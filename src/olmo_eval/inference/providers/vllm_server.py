"""vLLM Server provider for agent tasks."""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import httpx

from olmo_eval.common.beaker_status import BeakerStatusReporter
from olmo_eval.common.logging import get_logger
from olmo_eval.common.types import LMOutput, LMRequest, LogProbEntry, RequestType, SamplingParams
from olmo_eval.common.types.tools import ToolCall
from olmo_eval.inference.base import InferenceProvider
from olmo_eval.inference.errors import (
    CONTEXT_OVERFLOW_KEY,
    REQUEST_ERROR_KEY,
    is_context_overflow,
)
from olmo_eval.inference.hf_cache import refresh_hf_cache
from olmo_eval.inference.tokenizer_utils import encode_context_and_continuation
from olmo_eval.inference.utils import run_async

if TYPE_CHECKING:
    from openai import AsyncOpenAI

    from .vllm_server_utils import VLLMServerProcess

logger = get_logger(__name__)


class RemoteTokenizer:
    """Tokenizer that uses vLLM server's /tokenize and /detokenize endpoints.

    Provides a tokenizer-like interface without requiring transformers locally.
    This is useful when the vLLM server runs in an isolated environment.
    """

    def __init__(self, base_url: str, model_name: str) -> None:
        """Initialize the remote tokenizer.

        Args:
            base_url: Base URL of the vLLM server (e.g., "http://localhost:8000/v1").
            model_name: Model name for tokenization requests.
        """
        # Strip /v1 suffix if present for tokenize endpoints
        self._base_url = base_url.rstrip("/").removesuffix("/v1")
        self._model_name = model_name
        self._client: httpx.Client | None = None
        self._all_special_ids: set[int] | None = None
        # BOS/EOS token IDs - not available via API, set to None
        # For logprobs computation, empty context handling uses fallbacks
        self.bos_token_id: int | None = None
        self.eos_token_id: int | None = None

    def _get_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=30.0)
        return self._client

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """Encode text to token IDs using the remote server."""
        client = self._get_client()
        response = client.post(
            f"{self._base_url}/tokenize",
            json={
                "model": self._model_name,
                "prompt": text,
                "add_special_tokens": add_special_tokens,
            },
        )
        response.raise_for_status()
        data = response.json()
        return data.get("tokens", [])

    def decode(self, token_ids: list[int], skip_special_tokens: bool = False) -> str:
        """Decode token IDs to text using the remote server."""
        client = self._get_client()
        response = client.post(
            f"{self._base_url}/detokenize",
            json={
                "model": self._model_name,
                "tokens": token_ids,
            },
        )
        response.raise_for_status()
        data = response.json()
        return data.get("prompt", "")

    @property
    def all_special_ids(self) -> set[int]:
        """Get special token IDs (cached after first call)."""
        if self._all_special_ids is None:
            # vLLM doesn't expose special tokens via API, return empty set
            # Token inspection will still work, just without special token highlighting
            self._all_special_ids = set()
        return self._all_special_ids

    def __call__(
        self, text: str, return_tensors: str | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        """Tokenize text (for compatibility with HuggingFace tokenizer interface)."""
        tokens = self.encode(text)
        result: dict[str, Any] = {"input_ids": tokens}
        if return_tensors == "pt":
            import torch

            result["input_ids"] = torch.tensor([tokens])
        return result

    def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            self._client.close()
            self._client = None


class TokenizerBosOverride:
    """Tokenizer proxy that overrides add_bos_token without mutating the underlying tokenizer."""

    def __init__(self, tokenizer: Any, add_bos_token: bool) -> None:
        self._tokenizer = tokenizer
        self.add_bos_token = add_bos_token

    def __getattr__(self, name: str) -> Any:
        return getattr(self._tokenizer, name)


def _cut_at_stop_sequence(text: str, stop_sequences: list[str] | None) -> str:
    """Truncate text at the first non-empty stop sequence occurrence."""
    if not stop_sequences:
        return text

    smallest_idx = len(text)
    for stop_sequence in stop_sequences:
        if not stop_sequence:
            continue
        idx = text.find(stop_sequence)
        if idx != -1 and idx < smallest_idx:
            smallest_idx = idx
    return text[:smallest_idx]


def _completion_logprob_value(value: Any) -> float | None:
    """Extract a numeric logprob from OpenAI/vLLM completion logprob payloads."""
    if isinstance(value, dict):
        value = value.get("logprob", value.get("log_prob"))
    else:
        value = getattr(value, "logprob", value)

    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _completion_top_logprob_values(top_logprobs: Any) -> list[float]:
    """Extract numeric top-logprob values for one generated token."""
    if isinstance(top_logprobs, dict):
        candidates = top_logprobs.values()
    elif isinstance(top_logprobs, (list, tuple)):
        candidates = top_logprobs
    else:
        return []

    values: list[float] = []
    for candidate in candidates:
        if (logprob := _completion_logprob_value(candidate)) is not None:
            values.append(logprob)
    return values


def _completion_is_greedy(
    token_logprobs: list[float | None],
    top_logprobs: list[Any] | None,
) -> bool | None:
    """Best-effort greedy check for completion responses.

    Completion APIs expose generated-token logprobs rather than prompt_logprobs.
    When top_logprobs are present, a token is greedy if its emitted-token logprob
    is at least the highest returned top-logprob for that step.
    """
    if not token_logprobs or not top_logprobs:
        return None

    observed = False
    for token_logprob, top_logprob in zip(token_logprobs, top_logprobs, strict=False):
        chosen_logprob = _completion_logprob_value(token_logprob)
        top_values = _completion_top_logprob_values(top_logprob)
        if chosen_logprob is None or not top_values:
            continue

        observed = True
        if chosen_logprob + 1e-6 < max(top_values):
            return False

    return True if observed else None


# Enable with VLLM_DEBUG_REQUESTS=1
_DEBUG_REQUESTS = os.environ.get("VLLM_DEBUG_REQUESTS", "").lower() in ("1", "true", "yes")

# Disable retries for debugging with VLLM_DEBUG_NO_RETRY=1
_DEBUG_NO_RETRY = os.environ.get("VLLM_DEBUG_NO_RETRY", "").lower() in ("1", "true", "yes")


async def _log_request(request: httpx.Request) -> None:
    """Log outgoing HTTP request."""
    body = request.content.decode("utf-8", errors="replace") if request.content else ""
    # Truncate very long bodies
    if len(body) > 2000:
        body = body[:2000] + "... [truncated]"
    logger.info(f"vLLM request: {request.method} {request.url}\n  body: {body}")


async def _log_response(response: httpx.Response) -> None:
    """Log HTTP response, especially errors."""
    if response.status_code >= 400:
        # Read body for error details
        await response.aread()
        body = response.text[:1000] if response.text else "(empty)"
        logger.error(f"vLLM response error: {response.status_code} {response.url}\n  body: {body}")


class DebugTransport(httpx.AsyncHTTPTransport):
    """Transport wrapper that logs all HTTP errors with full tracebacks."""

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        try:
            return await super().handle_async_request(request)
        except Exception as e:
            import traceback

            logger.error(
                f"vLLM transport error: {type(e).__name__}: {e}\n"
                f"  URL: {request.url}\n"
                f"  Traceback:\n{traceback.format_exc()}"
            )
            raise


class VLLMServerProvider(InferenceProvider):
    """Provider that uses a vLLM server's OpenAI-compatible API.

    This provider can either connect to an existing vLLM server (if base_url
    is provided) or start and manage its own server subprocess.

    Tool schemas are sent to ``/v1/chat/completions``, but the server only
    returns parsed ``tool_calls`` when it was started with
    ``--enable-auto-tool-choice``; see :attr:`_tool_calls_parsed`.

    Example:
        # Auto-start server (managed lifecycle)
        provider = VLLMServerProvider("meta-llama/Llama-3.1-8B-Instruct")

        # Or connect to existing server
        provider = VLLMServerProvider("model", base_url="http://localhost:8000/v1")
    """

    supports_tools = True

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        timeout: float = 86400.0,
        max_concurrency: int = 32,
        max_retries: int = 3,
        tensor_parallel_size: int = 1,
        max_model_len: int | None = None,
        tokenizer: str | None = None,
        enable_auto_tool_choice: bool = False,
        tool_call_parser: str | None = None,
        trust_remote_code: bool = False,
        log_dir: str | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
        revision: str | None = None,
        force_download: bool = False,
        add_bos_token: bool | None = None,
        prompt_logprobs: int | None = None,
        completion_use_prompt_token_ids: bool | None = None,
        completion_client_side_stop_trim: bool | None = None,
        completion_sentencepiece_cleanup: bool | None = None,
        enable_lora: bool = False,
        lora_modules: list[str] | None = None,
        **server_kwargs: Any,
    ) -> None:
        """Initialize the provider.

        Args:
            model_name: Model identifier for requests.
            base_url: Base URL of existing vLLM server. If None, starts own server.
            timeout: Request timeout in seconds.
            max_concurrency: Maximum number of concurrent requests.
            max_retries: Maximum number of retries for transient errors.
            tensor_parallel_size: Number of GPUs for tensor parallelism (server mode).
            max_model_len: Authoritative maximum model context length. Passed to managed
                servers and used directly instead of metadata discovery for external servers.
            tokenizer: Tokenizer path override (server mode).
            enable_auto_tool_choice: Enable automatic tool choice (server mode).
            tool_call_parser: Tool call parser name (server mode).
            trust_remote_code: Trust remote code for model loading (server mode).
            log_dir: Directory to write server logs to (server mode).
            chat_template_kwargs: Extra kwargs for chat template (e.g., {"enable_thinking": false}).
            revision: HuggingFace revision/commit hash for managed server startup and
                local tokenizer loading.
            force_download: Force-refresh Hugging Face model/tokenizer cache entries
                before managed server startup and local tokenizer loads.
            add_bos_token: Optional provider-level BOS override for local tokenization paths.
                This matches the old oe-eval-internal runtime behavior and is not forwarded
                as a vLLM server CLI argument.
            prompt_logprobs: Number of prompt logprobs to request for loglikelihood scoring.
                Defaults to 5 in the current runtime.
            completion_use_prompt_token_ids: If True, locally tokenize completion prompts
                and send token IDs to /completions instead of raw prompt strings.
            completion_client_side_stop_trim: If True, trim completion text at the first
                stop sequence client-side after generation.
            completion_sentencepiece_cleanup: If True, replace SentencePiece space markers
                with actual spaces in completion outputs.
            enable_lora: Enable LoRA (Low-Rank Adaptation) support for model loading (server mode).
            lora_modules: List of LoRA module paths to load (format: "module_name=/path/to/lora").
                Only used if enable_lora is True (server mode). Raises an error if more than one
                is passed.
            **server_kwargs: Additional vLLM server arguments.
        """
        super().__init__(model_name)
        self._beaker_reporter = BeakerStatusReporter()
        self.timeout = timeout
        self.max_concurrency = max_concurrency
        self.max_retries = max_retries
        self.chat_template_kwargs = chat_template_kwargs
        self._add_bos_token = add_bos_token
        self._prompt_logprobs = prompt_logprobs if prompt_logprobs is not None else 5
        self._completion_use_prompt_token_ids = bool(completion_use_prompt_token_ids)
        self._completion_client_side_stop_trim = bool(completion_client_side_stop_trim)
        self._completion_sentencepiece_cleanup = bool(completion_sentencepiece_cleanup)
        self._tokenizer_path = tokenizer or model_name
        self._tokenizer_revision = revision
        self._tokenizer_trust_remote_code = trust_remote_code
        self._force_download = force_download
        self._client: AsyncOpenAI | None = None
        self._http_client: httpx.AsyncClient | None = None
        self._raw_http_client: httpx.AsyncClient | None = None
        self._openai_module: Any = None
        self._tokenizer: Any = None
        self._server: VLLMServerProcess | None = None  # type: ignore[possibly-unresolved-reference]
        if max_model_len is not None and max_model_len <= 0:
            raise ValueError("max_model_len must be positive when set")
        self._max_length: int | None = max_model_len
        self._max_length_source: str | None = "configured" if max_model_len is not None else None
        self._max_length_discovery_attempted = max_model_len is not None
        self._enable_lora = enable_lora
        self._lora_modules = lora_modules
        self._request_model_name = model_name
        if lora_modules and len(lora_modules) > 1:
            raise ValueError("the vllm server provider can accept only one lora module")
        if enable_lora and lora_modules:
            self._request_model_name = lora_modules[0].split("=", 1)[0]

        #: Whether replies will carry parsed ``tool_calls``: True for a managed
        #: server started with tool choice enabled, False for one started
        #: without it, and None for an external server, whose flags are not
        #: knowable from here.
        self._tool_calls_parsed: bool | None = None

        if base_url:
            # Connect to existing server
            self.base_url = base_url
        else:
            self._tool_calls_parsed = bool(enable_auto_tool_choice)
            # Start our own server
            from .vllm_server_utils import VLLMServerProcess

            # Build server kwargs
            srv_kwargs: dict[str, Any] = dict(server_kwargs)
            if force_download:
                refresh_kwargs = {
                    "revision": revision,
                    "cache_dir": srv_kwargs.get("download_dir") or srv_kwargs.get("cache_dir"),
                    "token": srv_kwargs.get("token"),
                    "force_download": True,
                }
                refresh_hf_cache(model_name, **refresh_kwargs)
                if tokenizer and tokenizer != model_name:
                    refresh_hf_cache(tokenizer, **refresh_kwargs)
            if max_model_len is not None:
                srv_kwargs["max_model_len"] = max_model_len
            if tokenizer:
                srv_kwargs["tokenizer"] = tokenizer
            if revision is not None:
                srv_kwargs["revision"] = revision
            if enable_auto_tool_choice:
                srv_kwargs["enable_auto_tool_choice"] = True
                if tool_call_parser:
                    srv_kwargs["tool_call_parser"] = tool_call_parser
            if trust_remote_code:
                srv_kwargs["trust_remote_code"] = True
            if enable_lora:
                srv_kwargs["enable_lora"] = True
            if lora_modules:
                srv_kwargs["lora_modules"] = lora_modules
            self._server = VLLMServerProcess(
                model_name=model_name,
                tensor_parallel_size=tensor_parallel_size,
                log_dir=log_dir,
                **srv_kwargs,
            )

            def _on_startup(msg: str) -> None:
                logger.info(msg)
                self._beaker_reporter.update(msg)

            self._server.start(progress_callback=_on_startup)
            self.base_url = self._server.base_url

    def close(self) -> None:
        """Close the provider and stop managed server if any."""
        server = getattr(self, "_server", None)
        if server is not None:
            server.stop()
            self._server = None

    def __del__(self) -> None:
        """Ensure server is stopped on garbage collection."""
        self.close()

    @property
    def max_length(self) -> int | None:
        """Get an authoritative model context length, if one is available."""
        if self._max_length is not None or self._max_length_discovery_attempted:
            return self._max_length

        self._max_length_discovery_attempted = True
        client = httpx.Client(timeout=30.0)
        try:
            resp = client.get(f"{self.base_url}/models")
            resp.raise_for_status()
            data = resp.json()
            if not isinstance(data, dict):
                raise ValueError("/models response is not a JSON object")
            model_records = data.get("data", [])
            if not isinstance(model_records, list):
                raise ValueError("/models response does not contain a list in 'data'")

            for model_info in model_records:
                if not isinstance(model_info, dict) or model_info.get("id") != self.model_name:
                    continue
                max_len = model_info.get("max_model_len")
                if max_len is None:
                    break
                discovered_max_length = int(max_len)
                if discovered_max_length <= 0:
                    raise ValueError("max_model_len reported by /models must be positive")
                self._max_length = discovered_max_length
                self._max_length_source = "server"
                break
        except (httpx.HTTPError, TypeError, ValueError) as exc:
            logger.warning(
                "Could not determine max_model_len from %s/models (%s); "
                "leaving the context length unknown",
                self.base_url,
                exc,
            )
            return None
        finally:
            client.close()

        if self._max_length is None:
            logger.warning(
                "Server model metadata does not provide max_model_len for %r; "
                "leaving the context length unknown",
                self.model_name,
            )
        return self._max_length

    def _get_or_create_client(self) -> AsyncOpenAI:
        """Get or create the AsyncOpenAI client."""
        if self._client is None:
            import openai
            from openai import AsyncOpenAI

            self._openai_module = openai

            # Configure connection pool limits to prevent exhaustion with large batches.
            limits = httpx.Limits(
                max_keepalive_connections=20,
                max_connections=100,
                keepalive_expiry=60.0,  # Close idle connections after 120s
            )

            # Build event hooks for debug logging if enabled
            event_hooks: dict[str, list[Any]] | None = None
            if _DEBUG_REQUESTS:
                logger.info("vLLM debug request logging enabled (VLLM_DEBUG_REQUESTS=1)")
                event_hooks = {
                    "request": [_log_request],
                    "response": [_log_response],
                }

            # Use debug transport when enabled to catch connection errors
            transport = DebugTransport() if _DEBUG_REQUESTS else None

            self._http_client = httpx.AsyncClient(
                transport=transport,
                limits=limits,
                timeout=self.timeout,
                event_hooks=event_hooks or {},
            )

            # Use 0 retries when debugging to see errors immediately
            effective_retries = 0 if _DEBUG_NO_RETRY else self.max_retries
            if _DEBUG_NO_RETRY:
                logger.info("vLLM debug: retries disabled (VLLM_DEBUG_NO_RETRY=1)")

            self._client = AsyncOpenAI(
                base_url=self.base_url,
                api_key="EMPTY",
                timeout=self.timeout,
                max_retries=effective_retries,
                http_client=self._http_client,
            )
        return self._client

    async def aclose(self) -> None:
        """Close the provider and release resources."""
        if self._raw_http_client is not None:
            await self._raw_http_client.aclose()
            self._raw_http_client = None
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None
        if self._client is not None:
            await self._client.close()
            self._client = None

    def _get_raw_http_client(self) -> httpx.AsyncClient:
        """Get or create raw HTTP client for direct vLLM API calls.

        Used for prompt_logprobs requests where we need integer-keyed
        token ID data that the OpenAI SDK converts to string keys.
        """
        if self._raw_http_client is None:
            self._raw_http_client = httpx.AsyncClient(timeout=self.timeout)
        return self._raw_http_client

    def get_openai_client(self) -> AsyncOpenAI:
        """Get the AsyncOpenAI client for this provider."""
        client = self._get_or_create_client()
        assert client is not None, "AsyncOpenAI client creation failed"
        return client

    def _get_local_tokenizer_load_kwargs(self) -> dict[str, Any]:
        """Build kwargs for local HuggingFace tokenizer loads."""
        tokenizer_kwargs: dict[str, Any] = {
            "trust_remote_code": self._tokenizer_trust_remote_code,
        }
        if self._tokenizer_revision is not None:
            tokenizer_kwargs["revision"] = self._tokenizer_revision
        if self._force_download:
            tokenizer_kwargs["force_download"] = True
        return tokenizer_kwargs

    def _get_tokenizer(self, *, require_local: bool = False) -> Any:
        """Get or create the tokenizer.

        Args:
            require_local: If True, loads HuggingFace tokenizer locally (slower but
                has BOS/EOS token IDs for logprobs computation). If False, uses
                the remote vLLM tokenization API (faster, no transformers needed).

        Returns:
            Tokenizer instance (RemoteTokenizer or HuggingFace AutoTokenizer).
        """
        if require_local:
            # Load HuggingFace tokenizer for full functionality (BOS/EOS handling)
            if self._tokenizer is None or isinstance(self._tokenizer, RemoteTokenizer):
                from transformers import AutoTokenizer

                self._tokenizer = AutoTokenizer.from_pretrained(
                    self._tokenizer_path,
                    **self._get_local_tokenizer_load_kwargs(),
                )
            return self._tokenizer

        # Use remote tokenizer by default (no transformers dependency)
        if self._tokenizer is None:
            self._tokenizer = RemoteTokenizer(self.base_url, self._request_model_name)
        return self._tokenizer

    def get_tokenizer(self) -> Any:
        """Get the tokenizer for this provider (for external use like inspection)."""
        return self._get_tokenizer(require_local=False)

    async def _generate_single_impl(
        self, request: LMRequest, params: SamplingParams
    ) -> list[LMOutput]:
        """Generate completions for a single request."""
        client = self._get_or_create_client()

        # Route to completions endpoint for COMPLETION requests without messages
        use_completions = (
            request.request_type == RequestType.COMPLETION
            and not request.messages
            and request.prompt
        )

        if use_completions:
            return await self._generate_completion(client, request, params)
        else:
            return await self._generate_chat(client, request, params)

    def describe_request(
        self,
        request: LMRequest,
        sampling_params: SamplingParams | None = None,
    ) -> dict[str, Any] | None:
        params = self._default_sampling_params(sampling_params)
        trace = super().describe_request(request, sampling_params)
        if trace is None:
            return None

        if request.request_type == RequestType.LOGLIKELIHOOD:
            trace["provider"] = "VLLMServerProvider"
            trace["endpoint"] = "/completions"
            trace["generation_kwargs"] = {
                "max_gen_toks": 1,
                "do_sample": False,
                "temperature": params.temperature,
                "prompt_logprobs": self._prompt_logprobs,
                "add_special_tokens": False,
            }
            trace["stop_sequences"] = []
            trace["input_mode"] = "prompt_token_ids"
            return trace

        use_completions = (
            request.request_type == RequestType.COMPLETION
            and not request.messages
            and request.prompt
        )
        if use_completions:
            trace["provider"] = "VLLMServerProvider"
            trace["endpoint"] = "/completions"
            trace["generation_kwargs"] = {
                "max_gen_toks": params.max_tokens,
                "do_sample": params.do_sample and params.temperature > 0,
                "temperature": params.temperature,
                "logprobs": 1,
                "num_samples": params.num_samples,
                "add_special_tokens": False,
            }
            if params.top_p is not None:
                trace["generation_kwargs"]["top_p"] = params.top_p
            if params.do_sample and params.temperature > 0 and params.top_k is not None:
                trace["generation_kwargs"]["top_k"] = params.top_k
            truncate_prompt_tokens, truncation_side = self._get_prompt_truncation(params)
            if truncate_prompt_tokens is not None:
                trace["generation_kwargs"]["truncate_prompt_tokens"] = truncate_prompt_tokens
            if truncation_side is not None:
                trace["generation_kwargs"]["truncation_side"] = truncation_side
            trace["stop_sequences"] = self._get_completion_stop_sequences(params) or []
            trace["input_mode"] = (
                "prompt_token_ids" if self._completion_use_prompt_token_ids else "text"
            )
            return trace

        trace["provider"] = "VLLMServerProvider"
        trace["endpoint"] = "/chat/completions"
        generation_kwargs: dict[str, Any] = {
            "max_gen_toks": params.max_tokens,
            "do_sample": params.do_sample and params.temperature > 0,
            "temperature": params.temperature,
            "logprobs": True,
            "top_logprobs": 1,
            "num_samples": params.num_samples,
        }
        if params.do_sample and params.temperature > 0 and params.top_k is not None:
            generation_kwargs["top_k"] = params.top_k
        if params.top_p is not None:
            generation_kwargs["top_p"] = params.top_p
        if self.chat_template_kwargs:
            generation_kwargs["chat_template_kwargs"] = dict(self.chat_template_kwargs)
        truncate_prompt_tokens, truncation_side = self._get_prompt_truncation(params)
        if truncate_prompt_tokens is not None:
            generation_kwargs["truncate_prompt_tokens"] = truncate_prompt_tokens
        if truncation_side is not None:
            generation_kwargs["truncation_side"] = truncation_side
        trace["generation_kwargs"] = generation_kwargs
        trace["stop_sequences"] = list(params.stop_sequences or ())
        trace["input_mode"] = "messages"
        return trace

    def _get_completion_eos_stop(self) -> str | None:
        """Best-effort EOS stop string for completion parity with oe-eval."""
        try:
            tokenizer = self._get_tokenizer(require_local=True)
        except (ImportError, Exception):
            return None

        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if eos_token_id is not None:
            try:
                eos_token = tokenizer.decode([eos_token_id], skip_special_tokens=False)
            except TypeError:
                eos_token = tokenizer.decode([eos_token_id])
            if eos_token:
                return eos_token

        eos_token = getattr(tokenizer, "eos_token", None)
        return str(eos_token) if eos_token else None

    def _get_completion_stop_sequences(self, params: SamplingParams) -> list[str] | None:
        """Build stop sequences for completions, including EOS when available."""
        stop_sequences = list(params.stop_sequences or ())
        eos_stop = self._get_completion_eos_stop()
        if eos_stop and eos_stop not in stop_sequences:
            stop_sequences.append(eos_stop)
        return stop_sequences or None

    def _get_completion_prompt_payload(self, prompt: str) -> str | list[int]:
        """Build completion prompt payload, optionally using local token IDs for parity."""
        if not self._completion_use_prompt_token_ids:
            return prompt

        tokenizer = self._get_tokenizer(require_local=True)
        return tokenizer.encode(prompt, add_special_tokens=bool(self._add_bos_token))

    def _get_prompt_truncation(
        self,
        params: SamplingParams,
    ) -> tuple[int | None, str | None]:
        """Return only caller-requested prompt truncation settings."""
        return params.truncate_prompt_tokens, params.truncation_side

    def _postprocess_completion_text(self, text: str, stop_sequences: list[str] | None) -> str:
        """Apply optional legacy completion post-processing."""
        if self._completion_sentencepiece_cleanup:
            text = text.replace("\u2581", " ")
        if self._completion_client_side_stop_trim:
            text = _cut_at_stop_sequence(text, stop_sequences)
        return text

    def _completion_metadata_from_logprobs(
        self,
        *,
        tokens: list[str],
        token_logprobs: list[float | None],
        top_logprobs: list[Any] | None,
        metadata: dict[str, Any],
    ) -> list[LogProbEntry] | None:
        """Convert completion logprob payload into standard entries and metadata."""
        logprob_entries: list[LogProbEntry] = []
        for token, logprob in zip(tokens, token_logprobs, strict=False):
            if logprob is not None:
                logprob_entries.append({"token": token, "logprob": logprob})

        if logprob_entries:
            sum_logits = sum(entry["logprob"] for entry in logprob_entries)
            num_tokens = len(logprob_entries)
            metadata.update(
                {
                    "sum_logits": sum_logits,
                    "num_tokens": num_tokens,
                    "num_tokens_all": num_tokens,
                }
            )
            if (is_greedy := _completion_is_greedy(token_logprobs, top_logprobs)) is not None:
                metadata["is_greedy"] = is_greedy
            return logprob_entries

        return None

    def _completion_usage_metadata(self, usage: Any, num_choices: int = 1) -> dict[str, Any]:
        """Extract per-output token counts from SDK or raw JSON response usage.

        Response usage sums completion tokens over all choices, so the count
        describes a single output only when the response has one choice.
        """
        if not usage:
            return {}

        if isinstance(usage, dict):
            prompt_tokens = usage.get("prompt_tokens")
            completion_tokens = usage.get("completion_tokens")
        else:
            prompt_tokens = getattr(usage, "prompt_tokens", None)
            completion_tokens = getattr(usage, "completion_tokens", None)

        metadata: dict[str, Any] = {}
        if prompt_tokens is not None:
            metadata["prompt_tokens"] = prompt_tokens
        if completion_tokens is not None and num_choices == 1:
            metadata["completion_tokens"] = completion_tokens
        return metadata

    def _build_completion_output(
        self,
        *,
        text: str,
        logprobs_payload: Any,
        usage: Any,
        stop_sequences: list[str] | None,
        finish_reason: str | None = None,
        num_choices: int = 1,
    ) -> LMOutput:
        """Create a standardized LMOutput from completion response payloads."""
        metadata = self._completion_usage_metadata(usage, num_choices)
        metadata["finish_reason"] = finish_reason
        processed_text = self._postprocess_completion_text(text, stop_sequences)

        logprob_entries: list[LogProbEntry] | None = None
        if logprobs_payload is not None:
            if isinstance(logprobs_payload, dict):
                tokens = logprobs_payload.get("tokens") or []
                token_logprobs = logprobs_payload.get("token_logprobs") or []
                top_logprobs = logprobs_payload.get("top_logprobs") or []
            else:
                tokens = getattr(logprobs_payload, "tokens", None) or []
                token_logprobs = getattr(logprobs_payload, "token_logprobs", None) or []
                top_logprobs = getattr(logprobs_payload, "top_logprobs", None) or []
            logprob_entries = self._completion_metadata_from_logprobs(
                tokens=tokens,
                token_logprobs=token_logprobs,
                top_logprobs=top_logprobs,
                metadata=metadata,
            )

        return LMOutput(text=processed_text, logprobs=logprob_entries, metadata=metadata)

    async def _generate_completion(
        self, client: AsyncOpenAI, request: LMRequest, params: SamplingParams
    ) -> list[LMOutput]:
        """Generate using the /v1/completions endpoint."""
        kwargs: dict[str, Any] = {
            "model": self._request_model_name,
            "prompt": request.prompt,
            "n": params.num_samples,
            "logprobs": 1,  # Request logprobs for metrics
        }
        # max_tokens=None means "generate to the context limit"; omit the field
        # rather than sending null, which some OpenAI-compatible servers reject.
        if params.max_tokens is not None:
            kwargs["max_tokens"] = params.max_tokens
        extra_body: dict[str, Any] = {"add_special_tokens": bool(self._add_bos_token)}

        # Always send temperature explicitly to avoid server defaults (OpenAI API defaults to 1.0)
        kwargs["temperature"] = params.temperature
        if params.top_p is not None:
            kwargs["top_p"] = params.top_p
        if params.do_sample and params.temperature > 0 and params.top_k is not None:
            extra_body["top_k"] = params.top_k
        stop_sequences = self._get_completion_stop_sequences(params)
        if stop_sequences:
            kwargs["stop"] = stop_sequences
        truncate_prompt_tokens, truncation_side = self._get_prompt_truncation(params)
        if truncate_prompt_tokens is not None:
            extra_body["truncate_prompt_tokens"] = truncate_prompt_tokens
        if truncation_side is not None:
            extra_body["truncation_side"] = truncation_side
        if self._completion_use_prompt_token_ids:
            http_client = self._get_raw_http_client()
            response = await http_client.post(
                f"{self.base_url}/completions",
                json={
                    **kwargs,
                    "prompt": self._get_completion_prompt_payload(request.prompt),
                    **extra_body,
                },
            )
            response.raise_for_status()
            data = response.json()
            usage = data.get("usage")
            choices = data.get("choices", [])
            return [
                self._build_completion_output(
                    text=choice.get("text") or "",
                    finish_reason=choice.get("finish_reason"),
                    logprobs_payload=choice.get("logprobs"),
                    usage=usage,
                    stop_sequences=stop_sequences,
                    num_choices=len(choices),
                )
                for choice in choices
            ]

        if extra_body:
            kwargs["extra_body"] = extra_body
        response = await client.completions.create(**kwargs)
        usage = getattr(response, "usage", None)
        return [
            self._build_completion_output(
                text=choice.text or "",
                finish_reason=getattr(choice, "finish_reason", None),
                logprobs_payload=getattr(choice, "logprobs", None),
                usage=usage,
                stop_sequences=stop_sequences,
                num_choices=len(response.choices),
            )
            for choice in response.choices
        ]

    #: Room left for the turn markers the chat template adds around a reply.
    _CONTEXT_MARGIN_TOKENS = 2

    #: What to ask for when the prompt already fills the window. The request
    #: will be rejected either way; this keeps the number recognisable.
    _NO_ROOM_MAX_TOKENS = 1000

    async def _fit_max_tokens(
        self, max_tokens: int, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None
    ) -> int:
        """Shrink ``max_tokens`` to the room the prompt leaves in the context window.

        Returns ``max_tokens`` unchanged when the window or the prompt length
        cannot be established, so a server that does not report them behaves as
        it did before.
        """
        context_length = self.max_length
        if context_length is None:
            return max_tokens

        prompt_tokens = await self._count_chat_tokens(messages, tools)
        if prompt_tokens is None:
            return max_tokens

        if context_length < prompt_tokens + self._CONTEXT_MARGIN_TOKENS:
            return self._NO_ROOM_MAX_TOKENS
        return min(max_tokens, context_length - prompt_tokens - self._CONTEXT_MARGIN_TOKENS)

    async def _count_chat_tokens(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None
    ) -> int | None:
        """Return the prompt's token count as the server's chat template renders it."""
        payload: dict[str, Any] = {"model": self._request_model_name, "messages": messages}
        if tools:
            payload["tools"] = tools
        if self.chat_template_kwargs:
            payload["chat_template_kwargs"] = dict(self.chat_template_kwargs)

        base_url = self.base_url.rstrip("/").removesuffix("/v1")
        try:
            async with httpx.AsyncClient(timeout=30.0) as http_client:
                response = await http_client.post(f"{base_url}/tokenize", json=payload)
                response.raise_for_status()
                data = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning("Could not count prompt tokens for max_tokens fitting: %s", exc)
            return None

        count = data.get("count")
        if count is None:
            tokens = data.get("tokens")
            count = len(tokens) if isinstance(tokens, list) else None
        return int(count) if count is not None else None

    def _refuse_tools_without_parsing(self, requests: Sequence[LMRequest]) -> None:
        """Fail when tool requests would go to a server that returns no tool calls."""
        if self._tool_calls_parsed is not False or not any(r.tools for r in requests):
            return
        raise ValueError(
            "This vLLM server was started without --enable-auto-tool-choice, so it "
            "answers in plain text and returns no tool_calls. Every instance of a "
            "tool-calling task would score as if the model called nothing. Remove the "
            "provider.kwargs.enable_auto_tool_choice=false override, or run a task "
            "variant that writes the functions into the prompt instead."
        )

    async def _generate_chat(
        self, client: AsyncOpenAI, request: LMRequest, params: SamplingParams
    ) -> list[LMOutput]:
        """Generate using the /v1/chat/completions endpoint."""
        # Build messages
        if request.messages:
            messages: list[dict[str, Any]] = [dict(m) for m in request.messages]
        else:
            messages = [{"role": "user", "content": request.prompt}]

        # Build tools if present
        tools = None
        if request.tools:
            tools = [t.to_openai() for t in request.tools]

        # Build request kwargs
        kwargs: dict[str, Any] = {
            "model": self._request_model_name,
            "messages": messages,
            "n": params.num_samples,
        }
        # max_tokens=None means "generate to the context limit"; omit the field
        # rather than sending null, which some OpenAI-compatible servers reject.
        max_tokens = params.max_tokens
        if max_tokens is not None and params.fit_max_tokens_to_context:
            max_tokens = await self._fit_max_tokens(max_tokens, messages, tools)
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        # Always send temperature explicitly to avoid server defaults (OpenAI API defaults to 1.0)
        kwargs["temperature"] = params.temperature
        if params.top_p is not None:
            kwargs["top_p"] = params.top_p
        extra_body: dict[str, Any] = {}
        if params.do_sample and params.temperature > 0 and params.top_k is not None:
            extra_body["top_k"] = params.top_k
        truncate_prompt_tokens, truncation_side = self._get_prompt_truncation(params)
        if truncate_prompt_tokens is not None:
            extra_body["truncate_prompt_tokens"] = truncate_prompt_tokens
        if truncation_side is not None:
            extra_body["truncation_side"] = truncation_side
        if params.stop_sequences:
            kwargs["stop"] = list(params.stop_sequences)
        if tools:
            kwargs["tools"] = tools
        # Always request logprobs for metrics computation
        # Both logprobs=True and top_logprobs are required for chat completions API
        kwargs["logprobs"] = True
        kwargs["top_logprobs"] = 1

        # Pass chat_template_kwargs via extra_body for vLLM
        if self.chat_template_kwargs:
            extra_body["chat_template_kwargs"] = self.chat_template_kwargs
        if extra_body:
            kwargs["extra_body"] = extra_body

        response = await client.chat.completions.create(**kwargs)

        # Capture usage for accurate metrics
        usage = getattr(response, "usage", None)

        outputs = []
        for choice in response.choices:
            text = choice.message.content or ""
            tool_calls = None
            if choice.message.tool_calls:
                tool_calls = [
                    ToolCall.create(
                        call_id=tc.id,
                        name=tc.function.name,
                        arguments=tc.function.arguments,
                    )
                    for tc in choice.message.tool_calls
                ]

            # Convert logprobs to standard format
            logprob_entries: list[LogProbEntry] | None = None
            metadata: dict[str, Any] = {"finish_reason": getattr(choice, "finish_reason", None)}

            metadata.update(self._completion_usage_metadata(usage, len(response.choices)))

            logprobs_data = getattr(choice, "logprobs", None)
            if logprobs_data and hasattr(logprobs_data, "content") and logprobs_data.content:
                logprob_entries = []
                for lp in logprobs_data.content:
                    entry: LogProbEntry = {"token": lp.token, "logprob": lp.logprob}
                    lp_bytes = getattr(lp, "bytes", None)
                    if lp_bytes is not None:
                        entry["bytes"] = lp_bytes
                    logprob_entries.append(entry)

                # Compute metadata from logprobs
                sum_logits = sum(entry["logprob"] for entry in logprob_entries)
                num_tokens = len(logprob_entries)
                metadata.update(
                    {
                        "sum_logits": sum_logits,
                        "num_tokens": num_tokens,
                        "num_tokens_all": num_tokens,
                    }
                )

            outputs.append(
                LMOutput(
                    text=text, logprobs=logprob_entries, metadata=metadata, tool_calls=tool_calls
                )
            )

        return outputs

    async def _generate_single_async(
        self, request: LMRequest, params: SamplingParams
    ) -> list[LMOutput]:
        """Generate completions for a single request."""
        return await self._generate_single_impl(request, params)

    async def agenerate(
        self,
        requests: list[LMRequest],
        sampling_params: SamplingParams | None = None,
    ) -> list[list[LMOutput]]:
        """Async generate completions via the vLLM server.

        This is the native async implementation that should be used
        in async contexts to avoid nested event loops.

        Args:
            requests: Batch of requests to process.
            sampling_params: Sampling configuration.

        Returns:
            List of output lists, one per request.
        """
        from olmo_eval.inference.dispatch import dispatch_concurrent

        # Checked for the whole batch before dispatch, which would otherwise catch
        # the refusal per request and hand back empty outputs that score as zero.
        self._refuse_tools_without_parsing(requests)
        params = self._default_sampling_params(sampling_params)

        async def process(req: LMRequest) -> list[LMOutput]:
            return await self._generate_or_mark_overflow(req, params)

        results = await dispatch_concurrent(
            requests,
            process,
            max_in_flight=self.max_concurrency,
            max_retries=self.max_retries,
        )
        return [r if r is not None else [] for r in results]

    async def _generate_or_mark_overflow(
        self, request: LMRequest, params: SamplingParams
    ) -> list[LMOutput]:
        """Generate, recording a context overflow on the reply instead of raising.

        A prompt too long for the window fails the same way every time, so it is
        not worth retrying; marking it lets the caller see why the request failed.
        """
        try:
            return await self._generate_single_async(request, params)
        except Exception as exc:
            if not is_context_overflow(exc):
                raise
            return [
                LMOutput(
                    text="",
                    metadata={
                        REQUEST_ERROR_KEY: f"context overflow: {exc}",
                        CONTEXT_OVERFLOW_KEY: True,
                    },
                )
            ]

    def generate(
        self,
        requests: list[LMRequest],
        sampling_params: SamplingParams | None = None,
    ) -> list[list[LMOutput]]:
        """Generate completions via the vLLM server (sync wrapper).

        For async contexts, prefer using agenerate() directly.

        Args:
            requests: Batch of requests to process.
            sampling_params: Sampling configuration.

        Returns:
            List of output lists, one per request.
        """
        return run_async(self.agenerate(requests, sampling_params))

    async def _logprobs_single_impl(
        self,
        request: LMRequest,
        params: SamplingParams | None = None,
    ) -> list[LMOutput]:
        """Compute logprobs for continuations using raw prompt_logprobs.

        Uses vLLM's prompt_logprobs feature with integer token ID keys
        (via raw HTTP) to avoid string-keyed dict collisions in is_greedy
        computation. This matches the inline vLLM provider's behavior exactly.

        Prefers local HuggingFace tokenizer for accurate context/continuation
        boundary computation (matches oe-eval-internal behavior). Falls back to
        RemoteTokenizer if transformers is not available.
        """
        # Prefer local tokenizer for exact boundary matching with inline vLLM
        try:
            tokenizer = self._get_tokenizer(require_local=True)
        except (ImportError, Exception):
            tokenizer = self._get_tokenizer(require_local=False)
        if self._add_bos_token is not None:
            tokenizer = TokenizerBosOverride(tokenizer, self._add_bos_token)
        params = self._default_sampling_params(params)

        # Get the context/prompt text
        context = request.prompt
        if request.messages:
            context = request.messages[0].get("content", "") if request.messages else ""

        http_client = self._get_raw_http_client()

        outputs = []
        cont_prompts = request.continuation_prompts
        for i, continuation in enumerate(request.continuations or ()):
            # Use per-continuation prompt when available (e.g. Trinh & Le partial eval)
            ctx = cont_prompts[i] if cont_prompts else context
            # Use shared utility for proper tokenization (handles BOS, trailing spaces)
            context_enc, continuation_enc = encode_context_and_continuation(
                tokenizer, ctx, continuation
            )

            # RemoteTokenizer doesn't have BOS/EOS token IDs, so for empty contexts
            # encode_context_and_continuation returns empty context_enc.
            # In this case, encode with add_special_tokens=True to get BOS from server.
            if not context_enc and context == "":
                context_enc = tokenizer.encode("", add_special_tokens=True)

            full_tokens = context_enc + continuation_enc
            # Loglikelihood scoring intentionally left-truncates only when the task,
            # provider configuration, or server metadata supplies an authoritative limit.
            max_len = request.max_length if request.max_length is not None else self.max_length
            if max_len is not None and max_len <= 1:
                raise ValueError("max_length must be greater than 1 for loglikelihood requests")
            overflow = 0
            if max_len is not None and len(full_tokens) > max_len - 1:
                full_tokens = full_tokens[-(max_len - 1) :]
                overflow = len(context_enc) + len(continuation_enc) - (max_len - 1)
                context_len = max(0, len(context_enc) - overflow)
            else:
                context_len = len(context_enc)

            # Use raw HTTP to get integer-keyed prompt_logprobs from vLLM.
            # The OpenAI SDK's top_logprobs uses string keys (decoded tokens),
            # which can collide when different token IDs decode to the same string.
            # prompt_logprobs preserves integer token IDs through JSON serialization
            # (as string representations of ints), avoiding this collision.
            resp = await http_client.post(
                f"{self.base_url}/completions",
                json={
                    "model": self._request_model_name,
                    "prompt": full_tokens,
                    "max_tokens": 1,
                    "temperature": params.temperature,
                    "prompt_logprobs": self._prompt_logprobs,
                    "add_special_tokens": False,
                },
            )
            resp.raise_for_status()
            data = resp.json()

            # Extract prompt_logprobs: list of dict[str(token_id), {logprob, ...}]
            choice = data["choices"][0]
            prompt_logprobs_raw = choice.get("prompt_logprobs") or []

            logprob_entries: list[LogProbEntry] = []
            total = 0.0
            is_greedy = True

            # Process continuation tokens only (skip context positions)
            cont_logprobs = prompt_logprobs_raw[context_len : context_len + len(continuation_enc)]
            cont_token_ids = full_tokens[context_len:]

            for token_id, token_probs in zip(cont_token_ids, cont_logprobs, strict=True):
                if not token_probs:
                    continue

                # JSON serializes integer dict keys as strings
                token_id_str = str(token_id)

                # Check is_greedy BEFORE the lp_obj gate so we catch non-greedy tokens
                # even when they aren't in the top-k returned by prompt_logprobs.
                # Keys are string-serialized integers (e.g., "128000"), not decoded tokens,
                # so no collision between different token IDs.
                if is_greedy:
                    max_tid = max(
                        token_probs.keys(),
                        key=lambda tid: (
                            token_probs[tid]["logprob"]
                            if isinstance(token_probs[tid], dict)
                            else token_probs[tid]
                        ),
                    )
                    if max_tid != token_id_str:
                        is_greedy = False

                lp_obj = token_probs.get(token_id_str)
                if lp_obj is None:
                    continue

                logprob_val = lp_obj["logprob"] if isinstance(lp_obj, dict) else lp_obj

                # Get decoded token string
                if isinstance(lp_obj, dict) and lp_obj.get("decoded_token"):
                    token_str = lp_obj["decoded_token"]
                else:
                    token_str = tokenizer.decode([token_id])

                logprob_entries.append(
                    {
                        "token": token_str,
                        "logprob": logprob_val,
                        "token_id": int(token_id),
                        "bytes": list(token_str.encode("utf-8")),
                    }
                )
                total += logprob_val

            num_tokens = len(logprob_entries)
            outputs.append(
                LMOutput(
                    text=continuation,
                    logprobs=logprob_entries if logprob_entries else None,
                    metadata={
                        "total_logprob": total,
                        "sum_logits": total,
                        "num_tokens": num_tokens,
                        # Context plus continuation as sent (after truncation), matching the
                        # offline vllm provider. Used for token accounting, not scoring.
                        "num_tokens_all": len(full_tokens),
                        "prompt_truncated_tokens": overflow,
                        "is_greedy": is_greedy,
                    },
                )
            )

        return outputs

    async def _logprobs_single_async(
        self,
        request: LMRequest,
        params: SamplingParams | None = None,
    ) -> list[LMOutput]:
        """Compute logprobs for a single request."""
        return await self._logprobs_single_impl(request, params)

    async def alogprobs(
        self,
        requests: list[LMRequest],
        sampling_params: SamplingParams | None = None,
    ) -> list[list[LMOutput]]:
        """Async compute logprobs for continuations.

        Args:
            requests: Batch of requests with continuations to score.

        Returns:
            List of output lists with logprobs populated.
        """
        from olmo_eval.inference.dispatch import dispatch_concurrent

        params = self._default_sampling_params(sampling_params)
        if params.truncate_prompt_tokens is not None or params.truncation_side is not None:
            logger.warning(
                "truncate_prompt_tokens or truncation_side has been set in the params, "
                "but is not supported for loglikelihood requests and will not be used."
            )
        results = await dispatch_concurrent(
            requests,
            lambda request: self._logprobs_single_async(request, params),
            max_in_flight=self.max_concurrency,
            max_retries=self.max_retries,
        )

        # Log if any requests failed (result is None)
        failed_count = sum(1 for r in results if r is None)
        if failed_count > 0:
            # Try to get the actual error by running one request directly
            try:
                await self._logprobs_single_async(requests[0], params)
            except Exception as e:
                logger.error(
                    f"alogprobs: {failed_count}/{len(requests)} requests failed. First error: {e!r}"
                )

        # Replace None with empty list for failed requests
        return [r if r is not None else [] for r in results]

    def logprobs(
        self,
        requests: list[LMRequest],
        sampling_params: SamplingParams | None = None,
    ) -> list[list[LMOutput]]:
        """Compute logprobs for continuations.

        Args:
            requests: Batch of requests with continuations to score.

        Returns:
            List of output lists with logprobs populated.
        """
        return run_async(self.alogprobs(requests, sampling_params))
