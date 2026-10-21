"""LiteLLM provider for API-based inference."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from olmo_eval.common.debug import is_debug_provider
from olmo_eval.common.logging import get_logger
from olmo_eval.common.openai_credentials import (
    OPENROUTER_BASE_URL,
    is_openai_model,
    openrouter_model_slug,
    resolve_openai_credentials,
)
from olmo_eval.common.types import LMOutput, LMRequest, LogProbEntry, RequestType, SamplingParams
from olmo_eval.inference.base import InferenceProvider
from olmo_eval.inference.retry import retry_with_backoff
from olmo_eval.inference.utils import run_async

if TYPE_CHECKING:
    from openai import AsyncOpenAI

# Maximum stop sequences supported by OpenAI-compatible APIs
_MAX_STOP_SEQUENCES = 4

# Request parameters that OpenAI accepts or rejects depending on the model.
_OPENAI_SAMPLING_PARAMS = frozenset(
    {
        "logprobs",
        "top_logprobs",
        "temperature",
        "top_p",
        "stop",
        "n",
        "max_completion_tokens",
        "max_tokens",
        "presence_penalty",
        "frequency_penalty",
        "reasoning_effort",
        "seed",
    }
)

logger = get_logger(__name__)


class LiteLLMProvider(InferenceProvider):
    """Provider using LiteLLM for unified API access to various providers."""

    # Environment variable to LiteLLM attribute mappings
    _API_KEY_MAPPINGS = {
        "OPENAI_API_KEY": "openai_api_key",
        "ANTHROPIC_API_KEY": "anthropic_api_key",
        "COHERE_API_KEY": "cohere_api_key",
        "TOGETHER_API_KEY": "together_api_key",
        "AZURE_API_KEY": "azure_api_key",
        "AZURE_API_BASE": "azure_api_base",
        "AZURE_API_VERSION": "azure_api_version",
        "LITELLM_PROXY_API_KEY": "litellm_proxy_api_key",
    }

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_base: str | None = None,
        max_concurrency: int = 32,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        **api_kwargs,
    ) -> None:
        """Initialize the provider.

        Args:
            model_name: Model identifier (e.g., "gpt-4", "claude-3-opus").
            base_url: Optional base URL for OpenAI-compatible endpoints.
            api_base: Optional API base for litellm.completion.
            max_concurrency: Maximum number of concurrent API requests.
            max_retries: Maximum number of retries for transient errors.
            retry_delay: Base delay in seconds between retries (exponential backoff).
            **api_kwargs: Additional arguments passed to litellm.completion.
        """
        try:
            import litellm
        except ImportError as e:
            raise ImportError(
                "litellm is required for LiteLLMProvider. Install with: uv pip install litellm"
            ) from e

        super().__init__(model_name)
        self._litellm = litellm
        self.base_url = base_url
        self.api_base = api_base
        self.max_concurrency = max_concurrency
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.api_kwargs = api_kwargs
        self._client: AsyncOpenAI | None = None  # Cached client
        self._request_overrides = self._openrouter_overrides(model_name, api_base)
        self._openai_params = self._openai_supported_params() if self._request_overrides else None

        if is_debug_provider():
            litellm._turn_on_debug()
        else:
            # litellm leaves this flag unannotated, so ty pins it to Literal[False];
            # it is a documented runtime setting, so assign through an untyped view.
            litellm_settings: Any = litellm
            litellm_settings.suppress_debug_info = True

        self._setup_api_keys()

    @staticmethod
    def _openrouter_overrides(model_name: str, api_base: str | None) -> dict[str, Any]:
        """Request fields that send an OpenAI model to OpenRouter.

        Empty unless the model targets OpenAI and only an OpenRouter key is available.
        """
        credentials = resolve_openai_credentials()
        if (
            credentials is None
            or not credentials.via_openrouter
            or not is_openai_model(model_name, api_base)
        ):
            return {}
        return {
            "model": f"openrouter/{openrouter_model_slug(model_name)}",
            "api_base": None,
            "api_key": credentials.api_key,
        }

    def _openai_supported_params(self) -> frozenset[str] | None:
        """Request parameters OpenAI accepts for this model, per LiteLLM's model data.

        LiteLLM only knows these limits for OpenAI's own endpoint, so requests routed
        through OpenRouter are filtered against them (see ``_drop_unsupported_params``).
        """
        name = self.model_name.removeprefix("openai/")
        try:
            params = self._litellm.get_supported_openai_params(
                model=name, custom_llm_provider="openai"
            )
        except Exception:
            return None
        return frozenset(params) if params else None

    def _drop_unsupported_params(self, kwargs: dict[str, Any]) -> None:
        """Remove parameters OpenAI rejects for this model from an OpenRouter request.

        Logprobs are always requested by this provider, so they are dropped whenever
        unsupported. Other parameters are dropped only when ``drop_params`` is set,
        matching LiteLLM's behavior against OpenAI's own endpoint.
        """
        if self._openai_params is None:
            return
        always = {"logprobs", "top_logprobs"}
        candidates = set(kwargs) if kwargs.get("drop_params") else always
        for key in (candidates & _OPENAI_SAMPLING_PARAMS) - self._openai_params:
            kwargs.pop(key, None)

    def _setup_api_keys(self) -> None:
        """Configure LiteLLM with API keys from environment."""
        for env_var, litellm_attr in self._API_KEY_MAPPINGS.items():
            value = os.getenv(env_var)
            if value:
                setattr(self._litellm, litellm_attr, value)

    @property
    def request_model_name(self) -> str:
        """LiteLLM model name sent with requests, which may differ from ``model_name``."""
        return self._request_overrides.get("model", self.model_name)

    @property
    def openai_client_model_name(self) -> str:
        """Model name to send with the client from ``get_openai_client``."""
        if self._request_overrides:
            return openrouter_model_slug(self.model_name)
        return self.model_name

    def get_openai_client(self) -> AsyncOpenAI:
        """Get an AsyncOpenAI client for the configured base_url, or for OpenRouter.

        Returns cached client on subsequent calls to avoid connection pool leaks.

        Returns:
            AsyncOpenAI client pointed at OpenRouter when requests are routed there,
            otherwise at base_url.

        Raises:
            NotImplementedError: If neither applies, since LiteLLM's own routing has
                no single OpenAI-compatible endpoint to hand out.
        """
        if self._client is not None:
            return self._client

        if self._request_overrides:
            base_url = OPENROUTER_BASE_URL
            api_key = self._request_overrides["api_key"]
        elif self.base_url is not None:
            base_url = self.base_url
            api_key = os.getenv("OPENAI_API_KEY", "EMPTY")
        else:
            raise NotImplementedError(
                f"{type(self).__name__} provides an OpenAI client only when base_url is set "
                "or requests are routed through OpenRouter."
            )

        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        return self._client

    async def _generate_single_impl(
        self, request: LMRequest, params: SamplingParams
    ) -> list[LMOutput]:
        """Generate completions for a single request."""
        # Build messages from request
        if request.messages:
            messages = [dict(m) for m in request.messages]
        else:
            messages = [{"role": "user", "content": request.prompt}]

        # Prepare API kwargs
        kwargs: dict[str, Any] = {
            "api_base": self.api_base,
            "model": self.model_name,
            "messages": messages,
            "n": params.num_samples,
            **self.api_kwargs,
            **self._request_overrides,
        }
        # max_tokens=None means "uncapped"; omit the field so the API uses its
        # own context-bounded default rather than receiving a literal null.
        if params.max_tokens is not None:
            kwargs["max_completion_tokens"] = params.max_tokens

        # Always send temperature explicitly to avoid server defaults (OpenAI API defaults to 1.0)
        kwargs["temperature"] = params.temperature
        if params.stop_sequences:
            kwargs["stop"] = list(params.stop_sequences)[:_MAX_STOP_SEQUENCES]
        # Always request logprobs for metrics computation
        kwargs["logprobs"] = True
        kwargs["top_logprobs"] = (
            1  # NOTE: workaround for litellm proxy issue https://github.com/BerriAI/litellm/issues/21932
        )
        self._drop_unsupported_params(kwargs)

        response = await self._litellm.acompletion(**kwargs)

        outputs = []
        for choice in response.choices:
            text = choice.message.content or ""

            # Convert logprobs to standard format
            logprob_entries: list[LogProbEntry] | None = None
            metadata: dict[str, Any] = {}
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
                metadata = {
                    "sum_logits": sum_logits,
                    "num_tokens": num_tokens,
                    "num_tokens_all": num_tokens,
                }

            metadata["finish_reason"] = getattr(choice, "finish_reason", None)
            outputs.append(LMOutput(text=text, logprobs=logprob_entries, metadata=metadata))

        return outputs

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
            trace["provider"] = "LiteLLMProvider"
            trace["endpoint"] = "litellm.acompletion"
            trace["generation_kwargs"] = {
                "max_gen_toks": 50,
                "do_sample": False,
                "temperature": params.temperature,
                "logprobs": True,
                "top_logprobs": 1,
            }
            trace["stop_sequences"] = []
            return trace

        trace["provider"] = "LiteLLMProvider"
        trace["endpoint"] = "litellm.acompletion"
        trace["generation_kwargs"] = {
            "max_gen_toks": params.max_tokens,
            "do_sample": params.do_sample and params.temperature > 0,
            "temperature": params.temperature,
            "logprobs": True,
            "top_logprobs": 1,
            "num_samples": params.num_samples,
        }
        if params.top_p is not None:
            trace["generation_kwargs"]["top_p"] = params.top_p
        trace["stop_sequences"] = list(params.stop_sequences or ())[:_MAX_STOP_SEQUENCES]
        return trace

    async def _generate_single_async(
        self, request: LMRequest, params: SamplingParams
    ) -> list[LMOutput]:
        """Generate completions for a single request."""
        return await retry_with_backoff(
            lambda: self._generate_single_impl(request, params),
            max_retries=self.max_retries,
            retry_delay=self.retry_delay,
            context=f"generate model={self.model_name}",
            sdk_module=self._litellm,
        )

    async def agenerate(
        self,
        requests: list[LMRequest],
        sampling_params: SamplingParams | None = None,
    ) -> list[list[LMOutput]]:
        """Async generate completions.

        This is the native async implementation that should be used
        in async contexts to avoid nested event loops.

        Args:
            requests: Batch of requests to process.
            sampling_params: Sampling configuration.

        Returns:
            List of output lists, one per request.
        """
        from olmo_eval.common.progress import ProgressLogger
        from olmo_eval.inference.utils import dispatch_requests

        logger.info(
            f"Sending {len(requests)} requests for {self.model_name}"
            f" with max_concurrency {self.max_concurrency}"
        )

        params = self._default_sampling_params(sampling_params)
        if params.truncate_prompt_tokens is not None or params.truncation_side is not None:
            logger.warning(
                "truncate_prompt_tokens or truncation_side has been set in the params, "
                "but is not supported for the LiteLLMProvider and will not be used."
            )
        progress = ProgressLogger(total=len(requests), desc="Generating", logger=logger)

        async def process(req: LMRequest) -> list[LMOutput]:
            return await self._generate_single_async(req, params)

        def on_progress(done: int, total: int) -> None:
            progress.update(1)

        results = await dispatch_requests(
            requests,
            process,
            max_in_flight=self.max_concurrency,
            max_retries=self.max_retries,
            on_progress=on_progress,
        )
        progress.close()
        return results

    def generate(
        self,
        requests: list[LMRequest],
        sampling_params: SamplingParams | None = None,
    ) -> list[list[LMOutput]]:
        """Generate completions (sync wrapper).

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
        """Compute logprobs for a single request."""
        params = self._default_sampling_params(params)
        if request.messages:
            default_content = request.messages[0].get("content", "") if request.messages else ""
        else:
            default_content = request.prompt

        outputs = []
        cont_prompts = request.continuation_prompts
        for i, continuation in enumerate(request.continuations or ()):
            content = cont_prompts[i] if cont_prompts else default_content

            kwargs: dict[str, Any] = {
                "api_base": self.api_base,
                "model": self.model_name,
                "messages": [{"role": "user", "content": content}],
                "max_completion_tokens": 50,
                "temperature": params.temperature,
                "logprobs": True,
                "top_logprobs": 1,  # NOTE: workaround for litellm proxy issue https://github.com/BerriAI/litellm/issues/21932
                **self.api_kwargs,
                **self._request_overrides,
            }
            self._drop_unsupported_params(kwargs)

            response = await self._litellm.acompletion(**kwargs)

            completion_logprobs: list[LogProbEntry] = []
            if response.choices:
                choice = response.choices[0]
                logprobs_data = getattr(choice, "logprobs", None)
                if logprobs_data and hasattr(logprobs_data, "content") and logprobs_data.content:
                    for lp in logprobs_data.content:
                        entry: LogProbEntry = {"token": lp.token, "logprob": lp.logprob}
                        lp_bytes = getattr(lp, "bytes", None)
                        if lp_bytes is not None:
                            entry["bytes"] = lp_bytes
                        completion_logprobs.append(entry)

            total = (
                sum(lp["logprob"] for lp in completion_logprobs[:5]) if completion_logprobs else 0.0
            )
            outputs.append(
                LMOutput(
                    text=continuation,
                    logprobs=completion_logprobs[:5] if completion_logprobs else None,
                    metadata={"total_logprob": total},
                )
            )

        return outputs

    async def _logprobs_single_async(
        self,
        request: LMRequest,
        params: SamplingParams | None = None,
    ) -> list[LMOutput]:
        """Compute logprobs for a single request."""
        return await retry_with_backoff(
            lambda: self._logprobs_single_impl(request, params),
            max_retries=self.max_retries,
            retry_delay=self.retry_delay,
            context=f"logprobs model={self.model_name}",
            sdk_module=self._litellm,
        )

    async def alogprobs(
        self,
        requests: list[LMRequest],
        sampling_params: SamplingParams | None = None,
    ) -> list[list[LMOutput]]:
        """Async compute logprobs for continuations.

        Note: Most API providers don't support true continuation logprobs.
        This implementation provides an approximation by generating a response
        and returning those logprobs.

        Args:
            requests: Batch of requests with continuations to score.

        Returns:
            List of output lists with logprobs populated.
        """
        from olmo_eval.common.progress import ProgressLogger
        from olmo_eval.inference.utils import dispatch_requests

        if self._openai_params is not None and "logprobs" not in self._openai_params:
            raise ValueError(
                f"{self.model_name} does not return logprobs, so it cannot score "
                "loglikelihood requests."
            )

        logger.info(
            f"Sending {len(requests)} logprob requests for {self.model_name}"
            f" with max_concurrency {self.max_concurrency}"
        )

        params = self._default_sampling_params(sampling_params)
        if params.truncate_prompt_tokens is not None or params.truncation_side is not None:
            logger.warning(
                "truncate_prompt_tokens or truncation_side has been set in the params, "
                "but is not supported for the LiteLLMProvider and will not be used."
            )
        progress = ProgressLogger(total=len(requests), desc="Logprobs", logger=logger)

        def on_progress(done: int, total: int) -> None:
            progress.update(1)

        results = await dispatch_requests(
            requests,
            lambda request: self._logprobs_single_async(request, params),
            max_in_flight=self.max_concurrency,
            max_retries=self.max_retries,
            on_progress=on_progress,
        )
        progress.close()
        return results

    def logprobs(
        self,
        requests: list[LMRequest],
        sampling_params: SamplingParams | None = None,
    ) -> list[list[LMOutput]]:
        """Compute logprobs for continuations (sync wrapper).

        Note: Most API providers don't support true continuation logprobs.
        This implementation provides an approximation by generating a response
        and returning those logprobs.

        For async contexts, prefer using alogprobs() directly.

        Args:
            requests: Batch of requests with continuations to score.

        Returns:
            List of output lists with logprobs populated.
        """
        return run_async(self.alogprobs(requests, sampling_params))
