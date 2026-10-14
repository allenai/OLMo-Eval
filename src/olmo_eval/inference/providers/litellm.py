"""LiteLLM provider for API-based inference."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from olmo_eval.common.debug import is_debug_provider
from olmo_eval.common.logging import get_logger
from olmo_eval.common.types import LMOutput, LMRequest, LogProbEntry, RequestType, SamplingParams
from olmo_eval.inference.base import InferenceProvider
from olmo_eval.inference.retry import retry_with_backoff
from olmo_eval.inference.utils import run_async

if TYPE_CHECKING:
    from openai import AsyncOpenAI

# Maximum stop sequences supported by OpenAI-compatible APIs
_MAX_STOP_SEQUENCES = 4

logger = get_logger(__name__)

# OpenAI fits high-detail images into a 2048x2048 square before tiling, so larger
# images can be downscaled client-side without changing what the model sees.
_MAX_IMAGE_SIDE = 2048


def _image_data_url(image: Any) -> str:
    """Encode a PIL image as a base64 PNG data URL, downscaled to the API's input bound."""
    import base64
    import io

    if image.mode not in ("RGB", "RGBA", "L", "LA"):
        image = image.convert("RGBA" if "A" in image.getbands() else "RGB")
    if max(image.size) > _MAX_IMAGE_SIDE:
        image = image.copy()
        image.thumbnail((_MAX_IMAGE_SIDE, _MAX_IMAGE_SIDE))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _attach_images(messages: list[dict[str, Any]], images: tuple[Any, ...]) -> None:
    """Put a request's images in the last user turn.

    A user turn whose content is a part list may mark image positions with
    ``{"type": "image"}`` parts (the Hugging Face chat-template convention); the
    images fill them in order. Otherwise the images precede the text, the order the
    local multimodal providers use, so an API model sees the same prompt layout as a
    local one.
    """
    from olmo_eval.common.images import resolve_images

    resolved = resolve_images(images)
    if not resolved:
        return
    user_turn = next((m for m in reversed(messages) if m.get("role") == "user"), None)
    if user_turn is None:
        raise ValueError("A request with images needs a user message to attach them to.")
    content = user_turn.get("content")
    parts = (
        list(content) if isinstance(content, list) else [{"type": "text", "text": content or ""}]
    )
    image_parts = [
        {"type": "image_url", "image_url": {"url": _image_data_url(image)}} for image in resolved
    ]
    slots = [i for i, part in enumerate(parts) if part.get("type") == "image"]
    if not slots:
        user_turn["content"] = image_parts + parts
        return
    if len(slots) != len(image_parts):
        raise ValueError(
            f"User turn marks {len(slots)} image position(s) but the request has "
            f"{len(image_parts)} image(s)."
        )
    for slot, image_part in zip(slots, image_parts, strict=True):
        parts[slot] = image_part
    user_turn["content"] = parts


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
            import litellm  # type: ignore[ty:unresolved-import]
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

        if is_debug_provider():
            litellm._turn_on_debug()
        else:
            litellm.suppress_debug_info = True

        self._setup_api_keys()

    def _setup_api_keys(self) -> None:
        """Configure LiteLLM with API keys from environment."""
        for env_var, litellm_attr in self._API_KEY_MAPPINGS.items():
            value = os.getenv(env_var)
            if value:
                setattr(self._litellm, litellm_attr, value)

    def get_openai_client(self) -> AsyncOpenAI | None:
        """Get an AsyncOpenAI client if base_url is configured.

        Returns cached client on subsequent calls to avoid connection pool leaks.

        Returns:
            AsyncOpenAI client if base_url is set, None otherwise.
        """
        if self.base_url is None:
            return None

        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(
                base_url=self.base_url,
                api_key=os.getenv("OPENAI_API_KEY", "EMPTY"),
            )

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
        if request.images:
            _attach_images(messages, request.images)

        # Prepare API kwargs
        kwargs: dict[str, Any] = {
            "api_base": self.api_base,
            "model": self.model_name,
            "messages": messages,
            "n": params.num_samples,
            **self.api_kwargs,
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
        from olmo_eval.inference.dispatch import dispatch_concurrent

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

        results = await dispatch_concurrent(
            requests,
            process,
            max_in_flight=self.max_concurrency,
            max_retries=self.max_retries,
            on_progress=on_progress,
        )
        progress.close()
        return [r if r is not None else [] for r in results]

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

            response = await self._litellm.acompletion(
                api_base=self.api_base,
                model=self.model_name,
                messages=[{"role": "user", "content": content}],
                max_completion_tokens=50,
                temperature=params.temperature,
                logprobs=True,
                top_logprobs=1,  # NOTE: workaround for litellm proxy issue https://github.com/BerriAI/litellm/issues/21932
                **self.api_kwargs,
            )

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
        from olmo_eval.inference.dispatch import dispatch_concurrent

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

        results = await dispatch_concurrent(
            requests,
            lambda request: self._logprobs_single_async(request, params),
            max_in_flight=self.max_concurrency,
            max_retries=self.max_retries,
            on_progress=on_progress,
        )
        progress.close()
        return [r if r is not None else [] for r in results]

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
