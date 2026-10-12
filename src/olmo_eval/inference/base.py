"""Inference provider base class and protocol definition."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from olmo_eval.common.types import LMOutput, LMRequest, RequestType, SamplingParams

if TYPE_CHECKING:
    from openai import AsyncOpenAI


class InferenceProvider(ABC):
    """Abstract base class for language model inference providers.

    All providers must implement `generate` and `logprobs` methods.
    Providers that support async operations should override `agenerate`
    and `alogprobs` for better performance in async contexts.
    """

    #: Whether this provider can render ``LMRequest.images``. Providers that
    #: cannot must not receive image requests: the harness rejects them so a
    #: vision task on a text provider fails loudly instead of scoring
    #: text-only completions.
    supports_images: bool = False

    model_name: str

    #: Whether this provider sends ``LMRequest.tools`` to the model. A provider
    #: that leaves this False drops them silently, which would score a
    #: tool-calling task as if the model had been shown no functions at all, so
    #: the harness refuses such a request instead.
    supports_tools: bool = False

    def __init__(self, model_name: str) -> None:
        """Initialize the provider.

        Args:
            model_name: Model identifier or path.
        """
        self.model_name = model_name

    @abstractmethod
    def generate(
        self,
        requests: list[LMRequest],
        sampling_params: SamplingParams | None = None,
    ) -> list[list[LMOutput]]:
        """Generate completions for a batch of requests.

        Args:
            requests: Batch of requests to process.
            sampling_params: Sampling configuration.

        Returns:
            List of output lists, one per request. Each inner list contains
            `sampling_params.num_samples` outputs.
        """
        ...

    @abstractmethod
    def logprobs(
        self,
        requests: list[LMRequest],
        sampling_params: SamplingParams | None = None,
    ) -> list[list[LMOutput]]:
        """Compute log probabilities for continuations.

        Args:
            requests: Batch of requests with continuations to score.
            sampling_params: Optional sampling configuration used for scoring requests.

        Returns:
            List of output lists. Each inner list has one LMOutput per
            continuation in the request, with logprobs populated.
        """
        ...

    async def agenerate(
        self,
        requests: list[LMRequest],
        sampling_params: SamplingParams | None = None,
    ) -> list[list[LMOutput]]:
        """Async version of generate.

        Args:
            requests: Batch of requests to process.
            sampling_params: Sampling configuration.

        Returns:
            List of output lists, one per request.

        Raises:
            NotImplementedError: If provider doesn't support async generation.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support async generation")

    async def alogprobs(
        self,
        requests: list[LMRequest],
        sampling_params: SamplingParams | None = None,
    ) -> list[list[LMOutput]]:
        """Async version of logprobs.

        Args:
            requests: Batch of requests with continuations to score.
            sampling_params: Optional sampling configuration used for scoring requests.

        Returns:
            List of output lists with logprobs populated.

        Raises:
            NotImplementedError: If provider doesn't support async logprobs.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support async logprobs")

    def describe_request(
        self,
        request: LMRequest,
        sampling_params: SamplingParams | None = None,
    ) -> dict[str, Any] | None:
        """Describe the effective request parameters sent to the provider.

        This trace is persisted into ``requests.jsonl`` so downstream overrides
        applied by the provider can be replayed later.
        """
        params = self._default_sampling_params(sampling_params)
        trace: dict[str, Any] = {
            "provider": type(self).__name__,
            "generation_kwargs": {},
            "stop_sequences": [],
        }

        if request.request_type == RequestType.LOGLIKELIHOOD:
            trace["mode"] = "logprobs"
            trace["generation_kwargs"] = {
                "temperature": params.temperature,
            }
            return trace

        trace["mode"] = "generate"
        generation_kwargs: dict[str, Any] = {
            "max_gen_toks": params.max_tokens,
            "do_sample": params.do_sample and params.temperature > 0,
            "temperature": params.temperature,
        }
        if params.top_p is not None:
            generation_kwargs["top_p"] = params.top_p
        if params.top_k is not None:
            generation_kwargs["top_k"] = params.top_k
        if params.num_samples != 1:
            generation_kwargs["num_samples"] = params.num_samples
        if params.logprobs is not None:
            generation_kwargs["logprobs"] = params.logprobs

        trace["generation_kwargs"] = generation_kwargs
        trace["stop_sequences"] = list(params.stop_sequences or ())
        return trace

    def _default_sampling_params(self, sampling_params: SamplingParams | None) -> SamplingParams:
        """Return sampling params with defaults applied."""
        return sampling_params or SamplingParams()

    def get_tokenizer(self) -> Any:
        """Get the tokenizer for this provider.

        Returns:
            The tokenizer instance.

        Raises:
            NotImplementedError: If provider doesn't support tokenizer access.
        """
        raise NotImplementedError(f"{type(self).__name__} does not provide tokenizer access")

    def get_openai_client(self) -> AsyncOpenAI:
        """Get an AsyncOpenAI client for this provider.

        Used by backends that need an OpenAI-compatible client
        (e.g., OpenAI Agents SDK).

        Returns:
            AsyncOpenAI client.

        Raises:
            NotImplementedError: If provider doesn't have an OpenAI-compatible interface.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not provide an OpenAI client. "
            f"Use a provider with OpenAI API support (e.g., VLLMServerProvider, LiteLLMProvider)."
        )


def require_tool_support(provider: InferenceProvider) -> None:
    """Fail before sending tools to a provider that would drop them.

    A provider that ignores ``tools`` shows the model a question with no
    functions attached, which it answers in prose, so every instance of a
    tool-calling task scores zero. That reads like a weak model rather than a
    misconfigured run.
    """
    if getattr(provider, "supports_tools", False):
        return
    raise ValueError(
        f"{type(provider).__name__} does not support tool requests, but this "
        "request carries function schemas, which would be dropped. Run this task "
        "with a provider that sends them (e.g. the default provider.kind="
        "vllm_server), or with a task variant that writes the functions into the "
        "prompt text instead."
    )
