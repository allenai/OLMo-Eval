"""Per-request failures returned by concurrent API providers."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from olmo_eval.common.types import LMOutput, LMRequest, RequestType
from olmo_eval.inference.errors import TerminalProviderError, request_error
from olmo_eval.inference.providers.litellm import LiteLLMProvider
from olmo_eval.inference.providers.vllm_server import VLLMServerProvider


@pytest.mark.parametrize("provider_type", [LiteLLMProvider, VLLMServerProvider])
@pytest.mark.parametrize("method", ["agenerate", "alogprobs"])
def test_failed_request_carries_exception_and_preserves_successes(provider_type, method) -> None:
    provider = provider_type.__new__(provider_type)
    provider.model_name = "mock"
    provider._tool_calls_parsed = None
    provider._openai_params = None
    provider.max_concurrency = 1
    provider.max_retries = 1
    requests = [
        LMRequest(request_type=RequestType.COMPLETION, prompt="ok"),
        LMRequest(request_type=RequestType.COMPLETION, prompt="bad"),
    ]

    async def process(request, params):
        if request.prompt == "bad":
            raise ValueError("400: rejected request")
        return [LMOutput(text="answer")]

    process_mock = AsyncMock(side_effect=process)
    attribute = "_generate_single_async" if method == "agenerate" else "_logprobs_single_async"
    setattr(provider, attribute, process_mock)

    outputs = asyncio.run(getattr(provider, method)(requests))
    assert outputs[0][0].text == "answer"
    assert request_error(outputs[0]) is None
    assert request_error(outputs[1]) == "ValueError: 400: rejected request"
    assert process_mock.await_count == 3


@pytest.mark.parametrize("provider_type", [LiteLLMProvider, VLLMServerProvider])
def test_loglikelihood_without_continuations_keeps_successful_empty_result(provider_type) -> None:
    provider = provider_type.__new__(provider_type)
    provider.model_name = "mock"
    provider._tool_calls_parsed = None
    provider._openai_params = None
    provider.max_concurrency = 1
    provider.max_retries = 0
    provider._logprobs_single_async = AsyncMock(return_value=[])
    request = LMRequest(request_type=RequestType.LOGLIKELIHOOD, prompt="ok", continuations=())

    assert asyncio.run(provider.alogprobs([request])) == [[]]


@pytest.mark.parametrize("provider_type", [LiteLLMProvider, VLLMServerProvider])
def test_terminal_failure_still_propagates(provider_type) -> None:
    provider = provider_type.__new__(provider_type)
    provider.model_name = "mock"
    provider._tool_calls_parsed = None
    provider._openai_params = None
    provider.max_concurrency = 1
    provider.max_retries = 1
    provider._generate_single_async = AsyncMock(side_effect=TerminalProviderError("engine died"))
    request = LMRequest(request_type=RequestType.COMPLETION, prompt="ok")

    with pytest.raises(TerminalProviderError, match="engine died"):
        asyncio.run(provider.agenerate([request]))


@pytest.mark.parametrize("provider_type", [LiteLLMProvider, VLLMServerProvider])
def test_failed_request_bounds_large_error_bodies(provider_type) -> None:
    provider = provider_type.__new__(provider_type)
    provider.model_name = "mock"
    provider._tool_calls_parsed = None
    provider._openai_params = None
    provider.max_concurrency = 1
    provider.max_retries = 0
    provider._generate_single_async = AsyncMock(side_effect=ValueError("gateway " + "x" * 10_000))
    request = LMRequest(request_type=RequestType.COMPLETION, prompt="q")

    outputs = asyncio.run(provider.agenerate([request]))
    error = request_error(outputs[0])

    assert error is not None and error.startswith("ValueError: gateway ")
    assert error.endswith("…")
    assert len(error) <= len("ValueError: ") + 501


def test_dispatch_requests_preserves_order_and_records_individual_failures() -> None:
    from olmo_eval.inference.errors import REQUEST_ERROR_KEY
    from olmo_eval.inference.utils import dispatch_requests

    async def process(request):
        if request.prompt == "failed":
            raise RuntimeError("service down")
        if request.prompt == "missing":
            return None
        await asyncio.sleep(0.01)
        return [LMOutput(text=request.prompt)]

    requests = [
        LMRequest(request_type=RequestType.COMPLETION, prompt=p)
        for p in ["first", "failed", "missing", "last"]
    ]
    outputs = asyncio.run(dispatch_requests(requests, process))
    assert [batch[0].text if batch else None for batch in outputs] == ["first", "", None, "last"]
    assert outputs[1][0].metadata[REQUEST_ERROR_KEY] == "RuntimeError: service down"
