"""Tests for VLLMServerProvider interaction with Beaker status reporting."""

import asyncio
from unittest.mock import AsyncMock, patch

from olmo_eval.common.types import LMOutput, LMRequest, RequestType


def test_agenerate_leaves_beaker_status_to_the_runner() -> None:
    """Per-call progress would overwrite the runner's cumulative count."""
    from olmo_eval.inference.providers.vllm_server import VLLMServerProvider

    with patch("olmo_eval.inference.providers.vllm_server.BeakerStatusReporter"):
        provider = VLLMServerProvider(
            "test-model", base_url="http://localhost:8000/v1", max_model_len=4096
        )
    reporter = provider._beaker_reporter
    reporter.reset_mock()

    requests = [LMRequest(request_type=RequestType.COMPLETION, prompt=f"p{i}") for i in range(3)]
    output = [LMOutput(text="out")]
    with patch.object(provider, "_generate_single_async", AsyncMock(return_value=output)):
        results = asyncio.run(provider.agenerate(requests))

    assert results == [output] * 3
    assert reporter.method_calls == []
