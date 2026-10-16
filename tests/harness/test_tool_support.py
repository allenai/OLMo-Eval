"""Tests for refusing requests whose tool schemas would be silently dropped."""

from __future__ import annotations

import asyncio
import queue
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from olmo_eval.common.types import (
    Instance,
    LMOutput,
    LMRequest,
    RequestType,
    SamplingParams,
    ToolSchema,
)
from olmo_eval.harness import Harness, HarnessConfig
from olmo_eval.harness.presets import get_harness_preset, list_harness_presets
from olmo_eval.inference.base import InferenceProvider
from olmo_eval.inference.providers.vllm_server import VLLMServerProvider
from olmo_eval.runners.asynq.processing import process_batch
from olmo_eval.runners.asynq.results import is_hard_failure
from olmo_eval.runners.asynq.types import QueueItem, ResultItem

TOOL = ToolSchema(name="lookup", description="Look something up.", parameters={})


class ToollessProvider(InferenceProvider):
    """A provider that would drop the tools it is handed."""

    def generate(
        self, requests: list[LMRequest], sampling_params: SamplingParams | None = None
    ) -> list[list[LMOutput]]:
        return [[LMOutput(text="")] for _ in requests]

    def logprobs(
        self, requests: list[LMRequest], sampling_params: SamplingParams | None = None
    ) -> list[list[LMOutput]]:
        return [[LMOutput(text="")] for _ in requests]


class ToolCarryingProvider(ToollessProvider):
    supports_tools = True


def harness_with(provider: InferenceProvider) -> Harness:
    harness = Harness(HarnessConfig(name="test"))
    harness._provider = provider
    return harness


def tool_request() -> LMRequest:
    return LMRequest(
        request_type=RequestType.CHAT,
        messages=({"role": "user", "content": "Look up the weather."},),
        tools=(TOOL,),
    )


def test_a_provider_declares_no_tool_support_by_default() -> None:
    assert InferenceProvider.supports_tools is False


def test_a_request_with_tools_is_refused_by_a_provider_that_would_drop_them() -> None:
    harness = harness_with(ToollessProvider("stub"))

    with pytest.raises(ValueError, match="does not support tool requests"):
        harness._apply_config(tool_request())


def test_a_request_without_tools_is_unaffected() -> None:
    harness = harness_with(ToollessProvider("stub"))
    request = LMRequest(
        request_type=RequestType.CHAT,
        messages=({"role": "user", "content": "Hello."},),
    )

    assert harness._apply_config(request).tools is None


TOOL_CARRYING_PRESETS = [
    name for name in list_harness_presets() if get_harness_preset(name).has_tools
]


def test_every_tool_carrying_preset_runs_through_a_scaffold() -> None:
    # The exemption below relies on this: a preset with its own tools hands them
    # to a scaffold, which sends them itself.
    assert TOOL_CARRYING_PRESETS
    assert all(get_harness_preset(name).scaffold for name in TOOL_CARRYING_PRESETS)


@pytest.mark.parametrize("preset", TOOL_CARRYING_PRESETS)
def test_a_scaffolded_harness_with_its_own_tools_is_not_refused(preset: str) -> None:
    # An API model resolves to a provider that does not send request tools, but
    # the scaffold sends the harness's tools over its own client.
    harness = Harness(get_harness_preset(preset))
    harness._provider = ToollessProvider("stub")
    request = LMRequest(
        request_type=RequestType.CHAT,
        messages=({"role": "user", "content": "Find it."},),
    )

    assert harness._apply_config(request).tools


def test_a_scaffold_is_trusted_with_request_tools_too() -> None:
    harness = Harness(HarnessConfig(name="test", scaffold="simple"))
    harness._provider = ToollessProvider("stub")

    assert harness._apply_config(tool_request()).tools == (TOOL,)


def test_a_provider_that_carries_tools_passes_them_through() -> None:
    harness = harness_with(ToolCarryingProvider("stub"))

    assert harness._apply_config(tool_request()).tools == (TOOL,)


def test_the_vllm_server_provider_carries_tools() -> None:
    assert VLLMServerProvider.supports_tools is True


def unstarted_server_provider(tool_calls_parsed: bool | None) -> VLLMServerProvider:
    """A provider object without the server its constructor would start."""
    provider = object.__new__(VLLMServerProvider)
    provider._tool_calls_parsed = tool_calls_parsed
    return provider


def test_a_server_started_without_tool_choice_refuses_a_tool_batch() -> None:
    provider = unstarted_server_provider(False)

    with pytest.raises(ValueError, match="enable-auto-tool-choice"):
        asyncio.run(provider.agenerate([tool_request()], SamplingParams()))


def test_the_refusal_fails_every_instance_instead_of_scoring_it() -> None:
    # Raised per request, the refusal was caught by dispatch and came back as an
    # empty output with no error, which scores zero. Raised for the batch, it
    # reaches the runner as a failure of each instance.
    provider = unstarted_server_provider(False)
    harness = SimpleNamespace(
        provider=SimpleNamespace(
            describe_request=Mock(return_value=None), agenerate=provider.agenerate
        ),
        _apply_config=lambda request: request,
        flush_metrics=Mock(),
    )
    items = [
        QueueItem(
            model_name="model",
            task_id="bfcl_simple",
            instance_idx=index,
            instance=Instance(question="Look up the weather."),
            request=tool_request(),
        )
        for index in range(2)
    ]
    result_queue: queue.Queue[ResultItem] = queue.Queue()

    asyncio.run(process_batch(items, harness, result_queue))  # type: ignore[arg-type]

    results = [result_queue.get_nowait() for _ in items]
    assert all(is_hard_failure(result) for result in results)
    assert all("enable-auto-tool-choice" in (result.error or "") for result in results)


def test_a_server_that_parses_tool_calls_takes_a_tool_batch() -> None:
    provider = unstarted_server_provider(True)

    provider._refuse_tools_without_parsing([tool_request()])


def test_an_external_server_is_given_the_benefit_of_the_doubt() -> None:
    # Whether an existing server parses tool calls is not knowable from here,
    # so the request goes out rather than being refused.
    provider = unstarted_server_provider(None)

    with pytest.raises(Exception) as caught:
        asyncio.run(
            provider._generate_chat(
                client=None,  # ty: ignore[invalid-argument-type]
                request=tool_request(),
                params=SamplingParams(),
            )
        )

    assert "enable-auto-tool-choice" not in str(caught.value)
