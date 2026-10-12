"""Tests for starting vLLM with tool-call parsing when tasks send tools."""

from __future__ import annotations

from olmo_eval.common.types import Instance, LMRequest, RequestType, ToolSchema
from olmo_eval.harness.config import HarnessConfig
from olmo_eval.inference.providers.config import ProviderConfig
from olmo_eval.runners.asynq.runner import AsyncEvalRunner
from olmo_eval.runners.asynq.types import QueueItem

TOOL = ToolSchema(name="lookup", description="Look something up.", parameters={})


class Runner:
    """The runner's decision, on a config, without the rest of the runner."""

    _enable_tool_parsing_for_tool_requests = AsyncEvalRunner._enable_tool_parsing_for_tool_requests

    def __init__(self, provider: ProviderConfig) -> None:
        self.harness_config = HarnessConfig(name="test", provider=provider)


def items(*, with_tools: bool) -> list[QueueItem]:
    return [
        QueueItem(
            model_name="model",
            task_id="bfcl_simple",
            instance_idx=0,
            instance=Instance(question="Look up the weather."),
            request=LMRequest(
                request_type=RequestType.CHAT,
                messages=({"role": "user", "content": "Look up the weather."},),
                tools=(TOOL,) if with_tools else None,
            ),
        )
    ]


def tool_choice_after(provider: ProviderConfig, *, with_tools: bool = True) -> object:
    runner = Runner(provider)
    runner._enable_tool_parsing_for_tool_requests(items(with_tools=with_tools))
    return dict(runner.harness_config.provider.kwargs).get("enable_auto_tool_choice")


def test_a_managed_server_parses_tool_calls_when_a_task_sends_tools() -> None:
    assert tool_choice_after(ProviderConfig(kind="vllm_server", model="m")) is True


def test_a_task_without_tools_leaves_the_server_alone() -> None:
    provider = ProviderConfig(kind="vllm_server", model="m")

    assert tool_choice_after(provider, with_tools=False) is None


def test_an_explicit_setting_is_kept() -> None:
    provider = ProviderConfig(
        kind="vllm_server", model="m", kwargs={"enable_auto_tool_choice": False}
    )

    assert tool_choice_after(provider) is False


def test_an_external_server_is_not_reconfigured() -> None:
    provider = ProviderConfig(kind="vllm_server", model="m", base_url="http://host:8000/v1")

    assert tool_choice_after(provider) is None


def test_other_providers_are_left_alone() -> None:
    assert tool_choice_after(ProviderConfig(kind="litellm", model="gpt-4o")) is None
