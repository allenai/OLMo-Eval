"""Tests for the single-bash-tool Vanillux scaffold."""

from __future__ import annotations

import json
from typing import Any

import pytest

from olmo_eval.common.execution import ExecutionResult
from olmo_eval.common.types import LMOutput, LMRequest, RequestType, SamplingParams
from olmo_eval.common.types.tools import ToolCall
from olmo_eval.harness.config import HarnessConfig
from olmo_eval.harness.scaffolds import get_scaffold
from olmo_eval.harness.scaffolds.vanillux import (
    DEFAULT_SAMPLING_PARAMS,
    OBSERVATION_HEAD_CHARS,
    OBSERVATION_MAX_CHARS,
    OBSERVATION_TAIL_CHARS,
    SUBMIT_MARKER,
    SYSTEM_TEMPLATE,
    TOO_LONG_HINT,
    format_observation,
    parse_action,
    render_instance,
    truncate_observation,
)
from olmo_eval.harness.tools import get_tools
from olmo_eval.inference.base import InferenceProvider
from olmo_eval.inference.errors import CONTEXT_OVERFLOW_KEY, REQUEST_ERROR_KEY


class ScriptedProvider(InferenceProvider):
    """Returns prepared replies in order, recording what it was asked."""

    supports_tools = True

    def __init__(self, replies: list[LMOutput]) -> None:
        super().__init__("scripted")
        self.replies = list(replies)
        self.requests: list[LMRequest] = []
        self.sampling_params: list[SamplingParams | None] = []

    def generate(self, requests, sampling_params=None):  # pragma: no cover - unused
        raise NotImplementedError

    def logprobs(self, requests, sampling_params=None):  # pragma: no cover - unused
        raise NotImplementedError

    async def agenerate(self, requests, sampling_params=None):
        self.requests.append(requests[0])
        self.sampling_params.append(sampling_params)
        reply = self.replies.pop(0) if self.replies else LMOutput(text="Nothing further.")
        return [[reply]]


class FakeBinding:
    """Stands in for an executor binding, replaying scripted command results."""

    def __init__(self, results: list[ExecutionResult] | None = None) -> None:
        self.results = list(results or [])
        self.commands: list[tuple[str, float | None]] = []
        self.released = False

    async def execute_in_session(self, command: str, timeout: float | None = None):
        self.commands.append((command, timeout))
        if self.results:
            return self.results.pop(0)
        return ExecutionResult(success=True, output=f"ran: {command}", exit_code=0)

    async def release(self) -> None:
        self.released = True


class FakeSandboxManager:
    def __init__(self, binding: FakeBinding) -> None:
        self.binding = binding
        self.acquired: list[Any] = []

    async def acquire_binding(self, capabilities=None):
        self.acquired.append(capabilities)
        return self.binding


def bash_reply(command: str, text: str = "THOUGHT: do it", call_id: str = "call-1") -> LMOutput:
    return LMOutput(
        text=text,
        tool_calls=[ToolCall.create(call_id, "bash", {"command": command})],
        metadata={"prompt_tokens": 10, "completion_tokens": 5},
    )


def make_config(**overrides: Any) -> HarnessConfig:
    fields: dict[str, Any] = {
        "name": "tb-test",
        "tools": get_tools(("bash",)),
        "scaffold": "vanillux",
        "max_turns": 5,
    }
    fields.update(overrides)
    return HarnessConfig(**fields)


REQUEST = LMRequest(
    request_type=RequestType.CHAT,
    messages=({"role": "user", "content": "Fix the build."},),
)


async def run(provider, binding: FakeBinding | None = None, config=None, **kwargs):
    binding = binding or FakeBinding()
    scaffold = get_scaffold("vanillux")
    manager = FakeSandboxManager(binding)
    scaffold.set_sandbox_manager(manager)
    result = await scaffold.run(provider, config or make_config(), REQUEST, **kwargs)
    return result, binding


class TestHelpers:
    def test_parse_action_reads_a_bash_command(self) -> None:
        action = parse_action(bash_reply("ls -la"))
        assert action.kind == "command"
        assert action.command == "ls -la"
        assert action.tool_call is not None

    def test_parse_action_recognizes_the_submit_marker(self) -> None:
        action = parse_action(bash_reply(f"echo {SUBMIT_MARKER}"))
        assert action.kind == "done"

    @pytest.mark.parametrize(
        "output",
        [
            LMOutput(text="no call here"),
            LMOutput(text="", tool_calls=[ToolCall.create("c", "python", {"code": "1"})]),
            LMOutput(text="", tool_calls=[ToolCall.create("c", "bash", "not json")]),
            LMOutput(text="", tool_calls=[ToolCall.create("c", "bash", {"command": "   "})]),
        ],
    )
    def test_parse_action_rejects_replies_without_a_usable_call(self, output) -> None:
        assert parse_action(output).kind == "no_tool_call"

    def test_short_output_is_left_alone(self) -> None:
        assert truncate_observation("hello") == "hello"

    def test_long_output_keeps_head_and_tail(self) -> None:
        output = "a" * OBSERVATION_HEAD_CHARS + "b" * 123 + "c" * OBSERVATION_TAIL_CHARS
        assert len(output) > OBSERVATION_MAX_CHARS
        truncated = truncate_observation(output)
        assert truncated.startswith(TOO_LONG_HINT)
        assert "a" * OBSERVATION_HEAD_CHARS in truncated
        assert "c" * OBSERVATION_TAIL_CHARS in truncated
        assert "123 chars elided" in truncated
        assert "b" * 123 not in truncated

    def test_observation_carries_the_exit_code(self) -> None:
        assert format_observation("out\n", 0) == "out\n\n(exit_code=0)"
        assert format_observation("", 2) == "(no output)\n\n(exit_code=2)"

    def test_instance_template_embeds_the_task(self) -> None:
        rendered = render_instance("Do the thing.")
        assert "Do the thing." in rendered
        assert SUBMIT_MARKER in rendered


@pytest.mark.anyio
async def test_commands_run_in_the_session_until_the_model_submits() -> None:
    provider = ScriptedProvider(
        [bash_reply("ls"), bash_reply("make"), bash_reply(f"echo {SUBMIT_MARKER}")]
    )

    result, binding = await run(provider)

    assert [c for c, _ in binding.commands] == ["ls", "make", f"echo {SUBMIT_MARKER}"]
    assert result.metadata["completion_reason"] == "submitted"
    assert result.metadata["steps"] == 3
    assert result.max_turns_reached is False
    assert result.error is None
    assert binding.released
    assert result.trajectory is not None
    assert [t.role for t in result.trajectory.turns] == ["assistant", "tool"] * 3


@pytest.mark.anyio
async def test_the_model_sees_prompts_tool_and_observations() -> None:
    provider = ScriptedProvider([bash_reply("cat README"), bash_reply(f"echo {SUBMIT_MARKER}")])
    binding = FakeBinding([ExecutionResult(success=False, output="boom", exit_code=1)])

    await run(provider, binding)

    first = provider.requests[0]
    assert first.messages[0] == {"role": "system", "content": SYSTEM_TEMPLATE}
    assert first.messages[1]["role"] == "user"
    assert "Fix the build." in first.messages[1]["content"]
    assert first.tools is not None and [t.name for t in first.tools] == ["bash"]

    second = provider.requests[1]
    assert second.messages[2]["role"] == "assistant"
    assert second.messages[2]["tool_calls"][0]["function"]["name"] == "bash"
    assert second.messages[3] == {
        "role": "tool",
        "tool_call_id": "call-1",
        "content": "boom\n\n(exit_code=1)",
    }


@pytest.mark.anyio
async def test_sampling_defaults_match_the_reference_agent() -> None:
    provider = ScriptedProvider([bash_reply(f"echo {SUBMIT_MARKER}")])

    await run(provider)

    assert provider.sampling_params == [DEFAULT_SAMPLING_PARAMS]
    assert DEFAULT_SAMPLING_PARAMS.temperature == 0.7
    assert DEFAULT_SAMPLING_PARAMS.top_p == 0.95
    assert DEFAULT_SAMPLING_PARAMS.max_tokens == 16384


@pytest.mark.anyio
async def test_explicit_sampling_params_are_passed_through() -> None:
    provider = ScriptedProvider([bash_reply(f"echo {SUBMIT_MARKER}")])
    params = SamplingParams(temperature=0.0, max_tokens=99)

    await run(provider, sampling_params=params)

    assert provider.sampling_params == [params]


@pytest.mark.anyio
async def test_a_reply_without_a_call_gets_a_format_error_and_the_run_goes_on() -> None:
    provider = ScriptedProvider(
        [LMOutput(text="I will just describe it."), bash_reply(f"echo {SUBMIT_MARKER}")]
    )

    result, binding = await run(provider)

    assert len(binding.commands) == 1
    follow_up = provider.requests[1].messages
    assert follow_up[2] == {"role": "assistant", "content": "I will just describe it."}
    assert follow_up[3]["role"] == "user"
    assert follow_up[3]["content"].startswith("Format error:")
    assert result.metadata["completion_reason"] == "submitted"
    assert result.metadata["format_errors"] == 0


@pytest.mark.anyio
async def test_a_call_to_another_tool_is_answered_as_that_call() -> None:
    wrong = LMOutput(text="", tool_calls=[ToolCall.create("c-9", "python", {"code": "1"})])
    provider = ScriptedProvider([wrong, bash_reply(f"echo {SUBMIT_MARKER}")])

    await run(provider)

    follow_up = provider.requests[1].messages
    assert follow_up[2]["tool_calls"][0]["id"] == "c-9"
    assert follow_up[3]["role"] == "tool"
    assert follow_up[3]["tool_call_id"] == "c-9"


@pytest.mark.anyio
async def test_a_call_with_broken_arguments_goes_back_as_text() -> None:
    # A truncated reply leaves arguments that are not JSON; sent back as a tool
    # call, the server would reject the whole conversation.
    broken = LMOutput(
        text="THOUGHT: run it",
        tool_calls=[ToolCall.create("c-7", "bash", '{"command": "cat big')],
    )
    provider = ScriptedProvider([broken, bash_reply(f"echo {SUBMIT_MARKER}")])

    result, binding = await run(provider)

    follow_up = provider.requests[1].messages
    assert follow_up[2]["role"] == "assistant"
    assert "tool_calls" not in follow_up[2]
    assert 'bash({"command": "cat big' in follow_up[2]["content"]
    assert follow_up[3]["role"] == "user"
    assert follow_up[3]["content"].startswith("Format error:")
    assert binding.commands == [(f"echo {SUBMIT_MARKER}", 120.0)]
    assert result.metadata["completion_reason"] == "submitted"


@pytest.mark.anyio
async def test_repeated_format_errors_end_the_run() -> None:
    provider = ScriptedProvider([LMOutput(text="nope")] * 5)

    result, binding = await run(provider, max_format_errors=2)

    assert result.metadata["completion_reason"] == "format_errors"
    assert len(provider.requests) == 2
    assert binding.commands == []


@pytest.mark.anyio
async def test_the_step_budget_is_the_configured_max_turns() -> None:
    provider = ScriptedProvider([bash_reply("ls")] * 10)

    result, binding = await run(provider, config=make_config(max_turns=3))

    assert len(binding.commands) == 3
    assert result.max_turns_reached is True
    assert result.metadata["completion_reason"] == "max_steps"


@pytest.mark.anyio
async def test_a_submit_marker_in_the_output_also_ends_the_run() -> None:
    provider = ScriptedProvider([bash_reply("cat notes"), bash_reply("ls")])
    binding = FakeBinding(
        [ExecutionResult(success=True, output=f"notes say {SUBMIT_MARKER}", exit_code=0)]
    )

    result, binding = await run(provider, binding)

    assert len(binding.commands) == 1
    assert result.metadata["completion_reason"] == "submitted"


@pytest.mark.anyio
async def test_command_timeout_reaches_the_session() -> None:
    provider = ScriptedProvider([bash_reply(f"echo {SUBMIT_MARKER}")])

    _, binding = await run(provider, command_timeout=7.5)

    assert binding.commands[0][1] == 7.5


@pytest.mark.anyio
async def test_an_exhausted_agent_timeout_stops_before_the_next_step() -> None:
    provider = ScriptedProvider([bash_reply("ls")] * 3)

    result, binding = await run(provider, agent_timeout=0.0)

    assert binding.commands == []
    assert provider.requests == []
    assert result.metadata["completion_reason"] == "timeout"


@pytest.mark.anyio
async def test_context_overflow_ends_the_run_without_an_error() -> None:
    overflow = LMOutput(
        text="", metadata={CONTEXT_OVERFLOW_KEY: True, REQUEST_ERROR_KEY: "too long"}
    )
    provider = ScriptedProvider([bash_reply("ls"), overflow])

    result, binding = await run(provider)

    assert len(binding.commands) == 1
    assert result.metadata["completion_reason"] == "context_overflow"
    assert result.error is None


@pytest.mark.anyio
async def test_a_provider_failure_is_reported_as_an_error() -> None:
    failed = LMOutput(text="", metadata={REQUEST_ERROR_KEY: "server exploded"})
    provider = ScriptedProvider([failed])

    result, _ = await run(provider)

    assert result.metadata["completion_reason"] == "error"
    assert result.error == "server exploded"


@pytest.mark.anyio
async def test_token_usage_is_summed_over_steps() -> None:
    provider = ScriptedProvider([bash_reply("ls"), bash_reply(f"echo {SUBMIT_MARKER}")])

    result, _ = await run(provider)

    assert result.metadata["usage"] == {"prompt_tokens": 20, "completion_tokens": 10}
    assert [t["cmd"] for t in result.metadata["timing"]] == ["ls", f"echo {SUBMIT_MARKER}"]


@pytest.mark.anyio
async def test_the_config_must_carry_a_session_bash_tool() -> None:
    provider = ScriptedProvider([])
    config = make_config(tools=get_tools(("execute_bash_session",)))

    with pytest.raises(ValueError, match="bash"):
        await run(provider, config=config)


@pytest.mark.anyio
async def test_a_custom_system_prompt_replaces_the_template() -> None:
    provider = ScriptedProvider([bash_reply(f"echo {SUBMIT_MARKER}")])

    await run(provider, config=make_config(system_prompt="Be terse."))

    assert provider.requests[0].messages[0] == {"role": "system", "content": "Be terse."}


def test_tool_call_arguments_round_trip_as_json() -> None:
    call = bash_reply("echo hi").tool_calls[0]
    assert json.loads(call.function.arguments) == {"command": "echo hi"}
