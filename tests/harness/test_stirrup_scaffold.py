"""Tests for the Stirrup-style agent scaffold."""

from __future__ import annotations

import asyncio
import json
import re
import shlex
from typing import Any

import pytest

from olmo_eval.common.execution import ExecutionResult
from olmo_eval.common.types import LMOutput, LMRequest, RequestType, SamplingParams
from olmo_eval.common.types.tools import ToolCall
from olmo_eval.harness.config import HarnessConfig
from olmo_eval.harness.scaffolds import get_scaffold
from olmo_eval.harness.scaffolds.stirrup import (
    BASE_SYSTEM_PROMPT,
    CONTINUE_MESSAGE,
    DEFAULT_SAMPLING_PARAMS,
    LAST_TURN_MESSAGE,
    MAX_OUTPUT_CHARS,
    SUMMARIZER_PROMPT,
    SUMMARIZER_TEXT_ONLY,
    SUMMARY_ACKNOWLEDGEMENT,
    CommandResult,
    build_system_prompt,
    format_shell_result,
    truncate_output,
    turns_remaining_message,
)
from olmo_eval.harness.tools import get_tools
from olmo_eval.inference.base import InferenceProvider
from olmo_eval.inference.errors import CONTEXT_OVERFLOW_KEY, REQUEST_ERROR_KEY

TOOLS = ("code_exec", "finish", "abandon_task_finish")


class ScriptedProvider(InferenceProvider):
    """Returns prepared replies in order, recording what it was asked."""

    supports_tools = True

    def __init__(self, replies: list[LMOutput], max_length: int | None = None) -> None:
        super().__init__("scripted")
        self.replies = list(replies)
        self.requests: list[LMRequest] = []
        self.sampling_params: list[SamplingParams | None] = []
        self.max_length = max_length

    def generate(self, requests, sampling_params=None):  # pragma: no cover - unused
        raise NotImplementedError

    def logprobs(self, requests, sampling_params=None):  # pragma: no cover - unused
        raise NotImplementedError

    async def agenerate(self, requests, sampling_params=None):
        self.requests.append(requests[0])
        self.sampling_params.append(sampling_params)
        reply = self.replies.pop(0) if self.replies else LMOutput(text="Nothing further.")
        return [[reply]]


#: A scripted command result for a command that never exits.
HANGS = None


class FakeExecutor:
    """Plays the sandbox side of ``run_command`` and answers ``test -f``.

    Each agent command takes the next scripted result, or succeeds and echoes
    itself; a ``HANGS`` result never writes an exit status.
    """

    def __init__(self, files: set[str] | None = None, results: list[Any] | None = None):
        self.files = files or set()
        self.results = list(results or [])
        self.commands: list[str] = []
        self.scripts: dict[str, str] = {}
        self.runs: dict[str, CommandResult | None] = {}
        self.killed: list[str] = []
        self.fail_with: Exception | None = None

    async def write_files(self, files, timeout=None):
        for path, content in files.items():
            self.scripts[path.rsplit("/", 1)[0]] = content

    async def execute_command(self, command, timeout=None, **_):
        if self.fail_with is not None:
            raise self.fail_with
        if command.startswith("test -f "):
            exists = shlex.split(command)[2] in self.files
            return ExecutionResult(success=exists, output="", exit_code=0 if exists else 1)
        run = re.search(r"/tmp/\.stirrup/\w+", command)
        assert run is not None, command
        run_dir = run.group(0)
        if command.startswith("setsid "):
            script = self.scripts[run_dir]
            self.commands.append(script)
            default = CommandResult(0, f"ran: {script}", "")
            self.runs[run_dir] = self.results.pop(0) if self.results else default
        elif command.startswith("cat "):
            result = self.runs[run_dir]
            return ExecutionResult(True, "" if result is HANGS else f"{result.exit_code}\n")
        elif command.startswith("size="):
            result = self.runs[run_dir]
            stream = re.search(r"/(stdout|stderr)\b", command).group(1)
            text = "" if result is HANGS else getattr(result, stream)
            return ExecutionResult(True, self._read(text))
        elif command.startswith("kill "):
            self.killed.append(run_dir)
        return ExecutionResult(True, "")

    @staticmethod
    def _read(text: str) -> str:
        """What the shell's capped read of an output file prints."""
        if len(text) <= MAX_OUTPUT_CHARS:
            return text
        half = MAX_OUTPUT_CHARS // 2
        return f"{len(text)}\n{text[:half]}\n---STIRRUP-OUTPUT-TAIL---\n{text[-half:]}"


class FakeBinding:
    def __init__(self, executor: FakeExecutor) -> None:
        self.executor = executor
        self.released = False

    async def release(self) -> None:
        self.released = True


class FakeSandboxManager:
    def __init__(self, files: set[str] | None = None, results: list[Any] | None = None):
        self.executor = FakeExecutor(files, results)
        self.binding = FakeBinding(self.executor)

    async def acquire_binding(self, capabilities=None):
        return self.binding


def call(name: str, arguments: dict[str, Any] | str, call_id: str = "c1") -> ToolCall:
    return ToolCall.create(call_id, name, arguments)


def reply(*calls: ToolCall, text: str = "", tokens: int = 10) -> LMOutput:
    return LMOutput(
        text=text,
        tool_calls=list(calls),
        metadata={"prompt_tokens": tokens, "completion_tokens": 0},
    )


def finish(paths: list[str], reason: str = "done", call_id: str = "f1") -> ToolCall:
    return call("finish", {"reason": reason, "paths": paths}, call_id)


REQUEST = LMRequest(
    request_type=RequestType.CHAT, messages=({"role": "user", "content": "Write the report."},)
)


def run(provider, manager=None, max_turns: int = 30, **config_overrides: Any):
    manager = manager or FakeSandboxManager()
    fields: dict[str, Any] = {
        "name": "stirrup-test",
        "tools": get_tools(TOOLS),
        "scaffold": "stirrup",
        "max_turns": max_turns,
        "system_prompt": "Task instructions.",
    }
    fields.update(config_overrides)
    scaffold = get_scaffold("stirrup")
    scaffold.set_sandbox_manager(manager)
    result = asyncio.run(scaffold.run(provider, HarnessConfig(**fields), REQUEST))
    return result, manager


def tool_messages(request: LMRequest) -> list[dict[str, Any]]:
    return [m for m in request.messages if m["role"] == "tool"]


class TestHelpers:
    def test_system_prompt_puts_the_task_instructions_after_the_base_prompt(self) -> None:
        prompt = build_system_prompt(500, "Be thorough.")
        assert prompt.startswith(BASE_SYSTEM_PROMPT.format(max_turns=500))
        assert " You are not able to interact with the user during the task." in prompt
        assert prompt.endswith("\n\nFollow these instructions from the User:\nBe thorough.")

    def test_turn_reminders_count_down_to_the_last_turn(self) -> None:
        assert "3 turns remaining" in turns_remaining_message(3)
        assert turns_remaining_message(1) == LAST_TURN_MESSAGE

    def test_truncation_keeps_both_ends_and_states_the_original_length(self) -> None:
        text = "a" * MAX_OUTPUT_CHARS + "b" * 100
        truncated = truncate_output(text)
        assert truncated.startswith("a" * (MAX_OUTPUT_CHARS // 2))
        assert truncated.endswith("b" * 100)
        assert f"original {len(text)} characters" in truncated
        assert truncate_output("short") == "short"

    def test_shell_results_are_xml(self) -> None:
        assert format_shell_result(CommandResult(2, "out", "err")) == (
            "<shell_results>\n<exit_code>2</exit_code>\n<stdout>out</stdout>"
            "\n<stderr>err</stderr>\n</shell_results>"
        )
        timed_out = format_shell_result(CommandResult(-1, "", "slow", timed_out=True))
        assert timed_out == (
            "<shell_results>\n<error_kind>timeout</error_kind>\n<details>slow</details>"
            "\n</shell_results>"
        )


class TestRun:
    def test_finish_ends_the_run_once_its_files_exist(self) -> None:
        provider = ScriptedProvider(
            [
                reply(call("code_exec", {"cmd": "python make.py"})),
                reply(finish(["/home/user/report.pdf"])),
            ]
        )
        result, manager = run(provider, FakeSandboxManager({"/home/user/report.pdf"}))

        assert result.metadata["completion_reason"] == "finished"
        assert result.metadata["finish"] == {
            "tool": "finish",
            "reason": "done",
            "paths": ["/home/user/report.pdf"],
        }
        assert result.final_output.text == "done"
        assert result.error is None
        assert manager.executor.commands == ["python make.py"]
        first = provider.requests[0]
        assert first.messages[0]["content"] == build_system_prompt(30, "Task instructions.")
        assert first.messages[1] == {"role": "user", "content": "Write the report."}
        assert [t.name for t in first.tools] == list(TOOLS)
        assert provider.sampling_params[0] == DEFAULT_SAMPLING_PARAMS
        command_result = tool_messages(provider.requests[1])[0]["content"]
        assert command_result == format_shell_result(CommandResult(0, "ran: python make.py", ""))
        assert manager.binding.released

    def test_finish_naming_a_missing_file_is_refused(self) -> None:
        provider = ScriptedProvider(
            [reply(finish(["/home/user/missing.pdf"])), reply(finish([], reason="nothing"))]
        )
        result, _ = run(provider)

        refusal = tool_messages(provider.requests[1])[0]["content"]
        assert refusal.startswith("ERROR: Files do not exist: ['/home/user/missing.pdf']")
        assert result.metadata["finish"]["reason"] == "nothing"
        assert result.metadata["steps"] == 2

    def test_abandon_ends_the_run(self) -> None:
        provider = ScriptedProvider([reply(call("abandon_task_finish", {"reason": "no inputs"}))])
        result, _ = run(provider)

        assert result.metadata["completion_reason"] == "abandoned"
        assert result.metadata["finish"] == {
            "tool": "abandon_task_finish",
            "reason": "no inputs",
            "paths": [],
        }

    def test_two_finish_calls_in_one_turn_are_both_refused_after_the_command(self) -> None:
        provider = ScriptedProvider(
            [
                reply(
                    finish([], call_id="f1"),
                    call("abandon_task_finish", {"reason": "x"}, "a1"),
                    call("code_exec", {"cmd": "ls"}, "e1"),
                ),
                reply(finish([], call_id="f2")),
            ]
        )
        result, manager = run(provider)

        results = tool_messages(provider.requests[1])
        assert [r["tool_call_id"] for r in results] == ["e1", "f1", "a1"]
        assert all("multiple finish tools" in r["content"] for r in results[1:])
        assert manager.executor.commands == ["ls"]
        assert result.metadata["completion_reason"] == "finished"

    def test_bad_calls_get_errors_and_keep_the_history_replayable(self) -> None:
        provider = ScriptedProvider(
            [
                reply(
                    call("code_exec", "not json", "e1"),
                    call("browse", {"url": "x"}, "b1"),
                    call("finish", {"reason": "done"}, "f1"),
                ),
                reply(finish([])),
            ]
        )
        run(provider)

        second = provider.requests[1]
        sent_calls = second.messages[2]["tool_calls"]
        assert sent_calls[0]["function"]["arguments"] == "{}"
        contents = [m["content"] for m in tool_messages(second)]
        assert contents == [
            "Tool arguments are not valid",
            "browse is not a valid tool",
            "Tool arguments are not valid",
        ]

    def test_a_reply_without_a_call_is_asked_to_continue(self) -> None:
        provider = ScriptedProvider([reply(text="Thinking out loud."), reply(finish([]))])
        run(provider)

        assert provider.requests[1].messages[-1] == {"role": "user", "content": CONTINUE_MESSAGE}

    def test_reminders_start_near_the_turn_limit_and_the_run_stops_there(self) -> None:
        provider = ScriptedProvider([reply(call("code_exec", {"cmd": "ls"})) for _ in range(3)])
        result, _ = run(provider, max_turns=3)

        assert result.metadata["completion_reason"] == "max_turns"
        assert result.max_turns_reached
        assert provider.requests[1].messages[-1]["content"] == turns_remaining_message(2)
        assert provider.requests[2].messages[-1]["content"] == LAST_TURN_MESSAGE
        # The first turn gets no reminder.
        assert provider.requests[0].messages[-1]["content"] == "Write the report."

    def test_a_full_context_is_summarized_and_the_history_replaced(self) -> None:
        provider = ScriptedProvider(
            [
                reply(call("code_exec", {"cmd": "ls"}), tokens=80),
                # The first summary attempt calls a tool, so it is asked again.
                reply(call("code_exec", {"cmd": "ls"})),
                LMOutput(text="Listed the files."),
                reply(finish([])),
            ],
            max_length=100,
        )
        result, manager = run(provider)

        assert provider.requests[1].messages[-1]["content"] == SUMMARIZER_PROMPT
        assert provider.requests[2].messages[-1]["content"].endswith(SUMMARIZER_TEXT_ONLY)
        after = provider.requests[3].messages
        assert [m["role"] for m in after] == ["system", "user", "user", "user"]
        assert "Listed the files." in after[2]["content"]
        assert after[3]["content"] == SUMMARY_ACKNOWLEDGEMENT
        assert result.metadata["summaries"] == 1
        assert manager.executor.commands == ["ls"]

    def test_a_summary_that_does_not_fit_ends_the_run_out_of_context(self) -> None:
        overflow = LMOutput(
            text="", metadata={REQUEST_ERROR_KEY: "too long", CONTEXT_OVERFLOW_KEY: True}
        )
        provider = ScriptedProvider(
            [reply(call("code_exec", {"cmd": "ls"}), tokens=80), overflow], max_length=100
        )
        result, _ = run(provider)

        assert result.metadata["completion_reason"] == "context_overflow"
        assert result.error is None

    def test_context_window_kwarg_overrides_the_provider(self) -> None:
        provider = ScriptedProvider(
            [reply(call("code_exec", {"cmd": "ls"}), tokens=80), reply(finish([]))],
            max_length=100,
        )
        result, _ = run(provider, scaffold_kwargs={"context_window_tokens": 1000})

        assert result.metadata["summaries"] == 0

    def test_a_command_past_its_timeout_is_killed_and_reported(self) -> None:
        provider = ScriptedProvider(
            [reply(call("code_exec", {"cmd": "sleep 9"})), reply(finish([]))]
        )
        _, manager = run(
            provider,
            FakeSandboxManager(results=[HANGS]),
            scaffold_kwargs={"command_timeout": 0.3},
        )

        content = tool_messages(provider.requests[1])[0]["content"]
        assert "<error_kind>timeout</error_kind>" in content
        assert "exceeded the 0.3s time limit" in content
        assert len(manager.executor.killed) == 1

    def test_command_output_keeps_streams_apart_and_truncates_long_ones(self) -> None:
        long_output = "a" * MAX_OUTPUT_CHARS + "b" * 10
        provider = ScriptedProvider([reply(call("code_exec", {"cmd": "make"})), reply(finish([]))])
        run(provider, FakeSandboxManager(results=[CommandResult(2, long_output, "warning")]))

        content = tool_messages(provider.requests[1])[0]["content"]
        assert "<exit_code>2</exit_code>" in content
        assert f"original {len(long_output)} characters" in content
        assert "b" * 10 + "</stdout>" in content
        assert "<stderr>warning</stderr>" in content

    def test_an_overflowing_request_drops_the_latest_turn_and_retries(self) -> None:
        overflow = LMOutput(
            text="", metadata={REQUEST_ERROR_KEY: "too long", CONTEXT_OVERFLOW_KEY: True}
        )
        provider = ScriptedProvider(
            [
                reply(call("code_exec", {"cmd": "first"}, "e1")),
                reply(call("code_exec", {"cmd": "second"}, "e2")),
                overflow,
                reply(finish([])),
            ]
        )
        result, _ = run(provider)

        retried = provider.requests[3].messages
        assert retried == provider.requests[1].messages
        assert result.metadata["completion_reason"] == "finished"
        assert result.metadata["dropped_turns"] == 1

    def test_a_sandbox_failure_ends_the_run_with_its_turns_recorded(self) -> None:
        manager = FakeSandboxManager()
        manager.executor.fail_with = RuntimeError("container gone")
        provider = ScriptedProvider([reply(call("code_exec", {"cmd": "ls"}))])
        result, _ = run(provider, manager)

        assert result.metadata["completion_reason"] == "error"
        assert result.error == "RuntimeError: container gone"
        assert [t.role for t in result.trajectory.turns] == ["user", "assistant"]
        assert manager.binding.released

    def test_context_overflow_and_server_errors_end_the_run(self) -> None:
        overflow = LMOutput(
            text="", metadata={REQUEST_ERROR_KEY: "too long", CONTEXT_OVERFLOW_KEY: True}
        )
        result, _ = run(ScriptedProvider([overflow]))
        assert result.metadata["completion_reason"] == "context_overflow"
        assert result.error is None

        failure = LMOutput(text="", metadata={REQUEST_ERROR_KEY: "server down"})
        result, _ = run(ScriptedProvider([failure]))
        assert result.metadata["completion_reason"] == "error"
        assert result.error == "server down"

    def test_trajectory_records_the_conversation(self) -> None:
        provider = ScriptedProvider([reply(call("code_exec", {"cmd": "ls"})), reply(finish([]))])
        result, _ = run(provider)

        roles = [turn.role for turn in result.trajectory.turns]
        assert roles == ["user", "assistant", "tool", "assistant", "tool"]
        assert json.loads(result.trajectory.turns[1].tool_calls[0].function.arguments) == {
            "cmd": "ls"
        }

    def test_tools_beyond_the_scaffolds_own_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="cannot run tools"):
            run(ScriptedProvider([]), tools=get_tools((*TOOLS, "bash")))
        with pytest.raises(ValueError, match="needs a 'finish' tool"):
            run(ScriptedProvider([]), tools=get_tools(("code_exec",)))
