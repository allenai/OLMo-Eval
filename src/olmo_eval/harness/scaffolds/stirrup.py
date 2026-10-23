"""Agent scaffold that follows Stirrup, Artificial Analysis's agent harness.

Artificial Analysis runs its agentic knowledge-work benchmarks (AA-Briefcase,
GDPval-AA) on Stirrup. The loop gives the model a ``code_exec`` tool that runs
each shell command on its own, so no shell state carries between calls, and two
finish tools: ``finish``, which ends the run once every deliverable path it names
is a real file, and ``abandon_task_finish``, which gives up with a reason. The
model may call several tools in one turn; ordinary calls run first and a turn
that calls both finish tools has both refused. A reply with no tool call is
answered with a request to continue. Near the turn limit the model is told how
many turns are left, and when a reply leaves the conversation past a fraction of
the context window, the model summarizes its progress and the history is
replaced by that summary. A request that does not fit the context window drops
the latest completed turn and is sent again.

Prompts, tool descriptions, limits and the result format are vendored from
Stirrup (MIT) at commit 247f24d56b2108235880ed2a2baea5d35b5a67ee. Stirrup's optional
``view_image`` tool is not offered.
"""

from __future__ import annotations

import json
import logging
import shlex
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from olmo_eval.common.types import LMOutput, LMRequest, RequestType, SamplingParams
from olmo_eval.common.types.tools import Function, ToolCall, ToolResult
from olmo_eval.common.types.trajectory import AgentTrajectory, AgentTurn
from olmo_eval.harness.config import HarnessConfig
from olmo_eval.harness.result import HarnessResult
from olmo_eval.harness.sandbox import Capability
from olmo_eval.harness.scaffolds import Scaffold, register_scaffold
from olmo_eval.inference.base import InferenceProvider, require_tool_support
from olmo_eval.inference.errors import is_context_overflow_output, request_error

if TYPE_CHECKING:
    from olmo_eval.common.types import ToolSchema
    from olmo_eval.harness.sandbox import SandboxExecutor

logger = logging.getLogger(__name__)

CODE_EXEC_TOOL_NAME = "code_exec"
FINISH_TOOL_NAME = "finish"
ABANDON_TOOL_NAME = "abandon_task_finish"
#: Tools whose successful call ends the run.
FINISH_TOOL_NAMES = (FINISH_TOOL_NAME, ABANDON_TOOL_NAME)

#: Stirrup's ``AGENT_MAX_TURNS``.
DEFAULT_MAX_TURNS = 30
#: Stirrup's ``SHELL_TIMEOUT``, in seconds.
DEFAULT_COMMAND_TIMEOUT = 300.0
#: Fraction of the context window a reply may fill before the history is summarized.
DEFAULT_SUMMARIZATION_CUTOFF = 0.7
#: Turns left at which the model starts being told how many remain.
TURNS_REMAINING_WARNING_THRESHOLD = 20
#: Characters of a command's stdout, and of its stderr, the model sees (Stirrup's
#: ``MAX_LENGTH_SHELL_STDOUT`` and ``MAX_LENGTH_SHELL_STDERR``).
MAX_OUTPUT_CHARS = 20_000

#: Timeout for the short requests that read a command's output and clean up after it.
CONTROL_TIMEOUT = 60.0

BASE_SYSTEM_PROMPT = (
    "You are an AI agent that will be given a specific task. You are to complete that task "
    "using the tools provided in {max_turns} steps. You will need to call a finish tool as "
    "your last step, where you will pass your finish reason and any required final fields "
    "for that tool.\n"
)
NO_USER_INTERACTION = " You are not able to interact with the user during the task."
USER_INSTRUCTIONS = "\n\nFollow these instructions from the User:\n{instructions}"

CONTINUE_MESSAGE = "Please continue the task"
LAST_TURN_MESSAGE = "This is the last turn. Please finish the task by calling a finish tool."
TURNS_REMAINING_MESSAGE = (
    "You have {turns} turns remaining to complete the task. Please continue. Remember you "
    "will need a separate turn to call a finish tool."
)

SUMMARIZER_PROMPT = """\
The context window is approaching its limit. Please create a concise summary of the \
conversation so far to preserve important information.

Your summary should include:

1. **Task Overview**: What is the main goal or objective?

2. **Progress Made**: What has been accomplished so far?
   - Key files created/modified (with paths)
   - Important functions/classes implemented
   - Tools used and their outcomes

3. **Current State**: Where are we now?
   - What is currently working?
   - What has been tested/verified?

4. **Next Steps**: What still needs to be done?
   - Outstanding TODOs (with specific file paths and line numbers if applicable)
   - Known issues or bugs to address
   - Features or functionality not yet implemented

5. **Important Context**: Any critical details that shouldn't be lost
   - Special configurations or setup requirements
   - Important variable names, API endpoints, or data structures
   - Edge cases or constraints to keep in mind
   - Dependencies or relationships between components

Keep the summary concise but comprehensive. Do not use any tools. Focus on actionable \
information that will allow smooth continuation of the work.
"""
SUMMARIZER_TEXT_ONLY = (
    "IMPORTANT: Respond with the summary as plain prose text only. Do NOT call any tools — "
    "a tool call cannot serve as a summary and will cause the summarization to fail."
)
SUMMARY_BRIDGE = """\
**Context Continuation**

Due to context window limitations, the previous conversation has been summarized. Below is \
a summary of what happened before:

---

{summary}

---

You should continue working on this task from where it was left off. All the progress, \
current state, and next steps are described in the summary above. Proceed with completing \
any outstanding work."""
SUMMARY_ACKNOWLEDGEMENT = "Got it, thanks!"

INVALID_ARGUMENTS_MESSAGE = "Tool arguments are not valid"

#: Counts kept in the result's metadata to show how a run spent its turns.
DIAGNOSTIC_COUNTS = (
    "no_tool_call_turns",
    "invalid_argument_calls",
    "unknown_tool_calls",
    "finish_refusals",
    "length_truncated_replies",
)

DEFAULT_SAMPLING_PARAMS = SamplingParams(
    temperature=0.7, top_p=0.95, max_tokens=32768, fit_max_tokens_to_context=True
)


class SummaryError(RuntimeError):
    """The model could not summarize the conversation."""


class SummaryOverflowError(SummaryError):
    """The summary request itself did not fit the context window."""


class SummaryEmptyError(SummaryError):
    """The model answered every summary request without any text."""


def build_system_prompt(max_turns: int, instructions: str | None) -> str:
    """Compose the system prompt the way Stirrup does: base prompt, then the task's own."""
    prompt = BASE_SYSTEM_PROMPT.format(max_turns=max_turns) + NO_USER_INTERACTION
    if instructions:
        prompt += USER_INSTRUCTIONS.format(instructions=instructions)
    return prompt


def turns_remaining_message(turns: int) -> str:
    """The reminder sent before a turn once few turns are left."""
    if turns == 1:
        return LAST_TURN_MESSAGE
    return TURNS_REMAINING_MESSAGE.format(turns=turns)


def truncate_output(text: str, max_chars: int = MAX_OUTPUT_CHARS) -> str:
    """Keep the start and end of a long output, noting how much was cut."""
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return _truncated(text[:half], text[-half:], f"{len(text)} characters", max_chars)


def _truncated(head: str, tail: str, original: str, max_chars: int) -> str:
    return (
        head + f"\n... This content has been truncated from an original {original} "
        f"to stay below {max_chars} characters ...\n" + tail
    )


@dataclass(frozen=True)
class CommandResult:
    """A command's exit code and output, each stream already truncated for the model.

    ``exit_code`` is -1 when the command timed out.
    """

    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False


def format_shell_result(result: CommandResult) -> str:
    """Render a command's result as the XML block the model reads."""
    if result.timed_out:
        return (
            "<shell_results>"
            "\n<error_kind>timeout</error_kind>"
            f"\n<details>{result.stderr}</details>"
            "\n</shell_results>"
        )
    return (
        "<shell_results>"
        f"\n<exit_code>{result.exit_code}</exit_code>"
        f"\n<stdout>{result.stdout}</stdout>"
        f"\n<stderr>{result.stderr}</stderr>"
        "\n</shell_results>"
    )


async def run_command(executor: SandboxExecutor, command: str, timeout: float) -> CommandResult:
    """Run a shell command in the sandbox and wait for it, up to ``timeout`` seconds.

    Commands run detached and are polled (see ``SandboxExecutor.run_detached``),
    because a request to the sandbox server is cut off after a few minutes however
    long a command is allowed. Each stream is truncated the way Stirrup does it.
    """
    run = await executor.run_detached(command, timeout)
    stdout = await _read_output(executor, run.stdout_path)
    stderr = await _read_output(executor, run.stderr_path)
    await executor.execute_command(f"rm -rf {run.run_dir}", timeout=CONTROL_TIMEOUT)
    if run.timed_out:
        note = f"Command timed out after {timeout:g} seconds"
        return CommandResult(-1, stdout, f"{stderr}\n{note}" if stderr else note, timed_out=True)
    if run.exit_code is None:
        note = "The command was killed before it reported an exit status."
        return CommandResult(-1, stdout, f"{stderr}\n{note}" if stderr else note)
    return CommandResult(run.exit_code, stdout, stderr)


async def _read_output(executor: SandboxExecutor, path: str) -> str:
    """Read an output file truncated for the model, fetching only the ends of a huge one.

    A file of up to four times the character budget is read whole, so it is cut by
    characters as Stirrup cuts it. Of a larger one, generous ends are fetched and
    cut by characters, and the note gives its size in bytes.
    """
    whole = 4 * MAX_OUTPUT_CHARS
    ends = 2 * MAX_OUTPUT_CHARS
    result = await executor.execute_command(
        f"[ -f {path} ] || exit 0; size=$(stat -c %s {path}); "
        f"if [ $size -le {whole} ]; then cat {path}; "
        f"else echo $size; head -c {ends} {path}; echo; echo {_TAIL_MARKER}; "
        f"tail -c {ends} {path}; fi",
        timeout=CONTROL_TIMEOUT,
    )
    text = result.output
    if f"\n{_TAIL_MARKER}\n" not in text:
        return truncate_output(text)
    size, _, rest = text.partition("\n")
    head, _, tail = rest.partition(f"\n{_TAIL_MARKER}\n")
    half = MAX_OUTPUT_CHARS // 2
    return _truncated(head[:half], tail[-half:], f"{size} bytes", MAX_OUTPUT_CHARS)


_TAIL_MARKER = "---STIRRUP-OUTPUT-TAIL---"


@dataclass(frozen=True)
class FinishCall:
    """What the model passed to the finish tool that ended the run."""

    tool: str
    reason: str
    paths: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"tool": self.tool, "reason": self.reason, "paths": list(self.paths)}


@dataclass
class _Conversation:
    """The messages sent to the model, what each one is, and the run's trajectory.

    ``kinds`` names each message: ``system``, ``task``, ``summary``, ``reminder``
    (a turn reminder), ``user``, ``assistant`` or ``tool``.
    """

    messages: list[dict[str, Any]] = field(default_factory=list)
    kinds: list[str] = field(default_factory=list)
    turns: list[AgentTurn] = field(default_factory=list)

    def add(self, message: dict[str, Any], kind: str) -> None:
        self.messages.append(message)
        self.kinds.append(kind)

    def add_user(self, content: str, kind: str = "user") -> None:
        self.add({"role": "user", "content": content}, kind)
        self.turns.append(AgentTurn.user(content=content))

    def drop_latest_turn(self) -> bool:
        """Drop the latest completed turn from the messages, as Stirrup does on overflow.

        A turn runs from its assistant message, or the turn reminder just before
        it, to the end of the conversation. The only turn since the task or the
        last summary is kept, so the model never loses all its progress.

        Returns:
            Whether a turn was dropped.
        """
        for index in range(len(self.kinds) - 1, -1, -1):
            kind = self.kinds[index]
            if kind in ("task", "summary"):
                return False
            if kind != "assistant":
                continue
            start = index - 1 if index > 0 and self.kinds[index - 1] == "reminder" else index
            earlier = self.kinds[:start]
            last_boundary = max(
                (i for i, k in enumerate(earlier) if k in ("task", "summary")), default=0
            )
            if "assistant" not in earlier[last_boundary:]:
                return False
            del self.messages[start:]
            del self.kinds[start:]
            return True
        return False


def _parse_arguments(call: ToolCall) -> dict[str, Any] | None:
    """A call's arguments as a dict, or None when they are not a JSON object."""
    raw = call.function.arguments
    if isinstance(raw, dict):
        return raw
    try:
        arguments = json.loads(raw) if raw else {}
    except (TypeError, ValueError):
        return None
    return arguments if isinstance(arguments, dict) else None


def _validated_finish(call: ToolCall, arguments: dict[str, Any] | None) -> FinishCall | None:
    """The finish call's parameters, or None when they do not match the tool's schema."""
    if arguments is None:
        return None
    reason = arguments.get("reason")
    if not isinstance(reason, str):
        return None
    if call.function.name == ABANDON_TOOL_NAME:
        return FinishCall(tool=ABANDON_TOOL_NAME, reason=reason)
    paths = arguments.get("paths")
    # A path with a NUL character cannot name a file, and the sandbox rejects it.
    if not isinstance(paths, list) or not all(isinstance(p, str) and "\0" not in p for p in paths):
        return None
    return FinishCall(tool=FINISH_TOOL_NAME, reason=reason, paths=tuple(paths))


def _replayable(call: ToolCall, arguments: dict[str, Any] | None) -> ToolCall:
    """The call as it can be sent back to a server that re-renders the history.

    An OpenAI-compatible server rejects a history whose call arguments are not a
    JSON object, so such a call is kept with empty arguments; its result already
    tells the model the arguments were invalid. A call with no arguments at all
    is sent back with an empty object.
    """
    if arguments is not None and call.function.arguments:
        return call
    return ToolCall(id=call.id, function=Function(name=call.function.name, arguments="{}"))


@register_scaffold("stirrup")
class StirrupScaffold(Scaffold):
    """Drive a model through a task with Stirrup's tools, turn reminders and summaries.

    Scaffold kwargs (``HarnessConfig.scaffold_kwargs`` or ``run(**kwargs)``):

    - ``command_timeout``: seconds each command may run (default 300).
    - ``agent_timeout``: wall-clock budget for the whole run in seconds. When set,
      no new turn starts past it and the last command is cut to fit.
    - ``context_window_tokens``: context size used to decide when to summarize.
      Defaults to the provider's ``max_length``; with neither, the history is
      never summarized.
    - ``summarization_cutoff``: fraction of the context window that triggers a
      summary (default 0.7).

    ``HarnessConfig.max_turns`` bounds the number of model replies (default 30)
    and ``HarnessConfig.system_prompt`` holds the task's instructions, which follow
    Stirrup's base prompt. The config's tools must include ``code_exec`` and
    ``finish``; ``abandon_task_finish`` is optional.

    The result's metadata carries ``completion_reason`` (``finished``,
    ``abandoned``, ``max_turns``, ``timeout``, ``context_overflow`` or ``error``)
    and, when the run ended through a finish tool, ``finish`` with its arguments.
    """

    name = "stirrup"

    def __init__(self) -> None:
        self._conversation: _Conversation | None = None

    async def run(
        self,
        provider: InferenceProvider,
        config: HarnessConfig,
        request: LMRequest,
        sampling_params: SamplingParams | None = None,
        trace_metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> HarnessResult:
        require_tool_support(provider)
        options = {**config.scaffold_kwargs, **kwargs}
        max_turns = config.max_turns or DEFAULT_MAX_TURNS
        command_timeout = float(options.get("command_timeout", DEFAULT_COMMAND_TIMEOUT))
        agent_timeout = options.get("agent_timeout")
        cutoff = float(options.get("summarization_cutoff", DEFAULT_SUMMARIZATION_CUTOFF))
        context_window = options.get("context_window_tokens") or getattr(
            provider, "max_length", None
        )
        task_id = (trace_metadata or {}).get("task_id", config.name)
        if not context_window:
            logger.warning(
                "Task %s: the context window is unknown, so the history is never summarized",
                task_id,
            )

        tool_names = {tool.name for tool in config.resolved_tools}
        for required in (CODE_EXEC_TOOL_NAME, FINISH_TOOL_NAME):
            if required not in tool_names:
                raise ValueError(f"The {self.name} scaffold needs a {required!r} tool.")
        unsupported = tool_names - {CODE_EXEC_TOOL_NAME, *FINISH_TOOL_NAMES}
        if unsupported:
            raise ValueError(
                f"The {self.name} scaffold cannot run tools {sorted(unsupported)}; it runs "
                f"{CODE_EXEC_TOOL_NAME}, {FINISH_TOOL_NAME} and {ABANDON_TOOL_NAME}."
            )
        tool_schemas = tuple(tool.schema for tool in config.resolved_tools)

        if self._sandbox_manager is None:
            if not config.sandboxes:
                raise ValueError(
                    f"The {self.name} scaffold needs a sandbox manager or sandbox configs."
                )
            from olmo_eval.harness.sandbox import SandboxManager

            self._sandbox_manager = SandboxManager(config.sandboxes, owner=config.name)
            await self._sandbox_manager.start()

        instruction = ""
        for message in reversed(request.messages):
            if message.get("role") == "user":
                instruction = str(message.get("content", ""))
                break
        system_message = {
            "role": "system",
            "content": build_system_prompt(max_turns, config.system_prompt),
        }
        conversation = _Conversation()
        self._conversation = conversation
        conversation.add(system_message, "system")
        conversation.add_user(instruction, "task")

        params = sampling_params or DEFAULT_SAMPLING_PARAMS
        usage = {"prompt_tokens": 0, "completion_tokens": 0}
        counts = dict.fromkeys(DIAGNOSTIC_COUNTS, 0)
        started = time.monotonic()
        deadline = started + float(agent_timeout) if agent_timeout is not None else None
        completion_reason = "max_turns"
        finish: FinishCall | None = None
        error: str | None = None
        summaries = 0
        dropped_turns = 0
        last_text = ""
        # Turns the conversation holds or has summarized; a dropped turn does not count.
        turns_taken = 0

        # Every command of the run goes to one container, so files persist between them.
        binding = await self._sandbox_manager.acquire_binding(Capability.BASH)
        try:
            while turns_taken < max_turns:
                if deadline is not None and time.monotonic() >= deadline:
                    completion_reason = "timeout"
                    break
                remaining = max_turns - turns_taken
                if turns_taken > 0 and remaining <= TURNS_REMAINING_WARNING_THRESHOLD:
                    conversation.add_user(turns_remaining_message(remaining), "reminder")

                replies = await self._generate(
                    provider, conversation.messages, tool_schemas, params
                )
                # A request that does not fit drops the latest turn and is sent again,
                # as Stirrup retries a step. The dropped turns are bounded, so a run
                # whose every request overflows still ends.
                while (
                    is_context_overflow_output(replies)
                    and dropped_turns < max_turns
                    and conversation.drop_latest_turn()
                ):
                    logger.info("Task %s overflowed the context; dropped a turn", task_id)
                    turns_taken -= 1
                    dropped_turns += 1
                    replies = await self._generate(
                        provider, conversation.messages, tool_schemas, params
                    )
                if is_context_overflow_output(replies):
                    logger.warning("Task %s outgrew the context window", task_id)
                    completion_reason = "context_overflow"
                    break
                failure = request_error(replies)
                if failure is not None or not replies:
                    completion_reason = "error"
                    error = failure or "The provider returned no output."
                    break

                output = replies[0]
                last_text = output.text or ""
                turns_taken += 1
                self._accumulate_usage(output, usage)
                if (output.metadata or {}).get("finish_reason") == "length":
                    counts["length_truncated_replies"] += 1
                if not output.tool_calls:
                    counts["no_tool_call_turns"] += 1
                finish = await self._take_turn(
                    conversation, output, binding.executor, command_timeout, deadline, counts
                )
                if finish is not None:
                    completion_reason = (
                        "finished" if finish.tool == FINISH_TOOL_NAME else "abandoned"
                    )
                    break

                used = self._context_used(output)
                if context_window and used / context_window >= cutoff and turns_taken < max_turns:
                    logger.info(
                        "Task %s used %.0f%% of the context window; summarizing",
                        task_id,
                        100 * used / context_window,
                    )
                    try:
                        summary, dropped = await self._summarize(
                            provider, conversation, tool_schemas, params
                        )
                    except SummaryOverflowError:
                        completion_reason = "context_overflow"
                        break
                    except SummaryEmptyError:
                        # The model's failure to summarize ends its run without a
                        # submission, as running out of context would.
                        completion_reason = "summary_failed"
                        break
                    except SummaryError as e:
                        completion_reason = "error"
                        error = str(e)
                        break
                    summaries += 1
                    turns_taken -= dropped
                    dropped_turns += dropped
                    # The system prompt and the task stay through every summary.
                    del conversation.messages[2:]
                    del conversation.kinds[2:]
                    conversation.add_user(SUMMARY_BRIDGE.format(summary=summary), "summary")
                    conversation.add_user(SUMMARY_ACKNOWLEDGEMENT)

                if not output.tool_calls and max_turns - turns_taken > (
                    TURNS_REMAINING_WARNING_THRESHOLD
                ):
                    conversation.add_user(CONTINUE_MESSAGE)
        except Exception as e:
            # A sandbox or transport failure ends the run as an error, not a score, and
            # its turns so far still reach the trajectory.
            logger.exception("Task %s failed", task_id)
            completion_reason = "error"
            error = f"{type(e).__name__}: {e}"
        finally:
            await binding.release()

        steps = sum(1 for turn in conversation.turns if turn.role == "assistant")
        metadata: dict[str, Any] = {
            "completion_reason": completion_reason,
            "steps": steps,
            "summaries": summaries,
            "dropped_turns": dropped_turns,
            **counts,
            "usage": dict(usage),
            "duration_s": round(time.monotonic() - started, 1),
        }
        if finish is not None:
            metadata["finish"] = finish.to_dict()
        logger.info(
            "Task %s finished after %s turns: %s (%s)",
            task_id,
            steps,
            completion_reason,
            ", ".join(f"{k}={v}" for k, v in counts.items() if v),
        )
        return HarnessResult(
            final_output=LMOutput(
                text=finish.reason if finish is not None else last_text,
                metadata={"completion_reason": completion_reason},
            ),
            trajectory=AgentTrajectory(turns=tuple(conversation.turns), metadata=dict(metadata)),
            max_turns_reached=completion_reason == "max_turns",
            error=error,
            metadata=metadata,
        )

    def trajectory_so_far(self) -> AgentTrajectory:
        """The turns of the run in progress, for a caller that has to cut it off."""
        if self._conversation is None:
            return AgentTrajectory(turns=())
        return AgentTrajectory(turns=tuple(self._conversation.turns))

    async def _generate(
        self,
        provider: InferenceProvider,
        messages: list[dict[str, Any]],
        tools: tuple[ToolSchema, ...],
        params: SamplingParams,
    ) -> list[LMOutput]:
        outputs = await provider.agenerate(
            [LMRequest(request_type=RequestType.CHAT, messages=tuple(messages), tools=tools)],
            params,
        )
        return list(outputs[0]) if outputs else []

    async def _take_turn(
        self,
        conversation: _Conversation,
        output: LMOutput,
        executor: SandboxExecutor,
        command_timeout: float,
        deadline: float | None,
        counts: dict[str, int],
    ) -> FinishCall | None:
        """Record a reply, run its tool calls, and return the finish call that ended the run.

        ``counts`` is updated with the calls that were invalid, unknown or refused.
        """
        text = output.text or ""
        calls = list(output.tool_calls or ())
        parsed = [(call, _parse_arguments(call)) for call in calls]
        conversation.turns.append(AgentTurn.assistant(content=text, tool_calls=calls))
        message: dict[str, Any] = {"role": "assistant", "content": text}
        if calls:
            message["tool_calls"] = [_replayable(c, a).to_openai() for c, a in parsed]
        conversation.add(message, "assistant")
        if not calls:
            return None

        finish_calls = [c for c in calls if c.function.name in FINISH_TOOL_NAMES]
        reject_finish = len(finish_calls) > 1
        finish: FinishCall | None = None
        results: list[ToolResult] = []
        # Ordinary calls run first, so nothing happens after a successful finish.
        ordered = sorted(parsed, key=lambda p: p[0].function.name in FINISH_TOOL_NAMES)
        for call, arguments in ordered:
            name = call.function.name
            if reject_finish and name in FINISH_TOOL_NAMES:
                content = (
                    f"Cannot call finish tool '{name}': multiple finish tools "
                    f"({sorted({c.function.name for c in finish_calls})}) were called in the "
                    "same turn. Only one finish tool may be called per turn — retry with a "
                    "single finish tool call."
                )
                results.append(ToolResult(tool_call_id=call.id, content=content, is_error=True))
                counts["finish_refusals"] += 1
                continue
            if name == CODE_EXEC_TOOL_NAME:
                result = await self._code_exec(call, arguments, executor, command_timeout, deadline)
            elif name in FINISH_TOOL_NAMES:
                result, finish = await self._finish(call, arguments, executor)
            else:
                result = ToolResult(
                    tool_call_id=call.id, content=f"{name} is not a valid tool", is_error=True
                )
                counts["unknown_tool_calls"] += 1
            if result.content == INVALID_ARGUMENTS_MESSAGE:
                counts["invalid_argument_calls"] += 1
            elif name in FINISH_TOOL_NAMES and finish is None:
                counts["finish_refusals"] += 1
            results.append(result)

        for result in results:
            conversation.add(result.to_openai(), "tool")
        conversation.turns.append(AgentTurn.tool(results))
        return finish

    @staticmethod
    async def _code_exec(
        call: ToolCall,
        arguments: dict[str, Any] | None,
        executor: SandboxExecutor,
        command_timeout: float,
        deadline: float | None,
    ) -> ToolResult:
        command = (arguments or {}).get("cmd")
        if not isinstance(command, str):
            return ToolResult(
                tool_call_id=call.id, content=INVALID_ARGUMENTS_MESSAGE, is_error=True
            )
        timeout = command_timeout
        if deadline is not None:
            timeout = max(1.0, min(command_timeout, deadline - time.monotonic()))
        result = await run_command(executor, command, timeout)
        return ToolResult(
            tool_call_id=call.id,
            content=format_shell_result(result),
            is_error=result.timed_out or result.exit_code != 0,
        )

    @staticmethod
    async def _finish(
        call: ToolCall, arguments: dict[str, Any] | None, executor: SandboxExecutor
    ) -> tuple[ToolResult, FinishCall | None]:
        """Run a finish tool, refusing a ``finish`` that names files that do not exist."""
        finish = _validated_finish(call, arguments)
        if finish is None:
            result = ToolResult(
                tool_call_id=call.id, content=INVALID_ARGUMENTS_MESSAGE, is_error=True
            )
            return result, None
        missing = []
        for path in finish.paths:
            check = await executor.execute_command(
                f"test -f {shlex.quote(path)}", timeout=CONTROL_TIMEOUT
            )
            if check.exit_code != 0:
                missing.append(path)
        if missing:
            content = (
                f"ERROR: Files do not exist: {missing}. Verify paths and ensure files were saved."
            )
            return ToolResult(tool_call_id=call.id, content=content, is_error=True), None
        return ToolResult(tool_call_id=call.id, content=finish.reason), finish

    async def _summarize(
        self,
        provider: InferenceProvider,
        conversation: _Conversation,
        tools: tuple[ToolSchema, ...],
        params: SamplingParams,
    ) -> tuple[str, int]:
        """Ask the model for a summary of its progress, escalating if it calls a tool.

        Some models answer the summary request with a tool call, so the request is
        repeated with an explicit text-only warning and then with the tools withheld.
        A request that does not fit the context window drops the latest turn and
        starts over.

        Returns:
            The summary, and how many turns were dropped to fit it.

        Raises:
            SummaryOverflowError: If the request does not fit the context window.
            SummaryEmptyError: If no attempt gives any text.
            SummaryError: If a request fails.
        """
        text_only = f"{SUMMARIZER_PROMPT}\n\n{SUMMARIZER_TEXT_ONLY}"
        tool_docs = "\n".join(f"- {tool.name}: {tool.description}" for tool in tools)
        no_tools = (
            f"{text_only}\n\nTools are disabled for this response. For reference, the tools "
            f"available earlier in the conversation were:\n{tool_docs}"
        )
        attempts = ((SUMMARIZER_PROMPT, tools), (text_only, tools), (no_tools, ()))
        dropped = 0
        attempt = 0
        while attempt < len(attempts):
            prompt, attempt_tools = attempts[attempt]
            messages = [*conversation.messages, {"role": "user", "content": prompt}]
            replies = await self._generate(provider, messages, attempt_tools, params)
            if is_context_overflow_output(replies):
                if not conversation.drop_latest_turn():
                    raise SummaryOverflowError("The summary request outgrew the context window.")
                dropped += 1
                attempt = 0
                continue
            failure = request_error(replies)
            if failure is not None or not replies:
                raise SummaryError(failure or "The provider returned no summary.")
            if (replies[0].metadata or {}).get("finish_reason") == "length":
                # The server left the summary too little room to finish, which is the
                # overflow Stirrup's fixed reply budget would have raised.
                if not conversation.drop_latest_turn():
                    raise SummaryOverflowError("The summary did not fit the context window.")
                dropped += 1
                attempt = 0
                continue
            summary = (replies[0].text or "").strip()
            if summary:
                return summary, dropped
            logger.warning("The summary reply had no text; asking again")
            attempt += 1
        raise SummaryEmptyError("The summarizer produced no text.")

    @staticmethod
    def _context_used(output: LMOutput) -> int:
        metadata = output.metadata or {}
        used = 0
        for key in ("prompt_tokens", "completion_tokens"):
            value = metadata.get(key)
            if isinstance(value, int | float):
                used += int(value)
        return used
