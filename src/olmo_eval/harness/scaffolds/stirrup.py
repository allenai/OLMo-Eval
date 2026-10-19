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
replaced by that summary.

Prompts, tool descriptions, limits and the result format are vendored from
Stirrup (MIT) at commit 247f24d56b2108235880ed2a2baea5d35b5a67ee. Two parts differ:
the sandbox returns a command's stdout and stderr as one stream, so a result
carries a single ``<output>`` element, and there is no ``view_image`` tool, which
Stirrup only offers to models that accept images.
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
#: Characters of a command's output the model sees (Stirrup's ``MAX_LENGTH_SHELL_STDOUT``).
MAX_OUTPUT_CHARS = 20_000

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

DEFAULT_SAMPLING_PARAMS = SamplingParams(
    temperature=0.7, top_p=0.95, max_tokens=32768, fit_max_tokens_to_context=True
)


class SummaryError(RuntimeError):
    """The model could not summarize the conversation."""


class SummaryOverflowError(SummaryError):
    """The summary request itself did not fit the context window."""


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
    return (
        text[: max_chars // 2]
        + f"\n... This content has been truncated from an original {len(text)} characters "
        f"to stay below {max_chars} characters ...\n" + text[-max_chars // 2 :]
    )


def format_shell_result(output: str, exit_code: int, timed_out: bool = False) -> str:
    """Render a command's result as the XML block the model reads."""
    if timed_out:
        return (
            "<shell_results>"
            "\n<error_kind>timeout</error_kind>"
            f"\n<details>{truncate_output(output)}</details>"
            "\n</shell_results>"
        )
    return (
        "<shell_results>"
        f"\n<exit_code>{exit_code}</exit_code>"
        f"\n<output>{truncate_output(output)}</output>"
        "\n</shell_results>"
    )


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
    """The messages sent to the model and the trajectory recorded for the run."""

    messages: list[dict[str, Any]]
    turns: list[AgentTurn] = field(default_factory=list)

    def add_user(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})
        self.turns.append(AgentTurn.user(content=content))


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
    if not isinstance(paths, list) or not all(isinstance(p, str) for p in paths):
        return None
    return FinishCall(tool=FINISH_TOOL_NAME, reason=reason, paths=tuple(paths))


def _replayable(call: ToolCall, arguments: dict[str, Any] | None) -> ToolCall:
    """The call as it can be sent back to a server that re-renders the history.

    An OpenAI-compatible server rejects a history whose call arguments are not a
    JSON object, so such a call is kept with empty arguments; its result already
    tells the model the arguments were invalid.
    """
    if arguments is not None:
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
        conversation = _Conversation(messages=[system_message])
        conversation.add_user(instruction)

        params = sampling_params or DEFAULT_SAMPLING_PARAMS
        usage = {"prompt_tokens": 0, "completion_tokens": 0}
        started = time.monotonic()
        deadline = started + float(agent_timeout) if agent_timeout is not None else None
        completion_reason = "max_turns"
        finish: FinishCall | None = None
        error: str | None = None
        summaries = 0
        last_text = ""

        for turn in range(max_turns):
            if deadline is not None and time.monotonic() >= deadline:
                completion_reason = "timeout"
                break
            remaining = max_turns - turn
            if turn > 0 and remaining <= TURNS_REMAINING_WARNING_THRESHOLD:
                conversation.add_user(turns_remaining_message(remaining))

            replies = await self._generate(provider, conversation.messages, tool_schemas, params)
            if is_context_overflow_output(replies):
                logger.warning("Task %s outgrew the context window at turn %s", task_id, turn)
                completion_reason = "context_overflow"
                break
            failure = request_error(replies)
            if failure is not None or not replies:
                completion_reason = "error"
                error = failure or "The provider returned no output."
                break

            output = replies[0]
            last_text = output.text or ""
            self._accumulate_usage(output, usage)
            finish = await self._take_turn(conversation, output, command_timeout, deadline)
            if finish is not None:
                completion_reason = "finished" if finish.tool == FINISH_TOOL_NAME else "abandoned"
                break

            used = self._context_used(output)
            if context_window and used / context_window >= cutoff and turn < max_turns - 1:
                logger.info(
                    "Task %s used %.0f%% of the context window; summarizing",
                    task_id,
                    100 * used / context_window,
                )
                try:
                    summary = await self._summarize(provider, conversation, tool_schemas, params)
                except SummaryOverflowError:
                    completion_reason = "context_overflow"
                    break
                except SummaryError as e:
                    completion_reason = "error"
                    error = str(e)
                    break
                summaries += 1
                # The system prompt and the task stay through every summary.
                conversation.messages = [system_message, conversation.messages[1]]
                conversation.add_user(SUMMARY_BRIDGE.format(summary=summary))
                conversation.add_user(SUMMARY_ACKNOWLEDGEMENT)

            if not output.tool_calls and remaining - 1 > TURNS_REMAINING_WARNING_THRESHOLD:
                conversation.add_user(CONTINUE_MESSAGE)

        steps = sum(1 for turn in conversation.turns if turn.role == "assistant")
        metadata: dict[str, Any] = {
            "completion_reason": completion_reason,
            "steps": steps,
            "summaries": summaries,
            "usage": dict(usage),
            "duration_s": round(time.monotonic() - started, 1),
        }
        if finish is not None:
            metadata["finish"] = finish.to_dict()
        logger.info("Task %s finished after %s turns: %s", task_id, steps, completion_reason)
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
        command_timeout: float,
        deadline: float | None,
    ) -> FinishCall | None:
        """Record a reply, run its tool calls, and return the finish call that ended the run."""
        text = output.text or ""
        calls = list(output.tool_calls or ())
        parsed = [(call, _parse_arguments(call)) for call in calls]
        conversation.turns.append(AgentTurn.assistant(content=text, tool_calls=calls))
        message: dict[str, Any] = {"role": "assistant", "content": text}
        if calls:
            message["tool_calls"] = [_replayable(c, a).to_openai() for c, a in parsed]
        conversation.messages.append(message)
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
                continue
            if name == CODE_EXEC_TOOL_NAME:
                result = await self._code_exec(call, arguments, command_timeout, deadline)
            elif name in FINISH_TOOL_NAMES:
                result, finish = await self._finish(call, arguments)
            else:
                result = ToolResult(
                    tool_call_id=call.id, content=f"{name} is not a valid tool", is_error=True
                )
            results.append(result)

        for result in results:
            conversation.messages.append(result.to_openai())
        conversation.turns.append(AgentTurn.tool(results))
        return finish

    async def _code_exec(
        self,
        call: ToolCall,
        arguments: dict[str, Any] | None,
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
        result = await self._sandbox_manager.execute_command(
            command, timeout, capabilities=Capability.BASH
        )
        timed_out = result.error == "timeout"
        return ToolResult(
            tool_call_id=call.id,
            content=format_shell_result(result.output, result.exit_code, timed_out),
            is_error=timed_out or result.exit_code != 0,
        )

    async def _finish(
        self, call: ToolCall, arguments: dict[str, Any] | None
    ) -> tuple[ToolResult, FinishCall | None]:
        """Run a finish tool, refusing a ``finish`` that names files that do not exist."""
        finish = _validated_finish(call, arguments)
        if finish is None:
            result = ToolResult(
                tool_call_id=call.id, content=INVALID_ARGUMENTS_MESSAGE, is_error=True
            )
            return result, None
        missing = [path for path in finish.paths if not await self._is_file(path)]
        if missing:
            content = (
                f"ERROR: Files do not exist: {missing}. Verify paths and ensure files were saved."
            )
            return ToolResult(tool_call_id=call.id, content=content, is_error=True), None
        return ToolResult(tool_call_id=call.id, content=finish.reason), finish

    async def _is_file(self, path: str) -> bool:
        result = await self._sandbox_manager.execute_command(
            f"test -f {shlex.quote(path)}", 30.0, capabilities=Capability.BASH
        )
        return result.exit_code == 0

    async def _summarize(
        self,
        provider: InferenceProvider,
        conversation: _Conversation,
        tools: tuple[ToolSchema, ...],
        params: SamplingParams,
    ) -> str:
        """Ask the model for a summary of its progress, escalating if it calls a tool.

        Some models answer the summary request with a tool call, so the request is
        repeated with an explicit text-only warning and then with the tools withheld.

        Raises:
            SummaryOverflowError: If the request does not fit the context window.
            SummaryError: If a request fails or no attempt gives any text.
        """
        text_only = f"{SUMMARIZER_PROMPT}\n\n{SUMMARIZER_TEXT_ONLY}"
        tool_docs = "\n".join(f"- {tool.name}: {tool.description}" for tool in tools)
        no_tools = (
            f"{text_only}\n\nTools are disabled for this response. For reference, the tools "
            f"available earlier in the conversation were:\n{tool_docs}"
        )
        attempts = ((SUMMARIZER_PROMPT, tools), (text_only, tools), (no_tools, ()))
        for prompt, attempt_tools in attempts:
            messages = [*conversation.messages, {"role": "user", "content": prompt}]
            replies = await self._generate(provider, messages, attempt_tools, params)
            if is_context_overflow_output(replies):
                raise SummaryOverflowError("The summary request outgrew the context window.")
            failure = request_error(replies)
            if failure is not None or not replies:
                raise SummaryError(failure or "The provider returned no summary.")
            summary = (replies[0].text or "").strip()
            if summary:
                return summary
            logger.warning("The summary reply had no text; asking again")
        raise SummaryError("The summarizer produced no text.")

    @staticmethod
    def _context_used(output: LMOutput) -> int:
        metadata = output.metadata or {}
        used = 0
        for key in ("prompt_tokens", "completion_tokens"):
            value = metadata.get(key)
            if isinstance(value, int | float):
                used += int(value)
        return used

    @staticmethod
    def _accumulate_usage(output: LMOutput, usage: dict[str, int]) -> None:
        metadata = output.metadata or {}
        for key in usage:
            value = metadata.get(key)
            if isinstance(value, int | float):
                usage[key] += int(value)
