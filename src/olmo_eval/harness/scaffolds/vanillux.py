"""Single-bash-tool agent scaffold with the mini-swe-agent prompts.

The loop is the one mini-swe-agent popularized and tmax's ``Vanillux2Agent``
ports to native tool calls: the model writes a THOUGHT and calls one ``bash``
tool per step, reads back the truncated output and exit code, and finishes by
echoing a submit marker. A reply without a valid call is answered with a
format-error message rather than ending the run. Shell state persists between
steps because commands run in the sandbox's persistent session.

Prompts are vendored from mini-swe-agent v2.2 (MIT) by way of tmax, with the
code-block directives replaced by instructions to call the ``bash`` tool.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any

from olmo_eval.common.types import LMOutput, LMRequest, RequestType, SamplingParams
from olmo_eval.common.types.tools import ToolCall, ToolResult
from olmo_eval.common.types.trajectory import AgentTrajectory, AgentTurn
from olmo_eval.harness.config import HarnessConfig
from olmo_eval.harness.result import HarnessResult
from olmo_eval.harness.sandbox import Capability
from olmo_eval.harness.scaffolds import Scaffold, register_scaffold
from olmo_eval.inference.base import InferenceProvider, require_tool_support
from olmo_eval.inference.errors import is_context_overflow_output, request_error

logger = logging.getLogger(__name__)

#: Name of the tool the model must call. Any other call is a format error.
BASH_TOOL_NAME = "bash"

#: The model ends the task by echoing this through the bash tool.
SUBMIT_MARKER = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"

DEFAULT_MAX_STEPS = 64
DEFAULT_MAX_FORMAT_ERRORS = 64
DEFAULT_COMMAND_TIMEOUT = 120.0
DEFAULT_SAMPLING_PARAMS = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=16384)

OBSERVATION_MAX_CHARS = 10_000
OBSERVATION_HEAD_CHARS = 5_000
OBSERVATION_TAIL_CHARS = 5_000

SYSTEM_TEMPLATE = """\
You are a helpful assistant that can interact with a computer.

Your response must include a THOUGHT section before your action where you
explain your reasoning. After the THOUGHT, you must call the `bash` tool
with EXACTLY ONE bash command (multiple commands chained with `&&` or `||`
count as a single action).

Failure to follow these rules — calling no tool, calling a tool other than
`bash`, or omitting the THOUGHT — will cause your response to be rejected.
"""

INSTANCE_TEMPLATE = """\
Please solve this task:

{task}

You can execute bash commands and edit files (with `sed`, `cat > file << 'EOF'`,
etc.) to implement the necessary changes.

## Recommended Workflow

This workflow should be done step-by-step so that you can iterate on your
changes and any possible problems.

1. Analyze the codebase / environment by finding and reading relevant files.
2. If applicable, create a script to reproduce the issue or expected behaviour.
3. Implement the change(s) by editing the source code or environment state.
4. Verify your fix works by running your script (or relevant test) again.
5. Test edge cases to ensure your fix is robust.
6. Submit your changes and finish your work by issuing the following command:
   `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`
   Do not combine it with any other command. After this command, you cannot
   continue working on this task.

## Important Rules

1. Every response must contain exactly one tool call to `bash`.
2. Directory and environment-variable changes ARE persistent across calls
   in this harness — you can `cd` and `export` and subsequent commands will
   see the change. (This differs from upstream mini-swe-agent's subshell
   model; treat the shell as a long-running login shell.)
3. Long-running commands: wrap with `timeout`, e.g. `timeout 30 <command>`.
4. Interactive commands are not possible. Use `yes`/`no` piping or
   non-interactive flags as appropriate.
5. Output may be truncated. Use `head`, `tail`, `grep`, `sed -n 'A,Bp'`,
   etc. to filter large outputs.

## Useful command examples

### Create a new file:
`cat <<'EOF' > newfile.py
import numpy as np
hello = "world"
print(hello)
EOF`

### Edit files with sed:
`sed -i 's/old_string/new_string/g' filename.py`        # all occurrences
`sed -i '1s/old_string/new_string/' filename.py`         # first on line 1
`sed -i '1,10s/old_string/new_string/g' filename.py`     # lines 1-10

### View file content:
`nl -ba filename.py | sed -n '10,20p'`
"""

TOO_LONG_HINT = """\
The output of your last command was too long.
Please try a different command that produces less output.
If you're looking at a file you can try use head, tail or sed to view a
smaller number of lines selectively. If you're using grep or find and it
produced too much output, you can use a more selective search pattern.
If you really need to see something from the full command's output, you
can redirect output to a file and then search in that file.
"""

FORMAT_ERROR_TEMPLATE = """\
Format error: {error}

Please always provide EXACTLY ONE call to the `bash` tool. If you want to
end the task, please issue the command `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`
via the `bash` tool, with no other content in the command.
"""

NO_TOOL_CALL_ERROR = "Your last response did not include a valid `bash` tool call."


def render_instance(task: str) -> str:
    """Wrap a task's instruction in the instance template."""
    return INSTANCE_TEMPLATE.replace("{task}", task)


def format_error_message(error: str) -> str:
    """Render the message sent back after a reply without a usable call."""
    return FORMAT_ERROR_TEMPLATE.replace("{error}", error)


def truncate_observation(output: str) -> str:
    """Keep the head and tail of a long output, with a hint in between."""
    if len(output) <= OBSERVATION_MAX_CHARS:
        return output
    elided = len(output) - OBSERVATION_HEAD_CHARS - OBSERVATION_TAIL_CHARS
    return (
        f"{TOO_LONG_HINT}\n\n"
        f"---- HEAD ({OBSERVATION_HEAD_CHARS} chars) ----\n"
        f"{output[:OBSERVATION_HEAD_CHARS]}\n"
        f"---- {elided} chars elided ----\n"
        f"---- TAIL ({OBSERVATION_TAIL_CHARS} chars) ----\n"
        f"{output[-OBSERVATION_TAIL_CHARS:]}"
    )


def format_observation(output: str, exit_code: int) -> str:
    """Render a command's result the way the model reads it."""
    body = truncate_observation(output.rstrip()) if output.strip() else "(no output)"
    return f"{body}\n\n(exit_code={exit_code})"


@dataclass(frozen=True)
class ParsedAction:
    """What a model reply asked for.

    ``kind`` is ``"command"`` for a bash call, ``"done"`` for a bash call that
    carries the submit marker, and ``"no_tool_call"`` for anything else.
    """

    kind: str
    command: str = ""
    tool_call: ToolCall | None = None


def parse_action(output: LMOutput) -> ParsedAction:
    """Read the first tool call of a reply as a bash action."""
    if not output.tool_calls:
        return ParsedAction(kind="no_tool_call")
    call = output.tool_calls[0]
    if call.function.name != BASH_TOOL_NAME:
        return ParsedAction(kind="no_tool_call", tool_call=call)
    raw = call.function.arguments
    try:
        arguments = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, ValueError):
        return ParsedAction(kind="no_tool_call", tool_call=call)
    if not isinstance(arguments, dict):
        return ParsedAction(kind="no_tool_call", tool_call=call)
    command = str(arguments.get("command") or "").strip()
    if not command:
        return ParsedAction(kind="no_tool_call", tool_call=call)
    kind = "done" if SUBMIT_MARKER in command else "command"
    return ParsedAction(kind=kind, command=command, tool_call=call)


@register_scaffold("vanillux")
class VanilluxScaffold(Scaffold):
    """Drive a model through a task with one ``bash`` tool and the mini-swe-agent prompts.

    Scaffold kwargs (``HarnessConfig.scaffold_kwargs`` or ``run(**kwargs)``):

    - ``command_timeout``: seconds each command may run (default 120).
    - ``max_format_errors``: consecutive replies without a valid call before
      the run stops (default 64).
    - ``agent_timeout``: wall-clock budget for the whole run in seconds. When
      set, no new step starts past it and the last command is cut to fit.

    ``HarnessConfig.max_turns`` bounds the number of steps (default 64). The
    config's tools must include a session-bound ``bash`` tool.
    """

    name = "vanillux"

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
        max_steps = config.max_turns or DEFAULT_MAX_STEPS
        command_timeout = float(options.get("command_timeout", DEFAULT_COMMAND_TIMEOUT))
        max_format_errors = int(options.get("max_format_errors", DEFAULT_MAX_FORMAT_ERRORS))
        agent_timeout = options.get("agent_timeout")
        params = sampling_params or DEFAULT_SAMPLING_PARAMS
        task_id = (trace_metadata or {}).get("task_id", config.name)

        bash_tool = next(
            (t for t in config.resolved_tools if t.name == BASH_TOOL_NAME and t.session), None
        )
        if bash_tool is None:
            raise ValueError(
                f"The {self.name} scaffold needs a session tool named {BASH_TOOL_NAME!r} "
                "in the harness config."
            )
        tool_schemas = (bash_tool.schema,)

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

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": config.system_prompt or SYSTEM_TEMPLATE},
            {"role": "user", "content": render_instance(instruction.strip())},
        ]
        turns: list[AgentTurn] = []
        timing: list[dict[str, Any]] = []
        usage = {"prompt_tokens": 0, "completion_tokens": 0}
        format_errors = 0
        steps = 0
        last_text = ""
        completion_reason = "max_steps"
        error: str | None = None
        started = time.monotonic()
        deadline = started + float(agent_timeout) if agent_timeout is not None else None

        binding = await self._sandbox_manager.acquire_binding(Capability.BASH)
        try:
            for step in range(max_steps):
                if deadline is not None and time.monotonic() >= deadline:
                    completion_reason = "timeout"
                    break

                llm_started = time.monotonic()
                outputs = await provider.agenerate(
                    [
                        LMRequest(
                            request_type=RequestType.CHAT,
                            messages=tuple(messages),
                            tools=tool_schemas,
                        )
                    ],
                    params,
                )
                llm_seconds = time.monotonic() - llm_started
                replies = outputs[0] if outputs else []
                if is_context_overflow_output(replies):
                    logger.warning("Task %s outgrew the context window at step %s", task_id, step)
                    completion_reason = "context_overflow"
                    break
                failure = request_error(replies)
                if failure is not None or not replies:
                    completion_reason = "error"
                    error = failure or "The provider returned no output."
                    break

                output = replies[0]
                steps += 1
                last_text = output.text or ""
                self._accumulate_usage(output, usage)
                action = parse_action(output)

                if action.kind == "no_tool_call":
                    format_errors += 1
                    self._record_format_error(messages, turns, output, action)
                    timing.append(
                        {"step": steps, "llm_s": round(llm_seconds, 1), "format_error": True}
                    )
                    if format_errors >= max_format_errors:
                        completion_reason = "format_errors"
                        break
                    continue

                format_errors = 0
                call = action.tool_call
                assert call is not None
                messages.append(
                    {"role": "assistant", "content": last_text, "tool_calls": [call.to_openai()]}
                )
                turns.append(AgentTurn.assistant(content=last_text, tool_calls=[call]))

                timeout = command_timeout
                if deadline is not None:
                    timeout = max(1.0, min(command_timeout, deadline - time.monotonic()))
                exec_started = time.monotonic()
                result = await binding.execute_in_session(action.command, timeout=timeout)
                exec_seconds = time.monotonic() - exec_started

                observation = format_observation(result.output, result.exit_code)
                messages.append({"role": "tool", "tool_call_id": call.id, "content": observation})
                turns.append(
                    AgentTurn.tool(
                        [
                            ToolResult(
                                tool_call_id=call.id,
                                content=observation,
                                is_error=result.exit_code != 0,
                            )
                        ]
                    )
                )
                timing.append(
                    {
                        "step": steps,
                        "llm_s": round(llm_seconds, 1),
                        "bash_s": round(exec_seconds, 1),
                        "exit_code": result.exit_code,
                        "cmd": action.command[:200],
                    }
                )

                if action.kind == "done" or SUBMIT_MARKER in observation:
                    completion_reason = "submitted"
                    break
        finally:
            await binding.release()

        metadata = {
            "completion_reason": completion_reason,
            "steps": steps,
            "format_errors": format_errors,
            "usage": dict(usage),
            "timing": timing,
            "duration_s": round(time.monotonic() - started, 1),
        }
        logger.info("Task %s finished after %s steps: %s", task_id, steps, completion_reason)
        final_output = LMOutput(text=last_text, metadata={"completion_reason": completion_reason})
        return HarnessResult(
            final_output=final_output,
            trajectory=AgentTrajectory(turns=tuple(turns), metadata=dict(metadata)),
            max_turns_reached=completion_reason == "max_steps",
            error=error,
            metadata=metadata,
        )

    @staticmethod
    def _accumulate_usage(output: LMOutput, usage: dict[str, int]) -> None:
        metadata = output.metadata or {}
        for key in usage:
            value = metadata.get(key)
            if isinstance(value, int | float):
                usage[key] += int(value)

    @staticmethod
    def _record_format_error(
        messages: list[dict[str, Any]],
        turns: list[AgentTurn],
        output: LMOutput,
        action: ParsedAction,
    ) -> None:
        """Keep the reply and answer it with the format-error message.

        A reply that called some tool with well-formed arguments gets the error
        as that call's result, so the conversation stays well formed. A call
        whose arguments are not a JSON object cannot go back to an
        OpenAI-compatible server as a tool call: the server re-renders the
        history through the chat template and rejects the request, which would
        end the run. Such a call is kept as text in the reply instead, and the
        error follows as a user message, as it does for a reply with no call.
        """
        text = output.text or ""
        content = format_error_message(NO_TOOL_CALL_ERROR)
        call = action.tool_call
        if call is not None and call.id and _arguments_are_an_object(call):
            messages.append(
                {"role": "assistant", "content": text, "tool_calls": [call.to_openai()]}
            )
            messages.append({"role": "tool", "tool_call_id": call.id, "content": content})
            turns.append(AgentTurn.assistant(content=text, tool_calls=[call]))
            turns.append(
                AgentTurn.tool([ToolResult(tool_call_id=call.id, content=content, is_error=True)])
            )
            return
        if call is not None:
            rendered = f"{call.function.name}({call.function.arguments})"
            text = "\n".join(part for part in (text, rendered) if part)
        messages.append({"role": "assistant", "content": text})
        messages.append({"role": "user", "content": content})
        turns.append(AgentTurn.assistant(content=text))
        turns.append(AgentTurn.user(content=content))


def _arguments_are_an_object(call: ToolCall) -> bool:
    """True when a call's arguments decode to a JSON object a server can re-render."""
    raw = call.function.arguments
    if isinstance(raw, dict):
        return True
    try:
        return isinstance(json.loads(raw), dict)
    except (TypeError, ValueError):
        return False
