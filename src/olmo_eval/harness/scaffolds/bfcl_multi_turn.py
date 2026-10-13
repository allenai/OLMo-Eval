"""Rollout scaffold for the BFCL multi-turn categories.

A multi-turn entry is a conversation, not a single request. For each user turn
the model is asked for calls, those calls are run against the entry's API
instances, their results go back as the next message, and the turn ends when
the model stops calling. What the model did is graded afterwards, by replaying
its calls against fresh instances, so the state built up here only serves to
answer the model.

Two of the categories perturb the conversation. A turn may hold back functions
the model needs, offering them partway through with a stock message; and a turn
may omit something the model needs, so that it should ask rather than act.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from olmo_eval.common.scorers.bfcl import DecodeError, decode_text, decode_tool_calls
from olmo_eval.common.scorers.bfcl.constants import Language
from olmo_eval.common.scorers.bfcl.multi_turn import (
    CONTEXT_OVERFLOW_METADATA_KEY,
    STEP_BUDGET_METADATA_KEY,
    build_instances,
    calls_from_decoded,
    execute_calls,
    render_call,
)
from olmo_eval.common.types import LMOutput, LMRequest, RequestType, SamplingParams, ToolSchema
from olmo_eval.harness.config import HarnessConfig
from olmo_eval.harness.result import HarnessResult
from olmo_eval.harness.scaffolds import Scaffold, register_scaffold
from olmo_eval.inference.base import InferenceProvider, require_tool_support
from olmo_eval.inference.errors import first_output, is_context_overflow_output, request_error

logger = logging.getLogger(__name__)

#: A turn stops once more than this many steps have run, so it can run one
#: more; the reference implementation counts the same way. Set ``max_steps``
#: in the harness's ``scaffold_kwargs`` to change it.
MAXIMUM_STEP_LIMIT = 20

#: What the user says when functions held back from a turn are offered.
ADDITIONAL_FUNCTION_PROMPT = (
    "I have updated some more functions you can choose from. What about now?"
)


@register_scaffold("bfcl_multi_turn")
class BFCLMultiTurnScaffold(Scaffold):
    """Drive one BFCL multi-turn entry to completion."""

    name = "bfcl_multi_turn"

    async def run(
        self,
        provider: InferenceProvider,
        config: HarnessConfig,
        request: LMRequest,
        sampling_params: SamplingParams | None = None,
        trace_metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> HarnessResult:
        payload = request.metadata or {}
        if "turns" not in payload:
            raise ValueError(
                "The bfcl_multi_turn scaffold needs a request carrying the entry's turns; "
                "it is meant for the bfcl_multi_turn_* tasks."
            )

        turns: list[list[dict[str, Any]]] = payload["turns"]
        held_back: dict[str, list[dict[str, Any]]] = payload.get("missed_function") or {}
        held_back_docs: dict[str, str] = payload.get("missed_function_docs") or {}
        language = Language(payload.get("language", Language.PYTHON))
        from_tool_calls = payload.get("call_source") == "tool_calls"
        if from_tool_calls:
            # The harness leaves tool support to a scaffold, and this one sends
            # its schemas through the provider, which would drop them.
            require_tool_support(provider)
        max_steps = int(kwargs.get("max_steps", MAXIMUM_STEP_LIMIT))

        instances = build_instances(
            payload["initial_config"],
            payload["involved_classes"],
            bool(payload.get("long_context", False)),
        )

        messages: list[dict[str, Any]] = [dict(m) for m in request.messages]
        tools: list[ToolSchema] = list(request.tools or ())
        calls_per_turn: list[list[list[dict[str, Any]]]] = []
        step_budget_exhausted = False
        context_overflow: str | None = None
        last_output = LMOutput(text="")

        for turn_index, turn_messages in enumerate(turns):
            if str(turn_index) in held_back:
                offered = held_back[str(turn_index)]
                announcement = ADDITIONAL_FUNCTION_PROMPT
                if from_tool_calls:
                    tools.extend(ToolSchema.from_openai(schema) for schema in offered)
                else:
                    # A prompted model reads its functions from the conversation,
                    # so the ones being offered have to appear in the message, in
                    # the form the rest of them were written in.
                    docs = held_back_docs.get(str(turn_index))
                    if docs is None:
                        raise ValueError(
                            "A prompted rollout needs the offered functions written out: "
                            f"turn {turn_index} holds functions back but carries no "
                            "'missed_function_docs' entry for them."
                        )
                    announcement = f"{docs}\n{announcement}"
                turn_messages = [{"role": "user", "content": announcement}]

            messages.extend(dict(m) for m in turn_messages)
            steps: list[list[dict[str, Any]]] = []

            while True:
                outputs = await provider.agenerate(
                    [
                        LMRequest(
                            request_type=RequestType.CHAT,
                            messages=tuple(messages),
                            tools=tuple(tools) or None,
                            system_prompt=request.system_prompt,
                        )
                    ],
                    sampling_params,
                )
                replies = outputs[0] if outputs else []
                if is_context_overflow_output(replies):
                    # The conversation no longer fits the model's window. The
                    # reference implementation grades such an entry wrong rather
                    # than abandoning the run, so the rollout ends here.
                    context_overflow = request_error(replies) or "context overflow"
                    logger.warning(
                        "BFCL multi-turn entry %s outgrew the context window on turn %s "
                        "and is graded wrong.",
                        (trace_metadata or {}).get("instance_id", "?"),
                        turn_index,
                    )
                    break
                last_output = first_output(replies)

                decoded = self._decode(last_output, from_tool_calls, language, payload)
                if not decoded:
                    self._record_assistant(messages, last_output, from_tool_calls)
                    break

                steps.append(decoded)
                calls = calls_from_decoded(decoded)
                results = execute_calls(calls, instances)
                self._record_assistant(messages, last_output, from_tool_calls)
                self._record_results(messages, last_output, calls, results, from_tool_calls)

                if len(steps) > max_steps:
                    step_budget_exhausted = True
                    logger.warning(
                        "BFCL multi-turn entry %s stopped after %s steps on turn %s.",
                        (trace_metadata or {}).get("instance_id", "?"),
                        len(steps),
                        turn_index,
                    )
                    break

            calls_per_turn.append(steps)
            if step_budget_exhausted or context_overflow is not None:
                break

        final = LMOutput(text=last_output.text, metadata=dict(last_output.metadata or {}))
        final.extracted_answer = calls_per_turn
        if step_budget_exhausted:
            final.metadata[STEP_BUDGET_METADATA_KEY] = True
        if context_overflow is not None:
            final.metadata[CONTEXT_OVERFLOW_METADATA_KEY] = context_overflow

        return HarnessResult(
            final_output=final,
            max_turns_reached=step_budget_exhausted,
            metadata={"turns": len(calls_per_turn)},
        )

    def _decode(
        self,
        output: LMOutput,
        from_tool_calls: bool,
        language: Language,
        payload: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Read one reply's calls, treating an unreadable reply as none."""
        try:
            if from_tool_calls:
                return decode_tool_calls(output.tool_calls, payload.get("name_map") or {})
            return decode_text(output.text, language)
        except DecodeError:
            return []

    def _record_assistant(
        self, messages: list[dict[str, Any]], output: LMOutput, from_tool_calls: bool
    ) -> None:
        message: dict[str, Any] = {"role": "assistant", "content": output.text or ""}
        if from_tool_calls and output.tool_calls:
            if all(_arguments_are_an_object(call) for call in output.tool_calls):
                message["tool_calls"] = [call.to_openai() for call in output.tool_calls]
            else:
                # A call whose arguments are not a JSON object cannot travel back to
                # an OpenAI-compatible endpoint as a tool call: vLLM re-renders the
                # history through the chat template and rejects the request (400),
                # which would drop the entry from the run instead of grading the
                # step wrong as the reference does. Keep the model's words as text,
                # the way a prompted rollout would see them, and let the rollout
                # continue; `_decode` has already read this step as no calls.
                rendered = "\n".join(
                    f"{call.function.name}({call.function.arguments})" for call in output.tool_calls
                )
                message["content"] = "\n".join(
                    part for part in (message["content"], rendered) if part
                )
        messages.append(message)

    def _record_results(
        self,
        messages: list[dict[str, Any]],
        output: LMOutput,
        calls: list[Any],
        results: list[str],
        from_tool_calls: bool,
    ) -> None:
        """Put the execution results where the model will read them next."""
        if from_tool_calls and output.tool_calls:
            for tool_call, result in zip(output.tool_calls, results, strict=False):
                messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result})
            return
        # A prompted model gets one tool message per result, named after the call
        # that produced it, as the reference implementation's handler for local
        # models sends them. The chat template decides how a tool message reads.
        for call, result in zip(calls, results, strict=False):
            messages.append({"role": "tool", "name": render_call(call), "content": result})


def _arguments_are_an_object(call: Any) -> bool:
    """True when a tool call's arguments decode to a JSON object the endpoint can re-render."""
    raw = call.function.arguments
    if raw is None or raw == "" or isinstance(raw, dict):
        return True
    try:
        return isinstance(json.loads(raw), dict)
    except (TypeError, ValueError):
        return False
