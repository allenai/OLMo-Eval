"""Scaffold that plays a scripted conversation against the model.

Some benchmarks write every user turn in advance but let the model answer each
one itself, so a later turn is asked with the model's own earlier replies in
the conversation. This scaffold sends the request's messages, then each
follow-up user message in turn, carrying every reply forward as an assistant
message. Each reply goes back as the model wrote it, so the chat template
decides what an earlier turn shows, such as whether a reasoning trace is kept.

The final output records each turn under ``turns``, in order, for the task to
score and the predictions file to keep. Its own fields describe the whole
conversation: ``prompt_tokens`` and ``completion_tokens`` add up every turn
(left out if any turn did not report them). Each turn's prompt holds the
conversation so far, so the prompt total counts earlier turns again: it is the
compute spent, not the length of the longest prompt. ``finish_reason`` is
``"length"`` if any turn hit the token limit or the conversation outgrew the
context before its last turn, else the last turn's.
"""

from __future__ import annotations

import logging
from typing import Any

from olmo_eval.common.types import (
    FOLLOW_UP_TURNS_KEY,
    TURNS_KEY,
    LMOutput,
    LMRequest,
    RequestType,
    SamplingParams,
)
from olmo_eval.harness.config import HarnessConfig
from olmo_eval.harness.result import HarnessResult
from olmo_eval.harness.scaffolds import Scaffold, register_scaffold
from olmo_eval.inference.base import InferenceProvider
from olmo_eval.inference.errors import first_output, is_context_overflow_output, request_error

logger = logging.getLogger(__name__)

_TOKEN_KEYS = ("prompt_tokens", "completion_tokens")


def _turn_record(output: LMOutput) -> dict[str, Any]:
    """What the predictions keep of one turn: its reply, finish reason and token counts."""
    metadata = output.metadata or {}
    record: dict[str, Any] = {"text": output.text, "finish_reason": metadata.get("finish_reason")}
    record.update({key: metadata[key] for key in _TOKEN_KEYS if key in metadata})
    return record


def _conversation_metadata(
    last: LMOutput, turns: list[dict[str, Any]], *, cut_short: bool
) -> dict[str, Any]:
    """The final output's metadata, with token counts and finish reason for every turn."""
    metadata = dict(last.metadata or {})
    for key in _TOKEN_KEYS:
        metadata.pop(key, None)
        if turns and all(isinstance(turn.get(key), int) for turn in turns):
            metadata[key] = sum(turn[key] for turn in turns)
    if cut_short or any(turn["finish_reason"] == "length" for turn in turns):
        metadata["finish_reason"] = "length"
    metadata[TURNS_KEY] = turns
    return metadata


@register_scaffold("multi_turn_chat")
class MultiTurnChatScaffold(Scaffold):
    """Answer each scripted user turn with the model's own conversation so far."""

    name = "multi_turn_chat"

    async def run(
        self,
        provider: InferenceProvider,
        config: HarnessConfig,
        request: LMRequest,
        sampling_params: SamplingParams | None = None,
        trace_metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> HarnessResult:
        follow_ups: list[dict[str, Any]] = list(
            (request.metadata or {}).get(FOLLOW_UP_TURNS_KEY, ())
        )
        messages: list[dict[str, Any]] = [dict(m) for m in request.messages]
        system_prompt = config.system_prompt or request.system_prompt
        turns: list[dict[str, Any]] = []
        last_output = LMOutput(text="")

        for turn_index in range(len(follow_ups) + 1):
            if turn_index > 0:
                messages.append({"role": "assistant", "content": last_output.text})
                messages.append(dict(follow_ups[turn_index - 1]))

            outputs = await provider.agenerate(
                [
                    LMRequest(
                        request_type=RequestType.CHAT,
                        messages=tuple(messages),
                        system_prompt=system_prompt,
                    )
                ],
                sampling_params,
            )
            generated = outputs[0] if outputs else []
            if is_context_overflow_output(generated):
                # Later turns only add to the conversation, so none of them can
                # be asked; the task scores them as unanswered.
                logger.warning(
                    "Conversation %s outgrew the context window on turn %s: %s",
                    (trace_metadata or {}).get("instance_id", "?"),
                    turn_index + 1,
                    request_error(generated),
                )
                break
            last_output = first_output(generated)
            turns.append(_turn_record(last_output))

        cut_short = len(turns) < len(follow_ups) + 1
        final = LMOutput(
            text=last_output.text,
            metadata=_conversation_metadata(last_output, turns, cut_short=cut_short),
        )
        return HarnessResult(final_output=final, metadata={"turns": len(turns)})
