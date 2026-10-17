"""Scaffold that plays a scripted conversation against the model.

Some benchmarks write every user turn in advance but let the model answer each
one itself, so a later turn is asked with the model's own earlier replies in
the conversation. This scaffold sends the request's messages, then each
follow-up user message in turn, carrying every reply forward as an assistant
message. Each reply goes back as the model wrote it, so the chat template
decides what an earlier turn shows, such as whether a reasoning trace is kept.
The replies are returned in order for the task to score.
"""

from __future__ import annotations

import logging
from typing import Any

from olmo_eval.common.types import (
    FOLLOW_UP_TURNS_KEY,
    REPLIES_KEY,
    TURN_FINISH_REASONS_KEY,
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
        replies: list[str] = []
        finish_reasons: list[str | None] = []
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
            replies.append(last_output.text)
            finish_reasons.append((last_output.metadata or {}).get("finish_reason"))

        final = LMOutput(text=last_output.text, metadata=dict(last_output.metadata or {}))
        final.metadata[REPLIES_KEY] = replies
        final.metadata[TURN_FINISH_REASONS_KEY] = finish_reasons
        return HarnessResult(final_output=final, metadata={"turns": len(replies)})
