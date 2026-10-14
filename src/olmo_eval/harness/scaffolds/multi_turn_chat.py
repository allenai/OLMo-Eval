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

from olmo_eval.common.types import LMOutput, LMRequest, RequestType, SamplingParams
from olmo_eval.harness.config import HarnessConfig
from olmo_eval.harness.result import HarnessResult
from olmo_eval.harness.scaffolds import Scaffold, register_scaffold
from olmo_eval.inference.base import InferenceProvider
from olmo_eval.inference.errors import first_output, is_context_overflow_output, request_error

logger = logging.getLogger(__name__)

#: ``LMRequest.metadata`` key holding the user messages sent after the first reply.
FOLLOW_UP_TURNS_KEY = "follow_up_turns"

#: ``LMOutput.metadata`` key holding the model's reply to each user turn, in order.
REPLIES_KEY = "replies"

#: ``LMOutput.metadata`` key set when the conversation outgrew the context window.
CONTEXT_OVERFLOW_KEY = "conversation_context_overflow"


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
        replies: list[str] = []
        context_overflow: str | None = None
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
                        system_prompt=request.system_prompt,
                    )
                ],
                sampling_params,
            )
            generated = outputs[0] if outputs else []
            if is_context_overflow_output(generated):
                # Later turns only add to the conversation, so none of them can
                # be asked; the task scores them as unanswered.
                context_overflow = request_error(generated) or "context overflow"
                logger.warning(
                    "Conversation %s outgrew the context window on turn %s.",
                    (trace_metadata or {}).get("instance_id", "?"),
                    turn_index + 1,
                )
                break
            last_output = first_output(generated)
            replies.append(last_output.text)

        final = LMOutput(text=last_output.text, metadata=dict(last_output.metadata or {}))
        final.metadata[REPLIES_KEY] = replies
        if context_overflow is not None:
            final.metadata[CONTEXT_OVERFLOW_KEY] = context_overflow

        return HarnessResult(final_output=final, metadata={"turns": len(replies)})
