"""Helpers for interpreting ``LMRequest`` fields consistently across providers."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from olmo_eval.common.types import LMRequest


def chat_messages_for_request(request: LMRequest) -> tuple[dict, ...]:
    """The request's chat turns, preserving a completion prompt as a user turn.

    A completion-style request carries its question in ``prompt`` with empty
    ``messages``; the harness may then inject a system message into ``messages``,
    which must not displace the question. The prompt is appended as a user turn
    whenever no user turn exists yet.
    """
    messages = tuple(request.messages or ())
    if request.prompt is not None and not any(m.get("role") == "user" for m in messages):
        messages = (*messages, {"role": "user", "content": request.prompt})
    return messages
