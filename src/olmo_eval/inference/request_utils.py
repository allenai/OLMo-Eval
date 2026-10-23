"""Helpers for interpreting ``LMRequest`` fields consistently across providers."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from olmo_eval.common.types import LMRequest

logger = logging.getLogger(__name__)
_DROPPED_SYSTEM_WARNED: list[bool] = []


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


def chat_with_images(
    messages: tuple[dict, ...], image_parts: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Chat turns as content-part lists, with ``image_parts`` attached in order.

    A turn whose ``content`` is a list of parts places an image at each ``{"type": "image"}``
    placeholder, so text and images can interleave as a benchmark specifies. Otherwise every
    image goes at the start of the first user turn (before its text), or into a leading user
    turn when there is none.

    :raises ValueError: When the placeholders do not match the number of images.
    """
    placed = any(isinstance(m.get("content"), list) for m in messages)
    remaining = list(image_parts)
    chat: list[dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content") or ""
        if isinstance(content, list):
            parts: list[dict[str, Any]] = []
            for part in content:
                if part.get("type") == "image":
                    if not remaining:
                        raise ValueError("more image placeholders than images in the request")
                    parts.append(remaining.pop(0))
                else:
                    parts.append(part)
        else:
            parts = [{"type": "text", "text": content}]
            if not placed and role == "user" and remaining:
                parts = [*remaining, *parts]
                remaining = []
        chat.append({"role": role, "content": parts})
    if remaining:
        if placed:
            raise ValueError("fewer image placeholders than images in the request")
        chat.insert(0, {"role": "user", "content": remaining})
    return chat


def template_has_system_role(chat_template: str | None) -> bool:
    """Whether a chat template renders system turns (it at least mentions the role)."""
    return chat_template is None or "system" in chat_template


def drop_system_turns(chat: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The conversation without its system turns, for chat templates with no system role.

    Such templates reject a system turn, and the model was never trained on system text,
    so moving it into the user turn puts unfamiliar text in front of the question. The rest
    of the conversation is sent unchanged; a warning is logged once.
    """
    kept = [m for m in chat if m["role"] != "system"]
    if len(kept) != len(chat) and not _DROPPED_SYSTEM_WARNED:
        _DROPPED_SYSTEM_WARNED.append(True)
        logger.warning("The chat template has no system role; dropping the system turn.")
    return kept
