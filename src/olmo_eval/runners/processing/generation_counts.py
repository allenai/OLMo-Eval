"""Per-task counts of generations that did not end cleanly."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from olmo_eval.common.multi_turn import turn_replies
from olmo_eval.common.types import RequestType, Response

GENERATION_COUNT_KEYS = (
    "generations",
    "cap_hit",
    "empty",
    "unclosed_think",
    "finish_reason_unknown",
)

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


def _unclosed_think(text: str) -> bool:
    return text.rfind(_THINK_OPEN) > text.rfind(_THINK_CLOSE)


def count_generations(responses: Sequence[Response]) -> dict[str, int] | None:
    """Count generation outcomes after scoring; None if there are no generations.

    A scripted conversation counts once, as unclosed if any of its turns is.
    """
    counts = dict.fromkeys(GENERATION_COUNT_KEYS, 0)
    for response in responses:
        if response.request.request_type == RequestType.LOGLIKELIHOOD:
            continue
        for output in response.outputs:
            metadata = output.metadata or {}
            text = output.text or ""
            raw = metadata.get("original_text", text)
            finish_reason = metadata.get("finish_reason")

            counts["generations"] += 1
            if finish_reason is None:
                counts["finish_reason_unknown"] += 1
            elif finish_reason == "length":
                counts["cap_hit"] += 1
            empty = metadata.get("scoring_text_empty", not text.strip())
            if empty and not output.has_tool_calls:
                counts["empty"] += 1
            replies = turn_replies(output) or [raw]
            if any(_unclosed_think(reply) for reply in replies):
                counts["unclosed_think"] += 1

    if counts["generations"] == 0:
        return None
    return counts


def sum_generation_counts(
    task_results: Mapping[str, Mapping[str, Any]], task_specs: Sequence[str]
) -> dict[str, int] | None:
    """Sum the counts of the given tasks; None if none carries counts."""
    total = dict.fromkeys(GENERATION_COUNT_KEYS, 0)
    found = False
    for spec in dict.fromkeys(task_specs):
        counts: Mapping[str, int] | None = (task_results.get(spec) or {}).get("generation_counts")
        if not counts:
            continue
        found = True
        for key in GENERATION_COUNT_KEYS:
            total[key] += counts.get(key, 0)
    return total if found else None


def format_generation_counts(counts: Mapping[str, int] | None) -> str:
    """Render counts compactly for log lines and the summary table."""
    if not counts or not counts.get("generations"):
        return "-"
    total = counts["generations"]
    parts = [
        f"cap-hit {counts.get('cap_hit', 0)}/{total}",
        f"empty {counts.get('empty', 0)}",
    ]
    unclosed = counts.get("unclosed_think", 0)
    if unclosed:
        parts.append(f"unclosed-think {unclosed}")
    unknown = counts.get("finish_reason_unknown", 0)
    if unknown:
        parts.append(f"finish-reason unknown {unknown}")
    return ", ".join(parts)


__all__ = [
    "GENERATION_COUNT_KEYS",
    "count_generations",
    "format_generation_counts",
    "sum_generation_counts",
]
