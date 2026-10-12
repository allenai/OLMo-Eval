"""Helpers shared by the ingest and read services."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from typing import Any

SCALE_TOLERANCE = 1e-6


def now_utc() -> datetime:
    return datetime.now(UTC)


def detect_scale(mean: float, score: float) -> float | None:
    """1 or 100 when ``score`` equals ``mean`` on that scale, else None."""
    for scale in (1.0, 100.0):
        if abs(mean * scale - score) <= SCALE_TOLERANCE + SCALE_TOLERANCE * abs(score):
            return scale
    return None


def _field(row: Any, name: str) -> Any:
    return row[name] if isinstance(row, Mapping) else getattr(row, name)


def latest_by_task[T](rows: Iterable[T]) -> dict[str, T]:
    """One task result per task name, the most recently uploaded when a run has several hashes.

    ``updated_at`` is the time of the last upload of the task result or its instances
    (finalizing does not change it); the larger id breaks an exact tie. Works on ORM objects
    and on row mappings.
    """
    out: dict[str, T] = {}
    for row in rows:
        name = _field(row, "task_name")
        prev = out.get(name)
        order = (_field(row, "updated_at"), _field(row, "id") or 0)
        if prev is None or order > (_field(prev, "updated_at"), _field(prev, "id") or 0):
            out[name] = row
    return out
