"""Per-task runtime derived from run timing and token counts.

Tasks run interleaved through shared inference workers, so a task result's ``duration_seconds``
(the run's processing start until the task finished) is not the task's own cost: every task of
a 20-task run can show about the same duration. ``derive_runtime`` gives each task result an
inference time and the basis it rests on, in this order of precedence:

- ``measured``: the run has exactly one task result and ``processing_seconds`` is known, so the
  run's processing time is the task's time.
- ``attributed``: the client recorded ``attributed_inference_seconds`` (per-batch wall time split
  by the task's share of each batch's tokens).
- ``estimated``: ``processing_seconds`` is known and every task result of the run has prompt and
  completion token totals: processing_seconds x (task tokens / run tokens), where tokens are
  prompt plus completion tokens.
- ``not_recorded``: none of the above; ``inference_seconds`` is null.

``startup_seconds`` is the run's ``startup_seconds``. Runs uploaded before it was recorded fall
back to the largest ``provider_init_seconds`` value (the slowest worker's start, which is when
every worker was ready).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from olmo_eval_api.schemas import api as a

# Task result columns derive_runtime reads (with tr.num_instances from TR_COLUMNS).
RUNTIME_TR_COLUMNS = """
    tr.first_request_at, tr.last_completed_at, tr.prompt_tokens_total,
    tr.completion_tokens_total, tr.attributed_inference_seconds
"""

# GPU type and count of a run (``runs r``): what olmo-eval detected, else the Beaker allocation.
GPU_TYPE_SQL = "(r.environment ->> 'gpu_type')"
GPU_COUNT_SQL = """
    COALESCE(
        CASE WHEN jsonb_typeof(r.environment -> 'gpu_count') = 'number'
             THEN (r.environment ->> 'gpu_count')::int END,
        CASE WHEN jsonb_typeof(r.beaker -> 'gpu_count') = 'number'
             THEN (r.beaker ->> 'gpu_count')::int END
    )
"""

_RUN_TIMING_SQL = """
    SELECT r.run_id, r.startup_seconds, r.processing_started_at, r.processing_seconds,
           r.provider_init_seconds, t.n, t.all_tokens, t.tokens
    FROM runs r
    LEFT JOIN LATERAL (
        SELECT count(*) AS n,
               bool_and(prompt_tokens_total IS NOT NULL AND completion_tokens_total IS NOT NULL)
                   AS all_tokens,
               sum(prompt_tokens_total + completion_tokens_total) AS tokens
        FROM task_results WHERE run_id = r.run_id
    ) t ON TRUE
    WHERE r.run_id = ANY(:ids)
"""


@dataclass(frozen=True)
class RunTiming:
    """What derive_runtime needs from a run and the token totals of all its task results."""

    startup_seconds: float | None = None
    processing_started_at: datetime | None = None
    processing_seconds: float | None = None
    n_task_results: int = 0
    # Every task result of the run has prompt and completion token totals.
    all_have_tokens: bool = False
    total_tokens: int | None = None


def run_startup_seconds(
    startup_seconds: float | None, provider_init_seconds: Mapping[str, Any] | None
) -> float | None:
    """The run's startup, falling back to the slowest provider init for older runs."""
    if startup_seconds is not None:
        return _finite(startup_seconds)
    values = [_finite(v) for v in (provider_init_seconds or {}).values()]
    known = [v for v in values if v is not None]
    return max(known) if known else None


def run_timing_from_row(row: Mapping[Any, Any]) -> RunTiming:
    tokens = row["tokens"]
    return RunTiming(
        startup_seconds=run_startup_seconds(row["startup_seconds"], row["provider_init_seconds"]),
        processing_started_at=row["processing_started_at"],
        processing_seconds=_finite(row["processing_seconds"]),
        n_task_results=int(row["n"] or 0),
        all_have_tokens=bool(row["all_tokens"]),
        total_tokens=int(tokens) if tokens is not None else None,
    )


async def load_run_timings(session: AsyncSession, run_ids: Iterable[str]) -> dict[str, RunTiming]:
    """RunTiming of several runs in one query, keyed by run_id."""
    ids = sorted(set(run_ids))
    if not ids:
        return {}
    rows = (await session.execute(text(_RUN_TIMING_SQL), {"ids": ids})).mappings().all()
    return {row["run_id"]: run_timing_from_row(row) for row in rows}


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _offset(at: datetime | None, start: datetime | None) -> float | None:
    if at is None or start is None:
        return None
    return (at - start).total_seconds()


def derive_runtime(tr: Mapping[Any, Any], run: RunTiming) -> a.TaskRuntime:
    """Runtime of one task result; see the module docstring for the rules."""
    prompt = tr["prompt_tokens_total"]
    completion = tr["completion_tokens_total"]
    task_tokens = prompt + completion if prompt is not None and completion is not None else None
    token_share = None
    if run.all_have_tokens and run.total_tokens and task_tokens is not None:
        token_share = task_tokens / run.total_tokens

    processing = run.processing_seconds
    attributed = _finite(tr["attributed_inference_seconds"])
    inference: float | None = None
    basis: a.RuntimeBasis = "not_recorded"
    if run.n_task_results == 1 and processing is not None:
        inference, basis = processing, "measured"
    elif attributed is not None:
        inference, basis = attributed, "attributed"
    elif processing is not None and token_share is not None:
        inference, basis = processing * token_share, "estimated"

    startup = run.startup_seconds
    n = tr["num_instances"]
    return a.TaskRuntime(
        inference_seconds=inference,
        basis=basis,
        startup_seconds=startup,
        with_startup_seconds=inference + startup
        if inference is not None and startup is not None
        else None,
        token_share=token_share,
        prompt_tokens_total=prompt,
        completion_tokens_total=completion,
        seconds_per_1k_instances=inference / n * 1000 if inference is not None and n else None,
        span_start_s=_offset(tr["first_request_at"], run.processing_started_at),
        span_end_s=_offset(tr["last_completed_at"], run.processing_started_at),
    )


async def runtimes_for(
    session: AsyncSession, trs: Iterable[Mapping[Any, Any]]
) -> dict[int, a.TaskRuntime]:
    """Runtime of each task result (keyed by id), loading the runs' timings in one query."""
    rows = list(trs)
    timings = await load_run_timings(session, (tr["run_id"] for tr in rows))
    empty = RunTiming()
    return {int(tr["id"]): derive_runtime(tr, timings.get(tr["run_id"], empty)) for tr in rows}


def runtime_stats(values: Sequence[float | None]) -> a.RuntimeStats:
    known = np.array([v for v in values if v is not None], dtype=np.float64)
    if not known.size:
        return a.RuntimeStats(n=0, median=None, p90=None, min=None, max=None)
    return a.RuntimeStats(
        n=int(known.size),
        median=float(np.median(known)),
        p90=float(np.percentile(known, 90)),
        min=float(known.min()),
        max=float(known.max()),
    )
