"""Subjects (runs or model checkpoints), their task results, and per-instance vectors.

``r:<run_id>`` uses that run's task results. ``m:<model_id>`` takes, per task name, the most
recent finalized task result of that model without an error (falling back to the most recent
with an error). With ``group``, only runs in that experiment group count (spec 4.5).
"""

from __future__ import annotations

import math
import re
from collections import OrderedDict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from olmo_eval_api.errors import bad_request, not_found
from olmo_eval_api.schemas import api as a
from olmo_eval_api.services.common import detect_scale, latest_by_task
from olmo_eval_api.services.queries import (
    PREFERRED_ORDER,
    fetch_models,
    fetch_runs,
    model_ref,
    run_summary,
)
from olmo_eval_api.services.runtime import RUNTIME_TR_COLUMNS
from olmo_eval_api.stats.paired import Side

SUBJECT_RE = re.compile(r"^(r:[a-z0-9]{6,32}|m:[0-9a-f]{12})$")

TR_COLUMNS = (
    """
    tr.id, tr.run_id, tr.task_name, tr.task_hash, tr.model_id, tr.run_created_at,
    tr.primary_metric, tr.score, tr.stderr, tr.score_is_mean, tr.instance_scale, tr.metric_kind,
    tr.higher_is_better, tr.display_format, tr.metrics, tr.metric_meta, tr.num_instances,
    tr.instances_processed, tr.instances_failed, tr.instances_stored, tr.error,
    tr.duration_seconds, tr.updated_at, tr.finalized_at,
"""
    + RUNTIME_TR_COLUMNS
)

TaskResultRow = Mapping[Any, Any]


@dataclass
class Subject:
    key: str
    kind: str  # "run" | "model"
    model: Mapping[Any, Any]
    run: Mapping[Any, Any] | None
    trs: dict[str, TaskResultRow] = field(default_factory=dict)
    label: str = ""

    @property
    def run_ids(self) -> list[str]:
        if self.run is not None:
            return [self.run["run_id"]]
        return sorted({tr["run_id"] for tr in self.trs.values()})

    @property
    def latest_at(self) -> datetime | None:
        if self.run is not None:
            return self.run["created_at"]
        dates = [tr["run_created_at"] for tr in self.trs.values()]
        return max(dates) if dates else None

    def info(self) -> a.SubjectInfo:
        return a.SubjectInfo(
            key=self.key,
            kind="run" if self.kind == "run" else "model",
            label=self.label,
            model=model_ref(self.model),
            run=run_summary(self.run) if self.run is not None else None,
            run_ids=self.run_ids,
            latest_at=self.latest_at,
        )


def validate_subject_key(key: str) -> str:
    if not SUBJECT_RE.match(key):
        raise bad_request(f"invalid subject key {key!r}")
    return key


async def run_task_results(session: AsyncSession, run_id: str) -> list[TaskResultRow]:
    sql = f"SELECT {TR_COLUMNS} FROM task_results tr WHERE tr.run_id = :rid ORDER BY tr.id"
    return list((await session.execute(text(sql), {"rid": run_id})).mappings().all())


async def runs_task_results(
    session: AsyncSession, run_ids: Sequence[str]
) -> dict[str, list[TaskResultRow]]:
    """Task results of several runs in one query, keyed by run_id."""
    out: dict[str, list[TaskResultRow]] = {rid: [] for rid in run_ids}
    if not run_ids:
        return out
    sql = f"SELECT {TR_COLUMNS} FROM task_results tr WHERE tr.run_id = ANY(:rids) ORDER BY tr.id"
    for row in (await session.execute(text(sql), {"rids": list(run_ids)})).mappings():
        out[row["run_id"]].append(row)
    return out


async def models_task_results(
    session: AsyncSession,
    model_ids: Sequence[str],
    group: str | None = None,
    task_names: Sequence[str] | None = None,
) -> dict[str, dict[str, TaskResultRow]]:
    """The model-subject task results (spec 4.5) of several models in one query."""
    out: dict[str, dict[str, TaskResultRow]] = {mid: {} for mid in model_ids}
    if not model_ids:
        return out
    filters = ["tr.model_id = ANY(:mids)", "tr.finalized_at IS NOT NULL"]
    params: dict[str, Any] = {"mids": list(model_ids)}
    if group is not None:
        filters.append("r.experiment_group = :grp")
        params["grp"] = group
    if task_names is not None:
        filters.append("tr.task_name = ANY(:tasks)")
        params["tasks"] = list(task_names)
    sql = f"""
        SELECT DISTINCT ON (tr.model_id, tr.task_name) {TR_COLUMNS}
        FROM task_results tr JOIN runs r ON r.run_id = tr.run_id
        WHERE {" AND ".join(filters)}
        ORDER BY tr.model_id, tr.task_name, {PREFERRED_ORDER}
    """
    for row in (await session.execute(text(sql), params)).mappings():
        out[row["model_id"]][row["task_name"]] = row
    return out


def base_label(model: Mapping[Any, Any]) -> str:
    label = model["m_series_label"]
    if model["m_step"] is not None:
        label += f" @ step {model['m_step']}"
    return label


def assign_labels(subjects: Sequence[Subject]) -> None:
    bases = [base_label(s.model) for s in subjects]
    counts: dict[str, int] = {}
    for b in bases:
        counts[b] = counts.get(b, 0) + 1
    for s, b in zip(subjects, bases, strict=True):
        if s.kind == "run" and counts[b] > 1 and s.run is not None and s.run["experiment_name"]:
            s.label = f"{b} · {s.run['experiment_name']}"
        else:
            s.label = b


async def resolve_subjects(
    session: AsyncSession,
    keys: Sequence[str],
    group: str | None = None,
    *,
    load_results: bool = True,
) -> tuple[list[Subject], list[str]]:
    """Subjects in request order (duplicates dropped) and the keys that do not exist."""
    keys = list(dict.fromkeys(keys))
    for key in keys:
        validate_subject_key(key)
    run_ids = [k[2:] for k in keys if k.startswith("r:")]
    model_ids = [k[2:] for k in keys if k.startswith("m:")]
    runs = await fetch_runs(session, run_ids)
    models = await fetch_models(session, model_ids)
    run_trs: dict[str, list[TaskResultRow]] = {}
    model_trs: dict[str, dict[str, TaskResultRow]] = {}
    if load_results:
        run_trs = await runs_task_results(session, list(runs))
        model_trs = await models_task_results(session, list(models), group)
    subjects: list[Subject] = []
    missing: list[str] = []
    for key in keys:
        ident = key[2:]
        if key.startswith("r:"):
            run = runs.get(ident)
            if run is None:
                missing.append(key)
                continue
            subject = Subject(key=key, kind="run", model=run, run=run)
            if load_results:
                subject.trs = latest_by_task(run_trs[ident])
        else:
            model = models.get(ident)
            if model is None:
                missing.append(key)
                continue
            subject = Subject(key=key, kind="model", model=model, run=None)
            if load_results:
                subject.trs = model_trs[ident]
        subjects.append(subject)
    assign_labels(subjects)
    return subjects, missing


async def resolve_one(
    session: AsyncSession, key: str, group: str | None = None, *, load_results: bool = True
) -> Subject:
    subjects, missing = await resolve_subjects(session, [key], group, load_results=load_results)
    if missing:
        raise not_found(f"Subject {key} not found")
    return subjects[0]


async def resolve_for_compare(
    session: AsyncSession, keys: Sequence[str], group: str | None
) -> list[Subject]:
    subjects, missing = await resolve_subjects(session, keys, group)
    if missing:
        raise bad_request(f"unknown subjects: {', '.join(missing)}")
    return subjects


# ---------------------------------------------------------------------------
# Per-instance vectors
# ---------------------------------------------------------------------------

Vector = tuple[np.ndarray, np.ndarray]
# Non-primary metric vectors keyed by (task_result_id, updated_at, metric key). Capped by the
# bytes of the arrays (an entry is two arrays of n elements, 16 bytes per instance).
_VECTOR_CACHE: OrderedDict[tuple[int, str, str], Vector] = OrderedDict()
_VECTOR_CACHE_MAX_BYTES = 128 * 1024 * 1024
_vector_cache_bytes = 0


def _vector_bytes(value: Vector) -> int:
    return int(value[0].nbytes + value[1].nbytes)


def _cache_get(key: tuple[int, str, str]) -> Vector | None:
    hit = _VECTOR_CACHE.get(key)
    if hit is not None:
        _VECTOR_CACHE.move_to_end(key)
    return hit


def _cache_put(key: tuple[int, str, str], value: Vector) -> None:
    global _vector_cache_bytes
    size = _vector_bytes(value)
    if size > _VECTOR_CACHE_MAX_BYTES // 4 or key in _VECTOR_CACHE:
        return
    _VECTOR_CACHE[key] = value
    _vector_cache_bytes += size
    while _vector_cache_bytes > _VECTOR_CACHE_MAX_BYTES:
        _, old = _VECTOR_CACHE.popitem(last=False)
        _vector_cache_bytes -= _vector_bytes(old)


async def load_primary_vectors(session: AsyncSession, tr_ids: Iterable[int]) -> dict[int, Vector]:
    ids = sorted(set(tr_ids))
    if not ids:
        return {}
    rows = (
        await session.execute(
            text(
                "SELECT task_result_id, key_hashes, scores FROM task_result_vectors "
                "WHERE task_result_id = ANY(:ids)"
            ),
            {"ids": ids},
        )
    ).all()
    return {
        int(r[0]): (np.frombuffer(r[1], "<i8").astype(np.int64), np.frombuffer(r[2], "<f8").copy())
        for r in rows
    }


async def load_metric_vectors(
    session: AsyncSession, trs: Iterable[TaskResultRow], metric_key: str
) -> dict[int, Vector]:
    """Vectors of a non-primary metric built from instance_results.metrics (cached in process)."""
    out: dict[int, Vector] = {}
    todo: list[int] = []
    for tr in trs:
        key = (int(tr["id"]), tr["updated_at"].isoformat(), metric_key)
        hit = _cache_get(key)
        if hit is not None:
            out[int(tr["id"])] = hit
        else:
            todo.append(int(tr["id"]))
    if todo:
        rows = (
            await session.execute(
                text(
                    "SELECT task_result_id, key_hash, (metrics ->> :k)::float8 "
                    "FROM instance_results WHERE task_result_id = ANY(:ids) "
                    "ORDER BY task_result_id, key_hash"
                ),
                {"k": metric_key, "ids": todo},
            )
        ).all()
        grouped: dict[int, tuple[list[int], list[float]]] = {i: ([], []) for i in todo}
        for tid, kh, val in rows:
            grouped[int(tid)][0].append(int(kh))
            grouped[int(tid)][1].append(math.nan if val is None else float(val))
        updated = {int(tr["id"]): tr["updated_at"].isoformat() for tr in trs}
        for tid, (ks, vs) in grouped.items():
            vec = (np.asarray(ks, dtype=np.int64), np.asarray(vs, dtype=np.float64))
            out[tid] = vec
            _cache_put((tid, updated[tid], metric_key), vec)
    return out


def meta_for(tr: TaskResultRow, metric_key: str | None) -> dict[str, Any]:
    if metric_key is None:
        return {}
    meta = (tr["metric_meta"] or {}).get(metric_key)
    if meta:
        return meta
    if metric_key == tr["primary_metric"]:
        return {
            "higher_is_better": tr["higher_is_better"],
            "display_format": tr["display_format"],
            "unit": None,
            "kind": tr["metric_kind"] or "unbounded",
        }
    return {}


class SideLoader:
    """Builds Sides for (task result, metric key) pairs with batched vector loading."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.requests: list[tuple[TaskResultRow, str]] = []
        self.primary: dict[int, Vector] = {}
        self.by_metric: dict[str, dict[int, Vector]] = {}

    def want(self, tr: TaskResultRow | None, metric_key: str | None) -> None:
        if tr is not None and metric_key is not None:
            self.requests.append((tr, metric_key))

    async def load(self) -> None:
        primary_ids = [tr["id"] for tr, key in self.requests if key == tr["primary_metric"]]
        self.primary = await load_primary_vectors(self.session, primary_ids)
        other: dict[str, list[TaskResultRow]] = {}
        for tr, key in self.requests:
            if key != tr["primary_metric"]:
                other.setdefault(key, []).append(tr)
        for key, trs in other.items():
            unique = {tr["id"]: tr for tr in trs}
            self.by_metric[key] = await load_metric_vectors(self.session, unique.values(), key)

    def side(self, tr: TaskResultRow | None, metric_key: str | None) -> Side | None:
        """None when the task result is missing or lacks the metric."""
        if tr is None or metric_key is None:
            return None
        meta = meta_for(tr, metric_key)
        if metric_key == tr["primary_metric"]:
            vec = self.primary.get(tr["id"])
            return Side(
                task_name=tr["task_name"],
                task_hash=tr["task_hash"],
                score=tr["score"],
                stderr=tr["stderr"],
                score_is_mean=bool(tr["score_is_mean"]),
                instance_scale=float(tr["instance_scale"] or 1.0),
                keys=vec[0] if vec else None,
                values=vec[1] if vec else None,
                kind=tr["metric_kind"] or meta.get("kind"),
                higher_is_better=tr["higher_is_better"],
                task_result_id=tr["id"],
            )
        metrics = tr["metrics"] or {}
        if metric_key not in metrics:
            return None
        score = metrics[metric_key]
        vec = self.by_metric.get(metric_key, {}).get(tr["id"])
        is_mean, scale, stderr = False, 1.0, None
        if vec is not None:
            vals = vec[1][np.isfinite(vec[1])]
            if vals.size and score is not None:
                found = detect_scale(float(vals.mean()), score)
                if found is not None:
                    is_mean, scale = True, found
                    if vals.size >= 2:
                        stderr = float(scale * vals.std(ddof=1) / math.sqrt(vals.size))
            if not vals.size:
                vec = None
        return Side(
            task_name=tr["task_name"],
            task_hash=tr["task_hash"],
            score=score,
            stderr=stderr,
            score_is_mean=is_mean,
            instance_scale=scale,
            keys=vec[0] if vec else None,
            values=vec[1] if vec else None,
            kind=meta.get("kind"),
            higher_is_better=meta.get("higher_is_better"),
            task_result_id=tr["id"],
        )
