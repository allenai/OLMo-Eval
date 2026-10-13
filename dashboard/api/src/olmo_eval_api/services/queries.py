"""Row loading and conversion shared by the read endpoints."""

from __future__ import annotations

import base64
import shlex
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timedelta
from typing import Any

import orjson
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from olmo_eval_api.errors import bad_request, not_found
from olmo_eval_api.schemas import api as a
from olmo_eval_api.services.common import now_utc
from olmo_eval_api.services.links import run_links

STALE_AFTER = timedelta(hours=24)
DEFAULT_LIMIT = 50
MAX_LIMIT = 500

# Runs that list and aggregate endpoints show (spec 4.1).
VISIBLE = "(r.upload_state = 'complete' OR r.status = 'running')"

# Whether a task result (alias tr) ran with an instance limit, such as a smoke run.
LIMITED = (
    "EXISTS (SELECT 1 FROM task_variants lv WHERE lv.task_name = tr.task_name "
    "AND lv.task_hash = tr.task_hash AND lv.task_limit IS NOT NULL)"
)
# Picks a model's representative task result: one without an error, then one without an
# instance limit, then the most recent run.
PREFERRED_ORDER = f"(tr.error IS NOT NULL), {LIMITED}, tr.run_created_at DESC, tr.id DESC"

MODEL_COLUMNS = """
    m.model_id AS m_model_id, m.name AS m_name, m.model_hash AS m_model_hash,
    m.series AS m_series, m.series_label AS m_series_label, m.family AS m_family,
    m.step AS m_step, m.tokens_seen AS m_tokens_seen, m.path AS m_path,
    m.revision AS m_revision, m.provider_kind AS m_provider_kind,
    m.provider_config AS m_provider_config, m.settings_hash AS m_settings_hash
"""
RUN_COLUMNS = (
    """
    r.run_id, r.launch_id, r.experiment_name, r.experiment_group, r.status, r.upload_state,
    r.author, r.uploaded_by, r.tags, r.created_at, r.updated_at, r.started_at, r.finished_at,
    r.duration_seconds, r.num_tasks, r.num_failed_tasks, r.num_instances, r.headline,
    r.beaker_experiment_id, r.beaker_result_dataset_id, r.beaker_workspace, r.beaker,
    r.git_repo, r.git_commit, r.git_branch, r.gcs_prefix,
"""
    + MODEL_COLUMNS
)
RUN_FROM = "runs r JOIN models m ON m.model_id = r.model_id"


# ---------------------------------------------------------------------------
# Cursors
# ---------------------------------------------------------------------------


def encode_cursor(values: Sequence[Any]) -> str:
    """Opaque cursor: base64url JSON of the last row's sort key and primary key."""
    tagged = [{"dt": v.isoformat()} if isinstance(v, datetime) else v for v in values]
    raw = orjson.dumps(tagged)
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def decode_cursor(cursor: str | None) -> list[Any] | None:
    if not cursor:
        return None
    try:
        padded = cursor + "=" * (-len(cursor) % 4)
        values = orjson.loads(base64.urlsafe_b64decode(padded))
    except Exception as exc:
        raise bad_request("invalid cursor") from exc
    if not isinstance(values, list):
        raise bad_request("invalid cursor")
    try:
        return [
            datetime.fromisoformat(v["dt"]) if isinstance(v, dict) and "dt" in v else v
            for v in values
        ]
    except (TypeError, ValueError) as exc:
        raise bad_request("invalid cursor") from exc


def decode_offset_cursor(cursor: str | None) -> int:
    """Row offset from a cursor made by ``encode_cursor([offset])``; 0 when absent."""
    values = decode_cursor(cursor)
    if not values:
        return 0
    start = values[0]
    if isinstance(start, bool) or not isinstance(start, int) or start < 0:
        raise bad_request("invalid cursor")
    return start


def decode_name_cursor(cursor: str | None) -> str | None:
    """Last name from a cursor made by ``encode_cursor([name])``."""
    values = decode_cursor(cursor)
    if not values:
        return None
    if not isinstance(values[0], str):
        raise bad_request("invalid cursor")
    return values[0]


def clamp_limit(limit: int | None, default: int = DEFAULT_LIMIT, maximum: int = MAX_LIMIT) -> int:
    if limit is None:
        return default
    if limit < 1 or limit > maximum:
        raise bad_request(f"limit must be between 1 and {maximum}")
    return limit


def keyset(
    columns: Sequence[tuple[str, bool]], cursor: list[Any] | None, prefix: str = "k"
) -> tuple[str, dict[str, Any]]:
    """WHERE fragment for keyset pagination with NULLS LAST on every column.

    ``columns`` is [(sql_expr, descending)]; the last one must be unique and non-null. The
    cursor holds one value per column (None for nulls). ORDER BY must be built with
    ``order_by(columns)``.
    """
    if cursor is None:
        return "TRUE", {}
    if len(cursor) != len(columns):
        raise bad_request("invalid cursor")
    params: dict[str, Any] = {}
    ors = []
    for i, (expr, desc) in enumerate(columns):
        eqs = []
        for j in range(i):
            e, _ = columns[j]
            v = cursor[j]
            if v is None:
                eqs.append(f"({e}) IS NULL")
            else:
                params[f"{prefix}{j}"] = v
                eqs.append(f"({e}) = :{prefix}{j}")
        v = cursor[i]
        if v is None:
            # Past a null there are only more nulls (NULLS LAST); nothing sorts after it here.
            continue
        params[f"{prefix}{i}"] = v
        op = "<" if desc else ">"
        cond = f"(({expr}) {op} :{prefix}{i} OR ({expr}) IS NULL)"
        ors.append("(" + " AND ".join([*eqs, cond]) + ")")
    return ("(" + " OR ".join(ors) + ")") if ors else "FALSE", params


def order_by(columns: Sequence[tuple[str, bool]]) -> str:
    return ", ".join(f"{e} {'DESC' if d else 'ASC'} NULLS LAST" for e, d in columns)


# ---------------------------------------------------------------------------
# Converters
# ---------------------------------------------------------------------------


def model_ref(row: Mapping[Any, Any], prefix: str = "m_") -> a.ModelRef:
    return a.ModelRef(
        model_id=row[f"{prefix}model_id"],
        name=row[f"{prefix}name"],
        model_hash=row[f"{prefix}model_hash"],
        series=row[f"{prefix}series"],
        series_label=row[f"{prefix}series_label"],
        family=row[f"{prefix}family"],
        step=row[f"{prefix}step"],
        tokens_seen=row[f"{prefix}tokens_seen"],
    )


def model_detail(row: Mapping[Any, Any], prefix: str = "m_") -> a.ModelDetail:
    ref = model_ref(row, prefix)
    return a.ModelDetail(
        **ref.model_dump(),
        path=row[f"{prefix}path"],
        revision=row[f"{prefix}revision"],
        provider_kind=row[f"{prefix}provider_kind"],
        provider_config=row[f"{prefix}provider_config"] or {},
        settings_hash=row[f"{prefix}settings_hash"],
    )


def is_stale(row: Mapping[Any, Any]) -> bool:
    return row["status"] == "running" and row["updated_at"] < now_utc() - STALE_AFTER


def run_summary_fields(row: Mapping[Any, Any]) -> dict[str, Any]:
    return {
        "run_id": row["run_id"],
        "launch_id": row["launch_id"],
        "experiment_name": row["experiment_name"],
        "experiment_group": row["experiment_group"],
        "status": row["status"],
        "upload_state": row["upload_state"],
        "stale": is_stale(row),
        "model": model_ref(row),
        "author": row["author"],
        "uploaded_by": row["uploaded_by"],
        "tags": list(row["tags"] or []),
        "created_at": row["created_at"],
        "started_at": row["started_at"],
        "finished_at": row["finished_at"],
        "duration_seconds": row["duration_seconds"],
        "num_tasks": row["num_tasks"],
        "num_failed_tasks": row["num_failed_tasks"],
        "num_instances": row["num_instances"],
        "headline": row["headline"],
        "beaker_experiment_id": row["beaker_experiment_id"],
        "beaker_workspace": row["beaker_workspace"],
        "git_commit": row["git_commit"],
        "git_branch": row["git_branch"],
        "gcs_prefix": row["gcs_prefix"],
        "links": run_links(row),
    }


def run_summary(row: Mapping[Any, Any]) -> a.RunSummary:
    return a.RunSummary(**run_summary_fields(row))


async def fetch_run_rows(
    session: AsyncSession,
    where: str = "TRUE",
    params: Mapping[Any, Any] | None = None,
    order: str = "r.created_at DESC, r.run_id DESC",
    limit: int | None = None,
    extra_columns: str = "",
) -> list[Mapping[Any, Any]]:
    sql = f"SELECT {RUN_COLUMNS}{extra_columns} FROM {RUN_FROM} WHERE {where} ORDER BY {order}"
    if limit is not None:
        sql += f" LIMIT {int(limit)}"
    return list((await session.execute(text(sql), dict(params or {}))).mappings().all())


async def fetch_runs(session: AsyncSession, run_ids: Iterable[str]) -> dict[str, Mapping[Any, Any]]:
    ids = list(dict.fromkeys(run_ids))
    if not ids:
        return {}
    rows = await fetch_run_rows(session, "r.run_id = ANY(:ids)", {"ids": ids})
    return {row["run_id"]: row for row in rows}


async def fetch_run_detail_row(session: AsyncSession, run_id: str) -> Mapping[Any, Any]:
    extra = """,
        r.notes, r.olmo_eval_version, r.git_dirty, r.beaker_job_id, r.environment, r.argv,
        r.task_specs, r.output_dir, r.harness_config, r.provider_init_seconds, r.errors, r.client,
        r.startup_seconds, r.processing_started_at, r.processing_seconds
    """
    rows = await fetch_run_rows(session, "r.run_id = :rid", {"rid": run_id}, extra_columns=extra)
    if not rows:
        raise not_found(f"Run {run_id} not found")
    return rows[0]


def reproduce_command(argv: Sequence[str] | None) -> str | None:
    return shlex.join(argv) if argv else None


def username(email: str | None) -> str | None:
    return email.split("@", 1)[0] if email else None


async def fetch_models(
    session: AsyncSession, model_ids: Iterable[str]
) -> dict[str, Mapping[Any, Any]]:
    ids = list(dict.fromkeys(model_ids))
    if not ids:
        return {}
    sql = f"SELECT {MODEL_COLUMNS} FROM models m WHERE m.model_id = ANY(:ids)"
    rows = (await session.execute(text(sql), {"ids": ids})).mappings().all()
    return {row["m_model_id"]: row for row in rows}


def metric_meta_or_none(meta: Mapping[Any, Any] | None) -> a.MetricMeta | None:
    if not meta:
        return None
    return a.MetricMeta(
        higher_is_better=meta.get("higher_is_better"),
        display_format=meta.get("display_format") or "raw",
        unit=meta.get("unit"),
        kind=meta.get("kind") or "unbounded",
    )
