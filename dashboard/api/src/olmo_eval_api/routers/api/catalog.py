"""Models, tasks, suites, leaderboards and score distributions."""

from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Annotated, Any

import numpy as np
from fastapi import APIRouter, Query
from sqlalchemy import text

from olmo_eval_api.errors import bad_request, not_found
from olmo_eval_api.schemas import api as a
from olmo_eval_api.services.cache import cache_key, get_cached, put_cached
from olmo_eval_api.services.common import latest_by_task
from olmo_eval_api.services.filters import like_contains
from olmo_eval_api.services.queries import (
    LIMITED,
    MODEL_COLUMNS,
    PREFERRED_ORDER,
    VISIBLE,
    clamp_limit,
    decode_cursor,
    decode_name_cursor,
    decode_offset_cursor,
    encode_cursor,
    fetch_models,
    fetch_runs,
    keyset,
    metric_meta_or_none,
    model_ref,
    order_by,
    run_summary,
)
from olmo_eval_api.services.read_suites import load_defs, score_suite, tree
from olmo_eval_api.services.runtime import (
    GPU_COUNT_SQL,
    GPU_TYPE_SQL,
    runtime_stats,
    runtimes_for,
)
from olmo_eval_api.services.subjects import (
    TR_COLUMNS,
    meta_for,
    models_task_results,
    resolve_subjects,
)
from olmo_eval_api.services.suites import SuiteNode, leaves
from olmo_eval_api.stats.mde import z

from .deps import SessionDep, UserDep

router = APIRouter()


def _csv(value: str | None) -> list[str]:
    return [v.strip() for v in (value or "").split(",") if v.strip()]


def _sort(sort: str, allowed: Mapping[str, tuple[str, bool]]) -> tuple[str, bool]:
    if sort not in allowed:
        raise bad_request(f"invalid sort {sort!r}; use one of {', '.join(allowed)}")
    return allowed[sort]


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

MODEL_SORTS = {
    "-last_run_at": ("last_run_at", True),
    "series": ("series", False),
    "-n_runs": ("n_runs", True),
}


@router.get("/models", response_model=a.ModelsListResponse)
async def list_models(
    session: SessionDep,
    user: UserDep,
    q: str | None = None,
    family: Annotated[list[str], Query()] = [],  # noqa: B006
    sort: str = "-last_run_at",
    cursor: str | None = None,
    limit: int | None = None,
) -> a.ModelsListResponse:
    limit = clamp_limit(limit)
    expr, desc = _sort(sort, MODEL_SORTS)
    where = [VISIBLE]
    params: dict[str, Any] = {}
    if q:
        for i, tok in enumerate(q.split()):
            params[f"q{i}"] = like_contains(tok)
            where.append(f"(m.series ILIKE :q{i} OR m.name ILIKE :q{i})")
    if family:
        where.append("m.family = ANY(:fam)")
        params["fam"] = family
    base = f"""
        WITH s AS (
            SELECT m.series, min(m.series_label) AS series_label,
                   mode() WITHIN GROUP (ORDER BY m.family) AS family,
                   count(DISTINCT m.model_id) AS n_checkpoints, count(r.run_id) AS n_runs,
                   max(m.step) AS latest_step, max(r.created_at) AS last_run_at
            FROM models m JOIN runs r ON r.model_id = m.model_id
            WHERE {" AND ".join(where)}
            GROUP BY m.series
        )
    """
    keys = [(expr, desc), ("series", desc)]
    cond, cparams = keyset(keys, decode_cursor(cursor))
    rows = (
        (
            await session.execute(
                text(
                    f"{base} SELECT * FROM s WHERE {cond} ORDER BY {order_by(keys)} "
                    f"LIMIT {limit + 1}"
                ),
                {**params, **cparams},
            )
        )
        .mappings()
        .all()
    )
    total = (await session.execute(text(f"{base} SELECT count(*) FROM s"), params)).scalar_one()
    next_cursor = None
    if len(rows) > limit:
        rows = rows[:limit]
        next_cursor = encode_cursor([rows[-1][expr], rows[-1]["series"]])
    series = [r["series"] for r in rows]
    latest: dict[str, str] = {}
    variants: dict[str, list[a.VariantCount]] = {}
    if series:
        for r in await session.execute(
            text(
                f"SELECT DISTINCT ON (m.series) m.series, r.run_id FROM runs r "
                f"JOIN models m ON m.model_id = r.model_id WHERE m.series = ANY(:s) AND {VISIBLE} "
                "ORDER BY m.series, r.created_at DESC"
            ),
            {"s": series},
        ):
            latest[r[0]] = r[1]
        for r in await session.execute(
            text(
                f"SELECT m.series, m.settings_hash, count(*) FROM runs r JOIN models m "
                f"ON m.model_id = r.model_id WHERE m.series = ANY(:s) AND {VISIBLE} "
                "GROUP BY 1, 2 ORDER BY 3 DESC"
            ),
            {"s": series},
        ):
            variants.setdefault(r[0], []).append(
                a.VariantCount(settings_hash=r[1], n_runs=int(r[2]))
            )
    runs = await fetch_runs(session, latest.values())
    items = [
        a.ModelSeriesRow(
            series=r["series"],
            series_label=r["series_label"],
            family=r["family"],
            n_checkpoints=int(r["n_checkpoints"]),
            n_runs=int(r["n_runs"]),
            latest_step=r["latest_step"],
            last_run_at=r["last_run_at"],
            latest_run=run_summary(runs[latest[r["series"]]]) if r["series"] in latest else None,
            variants=variants.get(r["series"], []),
        )
        for r in rows
    ]
    return a.ModelsListResponse(items=items, next_cursor=next_cursor, total=int(total))


def _differs(config: Mapping[Any, Any], reference: Mapping[Any, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in sorted(set(config) | set(reference)):
        if key in ("model", "alias", "revision", "tokenizer"):
            continue
        if config.get(key) != reference.get(key):
            out[key] = config.get(key)
    return out


@router.get("/models/series", response_model=a.ModelSeriesDetailResponse)
async def model_series(
    series: str, session: SessionDep, user: UserDep
) -> a.ModelSeriesDetailResponse:
    rows = (
        (
            await session.execute(
                text(
                    f"""
                    SELECT {MODEL_COLUMNS},
                           array_remove(array_agg(r.run_id ORDER BY r.created_at DESC), NULL)
                               AS run_ids,
                           max(r.created_at) AS latest_run_at
                    FROM models m
                    LEFT JOIN runs r ON r.model_id = m.model_id AND {VISIBLE}
                    WHERE m.series = :s
                    GROUP BY m.model_id
                    ORDER BY m.step ASC NULLS LAST, m.name
                    """
                ),
                {"s": series},
            )
        )
        .mappings()
        .all()
    )
    if not rows:
        raise not_found(f"Model series {series!r} not found")
    counts: dict[str, int] = {}
    configs: dict[str, Mapping[Any, Any]] = {}
    for r in rows:
        counts[r["m_settings_hash"]] = counts.get(r["m_settings_hash"], 0) + len(r["run_ids"])
        configs.setdefault(r["m_settings_hash"], r["m_provider_config"] or {})
    common = max(counts, key=lambda h: counts[h])
    families = [r["m_family"] for r in rows if r["m_family"]]
    return a.ModelSeriesDetailResponse(
        series=series,
        series_label=rows[0]["m_series_label"],
        family=statistics.mode(families) if families else None,
        n_runs=sum(counts.values()),
        checkpoints=[
            a.CheckpointRow(
                model=model_ref(r),
                settings_hash=r["m_settings_hash"],
                run_ids=list(r["run_ids"]),
                latest_run_at=r["latest_run_at"],
            )
            for r in rows
        ],
        variants=[
            a.VariantInfo(
                settings_hash=h,
                n_runs=n,
                differs={} if h == common else _differs(configs[h], configs[common]),
            )
            for h, n in sorted(counts.items(), key=lambda kv: -kv[1])
        ],
    )


def _panel_score(
    node: SuiteNode | None, task: str | None, trs: Mapping[str, Mapping[Any, Any]], metric: str
) -> tuple[float | None, float | None, int | None, Mapping[Any, Any] | None]:
    """(score, stderr, n, task result) for a task or suite panel from merged task results."""
    if node is not None:
        result = score_suite(node, trs)
        return result.score, result.stderr, result.n_instances or None, None
    tr = trs.get(task or "")
    if tr is None:
        return None, None, None, None
    if metric == "primary" or metric == tr["primary_metric"]:
        return tr["score"], tr["stderr"], tr["num_instances"], tr
    return (tr["metrics"] or {}).get(metric), None, tr["num_instances"], tr


@router.get("/models/progression", response_model=a.ProgressionResponse)
async def model_progression(
    session: SessionDep,
    user: UserDep,
    series: str,
    settings_hash: str | None = None,
    scope: str | None = None,
    tasks: str | None = None,
    metric: str = "primary",
    merge: Annotated[str, Query(pattern="^(latest|all)$")] = "latest",
    references: str | None = None,
) -> a.ProgressionResponse:
    params: dict[str, Any] = {"s": series}
    where = "m.series = :s"
    if settings_hash:
        where += " AND m.settings_hash = :h"
        params["h"] = settings_hash
    models = (
        (
            await session.execute(
                text(
                    f"SELECT {MODEL_COLUMNS} FROM models m WHERE {where} ORDER BY m.step NULLS LAST"
                ),
                params,
            )
        )
        .mappings()
        .all()
    )
    if not models:
        raise not_found(f"Model series {series!r} not found")
    defs = await load_defs(session)
    panels_spec: list[tuple[str, str, SuiteNode | None]] = []  # (kind, name, node)
    if scope:
        kind, _, name = scope.partition(":")
        if kind == "suite":
            if name not in defs:
                raise bad_request(f"unknown suite {name!r}")
            node = tree(name, defs)
            panels_spec.append(("suite", name, node))
            panels_spec.extend(("task", t, None) for t in leaves(node)[:24])
        elif kind == "task" and name:
            panels_spec.append(("task", name, None))
        else:
            raise bad_request("scope must be suite:<name> or task:<name>")
    for t in _csv(tasks):
        panels_spec.append(("task", t, None))
    model_ids = [m["m_model_id"] for m in models]
    if not panels_spec:
        top = (
            (
                await session.execute(
                    text(
                        """
                    SELECT s.suite_name FROM suite_results s
                    WHERE s.run_id = (SELECT r.run_id FROM runs r WHERE r.model_id = ANY(:ids)
                                      AND r.upload_state = 'complete'
                                      ORDER BY r.created_at DESC LIMIT 1)
                      AND s.parent_suite IS NULL ORDER BY s.id
                    """
                    ),
                    {"ids": model_ids},
                )
            )
            .scalars()
            .all()
        )
        for name in top:
            if name in defs:
                panels_spec.append(("suite", name, tree(name, defs)))
        if not panels_spec:
            frequent = (
                (
                    await session.execute(
                        text(
                            "SELECT task_name FROM task_results WHERE model_id = ANY(:ids) "
                            "AND finalized_at IS NOT NULL GROUP BY task_name "
                            "ORDER BY count(*) DESC, task_name LIMIT 12"
                        ),
                        {"ids": model_ids},
                    )
                )
                .scalars()
                .all()
            )
            panels_spec.extend(("task", t, None) for t in frequent)

    merged: dict[str, dict[str, Mapping[Any, Any]]] = {}
    all_trs: list[Mapping[Any, Any]] = []
    if merge == "latest":
        merged = await models_task_results(session, model_ids)
    else:
        all_trs = list(
            (
                await session.execute(
                    text(
                        f"SELECT {TR_COLUMNS} FROM task_results tr WHERE tr.model_id = ANY(:ids) "
                        "AND tr.finalized_at IS NOT NULL ORDER BY tr.run_created_at"
                    ),
                    {"ids": model_ids},
                )
            )
            .mappings()
            .all()
        )
    model_by_id = {m["m_model_id"]: m for m in models}
    ref_subjects, _ = await resolve_subjects(session, _csv(references))
    panels = []
    for kind, name, node in panels_spec:
        points: list[a.ProgressionPoint] = []
        meta = None
        if merge == "latest":
            for mid in model_ids:
                trs = merged[mid]
                score, stderr, n, tr = _panel_score(node, name, trs, metric)
                if score is None:
                    continue
                if tr is not None and meta is None:
                    meta = metric_meta_or_none(
                        meta_for(tr, tr["primary_metric"] if metric == "primary" else metric)
                    )
                contributing = [trs[t] for t in (leaves(node) if node else [name]) if t in trs]
                newest = max(contributing, key=lambda x: x["run_created_at"])
                m = model_by_id[mid]
                points.append(
                    a.ProgressionPoint(
                        model_id=mid,
                        run_id=newest["run_id"],
                        task_result_id=tr["id"] if tr is not None else None,
                        step=m["m_step"],
                        tokens_seen=m["m_tokens_seen"],
                        date=newest["run_created_at"],
                        score=score,
                        stderr=stderr,
                        n=n,
                    )
                )
        else:
            by_run: dict[str, list[Mapping[Any, Any]]] = {}
            for tr in all_trs:
                by_run.setdefault(tr["run_id"], []).append(tr)
            for run_trs in by_run.values():
                trs = latest_by_task(run_trs)
                score, stderr, n, tr = _panel_score(node, name, trs, metric)
                if score is None:
                    continue
                if tr is not None and meta is None:
                    meta = metric_meta_or_none(
                        meta_for(tr, tr["primary_metric"] if metric == "primary" else metric)
                    )
                any_tr = next(iter(trs.values()))
                m = model_by_id[any_tr["model_id"]]
                points.append(
                    a.ProgressionPoint(
                        model_id=any_tr["model_id"],
                        run_id=any_tr["run_id"],
                        task_result_id=tr["id"] if tr is not None else None,
                        step=m["m_step"],
                        tokens_seen=m["m_tokens_seen"],
                        date=any_tr["run_created_at"],
                        score=score,
                        stderr=stderr,
                        n=n,
                    )
                )
            points.sort(key=lambda p: (p.step is None, p.step or 0, p.date))
        if node is not None and meta is None:
            sample = next((trs for trs in merged.values() if trs), {})
            result = score_suite(node, sample) if sample else None
            meta = a.MetricMeta(
                higher_is_better=result.higher_is_better if result else None,
                display_format=result.display_format if result else "raw",
                unit=None,
                kind="bounded" if result and result.display_format == "percent" else "unbounded",
            )
        refs = []
        for s in ref_subjects:
            score, stderr, _, _ = _panel_score(node, name, s.trs, metric)
            refs.append(
                a.ProgressionReference(subject=s.key, label=s.label, score=score, stderr=stderr)
            )
        panels.append(
            a.ProgressionPanel(
                key=f"{kind}:{name}",
                kind="suite" if kind == "suite" else "task",
                name=name,
                meta=meta,
                points=points,
                references=refs,
            )
        )
    return a.ProgressionResponse(
        series=series, settings_hash=settings_hash, metric=metric, panels=panels
    )


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

TASK_SORTS = {
    "-n_runs": ("n_runs", True),
    "task_name": ("task_name", False),
    "-last_run_at": ("last_run_at", True),
}


@router.get("/tasks", response_model=a.TasksListResponse)
async def list_tasks(
    session: SessionDep,
    user: UserDep,
    q: str | None = None,
    suite: str | None = None,
    sort: str = "-n_runs",
    cursor: str | None = None,
    limit: int | None = None,
) -> a.TasksListResponse:
    limit = clamp_limit(limit)
    expr, desc = _sort(sort, TASK_SORTS)
    where = [VISIBLE]
    params: dict[str, Any] = {}
    if q:
        for i, tok in enumerate(q.split()):
            params[f"q{i}"] = like_contains(tok)
            where.append(f"tr.task_name ILIKE :q{i}")
    if suite:
        defs = await load_defs(session)
        members = leaves(tree(suite, defs)) if suite in defs else []
        where.append(
            "(tr.task_name = ANY(:members) OR EXISTS (SELECT 1 FROM task_variants v "
            "WHERE v.task_name = tr.task_name AND :suite = ANY(v.suites)))"
        )
        params.update(members=members, suite=suite)
    base = f"""
        WITH t AS (
            SELECT tr.task_name, count(DISTINCT tr.task_hash) AS n_variants,
                   count(DISTINCT tr.run_id) AS n_runs, count(DISTINCT tr.model_id) AS n_models,
                   max(tr.run_created_at) AS last_run_at
            FROM task_results tr JOIN runs r ON r.run_id = tr.run_id
            WHERE {" AND ".join(where)}
            GROUP BY tr.task_name
        )
    """
    keys = [(expr, desc), ("task_name", desc)] if expr != "task_name" else [(expr, desc)]
    cond, cparams = keyset(keys, decode_cursor(cursor))
    rows = (
        (
            await session.execute(
                text(
                    f"{base} SELECT * FROM t WHERE {cond} "
                    f"ORDER BY {order_by(keys)} LIMIT {limit + 1}"
                ),
                {**params, **cparams},
            )
        )
        .mappings()
        .all()
    )
    total = (await session.execute(text(f"{base} SELECT count(*) FROM t"), params)).scalar_one()
    next_cursor = None
    if len(rows) > limit:
        rows = rows[:limit]
        last = rows[-1]
        next_cursor = encode_cursor(
            [last[expr], last["task_name"]] if len(keys) == 2 else [last["task_name"]]
        )
    names = [r["task_name"] for r in rows]
    info: dict[str, Mapping[Any, Any]] = {}
    if names:
        info = {
            r["task_name"]: r
            for r in (
                await session.execute(
                    text(
                        """
                        SELECT v.task_name,
                               (array_agg(v.base_task ORDER BY v.last_seen_at DESC))[1]
                                   AS base_task,
                               (array_agg(v.primary_metric ORDER BY v.last_seen_at DESC))[1]
                                   AS primary_metric,
                               ARRAY(SELECT DISTINCT x FROM task_variants v2, unnest(v2.suites) x
                                     WHERE v2.task_name = v.task_name ORDER BY x) AS suites
                        FROM task_variants v WHERE v.task_name = ANY(:n)
                        GROUP BY v.task_name
                        """
                    ),
                    {"n": names},
                )
            ).mappings()
        }
    items = [
        a.TaskRow(
            task_name=r["task_name"],
            base_task=(info.get(r["task_name"]) or {}).get("base_task"),
            n_variants=int(r["n_variants"]),
            n_runs=int(r["n_runs"]),
            n_models=int(r["n_models"]),
            suites=list((info.get(r["task_name"]) or {}).get("suites") or []),
            primary_metric=(info.get(r["task_name"]) or {}).get("primary_metric"),
            last_run_at=r["last_run_at"],
        )
        for r in rows
    ]
    return a.TasksListResponse(items=items, next_cursor=next_cursor, total=int(total))


@router.get("/tasks/detail", response_model=a.TaskDetailResponse)
async def task_detail(task: str, session: SessionDep, user: UserDep) -> a.TaskDetailResponse:
    rows = (
        (
            await session.execute(
                text(
                    f"""
                    SELECT v.*, count(DISTINCT tr.run_id) FILTER (WHERE {VISIBLE}) AS n_runs,
                           percentile_disc(0.5) WITHIN GROUP (ORDER BY tr.instances_stored)
                               AS n_instances,
                           (array_agg(tr.metric_meta ORDER BY tr.updated_at DESC)
                               FILTER (WHERE tr.finalized_at IS NOT NULL))[1] AS latest_meta
                    FROM task_variants v
                    LEFT JOIN task_results tr ON tr.task_name = v.task_name
                        AND tr.task_hash = v.task_hash
                    LEFT JOIN runs r ON r.run_id = tr.run_id
                    WHERE v.task_name = :t
                    GROUP BY v.task_name, v.task_hash
                    ORDER BY n_runs DESC, v.last_seen_at DESC
                    """
                ),
                {"t": task},
            )
        )
        .mappings()
        .all()
    )
    if not rows:
        raise not_found(f"Task {task!r} not found")
    suites = sorted({s for r in rows for s in (r["suites"] or [])})
    return a.TaskDetailResponse(
        task_name=task,
        base_task=next((r["base_task"] for r in rows if r["base_task"]), None),
        suites=suites,
        variants=[
            a.TaskVariantInfo(
                task_hash=r["task_hash"],
                n_runs=int(r["n_runs"] or 0),
                primary_metric=r["primary_metric"],
                metric_meta=r["latest_meta"] or {},
                num_fewshot=r["num_fewshot"],
                limit=r["task_limit"],
                split=r["split"],
                n_instances=r["n_instances"],
                first_seen_at=r["first_seen_at"],
                last_seen_at=r["last_seen_at"],
                config=r["config"] or {},
            )
            for r in rows
        ],
    )


# ---------------------------------------------------------------------------
# Leaderboards
# ---------------------------------------------------------------------------


def _leaderboard_filters(
    family: list[str],
    group: str | None,
    user_filter: str | None,
    me: str,
    after: datetime | None,
    before: datetime | None,
    step_min: int | None,
    step_max: int | None,
) -> tuple[list[str], dict[str, Any]]:
    where = [VISIBLE, "tr.finalized_at IS NOT NULL"]
    params: dict[str, Any] = {}
    if family:
        where.append("m.family = ANY(:fam)")
        params["fam"] = family
    if group:
        where.append("r.experiment_group = :grp")
        params["grp"] = group
    if user_filter:
        who = me if user_filter == "me" else user_filter
        where.append("(r.author = :who OR split_part(r.uploaded_by, '@', 1) = :who)")
        params["who"] = who
    if after:
        where.append("r.created_at >= :after")
        params["after"] = after
    if before:
        where.append("r.created_at < :before")
        params["before"] = before
    if step_min is not None:
        where.append("m.step >= :smin")
        params["smin"] = step_min
    if step_max is not None:
        where.append("m.step <= :smax")
        params["smax"] = step_max
    return where, params


def _rank(
    entries: list[dict[str, Any]], higher_is_better: bool | None, alpha: float = 0.05
) -> list[dict[str, Any]]:
    sign = -1 if higher_is_better is not False else 1
    entries.sort(key=lambda e: (e["score"] is None, sign * (e["score"] or 0.0), str(e["date"])))
    zq = z(1 - alpha / 2)
    for e in entries:
        se = e.get("stderr")
        if e["score"] is not None and se is not None:
            e["ci_low"], e["ci_high"] = e["score"] - zq * se, e["score"] + zq * se
        else:
            e["ci_low"] = e["ci_high"] = None
    leader = entries[0] if entries else None
    for i, e in enumerate(entries):
        e["rank"] = i + 1
        if e is leader:
            e["tied_with_leader"] = e["score"] is not None
        elif leader and None not in (e["ci_low"], leader["ci_low"]):
            e["tied_with_leader"] = (
                e["ci_high"] >= leader["ci_low"] and e["ci_low"] <= leader["ci_high"]
            )
        else:
            e["tied_with_leader"] = False
    return entries


def _page(entries: list[dict[str, Any]], cursor: str | None, limit: int) -> tuple[list, str | None]:
    start = decode_offset_cursor(cursor)
    page = entries[start : start + limit]
    nxt = encode_cursor([start + limit]) if start + limit < len(entries) else None
    return page, nxt


LeaderboardFilters = dict[str, Any]


@router.get("/tasks/leaderboard", response_model=a.LeaderboardResponse)
async def task_leaderboard(
    session: SessionDep,
    user: UserDep,
    task: str,
    hash: str | None = None,  # noqa: A002
    metric: str = "primary",
    per_model: Annotated[str, Query(pattern="^(latest|all)$")] = "latest",
    family: Annotated[list[str], Query()] = [],  # noqa: B006
    group: str | None = None,
    user_filter: Annotated[str | None, Query(alias="user")] = None,
    after: datetime | None = None,
    before: datetime | None = None,
    step_min: int | None = None,
    step_max: int | None = None,
    gpu_type: str | None = None,
    cursor: str | None = None,
    limit: int | None = None,
) -> a.LeaderboardResponse:
    limit = clamp_limit(limit, default=100, maximum=1000)
    where, params = _leaderboard_filters(
        family, group, user_filter, user.username, after, before, step_min, step_max
    )
    where.append("tr.task_name = :task")
    params["task"] = task
    if hash:
        where.append("tr.task_hash = :hash")
        params["hash"] = hash
    if gpu_type:
        where.append(f"{GPU_TYPE_SQL} = :gpu")
        params["gpu"] = gpu_type
    distinct = "DISTINCT ON (tr.model_id)" if per_model == "latest" else ""
    order = f"tr.model_id, {PREFERRED_ORDER}" if per_model == "latest" else "tr.run_created_at DESC"
    rows = (
        (
            await session.execute(
                text(
                    f"""
                    SELECT {distinct} {TR_COLUMNS}, r.experiment_group, r.author,
                           {GPU_TYPE_SQL} AS gpu_type, {GPU_COUNT_SQL} AS gpu_count,
                           {MODEL_COLUMNS}
                    FROM task_results tr JOIN runs r ON r.run_id = tr.run_id
                    JOIN models m ON m.model_id = tr.model_id
                    WHERE {" AND ".join(where)} ORDER BY {order}
                    """
                ),
                params,
            )
        )
        .mappings()
        .all()
    )
    latest = max(rows, key=lambda r: r["updated_at"]) if rows else None
    metric_key = (latest["primary_metric"] if latest else None) if metric == "primary" else metric
    meta = meta_for(latest, metric_key) if latest else {}
    entries = []
    for r in rows:
        if metric == "primary" or metric == r["primary_metric"]:
            score, stderr = r["score"], r["stderr"]
        else:
            score, stderr = (r["metrics"] or {}).get(metric), None
        entries.append(
            {
                "subject": f"m:{r['model_id']}" if per_model == "latest" else f"r:{r['run_id']}",
                "model": model_ref(r),
                "run_id": r["run_id"],
                "task_result_id": r["id"],
                "task_hash": r["task_hash"],
                "score": score,
                "stderr": stderr,
                "n": r["num_instances"],
                "date": r["run_created_at"],
                "experiment_group": r["experiment_group"],
                "author": r["author"],
                "child_scores": None,
                "gpu_type": r["gpu_type"],
                "gpu_count": r["gpu_count"],
                "_row": r,
            }
        )
    ranked = _rank(entries, meta.get("higher_is_better"))
    page, nxt = _page(ranked, cursor, limit)
    # Runtime only for the page: it needs every task result of each run for the token shares.
    runtimes = await runtimes_for(session, [e["_row"] for e in page])
    for e in page:
        e["runtime"] = runtimes[e.pop("_row")["id"]]
    return a.LeaderboardResponse(
        kind="task",
        name=task,
        task_hash=hash,
        metric=metric,
        meta=metric_meta_or_none(meta),
        children=None,
        items=[a.LeaderboardRow(**e) for e in page],
        next_cursor=nxt,
        total=len(ranked),
    )


# ---------------------------------------------------------------------------
# Task runtime
# ---------------------------------------------------------------------------

# Most recent task results the runtime endpoint reads, and points it returns.
MAX_RUNTIME_RESULTS = 20_000
MAX_RUNTIME_POINTS = 2_000


@router.get("/tasks/{task_name:path}/runtime", response_model=a.TaskRuntimeResponse)
async def task_runtime(
    session: SessionDep,
    user: UserDep,
    task_name: str,
    hash: str | None = None,  # noqa: A002
    gpu_type: str | None = None,
    family: Annotated[list[str], Query()] = [],  # noqa: B006
    group: str | None = None,
    user_filter: Annotated[str | None, Query(alias="user")] = None,
) -> a.TaskRuntimeResponse:
    if hash is None:
        # The most-run variant, as on the task page (task_detail).
        hash = (  # noqa: A001
            await session.execute(
                text(
                    f"""
                    SELECT v.task_hash FROM task_variants v
                    LEFT JOIN task_results tr ON tr.task_name = v.task_name
                        AND tr.task_hash = v.task_hash
                    LEFT JOIN runs r ON r.run_id = tr.run_id
                    WHERE v.task_name = :t
                    GROUP BY v.task_name, v.task_hash
                    ORDER BY count(DISTINCT tr.run_id) FILTER (WHERE {VISIBLE}) DESC,
                             v.last_seen_at DESC
                    LIMIT 1
                    """
                ),
                {"t": task_name},
            )
        ).scalar()
        if hash is None:
            raise not_found(f"Task {task_name!r} not found")
    where, params = _leaderboard_filters(
        family, group, user_filter, user.username, None, None, None, None
    )
    where += ["tr.task_name = :task", "tr.task_hash = :hash"]
    params.update(task=task_name, hash=hash)
    gpu_rows = (
        await session.execute(
            text(
                f"""
                SELECT {GPU_TYPE_SQL} AS gpu_type, count(*) AS n
                FROM task_results tr JOIN runs r ON r.run_id = tr.run_id
                JOIN models m ON m.model_id = tr.model_id
                WHERE {" AND ".join(where)} AND {GPU_TYPE_SQL} IS NOT NULL
                GROUP BY 1 ORDER BY n DESC, gpu_type
                """
            ),
            params,
        )
    ).all()
    if gpu_type:
        where.append(f"{GPU_TYPE_SQL} = :gpu")
        params["gpu"] = gpu_type
    rows = (
        (
            await session.execute(
                text(
                    f"""
                    SELECT {TR_COLUMNS}, {GPU_TYPE_SQL} AS gpu_type,
                           {GPU_COUNT_SQL} AS gpu_count, {MODEL_COLUMNS}
                    FROM task_results tr JOIN runs r ON r.run_id = tr.run_id
                    JOIN models m ON m.model_id = tr.model_id
                    WHERE {" AND ".join(where)}
                    ORDER BY tr.run_created_at DESC, tr.id DESC
                    LIMIT {MAX_RUNTIME_RESULTS}
                    """
                ),
                params,
            )
        )
        .mappings()
        .all()
    )
    runtimes = await runtimes_for(session, rows)
    latest = max(rows, key=lambda r: r["updated_at"]) if rows else None
    meta = meta_for(latest, latest["primary_metric"]) if latest else {}

    points: list[a.TaskRuntimePoint] = []
    groups: dict[tuple[str, str | None, int | None], list[Mapping[Any, Any]]] = {}
    not_recorded = 0
    for r in rows:  # newest first
        runtime = runtimes[r["id"]]
        if runtime.basis == "not_recorded":
            not_recorded += 1
            continue
        groups.setdefault((r["model_id"], r["gpu_type"], r["gpu_count"]), []).append(r)
        if len(points) < MAX_RUNTIME_POINTS:
            points.append(
                a.TaskRuntimePoint(
                    run_id=r["run_id"],
                    task_result_id=r["id"],
                    model=model_ref(r),
                    date=r["run_created_at"],
                    gpu_type=r["gpu_type"],
                    gpu_count=r["gpu_count"],
                    score=r["score"],
                    n=r["num_instances"],
                    runtime=runtime,
                )
            )
    out_rows = []
    for (_, gpu, count), members in groups.items():
        rts = [runtimes[r["id"]] for r in members]
        out_rows.append(
            a.TaskRuntimeModelRow(
                model=model_ref(members[0]),
                gpu_type=gpu,
                gpu_count=count,
                runs=len(members),
                inference=runtime_stats([x.inference_seconds for x in rts]),
                with_startup=runtime_stats([x.with_startup_seconds for x in rts]),
                seconds_per_1k_instances=runtime_stats([x.seconds_per_1k_instances for x in rts]),
                latest_score=members[0]["score"],
            )
        )
    out_rows.sort(
        key=lambda x: (x.inference.median is None, x.inference.median or 0.0, x.model.name)
    )
    return a.TaskRuntimeResponse(
        task_name=task_name,
        task_hash=hash,
        meta=metric_meta_or_none(meta),
        gpu_types=[g for g, _ in gpu_rows],
        rows=out_rows,
        points=points,
        not_recorded=not_recorded,
    )


@router.get("/suites/leaderboard", response_model=a.LeaderboardResponse)
async def suite_leaderboard(
    session: SessionDep,
    user: UserDep,
    suite: str,
    per_model: Annotated[str, Query(pattern="^(latest|all)$")] = "latest",
    family: Annotated[list[str], Query()] = [],  # noqa: B006
    group: str | None = None,
    user_filter: Annotated[str | None, Query(alias="user")] = None,
    after: datetime | None = None,
    before: datetime | None = None,
    step_min: int | None = None,
    step_max: int | None = None,
    cursor: str | None = None,
    limit: int | None = None,
) -> a.LeaderboardResponse:
    limit = clamp_limit(limit, default=100, maximum=1000)
    defs = await load_defs(session)
    if suite not in defs:
        raise not_found(f"Suite {suite!r} not found")
    node = tree(suite, defs)
    names = leaves(node)
    where, params = _leaderboard_filters(
        family, group, user_filter, user.username, after, before, step_min, step_max
    )
    where.append("tr.task_name = ANY(:names)")
    params["names"] = names
    if per_model == "latest":
        sql = f"""
            SELECT DISTINCT ON (tr.model_id, tr.task_name) {TR_COLUMNS}
            FROM task_results tr JOIN runs r ON r.run_id = tr.run_id
            JOIN models m ON m.model_id = tr.model_id
            WHERE {" AND ".join(where)}
            ORDER BY tr.model_id, tr.task_name, {PREFERRED_ORDER}
        """
        group_key = "model_id"
    else:
        sql = f"""
            SELECT {TR_COLUMNS} FROM task_results tr JOIN runs r ON r.run_id = tr.run_id
            JOIN models m ON m.model_id = tr.model_id WHERE {" AND ".join(where)}
        """
        group_key = "run_id"
    rows = (await session.execute(text(sql), params)).mappings().all()
    grouped: dict[str, list[Mapping[Any, Any]]] = {}
    for r in rows:
        grouped.setdefault(r[group_key], []).append(r)
    models = await fetch_models(session, {r["model_id"] for r in rows})
    runs = await fetch_runs(session, {r["run_id"] for r in rows})
    entries = []
    hib_values = set()
    for gkey, trs_list in grouped.items():
        trs = latest_by_task(trs_list)
        result = score_suite(node, trs)
        if result.score is None:
            continue
        hib_values.add(result.higher_is_better)
        newest = max(trs.values(), key=lambda x: x["run_created_at"])
        child_scores: dict[str, float | None] = {}
        for child in node.children:
            if child.type == "task":
                tr = trs.get(child.name)
                child_scores[f"task:{child.name}"] = tr["score"] if tr is not None else None
            else:
                child_scores[f"suite:{child.name}"] = score_suite(child, trs).score
        run = runs[newest["run_id"]]
        entries.append(
            {
                "subject": f"m:{gkey}" if per_model == "latest" else f"r:{gkey}",
                "model": model_ref(models[newest["model_id"]]),
                "run_id": newest["run_id"],
                "task_result_id": None,
                "task_hash": None,
                "score": result.score,
                "stderr": result.stderr,
                "n": result.n_instances,
                "date": newest["run_created_at"],
                "experiment_group": run["experiment_group"],
                "author": run["author"],
                "child_scores": child_scores,
                "runtime": None,
                "gpu_type": None,
                "gpu_count": None,
            }
        )
    hib = hib_values.pop() if len(hib_values) == 1 else None
    ranked = _rank(entries, hib)
    page, nxt = _page(ranked, cursor, limit)
    return a.LeaderboardResponse(
        kind="suite",
        name=suite,
        task_hash=None,
        metric="primary",
        meta=None,
        children=[a.SuiteChild(type=c.type, name=c.name) for c in node.children],
        items=[a.LeaderboardRow(**e) for e in page],
        next_cursor=nxt,
        total=len(ranked),
    )


# ---------------------------------------------------------------------------
# Suites
# ---------------------------------------------------------------------------


@router.get("/suites", response_model=a.SuitesListResponse)
async def list_suites(
    session: SessionDep,
    user: UserDep,
    q: str | None = None,
    cursor: str | None = None,
    limit: int | None = None,
) -> a.SuitesListResponse:
    limit = clamp_limit(limit)
    defs = await load_defs(session)
    names = sorted(defs)
    if q:
        toks = q.lower().split()
        names = [n for n in names if all(t in n.lower() for t in toks)]
    stats = {
        r[0]: (int(r[1]), r[2])
        for r in await session.execute(
            text(
                f"SELECT s.suite_name, count(*), max(r.created_at) FROM suite_results s "
                f"JOIN runs r ON r.run_id = s.run_id WHERE {VISIBLE} GROUP BY 1"
            )
        )
    }
    start = 0
    after = decode_name_cursor(cursor)
    if after is not None:
        start = next((i for i, n in enumerate(names) if n > after), len(names))
    page = names[start : start + limit]
    nxt = encode_cursor([page[-1]]) if page and start + limit < len(names) else None
    items = []
    for name in page:
        agg, children, desc, _ = defs[name]
        n_runs, last = stats.get(name, (0, None))
        items.append(
            a.SuiteRow(
                suite_name=name,
                aggregation=agg,
                description=desc,
                n_children=len(children),
                n_tasks=len(leaves(tree(name, defs))),
                n_runs=n_runs,
                last_run_at=last,
            )
        )
    return a.SuitesListResponse(items=items, next_cursor=nxt, total=len(names))


def _tree_node(node: SuiteNode) -> a.SuiteTreeNode:
    return a.SuiteTreeNode(
        type=node.type,
        name=node.name,
        aggregation=node.aggregation,
        children=[_tree_node(c) for c in node.children],
    )


@router.get("/suites/detail", response_model=a.SuiteDetailResponse)
async def suite_detail(suite: str, session: SessionDep, user: UserDep) -> a.SuiteDetailResponse:
    defs = await load_defs(session)
    if suite not in defs:
        raise not_found(f"Suite {suite!r} not found")
    agg, _, desc, def_hash = defs[suite]
    seen, n_runs = (
        await session.execute(
            text(
                "SELECT (SELECT count(*) FROM suite_defs WHERE suite_name = :s), "
                "(SELECT count(*) FROM suite_results WHERE suite_name = :s)"
            ),
            {"s": suite},
        )
    ).one()
    return a.SuiteDetailResponse(
        suite_name=suite,
        aggregation=agg,
        description=desc,
        definition_hash=def_hash,
        definitions_seen=int(seen),
        tree=_tree_node(tree(suite, defs)),
        n_runs=int(n_runs),
    )


# ---------------------------------------------------------------------------
# Distributions
# ---------------------------------------------------------------------------


@router.post("/tasks/distributions", response_model=a.DistributionsResponse)
async def distributions(
    body: a.DistributionsRequest, session: SessionDep, user: UserDep
) -> a.DistributionsResponse:
    per_model = body.per_model or "latest"
    if per_model not in ("latest", "all"):
        raise bad_request("per_model must be latest or all")
    names = sorted({k.task_name for k in body.items})
    rows: Sequence[Mapping[Any, Any]] = []
    if names:
        rows = (
            (
                await session.execute(
                    text(
                        f"""
                        SELECT {TR_COLUMNS}, {MODEL_COLUMNS}, {LIMITED} AS is_limited
                        FROM task_results tr
                        JOIN runs r ON r.run_id = tr.run_id
                        JOIN models m ON m.model_id = tr.model_id
                        WHERE tr.task_name = ANY(:n) AND tr.finalized_at IS NOT NULL AND {VISIBLE}
                        ORDER BY tr.run_created_at DESC, tr.id DESC
                        """
                    ),
                    {"n": names},
                )
            )
            .mappings()
            .all()
        )
    key = cache_key("distributions", body.model_dump(mode="json"), rows)
    cached = await get_cached(session, key, a.DistributionsResponse)
    if cached is not None:
        # The key covers the task results, not model metadata, so refresh the top model.
        by_run = {r["run_id"]: r for r in rows}
        for item in cached.items:
            if item.top is not None and item.top.run_id in by_run:
                item.top.model = model_ref(by_run[item.top.run_id])
        return cached
    items = []
    for k in body.items:
        selected = [
            r
            for r in rows
            if r["task_name"] == k.task_name
            and (k.task_hash is None or r["task_hash"] == k.task_hash)
        ]
        if per_model == "latest":
            seen: set[str] = set()
            kept = []
            for r in sorted(selected, key=lambda r: (r["error"] is not None, r["is_limited"])):
                if r["model_id"] not in seen:
                    seen.add(r["model_id"])
                    kept.append(r)
            selected = kept
        values = []
        for r in selected:
            v = (
                r["score"]
                if k.metric in ("primary", r["primary_metric"])
                else (r["metrics"] or {}).get(k.metric)
            )
            if v is not None:
                values.append((float(v), r))
        values.sort(key=lambda x: x[0])
        scores = np.array([v for v, _ in values])
        latest = selected[0] if selected else None
        metric_key = (
            (latest["primary_metric"] if latest else None) if k.metric == "primary" else k.metric
        )
        hib = meta_for(latest, metric_key).get("higher_is_better") if latest else None
        top = None
        if values:
            best = values[0] if hib is False else values[-1]
            top = a.DistributionTop(
                run_id=best[1]["run_id"], model=model_ref(best[1]), score=best[0]
            )
        is_q = scores.size > 500
        out_scores = (
            np.quantile(scores, np.linspace(0, 1, 500)).tolist() if is_q else scores.tolist()
        )
        items.append(
            a.Distribution(
                task_name=k.task_name,
                task_hash=k.task_hash,
                metric=k.metric,
                n=int(scores.size),
                scores=out_scores,
                is_quantiles=is_q,
                higher_is_better=hib,
                top=top,
            )
        )
    response = a.DistributionsResponse(items=items)
    await put_cached(session, key, response)
    return response
