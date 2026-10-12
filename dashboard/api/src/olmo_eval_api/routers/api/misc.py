"""Identity, health, search, home, groups, subjects and saved views."""

from __future__ import annotations

import re
import uuid
from collections.abc import Mapping
from datetime import UTC, date, datetime, timedelta
from typing import Annotated, Any
from urllib.parse import quote

from fastapi import APIRouter, Query, Request, Response
from sqlalchemy import text

from olmo_eval_api.errors import bad_request, forbidden, not_found
from olmo_eval_api.routers.health import HealthResponse, health
from olmo_eval_api.schemas import api as a
from olmo_eval_api.services.filters import like_contains
from olmo_eval_api.services.queries import (
    MAX_LIMIT,
    VISIBLE,
    clamp_limit,
    decode_cursor,
    encode_cursor,
    fetch_run_rows,
    keyset,
    order_by,
    run_summary,
    username,
)
from olmo_eval_api.services.subjects import (
    assign_labels,
    models_task_results,
    resolve_one,
    resolve_subjects,
)

from .deps import SessionDep, UserDep

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def api_health(request: Request) -> HealthResponse:
    return await health(request)


@router.get("/me", response_model=a.MeResponse)
async def me(user: UserDep) -> a.MeResponse:
    return a.MeResponse(email=user.email, username=user.username, dev_mode=user.dev_mode)


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

SEARCH_TYPES = ("run", "model", "task", "suite", "group", "user", "id")
CROCKFORD_ID = re.compile(r"^[0-9A-HJKMNP-TV-Z]{26}$", re.IGNORECASE)
RUN_ID = re.compile(r"^[a-z0-9]{12}$")
COMMIT = re.compile(r"^[0-9a-f]{7,40}$")
MAX_HEATMAP_GROUPS = 50


def _like_all(column: str, tokens: list[str], params: dict[str, Any], prefix: str) -> str:
    parts = []
    for i, tok in enumerate(tokens):
        params[f"{prefix}{i}"] = like_contains(tok)
        parts.append(f"{column} ILIKE :{prefix}{i}")
    return " AND ".join(parts) or "TRUE"


@router.get("/search", response_model=a.SearchResponse)
async def search(
    session: SessionDep,
    user: UserDep,
    q: str,
    types: str | None = None,
    limit: Annotated[int, Query(ge=1, le=20)] = 5,
) -> a.SearchResponse:
    query = q.strip()
    if not query:
        raise bad_request("q must not be empty")
    wanted = [t for t in (types.split(",") if types else SEARCH_TYPES) if t]
    bad = set(wanted) - set(SEARCH_TYPES)
    if bad:
        raise bad_request(f"unknown search types: {', '.join(sorted(bad))}")
    tokens = query.lower().split()
    groups: list[a.SearchGroup] = []
    for kind in SEARCH_TYPES:
        if kind not in wanted:
            continue
        items = await _search_type(session, kind, query, tokens, limit)
        if items:
            groups.append(a.SearchGroup(type=kind, items=items))  # type: ignore[arg-type]
    return a.SearchResponse(query=query, groups=groups)


async def _search_type(
    session: Any, kind: str, query: str, tokens: list[str], limit: int
) -> list[a.SearchResult]:
    params: dict[str, Any] = {"q": query.lower(), "lim": limit}
    if kind == "run":
        cond = _like_all("r.search_text", tokens, params, "t")
        rows = await fetch_run_rows(
            session,
            f"{VISIBLE} AND {cond}",
            params,
            order="similarity(r.search_text, :q) DESC, r.created_at DESC",
            limit=limit,
        )
        return [
            a.SearchResult(
                type="run",
                key=r["run_id"],
                label=r["experiment_name"] or r["run_id"],
                secondary=" · ".join(x for x in (r["m_name"], r["author"]) if x) or None,
                href=f"/runs/{r['run_id']}",
                subject=f"r:{r['run_id']}",
            )
            for r in rows
        ]
    if kind == "model":
        cond = _like_all("(m.name || ' ' || m.series)", tokens, params, "t")
        rows = await session.execute(
            text(
                f"""
                SELECT m.series, min(m.series_label) AS label, count(*) AS n,
                       (array_agg(m.model_id ORDER BY m.last_seen_at DESC))[1] AS latest,
                       max(similarity(m.series, :q)) AS sim, max(m.last_seen_at) AS seen
                FROM models m WHERE {cond} GROUP BY m.series
                ORDER BY sim DESC, seen DESC LIMIT :lim
                """
            ),
            params,
        )
        return [
            a.SearchResult(
                type="model",
                key=r["series"],
                label=r["label"],
                secondary=f"{r['n']} checkpoint{'s' if r['n'] != 1 else ''} · {r['series']}",
                href=f"/models/{quote(r['series'], safe='')}",
                subject=f"m:{r['latest']}",
            )
            for r in rows.mappings()
        ]
    if kind == "task":
        cond = _like_all("task_name", tokens, params, "t")
        rows = await session.execute(
            text(
                f"SELECT task_name, max(last_seen_at) AS seen FROM task_variants WHERE {cond} "
                "GROUP BY task_name ORDER BY similarity(task_name, :q) DESC, seen DESC LIMIT :lim"
            ),
            params,
        )
        return [
            a.SearchResult(
                type="task",
                key=r[0],
                label=r[0],
                secondary=None,
                href=f"/tasks/{quote(r[0], safe='')}",
                subject=None,
            )
            for r in rows
        ]
    if kind == "suite":
        cond = _like_all("suite_name", tokens, params, "t")
        rows = await session.execute(
            text(
                f"SELECT suite_name, max(last_seen_at) AS seen FROM suite_defs WHERE {cond} "
                "GROUP BY suite_name ORDER BY similarity(suite_name, :q) DESC, seen DESC LIMIT :lim"
            ),
            params,
        )
        return [
            a.SearchResult(
                type="suite",
                key=r[0],
                label=r[0],
                secondary=None,
                href=f"/suites/{quote(r[0], safe='')}",
                subject=None,
            )
            for r in rows
        ]
    if kind == "group":
        cond = _like_all("r.experiment_group", tokens, params, "t")
        rows = await session.execute(
            text(
                f"SELECT r.experiment_group, count(*) AS n, max(r.created_at) AS last FROM runs r "
                f"WHERE r.experiment_group IS NOT NULL AND {VISIBLE} AND {cond} GROUP BY 1 "
                "ORDER BY similarity(r.experiment_group, :q) DESC, last DESC LIMIT :lim"
            ),
            params,
        )
        return [
            a.SearchResult(
                type="group",
                key=r[0],
                label=r[0],
                secondary=f"{r[1]} run{'s' if r[1] != 1 else ''}",
                href=f"/groups/{quote(r[0], safe='')}",
                subject=None,
            )
            for r in rows
        ]
    if kind == "user":
        cond = _like_all("u.name", tokens, params, "t")
        rows = await session.execute(
            text(
                f"""
                SELECT u.name, count(*) AS n FROM (
                    SELECT coalesce(r.author, split_part(r.uploaded_by, '@', 1)) AS name
                    FROM runs r WHERE {VISIBLE}
                ) u WHERE {cond} GROUP BY u.name
                ORDER BY similarity(u.name, :q) DESC, n DESC LIMIT :lim
                """
            ),
            params,
        )
        return [
            a.SearchResult(
                type="user",
                key=r[0],
                label=r[0],
                secondary=f"{r[1]} run{'s' if r[1] != 1 else ''}",
                href=f"/runs?user={quote(r[0], safe='')}",
                subject=None,
            )
            for r in rows
        ]
    return await _search_ids(session, query, limit)


async def _search_ids(session: Any, query: str, limit: int) -> list[a.SearchResult]:
    where = None
    params: dict[str, Any] = {}
    reason = ""
    if CROCKFORD_ID.match(query):
        where = (
            "(upper(r.beaker_experiment_id) = :id OR upper(r.beaker_job_id) = :id "
            "OR upper(r.beaker_result_dataset_id) = :id)"
        )
        params["id"] = query.upper()
        reason = "Beaker ID"
    elif query.startswith("gs://"):
        where = "(:uri LIKE r.gcs_prefix || '%' OR r.gcs_prefix LIKE :uri || '%')"
        params["uri"] = query
        reason = "GCS path"
    elif RUN_ID.match(query) and not COMMIT.match(query):
        where = "r.run_id = :id"
        params["id"] = query
        reason = "run ID"
    elif RUN_ID.match(query) or COMMIT.match(query):
        where = "(r.run_id = :id OR r.git_commit LIKE :prefix)"
        params.update(id=query, prefix=query.lower() + "%")
        reason = "run ID or commit"
    if where is None:
        return []
    rows = await fetch_run_rows(session, where, params, limit=limit)
    return [
        a.SearchResult(
            type="id",
            key=r["run_id"],
            label=r["experiment_name"] or r["run_id"],
            secondary=f"{reason} · {r['m_name']}",
            href=f"/runs/{r['run_id']}",
            subject=f"r:{r['run_id']}",
        )
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Home
# ---------------------------------------------------------------------------

STATUS_RANK = {"failed": 3, "partial": 2, "running": 1, "complete": 0}


@router.get("/activity", response_model=a.ActivityResponse)
async def activity(
    session: SessionDep,
    user: UserDep,
    user_filter: Annotated[str | None, Query(alias="user")] = None,
    cursor: str | None = None,
    limit: int | None = None,
) -> a.ActivityResponse:
    limit = clamp_limit(limit, default=20, maximum=100)
    where = [VISIBLE]
    params: dict[str, Any] = {}
    if user_filter:
        params["who"] = user.username if user_filter == "me" else user_filter
        where.append("(r.author = :who OR split_part(r.uploaded_by, '@', 1) = :who)")
    keys = [("created_at", True), ("key", True)]
    cond, cparams = keyset(keys, decode_cursor(cursor))
    rows = (
        await session.execute(
            text(
                f"""
                WITH g AS (
                    SELECT coalesce(r.launch_id, 'run:' || r.run_id) AS key,
                           max(r.created_at) AS created_at
                    FROM runs r WHERE {" AND ".join(where)} GROUP BY 1
                )
                SELECT key, created_at FROM g WHERE {cond}
                ORDER BY {order_by(keys)} LIMIT {limit + 1}
                """
            ),
            {**params, **cparams},
        )
    ).all()
    next_cursor = None
    if len(rows) > limit:
        rows = rows[:limit]
        next_cursor = encode_cursor([rows[-1][1], rows[-1][0]])
    group_keys = [r[0] for r in rows]
    launches = [k for k in group_keys if not k.startswith("run:")]
    singles = [k[4:] for k in group_keys if k.startswith("run:")]
    run_rows = await fetch_run_rows(
        session,
        f"{' AND '.join(where)} AND ((r.launch_id = ANY(:l)) OR (r.launch_id IS NULL "
        "AND r.run_id = ANY(:s)))",
        {**params, "l": launches, "s": singles},
    )
    by_group: dict[str, list[Mapping[Any, Any]]] = {}
    for r in run_rows:
        by_group.setdefault(r["launch_id"] or f"run:{r['run_id']}", []).append(r)
    errors: dict[str, str] = {}
    run_ids = [r["run_id"] for r in run_rows]
    if run_ids:
        for rid, err in await session.execute(
            text(
                "SELECT DISTINCT ON (run_id) run_id, error FROM task_results "
                "WHERE run_id = ANY(:r) AND error IS NOT NULL ORDER BY run_id, id"
            ),
            {"r": run_ids},
        ):
            errors[rid] = err
    items = []
    for key, created in rows:
        runs = by_group.get(key, [])
        if not runs:
            continue
        status = max((r["status"] for r in runs), key=lambda s: STATUS_RANK.get(s, 0))
        failure = next((errors[r["run_id"]] for r in runs if r["run_id"] in errors), None)
        first = runs[0]
        items.append(
            a.ActivityItem(
                key=key,
                launch_id=None if key.startswith("run:") else key,
                user=first["author"] or username(first["uploaded_by"]),
                created_at=created,
                runs=[run_summary(r) for r in runs],
                n_tasks=sum(r["num_tasks"] for r in runs),
                n_failed_tasks=sum(r["num_failed_tasks"] for r in runs),
                status=status,
                failure_reason=failure,
            )
        )
    return a.ActivityResponse(items=items, next_cursor=next_cursor)


@router.get("/stats/summary", response_model=a.StatsSummaryResponse)
async def stats_summary(
    session: SessionDep,
    user: UserDep,
    window_days: Annotated[int, Query(ge=1, le=90)] = 7,
) -> a.StatsSummaryResponse:
    start = datetime.now(UTC).date() - timedelta(days=window_days - 1)
    since = datetime.combine(start, datetime.min.time(), tzinfo=UTC)
    rows = (
        (
            await session.execute(
                text(
                    f"""
                SELECT (r.created_at AT TIME ZONE 'UTC')::date AS day, count(*) AS runs,
                       count(DISTINCT r.model_id) AS models, sum(r.num_instances) AS instances
                FROM runs r WHERE {VISIBLE} AND r.created_at >= :since GROUP BY 1
                """
                ),
                {"since": since},
            )
        )
        .mappings()
        .all()
    )
    totals = (
        await session.execute(
            text(
                f"SELECT count(*), count(DISTINCT r.model_id), coalesce(sum(r.num_instances), 0) "
                f"FROM runs r WHERE {VISIBLE} AND r.created_at >= :since"
            ),
            {"since": since},
        )
    ).one()
    by_day: dict[date, Mapping[Any, Any]] = {r["day"]: r for r in rows}
    daily = []
    for i in range(window_days):
        day = start + timedelta(days=i)
        r = by_day.get(day)
        daily.append(
            a.DailyStat(
                date=day.isoformat(),
                runs=int(r["runs"]) if r else 0,
                models=int(r["models"]) if r else 0,
                instances=int(r["instances"] or 0) if r else 0,
            )
        )
    return a.StatsSummaryResponse(
        window_days=window_days,
        runs=int(totals[0]),
        models=int(totals[1]),
        instances=int(totals[2]),
        daily=daily,
    )


# ---------------------------------------------------------------------------
# Groups
# ---------------------------------------------------------------------------


@router.get("/groups", response_model=a.GroupsListResponse)
async def list_groups(
    session: SessionDep,
    user: UserDep,
    q: str | None = None,
    active_days: Annotated[int | None, Query(ge=1)] = None,
    include_heatmap: bool = False,
    sort: Annotated[str, Query(pattern="^-last_run_at$")] = "-last_run_at",
    cursor: str | None = None,
    limit: int | None = None,
) -> a.GroupsListResponse:
    # Each heatmap costs several queries, so pages that include them stay small.
    limit = clamp_limit(limit, maximum=MAX_HEATMAP_GROUPS if include_heatmap else MAX_LIMIT)
    where = [VISIBLE, "r.experiment_group IS NOT NULL"]
    params: dict[str, Any] = {}
    if q:
        where.append(_like_all("r.experiment_group", q.lower().split(), params, "q"))
    having = ""
    if active_days:
        having = "HAVING max(r.created_at) >= now() - make_interval(days => :days)"
        params["days"] = active_days
    base = f"""
        WITH g AS (
            SELECT r.experiment_group AS name, count(DISTINCT r.run_id) AS n_runs,
                   count(DISTINCT r.model_id) AS n_models,
                   (SELECT count(DISTINCT t.task_name) FROM task_results t
                    JOIN runs r2 ON r2.run_id = t.run_id
                    WHERE r2.experiment_group = r.experiment_group) AS n_tasks,
                   (array_agg(DISTINCT coalesce(r.author, split_part(r.uploaded_by, '@', 1))))[1:10]
                       AS users,
                   min(r.created_at) AS first_run_at, max(r.created_at) AS last_run_at
            FROM runs r WHERE {" AND ".join(where)} GROUP BY r.experiment_group {having}
        )
    """
    keys = [("last_run_at", True), ("name", True)]
    cond, cparams = keyset(keys, decode_cursor(cursor))
    rows = (
        (
            await session.execute(
                text(
                    f"{base} SELECT * FROM g WHERE {cond} "
                    f"ORDER BY {order_by(keys)} LIMIT {limit + 1}"
                ),
                {**params, **cparams},
            )
        )
        .mappings()
        .all()
    )
    total = (await session.execute(text(f"{base} SELECT count(*) FROM g"), params)).scalar_one()
    next_cursor = None
    if len(rows) > limit:
        rows = rows[:limit]
        next_cursor = encode_cursor([rows[-1]["last_run_at"], rows[-1]["name"]])
    items = []
    for r in rows:
        heatmap = await _group_heatmap(session, r["name"]) if include_heatmap else None
        items.append(
            a.GroupRow(
                name=r["name"],
                n_runs=int(r["n_runs"]),
                n_models=int(r["n_models"]),
                n_tasks=int(r["n_tasks"]),
                users=[u for u in (r["users"] or []) if u],
                first_run_at=r["first_run_at"],
                last_run_at=r["last_run_at"],
                heatmap=heatmap,
            )
        )
    return a.GroupsListResponse(items=items, next_cursor=next_cursor, total=int(total))


async def _group_heatmap(session: Any, group: str) -> a.MiniHeatmap | None:
    latest = (
        (
            await session.execute(
                text(
                    f"""
                    SELECT DISTINCT ON (r.model_id) r.model_id, r.run_id, r.created_at
                    FROM runs r WHERE r.experiment_group = :g AND {VISIBLE}
                    ORDER BY r.model_id, r.created_at DESC
                    """
                ),
                {"g": group},
            )
        )
        .mappings()
        .all()
    )
    latest = sorted(latest, key=lambda r: r["created_at"], reverse=True)[:8]
    if not latest:
        return None
    run_ids = [r["run_id"] for r in latest]
    cols = (
        (
            await session.execute(
                text(
                    "SELECT suite_name FROM suite_results WHERE run_id = ANY(:r) "
                    "AND parent_suite IS NULL GROUP BY 1 ORDER BY count(*) DESC, 1 LIMIT 8"
                ),
                {"r": run_ids},
            )
        )
        .scalars()
        .all()
    )
    kind = "suite"
    if not cols:
        kind = "task"
        cols = (
            (
                await session.execute(
                    text(
                        "SELECT task_name FROM task_results WHERE run_id = ANY(:r) "
                        "GROUP BY 1 ORDER BY count(*) DESC, 1 LIMIT 8"
                    ),
                    {"r": run_ids},
                )
            )
            .scalars()
            .all()
        )
    if kind == "suite":
        values_sql = (
            "SELECT run_id, suite_name, score FROM suite_results "
            "WHERE run_id = ANY(:r) AND suite_name = ANY(:c)"
        )
    else:
        values_sql = (
            "SELECT DISTINCT ON (run_id, task_name) run_id, task_name, score FROM task_results "
            "WHERE run_id = ANY(:r) AND task_name = ANY(:c) "
            "ORDER BY run_id, task_name, updated_at DESC"
        )
    values = {
        (r[0], r[1]): r[2]
        for r in await session.execute(text(values_sql), {"r": run_ids, "c": list(cols)})
    }
    subjects, _ = await resolve_subjects(
        session, [f"m:{r['model_id']}" for r in latest], group, load_results=False
    )
    labels = {s.key: s.label for s in subjects}
    return a.MiniHeatmap(
        rows=[
            a.MiniHeatmapRowsItem(
                key=f"m:{r['model_id']}", label=labels.get(f"m:{r['model_id']}", "")
            )
            for r in latest
        ],
        columns=[a.MiniHeatmapColumnsItem(key=f"{kind}:{c}", label=c) for c in cols],
        values=[[values.get((r["run_id"], c)) for c in cols] for r in latest],
    )


@router.get("/groups/detail", response_model=a.GroupDetailResponse)
async def group_detail(group: str, session: SessionDep, user: UserDep) -> a.GroupDetailResponse:
    stats = (
        (
            await session.execute(
                text(
                    f"""
                    SELECT count(*) AS n_runs, count(DISTINCT r.model_id) AS n_models,
                           array_agg(DISTINCT coalesce(r.author, split_part(r.uploaded_by, '@', 1)))
                               AS users,
                           min(r.created_at) AS first_run_at, max(r.created_at) AS last_run_at,
                           array_agg(DISTINCT r.model_id) AS model_ids
                    FROM runs r WHERE r.experiment_group = :g AND {VISIBLE}
                    """
                ),
                {"g": group},
            )
        )
        .mappings()
        .one()
    )
    if not stats["n_runs"]:
        raise not_found(f"Group {group!r} not found")
    subjects, _ = await resolve_subjects(
        session, [f"m:{m}" for m in stats["model_ids"]], group, load_results=False
    )
    trs_by_model = await models_task_results(session, [s.key[2:] for s in subjects], group)
    for s in subjects:
        s.trs = trs_by_model[s.key[2:]]
    subjects.sort(key=lambda s: s.latest_at or datetime.min.replace(tzinfo=UTC), reverse=True)
    assign_labels(subjects)
    tasks = sorted({t for s in subjects for t in s.trs})
    coverage = []
    for s in subjects:
        for t in tasks:
            tr = s.trs.get(t)
            status = "missing" if tr is None else ("failed" if tr["error"] else "ok")
            coverage.append(
                a.CoverageCell(
                    subject=s.key,
                    task_name=t,
                    status=status,  # type: ignore[arg-type]
                    task_result_id=tr["id"] if tr is not None else None,
                    run_id=tr["run_id"] if tr is not None else None,
                )
            )
    return a.GroupDetailResponse(
        name=group,
        n_runs=int(stats["n_runs"]),
        n_models=int(stats["n_models"]),
        n_tasks=len(tasks),
        users=sorted(u for u in stats["users"] if u),
        first_run_at=stats["first_run_at"],
        last_run_at=stats["last_run_at"],
        subjects=[s.info() for s in subjects],
        tasks=tasks,
        coverage=coverage,
    )


# ---------------------------------------------------------------------------
# Subjects
# ---------------------------------------------------------------------------


@router.post("/subjects/resolve", response_model=a.ResolveSubjectsResponse)
async def subjects_resolve(
    body: a.ResolveSubjectsRequest, session: SessionDep, user: UserDep
) -> a.ResolveSubjectsResponse:
    subjects, missing = await resolve_subjects(session, body.subjects, body.group)
    return a.ResolveSubjectsResponse(items=[s.info() for s in subjects], missing=missing)


@router.get("/subjects/task-result", response_model=a.SubjectTaskResultResponse)
async def subject_task_result(
    session: SessionDep, user: UserDep, subject: str, task: str, group: str | None = None
) -> a.SubjectTaskResultResponse:
    s = await resolve_one(session, subject, group)
    tr = s.trs.get(task)
    return a.SubjectTaskResultResponse(
        subject=subject,
        task_name=task,
        task_result_id=tr["id"] if tr is not None else None,
        run_id=tr["run_id"] if tr is not None else None,
        task_hash=tr["task_hash"] if tr is not None else None,
    )


# ---------------------------------------------------------------------------
# Saved views
# ---------------------------------------------------------------------------


def _view(row: Mapping[Any, Any]) -> a.SavedView:
    return a.SavedView(
        id=row["id"],
        name=row["name"],
        owner_email=row["owner_email"],
        shared=row["shared"],
        page=row["page"],
        query=row["query"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


@router.get("/views", response_model=a.SavedViewsResponse)
async def list_views(
    session: SessionDep, user: UserDep, page: str | None = None
) -> a.SavedViewsResponse:
    where = "(owner_email = :me OR shared)"
    params: dict[str, Any] = {"me": user.email}
    if page:
        where += " AND page = :page"
        params["page"] = page
    rows = (
        (
            await session.execute(
                text(f"SELECT * FROM saved_views WHERE {where} ORDER BY updated_at DESC LIMIT 500"),
                params,
            )
        )
        .mappings()
        .all()
    )
    return a.SavedViewsResponse(
        mine=[_view(r) for r in rows if r["owner_email"] == user.email],
        shared=[_view(r) for r in rows if r["owner_email"] != user.email],
    )


MAX_VIEWS_PER_USER = 200


def _check_view(name: str | None, page: str | None, query: str | None) -> None:
    if name is not None and not 1 <= len(name) <= 120:
        raise bad_request("name must be 1-120 characters")
    if page is not None and not 1 <= len(page) <= 32:
        raise bad_request("page must be 1-32 characters")
    if query is not None and len(query) > 4000:
        raise bad_request("query must be at most 4000 characters")


@router.post("/views", response_model=a.SavedView)
async def create_view(body: a.CreateViewRequest, session: SessionDep, user: UserDep) -> a.SavedView:
    _check_view(body.name, body.page, body.query)
    owned = await session.scalar(
        text("SELECT count(*) FROM saved_views WHERE owner_email = :o"), {"o": user.email}
    )
    if owned >= MAX_VIEWS_PER_USER:
        raise bad_request(f"you can have at most {MAX_VIEWS_PER_USER} saved views")
    row = (
        (
            await session.execute(
                text(
                    "INSERT INTO saved_views (id, name, owner_email, shared, page, query, "
                    "created_at, updated_at) VALUES (:id, :n, :o, :s, :p, :q, now(), now()) "
                    "RETURNING *"
                ),
                {
                    "id": uuid.uuid4().hex,
                    "n": body.name,
                    "o": user.email,
                    "s": body.shared,
                    "p": body.page,
                    "q": body.query.removeprefix("?"),
                },
            )
        )
        .mappings()
        .one()
    )
    await session.commit()
    return _view(row)


async def _own_view(session: Any, view_id: str, email: str) -> Mapping[Any, Any]:
    row = (
        (await session.execute(text("SELECT * FROM saved_views WHERE id = :i"), {"i": view_id}))
        .mappings()
        .first()
    )
    if row is None:
        raise not_found(f"View {view_id} not found")
    if row["owner_email"] != email:
        raise forbidden("only the owner can change this view")
    return row


@router.patch("/views/{view_id}", response_model=a.SavedView)
async def update_view(
    view_id: str, body: a.UpdateViewRequest, session: SessionDep, user: UserDep
) -> a.SavedView:
    row = dict(await _own_view(session, view_id, user.email))
    _check_view(body.name, None, body.query)
    for field in ("name", "query", "shared"):
        value = getattr(body, field)
        if value is not None:
            row[field] = value.removeprefix("?") if field == "query" else value
    updated = (
        (
            await session.execute(
                text(
                    "UPDATE saved_views SET name = :n, query = :q, shared = :s, updated_at = now() "
                    "WHERE id = :i RETURNING *"
                ),
                {"n": row["name"], "q": row["query"], "s": row["shared"], "i": view_id},
            )
        )
        .mappings()
        .one()
    )
    await session.commit()
    return _view(updated)


@router.delete("/views/{view_id}", status_code=204)
async def delete_view(view_id: str, session: SessionDep, user: UserDep) -> Response:
    await _own_view(session, view_id, user.email)
    await session.execute(text("DELETE FROM saved_views WHERE id = :i"), {"i": view_id})
    await session.commit()
    return Response(status_code=204)
