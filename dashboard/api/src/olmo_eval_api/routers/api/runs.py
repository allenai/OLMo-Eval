"""/api/runs: list, facets, bulk tags, detail and patch."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Annotated, Any, cast

from fastapi import APIRouter, Depends
from sqlalchemy import CursorResult, text

from olmo_eval_api.errors import bad_request, not_found
from olmo_eval_api.schemas import api as a
from olmo_eval_api.schemas.ingest import RUN_ID_PATTERN, TAG_PATTERN
from olmo_eval_api.services.filters import RunFilters, build_where, run_filters
from olmo_eval_api.services.queries import (
    RUN_FROM,
    clamp_limit,
    decode_cursor,
    encode_cursor,
    fetch_run_detail_row,
    fetch_run_rows,
    keyset,
    model_detail,
    order_by,
    reproduce_command,
    run_summary,
    run_summary_fields,
)
from olmo_eval_api.services.read_suites import load_defs, score_suite, tree
from olmo_eval_api.services.search_text import SEARCH_TEXT_SQL
from olmo_eval_api.services.subjects import resolve_one
from olmo_eval_api.stats.paired import Side, unpaired

from .deps import SessionDep, UserDep

router = APIRouter()
MAX_COLS = 20
_TAG_RE = re.compile(TAG_PATTERN)
_RUN_ID_RE = re.compile(RUN_ID_PATTERN)

SORTS: dict[str, str] = {
    "created_at": "r.created_at",
    "model": "r.model_name",
    "step": "m.step",
    "group": "r.experiment_group",
    "user": "r.author",
    "duration": "r.duration_seconds",
    "num_tasks": "r.num_tasks",
}


def parse_cols(cols: str | None) -> list[tuple[str, str, str]]:
    """[(key, kind, name)] from "task:<name>,suite:<name>"."""
    if not cols:
        return []
    out = []
    for raw in cols.split(","):
        raw = raw.strip()
        if not raw:
            continue
        kind, _, name = raw.partition(":")
        if kind not in ("task", "suite") or not name:
            raise bad_request(f"invalid column {raw!r}; use task:<name> or suite:<name>")
        out.append((raw, kind, name))
    if len(out) > MAX_COLS:
        raise bad_request(f"at most {MAX_COLS} score columns")
    return list(dict.fromkeys(out))


def score_expr(kind: str, name_param: str) -> str:
    if kind == "task":
        return (
            "(SELECT t.score FROM task_results t WHERE t.run_id = r.run_id "
            f"AND t.task_name = {name_param} ORDER BY t.updated_at DESC, t.id DESC LIMIT 1)"
        )
    return (
        "(SELECT s.score FROM suite_results s WHERE s.run_id = r.run_id "
        f"AND s.suite_name = {name_param})"
    )


def parse_sort(sort: str) -> tuple[str, bool, dict[str, Any]]:
    desc = sort.startswith("-")
    key = sort.removeprefix("-")
    if key.startswith("score:"):
        kind, _, name = key.removeprefix("score:").partition(":")
        if kind not in ("task", "suite") or not name:
            raise bad_request(f"invalid sort {sort!r}")
        return score_expr(kind, ":sort_name"), desc, {"sort_name": name}
    if key not in SORTS:
        raise bad_request(f"invalid sort {sort!r}")
    return SORTS[key], desc, {}


async def column_meta(session: Any, cols: list[tuple[str, str, str]]) -> list[a.ScoreColumnMeta]:
    tasks = [n for _, k, n in cols if k == "task"]
    suites = [n for _, k, n in cols if k == "suite"]
    task_meta: dict[str, Mapping[Any, Any]] = {}
    suite_meta: dict[str, Mapping[Any, Any]] = {}
    if tasks:
        rows = await session.execute(
            text(
                "SELECT DISTINCT ON (task_name) task_name, display_format, higher_is_better "
                "FROM task_results WHERE task_name = ANY(:n) AND finalized_at IS NOT NULL "
                "ORDER BY task_name, updated_at DESC"
            ),
            {"n": tasks},
        )
        task_meta = {r["task_name"]: r for r in rows.mappings()}
    if suites:
        rows = await session.execute(
            text(
                "SELECT DISTINCT ON (s.suite_name) s.suite_name, s.display_format, "
                "s.higher_is_better FROM suite_results s JOIN runs r ON r.run_id = s.run_id "
                "WHERE s.suite_name = ANY(:n) ORDER BY s.suite_name, r.created_at DESC"
            ),
            {"n": suites},
        )
        suite_meta = {r["suite_name"]: r for r in rows.mappings()}
    out = []
    for key, kind, name in cols:
        meta = (task_meta if kind == "task" else suite_meta).get(name) or {}
        out.append(
            a.ScoreColumnMeta(
                key=key,
                kind="suite" if kind == "suite" else "task",
                name=name,
                display_format=meta.get("display_format") or "raw",
                higher_is_better=meta.get("higher_is_better"),
            )
        )
    return out


@router.get("/runs", response_model=a.RunsListResponse)
async def list_runs(
    session: SessionDep,
    user: UserDep,
    filters: Annotated[RunFilters, Depends(run_filters)],
    sort: str = "-created_at",
    cols: str | None = None,
    baseline: str | None = None,
    cursor: str | None = None,
    limit: int | None = None,
) -> a.RunsListResponse:
    limit = clamp_limit(limit)
    columns = parse_cols(cols)
    where, filter_params = build_where(filters, user.username)
    params = dict(filter_params)
    expr, desc, sort_params = parse_sort(sort)
    params.update(sort_params)
    keys = [(expr, desc), ("r.run_id", desc)]
    cond, cparams = keyset(keys, decode_cursor(cursor))
    params.update(cparams)
    rows = await fetch_run_rows(
        session,
        f"{where} AND {cond}",
        params,
        order=order_by(keys),
        limit=limit + 1,
        extra_columns=f", {expr} AS sort_value",
    )
    total = (
        await session.execute(text(f"SELECT count(*) FROM {RUN_FROM} WHERE {where}"), filter_params)
    ).scalar_one()
    next_cursor = None
    if len(rows) > limit:
        rows = rows[:limit]
        last = rows[-1]
        next_cursor = encode_cursor([last["sort_value"], last["run_id"]])

    meta = await column_meta(session, columns)
    scores = await page_scores(session, [r["run_id"] for r in rows], columns)
    baseline_info = None
    baseline_cells: dict[str, Side] = {}
    if baseline:
        subject = await resolve_one(session, baseline)
        baseline_info = subject.info()
        baseline_cells = await baseline_column_sides(session, subject, columns)
    items = []
    for row in rows:
        values = {}
        for m in meta:
            cell = scores.get((row["run_id"], m.key))
            side = cell[1] if cell else None
            delta = None
            base = baseline_cells.get(m.key)
            if base is not None and side is not None:
                delta = a.DeltaStats(**unpaired(side, base, 0.05, m.higher_is_better).as_dict())
            values[m.key] = (
                a.ScoreColumnValue(
                    score=cell[0]["score"],
                    stderr=cell[0]["stderr"],
                    n=cell[0]["n"],
                    task_result_id=cell[0]["task_result_id"],
                    delta=delta,
                )
                if cell
                else a.ScoreColumnValue(
                    score=None, stderr=None, n=None, task_result_id=None, delta=None
                )
            )
        items.append(a.RunRow(**run_summary_fields(row), scores=values))
    return a.RunsListResponse(
        items=items, next_cursor=next_cursor, total=total, columns=meta, baseline=baseline_info
    )


async def page_scores(
    session: Any, run_ids: list[str], columns: list[tuple[str, str, str]]
) -> dict[tuple[str, str], tuple[dict[str, Any], Side]]:
    out: dict[tuple[str, str], tuple[dict[str, Any], Side]] = {}
    tasks = [n for _, k, n in columns if k == "task"]
    suites = [n for _, k, n in columns if k == "suite"]
    if not run_ids:
        return out
    if tasks:
        rows = await session.execute(
            text(
                "SELECT DISTINCT ON (run_id, task_name) id, run_id, task_name, task_hash, score, "
                "stderr, num_instances, score_is_mean FROM task_results "
                "WHERE run_id = ANY(:r) AND task_name = ANY(:t) "
                "ORDER BY run_id, task_name, updated_at DESC, id DESC"
            ),
            {"r": run_ids, "t": tasks},
        )
        for r in rows.mappings():
            cell = {
                "score": r["score"],
                "stderr": r["stderr"],
                "n": r["num_instances"],
                "task_result_id": r["id"],
            }
            side = Side(r["task_name"], r["task_hash"], r["score"], r["stderr"], r["score_is_mean"])
            out[(r["run_id"], f"task:{r['task_name']}")] = (cell, side)
    if suites:
        rows = await session.execute(
            text(
                "SELECT run_id, suite_name, score, stderr, n_instances FROM suite_results "
                "WHERE run_id = ANY(:r) AND suite_name = ANY(:s)"
            ),
            {"r": run_ids, "s": suites},
        )
        for r in rows.mappings():
            cell = {
                "score": r["score"],
                "stderr": r["stderr"],
                "n": r["n_instances"],
                "task_result_id": None,
            }
            side = Side(r["suite_name"], None, r["score"], r["stderr"], False)
            out[(r["run_id"], f"suite:{r['suite_name']}")] = (cell, side)
    return out


async def baseline_column_sides(
    session: Any, subject: Any, columns: list[tuple[str, str, str]]
) -> dict[str, Side]:
    out: dict[str, Side] = {}
    defs = None
    for key, kind, name in columns:
        if kind == "task":
            tr = subject.trs.get(name)
            if tr is not None:
                out[key] = Side(name, tr["task_hash"], tr["score"], tr["stderr"], False)
            continue
        if subject.run is not None:
            row = (
                await session.execute(
                    text(
                        "SELECT score, stderr FROM suite_results WHERE run_id = :r "
                        "AND suite_name = :s"
                    ),
                    {"r": subject.run["run_id"], "s": name},
                )
            ).first()
            if row is not None:
                out[key] = Side(name, None, row[0], row[1], False)
                continue
        if defs is None:
            defs = await load_defs(session)
        if name in defs:
            result = score_suite(tree(name, defs), subject.trs)
            out[key] = Side(name, None, result.score, result.stderr, False)
    return out


FACETS: dict[str, str] = {
    "user": "coalesce(r.author, split_part(r.uploaded_by, '@', 1))",
    "family": "m.family",
    "group": "r.experiment_group",
    "workspace": "r.beaker_workspace",
    "status": (
        "CASE WHEN r.status <> 'running' AND r.upload_state = 'uploading' "
        "THEN 'uploading' ELSE r.status END"
    ),
    "model": "r.model_name",
}


@router.get("/runs/facets", response_model=a.RunsFacetsResponse)
async def run_facets(
    session: SessionDep,
    user: UserDep,
    filters: Annotated[RunFilters, Depends(run_filters)],
) -> a.RunsFacetsResponse:
    out: dict[str, list[a.FacetValue]] = {}
    for facet in ("user", "family", "group", "workspace", "tag", "status", "model"):
        where, params = build_where(filters, user.username, exclude=facet)
        if facet == "tag":
            sql = (
                f"SELECT tag AS value, count(*) AS n FROM {RUN_FROM} "
                f"CROSS JOIN LATERAL unnest(r.tags) AS tag WHERE {where} "
                "GROUP BY tag ORDER BY n DESC, value LIMIT 50"
            )
        else:
            expr = FACETS[facet]
            sql = (
                f"SELECT {expr} AS value, count(*) AS n FROM {RUN_FROM} WHERE {where} "
                f"AND {expr} IS NOT NULL GROUP BY 1 ORDER BY n DESC, value LIMIT 50"
            )
        rows = (await session.execute(text(sql), params)).all()
        out[facet] = [a.FacetValue(value=str(r[0]), count=int(r[1])) for r in rows]
    return a.RunsFacetsResponse(**out)


# A run has at most MAX_TAGS tags (the ingest protocol's limit too), each at most 64 characters
# (TAG_PATTERN).
MAX_TAGS = 50
MAX_NOTES_CHARS = 10_000


def _check_tags(tags: list[str]) -> list[str]:
    unique = list(dict.fromkeys(tags))
    if len(unique) > MAX_TAGS:
        raise bad_request(f"at most {MAX_TAGS} tags")
    for tag in unique:
        if not _TAG_RE.match(tag):
            raise bad_request(f"invalid tag {tag!r}")
    return unique


_NEW_TAGS_SQL = (
    "ARRAY(SELECT DISTINCT x FROM unnest(r.tags || CAST(:add AS text[])) AS x "
    "WHERE x <> ALL(CAST(:remove AS text[])) ORDER BY x)"
)


@router.post("/runs/tags", response_model=a.BulkTagResponse)
async def bulk_tag(body: a.BulkTagRequest, session: SessionDep, user: UserDep) -> a.BulkTagResponse:
    add = _check_tags(body.add)
    remove = _check_tags(body.remove)
    if not body.run_ids:
        return a.BulkTagResponse(updated=0)
    params = {"add": add, "remove": remove, "ids": list(body.run_ids)}
    too_many = (
        (
            await session.execute(
                text(
                    f"SELECT r.run_id FROM runs r WHERE r.run_id = ANY(:ids) "
                    f"AND cardinality({_NEW_TAGS_SQL}) > {MAX_TAGS} ORDER BY r.run_id LIMIT 5"
                ),
                params,
            )
        )
        .scalars()
        .all()
    )
    if too_many:
        raise bad_request(f"runs would have more than {MAX_TAGS} tags: {', '.join(too_many)}")
    result = await session.execute(
        text(
            f"UPDATE runs r SET tags = {_NEW_TAGS_SQL}, updated_at = now() "
            "WHERE r.run_id = ANY(:ids)"
        ),
        params,
    )
    await session.execute(
        text(f"UPDATE runs r SET search_text = {SEARCH_TEXT_SQL} WHERE r.run_id = ANY(:ids)"),
        {"ids": list(body.run_ids)},
    )
    await session.commit()
    return a.BulkTagResponse(updated=cast(CursorResult, result).rowcount or 0)


def suite_order(rows: Sequence[Mapping[Any, Any]]) -> list[Mapping[Any, Any]]:
    """Parents before children (pre-order), roots in insertion order."""
    by_parent: dict[str | None, list[Mapping[Any, Any]]] = {}
    names = {r["suite_name"] for r in rows}
    for r in rows:
        parent = r["parent_suite"] if r["parent_suite"] in names else None
        by_parent.setdefault(parent, []).append(r)
    out: list[Mapping[Any, Any]] = []

    def walk(parent: str | None) -> None:
        for r in by_parent.get(parent, []):
            out.append(r)
            walk(r["suite_name"])

    walk(None)
    return out


async def build_run_detail(session: Any, run_id: str) -> a.RunDetail:
    row = await fetch_run_detail_row(session, run_id)
    suites = (
        (
            await session.execute(
                text(
                    "SELECT suite_name, parent_suite, aggregation, description, children, "
                    "definition_hash FROM suite_results WHERE run_id = :r ORDER BY id"
                ),
                {"r": run_id},
            )
        )
        .mappings()
        .all()
    )
    siblings = []
    if row["launch_id"]:
        siblings = await fetch_run_rows(
            session,
            "r.launch_id = :l AND r.run_id <> :r",
            {"l": row["launch_id"], "r": run_id},
            limit=200,
        )
    beaker = row["beaker"]
    env = row["environment"] or {}
    return a.RunDetail(
        **run_summary_fields(row),
        model_detail=model_detail(row),
        git=a.GitDetail(
            repo=row["git_repo"],
            commit=row["git_commit"],
            branch=row["git_branch"],
            dirty=row["git_dirty"],
        ),
        beaker=a.BeakerDetail(**{k: beaker.get(k) for k in a.BeakerDetail.model_fields})
        if beaker
        else None,
        environment=a.EnvironmentDetail(
            **{k: env.get(k) for k in a.EnvironmentDetail.model_fields if k != "packages"},
            packages=env.get("packages") or {},
        ),
        olmo_eval_version=row["olmo_eval_version"],
        argv=list(row["argv"] or []),
        reproduce_command=reproduce_command(row["argv"]),
        task_specs=list(row["task_specs"] or []),
        output_dir=row["output_dir"],
        notes=row["notes"],
        errors=[a.RunError(task=e.get("task"), error=e.get("error", "")) for e in row["errors"]],
        provider_init_seconds=row["provider_init_seconds"],
        startup_seconds=row["startup_seconds"],
        processing_started_at=row["processing_started_at"],
        processing_seconds=row["processing_seconds"],
        suites_used=[
            a.SuiteDef(
                name=s["suite_name"],
                aggregation=s["aggregation"],
                description=s["description"],
                children=[a.SuiteChild(**c) for c in s["children"]],
                definition_hash=s["definition_hash"],
            )
            for s in suite_order(list(suites))
        ],
        siblings=[run_summary(s) for s in siblings],
        updated_at=row["updated_at"],
    )


def _check_run_id(run_id: str) -> str:
    if not _RUN_ID_RE.match(run_id):
        raise not_found(f"Run {run_id} not found")
    return run_id


@router.get("/runs/{run_id}", response_model=a.RunDetail)
async def get_run(run_id: str, session: SessionDep, user: UserDep) -> a.RunDetail:
    return await build_run_detail(session, _check_run_id(run_id))


@router.patch("/runs/{run_id}", response_model=a.RunDetail)
async def patch_run(
    run_id: str, body: a.RunPatchRequest, session: SessionDep, user: UserDep
) -> a.RunDetail:
    _check_run_id(run_id)
    await fetch_run_detail_row(session, run_id)
    fields = body.model_fields_set
    if body.notes is not None and len(body.notes) > MAX_NOTES_CHARS:
        raise bad_request(f"notes must be at most {MAX_NOTES_CHARS} characters")
    if "tags" in fields and body.tags is not None:
        await session.execute(
            text("UPDATE runs SET tags = :t, updated_at = now() WHERE run_id = :r"),
            {"t": _check_tags(body.tags), "r": run_id},
        )
    if "notes" in fields:
        await session.execute(
            text("UPDATE runs SET notes = :n, updated_at = now() WHERE run_id = :r"),
            {"n": body.notes, "r": run_id},
        )
    await session.execute(
        text(f"UPDATE runs r SET search_text = {SEARCH_TEXT_SQL} WHERE r.run_id = :r"),
        {"r": run_id},
    )
    await session.commit()
    return await build_run_detail(session, run_id)
