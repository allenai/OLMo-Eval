"""The instance list of a task result: filters, baseline join, sort and keyset pages."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from olmo_eval_api.errors import bad_request
from olmo_eval_api.services.filters import like_contains
from olmo_eval_api.services.queries import decode_cursor, encode_cursor, keyset, order_by

INSTANCE_SORTS = {
    "doc_id": ("ir.doc_id", False),
    "native_id": ("ir.native_id", False),
    "score": ("ir.primary_score", False),
    "-score": ("ir.primary_score", True),
    "delta": ("(ir.primary_score - b.primary_score)", False),
    "-delta": ("(ir.primary_score - b.primary_score)", True),
    "length": ("ir.completion_tokens", False),
    "-length": ("ir.completion_tokens", True),
}
HAS_FLAGS = {
    "scoring_error": "ir.has_scoring_error",
    "execution_result": "ir.has_execution_result",
    "judge_result": "ir.has_judge_result",
    "trajectory": "ir.has_trajectory",
}


def goodness_sql(col: str, higher_is_better: bool | None) -> str:
    return col if higher_is_better is not False else f"(1 - {col})"


@dataclass(frozen=True)
class InstanceFilters:
    baseline_tr_id: int | None
    baseline_requested: bool
    higher_is_better: bool | None
    threshold: float
    correct: str | None = None
    vs: str | None = None
    finish_reason: Sequence[str] = field(default_factory=list)
    len_min: int | None = None
    len_max: int | None = None
    score_min: float | None = None
    score_max: float | None = None
    has: Sequence[str] = field(default_factory=list)
    q: str | None = None


async def query_instances(
    session: AsyncSession,
    task_result_id: int,
    f: InstanceFilters,
    sort: str,
    cursor: str | None,
    limit: int,
) -> tuple[Sequence[Mapping[Any, Any]], int, str | None]:
    """One page of instances, the total matching count, and the next cursor.

    The count (an ILIKE scan for text search) runs on the first page only; the cursor carries
    it to later pages. A new filter starts again without a cursor, so it is counted again.
    """
    if sort not in INSTANCE_SORTS:
        raise bad_request(f"invalid sort {sort!r}")
    params: dict[str, Any] = {"tid": task_result_id, "thr": f.threshold}
    join = ""
    base_cols = "NULL::float8 AS b_score"
    if f.baseline_tr_id is not None:
        join = (
            "LEFT JOIN instance_results b "
            "ON b.task_result_id = :btid AND b.native_id = ir.native_id"
        )
        params["btid"] = f.baseline_tr_id
        base_cols = "b.primary_score AS b_score"
    elif sort in ("delta", "-delta") or f.vs:
        if not f.baseline_requested:
            raise bad_request("delta sort requires baseline")
        join = "LEFT JOIN (SELECT NULL::float8 AS primary_score) b ON FALSE"
    good = goodness_sql("ir.primary_score", f.higher_is_better)
    bgood = goodness_sql("b.primary_score", f.higher_is_better)
    is_correct = f"(ir.primary_score IS NOT NULL AND {good} >= :thr)"
    b_correct = f"(b.primary_score IS NOT NULL AND {bgood} >= :thr)"
    where = ["ir.task_result_id = :tid"]
    if f.correct == "correct":
        where.append(is_correct)
    elif f.correct == "incorrect":
        where.append(f"NOT {is_correct}")
    elif f.correct == "partial":
        where.append(f"({good} > 0 AND {good} < 1)")
    if f.vs:
        if f.baseline_tr_id is None:
            where.append("FALSE")
        elif f.vs == "gained":
            where.append(f"{is_correct} AND b.primary_score IS NOT NULL AND NOT {b_correct}")
        elif f.vs == "lost":
            where.append(f"NOT {is_correct} AND {b_correct} AND ir.primary_score IS NOT NULL")
        elif f.vs == "both_right":
            where.append(f"{is_correct} AND {b_correct}")
        elif f.vs == "both_wrong":
            where.append(
                f"ir.primary_score IS NOT NULL AND b.primary_score IS NOT NULL "
                f"AND NOT {is_correct} AND NOT {b_correct}"
            )
        elif f.vs == "changed":
            where.append(
                "b.native_id IS NOT NULL AND ir.primary_score IS DISTINCT FROM b.primary_score"
            )
    if f.finish_reason:
        where.append("ir.finish_reason = ANY(:fr)")
        params["fr"] = list(f.finish_reason)
    if f.len_min is not None:
        where.append("ir.completion_tokens >= :lmin")
        params["lmin"] = f.len_min
    if f.len_max is not None:
        where.append("ir.completion_tokens <= :lmax")
        params["lmax"] = f.len_max
    if f.score_min is not None:
        where.append("ir.primary_score >= :smin")
        params["smin"] = f.score_min
    if f.score_max is not None:
        where.append("ir.primary_score <= :smax")
        params["smax"] = f.score_max
    for flag in f.has:
        if flag not in HAS_FLAGS:
            raise bad_request(f"unknown has={flag}")
        where.append(HAS_FLAGS[flag])
    if f.q:
        params["q"] = like_contains(f.q)
        where.append(
            "(ir.prompt_preview ILIKE :q OR ir.output_preview ILIKE :q OR ir.label ILIKE :q "
            "OR ir.native_id ILIKE :q)"
        )
    where_sql = " AND ".join(where)
    expr, desc = INSTANCE_SORTS[sort]
    keys = [(expr, desc), ("ir.native_id", desc)] if expr != "ir.native_id" else [(expr, desc)]
    values = decode_cursor(cursor)
    total: int | None = None
    if values is not None and len(values) == len(keys) + 1:
        # Later pages carry the first page's total, so the count runs once per query.
        *values, carried = values
        if isinstance(carried, bool) or not isinstance(carried, int) or carried < 0:
            raise bad_request("invalid cursor")
        total = carried
    cond, cparams = keyset(keys, values)
    sql = f"""
        SELECT ir.*, {base_cols}, {expr} AS sort_value
        FROM instance_results ir {join}
        WHERE {where_sql} AND {cond}
        ORDER BY {order_by(keys)} LIMIT {limit + 1}
    """
    rows = list((await session.execute(text(sql), {**params, **cparams})).mappings().all())
    if total is None:
        count_sql = f"SELECT count(*) FROM instance_results ir {join} WHERE {where_sql}"
        total = int((await session.execute(text(count_sql), params)).scalar_one())
    next_cursor = None
    if len(rows) > limit:
        rows = rows[:limit]
        last = rows[-1]
        values = [last["sort_value"], last["native_id"]] if len(keys) == 2 else [last["native_id"]]
        next_cursor = encode_cursor([*values, total])
    return rows, total, next_cursor
