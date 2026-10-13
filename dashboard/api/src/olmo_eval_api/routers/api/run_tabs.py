"""Run detail tabs: task results, suites, instances, histograms, inference, configs, artifacts."""

from __future__ import annotations

import asyncio
import re
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from typing import Annotated, Any, Literal

import numpy as np
import orjson
from fastapi import APIRouter, Query
from sqlalchemy import text

from olmo_eval_api.errors import bad_request, not_found
from olmo_eval_api.schemas import api as a
from olmo_eval_api.schemas.ingest import RUN_ID_PATTERN
from olmo_eval_api.services.cache import cache_key, get_cached, put_cached
from olmo_eval_api.services.common import latest_by_task, now_utc
from olmo_eval_api.services.instances import INSTANCE_SORTS, InstanceFilters, query_instances
from olmo_eval_api.services.links import gcs_console_url, run_links, split_gs_uri
from olmo_eval_api.services.queries import (
    clamp_limit,
    fetch_models,
    fetch_run_rows,
    model_ref,
    run_summary,
)
from olmo_eval_api.services.read_suites import definition_hashes, load_defs, score_suite, tree
from olmo_eval_api.services.runtime import RunTiming, derive_runtime, load_run_timings
from olmo_eval_api.services.subjects import (
    TR_COLUMNS,
    SideLoader,
    Subject,
    base_label,
    resolve_one,
    run_task_results,
)
from olmo_eval_api.services.suites import leaf_weights, leaves
from olmo_eval_api.stats.histograms import box_stats, histogram
from olmo_eval_api.stats.paired import (
    Delta,
    PairAccumulator,
    bootstrap_summary,
    compare_pairs,
    contingency,
    improved,
    stratified_pairs,
)

from .deps import SessionDep, SettingsDep, StatsDep, StorageDep, ThresholdQuery, UserDep
from .runs import suite_order

router = APIRouter()
_RUN_ID_RE = re.compile(RUN_ID_PATTERN)
MAX_RECORD_BYTES = 25 * 1024 * 1024
MAX_BATCHES = 2000


async def _run_row(session: Any, run_id: str) -> Mapping[Any, Any]:
    if not _RUN_ID_RE.match(run_id):
        raise not_found(f"Run {run_id} not found")
    extra = ", r.environment, r.argv, r.harness_config, r.task_specs"
    rows = await fetch_run_rows(session, "r.run_id = :r", {"r": run_id}, extra_columns=extra)
    if not rows:
        raise not_found(f"Run {run_id} not found")
    return rows[0]


def object_uri(run: Mapping[Any, Any], path: str | None) -> str | None:
    return run["gcs_prefix"] + path if path else None


def delta_model(d: Delta) -> a.DeltaStats:
    return a.DeltaStats(**d.as_dict())


# ---------------------------------------------------------------------------
# Task results
# ---------------------------------------------------------------------------


@router.get("/runs/{run_id}/task-results", response_model=a.RunTaskResultsResponse)
async def run_task_results_endpoint(
    run_id: str,
    session: SessionDep,
    user: UserDep,
    stats: StatsDep,
    baseline: str | None = None,
) -> a.RunTaskResultsResponse:
    run = await _run_row(session, run_id)
    sql = f"""
        SELECT {TR_COLUMNS}, tr.error_summary, tr.mean_completion_tokens, tr.truncation_rate,
               tr.finish_reason_counts, tr.predictions_path, tr.requests_path,
               tv.base_task, tv.num_fewshot, tv.task_limit, tv.split, tv.suites
        FROM task_results tr
        JOIN task_variants tv ON tv.task_name = tr.task_name AND tv.task_hash = tr.task_hash
        WHERE tr.run_id = :r ORDER BY tr.id
    """
    trs = list((await session.execute(text(sql), {"r": run_id})).mappings().all())
    subject: Subject | None = None
    if baseline:
        subject = await resolve_one(session, baseline)
    used = list(trs) + (list(subject.trs.values()) if subject else [])
    key = cache_key(
        "run-task-results",
        {"run_id": run_id, "baseline": baseline, **stats.__dict__},
        used,
    )
    cached = None
    if subject is not None:
        cached = await get_cached(session, key, a.RunTaskResultsResponse)

    # Only the baseline cells come from the cache; rows and summaries are always current.
    cells: dict[int, a.BaselineCell] = {}
    if cached is not None:
        cells = {i.task_result_id: i.baseline for i in cached.items if i.baseline is not None}
    elif subject is not None:
        loader = SideLoader(session)
        pairs = []
        for tr in trs:
            base = subject.trs.get(tr["task_name"])
            if base is None or base["id"] == tr["id"] or tr["primary_metric"] is None:
                continue
            loader.want(tr, tr["primary_metric"])
            loader.want(base, tr["primary_metric"])
            pairs.append((tr, base))
        await loader.load()

        def build_cells() -> None:
            for tr, base in pairs:
                side_a = loader.side(tr, tr["primary_metric"])
                side_b = loader.side(base, tr["primary_metric"])
                if side_a is None or side_b is None:
                    continue
                d = compare_pairs(
                    [(side_a, side_b)],
                    task_name=tr["task_name"],
                    alpha=stats.alpha,
                    n_boot=stats.n_boot,
                    seed=stats.seed,
                    higher_is_better=tr["higher_is_better"],
                )[0]
                cont = contingency(side_a, side_b, stats.threshold)
                cells[tr["id"]] = a.BaselineCell(
                    task_result_id=base["id"],
                    run_id=base["run_id"],
                    task_hash=base["task_hash"],
                    score=side_b.score,
                    stderr=side_b.stderr,
                    n=int(base["num_instances"] or 0),
                    metrics=base["metrics"] or {},
                    delta=delta_model(d),
                    contingency=a.Contingency(**cont) if cont else None,
                )

        # Bootstraps are CPU-bound; a worker thread keeps the event loop serving other requests.
        await asyncio.to_thread(build_cells)
    timing = (await load_run_timings(session, [run_id])).get(run_id, RunTiming())
    items = [
        a.TaskResultRow(
            task_result_id=tr["id"],
            run_id=run_id,
            task_name=tr["task_name"],
            task_hash=tr["task_hash"],
            base_task=tr["base_task"],
            primary_metric=tr["primary_metric"],
            metric_meta=tr["metric_meta"] or {},
            score=tr["score"],
            stderr=tr["stderr"],
            score_is_mean=bool(tr["score_is_mean"]),
            instance_scale=float(tr["instance_scale"] or 1),
            n=int(tr["num_instances"] or 0),
            instances_processed=tr["instances_processed"],
            instances_failed=tr["instances_failed"],
            instances_stored=int(tr["instances_stored"] or 0),
            metrics=tr["metrics"] or {},
            error=tr["error"],
            error_summary=tr["error_summary"],
            duration_seconds=tr["duration_seconds"],
            runtime=derive_runtime(tr, timing),
            completion_tokens_total=tr["completion_tokens_total"],
            mean_completion_tokens=tr["mean_completion_tokens"],
            truncation_rate=tr["truncation_rate"],
            finish_reason_counts=tr["finish_reason_counts"] or {},
            num_fewshot=tr["num_fewshot"],
            limit=tr["task_limit"],
            split=tr["split"],
            suites=list(tr["suites"] or []),
            predictions_uri=object_uri(run, tr["predictions_path"]),
            requests_uri=object_uri(run, tr["requests_path"]),
            baseline=cells.get(tr["id"]),
        )
        for tr in trs
    ]
    response = a.RunTaskResultsResponse(
        run_id=run_id,
        baseline=subject.info() if subject else None,
        items=items,
        computed_at=cached.computed_at if cached is not None else now_utc(),
    )
    if subject is not None and cached is None:
        await put_cached(session, key, response)
    return response


# ---------------------------------------------------------------------------
# Suites
# ---------------------------------------------------------------------------


def suite_delta(
    run_trs: Mapping[str, Mapping[Any, Any]],
    base_trs: Mapping[str, Mapping[Any, Any]],
    node: Any,
    loader: SideLoader,
    stats: Any,
    corpus_delta: float | None,
    higher_is_better: bool | None,
) -> a.DeltaStats:
    tasks = []
    for name in leaves(node):
        tr, base = run_trs.get(name), base_trs.get(name)
        if tr is None or base is None or tr["primary_metric"] is None:
            continue
        side_a = loader.side(tr, tr["primary_metric"])
        side_b = loader.side(base, tr["primary_metric"])
        if side_a is not None and side_b is not None:
            tasks.append((name, {"a": side_a, "b": side_b}))
    n_inst = {n: int(tr["num_instances"] or 0) for n, tr in run_trs.items()}
    acc = stratified_pairs(
        tasks,
        [("a", "b")],
        lambda names: leaf_weights(node, n_inst, set(names)),
        alpha=stats.alpha,
        n_boot=stats.n_boot,
        seed=stats.seed,
        higher_is_better=higher_is_better,
    )[("a", "b")]
    return accumulator_delta(acc, stats, corpus_delta, higher_is_better)


def accumulator_delta(
    acc: PairAccumulator, stats: Any, corpus_delta: float | None, higher_is_better: bool | None
) -> a.DeltaStats:
    if not acc.weights:
        return a.DeltaStats(
            delta=corpus_delta,
            ci_low=None,
            ci_high=None,
            p_value=None,
            n_shared=0,
            method="none",
            significant=False,
            improved=improved(corpus_delta, higher_is_better),
            hash_mismatch=acc.hash_mismatch,
            alpha=stats.alpha,
            n_boot=None,
            note="no task has enough shared per-instance scores",
        )
    summary = bootstrap_summary(acc.boot, stats.alpha)
    lo, hi, p = summary if summary else (None, None, None)
    return a.DeltaStats(
        delta=acc.point,
        ci_low=lo,
        ci_high=hi,
        p_value=p,
        n_shared=acc.n_shared,
        method="paired_bootstrap",
        significant=lo is not None and hi is not None and (lo > 0 or hi < 0),
        improved=improved(acc.point, higher_is_better),
        hash_mismatch=acc.hash_mismatch,
        alpha=stats.alpha,
        n_boot=stats.n_boot,
        note=None,
    )


@router.get("/runs/{run_id}/suites", response_model=a.RunSuitesResponse)
async def run_suites(
    run_id: str,
    session: SessionDep,
    user: UserDep,
    stats: StatsDep,
    baseline: str | None = None,
) -> a.RunSuitesResponse:
    await _run_row(session, run_id)
    rows = list(
        (
            await session.execute(
                text("SELECT * FROM suite_results WHERE run_id = :r ORDER BY id"), {"r": run_id}
            )
        )
        .mappings()
        .all()
    )
    ordered = suite_order(rows)
    trs_list = await run_task_results(session, run_id)
    run_trs = latest_by_task(trs_list)
    subject = await resolve_one(session, baseline) if baseline else None
    stored_defs = {
        r["suite_name"]: (r["aggregation"], r["children"], r["description"], r["definition_hash"])
        for r in rows
    }
    defs = {**(await load_defs(session)), **stored_defs}
    key = cache_key(
        "run-suites",
        {"run_id": run_id, "baseline": baseline, **stats.__dict__},
        trs_list + (list(subject.trs.values()) if subject else []),
        definition_hashes([tree(r["suite_name"], defs) for r in ordered], defs),
    )
    cached = None
    if subject is not None:
        cached = await get_cached(session, key, a.RunSuitesResponse)
    loader = SideLoader(session)
    if subject is not None and cached is None:
        for name, tr in run_trs.items():
            base = subject.trs.get(name)
            if base is not None and tr["primary_metric"]:
                loader.want(tr, tr["primary_metric"])
                loader.want(base, tr["primary_metric"])
        await loader.load()
    depth: dict[str, int] = {}
    in_suites: set[str] = set()
    items: list[a.SuiteResultRow] = []
    # Only the baseline cells come from the cache; rows and summaries are always current.
    base_cells: dict[str, a.SuiteBaselineCell] = {}
    if cached is not None:
        base_cells = {i.name: i.baseline for i in cached.items if i.baseline is not None}
    elif subject is not None:
        base_trs = subject.trs

        def build_base_cells() -> None:
            for r in ordered:
                node = tree(r["suite_name"], defs)
                agg = score_suite(node, base_trs)
                corpus = (
                    r["score"] - agg.score
                    if r["score"] is not None and agg.score is not None
                    else None
                )
                delta = suite_delta(
                    run_trs, base_trs, node, loader, stats, corpus, r["higher_is_better"]
                )
                base_cells[r["suite_name"]] = a.SuiteBaselineCell(
                    score=agg.score, stderr=agg.stderr, delta=delta
                )

        # Bootstraps are CPU-bound; a worker thread keeps the event loop serving other requests.
        await asyncio.to_thread(build_base_cells)
    for r in ordered:
        parent = r["parent_suite"]
        depth[r["suite_name"]] = depth.get(parent, -1) + 1 if parent in depth else 0
        node = tree(r["suite_name"], defs)
        names = leaves(node)
        in_suites.update(names)
        base_cell = base_cells.get(r["suite_name"])
        items.append(
            a.SuiteResultRow(
                name=r["suite_name"],
                parent=parent,
                depth=depth[r["suite_name"]],
                aggregation=r["aggregation"],
                description=r["description"],
                children=[a.SuiteChild(**c) for c in r["children"]],
                score=r["score"],
                stderr=r["stderr"],
                num_tasks=int(r["num_tasks"] if r["num_tasks"] is not None else len(names)),
                n_instances=int(r["n_instances"] or 0),
                display_format=r["display_format"],
                higher_is_better=r["higher_is_better"],
                tasks_missing=list(r["tasks_missing"] or []),
                baseline=base_cell,
            )
        )
    response = a.RunSuitesResponse(
        run_id=run_id,
        baseline=subject.info() if subject else None,
        items=items,
        unsuited_tasks=sorted(set(run_trs) - in_suites),
        computed_at=cached.computed_at if cached is not None else now_utc(),
    )
    if subject is not None and cached is None:
        await put_cached(session, key, response)
    return response


# ---------------------------------------------------------------------------
# Instances
# ---------------------------------------------------------------------------


async def _task_result(session: Any, run_id: str, task_result_id: int) -> Mapping[Any, Any]:
    sql = f"""
        SELECT {TR_COLUMNS}, tr.finish_reason_counts, tr.predictions_path, tr.requests_path,
               tv.config
        FROM task_results tr
        JOIN task_variants tv ON tv.task_name = tr.task_name AND tv.task_hash = tr.task_hash
        WHERE tr.id = :i AND tr.run_id = :r
    """
    row = (await session.execute(text(sql), {"i": task_result_id, "r": run_id})).mappings().first()
    if row is None:
        raise not_found(f"Task result {task_result_id} not found in run {run_id}")
    return row


async def _baseline_tr(session: Any, baseline: str, task_name: str) -> Mapping[Any, Any] | None:
    subject = await resolve_one(session, baseline)
    return subject.trs.get(task_name)


@router.get(
    "/runs/{run_id}/task-results/{task_result_id}/instances",
    response_model=a.InstancesResponse,
)
async def run_instances(
    run_id: str,
    task_result_id: int,
    session: SessionDep,
    user: UserDep,
    baseline: str | None = None,
    correct: Annotated[str | None, Query(pattern="^(correct|incorrect|partial)$")] = None,
    threshold: ThresholdQuery = 0.5,
    vs: Annotated[
        str | None, Query(pattern="^(gained|lost|both_right|both_wrong|changed)$")
    ] = None,
    finish_reason: Annotated[list[str], Query()] = [],  # noqa: B006
    len_min: int | None = None,
    len_max: int | None = None,
    score_min: float | None = None,
    score_max: float | None = None,
    has: Annotated[list[str], Query()] = [],  # noqa: B006
    q: str | None = None,
    sort: str = "doc_id",
    cursor: str | None = None,
    limit: int | None = None,
) -> a.InstancesResponse:
    await _run_row(session, run_id)
    tr = await _task_result(session, run_id, task_result_id)
    limit = clamp_limit(limit, default=200, maximum=1000)
    kind = tr["metric_kind"] or "unbounded"
    hib = tr["higher_is_better"]
    bounded = kind in ("binary", "bounded")
    if correct and not bounded:
        raise bad_request("correct is not available for unbounded metrics")
    if vs and not baseline:
        raise bad_request("vs requires baseline")
    if vs and vs != "changed" and not bounded:
        raise bad_request(f"vs={vs} is not available for unbounded metrics")
    if sort not in INSTANCE_SORTS:
        raise bad_request(f"invalid sort {sort!r}")
    base_tr = await _baseline_tr(session, baseline, tr["task_name"]) if baseline else None
    rows, total, next_cursor = await query_instances(
        session,
        task_result_id,
        InstanceFilters(
            baseline_tr_id=base_tr["id"] if base_tr is not None else None,
            baseline_requested=bool(baseline),
            higher_is_better=hib,
            threshold=threshold,
            correct=correct,
            vs=vs,
            finish_reason=finish_reason,
            len_min=len_min,
            len_max=len_max,
            score_min=score_min,
            score_max=score_max,
            has=has,
            q=q,
        ),
        sort,
        cursor,
        limit,
    )

    def corr(score: float | None) -> bool | None:
        if not bounded:
            return None
        if score is None:
            return False
        g = score if hib is not False else 1 - score
        return g >= threshold

    items = []
    for r in rows:
        b = r["b_score"]
        items.append(
            a.InstanceRow(
                native_id=r["native_id"],
                doc_id=r["doc_id"],
                primary_score=r["primary_score"],
                correct=corr(r["primary_score"]),
                metrics=r["metrics"] or {},
                label=r["label"],
                extracted_answer=r["extracted_answer"],
                finish_reason=r["finish_reason"],
                completion_tokens=r["completion_tokens"],
                prompt_tokens=r["prompt_tokens"],
                num_outputs=r["num_outputs"],
                prompt_preview=r["prompt_preview"],
                output_preview=r["output_preview"],
                judge_verdict=r["judge_verdict"],
                has_scoring_error=r["has_scoring_error"],
                has_execution_result=r["has_execution_result"],
                has_judge_result=r["has_judge_result"],
                has_trajectory=r["has_trajectory"],
                baseline_score=b,
                baseline_correct=corr(b) if base_tr is not None else None,
                delta=(r["primary_score"] - b)
                if (b is not None and r["primary_score"] is not None)
                else None,
            )
        )
    return a.InstancesResponse(
        task_result_id=task_result_id,
        baseline_task_result_id=base_tr["id"] if base_tr is not None else None,
        kind=kind,
        threshold=threshold,
        items=items,
        next_cursor=next_cursor,
        total=total,
    )


# ---------------------------------------------------------------------------
# Histograms
# ---------------------------------------------------------------------------


@router.get(
    "/runs/{run_id}/task-results/{task_result_id}/histograms",
    response_model=a.HistogramsResponse,
)
async def run_histograms(
    run_id: str,
    task_result_id: int,
    session: SessionDep,
    user: UserDep,
    baseline: str | None = None,
    bins: Annotated[int, Query(ge=5, le=100)] = 30,
    threshold: ThresholdQuery = 0.5,
) -> a.HistogramsResponse:
    await _run_row(session, run_id)
    tr = await _task_result(session, run_id, task_result_id)
    kind = tr["metric_kind"] or "unbounded"
    binary = kind == "binary"
    rows = (
        await session.execute(
            text(
                "SELECT native_id, primary_score, completion_tokens FROM instance_results "
                "WHERE task_result_id = :i"
            ),
            {"i": task_result_id},
        )
    ).all()
    scores = np.array([np.nan if r[1] is None else r[1] for r in rows], dtype=float)
    score_hist = histogram(scores, bins, binary=binary) if rows else None
    base_hist = None
    delta_hist = None
    if baseline:
        base_tr = await _baseline_tr(session, baseline, tr["task_name"])
        if base_tr is not None:
            brows = (
                await session.execute(
                    text(
                        "SELECT native_id, primary_score FROM instance_results "
                        "WHERE task_result_id = :i"
                    ),
                    {"i": base_tr["id"]},
                )
            ).all()
            bscores = {r[0]: r[1] for r in brows}
            base_vals = np.array(
                [np.nan if v is None else v for v in bscores.values()], dtype=float
            )
            base_hist = histogram(base_vals, bins, binary=binary) if brows else None
            diffs = [
                r[1] - bscores[r[0]]
                for r in rows
                if r[1] is not None and bscores.get(r[0]) is not None
            ]
            delta_hist = histogram(diffs, bins) if diffs else None
    lengths = None
    tokens = np.array([np.nan if r[2] is None else r[2] for r in rows], dtype=float)
    finite = np.isfinite(tokens)
    if finite.any():
        lo, hi = float(tokens[finite].min()), float(tokens[finite].max())
        rng = (lo, hi) if hi > lo else (lo - 0.5, hi + 0.5)
        nb = bins if hi > lo else 1

        def hist(mask: np.ndarray) -> a.Histogram:
            counts, edges = np.histogram(tokens[mask & finite], bins=nb, range=rng)
            return a.Histogram(edges=[float(e) for e in edges], counts=[int(c) for c in counts])

        empty = a.Histogram(edges=[], counts=[])
        if kind in ("binary", "bounded"):
            good = scores if tr["higher_is_better"] is not False else 1 - scores
            with np.errstate(invalid="ignore"):
                is_correct = np.isfinite(scores) & (good >= threshold)
            lengths = a.LengthHistograms(
                correct=hist(is_correct), incorrect=hist(~is_correct), all=hist(finite)
            )
        else:
            lengths = a.LengthHistograms(correct=empty, incorrect=empty, all=hist(finite))
    config = tr["config"] or {}
    max_tokens = (config.get("sampling_params") or {}).get("max_tokens")
    return a.HistogramsResponse(
        task_result_id=task_result_id,
        metric=tr["primary_metric"],
        kind=kind,
        score=a.Histogram(**score_hist) if score_hist else None,
        baseline_score=a.Histogram(**base_hist) if base_hist else None,
        delta=a.Histogram(**delta_hist) if delta_hist else None,
        completion_tokens=lengths,
        finish_reasons=tr["finish_reason_counts"] or {},
        max_tokens=max_tokens if isinstance(max_tokens, int) else None,
    )


# ---------------------------------------------------------------------------
# Instance detail (full records from GCS)
# ---------------------------------------------------------------------------

# Parsed records keyed by (object key, offset, length, version). The version changes when the
# object is re-uploaded (see instance_detail), so a rewritten file is not served from the cache.
# Capped by entries and by the raw bytes they came from, so a few multi-MiB records cannot fill
# the container's memory.
_RECORD_CACHE: OrderedDict[tuple[str, int, int, str], dict[str, Any]] = OrderedDict()
_RECORD_CACHE_MAX = 256
_RECORD_CACHE_MAX_BYTES = 64 * 1024 * 1024
_record_cache_bytes = 0


def _cache_record(key: tuple[str, int, int, str], record: dict[str, Any]) -> None:
    global _record_cache_bytes
    length = key[2]
    if length > _RECORD_CACHE_MAX_BYTES // 8 or key in _RECORD_CACHE:
        return
    _RECORD_CACHE[key] = record
    _record_cache_bytes += length
    while len(_RECORD_CACHE) > _RECORD_CACHE_MAX or _record_cache_bytes > _RECORD_CACHE_MAX_BYTES:
        old, _ = _RECORD_CACHE.popitem(last=False)
        _record_cache_bytes -= old[2]


async def read_record(
    storage: Any,
    run: Mapping[Any, Any],
    path: str | None,
    offset: int | None,
    length: int | None,
    version: str = "",
    cacheable: bool = True,
) -> tuple[dict[str, Any] | None, str | None]:
    """One JSON record of a run file by byte range.

    ``version`` identifies the object's content (for the cache key); ``cacheable`` is False
    while a re-upload of the object may be in progress.
    """
    if not path:
        return None, "no file recorded for this task"
    if offset is None or length is None:
        return None, "byte offsets were not recorded for this instance"
    if offset < 0 or length <= 0:
        # A zero-length range would become an invalid Range header, which GCS may answer with
        # the whole object.
        return None, "the recorded byte range is empty"
    if length > MAX_RECORD_BYTES:
        return None, "record is larger than 25 MiB"
    _, prefix_key = split_gs_uri(run["gcs_prefix"])
    key = prefix_key + path
    cache_key_ = (key, offset, length, version)
    if cache_key_ in _RECORD_CACHE:
        _RECORD_CACHE.move_to_end(cache_key_)
        return _RECORD_CACHE[cache_key_], None
    try:
        data = await storage.read_range(key, offset, length)
    except FileNotFoundError:
        return None, f"{path} is not in storage"
    try:
        record = orjson.loads(data)
    except orjson.JSONDecodeError:
        return None, f"could not parse the record at byte {offset} of {path}"
    if not isinstance(record, dict):
        record = {"value": record}
    if cacheable:
        _cache_record(cache_key_, record)
    return record, None


@router.get("/instances/detail", response_model=a.InstanceDetailResponse)
async def instance_detail(
    task_result_id: int,
    native_id: str,
    session: SessionDep,
    user: UserDep,
    storage: StorageDep,
) -> a.InstanceDetailResponse:
    row = (
        (
            await session.execute(
                text(
                    "SELECT ir.*, tr.run_id, tr.task_name, tr.predictions_path, tr.requests_path, "
                    "tr.updated_at AS tr_updated_at, pa.md5_b64 AS pred_md5, "
                    "pa.uploaded AS pred_uploaded, ra.md5_b64 AS req_md5, "
                    "ra.uploaded AS req_uploaded "
                    "FROM instance_results ir JOIN task_results tr ON tr.id = ir.task_result_id "
                    "LEFT JOIN artifacts pa "
                    "ON pa.run_id = tr.run_id AND pa.path = tr.predictions_path "
                    "LEFT JOIN artifacts ra "
                    "ON ra.run_id = tr.run_id AND ra.path = tr.requests_path "
                    "WHERE ir.task_result_id = :i AND ir.native_id = :n"
                ),
                {"i": task_result_id, "n": native_id},
            )
        )
        .mappings()
        .first()
    )
    if row is None:
        raise not_found(f"Instance {native_id} not found in task result {task_result_id}")
    run = await _run_row(session, row["run_id"])
    # A re-upload rewrites the files and the task result; the artifact MD5 and the task
    # result's updated_at identify the content. While a signed artifact is not yet verified as
    # uploaded (between signing and complete), records are not cached.
    updated = row["tr_updated_at"].isoformat()
    prediction, why_p = await read_record(
        storage,
        run,
        row["predictions_path"],
        row["pred_offset"],
        row["pred_length"],
        version=f"{updated}|{row['pred_md5']}",
        cacheable=row["pred_uploaded"] is not False,
    )
    request, why_r = await read_record(
        storage,
        run,
        row["requests_path"],
        row["req_offset"],
        row["req_length"],
        version=f"{updated}|{row['req_md5']}",
        cacheable=row["req_uploaded"] is not False,
    )
    reasons = [f"prediction: {why_p}" if why_p else None, f"request: {why_r}" if why_r else None]
    reason = "; ".join(r for r in reasons if r) or None
    return a.InstanceDetailResponse(
        task_result_id=task_result_id,
        run_id=row["run_id"],
        task_name=row["task_name"],
        native_id=native_id,
        doc_id=row["doc_id"],
        primary_score=row["primary_score"],
        metrics=row["metrics"] or {},
        prediction=prediction,
        request=request,
        predictions_uri=object_uri(run, row["predictions_path"]),
        requests_uri=object_uri(run, row["requests_path"]),
        unavailable_reason=reason,
    )


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def _downsample(rows: list[Any], n: int) -> list[Any]:
    if len(rows) <= n:
        return rows
    idx = np.linspace(0, len(rows) - 1, n).round().astype(int)
    return [rows[i] for i in sorted(set(idx.tolist()))]


def _quantile_box(q: Mapping[Any, Any] | None) -> a.BoxStats | None:
    if not q:
        return None
    # Older clients sent quantiles only; the median stands in for a missing mean.
    mean = q.get("mean")
    return a.BoxStats(
        p5=q["p5"],
        p25=q["p25"],
        p50=q["p50"],
        p75=q["p75"],
        p95=q["p95"],
        mean=q["p50"] if mean is None else mean,
        n=q["n"],
    )


def _gpu_values(batches: Sequence[Mapping[Any, Any]], key: str) -> list[float]:
    values = [(b["gpu_summary"] or {}).get(key) for b in batches]
    return [float(v) for v in values if v is not None]


async def inference_kpis(
    session: Any, run: Mapping[Any, Any]
) -> tuple[a.InferenceKpis, list[Mapping[Any, Any]], Mapping[Any, Any] | None]:
    batches = list(
        (
            await session.execute(
                text("SELECT * FROM inference_batches WHERE run_id = :r ORDER BY seq"),
                {"r": run["run_id"]},
            )
        )
        .mappings()
        .all()
    )
    info = (
        (
            await session.execute(
                text("SELECT * FROM run_inference WHERE run_id = :r"), {"r": run["run_id"]}
            )
        )
        .mappings()
        .first()
    )
    env = run.get("environment") or {}
    beaker = run.get("beaker") or {}
    gpus = env.get("gpu_count") or beaker.get("gpu_count")
    duration = run["duration_seconds"]
    if batches:
        ts = [b["ts"] for b in batches]
        wall = (max(ts) - min(ts)).total_seconds() + float(batches[-1]["wall_clock_time_s"])
        total_requests = sum(b["total_requests"] for b in batches)
        failed = sum(b["failed_requests"] for b in batches)
        prompt = sum(b["total_prompt_tokens"] for b in batches)
        completion = sum(b["total_completion_tokens"] for b in batches)
        weighted = sum(b["mean_latency_s"] * b["total_requests"] for b in batches)
        mean_latency = weighted / total_requests if total_requests else None
        utils = _gpu_values(batches, "avg_utilization_pct")
        mems = _gpu_values(batches, "max_memory_used_mb")
    else:
        wall = duration
        total_requests = failed = prompt = completion = None
        mean_latency = None
        utils, mems = [], []
    ttft = ((info or {}).get("request_latency") or {}).get("ttft_s") if info else None
    kpis = a.InferenceKpis(
        wall_clock_s=wall,
        total_requests=total_requests,
        failed_requests=failed,
        prompt_tokens=prompt,
        completion_tokens=completion,
        output_tokens_per_second=(completion / wall) if completion and wall else None,
        mean_latency_s=mean_latency,
        ttft_p50_s=ttft["p50"] if ttft else None,
        ttft_p95_s=ttft["p95"] if ttft else None,
        gpu_utilization_mean_pct=float(np.mean(utils)) if utils else None,
        gpu_memory_max_mb=float(max(mems)) if mems else None,
        gpu_hours=(duration / 3600 * gpus) if duration and gpus else None,
    )
    return kpis, batches, info


@router.get("/runs/{run_id}/inference", response_model=a.InferenceResponse)
async def run_inference(
    run_id: str, session: SessionDep, user: UserDep, baseline: str | None = None
) -> a.InferenceResponse:
    run = await _run_row(session, run_id)
    kpis, batches, info = await inference_kpis(session, run)
    baseline_kpis = None
    if baseline:
        subject = await resolve_one(session, baseline)
        run_ids = subject.run_ids
        if run_ids:
            latest = max(
                (
                    await fetch_run_rows(
                        session,
                        "r.run_id = ANY(:ids)",
                        {"ids": run_ids},
                        extra_columns=", r.environment",
                    )
                ),
                key=lambda r: r["created_at"],
            )
            baseline_kpis, _, _ = await inference_kpis(session, latest)
    per_task_rows = (
        (
            await session.execute(
                text(
                    """
                    SELECT tr.id, tr.task_name, tr.instances_stored, tr.completion_tokens_total,
                           tr.prompt_tokens_total, tr.mean_completion_tokens, tr.truncation_rate,
                           tr.duration_seconds,
                           percentile_cont(ARRAY[0.05, 0.25, 0.5, 0.75, 0.95])
                               WITHIN GROUP (ORDER BY ir.completion_tokens) AS q,
                           avg(ir.completion_tokens) AS mean_tokens,
                           count(ir.completion_tokens) AS n_tokens
                    FROM task_results tr
                    LEFT JOIN instance_results ir ON ir.task_result_id = tr.id
                    WHERE tr.run_id = :r
                    GROUP BY tr.id ORDER BY tr.id
                    """
                ),
                {"r": run_id},
            )
        )
        .mappings()
        .all()
    )
    per_task = []
    for r in per_task_rows:
        box = None
        if r["n_tokens"]:
            q = r["q"]
            box = a.BoxStats(
                p5=q[0], p25=q[1], p50=q[2], p75=q[3], p95=q[4],
                mean=float(r["mean_tokens"]), n=int(r["n_tokens"]),
            )  # fmt: skip
        per_task.append(
            a.InferenceTaskRow(
                task_name=r["task_name"],
                task_result_id=r["id"],
                instances=int(r["instances_stored"] or 0),
                completion_tokens=r["completion_tokens_total"],
                prompt_tokens=r["prompt_tokens_total"],
                mean_completion_tokens=r["mean_completion_tokens"],
                truncation_rate=r["truncation_rate"],
                duration_seconds=r["duration_seconds"],
                completion_tokens_box=box,
            )
        )
    started = run["started_at"] or (batches[0]["ts"] if batches else None)
    by_task: dict[str, list[float]] = {}
    for b in batches:
        if b["task_name"]:
            by_task.setdefault(b["task_name"], []).append(b["mean_latency_s"])
    request_latency = None
    if info and info["request_latency"]:
        rl = info["request_latency"]
        request_latency = a.InferenceResponseRequestLatency(
            end_to_end_s=_quantile_box(rl.get("end_to_end_s")),
            ttft_s=_quantile_box(rl.get("ttft_s")),
            tpot_s=_quantile_box(rl.get("tpot_s")),
        )
    batch_box = box_stats([b["mean_latency_s"] for b in batches])
    return a.InferenceResponse(
        run_id=run_id,
        available=bool(batches) or info is not None,
        kpis=kpis,
        baseline_kpis=baseline_kpis,
        per_task=per_task,
        batch_latency=a.BoxStats(**batch_box) if batch_box else None,
        batch_latency_by_task=[
            a.InferenceResponseBatchLatencyByTaskItem(task_name=name, box=a.BoxStats(**box))
            for name, vals in sorted(by_task.items())
            if (box := box_stats(vals)) is not None
        ],
        request_latency=request_latency,
        batches=[
            a.InferenceBatchOut(
                seq=b["seq"],
                t=(b["ts"] - started).total_seconds() if started else 0.0,
                task_name=b["task_name"],
                total_requests=b["total_requests"],
                failed_requests=b["failed_requests"],
                total_prompt_tokens=b["total_prompt_tokens"],
                total_completion_tokens=b["total_completion_tokens"],
                wall_clock_time_s=b["wall_clock_time_s"],
                output_tokens_per_second=b["output_tokens_per_second"],
                mean_latency_s=b["mean_latency_s"],
            )
            for b in _downsample(batches, MAX_BATCHES)
        ],
        series=[a.SeriesOut(**s) for s in (info["series"] if info else [])],
        gpu_devices=[
            a.InferenceResponseGpuDevicesItem(**d) for d in (info["gpu_devices"] if info else [])
        ],
        started_at=started,
    )


# ---------------------------------------------------------------------------
# Configs, artifacts, baseline suggestions, download signing
# ---------------------------------------------------------------------------


@router.get("/runs/{run_id}/configs", response_model=a.RunConfigsResponse)
async def run_configs(run_id: str, session: SessionDep, user: UserDep) -> a.RunConfigsResponse:
    run = await _run_row(session, run_id)
    rows = (
        (
            await session.execute(
                text(
                    "SELECT tr.id, tr.task_name, tr.task_hash, tv.config FROM task_results tr "
                    "JOIN task_variants tv ON tv.task_name = tr.task_name "
                    "AND tv.task_hash = tr.task_hash WHERE tr.run_id = :r ORDER BY tr.id"
                ),
                {"r": run_id},
            )
        )
        .mappings()
        .all()
    )
    env = run["environment"] or {}
    return a.RunConfigsResponse(
        run_id=run_id,
        model_config=run["m_provider_config"] or {},
        harness_config=run["harness_config"],
        environment=a.EnvironmentDetail(
            **{k: env.get(k) for k in a.EnvironmentDetail.model_fields if k != "packages"},
            packages=env.get("packages") or {},
        ),
        argv=list(run["argv"] or []),
        task_configs=[
            a.TaskConfigEntry(
                task_result_id=r["id"],
                task_name=r["task_name"],
                task_hash=r["task_hash"],
                config=r["config"] or {},
            )
            for r in rows
        ],
    )


@router.get("/runs/{run_id}/artifacts", response_model=a.ArtifactsResponse)
async def run_artifacts(run_id: str, session: SessionDep, user: UserDep) -> a.ArtifactsResponse:
    run = await _run_row(session, run_id)
    rows = (
        (
            await session.execute(
                text("SELECT * FROM artifacts WHERE run_id = :r ORDER BY kind, path"),
                {"r": run_id},
            )
        )
        .mappings()
        .all()
    )
    beaker = run["beaker"]
    return a.ArtifactsResponse(
        run_id=run_id,
        gcs_prefix=run["gcs_prefix"],
        gcs_console=gcs_console_url(run["gcs_prefix"]),
        items=[
            a.ArtifactRow(
                path=r["path"],
                kind=r["kind"],
                task_name=r["task_name"],
                size_bytes=r["size_bytes"],
                uploaded=r["uploaded"],
                updated_at=r["updated_at"],
                gs_uri=run["gcs_prefix"] + r["path"],
                console_url=gcs_console_url(run["gcs_prefix"] + r["path"]),
            )
            for r in rows
        ],
        beaker=a.BeakerDetail(**{k: beaker.get(k) for k in a.BeakerDetail.model_fields})
        if beaker
        else None,
        links=a.RunLinks(**run_links(run)),
    )


@router.get("/runs/{run_id}/baseline-suggestions", response_model=a.BaselineSuggestionsResponse)
async def baseline_suggestions(
    run_id: str, session: SessionDep, user: UserDep
) -> a.BaselineSuggestionsResponse:
    run = await _run_row(session, run_id)
    items: list[a.BaselineSuggestion] = []
    if run["m_step"] is not None:
        prev = (
            await session.execute(
                text(
                    "SELECT m.model_id FROM models m WHERE m.series = :s AND "
                    "m.settings_hash = :h AND m.step < :step AND EXISTS (SELECT 1 FROM "
                    "task_results tr WHERE tr.model_id = m.model_id "
                    "AND tr.finalized_at IS NOT NULL) ORDER BY m.step DESC LIMIT 1"
                ),
                {"s": run["m_series"], "h": run["m_settings_hash"], "step": run["m_step"]},
            )
        ).scalar()
        if prev:
            model = (await fetch_models(session, [prev]))[prev]
            items.append(
                a.BaselineSuggestion(
                    subject=f"m:{prev}",
                    reason="previous_checkpoint",
                    label=base_label(model),
                    model=model_ref(model),
                    run=None,
                )
            )
    if run["experiment_group"]:
        same_group = await fetch_run_rows(
            session,
            "r.experiment_group = :g AND r.run_id <> :r AND "
            "(r.upload_state = 'complete' OR r.status = 'running')",
            {"g": run["experiment_group"], "r": run_id},
            limit=3,
        )
        for other in same_group:
            items.append(_run_suggestion(other, "same_group"))
    prev_run = await fetch_run_rows(
        session,
        """
        r.model_id = :m AND r.run_id <> :r AND r.created_at < :c AND r.upload_state = 'complete'
        AND (SELECT count(DISTINCT t.task_name) FROM task_results t
             WHERE t.run_id = r.run_id AND t.task_name IN
                 (SELECT task_name FROM task_results WHERE run_id = :r)) * 2
            >= GREATEST((SELECT count(DISTINCT task_name) FROM task_results WHERE run_id = :r), 1)
        """,
        {"m": run["m_model_id"], "r": run_id, "c": run["created_at"]},
        limit=1,
    )
    for other in prev_run:
        items.append(_run_suggestion(other, "previous_run_same_tasks"))
    return a.BaselineSuggestionsResponse(items=items)


def _run_suggestion(
    row: Mapping[Any, Any], reason: Literal["same_group", "previous_run_same_tasks"]
) -> a.BaselineSuggestion:
    summary = run_summary(row)
    label = base_label(row)
    if row["experiment_name"]:
        label += f" · {row['experiment_name']}"
    return a.BaselineSuggestion(
        subject=f"r:{row['run_id']}",
        reason=reason,
        label=label,
        model=summary.model,
        run=summary,
    )


@router.post("/artifacts/sign", response_model=a.SignDownloadResponse)
async def sign_download(
    body: a.SignDownloadRequest,
    session: SessionDep,
    user: UserDep,
    settings: SettingsDep,
    storage: StorageDep,
) -> a.SignDownloadResponse:
    bucket, key = split_gs_uri(body.gs_uri)
    prefix = f"{settings.results_prefix}runs/"
    if (
        not body.gs_uri.startswith("gs://")
        or bucket != settings.results_bucket
        or not key.startswith(prefix)
    ):
        raise bad_request(
            f"only objects under gs://{settings.results_bucket}/{prefix} can be signed"
        )
    run_id, _, rest = key.removeprefix(prefix).partition("/")
    if not rest or ".." in rest.split("/"):
        raise bad_request("gs_uri must name an object inside a run")
    exists = (
        await session.execute(text("SELECT 1 FROM runs WHERE run_id = :r"), {"r": run_id})
    ).first()
    if exists is None:
        raise not_found(f"Run {run_id} not found")
    signed = await storage.sign_download(key)
    return a.SignDownloadResponse(url=signed.url, expires_at=signed.expires_at)
