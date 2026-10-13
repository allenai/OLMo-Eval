"""/api/compare: matrix, pairwise, contingency and instance agreement (spec 4.5, 5)."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from fastapi import APIRouter
from sqlalchemy import text

from olmo_eval_api.errors import bad_request
from olmo_eval_api.schemas import api as a
from olmo_eval_api.services.cache import cache_key, get_cached, put_cached
from olmo_eval_api.services.common import now_utc
from olmo_eval_api.services.queries import decode_offset_cursor, encode_cursor
from olmo_eval_api.services.read_suites import (
    SuiteDefs,
    definition_hashes,
    load_defs,
    score_suite,
    tree,
)
from olmo_eval_api.services.runtime import runtimes_for
from olmo_eval_api.services.subjects import (
    SideLoader,
    Subject,
    meta_for,
    resolve_for_compare,
    validate_subject_key,
)
from olmo_eval_api.services.suites import SuiteNode, leaf_weights, leaves
from olmo_eval_api.stats.bootstrap import align, union_keys
from olmo_eval_api.stats.mde import mde80
from olmo_eval_api.stats.paired import (
    MAX_PAIR_RESAMPLES,
    MAX_PAIRWISE_WORK,
    Delta,
    Side,
    compare_pairs,
    contingency,
    corpus_delta,
    correctness,
    improved,
    pair_stats,
    resampled_win_rate,
    stratified_pairs,
)

from .deps import SessionDep, UserDep, check_stats, check_threshold
from .run_tabs import accumulator_delta

router = APIRouter()


@dataclass
class RowSpec:
    key: str
    kind: str  # "task" | "suite"
    name: str
    depth: int
    parent: str | None
    aggregation: str | None = None
    node: SuiteNode | None = None


def _suite_rows(node: SuiteNode, depth: int, parent: str | None, out: list[RowSpec]) -> None:
    out.append(
        RowSpec(f"suite:{node.name}", "suite", node.name, depth, parent, node.aggregation, node)
    )
    for child in node.children:
        if child.type == "suite":
            _suite_rows(child, depth + 1, node.name, out)
        else:
            out.append(RowSpec(f"task:{child.name}", "task", child.name, depth + 1, node.name))


async def resolve_scope(
    session: Any, scope: str, subjects: Sequence[Subject]
) -> tuple[list[RowSpec], SuiteNode | None, SuiteDefs | None]:
    if scope.startswith("task:"):
        name = scope.removeprefix("task:")
        return [RowSpec(scope, "task", name, 0, None)], None, None
    if scope.startswith("suite:"):
        defs = await load_defs(session)
        name = scope.removeprefix("suite:")
        if name not in defs:
            raise bad_request(f"unknown suite {name!r}")
        try:
            node = tree(name, defs)
        except ValueError as exc:
            raise bad_request(str(exc)) from exc
        rows: list[RowSpec] = []
        _suite_rows(node, 0, None, rows)
        return rows, node, defs
    if scope in ("all", "shared"):
        sets = [set(s.trs) for s in subjects]
        names = set.union(*sets) if scope == "all" else set.intersection(*sets)
        return [RowSpec(f"task:{n}", "task", n, 0, None) for n in sorted(names)], None, None
    raise bad_request("scope must be all, shared, suite:<name> or task:<name>")


def scope_hashes(node: SuiteNode | None, defs: SuiteDefs | None) -> dict[str, str | None]:
    """Cache-key part for a suite scope: the definition hash of every suite in its tree."""
    return definition_hashes([node], defs) if node is not None and defs is not None else {}


def live_subjects(
    cached: Sequence[a.SubjectInfo], by_key: Mapping[str, Subject]
) -> list[a.SubjectInfo]:
    """Current summaries (tags, status, labels) in the cached order; the cache key does not
    cover them."""
    return [by_key[x.key].info() for x in cached]


def row_metric(
    task: str, metric: str, subjects: Sequence[Subject], baseline: Subject | None
) -> str | None:
    if metric != "primary":
        return metric
    if baseline is not None and task in baseline.trs:
        return baseline.trs[task]["primary_metric"]
    for s in subjects:
        if task in s.trs:
            return s.trs[task]["primary_metric"]
    return None


def row_meta(
    task: str, key: str | None, subjects: Sequence[Subject], baseline: Subject | None
) -> dict:
    order = ([baseline] if baseline is not None else []) + list(subjects)
    for s in order:
        tr = s.trs.get(task)
        if tr is not None:
            meta = meta_for(tr, key)
            if meta:
                return meta
    return {}


def metric_kind_literal(kind: str | None) -> a.MetricKind:
    if kind == "binary":
        return "binary"
    if kind == "bounded":
        return "bounded"
    return "unbounded"


def to_meta(meta: Mapping[Any, Any]) -> a.MetricMeta | None:
    if not meta:
        return None
    return a.MetricMeta(
        higher_is_better=meta.get("higher_is_better"),
        display_format=meta.get("display_format") or "raw",
        unit=meta.get("unit"),
        kind=meta.get("kind") or "unbounded",
    )


def side_shared_score(side: Side, keys: np.ndarray) -> float | None:
    assert side.keys is not None and side.values is not None
    vals = align(keys, side.keys, side.scaled)
    vals = vals[np.isfinite(vals)]
    return float(vals.mean()) if vals.size else None


def common_keys(sides: Sequence[Side | None]) -> np.ndarray | None:
    if not sides or any(s is None or not s.pairable for s in sides):
        return None
    keys: np.ndarray | None = None
    for s in sides:
        assert s is not None and s.keys is not None and s.values is not None
        present = s.keys[np.isfinite(s.values)]
        keys = present if keys is None else np.intersect1d(keys, present, assume_unique=True)
    return keys


# ---------------------------------------------------------------------------
# Matrix
# ---------------------------------------------------------------------------


@router.post("/compare/matrix", response_model=a.MatrixResponse)
async def compare_matrix(
    body: a.MatrixRequest, session: SessionDep, user: UserDep
) -> a.MatrixResponse:
    stats = check_stats(body.alpha, body.n_boot, body.seed, None)
    keys = list(dict.fromkeys(body.subjects))
    if body.baseline:
        validate_subject_key(body.baseline)
        if body.baseline not in keys:
            keys.append(body.baseline)
    subjects = await resolve_for_compare(session, keys, body.group)
    by_key = {s.key: s for s in subjects}
    baseline = by_key.get(body.baseline) if body.baseline else None
    shared_only = bool(body.shared_only)
    request = body.model_dump(mode="json")
    used = [tr for s in subjects for tr in s.trs.values()]
    rows, node, defs = await resolve_scope(session, body.scope, subjects)
    if is_runtime_metric(body.metric):
        # No bootstraps, so nothing worth caching.
        return await _runtime_matrix(session, body, subjects, baseline, rows, stats.alpha)
    ckey = cache_key("matrix", request, used, scope_hashes(node, defs))
    cached = await get_cached(session, ckey, a.MatrixResponse)
    if cached is not None:
        return cached.model_copy(update={"subjects": live_subjects(cached.subjects, by_key)})

    task_names = list(dict.fromkeys(r.name for r in rows if r.kind == "task"))
    metric_keys = {t: row_metric(t, body.metric, subjects, baseline) for t in task_names}
    loader = SideLoader(session)
    for t in task_names:
        for s in subjects:
            loader.want(s.trs.get(t), metric_keys[t])
    await loader.load()
    sides: dict[str, dict[str, Side | None]] = {
        t: {s.key: loader.side(s.trs.get(t), metric_keys[t]) for s in subjects} for t in task_names
    }

    out_rows: list[a.MatrixRow] = []
    ses: list[float | None] = []
    coverage = {"complete": 0, "missing": 0, "failed": 0, "hash_mismatch": 0}

    def build_rows() -> None:
        for spec in rows:
            if spec.kind == "task":
                out_rows.append(
                    _matrix_task_row(
                        spec,
                        subjects,
                        baseline,
                        sides[spec.name],
                        metric_keys[spec.name],
                        body,
                        stats,
                        shared_only,
                        ses,
                        coverage,
                    )
                )
            else:
                assert spec.node is not None
                out_rows.append(_matrix_suite_row(spec, subjects, baseline, sides, body, stats))

    # Bootstraps are CPU-bound; a worker thread keeps the event loop serving other requests.
    await asyncio.to_thread(build_rows)
    response = a.MatrixResponse(
        subjects=[s.info() for s in subjects],
        baseline=body.baseline,
        scope=body.scope,
        metric=body.metric,
        rows=out_rows,
        coverage=a.Coverage(
            n_subjects=len(subjects),
            n_tasks=len(task_names),
            complete=coverage["complete"],
            missing=coverage["missing"],
            failed=coverage["failed"],
            hash_mismatch=coverage["hash_mismatch"],
        ),
        mde80=mde80(ses, stats.alpha),
        alpha=stats.alpha,
        computed_at=now_utc(),
    )
    await put_cached(session, ckey, response)
    return response


def _matrix_task_row(
    spec: RowSpec,
    subjects: Sequence[Subject],
    baseline: Subject | None,
    sides: Mapping[str, Side | None],
    metric_key: str | None,
    body: a.MatrixRequest,
    stats: Any,
    shared_only: bool,
    ses: list[float | None],
    coverage: dict[str, int],
) -> a.MatrixRow:
    meta = row_meta(spec.name, metric_key, subjects, baseline)
    hib = meta.get("higher_is_better")
    base_side = sides.get(baseline.key) if baseline is not None else None
    deltas: dict[str, Any] = {}
    if baseline is not None and base_side is not None:
        others: list[tuple[str, Side]] = [
            (s.key, sd)
            for s in subjects
            if s.key != baseline.key and (sd := sides.get(s.key)) is not None
        ]
        results = compare_pairs(
            [(sd, base_side) for _, sd in others],
            task_name=spec.name,
            alpha=stats.alpha,
            n_boot=stats.n_boot,
            seed=stats.seed,
            higher_is_better=hib,
        )
        for (key, _), d in zip(others, results, strict=True):
            deltas[key] = d
            if d.method == "paired_bootstrap":
                ses.append(d.se)
    keys = common_keys([sides.get(s.key) for s in subjects]) if shared_only else None
    ref_hash = None
    ref = base_side or next((sd for sd in sides.values() if sd is not None), None)
    if ref is not None:
        ref_hash = ref.task_hash
    cells = []
    for s in subjects:
        tr = s.trs.get(spec.name)
        side = sides.get(s.key)
        if tr is None or (side is None and tr["error"] is None):
            coverage["missing"] += 1
            cells.append(_empty_cell("missing", tr))
            continue
        if side is None or (tr["error"] and side.score is None):
            coverage["failed"] += 1
            cells.append(_empty_cell("failed", tr))
            continue
        coverage["complete"] += 1
        if ref_hash and side.task_hash and side.task_hash != ref_hash:
            coverage["hash_mismatch"] += 1
        score = side.score
        if keys is not None and keys.size:
            score = side_shared_score(side, keys)
        d = deltas.get(s.key)
        cells.append(
            a.MatrixCell(
                status="failed" if tr["error"] else "ok",
                score=score,
                stderr=side.stderr,
                n=int(tr["num_instances"] or 0),
                task_result_id=tr["id"],
                run_id=tr["run_id"],
                task_hash=tr["task_hash"],
                delta=a.DeltaStats(**d.as_dict()) if d is not None else None,
                children_missing=0,
            )
        )
    return a.MatrixRow(
        key=spec.key,
        kind="task",
        name=spec.name,
        depth=spec.depth,
        parent=spec.parent,
        aggregation=None,
        metric_key=metric_key,
        meta=to_meta(meta),
        cells=cells,
    )


def _empty_cell(status: a.CellStatus, tr: Mapping[Any, Any] | None) -> a.MatrixCell:
    return a.MatrixCell(
        status=status,
        score=None,
        stderr=None,
        n=None,
        task_result_id=tr["id"] if tr is not None else None,
        run_id=tr["run_id"] if tr is not None else None,
        task_hash=tr["task_hash"] if tr is not None else None,
        delta=None,
        children_missing=0,
    )


def _matrix_suite_row(
    spec: RowSpec,
    subjects: Sequence[Subject],
    baseline: Subject | None,
    sides: Mapping[str, Mapping[str, Side | None]],
    body: a.MatrixRequest,
    stats: Any,
) -> a.MatrixRow:
    node = spec.node
    assert node is not None
    names = leaves(node)
    results = {}
    for s in subjects:
        scores: dict[str, float | None] = {}
        stderrs: dict[str, float | None] = {}
        for n in names:
            if n in s.trs:
                sd = sides.get(n, {}).get(s.key)
                scores[n] = sd.score if sd is not None else None
                stderrs[n] = sd.stderr if sd is not None else None
        results[s.key] = score_suite(node, s.trs, scores, stderrs)
    hib_set = {r.higher_is_better for r in results.values() if r.present}
    hib = hib_set.pop() if len(hib_set) == 1 else None
    deltas: dict[str, a.DeltaStats] = {}
    if baseline is not None:
        others = [s.key for s in subjects if s.key != baseline.key]
        tasks = [(n, {k: v for k, v in sides.get(n, {}).items() if v is not None}) for n in names]
        n_inst = {n: int(baseline.trs[n]["num_instances"] or 0) for n in names if n in baseline.trs}
        acc = stratified_pairs(
            tasks,
            [(k, baseline.key) for k in others],
            lambda q: leaf_weights(node, n_inst, set(q)),
            alpha=stats.alpha,
            n_boot=stats.n_boot,
            seed=stats.seed,
            higher_is_better=hib,
        )
        base_score = results[baseline.key].score
        for k in others:
            sc = results[k].score
            corpus = sc - base_score if sc is not None and base_score is not None else None
            deltas[k] = accumulator_delta(acc[(k, baseline.key)], stats, corpus, hib)
    cells = []
    for s in subjects:
        r = results[s.key]
        status: a.CellStatus = "missing" if r.present == 0 else ("partial" if r.missing else "ok")
        cells.append(
            a.MatrixCell(
                status=status,
                score=r.score,
                stderr=r.stderr,
                n=r.n_instances if r.present else None,
                task_result_id=None,
                run_id=s.run["run_id"] if s.run is not None else None,
                task_hash=None,
                delta=deltas.get(s.key),
                children_missing=r.missing,
            )
        )
    any_result = next((r for r in results.values() if r.present), None)
    meta = (
        a.MetricMeta(
            higher_is_better=hib,
            display_format=any_result.display_format,
            unit=None,
            kind="bounded" if any_result.display_format == "percent" else "unbounded",
        )
        if any_result
        else None
    )
    return a.MatrixRow(
        key=spec.key,
        kind="suite",
        name=spec.name,
        depth=spec.depth,
        parent=spec.parent,
        aggregation=spec.aggregation,
        metric_key=None,
        meta=meta,
        cells=cells,
    )


# ---------------------------------------------------------------------------
# Runtime matrix
# ---------------------------------------------------------------------------

RUNTIME_METRICS = {"runtime:inference": "inference", "runtime:with_startup": "with_startup"}
RUNTIME_META = a.MetricMeta(
    higher_is_better=False, display_format="raw", unit="s", kind="unbounded"
)


def is_runtime_metric(metric: str) -> bool:
    if not metric.startswith("runtime:"):
        return False
    if metric not in RUNTIME_METRICS:
        raise bad_request(f"unknown runtime metric {metric!r}; use {' or '.join(RUNTIME_METRICS)}")
    return True


def reject_runtime_metric(metric: str) -> None:
    """Runtime has one value per task result and no instances, so nothing to pair."""
    if metric.startswith("runtime:"):
        raise bad_request(
            f"metric {metric!r} is not available here: runtime has no per-instance values, so "
            "pairwise and instance-level comparisons do not apply; use the matrix"
        )


def _runtime_delta(
    value: float | None, base: float | None, alpha: float, hash_mismatch: bool
) -> a.DeltaStats:
    # A plain difference: there is one runtime per task result, so no confidence interval.
    d = corpus_delta(value, base)
    return a.DeltaStats(
        **Delta(
            delta=d,
            method="none",
            improved=improved(d, False),
            hash_mismatch=hash_mismatch,
            alpha=alpha,
            note="runtime difference in seconds; no confidence interval",
        ).as_dict()
    )


async def _runtime_matrix(
    session: Any,
    body: a.MatrixRequest,
    subjects: Sequence[Subject],
    baseline: Subject | None,
    rows: Sequence[RowSpec],
    alpha: float,
) -> a.MatrixResponse:
    """Matrix of task runtimes in seconds (lower is better).

    A suite cell is the sum of its tasks' inference seconds; with startup, it adds the largest
    startup among those tasks' runs once (about what running the suite in one job takes).
    """
    mode = RUNTIME_METRICS[body.metric]
    names = {r.name for r in rows if r.kind == "task"}
    for r in rows:
        if r.node is not None:
            names.update(leaves(r.node))
    trs = [tr for s in subjects for n, tr in s.trs.items() if n in names]
    runtimes = await runtimes_for(session, trs)

    def task_value(tr: Mapping[Any, Any] | None) -> float | None:
        if tr is None:
            return None
        rt = runtimes[int(tr["id"])]
        return rt.inference_seconds if mode == "inference" else rt.with_startup_seconds

    def suite_value(s: Subject, leaf_names: Sequence[str]) -> tuple[float | None, int, int]:
        """(seconds, tasks with a runtime, tasks without one)."""
        known = [runtimes[int(s.trs[n]["id"])] for n in leaf_names if n in s.trs]
        known = [rt for rt in known if rt.inference_seconds is not None]
        if not known:
            return None, 0, len(leaf_names)
        total = sum(rt.inference_seconds or 0.0 for rt in known)
        if mode == "with_startup":
            startups = [rt.startup_seconds for rt in known if rt.startup_seconds is not None]
            if not startups:
                return None, 0, len(leaf_names)
            total += max(startups)
        return total, len(known), len(leaf_names) - len(known)

    coverage = {"complete": 0, "missing": 0, "failed": 0, "hash_mismatch": 0}
    task_names = list(dict.fromkeys(r.name for r in rows if r.kind == "task"))
    out_rows: list[a.MatrixRow] = []
    for spec in rows:
        cells: list[a.MatrixCell] = []
        if spec.kind == "task":
            base_tr = baseline.trs.get(spec.name) if baseline is not None else None
            base_value = task_value(base_tr)
            ref = base_tr or next((s.trs[spec.name] for s in subjects if spec.name in s.trs), None)
            ref_hash = ref["task_hash"] if ref is not None else None
            for s in subjects:
                tr = s.trs.get(spec.name)
                value = task_value(tr)
                if tr is None or value is None:
                    status: a.CellStatus = "failed" if tr is not None and tr["error"] else "missing"
                    coverage[status] += 1
                    cells.append(_empty_cell(status, tr))
                    continue
                coverage["complete"] += 1
                mismatch = bool(ref_hash and tr["task_hash"] != ref_hash)
                coverage["hash_mismatch"] += int(mismatch)
                delta = None
                if baseline is not None and s.key != baseline.key and base_value is not None:
                    hash_mismatch = base_tr is not None and tr["task_hash"] != base_tr["task_hash"]
                    delta = _runtime_delta(value, base_value, alpha, hash_mismatch)
                cells.append(
                    a.MatrixCell(
                        status="failed" if tr["error"] else "ok",
                        score=value,
                        stderr=None,
                        n=int(tr["num_instances"] or 0),
                        task_result_id=tr["id"],
                        run_id=tr["run_id"],
                        task_hash=tr["task_hash"],
                        delta=delta,
                        children_missing=0,
                    )
                )
        else:
            assert spec.node is not None
            leaf_names = leaves(spec.node)
            values = {s.key: suite_value(s, leaf_names) for s in subjects}
            base_value = values[baseline.key][0] if baseline is not None else None
            for s in subjects:
                value, present, missing = values[s.key]
                delta = None
                if baseline is not None and s.key != baseline.key and base_value is not None:
                    delta = _runtime_delta(value, base_value, alpha, False)
                cells.append(
                    a.MatrixCell(
                        status="missing" if not present else ("partial" if missing else "ok"),
                        score=value,
                        stderr=None,
                        n=None,
                        task_result_id=None,
                        run_id=s.run["run_id"] if s.run is not None else None,
                        task_hash=None,
                        delta=delta,
                        children_missing=missing,
                    )
                )
        out_rows.append(
            a.MatrixRow(
                key=spec.key,
                kind="task" if spec.kind == "task" else "suite",
                name=spec.name,
                depth=spec.depth,
                parent=spec.parent,
                aggregation=spec.aggregation,
                metric_key=body.metric,
                meta=RUNTIME_META,
                cells=cells,
            )
        )
    return a.MatrixResponse(
        subjects=[s.info() for s in subjects],
        baseline=body.baseline,
        scope=body.scope,
        metric=body.metric,
        rows=out_rows,
        coverage=a.Coverage(
            n_subjects=len(subjects),
            n_tasks=len(task_names),
            complete=coverage["complete"],
            missing=coverage["missing"],
            failed=coverage["failed"],
            hash_mismatch=coverage["hash_mismatch"],
        ),
        mde80=None,
        alpha=alpha,
        computed_at=now_utc(),
    )


# ---------------------------------------------------------------------------
# Pairwise
# ---------------------------------------------------------------------------


def restrict(side: Side, keys: np.ndarray) -> Side:
    assert side.keys is not None and side.values is not None
    keep = np.isin(side.keys, keys)
    return Side(
        task_name=side.task_name,
        task_hash=side.task_hash,
        score=side.score,
        stderr=side.stderr,
        score_is_mean=side.score_is_mean,
        instance_scale=side.instance_scale,
        keys=side.keys[keep],
        values=side.values[keep],
        kind=side.kind,
        higher_is_better=side.higher_is_better,
        task_result_id=side.task_result_id,
    )


@router.post("/compare/pairwise", response_model=a.PairwiseResponse)
async def compare_pairwise(
    body: a.PairwiseRequest, session: SessionDep, user: UserDep
) -> a.PairwiseResponse:
    stats = check_stats(body.alpha, body.n_boot, body.seed, None)
    reject_runtime_metric(body.metric)
    if len(set(body.subjects)) < 2:
        raise bad_request("pairwise needs at least two subjects")
    subjects = await resolve_for_compare(session, body.subjects, body.group)
    shared_only = True if body.shared_only is None else body.shared_only
    margin = float(body.margin or 0.0)
    if margin < 0:
        raise bad_request("margin must be >= 0")
    used = [tr for s in subjects for tr in s.trs.values()]
    rows, node, defs = await resolve_scope(session, body.scope, subjects)
    ckey = cache_key("pairwise", body.model_dump(mode="json"), used, scope_hashes(node, defs))
    cached = await get_cached(session, ckey, a.PairwiseResponse)
    if cached is not None:
        by_key = {s.key: s for s in subjects}
        return cached.model_copy(update={"subjects": live_subjects(cached.subjects, by_key)})

    task_names = list(dict.fromkeys(r.name for r in rows if r.kind == "task"))
    metric_keys = {t: row_metric(t, body.metric, subjects, None) for t in task_names}
    loader = SideLoader(session)
    for t in task_names:
        for s in subjects:
            loader.want(s.trs.get(t), metric_keys[t])
    await loader.load()
    tasks: list[tuple[str, dict[str, Side]]] = []
    metas = []
    skipped = 0
    for t in task_names:
        sides = {s.key: sd for s in subjects if (sd := loader.side(s.trs.get(t), metric_keys[t]))}
        if not sides:
            continue
        if shared_only:
            pairable = [sd for sd in sides.values() if sd.pairable]
            keys = common_keys(pairable) if len(pairable) == len(sides) else None
            if keys is not None:
                sides = {k: restrict(sd, keys) for k, sd in sides.items()}
        if not all(sd.pairable for sd in sides.values()):
            skipped += 1
        tasks.append((t, sides))
        metas.append(row_meta(t, metric_keys[t], subjects, None))
    directions = {m.get("higher_is_better") for m in metas if m}
    hib = directions.pop() if len(directions) == 1 else None
    formats = {m.get("display_format") for m in metas if m}
    display: a.DisplayFormat = "percent" if formats == {"percent"} else "raw"
    n_inst = {
        t: max(int(s.trs[t]["num_instances"] or 0) for s in subjects if t in s.trs)
        for t, _ in tasks
    }
    if node is not None:

        def weight_fn(names: Sequence[str]) -> dict[str, float]:
            return leaf_weights(node, n_inst, set(names))
    else:

        def weight_fn(names: Sequence[str]) -> dict[str, float]:
            return {n: 1.0 / len(names) for n in names}

    keys = [s.key for s in subjects]
    pairs = [(keys[i], keys[j]) for i in range(len(keys)) for j in range(i + 1, len(keys))]
    acc = await asyncio.to_thread(
        lambda: stratified_pairs(
            tasks,
            pairs,
            weight_fn,
            alpha=stats.alpha,
            n_boot=stats.n_boot,
            seed=stats.seed,
            higher_is_better=hib,
            margin=margin,
            max_work=MAX_PAIRWISE_WORK,
            max_pair_resamples=MAX_PAIR_RESAMPLES,
        )
    )
    pair_out: list[a.PairStats] = []
    win_boot: dict[tuple[str, str], np.ndarray] = {}
    deltas: dict[tuple[str, str], float | None] = {}
    ses = []
    used_tasks: set[str] = set()
    for (r, c), ac in acc.items():
        st = pair_stats(ac, stats.alpha, hib)
        used_tasks.update(ac.weights)
        if ac.weights and ac.var_terms > 0:
            ses.append(float(np.sqrt(ac.var_terms)))
        pair_out.append(a.PairStats(row=r, col=c, **st))
        mirrored = dict(st)
        mirrored.update(
            wins=st["losses"],
            losses=st["wins"],
            win_rate=1 - st["win_rate"] if st["n_contested"] else 0.5,
            delta=-st["delta"] if st["delta"] is not None else None,
            ci_low=-st["ci_high"] if st["ci_high"] is not None else None,
            ci_high=-st["ci_low"] if st["ci_low"] is not None else None,
            prob_row_better=1 - st["prob_row_better"]
            if st["prob_row_better"] is not None
            else None,
        )
        pair_out.append(a.PairStats(row=c, col=r, **mirrored))
        wr = resampled_win_rate(ac)
        win_boot[(r, c)] = wr
        win_boot[(c, r)] = 1 - wr
        deltas[(r, c)] = st["delta"]
        deltas[(c, r)] = mirrored["delta"]
    summaries = []
    for k in keys:
        others = [o for o in keys if o != k]
        rates = [p.win_rate for p in pair_out if p.row == k]
        mean_rate = float(np.mean(rates)) if rates else 0.5
        boots = np.stack([win_boot[(k, o)] for o in others], axis=1)
        per_resample = boots.mean(axis=1)
        lo, hi = np.quantile(per_resample, [stats.alpha / 2, 1 - stats.alpha / 2])
        ds = [d for o in others if (d := deltas[(k, o)]) is not None]
        summaries.append(
            a.PairwiseRowSummary(
                subject=k,
                mean_win_rate=mean_rate,
                ci_low=float(lo) if used_tasks else None,
                ci_high=float(hi) if used_tasks else None,
                mean_delta=float(np.mean(ds)) if ds else None,
            )
        )
    summaries.sort(key=lambda x: (-x.mean_win_rate, -(x.mean_delta or 0.0)))
    order = {x.subject: i for i, x in enumerate(summaries)}
    warnings = []
    if skipped:
        warnings.append(
            f"{skipped} task(s) have scores that are not per-instance means; they count toward "
            "wins and losses only where instances pair"
        )
    if any(ac.hash_mismatch for ac in acc.values()):
        warnings.append("some subjects ran different task configurations (task hashes differ)")
    missing = [t for t in task_names if t not in {x for x, _ in tasks}]
    if missing:
        warnings.append(f"{len(missing)} task(s) in scope have no results for any subject")
    response = a.PairwiseResponse(
        subjects=[s.info() for s in sorted(subjects, key=lambda s: order[s.key])],
        pairs=pair_out,
        rows=summaries,
        tasks_used=[t for t in task_names if t in used_tasks],
        mde80=mde80(ses, stats.alpha),
        alpha=stats.alpha,
        margin=margin,
        higher_is_better=hib,
        display_format=display,
        warnings=warnings,
        computed_at=now_utc(),
    )
    await put_cached(session, ckey, response)
    return response


# ---------------------------------------------------------------------------
# Contingency
# ---------------------------------------------------------------------------


@router.post("/compare/contingency", response_model=a.ContingencyResponse)
async def compare_contingency(
    body: a.ContingencyRequest, session: SessionDep, user: UserDep
) -> a.ContingencyResponse:
    threshold = check_threshold(body.threshold)
    reject_runtime_metric(body.metric)
    subjects = await resolve_for_compare(session, [body.a, body.b], body.group)
    if len(subjects) < 2:
        raise bad_request("a and b must be different subjects")
    sa, sb = subjects
    rows, _, _ = await resolve_scope(session, body.scope, subjects)
    task_names = list(dict.fromkeys(r.name for r in rows if r.kind == "task"))
    loader = SideLoader(session)
    metric_keys = {t: row_metric(t, body.metric, [sa, sb], sb) for t in task_names}
    for t in task_names:
        loader.want(sa.trs.get(t), metric_keys[t])
        loader.want(sb.trs.get(t), metric_keys[t])
    await loader.load()
    items = []
    totals = {"both_right": 0, "only_a": 0, "only_b": 0, "both_wrong": 0, "n_shared": 0, "net": 0}
    for t in task_names:
        side_a = loader.side(sa.trs.get(t), metric_keys[t])
        side_b = loader.side(sb.trs.get(t), metric_keys[t])
        if side_a is None and side_b is None:
            continue
        cont = contingency(side_a, side_b, threshold) if side_a and side_b else None
        if cont:
            for k in totals:
                totals[k] += cont[k]
        delta = (
            side_a.score - side_b.score
            if side_a and side_b and side_a.score is not None and side_b.score is not None
            else None
        )
        either = side_a if side_a is not None else side_b
        kind = metric_kind_literal(either.kind if either is not None else None)
        items.append(
            a.ContingencyTaskRow(
                task_name=t,
                kind=kind,
                task_hash_a=side_a.task_hash if side_a else None,
                task_hash_b=side_b.task_hash if side_b else None,
                task_result_id_a=side_a.task_result_id if side_a else None,
                task_result_id_b=side_b.task_result_id if side_b else None,
                contingency=a.Contingency(**cont) if cont else None,
                delta=delta,
            )
        )
    items.sort(key=lambda r: (r.contingency is None, r.contingency.net if r.contingency else 0))
    return a.ContingencyResponse(
        a=body.a, b=body.b, items=items, totals=a.Contingency(**totals, threshold=threshold)
    )


# ---------------------------------------------------------------------------
# Instance agreement grid
# ---------------------------------------------------------------------------


@router.post("/compare/instances", response_model=a.CompareInstancesResponse)
async def compare_instances(
    body: a.CompareInstancesRequest, session: SessionDep, user: UserDep
) -> a.CompareInstancesResponse:
    threshold = check_threshold(body.threshold)
    reject_runtime_metric(body.metric)
    include_previews = bool(body.include_previews)
    max_limit = 1000 if include_previews else 20_000
    limit = body.limit if body.limit is not None else 200
    if not 1 <= limit <= max_limit:
        raise bad_request(f"limit must be between 1 and {max_limit}")
    keys = list(dict.fromkeys(body.subjects))
    extra = [k for k in (body.a, body.b, body.baseline) if k and k not in keys]
    if extra:
        raise bad_request("a, b and baseline must be among subjects")
    if body.filter == "cell" and not (body.cell and body.a and body.b):
        raise bad_request("filter=cell needs cell, a and b")
    if body.filter == "baseline_wrong" and not body.baseline:
        raise bad_request("filter=baseline_wrong needs baseline")
    subjects = await resolve_for_compare(session, keys, body.group)
    task = body.task_name
    metric_key = row_metric(
        task,
        body.metric,
        subjects,
        next((s for s in subjects if s.key == body.baseline), None),
    )
    loader = SideLoader(session)
    for s in subjects:
        loader.want(s.trs.get(task), metric_key)
    await loader.load()
    sides = [loader.side(s.trs.get(task), metric_key) for s in subjects]
    meta = row_meta(task, metric_key, subjects, None)
    kind = meta.get("kind") or "unbounded"
    hib = meta.get("higher_is_better")
    universe = union_keys(*[sd.keys for sd in sides if sd is not None and sd.keys is not None])
    n = universe.size
    scores = np.full((n, len(subjects)), np.nan)
    for j, sd in enumerate(sides):
        if sd is not None and sd.keys is not None and sd.values is not None:
            scores[:, j] = align(universe, sd.keys, sd.values)
    present = np.isfinite(scores)
    corr = correctness(scores, kind, hib, threshold)
    corr_arr = np.zeros_like(present) if corr is None else corr
    n_correct = corr_arr.sum(axis=1)
    n_present = present.sum(axis=1)
    idx = {s.key: j for j, s in enumerate(subjects)}
    mask = np.ones(n, dtype=bool)
    if corr is None and body.filter in ("all_wrong", "baseline_wrong", "cell"):
        # Unbounded metrics have no right or wrong, so correctness filters match nothing.
        mask = np.zeros(n, dtype=bool)
    elif body.filter == "disagree":
        mask = (n_correct > 0) & (n_correct < n_present)
    elif body.filter == "all_wrong":
        mask = (n_correct == 0) & (n_present > 0)
    elif body.filter == "baseline_wrong":
        assert body.baseline is not None
        jb = idx[body.baseline]
        mask = present[:, jb] & ~corr_arr[:, jb]
    elif body.filter == "cell":
        assert body.a is not None and body.b is not None and body.cell is not None
        ja, jb = idx[body.a], idx[body.b]
        both = present[:, ja] & present[:, jb]
        ca, cb = corr_arr[:, ja], corr_arr[:, jb]
        cell = {
            "both_right": ca & cb,
            "only_a": ca & ~cb,
            "only_b": ~ca & cb,
            "both_wrong": ~ca & ~cb,
        }[body.cell]
        mask = both & cell
    selected = np.nonzero(mask)[0]
    tr_ids = [sd.task_result_id for sd in sides if sd is not None and sd.task_result_id]
    native = {}
    if tr_ids and selected.size:
        rows = await session.execute(
            text(
                "SELECT DISTINCT ON (key_hash) key_hash, native_id FROM instance_results "
                "WHERE task_result_id = ANY(:ids) AND key_hash = ANY(:keys) ORDER BY key_hash"
            ),
            {"ids": tr_ids, "keys": [int(k) for k in universe[selected]]},
        )
        native = {int(r[0]): r[1] for r in rows}
    ids = np.array(
        [native.get(int(universe[i]), str(int(universe[i]))) for i in selected], dtype=object
    )
    sort = body.sort or "disagreement"
    if sort == "native_id":
        order = np.argsort(ids, kind="stable")
    elif sort == "n_correct":
        order = np.lexsort((ids, -n_correct[selected]))
    else:
        order = np.lexsort(
            (ids, -n_correct[selected], np.abs(n_correct[selected] - n_present[selected] / 2))
        )
    selected = selected[order]
    ids = ids[order]
    counts = np.bincount(n_correct[mask].astype(int), minlength=len(subjects) + 1)
    start = decode_offset_cursor(body.cursor)
    page = selected[start : start + limit]
    page_ids = ids[start : start + limit]
    previews: dict[str, dict[int, tuple[str | None, str | None]]] = {}
    if include_previews and page.size:
        for s, sd in zip(subjects, sides, strict=True):
            if sd is None or not sd.task_result_id:
                continue
            rows = await session.execute(
                text(
                    "SELECT key_hash, prompt_preview, output_preview FROM instance_results "
                    "WHERE task_result_id = :i AND key_hash = ANY(:keys)"
                ),
                {"i": sd.task_result_id, "keys": [int(universe[i]) for i in page]},
            )
            previews[s.key] = {int(r[0]): (r[1], r[2]) for r in rows}
    items = []
    for i, nid in zip(page, page_ids, strict=True):
        k = int(universe[i])
        prompt = None
        outs = None
        if include_previews:
            outs = []
            for s in subjects:
                p = previews.get(s.key, {}).get(k)
                if p and prompt is None:
                    prompt = p[0]
                outs.append(p[1] if p else None)
        items.append(
            a.CompareInstanceRow(
                native_id=str(nid),
                scores=[
                    None if not present[i, j] else float(scores[i, j]) for j in range(len(subjects))
                ],
                correct=[
                    None if (corr is None or not present[i, j]) else bool(corr_arr[i, j])
                    for j in range(len(subjects))
                ],
                n_correct=int(n_correct[i]),
                n_present=int(n_present[i]),
                prompt_preview=prompt,
                output_previews=outs,
            )
        )
    next_cursor = encode_cursor([start + limit]) if start + limit < selected.size else None
    return a.CompareInstancesResponse(
        subjects=[
            a.CompareInstancesResponseSubjectsItem(
                key=s.key,
                task_result_id=s.trs[task]["id"] if task in s.trs else None,
                run_id=s.trs[task]["run_id"] if task in s.trs else None,
            )
            for s in subjects
        ],
        kind=kind,
        threshold=threshold,
        items=items,
        counts_by_n_correct=[int(c) for c in counts[: len(subjects) + 1]],
        total=int(selected.size),
        next_cursor=next_cursor,
    )
