"""Deltas with uncertainty between two subjects (spec 5.3-5.6, 5.8).

A *side* is one subject's result on one task: corpus score, stderr, whether the score is the
mean of the per-instance values, and the per-instance vector (sorted key hashes and raw values).
Per-instance values are multiplied by ``instance_scale`` so deltas are on the corpus scale.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from olmo_eval_api.stats.bootstrap import align, resampled_means, union_keys, weight_chunks
from olmo_eval_api.stats.mde import z
from olmo_eval_api.stats.sign_test import sign_test

MIN_SHARED = 20
# Pair columns are processed in blocks of at most this many (instance, pair) cells, so the
# working matrices stay under roughly 70 MB whatever the number of pairs.
MAX_BLOCK_CELLS = 2_000_000
# Limits for the all-pairs comparison. 50 subjects (1,225 pairs) on 20k-instance tasks with
# 2,000 resamples is 4.9e10 cells and fits; the accumulators hold 3 x 8 bytes per pair and
# resample, so 5M pair-resamples is about 120 MB.
MAX_PAIRWISE_WORK = 60_000_000_000
MAX_PAIR_RESAMPLES = 5_000_000


class WorkBudgetExceeded(ValueError):
    """The request needs more bootstrap work or memory than the server allows."""


def column_blocks(n_cols: int, n_rows: int, block_cells: int | None = None) -> Iterator[slice]:
    """Slices over ``n_cols`` columns with at most ``block_cells`` cells of ``n_rows`` each."""
    cells = MAX_BLOCK_CELLS if block_cells is None else block_cells
    step = max(1, cells // max(n_rows, 1))
    for start in range(0, n_cols, step):
        yield slice(start, min(start + step, n_cols))


@dataclass
class Side:
    task_name: str
    task_hash: str | None
    score: float | None
    stderr: float | None
    score_is_mean: bool
    instance_scale: float = 1.0
    keys: np.ndarray | None = None  # sorted int64 key hashes
    values: np.ndarray | None = None  # raw per-instance values, NaN for null
    kind: str | None = None
    higher_is_better: bool | None = None
    task_result_id: int | None = None

    @property
    def has_vector(self) -> bool:
        return self.keys is not None and self.keys.size > 0

    @property
    def scaled(self) -> np.ndarray:
        assert self.values is not None
        return self.values * self.instance_scale

    @property
    def pairable(self) -> bool:
        return self.score_is_mean and self.has_vector


@dataclass
class Delta:
    delta: float | None
    ci_low: float | None = None
    ci_high: float | None = None
    p_value: float | None = None
    n_shared: int = 0
    method: str = "none"
    significant: bool = False
    improved: bool | None = None
    hash_mismatch: bool = False
    alpha: float = 0.05
    n_boot: int | None = None
    note: str | None = None
    se: float | None = None  # standard error of the paired difference, for MDE80
    boot: np.ndarray | None = field(default=None, repr=False)

    def as_dict(self) -> dict[str, Any]:
        return {
            "delta": _f(self.delta),
            "ci_low": _f(self.ci_low),
            "ci_high": _f(self.ci_high),
            "p_value": _f(self.p_value),
            "n_shared": int(self.n_shared),
            "method": self.method,
            "significant": bool(self.significant),
            "improved": self.improved,
            "hash_mismatch": bool(self.hash_mismatch),
            "alpha": self.alpha,
            "n_boot": self.n_boot,
            "note": self.note,
        }


def _f(x: float | None) -> float | None:
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def improved(delta: float | None, higher_is_better: bool | None) -> bool | None:
    if delta is None or higher_is_better is None or delta == 0:
        return None
    return delta > 0 if higher_is_better else delta < 0


def corpus_delta(a: float | None, b: float | None) -> float | None:
    return a - b if a is not None and b is not None else None


def bootstrap_summary(boot: np.ndarray, alpha: float) -> tuple[float, float, float] | None:
    """Percentile CI and two-sided p-value of resampled deltas (NaN resamples dropped)."""
    boot = boot[np.isfinite(boot)]
    if boot.size == 0:
        return None
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2], method="linear")
    b = boot.size
    p = min(1.0, 2 * (min(int((boot <= 0).sum()), int((boot >= 0).sum())) + 1) / (b + 1))
    return float(lo), float(hi), p


def unpaired(
    a: Side, b: Side, alpha: float, higher_is_better: bool | None, hash_mismatch: bool = False
) -> Delta:
    delta = corpus_delta(a.score, b.score)
    if delta is None or a.stderr is None or b.stderr is None:
        return none_delta(
            a, b, alpha, higher_is_better, "standard error unavailable", hash_mismatch
        )
    se = math.sqrt(a.stderr**2 + b.stderr**2)
    if se == 0:
        return none_delta(a, b, alpha, higher_is_better, "zero standard error", hash_mismatch)
    zq = z(1 - alpha / 2)
    lo, hi = delta - zq * se, delta + zq * se
    p = 2 * (1 - _normal_cdf(abs(delta) / se))
    return Delta(
        delta=delta,
        ci_low=lo,
        ci_high=hi,
        p_value=p,
        method="unpaired",
        significant=lo > 0 or hi < 0,
        improved=improved(delta, higher_is_better),
        hash_mismatch=hash_mismatch,
        alpha=alpha,
        se=se,
    )


def _normal_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def none_delta(
    a: Side,
    b: Side,
    alpha: float,
    higher_is_better: bool | None,
    note: str,
    hash_mismatch: bool = False,
    n_shared: int = 0,
) -> Delta:
    delta = corpus_delta(a.score, b.score)
    return Delta(
        delta=delta,
        method="none",
        improved=improved(delta, higher_is_better),
        hash_mismatch=hash_mismatch,
        alpha=alpha,
        note=note,
        n_shared=n_shared,
    )


@dataclass
class PairPlan:
    """How one (a, b) pair on one task is compared."""

    a: Side
    b: Side
    method: str  # "paired_bootstrap", "insufficient", "unpaired", "none"
    n_shared: int
    note: str | None = None


def plan_pair(a: Side, b: Side) -> PairPlan:
    if a.pairable and b.pairable:
        assert a.keys is not None and b.keys is not None
        shared = _shared_count(a, b)
        if shared < MIN_SHARED:
            return PairPlan(a, b, "insufficient", shared, f"only {shared} shared instances")
        return PairPlan(a, b, "paired_bootstrap", shared)
    if a.has_vector and b.has_vector:
        return PairPlan(
            a, b, "none", 0, "score is not a per-instance mean, so no confidence interval"
        )
    if a.stderr is not None and b.stderr is not None:
        return PairPlan(a, b, "unpaired", 0)
    return PairPlan(a, b, "none", 0, "no per-instance scores or standard errors")


def _present(side: Side) -> np.ndarray:
    assert side.keys is not None and side.values is not None
    return side.keys[np.isfinite(side.values)]


def _shared_count(a: Side, b: Side) -> int:
    return int(np.intersect1d(_present(a), _present(b), assume_unique=True).size)


def paired_differences(a: Side, b: Side, union: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(d, mask) on ``union``: d = a - b on the corpus scale where both are present."""
    assert a.keys is not None and a.values is not None
    assert b.keys is not None and b.values is not None
    va = align(union, a.keys, a.scaled)
    vb = align(union, b.keys, b.scaled)
    mask = np.isfinite(va) & np.isfinite(vb)
    return np.where(mask, va - vb, 0.0), mask


def _boot_delta(
    plan: PairPlan,
    d: np.ndarray,
    boot: np.ndarray,
    alpha: float,
    n_boot: int,
    higher_is_better: bool | None,
    keep_boot: bool,
) -> Delta:
    delta = float(d.mean())
    summary = bootstrap_summary(boot, alpha)
    lo, hi, p = summary if summary else (None, None, None)
    se = float(d.std(ddof=1) / math.sqrt(d.size)) if d.size >= 2 else None
    mismatch = bool(plan.a.task_hash and plan.b.task_hash and plan.a.task_hash != plan.b.task_hash)
    return Delta(
        delta=delta,
        ci_low=lo,
        ci_high=hi,
        p_value=p,
        n_shared=plan.n_shared,
        method="paired_bootstrap",
        significant=lo is not None and hi is not None and (lo > 0 or hi < 0),
        improved=improved(delta, higher_is_better),
        hash_mismatch=mismatch,
        alpha=alpha,
        n_boot=n_boot,
        se=se,
        boot=boot.copy() if keep_boot else None,
    )


def compare_pairs(
    pairs: Sequence[tuple[Side, Side]],
    *,
    task_name: str,
    alpha: float,
    n_boot: int,
    seed: int,
    higher_is_better: bool | None,
    keep_boot: bool = False,
    block_cells: int | None = None,
) -> list[Delta]:
    """Deltas for several (a, b) pairs of one task.

    Every pair uses the weights of the union of the task's keys, so a pair's result does not
    depend on the other pairs or on how the pair columns are split into blocks.
    """
    plans = [plan_pair(a, b) for a, b in pairs]
    results: list[Delta | None] = [None] * len(plans)
    boot_idx = [i for i, p in enumerate(plans) if p.method == "paired_bootstrap"]
    for i, plan in enumerate(plans):
        a, b = plan.a, plan.b
        mismatch = bool(a.task_hash and b.task_hash and a.task_hash != b.task_hash)
        if plan.method == "unpaired":
            results[i] = unpaired(a, b, alpha, higher_is_better, mismatch)
        elif plan.method == "none":
            results[i] = none_delta(a, b, alpha, higher_is_better, plan.note or "", mismatch)
        elif plan.method == "insufficient":
            delta = corpus_delta(a.score, b.score)
            if plan.n_shared > 0:
                union = union_keys(a.keys, b.keys)
                d, m = paired_differences(a, b, union)
                delta = float(d[m].mean())
            results[i] = Delta(
                delta=delta,
                n_shared=plan.n_shared,
                method="insufficient",
                improved=improved(delta, higher_is_better),
                hash_mismatch=mismatch,
                alpha=alpha,
                note=plan.note,
            )
    if boot_idx:
        union = union_keys(*[k for i in boot_idx for k in (plans[i].a.keys, plans[i].b.keys)])
        for block in column_blocks(len(boot_idx), union.size, block_cells):
            idx = boot_idx[block]
            cols = [paired_differences(plans[i].a, plans[i].b, union) for i in idx]
            diffs = np.stack([c[0] for c in cols], axis=1)
            masks = np.stack([c[1] for c in cols], axis=1)
            del cols
            boot = resampled_means(union, diffs, masks, task_name, seed, n_boot)
            for j, i in enumerate(idx):
                d = diffs[masks[:, j], j]
                results[i] = _boot_delta(
                    plans[i], d, boot[:, j], alpha, n_boot, higher_is_better, keep_boot
                )
    return [r for r in results if r is not None]


def compare(
    a: Side,
    b: Side,
    *,
    alpha: float = 0.05,
    n_boot: int = 2000,
    seed: int = 0,
    higher_is_better: bool | None = None,
) -> Delta:
    """Delta for one pair; the direction defaults to side ``a``'s metric direction."""
    if higher_is_better is None:
        higher_is_better = a.higher_is_better
    return compare_pairs(
        [(a, b)],
        task_name=a.task_name,
        alpha=alpha,
        n_boot=n_boot,
        seed=seed,
        higher_is_better=higher_is_better,
    )[0]


# ---------------------------------------------------------------------------
# Correctness and contingency (spec 4.4, 5.8)
# ---------------------------------------------------------------------------


def correctness(
    values: np.ndarray, kind: str | None, higher_is_better: bool | None, threshold: float
) -> np.ndarray | None:
    """Boolean array (NaN values are not correct), or None for unbounded metrics."""
    if kind not in ("binary", "bounded"):
        return None
    goodness = values if higher_is_better is not False else 1.0 - values
    with np.errstate(invalid="ignore"):
        return np.isfinite(values) & (goodness >= threshold)


def contingency(a: Side, b: Side, threshold: float) -> dict[str, Any] | None:
    if not (a.has_vector and b.has_vector):
        return None
    kind = a.kind or b.kind
    if kind not in ("binary", "bounded") or (b.kind and b.kind not in ("binary", "bounded")):
        return None
    assert a.keys is not None and a.values is not None
    assert b.keys is not None and b.values is not None
    union = union_keys(a.keys, b.keys)
    va = align(union, a.keys, a.values)
    vb = align(union, b.keys, b.values)
    shared = np.isfinite(va) & np.isfinite(vb)
    hib = a.higher_is_better if a.higher_is_better is not None else b.higher_is_better
    ca = correctness(va[shared], kind, hib, threshold)
    cb = correctness(vb[shared], kind, hib, threshold)
    assert ca is not None and cb is not None
    both = int((ca & cb).sum())
    only_a = int((ca & ~cb).sum())
    only_b = int((~ca & cb).sum())
    neither = int((~ca & ~cb).sum())
    return {
        "both_right": both,
        "only_a": only_a,
        "only_b": only_b,
        "both_wrong": neither,
        "n_shared": int(shared.sum()),
        "net": only_a - only_b,
        "threshold": threshold,
    }


# ---------------------------------------------------------------------------
# Stratified (multi-task) comparison (spec 5.5, 5.6)
# ---------------------------------------------------------------------------


@dataclass
class PairAccumulator:
    """Running sums for one subject pair across the tasks of a scope."""

    n_boot: int
    weights: dict[str, float] = field(default_factory=dict)
    point: float = 0.0
    boot: np.ndarray = field(init=False)
    var_terms: float = 0.0
    n_shared: int = 0
    wins: int = 0
    losses: int = 0
    ties: int = 0
    win_num: np.ndarray = field(init=False)
    win_den: np.ndarray = field(init=False)
    corpus_a: dict[str, float | None] = field(default_factory=dict)
    corpus_b: dict[str, float | None] = field(default_factory=dict)
    hash_mismatch: bool = False

    def __post_init__(self) -> None:
        self.boot = np.zeros(self.n_boot)
        self.win_num = np.zeros(self.n_boot)
        self.win_den = np.zeros(self.n_boot)


def outcome_counts(d: np.ndarray, margin: float, higher_is_better: bool | None) -> np.ndarray:
    """Per-instance outcome from the row's point of view: 1 win, -1 loss, 0 tie."""
    s = d if higher_is_better is not False else -d
    return np.where(s > margin, 1, np.where(s < -margin, -1, 0))


def stratified_pairs(
    tasks: Sequence[tuple[str, Mapping[str, Side]]],
    pairs: Sequence[tuple[str, str]],
    weight_fn: Any,
    *,
    alpha: float,
    n_boot: int,
    seed: int,
    higher_is_better: bool | None,
    margin: float = 0.0,
    max_work: int | None = None,
    max_pair_resamples: int | None = None,
    block_cells: int | None = None,
) -> dict[tuple[str, str], PairAccumulator]:
    """Combine per-task bootstrap means across tasks for each (row, col) subject pair.

    ``tasks`` lists (task_name, {subject: Side}). ``weight_fn(qualifying_task_names)`` returns
    task weights summing to 1. Tasks qualify for a pair when both sides are per-instance means
    with at least MIN_SHARED shared instances.

    Pair columns are processed in blocks (``column_blocks``), so peak memory does not grow with
    the number of pairs. ``max_work`` caps the sum over tasks of pairs x union instances x
    resamples, and ``max_pair_resamples`` caps pairs x resamples (the accumulators hold three
    resample vectors per pair); either raises WorkBudgetExceeded before any bootstrap runs.
    """
    if max_pair_resamples is not None and len(pairs) * n_boot > max_pair_resamples:
        raise WorkBudgetExceeded(
            f"{len(pairs)} subject pairs x {n_boot} resamples exceeds the limit of "
            f"{max_pair_resamples:,}; compare fewer subjects or lower n_boot"
        )
    acc = {p: PairAccumulator(n_boot=n_boot) for p in pairs}
    qualifying: dict[tuple[str, str], list[str]] = {p: [] for p in pairs}
    for task_name, sides in tasks:
        for p in pairs:
            a, b = sides.get(p[0]), sides.get(p[1])
            if a is None or b is None:
                continue
            acc[p].corpus_a[task_name] = a.score
            acc[p].corpus_b[task_name] = b.score
            if a.task_hash and b.task_hash and a.task_hash != b.task_hash:
                acc[p].hash_mismatch = True
            if plan_pair(a, b).method == "paired_bootstrap":
                qualifying[p].append(task_name)
    for p in pairs:
        acc[p].weights = weight_fn(qualifying[p]) if qualifying[p] else {}

    work: list[tuple[str, Mapping[str, Side], list[tuple[str, str]], np.ndarray]] = []
    total = 0
    for task_name, sides in tasks:
        cols = [p for p in pairs if acc[p].weights.get(task_name)]
        if not cols:
            continue
        union = union_keys(*[k for p in cols for k in (sides[p[0]].keys, sides[p[1]].keys)])
        work.append((task_name, sides, cols, union))
        total += len(cols) * int(union.size) * n_boot
    if max_work is not None and total > max_work:
        raise WorkBudgetExceeded(
            f"this comparison needs {total:,} bootstrap cells (pairs x instances x resamples), "
            f"more than the limit of {max_work:,}; compare fewer subjects, narrow the scope or "
            "lower n_boot"
        )

    hib = higher_is_better
    for task_name, sides, all_cols, union in work:
        for block in column_blocks(len(all_cols), union.size, block_cells):
            cols = all_cols[block]
            _stratified_block(acc, task_name, sides, cols, union, n_boot, seed, margin, hib)
    return acc


def _stratified_block(
    acc: Mapping[tuple[str, str], PairAccumulator],
    task_name: str,
    sides: Mapping[str, Side],
    cols: Sequence[tuple[str, str]],
    union: np.ndarray,
    n_boot: int,
    seed: int,
    margin: float,
    higher_is_better: bool | None,
) -> None:
    """Add one task's contribution for a block of pair columns to their accumulators."""
    diffs_masks = [paired_differences(sides[p[0]], sides[p[1]], union) for p in cols]
    diffs = np.stack([dm[0] for dm in diffs_masks], axis=1)  # zero where not shared
    masks = np.stack([dm[1] for dm in diffs_masks], axis=1)
    del diffs_masks
    outcomes = np.stack(
        [outcome_counts(diffs[:, j], margin, higher_is_better) for j in range(len(cols))],
        axis=1,
    ).astype(np.int8)
    win = np.where(masks, outcomes == 1, False).astype(np.float64)
    contested = np.where(masks, outcomes != 0, False).astype(np.float64)
    m0 = masks.astype(np.float64)
    boot = np.empty((n_boot, len(cols)))
    win_num = np.empty((n_boot, len(cols)))
    win_den = np.empty((n_boot, len(cols)))
    for b0, w in weight_chunks(union, task_name, seed, n_boot):
        w64 = w.astype(np.float64)
        sl = slice(b0, b0 + w.shape[0])
        den = w64 @ m0
        with np.errstate(invalid="ignore", divide="ignore"):
            boot[sl] = np.where(den > 0, (w64 @ diffs) / den, np.nan)
        win_num[sl] = w64 @ win
        win_den[sl] = w64 @ contested
    for j, p in enumerate(cols):
        a_ = acc[p]
        wt = a_.weights[task_name]
        d = diffs[masks[:, j], j]
        a_.point += wt * float(d.mean())
        a_.boot = a_.boot + wt * boot[:, j]
        if d.size >= 2:
            a_.var_terms += wt * wt * float(d.var(ddof=1)) / d.size
        a_.n_shared += int(d.size)
        o = outcomes[masks[:, j], j]
        a_.wins += int((o == 1).sum())
        a_.losses += int((o == -1).sum())
        a_.ties += int((o == 0).sum())
        a_.win_num = a_.win_num + win_num[:, j]
        a_.win_den = a_.win_den + win_den[:, j]


def pair_stats(acc: PairAccumulator, alpha: float, higher_is_better: bool | None) -> dict[str, Any]:
    """PairStats fields (without row/col) from an accumulator."""
    contested = acc.wins + acc.losses
    win_rate = acc.wins / contested if contested else 0.5
    if acc.weights:
        summary = bootstrap_summary(acc.boot, alpha)
        lo, hi, p = summary if summary else (None, None, None)
        boot = acc.boot[np.isfinite(acc.boot)]
        if boot.size:
            better = boot > 0 if higher_is_better is not False else boot < 0
            prob = float(better.mean())
        else:
            prob = None
        return {
            "n_shared": acc.n_shared,
            "n_contested": contested,
            "wins": acc.wins,
            "losses": acc.losses,
            "ties": acc.ties,
            "win_rate": win_rate,
            "delta": acc.point,
            "ci_low": lo,
            "ci_high": hi,
            "p_bootstrap": p,
            "p_sign": sign_test(acc.wins, acc.losses),
            "prob_row_better": prob,
            "significant": lo is not None and hi is not None and (lo > 0 or hi < 0),
            "method": "paired_bootstrap",
        }
    diffs = [
        a - b
        for t in acc.corpus_a
        if (a := acc.corpus_a[t]) is not None and (b := acc.corpus_b.get(t)) is not None
    ]
    delta = sum(diffs) / len(diffs) if diffs else None
    return {
        "n_shared": 0,
        "n_contested": contested,
        "wins": acc.wins,
        "losses": acc.losses,
        "ties": acc.ties,
        "win_rate": win_rate,
        "delta": delta,
        "ci_low": None,
        "ci_high": None,
        "p_bootstrap": None,
        "p_sign": sign_test(acc.wins, acc.losses),
        "prob_row_better": None,
        "significant": False,
        "method": "none",
    }


def resampled_win_rate(acc: PairAccumulator) -> np.ndarray:
    num, den = acc.win_num, acc.win_den
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, 0.5)
