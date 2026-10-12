"""Statistics: bootstrap weights, paired deltas, sign test, MDE, stratified combination."""

from __future__ import annotations

import math
import time
from typing import Any

import httpx
import numpy as np
import pytest

from olmo_eval_api.services.derive import instance_key_hash
from olmo_eval_api.stats.bootstrap import poisson_weights, resampled_means
from olmo_eval_api.stats.histograms import box_stats, histogram
from olmo_eval_api.stats.mde import mde80, mde_factor
from olmo_eval_api.stats.paired import (
    Side,
    WorkBudgetExceeded,
    compare,
    compare_pairs,
    contingency,
    pair_stats,
    stratified_pairs,
    unpaired,
)
from olmo_eval_api.stats.sign_test import sign_test
from tests.factories import seed_run


def keys_for(n: int, prefix: str = "q") -> np.ndarray:
    return np.array([instance_key_hash(f"{prefix}{i}") for i in range(n)], dtype=np.int64)


def side(values: np.ndarray, keys: np.ndarray | None = None, **kw: Any) -> Side:
    keys = keys_for(len(values)) if keys is None else keys
    order = np.argsort(keys)
    vals = np.asarray(values, dtype=float)
    finite = vals[np.isfinite(vals)]
    return Side(
        task_name=kw.pop("task_name", "t"),
        task_hash=kw.pop("task_hash", "h"),
        score=kw.pop("score", float(finite.mean()) if finite.size else None),
        stderr=kw.pop("stderr", float(finite.std(ddof=1) / math.sqrt(finite.size))),
        score_is_mean=kw.pop("score_is_mean", True),
        keys=keys[order],
        values=vals[order],
        kind=kw.pop("kind", "binary"),
        higher_is_better=kw.pop("higher_is_better", True),
        **kw,
    )


def test_poisson_weights_deterministic_and_unit_mean() -> None:
    keys = keys_for(2000)
    w1 = poisson_weights(keys, "task", 0, 0, 50)
    w2 = poisson_weights(keys, "task", 0, 0, 50)
    assert w1.dtype == np.float32 and w1.shape == (50, 2000)
    assert np.array_equal(w1, w2)
    assert abs(float(w1.mean()) - 1.0) < 0.02
    assert abs(float(w1.var()) - 1.0) < 0.05
    # Weights depend only on (seed, task, resample, key): a subset of keys sees the same weights.
    assert np.array_equal(poisson_weights(keys[:10], "task", 0, 0, 50), w1[:, :10])
    assert np.array_equal(poisson_weights(keys, "task", 0, 20, 5), w1[20:25])
    assert not np.array_equal(poisson_weights(keys, "other", 0, 0, 50), w1)
    assert not np.array_equal(poisson_weights(keys, "task", 1, 0, 50), w1)


def test_integer_thresholds_match_float_cdf() -> None:
    from olmo_eval_api.stats.bootstrap import POISSON1_CDF

    rng = np.random.default_rng(0)
    y = rng.integers(0, 2**53, size=200_000, dtype=np.uint64)
    edges = np.array([int(np.ceil(c * 2.0**53)) for c in POISSON1_CDF], dtype=np.uint64)
    y = np.concatenate([y, edges[:-1], edges[:-1] - np.uint64(1)])
    as_float = np.searchsorted(POISSON1_CDF, y.astype(np.float64) * 2.0**-53, side="right")
    from olmo_eval_api.stats import bootstrap

    as_int = np.searchsorted(bootstrap._CDF_THRESHOLDS, y, side="right")
    assert np.array_equal(as_float, as_int)


def test_resampled_means_chunking_matches() -> None:
    from olmo_eval_api.stats import bootstrap

    keys = keys_for(300)
    rng = np.random.default_rng(0)
    d = rng.normal(size=(300, 2))
    m = np.ones_like(d, dtype=bool)
    m[:50, 1] = False
    full = resampled_means(keys, d, m, "t", 0, 500)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(bootstrap, "MAX_WEIGHT_CELLS", 300 * 7)  # forces 7-row chunks
        chunked = resampled_means(keys, d, m, "t", 0, 500)
    assert np.allclose(full, chunked)


def test_paired_delta_deterministic_and_independent_of_other_subjects() -> None:
    rng = np.random.default_rng(3)
    a = side((rng.random(200) < 0.6).astype(float))
    b = side((rng.random(200) < 0.5).astype(float))
    c = side((rng.random(300) < 0.4).astype(float), keys_for(300))
    d1 = compare(a, b)
    d2 = compare(a, b)
    assert d1.as_dict() == d2.as_dict()
    assert d1.method == "paired_bootstrap" and d1.n_shared == 200 and d1.n_boot == 2000
    assert a.values is not None and b.values is not None
    assert d1.delta is not None and d1.ci_low is not None and d1.ci_high is not None
    assert d1.delta == pytest.approx(float(a.values.mean() - b.values.mean()))
    assert d1.ci_low < d1.delta < d1.ci_high
    both = compare_pairs(
        [(a, b), (c, b)], task_name="t", alpha=0.05, n_boot=2000, seed=0, higher_is_better=True
    )
    assert both[0].ci_low == d1.ci_low and both[0].ci_high == d1.ci_high
    assert both[0].p_value == d1.p_value
    assert d1.improved is (d1.delta > 0)


def test_bootstrap_ci_coverage() -> None:
    """95% intervals cover the true difference (0.05) about 95% of the time."""
    rng = np.random.default_rng(7)
    sims, n, covered = 300, 200, 0
    for s in range(sims):
        a_vals = (rng.random(n) < 0.55).astype(float)
        b_vals = (rng.random(n) < 0.50).astype(float)
        d = compare(side(a_vals), side(b_vals), n_boot=400, seed=s)
        assert d.ci_low is not None and d.ci_high is not None
        covered += d.ci_low <= 0.05 <= d.ci_high
    assert 0.90 <= covered / sims <= 0.99


def test_insufficient_and_none_paths() -> None:
    a = side(np.ones(10))
    b = side(np.zeros(10))
    d = compare(a, b)
    assert d.method == "insufficient" and d.n_shared == 10 and d.delta == 1.0
    assert d.note and d.ci_low is None
    not_mean = side(np.ones(30), score=5.0, score_is_mean=False, stderr=None)
    d = compare(not_mean, side(np.zeros(30)))
    assert d.method == "none" and d.delta == pytest.approx(5.0)
    no_vec = Side("t", "h", 0.6, 0.02, True)
    d = compare(no_vec, Side("t", "h", 0.5, 0.02, True))
    assert d.method == "unpaired"
    d = compare(Side("t", "h", 0.6, None, False), Side("t", "h", 0.5, 0.01, True))
    assert d.method == "none"


def test_unpaired_known_values() -> None:
    d = unpaired(Side("t", "a", 0.6, 0.03, True), Side("t", "b", 0.5, 0.04, True), 0.05, True)
    assert d.delta == pytest.approx(0.1)
    assert d.ci_low == pytest.approx(0.1 - 1.959964 * 0.05, abs=1e-5)
    assert d.p_value == pytest.approx(2 * (1 - 0.9772499), abs=1e-5)  # z = 2
    assert d.significant is True and d.improved is True
    d = unpaired(Side("t", "a", 0.6, 0.03, True), Side("t", "b", 0.5, 0.04, True), 0.05, False)
    assert d.improved is False


def test_hash_mismatch_flag() -> None:
    rng = np.random.default_rng(0)
    a = side((rng.random(50) < 0.5).astype(float), task_hash="x")
    b = side((rng.random(50) < 0.5).astype(float), task_hash="y")
    assert compare(a, b).hash_mismatch is True


@pytest.mark.parametrize(
    ("wins", "losses", "expected"),
    [
        (5, 0, 0.0625),
        (0, 5, 0.0625),
        (3, 3, 1.0),
        (10, 2, 2 * (1 + 12 + 66) / 4096),
        (60, 40, 0.0569),
    ],
)
def test_sign_test_exact(wins: int, losses: int, expected: float) -> None:
    assert sign_test(wins, losses) == pytest.approx(expected, abs=5e-5)


def test_sign_test_large_n_is_finite() -> None:
    p = sign_test(6000, 4000)
    assert p is not None and 0 <= p < 1e-80
    assert sign_test(0, 0) is None


def test_mde_factor() -> None:
    assert mde_factor(0.05) == pytest.approx(2.8016, abs=1e-4)
    assert mde80([0.01, 0.02, 0.03], 0.05) == pytest.approx(2.8016 * 0.02, abs=1e-6)
    assert mde80([None], 0.05) is None


def test_contingency_counts() -> None:
    a = side(np.array([1, 1, 0, 0, 1], dtype=float))
    b = side(np.array([1, 0, 1, 0, np.nan]))
    c = contingency(a, b, 0.5)
    assert c == {
        "both_right": 1,
        "only_a": 1,
        "only_b": 1,
        "both_wrong": 1,
        "n_shared": 4,
        "net": 0,
        "threshold": 0.5,
    }
    lower = side(np.array([0.1, 0.9]), kind="bounded", higher_is_better=False)
    other = side(np.array([0.9, 0.9]), kind="bounded", higher_is_better=False)
    c = contingency(lower, other, 0.5)
    assert c is not None and c["only_a"] == 1 and c["both_wrong"] == 1
    assert contingency(side(np.ones(3), kind="unbounded"), side(np.ones(3)), 0.5) is None


def test_stratified_weights_and_pair_stats() -> None:
    rng = np.random.default_rng(5)
    tasks = []
    means = {}
    for name, n in (("t1", 100), ("t2", 300)):
        keys = keys_for(n, name)
        a = side((rng.random(n) < 0.7).astype(float), keys, task_name=name)
        b = side((rng.random(n) < 0.5).astype(float), keys, task_name=name)
        assert a.values is not None and b.values is not None
        means[name] = float(np.nanmean(a.values) - np.nanmean(b.values))
        tasks.append((name, {"A": a, "B": b}))
    acc = stratified_pairs(
        tasks,
        [("A", "B")],
        lambda names: {"t1": 0.25, "t2": 0.75} if set(names) == {"t1", "t2"} else {},
        alpha=0.05,
        n_boot=1000,
        seed=0,
        higher_is_better=True,
    )[("A", "B")]
    assert acc.point == pytest.approx(0.25 * means["t1"] + 0.75 * means["t2"])
    assert acc.n_shared == 400
    st = pair_stats(acc, 0.05, True)
    assert st["method"] == "paired_bootstrap" and st["significant"]
    assert st["wins"] + st["losses"] + st["ties"] == 400
    assert st["win_rate"] == pytest.approx(st["wins"] / (st["wins"] + st["losses"]))
    assert 0.9 < st["prob_row_better"] <= 1.0
    # A task with too few shared instances does not qualify.
    small = [("t3", {"A": side(np.ones(5)), "B": side(np.zeros(5))})]
    acc = stratified_pairs(
        small, [("A", "B")], lambda names: {}, alpha=0.05, n_boot=200, seed=0, higher_is_better=True
    )[("A", "B")]
    assert pair_stats(acc, 0.05, True)["method"] == "none"


def _subject_tasks(n_subjects: int) -> list[tuple[str, dict[str, Side]]]:
    """Two tasks; subjects cover overlapping but different instance sets."""
    rng = np.random.default_rng(11)
    tasks = []
    for name, n in (("t1", 120), ("t2", 80)):
        sides = {}
        for k in range(n_subjects):
            keys = keys_for(n + 10 * k, name)[10 * k :]
            vals = (rng.random(keys.size) < 0.4 + 0.05 * k).astype(float)
            vals[rng.random(keys.size) < 0.05] = np.nan
            sides[f"s{k}"] = side(vals, keys, task_name=name)
        tasks.append((name, sides))
    return tasks


def test_stratified_blocks_match_unblocked() -> None:
    tasks = _subject_tasks(5)
    names = [f"s{k}" for k in range(5)]
    pairs = [(names[i], names[j]) for i in range(5) for j in range(i + 1, 5)]

    def run(block_cells: int | None) -> dict:
        return stratified_pairs(
            tasks,
            pairs,
            lambda q: {n: 1 / len(q) for n in q},
            alpha=0.05,
            n_boot=300,
            seed=3,
            higher_is_better=True,
            margin=0.0,
            block_cells=block_cells,
        )

    whole, blocked = run(None), run(1)  # block_cells=1 puts every pair in its own block
    for p in pairs:
        a, b = whole[p], blocked[p]
        assert a.weights == b.weights
        assert (a.point, a.var_terms, a.n_shared) == (b.point, b.var_terms, b.n_shared)
        assert (a.wins, a.losses, a.ties) == (b.wins, b.losses, b.ties)
        np.testing.assert_array_equal(a.boot, b.boot)
        np.testing.assert_array_equal(a.win_num, b.win_num)
        np.testing.assert_array_equal(a.win_den, b.win_den)


def test_compare_pairs_blocks_match_unblocked() -> None:
    sides = _subject_tasks(6)[0][1]
    base = sides.pop("s0")
    pairs = [(sd, base) for sd in sides.values()]
    kw: dict[str, Any] = dict(
        task_name="t1", alpha=0.05, n_boot=300, seed=1, higher_is_better=True, keep_boot=True
    )
    whole = compare_pairs(pairs, **kw)
    blocked = compare_pairs(pairs, block_cells=1, **kw)
    for a, b in zip(whole, blocked, strict=True):
        assert a.as_dict() == b.as_dict()
        assert a.boot is not None and b.boot is not None
        np.testing.assert_array_equal(a.boot, b.boot)


def test_stratified_work_budget() -> None:
    tasks = _subject_tasks(4)
    pairs = [("s0", "s1"), ("s0", "s2"), ("s1", "s2"), ("s2", "s3")]
    kw: dict[str, Any] = dict(alpha=0.05, n_boot=200, seed=0, higher_is_better=True)

    def weights(q: Any) -> dict[str, float]:
        return {n: 1 / len(q) for n in q}

    # Union sizes are 150 (t1) and 110 (t2); all four pairs qualify on both tasks.
    needed = 4 * (150 + 110) * 200
    stratified_pairs(tasks, pairs, weights, max_work=needed, **kw)
    with pytest.raises(WorkBudgetExceeded, match="bootstrap cells"):
        stratified_pairs(tasks, pairs, weights, max_work=needed - 1, **kw)
    with pytest.raises(WorkBudgetExceeded, match="resamples"):
        stratified_pairs(tasks, pairs, weights, max_pair_resamples=4 * 200 - 1, **kw)


def test_histograms_and_box() -> None:
    assert histogram([0, 1, 1], 10, binary=True) == {"edges": [-0.5, 0.5, 1.5], "counts": [1, 2]}
    h = histogram(np.arange(10.0), 5)
    assert h["counts"] == [2, 2, 2, 2, 2] and len(h["edges"]) == 6
    assert histogram([3.0, 3.0], 5) == {"edges": [2.5, 3.5], "counts": [2]}
    b = box_stats([1, 2, 3, 4, 5])
    assert b is not None and b["p50"] == 3 and b["n"] == 5
    assert box_stats([]) is None


def test_matrix_scale_performance() -> None:
    """10 subjects x 200 tasks x 500 instances: all deltas against one baseline."""
    rng = np.random.default_rng(0)
    keys = keys_for(500)
    start = time.perf_counter()
    for t in range(200):
        base = side((rng.random(500) < 0.5).astype(float), keys, task_name=f"task{t}")
        others = [
            side((rng.random(500) < 0.55).astype(float), keys, task_name=f"task{t}")
            for _ in range(9)
        ]
        compare_pairs(
            [(o, base) for o in others],
            task_name=f"task{t}",
            alpha=0.05,
            n_boot=2000,
            seed=0,
            higher_is_better=True,
        )
    elapsed = time.perf_counter() - start
    assert elapsed < 5, f"took {elapsed:.2f}s"


async def test_cache_key_changes_on_reupload(client: httpx.AsyncClient, session: Any) -> None:
    from olmo_eval_api.services.cache import cache_key
    from olmo_eval_api.services.subjects import run_task_results

    await seed_run(client, run_id="cache0000001", tasks={"t": [1.0, 0.0] * 15})
    trs1 = await run_task_results(session, "cache0000001")
    k1 = cache_key("x", {"a": 1}, trs1)
    assert k1 == cache_key("x", {"a": 1}, trs1)
    assert k1 != cache_key("x", {"a": 2}, trs1)
    await seed_run(client, run_id="cache0000001", tasks={"t": [1.0, 1.0] * 15})
    session.expire_all()
    trs2 = await run_task_results(session, "cache0000001")
    assert trs1[0]["id"] == trs2[0]["id"]
    assert cache_key("x", {"a": 1}, trs2) != k1


def test_vector_cache_is_capped_by_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    from olmo_eval_api.services import subjects

    monkeypatch.setattr(subjects, "_VECTOR_CACHE", type(subjects._VECTOR_CACHE)())
    monkeypatch.setattr(subjects, "_vector_cache_bytes", 0)
    monkeypatch.setattr(subjects, "_VECTOR_CACHE_MAX_BYTES", 16 * 1000 * 4)

    def vec(n: int) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros(n, dtype=np.int64), np.zeros(n, dtype=np.float64)

    for i in range(10):
        subjects._cache_put((i, "t", "m"), vec(1000))  # 16 kB each
    assert list(subjects._VECTOR_CACHE) == [(i, "t", "m") for i in (6, 7, 8, 9)]
    assert subjects._vector_cache_bytes == 4 * 16_000
    subjects._cache_put((99, "t", "m"), vec(2000))  # over a quarter of the cap: not cached
    assert subjects._cache_get((99, "t", "m")) is None
