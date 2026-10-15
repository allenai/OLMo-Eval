"""Derived identity and name helpers (spec 2.6), settings defaults, and suite math."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import pytest

from olmo_eval_api.services import derive
from olmo_eval_api.services.suites import (
    aggregate,
    build_tree,
    leaf_weights,
    leaves,
    propagate_stderr,
)
from olmo_eval_api.settings import Settings


def test_incomplete_gap_headline_does_not_fall_back_to_task_mean() -> None:
    from olmo_eval_api.db.models import Run, SuiteResult, TaskResult
    from olmo_eval_api.services.ingest import _headline

    run = Run(run_id="run", model_id="model", task_specs=["pair"])
    gap = SuiteResult(run_id="run", parent_suite=None, score=None, aggregation="gap")
    tasks = [
        TaskResult(run_id="run", score=value, display_format="percent", higher_is_better=True)
        for value in (0.5, 1.0)
    ]
    assert _headline(run, [gap], tasks) is None
    bare = _headline(run, [], tasks)
    assert bare is not None and bare["score"] == 0.75


@pytest.mark.parametrize(
    "higher_is_better,first", [(True, "r:run00002"), (False, "r:run00001"), (None, "r:run00002")]
)
def test_pairwise_ties_rank_by_metric_improvement(higher_is_better, first) -> None:
    from olmo_eval_api.routers.api.compare import pairwise_order_key
    from olmo_eval_api.schemas.api import PairwiseRowSummary

    rows = [
        PairwiseRowSummary(
            subject=key, mean_win_rate=0.5, mean_delta=delta, ci_low=None, ci_high=None
        )
        for key, delta in [("r:run00001", -1.0), ("r:run00002", 1.0)]
    ]
    assert (
        sorted(rows, key=lambda row: pairwise_order_key(row, higher_is_better))[0].subject == first
    )
    rows[0].mean_win_rate = 1.0
    assert (
        sorted(rows, key=lambda row: pairwise_order_key(row, higher_is_better))[0].subject
        == "r:run00001"
    )


def test_model_id_and_key_hash() -> None:
    assert (
        derive.model_id("meta-llama/Llama-3.1-8B-Instruct", "16614dd9d1ac227a")
        == (derive.sha256_hex("meta-llama/Llama-3.1-8B-Instruct\x0016614dd9d1ac227a")[:12])
    )
    h = derive.instance_key_hash("q1")
    assert -(2**63) <= h < 2**63
    assert h == derive.instance_key_hash("q1") != derive.instance_key_hash("q2")


@pytest.mark.parametrize(
    ("candidates", "expected"),
    [
        ((None, "allenai/OLMo-2-7B-stage1-step239000-tokens1003B", None), 239000),
        (("step1000", "anything", None), 1000),
        ((None, "model_step_42", None), 42),
        ((None, "mystep5", None), None),
        ((None, "Qwen/Qwen3-8B", "/weka/ckpt/step12345-hf"), 12345),
    ],
)
def test_derive_step(candidates: tuple, expected: int | None) -> None:
    assert derive.derive_step(*candidates) == expected


def test_derive_tokens() -> None:
    assert derive.derive_tokens("OLMo-2-7B-stage1-step239000-tokens1003B") == 1_003_000_000_000
    assert derive.derive_tokens("x-tokens2.5T") == 2_500_000_000_000
    assert derive.derive_tokens("plain") is None


@pytest.mark.parametrize(
    ("name", "series"),
    [
        ("allenai/OLMo-2-7B-stage1-step239000-tokens1003B", "allenai/OLMo-2-7B"),
        ("olmo3-7b/step12000-hf", "olmo3-7b"),
        ("my-model_step500", "my-model"),
        ("Qwen/Qwen3-8B", "Qwen/Qwen3-8B"),
        ("step100", "step100"),
    ],
)
def test_derive_series(name: str, series: str) -> None:
    assert derive.derive_series(name) == series


def test_series_label() -> None:
    assert derive.series_label("allenai/OLMo-2-7B") == "OLMo-2-7B"
    assert derive.series_label("olmo3") == "olmo3"


@pytest.mark.parametrize(
    ("name", "path", "family"),
    [
        ("allenai/OLMo-3-7B", "", "olmo3"),
        ("meta-llama/Llama-3.1-8B", "", "llama3.1"),
        ("Qwen/Qwen3-8B", "", "qwen3"),
        ("", "/weka/models/gemma-2-9b", "gemma2"),
        ("custom-thing_v2", "", "custom"),
        ("mistral_7b", "", "mistral7"),
    ],
)
def test_derive_family(name: str, path: str, family: str | None) -> None:
    assert derive.derive_family(name, path) == family


def test_settings_hash_ignores_paths_and_operational_keys() -> None:
    a = {
        "kind": "vllm",
        "model": "/ckpt/step100",
        "num_instances": 4,
        "kwargs": {"tensor_parallel_size": 2, "dtype": "bfloat16"},
    }
    b = {
        "kind": "vllm",
        "model": "/ckpt/step200",
        "num_instances": 1,
        "kwargs": {"tensor_parallel_size": 8, "dtype": "bfloat16"},
    }
    c = {**b, "kwargs": {"dtype": "float16"}}
    assert derive.settings_hash(a) == derive.settings_hash(b) != derive.settings_hash(c)
    assert len(derive.settings_hash(a)) == 12


def test_metric_kind_and_flatten() -> None:
    assert derive.metric_kind([0.0, 1.0, 1.0]) == "binary"
    assert derive.metric_kind([0.0, 0.5]) == "bounded"
    assert derive.metric_kind([-1.0, 1.0]) == "unbounded"
    flat = derive.flatten_metrics({"acc": {"default": 0.5, "x": float("nan")}})
    assert flat == {"acc:default": 0.5, "acc:x": None}
    assert derive.metric_name("acc:default") == "acc"


def test_settings_defaults_per_env() -> None:
    prod = Settings(skiff_env="prod")
    assert prod.db_name == "olmo_eval"
    assert prod.results_prefix == ""
    assert prod.storage_backend == "gcs"
    assert prod.dashboard_base_url == "https://olmo-eval.allen.ai"
    branch = Settings(skiff_env="chrisg-skiff2-dashboard")
    assert branch.db_name == "olmo_eval_dev"
    assert branch.results_prefix == "dev/"
    assert branch.dashboard_base_url == (
        "https://chrisg-skiff2-dashboard-ui.olmo-eval.apps.allenai.org"
    )
    assert branch.gcs_prefix("abc123") == "gs://ai2-skiff2-olmo-eval-results/dev/runs/abc123/"
    local = Settings(skiff_env="local", db_url="postgresql+asyncpg://x")
    assert local.storage_backend == "local"
    assert local.dashboard_base_url == "http://localhost:5173"
    assert Settings(skiff_env="local").storage_backend == "gcs"
    with pytest.raises(ValueError):
        Settings(skiff_env="prod", api_mode="all")


def test_settings_refuse_local_mode_on_cloud_run(monkeypatch: pytest.MonkeyPatch) -> None:
    # SKIFF_ENV defaults to local, which enables dev identities; Cloud Run must never get that.
    monkeypatch.delenv("SKIFF_ENV", raising=False)
    monkeypatch.setenv("K_SERVICE", "ingest")
    with pytest.raises(ValueError, match="Cloud Run"):
        Settings()
    assert Settings(skiff_env="prod").k_service == "ingest"


def test_settings_csv_lists(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INGEST_ALLOWED_SERVICE_ACCOUNTS", "a@x.iam.gserviceaccount.com, B@y.com")
    monkeypatch.setenv("INGEST_ALLOWED_DOMAINS", "allenai.org,Example.com")
    s = Settings()
    assert s.ingest_allowed_service_accounts == ["a@x.iam.gserviceaccount.com", "b@y.com"]
    assert s.ingest_allowed_domains == ["allenai.org", "example.com"]


DEFS = {
    "root": ("average", [{"type": "suite", "name": "mc"}, {"type": "task", "name": "gen"}]),
    "mc": ("average", [{"type": "task", "name": "a"}, {"type": "task", "name": "b"}]),
    "w": ("weighted_average", [{"type": "task", "name": "a"}, {"type": "task", "name": "b"}]),
    "cyc": ("average", [{"type": "suite", "name": "cyc"}]),
    "aoa": (
        "average_of_averages",
        [{"type": "suite", "name": "w"}, {"type": "task", "name": "gen"}],
    ),
}


def test_suite_tree_and_weights() -> None:
    tree = build_tree("root", DEFS)
    assert leaves(tree) == ["a", "b", "gen"]
    # average weighs every expanded task equally, as olmo-eval does.
    w = leaf_weights(tree, {}, {"a", "b", "gen"})
    assert w == pytest.approx({"a": 1 / 3, "b": 1 / 3, "gen": 1 / 3})
    w = leaf_weights(tree, {}, {"a", "gen"})
    assert w == pytest.approx({"a": 0.5, "gen": 0.5})
    # average_of_averages gives each child an equal share; a weighted child splits by instances.
    aoa = leaf_weights(build_tree("aoa", DEFS), {"a": 30, "b": 10}, {"a", "b", "gen"})
    assert aoa == pytest.approx({"a": 0.375, "b": 0.125, "gen": 0.5})
    assert leaf_weights(tree, {}, set()) == {}
    weighted = leaf_weights(build_tree("w", DEFS), {"a": 30, "b": 10}, {"a", "b"})
    assert weighted == pytest.approx({"a": 0.75, "b": 0.25})
    with pytest.raises(ValueError):
        build_tree("cyc", DEFS)
    unknown = build_tree("missing", DEFS)
    assert unknown.children == () and leaves(unknown) == []


def test_aggregate_and_stderr() -> None:
    tree = build_tree("root", DEFS)
    assert aggregate(tree, {"a": 0.2, "b": 0.4, "gen": 0.9}, {}) == pytest.approx(0.5)
    assert aggregate(tree, {"a": None, "b": None, "gen": None}, {}) is None
    display = build_tree("d", {"d": ("display_only", [{"type": "task", "name": "a"}])})
    assert aggregate(display, {"a": 1.0}, {}) is None
    se = propagate_stderr({"a": 0.5, "b": 0.5}, {"a": 0.1, "b": 0.2})
    assert se == pytest.approx(math.sqrt(0.25 * 0.01 + 0.25 * 0.04))
    assert propagate_stderr({"a": 1.0}, {"a": None}) is None


def test_weighted_suite_with_missing_counts_has_no_fallback_score() -> None:
    tree = build_tree("w", DEFS)
    scores = {"a": 0.2, "b": 0.9}
    assert aggregate(tree, scores, {}) is None
    assert aggregate(tree, scores, {"a": 30}) is None
    assert aggregate(tree, scores, {"a": 30, "b": 0}) is None
    assert aggregate(tree, scores, {"a": 30, "b": 10}) == pytest.approx(0.375)


def test_suite_generation_counts_match_scored_members_and_omit_legacy_counts() -> None:
    from olmo_eval_api.services.read_suites import suite_generation_counts

    tasks = {
        "a": {
            "score": 0.5,
            "metrics": {"acc:x": 0.5},
            "generation_counts": {"generations": 4, "cap_hit": 2},
        },
        "b": {"metrics": {}, "error": "failed", "generation_counts": {"generations": 50}},
        "gen": {
            "score": 0.5,
            "metrics": {"acc:x": 0.5},
            "generation_counts": {"generations": 6, "empty": 1},
        },
        "outside": {"metrics": {"acc:x": 1.0}, "generation_counts": {"generations": 99}},
    }
    node = build_tree("root", DEFS)
    assert suite_generation_counts(node, tasks) == {"generations": 10, "cap_hit": 2, "empty": 1}
    assert suite_generation_counts(node, {"a": {"metrics": {"acc:x": 0.5}}}) is None


def test_quantile_box_uses_mean_when_sent() -> None:
    from olmo_eval_api.routers.api.run_tabs import _quantile_box

    q = {"p5": 1.0, "p25": 2.0, "p50": 3.0, "p75": 4.0, "p95": 5.0, "n": 10}
    box = _quantile_box(q)
    assert box is not None and box.mean == 3.0
    box = _quantile_box({**q, "mean": 3.5})
    assert box is not None and box.mean == 3.5


def test_gap_suite_ranks_by_companion_and_uses_its_uncertainty() -> None:
    from olmo_eval_api.services.read_suites import score_suite

    tree = build_tree(
        "omega:dev",
        {"omega:dev": ("gap", [{"type": "task", "name": "in"}, {"type": "task", "name": "out"}])},
    )
    assert aggregate(tree, {"in": 0.6, "out": 0.45}, {}) == 0.45
    assert aggregate(tree, {"in": 0.0, "out": 0.0}, {}) == 0.0
    assert aggregate(tree, {"in": 0.1, "out": 0.45}, {}) == 0.45
    assert aggregate(tree, {"in": None, "out": 0.45}, {}) is None
    assert aggregate(tree, {"in": 0.6}, {}) is None
    weights = leaf_weights(tree, {}, {"in", "out"})
    assert weights == {"out": 1.0}
    assert leaf_weights(tree, {}, {"in", "out"}, metric_keys={"in": "strict", "out": "flex"}) == {}
    assert (
        leaf_weights(tree, {}, {"in", "out"}, metric_keys={"in": "strict", "out": "strict"})
        == weights
    )
    assert propagate_stderr(weights, {"in": 0.3, "out": 0.02}) == 0.02
    task_results = {
        name: {
            "score": score,
            "stderr": 0.02,
            "num_instances": 500,
            "display_format": "percent",
            "higher_is_better": True,
            "primary_metric": "exact_match:exact_match",
        }
        for name, score in (("in", 0.6), ("out", 0.45))
    }
    assert score_suite(tree, task_results).score == 0.45
    task_results["in"]["primary_metric"] = "exact_match_flex:exact_match_flex"
    assert score_suite(tree, task_results).score is None
    overrides = {"in": 0.6, "out": 0.45}
    assert score_suite(tree, task_results, overrides).score is None
    assert (
        score_suite(
            tree,
            task_results,
            overrides,
            metric_keys={"in": "exact_match:exact_match", "out": "exact_match:exact_match"},
        ).score
        == 0.45
    )


def test_gap_matrix_overrides_validate_metrics_before_scores_and_deltas() -> None:
    from olmo_eval_api.routers.api.compare import RowSpec, _matrix_suite_row
    from olmo_eval_api.schemas.api import MatrixRequest
    from olmo_eval_api.services.subjects import Subject
    from olmo_eval_api.stats.paired import Side

    node = build_tree(
        "pair", {"pair": ("gap", [{"type": "task", "name": "in"}, {"type": "task", "name": "out"}])}
    )
    subject = Subject(
        key="r:model001",
        kind="run",
        model={},
        run={"run_id": "model001"},
        trs={
            name: {
                "score": score,
                "stderr": 0.02,
                "num_instances": 500,
                "display_format": "percent",
                "higher_is_better": True,
                "primary_metric": metric,
            }
            for name, score, metric in (("in", 0.6, "strict"), ("out", 0.45, "flex"))
        },
    )
    sides = {
        name: {subject.key: Side(name, "hash", tr["score"], 0.02, False)}
        for name, tr in subject.trs.items()
    }
    spec = RowSpec("suite:pair", "suite", "pair", 0, None, "gap", node)
    body = MatrixRequest(subjects=[subject.key], scope="suite:pair", metric="primary")
    row = _matrix_suite_row(
        spec, [subject], subject, sides, body, None, metric_keys={"in": "strict", "out": "flex"}
    )
    assert row.cells[0].score is None
    assert row.cells[0].status == "partial"
    assert row.cells[0].delta is None
    row = _matrix_suite_row(
        spec, [subject], None, sides, body, None, metric_keys={"in": "strict", "out": "strict"}
    )
    assert row.cells[0].score == 0.45


def test_primary_gap_metrics_come_from_one_complete_subject() -> None:
    from olmo_eval_api.routers.api.compare import scope_metric_keys
    from olmo_eval_api.services.read_suites import score_suite
    from olmo_eval_api.services.subjects import Subject

    node = build_tree(
        "pair", {"pair": ("gap", [{"type": "task", "name": "in"}, {"type": "task", "name": "out"}])}
    )
    complete: dict[str, Mapping[Any, Any]] = {
        name: {
            "primary_metric": "strict",
            "score": score,
            "stderr": 0.02,
            "num_instances": 500,
            "display_format": "percent",
            "higher_is_better": True,
        }
        for name, score in (("in", 0.6), ("out", 0.45))
    }
    subjects = [
        Subject("r:model001", "run", {}, None, trs={"in": complete["in"]}),
        Subject(
            "r:model002",
            "run",
            {},
            None,
            trs={"out": {**complete["out"], "primary_metric": "flex"}},
        ),
        Subject("r:model003", "run", {}, None, trs=complete),
    ]
    keys = scope_metric_keys(["in", "out"], "primary", subjects, None, node)
    assert keys == {"in": "strict", "out": "strict"}
    assert score_suite(node, complete, metric_keys=keys).score == 0.45
    assert scope_metric_keys(["in", "out"], "chosen", subjects, None, node) == {
        "in": "chosen",
        "out": "chosen",
    }


@pytest.mark.parametrize(
    "metrics,expected",
    [(("strict", "strict"), "strict"), (("strict", "flex"), None), (("strict", None), None)],
)
def test_gap_baseline_metric_identity_requires_a_complete_shared_primary(metrics, expected):
    from olmo_eval_api.routers.api.runs import gap_metric_identity

    tasks = {
        name: {"score": 0.0, "primary_metric": metric}
        for name, metric in zip(("in", "out"), metrics, strict=True)
    }
    assert gap_metric_identity(tasks, ["in", "out"]) == expected
    assert gap_metric_identity({"out": tasks["out"]}, ["in", "out"]) is None


@pytest.mark.parametrize(
    "changes",
    [
        {},
        {"_suite_aggregation": "average"},
        {"_suite_hashes": {"pair": "v2"}},
        {"_gap_metric": "flex"},
    ],
)
def test_gap_columns_require_the_same_stored_definition_and_metric(changes):
    from olmo_eval_api.routers.api.runs import suite_columns_compatible

    row = {
        "_suite_aggregation": "gap",
        "_suite_hashes": {"pair": "v1", "nested": "child-v1"},
        "_gap_metric": "strict",
    }
    assert suite_columns_compatible(row, {**row, **changes}) is (not changes)


def test_gap_selected_metric_controls_score_format_and_direction() -> None:
    from olmo_eval_api.services.read_suites import score_suite

    node = build_tree(
        "pair", {"pair": ("gap", [{"type": "task", "name": "in"}, {"type": "task", "name": "out"}])}
    )
    tasks = {
        name: {
            "primary_metric": "loss:default",
            "score": 2.0,
            "stderr": 0.3,
            "num_instances": 10,
            "display_format": "raw",
            "higher_is_better": True,
            "metrics": {"error:default": value},
            "metric_meta": {
                "error:default": {
                    "display_format": "percent",
                    "higher_is_better": False,
                    "stderr": 0.02,
                }
            },
        }
        for name, value in (("in", 0.2), ("out", 0.1))
    }
    result = score_suite(node, tasks, metric_keys={"in": "error:default", "out": "error:default"})
    assert result.score == 0.1 and result.stderr == 0.02
    assert result.display_format == "percent" and result.higher_is_better is False


def test_missing_selected_metric_does_not_change_scored_suite_units() -> None:
    from olmo_eval_api.services.read_suites import score_suite

    node = build_tree(
        "root",
        {"root": ("average", [{"type": "task", "name": "a"}, {"type": "task", "name": "b"}])},
    )
    rows = {
        name: {
            "primary_metric": "primary",
            "score": 0.5,
            "stderr": 0.02,
            "num_instances": 10,
            "display_format": "percent",
            "higher_is_better": True,
            "metrics": {},
            "metric_meta": {},
        }
        for name in ("a", "b")
    }
    rows["a"]["metrics"] = {"selected": 0.4}
    rows["a"]["metric_meta"] = {
        "selected": {"display_format": "percent", "higher_is_better": False}
    }
    result = score_suite(node, rows, metric_keys={"a": "selected", "b": "selected"})
    assert result.score == 0.4
    assert result.display_format == "percent" and result.higher_is_better is False


def test_gap_metric_selection_requires_a_complete_primary_pair() -> None:
    from olmo_eval_api.routers.api.compare import scope_metric_keys
    from olmo_eval_api.services.subjects import Subject

    node = build_tree(
        "pair", {"pair": ("gap", [{"type": "task", "name": "in"}, {"type": "task", "name": "out"}])}
    )
    subjects = [
        Subject(
            "r:in000001", "run", {}, None, trs={"in": {"primary_metric": "strict", "score": 0.5}}
        ),
        Subject(
            "r:out00001", "run", {}, None, trs={"out": {"primary_metric": "strict", "score": 0.4}}
        ),
    ]
    assert scope_metric_keys(["in", "out"], "primary", subjects, None, node) == {
        "in": None,
        "out": None,
    }


@pytest.mark.parametrize("out_metric,out_score", [("flex", 0.4), ("strict", None)])
def test_incomplete_gap_keeps_individual_task_matrix_scores(out_metric, out_score) -> None:
    from olmo_eval_api.routers.api.compare import (
        RowSpec,
        _matrix_suite_row,
        _matrix_task_row,
        scope_metric_keys,
    )
    from olmo_eval_api.schemas.api import MatrixRequest
    from olmo_eval_api.services.subjects import Subject
    from olmo_eval_api.stats.paired import Side

    node = build_tree(
        "pair", {"pair": ("gap", [{"type": "task", "name": "in"}, {"type": "task", "name": "out"}])}
    )
    subject = Subject(
        "r:model001",
        "run",
        {},
        {"run_id": "model001"},
        trs={
            name: {
                "id": i,
                "run_id": "model001",
                "task_hash": "hash",
                "primary_metric": metric,
                "score": score,
                "stderr": 0.02,
                "num_instances": 500,
                "error": None,
                "metric_meta": None,
                "metric_kind": "bounded",
                "display_format": "percent",
                "higher_is_better": True,
            }
            for i, (name, metric, score) in enumerate(
                [("in", "strict", 0.5), ("out", out_metric, out_score)], start=1
            )
        },
    )
    task_keys = scope_metric_keys(
        ["in", "out"], "primary", [subject], None, node, require_gap_anchor=False
    )
    suite_keys = scope_metric_keys(["in", "out"], "primary", [subject], None, node)
    assert task_keys == {"in": "strict", "out": out_metric}
    assert suite_keys == {"in": None, "out": None}
    sides = {
        name: {subject.key: Side(name, "hash", tr["score"], 0.02, False)}
        for name, tr in subject.trs.items()
    }
    body = MatrixRequest(subjects=[subject.key], scope="suite:pair", metric="primary")
    coverage = dict.fromkeys(["complete", "missing", "failed", "hash_mismatch"], 0)
    for name, tr in subject.trs.items():
        if tr["score"] is None:
            continue
        row = _matrix_task_row(
            RowSpec(f"task:{name}", "task", name, 1, "suite:pair"),
            [subject],
            None,
            sides[name],
            task_keys[name],
            body,
            None,
            False,
            [],
            coverage,
        )
        assert row.cells[0].score == tr["score"]
        assert row.cells[0].status == "ok"
        assert row.metric_key == tr["primary_metric"]
        assert row.meta is not None and row.meta.display_format == "percent"
    assert coverage["missing"] == 0
    suite_row = _matrix_suite_row(
        RowSpec("suite:pair", "suite", "pair", 0, None, "gap", node),
        [subject],
        None,
        sides,
        body,
        None,
        metric_keys=suite_keys,
    )
    assert suite_row.cells[0].score is None
    assert suite_row.cells[0].status == "partial"
    assert suite_row.cells[0].delta is None
    assert leaf_weights(node, {}, {"in", "out"}, metric_keys=suite_keys) == {}


@pytest.mark.parametrize("reference_format,reference_direction", [("raw", None), ("percent", True)])
def test_gap_format_and_direction_come_from_its_companion(
    reference_format, reference_direction
) -> None:
    from olmo_eval_api.services.read_suites import score_suite

    node = build_tree(
        "pair", {"pair": ("gap", [{"type": "task", "name": "in"}, {"type": "task", "name": "out"}])}
    )
    tasks = {
        name: {
            "score": score,
            "stderr": stderr,
            "num_instances": 500,
            "primary_metric": "error_rate",
            "display_format": fmt,
            "higher_is_better": direction,
        }
        for name, score, stderr, fmt, direction in (
            ("in", 0.6, 0.03, reference_format, reference_direction),
            ("out", 0.4, 0.02, "percent", False),
        )
    }
    result = score_suite(node, tasks)
    assert result.score == 0.4
    assert result.stderr == pytest.approx(0.02)
    assert result.display_format == "percent"
    assert result.higher_is_better is False


def test_suite_counts_include_derived_scores_even_without_corpus_metrics_or_with_errors() -> None:
    from olmo_eval_api.services.read_suites import suite_generation_counts

    node = build_tree(
        "derived",
        {
            "derived": (
                "average",
                [
                    {"type": "task", "name": "a"},
                    {"type": "task", "name": "b"},
                    {"type": "task", "name": "c"},
                ],
            )
        },
    )
    tasks = {
        "a": {"score": 0.5, "metrics": {}, "generation_counts": {"generations": 4}},
        "b": {"score": 0.6, "error": "partial", "generation_counts": {"generations": 6}},
        "c": {"score": None, "metrics": {"acc:x": 0.5}, "generation_counts": {"generations": 99}},
    }
    assert suite_generation_counts(node, tasks) == {"generations": 10}


def test_gap_progression_points_use_one_population_metric() -> None:
    from olmo_eval_api.routers.api.catalog import _panel_score
    from olmo_eval_api.services.read_suites import primary_gap_metric_keys

    node = build_tree(
        "pair", {"pair": ("gap", [{"type": "task", "name": "in"}, {"type": "task", "name": "out"}])}
    )
    checkpoints: list[dict[str, dict[str, Any]]] = [
        {
            name: {
                "score": score,
                "stderr": 0.0,
                "num_instances": 32,
                "primary_metric": metric,
                "display_format": "percent",
                "higher_is_better": True,
                "metrics": {metric: score},
            }
            for name in ["in", "out"]
        }
        for metric, score in [("strict", 0.4), ("flex", 0.9)]
    ]
    keys = primary_gap_metric_keys(node, checkpoints)
    assert _panel_score(node, None, checkpoints[0], "primary", keys)[0] == 0.4
    assert _panel_score(node, None, checkpoints[1], "primary", keys)[0] is None
    checkpoints[1]["in"]["metrics"]["strict"] = 0.5
    checkpoints[1]["out"]["metrics"]["strict"] = 0.6
    assert _panel_score(node, None, checkpoints[1], "primary", keys)[0] == 0.6


def test_shared_gap_task_anchors_are_selected_per_matrix_row() -> None:
    from olmo_eval_api.routers.api.compare import _suite_rows, matrix_metric_keys
    from olmo_eval_api.services.subjects import Subject

    node = build_tree(
        "both",
        {
            "both": ("none", [{"type": "suite", "name": "a"}, {"type": "suite", "name": "b"}]),
            "a": ("gap", [{"type": "task", "name": "in"}, {"type": "task", "name": "out_a"}]),
            "b": ("gap", [{"type": "task", "name": "in"}, {"type": "task", "name": "out_b"}]),
        },
    )
    subjects = [
        Subject(
            "r:strict",
            "run",
            {},
            None,
            trs={
                name: {"primary_metric": "strict", "score": score}
                for name, score in [("in", 0.6), ("out_a", 0.4), ("out_b", None)]
            },
        ),
        Subject(
            "r:flex",
            "run",
            {},
            None,
            trs={
                name: {"primary_metric": "flex", "score": score}
                for name, score in [("in", 0.7), ("out_a", None), ("out_b", 0.5)]
            },
        ),
    ]
    rows = []
    _suite_rows(node, 0, None, rows)
    selected = matrix_metric_keys(rows, "primary", subjects, None)
    by_suite = {
        row.name: keys for row, keys in zip(rows, selected, strict=True) if row.kind == "suite"
    }
    assert by_suite["a"] == {"in": "strict", "out_a": "strict"}
    assert by_suite["b"] == {"in": "flex", "out_b": "flex"}
    in_tasks = [
        (row.parent, keys["in"])
        for row, keys in zip(rows, selected, strict=True)
        if row.kind == "task" and row.name == "in"
    ]
    assert in_tasks == [("a", "strict"), ("b", "flex")]


@pytest.mark.parametrize("missing_companion", [False, True])
def test_matrix_parent_keeps_leaves_from_unanchored_gap_child(missing_companion):
    from olmo_eval_api.routers.api.compare import RowSpec, _matrix_suite_row, matrix_metric_keys
    from olmo_eval_api.schemas.api import MatrixRequest
    from olmo_eval_api.services.subjects import Subject
    from olmo_eval_api.stats.paired import Side

    node = build_tree(
        "root",
        {
            "root": (
                "average",
                [{"type": "suite", "name": "pair"}, {"type": "task", "name": "other"}],
            ),
            "pair": ("gap", [{"type": "task", "name": "in"}, {"type": "task", "name": "out"}]),
        },
    )
    trs: dict[str, Mapping[Any, Any]] = {
        name: {
            "score": score,
            "stderr": 0.02,
            "num_instances": 10,
            "display_format": "percent",
            "higher_is_better": True,
            "primary_metric": metric,
        }
        for name, score, metric in [("in", 0.2, "acc"), ("out", 0.4, "f1"), ("other", 0.8, "acc")]
    }
    if missing_companion:
        del trs["out"]
    subject = Subject("r:model001", "run", {}, None, trs=trs)
    spec = RowSpec("suite:root", "suite", "root", 0, None, "average", node)
    checks = matrix_metric_keys([spec], "primary", [subject], None)[0]
    sides_keys = matrix_metric_keys([spec], "primary", [subject], None, require_gap_anchor=False)[0]
    assert checks["in"] == "acc"
    sides = {
        name: {subject.key: Side(name, "hash", trs[name]["score"], 0.02, False)}
        for name, key in sides_keys.items()
        if name in trs and key is not None
    }
    row = _matrix_suite_row(
        spec,
        [subject],
        None,
        sides,
        MatrixRequest(subjects=[subject.key], scope="suite:root", metric="primary"),
        None,
        metric_keys=checks,
    )
    assert row.cells[0].score == pytest.approx(sum(tr["score"] for tr in trs.values()) / len(trs))
    assert row.meta is not None
    assert row.meta.display_format == "percent"
    assert row.meta.higher_is_better is True


def test_gap_progression_meta_prefers_the_primary_anchor_subject():
    from olmo_eval_api.routers.api.catalog import _suite_panel_meta
    from olmo_eval_api.services.read_suites import primary_gap_metric_keys

    node = build_tree(
        "pair", {"pair": ("gap", [{"type": "task", "name": "in"}, {"type": "task", "name": "out"}])}
    )
    early = {
        name: {
            "score": 0.3,
            "num_instances": 10,
            "primary_metric": metric,
            "metrics": {"strict": 0.3},
            "display_format": "percent",
            "higher_is_better": True,
        }
        for name, metric in [("in", "flex"), ("out", "strict")]
    }
    anchor = {
        name: {
            "score": 0.5,
            "num_instances": 10,
            "primary_metric": "strict",
            "metrics": {"strict": 0.5},
            "display_format": "percent",
            "higher_is_better": True,
        }
        for name in ["in", "out"]
    }
    keys = primary_gap_metric_keys(node, [early, anchor])
    assert keys == {"in": "strict", "out": "strict"}
    meta = _suite_panel_meta(node, [early, anchor], keys)
    assert meta.display_format == "percent"
    assert meta.higher_is_better is True


@pytest.mark.parametrize("secondary", [False, True])
def test_generated_gap_task_panel_uses_its_parent_anchor(secondary):
    from olmo_eval_api.routers.api.catalog import _panel_score

    tr = {
        "score": 0.9,
        "stderr": 0.01,
        "num_instances": 10,
        "primary_metric": "flex",
        "metrics": {"flex": 0.9, **({"strict": 0.4} if secondary else {})},
    }
    result = _panel_score(None, "in", {"in": tr}, "primary", {"in": "strict"})
    assert result[0] == (0.4 if secondary else None)
