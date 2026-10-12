"""Derived identity and name helpers (spec 2.6), settings defaults, and suite math."""

from __future__ import annotations

import math

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


def test_quantile_box_uses_mean_when_sent() -> None:
    from olmo_eval_api.routers.api.run_tabs import _quantile_box

    q = {"p5": 1.0, "p25": 2.0, "p50": 3.0, "p75": 4.0, "p95": 5.0, "n": 10}
    box = _quantile_box(q)
    assert box is not None and box.mean == 3.0
    box = _quantile_box({**q, "mean": 3.5})
    assert box is not None and box.mean == 3.5
