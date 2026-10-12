"""Run manifests written by the runners, metadata resolution, and inference payloads."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from olmo_eval.common.metrics.base import AccuracyMetric
from olmo_eval.harness.config import HarnessConfig
from olmo_eval.inference.providers.config import ProviderConfig
from olmo_eval.runners.asynq.runner import AsyncEvalRunner
from olmo_eval.upload.inference import build_inference_payload, downsample
from olmo_eval.upload.manifest import (
    RUN_SECRET_NAME,
    ManifestError,
    degraded_manifest,
    read_manifest,
    run_secret,
    run_status,
    to_utc_iso,
)
from olmo_eval.upload.metadata import metric_meta_from_metrics, suite_results
from olmo_eval.upload.uploader import dry_run
from olmo_eval.upload.validation import PayloadValidator
from tests.upload.fixtures import (
    ARC_CHALLENGE,
    ARC_EASY,
    ARC_SUITE,
    CONTRACT_SCHEMA,
    MODEL,
    MODEL_DIR,
    write_inference,
    write_jsonl,
)

VALIDATOR = PayloadValidator(CONTRACT_SCHEMA)


def test_run_status() -> None:
    assert run_status({"a": None, "b": None}) == "complete"
    assert run_status({"a": None, "b": "boom"}) == "partial"
    assert run_status({"a": "boom"}) == "failed"
    assert run_status({}) == "failed"
    assert run_status({"a": None}, [{"task": None, "error": "x"}]) == "partial"


def test_to_utc_iso() -> None:
    assert to_utc_iso("2026-10-01T10:00:00") == "2026-10-01T10:00:00+00:00"
    assert to_utc_iso("2026-10-01T03:00:00-07:00") == "2026-10-01T10:00:00+00:00"
    assert to_utc_iso("garbage") is None


def test_metric_meta_inherits_and_forces_direction() -> None:
    accuracy = AccuracyMetric(name="accuracy", scorer=object)  # type: ignore[arg-type]
    meta = metric_meta_from_metrics(
        [accuracy],
        ["accuracy", "domain__Law__accuracy", "judge_parsing_errors", "pass_at_1", "f1_score"],
    )
    assert meta["accuracy"] == {
        "higher_is_better": True,
        "display_format": "percent",
        "unit": "proportion",
    }
    assert meta["domain__Law__accuracy"] == meta["accuracy"]
    assert meta["judge_parsing_errors"]["higher_is_better"] is False
    assert meta["pass_at_1"] == {
        "higher_is_better": None,
        "display_format": "percent",
        "unit": None,
    }
    assert meta["f1_score"]["display_format"] == "raw"


def test_suite_results_use_registered_definitions() -> None:
    suites = suite_results(
        {
            f"{ARC_SUITE}@high": {
                "metrics": {"primary_score": {"average": 0.6}, "acc": {"x": float("nan")}},
                "tasks": [f"{ARC_CHALLENGE}@high", f"{ARC_EASY}@high"],
                "num_tasks": 2,
                "aggregation": "average",
                "primary_metric": "primary_score:average",
            },
            "not_a_registered_suite": {
                "metrics": {},
                "tasks": ["gsm8k@high"],
                "num_tasks": 1,
                "aggregation": "average",
                "parent_suite": f"{ARC_SUITE}@high",
            },
        }
    )
    arc, other = suites
    assert arc["name"] == ARC_SUITE
    assert arc["score"] == 0.6
    assert arc["metrics"]["acc"] == {"x": None}
    assert {c["name"] for c in arc["children"]} == {ARC_CHALLENGE, ARC_EASY}
    assert other["children"] == [{"type": "task", "name": "gsm8k"}]
    assert other["parent"] == ARC_SUITE
    assert other["score"] is None


def test_suite_results_add_nested_suites_the_runner_does_not_report() -> None:
    from olmo_eval.evals.suites import get_suite
    from olmo_eval.runners.processing.aggregation import compute_suite_aggregations

    tasks = {
        spec: {"metrics": {"acc": {"x": 0.5}}, "primary_metric": "acc:x", "num_instances": 10}
        for spec in get_suite("biology").expand()
    }
    aggregations = compute_suite_aggregations(["biology"], tasks)
    assert list(aggregations) == ["biology"]

    by_name = {s["name"]: s for s in suite_results(aggregations, tasks)}

    assert set(by_name) == {"biology", "lab_bench", "geneturing"}
    lab = by_name["lab_bench"]
    assert lab["parent"] == "biology"
    assert lab["score"] == 0.5
    assert lab["children"] and all(c["type"] == "task" for c in lab["children"])
    unscored = {s["name"]: s for s in suite_results(aggregations)}["geneturing"]
    assert unscored["score"] is None and unscored["children"]


def test_degraded_manifest_for_external_runner_metrics() -> None:
    metrics = {
        "timestamp": "2026-10-01T10:00:00+00:00",
        "config": {"kind": "vllm_server", "model": MODEL, "alias": "olmo-1b"},
        "tasks": [{"task": "tau2_bench", "metrics": {"pass^1": {"external": 0.4}}}],
        "summary": {},
        "errors": [],
        "experiment_id": "abc123def456",
        "experiment_duration_seconds": 60.0,
    }
    manifest = degraded_manifest("/tmp/x", metrics)
    assert manifest["model"]["name"] == "olmo-1b"
    assert manifest["model"]["path"] == MODEL
    assert manifest["run"]["harness_config"] is None
    assert manifest["run"]["started_at"] == "2026-10-01T09:59:00+00:00"
    assert manifest["tasks"]["tau2_bench"]["base_task"] is None


def test_degraded_manifest_needs_an_experiment_id() -> None:
    with pytest.raises(ManifestError, match="experiment_id"):
        degraded_manifest("/tmp/x", {"tasks": []})


def _runner(tmp_path: Path) -> AsyncEvalRunner:
    from olmo_eval.upload import UploadConfig

    return AsyncEvalRunner(
        harness_config=HarnessConfig(
            name="default", provider=ProviderConfig(kind="vllm_server", model=MODEL)
        ),
        task_specs=[ARC_SUITE],
        output_dir=str(tmp_path),
        experiment_name="exp",
        experiment_group="grp",
        upload_config=UploadConfig(enabled=False, tags=("t1",)),
    )


def test_runner_writes_start_and_final_manifests(tmp_path: Path) -> None:
    from olmo_eval.evals.tasks.common import get_task

    runner = _runner(tmp_path)
    runner._start_run_record("abc123def456")

    start = read_manifest(tmp_path)
    assert start is not None
    assert start["run"]["status"] == "running"
    assert start["run"]["started_at"]
    assert start["run"]["tags"] == ["t1"]
    assert start["run"]["task_specs"] == [ARC_SUITE]
    assert start["model"]["provider_kind"] == "vllm_server"
    assert start["model"]["model_hash"]

    runner._prepared_tasks = {ARC_EASY: get_task(ARC_EASY)}
    results_dict = {
        "model": MODEL,
        "model_path": MODEL,
        "provider": "vllm_server",
        "model_config": {"kind": "vllm_server", "model": MODEL},
        "harness_config": {"name": "default"},
        "tasks": {
            ARC_EASY: {"metrics": {"accuracy": {"LogprobScorer": 0.5}}},
            ARC_CHALLENGE: {"error": "dataset failed to load"},
        },
        "errors": [{"task": ARC_CHALLENGE, "error": "dataset failed to load"}],
        "suites": {
            ARC_SUITE: {
                "metrics": {"primary_score": {"average": 0.5}},
                "tasks": [ARC_EASY],
                "num_tasks": 1,
                "aggregation": "average",
                "primary_metric": "primary_score:average",
            }
        },
    }
    runner._write_final_manifest(results_dict, "abc123def456", "0123456789abcdef", 42.0, None)

    final = read_manifest(tmp_path)
    assert final is not None
    assert final["run"]["status"] == "partial"
    assert final["run"]["started_at"] == start["run"]["started_at"]
    assert final["run"]["duration_seconds"] == 42.0
    assert final["run"]["errors"] == [{"task": ARC_CHALLENGE, "error": "dataset failed to load"}]
    easy = final["tasks"][ARC_EASY]
    assert easy["base_task"] == "arc_easy"
    assert ARC_SUITE in easy["suites"]
    assert easy["metric_meta"]["accuracy"]["higher_is_better"] is True
    assert final["tasks"][ARC_CHALLENGE]["error"] == "dataset failed to load"
    assert final["suites"][0]["name"] == ARC_SUITE

    request = {
        "protocol_version": 1,
        "client": {"name": "olmo-eval", "version": "test"},
        "run": final["run"],
        "model": final["model"],
    }
    assert VALIDATOR.errors("RunUpsertRequest", request) == []
    complete = {
        "status": "partial",
        "finished_at": final["run"]["finished_at"],
        "duration_seconds": 42.0,
        "suites": final["suites"],
        "expected_task_results": 2,
        "expected_artifacts": 0,
    }
    assert VALIDATOR.errors("CompleteRequest", complete) == []


def test_runner_manifest_failure_does_not_raise(tmp_path: Path) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("x")
    runner = _runner(blocker / "sub")  # cannot create a directory under a file
    runner._start_run_record("abc123def456")
    runner._write_final_manifest({"tasks": {}}, "abc123def456", None, None, None)


def test_external_runner_writes_predictions_and_manifest(tmp_path: Path) -> None:
    from olmo_eval.evals.external import ExternalEvalResult
    from olmo_eval.runners.external.runner import ExternalEvalRunner

    runner = ExternalEvalRunner(
        provider_config=ProviderConfig(kind="vllm_server", model=MODEL),
        external_eval_names=["tau2_bench", "scicode"],
        output_dir=str(tmp_path),
    )
    results = {
        "tau2_bench": ExternalEvalResult(
            name="tau2_bench",
            metrics={"pass^1": 0.5},
            predictions=[{"native_id": "t1", "instance_metrics": {"pass^1": {"external": 1.0}}}],
        ),
        "scicode": ExternalEvalResult.from_error("scicode", "container failed"),
    }

    runner._save_results(results, total_duration=30.0)

    preds = list((tmp_path / "predictions").rglob("tau2_bench_*-predictions.jsonl"))
    assert len(preds) == 1
    manifest = read_manifest(tmp_path)
    assert manifest is not None
    assert manifest["run"]["status"] == "partial"
    assert manifest["run"]["task_specs"] == ["tau2_bench", "scicode"]
    assert manifest["tasks"]["scicode"]["error"] == "container failed"

    report = dry_run(tmp_path)
    assert report.errors == []
    assert report.instances == 1


def test_inference_payload(tmp_path: Path) -> None:
    write_inference(tmp_path)
    write_jsonl(
        tmp_path / "metrics" / "vllm_server_metrics.jsonl",
        [
            {
                "timestamp": "2026-10-01T09:59:05+00:00",
                "error": None,
                "metrics": {
                    "vllm:num_requests_running": {"labels": {}, "value": 7.0},
                    "vllm:kv_cache_usage_perc": {"labels": {}, "value": 0.25},
                },
            },
            {"timestamp": "2026-10-01T09:59:06+00:00", "error": "timeout", "metrics": {}},
        ],
    )

    payload = build_inference_payload(tmp_path, "2026-10-01T09:59:00+00:00")

    assert payload is not None
    assert VALIDATOR.errors("InferenceUploadRequest", payload) == []
    assert [b["seq"] for b in payload["batches"]] == [0, 1, 2]
    assert payload["gpu_devices"] == [
        {"device_id": 0, "name": "NVIDIA H100 80GB HBM3", "memory_total_mb": 81559.0}
    ]
    series = {(s["name"], s["device"]): s for s in payload["series"]}
    util = series[("gpu_utilization_pct", "0")]
    assert util["unit"] == "%"
    assert util["points"][0] == [0.0, 40.0]
    assert series[("vllm_kv_cache_usage_pct", None)]["points"] == [[5.0, 25.0]]
    assert series[("vllm_num_requests_running", None)]["points"] == [[5.0, 7.0]]
    latency = payload["request_latency"]
    assert latency["end_to_end_s"]["n"] == 12
    e2e = latency["end_to_end_s"]
    assert e2e["p5"] <= e2e["mean"] <= e2e["p95"]
    assert latency["tpot_s"] is None
    assert payload["source_paths"][-1] == "metrics/vllm_server_metrics.jsonl"


def test_inference_payload_leaves_out_earlier_runs_and_other_models(tmp_path: Path) -> None:
    write_inference(tmp_path)
    other = tmp_path / "metrics" / "vllm_server_other-model-inference.jsonl"
    other.write_bytes(
        (tmp_path / "metrics" / f"vllm_server_{MODEL_DIR}-inference.jsonl").read_bytes()
    )

    # Batches stamped 09:59:00-02 predate a run that started at 10:30.
    assert build_inference_payload(tmp_path, "2026-10-01T10:30:00+00:00", [MODEL]) is None

    payload = build_inference_payload(tmp_path, "2026-10-01T09:59:01+00:00", [MODEL])
    assert payload is not None
    assert payload["source_paths"] == [f"metrics/vllm_server_{MODEL_DIR}-inference.jsonl"]
    assert len(payload["batches"]) == 3  # within the one-minute clock slack


def test_no_inference_files(tmp_path: Path) -> None:
    assert build_inference_payload(tmp_path) is None


def test_downsample_caps_points() -> None:
    points = [(float(i), float(i % 10)) for i in range(5000)]
    reduced = downsample(points, max_points=500)
    assert len(reduced) == 500
    assert reduced[0][0] < reduced[-1][0]
    assert all(0 <= y <= 9 for _, y in reduced)


def test_contract_examples_match_the_client_shapes() -> None:
    """Keys the client sends match the contract examples for each definition."""
    examples = CONTRACT_SCHEMA.parent / "examples" / "ingest"
    for name, definition in [
        ("run-upsert-final.request.json", "RunUpsertRequest"),
        ("task-result.request.json", "TaskResultIn"),
        ("instances.request.json", "InstanceBatchRequest"),
        ("complete-with-suites.request.json", "CompleteRequest"),
    ]:
        example = json.loads((examples / name).read_text())
        assert VALIDATOR.errors(definition, example) == [], name


def test_metric_meta_for_live_task() -> None:
    from olmo_eval.evals.tasks.common import get_task
    from olmo_eval.upload.metadata import metric_meta_for_task

    meta = metric_meta_for_task(get_task(ARC_EASY), ["accuracy"])
    assert meta["accuracy"]["display_format"] == "percent"
    assert metric_meta_for_task(SimpleNamespace(), ["x"]) == {}


def test_finalize_writes_metrics_manifest_then_uploads(tmp_path: Path) -> None:
    from unittest.mock import patch

    from olmo_eval.upload import UploadConfig

    runner = _runner(tmp_path)
    runner.upload_config = UploadConfig(enabled=True, api_url="https://ingest.test")
    runner._start_run_record = lambda experiment_id: None  # type: ignore[method-assign]
    results_dict = {
        "model": MODEL,
        "model_path": MODEL,
        "provider": "vllm_server",
        "model_config": {"kind": "vllm_server", "model": MODEL},
        "tasks": {ARC_EASY: {"metrics": {"accuracy": {"LogprobScorer": 0.5}}, "num_instances": 2}},
        "summary": {},
        "errors": [],
        "timestamp": "2026-10-01T10:00:00",
    }

    def check_files(output_dir, config):
        assert (Path(output_dir) / "metrics.json").exists()
        assert read_manifest(output_dir)["run"]["status"] == "complete"
        assert config is runner.upload_config

    with patch("olmo_eval.upload.upload_results_dir", side_effect=check_files) as upload:
        runner._finalize_and_save(results_dict, experiment_id="abc123def456")

    upload.assert_called_once()


def test_finalize_skips_upload_when_disabled(tmp_path: Path) -> None:
    from unittest.mock import patch

    runner = _runner(tmp_path)
    results_dict = {"model": MODEL, "model_config": {}, "tasks": {}, "errors": []}

    with patch("olmo_eval.upload.upload_results_dir") as upload:
        runner._finalize_and_save(results_dict, experiment_id="abc123def456")

    upload.assert_not_called()
    assert read_manifest(tmp_path) is not None


def test_run_secret_is_created_once(tmp_path: Path) -> None:
    first = run_secret(tmp_path)
    assert first and len(first) >= 32
    assert run_secret(tmp_path) == first
    assert (tmp_path / RUN_SECRET_NAME).stat().st_mode & 0o777 == 0o600


def test_run_secret_in_a_read_only_directory(tmp_path: Path) -> None:
    tmp_path.chmod(0o500)
    try:
        assert run_secret(tmp_path) is None
    finally:
        tmp_path.chmod(0o700)


def test_inference_truncation_is_reported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from olmo_eval.upload import inference

    monkeypatch.setattr(inference, "MAX_BATCHES", 2)
    write_inference(tmp_path, batches=3)
    warnings: list[str] = []

    payload = build_inference_payload(tmp_path, "2026-10-01T09:59:00+00:00", warnings=warnings)

    assert payload is not None and len(payload["batches"]) == 2
    assert len(warnings) == 1 and "left out 1" in warnings[0]
    assert payload["request_latency"]["end_to_end_s"]["n"] == 12


def test_name_based_display_format_matches_metric_objects() -> None:
    from olmo_eval.common.metrics.base import display_format_for_name

    assert display_format_for_name("accuracy") == "percentage"
    assert display_format_for_name("tool_accuracy") == "percentage"
    assert display_format_for_name("pass_at_1") == "percentage"
    assert display_format_for_name("bits_per_byte") == "raw"
    meta = metric_meta_from_metrics([], ["domain__Law__accuracy", "domain__Law__bleu"])
    assert meta["domain__Law__accuracy"]["display_format"] == "percent"
    assert meta["domain__Law__bleu"]["display_format"] == "raw"


def test_secret_looking_config_values_are_redacted() -> None:
    from olmo_eval.upload.manifest import REDACTED, json_safe

    config = {
        "provider": {
            "kind": "litellm",
            "api_key": "sk-live-123",
            "kwargs": {"Authorization": "Bearer abc", "hf-token": "hf_xyz", "max_tokens": 256},
            "required_secrets": ["OPENAI_API_KEY"],
        },
        "eos_token": "</s>",
        "token": "",
        "password": None,
    }
    cleaned = json_safe(config)
    assert cleaned["provider"]["api_key"] == REDACTED
    assert cleaned["provider"]["kwargs"] == {
        "Authorization": REDACTED,
        "hf-token": REDACTED,
        "max_tokens": 256,
    }
    # Secret names are references, not values.
    assert cleaned["provider"]["required_secrets"] == ["OPENAI_API_KEY"]
    assert cleaned["eos_token"] == "</s>"
    assert cleaned["token"] == "" and cleaned["password"] is None


def test_secret_looking_cli_values_are_redacted() -> None:
    from olmo_eval.upload.manifest import REDACTED, redact_argv

    argv = [
        "olmo-eval", "run", "-m", "gpt", "-o", "provider.api_key=sk-1", "--hf-token", "hf_2",
        "--auth-token=t3", "-o", "limit=10", "--tag", "x",
    ]  # fmt: skip
    assert redact_argv(argv) == [
        "olmo-eval", "run", "-m", "gpt", "-o", f"provider.api_key={REDACTED}", "--hf-token",
        REDACTED, f"--auth-token={REDACTED}", "-o", "limit=10", "--tag", "x",
    ]  # fmt: skip
    # A boolean flag keeps the option after it.
    assert redact_argv(["--no-hf-token", "-m", "gpt"]) == ["--no-hf-token", "-m", "gpt"]


def test_run_info_redacts_argv() -> None:
    from olmo_eval.upload.manifest import build_run_info

    run = build_run_info(
        run_id="abc123def456",
        status="running",
        started_at=None,
        argv=["olmo-eval", "run", "-o", "provider.api_key=sk-1"],
        harness_config={"provider": {"api_key": "sk-1"}},
        capture_provenance=False,
    )
    assert "sk-1" not in json.dumps(run)
