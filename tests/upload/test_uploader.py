"""The upload protocol against a fake ingest service."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from olmo_eval.upload import (
    UploadConfig,
    mark_run_failed,
    register_run_start,
    upload_results_dir,
)
from olmo_eval.upload.auth import NO_CREDENTIALS_MESSAGE, UploadAuthError
from olmo_eval.upload.client import IngestClient
from olmo_eval.upload.manifest import RUN_SECRET_NAME
from olmo_eval.upload.uploader import build_plan, dry_run
from tests.upload.fixtures import (
    API_URL,
    ARC_CHALLENGE,
    ARC_EASY,
    ARC_SUITE,
    RUN_ID,
    FakeIngest,
    StaticTokens,
    make_arc_run,
    write_manifest,
)

CONFIG = UploadConfig(enabled=True, api_url=API_URL, tags=("cli-tag",))


@pytest.fixture
def server() -> FakeIngest:
    return FakeIngest()


@pytest.fixture
def sleeps() -> list[float]:
    return []


@pytest.fixture
def client(server: FakeIngest, sleeps: list[float]) -> IngestClient:
    return IngestClient(API_URL, StaticTokens(), transport=server.transport(), sleep=sleeps.append)


def test_happy_path_uploads_everything(tmp_path: Path, server: FakeIngest, client) -> None:
    out = make_arc_run(tmp_path / "run")

    result = upload_results_dir(out, CONFIG, client=client)

    assert result.ok, result.error
    assert result.run_id == RUN_ID
    assert result.dashboard_url == f"https://dash.test/runs/{RUN_ID}"
    assert server.schema_errors == []
    assert set(server.tokens) == {"test-token"}

    # Order: run, artifacts, task results with instances, inference, complete.
    paths = [p for _, p in server.calls]
    assert paths[0] == f"/v1/runs/{RUN_ID}"
    assert paths[1] == f"/v1/runs/{RUN_ID}/artifacts:sign"
    assert paths[-2] == f"/v1/runs/{RUN_ID}/inference"
    assert paths[-1] == f"/v1/runs/{RUN_ID}/complete"

    run = server.runs[RUN_ID]["run"]
    assert run["status"] == "complete"
    assert run["tags"] == ["smoke", "cli-tag"]

    by_name = {t["task_name"]: (tid, t) for tid, t in server.task_results.items()}
    easy_id, easy = by_name[ARC_EASY]
    challenge_id, challenge = by_name[ARC_CHALLENGE]
    assert easy["instance_count"] == len(server.instances[easy_id]) == 6
    assert easy["requests_path"] is not None
    assert easy["suites"] == [ARC_SUITE]
    assert easy["metric_meta"]["accuracy"]["higher_is_better"] is True
    assert challenge["requests_path"] is None
    assert len(server.instances[challenge_id]) == 6

    uploaded = set(server.objects)
    assert {"metrics.json", "manifest.json", "logs/vllm_server_8000.log"} <= uploaded
    assert any(p.startswith("predictions/") for p in uploaded)
    assert any(p.startswith("metrics/") for p in uploaded)

    assert server.inference is not None
    assert len(server.inference["batches"]) == 3
    assert server.completed is not None
    assert server.completed["expected_task_results"] == 2
    assert server.completed["expected_artifacts"] == len(uploaded)
    assert server.completed["suites"][0]["name"] == ARC_SUITE


def test_non_finite_task_values_are_uploaded_as_null(
    tmp_path: Path, server: FakeIngest, client
) -> None:
    out = make_arc_run(tmp_path / "run")
    metrics_path = out / "metrics.json"
    data = json.loads(metrics_path.read_text())
    row = data["tasks"][0]
    row["metrics"]["accuracy"] = {"LogprobScorer": float("nan"), "other": float("inf")}
    row["config"]["temperature"] = float("-inf")
    row["config"]["nested"] = [{"value": float("nan")}]
    row["duration_seconds"] = float("nan")
    row["error_summary"] = {"rate": float("nan")}
    metrics_path.write_text(json.dumps(data))

    result = upload_results_dir(out, CONFIG, client=client)

    assert result.ok, result.error
    assert server.schema_errors == []
    task = next(t for t in server.task_results.values() if t["task_name"] == row["task"])
    assert task["metrics"]["accuracy"] == {"LogprobScorer": None, "other": None}
    assert task["config"]["temperature"] is None
    assert task["config"]["nested"] == [{"value": None}]
    assert task["duration_seconds"] is None
    assert task["error_summary"] == {"rate": None}


def test_every_request_carries_the_run_secret(tmp_path: Path, server: FakeIngest, client) -> None:
    out = make_arc_run(tmp_path / "run")

    assert register_run_start(out, CONFIG, client=client)
    result = upload_results_dir(out, CONFIG, client=client)

    assert result.ok, result.error
    secret_file = out / RUN_SECRET_NAME
    assert secret_file.stat().st_mode & 0o777 == 0o600
    assert set(server.run_secrets) == {secret_file.read_text()}
    assert RUN_SECRET_NAME not in server.objects


def test_reupload_skips_existing_artifacts(tmp_path: Path, server: FakeIngest, client) -> None:
    out = make_arc_run(tmp_path / "run")
    assert upload_results_dir(out, CONFIG, client=client).ok
    first_objects = dict(server.objects)
    task_ids = set(server.task_results)

    assert upload_results_dir(out, CONFIG, client=client).ok

    assert server.objects == first_objects
    assert set(server.task_results) == task_ids  # same (task_name, task_hash) keys
    assert server.schema_errors == []


def test_reupload_skips_instances_of_unchanged_task_results(
    tmp_path: Path, server: FakeIngest, client
) -> None:
    out = make_arc_run(tmp_path / "run")
    assert upload_results_dir(out, CONFIG, client=client).ok
    instances = {tid: list(rows) for tid, rows in server.instances.items()}
    server.calls.clear()

    assert upload_results_dir(out, CONFIG, client=client).ok

    assert not any("/instances" in path for _, path in server.calls)
    assert dict(server.instances) == instances


def test_retries_transient_errors(
    tmp_path: Path, server: FakeIngest, client, sleeps: list[float]
) -> None:
    out = make_arc_run(tmp_path / "run")
    server.fail[f"PUT /v1/runs/{RUN_ID}"] = [503, 502]
    server.fail[f"POST /v1/runs/{RUN_ID}/complete"] = [429]

    result = upload_results_dir(out, CONFIG, client=client)

    assert result.ok, result.error
    assert len(sleeps) == 3
    assert 1.0 <= sleeps[0] <= 1.2
    assert 2.0 <= sleeps[1] <= 2.4


def test_gives_up_after_max_retries(tmp_path: Path, server: FakeIngest, sleeps) -> None:
    out = make_arc_run(tmp_path / "run")
    server.fail[f"PUT /v1/runs/{RUN_ID}"] = [503] * 10
    client = IngestClient(
        API_URL, StaticTokens(), transport=server.transport(), sleep=sleeps.append, max_retries=2
    )

    result = upload_results_dir(out, CONFIG, client=client)

    assert not result.ok
    assert "HTTP 503" in (result.error or "")
    assert len(sleeps) == 2


def test_resigns_once_when_gcs_rejects_the_url(tmp_path: Path, server: FakeIngest, client) -> None:
    out = make_arc_run(tmp_path / "run")
    server.gcs_fail["metrics.json"] = [403]

    result = upload_results_dir(out, CONFIG, client=client)

    assert result.ok, result.error
    assert "metrics.json" in server.objects
    assert server.sign_requests == 2


def test_artifact_failure_is_reported(tmp_path: Path, server: FakeIngest, client, caplog) -> None:
    out = make_arc_run(tmp_path / "run")
    server.gcs_fail["metrics.json"] = [403, 403]

    with caplog.at_level(logging.ERROR):
        result = upload_results_dir(out, CONFIG, client=client)

    assert not result.ok
    assert "metrics.json" in (result.error or "")
    # The rest of the protocol still ran, so the run is visible in the dashboard.
    assert server.completed is not None
    assert f"olmo-eval results upload {out}" in caplog.text


def test_client_error_fails_fast_and_never_raises(
    tmp_path: Path, server: FakeIngest, client, sleeps, caplog
) -> None:
    out = make_arc_run(tmp_path / "run")
    server.fail[f"PUT /v1/runs/{RUN_ID}"] = [409]
    server.error_body = {
        "error": {"code": "conflict", "message": "model mismatch", "request_id": "req-9"}
    }

    with caplog.at_level(logging.ERROR):
        result = upload_results_dir(out, CONFIG, client=client)

    assert not result.ok
    assert "model mismatch" in (result.error or "")
    assert "req-9" in (result.error or "")
    assert sleeps == []
    assert f"Results are saved in {out}" in caplog.text
    assert "--api-url https://ingest.test" in caplog.text
    assert "--tag cli-tag" in caplog.text
    # Local results are untouched.
    assert (out / "metrics.json").exists()
    assert (out / "manifest.json").exists()


def test_auth_failure_never_raises(tmp_path: Path, server: FakeIngest, caplog) -> None:
    class NoCredentials:
        def token(self) -> str:
            raise UploadAuthError(NO_CREDENTIALS_MESSAGE)

    out = make_arc_run(tmp_path / "run")
    client = IngestClient(API_URL, NoCredentials(), transport=server.transport())

    with caplog.at_level(logging.ERROR):
        result = upload_results_dir(out, CONFIG, client=client)

    assert not result.ok
    assert result.error == NO_CREDENTIALS_MESSAGE
    assert "olmo-eval results upload" in caplog.text


def test_access_token_never_logged(tmp_path: Path, server: FakeIngest, sleeps, caplog) -> None:
    secret = "ya29.secret-access-token"
    out = make_arc_run(tmp_path / "run")
    server.fail[f"PUT /v1/runs/{RUN_ID}"] = [503]
    server.fail[f"POST /v1/runs/{RUN_ID}/complete"] = [403]
    server.gcs_fail["metrics.json"] = [403, 403]
    client = IngestClient(
        API_URL, StaticTokens(secret), transport=server.transport(), sleep=sleeps.append
    )

    with caplog.at_level(logging.DEBUG):
        result = upload_results_dir(out, CONFIG, client=client)

    assert not result.ok
    assert set(server.tokens) == {secret}
    assert caplog.text
    assert secret not in caplog.text
    assert secret not in (result.error or "")


def test_files_from_earlier_runs_in_a_reused_directory_are_skipped(tmp_path: Path) -> None:
    import os

    out = make_arc_run(tmp_path / "run")
    stray = out / "predictions" / "old-model" / "gsm8k_123456-predictions.jsonl"
    stray.parent.mkdir(parents=True)
    stray.write_text("{}\n")
    referenced = next((out / "predictions").rglob("arc_easy*-predictions.jsonl"))
    long_ago = 1_600_000_000  # 2020, before the fixture run started
    os.utime(stray, (long_ago, long_ago))
    os.utime(referenced, (long_ago, long_ago))

    plan = build_plan(out)

    paths = {a["path"] for a in plan.artifacts}
    assert stray.relative_to(out).as_posix() not in paths
    # An earlier run's predictions for the same model and task are not this run's.
    assert referenced.relative_to(out).as_posix() not in paths
    easy = next(t for t in plan.tasks if t.payload["task_name"] == ARC_EASY)
    assert easy.payload["predictions_path"] is None
    assert easy.payload["instance_count"] == 0
    assert any("last modified before this run started" in w for w in plan.warnings)


def test_reupload_of_the_same_run_keeps_its_files(tmp_path: Path) -> None:
    out = make_arc_run(tmp_path / "run")
    first = build_plan(out)

    again = build_plan(out)

    assert {a["path"] for a in again.artifacts} == {a["path"] for a in first.artifacts}
    assert [t.payload["predictions_path"] for t in again.tasks] == [
        t.payload["predictions_path"] for t in first.tasks
    ]


def test_metrics_from_an_earlier_run_are_ignored(tmp_path: Path) -> None:
    out = make_arc_run(tmp_path / "run")
    metrics_path = out / "metrics.json"
    data = json.loads(metrics_path.read_text())
    data["experiment_id"] = "earlier00001"
    metrics_path.write_text(json.dumps(data))

    plan = build_plan(out)

    assert plan.run_id == RUN_ID
    assert plan.tasks == []
    assert "metrics.json" not in {a["path"] for a in plan.artifacts}
    assert any("belongs to run earlier00001" in w for w in plan.warnings)


def test_missing_directory_never_raises(tmp_path: Path, client) -> None:
    result = upload_results_dir(tmp_path / "nope", CONFIG, client=client)
    assert not result.ok
    assert "not a directory" in (result.error or "")


def test_deadline_stops_the_upload(tmp_path: Path, server: FakeIngest) -> None:
    out = make_arc_run(tmp_path / "run")
    client = IngestClient(API_URL, StaticTokens(), transport=server.transport(), deadline_s=0)

    result = upload_results_dir(out, CONFIG, client=client)

    assert not result.ok
    assert "deadline" in (result.error or "").lower()
    assert server.calls == []


def test_degraded_upload_without_manifest(tmp_path: Path, server: FakeIngest, client) -> None:
    out = make_arc_run(tmp_path / "run", with_manifest=False)

    result = upload_results_dir(out, CONFIG, client=client)

    assert result.ok, result.error
    assert server.schema_errors == []
    run = server.runs[RUN_ID]["run"]
    assert run["finished_at"] == "2026-10-01T10:00:00+00:00"  # naive timestamp read as UTC
    assert run["started_at"] == "2026-10-01T09:58:00+00:00"
    assert run["git"]["commit"] is None  # provenance of the original run is unknown
    model = server.runs[RUN_ID]["model"]
    assert model["path"] == "allenai/OLMo-2-0425-1B"
    assert model["provider_kind"] == "vllm_server"
    # The suite in metrics.json "summary" is rebuilt from the registry.
    assert server.completed is not None
    suites = server.completed["suites"]
    assert [s["name"] for s in suites] == [ARC_SUITE]
    assert suites[0]["score"] == pytest.approx((0.5 + 2 / 6) / 2)
    by_name = {t["task_name"]: t for t in server.task_results.values()}
    assert by_name[ARC_EASY]["base_task"] == "arc_easy"
    assert ARC_SUITE in by_name[ARC_EASY]["suites"]
    assert by_name[ARC_EASY]["metric_meta"]["accuracy"]["display_format"] == "percent"


def test_running_manifest_gets_a_final_status(tmp_path: Path) -> None:
    out = make_arc_run(tmp_path / "run")
    write_manifest(out, status="running")

    plan = build_plan(out)

    assert plan.run_request["run"]["status"] == "complete"
    assert plan.complete["status"] == "complete"
    assert plan.complete["finished_at"] is not None


def test_failed_tasks_make_a_partial_run(tmp_path: Path) -> None:
    out = make_arc_run(tmp_path / "run", with_manifest=False)
    metrics = json.loads((out / "metrics.json").read_text())
    metrics["errors"] = [{"task": ARC_CHALLENGE, "error": "boom"}]
    (out / "metrics.json").write_text(json.dumps(metrics))

    plan = build_plan(out)

    assert plan.complete["status"] == "partial"
    errors = {t.payload["task_name"]: t.payload["error"] for t in plan.tasks}
    assert errors == {ARC_EASY: None, ARC_CHALLENGE: "boom"}


def test_dry_run_validates_against_the_contract(tmp_path: Path) -> None:
    out = make_arc_run(tmp_path / "run")

    report = dry_run(out, ["extra"])

    assert report.errors == []
    assert report.instances == 12
    assert "extra" in report.plan.run_request["run"]["tags"]


def test_register_run_start(tmp_path: Path, server: FakeIngest, client) -> None:
    out = tmp_path / "run"
    out.mkdir()
    write_manifest(out, status="running")

    assert register_run_start(out, CONFIG, client=client)

    assert server.calls == [("GET", "/v1/whoami"), ("PUT", f"/v1/runs/{RUN_ID}")]
    assert server.runs[RUN_ID]["run"]["status"] == "running"
    assert server.schema_errors == []


def test_register_run_start_is_best_effort(tmp_path: Path, server: FakeIngest, caplog) -> None:
    out = tmp_path / "run"
    out.mkdir()
    write_manifest(out, status="running")
    server.fail["GET /v1/whoami"] = [403]
    server.error_body = {"error": {"code": "forbidden", "message": "nope", "request_id": "r"}}
    client = IngestClient(API_URL, StaticTokens(), transport=server.transport())

    with caplog.at_level(logging.WARNING):
        assert not register_run_start(out, CONFIG, client=client)

    assert "Could not register the run" in caplog.text


def test_register_run_start_disabled(tmp_path: Path, server: FakeIngest, client) -> None:
    disabled = UploadConfig(enabled=False, api_url=API_URL)
    assert not register_run_start(tmp_path, disabled, client=client)
    assert server.calls == []


def test_mark_run_failed(tmp_path: Path, server: FakeIngest, client) -> None:
    out = tmp_path / "run"
    out.mkdir()
    write_manifest(out, status="running")

    result = mark_run_failed(out, CONFIG, error="CUDA out of memory", client=client)

    assert result.ok
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["run"]["status"] == "failed"
    assert manifest["run"]["errors"][-1] == {"task": None, "error": "CUDA out of memory"}
    assert manifest["run"]["duration_seconds"] is not None
    assert server.runs[RUN_ID]["run"]["status"] == "failed"
    assert server.completed is not None
    assert server.completed["expected_task_results"] == 0
    assert server.schema_errors == []


def test_mark_run_failed_ignores_a_manifest_from_an_earlier_run(
    tmp_path: Path, server, client
) -> None:
    out = tmp_path / "run"
    out.mkdir()
    write_manifest(out, status="running")
    started = json.loads((out / "manifest.json").read_text())["run"]["started_at"]
    later = datetime.fromisoformat(started) + timedelta(hours=1)

    result = mark_run_failed(out, CONFIG, error="boom", since=later, client=client)

    assert not result.ok
    assert server.calls == []
    assert json.loads((out / "manifest.json").read_text())["run"]["status"] == "running"


def test_mark_run_failed_ignores_finished_runs(tmp_path: Path, server, client) -> None:
    out = tmp_path / "run"
    out.mkdir()
    write_manifest(out, status="complete")

    result = mark_run_failed(out, CONFIG, error="gate", client=client)

    assert not result.ok
    assert server.calls == []
    assert json.loads((out / "manifest.json").read_text())["run"]["status"] == "complete"


def test_mark_run_failed_without_upload_still_updates_manifest(tmp_path: Path, server) -> None:
    out = tmp_path / "run"
    out.mkdir()
    write_manifest(out, status="running")

    mark_run_failed(out, UploadConfig(enabled=False), error="boom")

    assert json.loads((out / "manifest.json").read_text())["run"]["status"] == "failed"
