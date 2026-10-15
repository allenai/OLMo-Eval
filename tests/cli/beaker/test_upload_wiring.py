"""Beaker launch carries dashboard upload settings and the uploader key into jobs."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from olmo_eval.cli.beaker.config_loader import LaunchConfig
from olmo_eval.cli.beaker.experiment_plan import ExperimentPlan
from olmo_eval.cli.beaker.job_assembler import (
    JobConfigAssembler,
    assemble_external_eval_job,
    upload_job_settings,
)


def _assemble(**config_overrides):
    launch_config = LaunchConfig(
        name="test",
        model_specs=["test-model"],
        task_specs=["arc_easy"],
        cluster="ai2/jupiter",
        workspace="ai2/test",
        budget="ai2/oe-other",
        **config_overrides,
    )
    exp = ExperimentPlan(
        name="test-exp",
        model_spec="test-model",
        priority="high",
        tasks=["arc_easy"],
        original_task_specs=["arc_easy"],
        total_expanded_tasks=1,
        num_gpus=1,
    )
    assembler = JobConfigAssembler(
        config=launch_config,
        effective_image="beaker://user/image",
        effective_groups=["group-a"],
        beaker_username="test-user",
        common_secrets=[],
        task_secrets=[],
        inject_aws_credentials=False,
        inject_gcs_credentials=True,
    )
    with patch("olmo_eval.cli.beaker.job_assembler.cluster_has_weka", return_value=False):
        return launch_config, assembler.assemble(exp)


def test_job_command_and_env_carry_upload_settings() -> None:
    launch_config, job = _assemble(tags=["sweep-1"], api_url="https://dev-ingest.example")

    assert "--upload" in job.command
    assert job.command[job.command.index("--tag") + 1] == "sweep-1"
    assert not any(arg.startswith(("--store", "--s3")) for arg in job.command)
    env = job.env_vars
    assert env["OLMO_EVAL_LAUNCH_ID"] == launch_config.launch_id
    assert env["OLMO_EVAL_API_URL"] == "https://dev-ingest.example"
    assert env["OLMO_EVAL_BEAKER_CLUSTER"] == "ai2/jupiter"
    assert env["OLMO_EVAL_BEAKER_PRIORITY"] == "high"
    assert env["OLMO_EVAL_BEAKER_IMAGE"] == "beaker://user/image"
    assert env["OLMO_EVAL_BEAKER_BUDGET"] == "ai2/oe-other"
    assert not {"PGHOST", "PGPORT", "PGDATABASE", "PGUSER", "OLMO_S3_BUCKET"} & set(env)
    assert not any(s.env_var in {"DB_SECRET_ARN", "PGHOST"} for s in job.env_secrets)
    assert job.inject_gcs_credentials is True
    assert job.inject_upload_credentials is True
    assert "storage" not in job.extras and "postgres" not in job.extras


def test_no_upload_is_forwarded_and_api_url_is_optional() -> None:
    _, job = _assemble(upload=False)

    assert "--no-upload" in job.command
    assert "--upload" not in job.command
    assert "OLMO_EVAL_API_URL" not in job.env_vars


def test_launch_ids_differ_between_launches() -> None:
    first, _ = _assemble()
    second, _ = _assemble()
    assert first.launch_id != second.launch_id
    assert len(first.launch_id) == 12


def test_upload_job_settings_resolves_cluster_list() -> None:
    args, env = upload_job_settings(
        upload=True, launch_id="abc", clusters=["ai2/jupiter", "ai2/ceres"]
    )
    assert args == ["--upload"]
    assert env == {
        "OLMO_EVAL_LAUNCH_ID": "abc",
        "OLMO_EVAL_BEAKER_CLUSTER": "ai2/jupiter,ai2/ceres",
    }


def test_external_eval_job_carries_upload_settings() -> None:
    job = assemble_external_eval_job(
        name="test",
        model="test-model",
        external_evals=["tau2_bench"],
        cluster="ai2/jupiter",
        num_gpus=1,
        workspace="ai2/test",
        beaker_image="test-image",
        tags=["ext"],
        launch_id="launch123456",
    )

    assert "--upload" in job.command
    assert job.command[job.command.index("--tag") : job.command.index("--tag") + 2] == [
        "--tag",
        "ext",
    ]
    assert job.env_vars["OLMO_EVAL_LAUNCH_ID"] == "launch123456"
    assert not any(arg.startswith(("--store", "--s3")) for arg in job.command)


def test_upload_does_not_inject_personal_google_credentials() -> None:
    from olmo_eval.cli.beaker.credentials import CredentialManager

    launcher = SimpleNamespace(beaker=SimpleNamespace(user_name="test-user"))
    with patch("olmo_eval.launch.beaker.gcs.get_local_gcs_credentials", return_value=None):
        manager = CredentialManager(["Qwen/Qwen2-0.5B"], None, None)
        assert manager.detect_and_setup(launcher) == (False, False)

        manager = CredentialManager(["gs://bucket/model"], None, None)
        assert manager.detect_and_setup(launcher) == (False, True)


def test_no_upload_skips_the_uploader_key() -> None:
    _, job = _assemble(upload=False)
    assert job.inject_upload_credentials is False


def test_launch_stops_without_google_credentials(capsys) -> None:
    from olmo_eval.cli.beaker.launch import _require_upload_credentials

    with patch("olmo_eval.upload.auth.has_local_google_credentials", return_value=False):
        with pytest.raises(SystemExit):
            _require_upload_credentials(upload=True, dry_run=False)
        _require_upload_credentials(upload=True, dry_run=True)  # warns only
        _require_upload_credentials(upload=False, dry_run=False)


def test_launch_cli_resolves_upload_from_env(monkeypatch) -> None:
    """A local OLMO_EVAL_UPLOAD=0 reaches the job as --no-upload."""
    from click.testing import CliRunner

    from olmo_eval.cli.beaker.launch import launch

    captured = {}

    def fake_external(**kwargs):
        captured.update(kwargs)

    monkeypatch.setenv("OLMO_EVAL_UPLOAD", "0")
    monkeypatch.setenv("OLMO_EVAL_API_URL", "https://dev-ingest.example")
    with patch("olmo_eval.cli.beaker.launch._launch_external_evals", side_effect=fake_external):
        result = CliRunner().invoke(
            launch, ["-m", "test-model", "-E", "tau2_bench", "-c", "h100", "--tag", "x"]
        )

    assert result.exit_code == 0, result.output
    assert captured["upload"] is False
    assert captured["api_url"] == "https://dev-ingest.example"
    assert captured["tags"] == ["x"]
