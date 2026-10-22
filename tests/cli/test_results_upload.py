"""olmo-eval results upload."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from olmo_eval.cli import main
from olmo_eval.upload import UploadResult
from tests.upload.fixtures import RUN_ID, make_arc_run


def test_dry_run_validates_without_network(tmp_path: Path) -> None:
    out = make_arc_run(tmp_path / "run")

    with patch("olmo_eval.upload.uploader.make_client") as make_client:
        result = CliRunner().invoke(main, ["results", "upload", str(out), "--dry-run"])

    assert result.exit_code == 0, result.output
    make_client.assert_not_called()
    assert RUN_ID in result.output
    assert "Instances: 12" in result.output
    assert "All payloads are valid" in result.output
    assert "contract schema" in result.output


def test_dry_run_reports_unusable_directories(tmp_path: Path) -> None:
    result = CliRunner().invoke(main, ["results", "upload", str(tmp_path), "--dry-run"])
    assert result.exit_code == 1
    assert "neither manifest.json nor metrics.json" in result.output


def test_upload_passes_flags_and_reports_the_dashboard_url(tmp_path: Path) -> None:
    out = make_arc_run(tmp_path / "run")
    ok = UploadResult(ok=True, run_id=RUN_ID, dashboard_url="https://dash.test/runs/x")

    with patch("olmo_eval.upload.upload_results_dir", return_value=ok) as upload:
        result = CliRunner().invoke(
            main,
            ["results", "upload", str(out), "--api-url", "http://localhost:8000", "--tag", "a"],
        )

    assert result.exit_code == 0, result.output
    config = upload.call_args.args[1]
    assert config.enabled is True
    assert config.api_url == "http://localhost:8000"
    assert config.tags == ("a",)
    assert "https://dash.test/runs/x" in result.output


def test_upload_failure_exits_nonzero(tmp_path: Path) -> None:
    out = make_arc_run(tmp_path / "run")
    failed = UploadResult(ok=False, error="HTTP 503")

    with patch("olmo_eval.upload.upload_results_dir", return_value=failed):
        result = CliRunner().invoke(main, ["results", "upload", str(out)])

    assert result.exit_code == 1
    assert "HTTP 503" in result.output


def test_invalid_tag_is_rejected(tmp_path: Path) -> None:
    result = CliRunner().invoke(main, ["results", "upload", str(tmp_path), "--tag", "bad tag"])
    assert result.exit_code == 2
    assert "Invalid tag" in result.output


def test_removed_commands_are_gone() -> None:
    runner = CliRunner()
    assert runner.invoke(main, ["metrics", "--help"]).exit_code != 0
    assert runner.invoke(main, ["results", "query", "--help"]).exit_code != 0
    run_help = runner.invoke(main, ["run", "--help"]).output
    assert "--upload / --no-upload" in run_help
    assert "--store" not in run_help
    assert "--s3-bucket" not in run_help


def test_run_crash_records_a_failed_run(tmp_path: Path, monkeypatch) -> None:
    from types import SimpleNamespace

    class CrashingRunner(SimpleNamespace):
        def validate(self) -> None:
            pass

        def run(self) -> None:
            raise RuntimeError("worker died")

    runner = CrashingRunner(output_dir=str(tmp_path))
    argv = ["run", "-m", "mock", "-t", "arc_easy", "-O", str(tmp_path), "--no-upload"]
    monkeypatch.setattr("sys.argv", ["olmo-eval", *argv])
    with (
        patch("olmo_eval.cli.run.factory.RunnerFactory.create", return_value=runner),
        patch("olmo_eval.upload.mark_run_failed") as mark_failed,
    ):
        result = CliRunner().invoke(main, argv)

    assert result.exit_code == 1
    mark_failed.assert_called_once()
    assert mark_failed.call_args.args[0] == str(tmp_path)
    assert mark_failed.call_args.args[1].enabled is False
    assert mark_failed.call_args.kwargs["error"] == "worker died"
    assert mark_failed.call_args.kwargs["since"] is not None


@pytest.mark.parametrize(
    ("crash", "exit_code", "error"),
    [
        ("interrupt", 130, "Run interrupted"),
        ("sigterm", 143, "Run terminated (SIGTERM)"),
    ],
)
def test_run_killed_by_a_signal_records_a_failed_run(
    tmp_path: Path, monkeypatch, crash: str, exit_code: int, error: str
) -> None:
    import os
    import signal
    from types import SimpleNamespace

    class KilledRunner(SimpleNamespace):
        def validate(self) -> None:
            pass

        def run(self) -> None:
            if crash == "interrupt":
                raise KeyboardInterrupt
            os.kill(os.getpid(), signal.SIGTERM)

    runner = KilledRunner(output_dir=str(tmp_path))
    argv = ["run", "-m", "mock", "-t", "arc_easy", "-O", str(tmp_path), "--no-upload"]
    monkeypatch.setattr("sys.argv", ["olmo-eval", *argv])
    previous = signal.getsignal(signal.SIGTERM)
    with (
        patch("olmo_eval.cli.run.factory.RunnerFactory.create", return_value=runner),
        patch("olmo_eval.upload.mark_run_failed") as mark_failed,
    ):
        result = CliRunner().invoke(main, argv)

    assert result.exit_code == exit_code
    assert mark_failed.call_args.kwargs["error"] == error
    assert signal.getsignal(signal.SIGTERM) == previous
