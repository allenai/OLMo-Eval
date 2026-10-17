"""metrics.json reaches the Beaker result dataset when -O points elsewhere."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from olmo_eval.runners.processing.metrics import (
    RESULT_DIR_METRICS_SUBDIR,
    write_metrics_json,
)


@pytest.fixture
def beaker_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """A fake Beaker job whose result dataset is mounted at tmp_path/results."""
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    monkeypatch.setenv("BEAKER_JOB_ID", "01TESTJOB")
    monkeypatch.setenv("OLMO_RESULT_DIR", str(result_dir))
    monkeypatch.delenv("RESULTS_DIR", raising=False)
    return result_dir


def _write(output_dir: Path, experiment_id: str | None = "exp1") -> None:
    write_metrics_json(str(output_dir), {"tasks": {}}, experiment_id=experiment_id)


def _copies(result_dir: Path) -> list[Path]:
    return sorted((result_dir / RESULT_DIR_METRICS_SUBDIR).glob("*/metrics.json"))


def test_output_outside_result_dir_is_copied(beaker_env: Path, tmp_path: Path) -> None:
    output_dir = tmp_path / "weka" / "run"
    _write(output_dir)

    copy = beaker_env / RESULT_DIR_METRICS_SUBDIR / "exp1" / "metrics.json"
    assert copy.read_text() == (output_dir / "metrics.json").read_text()
    assert json.loads(copy.read_text())["experiment_id"] == "exp1"
    assert not (beaker_env / "metrics.json").exists()


def test_only_metrics_json_is_copied(beaker_env: Path, tmp_path: Path) -> None:
    output_dir = tmp_path / "weka" / "run"
    (output_dir / "predictions").mkdir(parents=True)
    (output_dir / "predictions" / "task_predictions.jsonl").write_text("{}\n")
    _write(output_dir)

    copied = [p.relative_to(beaker_env).as_posix() for p in beaker_env.rglob("*") if p.is_file()]
    assert copied == [f"{RESULT_DIR_METRICS_SUBDIR}/exp1/metrics.json"]


def test_runs_sharing_a_job_keep_separate_copies(beaker_env: Path, tmp_path: Path) -> None:
    _write(tmp_path / "a", experiment_id="exp-a")
    _write(tmp_path / "b", experiment_id="exp-b")
    _write(tmp_path / "c", experiment_id=None)

    copies = _copies(beaker_env)
    assert len(copies) == 3
    ids = {json.loads(p.read_text()).get("experiment_id") for p in copies}
    assert ids == {"exp-a", "exp-b", None}


def test_existing_copy_is_not_overwritten(beaker_env: Path, tmp_path: Path) -> None:
    existing = beaker_env / RESULT_DIR_METRICS_SUBDIR / "exp1" / "metrics.json"
    existing.parent.mkdir(parents=True)
    existing.write_text("original")

    _write(tmp_path / "out")

    assert existing.read_text() == "original"


@pytest.mark.parametrize("subdir", ["", "planner"])
def test_output_inside_result_dir_is_not_copied(beaker_env: Path, subdir: str) -> None:
    output_dir = beaker_env / subdir if subdir else beaker_env
    _write(output_dir)

    assert (output_dir / "metrics.json").is_file()
    assert not (beaker_env / RESULT_DIR_METRICS_SUBDIR).exists()


def test_off_beaker_nothing_is_copied(
    beaker_env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("BEAKER_JOB_ID")
    _write(tmp_path / "out")

    assert not (beaker_env / RESULT_DIR_METRICS_SUBDIR).exists()


def test_gantry_results_dir_is_used_without_olmo_result_dir(
    beaker_env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OLMO_RESULT_DIR")
    monkeypatch.setenv("RESULTS_DIR", str(beaker_env))
    _write(tmp_path / "out")

    assert len(_copies(beaker_env)) == 1


def test_missing_result_dir_warns(
    beaker_env: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("OLMO_RESULT_DIR", str(tmp_path / "absent"))
    with caplog.at_level(logging.WARNING):
        _write(tmp_path / "out")

    assert (tmp_path / "out" / "metrics.json").is_file()
    assert not (tmp_path / "absent").exists()
    assert "does not exist" in caplog.text


def test_failed_copy_warns_and_keeps_the_run(
    beaker_env: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise PermissionError("read-only result dir")

    monkeypatch.setattr("olmo_eval.runners.processing.metrics.shutil.copyfile", fail)
    with caplog.at_level(logging.WARNING):
        _write(tmp_path / "out")

    assert (tmp_path / "out" / "metrics.json").is_file()
    assert "Could not copy metrics.json" in caplog.text
