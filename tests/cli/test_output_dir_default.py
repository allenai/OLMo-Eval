"""Tests for the run command's default output directory."""

import click
import pytest
from click.testing import CliRunner

from olmo_eval.cli.run import options
from olmo_eval.common.constants.infrastructure import LOCAL_RESULT_DIR


@click.command()
@options.output_options
def _show_output_dir(output_dir: str, **_: object) -> None:
    click.echo(output_dir)


def _invoke(*args: str) -> str:
    result = CliRunner().invoke(_show_output_dir, list(args))
    assert result.exit_code == 0, result.output
    return result.output.strip()


@pytest.fixture
def result_dir(monkeypatch, tmp_path):
    path = tmp_path / "results"
    path.mkdir()
    monkeypatch.setattr(options, "BEAKER_RESULT_DIR", str(path))
    return str(path)


def test_beaker_job_defaults_to_the_result_dir(monkeypatch, result_dir) -> None:
    monkeypatch.setenv("BEAKER_JOB_ID", "01JOB")
    assert _invoke() == result_dir


def test_explicit_output_dir_wins_on_beaker(monkeypatch, result_dir, tmp_path) -> None:
    monkeypatch.setenv("BEAKER_JOB_ID", "01JOB")
    assert _invoke("-O", str(tmp_path / "weka")) == str(tmp_path / "weka")


def test_local_run_defaults_to_the_local_result_dir(monkeypatch, result_dir) -> None:
    monkeypatch.delenv("BEAKER_JOB_ID", raising=False)
    assert _invoke() == LOCAL_RESULT_DIR


def test_beaker_job_without_a_result_dir_keeps_the_local_default(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("BEAKER_JOB_ID", "01JOB")
    monkeypatch.setattr(options, "BEAKER_RESULT_DIR", str(tmp_path / "missing"))
    assert _invoke() == LOCAL_RESULT_DIR
