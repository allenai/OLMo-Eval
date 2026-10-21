"""Tests for package metadata."""

import tomllib
from pathlib import Path

import olmo_eval


def test_version_matches_pyproject():
    pyproject = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())

    assert olmo_eval.__version__ == pyproject["project"]["version"]
