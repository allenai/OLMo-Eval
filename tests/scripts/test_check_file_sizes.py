"""The file size check must block oversized tracked files and nothing else."""

import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/check_file_sizes.sh"


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    _git(tmp_path, "init", "-q")
    return tmp_path


def _add(repo: Path, name: str, size: int) -> None:
    path = repo / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    _git(repo, "add", name)


def _run(repo: Path, max_bytes: int = 100) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=repo,
        env={**os.environ, "MAX_BYTES": str(max_bytes)},
        capture_output=True,
        text=True,
    )


def test_passes_when_all_files_are_within_limit(repo: Path):
    _add(repo, "small.txt", 100)
    _add(repo, "dir with space/also small.txt", 10)
    result = _run(repo)
    assert result.returncode == 0, result.stderr
    assert "All 2 tracked files" in result.stdout


def test_fails_and_names_oversized_file(repo: Path):
    _add(repo, "small.txt", 10)
    _add(repo, "data/big.jsonl", 101)
    result = _run(repo)
    assert result.returncode == 1
    assert "data/big.jsonl" in result.stderr
    assert "small.txt" not in result.stderr


def test_checks_staged_content_not_working_tree(repo: Path):
    _add(repo, "file.txt", 101)
    (repo / "file.txt").write_bytes(b"x")
    assert _run(repo).returncode == 1


def test_ignores_untracked_files(repo: Path):
    _add(repo, "small.txt", 10)
    (repo / "untracked.bin").write_bytes(b"x" * 1000)
    assert _run(repo).returncode == 0


def test_excluded_paths_may_exceed_limit(repo: Path):
    _add(repo, "uv.lock", 1000)
    _add(repo, "nested/uv.lock", 1000)
    result = _run(repo)
    assert result.returncode == 1
    assert "nested/uv.lock" in result.stderr
    assert not any(line.endswith("  uv.lock") for line in result.stderr.splitlines())
