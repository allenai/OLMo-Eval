"""Provenance capture: Beaker env, git, author."""

from __future__ import annotations

import os
import sys
from importlib.metadata import PackageNotFoundError
from pathlib import Path

import pytest

from olmo_eval.upload import provenance
from olmo_eval.upload.provenance import (
    beaker_info,
    cluster_from_hostname,
    environment_info,
    get_author,
    git_info,
    interpreter_package_versions,
    package_versions,
    parse_github_repo,
    resolve_cluster,
)

BEAKER_ENV = {
    "BEAKER_EXPERIMENT_ID": "01M3TFPE2PQ9DDAZD3YT52V0JZ",
    "BEAKER_WORKLOAD_ID": "01M3TFPE2PQ9DDAZD3YT52V0JZ",
    "BEAKER_JOB_ID": "01M3TFPE69KR939WRYMF981EP3",
    "BEAKER_TASK_ID": "01M3TFPE2TRWCBD3EHFRFWQM11",
    "BEAKER_WORKSPACE_ID": "01KG8QW7T4RYEW59EFXSJ7029N",
    "BEAKER_RESULT_DATASET_ID": "01M3TFPE2X873CCF4KMSKRME4D",
    "BEAKER_NODE_HOSTNAME": "jupiter-cs-aus-111.reviz.ai2.in",
    "BEAKER_ASSIGNED_GPU_COUNT": "4",
    "BEAKER_ASSIGNED_CPU_COUNT": "90",
    "BEAKER_WORKSPACE": "ai2/olmo-eval-debug",
    "OLMO_EVAL_BEAKER_CLUSTER": "ai2/jupiter,ai2/ceres",
    "OLMO_EVAL_BEAKER_PRIORITY": "high",
    "OLMO_EVAL_BEAKER_IMAGE": "beaker://petew/olmo-eval-vllm",
    "OLMO_EVAL_BEAKER_BUDGET": "ai2/oe-other",
    "GITHUB_REPO": "allenai/olmo-eval-internal",
    "GIT_REF": "e81a69bf44a4d7a86033d7ccc7fb7d0e41db48a7",
    "GIT_BRANCH": "maliam/omniscience",
}


def test_beaker_info_from_job_env() -> None:
    info = beaker_info(BEAKER_ENV)
    assert info == {
        "experiment_id": "01M3TFPE2PQ9DDAZD3YT52V0JZ",
        "workload_id": "01M3TFPE2PQ9DDAZD3YT52V0JZ",
        "job_id": "01M3TFPE69KR939WRYMF981EP3",
        "task_id": "01M3TFPE2TRWCBD3EHFRFWQM11",
        "workspace": "ai2/olmo-eval-debug",
        "workspace_id": "01KG8QW7T4RYEW59EFXSJ7029N",
        "result_dataset_id": "01M3TFPE2X873CCF4KMSKRME4D",
        "node_hostname": "jupiter-cs-aus-111.reviz.ai2.in",
        "cluster": "ai2/jupiter",
        "priority": "high",
        "image": "beaker://petew/olmo-eval-vllm",
        "budget": "ai2/oe-other",
        "gpu_count": 4,
        "cpu_count": 90,
    }


def test_beaker_info_is_none_outside_beaker() -> None:
    assert beaker_info({"BEAKER_WORKSPACE": "ai2/x"}) is None


def test_beaker_info_tolerates_bad_numbers() -> None:
    env = {"BEAKER_EXPERIMENT_ID": "x", "BEAKER_ASSIGNED_GPU_COUNT": "n/a"}
    info = beaker_info(env)
    assert info is not None
    assert info["gpu_count"] is None
    assert info["cluster"] is None


@pytest.mark.parametrize(
    ("launch", "host", "expected"),
    [
        ("ai2/saturn", "jupiter-cs-1", "ai2/saturn"),
        ("ai2/jupiter,ai2/ceres", "ceres-cs-9.reviz.ai2.in", "ai2/ceres"),
        ("ai2/jupiter,ai2/ceres", "neptune-1", "ai2/neptune"),
        (None, "titan-cs-3", "ai2/titan"),
        (None, None, None),
    ],
)
def test_resolve_cluster(launch, host, expected) -> None:
    assert resolve_cluster(launch, host) == expected


def test_cluster_from_hostname() -> None:
    assert cluster_from_hostname("jupiter-cs-aus-111.reviz.ai2.in") == "ai2/jupiter"
    assert cluster_from_hostname("") is None


def test_git_info_in_beaker() -> None:
    assert git_info(BEAKER_ENV) == {
        "repo": "allenai/olmo-eval-internal",
        "commit": "e81a69bf44a4d7a86033d7ccc7fb7d0e41db48a7",
        "branch": "maliam/omniscience",
        "dirty": False,
    }


def test_git_info_locally_has_the_expected_shape() -> None:
    info = git_info({})
    assert set(info) == {"repo", "commit", "branch", "dirty"}


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("git@github.com:allenai/olmo-eval.git", "allenai/olmo-eval"),
        ("https://github.com/allenai/olmo-eval.git", "allenai/olmo-eval"),
        ("https://github.com/allenai/olmo-eval", "allenai/olmo-eval"),
        ("ssh://git@github.com/allenai/olmo-eval-internal.git\n", "allenai/olmo-eval-internal"),
        ("not a url", None),
    ],
)
def test_parse_github_repo(url, expected) -> None:
    assert parse_github_repo(url) == expected


def test_author_prefers_beaker(monkeypatch) -> None:
    monkeypatch.setenv("BEAKER_AUTHOR", "maliam")
    monkeypatch.setenv("USER", "local")
    assert get_author() == "maliam"
    monkeypatch.delenv("BEAKER_AUTHOR")
    assert get_author() == "local"


def test_environment_info_shape() -> None:
    info = environment_info()
    assert info["python_version"]
    assert "olmo-eval" in info["packages"]
    assert set(info) == {
        "hostname",
        "platform",
        "python_version",
        "packages",
        "cuda_version",
        "gpu_type",
        "gpu_count",
    }


ISOLATED_OUTPUT = '{"vllm": "0.19.0", "torch": "2.10.0", "transformers": "5.9.0"}\n'


@pytest.fixture
def fake_vllm_python(tmp_path: Path) -> str:
    """A stand-in for /opt/vllm-venv/bin/python in a venv other than the running one.

    Like a real venv interpreter, it is a symlink to the same base Python as the running
    interpreter, so only the venv directory tells the two apart.
    """
    path = tmp_path / "vllm-venv" / "bin" / "python"
    path.parent.mkdir(parents=True)
    path.symlink_to(os.path.realpath(sys.executable))
    return str(path)


@pytest.fixture(autouse=True)
def _clear_interpreter_cache():
    interpreter_package_versions.cache_clear()
    yield
    interpreter_package_versions.cache_clear()


def _main_env(monkeypatch, installed: dict[str, str]) -> None:
    def fake_version(name: str) -> str:
        if name not in installed:
            raise PackageNotFoundError(name)
        return installed[name]

    monkeypatch.setattr(provenance, "version", fake_version)


def test_package_versions_records_the_separate_vllm_venv(monkeypatch, fake_vllm_python) -> None:
    calls = []

    def fake_run(args, cwd=None, timeout=5.0):
        calls.append(args)
        return ISOLATED_OUTPUT

    monkeypatch.setattr(provenance, "_run", fake_run)
    _main_env(monkeypatch, {"olmo-eval": "0.1.0", "transformers": "5.8.0"})

    versions = package_versions({"VLLM_PYTHON": fake_vllm_python})

    assert calls[0][:2] == [fake_vllm_python, "-I"]
    assert calls[0][-3:] == ["vllm", "torch", "transformers"]
    assert versions == {
        "olmo-eval": "0.1.0",
        "transformers": "5.8.0",
        "vllm": "0.19.0",
        "vllm-venv/vllm": "0.19.0",
        "vllm-venv/torch": "2.10.0",
        "vllm-venv/transformers": "5.9.0",
    }


def test_package_versions_records_the_serving_vllm(monkeypatch, fake_vllm_python) -> None:
    monkeypatch.setattr(provenance, "_run", lambda *a, **k: ISOLATED_OUTPUT)
    _main_env(monkeypatch, {"vllm": "0.18.0"})

    versions = package_versions({"VLLM_PYTHON": fake_vllm_python})

    assert versions["vllm"] == "0.19.0"
    assert versions["vllm-venv/vllm"] == "0.19.0"


def test_package_versions_without_vllm_python(monkeypatch) -> None:
    def fail_run(*args, **kwargs):
        raise AssertionError("no interpreter should be spawned")

    monkeypatch.setattr(provenance, "_run", fail_run)
    _main_env(monkeypatch, {"olmo-eval": "0.1.0"})
    assert package_versions({}) == {"olmo-eval": "0.1.0"}


def test_package_versions_ignores_vllm_python_that_is_this_interpreter(monkeypatch) -> None:
    def fail_run(*args, **kwargs):
        raise AssertionError("must not spawn the current interpreter")

    monkeypatch.setattr(provenance, "_run", fail_run)
    versions = package_versions({"VLLM_PYTHON": sys.executable})
    assert not any(key.startswith("vllm-venv/") for key in versions)


def test_package_versions_ignores_missing_vllm_python(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(provenance, "_run", lambda *a, **k: ISOLATED_OUTPUT)
    versions = package_versions({"VLLM_PYTHON": str(tmp_path / "missing")})
    assert not any(key.startswith("vllm-venv/") for key in versions)


@pytest.mark.parametrize("output", [None, "", "not json", "[1, 2]"])
def test_package_versions_tolerates_a_failing_vllm_venv(
    monkeypatch, fake_vllm_python, output
) -> None:
    monkeypatch.setattr(provenance, "_run", lambda *a, **k: output)
    _main_env(monkeypatch, {"olmo-eval": "0.1.0"})
    assert package_versions({"VLLM_PYTHON": fake_vllm_python}) == {"olmo-eval": "0.1.0"}


def test_interpreter_package_versions_queries_a_real_interpreter() -> None:
    versions = interpreter_package_versions(sys.executable, ("olmo-eval", "no-such-dist-xyz"))
    assert "olmo-eval" in versions
    assert "no-such-dist-xyz" not in versions
