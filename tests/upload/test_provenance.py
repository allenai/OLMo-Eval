"""Provenance capture: Beaker env, git, author."""

from __future__ import annotations

import pytest

from olmo_eval.upload.provenance import (
    beaker_info,
    cluster_from_hostname,
    environment_info,
    get_author,
    git_info,
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
