"""Where and how a run happened: author, git, Beaker job, and machine.

Every helper returns None (or an empty value) instead of raising, so a missing
tool or variable never blocks an upload.
"""

from __future__ import annotations

import os
import platform
import re
import socket
import subprocess
from collections.abc import Mapping
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

TRACKED_PACKAGES = ("olmo-eval", "vllm", "torch", "transformers", "datasets", "litellm", "sglang")


def get_author() -> str:
    """BEAKER_AUTHOR (set by olmo-eval beaker launch), else the local user name."""
    import getpass

    author = (
        os.environ.get("BEAKER_AUTHOR")
        or os.environ.get("USER")
        or os.environ.get("USERNAME")
        or os.environ.get("LOGNAME")
    )
    if author:
        return author
    try:
        return getpass.getuser()
    except Exception:
        return "unknown"


def olmo_eval_version() -> str | None:
    try:
        return version("olmo-eval")
    except PackageNotFoundError:
        return None


def package_versions() -> dict[str, str]:
    """Installed versions of the packages that most affect results."""
    versions: dict[str, str] = {}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            continue
        except Exception:
            continue
    return versions


def _run(args: list[str], cwd: Path | None = None, timeout: float = 5.0) -> str | None:
    try:
        result = subprocess.run(
            args, cwd=cwd, capture_output=True, text=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout


def parse_github_repo(remote_url: str) -> str | None:
    """Turn a git remote URL into "owner/name"."""
    match = re.search(r"[:/]([^/:]+)/([^/]+?)(?:\.git)?/?$", remote_url.strip())
    if not match:
        return None
    return f"{match.group(1)}/{match.group(2)}"


@lru_cache(maxsize=1)
def _local_git_info() -> dict[str, Any]:
    import olmo_eval

    cwd = Path(olmo_eval.__file__).resolve().parent
    commit = _run(["git", "rev-parse", "HEAD"], cwd=cwd)
    if commit is None:
        return {"repo": None, "commit": None, "branch": None, "dirty": None}
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=cwd)
    status = _run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=cwd)
    remote = _run(["git", "config", "--get", "remote.origin.url"], cwd=cwd)
    branch_name = branch.strip() if branch else None
    return {
        "repo": parse_github_repo(remote) if remote else None,
        "commit": commit.strip() or None,
        "branch": None if branch_name in (None, "", "HEAD") else branch_name,
        "dirty": None if status is None else bool(status.strip()),
    }


def git_info(env: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Repository, commit, branch and dirty flag.

    Inside Beaker, gantry sets GITHUB_REPO, GIT_REF and GIT_BRANCH and the checkout
    is clean. Elsewhere the values come from the git checkout that holds olmo_eval.
    """
    env = os.environ if env is None else env
    if env.get("GIT_REF") or env.get("GITHUB_REPO"):
        return {
            "repo": env.get("GITHUB_REPO") or None,
            "commit": env.get("GIT_REF") or None,
            "branch": env.get("GIT_BRANCH") or None,
            "dirty": False,
        }
    return dict(_local_git_info())


def _int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    try:
        parsed = int(float(value))
    except ValueError:
        return None
    return parsed if parsed >= 0 else None


def _number(value: str | None) -> float | int | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    return int(parsed) if parsed.is_integer() else parsed


def cluster_from_hostname(hostname: str | None) -> str | None:
    """Beaker node hostnames start with the cluster name, e.g. jupiter-cs-aus-111."""
    if not hostname:
        return None
    prefix = hostname.split("-", 1)[0].split(".", 1)[0]
    return f"ai2/{prefix}" if prefix else None


def resolve_cluster(launch_clusters: str | None, hostname: str | None) -> str | None:
    """The cluster a job ran on.

    OLMO_EVAL_BEAKER_CLUSTER lists every cluster the launch allowed. With one entry
    it is the answer; with several, the node hostname picks among them.
    """
    from_host = cluster_from_hostname(hostname)
    candidates = [c.strip() for c in (launch_clusters or "").split(",") if c.strip()]
    if len(candidates) == 1:
        return candidates[0]
    if from_host and candidates:
        short = from_host.split("/", 1)[-1]
        for candidate in candidates:
            if candidate.split("/", 1)[-1] == short:
                return candidate
    return from_host


def beaker_info(env: Mapping[str, str] | None = None) -> dict[str, Any] | None:
    """Beaker job metadata from the environment, or None outside Beaker."""
    env = os.environ if env is None else env
    if not env.get("BEAKER_EXPERIMENT_ID"):
        return None
    hostname = env.get("BEAKER_NODE_HOSTNAME") or None

    def get(name: str) -> str | None:
        return env.get(name) or None

    return {
        "experiment_id": get("BEAKER_EXPERIMENT_ID"),
        "workload_id": get("BEAKER_WORKLOAD_ID"),
        "job_id": get("BEAKER_JOB_ID"),
        "task_id": get("BEAKER_TASK_ID"),
        "workspace": get("BEAKER_WORKSPACE"),
        "workspace_id": get("BEAKER_WORKSPACE_ID"),
        "result_dataset_id": get("BEAKER_RESULT_DATASET_ID"),
        "node_hostname": hostname,
        "cluster": resolve_cluster(env.get("OLMO_EVAL_BEAKER_CLUSTER"), hostname),
        "priority": get("OLMO_EVAL_BEAKER_PRIORITY"),
        "image": get("OLMO_EVAL_BEAKER_IMAGE"),
        "budget": get("OLMO_EVAL_BEAKER_BUDGET"),
        "gpu_count": _int(env.get("BEAKER_ASSIGNED_GPU_COUNT")),
        "cpu_count": _number(env.get("BEAKER_ASSIGNED_CPU_COUNT")),
    }


@lru_cache(maxsize=1)
def _gpu_info() -> tuple[str | None, int | None, str | None]:
    names_out = _run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
    names = [line.strip() for line in (names_out or "").splitlines() if line.strip()]
    cuda_version = None
    if names:
        plain = _run(["nvidia-smi"]) or ""
        match = re.search(r"CUDA Version:\s*([0-9.]+)", plain)
        cuda_version = match.group(1) if match else None
    gpu_type = names[0] if names else None
    return gpu_type, (len(names) if names else None), cuda_version


def environment_info() -> dict[str, Any]:
    """Host, Python, package and GPU details."""
    try:
        hostname: str | None = socket.gethostname()
    except OSError:
        hostname = None
    gpu_type, gpu_count, cuda_version = _gpu_info()
    return {
        "hostname": hostname,
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "packages": package_versions(),
        "cuda_version": cuda_version,
        "gpu_type": gpu_type,
        "gpu_count": gpu_count,
    }
