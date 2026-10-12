"""Links out to Beaker, the GCS console and GitHub."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from urllib.parse import quote

from olmo_eval_api.settings import GCP_PROJECT

BEAKER = "https://beaker.org"
CONSOLE = "https://console.cloud.google.com/storage/browser"


def split_gs_uri(uri: str) -> tuple[str, str]:
    """gs://bucket/key -> (bucket, key)."""
    rest = uri.removeprefix("gs://")
    bucket, _, key = rest.partition("/")
    return bucket, key


def gcs_console_url(gs_uri: str) -> str:
    """Console browser URL for a prefix (gs://bucket/dir/) or an object."""
    bucket, key = split_gs_uri(gs_uri)
    if not key or key.endswith("/"):
        return f"{CONSOLE}/{bucket}/{quote(key.rstrip('/'))}?project={GCP_PROJECT}"
    return f"{CONSOLE}/_details/{bucket}/{quote(key)}?project={GCP_PROJECT}"


def run_links(run: Mapping[Any, Any]) -> dict[str, str | None]:
    experiment = run.get("beaker_experiment_id")
    dataset = run.get("beaker_result_dataset_id")
    workspace = run.get("beaker_workspace")
    repo = run.get("git_repo")
    commit = run.get("git_commit")
    github = None
    if repo and commit and "/" in repo:
        github = f"https://github.com/{repo.removeprefix('https://github.com/')}/commit/{commit}"
    return {
        "beaker_experiment": f"{BEAKER}/ex/{experiment}" if experiment else None,
        "beaker_result_dataset": f"{BEAKER}/ds/{dataset}" if dataset else None,
        "beaker_workspace": f"{BEAKER}/ws/{workspace}" if workspace else None,
        "gcs_console": gcs_console_url(run["gcs_prefix"]) if run.get("gcs_prefix") else None,
        "github_commit": github,
    }
