"""Infrastructure constants for Beaker, clusters, and storage mounts.

This module contains configuration constants for AI2's compute infrastructure,
including Beaker workspace settings, cluster definitions, and Weka storage mounts.
"""

from enum import StrEnum

# =============================================================================
# Beaker Configuration
# =============================================================================

BEAKER_DEFAULT_WORKSPACE = "ai2/oe-data"
"""Default Beaker workspace for evaluation jobs."""

BEAKER_DEFAULT_BUDGET = "ai2/oe-other"
"""Default budget allocation for Beaker jobs."""

BEAKER_DEFAULT_PRIORITY = "normal"
"""Default job priority level."""

BEAKER_DEFAULT_IMAGE = "ai2-tylerm/olmo-eval-cu1281-trc2100-amd64"
"""Default Docker image for Beaker evaluation jobs."""

BEAKER_SANDBOX_IMAGE = "ai2-tylerm/olmo-eval-cu1281-trc2100-amd64-sandbox"
"""Docker image for Beaker evaluation jobs with sandbox support (includes Podman)."""

BEAKER_RESULT_DIR = "/results"
"""Default directory for evaluation results in Beaker jobs."""

LOCAL_RESULT_DIR = "/tmp/results/"
"""Default directory for evaluation results when running locally."""

BEAKER_UV_CACHE_DIR = "/weka/oe-eval-default/olmo-eval-pypi-cache"
"""Default UV cache directory for Beaker jobs (on Weka shared storage)."""

DEFAULT_MAX_GPUS_PER_NODE = 8
"""Default maximum GPUs available per node for experiment splitting."""


class BeakerPriority(StrEnum):
    """Valid Beaker job priority levels."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


# Version constraints for Beaker tooling
BEAKER_PY_MIN_VERSION = "2.5.0"
BEAKER_PY_MAX_VERSION = "3"
BEAKER_GANTRY_MIN_VERSION = "3.2.0"
BEAKER_GANTRY_MAX_VERSION = "4"


# =============================================================================
# Cluster Definitions
# =============================================================================

BEAKER_KNOWN_CLUSTERS: dict[str, list[str]] = {
    "aus": [
        "ai2/jupiter",
        "ai2/neptune",
        "ai2/saturn",
        "ai2/ceres",
    ],
    "aus80g": [
        "ai2/jupiter",
        "ai2/saturn",
        "ai2/ceres",
    ],
    "h100": [
        "ai2/jupiter",
        "ai2/ceres",
    ],
    "a100": [
        "ai2/saturn",
    ],
    "l40": [
        "ai2/neptune",
    ],
    "80g": [
        "ai2/jupiter",
        "ai2/saturn",
        "ai2/ceres",
    ],
}
"""Mapping of cluster group aliases to their constituent Beaker clusters."""


NEW_CLUSTER_ALIASES: dict[str, str] = {
    "ai2/allennlp-elanding-a100-40g": "ai2/allennlp",
    "ai2/allennlp-dev-a100-sea": "ai2/allennlp",
    "ai2/ceres-dev-h100-aus-ib": "ai2/ceres",
    "ai2/ceres-cirrascale": "ai2/ceres",
    "ai2/jupiter-batch-h100-aus-ib": "ai2/jupiter",
    "ai2/jupiter-cirrascale-2": "ai2/jupiter",
    "ai2/neptune-dev-l40-aus": "ai2/neptune",
    "ai2/neptune-cirrascale": "ai2/neptune",
    "ai2/phobos-dev-aus": "ai2/phobos",
    "ai2/phobos-cirrascale": "ai2/phobos",
    "ai2/prior-dev-a6000-sea": "ai2/prior",
    "ai2/prior-elanding": "ai2/prior",
    "ai2/prior-elanding-rtx8000": "ai2/prior-rtx8000",
    "ai2/prior-dev-rtx8000-sea": "ai2/prior-rtx8000",
    "ai2/rhea-dev-a6000-aus": "ai2/rhea",
    "ai2/rhea-cirrascale": "ai2/rhea",
    "ai2/saturn-dev-a100-aus": "ai2/saturn",
    "ai2/saturn-cirrascale": "ai2/saturn",
    "ai2/titan-batch-b200-aus-ib": "ai2/titan",
    "ai2/titan-cirrascale": "ai2/titan",
    "ai2/triton-dev-l40-aus": "ai2/triton",
    "ai2/triton-cirrascale": "ai2/triton",
}
"""Mapping of specific cluster names to their canonical short names."""


# =============================================================================
# Weka Storage Mounts
# =============================================================================

WEKA_CLUSTERS: set[str] = {
    "ai2/jupiter",
    "ai2/saturn",
    "ai2/ceres",
    "ai2/neptune",
    "ai2/titan",
    "ai2/holmes",
}
"""Clusters with Weka storage available.

Uses canonical cluster names (the short form from NEW_CLUSTER_ALIASES values
and BEAKER_KNOWN_CLUSTERS entries, e.g. ``"ai2/jupiter"``).
"""


WEKA_MOUNTS: tuple[str, ...] = (
    "ai1-default",
    "climate-default",
    "dfive-default",
    "mosaic-default",
    "nora-default",
    "nsf-uchicago-apto",
    "oe-adapt-default",
    "oe-data-default",
    "oe-eval-default",
    "oe-training-default",
    "prior-default",
    "reviz-default",
    "skylight-default",
)
"""Available Weka storage mount points for Beaker jobs."""


DEFAULT_WEKA_BUCKETS: tuple[str, ...] = (
    "oe-training-default",
    "oe-eval-default",
    "oe-adapt-default",
)
"""Weka buckets mounted on every eval job; ``--weka-bucket`` adds to these."""


def cluster_has_weka(cluster: str) -> bool:
    """Check whether a cluster has Weka storage available.

    Handles aliases (e.g. ``"aus"``), legacy names
    (e.g. ``"ai2/jupiter-cirrascale-2"``), and canonical names
    (e.g. ``"ai2/jupiter"``).  If *cluster* is an alias that maps to
    multiple clusters, returns ``True`` only when **all** of them have Weka.
    """
    # Resolve legacy name → canonical
    canonical = NEW_CLUSTER_ALIASES.get(cluster, cluster)

    # Alias that expands to a group of clusters
    if canonical in BEAKER_KNOWN_CLUSTERS:
        return all(c in WEKA_CLUSTERS for c in BEAKER_KNOWN_CLUSTERS[canonical])

    return canonical in WEKA_CLUSTERS


# =============================================================================
# OE-Eval Configuration
# =============================================================================

OE_EVAL_GIT_URL = "git@github.com:allenai/oe-eval-internal.git"
"""Git URL for the oe-eval-internal repository."""

OE_EVAL_COMMIT_HASH: str | None = None
"""Pinned commit hash for oe-eval (None means latest)."""

OE_EVAL_LAUNCH_COMMAND = "oe_eval/launch.py"
"""Entry point script for launching evaluations."""


# =============================================================================
# Backend Dependencies
# =============================================================================

BACKEND_OPTIONAL_GROUPS: dict[str, str | None] = {
    "vllm": "vllm",
    "vllm_server": "vllm",
    "hf": "hf",
    "olmo_core": "olmo_core",
    "olmo_core_vlm": "olmo_core_vlm",
    "litellm": "litellm",
    "mock": None,
}
"""Mapping of backend types to their pyproject.toml optional dependency group names.

Used for auto-installing backend dependencies at runtime when launching
Beaker jobs via `pip install .[group_name]`.
"""
