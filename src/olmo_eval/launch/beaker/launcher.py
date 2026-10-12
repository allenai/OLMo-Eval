"""Unified Beaker launcher for olmo-eval jobs using beaker-gantry.

Dataclass-based API for submitting evaluation jobs
to Beaker using beaker-gantry's Python API for reproducible experiments.

Example:
    config = BeakerJobConfig(
        name="eval-llama3-mmlu",
        command=["olmo-eval", "run", "-m", "llama3.1-8b", "-t", "mmlu"],
        cluster="ai2/ceres",
        workspace="ai2/oe-data",
        budget="ai2/oe-other",
        num_gpus=1,
    )
    launcher = BeakerLauncher(workspace="ai2/oe-data")
    experiment = launcher.launch(config)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import UTC
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

from olmo_eval.common.constants.infrastructure import (
    BEAKER_DEFAULT_IMAGE,
    BEAKER_DEFAULT_WORKSPACE,
    BEAKER_KNOWN_CLUSTERS,
    NEW_CLUSTER_ALIASES,
)
from olmo_eval.common.repr import hide_unset

if TYPE_CHECKING:
    from beaker import Beaker, BeakerExperiment, BeakerGroup

log = logging.getLogger(__name__)

__all__ = [
    "BeakerEnvSecret",
    "BeakerWekaBucket",
    "BeakerJobConfig",
    "BeakerLauncher",
    "build_install_command",
    "calculate_experiment_splits",
    "normalize_provider_package",
    "parse_install_spec",
    "parse_task_with_priority",
    "validate_priority_configuration",
    "print_experiment_config",
    "resolve_clusters",
    "validate_min_runtime",
]

# Rich console for pretty printing
_console = Console()

_DIRECT_URL_ARTIFACT_SUFFIXES = (".whl", ".zip", ".tar.gz", ".tar.bz2", ".tar.xz", ".tgz")


def print_experiment_config(
    spec_dict: dict[str, Any],
    name: str | None = None,
    show_header: bool = True,
) -> None:
    """Pretty print a Beaker experiment spec with colorization.

    Uses Rich library for syntax-highlighted JSON output with optional
    header panel showing experiment metadata.

    Args:
        spec_dict: The experiment spec as a dictionary.
        name: Optional experiment name to display in header.
        show_header: Whether to show the header panel.

    Example:
        spec = launcher.build_spec(config)
        print_experiment_config(spec.to_json(), name=config.name)
    """
    # Extract key info for header
    tasks = spec_dict.get("tasks", [])
    task_spec = tasks[0] if tasks else {}
    context = task_spec.get("context", {})
    resources = task_spec.get("resources", {})
    constraints = task_spec.get("constraints", {})
    command = task_spec.get("command", [])

    # Build header with key metadata
    if show_header:
        header_lines = []
        if name:
            header_lines.append(f"[bold cyan]Experiment:[/] {name}")

        # Extract model and tasks from command
        model = None
        task_names = []
        for i, arg in enumerate(command):
            if arg == "-m" and i + 1 < len(command):
                model = command[i + 1]
            elif arg == "-t" and i + 1 < len(command):
                task_names.append(command[i + 1])

        if model:
            header_lines.append(f"[bold blue]Model:[/] {model}")
        if task_names:
            header_lines.append(f"[bold blue]Tasks:[/] {', '.join(task_names)}")

        # Resource info
        priority = context.get("priority", "normal")
        priority_color = {
            "low": "dim",
            "normal": "white",
            "high": "yellow",
            "urgent": "red bold",
        }.get(priority, "white")
        header_lines.append(f"[bold blue]Priority:[/] [{priority_color}]{priority}[/]")

        gpu_count = resources.get("gpuCount", 1)
        header_lines.append(f"[bold blue]GPUs:[/] {gpu_count}")

        clusters = constraints.get("cluster", [])
        if clusters:
            header_lines.append(f"[bold blue]Clusters:[/] {', '.join(clusters)}")

        min_runtime_ns = context.get("minRuntime")
        if min_runtime_ns is not None:
            min_runtime = _format_duration(min_runtime_ns // 1_000_000_000)
            header_lines.append(f"[bold blue]Min runtime:[/] {min_runtime}")
        else:
            preemptible = context.get("preemptible", True)
            preempt_str = "[green]yes[/]" if preemptible else "[red]no[/]"
            header_lines.append(f"[bold blue]Preemptible:[/] {preempt_str}")

        header_text = Text.from_markup("\n".join(header_lines))
        _console.print(Panel(header_text, title="[bold]Beaker Experiment[/]", border_style="blue"))

    # Print JSON with syntax highlighting
    json_str = json.dumps(spec_dict, indent=2)
    syntax = Syntax(json_str, "json", theme="monokai", line_numbers=False)
    _console.print(syntax)


# Valid Beaker priority levels
VALID_PRIORITIES = ("low", "normal", "high", "urgent")


def parse_task_with_priority(task_spec: str, default_priority: str = "normal") -> tuple[str, str]:
    """Parse task spec with optional @priority suffix.

    Format: task_name[@priority] or task_name:variant[@priority]

    Examples:
        - "mmlu" -> ("mmlu", "normal")
        - "mmlu@high" -> ("mmlu", "high")
        - "arc_easy:mc@high" -> ("arc_easy:mc", "high")

    Args:
        task_spec: Task specification, optionally with @priority suffix.
        default_priority: Priority to use if not specified in task_spec.

    Returns:
        Tuple of (task_name, priority).

    Raises:
        ValueError: If priority is not valid.
    """
    if "@" in task_spec:
        task_name, priority = task_spec.rsplit("@", 1)
        if priority not in VALID_PRIORITIES:
            raise ValueError(
                f"Invalid priority '{priority}'. Must be one of: {', '.join(VALID_PRIORITIES)}"
            )
        return task_name, priority
    return task_spec, default_priority


def validate_priority_configuration(
    tasks: tuple[str, ...] | list[str],
    default_priority: str = "normal",
) -> dict[str, list[str]]:
    """Group tasks by priority.

    Tasks can specify priority via @priority suffix (e.g., "mmlu@high").
    Tasks without @priority use the default_priority.

    Args:
        tasks: Task specifications (may include @priority suffixes).
        default_priority: Default priority for tasks without @priority suffix.

    Returns:
        Dictionary mapping priority levels to lists of task names.

    Examples:
        >>> validate_priority_configuration(["mmlu", "gsm8k"])
        {"normal": ["mmlu", "gsm8k"]}

        >>> validate_priority_configuration(["mmlu@high", "gsm8k@normal"])
        {"high": ["mmlu"], "normal": ["gsm8k"]}

        >>> validate_priority_configuration(["mmlu", "gsm8k"], "high")
        {"high": ["mmlu", "gsm8k"]}
    """
    from collections import defaultdict

    tasks_by_priority: dict[str, list[str]] = defaultdict(list)

    for task_spec in tasks:
        if "@" in task_spec:
            task_name, task_priority = parse_task_with_priority(task_spec)
            tasks_by_priority[task_priority].append(task_name)
        else:
            tasks_by_priority[default_priority].append(task_spec)

    return dict(tasks_by_priority)


def calculate_experiment_splits(
    tasks: list[str],
    gpus_per_model: int,
    parallelism: int,
    max_gpus_per_node: int,
) -> list[dict[str, Any]]:
    """Calculate how to split tasks across experiments based on GPU constraints.

    When the total GPU requirement (gpus_per_model × parallelism) exceeds the
    maximum GPUs available per node, tasks are distributed across multiple experiments.
    Each experiment runs a subset of tasks with reduced parallelism.

    Args:
        tasks: List of expanded task specs to run.
        gpus_per_model: GPUs required per model instance.
        parallelism: Number of model instances desired.
        max_gpus_per_node: Maximum GPUs available per node.

    Returns:
        List of dicts, each containing:
        - tasks: subset of tasks for this experiment
        - num_gpus: GPUs to request for this experiment
        - parallelism: effective parallelism for this split
    """
    import math

    total_gpus_needed = gpus_per_model * parallelism

    if total_gpus_needed <= max_gpus_per_node:
        # Fits on single node - no splitting needed
        return [
            {
                "tasks": tasks,
                "num_gpus": total_gpus_needed,
                "parallelism": parallelism,
            }
        ]

    # Need to split across multiple experiments
    # Calculate how many instances can fit per experiment
    instances_per_experiment = max_gpus_per_node // gpus_per_model
    if instances_per_experiment < 1:
        # Edge case: single model instance needs more GPUs than max
        # Fall back to 1 instance per experiment
        instances_per_experiment = 1

    gpus_per_experiment = instances_per_experiment * gpus_per_model
    num_experiments = math.ceil(parallelism / instances_per_experiment)

    # Distribute tasks across experiments
    tasks_per_experiment = math.ceil(len(tasks) / num_experiments)
    splits = []

    for i in range(num_experiments):
        start_idx = i * tasks_per_experiment
        end_idx = min(start_idx + tasks_per_experiment, len(tasks))
        split_tasks = tasks[start_idx:end_idx]

        if split_tasks:  # Only add if there are tasks
            splits.append(
                {
                    "tasks": split_tasks,
                    "num_gpus": gpus_per_experiment,
                    "parallelism": instances_per_experiment,
                }
            )

    return splits


@dataclass
class BeakerEnvSecret:
    """Environment variable sourced from a Beaker secret.

    Attributes:
        name: Environment variable name to set in the container.
        secret: Name of the secret in Beaker's secret store.
    """

    name: str
    secret: str


@dataclass
class BeakerWekaBucket:
    """Weka bucket mount configuration.

    Attributes:
        bucket: Weka bucket name (e.g., "oe-eval-default").
        mount: Mount path in the container. Defaults to /weka/{bucket}.
    """

    bucket: str
    mount: str | None = None

    def __post_init__(self) -> None:
        if self.mount is None:
            self.mount = f"/weka/{self.bucket}"


@hide_unset()
@dataclass
class BeakerJobConfig:
    """Configuration for a Beaker evaluation job.

    This dataclass provides sensible defaults while allowing full customization
    of job parameters. Use `BeakerLauncher.launch()` to submit jobs.

    Example:
        config = BeakerJobConfig(
            name="eval-llama3-mmlu",
            command=["olmo-eval", "run", "-m", "llama3.1-8b", "-t", "mmlu"],
            cluster="ai2/ceres",
            workspace="ai2/oe-data",
            budget="ai2/oe-other",
            num_gpus=1,
        )

    Attributes:
        name: Experiment name (required).
        command: Command to run in the container (required).
        cluster: Cluster alias ("h100", "a100", "aus") or full name(s) (required).
        workspace: Beaker workspace (required).
        budget: Beaker budget. If None, gantry uses the workspace's bound budget.
        num_gpus: Number of GPUs to request.
        shared_memory: Shared memory size (e.g., "10GiB").
        priority: Job priority level.
        preemptible: Whether the job can be preempted. Ignored and set to None when
            min_runtime is set; combining False with min_runtime raises ValueError.
        min_runtime: Minimum time the job runs before Beaker may preempt it (e.g., "2h").
            Replaces preemptible, which Beaker has deprecated.
        timeout: Job timeout (e.g., "24h", "30m").
        retries: Number of retries on failure.
        beaker_image: Container image to use.
        description: Optional job description.
        weka_buckets: Weka storage mounts.
        nfs: Whether to mount NFS.
        env_vars: Additional environment variables.
        env_secrets: Environment variables from Beaker secrets.
        result_path: Path for job results.
    """

    # Required
    name: str
    command: list[str]
    cluster: str | list[str]  # Cluster alias ("h100", "a100", "aus") or full name(s)
    workspace: str  # Beaker workspace
    budget: str | None = None  # Beaker budget; None uses the workspace's bound budget

    # Resources
    num_gpus: int = 0
    shared_memory: str = "10GiB"

    # Job settings
    priority: str = "normal"
    preemptible: bool | None = True
    min_runtime: str | None = None
    timeout: str | None = "24h"
    retries: int | None = None

    # Beaker settings
    beaker_image: str = BEAKER_DEFAULT_IMAGE
    description: str | None = None

    # Storage - defaults include common eval buckets
    weka_buckets: list[BeakerWekaBucket] = field(
        default_factory=lambda: [
            BeakerWekaBucket("oe-training-default"),
            BeakerWekaBucket("oe-eval-default"),
            BeakerWekaBucket("oe-adapt-default"),
        ]
    )
    nfs: bool = False

    # Environment - defaults include HuggingFace cache on Weka
    env_vars: dict[str, str] = field(
        default_factory=lambda: {
            "HF_HOME": "/weka/oe-eval-default/oyvindt/hf-cache",
            "HF_HUB_CACHE": "/weka/oe-eval-default/oyvindt/hf-cache",
            "INSPECT_CACHE_DIR": "/weka/oe-training-default/olmo-eval/inspect-cache",
        }
    )
    env_secrets: list[BeakerEnvSecret] = field(default_factory=list)

    # Result path
    result_path: str = "/results"

    # Optional dependency groups to install at runtime (e.g., ["vllm", "s3"])
    extras: list[str] = field(default_factory=list)

    # Group assignment - experiment will be added to these groups at creation time
    groups: list[str] | None = None

    # AWS S3 access - when True, injects user's AWS credentials as env secrets
    inject_aws_credentials: bool = False

    # GCS access - when True, injects user's GCS credentials as env secret
    inject_gcs_credentials: bool = False

    # Results upload - when True, injects the shared uploader service account key so the
    # job can upload results to the dashboard's ingest service
    inject_upload_credentials: bool = False

    # Provider-specific dependencies (from provider config)
    provider_packages: list[str] | None = None

    # Task-specific packages to install at runtime
    task_packages: list[str] | None = None

    # Follow mode - when True, wait for experiment to complete
    follow: bool = False

    # Sandbox mode - when True, use Podman-enabled base image and set sandbox env vars
    enable_sandbox: bool = False

    # Registry mirror setup script to run during install (for sandbox jobs)
    setup_registry_mirror: bool = False

    # Install vLLM in isolated venv (for server mode to avoid dependency conflicts)
    # When True, vLLM is installed in /opt/vllm-venv and VLLM_PYTHON points to it.
    # Use this for external evals that run vLLM as a server subprocess.
    vllm_isolated_venv: bool = False

    # Setup Modal secret for GCP Artifact Registry access (for Modal sandboxes)
    # When True, runs setup_modal_gcp_secret script during install.
    # Requires MODAL_GCP_SECRET_NAME env var to be set with the secret name.
    setup_modal_gcp_secret: bool = False

    def __post_init__(self) -> None:
        if self.min_runtime is not None:
            if self.preemptible is False:
                raise ValueError(
                    "preemptible=False cannot be combined with min_runtime; set min_runtime alone"
                )
            self.min_runtime = validate_min_runtime(self.min_runtime)
            self.preemptible = None
        elif self.preemptible is None:
            self.preemptible = True


def resolve_clusters(cluster: str | list[str]) -> list[str]:
    """Resolve cluster aliases to full cluster names.

    Supports:
    - Aliases: "h100", "a100", "aus", "goog", "80g", etc.
    - Full names: "ai2/jupiter", "ai2/saturn", etc.
    - Legacy names: "ai2/jupiter-cirrascale-2" -> "ai2/jupiter"

    Args:
        cluster: Single cluster or list of clusters/aliases.

    Returns:
        List of resolved cluster names.
    """
    clusters = [cluster] if isinstance(cluster, str) else list(cluster)
    resolved: list[str] = []

    for c in clusters:
        # Check if it's a legacy name that needs aliasing
        if c in NEW_CLUSTER_ALIASES:
            c = NEW_CLUSTER_ALIASES[c]

        # Check if it's a known alias group
        if c in BEAKER_KNOWN_CLUSTERS:
            resolved.extend(BEAKER_KNOWN_CLUSTERS[c])
        else:
            resolved.append(c)

    return list(set(resolved))  # Deduplicate


def parse_install_spec(package: str) -> tuple[str, list[str]]:
    """Parse a package specifier into package URL and install flags.

    Separates pip/uv install flags (like --no-build-isolation) from the
    package specifier. Flags must start with '-' and come after the package.

    Args:
        package: Package specifier with optional install flags.

    Returns:
        Tuple of (package_spec, install_flags).

    Examples:
        >>> parse_install_spec("git+https://github.com/user/repo@v1.0 --no-build-isolation")
        ("git+https://github.com/user/repo@v1.0", ["--no-build-isolation"])
        >>> parse_install_spec("vllm==0.14.0")
        ("vllm==0.14.0", [])
    """
    parts = package.split()
    if len(parts) == 1:
        return package, []

    # Find where flags start (first part starting with -)
    pkg_parts = []
    flags = []
    in_flags = False
    for part in parts:
        if part.startswith("-") or in_flags:
            in_flags = True
            flags.append(part)
        else:
            pkg_parts.append(part)

    return " ".join(pkg_parts), flags


def _is_direct_url_artifact(package: str) -> bool:
    if "://" not in package:
        return False
    path = urlsplit(package).path.lower()
    return path.endswith(_DIRECT_URL_ARTIFACT_SUFFIXES)


def normalize_provider_package(package: str) -> str:
    """Normalize a provider package specifier for pip installation.

    Handles various package formats:
    - GitHub URL: https://github.com/user/repo -> git+https://github.com/user/repo
    - GitHub URL with branch: https://github.com/user/repo@branch -> git+https://github.com/user/repo@branch
    - Git URL (explicit): git+https://github.com/user/repo@tag (unchanged)
    - Local path: /path/to/local/package (unchanged)
    - PyPI version: vllm==0.14.0 (unchanged)
    - PyPI with extras: vllm[runai]==0.14.0 (unchanged)

    Note: This function strips install flags. Use parse_install_spec() first
    if you need to preserve flags.

    Args:
        package: Package specifier string (may include install flags).

    Returns:
        Normalized package specifier suitable for pip install (without flags).
    """
    # Strip any install flags first
    pkg_spec, _ = parse_install_spec(package)

    # Named direct references ("name @ url") should preserve the package name
    # while still normalizing the URL/source portion.
    if " @ " in pkg_spec:
        name, source = pkg_spec.split(" @ ", 1)
        return f"{name} @ {normalize_provider_package(source)}"

    # Already a git+ URL, return as-is
    if pkg_spec.startswith("git+"):
        return pkg_spec

    # GitHub release assets and other direct wheel/archive URLs are installable
    # artifacts, not Git repositories.
    if _is_direct_url_artifact(pkg_spec):
        return pkg_spec

    # GitHub or GitLab URLs need git+ prefix
    if "github.com" in pkg_spec or "gitlab.com" in pkg_spec:
        return f"git+{pkg_spec}"

    # Everything else (local paths, PyPI specs) passes through unchanged
    return pkg_spec


def _strip_package_extras(package_name: str) -> str:
    """Return the base distribution name from a package name with optional extras."""
    return package_name.split("[", 1)[0].strip()


def _normalize_distribution_name(package_name: str) -> str:
    """Return a PEP 503-normalized distribution name."""
    return re.sub(r"[-_.]+", "-", package_name).lower()


def _normalize_install_target_name(package_name: str) -> str | None:
    """Return the canonical distribution name used by uv package flags."""
    base_name = _strip_package_extras(package_name)
    return _normalize_distribution_name(base_name) if base_name else None


def _infer_wheel_distribution_name(filename: str) -> str | None:
    """Infer the distribution name from a wheel filename."""
    if not filename.lower().endswith(".whl"):
        return None
    distribution = filename[:-4].split("-", 1)[0]
    return _normalize_distribution_name(distribution) if distribution else None


def _infer_install_target_name_from_normalized(package: str) -> str | None:
    """Infer a distribution name from an already-normalized package spec."""
    normalized = package

    # PEP 508 direct references: "name @ URL"
    if " @ " in normalized:
        return _normalize_install_target_name(normalized.split(" @ ", 1)[0].strip())

    # Direct URLs and local paths: prefer an explicit egg fragment, otherwise
    # fall back to the trailing path component (e.g., transformers.git -> transformers).
    if normalized.startswith("git+") or "://" in normalized or normalized.startswith("/"):
        match = re.search(r"(?:[#&]egg=)([A-Za-z0-9_.-]+)", normalized)
        if match:
            return _normalize_install_target_name(match.group(1))

        path_part = normalized.split("#", 1)[0].rsplit("/", 1)[-1]
        if "@" in path_part:
            path_part = path_part.split("@", 1)[0]
        wheel_distribution = _infer_wheel_distribution_name(path_part)
        if wheel_distribution:
            return wheel_distribution
        if path_part.endswith(".git"):
            path_part = path_part[:-4]
        return _normalize_install_target_name(path_part)

    match = re.match(r"^\s*([A-Za-z0-9_.-]+)", normalized)
    return _normalize_install_target_name(match.group(1)) if match else None


def bind_direct_source_to_package(package: str) -> str:
    """Attach a direct source URL/path to an inferred package name when possible.

    uv's resolver pins URL dependencies when the URL is declared for a specific
    package (for example, ``transformers @ git+https://...``). Provider overrides
    are intended to replace an existing package, so we make that association
    explicit for direct-source specs.
    """
    normalized = normalize_provider_package(package)
    if " @ " in normalized or not is_direct_source_package(normalized):
        return normalized

    package_name = _infer_install_target_name_from_normalized(normalized)
    if package_name:
        return f"{package_name} @ {normalized}"
    return normalized


def infer_install_target_name(package: str) -> str | None:
    """Infer the distribution name for a package specifier.

    This is used to target uv's package-specific reinstall flags when a runtime
    dependency should override an already-installed distribution.

    Args:
        package: Package specifier with or without install flags.

    Returns:
        Best-effort inferred distribution name, or None if it cannot be derived.
    """
    pkg_spec, _ = parse_install_spec(package)
    normalized = bind_direct_source_to_package(pkg_spec)
    return _infer_install_target_name_from_normalized(normalized)


def is_direct_source_package(package: str) -> bool:
    """Whether a package spec resolves from a direct source instead of an index."""
    pkg_spec, _ = parse_install_spec(package)
    normalized = normalize_provider_package(pkg_spec)
    return (
        normalized.startswith("git+")
        or " @ " in normalized
        or "://" in normalized
        or normalized.startswith("/")
    )


def build_install_command(
    package: str,
    constraints: str | None = None,
    venv_path: str | None = None,
    force_reinstall: bool = False,
) -> str:
    """Build a uv pip install command for a package with optional flags.

    Handles package specifiers that include install flags like --no-build-isolation.

    Args:
        package: Package specifier with optional install flags.
        constraints: Optional constraints file path.
        venv_path: Optional virtualenv path to target via `uv pip --python`.
        force_reinstall: Whether to force a provider override to replace an
            existing installed distribution in the target environment.

    Returns:
        Complete uv pip install command string.
    """
    pkg_spec, flags = parse_install_spec(package)
    normalized = (
        bind_direct_source_to_package(pkg_spec)
        if force_reinstall
        else normalize_provider_package(pkg_spec)
    )

    cmd_parts = ["uv"]
    if force_reinstall and is_direct_source_package(normalized):
        # Provider source overrides should not inherit the repo's uv config or
        # a shared cache entry that can silently re-lay an old wheel.
        cmd_parts.extend(["--no-config", "--no-cache"])
    cmd_parts.extend(["pip", "install"])
    if venv_path:
        cmd_parts.extend(["--python", f"{venv_path}/bin/python"])
    reinstall_flags = {
        "--force-reinstall",
        "--refresh",
        "--refresh-package",
        "--reinstall",
        "--reinstall-package",
    }
    if force_reinstall and not any(flag in reinstall_flags for flag in flags):
        if is_direct_source_package(normalized):
            cmd_parts.append("--refresh")
        package_name = infer_install_target_name(normalized)
        if package_name:
            cmd_parts.extend(["--refresh-package", package_name])
            cmd_parts.extend(["--reinstall-package", package_name])
        else:
            cmd_parts.extend(["--refresh", "--reinstall"])
    cmd_parts.extend(flags)
    cmd_parts.append(f"'{normalized}'")
    if constraints:
        cmd_parts.extend(["-c", constraints])

    return " ".join(cmd_parts)


def _parse_timeout(timeout: str) -> int:
    """Parse timeout string to nanoseconds.

    Supports formats: "24h", "30m", "1h30m", "90s".

    Args:
        timeout: Timeout string.

    Returns:
        Timeout in nanoseconds.
    """
    total_ns = 0
    patterns = [
        (r"(\d+)h", 3600_000_000_000),
        (r"(\d+)m", 60_000_000_000),
        (r"(\d+)s", 1_000_000_000),
    ]

    for pattern, multiplier in patterns:
        match = re.search(pattern, timeout)
        if match:
            total_ns += int(match.group(1)) * multiplier

    return total_ns if total_ns else 86400_000_000_000  # Default 24h


_MIN_RUNTIME_PATTERN = re.compile(r"(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?")


def validate_min_runtime(min_runtime: str) -> str:
    """Validate a Beaker min runtime string such as "2h", "30m", or "1h30m".

    Units are required and the duration must be positive. Beaker checks the allowed range
    when the experiment is created.

    Args:
        min_runtime: Duration string using h, m, and s units.

    Returns:
        The duration string with surrounding whitespace removed.

    Raises:
        ValueError: If the string is not a positive duration in h, m, and s units.
    """
    value = min_runtime.strip()
    match = _MIN_RUNTIME_PATTERN.fullmatch(value)
    if not value or match is None:
        raise ValueError(
            f"invalid min runtime {min_runtime!r}; expected a duration like '2h', '30m', or '1h30m'"
        )
    if not any(int(part) for part in match.groups() if part):
        raise ValueError(
            f"min runtime {min_runtime!r} must be greater than zero; "
            "omit it to allow preemption at any time"
        )
    return value


def _format_duration(seconds: int) -> str:
    """Format a duration in seconds as h/m/s units, such as "1h30m"."""
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    parts = [f"{n}{unit}" for n, unit in ((hours, "h"), (minutes, "m"), (secs, "s")) if n]
    return "".join(parts) or "0s"


class BeakerLauncher:
    """Launches evaluation jobs on Beaker using beaker-gantry.

    This class provides a clean API for submitting Beaker experiments
    using gantry's Python API for reproducible, git-tracked experiments.

    Example:
        launcher = BeakerLauncher()

        # Launch the job
        experiment = launcher.launch(config)
        print(f"Experiment: {launcher.experiment_url(experiment)}")

        # Or do a dry run
        launcher.launch(config, dry_run=True)

    Attributes:
        beaker: Lazy-initialized Beaker client (for group operations).
    """

    def __init__(self, workspace: str | None = None) -> None:
        """Initialize the launcher.

        Args:
            workspace: Beaker workspace for the launcher. Defaults to BEAKER_DEFAULT_WORKSPACE
                for read-only operations (results, watch, group info). When launching jobs,
                callers should always provide an explicit workspace.
        """
        self._workspace = workspace or BEAKER_DEFAULT_WORKSPACE
        self._beaker: Beaker | None = None

    @property
    def beaker(self) -> Beaker:
        """Lazy-initialized Beaker client."""
        if self._beaker is None:
            from beaker import Beaker

            self._beaker = Beaker.from_env(default_workspace=self._workspace)
        return self._beaker

    def _default_github_token_secret(self) -> str:
        """Return the user-scoped GitHub token secret name for Gantry clones."""
        return f"{self.beaker.user_name}_GITHUB_TOKEN"

    def beaker_token_secret(self, workspace: str, dry_run: bool) -> str | None:
        """Return the user's BEAKER_TOKEN secret if it exists in ``workspace``.

        The secret is never created: it holds a personal Beaker token, and it only
        enables cosmetic status updates. A dry run does not query Beaker.
        """
        from olmo_eval.launch.beaker.secrets import beaker_token_secret_name, secret_exists

        name = beaker_token_secret_name(self.beaker.user_name)
        enable_hint = (
            f"To enable them, create the secret: beaker secret write --workspace {workspace} "
            f"{name} <your Beaker token>"
        )
        if dry_run:
            _console.print(
                f"[dim]BEAKER_TOKEN will be injected from secret {name} if it exists in "
                f"{workspace}; otherwise job status updates in the Beaker UI are disabled. "
                "Not checked in a dry run.[/dim]"
            )
            return None
        try:
            exists = secret_exists(
                self.beaker, name, workspace if workspace != self._workspace else None
            )
        except Exception as e:
            log.debug(f"Could not check for Beaker secret {name}: {e}")
            exists = False
        if exists:
            return name
        _console.print(
            f"[yellow]Warning:[/yellow] Secret {name} not found in {workspace}, so job status "
            f"updates in the Beaker UI are disabled. {enable_hint}"
        )
        return None

    def _build_install_cmd(
        self,
        extras: list[str],
        env_exports: dict[str, str] | None = None,
        provider_packages: list[str] | None = None,
        task_packages: list[str] | None = None,
        setup_registry_mirror: bool = False,
        enable_sandbox: bool = False,
        vllm_isolated_venv: bool = False,
        setup_modal_gcp_secret: bool = False,
    ) -> str:
        """Build installation command for gantry's install_cmd parameter.

        Gantry clones the source code to /gantry-runtime, so we:
        1. Install olmo-eval from the cloned source with optional extras
        2. Optionally install provider-specific and task-specific dependencies

        When vllm_isolated_venv is True, vLLM is installed in a isolated venv
        (/opt/vllm-venv) to avoid dependency conflicts. The vLLM server runs as
        a subprocess using VLLM_PYTHON env var, while the main app uses /opt/venv.

        Args:
            extras: Optional dependency group names from pyproject.toml.
            env_exports: Optional dict of environment variables to export before running.
            provider_packages: Optional list of provider-specific dependencies.
            task_packages: Optional list of task-specific packages to install.
            setup_registry_mirror: If True, run setup_dockerio_mirror script with MIRROR_HOSTS.
            enable_sandbox: If True, set up /dev/net/tun and Artifact Registry auth.
            vllm_isolated_venv: If True, install vLLM in isolated venv for server mode.
            setup_modal_gcp_secret: If True, run setup_modal_gcp_secret to create Modal secret.

        Returns:
            Shell command string for installation.
        """
        has_vllm = "vllm" in extras
        use_isolated_vllm_venv = vllm_isolated_venv and (has_vllm or bool(provider_packages))
        vllm_venv: str | None = None

        # Build the install steps
        # Export UV_PROJECT_ENVIRONMENT so all uv commands use Docker's /opt/venv
        steps = ["export UV_PROJECT_ENVIRONMENT=/opt/venv"]

        # Set up /dev/net/tun for pasta networking (sandbox jobs)
        if enable_sandbox:
            steps.append(
                "mkdir -p /dev/net && "
                "[ -e /dev/net/tun ] || mknod /dev/net/tun c 10 200 && "
                "chmod 666 /dev/net/tun"
            )

        # Set up registry mirror for Docker Hub if configured (for sandbox jobs)
        if setup_registry_mirror:
            script = "/gantry-runtime/src/olmo_eval/launch/beaker/podman/setup_dockerio_mirror"
            steps.append(f'if [ -n "$MIRROR_HOSTS" ]; then {script} "$MIRROR_HOSTS"; fi')

        # Set up Artifact Registry auth for sandbox image caching
        # Skips itself unless GOOGLE_APPLICATION_CREDENTIALS holds a service account key
        if enable_sandbox:
            script = "/gantry-runtime/src/olmo_eval/launch/beaker/scripts/setup_artifact_registry"
            steps.append(f"source {script}")

        # Export additional environment variables (e.g., UV_CACHE_DIR)
        if env_exports:
            for key, value in env_exports.items():
                steps.append(f"export {key}={value}")

        runtime_torch_version = (env_exports or {}).get("OLMO_EVAL_RUNTIME_TORCH_VERSION")
        runtime_torch_index_url = (env_exports or {}).get("OLMO_EVAL_RUNTIME_TORCH_INDEX_URL")

        def runtime_torch_spec() -> str:
            spec = f"torch=={runtime_torch_version}"
            if runtime_torch_index_url:
                spec = f"{spec} --index-url {runtime_torch_index_url}"
            return spec

        def provider_package_spec(package: str) -> str:
            if not runtime_torch_index_url:
                return package

            _, flags = parse_install_spec(package)
            extra_flags: list[str] = []
            if "--index-url" not in flags and "--extra-index-url" not in flags:
                extra_flags.extend(["--extra-index-url", runtime_torch_index_url])
            if "--index-strategy" not in flags:
                extra_flags.extend(["--index-strategy", "unsafe-best-match"])
            return " ".join([package, *extra_flags]) if extra_flags else package

        # Optional runtime Torch override for provider forks that require a
        # different Torch build than the base image. This must happen before
        # constraints are generated so provider installs resolve against the
        # replacement Torch/CUDA package set. In isolated vLLM mode, the
        # overridden packages are symlinked into /opt/vllm-venv below.
        if runtime_torch_version:
            steps.append(build_install_command(runtime_torch_spec(), force_reinstall=True))

        # Generate constraints from pre-installed CUDA packages to prevent uv from changing them.
        # CUTLASS DSL is the exception: vLLM pins this package independently of
        # Torch's CUDA package set, so constraining an image's older build makes
        # an otherwise compatible vLLM impossible to resolve.
        constraints = "/tmp/cuda-constraints.txt"
        steps.append(
            # Match torch itself, but not companion packages such as
            # torchvision or torchaudio. Provider releases pin those companions
            # alongside their Torch generation and must be allowed to replace
            # the versions inherited from the base image.
            f"uv pip freeze -q | grep -E '^(torch(==| @ )|nvidia-)' "
            f"| grep -vE '^nvidia-cutlass-dsl' > {constraints}"
        )

        # Install vLLM in isolated venv when requested (for server mode)
        if use_isolated_vllm_venv:
            vllm_venv = "/opt/vllm-venv"
            steps.append(f"uv venv {vllm_venv}")
            # Symlink torch and nvidia packages from main venv (already installed)
            steps.append(
                f"for pkg in /opt/venv/lib/python*/site-packages/torch* "
                f"/opt/venv/lib/python*/site-packages/nvidia*; do "
                f'ln -sf "$pkg" {vllm_venv}/lib/python*/site-packages/; done'
            )
            # Install the project's bundled vLLM dependency set when requested.
            # Custom provider packages (e.g., a vLLM fork) are installed below
            # into the same isolated environment.
            if has_vllm:
                # Build a lockfile constraint set scoped to vLLM. Keep torch/CUDA
                # out so vLLM can pick its required build, and drop direct refs
                # that uv exports as commit-pinned URLs.
                vllm_lock_constraints = "/tmp/vllm-lock-constraints.txt"
                steps.append(
                    f"cd /gantry-runtime && uv export --extra vllm "
                    f"--no-default-groups --no-hashes --no-emit-project "
                    f"--no-header --no-annotate --frozen 2>/dev/null "
                    f"| grep -vE '^(torch|nvidia-)' | grep -vE ' @ ' "
                    f"> {vllm_lock_constraints}"
                )
                steps.append(
                    f"cd /gantry-runtime && uv pip install "
                    f"--python {vllm_venv}/bin/python "
                    f'--cache-dir "$UV_CACHE_DIR" '
                    f"-c {vllm_lock_constraints} -e '.[vllm]'"
                )
            # Set VLLM_PYTHON so VLLMServerProcess uses the isolated venv
            steps.append(f"export VLLM_PYTHON={vllm_venv}/bin/python")

        # Install main package (without vllm extra only when using isolated venv)
        main_extras = [e for e in extras if e != "vllm"] if use_isolated_vllm_venv else list(extras)

        # vllm_server mode needs HuggingFace tokenizer for accurate logprobs boundary
        if use_isolated_vllm_venv and "hf" not in main_extras:
            main_extras.append("hf")

        # Always install the beaker extra so BeakerStatusReporter can push progress
        # updates to the workload description.
        if "beaker" not in main_extras:
            main_extras.append("beaker")
        if main_extras:
            extras_str = ",".join(main_extras)
            install_cmd = f"uv pip install -e '.[{extras_str}]' -c {constraints}"
            steps.append(f"cd /gantry-runtime && {install_cmd}")
        else:
            steps.append(f"cd /gantry-runtime && uv pip install -e . -c {constraints}")

        # Install provider-specific dependencies
        if provider_packages:
            for pkg in provider_packages:
                steps.append(
                    build_install_command(
                        provider_package_spec(pkg),
                        constraints,
                        venv_path=vllm_venv if use_isolated_vllm_venv else None,
                        force_reinstall=True,
                    )
                )

        # Install task-specific dependencies
        if task_packages:
            for pkg in task_packages:
                steps.append(build_install_command(pkg, constraints))

        # Set up Modal secret for GCP Artifact Registry (Modal sandboxes)
        if setup_modal_gcp_secret:
            script = "/gantry-runtime/src/olmo_eval/launch/beaker/scripts/setup_modal_gcp_secret"
            steps.append(f"source {script}")

        return " && ".join(steps)

    def launch(self, config: BeakerJobConfig, dry_run: bool = False) -> BeakerExperiment | None:
        """Launch an experiment on Beaker using gantry.

        Args:
            config: Job configuration.
            dry_run: If True, print spec and exit without launching.

        Returns:
            Experiment object if launched, None if dry_run.
        """
        from gantry.api import launch_experiment

        clusters = resolve_clusters(config.cluster)

        # Build env vars that need to be exported in the install command (before uv runs)
        env_exports: dict[str, str] = {}
        install_env_keys = (
            "UV_CACHE_DIR",
            "OLMO_EVAL_RUNTIME_TORCH_VERSION",
            "OLMO_EVAL_RUNTIME_TORCH_INDEX_URL",
        )
        for key in install_env_keys:
            if key in config.env_vars:
                env_exports[key] = config.env_vars[key]

        # Build separate install command for gantry's install_cmd parameter
        install_cmd = self._build_install_cmd(
            config.extras,
            env_exports,
            config.provider_packages,
            config.task_packages,
            config.setup_registry_mirror,
            config.enable_sandbox,
            config.vllm_isolated_venv,
            config.setup_modal_gcp_secret,
        )

        # Build weka mounts as tuples: (bucket, mount_path)
        weka_mounts: list[tuple[str, str]] = []
        for bucket in config.weka_buckets:
            assert bucket.mount is not None  # Set by __post_init__
            weka_mounts.append((bucket.bucket, bucket.mount))

        # Build env secrets as tuples: (env_var_name, secret_name)
        env_secrets: list[tuple[str, str]] = [
            (secret.name, secret.secret) for secret in config.env_secrets
        ]

        # BEAKER_TOKEN lets the running job post status updates to the workload
        # description via BeakerStatusReporter.
        if not any(name == "BEAKER_TOKEN" for name, _ in env_secrets):
            token_secret = self.beaker_token_secret(config.workspace, dry_run)
            if token_secret:
                env_secrets.append(("BEAKER_TOKEN", token_secret))

        # Inject AWS credentials if requested
        if config.inject_aws_credentials:
            from olmo_eval.launch.beaker.aws import ensure_aws_secrets

            aws_secrets = ensure_aws_secrets(config.workspace)
            env_secrets.extend(aws_secrets)
            log.info("Injecting AWS credentials for S3 access")

        # Inject GCS credentials if requested
        google_credentials_secret: str | None = None
        if config.inject_gcs_credentials and dry_run:
            # A dry run shows the secret name without writing credentials to Beaker.
            google_credentials_secret = f"{self.beaker.user_name}_GOOGLE_CREDENTIALS"
        elif config.inject_gcs_credentials:
            from olmo_eval.launch.beaker.gcs import ensure_gcs_secrets

            google_credentials_secret = ensure_gcs_secrets(config.workspace)
            log.info("Injecting GCS credentials for GCS access")

        # Inject the uploader key for results upload. A dry run shows the secret name
        # without reading the key or writing it to Beaker.
        if config.inject_upload_credentials:
            from olmo_eval.launch.beaker.uploader import (
                UPLOADER_SECRET_NAME,
                ensure_uploader_secret,
            )
            from olmo_eval.upload.config import UPLOAD_CREDENTIALS_ENV

            secret = UPLOADER_SECRET_NAME if dry_run else ensure_uploader_secret(config.workspace)
            env_secrets.append((UPLOAD_CREDENTIALS_ENV, secret))

        # Build env vars as tuples: (name, value)
        env_vars: list[tuple[str, str]] = list(config.env_vars.items())

        if config.enable_sandbox:
            log.info("Enabling sandbox mode with Podman subcontainers")

        # Build mounts for NFS if requested
        mounts: list[tuple[str, str]] | None = None
        if config.nfs:
            mounts = [("/net/nfs.cirrascale", "/net/nfs.cirrascale")]

        # Build group names list if groups are specified
        group_names: list[str] | None = config.groups if config.groups else None

        # Print config summary for dry run
        if dry_run:
            self._print_dry_run_config(config, clusters)

        # Launch the experiment (or show spec if dry_run)
        workload = launch_experiment(
            args=config.command,
            name=config.name,
            description=config.description,
            workspace=config.workspace,
            group_names=group_names,
            clusters=clusters,
            gpus=config.num_gpus,
            shared_memory=config.shared_memory,
            priority=config.priority,
            preemptible=config.preemptible,
            min_runtime=config.min_runtime,
            task_timeout=config.timeout,
            retries=config.retries,
            budget=config.budget,
            beaker_image=config.beaker_image,
            weka=weka_mounts if weka_mounts else None,
            gh_token_secret=self._default_github_token_secret(),
            env_vars=env_vars if env_vars else None,
            env_secrets=env_secrets if env_secrets else None,
            google_credentials_secret=google_credentials_secret,
            mounts=mounts,
            results=config.result_path,
            install=install_cmd,
            no_python=True,  # Use pre-built image, skip Python setup
            dry_run=dry_run,
            timeout=(
                99999999 if config.follow else 0
            ),  # only way to follow the experiment without canceling
            yes=True,  # Skip confirmation prompts
        )

        if dry_run:
            return None

        if workload is None:
            log.warning("Gantry returned None workload - experiment may not have been created")
            return None

        # Get the experiment from the workload
        # The workload contains an embedded experiment object from gantry
        experiment = workload.experiment
        log.info(f"Experiment submitted: {self.experiment_url(experiment)}")
        return experiment

    def _print_dry_run_config(self, config: BeakerJobConfig, clusters: list[str]) -> None:
        """Print a summary of the config for dry run mode."""
        # Build header with key metadata
        header_lines = []
        header_lines.append(f"[bold cyan]Experiment:[/] {config.name}")

        # Extract model and tasks from command
        model = None
        task_names = []
        for i, arg in enumerate(config.command):
            if arg == "-m" and i + 1 < len(config.command):
                model = config.command[i + 1]
            elif arg == "-t" and i + 1 < len(config.command):
                task_names.append(config.command[i + 1])

        if model:
            header_lines.append(f"[bold blue]Model:[/] {model}")
        if task_names:
            header_lines.append(f"[bold blue]Tasks:[/] {', '.join(task_names)}")

        # Resource info
        priority = config.priority
        priority_color = {
            "low": "dim",
            "normal": "white",
            "high": "yellow",
            "urgent": "red bold",
        }.get(priority, "white")
        header_lines.append(f"[bold blue]Priority:[/] [{priority_color}]{priority}[/]")
        header_lines.append(f"[bold blue]GPUs:[/] {config.num_gpus}")
        header_lines.append(f"[bold blue]Clusters:[/] {', '.join(clusters)}")
        header_lines.append(f"[bold blue]Image:[/] {config.beaker_image}")

        if config.min_runtime is not None:
            header_lines.append(f"[bold blue]Min runtime:[/] {config.min_runtime}")
        else:
            preempt_str = "[green]yes[/]" if config.preemptible else "[red]no[/]"
            header_lines.append(f"[bold blue]Preemptible:[/] {preempt_str}")

        header_text = Text.from_markup("\n".join(header_lines))
        _console.print(Panel(header_text, title="[bold]Beaker Experiment[/]", border_style="blue"))

    def experiment_url(self, experiment: BeakerExperiment) -> str:
        """Get the Beaker URL for an experiment.

        Args:
            experiment: The experiment object.

        Returns:
            URL to view the experiment in Beaker.
        """
        return self.beaker.experiment.url(experiment)

    # -------------------------------------------------------------------------
    # Group Management
    # -------------------------------------------------------------------------

    def get_or_create_group(
        self,
        name: str,
        workspace: str | None = None,
        description: str | None = None,
    ) -> BeakerGroup:
        """Get existing group or create a new one.

        Args:
            name: Group name (can be short name or user-qualified).
            workspace: Workspace for the group. Uses default if None.
            description: Optional description for new groups.

        Returns:
            The existing or newly created group.
        """
        from beaker.exceptions import BeakerGroupConflict, BeakerGroupNotFound

        # Try to get existing group - needs user-qualified name for lookup
        # If name doesn't contain "/", try with current user prefix
        qualified_name = f"{self.beaker.user_name}/{name}" if "/" not in name else name

        try:
            return self.beaker.group.get(qualified_name)
        except BeakerGroupNotFound:
            try:
                # Get workspace object for the API call
                ws_name = workspace or self._workspace
                ws_obj = self.beaker.workspace.get(ws_name) if ws_name else None
                return self.beaker.group.create(
                    name=name,  # Create uses short name
                    workspace=ws_obj,
                    description=description,
                )
            except BeakerGroupConflict:
                # Group was created between our get and create calls, fetch it
                return self.beaker.group.get(qualified_name)

    def add_experiments_to_group(
        self,
        group: str | BeakerGroup,
        experiment_ids: list[str],
    ) -> BeakerGroup:
        """Add experiments to a group.

        Args:
            group: Group name or object.
            experiment_ids: List of experiment IDs to add.

        Returns:
            Updated group object.
        """
        # Convert string to Group object if needed
        group_obj = self.beaker.group.get(group) if isinstance(group, str) else group
        return self.beaker.group.update(
            group_obj,
            add_experiment_ids=experiment_ids,
        )

    def _get_experiment_ids_from_group(self, group: str | BeakerGroup) -> set[str]:
        """Get unique experiment IDs from a group using task metrics.

        Args:
            group: Group name or object.

        Returns:
            Set of experiment IDs.
        """
        if isinstance(group, str):
            group = self.beaker.group.get(group)
        task_metrics = self.beaker.group.list_task_metrics(group)
        return {tm.experiment_id for tm in task_metrics}

    def get_group_status(self, group: str | BeakerGroup) -> dict[str, int]:
        """Get status summary for all experiments in a group.

        Args:
            group: Group name or object.

        Returns:
            Dictionary mapping status to count.
        """
        from beaker import BeakerWorkloadStatus

        experiment_ids = self._get_experiment_ids_from_group(group)
        status_counts = {
            "succeeded": 0,
            "failed": 0,
            "running": 0,
            "pending": 0,
            "canceled": 0,
        }

        for exp_id in experiment_ids:
            # Get the workload status
            workload = self.beaker.workload.get(exp_id)
            status = workload.status

            if status == BeakerWorkloadStatus.succeeded:
                status_counts["succeeded"] += 1
            elif status == BeakerWorkloadStatus.failed:
                status_counts["failed"] += 1
            elif status == BeakerWorkloadStatus.canceled:
                status_counts["canceled"] += 1
            elif status in (
                BeakerWorkloadStatus.running,
                BeakerWorkloadStatus.uploading_results,
            ):
                status_counts["running"] += 1
            else:
                # queued, initializing, submitted
                status_counts["pending"] += 1

        return status_counts

    def get_group_experiments(
        self,
        group: str | BeakerGroup,
    ) -> list[BeakerExperiment]:
        """Get all experiments in a group.

        Args:
            group: Group name or object.

        Returns:
            List of experiment objects.
        """
        experiment_ids = self._get_experiment_ids_from_group(group)
        experiments = []
        for exp_id in experiment_ids:
            workload = self.beaker.workload.get(exp_id)
            experiments.append(workload.experiment)
        return experiments

    def export_group_metrics(self, group: str | BeakerGroup) -> str:
        """Export group metrics as CSV.

        Args:
            group: Group name or object.

        Returns:
            CSV string with experiment metrics.
        """
        # Convert string to Group object if needed
        group_obj = self.beaker.group.get(group) if isinstance(group, str) else group
        return self.beaker.group.export_metrics(group_obj)

    def cancel_group(self, group: str | BeakerGroup) -> dict[str, int]:
        """Cancel all active experiments in a group.

        Stops all running and pending experiments in the group. Experiments that
        have already completed (succeeded, failed, or canceled) are skipped.

        Args:
            group: Group name or object.

        Returns:
            Dictionary with counts: {"canceled": N, "skipped": M, "failed": K}
        """
        from beaker import BeakerWorkloadStatus
        from beaker.exceptions import BeakerExperimentConflict, BeakerExperimentNotFound

        experiment_ids = self._get_experiment_ids_from_group(group)
        results = {"canceled": 0, "skipped": 0, "failed": 0}

        for exp_id in experiment_ids:
            # Check current status before attempting to cancel
            workload = self.beaker.workload.get(exp_id)
            status = workload.status

            # Skip already-completed experiments
            if status in (
                BeakerWorkloadStatus.succeeded,
                BeakerWorkloadStatus.failed,
                BeakerWorkloadStatus.canceled,
            ):
                results["skipped"] += 1
                continue

            # Attempt to cancel the workload
            try:
                list(self.beaker.workload.cancel(workload))
                results["canceled"] += 1
            except (BeakerExperimentNotFound, BeakerExperimentConflict):
                # Already completed or canceled while we were iterating
                results["skipped"] += 1
            except Exception:
                results["failed"] += 1

        return results

    def get_group_url(self, group: BeakerGroup) -> str:
        """Get the Beaker URL for a group.

        Args:
            group: The group object.

        Returns:
            URL to view the group in Beaker.
        """
        return f"https://beaker.org/gr/{group.id}"

    # -------------------------------------------------------------------------
    # Experiment Following / Watching
    # -------------------------------------------------------------------------

    def follow_experiment(
        self,
        experiment_id: str,
        tail: bool = False,
    ) -> int:
        """Follow an experiment's logs until completion.

        Streams logs in real-time, showing startup events and job output.

        Args:
            experiment_id: Beaker experiment ID.
            tail: If True, only show last 10 seconds of logs (useful for
                  attaching to already-running experiments).

        Returns:
            Exit code: 0 for success, non-zero for failure.
        """
        import time
        from datetime import datetime, timedelta

        # Get the workload (experiment)
        workload = self.beaker.workload.get(experiment_id)
        workload_url = self.beaker.workload.url(workload)

        # Phase 1: Wait for job creation
        job = self.beaker.workload.get_latest_job(workload)
        if job is None:
            _console.print("[dim]Waiting for job to be created...[/dim]")
            while job is None:
                time.sleep(1.0)
                workload = self.beaker.workload.get(experiment_id)
                job = self.beaker.workload.get_latest_job(workload)

        # Phase 2: Wait for job to start, showing events
        events_seen: set[str] = set()
        assert job is not None  # We waited for job creation above
        job = self.beaker.job.get(job.id)
        while not (job.status.finalized or job.status.started):
            for event in self.beaker.job.list_summarized_events(job):
                # Use latest_message as key since events don't have stable IDs
                event_key = f"{event.status}:{event.latest_message}"
                if event_key not in events_seen:
                    events_seen.add(event_key)
                    _console.print(f"[cyan]>[/cyan] {event.latest_message}")
            time.sleep(1.0)
            job = self.beaker.job.get(job.id)

        # Phase 3: Stream logs
        _console.print("\n[bold]Logs:[/bold]")
        try:
            since = datetime.now(UTC) - timedelta(seconds=10) if tail else None
            log_iter = iter(self.beaker.job.logs(job, follow=True, since=since))

            # Wait for first log entry with spinner
            first_entry = None
            with _console.status("[dim]Waiting for job to start...[/dim]"):
                first_entry = next(log_iter, None)

            # Print first entry if we got one
            if first_entry and first_entry.message:
                line = first_entry.message.decode(errors="ignore").rstrip("\n")
                if line:
                    print(line)

            # Continue with remaining logs
            for log_entry in log_iter:
                if log_entry.message:
                    line = log_entry.message.decode(errors="ignore").rstrip("\n")
                    if line:
                        print(line)
        except KeyboardInterrupt:
            _console.print("\n[yellow]Interrupted. Experiment continues running.[/yellow]")
            _console.print(f"[dim]View at: {workload_url}[/dim]")
            return 130  # Standard exit code for SIGINT

        # Phase 4: Check exit status
        _console.print()
        job = self.beaker.job.get(job.id)
        exit_code = job.status.exit_code

        if exit_code is None:
            _console.print(f"[red]Experiment failed[/red]: {workload_url}")
            return 1
        elif exit_code > 0:
            _console.print(f"[red]Experiment exited with code {exit_code}[/red]")
            return exit_code
        else:
            _console.print("[green]Experiment completed successfully[/green]")
            return 0

    def get_experiment_url(self, experiment_id: str) -> str:
        """Get the Beaker URL for an experiment by ID.

        Args:
            experiment_id: The experiment ID.

        Returns:
            URL to view the experiment in Beaker.
        """
        return f"https://beaker.org/ex/{experiment_id}"
