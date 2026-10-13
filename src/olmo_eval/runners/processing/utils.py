"""Common utility functions for ID generation, sanitization, and metadata extraction."""

from __future__ import annotations

import json
import uuid
from typing import Any

from olmo_eval.common.types import SamplingParams


def generate_experiment_id() -> str:
    """Generate a unique experiment ID.

    Returns a short 12-character hex string from UUID4. Since experiments
    are partitioned by model/task, collision risk is minimal.

    Returns:
        A 12-character hex string to uniquely identify an experiment.

    Example:
        >>> exp_id = generate_experiment_id()
        >>> len(exp_id)
        12
    """
    return uuid.uuid4().hex[:12]


def sanitize_spec_for_filename(spec: str) -> str:
    """Sanitize a task spec string to be safe for use in filenames.

    Replaces characters that are not allowed or problematic in filenames:
    - ':' -> '_' (used in task variants like arc_challenge:bpb)
    - '/' -> '_' (could appear in model names or paths)
    - '\\' -> '_' (Windows path separator)
    - ' ' -> '_' (spaces)

    Args:
        spec: Task specification string (e.g., "arc_challenge:bpb")

    Returns:
        Sanitized string safe for use in filenames (e.g., "arc_challenge_bpb")

    Examples:
        >>> sanitize_spec_for_filename("arc_challenge:bpb")
        'arc_challenge_bpb'
        >>> sanitize_spec_for_filename("mmlu/history")
        'mmlu_history'
    """
    # Replace problematic characters with underscores
    result = spec.replace(":", "_").replace("/", "_").replace("\\", "_").replace(" ", "_")
    return result


def serialize_sampling_params(params: SamplingParams | None) -> dict[str, Any] | None:
    """Serialize SamplingParams to a dictionary for JSON output.

    Args:
        params: SamplingParams instance or None

    Returns:
        Dictionary representation or None if params is None
    """
    if params is None:
        return None
    return {
        "temperature": params.temperature,
        "max_tokens": params.max_tokens,
        "top_p": params.top_p,
        "top_k": params.top_k,
        "num_samples": params.num_samples,
    }


def compute_task_hash(config: dict) -> str:
    """Compute a hash from task config.

    Args:
        config: Task configuration dictionary

    Returns:
        16-character hex string hash of the config
    """
    import hashlib

    config_str = json.dumps(config, sort_keys=True)
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]


def log_task_metrics(
    metrics: dict[str, dict[str, float]],
    task_spec: str,
    logger: Any,
    console: Any | None = None,
) -> None:
    """Log task metrics in a consistent format.

    Handles nested metrics structure: {metric_name: {scorer_name: value}}.

    Args:
        metrics: Nested metrics dictionary.
        task_spec: Task specification string for the log header.
        logger: Logger instance to use.
        console: Optional rich console for formatted output.
    """
    if not metrics:
        return

    logger.info(f"** Task metrics for {task_spec}: **")
    for metric_name, scorers in metrics.items():
        for scorer_name, value in scorers.items():
            line = f"  {metric_name}:{scorer_name}: {value:.4f}"
            logger.info(line)
            if console:
                console.print(line)


def make_metric_key(metric_name: str, scorer_name: str) -> str:
    """Create a metric key in "metric:scorer" format.

    Args:
        metric_name: The metric name (e.g., "accuracy").
        scorer_name: The scorer name (e.g., "exact_match").

    Returns:
        Combined key in "metric:scorer" format.
    """
    return f"{metric_name}:{scorer_name}"


def parse_metric_key(key: str) -> tuple[str, str] | None:
    """Parse a metric key in "metric:scorer" format.

    Args:
        key: The combined key (e.g., "accuracy:exact_match").

    Returns:
        Tuple of (metric_name, scorer_name), or None if invalid format.
    """
    if not key or ":" not in key:
        return None
    parts = key.split(":", 1)
    return (parts[0], parts[1])


def extract_score_from_metrics(
    metrics: dict[str, dict[str, float]] | None,
    primary_metric: str | None,
) -> float | None:
    """Extract a score from nested metrics using a primary_metric identifier.

    Args:
        metrics: Nested metrics dict {metric_name: {scorer_name: score}}.
        primary_metric: The metric identifier in "metric:scorer" format.

    Returns:
        The score value, or None if not found.
    """
    if not metrics or not primary_metric:
        return None
    parsed = parse_metric_key(primary_metric)
    if not parsed:
        return None
    metric_name, scorer_name = parsed
    if metric_name in metrics and scorer_name in metrics[metric_name]:
        return metrics[metric_name][scorer_name]
    return None


def get_primary_metric(
    metrics: dict[str, dict[str, float]],
    preferred: str | None = None,
) -> tuple[str, float] | None:
    """Get the primary metric identifier and value from a nested metrics dict.

    Metrics are in nested format: {metric_name: {scorer_name: score}}.
    The returned metric identifier uses "metric_name:scorer_name" format.

    Priority:
    1. User-specified preferred (in "metric:scorer" format) if present
    2. "accuracy" with first scorer if present
    3. First metric:scorer alphabetically (for determinism)

    Args:
        metrics: Nested dictionary {metric_name: {scorer_name: value}}
        preferred: Optional preferred metric in "metric:scorer" format

    Returns:
        Tuple of ("metric:scorer", value), or None if metrics is empty
    """
    if not metrics:
        return None

    # Use preferred if specified and present (format: "metric:scorer")
    if preferred:
        parsed = parse_metric_key(preferred)
        if parsed:
            metric_name, scorer_name = parsed
            if metric_name in metrics and scorer_name in metrics[metric_name]:
                return (preferred, metrics[metric_name][scorer_name])

    # Default fallback: accuracy first (with first scorer alphabetically)
    if "accuracy" in metrics:
        scorers = metrics["accuracy"]
        if scorers:
            scorer_name = sorted(scorers.keys())[0]
            return (f"accuracy:{scorer_name}", scorers[scorer_name])

    # Fallback: first metric:scorer alphabetically for determinism
    metric_name = sorted(metrics.keys())[0]
    scorers = metrics[metric_name]
    if scorers:
        scorer_name = sorted(scorers.keys())[0]
        return (f"{metric_name}:{scorer_name}", scorers[scorer_name])

    # No valid metrics found
    return None


def get_metric_metadata(task: Any) -> str | None:
    """Extract primary metric identifier from task config.

    Args:
        task: Task instance with config.metrics

    Returns:
        Primary metric in "metric_name:scorer_name" format, or None.
    """
    primary_metric = task.config.get_primary_metric()
    if primary_metric is None:
        return None

    # Get scorer name for the primary metric
    scorer_name = "default"
    if hasattr(primary_metric, "scorer") and primary_metric.scorer is not None:
        scorer_name = primary_metric.scorer().name

    return f"{primary_metric.name}:{scorer_name}"
