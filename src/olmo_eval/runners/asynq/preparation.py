"""Task preparation functions for async evaluation runners."""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from olmo_eval.common.logging import get_logger
from olmo_eval.common.types import Response, SamplingParams
from olmo_eval.evals.tasks.common import Task, get_task
from olmo_eval.runners.asynq.types import QueueItem, TaskTracker, summarize_failed_instances
from olmo_eval.runners.common.types import DEFAULT_MAX_HARD_FAILURE_RATE, TaskResult
from olmo_eval.runners.io.builders import build_predictions, build_requests_from_responses
from olmo_eval.runners.processing.generation_counts import count_generations
from olmo_eval.runners.processing.utils import get_metric_metadata

logger = get_logger(__name__)


def prepare_task_items(
    spec: str,
    model_name: str,
    overrides: dict[str, Any] | None,
    sampling_overrides: dict[str, Any] | None = None,
) -> tuple[Task, list[QueueItem]]:
    """Prepare a task and its queue items.

    Args:
        spec: Task specification string
        model_name: Model name this task is for
        overrides: Optional config overrides (num_fewshot, limit, fewshot_seed)
        sampling_overrides: Optional overrides for sampling params (temperature, max_tokens, etc.)

    Returns:
        Tuple of (Task instance for scoring, list of QueueItems)

    """
    task = get_task(spec)

    if overrides:
        task.config = replace(task.config, **overrides)

    # Build sampling params from overrides
    existing_params = task.config.sampling_params or SamplingParams()

    # Apply sampling_overrides
    if sampling_overrides:
        for key, value in sampling_overrides.items():
            if hasattr(existing_params, key):
                existing_params = replace(existing_params, **{key: value})

    # Always update task config with final sampling params (so finalize_task captures them)
    task.config = replace(task.config, sampling_params=existing_params)

    instances = list(task.instances)
    if task.config.limit and len(instances) > task.config.limit:
        # Use random.sample for reproducible random sampling (matches oe-eval-internal behavior)
        rng = random.Random(task.config.seed)
        instances = rng.sample(instances, task.config.limit)

    items = [
        QueueItem(
            model_name=model_name,
            task_id=spec,
            instance_idx=idx,
            instance=inst,
            request=task.format_request(inst),
            sampling_params=task.get_sampling_params(inst) or existing_params,
        )
        for idx, inst in enumerate(instances)
    ]

    return task, items


def build_requests_from_items(items: list[QueueItem], task_name: str) -> list[dict]:
    """Build request objects from queue items for early writing.

    Args:
        items: List of QueueItems (with instance, request, sampling_params)
        task_name: Name of the task

    Returns:
        List of request dicts suitable for JSONL output
    """
    from olmo_eval.runners.io.builders import build_requests

    instances = [item.instance for item in items]
    requests = [item.request for item in items]
    sampling_params = items[0].sampling_params if items else None

    return build_requests(instances, requests, task_name, sampling_params)


async def finalize_task(tracker: TaskTracker) -> TaskResult:
    """Finalize a task tracker into a TaskResult.

    Only the task-level error path is exercised by the async runner, which calls
    this for trackers that failed before any instance ran; every task that
    actually dispatched instances goes through compute_task_metrics instead. The
    hard-failure gate therefore lives in compute_task_metrics and not here.

    Args:
        tracker: Completed TaskTracker

    Returns:
        TaskResult with metrics and predictions
    """
    import time

    duration = time.time() - tracker.start_time

    # Task-level error (e.g., prep failed) - no results possible. Instance
    # accounting stays at zero: nothing was dispatched, so nothing was processed.
    if tracker.error:
        return TaskResult(
            spec=tracker.spec,
            config={},
            num_instances=0,
            metrics={},
            error=tracker.error,
            duration_seconds=duration,
            error_summary=tracker.error,
        )

    if tracker.task is None:
        return TaskResult(
            spec=tracker.spec,
            config={},
            num_instances=0,
            metrics={},
            error="Task preparation failed",
            duration_seconds=duration,
            error_summary="Task preparation failed",
        )

    error_summary = summarize_failed_instances(tracker.failed_instances)

    # Check if we have any successful responses
    if not tracker.responses:
        # All instances failed
        summary = error_summary or "All instances failed"
        return TaskResult(
            spec=tracker.spec,
            config=tracker.task.config.to_dict(),
            num_instances=tracker.total_instances,
            metrics={},
            error=summary,
            duration_seconds=duration,
            error_summary=summary,
        )

    # Sort responses by index (only successful ones)
    responses = [tracker.responses[i] for i in sorted(tracker.responses.keys())]

    # Score and compute metrics
    tracker.task.strip_thinking_traces(responses)
    scored = await tracker.task.score_responses(responses)
    metrics = tracker.task.compute_metrics(scored)
    infrastructure_failures = _count_infrastructure_scoring_errors(scored)

    # Build predictions
    predictions = build_predictions(scored, metrics=tracker.task.config.metrics)
    requests = build_requests_from_responses(scored, tracker.task.config.name)

    # Get task config for serialization
    task_config = tracker.task.config

    # Extract metric metadata (returns "metric:scorer" format)
    primary_metric = get_metric_metadata(tracker.task)

    # Add warning about failed instances if any
    if error_summary:
        # Log failed instances but still return partial results
        logger.warning(
            f"Task {tracker.spec} completed with failures: {error_summary}. "
            f"Computed metrics on {len(responses)}/{tracker.total_instances} instances."
        )

    return TaskResult(
        spec=tracker.spec,
        config=task_config.to_dict(),
        num_instances=len(responses),
        metrics=metrics,
        duration_seconds=duration,
        predictions=predictions,
        requests=requests,
        primary_metric=primary_metric,
        error=_infrastructure_error(infrastructure_failures),
        error_summary=error_summary,
        instances_processed=tracker.total_instances,
        instances_failed=len(tracker.failed_instances),
        generation_counts=count_generations(scored),
    )


def compute_task_metrics(
    spec: str,
    task: Task,
    scored_responses: list[Response],
    failed_instances: dict[int, str],
    total_instances: int,
    duration_seconds: float,
    max_hard_failure_rate: float = DEFAULT_MAX_HARD_FAILURE_RATE,
) -> TaskResult:
    """Compute metrics from pre-scored responses.

    Args:
        spec: Task specification string.
        task: Task instance for metric computation.
        scored_responses: List of already-scored responses.
        failed_instances: Dict of instance_idx -> error message for failures.
        total_instances: Total number of instances in the task.
        duration_seconds: Duration of the task.
        max_hard_failure_rate: Fraction of instances allowed to hard-fail before the
            task is marked failed.

    Returns:
        TaskResult with metrics and predictions.
    """
    instances_failed = len(failed_instances)
    error_summary = summarize_failed_instances(failed_instances)

    if not scored_responses:
        summary = error_summary or "No responses"
        gate_error = hard_failure_gate_error(
            instances_saved=0,
            instances_failed=instances_failed,
            instances_processed=total_instances,
            first_error=_first_instance_error(failed_instances),
            max_hard_failure_rate=max_hard_failure_rate,
        )
        return TaskResult(
            spec=spec,
            config=task.config.to_dict(),
            num_instances=0,
            metrics={},
            error=gate_error or summary,
            duration_seconds=duration_seconds,
            error_summary=summary,
            instances_processed=total_instances,
            instances_failed=instances_failed,
            hard_failure_rate_exceeded=gate_error is not None,
        )

    # Compute metrics from pre-scored responses
    metrics = task.compute_metrics(scored_responses)
    infrastructure_failures = _count_infrastructure_scoring_errors(scored_responses)

    # Build predictions
    predictions = build_predictions(scored_responses, metrics=task.config.metrics)
    requests = build_requests_from_responses(scored_responses, task.config.name)

    # Extract metric metadata
    primary_metric = get_metric_metadata(task)

    # Add warning about failed instances if any
    if error_summary:
        logger.warning(
            f"Task {spec} completed with failures: {error_summary}. "
            f"Computed metrics on {len(scored_responses)}/{total_instances} instances."
        )

    gate_error = hard_failure_gate_error(
        instances_saved=len(scored_responses),
        instances_failed=instances_failed,
        instances_processed=total_instances,
        first_error=_first_instance_error(failed_instances),
        max_hard_failure_rate=max_hard_failure_rate,
    )

    return TaskResult(
        spec=spec,
        config=task.config.to_dict(),
        num_instances=len(scored_responses),
        metrics=metrics,
        duration_seconds=duration_seconds,
        predictions=predictions,
        requests=requests,
        primary_metric=primary_metric,
        error=_combine_errors(gate_error, _infrastructure_error(infrastructure_failures)),
        error_summary=error_summary,
        instances_processed=total_instances,
        instances_failed=instances_failed,
        hard_failure_rate_exceeded=gate_error is not None,
        generation_counts=count_generations(scored_responses),
    )


def hard_failure_gate_error(
    *,
    instances_saved: int,
    instances_failed: int,
    instances_processed: int,
    first_error: str | None,
    max_hard_failure_rate: float,
) -> str | None:
    """Describe a breach of the hard-failure-rate budget, if there is one.

    A hard failure is an instance that came back with an error and no outputs at all,
    so it contributes nothing to the metrics. Soft failures (an error alongside a
    usable output, e.g. MaxTurnsExceeded with a fallback answer) are scored upstream
    and never reach this function.

    Two separate conditions fail a task. A task that saved no instances at all
    produced no metrics, and there is no budget at which that is a result worth
    publishing, so it fails even when the threshold has been opened all the way up
    to 1.0. Otherwise the hard-failure rate is compared against the budget, and
    that comparison is strict: a task sitting exactly on the threshold passes.

    Args:
        instances_saved: Instances that were scored and saved.
        instances_failed: Instances that hard-failed.
        instances_processed: Instances the runner saw (saved plus hard-failed).
        first_error: First underlying instance error, quoted if present so the
            message says why the instances failed and not just how many.
        max_hard_failure_rate: Maximum tolerated fraction of hard failures.

    Returns:
        An error string describing the breach, or None when within budget.
    """
    if instances_processed <= 0:
        return None

    detail = f" First failure: {first_error}" if first_error else ""

    if instances_saved <= 0:
        return (
            f"No instances were saved out of {instances_processed}: the task produced "
            f"no metrics ({instances_failed} hard-failed), which fails at any hard "
            f"failure rate budget.{detail}"
        )

    if instances_failed <= 0:
        return None
    rate = instances_failed / instances_processed
    if rate <= max_hard_failure_rate:
        return None
    return (
        f"Hard failure rate {rate:.1%} exceeds the maximum of {max_hard_failure_rate:.1%}: "
        f"saved {instances_saved} of {instances_processed} instances, "
        f"{instances_failed} hard-failed.{detail}"
    )


def _first_instance_error(failed_instances: dict[int, str]) -> str | None:
    """The first recorded instance error, used to make gate messages actionable."""
    return next(iter(failed_instances.values()), None)


def _infrastructure_error(infrastructure_failures: int) -> str | None:
    """Describe output scores lost to sandbox infrastructure, if any."""
    if not infrastructure_failures:
        return None
    return (
        f"Incomplete: {infrastructure_failures} output score(s) failed due to "
        "sandbox infrastructure"
    )


def _combine_errors(*errors: str | None) -> str | None:
    """Join the task-level errors that are set, preserving order."""
    present = [error for error in errors if error]
    return " ".join(present) if present else None


def _count_infrastructure_scoring_errors(responses: Sequence[Response]) -> int:
    """Count output scores that failed specifically because of sandbox infrastructure."""
    count = 0
    for response in responses:
        for output in response.outputs:
            errors = output.metadata.get("scoring_errors")
            if not isinstance(errors, dict):
                continue
            count += sum(
                1
                for error in errors.values()
                if isinstance(error, dict) and error.get("infrastructure") == "true"
            )
    return count


__all__ = [
    "prepare_task_items",
    "build_requests_from_items",
    "finalize_task",
    "compute_task_metrics",
    "hard_failure_gate_error",
]
