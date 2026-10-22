"""Task preparation functions for async evaluation runners."""

from __future__ import annotations

import random
from dataclasses import replace
from typing import Any

from olmo_eval.common.logging import get_logger
from olmo_eval.common.scoring_failures import (
    ScoringFailures,
    count_scoring_failures,
    measurable_responses,
)
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


async def finalize_task(
    tracker: TaskTracker, max_hard_failure_rate: float = DEFAULT_MAX_HARD_FAILURE_RATE
) -> TaskResult:
    """Finalize a task tracker into a TaskResult.

    Only the task-level error path is exercised by the async runner, which calls
    this for trackers that failed before any instance ran; every task that
    actually dispatched instances goes through compute_task_metrics, which this
    also uses once it has scored the tracker's responses.

    Args:
        tracker: Completed TaskTracker
        max_hard_failure_rate: The failure budget, as for compute_task_metrics.

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

    tracker.task.strip_thinking_traces(responses)
    scored = await tracker.task.score_responses(responses)
    return compute_task_metrics(
        spec=tracker.spec,
        task=tracker.task,
        scored_responses=list(scored),
        failed_instances=tracker.failed_instances,
        total_instances=tracker.total_instances,
        duration_seconds=duration,
        max_hard_failure_rate=max_hard_failure_rate,
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
    """Compute metrics from pre-scored responses and apply the failure budget.

    Unscored instances are left out of the metrics. Instances with no output
    (``failed_instances``) and unscored instances both count against
    ``max_hard_failure_rate``; a task over it, or with nothing scored, is failed and
    its metrics are not published.

    Args:
        spec: Task specification string.
        task: Task instance for metric computation.
        scored_responses: List of already-scored responses.
        failed_instances: Dict of instance_idx -> error message for instances that
            came back with no output.
        total_instances: Total number of instances in the task.
        duration_seconds: Duration of the task.
        max_hard_failure_rate: Fraction of instances allowed to have no output or no
            scored output before the task is marked failed.

    Returns:
        TaskResult with metrics and predictions.
    """
    instances_failed = len(failed_instances)
    error_summary = summarize_failed_instances(failed_instances)

    if not scored_responses:
        summary = error_summary or "No responses"
        budget_error = failure_budget_error(
            instances_measured=0,
            instances_failed=instances_failed,
            instances_unscored=0,
            instances_processed=total_instances,
            first_error=_first_instance_error(failed_instances),
            max_failure_rate=max_hard_failure_rate,
        )
        return TaskResult(
            spec=spec,
            config=task.config.to_dict(),
            num_instances=0,
            metrics={},
            error=budget_error or summary,
            duration_seconds=duration_seconds,
            error_summary=summary,
            instances_processed=total_instances,
            instances_failed=instances_failed,
            failure_budget_exceeded=budget_error is not None,
        )

    # Instances no scorer could score are left out rather than counted as zeros; the
    # task's compute_metrics leaves out, per metric, those its own scorer could not.
    scorers = tuple(task._get_scorers())
    metrics = task.compute_metrics(measurable_responses(scored_responses, scorers=scorers))
    # Counted after compute_metrics, which can itself record scoring errors (a task-level
    # evaluator that failed, for example).
    failures = count_scoring_failures(scored_responses, scorers)
    # As for the budget, an instance counts as scored only if every scorer scored it.
    instances_scored = len(scored_responses) - failures.instances_unscored

    # Build predictions
    predictions = build_predictions(scored_responses, metrics=task.config.metrics)
    requests = build_requests_from_responses(scored_responses, task.config.name)

    # Extract metric metadata
    primary_metric = get_metric_metadata(task)

    if error_summary:
        logger.warning(
            f"Task {spec} completed with failures: {error_summary}. "
            f"Computed metrics on {instances_scored}/{total_instances} instances."
        )
    if failures:
        logger.warning(
            f"Task {spec}: {_describe_scoring_failures(failures)} "
            f"First scoring error: {failures.first_error}"
        )

    budget_error = failure_budget_error(
        instances_measured=instances_scored,
        instances_failed=instances_failed,
        instances_unscored=failures.instances_unscored,
        instances_processed=total_instances,
        first_error=_first_instance_error(failed_instances) or failures.first_error,
        max_failure_rate=max_hard_failure_rate,
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
        error=budget_error,
        error_summary=error_summary,
        instances_processed=total_instances,
        instances_failed=instances_failed,
        instances_unscored=failures.instances_unscored,
        failure_budget_exceeded=budget_error is not None,
        scoring_failures=failures.to_dict() if failures else None,
        generation_counts=count_generations(scored_responses),
    )


def _describe_scoring_failures(failures: ScoringFailures) -> str:
    """One sentence on how many instances and outputs could not be scored."""
    parts = []
    if failures.instances_unscored:
        parts.append(
            f"{failures.instances_unscored} instance(s) could not be scored and are left "
            "out of the metrics."
        )
    if failures.instances_partially_scored:
        parts.append(
            f"{failures.instances_partially_scored} instance(s) were scored from only some "
            "of their outputs."
        )
    parts.append(f"{failures.outputs_unscored} output(s) failed scoring in all.")
    return " ".join(parts)


def failure_budget_error(
    *,
    instances_measured: int,
    instances_failed: int,
    instances_unscored: int,
    instances_processed: int,
    first_error: str | None,
    max_failure_rate: float,
) -> str | None:
    """Describe a breach of the task's failure budget, if there is one.

    An instance counts against the budget when it came back with no output
    (``instances_failed``) or when none of its outputs could be scored
    (``instances_unscored``). Either way it contributes nothing to the metrics.

    Two separate conditions fail a task. A task with nothing scored produced no
    metrics, and there is no budget at which that is a result worth publishing, so
    it fails even when the budget has been opened all the way up to 1.0. Otherwise
    the failure rate is compared against the budget, and that comparison is strict:
    a task sitting exactly on the budget passes.

    Args:
        instances_measured: Instances whose scores count toward the metrics.
        instances_failed: Instances that came back with no output.
        instances_unscored: Saved instances none of whose outputs could be scored.
        instances_processed: Instances the runner saw.
        first_error: First underlying error, quoted if present so the message says
            why the instances failed and not just how many.
        max_failure_rate: Maximum tolerated fraction of failed and unscored instances.

    Returns:
        An error string describing the breach, or None when within budget.
    """
    if instances_processed <= 0:
        return None

    detail = f" First failure: {first_error}" if first_error else ""
    breakdown = (
        f"{instances_failed} produced no output and {instances_unscored} could not be scored"
    )

    if instances_measured <= 0:
        return (
            f"No instance was scored out of {instances_processed} ({breakdown}). A task "
            f"with nothing scored fails at any failure budget.{detail}"
        )

    lost = instances_failed + instances_unscored
    if lost <= 0:
        return None
    rate = lost / instances_processed
    if rate <= max_failure_rate:
        return None
    return (
        f"{lost} of {instances_processed} instances ({rate:.1%}) failed, above the failure "
        f"budget of {max_failure_rate:.1%}: {breakdown}.{detail}"
    )


def _first_instance_error(failed_instances: dict[int, str]) -> str | None:
    """The first recorded instance error, used to make gate messages actionable."""
    return next(iter(failed_instances.values()), None)


__all__ = [
    "prepare_task_items",
    "build_requests_from_items",
    "finalize_task",
    "compute_task_metrics",
    "failure_budget_error",
]
