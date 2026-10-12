"""Result processing and aggregation for async evaluation runner."""

from __future__ import annotations

import asyncio
import queue
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from olmo_eval.common.beaker_status import BeakerStatusReporter
from olmo_eval.common.logging import get_logger
from olmo_eval.common.progress import ProgressLogger
from olmo_eval.common.types import Response
from olmo_eval.common.types.trajectory import AgentTrajectory
from olmo_eval.runners.asynq.monitoring import check_workers_alive
from olmo_eval.runners.asynq.preparation import compute_task_metrics, finalize_task
from olmo_eval.runners.asynq.types import (
    WORKER_FATAL,
    ResultItem,
    TaskTracker,
)
from olmo_eval.runners.common.constants import HardFailureRateExceeded
from olmo_eval.runners.common.types import (
    DEFAULT_MAX_HARD_FAILURE_RATE,
    TASK_COST_KEYS,
    TaskResult,
)
from olmo_eval.runners.processing.aggregation import compute_suite_aggregations
from olmo_eval.runners.processing.utils import compute_task_hash

if TYPE_CHECKING:
    import multiprocessing as mp

    from olmo_eval.common.execution import ScoringContext
    from olmo_eval.evals.tasks.common import Task

logger = get_logger(__name__)


def is_hard_failure(result_item: ResultItem) -> bool:
    """Whether an instance result contributes nothing to the metrics.

    A hard failure carries an error and no outputs at all. A soft failure (an
    error alongside a usable output, e.g. MaxTurnsExceeded with a fallback
    answer) still gets scored, so it is not a hard failure.
    """
    return bool(result_item.error) and not result_item.outputs


InstanceKey = tuple[str, str, int]


def utc_iso(timestamp: float | None) -> str | None:
    """POSIX seconds as an ISO 8601 UTC timestamp."""
    if timestamp is None:
        return None
    return datetime.fromtimestamp(timestamp, UTC).isoformat()


def apply_task_cost(result: TaskResult, tracker: TaskTracker, completed_at: float) -> None:
    """Copy the tracker's request timing and token totals onto a finished TaskResult."""
    result.last_completed_at = utc_iso(completed_at)
    if tracker.results_recorded == 0:
        return
    result.first_request_at = utc_iso(tracker.first_request_at)
    result.prompt_tokens_total = tracker.prompt_tokens_total
    result.completion_tokens_total = tracker.completion_tokens_total
    result.attributed_inference_seconds = tracker.attributed_inference_seconds


def _build_pending_instance_keys(trackers: dict[str, TaskTracker]) -> set[InstanceKey]:
    """Build the exact set of inference results expected from workers."""
    return {
        (tracker.model_name, spec, instance_idx)
        for spec, tracker in trackers.items()
        for instance_idx in range(tracker.total_instances)
    }


def _consume_pending_instance(pending_instances: set[InstanceKey], result_item: ResultItem) -> None:
    """Mark a result received, rejecting duplicate or unexpected identities."""
    key = (result_item.model_name, result_item.task_id, result_item.instance_idx)
    if key not in pending_instances:
        raise RuntimeError(
            "Received duplicate or unexpected inference result for "
            f"{result_item.task_id}[{result_item.instance_idx}]"
        )
    pending_instances.remove(key)


def _format_pending_instances(pending_instances: set[InstanceKey]) -> str:
    """Summarize unresolved inference results for failure messages."""
    ordered = sorted(pending_instances)
    preview = ", ".join(
        f"{model_name}:{spec}[{instance_idx}]" for model_name, spec, instance_idx in ordered[:10]
    )
    if len(ordered) > 10:
        preview += f", ... and {len(ordered) - 10} more"
    return f"{len(ordered)} inference result(s) pending ({preview})"


def _format_worker_failure(error: object, pending_instances: set[InstanceKey]) -> str:
    """Combine a fatal worker error with unresolved-instance accounting."""
    return f"Worker process crashed with {_format_pending_instances(pending_instances)}: {error}"


def _format_scoring_error(exc: Exception, *, phase: str) -> dict[str, str]:
    """Build a JSON-serializable scoring error payload."""
    error = {
        "phase": phase,
        "type": type(exc).__qualname__,
    }
    message = str(exc).strip()
    if message:
        error["message"] = message
    return error


def _record_scoring_failure(
    response: Response,
    *,
    scorer_names: list[str],
    error: dict[str, str],
) -> Response:
    """Annotate a response when scoring aborts before per-output results exist."""
    for scorer_name in scorer_names:
        response.scores.setdefault(scorer_name, 0.0)

    for output in response.outputs:
        if output.metadata is None:
            output.metadata = {}
        existing = output.metadata.get("scoring_errors")
        if not isinstance(existing, dict):
            existing = {}
            output.metadata["scoring_errors"] = existing
        existing["__response__"] = error
        for scorer_name in scorer_names:
            output.metadata.setdefault(f"score:{scorer_name}", 0.0)

    return response


async def process_results(
    trackers: dict[str, TaskTracker],
    result_queue: mp.Queue,
    workers: list[mp.process.BaseProcess],
    scoring_context: ScoringContext,
    scoring_concurrency: int,
    total_tasks: int,
    total_instances: int,
    model_name: str,
    save_predictions: bool,
    write_predictions_fn: Any,
    save_requests: bool,
    write_requests_fn: Any,
    max_hard_failure_rate: float = DEFAULT_MAX_HARD_FAILURE_RATE,
) -> dict[str, TaskResult]:
    """Process results from workers with inline async scoring.

    As each instance result comes in, it's scored directly in this event loop
    using asyncio tasks, eliminating cross-process serialization.

    Args:
        trackers: Task trackers keyed by spec.
        result_queue: Queue receiving results from inference workers.
        workers: List of inference worker processes.
        scoring_context: Scoring context with execution environment.
        scoring_concurrency: Maximum concurrent scoring operations.
        total_tasks: Total number of tasks.
        total_instances: Total number of instances across all tasks.
        model_name: Model name for reporting.
        save_predictions: Whether to save predictions.
        write_predictions_fn: Function to write predictions.
        max_hard_failure_rate: Fraction of a task's instances allowed to hard-fail
            before the task is marked failed.

    Returns:
        Dict mapping task spec to TaskResult.
    """
    results: dict[str, TaskResult] = {}
    tasks_complete = 0

    # Track scored responses per task: spec -> {idx: scored_response}
    scored_responses: dict[str, dict[int, Response]] = {spec: {} for spec in trackers}
    instances_scored: dict[str, int] = {spec: 0 for spec in trackers}

    # Progress tracking for scoring
    scoring_progress = ProgressLogger(
        total=total_instances, desc="Scored", logger=logger, color="cyan"
    )

    # Semaphore to limit concurrent scoring operations
    scoring_semaphore = asyncio.Semaphore(scoring_concurrency)
    in_flight_scoring: set[asyncio.Task[None]] = set()

    # Determine which tasks need the scoring worker (async scorers like sandboxed
    # code execution). All other tasks are scored inline to avoid mp.Queue overhead.
    tasks_needing_async_scoring: set[str] = set()
    for spec, tracker in trackers.items():
        if tracker.task is not None and tracker.task._has_async_scorers():
            tasks_needing_async_scoring.add(spec)
    if tasks_needing_async_scoring:
        logger.info(f"Tasks using async scoring worker: {tasks_needing_async_scoring}")

    def check_task_completion(spec: str) -> None:
        """Check if task is complete and finalize if so."""
        nonlocal tasks_complete
        tracker = trackers[spec]
        expected = tracker.total_instances - len(tracker.failed_instances)

        if instances_scored[spec] >= expected and spec not in results:
            responses_list = [
                scored_responses[spec][i] for i in sorted(scored_responses[spec].keys())
            ]
            assert tracker.task is not None
            completed_at = time.time()
            task_result = compute_task_metrics(
                spec=spec,
                task=tracker.task,
                scored_responses=responses_list,
                failed_instances=tracker.failed_instances,
                total_instances=tracker.total_instances,
                duration_seconds=completed_at - tracker.start_time,
                max_hard_failure_rate=max_hard_failure_rate,
            )
            apply_task_cost(task_result, tracker, completed_at)
            results[spec] = task_result
            tasks_complete += 1
            _report_task_completion(model_name, task_result)
            if save_predictions and task_result.predictions:
                task_hash = compute_task_hash(task_result.config)
                write_predictions_fn(
                    model_name, task_result.spec, task_result.predictions, task_hash
                )
            if save_requests and task_result.requests:
                task_hash = compute_task_hash(task_result.config)
                write_requests_fn(model_name, task_result.spec, task_result.requests, task_hash)

    async def score_and_store(
        spec: str,
        instance_idx: int,
        response: Response,
        task: Task,
    ) -> None:
        """Score a single response and store the result."""
        async with scoring_semaphore:
            try:
                task.strip_thinking_traces([response])
                scored_list = await task.score_responses([response], context=scoring_context)
                scored = scored_list[0] if scored_list else response
            except Exception as e:
                error = _format_scoring_error(e, phase="response")
                logger.warning(
                    "Failed to score %s[%s]: %s",
                    spec,
                    instance_idx,
                    error.get("message", error["type"]),
                )
                scored = _record_scoring_failure(
                    response,
                    scorer_names=list(task._get_scorers()),
                    error=error,
                )

        scored_responses[spec][instance_idx] = scored
        instances_scored[spec] += 1
        scoring_progress.update(1)
        check_task_completion(spec)

    def _reap_done_tasks() -> None:
        """Remove completed scoring tasks from in_flight set."""
        done = {t for t in in_flight_scoring if t.done()}
        for t in done:
            in_flight_scoring.discard(t)
            # Surface any unexpected exceptions
            if t.exception() is not None:
                logger.error(f"Scoring task failed unexpectedly: {t.exception()}")

    # Pre-add error tasks to results (these don't need scoring)
    for spec, tracker in trackers.items():
        if tracker.error:
            task_result = await finalize_task(tracker)
            results[spec] = task_result
            tasks_complete += 1
            _report_task_completion(model_name, task_result)

    pending_instances = _build_pending_instance_keys(trackers)
    if len(pending_instances) != total_instances:
        raise RuntimeError(
            "Inference accounting mismatch before processing: "
            f"runner expected {total_instances} instance(s), "
            f"task trackers describe {len(pending_instances)}"
        )
    reporter = BeakerStatusReporter()
    report_progress = reporter.progress_callback("Processed")
    last_health_check = time.time()
    health_check_interval = 5.0
    workers_exited = False

    try:
        # Collect instance results and score each inline
        while pending_instances:
            _reap_done_tasks()

            try:
                result_item: ResultItem = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: result_queue.get(timeout=1.0)
                )
            except queue.Empty:
                if workers_exited:
                    exit_codes = [worker.exitcode for worker in workers]
                    fatal_message = (
                        f"All inference workers exited (exit codes: {exit_codes}) with "
                        f"{_format_pending_instances(pending_instances)}"
                    )
                    logger.error(f"FATAL: {fatal_message}")
                    raise RuntimeError(fatal_message) from None
                workers_exited = all(worker.exitcode is not None for worker in workers)
                if time.time() - last_health_check > health_check_interval:
                    try:
                        check_workers_alive(workers, result_queue)
                    except RuntimeError as e:
                        fatal_message = _format_worker_failure(e, pending_instances)
                        logger.error(f"FATAL: {fatal_message}")
                        raise RuntimeError(fatal_message) from e
                    last_health_check = time.time()
                continue

            # Check for fatal worker crash
            if result_item.task_id == WORKER_FATAL:
                fatal_message = _format_worker_failure(result_item.error, pending_instances)
                logger.error(f"FATAL: {fatal_message}")
                raise RuntimeError(fatal_message)

            _consume_pending_instance(pending_instances, result_item)
            report_progress(total_instances - len(pending_instances), total_instances)
            tracker = trackers[result_item.task_id]

            if tracker.error:
                continue

            tracker.record_usage(result_item)

            if is_hard_failure(result_item):
                # is_hard_failure() guarantees an error is set; the `or ""` is for
                # the type checker, which cannot see that through the predicate.
                instance_error = result_item.error or ""
                logger.warning(
                    f"Instance {result_item.instance_idx} failed for "
                    f"{result_item.task_id}: {instance_error}"
                )
                tracker.add_failure(result_item.instance_idx, instance_error)
                scoring_progress.update(1)
                check_task_completion(result_item.task_id)
                continue

            if result_item.error:
                # Soft error - log warning but continue to scoring
                logger.debug(
                    f"Instance {result_item.instance_idx} completed with warning for "
                    f"{result_item.task_id}: {result_item.error}"
                )

            # Build response
            trajectory = None
            if result_item.outputs:
                meta = result_item.outputs[0].metadata or {}
                traj_dict = meta.get("trajectory")
                if traj_dict:
                    trajectory = AgentTrajectory.from_dict(traj_dict)

            assert result_item.instance is not None
            assert result_item.request is not None
            response = Response(
                instance=result_item.instance,
                request=result_item.request,
                outputs=result_item.outputs,
                trajectory=trajectory,
                request_trace=result_item.request_trace,
            )

            # Score inline as an async task
            assert tracker.task is not None
            scoring_task = asyncio.create_task(
                score_and_store(
                    spec=result_item.task_id,
                    instance_idx=result_item.instance_idx,
                    response=response,
                    task=tracker.task,
                )
            )
            in_flight_scoring.add(scoring_task)

        # Wait for all in-flight scoring to complete
        if in_flight_scoring:
            await asyncio.gather(*in_flight_scoring, return_exceptions=True)

        # Final check — all tasks should be complete
        if tasks_complete < total_tasks:
            # Some tasks may have had all instances fail during inference
            for spec in trackers:
                if spec not in results:
                    check_task_completion(spec)
    finally:
        scoring_progress.close()
        report_progress(total_instances - len(pending_instances), total_instances, force=True)
        reporter.flush()

    return results


def _format_instance_accounting(result: TaskResult) -> str:
    """Render saved-vs-processed instance counts for the task summary line.

    Matches the summary table: a task that never processed an instance (a
    task-level failure) reports "-" rather than an invented count.
    """
    if not result.instances_processed:
        return "instances: -"
    accounting = f"{result.num_instances}/{result.instances_processed} instances saved"
    if result.instances_failed:
        accounting += f", {result.instances_failed} hard-failed"
    return accounting


def _report_task_completion(model_name: str, result: TaskResult) -> None:
    """Report when a task completes."""
    label = f"{model_name}:{result.spec}"
    accounting = _format_instance_accounting(result)
    if result.error:
        logger.error(f"\u2717 {label} ({accounting}) (ERROR: {result.error})")
    else:
        logger.info(f"\u2713 {label} ({accounting}, {result.duration_seconds:.1f}s)")


def aggregate_results(
    results: dict[str, TaskResult],
    expanded_tasks: list[str],
    task_specs: list[str],
    provider_config: Any,
    attention_backend: str | None,
    harness_config: Any | None = None,
) -> dict[str, Any]:
    """Aggregate results and prepare final output.

    Args:
        results: Dict mapping task spec to TaskResult.
        expanded_tasks: List of all expanded task specs.
        task_specs: Original task specs (may include suite names).
        provider_config: Provider configuration.
        attention_backend: Attention backend name.
        harness_config: Optional harness configuration.

    Returns:
        Dict with full results including task metrics, errors, and summary.
    """
    model_config = provider_config.to_dict() if hasattr(provider_config, "to_dict") else {}
    if attention_backend:
        model_config["attention_backend"] = attention_backend
    results_dict: dict[str, Any] = {
        "model": provider_config.alias or provider_config.model,
        "model_path": provider_config.model,
        "provider": str(provider_config.kind),
        "model_config": model_config,
        "tasks": {},
        "summary": {},
        "errors": [],
        "timestamp": datetime.now().isoformat(),
    }
    if harness_config is not None:
        results_dict["harness_config"] = (
            harness_config.to_dict() if hasattr(harness_config, "to_dict") else harness_config
        )

    # Process each task result
    for spec, task_result in results.items():
        task_data: dict[str, Any] = {}

        # Instance counts describe what happened, not whether it was published, so
        # they are recorded for failed tasks too. Metrics are not: a score computed
        # on a subset must not read as the task's score.
        task_data["num_instances"] = task_result.num_instances

        if task_result.error:
            task_data["error"] = task_result.error
            results_dict["errors"].append({"task": spec, "error": task_result.error})
        else:
            task_data["metrics"] = task_result.metrics
            task_data["duration_seconds"] = task_result.duration_seconds
            task_data["primary_metric"] = task_result.primary_metric
            if task_result.predictions:
                task_data["predictions"] = task_result.predictions

            # Get the primary metric for the summary
            if task_result.primary_metric and task_result.metrics:
                from olmo_eval.runners.processing.utils import get_primary_metric

                primary = get_primary_metric(task_result.metrics, task_result.primary_metric)
                if primary:
                    results_dict["summary"][spec] = {
                        "metric": primary[0],
                        "score": primary[1],
                    }

        # Saved-vs-processed accounting is recorded for every task, including the
        # ones that errored, so a partial run cannot look complete downstream.
        task_data["instances_saved"] = task_result.num_instances
        task_data["instances_processed"] = task_result.instances_processed
        task_data["instances_failed"] = task_result.instances_failed
        if task_result.error_summary:
            task_data["error_summary"] = task_result.error_summary

        for key in TASK_COST_KEYS:
            task_data[key] = getattr(task_result, key)

        task_data["config"] = task_result.config
        task_data["task_hash"] = (
            compute_task_hash(task_result.config) if task_result.config else None
        )
        results_dict["tasks"][spec] = task_data

    # Compute suite aggregations
    suite_aggs = compute_suite_aggregations(task_specs, results_dict["tasks"])
    if suite_aggs:
        results_dict["suites"] = suite_aggs

    return results_dict


def check_hard_failure_gate(results: dict[str, TaskResult]) -> None:
    """Fail the run when any task exceeded its hard-failure-rate budget.

    Call this only after results have been written, so the failing run still leaves
    metrics.json and predictions behind for diagnosis.

    Args:
        results: Dict mapping task spec to TaskResult.

    Raises:
        HardFailureRateExceeded: If at least one task tripped the gate.
    """
    tripped = sorted(
        (result for result in results.values() if result.hard_failure_rate_exceeded),
        key=lambda result: result.spec,
    )
    if not tripped:
        return
    details = "; ".join(f"{result.spec}: {result.error}" for result in tripped)
    raise HardFailureRateExceeded(
        f"{len(tripped)} task(s) exceeded the hard failure rate threshold: {details}"
    )
