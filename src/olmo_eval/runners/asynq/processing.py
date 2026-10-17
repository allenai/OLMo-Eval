"""Request processing for async evaluation runners."""

from __future__ import annotations

import logging
import multiprocessing as mp
import time
from typing import TYPE_CHECKING

from olmo_eval.common.logging import get_logger
from olmo_eval.inference.errors import classify_terminal_provider_error, request_error
from olmo_eval.inference.metrics.core.stats import compute_batch_hash
from olmo_eval.runners.asynq.share_clock import ShareClock
from olmo_eval.runners.asynq.types import QueueItem, ResultItem

if TYPE_CHECKING:
    from olmo_eval.harness import Harness

logger = get_logger(__name__)

_NO_OUTPUT_ERROR = "Provider returned no output for this request (see the provider's logged error)"


def _get_native_ids(items: list[QueueItem]) -> list[str]:
    """Extract native IDs from queue items for batch hashing."""
    return [f"{item.task_id}:{item.instance_idx}" for item in items]


def _flush_batch_metrics(harness: Harness, items: list[QueueItem], log: logging.Logger) -> object:
    """Flush batch metrics without letting telemetry failures interrupt inference.

    Returns what harness.flush_metrics returned, or None if flushing failed.
    """
    try:
        return harness.flush_metrics(compute_batch_hash(_get_native_ids(items)))
    except Exception as e:
        log.warning(f"Failed to flush batch metrics: {e}")
        return None


def attribute_batch_seconds(seconds: float, batch: object, num_requests: int) -> list[float]:
    """Split a batch's share of worker time across its requests.

    Each request gets seconds * (its prompt + completion tokens) / (the batch's
    tokens), using the per-request metrics the provider recorded for this batch.
    Without such metrics, or when they do not line up one to one with the batch's
    requests, every request gets an equal share.
    """
    from olmo_eval.inference.metrics.core.schema import BatchMetrics

    if isinstance(batch, BatchMetrics) and len(batch.requests) == num_requests:
        tokens = [r.prompt_tokens + r.completion_tokens for r in batch.requests]
        total = sum(tokens)
        if total > 0:
            return [seconds * count / total for count in tokens]
    return [seconds / num_requests] * num_requests


def _format_cause(cause: BaseException) -> str:
    """Format the cause of an exception with fallback for empty strings."""
    cause_type = type(cause).__qualname__
    cause_str = str(cause)

    # If str(cause) is non-empty, use it
    if cause_str:
        return f"{cause_type}: {cause_str}"

    # Try args (often contains the real message)
    if cause.args:
        args_str = ", ".join(str(a) for a in cause.args if a)
        if args_str:
            return f"{cause_type}: {args_str}"

    # Just return the type name
    return cause_type


def _format_error_detail(exc: Exception) -> str:
    """Format exception with HTTP details for debugging."""
    parts = [f"type: {type(exc).__qualname__}"]

    # HTTP status code
    status = getattr(exc, "status_code", None)
    if status is not None:
        parts.append(f"status_code: {status}")

    # Request URL from response
    response = getattr(exc, "response", None)
    if response is not None:
        url = getattr(response, "url", None)
        if url is not None:
            parts.append(f"url: {url}")

    # Error message
    message = getattr(exc, "message", None) or str(exc)
    if len(message) > 500:
        message = message[:500] + "..."
    parts.append(f"message: {message}")

    # Root cause (e.g., httpx.ReadTimeout)
    cause = exc.__cause__
    if cause is not None:
        parts.append(f"cause: {_format_cause(cause)}")

    return " | ".join(parts)


async def process_chat_request(
    item: QueueItem,
    harness: Harness,
    result_queue: mp.Queue,
    worker_logger: logging.Logger | None = None,
    clock: ShareClock | None = None,
) -> None:
    """Process a single CHAT request via harness.run().

    CHAT requests use the async harness.run() method which handles agentic
    loops with tool calls. These must be processed individually.

    Args:
        item: Queue item to process (must be CHAT type).
        harness: Harness instance for execution.
        result_queue: Queue to put results.
        worker_logger: Logger with worker identification.
        clock: The worker's clock; the result carries the rollout's share of
            worker time. Without one the rollout is timed on its own.
    """
    log = worker_logger or logger
    from dataclasses import replace as dataclass_replace

    # Build trace metadata for observability
    trace_metadata = {
        "task_id": item.task_id,
        "instance_idx": item.instance_idx,
        "instance_id": item.instance.metadata.get("id", str(item.instance_idx)),
        "date_cutoff": item.instance.metadata.get("retrieval_date_cutoff"),
    }
    prepared_request = harness._apply_config(item.request)
    request_trace = harness.provider.describe_request(prepared_request, item.sampling_params)
    sent_at = time.time()
    if clock is None:
        clock = ShareClock()
    token = clock.start()
    seconds: float | None = None

    try:
        try:
            harness_result = await harness.run(
                item.request, item.sampling_params, trace_metadata=trace_metadata
            )
        finally:
            seconds = clock.stop(token)
        final_output = harness_result.final_output

        if harness_result.trajectory is not None:
            output_with_metadata = dataclass_replace(
                final_output,
                metadata={
                    **(final_output.metadata or {}),
                    "trajectory": harness_result.trajectory.to_dict(),
                    "max_turns_reached": harness_result.max_turns_reached,
                    "total_tool_calls": harness_result.total_tool_calls,
                    "num_turns": harness_result.num_turns,
                },
            )
        else:
            output_with_metadata = final_output

        provider_error = request_error([final_output])
        result_queue.put(
            ResultItem(
                model_name=item.model_name,
                task_id=item.task_id,
                instance_idx=item.instance_idx,
                instance=item.instance,
                request=prepared_request,
                request_trace=request_trace,
                outputs=[] if provider_error is not None else [output_with_metadata],
                error=provider_error or harness_result.error,
                attempt=item.attempt,
                sent_at=sent_at,
                attributed_seconds=seconds,
            )
        )

    except Exception as e:
        import traceback

        terminal_error = classify_terminal_provider_error(e)
        if terminal_error is not None:
            raise terminal_error from e

        error_detail = _format_error_detail(e)
        full_tb = traceback.format_exc()
        log.error(f"Error on CHAT instance {item.instance_idx}: {error_detail}\n{full_tb}")

        result_queue.put(
            ResultItem(
                model_name=item.model_name,
                task_id=item.task_id,
                instance_idx=item.instance_idx,
                instance=item.instance,
                request=prepared_request,
                request_trace=request_trace,
                outputs=[],
                error=error_detail,
                attempt=item.attempt,
                sent_at=sent_at,
                attributed_seconds=seconds,
            )
        )


async def process_batch(
    items: list[QueueItem],
    harness: Harness,
    result_queue: mp.Queue,
    worker_logger: logging.Logger | None = None,
    clock: ShareClock | None = None,
) -> None:
    """Process a batch of COMPLETION or LOGLIKELIHOOD requests.

    All items must have the same request_type and sampling_params.
    Calls the underlying provider once for the entire batch using the harness-
    transformed requests.

    Args:
        items: List of queue items to process (same type and sampling_params).
        harness: Harness instance for execution.
        result_queue: Queue to put results.
        worker_logger: Logger with worker identification.
        clock: The worker's clock; each result carries its share of the batch's
            share of worker time. Without one the batch is timed on its own.
    """
    from olmo_eval.common.types import RequestType

    log = worker_logger or logger

    if not items:
        return

    request_type = items[0].request.request_type
    sampling_params = items[0].sampling_params
    requests = [item.request for item in items]
    prepared_requests = [harness._apply_config(request) for request in requests]
    request_traces = [
        harness.provider.describe_request(request, sampling_params) for request in prepared_requests
    ]

    # Results are queued after the batch ends, so each one can carry its share of
    # the batch's worker time.
    results: list[ResultItem] = []
    sent_at = time.time()
    if clock is None:
        clock = ShareClock()
    token = clock.start()
    try:
        if request_type == RequestType.LOGLIKELIHOOD:
            all_outputs = await harness.provider.alogprobs(prepared_requests, sampling_params)
        else:
            all_outputs = await harness.provider.agenerate(prepared_requests, sampling_params)

        # An empty result is a fallback failure for requests that expected output.
        for item, prepared_request, request_trace, outputs in zip(
            items, prepared_requests, request_traces, all_outputs, strict=True
        ):
            error = request_error(outputs)
            expects_output = request_type != RequestType.LOGLIKELIHOOD or bool(
                prepared_request.continuations
            )
            if error is None and expects_output and not outputs:
                error = _NO_OUTPUT_ERROR
            if error is not None:
                log.warning(f"Instance {item.instance_idx} failed: {error}")
            results.append(
                ResultItem(
                    model_name=item.model_name,
                    task_id=item.task_id,
                    instance_idx=item.instance_idx,
                    instance=item.instance,
                    request=prepared_request,
                    request_trace=request_trace,
                    outputs=[] if error is not None else outputs,
                    error=error,
                    attempt=item.attempt,
                    sent_at=sent_at,
                )
            )

    except Exception as e:
        # A failed batch still took worker time; split it evenly across its items.
        share = clock.stop(token) / len(items)
        terminal_error = classify_terminal_provider_error(e)
        if terminal_error is not None:
            raise terminal_error from e

        # Batch failed - report the results already built, then errors for the rest
        for result in results:
            result.attributed_seconds = share
            result_queue.put(result)
        reported = len(results)
        error_detail = _format_error_detail(e)
        log.error(
            f"Batch error ({len(items) - reported} of {len(items)} items affected): {error_detail}"
        )

        for item, prepared_request, request_trace in zip(
            items[reported:], prepared_requests[reported:], request_traces[reported:], strict=True
        ):
            result_queue.put(
                ResultItem(
                    model_name=item.model_name,
                    task_id=item.task_id,
                    instance_idx=item.instance_idx,
                    instance=item.instance,
                    request=prepared_request,
                    request_trace=request_trace,
                    outputs=[],
                    error=error_detail,
                    attempt=item.attempt,
                    sent_at=sent_at,
                    attributed_seconds=share,
                )
            )
    except BaseException:
        clock.stop(token)
        raise
    else:
        batch_seconds = clock.stop(token)
        batch = _flush_batch_metrics(harness, items, log)
        for result, share in zip(
            results, attribute_batch_seconds(batch_seconds, batch, len(results)), strict=True
        ):
            result.attributed_seconds = share
            result_queue.put(result)


async def process_items(
    items: list[QueueItem],
    harness: Harness,
    result_queue: mp.Queue,
    max_concurrency: int | None = None,
    worker_logger: logging.Logger | None = None,
    show_progress: bool = True,
    clock: ShareClock | None = None,
) -> None:
    """Process queue items, batching where possible.

    COMPLETION and LOGLIKELIHOOD requests are grouped by sampling_params and
    processed in batches. CHAT requests are processed individually with async
    concurrency.

    Args:
        items: Queue items to process.
        harness: Harness instance for execution.
        result_queue: Queue to put results.
        max_concurrency: Maximum concurrent CHAT requests.
        worker_logger: Logger with worker identification.
        show_progress: Whether to show progress logging (disable for streaming).
        clock: The worker's clock, shared by every batch and rollout the worker
            runs so overlapping work splits worker time. Defaults to a clock for
            these items alone.
    """
    from olmo_eval.common.types import RequestType, SamplingParams

    log = worker_logger or logger
    if clock is None:
        clock = ShareClock()

    chat_items: list[QueueItem] = []
    batchable_items: list[QueueItem] = []

    has_scaffold = bool(harness.config.scaffold)

    for item in items:
        if item.request.request_type == RequestType.CHAT and has_scaffold:
            chat_items.append(item)
        else:
            batchable_items.append(item)

    if batchable_items:
        batches: dict[tuple[RequestType, SamplingParams | None], list[QueueItem]] = {}
        for item in batchable_items:
            key = (item.request.request_type, item.sampling_params)
            if key not in batches:
                batches[key] = []
            batches[key].append(item)

        for batch in batches.values():
            await process_batch(batch, harness, result_queue, log, clock)

    if chat_items:
        from olmo_eval.inference.dispatch import dispatch_concurrent

        async def process(item: QueueItem) -> None:
            await process_chat_request(item, harness, result_queue, log, clock)

        if show_progress:
            from olmo_eval.common.progress import ProgressLogger

            progress = ProgressLogger(
                total=len(chat_items), desc="Processed", logger=log, color="green"
            )

            def on_progress(done: int, total: int) -> None:
                progress.update(1)

            try:
                await dispatch_concurrent(
                    chat_items,
                    process,
                    max_in_flight=max_concurrency or len(chat_items),
                    on_progress=on_progress,
                )
            finally:
                progress.close()
        else:
            await dispatch_concurrent(
                chat_items,
                process,
                max_in_flight=max_concurrency or len(chat_items),
            )

        _flush_batch_metrics(harness, chat_items, log)


__all__ = [
    "attribute_batch_seconds",
    "process_chat_request",
    "process_batch",
    "process_items",
]
