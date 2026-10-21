"""Data structures for async evaluation runners."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from olmo_eval.common.types import Instance, LMOutput, LMRequest, Response, SamplingParams

if TYPE_CHECKING:
    from olmo_eval.evals.tasks.common import Task

# Sentinel value for fatal worker errors
WORKER_FATAL = "__WORKER_FATAL__"

# Default concurrency for scoring worker
DEFAULT_SCORING_CONCURRENCY = 8


def summarize_failed_instances(failed_instances: dict[int, str]) -> str | None:
    """Summarize hard-failed instances as one human-readable line.

    Args:
        failed_instances: Dict of instance_idx -> error message.

    Returns:
        A summary string, or None when nothing failed.
    """
    if not failed_instances:
        return None
    if len(failed_instances) == 1:
        idx, err = next(iter(failed_instances.items()))
        return f"Instance {idx} failed: {err}"
    first_error = next(iter(failed_instances.values()))
    return f"{len(failed_instances)} instances failed (first: {first_error})"


@dataclass
class QueueItem:
    """Single instance ready for generation."""

    model_name: str  # Which model this is for
    task_id: str  # Task spec string
    instance_idx: int  # Index within task's instance list
    instance: Instance
    request: LMRequest  # Pre-formatted request
    sampling_params: SamplingParams | None = None
    attempt: int = 0  # Retry attempt number


@dataclass
class TaskTracker:
    """Tracks completion state for a single (model, task) pair."""

    model_name: str  # Which model this is for
    spec: str
    task: Task | None  # None if task prep failed
    total_instances: int
    completed_count: int = 0
    responses: dict[int, Response] = field(default_factory=dict)
    failed_instances: dict[int, str] = field(default_factory=dict)  # idx -> error message
    error: str | None = None  # Task-level error (e.g., prep failed)
    start_time: float = field(default_factory=time.time)
    # Per-task cost, folded in from every worker result by record_usage()
    results_recorded: int = 0
    first_request_at: float | None = None  # Earliest ResultItem.sent_at (POSIX seconds)
    prompt_tokens_total: int | None = 0  # None once any result's count is unknown
    completion_tokens_total: int | None = 0
    attributed_inference_seconds: float | None = 0.0

    def record_usage(self, result: ResultItem) -> None:
        """Fold one worker result into the task's request timing and token totals.

        Hard-failed results (no outputs) add no tokens. A result whose outputs do
        not report a count, or that has no batch attribution, makes that total None.
        """
        from olmo_eval.common.types import RequestType
        from olmo_eval.runners.common.usage import request_token_usage

        self.results_recorded += 1
        if result.sent_at is not None and (
            self.first_request_at is None or result.sent_at < self.first_request_at
        ):
            self.first_request_at = result.sent_at

        loglikelihood = (
            result.request is not None and result.request.request_type == RequestType.LOGLIKELIHOOD
        )
        prompt, completion = request_token_usage(
            [output.metadata or {} for output in result.outputs], loglikelihood=loglikelihood
        )
        if self.prompt_tokens_total is not None:
            self.prompt_tokens_total = None if prompt is None else self.prompt_tokens_total + prompt
        if self.completion_tokens_total is not None:
            self.completion_tokens_total = (
                None if completion is None else self.completion_tokens_total + completion
            )
        if self.attributed_inference_seconds is not None:
            self.attributed_inference_seconds = (
                None
                if result.attributed_seconds is None
                else self.attributed_inference_seconds + result.attributed_seconds
            )

    def is_complete(self) -> bool:
        """Check if task is complete (all instances done, including failed ones)."""
        if self.error is not None:
            return True  # Task-level error stops everything
        processed = self.completed_count + len(self.failed_instances)
        return processed >= self.total_instances

    def add_response(self, idx: int, response: Response) -> bool:
        """Add a response. Returns True if task is now complete."""
        self.responses[idx] = response
        self.completed_count += 1
        return self.is_complete()

    def add_failure(self, idx: int, error: str) -> bool:
        """Record a failed instance. Returns True if task is now complete."""
        self.failed_instances[idx] = error
        return self.is_complete()


@dataclass
class ResultItem:
    """Result for a single instance from the worker."""

    model_name: str  # Which model produced this result
    task_id: str
    instance_idx: int
    instance: Instance | None  # None only for fatal error signals
    request: LMRequest | None  # None only for fatal error signals
    outputs: list[LMOutput]
    error: str | None = None
    attempt: int = 0
    request_trace: dict[str, Any] | None = None
    # When the worker sent the request to the provider (POSIX seconds)
    sent_at: float | None = None
    # This request's share of its batch's inference wall time, from batch metrics.
    # None when metrics are disabled or the batch could not be attributed.
    attributed_seconds: float | None = None


__all__ = [
    "DEFAULT_SCORING_CONCURRENCY",
    "WORKER_FATAL",
    "QueueItem",
    "TaskTracker",
    "ResultItem",
    "summarize_failed_instances",
]
