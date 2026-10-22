"""Core types and constants for evaluation runners."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Filename suffixes for output files (consistent across all runners)
PREDICTIONS_SUFFIX = "-predictions.jsonl"
REQUESTS_SUFFIX = "-requests.jsonl"

# The failure budget: fraction of a task's instances allowed to contribute nothing to the
# metrics before the task is marked failed. Two kinds of instance count against it: a
# hard failure, which came back with an error and no outputs at all, and an unscored
# instance, none of whose outputs could be scored. Soft failures (an error alongside a
# usable output, e.g. MaxTurnsExceeded with a fallback answer) are still scored and do
# not count here. Independently of this budget, a task with nothing scored always fails:
# it produced no metrics, so there is nothing to publish at any rate.
DEFAULT_MAX_HARD_FAILURE_RATE = 0.05

# Per-task cost keys written to each metrics.json task entry. Tasks share inference
# workers, so duration_seconds (from the shared processing start) is not a task's own
# cost; these are.
#   first_request_at: when the task's first request was sent to a worker
#   last_completed_at: when the task's last instance finished (inference and scoring)
#   prompt_tokens_total / completion_tokens_total: summed over every request of the
#       task as the provider reported them (rules in runners/common/usage.py)
#   attributed_inference_seconds: per-batch inference wall time split by token share
TASK_COST_KEYS = (
    "first_request_at",
    "last_completed_at",
    "prompt_tokens_total",
    "completion_tokens_total",
    "attributed_inference_seconds",
)

# Run-level timing keys written to metrics.json by the async runner.
#   startup_seconds: from the start of AsyncEvalRunner.run_async (before task data
#       loading and provider init) until every inference worker was ready
#   processing_started_at: ISO 8601 UTC time when every worker was ready
#   processing_seconds: from processing_started_at until the last task finished
RUN_TIMING_KEYS = ("startup_seconds", "processing_started_at", "processing_seconds")


@dataclass
class TaskResult:
    """Result from executing a single task.

    Metrics are stored in a nested structure: {metric_name: {scorer_name: score}}.
    The primary_metric uses "metric_name:scorer_name" format.
    """

    spec: str
    config: dict[str, Any]
    num_instances: int  # Instances that were scored and saved
    metrics: dict[str, dict[str, float]]
    error: str | None = None
    duration_seconds: float = 0.0
    predictions: list[dict] | None = None
    requests: list[dict] | None = None  # oe-eval compatible request objects
    primary_metric: str | None = None  # Format: "metric_name:scorer_name"
    # Failure accounting. `instances_processed` counts every instance the runner saw
    # (saved plus hard-failed), so `num_instances` vs `instances_processed` is the
    # saved-vs-processed gap that a partially failed run would otherwise hide.
    # `instances_failed` came back with no output; `instances_unscored` were saved but
    # no output could be scored. Both count against the failure budget.
    error_summary: str | None = None
    instances_processed: int = 0
    instances_failed: int = 0
    instances_unscored: int = 0
    failure_budget_exceeded: bool = False
    # See olmo_eval.common.scoring_failures; None when nothing failed scoring.
    scoring_failures: dict[str, Any] | None = None
    # See olmo_eval.runners.processing.generation_counts.
    generation_counts: dict[str, int] | None = None
    # Per-task cost (see TASK_COST_KEYS). Timestamps are ISO 8601 UTC. None when the
    # runner could not measure the value.
    first_request_at: str | None = None
    last_completed_at: str | None = None
    prompt_tokens_total: int | None = None
    completion_tokens_total: int | None = None
    attributed_inference_seconds: float | None = None

    @property
    def failure_rate(self) -> float:
        """Fraction of processed instances with no output or no scored output."""
        if self.instances_processed <= 0:
            return 0.0
        return (self.instances_failed + self.instances_unscored) / self.instances_processed

    def to_dict(self, include_predictions: bool = False) -> dict[str, Any]:
        """Serialize to dictionary for JSON output.

        Args:
            include_predictions: Whether to include predictions in the output.
                Defaults to False since predictions are typically written separately.

        Returns:
            Dictionary with task result data.
        """
        result: dict[str, Any] = {
            "config": self.config,
            "num_instances": self.num_instances,
            "metrics": self.metrics,
            "duration_seconds": self.duration_seconds,
            "instances_saved": self.num_instances,
            "instances_processed": self.instances_processed,
            "instances_failed": self.instances_failed,
        }
        if self.primary_metric:
            result["primary_metric"] = self.primary_metric
        if self.error:
            result["error"] = self.error
        if self.error_summary:
            result["error_summary"] = self.error_summary
        if self.generation_counts is not None:
            result["generation_counts"] = self.generation_counts
        if self.scoring_failures is not None:
            result["scoring_failures"] = self.scoring_failures
        if include_predictions and self.predictions:
            result["predictions"] = self.predictions
        return result
