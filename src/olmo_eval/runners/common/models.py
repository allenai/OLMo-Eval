"""Dataclasses for runner configuration and metrics output."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from olmo_eval.common.utils import Serializable


@dataclass
class ModelMetadata(Serializable):
    """Model metadata for metrics.json output format."""

    model: str
    provider: str
    dtype: str = "auto"
    tokenizer: str | None = None
    revision: str | None = None
    attention_backend: str | None = None


@dataclass
class TaskMetricsEntry(Serializable):
    """A task entry in the metrics output.

    Metrics are stored in a nested structure: {metric_name: {scorer_name: score}}.
    The primary_metric uses "metric_name:scorer_name" format.
    """

    task: str
    metrics: dict[str, dict[str, float]]
    num_instances: int
    model: str | None = None  # Only set for multi-model format
    primary_metric: str | None = None  # Format: "metric_name:scorer_name"
    config: dict[str, Any] | None = None
    duration_seconds: float | None = None
    task_hash: str | None = None
    # Saved-vs-processed accounting; see TaskResult for the exact definitions.
    instances_saved: int | None = None
    instances_processed: int | None = None
    instances_failed: int | None = None
    error_summary: str | None = None
    # Per-task cost; see TASK_COST_KEYS in runners/common/types.py
    first_request_at: str | None = None
    last_completed_at: str | None = None
    prompt_tokens_total: int | None = None
    completion_tokens_total: int | None = None
    attributed_inference_seconds: float | None = None


@dataclass
class ScoreSummary(Serializable):
    """Summary entry with metric:scorer identifier and score.

    The metric field uses "metric_name:scorer_name" format to uniquely
    identify which metric and scorer produced this score.
    """

    metric: str  # Format: "metric_name:scorer_name"
    score: float


@dataclass
class MetricsOutput(Serializable):
    """Top-level metrics.json output structure."""

    timestamp: str
    config: dict[str, Any]  # ProviderConfig.to_dict() or {"models": {name: config}}
    tasks: list[dict[str, Any]]  # List of TaskMetricsEntry.to_dict()
    summary: dict[str, Any]  # task_name -> ScoreSummary or model -> task -> ScoreSummary
    errors: list[dict[str, Any]] = field(default_factory=list)
    # Experiment identification fields for querying results
    experiment_id: str | None = None
    experiment_name: str | None = None
    experiment_group: str | None = None
    # Duration metrics
    experiment_duration_seconds: float | None = None
    provider_init_seconds: dict[str, float] | None = None  # model_name -> init_time
    # Run timing; see RUN_TIMING_KEYS in runners/common/types.py
    startup_seconds: float | None = None
    processing_started_at: str | None = None
    processing_seconds: float | None = None
