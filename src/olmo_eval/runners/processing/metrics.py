"""Metrics building and writing utilities."""

from __future__ import annotations

import json
import os
from typing import Any

from olmo_eval.common.logging import get_logger
from olmo_eval.runners.common.models import (
    MetricsOutput,
    ModelMetadata,
    ScoreSummary,
    TaskMetricsEntry,
)
from olmo_eval.runners.common.types import RUN_TIMING_KEYS, TASK_COST_KEYS
from olmo_eval.runners.processing.utils import get_primary_metric

logger = get_logger("runners.metrics")


def build_single_model_metrics(
    results: dict[str, Any],
    experiment_id: str | None = None,
    experiment_name: str | None = None,
    experiment_group: str | None = None,
    model_hash: str | None = None,
    experiment_duration_seconds: float | None = None,
    provider_init_seconds: dict[str, float] | None = None,
) -> MetricsOutput:
    """Build metrics output for single-model format.

    Args:
        results: Results dictionary with 'tasks', 'model_config', 'harness_config', etc.
        experiment_id: Unique ID for this experiment run.
        experiment_name: Human-readable experiment name.
        experiment_group: Group for related experiments.
        model_hash: Hash of model config.
        experiment_duration_seconds: Total time for the experiment.
        provider_init_seconds: Dict mapping model name to provider init time.

    Returns:
        MetricsOutput instance ready for serialization.
    """
    # Use full harness_config if available, otherwise build from model_config
    if "harness_config" in results:
        hc = results["harness_config"]
        if hasattr(hc, "to_dict"):
            config_dict = hc.to_dict()
        elif isinstance(hc, dict):
            config_dict = hc
        else:
            config_dict = dict(hc)
    else:
        # Fallback to legacy model_config format
        model_cfg = results.get("model_config", {})
        config = ModelMetadata(
            model=model_cfg.get("model", results.get("model", "")),
            provider=model_cfg.get("provider", results.get("provider", "")),
            dtype=model_cfg.get("dtype", "auto"),
            tokenizer=model_cfg.get("tokenizer"),
            revision=model_cfg.get("revision"),
            attention_backend=model_cfg.get("attention_backend"),
        )
        config_dict = config.to_dict()

    # Add model_hash to config
    if model_hash is not None:
        config_dict["model_hash"] = model_hash

    # Build task entries
    tasks_list: list[TaskMetricsEntry] = []
    for task_name, task_data in results.get("tasks", {}).items():
        entry = TaskMetricsEntry(
            task=task_name,
            metrics=task_data.get("metrics", {}),
            num_instances=task_data.get("num_instances", 0),
            primary_metric=task_data.get("primary_metric"),
            config=task_data.get("config"),
            duration_seconds=task_data.get("duration_seconds"),
            task_hash=task_data.get("task_hash"),
            instances_saved=task_data.get("instances_saved"),
            instances_processed=task_data.get("instances_processed"),
            instances_failed=task_data.get("instances_failed"),
            error_summary=task_data.get("error_summary"),
            **{key: task_data.get(key) for key in TASK_COST_KEYS},
        )
        tasks_list.append(entry)

    # Build summary with primary metric for each task
    summary: dict[str, ScoreSummary] = {}
    for task_name, task_data in results.get("tasks", {}).items():
        metrics = task_data.get("metrics", {})
        preferred = task_data.get("primary_metric")
        primary = get_primary_metric(metrics, preferred)
        if primary:
            metric_scorer, score = primary
            summary[task_name] = ScoreSummary(metric=metric_scorer, score=score)

    # Add suite summaries
    if "suites" in results:
        for suite_name, suite_data in results["suites"].items():
            metrics = suite_data.get("metrics", {})
            preferred = suite_data.get("primary_metric")
            primary = get_primary_metric(metrics, preferred)
            if primary:
                metric_scorer, score = primary
                summary[suite_name] = ScoreSummary(metric=metric_scorer, score=score)

    return MetricsOutput(
        timestamp=results.get("timestamp", ""),
        config=config_dict,
        tasks=[t.to_dict() for t in tasks_list],
        summary={k: v.to_dict() for k, v in summary.items()},
        errors=results.get("errors", []),
        experiment_id=experiment_id,
        experiment_name=experiment_name,
        experiment_group=experiment_group,
        experiment_duration_seconds=experiment_duration_seconds,
        provider_init_seconds=provider_init_seconds,
        **{key: results.get(key) for key in RUN_TIMING_KEYS},
    )


def build_multi_model_metrics(
    results: dict[str, Any],
    experiment_id: str | None = None,
    experiment_name: str | None = None,
    experiment_group: str | None = None,
    experiment_duration_seconds: float | None = None,
    provider_init_seconds: dict[str, float] | None = None,
) -> MetricsOutput:
    """Build metrics output for multi-model format.

    Args:
        results: Results dictionary with 'models' key containing per-model data.
        experiment_id: Unique ID for this experiment run.
        experiment_name: Human-readable experiment name.
        experiment_group: Group for related experiments.
        experiment_duration_seconds: Total time for the experiment.
        provider_init_seconds: Dict mapping model name to provider init time.

    Returns:
        MetricsOutput instance ready for serialization.
    """
    # Build config for each model (include model_hash per model)
    models_config: dict[str, dict[str, Any]] = {}
    for model_name, model_data in results.get("models", {}).items():
        model_cfg = model_data.get("model_config", {})
        config = ModelMetadata(
            model=model_cfg.get("model", model_data.get("model", "")),
            provider=model_cfg.get("provider", model_data.get("provider", "")),
            dtype=model_cfg.get("dtype", "auto"),
            tokenizer=model_cfg.get("tokenizer"),
            revision=model_cfg.get("revision"),
            attention_backend=model_cfg.get("attention_backend"),
        )
        config_dict = config.to_dict()
        # Include model_hash in per-model config if available
        if "_model_hash" in model_data:
            config_dict["model_hash"] = model_data["_model_hash"]
        models_config[model_name] = config_dict

    # Build task entries - flatten (model, task) pairs
    tasks_list: list[TaskMetricsEntry] = []
    for model_name, model_data in results.get("models", {}).items():
        for task_name, task_data in model_data.get("tasks", {}).items():
            entry = TaskMetricsEntry(
                task=task_name,
                model=model_name,
                metrics=task_data.get("metrics", {}),
                num_instances=task_data.get("num_instances", 0),
                primary_metric=task_data.get("primary_metric"),
                config=task_data.get("config"),
                duration_seconds=task_data.get("duration_seconds"),
                task_hash=task_data.get("task_hash"),
                instances_saved=task_data.get("instances_saved"),
                instances_processed=task_data.get("instances_processed"),
                instances_failed=task_data.get("instances_failed"),
                error_summary=task_data.get("error_summary"),
                **{key: task_data.get(key) for key in TASK_COST_KEYS},
            )
            tasks_list.append(entry)

    # Build summary with primary metric for each (model, task) pair
    summary: dict[str, dict[str, ScoreSummary]] = {}
    for model_name, model_data in results.get("models", {}).items():
        summary[model_name] = {}
        for task_name, task_data in model_data.get("tasks", {}).items():
            metrics = task_data.get("metrics", {})
            preferred = task_data.get("primary_metric")
            primary = get_primary_metric(metrics, preferred)
            if primary:
                metric_scorer, score = primary
                summary[model_name][task_name] = ScoreSummary(metric=metric_scorer, score=score)

        # Add suite summaries to this model's summary
        if "suites" in model_data:
            for suite_name, suite_data in model_data["suites"].items():
                metrics = suite_data.get("metrics", {})
                preferred = suite_data.get("primary_metric")
                primary = get_primary_metric(metrics, preferred)
                if primary:
                    metric_scorer, score = primary
                    summary[model_name][suite_name] = ScoreSummary(
                        metric=metric_scorer, score=score
                    )

    return MetricsOutput(
        timestamp=results.get("timestamp", ""),
        config={"models": models_config},
        tasks=[t.to_dict() for t in tasks_list],
        summary={
            model: {task: s.to_dict() for task, s in tasks.items()}
            for model, tasks in summary.items()
        },
        errors=results.get("errors", []),
        experiment_id=experiment_id,
        experiment_name=experiment_name,
        experiment_group=experiment_group,
        experiment_duration_seconds=experiment_duration_seconds,
        provider_init_seconds=provider_init_seconds,
        **{key: results.get(key) for key in RUN_TIMING_KEYS},
    )


def write_metrics_json(
    output_dir: str,
    results: dict[str, Any],
    multi_model: bool = False,
    experiment_id: str | None = None,
    experiment_name: str | None = None,
    experiment_group: str | None = None,
    model_hash: str | None = None,
    experiment_duration_seconds: float | None = None,
    provider_init_seconds: dict[str, float] | None = None,
) -> None:
    """Write metrics.json to the output directory.

    Args:
        output_dir: Directory to write metrics.json to.
        results: Results dictionary from the runner.
        multi_model: If True, use multi-model format with results["models"],
                    otherwise use single-model format with results["tasks"].
        experiment_id: Unique ID for this experiment run.
        experiment_name: Human-readable experiment name.
        experiment_group: Group for related experiments.
        model_hash: Hash of model config (single-model only).
        experiment_duration_seconds: Total time for the experiment.
        provider_init_seconds: Dict mapping model name to provider init time.
    """
    metrics_file = os.path.join(output_dir, "metrics.json")

    if multi_model:
        metrics_output = build_multi_model_metrics(
            results,
            experiment_id=experiment_id,
            experiment_name=experiment_name,
            experiment_group=experiment_group,
            experiment_duration_seconds=experiment_duration_seconds,
            provider_init_seconds=provider_init_seconds,
        )
    else:
        metrics_output = build_single_model_metrics(
            results,
            experiment_id=experiment_id,
            experiment_name=experiment_name,
            experiment_group=experiment_group,
            model_hash=model_hash,
            experiment_duration_seconds=experiment_duration_seconds,
            provider_init_seconds=provider_init_seconds,
        )

    os.makedirs(output_dir, exist_ok=True)
    with open(metrics_file, "w") as f:
        json.dump(metrics_output.to_dict(), f, indent=2)

    logger.info(f"Metrics written to {metrics_file}")


def log_summary(results: dict[str, Any], multi_model: bool = False) -> None:
    """Log summary of all task scores.

    Args:
        results: Results dictionary from the runner.
        multi_model: If True, iterate results["models"][model]["tasks"],
                    otherwise iterate results["tasks"] directly.
    """
    from rich.console import Console
    from rich.table import Table

    console = Console(force_terminal=True, width=120)

    table = Table(title="Results Summary")
    table.add_column("Task", style="cyan")
    table.add_column("Status")
    table.add_column("Instances")
    table.add_column("Metric")
    table.add_column("Result")

    def format_instances(task_data: dict[str, Any]) -> str:
        """Render saved-vs-processed counts, so a partial task cannot read as whole."""
        processed = task_data.get("instances_processed")
        if not processed:
            return "-"
        saved = task_data.get("instances_saved", task_data.get("num_instances", 0))
        failed = task_data.get("instances_failed") or 0
        instances = f"{saved}/{processed}"
        if failed:
            instances += f" ({failed} failed)"
        return instances

    def add_task_row(name: str, task_data: dict[str, Any]) -> None:
        metrics = task_data.get("metrics", {})
        error = task_data.get("error")
        preferred = task_data.get("primary_metric")
        primary = get_primary_metric(metrics, preferred)

        metric_name = primary[0] if primary else (preferred or "-")
        instances = format_instances(task_data)

        if error:
            table.add_row(name, "[red]Failed[/red]", instances, metric_name, str(error))
        elif primary:
            table.add_row(
                name, "[green]Success[/green]", instances, metric_name, f"{primary[1]:.4f}"
            )
        else:
            table.add_row(name, "[green]Success[/green]", instances, metric_name, "-")

    def _get_collapsed_tasks(suites: dict[str, Any]) -> set[str]:
        """Identify tasks collapsed into a sub-suite average.

        When a parent suite uses AVERAGE_OF_AVERAGES and a child suite averages
        its own tasks, the child's individual tasks are represented by the
        sub-suite row and should not appear separately.
        """
        from olmo_eval.evals.suites.registry import AggregationStrategy

        averaging = {
            AggregationStrategy.AVERAGE.value,
            AggregationStrategy.WEIGHTED_AVERAGE.value,
        }
        collapsed: set[str] = set()
        for suite_data in suites.values():
            if suite_data.get("parent_suite") and suite_data.get("aggregation") in averaging:
                collapsed.update(suite_data.get("tasks", []))
        return collapsed

    if multi_model:
        for model_name, model_data in results.get("models", {}).items():
            collapsed = _get_collapsed_tasks(model_data.get("suites", {}))
            for task_name, task_data in model_data.get("tasks", {}).items():
                if task_name not in collapsed:
                    add_task_row(f"{model_name}:{task_name}", task_data)
            for suite_name, suite_data in model_data.get("suites", {}).items():
                add_task_row(f"{model_name}:{suite_name}", suite_data)
    else:
        collapsed = _get_collapsed_tasks(results.get("suites", {}))
        for task_name, task_data in results["tasks"].items():
            if task_name not in collapsed:
                add_task_row(task_name, task_data)
        for suite_name, suite_data in results.get("suites", {}).items():
            add_task_row(suite_name, suite_data)

    console.print(table)
