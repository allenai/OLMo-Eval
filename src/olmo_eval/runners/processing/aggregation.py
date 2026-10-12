"""Suite aggregation utilities for computing aggregate metrics across tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from olmo_eval.common.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ChildAverageResult:
    """Result from computing a child average."""

    metrics: dict[str, dict[str, float]]  # Nested: {metric: {scorer: value}}
    tasks: list[str]
    primary_score: float | None = None  # Average of primary metric values
    # If child was a Suite, include its info for separate reporting
    nested_suite: Any | None = None  # Suite or None
    nested_suite_key: str | None = None  # Key to use in results (with suffixes)


def _flatten_nested_metrics(metrics: dict[str, dict[str, float]]) -> dict[str, float]:
    """Flatten nested metrics to simple dict for aggregation.

    Converts {metric: {scorer: value}} to {"metric:scorer": value}.
    """
    result: dict[str, float] = {}
    for metric_name, scorers in metrics.items():
        for scorer_name, value in scorers.items():
            result[f"{metric_name}:{scorer_name}"] = value
    return result


def _unflatten_metrics(flat_metrics: dict[str, float]) -> dict[str, dict[str, float]]:
    """Convert flat metrics back to nested structure.

    Converts {"metric:scorer": value} to {metric: {scorer: value}}.
    """
    from olmo_eval.runners.processing.utils import parse_metric_key

    result: dict[str, dict[str, float]] = {}
    for key, value in flat_metrics.items():
        parsed = parse_metric_key(key)
        if parsed:
            metric_name, scorer_name = parsed
        else:
            metric_name, scorer_name = key, "default"
        if metric_name not in result:
            result[metric_name] = {}
        result[metric_name][scorer_name] = value
    return result


def _weighted_mean(values: list[float], weights: list[float]) -> float:
    """Instance-weighted mean. Callers guarantee one positive weight per value."""
    return sum(value * weight for value, weight in zip(values, weights, strict=True)) / sum(weights)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _task_weight(task_data: dict[str, Any]) -> float:
    """Instance count to weight one task by, or 0.0 when there is no usable count."""
    num_instances = task_data.get("num_instances")
    if not isinstance(num_instances, (int, float)) or isinstance(num_instances, bool):
        return 0.0
    return float(num_instances) if num_instances > 0 else 0.0


def _tasks_missing_instance_counts(
    task_specs: list[str],
    task_results: dict[str, dict[str, Any]],
) -> list[str]:
    return sorted(spec for spec in task_specs if _task_weight(task_results[spec]) <= 0)


def _log_omitted_weighted_suite(suite_name: str, task_specs: list[str]) -> None:
    logger.warning(
        f"Suite {suite_name}: no instance count for {', '.join(task_specs)}. "
        f"Omitting the aggregate - an instance-weighted suite reports a weighted "
        f"mean or no score at all."
    )


def _contributing_task_specs(
    task_specs: list[str],
    task_results: dict[str, dict[str, Any]],
) -> list[str]:
    """The specs that published metrics, in suite order."""
    return [spec for spec in task_specs if (task_results.get(spec) or {}).get("metrics")]


def _extract_primary_score(
    task_data: dict[str, Any],
) -> float | None:
    """Extract the primary metric value from a task result."""
    from olmo_eval.runners.processing.utils import extract_score_from_metrics

    primary_metric_key = task_data.get("primary_metric")
    if not primary_metric_key:
        return None
    return extract_score_from_metrics(task_data.get("metrics", {}), primary_metric_key)


def _log_tasks_excluded_from_suite(
    suite_name: str,
    task_specs: list[str],
    task_results: dict[str, dict[str, Any]],
) -> None:
    """Name the failed tasks that a suite average leaves out.

    A failed task publishes no metrics, so it drops out of the average on its
    own. That is deliberate - a score computed on a subset of instances must not
    pollute a suite average - but it silently changes what the average covers, so
    say it once per excluded task instead of letting the reader infer it from
    num_tasks.
    """
    for spec in task_specs:
        task_data = task_results.get(spec)
        if task_data is None or task_data.get("metrics"):
            continue
        error = task_data.get("error")
        if not error:
            continue
        logger.warning(
            f"Suite {suite_name}: excluding failed task {spec} from the aggregate "
            f"({error}). The suite average covers the remaining tasks only."
        )


def _compute_child_average(
    child: str | Any,  # str or Suite
    priority_suffix: str,
    task_results: dict[str, dict[str, Any]],
) -> ChildAverageResult | None:
    """Compute average metrics for a single child (task string or nested Suite).

    Returns:
        ChildAverageResult with metrics and task info, or None if no results found.
    """
    from olmo_eval.evals.suites.registry import AggregationStrategy, Suite

    if isinstance(child, Suite):
        # Child is a nested Suite - average all its expanded tasks, weighting
        # each by its instance count when the child asks to be weighted.
        weighted = child.aggregation == AggregationStrategy.WEIGHTED_AVERAGE
        child_specs = [f"{task_spec}{priority_suffix}" for task_spec in child.expand()]
        tasks_included = _contributing_task_specs(child_specs, task_results)
        if not tasks_included:
            return None

        if weighted:
            missing = _tasks_missing_instance_counts(tasks_included, task_results)
            if missing:
                _log_omitted_weighted_suite(child.name, missing)
                return None

        child_metrics: dict[str, list[float]] = {}
        child_metric_weights: dict[str, list[float]] = {}
        primary_scores: list[float] = []
        primary_score_weights: list[float] = []

        for full_task_spec in tasks_included:
            task_data = task_results[full_task_spec]

            # Flatten nested metrics for averaging
            flat_metrics = _flatten_nested_metrics(task_data.get("metrics", {}))
            if not flat_metrics:
                continue

            task_weight = _task_weight(task_data)
            for metric_key, value in flat_metrics.items():
                if metric_key not in child_metrics:
                    child_metrics[metric_key] = []
                    child_metric_weights[metric_key] = []
                child_metrics[metric_key].append(value)
                child_metric_weights[metric_key].append(task_weight)

            primary_value = _extract_primary_score(task_data)
            if primary_value is not None:
                primary_scores.append(primary_value)
                primary_score_weights.append(task_weight)

        if not child_metrics:
            return None

        averaged_flat = {
            name: (_weighted_mean(vals, child_metric_weights[name]) if weighted else _mean(vals))
            for name, vals in child_metrics.items()
        }
        averaged = _unflatten_metrics(averaged_flat)
        if not primary_scores:
            avg_primary = None
        elif weighted:
            avg_primary = _weighted_mean(primary_scores, primary_score_weights)
        else:
            avg_primary = _mean(primary_scores)
        # Build the key for this nested suite (with suffix)
        nested_key = f"{child.name}{priority_suffix}"
        return ChildAverageResult(
            metrics=averaged,
            tasks=tasks_included,
            primary_score=avg_primary,
            nested_suite=child,
            nested_suite_key=nested_key,
        )
    else:
        # Child is a task string - get its metrics directly
        full_task_spec = f"{child}{priority_suffix}"
        if full_task_spec not in task_results:
            return None

        task_data = task_results[full_task_spec]
        metrics = task_data.get("metrics", {})
        if not metrics:
            return None

        return ChildAverageResult(
            metrics=dict(metrics),
            tasks=[full_task_spec],
            primary_score=_extract_primary_score(task_data),
            nested_suite=None,
            nested_suite_key=None,
        )


def compute_suite_aggregations(
    task_specs: list[str],
    task_results: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Compute aggregated metrics for suites in the task specs.

    For each suite in task_specs, computes aggregated metrics based on the
    suite's aggregation strategy:
    - AVERAGE: Simple average of all expanded task scores
    - WEIGHTED_AVERAGE: Average of all expanded task scores, each weighted by
      the task's instance count
    - AVERAGE_OF_AVERAGES: Average over children, where nested suites are
      averaged first (each child gets equal weight)

    Handles specs with priority suffixes (@priority).
    When a suite has these suffixes, they are propagated to expanded task lookups.

    Args:
        task_specs: Original task specs (may include suite names with priority)
        task_results: Dict mapping task spec -> {"metrics": {...}, ...}

    Returns:
        Dict mapping suite name -> {"metrics": {...}, "tasks": [...], "aggregation": ...}
    """
    from olmo_eval.evals.suites import get_suite, suite_exists
    from olmo_eval.evals.suites.registry import AggregationStrategy

    suite_aggregations: dict[str, dict[str, Any]] = {}

    for spec in task_specs:
        # Parse out priority suffix (e.g., "suite@high" -> "suite", "@high")
        priority_suffix = ""
        base_spec = spec
        if "@" in spec:
            base_spec, priority = spec.rsplit("@", 1)
            priority_suffix = f"@{priority}"

        # Check if the base spec (without priority) is a suite
        if not suite_exists(base_spec):
            continue

        suite = get_suite(base_spec)
        if suite.aggregation == AggregationStrategy.NONE:
            continue

        _log_tasks_excluded_from_suite(
            spec,
            [f"{task_spec}{priority_suffix}" for task_spec in suite.expand()],
            task_results,
        )

        if suite.aggregation == AggregationStrategy.AVERAGE_OF_AVERAGES:
            # Average of averages: each child (task or nested suite) gets equal weight
            # Process each child separately, then average the child averages
            child_averages: dict[str, list[float]] = {}  # Flat "metric:scorer" -> values
            child_primary_scores: list[float] = []
            all_tasks_included: list[str] = []
            children_included = 0
            nested_suites_included: list[str] = []

            for child in suite.tasks:
                result = _compute_child_average(child, priority_suffix, task_results)
                if result is None:
                    continue

                all_tasks_included.extend(result.tasks)
                children_included += 1

                if result.primary_score is not None:
                    child_primary_scores.append(result.primary_score)

                # Flatten nested metrics for aggregation
                flat_metrics = _flatten_nested_metrics(result.metrics)
                for metric_key, value in flat_metrics.items():
                    if metric_key not in child_averages:
                        child_averages[metric_key] = []
                    child_averages[metric_key].append(value)

                # If this child is a nested Suite, also report its aggregation separately
                if result.nested_suite is not None and result.nested_suite_key:
                    nested_suites_included.append(result.nested_suite_key)
                    nested_result: dict[str, Any] = {
                        "metrics": result.metrics,
                        "tasks": result.tasks,
                        "num_tasks": len(result.tasks),
                        "aggregation": result.nested_suite.aggregation.value,
                        "parent_suite": spec,
                    }
                    if result.primary_score is not None:
                        nested_result["metrics"] = dict(result.metrics)
                        nested_result["metrics"]["primary_score"] = {
                            "average": result.primary_score
                        }
                        nested_result["primary_metric"] = "primary_score:average"
                    suite_aggregations[result.nested_suite_key] = nested_result

            if not child_averages:
                continue

            # Average the child averages (each child weighted equally)
            averaged_flat = {
                name: sum(values) / len(values) for name, values in child_averages.items()
            }
            aggregated_metrics = _unflatten_metrics(averaged_flat)

            suite_result: dict[str, Any] = {
                "metrics": aggregated_metrics,
                "tasks": all_tasks_included,
                "num_tasks": len(all_tasks_included),
                "num_children": children_included,
                "nested_suites": nested_suites_included,
                "aggregation": suite.aggregation.value,
            }

            if child_primary_scores:
                avg_primary = sum(child_primary_scores) / len(child_primary_scores)
                aggregated_metrics["primary_score"] = {"average": avg_primary}
                suite_result["primary_metric"] = "primary_score:average"

            suite_aggregations[spec] = suite_result
        else:
            # AVERAGE, WEIGHTED_AVERAGE or DISPLAY_ONLY: average all expanded
            # tasks, weighting each by its instance count for WEIGHTED_AVERAGE.
            weighted = suite.aggregation == AggregationStrategy.WEIGHTED_AVERAGE
            suite_specs = [f"{task_spec}{priority_suffix}" for task_spec in suite.expand()]
            tasks_included = _contributing_task_specs(suite_specs, task_results)
            if not tasks_included:
                continue

            if weighted:
                missing = _tasks_missing_instance_counts(tasks_included, task_results)
                if missing:
                    _log_omitted_weighted_suite(spec, missing)
                    continue

            suite_metrics: dict[str, list[float]] = {}  # Flat "metric:scorer" -> values
            suite_metric_weights: dict[str, list[float]] = {}
            task_primary_scores: list[float] = []
            primary_score_weights: list[float] = []

            for full_task_spec in tasks_included:
                task_data = task_results[full_task_spec]
                task_weight = _task_weight(task_data)

                # Flatten nested metrics for averaging
                flat_metrics = _flatten_nested_metrics(task_data.get("metrics", {}))
                for metric_key, value in flat_metrics.items():
                    if metric_key not in suite_metrics:
                        suite_metrics[metric_key] = []
                        suite_metric_weights[metric_key] = []
                    suite_metrics[metric_key].append(value)
                    suite_metric_weights[metric_key].append(task_weight)

                primary_value = _extract_primary_score(task_data)
                if primary_value is not None:
                    task_primary_scores.append(primary_value)
                    primary_score_weights.append(task_weight)

            if not suite_metrics:
                continue

            # Compute averages and unflatten back to nested structure
            averaged_flat = {
                name: (
                    _weighted_mean(values, suite_metric_weights[name])
                    if weighted
                    else _mean(values)
                )
                for name, values in suite_metrics.items()
            }
            aggregated_metrics = _unflatten_metrics(averaged_flat)

            avg_suite_result: dict[str, Any] = {
                "metrics": aggregated_metrics,
                "tasks": tasks_included,
                "num_tasks": len(tasks_included),
                "aggregation": suite.aggregation.value,
            }

            if task_primary_scores:
                avg_primary = (
                    _weighted_mean(task_primary_scores, primary_score_weights)
                    if weighted
                    else _mean(task_primary_scores)
                )
                aggregated_metrics["primary_score"] = {"average": avg_primary}
                avg_suite_result["primary_metric"] = "primary_score:average"

            suite_aggregations[spec] = avg_suite_result

    return suite_aggregations
