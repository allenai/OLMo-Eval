"""Metric metadata and suite definitions resolved from the olmo-eval registries.

The ingest service cannot import olmo_eval, so the client resolves everything
that lives on Python objects (metric direction, display format, suite trees)
and sends it with the results.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from olmo_eval.common.logging import get_logger
from olmo_eval.common.metrics.base import display_format_for_name

logger = get_logger("upload.metadata")

# Lower-is-better metrics whose Metric class does not say so.
_LOWER_IS_BETTER_SUFFIXES = ("hallucination_rate", "parsing_error", "parsing_errors", "error_rate")


def strip_priority(spec: str) -> str:
    """Drop an "@priority" suffix from a task or suite spec."""
    return spec.rsplit("@", 1)[0] if "@" in spec else spec


def base_task_name(spec: str) -> str | None:
    """Registered base task for a spec, e.g. "gsm8k" for "gsm8k:cot:olmo3"."""
    from olmo_eval.evals.tasks.common import list_tasks

    available = set(list_tasks())
    parts = strip_priority(spec).split(":")
    for i in range(len(parts), 0, -1):
        candidate = ":".join(parts[:i])
        if candidate in available:
            return candidate
    return None


def _display_format(value: str | None) -> str | None:
    if value in ("percentage", "percent"):
        return "percent"
    if value == "raw":
        return "raw"
    return None


def _name_display_format(name: str) -> str:
    """The display format of a metric reported without a Metric object."""
    return _display_format(display_format_for_name(name.rsplit("__", 1)[-1])) or "raw"


def _force_direction(meta: dict[str, dict[str, Any]]) -> None:
    for name, entry in meta.items():
        if name.endswith(_LOWER_IS_BETTER_SUFFIXES):
            entry["higher_is_better"] = False


def metric_meta_from_metrics(
    metrics: Iterable[Any], metric_names: Iterable[str] = ()
) -> dict[str, dict[str, Any]]:
    """Build {metric_name: MetricMetaIn} from live Metric objects.

    Metric names reported by the task that have no Metric object (for example
    per-subset names such as ``domain__Law__accuracy``) inherit the metadata of
    the Metric whose name matches their last ``__`` segment, or get name-based
    defaults with unknown direction.
    """
    meta: dict[str, dict[str, Any]] = {}
    for metric in metrics:
        name = getattr(metric, "name", None)
        if not isinstance(name, str):
            continue
        try:
            entry = {
                "higher_is_better": bool(metric.higher_is_better()),
                "display_format": _display_format(metric.display_format()),
                "unit": metric.metric_unit(),
            }
        except Exception:
            continue
        meta[str(name)] = entry
    for name in metric_names:
        if name in meta:
            continue
        leaf = name.rsplit("__", 1)[-1]
        if leaf in meta:
            meta[name] = dict(meta[leaf])
        else:
            meta[name] = {
                "higher_is_better": None,
                "display_format": _name_display_format(name),
                "unit": None,
            }
    _force_direction(meta)
    return meta


def metric_meta_for_task(task: Any, metric_names: Iterable[str] = ()) -> dict[str, Any]:
    """Metric metadata for a prepared Task object. Returns {} on any error."""
    try:
        return metric_meta_from_metrics(task.config.metrics, metric_names)
    except Exception as e:
        logger.debug(f"Could not resolve metric metadata: {e}")
        return {}


def metric_meta_for_spec(spec: str, metric_names: Iterable[str] = ()) -> dict[str, Any]:
    """Metric metadata re-resolved from the task registry (used for old directories)."""
    try:
        from olmo_eval.evals.tasks.common import get_task

        task = get_task(strip_priority(spec))
    except Exception:
        return metric_meta_from_metrics((), metric_names) if metric_names else {}
    return metric_meta_for_task(task, metric_names)


def suites_containing(spec: str) -> list[str]:
    """Registered suites whose expanded tasks include this spec (static membership)."""
    from olmo_eval.evals.suites import get_suite, list_suites

    target = strip_priority(spec)
    found = []
    for name in list_suites():
        try:
            if target in get_suite(name).expanded_tasks:
                found.append(name)
        except Exception:
            continue
    return found


def _suite_children(name: str, contributing: Iterable[str]) -> list[dict[str, str]]:
    from olmo_eval.evals.suites import get_suite, suite_exists
    from olmo_eval.evals.suites.registry import Suite

    if suite_exists(name):
        children = []
        for child in get_suite(name).tasks:
            if isinstance(child, Suite):
                children.append({"type": "suite", "name": child.name})
            else:
                children.append({"type": "task", "name": child})
        return children
    return [{"type": "task", "name": strip_priority(spec)} for spec in contributing]


def _suite_description(name: str) -> str | None:
    from olmo_eval.evals.suites import get_suite, suite_exists

    if not suite_exists(name):
        return None
    return get_suite(name).description or None


def clean_metrics(metrics: Any) -> dict[str, dict[str, float | None]]:
    """Nested metrics with every non-numeric or non-finite value replaced by None."""
    cleaned: dict[str, dict[str, float | None]] = {}
    if not isinstance(metrics, Mapping):
        return cleaned
    for metric, scorers in metrics.items():
        if not isinstance(scorers, Mapping):
            continue
        cleaned[str(metric)] = {
            str(scorer): (
                float(value)
                if isinstance(value, (int, float)) and value == value and abs(value) != float("inf")
                else None
            )
            for scorer, value in scorers.items()
        }
    return cleaned


def _suite_result(raw_name: str, data: Mapping[str, Any]) -> dict[str, Any]:
    name = strip_priority(raw_name)
    metrics = clean_metrics(data.get("metrics"))
    score = metrics.get("primary_score", {}).get("average")
    parent = data.get("parent_suite")
    num_tasks = data.get("num_tasks")
    return {
        "name": name,
        "aggregation": str(data.get("aggregation") or "average"),
        "parent": strip_priority(parent) if isinstance(parent, str) else None,
        "description": _suite_description(name),
        "children": _suite_children(name, data.get("tasks") or []),
        "metrics": metrics,
        "primary_metric": data.get("primary_metric"),
        "score": score,
        "num_tasks": num_tasks if isinstance(num_tasks, int) and num_tasks >= 0 else None,
    }


def _nested_suites(name: str) -> list[tuple[str, str, str]]:
    """(child name, parent name, aggregation) for every suite nested under a registered suite."""
    from olmo_eval.evals.suites import get_suite, suite_exists
    from olmo_eval.evals.suites.registry import Suite

    if not suite_exists(name):
        return []
    out: list[tuple[str, str, str]] = []

    def walk(suite: Suite, stack: tuple[str, ...]) -> None:
        for child in suite.tasks:
            if isinstance(child, Suite) and child.name not in stack:
                out.append((child.name, suite.name, child.aggregation.value))
                walk(child, (*stack, child.name))

    walk(get_suite(name), (name,))
    return out


def suite_results(
    suite_aggregations: Mapping[str, Mapping[str, Any]],
    task_results: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Convert runner suite aggregations into SuiteResultIn payloads.

    The runner only reports nested suites under average_of_averages parents. The server
    needs a definition for every nested suite to build the tree, so the missing ones are
    added here, scored from ``task_results`` when given (the same way the runner scores
    a top-level suite) and otherwise sent as definitions with no score.
    """
    results = [_suite_result(raw, data) for raw, data in suite_aggregations.items()]
    seen = {r["name"] for r in results}
    for raw_name in suite_aggregations:
        suffix = f"@{raw_name.rsplit('@', 1)[1]}" if "@" in raw_name else ""
        for child, parent, aggregation in _nested_suites(strip_priority(raw_name)):
            if child in seen:
                continue
            seen.add(child)
            data: dict[str, Any] = {}
            if task_results is not None:
                computed = recompute_suite_aggregations([f"{child}{suffix}"], task_results)
                data = dict(computed.get(f"{child}{suffix}") or {})
            data["parent_suite"] = parent
            data.setdefault("aggregation", aggregation)
            results.append(_suite_result(child, data))
    return results


def recompute_suite_aggregations(
    task_specs: Iterable[str], tasks: Mapping[str, Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Rebuild suite aggregations from task results, the same way the runner does."""
    from olmo_eval.runners.processing.aggregation import compute_suite_aggregations

    try:
        return compute_suite_aggregations(list(task_specs), {k: dict(v) for k, v in tasks.items()})
    except Exception as e:
        logger.warning(f"Could not rebuild suite aggregations: {e}")
        return {}
