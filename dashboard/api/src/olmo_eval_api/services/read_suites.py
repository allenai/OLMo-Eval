"""Suite definitions and subject-level suite aggregation for the read endpoints."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from sqlalchemy.ext.asyncio import AsyncSession

from olmo_eval_api.services.suites import (
    NO_SCORE_AGGREGATIONS,
    SuiteNode,
    aggregate,
    build_tree,
    current_suite_defs,
    leaf_weights,
    leaves,
    propagate_stderr,
)

SuiteDefs = dict[str, tuple[str, list[dict], str | None, str]]


def suite_generation_counts(
    node: SuiteNode, tasks: Mapping[str, Mapping[Any, Any]]
) -> dict[str, int] | None:
    """Sum generation diagnostics over the suite's contributing task results."""
    total: dict[str, int] = {}
    for name in leaves(node):
        task = tasks.get(name)
        if task is None or task.get("score") is None:
            continue
        counts = task.get("generation_counts")
        if counts is not None:
            for key, value in counts.items():
                total[key] = total.get(key, 0) + value
    return total or None


async def load_defs(session: AsyncSession) -> SuiteDefs:
    """Current definition of every suite: name -> (aggregation, children, description, hash)."""
    return await current_suite_defs(session)


def tree(name: str, defs: SuiteDefs) -> SuiteNode:
    return build_tree(name, {k: (v[0], v[1]) for k, v in defs.items()})


def definition_hashes(nodes: Iterable[SuiteNode], defs: SuiteDefs) -> dict[str, str | None]:
    """Definition hash of every suite in the trees (None for an undefined suite), for cache
    keys: a response computed from these trees is stale once any of them changes."""
    out: dict[str, str | None] = {}

    def walk(node: SuiteNode) -> None:
        if node.type != "suite":
            return
        out[node.name] = defs[node.name][3] if node.name in defs else None
        for child in node.children:
            walk(child)

    for node in nodes:
        walk(node)
    return out


@dataclass
class SuiteScore:
    score: float | None
    stderr: float | None
    n_instances: int
    present: int
    missing: int
    display_format: Literal["percent", "raw"]
    higher_is_better: bool | None


def primary_gap_metric_keys(
    node: SuiteNode, subjects: Iterable[Mapping[str, Mapping[Any, Any]]]
) -> dict[str, str | None] | None:
    """Choose one complete GAP primary metric for a population of subjects."""
    if node.aggregation != "gap":
        return None
    names = leaves(node)
    # Population order fixes one identity; metrics from other identities remain unscored.
    for tasks in subjects:
        if all(name in tasks and tasks[name].get("score") is not None for name in names):
            keys = {tasks[name].get("primary_metric") for name in names}
            if len(keys) == 1 and None not in keys:
                return {name: tasks[name]["primary_metric"] for name in names}
    return dict.fromkeys(names)


def selected_metric_meta(task: Mapping[Any, Any], key: str | None) -> Mapping[str, Any]:
    """Metadata for a selected task metric, with primary-only legacy fallback."""
    metadata = (task.get("metric_meta") or {}).get(key)
    if key == task.get("primary_metric"):
        return {**task, **(metadata or {})}
    return metadata or {}


def score_suite(
    node: SuiteNode,
    trs: Mapping[str, Mapping[Any, Any]],
    scores: Mapping[str, float | None] | None = None,
    stderrs: Mapping[str, float | None] | None = None,
    metric_keys: Mapping[str, str | None] | None = None,
) -> SuiteScore:
    """Aggregate a subject's task results over a suite tree.

    ``scores``/``stderrs`` override the task results' primary score and stderr (used for a
    selected metric). ``metric_keys`` identifies each selected metric so gap
    suites can verify that both tasks report the same one.
    """
    names = leaves(node)
    present = {n: trs[n] for n in names if n in trs}
    selected = {
        n: metric_keys.get(n) if metric_keys is not None else tr.get("primary_metric")
        for n, tr in present.items()
    }
    metadata = {n: selected_metric_meta(tr, selected[n]) for n, tr in present.items()}
    leaf_scores = {
        n: (
            scores[n]
            if scores is not None and n in scores
            else tr["score"]
            if metric_keys is None or selected[n] == tr.get("primary_metric")
            else (tr.get("metrics") or {}).get(selected[n])
        )
        for n, tr in present.items()
    }
    leaf_ses = {
        n: (stderrs[n] if stderrs is not None and n in stderrs else metadata[n].get("stderr"))
        for n in present
    }
    n_inst = {n: int(tr["num_instances"] or 0) for n, tr in present.items()}
    score = aggregate(node, leaf_scores, n_inst)
    if node.aggregation == "gap":
        metrics = {
            metric_keys.get(name) if metric_keys is not None else tr.get("primary_metric")
            for name, tr in present.items()
            if leaf_scores[name] is not None
        }
        if len(metrics) != 1 or None in metrics:
            score = None
    scored = {n for n, value in leaf_scores.items() if value is not None}
    weights = leaf_weights(node, n_inst, scored, metric_keys=metric_keys)
    stderr = None
    if score is not None and node.aggregation not in NO_SCORE_AGGREGATIONS:
        stderr = propagate_stderr(weights, leaf_ses)
    metadata_names = weights if node.aggregation == "gap" else scored
    scored_metadata = [metadata[name] for name in metadata_names]
    formats = {meta.get("display_format", "raw") for meta in scored_metadata}
    directions = {meta.get("higher_is_better") for meta in scored_metadata}
    return SuiteScore(
        score=score,
        stderr=stderr,
        n_instances=sum(n_inst.values()),
        present=len(present),
        missing=len(names) - len(present),
        display_format="percent" if formats == {"percent"} else "raw",
        higher_is_better=directions.pop() if len(directions) == 1 else None,
    )
