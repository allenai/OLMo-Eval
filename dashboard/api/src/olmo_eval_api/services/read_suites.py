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


def score_suite(
    node: SuiteNode,
    trs: Mapping[str, Mapping[Any, Any]],
    scores: Mapping[str, float | None] | None = None,
    stderrs: Mapping[str, float | None] | None = None,
) -> SuiteScore:
    """Aggregate a subject's task results over a suite tree.

    ``scores``/``stderrs`` override the task results' primary score and stderr (used for a
    non-primary metric).
    """
    names = leaves(node)
    present = {n: trs[n] for n in names if n in trs}
    leaf_scores = {
        n: (scores[n] if scores is not None and n in scores else tr["score"])
        for n, tr in present.items()
    }
    leaf_ses = {
        n: (stderrs[n] if stderrs is not None and n in stderrs else tr["stderr"])
        for n, tr in present.items()
    }
    n_inst = {n: int(tr["num_instances"] or 0) for n, tr in present.items()}
    score = aggregate(node, leaf_scores, n_inst)
    stderr = None
    if score is not None and node.aggregation not in NO_SCORE_AGGREGATIONS:
        scored = {n for n, v in leaf_scores.items() if v is not None}
        stderr = propagate_stderr(leaf_weights(node, n_inst, scored), leaf_ses)
    formats = {tr["display_format"] for tr in present.values()}
    directions = {tr["higher_is_better"] for tr in present.values()}
    return SuiteScore(
        score=score,
        stderr=stderr,
        n_instances=sum(n_inst.values()),
        present=len(present),
        missing=len(names) - len(present),
        display_format="percent" if formats == {"percent"} else "raw",
        higher_is_better=directions.pop() if len(directions) == 1 else None,
    )
