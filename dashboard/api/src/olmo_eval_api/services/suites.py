"""Suite trees: leaves, weights, aggregation and stderr propagation (spec 2.6).

Also the stored suite definitions (``suite_defs``) that ingest writes and the read endpoints
build trees from.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from olmo_eval_api.services.derive import canonical_json, sha256_hex

NO_SCORE_AGGREGATIONS = {"none", "display_only"}


@dataclass(frozen=True)
class SuiteNode:
    name: str
    type: Literal["task", "suite"]
    aggregation: str | None
    children: tuple[SuiteNode, ...]


def build_tree(root: str, defs: Mapping[str, tuple[str, list[dict]]]) -> SuiteNode:
    """Build the tree under ``root``.

    ``defs`` maps suite name -> (aggregation, children[{type, name}]). Unknown child suites
    become empty suites. Cycles raise ValueError.
    """

    def build(name: str, stack: tuple[str, ...]) -> SuiteNode:
        if name in stack:
            raise ValueError(f"suite cycle: {' -> '.join((*stack, name))}")
        if name not in defs:
            return SuiteNode(name=name, type="suite", aggregation=None, children=())
        aggregation, children = defs[name]
        nodes = []
        for child in children:
            if child.get("type") == "suite":
                nodes.append(build(child["name"], (*stack, name)))
            else:
                nodes.append(
                    SuiteNode(name=child["name"], type="task", aggregation=None, children=())
                )
        return SuiteNode(name=name, type="suite", aggregation=aggregation, children=tuple(nodes))

    return build(root, ())


def leaves(node: SuiteNode) -> list[str]:
    """Task names in pre-order, deduplicated."""
    out: list[str] = []
    seen: set[str] = set()

    def walk(n: SuiteNode) -> None:
        if n.type == "task":
            if n.name not in seen:
                seen.add(n.name)
                out.append(n.name)
            return
        for child in n.children:
            walk(child)

    walk(node)
    return out


def _flat_weights(
    node: SuiteNode, n_instances: Mapping[str, int], present: set[str]
) -> dict[str, float]:
    """Weights over the present leaves: by instance count for weighted_average, else equal."""
    names = [name for name in leaves(node) if name in present]
    if not names:
        return {}
    if node.aggregation == "weighted_average":
        total = sum(max(n_instances.get(name, 0), 0) for name in names)
        if total > 0:
            return {name: max(n_instances.get(name, 0), 0) / total for name in names}
    return {name: 1.0 / len(names) for name in names}


def leaf_weights(
    node: SuiteNode, n_instances: Mapping[str, int], present: set[str]
) -> dict[str, float]:
    """Weights over present leaves summing to 1 (empty when no leaf is present).

    These match olmo-eval's compute_suite_aggregations:

    - average, display_only, none: equal weight for every present leaf in the expanded suite.
    - weighted_average: proportional to n_instances over the present leaves.
    - average_of_averages: equal weight for each direct child with a present leaf. A child suite
      spreads its share over its own leaves, by instance count when it is weighted_average and
      equally otherwise.
    """
    if node.type == "task":
        return {node.name: 1.0} if node.name in present else {}
    if node.aggregation != "average_of_averages":
        return _flat_weights(node, n_instances, present)
    child_weights = []
    for child in node.children:
        if child.type == "task":
            weights = {child.name: 1.0} if child.name in present else {}
        else:
            weights = _flat_weights(child, n_instances, present)
        if weights:
            child_weights.append(weights)
    if not child_weights:
        return {}
    out: dict[str, float] = {}
    share = 1.0 / len(child_weights)
    for weights in child_weights:
        for name, w in weights.items():
            out[name] = out.get(name, 0.0) + share * w
    return out


def aggregate(
    node: SuiteNode, scores: Mapping[str, float | None], n_instances: Mapping[str, int]
) -> float | None:
    if node.type == "suite" and node.aggregation in NO_SCORE_AGGREGATIONS:
        return None
    present = {name for name, score in scores.items() if score is not None}
    weights = leaf_weights(node, n_instances, present)
    if not weights:
        return None
    total = 0.0
    for name, w in weights.items():
        value = scores[name]
        assert value is not None
        total += w * value
    return total


def propagate_stderr(
    weights: Mapping[str, float], stderrs: Mapping[str, float | None]
) -> float | None:
    if not weights:
        return None
    total = 0.0
    for name, w in weights.items():
        se = stderrs.get(name)
        if se is None:
            return None
        total += w * w * se * se
    return math.sqrt(total)


def walk_suites(node: SuiteNode, depth: int = 0, parent: str | None = None):
    """Yield (node, depth, parent) for suite nodes in pre-order."""
    if node.type != "suite":
        return
    yield node, depth, parent
    for child in node.children:
        yield from walk_suites(child, depth + 1, node.name)


async def current_suite_defs(
    session: AsyncSession, names: set[str] | None = None
) -> dict[str, tuple[str, list[dict], str | None, str]]:
    """Current definition per suite name: (aggregation, children, description, hash)."""
    sql = """
        SELECT DISTINCT ON (suite_name) suite_name, aggregation, children, description,
               definition_hash
        FROM suite_defs {where}
        ORDER BY suite_name, last_seen_at DESC
    """
    if names is None:
        rows = (await session.execute(text(sql.format(where="")))).all()
    else:
        if not names:
            return {}
        rows = (
            await session.execute(
                text(sql.format(where="WHERE suite_name = ANY(:names)")), {"names": list(names)}
            )
        ).all()
    return {r[0]: (r[1], r[2], r[3], r[4]) for r in rows}


def suite_definition_hash(aggregation: str, children: list[dict]) -> str:
    payload = {"aggregation": aggregation, "children": children}
    return sha256_hex(canonical_json(payload))[:16]
