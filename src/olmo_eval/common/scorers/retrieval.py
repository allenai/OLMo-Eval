"""Ranking scorers for re-ranking tasks.

HELMET scores passage re-ranking with NDCG@10, computed per query with
`pytrec_eval` and then averaged. The formula is reimplemented here rather than
taking on a compiled dependency; it is verified to agree with `pytrec_eval`
exactly (linear gain, `rel / log2(rank + 1)`, ideal ranking from the full
qrels), so scores stay comparable to published numbers.
"""

import math
from dataclasses import dataclass
from typing import Any

from olmo_eval.common.scorers.base import Scorer
from olmo_eval.common.types import Instance, LMOutput


def _relevance_labels(instance: Instance) -> dict[str, int]:
    """Read a query's relevance judgements off the instance.

    Accepts either a mapping, or HELMET's `[[doc_id, label], ...]` pair list.
    """
    metadata: dict[str, Any] = instance.metadata or {}
    qrels = metadata.get("qrels", metadata.get("qrel"))
    if qrels is None:
        return {}
    if isinstance(qrels, dict):
        return {str(k): int(v) for k, v in qrels.items()}
    return {str(doc_id): int(label) for doc_id, label in qrels}


def ndcg_at_k(ranking: list[str], relevance: dict[str, int], k: int) -> float:
    """NDCG@k for one ranked list, matching trec_eval's `ndcg_cut`.

    Gain is linear in the relevance label and the ideal ranking is drawn from
    every judged document, not just the ones the model returned -- so failing
    to retrieve a relevant document costs score rather than being ignored.
    """
    if not relevance:
        return 0.0

    dcg = sum(
        relevance.get(doc_id, 0) / math.log2(rank + 2) for rank, doc_id in enumerate(ranking[:k])
    )
    ideal = sorted(relevance.values(), reverse=True)
    idcg = sum(rel / math.log2(rank + 2) for rank, rel in enumerate(ideal[:k]))

    if idcg == 0:
        return 0.0
    return dcg / idcg


@dataclass(frozen=True, slots=True)
class NDCGScorer(Scorer):
    """NDCG@k over a predicted document ranking.

    Expects `output.extracted_answer` to be the ranked list of document ids the
    model produced, and the query's relevance judgements in instance metadata
    under `qrels` (mapping) or `qrel` (HELMET's pair list).
    """

    name: str = "ndcg_at_10"
    k: int = 10

    def score(self, instance: Instance, output: LMOutput) -> float:
        ranking = output.extracted_answer
        if not isinstance(ranking, (list, tuple)):
            return 0.0
        return ndcg_at_k([str(doc_id) for doc_id in ranking], _relevance_labels(instance), self.k)
