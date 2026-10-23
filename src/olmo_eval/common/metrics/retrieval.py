"""Ranking metrics for re-ranking tasks."""

from collections.abc import Sequence
from dataclasses import dataclass

from olmo_eval.common.metrics.base import Metric
from olmo_eval.common.scorers.base import Scorer
from olmo_eval.common.scorers.retrieval import NDCGScorer
from olmo_eval.common.types import Response


@dataclass(frozen=True, slots=True)
class NDCGMetric(Metric):
    """Mean NDCG@10 across queries.

    HELMET's headline metric for passage re-ranking (msmarco_rerank_psg): NDCG
    is computed per query and then averaged, so every query counts equally
    regardless of how many candidates it has.
    """

    name: str = "ndcg_at_10"
    scorer: type[Scorer] | Scorer = NDCGScorer

    def compute(self, responses: Sequence[Response]) -> float:
        if not responses:
            return 0.0
        scorer = self.scorer() if isinstance(self.scorer, type) else self.scorer
        total = sum(r.scores.get(scorer.name, 0.0) for r in responses)
        return total / len(responses)
