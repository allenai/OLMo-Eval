"""ROUGE-based metrics."""

from collections.abc import Sequence
from dataclasses import dataclass

from olmo_eval.common.metrics.base import Metric
from olmo_eval.common.scorers.base import Scorer
from olmo_eval.common.scorers.rouge import RougeLF1Scorer, RougeLRecallScorer
from olmo_eval.common.types import Response


@dataclass(frozen=True, slots=True)
class RougeLF1Metric(Metric):
    """Mean ROUGE-L F1 across all responses.

    Used for free-form long-context QA where the answer is a phrase rather
    than an exact string, e.g. HELMET's infbench_qa_eng.
    """

    name: str = "rougeL_f1"
    scorer: type[Scorer] = RougeLF1Scorer

    def compute(self, responses: Sequence[Response]) -> float:
        if not responses:
            return 0.0
        scorer_name = self.scorer().name
        total = sum(r.scores.get(scorer_name, 0.0) for r in responses)
        return total / len(responses)


@dataclass(frozen=True, slots=True)
class RougeLRecallMetric(Metric):
    """Mean ROUGE-L recall across all responses."""

    name: str = "rougeL_recall"
    scorer: type[Scorer] = RougeLRecallScorer

    def compute(self, responses: Sequence[Response]) -> float:
        if not responses:
            return 0.0
        scorer_name = self.scorer().name
        total = sum(r.scores.get(scorer_name, 0.0) for r in responses)
        return total / len(responses)
