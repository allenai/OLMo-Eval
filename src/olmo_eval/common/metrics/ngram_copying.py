"""Metrics for n-gram copying: bits per byte restricted to repeated-span positions."""

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

from olmo_eval.common.scorers import NGramCopyingBPBScorer
from olmo_eval.common.types import Response

from .base import Metric

#: Default set of n-gram-length thresholds. Add new values here to extend coverage.
NGRAM_COPYING_K_VALUES: tuple[int, ...] = (1, 2, 3, 4, 5)


def _ngram_copying_masked_totals(
    response: Response, scorer: NGramCopyingBPBScorer
) -> tuple[float, int] | None:
    """Look up the masked (logprob, bytes) totals for a response's document output."""
    if not response.outputs:
        return None
    return scorer.masked_totals(response.outputs[0])


@dataclass(frozen=True, slots=True)
class NGramCopyingBPBMetricByteAvg(Metric):
    """Corpus-level BPB restricted to length-k+ repeated n-grams.

    Sums masked logprobs and masked bytes independently across all documents in
    the corpus, then divides once. Documents with no positions meeting the
    threshold contribute nothing to either sum, so they're handled without any
    special-casing rather than being explicitly excluded.
    """

    name: str = field(init=False)
    k: int = 1
    scorer: NGramCopyingBPBScorer = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", f"ngram_copying_bpb_byte_avg_k{self.k}")
        object.__setattr__(self, "scorer", NGramCopyingBPBScorer(k=self.k))

    def compute(self, responses: Sequence[Response]) -> float:
        total_logprob = 0.0
        total_bytes = 0
        for response in responses:
            totals = _ngram_copying_masked_totals(response, self.scorer)
            if totals is None:
                continue
            logprob, num_bytes = totals
            if num_bytes == 0:
                continue
            total_logprob += logprob
            total_bytes += num_bytes

        if total_bytes == 0:
            return 0.0
        return -total_logprob / (total_bytes * math.log(2))

    def compute_instance(self, response: Response) -> float | None:
        totals = _ngram_copying_masked_totals(response, self.scorer)
        if totals is None:
            return None
        total_logprob, total_bytes = totals
        if total_bytes == 0:
            return None
        return -total_logprob / (total_bytes * math.log(2))

    def supports_scorer_fallback(self) -> bool:
        return False

    def higher_is_better(self) -> bool:
        return False
