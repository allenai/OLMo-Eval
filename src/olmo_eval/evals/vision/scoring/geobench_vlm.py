"""GEOBench-VLM multiple-choice scoring.

The official evaluation scripts (``eval_geobenchvlm/*_cls_*.py`` in
The-AI-Alliance/GEO-Bench-VLM) keep a response's answer as its first character when that
character is one of ``A``-``E`` (after stripping the chat-role prefix and whitespace), and
record no answer otherwise. :func:`parse_geobench_answer` is that rule; an unparsed reply
scores 0.
"""

from __future__ import annotations

from dataclasses import dataclass

from olmo_eval.common.scorers.base import Scorer, set_scorer_result
from olmo_eval.common.types import Instance, LMOutput
from olmo_eval.evals.vision.scoring.common import response_text

VALID_CHOICES = frozenset("ABCDE")


def parse_geobench_answer(response: str) -> str | None:
    """The official answer rule: the stripped response's first character if it is A-E."""
    response = response.strip()
    return response[0] if response and response[0] in VALID_CHOICES else None


@dataclass(frozen=True, slots=True)
class GeobenchMcScorer(Scorer):
    """1.0 when the parsed letter equals the gold option letter, else 0.0.

    Records the parsed letter (``None`` when unparsed) as the scorer result.
    """

    name: str = "geobench_mc"

    def score(self, instance: Instance, output: LMOutput) -> float:
        pred = parse_geobench_answer(response_text(output))
        set_scorer_result(output, self.name, {"pred": pred})
        return float(pred is not None and pred == instance.metadata["answer"])
