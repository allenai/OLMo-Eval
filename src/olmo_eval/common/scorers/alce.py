"""Answer-correctness scorers for ALCE (HELMET's Cite category).

Ports the non-citation metrics from ALCE's `eval_alce.py`: `str_em` for ASQA
and `qampari_rec_top5` for QAMPARI. These score *what* the model answered.
ALCE's other half -- whether each claim is actually supported by the sources it
cites -- is measured by AutoAIS, an NLI model, and is not implemented here.

Both metrics run against a normalized generation: newlines collapsed to spaces,
chat end-markers dropped, and inline `[1][2]` citation markers stripped, so a
well-cited answer is not penalized for carrying its citations inline.

Scores are proportions in [0, 1]; ALCE reports the same numbers scaled to
0-100, so multiply by 100 to compare against published figures.
"""

import re
from dataclasses import dataclass
from typing import Any

from olmo_eval.common.scorers.base import Scorer, _squad_normalize_answer
from olmo_eval.common.types import Instance, LMOutput


def remove_citations(text: str) -> str:
    """Strip inline `[1]`-style citation markers. Verbatim from ALCE's utils."""
    return re.sub(r"\[\d+", "", re.sub(r" \[\d+", "", text)).replace(" |", "").replace("]", "")


def normalize_generation(text: str) -> str:
    """Apply ALCE's pre-scoring cleanup to a generation.

    Mirrors the preprocessing in `eval_alce.py`'s main(): collapse newlines,
    drop the chat end-marker, then remove citations.
    """
    text = re.sub(r"\n+", " ", text or "")
    text = text.replace("<|im_end|>", "")
    return remove_citations(text)


def exact_presence(short_answers: list[str], context: str) -> bool:
    """True if any acceptable short answer appears in the context.

    Port of ALCE's `exact_presence`; the normalization matches HELMET's
    `normalize_answer`.
    """
    normalized_context = _squad_normalize_answer(context)
    return any(_squad_normalize_answer(sa) in normalized_context for sa in short_answers)


@dataclass(frozen=True, slots=True)
class AlceStrEmScorer(Scorer):
    """ALCE's STR-EM: the fraction of an ASQA question's sub-answers covered.

    ASQA questions are ambiguous and have several disambiguated sub-questions,
    each with its own acceptable short answers. A response scores the fraction
    of those sub-questions it answers, so partial coverage gets partial credit
    rather than being all-or-nothing.
    """

    name: str = "str_em"

    def score(self, instance: Instance, output: LMOutput) -> float:
        qa_pairs: list[dict[str, Any]] = (instance.metadata or {}).get("qa_pairs") or []
        if not qa_pairs:
            return 0.0

        generation = normalize_generation(output.text or "")
        hits = sum(
            1.0 for pair in qa_pairs if exact_presence(pair.get("short_answers") or [], generation)
        )
        return hits / len(qa_pairs)


@dataclass(frozen=True, slots=True)
class AlceQampariRecTop5Scorer(Scorer):
    """ALCE's QAMPARI recall@top-5.

    QAMPARI questions have many valid answers, so the model emits a
    comma-separated list. Recall is capped at five: a question with twenty gold
    answers is satisfied by finding five, which keeps questions with long answer
    sets from dominating the average.
    """

    name: str = "qampari_rec_top5"

    def score(self, instance: Instance, output: LMOutput) -> float:
        answers: list[list[str]] = (instance.metadata or {}).get("qampari_answers") or []
        if not answers:
            return 0.0

        generation = normalize_generation(output.text or "")
        # the model answers as a comma-separated list; trailing punctuation is
        # stripped before splitting, as upstream does
        predictions = [
            _squad_normalize_answer(part.strip())
            for part in generation.rstrip().rstrip(".").rstrip(",").split(",")
        ]
        predictions = [p for p in predictions if p]

        normalized_answers = [[_squad_normalize_answer(a) for a in alias] for alias in answers]
        found = sum(
            1 for alias_group in normalized_answers if any(a in predictions for a in alias_group)
        )
        return min(5, found) / min(5, len(normalized_answers))
