"""ROUGE-based scorers.

Backed by Google Research's `rouge_score` package with `use_stemmer=True`,
matching how HELMET (and the InfiniteBench work it draws from) computes ROUGE,
so scores stay comparable to published numbers. Hand-rolling ROUGE-L is easy
to get subtly wrong -- tokenization and stemming both move the number -- which
is why the reference implementation is used instead.
"""

from dataclasses import dataclass
from functools import cache
from typing import Any

from olmo_eval.common.scorers.base import Scorer
from olmo_eval.common.types import Instance, LMOutput


@cache
def _get_rouge_scorer(rouge_type: str):
    """Build (and cache) a RougeScorer.

    Construction loads a stemmer, so it's cached rather than rebuilt per
    instance -- these scorers are called once per response.
    """
    from rouge_score import rouge_scorer

    return rouge_scorer.RougeScorer([rouge_type], use_stemmer=True)


def _reference_answers(instance: Instance) -> list[str]:
    """Collect every acceptable gold answer for an instance.

    Handles both multi-reference metadata conventions in this repo
    (`all_answers`, used by the SQuAD scorer, and `all_gold_answers`, used by
    RULER/HELMET tasks), falling back to the single `gold_answer`.
    """
    metadata: dict[str, Any] = instance.metadata or {}
    for key in ("all_answers", "all_gold_answers"):
        answers = metadata.get(key)
        if answers:
            return [str(a) for a in answers]

    gold = instance.gold_answer
    if gold is None:
        return []
    if isinstance(gold, (list, tuple)):
        return [str(a) for a in gold]
    return [str(gold)]


def _prediction_candidates(output: LMOutput) -> list[str]:
    """The extracted answer and the raw generation, deduplicated.

    HELMET scores both and keeps the better (its `default_post_process` takes
    the max of metrics on the raw and the parsed output), so scoring only the
    extracted answer would under-credit a multi-line generation whose
    answer-prefix parse picked the wrong line.
    """
    candidates = []
    for value in (output.extracted_answer, output.text):
        if value is None:
            continue
        text = str(value)
        if text and text not in candidates:
            candidates.append(text)
    return candidates


@dataclass(frozen=True, slots=True)
class RougeLF1Scorer(Scorer):
    """Max ROUGE-L F-measure over all reference answers.

    Taking the max over references mirrors HELMET's `calculate_metrics`, which
    scores against every acceptable answer and keeps the best; taking it over
    the raw and extracted generations mirrors its `default_post_process`.
    """

    name: str = "rougeL_f1"

    def score(self, instance: Instance, output: LMOutput) -> float:
        predictions = _prediction_candidates(output)
        references = _reference_answers(instance)
        if not predictions or not references:
            return 0.0

        scorer = _get_rouge_scorer("rougeL")
        return max(
            scorer.score(target=reference, prediction=prediction)["rougeL"].fmeasure
            for reference in references
            for prediction in predictions
        )


@dataclass(frozen=True, slots=True)
class RougeLRecallScorer(Scorer):
    """Max ROUGE-L recall over all reference answers.

    HELMET reports recall alongside F1 for some summarization-style tasks
    (e.g. qmsum uses rougeL_recall), so both are provided here.
    """

    name: str = "rougeL_recall"

    def score(self, instance: Instance, output: LMOutput) -> float:
        predictions = _prediction_candidates(output)
        references = _reference_answers(instance)
        if not predictions or not references:
            return 0.0

        scorer = _get_rouge_scorer("rougeL")
        return max(
            scorer.score(target=reference, prediction=prediction)["rougeL"].recall
            for reference in references
            for prediction in predictions
        )
