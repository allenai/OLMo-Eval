"""Substring matching scorers."""

from dataclasses import dataclass

from ..types import Instance, LMOutput
from .base import Scorer, _squad_normalize_answer


@dataclass(frozen=True, slots=True)
class SubstringRecallScorer(Scorer):
    """Substring recall scorer.

    Computes recall as the fraction of gold answer strings that appear
    (case-insensitive substring match) in the prediction.

    This scorer handles both single and multiple gold answers:
    - Single answer: Returns 1.0 if found, 0.0 otherwise
    - Multiple answers (list): Returns fraction of answers found

    Useful for tasks where the model needs to recall specific information,
    such as RULER (NIAH, VT, CWE, FWE) and similar retrieval tasks.
    """

    name: str = "substring_recall"
    case_sensitive: bool = False

    def score(self, instance: Instance, output: LMOutput) -> float:
        """Score recall of gold answers in the prediction.

        Args:
            instance: Instance with gold_answer as list of strings
            output: LMOutput with text prediction

        Returns:
            Fraction of gold answers found in prediction (0.0 to 1.0)
        """
        if instance.gold_answer is None or output.text is None:
            return 0.0

        # Handle both list and single string gold answers
        if isinstance(instance.gold_answer, list):
            gold_answers = instance.gold_answer
        else:
            gold_answers = [str(instance.gold_answer)]

        if len(gold_answers) == 0:
            return 0.0

        # Apply case sensitivity
        prediction = output.text if self.case_sensitive else output.text.lower()
        matches = sum(
            1
            for answer in gold_answers
            if (str(answer) if self.case_sensitive else str(answer).lower()) in prediction
        )

        return matches / len(gold_answers)


@dataclass(frozen=True, slots=True)
class SubstringExactMatchScorer(Scorer):
    """Substring exact match: 1.0 if *any* gold answer appears in the prediction.

    Distinct from `SubstringRecallScorer`, which returns the *fraction* of gold
    answers found. That difference matters whenever the gold answers are
    aliases for one another rather than facts to be jointly recalled: an
    open-domain QA answer like Natural Questions' carries several surface forms
    of a single answer, and producing one of them is fully correct, not 1/N
    correct.

    Normalization (lowercase, strip punctuation, drop articles, collapse
    whitespace) matches HELMET's `normalize_answer`, so this reproduces its
    `substring_exact_match` metric for the RAG tasks.
    """

    name: str = "substring_exact_match"

    def score(self, instance: Instance, output: LMOutput) -> float:
        if output.text is None:
            return 0.0

        metadata = instance.metadata or {}
        golds = metadata.get("all_gold_answers")
        if not golds:
            gold = instance.gold_answer
            if gold is None:
                return 0.0
            golds = gold if isinstance(gold, (list, tuple)) else [gold]
        if not golds:
            return 0.0

        prediction = _squad_normalize_answer(str(output.text))
        return float(any(_squad_normalize_answer(str(gold)) in prediction for gold in golds))
