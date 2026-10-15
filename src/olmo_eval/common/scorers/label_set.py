"""Label-set matching scorers."""

from dataclasses import dataclass

from ..types import Instance, LMOutput
from .base import Scorer


@dataclass(frozen=True, slots=True)
class LabelSetMatchScorer(Scorer):
    """Score 1.0 if the predicted label set equals the gold label set.

    Both the prediction and the gold answer are delimited label lists (for
    example ``A,C``); order and duplicates are ignored and labels are compared
    verbatim.
    """

    name: str = "label_set_match"
    separator: str = ","

    def score(self, instance: Instance, output: LMOutput) -> float:
        if instance.gold_answer is None:
            return 0.0
        gold = {label for label in instance.gold_answer.split(self.separator) if label}
        pred = output.extracted_answer
        predicted = set(str(pred).split(self.separator)) if pred else set()
        return 1.0 if predicted == gold else 0.0
