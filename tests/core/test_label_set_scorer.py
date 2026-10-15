"""Tests for LabelSetMatchScorer."""

import pytest

from olmo_eval.common.scorers import LabelSetMatchScorer
from olmo_eval.common.types import Instance, LMOutput


def _instance(gold: str | None) -> Instance:
    return Instance(question="q", gold_answer=gold, metadata={"id": "x"})


class TestLabelSetMatchScorer:
    """Tests for order-insensitive label-set scoring."""

    @pytest.mark.parametrize(
        ("pred", "gold", "expected"),
        [
            ("A,C", "A,C", 1.0),
            ("C,A", "A,C", 1.0),
            ("A,C,C", "A,C", 1.0),
            ("A", "A,C", 0.0),
            ("A,B,C", "A,C", 0.0),
            ("A, C", "A,C", 0.0),
            ("a,c", "A,C", 0.0),
            (None, "A,C", 0.0),
            ("", "A,C", 0.0),
        ],
    )
    def test_score(self, pred, gold, expected):
        output = LMOutput(text="", extracted_answer=pred)
        assert LabelSetMatchScorer().score(_instance(gold), output) == expected

    def test_missing_gold(self):
        output = LMOutput(text="", extracted_answer="A")
        assert LabelSetMatchScorer().score(_instance(None), output) == 0.0

    def test_custom_separator(self):
        output = LMOutput(text="", extracted_answer="C;A")
        assert LabelSetMatchScorer(separator=";").score(_instance("A;C"), output) == 1.0

    def test_name(self):
        assert LabelSetMatchScorer().name == "label_set_match"
