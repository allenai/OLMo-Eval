"""BitsPerByteScorer normalizes by the text the logprobs describe."""

from __future__ import annotations

import math

from olmo_eval.common.scorers import BitsPerByteScorer
from olmo_eval.common.types import Instance, LMOutput

_LOGPROBS = [{"token": "x", "logprob": -1.0} for _ in range(5)]


def test_bits_per_byte_uses_output_text() -> None:
    output = LMOutput(text="abcd", logprobs=_LOGPROBS)

    score = BitsPerByteScorer().score(Instance(question="Q", gold_answer="A"), output)

    assert score == 5.0 / (4 * math.log(2))


def test_bits_per_byte_uses_original_text_after_strip_thinking() -> None:
    original = "<think>reasoning</think>abcd"
    output = LMOutput(text="abcd", logprobs=_LOGPROBS, metadata={"original_text": original})

    score = BitsPerByteScorer().score(Instance(question="Q", gold_answer="A"), output)

    assert score == 5.0 / (len(original.encode("utf-8")) * math.log(2))
