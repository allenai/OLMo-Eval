"""Extraction functions for handling thinking/reasoning traces"""

import re

_THINK_END = "</think>"
_ANSWER_OPEN = re.compile(r"(?ms)^\s*<answer>\s*")
_ANSWER_CLOSE = re.compile(r"(?ms)</answer>\s*$")


def _after_last_think(text: str) -> str:
    """Text after the last ``</think>``, with ``<answer>`` tags removed.

    Equivalent to the reference harness's ``r1_style`` regexes
    (``re.sub("(?ms).*</think>", "", text)`` then the tag strips) but linear
    in the answer length; the greedy regex backtracks quadratically on long
    answers.
    """
    answer = text.rpartition(_THINK_END)[2] if _THINK_END in text else text
    answer = _ANSWER_OPEN.sub("", answer)
    return _ANSWER_CLOSE.sub("", answer)


def extract_think_answer(text: str) -> str | None:
    """Deepseek-R1 style answer: drop the ``<think>`` trace, keep the rest.

    Text without a closing tag is returned unchanged (the trace is treated as
    the answer), matching the reference implementation.
    """
    return _after_last_think(text)


def extract_think_answer_only(text: str) -> str | None:
    """Like :func:`extract_think_answer`, but an empty string when no trace closed."""
    if _THINK_END not in text:
        return ""
    return _after_last_think(text)
