"""Extraction functions for handling thinking/reasoning traces"""

import re

_THINK_END = "</think>"
_ANSWER_OPEN = re.compile(r"(?ms)^\s*<answer>\s*")
_ANSWER_CLOSE = re.compile(r"(?ms)</answer>\s*$")


def _after_last_think(text: str) -> str:
    """Text after the last ``</think>``, with ``<answer>`` tags removed.

    Equivalent to the reference harness's ``r1_style`` regexes
    (``re.sub("(?ms).*</think>", "", text)`` then the tag strips) but linear
    in the thinking prefix. Missing answer tags are checked literally before
    substitution. An answer tag after a long blank prefix can still make the
    whitespace substitution quadratic.
    """
    answer = text.rpartition(_THINK_END)[2] if _THINK_END in text else text
    # A multiline whitespace prefix can rescan long runs of blank lines when
    # the tag is absent. Literal checks make those negative scans linear.
    if "<answer>" in answer:
        answer = _ANSWER_OPEN.sub("", answer)
    if "</answer>" in answer:
        answer = _ANSWER_CLOSE.sub("", answer)
    return answer


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
