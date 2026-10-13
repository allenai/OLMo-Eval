"""Tests for the reasoning-trace extractors."""

import re
import time

import pytest

from olmo_eval.evals.extract import (
    extract_think_answer,
    extract_think_answer_if_closed,
    extract_think_answer_only,
)

# The reference harness's r1_style regexes, kept here as the oracle.


def _reference(text: str) -> str:
    answer = re.sub("(?ms).*</think>", "", text)
    answer = re.sub("(?ms)^\\s*<answer>\\s*", "", answer)
    return re.sub("(?ms)</answer>\\s*$", "", answer)


@pytest.mark.parametrize(
    "text",
    [
        "<think>plan</think>\n\nFirst line.\n\nSecond line.",
        "<think>a</think> draft <think>b</think> 4",
        "<think>unterminated",
        "no trace at all",
        "",
        "<think>plan</think>\n<answer>4</answer>",
        "<think>plan</think>  <answer>\n4\n</answer>\n",
        "<think>x</think>\n\n<<title>>\n\nbody with `</answer>` mid-text </answer>",
    ],
)
def test_matches_reference_regexes(text: str) -> None:
    assert extract_think_answer(text) == _reference(text)


def test_keeps_text_after_trace_verbatim() -> None:
    assert extract_think_answer("<think>p</think>\n\n  4 ") == "\n\n  4 "


def test_only_variant_is_empty_without_closed_trace() -> None:
    assert extract_think_answer_only("<think>still going") == ""
    assert extract_think_answer_only("plain") == ""
    assert extract_think_answer_only("<think>p</think>\n4") == "\n4"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("<think>p</think>\n4", "\n4"),
        ("p</think>\n4", "\n4"),
        ("plain", "plain"),
        ("<answer>4</answer>", "4"),
        ("<think>still going", ""),
        ("<think>a</think> draft <think>b</think> 4", " 4"),
        ("<think>a</think> draft <think>still going", ""),
        ("", ""),
    ],
    ids=[
        "closed-trace",
        "opening-tag-in-prompt",
        "no-trace",
        "answer-tags",
        "unclosed-trace",
        "several-closed-traces",
        "reopened-trace",
        "empty",
    ],
)
def test_if_closed_variant_is_empty_while_a_trace_is_open(text: str, expected: str) -> None:
    assert extract_think_answer_if_closed(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "<think>p</think>\n4",
        "<think>p</think>",
        "<think>a</think> draft <think>still going",
        "<think>still going",
        "plain",
    ],
)
def test_if_closed_variant_is_unchanged_by_stripping_closed_traces_first(text: str) -> None:
    stripped = extract_think_answer(text) if "</think>" in text else text

    assert extract_think_answer_if_closed(stripped) == extract_think_answer_if_closed(text)


def test_linear_on_long_answers() -> None:
    text = "<think>" + "x" * 1000 + "</think>" + ("answer text. " * 8000)
    start = time.perf_counter()
    extract_think_answer(text)
    assert time.perf_counter() - start < 0.05
