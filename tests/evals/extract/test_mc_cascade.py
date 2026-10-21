"""Parity and long-response regression tests for the shared MC cascade."""

import subprocess
import sys

import pytest

from olmo_eval.evals.extract.answer_format import (
    OLMO_3_ANSWER_REGEX_TEMPLATES,
    extract_answer_with_format,
)
from olmo_eval.evals.tasks.gpqa import _extract_cot_answer as gpqa
from olmo_eval.evals.tasks.mmlu_pro import _extract_cot_answer as mmlu_pro


# Synthetic golden cases pin down answer priority and format scores.
def public_cascade(text):
    result = extract_answer_with_format(
        text,
        answer_format_regex=r"Therefore, the answer is \(([ABCD])\)",
        answer_regexes=(r"\(?([ABCD])\)?",),
        answer_regexes_templates=OLMO_3_ANSWER_REGEX_TEMPLATES,
    )
    return result._replace(answer=result.answer.replace("(", "").replace(")", ""))


@pytest.mark.parametrize("extract", (gpqa, mmlu_pro, public_cascade))
@pytest.mark.parametrize(
    ("text", "expected"),
    (
        ("Therefore, the answer is (C)", ("C", 1.0)),
        ("therefore the answer is: d", ("d", 1.0)),
        ("So the answer is (B).", ("B", 0.5)),
        ("The correct answer is: C", ("C", 0.5)),
        ("The answer is (A). Final: \\boxed{C}", ("A", 0.5)),
        ("Answer: d", ("d", 0.5)),
        ("C is correct", ("C", 0.5)),
        (r"\boxed{A}" + "\n" + r"\boxed{D}", ("D", 0.5)),
        ("Choose (C)", ("C", 0.5)),
        ("a", ("a", 0.5)),
        ("BAZ", ("A", 0.2)),
        ("纯中文回答", ("", 0.0)),
    ),
)
def test_mc_cascade_matches_reference_golden_results(extract, text, expected):
    assert extract(text) == expected


def test_long_cjk_responses_without_letters_complete():
    # The omitted greedy pattern was quadratic on this response. A subprocess
    # bounds the regression's runtime while allowing ample time for imports.
    code = """
from olmo_eval.evals.tasks.gpqa import _extract_cot_answer as gpqa
from olmo_eval.evals.tasks.mmlu_pro import _extract_cot_answer as mmlu_pro
import time
text = '推理' * 50000
started = time.perf_counter()
assert gpqa(text) == ('', 0.0)
assert mmlu_pro(text) == ('', 0.0)
assert time.perf_counter() - started < 2
"""
    subprocess.run([sys.executable, "-c", code], check=True, timeout=120)


@pytest.mark.parametrize("use_last", [False, True])
def test_empty_capable_answer_regex_keeps_reference_format_score(use_last):
    result = extract_answer_with_format(
        "x A!",
        answer_regexes=(r"(?=x)|A",),
        answer_regexes_templates=OLMO_3_ANSWER_REGEX_TEMPLATES,
        use_last_prefix_match=use_last,
    )
    assert result == ("A", 0.5)
