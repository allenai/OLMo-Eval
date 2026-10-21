"""Direct tests for the regex-cascade answer extractor."""

from __future__ import annotations

import unittest

from olmo_eval.evals.extract import ExtractedAnswer, extract_answer_with_format

_FORMAT = r"Therefore, the final answer is \\boxed\{(.*)\}"
_ANSWERS = (r"[\s\S]*?\\boxed\{(.*?)\}[\s\S]*?", r"(.*)\.?")
_PREFIXES = (r"(?i)the final answer is", r"(?i)the answer is")
_TEMPLATES = (r"(?i)answer:\s*($ANS$)", r"(?i)\(($ANS$)\)")


class TestExtractAnswerWithFormat(unittest.TestCase):
    def test_format_regex_scores_full(self) -> None:
        result = extract_answer_with_format(
            "Therefore, the final answer is \\boxed{42}", answer_format_regex=_FORMAT
        )
        self.assertEqual(result, ExtractedAnswer("42", 1.0))

    def test_format_regex_last_match_wins(self) -> None:
        # On separate lines: greedy (.*) does not cross newlines, so there
        # are two matches and the last wins (reference behavior).
        text = (
            "Therefore, the final answer is \\boxed{41}\nTherefore, the final answer is \\boxed{42}"
        )
        result = extract_answer_with_format(text, answer_format_regex=_FORMAT)
        self.assertEqual(result.answer, "42")

    def test_template_branch_first_template_scores_full(self) -> None:
        result = extract_answer_with_format(
            "Answer: B",
            answer_regexes=(r"[A-D]",),
            answer_regexes_templates=_TEMPLATES,
        )
        self.assertEqual(result, ExtractedAnswer("B", 1.0))

    def test_template_branch_later_template_scores_half(self) -> None:
        result = extract_answer_with_format(
            "The right choice is (C) here.",
            answer_regexes=(r"[A-D]",),
            answer_regexes_templates=_TEMPLATES,
        )
        self.assertEqual(result, ExtractedAnswer("C", 0.5))

    def test_prefix_branch_first_prefix_scores_full(self) -> None:
        result = extract_answer_with_format(
            "The final answer is \\boxed{7}",
            answer_regexes=_ANSWERS,
            prefix_regexes=_PREFIXES,
        )
        self.assertEqual(result, ExtractedAnswer("7", 1.0))

    def test_prefix_branch_later_prefix_scores_half(self) -> None:
        result = extract_answer_with_format(
            "The answer is \\boxed{7}",
            answer_regexes=_ANSWERS,
            prefix_regexes=_PREFIXES,
        )
        self.assertEqual(result, ExtractedAnswer("7", 0.5))

    def test_raw_fallback_first_regex_scores_point_two(self) -> None:
        result = extract_answer_with_format(
            "So we get \\boxed{9} somewhere", answer_regexes=_ANSWERS
        )
        self.assertEqual(result, ExtractedAnswer("9", 0.2))

    def test_raw_fallback_later_regex_scores_point_one(self) -> None:
        # No \boxed{}, so only the catch-all second regex matches.
        result = extract_answer_with_format(
            "plain text", answer_regexes=(r"\\boxed\{(.*?)\}", r"(plain \w+)")
        )
        self.assertEqual(result, ExtractedAnswer("plain text", 0.1))

    def test_use_last_prefix_match_false_takes_first(self) -> None:
        text = "The answer is \\boxed{1}. Later: the answer is \\boxed{2}."
        last = extract_answer_with_format(text, answer_regexes=_ANSWERS, prefix_regexes=_PREFIXES)
        first = extract_answer_with_format(
            text, answer_regexes=_ANSWERS, prefix_regexes=_PREFIXES, use_last_prefix_match=False
        )
        self.assertEqual(last.answer, "2")
        self.assertEqual(first.answer, "1")

    def test_use_last_raw_match_false_takes_first(self) -> None:
        text = "\\boxed{1} then \\boxed{2}"
        last = extract_answer_with_format(text, answer_regexes=_ANSWERS)
        first = extract_answer_with_format(text, answer_regexes=_ANSWERS, use_last_raw_match=False)
        self.assertEqual(last.answer, "2")
        self.assertEqual(first.answer, "1")

    def test_nothing_matches(self) -> None:
        self.assertEqual(
            extract_answer_with_format("", answer_regexes=(r"\\boxed\{(.+?)\}",)),
            ExtractedAnswer("", 0.0),
        )


if __name__ == "__main__":
    unittest.main()
