"""Tests for the SciBench task.

The reference grader is the specification, so most of these assert behaviour copied
from https://github.com/mandyyyyii/scibench: a 5 percent relative tolerance, a unit
field that can carry a power-of-ten factor, and a boxed-answer parser that keys off
the substring ``oxed``.
"""

from __future__ import annotations

import unittest

from olmo_eval.common.types import Instance, LMOutput, RequestType
from olmo_eval.evals.tasks.common import get_task
from olmo_eval.evals.tasks.scibench import (
    SciBenchToleranceScorer,
    collapse_scientific_notation,
    parse_answer,
)

_SCORER = SciBenchToleranceScorer()


def _doc(**overrides: object) -> dict[str, object]:
    doc = {
        "problem_text": "Predict the pressure exerted by the ethane.",
        "answer_latex": " 50.7",
        "answer_number": "50.7",
        "unit": "$\\mathrm{atm}$ ",
        "problemid": " e1.17(a)(a)",
        "source": "atkins",
        "solution": "",
        "comment": "",
    }
    doc.update(overrides)
    return doc


def _boxed(value: str) -> LMOutput:
    return LMOutput(
        text=f"Working it through. The answer is therefore \\boxed{{{value}}}.",
        metadata={},
    )


class TestSciBenchRegistration(unittest.TestCase):
    def test_registered_with_expected_config(self) -> None:
        task = get_task("scibench")
        self.assertEqual(task.request_type, RequestType.CHAT)
        self.assertEqual(task.config.num_fewshot, 0)
        self.assertEqual(task.config.sampling_params.temperature, 0.0)
        self.assertIsNone(task.config.sampling_params.max_tokens)
        self.assertTrue(task.config.strip_thinking)
        primary = task.config.get_primary_metric()
        assert primary is not None
        self.assertEqual(primary.name, "exact_match")

    def test_data_source_is_pinned(self) -> None:
        source = get_task("scibench").config.data_source
        assert source is not None
        self.assertEqual(source.path, "xw27/scibench")
        self.assertEqual(source.revision, "93931252bc1b71d495e67390235940643d926958")

    def test_per_textbook_subset_metrics_are_present(self) -> None:
        names = {m.name for m in get_task("scibench").config.metrics}
        self.assertIn("macro_accuracy", names)
        self.assertIn("source__atkins", names)
        self.assertIn("source__thermo", names)


class TestSciBenchProcessDoc(unittest.TestCase):
    def setUp(self) -> None:
        self.task = get_task("scibench")

    def test_prompt_states_the_unit_and_keeps_reference_spacing(self) -> None:
        instance = self.task.process_doc(_doc())
        assert instance is not None
        self.assertEqual(
            instance.question,
            "Predict the pressure exerted by the ethane. "
            "The unit of the answer is $\\mathrm{atm}$ .",
        )
        self.assertEqual(float(instance.gold_answer or ""), 50.7)
        self.assertIsNone(instance.metadata["ten_exponent"])
        self.assertEqual(instance.metadata["source"], "atkins")
        self.assertFalse(instance.metadata["has_solution"])

    def test_factor_with_a_trailing_unit_is_stripped_and_multiplied_out(self) -> None:
        instance = self.task.process_doc(_doc(answer_number="2", unit=" $10^6$ m"))
        assert instance is not None
        # The reference regex consumes the factor and its trailing "$" but not the
        # space that follows, so the remaining unit keeps a leading space.
        self.assertIn("The unit of the answer is  m.", instance.question)
        self.assertNotIn("10^6", instance.question)
        self.assertEqual(float(instance.gold_answer or ""), 2e6)
        self.assertEqual(instance.metadata["ten_exponent"], 6.0)

    def test_negative_exponent(self) -> None:
        instance = self.task.process_doc(
            _doc(answer_number="1.91", unit=" $10^{-47} \\mathrm{~kg}")
        )
        assert instance is not None
        self.assertAlmostEqual(float(instance.gold_answer or ""), 1.91e-47)

    def test_unit_that_is_only_a_factor_stays_in_the_prompt(self) -> None:
        # The reference keeps the exponent visible and compares on the mantissa.
        instance = self.task.process_doc(_doc(answer_number="4.16", unit="$10^{42}$"))
        assert instance is not None
        self.assertIn("The unit of the answer is $10^{42}$.", instance.question)
        self.assertEqual(float(instance.gold_answer or ""), 4.16)
        self.assertIsNone(instance.metadata["ten_exponent"])

    def test_trailing_space_after_a_factor_counts_as_a_unit(self) -> None:
        # "$10^6$ " has a space after the factor, so the reference treats it as a real
        # suffix and multiplies. Stripping the unit first would lose the 10^6.
        instance = self.task.process_doc(_doc(answer_number="1.27", unit="$10^6$ "))
        assert instance is not None
        self.assertEqual(float(instance.gold_answer or ""), 1.27e6)
        self.assertEqual(instance.metadata["ten_exponent"], 6.0)

    def test_unicode_minus_is_normalized(self) -> None:
        instance = self.task.process_doc(_doc(answer_number="−2"))
        assert instance is not None
        self.assertEqual(float(instance.gold_answer or ""), -2.0)

    def test_thousands_separator_is_normalized(self) -> None:
        instance = self.task.process_doc(_doc(answer_number="89,034.79"))
        assert instance is not None
        self.assertAlmostEqual(float(instance.gold_answer or ""), 89034.79)

    def test_blank_unit_is_allowed(self) -> None:
        instance = self.task.process_doc(_doc(unit=""))
        assert instance is not None
        self.assertTrue(instance.question.endswith("The unit of the answer is ."))

    def test_solution_rows_are_flagged(self) -> None:
        instance = self.task.process_doc(_doc(solution="First, apply the ideal gas law."))
        assert instance is not None
        self.assertTrue(instance.metadata["has_solution"])

    def test_unusable_rows_are_dropped(self) -> None:
        self.assertIsNone(self.task.process_doc(_doc(answer_number="")))
        self.assertIsNone(self.task.process_doc(_doc(answer_number="see figure")))
        self.assertIsNone(self.task.process_doc(_doc(problem_text="  ")))


class TestSciBenchAnswerParsing(unittest.TestCase):
    def test_takes_the_last_boxed_value(self) -> None:
        text = "First \\boxed{1.0} then \\boxed{2.0}"
        self.assertEqual(parse_answer(text), "2.0")

    def test_box_without_the_ed_is_not_matched(self) -> None:
        # The reference searches for the substring "oxed", so "\\box{}" finds nothing.
        self.assertEqual(parse_answer("so \\box{3.5}"), "")

    def test_keeps_the_right_hand_side_of_an_equation(self) -> None:
        self.assertEqual(parse_answer("\\boxed{p = 50.7}"), "50.7")

    def test_missing_box_yields_empty(self) -> None:
        self.assertEqual(parse_answer("the answer is 50.7"), "")
        self.assertEqual(parse_answer(""), "")

    def test_unbalanced_braces_yield_empty(self) -> None:
        self.assertEqual(parse_answer("\\boxed{50.7"), "")

    def test_scientific_notation_is_collapsed(self) -> None:
        self.assertEqual(float(collapse_scientific_notation("2.5 \\times 10^6")), 2.5e6)
        self.assertEqual(float(collapse_scientific_notation("2.5 * 10^{3}")), 2500.0)

    def test_plain_decimals_pass_through_collapse_unchanged(self) -> None:
        self.assertEqual(collapse_scientific_notation("2500000"), "2500000")
        self.assertEqual(collapse_scientific_notation("not a number"), "not a number")


class TestSciBenchScorer(unittest.TestCase):
    def _instance(self, gold: float, exponent: float | None = None) -> Instance:
        return Instance(
            question="q",
            gold_answer=repr(gold),
            metadata={"id": "t", "source": "atkins", "ten_exponent": exponent},
        )

    def test_exact_value_scores_one(self) -> None:
        self.assertEqual(_SCORER.score(self._instance(50.7), _boxed("50.7")), 1.0)

    def test_inside_five_percent_scores_one(self) -> None:
        # 1.52 against 1.5 is 1.3 percent out. The Minerva scorer rejects this pair,
        # which is why SciBench needs a tolerance scorer of its own.
        self.assertEqual(_SCORER.score(self._instance(1.5), _boxed("1.52")), 1.0)

    def test_outside_five_percent_scores_zero(self) -> None:
        self.assertEqual(_SCORER.score(self._instance(1.5), _boxed("1.6")), 0.0)

    def test_tolerance_boundary(self) -> None:
        # math.isclose scales the tolerance by the larger magnitude, so the window is
        # wider above the gold than below it: the cutoff is 100/0.95 = 105.26.
        self.assertEqual(_SCORER.score(self._instance(100.0), _boxed("105.2")), 1.0)
        self.assertEqual(_SCORER.score(self._instance(100.0), _boxed("105.3")), 0.0)
        self.assertEqual(_SCORER.score(self._instance(100.0), _boxed("95.3")), 1.0)
        self.assertEqual(_SCORER.score(self._instance(100.0), _boxed("94.9")), 0.0)

    def test_gold_of_zero_demands_an_exact_match(self) -> None:
        # abs_tol keeps its default of zero, so nothing near zero is close to zero.
        self.assertEqual(_SCORER.score(self._instance(0.0), _boxed("0")), 1.0)
        self.assertEqual(_SCORER.score(self._instance(0.0), _boxed("0.001")), 0.0)

    def test_commas_in_the_model_answer_are_stripped(self) -> None:
        self.assertEqual(_SCORER.score(self._instance(89034.79), _boxed("89,034.79")), 1.0)

    def test_falls_back_to_the_first_token(self) -> None:
        self.assertEqual(_SCORER.score(self._instance(50.7), _boxed("50.7 atm")), 1.0)

    def test_negative_values(self) -> None:
        self.assertEqual(_SCORER.score(self._instance(-2.0), _boxed("-2.0")), 1.0)
        self.assertEqual(_SCORER.score(self._instance(-2.0), _boxed("2.0")), 0.0)

    def test_scientific_notation_only_collapses_for_factor_rows(self) -> None:
        with_factor = self._instance(2e6, exponent=6.0)
        self.assertEqual(_SCORER.score(with_factor, _boxed("2 \\times 10^6")), 1.0)
        self.assertEqual(_SCORER.score(with_factor, _boxed("2000000")), 1.0)
        # Without a factor the reference leaves the answer alone, so the mantissa and
        # exponent are compared as written and do not parse as a number.
        plain = self._instance(4.16)
        self.assertEqual(_SCORER.score(plain, _boxed("4.16 \\times 10^{42}")), 1.0)
        self.assertEqual(_SCORER.score(plain, _boxed("9.9 \\times 10^{42}")), 0.0)

    def test_missing_or_unparsable_answer_scores_zero(self) -> None:
        self.assertEqual(_SCORER.score(self._instance(50.7), _boxed("about fifty")), 0.0)
        self.assertEqual(
            _SCORER.score(self._instance(50.7), LMOutput(text="no box here", metadata={})),
            0.0,
        )

    def test_missing_gold_scores_zero(self) -> None:
        blank = Instance(question="q", gold_answer=None, metadata={"ten_exponent": None})
        self.assertEqual(_SCORER.score(blank, _boxed("1.0")), 0.0)

    def test_extracted_value_is_recorded_for_inspection(self) -> None:
        output = _boxed("50.7")
        _SCORER.score(self._instance(50.7), output)
        self.assertEqual(output.metadata["extracted_value"], "50.7")


class TestSciBenchExtractAnswer(unittest.TestCase):
    def test_extract_answer_returns_the_boxed_value(self) -> None:
        task = get_task("scibench")
        self.assertEqual(task.extract_answer(_boxed("50.7")), "50.7")


if __name__ == "__main__":
    unittest.main()
