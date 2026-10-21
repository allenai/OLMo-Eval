"""Tests for the vendored image-QA scoring/prompt utilities.

Pure unit tests — no I/O, no GPU, no API key.  Expected values mirror the
mm_olmo reference implementation (``olmo/eval/vqa.py``,
``molmo_prediction_evaluators.PointCountEval``, ``mmmu_eval_utils.py``,
``math_vista_utils.py``, ``data_formatter.py``).
"""

from __future__ import annotations

import pytest

from olmo_eval.evals.vision.scoring import (
    LONG_CAPTION_TEMPLATES,
    POINT_COUNT_TEMPLATES,
    anls_metric,
    clean_prediction,
    dense_caption_question,
    extract_image_points,
    format_mc_question,
    levenshtein,
    math_vista_score_offline,
    mmmu_score,
    parse_count,
    parse_multi_choice_response,
    parse_open_response,
    pixmo_count_question,
    preprocess_answer,
    real_world_qa_score,
    relaxed_correctness,
    scifi_relaxed_correctness,
    select_mc_option,
    vqa_score,
)

# ---------------------------------------------------------------------------
# levenshtein
# ---------------------------------------------------------------------------


class TestLevenshtein:
    @pytest.mark.parametrize(
        ("a", "b", "expected"),
        [
            ("", "", 0),
            ("abc", "abc", 0),
            ("abc", "", 3),
            ("", "abc", 3),
            ("kitten", "sitting", 3),
            ("flaw", "lawn", 2),
            ("gumbo", "gambol", 2),
            ("a", "b", 1),
        ],
    )
    def test_reference_distances(self, a: str, b: str, expected: int) -> None:
        assert levenshtein(a, b) == expected
        assert levenshtein(b, a) == expected


# ---------------------------------------------------------------------------
# VQA v2 normalization + vqa_score
# ---------------------------------------------------------------------------


class TestVqaScore:
    def test_three_matches_is_full_credit(self) -> None:
        answers = ["red", "red", "red", "blue"] + ["green"] * 6
        assert vqa_score(answers, "red") == 1.0

    def test_one_match_is_third_credit(self) -> None:
        answers = ["red"] + ["blue"] * 9
        assert vqa_score(answers, "red") == pytest.approx(1 / 3)

    def test_number_word_normalization(self) -> None:
        # "two" and "2" normalize to the same answer
        assert vqa_score(["2"] * 10, "two") == 1.0

    def test_article_removal(self) -> None:
        assert vqa_score(["dog"] * 10, "a dog") == 1.0

    def test_contraction_normalization(self) -> None:
        assert preprocess_answer("dont") == "don't"

    def test_punctuation_stripped(self) -> None:
        assert vqa_score(["yes"] * 10, "yes.") == 1.0

    def test_no_match(self) -> None:
        assert vqa_score(["red"] * 10, "blue") == 0.0


# ---------------------------------------------------------------------------
# clean_prediction (VqaEval cleanup)
# ---------------------------------------------------------------------------


class TestCleanPrediction:
    def test_answer_prefix_split(self) -> None:
        assert clean_prediction("Reasoning blah. Answer: 42") == "42"

    def test_multiline_majority_vote(self) -> None:
        assert clean_prediction("cat\ndog\ncat") == "cat"

    def test_multiline_tie_takes_first(self) -> None:
        assert clean_prediction("dog\ncat") == "dog"

    def test_whitespace_collapse(self) -> None:
        assert clean_prediction("  a   b  ") == "a b"


# ---------------------------------------------------------------------------
# ANLS
# ---------------------------------------------------------------------------


class TestAnls:
    def test_exact(self) -> None:
        assert anls_metric("hello", "hello") == 1.0

    def test_case_insensitive(self) -> None:
        assert anls_metric("Hello", "hello") == 1.0

    def test_below_threshold_scores_zero(self) -> None:
        # distance 3 over max-len 5 = 0.6 >= 0.5 -> 0
        assert anls_metric("abcde", "abxyz") == 0

    def test_above_threshold_partial(self) -> None:
        # distance 1 over max-len 5 = 0.2 -> 0.8
        assert anls_metric("abcde", "abcdx") == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# ChartQA relaxed correctness
# ---------------------------------------------------------------------------


class TestRelaxedCorrectness:
    def test_numeric_within_5pct(self) -> None:
        assert relaxed_correctness("100", "104")
        assert not relaxed_correctness("100", "106")

    def test_percent_parsing(self) -> None:
        assert relaxed_correctness("0.07", "7%")

    def test_non_numeric_exact(self) -> None:
        assert relaxed_correctness("Yes", "yes")
        assert not relaxed_correctness("Yes", "yes!")

    def test_zero_target_falls_back_to_exact(self) -> None:
        # target "0" is falsy as float -> exact-match branch
        assert relaxed_correctness("0", "0")
        assert not relaxed_correctness("0", "0.001")


class TestScifiRelaxedCorrectness:
    def test_answer_prefix(self) -> None:
        assert scifi_relaxed_correctness("42", "the answer: 42")

    def test_word_to_number(self) -> None:
        assert scifi_relaxed_correctness("3", "three")

    def test_comma_removal(self) -> None:
        assert scifi_relaxed_correctness("1000", "1,000")

    def test_div_100_normalization(self) -> None:
        assert scifi_relaxed_correctness("0.5", "50")

    def test_list_target(self) -> None:
        assert scifi_relaxed_correctness("[2007, 2008]", "between 2007 and 2008")
        assert not scifi_relaxed_correctness("[2007, 2008]", "between 2007 and 2009")

    def test_string_containment(self) -> None:
        assert scifi_relaxed_correctness("cat", "it is a cat indeed")

    def test_empty_prediction(self) -> None:
        assert not scifi_relaxed_correctness("1", "")


# ---------------------------------------------------------------------------
# select_mc_option
# ---------------------------------------------------------------------------


class TestSelectMcOption:
    OPTIONS = ["A", "B", "C", "D"]

    def test_exact(self) -> None:
        assert select_mc_option("b", self.OPTIONS) == 1

    def test_target_starts_with_option(self) -> None:
        assert select_mc_option("C. some text", self.OPTIONS) == 2

    def test_option_starts_with_target(self) -> None:
        options = ["apple pie", "banana split", "cherry cake"]
        assert select_mc_option("banana", options) == 1

    def test_containment(self) -> None:
        options = ["the red car", "the blue boat", "the green tree"]
        assert select_mc_option("blue", options) == 1

    def test_edit_distance_fallback(self) -> None:
        options = ["alpha", "beta", "gamma"]
        assert select_mc_option("btea", options) == 1

    def test_full_option_text(self) -> None:
        options = ["moon", "none of the above", "earth", "sun"]
        assert select_mc_option("B. none of the above", ["A", "B", "C", "D"]) == 1
        assert select_mc_option("none of the above", options) == 1


# ---------------------------------------------------------------------------
# RealWorldQA
# ---------------------------------------------------------------------------


class TestRealWorldQa:
    def test_mc_letter(self) -> None:
        assert real_world_qa_score("B", "B", "multiple_choice") == 1.0
        assert real_world_qa_score("B", "C", "multiple_choice") == 0.0

    def test_short_answer_normalized(self) -> None:
        assert real_world_qa_score("two", "2", "short_answer") == 1.0


# ---------------------------------------------------------------------------
# MMMU parsing + scoring
# ---------------------------------------------------------------------------


class TestMmmuParsing:
    CHOICES = ["A", "B", "C", "D"]
    INDEX2ANS = {"A": "moon", "B": "sun", "C": "earth", "D": "mars"}

    def test_paren_format(self) -> None:
        assert parse_multi_choice_response("The answer is (B)", self.CHOICES, self.INDEX2ANS) == "B"

    def test_dot_format(self) -> None:
        assert parse_multi_choice_response("B. sun", self.CHOICES, self.INDEX2ANS) == "B"

    def test_bare_letter(self) -> None:
        assert parse_multi_choice_response("B", self.CHOICES, self.INDEX2ANS) == "B"

    def test_content_match_long_response(self) -> None:
        resp = "I believe from the diagram that it must be the sun shining"
        assert parse_multi_choice_response(resp, self.CHOICES, self.INDEX2ANS) == "B"

    def test_multiple_candidates_takes_last(self) -> None:
        resp = "(A) is wrong, the answer is (C)"
        assert parse_multi_choice_response(resp, self.CHOICES, self.INDEX2ANS) == "C"

    def test_unparseable_is_deterministic(self) -> None:
        first = parse_multi_choice_response("?!", self.CHOICES, self.INDEX2ANS, stable_id="x1")
        for _ in range(3):
            assert (
                parse_multi_choice_response("?!", self.CHOICES, self.INDEX2ANS, stable_id="x1")
                == first
            )

    def test_open_number_extraction(self) -> None:
        preds = parse_open_response("So the result is 14.")
        assert 14.0 in preds

    def test_open_comma_number(self) -> None:
        preds = parse_open_response("The total is 1,234")
        assert 1234.0 in preds

    def test_mmmu_score_mc(self) -> None:
        score = mmmu_score(
            ["B"],
            "The answer is (B)",
            question_type="multiple-choice",
            options=["moon", "sun", "earth", "mars"],
        )
        assert score == 1.0

    def test_mmmu_score_open(self) -> None:
        assert mmmu_score(["14"], "The answer is 14", question_type="open", options=[]) == 1.0
        assert mmmu_score(["14"], "The answer is 15", question_type="open", options=[]) == 0.0


# ---------------------------------------------------------------------------
# MathVista offline scoring
# ---------------------------------------------------------------------------


class TestMathVistaOffline:
    def test_mc(self) -> None:
        assert math_vista_score_offline(
            "B",
            question_type="multi_choice",
            answer_type="text",
            choices=["3/11", "8/11", "6/11", "3/5"],
            precision=None,
            target="8/11",
        )

    def test_mc_full_text(self) -> None:
        assert math_vista_score_offline(
            "8/11",
            question_type="multi_choice",
            answer_type="text",
            choices=["3/11", "8/11", "6/11", "3/5"],
            precision=None,
            target="8/11",
        )

    def test_integer(self) -> None:
        assert math_vista_score_offline(
            "14",
            question_type="free_form",
            answer_type="integer",
            choices=[],
            precision=None,
            target="14",
        )

    def test_float_precision(self) -> None:
        assert math_vista_score_offline(
            "0.59999",
            question_type="free_form",
            answer_type="float",
            choices=[],
            precision=1,
            target="0.6",
        )

    def test_wrong_integer(self) -> None:
        assert not math_vista_score_offline(
            "13",
            question_type="free_form",
            answer_type="integer",
            choices=[],
            precision=None,
            target="14",
        )


# ---------------------------------------------------------------------------
# Count parsing (PointCountEval ladder)
# ---------------------------------------------------------------------------


class TestParseCount:
    def test_last_token_int(self) -> None:
        assert parse_count("There are 7") == 7

    def test_trailing_period(self) -> None:
        assert parse_count("Counting shows 12.") == 12

    def test_number_word(self) -> None:
        assert parse_count("there are three") == 3

    def test_a_total_of(self) -> None:
        pred = (
            'Counting the <points coords="1 1 215 453, 1 2 305 410" alt="people">people'
            "</points> shows a total of 8."
        )
        assert parse_count(pred) == 8

    def test_none_means_zero(self) -> None:
        assert parse_count("There are none.") == 0

    def test_points_fallback(self) -> None:
        pred = '<points coords="1 1 215 453, 1 2 305 410, 1 3 100 200" alt="cats">cats</points>'
        assert parse_count(pred) == 3

    def test_no_points_no_number(self) -> None:
        assert parse_count("I cannot tell") == 0

    def test_extract_image_points_unified(self) -> None:
        text = '<points coords="1 1 215 453, 1 2 305 410" alt="x">x</points>'
        assert len(extract_image_points(text, 100, 100)) == 2

    def test_extract_image_points_out_of_bounds_filtered(self) -> None:
        # 4-digit coords > 1000 scale past the 100x100 bounds and are dropped
        text = '<points coords="1 1 2150 4530" alt="x">x</points>'
        assert len(extract_image_points(text, 100, 100)) == 0


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------


class TestPromptTemplates:
    def test_template_count(self) -> None:
        assert len(POINT_COUNT_TEMPLATES) == 60

    # Pinned (arrow_idx, label) -> question triples captured from the released
    # Molmo2-4B predictions-ck2000-pixmo_count_counting-validation dump; the
    # full 540-prompt parity is asserted in the dump-parity test suite.
    @pytest.mark.parametrize(
        ("arrow_idx", "label", "expected"),
        [
            (
                0,
                "cows",
                "How many cows are there in the image? Point to them and output the total count.",
            ),
            (1, "people", "How many people do you see?"),
            (17, "people", "how many people."),
            (
                539,
                "people",
                "Can you see any people in the image? Point to them and output the total count.",
            ),
        ],
    )
    def test_pixmo_count_question_pinned(self, arrow_idx: int, label: str, expected: str) -> None:
        assert pixmo_count_question(label, arrow_idx) == expected

    def test_pixmo_count_label_lowercased(self) -> None:
        assert pixmo_count_question("People", 1) == pixmo_count_question("people", 1)

    def test_long_caption_template_count(self) -> None:
        assert len(LONG_CAPTION_TEMPLATES) == 23

    # Pinned (line_idx -> dense-caption prompt) pairs for mm_olmo's dense-caption
    # generation pipeline (`launch_scripts/eval.py`, loader seed 6198 — verified against a
    # real DenseCaptionEval-test dump's `_getter_seed`), under the released Molmo2-4B config
    # (prompt_templates="uber_model_v2", system_prompt="demo_or_style_v2"): a seeded
    # per-example pick from the long_caption templates, with no style prefix.
    @pytest.mark.parametrize(
        ("idx", "expected"),
        [
            (0, "What can be seen in this image?"),
            (1, "Construct a long caption for this image"),
            (2, "caption the picture"),
            (3, "Generate a long caption about this image."),
            (6, "What do you see in the image?"),
        ],
    )
    def test_dense_caption_question_pinned(self, idx: int, expected: str) -> None:
        assert dense_caption_question(idx) == expected

    def test_format_mc_question_labelled(self) -> None:
        text, option_names = format_mc_question("What is X?", ["moon", "sun"])
        assert text == "What is X?\nOnly return the correct answer option.\nA. moon\nB. sun"
        assert option_names == "AB"

    def test_format_mc_question_unlabelled(self) -> None:
        text, option_names = format_mc_question("What is X?", ["P", "Q"], labelled=False)
        assert text == "What is X?\nOnly return the correct answer option.\nP\nQ"
        assert option_names == ["P", "Q"]


# ---------------------------------------------------------------------------
# Metric aggregation (synthetic responses)
# ---------------------------------------------------------------------------

from olmo_eval.common.types import Instance, LMOutput, Response  # noqa: E402
from olmo_eval.evals.vision.scoring.counting import PointCountScorer  # noqa: E402
from olmo_eval.evals.vision.scoring.image_qa import (  # noqa: E402
    Ai2dScorer,
    RelaxedCorrectnessScorer,
)
from olmo_eval.evals.vision.tasks.image_qa import (  # noqa: E402
    Ai2dMetric,
    ChartQaSubsetMetric,
    MeanScorerMetric,
    PointCountCategoryAverageMetric,
    PointCountMetric,
    PointCountPerCountMetric,
)


def _response(metadata: dict, score_name: str, score: float, output_metadata: dict | None = None):
    instance = Instance(question="q", metadata=metadata)
    output = LMOutput(text="", metadata=output_metadata or {})
    response = Response(instance=instance, request=None, outputs=[output])
    response.scores[score_name] = score
    return response


class TestMeanScorerMetric:
    def test_mean(self) -> None:
        scorer = RelaxedCorrectnessScorer()
        metric = MeanScorerMetric(name="relaxed_correctness", scorer=scorer)
        responses = [
            _response({}, scorer.name, 1.0),
            _response({}, scorer.name, 0.0),
        ]
        assert metric.compute(responses) == pytest.approx(0.5)

    def test_empty(self) -> None:
        metric = MeanScorerMetric(name="x", scorer=RelaxedCorrectnessScorer())
        assert metric.compute([]) == 0.0


class TestChartQaSubsetMetric:
    def test_subset_split(self) -> None:
        scorer = RelaxedCorrectnessScorer()
        responses = [
            _response({"is_human": True}, scorer.name, 1.0),
            _response({"is_human": True}, scorer.name, 0.0),
            _response({"is_human": False}, scorer.name, 1.0),
            _response({"is_human": False}, scorer.name, 1.0),
        ]
        m_all = ChartQaSubsetMetric(name="relaxed_correctness", scorer=scorer, subset="all")
        m_human = ChartQaSubsetMetric(
            name="relaxed_correctness_human", scorer=scorer, subset="human"
        )
        m_aug = ChartQaSubsetMetric(name="relaxed_correctness_aug", scorer=scorer, subset="aug")
        assert m_all.compute(responses) == pytest.approx(0.75)
        assert m_human.compute(responses) == pytest.approx(0.5)
        assert m_aug.compute(responses) == pytest.approx(1.0)


class TestPointCountMetrics:
    def _responses(self):
        scorer = PointCountScorer()
        rows = [
            # (gt count, correct, close)
            (2, 1.0, 1.0),
            (2, 0.0, 1.0),
            (3, 1.0, 1.0),
            (5, 0.0, 0.0),
        ]
        responses = []
        for count, correct, close in rows:
            responses.append(
                _response(
                    {"count": count},
                    scorer.name,
                    correct,
                    output_metadata={
                        "point_count_result": {
                            "correct": correct,
                            "close": close,
                            "valid": 1.0,
                            "pred_count": 0,
                        }
                    },
                )
            )
        return scorer, responses

    def test_correct_close_valid(self) -> None:
        scorer, responses = self._responses()
        assert PointCountMetric(name="correct", scorer=scorer, kind="correct").compute(
            responses
        ) == pytest.approx(0.5)
        assert PointCountMetric(name="close", scorer=scorer, kind="close").compute(
            responses
        ) == pytest.approx(0.75)
        assert PointCountMetric(name="valid", scorer=scorer, kind="valid").compute(
            responses
        ) == pytest.approx(1.0)

    def test_per_count(self) -> None:
        scorer, responses = self._responses()
        assert PointCountPerCountMetric(name="correct_2", scorer=scorer, k=2).compute(
            responses
        ) == pytest.approx(0.5)
        assert PointCountPerCountMetric(name="correct_3", scorer=scorer, k=3).compute(
            responses
        ) == pytest.approx(1.0)
        # absent count -> 0.0
        assert (
            PointCountPerCountMetric(name="correct_9", scorer=scorer, k=9).compute(responses) == 0.0
        )

    def test_per_category_average(self) -> None:
        scorer, responses = self._responses()
        # means: k=2 -> 0.5, k=3 -> 1.0, k=5 -> 0.0; macro avg = 0.5
        assert PointCountCategoryAverageMetric(name="per_category_average", scorer=scorer).compute(
            responses
        ) == pytest.approx(0.5)


class TestAi2dMetric:
    def _response(self, is_correct: float, abc_label: bool, transparent_box: bool):
        return _response(
            {},
            "mc_ai2d",
            is_correct,
            output_metadata={
                "ai2d_result": {
                    "is_correct": is_correct,
                    "abc_label": abc_label,
                    "has_transparent_box": transparent_box,
                }
            },
        )

    def test_routing(self) -> None:
        scorer = Ai2dScorer()
        responses = [
            self._response(1.0, abc_label=False, transparent_box=False),  # both
            self._response(0.0, abc_label=True, transparent_box=False),  # opaque only
            self._response(1.0, abc_label=True, transparent_box=True),  # transparent only
        ]
        opaque = Ai2dMetric(name="mc_ai2d_opaque", scorer=scorer, transparent=False)
        transparent = Ai2dMetric(name="mc_ai2d_transparent", scorer=scorer, transparent=True)
        assert opaque.compute(responses) == pytest.approx(0.5)  # (1.0 + 0.0) / 2
        assert transparent.compute(responses) == pytest.approx(1.0)  # (1.0 + 1.0) / 2
