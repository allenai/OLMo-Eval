"""Tests for DeepSearchQA grading of full responses with the notebook prompt."""

import json
from unittest.mock import AsyncMock

import pytest

from olmo_eval.common.types import Instance, LMOutput, LMRequest, RequestType, Response
from olmo_eval.evals.tasks import deepsearchqa, deepsearchqa_official_judge
from olmo_eval.evals.tasks.common import get_task, task_exists
from olmo_eval.evals.tasks.deepsearchqa_official_judge import (
    compute_deepsearchqa_official_scores,
    parse_deepsearchqa_official_judge_response,
)


@pytest.fixture
def task():
    return get_task("deepsearchqa_official_judge")


def _doc(
    problem: str = "Which countries border Chad?",
    problem_category: str = "Geography",
    answer: str = "Libya, Sudan, Niger",
    answer_type: str = "Set Answer",
):
    return {
        "problem": problem,
        "problem_category": problem_category,
        "answer": answer,
        "answer_type": answer_type,
    }


def _response(instance: Instance, text: str = "Libya, Sudan, and Niger border Chad.") -> Response:
    return Response(
        instance=instance,
        request=LMRequest(request_type=RequestType.CHAT, messages=()),
        outputs=[LMOutput(text=text)],
        scores={},
    )


class TestRegistration:
    def test_task_and_variant_are_registered(self):
        assert task_exists("deepsearchqa_official_judge")
        assert task_exists("deepsearchqa_official_judge:mini")

        full = get_task("deepsearchqa_official_judge")
        mini = get_task("deepsearchqa_official_judge:mini")
        assert full.config.data_source.path == "google/deepsearchqa"
        assert full.config.data_source.subset == "deepsearchqa"
        assert full.config.data_source.split == "eval"
        assert mini.config.limit == 50

    def test_primary_and_secondary_metrics(self, task):
        assert task.config.get_primary_metric().name == "deepsearchqa_official_f1"
        assert {metric.name for metric in task.config.metrics} == {
            "deepsearchqa_official_f1",
            "deepsearchqa_official_precision",
            "deepsearchqa_official_recall",
            "deepsearchqa_official_exact_match",
        }


class TestProcessDoc:
    """Data loading is inherited from DeepSearchQABase; spot-check it here too."""

    def test_maps_real_schema(self, task):
        instance = task.process_doc(_doc(), index=3)
        assert instance is not None
        assert instance.gold_answer == "Libya, Sudan, Niger"
        assert "gold_items" not in instance.metadata

    def test_restores_no_answer_reference_after_csv_loading(self, task):
        instance = task.process_doc(_doc(answer="", answer_type="Set Answer"))
        assert instance is not None
        assert instance.gold_answer == "None"


class TestFormatRequest:
    def test_prompt_has_no_final_answer_requirement(self, task):
        instance = task.process_doc(_doc())
        request = task.format_request(instance)
        prompt = request.messages[0]["content"]
        assert "FINAL ANSWER" not in prompt
        assert "Which countries border Chad?" in prompt


class TestExtractAnswer:
    def test_returns_raw_stripped_text(self, task):
        output = LMOutput(text="  Libya, Sudan, and Niger.  ")
        assert task.extract_answer(output) == "Libya, Sudan, and Niger."


class TestParseJudgeResponse:
    def test_counts_true_correctness_details(self):
        raw = (
            '{"Answer Correctness": {"Explanation": "...", '
            '"Correctness Details": {"Libya": true, "Sudan": true, "Niger": false}, '
            '"Excessive Answers": []}}'
        )
        assert parse_deepsearchqa_official_judge_response(raw) == (3, 2, 0)

    def test_counts_excessive_answers(self):
        raw = (
            '{"Answer Correctness": {"Explanation": "...", '
            '"Correctness Details": {"Libya": true}, '
            '"Excessive Answers": ["Egypt", "Nigeria"]}}'
        )
        assert parse_deepsearchqa_official_judge_response(raw) == (1, 1, 2)

    def test_uses_judges_expected_answer_count(self):
        raw = (
            '{"Answer Correctness": {"Explanation": "...", "Correctness Details": '
            '{"Libya": true, "Sudan": true, "Extra": true}, "Excessive Answers": []}}'
        )
        assert parse_deepsearchqa_official_judge_response(raw) == (3, 3, 0)

    def test_tolerates_surrounding_prose(self):
        raw = (
            "Here is my rating:\n"
            '{"Answer Correctness": {"Explanation": "...", "Correctness Details": {"Libya": true}, '
            '"Excessive Answers": []}}\nDone.'
        )
        assert parse_deepsearchqa_official_judge_response(raw) == (1, 1, 0)

    def test_returns_none_for_malformed_json(self):
        assert parse_deepsearchqa_official_judge_response("not json") is None

    def test_returns_none_when_answer_correctness_missing(self):
        raw = '{"Correctness Details": {"Libya": true}, "Excessive Answers": []}'
        assert parse_deepsearchqa_official_judge_response(raw) is None

    def test_returns_none_when_excessive_answers_is_not_a_list(self):
        raw = (
            '{"Answer Correctness": {"Explanation": "...", "Correctness Details": {"Libya": true}, '
            '"Excessive Answers": "none"}}'
        )
        assert parse_deepsearchqa_official_judge_response(raw) is None

    @pytest.mark.parametrize("value", [1, "true", None])
    def test_rejects_non_boolean_correctness_values(self, value):
        raw = json.dumps(
            {"Answer Correctness": {"Explanation": "...", "Correctness Details": {"Libya": value}}}
        )
        assert parse_deepsearchqa_official_judge_response(raw) is None

    def test_rejects_non_string_excessive_answers(self):
        raw = json.dumps(
            {
                "Answer Correctness": {
                    "Explanation": "...",
                    "Correctness Details": {},
                    "Excessive Answers": [None],
                }
            }
        )
        assert parse_deepsearchqa_official_judge_response(raw) is None

    def test_missing_excessive_answers_defaults_to_empty_list(self):
        raw = (
            '{"Answer Correctness": {"Explanation": "...", "Correctness Details": {"Libya": true}}}'
        )
        assert parse_deepsearchqa_official_judge_response(raw) == (1, 1, 0)


class TestComputeScores:
    def test_perfect_match(self):
        scores = compute_deepsearchqa_official_scores(num_gold=3, num_matched=3, num_excessive=0)
        assert scores == {
            "deepsearchqa_official_precision": 1.0,
            "deepsearchqa_official_recall": 1.0,
            "deepsearchqa_official_f1": 1.0,
            "deepsearchqa_official_exact_match": 1.0,
        }

    def test_excessive_answer_hurts_precision_not_recall(self):
        scores = compute_deepsearchqa_official_scores(num_gold=2, num_matched=2, num_excessive=1)
        assert scores["deepsearchqa_official_recall"] == 1.0
        assert scores["deepsearchqa_official_precision"] == pytest.approx(2 / 3)
        assert scores["deepsearchqa_official_exact_match"] == 0.0

    def test_missing_gold_item_hurts_recall_not_precision(self):
        scores = compute_deepsearchqa_official_scores(num_gold=3, num_matched=2, num_excessive=0)
        assert scores["deepsearchqa_official_precision"] == 1.0
        assert scores["deepsearchqa_official_recall"] == pytest.approx(2 / 3)
        assert scores["deepsearchqa_official_exact_match"] == 0.0

    def test_empty_gold_set_and_no_excessive_answers_is_perfect(self):
        scores = compute_deepsearchqa_official_scores(num_gold=0, num_matched=0, num_excessive=0)
        assert scores == {
            "deepsearchqa_official_precision": 1.0,
            "deepsearchqa_official_recall": 1.0,
            "deepsearchqa_official_f1": 1.0,
            "deepsearchqa_official_exact_match": 1.0,
        }

    def test_empty_gold_set_with_excessive_answers_scores_zero(self):
        scores = compute_deepsearchqa_official_scores(num_gold=0, num_matched=0, num_excessive=2)
        assert scores == {
            "deepsearchqa_official_precision": 0.0,
            "deepsearchqa_official_recall": 1.0,
            "deepsearchqa_official_f1": 0.0,
            "deepsearchqa_official_exact_match": 0.0,
        }


class TestScoreResponses:
    @pytest.mark.anyio
    async def test_scores_a_fully_correct_response(self, task, monkeypatch):
        instance = task.process_doc(_doc())
        response = _response(instance, text="Libya, Sudan, and Niger all border Chad.")
        calls = []

        async def judge(prompt):
            calls.append(prompt)
            return (
                '{"Answer Correctness": {"Explanation": "...", "Correctness Details": '
                '{"Libya": true, "Sudan": true, "Niger": true}, "Excessive Answers": []}}'
            )

        monkeypatch.setattr(
            deepsearchqa_official_judge,
            "build_deepsearchqa_official_judge_fn",
            lambda _config: judge,
        )
        await task.score_responses([response])

        assert len(calls) == 1
        assert "FINAL ANSWER" not in calls[0]
        assert response.scores["deepsearchqa_official_f1"] == 1.0
        assert response.outputs[0].metadata["deepsearchqa_official_num_matched"] == 3
        assert response.outputs[0].metadata["score:deepsearchqa_official_f1"] == 1.0

    @pytest.mark.anyio
    async def test_empty_response_skips_the_judge_call(self, task, monkeypatch):
        instance = task.process_doc(_doc())
        response = _response(instance, text="")
        calls = []

        async def judge(prompt):
            calls.append(prompt)
            return (
                '{"Answer Correctness": {"Explanation": "...", '
                '"Correctness Details": {}, "Excessive Answers": []}}'
            )

        monkeypatch.setattr(
            deepsearchqa_official_judge,
            "build_deepsearchqa_official_judge_fn",
            lambda _config: judge,
        )
        await task.score_responses([response])

        assert calls == []
        assert response.scores["deepsearchqa_official_f1"] == 0.0

    @pytest.mark.anyio
    async def test_empty_gold_set_with_explicit_no_answer_is_judged(self, task, monkeypatch):
        instance = task.process_doc(_doc(answer="", answer_type="Set Answer"))
        response = _response(instance, text="I could not find any such country.")
        calls = []

        async def judge(prompt):
            calls.append(prompt)
            return (
                '{"Answer Correctness": {"Explanation": "...", '
                '"Correctness Details": {}, "Excessive Answers": []}}'
            )

        monkeypatch.setattr(
            deepsearchqa_official_judge,
            "build_deepsearchqa_official_judge_fn",
            lambda _config: judge,
        )
        await task.score_responses([response])

        assert len(calls) == 1
        assert "<answer>\nNone\n</answer>" in calls[0]
        assert response.scores["deepsearchqa_official_f1"] == 1.0

    @pytest.mark.anyio
    async def test_retries_after_a_malformed_judge_reply(self, task, monkeypatch):
        monkeypatch.setattr(deepsearchqa.asyncio, "sleep", lambda _seconds: _no_sleep())

        instance = task.process_doc(_doc(answer="New Zealand", answer_type="Single Answer"))
        response = _response(instance, text="It's New Zealand.")
        valid = (
            '{"Answer Correctness": {"Explanation": "...", '
            '"Correctness Details": {"New Zealand": true}, '
            '"Excessive Answers": []}}'
        )
        outputs = iter(["not json", valid])
        calls = []

        async def judge(prompt):
            calls.append(prompt)
            return next(outputs)

        monkeypatch.setattr(
            deepsearchqa_official_judge,
            "build_deepsearchqa_official_judge_fn",
            lambda _config: judge,
        )
        await task.score_responses([response])

        assert len(calls) == 2
        assert response.scores["deepsearchqa_official_f1"] == 1.0

    @pytest.mark.anyio
    async def test_exhausted_retries_score_zero_without_raising(self, task, monkeypatch):
        monkeypatch.setattr(deepsearchqa.asyncio, "sleep", lambda _seconds: _no_sleep())

        instance = task.process_doc(_doc(answer="New Zealand", answer_type="Single Answer"))
        response = _response(instance, text="It's New Zealand.")

        async def judge(_prompt):
            return "not json"

        monkeypatch.setattr(
            deepsearchqa_official_judge,
            "build_deepsearchqa_official_judge_fn",
            lambda _config: judge,
        )
        await task.score_responses([response])

        assert response.scores["deepsearchqa_official_f1"] == 0.0
        assert response.scores["deepsearchqa_official_recall"] == 0.0

    @pytest.mark.anyio
    @pytest.mark.parametrize("answer", ["Hasbro, Inc.", "209,550,294"])
    async def test_single_answer_with_commas_receives_full_credit(self, task, monkeypatch, answer):
        response = _response(
            task.process_doc(_doc(answer=answer, answer_type="Single Answer")), text=answer
        )
        judge = AsyncMock(
            return_value=json.dumps(
                {
                    "Answer Correctness": {
                        "Explanation": "...",
                        "Correctness Details": {answer: True},
                    }
                }
            )
        )
        monkeypatch.setattr(
            deepsearchqa_official_judge,
            "build_deepsearchqa_official_judge_fn",
            lambda _config: judge,
        )

        await task.score_responses([response])

        judge.assert_awaited_once()
        assert f"<answer>\n{answer}\n</answer>" in judge.call_args.args[0]
        assert set(response.scores.values()) == {1.0}

    @pytest.mark.anyio
    async def test_set_answer_uses_all_correctness_details_for_recall(self, task, monkeypatch):
        gold = ["Alice Smith (Boston, Massachusetts)", "Bob Jones (Austin, Texas)"]
        answer = "\n".join(gold)
        response = _response(task.process_doc(_doc(answer=answer)), text=gold[0])
        judge = AsyncMock(
            return_value=json.dumps(
                {
                    "Answer Correctness": {
                        "Explanation": "...",
                        "Correctness Details": {gold[0]: True, gold[1]: False},
                    }
                }
            )
        )
        monkeypatch.setattr(
            deepsearchqa_official_judge,
            "build_deepsearchqa_official_judge_fn",
            lambda _config: judge,
        )

        await task.score_responses([response])

        assert f"<answer>\n{answer}\n</answer>" in judge.call_args.args[0]
        assert response.scores["deepsearchqa_official_precision"] == 1.0
        assert response.scores["deepsearchqa_official_recall"] == 0.5
        assert response.scores["deepsearchqa_official_f1"] == pytest.approx(2 / 3)


async def _no_sleep() -> None:
    return None
