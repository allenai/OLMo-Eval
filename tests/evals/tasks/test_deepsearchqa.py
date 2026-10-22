"""Tests for the DeepSearchQA multi-step search question answering task."""

import pytest

from olmo_eval.common.types import Instance, LMOutput, LMRequest, RequestType, Response
from olmo_eval.evals.tasks import deepsearchqa
from olmo_eval.evals.tasks.common import get_task, task_exists
from olmo_eval.evals.tasks.deepsearchqa import (
    compute_deepsearchqa_scores,
    extract_final_answer,
    parse_deepsearchqa_judge_response,
    split_answer_set,
)


@pytest.fixture
def task():
    return get_task("deepsearchqa")


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


def _response(instance: Instance, text: str = "FINAL ANSWER: Libya, Sudan, Niger") -> Response:
    return Response(
        instance=instance,
        request=LMRequest(request_type=RequestType.CHAT, messages=()),
        outputs=[LMOutput(text=text)],
        scores={},
    )


class TestRegistration:
    def test_task_and_variant_are_registered(self):
        assert task_exists("deepsearchqa")
        assert task_exists("deepsearchqa:mini")

        full = get_task("deepsearchqa")
        mini = get_task("deepsearchqa:mini")
        assert full.config.data_source.path == "google/deepsearchqa"
        assert full.config.data_source.subset == "deepsearchqa"
        assert full.config.data_source.split == "eval"
        assert mini.config.limit == 50

    def test_primary_and_secondary_metrics(self, task):
        assert task.config.get_primary_metric().name == "deepsearchqa_f1"
        assert {metric.name for metric in task.config.metrics} == {
            "deepsearchqa_f1",
            "deepsearchqa_precision",
            "deepsearchqa_recall",
            "deepsearchqa_exact_match",
        }


class TestProcessDoc:
    def test_maps_real_schema_and_splits_answer_set(self, task):
        instance = task.process_doc(_doc(), index=3)

        assert instance is not None
        assert instance.question == "Which countries border Chad?"
        assert instance.gold_answer == "Libya, Sudan, Niger"
        assert instance.metadata["id"] == "deepsearchqa_3"
        assert instance.metadata["index"] == 3
        assert instance.metadata["problem_category"] == "Geography"
        assert instance.metadata["answer_type"] == "Set Answer"
        assert instance.metadata["gold_items"] == ["Libya", "Sudan", "Niger"]

    def test_single_answer_becomes_one_item_set(self, task):
        instance = task.process_doc(_doc(answer="New Zealand", answer_type="Single Answer"))
        assert instance is not None
        assert instance.metadata["gold_items"] == ["New Zealand"]

    def test_skips_missing_problem(self, task):
        assert task.process_doc(_doc(problem="")) is None

    def test_skips_answer_that_is_only_junk_separators(self, task):
        assert task.process_doc(_doc(answer=" , ,")) is None

    def test_missing_answer_on_single_answer_is_dropped_as_malformed(self, task):
        assert task.process_doc(_doc(answer="", answer_type="Single Answer")) is None
        assert task.process_doc(_doc(answer=None, answer_type="Single Answer")) is None

    def test_missing_answer_on_set_answer_becomes_empty_gold_set(self, task):
        # The source CSV spells "no items satisfy every constraint" as the literal
        # text "None" on Set Answer rows; the CSV loader turns that into a null.
        for missing in ("", None):
            instance = task.process_doc(_doc(answer=missing, answer_type="Set Answer"))
            assert instance is not None
            assert instance.metadata["gold_items"] == []
            assert instance.gold_answer == ""


class TestSplitAnswerSet:
    def test_splits_and_strips_items(self):
        assert split_answer_set("Libya, Sudan,Niger") == ["Libya", "Sudan", "Niger"]

    def test_drops_empty_items(self):
        assert split_answer_set(" New Zealand ,, ") == ["New Zealand"]

    def test_empty_string_yields_empty_list(self):
        assert split_answer_set("") == []


class TestExtractFinalAnswer:
    def test_extracts_text_after_marker(self):
        text = "I searched several sources.\nFINAL ANSWER: Libya, Sudan, Niger"
        assert extract_final_answer(text) == "Libya, Sudan, Niger"

    def test_tolerates_markdown_emphasis_and_missing_colon(self):
        text = "Reasoning...\n**FINAL ANSWER** Libya, Sudan, Niger"
        assert extract_final_answer(text) == "Libya, Sudan, Niger"

    def test_falls_back_to_full_text_when_marker_absent(self):
        assert extract_final_answer(" Just an answer. ") == "Just an answer."

    def test_uses_last_marker_when_repeated(self):
        text = "FINAL ANSWER: draft\nMore thinking.\nFINAL ANSWER: Libya, Sudan, Niger"
        assert extract_final_answer(text) == "Libya, Sudan, Niger"


class TestParseJudgeResponse:
    def test_parses_valid_json(self):
        raw = '{"matched_gold_indices": [0, 1], "matched_submitted_indices": [0, 2]}'
        parsed = parse_deepsearchqa_judge_response(raw, num_gold=2, num_pred=3)
        assert parsed == ({0, 1}, {0, 2})

    def test_tolerates_surrounding_prose(self):
        raw = (
            "Here is my judgment:\n"
            '{"matched_gold_indices": [0], "matched_submitted_indices": [0]}\n'
            "Done."
        )
        assert parse_deepsearchqa_judge_response(raw, num_gold=1, num_pred=1) == ({0}, {0})

    def test_drops_out_of_range_indices(self):
        raw = '{"matched_gold_indices": [0, 5], "matched_submitted_indices": [-1, 0]}'
        assert parse_deepsearchqa_judge_response(raw, num_gold=2, num_pred=2) == ({0}, {0})

    def test_returns_none_for_malformed_json(self):
        assert parse_deepsearchqa_judge_response("not json", num_gold=1, num_pred=1) is None

    def test_returns_none_when_fields_are_not_lists(self):
        raw = '{"matched_gold_indices": "0", "matched_submitted_indices": [0]}'
        assert parse_deepsearchqa_judge_response(raw, num_gold=1, num_pred=1) is None


class TestComputeScores:
    def test_perfect_match(self):
        scores = compute_deepsearchqa_scores(
            num_gold=2, num_pred=2, num_matched_gold=2, num_matched_pred=2
        )
        assert scores == {
            "deepsearchqa_precision": 1.0,
            "deepsearchqa_recall": 1.0,
            "deepsearchqa_f1": 1.0,
            "deepsearchqa_exact_match": 1.0,
        }

    def test_extra_predicted_item_hurts_precision_not_recall(self):
        scores = compute_deepsearchqa_scores(
            num_gold=2, num_pred=3, num_matched_gold=2, num_matched_pred=2
        )
        assert scores["deepsearchqa_recall"] == 1.0
        assert scores["deepsearchqa_precision"] == pytest.approx(2 / 3)
        assert scores["deepsearchqa_exact_match"] == 0.0

    def test_missing_gold_item_hurts_recall_not_precision(self):
        scores = compute_deepsearchqa_scores(
            num_gold=3, num_pred=2, num_matched_gold=2, num_matched_pred=2
        )
        assert scores["deepsearchqa_precision"] == 1.0
        assert scores["deepsearchqa_recall"] == pytest.approx(2 / 3)
        assert scores["deepsearchqa_exact_match"] == 0.0

    def test_no_predicted_items_scores_zero(self):
        scores = compute_deepsearchqa_scores(
            num_gold=2, num_pred=0, num_matched_gold=0, num_matched_pred=0
        )
        assert scores == {
            "deepsearchqa_precision": 0.0,
            "deepsearchqa_recall": 0.0,
            "deepsearchqa_f1": 0.0,
            "deepsearchqa_exact_match": 0.0,
        }

    def test_empty_gold_set_and_empty_prediction_is_a_perfect_score(self):
        scores = compute_deepsearchqa_scores(
            num_gold=0, num_pred=0, num_matched_gold=0, num_matched_pred=0
        )
        assert scores == {
            "deepsearchqa_precision": 1.0,
            "deepsearchqa_recall": 1.0,
            "deepsearchqa_f1": 1.0,
            "deepsearchqa_exact_match": 1.0,
        }

    def test_empty_gold_set_with_hallucinated_prediction_scores_zero(self):
        scores = compute_deepsearchqa_scores(
            num_gold=0, num_pred=2, num_matched_gold=0, num_matched_pred=0
        )
        assert scores == {
            "deepsearchqa_precision": 0.0,
            "deepsearchqa_recall": 1.0,
            "deepsearchqa_f1": 0.0,
            "deepsearchqa_exact_match": 0.0,
        }


class TestScoreResponses:
    @pytest.mark.anyio
    async def test_scores_a_fully_correct_response(self, task, monkeypatch):
        instance = task.process_doc(_doc())
        response = _response(instance, text="FINAL ANSWER: Libya, Sudan, Niger")
        calls = []

        async def judge(prompt):
            calls.append(prompt)
            return '{"matched_gold_indices": [0, 1, 2], "matched_submitted_indices": [0, 1, 2]}'

        monkeypatch.setattr(deepsearchqa, "build_deepsearchqa_judge_fn", lambda: judge)
        await task.score_responses([response])

        assert len(calls) == 1
        assert response.scores["deepsearchqa_f1"] == 1.0
        assert response.scores["deepsearchqa_exact_match"] == 1.0
        assert response.outputs[0].metadata["deepsearchqa_predicted_items"] == [
            "Libya",
            "Sudan",
            "Niger",
        ]
        assert response.outputs[0].metadata["score:deepsearchqa_f1"] == 1.0

    @pytest.mark.anyio
    async def test_empty_prediction_skips_the_judge_call(self, task, monkeypatch):
        instance = task.process_doc(_doc())
        response = _response(instance, text="")
        calls = []

        async def judge(prompt):
            calls.append(prompt)
            return '{"matched_gold_indices": [], "matched_submitted_indices": []}'

        monkeypatch.setattr(deepsearchqa, "build_deepsearchqa_judge_fn", lambda: judge)
        await task.score_responses([response])

        assert calls == []
        assert response.scores["deepsearchqa_f1"] == 0.0

    @pytest.mark.anyio
    async def test_retries_after_a_malformed_judge_reply(self, task, monkeypatch):
        monkeypatch.setattr(deepsearchqa.asyncio, "sleep", lambda _seconds: _no_sleep())

        instance = task.process_doc(_doc(answer="New Zealand", answer_type="Single Answer"))
        response = _response(instance, text="FINAL ANSWER: New Zealand")
        valid = '{"matched_gold_indices": [0], "matched_submitted_indices": [0]}'
        outputs = iter(["not json", valid])
        calls = []

        async def judge(prompt):
            calls.append(prompt)
            return next(outputs)

        monkeypatch.setattr(deepsearchqa, "build_deepsearchqa_judge_fn", lambda: judge)
        await task.score_responses([response])

        assert len(calls) == 2
        assert response.scores["deepsearchqa_f1"] == 1.0

    @pytest.mark.anyio
    async def test_empty_gold_set_with_correctly_empty_prediction_skips_judge(
        self, task, monkeypatch
    ):
        instance = task.process_doc(_doc(answer="", answer_type="Set Answer"))
        response = _response(instance, text="")
        calls = []

        async def judge(prompt):
            calls.append(prompt)
            return '{"matched_gold_indices": [], "matched_submitted_indices": []}'

        monkeypatch.setattr(deepsearchqa, "build_deepsearchqa_judge_fn", lambda: judge)
        await task.score_responses([response])

        assert calls == []
        assert response.scores["deepsearchqa_f1"] == 1.0
        assert response.scores["deepsearchqa_exact_match"] == 1.0

    @pytest.mark.anyio
    async def test_empty_gold_set_with_hallucinated_prediction_skips_judge(self, task, monkeypatch):
        instance = task.process_doc(_doc(answer="", answer_type="Set Answer"))
        response = _response(instance, text="FINAL ANSWER: Some hallucinated item")
        calls = []

        async def judge(prompt):
            calls.append(prompt)
            return '{"matched_gold_indices": [], "matched_submitted_indices": []}'

        monkeypatch.setattr(deepsearchqa, "build_deepsearchqa_judge_fn", lambda: judge)
        await task.score_responses([response])

        assert calls == []
        assert response.scores["deepsearchqa_f1"] == 0.0
        assert response.scores["deepsearchqa_recall"] == 1.0
        assert response.scores["deepsearchqa_precision"] == 0.0

    @pytest.mark.anyio
    async def test_exhausted_retries_score_zero_without_raising(self, task, monkeypatch):
        monkeypatch.setattr(deepsearchqa.asyncio, "sleep", lambda _seconds: _no_sleep())

        instance = task.process_doc(_doc(answer="New Zealand", answer_type="Single Answer"))
        response = _response(instance, text="FINAL ANSWER: New Zealand")

        async def judge(_prompt):
            return "not json"

        monkeypatch.setattr(deepsearchqa, "build_deepsearchqa_judge_fn", lambda: judge)
        await task.score_responses([response])

        assert response.scores["deepsearchqa_f1"] == 0.0
        assert response.scores["deepsearchqa_recall"] == 0.0


async def _no_sleep() -> None:
    return None
