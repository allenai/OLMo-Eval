"""Tests for the DeepSearchQA multi-step search question answering task."""

import asyncio
import json
import logging
from unittest.mock import AsyncMock, Mock

import pytest

from olmo_eval.common.types import (
    Instance,
    LMOutput,
    LMRequest,
    RequestType,
    Response,
    compute_task_hash,
)
from olmo_eval.evals.tasks import deepsearchqa, deepsearchqa_official_judge
from olmo_eval.evals.tasks.common import get_task, task_exists
from olmo_eval.evals.tasks.deepsearchqa import (
    compute_deepsearchqa_scores,
    extract_final_answer,
    parse_deepsearchqa_judge_response,
)
from olmo_eval.runners.io.builders import build_predictions


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


def _judgment(details, excessive=()) -> str:
    return json.dumps(
        {
            "Answer Correctness": {
                "Explanation": "Compared the expected and submitted answers.",
                "Correctness Details": details,
                "Excessive Answers": list(excessive),
            }
        }
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


@pytest.mark.parametrize("task_name", ["deepsearchqa", "deepsearchqa_official_judge"])
class TestJudgeConfiguration:
    @pytest.fixture(autouse=True)
    def clear_judge_override(self, monkeypatch):
        monkeypatch.delenv("OLMO_EVAL_JUDGE", raising=False)

    def test_defaults_are_serialized_for_full_and_mini_tasks(self, task_name):
        for spec in (task_name, f"{task_name}:mini"):
            config = get_task(spec).config.to_dict()
            assert config["judge_model"] == "gpt-5.5"
            assert config["judge_reasoning_effort"] == "medium"
            assert config["judge_max_tokens"] == (8192 if task_name == "deepsearchqa" else 1024)

    @pytest.mark.parametrize(
        ("env_spec", "model", "effort"),
        [("gpt-4.1-mini", "gpt-4.1-mini", None), ("gpt-5:high", "gpt-5", "high")],
    )
    def test_environment_override_changes_recorded_config_and_hash(
        self, task_name, monkeypatch, env_spec, model, effort
    ):
        original = get_task(task_name).config.to_dict()
        monkeypatch.setenv("OLMO_EVAL_JUDGE", env_spec)

        overridden = get_task(task_name).config.to_dict()

        assert overridden["judge_model"] == model
        assert overridden["judge_reasoning_effort"] == effort
        assert overridden["judge_max_tokens"] == original["judge_max_tokens"]
        assert compute_task_hash(original) != compute_task_hash(overridden)
        monkeypatch.delenv("OLMO_EVAL_JUDGE")
        assert get_task(task_name).config.to_dict() == original

    @pytest.mark.parametrize(
        "override",
        [
            {"judge_model": "another-judge"},
            {"judge_reasoning_effort": "high"},
            {"judge_max_tokens": 4096},
        ],
    )
    def test_each_judge_setting_affects_the_hash(self, task_name, override):
        original = get_task(task_name).config.to_dict()
        changed = get_task(task_name, config_overrides=override).config.to_dict()

        assert compute_task_hash(original) != compute_task_hash(changed)
        for name, value in override.items():
            assert changed[name] == value

    @pytest.mark.anyio
    @pytest.mark.parametrize("env_spec", [None, "environment-judge:high"])
    async def test_grading_uses_recorded_settings_after_environment_changes(
        self, task_name, monkeypatch, env_spec
    ):
        if env_spec is not None:
            monkeypatch.setenv("OLMO_EVAL_JUDGE", env_spec)
        task = get_task(
            task_name,
            config_overrides={
                "judge_model": "configured-judge",
                "judge_reasoning_effort": "low",
                "judge_max_tokens": 2345,
            },
        )
        recorded = task.config.to_dict()
        assert recorded["judge_model"] == (
            "configured-judge" if env_spec is None else "environment-judge"
        )
        assert recorded["judge_reasoning_effort"] == ("low" if env_spec is None else "high")
        monkeypatch.setenv("OLMO_EVAL_JUDGE", "different-judge:medium")
        module = deepsearchqa if task_name == "deepsearchqa" else deepsearchqa_official_judge
        verdict = _judgment({"New Zealand": True})
        judge = AsyncMock(return_value=verdict)
        builder = Mock(return_value=judge)
        monkeypatch.setattr(module, "build_openai_judge_fn", builder)
        response = _response(
            task.process_doc(_doc(answer="New Zealand", answer_type="Single Answer")),
            text="FINAL ANSWER: New Zealand",
        )

        await task.score_responses([response])

        builder.assert_called_once_with(
            model=recorded["judge_model"],
            reasoning_effort=recorded["judge_reasoning_effort"],
            max_tokens=2345,
            temperature=0.0,
            scorer_name="DeepSearchQA"
            if task_name == "deepsearchqa"
            else "DeepSearchQAOfficialJudge",
        )
        judge.assert_awaited_once()
        assert set(response.scores.values()) == {1.0}
        assert task.config.to_dict() == recorded

    def test_rejects_environment_override_without_a_model(self, task_name, monkeypatch):
        monkeypatch.setenv("OLMO_EVAL_JUDGE", ":medium")
        with pytest.raises(ValueError, match="must name a judge model"):
            get_task(task_name)


class TestProcessDoc:
    def test_maps_real_schema_and_preserves_answer_text(self, task):
        instance = task.process_doc(_doc(), index=3)

        assert instance is not None
        assert instance.question == "Which countries border Chad?"
        assert instance.gold_answer == "Libya, Sudan, Niger"
        assert instance.metadata["id"] == "deepsearchqa_3"
        assert instance.metadata["index"] == 3
        assert instance.metadata["problem_category"] == "Geography"
        assert instance.metadata["answer_type"] == "Set Answer"
        assert "gold_items" not in instance.metadata

    @pytest.mark.parametrize("answer", ["New Zealand", "Hasbro, Inc.", "209,550,294"])
    def test_preserves_single_answers_containing_commas(self, task, answer):
        instance = task.process_doc(_doc(answer=answer, answer_type="Single Answer"))
        assert instance is not None
        assert instance.gold_answer == answer

    def test_skips_missing_problem(self, task):
        assert task.process_doc(_doc(problem="")) is None

    def test_skips_answer_that_is_only_junk_separators(self, task):
        assert task.process_doc(_doc(answer=" , ,")) is None

    def test_missing_answer_on_single_answer_is_dropped_as_malformed(self, task):
        assert task.process_doc(_doc(answer="", answer_type="Single Answer")) is None
        assert task.process_doc(_doc(answer=None, answer_type="Single Answer")) is None

    def test_restores_no_answer_reference_after_csv_loading(self, task):
        # The source CSV spells "no items satisfy every constraint" as the literal
        # text "None" on Set Answer rows; the CSV loader turns that into a null.
        for missing in ("", None, "None"):
            instance = task.process_doc(_doc(answer=missing, answer_type="Set Answer"))
            assert instance is not None
            assert instance.gold_answer == "None"


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


@pytest.mark.parametrize(
    "parse",
    [
        parse_deepsearchqa_judge_response,
        deepsearchqa_official_judge.parse_deepsearchqa_official_judge_response,
    ],
    ids=["final_answer", "full_response"],
)
class TestParseJudgeResponse:
    def test_counts_expected_matched_and_excessive_answers(self, parse):
        raw = _judgment({"Paris": True, "London": False, "Tokyo": True}, ["Rome", "Berlin"])
        assert parse(raw) == (3, 2, 2)

    def test_tolerates_surrounding_prose(self, parse):
        raw = "Here is my judgment:\n```json\n" + _judgment({"Libya": True}) + "\n```\nDone."
        assert parse(raw) == (1, 1, 0)

    @pytest.mark.parametrize("value", [1, "true", None])
    def test_rejects_non_boolean_correctness(self, parse, value):
        assert parse(_judgment({"Libya": value})) is None

    @pytest.mark.parametrize("value", [None, 1, {}])
    def test_rejects_invalid_explanation(self, parse, value):
        data = json.loads(_judgment({"Libya": True}))
        data["Answer Correctness"]["Explanation"] = value
        assert parse(json.dumps(data)) is None

    def test_requires_explanation(self, parse):
        data = json.loads(_judgment({"Libya": True}))
        del data["Answer Correctness"]["Explanation"]
        assert parse(json.dumps(data)) is None

    def test_rejects_non_string_excessive_answers(self, parse):
        assert parse(_judgment({"Libya": True}, [None])) is None

    @pytest.mark.parametrize(
        "raw",
        ["not json", "null", '{"matched_gold_indices": [0], "matched_submitted_indices": [0]}'],
    )
    def test_rejects_invalid_or_old_schema(self, parse, raw):
        assert parse(raw) is None

    def test_missing_excessive_answers_defaults_to_empty_list(self, parse):
        data = json.loads(_judgment({"Libya": True}))
        del data["Answer Correctness"]["Excessive Answers"]
        assert parse(json.dumps(data)) == (1, 1, 0)

    def test_accepts_empty_correctness_details(self, parse):
        assert parse(_judgment({})) == (0, 0, 0)


@pytest.mark.parametrize(
    "build_prompt",
    [
        deepsearchqa.build_deepsearchqa_judge_prompt,
        deepsearchqa_official_judge.build_deepsearchqa_official_judge_prompt,
    ],
)
def test_prompt_contains_valid_judge_example_and_preserves_answer_text(build_prompt):
    question = 'Which company uses the symbol "{ABC}"?'
    answer = 'Example, Inc.; "{ABC}"'
    response = "The company is Example, Inc. ({ABC})."
    prompt = build_prompt(question, "Single Answer", answer, response)

    output_format = (
        prompt.split("**Output Format:**", 1)[1].split("User Prompt (Wrapped", 1)[0].strip()
    )
    example_text = output_format[output_format.index("{") :]
    shape, _ = json.JSONDecoder().raw_decode(example_text)
    assert set(shape) == {"Answer Correctness"}
    correctness = shape["Answer Correctness"]
    assert set(correctness) == {"Explanation", "Correctness Details", "Excessive Answers"}
    assert isinstance(correctness["Explanation"], str)
    assert isinstance(correctness["Correctness Details"], dict)
    assert all(isinstance(value, bool) for value in correctness["Correctness Details"].values())
    assert isinstance(correctness["Excessive Answers"], list)
    assert parse_deepsearchqa_judge_response(example_text) == (2, 2, 1)
    assert f"<prompt>\n{question}\n</prompt>" in prompt
    assert f"<answer>\n{answer}\n</answer>" in prompt
    assert f"<response>\n{response}\n</response>" in prompt


class TestComputeScores:
    def test_perfect_match(self):
        scores = compute_deepsearchqa_scores(num_gold=2, num_matched=2, num_excessive=0)
        assert scores == {
            "deepsearchqa_precision": 1.0,
            "deepsearchqa_recall": 1.0,
            "deepsearchqa_f1": 1.0,
            "deepsearchqa_exact_match": 1.0,
        }

    def test_extra_predicted_item_hurts_precision_not_recall(self):
        scores = compute_deepsearchqa_scores(num_gold=2, num_matched=2, num_excessive=1)
        assert scores["deepsearchqa_recall"] == 1.0
        assert scores["deepsearchqa_precision"] == pytest.approx(2 / 3)
        assert scores["deepsearchqa_exact_match"] == 0.0

    def test_missing_gold_item_hurts_recall_not_precision(self):
        scores = compute_deepsearchqa_scores(num_gold=3, num_matched=2, num_excessive=0)
        assert scores["deepsearchqa_precision"] == 1.0
        assert scores["deepsearchqa_recall"] == pytest.approx(2 / 3)
        assert scores["deepsearchqa_exact_match"] == 0.0

    def test_no_predicted_items_scores_zero(self):
        scores = compute_deepsearchqa_scores(num_gold=2, num_matched=0, num_excessive=0)
        assert scores == {
            "deepsearchqa_precision": 0.0,
            "deepsearchqa_recall": 0.0,
            "deepsearchqa_f1": 0.0,
            "deepsearchqa_exact_match": 0.0,
        }

    def test_empty_gold_set_and_empty_prediction_is_a_perfect_score(self):
        scores = compute_deepsearchqa_scores(num_gold=0, num_matched=0, num_excessive=0)
        assert scores == {
            "deepsearchqa_precision": 1.0,
            "deepsearchqa_recall": 1.0,
            "deepsearchqa_f1": 1.0,
            "deepsearchqa_exact_match": 1.0,
        }

    def test_empty_gold_set_with_hallucinated_prediction_scores_zero(self):
        scores = compute_deepsearchqa_scores(num_gold=0, num_matched=0, num_excessive=2)
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
            return _judgment({"Libya": True, "Sudan": True, "Niger": True})

        monkeypatch.setattr(deepsearchqa, "build_deepsearchqa_judge_fn", lambda _config: judge)
        await task.score_responses([response])

        assert len(calls) == 1
        assert response.scores["deepsearchqa_f1"] == 1.0
        assert response.scores["deepsearchqa_exact_match"] == 1.0
        assert (
            response.outputs[0].metadata["deepsearchqa_submitted_answer"] == "Libya, Sudan, Niger"
        )
        assert response.outputs[0].metadata["score:deepsearchqa_f1"] == 1.0

    @pytest.mark.anyio
    async def test_empty_prediction_skips_the_judge_call(self, task, monkeypatch):
        instance = task.process_doc(_doc())
        response = _response(instance, text="")
        calls = []

        async def judge(prompt):
            calls.append(prompt)
            return _judgment({})

        monkeypatch.setattr(deepsearchqa, "build_deepsearchqa_judge_fn", lambda _config: judge)
        await task.score_responses([response])

        assert calls == []
        assert response.scores["deepsearchqa_f1"] == 0.0

    @pytest.mark.anyio
    async def test_retries_after_a_malformed_judge_reply(self, task, monkeypatch):
        monkeypatch.setattr(deepsearchqa.asyncio, "sleep", lambda _seconds: _no_sleep())

        instance = task.process_doc(_doc(answer="New Zealand", answer_type="Single Answer"))
        response = _response(instance, text="FINAL ANSWER: New Zealand")
        valid = _judgment({"New Zealand": True})
        outputs = iter(["not json", valid])
        calls = []

        async def judge(prompt):
            calls.append(prompt)
            return next(outputs)

        monkeypatch.setattr(deepsearchqa, "build_deepsearchqa_judge_fn", lambda _config: judge)
        await task.score_responses([response])

        assert len(calls) == 2
        assert response.scores["deepsearchqa_f1"] == 1.0

    @pytest.mark.anyio
    async def test_no_answer_reference_with_blank_prediction_scores_zero(self, task, monkeypatch):
        instance = task.process_doc(_doc(answer="", answer_type="Set Answer"))
        response = _response(instance, text="")
        calls = []

        async def judge(prompt):
            calls.append(prompt)
            return _judgment({})

        monkeypatch.setattr(deepsearchqa, "build_deepsearchqa_judge_fn", lambda _config: judge)
        await task.score_responses([response])

        assert calls == []
        assert response.scores["deepsearchqa_f1"] == 0.0
        assert response.scores["deepsearchqa_exact_match"] == 0.0

    @pytest.mark.anyio
    async def test_empty_gold_set_with_hallucinated_prediction_is_judged(self, task, monkeypatch):
        instance = task.process_doc(_doc(answer="", answer_type="Set Answer"))
        response = _response(instance, text="FINAL ANSWER: Some hallucinated item")
        calls = []

        async def judge(prompt):
            calls.append(prompt)
            return _judgment({}, ["Some hallucinated item"])

        monkeypatch.setattr(deepsearchqa, "build_deepsearchqa_judge_fn", lambda _config: judge)
        await task.score_responses([response])

        assert len(calls) == 1
        assert "<answer>\nNone\n</answer>" in calls[0]
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

        monkeypatch.setattr(deepsearchqa, "build_deepsearchqa_judge_fn", lambda _config: judge)
        await task.score_responses([response])

        assert response.scores["deepsearchqa_f1"] == 0.0
        assert response.scores["deepsearchqa_recall"] == 0.0

    @pytest.mark.anyio
    @pytest.mark.parametrize("answer", ["Hasbro, Inc.", "209,550,294"])
    async def test_single_answer_with_commas_receives_full_credit(self, task, monkeypatch, answer):
        response = _response(
            task.process_doc(_doc(answer=answer, answer_type="Single Answer")),
            text=f"FINAL ANSWER: {answer}",
        )
        judge = AsyncMock(return_value=_judgment({answer: True}))
        monkeypatch.setattr(deepsearchqa, "build_deepsearchqa_judge_fn", lambda _config: judge)

        await task.score_responses([response])

        judge.assert_awaited_once()
        prompt = judge.call_args.args[0]
        assert f"<answer>\n{answer}\n</answer>" in prompt
        assert f"<response>\n{answer}\n</response>" in prompt
        assert set(response.scores.values()) == {1.0}

    @pytest.mark.anyio
    async def test_set_answer_counts_complete_items_with_internal_commas(self, task, monkeypatch):
        gold = ["Alice Smith (Boston, Massachusetts)", "Bob Jones (Austin, Texas)"]
        answer = "\n".join(gold)
        response = _response(task.process_doc(_doc(answer=answer)), text=f"FINAL ANSWER: {gold[0]}")
        judge = AsyncMock(return_value=_judgment({gold[0]: True, gold[1]: False}))
        monkeypatch.setattr(deepsearchqa, "build_deepsearchqa_judge_fn", lambda _config: judge)

        await task.score_responses([response])

        assert f"<answer>\n{answer}\n</answer>" in judge.call_args.args[0]
        assert response.scores["deepsearchqa_precision"] == 1.0
        assert response.scores["deepsearchqa_recall"] == 0.5
        assert response.scores["deepsearchqa_f1"] == pytest.approx(2 / 3)

    @pytest.mark.anyio
    @pytest.mark.parametrize("text", ["FINAL ANSWER: None", "No items meet the constraints."])
    async def test_explicit_no_answer_is_sent_to_the_judge(self, task, monkeypatch, text):
        response = _response(task.process_doc(_doc(answer=None)), text=text)
        judge = AsyncMock(return_value=_judgment({}))
        monkeypatch.setattr(deepsearchqa, "build_deepsearchqa_judge_fn", lambda _config: judge)

        await task.score_responses([response])

        judge.assert_awaited_once()
        assert response.scores["deepsearchqa_exact_match"] == 1.0


async def _no_sleep() -> None:
    return None


@pytest.fixture(
    params=[
        ("deepsearchqa", deepsearchqa, "build_deepsearchqa_judge_fn"),
        (
            "deepsearchqa_official_judge",
            deepsearchqa_official_judge,
            "build_deepsearchqa_official_judge_fn",
        ),
    ],
    ids=["final_answer", "full_response"],
)
def judge_task(request, monkeypatch):
    task_name, module, factory = request.param
    judge = AsyncMock()
    sleep = AsyncMock()
    monkeypatch.setattr(module, factory, lambda _config: judge)
    monkeypatch.setattr(deepsearchqa.asyncio, "sleep", sleep)
    return get_task(task_name), judge, sleep


@pytest.mark.anyio
@pytest.mark.parametrize("gold_answer", ["New Zealand", "None", None])
@pytest.mark.parametrize("text", ["", " \n\t", None], ids=["blank", "whitespace", "no_output"])
async def test_blank_response_is_incorrect_for_every_reference(judge_task, gold_answer, text):
    task, judge, sleep = judge_task
    response = _response(task.process_doc(_doc(answer=gold_answer)), text=text or "")
    if text is None:
        response.outputs.clear()

    await task.score_responses([response])

    assert set(response.scores.values()) == {0.0}
    judge.assert_not_awaited()
    sleep.assert_not_awaited()


@pytest.mark.anyio
async def test_notebook_verdict_scores_missing_and_excessive_answers(judge_task):
    task, judge, sleep = judge_task
    raw = "Draft: consider Rome.\nFINAL ANSWER: Tokyo, Paris, Rome, Berlin."
    response = _response(task.process_doc(_doc(answer="Paris; London; Tokyo")), text=raw)
    judge.return_value = _judgment(
        {"Paris": True, "London": False, "Tokyo": True}, ["Rome", "Berlin"]
    )

    await task.score_responses([response])

    prefix = "deepsearchqa" if task.config.name == "deepsearchqa" else "deepsearchqa_official"
    assert response.scores[f"{prefix}_precision"] == 0.5
    assert response.scores[f"{prefix}_recall"] == pytest.approx(2 / 3)
    assert response.scores[f"{prefix}_f1"] == pytest.approx(4 / 7)
    assert response.scores[f"{prefix}_exact_match"] == 0.0
    judged_answer = "Tokyo, Paris, Rome, Berlin." if task.config.name == "deepsearchqa" else raw
    assert f"<response>\n{judged_answer}\n</response>" in judge.call_args.args[0]
    judge.assert_awaited_once()
    sleep.assert_not_awaited()


@pytest.mark.anyio
async def test_empty_correctness_details_for_nonempty_reference_are_retried(judge_task):
    task, judge, sleep = judge_task
    response = _response(task.process_doc(_doc(answer="New Zealand")), text="New Zealand")
    judge.side_effect = [_judgment({}), _judgment({"New Zealand": True})]

    await task.score_responses([response])

    assert judge.await_count == 2
    sleep.assert_awaited_once_with(1)
    assert set(response.scores.values()) == {1.0}
    assert "judge_result" not in response.outputs[0].metadata


class TestJudgeFailures:
    @pytest.mark.anyio
    @pytest.mark.parametrize("gold_answer", ["New Zealand", None], ids=["nonempty", "empty"])
    @pytest.mark.parametrize(
        "replies",
        [
            ["not json"] * 3,
            [""] * 3,
            [TimeoutError("request timed out")] * 3,
            ["", "not json", ConnectionError("connection dropped")],
        ],
        ids=["invalid", "empty", "api_error", "mixed"],
    )
    async def test_exhaustion_warns_and_saves_all_attempts(
        self, judge_task, caplog, gold_answer, replies
    ):
        task, judge, sleep = judge_task
        judge.side_effect = replies
        response = _response(task.process_doc(_doc(answer=gold_answer), index=17))

        with caplog.at_level(logging.WARNING):
            await task.score_responses([response])

        assert judge.await_count == 3
        assert [call.args[0] for call in sleep.await_args_list] == [1, 2]
        assert set(response.scores.values()) == {0.0}
        saved = json.loads(json.dumps(build_predictions([response], metrics=task.config.metrics)))
        output = saved[0]["model_output"][0]
        result = output["judge_result"]
        assert result["status"] == "failed"
        assert result["attempts"] == 3
        assert result["instance_id"] == "deepsearchqa_17"
        assert result["task"] == task.config.name
        assert len(result["failures"]) == 3
        for attempt, (reply, failure) in enumerate(
            zip(replies, result["failures"], strict=True), start=1
        ):
            assert failure["attempt"] == attempt
            if isinstance(reply, Exception):
                assert failure["type"] == type(reply).__name__
                assert failure["message"] == str(reply)
            else:
                assert failure["response"] == reply
                assert failure["type"] == (
                    "InvalidJudgeResponse" if reply else "EmptyJudgeResponse"
                )
        assert set(output["scoring_errors"]) == set(response.scores)
        assert all(
            error["type"] == "JudgeAttemptsExhausted" for error in output["scoring_errors"].values()
        )
        assert len(caplog.records) == 1
        assert task.config.name in caplog.text
        assert "deepsearchqa_17" in caplog.text
        assert "3 attempts" in caplog.text

    @pytest.mark.anyio
    async def test_recovery_is_not_marked_as_failure(self, judge_task, caplog):
        task, judge, sleep = judge_task
        valid = _judgment({"New Zealand": True})
        judge.side_effect = [TimeoutError("transient"), "", valid]
        response = _response(task.process_doc(_doc(answer="New Zealand")))

        with caplog.at_level(logging.WARNING):
            await task.score_responses([response])

        assert judge.await_count == 3
        assert sleep.await_count == 2
        assert set(response.scores.values()) == {1.0}
        assert "scoring_errors" not in response.outputs[0].metadata
        assert not caplog.records

    @pytest.mark.anyio
    async def test_cancellation_propagates(self, judge_task, caplog):
        task, judge, sleep = judge_task
        judge.side_effect = asyncio.CancelledError()
        response = _response(task.process_doc(_doc()))

        with pytest.raises(asyncio.CancelledError):
            await task.score_responses([response])

        judge.assert_awaited_once()
        sleep.assert_not_awaited()
        assert response.scores == {}
        assert not caplog.records
