"""Tests for the Humanity's Last Exam tasks."""

import json
import logging
import math

import pytest

from olmo_eval.common.scorers import ScoringIncompleteError
from olmo_eval.common.types import LMOutput, RequestType, Response
from olmo_eval.evals.tasks import hle
from olmo_eval.evals.tasks.common import OutputScoreAggregation, get_task, task_exists
from olmo_eval.evals.tasks.hle import (
    HLE_DEFAULT_JUDGE_MODEL,
    HLE_SYSTEM_PROMPT,
    calibration_error,
    parse_judge_reply,
    visible_response,
)
from olmo_eval.runners.asynq.preparation import compute_task_metrics
from olmo_eval.runners.asynq.results import _format_scoring_error, _record_scoring_failure
from olmo_eval.runners.io.builders import build_predictions

HLE_TASKS = ("hle:text", "hle:text:verified")
IMAGE_URI = "data:image/png;base64,iVBORw0KGgo="


def _doc(image: str = "", category: str = "Physics", answer_type: str = "exactMatch") -> dict:
    return {
        "id": "abc123",
        "question": "What is the answer?",
        "image": image,
        "answer": "42",
        "answer_type": answer_type,
        "raw_subject": "Physics",
        "category": category,
    }


def _verified_doc(verified_class: str = "Gold subset", image: str = "") -> dict:
    record = {**_doc(image=image), "Verified_Classes": verified_class}
    return {
        "id": record["id"],
        "Verified_Classes": verified_class,
        "question": record["question"],
        "answer": record["answer"],
        "json": json.dumps(record),
    }


def _response(
    task, *texts: str, category: str = "Physics", answer_type: str = "exactMatch"
) -> Response:
    instance = task.process_doc(_doc(category=category, answer_type=answer_type))
    assert instance is not None
    outputs = [LMOutput(text=text) for text in texts or ("Explanation: x\nAnswer: 42",)]
    return Response(instance=instance, request=task.format_request(instance), outputs=outputs)


def _scripted_judge(*replies: str):
    remaining = iter(replies)
    calls: list[str] = []

    async def judge(prompt: str, **_kwargs) -> str:
        calls.append(prompt)
        return next(remaining)

    return judge, calls


def _reply(correct: str = "yes", confidence: str = "80") -> str:
    return (
        "extracted_final_answer: 42\n\n"
        "reasoning: The answers match.\n\n"
        f"correct: {correct}\n\n"
        f"confidence: {confidence}"
    )


@pytest.fixture(autouse=True)
def _no_judge_env(monkeypatch):
    monkeypatch.delenv("OLMO_EVAL_JUDGE", raising=False)


@pytest.fixture(autouse=True)
def _no_backoff_sleep(monkeypatch):
    async def sleep(_delay):
        return None

    monkeypatch.setattr(hle.asyncio, "sleep", sleep)


@pytest.fixture
def task():
    return get_task("hle:text")


def _use_judge(monkeypatch, judge) -> None:
    monkeypatch.setattr(hle, "build_openai_judge_fn", lambda **_: judge)


def _calibration_metric(task):
    return next(m for m in task.config.metrics if m.name == "calibration_error_all_bins")


class TestRegistration:
    def test_text_variants_are_registered_against_pinned_sources(self):
        for name in HLE_TASKS:
            assert task_exists(name)
        # The multimodal variants need provider image support and are not registered yet.
        assert not task_exists("hle")
        assert not task_exists("hle:verified")

        source = get_task("hle:text").config.data_source
        assert source.path == "cais/hle"
        assert source.revision == hle.HLE_REVISION
        verified = get_task("hle:text:verified").config.data_source
        assert verified.path == "skylenage-ai/HLE-Verified"
        assert verified.revision == hle.HLE_VERIFIED_REVISION
        assert verified.data_files == "data/Gold_subset.part*.parquet"

    def test_judge_defaults_and_secrets(self):
        for name in HLE_TASKS:
            config = get_task(name).config
            assert config.judge_model == HLE_DEFAULT_JUDGE_MODEL
            assert config.judge_max_tokens == hle.HLE_JUDGE_MAX_TOKENS
            assert config.required_secrets == ("OPENAI_API_KEY",)
            assert config.output_score_aggregation == OutputScoreAggregation.MEAN

    def test_metrics_cover_categories_answer_types_and_calibration(self, task):
        names = {metric.name for metric in task.config.metrics}
        assert task.config.get_primary_metric().name == "accuracy"
        assert "calibration_error_all_bins" in names
        assert "truncation_rate" in names
        assert {f"accuracy_{suffix}" for suffix in hle.HLE_CATEGORIES.values()} <= names
        assert {"accuracy_exact_match", "accuracy_multiple_choice"} <= names

    def test_default_generation_samples_rather_than_decoding_greedily(self, task):
        params = task.config.sampling_params
        assert params.temperature == 0.6
        assert params.top_p == 0.95
        assert params.max_tokens == 32768

    def test_calibration_error_is_lower_is_better(self, task):
        assert _calibration_metric(task).higher_is_better() is False
        assert task.config.get_primary_metric().higher_is_better() is True


class TestJudgeConfiguration:
    def test_task_override_is_passed_to_the_judge_builder(self, monkeypatch):
        captured = {}

        def fake_builder(**kwargs):
            captured.update(kwargs)
            return object()

        monkeypatch.setattr(hle, "build_openai_judge_fn", fake_builder)
        task = get_task(
            "hle:text",
            config_overrides={"judge_model": "judge-a", "judge_reasoning_effort": "high"},
        )
        task._get_judge_fn()

        assert captured["model"] == "judge-a"
        assert captured["reasoning_effort"] == "high"
        assert captured["max_tokens"] == hle.HLE_JUDGE_MAX_TOKENS

    def test_environment_override_applies_to_the_default_judge(self, monkeypatch):
        monkeypatch.setenv("OLMO_EVAL_JUDGE", "gpt-5:high")

        config = get_task("hle:text").config

        assert config.judge_model == "gpt-5"
        assert config.judge_reasoning_effort == "high"

    def test_environment_model_keeps_an_explicit_reasoning_effort(self, monkeypatch):
        monkeypatch.setenv("OLMO_EVAL_JUDGE", "gpt-5:high")

        config = get_task("hle:text", config_overrides={"judge_reasoning_effort": "low"}).config

        assert config.judge_model == "gpt-5"
        assert config.judge_reasoning_effort == "low"

    def test_task_override_wins_over_the_environment(self, monkeypatch):
        monkeypatch.setenv("OLMO_EVAL_JUDGE", "gpt-5:high")

        config = get_task("hle:text", config_overrides={"judge_model": "judge-a"}).config

        assert config.judge_model == "judge-a"
        assert config.judge_reasoning_effort is None


class TestProcessDoc:
    def test_maps_the_schema(self, task):
        instance = task.process_doc(_doc(), index=3)

        assert instance is not None
        assert instance.question == "What is the answer?"
        assert instance.gold_answer == "42"
        assert instance.metadata["id"] == "abc123"
        assert instance.metadata["category"] == "Physics"
        assert instance.metadata["answer_type"] == "exactMatch"

    def test_drops_image_questions(self, task):
        assert task.process_doc(_doc(image=IMAGE_URI)) is None

    def test_verified_variant_reads_the_full_record_and_keeps_only_text_gold(self):
        verified = get_task("hle:text:verified")

        gold = verified.process_doc(_verified_doc())
        assert gold is not None
        assert gold.metadata["answer_type"] == "exactMatch"
        assert verified.process_doc(_verified_doc(verified_class="Revision subset")) is None
        assert verified.process_doc(_verified_doc(image=IMAGE_URI)) is None


def test_request_uses_the_official_system_prompt(task):
    instance = task.process_doc(_doc())
    assert instance is not None

    request = task.format_request(instance)

    assert request.request_type == RequestType.CHAT
    assert request.messages == (
        {"role": "system", "content": HLE_SYSTEM_PROMPT},
        {"role": "user", "content": "What is the answer?"},
    )
    assert request.images is None


class TestVisibleResponse:
    def test_drops_a_closed_reasoning_block(self):
        assert visible_response("<think>work</think>\nAnswer: 42") == "Answer: 42"

    def test_unclosed_reasoning_block_has_no_final_response(self):
        assert visible_response("<think>so the answer is probably 41") == ""

    def test_untagged_answers_are_kept(self):
        assert visible_response("  Answer: 42\n") == "Answer: 42"


class TestJudgeParsing:
    def test_parses_the_official_fields(self):
        verdict = parse_judge_reply(_reply("yes", "85"))

        assert verdict is not None
        assert verdict.correct is True
        assert verdict.confidence == pytest.approx(0.85)
        assert verdict.extracted_final_answer == "42"

    def test_tolerates_markdown_emphasis_and_percent_signs(self):
        verdict = parse_judge_reply("**correct:** no\n**confidence:** 30%")

        assert verdict is not None
        assert verdict.correct is False
        assert verdict.confidence == pytest.approx(0.3)

    def test_missing_confidence_counts_as_full_confidence(self):
        verdict = parse_judge_reply("correct: yes")

        assert verdict is not None
        assert verdict.confidence == 1.0

    def test_no_verdict_is_unparseable(self):
        assert parse_judge_reply("The response looks right to me.") is None

    def test_does_not_read_the_word_correct_inside_the_reasoning(self):
        verdict = parse_judge_reply("reasoning: the answer is correct: no doubt\ncorrect: no")

        assert verdict is not None
        assert verdict.correct is False


class TestCalibrationError:
    def test_perfect_calibration_is_zero(self):
        assert calibration_error([1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0], bin_size=2) == 0.0

    def test_fewer_items_than_a_bin_form_one_bin(self):
        assert calibration_error([0.9, 0.9], [1.0, 0.0], bin_size=100) == pytest.approx(0.4)

    def test_bins_are_rms_weighted_and_the_last_absorbs_the_remainder(self):
        confidences = [0.1, 0.1, 0.9, 0.9, 0.9]
        correct = [0.0, 0.0, 1.0, 0.0, 0.0]
        # Bins of two: the first holds two items at confidence 0.1 and accuracy 0; the
        # last takes the remaining three, at confidence 0.9 and accuracy 1/3.
        expected = math.sqrt(2 / 5 * 0.1**2 + 3 / 5 * (0.9 - 1 / 3) ** 2)

        assert calibration_error(confidences, correct, bin_size=2) == pytest.approx(expected)

    def test_empty_input_is_zero(self):
        assert calibration_error([], []) == 0.0


class TestScoring:
    @pytest.mark.anyio
    async def test_judges_the_visible_response_and_keeps_the_raw_reply(self, task, monkeypatch):
        response = _response(task, "<think>hidden</think>\nExplanation: x\nAnswer: 42")
        judge, calls = _scripted_judge(_reply("yes", "80"))
        _use_judge(monkeypatch, judge)

        await task.score_responses([response])

        assert len(calls) == 1
        assert "[response]: Explanation: x\nAnswer: 42" in calls[0]
        assert "hidden" not in calls[0]
        assert "[correct_answer]: 42" in calls[0]
        assert response.scores["accuracy"] == 1.0
        saved = build_predictions([response])[0]["model_output"][0]["judge_result"]
        assert saved == {
            "scorer": "hle_judge",
            "truncated": False,
            "raw_judge_response": _reply("yes", "80"),
            "correct": True,
            "confidence": pytest.approx(0.8),
            "extracted_final_answer": "42",
            "parse_error": False,
        }

    @pytest.mark.anyio
    async def test_unfinished_reasoning_is_scored_wrong_without_the_judge(self, task, monkeypatch):
        response = _response(task, "<think>so the answer is 42")
        judge, calls = _scripted_judge()
        _use_judge(monkeypatch, judge)

        await task.score_responses([response])

        assert calls == []
        assert response.scores["accuracy"] == 0.0
        judge_result = response.outputs[0].metadata["judge_result"]
        assert judge_result["no_final_response"] is True
        assert judge_result["correct"] is False

    @pytest.mark.anyio
    async def test_unparseable_reply_is_a_scoring_error_not_a_wrong_answer(
        self, task, monkeypatch, caplog
    ):
        response = _response(task, "Answer: 42", "Answer: 42")
        judge, calls = _scripted_judge("garbage", "more garbage", "nope", _reply("yes", "90"))
        _use_judge(monkeypatch, judge)

        with caplog.at_level(logging.WARNING, logger=hle.__name__):
            await task.score_responses([response])

        assert len(calls) == hle.HLE_JUDGE_ATTEMPTS + 1
        failed = [o for o in response.outputs if "scoring_errors" in o.metadata]
        judged = [o for o in response.outputs if "scoring_errors" not in o.metadata]
        assert len(failed) == len(judged) == 1
        error = failed[0].metadata["scoring_errors"]["hle_judge"]
        assert error["type"] == "HLEJudgeParseError"
        assert failed[0].metadata["judge_result"]["parse_error"] is True
        assert failed[0].metadata["judge_result"]["raw_judge_response"] in {"nope", "garbage"}
        assert "score:confidence" not in failed[0].metadata
        assert judged[0].metadata["judge_result"]["correct"] is True
        # The failed sample leaves the instance average rather than counting as wrong.
        assert response.scores["accuracy"] == 1.0
        assert "failed on 1/2 output(s)" in caplog.text

    @pytest.mark.anyio
    async def test_a_batch_of_malformed_replies_fails_instead_of_scoring(self, task, monkeypatch):
        responses = [_response(task), _response(task)]
        judge, _ = _scripted_judge(*["garbage"] * (2 * hle.HLE_JUDGE_ATTEMPTS))
        _use_judge(monkeypatch, judge)

        with pytest.raises(ScoringIncompleteError, match="failed for every sample"):
            await task.score_responses(responses)

        for response in responses:
            output = response.outputs[0]
            assert output.metadata["scoring_errors"]["hle_judge"]["type"] == "HLEJudgeParseError"
            assert output.metadata["judge_result"]["raw_judge_response"] == "garbage"
            assert "accuracy" not in response.scores

    @pytest.mark.anyio
    async def test_metrics_slice_by_category_and_answer_type(self, task, monkeypatch):
        responses = [
            _response(task, category="Physics"),
            _response(task, category="Physics", answer_type="multipleChoice"),
            _response(task, category="Math"),
        ]
        judge, _ = _scripted_judge(_reply("yes", "90"), _reply("no", "90"), _reply("yes", "90"))
        _use_judge(monkeypatch, judge)

        await task.score_responses(responses)

        metrics = {metric.name: metric.compute(responses) for metric in task.config.metrics}
        assert metrics["accuracy"] == pytest.approx(2 / 3)
        assert metrics["accuracy_physics"] == 0.5
        assert metrics["accuracy_math"] == 1.0
        assert metrics["accuracy_chemistry"] == 0.0
        assert metrics["accuracy_multiple_choice"] == 0.0
        assert metrics["accuracy_exact_match"] == 1.0
        assert metrics["calibration_error_all_bins"] == pytest.approx(abs(0.9 - 2 / 3))

    @pytest.mark.anyio
    async def test_calibration_pairs_each_sample_with_its_own_confidence(self, task, monkeypatch):
        # Every question has a right answer at 0% confidence and a wrong one at 100%:
        # maximally miscalibrated, although per-question averages look calibrated. Two
        # hundred outputs fill two official-size bins, one at each confidence.
        responses = [_response(task, "Answer: a", "Answer: b") for _ in range(100)]
        judge, calls = _scripted_judge(*[_reply("yes", "0"), _reply("no", "100")] * 100)
        _use_judge(monkeypatch, judge)

        await task.score_responses(responses)

        assert len(calls) == 200
        assert _calibration_metric(task).compute(responses) == pytest.approx(1.0)
        assert all(response.scores["accuracy"] == 0.5 for response in responses)

    @pytest.mark.anyio
    async def test_truncated_answer_scores_wrong_without_the_judge(self, task, monkeypatch, caplog):
        # A visible answer that ran into the token limit still counts as wrong: the
        # model has to finish within its budget.
        truncated = _response(task, "Explanation: partial\nAnswer: 42")
        truncated.outputs[0].metadata["finish_reason"] = "length"
        finished = _response(task, "Answer: 42")
        judge, calls = _scripted_judge(_reply("yes", "90"))
        _use_judge(monkeypatch, judge)

        with caplog.at_level(logging.WARNING, logger=hle.__name__):
            await task.score_responses([truncated, finished])

        assert len(calls) == 1
        assert truncated.scores["accuracy"] == 0.0
        judge_result = truncated.outputs[0].metadata["judge_result"]
        assert judge_result["truncated"] is True
        assert judge_result["correct"] is False
        assert finished.outputs[0].metadata["judge_result"]["truncated"] is False
        assert "hit the token limit on 1 output(s)" in caplog.text
        metrics = {m.name: m.compute([truncated, finished]) for m in task.config.metrics}
        assert metrics["truncation_rate"] == 0.5
        assert metrics["accuracy"] == 0.5

    @pytest.mark.anyio
    async def test_question_without_output_scores_wrong_and_is_logged(
        self, task, monkeypatch, caplog
    ):
        response = _response(task)
        response.outputs = []
        judge, calls = _scripted_judge()
        _use_judge(monkeypatch, judge)

        with caplog.at_level(logging.WARNING, logger=hle.__name__):
            await task.score_responses([response])

        assert calls == []
        assert response.scores == {"accuracy": 0.0, "truncation_rate": 1.0}
        assert "no model output for 1 question(s), e.g. abc123" in caplog.text


async def _score_like_the_runner(task, response: Response) -> Response:
    """Score one instance the way the async runner does, including its failure path."""
    try:
        return (await task.score_responses([response]))[0]
    except Exception as exc:
        return _record_scoring_failure(
            response,
            scorer_names=list(task._get_scorers()),
            error=_format_scoring_error(exc, phase="response"),
        )


class TestRunnerFinalization:
    @pytest.mark.anyio
    async def test_judge_failures_mark_the_result_incomplete(self, task, monkeypatch):
        responses = [_response(task), _response(task)]
        judge, _ = _scripted_judge(*["garbage"] * (2 * hle.HLE_JUDGE_ATTEMPTS))
        _use_judge(monkeypatch, judge)

        scored = [await _score_like_the_runner(task, response) for response in responses]
        result = compute_task_metrics("hle:text", task, scored, {}, len(scored), 0.0)

        assert result.error is not None
        assert result.error.startswith("Incomplete: 2 output score(s)")
        assert result.metrics["calibration_error_all_bins"] == {"hle_judge": 0.0}

    @pytest.mark.anyio
    async def test_unjudged_instance_is_left_out_of_accuracy(self, task, monkeypatch):
        # One question judged correct, one whose judge never gave a verdict: the
        # unjudged one must not count as a wrong answer, in aggregate or per instance.
        judged, failed = _response(task, category="Physics"), _response(task, category="Physics")
        judge, _ = _scripted_judge(_reply("yes", "90"), *["garbage"] * hle.HLE_JUDGE_ATTEMPTS)
        _use_judge(monkeypatch, judge)

        scored = [await _score_like_the_runner(task, judged)]
        scored.append(await _score_like_the_runner(task, failed))
        result = compute_task_metrics("hle:text", task, scored, {}, len(scored), 0.0)

        assert result.error is not None and result.error.startswith("Incomplete:")
        assert result.metrics["accuracy"] == {"hle_judge": 1.0}
        assert result.metrics["accuracy_physics"] == {"hle_judge": 1.0}
        judged_prediction, failed_prediction = result.predictions
        assert judged_prediction["instance_metrics"]["accuracy"]["hle_judge"] == 1.0
        assert "accuracy" not in failed_prediction["instance_metrics"]
        assert "accuracy_physics" not in failed_prediction["instance_metrics"]

    @pytest.mark.anyio
    async def test_an_api_error_after_retries_is_incomplete_not_wrong(self, task, monkeypatch):
        async def judge(_prompt: str, **_kwargs) -> str:
            raise ValueError("judge unavailable")

        _use_judge(monkeypatch, judge)
        monkeypatch.setattr(hle, "retry_with_backoff", lambda call, **_: call())

        scored = [await _score_like_the_runner(task, _response(task))]
        result = compute_task_metrics("hle:text", task, scored, {}, 1, 0.0)

        assert result.error is not None
        assert result.error.startswith("Incomplete: 1 output score(s)")
        error = scored[0].outputs[0].metadata["scoring_errors"]["hle_judge"]
        assert "ValueError: judge unavailable" in error["message"]

    @pytest.mark.anyio
    async def test_a_clean_run_has_no_error(self, task, monkeypatch):
        judge, _ = _scripted_judge(_reply("yes", "90"))
        _use_judge(monkeypatch, judge)

        scored = [await _score_like_the_runner(task, _response(task))]
        result = compute_task_metrics("hle:text", task, scored, {}, 1, 0.0)

        assert result.error is None
        assert result.metrics["accuracy"] == {"hle_judge": 1.0}


@pytest.mark.anyio
async def test_slice_metrics_are_stored_per_instance(task, monkeypatch):
    responses = [
        _response(task, category="Math"),
        _response(task, category="Physics", answer_type="multipleChoice"),
    ]
    judge, _ = _scripted_judge(_reply("yes", "90"), _reply("no", "90"))
    _use_judge(monkeypatch, judge)

    await task.score_responses(responses)
    math_prediction, physics_prediction = build_predictions(responses, metrics=task.config.metrics)

    math_metrics = math_prediction["instance_metrics"]
    assert math_metrics["accuracy"]["hle_judge"] == 1.0
    assert math_metrics["accuracy_math"]["hle_judge"] == 1.0
    assert math_metrics["accuracy_exact_match"]["hle_judge"] == 1.0
    assert "accuracy_physics" not in math_metrics
    assert "accuracy_multiple_choice" not in math_metrics

    physics_metrics = physics_prediction["instance_metrics"]
    assert physics_metrics["accuracy_physics"]["hle_judge"] == 0.0
    assert physics_metrics["accuracy_multiple_choice"]["hle_judge"] == 0.0
    assert "accuracy_math" not in physics_metrics
