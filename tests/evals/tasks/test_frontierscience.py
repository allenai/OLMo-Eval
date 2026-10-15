"""Tests for the FrontierScience olympiad and research tracks."""

import asyncio
import gc
import logging
import math
import types
import weakref
from collections import Counter

import pytest

from olmo_eval.common.execution import ScoringContext
from olmo_eval.common.types import Instance, LMOutput, LMRequest, RequestType, Response
from olmo_eval.evals.tasks import frontierscience
from olmo_eval.evals.tasks.common import OutputScoreAggregation, get_task, task_exists
from olmo_eval.evals.tasks.frontierscience import (
    FRONTIERSCIENCE_DEFAULT_JUDGE_MODEL,
    FRONTIERSCIENCE_OLYMPIAD_JUDGE_PROMPT,
    FRONTIERSCIENCE_RESEARCH_JUDGE_PROMPT,
    extract_final_answer,
    parse_olympiad_verdict,
    parse_research_points,
    strip_reasoning,
)
from olmo_eval.runners.io.builders import build_predictions
from olmo_eval.runners.processing.utils import compute_task_hash

OLYMPIAD_INSTRUCTION = (
    "Think step by step and solve the problem below. At the end of your response, write your "
    "final answer on a new line starting with “FINAL ANSWER”."
)

#: Captured before ``_no_backoff_sleep`` replaces ``asyncio.sleep`` module-wide, so
#: concurrency tests can still yield to the event loop for real.
_real_sleep = asyncio.sleep


class RateLimitError(Exception):
    pass


class APITimeoutError(Exception):
    pass


class ServerError(Exception):
    status_code = 503


@pytest.fixture
def olympiad():
    return get_task("frontierscience_olympiad")


@pytest.fixture
def research():
    return get_task("frontierscience_research")


@pytest.fixture
def olympiad_mc():
    return get_task("frontierscience_olympiad:mc")


def _doc(
    problem: str = f"Find the average temperature of the Sun.\n\n{OLYMPIAD_INSTRUCTION}",
    answer: str = "`\\( 2.31 \\times 10^6 K\\)`",
    subject: str = "physics",
    task_group_id: str = "bb0539ef-d9fd-4215-bf16-b0eca44a8778",
) -> dict:
    return {
        "problem": problem,
        "answer": answer,
        "subject": subject,
        "task_group_id": task_group_id,
    }


def _response(task, subject: str = "physics", text: str = "An attempt.") -> Response:
    instance = task.process_doc(_doc(subject=subject))
    assert instance is not None
    return Response(
        instance=instance,
        request=LMRequest(request_type=RequestType.CHAT, messages=()),
        outputs=[LMOutput(text=text)],
    )


def _replies(task, *replies: str):
    """Patch the task's judge with a scripted, non-sleeping reply sequence."""
    remaining = iter(replies)
    calls: list[str] = []

    async def judge(prompt: str, **_kwargs) -> str:
        calls.append(prompt)
        return next(remaining)

    return judge, calls


@pytest.fixture(autouse=True)
def _no_backoff_sleep(monkeypatch):
    async def sleep(_delay):
        return None

    monkeypatch.setattr(frontierscience.asyncio, "sleep", sleep)


class TestRegistration:
    def test_both_tracks_and_paper_variants_are_registered(self):
        assert task_exists("frontierscience_olympiad")
        assert task_exists("frontierscience_research")

        olympiad = get_task("frontierscience_olympiad")
        research = get_task("frontierscience_research")
        assert olympiad.config.data_source.path == "openai/frontierscience"
        assert olympiad.config.data_source.data_files == "olympiad/test.jsonl"
        assert research.config.data_source.data_files == "research/test.jsonl"
        assert olympiad.config.data_source.revision == research.config.data_source.revision

        paper_olympiad = get_task("frontierscience_olympiad:paper")
        paper_research = get_task("frontierscience_research:paper")
        assert paper_olympiad.config.sampling_params.num_samples == 20
        assert paper_research.config.sampling_params.num_samples == 30
        assert paper_olympiad.config.sampling_params.temperature == 1.0

    def test_verified_mc_task_and_base_variant_are_registered(self):
        assert task_exists("frontierscience_olympiad:mc")
        assert task_exists("frontierscience_olympiad:mc:olmo3base")

        task = get_task("frontierscience_olympiad:mc")
        assert task.config.data_source.path == "frontierscience_olympiad_mc.jsonl"
        assert task.config.required_secrets == ()
        assert task.config.get_primary_metric().name == "accuracy"
        assert {metric.name for metric in task.config.metrics} == {
            "accuracy",
            "accuracy_biology",
            "accuracy_chemistry",
            "accuracy_physics",
            "gold_probability",
            "gold_probability_biology",
            "gold_probability_chemistry",
            "gold_probability_physics",
            "logprob_margin",
            "logprob_margin_biology",
            "logprob_margin_chemistry",
            "logprob_margin_physics",
        }

    def test_trials_average_rather_than_take_the_best_sample(self, olympiad, research):
        assert olympiad.config.output_score_aggregation == OutputScoreAggregation.MEAN
        assert research.config.output_score_aggregation == OutputScoreAggregation.MEAN

    def test_metrics_cover_the_primary_and_each_subject(self, olympiad, research):
        assert olympiad.config.get_primary_metric().name == "accuracy"
        assert {metric.name for metric in olympiad.config.metrics} == {
            "accuracy",
            "accuracy_biology",
            "accuracy_chemistry",
            "accuracy_physics",
        }

        assert research.config.get_primary_metric().name == "success_rate"
        assert {metric.name for metric in research.config.metrics} == {
            "success_rate",
            "rubric_score",
            "success_rate_biology",
            "success_rate_chemistry",
            "success_rate_physics",
            "rubric_score_biology",
            "rubric_score_chemistry",
            "rubric_score_physics",
        }

    def test_both_tracks_declare_the_judge_api_key(self, olympiad, research):
        assert olympiad.config.required_secrets == ("OPENAI_API_KEY",)
        assert research.config.required_secrets == ("OPENAI_API_KEY",)

    def test_judge_configuration_is_serialized(self, olympiad, research):
        olympiad_config = olympiad.config.to_dict()
        research_config = research.config.to_dict()

        assert olympiad_config["judge_model"] == FRONTIERSCIENCE_DEFAULT_JUDGE_MODEL
        assert olympiad_config["judge_reasoning_effort"] == "high"
        assert olympiad_config["judge_max_tokens"] == 8192
        assert research_config["judge_max_tokens"] == 32768

    def test_judge_configuration_changes_the_task_hash(self, monkeypatch):
        monkeypatch.delenv("OLMO_EVAL_JUDGE", raising=False)
        first = get_task("frontierscience_olympiad", config_overrides={"judge_model": "judge-a"})
        second = get_task("frontierscience_olympiad", config_overrides={"judge_model": "judge-b"})

        assert compute_task_hash(first.config.to_dict()) != compute_task_hash(
            second.config.to_dict()
        )


class TestProcessDoc:
    def test_maps_the_real_schema(self, olympiad):
        instance = olympiad.process_doc(_doc(), index=2)

        assert instance is not None
        assert instance.question.endswith("“FINAL ANSWER”.")
        assert instance.gold_answer == "`\\( 2.31 \\times 10^6 K\\)`"
        assert instance.metadata["id"] == "bb0539ef-d9fd-4215-bf16-b0eca44a8778:2"
        assert instance.metadata["task_group_id"] == "bb0539ef-d9fd-4215-bf16-b0eca44a8778"
        assert instance.metadata["subject"] == "physics"
        assert instance.metadata["index"] == 2

    def test_falls_back_to_a_positional_id(self, olympiad):
        instance = olympiad.process_doc(_doc(task_group_id=""), index=5)

        assert instance is not None
        assert instance.metadata["id"] == "frontierscience_olympiad:5"

    def test_ids_are_unique_even_though_task_group_ids_are_not(self, research):
        """The Research gold set duplicates one task_group_id byte-for-byte."""
        first = research.process_doc(_doc(task_group_id="dup"), index=11)
        second = research.process_doc(_doc(task_group_id="dup"), index=12)

        assert first is not None and second is not None
        assert first.metadata["id"] != second.metadata["id"]
        assert first.metadata["task_group_id"] == second.metadata["task_group_id"] == "dup"

    def test_skips_rows_missing_a_problem_or_answer(self, olympiad):
        assert olympiad.process_doc(_doc(problem="  ")) is None
        assert olympiad.process_doc(_doc(answer="")) is None

    def test_warns_on_an_unexpected_subject(self, olympiad, caplog):
        with caplog.at_level(logging.WARNING, logger=frontierscience.__name__):
            instance = olympiad.process_doc(_doc(subject="Geology"), index=3)

        assert instance is not None
        assert instance.metadata["subject"] == "geology"
        assert "unexpected subject 'geology'" in caplog.text


class TestPrompts:
    def test_generation_prompt_sends_the_problem_verbatim(self, olympiad):
        instance = olympiad.process_doc(_doc())
        assert instance is not None
        request = olympiad.format_request(instance)

        assert request.request_type == RequestType.CHAT
        assert len(request.messages) == 1
        assert request.messages[0] == {"role": "user", "content": instance.question}

    def test_olympiad_judge_prompt_matches_the_paper(self):
        rendered = FRONTIERSCIENCE_OLYMPIAD_JUDGE_PROMPT.format(
            problem="P", reference_answer="R", answer="A"
        )

        assert rendered.startswith("You are grading an attempted answer to a science olympiad")
        assert "6.69 ≈ 6.7" in rendered
        assert "\n***\n\nThe problem: P\n\n***\n\nThe reference answer: R\n" in rendered
        assert "\n***\n\nThe attempted answer: A\n" in rendered
        assert rendered.endswith('If it is incorrect, write "VERDICT: INCORRECT".')

    def test_research_judge_prompt_matches_the_paper(self):
        rendered = FRONTIERSCIENCE_RESEARCH_JUDGE_PROMPT.format(
            problem="P", rubric="Points: 10, Item: X", answer="A"
        )

        assert rendered.startswith("You are grading a science exam.")
        # The paper's own spelling; the prompt is reproduced as published.
        assert "Evaluate the attemped answer against the provided rubric." in rendered
        assert "\n***\n\nThe rubric: Points: 10, Item: X\n" in rendered
        assert rendered.endswith("For example, `VERDICT: 2.5` or `VERDICT: 8`.")

    def test_latex_braces_in_substituted_values_survive_formatting(self):
        rendered = FRONTIERSCIENCE_OLYMPIAD_JUDGE_PROMPT.format(
            problem="Find \\frac{a}{b}",
            reference_answer="\\(U=CT\\)",
            answer="{unbalanced",
        )

        assert "Find \\frac{a}{b}" in rendered
        assert "The attempted answer: {unbalanced" in rendered


class TestAnswerExtraction:
    def test_strips_a_closed_reasoning_block(self):
        assert strip_reasoning("<think>plan</think>\n\nThe answer.") == "The answer."
        assert strip_reasoning("  no tags  ") == "no tags"

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("Work...\nFINAL ANSWER\n42", "42"),
            ("Work...\nFINAL ANSWER: 42", "42"),
            ("Work...\n**FINAL ANSWER**\n\\(U=CT\\)", "\\(U=CT\\)"),
            ("Work...\nfinal answer 42", "42"),
            ("A first FINAL ANSWER 1\nthen FINAL ANSWER 2", "2"),
        ],
    )
    def test_keeps_only_the_last_final_answer(self, text, expected):
        assert extract_final_answer(text) == expected

    def test_falls_back_to_the_visible_response(self):
        assert extract_final_answer("No marker here.") == "No marker here."
        assert extract_final_answer("Only a marker\nFINAL ANSWER") == "Only a marker\nFINAL ANSWER"

    def test_olympiad_extracts_the_final_answer_and_research_keeps_the_derivation(
        self, olympiad, research
    ):
        output = LMOutput(text="<think>x</think>\nStep 1.\n\nFINAL ANSWER\n42")

        assert olympiad.extract_answer(output) == "42"
        assert research.extract_answer(output) == "Step 1.\n\nFINAL ANSWER\n42"


class TestVerdictParsing:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("reasoning\nVERDICT: CORRECT", 1.0),
            ("reasoning\nVERDICT: INCORRECT", 0.0),
            ("verdict: correct", 1.0),
            ("**VERDICT: INCORRECT**", 0.0),
            ("VERDICT: CORRECT\nactually VERDICT: INCORRECT", 0.0),
            ("no verdict at all", None),
            ("VERDICT: MAYBE", None),
        ],
    )
    def test_olympiad_verdicts(self, raw, expected):
        assert parse_olympiad_verdict(raw) == expected

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("tally\nVERDICT: 8", 8.0),
            ("tally\nVERDICT: 2.5", 2.5),
            ("`VERDICT: 10`", 10.0),
            ("**VERDICT: 0**", 0.0),
            ("VERDICT: 7/10", 7.0),
            ("VERDICT: 12", 10.0),
            ("VERDICT: -3", 0.0),
            ("no verdict at all", None),
        ],
    )
    def test_research_points(self, raw, expected):
        assert parse_research_points(raw) == expected


class TestOlympiadScoring:
    @pytest.mark.anyio
    async def test_judges_the_extracted_answer_against_the_reference(self, olympiad, monkeypatch):
        response = _response(olympiad, text="<think>x</think>\nWork.\n\nFINAL ANSWER\n42")
        judge, calls = _replies(olympiad, "reasoning\nVERDICT: CORRECT")
        monkeypatch.setattr(frontierscience, "build_frontierscience_judge_fn", lambda **_: judge)

        await olympiad.score_responses([response])

        assert len(calls) == 1
        assert "The attempted answer: 42" in calls[0]
        assert "The reference answer: `\\( 2.31 \\times 10^6 K\\)`" in calls[0]
        assert response.scores["accuracy"] == 1.0
        assert response.outputs[0].metadata["score:accuracy"] == 1.0
        assert response.outputs[0].metadata["score:frontierscience_judge"] == 1.0
        assert response.outputs[0].metadata["frontierscience_judge_parse_error"] is False
        saved = build_predictions([response])[0]["model_output"][0]["judge_result"]
        assert saved == {
            "scorer": "frontierscience_judge",
            "parsed_outcome": 1.0,
            "parse_error": False,
        }

    @pytest.mark.anyio
    async def test_subject_metrics_average_only_their_own_slice(self, olympiad, monkeypatch):
        responses = [
            _response(olympiad, subject="physics"),
            _response(olympiad, subject="physics"),
            _response(olympiad, subject="chemistry"),
        ]
        judge, _ = _replies(
            olympiad,
            "VERDICT: CORRECT",
            "VERDICT: INCORRECT",
            "VERDICT: CORRECT",
        )
        monkeypatch.setattr(frontierscience, "build_frontierscience_judge_fn", lambda **_: judge)

        await olympiad.score_responses(responses)

        metrics = {metric.name: metric.compute(responses) for metric in olympiad.config.metrics}
        assert metrics["accuracy"] == pytest.approx(2 / 3)
        assert metrics["accuracy_physics"] == 0.5
        assert metrics["accuracy_chemistry"] == 1.0
        # No biology instances in this slice, so the subject metric has nothing to average.
        assert metrics["accuracy_biology"] == 0.0

    @pytest.mark.anyio
    async def test_retries_then_scores_zero_on_an_unparseable_verdict(
        self, olympiad, monkeypatch, caplog
    ):
        response = _response(olympiad)
        judge, calls = _replies(olympiad, "garbage", "still garbage", "nope")
        monkeypatch.setattr(frontierscience, "build_frontierscience_judge_fn", lambda **_: judge)

        with caplog.at_level(logging.WARNING, logger=frontierscience.__name__):
            await olympiad.score_responses([response])

        assert len(calls) == 3
        assert response.scores["accuracy"] == 0.0
        assert response.outputs[0].metadata["frontierscience_judge_parse_error"] is True
        assert "no parseable verdict for 1/1 output(s)" in caplog.text
        saved = build_predictions([response])[0]["model_output"][0]["judge_result"]
        assert saved["parsed_outcome"] is None
        assert saved["parse_error"] is True
        assert saved["raw_judge_response"] == "nope"

    @pytest.mark.anyio
    async def test_warns_when_the_provider_returned_no_output(self, olympiad, monkeypatch, caplog):
        """A provider that fails every request must not look like a genuine zero."""
        responses = [_response(olympiad), _response(olympiad)]
        for response in responses:
            response.outputs = []
        judge, calls = _replies(olympiad)
        monkeypatch.setattr(frontierscience, "build_frontierscience_judge_fn", lambda **_: judge)

        with caplog.at_level(logging.WARNING, logger=frontierscience.__name__):
            await olympiad.score_responses(responses)

        assert calls == []
        assert [response.scores["accuracy"] for response in responses] == [0.0, 0.0]
        assert "no visible model output for 2/2 instance(s)" in caplog.text
        assert "capability measurement" in caplog.text

    @pytest.mark.anyio
    async def test_warns_when_reasoning_consumed_the_whole_budget(
        self, olympiad, monkeypatch, caplog
    ):
        """An output exists but its text is empty: reasoning was routed away from content."""
        answered = _response(olympiad, text="Work.\n\nFINAL ANSWER\n42")
        truncated = _response(olympiad, text="")
        judge, calls = _replies(olympiad, "VERDICT: CORRECT", "VERDICT: INCORRECT")
        monkeypatch.setattr(frontierscience, "build_frontierscience_judge_fn", lambda **_: judge)

        with caplog.at_level(logging.WARNING, logger=frontierscience.__name__):
            await olympiad.score_responses([answered, truncated])

        # The empty output is still judged, so the count is what flags the run.
        assert len(calls) == 2
        assert answered.scores["accuracy"] == 1.0
        assert truncated.scores["accuracy"] == 0.0
        assert "no visible model output for 1/2 instance(s)" in caplog.text
        assert "routes reasoning away" in caplog.text

    @pytest.mark.anyio
    async def test_recovers_on_a_retry(self, olympiad, monkeypatch):
        response = _response(olympiad)
        judge, calls = _replies(olympiad, "garbage", "VERDICT: CORRECT")
        monkeypatch.setattr(frontierscience, "build_frontierscience_judge_fn", lambda **_: judge)

        await olympiad.score_responses([response])

        assert len(calls) == 2
        assert response.scores["accuracy"] == 1.0
        assert response.outputs[0].metadata["frontierscience_judge_parse_error"] is False

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        "error",
        [
            RateLimitError("429: try again"),
            APITimeoutError("request timed out"),
            ServerError("service unavailable"),
        ],
    )
    async def test_retries_a_transient_api_error(self, olympiad, monkeypatch, error):
        response = _response(olympiad)
        calls = 0

        async def judge(_prompt: str, **_kwargs) -> str:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise error
            return "VERDICT: CORRECT"

        monkeypatch.setattr(frontierscience, "build_frontierscience_judge_fn", lambda **_: judge)

        await olympiad.score_responses([response])

        assert calls == 2
        assert response.scores["accuracy"] == 1.0

    @pytest.mark.anyio
    async def test_transient_error_does_not_consume_a_parse_attempt(self, olympiad, monkeypatch):
        response = _response(olympiad)
        replies: list[str | Exception] = [
            RateLimitError("429: try again"),
            "garbage",
            "still garbage",
            "VERDICT: CORRECT",
        ]
        calls = 0

        async def judge(_prompt: str, **_kwargs) -> str:
            nonlocal calls
            reply = replies[calls]
            calls += 1
            if isinstance(reply, Exception):
                raise reply
            return reply

        monkeypatch.setattr(frontierscience, "build_frontierscience_judge_fn", lambda **_: judge)

        await olympiad.score_responses([response])

        assert calls == 4
        assert response.scores["accuracy"] == 1.0

    @pytest.mark.anyio
    async def test_multiple_trials_average_rather_than_take_the_best(self, olympiad, monkeypatch):
        response = _response(olympiad)
        response.outputs = [
            LMOutput(text="FINAL ANSWER 1"),
            LMOutput(text="FINAL ANSWER 2"),
            LMOutput(text="FINAL ANSWER 3"),
            LMOutput(text="FINAL ANSWER 4"),
        ]
        judge, _ = _replies(
            olympiad,
            "VERDICT: CORRECT",
            "VERDICT: INCORRECT",
            "VERDICT: INCORRECT",
            "VERDICT: INCORRECT",
        )
        monkeypatch.setattr(frontierscience, "build_frontierscience_judge_fn", lambda **_: judge)

        await olympiad.score_responses([response])

        assert response.scores["accuracy"] == 0.25


class TestResearchScoring:
    @pytest.mark.anyio
    async def test_grades_the_whole_answer_against_the_rubric(self, research, monkeypatch):
        response = _response(research, text="<think>x</think>\nA full derivation.")
        judge, calls = _replies(research, "item-by-item\nVERDICT: 7.5")
        monkeypatch.setattr(frontierscience, "build_frontierscience_judge_fn", lambda **_: judge)

        await research.score_responses([response])

        assert len(calls) == 1
        assert "The attempted answer: A full derivation." in calls[0]
        assert response.scores["success_rate"] == 1.0
        assert response.scores["rubric_score"] == 0.75
        saved = build_predictions([response])[0]["model_output"][0]["judge_result"]
        assert saved["rubric_points"] == 7.5
        assert saved["parse_error"] is False

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("points", "expected_success"),
        [("6.9", 0.0), ("7", 1.0), ("7.1", 1.0)],
    )
    async def test_success_threshold_is_seven_of_ten(
        self, research, monkeypatch, points, expected_success
    ):
        response = _response(research)
        judge, _ = _replies(research, f"VERDICT: {points}")
        monkeypatch.setattr(frontierscience, "build_frontierscience_judge_fn", lambda **_: judge)

        await research.score_responses([response])

        assert response.scores["success_rate"] == expected_success
        assert response.scores["rubric_score"] == pytest.approx(float(points) / 10)

    @pytest.mark.anyio
    async def test_reports_both_metric_families_per_subject(self, research, monkeypatch):
        responses = [
            _response(research, subject="biology"),
            _response(research, subject="biology"),
            _response(research, subject="chemistry"),
        ]
        judge, _ = _replies(research, "VERDICT: 8", "VERDICT: 2", "VERDICT: 10")
        monkeypatch.setattr(frontierscience, "build_frontierscience_judge_fn", lambda **_: judge)

        await research.score_responses(responses)

        metrics = {metric.name: metric.compute(responses) for metric in research.config.metrics}
        assert metrics["success_rate"] == pytest.approx(2 / 3)
        assert metrics["success_rate_biology"] == 0.5
        assert metrics["success_rate_chemistry"] == 1.0
        assert metrics["rubric_score"] == pytest.approx(2 / 3)
        assert metrics["rubric_score_biology"] == 0.5
        assert metrics["rubric_score_chemistry"] == 1.0

    @pytest.mark.anyio
    async def test_unparseable_verdict_scores_zero_on_both_metrics(self, research, monkeypatch):
        response = _response(research)
        judge, calls = _replies(research, "garbage", "garbage", "garbage")
        monkeypatch.setattr(frontierscience, "build_frontierscience_judge_fn", lambda **_: judge)

        await research.score_responses([response])

        assert len(calls) == 3
        assert response.scores == {"success_rate": 0.0, "rubric_score": 0.0}


class TestRunnerShapedScoring:
    """The runner calls score_responses once per response, many calls concurrently."""

    @pytest.mark.anyio
    async def test_judge_concurrency_is_bounded_across_calls(self, olympiad, monkeypatch):
        context = types.SimpleNamespace(scoring_concurrency=2)
        in_flight = 0
        peak = 0

        async def judge(_prompt: str, **_kwargs) -> str:
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            try:
                # A real yield, long enough that any unbounded callers pile up here.
                await _real_sleep(0.02)
                return "VERDICT: CORRECT"
            finally:
                in_flight -= 1

        monkeypatch.setattr(frontierscience, "build_frontierscience_judge_fn", lambda **_: judge)

        # Six separate score_responses calls, as the runner would issue them.
        await asyncio.gather(
            *(olympiad.score_responses([_response(olympiad)], context=context) for _ in range(6))
        )

        assert peak <= 2

    @pytest.mark.anyio
    async def test_judge_is_built_once_per_task(self, olympiad, monkeypatch):
        builds = 0

        async def judge(_prompt: str, **_kwargs) -> str:
            return "VERDICT: CORRECT"

        def build(**_kwargs):
            nonlocal builds
            builds += 1
            return judge

        monkeypatch.setattr(frontierscience, "build_frontierscience_judge_fn", build)

        for _ in range(3):
            await olympiad.score_responses([_response(olympiad)])

        assert builds == 1

    @pytest.mark.anyio
    async def test_concurrency_is_shared_across_tracks(self, olympiad, research, monkeypatch):
        """Both tracks are scored in the same run and share one judge budget."""
        context = types.SimpleNamespace(scoring_concurrency=2)
        in_flight = 0
        peak = 0

        async def judge(_prompt: str, **_kwargs) -> str:
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            try:
                await _real_sleep(0.02)
                return "VERDICT: CORRECT"
            finally:
                in_flight -= 1

        monkeypatch.setattr(frontierscience, "build_frontierscience_judge_fn", lambda **_: judge)

        await asyncio.gather(
            *(olympiad.score_responses([_response(olympiad)], context=context) for _ in range(3)),
            *(research.score_responses([_response(research)], context=context) for _ in range(3)),
        )

        assert peak <= 2

    @pytest.mark.anyio
    async def test_limiter_is_scoped_to_the_run_not_the_process(self, olympiad, monkeypatch):
        """A second run must get its own limiter at its own configured concurrency."""

        async def judge(_prompt: str, **_kwargs) -> str:
            return "VERDICT: CORRECT"

        monkeypatch.setattr(frontierscience, "build_frontierscience_judge_fn", lambda **_: judge)

        first = ScoringContext(scoring_concurrency=2)
        second = ScoringContext(scoring_concurrency=5)
        await olympiad.score_responses([_response(olympiad)], context=first)
        await olympiad.score_responses([_response(olympiad)], context=second)

        first_limiter = frontierscience.get_judge_semaphore(first, 2)
        second_limiter = frontierscience.get_judge_semaphore(second, 5)
        assert first_limiter is not second_limiter
        assert first_limiter._value == 2
        assert second_limiter._value == 5

    def test_limiter_does_not_outlive_its_run(self):
        """The limiter must not be reachable from module state once the run is gone."""
        context = ScoringContext(scoring_concurrency=3)
        semaphore = frontierscience.get_judge_semaphore(context, 3)
        ref = weakref.ref(semaphore)

        del semaphore, context
        gc.collect()

        assert ref() is None

    @pytest.mark.anyio
    async def test_cancellation_is_not_swallowed_as_a_sample_failure(self, olympiad, monkeypatch):
        response = _response(olympiad)

        async def judge(_prompt: str, **_kwargs) -> str:
            raise asyncio.CancelledError()

        monkeypatch.setattr(frontierscience, "build_frontierscience_judge_fn", lambda **_: judge)

        with pytest.raises(asyncio.CancelledError):
            await olympiad.score_responses([response])

    @pytest.mark.anyio
    async def test_one_failed_judge_call_keeps_the_other_samples(self, olympiad, monkeypatch):
        """A non-retryable failure on one trial must not discard the whole instance."""
        response = _response(olympiad)
        response.outputs = [LMOutput(text=f"FINAL ANSWER {i}") for i in range(3)]
        failed_calls = 0

        class BadRequestError(Exception):
            pass

        async def judge(prompt: str, **_kwargs) -> str:
            nonlocal failed_calls
            if "The attempted answer: 1" in prompt:
                failed_calls += 1
                raise BadRequestError("invalid request")
            return "VERDICT: CORRECT"

        monkeypatch.setattr(frontierscience, "build_frontierscience_judge_fn", lambda **_: judge)

        await olympiad.score_responses([response])

        # Mean over the two samples that were judged, not 1/3.
        assert response.scores["accuracy"] == 1.0
        assert failed_calls == 1
        # Recorded through the standard persisted scoring-error channel.
        errors = response.outputs[1].metadata["scoring_errors"]["frontierscience_judge"]
        assert errors["phase"] == "judge"
        assert "invalid request" in errors["message"]

    @pytest.mark.anyio
    async def test_total_judge_failure_raises_and_the_runner_scores_it_zero(
        self, olympiad, monkeypatch
    ):
        """Raising cannot exclude the instance; document what the runner actually does."""
        from olmo_eval.runners.asynq.results import _record_scoring_failure

        response = _response(olympiad)
        calls = 0

        async def judge(_prompt: str, **_kwargs) -> str:
            nonlocal calls
            calls += 1
            raise RuntimeError("rate limit exceeded")

        monkeypatch.setattr(frontierscience, "build_frontierscience_judge_fn", lambda **_: judge)

        with pytest.raises(RuntimeError, match="failed for every sample"):
            await olympiad.score_responses([response])
        assert calls == 4

        # The runner catches, annotates, and keeps the response, so the instance lands
        # in the metric as 0.0 rather than being dropped from the denominator.
        _record_scoring_failure(
            response,
            scorer_names=list(olympiad._get_scorers()),
            error={"phase": "response", "type": "RuntimeError"},
        )
        assert response.outputs[0].metadata["scoring_errors"]["__response__"]["phase"] == "response"
        assert olympiad.config.get_primary_metric().compute([response]) == 0.0


class TestComputeMetrics:
    @pytest.mark.anyio
    async def test_metrics_land_under_the_shared_scorer_channel(self, research, monkeypatch):
        response = _response(research)
        judge, _ = _replies(research, "VERDICT: 8")
        monkeypatch.setattr(frontierscience, "build_frontierscience_judge_fn", lambda **_: judge)

        await research.score_responses([response])
        computed = research.compute_metrics([response])

        assert computed["success_rate"] == {"frontierscience_judge": 1.0}
        assert computed["rubric_score"] == {"frontierscience_judge": 0.8}


def test_judge_builder_uses_serialized_config(monkeypatch):
    captured = {}

    async def sentinel(_prompt):
        return ""

    def fake_builder(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.delenv("OLMO_EVAL_JUDGE", raising=False)
    monkeypatch.setattr(frontierscience, "build_openai_judge_fn", fake_builder)
    task = get_task(
        "frontierscience_olympiad",
        config_overrides={
            "judge_model": "serialized-judge",
            "judge_reasoning_effort": "medium",
            "judge_max_tokens": 1234,
        },
    )
    assert isinstance(task, frontierscience.FrontierScienceOlympiad)

    built = task._get_judge_fn()

    assert built is sentinel
    assert captured == {
        "model": "serialized-judge",
        "temperature": 0.0,
        "max_tokens": 1234,
        "scorer_name": "FrontierScienceOlympiad",
        "reasoning_effort": "medium",
    }


@pytest.mark.parametrize(
    ("env_spec", "expected_model", "expected_effort"),
    [
        ("judge-override", "judge-override", None),
        ("gpt-5:high", "gpt-5", "high"),
    ],
)
def test_environment_judge_override_is_resolved_into_config(
    monkeypatch, env_spec, expected_model, expected_effort
):
    monkeypatch.setenv("OLMO_EVAL_JUDGE", env_spec)

    config = get_task("frontierscience_olympiad").config.to_dict()

    assert config["judge_model"] == expected_model
    assert config["judge_reasoning_effort"] == expected_effort
    assert config["judge_max_tokens"] == 8192


def test_metric_subject_slice_stays_out_of_pairwise_scorer_fallback():
    accuracy = next(
        metric
        for metric in get_task("frontierscience_olympiad").config.metrics
        if metric.name == "accuracy_physics"
    )
    instance = Instance(question="q", metadata={"subject": "physics"})
    response = Response(
        instance=instance,
        request=LMRequest(request_type=RequestType.CHAT, messages=()),
        outputs=[LMOutput(text="")],
        scores={"accuracy": 1.0},
    )

    assert accuracy.compute([response]) == 1.0
    assert accuracy.compute_instance(response) is None
    assert accuracy.display_format() == "percentage"


class TestVerifiedMultipleChoice:
    def test_release_has_expected_rows_subjects_and_balanced_labels(self, olympiad_mc):
        instances = list(olympiad_mc.instances)

        assert len(instances) == 83
        assert len({instance.metadata["task_group_id"] for instance in instances}) == 83
        assert Counter(instance.metadata["subject"] for instance in instances) == {
            "physics": 40,
            "chemistry": 34,
            "biology": 9,
        }
        assert Counter(instance.gold_answer for instance in instances) == {
            "A": 21,
            "B": 20,
            "C": 21,
            "D": 21,
        }
        assert max(Counter(instance.gold_answer for instance in instances).values()) / 83 == (
            21 / 83
        )

        for instance in instances:
            assert len(instance.choices or ()) == 4
            assert instance.gold_answer == chr(ord("A") + instance.metadata["gold_idx"])
            assert instance.metadata["gold_text"] == instance.choices[instance.metadata["gold_idx"]]
            assert "Think step by step and solve the problem below" not in instance.question

    def test_prompt_uses_displayed_choices_and_label_continuations(self, olympiad_mc):
        instance = next(iter(olympiad_mc.instances))

        request = olympiad_mc.format_request(instance)

        assert request.request_type == RequestType.LOGLIKELIHOOD
        assert request.prompt is not None
        assert request.prompt.startswith("Problem:\n")
        assert "\n\nChoices:\nA. " + instance.choices[0] in request.prompt
        assert "\nB. " + instance.choices[1] in request.prompt
        assert request.prompt.endswith("\n\nAnswer:")
        assert request.continuations == (" A", " B", " C", " D")

    def test_metrics_report_accuracy_probability_margin_and_subject_slices(self, olympiad_mc):
        instance = Instance(
            question="q",
            choices=("a", "b", "c", "d"),
            gold_answer="B",
            metadata={"gold_idx": 1, "subject": "physics"},
        )
        scores = (-2.0, -1.0, -3.0, -4.0)
        response = Response(
            instance=instance,
            request=LMRequest(request_type=RequestType.LOGLIKELIHOOD, prompt="q"),
            outputs=[LMOutput(text="", logprobs=[{"logprob": score}]) for score in scores],
        )
        metrics = {metric.name: metric for metric in olympiad_mc.config.metrics}

        assert metrics["accuracy"].compute([response]) == 1.0
        assert metrics["accuracy_physics"].compute([response]) == 1.0
        assert metrics["accuracy_biology"].compute([response]) == 0.0
        expected_probability = math.exp(-1.0) / sum(math.exp(score) for score in scores)
        assert metrics["gold_probability"].compute([response]) == pytest.approx(
            expected_probability
        )
        assert metrics["logprob_margin"].compute([response]) == 1.0
        assert metrics["logprob_margin"].metric_unit() == "logprob"
