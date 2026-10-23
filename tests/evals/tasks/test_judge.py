"""Tests for the shared judged-task base."""

import asyncio
import types
from collections.abc import Iterator
from typing import Any

import pytest

from olmo_eval.common.multi_turn import scripted_chat_request
from olmo_eval.common.scorers import llm_judge
from olmo_eval.common.scorers.llm_judge import parse_judge_spec
from olmo_eval.common.scoring_failures import ScoringState, scoring_state
from olmo_eval.common.types import (
    TURNS_KEY,
    Instance,
    LMOutput,
    LMRequest,
    RequestType,
    Response,
)
from olmo_eval.evals.tasks.common import OutputScoreAggregation, TaskConfig
from olmo_eval.evals.tasks.common import judge as judge_module
from olmo_eval.evals.tasks.common.judge import (
    JUDGE_ATTEMPTS,
    JudgedMacroScoreMetric,
    JudgedScoreMetric,
    JudgedTask,
    TruncationRateMetric,
    is_truncated,
    visible_response,
)
from olmo_eval.evals.tasks.hle import HLEAccuracyMetric, HLETruncationRateMetric
from olmo_eval.inference.retry import retry_with_backoff
from tests.evals.tasks.judge_stubs import use_judge

pytestmark = pytest.mark.usefixtures("judge_test_env")

DEFAULT_JUDGE = "judge-default"


class _PassFailTask(JudgedTask):
    """Passes an output when the judge replies PASS; FAIL is a fail, anything else unparsed."""

    judge_model = DEFAULT_JUDGE

    @property
    def instances(self) -> Iterator[Instance]:
        yield from ()

    def format_request(self, instance: Instance) -> LMRequest:
        return LMRequest(request_type=RequestType.CHAT, messages=())

    async def _score_output(self, response: Response, output: LMOutput, judge_fn) -> float:
        judge_result: dict[str, Any] = {"scorer": self.judge_scorer_name}
        verdict = await self._judge_output(
            output,
            judge_fn,
            self._judged_answer(output),
            lambda raw: raw if raw in {"PASS", "FAIL"} else None,
            judge_result,
        )
        return self._record_judgement(
            output, judge_result, {"accuracy": 1.0 if verdict == "PASS" else 0.0}
        )


class _RatingTask(_PassFailTask):
    """Rates each output 1-10 in two concurrent judge calls with a system prompt.

    Each call's rating is kept as its own score, next to their mean.
    """

    judge_reasoning_effort = "medium"
    judge_score_key = "rating"
    judge_extra_score_keys = ("forward", "reversed")
    unanswered_score = 1.0

    async def _score_output(self, response: Response, output: LMOutput, judge_fn) -> float:
        judge_result: dict[str, Any] = {}
        ratings = await self._judge_together(
            *(
                self._judge_output(
                    output,
                    judge_fn,
                    f"{self._judged_answer(output)} ({order})",
                    lambda raw: float(raw) if raw.isdigit() else None,
                    judge_result,
                    system_prompt="Rate 1-10.",
                    raw_key=f"raw_judge_response_{order}",
                )
                for order in ("forward", "reversed")
            )
        )
        forward, reversed_ = ratings
        return self._record_judgement(
            output,
            judge_result,
            {"rating": (forward + reversed_) / 2, "forward": forward, "reversed": reversed_},
        )


def _config(**overrides) -> TaskConfig:
    values: dict[str, Any] = {
        "name": "pass_fail",
        "judge_model": DEFAULT_JUDGE,
        "judge_max_tokens": 64,
        "output_score_aggregation": OutputScoreAggregation.MEAN,
    }
    values.update(overrides)
    return TaskConfig(**values)


def _response(*texts: str, group: str = "a") -> Response:
    instance = Instance(question="q", gold_answer=None, metadata={"id": f"id-{group}", "g": group})
    return Response(
        instance=instance,
        request=LMRequest(request_type=RequestType.CHAT, messages=()),
        outputs=[LMOutput(text=text) for text in texts],
    )


def _scored(accuracy: float | None, group: str = "a", truncation_rate: float = 0.0) -> Response:
    response = _response("x", group=group)
    if accuracy is not None:
        response.scores["accuracy"] = accuracy
    response.scores["truncation_rate"] = truncation_rate
    return response


def _scripted_judge(replies: dict[str, list[str]]):
    """A judge that answers each prompt from its own queue of replies."""
    calls: list[str] = []

    async def judge(prompt: str, *, system_prompt: str | None = None) -> str:
        calls.append(prompt if system_prompt is None else f"[{system_prompt}] {prompt}")
        reply = replies[prompt].pop(0)
        if reply == "raise":
            raise ValueError("judge unavailable")
        return reply

    return judge, calls


class TestHelpers:
    def test_parse_judge_spec_splits_model_and_effort(self):
        assert parse_judge_spec("gpt-5:high") == ("gpt-5", "high")
        assert parse_judge_spec("gpt-4o") == ("gpt-4o", None)
        assert parse_judge_spec("gpt-4o:") == ("gpt-4o", None)

    def test_parse_judge_spec_requires_a_model(self):
        with pytest.raises(ValueError, match="must name a judge model"):
            parse_judge_spec(":high")

    def test_visible_response_drops_a_closed_reasoning_block(self):
        assert visible_response("<think>work</think>\nAnswer: 42") == "Answer: 42"

    def test_visible_response_is_empty_for_an_unclosed_reasoning_block(self):
        assert visible_response("<think>so the answer is probably 41") == ""

    def test_visible_response_keeps_untagged_text(self):
        assert visible_response("  Answer: 42\n") == "Answer: 42"

    def test_is_truncated_reads_the_finish_reason(self):
        assert is_truncated(LMOutput(text="x", metadata={"finish_reason": "length"}))
        assert not is_truncated(LMOutput(text="x", metadata={"finish_reason": "stop"}))
        assert not is_truncated(LMOutput(text="x"))


class TestJudgeConfiguration:
    def test_environment_override_replaces_the_default_judge(self, monkeypatch):
        monkeypatch.setenv("OLMO_EVAL_JUDGE", "gpt-5:high")

        config = _PassFailTask(_config()).config

        assert config.judge_model == "gpt-5"
        assert config.judge_reasoning_effort == "high"

    def test_task_override_wins_over_the_environment(self, monkeypatch):
        monkeypatch.setenv("OLMO_EVAL_JUDGE", "gpt-5:high")

        config = _PassFailTask(_config(judge_model="judge-a")).config

        assert config.judge_model == "judge-a"
        assert config.judge_reasoning_effort is None

    def test_environment_effort_replaces_a_task_default_effort(self, monkeypatch):
        monkeypatch.setenv("OLMO_EVAL_JUDGE", "gpt-5:high")

        config = _RatingTask(_config(judge_reasoning_effort="medium")).config

        assert (config.judge_model, config.judge_reasoning_effort) == ("gpt-5", "high")

    def test_environment_model_without_an_effort_drops_a_task_default_effort(self, monkeypatch):
        monkeypatch.setenv("OLMO_EVAL_JUDGE", "gpt-4o")

        config = _RatingTask(_config(judge_reasoning_effort="medium")).config

        assert (config.judge_model, config.judge_reasoning_effort) == ("gpt-4o", None)

    def test_an_effort_override_equal_to_the_default_counts_as_unset(self, monkeypatch):
        monkeypatch.setenv("OLMO_EVAL_JUDGE", "gpt-4o")

        config = _RatingTask(_config(judge_reasoning_effort="medium")).config

        assert config.judge_reasoning_effort is None

    def test_task_effort_override_wins_over_the_environment(self, monkeypatch):
        monkeypatch.setenv("OLMO_EVAL_JUDGE", "gpt-5:high")

        config = _RatingTask(_config(judge_reasoning_effort="low")).config

        assert (config.judge_model, config.judge_reasoning_effort) == ("gpt-5", "low")

    def test_judge_is_built_once_from_the_config(self, monkeypatch):
        built: list[dict[str, Any]] = []

        def builder(**kwargs):
            built.append(kwargs)
            return object()

        monkeypatch.setattr(judge_module, "build_openai_judge_fn", builder)
        task = _PassFailTask(_config(judge_reasoning_effort="low"))

        assert task._get_judge_fn() is task._get_judge_fn()
        assert len(built) == 1
        assert built[0]["model"] == DEFAULT_JUDGE
        assert built[0]["max_tokens"] == 64
        assert built[0]["reasoning_effort"] == "low"
        assert built[0]["temperature"] == 0.0

    def test_incomplete_judge_configuration_is_rejected(self):
        task = _PassFailTask(_config(judge_max_tokens=None))

        with pytest.raises(ValueError, match="complete judge configuration"):
            task._get_judge_fn()

    def test_judge_limiter_is_shared_through_the_scoring_context(self):
        context = types.SimpleNamespace(scoring_concurrency=2)
        first = _PassFailTask(_config())
        second = _PassFailTask(_config(name="other"))

        semaphore = first._get_judge_semaphore(context)

        assert second._get_judge_semaphore(context) is semaphore
        assert first._get_judge_semaphore(None) is not semaphore


class TestScoreResponses:
    @pytest.mark.anyio
    async def test_averages_judged_samples_and_reports_truncation(self, monkeypatch):
        judge, _ = _scripted_judge({"good": ["PASS"], "bad": ["FAIL"]})
        use_judge(monkeypatch, judge)
        response = _response("good", "bad")

        await _PassFailTask(_config()).score_responses([response])

        assert response.scores == {"accuracy": 0.5, "truncation_rate": 0.0}
        assert [output.metadata["score:judge"] for output in response.outputs] == [1.0, 0.0]
        assert response.outputs[0].metadata["judge_result"]["raw_judge_response"] == "PASS"

    @pytest.mark.anyio
    async def test_retries_an_unparseable_reply(self, monkeypatch):
        judge, calls = _scripted_judge({"good": ["hmm", "PASS"]})
        use_judge(monkeypatch, judge)
        response = _response("good")

        await _PassFailTask(_config()).score_responses([response])

        assert len(calls) == 2
        assert response.scores["accuracy"] == 1.0

    @pytest.mark.anyio
    async def test_a_failed_sample_leaves_the_instance_average(self, monkeypatch):
        judge, _ = _scripted_judge({"good": ["PASS"], "broken": ["raise"]})
        use_judge(monkeypatch, judge)
        response = _response("good", "broken")

        await _PassFailTask(_config()).score_responses([response])

        assert response.scores["accuracy"] == 1.0
        error = response.outputs[1].metadata["scoring_errors"]["judge"]
        assert error["infrastructure"] == "true"
        assert "ValueError: judge unavailable" in error["message"]

    @pytest.mark.anyio
    async def test_an_instance_with_no_judged_sample_is_unscored(self, monkeypatch):
        judge, _ = _scripted_judge({"garbled": ["?"] * JUDGE_ATTEMPTS})
        use_judge(monkeypatch, judge)
        response = _response("garbled")

        await _PassFailTask(_config()).score_responses([response])

        assert scoring_state(response) is ScoringState.UNSCORED

        output = response.outputs[0]
        assert output.metadata["scoring_errors"]["judge"]["type"] == "JudgeParseError"
        assert output.metadata["judge_result"]["parse_error"] is True
        assert output.metadata["judge_result"]["raw_judge_response"] == "?"
        assert "accuracy" not in response.scores

    @pytest.mark.anyio
    async def test_a_custom_score_key_with_a_system_prompt_and_two_judge_calls(self, monkeypatch):
        judge, calls = _scripted_judge({"answer (forward)": ["8"], "answer (reversed)": ["6"]})
        use_judge(monkeypatch, judge)
        response = _response("answer")

        await _RatingTask(_config()).score_responses([response])

        assert sorted(calls) == [
            "[Rate 1-10.] answer (forward)",
            "[Rate 1-10.] answer (reversed)",
        ]
        assert response.scores == {
            "rating": 7.0,
            "forward": 8.0,
            "reversed": 6.0,
            "truncation_rate": 0.0,
        }
        output = response.outputs[0]
        assert output.metadata["score:judge"] == 7.0
        assert output.metadata["judge_result"] == {
            "raw_judge_response_forward": "8",
            "raw_judge_response_reversed": "6",
        }

    @pytest.mark.anyio
    async def test_extra_scores_are_aggregated_over_the_judged_samples(self, monkeypatch):
        judge, _ = _scripted_judge(
            {
                "a (forward)": ["8"],
                "a (reversed)": ["6"],
                "b (forward)": ["4"],
                "b (reversed)": ["2"],
                "c (forward)": ["?"] * JUDGE_ATTEMPTS,
                "c (reversed)": ["9"],
            }
        )
        use_judge(monkeypatch, judge)
        response = _response("a", "b", "c")

        await _RatingTask(_config()).score_responses([response])

        assert response.scores["rating"] == 5.0
        assert (response.scores["forward"], response.scores["reversed"]) == (6.0, 4.0)
        failed = response.outputs[2].metadata
        assert failed["judge_result"]["raw_judge_response_reversed"] == "9"
        assert "score:forward" not in failed

    @pytest.mark.anyio
    async def test_an_instance_without_outputs_gets_the_tasks_unanswered_score(self, monkeypatch):
        judge, _ = _scripted_judge({})
        use_judge(monkeypatch, judge)
        response = _response()

        await _RatingTask(_config()).score_responses([response])

        assert response.scores == {
            "rating": 1.0,
            "forward": 1.0,
            "reversed": 1.0,
            "truncation_rate": 1.0,
        }

    @pytest.mark.anyio
    async def test_a_conversation_cut_short_by_the_context_counts_as_truncated(
        self, monkeypatch, caplog
    ):
        judge, _ = _scripted_judge({"first": ["PASS"], "both": ["PASS"]})
        use_judge(monkeypatch, judge)
        cut_short, finished = _response("first"), _response("both")
        for response, replies in ((cut_short, ["first"]), (finished, ["one", "both"])):
            response.request = scripted_chat_request(["turn 1", "turn 2"])
            response.outputs[0].metadata[TURNS_KEY] = [{"text": r} for r in replies]

        with caplog.at_level("WARNING", logger=judge_module.__name__):
            await _PassFailTask(_config()).score_responses([cut_short, finished])

        assert cut_short.scores["truncation_rate"] == 1.0
        assert finished.scores["truncation_rate"] == 0.0
        assert "hit the token or context limit on 1 output(s)" in caplog.text

    @pytest.mark.anyio
    async def test_an_instance_without_outputs_scores_as_fully_truncated(self, monkeypatch):
        judge, calls = _scripted_judge({})
        use_judge(monkeypatch, judge)
        response = _response()

        await _PassFailTask(_config()).score_responses([response])

        assert calls == []
        assert response.scores == {"accuracy": 0.0, "truncation_rate": 1.0}


class TestJudgeCalls:
    @pytest.mark.anyio
    async def test_scoring_concurrency_bounds_calls_however_many_each_output_makes(
        self, monkeypatch
    ):
        in_flight = peak = 0

        async def judge(prompt: str, *, system_prompt: str | None = None) -> str:
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            # Yield to the event loop so that any other admitted call can start.
            loop = asyncio.get_running_loop()
            future = loop.create_future()
            loop.call_soon(future.set_result, None)
            await future
            in_flight -= 1
            return "5"

        use_judge(monkeypatch, judge)
        responses = [_response("a", "b"), _response("c", "d", group="b")]

        await _RatingTask(_config()).score_responses(
            responses, context=types.SimpleNamespace(scoring_concurrency=3)
        )

        assert peak == 3
        assert all(response.scores["rating"] == 5.0 for response in responses)

    @pytest.mark.anyio
    async def test_a_failed_call_waits_for_the_outputs_other_calls(self, monkeypatch):
        judge, calls = _scripted_judge({"answer (forward)": ["raise"], "answer (reversed)": ["6"]})
        use_judge(monkeypatch, judge)
        response = _response("answer")

        await _RatingTask(_config()).score_responses([response])

        assert len(calls) == 2
        error = response.outputs[0].metadata["scoring_errors"]["judge"]
        assert "ValueError: judge unavailable" in error["message"]
        assert "score:judge" not in response.outputs[0].metadata
        assert scoring_state(response) is ScoringState.UNSCORED

    @pytest.mark.anyio
    async def test_a_transient_api_error_is_retried(self, monkeypatch):
        monkeypatch.setattr(llm_judge, "retry_with_backoff", retry_with_backoff)
        replies: list[str | Exception] = [TimeoutError("request timed out"), "PASS"]

        async def judge(prompt: str, *, system_prompt: str | None = None) -> str:
            reply = replies.pop(0)
            if isinstance(reply, Exception):
                raise reply
            return reply

        use_judge(monkeypatch, judge)
        response = _response("good")

        await _PassFailTask(_config()).score_responses([response])

        assert replies == []
        assert response.scores["accuracy"] == 1.0

    @pytest.mark.anyio
    async def test_an_unrecorded_extra_score_is_named(self, monkeypatch):
        class _ForgetfulTask(_RatingTask):
            judge_extra_score_keys = ("forward", "reversed", "consensus")

        judge, _ = _scripted_judge({"answer (forward)": ["8"], "answer (reversed)": ["6"]})
        use_judge(monkeypatch, judge)
        response = _response("answer")

        await _ForgetfulTask(_config()).score_responses([response])

        error = response.outputs[0].metadata["scoring_errors"]["judge"]
        assert "recorded no consensus score" in error["message"]


class TestMetrics:
    def test_score_metric_leaves_out_unjudged_instances(self):
        responses = [_scored(1.0), _scored(0.0), _scored(None)]

        assert JudgedScoreMetric().compute(responses) == 0.5
        assert JudgedScoreMetric().compute_instance(responses[2]) is None

    def test_score_metric_restricts_to_a_slice(self):
        metric = JudgedScoreMetric(name="accuracy_b", metadata_key="g", metadata_value="b")
        responses = [_scored(1.0, group="a"), _scored(0.0, group="b")]

        assert metric.compute(responses) == 0.0
        assert metric.compute_instance(responses[0]) is None
        assert metric.compute([responses[0]]) == 0.0

    def test_truncation_rate_is_lower_is_better(self):
        metric = TruncationRateMetric()
        responses = [_scored(1.0, truncation_rate=0.0), _scored(0.0, truncation_rate=1.0)]

        assert metric.compute(responses) == 0.5
        assert metric.higher_is_better() is False

    def test_score_metric_can_display_raw_units(self):
        metric = JudgedScoreMetric(name="rating", score_key="rating", display="raw")
        response = _response("x")
        response.scores["rating"] = 7.5

        assert metric.compute([response]) == 7.5
        assert metric.display_format() == "raw"
        assert metric.metric_unit() == "rating"
        assert JudgedScoreMetric().display_format() == "percentage"
        assert JudgedScoreMetric().metric_unit() == "proportion"

    def test_score_metric_can_name_its_unit(self):
        metric = JudgedScoreMetric(name="score", display="raw", unit="ten_point_rating")

        assert metric.metric_unit() == "ten_point_rating"

    def test_serialization_ignores_the_new_fields(self):
        # Task hashes include each metric's serialized form, so HLE's must not change.
        assert HLEAccuracyMetric().to_dict() == {
            "type": "HLEAccuracyMetric",
            "name": "accuracy",
            "scorer": "HLEScorer",
        }
        assert HLETruncationRateMetric().to_dict() == {
            "type": "HLETruncationRateMetric",
            "name": "truncation_rate",
            "scorer": "HLEScorer",
        }

    def test_macro_metric_requires_a_key_and_groups(self):
        with pytest.raises(ValueError, match="metadata_key and at least one group"):
            JudgedMacroScoreMetric()
        with pytest.raises(ValueError, match="metadata_key and at least one group"):
            JudgedMacroScoreMetric(metadata_key="g")

    def test_macro_metric_weights_groups_equally(self):
        metric = JudgedMacroScoreMetric(metadata_key="g", groups=("a", "b", "c"))
        responses = [
            _scored(1.0, group="a"),
            _scored(1.0, group="a"),
            _scored(1.0, group="a"),
            _scored(0.0, group="b"),
            _scored(None, group="c"),
        ]

        # Group a passes, group b fails, and group c has nothing judged.
        assert metric.compute(responses) == 0.5
        assert metric.compute_instance(responses[0]) is None
        assert metric.compute([]) == 0.0
