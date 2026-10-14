"""Tests for the shared judged-task base."""

import types
from collections.abc import Iterator
from typing import Any

import pytest

from olmo_eval.common.scorers import ScoringIncompleteError
from olmo_eval.common.types import Instance, LMOutput, LMRequest, RequestType, Response
from olmo_eval.evals.tasks.common import OutputScoreAggregation, TaskConfig
from olmo_eval.evals.tasks.common import judge as judge_module
from olmo_eval.evals.tasks.common.judge import (
    JUDGE_ATTEMPTS,
    JudgedMacroScoreMetric,
    JudgedScoreMetric,
    JudgedTask,
    TruncationRateMetric,
    is_truncated,
    parse_judge_spec,
    visible_response,
)

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

    async def judge(prompt: str, **_kwargs) -> str:
        calls.append(prompt)
        reply = replies[prompt].pop(0)
        if reply == "raise":
            raise ValueError("judge unavailable")
        return reply

    return judge, calls


@pytest.fixture(autouse=True)
def _no_judge_env(monkeypatch):
    monkeypatch.delenv("OLMO_EVAL_JUDGE", raising=False)


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch):
    async def sleep(_delay):
        return None

    monkeypatch.setattr(judge_module.asyncio, "sleep", sleep)
    monkeypatch.setattr(judge_module, "retry_with_backoff", lambda call, **_: call())


def _use_judge(monkeypatch, judge) -> None:
    monkeypatch.setattr(judge_module, "build_openai_judge_fn", lambda **_: judge)


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
        _use_judge(monkeypatch, judge)
        response = _response("good", "bad")

        await _PassFailTask(_config()).score_responses([response])

        assert response.scores == {"accuracy": 0.5, "truncation_rate": 0.0}
        assert [output.metadata["score:judge"] for output in response.outputs] == [1.0, 0.0]
        assert response.outputs[0].metadata["judge_result"]["raw_judge_response"] == "PASS"

    @pytest.mark.anyio
    async def test_retries_an_unparseable_reply(self, monkeypatch):
        judge, calls = _scripted_judge({"good": ["hmm", "PASS"]})
        _use_judge(monkeypatch, judge)
        response = _response("good")

        await _PassFailTask(_config()).score_responses([response])

        assert len(calls) == 2
        assert response.scores["accuracy"] == 1.0

    @pytest.mark.anyio
    async def test_a_failed_sample_leaves_the_instance_average(self, monkeypatch):
        judge, _ = _scripted_judge({"good": ["PASS"], "broken": ["raise"]})
        _use_judge(monkeypatch, judge)
        response = _response("good", "broken")

        await _PassFailTask(_config()).score_responses([response])

        assert response.scores["accuracy"] == 1.0
        error = response.outputs[1].metadata["scoring_errors"]["judge"]
        assert error["infrastructure"] == "true"
        assert "ValueError: judge unavailable" in error["message"]

    @pytest.mark.anyio
    async def test_an_instance_with_no_judged_sample_is_incomplete(self, monkeypatch):
        judge, _ = _scripted_judge({"garbled": ["?"] * JUDGE_ATTEMPTS})
        _use_judge(monkeypatch, judge)
        response = _response("garbled")

        with pytest.raises(ScoringIncompleteError, match="failed for every sample"):
            await _PassFailTask(_config()).score_responses([response])

        output = response.outputs[0]
        assert output.metadata["scoring_errors"]["judge"]["type"] == "JudgeParseError"
        assert output.metadata["judge_result"]["parse_error"] is True
        assert output.metadata["judge_result"]["raw_judge_response"] == "?"
        assert "accuracy" not in response.scores

    @pytest.mark.anyio
    async def test_an_instance_without_outputs_scores_as_fully_truncated(self, monkeypatch):
        judge, calls = _scripted_judge({})
        _use_judge(monkeypatch, judge)
        response = _response()

        await _PassFailTask(_config()).score_responses([response])

        assert calls == []
        assert response.scores == {"accuracy": 0.0, "truncation_rate": 1.0}


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
