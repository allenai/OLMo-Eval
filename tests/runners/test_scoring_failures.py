"""Tests for how the runner treats instances whose scores could not be computed."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from olmo_eval.common.scoring_failures import (
    MAX_ERROR_DESCRIPTION_CHARS,
    ScoringState,
    count_scoring_failures,
    describe_error,
    measurable_responses,
    scoring_state,
)
from olmo_eval.common.types import Instance, LMOutput, LMRequest, RequestType, Response
from olmo_eval.evals.tasks.common import TaskConfig
from olmo_eval.runners.asynq.preparation import compute_task_metrics

ERROR = {"phase": "execution", "type": "RuntimeError", "message": "scorer crashed"}


def _output(*, failed: bool = False) -> LMOutput:
    if failed:
        return LMOutput(text="out", metadata={"scoring_errors": {"code_exec": dict(ERROR)}})
    return LMOutput(text="out", metadata={"score:code_exec": 1.0})


def _response(*outputs: LMOutput, scoring_error: dict[str, str] | None = None) -> Response:
    return Response(
        instance=Instance(question="q", gold_answer="a"),
        request=LMRequest(request_type=RequestType.COMPLETION, prompt="q"),
        outputs=list(outputs),
        scoring_error=scoring_error,
    )


class _RecordingTask:
    """Stands in for a task, recording which responses its metrics were computed on."""

    config = TaskConfig(name="test", data_source="test/data")

    def __init__(self) -> None:
        self.measured: list[Response] = []

    def _get_scorers(self) -> dict[str, object]:
        return {"code_exec": object()}

    def compute_metrics(self, responses: list[Response]) -> dict[str, dict[str, float]]:
        self.measured = list(responses)
        return {"accuracy": {"code_exec": 1.0}}


def test_an_instance_is_scored_partially_scored_or_unscored() -> None:
    assert scoring_state(_response(_output(), _output())) is ScoringState.SCORED
    assert scoring_state(_response(_output(), _output(failed=True))) is ScoringState.PARTIAL
    assert scoring_state(_response(_output(failed=True))) is ScoringState.UNSCORED


def test_an_instance_without_outputs_is_unscored_only_if_scoring_it_failed() -> None:
    assert scoring_state(_response()) is ScoringState.SCORED
    assert scoring_state(_response(scoring_error=dict(ERROR))) is ScoringState.UNSCORED


def test_only_unscored_instances_are_left_out_of_the_metrics() -> None:
    scored = _response(_output())
    partial = _response(_output(), _output(failed=True))
    unscored = _response(_output(failed=True))

    assert measurable_responses([scored, partial, unscored]) == [scored, partial]


def test_the_runner_computes_metrics_without_the_unscored_instances() -> None:
    task = _RecordingTask()
    scored, unscored = _response(_output()), _response(_output(failed=True))

    result = compute_task_metrics("test", task, [scored, unscored], {}, 2, 0.0)  # type: ignore[arg-type]

    assert task.measured == [scored]
    # Both instances are still saved to the predictions, the failed one with its error.
    assert result.num_instances == 2
    assert result.predictions is not None
    assert result.predictions[1]["model_output"][0]["scoring_errors"]["code_exec"] == ERROR


def test_failures_are_counted_by_instance_output_and_type() -> None:
    infra = {"type": "SandboxInfrastructureError", "infrastructure": "true"}
    responses = [
        _response(_output(failed=True)),
        _response(_output(), _output(failed=True)),
        _response(
            LMOutput(text="o", metadata={"scoring_errors": {"code_exec": infra}}),
        ),
        _response(scoring_error={"type": "KeyError", "message": "'gold'"}),
        _response(_output()),
    ]

    failures = count_scoring_failures(responses)

    assert failures.to_dict() == {
        "instances_unscored": 3,
        "instances_partially_scored": 1,
        "outputs_unscored": 3,
        "infrastructure": 1,
        "by_type": {"KeyError": 1, "RuntimeError": 2, "SandboxInfrastructureError": 1},
        "first_error": "RuntimeError: scorer crashed",
    }


def test_the_quoted_error_is_capped() -> None:
    error = {"type": "ValueError", "message": "x" * 10_000}

    described = describe_error(error)

    assert len(described) == MAX_ERROR_DESCRIPTION_CHARS
    assert described.startswith("ValueError: xxx")
    assert described.endswith("…")


def test_the_task_line_and_metrics_json_report_the_failures() -> None:
    from olmo_eval.runners.asynq.results import _format_instance_accounting, aggregate_results
    from olmo_eval.runners.processing.metrics import build_single_model_metrics

    responses = [_response(_output()) for _ in range(30)]
    responses += [_response(_output(failed=True)), _response(_output(), _output(failed=True))]
    result = compute_task_metrics("t", _RecordingTask(), responses, {}, 32, 0.0)  # type: ignore[arg-type]

    assert result.error is None
    assert "1 unscored, 1 partially scored" in _format_instance_accounting(result)

    provider = SimpleNamespace(alias="m", model="org/m", kind="vllm_server", to_dict=dict)
    results = aggregate_results(
        results={"t": result},
        expanded_tasks=["t"],
        task_specs=["t"],
        provider_config=provider,
        attention_backend=None,
    )
    entry = build_single_model_metrics(results).to_dict()["tasks"][0]
    assert entry["scoring_failures"]["instances_unscored"] == 1
    assert entry["scoring_failures"]["instances_partially_scored"] == 1


def test_the_summary_warns_about_published_tasks_only() -> None:
    from olmo_eval.runners.processing.metrics import scoring_failure_warnings

    failures = {
        "instances_unscored": 1,
        "instances_partially_scored": 2,
        "outputs_unscored": 3,
        "first_error": "KeyError: 'gold'",
    }
    results = {
        "tasks": {
            "published": {"metrics": {}, "scoring_failures": failures},
            "over_budget": {"error": "over budget", "scoring_failures": failures},
            "clean": {"metrics": {}},
        }
    }

    assert scoring_failure_warnings(results) == [
        "published: 1 instance(s) could not be scored and are left out of its metrics; "
        "2 instance(s) were scored from only some of their outputs. "
        "First scoring error: KeyError: 'gold'"
    ]


class _EvaluatorFailsTask(_RecordingTask):
    """A task whose compute_metrics finds it cannot score anything, as OmniDocBench does."""

    def compute_metrics(self, responses: list[Response]) -> dict[str, dict[str, float]]:
        for response in responses:
            for output in response.outputs:
                output.metadata["scoring_errors"] = {"code_exec": dict(ERROR)}
        return {}


def test_errors_recorded_while_computing_metrics_still_fail_the_task() -> None:
    responses = [_response(_output()) for _ in range(3)]

    result = compute_task_metrics(
        "test",
        _EvaluatorFailsTask(),  # type: ignore[arg-type]
        responses,
        {},
        3,
        0.0,
        max_hard_failure_rate=1.0,
    )

    assert result.failure_budget_exceeded is True
    assert result.error is not None and result.error.startswith("No instance was scored")
    assert result.instances_unscored == 3


def test_a_failure_on_a_response_without_outputs_reaches_the_predictions() -> None:
    from olmo_eval.upload.extract import build_instance_row

    failed = _response(scoring_error=dict(ERROR))
    result = compute_task_metrics(
        "test",
        _RecordingTask(),  # type: ignore[arg-type]
        [_response(_output()), failed],
        {},
        2,
        0.0,
        max_hard_failure_rate=1.0,
    )

    assert result.predictions is not None
    record = result.predictions[1]
    assert record["scoring_error"] == ERROR
    assert build_instance_row(record, "1", None, 0, 0, None)["has_scoring_error"] is True


class _TwoScorerTask(_RecordingTask):
    def _get_scorers(self) -> dict[str, object]:
        return {"code_exec": object(), "judge": object()}


def test_a_scorer_that_scored_nothing_fails_the_task_at_any_budget() -> None:
    judge_failed = {"judge": {"type": "JudgeParseError", "infrastructure": "true"}}
    responses = [
        _response(
            LMOutput(
                text="o", metadata={"score:code_exec": 1.0, "scoring_errors": dict(judge_failed)}
            )
        )
        for _ in range(3)
    ]

    result = compute_task_metrics(
        "test",
        _TwoScorerTask(),  # type: ignore[arg-type]
        responses,
        {},
        3,
        0.0,
        max_hard_failure_rate=1.0,
    )

    assert result.failure_budget_exceeded is True
    assert result.error is not None and result.error.startswith("No instance was scored")


class _ScoringTask(_RecordingTask):
    """A task the tracker path can score: one output fails, as a raising scorer would."""

    def strip_thinking_traces(self, responses: list[Response]) -> None:
        pass

    async def score_responses(self, responses: list[Response], context=None) -> list[Response]:
        return responses


@pytest.mark.anyio
async def test_finalize_task_applies_the_configured_budget() -> None:
    from olmo_eval.runners.asynq.preparation import finalize_task
    from olmo_eval.runners.asynq.types import TaskTracker

    def tracker() -> TaskTracker:
        responses = [_response(_output()), _response(_output(failed=True))]
        return TaskTracker(
            model_name="m",
            spec="test",
            task=_ScoringTask(),  # type: ignore[arg-type]
            total_instances=2,
            responses=dict(enumerate(responses)),
        )

    assert (await finalize_task(tracker())).failure_budget_exceeded is True
    assert (await finalize_task(tracker(), max_hard_failure_rate=0.5)).error is None


def test_the_summary_prints_warnings_as_plain_text(capsys) -> None:
    from olmo_eval.runners.processing.metrics import log_summary

    failures = {"instances_unscored": 1, "first_error": "KeyError: '[gold]'"}
    log_summary({"tasks": {"t": {"metrics": {}, "scoring_failures": failures}}})

    printed = capsys.readouterr().out
    assert "[yellow]" not in printed
    assert "KeyError: '[gold]'" in printed
