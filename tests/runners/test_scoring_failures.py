"""Tests for how the runner treats instances whose scores could not be computed."""

from __future__ import annotations

from olmo_eval.common.types import Instance, LMOutput, LMRequest, RequestType, Response
from olmo_eval.evals.tasks.common import TaskConfig
from olmo_eval.runners.asynq.preparation import compute_task_metrics
from olmo_eval.runners.processing.scoring_failures import (
    ScoringState,
    measurable_responses,
    scoring_state,
)

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
