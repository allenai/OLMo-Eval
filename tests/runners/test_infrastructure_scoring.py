"""Tests that scoring failures count against the failure budget instead of as zeros."""

from __future__ import annotations

import pytest

from olmo_eval.common.scoring_failures import ScoringState, scoring_state
from olmo_eval.common.types import Instance, LMOutput, LMRequest, RequestType, Response
from olmo_eval.evals.tasks.common import TaskConfig
from olmo_eval.runners.asynq.preparation import compute_task_metrics


class _Task:
    config = TaskConfig(name="test", data_source="test/data")

    def _get_scorers(self) -> dict[str, object]:
        return {"code_exec": object()}

    def compute_metrics(self, _responses: list[Response]) -> dict[str, dict[str, float]]:
        return {"accuracy": {"code_exec": 0.0}}


def _response(error: dict[str, str] | None) -> Response:
    """A response with one output, carrying ``error`` as its code_exec scoring error if set."""
    return Response(
        instance=Instance(question="question", gold_answer="answer"),
        request=LMRequest(request_type=RequestType.COMPLETION, prompt="question"),
        outputs=[
            LMOutput(
                text="output",
                metadata={} if error is None else {"scoring_errors": {"code_exec": error}},
            )
        ],
    )


def _result(*responses: Response, total: int | None = None, failed: int = 0, budget=0.05):
    return compute_task_metrics(
        "test:variant",
        _Task(),  # type: ignore[arg-type]
        list(responses),
        {index: "no output" for index in range(failed)},
        total if total is not None else len(responses) + failed,
        1.0,
        max_hard_failure_rate=budget,
    )


def _scored() -> Response:
    response = _response(None)
    response.outputs[0].metadata["score:code_exec"] = 1.0
    return response


INFRA = {"type": "SandboxInfrastructureError", "infrastructure": "true"}
BUG = {"type": "KeyError", "message": "'answer'"}


def test_a_scoring_failure_within_the_budget_publishes_the_metrics() -> None:
    result = _result(_response(INFRA), *[_scored() for _ in range(19)])

    assert result.error is None
    assert result.failure_budget_exceeded is False
    assert result.metrics == {"accuracy": {"code_exec": 0.0}}
    assert result.instances_unscored == 1
    assert result.failure_rate == pytest.approx(0.05)
    assert result.scoring_failures == {
        "instances_unscored": 1,
        "instances_partially_scored": 0,
        "outputs_unscored": 1,
        "infrastructure": 1,
        "by_type": {"SandboxInfrastructureError": 1},
        "first_error": "SandboxInfrastructureError",
    }
    assert result.to_dict()["scoring_failures"] == result.scoring_failures


def test_a_scoring_failure_of_any_kind_counts_against_the_budget() -> None:
    result = _result(_response(BUG), *[_scored() for _ in range(9)])

    assert result.failure_budget_exceeded is True
    assert result.error == (
        "1 of 10 instances (10.0%) failed, above the failure budget of 5.0%: 0 produced no "
        "output and 1 could not be scored. First failure: KeyError: 'answer'"
    )


def test_instances_with_no_output_and_unscored_instances_share_one_budget() -> None:
    # 2 with no output + 2 unscored = 4 of 40: 10%, over a budget that each alone fits.
    responses = [_response(BUG), _response(BUG), *[_scored() for _ in range(36)]]

    assert _result(*responses, failed=2, budget=0.06).failure_budget_exceeded is True
    assert _result(*responses, failed=2, budget=0.10).failure_budget_exceeded is False


def test_a_task_with_nothing_scored_fails_at_any_budget() -> None:
    result = _result(_response(INFRA), _response(BUG), budget=1.0)

    assert result.failure_budget_exceeded is True
    assert result.error.startswith("No instance was scored out of 2")


def test_a_partially_scored_instance_is_not_a_failure() -> None:
    partial = _scored()
    partial.outputs.append(LMOutput(text="b", metadata={"scoring_errors": {"code_exec": BUG}}))

    result = _result(partial)

    assert result.error is None
    assert result.scoring_failures["instances_partially_scored"] == 1
    assert result.scoring_failures["outputs_unscored"] == 1
    assert result.instances_unscored == 0


def test_a_clean_task_reports_no_scoring_failures() -> None:
    result = _result(_scored())

    assert result.error is None
    assert result.scoring_failures is None
    assert "scoring_failures" not in result.to_dict()


@pytest.mark.parametrize("environment", ["missing", "empty", "other_pool", "depleted"])
def test_missing_sandbox_invalidates_livecodebench_samples(
    environment: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    import asyncio
    import queue

    from olmo_eval.common.execution import ScoringContext
    from olmo_eval.evals.tasks.common import get_task
    from olmo_eval.harness.sandbox import SandboxManager
    from olmo_eval.runners.asynq.results import process_results
    from olmo_eval.runners.asynq.types import ResultItem, TaskTracker
    from tests.core.harness.sandbox_stubs import TrackingExecutor, make_manager

    manager = None if environment == "missing" else SandboxManager([])
    if environment == "other_pool":
        manager = make_manager(TrackingExecutor("healthy", capabilities=frozenset({"other"})))
        assert manager.is_running
    if environment == "depleted":
        executor = TrackingExecutor("broken", max_concurrency=1, fail_and_quarantine=True)
        manager = make_manager(executor)
        monkeypatch.setattr(
            "olmo_eval.evals.tasks.livecodebench._stage_problem", lambda *_args: "{}"
        )
        assert manager.is_running

    task = get_task("livecodebench:lite")
    instance = Instance(question="Print one", gold_answer="", metadata={"id": "sandbox-missing"})
    request = task.format_request(instance)
    tracker = TaskTracker(model_name="m", spec="livecodebench:lite", task=task, total_instances=1)
    result_queue: queue.Queue = queue.Queue()
    result_queue.put(
        ResultItem(
            model_name="m",
            task_id="livecodebench:lite",
            instance_idx=0,
            instance=instance,
            request=request,
            outputs=[LMOutput(text="```python\nprint(1)\n```") for _ in range(8)],
        )
    )
    results = asyncio.run(
        process_results(
            trackers={"livecodebench:lite": tracker},
            result_queue=result_queue,  # type: ignore[arg-type]
            workers=[],
            scoring_context=ScoringContext(execution_env=manager),
            scoring_concurrency=1,
            total_tasks=1,
            total_instances=1,
            model_name="m",
            save_predictions=False,
            write_predictions_fn=None,
            save_requests=False,
            write_requests_fn=None,
        )
    )
    result = results["livecodebench:lite"]
    assert result.error.startswith("No instance was scored out of 1")
    assert result.scoring_failures["outputs_unscored"] == 8
    assert result.scoring_failures["infrastructure"] == 8
    if environment == "depleted":
        assert executor.calls == 1
    for output in result.predictions[0]["model_output"]:
        errors = output["scoring_errors"]
        assert len(errors) == 1
        error = next(iter(errors.values()))
        assert error["type"] == (
            "SandboxRequiredError" if environment == "missing" else "SandboxInfrastructureError"
        )
        assert error["infrastructure"] == "true"


def test_a_response_failure_leaves_every_output_unscored() -> None:
    from olmo_eval.runners.asynq.results import _record_scoring_failure

    response = _response({"type": "ScoringIncompleteError", "infrastructure": "true"})
    response.outputs.append(LMOutput(text="ungraded"))
    response.outputs.append(LMOutput(text="graded", metadata={"score:code_exec": 1.0}))
    response.scores["code_exec"] = 1.0

    _record_scoring_failure(response, error={"type": "KeyError", "message": "'answer'"})

    # The raise interrupted scoring, so even the output that had a score is not trusted,
    # and no score of it reaches the predictions.
    for output in response.outputs:
        assert output.metadata["scoring_errors"]["__response__"]["type"] == "KeyError"
        assert not any(key.startswith("score:") for key in output.metadata)
    assert response.scores == {}
    assert scoring_state(response) is ScoringState.UNSCORED


def test_a_failed_response_without_outputs_keeps_the_error() -> None:
    from olmo_eval.runners.asynq.results import _record_scoring_failure

    response = _response(None)
    response.outputs = []

    _record_scoring_failure(response, error={"type": "KeyError"})

    assert response.scoring_error == {"type": "KeyError"}
