"""Tests that sandbox infrastructure failures invalidate task results."""

from __future__ import annotations

import pytest

from olmo_eval.common.types import Instance, LMOutput, LMRequest, RequestType, Response
from olmo_eval.evals.tasks.common import TaskConfig
from olmo_eval.runners.asynq.preparation import compute_task_metrics


class _Task:
    config = TaskConfig(name="test", data_source="test/data")

    def compute_metrics(self, _responses: list[Response]) -> dict[str, dict[str, float]]:
        return {"accuracy": {"code_exec": 0.0}}


def _response(error: dict[str, str]) -> Response:
    return Response(
        instance=Instance(question="question", gold_answer="answer"),
        request=LMRequest(request_type=RequestType.COMPLETION, prompt="question"),
        outputs=[
            LMOutput(
                text="output",
                metadata={"scoring_errors": {"code_exec": error}},
            )
        ],
    )


def test_infrastructure_scoring_error_marks_task_incomplete() -> None:
    result = compute_task_metrics(
        "test:variant",
        _Task(),  # type: ignore[arg-type]
        [_response({"type": "SandboxInfrastructureError", "infrastructure": "true"})],
        {},
        1,
        1.0,
    )

    assert result.error == "Incomplete: 1 output score(s) failed due to sandbox infrastructure"
    assert result.to_dict()["error"] == result.error


def test_model_scoring_error_does_not_mark_task_incomplete() -> None:
    result = compute_task_metrics(
        "test:variant",
        _Task(),  # type: ignore[arg-type]
        [_response({"type": "AssertionError", "message": "tests failed"})],
        {},
        1,
        1.0,
    )

    assert result.error is None


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
    assert result.error == "Incomplete: 8 output score(s) failed due to sandbox infrastructure"
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


def test_response_failure_counts_only_outputs_without_a_recorded_verdict() -> None:
    from olmo_eval.runners.asynq.results import _record_scoring_failure

    response = _response({"type": "ScoringIncompleteError", "infrastructure": "true"})
    response.outputs.append(LMOutput(text="ungraded"))
    response.outputs.append(LMOutput(text="graded", metadata={"score:code_exec": 1.0}))
    scored = _record_scoring_failure(
        response,
        scorer_names=["code_exec"],
        error={"type": "ScoringIncompleteError", "infrastructure": "true"},
    )
    result = compute_task_metrics("test", _Task(), [scored], {}, 1, 0.0)  # type: ignore[arg-type]
    assert result.error == "Incomplete: 2 output score(s) failed due to sandbox infrastructure"
