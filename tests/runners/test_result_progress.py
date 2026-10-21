"""Tests for Beaker progress reporting from the parent result consumer."""

from __future__ import annotations

import asyncio
import logging
import queue
import subprocess
import sys
from collections.abc import Iterator, Sequence

import pytest

from olmo_eval.common.execution.environment import ScoringContext
from olmo_eval.common.types import Instance, LMOutput, LMRequest, RequestType, Response
from olmo_eval.evals.tasks.common import Task, TaskConfig
from olmo_eval.runners.asynq.batching import BatchConfig, BatchedStrategy, StreamingStrategy
from olmo_eval.runners.asynq.results import process_results
from olmo_eval.runners.asynq.types import WORKER_FATAL, QueueItem, ResultItem, TaskTracker


class _EchoTask(Task):
    @property
    def instances(self) -> Iterator[Instance]:
        yield Instance(question="Q", gold_answer="A")

    def format_request(self, instance: Instance) -> LMRequest:
        return LMRequest(request_type=RequestType.COMPLETION, prompt=instance.question)

    async def score_responses(
        self, responses: Sequence[Response], context: ScoringContext | None = None
    ) -> Sequence[Response]:
        for response in responses:
            response.scores["echo"] = 1.0
        return responses


class _Reporter:
    instances: list[_Reporter]

    def __init__(self) -> None:
        self.calls: list[tuple] = []
        _Reporter.instances.append(self)

    def flush(self) -> None:
        self.calls.append(("flush",))

    def progress_callback(self, label: str):
        assert label == "Processed"

        def _cb(count: int, total: int, *, force: bool = False) -> None:
            self.calls.append((count, total, force))

        return _cb


@pytest.fixture
def reporters(monkeypatch: pytest.MonkeyPatch) -> list[_Reporter]:
    _Reporter.instances = []
    monkeypatch.setattr("olmo_eval.runners.asynq.results.BeakerStatusReporter", _Reporter)
    monkeypatch.setattr("olmo_eval.common.beaker_status.BeakerStatusReporter", _Reporter)
    return _Reporter.instances


def _result(task: Task, idx: int) -> ResultItem:
    instance = Instance(question=f"Q{idx}", gold_answer="A")
    return ResultItem(
        model_name="m",
        task_id="echo",
        instance_idx=idx,
        instance=instance,
        request=task.format_request(instance),
        outputs=[LMOutput(text="A")],
    )


def _process(result_queue: queue.Queue, total: int) -> None:
    task = _EchoTask(TaskConfig(name="echo", data_source="test/dataset"))
    asyncio.run(
        process_results(
            trackers={
                "echo": TaskTracker(model_name="m", spec="echo", task=task, total_instances=total)
            },
            result_queue=result_queue,  # type: ignore[arg-type]
            workers=[],
            scoring_context=ScoringContext(),
            scoring_concurrency=4,
            total_tasks=1,
            total_instances=total,
            model_name="m",
            save_predictions=False,
            write_predictions_fn=None,
            save_requests=False,
            write_requests_fn=None,
        )
    )


def test_reports_one_global_count_for_uneven_worker_results(reporters: list[_Reporter]) -> None:
    """Two workers that took 64 and 36 items still produce one count out of 100."""
    task = _EchoTask(TaskConfig(name="echo", data_source="test/dataset"))
    worker_a = list(range(0, 64))
    worker_b = list(range(64, 100))
    order = [idx for pair in zip(worker_a, worker_b, strict=False) for idx in pair]
    order += worker_a[len(worker_b) :]
    result_queue: queue.Queue[ResultItem] = queue.Queue()
    for idx in order:
        result_queue.put(_result(task, idx))

    _process(result_queue, total=100)

    assert len(reporters) == 1
    assert reporters[0].calls == [(n, 100, False) for n in range(1, 101)] + [
        (100, 100, True),
        ("flush",),
    ]


def test_forces_final_report_when_a_worker_dies(reporters: list[_Reporter]) -> None:
    task = _EchoTask(TaskConfig(name="echo", data_source="test/dataset"))
    result_queue: queue.Queue[ResultItem] = queue.Queue()
    for idx in range(3):
        result_queue.put(_result(task, idx))
    result_queue.put(
        ResultItem(
            model_name="m",
            task_id=WORKER_FATAL,
            instance_idx=-1,
            instance=None,
            request=None,
            outputs=[],
            error="engine died",
        )
    )

    with pytest.raises(RuntimeError, match="engine died"):
        _process(result_queue, total=10)

    assert reporters[0].calls == [
        (1, 10, False),
        (2, 10, False),
        (3, 10, False),
        (3, 10, True),
        ("flush",),
    ]


@pytest.mark.parametrize(
    "strategy",
    [
        BatchedStrategy(BatchConfig(chunk_size=2, chunk_timeout=0.05)),
        StreamingStrategy(BatchConfig.streaming()),
    ],
)
def test_workers_leave_progress_to_the_parent(
    monkeypatch: pytest.MonkeyPatch, reporters: list[_Reporter], strategy
) -> None:
    """Per-worker counts would overwrite the parent's global count."""

    async def noop(*_args: object, **_kwargs: object) -> None:
        return None

    monkeypatch.setattr("olmo_eval.runners.asynq.processing.process_items", noop)
    item_queue: queue.Queue[QueueItem | None] = queue.Queue()
    for idx in range(5):
        item_queue.put(
            QueueItem(
                model_name="m",
                task_id="echo",
                instance_idx=idx,
                instance=Instance(question="Q", gold_answer="A"),
                request=LMRequest(request_type=RequestType.COMPLETION, prompt="Q"),
            )
        )
    item_queue.put(None)

    asyncio.run(
        strategy.run(
            item_queue=item_queue,  # type: ignore[arg-type]
            harness=object(),  # type: ignore[arg-type]
            result_queue=queue.Queue(),  # type: ignore[arg-type]
            max_concurrency=2,
            worker_logger=logging.getLogger(__name__),
            total_instances=10,
            num_workers=2,
        )
    )

    assert reporters == []


def test_reporter_is_a_noop_without_the_beaker_extra() -> None:
    """Local runs without beaker-py installed must still import and run the reporter."""
    code = """
import os, sys
sys.modules["beaker"] = None
sys.modules["beaker.exceptions"] = None
os.environ["BEAKER_WORKLOAD_ID"] = "wl_123"
from olmo_eval.common.beaker_status import BeakerStatusReporter
import olmo_eval.runners.asynq.results
reporter = BeakerStatusReporter()
assert reporter._client is None
reporter.update("hello", force=True)
reporter.progress_callback("Processed")(1, 2, force=True)
print("ok")
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
