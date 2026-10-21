"""Regression tests for fatal async inference failures."""

from __future__ import annotations

import asyncio
import logging
import queue
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from olmo_eval.common.types import Instance, LMOutput, LMRequest, RequestType
from olmo_eval.inference.errors import REQUEST_ERROR_KEY, TerminalProviderError
from olmo_eval.runners.asynq.batching import BatchConfig, StreamingStrategy
from olmo_eval.runners.asynq.monitoring import check_workers_alive
from olmo_eval.runners.asynq.processing import process_batch, process_chat_request
from olmo_eval.runners.asynq.results import (
    _build_pending_instance_keys,
    _consume_pending_instance,
    _format_worker_failure,
)
from olmo_eval.runners.asynq.types import QueueItem, ResultItem, TaskTracker


def _engine_dead_error() -> Exception:
    error_type = type(
        "EngineDeadError",
        (Exception,),
        {"__module__": "vllm.v1.engine.exceptions"},
    )
    return error_type("EngineCore encountered an issue")


def _queue_item(instance_idx: int) -> QueueItem:
    return QueueItem(
        model_name="model",
        task_id="task",
        instance_idx=instance_idx,
        instance=Instance(question=f"question {instance_idx}", gold_answer="answer"),
        request=LMRequest(
            request_type=RequestType.COMPLETION,
            prompt=f"question {instance_idx}",
        ),
    )


def _harness(error: Exception) -> SimpleNamespace:
    provider = SimpleNamespace(
        describe_request=Mock(return_value=None),
        agenerate=AsyncMock(side_effect=error),
    )
    return SimpleNamespace(
        provider=provider,
        _apply_config=lambda request: request,
        flush_metrics=Mock(),
        run=AsyncMock(side_effect=error),
    )


def test_terminal_processing_paths_propagate_without_instance_failures() -> None:
    for chat in (False, True):
        result_queue: queue.Queue[ResultItem] = queue.Queue()
        harness = _harness(_engine_dead_error())
        if chat:
            request = process_chat_request(
                _queue_item(0),
                harness,
                result_queue,  # type: ignore[arg-type]
            )
        else:
            request = process_batch(
                [_queue_item(0)],
                harness,
                result_queue,  # type: ignore[arg-type]
            )

        with pytest.raises(TerminalProviderError, match="EngineDeadError"):
            asyncio.run(request)
        assert result_queue.empty()


def test_marked_request_fails_alone_in_its_batch() -> None:
    marked = LMOutput(text="", metadata={REQUEST_ERROR_KEY: "prompt too long"})
    harness = SimpleNamespace(
        provider=SimpleNamespace(
            describe_request=Mock(return_value=None),
            agenerate=AsyncMock(return_value=[[LMOutput(text="ok")], [marked]]),
        ),
        _apply_config=lambda request: request,
        flush_metrics=Mock(),
    )
    result_queue: queue.Queue[ResultItem] = queue.Queue()

    asyncio.run(process_batch([_queue_item(0), _queue_item(1)], harness, result_queue))  # type: ignore[arg-type]

    results = {r.instance_idx: r for r in (result_queue.get_nowait() for _ in range(2))}
    assert results[0].error is None and [o.text for o in results[0].outputs] == ["ok"]
    assert results[1].error == "prompt too long" and results[1].outputs == []


def test_marked_chat_request_fails_its_instance() -> None:
    marked = LMOutput(text="", metadata={REQUEST_ERROR_KEY: "prompt too long"})
    harness = SimpleNamespace(
        provider=SimpleNamespace(describe_request=Mock(return_value=None)),
        _apply_config=lambda request: request,
        run=AsyncMock(
            return_value=SimpleNamespace(final_output=marked, trajectory=None, error=None)
        ),
    )
    result_queue: queue.Queue[ResultItem] = queue.Queue()

    asyncio.run(process_chat_request(_queue_item(0), harness, result_queue))  # type: ignore[arg-type]

    result = result_queue.get_nowait()
    assert result.error == "prompt too long" and result.outputs == []


def test_streaming_strategy_propagates_fast_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fail(*_args: object, **_kwargs: object) -> None:
        raise TerminalProviderError("engine died")

    async def run() -> None:
        monkeypatch.setattr("olmo_eval.runners.asynq.processing.process_items", fail)
        item_queue: queue.Queue[QueueItem | None] = queue.Queue()
        result_queue: queue.Queue[ResultItem] = queue.Queue()
        item_queue.put(_queue_item(0))

        with pytest.raises(TerminalProviderError, match="engine died"):
            await StreamingStrategy(BatchConfig.streaming()).run(
                item_queue=item_queue,  # type: ignore[arg-type]
                harness=object(),  # type: ignore[arg-type]
                result_queue=result_queue,  # type: ignore[arg-type]
                max_concurrency=1,
                worker_logger=logging.getLogger(__name__),
                total_instances=1,
            )

    asyncio.run(run())


def test_pending_identities_reject_duplicates_and_appear_in_fatal_error() -> None:
    trackers = {
        "task": TaskTracker(
            model_name="model",
            spec="task",
            task=None,
            total_instances=2,
        )
    }
    pending = _build_pending_instance_keys(trackers)
    message = _format_worker_failure("terminal failure", pending)
    assert "2 inference result(s) pending" in message
    assert "model:task[0]" in message
    assert "model:task[1]" in message

    result = ResultItem(
        model_name="model",
        task_id="task",
        instance_idx=0,
        instance=None,
        request=None,
        outputs=[],
    )

    _consume_pending_instance(pending, result)

    assert pending == {("model", "task", 1)}
    with pytest.raises(RuntimeError, match="duplicate or unexpected"):
        _consume_pending_instance(pending, result)


def test_single_failed_worker_is_fatal_while_other_workers_are_alive() -> None:
    workers = [SimpleNamespace(exitcode=1), SimpleNamespace(exitcode=None)]

    with pytest.raises(RuntimeError, match="worker 0 exited with code 1"):
        check_workers_alive(
            workers,  # type: ignore[arg-type]
            queue.Queue(),  # type: ignore[arg-type]
        )
