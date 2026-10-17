"""Per-task cost in the async runner: request timing, token totals and time attribution."""

from __future__ import annotations

import asyncio
import queue
from collections.abc import Callable, Iterator, Sequence
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from olmo_eval.common.execution.environment import ScoringContext
from olmo_eval.common.types import Instance, LMOutput, LMRequest, RequestType, Response
from olmo_eval.evals.tasks.common import Task, TaskConfig
from olmo_eval.inference.metrics.core.schema import BatchMetrics, RequestMetrics
from olmo_eval.runners.asynq.batching import BatchConfig, BatchedStrategy, StreamingStrategy
from olmo_eval.runners.asynq.processing import (
    attribute_batch_seconds,
    process_batch,
    process_chat_request,
    process_items,
)
from olmo_eval.runners.asynq.results import aggregate_results, process_results, utc_iso
from olmo_eval.runners.asynq.runner import _processing_seconds
from olmo_eval.runners.asynq.share_clock import ShareClock
from olmo_eval.runners.asynq.types import QueueItem, ResultItem, TaskTracker
from olmo_eval.runners.common.types import RUN_TIMING_KEYS, TASK_COST_KEYS, TaskResult
from olmo_eval.runners.common.usage import (
    instance_prompt_tokens,
    loglikelihood_prompt_tokens,
    request_token_usage,
)
from olmo_eval.runners.io.builders import build_predictions
from olmo_eval.runners.processing.metrics import build_single_model_metrics

# ---------------------------------------------------------------------------
# Token counting rules
# ---------------------------------------------------------------------------


def _ll(num_tokens: int, num_tokens_all: int) -> dict:
    return {"sum_logits": -1.0, "num_tokens": num_tokens, "num_tokens_all": num_tokens_all}


def test_loglikelihood_request_sums_context_plus_continuation_per_choice() -> None:
    outputs = [_ll(2, 30), _ll(3, 31), _ll(1, 29)]
    assert request_token_usage(outputs, loglikelihood=True) == (90, 0)


def test_loglikelihood_without_context_count_is_unknown() -> None:
    # vllm_server and huggingface report num_tokens_all == num_tokens (continuation only)
    assert loglikelihood_prompt_tokens(_ll(3, 3)) is None
    assert request_token_usage([_ll(2, 30), _ll(3, 3)], loglikelihood=True) == (None, 0)


def test_generation_uses_provider_counts() -> None:
    samples = [
        {"prompt_tokens": 50, "completion_tokens": 10},
        {"prompt_tokens": 50, "completion_tokens": 12},
    ]
    assert request_token_usage(samples, loglikelihood=False) == (50, 22)
    assert request_token_usage([{"completion_tokens": 7}], loglikelihood=False) == (None, 7)
    assert request_token_usage([{"prompt_tokens": 5}], loglikelihood=False) == (5, None)


def test_multi_turn_agent_counts_are_unknown() -> None:
    output = {"prompt_tokens": 500, "completion_tokens": 30, "num_turns": 3}
    assert request_token_usage([output], loglikelihood=False) == (None, None)


def test_request_without_outputs_adds_nothing() -> None:
    assert request_token_usage([], loglikelihood=False) == (0, 0)


def test_instance_prompt_tokens_from_predictions_outputs() -> None:
    assert instance_prompt_tokens([_ll(2, 30), _ll(3, 31)]) == 30
    assert instance_prompt_tokens([{"prompt_tokens": 64, "completion_tokens": 9}]) == 64
    # vllm offline generation: num_tokens_all counts generated tokens, not the prompt
    generated = {"finish_reason": "stop", "completion_tokens": 9, **_ll(9, 9)}
    assert instance_prompt_tokens([generated]) is None
    assert instance_prompt_tokens([]) is None


def test_predictions_keep_provider_prompt_tokens() -> None:
    response = Response(
        instance=Instance(question="q", gold_answer="a"),
        request=LMRequest(request_type=RequestType.COMPLETION, prompt="q"),
        outputs=[
            LMOutput(
                text="a",
                metadata={"prompt_tokens": 12, "completion_tokens": 3, "finish_reason": "stop"},
            )
        ],
    )
    (prediction,) = build_predictions([response])
    assert prediction["model_output"][0]["prompt_tokens"] == 12


# ---------------------------------------------------------------------------
# Tracker accumulation
# ---------------------------------------------------------------------------


def _result(
    idx: int,
    outputs: list[LMOutput],
    *,
    request_type: RequestType = RequestType.LOGLIKELIHOOD,
    sent_at: float | None = None,
    attributed: float | None = None,
    error: str | None = None,
    task_id: str = "t",
) -> ResultItem:
    instance = Instance(question=f"q{idx}", gold_answer="a")
    return ResultItem(
        model_name="m",
        task_id=task_id,
        instance_idx=idx,
        instance=instance,
        request=LMRequest(request_type=request_type, prompt=f"q{idx}", continuations=("a",)),
        outputs=outputs,
        error=error,
        sent_at=sent_at,
        attributed_seconds=attributed,
    )


def _tracker(total: int = 3) -> TaskTracker:
    return TaskTracker(model_name="m", spec="t", task=None, total_instances=total)


def test_tracker_sums_tokens_and_keeps_earliest_send_time() -> None:
    tracker = _tracker()
    tracker.record_usage(_result(0, [LMOutput("a", metadata=_ll(2, 20))], sent_at=105.0))
    tracker.record_usage(_result(1, [LMOutput("a", metadata=_ll(2, 25))], sent_at=101.0))
    # A hard failure (no outputs) still counts as a result but adds no tokens
    tracker.record_usage(_result(2, [], sent_at=103.0, error="boom"))

    assert tracker.first_request_at == 101.0
    assert tracker.prompt_tokens_total == 45
    assert tracker.completion_tokens_total == 0
    assert tracker.attributed_inference_seconds is None  # no batch metrics


def test_tracker_total_becomes_null_when_any_count_is_unknown() -> None:
    tracker = _tracker()
    gen = RequestType.COMPLETION
    tracker.record_usage(
        _result(
            0,
            [LMOutput("x", metadata={"prompt_tokens": 10, "completion_tokens": 4})],
            request_type=gen,
            attributed=0.5,
        )
    )
    tracker.record_usage(
        _result(
            1, [LMOutput("x", metadata={"completion_tokens": 6})], request_type=gen, attributed=0.25
        )
    )
    assert tracker.prompt_tokens_total is None
    assert tracker.completion_tokens_total == 10
    assert tracker.attributed_inference_seconds == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# Worker time attribution
# ---------------------------------------------------------------------------


class _FakeTime:
    """A settable stand-in for time.monotonic."""

    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def test_share_clock_gives_a_lone_unit_its_wall_time() -> None:
    now = _FakeTime()
    clock = ShareClock(now)
    token = clock.start()
    now.t = 2.5
    assert clock.stop(token) == pytest.approx(2.5)


def test_share_clock_splits_overlap_and_skips_idle_time() -> None:
    now = _FakeTime()
    clock = ShareClock(now)
    a = clock.start()
    now.t = 2.0
    b = clock.start()  # a alone for 2s
    now.t = 6.0
    share_a = clock.stop(a)  # 4s shared by two
    now.t = 7.0
    share_b = clock.stop(b)  # b alone for 1s
    now.t = 20.0  # idle: credited to no one
    c = clock.start()
    now.t = 21.0
    share_c = clock.stop(c)

    assert share_a == pytest.approx(4.0)
    assert share_b == pytest.approx(3.0)
    assert share_c == pytest.approx(1.0)
    assert share_a + share_b + share_c == pytest.approx(8.0)  # busy time, not 21s


def _request_metrics(prompt: int, completion: int) -> RequestMetrics:
    return RequestMetrics(
        request_id="r",
        prompt_tokens=prompt,
        completion_tokens=completion,
        end_to_end_latency_s=1.0,
        tokens_per_second=0.0,
    )


def _batch(wall: float, requests: Sequence[RequestMetrics]) -> BatchMetrics:
    return BatchMetrics(
        total_requests=len(requests),
        successful_requests=len(requests),
        failed_requests=0,
        total_prompt_tokens=sum(r.prompt_tokens for r in requests),
        total_completion_tokens=sum(r.completion_tokens for r in requests),
        wall_clock_time_s=wall,
        output_tokens_per_second=0.0,
        mean_latency_s=0.0,
        requests=tuple(requests),
    )


def test_attribution_splits_batch_time_by_token_share() -> None:
    # The batch's own wall time in its metrics is ignored; the worker clock's share is split.
    batch = _batch(99.0, [_request_metrics(30, 10), _request_metrics(10, 10)])
    assert attribute_batch_seconds(4.0, batch, 2) == [pytest.approx(8 / 3), pytest.approx(4 / 3)]


def test_attribution_splits_evenly_without_matching_request_metrics() -> None:
    batch = _batch(4.0, [_request_metrics(30, 10)])
    assert attribute_batch_seconds(4.0, batch, 2) == [2.0, 2.0]
    assert attribute_batch_seconds(4.0, None, 2) == [2.0, 2.0]
    assert attribute_batch_seconds(4.0, Mock(), 1) == [4.0]
    empty = _batch(3.0, [_request_metrics(0, 0), _request_metrics(0, 0)])
    assert attribute_batch_seconds(3.0, empty, 2) == [1.5, 1.5]


def _queue_item(
    idx: int, task_id: str, request_type: RequestType = RequestType.COMPLETION
) -> QueueItem:
    return QueueItem(
        model_name="m",
        task_id=task_id,
        instance_idx=idx,
        instance=Instance(question=f"q{idx}", gold_answer="a"),
        request=LMRequest(request_type=request_type, prompt=f"q{idx}"),
    )


def _timed(now: _FakeTime, seconds: float, value: object) -> AsyncMock:
    """An async callable that advances the fake clock by ``seconds`` and returns ``value``."""

    async def call(*_args: object, **_kwargs: object) -> object:
        now.t += seconds
        return value

    return AsyncMock(side_effect=call)


def test_process_batch_stamps_send_time_and_attribution(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("olmo_eval.runners.asynq.processing.time.time", lambda: 1000.0)
    now = _FakeTime()
    batch = _batch(99.0, [_request_metrics(10, 10), _request_metrics(30, 10)])
    harness = SimpleNamespace(
        provider=SimpleNamespace(
            describe_request=Mock(return_value=None),
            agenerate=_timed(now, 6.0, [[LMOutput("a")], [LMOutput("b")]]),
        ),
        _apply_config=lambda request: request,
        flush_metrics=Mock(return_value=batch),
    )
    result_queue: queue.Queue[ResultItem] = queue.Queue()

    items = [_queue_item(0, "easy"), _queue_item(1, "hard")]
    asyncio.run(process_batch(items, harness, result_queue, clock=ShareClock(now)))  # type: ignore[arg-type]

    results = [result_queue.get_nowait() for _ in range(2)]
    assert [r.sent_at for r in results] == [1000.0, 1000.0]
    assert [r.attributed_seconds for r in results] == [2.0, 4.0]


def test_failed_batch_still_attributes_its_time() -> None:
    now = _FakeTime()

    async def fail(*_args: object, **_kwargs: object) -> None:
        now.t += 3.0
        raise RuntimeError("server hiccup")

    harness = SimpleNamespace(
        provider=SimpleNamespace(
            describe_request=Mock(return_value=None), agenerate=AsyncMock(side_effect=fail)
        ),
        _apply_config=lambda request: request,
        flush_metrics=Mock(return_value=None),
    )
    result_queue: queue.Queue[ResultItem] = queue.Queue()

    items = [_queue_item(0, "t"), _queue_item(1, "t"), _queue_item(2, "t")]
    asyncio.run(process_batch(items, harness, result_queue, clock=ShareClock(now)))  # type: ignore[arg-type]

    results = [result_queue.get_nowait() for _ in range(3)]
    assert all(r.error for r in results)
    assert [r.attributed_seconds for r in results] == [1.0, 1.0, 1.0]


async def _until(condition: Callable[[], bool]) -> None:
    """Yield to the event loop until ``condition`` holds (fails instead of hanging)."""
    for _ in range(1000):
        if condition():
            return
        await asyncio.sleep(0)
    raise AssertionError("condition never held")


def _rollout(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        final_output=LMOutput(text),
        trajectory=None,
        error=None,
        max_turns_reached=False,
        total_tool_calls=0,
        num_turns=1,
    )


def test_chat_rollout_carries_its_time_on_success_and_failure() -> None:
    now = _FakeTime()
    clock = ShareClock(now)
    result_queue: queue.Queue[ResultItem] = queue.Queue()

    async def broken(*_args: object, **_kwargs: object) -> None:
        now.t += 2.0
        raise RuntimeError("sandbox lost")

    for run in (_timed(now, 5.0, _rollout("ok")), AsyncMock(side_effect=broken)):
        harness = SimpleNamespace(
            provider=SimpleNamespace(describe_request=Mock(return_value=None)),
            _apply_config=lambda request: request,
            run=run,
        )
        item = _queue_item(0, "t", RequestType.CHAT)
        asyncio.run(process_chat_request(item, harness, result_queue, clock=clock))  # type: ignore[arg-type]

    ok, failed = result_queue.get_nowait(), result_queue.get_nowait()
    assert ok.error is None and ok.attributed_seconds == pytest.approx(5.0)
    assert failed.error and failed.attributed_seconds == pytest.approx(2.0)


def test_concurrent_rollouts_split_worker_time() -> None:
    """Overlapping rollouts share the worker's time instead of each claiming its latency."""
    now = _FakeTime()
    finish = [asyncio.Event() for _ in range(3)]
    started: list[int] = []

    async def run(request: LMRequest, *_args: object, **_kwargs: object) -> SimpleNamespace:
        idx = int(request.prompt[1:])
        started.append(idx)
        await finish[idx].wait()
        return _rollout("done")

    harness = SimpleNamespace(
        config=SimpleNamespace(scaffold="multi_turn_chat"),
        provider=SimpleNamespace(describe_request=Mock(return_value=None)),
        _apply_config=lambda request: request,
        run=run,
        flush_metrics=Mock(return_value=None),
    )
    result_queue: queue.Queue[ResultItem] = queue.Queue()
    items = [_queue_item(i, f"task{i}", RequestType.CHAT) for i in range(3)]

    async def drive() -> None:
        processing = asyncio.create_task(
            process_items(items, harness, result_queue, clock=ShareClock(now))  # type: ignore[arg-type]
        )
        await _until(lambda: len(started) == 3)  # all three rollouts start at t=0
        for idx, t in enumerate((3.0, 5.0, 9.0)):
            now.t = t
            finish[idx].set()
            await _until(lambda idx=idx: result_queue.qsize() == idx + 1)
        await processing

    asyncio.run(drive())

    results = sorted((result_queue.get_nowait() for _ in range(3)), key=lambda r: r.instance_idx)
    # 0-3s: three in flight (1s each); 3-5s: two (1s each); 5-9s: one (4s).
    assert [r.attributed_seconds for r in results] == [
        pytest.approx(1.0),
        pytest.approx(2.0),
        pytest.approx(6.0),
    ]
    assert sum(r.attributed_seconds or 0.0 for r in results) == pytest.approx(9.0)


def test_overlapping_batches_split_worker_time() -> None:
    """Streaming runs single-item batches concurrently; their shares sum to busy time."""
    now = _FakeTime()
    clock = ShareClock(now)
    release = [asyncio.Event() for _ in range(2)]

    async def agenerate(requests: list[LMRequest], *_args: object) -> list[list[LMOutput]]:
        await release[int(requests[0].prompt[1:])].wait()
        return [[LMOutput("x")]]

    harness = SimpleNamespace(
        provider=SimpleNamespace(describe_request=Mock(return_value=None), agenerate=agenerate),
        _apply_config=lambda request: request,
        flush_metrics=Mock(return_value=None),
    )
    result_queue: queue.Queue[ResultItem] = queue.Queue()

    async def drive() -> None:
        batches = [
            asyncio.create_task(
                process_batch([_queue_item(i, "t")], harness, result_queue, clock=clock)  # type: ignore[arg-type]
            )
            for i in range(2)
        ]
        await _until(lambda: clock._active == 2)
        now.t = 4.0
        release[0].set()
        await _until(lambda: result_queue.qsize() == 1)
        now.t = 6.0
        release[1].set()
        await asyncio.gather(*batches)

    asyncio.run(drive())

    shares = sorted(result_queue.get_nowait().attributed_seconds or 0.0 for _ in range(2))
    # Each call took 4s or 6s on its own, but the worker was busy for 6s in total.
    assert shares == [pytest.approx(2.0), pytest.approx(4.0)]


def test_cancelled_batch_leaves_the_clock() -> None:
    now = _FakeTime()
    clock = ShareClock(now)

    async def hang(*_args: object) -> None:
        await asyncio.Event().wait()

    harness = SimpleNamespace(
        provider=SimpleNamespace(describe_request=Mock(return_value=None), agenerate=hang),
        _apply_config=lambda request: request,
    )

    async def drive() -> None:
        batch = asyncio.create_task(
            process_batch([_queue_item(0, "t")], harness, queue.Queue(), clock=clock)  # type: ignore[arg-type]
        )
        await _until(lambda: clock._active == 1)
        batch.cancel()
        with pytest.raises(asyncio.CancelledError):
            await batch

    asyncio.run(drive())
    # A later unit alone gets its full wall time, so the cancelled one no longer counts.
    token = clock.start()
    now.t += 2.0
    assert clock.stop(token) == pytest.approx(2.0)


@pytest.mark.parametrize(
    "strategy",
    [
        BatchedStrategy(BatchConfig(chunk_size=2, chunk_timeout=0.05)),
        StreamingStrategy(BatchConfig.streaming()),
    ],
)
def test_strategies_share_one_clock_per_worker(
    strategy: BatchedStrategy | StreamingStrategy, monkeypatch: pytest.MonkeyPatch
) -> None:
    clocks: list[ShareClock | None] = []

    async def capture(*_args: object, clock: ShareClock | None = None, **_kwargs: object) -> None:
        clocks.append(clock)

    monkeypatch.setattr("olmo_eval.runners.asynq.processing.process_items", capture)
    item_queue: queue.Queue[QueueItem | None] = queue.Queue()
    for i in range(3):
        item_queue.put(_queue_item(i, "t"))
    item_queue.put(None)

    asyncio.run(
        strategy.run(
            item_queue=item_queue,  # type: ignore[arg-type]
            harness=object(),  # type: ignore[arg-type]
            result_queue=queue.Queue(),  # type: ignore[arg-type]
            max_concurrency=None,
            worker_logger=Mock(),
            total_instances=3,
        )
    )
    assert clocks and all(clock is strategy.clock for clock in clocks)


# ---------------------------------------------------------------------------
# Task timing through process_results, metrics.json and run-level timing
# ---------------------------------------------------------------------------


class _LLTask(Task):
    @property
    def instances(self) -> Iterator[Instance]:
        yield Instance(question="Q", gold_answer="A")

    def format_request(self, instance: Instance) -> LMRequest:
        return LMRequest(
            request_type=RequestType.LOGLIKELIHOOD, prompt=instance.question, continuations=("A",)
        )

    async def score_responses(
        self, responses: Sequence[Response], context: ScoringContext | None = None
    ) -> Sequence[Response]:
        for response in responses:
            response.scores["echo"] = 1.0
        return responses


def _ts(value: str) -> float:
    return datetime.fromisoformat(value).timestamp()


def test_task_spans_fall_within_processing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "olmo_eval.runners.asynq.results.BeakerStatusReporter",
        lambda: SimpleNamespace(
            progress_callback=lambda label: lambda *a, **k: None, flush=lambda: None
        ),
    )
    task = _LLTask(TaskConfig(name="ll", data_source="test/dataset"))
    import time

    processing_start = time.time()
    trackers = {}
    for spec in ("a", "b"):
        tracker = TaskTracker(model_name="m", spec=spec, task=task, total_instances=2)
        tracker.start_time = processing_start
        trackers[spec] = tracker

    result_queue: queue.Queue[ResultItem] = queue.Queue()
    for spec, offset in (("a", 0.0), ("b", 0.5)):
        for idx in range(2):
            outputs = [LMOutput("A", metadata=_ll(1, 10 + idx))]
            result_queue.put(
                _result(idx, outputs, sent_at=processing_start + offset + idx * 0.1, task_id=spec)
            )

    results = asyncio.run(
        process_results(
            trackers=trackers,
            result_queue=result_queue,  # type: ignore[arg-type]
            workers=[],
            scoring_context=ScoringContext(),
            scoring_concurrency=4,
            total_tasks=2,
            total_instances=4,
            model_name="m",
            save_predictions=False,
            write_predictions_fn=None,
            save_requests=False,
            write_requests_fn=None,
        )
    )
    end = time.time()

    for spec, offset in (("a", 0.0), ("b", 0.5)):
        result = results[spec]
        assert result.first_request_at == utc_iso(processing_start + offset)
        assert result.last_completed_at is not None
        last = _ts(result.last_completed_at)
        assert processing_start <= last <= end
        assert last == pytest.approx(processing_start + result.duration_seconds)
        assert result.prompt_tokens_total == 21
        assert result.completion_tokens_total == 0
        assert result.attributed_inference_seconds is None

    processing_seconds = _processing_seconds(results, processing_start, end)
    assert 0 <= processing_seconds <= end - processing_start
    assert processing_seconds == pytest.approx(
        max(_ts(r.last_completed_at or "") for r in results.values()) - processing_start
    )


def test_processing_seconds_falls_back_without_task_end_times() -> None:
    assert _processing_seconds({}, 100.0, 130.0) == 30.0


def _task_result(spec: str, **cost: object) -> TaskResult:
    return TaskResult(
        spec=spec,
        config={"name": spec},
        num_instances=1,
        metrics={"acc": {"s": 1.0}},
        duration_seconds=2.0,
        instances_processed=1,
        **cost,  # type: ignore[arg-type]
    )


def test_metrics_json_carries_task_cost_and_run_timing() -> None:
    cost = {
        "first_request_at": "2026-10-01T10:00:01+00:00",
        "last_completed_at": "2026-10-01T10:00:03+00:00",
        "prompt_tokens_total": 120,
        "completion_tokens_total": 0,
        "attributed_inference_seconds": 1.25,
    }
    provider = SimpleNamespace(alias=None, model="m", kind="vllm", to_dict=lambda: {})
    results_dict = aggregate_results({"t": _task_result("t", **cost)}, ["t"], ["t"], provider, None)
    assert {key: results_dict["tasks"]["t"][key] for key in TASK_COST_KEYS} == cost

    results_dict.update(
        startup_seconds=42.0,
        processing_started_at="2026-10-01T10:00:00+00:00",
        processing_seconds=3.5,
    )
    output = build_single_model_metrics(results_dict, experiment_id="e").to_dict()
    assert {key: output[key] for key in RUN_TIMING_KEYS} == {
        "startup_seconds": 42.0,
        "processing_started_at": "2026-10-01T10:00:00+00:00",
        "processing_seconds": 3.5,
    }
    (entry,) = output["tasks"]
    assert {key: entry[key] for key in TASK_COST_KEYS} == cost
    # Existing fields are unchanged
    assert entry["duration_seconds"] == 2.0
    assert "experiment_id" in output


def test_metrics_json_omits_unmeasured_cost() -> None:
    provider = SimpleNamespace(alias=None, model="m", kind="vllm", to_dict=lambda: {})
    results_dict = aggregate_results({"t": _task_result("t")}, ["t"], ["t"], provider, None)
    output = build_single_model_metrics(results_dict).to_dict()
    assert not set(RUN_TIMING_KEYS) & set(output)
    assert not set(TASK_COST_KEYS) & set(output["tasks"][0])
