"""Per-task cost in the async runner: request timing, token totals and batch attribution."""

from __future__ import annotations

import asyncio
import queue
from collections.abc import Iterator, Sequence
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from olmo_eval.common.execution.environment import ScoringContext
from olmo_eval.common.types import Instance, LMOutput, LMRequest, RequestType, Response
from olmo_eval.evals.tasks.common import Task, TaskConfig
from olmo_eval.inference.metrics.core.schema import BatchMetrics, RequestMetrics
from olmo_eval.runners.asynq.processing import attribute_batch_seconds, process_batch
from olmo_eval.runners.asynq.results import aggregate_results, process_results, utc_iso
from olmo_eval.runners.asynq.runner import _processing_seconds
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
# Batch attribution in the worker
# ---------------------------------------------------------------------------


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


def test_attribution_splits_wall_time_by_token_share() -> None:
    batch = _batch(4.0, [_request_metrics(30, 10), _request_metrics(10, 10)])
    assert attribute_batch_seconds(batch, 2) == [pytest.approx(8 / 3), pytest.approx(4 / 3)]


def test_attribution_needs_matching_request_metrics() -> None:
    batch = _batch(4.0, [_request_metrics(30, 10)])
    assert attribute_batch_seconds(batch, 2) == [None, None]
    assert attribute_batch_seconds(None, 2) == [None, None]
    assert attribute_batch_seconds(Mock(), 1) == [None]
    empty = _batch(3.0, [_request_metrics(0, 0), _request_metrics(0, 0)])
    assert attribute_batch_seconds(empty, 2) == [1.5, 1.5]


def _queue_item(idx: int, task_id: str) -> QueueItem:
    return QueueItem(
        model_name="m",
        task_id=task_id,
        instance_idx=idx,
        instance=Instance(question=f"q{idx}", gold_answer="a"),
        request=LMRequest(request_type=RequestType.COMPLETION, prompt=f"q{idx}"),
    )


def test_process_batch_stamps_send_time_and_attribution(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("olmo_eval.runners.asynq.processing.time.time", lambda: 1000.0)
    batch = _batch(6.0, [_request_metrics(10, 10), _request_metrics(30, 10)])
    harness = SimpleNamespace(
        provider=SimpleNamespace(
            describe_request=Mock(return_value=None),
            agenerate=AsyncMock(return_value=[[LMOutput("a")], [LMOutput("b")]]),
        ),
        _apply_config=lambda request: request,
        flush_metrics=Mock(return_value=batch),
    )
    result_queue: queue.Queue[ResultItem] = queue.Queue()

    items = [_queue_item(0, "easy"), _queue_item(1, "hard")]
    asyncio.run(process_batch(items, harness, result_queue))  # type: ignore[arg-type]

    results = [result_queue.get_nowait() for _ in range(2)]
    assert [r.sent_at for r in results] == [1000.0, 1000.0]
    assert [r.attributed_seconds for r in results] == [2.0, 4.0]


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
