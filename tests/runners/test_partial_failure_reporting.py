"""Regression tests for partial-failure reporting (issue #310 / U0005).

Seeded from a reproduction of 27B attempt-1 (experiment 01M0VW7G0WRRN1452DXCWMW8W6):
a healthy vLLM server rejected 47 of 63 requests with a per-request 400 (context
length). The harness logged "Processed 63/63", saved 16 predictions, printed a green
Success row and exited 0, because compute_task_metrics built an error summary from the
failed instances, logged it, and then dropped it on the floor.

The tests below pin the three halves of the fix:
  1. the error summary is attached to the TaskResult,
  2. saved-vs-processed accounting reaches the summary table, metrics.json and storage,
  3. a hard-failure rate above the configured threshold fails the task and the run,
     and the run is failed only after the results have been written.
"""

from __future__ import annotations

import asyncio
import logging
import queue
from types import SimpleNamespace

import pytest

from olmo_eval.cli.run.config import _apply_dotlist_overrides
from olmo_eval.cli.utils import process_ordered_args, reconstruct_ordered_args
from olmo_eval.common.metrics import AccuracyMetric
from olmo_eval.common.types import Instance, LMOutput, LMRequest, RequestType, Response
from olmo_eval.evals.suites import get_suite
from olmo_eval.evals.tasks.common import TaskConfig
from olmo_eval.harness.config import HarnessConfig
from olmo_eval.inference.errors import classify_terminal_provider_error
from olmo_eval.runners.asynq.preparation import compute_task_metrics, finalize_task
from olmo_eval.runners.asynq.processing import process_chat_request
from olmo_eval.runners.asynq.results import (
    _report_task_completion,
    aggregate_results,
    check_hard_failure_gate,
    is_hard_failure,
)
from olmo_eval.runners.asynq.runner import AsyncEvalRunner
from olmo_eval.runners.asynq.types import QueueItem, ResultItem, TaskTracker
from olmo_eval.runners.common.constants import HardFailureRateExceeded
from olmo_eval.runners.common.types import DEFAULT_MAX_HARD_FAILURE_RATE, TaskResult
from olmo_eval.runners.processing.aggregation import compute_suite_aggregations
from olmo_eval.runners.processing.metrics import build_single_model_metrics, log_summary
from olmo_eval.upload.uploader import _task_payload

SPEC = "deepscholar_bench:probe"
PRIMARY_METRIC = "accuracy:exact_match"
CLEAN_METRICS = {"accuracy": {"exact_match": 0.72}}
TOTAL_INSTANCES = 63
SAVED_INSTANCES = 16
HARD_FAILURES = TOTAL_INSTANCES - SAVED_INSTANCES  # 47


# ---------------------------------------------------------------------------
# Stubs reproducing the attempt-1 shape
# ---------------------------------------------------------------------------


def _context_length_400() -> Exception:
    """A per-request vLLM 400, as seen in 27B attempt-1.

    Deliberately NOT an EngineDeadError: the provider is healthy, it is these
    particular requests that are rejected.
    """
    error_type = type(
        "VLLMValidationError",
        (Exception,),
        {"__module__": "vllm.entrypoints.openai.protocol"},
    )
    return error_type(
        "This model's maximum context length is 40960 tokens. "
        "However, you requested 52731 tokens. (HTTP 400)"
    )


def _queue_item(instance_idx: int) -> QueueItem:
    return QueueItem(
        model_name="qwen3-8b-27b",
        task_id=SPEC,
        instance_idx=instance_idx,
        instance=Instance(question=f"q{instance_idx}", gold_answer="a"),
        request=LMRequest(request_type=RequestType.CHAT, prompt=f"q{instance_idx}"),
    )


def _stub_harness(
    hard_failing: set[int],
    soft_failing: set[int] | None = None,
) -> SimpleNamespace:
    """A harness whose provider is healthy but 400s a fixed subset of requests.

    Instances in `soft_failing` come back with an error AND a usable output, the
    MaxTurnsExceeded shape that must keep scoring.
    """
    soft = soft_failing or set()

    async def run(request, sampling_params, trace_metadata=None):
        idx = trace_metadata["instance_idx"]
        if idx in hard_failing:
            raise _context_length_400()
        return SimpleNamespace(
            final_output=LMOutput(text=f"answer {idx}", metadata={}),
            trajectory=None,
            error="MaxTurnsExceeded" if idx in soft else None,
            max_turns_reached=idx in soft,
            total_tool_calls=0,
            num_turns=1,
        )

    return SimpleNamespace(
        provider=SimpleNamespace(describe_request=lambda *a, **k: None),
        _apply_config=lambda request: request,
        run=run,
    )


class _StubTask:
    """Minimal task with one metric, so a clean run produces a primary score."""

    config = TaskConfig(
        name="deepscholar_bench",
        data_source="test/data",
        metrics=(AccuracyMetric(),),
    )

    def compute_metrics(self, responses: list[Response]) -> dict[str, dict[str, float]]:
        return CLEAN_METRICS


def _scored_response(idx: int) -> Response:
    return Response(
        instance=Instance(question=f"q{idx}", gold_answer="a"),
        request=LMRequest(request_type=RequestType.CHAT, prompt=f"q{idx}"),
        outputs=[LMOutput(text=f"answer {idx}", metadata={})],
    )


def _run_inference(
    total: int,
    hard_failing: set[int],
    soft_failing: set[int] | None = None,
) -> list[ResultItem]:
    """Drive the real per-instance chat path and collect what the workers emit."""
    harness = _stub_harness(hard_failing, soft_failing)
    result_queue: queue.Queue[ResultItem] = queue.Queue()

    async def drive() -> None:
        for idx in range(total):
            await process_chat_request(
                _queue_item(idx),
                harness,  # type: ignore[arg-type]
                result_queue,  # type: ignore[arg-type]
            )

    asyncio.run(drive())

    items: list[ResultItem] = []
    while not result_queue.empty():
        items.append(result_queue.get())
    return items


def _task_result_from(items: list[ResultItem], **kwargs) -> TaskResult:
    """Fold worker results into a TaskResult exactly as process_results would.

    Routing uses the production predicate, so a change to what counts as a hard
    failure shows up here instead of being masked by a copy of the rule.
    """
    hard_failures = {
        item.instance_idx: (item.error or "") for item in items if is_hard_failure(item)
    }
    scored = [_scored_response(item.instance_idx) for item in items if not is_hard_failure(item)]
    return compute_task_metrics(
        spec=SPEC,
        task=_StubTask(),  # type: ignore[arg-type]
        scored_responses=scored,
        failed_instances=hard_failures,
        total_instances=len(items),
        duration_seconds=1234.0,
        **kwargs,
    )


def _synthetic_result(total: int, failures: int, **kwargs) -> TaskResult:
    """A TaskResult for `failures` of `total` instances hard-failing."""
    return compute_task_metrics(
        spec=SPEC,
        task=_StubTask(),  # type: ignore[arg-type]
        scored_responses=[_scored_response(i) for i in range(total - failures)],
        failed_instances={i: "boom" for i in range(total - failures, total)},
        total_instances=total,
        duration_seconds=1.0,
        **kwargs,
    )


def _aggregate(result: TaskResult) -> dict:
    provider_config = SimpleNamespace(
        alias="qwen3-8b-27b",
        model="Qwen/Qwen3-8B",
        kind="vllm_server",
        to_dict=lambda: {},
    )
    return aggregate_results(
        results={SPEC: result},
        expanded_tasks=[SPEC],
        task_specs=[SPEC],
        provider_config=provider_config,
        attention_backend=None,
    )


# ---------------------------------------------------------------------------
# The premise: a per-request 400 is not terminal, so it lands as a hard failure
# ---------------------------------------------------------------------------


def test_per_request_400_is_not_a_terminal_provider_error() -> None:
    assert classify_terminal_provider_error(_context_length_400()) is None


def test_hard_and_soft_failures_stay_distinct() -> None:
    items = _run_inference(
        TOTAL_INSTANCES,
        hard_failing=set(range(SAVED_INSTANCES, TOTAL_INSTANCES)),
        soft_failing={0, 1},
    )
    hard = [item for item in items if is_hard_failure(item)]
    soft = [item for item in items if item.error and item.outputs]

    assert len(items) == TOTAL_INSTANCES
    assert len(hard) == HARD_FAILURES
    assert len(soft) == 2
    # The production predicate must not sweep soft failures in with hard ones.
    assert all(not is_hard_failure(item) for item in soft)


# ---------------------------------------------------------------------------
# 47/63 hard failures: the attempt-1 case
# ---------------------------------------------------------------------------


def test_partial_hard_failure_marks_task_failed() -> None:
    items = _run_inference(
        TOTAL_INSTANCES, hard_failing=set(range(SAVED_INSTANCES, TOTAL_INSTANCES))
    )
    result = _task_result_from(items)

    assert result.hard_failure_rate_exceeded is True
    assert result.error is not None
    assert "Hard failure rate" in result.error
    # The summary compute_task_metrics used to drop is now attached.
    assert result.error_summary is not None
    assert "47 instances failed" in result.error_summary
    assert "maximum context length" in result.error_summary

    assert result.num_instances == SAVED_INSTANCES
    assert result.instances_processed == TOTAL_INSTANCES
    assert result.instances_failed == HARD_FAILURES
    assert result.hard_failure_rate == pytest.approx(HARD_FAILURES / TOTAL_INSTANCES)


def test_partial_hard_failure_fails_the_run() -> None:
    items = _run_inference(
        TOTAL_INSTANCES, hard_failing=set(range(SAVED_INSTANCES, TOTAL_INSTANCES))
    )
    result = _task_result_from(items)

    with pytest.raises(HardFailureRateExceeded) as excinfo:
        check_hard_failure_gate({SPEC: result})
    assert SPEC in str(excinfo.value)


def test_partial_hard_failure_is_recorded_in_the_results_dict() -> None:
    items = _run_inference(
        TOTAL_INSTANCES, hard_failing=set(range(SAVED_INSTANCES, TOTAL_INSTANCES))
    )
    results = _aggregate(_task_result_from(items))
    task_entry = results["tasks"][SPEC]

    assert results["errors"], "a failed task must be recorded at run level"
    assert "error" in task_entry
    assert task_entry["instances_saved"] == SAVED_INSTANCES
    assert task_entry["instances_processed"] == TOTAL_INSTANCES
    assert task_entry["instances_failed"] == HARD_FAILURES
    assert "47 instances failed" in task_entry["error_summary"]


def test_gated_task_score_is_absent_from_the_summary() -> None:
    """A score computed on a subset must not be published as the task's score."""
    results = _aggregate(_synthetic_result(total=63, failures=47))

    assert SPEC not in results["summary"]
    assert "metrics" not in results["tasks"][SPEC]
    # The accounting still says what happened, so the exclusion is explainable.
    assert results["tasks"][SPEC]["instances_saved"] == 16
    assert results["tasks"][SPEC]["instances_processed"] == 63


def test_clean_task_score_is_present_in_the_summary() -> None:
    results = _aggregate(_synthetic_result(total=63, failures=0))

    assert SPEC in results["summary"]
    assert results["summary"][SPEC]["metric"] == PRIMARY_METRIC
    assert results["summary"][SPEC]["score"] == pytest.approx(0.72)
    assert results["tasks"][SPEC]["metrics"] == CLEAN_METRICS


# ---------------------------------------------------------------------------
# 100% hard failures: issue #310's BadRequestError case
# ---------------------------------------------------------------------------


def test_total_failure_marks_task_failed_and_fails_the_run() -> None:
    items = _run_inference(TOTAL_INSTANCES, hard_failing=set(range(TOTAL_INSTANCES)))
    result = _task_result_from(items)

    assert result.hard_failure_rate_exceeded is True
    assert result.error is not None
    assert "No instances were saved" in result.error
    assert result.error_summary is not None
    assert "63 instances failed" in result.error_summary
    assert result.num_instances == 0
    assert result.instances_processed == TOTAL_INSTANCES
    assert result.instances_failed == TOTAL_INSTANCES
    assert result.hard_failure_rate == pytest.approx(1.0)

    with pytest.raises(HardFailureRateExceeded):
        check_hard_failure_gate({SPEC: result})


# ---------------------------------------------------------------------------
# 0% hard failures: a clean run stays green and exits 0
# ---------------------------------------------------------------------------


def test_clean_run_stays_green() -> None:
    items = _run_inference(TOTAL_INSTANCES, hard_failing=set())
    result = _task_result_from(items)

    assert result.hard_failure_rate_exceeded is False
    assert result.error is None
    assert result.error_summary is None
    assert result.num_instances == TOTAL_INSTANCES
    assert result.instances_processed == TOTAL_INSTANCES
    assert result.instances_failed == 0
    assert result.hard_failure_rate == pytest.approx(0.0)

    check_hard_failure_gate({SPEC: result})  # must not raise

    results = _aggregate(result)
    assert results["errors"] == []
    assert results["tasks"][SPEC]["metrics"] == CLEAN_METRICS


def test_soft_failures_alone_keep_the_task_green() -> None:
    """MaxTurnsExceeded with a fallback answer still scores (preserved from PR#300)."""
    items = _run_inference(
        TOTAL_INSTANCES, hard_failing=set(), soft_failing=set(range(TOTAL_INSTANCES))
    )
    result = _task_result_from(items)

    assert result.instances_failed == 0
    assert result.num_instances == TOTAL_INSTANCES
    assert result.hard_failure_rate_exceeded is False
    assert result.error is None
    check_hard_failure_gate({SPEC: result})


# ---------------------------------------------------------------------------
# Threshold boundary and configurability
# ---------------------------------------------------------------------------


def test_default_threshold_is_five_percent() -> None:
    assert DEFAULT_MAX_HARD_FAILURE_RATE == 0.05


def test_rate_exactly_at_threshold_passes() -> None:
    result = _synthetic_result(total=20, failures=1)  # exactly 0.05

    assert result.hard_failure_rate == pytest.approx(0.05)
    assert result.hard_failure_rate_exceeded is False
    assert result.error is None
    # The summary is still attached even though the task passes.
    assert result.error_summary == "Instance 19 failed: boom"
    check_hard_failure_gate({SPEC: result})


def test_rate_just_under_threshold_passes() -> None:
    result = _synthetic_result(total=63, failures=3)  # 0.0476

    assert result.hard_failure_rate == pytest.approx(3 / 63)
    assert result.hard_failure_rate_exceeded is False
    assert result.error is None
    assert result.error_summary is not None
    check_hard_failure_gate({SPEC: result})


def test_rate_just_over_threshold_fails() -> None:
    result = _synthetic_result(total=63, failures=4)  # 0.0635

    assert result.hard_failure_rate_exceeded is True
    assert result.error is not None
    with pytest.raises(HardFailureRateExceeded):
        check_hard_failure_gate({SPEC: result})


def test_threshold_can_be_raised_to_tolerate_failures() -> None:
    result = _synthetic_result(total=63, failures=47, max_hard_failure_rate=0.9)

    assert result.hard_failure_rate_exceeded is False
    assert result.error is None
    # Tolerated, but never silent: the summary and the accounting survive.
    assert result.error_summary is not None
    assert result.instances_failed == 47
    assert result.instances_processed == 63
    check_hard_failure_gate({SPEC: result})


def test_all_failed_task_fails_even_at_the_maximum_threshold() -> None:
    """The opt-out tolerates partial failure, not a task that produced nothing."""
    result = _synthetic_result(total=63, failures=63, max_hard_failure_rate=1.0)

    assert result.num_instances == 0
    assert result.hard_failure_rate_exceeded is True
    assert result.error is not None
    assert "No instances were saved" in result.error
    assert "any hard failure rate budget" in result.error
    with pytest.raises(HardFailureRateExceeded):
        check_hard_failure_gate({SPEC: result})


def test_partial_failure_stays_quiet_at_the_maximum_threshold() -> None:
    """One survivor is still a result, so 1.0 tolerates it."""
    result = _synthetic_result(total=63, failures=62, max_hard_failure_rate=1.0)

    assert result.num_instances == 1
    assert result.hard_failure_rate_exceeded is False
    assert result.error is None
    assert result.error_summary is not None
    check_hard_failure_gate({SPEC: result})


def test_threshold_can_be_lowered_to_zero_tolerance() -> None:
    result = _synthetic_result(total=63, failures=1, max_hard_failure_rate=0.0)

    assert result.hard_failure_rate_exceeded is True
    assert result.error is not None
    with pytest.raises(HardFailureRateExceeded):
        check_hard_failure_gate({SPEC: result})


# ---------------------------------------------------------------------------
# The -o knob: the real CLI path, not just serialization
# ---------------------------------------------------------------------------


def _harness_overrides_from(argv: list[str]) -> list[str]:
    """Run the CLI's own argument association over a command line."""
    _, harness_overrides = process_ordered_args(reconstruct_ordered_args(argv))
    return harness_overrides


def test_max_hard_failure_rate_is_accepted_as_a_harness_override() -> None:
    """Registered in HARNESS_CONFIG_FIELDS, so the documented -o does not hard-error."""
    argv = [
        "run",
        "--harness",
        "default",
        "-o",
        "max_hard_failure_rate=0.2",
        "-m",
        "Qwen/Qwen3-8B",
        "-t",
        "deepscholar_bench",
    ]

    assert _harness_overrides_from(argv) == ["max_hard_failure_rate=0.2"]


def test_cli_override_reaches_harness_config() -> None:
    """The full CLI wiring: argv -> overrides -> to_dict -> from_dict."""
    argv = ["run", "--harness", "default", "-o", "max_hard_failure_rate=0.2"]

    harness_dict = HarnessConfig(name="default").to_dict()
    harness_dict = _apply_dotlist_overrides(harness_dict, _harness_overrides_from(argv))
    config = HarnessConfig.from_dict(harness_dict)

    assert config.max_hard_failure_rate == pytest.approx(0.2)
    # Must be a number, not the raw string, or the gate comparison would blow up.
    assert isinstance(config.max_hard_failure_rate, float)


def test_runner_resolves_the_configured_threshold() -> None:
    runner = AsyncEvalRunner.__new__(AsyncEvalRunner)

    runner.harness_config = HarnessConfig(name="default")
    assert runner.max_hard_failure_rate == DEFAULT_MAX_HARD_FAILURE_RATE

    runner.harness_config = HarnessConfig(name="default", max_hard_failure_rate=0.2)
    assert runner.max_hard_failure_rate == pytest.approx(0.2)


def test_harness_config_serializes_the_threshold() -> None:
    """Serialization round-trip, so a harness YAML can carry the setting."""
    config = HarnessConfig(name="test", max_hard_failure_rate=0.25)
    assert config.to_dict()["max_hard_failure_rate"] == 0.25
    assert HarnessConfig.from_dict(config.to_dict()).max_hard_failure_rate == 0.25

    default_config = HarnessConfig(name="test")
    assert default_config.max_hard_failure_rate is None
    assert "max_hard_failure_rate" not in default_config.to_dict()


# ---------------------------------------------------------------------------
# Wiring: the gate fires only after the results are on disk
# ---------------------------------------------------------------------------


def _runner_with_recorded_finalize(calls: list[str]) -> AsyncEvalRunner:
    runner = AsyncEvalRunner.__new__(AsyncEvalRunner)

    def fake_finalize_and_save(results_dict, **kwargs):
        calls.append("finalize_and_save")
        return results_dict

    runner._finalize_and_save = fake_finalize_and_save  # type: ignore[method-assign]
    return runner


def test_gate_runs_after_results_are_saved() -> None:
    """A run that trips the gate must still leave metrics.json and predictions behind."""
    calls: list[str] = []
    runner = _runner_with_recorded_finalize(calls)
    gated = _synthetic_result(total=63, failures=47)

    with pytest.raises(HardFailureRateExceeded):
        runner._finalize_and_gate({SPEC: gated}, {"tasks": {}}, experiment_id="exp")

    assert calls == ["finalize_and_save"], "results must be written before the gate fires"


def test_gate_returns_the_finalized_results_on_a_clean_run() -> None:
    calls: list[str] = []
    runner = _runner_with_recorded_finalize(calls)
    clean = _synthetic_result(total=63, failures=0)
    results_dict = {"tasks": {SPEC: {}}}

    returned = runner._finalize_and_gate({SPEC: clean}, results_dict, experiment_id="exp")

    assert returned is results_dict
    assert calls == ["finalize_and_save"]


# ---------------------------------------------------------------------------
# finalize_task: the one path the async runner actually reaches
# ---------------------------------------------------------------------------


def test_task_level_error_carries_its_summary() -> None:
    """The async runner calls finalize_task only for trackers that failed outright."""
    tracker = TaskTracker(
        model_name="qwen3-8b-27b",
        spec=SPEC,
        task=None,
        total_instances=63,
        error="Task preparation failed: dataset not found",
    )

    result = asyncio.run(finalize_task(tracker))

    assert result.error == "Task preparation failed: dataset not found"
    assert result.error_summary == "Task preparation failed: dataset not found"
    # Nothing was dispatched, so there is no instance accounting to report.
    # num_instances counts what was saved, so it cannot exceed what was processed.
    assert result.num_instances == 0
    assert result.instances_processed == 0
    assert result.instances_failed == 0
    assert result.hard_failure_rate_exceeded is False


# ---------------------------------------------------------------------------
# What the operator sees
# ---------------------------------------------------------------------------


def test_task_completion_line_reports_saved_vs_processed(caplog) -> None:
    caplog.set_level(logging.INFO)
    result = _synthetic_result(total=63, failures=47)

    _report_task_completion("qwen3-8b-27b", result)

    logged = " ".join(record.getMessage() for record in caplog.records)
    assert "16/63 instances saved" in logged
    assert "47 hard-failed" in logged


def test_task_completion_line_and_table_agree_on_no_accounting(caplog, capsys) -> None:
    """A task that never processed an instance reports "-" in both places."""
    caplog.set_level(logging.INFO)
    tracker = TaskTracker(
        model_name="qwen3-8b-27b",
        spec=SPEC,
        task=None,
        total_instances=63,
        error="Task preparation failed",
    )
    result = asyncio.run(finalize_task(tracker))
    assert result.num_instances == 0
    assert result.instances_processed == 0

    _report_task_completion("qwen3-8b-27b", result)
    logged = " ".join(record.getMessage() for record in caplog.records)
    assert "instances: -" in logged
    assert "63/63" not in logged

    log_summary(_aggregate(result))
    out = capsys.readouterr().out
    assert "63/63" not in out


def test_summary_table_reports_saved_vs_processed(capsys) -> None:
    results = _aggregate(_synthetic_result(total=63, failures=47))

    log_summary(results)

    out = capsys.readouterr().out
    assert "16/63" in out
    assert "Failed" in out


def test_metrics_json_carries_the_accounting() -> None:
    results = _aggregate(_synthetic_result(total=63, failures=47))

    metrics_output = build_single_model_metrics(results).to_dict()
    entry = metrics_output["tasks"][0]

    assert entry["instances_saved"] == 16
    assert entry["instances_processed"] == 63
    assert entry["instances_failed"] == 47
    # num_instances is the saved count, so it must not read 0 next to instances_saved 16.
    assert entry["num_instances"] == 16
    assert "47 instances failed" in entry["error_summary"]
    assert metrics_output["errors"], "the failed task must appear in metrics.json errors"


def test_metrics_json_accounting_on_a_clean_run() -> None:
    results = _aggregate(_synthetic_result(total=63, failures=0))

    entry = build_single_model_metrics(results).to_dict()["tasks"][0]

    assert entry["instances_saved"] == 63
    assert entry["instances_processed"] == 63
    assert entry["instances_failed"] == 0
    assert entry["num_instances"] == 63
    assert "error_summary" not in entry


# ---------------------------------------------------------------------------
# Downstream: uploaded task rows and suite averages
# ---------------------------------------------------------------------------


def _uploaded_task_row(results: dict, tmp_path) -> dict:
    metrics_output = build_single_model_metrics(results).to_dict()
    run_errors = {e["task"]: e["error"] for e in metrics_output["errors"]}
    row = metrics_output["tasks"][0]
    return _task_payload(tmp_path, row, None, run_errors, metrics_output["summary"]).payload


def test_uploaded_task_row_explains_a_gated_task(tmp_path) -> None:
    """The uploaded row must say what happened, not just carry empty metrics."""
    results = _aggregate(_synthetic_result(total=63, failures=47))
    results["tasks"][SPEC]["task_hash"] = "deadbeef"

    row = _uploaded_task_row(results, tmp_path)

    assert row["num_instances"] == 16
    assert row["instances_processed"] == 63
    assert row["instances_failed"] == 47
    assert row["error"].startswith("Hard failure rate")
    assert "47 instances failed" in row["error_summary"]["summary"]


def test_uploaded_task_row_on_a_clean_run(tmp_path) -> None:
    results = _aggregate(_synthetic_result(total=63, failures=0))
    results["tasks"][SPEC]["task_hash"] = "deadbeef"

    row = _uploaded_task_row(results, tmp_path)

    assert row["num_instances"] == 63
    assert row["instances_processed"] == 63
    assert row["instances_failed"] == 0
    assert row["error"] is None
    assert row["error_summary"] is None


def test_suite_average_names_the_task_it_excluded(caplog) -> None:
    """Excluding a failed member changes what the average covers, so say so."""
    caplog.set_level(logging.WARNING)
    suite_name = "mt_mbpp_v2fix"
    expanded = get_suite(suite_name).expand()
    assert len(expanded) >= 2, "this test needs a suite with more than one task"

    task_results: dict[str, dict] = {
        task: {"metrics": {"accuracy": {"exact_match": 0.75}}} for task in expanded
    }
    failed_task = expanded[0]
    task_results[failed_task] = {
        "error": "Hard failure rate 74.6% exceeds the maximum of 5.0%",
        "instances_saved": 16,
        "instances_processed": 63,
        "instances_failed": 47,
    }

    aggregations = compute_suite_aggregations([suite_name], task_results)

    logged = " ".join(record.getMessage() for record in caplog.records)
    assert suite_name in logged
    assert failed_task in logged
    assert "Hard failure rate" in logged
    # And the average genuinely excludes it.
    assert aggregations[suite_name]["num_tasks"] == len(expanded) - 1
    assert failed_task not in aggregations[suite_name]["tasks"]


def test_suite_average_stays_quiet_when_nothing_failed(caplog) -> None:
    caplog.set_level(logging.WARNING)
    suite_name = "mt_mbpp_v2fix"
    expanded = get_suite(suite_name).expand()
    task_results = {task: {"metrics": {"accuracy": {"exact_match": 0.75}}} for task in expanded}

    compute_suite_aggregations([suite_name], task_results)

    assert not [r for r in caplog.records if "excluding failed task" in r.getMessage()]
