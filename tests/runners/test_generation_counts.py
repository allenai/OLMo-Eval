"""Tests for per-task generation counts."""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from typing import Any

import pytest

from olmo_eval.common.metrics import AccuracyMetric
from olmo_eval.common.types import (
    TURNS_KEY,
    Function,
    Instance,
    LMOutput,
    LMRequest,
    RequestType,
    Response,
    ToolCall,
)
from olmo_eval.evals.tasks.common import TaskConfig, get_task
from olmo_eval.runners.asynq.preparation import compute_task_metrics, finalize_task
from olmo_eval.runners.asynq.results import _report_task_completion, aggregate_results
from olmo_eval.runners.asynq.types import TaskTracker
from olmo_eval.runners.processing.aggregation import compute_suite_aggregations
from olmo_eval.runners.processing.generation_counts import (
    count_generations,
    format_generation_counts,
    sum_generation_counts,
)
from olmo_eval.runners.processing.metrics import (
    build_multi_model_metrics,
    build_single_model_metrics,
    log_summary,
)
from olmo_eval.upload.uploader import _task_payload

SPEC = "ifeval_ood"
CLEAN_METRICS = {"accuracy": {"exact_match": 0.5}}


def _response(
    *outputs: tuple[str, str | None],
    request_type: RequestType = RequestType.CHAT,
) -> Response:
    """A response whose outputs are (text, finish_reason) pairs."""
    return Response(
        instance=Instance(question="q", gold_answer="a"),
        request=LMRequest(request_type=request_type, prompt="q"),
        outputs=[
            LMOutput(
                text=text,
                metadata={} if finish_reason is None else {"finish_reason": finish_reason},
            )
            for text, finish_reason in outputs
        ],
    )


def _stripped(responses: list[Response]) -> list[Response]:
    """Strip thinking traces with a real reasoning-regime task, as the runner does."""
    get_task(SPEC).strip_thinking_traces(responses)
    return responses


def _reasoning_batch() -> list[Response]:
    """One of each outcome, in the shapes a thinking model produces."""
    return _stripped(
        [
            # Clean.
            _response(("<think>plan</think>\nFinal answer.", "stop")),
            # Cut off in a template-opened trace.
            _response(("step 1, step 2, step 3", "length")),
            # Cut off in a trace the model opened.
            _response(("<think>still thinking", "length")),
            # Empty answer.
            _response(("<think>plan</think>\n  ", "stop")),
        ]
    )


class _StubTask:
    config = TaskConfig(name=SPEC, data_source="test/data", metrics=(AccuracyMetric(),))

    def strip_thinking_traces(self, responses: list[Response]) -> None:
        get_task(SPEC).strip_thinking_traces(responses)

    async def score_responses(self, responses: list[Response]) -> list[Response]:
        return responses

    def _get_scorers(self) -> dict[str, object]:
        return {"code_exec": object()}

    def compute_metrics(self, responses: list[Response]) -> dict[str, dict[str, float]]:
        return CLEAN_METRICS


def _task_result(responses: list[Response]):
    return compute_task_metrics(
        spec=SPEC,
        task=_StubTask(),  # type: ignore[arg-type]
        scored_responses=responses,
        failed_instances={},
        total_instances=len(responses),
        duration_seconds=1.0,
    )


def _aggregate(result, spec: str = SPEC) -> dict[str, Any]:
    provider_config = SimpleNamespace(
        alias="m", model="org/m", kind="vllm_server", to_dict=lambda: {}
    )
    return aggregate_results(
        results={spec: result},
        expanded_tasks=[spec],
        task_specs=[spec],
        provider_config=provider_config,
        attention_backend=None,
    )


def test_counts_each_outcome() -> None:
    counts = count_generations(_reasoning_batch())

    assert counts == {
        "generations": 4,
        "cap_hit": 2,
        "empty": 1,
        "unclosed_think": 1,
        "finish_reason_unknown": 0,
    }


def test_an_unclosed_trace_on_an_earlier_turn_counts_the_conversation_once() -> None:
    response = _response(("Answer.", "stop"))
    response.outputs[0].metadata[TURNS_KEY] = [
        {"text": "<think>still reasoning", "finish_reason": "length"},
        {"text": "<think>also unfinished", "finish_reason": "length"},
        {"text": "Answer.", "finish_reason": "stop"},
    ]

    counts = count_generations([response])

    assert counts is not None
    assert (counts["generations"], counts["unclosed_think"]) == (1, 1)


def test_closed_trace_opened_by_the_template_is_not_unclosed() -> None:
    """With the opener in the prompt, a closed trace shows only </think>."""
    counts = count_generations(_stripped([_response(("reasoning</think>\nAnswer.", "stop"))]))

    assert counts is not None
    assert counts["unclosed_think"] == 0
    assert counts["empty"] == 0


def test_missing_finish_reason_is_counted_as_unknown_not_as_complete() -> None:
    counts = count_generations([_response(("answer", None), ("", None))])

    assert counts is not None
    assert counts["finish_reason_unknown"] == 2
    assert counts["cap_hit"] == 0
    assert counts["empty"] == 1


def test_every_sample_of_a_multi_sample_request_is_counted() -> None:
    counts = count_generations([_response(("a", "stop"), ("b", "length"), ("c", "length"))])

    assert counts is not None
    assert counts["generations"] == 3
    assert counts["cap_hit"] == 2


def test_tool_call_output_with_no_text_is_not_empty() -> None:
    """Native tool calling returns text="" beside the structured calls it scores."""
    response = _response(("", "tool_calls"))
    response.outputs[0].tool_calls = [
        ToolCall(id="call_0", function=Function(name="get_weather", arguments='{"city": "Paris"}'))
    ]

    counts = count_generations([response])

    assert counts is not None
    assert counts["empty"] == 0


def test_loglikelihood_tasks_have_no_counts() -> None:
    responses = [_response((" yes", None), request_type=RequestType.LOGLIKELIHOOD)]

    assert count_generations(responses) is None


def test_format_lists_every_count_and_flags_unknown_only_when_present() -> None:
    assert format_generation_counts(count_generations(_reasoning_batch())) == (
        "cap-hit 2/4, empty 1, unclosed-think 1"
    )
    assert "finish-reason unknown 1" in format_generation_counts(
        count_generations([_response(("answer", None))])
    )
    assert format_generation_counts(None) == "-"
    assert "unclosed-think" not in format_generation_counts(
        count_generations([_response(("template-opened reasoning", "length"))])
    )


def test_compute_task_metrics_attaches_counts() -> None:
    result = _task_result(_reasoning_batch())

    assert result.generation_counts is not None
    assert result.generation_counts["cap_hit"] == 2
    assert result.to_dict()["generation_counts"] == result.generation_counts


def test_finalize_task_attaches_counts_after_stripping() -> None:
    """finalize_task strips traces itself, so a closed empty answer counts as empty."""
    raw = [
        _response(("<think>plan</think>\n  ", "stop")),
        _response(("step 1, step 2", "length")),
    ]
    tracker = TaskTracker(
        model_name="m",
        spec=SPEC,
        task=_StubTask(),  # type: ignore[arg-type]
        total_instances=len(raw),
        responses=dict(enumerate(raw)),
    )

    result = asyncio.run(finalize_task(tracker))

    assert result.generation_counts is not None
    assert result.generation_counts["empty"] == 1
    assert result.generation_counts["cap_hit"] == 1


def test_task_completion_line_reports_counts(caplog) -> None:
    caplog.set_level(logging.INFO)

    _report_task_completion("m", _task_result(_reasoning_batch()))

    logged = " ".join(record.getMessage() for record in caplog.records)
    assert "4/4 instances saved; cap-hit 2/4, empty 1, unclosed-think 1" in logged


def test_summary_table_reports_counts_beside_the_score(capsys) -> None:
    log_summary(_aggregate(_task_result(_reasoning_batch())))

    out = capsys.readouterr().out
    assert "Generations" in out
    assert "cap-hit 2/4" in out
    assert "0.5000" in out


def test_metrics_json_carries_counts() -> None:
    results = _aggregate(_task_result(_reasoning_batch()))

    entry = build_single_model_metrics(results).to_dict()["tasks"][0]

    assert entry["generation_counts"]["cap_hit"] == 2
    assert entry["generation_counts"]["generations"] == 4


def test_metrics_json_omits_counts_for_loglikelihood_tasks() -> None:
    responses = [_response((" yes", None), request_type=RequestType.LOGLIKELIHOOD)]
    results = _aggregate(_task_result(responses))

    entry = build_single_model_metrics(results).to_dict()["tasks"][0]

    assert "generation_counts" not in entry


def test_uploaded_task_row_carries_counts(tmp_path) -> None:
    results = _aggregate(_task_result(_reasoning_batch()))
    row = results["tasks"][SPEC]
    row["task"] = SPEC
    row["task_hash"] = "deadbeef"
    task = _task_payload(tmp_path, row, None, {}, {})
    assert task.payload["generation_counts"]["cap_hit"] == 2
    assert task.payload["generation_counts"]["generations"] == 4


def test_upload_omits_generation_counts_when_unavailable(tmp_path) -> None:
    results = _aggregate(
        _task_result([_response(("yes", None), request_type=RequestType.LOGLIKELIHOOD)])
    )
    row = results["tasks"][SPEC]
    row.update(task=SPEC, task_hash="deadbeef")
    assert "generation_counts" not in _task_payload(tmp_path, row, None, {}, {}).payload


def _task_data(cap_hit: int, generations: int) -> dict[str, Any]:
    return {
        "metrics": CLEAN_METRICS,
        "primary_metric": "accuracy:exact_match",
        "num_instances": generations,
        "generation_counts": {
            "generations": generations,
            "cap_hit": cap_hit,
            "empty": 0,
            "unclosed_think": 0,
            "finish_reason_unknown": 0,
        },
    }


def test_average_suite_sums_member_counts() -> None:
    task_results = {
        "ifeval_mt_wildchat_unused_withRewrite": _task_data(cap_hit=1, generations=10),
        "ifeval_mt_ood_wildchat_unused_withRewrite": _task_data(cap_hit=2, generations=10),
        "ifeval_ood": _task_data(cap_hit=291, generations=300),
    }

    suite = compute_suite_aggregations(["ifbench"], task_results)["ifbench"]

    assert suite["generation_counts"]["cap_hit"] == 294
    assert suite["generation_counts"]["generations"] == 320


def test_average_of_averages_suite_sums_member_counts() -> None:
    task_results = {
        "aime_2024:pass_at_32": _task_data(cap_hit=5, generations=960),
        "aime_2025:pass_at_32": _task_data(cap_hit=7, generations=960),
    }

    suite = compute_suite_aggregations(["aime_2022_to_2026"], task_results)["aime_2022_to_2026"]

    assert suite["generation_counts"]["cap_hit"] == 12
    assert suite["generation_counts"]["generations"] == 1920


def test_metrics_json_summary_carries_suite_counts() -> None:
    task_results = {
        "ifeval_mt_wildchat_unused_withRewrite": _task_data(cap_hit=1, generations=10),
        "ifeval_mt_ood_wildchat_unused_withRewrite": _task_data(cap_hit=2, generations=10),
        "ifeval_ood": _task_data(cap_hit=291, generations=300),
    }
    results = {
        "tasks": task_results,
        "suites": compute_suite_aggregations(["ifbench"], task_results),
    }

    single = build_single_model_metrics(results).to_dict()["summary"]
    multi = build_multi_model_metrics({"models": {"m": results}}).to_dict()["summary"]["m"]

    for summary in (single, multi):
        assert summary["ifbench"]["generation_counts"]["cap_hit"] == 294
        assert "generation_counts" not in summary["ifeval_ood"]


def test_sum_is_none_when_no_member_has_counts() -> None:
    assert sum_generation_counts({"a": {"metrics": CLEAN_METRICS}}, ["a", "missing"]) is None


def test_overlapping_suite_members_count_each_task_once() -> None:
    counts = sum_generation_counts(
        {
            "shared": _task_data(cap_hit=2, generations=10),
            "other": _task_data(cap_hit=1, generations=5),
        },
        ["shared", "other", "shared"],
    )

    assert counts is not None
    assert counts["generations"] == 15
    assert counts["cap_hit"] == 3


@pytest.mark.parametrize("spec", ["sage_short_form", "sage_open_ended"])
@pytest.mark.parametrize(
    "text,expected_empty",
    [("<think>target</think>  ", True), ("<think>target", True), ("answer<think>target", False)],
)
@pytest.mark.parametrize("has_tool_call", [False, True])
def test_sage_empty_counts_match_its_scored_text(spec, text, expected_empty, has_tool_call):
    from olmo_eval.evals.tasks.sage import make_gold

    task = get_task(spec)
    assert task.config.strip_thinking is False
    response = _response((text, "stop"))
    gold = make_gold("gold", "target")
    response.instance.metadata.update({"gold": gold, "golds": [(gold, 1)]})
    if has_tool_call:
        response.outputs[0].tool_calls = [
            ToolCall(id="call", function=Function(name="search", arguments="{}"))
        ]
    scored = asyncio.run(task.score_responses([response]))
    counts = count_generations(scored)
    assert counts is not None
    assert counts["empty"] == int(expected_empty and not has_tool_call)
    assert response.outputs[0].text == text
