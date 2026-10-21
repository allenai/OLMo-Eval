"""Tests for the single-turn IFEval task."""

from typing import Any

import pytest

from olmo_eval.common.scorers import IFEvalScorer
from olmo_eval.common.types import Instance, LMOutput, RequestType
from olmo_eval.evals.tasks.common import get_task

# A real google/IFEval row shape: kwargs arrive null-padded to the dataset's
# full schema, with one entry per instruction id.
_DOC: dict[str, Any] = {
    "key": 1000,
    "prompt": (
        "Write a resume for a software engineer. Your answer must contain "
        "exactly 3 bullet points, such as: * This is a point."
    ),
    "instruction_id_list": ["detectable_format:number_bullet_lists"],
    "kwargs": [{"num_bullets": 3, "num_placeholders": None, "language": None}],
}
_THREE_BULLETS = "* Led backend\n* Shipped v2\n* Cut latency"


def _instance() -> Instance:
    instance = get_task("ifeval").process_doc(dict(_DOC), index=0)
    assert instance is not None
    return instance


def _score(text: str) -> tuple[float, dict[str, Any]]:
    """Score through the task's extraction step, as the pipeline does."""
    task = get_task("ifeval")
    output = LMOutput(text=text)
    output.extracted_answer = task.extract_answer(output)
    score = IFEvalScorer().score(_instance(), output)
    return score, output.metadata["ifeval"]


def test_registered() -> None:
    task = get_task("ifeval")
    assert task.request_type == RequestType.CHAT
    params = task.config.sampling_params
    assert params.max_tokens is None
    assert params.temperature == 0.6
    assert params.top_p == 0.95


def test_metrics() -> None:
    """All four IFEval aggregations, with loose prompt-level primary."""
    task = get_task("ifeval")
    assert {metric.name for metric in task.config.metrics} == {
        "prompt_level_strict_acc",
        "prompt_level_loose_acc",
        "inst_level_strict_acc",
        "inst_level_loose_acc",
    }
    primary = task.config.get_primary_metric()
    assert primary is not None
    assert primary.name == "prompt_level_loose_acc"


def test_process_doc_drops_null_padded_kwargs() -> None:
    """Absent kwargs arrive as None and must not reach the verifier."""
    instance = _instance()
    assert instance.metadata["kwargs"] == [{"num_bullets": 3}]
    assert instance.metadata["instruction_id_list"] == ["detectable_format:number_bullet_lists"]
    assert instance.metadata["key"] == 1000
    assert instance.metadata["prompt"] == _DOC["prompt"]
    assert instance.gold_answer is None


def test_process_doc_without_instructions() -> None:
    instance = get_task("ifeval").process_doc({"key": 7, "prompt": "Hello."}, index=0)
    assert instance is not None
    assert instance.metadata["instruction_id_list"] == []
    assert instance.metadata["kwargs"] == []


def test_format_request_is_single_user_message() -> None:
    request = get_task("ifeval").format_request(_instance())
    assert request.request_type == RequestType.CHAT
    assert len(request.messages) == 1
    assert request.messages[0]["role"] == "user"
    assert request.messages[0]["content"] == _DOC["prompt"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("  spaced  ", "  spaced  "),
        ("<think>draft the bullets</think>" + _THREE_BULLETS, _THREE_BULLETS),
        ("<think>never closed", "<think>never closed"),
    ],
)
def test_extract_answer_drops_think_block(text: str, expected: str) -> None:
    assert get_task("ifeval").extract_answer(LMOutput(text=text)) == expected


def test_compliant_response() -> None:
    score, result = _score(_THREE_BULLETS)
    assert score == 1.0
    assert result["strict"] == [True]
    assert result["loose"] == [True]


def test_noncompliant_response() -> None:
    score, result = _score("I led the backend team and shipped v2.")
    assert score == 0.0
    assert result["strict"] == [False]
    assert result["loose"] == [False]


def test_think_block_is_not_verified() -> None:
    """Bullets drafted while thinking must not count toward the constraint."""
    score, result = _score("<think>* a\n* b\n* c\n* d</think>" + _THREE_BULLETS)
    assert score == 1.0
    assert result["strict"] == [True]


def test_scorer_falls_back_to_text_without_extraction() -> None:
    output = LMOutput(text=_THREE_BULLETS)
    assert IFEvalScorer().score(_instance(), output) == 1.0


def test_no_instructions_scores_zero() -> None:
    instance = get_task("ifeval").process_doc({"key": 7, "prompt": "Hello."}, index=0)
    assert instance is not None
    assert IFEvalScorer().score(instance, LMOutput(text="hi")) == 0.0


def test_verify_instructions_records_instructions_it_cannot_build() -> None:
    from ifbench import instructions_registry

    from olmo_eval.common.scorers.ifeval import verify_instructions

    result = verify_instructions(
        "Rain on the roof",
        ["punctuation:no_comma", "not:an_instruction"],
        [{}, {}],
        "prompt",
        instructions_registry.INSTRUCTION_DICT,
    )
    assert result["strict"] == [True, False]
    assert result["loose"] == [True, False]
    assert set(result["errors"]) == {"not:an_instruction"}
