"""Tests for the Multi-IF task, its scorer and its metrics."""

from __future__ import annotations

import json
from typing import Any

import pytest

from olmo_eval.common.metrics import MultiIFAccuracy
from olmo_eval.common.scorers import MultiIFScorer
from olmo_eval.common.scorers.base import get_scorer_result
from olmo_eval.common.types import (
    FOLLOW_UP_TURNS_KEY,
    TURNS_KEY,
    Instance,
    LMOutput,
    RequestType,
    Response,
)
from olmo_eval.evals.tasks.common import get_task

_TURN_INSTRUCTIONS: list[tuple[str, list[str], list[dict[str, Any]]]] = [
    ("Write a haiku about rain. Use no commas.", ["punctuation:no_comma"], [{}]),
    (
        'End your reply with "Done."',
        ["punctuation:no_comma", "startend:end_checker"],
        [{}, {"end_phrase": "Done."}],
    ),
    (
        "Wrap your entire reply in double quotes.",
        ["punctuation:no_comma", "startend:end_checker", "startend:quotation"],
        [{}, {"end_phrase": "Done."}, {}],
    ),
]


def _doc(turns: int = 3, language: str = "English", key: str = "1:1:en") -> dict[str, Any]:
    """A ``facebook/Multi-IF`` row: every turn field is a JSON-encoded string."""
    doc: dict[str, Any] = {"key": key, "language": language, "turns": None, "responses": None}
    for index in range(1, 4):
        if index <= turns:
            prompt, ids, kwargs = _TURN_INSTRUCTIONS[index - 1]
            doc[f"turn_{index}_prompt"] = json.dumps({"role": "user", "content": prompt})
            doc[f"turn_{index}_instruction_id_list"] = json.dumps(ids)
            doc[f"turn_{index}_kwargs"] = json.dumps([json.dumps(kw) for kw in kwargs])
        else:
            doc[f"turn_{index}_prompt"] = None
            doc[f"turn_{index}_instruction_id_list"] = None
            doc[f"turn_{index}_kwargs"] = None
    return doc


def _instance(**kwargs: Any) -> Instance:
    instance = get_task("multi_if").process_doc(_doc(**kwargs))
    assert instance is not None
    return instance


def _response(instance: Instance, replies: list[str] | None) -> Response:
    """A response scored the way the pipeline does: extract, then score."""
    task = get_task("multi_if")
    metadata = {} if replies is None else {TURNS_KEY: [{"text": r} for r in replies]}
    output = LMOutput(text=(replies or [""])[-1], metadata=metadata)
    output.extracted_answer = task.extract_answer(output)
    response = Response(instance=instance, request=task.format_request(instance), outputs=[output])
    response.scores[MultiIFScorer.name] = MultiIFScorer().score(instance, output)
    return response


FOLLOWS_ALL = ["Rain on the roof", "Rain on the roof. Done.", '"Rain on the roof. Done."']


def test_registered() -> None:
    task = get_task("multi_if")
    assert task.request_type == RequestType.CHAT
    assert task.required_scaffold == "multi_turn_chat"
    assert task.config.strip_thinking
    params = task.config.sampling_params
    assert params.max_tokens is None
    assert params.temperature == 0.6
    assert params.top_p == 0.95


def test_paper_variant_uses_the_papers_sampling() -> None:
    params = get_task("multi_if:paper").config.sampling_params
    assert (params.max_tokens, params.temperature, params.top_p) == (1024, 0.6, 0.9)


def test_metrics() -> None:
    task = get_task("multi_if")
    names = {metric.name for metric in task.config.metrics}
    assert {"accuracy", "turn_1_accuracy", "turn_2_accuracy", "turn_3_accuracy"} <= names
    assert {"english_accuracy", "chinese_accuracy", "hindi_accuracy"} <= names
    primary = task.config.get_primary_metric()
    assert primary is not None
    assert primary.name == "accuracy"


def test_process_doc_decodes_each_turn() -> None:
    instance = _instance()
    assert instance.metadata["id"] == "1:1:en"
    assert instance.metadata["language"] == "English"
    assert instance.question == _TURN_INSTRUCTIONS[0][0]
    assert instance.metadata["turns"][2] == {
        "prompt": _TURN_INSTRUCTIONS[2][0],
        "instruction_id_list": _TURN_INSTRUCTIONS[2][1],
        "kwargs": _TURN_INSTRUCTIONS[2][2],
    }


def test_process_doc_stops_at_the_conversations_last_turn() -> None:
    assert len(_instance(turns=2).metadata["turns"]) == 2


def test_process_doc_drops_null_kwargs() -> None:
    doc = _doc(turns=1)
    doc["turn_1_kwargs"] = json.dumps([json.dumps({"num_words": None})])
    instance = get_task("multi_if").process_doc(doc)
    assert instance is not None
    assert instance.metadata["turns"][0]["kwargs"] == [{}]


def test_format_request_sends_the_first_turn_and_scripts_the_rest() -> None:
    request = get_task("multi_if").format_request(_instance())
    assert request.messages == ({"role": "user", "content": _TURN_INSTRUCTIONS[0][0]},)
    assert request.metadata == {
        FOLLOW_UP_TURNS_KEY: [
            {"role": "user", "content": _TURN_INSTRUCTIONS[1][0]},
            {"role": "user", "content": _TURN_INSTRUCTIONS[2][0]},
        ]
    }


def test_extract_answer_strips_each_turns_reasoning_and_whitespace() -> None:
    output = LMOutput(
        text="", metadata={TURNS_KEY: [{"text": "<think>plan</think>\n\na"}, {"text": " b\n"}]}
    )
    assert get_task("multi_if").extract_answer(output) == ["a", "b"]


def test_extraction_reads_replies_after_trace_stripping() -> None:
    # Trace stripping rewrites the output's text, the last reply; each turn's
    # reply is still read from the replies the scaffold recorded.
    task = get_task("multi_if")
    instance = _instance(turns=2)
    output = LMOutput(
        text="<think>b</think>Done.",
        metadata={TURNS_KEY: [{"text": "<think>a</think>Rain"}, {"text": "<think>b</think>Done."}]},
    )
    response = Response(instance=instance, request=task.format_request(instance), outputs=[output])
    task.strip_thinking_traces([response])
    task._extract_answers([response])
    assert response.outputs[0].text == "Done."
    assert response.outputs[0].extracted_answer == ["Rain", "Done."]


def test_extract_answer_without_replies() -> None:
    assert get_task("multi_if").extract_answer(LMOutput(text="hello")) is None


def test_scorer_checks_each_turn_against_its_own_instructions() -> None:
    response = _response(_instance(), FOLLOWS_ALL)
    result = get_scorer_result(response.outputs[0], MultiIFScorer.name)
    assert result is not None
    assert [turn["loose"] for turn in result["turns"]] == [[True], [True, True], [True] * 3]
    assert response.scores[MultiIFScorer.name] == 1.0


def test_scorer_counts_turns_whose_instructions_all_pass() -> None:
    replies = ["Rain, on the roof", "Rain on the roof. Done.", "Rain on the roof. Done."]
    response = _response(_instance(), replies)
    result = get_scorer_result(response.outputs[0], MultiIFScorer.name)
    assert result is not None
    assert [turn["strict"] for turn in result["turns"]] == [
        [False],
        [True, True],
        [True, True, False],
    ]
    assert response.scores[MultiIFScorer.name] == pytest.approx(1 / 3)


def test_scorer_fails_turns_the_conversation_never_reached() -> None:
    response = _response(_instance(), FOLLOWS_ALL[:1])
    result = get_scorer_result(response.outputs[0], MultiIFScorer.name)
    assert result is not None
    assert [turn["loose"] for turn in result["turns"]] == [[True], [False] * 2, [False] * 3]


def test_scorer_requires_the_scaffolds_replies() -> None:
    with pytest.raises(ValueError, match="multi_turn_chat"):
        _response(_instance(), None)


def test_turn_accuracy_averages_the_four_ifeval_aggregations() -> None:
    # Turn 3: one conversation passes all three instructions, the other passes
    # two of three, so prompt-level is 1/2 and instruction-level 5/6 under
    # both strict and loose scoring.
    responses = [
        _response(_instance(), FOLLOWS_ALL),
        _response(_instance(), FOLLOWS_ALL[:2] + ["Rain on the roof. Done."]),
    ]
    metric = MultiIFAccuracy(name="turn_3_accuracy", turn=3)
    assert metric.compute(responses) == pytest.approx((1 / 2 + 5 / 6) / 2)


def test_accuracy_averages_turns_over_the_conversations_reaching_each() -> None:
    responses = [
        _response(_instance(), FOLLOWS_ALL),
        _response(_instance(turns=2), ["Rain, on the roof", "Rain on the roof. Done."]),
    ]
    # Turn 1: 1/2 of prompts and instructions; turn 2: all pass; turn 3: only
    # the first conversation reaches it, and it passes.
    assert MultiIFAccuracy().compute(responses) == pytest.approx((0.5 + 1.0 + 1.0) / 3)


def test_language_accuracy_keeps_only_that_language() -> None:
    responses = [
        _response(_instance(language="English"), FOLLOWS_ALL),
        _response(_instance(language="French", key="1:1:fr"), ["Rain, roof", "x", "y"]),
    ]
    assert MultiIFAccuracy(name="english_accuracy", language="English").compute(responses) == 1.0
    assert MultiIFAccuracy(name="french_accuracy", language="French").compute(responses) < 0.5


def test_languages_are_weighted_equally() -> None:
    # English passes one of two conversations on turn 1 and French passes its
    # only one, so the languages score 0.5 and 1.0. Pooling would give 2/3.
    responses = [
        _response(_instance(language="English"), FOLLOWS_ALL),
        _response(_instance(language="English"), ["Rain, on the roof", "x", "y"]),
        _response(_instance(language="French", key="1:1:fr"), FOLLOWS_ALL),
    ]
    assert MultiIFAccuracy(name="turn_1_accuracy", turn=1).compute(responses) == 0.75


def test_unscored_conversations_count_as_failures() -> None:
    instance = _instance()
    unscored = Response(
        instance=instance, request=get_task("multi_if").format_request(instance), outputs=[]
    )
    responses = [_response(_instance(), FOLLOWS_ALL), unscored]
    assert MultiIFAccuracy(name="turn_1_accuracy", turn=1).compute(responses) == 0.5


def test_metric_has_no_per_instance_value() -> None:
    response = _response(_instance(), FOLLOWS_ALL)
    assert MultiIFAccuracy().compute_instance(response) is None
