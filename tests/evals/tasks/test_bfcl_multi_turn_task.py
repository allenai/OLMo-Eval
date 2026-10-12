"""Tests for the BFCL multi-turn tasks."""

from __future__ import annotations

from typing import Any

import pytest

from olmo_eval.common.scorers.base import get_scorer_result
from olmo_eval.common.types import LMOutput, RequestType
from olmo_eval.evals.suites import get_suite
from olmo_eval.evals.tasks.bfcl.multi_turn import (
    FUNC_DOC_FILES,
    MULTI_TURN_CATEGORIES,
    BFCLMultiTurnScorer,
    BFCLMultiTurnTask,
)
from olmo_eval.evals.tasks.common import get_task

FS_DOCS = [
    {
        "name": "mkdir",
        "description": "Make a directory.",
        "parameters": {
            "type": "dict",
            "properties": {"dir_name": {"type": "string", "description": "Name."}},
            "required": ["dir_name"],
        },
    },
    {
        "name": "sort",
        "description": "Sort a file.",
        "parameters": {
            "type": "dict",
            "properties": {"file_name": {"type": "string", "description": "File."}},
            "required": ["file_name"],
        },
    },
]

ENTRY: dict[str, Any] = {
    "id": "multi_turn_base_0",
    "question": [[{"role": "user", "content": "Make a temp directory."}], []],
    "initial_config": {
        "GorillaFileSystem": {"root": {"ws": {"type": "directory", "contents": {}}}}
    },
    "involved_classes": ["GorillaFileSystem"],
}


def task_with_stubbed_data(spec: str = "bfcl_multi_turn_base") -> BFCLMultiTurnTask:
    """A task whose per-class documents and answers are supplied, not fetched."""
    task = get_task(spec)
    assert isinstance(task, BFCLMultiTurnTask)
    task._func_docs = {"GorillaFileSystem": FS_DOCS}
    task._answers = {"multi_turn_base_0": [["mkdir(dir_name='temp')"], []]}
    return task


def test_every_api_class_has_a_documented_file() -> None:
    from olmo_eval.common.scorers.bfcl.multi_turn.api import API_CLASSES

    assert set(FUNC_DOC_FILES) == set(API_CLASSES)


def test_functions_are_assembled_from_the_classes_an_entry_involves() -> None:
    # A multi-turn entry carries no function documents of its own.
    assert "function" not in ENTRY
    instance = task_with_stubbed_data().process_doc(ENTRY)

    assert instance is not None
    assert {tool.name for tool in instance.tools or ()} == {"mkdir", "sort"}


def test_a_held_back_function_is_kept_out_of_the_visible_tools() -> None:
    entry = {**ENTRY, "missed_function": {"1": ["sort"]}}
    instance = task_with_stubbed_data().process_doc(entry)

    assert instance is not None
    assert {tool.name for tool in instance.tools or ()} == {"mkdir"}
    offered = instance.metadata["missed_function"]["1"]
    assert [schema["function"]["name"] for schema in offered] == ["sort"]


def test_a_held_back_function_is_written_out_the_way_the_prompt_writes_the_rest() -> None:
    entry = {**ENTRY, "missed_function": {"1": ["sort"]}}
    task = task_with_stubbed_data("bfcl_multi_turn_miss_func:prompt")
    instance = task.process_doc(entry)
    assert instance is not None

    request = task.format_request(instance)
    assert request.metadata is not None
    # Read it off the request, because that is what reaches the scaffold.
    offered = request.metadata["missed_function_docs"]["1"]
    system_prompt = request.system_prompt or ""

    # Both describe a function the same way, so the offer does not switch
    # dialects on the model partway through the conversation.
    assert "'name': 'sort'" in offered
    assert "'name': 'mkdir'" in system_prompt
    assert "'type': 'function'" not in offered
    assert "'type': 'function'" not in system_prompt


def test_a_held_back_name_still_maps_back_to_the_dataset_spelling() -> None:
    entry = {**ENTRY, "missed_function": {"1": ["sort"]}}
    instance = task_with_stubbed_data().process_doc(entry)

    assert instance is not None
    assert "sort" in instance.metadata["name_map"].values()


def test_the_request_hands_the_scaffold_the_rollout_payload() -> None:
    task = task_with_stubbed_data()
    instance = task.process_doc(ENTRY)
    assert instance is not None

    request = task.format_request(instance)

    assert request.request_type == RequestType.CHAT
    assert request.metadata is not None
    assert request.metadata["turns"] == instance.metadata["turns"]
    assert request.metadata["involved_classes"] == ["GorillaFileSystem"]
    assert request.metadata["call_source"] == "tool_calls"
    assert request.tools is not None


def test_the_prompt_regime_writes_the_functions_into_a_system_prompt() -> None:
    task = task_with_stubbed_data("bfcl_multi_turn_base:prompt")
    instance = task.process_doc(ENTRY)
    assert instance is not None

    request = task.format_request(instance)

    assert request.system_prompt is not None
    assert "mkdir" in request.system_prompt
    assert request.metadata is not None
    assert request.metadata["call_source"] == "text"


def test_the_prompt_regime_sends_no_tool_schemas() -> None:
    # The functions are in the prompt; sending them again as schemas would
    # give the model them twice and invite calls this regime cannot read.
    task = task_with_stubbed_data("bfcl_multi_turn_base:prompt")
    instance = task.process_doc(ENTRY)
    assert instance is not None

    assert task.format_request(instance).tools is None


def test_the_tool_regime_sends_schemas_and_no_system_prompt() -> None:
    task = task_with_stubbed_data()
    instance = task.process_doc(ENTRY)
    assert instance is not None
    request = task.format_request(instance)

    assert request.tools is not None
    assert request.system_prompt is None


def test_a_rollout_matching_the_ground_truth_scores_one() -> None:
    task = task_with_stubbed_data()
    instance = task.process_doc(ENTRY)
    assert instance is not None
    output = LMOutput(text="")
    output.extracted_answer = [[[{"mkdir": {"dir_name": "temp"}}]], []]

    assert BFCLMultiTurnScorer().score(instance, output) == 1.0


def test_a_rollout_that_outgrew_the_context_window_scores_zero() -> None:
    task = task_with_stubbed_data()
    instance = task.process_doc(ENTRY)
    assert instance is not None
    # Even calls that would have passed count for nothing once the conversation
    # overflowed, as in the reference implementation.
    output = LMOutput(text="", metadata={"bfcl_context_overflow": "context overflow: too long"})
    output.extracted_answer = [[[{"mkdir": {"dir_name": "temp"}}]], []]

    assert BFCLMultiTurnScorer().score(instance, output) == 0.0
    recorded = get_scorer_result(output, "bfcl_multi_turn")
    assert recorded is not None
    assert recorded["error_type"] == "multi_turn:context_overflow"


def test_a_rollout_stopped_before_its_last_turn_scores_zero() -> None:
    # The reviewer's reproduction: the ground truth's remaining turn is empty, so
    # padding it would pass the entry, where the reference fails it outright.
    task = task_with_stubbed_data()
    instance = task.process_doc(ENTRY)
    assert instance is not None
    output = LMOutput(text="", metadata={"bfcl_step_budget_exhausted": True})
    output.extracted_answer = [[[{"mkdir": {"dir_name": "temp"}}]]]

    assert BFCLMultiTurnScorer().score(instance, output) == 0.0
    recorded = get_scorer_result(output, "bfcl_multi_turn")
    assert recorded is not None
    assert recorded["error_type"] == "multi_turn:force_terminated"
    assert recorded["step_budget_exhausted"] is True


def test_a_passing_rollout_is_recorded_for_the_predictions() -> None:
    task = task_with_stubbed_data()
    instance = task.process_doc(ENTRY)
    assert instance is not None
    output = LMOutput(text="")
    output.extracted_answer = [[[{"mkdir": {"dir_name": "temp"}}]], []]

    BFCLMultiTurnScorer().score(instance, output)

    assert get_scorer_result(output, "bfcl_multi_turn") == {
        "valid": True,
        "step_budget_exhausted": False,
    }


def test_calling_on_a_turn_that_expects_nothing_is_not_penalised() -> None:
    # The reference implementation defines an irrelevance check for such a turn
    # but never runs it.
    task = task_with_stubbed_data()
    instance = task.process_doc(ENTRY)
    assert instance is not None
    output = LMOutput(text="")
    output.extracted_answer = [
        [[{"mkdir": {"dir_name": "temp"}}]],
        [[{"mkdir": {"dir_name": "x"}}]],
    ]

    assert BFCLMultiTurnScorer().score(instance, output) == 1.0


def test_a_reply_with_no_rollout_is_refused() -> None:
    # Without the scaffold there is no rollout to grade, and a zero here would
    # read as a weak model rather than a missing harness.
    task = task_with_stubbed_data()
    instance = task.process_doc(ENTRY)
    assert instance is not None

    with pytest.raises(ValueError, match="scaffold=bfcl_multi_turn"):
        BFCLMultiTurnScorer().score(instance, LMOutput(text="some prose"))


def test_every_multi_turn_category_is_registered() -> None:
    for category in MULTI_TURN_CATEGORIES:
        assert get_task(f"bfcl_multi_turn_{category}") is not None
        assert get_task(f"bfcl_multi_turn_{category}:prompt") is not None


def test_composite_is_not_registered() -> None:
    # Its ground truth cannot be executed against any version of the classes.
    from olmo_eval.evals.tasks.common import list_tasks

    assert "bfcl_multi_turn_composite" not in list_tasks()


def test_the_multi_turn_suite_averages_the_four_categories() -> None:
    assert get_suite("bfcl:multi_turn").expand() == tuple(
        f"bfcl_multi_turn_{category}" for category in MULTI_TURN_CATEGORIES
    )


def test_there_is_no_base_regime_for_multi_turn() -> None:
    from olmo_eval.evals.suites import list_suites

    assert "bfcl:multi_turn:base" not in list_suites()


def test_the_multi_turn_rollout_samples_the_way_the_leaderboard_does() -> None:
    params = get_task("bfcl_multi_turn_base").config.sampling_params

    assert params.max_tokens == 4096
    assert params.temperature == 0.001
    assert params.fit_max_tokens_to_context
