"""Tests for BFCL multi-turn execution and grading."""

from __future__ import annotations

import pytest

from olmo_eval.common.scorers.bfcl.multi_turn import (
    Call,
    build_instances,
    calls_from_decoded,
    execute_calls,
    is_empty_execute_response,
    multi_turn_checker,
    parse_call_strings,
    state_checker,
)
from olmo_eval.common.scorers.bfcl.multi_turn.api import API_CLASSES, STATELESS_CLASSES

FS_CONFIG = {
    "GorillaFileSystem": {
        "root": {
            "workspace": {
                "type": "directory",
                "contents": {"notes.txt": {"type": "file", "content": "budget analysis"}},
            }
        }
    }
}


def fs_instances() -> dict:
    return build_instances(FS_CONFIG, ["GorillaFileSystem"])


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def test_every_involved_class_is_known() -> None:
    assert set(API_CLASSES) == {
        "GorillaFileSystem",
        "MathAPI",
        "MessageAPI",
        "TwitterAPI",
        "TicketAPI",
        "TradingBot",
        "TravelAPI",
        "VehicleControlAPI",
    }
    assert {"MathAPI"} == STATELESS_CLASSES


def test_a_scenario_is_loaded_into_the_instance() -> None:
    instances = fs_instances()

    assert execute_calls([Call("ls")], instances) == [
        '{"current_directory_content": ["notes.txt"]}'
    ]


def test_calls_see_the_state_earlier_calls_left() -> None:
    instances = fs_instances()

    execute_calls([Call("mkdir", kwargs={"dir_name": "temp"})], instances)
    execute_calls([Call("cd", kwargs={"folder": "temp"})], instances)
    results = execute_calls([Call("ls")], instances)

    assert results == ['{"current_directory_content": []}']


def test_two_rollouts_do_not_share_state() -> None:
    first, second = fs_instances(), fs_instances()

    execute_calls([Call("mkdir", kwargs={"dir_name": "temp"})], first)

    assert "temp" in execute_calls([Call("ls")], first)[0]
    assert "temp" not in execute_calls([Call("ls")], second)[0]


def test_a_positional_argument_reaches_the_method() -> None:
    # Ground truth paths write some arguments positionally.
    calls = parse_call_strings(["sort('notes.txt')"])

    assert calls == [Call("sort", args=("notes.txt",))]
    assert "Error" not in execute_calls(calls, fs_instances())[0]


def test_an_unknown_function_is_recorded_not_raised() -> None:
    results = execute_calls([Call("launch_missiles")], fs_instances())

    assert len(results) == 1
    assert "not available" in results[0]


def test_a_call_that_raises_does_not_stop_the_ones_after_it() -> None:
    # A missing required argument raises. The classes report their own domain
    # failures as a result instead, which is not an execution error.
    results = execute_calls([Call("cd"), Call("ls")], fs_instances())

    assert results[0].startswith("Error during execution")
    assert "notes.txt" in results[1]


def test_a_domain_failure_is_returned_as_the_calls_result() -> None:
    results = execute_calls([Call("cd", kwargs={"folder": "nope"})], fs_instances())

    assert not results[0].startswith("Error during execution")
    assert "No such directory" in results[0]


def test_an_unparseable_ground_truth_call_becomes_one_failed_call() -> None:
    # A few ground truth entries carry an apostrophe inside a quoted argument.
    calls = parse_call_strings(["send_message('receiver_id='USR005',message='hi')"])
    results = execute_calls(calls, fs_instances())

    assert len(results) == 1
    assert "could not parse" in results[0]


def test_a_decoded_reply_becomes_calls() -> None:
    assert calls_from_decoded([{"cd": {"folder": "temp"}}]) == [
        Call("cd", kwargs={"folder": "temp"})
    ]


def test_emptiness_of_a_turn() -> None:
    assert is_empty_execute_response([])
    assert is_empty_execute_response([[]])
    assert not is_empty_execute_response([Call("ls")])


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------

PATH = [["mkdir(dir_name='temp')", "cd(folder='temp')"], ["touch(file_name='a.txt')"]]


def check(model_calls_per_turn: list[list[list[Call]]], path: list[list[str]] = PATH) -> dict:
    return multi_turn_checker(model_calls_per_turn, path, FS_CONFIG, ["GorillaFileSystem"])


def perfect(path: list[list[str]] = PATH) -> list[list[list[Call]]]:
    return [[parse_call_strings(turn)] if turn else [] for turn in path]


def test_a_rollout_matching_the_ground_truth_passes() -> None:
    assert check(perfect())["valid"]


def test_the_same_calls_split_across_steps_still_pass() -> None:
    # The model may take several steps within a turn to get there.
    split = [
        [parse_call_strings(["mkdir(dir_name='temp')"]), parse_call_strings(["cd(folder='temp')"])],
        [parse_call_strings(["touch(file_name='a.txt')"])],
    ]

    assert check(split)["valid"]


def test_a_turn_with_no_calls_fails_when_one_was_expected() -> None:
    result = check([[], [parse_call_strings(["touch(file_name='a.txt')"])]])

    assert not result["valid"]
    assert result["error_type"] == "multi_turn:empty_turn_model_response"


def test_leaving_the_wrong_state_fails() -> None:
    wrong = perfect()
    wrong[1] = [parse_call_strings(["touch(file_name='b.txt')"])]
    result = check(wrong)

    assert not result["valid"]
    assert result["error_type"] == "multi_turn:instance_state_mismatch"


def test_calling_on_a_turn_that_expects_nothing_is_not_graded() -> None:
    # The reference implementation defines an irrelevance check for these turns
    # but never runs it, so a call there is not penalised.
    path = [["mkdir(dir_name='temp')"], []]
    model = [parse_call_strings(["mkdir(dir_name='temp')"])], [parse_call_strings(["ls()"])]

    assert check([list(model[0]), list(model[1])], path)["valid"]


def test_a_call_ahead_of_its_turn_still_passes() -> None:
    # The reviewer's reproduction against the reference: the model adds on both
    # turns although the ground truth only adds on the second.
    path = [[], ["add(a=1, b=2)"]]
    rollout = [[parse_call_strings(["add(a=1, b=2)"])], [parse_call_strings(["add(a=1, b=2)"])]]

    assert multi_turn_checker(rollout, path, {}, ["MathAPI"])["valid"]


def test_a_call_on_an_ungraded_turn_still_counts_toward_the_next_turns_state() -> None:
    path = [[], ["cd(folder='temp')"]]
    # The ungraded turn's mkdir is still executed, so at the graded turn the
    # model's file system has a folder the ground truth's does not.
    early = [
        [parse_call_strings(["mkdir(dir_name='temp')"])],
        [parse_call_strings(["cd(folder='temp')"])],
    ]

    assert not check(early, path)["valid"]


def test_declining_on_a_turn_that_expects_nothing_passes() -> None:
    path = [["mkdir(dir_name='temp')"], []]

    assert check([[parse_call_strings(["mkdir(dir_name='temp')"])], []], path)["valid"]


def test_state_comparison_ignores_private_attributes() -> None:
    a, b = fs_instances(), fs_instances()
    a["GorillaFileSystem"]._private_marker = object()

    assert state_checker(a, b)["valid"]


@pytest.mark.parametrize("class_name", sorted(API_CLASSES))
def test_every_class_loads_a_scenario_and_exposes_methods(class_name: str) -> None:
    instances = build_instances({}, [class_name])
    instance = instances[class_name]
    public = [name for name in dir(instance) if not name.startswith("_")]

    assert public, f"{class_name} exposes no callable methods"
