"""Grade a BFCL multi-turn rollout against its ground truth path.

A turn is right when the model left the involved instances in the state the
ground truth path leaves them in, and when everything the ground truth's calls
returned also came back from the model's. Nothing the model said is read.

The response comparison is unordered and cumulative: many turns can be carried
out in more than one order, and a model that fetched something in an earlier
turn need not fetch it again.
"""

from __future__ import annotations

from typing import Any

from .execution import (
    Call,
    build_instances,
    execute_calls,
    is_empty_execute_response,
    parse_call_strings,
)

CheckResult = dict[str, Any]

#: Rollout metadata recording that the conversation outgrew the model's context
#: window before it finished. The reference implementation records the failure
#: as the model's answer and grades the entry wrong, so it scores zero here too.
CONTEXT_OVERFLOW_METADATA_KEY = "bfcl_context_overflow"
CONTEXT_OVERFLOW_ERROR_TYPE = "multi_turn:context_overflow"

#: Rollout metadata recording that a turn ran out of steps. The rollout stops
#: there, and the reference implementation fails such an entry outright when it
#: was stopped before its last turn.
STEP_BUDGET_METADATA_KEY = "bfcl_step_budget_exhausted"
FORCE_TERMINATED_ERROR_TYPE = "multi_turn:force_terminated"


def _fail(message: str, error_type: str, **details: Any) -> CheckResult:
    return {"valid": False, "error_message": message, "error_type": error_type, "details": details}


def compare_instances(model_instance: Any, truth_instance: Any) -> tuple[bool, dict[str, Any]]:
    """Compare the public attributes of two instances of the same class."""
    differences: dict[str, Any] = {}
    for attr_name in vars(truth_instance):
        if attr_name.startswith("_"):
            continue
        model_attr = getattr(model_instance, attr_name, None)
        truth_attr = getattr(truth_instance, attr_name)
        if model_attr != truth_attr:
            differences[attr_name] = {"model": model_attr, "ground_truth": truth_attr}
    return not differences, differences


def state_checker(model_instances: dict[str, Any], truth_instances: dict[str, Any]) -> CheckResult:
    """Check that every involved instance ended the turn in the expected state."""
    for class_name, truth_instance in truth_instances.items():
        model_instance = model_instances.get(class_name)
        if model_instance is None:
            return _fail(
                f"No model instance for {class_name}.", "multi_turn:instance_state_mismatch"
            )
        valid, differences = compare_instances(model_instance, truth_instance)
        if not valid:
            return _fail(
                f"Model instance for {class_name} does not match the ground truth state.",
                "multi_turn:instance_state_mismatch",
                differences=differences,
            )
    return {"valid": True}


def missing_responses(expected: list[str], produced: list[str]) -> list[str]:
    """Return the expected results that the model's results do not cover.

    Duplicates count, so a result the ground truth produced twice has to appear
    twice among the model's.
    """
    remaining = list(produced)
    missing: list[str] = []
    for item in expected:
        try:
            remaining.remove(item)
        except ValueError:
            missing.append(item)
    return missing


def response_checker(
    model_responses: list[str], truth_responses: list[str], turn_index: int
) -> CheckResult:
    """Check the model's results so far cover this turn's expected results."""
    missing = missing_responses(truth_responses, model_responses)
    if missing:
        return _fail(
            f"Model execution results do not cover the ground truth results for turn {turn_index}.",
            "multi_turn:execution_response_mismatch",
            missing_items=missing,
        )
    return {"valid": True}


def multi_turn_checker(
    model_calls_per_turn: list[list[list[Call]]],
    truth_path: list[list[str]],
    initial_config: dict[str, Any],
    involved_classes: list[str],
    long_context: bool = False,
) -> CheckResult:
    """Grade a whole rollout, turn by turn, stopping at the first failure.

    The model's calls are replayed against instances of their own so that the
    state they build up is compared with, not shared with, the ground truth's.
    """
    model_instances = build_instances(initial_config, involved_classes, long_context)
    truth_instances = build_instances(initial_config, involved_classes, long_context)
    model_results_so_far: list[str] = []

    for turn_index, truth_turn in enumerate(truth_path):
        steps = model_calls_per_turn[turn_index] if turn_index < len(model_calls_per_turn) else []
        for step in steps:
            model_results_so_far.extend(execute_calls(step, model_instances))

        truth_results = execute_calls(parse_call_strings(truth_turn), truth_instances)

        if not truth_turn:
            # A turn the ground truth leaves alone is not graded, as in the
            # reference implementation. Its calls were still executed above, so
            # that the state compared at the next turn includes them.
            continue

        if is_empty_execute_response([call for step in steps for call in step]):
            return _fail(
                f"Model made no call on turn {turn_index}, where the ground truth makes one.",
                "multi_turn:empty_turn_model_response",
                turn=turn_index,
            )

        state_result = state_checker(model_instances, truth_instances)
        if not state_result["valid"]:
            state_result["details"]["turn"] = turn_index
            return state_result

        response_result = response_checker(model_results_so_far, truth_results, turn_index)
        if not response_result["valid"]:
            return response_result

    return {"valid": True}
