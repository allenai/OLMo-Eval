"""Run BFCL multi-turn calls against the stateful API instances.

A turn is graded on what the calls did, so every call has to actually run. The
instances live for the length of one rollout: a call in a later turn sees the
state earlier calls left behind, which is most of what these categories
measure.

The reference implementation renders each call back to source and ``eval``s it
with the instances in ``globals()``, guarded by a list of function names it
refuses. Calls arrive here already decoded, as a name and its arguments, so
they are dispatched by looking the method up on the instance that defines it.
A model can then only reach methods the involved classes actually have, and
nothing it writes is executed as code.
"""

from __future__ import annotations

import ast
import inspect
import json
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from ..decoding import call_name, resolve_ast_by_type
from .api import API_CLASSES, STATELESS_CLASSES


@dataclass(frozen=True, slots=True)
class Call:
    """One call to make against the instances.

    Arguments are kept as written. A ground truth path may pass one
    positionally, while a decoded model reply carries only named ones, so both
    forms have to survive as far as the method that receives them.
    """

    name: str
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)


def parse_call_strings(call_strings: list[str]) -> list[Call | str]:
    """Parse the call strings a test entry's ground truth path is written in.

    A string that will not parse is returned as itself, so that one unreadable
    call is recorded as a failed call rather than losing the whole path. A few
    ground truth entries carry an apostrophe inside a single-quoted argument
    and do not parse as Python at all; the reference implementation reaches the
    same outcome by letting its ``eval`` raise.
    """
    calls: list[Call | str] = []
    for call_string in call_strings:
        try:
            parsed = ast.parse(call_string.strip(), mode="eval").body
        except SyntaxError:
            calls.append(call_string)
            continue
        if not isinstance(parsed, ast.Call):
            calls.append(call_string)
            continue
        calls.append(
            Call(
                name=call_name(parsed),
                args=tuple(resolve_ast_by_type(arg) for arg in parsed.args),
                kwargs={
                    keyword.arg: resolve_ast_by_type(keyword.value)
                    for keyword in parsed.keywords
                    if keyword.arg is not None
                },
            )
        )
    return calls


def render_call(call: Call) -> str:
    """Write a call out as the reference implementation does, e.g. ``cd(folder='temp')``."""
    arguments = [repr(arg) for arg in call.args]
    arguments += [f"{name}={value!r}" for name, value in call.kwargs.items()]
    return f"{call.name}({','.join(arguments)})"


def calls_from_decoded(decoded: list[dict[str, Any]]) -> list[Call]:
    """Convert a decoded model reply into calls.

    Decoding keeps only named arguments, which is what the benchmark asks a
    model to write, so nothing is lost here that a reply should have carried.
    """
    calls: list[Call] = []
    for item in decoded or []:
        if not isinstance(item, dict) or len(item) != 1:
            raise ValueError(f"Malformed call: {item!r}")
        name, arguments = next(iter(item.items()))
        if not isinstance(arguments, dict):
            raise ValueError(f"Arguments for {name!r} are not named.")
        calls.append(Call(name=name, kwargs=dict(arguments)))
    return calls


class UnknownFunctionError(LookupError):
    """Raised when a call names no method on any involved instance."""


def build_instances(
    initial_config: dict[str, Any],
    involved_classes: list[str],
    long_context: bool = False,
) -> dict[str, Any]:
    """Create the instances one rollout acts on, loaded with its scenario.

    The configuration is copied per instance, so the model's rollout and the
    ground truth's cannot see each other's state.
    """
    instances: dict[str, Any] = {}
    for class_name in involved_classes:
        try:
            class_ = API_CLASSES[class_name]
        except KeyError as exc:
            raise LookupError(f"No API class named {class_name!r}.") from exc
        instance = class_()
        if class_name not in STATELESS_CLASSES:
            instance._load_scenario(
                deepcopy(initial_config.get(class_name, {})), long_context=long_context
            )
        instances[class_name] = instance
    return instances


def build_method_table(instances: dict[str, Any]) -> dict[str, Any]:
    """Map every public method of every instance to the bound method.

    BFCL's calls name a method without saying which class it belongs to, so the
    involved classes share one namespace, as they do in the prompt.
    """
    methods: dict[str, Any] = {}
    for instance in instances.values():
        for name, method in inspect.getmembers(instance, predicate=inspect.ismethod):
            if not name.startswith("_"):
                methods.setdefault(name, method)
    return methods


def _format_result(result: Any) -> str:
    """Render a return value the way the response check expects to read it."""
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        try:
            return json.dumps(result)
        except (TypeError, ValueError):
            return str(result)
    return str(result)


def execute_calls(calls: Sequence[Call | str], instances: dict[str, Any]) -> list[str]:
    """Run calls in order against ``instances``, returning their results.

    A call that raises is recorded as its error rather than stopping the
    rollout, because the model is allowed to recover from one in a later step.
    """
    methods = build_method_table(instances)
    results: list[str] = []

    for call in calls:
        if isinstance(call, str):
            results.append(f"Error during execution: could not parse {call[:80]!r}")
            continue
        try:
            # A dotted name is written for the reader's benefit; the method is
            # found by its own name, as it is in the prompt.
            method = methods.get(call.name) or methods.get(call.name.rsplit(".", 1)[-1])
            if method is None:
                raise UnknownFunctionError(f"Function {call.name!r} is not available.")
            results.append(_format_result(method(*call.args, **call.kwargs)))
        except Exception as exc:
            results.append(f"Error during execution: {exc}")

    return results


def is_empty_execute_response(calls: list[Any]) -> bool:
    """Whether a turn's steps amount to no call at all."""
    if not calls:
        return True
    return len(calls) == 1 and not calls[0]
