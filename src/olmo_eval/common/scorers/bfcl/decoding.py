"""Turn a model's reply into the list of calls the BFCL checker compares.

Whatever regime produced the reply — native tool calls from an OpenAI-style
endpoint, or the ``[func(arg=value)]`` text BFCL asks a prompted model for —
decoding ends at the same shape::

    [{"function_name": {"param": value, ...}}, ...]

A reply that cannot be read as calls raises :class:`DecodeError`, which the
relevance categories treat as the model declining to call anything.
"""

from __future__ import annotations

import ast
import json
import operator
import re
from typing import Any

from .constants import Language

#: Largest exponent evaluated while folding an arithmetic argument, so that a
#: reply containing ``10**10**10`` cannot stall scoring.
_MAX_POW_EXPONENT = 1000

_BINARY_OPERATORS: dict[type[ast.operator], Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

#: Tool-call objects a lenient decode will accept, as (name key, arguments key).
_JSON_CALL_KEYS: tuple[tuple[str, str], ...] = (
    ("name", "arguments"),
    ("name", "parameters"),
    ("function", "parameters"),
    ("function", "arguments"),
)

_TOOL_CALL_TAG_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_FENCE_RE = re.compile(r"```(?:json|python)?\s*(.*?)```", re.DOTALL)


class DecodeError(ValueError):
    """Raised when a reply cannot be read as a list of function calls."""


def _fold_arithmetic(node: ast.AST) -> int | float | complex:
    """Evaluate an arithmetic expression over numeric literals.

    The reference implementation calls ``eval`` on the unparsed expression,
    which would run whatever a model wrote. Folding the operators here covers
    the arithmetic that actually appears in replies without executing
    model-authored code.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float, complex)):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        operand = _fold_arithmetic(node.operand)
        return operand if isinstance(node.op, ast.UAdd) else -operand
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        left = _fold_arithmetic(node.left)
        right = _fold_arithmetic(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_POW_EXPONENT:
            raise ValueError("Exponent is too large to evaluate.")
        return _BINARY_OPERATORS[type(node.op)](left, right)
    raise ValueError(f"Not a numeric expression: {ast.dump(node)}")


def resolve_ast_by_type(value: ast.AST) -> Any:
    """Convert one argument's syntax node to the value the checker compares."""
    if isinstance(value, ast.Constant):
        return "..." if value.value is Ellipsis else value.value
    if isinstance(value, ast.UnaryOp):
        # Mirrors the reference, which negates the operand whatever the
        # operator is; in practice this node is a negative number literal.
        return -value.operand.value  # ty: ignore[unresolved-attribute]
    if isinstance(value, ast.List):
        return [resolve_ast_by_type(element) for element in value.elts]
    if isinstance(value, ast.Dict):
        return {
            resolve_ast_by_type(key): resolve_ast_by_type(item)
            for key, item in zip(value.keys, value.values, strict=False)
            if key is not None
        }
    if isinstance(value, ast.BinOp):
        try:
            return _fold_arithmetic(value)
        except (ValueError, ArithmeticError, TypeError):
            return ast.unparse(value)
    if isinstance(value, ast.Name):
        return value.id
    if isinstance(value, ast.Call):
        return ast.unparse(value) if not value.keywords else resolve_ast_call(value)
    if isinstance(value, ast.Tuple):
        return tuple(resolve_ast_by_type(element) for element in value.elts)
    if isinstance(value, (ast.Lambda, ast.Subscript)):
        return ast.unparse(value)
    raise DecodeError(f"Unsupported AST type: {type(value)}")


def call_name(node: ast.Call) -> str:
    """The dotted name a call node calls, e.g. ``math.factorial``."""
    parts: list[str] = []
    func: ast.expr = node.func
    while isinstance(func, ast.Attribute):
        parts.append(func.attr)
        func = func.value
    if isinstance(func, ast.Name):
        parts.append(func.id)
    return ".".join(reversed(parts))


def resolve_ast_call(elem: ast.Call) -> dict[str, dict[str, Any]]:
    """Convert one call node to ``{name: {param: value}}``."""
    args = {
        keyword.arg: resolve_ast_by_type(keyword.value)
        for keyword in elem.keywords
        if keyword.arg is not None
    }
    return {call_name(elem): args}


def parse_python_calls(input_str: str) -> list[dict[str, Any]]:
    """Parse ``[func(arg=value), ...]`` written in Python syntax."""
    parsed = ast.parse(input_str.strip().strip("'"), mode="eval")
    if isinstance(parsed.body, ast.Call):
        return [resolve_ast_call(parsed.body)]
    elements = getattr(parsed.body, "elts", None)
    if elements is None:
        raise DecodeError("The reply is not a call or a list of calls.")
    calls = []
    for element in elements:
        if not isinstance(element, ast.Call):
            raise DecodeError("A list element is not a function call.")
        calls.append(resolve_ast_call(element))
    return calls


def parse_calls(input_str: str, language: Language = Language.PYTHON) -> list[dict[str, Any]]:
    """Parse a bracketed list of calls written in ``language``."""
    if language is Language.PYTHON:
        return parse_python_calls(input_str)
    # The Java and JavaScript grammars parse a bare call, so the brackets the
    # prompt asks for are removed first.
    from .java_js import parse_java_function_call, parse_javascript_function_call

    if language is Language.JAVA:
        return parse_java_function_call(input_str[1:-1])
    return parse_javascript_function_call(input_str[1:-1])


def decode_text(text: str, language: Language = Language.PYTHON) -> list[dict[str, Any]]:
    """Decode a reply written in the ``[func(arg=value)]`` format BFCL asks for.

    Brackets are supplied when the reply omits them, matching the reference
    implementation, so a model that answers with a single bare call is not
    marked wrong for the missing list.
    """
    result = (text or "").strip("`\n ")
    if not result.startswith("["):
        result = "[" + result
    if not result.endswith("]"):
        result = result + "]"
    try:
        return parse_calls(result, language)
    except (DecodeError, ImportError):
        # A missing grammar is a setup problem, not a reply the model got
        # wrong, so it is left to surface as itself.
        raise
    except Exception as exc:
        raise DecodeError(str(exc)) from exc


def _calls_from_json_payload(payload: Any) -> list[dict[str, Any]] | None:
    """Read a JSON tool-call list, in any of the shapes models emit."""
    items = payload if isinstance(payload, list) else [payload]
    calls: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            return None
        for name_key, args_key in _JSON_CALL_KEYS:
            name = item.get(name_key)
            if not isinstance(name, str):
                continue
            arguments = item.get(args_key, {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    return None
            if not isinstance(arguments, dict):
                return None
            calls.append({name: arguments})
            break
        else:
            return None
    return calls


def _json_candidates(text: str) -> list[str]:
    """Yield the substrings of a reply that might hold a JSON tool call."""
    candidates = [match.group(1) for match in _TOOL_CALL_TAG_RE.finditer(text)]
    candidates.extend(match.group(1) for match in _FENCE_RE.finditer(text))
    array = re.search(r"\[.*\]", text, re.DOTALL)
    if array:
        candidates.append(array.group(0))
    obj = re.search(r"\{.*\}", text, re.DOTALL)
    if obj:
        candidates.append(obj.group(0))
    candidates.append(text)
    return candidates


def decode_text_lenient(text: str, language: Language = Language.PYTHON) -> list[dict[str, Any]]:
    """Decode a reply, also accepting the JSON tool-call formats models emit.

    A base model has never been taught BFCL's ``[func(arg=value)]`` format, and
    few-shot examples do not always override a format it saw often during
    pretraining. Accepting a JSON tool-call list as well keeps the measurement
    on whether the right function was chosen with the right arguments rather
    than on which surface form the model reached for.
    """
    try:
        return decode_text(text, language)
    except DecodeError as exc:
        first_error = exc

    for candidate in _json_candidates(text or ""):
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        calls = _calls_from_json_payload(payload)
        if calls is not None:
            return calls

    raise first_error


def decode_tool_calls(
    tool_calls: list[Any] | None,
    name_map: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Decode native tool calls returned by an OpenAI-compatible endpoint.

    ``name_map`` restores the function names as the dataset spells them, since
    the schemas sent to the endpoint use names an OpenAI tool name must match.
    """
    calls: list[dict[str, Any]] = []
    for tool_call in tool_calls or []:
        name = tool_call.function.name or ""
        raw_arguments = tool_call.function.arguments or "{}"
        try:
            arguments = (
                json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
            )
        except json.JSONDecodeError as exc:
            raise DecodeError(f"Arguments for {name!r} are not valid JSON: {exc}") from exc
        if not isinstance(arguments, dict):
            raise DecodeError(f"Arguments for {name!r} are not an object.")
        calls.append({(name_map or {}).get(name, name): arguments})
    return calls


def is_function_calling_format(decoded: Any) -> bool:
    """Whether a decoded reply has the shape the checker expects.

    An empty list passes: it is a well-formed reply that called nothing.
    """
    if not isinstance(decoded, list):
        return False
    for item in decoded:
        if not isinstance(item, dict) or len(item) != 1:
            return False
        if not isinstance(next(iter(item.values())), dict):
            return False
    return True


def is_empty_output(decoded: Any) -> bool:
    """Whether a decoded reply amounts to no function call at all."""
    if not is_function_calling_format(decoded):
        return True
    if not decoded:
        return True
    return len(decoded) == 1 and not decoded[0]
