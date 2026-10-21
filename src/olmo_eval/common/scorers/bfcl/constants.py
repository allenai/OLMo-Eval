"""Type tables and category names shared by the BFCL decoder and checker.

The tables mirror the reference implementation's so that a call the reference
harness accepts is accepted here too. Gorilla's function documents use a type
vocabulary of their own (``dict``, ``float``, ``ArrayList``, ...) that has to be
mapped twice: once onto JSON Schema, for the schemas a model is shown, and once
onto Python, for the values the checker compares.

Reference: https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard
"""

from __future__ import annotations

from enum import StrEnum


class Language(StrEnum):
    """Source language a category's function calls are written in."""

    PYTHON = "python"
    JAVA = "java"
    JAVASCRIPT = "javascript"


#: Categories scored only on whether the model called a function at all, with
#: no possible answers to compare against.
RELEVANCE_CATEGORIES: tuple[str, ...] = (
    "irrelevance",
    "live_irrelevance",
    "live_relevance",
)

#: Categories where calling no function is the correct behavior.
IRRELEVANCE_CATEGORIES: tuple[str, ...] = ("irrelevance", "live_irrelevance")


def language_for_category(test_category: str) -> Language:
    """Return the language a category's calls are written in."""
    if test_category == "java":
        return Language.JAVA
    if test_category == "javascript":
        return Language.JAVASCRIPT
    return Language.PYTHON


#: Gorilla's type names mapped onto the JSON Schema types a tool schema may use.
GORILLA_TO_OPENAPI: dict[str, str] = {
    "integer": "integer",
    "number": "number",
    "float": "number",
    "string": "string",
    "boolean": "boolean",
    "bool": "boolean",
    "array": "array",
    "list": "array",
    "dict": "object",
    "object": "object",
    "tuple": "array",
    "any": "string",
    "byte": "integer",
    "short": "integer",
    "long": "integer",
    "double": "number",
    "char": "string",
    "ArrayList": "array",
    "Array": "array",
    "HashMap": "object",
    "Hashtable": "object",
    "Queue": "array",
    "Stack": "array",
    "Any": "string",
    "String": "string",
    "Bigint": "integer",
}

#: Gorilla's type names mapped onto the Python types the checker compares with.
PYTHON_TYPE_MAPPING: dict[str, type] = {
    "string": str,
    "integer": int,
    "float": float,
    "boolean": bool,
    "array": list,
    "tuple": list,
    "dict": dict,
    "any": str,
}

#: Python types whose element values are checked one level deep.
PYTHON_NESTED_TYPE_CHECK_LIST: tuple[str, ...] = ("array", "tuple")

#: Types whose declared element type drives the value conversion.
NESTED_CONVERSION_TYPE_LIST: tuple[str, ...] = ("Array", "ArrayList", "array")

JAVA_TYPE_CONVERSION: dict[str, type] = {
    "byte": int,
    "short": int,
    "integer": int,
    "float": float,
    "double": float,
    "long": int,
    "boolean": bool,
    "char": str,
    "Array": list,
    "ArrayList": list,
    "Set": set,
    "HashMap": dict,
    "Hashtable": dict,
    # A Queue could be checked against queue.Queue; a list is close enough.
    "Queue": list,
    "Stack": list,
    "String": str,
    "any": str,
}

JS_TYPE_CONVERSION: dict[str, type] = {
    "String": str,
    "integer": int,
    "float": float,
    "Bigint": int,
    "Boolean": bool,
    "dict": dict,
    "array": list,
    "any": str,
}
