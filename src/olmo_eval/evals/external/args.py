"""Parsing of external-eval arguments, which arrive as strings from the command line."""

from __future__ import annotations

from typing import Any

#: Values that mean "not set" for an optional argument.
_UNSET = (None, "", "none", "None")


def as_bool(value: Any) -> bool:
    """Read a flag given as a bool, a 0/1 or a string such as ``true``."""
    return value in (True, "true", "True", "1", 1)


def as_optional_int(value: Any) -> int | None:
    """Read an integer, or None when the argument is unset."""
    return None if value in _UNSET else int(value)


def as_optional_float(value: Any) -> float | None:
    """Read a number, or None when the argument is unset."""
    return None if value in _UNSET else float(value)


def as_list(value: Any) -> list[str] | None:
    """Read a comma-separated string or a sequence as a list, or None when empty."""
    if value is None:
        return None
    if isinstance(value, str):
        items = [v.strip() for v in value.split(",") if v.strip()]
    else:
        items = [str(v) for v in value]
    return items or None
