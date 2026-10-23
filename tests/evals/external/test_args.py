"""Tests for parsing external-eval arguments."""

from __future__ import annotations

import pytest

from olmo_eval.evals.external.args import as_bool, as_list, as_optional_float, as_optional_int


@pytest.mark.parametrize(
    ("value", "expected"),
    [(True, True), ("true", True), ("1", True), (1, True), ("false", False), (None, False)],
)
def test_as_bool(value, expected) -> None:
    assert as_bool(value) is expected


@pytest.mark.parametrize("value", [None, "", "none", "None"])
def test_unset_values_parse_as_none(value) -> None:
    assert as_optional_int(value) is None
    assert as_optional_float(value) is None


def test_numbers_parse_from_strings() -> None:
    assert as_optional_int("12") == 12
    assert as_optional_float("0.5") == 0.5


def test_as_list_splits_strings_and_keeps_sequences() -> None:
    assert as_list("a, b,,c") == ["a", "b", "c"]
    assert as_list(["x", 2]) == ["x", "2"]
    assert as_list("") is None
    assert as_list(None) is None
