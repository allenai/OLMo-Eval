"""Sampling overrides must remain usable as async batching keys."""

from dataclasses import replace

import pytest

from olmo_eval.cli.run.config import _parse_override_value
from olmo_eval.common.types import RequestType, SamplingParams


@pytest.mark.parametrize("encoded,expected", [("[]", ()), ('["END", "STOP"]', ("END", "STOP"))])
def test_cli_stop_sequences_group_with_equivalent_tuple(encoded, expected):
    original = SamplingParams(stop_sequences=("Question:", "\n\n"))
    parsed = _parse_override_value(encoded)
    overridden = replace(original, stop_sequences=parsed)
    equivalent = SamplingParams(stop_sequences=expected)

    batches = {(RequestType.COMPLETION, overridden): ["first"]}
    batches[(RequestType.COMPLETION, equivalent)].append("second")
    assert len(batches) == 1
    assert overridden.stop_sequences == expected
    assert original.stop_sequences == ("Question:", "\n\n")
    parsed.append("LATER")
    assert overridden.stop_sequences == expected


@pytest.mark.parametrize("stops", [None, (), ("END",)])
def test_existing_immutable_stop_sequences_are_preserved(stops):
    params = SamplingParams(stop_sequences=stops)
    assert params.stop_sequences is stops
    assert {params: "batch"}[replace(params)] == "batch"
