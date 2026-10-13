"""Tests for recognising a prompt too long for the model's context window."""

from __future__ import annotations

import pytest

from olmo_eval.common.types import LMOutput
from olmo_eval.inference.errors import (
    CONTEXT_OVERFLOW_KEY,
    REQUEST_ERROR_KEY,
    first_output,
    is_context_overflow,
    is_context_overflow_output,
)

VLLM_MESSAGE = (
    "Error code: 400 - {'error': {'message': \"This model's maximum context length is "
    "40960 tokens. However, you requested 1000 output tokens and your prompt contains "
    'at least 39961 input tokens"}}'
)


class CodedError(Exception):
    def __init__(self, code: str) -> None:
        super().__init__("request rejected")
        self.code = code


def test_the_vllm_overflow_message_is_recognised() -> None:
    assert is_context_overflow(ValueError(VLLM_MESSAGE))


def test_the_openai_overflow_code_is_recognised() -> None:
    assert is_context_overflow(CodedError("context_length_exceeded"))


def test_an_overflow_wrapped_by_another_error_is_recognised() -> None:
    wrapped = RuntimeError("provider call failed")
    wrapped.__cause__ = ValueError(VLLM_MESSAGE)

    assert is_context_overflow(wrapped)


def test_other_bad_requests_are_not_overflow() -> None:
    assert not is_context_overflow(ValueError("Error code: 400 - invalid tool schema"))
    assert not is_context_overflow(CodedError("invalid_request_error"))


def test_a_marked_reply_reads_as_overflow() -> None:
    marked = LMOutput(text="", metadata={REQUEST_ERROR_KEY: "x", CONTEXT_OVERFLOW_KEY: True})

    assert is_context_overflow_output([marked])
    assert not is_context_overflow_output([LMOutput(text="", metadata={REQUEST_ERROR_KEY: "x"})])
    assert not is_context_overflow_output([])


def test_first_output_returns_a_served_reply() -> None:
    assert first_output([LMOutput(text="ok")]).text == "ok"


def test_first_output_raises_on_a_marked_failure() -> None:
    marked = LMOutput(text="", metadata={REQUEST_ERROR_KEY: "context overflow: too long"})

    with pytest.raises(RuntimeError, match="context overflow"):
        first_output([marked])


def test_first_output_raises_when_nothing_came_back() -> None:
    with pytest.raises(RuntimeError, match="no output"):
        first_output([])
