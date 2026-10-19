"""Tests for terminal inference-provider error handling."""

from __future__ import annotations

import asyncio
import logging

import pytest

from olmo_eval.inference.dispatch import dispatch_concurrent, dispatch_concurrent_results
from olmo_eval.inference.errors import (
    TerminalProviderError,
    classify_terminal_provider_error,
)


def _engine_dead_error(message: str = "engine died") -> Exception:
    error_type = type(
        "EngineDeadError",
        (Exception,),
        {"__module__": "vllm.v1.engine.exceptions"},
    )
    return error_type(message)


def test_classifies_only_vllm_engine_death_as_terminal() -> None:
    wrapped = RuntimeError("provider call failed")
    wrapped.__cause__ = _engine_dead_error()

    for error in (_engine_dead_error(), wrapped):
        classified = classify_terminal_provider_error(error)
        assert classified is not None
        assert "EngineDeadError" in str(classified)
    assert classify_terminal_provider_error(RuntimeError("request failed")) is None


def test_dispatch_propagates_terminal_error_and_cancels_siblings() -> None:
    async def run() -> None:
        sibling_started = asyncio.Event()
        sibling_cancelled = asyncio.Event()

        async def process(item: int) -> None:
            if item == 0:
                await sibling_started.wait()
                raise TerminalProviderError("engine died")

            sibling_started.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                sibling_cancelled.set()
                raise

        with pytest.raises(TerminalProviderError, match="engine died"):
            await dispatch_concurrent([0, 1], process, max_in_flight=2, max_retries=3)

        assert sibling_cancelled.is_set()

    asyncio.run(run())


def test_dispatch_logs_one_summary_naming_the_first_non_terminal_error(caplog) -> None:
    async def process(item: int) -> int:
        if item % 2:
            raise ValueError(f"request {item} rejected")
        return item

    with caplog.at_level(logging.ERROR, logger="olmo_eval.inference.dispatch"):
        results = asyncio.run(dispatch_concurrent([0, 1, 2, 3], process, max_in_flight=1))

    assert results == [0, None, 2, None]
    summaries = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
    assert summaries == [
        "dispatch: 2/4 requests failed; first error: ValueError: request 1 rejected"
    ]


def test_dispatch_logs_no_summary_when_nothing_failed(caplog) -> None:
    async def process(item: int) -> int:
        return item

    with caplog.at_level(logging.ERROR, logger="olmo_eval.inference.dispatch"):
        asyncio.run(dispatch_concurrent([0, 1], process))

    assert not [r for r in caplog.records if r.levelno == logging.ERROR]


def test_dispatch_preserves_order_and_final_errors_after_retries() -> None:
    calls: dict[int, int] = {}
    error = ValueError("rejected")

    async def process(item: int) -> int:
        calls[item] = calls.get(item, 0) + 1
        if item == 1:
            raise error
        if item == 2 and calls[item] == 1:
            raise ValueError("transient")
        return item

    results = asyncio.run(dispatch_concurrent_results([0, 1, 2], process, max_retries=1))
    assert [result.index for result in results] == [0, 1, 2]
    assert [result.result for result in results] == [0, None, 2]
    assert [result.error for result in results] == [None, error, None]
    assert calls == {0: 1, 1: 2, 2: 2}


def test_single_request_failure_has_no_duplicate_error_summary(caplog) -> None:
    async def process(item: int) -> int:
        raise ValueError("rejected")

    with caplog.at_level(logging.WARNING, logger="olmo_eval.inference.dispatch"):
        assert asyncio.run(dispatch_concurrent([0], process)) == [None]
    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == logging.WARNING


def test_dispatch_summary_preserves_falsey_first_exception(caplog) -> None:
    class FalseyError(Exception):
        def __bool__(self) -> bool:
            return False

    async def process(item: int) -> int:
        if item == 0:
            raise FalseyError("first")
        raise ValueError("second")

    with caplog.at_level(logging.ERROR, logger="olmo_eval.inference.dispatch"):
        asyncio.run(dispatch_concurrent([0, 1], process, max_in_flight=1))
    assert "first error: FalseyError: first" in caplog.text
