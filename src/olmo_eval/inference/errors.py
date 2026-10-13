"""Inference provider error classification."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from olmo_eval.common.types import LMOutput


class TerminalProviderError(RuntimeError):
    """A provider failure after which its worker cannot serve more requests."""


# Avoid importing optional GPU-only provider packages just to classify errors.
_TERMINAL_PROVIDER_ERRORS = {"vllm.v1.engine.exceptions.EngineDeadError"}


def _exception_chain(exc: BaseException) -> Iterator[BaseException]:
    """An exception followed by its causes and contexts, each once."""
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def classify_terminal_provider_error(exc: BaseException) -> TerminalProviderError | None:
    """Return a terminal provider error found in an exception chain, if any."""
    for current in _exception_chain(exc):
        if isinstance(current, TerminalProviderError):
            return current

        error_type = type(current)
        error_name = f"{error_type.__module__}.{error_type.__qualname__}"
        if error_name in _TERMINAL_PROVIDER_ERRORS:
            return TerminalProviderError(f"{error_name}: {current}")

    return None


#: ``LMOutput.metadata`` key a provider sets when it cannot serve one request of a
#: batch. Raising instead would fail every request in the batch; the runner records
#: the marked request as a failed instance rather than scoring its output.
REQUEST_ERROR_KEY = "request_error"


def request_error(outputs: Sequence[LMOutput]) -> str | None:
    """The failure a provider recorded for a request it could not serve, if any."""
    if outputs and all(REQUEST_ERROR_KEY in (output.metadata or {}) for output in outputs):
        return str(outputs[0].metadata[REQUEST_ERROR_KEY])
    return None


#: ``LMOutput.metadata`` key a provider sets, beside ``REQUEST_ERROR_KEY``, when the
#: request failed because its prompt left no room in the model's context window.
#: Retrying cannot help, and a caller that grows a conversation may want to treat
#: it as the end of that conversation rather than as a fault.
CONTEXT_OVERFLOW_KEY = "context_overflow"

_CONTEXT_OVERFLOW_CODES = {"context_length_exceeded"}
_CONTEXT_OVERFLOW_MESSAGES = ("maximum context length",)


def is_context_overflow(exc: BaseException) -> bool:
    """Whether an exception chain reports a prompt too long for the context window."""
    for current in _exception_chain(exc):
        if getattr(current, "code", None) in _CONTEXT_OVERFLOW_CODES:
            return True
        message = str(current).lower()
        if any(marker in message for marker in _CONTEXT_OVERFLOW_MESSAGES):
            return True
    return False


def is_context_overflow_output(outputs: Sequence[LMOutput]) -> bool:
    """Whether a provider marked a request as failing on context overflow."""
    return bool(outputs) and all(
        (output.metadata or {}).get(CONTEXT_OVERFLOW_KEY) for output in outputs
    )


def first_output(outputs: Sequence[LMOutput]) -> LMOutput:
    """The first output of a request, raising if the provider marked it failed.

    For a caller that reads a reply directly rather than through the runner,
    which records a marked request as a failed instance on its own.
    """
    error = request_error(outputs)
    if error is not None or not outputs:
        raise RuntimeError(error or "The provider returned no output for the request.")
    return outputs[0]


__all__ = [
    "CONTEXT_OVERFLOW_KEY",
    "REQUEST_ERROR_KEY",
    "TerminalProviderError",
    "classify_terminal_provider_error",
    "first_output",
    "is_context_overflow",
    "is_context_overflow_output",
    "request_error",
]
