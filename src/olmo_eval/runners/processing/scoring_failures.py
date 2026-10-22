"""Instances whose scores could not be computed.

A scorer that cannot reach a verdict for an output, because a judge call failed, a
sandbox was lost, or the scorer raised, records the error under the output's
``scoring_errors`` metadata and gives the output no score. An instance is:

- **scored** when every output was scored;
- **partially scored** when some outputs were scored and some were not; its score
  comes from the outputs that were;
- **unscored** when no output was scored, or when it has no outputs and scoring it
  raised. It is left out of the task's metrics.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from olmo_eval.common.types import LMOutput, Response

SCORING_ERRORS_KEY = "scoring_errors"

#: Longest first-error message kept in task results and run summaries.
MAX_ERROR_MESSAGE_CHARS = 500


class ScoringState(StrEnum):
    SCORED = "scored"
    PARTIAL = "partial"
    UNSCORED = "unscored"


def output_scoring_errors(output: LMOutput) -> list[dict[str, str]]:
    """The scoring errors recorded on an output, in the order they were recorded."""
    errors = (output.metadata or {}).get(SCORING_ERRORS_KEY)
    if not isinstance(errors, dict):
        return []
    return [error for error in errors.values() if isinstance(error, dict)]


def scoring_state(response: Response) -> ScoringState:
    """Whether a response's outputs were scored, some of them, or none."""
    if not response.outputs:
        return ScoringState.UNSCORED if response.scoring_error else ScoringState.SCORED
    failed = sum(1 for output in response.outputs if output_scoring_errors(output))
    if failed == 0:
        return ScoringState.SCORED
    if failed == len(response.outputs):
        return ScoringState.UNSCORED
    return ScoringState.PARTIAL


def measurable_responses(responses: Sequence[Response]) -> list[Response]:
    """The responses that count toward the task's metrics: all but the unscored."""
    return [r for r in responses if scoring_state(r) is not ScoringState.UNSCORED]


@dataclass
class ScoringFailures:
    """Counts of a task's scoring failures, for its results and run summary.

    ``outputs_unscored`` counts outputs with a scoring error, across scored, partially
    scored and unscored instances. ``infrastructure`` counts those whose failure is
    flagged as infrastructure. ``by_type`` counts them by exception type, taking each
    output's first error; a response with no outputs counts once.
    """

    instances_unscored: int = 0
    instances_partially_scored: int = 0
    outputs_unscored: int = 0
    infrastructure: int = 0
    by_type: dict[str, int] = field(default_factory=dict)
    first_error: str | None = None

    def __bool__(self) -> bool:
        return bool(self.instances_unscored or self.outputs_unscored)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "instances_unscored": self.instances_unscored,
            "instances_partially_scored": self.instances_partially_scored,
            "outputs_unscored": self.outputs_unscored,
            "infrastructure": self.infrastructure,
            "by_type": dict(self.by_type),
        }
        if self.first_error is not None:
            result["first_error"] = self.first_error
        return result


def describe_error(error: dict[str, str]) -> str:
    """``Type: message`` for a scoring error, with the message capped in length."""
    text = error.get("type", "Error")
    message = error.get("message")
    if message:
        if len(message) > MAX_ERROR_MESSAGE_CHARS:
            message = message[: MAX_ERROR_MESSAGE_CHARS - 1] + "…"
        text += f": {message}"
    return text


def count_scoring_failures(responses: Sequence[Response]) -> ScoringFailures:
    """Count the scoring failures recorded on a task's responses."""
    failures = ScoringFailures()
    by_type: Counter[str] = Counter()

    def record(error: dict[str, str], *, flagged: bool) -> None:
        by_type[error.get("type", "Error")] += 1
        failures.infrastructure += flagged
        if failures.first_error is None:
            failures.first_error = describe_error(error)

    for response in responses:
        state = scoring_state(response)
        if state is ScoringState.UNSCORED:
            failures.instances_unscored += 1
        elif state is ScoringState.PARTIAL:
            failures.instances_partially_scored += 1
        if not response.outputs and response.scoring_error:
            error = response.scoring_error
            record(error, flagged=error.get("infrastructure") == "true")
        for output in response.outputs:
            errors = output_scoring_errors(output)
            if not errors:
                continue
            failures.outputs_unscored += 1
            record(errors[0], flagged=any(e.get("infrastructure") == "true" for e in errors))
    failures.by_type = dict(sorted(by_type.items()))
    return failures


def format_scoring_failures(failures: dict[str, Any] | None) -> str:
    """Render a task's scoring failures compactly for log lines and the summary table."""
    if not failures:
        return ""
    parts = []
    if failures.get("instances_unscored"):
        parts.append(f"{failures['instances_unscored']} unscored")
    if failures.get("instances_partially_scored"):
        parts.append(f"{failures['instances_partially_scored']} partially scored")
    return ", ".join(parts)


__all__ = [
    "MAX_ERROR_MESSAGE_CHARS",
    "SCORING_ERRORS_KEY",
    "ScoringFailures",
    "ScoringState",
    "count_scoring_failures",
    "describe_error",
    "format_scoring_failures",
    "measurable_responses",
    "output_scoring_errors",
    "scoring_state",
]
