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

from collections.abc import Sequence
from enum import StrEnum

from olmo_eval.common.types import LMOutput, Response

SCORING_ERRORS_KEY = "scoring_errors"


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


__all__ = [
    "SCORING_ERRORS_KEY",
    "ScoringState",
    "measurable_responses",
    "output_scoring_errors",
    "scoring_state",
]
