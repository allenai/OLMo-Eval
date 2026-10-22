"""Instances whose scores could not be computed.

A scorer that cannot reach a verdict for an output, because a judge call failed, a
sandbox was lost, or the scorer raised, records the error under the output's
``scoring_errors`` metadata, keyed by the scorer's name, and gives the output no score
from that scorer. An error under any other key, such as the runner's ``__response__``,
applies to every scorer.

For one scorer, an instance is:

- **scored** when every output was scored;
- **partially scored** when some outputs were scored and some were not; its score
  comes from the outputs that were;
- **unscored** when no output was scored, or when it has no outputs and scoring it
  raised. It is left out of that scorer's metrics.

Across a task, an instance is unscored if any of its scorers left it unscored, and it
then counts against the task's failure budget.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from olmo_eval.common.types import LMOutput, Response

SCORING_ERRORS_KEY = "scoring_errors"

#: Key of the error the runner records when a task's ``score_responses`` raised.
RESPONSE_ERROR_KEY = "__response__"

#: Longest first-error description kept in task results and run summaries.
MAX_ERROR_DESCRIPTION_CHARS = 500


class ScoringState(StrEnum):
    SCORED = "scored"
    PARTIAL = "partial"
    UNSCORED = "unscored"


def output_scoring_errors(output: LMOutput) -> dict[str, dict[str, str]]:
    """The scoring errors recorded on an output, keyed as recorded."""
    errors = (output.metadata or {}).get(SCORING_ERRORS_KEY)
    if not isinstance(errors, dict):
        return {}
    return {key: error for key, error in errors.items() if isinstance(error, dict)}


def _failed(output: LMOutput, scorer: str | None, scorers: Collection[str]) -> bool:
    """Whether an output has no score from ``scorer``, or from any scorer if it is None."""
    errors = output_scoring_errors(output)
    if scorer is None:
        return bool(errors)
    return any(key == scorer or key not in scorers for key in errors)


def scoring_state(
    response: Response, scorer: str | None = None, scorers: Collection[str] = ()
) -> ScoringState:
    """How far ``scorer`` scored a response; with no scorer, how far every scorer did.

    ``scorers`` names all of the task's scorers, so an error keyed by none of them
    counts against each one.
    """
    if not response.outputs:
        return ScoringState.UNSCORED if response.scoring_error else ScoringState.SCORED
    failed = sum(1 for output in response.outputs if _failed(output, scorer, scorers))
    if failed == 0:
        return ScoringState.SCORED
    if failed == len(response.outputs):
        return ScoringState.UNSCORED
    return ScoringState.PARTIAL


def task_scoring_state(response: Response, scorers: Collection[str]) -> ScoringState:
    """An instance's state across the task: unscored if any scorer left it unscored."""
    if not scorers:
        return scoring_state(response)
    states = {scoring_state(response, scorer, scorers) for scorer in scorers}
    if ScoringState.UNSCORED in states:
        return ScoringState.UNSCORED
    if ScoringState.PARTIAL in states:
        return ScoringState.PARTIAL
    return ScoringState.SCORED


def measurable_responses(
    responses: Sequence[Response], scorer: str | None = None, scorers: Collection[str] = ()
) -> list[Response]:
    """The responses ``scorer`` scored at least partly; with no scorer, any scorer did."""
    if scorer is None and scorers:
        return [
            r
            for r in responses
            if any(scoring_state(r, s, scorers) is not ScoringState.UNSCORED for s in scorers)
        ]
    return [r for r in responses if scoring_state(r, scorer, scorers) is not ScoringState.UNSCORED]


@dataclass
class ScoringFailures:
    """Counts of a task's scoring failures, for its results and run summary.

    ``outputs_unscored`` counts outputs with a scoring error, across all instances.
    ``infrastructure`` counts those whose failure is flagged as infrastructure.
    ``by_type`` counts them by exception type, taking each output's first error; a
    response with no outputs counts once.
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
    """``Type: message`` for a scoring error, capped at ``MAX_ERROR_DESCRIPTION_CHARS``."""
    text = error.get("type", "Error")
    if error.get("message"):
        text += f": {error['message']}"
    if len(text) > MAX_ERROR_DESCRIPTION_CHARS:
        text = text[: MAX_ERROR_DESCRIPTION_CHARS - 1] + "…"
    return text


def count_scoring_failures(
    responses: Sequence[Response], scorers: Collection[str] = ()
) -> ScoringFailures:
    """Count the scoring failures recorded on a task's responses."""
    failures = ScoringFailures()
    by_type: Counter[str] = Counter()

    def record(error: dict[str, str], *, flagged: bool) -> None:
        by_type[error.get("type", "Error")] += 1
        failures.infrastructure += flagged
        if failures.first_error is None:
            failures.first_error = describe_error(error)

    for response in responses:
        state = task_scoring_state(response, scorers)
        if state is ScoringState.UNSCORED:
            failures.instances_unscored += 1
        elif state is ScoringState.PARTIAL:
            failures.instances_partially_scored += 1
        if not response.outputs and response.scoring_error:
            error = response.scoring_error
            record(error, flagged=error.get("infrastructure") == "true")
        for output in response.outputs:
            errors = list(output_scoring_errors(output).values())
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
    "MAX_ERROR_DESCRIPTION_CHARS",
    "RESPONSE_ERROR_KEY",
    "SCORING_ERRORS_KEY",
    "ScoringFailures",
    "ScoringState",
    "count_scoring_failures",
    "describe_error",
    "format_scoring_failures",
    "measurable_responses",
    "output_scoring_errors",
    "scoring_state",
    "task_scoring_state",
]
