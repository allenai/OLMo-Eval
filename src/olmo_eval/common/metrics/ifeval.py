"""IFBench / IFEval / Multi-IF metrics.

The four IFEval metrics share :class:`IFEvalScorer`, which writes
per-instruction strict and loose pass lists to ``output.metadata["ifeval"]``.
Each metric aggregates that side-band data along the prompt or instruction
axis. :class:`MultiIFAccuracy` combines the same four aggregations per
conversation turn, from the lists :class:`MultiIFScorer` records per turn.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from olmo_eval.common.metrics.base import Metric
from olmo_eval.common.scorers import IFEvalScorer, MultiIFScorer, Scorer
from olmo_eval.common.scorers.base import get_scorer_result
from olmo_eval.common.types import Response


def _iter_results(responses: Sequence[Response], key: str) -> list[list[bool]]:
    out: list[list[bool]] = []
    for response in responses:
        if not response.outputs:
            out.append([])
            continue
        meta = response.outputs[0].metadata or {}
        ifeval = meta.get("ifeval") or {}
        out.append(list(ifeval.get(key, [])))
    return out


def _prompt_level(results: list[list[bool]]) -> float:
    if not results:
        return 0.0
    correct = sum(1 for r in results if r and all(r))
    return correct / len(results)


def _instruction_level(results: list[list[bool]]) -> float:
    total = sum(len(r) for r in results)
    if total == 0:
        return 0.0
    correct = sum(1 for r in results for v in r if v)
    return correct / total


@dataclass(frozen=True, slots=True)
class IFEvalPromptStrictAccuracy(Metric):
    """Fraction of prompts where every instruction passes under strict scoring."""

    name: str = "prompt_level_strict_acc"
    scorer: type[Scorer] = IFEvalScorer

    def compute(self, responses: Sequence[Response]) -> float:
        return _prompt_level(_iter_results(responses, "strict"))


@dataclass(frozen=True, slots=True)
class IFEvalPromptLooseAccuracy(Metric):
    """Fraction of prompts where every instruction passes under loose scoring."""

    name: str = "prompt_level_loose_acc"
    scorer: type[Scorer] = IFEvalScorer

    def compute(self, responses: Sequence[Response]) -> float:
        return _prompt_level(_iter_results(responses, "loose"))


@dataclass(frozen=True, slots=True)
class IFEvalInstStrictAccuracy(Metric):
    """Fraction of individual instructions passing under strict scoring."""

    name: str = "inst_level_strict_acc"
    scorer: type[Scorer] = IFEvalScorer

    def compute(self, responses: Sequence[Response]) -> float:
        return _instruction_level(_iter_results(responses, "strict"))


@dataclass(frozen=True, slots=True)
class IFEvalInstLooseAccuracy(Metric):
    """Fraction of individual instructions passing under loose scoring."""

    name: str = "inst_level_loose_acc"
    scorer: type[Scorer] = IFEvalScorer

    def compute(self, responses: Sequence[Response]) -> float:
        return _instruction_level(_iter_results(responses, "loose"))


def _multi_if_turns(response: Response) -> list[dict[str, list[bool]]]:
    """Per-turn pass lists for one conversation; every instruction fails when unscored."""
    result = (
        get_scorer_result(response.outputs[0], MultiIFScorer.name) if response.outputs else None
    )
    if result is not None:
        return result["turns"]
    return [
        {"strict": [False] * len(ids), "loose": [False] * len(ids)}
        for ids in (turn["instruction_id_list"] for turn in response.instance.metadata["turns"])
    ]


def _multi_if_turn_score(turns: list[dict[str, list[bool]]]) -> float:
    """Multi-IF's score for a set of turns: the mean of the four IFEval accuracies."""
    strict = [turn["strict"] for turn in turns]
    loose = [turn["loose"] for turn in turns]
    return (
        _prompt_level(strict)
        + _instruction_level(strict)
        + _prompt_level(loose)
        + _instruction_level(loose)
    ) / 4


@dataclass(frozen=True, slots=True)
class MultiIFAccuracy(Metric):
    """Multi-IF accuracy for one turn, or averaged over every turn.

    A turn's score in one language is the mean of the prompt- and
    instruction-level accuracies under strict and loose scoring, over that
    language's conversations that reach the turn. Languages are then averaged
    with equal weight, as in the paper's tables. Without a ``turn``, the
    per-turn scores are averaged with equal weight. A ``language`` restricts
    the score to that language.
    """

    name: str = "accuracy"
    scorer: type[Scorer] = MultiIFScorer
    turn: int | None = None
    language: str | None = None

    def compute(self, responses: Sequence[Response]) -> float:
        # language -> turn number -> that turn's pass lists, one per conversation
        by_language: dict[str, dict[int, list[dict[str, list[bool]]]]] = {}
        for response in responses:
            language = response.instance.metadata.get("language", "")
            if self.language is not None and language != self.language:
                continue
            by_turn = by_language.setdefault(language, {})
            for index, turn in enumerate(_multi_if_turns(response), start=1):
                by_turn.setdefault(index, []).append(turn)

        turns = (
            {self.turn} if self.turn is not None else {t for b in by_language.values() for t in b}
        )
        turn_scores = []
        for turn in sorted(turns):
            language_scores = [
                _multi_if_turn_score(by_turn[turn])
                for by_turn in by_language.values()
                if turn in by_turn
            ]
            if language_scores:
                turn_scores.append(sum(language_scores) / len(language_scores))
        return sum(turn_scores) / len(turn_scores) if turn_scores else 0.0

    def compute_instance(self, response: Response) -> float | None:
        """Instruction-level accuracies pool across conversations, so no instance value exists."""
        return None

    def supports_scorer_fallback(self) -> bool:
        return False
