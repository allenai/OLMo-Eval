"""Shared base for tasks whose outputs a model judge grades.

A judged task generates as usual and grades every output in ``score_responses``.
The judge model, reasoning effort, and token budget are task configuration, so they
are part of the task hash: runs with different judges are stored and compared as
separate results. ``-o judge_model=<model>`` (and ``-o judge_reasoning_effort=<effort>``)
changes the judge of one task; ``OLMO_EVAL_JUDGE=<model>[:<effort>]`` changes every
judged task in a run whose judge was left at its default. A task override wins.

Judge failures are kept apart from wrong answers. A judge call that fails after its
retries, or a reply that never parses, is recorded as an incomplete score with the
raw reply and left out of the instance average; the run's result is flagged
incomplete. Metrics built on :class:`JudgedScoreMetric` leave such instances out.
"""

from __future__ import annotations

import asyncio
import logging
import os
from abc import abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any, ClassVar, TypeVar

from olmo_eval.common.metrics import Metric
from olmo_eval.common.scorers import ScoringIncompleteError
from olmo_eval.common.scorers.base import Scorer
from olmo_eval.common.scorers.llm_judge import JudgeFn, build_openai_judge_fn
from olmo_eval.common.types import Instance, LMOutput, Response
from olmo_eval.inference.retry import retry_with_backoff

from .base import Task, TaskConfig, _format_scoring_error, _store_output_score

logger = logging.getLogger(__name__)

V = TypeVar("V")

#: Judge calls per output before an unparseable reply is given up on.
JUDGE_ATTEMPTS = 3
#: Judge concurrency when the scoring context does not set one.
DEFAULT_JUDGE_CONCURRENCY = 16

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"

#: Where the run-wide judge limiter is kept on the scoring context.
_JUDGE_SEMAPHORE_ATTR = "_judged_task_semaphore"


class JudgeParseError(ScoringIncompleteError):
    """The judge gave no parseable verdict after every attempt."""


def parse_judge_spec(spec: str) -> tuple[str, str | None]:
    """Split ``model[:reasoning_effort]`` into a model and an optional effort."""
    model, separator, effort = spec.partition(":")
    if not model:
        raise ValueError("OLMO_EVAL_JUDGE must name a judge model")
    return model, effort if separator and effort else None


def visible_response(text: str) -> str:
    """Return the response after any reasoning block, or "" if the block never closed.

    Text without a ``<think>`` tag is returned as is, stripped.
    """
    if _THINK_CLOSE in text:
        return text.rpartition(_THINK_CLOSE)[2].strip()
    if _THINK_OPEN in text:
        return ""
    return text.strip()


def is_truncated(output: LMOutput) -> bool:
    """Whether generation stopped at the token limit."""
    return (output.metadata or {}).get("finish_reason") == "length"


@dataclass(frozen=True)
class JudgeScorer(Scorer):
    """Reads the score a judged task stored on an output.

    Judged tasks grade outputs in ``score_responses``; this scorer names the score
    channel and serves the stored value. Subclass it to give a task its own channel
    name.
    """

    name: str = "judge"

    def score(self, instance: Instance, output: LMOutput) -> float:
        value = (output.metadata or {}).get(f"score:{self.name}", 0.0)
        return float(value) if isinstance(value, (int, float)) else 0.0


@dataclass(frozen=True)
class JudgedScoreMetric(Metric):
    """Mean of a judged response score, optionally restricted to one metadata value.

    An instance whose judging failed has no score and is left out rather than
    counted as zero. A slice with no scored instances reports 0.0.
    """

    name: str = "accuracy"
    scorer: type[Scorer] | Scorer = JudgeScorer
    score_key: str = "accuracy"
    metadata_key: str | None = None
    metadata_value: str | None = None

    def compute_instance(self, response: Response) -> float | None:
        if (
            self.metadata_key is not None
            and response.instance.metadata.get(self.metadata_key) != self.metadata_value
        ):
            return None
        value = response.scores.get(self.score_key)
        return float(value) if isinstance(value, (int, float)) else None

    def compute(self, responses: Sequence[Response]) -> float:
        values = [
            value
            for response in responses
            if (value := self.compute_instance(response)) is not None
        ]
        return sum(values) / len(values) if values else 0.0

    def supports_scorer_fallback(self) -> bool:
        return False

    def display_format(self) -> str:
        return "percentage"

    def metric_unit(self) -> str:
        return "proportion"


@dataclass(frozen=True)
class TruncationRateMetric(JudgedScoreMetric):
    """Share of instances with no complete response; lower is better."""

    name: str = "truncation_rate"
    score_key: str = "truncation_rate"

    def higher_is_better(self) -> bool:
        return False


@dataclass(frozen=True)
class JudgedMacroScoreMetric(Metric):
    """Unweighted mean of a judged response score over the groups of one metadata key.

    Each group contributes equally however many instances it holds. Instances whose
    judging failed are left out of their group, and a group with no scored instances
    is left out of the mean.
    """

    name: str = "macro_accuracy"
    scorer: type[Scorer] | Scorer = JudgeScorer
    score_key: str = "accuracy"
    metadata_key: str = ""
    groups: tuple[str, ...] = ()

    def compute(self, responses: Sequence[Response]) -> float:
        group_means: list[float] = []
        for group in self.groups:
            values = [
                float(value)
                for response in responses
                if response.instance.metadata.get(self.metadata_key) == group
                and isinstance(value := response.scores.get(self.score_key), (int, float))
            ]
            if values:
                group_means.append(sum(values) / len(values))
        return sum(group_means) / len(group_means) if group_means else 0.0

    def compute_instance(self, response: Response) -> float | None:
        """A macro average has no per-instance equivalent."""
        return None

    def supports_scorer_fallback(self) -> bool:
        return False

    def display_format(self) -> str:
        return "percentage"

    def metric_unit(self) -> str:
        return "proportion"


class JudgedTask(Task):
    """A task whose outputs an OpenAI-compatible model judge grades.

    Subclasses set ``judge_model`` and ``judge_max_tokens`` (and optionally
    ``judge_reasoning_effort``) as task configuration, name their score channel with
    ``judge_scorer_name``, and implement :meth:`_score_output`. Each instance gets
    ``accuracy``, the aggregate over its judged outputs, and ``truncation_rate``.
    """

    required_secrets = ("OPENAI_API_KEY",)

    #: Score channel carrying each output's accuracy; matches the task's scorer name.
    judge_scorer_name: ClassVar[str] = JudgeScorer.name

    def __init__(self, config: TaskConfig) -> None:
        # The environment override applies only where the task's own judge model was
        # left at its default, so ``-o judge_model=...`` wins.
        spec = os.getenv("OLMO_EVAL_JUDGE")
        if spec and config.judge_model == getattr(type(self), "judge_model", None):
            model, reasoning_effort = parse_judge_spec(spec)
            config = replace(
                config,
                judge_model=model,
                judge_reasoning_effort=config.judge_reasoning_effort or reasoning_effort,
            )
        super().__init__(config)

    @abstractmethod
    async def _score_output(self, response: Response, output: LMOutput, judge_fn: JudgeFn) -> float:
        """Judge one output, record its judge details, and return its accuracy.

        Implementations record their result with :meth:`_record_judgement` and call
        the judge through :meth:`_judge_output`, which raises when no verdict parses.
        """
        ...

    def extract_answer(self, output: LMOutput) -> str:
        """Hand the judge the visible response."""
        return visible_response(output.text)

    def _judged_answer(self, output: LMOutput) -> str:
        """Return the answer the judge should see for an output."""
        answer = output.extracted_answer
        return answer if isinstance(answer, str) else visible_response(output.text)

    def _get_judge_fn(self) -> JudgeFn:
        """Return this task's judge, built once so its request-shape cache persists."""
        judge_fn = getattr(self, "_judge_fn", None)
        if judge_fn is None:
            if self.config.judge_model is None or self.config.judge_max_tokens is None:
                raise ValueError(f"{type(self).__name__} requires a complete judge configuration")
            judge_fn = build_openai_judge_fn(
                model=self.config.judge_model,
                scorer_name=type(self).__name__,
                max_tokens=self.config.judge_max_tokens,
                temperature=0.0,
                reasoning_effort=self.config.judge_reasoning_effort,
            )
            self._judge_fn = judge_fn
        return judge_fn

    def _get_judge_semaphore(self, context: Any) -> asyncio.Semaphore:
        """Return the judge limiter, shared by every judged task in the run when possible.

        The scoring context lives exactly as long as the run, so the limiter is kept
        there; without one, each task gets its own.
        """
        owner = context if context is not None else self
        semaphore = getattr(owner, _JUDGE_SEMAPHORE_ATTR, None)
        if semaphore is None:
            concurrency = getattr(context, "scoring_concurrency", DEFAULT_JUDGE_CONCURRENCY)
            semaphore = asyncio.Semaphore(max(1, concurrency))
            setattr(owner, _JUDGE_SEMAPHORE_ATTR, semaphore)
        return semaphore

    async def _judge(
        self, judge_fn: JudgeFn, prompt: str, parse: Callable[[str], V | None]
    ) -> tuple[V | None, str]:
        """Call the judge, retrying API failures and unparseable replies.

        Returns the parsed verdict, or None if no attempt parsed, with the last raw reply.
        """
        raw = ""
        for attempt in range(JUDGE_ATTEMPTS):
            raw = await retry_with_backoff(
                lambda: judge_fn(prompt), context=f"{self.config.name} judge"
            )
            verdict = parse(raw)
            if verdict is not None:
                return verdict, raw
            if attempt < JUDGE_ATTEMPTS - 1:
                await asyncio.sleep(2**attempt)
        return None, raw

    async def _judge_output(
        self,
        output: LMOutput,
        judge_fn: JudgeFn,
        prompt: str,
        parse: Callable[[str], V | None],
        judge_result: dict[str, Any],
    ) -> V:
        """Judge one output's prompt and keep the raw reply in ``judge_result``.

        Raises :class:`JudgeParseError`, after recording ``judge_result`` on the
        output, when no attempt gives a parseable verdict.
        """
        verdict, raw = await self._judge(judge_fn, prompt, parse)
        judge_result["raw_judge_response"] = raw
        if verdict is None:
            judge_result["parse_error"] = True
            output.metadata["judge_result"] = judge_result
            raise JudgeParseError(f"no parseable verdict after {JUDGE_ATTEMPTS} judge attempts")
        return verdict

    def _record_judgement(
        self, output: LMOutput, judge_result: dict[str, Any], scores: dict[str, float]
    ) -> float:
        """Store an output's judge details and scores, and return its accuracy.

        ``scores`` must hold ``accuracy``; every entry is kept as ``score:<key>``.
        """
        output.metadata["judge_result"] = judge_result
        for key, value in scores.items():
            output.metadata[f"score:{key}"] = value
        _store_output_score(output, scorer_name=self.judge_scorer_name, score=scores["accuracy"])
        return scores["accuracy"]

    def _warn_about_missing_responses(self, responses: Sequence[Response]) -> None:
        """Log outputs that score wrong because the model gave no complete response."""
        name = self.config.name
        truncated = sum(
            1 for response in responses for output in response.outputs if is_truncated(output)
        )
        if truncated:
            logger.warning(
                "%s generation hit the token limit on %d output(s); check truncation_rate "
                "before reading accuracy as capability.",
                name,
                truncated,
            )
        unanswered = [response for response in responses if not response.outputs]
        if unanswered:
            logger.warning(
                "%s got no model output for %d instance(s), e.g. %s; they score wrong. A "
                "prompt longer than the context leaves room for no answer.",
                name,
                len(unanswered),
                unanswered[0].instance.metadata.get("id"),
            )

    async def score_responses(
        self,
        responses: Sequence[Response],
        context: Any = None,
    ) -> Sequence[Response]:
        """Judge every output concurrently, then aggregate each instance over its samples."""
        self._extract_answers(responses)
        judge_fn = self._get_judge_fn()
        semaphore = self._get_judge_semaphore(context)
        self._warn_about_missing_responses(responses)

        jobs = [
            (response_index, output_index)
            for response_index, response in enumerate(responses)
            for output_index in range(len(response.outputs))
        ]

        async def run(response_index: int, output_index: int) -> float:
            response = responses[response_index]
            async with semaphore:
                try:
                    return await self._score_output(
                        response, response.outputs[output_index], judge_fn
                    )
                except ScoringIncompleteError:
                    raise
                except Exception as exc:
                    # A judge call that failed after its retries left the output unscored,
                    # which the runner must report as incomplete rather than as wrong.
                    raise ScoringIncompleteError(
                        f"{self.config.name} judge call failed: {type(exc).__name__}: {exc}"
                    ) from exc

        results = await asyncio.gather(*(run(*job) for job in jobs), return_exceptions=True)

        accuracy_by_response: list[dict[int, float]] = [{} for _ in responses]
        judge_errors: list[Exception] = []
        failed_by_response = [0] * len(responses)
        for (response_index, output_index), accuracy in zip(jobs, results, strict=True):
            if isinstance(accuracy, BaseException):
                # Cancellation is control flow, not data about a sample.
                if not isinstance(accuracy, Exception):
                    raise accuracy
                judge_errors.append(accuracy)
                failed_by_response[response_index] += 1
                _store_output_score(
                    responses[response_index].outputs[output_index],
                    scorer_name=self.judge_scorer_name,
                    score=0.0,
                    scoring_error=_format_scoring_error(accuracy, phase="judge"),
                )
                continue
            accuracy_by_response[response_index][output_index] = accuracy

        if judge_errors:
            logger.warning(
                "%s judging failed on %d/%d output(s) (API errors or no parseable verdict); "
                "they are recorded as scoring errors and excluded from their instance average. "
                "First error: %s",
                self.config.name,
                len(judge_errors),
                len(jobs),
                judge_errors[0],
            )
        for response_index, response in enumerate(responses):
            if response.outputs and failed_by_response[response_index] == len(response.outputs):
                raise ScoringIncompleteError(
                    f"{self.config.name} judging failed for every sample of instance "
                    f"{response.instance.metadata.get('id', response_index)}: {judge_errors[0]}"
                ) from judge_errors[0]

        for response, accuracy in zip(responses, accuracy_by_response, strict=True):
            response.scores["accuracy"] = self._aggregate_output_scores(accuracy)
            response.scores["truncation_rate"] = (
                sum(is_truncated(output) for output in response.outputs) / len(response.outputs)
                if response.outputs
                else 1.0
            )
        return responses
