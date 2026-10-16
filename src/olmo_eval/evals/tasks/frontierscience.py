"""FrontierScience expert-level scientific reasoning (arXiv 2601.21165).

Two tracks from the open-sourced gold set of ``openai/frontierscience``:

``frontierscience_olympiad``
    100 short-answer olympiad problems (IPhO / IChO / IBO level). A model judge
    compares the attempted answer against the reference answer and returns a
    CORRECT / INCORRECT verdict, so the primary metric is accuracy.

``frontierscience_olympiad:verified``
    The 83 of those 100 problems whose gold answers survived manual review.
    Everything else matches ``frontierscience_olympiad`` exactly -- same problem
    text, same upstream gold answer, same judge -- so a verified score is the
    full-set score restricted to those problems, not a separate measurement.

``frontierscience_olympiad:mc``
    The same verified subset with four curated choices per problem. It scores the
    log-likelihood of the labels A--D and therefore needs no judge.
    This recognition task is intended for base-model tracking and is not directly
    comparable with the free-response Olympiad score.
    Problem and answer notation includes display edits. Distractor provenance
    preserves the curation text; its zero-based ``choice_index`` identifies the
    corresponding final displayed choice.

``frontierscience_research``
    60 open-ended PhD-level research sub-tasks. Each carries a rubric totaling
    10 points that credits intermediate derivations as well as the final result.
    A model judge tallies the points earned; the paper marks a response a
    success at seven or more points, which is the primary metric here. The mean
    fraction of rubric points earned is reported alongside it because it is the
    more sensitive signal on a benchmark where frontier models score around 25%.

All tracks report the primary metric overall and per subject (biology,
chemistry, physics).

The problem statements already end with the benchmark's own answer-format
instruction, so the generation prompt is the ``problem`` field verbatim with
nothing prepended. Olympiad responses are cut down to the text after the final
``FINAL ANSWER`` marker the dataset asks for; research responses are graded
whole, since the rubric credits the derivation.

Judge prompts are reconstructed from the paper's Appendix B listings, including
its ``attemped`` typo. The paper typesets one example as math, so the exact
characters behind ``6.69 ~ 6.7`` are unrecoverable; the rendered form is used.

The paper judges with GPT-5 at high reasoning effort. The deliberate project
default is the dated ``gpt-5.5-2026-04-23:high`` snapshot; set
``OLMO_EVAL_JUDGE=gpt-5:high`` to reproduce the published numbers, or a cheaper
spec such as ``gpt-5-mini`` for iteration. The resolved judge configuration is
serialized with the task because judge strictness moves these scores.

The paper averages Olympiad over 20 independent trials and Research over 30. The
``:paper`` variants request that many samples per problem, and per-instance
scores are the mean over samples rather than the harness default of the max.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, replace
from functools import cache
from importlib.resources import files
from typing import Any

from olmo_eval.common.metrics import Metric
from olmo_eval.common.scorers.base import LogprobScorer, Scorer
from olmo_eval.common.scorers.llm_judge import JudgeFn, build_openai_judge_fn
from olmo_eval.common.types import (
    Instance,
    LMOutput,
    LMRequest,
    RequestType,
    Response,
    SamplingParams,
    Split,
)
from olmo_eval.data import DataSource
from olmo_eval.evals.tasks.common import (
    OutputScoreAggregation,
    Task,
    TaskConfig,
    register,
    register_variant,
)
from olmo_eval.evals.tasks.common.base import _format_scoring_error, _store_output_score
from olmo_eval.inference.retry import retry_with_backoff

logger = logging.getLogger(__name__)

FRONTIERSCIENCE_REPO = "openai/frontierscience"
# Pinned so a Hub update cannot silently change the gold set mid-comparison.
FRONTIERSCIENCE_REVISION = "25ed67db7da8f4591484e764008ff585544f5a30"
FRONTIERSCIENCE_VERIFIED_IDS_FILE = "frontierscience_olympiad_verified_ids.json"
FRONTIERSCIENCE_VERIFIED_MC_FILE = "frontierscience_olympiad_mc.jsonl"
#: Olympiad problems left after manual review of the published gold answers.
FRONTIERSCIENCE_VERIFIED_COUNT = 83

FRONTIERSCIENCE_DEFAULT_JUDGE_MODEL = "gpt-5.5-2026-04-23"
FRONTIERSCIENCE_DEFAULT_JUDGE_REASONING_EFFORT = "high"
FRONTIERSCIENCE_JUDGE_ATTEMPTS = 3
FRONTIERSCIENCE_DEFAULT_SCORING_CONCURRENCY = 8

FRONTIERSCIENCE_SUBJECTS = ("biology", "chemistry", "physics")

#: Every research rubric in the gold set sums to exactly this many points.
FRONTIERSCIENCE_RUBRIC_TOTAL = 10.0
#: The paper counts a research response as a success at this many rubric points.
FRONTIERSCIENCE_SUCCESS_THRESHOLD = 7.0

#: Scorer channel name shared by every FrontierScience metric.
FRONTIERSCIENCE_SCORER_NAME = "frontierscience_judge"

FRONTIERSCIENCE_OLYMPIAD_JUDGE_PROMPT = (
    "You are grading an attempted answer to a science olympiad problem. You will be given the "
    "problem, attempted answer, and reference answer. Evaluate the solution against the provided "
    "reference solution, ensuring it is complete and matches the reference solution. Pay close "
    "attention to detail and grade it strictly, but fairly.\n\n"
    "The reference answer is either a single number or expression in latex formatting, a chemical "
    "formula, a compound name, or a phrase referring to a specific name, entity, or method.\n\n"
    "Mark the attempted answer as correct if it fully matches the reference answer or is otherwise "
    "equivalent (e.g., an equivalent algebraic expression, a numerical number within 1 decimal "
    "place rounding of the reference answer (e.g., 6.69 ≈ 6.7), an equivalent name for a "
    "compound/formula, equivalent when accounting for units, etc.).\n"
    "Mark it as incorrect if it is not equivalent to the reference answer.\n\n"
    "***\n\n"
    "The problem: {problem}\n\n"
    "***\n\n"
    "The reference answer: {reference_answer}\n\n"
    "***\n\n"
    "The attempted answer: {answer}\n\n"
    "***\n\n"
    "First, think step-by-step about whether the attempted answer matches the reference answer.\n"
    'If the attempted answer is correct, write "VERDICT: CORRECT" in the last line of your '
    'response, with no other text or formatting. If it is incorrect, write "VERDICT: INCORRECT".'
)

FRONTIERSCIENCE_RESEARCH_JUDGE_PROMPT = (
    "You are grading a science exam.\n\n"
    "You will be given the problem, attempted answer, and a rubric to grade the answer. The rubric "
    "will total up to 10 points.\n\n"
    # "attemped" reproduces the paper's spelling; the prompt is kept verbatim.
    "Evaluate the attemped answer against the provided rubric. Pay close attention to detail and "
    "grade it strictly, but fairly.\n"
    "Only evaluate against the rubric, as you yourself should not make any judgements (e.g., even "
    "if you think the answer is correct but rubric is wrong, you should treat the rubric as the "
    "gold standard).\n"
    "Return the absolute total number of points earned (it can be a decimal based on the "
    "rubric).\n"
    "***\n\n"
    "The problem: {problem}\n\n"
    "***\n\n"
    "The rubric: {rubric}\n\n"
    "***\n\n"
    "The attempted answer: {answer}\n\n"
    "***\n\n"
    "First, think step-by-step about each rubric item. Explain your reasoning for each rubric "
    "item.\n"
    "Then, tally the points up and write `VERDICT: <total_points>` in the last line of your "
    "response, no other text. For example, `VERDICT: 2.5` or `VERDICT: 8`."
)

_OLYMPIAD_VERDICT = re.compile(r"VERDICT:\s*\**\s*(INCORRECT|CORRECT)\b", re.IGNORECASE)
_RESEARCH_VERDICT = re.compile(r"VERDICT:\s*\**\s*`?\s*(-?\d+(?:\.\d+)?)")

# Tolerates the markdown emphasis and trailing colon models add around the marker.
_FINAL_ANSWER_MARKER = re.compile(r"\**\s*FINAL\s+ANSWER\s*\**\s*:?\s*", re.IGNORECASE)
_MC_ANSWER_INSTRUCTION = re.compile(
    r"\s*Think step by step and solve the problem below\..*?"
    r"without any extra commentary or providing multiple answer attempts\.\s*$",
    re.DOTALL,
)


def strip_reasoning(text: str) -> str:
    """Drop a closed reasoning block so only the visible answer reaches the judge."""
    close_tag = "</think>"
    index = text.rfind(close_tag)
    if index >= 0:
        return text[index + len(close_tag) :].strip()
    return text.strip()


def extract_final_answer(text: str) -> str:
    """Return the text after the benchmark's ``FINAL ANSWER`` marker.

    Falls back to the whole visible response when the marker is absent or
    nothing follows it, so a judge still sees the model's attempt.
    """
    visible = strip_reasoning(text)
    matches = list(_FINAL_ANSWER_MARKER.finditer(visible))
    if not matches:
        return visible
    return visible[matches[-1].end() :].strip() or visible


def parse_olympiad_verdict(raw: str) -> float | None:
    """Map an olympiad judge reply onto 1.0 / 0.0, or None if no verdict parses."""
    matches = list(_OLYMPIAD_VERDICT.finditer(raw))
    if not matches:
        return None
    return 0.0 if matches[-1].group(1).upper() == "INCORRECT" else 1.0


def parse_research_points(raw: str) -> float | None:
    """Read the rubric point total from a research judge reply, clamped to 0-10."""
    matches = list(_RESEARCH_VERDICT.finditer(raw))
    if not matches:
        return None
    points = float(matches[-1].group(1))
    return min(max(points, 0.0), FRONTIERSCIENCE_RUBRIC_TOTAL)


@cache
def verified_task_group_ids() -> frozenset[str]:
    """Return the Olympiad ``task_group_id`` values that passed manual answer review.

    The same 83 ids back the multiple-choice adaptation of this subset, so the two
    tracks cover identical problems and the file is the single place that changes
    if a problem is later added back or removed.
    """
    resource = files(__package__).joinpath(FRONTIERSCIENCE_VERIFIED_IDS_FILE)
    ids = frozenset(json.loads(resource.read_text(encoding="utf-8"))["task_group_ids"])
    if len(ids) != FRONTIERSCIENCE_VERIFIED_COUNT:
        raise ValueError(
            f"{FRONTIERSCIENCE_VERIFIED_IDS_FILE} holds {len(ids)} distinct ids, "
            f"expected {FRONTIERSCIENCE_VERIFIED_COUNT}"
        )
    return ids


def _parse_judge_spec(spec: str) -> tuple[str, str | None]:
    """Split ``model[:reasoning_effort]`` into serialized config values."""
    model, separator, effort = spec.partition(":")
    if not model:
        raise ValueError("OLMO_EVAL_JUDGE must name a judge model")
    return model, effort if separator and effort else None


def build_frontierscience_judge_fn(
    *,
    scorer_name: str,
    model: str,
    reasoning_effort: str | None,
    max_tokens: int,
) -> JudgeFn:
    """Build a judge from the task's serialized judge configuration."""
    return build_openai_judge_fn(
        model=model,
        temperature=0.0,
        # A reasoning judge that spends this budget truncates before its verdict
        # line; that call is retried and then scored 0.0.
        max_tokens=max_tokens,
        scorer_name=scorer_name,
        reasoning_effort=reasoning_effort,
    )


#: Where the shared judge limiter is stashed on the scoring context.
#:
#: Judge concurrency is one budget per run, not per task: every FrontierScience track
#: talks to the same judge API and both tracks are typically scored at once. The
#: scoring context is the object with exactly that lifetime -- shared across tasks,
#: discarded with the run. A module-level cache keyed by event loop would instead
#: leak, because a contended semaphore binds its own loop and the cached value then
#: keeps its weak key alive, and it would hand a second run on the same loop the
#: first run's concurrency setting.
_JUDGE_SEMAPHORE_ATTR = "_frontierscience_judge_semaphore"


def get_judge_semaphore(context: Any, concurrency: int) -> asyncio.Semaphore:
    """Return the judge limiter shared by every FrontierScience task in this run."""
    semaphore = getattr(context, _JUDGE_SEMAPHORE_ATTR, None)
    if semaphore is None:
        semaphore = asyncio.Semaphore(max(1, concurrency))
        setattr(context, _JUDGE_SEMAPHORE_ATTR, semaphore)
    return semaphore


async def judge_with_retries(
    judge_fn: JudgeFn,
    prompt: str,
    parse: Callable[[str], float | None],
) -> tuple[float | None, str]:
    """Retry transient API failures within each parse attempt."""
    raw = ""
    for attempt in range(FRONTIERSCIENCE_JUDGE_ATTEMPTS):
        raw = await retry_with_backoff(
            lambda: judge_fn(prompt),
            context="frontierscience judge",
        )
        value = parse(raw)
        if value is not None:
            return value, raw
        if attempt < FRONTIERSCIENCE_JUDGE_ATTEMPTS - 1:
            await asyncio.sleep(2**attempt)
    return None, raw


@dataclass(frozen=True)
class FrontierScienceScorer(Scorer):
    """Placeholder scorer; FrontierScience scores are computed in ``score_responses``."""

    name: str = FRONTIERSCIENCE_SCORER_NAME

    def score(self, instance: Instance, output: LMOutput) -> float:
        value = (output.metadata or {}).get(f"score:{self.name}", 0.0)
        return float(value) if isinstance(value, (int, float)) else 0.0


@dataclass(frozen=True)
class FrontierScienceMetric(Metric):
    """Mean of a task-computed response score, optionally restricted to one subject.

    All FrontierScience metrics are proportions on a 0-1 scale, so they display as
    percentages even where the metric name does not follow the framework's naming
    heuristics. A subject slice with no instances in the scored set has nothing to
    average and reports 0.0.
    """

    name: str = ""
    scorer: type[Scorer] | Scorer = FrontierScienceScorer
    score_key: str = ""
    subject: str | None = None

    def compute(self, responses: Sequence[Response]) -> float:
        selected = [
            response
            for response in responses
            if self.subject is None or response.instance.metadata.get("subject") == self.subject
        ]
        if not selected:
            return 0.0
        key = self.score_key or self.name
        return sum(response.scores.get(key, 0.0) for response in selected) / len(selected)

    def supports_scorer_fallback(self) -> bool:
        # The shared scorer channel carries the primary metric only, so a subject
        # slice must never fall back to it.
        return False

    def display_format(self) -> str:
        return "percentage"

    def metric_unit(self) -> str:
        return "proportion"


def _subject_metrics(score_key: str) -> tuple[FrontierScienceMetric, ...]:
    """Build one per-subject metric per FrontierScience subject for a score key."""
    return tuple(
        FrontierScienceMetric(
            name=f"{score_key}_{subject}",
            score_key=score_key,
            subject=subject,
        )
        for subject in FRONTIERSCIENCE_SUBJECTS
    )


@dataclass(frozen=True)
class FrontierScienceMCMetric(Metric):
    """A label-logprob metric, optionally restricted to one science subject."""

    name: str = ""
    scorer: type[Scorer] | Scorer = LogprobScorer
    metric_kind: str = "accuracy"
    subject: str | None = None

    def compute_instance(self, response: Response) -> float | None:
        if self.subject is not None and response.instance.metadata.get("subject") != self.subject:
            return None
        gold_idx = response.instance.metadata.get("gold_idx")
        choices = response.instance.choices or ()
        if (
            not isinstance(gold_idx, int)
            or not response.outputs
            or len(response.outputs) != len(choices)
        ):
            return None

        scorer = self.scorer() if isinstance(self.scorer, type) else self.scorer
        scores = [scorer.score(response.instance, output) for output in response.outputs]
        if self.metric_kind == "accuracy":
            return 1.0 if scores.index(max(scores)) == gold_idx else 0.0
        if self.metric_kind == "gold_probability":
            largest = max(scores)
            weights = [math.exp(score - largest) for score in scores]
            return weights[gold_idx] / sum(weights)
        if self.metric_kind == "logprob_margin":
            return scores[gold_idx] - max(
                score for index, score in enumerate(scores) if index != gold_idx
            )
        raise ValueError(f"unknown FrontierScience MC metric kind: {self.metric_kind}")

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
        return "percentage" if self.metric_kind != "logprob_margin" else "raw"

    def metric_unit(self) -> str:
        return "proportion" if self.metric_kind != "logprob_margin" else "logprob"


def _mc_metrics(metric_kind: str, name: str) -> tuple[FrontierScienceMCMetric, ...]:
    return (
        FrontierScienceMCMetric(name=name, metric_kind=metric_kind),
        *(
            FrontierScienceMCMetric(
                name=f"{name}_{subject}",
                metric_kind=metric_kind,
                subject=subject,
            )
            for subject in FRONTIERSCIENCE_SUBJECTS
        ),
    )


OLYMPIAD_ACCURACY = FrontierScienceMetric(name="accuracy", score_key="accuracy")
OLYMPIAD_METRICS = (OLYMPIAD_ACCURACY, *_subject_metrics("accuracy"))

MC_ACCURACY = _mc_metrics("accuracy", "accuracy")
MC_GOLD_PROBABILITY = _mc_metrics("gold_probability", "gold_probability")
MC_LOGPROB_MARGIN = _mc_metrics("logprob_margin", "logprob_margin")
MC_METRICS = (*MC_ACCURACY, *MC_GOLD_PROBABILITY, *MC_LOGPROB_MARGIN)

RESEARCH_SUCCESS_RATE = FrontierScienceMetric(name="success_rate", score_key="success_rate")
RESEARCH_RUBRIC_SCORE = FrontierScienceMetric(name="rubric_score", score_key="rubric_score")
RESEARCH_METRICS = (
    RESEARCH_SUCCESS_RATE,
    RESEARCH_RUBRIC_SCORE,
    *_subject_metrics("success_rate"),
    *_subject_metrics("rubric_score"),
)

_OLYMPIAD_SAMPLING = SamplingParams(temperature=0.0, max_tokens=32768)
_RESEARCH_SAMPLING = SamplingParams(temperature=0.0, max_tokens=32768)


class _FrontierScience(Task, ABC):
    """Shared FrontierScience loading, prompting, and judge orchestration."""

    split = Split.TRAIN  # HF JSON files load as a single "train" split
    # The paper averages each problem over independent trials, so multi-sample
    # runs must average rather than take the best sample.
    output_score_aggregation = OutputScoreAggregation.MEAN
    required_secrets = ("OPENAI_API_KEY",)

    #: Response score keys this track populates, primary first.
    score_keys: tuple[str, ...] = ()
    judge_model = FRONTIERSCIENCE_DEFAULT_JUDGE_MODEL
    judge_reasoning_effort = FRONTIERSCIENCE_DEFAULT_JUDGE_REASONING_EFFORT
    judge_max_tokens = 8192

    def __init__(self, config: TaskConfig) -> None:
        # Preserve the project-wide environment override, but resolve it before
        # hashing/artifact creation rather than consulting it during scoring.
        if spec := os.getenv("OLMO_EVAL_JUDGE"):
            model, reasoning_effort = _parse_judge_spec(spec)
            config = replace(
                config,
                judge_model=model,
                judge_reasoning_effort=reasoning_effort,
            )
        super().__init__(config)

    @property
    def instances(self) -> Iterator[Instance]:
        yield from self._load_instances_cached()

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        problem = str(doc.get("problem") or "").strip()
        answer = str(doc.get("answer") or "").strip()
        if not problem or not answer:
            return None

        subject = str(doc.get("subject") or "").strip().lower()
        if subject and subject not in FRONTIERSCIENCE_SUBJECTS:
            logger.warning("FrontierScience: unexpected subject %r at index %d", subject, index)

        # task_group_id is not unique: the Research gold set has 60 rows but 59 distinct
        # values (one row is duplicated byte-for-byte). Anything that keys on the
        # instance id -- pairwise storage, pass@k grouping -- would silently merge the
        # pair, so the row index goes into the id and the raw value is kept alongside.
        task_group_id = str(doc.get("task_group_id") or "")
        return Instance(
            question=problem,
            gold_answer=answer,
            metadata={
                "id": f"{task_group_id}:{index}"
                if task_group_id
                else f"{self.config.name}:{index}",
                "task_group_id": task_group_id,
                "subject": subject,
                "index": index,
            },
        )

    def format_request(self, instance: Instance) -> LMRequest:
        """Send the problem verbatim; it already carries the answer-format instruction."""
        return LMRequest(
            request_type=RequestType.CHAT,
            messages=({"role": "user", "content": instance.question},),
        )

    def _get_judge_fn(self) -> JudgeFn:
        """Return the run's single judge, built once per task.

        The runner scores one instance per ``score_responses`` call, so building a
        judge per call would give every instance a fresh client and a fresh copy of
        the adaptive request-shape cache in ``build_openai_judge_fn`` -- making each
        instance re-discover the same per-model parameter rejections.
        """
        judge_fn = getattr(self, "_judge_fn", None)
        if judge_fn is None:
            if self.config.judge_model is None or self.config.judge_max_tokens is None:
                raise ValueError("FrontierScience requires a complete judge configuration")
            judge_fn = build_frontierscience_judge_fn(
                scorer_name=type(self).__name__,
                model=self.config.judge_model,
                reasoning_effort=self.config.judge_reasoning_effort,
                max_tokens=self.config.judge_max_tokens,
            )
            self._judge_fn = judge_fn
        return judge_fn

    def _get_judge_semaphore(self, context: Any) -> asyncio.Semaphore:
        """Return the judge limiter, shared run-wide when a scoring context exists."""
        if context is not None:
            return get_judge_semaphore(context, context.scoring_concurrency)

        # Without a scoring context there is no run-scoped object to hang the limiter
        # on, so fall back to one per task. Bounded, just not shared across tracks.
        semaphore = getattr(self, "_judge_semaphore", None)
        if semaphore is None:
            semaphore = asyncio.Semaphore(FRONTIERSCIENCE_DEFAULT_SCORING_CONCURRENCY)
            self._judge_semaphore = semaphore
        return semaphore

    @abstractmethod
    async def _score_output(
        self,
        response: Response,
        output: LMOutput,
        judge_fn: JudgeFn,
    ) -> dict[str, float]:
        """Judge one output, record its judge details, and return its score keys."""
        ...

    async def score_responses(
        self,
        responses: Sequence[Response],
        context: Any = None,
    ) -> Sequence[Response]:
        """Judge every output concurrently, then average each instance over its samples."""
        self._extract_answers(responses)
        judge_fn = self._get_judge_fn()
        semaphore = self._get_judge_semaphore(context)

        jobs = [
            (response_index, output_index)
            for response_index, response in enumerate(responses)
            for output_index in range(len(response.outputs))
        ]

        # A reasoning model that spends its whole token budget before emitting any
        # visible text leaves an output whose text is empty, which scores 0.0 and is
        # indistinguishable from a genuine miss unless it is announced. (A provider
        # request that fails outright never reaches scoring: the runner counts that
        # instance as failed and drops it from the denominator.)
        unanswered = sum(
            1
            for response in responses
            if not response.outputs or not any(output.text.strip() for output in response.outputs)
        )
        if unanswered:
            logger.warning(
                "FrontierScience has no visible model output for %d/%d instance(s); each scores "
                "0.0. Check the generation length and whether the provider routes reasoning away "
                "from the response text before reading this score as a capability measurement.",
                unanswered,
                len(responses),
            )

        async def run(response_index: int, output_index: int) -> dict[str, float]:
            response = responses[response_index]
            async with semaphore:
                return await self._score_output(response, response.outputs[output_index], judge_fn)

        # A judge call can still fail after its own retries (rate limit, timeout). Keep
        # the exceptions so one bad call costs its own sample rather than every sample
        # for that instance, which matters most for the multi-trial :paper variants.
        results = await asyncio.gather(*(run(*job) for job in jobs), return_exceptions=True)

        collected: list[dict[str, dict[int, float]]] = [
            {key: {} for key in self.score_keys} for _ in responses
        ]
        judge_errors: list[Exception] = []
        failed_by_response: list[int] = [0] * len(responses)
        for (response_index, output_index), scores in zip(jobs, results, strict=True):
            if isinstance(scores, BaseException):
                # Cancellation and other non-Exception failures are control flow, not
                # data about a sample; swallowing them would turn a cancelled run into
                # a run with quietly missing trials.
                if not isinstance(scores, Exception):
                    raise scores
                judge_errors.append(scores)
                failed_by_response[response_index] += 1
                # Record through the standard scorer-error channel so an omitted
                # trial stays auditable after the run.
                _store_output_score(
                    responses[response_index].outputs[output_index],
                    scorer_name=FRONTIERSCIENCE_SCORER_NAME,
                    score=0.0,
                    scoring_error=_format_scoring_error(scores, phase="judge"),
                )
                continue
            for key, value in scores.items():
                collected[response_index][key][output_index] = value

        # An instance whose every judge call failed has no score of its own. Raising
        # hands it to the runner, which annotates every output with the standard
        # ``__response__`` scoring error and logs the failure. Note the instance still
        # scores 0.0 and stays in the denominator -- the runner keeps the response and
        # has no path for excluding a scoring failure from a metric.
        for response_index, response in enumerate(responses):
            if response.outputs and failed_by_response[response_index] == len(response.outputs):
                raise RuntimeError(
                    "FrontierScience judge failed for every sample of instance "
                    f"{response.instance.metadata.get('id', response_index)}"
                ) from judge_errors[0]

        if judge_errors:
            logger.warning(
                "FrontierScience judge raised on %d/%d sample(s) after retries; those samples are "
                "excluded from their instance average. First error: %s",
                len(judge_errors),
                len(jobs),
                judge_errors[0],
            )

        for response, per_key in zip(responses, collected, strict=True):
            for key, output_scores in per_key.items():
                response.scores[key] = self._aggregate_output_scores(output_scores)

        unparsed = sum(
            1
            for response in responses
            for output in response.outputs
            if output.metadata.get("frontierscience_judge_parse_error")
        )
        if unparsed:
            logger.warning(
                "FrontierScience judge returned no parseable verdict for %d/%d output(s) after "
                "%d attempts; those outputs scored 0.0 and stay in the denominator.",
                unparsed,
                len(jobs),
                FRONTIERSCIENCE_JUDGE_ATTEMPTS,
            )
        return responses

    def _record(
        self,
        output: LMOutput,
        *,
        scores: dict[str, float],
        raw_judge_response: str,
        parse_error: bool,
        parsed_result: dict[str, float | None],
    ) -> None:
        """Persist per-output judge details and score channels."""
        output.metadata["frontierscience_judge_parse_error"] = parse_error
        judge_result: dict[str, Any] = {
            "scorer": FRONTIERSCIENCE_SCORER_NAME,
            **parsed_result,
            "parse_error": parse_error,
        }
        # Successful research replies can contain a long rubric walkthrough. The
        # parsed fields are sufficient to audit those; retain raw text only when
        # parsing failed and the response itself is needed for diagnosis.
        if parse_error:
            judge_result["raw_judge_response"] = raw_judge_response
        output.metadata["judge_result"] = judge_result
        for key, value in scores.items():
            output.metadata[f"score:{key}"] = value
        _store_output_score(
            output,
            scorer_name=FRONTIERSCIENCE_SCORER_NAME,
            score=scores[self.score_keys[0]],
        )


@register("frontierscience_olympiad")
class FrontierScienceOlympiad(_FrontierScience):
    """FrontierScience Olympiad track scored by a model equivalence checker."""

    data_source = DataSource(
        path=FRONTIERSCIENCE_REPO,
        data_files="olympiad/test.jsonl",
        revision=FRONTIERSCIENCE_REVISION,
        split="train",
    )
    metrics = OLYMPIAD_METRICS
    primary_metric = OLYMPIAD_ACCURACY
    sampling_params = _OLYMPIAD_SAMPLING
    score_keys = ("accuracy",)
    judge_max_tokens = 8192

    def extract_answer(self, output: LMOutput) -> str:
        """Keep only what follows the dataset's ``FINAL ANSWER`` marker."""
        return extract_final_answer(output.text)

    async def _score_output(
        self,
        response: Response,
        output: LMOutput,
        judge_fn: JudgeFn,
    ) -> dict[str, float]:
        answer = output.extracted_answer
        if not isinstance(answer, str):
            answer = extract_final_answer(output.text)

        prompt = FRONTIERSCIENCE_OLYMPIAD_JUDGE_PROMPT.format(
            problem=response.instance.question,
            reference_answer=response.instance.gold_answer or "",
            answer=answer,
        )
        verdict, raw = await judge_with_retries(judge_fn, prompt, parse_olympiad_verdict)
        scores = {"accuracy": 0.0 if verdict is None else verdict}
        self._record(
            output,
            scores=scores,
            raw_judge_response=raw,
            parse_error=verdict is None,
            parsed_result={"parsed_outcome": verdict},
        )
        return scores


@register("frontierscience_olympiad:verified")
class FrontierScienceOlympiadVerified(FrontierScienceOlympiad):
    """Olympiad restricted to the problems whose published gold answers held up.

    Reviewing all 100 answers to build the multiple-choice adaptation flagged 17 as
    ambiguously worded or wrong, which the judge scores as model errors. Problems
    and gold answers are still the upstream ones verbatim; only the row set differs.
    """

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        if str(doc.get("task_group_id") or "") not in verified_task_group_ids():
            return None
        return super().process_doc(doc, index)


@register("frontierscience_olympiad:mc")
class FrontierScienceOlympiadMC(Task):
    """Judge-free multiple-choice adaptation of the verified Olympiad subset.

    Choices and display text come from the curated release; which problems appear
    is decided by the shared verified index, so this track and
    ``frontierscience_olympiad:verified`` always cover the same problems.
    """

    data_source = DataSource(path=FRONTIERSCIENCE_VERIFIED_MC_FILE, split="test")
    split = Split.TEST
    metrics = MC_METRICS
    primary_metric = MC_ACCURACY[0]
    num_fewshot = 0
    sampling_params = SamplingParams(temperature=0.0, max_tokens=1)

    @property
    def instances(self) -> Iterator[Instance]:
        if self._instances_cache is None:
            self._instances_cache = self._load_verified_rows()
        yield from self._instances_cache

    def _load_verified_rows(self) -> list[Instance]:
        """Read the curated release, keeping the rows the verified index names."""
        instances: list[Instance] = []
        resource = files(__package__).joinpath(FRONTIERSCIENCE_VERIFIED_MC_FILE)
        with resource.open(encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if not line.strip():
                    continue
                instance = self.process_doc(json.loads(line), index)
                if instance is not None:
                    instances.append(instance)

        missing = sorted(
            verified_task_group_ids()
            - {instance.metadata["task_group_id"] for instance in instances}
        )
        if missing:
            raise ValueError(
                f"{FRONTIERSCIENCE_VERIFIED_MC_FILE} has no row for {len(missing)} verified "
                f"problem(s), starting with {missing[0]}"
            )
        return instances

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        # Filtering on the index rather than on the file means a problem dropped
        # from the verified subset leaves both tracks without a release rebuild.
        task_group_id = str(doc.get("task_group_id") or "")
        if task_group_id not in verified_task_group_ids():
            return None

        raw_problem = str(doc.get("problem") or "")
        problem, substitutions = _MC_ANSWER_INSTRUCTION.subn("", raw_problem)
        choices = tuple(str(choice).strip() for choice in doc.get("choices", ()))
        if not problem.strip() or not choices:
            return None
        if substitutions != 1:
            raise ValueError(f"expected one answer-instruction suffix at MC row {index}")
        if len(choices) != 4 or any(not choice for choice in choices):
            raise ValueError(f"expected four nonempty choices at MC row {index}")

        gold_idx = doc.get("correct_choice_index")
        if not isinstance(gold_idx, int) or not 0 <= gold_idx < len(choices):
            raise ValueError(f"invalid correct_choice_index at MC row {index}")
        gold_label = chr(ord("A") + gold_idx)
        if doc.get("correct_choice_label") != gold_label:
            raise ValueError(f"choice index/label mismatch at MC row {index}")
        if str(doc.get("reference_answer") or "").strip() != choices[gold_idx]:
            raise ValueError(f"reference answer/choice mismatch at MC row {index}")

        subject = str(doc.get("subject") or "").strip().lower()
        if subject not in FRONTIERSCIENCE_SUBJECTS:
            raise ValueError(f"unexpected subject {subject!r} at MC row {index}")
        return Instance(
            question=problem.strip(),
            choices=choices,
            gold_answer=gold_label,
            metadata={
                "id": task_group_id,
                "task_group_id": task_group_id,
                "subject": subject,
                "index": index,
                "gold_idx": gold_idx,
                "gold_text": choices[gold_idx],
            },
        )

    def format_request(self, instance: Instance) -> LMRequest:
        choices = instance.choices or ()
        rendered_choices = "\n".join(
            f"{chr(ord('A') + index)}. {choice}" for index, choice in enumerate(choices)
        )
        prompt = f"Problem:\n{instance.question}\n\nChoices:\n{rendered_choices}\n\nAnswer:"
        return LMRequest(
            request_type=RequestType.LOGLIKELIHOOD,
            prompt=prompt,
            continuations=tuple(f" {chr(ord('A') + index)}" for index in range(len(choices))),
        )

    def extract_answer(self, output: LMOutput) -> None:
        return None


@register("frontierscience_research")
class FrontierScienceResearch(_FrontierScience):
    """FrontierScience Research track scored against 10-point model-judged rubrics."""

    data_source = DataSource(
        path=FRONTIERSCIENCE_REPO,
        data_files="research/test.jsonl",
        revision=FRONTIERSCIENCE_REVISION,
        split="train",
    )
    metrics = RESEARCH_METRICS
    primary_metric = RESEARCH_SUCCESS_RATE
    sampling_params = _RESEARCH_SAMPLING
    score_keys = ("success_rate", "rubric_score")
    # Rubrics run up to 18 items and the judge reasons through each one before
    # tallying, so the reply needs far more room than the olympiad verdict.
    judge_max_tokens = 32768
    success_threshold: float = FRONTIERSCIENCE_SUCCESS_THRESHOLD

    def extract_answer(self, output: LMOutput) -> str:
        """Grade the whole visible response; the rubric credits the derivation."""
        return strip_reasoning(output.text)

    async def _score_output(
        self,
        response: Response,
        output: LMOutput,
        judge_fn: JudgeFn,
    ) -> dict[str, float]:
        answer = output.extracted_answer
        if not isinstance(answer, str):
            answer = strip_reasoning(output.text)

        prompt = FRONTIERSCIENCE_RESEARCH_JUDGE_PROMPT.format(
            problem=response.instance.question,
            rubric=response.instance.gold_answer or "",
            answer=answer,
        )
        points, raw = await judge_with_retries(judge_fn, prompt, parse_research_points)
        earned = 0.0 if points is None else points
        scores = {
            "success_rate": 1.0 if earned >= self.success_threshold else 0.0,
            "rubric_score": earned / FRONTIERSCIENCE_RUBRIC_TOTAL,
        }
        self._record(
            output,
            scores=scores,
            raw_judge_response=raw,
            parse_error=points is None,
            parsed_result={"rubric_points": points},
        )
        return scores


# The paper's scoring protocol: 20 Olympiad trials and 30 Research trials per
# problem, averaged. It does not state a temperature; 1.0 is the API default for
# the models it evaluates and is what makes independent trials differ.
register_variant(
    "frontierscience_olympiad",
    "paper",
    sampling_params=replace(_OLYMPIAD_SAMPLING, temperature=1.0, num_samples=20),
)
register_variant(
    "frontierscience_olympiad:verified",
    "paper",
    sampling_params=replace(_OLYMPIAD_SAMPLING, temperature=1.0, num_samples=20),
)
register_variant(
    "frontierscience_research",
    "paper",
    sampling_params=replace(_RESEARCH_SAMPLING, temperature=1.0, num_samples=30),
)
register_variant("frontierscience_olympiad:mc", "olmo3base")
