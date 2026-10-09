"""Humanity's Last Exam (HLE, arXiv 2501.14249), text-only questions.

HLE has 2,500 expert-written, closed-ended questions across eight categories; about
42% are mathematics. Each has a short reference answer, either an exact-match string
or a multiple-choice letter. 342 questions include an image; these tasks drop them,
so they run on any model. A model judge grades the responses, so both tasks need
``OPENAI_API_KEY``.

Variants
--------
``hle:text``
    The 2,158 questions without an image, from ``cais/hle``. This is the question set
    behind public text-only HLE results, such as the Artificial Analysis index, but
    scores are only directly comparable when the judge matches theirs (see Judge).
    ``cais/hle`` is gated: accept its terms on Hugging Face with the account behind
    your ``HF_TOKEN``.

``hle:text:verified``
    The 575 text-only questions in the Gold subset of HLE-Verified (arXiv
    2602.13964). That audit found many HLE items with wrong reference answers or
    flawed questions, and kept as Gold only items whose question, answer, and
    rationale held up unchanged. Records come from ``skylenage-ai/HLE-Verified``
    because the audit repaired two Gold question statements; every Gold answer
    matches ``cais/hle``. Prefer this variant for measuring capability. Only 13
    chemistry questions survived the audit, so that slice is noisy.

Both datasets are pinned to a fixed Hub revision.

Judge
-----
The default judge is ``gpt-6-luna``; the official HLE scorer uses
``o3-mini-2025-01-31``. ``gpt-6-luna`` is an alias, not a dated snapshot, so the
model behind it can change without the task hash changing; pass a dated model with
``-o judge_model`` when results must stay comparable across time. Change the judge
for one task with ``-o judge_model=<model>`` (and ``-o judge_reasoning_effort=<effort>``
for reasoning judges), or for every judge task in a run with
``OLMO_EVAL_JUDGE=<model>[:<effort>]``; a task override wins. The judge
settings are part of the task configuration and hash, so runs with different judges
are stored and compared as separate results.

Examples::

    olmo-eval run -m <model> -t hle:text:verified
    olmo-eval run -m <model> -t hle:text:verified -o judge_model=o3-mini-2025-01-31
    uv run olmo-eval beaker launch -n hle-text -m <model> -t hle:text

Scoring
-------
The model gets the official system prompt (explanation, answer, confidence). The
judge gets the official judge prompt with the model's visible response. A closed
reasoning block is removed; a reasoning block that never closes means the model gave
no final response, which is scored wrong without calling the judge. An answer that hit
the generation token limit is also scored wrong without the judge: a model is expected
to finish within its budget. Such outputs carry ``truncated: true`` in their
``judge_result``, and a question the server returned nothing for (for example, a
prompt longer than the context) scores wrong and is logged. The judge's
``correct:`` and ``confidence:`` lines are parsed from plain text, so any chat model
can judge. A judge call that fails, or gives no parseable verdict after three
attempts, is recorded as an incomplete score with the raw reply and left out of
accuracy rather than counted as a wrong answer; the run's result is flagged
incomplete. Every judge reply is kept
in the output's ``judge_result``.

Metrics:

- ``accuracy`` (primary), overall, per category (``accuracy_<category>``), and per
  answer type (``accuracy_exact_match``, ``accuracy_multiple_choice``).
- ``calibration_error_all_bins``: RMS calibration error of the stated confidences
  over individual outputs, in bins of 100 as in the official scorer; lower is
  better. The official code leaves its highest-confidence bin out of the sum; this
  metric keeps it, hence the distinct name.
- ``truncation_rate``: share of questions with no complete response, because
  generation hit the token limit or the server returned nothing; lower is better.
  These questions already count as wrong in ``accuracy``; a high rate means the
  budget, not the model's knowledge, is limiting the score.

Generation samples at temperature 0.6 and top_p 0.95 with a 32,768-token budget.
Greedy decoding sends thinking models into repetition loops that run out the budget.
Raise the budget with ``-o max_tokens=<n>``, keeping prompt plus budget within the
model's context: the longest text-only HLE prompt is about 13.6k tokens.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from typing import Any

from olmo_eval.common.metrics import Metric
from olmo_eval.common.scorers import ScoringIncompleteError
from olmo_eval.common.scorers.base import Scorer
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
)
from olmo_eval.evals.tasks.common.base import _format_scoring_error, _store_output_score
from olmo_eval.inference.retry import retry_with_backoff

logger = logging.getLogger(__name__)

HLE_REPO = "cais/hle"
# Pinned so a Hub update cannot silently change the question set mid-comparison.
HLE_REVISION = "5a81a4c7271a2a2a312b9a690f0c2fde837e4c29"

HLE_VERIFIED_REPO = "skylenage-ai/HLE-Verified"
HLE_VERIFIED_REVISION = "0bc83643672d4f68a5f89998617a639d85e7318b"
HLE_VERIFIED_GOLD_FILES = "data/Gold_subset.part*.parquet"
HLE_VERIFIED_GOLD_CLASS = "Gold subset"

# An alias rather than a dated snapshot: OpenAI publishes no dated gpt-6-luna model
# (checked 2026-10-05), so the model behind this name can change under the same hash.
HLE_DEFAULT_JUDGE_MODEL = "gpt-6-luna"
HLE_JUDGE_MAX_TOKENS = 8192
HLE_JUDGE_ATTEMPTS = 3
HLE_DEFAULT_SCORING_CONCURRENCY = 16

#: Official calibration bin size, in outputs.
HLE_CALIBRATION_BIN_SIZE = 100

HLE_SCORER_NAME = "hle_judge"

#: Upstream category label -> metric suffix.
HLE_CATEGORIES = {
    "Math": "math",
    "Physics": "physics",
    "Chemistry": "chemistry",
    "Biology/Medicine": "biology_medicine",
    "Computer Science/AI": "computer_science_ai",
    "Engineering": "engineering",
    "Humanities/Social Science": "humanities_social_science",
    "Other": "other",
}

HLE_ANSWER_TYPES = {"exactMatch": "exact_match", "multipleChoice": "multiple_choice"}

# Verbatim from the official ``run_model_predictions.py``, which uses the same
# prompt for exact-match and multiple-choice questions.
HLE_SYSTEM_PROMPT = (
    "Your response should be in the following format:\n"
    "Explanation: {your explanation for your answer choice}\n"
    "Answer: {your chosen answer}\n"
    "Confidence: {your confidence score between 0% and 100% for your answer}"
)

# Verbatim from the official ``run_judge_results.py``, including its doubled "if
# there" and the ``0|\%|`` escaping.
HLE_JUDGE_PROMPT = (
    "Judge whether the following [response] to [question] is correct or not based on the "
    "precise and unambiguous [correct_answer] below.\n\n"
    "[question]: {question}\n\n"
    "[response]: {response}\n\n"
    "Your judgement must be in the format and criteria specified below:\n\n"
    "extracted_final_answer: The final exact answer extracted from the [response]. Put the "
    "extracted answer as 'None' if there is no exact, final answer to extract from the "
    "response.\n\n"
    "[correct_answer]: {correct_answer}\n\n"
    "reasoning: Explain why the extracted_final_answer is correct or incorrect based on "
    "[correct_answer], focusing only on if there are meaningful differences between "
    "[correct_answer] and the extracted_final_answer. Do not comment on any background to the "
    "problem, do not attempt to solve the problem, do not argue for any answer different than "
    "[correct_answer], focus only on whether the answers match.\n\n"
    "correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, "
    "or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. "
    "if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted "
    "answer is incorrect.\n\n\n"
    "confidence: The extracted confidence score between 0|\\%| and 100|\\%| from [response]. "
    "Put 100 if there is no confidence score available."
)

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"

# Judges wrap field names in markdown emphasis and sometimes quote the value.
_FIELD_PREFIX = r"^[\s>*_#-]*{name}[\s*_]*:[\s*_]*"
_CORRECT_FIELD = re.compile(
    _FIELD_PREFIX.format(name="correct") + r"['\"`]?(yes|no)\b",
    re.IGNORECASE | re.MULTILINE,
)
_CONFIDENCE_FIELD = re.compile(
    _FIELD_PREFIX.format(name="confidence") + r"(\d+(?:\.\d+)?)",
    re.IGNORECASE | re.MULTILINE,
)
_EXTRACTED_ANSWER_FIELD = re.compile(
    _FIELD_PREFIX.format(name="extracted_final_answer") + r"(.*)$",
    re.IGNORECASE | re.MULTILINE,
)


@dataclass(frozen=True)
class JudgeVerdict:
    """The fields read from one judge reply."""

    correct: bool
    confidence: float
    extracted_final_answer: str | None


class HLEJudgeParseError(ScoringIncompleteError):
    """The judge gave no parseable verdict after every attempt."""


def parse_judge_reply(raw: str) -> JudgeVerdict | None:
    """Read the verdict, confidence, and extracted answer from a judge reply.

    Returns None when no ``correct:`` verdict parses. A missing confidence counts as
    100%, which is what the judge prompt tells the judge to report when the response
    states none. Confidence is returned on a 0-1 scale.
    """
    verdicts = list(_CORRECT_FIELD.finditer(raw))
    if not verdicts:
        return None
    confidences = list(_CONFIDENCE_FIELD.finditer(raw))
    confidence = float(confidences[-1].group(1)) if confidences else 100.0
    answers = list(_EXTRACTED_ANSWER_FIELD.finditer(raw))
    return JudgeVerdict(
        correct=verdicts[-1].group(1).lower() == "yes",
        confidence=min(max(confidence, 0.0), 100.0) / 100.0,
        extracted_final_answer=answers[0].group(1).strip() if answers else None,
    )


def visible_response(text: str) -> str:
    """Return the response after any reasoning block, or "" if the block never closed.

    Text without a ``<think>`` tag is returned as is.
    """
    if _THINK_CLOSE in text:
        return text.rpartition(_THINK_CLOSE)[2].strip()
    if _THINK_OPEN in text:
        return ""
    return text.strip()


def calibration_error(
    confidences: Sequence[float],
    correct: Sequence[float],
    bin_size: int = HLE_CALIBRATION_BIN_SIZE,
) -> float:
    """RMS calibration error over equal-count bins of ascending confidence.

    Bins hold ``bin_size`` items, the last bin absorbs the remainder, and fewer than
    ``bin_size`` items form a single bin. Returns a 0-1 value.
    """
    if len(confidences) != len(correct):
        raise ValueError("confidences and correct must have the same length")
    total = len(confidences)
    if total == 0:
        return 0.0
    pairs = sorted(zip(confidences, correct, strict=True), key=lambda pair: pair[0])
    num_bins = max(1, total // bin_size)
    squared_error = 0.0
    for bin_index in range(num_bins):
        start = bin_index * bin_size
        end = total if bin_index == num_bins - 1 else start + bin_size
        members = pairs[start:end]
        mean_confidence = sum(pair[0] for pair in members) / len(members)
        mean_correct = sum(pair[1] for pair in members) / len(members)
        squared_error += len(members) / total * (mean_confidence - mean_correct) ** 2
    return math.sqrt(squared_error)


def _parse_judge_spec(spec: str) -> tuple[str, str | None]:
    """Split ``model[:reasoning_effort]``."""
    model, separator, effort = spec.partition(":")
    if not model:
        raise ValueError("OLMO_EVAL_JUDGE must name a judge model")
    return model, effort if separator and effort else None


_JUDGE_SEMAPHORE_ATTR = "_hle_judge_semaphore"


@dataclass(frozen=True)
class HLEScorer(Scorer):
    """Placeholder scorer; HLE scores are computed in ``score_responses``."""

    name: str = HLE_SCORER_NAME

    def score(self, instance: Instance, output: LMOutput) -> float:
        value = (output.metadata or {}).get(f"score:{self.name}", 0.0)
        return float(value) if isinstance(value, (int, float)) else 0.0


@dataclass(frozen=True)
class HLEAccuracyMetric(Metric):
    """Mean judged accuracy, optionally restricted to instances with one metadata value."""

    name: str = "accuracy"
    scorer: type[Scorer] | Scorer = HLEScorer
    metadata_key: str | None = None
    metadata_value: str | None = None

    def compute_instance(self, response: Response) -> float | None:
        """Return the instance's accuracy if it is in this slice and was judged, else None.

        An instance whose judging failed has no accuracy score; it is left out rather
        than counted as wrong.
        """
        if (
            self.metadata_key is not None
            and response.instance.metadata.get(self.metadata_key) != self.metadata_value
        ):
            return None
        value = response.scores.get("accuracy")
        return float(value) if isinstance(value, (int, float)) else None

    def compute(self, responses: Sequence[Response]) -> float:
        values = [
            value
            for response in responses
            if (value := self.compute_instance(response)) is not None
        ]
        return sum(values) / len(values) if values else 0.0

    def supports_pairwise_scorer_fallback(self) -> bool:
        return False

    def pairwise_display_format(self) -> str:
        return "percentage"

    def pairwise_unit(self) -> str:
        return "proportion"


@dataclass(frozen=True)
class HLECalibrationErrorMetric(Metric):
    """RMS calibration error of each judged output's stated confidence; lower is better.

    Reads per-output scores so that each answer stays paired with its own confidence
    when a question has several samples. Outputs whose judging failed are left out.
    """

    name: str = "calibration_error_all_bins"
    scorer: type[Scorer] | Scorer = HLEScorer

    def compute(self, responses: Sequence[Response]) -> float:
        confidences: list[float] = []
        correct: list[float] = []
        for response in responses:
            for output in response.outputs:
                metadata = output.metadata or {}
                if "score:confidence" not in metadata or "score:accuracy" not in metadata:
                    continue
                confidences.append(float(metadata["score:confidence"]))
                correct.append(float(metadata["score:accuracy"]))
        return calibration_error(confidences, correct)

    def supports_pairwise_scorer_fallback(self) -> bool:
        return False

    def pairwise_higher_is_better(self) -> bool:
        return False

    def pairwise_display_format(self) -> str:
        return "percentage"

    def pairwise_unit(self) -> str:
        return "proportion"


@dataclass(frozen=True)
class HLETruncationRateMetric(Metric):
    """Share of questions with no complete response; lower is better."""

    name: str = "truncation_rate"
    scorer: type[Scorer] | Scorer = HLEScorer

    def compute(self, responses: Sequence[Response]) -> float:
        values = [
            value
            for response in responses
            if (value := self.compute_instance(response)) is not None
        ]
        return sum(values) / len(values) if values else 0.0

    def supports_pairwise_scorer_fallback(self) -> bool:
        return False

    def pairwise_higher_is_better(self) -> bool:
        return False

    def pairwise_display_format(self) -> str:
        return "percentage"

    def pairwise_unit(self) -> str:
        return "proportion"


HLE_ACCURACY = HLEAccuracyMetric()
HLE_METRICS = (
    HLE_ACCURACY,
    HLECalibrationErrorMetric(),
    HLETruncationRateMetric(),
    *(
        HLEAccuracyMetric(
            name=f"accuracy_{suffix}", metadata_key="category", metadata_value=category
        )
        for category, suffix in HLE_CATEGORIES.items()
    ),
    *(
        HLEAccuracyMetric(
            name=f"accuracy_{suffix}", metadata_key="answer_type", metadata_value=answer_type
        )
        for answer_type, suffix in HLE_ANSWER_TYPES.items()
    ),
)


class _HLE(Task):
    """Shared HLE loading, prompting, and judge orchestration for the text-only tasks."""

    split = Split.TEST
    metrics = HLE_METRICS
    primary_metric = HLE_ACCURACY
    sampling_params = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=32768)
    output_score_aggregation = OutputScoreAggregation.MEAN
    required_secrets = ("OPENAI_API_KEY",)

    judge_model = HLE_DEFAULT_JUDGE_MODEL
    judge_max_tokens = HLE_JUDGE_MAX_TOKENS

    def __init__(self, config: TaskConfig) -> None:
        # The project-wide environment override applies only where the task's own
        # judge model was left at its default, so ``-o judge_model=...`` wins.
        spec = os.getenv("OLMO_EVAL_JUDGE")
        if spec and config.judge_model == type(self).judge_model:
            model, reasoning_effort = _parse_judge_spec(spec)
            config = replace(
                config,
                judge_model=model,
                judge_reasoning_effort=config.judge_reasoning_effort or reasoning_effort,
            )
        super().__init__(config)

    @property
    def instances(self) -> Iterator[Instance]:
        yield from self._load_instances_cached()

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        question = str(doc.get("question") or "").strip()
        answer = str(doc.get("answer") or "").strip()
        # Image questions need the image to be answerable; these tasks are text-only.
        if not question or not answer or doc.get("image"):
            return None

        category = str(doc.get("category") or "")
        if category not in HLE_CATEGORIES:
            logger.warning("HLE: unexpected category %r at index %d", category, index)
        return Instance(
            question=question,
            gold_answer=answer,
            metadata={
                "id": str(doc.get("id") or f"hle:{index}"),
                "category": category,
                "raw_subject": str(doc.get("raw_subject") or ""),
                "answer_type": str(doc.get("answer_type") or ""),
                "index": index,
            },
        )

    def format_request(self, instance: Instance) -> LMRequest:
        return LMRequest(
            request_type=RequestType.CHAT,
            messages=(
                {"role": "system", "content": HLE_SYSTEM_PROMPT},
                {"role": "user", "content": instance.question},
            ),
        )

    def extract_answer(self, output: LMOutput) -> str:
        """Hand the judge the visible response; the judge extracts the answer itself."""
        return visible_response(output.text)

    def _get_judge_fn(self) -> JudgeFn:
        """Return this task's judge, built once so its request-shape cache persists."""
        judge_fn = getattr(self, "_judge_fn", None)
        if judge_fn is None:
            if self.config.judge_model is None or self.config.judge_max_tokens is None:
                raise ValueError("HLE requires a complete judge configuration")
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
        """Return the judge limiter, shared by every HLE task in the run when possible."""
        owner = context if context is not None else self
        semaphore = getattr(owner, _JUDGE_SEMAPHORE_ATTR, None)
        if semaphore is None:
            concurrency = getattr(context, "scoring_concurrency", HLE_DEFAULT_SCORING_CONCURRENCY)
            semaphore = asyncio.Semaphore(max(1, concurrency))
            setattr(owner, _JUDGE_SEMAPHORE_ATTR, semaphore)
        return semaphore

    async def _judge(self, judge_fn: JudgeFn, prompt: str) -> tuple[JudgeVerdict | None, str]:
        """Call the judge, retrying API failures and unparseable replies."""
        raw = ""
        for attempt in range(HLE_JUDGE_ATTEMPTS):
            raw = await retry_with_backoff(lambda: judge_fn(prompt), context="hle judge")
            verdict = parse_judge_reply(raw)
            if verdict is not None:
                return verdict, raw
            if attempt < HLE_JUDGE_ATTEMPTS - 1:
                await asyncio.sleep(2**attempt)
        return None, raw

    async def _score_output(
        self, response: Response, output: LMOutput, judge_fn: JudgeFn
    ) -> dict[str, float]:
        """Judge one output and record its scores; raise if the judge gave no verdict."""
        answer = output.extracted_answer
        if not isinstance(answer, str):
            answer = visible_response(output.text)

        truncated = output.metadata.get("finish_reason") == "length"
        judge_result: dict[str, Any] = {"scorer": HLE_SCORER_NAME, "truncated": truncated}
        verdict: JudgeVerdict | None
        if truncated or not answer:
            # Nothing finished to grade. Score it wrong at 100% confidence, as the judge
            # prompt directs for a response that states no confidence.
            verdict = JudgeVerdict(correct=False, confidence=1.0, extracted_final_answer=None)
            judge_result["no_final_response"] = True
        else:
            prompt = HLE_JUDGE_PROMPT.format(
                question=response.instance.question,
                response=answer,
                correct_answer=response.instance.gold_answer or "",
            )
            verdict, raw = await self._judge(judge_fn, prompt)
            judge_result["raw_judge_response"] = raw

        if verdict is None:
            judge_result["parse_error"] = True
            output.metadata["judge_result"] = judge_result
            raise HLEJudgeParseError(
                f"no parseable verdict after {HLE_JUDGE_ATTEMPTS} judge attempts"
            )

        judge_result.update(
            correct=verdict.correct,
            confidence=verdict.confidence,
            extracted_final_answer=verdict.extracted_final_answer,
            parse_error=False,
        )
        output.metadata["judge_result"] = judge_result
        scores = {"accuracy": 1.0 if verdict.correct else 0.0, "confidence": verdict.confidence}
        for key, value in scores.items():
            output.metadata[f"score:{key}"] = value
        _store_output_score(output, scorer_name=HLE_SCORER_NAME, score=scores["accuracy"])
        return scores

    async def score_responses(
        self,
        responses: Sequence[Response],
        context: Any = None,
    ) -> Sequence[Response]:
        """Judge every output concurrently, then average each instance over its samples."""
        self._extract_answers(responses)
        judge_fn = self._get_judge_fn()
        semaphore = self._get_judge_semaphore(context)

        truncated = sum(
            1
            for response in responses
            for output in response.outputs
            if output.metadata.get("finish_reason") == "length"
        )
        if truncated:
            logger.warning(
                "HLE generation hit the token limit on %d output(s); they score wrong "
                "without judging. Check truncation_rate before reading accuracy as capability.",
                truncated,
            )
        unanswered = [response for response in responses if not response.outputs]
        if unanswered:
            logger.warning(
                "HLE got no model output for %d question(s), e.g. %s; they score wrong. A "
                "prompt longer than the context leaves room for no answer.",
                len(unanswered),
                unanswered[0].instance.metadata.get("id"),
            )

        jobs = [
            (response_index, output_index)
            for response_index, response in enumerate(responses)
            for output_index in range(len(response.outputs))
        ]

        async def run(response_index: int, output_index: int) -> dict[str, float]:
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
                        f"HLE judge call failed: {type(exc).__name__}: {exc}"
                    ) from exc

        results = await asyncio.gather(*(run(*job) for job in jobs), return_exceptions=True)

        accuracy_by_response: list[dict[int, float]] = [{} for _ in responses]
        judge_errors: list[Exception] = []
        failed_by_response = [0] * len(responses)
        for (response_index, output_index), scores in zip(jobs, results, strict=True):
            if isinstance(scores, BaseException):
                if not isinstance(scores, Exception):
                    raise scores
                judge_errors.append(scores)
                failed_by_response[response_index] += 1
                _store_output_score(
                    responses[response_index].outputs[output_index],
                    scorer_name=HLE_SCORER_NAME,
                    score=0.0,
                    scoring_error=_format_scoring_error(scores, phase="judge"),
                )
                continue
            accuracy_by_response[response_index][output_index] = scores["accuracy"]

        if judge_errors:
            logger.warning(
                "HLE judging failed on %d/%d output(s) (API errors or no parseable verdict); "
                "they are recorded as scoring errors and excluded from their instance average. "
                "First error: %s",
                len(judge_errors),
                len(jobs),
                judge_errors[0],
            )
        for response_index, response in enumerate(responses):
            if response.outputs and failed_by_response[response_index] == len(response.outputs):
                raise ScoringIncompleteError(
                    "HLE judging failed for every sample of instance "
                    f"{response.instance.metadata.get('id', response_index)}: {judge_errors[0]}"
                ) from judge_errors[0]

        for response, accuracy in zip(responses, accuracy_by_response, strict=True):
            response.scores["accuracy"] = self._aggregate_output_scores(accuracy)
            response.scores["truncation_rate"] = (
                sum(output.metadata.get("finish_reason") == "length" for output in response.outputs)
                / len(response.outputs)
                if response.outputs
                else 1.0
            )
        return responses


@register("hle:text")
class HLEText(_HLE):
    """Humanity's Last Exam, the questions without an image."""

    data_source = DataSource(path=HLE_REPO, revision=HLE_REVISION, split="test")


@register("hle:text:verified")
class HLETextVerified(_HLE):
    """The text-only questions in the HLE-Verified Gold subset."""

    data_source = DataSource(
        path=HLE_VERIFIED_REPO,
        data_files=HLE_VERIFIED_GOLD_FILES,
        revision=HLE_VERIFIED_REVISION,
        split="train",
    )
    split = Split.TRAIN  # Parquet data files load as a single "train" split

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        # The flat columns omit the image and answer type; the full record is in ``json``.
        record = json.loads(doc["json"]) if doc.get("json") else doc
        if record.get("Verified_Classes", HLE_VERIFIED_GOLD_CLASS) != HLE_VERIFIED_GOLD_CLASS:
            return None
        return super().process_doc(record, index)
