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

import json
import logging
import math
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

from olmo_eval.common.metrics import Metric
from olmo_eval.common.scorers.base import Scorer
from olmo_eval.common.scorers.llm_judge import JudgeFn
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
from olmo_eval.evals.tasks.common import OutputScoreAggregation, register
from olmo_eval.evals.tasks.common.judge import (
    JudgedScoreMetric,
    JudgedTask,
    JudgeScorer,
    TruncationRateMetric,
    is_truncated,
)

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


@dataclass(frozen=True)
class HLEScorer(JudgeScorer):
    """Serves the HLE judge's stored per-output accuracy."""

    name: str = HLE_SCORER_NAME


@dataclass(frozen=True)
class HLEAccuracyMetric(JudgedScoreMetric):
    """Mean judged accuracy, optionally restricted to instances with one metadata value."""

    scorer: type[Scorer] | Scorer = HLEScorer


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

    def supports_scorer_fallback(self) -> bool:
        return False

    def higher_is_better(self) -> bool:
        return False

    def display_format(self) -> str:
        return "percentage"

    def metric_unit(self) -> str:
        return "proportion"


@dataclass(frozen=True)
class HLETruncationRateMetric(TruncationRateMetric):
    """Share of questions with no complete response; lower is better."""

    scorer: type[Scorer] | Scorer = HLEScorer


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


class _HLE(JudgedTask):
    """Shared HLE loading, prompting, and judging for the text-only tasks."""

    split = Split.TEST
    metrics = HLE_METRICS
    primary_metric = HLE_ACCURACY
    sampling_params = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=32768)
    output_score_aggregation = OutputScoreAggregation.MEAN

    judge_model = HLE_DEFAULT_JUDGE_MODEL
    judge_max_tokens = HLE_JUDGE_MAX_TOKENS
    judge_scorer_name = HLE_SCORER_NAME

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

    async def _score_output(self, response: Response, output: LMOutput, judge_fn: JudgeFn) -> float:
        """Judge one output; an unfinished response scores wrong without the judge."""
        answer = self._judged_answer(output)
        truncated = is_truncated(output)
        judge_result: dict[str, Any] = {"scorer": HLE_SCORER_NAME, "truncated": truncated}
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
            verdict = await self._judge_output(
                output, judge_fn, prompt, parse_judge_reply, judge_result
            )

        judge_result.update(
            correct=verdict.correct,
            confidence=verdict.confidence,
            extracted_final_answer=verdict.extracted_final_answer,
            parse_error=False,
        )
        return self._record_judgement(
            output,
            judge_result,
            {"accuracy": 1.0 if verdict.correct else 0.0, "confidence": verdict.confidence},
        )


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
