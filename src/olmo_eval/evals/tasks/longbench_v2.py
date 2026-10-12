"""LongBench v2 long-context multiple-choice tasks.

LongBench v2 (zai-org/LongBench-v2) has 503 four-way multiple-choice questions
whose contexts range from 8k to 2M words across six domains: single-document
QA, multi-document QA, long in-context learning, long-dialogue history
understanding, code repository understanding, and long structured data
understanding. The prompt and answer extraction follow the reference
implementation's zero-shot (non-CoT) setting so the full-suite number is
comparable with published results.

Tasks:
    longbench_v2       All 503 questions. Reports the standard full-suite
                       accuracy alongside accuracy on the held-out subset that
                       excludes the development domains.
    longbench_v2_dev   The code repository and structured data questions only,
                       for use in the development loop.

Paper: https://arxiv.org/abs/2412.15204
Reference implementation: https://github.com/THUDM/LongBench
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

from olmo_eval.common.metrics import AccuracyMetric, Metric
from olmo_eval.common.scorers import MultipleChoiceScorer
from olmo_eval.common.scorers.base import Scorer
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
from olmo_eval.evals.tasks.common import Task, register

logger = logging.getLogger(__name__)

LONGBENCH_V2_REPO = "zai-org/LongBench-v2"
#: Dataset revision, pinned so the question set behind a stored result cannot change.
LONGBENCH_V2_REVISION = "2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9"

#: Domains held back for the development loop. Questions in these domains are
#: excluded from the held-out score reported by the full task.
DEV_DOMAINS: frozenset[str] = frozenset(
    {
        "Code Repository Understanding",
        "Long Structured Data Understanding",
    }
)

#: Metadata key recording whether an instance belongs to the dev or held-out group.
SPLIT_GROUP_KEY = "split_group"
DEV_GROUP = "dev"
HELDOUT_GROUP = "heldout"

_LETTERS = ("A", "B", "C", "D")

# Zero-shot prompt from the reference implementation (prompts/0shot.txt).
_PROMPT_TEMPLATE = (
    "Please read the following text and answer the question below.\n"
    "\n"
    "<text>\n"
    "{context}\n"
    "</text>\n"
    "\n"
    "What is the correct answer to this question: {question}\n"
    "Choices:\n"
    "(A) {choice_a}\n"
    "(B) {choice_b}\n"
    "(C) {choice_c}\n"
    "(D) {choice_d}\n"
    "\n"
    'Format your response as follows: "The correct answer is (insert answer here)".'
)

_ANSWER_PAREN = re.compile(r"The correct answer is \(([A-D])\)")
_ANSWER_BARE = re.compile(r"The correct answer is ([A-D])")


def extract_answer(text: str) -> str | None:
    """Return the answer letter using the reference implementation's extraction.

    Markdown emphasis is dropped first, then the parenthesized form is tried
    before the bare letter. Responses without the requested phrasing yield
    ``None`` and score as incorrect, as in the reference scoring.
    """
    text = text.replace("*", "")
    for pattern in (_ANSWER_PAREN, _ANSWER_BARE):
        match = pattern.search(text)
        if match:
            return match.group(1)
    return None


@dataclass(frozen=True, slots=True)
class SplitGroupAccuracyMetric(Metric):
    """Mean accuracy over the instances in one split group.

    Lets a single run of the full benchmark also report the score on a subset,
    so the dev and held-out numbers come from the same predictions.
    """

    name: str = "heldout_accuracy"
    scorer: type[Scorer] | Scorer = MultipleChoiceScorer
    split_group: str = HELDOUT_GROUP

    def _in_group(self, response: Response) -> bool:
        return response.instance.metadata.get(SPLIT_GROUP_KEY) == self.split_group

    def compute(self, responses: Sequence[Response]) -> float:
        scorer_name = self.scorer().name
        scores = [r.scores.get(scorer_name, 0.0) for r in responses if self._in_group(r)]
        if not scores:
            logger.warning(
                "%s: no instances in split group %r, reporting 0.0. This happens when a "
                "limited run samples no question from the group; the value is not a score.",
                self.name,
                self.split_group,
            )
            return 0.0
        return sum(scores) / len(scores)

    def compute_instance(self, response: Response) -> float | None:
        if not self._in_group(response):
            return None
        score = response.scores.get(self.scorer().name)
        return float(score) if score is not None else None

    def supports_scorer_fallback(self) -> bool:
        return False

    def to_dict(self) -> dict[str, Any]:
        """Serialize including the group, so differently scoped metrics hash differently."""
        # Named base call, not super(): a slots dataclass is a rebuilt class, so the
        # zero-argument form resolves against a stale __class__ cell before Python 3.13.
        return {**Metric.to_dict(self), "split_group": self.split_group}


_ACCURACY = AccuracyMetric(scorer=MultipleChoiceScorer)
_HELDOUT_ACCURACY = SplitGroupAccuracyMetric(name="heldout_accuracy", split_group=HELDOUT_GROUP)
_DEV_ACCURACY = SplitGroupAccuracyMetric(name="dev_accuracy", split_group=DEV_GROUP)

# Sampling follows the reference implementation's direct-answer setting. The
# prompt is left-truncated to the serving model's context window, so questions
# longer than the window keep the question and choices at the end of the prompt.
_SAMPLING = SamplingParams(
    max_tokens=128,
    temperature=0.1,
    truncate_prompt_tokens=-1,
    truncation_side="left",
)


class LongBenchV2Task(Task):
    """Base class for LongBench v2 tasks."""

    data_source = DataSource(path=LONGBENCH_V2_REPO, revision=LONGBENCH_V2_REVISION)
    split = Split.TRAIN  # HF dataset only has a train split
    metrics = (_ACCURACY, _HELDOUT_ACCURACY, _DEV_ACCURACY)
    primary_metric = _ACCURACY
    sampling_params = _SAMPLING

    #: Restrict instances to these domains; ``None`` keeps every question.
    domains: frozenset[str] | None = None

    @property
    def request_type(self) -> RequestType:
        return RequestType.CHAT

    @property
    def instances(self) -> Iterator[Instance]:
        yield from self._load_instances_cached()

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        domain = str(doc.get("domain") or "")
        if self.domains is not None and domain not in self.domains:
            return None

        question = str(doc.get("question") or "").strip()
        choices = tuple(str(doc.get(f"choice_{letter}") or "").strip() for letter in _LETTERS)
        answer = str(doc.get("answer") or "").strip().upper()
        if not question or not all(choices) or answer not in _LETTERS:
            logger.warning(
                "Skipping malformed LongBench v2 document %r: question, four choices, and an "
                "A-D answer are all required.",
                doc.get("_id", index),
            )
            return None

        prompt = _PROMPT_TEMPLATE.format(
            context=str(doc.get("context") or "").strip(),
            question=question,
            choice_a=choices[0],
            choice_b=choices[1],
            choice_c=choices[2],
            choice_d=choices[3],
        )

        return Instance(
            question=prompt,
            gold_answer=answer,
            choices=choices,
            metadata={
                "id": doc.get("_id", index),
                "index": index,
                "domain": domain,
                "sub_domain": doc.get("sub_domain", ""),
                "difficulty": doc.get("difficulty", ""),
                "length": doc.get("length", ""),
                SPLIT_GROUP_KEY: DEV_GROUP if domain in DEV_DOMAINS else HELDOUT_GROUP,
            },
        )

    def format_request(self, instance: Instance) -> LMRequest:
        return LMRequest(
            request_type=RequestType.CHAT,
            messages=({"role": "user", "content": instance.question},),
        )

    def extract_answer(self, output: LMOutput) -> str | None:
        return extract_answer(output.text or "")


@register("longbench_v2")
class LongBenchV2(LongBenchV2Task):
    """Full LongBench v2 with full-suite, held-out, and dev-subset accuracy."""


@register("longbench_v2_dev")
class LongBenchV2Dev(LongBenchV2Task):
    """Development slice: code repository and long structured data questions."""

    domains = DEV_DOMAINS
    metrics = (_ACCURACY,)
    primary_metric = _ACCURACY
