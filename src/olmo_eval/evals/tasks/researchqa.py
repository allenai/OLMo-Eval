"""ResearchQA agentic scholarly question answering (arXiv 2509.00496).

The model searches the scholarly web and writes an approximately 250-word
answer. Coverage is judged with the verbatim prompt from the official
``compute_coverage.py`` scorer: rubric items are sent in batches of at most
eight and receive one of five ordinal labels, normalized onto a 0--1 scale.
The primary metric is an unweighted mean of per-answer rubric means. The paper
reports this value multiplied by 100; for example, its best system's 75.29%
coverage is ``0.7529`` here. Per-type metrics pool rubric items across answers,
so answers with more items of a type receive proportionally more weight.

Unlike the official scorer, which drops an entire answer after three malformed
judge outputs, this task assigns the affected rubric items 0.0 and keeps the
answer in the denominator. This prevents parse failures from inflating scores.

The deliberate project default, ``gpt-5.4-mini``, deviates from the paper's
official ``gpt-4.1-mini`` judge at temperature 0. To reproduce the published
leaderboard exactly, set ``OLMO_EVAL_JUDGE=gpt-4.1-mini``.
"""

from __future__ import annotations

import asyncio
import logging
import os
from abc import ABC
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any, cast

from olmo_eval.common.metrics import Metric
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
from olmo_eval.evals.tasks.common import Task, register, register_variant

logger = logging.getLogger(__name__)

RESEARCHQA_REPO = "realliyifei/ResearchQA"
RESEARCHQA_DEFAULT_JUDGE_SPEC = "gpt-5.4-mini"
RESEARCHQA_RUBRIC_BATCH_SIZE = 8
RESEARCHQA_JUDGE_ATTEMPTS = 3

RESEARCHQA_TYPE_METRICS = {
    "coverage_citation": "Citation",
    "coverage_limitation": "Limitation",
    "coverage_comparison": "Comparison",
    "coverage_example": "Example",
    "coverage_impact": "Impact",
    "coverage_other": "Other",
}

RESEARCHQA_LABEL_SCORES = {
    "Not at all": 1,
    "Barely": 2,
    "Moderately": 3,
    "Mostly": 4,
    "Completely": 5,
}

RESEARCHQA_GENERATION_PROMPT = (
    "You are answering a scholarly research question. Use the available research tools "
    "to find reliable sources that support your answer.\n\n"
    "Available tools:\n"
    "- semantic_scholar_snippet_search\n"
    "- serper_google_webpage_search\n"
    "- serper_fetch_webpage_content\n\n"
    "Format and length requirements:\n"
    "- Write a well-organized, citation-supported answer in about 250 words.\n"
    "- Synthesize the evidence directly around the question and cite the sources you use.\n\n"
    "{date_cutoff}"
    "Workflow (follow this order):\n"
    "1. Search for relevant scholarly sources using semantic_scholar_snippet_search and "
    "serper_google_webpage_search.\n"
    "2. Fetch and read promising source content using serper_fetch_webpage_content.\n"
    "3. Only after searching, fetching, and reading, write the answer.\n"
    "Search first, then fetch and read, and only then answer. Do not answer from memory.\n\n"
    "Question: "
)

# Verbatim from ResearchQA's official compute_coverage.py:build_prompt().
RESEARCHQA_COVERAGE_JUDGE_PROMPT = (
    "Please judge the following questions based on the response below.\n"
    "For each question, select one of the following ratings to indicate the extent to which "
    "the response addresses the question:\n"
    "Not at all, Barely, Moderately, Mostly, Completely\n\n"
    "Definitions:\n"
    "- Not at all: *totally uninferable*\n"
    "- Barely: *unmentioned but inferrable*\n"
    "- Moderately: *mentioned but misses important details*\n"
    "- Mostly: *mentioned but misses some details*\n"
    "- Completely: *mentioned with sufficient details*\n\n"
    "Only output one of the five phrases for each question, separated by newlines, and "
    "nothing else.\n\n"
    "Response: {answer}\n"
    "Questions:\n"
    "{questions}\n\n"
    "Output:"
)


def normalize_coverage_label(label: str) -> float:
    """Map an official ResearchQA judge label onto the 0--1 coverage scale."""
    return (RESEARCHQA_LABEL_SCORES[label] - 1) / 4


def build_coverage_judge_prompt(answer: str, rubric_items: Sequence[str]) -> str:
    """Format one official coverage-judge batch with raw newline-separated items."""
    questions = "\n".join(rubric_items)
    return RESEARCHQA_COVERAGE_JUDGE_PROMPT.format(answer=answer, questions=questions)


def parse_coverage_labels(raw: str, expected_count: int) -> list[str] | None:
    """Parse the official newline-separated output, rejecting malformed batches."""
    labels = [line.strip() for line in raw.splitlines() if line.strip()]
    if len(labels) != expected_count:
        return None
    if any(label not in RESEARCHQA_LABEL_SCORES for label in labels):
        return None
    return labels


def build_researchqa_judge_fn() -> JudgeFn:
    """Build the task-local judge from ``$OLMO_EVAL_JUDGE`` or its own default."""
    spec = os.getenv("OLMO_EVAL_JUDGE", RESEARCHQA_DEFAULT_JUDGE_SPEC)
    model, separator, effort = spec.partition(":")
    return build_openai_judge_fn(
        model=model,
        temperature=0.0,
        # A reasoning judge can spend this budget and truncate the label list; that batch scores 0.
        max_tokens=4096,
        scorer_name="ResearchQA",
        reasoning_effort=(effort if separator else None),
    )


def _type_item_count(response: Response, rubric_type: str) -> int:
    rubric = response.instance.metadata.get("rubric", [])
    if not isinstance(rubric, list):
        return 0
    return sum(
        1
        for item in rubric
        if isinstance(item, dict)
        and isinstance(item.get("type"), list)
        and rubric_type in item["type"]
    )


@dataclass(frozen=True)
class ResearchQAScorer(Scorer):
    """Placeholder scorer; ResearchQA scores are computed in ``score_responses``."""

    name: str = "researchqa"
    score_key: str = "rubric_coverage"

    def score(self, instance: Instance, output: LMOutput) -> float:
        return (output.metadata or {}).get(self.score_key, 0.0)


class ResearchQAMetricBase(Metric, ABC):
    """Base for ResearchQA metrics precomputed by the task."""

    scorer: type[Scorer] = ResearchQAScorer

    def display_format(self) -> str:
        return "percentage"

    def metric_unit(self) -> str:
        return "proportion"


@dataclass(frozen=True)
class RubricCoverageMetric(ResearchQAMetricBase):
    """Mean per-answer rubric coverage, matching the official aggregation."""

    name: str = "rubric_coverage"
    scorer: type[Scorer] = ResearchQAScorer

    def compute(self, responses: Sequence[Response]) -> float:
        if not responses:
            return 0.0
        return sum(response.scores.get(self.name, 0.0) for response in responses) / len(responses)


@dataclass(frozen=True)
class CoverageTypeMetric(ResearchQAMetricBase):
    """Coverage pooled over all rubric items carrying one type label."""

    name: str = ""
    scorer: type[Scorer] = ResearchQAScorer
    rubric_type: str = ""

    def compute(self, responses: Sequence[Response]) -> float:
        weighted_score = 0.0
        item_count = 0
        for response in responses:
            response_item_count = _type_item_count(response, self.rubric_type)
            weighted_score += response.scores.get(self.name, 0.0) * response_item_count
            item_count += response_item_count
        return weighted_score / item_count if item_count else 0.0

    def supports_scorer_fallback(self) -> bool:
        return False


RUBRIC_COVERAGE_METRIC = RubricCoverageMetric()
RESEARCHQA_METRICS = (
    RUBRIC_COVERAGE_METRIC,
    *(
        CoverageTypeMetric(name=metric_name, rubric_type=rubric_type)
        for metric_name, rubric_type in RESEARCHQA_TYPE_METRICS.items()
    ),
)


def _validated_rubric(doc: dict[str, Any]) -> list[dict[str, Any]] | None:
    """Validate the nested HF schema without normalizing scientific text."""
    raw_rubric = doc.get("rubric")
    if not isinstance(raw_rubric, list) or not raw_rubric:
        return None

    rubric: list[dict[str, Any]] = []
    for item_index, raw_item in enumerate(raw_rubric):
        if not isinstance(raw_item, dict):
            raise TypeError(f"rubric[{item_index}] must be a mapping")
        item = cast(dict[str, Any], raw_item)

        rubric_item = item.get("rubric_item")
        if not isinstance(rubric_item, str) or not rubric_item:
            raise ValueError(f"rubric[{item_index}].rubric_item must be a non-empty string")

        rubric_types = item.get("type")
        if not isinstance(rubric_types, list) or not all(
            isinstance(rubric_type, str) for rubric_type in rubric_types
        ):
            raise TypeError(f"rubric[{item_index}].type must be a list of strings")

        citation_metadata = item.get("citation_metadata")
        if citation_metadata is not None and not isinstance(citation_metadata, dict):
            raise TypeError(f"rubric[{item_index}].citation_metadata must be a mapping or None")

        rubric.append(
            {
                "rubric_item": rubric_item,
                "type": list(rubric_types),
                "citation_metadata": (
                    dict(citation_metadata) if citation_metadata is not None else None
                ),
            }
        )
    return rubric


@register("researchqa")
class ResearchQA(Task):
    """ResearchQA generation with a deliberately non-paper-default coverage judge.

    The default is ``gpt-5.4-mini``; the paper's official judge is
    ``gpt-4.1-mini`` at temperature 0. Set
    ``OLMO_EVAL_JUDGE=gpt-4.1-mini`` for exact published-leaderboard
    reproduction.
    """

    split = Split.TEST
    data_source = DataSource(path=RESEARCHQA_REPO, subset="default", split="test")
    metrics = RESEARCHQA_METRICS
    primary_metric = RUBRIC_COVERAGE_METRIC
    sampling_params = SamplingParams(temperature=0.0, max_tokens=4096)
    required_secrets = ("OPENAI_API_KEY",)

    @property
    def instances(self) -> Iterator[Instance]:
        split = (
            self.config.data_source.split
            if isinstance(self.config.data_source, DataSource)
            else None
        )
        yield from self._load_instances_cached(split=split)

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        query = doc.get("query")
        if not isinstance(query, str) or not query:
            return None

        rubric = _validated_rubric(doc)
        if rubric is None:
            return None

        researchqa_id = doc.get("id")
        if not isinstance(researchqa_id, str) or not researchqa_id:
            researchqa_id = f"researchqa_{index}"

        date = doc.get("date")
        return Instance(
            question=query,
            metadata={
                "id": researchqa_id,
                "case_id": researchqa_id,
                "general_domain": doc.get("general_domain", ""),
                "subdomain": doc.get("subdomain", ""),
                "field": doc.get("field", ""),
                "date": None if date is None else str(date),
                "retrieval_date_cutoff": None if date is None else str(date),
                "rubric": rubric,
                "index": index,
            },
        )

    def format_request(self, instance: Instance) -> LMRequest:
        date = instance.metadata.get("date")
        date_cutoff = (
            f"Prefer sources available on or before {date}.\n\n"
            if isinstance(date, str) and date
            else ""
        )
        generation_prompt = RESEARCHQA_GENERATION_PROMPT.format(date_cutoff=date_cutoff)
        return LMRequest(
            request_type=RequestType.CHAT,
            messages=(
                {
                    "role": "user",
                    "content": generation_prompt + instance.question,
                },
            ),
        )

    def extract_answer(self, output: LMOutput) -> Any:
        """Strip a leading ``<think>...</think>`` reasoning block before scoring."""
        text = output.text
        think_end = text.find("</think>")
        if think_end >= 0:
            text = text[think_end + len("</think>") :]
        return text.strip()

    async def score_responses(
        self,
        responses: Sequence[Response],
        context: Any = None,
    ) -> Sequence[Response]:
        """Judge rubric coverage, retaining zero-scored parse failures."""
        self._extract_answers(responses)
        judge_fn = build_researchqa_judge_fn()

        failed_batches = 0
        failed_items = 0
        affected_instances = 0
        for response in responses:
            scores, labels, item_scores, response_failed_batches = await self._score_single(
                response, judge_fn
            )
            response.scores.update(scores)
            failed_batches += response_failed_batches
            if response_failed_batches:
                affected_instances += 1
                failed_items += sum(label is None for label in labels)

            if response.outputs:
                output = response.outputs[0]
                output.metadata["researchqa_rubric_labels"] = labels
                output.metadata["researchqa_rubric_scores"] = item_scores
                for metric_name, score in scores.items():
                    output.metadata[f"score:{metric_name}"] = score

        if failed_batches:
            logger.warning(
                "ResearchQA judge had %d unparseable batch(es) covering %d rubric item(s) "
                "across %d instance(s) after %d attempts; assigned those items 0.0 and "
                "kept every instance in the denominator.",
                failed_batches,
                failed_items,
                affected_instances,
                RESEARCHQA_JUDGE_ATTEMPTS,
            )
        return responses

    async def _score_single(
        self,
        response: Response,
        judge_fn: JudgeFn,
    ) -> tuple[dict[str, float], list[str | None], list[float], int]:
        """Score one answer and return metrics plus per-rubric judge details."""
        rubric = response.instance.metadata.get("rubric", [])
        if not isinstance(rubric, list) or not rubric:
            return self._scores_from_items([], []), [], [], 0

        output = response.outputs[0] if response.outputs else None
        if output is None:
            zeros = [0.0] * len(rubric)
            return self._scores_from_items(rubric, zeros), [None] * len(rubric), zeros, 0

        answer = output.extracted_answer
        if not isinstance(answer, str):
            answer = output.text

        all_labels: list[str | None] = []
        all_scores: list[float] = []
        failed_batches = 0
        for start in range(0, len(rubric), RESEARCHQA_RUBRIC_BATCH_SIZE):
            batch = rubric[start : start + RESEARCHQA_RUBRIC_BATCH_SIZE]
            rubric_items = [item["rubric_item"] for item in batch]
            prompt = build_coverage_judge_prompt(answer, rubric_items)

            labels = None
            for attempt in range(RESEARCHQA_JUDGE_ATTEMPTS):
                raw = await judge_fn(prompt)
                labels = parse_coverage_labels(raw, len(batch))
                if labels is not None:
                    break
                if attempt < RESEARCHQA_JUDGE_ATTEMPTS - 1:
                    await asyncio.sleep(2**attempt)

            if labels is None:
                failed_batches += 1
                all_labels.extend([None] * len(batch))
                all_scores.extend([0.0] * len(batch))
            else:
                all_labels.extend(labels)
                all_scores.extend(normalize_coverage_label(label) for label in labels)

        return (
            self._scores_from_items(rubric, all_scores),
            all_labels,
            all_scores,
            failed_batches,
        )

    @staticmethod
    def _scores_from_items(
        rubric: Sequence[dict[str, Any]], item_scores: Sequence[float]
    ) -> dict[str, float]:
        primary = sum(item_scores) / len(rubric) if rubric else 0.0
        scores = {"rubric_coverage": primary}
        for metric_name, rubric_type in RESEARCHQA_TYPE_METRICS.items():
            type_scores = [
                score
                for item, score in zip(rubric, item_scores, strict=True)
                if rubric_type in item["type"]
            ]
            scores[metric_name] = sum(type_scores) / len(type_scores) if type_scores else 0.0
        return scores


register_variant(
    "researchqa",
    "mini",
    data_source=DataSource(
        path=RESEARCHQA_REPO,
        subset="default",
        split="test_mini",
    ),
)
