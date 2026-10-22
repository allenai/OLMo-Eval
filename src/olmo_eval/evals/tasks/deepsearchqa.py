"""DeepSearchQA: deep-research agent evaluation (Google DeepMind, arXiv 2601.20975).

DeepSearchQA is a 900-problem benchmark of hand-crafted, multi-step
information-seeking questions spanning 17 domains (Politics, Finance, Science,
Health, History, Geography, Media, ...). Each problem is either a
``Single Answer`` (one entity/value) or a ``Set Answer`` (an enumeration or
composite answer with multiple required items); on HuggingFace both are
stored as one comma-joined ``answer`` string
(``google/deepsearchqa``, config ``deepsearchqa``, split ``eval``). A handful
of ``Set Answer`` rows encode "no items satisfy every constraint" as the
literal text ``None``, which HuggingFace's CSV loader turns into a null; these
are scored as an empty gold answer set rather than dropped.

Grading follows the paper's outcome-based, set-comparison methodology: an
LLM judge decides, per item, whether a submitted answer is semantically
equivalent to a ground-truth answer (and vice versa), and the task reports
Precision (``|S ∩ G| / |S|``), Recall (``|S ∩ G| / |G|``), their harmonic
mean F1 (the primary metric), and an all-or-nothing Exact Match rate. The
official grader (Gemini 2.5 Flash, zero-shot) and its exact prompt *are*
published, in Appendix A of the technical report. This task nonetheless uses
an independently written judge prompt against olmo-eval's OpenAI-based judge
infrastructure (``build_openai_judge_fn`` is OpenAI-only), so absolute
numbers are not directly comparable to the public leaderboard: the official
prompt grades the model's raw free-form response directly, while this task
requires a discrete ``FINAL ANSWER: ...`` line and matches two pre-extracted
item lists by index. See ``deepsearchqa_official_judge`` for a second task
that instead mirrors the official prompt and grading mechanic. Set
``OLMO_EVAL_JUDGE=<model>[:<effort>]`` to change the judge model.

The task asks the model to end its response with a ``FINAL ANSWER: ...`` line
listing every item in its answer set, comma-separated; this line (rather than
the full response) is what gets judged. This benchmark is about finding
information via search rather than about knowledge already in the model's
parameters, so it is intended to be run with search tools attached via the
Harness abstraction (e.g. ``--harness dr_tulu``); running without tools mostly
measures parametric recall on questions designed to require live search.

Usage:
    # Tool-augmented (intended way to run this benchmark)
    olmo-eval run -m llama3.1-8b -t deepsearchqa --harness dr_tulu

    # Baseline, no search tools (for reference only)
    olmo-eval run -m llama3.1-8b -t deepsearchqa

    # Quick iteration on a 50-instance subset
    olmo-eval run -m llama3.1-8b -t deepsearchqa:mini --harness dr_tulu
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

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
)
from olmo_eval.data import DataSource
from olmo_eval.evals.tasks.common import Task, register, register_variant

logger = logging.getLogger(__name__)

DEEPSEARCHQA_REPO = "google/deepsearchqa"
DEEPSEARCHQA_CONFIG = "deepsearchqa"
DEEPSEARCHQA_SPLIT = "eval"

DEEPSEARCHQA_DEFAULT_JUDGE_SPEC = "gpt-5.5:medium"
DEEPSEARCHQA_JUDGE_ATTEMPTS = 3

DEEPSEARCHQA_GENERATION_PROMPT = """\
Answer the following question. It may require an exhaustive search across \
multiple sources rather than a single lookup. If you have access to search \
tools, use them to find and verify each candidate answer instead of \
answering from memory alone.

The question may have exactly one correct answer, or it may require a \
complete set of items that together satisfy every constraint it states \
(watch for words like "all", "each", or "list"). Do not stop at the first \
plausible answer if the question implies there may be more than one.

Finish your response with a line of exactly this form:
FINAL ANSWER: <item 1>, <item 2>, ...

List every item belonging to your answer set on that line, separated by \
commas, and put nothing else on it. If there is a single correct answer, \
list just that one item.

Question: {question}"""

# Tolerates markdown emphasis and a missing/extra colon around the marker.
_FINAL_ANSWER_MARKER = re.compile(r"\**\s*FINAL\s+ANSWER\s*\**\s*:?\s*", re.IGNORECASE)


def extract_final_answer(text: str) -> str:
    """Return the text after the last ``FINAL ANSWER`` marker.

    Falls back to the full response when the marker is absent, so the judge
    still sees the model's attempt.
    """
    matches = list(_FINAL_ANSWER_MARKER.finditer(text))
    if not matches:
        return text.strip()
    return text[matches[-1].end() :].strip() or text.strip()


def split_answer_set(text: str) -> list[str]:
    """Split a comma-delimited answer string into non-empty, stripped items."""
    return [item.strip() for item in text.split(",") if item.strip()]


def _numbered_list(items: Sequence[str]) -> str:
    return "\n".join(f"{index}. {item}" for index, item in enumerate(items))


DEEPSEARCHQA_JUDGE_PROMPT = """\
You are grading a submitted answer against a ground-truth answer for the \
question below. The answer may be a single item or a set of items. Two \
items are equivalent if they refer to the same real-world entity, value, or \
fact, even when worded differently (abbreviations, alternate spellings, \
reordering, or trivial formatting differences do not matter). Judge \
semantic equivalence, not surface string similarity.

Question: {question}

Ground-truth answer set ({num_gold} item(s)):
{gold_list}

Submitted answer set ({num_pred} item(s)):
{pred_list}

For every ground-truth item, decide whether it is semantically covered by \
at least one submitted item. Separately, for every submitted item, decide \
whether it matches at least one ground-truth item.

Respond with only a JSON object of exactly this form, and nothing else:
{{"matched_gold_indices": [...], "matched_submitted_indices": [...]}}
Both lists hold the 0-based indices (into the lists above) of the items \
that were matched."""

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def build_deepsearchqa_judge_fn() -> JudgeFn:
    """Build the task's judge from ``$OLMO_EVAL_JUDGE`` or its own default."""
    import os

    spec = os.getenv("OLMO_EVAL_JUDGE", DEEPSEARCHQA_DEFAULT_JUDGE_SPEC)
    model, separator, effort = spec.partition(":")
    return build_openai_judge_fn(
        model=model,
        temperature=0.0,
        max_tokens=1024,
        scorer_name="DeepSearchQA",
        reasoning_effort=(effort if separator else None),
    )


def build_deepsearchqa_judge_prompt(
    question: str, gold_items: Sequence[str], pred_items: Sequence[str]
) -> str:
    """Format the item-matching judge prompt for one instance."""
    return DEEPSEARCHQA_JUDGE_PROMPT.format(
        question=question,
        num_gold=len(gold_items),
        gold_list=_numbered_list(gold_items),
        num_pred=len(pred_items),
        pred_list=_numbered_list(pred_items),
    )


def parse_deepsearchqa_judge_response(
    raw: str, num_gold: int, num_pred: int
) -> tuple[set[int], set[int]] | None:
    """Parse the judge's matched-index JSON, or None if it is unparseable."""
    match = _JSON_OBJECT.search(raw)
    if match is None:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None

    gold_raw, pred_raw = data.get("matched_gold_indices"), data.get("matched_submitted_indices")
    if not isinstance(gold_raw, list) or not isinstance(pred_raw, list):
        return None

    try:
        matched_gold = {int(i) for i in gold_raw if 0 <= int(i) < num_gold}
        matched_pred = {int(i) for i in pred_raw if 0 <= int(i) < num_pred}
    except (TypeError, ValueError):
        return None
    return matched_gold, matched_pred


def compute_deepsearchqa_scores(
    num_gold: int, num_pred: int, num_matched_gold: int, num_matched_pred: int
) -> dict[str, float]:
    """Compute precision/recall/F1/exact-match from item-level match counts."""
    if num_gold == 0:
        # Gold answer is the empty set: a correctly-empty prediction is a perfect score,
        # any predicted item is an unwarranted hallucination.
        is_correct = num_pred == 0
        return {
            "deepsearchqa_precision": 1.0 if is_correct else 0.0,
            "deepsearchqa_recall": 1.0,
            "deepsearchqa_f1": 1.0 if is_correct else 0.0,
            "deepsearchqa_exact_match": 1.0 if is_correct else 0.0,
        }

    recall = num_matched_gold / num_gold
    precision = num_matched_pred / num_pred if num_pred else 0.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    exact_match = 1.0 if (num_pred > 0 and num_matched_pred == num_pred and recall == 1.0) else 0.0
    return {
        "deepsearchqa_precision": precision,
        "deepsearchqa_recall": recall,
        "deepsearchqa_f1": f1,
        "deepsearchqa_exact_match": exact_match,
    }


@dataclass(frozen=True)
class DeepSearchQAScorer(Scorer):
    """Placeholder scorer; DeepSearchQA scores are computed in ``score_responses``."""

    name: str = "deepsearchqa_f1"
    score_key: str = "deepsearchqa_f1"

    def score(self, instance: Instance, output: LMOutput) -> float:
        return (output.metadata or {}).get(self.score_key, 0.0)


@dataclass(frozen=True)
class DeepSearchQAMetric(Metric):
    """Mean of a precomputed DeepSearchQA score across responses."""

    name: str = "deepsearchqa_f1"
    scorer: type[Scorer] | Scorer = field(
        default_factory=lambda: DeepSearchQAScorer(score_key="deepsearchqa_f1")
    )

    def compute(self, responses: Sequence[Response]) -> float:
        if not responses:
            return 0.0
        return sum(response.scores.get(self.name, 0.0) for response in responses) / len(responses)

    def pairwise_display_format(self) -> str:
        return "percentage"

    def pairwise_unit(self) -> str:
        return "proportion"


def _metric(name: str) -> DeepSearchQAMetric:
    return DeepSearchQAMetric(name=name, scorer=DeepSearchQAScorer(score_key=name))


F1_METRIC = _metric("deepsearchqa_f1")
PRECISION_METRIC = _metric("deepsearchqa_precision")
RECALL_METRIC = _metric("deepsearchqa_recall")
EXACT_MATCH_METRIC = _metric("deepsearchqa_exact_match")
DEEPSEARCHQA_METRICS = (F1_METRIC, PRECISION_METRIC, RECALL_METRIC, EXACT_MATCH_METRIC)


class DeepSearchQABase(Task):
    """Shared data loading for the DeepSearchQA task family.

    Subclasses differ only in generation prompt and grading mechanic; the
    dataset, schema, and empty-gold-set handling are identical across them.
    """

    data_source = DataSource(
        path=DEEPSEARCHQA_REPO, subset=DEEPSEARCHQA_CONFIG, split=DEEPSEARCHQA_SPLIT
    )
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
        question = doc.get("problem")
        answer = doc.get("answer")
        answer_type = doc.get("answer_type", "")
        if not question:
            return None

        if not answer:
            # The source CSV encodes "no items satisfy every constraint" as the literal
            # text "None" on Set Answer rows; the HF CSV loader coerces that string to a
            # null, so it arrives here indistinguishable from a missing field. A missing
            # answer only makes sense as an intentional empty answer set for Set Answer
            # rows; treat anything else as a malformed row.
            if answer_type != "Set Answer":
                return None
            gold_items: list[str] = []
        else:
            gold_items = split_answer_set(answer)
            if not gold_items:
                return None

        return Instance(
            question=question,
            gold_answer=answer or "",
            metadata={
                "id": f"deepsearchqa_{index}",
                "index": index,
                "problem_category": doc.get("problem_category", ""),
                "answer_type": answer_type,
                "gold_items": gold_items,
            },
        )


@register("deepsearchqa")
class DeepSearchQA(DeepSearchQABase):
    """DeepSearchQA multi-step search question answering, graded by item-set F1."""

    metrics = DEEPSEARCHQA_METRICS
    primary_metric = F1_METRIC

    def format_request(self, instance: Instance) -> LMRequest:
        prompt = DEEPSEARCHQA_GENERATION_PROMPT.format(question=instance.question)
        return LMRequest(
            request_type=RequestType.CHAT,
            messages=({"role": "user", "content": prompt},),
        )

    def extract_answer(self, output: LMOutput) -> str:
        """Keep only the comma-separated list after the ``FINAL ANSWER`` marker."""
        return extract_final_answer(output.text)

    async def score_responses(
        self,
        responses: Sequence[Response],
        context: Any = None,
    ) -> Sequence[Response]:
        """Judge each response's answer set against the gold answer set."""
        self._extract_answers(responses)
        judge_fn = build_deepsearchqa_judge_fn()

        failed_instances = 0
        for response in responses:
            scores, judge_metadata, judge_failed = await self._score_single(response, judge_fn)
            response.scores.update(scores)
            failed_instances += judge_failed

            if response.outputs:
                output = response.outputs[0]
                output.metadata.update(judge_metadata)
                for metric_name, score in scores.items():
                    output.metadata[f"score:{metric_name}"] = score

        if failed_instances:
            logger.warning(
                "DeepSearchQA judge returned no parseable verdict for %d instance(s) after "
                "%d attempts; scored those 0.0 across all metrics.",
                failed_instances,
                DEEPSEARCHQA_JUDGE_ATTEMPTS,
            )
        return responses

    async def _score_single(
        self,
        response: Response,
        judge_fn: JudgeFn,
    ) -> tuple[dict[str, float], dict[str, Any], int]:
        """Score one response, returning metrics, judge details, and a failure count."""
        gold_items = response.instance.metadata.get("gold_items", [])
        output = response.outputs[0] if response.outputs else None

        pred_text = ""
        if output is not None:
            extracted = output.extracted_answer
            pred_text = extracted if isinstance(extracted, str) else output.text
        pred_items = split_answer_set(pred_text)

        base_metadata = {
            "deepsearchqa_gold_items": gold_items,
            "deepsearchqa_predicted_items": pred_items,
        }

        if not pred_items or not gold_items:
            # Nothing to match on one side (an empty prediction, or a gold answer
            # that is itself the empty set); the outcome is already determined.
            scores = compute_deepsearchqa_scores(len(gold_items), len(pred_items), 0, 0)
            return scores, base_metadata, 0

        prompt = build_deepsearchqa_judge_prompt(response.instance.question, gold_items, pred_items)

        parsed = None
        for attempt in range(DEEPSEARCHQA_JUDGE_ATTEMPTS):
            raw = await judge_fn(prompt)
            parsed = parse_deepsearchqa_judge_response(raw, len(gold_items), len(pred_items))
            if parsed is not None:
                break
            if attempt < DEEPSEARCHQA_JUDGE_ATTEMPTS - 1:
                await asyncio.sleep(2**attempt)

        matched_gold, matched_pred = parsed if parsed is not None else (set(), set())
        scores = compute_deepsearchqa_scores(
            len(gold_items), len(pred_items), len(matched_gold), len(matched_pred)
        )
        metadata = {
            **base_metadata,
            "deepsearchqa_matched_gold_indices": sorted(matched_gold),
            "deepsearchqa_matched_submitted_indices": sorted(matched_pred),
        }
        return scores, metadata, 0 if parsed is not None else 1


register_variant("deepsearchqa", "mini", limit=50)
