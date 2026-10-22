"""DeepSearchQA graded with the official paper prompt (Google DeepMind, arXiv 2601.20975).

This is a second implementation of the ``deepsearchqa`` task (see
``deepsearchqa.py`` for the dataset, taxonomy, and background) that instead
mirrors the grading methodology published in Appendix A of the technical
report: the judge grades the model's raw, free-form response directly rather
than a discrete ``FINAL ANSWER: ...`` line, and reports per-gold-item
"Correctness Details" plus a free-form "Excessive Answers" list rather than
matching two pre-extracted item lists by index.

Concretely, this changes two things relative to ``deepsearchqa``:
  * The generation prompt does not ask for a ``FINAL ANSWER:`` line; the full
    response is what gets judged.
  * The judge prompt and parsing follow the official schema: a boolean per
    expected gold item (did the response contain it), plus a free list of
    excessive items the response claimed that are not part of the gold
    answer. Precision/recall/F1/exact-match are derived from those counts
    the same way the paper defines them.

As with ``deepsearchqa``, the official grader is Gemini 2.5 Flash;
``build_openai_judge_fn`` is OpenAI-only, so this task uses an OpenAI model
against the official prompt text instead, and absolute numbers are still not
directly comparable to the public Kaggle leaderboard.

Usage:
    olmo-eval run -m llama3.1-8b -t deepsearchqa_official_judge --harness dr_tulu
    olmo-eval run -m llama3.1-8b -t deepsearchqa_official_judge:mini --harness dr_tulu
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from olmo_eval.common.metrics import Metric
from olmo_eval.common.scorers.base import Scorer
from olmo_eval.common.scorers.llm_judge import JudgeFn, build_openai_judge_fn
from olmo_eval.common.types import Instance, LMOutput, LMRequest, RequestType, Response
from olmo_eval.evals.tasks.common import register, register_variant
from olmo_eval.evals.tasks.deepsearchqa import (
    DEEPSEARCHQA_DEFAULT_JUDGE_SPEC,
    DEEPSEARCHQA_JUDGE_ATTEMPTS,
    DeepSearchQABase,
)

logger = logging.getLogger(__name__)

DEEPSEARCHQA_OFFICIAL_GENERATION_PROMPT = """\
Answer the following question. It may require an exhaustive search across \
multiple sources rather than a single lookup. If you have access to search \
tools, use them to find and verify each candidate answer instead of \
answering from memory alone.

The question may have exactly one correct answer, or it may require a \
complete set of items that together satisfy every constraint it states. Do \
not stop at the first plausible answer if the question implies there may be \
more than one.

Question: {question}"""

# Verbatim from DeepSearchQA paper (arXiv 2601.20975), Appendix A: "Grader Prompt".
DEEPSEARCHQA_OFFICIAL_JUDGE_PROMPT = """\
Your task is to evaluate whether a given "AI Response" for a specific "User Prompt"
arrived at the correct answer.

**Answer Correctness Task**
* **Purpose:** Assess whether the AI response provides the correct answer(s) based on
the provided "Correct Answer" and "Prompt Type".
* **Process:**
    * Identify the "Prompt Type": "{prompt_type}".
    * Refer to the "Correct Answer": "{answer}".
    * Based on the "Prompt Type", determine if the "AI Response" contains the expected
answer(s).
        * **'Single Answer'**: Check if the response provides the answer that addresses
the user's question. It does not have to match the exact wording of the provided
answer.
        * **'Set Answer'**: Check if the response includes *each* item from the provided
ground truth answers. The order might not matter unless specified otherwise. The
response might include more answers than the list. Determine the correctness *
only* based on the list first and then check if the response includes answers not
in the list.
    * **Explanation:** Provide a brief explanation justifying your assessment of answer
correctness, referencing specific parts of the AI response and the correct answer.
    * **Correctness Details:** Provide a dictionary, one key for each expected answer
part, and value is a boolean indicating whether each expected answer part was found.
        * For 'Set Answer', this will be a list of attributes, one for each item/part in
the "Correct Answer". Each key will be a string indicating the expected answer
part, and the value will be a boolean indicating whether that part was found in
the response.
    * **Excessive Answers:** Provide a list of strings, each indicating an excessive
answer part. If the response provides answers that are **not** in the "Correct Answer
" list, add these answers as excessive answers. Return an empty list when there's no
excessive answers in the response.

**Output Format:**
Your evaluation *must* be structured as a nested JSON dictionary with the following top-
level keys: `"Answer Correctness"`. Please return NULL if any of "Prompt", "AI Response"
or "Correct Answer" is empty.
The value for `"Answer Correctness"` should be a dictionary containing `"Explanation"` (
a string), `"Correctness Details"` (a dictionary where each key is the expected correct
answer, and the value is a boolean indicating whether the response contains the correct
answer), and `"Excessive Answers"` (a list of strings indicating the excessive answers).

Make sure you return a valid JSON string. Pay special attention to quotes, commas and
special characters in the JSON string. Make sure to escape all special characters and
quotes in the JSON string.

User Prompt (Wrapped in <prompt> and </prompt>):
<prompt>
{prompt}
</prompt>
--------------------
Correct Answer (Wrapped in <answer> and </answer>):
Prompt Type: {prompt_type}
<answer>
{answer}
</answer>
--------------------
AI assistant response (Wrapped in <response> and </response>):
<response>
{response}
</response>
--------------------
Rating:"""

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def build_deepsearchqa_official_judge_fn() -> JudgeFn:
    """Build the task's judge from ``$OLMO_EVAL_JUDGE`` or its own default."""
    import os

    spec = os.getenv("OLMO_EVAL_JUDGE", DEEPSEARCHQA_DEFAULT_JUDGE_SPEC)
    model, separator, effort = spec.partition(":")
    return build_openai_judge_fn(
        model=model,
        temperature=0.0,
        max_tokens=1024,
        scorer_name="DeepSearchQAOfficialJudge",
        reasoning_effort=(effort if separator else None),
    )


def build_deepsearchqa_official_judge_prompt(
    prompt: str, prompt_type: str, answer: str, response: str
) -> str:
    """Format the official Appendix A grader prompt for one instance."""
    return DEEPSEARCHQA_OFFICIAL_JUDGE_PROMPT.format(
        prompt=prompt,
        prompt_type=prompt_type or "Set Answer",
        answer=answer,
        response=response,
    )


def parse_deepsearchqa_official_judge_response(raw: str, num_gold: int) -> tuple[int, int] | None:
    """Parse the official schema's match/excessive counts, or None if unparseable.

    The judge echoes gold items back as dictionary keys, which an LLM cannot
    be relied on to reproduce verbatim (paraphrases, casing, punctuation).
    Rather than reconciling those keys against ``gold_items`` by text — which
    would reintroduce the same surface-matching problem the judge exists to
    avoid — this counts how many of the (at most ``num_gold``) entries were
    marked true, matching the official recall/precision formulas without
    needing per-item traceability.
    """
    match = _JSON_OBJECT.search(raw)
    if match is None:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None

    correctness = data.get("Answer Correctness")
    if not isinstance(correctness, dict):
        return None

    details = correctness.get("Correctness Details")
    excessive = correctness.get("Excessive Answers")
    if not isinstance(details, dict) or not isinstance(excessive, list):
        return None

    num_matched = min(sum(1 for v in details.values() if v is True), num_gold)
    return num_matched, len(excessive)


def compute_deepsearchqa_official_scores(
    num_gold: int, num_matched: int, num_excessive: int
) -> dict[str, float]:
    """Compute precision/recall/F1/exact-match from official match/excessive counts."""
    if num_gold == 0:
        is_correct = num_excessive == 0
        return {
            "deepsearchqa_official_precision": 1.0 if is_correct else 0.0,
            "deepsearchqa_official_recall": 1.0,
            "deepsearchqa_official_f1": 1.0 if is_correct else 0.0,
            "deepsearchqa_official_exact_match": 1.0 if is_correct else 0.0,
        }

    recall = num_matched / num_gold
    num_submitted = num_matched + num_excessive
    precision = num_matched / num_submitted if num_submitted else 0.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    exact_match = 1.0 if (num_matched == num_gold and num_excessive == 0) else 0.0
    return {
        "deepsearchqa_official_precision": precision,
        "deepsearchqa_official_recall": recall,
        "deepsearchqa_official_f1": f1,
        "deepsearchqa_official_exact_match": exact_match,
    }


@dataclass(frozen=True)
class DeepSearchQAOfficialScorer(Scorer):
    """Placeholder scorer; scores are computed in ``score_responses``."""

    name: str = "deepsearchqa_official_f1"
    score_key: str = "deepsearchqa_official_f1"

    def score(self, instance: Instance, output: LMOutput) -> float:
        return (output.metadata or {}).get(self.score_key, 0.0)


@dataclass(frozen=True)
class DeepSearchQAOfficialMetric(Metric):
    """Mean of a precomputed DeepSearchQA-official score across responses."""

    name: str = "deepsearchqa_official_f1"
    scorer: type[Scorer] | Scorer = field(
        default_factory=lambda: DeepSearchQAOfficialScorer(score_key="deepsearchqa_official_f1")
    )

    def compute(self, responses: Sequence[Response]) -> float:
        if not responses:
            return 0.0
        return sum(response.scores.get(self.name, 0.0) for response in responses) / len(responses)

    def pairwise_display_format(self) -> str:
        return "percentage"

    def pairwise_unit(self) -> str:
        return "proportion"


def _metric(name: str) -> DeepSearchQAOfficialMetric:
    return DeepSearchQAOfficialMetric(name=name, scorer=DeepSearchQAOfficialScorer(score_key=name))


F1_METRIC = _metric("deepsearchqa_official_f1")
PRECISION_METRIC = _metric("deepsearchqa_official_precision")
RECALL_METRIC = _metric("deepsearchqa_official_recall")
EXACT_MATCH_METRIC = _metric("deepsearchqa_official_exact_match")
DEEPSEARCHQA_OFFICIAL_METRICS = (F1_METRIC, PRECISION_METRIC, RECALL_METRIC, EXACT_MATCH_METRIC)


@register("deepsearchqa_official_judge")
class DeepSearchQAOfficialJudge(DeepSearchQABase):
    """DeepSearchQA graded with the official paper prompt and free-form response."""

    metrics = DEEPSEARCHQA_OFFICIAL_METRICS
    primary_metric = F1_METRIC

    def format_request(self, instance: Instance) -> LMRequest:
        prompt = DEEPSEARCHQA_OFFICIAL_GENERATION_PROMPT.format(question=instance.question)
        return LMRequest(
            request_type=RequestType.CHAT,
            messages=({"role": "user", "content": prompt},),
        )

    def extract_answer(self, output: LMOutput) -> str:
        """No extraction: the official grader judges the full raw response."""
        return output.text.strip()

    async def score_responses(
        self,
        responses: Sequence[Response],
        context: Any = None,
    ) -> Sequence[Response]:
        """Judge each response's raw text against the gold answer with the official prompt."""
        self._extract_answers(responses)
        judge_fn = build_deepsearchqa_official_judge_fn()

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
                "DeepSearchQAOfficialJudge judge returned no parseable verdict for %d "
                "instance(s) after %d attempts; scored those 0.0 across all metrics.",
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
        answer_type = response.instance.metadata.get("answer_type", "")
        output = response.outputs[0] if response.outputs else None

        pred_text = ""
        if output is not None:
            extracted = output.extracted_answer
            pred_text = extracted if isinstance(extracted, str) else output.text
        pred_text = pred_text.strip()

        base_metadata: dict[str, Any] = {"deepsearchqa_official_gold_items": gold_items}

        if not pred_text:
            # An empty response can't contain any gold item or excessive claim.
            scores = compute_deepsearchqa_official_scores(len(gold_items), 0, 0)
            return scores, base_metadata, 0

        answer_text = ", ".join(gold_items) if gold_items else "None"
        prompt = build_deepsearchqa_official_judge_prompt(
            response.instance.question, answer_type, answer_text, pred_text
        )

        parsed = None
        for attempt in range(DEEPSEARCHQA_JUDGE_ATTEMPTS):
            raw = await judge_fn(prompt)
            parsed = parse_deepsearchqa_official_judge_response(raw, len(gold_items))
            if parsed is not None:
                break
            if attempt < DEEPSEARCHQA_JUDGE_ATTEMPTS - 1:
                await asyncio.sleep(2**attempt)

        num_matched, num_excessive = parsed if parsed is not None else (0, 0)
        scores = compute_deepsearchqa_official_scores(len(gold_items), num_matched, num_excessive)
        metadata = {
            **base_metadata,
            "deepsearchqa_official_num_matched": num_matched,
            "deepsearchqa_official_num_excessive": num_excessive,
        }
        return scores, metadata, 0 if parsed is not None else 1


register_variant("deepsearchqa_official_judge", "mini", limit=50)
