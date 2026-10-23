"""DeepSearchQA grading of full responses with the official notebook rubric.

This variant shares the notebook judge prompt and verdict parser with
``deepsearchqa``, expressing the output format as a JSON template. Its generation
prompt allows free-form answers, and the judge receives the full response.

The notebook uses Gemini 2.5 Flash; olmo-eval uses an OpenAI judge, so scores
are not directly comparable to the public leaderboard.

Usage:
    olmo-eval run -m llama3.1-8b -t deepsearchqa_official_judge --harness dr_tulu
    olmo-eval run -m llama3.1-8b -t deepsearchqa_official_judge:mini --harness dr_tulu
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from olmo_eval.common.metrics import Metric
from olmo_eval.common.scorers.base import Scorer
from olmo_eval.common.scorers.llm_judge import JudgeFn, build_openai_judge_fn
from olmo_eval.common.types import Instance, LMOutput, LMRequest, RequestType, Response
from olmo_eval.evals.tasks.common import TaskConfig, register, register_variant
from olmo_eval.evals.tasks.deepsearchqa import (
    DEEPSEARCHQA_JUDGE_ATTEMPTS,
    DEEPSEARCHQA_JUDGE_PROMPT,
    DeepSearchQABase,
    call_deepsearchqa_judge,
    parse_deepsearchqa_judge_response,
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

DEEPSEARCHQA_OFFICIAL_JUDGE_PROMPT = DEEPSEARCHQA_JUDGE_PROMPT


def build_deepsearchqa_official_judge_fn(config: TaskConfig) -> JudgeFn:
    """Build the task's judge from its recorded configuration."""
    if config.judge_model is None or config.judge_max_tokens is None:
        raise ValueError("DeepSearchQAOfficialJudge requires a complete judge configuration")
    return build_openai_judge_fn(
        model=config.judge_model,
        temperature=0.0,
        max_tokens=config.judge_max_tokens,
        scorer_name="DeepSearchQAOfficialJudge",
        reasoning_effort=config.judge_reasoning_effort,
    )


def build_deepsearchqa_official_judge_prompt(
    prompt: str, prompt_type: str, answer: str, response: str
) -> str:
    """Format the notebook grader prompt for one instance."""
    return DEEPSEARCHQA_OFFICIAL_JUDGE_PROMPT.format(
        prompt=prompt,
        prompt_type=prompt_type or "Set Answer",
        answer=answer,
        response=response,
    )


def parse_deepsearchqa_official_judge_response(raw: str) -> tuple[int, int, int] | None:
    """Return the judge's expected, matched, and excessive answer counts."""
    return parse_deepsearchqa_judge_response(raw)


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
    judge_max_tokens = 1024

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
        judge_fn = build_deepsearchqa_official_judge_fn(self.config)

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

        if failed_instances and len(responses) > 1:
            logger.warning(
                "DeepSearchQAOfficialJudge judge failed for %d/%d instances after %d attempts "
                "each; failure details were saved and those instances were scored 0.0.",
                failed_instances,
                len(responses),
                DEEPSEARCHQA_JUDGE_ATTEMPTS,
            )
        return responses

    async def _score_single(
        self,
        response: Response,
        judge_fn: JudgeFn,
    ) -> tuple[dict[str, float], dict[str, Any], int]:
        """Score one response, returning metrics, judge details, and a failure count."""
        gold_answer = response.instance.gold_answer or ""
        answer_type = response.instance.metadata.get("answer_type", "")
        output = response.outputs[0] if response.outputs else None

        pred_text = ""
        if output is not None:
            extracted = output.extracted_answer
            pred_text = extracted if isinstance(extracted, str) else output.text
        pred_text = pred_text.strip()

        base_metadata: dict[str, Any] = {"deepsearchqa_official_gold_answer": gold_answer}

        if not pred_text:
            scores = {metric.name: 0.0 for metric in DEEPSEARCHQA_OFFICIAL_METRICS}
            return scores, base_metadata, 0

        prompt = build_deepsearchqa_official_judge_prompt(
            response.instance.question, answer_type, gold_answer or "None", pred_text
        )

        def parse(raw: str) -> tuple[int, int, int] | None:
            parsed = parse_deepsearchqa_official_judge_response(raw)
            if parsed is not None and parsed[0] == 0 and gold_answer not in ("", "None"):
                return None
            return parsed

        parsed, failures = await call_deepsearchqa_judge(judge_fn, prompt, parse)
        if parsed is None:
            metadata = {**base_metadata, **self._judge_failure_metadata(response, failures)}
            return {metric.name: 0.0 for metric in DEEPSEARCHQA_OFFICIAL_METRICS}, metadata, 1

        num_gold, num_matched, num_excessive = parsed
        scores = compute_deepsearchqa_official_scores(num_gold, num_matched, num_excessive)
        metadata = {
            **base_metadata,
            "deepsearchqa_official_num_gold": num_gold,
            "deepsearchqa_official_num_matched": num_matched,
            "deepsearchqa_official_num_excessive": num_excessive,
        }
        return scores, metadata, 0


register_variant("deepsearchqa_official_judge", "mini", limit=50)
