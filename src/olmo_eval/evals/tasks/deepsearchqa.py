"""DeepSearchQA: deep-research agent evaluation (Google DeepMind, arXiv 2601.20975).

DeepSearchQA is a 900-problem benchmark of hand-crafted, multi-step
information-seeking questions spanning 17 domains (Politics, Finance, Science,
Health, History, Geography, Media, ...). Each problem is either a
``Single Answer`` (one entity/value) or a ``Set Answer`` (an enumeration or
composite answer with multiple required items); on HuggingFace both are
stored as free-form text in the ``answer`` column
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
import os
import re
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field, replace
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
from olmo_eval.evals.tasks.common import Task, TaskConfig, register, register_variant

logger = logging.getLogger(__name__)

DEEPSEARCHQA_REPO = "google/deepsearchqa"
DEEPSEARCHQA_CONFIG = "deepsearchqa"
DEEPSEARCHQA_SPLIT = "eval"

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


# Official starter notebook rubric, with the output format expressed as a JSON template.
DEEPSEARCHQA_JUDGE_PROMPT = """\
Your task is to evaluate whether a given "AI Response" for a specific "User Prompt" arrived at \
the correct answer.

**Answer Correctness Task**

*   **Purpose:** Assess whether the AI response provides the correct answer(s) based on the \
provided "Correct Answer" and "Prompt Type".
*   **Process:**
    *   Identify the "Prompt Type": "<prompt_type>".
    *   Refer to the "Correct Answer": "<answer>".
    *   Based on the "Prompt Type", determine if the "AI Response" contains the expected answer(s).
        *   **'Single Answer'**: Check if the response provides the answer that addresses the \
user's question. It does not have to match the exact wording of the provided answer.
        *   **'Set Answer'**: Check if the response includes *each* item from the provided ground \
truth answers. The order might not matter unless specified otherwise. The response might include \
more answers than the list. Determine the correctness *only* based on the list first and then \
check if the response includes answers not in the list.
    *   **Explanation:** Provide a brief explanation justifying your assessment of answer \
correctness, referencing specific parts of the AI response and the correct answer.
    *   **Correctness Details:** Provide a dictionary, one key for each expected answer part, and \
value is a boolean indicating whether each expected answer part was found.
        *   For 'Set Answer', this will be a list of attributes, one for each item/part in the \
"Correct Answer". Each key will be a string indicating the expected answer part, and the value \
will be a boolean indicating whether that part was found in the response.
    *   **Excessive Answers:** Provide a list of strings, each indicating an excessive answer \
part. If the response provides answers that are **not** in the "Correct Answer" list, add these \
answers as excessive answers. Return an empty list when there's no excessive answers in the \
response.

**Output Format:**

Your evaluation *must* be structured as a nested JSON dictionary with the following top-level \
keys: `"Answer Correctness"`. Please return NULL if any of "Prompt", "AI Response" or "Correct \
Answer" is empty.
The value for `"Answer Correctness"` should be a dictionary containing `"Explanation"` (a \
string), `"Correctness Details"` (a dictionary where each key is the expected correct answer, and \
the value is a boolean indicating whether the response contains the correct answer), and \
`"Excessive Answers"` (a list of strings indicating the excessive answers).

Make sure you return a valid JSON object of exactly this form like the example below and nothing \
else:
{{
  "Answer Correctness": {{
    "Explanation": "The response correctly identified Belgium and France but also includes an \
excessive answer, Italy.",
    "Correctness Details": {{
      "Belgium": true,
      "France": true
    }},
    "Excessive Answers": [ "Italy" ]
  }}
}}

**Now, proceed with the evaluation using the provided User Prompt, AI Response, and Correct \
Answer.**

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


def build_deepsearchqa_judge_fn(config: TaskConfig) -> JudgeFn:
    """Build the task's judge from its recorded configuration."""
    if config.judge_model is None or config.judge_max_tokens is None:
        raise ValueError("DeepSearchQA requires a complete judge configuration")
    return build_openai_judge_fn(
        model=config.judge_model,
        temperature=0.0,
        max_tokens=config.judge_max_tokens,
        scorer_name="DeepSearchQA",
        reasoning_effort=config.judge_reasoning_effort,
    )


def build_deepsearchqa_judge_prompt(
    question: str, answer_type: str, gold_answer: str, pred_answer: str
) -> str:
    """Format the notebook judge prompt for one instance."""
    return DEEPSEARCHQA_JUDGE_PROMPT.format(
        prompt=question,
        prompt_type=answer_type or "Set Answer",
        answer=gold_answer,
        response=pred_answer,
    )


def parse_deepsearchqa_judge_response(raw: str) -> tuple[int, int, int] | None:
    """Return the judge's expected, matched, and excessive answer counts."""
    decoder = json.JSONDecoder()
    data: Any | None = None
    for match in re.finditer(r"\{", raw):
        try:
            candidate, _end = decoder.raw_decode(raw[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict):
            data = candidate
            break
    if data is None:
        return None

    correctness = data.get("Answer Correctness")
    if not isinstance(correctness, dict) or not isinstance(correctness.get("Explanation"), str):
        return None

    details = correctness.get("Correctness Details")
    excessive = correctness.get("Excessive Answers", [])
    if not isinstance(details, dict) or not isinstance(excessive, list):
        return None
    if any(not isinstance(value, bool) for value in details.values()):
        return None
    if any(not isinstance(item, str) for item in excessive):
        return None

    return len(details), sum(details.values()), len(excessive)


async def call_deepsearchqa_judge[T](
    judge_fn: JudgeFn,
    prompt: str,
    parse: Callable[[str], T | None],
) -> tuple[T | None, list[dict[str, Any]]]:
    """Retry unsuccessful judge attempts and retain their diagnostics."""
    failures: list[dict[str, Any]] = []
    for attempt in range(1, DEEPSEARCHQA_JUDGE_ATTEMPTS + 1):
        raw: str | None = None
        try:
            raw = await judge_fn(prompt)
            parsed = parse(raw)
        except Exception as exc:
            failure: dict[str, Any] = {
                "attempt": attempt,
                "type": type(exc).__name__,
                "message": str(exc),
            }
            if raw is not None:
                failure["response"] = raw
        else:
            if parsed is not None:
                return parsed, failures
            failure = {
                "attempt": attempt,
                "type": "InvalidJudgeResponse" if raw else "EmptyJudgeResponse",
                "message": "Judge returned an invalid or incomplete verdict.",
                "response": raw,
            }
        failures.append(failure)
        if attempt < DEEPSEARCHQA_JUDGE_ATTEMPTS:
            await asyncio.sleep(2 ** (attempt - 1))
    return None, failures


def compute_deepsearchqa_scores(
    num_gold: int, num_matched: int, num_excessive: int
) -> dict[str, float]:
    """Compute precision/recall/F1/exact-match from judge answer counts."""
    if num_gold == 0:
        is_correct = num_excessive == 0
        return {
            "deepsearchqa_precision": 1.0 if is_correct else 0.0,
            "deepsearchqa_recall": 1.0,
            "deepsearchqa_f1": 1.0 if is_correct else 0.0,
            "deepsearchqa_exact_match": 1.0 if is_correct else 0.0,
        }

    recall = num_matched / num_gold
    num_submitted = num_matched + num_excessive
    precision = num_matched / num_submitted if num_submitted else 0.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    exact_match = 1.0 if (num_matched == num_gold and num_excessive == 0) else 0.0
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
    dataset, schema, and no-answer reference handling are identical across them.
    """

    data_source = DataSource(
        path=DEEPSEARCHQA_REPO, subset=DEEPSEARCHQA_CONFIG, split=DEEPSEARCHQA_SPLIT
    )
    sampling_params = SamplingParams(temperature=0.0, max_tokens=4096)
    required_secrets = ("OPENAI_API_KEY",)
    judge_model = "gpt-5.5"
    judge_reasoning_effort = "medium"
    judge_max_tokens = 8192

    def __init__(self, config: TaskConfig) -> None:
        if spec := os.getenv("OLMO_EVAL_JUDGE"):
            model, _, effort = spec.partition(":")
            if not model:
                raise ValueError("OLMO_EVAL_JUDGE must name a judge model")
            config = replace(config, judge_model=model, judge_reasoning_effort=effort or None)
        super().__init__(config)

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
            # HF converts the source CSV's literal "None" to null. Restore the
            # no-answer reference for Set Answer rows.
            if answer_type != "Set Answer":
                return None
            answer = "None"
        elif not isinstance(answer, str) or not answer.strip(" ,"):
            return None

        return Instance(
            question=question,
            gold_answer=answer,
            metadata={
                "id": f"deepsearchqa_{index}",
                "index": index,
                "problem_category": doc.get("problem_category", ""),
                "answer_type": answer_type,
            },
        )

    def _judge_failure_metadata(
        self, response: Response, failures: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Report exhausted judge attempts and build persistent failure metadata."""
        instance_id = str(response.instance.metadata.get("id", "unknown"))
        last_failure = failures[-1]
        message = (
            f"Judge failed after {len(failures)} attempts: "
            f"{last_failure['type']}: {last_failure['message']}"
        )
        logger.warning(
            "%s judge failed for instance %s after %d attempts; "
            "last failure: %s: %s. Assigned 0.0 across all metrics; "
            "attempt details are saved in model_output[].judge_result.",
            self.config.name,
            instance_id,
            len(failures),
            last_failure["type"],
            last_failure["message"],
        )
        return {
            "judge_result": {
                "status": "failed",
                "task": self.config.name,
                "instance_id": instance_id,
                "attempts": len(failures),
                "failures": failures,
            },
            "scoring_errors": {
                metric.name: {
                    "phase": "judge",
                    "type": "JudgeAttemptsExhausted",
                    "message": message,
                }
                for metric in self.config.metrics
            },
        }


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
        """Extract the final answer submitted for grading."""
        return extract_final_answer(output.text)

    async def score_responses(
        self,
        responses: Sequence[Response],
        context: Any = None,
    ) -> Sequence[Response]:
        """Judge each response's answer set against the gold answer set."""
        self._extract_answers(responses)
        judge_fn = build_deepsearchqa_judge_fn(self.config)

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
                "DeepSearchQA judge failed for %d/%d instances after %d attempts each; "
                "failure details were saved and those instances were scored 0.0.",
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

        base_metadata = {
            "deepsearchqa_gold_answer": gold_answer,
            "deepsearchqa_submitted_answer": pred_text,
        }

        if not pred_text:
            scores = {metric.name: 0.0 for metric in DEEPSEARCHQA_METRICS}
            return scores, base_metadata, 0

        prompt = build_deepsearchqa_judge_prompt(
            response.instance.question, answer_type, gold_answer or "None", pred_text
        )

        def parse(raw: str) -> tuple[int, int, int] | None:
            parsed = parse_deepsearchqa_judge_response(raw)
            if parsed is not None and parsed[0] == 0 and gold_answer not in ("", "None"):
                return None
            return parsed

        parsed, failures = await call_deepsearchqa_judge(judge_fn, prompt, parse)
        if parsed is None:
            metadata = {**base_metadata, **self._judge_failure_metadata(response, failures)}
            return {metric.name: 0.0 for metric in DEEPSEARCHQA_METRICS}, metadata, 1

        scores = compute_deepsearchqa_scores(*parsed)
        return scores, base_metadata, 0


register_variant("deepsearchqa", "mini", limit=50)
