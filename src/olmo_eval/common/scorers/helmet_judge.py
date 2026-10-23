"""LLM-as-judge scorers for HELMET's model-graded tasks.

HELMET grades its LongQA and Summarization tasks with GPT-4 rather than string
overlap, because the references are short phrases or paragraph summaries that
a correct answer can express many ways. This ports the LongQA rubric from
HELMET's `scripts/eval_gpt4_longqa.py`; the judge model and temperature match
upstream so scores stay comparable to published numbers.
"""

import json
import re
from dataclasses import dataclass
from typing import Any

from olmo_eval.common.execution import ScoringContext
from olmo_eval.common.scorers.helmet_summ_prompts import (
    FLUENCY_PROMPT,
    FLUENCY_PROMPT_BOOK,
    PRECISION_PROMPT,
    PRECISION_PROMPT_BOOK,
    RECALL_PROMPT,
    RECALL_PROMPT_BOOK,
)
from olmo_eval.common.scorers.llm_judge import LLMJudgeScorer, build_openai_judge_fn
from olmo_eval.common.types import Instance, LMOutput

# HELMET grades with gpt-4o at temperature 0.1 (scripts/eval_gpt4_longqa.py).
HELMET_JUDGE_MODEL = "gpt-4o-2024-05-13"
HELMET_JUDGE_TEMPERATURE = 0.1
# The rubric asks the judge to reason step by step before emitting JSON, so it
# needs far more room than the 10-token default used for letter-grade judges.
HELMET_JUDGE_MAX_TOKENS = 1024

# Verbatim from HELMET's scripts/eval_gpt4_longqa.py.
LONGQA_JUDGE_PROMPT = """Please act as an impartial judge and evaluate the quality of the provided answer which attempts to answer the provided question based on a provided context.
Although you are not given the context, you will be given a set of correct answers that achieves full scores on all metrics, and you need to assess the provided answers using the correct answers.

Below is your grading rubric:

Fluency:
- Score 0 (incoherent, repetitive, or incomplete): Incoherent sentences, repetitive sentences (even if not by exact words), incomplete answers, or gibberish. Note that even if the answer is coherent, if it is repetitive or incomplete, it should be given a score of 0.
- Score 1 (coherent, non-repetitive answer): Coherent, non-repetitive, fluent, grammatically correct answers.

Correctness:
- Score 0 (Incorrect): The answer does not agree with the provided correct answers at all.
- Score 1 (partly correct): Partly agree with one of the provided correct answers (for example, the question asks for a date and a person; the answer gets the date right but the person wrong).
- Score 2 (correct but not fully relevant): Fully agrees with one of the provided correct answers but mentions other completely irrelevant information. Note that extra details provided in the answer, even if not mentioned in the correct answers, should NOT be seen as irrelevant as long as they are relevant to the question to a reasonable extend.
- Score 3 (correct and relevant): Fully agrees with one of the provided correct answers and only provides information relevant to the question. Note that if the answer is longer than the correct answer, as long as everything in the answer is relevant to the question, it should still be given score 3. For example, if the correct answer is "the North Pole" and the answer is "They are headed for the North Pole", it should still be given a score of 3.

Now, read the following question, answer, and correct answers. First think step-by-step and provide your reasoning and assessment on the answer. Then output your score in the following json format: {{"fluency": 0, "correctness": 1}}.

Question: {question}
Correct answers: {correct_answers}
Answer: {parsed_output}
"""  # noqa: E501

# fluency (0-1) x correctness (0-3); dividing by this keeps the scorer inside
# the [0, 1] contract the base class documents, and multiplying a reported
# score back by it recovers HELMET's raw "gpt-4-score".
LONGQA_MAX_RAW_SCORE = 3.0


def parse_judge_json(text: str) -> dict | None:
    """Pull the last JSON object out of a judge response.

    Mirrors HELMET's `parse_json`: the rubric asks for reasoning followed by a
    JSON verdict, so the final object is the score.
    """
    matches = re.findall(r"\{.*?\}", text, re.DOTALL)
    if not matches:
        return None
    try:
        return json.loads(matches[-1])
    except json.JSONDecodeError:
        return None


@dataclass(frozen=True)
class _HelmetJudgeScorer(LLMJudgeScorer):
    """Shared judge settings and dispatch for HELMET's judged tasks.

    The judge model, temperature and token budget are fields, so they reach
    `to_dict` and the task hash, and both scoring paths read them: the default
    OpenAI `judge_fn` is built from them, and the provider path passes the
    temperature and budget with each request. On the provider path the model
    is whichever one the provider serves. A caller-supplied `judge_fn` is used
    as given, so these fields do not describe it.
    """

    judge_model: str = HELMET_JUDGE_MODEL
    judge_temperature: float = HELMET_JUDGE_TEMPERATURE
    judge_max_tokens: int = HELMET_JUDGE_MAX_TOKENS

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.provider_name is None and self.judge_fn is None:
            judge_fn = build_openai_judge_fn(
                model=self.judge_model,
                scorer_name=type(self).__name__,
                max_tokens=self.judge_max_tokens,
                temperature=self.judge_temperature,
            )
            object.__setattr__(self, "judge_fn", judge_fn)

    async def _judge(self, prompt: str, context: ScoringContext) -> str:
        if self.provider_name is not None:
            return await self._score_with_provider(
                prompt,
                context,
                temperature=self.judge_temperature,
                max_tokens=self.judge_max_tokens,
            )
        return await self._score_with_judge_fn(prompt)


@dataclass(frozen=True)
class HelmetLongQAJudgeScorer(_HelmetJudgeScorer):
    """HELMET's LongQA judge: fluency x correctness, normalized to [0, 1].

    HELMET's headline `gpt-4-score` is the product of a 0/1 fluency judgement
    and a 0-3 correctness judgement, so a disfluent answer scores zero however
    correct it is. That product is divided by its maximum of 3 here, both to
    respect the [0, 1] range the base class documents and so this task doesn't
    dominate a suite average against metrics that are already proportions --
    multiply a reported score by 3 to recover HELMET's number.

    An unparseable judge response scores 0.0. HELMET instead leaves such
    responses out of the average, so the two can differ; each instance's
    `judge_result` records the raw response and a `parse_error` flag, so
    failures can be counted and excluded after the fact.
    """

    name: str = "gpt4_score"

    def format_judge_prompt(self, instance: Instance, output: LMOutput) -> str:
        metadata = instance.metadata or {}
        correct_answers = metadata.get("all_gold_answers") or instance.gold_answer or []
        if isinstance(correct_answers, str):
            correct_answers = [correct_answers]

        return LONGQA_JUDGE_PROMPT.format(
            question=metadata.get("judge_question", instance.question),
            correct_answers=list(correct_answers),
            parsed_output=output.extracted_answer or output.text,
        )

    def parse_judge_response(self, response: str) -> float:
        score = self._parse_verdict(response)
        return 0.0 if score is None else score

    @staticmethod
    def _parse_verdict(response: str) -> float | None:
        """Return the normalized score, or None if the verdict is unusable."""
        scores = parse_judge_json(response)
        if scores is None or "fluency" not in scores or "correctness" not in scores:
            return None
        try:
            fluency = float(scores["fluency"])
            correctness = float(scores["correctness"])
        except (TypeError, ValueError):
            return None
        return (fluency * correctness) / LONGQA_MAX_RAW_SCORE

    async def ascore_with_context(
        self, instance: Instance, output: LMOutput, context: ScoringContext
    ) -> float:
        self._validate_provider(context)
        response = await self._judge(self.format_judge_prompt(instance, output), context)
        score = self._parse_verdict(response)
        output.metadata["judge_result"] = {
            "raw_judge_response": response,
            "parse_error": score is None,
        }
        return 0.0 if score is None else score


@dataclass(frozen=True)
class HelmetSummJudgeScorer(_HelmetJudgeScorer):
    """HELMET's summarization judge: fluency-gated F1 over key points.

    Ports `scripts/eval_gpt4_summ.py`. Unlike the LongQA judge this needs three
    separate judge calls per summary, because the rubrics grade different
    things and would interfere if merged into one prompt:

    - fluency   -> 0/1, is the summary coherent and non-repetitive
    - recall    -> how many of the pre-extracted key points it covers
    - precision -> how many of its own sentences the expert summary supports

    which combine as HELMET's `gpt-4-f1`::

        recall = found_key_points / total_key_points
        precision = supported_sentences / sentence_count
        f1 = fluency * 2 * recall * precision / (recall + precision)

    Fluency multiplies rather than averages, so a disfluent summary scores zero
    however much it covers -- that is deliberate upstream, since degenerate
    repetition can otherwise score well on recall.

    The key points are not derived at scoring time: they were extracted ahead of
    time and ship with HELMET's data, reaching the scorer via instance metadata.
    An instance without them cannot be graded and scores 0.0 without calling
    the judge; its `judge_result` says so. As with the LongQA judge, an
    unusable verdict scores 0.0 and sets `parse_error` in `judge_result`,
    alongside the three raw responses.

    `is_book` selects the novel-summary rubric variants (infbench_sum_eng) over
    the civil-lawsuit ones (multi_lexsum).
    """

    name: str = "gpt4_f1"
    is_book: bool = False

    def _prompts(self, instance: Instance, output: LMOutput) -> tuple[str, str, str]:
        metadata = instance.metadata or {}
        summary = (output.text or "").strip()
        key_points = metadata.get("keypoints") or []
        numbered = "\n".join(f"{i + 1}. {kp}" for i, kp in enumerate(key_points))
        # the expert reference: the long summary for lawsuits, the gold summary
        # itself for books, matching how upstream fills each template
        expert = metadata.get("expert_summary") or ""

        if self.is_book:
            return (
                FLUENCY_PROMPT_BOOK.format(text=summary),
                RECALL_PROMPT_BOOK.format(keypoints=numbered, summary=summary),
                PRECISION_PROMPT_BOOK.format(expert_summary=expert, summary=summary),
            )
        return (
            FLUENCY_PROMPT.format(text=summary),
            RECALL_PROMPT.format(keypoints=numbered, summary=summary),
            PRECISION_PROMPT.format(expert_summary=expert, summary=summary),
        )

    def format_judge_prompt(self, instance: Instance, output: LMOutput) -> str:
        """Unused: this scorer issues three prompts, see `ascore_with_context`."""
        return self._prompts(instance, output)[0]

    def parse_judge_response(self, response: str) -> float:
        """Unused: the three responses are combined in `ascore_with_context`."""
        raise RuntimeError("HelmetSummJudgeScorer combines three judge responses.")

    @staticmethod
    def combine(
        fluency: dict | None,
        recall: dict | None,
        precision: dict | None,
        num_key_points: int,
    ) -> float:
        """Combine the three verdicts into HELMET's gpt-4-f1 (0.0 if any is unusable)."""
        score = HelmetSummJudgeScorer._combine_verdicts(fluency, recall, precision, num_key_points)
        return 0.0 if score is None else score

    @staticmethod
    def _combine_verdicts(
        fluency: dict | None,
        recall: dict | None,
        precision: dict | None,
        num_key_points: int,
    ) -> float | None:
        """Return HELMET's gpt-4-f1, or None if any verdict is unusable."""
        if fluency is None or recall is None or precision is None:
            return None
        try:
            fluency_score = float(fluency["fluency"])
            found = float(recall["recall"])
            supported = float(precision["precision"])
            sentence_count = float(precision["sentence_count"])
        except (KeyError, TypeError, ValueError):
            return None

        rec = found / num_key_points if num_key_points > 0 else 0.0
        prec = supported / sentence_count if sentence_count > 0 else 0.0
        if rec + prec <= 0:
            return 0.0
        return fluency_score * 2 * (rec * prec) / (rec + prec)

    async def ascore_with_context(
        self, instance: Instance, output: LMOutput, context: ScoringContext
    ) -> float:
        self._validate_provider(context)
        key_points = (instance.metadata or {}).get("keypoints") or []
        if not key_points:
            output.metadata["judge_result"] = {"missing_keypoints": True, "parse_error": False}
            return 0.0

        responses: dict[str, Any] = {}
        verdicts = []
        for rubric, prompt in zip(
            ("fluency", "recall", "precision"), self._prompts(instance, output), strict=True
        ):
            responses[rubric] = await self._judge(prompt, context)
            verdicts.append(parse_judge_json(responses[rubric]))

        score = self._combine_verdicts(*verdicts, num_key_points=len(key_points))
        output.metadata["judge_result"] = {
            "raw_judge_responses": responses,
            "parse_error": score is None,
        }
        return 0.0 if score is None else score
