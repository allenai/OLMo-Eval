"""AstaBench ScholArQA-CS2 scientific question answering task.

Evaluates models on generating well-cited research reports with inline citations.
Scoring uses 4 LLM-as-judge metrics (each 25% weight):
ingredient_recall, answer_precision, citation_precision, citation_recall.

Reference implementation: https://github.com/allenai/asta-bench
"""

from __future__ import annotations

import difflib
import logging
import re
from abc import ABC
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

from olmo_eval.common.metrics import Metric
from olmo_eval.common.scorers.base import Scorer
from olmo_eval.common.scorers.citation import (
    extract_json_from_response,
    score_citations_for_sections,
)
from olmo_eval.common.scorers.llm_judge import JudgeFn, build_default_judge_fn
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

# =============================================================================
# Constants
# =============================================================================

ASTA_BENCH_REPO = "allenai/asta-bench"
ASTA_BENCH_REVISION = "a600dc767f850385f4664772e3ba7a7f8be17d5e"

# =============================================================================
# Prompt Templates (from astabench evals/sqa/task.py, rubric.py, precision_eval.py, citation_eval.py)
# =============================================================================

# from astabench evals/sqa/task.py:json_to_sample
SQA_GENERATION_PROMPT = """Generate a report answering the following research question. Be sure to include inline citations for each claim. Return your result as valid JSON with a single key `sections` which is a list of sections, each having keys `title`, `text`, and `citations`. Each entry in `citations` should have a JSON list of `snippets` extracted from the reference document and an `id`, each of which appears exactly in the text. Each `id` should be an inline citation as it appears in the text (with wrapping parentheses or square brackets if appropriate). Each citation should have a `title` if one is available. Any additional information about the citation should go under `metadata`. Do not create a References section.

Here is an example `section` to help you with formatting:

        {
          "title": "Background",
          "text": "Convolutional neural networks (CNNs) have achieved state-of-the-art results in image classification [1][2].",
          "citations": [
            {
              "id": "[1]",
              "snippets": [
                "CNNs have become the standard for many visual tasks."
              ],
              "title": "ImageNet Classification with Deep Convolutional Neural Networks",
              "metadata": {
                "authors": "Krizhevsky, A. et al.",
                "year": 2012,
                "arxiv": "1207.0580"
              }
            },
            {
              "id": "[2]",
              "snippets": [
                "Significant improvements in image recognition have been observed with CNNs."
              ],
              "title": "Very Deep Convolutional Networks for Large-Scale Image Recognition",
              "metadata": {
                "authors": "Simonyan, K. & Zisserman, A.",
                "year": 2014,
                "arxiv": "1409.1556"
              }
            }
          ]
        }

Question: """

# from astabench rubric.py:_assess_properties_jointly
INGREDIENT_SYSTEM_PROMPT = """You will be given a question someone asked (in <question></question> tags) and the corresponding response (in <response></response> tags) given to them by an assistant.
        You will then be given an enumerated list of criteria by which to evaluate the response. Each criterion specifies requirements that the answer must satisfy. You will assign a score accordingly (see below).
        You will also be given a list of examples (in <examples></examples> tags, below each criterion) that illustrate the type of details that would satisfy the criterion. We do NOT expect any of the specified details to necessarily appear in the answer. These are strictly to be used as guidance for locating the answers that satisfy the set requirement.

        For each criterion, return a score of 0, 1 or 2 indicating how appropriate the response is based on the given criterion. 0 means the response does not meet the criterion, 1 means the response somewhat meets the criterion, 2 means the response perfectly meets the criterion. Judge only the specified aspect(s) delimited by the criterion, not any other qualities of the answer.

        Scoring Example 1:
        <question>Common medical NLP papers on clinical text benchmarks</question>
        <response>The application of natural language processing (NLP) and machine learning to medical text presents tremendous opportunities for healthcare tasks such as prediction ... [TRUNCATED]</response>
        Criteria:
        <criterion>
        1. Detail the well-known medical NLP datasets
        <examples>
        i2b2 includes datasets focused on temporal relations in clinical narratives, CRAFT Corpus is a collection of 97 full-length, open-access biomedical journal articles with semantic and syntactic annotations.]
        </examples>
        </criterion>
        <criterion>
        2. ... [TRUNCATED]
        <examples>
        ...[TRUNCATED]
        </examples>
        </criterion>

        A 2 point answer would fully satisfy the criterion #1. For example, it would include specific names with some details of well-known medical datasets for ML like those mentioned in the examples.
        A 1 point answer would only partially satisfy the criterion #1. For example, a dataset (like those in examples) may be mentioned, but no detail would be provided. Or datasets may be simply listed without further discussion.
        A 0 point answer would not mention datasets at all.

        Scoring Example 2:
        <question>What are some of the documentation methods used in Linguistics fieldwork.</question>
        <response>Language documentation, also called documentary linguistics, is a specialized subfield of linguistics ... [TRUNCATED]</response>
        Criteria:
        <criterion>
        1. ... [TRUNCATED]
        <examples>
        ...[TRUNCATED]
        </examples>
        </criterion>
        <criterion>
        2. Cover elicitation techniques for capturing specific linguistic data.
        <examples>
        structured interviews, elicitations based on standard word lists, prompted speech tasks
        </examples>
        </criterion>

        A 2 point answer to criterion #2 would contain common elicitation techniques like (but not limited to) those mentioned in the examples. The answer specifics don't have to match exactly with the examples, but examples show the types of instances that would count towards satisfying the criterion.
        A 1 point answer to criterion #2 be incomplete in some way. For example, the answer might mention "elicitation sessions" during a discussion on audio recording, but it fails to specifically address the requirement. Or the answer gives a list of standard word lists in the answer as resources, but fails to tie this information to elicitation.
        A 0 point answer to criterion #2 would simply not include the discussion in any way. For example, if an answer focuses only on data handling (post elicitation) techniques, it would miss out on techniques for documentation interview itself.

        Scoring Example 3:

        <question>How do transformer models differ from recurrent neural networks (RNNs)?</question>
        <response>Transformer models use self-attention mechanisms to process input, while RNNs process input sequentially. Transformers are better at handling long-range dependencies in data because they don't rely on previous time steps to pass information. RNNs may suffer from vanishing gradients and have trouble with long-term dependencies.</response>
        Criteria:
        <criterion>
        1. Must compare how the architecture and data processing flow differ between transformers and RNNs. <examples>
        Transformers use parallel processing and self-attention; RNNs process input tokens one at a time in sequence. Transformers can look at the entire input sequence at once, while RNNs have to pass information step by step.
        </examples>
        </criterion>

        A 2 point answer would accurately and distinctly contrast both architecture and sequence-processing style of both model families (e.g., parallelism vs. sequential processing, use of self-attention vs. recurrence).

        A 1 point answer would provide a partial or imprecise comparison, perhaps only mentioning one difference, or being vague (e.g., "Transformers work differently from RNNs in how they process text" without further elaboration).

        A 0 point answer would explain only one architecture (e.g., only transformers), or describe both but fail to contrast them on the asked criteria.

        Return your result as a JSON object with a single key `scores` whose value is a list of objects, each having keys `criteria_idx`, `reasoning`, `score` and `evidence` from the text supporting the claim."""

# from astabench precision_eval.py:PrecisionEval
PRECISION_EVAL_PROMPT = """You are given a query and a corresponding long answer.

Goal: find irrelevant paragraphs in the answer. These are paragraphs that don't directly answer the query and shouldn't be in the answer.

For instance, if the query is about datasets for scientific question answering, a paragraph about multilingual question answering datasets that don't contain scientific text would be considered irrelevant.

Explicitly consider whether something may be indirectly relevant. For example, if the question is about the conditions of horses in South Africa, a paragraph about general animal welfare in South Africa is potentially relevant while not being precisely about horses. On the other hand, a paragraph about pig welfare in South Africa is irrelevant.

Note that subtle differences can make the text irrelevant to the query. For instance, text about scientific survey paper generation is not relevant to a query about automatic paper review generation. Even though they seem related, they are about very different tasks.

Also, useful background in general is relevant. If the question is about an approach to creating liver-related proteins, some information about liver-related proteins could contextualize other parts of the answer. If a paragraph contextualizes another part of the answer, then it is relevant.

Go through the answer and output a list of irrelevant paragraphs. Every single paragraph needs to be considered, one by one. Our goal is to catch all the irrelevant paragraphs, so please be thorough.
Return your result as a JSON object with a single key `irrelevant_paragraphs` whose value is a list of objects, each having keys `reason`, and `answer_text` as follows:
{{"irrelevant_paragraphs":[
{{
"reason": "discuss why something is irrelevant (not indirectly relevant)",
"answer_text": "exact ENTIRE paragraph (not just a part of it) from the answer that is irrelevant"
}},
...
]
}}
Make sure all the irrelevant paragraphs are included.

Question: {query}
Answer: {answer}
"""

# =============================================================================
# JSON / Response Parsing Helpers (from astabench evals/sqa/task.py)
# =============================================================================


def normalize_agent_response_dict(response_dict: dict[str, Any]) -> dict[str, Any]:
    """Normalize response format to always have top-level `sections` key."""
    if not response_dict.get("sections") and response_dict.get("response"):
        return response_dict["response"]
    return response_dict


def format_report(response_dict: dict[str, Any]) -> str:
    """Extract plain text from structured response sections."""
    report = ""
    for section in response_dict.get("sections", []):
        text = section.get("text", "")
        if text:
            report += f"{text}\n\n"
        table = section.get("table")
        if table and isinstance(table, dict):
            table_text = table.get("text", "")
            if table_text:
                report += f"{table_text}\n\n"
    return report.strip()


# =============================================================================
# Scoring Helpers (ported from astabench precision_eval.py, rubric.py)
# =============================================================================


# from astabench precision_eval.py:PrecisionEval.score
def compute_precision_score(
    parsed_json: dict[str, Any], answer: str
) -> tuple[float, dict[str, Any]]:
    """Compute answer precision from judge output identifying irrelevant paragraphs."""
    metadata: dict[str, Any] = {}
    irrelevant_texts = [
        item["answer_text"] for item in parsed_json.get("irrelevant_paragraphs", [])
    ]
    metadata["irrelevant_texts"] = irrelevant_texts

    paragraphs = re.split(r"\n\s*\n", answer.strip())
    metadata["answer_paragraphs"] = paragraphs

    if not paragraphs:
        return 1.0, metadata

    matching_paragraph_indices: set[int] = set()
    for sentence in irrelevant_texts:
        for idx, para in enumerate(paragraphs):
            if difflib.SequenceMatcher(None, sentence.strip(), para).ratio() >= 0.85:
                matching_paragraph_indices.add(idx)
                break

    score = 1 - (len(matching_paragraph_indices) / len(paragraphs))
    return score, metadata


# from astabench rubric.py:_assess_properties_jointly
def compute_ingredient_score(
    parsed_json: dict[str, Any],
    ingredients: list[dict[str, Any]],
) -> float:
    """Compute ingredient recall from joint assessment judge output."""
    has_criterion = [ing for ing in ingredients if ing.get("criterion")]
    if not has_criterion:
        return 0.0

    # Normalize weights so they sum to 1
    raw_weights = {ing["name"]: ing["weight"] for ing in has_criterion}
    total_weight = sum(raw_weights.values())
    if total_weight == 0:
        return 0.0
    weights = {k: v / total_weight for k, v in raw_weights.items()}

    scores_list = parsed_json.get("scores", [])
    score_components: dict[str, float] = {}
    for item in scores_list:
        idx = item.get("criteria_idx", 0) - 1  # 1-indexed in prompt
        if 0 <= idx < len(has_criterion):
            name = has_criterion[idx]["name"]
            score_components[name] = item.get("score", 0) / 2  # Normalize 0-2 to 0-1

    return sum(weights.get(k, 0) * score_components.get(k, 0) for k in weights)


# =============================================================================
# Scorer + Metrics
# =============================================================================


@dataclass(frozen=True)
class SQAJudgeScorer(Scorer):
    """Placeholder scorer for SQA metrics.

    Scoring is done in AstaBenchSQA.score_responses, not via this scorer.
    The score() method reads pre-computed scores from output metadata.
    """

    name: str = "sqa_judge"
    score_key: str = "global_avg"

    def score(self, instance: Instance, output: LMOutput) -> float:
        return (output.metadata or {}).get(self.score_key, 0.0)


class SQAMetricBase(Metric, ABC):
    """Base for SQA metrics that read from response.scores."""

    scorer: type[Scorer] = SQAJudgeScorer

    def compute(self, responses: Sequence[Response]) -> float:
        if not responses:
            return 0.0
        total = sum(r.scores.get(self.name, 0.0) for r in responses)
        return total / len(responses)


@dataclass(frozen=True)
class IngredientRecallMetric(SQAMetricBase):
    name: str = "ingredient_recall"
    scorer: type[Scorer] = SQAJudgeScorer


@dataclass(frozen=True)
class AnswerPrecisionMetric(SQAMetricBase):
    name: str = "answer_precision"
    scorer: type[Scorer] = SQAJudgeScorer


@dataclass(frozen=True)
class CitationPrecisionMetric(SQAMetricBase):
    name: str = "citation_precision"
    scorer: type[Scorer] = SQAJudgeScorer


@dataclass(frozen=True)
class CitationRecallMetric(SQAMetricBase):
    name: str = "citation_recall"
    scorer: type[Scorer] = SQAJudgeScorer


@dataclass(frozen=True)
class GlobalAvgMetric(SQAMetricBase):
    name: str = "global_avg"
    scorer: type[Scorer] = SQAJudgeScorer


SQA_METRICS = (
    GlobalAvgMetric(),
    IngredientRecallMetric(),
    AnswerPrecisionMetric(),
    CitationPrecisionMetric(),
    CitationRecallMetric(),
)

SQA_METRIC_LABELS = [
    "ingredient_recall",
    "answer_precision",
    "citation_precision",
    "citation_recall",
]

# =============================================================================
# Task
# =============================================================================


@register("astabench_scholarqa")
class AstaBenchSQA(Task):
    """AstaBench ScholArQA-CS2 scientific question answering task."""

    split = Split.TRAIN  # HF JSON files load as a single "train" split
    data_source = DataSource(
        path=ASTA_BENCH_REPO,
        data_files="tasks/sqa/rubrics_v1_recomputed.json",
        revision=ASTA_BENCH_REVISION,
        split="train",  # JSON files load as a single "train" split in HF
    )
    metrics = SQA_METRICS
    primary_metric = GlobalAvgMetric()
    sampling_params = SamplingParams(temperature=0.0, max_tokens=4096)
    # The LLM-judge scorer needs an OpenAI key at scoring time; the launcher
    # mounts {user}_OPENAI_API_KEY, or {user}_OPENROUTER_API_KEY in its place.
    required_secrets = ("OPENAI_API_KEY",)

    @property
    def instances(self) -> Iterator[Instance]:
        yield from self._load_instances_cached()

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        question = doc.get("question", "")
        if not question:
            return None

        # The HF dataset stores ingredients as a top-level list of dicts,
        # each with keys: criterion, examples, name, weight.
        ingredients = doc.get("ingredients", [])

        return Instance(
            question=question,
            metadata={
                "case_id": doc.get("case_id", f"sqa_{index}"),
                "annotator": doc.get("annotator", ""),
                "rubric_config": {"ingredients": ingredients},
                "index": index,
            },
        )

    def format_request(self, instance: Instance) -> LMRequest:
        prompt_text = SQA_GENERATION_PROMPT + instance.question
        return LMRequest(
            request_type=RequestType.CHAT,
            messages=({"role": "user", "content": prompt_text},),
        )

    def extract_answer(self, output: LMOutput) -> Any:
        """Parse JSON from model output and store structured response."""
        text = output.text
        # Strip <think>...</think> blocks (e.g. Qwen3) so that JSON
        # extraction doesn't grab braces from the reasoning trace.
        think_end = text.find("</think>")
        if think_end >= 0:
            text = text[think_end + len("</think>") :]
        parsed = extract_json_from_response(text)
        if parsed is not None:
            parsed = normalize_agent_response_dict(parsed)
        output.metadata["parsed_response"] = parsed
        return parsed

    async def score_responses(
        self,
        responses: Sequence[Response],
        context: Any = None,
    ) -> Sequence[Response]:
        """Score all responses using LLM-as-judge for 4 metrics.

        Overrides Task.score_responses because SQA needs multiple judge calls
        per instance producing 4+ scores (doesn't fit single-float Scorer.score).
        Scoring logic ported from astabench evals/sqa/task.py:SQATask.score.
        """
        self._extract_answers(responses)

        judge_fn = build_default_judge_fn(scorer_name="AstaBenchSQA")

        for response in responses:
            scores = await self._score_single(response, judge_fn)
            response.scores.update(scores)

        return responses

    async def _score_single(self, response: Response, judge_fn: JudgeFn) -> dict[str, float]:
        """Score a single response across all 4 metrics + global_avg."""
        output = response.outputs[0] if response.outputs else None
        if output is None:
            return {k: 0.0 for k in SQA_METRIC_LABELS + ["global_avg"]}

        parsed = output.metadata.get("parsed_response")
        if not parsed or not parsed.get("sections"):
            return {k: 0.0 for k in SQA_METRIC_LABELS + ["global_avg"]}

        rubric_config = response.instance.metadata.get("rubric_config", {})
        question = response.instance.question
        answer_text = format_report(parsed)

        # 1. Ingredient recall
        ingredient_score = await self._score_ingredients(
            judge_fn, question, answer_text, rubric_config
        )

        # 2. Answer precision
        precision_score = await self._score_precision(judge_fn, question, answer_text)

        # 3. Citation precision & recall
        citation_scores = await self._score_citations(judge_fn, parsed)

        scores = {
            "ingredient_recall": ingredient_score,
            "answer_precision": precision_score,
            "citation_precision": citation_scores.get("citation_precision", 0.0),
            "citation_recall": citation_scores.get("citation_recall", 0.0),
        }
        scores["global_avg"] = sum(scores[k] for k in SQA_METRIC_LABELS) / len(SQA_METRIC_LABELS)
        return scores

    async def _score_ingredients(
        self,
        judge_fn: JudgeFn,
        question: str,
        answer: str,
        rubric_config: dict[str, Any],
    ) -> float:
        """Score ingredient recall. From astabench rubric.py:_assess_properties_jointly."""
        ingredients = rubric_config.get("ingredients", [])
        has_criterion = [ing for ing in ingredients if ing.get("criterion")]
        if not has_criterion:
            return 0.0

        # Build criteria text
        criteria_parts = []
        for i, ing in enumerate(has_criterion, 1):
            examples_text = "\n".join(ing.get("examples", []))
            criteria_parts.append(
                f"<criterion>\n{i}. {ing['criterion']}\n<examples>\n{examples_text}\n</examples>\n</criterion>"
            )
        criteria = "\n".join(criteria_parts)

        user_prompt = (
            f"<question>{question}</question>\n<response>{answer}</response>\nCriteria:\n{criteria}"
        )
        raw = await judge_fn(user_prompt, system_prompt=INGREDIENT_SYSTEM_PROMPT)
        parsed = extract_json_from_response(raw)
        if not parsed:
            return 0.0

        return compute_ingredient_score(parsed, ingredients)

    async def _score_precision(
        self,
        judge_fn: JudgeFn,
        question: str,
        answer: str,
    ) -> float:
        """Score answer precision. From astabench precision_eval.py:PrecisionEval."""
        prompt = PRECISION_EVAL_PROMPT.format(query=question, answer=answer)
        raw = await judge_fn(prompt)
        parsed = extract_json_from_response(raw)
        if not parsed:
            return 1.0  # No irrelevant paragraphs identified = perfect precision

        score, _ = compute_precision_score(parsed, answer)
        return score

    async def _score_citations(
        self,
        judge_fn: JudgeFn,
        parsed_response: dict[str, Any],
    ) -> dict[str, float]:
        """Score citation precision and recall via the shared citation scorer."""
        return await score_citations_for_sections(judge_fn, parsed_response)


# =============================================================================
# Variant: test split
# =============================================================================

register_variant(
    "astabench_scholarqa",
    "test",
    data_source=DataSource(
        path=ASTA_BENCH_REPO,
        data_files="tasks/sqa/rubrics_v2_recomputed.json",
        revision=ASTA_BENCH_REVISION,
        split="train",
    ),
)
