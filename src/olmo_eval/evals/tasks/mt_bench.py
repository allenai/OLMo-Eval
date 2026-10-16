"""MT-Bench: two-turn conversations graded 1-10 by a model judge.

MT-Bench (Zheng et al., 2023, https://arxiv.org/abs/2306.05685) has 80 questions in
eight categories: writing, roleplay, reasoning, math, coding, extraction, STEM and
humanities. Each question has two user turns, and the second follows up on the
model's own answer to the first. That needs the ``multi_turn_chat`` scaffold::

    uv run olmo-eval run -m my-model --harness multi_turn_chat -t mt_bench

Data
----
``mt_bench.jsonl`` holds FastChat's ``question.jsonl``, with the GPT-4 reference
answers from ``reference_answer/gpt-4.jsonl`` attached to the math, reasoning and
coding questions, which are the ones FastChat grades against a reference. Both files
are taken from FastChat at commit ``587d5cfa1609a43d192cedb8441cac3c17db105d``.

Judge
-----
The judge gets FastChat's single-answer grading prompts verbatim. The first turn is
graded on its own. The second is graded with the whole conversation in view, and the
judge is told to focus on the second answer. Math, reasoning and coding prompts
include the reference answers. The judge replies with a ``[[rating]]`` from 1 to 10.
The default judge is ``gpt-5.5-2026-04-23`` at medium reasoning effort. It is a dated
snapshot, so the judge cannot change under an unchanged task hash. The paper judges
with ``gpt-4`` (``gpt-4-0613``), which OpenAI retires on 2026-10-23, so scores here are
not directly comparable with published MT-Bench numbers. Change the judge with
``-o judge_model=<model>`` (and ``-o judge_reasoning_effort=<effort>``), or for every
judged task in a run with ``OLMO_EVAL_JUDGE=<model>[:<effort>]``; a task override
wins. The judge settings are part of the task hash.

Scoring
-------
The judge sees each answer after any closed reasoning block. A turn with no visible
answer gets the lowest rating, 1, without calling the judge. That covers an empty
reply, a reasoning block that never closed, a second turn the conversation never
reached because it outgrew the context window.
A judge call that fails, or gives no rating after three attempts, leaves the question
unscored and the run's result is flagged incomplete.

Metrics are on FastChat's 1-10 scale:

- ``score`` (primary): the mean rating over both turns of every question. This is
  the number FastChat's ``show_result.py`` reports.
- ``score_turn_1`` and ``score_turn_2``: the mean rating on each turn.
- ``score_<category>``: the mean rating over both turns in one category.
- ``truncation_rate``: share of questions where a turn hit the generation token
  limit; lower is better.

Generation defaults to this repo's reasoning regime: temperature 0.6, top_p 0.95, no
token cap. ``mt_bench:paper`` uses FastChat's settings instead: up to 1,024 new
tokens, at a temperature that depends on the category (0.7 for writing and
roleplay, 0.1 for STEM and humanities, greedy otherwise). It rejects a temperature
override, which would otherwise be recorded but not used.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from importlib.resources import files
from typing import Any, ClassVar

from olmo_eval.common.scorers.base import Scorer
from olmo_eval.common.scorers.llm_judge import JudgeFn
from olmo_eval.common.types import (
    FOLLOW_UP_TURNS_KEY,
    REPLIES_KEY,
    TURN_FINISH_REASONS_KEY,
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
    JudgeParseError,
    JudgeScorer,
    TruncationRateMetric,
    visible_response,
)

MT_BENCH_DATA_FILE = "mt_bench.jsonl"
MT_BENCH_QUESTION_COUNT = 80
MT_BENCH_TURNS = (1, 2)

MT_BENCH_DEFAULT_JUDGE_MODEL = "gpt-5.5-2026-04-23"
MT_BENCH_DEFAULT_JUDGE_REASONING_EFFORT = "medium"
# Room for the judge's reasoning as well as FastChat's 2,048-token judgment.
MT_BENCH_JUDGE_MAX_TOKENS = 8192

MT_BENCH_SCORER_NAME = "mt_bench_judge"

MT_BENCH_CATEGORIES = (
    "writing",
    "roleplay",
    "reasoning",
    "math",
    "coding",
    "extraction",
    "stem",
    "humanities",
)
#: Categories FastChat grades against a reference answer.
MT_BENCH_REFERENCE_CATEGORIES = frozenset({"math", "reasoning", "coding"})

MIN_RATING = 1.0
MAX_RATING = 10.0

#: FastChat's generation temperature for each category.
FASTCHAT_TEMPERATURES: Mapping[str, float] = {
    "writing": 0.7,
    "roleplay": 0.7,
    "extraction": 0.0,
    "math": 0.0,
    "coding": 0.0,
    "reasoning": 0.0,
    "stem": 0.1,
    "humanities": 0.1,
}
FASTCHAT_MAX_NEW_TOKENS = 1024


@dataclass(frozen=True)
class JudgePrompt:
    """A FastChat judge prompt: the judge's system prompt and its user template."""

    system_prompt: str
    template: str


# The four single-answer grading prompts, verbatim from FastChat's
# ``data/judge_prompts.jsonl`` (``single-v1``, ``single-math-v1``,
# ``single-v1-multi-turn`` and ``single-math-v1-multi-turn``).
_RATING_INSTRUCTION = (
    "After providing your explanation, you must rate the response on a scale of 1 to 10 "
    'by strictly following this format: "[[rating]]", for example: "Rating: [[5]]".'
)

SINGLE_TURN_PROMPT = JudgePrompt(
    system_prompt="You are a helpful assistant.",
    template=(
        "[Instruction]\n"
        "Please act as an impartial judge and evaluate the quality of the response provided "
        "by an AI assistant to the user question displayed below. Your evaluation should "
        "consider factors such as the helpfulness, relevance, accuracy, depth, creativity, "
        "and level of detail of the response. Begin your evaluation by providing a short "
        f"explanation. Be as objective as possible. {_RATING_INSTRUCTION}\n\n"
        "[Question]\n{question}\n\n"
        "[The Start of Assistant's Answer]\n{answer}\n[The End of Assistant's Answer]"
    ),
)

SINGLE_TURN_REFERENCE_PROMPT = JudgePrompt(
    system_prompt="You are a helpful assistant.",
    template=(
        "[Instruction]\n"
        "Please act as an impartial judge and evaluate the quality of the response provided "
        "by an AI assistant to the user question displayed below. Your evaluation should "
        "consider correctness and helpfulness. You will be given a reference answer and the "
        "assistant's answer. Begin your evaluation by comparing the assistant's answer with "
        "the reference answer. Identify and correct any mistakes. Be as objective as "
        f"possible. {_RATING_INSTRUCTION}\n\n"
        "[Question]\n{question}\n\n"
        "[The Start of Reference Answer]\n{ref_answer_1}\n[The End of Reference Answer]\n\n"
        "[The Start of Assistant's Answer]\n{answer}\n[The End of Assistant's Answer]"
    ),
)

_CONVERSATION = (
    "<|The Start of Assistant A's Conversation with User|>\n\n"
    "### User:\n{question_1}\n\n"
    "### Assistant A:\n{answer_1}\n\n"
    "### User:\n{question_2}\n\n"
    "### Assistant A:\n{answer_2}\n\n"
    "<|The End of Assistant A's Conversation with User|>"
)

SECOND_TURN_PROMPT = JudgePrompt(
    system_prompt=(
        "Please act as an impartial judge and evaluate the quality of the response provided "
        "by an AI assistant to the user question displayed below. Your evaluation should "
        "consider factors such as the helpfulness, relevance, accuracy, depth, creativity, "
        "and level of detail of the response. You evaluation should focus on the "
        "assistant's answer to the second user question. Begin your evaluation by providing "
        f"a short explanation. Be as objective as possible. {_RATING_INSTRUCTION}\n\n"
    ),
    template=_CONVERSATION,
)

SECOND_TURN_REFERENCE_PROMPT = JudgePrompt(
    system_prompt=(
        "Please act as an impartial judge and evaluate the quality of the response provided "
        "by an AI assistant to the user question. Your evaluation should consider correctness "
        "and helpfulness. You will be given a reference answer and the assistant's answer. "
        "You evaluation should focus on the assistant's answer to the second question. Begin "
        "your evaluation by comparing the assistant's answer with the reference answer. "
        "Identify and correct any mistakes. Be as objective as possible. "
        f"{_RATING_INSTRUCTION}\n\n"
    ),
    template=(
        "<|The Start of Reference Answer|>\n\n"
        "### User:\n{question_1}\n\n"
        "### Reference answer:\n{ref_answer_1}\n\n"
        "### User:\n{question_2}\n\n"
        "### Reference answer:\n{ref_answer_2}\n\n"
        "<|The End of Reference Answer|>\n\n\n" + _CONVERSATION
    ),
)

# FastChat reads the first ``[[rating]]``, falling back to the first ``[rating]``.
_RATING = re.compile(r"\[\[(\d+\.?\d*)\]\]")
_RATING_BACKUP = re.compile(r"\[(\d+\.?\d*)\]")


def parse_rating(raw: str) -> float | None:
    """Read the judge's 1-10 rating, or None if the reply gives none in range."""
    match = _RATING.search(raw) or _RATING_BACKUP.search(raw)
    if match is None:
        return None
    rating = float(match.group(1))
    return rating if MIN_RATING <= rating <= MAX_RATING else None


def judge_prompt(instance: Instance, answers: Sequence[str], turn: int) -> JudgePrompt:
    """Return the judge prompt for one turn, filled in with the question and answers."""
    questions: list[str] = instance.metadata["turns"]
    reference: list[str] | None = instance.metadata["reference"]
    if turn == 1:
        prompt = SINGLE_TURN_PROMPT if reference is None else SINGLE_TURN_REFERENCE_PROMPT
        fields = {"question": questions[0], "answer": answers[0]}
    else:
        prompt = SECOND_TURN_PROMPT if reference is None else SECOND_TURN_REFERENCE_PROMPT
        fields = {
            "question_1": questions[0],
            "answer_1": answers[0],
            "question_2": questions[1],
            "answer_2": answers[1],
        }
    if reference is not None:
        fields.update(ref_answer_1=reference[0], ref_answer_2=reference[1])
    return replace(prompt, template=prompt.template.format(**fields))


@dataclass(frozen=True)
class MTBenchScorer(JudgeScorer):
    """Serves the MT-Bench judge's stored per-output rating."""

    name: str = MT_BENCH_SCORER_NAME


@dataclass(frozen=True)
class MTBenchScoreMetric(JudgedScoreMetric):
    """Mean 1-10 judge rating, optionally restricted to one category."""

    name: str = "score"
    scorer: type[Scorer] | Scorer = MTBenchScorer
    score_key: str = "score"

    def display_format(self) -> str:
        return "raw"

    def metric_unit(self) -> str:
        return "mt_bench_rating"


MT_BENCH_SCORE = MTBenchScoreMetric()
MT_BENCH_METRICS = (
    MT_BENCH_SCORE,
    *(
        MTBenchScoreMetric(name=f"score_turn_{turn}", score_key=f"turn_{turn}")
        for turn in MT_BENCH_TURNS
    ),
    *(
        MTBenchScoreMetric(
            name=f"score_{category}", metadata_key="category", metadata_value=category
        )
        for category in MT_BENCH_CATEGORIES
    ),
    TruncationRateMetric(scorer=MTBenchScorer),
)


@register("mt_bench")
class MTBench(JudgedTask):
    """MT-Bench two-turn questions, each turn rated 1-10 by a model judge."""

    data_source = DataSource(path=MT_BENCH_DATA_FILE, split="test")
    split = Split.TEST
    metrics = MT_BENCH_METRICS
    primary_metric = MT_BENCH_SCORE
    # max_tokens=None: generate up to the context limit.
    sampling_params = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=None)
    output_score_aggregation = OutputScoreAggregation.MEAN

    judge_model = MT_BENCH_DEFAULT_JUDGE_MODEL
    judge_reasoning_effort = MT_BENCH_DEFAULT_JUDGE_REASONING_EFFORT
    judge_max_tokens = MT_BENCH_JUDGE_MAX_TOKENS
    judge_scorer_name = MT_BENCH_SCORER_NAME
    judge_score_key = "score"

    #: Generation temperature per category, replacing the task-wide temperature.
    category_temperatures: ClassVar[Mapping[str, float] | None] = None

    @property
    def instances(self) -> Iterator[Instance]:
        if self._instances_cache is None:
            resource = files(__package__).joinpath(MT_BENCH_DATA_FILE)
            with resource.open(encoding="utf-8") as handle:
                docs = [json.loads(line) for line in handle if line.strip()]
            self._instances_cache = [
                instance
                for index, doc in enumerate(docs)
                if (instance := self.process_doc(doc, index)) is not None
            ]
        yield from self._instances_cache

    @property
    def request_type(self) -> RequestType:
        return RequestType.CHAT

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        turns = [str(turn) for turn in doc["turns"]]
        if len(turns) != len(MT_BENCH_TURNS):
            raise ValueError(f"MT-Bench question {doc['question_id']} has {len(turns)} turns")
        reference = doc.get("reference")
        return Instance(
            question=turns[0],
            gold_answer=None,
            metadata={
                "id": str(doc["question_id"]),
                "category": str(doc["category"]),
                "turns": turns,
                "reference": [str(answer) for answer in reference] if reference else None,
            },
        )

    def format_request(self, instance: Instance) -> LMRequest:
        first, second = instance.metadata["turns"]
        return LMRequest(
            request_type=RequestType.CHAT,
            messages=({"role": "user", "content": first},),
            metadata={FOLLOW_UP_TURNS_KEY: [{"role": "user", "content": second}]},
        )

    def get_sampling_params(self, instance: Instance) -> SamplingParams | None:
        params = self.config.sampling_params
        if self.category_temperatures is None or params is None:
            return params
        if params.temperature != type(self).sampling_params.temperature:
            raise ValueError(
                f"{self.config.name} sets the temperature per category; "
                "drop the temperature override."
            )
        temperature = self.category_temperatures[instance.metadata["category"]]
        return replace(params, temperature=temperature, do_sample=temperature > 0)

    def extract_answer(self, output: LMOutput) -> Any:
        """The visible answer to each turn, as a list, or None without the scaffold's replies."""
        replies = (output.metadata or {}).get(REPLIES_KEY)
        if replies is None:
            return None
        return [visible_response(reply) for reply in replies]

    async def score_responses(
        self,
        responses: Sequence[Response],
        context: Any = None,
    ) -> Sequence[Response]:
        """Judge both turns of every output, then aggregate each turn per instance.

        An output counts as truncated when any of its turns hit the token limit.
        """
        if any(
            REPLIES_KEY not in (output.metadata or {}) for r in responses for output in r.outputs
        ):
            raise ValueError(
                "An MT-Bench output has no per-turn replies. The task has to run under a "
                "harness with scaffold=multi_turn_chat, for example `--harness multi_turn_chat`."
            )
        await super().score_responses(responses, context)
        for response in responses:
            if not response.outputs:
                # No model output at all answers neither turn.
                response.scores["score"] = MIN_RATING
                response.scores.update({f"turn_{turn}": MIN_RATING for turn in MT_BENCH_TURNS})
                continue
            response.scores["truncation_rate"] = sum(
                "length" in output.metadata.get(TURN_FINISH_REASONS_KEY, ())
                for output in response.outputs
            ) / len(response.outputs)
            for turn in MT_BENCH_TURNS:
                ratings = {
                    index: output.metadata[f"score:turn_{turn}"]
                    for index, output in enumerate(response.outputs)
                    if f"score:turn_{turn}" in output.metadata
                }
                if ratings:
                    response.scores[f"turn_{turn}"] = self._aggregate_output_scores(ratings)
        return responses

    async def _score_output(self, response: Response, output: LMOutput, judge_fn: JudgeFn) -> float:
        """Rate both turns; a turn without a visible answer gets the lowest rating."""
        answers = list(output.extracted_answer or [])
        answers += [""] * (len(MT_BENCH_TURNS) - len(answers))
        # One turn at a time, so each output holds one judge call against the limiter.
        turn_results = [
            await self._rate_turn(response.instance, answers, turn, judge_fn)
            for turn in MT_BENCH_TURNS
        ]
        judge_result: dict[str, Any] = {"scorer": MT_BENCH_SCORER_NAME, "turns": turn_results}
        ratings = [result["rating"] for result in turn_results]
        if None in ratings:
            judge_result["parse_error"] = True
            output.metadata["judge_result"] = judge_result
            raise JudgeParseError("no parseable rating after every judge attempt")

        judge_result["parse_error"] = False
        scores = {
            f"turn_{turn}": rating for turn, rating in zip(MT_BENCH_TURNS, ratings, strict=True)
        }
        scores["score"] = sum(ratings) / len(ratings)
        return self._record_judgement(output, judge_result, scores)

    async def _rate_turn(
        self, instance: Instance, answers: Sequence[str], turn: int, judge_fn: JudgeFn
    ) -> dict[str, Any]:
        """Judge one turn and return its rating (None if unparseable) with the details."""
        if not answers[turn - 1]:
            return {"turn": turn, "rating": MIN_RATING, "no_final_response": True}
        prompt = judge_prompt(instance, answers, turn)
        rating, raw = await self._judge(
            judge_fn, prompt.template, parse_rating, system_prompt=prompt.system_prompt
        )
        return {
            "turn": turn,
            "rating": rating,
            "no_final_response": False,
            "raw_judge_response": raw,
        }


@register("mt_bench:paper")
class MTBenchPaper(MTBench):
    """MT-Bench with FastChat's generation settings."""

    sampling_params = SamplingParams(max_tokens=FASTCHAT_MAX_NEW_TOKENS)
    category_temperatures = FASTCHAT_TEMPERATURES
