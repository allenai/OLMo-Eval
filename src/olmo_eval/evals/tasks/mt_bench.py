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
with ``gpt-4`` (``gpt-4-0613``), which OpenAI is retiring, so scores here are not
directly comparable with published MT-Bench numbers. The judge can be changed like
that of any judged task (see :mod:`olmo_eval.evals.tasks.common.judge`).

Scoring
-------
The judge sees each answer after any closed reasoning block. A turn with no visible
answer gets the lowest rating, 1, without calling the judge. That covers an empty
reply, a reasoning block that never closed, and a second turn the conversation never
reached because it outgrew the context window. The second turn is still judged when
the first has no visible answer; the judge then sees an empty first answer.

A judge call that fails, or a reply with no rating from 1 to 10 after every retry,
leaves the question unscored, and the run's result is flagged incomplete. FastChat
instead keeps an out-of-range rating and drops only the turn it could not rate, so
the two give the same scores when every turn gets a rating from 1 to 10.

Metrics are on FastChat's 1-10 scale:

- ``score`` (primary): the mean rating over both turns of every question, the
  number FastChat's ``show_result.py`` reports.
- ``score_turn_1`` and ``score_turn_2``: the mean rating on each turn.
- ``score_<category>``: the mean rating over both turns in one category.
- ``truncation_rate``: share of questions where a turn hit the generation token
  limit or the conversation outgrew the context; lower is better.

Generation defaults to this repo's reasoning regime: temperature 0.6, top_p 0.95, no
token cap. ``mt_bench:paper`` uses FastChat's settings instead: up to 1,024 new
tokens, at a temperature that depends on the category (0.7 for writing and
roleplay, 0.1 for STEM and humanities, greedy otherwise). It rejects a temperature
override, which would otherwise be recorded but not used.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, ClassVar

from olmo_eval.common.multi_turn import missing_replies_error, scripted_chat_request, turn_replies
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

MT_BENCH_MIN_RATING = 1.0
MT_BENCH_MAX_RATING = 10.0

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
    return rating if MT_BENCH_MIN_RATING <= rating <= MT_BENCH_MAX_RATING else None


def judge_prompt(instance: Instance, answers: Sequence[str], turn: int) -> tuple[str, str]:
    """Return the judge's system prompt and its prompt for one turn of a question."""
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
    return prompt.system_prompt, prompt.template.format(**fields)


@dataclass(frozen=True)
class MTBenchScorer(JudgeScorer):
    """Serves the MT-Bench judge's stored per-output rating."""

    name: str = MT_BENCH_SCORER_NAME


def _rating_metric(name: str, score_key: str = "score", **fields: Any) -> JudgedScoreMetric:
    """A mean 1-10 judge rating, reported on its own scale."""
    return JudgedScoreMetric(
        name=name,
        scorer=MTBenchScorer,
        score_key=score_key,
        display="raw",
        unit="mt_bench_rating",
        **fields,
    )


MT_BENCH_SCORE = _rating_metric("score")
MT_BENCH_METRICS = (
    MT_BENCH_SCORE,
    *(_rating_metric(f"score_turn_{turn}", score_key=f"turn_{turn}") for turn in MT_BENCH_TURNS),
    *(
        _rating_metric(f"score_{category}", metadata_key="category", metadata_value=category)
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
    judge_extra_score_keys = tuple(f"turn_{turn}" for turn in MT_BENCH_TURNS)
    # No model output answers neither turn.
    unanswered_score = MT_BENCH_MIN_RATING

    #: Generation temperature per category, replacing the task-wide temperature.
    category_temperatures: ClassVar[Mapping[str, float] | None] = None

    @property
    def instances(self) -> Iterator[Instance]:
        if self._instances_cache is None:
            instances = list(self._load_packaged_instances(__package__, MT_BENCH_DATA_FILE))
            if len(instances) != MT_BENCH_QUESTION_COUNT:
                raise ValueError(
                    f"{MT_BENCH_DATA_FILE} holds {len(instances)} questions, "
                    f"expected {MT_BENCH_QUESTION_COUNT}"
                )
            misreferenced = [
                instance.metadata["id"]
                for instance in instances
                if (instance.metadata["reference"] is not None)
                != (instance.metadata["category"] in MT_BENCH_REFERENCE_CATEGORIES)
            ]
            if misreferenced:
                raise ValueError(
                    f"{MT_BENCH_DATA_FILE} gives reference answers to the wrong questions: "
                    f"{', '.join(misreferenced)}"
                )
            self._instances_cache = instances
        yield from self._instances_cache

    @property
    def request_type(self) -> RequestType:
        return RequestType.CHAT

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        question_id = doc["question_id"]
        category = str(doc["category"])
        turns = [str(turn) for turn in doc["turns"]]
        if len(turns) != len(MT_BENCH_TURNS):
            raise ValueError(f"MT-Bench question {question_id} has {len(turns)} turns")
        reference = doc.get("reference")
        return Instance(
            question=turns[0],
            gold_answer=None,
            metadata={
                "id": str(question_id),
                "category": category,
                "turns": turns,
                "reference": [str(answer) for answer in reference] if reference else None,
            },
        )

    def format_request(self, instance: Instance) -> LMRequest:
        return scripted_chat_request(instance.metadata["turns"])

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
        replies = turn_replies(output)
        return None if replies is None else [visible_response(reply) for reply in replies]

    async def score_responses(
        self,
        responses: Sequence[Response],
        context: Any = None,
    ) -> Sequence[Response]:
        if any(turn_replies(output) is None for r in responses for output in r.outputs):
            raise missing_replies_error("MT-Bench")
        return await super().score_responses(responses, context)

    async def _score_output(self, response: Response, output: LMOutput, judge_fn: JudgeFn) -> float:
        """Rate both turns; a turn without a visible answer gets the lowest rating."""
        answers = list(output.extracted_answer or [])
        answers += [""] * (len(MT_BENCH_TURNS) - len(answers))
        judge_result: dict[str, Any] = {"scorer": MT_BENCH_SCORER_NAME}
        ratings = await self._judge_together(
            *(
                self._rate_turn(response.instance, output, answers, turn, judge_fn, judge_result)
                for turn in MT_BENCH_TURNS
            )
        )
        judge_result["parse_error"] = False
        scores = {
            f"turn_{turn}": rating for turn, rating in zip(MT_BENCH_TURNS, ratings, strict=True)
        }
        scores["score"] = sum(ratings) / len(ratings)
        return self._record_judgement(output, judge_result, scores)

    async def _rate_turn(
        self,
        instance: Instance,
        output: LMOutput,
        answers: Sequence[str],
        turn: int,
        judge_fn: JudgeFn,
        judge_result: dict[str, Any],
    ) -> float:
        """Rate one turn, recording its rating and the judge's reply in ``judge_result``."""
        if answers[turn - 1]:
            system_prompt, prompt = judge_prompt(instance, answers, turn)
            rating = await self._judge_output(
                output,
                judge_fn,
                prompt,
                parse_rating,
                judge_result,
                system_prompt=system_prompt,
                raw_key=f"raw_judge_response_turn_{turn}",
            )
        else:
            judge_result[f"no_final_response_turn_{turn}"] = True
            rating = MT_BENCH_MIN_RATING
        judge_result[f"rating_turn_{turn}"] = rating
        return rating


@register("mt_bench:paper")
class MTBenchPaper(MTBench):
    """MT-Bench with FastChat's generation settings."""

    sampling_params = SamplingParams(max_tokens=FASTCHAT_MAX_NEW_TOKENS)
    category_temperatures = FASTCHAT_TEMPERATURES
