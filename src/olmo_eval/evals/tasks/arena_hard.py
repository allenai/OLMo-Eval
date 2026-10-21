"""Arena-Hard v2.0 (arXiv 2406.11939): open-ended chat quality judged against a baseline.

Arena-Hard-Auto v2.0 holds 750 prompts from Chatbot Arena: 500 hard prompts (253
coding, 247 math) and 250 creative-writing prompts. A model judge compares each
response with a fixed baseline model's response, so the score is a win rate against
that baseline, with 50% meaning parity.

Tasks
-----
``arena_hard_v2:hard_prompt``
    The 500 hard prompts against ``o3-mini-2025-01-31``. The primary metric is the
    style-controlled win rate, the configuration upstream recommends.

``arena_hard_v2:creative_writing``
    The 250 creative-writing prompts against ``gemini-2.0-flash-001``. The primary
    metric is the plain win rate, which is what upstream reports for this category.

The ``arena_hard_v2`` suite runs both. Questions and baseline answers come from
``lmarena-ai/arena-hard-auto`` at a pinned revision. The judge prompts and verdict
parsing follow the ``lmarena/arena-hard-auto`` code at commit ``196f6b82``.

Examples::

    olmo-eval run -m <model> -t arena_hard_v2
    olmo-eval run -m <model> -t arena_hard_v2:hard_prompt -o judge_model=gpt-4.1-mini

Judge
-----
The default judge is ``gpt-4.1-2025-04-14``, the snapshot behind upstream's GPT-4.1
leaderboard, called at temperature 0 with upstream's prompts. Upstream's official
judge is Gemini 2.5, whose scores run higher; compare a score only with others from
the same judge. Change the judge with ``-o judge_model=<model>`` (and
``-o judge_reasoning_effort=<effort>``), or for every judge task in a run with
``OLMO_EVAL_JUDGE=<model>[:<effort>]``; a task override wins. The judge settings
are part of the task configuration and hash.

Scoring
-------
Each response is judged twice, once in each answer position, to cancel position
bias. A verdict of ``A>>B`` or ``B>>A`` counts three times, as upstream weights it,
and a tie counts as half a win. The judge sees the response after any closed
reasoning block; a reasoning block that never closes leaves an empty response. A
judge call that fails, or gives no parseable verdict after three attempts, is
recorded as an incomplete score and left out of the metrics. A prompt the model gave
no output for, such as one longer than its context, counts as a decisive loss in both
games. Responses with no visible answer get neutral style, so that style control
does not excuse their losses as the losses of a short answer.

Metrics:

- ``win_rate``: the share of weighted battles the model wins.
- ``win_rate_style_controlled``: the win rate with response length and markdown
  density (headers, list items, bold spans) held at the leaderboard's average, so
  that a model is not rewarded only for writing longer or more formatted answers.
  Upstream fits one Bradley-Terry model over every leaderboard model at once. This
  task keeps the style coefficients and feature scaling from that joint fit on
  upstream's published GPT-4.1 judgments and fits only the evaluated model's
  strength, so a score does not depend on which other models were run. The
  constants are reproduced by ``scripts/internal/fit_arena_hard_style_model.py``.
  They were fit to GPT-4.1 verdicts, so under another judge read this metric as
  an approximation.
- ``win_rate_<subcategory>``: the plain win rate on the coding and math prompts.
- ``truncation_rate``: share of prompts whose response hit the generation token
  limit; lower is better. The judge still sees whatever visible text they contain.

Each output's verdicts, battle outcomes, and style features are kept in its
``judge_result``, so the metrics can be recomputed from saved predictions.

Generation samples at temperature 0.6 and top_p 0.95, since greedy decoding sends
reasoning models into repetition loops. The budget is 32,768 tokens, shortened to
fit models with a smaller context window.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass, replace
from functools import cache, cached_property
from typing import Any, ClassVar

import tiktoken

from olmo_eval.common.scorers.base import Scorer
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
from olmo_eval.data import DataLoader, DataSource
from olmo_eval.evals.tasks.common import OutputScoreAggregation, register
from olmo_eval.evals.tasks.common.judge import (
    JudgedScoreMetric,
    JudgedTask,
    JudgeScorer,
    TruncationRateMetric,
)

ARENA_HARD_REPO = "lmarena-ai/arena-hard-auto"
# Pinned so a Hub update cannot silently change the prompts or baseline answers.
ARENA_HARD_REVISION = "15f3746e21432264ce9b453999bde4f3c946d2e6"
ARENA_HARD_DATA_DIR = "data/arena-hard-v2.0"

ARENA_HARD_DEFAULT_JUDGE_MODEL = "gpt-4.1-2025-04-14"
ARENA_HARD_JUDGE_MAX_TOKENS = 16000

ARENA_HARD_SCORER_NAME = "arena_hard_judge"

#: How many battles a decisive (``>>``) verdict counts as.
STRONG_VERDICT_WEIGHT = 3

#: Verdict label -> outcomes for Assistant A, verbatim from upstream ``show_result.py``.
VERDICT_OUTCOMES: dict[str, tuple[float, ...]] = {
    "A>B": (1.0,),
    "A>>B": (1.0,) * STRONG_VERDICT_WEIGHT,
    "A=B": (0.5,),
    "A<<B": (0.0,) * STRONG_VERDICT_WEIGHT,
    "A<B": (0.0,),
    "B>A": (0.0,),
    "B>>A": (0.0,) * STRONG_VERDICT_WEIGHT,
    "B=A": (0.5,),
    "B<<A": (1.0,) * STRONG_VERDICT_WEIGHT,
    "B<A": (1.0,),
}

# Upstream tries these in order on the upper-cased reply and keeps the last match.
_VERDICT_PATTERNS = (re.compile(r"\[\[([AB<>=]+)\]\]"), re.compile(r"\[([AB<>=]+)\]"))

# Verbatim from upstream ``config/arena-hard-v2.0.yaml``.
ARENA_HARD_JUDGE_PROMPT = (
    "<|User Prompt|>\n{question}\n\n"
    "<|The Start of Assistant A's Answer|>\n{answer_a}\n<|The End of Assistant A's Answer|>\n\n"
    "<|The Start of Assistant B's Answer|>\n{answer_b}\n<|The End of Assistant B's Answer|>"
)

# Verbatim from upstream ``utils/judge_utils.py``. The creative-writing prompt is the
# same except that the judge does not write its own answer first.
_JUDGE_PROMPT_OPENING = (
    "Please act as an impartial judge and evaluate the quality of the responses provided by "
    "two AI assistants to the user prompt displayed below. You will be given assistant A's "
    "answer and assistant B's answer. Your job is to evaluate which assistant's answer is "
    "better.\n\n"
)
_JUDGE_PROMPT_OWN_ANSWER = (
    "Begin your evaluation by generating your own answer to the prompt. You must provide "
    "your answers before judging any answers.\n\n"
    "When evaluating the assistants' answers, compare both assistants' answers with your "
    "answer. You must identify and correct any mistakes or inaccurate information.\n\n"
)
_JUDGE_PROMPT_COMPARE_ONLY = (
    "When evaluating the assistants' answers, compare both assistants' answers. You must "
    "identify and correct any mistakes or inaccurate information.\n\n"
)
_JUDGE_PROMPT_CRITERIA = (
    "Then consider if the assistant's answers are helpful, relevant, and concise. Helpful "
    "means the answer correctly responds to the prompt or follows the instructions. Note when "
    "user prompt has any ambiguity or more than one interpretation, it is more helpful and "
    "appropriate to ask for clarifications or more information from the user than providing "
    "an answer based on assumptions. Relevant means all parts of the response closely connect "
    "or are appropriate to what is being asked. Concise means the response is clear and not "
    "verbose or excessive.\n\n"
    "Then consider the creativity and novelty of the assistant's answers when needed. "
    "Finally, identify any missing important information in the assistants' answers that "
    "would be beneficial to include when responding to the user prompt.\n\n"
    "After providing your explanation, you must output only one of the following choices as "
    "your final verdict with a label:\n\n"
    "1. Assistant A is significantly better: [[A>>B]]\n"
    "2. Assistant A is slightly better: [[A>B]]\n"
    "3. Tie, relatively the same: [[A=B]]\n"
    "4. Assistant B is slightly better: [[B>A]]\n"
    "5. Assistant B is significantly better: [[B>>A]]\n\n"
    'Example output: "My final verdict is tie: [[A=B]]".'
)
HARD_PROMPT_JUDGE_SYSTEM_PROMPT = (
    _JUDGE_PROMPT_OPENING + _JUDGE_PROMPT_OWN_ANSWER + _JUDGE_PROMPT_CRITERIA
)
CREATIVE_WRITING_JUDGE_SYSTEM_PROMPT = (
    _JUDGE_PROMPT_OPENING + _JUDGE_PROMPT_COMPARE_ONLY + _JUDGE_PROMPT_CRITERIA
)

# Upstream counts markdown with fenced code blocks removed. Its ``findall`` returns
# each block's contents, which are then deleted wherever they occur.
_CODE_BLOCK = re.compile(r"```([^`]*)```")
_HEADER_PATTERNS = tuple(re.compile(rf"^#{{{level}}}\s", re.MULTILINE) for level in range(1, 7))
_LIST_PATTERNS = (
    re.compile(r"^\s*\d+\.\s", re.MULTILINE),
    re.compile(r"^\s*[-*+]\s", re.MULTILINE),
)
_BOLD_PATTERNS = (re.compile(r"\*\*[^*\n]+\*\*"), re.compile(r"__[^_\n]+__"))

#: Tokenizer upstream measures response length with (``tiktoken`` for ``gpt-4o``).
STYLE_TOKENIZER = "o200k_base"


@dataclass(frozen=True)
class StyleCounts:
    """Length and markdown element counts of one response."""

    tokens: int
    headers: int
    lists: int
    bold: int


@cache
def _style_encoding() -> tiktoken.Encoding:
    return tiktoken.get_encoding(STYLE_TOKENIZER)


def count_tokens(text: str) -> int:
    """Count tokens the way upstream measures response length."""
    return len(_style_encoding().encode(text, disallowed_special=()))


def style_counts(text: str) -> StyleCounts:
    """Measure a response's length and markdown elements as upstream does."""
    markdown = text
    for block in _CODE_BLOCK.findall(text):
        markdown = markdown.replace(block, "")
    return StyleCounts(
        tokens=count_tokens(text),
        headers=sum(len(p.findall(markdown)) for p in _HEADER_PATTERNS),
        lists=sum(len(p.findall(markdown)) for p in _LIST_PATTERNS),
        bold=sum(len(p.findall(markdown)) for p in _BOLD_PATTERNS),
    )


def style_features(model: StyleCounts, baseline: StyleCounts) -> tuple[float, ...]:
    """Upstream's style contrasts between a response and the baseline's.

    The first is the relative length difference; the rest compare header, list, and
    bold density per token. Each lies in (-1, 1) and is 0 when the styles match.
    """
    total = model.tokens + baseline.tokens
    features = [(model.tokens - baseline.tokens) / total if total else 0.0]
    for model_count, baseline_count in (
        (model.headers, baseline.headers),
        (model.lists, baseline.lists),
        (model.bold, baseline.bold),
    ):
        model_density = model_count / (model.tokens + 1)
        baseline_density = baseline_count / (baseline.tokens + 1)
        features.append((model_density - baseline_density) / (model_density + baseline_density + 1))
    return tuple(features)


def parse_verdict(reply: str) -> str | None:
    """Read the judge's final verdict label, or None when it gave no known label."""
    upper = reply.upper()
    for pattern in _VERDICT_PATTERNS:
        matches = [match for match in pattern.findall(upper) if match]
        if matches:
            label = matches[-1].strip("\n")
            return label if label in VERDICT_OUTCOMES else None
    return None


def battle_outcomes(baseline_first: str, model_first: str) -> tuple[float, ...]:
    """The model's weighted battle outcomes from the two games' verdicts.

    ``baseline_first`` is the verdict with the baseline as Assistant A and
    ``model_first`` the verdict with the positions swapped. 1 is a model win.
    """
    return VERDICT_OUTCOMES[model_first] + tuple(
        1.0 - outcome for outcome in VERDICT_OUTCOMES[baseline_first]
    )


def sigmoid(x: float) -> float:
    """The logistic function, computed without overflow."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def fit_win_rate(outcomes: Sequence[float], offsets: Sequence[float]) -> float:
    """Fit a model's Bradley-Terry strength against the baseline given fixed style offsets.

    Maximizes the likelihood of ``outcome ~ sigmoid(strength + offset)`` over battles
    and returns ``sigmoid(strength)``, the win rate at zero offset. The score equation
    is monotone in the strength, so bisection finds the maximum, including the 0 or 1
    limit when every battle went one way.
    """
    if len(outcomes) != len(offsets):
        raise ValueError("outcomes and offsets must have the same length")
    if not outcomes:
        return 0.0
    target = sum(outcomes)
    low, high = -40.0, 40.0
    for _ in range(100):
        middle = (low + high) / 2
        expected = sum(sigmoid(middle + offset) for offset in offsets)
        if expected < target:
            low = middle
        else:
            high = middle
    return sigmoid((low + high) / 2)


@dataclass(frozen=True)
class StyleModel:
    """Style coefficients and feature scaling from upstream's joint leaderboard fit."""

    means: tuple[float, ...]
    stds: tuple[float, ...]
    coefs: tuple[float, ...]

    def offset(self, features: Sequence[float]) -> float:
        """The style term a battle contributes to the Bradley-Terry logit."""
        return sum(
            coef * (feature - mean) / std
            for feature, mean, std, coef in zip(
                features, self.means, self.stds, self.coefs, strict=True
            )
        )


# Reproduced by scripts/internal/fit_arena_hard_style_model.py from the GPT-4.1
# judgments at ARENA_HARD_REVISION.
HARD_PROMPT_STYLE_MODEL = StyleModel(
    means=(0.0318839, 0.00184698, 0.00847869, 0.00859469),
    stds=(0.356366, 0.00459669, 0.0122768, 0.0131325),
    coefs=(1.18056, 0.0666117, 0.040301, 0.0769705),
)
CREATIVE_WRITING_STYLE_MODEL = StyleModel(
    means=(0.00635942, 0.000418131, -0.00126771, -0.00286982),
    stds=(0.38662, 0.00222501, 0.0109714, 0.0171662),
    coefs=(1.24336, 0.0128502, -0.1984, -0.0310358),
)


@dataclass(frozen=True)
class ArenaHardScorer(JudgeScorer):
    """Serves the win rate the Arena-Hard judge stored for each output."""

    name: str = ARENA_HARD_SCORER_NAME


#: Battles of a prompt the model gave no output for: decisive losses in both games.
UNANSWERED_BATTLE_OUTCOMES = battle_outcomes("A>>B", "B>>A")
#: Style contrasts of a response with no visible answer. Neutral, because the style
#: model expects a short answer to lose and would otherwise credit the model for the
#: losses an unfinished or missing response earns.
NO_ANSWER_STYLE_FEATURES = (0.0, 0.0, 0.0, 0.0)


def _judged_battles(response: Response) -> Iterator[tuple[list[float], list[float]]]:
    """Yield each judged output's battle outcomes and style features.

    A prompt with no model output counts as decisive losses in both games, so prompts
    the model could not answer lower the win rate instead of dropping out.
    """
    if not response.outputs:
        yield list(UNANSWERED_BATTLE_OUTCOMES), list(NO_ANSWER_STYLE_FEATURES)
        return
    for output in response.outputs:
        judge_result = (output.metadata or {}).get("judge_result")
        if isinstance(judge_result, dict) and "battle_outcomes" in judge_result:
            yield judge_result["battle_outcomes"], judge_result["style_features"]


@dataclass(frozen=True)
class ArenaHardWinRateMetric(JudgedScoreMetric):
    """Win rate against the baseline, pooled over every judged battle.

    Decisive verdicts count as several battles, so pooling weights prompts unequally,
    as upstream does. With a ``style_model``, the rate is fitted with response style
    held at the leaderboard average; a fitted strength has no per-instance
    equivalent. ``metadata_key`` and ``metadata_value`` restrict it to one slice.
    """

    name: str = "win_rate"
    scorer: type[Scorer] | Scorer = ArenaHardScorer
    score_key: str = "win_rate"
    style_model: StyleModel | None = None

    def compute_instance(self, response: Response) -> float | None:
        if self.style_model is not None:
            return None
        return super().compute_instance(response)

    def compute(self, responses: Sequence[Response]) -> float:
        outcomes: list[float] = []
        offsets: list[float] = []
        for response in responses:
            if not self.in_slice(response):
                continue
            for battles, features in _judged_battles(response):
                outcomes.extend(battles)
                if self.style_model is not None:
                    offsets.extend([self.style_model.offset(features)] * len(battles))
        if self.style_model is not None:
            return fit_win_rate(outcomes, offsets)
        return sum(outcomes) / len(outcomes) if outcomes else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize, including the style constants, so refitting them changes the task hash."""
        serialized = super().to_dict()
        if self.style_model is not None:
            serialized["style_model"] = asdict(self.style_model)
        return serialized


def baseline_answer(record: dict[str, Any]) -> str:
    """The answer text of one record in an upstream ``model_answer`` file."""
    return str(record["messages"][-1]["content"]["answer"])


@dataclass(frozen=True)
class ArenaHardCategory:
    """One Arena-Hard category: its prompts, baseline, and judge instructions."""

    name: str
    baseline_model: str
    judge_system_prompt: str

    @property
    def baseline_answers_file(self) -> str:
        """The baseline model's answers, relative to the dataset repository root."""
        return f"{ARENA_HARD_DATA_DIR}/model_answer/{self.baseline_model}.jsonl"


HARD_PROMPT = ArenaHardCategory(
    name="hard_prompt",
    baseline_model="o3-mini-2025-01-31",
    judge_system_prompt=HARD_PROMPT_JUDGE_SYSTEM_PROMPT,
)
CREATIVE_WRITING = ArenaHardCategory(
    name="creative_writing",
    baseline_model="gemini-2.0-flash-001",
    judge_system_prompt=CREATIVE_WRITING_JUDGE_SYSTEM_PROMPT,
)

#: Game name -> whether the baseline answer is Assistant A in that game.
_GAMES = {"baseline_first": True, "model_first": False}


class _ArenaHardV2(JudgedTask):
    """Shared Arena-Hard v2.0 loading, prompting, and pairwise judging."""

    category: ClassVar[ArenaHardCategory]

    data_source = DataSource(
        path=ARENA_HARD_REPO,
        data_files=f"{ARENA_HARD_DATA_DIR}/question.jsonl",
        revision=ARENA_HARD_REVISION,
        split="train",
    )
    split = Split.TRAIN  # JSONL data files load as a single "train" split
    sampling_params = SamplingParams(
        temperature=0.6, top_p=0.95, max_tokens=32768, fit_max_tokens_to_context=True
    )
    output_score_aggregation = OutputScoreAggregation.MEAN

    judge_model = ARENA_HARD_DEFAULT_JUDGE_MODEL
    judge_max_tokens = ARENA_HARD_JUDGE_MAX_TOKENS
    judge_scorer_name = ARENA_HARD_SCORER_NAME
    judge_score_key = "win_rate"

    @property
    def instances(self) -> Iterator[Instance]:
        yield from self._load_instances_cached()

    @cached_property
    def baseline_answers(self) -> dict[str, str]:
        """The baseline model's answers, keyed by question id."""
        source = replace(self.config.data_source, data_files=self.category.baseline_answers_file)
        return {str(record["uid"]): baseline_answer(record) for record in DataLoader().load(source)}

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        if doc.get("category") != self.category.name:
            return None
        uid = str(doc["uid"])
        baseline = self.baseline_answers.get(uid)
        if baseline is None:
            raise ValueError(
                f"Arena-Hard question {uid} has no {self.category.baseline_model} answer"
            )
        return Instance(
            question=str(doc["prompt"]),
            metadata={
                "id": uid,
                "category": self.category.name,
                "subcategory": str(doc.get("subcategory") or ""),
                "baseline_model": self.category.baseline_model,
                "baseline_answer": baseline,
                "index": index,
            },
        )

    def format_request(self, instance: Instance) -> LMRequest:
        """Send the prompt alone, as upstream generates answers."""
        return LMRequest(
            request_type=RequestType.CHAT,
            messages=({"role": "user", "content": instance.question},),
        )

    async def _score_output(self, response: Response, output: LMOutput, judge_fn: JudgeFn) -> float:
        """Judge one output in both positions, record its battles, and return its win rate."""
        answer = self._judged_answer(output)
        instance = response.instance
        baseline = instance.metadata["baseline_answer"]
        # Measured before judging so a failure here does not waste judge calls.
        features = (
            style_features(style_counts(answer), style_counts(baseline))
            if answer
            else NO_ANSWER_STYLE_FEATURES
        )
        judge_result: dict[str, Any] = {
            "scorer": ARENA_HARD_SCORER_NAME,
            "baseline_model": instance.metadata["baseline_model"],
        }

        async def judge_game(game: str, baseline_is_a: bool) -> str:
            answer_a, answer_b = (baseline, answer) if baseline_is_a else (answer, baseline)
            prompt = ARENA_HARD_JUDGE_PROMPT.format(
                question=instance.question, answer_a=answer_a, answer_b=answer_b
            )
            verdict = await self._judge_output(
                output,
                judge_fn,
                prompt,
                parse_verdict,
                judge_result,
                system_prompt=self.category.judge_system_prompt,
                raw_key=f"raw_judge_response_{game}",
            )
            judge_result[f"verdict_{game}"] = verdict
            return verdict

        baseline_first, model_first = await self._judge_together(
            *(judge_game(game, baseline_is_a) for game, baseline_is_a in _GAMES.items())
        )
        outcomes = battle_outcomes(baseline_first, model_first)
        judge_result.update(
            battle_outcomes=list(outcomes),
            style_features=list(features),
            parse_error=False,
        )
        return self._record_judgement(
            output, judge_result, {"win_rate": sum(outcomes) / len(outcomes)}
        )


HARD_PROMPT_WIN_RATE_STYLE_CONTROLLED = ArenaHardWinRateMetric(
    name="win_rate_style_controlled", style_model=HARD_PROMPT_STYLE_MODEL
)
CREATIVE_WRITING_WIN_RATE = ArenaHardWinRateMetric()


@register("arena_hard_v2:hard_prompt")
class ArenaHardV2HardPrompt(_ArenaHardV2):
    """Arena-Hard v2.0 hard prompts, judged against o3-mini."""

    category = HARD_PROMPT
    metrics = (
        HARD_PROMPT_WIN_RATE_STYLE_CONTROLLED,
        ArenaHardWinRateMetric(),
        *(
            ArenaHardWinRateMetric(
                name=f"win_rate_{subcategory}",
                metadata_key="subcategory",
                metadata_value=subcategory,
            )
            for subcategory in ("coding", "math")
        ),
        TruncationRateMetric(scorer=ArenaHardScorer),
    )
    primary_metric = HARD_PROMPT_WIN_RATE_STYLE_CONTROLLED


@register("arena_hard_v2:creative_writing")
class ArenaHardV2CreativeWriting(_ArenaHardV2):
    """Arena-Hard v2.0 creative-writing prompts, judged against Gemini 2.0 Flash."""

    category = CREATIVE_WRITING
    metrics = (
        CREATIVE_WRITING_WIN_RATE,
        ArenaHardWinRateMetric(
            name="win_rate_style_controlled", style_model=CREATIVE_WRITING_STYLE_MODEL
        ),
        TruncationRateMetric(scorer=ArenaHardScorer),
    )
    primary_metric = CREATIVE_WRITING_WIN_RATE
