"""MultiChallenge: multi-turn conversations graded against per-conversation rubrics.

MultiChallenge (arXiv 2501.17399) tests whether a model handles a realistic
multi-turn conversation. Each item is a conversation that ends in a user turn, and
the model writes the next assistant turn. A model judge then answers a YES/NO
question about that response, written for that conversation, and the response
passes when the judge's answer matches the item's expected answer. Items cover four
axes:

- ``INFERENCE_MEMORY``: use details the user gave in earlier turns without being
  reminded.
- ``INSTRUCTION_RETENTION``: keep following an instruction from the first turn.
- ``SELF_COHERENCE``: stay consistent with earlier assistant turns instead of
  agreeing with a user who contradicts them.
- ``RELIABLE_VERSION_EDITING``: edit the right version of a document that went
  through several revisions.

Data
----
``ScaleAI/MultiChallenge``, pinned to a fixed Hub revision: 266 conversations.
This is the set Scale revised in its 2026 update, which tightened 54 rubrics and
removed ambiguous items from the paper's original 273. Scores from before and after
the update are not comparable.

Judge
-----
The judge gets the paper's prompt verbatim: the model's response and the rubric
question, with the instruction to be very strict. It sees only the final response,
not the conversation. The paper judges with ``gpt-4o-2024-08-06``, a dated
snapshot, which is the default here. Scale's current leaderboard judges with
Gemini 2.5 Pro, so its numbers are not directly comparable. The paper asks for a
structured reply; here the verdict is read from plain text, so any chat model can
judge. Change the judge with ``-o judge_model=<model>`` (and
``-o judge_reasoning_effort=<effort>`` for reasoning judges), or for every judge task
in a run with ``OLMO_EVAL_JUDGE=<model>[:<effort>]``; a task override wins. The
judge settings are part of the task hash.

Examples::

    olmo-eval run -m <model> -t multichallenge
    olmo-eval run -m <model> -t multichallenge -o judge_model=gpt-4.1-2025-04-14

Scoring
-------
A closed reasoning block is removed before judging. A reasoning block that never
closes, or a response that hit the generation token limit, scores as a fail without
calling the judge. A judge call that fails, or gives no parseable verdict after
three attempts, is recorded as an incomplete score and left out of the metrics; the
run's result is flagged incomplete. Upstream instead counts a judge error as a fail.

Metrics:

- ``macro_accuracy`` (primary): the mean of the four per-axis pass rates. This is
  the paper's overall score.
- ``accuracy``: the pass rate over all conversations.
- ``accuracy_<axis>``: the pass rate on each axis.
- ``truncation_rate``: share of conversations with no complete response; lower is
  better. These already count as fails.

With several samples per conversation, each conversation scores the mean over its
samples. The paper's ``--attempts`` setting instead passes a conversation when any
attempt passes; ``-o output_score_aggregation=max`` reproduces that.

Generation samples at temperature 0.6 and top_p 0.95 up to the context limit, the
regime this repo's other post-training tasks use; the upstream example generates at
temperature 0. The longest conversation is about 8.3k tokens with the Qwen3
tokenizer; the median is about 1.3k.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

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
    JudgedMacroScoreMetric,
    JudgedScoreMetric,
    JudgedTask,
    JudgeScorer,
    TruncationRateMetric,
    is_truncated,
)

logger = logging.getLogger(__name__)

MULTICHALLENGE_REPO = "ScaleAI/MultiChallenge"
# Pinned so a Hub update cannot silently change the conversations mid-comparison.
MULTICHALLENGE_REVISION = "670fd81f270314d0f8f82b34fcfe2187fd28f699"

MULTICHALLENGE_DEFAULT_JUDGE_MODEL = "gpt-4o-2024-08-06"
MULTICHALLENGE_JUDGE_MAX_TOKENS = 4096

MULTICHALLENGE_SCORER_NAME = "multichallenge_judge"

#: Upstream axis label -> metric suffix.
MULTICHALLENGE_AXES = {
    "INFERENCE_MEMORY": "inference_memory",
    "INSTRUCTION_RETENTION": "instruction_retention",
    "SELF_COHERENCE": "self_coherence",
    "RELIABLE_VERSION_EDITING": "reliable_version_editing",
}

# Verbatim from the official ``src/evaluator.py``.
MULTICHALLENGE_JUDGE_PROMPT = (
    "You are tasked with evaluating a model response to see if it meets a specific criteria.\n"
    "The criteria will always be YES/NO evaluation.\n\n"
    "The model response is as follows:\n"
    "<MODEL_RESPONSE>\n"
    "{response}\n"
    "</MODEL_RESPONSE>\n\n"
    "The criteria that the model response must meet is as follows. Be VERY STRICT!:\n"
    "<CRITERIA>\n"
    "{criteria}\n"
    "</CRITERIA>\n\n"
    'Print your reasoning followed by your verdict, either "YES" or "NO".'
)

# A verdict label counts only at the start of a line, so prose that mentions a
# verdict ("merits a YES verdict. No issues remain.") is not read as one. Judges wrap
# the label in markdown emphasis or headings and sometimes quote the value.
_LABELED_VERDICT = re.compile(
    r"^[\s>*_#-]*(?:final\s+)?verdict[\s*_]*:?[\s*_]*['\"`]?(YES|NO)\b",
    re.IGNORECASE | re.MULTILINE,
)
# Without a label, only an uppercase YES/NO counts, so the reasoning's prose does not.
_BARE_VERDICT = re.compile(r"\b(YES|NO)\b")


def parse_verdict(raw: str) -> str | None:
    """Read the judge's YES/NO verdict, or None if the reply gives none.

    A labeled verdict wins; otherwise the last uppercase YES or NO is taken, since
    the prompt asks for the reasoning first and the verdict last.
    """
    matches = list(_LABELED_VERDICT.finditer(raw)) or list(_BARE_VERDICT.finditer(raw))
    return matches[-1].group(1).upper() if matches else None


def _messages(conversation: Any) -> tuple[dict[str, str], ...]:
    """Return the conversation as chat messages.

    The Hub stores the conversation column-wise, as ``{"role": [...], "content":
    [...]}``; a list of message dicts is accepted as well.
    """
    if isinstance(conversation, Mapping):
        turns = zip(conversation["role"], conversation["content"], strict=True)
        return tuple({"role": role, "content": content} for role, content in turns)
    return tuple({"role": turn["role"], "content": turn["content"]} for turn in conversation)


@dataclass(frozen=True)
class MultiChallengeScorer(JudgeScorer):
    """Serves the MultiChallenge judge's stored per-output pass."""

    name: str = MULTICHALLENGE_SCORER_NAME


MULTICHALLENGE_MACRO_ACCURACY = JudgedMacroScoreMetric(
    scorer=MultiChallengeScorer,
    metadata_key="axis",
    groups=tuple(MULTICHALLENGE_AXES),
)
MULTICHALLENGE_METRICS = (
    MULTICHALLENGE_MACRO_ACCURACY,
    JudgedScoreMetric(scorer=MultiChallengeScorer),
    TruncationRateMetric(scorer=MultiChallengeScorer),
    *(
        JudgedScoreMetric(
            name=f"accuracy_{suffix}",
            scorer=MultiChallengeScorer,
            metadata_key="axis",
            metadata_value=axis,
        )
        for axis, suffix in MULTICHALLENGE_AXES.items()
    ),
)


@register("multichallenge")
class MultiChallenge(JudgedTask):
    """MultiChallenge multi-turn conversations, judged against each item's rubric."""

    data_source = DataSource(
        path=MULTICHALLENGE_REPO, revision=MULTICHALLENGE_REVISION, split="test"
    )
    split = Split.TEST
    metrics = MULTICHALLENGE_METRICS
    primary_metric = MULTICHALLENGE_MACRO_ACCURACY
    # max_tokens=None: generate up to the context limit.
    sampling_params = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=None)
    output_score_aggregation = OutputScoreAggregation.MEAN

    judge_model = MULTICHALLENGE_DEFAULT_JUDGE_MODEL
    judge_max_tokens = MULTICHALLENGE_JUDGE_MAX_TOKENS
    judge_scorer_name = MULTICHALLENGE_SCORER_NAME

    @property
    def instances(self) -> Iterator[Instance]:
        yield from self._load_instances_cached()

    @property
    def request_type(self) -> RequestType:
        return RequestType.CHAT

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        messages = _messages(doc["conversation"])
        if not messages or messages[-1]["role"] != "user":
            logger.warning("MultiChallenge: conversation %d does not end in a user turn", index)
            return None

        axis = str(doc["axis"])
        if axis not in MULTICHALLENGE_AXES:
            logger.warning("MultiChallenge: unexpected axis %r at index %d", axis, index)
        pass_criteria = str(doc["pass_criteria"]).strip().upper()
        return Instance(
            question=str(doc["target_question"]),
            gold_answer=pass_criteria,
            metadata={
                "id": str(doc["question_id"]),
                "axis": axis,
                "messages": messages,
                "index": index,
            },
        )

    def format_request(self, instance: Instance) -> LMRequest:
        return LMRequest(
            request_type=RequestType.CHAT,
            messages=tuple(instance.metadata["messages"]),
        )

    async def _score_output(self, response: Response, output: LMOutput, judge_fn: JudgeFn) -> float:
        """Judge one output; an unfinished response fails without the judge."""
        answer = self._judged_answer(output)
        truncated = is_truncated(output)
        pass_criteria = response.instance.gold_answer
        judge_result: dict[str, Any] = {
            "scorer": MULTICHALLENGE_SCORER_NAME,
            "truncated": truncated,
            "pass_criteria": pass_criteria,
        }
        if truncated or not answer:
            judge_result["no_final_response"] = True
            passed = False
        else:
            prompt = MULTICHALLENGE_JUDGE_PROMPT.format(
                response=answer, criteria=response.instance.question
            )
            verdict = await self._judge_output(
                output, judge_fn, prompt, parse_verdict, judge_result
            )
            judge_result["verdict"] = verdict
            passed = verdict == pass_criteria

        judge_result.update(passed=passed, parse_error=False)
        return self._record_judgement(output, judge_result, {"accuracy": 1.0 if passed else 0.0})
