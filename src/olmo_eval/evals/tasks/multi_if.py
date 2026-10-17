"""Multi-IF: multi-turn, multilingual instruction following.

Dataset: ``facebook/Multi-IF`` (4,501 conversations in English, Chinese,
French, Hindi, Italian, Portuguese, Russian and Spanish). Each conversation has
two or three user turns, and each turn adds IFEval instructions to the ones
before it. The model answers every turn itself, so a later turn is asked with
its own earlier replies in the conversation. That needs the ``multi_turn_chat``
scaffold:

    uv run olmo-eval run -m my-model --harness multi_turn_chat -t multi_if

Each reply is checked against its turn's instructions with the multilingual
verifiers from :mod:`olmo_eval.common.scorers.multilingual_instructions`. A
turn scores the mean of the prompt- and instruction-level accuracies under
strict and loose scoring, averaged over languages with equal weight, as in the
paper's tables (He et al., 2024, https://arxiv.org/abs/2410.15553). The
primary metric averages the three turns.

Defaults follow the post-training reasoning regime shared by this repo's other
instruction-following tasks, and reasoning inside ``<think>`` tags is removed
before verification. Generation is bounded only by the context window, so a
very long reply can leave no room for later turns, which then score as
unanswered. The ``:paper`` variant uses the paper's sampling, with 1,024
tokens per reply.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

from olmo_eval.common.metrics import MultiIFAccuracy
from olmo_eval.common.multi_turn import scripted_chat_request, turn_replies
from olmo_eval.common.types import (
    Instance,
    LMOutput,
    LMRequest,
    RequestType,
    SamplingParams,
    Split,
)
from olmo_eval.data import DataSource
from olmo_eval.evals.extract import extract_think_answer
from olmo_eval.evals.tasks.common import Task, register, register_variant

_DATASET_PATH = "facebook/Multi-IF"
_DATASET_REVISION = "0ab97ce0b45c7f57772e8ba2ac1616f4b00bd3aa"
_MAX_TURNS = 3
LANGUAGES = (
    "Chinese",
    "English",
    "French",
    "Hindi",
    "Italian",
    "Portuguese",
    "Russian",
    "Spanish",
)

_PRIMARY_METRIC = MultiIFAccuracy()


def _parse_turn(doc: dict[str, Any], turn: int) -> dict[str, Any] | None:
    """One turn's user prompt and instructions, or None past the conversation's end."""
    raw_prompt = doc.get(f"turn_{turn}_prompt")
    if not raw_prompt or raw_prompt == "None":
        return None
    kwargs_list = [json.loads(kwargs) for kwargs in json.loads(doc[f"turn_{turn}_kwargs"])]
    return {
        "prompt": json.loads(raw_prompt)["content"],
        "instruction_id_list": json.loads(doc[f"turn_{turn}_instruction_id_list"]),
        "kwargs": [{k: v for k, v in kwargs.items() if v is not None} for kwargs in kwargs_list],
    }


@register("multi_if")
class MultiIF(Task):
    data_source = DataSource(path=_DATASET_PATH, split="train", revision=_DATASET_REVISION)
    split = Split.TRAIN
    required_scaffold = "multi_turn_chat"
    metrics = (
        _PRIMARY_METRIC,
        *(
            MultiIFAccuracy(name=f"turn_{turn}_accuracy", turn=turn)
            for turn in range(1, _MAX_TURNS + 1)
        ),
        *(
            MultiIFAccuracy(name=f"{language.lower()}_accuracy", language=language)
            for language in LANGUAGES
        ),
    )
    primary_metric = _PRIMARY_METRIC
    strip_thinking = True
    # max_tokens=None: generate up to the context limit.
    sampling_params = SamplingParams(
        max_tokens=None,
        temperature=0.6,
        top_p=0.95,
    )

    @property
    def instances(self) -> Iterator[Instance]:
        yield from self._load_instances_cached()

    @property
    def request_type(self) -> RequestType:
        return RequestType.CHAT

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        turns = []
        for turn in range(1, _MAX_TURNS + 1):
            parsed = _parse_turn(doc, turn)
            if parsed is None:
                break
            turns.append(parsed)
        if not turns:
            return None
        return Instance(
            question=turns[0]["prompt"],
            gold_answer=None,
            metadata={
                "id": doc["key"],
                "key": doc["key"],
                "language": doc["language"],
                "turns": turns,
            },
        )

    def format_request(self, instance: Instance) -> LMRequest:
        turns = instance.metadata["turns"]
        return scripted_chat_request([turn["prompt"] for turn in turns])

    def extract_answer(self, output: LMOutput) -> list[str] | None:
        """The reply to each turn, trimmed as the reference implementation does.

        Each reply's reasoning trace is dropped first when ``strip_thinking`` is set.
        """
        replies = turn_replies(output)
        if replies is None:
            return None
        if self.config.strip_thinking:
            replies = [extract_think_answer(reply) or "" for reply in replies]
        return [reply.strip() for reply in replies]


# The paper's generation settings (Table 3).
register_variant(
    "multi_if",
    "paper",
    sampling_params=SamplingParams(max_tokens=1024, temperature=0.6, top_p=0.9),
)
