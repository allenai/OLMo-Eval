"""PopQA: open-domain QA over entities spanning a wide popularity range.

PopQA (https://arxiv.org/abs/2212.10511) probes factual recall with
template-generated questions from Wikidata triples, weighted toward
long-tail entities. A response is correct if any accepted alias of the
answer appears in it.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from olmo_eval.common.formatters import ChatFormatter
from olmo_eval.common.metrics import AccuracyMetric
from olmo_eval.common.scorers.base import Scorer
from olmo_eval.common.types import (
    Instance,
    LMOutput,
    LMRequest,
    RequestType,
    SamplingParams,
    Split,
)
from olmo_eval.data import DataSource
from olmo_eval.evals.tasks.common import Task, register, register_variant
from olmo_eval.evals.tasks.constants.popqa import POPQA_FIXED_FEWSHOT


def _format_query(question: str) -> str:
    return f"Q: {question} A:"


@dataclass(frozen=True, slots=True)
class PopQAContainsScorer(Scorer):
    """Alias containment scorer for PopQA.

    A response is correct if any accepted alias appears in it verbatim,
    lowercased, or capitalized — the matching rule used by the reference
    implementation. With ``first_line`` set, only the first non-blank line
    is searched: models that ramble past their answer invent follow-up Q/A
    pairs full of plausible entity names, and whole-output containment
    credits those accidental hits. Leading blank lines are skipped rather
    than treated as the answer — a reasoning model's answer starts on the
    line after the stripped ``<think>`` trace.
    """

    name: str = "popqa_contains"
    first_line: bool = False

    def score(self, instance: Instance, output: LMOutput) -> float:
        response = output.text or ""
        if self.first_line:
            response = next(
                (line for line in response.split("\n") if line.strip()),
                "",
            )
        aliases = instance.metadata.get("aliases") or []
        if not aliases and instance.gold_answer:
            aliases = [instance.gold_answer]
        for alias in aliases:
            alias = str(alias)
            if alias in response or alias.lower() in response or alias.capitalize() in response:
                return 1.0
        return 0.0


_FIRST_LINE = PopQAContainsScorer(name="popqa_contains_first_line", first_line=True)
_CONTAINMENT = PopQAContainsScorer()

# First-line containment is the primary metric: it is robust to ramble-rate
# differences between models, where whole-output containment awards verbose
# models free credit. Whole-output containment is what the reference
# implementation computes, so it stays reported for cross-harness and
# historical comparison.
_FIRST_LINE_ACCURACY = AccuracyMetric(name="exact_match_first_line", scorer=_FIRST_LINE)
_CONTAINMENT_ACCURACY = AccuracyMetric(name="exact_match_containment", scorer=_CONTAINMENT)


@register("popqa")
class PopQA(Task):
    data_source = DataSource(path="akariasai/PopQA", split="test")
    split = Split.TEST
    metrics = (_FIRST_LINE_ACCURACY, _CONTAINMENT_ACCURACY)
    primary_metric = _FIRST_LINE_ACCURACY
    # A completed trace fits the 15-token budget (``<think>USA</think>\nFrance``
    # is 8 tokens), and scoring one credits the reasoning rather than the answer,
    # so the strip is enabled here as well as on the reasoning variant.
    strip_thinking = True
    num_fewshot = 15
    sampling_params = SamplingParams(
        max_tokens=15,
        temperature=0.0,
        do_sample=False,
        stop_sequences=("\n\n",),
    )

    @property
    def instances(self) -> Iterator[Instance]:
        yield from self._load_instances_cached()

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        question = str(doc.get("question", ""))
        if not question:
            return None
        aliases = json.loads(doc["possible_answers"])

        return Instance(
            question=_format_query(question),
            gold_answer=str(doc["obj"]),
            metadata={
                "id": doc.get("id", index),
                "template_id": doc.get("prop_id"),
                "subject_popularity": doc.get("s_pop"),
                "aliases": aliases,
            },
        )

    def _build_fewshot(self) -> list[Instance]:
        num_fewshot = self.config.num_fewshot or 0
        if num_fewshot <= 0:
            return []
        instances = [
            Instance(
                question=_format_query(str(doc["question"])),
                gold_answer=str(doc["answer"]),
                metadata={"id": f"popqa_fixed_{index}"},
            )
            for index, doc in enumerate(POPQA_FIXED_FEWSHOT)
        ]
        return instances[:num_fewshot]

    def format_request(self, instance: Instance) -> LMRequest:
        parts = [f"{ex.question} {ex.gold_answer}" for ex in self.get_fewshot()]
        parts.append(instance.question)
        prompt = "\n\n".join(parts)
        # Few-shot examples stay in a single user message under chat format,
        # matching the reference implementation.
        if self.request_type == RequestType.CHAT:
            return LMRequest(
                request_type=RequestType.CHAT,
                messages=({"role": "user", "content": prompt},),
            )
        return LMRequest(request_type=RequestType.COMPLETION, prompt=prompt)


# Chat variant for instruct and reasoning models. Drops stop sequences and
# lifts the token cap (None = generate to the model's context limit) so
# chain-of-thought reasoning is not truncated on any context size.
register_variant(
    "popqa",
    "chat",
    formatter=ChatFormatter(),
    sampling_params=SamplingParams(
        max_tokens=None,
        temperature=0.6,
        top_p=0.95,
    ),
)
