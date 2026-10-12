"""OMEGA-500: out-of-distribution math reasoning.

OMEGA (https://arxiv.org/abs/2506.18880) evaluates exploratory,
compositional, and transformative generalization in math. OMEGA-500 is a
500-problem batch generated from the exploratory templates. Answers are
extracted from the response via a regex cascade and compared to the ground
truth, with a strict metric that demotes answers not stated in the
requested format and a flex metric that accepts them.
"""

from __future__ import annotations

import re
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
    SamplingParams,
    Split,
)
from olmo_eval.data import DataSource
from olmo_eval.evals.extract import ExtractedAnswer, extract_answer_with_format
from olmo_eval.evals.tasks.common import Task, register

#: The AllenAI snapshot, which corrects some of the original questions and answers.
OMEGA_500_REVISION = "113a7eb896b8c1f7d781eb5f00713972074bae42"

_BOXED_SUFFIX = "\n\nPresent the answer in LaTex format: \\boxed{Your answer}"

_TEMPLATE_SUFFIX = (
    '\n\nShow your work and conclude with "Therefore, the final answer is '
    '\\boxed{answer}." where answer is just the final solution that solves the '
    'problem. E.g., if the answer is 6, conclude with "Therefore, the final '
    'answer is \\boxed{6}."'
)

# The suffix lives in the formatter (braces escaped for str.format) so
# register_variant can swap the prompt template; see the rlzero/midtrain
# regimes upstream, which use different templates and stop sequences.
_FORMATTER = ChatFormatter(
    user_template="{question}" + _TEMPLATE_SUFFIX.replace("{", "{{").replace("}", "}}")
)

_ANSWER_FORMAT_REGEX = r"Therefore, the final answer is \\boxed\{(.*)\}"
_PREFIX_REGEXES = (
    r"(?i)Therefore,? the final answer is",
    r"(?i)Therefore,? the answer is",
    r"(?i)the final answer is",
    r"(?i)the answer is",
    r"(?i)answer is",
    r"(?i)answer is:",
    r"(?i)answer:",
)
_ANSWER_REGEXES = (r"[\s\S]*?\\boxed\{(.*?)\}[\s\S]*?", r"(.*)\.?")
_FORMAT_CORRECT_CUTOFF = 0.4

# Mathy delimiters stripped before extraction. Ported verbatim from the
# reference implementation, whose substitution is unanchored and per-line:
# on each line the outermost left/right pair is dropped wherever it
# appears, not only when it surrounds the whole answer.
_DELIMITERS_TO_STRIP = (
    ("$", "$"),
    ("\\(", "\\)"),
    ("**", "**"),
    ("***", "***"),
    ("\\[", "\\]"),
    ("\\[\n", "\n\\]"),
)


def _extract(continuation: str) -> ExtractedAnswer:
    """Normalize a continuation and extract the final answer."""
    output = re.sub(r"(\d),(\d)", r"\1\2", continuation)
    res = re.sub(r"\.\s*$", "", output).strip()
    for left, right in _DELIMITERS_TO_STRIP:
        res = re.sub(f"{re.escape(left)}(.*){re.escape(right)}", "\\1", res).strip()
    # The leading wildcard search is quadratic when no boxed opener exists.
    # Keep its cascade slot (and format score) with an impossible pattern.
    answer_regexes = _ANSWER_REGEXES if "\\boxed{" in res else (r"(?!)", _ANSWER_REGEXES[1])
    return extract_answer_with_format(
        res,
        answer_format_regex=_ANSWER_FORMAT_REGEX,
        answer_regexes=answer_regexes,
        prefix_regexes=_PREFIX_REGEXES,
    )


@dataclass(frozen=True, slots=True)
class OmegaExactMatchScorer(Scorer):
    """Case-insensitive exact match on the extracted answer.

    The strict form treats an answer as missing when it was not stated in
    the requested format; the flex form accepts it regardless.
    """

    name: str = ""
    flex: bool = False

    def __post_init__(self) -> None:
        # The name keys scorer storage and deduplication, so it is derived
        # from ``flex`` unless set explicitly — a flex scorer left with the
        # strict default name would be silently dropped as a duplicate.
        if not self.name:
            object.__setattr__(self, "name", "exact_match_flex" if self.flex else "exact_match")

    def score(self, instance: Instance, output: LMOutput) -> float:
        answer, format_correct = _extract(output.text or "")
        # Whether the answer was stated in the requested format, independent
        # of correctness — distinguishes "wrong" from "badly formatted".
        output.metadata["answer_format_correct"] = format_correct
        if not self.flex and format_correct < _FORMAT_CORRECT_CUTOFF:
            answer = ""
        gold = str(instance.gold_answer or "")
        return 1.0 if answer.lower() == gold.lower() else 0.0


_STRICT = OmegaExactMatchScorer()
_FLEX = OmegaExactMatchScorer(flex=True)


@register("omega_500")
class Omega500(Task):
    data_source = DataSource(path="saumyamalik/omega-500")
    split = Split.TRAIN
    formatter = _FORMATTER
    metrics = (
        AccuracyMetric(name="exact_match", scorer=_STRICT),
        AccuracyMetric(name="exact_match_flex", scorer=_FLEX),
    )
    primary_metric = AccuracyMetric(name="exact_match_flex", scorer=_FLEX)
    strip_thinking = True
    # Defaults mirror oe-eval's ``omega_500:0-shot-chat_deepseek`` — the
    # OLMO_3 suite entry and the config the parity certification ran on.
    # (Plain ``0-shot-chat`` upstream uses temperature 0.7 with no top_p;
    # register a variant if that regime is ever needed.) max_tokens=None
    # generates to the model's context limit, matching the reference
    # regime's effective behavior on any context size.
    sampling_params = SamplingParams(
        max_tokens=None,
        temperature=0.6,
        top_p=0.95,
    )

    @property
    def instances(self) -> Iterator[Instance]:
        yield from self._load_instances_cached()

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        question = str(doc["messages"][0]["content"]).replace(_BOXED_SUFFIX, "")
        return Instance(
            question=question,
            gold_answer=str(doc["ground_truth"]),
            metadata={
                "id": doc.get("index", index),
                "family": doc.get("family", doc.get("setting_key")),
                "dataset": doc.get("dataset"),
            },
        )

    def format_request(self, instance: Instance) -> LMRequest:
        assert self.config.formatter is not None
        return self.config.formatter.format(instance, self.get_fewshot())

    def extract_answer(self, output: LMOutput) -> str:
        return _extract(output.text or "").answer


@register("omega_500:hillclimb")
class Omega500HillClimb(Omega500):
    """OMEGA-500 on the corrected AllenAI snapshot, scored strictly at a 32K budget.

    Instances keep the dataset's own IDs, so they stay stable if rows move.
    """

    data_source = DataSource(path="allenai/omega-500", revision=OMEGA_500_REVISION)
    primary_metric = AccuracyMetric(name="exact_match", scorer=_STRICT)
    sampling_params = SamplingParams(max_tokens=32768, temperature=0.6, top_p=0.95)

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        instance = super().process_doc(doc, index)
        assert instance is not None
        instance.metadata["id"] = doc["id"]
        return instance
