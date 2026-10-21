"""MMLU-Pro: multiple-choice questions with up to ten options, one task per category.

The dataset (``TIGER-Lab/MMLU-Pro``) ships as a single configuration with a
``category`` column, so each per-category task loads the split and keeps the
rows for its category. Three recipes are registered per category, mirroring
the reference harness:

- ``mmlu_pro_{category}``: 5-shot multiple choice scored by the log-probability
  of the answer letter (aliases ``:mc`` and ``:olmo3base``).
- ``mmlu_pro_{category}:rc``: 5-shot cloze scored by character-normalized
  log-probability of the answer text (``:bpb`` reports bits per byte).
- ``mmlu_pro_{category}:cot``: 0-shot chat with concise reasoning and a
  "Therefore, the answer is (X)" conclusion, scored by exact match on the
  extracted letter.

Few-shot exemplars are the dataset's ``validation`` rows for the category, in
dataset order. The CoT recipe samples and extracts answers the same way as the
MMLU and GPQA chain-of-thought tasks: the shared regex cascade, widened to the
ten answer letters this benchmark uses.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from olmo_eval.common.formatters import ChatFormatter, MultipleChoiceLogprobFormatter
from olmo_eval.common.metrics import (
    AccuracyMetric,
    BPBMetricInstanceAvg,
    LogprobMCAccuracyMetric,
    LogprobPerCharMCAccuracyMetric,
)
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
from olmo_eval.evals.extract import (
    OLMO_3_ANSWER_REGEX_TEMPLATES,
    ExtractedAnswer,
    extract_answer_with_format,
    extract_think_answer,
)
from olmo_eval.evals.tasks.common import Task, register, register_variant
from olmo_eval.evals.tasks.mmlu import _format_rc, _make_mcq_prompt

MMLU_PRO_PATH = "TIGER-Lab/MMLU-Pro"
MMLU_PRO_REVISION = "b189ec765aa7ed75c8acfea42df31fdae71f97be"

MMLU_PRO_CATEGORIES: tuple[str, ...] = (
    "math",
    "health",
    "physics",
    "business",
    "biology",
    "chemistry",
    "computer science",
    "economics",
    "engineering",
    "philosophy",
    "other",
    "history",
    "psychology",
    "law",
)

_CHOICE_LABELS = "ABCDEFGHIJKLMNO"


def category_slug(category: str) -> str:
    """Task-name slug for a dataset category."""
    return category.replace(" ", "_")


def _parse_doc(doc: dict[str, Any], category: str) -> tuple[str, list[str], int] | None:
    """Return (question, options, gold index) for a doc in ``category``, else None."""
    if doc.get("category") != category:
        return None
    question = str(doc.get("question") or "")
    options = [str(option) for option in (doc.get("options") or [])]
    gold_idx = doc.get("answer_index")
    if not question or not options or len(options) > len(_CHOICE_LABELS):
        return None
    if not isinstance(gold_idx, int) or not 0 <= gold_idx < len(options):
        return None
    return question, options, gold_idx


def _metadata(
    doc: dict[str, Any], index: int, category: str, gold_idx: int | None = None
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "id": doc.get("question_id", index),
        "index": index,
        "category": category,
    }
    if gold_idx is not None:
        metadata["gold_idx"] = gold_idx
    return metadata


class MMLUProTask(Task):
    """Shared loading and few-shot behavior for one MMLU-Pro category."""

    split = Split.TEST
    fewshot_split: str = "validation"
    fewshot_sample: bool = False
    category: str = ""

    @property
    def instances(self) -> Iterator[Instance]:
        yield from self._load_instances_cached()

    def extract_answer(self, output: LMOutput) -> Any:
        return None

    def _build_fewshot(self) -> list[Instance]:
        """First ``num_fewshot`` validation rows of this category, in dataset order."""
        k = self.config.num_fewshot
        if not k:
            return []
        all_fewshot = self._build_fewshot_from_source(
            split=self.fewshot_split,
            sample=self.fewshot_sample,
            fallback_splits=[],
        )
        return all_fewshot[:k]


class MMLUProMCTask(MMLUProTask):
    """Multiple choice with lettered options and letter continuations."""

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        parsed = _parse_doc(doc, self.category)
        if parsed is None:
            return None
        question, options, gold_idx = parsed
        return Instance(
            question=_make_mcq_prompt(question, options, label_prefix=" "),
            gold_answer=_CHOICE_LABELS[gold_idx],
            choices=tuple(_CHOICE_LABELS[: len(options)]),
            metadata=_metadata(doc, index, self.category, gold_idx),
        )

    def format_request(self, instance: Instance) -> LMRequest:
        formatter = self.config.formatter
        if formatter is None:
            raise ValueError("MMLU-Pro MC task requires a formatter")
        return formatter.format(instance, self.get_fewshot())


class MMLUProRCTask(MMLUProTask):
    """Cloze form: the question alone, with option texts as continuations."""

    @property
    def request_type(self) -> RequestType:
        return RequestType.LOGLIKELIHOOD

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        parsed = _parse_doc(doc, self.category)
        if parsed is None:
            return None
        question, options, gold_idx = parsed
        return Instance(
            question=question,
            gold_answer=options[gold_idx],
            choices=tuple(options),
            metadata=_metadata(doc, index, self.category, gold_idx),
        )

    def format_request(self, instance: Instance) -> LMRequest:
        parts = [_format_rc(ex.question, ex.gold_answer or "") for ex in self.get_fewshot()]
        parts.append(_format_rc(instance.question))
        return LMRequest(
            request_type=RequestType.LOGLIKELIHOOD,
            prompt="\n\n".join(parts),
            continuations=tuple(f" {c}" for c in (instance.choices or ())),
        )


_MC_FORMATTER = MultipleChoiceLogprobFormatter(
    template="{question}",
    label_prefix=" ",
    answer_suffix="",
    fewshot_separator="\n\n",
    description="",
)

_COT_DESCRIPTION = (
    "Answer the following multiple-choice question by giving the correct answer letter in "
    "parentheses. Provide CONCISE reasoning for the answer, and make sure to finish the "
    'response with "Therefore, the answer is (ANSWER_LETTER)" where (ANSWER_LETTER) is one of '
    "(A), (B), (C), (D), (E), etc.\n\n"
)

_COT_FINAL_DESCRIPTION = (
    "\nAnswer the above question and REMEMBER to finish your response with the exact phrase "
    '"Therefore, the answer is (ANSWER_LETTER)" where (ANSWER_LETTER) is one of (A), (B), (C), '
    "(D), (E), etc."
)

_COT_FORMATTER = ChatFormatter(
    user_template=_COT_DESCRIPTION + "{question}" + _COT_FINAL_DESCRIPTION,
)

# max_tokens=None generates to the model's context limit; temperature and
# top_p match the MMLU and GPQA chain-of-thought tasks, since greedy decoding
# degrades reasoning models.
_COT_SAMPLING = SamplingParams(
    max_tokens=None,
    temperature=0.6,
    top_p=0.95,
)

_COT_ANSWER_FORMAT_REGEX = r"Therefore, the answer is \(([A-J])\)"
_COT_ANSWER_REGEXES = (r"\(?([A-J])\)?",)


def _extract_cot_answer(text: str) -> ExtractedAnswer:
    """Answer letter and format score from a chain-of-thought response.

    Reasoning enclosed in ``<think>`` tags is dropped first. The requested
    phrasing scores 1.0; the shared regex templates score 1.0 or 0.5 by
    position in the cascade; a bare letter scores 0.2 or lower.
    """
    extracted = extract_answer_with_format(
        extract_think_answer(text or "") or "",
        answer_format_regex=_COT_ANSWER_FORMAT_REGEX,
        answer_regexes=_COT_ANSWER_REGEXES,
        answer_regexes_templates=OLMO_3_ANSWER_REGEX_TEMPLATES,
    )
    return extracted._replace(answer=re.sub(r"\(|\)", "", extracted.answer))


def _extract_cot_letter(text: str) -> str:
    return _extract_cot_answer(text).answer


@dataclass(frozen=True, slots=True)
class MMLUProCoTExactMatchScorer(Scorer):
    """Case-insensitive exact match on the extracted answer letter."""

    name: str = "exact_match"

    def score(self, instance: Instance, output: LMOutput) -> float:
        answer, format_correct = _extract_cot_answer(output.text or "")
        # Whether the letter was stated in the requested format, independent
        # of correctness: a low score means fallback extraction was needed.
        output.metadata["answer_format_correct"] = format_correct
        gold = str(instance.gold_answer or "")
        return 1.0 if answer.upper() == gold.upper() else 0.0


_COT_ACCURACY = AccuracyMetric(name="exact_match", scorer=MMLUProCoTExactMatchScorer)


class MMLUProCoTTask(MMLUProTask):
    """0-shot chain-of-thought in chat format for one category."""

    formatter = _COT_FORMATTER
    metrics = (_COT_ACCURACY,)
    primary_metric = _COT_ACCURACY
    sampling_params = _COT_SAMPLING
    num_fewshot = 0
    strip_thinking = True
    answer_extractor = _extract_cot_letter

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        parsed = _parse_doc(doc, self.category)
        if parsed is None:
            return None
        question, options, gold_idx = parsed
        query = f"Question: {question}\n" + "".join(
            f" ({label}) {option}\n" for label, option in zip(_CHOICE_LABELS, options, strict=False)
        )
        return Instance(
            question=query,
            gold_answer=_CHOICE_LABELS[gold_idx],
            choices=tuple(options),
            metadata=_metadata(doc, index, self.category),
        )

    def extract_answer(self, output: LMOutput) -> str | None:
        return _extract_cot_letter(output.text or "") or None

    def format_request(self, instance: Instance) -> LMRequest:
        formatter = self.config.formatter
        if formatter is None:
            raise ValueError("MMLU-Pro CoT task requires a formatter")
        return formatter.format(instance, None)


for _category in MMLU_PRO_CATEGORIES:
    _slug = category_slug(_category)
    _name = f"mmlu_pro_{_slug}"
    _source = DataSource(path=MMLU_PRO_PATH, revision=MMLU_PRO_REVISION)

    _mc_cls = type(
        f"MMLUPro_{_slug}",
        (MMLUProMCTask,),
        {
            "category": _category,
            "data_source": _source,
            "formatter": _MC_FORMATTER,
            "metrics": (LogprobMCAccuracyMetric(),),
            "primary_metric": LogprobMCAccuracyMetric(),
            "num_fewshot": 5,
            "sampling_params": SamplingParams(max_tokens=1, temperature=0.0),
            "__module__": __name__,
            "__qualname__": f"MMLUPro_{_slug}",
        },
    )
    register(_name)(_mc_cls)
    register_variant(_name, "mc")
    register_variant(_name, "olmo3base")

    _rc_cls = type(
        f"MMLUProRC_{_slug}",
        (MMLUProRCTask,),
        {
            "category": _category,
            "data_source": _source,
            "metrics": (LogprobPerCharMCAccuracyMetric(),),
            "primary_metric": LogprobPerCharMCAccuracyMetric(),
            "num_fewshot": 5,
            "sampling_params": SamplingParams(max_tokens=1, temperature=0.0),
            "__module__": __name__,
            "__qualname__": f"MMLUProRC_{_slug}",
        },
    )
    register(f"{_name}:rc")(_rc_cls)
    register_variant(f"{_name}:rc", "olmo3base")
    register_variant(
        f"{_name}:rc",
        "bpb",
        metrics=(BPBMetricInstanceAvg(),),
        primary_metric=BPBMetricInstanceAvg(),
    )

    _cot_cls = type(
        f"MMLUProCoT_{_slug}",
        (MMLUProCoTTask,),
        {
            "category": _category,
            "data_source": _source,
            "__module__": __name__,
            "__qualname__": f"MMLUProCoT_{_slug}",
        },
    )
    register(f"{_name}:cot")(_cot_cls)
