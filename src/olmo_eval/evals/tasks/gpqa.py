"""GPQA (Graduate-Level Google-Proof Q&A) evaluation tasks.

GPQA (Idavidrein/gpqa) is a multiple-choice benchmark with 4-way questions
across biology, physics, and chemistry. Three quality-tier subsets are available
(diamond/main/extended) and each question has a fine-grained ``Subdomain`` field.

Tasks:
    gpqa_{subset}            Full subset (diamond/main/extended)
    gpqa_{subset}_{subject}  Filtered by broad subject (biology/chemistry/physics)

Variants: ``:mc`` (logprob-based), ``:bpb`` (bits-per-byte), and ``:cot``
(0-shot chain-of-thought, post-training regime) on the full subsets.
"""

from __future__ import annotations

import logging
import random
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

from olmo_eval.common.formatters import MCQAChatFormatter, MultipleChoiceFormatter, PPLFormatter
from olmo_eval.common.metrics import AccuracyMetric, BPBMetricInstanceAvg
from olmo_eval.common.scorers import MultipleChoiceScorer
from olmo_eval.common.scorers.base import Scorer
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
from olmo_eval.evals.extract import (
    ANSWER_LINE_PATTERN,
    OLMO_3_ANSWER_REGEX_TEMPLATES,
    ExtractedAnswer,
    extract_answer_with_format,
)
from olmo_eval.evals.tasks.common import Task, register, register_variant

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

_PAREN_LETTER = re.compile(r"\(([A-Z])\)")
_LAST_CAPITAL = re.compile(r"\b([A-Z])\b")


# ---------------------------------------------------------------------------
# Subject grouping
# ---------------------------------------------------------------------------

_SUBJECT_GROUPS: dict[str, set[str]] = {
    "biology": {
        "Molecular Biology",
        "Genetics",
    },
    "chemistry": {
        "Organic Chemistry",
        "Chemistry",
        "Chemistry (general)",
        "Analytical Chemistry",
        "Inorganic Chemistry",
        "Physical Chemistry",
    },
    "physics": {
        "Quantum Mechanics",
        "Physics",
        "Physics (general)",
        "Astrophysics",
        "High-energy particle physics",
        "Condensed Matter Physics",
        "Electromagnetism and Photonics",
        "Relativistic Mechanics",
        "Statistical Mechanics",
        "Optics and Acoustics",
    },
}

# Reverse lookup: subdomain string -> group name
_SUBDOMAIN_TO_GROUP: dict[str, str] = {
    subdomain: group for group, subdomains in _SUBJECT_GROUPS.items() for subdomain in subdomains
}

_WARNED_SUBDOMAINS: set[str] = set()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an expert scientist. Answer the following multiple choice question by \
reasoning through the problem step by step, then selecting the best answer.

End your response with "ANSWER: X" where X is the letter of your chosen answer."""

_DEFAULT_ACCURACY = AccuracyMetric(scorer=MultipleChoiceScorer)
_DEFAULT_METRICS = (_DEFAULT_ACCURACY,)
_DEFAULT_SAMPLING = SamplingParams(temperature=0.0, max_tokens=1024)


# ---------------------------------------------------------------------------
# Base task
# ---------------------------------------------------------------------------


class GPQATask(Task):
    """Base class for GPQA tasks."""

    split = Split.TRAIN
    subject_group: str | None = None

    @property
    def instances(self) -> Iterator[Instance]:
        if self._instances_cache is None:
            self._instances_cache = []
            loader = DataLoader()
            source = self.config.get_data_source()
            for idx, doc in enumerate(loader.load(source)):
                instance = self.process_doc(doc, idx)
                if instance is not None:
                    self._instances_cache.append(instance)
        yield from self._instances_cache

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        question = doc.get("Question", "")
        correct = doc.get("Correct Answer", "")
        if not question or not correct:
            return None

        # Filter by subject group if set
        subdomain = doc.get("Subdomain", "")
        if self.subject_group is not None:
            group = _SUBDOMAIN_TO_GROUP.get(subdomain)
            if group is None and subdomain not in _WARNED_SUBDOMAINS:
                _WARNED_SUBDOMAINS.add(subdomain)
                log.warning("GPQA: unmapped subdomain %r", subdomain)
            if group != self.subject_group:
                return None

        # Build choices
        incorrect = [
            doc.get("Incorrect Answer 1", ""),
            doc.get("Incorrect Answer 2", ""),
            doc.get("Incorrect Answer 3", ""),
        ]
        choices = [_clean_text(correct)] + [_clean_text(a) for a in incorrect if a]

        # Deterministic per-question shuffle (seed=111 matches old oe-eval convention)
        rng = random.Random(f"{self.config.seed}:{index}")
        rng.shuffle(choices)

        gold_idx = choices.index(_clean_text(correct))
        gold_letter = chr(ord("A") + gold_idx)

        metadata: dict[str, Any] = {
            "index": index,
            "gold_idx": gold_idx,
            "gold_text": _clean_text(correct),
            "subdomain": subdomain,
        }
        explanation = doc.get("Explanation")
        if explanation:
            metadata["explanation"] = explanation

        return Instance(
            question=_clean_text(question),
            gold_answer=gold_letter,
            choices=tuple(choices),
            metadata=metadata,
        )

    @property
    def request_type(self) -> RequestType:
        if self.config.formatter is not None:
            return self.config.formatter.request_type
        return RequestType.CHAT

    def format_request(self, instance: Instance) -> LMRequest:
        if self.config.formatter is not None:
            return self.config.formatter.format(instance, self.get_fewshot())
        return LMRequest(
            request_type=RequestType.CHAT,
            messages=({"role": "user", "content": instance.question},),
        )

    def _extract_answers(self, responses: Sequence[Response]) -> None:
        for response in responses:
            if (
                response.request.request_type == RequestType.LOGLIKELIHOOD
                and len(response.outputs) > 1
            ):
                # MC logprob mode: pick the continuation with highest logprob
                best_idx = 0
                best_logprob = float("-inf")
                for i, output in enumerate(response.outputs):
                    logprob = (
                        output.metadata.get("total_logprob", float("-inf"))
                        if output.metadata
                        else float("-inf")
                    )
                    if logprob > best_logprob:
                        best_logprob = logprob
                        best_idx = i
                letter = chr(ord("A") + best_idx)
                for output in response.outputs:
                    output.extracted_answer = letter
            else:
                for output in response.outputs:
                    output.extracted_answer = self.extract_answer(output)

    def extract_answer(self, output: LMOutput) -> str | None:
        text = output.text

        # 1. Try "ANSWER: X"
        matches = list(ANSWER_LINE_PATTERN.finditer(text))
        if matches:
            return matches[-1].group(1).upper()

        # 2. Try parenthesized letter "(X)"
        paren = list(_PAREN_LETTER.finditer(text))
        if paren:
            return paren[-1].group(1).upper()

        # 3. Fallback: last standalone capital letter
        caps = list(_LAST_CAPITAL.finditer(text))
        if caps:
            last = caps[-1].group(1).upper()
            if last in {"A", "B", "C", "D"}:
                return last

        return None


# ---------------------------------------------------------------------------
# 0-shot CoT task (post-training regime)
# ---------------------------------------------------------------------------

_COT_FINAL_DESCRIPTION = (
    "\n\nPlease reason step by step, and put your final answer within \\boxed{}. "
    "Please only provide the letter of the answer in the box."
)

_COT_ANSWER_FORMAT_REGEX = r"Therefore, the answer is \(([A-D])\)"
_COT_ANSWER_REGEXES = (r"\(?([A-D])\)?",)

_COT_SHUFFLING_SEED = 111

# max_tokens=None generates to the model's context limit, matching the
# reference regime's effective behavior on any context size.
_COT_SAMPLING = SamplingParams(
    max_tokens=None,
    temperature=0.6,
    top_p=0.95,
)


def _extract_cot_answer(text: str) -> ExtractedAnswer:
    extracted = extract_answer_with_format(
        text,
        answer_format_regex=_COT_ANSWER_FORMAT_REGEX,
        answer_regexes=_COT_ANSWER_REGEXES,
        answer_regexes_templates=OLMO_3_ANSWER_REGEX_TEMPLATES,
    )
    return extracted._replace(answer=re.sub(r"\(|\)", "", extracted.answer))


@dataclass(frozen=True, slots=True)
class GPQACoTExactMatchScorer(Scorer):
    """Exact match on the answer letter extracted from a CoT response."""

    name: str = "exact_match"

    def score(self, instance: Instance, output: LMOutput) -> float:
        answer, format_correct = _extract_cot_answer(output.text or "")
        # Whether the letter was stated in the requested format, independent
        # of correctness: a low score means fallback extraction was needed.
        output.metadata["answer_format_correct"] = format_correct
        gold = str(instance.gold_answer or "")
        return 1.0 if answer.upper() == gold.upper() else 0.0


_COT_ACCURACY = AccuracyMetric(name="exact_match", scorer=GPQACoTExactMatchScorer)


def _cot_preprocess(text: str | None) -> str:
    """Text normalization used by the reference implementation's CoT regime."""
    if text is None:
        return " "
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    text = text.replace("  ", " ")
    return text


class GPQACoTTask(GPQATask):
    """0-shot chain-of-thought GPQA with inline answer choices.

    Prompt construction and choice shuffling match the reference
    implementation's post-training regime, so results are comparable
    with oe-eval's ``gpqa:0shot_cot`` numbers.
    """

    metrics = (_COT_ACCURACY,)
    primary_metric = _COT_ACCURACY
    sampling_params = _COT_SAMPLING
    strip_thinking = True

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        question = doc.get("Question", "")
        correct = doc.get("Correct Answer", "")
        if not question or not correct:
            return None

        choices = [
            _cot_preprocess(doc.get("Incorrect Answer 1")),
            _cot_preprocess(doc.get("Incorrect Answer 2")),
            _cot_preprocess(doc.get("Incorrect Answer 3")),
            _cot_preprocess(correct),
        ]
        random.Random(_COT_SHUFFLING_SEED + index).shuffle(choices)
        gold_idx = choices.index(_cot_preprocess(correct))
        gold_letter = chr(ord("A") + gold_idx)

        choice_labels = ["A", "B", "C", "D"]
        query = "Question: " + question + "\nChoices:\n"
        query += "".join(
            f" ({key}) {choice}\n" for key, choice in zip(choice_labels, choices, strict=True)
        )

        return Instance(
            question=query + _COT_FINAL_DESCRIPTION,
            gold_answer=gold_letter,
            choices=tuple(choices),
            metadata={
                "index": index,
                "gold_idx": gold_idx,
                "gold_text": _cot_preprocess(correct),
                "subdomain": doc.get("Subdomain", ""),
            },
        )

    def extract_answer(self, output: LMOutput) -> str | None:
        return _extract_cot_answer(output.text or "").answer or None


_TITLE_MARKER = re.compile(r"\s*\[title\]\s*", re.IGNORECASE)
_MULTI_SPACE = re.compile(r" {2,}")


def _clean_text(text: str) -> str:
    """Normalize GPQA text while preserving bracketed scientific content."""
    text = _TITLE_MARKER.sub(". ", text)
    text = _MULTI_SPACE.sub(" ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

_SUBSETS = ("gpqa_diamond", "gpqa_main", "gpqa_extended")
_SUBJECTS = ("biology", "chemistry", "physics")

for _subset in _SUBSETS:
    # Full-subset task (no filtering)
    _cls = type(
        _subset.title().replace("_", ""),
        (GPQATask,),
        {
            "__module__": __name__,
            "__qualname__": _subset.title().replace("_", ""),
            "data_source": DataSource(path="Idavidrein/gpqa", subset=_subset),
            "formatter": MCQAChatFormatter(system_prompt=_SYSTEM_PROMPT),
            "metrics": _DEFAULT_METRICS,
            "primary_metric": _DEFAULT_ACCURACY,
            "sampling_params": _DEFAULT_SAMPLING,
        },
    )
    globals()[_cls.__name__] = _cls
    register(_subset)(_cls)
    register_variant(_subset, "mc", formatter=MultipleChoiceFormatter(), metrics=_DEFAULT_METRICS)
    register_variant(_subset, "bpb", formatter=PPLFormatter(), metrics=(BPBMetricInstanceAvg(),))

    # 0-shot CoT (post-training regime): needs its own doc processing, so it is
    # a separate task class registered under a variant-style name.
    _cot_cls = type(
        _subset.title().replace("_", "") + "CoT",
        (GPQACoTTask,),
        {
            "__module__": __name__,
            "__qualname__": _subset.title().replace("_", "") + "CoT",
            "data_source": DataSource(path="Idavidrein/gpqa", subset=_subset),
        },
    )
    globals()[_cot_cls.__name__] = _cot_cls
    register(f"{_subset}:cot")(_cot_cls)

    # Subject-filtered tasks
    for _subject in _SUBJECTS:
        _task_name = f"{_subset}_{_subject}"
        _class_name = _task_name.title().replace("_", "")
        _cls_subj = type(
            _class_name,
            (GPQATask,),
            {
                "__module__": __name__,
                "__qualname__": _class_name,
                "data_source": DataSource(path="Idavidrein/gpqa", subset=_subset),
                "subject_group": _subject,
                "formatter": MCQAChatFormatter(system_prompt=_SYSTEM_PROMPT),
                "metrics": _DEFAULT_METRICS,
                "primary_metric": _DEFAULT_ACCURACY,
                "sampling_params": _DEFAULT_SAMPLING,
            },
        )
        globals()[_class_name] = _cls_subj
        register(_task_name)(_cls_subj)
        register_variant(
            _task_name, "mc", formatter=MultipleChoiceFormatter(), metrics=_DEFAULT_METRICS
        )
        register_variant(
            _task_name, "bpb", formatter=PPLFormatter(), metrics=(BPBMetricInstanceAvg(),)
        )
