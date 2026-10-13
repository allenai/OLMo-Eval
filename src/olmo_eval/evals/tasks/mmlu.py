from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from olmo_eval.common.formatters import MultipleChoiceLogprobFormatter
from olmo_eval.common.metrics import (
    BPBMetricInstanceAvg,
    LogprobMCAccuracyMetric,
    LogprobPerCharMCAccuracyMetric,
)
from olmo_eval.common.types import (
    Instance,
    LMOutput,
    LMRequest,
    RequestType,
    SamplingParams,
    Split,
)
from olmo_eval.data import DataSource
from olmo_eval.evals.tasks.common import Task, TaskConfig, register, register_variant

_STEM = (
    "abstract_algebra",
    "astronomy",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "electrical_engineering",
    "elementary_mathematics",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_mathematics",
    "high_school_physics",
    "high_school_statistics",
    "machine_learning",
)

_HUMANITIES = (
    "formal_logic",
    "high_school_european_history",
    "high_school_us_history",
    "high_school_world_history",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "moral_disputes",
    "moral_scenarios",
    "philosophy",
    "prehistory",
    "professional_law",
    "world_religions",
)

_SOCIAL_SCIENCES = (
    "econometrics",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_microeconomics",
    "high_school_psychology",
    "human_sexuality",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
)

_OTHER = (
    "anatomy",
    "business_ethics",
    "clinical_knowledge",
    "college_medicine",
    "global_facts",
    "human_aging",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "nutrition",
    "professional_accounting",
    "professional_medicine",
    "virology",
)

MMLU_SUBJECTS = _STEM + _HUMANITIES + _SOCIAL_SCIENCES + _OTHER

DEFAULT_MMLU_PATH = "cais/mmlu"


def _make_mcq_prompt(question: str, choices: list[str], label_prefix: str = " ") -> str:
    choice_labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    label_format = label_prefix + "A."
    choices_text = "\n".join(
        f"{label_format.replace('A', label)} {text}"
        for label, text in zip(choice_labels, choices, strict=False)
    )
    return f"Question: {question}\n{choices_text}\nAnswer:"


def _answer_to_index_and_letter(answer: int | str) -> tuple[int, str]:
    """Convert dataset answer (int 0-3 or str A-D) to (gold_idx, letter)."""
    if isinstance(answer, int):
        if not 0 <= answer <= 4:
            raise ValueError(f"MMLU answer index must be 0-4, got {answer}")
        return answer, chr(ord("A") + answer)
    s = str(answer).strip().upper()
    if len(s) != 1 or s not in "ABCDE":
        raise ValueError(f"MMLU answer letter must be A-E, got {answer!r}")
    return ord(s) - ord("A"), s


def _format_subject(subject: str) -> str:
    return " ".join(subject.split("_"))


def _format_rc(question: str, answer: str | None = None) -> str:
    """Format a cloze-style prompt for ranked classification."""
    prompt = f"Question: {question}\nAnswer:"
    if answer:
        prompt += f" {answer}"
    return prompt


class MMLUMCTask(Task):
    default_source: str = DEFAULT_MMLU_PATH
    fewshot_split: str = "dev"
    fewshot_sample: bool = False  # Fixed order (first k) as in reference

    def __init__(self, config: TaskConfig) -> None:
        super().__init__(config)

    @property
    def instances(self) -> Iterator[Instance]:
        """Yield instances from the test split."""
        yield from self._load_instances_cached(split="test")

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        """Convert a cais/mmlu document to an Instance."""
        question = doc.get("question", "")
        choices = list(doc.get("choices", []))
        if not choices or len(choices) > 5:
            return None
        answer = doc.get("answer", 0)
        try:
            gold_idx, letter = _answer_to_index_and_letter(answer)
        except ValueError:
            return None
        if gold_idx >= len(choices):
            return None
        choice_labels = [chr(ord("A") + i) for i in range(len(choices))]
        query = _make_mcq_prompt(question, choices, label_prefix=" ")
        return Instance(
            question=query,
            gold_answer=letter,
            choices=tuple(choice_labels),
            metadata={
                "id": index,
                "gold_idx": gold_idx,
            },
        )

    def format_request(self, instance: Instance) -> LMRequest:
        """Format using the task's formatter (with fewshot if configured)."""
        formatter = self.config.formatter
        if formatter is None:
            raise ValueError("MMLU MC task requires a formatter (e.g. MultipleChoiceFormatter)")
        return formatter.format(instance, self.get_fewshot())

    def extract_answer(self, output: LMOutput) -> Any:
        """Not used for logprob-based MC; scoring uses MultipleChoiceLogprobScorer."""
        return None

    def _build_fewshot(self) -> list[Instance]:
        """Few-shot from dev split in fixed order (first k), matching reference."""
        all_fewshot = self._build_fewshot_from_source(
            split=self.fewshot_split,
            sample=self.fewshot_sample,
            fallback_splits=[],
        )
        return all_fewshot[: self.config.num_fewshot]


class MMLURCTask(Task):
    """MMLU ranked classification (cloze) variant.

    Uses cloze-style prompts ("Question: ...\\nAnswer:") with raw text choices
    as continuations, scored by character-normalized logprob (acc_per_char).
    """

    default_source: str = DEFAULT_MMLU_PATH
    fewshot_split: str = "dev"
    fewshot_sample: bool = False
    subject: str = ""

    def __init__(self, config: TaskConfig) -> None:
        super().__init__(config)

    @property
    def instances(self) -> Iterator[Instance]:
        yield from self._load_instances_cached(split="test")

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        question = doc.get("question", "")
        choices = list(doc.get("choices", []))
        if not choices or len(choices) > 5:
            return None
        answer = doc.get("answer", 0)
        try:
            gold_idx, _letter = _answer_to_index_and_letter(answer)
        except ValueError:
            return None
        if gold_idx >= len(choices):
            return None
        return Instance(
            question=question,
            gold_answer=choices[gold_idx],
            choices=tuple(choices),
            metadata={
                "id": index,
                "gold_idx": gold_idx,
            },
        )

    def format_request(self, instance: Instance) -> LMRequest:
        fewshot = self.get_fewshot()
        subject_text = _format_subject(self.subject) if self.subject else ""
        description = (
            f"The following are multiple choice questions (with answers) about {subject_text}.\n\n"
            if subject_text
            else ""
        )

        parts: list[str] = []
        for ex in fewshot:
            answer = ex.gold_answer or ""
            parts.append(_format_rc(ex.question, answer))

        parts.append(_format_rc(instance.question))

        prompt = "\n\n".join(parts)
        if description:
            prompt = description + prompt

        continuations = tuple(f" {c}" for c in (instance.choices or ()))

        return LMRequest(
            request_type=RequestType.LOGLIKELIHOOD,
            prompt=prompt,
            continuations=continuations,
        )

    def extract_answer(self, output: LMOutput) -> Any:
        return None

    def _build_fewshot(self) -> list[Instance]:
        all_fewshot = self._build_fewshot_from_source(
            split=self.fewshot_split,
            sample=self.fewshot_sample,
            fallback_splits=[],
        )
        return all_fewshot[: self.config.num_fewshot]


def _make_formatter(subject: str) -> MultipleChoiceLogprobFormatter:
    subject_text = _format_subject(subject)
    description = (
        f"The following are multiple choice questions (with answers) about {subject_text}.\n\n"
    )
    return MultipleChoiceLogprobFormatter(
        template="{question}",
        label_prefix=" ",
        answer_suffix="",  # "Answer:" is already in the question text from _make_mcq_prompt
        fewshot_separator="\n\n",
        description=description,
    )


# Register one task per subject (mmlu_abstract_algebra, mmlu_anatomy, ...)
for _subject in MMLU_SUBJECTS:
    _formatter = _make_formatter(_subject)

    # MC variant (base task)
    _cls = type(
        f"MMLU_{_subject}",
        (MMLUMCTask,),
        {
            "data_source": DataSource(path=DEFAULT_MMLU_PATH, subset=_subject, split="test"),
            "formatter": _formatter,
            "metrics": (LogprobMCAccuracyMetric(),),
            "primary_metric": LogprobMCAccuracyMetric(),
            "num_fewshot": 5,
            "split": Split.TEST,
            "sampling_params": SamplingParams(max_tokens=1, temperature=0.0),
            "__module__": __name__,
            "__qualname__": f"MMLU_{_subject}",
        },
    )
    register(f"mmlu_{_subject}")(_cls)
    register_variant(f"mmlu_{_subject}", "mc")
    register_variant(f"mmlu_{_subject}", "olmo3base")
    globals()[f"MMLU_{_subject}"] = _cls

    # RC variant (cloze format with acc_per_char)
    _rc_cls = type(
        f"MMLU_RC_{_subject}",
        (MMLURCTask,),
        {
            "subject": _subject,
            "data_source": DataSource(path=DEFAULT_MMLU_PATH, subset=_subject, split="test"),
            "metrics": (LogprobPerCharMCAccuracyMetric(),),
            "primary_metric": LogprobPerCharMCAccuracyMetric(),
            "num_fewshot": 5,
            "split": Split.TEST,
            "sampling_params": SamplingParams(max_tokens=1, temperature=0.0),
            "__module__": __name__,
            "__qualname__": f"MMLU_RC_{_subject}",
        },
    )
    register(f"mmlu_{_subject}:rc")(_rc_cls)
    register_variant(f"mmlu_{_subject}:rc", "olmo3base")
    register_variant(
        f"mmlu_{_subject}:rc",
        "bpb",
        metrics=(BPBMetricInstanceAvg(),),
        primary_metric=BPBMetricInstanceAvg(),
    )
    globals()[f"MMLU_RC_{_subject}"] = _rc_cls
