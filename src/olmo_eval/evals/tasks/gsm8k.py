import re
from collections.abc import Iterator
from typing import Any

from olmo_eval.common.formatters import ChatFormatter
from olmo_eval.common.metrics import AccuracyMetric, PassAtKMetric
from olmo_eval.common.scorers import ExactMatchScorer
from olmo_eval.common.types import Instance, LMOutput, LMRequest, RequestType, SamplingParams
from olmo_eval.data import DataSource
from olmo_eval.evals.extract.math import last_boxed_only_string
from olmo_eval.evals.tasks.common import (
    OutputScoreAggregation,
    Task,
    register,
    register_variant,
)
from olmo_eval.evals.tasks.constants.gsm_symbolic import GSM8K_FIXED_FEWSHOT

GSM8K_PLATINUM_REPO = "madrylab/gsm8k-platinum"
GSM8K_PLATINUM_REVISION = "e762492455a1cf7967de89f05b6bef72fc713b66"

_NUMBER_RE = re.compile(r"[-+]?\d*\.\d+|[-+]?\d+")
_COMMA_IN_NUMBER_RE = re.compile(r"(\d),(\d)")


def _extract_last_number(text: str) -> str | None:
    output = _COMMA_IN_NUMBER_RE.sub(r"\1\2", text)
    numbers = _NUMBER_RE.findall(output)
    return numbers[-1] if numbers else None


def _strip_trailing_zeros(number: str) -> str:
    """Drop trailing decimal zeros so ``57.00`` matches a gold answer of ``57``."""
    if "." not in number:
        return number
    return number.rstrip("0").rstrip(".") or "0"


def _extract_boxed_or_last_number(text: str) -> str | None:
    """Return the number in the last ``\\boxed{}``, else the last number in the text."""
    boxed = last_boxed_only_string(text)
    number = None
    if boxed is not None and "{" in boxed:
        number = _extract_last_number(boxed[boxed.index("{") + 1 : -1])
    if number is None:
        number = _extract_last_number(text)
    return None if number is None else _strip_trailing_zeros(number)


def _clean_short_answer(text: str) -> str:
    output = _COMMA_IN_NUMBER_RE.sub(r"\1\2", text)
    numbers = _NUMBER_RE.findall(output)
    return numbers[-1] if numbers else text


@register("gsm8k")
class GSM8K(Task):
    data_source = DataSource(path="gsm8k", subset="main")
    metrics = (AccuracyMetric(scorer=ExactMatchScorer),)
    num_fewshot = 8
    sampling_params = SamplingParams(
        max_tokens=512,
        temperature=0,
        stop_sequences=("Question:", "\n\n"),
    )

    @property
    def instances(self) -> Iterator[Instance]:
        yield from self._load_instances_cached()

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        question = doc["question"]
        answer = doc["answer"]
        short_answer = answer.split("####")[-1].strip()
        cleaned = _clean_short_answer(short_answer)

        return Instance(
            question=question,
            gold_answer=cleaned,
            metadata={
                "id": index,
                "answer": answer,
                "short_answer": short_answer,
            },
        )

    def _build_fewshot(self) -> list[Instance]:
        instances = []
        for doc in GSM8K_FIXED_FEWSHOT:
            instances.append(
                Instance(
                    question=doc["question"],
                    gold_answer=doc["short_answer"],
                    metadata={
                        "answer": doc["answer"],
                        "short_answer": doc["short_answer"],
                    },
                )
            )
        return instances[: self.config.num_fewshot]

    def format_request(self, instance: Instance) -> LMRequest:
        fewshot = self.get_fewshot()
        if self.config.formatter is not None:
            return self.config.formatter.format(instance, fewshot)

        parts: list[str] = []
        for ex in fewshot:
            parts.append(f"Question: {ex.question}\nAnswer: {ex.metadata['answer']}")
        parts.append(f"Question: {instance.question}\nAnswer:")
        prompt = "\n\n".join(parts)

        return LMRequest(request_type=RequestType.COMPLETION, prompt=prompt)

    def extract_answer(self, output: LMOutput) -> str | None:
        if self.config.answer_extractor is not None:
            return self.config.answer_extractor(output.text)
        return _extract_last_number(output.text)


# GSM8K-Platinum is the GSM8K test set with ambiguous or flawed questions
# removed and mislabeled answers corrected (https://gradientscience.org/gsm8k-platinum/).
register_variant(
    "gsm8k",
    "platinum",
    data_source=DataSource(
        path=GSM8K_PLATINUM_REPO,
        subset="main",
        revision=GSM8K_PLATINUM_REVISION,
    ),
)

# Zero-shot chat variant for instruct and reasoning models. Drops stop
# sequences and lifts the token cap so chain-of-thought is not truncated.
register_variant(
    "gsm8k",
    "chat",
    formatter=ChatFormatter(
        user_template=(
            "{question}\n\nPlease reason step by step, "
            "and put your final answer within \\boxed{{}}."
        ),
    ),
    num_fewshot=0,
    sampling_params=SamplingParams(
        max_tokens=None,
        temperature=0.6,
        top_p=0.95,
    ),
    strip_thinking=True,
    answer_extractor=_extract_boxed_or_last_number,
)

register_variant(
    "gsm8k",
    "olmo3base",
    metrics=(
        AccuracyMetric(scorer=ExactMatchScorer),
        PassAtKMetric(k=1, scorer=ExactMatchScorer),
        PassAtKMetric(k=2, scorer=ExactMatchScorer),
        PassAtKMetric(k=4, scorer=ExactMatchScorer),
        PassAtKMetric(k=8, scorer=ExactMatchScorer),
    ),
    primary_metric=PassAtKMetric(k=1, scorer=ExactMatchScorer),
    sampling_params=SamplingParams(
        max_tokens=512,
        temperature=0.6,
        top_p=0.6,
        stop_sequences=("Question:", "</s>", "<|im_end|>", "\n\n"),
        num_samples=8,
    ),
    output_score_aggregation=OutputScoreAggregation.FIRST,
)
