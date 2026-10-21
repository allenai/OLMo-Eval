from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from olmo_eval.common.formatters import ChatFormatter, CompletionFormatter, PPLFormatter
from olmo_eval.common.metrics import AccuracyMetric, BPBMetricInstanceAvg, PassAtKMetric
from olmo_eval.common.scorers import MinervaMathScorer
from olmo_eval.common.types import Instance, LMOutput, LMRequest, RequestType, SamplingParams
from olmo_eval.data import DataLoader, DataSource
from olmo_eval.evals.extract import MathExtractor
from olmo_eval.evals.tasks.common import Task, register, register_variant
from olmo_eval.evals.tasks.constants.minerva_math import MINERVA_MATH_FIXED_FEWSHOT


@dataclass(slots=True)
class MinervaCompletionFormatter(CompletionFormatter):
    """CompletionFormatter that omits answer_prefix from the final (test) instance."""

    def format(
        self,
        instance: Instance,
        fewshot: list[Instance] | None = None,
    ) -> LMRequest:
        parts: list[str] = []
        for ex in fewshot or []:
            example = self.template.format(question=ex.question)
            if self.fewshot_answer_key and self.fewshot_answer_key in ex.metadata:
                answer = ex.metadata[self.fewshot_answer_key]
            else:
                answer = ex.gold_answer
            if answer:
                example += self.answer_prefix + str(answer)
            parts.append(example)
        parts.append(self.template.format(question=instance.question))
        prompt = self.fewshot_separator.join(parts)
        return LMRequest(request_type=self.request_type, prompt=prompt)


MATH_SUBSETS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]


class MinervaMathTask(Task):
    fewshot_split: str = "train"
    formatter = CompletionFormatter(
        template="Problem:\n{question}\n\nSolution:",
        fewshot_answer_key="solution_text",
    )
    metrics = (AccuracyMetric(scorer=MinervaMathScorer),)
    num_fewshot = 4
    sampling_params = SamplingParams(
        max_tokens=1024, temperature=0, stop_sequences=("Problem:", "\n\n")
    )
    dependencies = ["antlr4-python3-runtime>=4.11,<4.12"]

    def _build_fewshot(self) -> list[Instance]:
        """Use fixed 4 examples when fewshot_source is 'minerva_math_fixed'."""
        if getattr(self.config, "fewshot_source", None) == "minerva_math_fixed":
            return self._build_fixed_fewshot()
        return super()._build_fewshot()

    def _build_fixed_fewshot(self) -> list[Instance]:
        """Build 4 fixed few-shot instances from MINERVA_MATH_FIXED_FEWSHOT."""
        instances = []
        for doc in MINERVA_MATH_FIXED_FEWSHOT:
            solution_text = doc["solution"]
            extracted = MathExtractor.extract_answer(solution_text)
            primary = extracted[0] if extracted else None
            instances.append(
                Instance(
                    question=doc["problem"],
                    gold_answer=primary,
                    metadata={
                        "solution_text": solution_text,
                        "all_gold_answers": extracted if extracted else [],
                    },
                )
            )
        return instances

    @property
    def instances(self) -> Iterator[Instance]:
        if self._instances_cache is None:
            self._instances_cache = []
            loader = DataLoader()
            source = self._get_source_for_split("test")
            for index, doc in enumerate(loader.load(source)):
                instance = self.process_doc(doc, index)
                if instance is not None:
                    self._instances_cache.append(instance)
        yield from self._instances_cache

    def _get_source_for_split(self, split: str) -> DataSource:
        return self.config.get_data_source(split=split)

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        solution_text = doc["solution"]
        extracted_answers = MathExtractor.extract_answer(solution_text)

        primary_answer = extracted_answers[0] if extracted_answers else None
        all_gold_answers = extracted_answers if extracted_answers else []

        return Instance(
            question=doc["problem"],
            gold_answer=primary_answer,
            metadata={
                "id": doc.get("index", index),
                "level": doc.get("level"),
                "type": doc.get("type"),
                "solution_text": solution_text,
                "all_gold_answers": all_gold_answers,
            },
        )

    def format_request(self, instance: Instance) -> LMRequest:
        assert self.config.formatter is not None
        return self.config.formatter.format(instance, self.get_fewshot())

    def extract_answer(self, output: LMOutput) -> str | None:
        answers = MathExtractor.extract_answer(output.text)
        # Store all extracted answers in metadata for flexible scorers
        output.metadata["all_extracted_answers"] = answers if answers else []
        return answers[0] if answers else None


class Math500Task(MinervaMathTask):
    def _get_source_for_split(self, split: str) -> DataSource:
        # MATH-500 only has test split; use full MATH for train/dev
        if split != "test":
            return DataSource(
                path="EleutherAI/hendrycks_math",
                subset="algebra",  # Use algebra subset for few-shot
                split=split,
            )
        return self.config.get_data_source(split=split)

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        solution_text = doc["solution"]
        extracted_answers = MathExtractor.extract_answer(solution_text)

        gold_answer = extracted_answers[0] if extracted_answers else None
        all_gold_answers = extracted_answers if extracted_answers else []

        return Instance(
            question=doc["problem"],
            gold_answer=gold_answer,
            metadata={
                "id": doc.get("index", index),
                "level": doc.get("level"),
                "type": doc.get("type", doc.get("subject")),
                "solution_text": doc.get("solution"),
                "all_gold_answers": all_gold_answers,
            },
        )


for subset in MATH_SUBSETS:
    task_name = f"minerva_math_{subset}"
    class_name = f"MinervaMath_{subset.title().replace('_', '')}"

    task_cls = type(
        class_name,
        (MinervaMathTask,),
        {
            "__module__": __name__,
            "__qualname__": class_name,
            "data_source": DataSource(
                path="EleutherAI/hendrycks_math",
                subset=subset,
            ),
        },
    )
    globals()[class_name] = task_cls
    register(task_name)(task_cls)


@register("math500")
class Math500(Math500Task):
    data_source = DataSource(path="HuggingFaceH4/MATH-500")


for _subset in MATH_SUBSETS:
    _task_name = f"minerva_math_{_subset}"

    register_variant(
        _task_name,
        "olmo3",
        fewshot_source="minerva_math_fixed",
        metrics=(
            AccuracyMetric(scorer=MinervaMathScorer),
            PassAtKMetric(k=1, scorer=MinervaMathScorer),
            PassAtKMetric(k=2, scorer=MinervaMathScorer),
            PassAtKMetric(k=4, scorer=MinervaMathScorer),
        ),
        primary_metric=PassAtKMetric(k=1, scorer=MinervaMathScorer),
        sampling_params=SamplingParams(
            max_tokens=1024,
            temperature=0.6,
            top_p=0.6,
            stop_sequences=("Problem:", "\n\n"),
            num_samples=4,
        ),
    )

    register_variant(
        _task_name,
        "olmes_n4_v2",
        fewshot_source="minerva_math_fixed",
        metrics=(
            AccuracyMetric(scorer=MinervaMathScorer),
            PassAtKMetric(k=1, scorer=MinervaMathScorer),
            PassAtKMetric(k=2, scorer=MinervaMathScorer),
            PassAtKMetric(k=4, scorer=MinervaMathScorer),
        ),
        primary_metric=PassAtKMetric(k=1, scorer=MinervaMathScorer),
        sampling_params=SamplingParams(
            max_tokens=1024,
            temperature=0.6,
            top_p=0.6,
            stop_sequences=("Problem:", "\n\n"),
            num_samples=4,
        ),
    )

    register_variant(
        _task_name,
        "olmo3base",
        fewshot_source="minerva_math_fixed",
        formatter=MinervaCompletionFormatter(
            template="Problem:\n{question}\n\nSolution:",
            answer_prefix=" ",
            fewshot_answer_key="solution_text",
        ),
        metrics=(
            AccuracyMetric(scorer=MinervaMathScorer),
            PassAtKMetric(k=1, scorer=MinervaMathScorer),
            PassAtKMetric(k=2, scorer=MinervaMathScorer),
            PassAtKMetric(k=4, scorer=MinervaMathScorer),
        ),
        primary_metric=PassAtKMetric(k=1, scorer=MinervaMathScorer),
        sampling_params=SamplingParams(
            max_tokens=1024,
            temperature=0.6,
            top_p=0.6,
            stop_sequences=("Problem:", "\n\n"),
            num_samples=4,
        ),
    )

    register_variant(
        _task_name,
        "olmo3base_gen",
        fewshot_source="minerva_math_fixed",
        formatter=MinervaCompletionFormatter(
            template="Problem:\n{question}\n\nSolution:",
            answer_prefix=" ",
            fewshot_answer_key="solution_text",
        ),
        metrics=(
            AccuracyMetric(scorer=MinervaMathScorer),
            PassAtKMetric(k=1, scorer=MinervaMathScorer),
            PassAtKMetric(k=2, scorer=MinervaMathScorer),
            PassAtKMetric(k=4, scorer=MinervaMathScorer),
        ),
        primary_metric=PassAtKMetric(k=1, scorer=MinervaMathScorer),
        sampling_params=SamplingParams(
            max_tokens=1024,
            temperature=0.6,
            top_p=0.6,
            stop_sequences=("Problem:", "\n\n"),
            num_samples=4,
        ),
    )

    register_variant(
        _task_name,
        "bpb",
        formatter=PPLFormatter(),
        metrics=(BPBMetricInstanceAvg(),),
        primary_metric=BPBMetricInstanceAvg(),
    )

    register_variant(
        _task_name,
        "olmes",
        fewshot_source="minerva_math_fixed",
    )

    register_variant(
        _task_name,
        "olmes",
        fewshot_source="minerva_math_fixed",
    )

register_variant(
    "math500",
    "bpb",
    formatter=PPLFormatter(),
    metrics=(BPBMetricInstanceAvg(),),
    primary_metric=BPBMetricInstanceAvg(),
)


class MinervaMathBPBTask(MinervaMathTask):
    formatter = PPLFormatter()
    metrics = (BPBMetricInstanceAvg(),)
    primary_metric = BPBMetricInstanceAvg()

    def format_request(self, instance: Instance) -> LMRequest:
        fewshot = self.get_fewshot()
        parts: list[str] = []
        for ex in fewshot:
            text = f"Problem:\n{ex.question}\n\nSolution:"
            solution = ex.metadata.get("solution_text", ex.gold_answer or "")
            if solution:
                text += " " + str(solution)
            parts.append(text)
        parts.append(f"Problem:\n{instance.question}\n\nSolution:")
        prompt = "\n\n".join(parts)

        gold_text = instance.metadata.get("solution_text") or instance.gold_answer
        if gold_text is None:
            raise ValueError("BPB task requires a gold answer")
        if not gold_text.startswith(("\n", " ")):
            gold_text = " " + gold_text

        return LMRequest(
            request_type=RequestType.LOGLIKELIHOOD,
            prompt=prompt,
            continuations=(gold_text,),
            max_length=self.config.max_length,
        )


for _bpb_subset in MATH_SUBSETS:
    _bpb_task_name = f"minerva_math_{_bpb_subset}:bpb"
    _bpb_class_name = f"MinervaMathBPB_{_bpb_subset.title().replace('_', '')}"

    _bpb_cls = type(
        _bpb_class_name,
        (MinervaMathBPBTask,),
        {
            "__module__": __name__,
            "__qualname__": _bpb_class_name,
            "data_source": DataSource(
                path="EleutherAI/hendrycks_math",
                subset=_bpb_subset,
            ),
        },
    )
    globals()[_bpb_class_name] = _bpb_cls
    register(_bpb_task_name)(_bpb_cls)

    register_variant(_bpb_task_name, "olmo3base", fewshot_source="minerva_math_fixed")
    register_variant(_bpb_task_name, "olmes", fewshot_source="minerva_math_fixed")


# Olmo 3 paper protocol (oe-eval ``minerva_math_<subset>::olmo3:adapt``): zero-shot chat with
# the boxed-answer instruction, T=0.6 / top-p 0.95, no stop sequences, thinking trace stripped
# before scoring. max_tokens is 32768 rather than oe-eval's 131072 so the request fits a
# 65536-token context; override with ``-o max_tokens=...`` for bigger models.
OLMO3_ADAPT_BOXED_SUFFIX = "\n\nPresent the answer in LaTex format: \\boxed{{Your answer}}"
_OLMO3_ADAPT_FORMATTER = ChatFormatter(user_template="{question}" + OLMO3_ADAPT_BOXED_SUFFIX)
_OLMO3_ADAPT_SAMPLING = SamplingParams(max_tokens=32768, temperature=0.6, top_p=0.95)

for _subset in MATH_SUBSETS:
    register_variant(
        f"minerva_math_{_subset}",
        "olmo3adapt",
        formatter=_OLMO3_ADAPT_FORMATTER,
        num_fewshot=0,
        metrics=(AccuracyMetric(scorer=MinervaMathScorer),),
        primary_metric=AccuracyMetric(scorer=MinervaMathScorer),
        sampling_params=_OLMO3_ADAPT_SAMPLING,
        strip_thinking=True,
    )

# H101 (olmo-post-training-ledger): the olmo3adapt protocol with the boxed-answer line dropped. Identical to
# ``olmo3adapt`` except that the user message is the problem text alone.
_OLMO3_ADAPT_NOBOXED_FORMATTER = ChatFormatter(user_template="{question}")

for _subset in MATH_SUBSETS:
    register_variant(
        f"minerva_math_{_subset}",
        "olmo3adapt_noboxed",
        formatter=_OLMO3_ADAPT_NOBOXED_FORMATTER,
        num_fewshot=0,
        metrics=(AccuracyMetric(scorer=MinervaMathScorer),),
        primary_metric=AccuracyMetric(scorer=MinervaMathScorer),
        sampling_params=_OLMO3_ADAPT_SAMPLING,
        strip_thinking=True,
    )
