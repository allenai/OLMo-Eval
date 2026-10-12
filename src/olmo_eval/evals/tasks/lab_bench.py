"""LAB-Bench evaluation tasks.

LAB-Bench (futurehouse/lab-bench) is a biology research benchmark with 8 subtasks.
This module implements the 6 text-only subtasks. The 2 image-based subtasks
(FigQA, TableQA) are deferred until the framework supports multimodal inputs.

Subtasks:
    lab_bench_litqa2            LitQA2 — literature-based QA (199 questions)
    lab_bench_dbqa              DbQA — biological database QA (520 questions)
    lab_bench_seqqa             SeqQA — DNA/protein sequence analysis (600 questions)
    lab_bench_protocolqa        ProtocolQA — lab protocol QA (108 questions)
    lab_bench_suppqa            SuppQA — supplementary material QA (82 questions)
    lab_bench_cloning_scenarios CloningScenarios — molecular cloning (33 questions)

Each task supports :mc (logprob-based) and :bpb (bits-per-byte) variants.
"""

from __future__ import annotations

import random
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

from olmo_eval.common.formatters import MCQAChatFormatter, MultipleChoiceFormatter, PPLFormatter
from olmo_eval.common.metrics import (
    AccuracyMetric,
    BPBMetricInstanceAvg,
    LogprobPerCharMCAccuracyMetric,
    Metric,
)
from olmo_eval.common.scorers import MultipleChoiceScorer, Scorer
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
from olmo_eval.evals.extract import ANSWER_LINE_PATTERN
from olmo_eval.evals.tasks.common import Task, register, register_variant

# Matches REFUSE_CHOICE from the official LAB-Bench evaluation code
_REFUSE_CHOICE = "Insufficient information to answer the question"


def _is_refusal(r: Response) -> bool:
    """Check if a response chose the refuse option."""
    refuse_letter = chr(ord("A") + r.instance.metadata["refuse_idx"])
    extracted = r.outputs[0].extracted_answer if r.outputs else None
    return extracted is not None and extracted.strip().upper() == refuse_letter


@dataclass(frozen=True, slots=True)
class PrecisionMetric(Metric):
    """Accuracy excluding refusals (correct / non-refused).

    Matches the "precision" metric from the LAB-Bench evaluation protocol.
    """

    name: str = "precision"
    scorer: type[Scorer] = MultipleChoiceScorer

    def compute(self, responses: Sequence[Response]) -> float:
        if not responses:
            return 0.0
        scorer_name = self.scorer().name
        committed = [r for r in responses if not _is_refusal(r)]
        if not committed:
            return 0.0
        return sum(r.scores.get(scorer_name, 0.0) for r in committed) / len(committed)

    def supports_scorer_fallback(self) -> bool:
        # Precision excludes refusals before averaging, so raw MC scorer values
        # are not interchangeable with the stored task metric.
        return False

    def display_format(self) -> str:
        return "percentage"


@dataclass(frozen=True, slots=True)
class CoverageMetric(Metric):
    """Fraction of responses where the model committed (did not refuse).

    Matches the "coverage" metric from the LAB-Bench evaluation protocol.
    """

    name: str = "coverage"
    scorer: type[Scorer] = MultipleChoiceScorer

    def compute(self, responses: Sequence[Response]) -> float:
        if not responses:
            return 0.0
        committed = sum(1 for r in responses if not _is_refusal(r))
        return committed / len(responses)

    def supports_scorer_fallback(self) -> bool:
        # Coverage is derived from refusal handling, not the MC scorer signal.
        return False

    def display_format(self) -> str:
        return "percentage"


def _format_lab_bench_rc(question: str, answer: str | None = None) -> str:
    prompt = f"Question: {question}\nAnswer:"
    if answer:
        prompt += f" {answer}"
    return prompt


class LabBenchTask(Task):
    split = Split.TRAIN
    fewshot_split = "train"

    def __init_subclass__(cls, subset: str | None = None, **kwargs):
        super().__init_subclass__(**kwargs)
        if subset is None:
            return
        snake = re.sub(r"([a-z])([A-Z][a-z])", r"\1_\2", subset).lower()
        name = f"lab_bench_{snake}"
        cls.data_source = DataSource(path="futurehouse/lab-bench", subset=subset)
        cls.formatter = MCQAChatFormatter(system_prompt=_SYSTEM_PROMPT)
        cls.metrics = _DEFAULT_METRICS
        cls.primary_metric = _DEFAULT_ACCURACY
        cls.sampling_params = _DEFAULT_SAMPLING
        register(name)(cls)
        register_variant(name, "mc", formatter=MultipleChoiceFormatter(), metrics=_DEFAULT_METRICS)
        register_variant(name, "bpb", formatter=PPLFormatter(), metrics=(BPBMetricInstanceAvg(),))
        register_variant(
            name,
            "olmo3base",
            formatter=None,
            num_fewshot=3,
            fewshot_seed=1234,
            metrics=(LogprobPerCharMCAccuracyMetric(),),
        )
        # Register name:bpb as a separate task (like drop:bpb) so that
        # name:bpb:olmo3base works reliably with stacked variants. This
        # matches the old oe-eval-internal lab_bench_*:bpb config: RC format,
        # 3-shot, seed 1234, BPB metric.
        import sys

        bpb_name = f"{name}:bpb"
        bpb_cls = type(
            f"{cls.__name__}BPB",
            (cls,),
            {
                "formatter": None,
                "metrics": (BPBMetricInstanceAvg(),),
                "primary_metric": BPBMetricInstanceAvg(),
                "num_fewshot": 3,
                "fewshot_seed": 1234,
                "__module__": cls.__module__,
                "__qualname__": f"{cls.__name__}BPB",
            },
        )
        setattr(sys.modules[cls.__module__], f"{cls.__name__}BPB", bpb_cls)
        register(bpb_name)(bpb_cls)
        register_variant(bpb_name, "olmo3base")

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
        """Convert a LAB-Bench document to an Instance with shuffled choices."""
        question = doc.get("question", "")
        ideal = doc.get("ideal", "")
        distractors = doc.get("distractors", [])

        if not question or not ideal:
            return None

        rc_mode = self.config.formatter is None

        if rc_mode:
            # RC mode (olmo3base): match old oe-eval-internal behavior exactly.
            # No refuse option, no shuffling, distractors + [ideal] order.
            choices = list(distractors) + [ideal]
            gold_idx = len(distractors)
            refuse_idx = -1
        else:
            # Build choices: ideal + distractors (deduplicate in case ideal appears in distractors)
            choices = [ideal] + [d for d in distractors if d != ideal]

            # Inject refuse option per the LAB-Bench evaluation protocol
            choices.append(_REFUSE_CHOICE)

            # Deterministic per-question shuffle
            rng = random.Random(f"{self.config.seed}:{index}")
            rng.shuffle(choices)

            gold_idx = choices.index(ideal)
            refuse_idx = choices.index(_REFUSE_CHOICE)

        gold_letter = chr(ord("A") + gold_idx)

        metadata: dict[str, Any] = {
            "id": doc.get("id", f"lab_bench_{index}"),
            "index": index,
            "gold_idx": gold_idx,
            "gold_text": ideal,
            "refuse_idx": refuse_idx,
        }

        return Instance(
            question=question,
            gold_answer=gold_letter,
            choices=tuple(choices),
            metadata=metadata,
        )

    @property
    def request_type(self) -> RequestType:
        if self.config.formatter is not None:
            return self.config.formatter.request_type
        return RequestType.LOGLIKELIHOOD

    def format_request(self, instance: Instance) -> LMRequest:
        if self.config.formatter is not None:
            return self.config.formatter.format(instance, self.get_fewshot())
        # RC format (cloze): "Question: {q}\nAnswer:" with choice text continuations
        fewshot = self.get_fewshot()
        parts: list[str] = []
        for ex in fewshot:
            answer = ex.metadata.get("gold_text", ex.gold_answer or "")
            parts.append(_format_lab_bench_rc(ex.question, answer))
        parts.append(_format_lab_bench_rc(instance.question))
        continuations = tuple(f" {c}" for c in (instance.choices or ()))
        prompt = "\n\n".join(parts)
        return LMRequest(
            request_type=RequestType.LOGLIKELIHOOD,
            prompt=prompt,
            continuations=continuations,
        )

    def _extract_answers(self, responses: Sequence[Response]) -> None:
        """Extract answers, with argmax for logprob-based MC scoring."""
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
                # Chat mode: parse "ANSWER: X" from generated text
                for output in response.outputs:
                    output.extracted_answer = self.extract_answer(output)

    def extract_answer(self, output: LMOutput) -> str | None:
        """Extract the last ``ANSWER: X`` letter from model output."""
        matches = list(ANSWER_LINE_PATTERN.finditer(output.text))
        return matches[-1].group(1).upper() if matches else None


_SYSTEM_PROMPT = """\
You are a scientific research assistant. Answer the following multiple choice \
question by reasoning through the options and selecting the best answer.

End your response with "ANSWER: X" where X is the letter of your chosen answer."""

_DEFAULT_ACCURACY = AccuracyMetric(scorer=MultipleChoiceScorer)
_DEFAULT_METRICS = (_DEFAULT_ACCURACY, PrecisionMetric(), CoverageMetric())
_DEFAULT_SAMPLING = SamplingParams(temperature=0.0, max_tokens=1024)


# =============================================================================
# Task Registrations
# =============================================================================


class LitQA2(LabBenchTask, subset="LitQA2"): ...


class DbQA(LabBenchTask, subset="DbQA"): ...


class SeqQA(LabBenchTask, subset="SeqQA"): ...


class SuppQA(LabBenchTask, subset="SuppQA"): ...


class CloningScenarios(LabBenchTask, subset="CloningScenarios"): ...


class ProtocolQA(LabBenchTask, subset="ProtocolQA"):
    """Prepends the protocol text to the question (chat/MC modes only)."""

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        # Only prepend protocol for chat/MC evaluation; RC mode (formatter=None)
        # matches the old oe-eval-internal behaviour which uses the bare question.
        if self.config.formatter is not None:
            protocol = doc.get("protocol", "")
            if protocol:
                question = doc.get("question", "")
                doc = {**doc, "question": f"Protocol:\n{protocol}\n\nQuestion: {question}"}
        return super().process_doc(doc, index)
