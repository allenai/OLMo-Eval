"""

Base Task structure for implementing safety evaluations

Inherited by: BBQ, Do_Anything_Now, HarmBench, StrongReject,
TrustLLM JailbreakTrigger, WildGuard, WildJailbreak, WMDP, XSTest

"""

import logging
from collections.abc import Iterator

from olmo_eval.common.formatters import ChatFormatter, CompletionFormatter
from olmo_eval.common.metrics import (
    AccuracyMetric,
    MacroSubsetAccuracyMetric,
    SafetyErrorMetric,
    SubsetAccuracyMetric,
)
from olmo_eval.common.scorers import SafetyScorer
from olmo_eval.common.types import Instance, LMRequest, RequestType, SamplingParams
from olmo_eval.data import DataLoader, DataSource
from olmo_eval.evals.extract import extract_think_answer, extract_think_answer_only
from olmo_eval.evals.tasks.common import Task, register_variant

logger = logging.getLogger(__name__)

judge_sampling = SamplingParams(max_tokens=None, temperature=0.6, top_p=0.95)
base_sampling = SamplingParams(
    max_tokens=1024,
    temperature=0.6,
    top_p=0.6,
    stop_sequences=("Question:", "</s>", "<|im_end|>", "\n\n"),
)


class SafetyBase(Task):
    """Base class for safety evaluation tasks."""

    data_source: DataSource | None = None
    fewshot_split: str | None = None
    fewshot_sample: bool = False
    sampling_params: SamplingParams = judge_sampling
    formatter = ChatFormatter()
    answer_extractor = extract_think_answer

    @property
    def instances(self) -> Iterator[Instance]:
        """Yield instances from the dataset."""
        if self._instances_cache is None:
            self._instances_cache = []
            loader = DataLoader()
            source = self.config.get_data_source()

            for idx, doc in enumerate(loader.load(source)):
                instance = self.process_doc(doc, idx)
                if instance is not None:
                    self._instances_cache.append(instance)

        yield from self._instances_cache

    @property
    def request_type(self) -> RequestType:
        """Return the request type for this task."""
        if self.config.formatter is not None:
            return self.config.formatter.request_type
        return RequestType.CHAT

    def format_request(self, instance: Instance) -> LMRequest:
        """Format an instance into an LM request.

        Delegates to the configured formatter (ChatFormatter by default).
        """
        if self.config.formatter is not None:
            return self.config.formatter.format(instance)
        # Fallback: create a simple chat request
        return LMRequest(
            request_type=RequestType.CHAT,
            messages=({"role": "user", "content": instance.question},),
        )


def safety_metrics(scorer, subsets: tuple[str, ...], macro_subsets: tuple[str, ...] = ()):
    """Build the full metric tuple for a safety judge scorer.

    ``macro_subsets`` adds an unweighted mean over those subsets, for tasks whose
    halves must count equally rather than by instance volume.
    """
    metrics = (
        AccuracyMetric(scorer=scorer),
        SafetyErrorMetric(scorer=scorer),
        *(SubsetAccuracyMetric(name=name, scorer=scorer) for name in subsets),
    )
    if macro_subsets:
        metrics = (*metrics, MacroSubsetAccuracyMetric(subsets=macro_subsets, scorer=scorer))
    return metrics


def make_mcq_prompt(question: str, choices: list[str], label_prefix: str = " ") -> str:
    choice_labels = "ABCD"
    label_format = label_prefix + "A."
    choices_text = "\n".join(
        f"{label_format.replace('A', label)} {text}"
        for label, text in zip(choice_labels, choices, strict=False)
    )
    return f"Question: {question}\n{choices_text}\nAnswer:"


def register_safety_variants(
    eval_name: str,
    subsets: tuple[str, ...],
    scorer=None,
    scorer_name=None,
    macro_subsets: tuple[str, ...] = (),
):
    """
    Build the four variants that the base wildguard safety tasks use.

    ``macro_subsets`` makes the primary metric an unweighted mean over those
    subsets instead of an instance-weighted mean across the whole task.
    """

    # Initialize the safety scorer
    if scorer is None:
        scorer = SafetyScorer(
            provider_name="wg_judge",
            judge_format="wildguard",
            judge_request_type=RequestType.COMPLETION,
        )

    if scorer_name is None:
        scorer_name = "wg_judge"

    def primary_metric(variant_scorer):
        if macro_subsets:
            return MacroSubsetAccuracyMetric(subsets=macro_subsets, scorer=variant_scorer)
        return AccuracyMetric(scorer=variant_scorer)

    # OpenAI judge variant - uses OpenAI API as the judge
    register_variant(
        eval_name,
        "openai_judge",
        metrics=safety_metrics(SafetyScorer, subsets, macro_subsets),
        primary_metric=primary_metric(SafetyScorer),
        sampling_params=judge_sampling,
        required_secrets=("OPENAI_API_KEY",),
    )

    register_variant(
        eval_name,
        scorer_name,
        metrics=safety_metrics(scorer, subsets, macro_subsets),
        primary_metric=primary_metric(scorer),
        sampling_params=judge_sampling,
    )

    register_variant(
        eval_name,
        f"{scorer_name}_thinking",
        metrics=safety_metrics(scorer, subsets, macro_subsets),
        primary_metric=primary_metric(scorer),
        sampling_params=judge_sampling,
        answer_extractor=extract_think_answer_only,
    )

    register_variant(
        eval_name,
        "base",
        metrics=safety_metrics(scorer, subsets, macro_subsets),
        primary_metric=primary_metric(scorer),
        sampling_params=base_sampling,
        formatter=CompletionFormatter(template="Question: {question}\nAnswer:"),
    )
