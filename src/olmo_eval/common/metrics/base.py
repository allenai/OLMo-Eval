"""Metric base class and implementations."""

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from olmo_eval.common.scorers import (
    BitsPerByteScorer,
    ExactMatchScorer,
    F1Scorer,
    LogprobScorer,
    PerplexityScorer,
    SafetyScorer,
    Scorer,
    SQuADF1Scorer,
    SubstringRecallScorer,
    ToolCallScorer,
)
from olmo_eval.common.types import Response
from olmo_eval.common.utils import compute_pass_at_k, compute_pass_pow_k


def display_format_for_name(name: str) -> str:
    """Default display format for a metric name: ``percentage`` or ``raw``.

    Proportions (accuracy, F1, recall, pass@k) are ``percentage``; everything else keeps its
    native units.
    """
    if name in {"accuracy", "f1", "recall", "tool_accuracy"} or name.endswith("_accuracy"):
        return "percentage"
    if name.startswith(("pass_at_", "pass_pow_")):
        return "percentage"
    return "raw"


@dataclass(frozen=True)
class Metric(ABC):
    """Abstract base class for aggregating scores across responses.

    Subclasses must define:
        - name: str class attribute identifying the metric
        - scorer: type[Scorer] class attribute for the associated scorer
        - compute(): method to aggregate scores from responses
    """

    name: str = ""
    scorer: type[Scorer] | Scorer = ExactMatchScorer

    @abstractmethod
    def compute(self, responses: Sequence[Response]) -> float:
        """Compute aggregate metric from scored responses."""
        ...

    def compute_instance(self, response: Response) -> float | None:
        """Compute the per-instance value that should back exact metric storage.

        The default path is conservative:
        - if a task stores its exact per-instance metric directly in ``response.scores``
          under the metric name, prefer that;
        - otherwise, only reuse the scorer channel when ``supports_scorer_fallback`` allows it.

        Metrics with richer per-instance semantics should override this method.
        """
        metric_value = response.scores.get(self.name)
        if isinstance(metric_value, (int, float)):
            return float(metric_value)

        if not self.supports_scorer_fallback():
            return None

        try:
            scorer_name = self.scorer().name
        except Exception:
            return None
        scorer_value = response.scores.get(scorer_name)
        if isinstance(scorer_value, (int, float)):
            return float(scorer_value)
        return None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary."""
        scorer = self.scorer
        scorer_name = (
            scorer.__name__
            if isinstance(scorer, type)
            else scorer.to_dict()
            if hasattr(scorer, "to_dict")
            else asdict(scorer)
        )
        return {"type": self.__class__.__name__, "name": self.name, "scorer": scorer_name}

    def supports_scorer_fallback(self) -> bool:
        """Whether scorer-level instance values match the metric's per-instance signal.

        When a prediction has no exact per-instance value for a metric key (for example
        ``accuracy:exact_match``), ``augment_prediction_instance_metrics`` fills it from the
        scorer channel (for example ``exact_match:exact_match``). That fallback is only valid
        when the task-level metric is literally an average of per-instance scorer values.

        Metrics that aggregate across multiple continuations, weight instances unevenly,
        or otherwise derive the final value from richer response structure should override
        this to ``False``.
        """
        return True

    def higher_is_better(self) -> bool:
        """Whether larger metric values mean better model quality."""
        return True

    def display_format(self) -> str:
        """Return the display format the dashboard uses for this metric.

        The uploader sends it as metric metadata. ``percentage`` means the metric is
        naturally interpreted on a 0-1 scale and should be rendered as percentages /
        percentage-point deltas. ``raw`` means the metric should stay in its native
        numeric units.
        """
        return display_format_for_name(self.name)

    def metric_unit(self) -> str:
        """Return the unit family used to decide whether suite pooling is comparable."""
        if self.display_format() == "percentage":
            return "proportion"
        return self.name


@dataclass(frozen=True, slots=True)
class AccuracyMetric(Metric):
    """Mean accuracy across all responses for a given scorer."""

    name: str = "accuracy"
    scorer: type[Scorer] | Scorer = ExactMatchScorer

    def compute(self, responses: Sequence[Response]) -> float:
        if not responses:
            return 0.0
        scorer_name = self.scorer().name
        total = sum(r.scores.get(scorer_name, 0.0) for r in responses)
        return total / len(responses)


@dataclass(frozen=True, slots=True)
class F1Metric(Metric):
    """Mean F1 score across all responses."""

    name: str = "f1"
    scorer: type[Scorer] = F1Scorer

    def compute(self, responses: Sequence[Response]) -> float:
        if not responses:
            return 0.0
        scorer_name = self.scorer().name
        total = sum(r.scores.get(scorer_name, 0.0) for r in responses)
        return total / len(responses)


def _select_gold_output(response: Response):
    """Select the gold/correct output from a response.

    For multi-output responses (e.g. multiple choice), uses
    ``instance.metadata["gold_idx"]`` to pick the correct one,
    falling back to the first output.  Returns None if the response
    has no outputs or the selected output has no logprobs / zero bytes.
    """
    outputs = response.outputs
    if not outputs:
        return None

    if len(outputs) > 1:
        gold_idx = response.instance.metadata.get("gold_idx")
        if gold_idx is not None and 0 <= gold_idx < len(outputs):
            output = outputs[gold_idx]
        else:
            output = outputs[0]
    else:
        output = outputs[0]

    if output.logprobs is None:
        return None

    num_bytes = len(output.text.encode("utf-8")) if output.text else 0
    if num_bytes == 0:
        return None

    return output


@dataclass(frozen=True, slots=True)
class SQuADF1Metric(Metric):
    """Mean SQuAD-style F1 score: max F1 over all reference answers."""

    name: str = "f1"
    scorer: type[Scorer] = SQuADF1Scorer

    def compute(self, responses: Sequence[Response]) -> float:
        if not responses:
            return 0.0
        scorer_name = self.scorer().name
        total = sum(r.scores.get(scorer_name, 0.0) for r in responses)
        return total / len(responses)


@dataclass(frozen=True, slots=True)
class RecallMetric(Metric):
    """Recall metric.

    Aggregates recall scores across instances by averaging them.
    Works with any recall-based scorer (e.g., SubstringRecallScorer).

    Used for tasks that measure information retrieval or recall,
    such as RULER and similar long-context benchmarks.
    """

    name: str = "recall"
    scorer: type[Scorer] = SubstringRecallScorer

    def compute(self, responses: Sequence[Response]) -> float:
        """Compute average recall across all responses.

        Args:
            responses: Sequence of Response objects with scores

        Returns:
            Average recall score (0.0 to 1.0)
        """
        if not responses:
            return 0.0
        scorer_name = self.scorer().name
        total = sum(r.scores.get(scorer_name, 0.0) for r in responses)
        return total / len(responses)


@dataclass(frozen=True, slots=True)
class BPBMetricInstanceAvg(Metric):
    """Arithmetic mean of per-instance bits-per-byte.

    Computes per-instance BPB as: -logprob / (num_bytes * log(2)),
    then returns the simple (unweighted) mean across all instances.
    """

    name: str = "bits_per_byte"
    scorer: type[Scorer] = BitsPerByteScorer

    def compute(self, responses: Sequence[Response]) -> float:
        if not responses:
            return 0.0

        scorer = self.scorer()
        bpb_values: list[float] = []

        for response in responses:
            output = _select_gold_output(response)
            if output is None:
                continue
            bpb_values.append(scorer.score(response.instance, output))

        if not bpb_values:
            return 0.0

        return sum(bpb_values) / len(bpb_values)

    def compute_instance(self, response: Response) -> float | None:
        output = _select_gold_output(response)
        if output is None:
            return None
        return float(self.scorer().score(response.instance, output))

    def supports_scorer_fallback(self) -> bool:
        # Keep BPB handling conservative: read the exact stored metric key.
        return False

    def higher_is_better(self) -> bool:
        return False


@dataclass(frozen=True, slots=True)
class BPBMetricByteAvg(Metric):
    """Byte-weighted (corpus-level) bits-per-byte.

    Each instance's BPB is computed via the scorer, then weighted by its byte
    count so that longer texts contribute proportionally more to the result.
    Equivalent to ``-sum(logprobs) / (sum(bytes) * log(2))`` across the corpus.
    """

    name: str = "bits_per_byte"
    scorer: type[Scorer] = BitsPerByteScorer

    def compute(self, responses: Sequence[Response]) -> float:
        if not responses:
            return 0.0

        scorer = self.scorer()
        weighted_sum = 0.0
        total_bytes = 0

        for response in responses:
            output = _select_gold_output(response)
            if output is None:
                continue

            num_bytes = len(output.text.encode("utf-8")) if output.text else 0
            if num_bytes == 0:
                continue
            bpb = scorer.score(response.instance, output)
            weighted_sum += bpb * num_bytes
            total_bytes += num_bytes

        if total_bytes == 0:
            return 0.0

        return weighted_sum / total_bytes

    def compute_instance(self, response: Response) -> float | None:
        output = _select_gold_output(response)
        if output is None:
            return None
        return float(self.scorer().score(response.instance, output))

    def supports_scorer_fallback(self) -> bool:
        # Byte-weighted BPB is a corpus aggregate, so scorer-level per-instance values
        # are not equivalent to the stored task metric.
        return False

    def higher_is_better(self) -> bool:
        return False


@dataclass(frozen=True, slots=True)
class MeanPerplexityMetric(Metric):
    """Mean perplexity of the gold/correct completion.

    For tasks with multiple continuations (e.g., multiple choice), this returns
    the perplexity of the correct continuation using `instance.metadata["gold_idx"]`.
    For single-continuation tasks, it returns the perplexity of that continuation.
    """

    name: str = "perplexity"
    scorer: type[Scorer] = PerplexityScorer

    def compute(self, responses: Sequence[Response]) -> float:
        if not responses:
            return 0.0

        scorer = self.scorer()
        total = 0.0

        for response in responses:
            outputs = response.outputs
            if not outputs:
                continue

            if len(outputs) > 1:
                # Multiple outputs: select the gold/correct continuation
                gold_idx = response.instance.metadata.get("gold_idx")
                if gold_idx is not None and 0 <= gold_idx < len(outputs):
                    output = outputs[gold_idx]
                else:
                    # Fallback to first output if gold_idx not available
                    output = outputs[0]
            else:
                # Single output: use it directly
                output = outputs[0]

            total += scorer.score(response.instance, output)

        return total / len(responses)

    def compute_instance(self, response: Response) -> float | None:
        outputs = response.outputs
        if not outputs:
            return None

        if len(outputs) > 1:
            gold_idx = response.instance.metadata.get("gold_idx")
            if gold_idx is not None and 0 <= gold_idx < len(outputs):
                output = outputs[gold_idx]
            else:
                output = outputs[0]
        else:
            output = outputs[0]

        if output.logprobs is None:
            return None
        return float(self.scorer().score(response.instance, output))

    def higher_is_better(self) -> bool:
        return False


@dataclass(frozen=True, slots=True)
class PassAtKMetric(Metric):
    """Compute pass@k metric across multiple samples per task.

    The probability that at least one of k samples passes.
    """

    name: str = field(init=False)
    k: int = 1
    scorer: type[Scorer] = field(kw_only=True)

    def __post_init__(self) -> None:
        # Use unique name per k value so metrics don't overwrite each other
        object.__setattr__(self, "name", f"pass_at_{self.k}")

    def compute(self, responses: Sequence[Response]) -> float:
        """Compute pass@k across all tasks."""
        if not responses:
            return 0.0

        scorer_name = self.scorer().name
        score_key = f"score:{scorer_name}"

        # Group per-output scores by task ID
        task_results: dict[str, list[float]] = {}
        for r in responses:
            task_id = r.instance.metadata.get("id", "unknown")
            if task_id not in task_results:
                task_results[task_id] = []
            # Use per-output scores stored during scoring (one score per sample)
            for output in r.outputs:
                if output.metadata and score_key in output.metadata:
                    task_results[task_id].append(output.metadata[score_key])
                else:
                    # Fallback: single-sample response, use response-level score
                    task_results[task_id].append(r.scores.get(scorer_name, 0.0))
                    break

        # Compute pass@k for each task
        pass_at_k_values = []
        for scores in task_results.values():
            n = len(scores)
            c = sum(1 for s in scores if s > 0.5)  # Count passing
            pass_at_k_values.append(compute_pass_at_k(n, c, min(self.k, n)))

        return sum(pass_at_k_values) / len(pass_at_k_values) if pass_at_k_values else 0.0

    def compute_instance(self, response: Response) -> float | None:
        scorer_name = self.scorer().name
        score_key = f"score:{scorer_name}"
        sample_scores: list[float] = []
        for output in response.outputs:
            score = (output.metadata or {}).get(score_key)
            if isinstance(score, (int, float)):
                sample_scores.append(float(score))

        if not sample_scores:
            scorer_score = response.scores.get(scorer_name)
            if isinstance(scorer_score, (int, float)) and len(response.outputs) <= 1:
                sample_scores.append(float(scorer_score))

        if not sample_scores:
            return None

        n = len(sample_scores)
        c = sum(1 for score in sample_scores if score > 0.5)
        return float(compute_pass_at_k(n, c, min(self.k, n)))

    def supports_scorer_fallback(self) -> bool:
        return self.k == 1


@dataclass(frozen=True, slots=True)
class PassPowKMetric(Metric):
    """Compute pass^k metric (all k trials succeed).

    The probability that k consecutive runs all succeed. Computed as (success_rate)^k.
    """

    name: str = field(init=False)
    k: int = 1
    scorer: type[Scorer] = field(kw_only=True)

    def __post_init__(self) -> None:
        # Use unique name per k value so metrics don't overwrite each other
        object.__setattr__(self, "name", f"pass_pow_{self.k}")

    def compute(self, responses: Sequence[Response]) -> float:
        """Compute pass^k across all tasks."""
        if not responses:
            return 0.0

        scorer_name = self.scorer().name
        score_key = f"score:{scorer_name}"

        # Group per-output scores by task ID
        task_results: dict[str, list[float]] = {}
        for r in responses:
            task_id = r.instance.metadata.get("id", "unknown")
            if task_id not in task_results:
                task_results[task_id] = []
            # Use per-output scores stored during scoring (one score per sample)
            for output in r.outputs:
                if output.metadata and score_key in output.metadata:
                    task_results[task_id].append(output.metadata[score_key])
                else:
                    # Fallback: single-sample response, use response-level score
                    task_results[task_id].append(r.scores.get(scorer_name, 0.0))
                    break

        # Compute pass^k for each task
        pass_pow_k_values = []
        for scores in task_results.values():
            n = len(scores)
            c = sum(1 for s in scores if s > 0.5)  # Count passing
            pass_pow_k_values.append(compute_pass_pow_k(n, c, self.k))

        return sum(pass_pow_k_values) / len(pass_pow_k_values) if pass_pow_k_values else 0.0

    def compute_instance(self, response: Response) -> float | None:
        scorer_name = self.scorer().name
        score_key = f"score:{scorer_name}"
        sample_scores: list[float] = []
        for output in response.outputs:
            score = (output.metadata or {}).get(score_key)
            if isinstance(score, (int, float)):
                sample_scores.append(float(score))

        if not sample_scores:
            scorer_score = response.scores.get(scorer_name)
            if isinstance(scorer_score, (int, float)) and len(response.outputs) <= 1:
                sample_scores.append(float(scorer_score))

        if not sample_scores:
            return None

        n = len(sample_scores)
        c = sum(1 for score in sample_scores if score > 0.5)
        return float(compute_pass_pow_k(n, c, self.k))

    def supports_scorer_fallback(self) -> bool:
        return self.k == 1


@dataclass(frozen=True, slots=True)
class LogprobMCAccuracyMetric(Metric):
    """Multiple-choice accuracy via logprob argmax.

    Picks the continuation with the highest total logprob and checks whether
    its index matches ``instance.metadata["gold_idx"]``.
    """

    name: str = "accuracy"
    scorer: type[Scorer] = LogprobScorer

    def compute(self, responses: Sequence[Response]) -> float:
        if not responses:
            return 0.0
        scorer = self.scorer()
        correct = 0
        for response in responses:
            gold_idx = response.instance.metadata.get("gold_idx")
            if gold_idx is None or not response.outputs:
                continue
            logprob_sums = [scorer.score(response.instance, o) for o in response.outputs]
            if logprob_sums.index(max(logprob_sums)) == gold_idx:
                correct += 1
        return correct / len(responses)

    def compute_instance(self, response: Response) -> float | None:
        gold_idx = response.instance.metadata.get("gold_idx")
        if gold_idx is None or not response.outputs:
            return None
        scorer = self.scorer()
        logprob_sums = [scorer.score(response.instance, output) for output in response.outputs]
        return 1.0 if logprob_sums.index(max(logprob_sums)) == gold_idx else 0.0

    def supports_scorer_fallback(self) -> bool:
        return False


@dataclass(frozen=True, slots=True)
class LogprobUncondMCAccuracyMetric(Metric):
    """Multiple-choice accuracy with unconditional normalization (acc_uncond).

    Expects responses where the first half of outputs are conditioned (full
    context) and the second half are unconditional (e.g. just "Answer:").
    The number of actual choices is stored in ``instance.metadata["num_choices"]``.

    For each choice *i*, computes::

        score_i = sum_logprob_conditioned[i] - sum_logprob_unconditional[i]

    Picks the choice with the highest score and checks whether it matches
    ``instance.metadata["gold_idx"]``.

    This matches the ``acc_uncond`` metric from oe-eval-internal's MCAccuracy.
    """

    name: str = "accuracy"
    scorer: type[Scorer] = LogprobScorer

    def compute(self, responses: Sequence[Response]) -> float:
        if not responses:
            return 0.0
        correct = 0
        for response in responses:
            gold_idx = response.instance.metadata.get("gold_idx")
            num_choices = response.instance.metadata.get("num_choices")
            if gold_idx is None or num_choices is None or not response.outputs:
                continue
            outputs = response.outputs
            cond_outputs = outputs[:num_choices]
            uncond_outputs = outputs[num_choices:]
            scores = []
            for cond, uncond in zip(cond_outputs, uncond_outputs, strict=True):
                cond_lp = sum(lp["logprob"] for lp in (cond.logprobs or []))
                uncond_lp = sum(lp["logprob"] for lp in (uncond.logprobs or []))
                scores.append(cond_lp - uncond_lp)
            if scores.index(max(scores)) == gold_idx:
                correct += 1
        return correct / len(responses)

    def compute_instance(self, response: Response) -> float | None:
        gold_idx = response.instance.metadata.get("gold_idx")
        num_choices = response.instance.metadata.get("num_choices")
        if gold_idx is None or num_choices is None or not response.outputs:
            return None
        outputs = response.outputs
        cond_outputs = outputs[:num_choices]
        uncond_outputs = outputs[num_choices:]
        if len(cond_outputs) != len(uncond_outputs) or not cond_outputs:
            return None

        scores: list[float] = []
        for cond, uncond in zip(cond_outputs, uncond_outputs, strict=True):
            if cond.logprobs is None or uncond.logprobs is None:
                return None
            cond_lp = sum(lp["logprob"] for lp in cond.logprobs)
            uncond_lp = sum(lp["logprob"] for lp in uncond.logprobs)
            scores.append(cond_lp - uncond_lp)
        return 1.0 if scores.index(max(scores)) == gold_idx else 0.0

    def supports_scorer_fallback(self) -> bool:
        return False


@dataclass(frozen=True, slots=True)
class LogprobPerCharMCAccuracyMetric(Metric):
    """Multiple-choice accuracy via character-length-normalized logprob argmax.

    For each continuation, divides the total logprob by the number of characters
    in the continuation text, then picks the continuation with the highest
    normalized logprob. Checks whether its index matches
    ``instance.metadata["gold_idx"]``.

    This matches the ``acc_per_char`` metric from oe-eval-internal's MCAccuracy.
    """

    name: str = "accuracy"
    scorer: type[Scorer] = LogprobScorer

    def compute(self, responses: Sequence[Response]) -> float:
        if not responses:
            return 0.0
        scorer = self.scorer()
        correct = 0
        for response in responses:
            gold_idx = response.instance.metadata.get("gold_idx")
            if gold_idx is None or not response.outputs:
                continue
            logprob_per_char = []
            for o in response.outputs:
                total_logprob = scorer.score(response.instance, o)
                num_chars = max(len(o.text) if o.text else 0, 1)
                logprob_per_char.append(total_logprob / num_chars)
            if logprob_per_char.index(max(logprob_per_char)) == gold_idx:
                correct += 1
        return correct / len(responses)

    def compute_instance(self, response: Response) -> float | None:
        gold_idx = response.instance.metadata.get("gold_idx")
        if gold_idx is None or not response.outputs:
            return None
        scorer = self.scorer()
        logprob_per_char: list[float] = []
        for output in response.outputs:
            total_logprob = scorer.score(response.instance, output)
            num_chars = max(len(output.text) if output.text else 0, 1)
            logprob_per_char.append(total_logprob / num_chars)
        return 1.0 if logprob_per_char.index(max(logprob_per_char)) == gold_idx else 0.0

    def supports_scorer_fallback(self) -> bool:
        return False


@dataclass(frozen=True, slots=True)
class LogprobPerTokenMCAccuracyMetric(Metric):
    """Multiple-choice accuracy via token-length-normalized logprob argmax.

    For each continuation, divides the total logprob by the number of tokens,
    then picks the continuation with the highest normalized logprob. Checks
    whether its index matches ``instance.metadata["gold_idx"]``.

    This matches the ``acc_per_token`` metric from oe-eval-internal's MCAccuracy.
    """

    name: str = "accuracy"
    scorer: type[Scorer] = LogprobScorer

    def compute(self, responses: Sequence[Response]) -> float:
        if not responses:
            return 0.0
        scorer = self.scorer()
        correct = 0
        for response in responses:
            gold_idx = response.instance.metadata.get("gold_idx")
            if gold_idx is None or not response.outputs:
                continue
            logprob_per_token = []
            for o in response.outputs:
                total_logprob = scorer.score(response.instance, o)
                num_tokens = max(len(o.logprobs) if o.logprobs else 0, 1)
                logprob_per_token.append(total_logprob / num_tokens)
            if logprob_per_token.index(max(logprob_per_token)) == gold_idx:
                correct += 1
        return correct / len(responses)

    def compute_instance(self, response: Response) -> float | None:
        gold_idx = response.instance.metadata.get("gold_idx")
        if gold_idx is None or not response.outputs:
            return None
        scorer = self.scorer()
        logprob_per_token: list[float] = []
        for output in response.outputs:
            total_logprob = scorer.score(response.instance, output)
            num_tokens = max(len(output.logprobs) if output.logprobs else 0, 1)
            logprob_per_token.append(total_logprob / num_tokens)
        return 1.0 if logprob_per_token.index(max(logprob_per_token)) == gold_idx else 0.0

    def supports_scorer_fallback(self) -> bool:
        return False


@dataclass(frozen=True, slots=True)
class GreedyAccuracyMetric(Metric):
    """Greedy decoding accuracy for single-continuation tasks.

    Checks whether the model would greedily decode the expected continuation
    token-by-token, using the ``is_greedy`` flag computed by the inference provider.

    This is the correct metric for tasks like LAMBADA where there is only one
    continuation per instance and we want to know if the model would have
    produced that exact continuation via greedy decoding.
    """

    name: str = "greedy_accuracy"
    scorer: type[Scorer] = LogprobScorer

    def compute(self, responses: Sequence[Response]) -> float:
        if not responses:
            return 0.0
        correct = 0
        for response in responses:
            if not response.outputs:
                continue
            # For single-continuation tasks, check the gold continuation
            gold_idx = response.instance.metadata.get("gold_idx", 0)
            if gold_idx < len(response.outputs):
                output = response.outputs[gold_idx]
                if output.metadata.get("is_greedy", False):
                    correct += 1
        return correct / len(responses)

    def compute_instance(self, response: Response) -> float | None:
        if not response.outputs:
            return None
        gold_idx = response.instance.metadata.get("gold_idx", 0)
        if gold_idx >= len(response.outputs):
            return None
        return 1.0 if response.outputs[gold_idx].metadata.get("is_greedy", False) else 0.0

    def supports_scorer_fallback(self) -> bool:
        return False


@dataclass(frozen=True, slots=True)
class ToolAccuracyMetric(Metric):
    """Mean tool call accuracy across all responses."""

    name: str = "tool_accuracy"
    scorer: type[Scorer] = ToolCallScorer

    def compute(self, responses: Sequence[Response]) -> float:
        if not responses:
            return 0.0
        scorer_name = self.scorer().name
        total = sum(r.scores.get(scorer_name, 0.0) for r in responses)
        return total / len(responses)


@dataclass(frozen=True, slots=True)
class CorpusPerplexityMetric(Metric):
    """Corpus-level (token-weighted) perplexity.

    This is the standard token-weighted aggregation across documents.
    Documents that exceed the model's context length are truncated.
    """

    name: str = "corpus_perplexity"
    scorer: type[Scorer] = LogprobScorer

    def compute(self, responses: Sequence[Response]) -> float:
        if not responses:
            return 0.0

        scorer = self.scorer()
        total_logprob = 0.0
        total_tokens = 0

        for response in responses:
            outputs = response.outputs
            if not outputs:
                continue
            # This should contain the output for the entire doc
            output = outputs[0]
            if output.logprobs is None:
                continue
            total_logprob += scorer.score(response.instance, output)
            total_tokens += len(output.logprobs)

        if total_tokens == 0:
            return 0.0

        avg_logprob = total_logprob / total_tokens
        return math.exp(-avg_logprob)

    def compute_instance(self, response: Response) -> float | None:
        if not response.outputs:
            return None
        output = response.outputs[0]
        if output.logprobs is None or not output.logprobs:
            return None
        total_logprob = self.scorer().score(response.instance, output)
        avg_logprob = total_logprob / len(output.logprobs)
        return float(math.exp(-avg_logprob))

    def supports_scorer_fallback(self) -> bool:
        return False

    def higher_is_better(self) -> bool:
        return False


@dataclass(frozen=True, slots=True)
class SubsetAccuracyMetric(Metric):
    """
    Calculates the accuracy across responses for a given
    subset. For use to avoid double calculation in cases
    where a single instance might belong to multiple subsets
    """

    # defaults to all responses
    # expected format is subset_name__subset_value ie functional_category__copyright
    name: str = "any__any"
    scorer: type[Scorer] | Scorer = SafetyScorer

    def compute_instance(self, response: Response) -> float | None:
        subset, cat = self.name.split("__")
        if subset != "any" and response.instance.metadata.get(subset) != cat:
            return None
        score = response.scores.get(self.scorer().name)
        return float(score) if score is not None else None

    def supports_scorer_fallback(self) -> bool:
        return False

    def compute(self, responses: Sequence[Response]) -> float:
        """Compute aggregate metric from scored responses."""
        if not responses:
            return 0.0
        scorer_name = self.scorer().name

        if self.name == "any__any":
            total = sum(r.scores.get(scorer_name, 0.0) for r in responses)
            return total / len(responses)

        subset, cat = self.name.split("__")
        subset_responses = []
        for r in responses:
            if r.instance.metadata[subset] == cat:
                subset_responses.append(r.scores.get(scorer_name, 0.0))

        return sum(subset_responses) / len(subset_responses) if subset_responses else -1


@dataclass(frozen=True, slots=True)
class MacroSubsetAccuracyMetric(Metric):
    """
    Unweighted mean of per-subset accuracies.

    Every subset contributes equally regardless of how many instances it holds,
    so a small subset is not diluted by a larger one. Subsets with no matching
    instances are skipped rather than counted as zero.
    """

    name: str = "macro_accuracy"
    scorer: type[Scorer] | Scorer = SafetyScorer
    # Subset names, each formatted as ``metadata_key__value``.
    subsets: tuple[str, ...] = ()

    def compute(self, responses: Sequence[Response]) -> float:
        if not responses:
            return 0.0
        scorer_name = self.scorer().name

        subset_means: list[float] = []
        for subset_name in self.subsets:
            subset, cat = subset_name.split("__")
            scores = [
                r.scores.get(scorer_name, 0.0)
                for r in responses
                if r.instance.metadata.get(subset) == cat
            ]
            if scores:
                subset_means.append(sum(scores) / len(scores))

        return sum(subset_means) / len(subset_means) if subset_means else 0.0

    def compute_instance(self, response: Response) -> float | None:
        """A macro average has no per-instance equivalent."""
        return None

    def supports_scorer_fallback(self) -> bool:
        return False


@dataclass(frozen=True, slots=True)
class SafetyErrorMetric(Metric):
    """
    Counts the number of parsing errors from LLM-as-a-judge
    Designed for the metadata format from the safety evals
    """

    scorer: type[Scorer] | Scorer = SafetyScorer
    name: str = "parsing_error"

    def compute(self, responses: Sequence[Response]) -> float:
        """Compute aggregate metric from scored responses."""
        if not responses:
            return 0.0

        total = sum(int(r.instance.metadata.get("is_parsing_error", False)) for r in responses)

        return total

    def compute_instance(self, response: Response) -> float | None:
        """Compute the per-instance value"""

        return float(response.instance.metadata.get("is_parsing_error", False))

    def supports_scorer_fallback(self) -> bool:
        return False
