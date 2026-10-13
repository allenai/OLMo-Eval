"""Base Task class and configuration."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, ClassVar, NamedTuple

from olmo_eval.common.formatters import Formatter
from olmo_eval.common.metrics import Metric
from olmo_eval.common.repr import hide_unset
from olmo_eval.common.scorers import Scorer, ScoringIncompleteError
from olmo_eval.common.types import (
    Instance,
    LMOutput,
    LMRequest,
    MetricName,
    RequestType,
    Response,
    SamplingParams,
    Split,
)

if TYPE_CHECKING:
    from olmo_eval.common.execution import ScoringContext
    from olmo_eval.data import DataSource

logger = logging.getLogger(__name__)


def _format_scoring_error(exc: Exception, *, phase: str) -> dict[str, str]:
    """Build a JSON-serializable scoring error payload."""
    error = {
        "phase": phase,
        "type": type(exc).__qualname__,
    }
    message = str(exc).strip()
    if message:
        error["message"] = message
    if isinstance(exc, ScoringIncompleteError):
        error["infrastructure"] = "true"
    else:
        try:
            from olmo_eval.harness.sandbox import SandboxInfrastructureError

            if isinstance(exc, SandboxInfrastructureError):
                error["infrastructure"] = "true"
        except ImportError:
            pass
    return error


def _store_output_score(
    output: LMOutput,
    *,
    scorer_name: str,
    score: float,
    scoring_error: dict[str, str] | None = None,
) -> None:
    """Store a per-output scorer result on output metadata."""
    if output.metadata is None:
        output.metadata = {}
    output.metadata[f"score:{scorer_name}"] = score
    if scoring_error is not None:
        existing = output.metadata.get("scoring_errors")
        if not isinstance(existing, dict):
            existing = {}
            output.metadata["scoring_errors"] = existing
        existing[scorer_name] = scoring_error


class OutputScore(NamedTuple):
    """Scoring result for a single output (execution / context scorers)."""

    resp_idx: int
    scorer_name: str
    out_idx: int
    score: float
    scoring_error: dict[str, str] | None


class ResponseScore(NamedTuple):
    """Scoring result for a whole response (process scorers)."""

    resp_idx: int
    scorer_name: str
    output_scores: dict[int, float]
    output_errors: dict[int, dict[str, str]]


@dataclass(frozen=True, slots=True)
class SandboxEnv:
    """Named sandbox environment with dependencies for code execution scoring.

    Tasks sharing the same name share a container. Different names get
    isolated containers with only their declared dependencies installed.
    """

    name: str
    dependencies: tuple[str, ...] = ()
    dockerfile_extra: tuple[str, ...] = ()

    @property
    def capability(self) -> frozenset[str]:
        """Capability tag used to route execution to this sandbox."""
        return frozenset({f"sandbox:{self.name}"})


class OutputScoreAggregation(StrEnum):
    """How per-output scorer values collapse to a response-level score."""

    MAX = "max"
    FIRST = "first"
    MEAN = "mean"


@hide_unset()
@dataclass
class TaskConfig:
    """Configuration for a task.

    Examples:
        # With DataSource object
        >>> from olmo_eval.data import DataSource
        >>> config = TaskConfig(
        ...     name="arc_challenge",
        ...     data_source=DataSource(path="allenai/ai2_arc", subset="ARC-Challenge"),
        ... )

        # With URI string
        >>> config = TaskConfig(
        ...     name="mmlu_math",
        ...     data_source="hf://cais/mmlu?subset=abstract_algebra",
        ... )
    """

    name: str

    # Data source configuration
    data_source: DataSource | str | None = None
    fewshot_source: DataSource | str | None = None

    # Task configuration
    formatter: Formatter | None = None
    metrics: tuple[Metric, ...] = ()
    num_fewshot: int = 0
    fewshot_seed: int = 42
    limit: int | None = None
    seed: int = 42
    split: Split = Split.TEST
    primary_metric: MetricName | Metric | None = None
    sampling_params: SamplingParams | None = None
    output_score_aggregation: OutputScoreAggregation = OutputScoreAggregation.MAX

    #: Maximum prompt length in tokens for loglikelihood truncation (matches oe-eval max_length).
    #: When set, prompts exceeding this length are left-truncated before scoring.
    max_length: int | None = None
    answer_extractor: Callable[[str], str] | None = None

    #: Drop ``<think>...</think>`` traces before scoring; see :meth:`Task.strip_thinking_traces`.
    strip_thinking: bool = False

    #: Runtime dependencies to install for this task (package specs like "pkg==1.0" or git URLs)
    dependencies: list[str] | None = None

    #: Sandbox environment for code execution scoring. None = use default sandbox.
    sandbox_env: SandboxEnv | None = None

    #: Scheduler-only weight hint for shared sandbox allocation.
    sandbox_allocation_weight: float = 1.0

    #: Env-var names this task needs at runtime (e.g. an LLM-judge API key). The
    #: beaker launcher mounts each as the user-scoped secret ``{user}_{NAME}``.
    required_secrets: tuple[str, ...] = ()

    # LLM-as-judge configuration. These stay optional so adding them does not
    # perturb hashes for tasks that do not use a judge.
    judge_model: str | None = None
    judge_reasoning_effort: str | None = None
    judge_max_tokens: int | None = None

    #: How a task that builds its prompt from a bare label should render it. The
    #: vision tasks read these; they follow the checkpoint, since instruction-tuned
    #: and pretrain models were trained on different prompt forms.
    prompt_templates: str | None = None
    system_prompt_style: str | None = None

    def __post_init__(self) -> None:
        """Validate scheduler-only sandbox allocation hints."""
        if isinstance(self.output_score_aggregation, str):
            try:
                self.output_score_aggregation = OutputScoreAggregation(
                    self.output_score_aggregation
                )
            except ValueError as exc:
                valid = ", ".join(option.value for option in OutputScoreAggregation)
                raise ValueError(
                    f"output_score_aggregation must be one of: {valid}; "
                    f"got {self.output_score_aggregation!r}"
                ) from exc

        try:
            weight = float(self.sandbox_allocation_weight)
        except (TypeError, ValueError) as exc:
            raise ValueError("sandbox_allocation_weight must be a positive finite float") from exc

        if not math.isfinite(weight) or weight <= 0:
            raise ValueError("sandbox_allocation_weight must be a positive finite float")
        self.sandbox_allocation_weight = weight

    def get_data_source(self, split: str | None = None) -> DataSource:
        """Get the data source for a specific split.

        Args:
            split: The split to use. If None, uses the config's default split.

        Returns:
            A DataSource configured for the specified split.

        Raises:
            ValueError: If no data source is configured.
        """
        from olmo_eval.data import DataSource

        if split is None:
            split = self.split.value

        if isinstance(self.data_source, str):
            return DataSource.from_uri(self.data_source, split=split)
        elif isinstance(self.data_source, DataSource):
            return self.data_source.with_split(split)
        raise ValueError("No data source configured for this task")

    def get_fewshot_source(self, split: str = "dev") -> DataSource | None:
        """Get the data source for few-shot examples.

        Args:
            split: The split to use for few-shot examples (default: "dev").

        Returns:
            A DataSource for few-shot examples, or None if not configured.
        """
        from olmo_eval.data import DataSource

        if self.fewshot_source is not None:
            if isinstance(self.fewshot_source, str):
                return DataSource.from_uri(self.fewshot_source, split=split)
            return self.fewshot_source.with_split(split)

        # Fall back to main data source with different split
        try:
            return self.get_data_source(split=split)
        except ValueError:
            return None

    #: Sampling fields added after task hashes were first stored. Each is emitted
    #: only when it differs from its default, so that tasks which do not use it
    #: keep the hash they had before the field existed.
    _SAMPLING_FIELDS_EMITTED_WHEN_SET: ClassVar[dict[str, Any]] = {
        "fit_max_tokens_to_context": False,
    }

    def _serialize_sampling_params(self) -> dict[str, Any] | None:
        from dataclasses import asdict

        if not self.sampling_params:
            return None
        serialized = asdict(self.sampling_params)
        for name, default in self._SAMPLING_FIELDS_EMITTED_WHEN_SET.items():
            if serialized.get(name) == default:
                serialized.pop(name, None)
        return serialized

    def to_dict(self) -> dict[str, Any]:
        """Serialize config to a dictionary for hashing and storage.

        Returns:
            Dictionary with all config values serialized.
        """

        def serialize_data_source(ds: Any) -> Any:
            if ds is None:
                return None
            if isinstance(ds, str):
                return ds
            return ds.to_dict()

        def serialize_primary_metric(pm: Any) -> Any:
            if pm is None:
                return None
            # MetricName is a str Enum
            if hasattr(pm, "value"):
                return pm.value
            # Metric instance
            if hasattr(pm, "to_dict"):
                return pm.to_dict()
            return str(pm)

        serialized = {
            "name": self.name,
            "data_source": serialize_data_source(self.data_source),
            "fewshot_source": serialize_data_source(self.fewshot_source),
            "formatter": self.formatter.to_dict() if self.formatter else None,
            "metrics": [m.to_dict() for m in self.metrics],
            "num_fewshot": self.num_fewshot,
            "fewshot_seed": self.fewshot_seed,
            "limit": self.limit,
            "seed": self.seed,
            "split": self.split.value,
            "primary_metric": serialize_primary_metric(self.get_primary_metric()),
            "sampling_params": self._serialize_sampling_params(),
            "output_score_aggregation": self.output_score_aggregation.value,
            "max_length": self.max_length,
            "answer_extractor": getattr(self.answer_extractor, "__name__", None),
            "dependencies": self.dependencies,
        }
        # Emitted only when set so that task hashes of runs without it are
        # unchanged from before the field existed.
        if self.prompt_templates is not None or self.system_prompt_style is not None:
            serialized["prompt_templates"] = self.prompt_templates
            serialized["system_prompt_style"] = self.system_prompt_style
        if self.strip_thinking:
            serialized["strip_thinking"] = True
        if any(
            value is not None
            for value in (
                self.judge_model,
                self.judge_reasoning_effort,
                self.judge_max_tokens,
            )
        ):
            serialized.update(
                {
                    "judge_model": self.judge_model,
                    "judge_reasoning_effort": self.judge_reasoning_effort,
                    "judge_max_tokens": self.judge_max_tokens,
                }
            )
        return serialized

    def get_primary_metric(self) -> Metric | None:
        """Get the effective primary metric for this task.

        Returns the explicitly set primary_metric if available, otherwise
        returns the single metric if exactly one is defined. Returns None
        if no metrics are defined or multiple metrics exist without an
        explicit primary.

        Returns:
            The primary Metric instance, or None.
        """
        if self.primary_metric is not None:
            # If it's a Metric instance, return it directly
            if isinstance(self.primary_metric, Metric):
                return self.primary_metric
            # If it's a MetricName enum, we can't resolve it to an instance here
            return None
        # Default to single metric if only one is defined
        if len(self.metrics) == 1:
            return self.metrics[0]
        return None


class Task(ABC):
    """Abstract base class for evaluation tasks.

    Tasks can either:
    1. Override `instances` property directly (legacy approach)
    2. Implement `process_doc()` and use the default `_load_instances()` helper

    The second approach allows tasks to benefit from the unified data loading
    infrastructure that supports HuggingFace, local files, S3, and GCS sources.

    Example using process_doc:
        >>> class MyTask(Task):
        ...     def process_doc(self, doc: dict) -> Instance:
        ...         return Instance(
        ...             question=doc["question"],
        ...             gold_answer=doc["answer"],
        ...         )
        ...
        ...     @property
        ...     def instances(self) -> Iterator[Instance]:
        ...         yield from self._load_instances()
    """

    def __init__(self, config: TaskConfig) -> None:
        self.config = config
        self._fewshot_cache: list[Instance] | None = None
        self._instances_cache: list[Instance] | None = None
        self._scorers_cache: dict[str, Scorer] | None = None
        self._has_async_cache: bool | None = None

    @property
    def request_type(self) -> RequestType:
        """The type of request this task produces."""
        if self.config.formatter is not None:
            return self.config.formatter.request_type
        return RequestType.COMPLETION

    def get_sampling_params(self, instance: Instance) -> SamplingParams | None:
        """Get sampling params for a specific instance.

        Override to provide instance-specific sampling params (e.g., per-instance stop sequences).
        Default returns the task-level sampling params.
        """
        return self.config.sampling_params

    @property
    @abstractmethod
    def instances(self) -> Iterator[Instance]:
        """Yield instances from the dataset.

        Subclasses must implement this. They can either:
        1. Implement custom loading logic directly
        2. Use the helper: `yield from self._load_instances()`
        """
        ...

    @abstractmethod
    def format_request(self, instance: Instance) -> LMRequest:
        """Format an instance into an LM request."""
        ...

    def extract_answer(self, output: LMOutput) -> Any:
        """Extract the answer from model output.

        If ``config.answer_extractor`` is set, it is called on ``output.text``.
        Subclasses can override this method directly for task-level defaults;
        variants can override via the config field.
        """
        if self.config.answer_extractor is not None:
            return self.config.answer_extractor(output.text)
        return output.text

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        """Convert a raw document to an Instance.

        Override this method to define how documents are converted to instances.
        Return None to skip the document.

        This is used by `_load_instances()` when using the unified data loader.

        Args:
            doc: A raw document dictionary from the dataset.
            index: The index of the document in the dataset.

        Returns:
            An Instance, or None to skip this document.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement process_doc() "
            "to use the unified data loading infrastructure"
        )

    def _load_instances(self, split: str | None = None) -> Iterator[Instance]:
        """Load and process instances from the configured data source.

        This helper method uses the unified DataLoader to fetch documents
        and calls `process_doc()` to convert them to instances.

        Subclasses can use this in their `instances` property:
            @property
            def instances(self) -> Iterator[Instance]:
                yield from self._load_instances()

        Args:
            split: Optional split override. If None, uses config.split.

        Yields:
            Instance objects from the dataset.
        """
        from olmo_eval.data import DataLoader

        loader = DataLoader()
        source = self.config.get_data_source(split=split)

        for index, doc in enumerate(loader.load(source)):
            instance = self.process_doc(doc, index)
            if instance is not None:
                yield instance

    def _load_instances_cached(self, split: str | None = None) -> Iterator[Instance]:
        """Load instances with caching.

        Same as `_load_instances()` but caches results after first call.

        Args:
            split: Optional split override.

        Yields:
            Instance objects from the dataset.
        """
        if self._instances_cache is None:
            self._instances_cache = list(self._load_instances(split=split))
        yield from self._instances_cache

    def get_fewshot(self) -> list[Instance]:
        """Get few-shot examples (cached after first call).

        Returns no examples when ``num_fewshot`` is 0, whatever pool the task
        would otherwise draw from.
        """
        if self._fewshot_cache is None:
            self._fewshot_cache = self._build_fewshot() if self.config.num_fewshot else []
        return self._fewshot_cache

    # Class attributes for fewshot configuration (can be overridden by subclasses)
    fewshot_split: str = "dev"
    fewshot_sample: bool = True

    def _build_fewshot(self) -> list[Instance]:
        """Build few-shot examples. Override fewshot_split/fewshot_sample for custom behavior."""
        return self._build_fewshot_from_source(
            split=self.fewshot_split,
            sample=self.fewshot_sample,
        )

    def _build_fewshot_from_source(
        self,
        split: str = "dev",
        sample: bool = True,
        fallback_splits: list[str] | None = None,
    ) -> list[Instance]:
        """Build few-shot examples using the unified data loader.

        Args:
            split: Primary split to use for few-shot examples.
            sample: If True, randomly sample num_fewshot examples. If False, return all.
            fallback_splits: Optional list of splits to try if primary split fails/empty.

        Returns:
            List of Instance objects for few-shot prompting.
        """
        import random

        from olmo_eval.data import DataLoader

        if sample and self.config.num_fewshot == 0:
            return []

        splits_to_try = [split] + (fallback_splits or [])
        all_instances: list[Instance] = []

        loader = DataLoader()
        for try_split in splits_to_try:
            try:
                source = self._get_source_for_split(try_split)
                all_instances = [
                    inst
                    for doc in loader.load(source)
                    if (inst := self.process_doc(doc)) is not None
                ]
                if all_instances:
                    break
            except Exception:
                continue

        if not all_instances:
            return []

        if sample and self.config.num_fewshot:
            rng = random.Random(self.config.fewshot_seed)
            return rng.sample(all_instances, min(self.config.num_fewshot, len(all_instances)))

        return all_instances

    def _get_source_for_split(self, split: str) -> DataSource:
        """Get data source for a specific split."""
        return self.config.get_data_source(split=split)

    async def score_responses(
        self,
        responses: Sequence[Response],
        context: ScoringContext | None = None,
    ) -> Sequence[Response]:
        """Apply all scorers to extract answers and compute scores.

        Args:
            responses: Responses to score.
            context: Optional scoring context with execution environment for
                scorers that need sandboxed execution.

        Returns:
            The scored responses.

        Subclasses needing custom answer extraction should override
        `_extract_answers()` rather than this method.
        """
        self._extract_answers(responses)

        # Check if any scorers need async execution
        has_async_scorers = self._has_async_scorers()

        if has_async_scorers and context is not None:
            # Run async scoring
            await self._apply_scorers_async(responses, context)
        else:
            # Run sync scoring
            self._apply_scorers(responses)

        return responses

    def _get_scorers(self) -> dict[str, Scorer]:
        """Return scorer instances, keyed by name. Cached after first call."""
        if self._scorers_cache is None:
            scorers_by_name: dict[str, Scorer] = {}
            for metric in self.config.metrics:
                if hasattr(metric, "scorer") and metric.scorer is not None:
                    scorer_instance = metric.scorer()
                    if scorer_instance.name not in scorers_by_name:
                        scorers_by_name[scorer_instance.name] = scorer_instance
            self._scorers_cache = scorers_by_name
        return self._scorers_cache

    def _has_async_scorers(self) -> bool:
        """Check if any configured scorers require async execution.

        This includes ExecutionScorer (sandboxed code), ProcessScorer
        (CPU-bound subprocess work), and ContextScorer (LLM judges).
        """
        if self._has_async_cache is None:
            from olmo_eval.common.scorers.execution import ContextScorer, ExecutionScorer

            self._has_async_cache = any(
                isinstance(s, (ExecutionScorer, ContextScorer))
                or getattr(s, "requires_async", False)
                for s in self._get_scorers().values()
            )
        return self._has_async_cache

    def strip_thinking_traces(self, responses: Sequence[Response]) -> None:
        """Drop ``<think>...</think>`` traces so scorers see only the final answer.

        No-op unless ``config.strip_thinking`` is set. Runners call this before
        ``score_responses`` (which tasks may override). Idempotent: a stripped
        output has no ``</think>`` left, so a second pass leaves it alone.

        Mirrors the reference harness's ``r1_style`` processing: everything
        through the *last* ``</think>`` goes, the text after it is kept
        byte-for-byte, and an unterminated trace is left as-is. Whitespace
        matters — the IFEval loose variants drop the response's first line, and
        paragraph checks index on blank-line splits — so nothing is trimmed.
        Only ``outputs[*].text`` is touched; trajectories and request traces
        keep the trace.
        """
        if not self.config.strip_thinking:
            return
        from olmo_eval.evals.extract import extract_think_answer

        for response in responses:
            for output in response.outputs:
                text = output.text or ""
                if "</think>" not in text:
                    continue
                output.metadata.setdefault("original_text", text)
                output.text = extract_think_answer(text) or ""

    def _extract_answers(self, responses: Sequence[Response]) -> None:
        """Extract answers from outputs. Override for complex multi-output logic."""
        for response in responses:
            for output in response.outputs:
                output.extracted_answer = self.extract_answer(output)

    def _apply_scorers(self, responses: Sequence[Response]) -> None:
        """Run all scorers synchronously and populate response.scores."""
        scorers_by_name = self._get_scorers()

        for response in responses:
            for scorer in scorers_by_name.values():
                scores = [scorer.score(response.instance, o) for o in response.outputs]
                for i, output in enumerate(response.outputs):
                    if output.metadata is None:
                        output.metadata = {}
                    output.metadata[f"score:{scorer.name}"] = scores[i] if i < len(scores) else 0.0
                response.scores[scorer.name] = self._aggregate_output_scores(
                    dict(enumerate(scores))
                )

    async def _apply_scorers_async(
        self,
        responses: Sequence[Response],
        context: ScoringContext,
    ) -> None:
        """Run all scorers with context and populate response.scores.

        ExecutionScorer subclasses are scored via ascore() with the execution
        environment. ProcessScorer subclasses are dispatched through the
        process pool manager in the scoring context. ContextScorer subclasses
        (e.g., LLM judges) are scored via ascore_with_context() with the full
        scoring context. Regular scorers use score() synchronously.

        Args:
            responses: Responses to score.
            context: Scoring context with execution environment and concurrency settings.

        Raises:
            SandboxRequiredError: If an ExecutionScorer is used without a valid
                execution environment in the context.
        """
        from olmo_eval.common.scorers.base import ProcessScorer
        from olmo_eval.common.scorers.execution import (
            ContextScorer,
            ExecutionScorer,
            SandboxRequiredError,
        )

        execution_env = context.execution_env if context.has_execution_env else None
        process_pool_manager = (
            context.process_pool_manager if context.has_process_pool_manager else None
        )
        scorers_by_name = self._get_scorers()

        # Route to task-specific sandbox if sandbox_env is configured
        sandbox_cap = self.config.sandbox_env.capability if self.config.sandbox_env else None
        if execution_env is not None and hasattr(execution_env, "for_capabilities"):
            from olmo_eval.harness.sandbox.config import Capability

            task_executor = execution_env.for_capabilities(  # ty: ignore[call-non-callable]
                sandbox_cap or Capability.DEFAULT
            )
        elif sandbox_cap and execution_env is not None and hasattr(execution_env, "get_executor"):
            task_executor = execution_env.get_executor(sandbox_cap)  # ty: ignore[call-non-callable]
        else:
            task_executor = execution_env

        # Separate scorers by type: execution (sandbox), process (CPU), context (LLM judge), sync
        execution_scorers: dict[str, ExecutionScorer] = {}
        process_scorers: dict[str, ProcessScorer] = {}
        context_scorers: dict[str, ContextScorer] = {}
        sync_scorers: dict[str, Scorer] = {}
        for name, scorer in scorers_by_name.items():
            if isinstance(scorer, ExecutionScorer):
                if task_executor is None:
                    raise SandboxRequiredError(
                        f"{scorer.__class__.__name__} requires a sandbox. "
                        "Configure sandboxes in HarnessConfig."
                    )
                execution_scorers[name] = scorer
            elif isinstance(scorer, ProcessScorer):
                if process_pool_manager is None:
                    raise RuntimeError(
                        f"{scorer.__class__.__name__} requires process scoring pools. "
                        "Configure scoring process pools in HarnessConfig."
                    )
                process_scorers[name] = scorer
            elif isinstance(scorer, ContextScorer):
                context_scorers[name] = scorer
            else:
                sync_scorers[name] = scorer

        # Apply sync scorers first (fast, no concurrency needed)
        for response in responses:
            for scorer in sync_scorers.values():
                scores = [scorer.score(response.instance, o) for o in response.outputs]
                # Store individual scores in output metadata for pass@k expansion
                for i, output in enumerate(response.outputs):
                    _store_output_score(
                        output,
                        scorer_name=scorer.name,
                        score=scores[i] if i < len(scores) else 0.0,
                    )
                response.scores[scorer.name] = self._aggregate_output_scores(
                    dict(enumerate(scores))
                )

        # Apply async scorers (execution, process, and context) concurrently
        async_scorers_exist = (
            bool(execution_scorers) or bool(process_scorers) or bool(context_scorers)
        )
        if async_scorers_exist:
            # Shared semaphore for sandbox execution — scoped by capability, sized
            # to max_concurrency * running_instances. Prevents overloading the pool.
            exec_semaphore: asyncio.Semaphore | contextlib.nullcontext[None] = (
                contextlib.nullcontext()
            )
            if execution_env is not None and hasattr(execution_env, "get_execution_semaphore"):
                from olmo_eval.harness.sandbox.config import Capability

                sem = execution_env.get_execution_semaphore(sandbox_cap or Capability.DEFAULT)  # ty: ignore[call-non-callable]
                if sem is not None:
                    exec_semaphore = sem

            # Per-call semaphore for context scorers (LLM judges) — they don't
            # consume sandbox slots so per-call throttling is appropriate.
            ctx_semaphore = asyncio.Semaphore(context.scoring_concurrency)

            async def score_execution(
                resp_idx: int, scorer: ExecutionScorer, out_idx: int
            ) -> OutputScore:
                response = responses[resp_idx]
                output = response.outputs[out_idx]
                try:
                    async with exec_semaphore:
                        assert task_executor is not None
                        score = await scorer.ascore(response.instance, output, task_executor)
                except Exception as exc:
                    error = _format_scoring_error(exc, phase="execution")
                    instance_id = response.instance.metadata.get("id", str(resp_idx))
                    logger.warning(
                        "Async scoring failed for instance %s output %s with scorer %s: %s",
                        instance_id,
                        out_idx,
                        scorer.name,
                        error.get("message", error["type"]),
                    )
                    return OutputScore(resp_idx, scorer.name, out_idx, 0.0, error)
                return OutputScore(resp_idx, scorer.name, out_idx, score, None)

            async def score_context(
                resp_idx: int, scorer: ContextScorer, out_idx: int
            ) -> OutputScore:
                response = responses[resp_idx]
                output = response.outputs[out_idx]
                try:
                    async with ctx_semaphore:
                        score = await scorer.ascore_with_context(response.instance, output, context)
                except Exception as exc:
                    error = _format_scoring_error(exc, phase="context")
                    instance_id = response.instance.metadata.get("id", str(resp_idx))
                    logger.warning(
                        "Context scoring failed for instance %s output %s with scorer %s: %s",
                        instance_id,
                        out_idx,
                        scorer.name,
                        error.get("message", error["type"]),
                    )
                    return OutputScore(resp_idx, scorer.name, out_idx, 0.0, error)
                return OutputScore(resp_idx, scorer.name, out_idx, score, None)

            async def score_process(
                resp_idx: int,
                scorer: ProcessScorer,
            ) -> ResponseScore:
                response = responses[resp_idx]
                assert process_pool_manager is not None
                results = await process_pool_manager.score_outputs(
                    scorer,
                    response.instance,
                    list(response.outputs),
                )
                output_scores = {out_idx: result.score for out_idx, result in enumerate(results)}
                output_errors = {
                    out_idx: result.error
                    for out_idx, result in enumerate(results)
                    if result.error is not None
                }
                return ResponseScore(resp_idx, scorer.name, output_scores, output_errors)

            tasks = []
            for resp_idx, response in enumerate(responses):
                # Add execution scorer tasks
                for scorer in execution_scorers.values():
                    for out_idx in range(len(response.outputs)):
                        tasks.append(score_execution(resp_idx, scorer, out_idx))
                # Add context scorer tasks
                for scorer in context_scorers.values():
                    for out_idx in range(len(response.outputs)):
                        tasks.append(score_context(resp_idx, scorer, out_idx))
                # Add process scorer tasks (one job per response/scorer)
                for scorer in process_scorers.values():
                    tasks.append(score_process(resp_idx, scorer))

            # Run all scoring tasks concurrently
            results = await asyncio.gather(*tasks)

            # Store individual scores in output metadata for pass@k expansion
            # Structure: {resp_idx: {scorer_name: {out_idx: score}}}
            scores_by_response: dict[int, dict[str, dict[int, float]]] = {}
            scoring_errors_by_response: dict[int, dict[int, dict[str, dict[str, str]]]] = {}
            for result in results:
                if isinstance(result, OutputScore):
                    resp_idx, scorer_name, out_idx, score, scoring_error = result
                    if resp_idx not in scores_by_response:
                        scores_by_response[resp_idx] = {}
                    if scorer_name not in scores_by_response[resp_idx]:
                        scores_by_response[resp_idx][scorer_name] = {}
                    scores_by_response[resp_idx][scorer_name][out_idx] = score
                    if scoring_error is not None:
                        if resp_idx not in scoring_errors_by_response:
                            scoring_errors_by_response[resp_idx] = {}
                        if out_idx not in scoring_errors_by_response[resp_idx]:
                            scoring_errors_by_response[resp_idx][out_idx] = {}
                        scoring_errors_by_response[resp_idx][out_idx][scorer_name] = scoring_error
                else:
                    resp_idx, scorer_name, output_scores, output_errors = result
                    if resp_idx not in scores_by_response:
                        scores_by_response[resp_idx] = {}
                    scores_by_response[resp_idx][scorer_name] = dict(output_scores)
                    if output_errors:
                        if resp_idx not in scoring_errors_by_response:
                            scoring_errors_by_response[resp_idx] = {}
                        for out_idx, scoring_error in output_errors.items():
                            if out_idx not in scoring_errors_by_response[resp_idx]:
                                scoring_errors_by_response[resp_idx][out_idx] = {}
                            scoring_errors_by_response[resp_idx][out_idx][scorer_name] = (
                                scoring_error
                            )

            # Store individual scores in output metadata and aggregate to response
            for resp_idx, scorer_scores in scores_by_response.items():
                response = responses[resp_idx]
                for scorer_name, output_scores in scorer_scores.items():
                    # Store per-output scores in metadata
                    for out_idx, score in output_scores.items():
                        output = response.outputs[out_idx]
                        scoring_error = None
                        if (
                            resp_idx in scoring_errors_by_response
                            and out_idx in scoring_errors_by_response[resp_idx]
                            and scorer_name in scoring_errors_by_response[resp_idx][out_idx]
                        ):
                            scoring_error = scoring_errors_by_response[resp_idx][out_idx][
                                scorer_name
                            ]
                        _store_output_score(
                            output,
                            scorer_name=scorer_name,
                            score=score,
                            scoring_error=scoring_error,
                        )
                    response.scores[scorer_name] = self._aggregate_output_scores(output_scores)

    def _aggregate_output_scores(self, scores: dict[int, float]) -> float:
        """Collapse per-output scorer values into one response-level score."""
        if not scores:
            return 0.0
        ordered_scores = [scores[index] for index in sorted(scores)]

        aggregation = self.config.output_score_aggregation
        if aggregation == OutputScoreAggregation.MAX:
            return max(ordered_scores)
        if aggregation == OutputScoreAggregation.FIRST:
            return ordered_scores[0]
        if aggregation == OutputScoreAggregation.MEAN:
            return sum(ordered_scores) / len(ordered_scores)
        raise ValueError(f"Unsupported output_score_aggregation: {aggregation}")

    def _expand_multi_output_responses(self, responses: Sequence[Response]) -> list[Response]:
        """Expand multi-output responses into individual responses for pass@k.

        When num_samples > 1, each Response has multiple outputs but scores are
        aggregated to max. For pass@k computation, we need N separate Response
        objects to count individual passing samples. This method expands each
        multi-output Response into N single-output Responses, preserving
        individual scores stored in output.metadata during scoring.

        Args:
            responses: Responses with potentially multiple outputs each.

        Returns:
            Expanded list where each Response has exactly one output.
        """
        expanded: list[Response] = []
        for response in responses:
            if len(response.outputs) <= 1:
                expanded.append(response)
            else:
                for output in response.outputs:
                    new_response = Response(
                        instance=response.instance,
                        request=response.request,
                        outputs=[output],
                        trajectory=response.trajectory,
                    )
                    # Copy scores from output metadata
                    for key, value in (output.metadata or {}).items():
                        if key.startswith("score:"):
                            scorer_name = key[6:]
                            new_response.scores[scorer_name] = value
                    expanded.append(new_response)
        return expanded

    def compute_metrics(self, responses: Sequence[Response]) -> dict[str, dict[str, float]]:
        """Compute all metrics from scored responses.

        Returns metrics in nested structure: {metric_name: {scorer_name: score}}.
        This allows multiple scorers to produce the same metric (e.g., accuracy)
        while preserving which scorer produced which value.
        """
        from olmo_eval.common.metrics import PassAtKMetric, PassPowKMetric

        # Expand multi-output responses for pass@k/pass^k metrics only
        expanded_responses: Sequence[Response] | None = None

        result: dict[str, dict[str, float]] = {}
        for metric in self.config.metrics:
            scorer_name = (
                metric.scorer().name if hasattr(metric, "scorer") and metric.scorer else "default"
            )
            if metric.name not in result:
                result[metric.name] = {}

            # Use expanded responses for pass metrics, original for others
            if isinstance(metric, (PassAtKMetric, PassPowKMetric)):
                if expanded_responses is None:
                    expanded_responses = self._expand_multi_output_responses(responses)
                result[metric.name][scorer_name] = metric.compute(expanded_responses)
            else:
                result[metric.name][scorer_name] = metric.compute(responses)
        return result
