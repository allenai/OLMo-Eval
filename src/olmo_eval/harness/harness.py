"""Harness: runtime orchestration layer for an evaluation run.

The Harness combines the primary inference provider with runtime behavior such as
tools, system prompts, auxiliary providers, metrics collection, and optional
scaffolds.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from olmo_eval.common.types import LMOutput, LMRequest, SamplingParams
from olmo_eval.inference.base import require_tool_support

from .config import HarnessConfig
from .result import HarnessResult
from .scaffolds import Scaffold, get_scaffold

if TYPE_CHECKING:
    from olmo_eval.inference.base import InferenceProvider
    from olmo_eval.inference.metrics.core.schema import BatchMetrics


class Harness:
    """Runtime orchestration layer for an evaluation run.

    The Harness combines the primary inference provider with execution policy
    such as system prompts, tools, auxiliary providers, sandboxing, metrics
    collection, and an optional scaffold for multi-turn control. This lets the
    same task run in plain, tool-using, or scaffolded modes without changing
    the task definition.

    For multi-turn execution with run(), a scaffold must be configured.
    """

    def __init__(self, config: HarnessConfig) -> None:
        """Initialize the Harness.

        Args:
            config: Configuration specifying provider, tools, system prompt, etc.
                The provider is created from config.provider.
        """
        self.config = config
        self._provider: InferenceProvider | None = None
        self._scaffold: Scaffold | None = None

    @property
    def provider(self) -> InferenceProvider:
        """Get or create the inference provider.

        The provider is lazily created from config.provider on first access.
        If metrics configuration is present and enabled, the provider is wrapped
        with instrumentation for metrics collection.
        """
        if self._provider is None:
            provider = self.config.provider.create_provider()
            if self.config.metrics is not None and self.config.metrics.enabled:
                from olmo_eval.inference.metrics import InstrumentedProvider

                instrumented = InstrumentedProvider(provider)
                if self.config.metrics.collect_gpu:
                    instrumented.enable_gpu_monitoring(interval_s=1.0)
                self._provider = instrumented  # type: ignore[ty:invalid-assignment]
            else:
                self._provider = provider
        return self._provider  # type: ignore[ty:invalid-return-type]

    @property
    def scaffold(self) -> Scaffold:
        """Get or create the scaffold.

        The scaffold is lazily created from config.scaffold on first access.

        Raises:
            RuntimeError: If no scaffold is configured.
        """
        if self._scaffold is None:
            if not self.config.scaffold:
                raise RuntimeError(
                    "No scaffold configured. Set config.scaffold to use run(). "
                    "For single-turn generation, use generate() or agenerate() instead."
                )
            self._scaffold = get_scaffold(self.config.scaffold)
        return self._scaffold

    @property
    def model_name(self) -> str:
        """Get the model name from the provider.

        Returns:
            Model name string.
        """
        return self.provider.model_name

    # ─────────────────────────────────────────────────────────
    # Single-turn interface (same as Provider, but with config)
    # ─────────────────────────────────────────────────────────

    def _check_images_supported(self, requests: list[LMRequest]) -> None:
        if getattr(self.provider, "supports_images", False):
            return
        n = sum(1 for r in requests if r.images)
        if n:
            raise ValueError(
                f"{type(self.provider).__name__} does not support image requests, but "
                f"{n} of {len(requests)} requests carry images. Run this task with a "
                "multimodal provider (e.g. provider.kind=olmo_core_vlm or an hf "
                "multimodal preset)."
            )

    def generate(
        self,
        requests: list[LMRequest],
        sampling_params: SamplingParams | None = None,
    ) -> list[list[LMOutput]]:
        """Single-turn generation with config injected.

        Applies the harness configuration (tools, system prompt) to each request
        and passes them to the provider.

        Args:
            requests: List of requests to process.
            sampling_params: Optional sampling parameters.

        Returns:
            List of output lists, one per request.
        """
        transformed = [self._apply_config(r) for r in requests]
        self._check_images_supported(transformed)
        return self.provider.generate(transformed, sampling_params)

    def logprobs(
        self,
        requests: list[LMRequest],
        sampling_params: SamplingParams | None = None,
    ) -> list[list[LMOutput]]:
        """Log probability computation.

        Args:
            requests: List of requests with continuations to score.

        Returns:
            List of output lists with logprobs populated.
        """
        transformed = [self._apply_config(r) for r in requests]
        self._check_images_supported(transformed)
        return self.provider.logprobs(transformed, sampling_params)

    async def agenerate(
        self,
        requests: list[LMRequest],
        sampling_params: SamplingParams | None = None,
    ) -> list[list[LMOutput]]:
        """Async single-turn generation with config injected."""
        transformed = [self._apply_config(r) for r in requests]
        self._check_images_supported(transformed)
        return await self.provider.agenerate(transformed, sampling_params)

    async def alogprobs(
        self,
        requests: list[LMRequest],
        sampling_params: SamplingParams | None = None,
    ) -> list[list[LMOutput]]:
        """Async log probability computation."""
        transformed = [self._apply_config(r) for r in requests]
        self._check_images_supported(transformed)
        return await self.provider.alogprobs(transformed, sampling_params)

    # ─────────────────────────────────────────────────────────
    # Multi-turn interface (delegates to scaffold)
    # ─────────────────────────────────────────────────────────

    async def run(
        self,
        request: LMRequest,
        sampling_params: SamplingParams | None = None,
        trace_metadata: dict[str, Any] | None = None,
    ) -> HarnessResult:
        """Multi-turn execution via configured scaffold.

        Runs an agent loop that:
        1. Sends the request to the model
        2. If the response has tool calls, executes them
        3. Appends results and continues until done or max_turns

        Args:
            request: Initial request to start the conversation.
            sampling_params: Optional sampling parameters.
            trace_metadata: Optional metadata for tracing (e.g., instance_id, task_id).

        Returns:
            HarnessResult with trajectory and final output.

        Raises:
            RuntimeError: If no scaffold is configured.
        """
        return await self.scaffold.run(
            self.provider,
            self.config,
            request,
            sampling_params,
            trace_metadata,
            **self.config.scaffold_kwargs,
        )

    async def cleanup(self) -> None:
        """Clean up resources held by the harness and its scaffold."""
        self.shutdown_reporters()
        if self._scaffold is not None:
            await self._scaffold.cleanup()

    def flush_metrics(self, batch_hash: str, clear: bool = True) -> BatchMetrics | None:
        """Flush collected metrics to configured reporters.

        Call this after each batch to write metrics incrementally.
        Only has effect if metrics is enabled and the provider is instrumented.

        Args:
            batch_hash: Batch hash computed from native instance IDs.
            clear: If True, clear collected metrics after reporting.

        Returns:
            The reported batch with its per-request metrics attached (in the order
            the provider recorded them), or None when nothing was reported.
        """
        if self.config.metrics is None or not self.config.metrics.enabled:
            return None

        # Check if provider is instrumented
        if self._provider is None:
            return None

        from olmo_eval.inference.metrics import InstrumentedProvider

        if not isinstance(self._provider, InstrumentedProvider):
            return None

        metrics = self._provider.get_metrics()
        if not metrics:
            return None

        # Initialize reporters and report
        from olmo_eval.inference.metrics.core.stats import compute_batch_metrics

        # Get GPU snapshots collected during inference
        gpu_snapshots = self._provider.get_gpu_snapshots()

        batch = compute_batch_metrics(
            metrics,
            wall_clock_s=0.0,
            batch_hash=batch_hash,
            config=self.config.metrics,
            gpu_snapshots=gpu_snapshots,
        )

        # Get or create cached reporters (reuse connections across batches)
        reporters = self._get_reporters()

        for reporter in reporters:
            try:
                reporter.report_batch(batch)
                reporter.flush()
            except Exception as e:
                import logging

                logging.getLogger(__name__).warning(f"Failed to report metrics: {e}")

        # Clear metrics after reporting to avoid double-counting
        if clear:
            self._provider.clear_metrics()

        from dataclasses import replace

        return replace(batch, requests=tuple(metrics))

    def initialize_reporters(self) -> None:
        """Initialize metrics reporters eagerly.

        Call this at job start to surface reporter setup errors early rather than
        waiting until the first batch is processed. This is optional - reporters
        will be lazily initialized on first use if not called.
        """
        self._get_reporters()

    def _get_reporters(self) -> list[Any]:
        """Get or create cached metrics reporters."""
        if not hasattr(self, "_reporters"):
            from olmo_eval.inference.metrics.core.registry import reporter_registry

            self._reporters: list[Any] = []
            if self.config.metrics is not None:
                for reporter_config in self.config.metrics.reporters:
                    resolved = self._resolve_reporter_config(reporter_config)
                    if resolved is not None:
                        reporter = reporter_registry.create(resolved)
                        # Initialize reporters that support eager connection
                        init_fn = getattr(reporter, "initialize", None)
                        if callable(init_fn):
                            init_fn()
                        self._reporters.append(reporter)
        return self._reporters

    def shutdown_reporters(self) -> None:
        """Shutdown cached metrics reporters."""
        import contextlib

        if hasattr(self, "_reporters"):
            for reporter in self._reporters:
                with contextlib.suppress(Exception):
                    reporter.shutdown()
            self._reporters = []

    def _resolve_reporter_config(
        self, reporter_config: str | dict[str, Any]
    ) -> str | dict[str, Any] | None:
        """Resolve reporter config, adding path for file reporter if needed."""
        if isinstance(reporter_config, str):
            name = reporter_config
            config_dict: dict[str, Any] = {}
        else:
            name = reporter_config.get("name", "console")
            config_dict = dict(reporter_config)

        # For file reporter, resolve path from metrics config if not set
        if name == "file" and "path" not in config_dict:
            path = self.config.metrics.get_metrics_path() if self.config.metrics else None
            if path is None:
                import logging

                logging.getLogger(__name__).warning(
                    "File reporter requires output_dir in MetricsConfig. Skipping."
                )
                return None
            config_dict["name"] = "file"
            config_dict["path"] = path
            return config_dict

        return reporter_config

    # ─────────────────────────────────────────────────────────
    # Config application (used by scaffolds)
    # ─────────────────────────────────────────────────────────

    def _apply_config(self, request: LMRequest) -> LMRequest:
        """Inject tool schemas and system prompt from config.

        This transforms a request by adding:
        - Tool schemas (if config has tools)
        - System prompt (if configured and not already present)
        - Formatted prompt with chat template (if a chat request)

        Args:
            request: Original request.

        Returns:
            New request with config applied.
        """

        messages = self._inject_system_prompt(request.messages)
        tools = self.config.tool_schemas if self.config.has_tools else request.tools
        # A scaffold sends its tools itself; only a request the provider sends
        # directly depends on the provider honouring its tool schemas.
        if request.tools and not self.config.scaffold:
            require_tool_support(self.provider)

        return LMRequest(
            request_type=request.request_type,
            messages=messages,
            prompt=request.prompt,
            continuations=request.continuations,
            continuation_prompts=request.continuation_prompts,
            tools=tools,
            system_prompt=self.config.system_prompt or request.system_prompt,
            max_length=request.max_length,
            images=request.images,
            metadata=request.metadata,
        )

    def _inject_system_prompt(
        self, messages: tuple[dict[str, Any], ...]
    ) -> tuple[dict[str, Any], ...]:
        """Add system prompt to messages if configured and not present.

        Args:
            messages: Original message tuple.

        Returns:
            Messages with system prompt prepended if needed.
        """
        if not self.config.system_prompt:
            return messages

        # Check if messages already start with a system message
        if messages and messages[0].get("role") == "system":
            return messages

        # Prepend system message
        system_msg: dict[str, Any] = {
            "role": "system",
            "content": self.config.system_prompt,
        }
        return (system_msg,) + messages


def create_harness(
    config: HarnessConfig | dict[str, Any],
) -> Harness:
    """Create a Harness from config.

    Convenience function that handles config creation/parsing.

    Args:
        config: HarnessConfig instance or dict.

    Returns:
        Configured Harness instance.
    """
    if isinstance(config, dict):
        return Harness(HarnessConfig.from_dict(cast(dict[str, Any], config)))
    return Harness(config)
