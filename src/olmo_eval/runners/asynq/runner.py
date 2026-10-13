"""Async evaluation runner with instance-level queuing."""

from __future__ import annotations

import asyncio
import contextlib
import multiprocessing as mp
import random
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from rich.console import Console

from olmo_eval.common.configs import expand_tasks
from olmo_eval.common.constants.infrastructure import BEAKER_RESULT_DIR
from olmo_eval.common.logging import configure_worker_logging, get_logger
from olmo_eval.harness.config import HarnessConfig, ProviderConfig
from olmo_eval.runners.asynq.monitoring import (
    terminate_workers,
    wait_for_init_times,
    wait_for_workers_ready,
)
from olmo_eval.runners.asynq.preparation import prepare_task_items
from olmo_eval.runners.asynq.results import (
    aggregate_results,
    check_hard_failure_gate,
    process_results,
    utc_iso,
)
from olmo_eval.runners.asynq.types import QueueItem, TaskTracker
from olmo_eval.runners.common.base import BaseEvalRunner
from olmo_eval.runners.common.mixins import RunnerResultsMixin
from olmo_eval.runners.common.types import DEFAULT_MAX_HARD_FAILURE_RATE, TaskResult
from olmo_eval.runners.processing.utils import generate_experiment_id

if TYPE_CHECKING:
    from olmo_eval.common.execution import ProcessScoringPoolConfig
    from olmo_eval.evals.tasks.common import SandboxEnv, Task
    from olmo_eval.harness.sandbox import SandboxConfig
    from olmo_eval.upload import UploadConfig

logger = get_logger(__name__)
runner_logger = configure_worker_logging("runner")
console = Console(force_terminal=True, width=120)

_DEFAULT_SANDBOX_ENV = "__default__"
_DEFAULT_PROCESS_POOL = "cpu"


@dataclass(frozen=True)
class SandboxDemand:
    """Execution-aware sandbox demand summary for allocation and logging."""

    sandbox_envs: list[SandboxEnv]
    env_sandbox_items: dict[str, int]
    env_work_units: dict[str, float]
    env_task_count: dict[str, int]
    needs_default: bool


@dataclass(frozen=True)
class SandboxPlan:
    """Resolved sandbox configs and allocation data for relevant sandbox envs."""

    sandboxes: list[SandboxConfig]
    env_sandbox_items: dict[str, int]
    env_work_units: dict[str, float]
    env_task_count: dict[str, int]
    allocated: dict[str, int]
    min_required: dict[str, int]
    budget: int
    needs_default: bool


@dataclass(frozen=True)
class ProcessPoolDemand:
    """Observed process-scoring workload for a named process pool."""

    task_count: int
    scorer_count: int
    response_jobs: int


@dataclass(frozen=True)
class ProcessPoolPlan:
    """Resolved process pools and demand data for process-backed scorers."""

    pools: dict[str, ProcessScoringPoolConfig]
    demand: dict[str, ProcessPoolDemand]
    auto_created: frozenset[str]


def _processing_seconds(
    results: Mapping[str, TaskResult], processing_start: float, fallback_end: float
) -> float:
    """Seconds from processing start until the last task finished.

    Uses the latest task last_completed_at, or fallback_end when no task has one.
    """
    from datetime import datetime

    ends = [
        datetime.fromisoformat(r.last_completed_at).timestamp()
        for r in results.values()
        if r.last_completed_at
    ]
    return max([*ends, processing_start] if ends else [fallback_end]) - processing_start


def _materialize_sandbox_instances(sandboxes: Sequence[SandboxConfig]) -> list[SandboxConfig]:
    """Replace auto-managed sandbox instance counts with their runtime defaults."""
    return [
        replace(cfg, instances=cfg.resolved_instances) if cfg.instances is None else cfg
        for cfg in sandboxes
    ]


def _count_execution_scorers(task: Task) -> int:
    """Count unique execution scorers after task-level deduplication."""
    from olmo_eval.common.scorers.execution import ExecutionScorer

    return sum(1 for scorer in task._get_scorers().values() if isinstance(scorer, ExecutionScorer))


def _count_process_scorers_by_pool(task: Task) -> dict[str, int]:
    """Count unique process scorers per pool after task-level deduplication."""
    from olmo_eval.common.execution import serialize_process_scorer
    from olmo_eval.common.scorers import ProcessScorer

    counts: dict[str, int] = {}
    for scorer in task._get_scorers().values():
        if isinstance(scorer, ProcessScorer):
            spec = serialize_process_scorer(scorer)
            pool_name = spec.pool_name
            counts[pool_name] = counts.get(pool_name, 0) + 1
    return counts


def _collect_process_pool_demand(
    trackers: Mapping[str, TaskTracker],
    expanded_tasks: Sequence[str] | None = None,
) -> dict[str, ProcessPoolDemand]:
    """Collect per-pool demand for process-backed scorers."""
    from olmo_eval.common.execution import ProcessScoringConfigError

    relevant_specs = set(expanded_tasks) if expanded_tasks else None
    task_counts: dict[str, int] = {}
    scorer_counts: dict[str, int] = {}
    response_jobs: dict[str, int] = {}

    for spec, tracker in trackers.items():
        if relevant_specs is not None and spec not in relevant_specs:
            continue
        if tracker.task is None:
            continue

        try:
            pool_counts = _count_process_scorers_by_pool(tracker.task)
        except ProcessScoringConfigError as exc:
            raise ValueError(
                f"Task '{spec}' has invalid process scorer configuration: {exc}"
            ) from exc
        if not pool_counts:
            continue

        for pool_name, scorer_count in pool_counts.items():
            task_counts[pool_name] = task_counts.get(pool_name, 0) + 1
            scorer_counts[pool_name] = scorer_counts.get(pool_name, 0) + scorer_count
            response_jobs[pool_name] = response_jobs.get(pool_name, 0) + (
                tracker.total_instances * scorer_count
            )

    return {
        pool_name: ProcessPoolDemand(
            task_count=task_counts[pool_name],
            scorer_count=scorer_counts[pool_name],
            response_jobs=response_jobs[pool_name],
        )
        for pool_name in task_counts
    }


def _plan_process_scoring_pools(
    configured_pools: Mapping[str, ProcessScoringPoolConfig],
    expanded_tasks: Sequence[str],
    trackers: Mapping[str, TaskTracker],
    scoring_concurrency: int | None,
) -> ProcessPoolPlan | None:
    """Resolve the process pools needed by the selected tasks."""
    from olmo_eval.common.execution import ProcessScoringPoolConfig, default_process_pool_workers

    demand = _collect_process_pool_demand(trackers, expanded_tasks)
    if not demand:
        return None

    referenced = set(demand)
    explicit = dict(configured_pools)
    auto_created: set[str] = set()

    if not explicit:
        non_default = sorted(name for name in referenced if name != _DEFAULT_PROCESS_POOL)
        if non_default:
            joined = ", ".join(non_default)
            raise ValueError(
                "Process scoring pool(s) "
                f"{joined} are required by the selected tasks but no "
                "scoring_process_pools were configured. Only the default "
                f"'{_DEFAULT_PROCESS_POOL}' pool can be auto-created."
            )
        workers = scoring_concurrency or default_process_pool_workers()
        explicit[_DEFAULT_PROCESS_POOL] = ProcessScoringPoolConfig(workers=workers)
        auto_created.add(_DEFAULT_PROCESS_POOL)

    missing = sorted(name for name in referenced if name not in explicit)
    if missing:
        joined = ", ".join(missing)
        raise ValueError(
            "Missing process scoring pool configuration for: "
            f"{joined}. Add these under HarnessConfig.scoring_process_pools."
        )

    pools = {name: explicit[name] for name in sorted(referenced)}
    return ProcessPoolPlan(
        pools=pools,
        demand=demand,
        auto_created=frozenset(auto_created),
    )


def _shutdown_process_scoring_pools(process_pool_manager: Any) -> None:
    runner_logger.info("Stopping process scoring pools...")
    start_time = time.time()
    try:
        process_pool_manager.shutdown()
    except Exception:
        runner_logger.warning("Failed to stop process scoring pools", exc_info=True)
    else:
        runner_logger.info(f"Stopped process scoring pools in {time.time() - start_time:.1f}s")


def _collect_sandbox_demand(
    trackers: Mapping[str, TaskTracker],
    expanded_tasks: Sequence[str] | None = None,
) -> SandboxDemand:
    """Collect execution-aware sandbox demand and task counts per environment."""
    env_sandbox_items: dict[str, int] = {}
    env_work_units: dict[str, float] = {}
    env_task_count: dict[str, int] = {}
    named_envs: dict[str, SandboxEnv] = {}
    needs_default = False
    relevant_specs = set(expanded_tasks) if expanded_tasks else None

    for spec, tracker in trackers.items():
        if relevant_specs is not None and spec not in relevant_specs:
            continue
        if tracker.task is None:
            continue
        execution_scorer_count = _count_execution_scorers(tracker.task)
        if execution_scorer_count <= 0:
            continue

        tcfg = tracker.task.config
        num_samples = tcfg.sampling_params.num_samples if tcfg.sampling_params else 1
        sandbox_items = tracker.total_instances * num_samples * execution_scorer_count

        if tcfg.sandbox_env is None:
            env_key = _DEFAULT_SANDBOX_ENV
            needs_default = True
        else:
            senv = tcfg.sandbox_env
            if senv.name in named_envs and named_envs[senv.name] != senv:
                raise ValueError(
                    f"Conflicting sandbox_env for '{senv.name}': "
                    f"{named_envs[senv.name].dependencies} vs {senv.dependencies}"
                )
            named_envs[senv.name] = senv
            env_key = senv.name

        env_sandbox_items[env_key] = env_sandbox_items.get(env_key, 0) + sandbox_items
        env_work_units[env_key] = env_work_units.get(env_key, 0.0) + (
            sandbox_items * tcfg.sandbox_allocation_weight
        )
        env_task_count[env_key] = env_task_count.get(env_key, 0) + 1

    return SandboxDemand(
        sandbox_envs=list(named_envs.values()),
        env_sandbox_items=env_sandbox_items,
        env_work_units=env_work_units,
        env_task_count=env_task_count,
        needs_default=needs_default,
    )


def _format_work_units(work_units: float) -> str:
    """Format weighted sandbox work units for logs and tables."""
    if float(work_units).is_integer():
        return str(int(work_units))
    return f"{work_units:.1f}"


def _allocate_auto_sandbox_instances(
    auto_env_keys: Sequence[str],
    env_work_units: Mapping[str, float],
    sandbox_pool_instances: int | None,
) -> dict[str, int]:
    """Allocate the shared sandbox pool across auto-managed environments."""
    if not auto_env_keys:
        return {}
    if sandbox_pool_instances is None:
        return {env_key: 1 for env_key in auto_env_keys}

    budget = sandbox_pool_instances
    min_budget = len(auto_env_keys)
    if budget < min_budget:
        logger.warning(
            f"Sandbox pool budget ({budget}) is less than the number of "
            f"auto-allocated environments ({min_budget}); clamping to {min_budget}"
        )
        budget = min_budget

    total_demand = sum(env_work_units.get(env_key, 0.0) for env_key in auto_env_keys)
    if total_demand <= 0:
        base = budget // len(auto_env_keys)
        remainder = budget % len(auto_env_keys)
        allocated: dict[str, int] = {}
        for i, env_key in enumerate(auto_env_keys):
            allocated[env_key] = base + (1 if i < remainder else 0)
        return allocated

    distributable = budget - len(auto_env_keys)
    allocated = {}
    for env_key in auto_env_keys:
        demand = env_work_units.get(env_key, 0.0)
        extra = max(0, round(distributable * demand / total_demand))
        allocated[env_key] = 1 + extra

    diff = budget - sum(allocated.values())
    if diff != 0:
        top = max(auto_env_keys, key=lambda key: env_work_units.get(key, 0.0))
        allocated[top] = max(1, allocated[top] + diff)
    return allocated


def _plan_sandbox_configs(
    base_sandboxes: Sequence[SandboxConfig],
    expanded_tasks: Sequence[str],
    trackers: Mapping[str, TaskTracker],
    sandbox_pool_instances: int | None,
    sandbox_pool_min_instances: int | None = None,
) -> SandboxPlan | None:
    """Resolve relevant sandbox configs and concrete executor counts."""
    from olmo_eval.harness.sandbox.image import dependencies_to_dockerfile_extra

    sandbox_demand = _collect_sandbox_demand(trackers, expanded_tasks)
    sandbox_envs = sandbox_demand.sandbox_envs
    needs_default = sandbox_demand.needs_default
    if not sandbox_envs and not needs_default:
        return None

    template = base_sandboxes[0]
    used_caps = {senv.capability for senv in sandbox_envs}
    sandboxes = [
        cfg
        for i, cfg in enumerate(base_sandboxes)
        if (i == 0 and needs_default) or (i != 0 and cfg.capabilities in used_caps)
    ]

    env_to_index: dict[str, int] = {}
    if needs_default and sandboxes:
        env_to_index[_DEFAULT_SANDBOX_ENV] = 0

    for senv in sandbox_envs:
        extra = dependencies_to_dockerfile_extra(senv.dependencies)
        match_idx = next(
            (i for i, cfg in enumerate(sandboxes) if cfg.capabilities == senv.capability),
            None,
        )
        if match_idx is not None:
            matched = sandboxes[match_idx]
            sandboxes[match_idx] = replace(
                matched,
                dockerfile_extra=matched.dockerfile_extra + extra + senv.dockerfile_extra,
            )
            env_to_index[senv.name] = match_idx
        else:
            sandboxes.append(
                replace(
                    template,
                    capabilities=senv.capability,
                    dockerfile_extra=template.dockerfile_extra + extra + senv.dockerfile_extra,
                    inject_swerex=True,
                    instances=None,
                )
            )
            env_to_index[senv.name] = len(sandboxes) - 1

    explicit_allocations: dict[str, int] = {}
    for env_key, idx in env_to_index.items():
        instances = sandboxes[idx].instances
        if instances is not None:
            explicit_allocations[env_key] = instances
    auto_env_keys = [env_key for env_key in env_to_index if env_key not in explicit_allocations]
    if (
        sandbox_pool_instances is None
        and _DEFAULT_SANDBOX_ENV in explicit_allocations
        and auto_env_keys
    ):
        auto_env_list = ", ".join(sorted(auto_env_keys))
        logger.warning(
            "Default sandbox has explicit instances=%s while sandbox envs [%s] are "
            "auto-managed with no shared sandbox pool; those envs will default to one "
            "executor each. Use sandbox_pool_instances or a global "
            "'sandboxes={...,\"instances\":N}' override to distribute capacity across all "
            "sandbox envs.",
            explicit_allocations[_DEFAULT_SANDBOX_ENV],
            auto_env_list,
        )
    auto_allocations = _allocate_auto_sandbox_instances(
        auto_env_keys,
        sandbox_demand.env_work_units,
        sandbox_pool_instances,
    )
    auto_min_requirements = (
        _allocate_auto_sandbox_instances(
            auto_env_keys,
            sandbox_demand.env_work_units,
            sandbox_pool_min_instances,
        )
        if sandbox_pool_min_instances is not None
        else {}
    )
    allocated: dict[str, int] = dict(explicit_allocations)
    allocated.update(auto_allocations)

    materialized_sandboxes = list(sandboxes)
    min_required: dict[str, int] = {}
    for env_key, idx in env_to_index.items():
        original = materialized_sandboxes[idx]
        resolved_instances = allocated[env_key]
        resolved_min = auto_min_requirements.get(env_key, original.min_instances)
        if resolved_min is not None:
            resolved_min = min(resolved_min, resolved_instances)
        min_required[env_key] = resolved_min if resolved_min is not None else resolved_instances
        materialized_sandboxes[idx] = replace(
            original,
            instances=resolved_instances,
            min_instances=resolved_min,
        )

    return SandboxPlan(
        sandboxes=materialized_sandboxes,
        env_sandbox_items=sandbox_demand.env_sandbox_items,
        env_work_units=sandbox_demand.env_work_units,
        env_task_count=sandbox_demand.env_task_count,
        allocated=allocated,
        min_required=min_required,
        budget=sum(allocated.values()),
        needs_default=needs_default,
    )


def _determine_scoring_limits(
    harness_config: HarnessConfig,
    sandbox_configs_list: Sequence[dict[str, Any]] | None,
    process_pool_plan: ProcessPoolPlan | None,
) -> tuple[int, int]:
    """Return (response_concurrency, context_concurrency) for scoring."""
    import os

    sandbox_pool_size = (
        sum((cfg.get("instances") or 1) for cfg in sandbox_configs_list)
        if sandbox_configs_list is not None
        else 0
    )
    process_pool_size = (
        max((config.workers for config in process_pool_plan.pools.values()), default=0)
        if process_pool_plan is not None
        else 0
    )
    explicit = harness_config.scoring_concurrency

    if explicit is not None:
        context_concurrency = max(1, explicit)
        response_concurrency = max(context_concurrency, sandbox_pool_size, process_pool_size)
        return response_concurrency, context_concurrency

    derived = max(sandbox_pool_size, process_pool_size)
    if derived <= 0:
        cpu_count = os.cpu_count() or 4
        derived = max(1, int(cpu_count * 0.75))

    return derived, derived


@dataclass
class AsyncEvalRunner(RunnerResultsMixin, BaseEvalRunner):
    """Async evaluation runner with instance-level queuing.

    Uses a single model with instance-level queuing where instances from all
    tasks are mixed together, enabling better GPU utilization and early
    completion reporting.
    """

    # Harness configuration (includes provider, tools, system prompt)
    harness_config: HarnessConfig = field(default_factory=lambda: HarnessConfig(name="default"))

    # Task configuration
    task_specs: list[str] = field(default_factory=list)
    task_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Output configuration
    output_dir: str = BEAKER_RESULT_DIR

    # vLLM-specific configuration
    attention_backend: str | None = None

    # Dashboard upload configuration; None writes the manifest but uploads nothing
    upload_config: UploadConfig | None = None

    # Experiment metadata
    experiment_name: str | None = None
    experiment_group: str | None = None

    # Output persistence options
    save_predictions: bool = True
    save_requests: bool = True

    # Shuffle seed for deterministic instance ordering (enables future checkpointing)
    shuffle_seed: int = 42

    # Instance inspection options
    inspect_instance: bool = False
    inspect_formatted: bool = False
    inspect_tokens: bool = False
    inspect_response: bool = False
    inspect_request: bool = False

    # Configuration for print_config display
    _mode_name: str = "Async Mode"
    _mode_description: str = "Async (All-at-once)"

    # Run record state for the manifest written at start and finalize
    _started_at: str | None = field(default=None, init=False, repr=False)
    _prepared_tasks: dict[str, Task] = field(default_factory=dict, init=False, repr=False)

    @property
    def provider_config(self) -> ProviderConfig:
        """Get the provider config from harness config."""
        return self.harness_config.provider

    @property
    def model_name(self) -> str:
        """Get the model name from provider config."""
        return self.provider_config.model

    def validate(self) -> None:
        """Validate runner configuration."""
        from olmo_eval.runners.common.constants import ValidationError

        if not self.provider_config.model:
            raise ValidationError("provider_config.model is required")

        if not self.task_specs:
            raise ValidationError("task_specs is required")

        # Validate task specs
        errors = self._validate_task_specs()
        if errors:
            raise ValidationError("\n".join(errors))

    def print_config(self) -> None:
        """Print runner configuration."""
        from rich.table import Table

        from olmo_eval.common.configs import expand_tasks

        table = Table(title=f"Run Configuration ({self._mode_name})")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="white")

        table.add_row("Model", self.model_name)
        table.add_row("Provider", self.provider_config.get_provider_name())
        table.add_row("Mode", self._mode_description)
        table.add_row("Output Dir", self.output_dir)

        console.print(table)

        expanded = expand_tasks(self.task_specs)
        console.print(f"\n[bold]Tasks:[/bold] {len(expanded)}")
        for spec in expanded:
            console.print(f"  - {spec}")

    def run(self) -> dict[str, Any]:
        """Sync wrapper for async execution."""
        return asyncio.run(self.run_async())

    async def run_async(self) -> dict[str, Any]:
        """Execute evaluations using instance-level queuing."""
        # Track experiment start time. This is also the anchor for startup_seconds: it
        # comes before task data loading, worker spawn and provider init, and after CLI
        # startup (imports and config resolution), which it does not cover.
        experiment_start = time.time()

        # Generate experiment ID early so metrics can include it
        experiment_id = generate_experiment_id()
        self._start_run_record(experiment_id)

        # Prepare tasks
        expanded_tasks, trackers, items = self._prepare_tasks()
        self._prepared_tasks = {
            spec: tracker.task for spec, tracker in trackers.items() if tracker.task is not None
        }
        total_instances = len(items)
        runner_logger.info(f"Total instances: {total_instances}")

        # Update harness metrics config with experiment metadata before starting workers
        self._update_metrics_config(experiment_id)
        self._enable_tool_parsing_for_tool_requests(items)

        # Setup multiprocessing
        ctx = mp.get_context("spawn")
        item_queue: mp.Queue = ctx.Queue()
        result_queue: mp.Queue = ctx.Queue()
        total_gpus = self._get_gpu_count()

        # Queue for workers to report init times (replaces mp.Manager dict to
        # avoid a long-lived server process that can zombie and hang on shutdown)
        init_queue: mp.Queue = ctx.Queue()
        start_event = ctx.Event()

        # Shuffle with seed for deterministic ordering (enables future checkpointing)
        rng = random.Random(self.shuffle_seed)
        rng.shuffle(items)
        for item in items:
            item_queue.put(item)

        # Start workers
        workers: list[mp.process.BaseProcess] = []
        sandbox_manager = None
        process_pool_manager = None
        inference_manager = None

        try:
            from olmo_eval.inference.gpu_planner import GPUPlanner
            from olmo_eval.inference.provider_manager import (
                ProviderManager,
                validate_inference_workers,
            )

            # Use GPU planner to allocate GPUs for main workers and auxiliary providers
            planner = GPUPlanner.from_harness_config(self.harness_config, total_gpus)
            gpu_plan = planner.plan()

            # Get main worker GPUs (flattened list from all worker allocations)
            main_gpu_ids: list[int] = []
            for alloc in gpu_plan.main_workers:
                main_gpu_ids.extend(alloc.gpu_ids)

            # Number of inference workers is determined by provider.num_instances
            num_inference_workers = self.harness_config.provider.num_instances

            # Validate configuration
            validate_inference_workers(num_inference_workers, len(main_gpu_ids))

            provider_manager = ProviderManager(
                harness_config=self.harness_config,
                num_inference_workers=num_inference_workers,
                gpu_ids=main_gpu_ids,
                item_queue=item_queue,
                result_queue=result_queue,
                output_dir=self.output_dir,
            )

            # Add poison pills (one per worker)
            provider_manager.add_poison_pills()

            # Start workers
            workers = provider_manager.start(ctx, total_instances, init_queue, start_event)

            # Start auxiliary inference servers if configured
            registry_config: dict[str, list[dict[str, Any]]] | None = None
            if self.harness_config.auxiliary_providers:
                from olmo_eval.inference.manager import InferenceManager

                auxiliary_gpus = gpu_plan.get_auxiliary_gpus()

                import os

                inference_manager = InferenceManager(
                    configs=dict(self.harness_config.auxiliary_providers),
                    available_gpu_ids=auxiliary_gpus,
                    log_dir=os.path.join(self.output_dir, "logs"),
                )
                registry_config = inference_manager.start()
                runner_logger.info(f"Auxiliary providers ready: {list(registry_config.keys())}")

            # Prepare sandbox configs for scoring worker if configured
            sandbox_configs_list = None
            if self.harness_config.sandboxes:
                sandboxes = list(self.harness_config.sandboxes)

                sandbox_plan = _plan_sandbox_configs(
                    self.harness_config.sandboxes,
                    expanded_tasks,
                    trackers,
                    self.harness_config.sandbox_pool_instances,
                    self.harness_config.sandbox_pool_min_instances,
                )
                if sandbox_plan is not None:
                    sandboxes = sandbox_plan.sandboxes

                    for env_key in sorted(sandbox_plan.env_work_units):
                        display_name = "default" if env_key == _DEFAULT_SANDBOX_ENV else env_key
                        work_units = _format_work_units(
                            sandbox_plan.env_work_units.get(env_key, 0.0)
                        )
                        runner_logger.info(
                            f"Sandbox env '{display_name}' "
                            f"({sandbox_plan.allocated.get(env_key, 1)} executors, "
                            f"{sandbox_plan.env_sandbox_items.get(env_key, 0)} sandbox items, "
                            f"{work_units} work units)"
                        )

                    # Print sandbox distribution summary
                    from rich.table import Table

                    dist_table = Table(title="Sandbox Distribution")
                    dist_table.add_column("Env", style="cyan")
                    dist_table.add_column("Tasks", justify="right")
                    dist_table.add_column("Sandbox Items", justify="right")
                    dist_table.add_column("Work Units", justify="right")
                    dist_table.add_column("Executors", justify="right")
                    dist_table.add_column("Share", justify="right")

                    for env_key in sorted(sandbox_plan.env_work_units):
                        display_name = "default" if env_key == _DEFAULT_SANDBOX_ENV else env_key
                        executors = sandbox_plan.allocated.get(env_key, 0)
                        share = (
                            f"{executors / sandbox_plan.budget * 100:.1f}%"
                            if sandbox_plan.budget > 0
                            else "0.0%"
                        )
                        dist_table.add_row(
                            display_name,
                            str(sandbox_plan.env_task_count.get(env_key, 0)),
                            str(sandbox_plan.env_sandbox_items.get(env_key, 0)),
                            _format_work_units(sandbox_plan.env_work_units.get(env_key, 0.0)),
                            str(executors),
                            share,
                        )

                    console.print(dist_table)
                else:
                    sandboxes = _materialize_sandbox_instances(sandboxes)

                # Pre-build unique sandbox images in parallel to avoid
                # duplicate builds when multiple executors share the same image
                from concurrent.futures import ThreadPoolExecutor

                from olmo_eval.harness.sandbox.image import get_swerex_image

                # Dedupe by (image, dockerfile_extra) — configs sharing these produce the same image
                unique_builds: dict[tuple[str, tuple[str, ...]], Any] = {}
                for cfg in sandboxes:
                    if cfg.inject_swerex:
                        key = (cfg.image, cfg.dockerfile_extra)
                        if key not in unique_builds:
                            unique_builds[key] = cfg

                if unique_builds:
                    runner_logger.info(
                        f"Pre-building {len(unique_builds)} unique sandbox image(s)..."
                    )

                    def _build(cfg: Any) -> None:
                        require_registry = cfg.mode.value == "modal"
                        get_swerex_image(
                            cfg.image,
                            cfg.container_runtime,
                            cfg.dockerfile_extra,
                            require_registry=require_registry,
                        )

                    with ThreadPoolExecutor(max_workers=len(unique_builds)) as pool:
                        list(pool.map(_build, unique_builds.values()))

                sandbox_configs_list = [s.to_dict() for s in sandboxes]

            process_pool_plan = _plan_process_scoring_pools(
                self.harness_config.scoring_process_pools,
                expanded_tasks,
                trackers,
                self.harness_config.scoring_concurrency,
            )
            if process_pool_plan is not None:
                from olmo_eval.common.execution import ProcessPoolManager

                for pool_name in sorted(process_pool_plan.demand):
                    config = process_pool_plan.pools[pool_name]
                    demand = process_pool_plan.demand[pool_name]
                    auto = " (auto-created)" if pool_name in process_pool_plan.auto_created else ""
                    runner_logger.info(
                        f"Process scoring pool '{pool_name}'{auto} "
                        f"({config.workers} workers, start_method={config.start_method}, "
                        f"max_tasks_per_child={config.max_tasks_per_child}, "
                        f"{demand.response_jobs} response jobs across "
                        f"{demand.task_count} task(s), {demand.scorer_count} scorer(s))"
                    )
                process_pool_manager = ProcessPoolManager(process_pool_plan.pools)

            response_scoring_concurrency, context_scoring_concurrency = _determine_scoring_limits(
                self.harness_config,
                sandbox_configs_list,
                process_pool_plan,
            )
            runner_logger.info(f"Response scoring concurrency: {response_scoring_concurrency}")
            runner_logger.info(f"Context scoring concurrency: {context_scoring_concurrency}")

            # Initialize sandbox manager inline (no separate scorer process)
            sandbox_manager = None
            if sandbox_configs_list is not None:
                from olmo_eval.harness.sandbox import SandboxConfig, SandboxManager

                sandbox_configs = [SandboxConfig.from_dict(d) for d in sandbox_configs_list]
                runner_logger.info(
                    f"Initializing sandbox manager with {len(sandbox_configs)} config(s)..."
                )
                sandbox_manager = SandboxManager(sandbox_configs, owner="scorer")
                await sandbox_manager.start()
                runner_logger.info("Sandbox manager ready")

            # Create provider registry for auxiliary providers
            provider_registry = None
            if registry_config:
                from olmo_eval.inference.registry import ProviderRegistry

                provider_registry = ProviderRegistry.from_serialized(registry_config)
                if provider_registry:
                    runner_logger.info(
                        f"Provider registry ready with providers: {provider_registry.names}"
                    )

            from olmo_eval.common.execution import ScoringContext

            scoring_context = ScoringContext(
                execution_env=sandbox_manager,
                scoring_concurrency=context_scoring_concurrency,
                inference_pool=provider_registry,
                process_pool_manager=process_pool_manager,
            )

            # Ensure inference workers are ready before dispatching
            wait_for_workers_ready(workers, result_queue, startup_timeout=60.0)

            # Wait for workers to report their init times (also checks for crashes)
            provider_init_seconds = wait_for_init_times(
                init_queue, num_inference_workers, workers=workers, result_queue=result_queue
            )
            if len(provider_init_seconds) != num_inference_workers:
                raise RuntimeError(
                    f"Expected {num_inference_workers} inference worker(s) to initialize "
                    f"before starting work, got {len(provider_init_seconds)}."
                )

            runner_logger.info(
                f"All {num_inference_workers} inference worker(s) ready; starting queued work"
            )
            start_event.set()

            # Reset tracker start times now that workers are ready
            # This ensures task duration only measures actual processing time
            processing_start = time.time()
            for tracker in trackers.values():
                tracker.start_time = processing_start
            startup_seconds = processing_start - experiment_start

            # Process results
            results = await self._process_results(
                trackers,
                result_queue,
                workers,
                scoring_context,
                response_scoring_concurrency,
                len(expanded_tasks),
                total_instances,
            )
            processing_seconds = _processing_seconds(results, processing_start, time.time())

            # Wait for all workers
            for worker in workers:
                worker.join(timeout=10)
                if worker.is_alive():
                    worker.terminate()
                    worker.join()

            # Optionally inspect first response of each task
            if self.inspect_response:
                from olmo_eval.common.inspection import inspect_response

                for spec, tracker in trackers.items():
                    if tracker.responses:
                        first_response = next(iter(tracker.responses.values()))
                        console.print()
                        inspect_response(
                            first_response,
                            console=console,
                            task_name=spec,
                        )
                        break  # Only show first task's first response

            # Compute experiment duration
            experiment_duration_seconds = time.time() - experiment_start

            # Aggregate and save results
            results_dict = self._aggregate_results(results, expanded_tasks)
            results_dict["startup_seconds"] = startup_seconds
            results_dict["processing_started_at"] = utc_iso(processing_start)
            results_dict["processing_seconds"] = processing_seconds
            return self._finalize_and_gate(
                results,
                results_dict,
                experiment_id=experiment_id,
                experiment_duration_seconds=experiment_duration_seconds,
                provider_init_seconds=provider_init_seconds,
            )
        finally:
            terminate_workers(workers)
            if sandbox_manager is not None:
                runner_logger.info("Stopping sandbox manager...")
                with contextlib.suppress(Exception):
                    await sandbox_manager.stop()
            if process_pool_manager is not None:
                _shutdown_process_scoring_pools(process_pool_manager)
            if inference_manager is not None:
                inference_manager.shutdown()
            for q in [item_queue, result_queue, init_queue]:
                q.cancel_join_thread()

    def _prepare_tasks(
        self,
    ) -> tuple[list[str], dict[str, TaskTracker], list[QueueItem]]:
        """Prepare all tasks and return tracking data structures."""
        expanded_tasks = expand_tasks(self.task_specs)

        if not expanded_tasks:
            raise ValueError("No tasks to run after expansion. Check task_specs configuration.")

        trackers: dict[str, TaskTracker] = {}
        items: list[QueueItem] = []

        def prepare_one(spec: str) -> tuple[str, TaskTracker, list[QueueItem]]:
            try:
                overrides, sampling_overrides = self._build_task_overrides(spec)
                task, task_items = prepare_task_items(
                    spec,
                    self.model_name,
                    overrides or None,
                    sampling_overrides=sampling_overrides or None,
                )
                tracker = TaskTracker(
                    model_name=self.model_name,
                    spec=spec,
                    task=task,
                    total_instances=len(task_items),
                )
                return (spec, tracker, task_items)
            except Exception as e:
                tracker = TaskTracker(
                    model_name=self.model_name,
                    spec=spec,
                    task=None,
                    total_instances=0,
                    error=str(e),
                )
                return (spec, tracker, [])

        from rich.table import Table

        # tqdm creates its class-level write lock lazily, so concurrent first uses can
        # race and leave a loader reading `tqdm._lock` before it exists. Dataset loads
        # below run in parallel, so claim the lock once up front.
        with contextlib.suppress(ImportError):
            import tqdm as tqdm_module

            tqdm_module.tqdm.get_lock()

        # Collect prepared tasks in parallel, but accumulate results for deterministic ordering
        prepared_results: dict[str, tuple[TaskTracker, list[QueueItem]]] = {}
        with ThreadPoolExecutor(max_workers=min(32, len(expanded_tasks))) as executor:
            futures = {executor.submit(prepare_one, spec): spec for spec in expanded_tasks}
            for future in as_completed(futures):
                spec, tracker, task_items = future.result()
                trackers[spec] = tracker
                prepared_results[spec] = (tracker, task_items)

        # Add items in deterministic task spec order (not completion order)
        # This ensures shuffle produces identical results across runs
        for spec in expanded_tasks:
            tracker, task_items = prepared_results[spec]
            items.extend(task_items)

        # Print task preparation summary table
        table = Table(title="Tasks")
        table.add_column("Task", style="cyan")
        table.add_column("Instances", justify="right")
        table.add_column("Status")

        for spec in expanded_tasks:
            tracker = trackers[spec]
            if tracker.error:
                table.add_row(spec, "-", f"[red]ERROR: {tracker.error}[/red]")
            else:
                table.add_row(spec, str(tracker.total_instances), "[green]Ready[/green]")

        console.print(table)

        # Fail fast if any tasks failed to prepare
        failed_tasks = [spec for spec, tracker in trackers.items() if tracker.error]
        if failed_tasks:
            error_details = "\n".join(
                f"  - {spec}: {trackers[spec].error}" for spec in failed_tasks
            )
            raise RuntimeError(f"Failed to prepare {len(failed_tasks)} task(s):\n{error_details}")

        # Optionally inspect first instance of each task
        if (
            self.inspect_instance
            or self.inspect_formatted
            or self.inspect_tokens
            or self.inspect_request
        ):
            from olmo_eval.common.inspection import inspect_task_instances

            inspect_task_instances(
                trackers,
                self.provider_config,
                inspect_instance_flag=self.inspect_instance,
                inspect_formatted=self.inspect_formatted,
                inspect_tokens_flag=self.inspect_tokens,
                inspect_request_flag=self.inspect_request,
                console=console,
            )

        return expanded_tasks, trackers, items

    def _build_task_overrides(self, spec: str) -> tuple[dict[str, Any], dict[str, Any]]:
        """Build task and sampling overrides for a given task spec.

        Returns:
            Tuple of (task_overrides, sampling_overrides)
        """
        from dataclasses import fields

        from olmo_eval.common.types import SamplingParams
        from olmo_eval.evals.tasks.common.base import TaskConfig

        task_overrides: dict[str, Any] = {}
        sampling_overrides: dict[str, Any] = {}

        # Get field names from dataclasses
        task_fields = {f.name for f in fields(TaskConfig)}
        sampling_fields = {f.name for f in fields(SamplingParams)}

        # Apply per-task overrides
        per_task = self.task_overrides.get(spec, {})
        for key, value in per_task.items():
            if key in task_fields:
                task_overrides[key] = value
            elif key in sampling_fields:
                sampling_overrides[key] = value

        return task_overrides, sampling_overrides

    def _get_gpu_count(self) -> int:
        """Get total number of available GPUs."""
        try:
            import torch

            return torch.cuda.device_count()
        except ImportError:
            return 0

    def _enable_tool_parsing_for_tool_requests(self, items: Sequence[QueueItem]) -> None:
        """Start a managed vLLM server that returns tool calls when tasks send tools.

        A harness with its own tools already gets this from the worker. A task
        that puts function schemas on its requests needs it too, or the server
        answers in plain text. A setting given explicitly is left as it is.

        This must be called before starting workers, which read the setting from
        the serialized config.
        """
        provider = self.harness_config.provider
        if str(provider.kind) != "vllm_server" or provider.base_url:
            return
        if "enable_auto_tool_choice" in (provider.kwargs or {}):
            return
        if not any(item.request.tools for item in items):
            return
        self.harness_config = self.harness_config.with_provider_overrides(
            enable_auto_tool_choice=True
        )
        runner_logger.info("Tasks send function schemas; enabling tool-call parsing on vLLM")

    def _update_metrics_config(self, experiment_id: str) -> None:
        """Update harness metrics config with experiment metadata.

        This must be called before starting workers so the serialized config
        includes all metadata for metrics reporting.
        """
        from olmo_eval.common.types import compute_model_hash

        if self.harness_config.metrics is None or not self.harness_config.metrics.enabled:
            return

        # Compute model hash from provider config
        model_hash = compute_model_hash(self.harness_config.provider.to_dict())

        # Update metrics config with all available metadata
        updated_metrics = self.harness_config.metrics.with_metadata(
            experiment_id=experiment_id,
            experiment_name=self.experiment_name,
            experiment_group=self.experiment_group,
            model_name=self.model_name,
            model_hash=model_hash,
        )

        # Replace harness config with updated metrics
        self.harness_config = self.harness_config.with_metrics(updated_metrics)

    @property
    def max_hard_failure_rate(self) -> float:
        """Fraction of a task's instances allowed to hard-fail before it is failed."""
        configured = self.harness_config.max_hard_failure_rate
        return DEFAULT_MAX_HARD_FAILURE_RATE if configured is None else configured

    async def _process_results(
        self,
        trackers: dict[str, TaskTracker],
        result_queue: mp.Queue,
        workers: list[mp.process.BaseProcess],
        scoring_context: Any,
        scoring_concurrency: int,
        total_tasks: int,
        total_instances: int,
    ) -> dict[str, Any]:
        """Process results from workers with inline async scoring."""
        return await process_results(
            trackers=trackers,
            result_queue=result_queue,
            workers=workers,
            scoring_context=scoring_context,
            scoring_concurrency=scoring_concurrency,
            total_tasks=total_tasks,
            total_instances=total_instances,
            model_name=self.model_name,
            save_predictions=self.save_predictions,
            write_predictions_fn=self._write_predictions,
            save_requests=self.save_requests,
            write_requests_fn=self._write_requests,
            max_hard_failure_rate=self.max_hard_failure_rate,
        )

    def _aggregate_results(
        self,
        results: dict[str, Any],
        expanded_tasks: list[str],
    ) -> dict[str, Any]:
        """Aggregate results and prepare final output."""
        return aggregate_results(
            results=results,
            expanded_tasks=expanded_tasks,
            task_specs=self.task_specs,
            provider_config=self.provider_config,
            attention_backend=self.attention_backend,
            harness_config=self.harness_config,
        )

    def _finalize_and_gate(
        self,
        results: dict[str, TaskResult],
        results_dict: dict[str, Any],
        experiment_id: str,
        experiment_duration_seconds: float | None = None,
        provider_init_seconds: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Write the results, then fail the run if a task blew its budget.

        The order is the point: a run that trips the gate must still leave its
        metrics.json and predictions on disk to be diagnosed from, so the gate is
        checked only after _finalize_and_save has returned.
        """
        final_results = self._finalize_and_save(
            results_dict,
            experiment_id=experiment_id,
            experiment_duration_seconds=experiment_duration_seconds,
            provider_init_seconds=provider_init_seconds,
        )
        check_hard_failure_gate(results)
        return final_results

    def _finalize_and_save(
        self,
        results_dict: dict[str, Any],
        experiment_id: str,
        experiment_duration_seconds: float | None = None,
        provider_init_seconds: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Log summary, write metrics.json and the manifest, then upload the results."""
        from olmo_eval.common.types import compute_model_hash

        self._log_summary(results_dict)

        model_hash = compute_model_hash(results_dict.get("model_config", {}))
        results_dict["_model_hash"] = model_hash

        self._write_metrics_json(
            results=results_dict,
            experiment_id=experiment_id,
            experiment_name=self.experiment_name,
            experiment_group=self.experiment_group,
            model_hash=model_hash,
            experiment_duration_seconds=experiment_duration_seconds,
            provider_init_seconds=provider_init_seconds,
        )

        results_dict["_experiment_id"] = experiment_id

        self._write_final_manifest(
            results_dict,
            experiment_id=experiment_id,
            model_hash=model_hash,
            experiment_duration_seconds=experiment_duration_seconds,
            provider_init_seconds=provider_init_seconds,
        )
        if self.upload_config is not None and self.upload_config.enabled:
            from olmo_eval.upload import upload_results_dir

            upload_results_dir(self.output_dir, self.upload_config)

        return results_dict

    def _start_run_record(self, experiment_id: str) -> None:
        """Write manifest.json with status "running" and register the run (best effort)."""
        from olmo_eval.upload import register_run_start
        from olmo_eval.upload.manifest import (
            build_manifest,
            build_run_info,
            model_info_from_provider,
            utc_now,
            write_manifest,
        )

        self._started_at = utc_now()
        try:
            run = build_run_info(
                run_id=experiment_id,
                status="running",
                started_at=self._started_at,
                experiment_name=self.experiment_name,
                experiment_group=self.experiment_group,
                task_specs=self.task_specs,
                output_dir=self.output_dir,
                harness_config=self.harness_config.to_dict(),
                tags=self.upload_config.tags if self.upload_config else (),
            )
            model = model_info_from_provider(self.provider_config, self.attention_backend)
            write_manifest(self.output_dir, build_manifest(run, model))
        except Exception as e:
            logger.warning(f"Could not write the run manifest: {e}")
            return
        if self.upload_config is not None and self.upload_config.enabled:
            register_run_start(self.output_dir, self.upload_config)

    def _write_final_manifest(
        self,
        results_dict: dict[str, Any],
        experiment_id: str,
        model_hash: str | None,
        experiment_duration_seconds: float | None,
        provider_init_seconds: dict[str, float] | None,
    ) -> None:
        """Rewrite manifest.json with the final status, task metadata and suites."""
        from olmo_eval.upload.manifest import (
            build_manifest,
            build_model_info,
            build_run_info,
            build_task_entry,
            run_status,
            utc_now,
            write_manifest,
        )
        from olmo_eval.upload.metadata import (
            metric_meta_for_spec,
            metric_meta_for_task,
            suite_results,
        )

        try:
            tasks: dict[str, dict[str, Any]] = {}
            for spec, data in results_dict.get("tasks", {}).items():
                names = list((data.get("metrics") or {}).keys())
                task = self._prepared_tasks.get(spec)
                meta = (
                    metric_meta_for_task(task, names)
                    if task is not None
                    else metric_meta_for_spec(spec, names)
                )
                tasks[spec] = build_task_entry(spec, meta, data.get("error"))
            run = build_run_info(
                run_id=experiment_id,
                status=run_status({spec: t["error"] for spec, t in tasks.items()}),
                started_at=self._started_at,
                finished_at=utc_now(),
                duration_seconds=experiment_duration_seconds,
                experiment_name=self.experiment_name,
                experiment_group=self.experiment_group,
                task_specs=self.task_specs,
                output_dir=self.output_dir,
                harness_config=results_dict.get("harness_config"),
                provider_init_seconds=provider_init_seconds,
                errors=results_dict.get("errors", []),
                tags=self.upload_config.tags if self.upload_config else (),
            )
            model = build_model_info(
                name=results_dict.get("model") or self.model_name,
                path=results_dict.get("model_path") or self.model_name,
                model_hash=model_hash,
                revision=self.provider_config.revision,
                provider_kind=results_dict.get("provider") or str(self.provider_config.kind),
                provider_config=results_dict.get("model_config", {}),
            )
            suites = suite_results(results_dict.get("suites") or {}, results_dict.get("tasks"))
            write_manifest(self.output_dir, build_manifest(run, model, tasks, suites))
        except Exception as e:
            logger.warning(f"Could not write the run manifest: {e}")
