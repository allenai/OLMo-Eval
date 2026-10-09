"""Configuration loading for Beaker launch command."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import beaker
from rich.console import Console

if TYPE_CHECKING:
    from olmo_eval.launch import EvalConfig

console = Console()


@dataclass
class LaunchConfig:
    """Parsed and validated configuration for Beaker launch."""

    name: str
    model_specs: list[str]
    task_specs: list[str]
    cluster: str
    workspace: str
    budget: str | None = None

    task_overrides: dict[str, list[str]] = field(default_factory=dict)

    max_gpus_per_node: int = 8
    priority: str = "normal"
    preemptible: bool | None = True
    min_runtime: str | None = None
    timeout: str = "6h"
    retries: int | None = None
    image: str | None = None
    groups: list[str] = field(default_factory=list)

    gpus: int = 0

    s3_bucket: str = "ai2-llm"
    s3_prefix: str = "olmo-eval"
    s3_endpoint_url: str | None = None
    s3_region: str = "us-east-1"

    store: bool = False
    debug_requests: bool = False
    debug_provider: bool = False
    force_download_model: bool = False
    save_predictions: bool = True
    save_requests: bool = True
    inspect_instance: bool = False
    inspect_formatted: bool = False
    inspect_tokens: bool = False
    inspect_response: bool = False
    inspect_request: bool = False

    harness: str | None = None
    harness_overrides: list[str] = field(default_factory=list)

    uv_cache_dir: str | None = None

    # Secret env overrides: maps beaker_secret_name -> env_var_name
    # Allows overriding default {username}_{ENV_VAR} pattern
    secret_env_overrides: dict[str, str] = field(default_factory=dict)

    # User-supplied plain env vars forwarded/set on the job via --env/-e.
    # These win over infra and store defaults.
    env_vars: dict[str, str] = field(default_factory=dict)


def resolve_preemption(
    preemptible: bool | None, min_runtime: str | None
) -> tuple[bool | None, str | None]:
    """Resolve the preemption settings from one source into (preemptible, min_runtime).

    A min runtime replaces preemptible, so the returned preemptible is None when a min
    runtime is set. Without either setting, jobs are preemptible.

    Raises:
        ValueError: If both are set, or the min runtime is not a valid duration.
    """
    from olmo_eval.launch.beaker import validate_min_runtime

    if min_runtime is None:
        return (True if preemptible is None else preemptible), None
    if preemptible is not None:
        raise ValueError("preemptible and min_runtime cannot both be set; set min_runtime alone")
    return None, validate_min_runtime(min_runtime)


class LaunchConfigLoader:
    """Loads and merges configuration from YAML file and CLI arguments."""

    def __init__(self, config_path: str | None, cli_args: dict[str, Any]):
        self.config_path = config_path
        self.cli_args = cli_args

    def load(self) -> LaunchConfig:
        """Load and merge configuration."""
        from olmo_eval.common.constants.infrastructure import DEFAULT_MAX_GPUS_PER_NODE
        from olmo_eval.launch import EvalConfig

        cfg: EvalConfig | None = None

        if self.config_path:
            try:
                cfg = EvalConfig.from_yaml(self.config_path)
            except FileNotFoundError as e:
                console.print(f"[red]Error:[/red] {e}")
                raise SystemExit(1) from None
            except Exception as e:
                console.print(f"[red]Config error:[/red] {e}")
                raise SystemExit(1) from None

        cli_name = self.cli_args.get("name")
        cli_model = self.cli_args.get("model", ())
        cli_task = self.cli_args.get("task", ())
        cli_task_overrides: dict[str, list[str]] = self.cli_args.get("task_overrides", {})
        cli_cluster = self.cli_args.get("cluster")
        cli_max_gpus_per_node = self.cli_args.get("max_gpus_per_node")
        cli_priority = self.cli_args.get("priority")
        cli_preemptible = self.cli_args.get("preemptible")
        cli_min_runtime = self.cli_args.get("min_runtime")
        cli_timeout = self.cli_args.get("timeout")
        cli_retries = self.cli_args.get("retries")
        cli_workspace = self.cli_args.get("workspace")
        cli_budget = self.cli_args.get("budget")
        cli_image = self.cli_args.get("image")
        cli_groups = self.cli_args.get("group", ())
        cli_gpus = self.cli_args.get("gpus")

        if cfg is not None:
            name = cli_name or cfg.name
            task_specs = list(cli_task) if cli_task else list(cfg.tasks)
            retries = cli_retries if cli_retries is not None else cfg.retries
            workspace = cli_workspace or cfg.workspace
            budget = cli_budget or cfg.budget
            model_specs = list(cli_model) if cli_model else list(cfg.models)
            cluster = cli_cluster if cli_cluster is not None else cfg.cluster
            max_gpus_per_node = (
                cli_max_gpus_per_node
                if cli_max_gpus_per_node is not None
                else cfg.max_gpus_per_node
            )
            priority = cli_priority if cli_priority is not None else cfg.priority
            # Preemption settings come from one source: the CLI if it sets either, else the file.
            if cli_preemptible is not None or cli_min_runtime is not None:
                preemptible, min_runtime = cli_preemptible, cli_min_runtime
            else:
                preemptible, min_runtime = cfg.preemptible, cfg.min_runtime
            timeout = cli_timeout if cli_timeout is not None else cfg.timeout
            gpus = cli_gpus if cli_gpus is not None else cfg.gpus
            image = cli_image or cfg.beaker_image
        else:
            name = cli_name
            task_specs = list(cli_task)
            retries = cli_retries
            workspace = cli_workspace
            budget = cli_budget
            model_specs = list(cli_model) if cli_model else []
            cluster = cli_cluster
            max_gpus_per_node = cli_max_gpus_per_node
            priority = cli_priority
            preemptible = cli_preemptible
            min_runtime = cli_min_runtime
            timeout = cli_timeout
            gpus = cli_gpus  # None means auto-detect from provider
            image = cli_image

        # Auto-detect GPU requirements from provider and harness if not explicitly set
        harness_name = self.cli_args.get("harness")
        harness_overrides = self.cli_args.get("harness_overrides", [])

        if gpus is None and model_specs:
            gpus = max(
                self._detect_gpu_requirement(model_spec, harness_name, harness_overrides)
                for model_spec in model_specs
            )
        elif gpus is None:
            gpus = 0  # Default to 0 if no model specified

        if model_specs:
            self._validate_gpu_requirement(gpus, model_specs, harness_name, harness_overrides)

        max_gpus_per_node = (
            max_gpus_per_node if max_gpus_per_node is not None else DEFAULT_MAX_GPUS_PER_NODE
        )
        priority = priority or "normal"
        try:
            preemptible, min_runtime = resolve_preemption(preemptible, min_runtime)
        except ValueError as e:
            console.print(f"[red]Error:[/red] {e}")
            raise SystemExit(1) from None
        timeout = timeout or "24h"

        self._validate_required(model_specs, task_specs, cluster, workspace, budget)

        # Auto-generate name if not provided
        if not name:
            name = self._generate_experiment_name(model_specs, task_specs)

        assert name is not None
        assert cluster is not None
        assert workspace is not None

        from olmo_eval.launch.beaker.constants import DEFAULT_S3_BUCKET, DEFAULT_S3_PREFIX

        s3_bucket = self.cli_args.get("s3_bucket") or DEFAULT_S3_BUCKET
        s3_prefix = self.cli_args.get("s3_prefix") or DEFAULT_S3_PREFIX

        effective_groups = list(cli_groups)
        if cfg is not None and cfg.groups:
            for g in cfg.groups:
                if g not in effective_groups:
                    effective_groups.append(g)

        return LaunchConfig(
            name=name,
            model_specs=model_specs,
            task_specs=task_specs,
            cluster=cluster,
            workspace=workspace,
            budget=budget,
            task_overrides=cli_task_overrides,
            max_gpus_per_node=max_gpus_per_node,
            priority=priority,
            preemptible=preemptible,
            min_runtime=min_runtime,
            timeout=timeout,
            retries=retries,
            image=image,
            groups=effective_groups,
            gpus=gpus,
            s3_bucket=s3_bucket,
            s3_prefix=s3_prefix,
            s3_endpoint_url=self.cli_args.get("s3_endpoint_url"),
            s3_region=self.cli_args.get("s3_region", "us-east-1"),
            store=self.cli_args.get("store", False),
            debug_requests=self.cli_args.get("debug_requests", False),
            debug_provider=self.cli_args.get("debug_provider", False),
            force_download_model=self.cli_args.get("force_download_model", False),
            save_predictions=self.cli_args.get("save_predictions", True),
            save_requests=self.cli_args.get("save_requests", True),
            inspect_instance=self.cli_args.get("inspect_instance", False),
            inspect_formatted=self.cli_args.get("inspect_formatted", False),
            inspect_tokens=self.cli_args.get("inspect_tokens", False),
            inspect_response=self.cli_args.get("inspect_response", False),
            inspect_request=self.cli_args.get("inspect_request", False),
            harness=self.cli_args.get("harness"),
            harness_overrides=self.cli_args.get("harness_overrides", []),
            uv_cache_dir=self.cli_args.get("uv_cache_dir"),
            secret_env_overrides=self.cli_args.get("secret_env_overrides", {}),
            env_vars=self.cli_args.get("env_vars", {}),
        )

    def _validate_required(
        self,
        model_specs: list[str],
        task_specs: list[str],
        cluster: str | None,
        workspace: str | None,
        budget: str | None,
    ) -> None:
        if not model_specs:
            console.print("[red]Error:[/red] --model/-m is required")
            raise SystemExit(1) from None
        if not task_specs:
            console.print("[red]Error:[/red] --task/-t is required")
            raise SystemExit(1) from None
        if not cluster:
            console.print("[red]Error:[/red] --cluster/-c is required")
            raise SystemExit(1) from None
        if not workspace:
            console.print("[red]Error:[/red] --workspace/-w is required")
            raise SystemExit(1) from None
        if not budget and not self._workspace_has_bound_budget(workspace):
            console.print(
                "[red]Error:[/red] --budget/-B is required when the workspace has no bound budget"
            )
            raise SystemExit(1) from None

    def _workspace_has_bound_budget(self, workspace: str) -> bool:
        client = beaker.Beaker.from_env(default_workspace=workspace)
        return bool(client.workspace.get().budget_id)

    def _generate_experiment_name(self, model_specs: list[str], task_specs: list[str]) -> str:
        """Generate experiment name from model and task specs.

        Format:
        - Single model, 1-2 tasks: {model}-{task1}-{task2}
        - Single model, 3+ tasks: {model}-{task1}-and-{N}-more
        - Multi-model: {tasks_part} only (each model name is appended per-experiment)
        """
        from olmo_eval.evals.tasks.common.registry import get_base_task_name
        from olmo_eval.launch import get_model_short_name, sanitize_beaker_name

        name_task_specs = [get_base_task_name(task_spec) for task_spec in task_specs]

        if len(name_task_specs) <= 2:
            tasks_part = "-".join(name_task_specs)
        else:
            tasks_part = f"{name_task_specs[0]}-and-{len(name_task_specs) - 1}-more"

        if len(model_specs) > 1:
            return sanitize_beaker_name(tasks_part)

        model_name = get_model_short_name(model_specs[0]) if model_specs else "eval"
        return sanitize_beaker_name(f"{model_name}-{tasks_part}")

    def _detect_gpu_requirement(
        self,
        model_spec: str,
        harness_name: str | None = None,
        harness_overrides: list[str] | None = None,
    ) -> int:
        """Detect GPU requirement based on provider and harness configuration.

        Calculates total GPUs needed for:
        - Main model (tensor_parallel_size × num_instances)
        - Auxiliary providers (each with their own tensor_parallel_size × num_instances)

        Args:
            model_spec: Model specification (name or preset).
            harness_name: Optional harness preset name.
            harness_overrides: Optional list of harness override strings.

        Returns:
            Total number of GPUs required.
        """
        from olmo_eval.common.configs import get_provider_config

        # Build effective harness config with overrides applied
        harness_config = self._get_effective_harness_config(harness_name, harness_overrides)

        try:
            provider_config = get_provider_config(model_spec)
        except Exception:
            # If the model cannot be resolved, retain the historical safe default
            # of one local GPU (plus any auxiliary provider requirements).
            provider_config = None

        if harness_config and provider_config:
            from olmo_eval.cli.run.config import merge_model_provider

            harness_config = merge_model_provider(
                harness_config, provider_config, harness_overrides
            )

        # Calculate main provider GPU requirements
        main_instances = 1
        main_tp = 1
        main_provider = harness_config.provider if harness_config else provider_config
        if main_provider:
            main_tp = main_provider.kwargs.get("tensor_parallel_size", 1)
            main_instances = main_provider.num_instances
        main_gpus = (
            main_instances * main_tp
            if main_provider is None or main_provider.requires_local_gpu
            else 0
        )

        # Calculate auxiliary provider GPU requirements
        aux_gpus = 0
        if harness_config and harness_config.auxiliary_providers:
            for config in harness_config.auxiliary_providers.values():
                if not config.requires_local_gpu:
                    # API-backed or external server - no GPUs needed
                    continue
                num_instances = config.num_instances
                tensor_parallel = config.kwargs.get("tensor_parallel_size", 1)
                aux_gpus += num_instances * tensor_parallel

        return main_gpus + aux_gpus

    def _validate_gpu_requirement(
        self,
        gpus: int,
        model_specs: list[str],
        harness_name: str | None = None,
        harness_overrides: list[str] | None = None,
    ) -> None:
        """Reject a GPU allocation too small for main or auxiliary providers."""
        required_gpus = max(
            self._detect_gpu_requirement(model_spec, harness_name, harness_overrides)
            for model_spec in model_specs
        )
        if gpus < required_gpus:
            console.print(
                f"[red]Error:[/red] GPU count ({gpus}) is insufficient for the configured "
                f"provider instances; at least {required_gpus} GPUs are required "
                "across main and auxiliary providers"
            )
            raise SystemExit(1)

    def _get_effective_harness_config(
        self,
        harness_name: str | None,
        harness_overrides: list[str] | None,
    ):
        """Get harness config with overrides applied.

        Args:
            harness_name: Optional harness preset name.
            harness_overrides: Optional list of harness override strings.

        Returns:
            HarnessConfig with overrides applied, or None if no harness specified.
        """
        if not harness_name:
            return None

        from olmo_eval.cli.run.config import _apply_harness_overrides
        from olmo_eval.harness import get_harness_preset

        try:
            harness_config = get_harness_preset(harness_name)
        except Exception:
            return None

        if harness_overrides:
            harness_config = _apply_harness_overrides(harness_config, harness_overrides)

        return harness_config
