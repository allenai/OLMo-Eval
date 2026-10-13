"""Configuration building for the run command."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from rich.console import Console

from olmo_eval.common.types import ProviderKind
from olmo_eval.evals.suites import get_suite, suite_exists
from olmo_eval.harness.config import HarnessConfig, ProviderConfig

console = Console()


def _parse_override_value(value: str) -> Any:
    """Parse an override value, supporting JSON, OmegaConf lists, bool, int, float, and string."""
    import json

    # Try JSON first (for dicts and arrays with quoted strings)
    if value.startswith("{") or value.startswith("["):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass

        # Try OmegaConf-style list: [item1,item2] without quotes
        if value.startswith("[") and value.endswith("]"):
            inner = value[1:-1].strip()
            if inner:
                # Split by comma, but be careful with URLs containing commas (rare)
                items = [item.strip() for item in inner.split(",")]
                return items
            return []

    # Parse bool, int, float, or string
    if value.lower() == "true":
        return True
    elif value.lower() == "false":
        return False
    else:
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return value


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge override into base dict."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def _merge_dict_into_list_items(items: list[Any], override: dict[str, Any], key_path: str) -> None:
    """Deep-merge a dict override into every dict item in a list."""
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(
                f"Invalid override path '{key_path}': cannot merge dict into list item "
                f"{idx} of type {type(item).__name__}"
            )
        _deep_merge(item, copy.deepcopy(override))


def _apply_dotlist_overrides(base_dict: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply dotlist overrides to a dictionary, handling list indices.

    Unlike OmegaConf.from_dotlist which treats numeric keys as dict keys,
    this function treats them as list indices (e.g., sandboxes.0.mode=modal).

    Supports JSON values for nested structures:
        sandboxes.0='{"mode":"modal"}'
        sandboxes='{"mode":"modal","instances":64}'

    A dict override applied directly to the top-level sandboxes key is treated
    specially:
    - most fields are deep-merged into each existing sandbox config
    - the special fields ``instances`` and ``min_instances`` are peeled off into
      the shared ``sandbox_pool_instances`` budget and
      ``sandbox_pool_min_instances`` minimum instead of being applied per sandbox

    Args:
        base_dict: The base dictionary. It is not modified.
        overrides: List of dotlist strings (e.g., ["sandboxes.0.mode=modal"]).

    Returns:
        A new dictionary with the overrides applied.
    """
    base_dict = copy.deepcopy(base_dict)
    for override in overrides:
        if "=" not in override:
            continue
        key_path, value = override.split("=", 1)
        keys = key_path.split(".")
        parsed_value = _parse_override_value(value)

        # Navigate to the target location and set the value
        target = base_dict
        for i, key in enumerate(keys[:-1]):
            path_so_far = ".".join(keys[: i + 1])
            # Check if key is a numeric index for list access
            if key.isdigit():
                idx = int(key)
                if not isinstance(target, list):
                    raise ValueError(
                        f"Invalid override path '{key_path}': '{path_so_far}' uses numeric "
                        f"index but target is {type(target).__name__}, not list"
                    )
                if idx >= len(target):
                    raise ValueError(
                        f"Invalid override path '{key_path}': index {idx} out of bounds "
                        f"for list at '{'.'.join(keys[:i])}' (length {len(target)})"
                    )
                target = target[idx]
            else:
                if not isinstance(target, dict):
                    raise ValueError(
                        f"Invalid override path '{key_path}': "
                        f"'{path_so_far}' expects dict but found {type(target).__name__}"
                    )
                if key not in target or target[key] is None:
                    # Create nested dict if needed
                    target[key] = {}
                target = target[key]

        # Set the final value
        final_key = keys[-1]
        if final_key.isdigit():
            idx = int(final_key)
            if not isinstance(target, list):
                raise ValueError(
                    f"Invalid override path '{key_path}': final key '{final_key}' is numeric "
                    f"but target is {type(target).__name__}, not list"
                )
            if idx >= len(target):
                raise ValueError(
                    f"Invalid override path '{key_path}': "
                    f"index {idx} out of bounds for list (length {len(target)})"
                )
            if isinstance(parsed_value, dict) and isinstance(target[idx], dict):
                _deep_merge(target[idx], parsed_value)
            else:
                target[idx] = parsed_value
        elif isinstance(target, dict):
            if isinstance(parsed_value, dict) and isinstance(target.get(final_key), dict):
                _deep_merge(target[final_key], parsed_value)
            elif (
                final_key == "sandboxes"
                and isinstance(parsed_value, dict)
                and isinstance(target.get(final_key), list)
            ):
                override_dict = copy.deepcopy(parsed_value)
                if "instances" in override_dict:
                    base_dict["sandbox_pool_instances"] = override_dict.pop("instances")
                if "min_instances" in override_dict:
                    base_dict["sandbox_pool_min_instances"] = override_dict.pop("min_instances")
                if override_dict:
                    _merge_dict_into_list_items(target[final_key], override_dict, key_path)
            else:
                target[final_key] = parsed_value
        else:
            raise ValueError(
                f"Invalid override path '{key_path}': "
                f"cannot set key '{final_key}' on {type(target).__name__}"
            )

    return base_dict


def provider_kind_override(overrides: list[str]) -> str | None:
    """Return the provider kind set by a CLI override such as ``provider.kind=vllm_server``."""
    provider_overrides = [
        o for o in overrides if o.split("=", 1)[0].strip() in ("provider", "provider.kind")
    ]
    if not provider_overrides:
        return None
    provider = _apply_dotlist_overrides({"provider": {}}, provider_overrides)["provider"]
    kind = provider.get("kind") if isinstance(provider, dict) else None
    return str(kind) if kind else None


def merge_model_provider(
    harness_config: HarnessConfig,
    provider_config: ProviderConfig,
    overrides: list[str] | None = None,
) -> HarnessConfig:
    """Merge a model's provider into a harness config, honoring a CLI provider.kind override.

    A harness preset's own provider kind yields to the model preset's kind, but an
    explicit ``provider.kind`` override always wins. Scaffolds talk to the model through
    an OpenAI client, so a model preset's in-process vllm kind runs as vllm_server under
    a scaffold.
    """
    merged = harness_config.merge_provider(provider_config)
    kind = provider_kind_override(overrides or [])
    if kind is None and merged.scaffold and merged.provider.kind == ProviderKind.VLLM:
        kind = ProviderKind.VLLM_SERVER
    return merged.with_provider_overrides(kind=kind) if kind else merged


def _apply_harness_overrides(harness_config: HarnessConfig, overrides: list[str]) -> HarnessConfig:
    """Apply CLI overrides to harness config.

    Args:
        harness_config: Base HarnessConfig to modify.
        overrides: List of dotlist override strings (e.g., ["sandbox.mode=docker"]).
            Supports list indices like "sandboxes.0.mode=modal".
            Supports JSON values like 'sandboxes.0={"mode":"modal"}'.
            Supports shared sandbox overrides like
            'sandboxes={"mode":"modal","instances":64,"min_instances":24}'
            where non-pool fields are applied to each sandbox config and
            instances/min_instances set the shared sandbox pool budget/minimum.

    Returns:
        New HarnessConfig with overrides applied.
    """
    harness_dict = _apply_dotlist_overrides(harness_config.to_dict(), overrides)
    return HarnessConfig.from_dict(harness_dict)


@dataclass
class RunConfig:
    """Parsed and validated configuration for an evaluation run.

    This is the fully-formed configuration after CLI parsing and override
    application.
    """

    harness_config: HarnessConfig
    task_specs: list[str] = field(default_factory=list, repr=False)
    # Per-task overrides (task_spec -> overrides dict)
    # These are applied when preparing tasks since tasks are loaded by spec
    task_overrides: dict[str, dict[str, Any]] = field(default_factory=dict, repr=False)

    output_dir: str = "/tmp/results/"

    # Experiment metadata
    experiment_name: str | None = None
    experiment_group: str | None = None

    # Output options
    save_predictions: bool = True
    save_requests: bool = True

    # Debug/inspection options
    inspect_instance: bool = False
    inspect_formatted: bool = False
    inspect_tokens: bool = False
    inspect_response: bool = False
    inspect_request: bool = False

    @property
    def model_name(self) -> str:
        """Get the model name from the harness config."""
        return self.harness_config.provider.model

    @property
    def provider_config(self) -> ProviderConfig:
        """Get the provider config from the harness config."""
        return self.harness_config.provider

    @property
    def tasks(self) -> list:
        """Resolve task_specs to TaskConfig objects for display."""
        from olmo_eval.common.configs import expand_tasks
        from olmo_eval.evals.tasks.common import get_task

        return [get_task(spec).config for spec in expand_tasks(self.task_specs)]

    def __rich_repr__(self):
        """Rich repr that shows resolved tasks instead of task_specs."""
        from dataclasses import fields

        for f in fields(self):
            if f.name == "task_specs":
                yield "tasks", self.tasks
            elif not f.repr:
                continue
            else:
                yield f.name, getattr(self, f.name)


class RunConfigBuilder:
    """Builds and validates run configuration from CLI arguments."""

    def __init__(
        self,
        model: str,
        task: tuple[str, ...],
        output_dir: str,
        num_gpus: int = 1,
        parallelism: int = 1,
        experiment_name: str | None = None,
        experiment_group: str | None = None,
        save_predictions: bool = True,
        save_requests: bool = True,
        inspect_instance: bool = False,
        inspect_formatted: bool = False,
        inspect_tokens: bool = False,
        inspect_response: bool = False,
        inspect_request: bool = False,
        cli_task_overrides: dict[str, list[str]] | None = None,
        harness_preset: str | None = None,
        harness_config_path: str | None = None,
        cli_harness_overrides: list[str] | None = None,
        force_download_model: bool = False,
    ):
        """Initialize the builder with raw CLI arguments.

        Args:
            model: Model name/path from -m flag.
            task: Tuple of task specs from -t flags.
            output_dir: Output directory for results.
            cli_task_overrides: Per-task overrides from -o flags (task_spec -> [overrides]).
            harness_preset: Name of a harness preset (e.g., "search").
            harness_config_path: Path to a harness config YAML/JSON file.
            cli_harness_overrides: Harness overrides from -o flags after --harness.
            force_download_model: Force-refresh Hugging Face model/tokenizer cache before loading.
            ... (other standard args)
        """
        self.model = model
        self.task = task
        self.output_dir = output_dir
        self.num_gpus = num_gpus
        self.parallelism = parallelism
        self.experiment_name = experiment_name
        self.experiment_group = experiment_group
        self.save_predictions = save_predictions
        self.save_requests = save_requests
        self.inspect_instance = inspect_instance
        self.inspect_formatted = inspect_formatted
        self.inspect_tokens = inspect_tokens
        self.inspect_response = inspect_response
        self.inspect_request = inspect_request
        self.cli_task_overrides = cli_task_overrides or {}
        self.harness_preset = harness_preset
        self.harness_config_path = harness_config_path
        self.cli_harness_overrides = cli_harness_overrides or []
        self.force_download_model = force_download_model

    def build(self) -> RunConfig:
        """Parse inputs and build configuration.

        Returns:
            RunConfig with parsed and validated settings.
        """
        # Build task specs and overrides from CLI -o flags. Suite-level overrides
        # are propagated to each constituent task spec.
        task_specs: list[str] = list(self.task)
        resolved_cli_overrides: dict[str, list[str]] = {}
        for task_spec, cli_overrides in self.cli_task_overrides.items():
            if not cli_overrides:
                continue
            if "@" in task_spec:
                base_spec, priority = task_spec.rsplit("@", 1)
                priority_suffix = f"@{priority}"
            else:
                base_spec = task_spec
                priority_suffix = ""
            if suite_exists(base_spec):
                for child_spec in get_suite(base_spec).expand():
                    key = f"{child_spec}{priority_suffix}"
                    resolved_cli_overrides.setdefault(key, []).extend(cli_overrides)
            else:
                resolved_cli_overrides.setdefault(task_spec, []).extend(cli_overrides)

        task_overrides: dict[str, dict[str, Any]] = {}
        for task_spec, cli_overrides in resolved_cli_overrides.items():
            task_overrides[task_spec] = _apply_dotlist_overrides({}, cli_overrides)

        # Resolve harness configuration with provider config built in
        harness_config = self._resolve_harness_config(self.model)

        return RunConfig(
            harness_config=harness_config,
            task_specs=task_specs,
            task_overrides=task_overrides,
            output_dir=self.output_dir,
            experiment_name=self.experiment_name,
            experiment_group=self.experiment_group,
            save_predictions=self.save_predictions,
            save_requests=self.save_requests,
            inspect_instance=self.inspect_instance,
            inspect_formatted=self.inspect_formatted,
            inspect_tokens=self.inspect_tokens,
            inspect_response=self.inspect_response,
            inspect_request=self.inspect_request,
        )

    def _resolve_harness_config(self, model_name: str) -> HarnessConfig:
        """Resolve harness configuration with provider fully configured.

        Args:
            model_name: Model name/path.

        Returns:
            HarnessConfig with provider configured.

        Raises:
            SystemExit: If harness preset or config file is invalid.
        """
        if self.harness_preset and self.harness_config_path:
            console.print("[red]Error:[/red] Cannot specify both --harness and --harness-config")
            raise SystemExit(1)

        harness_config: HarnessConfig

        if self.harness_preset:
            try:
                from olmo_eval.harness import get_harness_preset

                harness_config = get_harness_preset(self.harness_preset)
            except ValueError as e:
                console.print(f"[red]Error:[/red] {e}")
                raise SystemExit(1) from None

        elif self.harness_config_path:
            import json

            import yaml

            try:
                with open(self.harness_config_path) as f:
                    if self.harness_config_path.endswith(".json"):
                        config_dict = json.load(f)
                    else:
                        config_dict = yaml.safe_load(f)

                harness_config = HarnessConfig.from_dict(config_dict)
                console.print(f"[dim]Using harness config: {self.harness_config_path}[/dim]")
            except FileNotFoundError:
                console.print(
                    f"[red]Error:[/red] Harness config file not found: {self.harness_config_path}"
                )
                raise SystemExit(1) from None
            except (json.JSONDecodeError, yaml.YAMLError) as e:
                console.print(f"[red]Error:[/red] Invalid harness config file: {e}")
                raise SystemExit(1) from None
            except (KeyError, TypeError) as e:
                console.print(f"[red]Error:[/red] Invalid harness config format: {e}")
                raise SystemExit(1) from None

        else:
            harness_config = HarnessConfig(name="default")

        # Apply CLI overrides to harness config
        if self.cli_harness_overrides:
            harness_config = _apply_harness_overrides(harness_config, self.cli_harness_overrides)

        from olmo_eval.common.configs import get_provider_config

        provider_config = get_provider_config(model_name)
        harness_config = merge_model_provider(
            harness_config, provider_config, self.cli_harness_overrides
        )
        if self.force_download_model:
            harness_config = harness_config.with_provider_overrides(force_download=True)

        return harness_config
