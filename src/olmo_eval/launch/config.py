"""YAML-based configuration for Beaker launch jobs.

Provides OmegaConf-based configuration loading for composing complex
evaluation experiments from YAML files.

Example config (eval_config.yaml):
    name: eval-llama-suite
    models:
      - llama3.1-8b
      - olmo-2-7b
    tasks:
      - mmlu
      - gsm8k
    cluster: h100
    priority: normal

Example usage:
    config = EvalConfig.from_yaml("eval_config.yaml")
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import MISSING, OmegaConf

from olmo_eval.common.constants.infrastructure import DEFAULT_MAX_GPUS_PER_NODE


def get_model_short_name(model_name: str, alias: str | None = None) -> str:
    """Get a short display name for a model suitable for experiment naming.

    Args:
        model_name: Model name or path.
        alias: Optional short name override.

    Returns:
        Short name suitable for use in experiment names.
    """
    if alias:
        return alias.lower()

    name = model_name.rstrip("/")
    parts = name.split("/")

    # Handle checkpoint paths: s3://bucket/checkpoints/{owner}/{model}/{step}
    # or /weka/oe-training-default/checkpoints/{owner}/{model}/{step}
    # Returns everything after "checkpoints/"
    try:
        checkpoint_idx = parts.index("checkpoints")
        post_checkpoint = parts[checkpoint_idx + 1 :]
        if post_checkpoint:
            return "/".join(post_checkpoint).lower()
    except (ValueError, IndexError):
        pass

    short_name = parts[-1]

    if not short_name or len(short_name) > 32:
        short_name = name[-16:].lstrip("/").lstrip("-").lstrip("_")

    return short_name.lower()


def sanitize_beaker_name(name: str) -> str:
    """Sanitize a string for use in Beaker names.

    Beaker names can only contain letters, digits, periods, dashes, and
    underscores, and cannot start with a dash.
    """
    # Replace invalid characters with underscores
    sanitized = name.replace(":", "_").replace("@", "_")
    # Ensure name doesn't start with a dash
    if sanitized.startswith("-"):
        sanitized = "_" + sanitized[1:]
    return sanitized


def _shorten_task_name(task: str, max_len: int = 8) -> str:
    """Shorten a single task name for use in experiment naming."""
    if "@" in task:
        task = task.split("@")[0]
    if ":" in task:
        task = task.split(":")[0]
    task = task.replace("_challenge", "").replace("_easy", "")
    if len(task) > max_len:
        task = task[:max_len]
    return task.lower().rstrip("_").rstrip("-")


def get_tasks_short_name(tasks: list[str], max_total_len: int = 24) -> str:
    """Generate a short identifier from a list of task names."""
    if not tasks:
        return "notasks"

    if len(tasks) == 1:
        return _shorten_task_name(tasks[0], max_len=max_total_len)

    if len(tasks) <= 3:
        shortened = [_shorten_task_name(t, max_len=8) for t in tasks]
        result = "_".join(shortened)
        if len(result) > max_total_len:
            first = _shorten_task_name(tasks[0], max_len=12)
            return f"{first}_{len(tasks) - 1}more"
        return result

    first = _shorten_task_name(tasks[0], max_len=12)
    return f"{first}_{len(tasks) - 1}more"


@dataclass
class EvalConfig:
    """Configuration for launching Beaker evaluation jobs.

    This dataclass can be loaded from YAML files using OmegaConf.
    Models are specified as simple strings (model names/paths).

    Example:
        models:
          - llama3.1-8b
          - meta-llama/Llama-3.1-70B
    """

    name: str = MISSING
    models: list[str] = MISSING
    tasks: list[str] = MISSING

    cluster: str | None = None
    max_gpus_per_node: int = DEFAULT_MAX_GPUS_PER_NODE

    priority: str = "normal"
    preemptible: bool | None = None
    min_runtime: str | None = None
    timeout: str = "24h"
    retries: int | None = None

    gpus: int = 1

    workspace: str | None = None
    budget: str | None = None
    beaker_image: str | None = None
    description: str | None = None
    groups: list[str] | None = None

    @classmethod
    def from_yaml(cls, path: str | Path, overrides: list[str] | None = None) -> EvalConfig:
        """Load configuration from a YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        file_config = OmegaConf.load(path)
        schema = OmegaConf.structured(cls)
        merged = OmegaConf.merge(schema, file_config)

        if overrides:
            override_config = OmegaConf.from_dotlist(overrides)
            merged = OmegaConf.merge(merged, override_config)

        return OmegaConf.to_object(merged)  # type: ignore[ty:invalid-return-type]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvalConfig:
        """Create configuration from a dictionary."""
        schema = OmegaConf.structured(cls)
        merged = OmegaConf.merge(schema, OmegaConf.create(data))
        return OmegaConf.to_object(merged)  # type: ignore[ty:invalid-return-type]

    def to_yaml(self, path: str | Path | None = None) -> str:
        """Export configuration to YAML."""
        config = OmegaConf.structured(self)
        yaml_str = OmegaConf.to_yaml(config)
        if path:
            Path(path).write_text(yaml_str)
        return yaml_str
