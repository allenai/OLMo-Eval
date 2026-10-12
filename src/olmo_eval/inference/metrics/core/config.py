"""MetricsConfig for embedding in HarnessConfig."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from olmo_eval.common.repr import hide_unset

# Default subdirectory for metrics files within output_dir
METRICS_SUBDIR = "metrics"
METRICS_FILENAME_SUFFIX = "-inference.jsonl"


class ReporterType(StrEnum):
    """Available metrics reporters."""

    CONSOLE = "console"
    FILE = "file"


@hide_unset()
@dataclass(frozen=True)
class MetricsConfig:
    """Configuration for metrics collection.

    Can be embedded in HarnessConfig for automatic instrumentation,
    or used directly with collect_metrics().

    Core metadata fields identify the run each batch belongs to:
    - experiment_id, experiment_name, experiment_group: identify the eval run
    - model_name, model_hash: identify the model
    - task_name, task_hash: identify the task (optional, set per-batch)

    The `tags` field is for user-defined key-value pairs beyond core metadata.
    """

    enabled: bool = True
    reporters: tuple[str | dict[str, Any], ...] = (ReporterType.FILE,)
    collect_gpu: bool = False

    # vLLM server metrics monitoring
    collect_vllm_server: bool = True  # Poll /metrics endpoint from vLLM server
    vllm_poll_interval_s: float = 10.0  # Polling interval in seconds

    # Output directory (set at runtime, used by file-based reporters)
    output_dir: str | None = None

    # Provider identification
    provider_kind: str | None = None  # e.g., "vllm", "litellm", "hf"

    # Core metadata (mirrors evaluation schema for joins)
    experiment_id: str | None = None
    experiment_name: str | None = None
    experiment_group: str | None = None
    model_name: str | None = None
    model_hash: str | None = None
    task_name: str | None = None
    task_hash: str | None = None
    workspace: str | None = None
    author: str | None = None

    # User-defined tags (for special filtering beyond core metadata)
    tags: dict[str, str] = field(default_factory=dict)

    def get_metrics_path(self) -> str | None:
        """Get the resolved path for metrics file.

        Returns:
            Path like {output_dir}/metrics/{provider_kind}_{model_name}-inference.jsonl,
            or None if output_dir is not set.
        """
        if not self.output_dir:
            return None

        # Build filename from provider and model
        provider = self.provider_kind or "unknown"
        model = self.model_name or "model"

        # Sanitize for filesystem
        safe_provider = provider.replace("/", "_").replace("\\", "_")
        safe_model = model.replace("/", "_").replace("\\", "_")

        filename = f"{safe_provider}_{safe_model}{METRICS_FILENAME_SUFFIX}"

        return os.path.join(self.output_dir, METRICS_SUBDIR, filename)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        d: dict[str, Any] = {
            "enabled": self.enabled,
            "reporters": list(self.reporters),
            "collect_gpu": self.collect_gpu,
            "collect_vllm_server": self.collect_vllm_server,
            "vllm_poll_interval_s": self.vllm_poll_interval_s,
        }
        # Only include non-None fields
        if self.output_dir is not None:
            d["output_dir"] = self.output_dir
        if self.provider_kind is not None:
            d["provider_kind"] = self.provider_kind
        if self.experiment_id is not None:
            d["experiment_id"] = self.experiment_id
        if self.experiment_name is not None:
            d["experiment_name"] = self.experiment_name
        if self.experiment_group is not None:
            d["experiment_group"] = self.experiment_group
        if self.model_name is not None:
            d["model_name"] = self.model_name
        if self.model_hash is not None:
            d["model_hash"] = self.model_hash
        if self.task_name is not None:
            d["task_name"] = self.task_name
        if self.task_hash is not None:
            d["task_hash"] = self.task_hash
        if self.workspace is not None:
            d["workspace"] = self.workspace
        if self.author is not None:
            d["author"] = self.author
        if self.tags:
            d["tags"] = dict(self.tags)
        return d

    def validate(self) -> None:
        """Validate the metrics configuration.

        Raises:
            ValueError: If any reporter name is invalid.
        """
        if not self.enabled:
            return

        from .registry import reporter_registry

        reporter_registry.validate(self.reporters)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MetricsConfig:
        """Create from dictionary."""
        reporters = data.get("reporters", [ReporterType.FILE])
        config = cls(
            enabled=data.get("enabled", True),
            reporters=tuple(reporters),
            collect_gpu=data.get("collect_gpu", False),
            collect_vllm_server=data.get("collect_vllm_server", True),
            vllm_poll_interval_s=data.get("vllm_poll_interval_s", 10.0),
            output_dir=data.get("output_dir"),
            provider_kind=data.get("provider_kind"),
            experiment_id=data.get("experiment_id"),
            experiment_name=data.get("experiment_name"),
            experiment_group=data.get("experiment_group"),
            model_name=data.get("model_name"),
            model_hash=data.get("model_hash"),
            task_name=data.get("task_name"),
            task_hash=data.get("task_hash"),
            workspace=data.get("workspace"),
            author=data.get("author"),
            tags=data.get("tags", {}),
        )
        config.validate()
        return config

    def with_output_dir(self, output_dir: str) -> MetricsConfig:
        """Create a new config with output_dir set."""
        return MetricsConfig(
            enabled=self.enabled,
            reporters=self.reporters,
            collect_gpu=self.collect_gpu,
            collect_vllm_server=self.collect_vllm_server,
            vllm_poll_interval_s=self.vllm_poll_interval_s,
            output_dir=output_dir,
            provider_kind=self.provider_kind,
            experiment_id=self.experiment_id,
            experiment_name=self.experiment_name,
            experiment_group=self.experiment_group,
            model_name=self.model_name,
            model_hash=self.model_hash,
            task_name=self.task_name,
            task_hash=self.task_hash,
            workspace=self.workspace,
            author=self.author,
            tags=self.tags,
        )

    def with_metadata(
        self,
        provider_kind: str | None = None,
        experiment_id: str | None = None,
        experiment_name: str | None = None,
        experiment_group: str | None = None,
        model_name: str | None = None,
        model_hash: str | None = None,
        task_name: str | None = None,
        task_hash: str | None = None,
        workspace: str | None = None,
        author: str | None = None,
    ) -> MetricsConfig:
        """Create a new config with updated metadata fields.

        Only non-None arguments override existing values.
        """
        return MetricsConfig(
            enabled=self.enabled,
            reporters=self.reporters,
            collect_gpu=self.collect_gpu,
            collect_vllm_server=self.collect_vllm_server,
            vllm_poll_interval_s=self.vllm_poll_interval_s,
            output_dir=self.output_dir,
            provider_kind=provider_kind or self.provider_kind,
            experiment_id=experiment_id or self.experiment_id,
            experiment_name=experiment_name or self.experiment_name,
            experiment_group=experiment_group or self.experiment_group,
            model_name=model_name or self.model_name,
            model_hash=model_hash or self.model_hash,
            task_name=task_name or self.task_name,
            task_hash=task_hash or self.task_hash,
            workspace=workspace or self.workspace,
            author=author or self.author,
            tags=self.tags,
        )

    def with_tags(self, **new_tags: str) -> MetricsConfig:
        """Create a new config with additional tags merged in."""
        merged_tags = {**self.tags, **new_tags}
        return MetricsConfig(
            enabled=self.enabled,
            reporters=self.reporters,
            collect_gpu=self.collect_gpu,
            collect_vllm_server=self.collect_vllm_server,
            vllm_poll_interval_s=self.vllm_poll_interval_s,
            output_dir=self.output_dir,
            provider_kind=self.provider_kind,
            experiment_id=self.experiment_id,
            experiment_name=self.experiment_name,
            experiment_group=self.experiment_group,
            model_name=self.model_name,
            model_hash=self.model_hash,
            task_name=self.task_name,
            task_hash=self.task_hash,
            workspace=self.workspace,
            author=self.author,
            tags=merged_tags,
        )
