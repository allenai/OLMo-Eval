"""Metrics data schemas.

Frozen dataclasses representing collected metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass(frozen=True)
class RequestMetrics:
    """Metrics for a single inference request."""

    request_id: str
    prompt_tokens: int
    completion_tokens: int
    end_to_end_latency_s: float
    tokens_per_second: float
    time_to_first_token_s: float | None = None
    time_per_output_token_s: float | None = None
    finish_reason: str | None = None
    model: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "request_id": self.request_id,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "end_to_end_latency_s": self.end_to_end_latency_s,
            "tokens_per_second": self.tokens_per_second,
            "time_to_first_token_s": self.time_to_first_token_s,
            "time_per_output_token_s": self.time_per_output_token_s,
            "finish_reason": self.finish_reason,
            "model": self.model,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass(frozen=True)
class BatchMetrics:
    """Aggregated metrics for a batch of requests.

    Core metadata fields identify the run, model and task the batch belongs to.
    """

    # Aggregate statistics
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_prompt_tokens: int
    total_completion_tokens: int
    wall_clock_time_s: float
    output_tokens_per_second: float
    mean_latency_s: float

    # Batch identification
    batch_hash: str | None = None  # Hash of request IDs for reproducibility

    # Core metadata
    experiment_id: str | None = None
    experiment_name: str | None = None
    experiment_group: str | None = None
    model_name: str | None = None
    model_hash: str | None = None
    task_name: str | None = None
    task_hash: str | None = None
    workspace: str | None = None
    author: str | None = None
    provider_kind: str | None = None

    # User-defined tags
    tags: dict[str, str] = field(default_factory=dict)

    # Detailed data
    requests: tuple[RequestMetrics, ...] = ()
    gpu_snapshots: tuple[GPUSnapshot, ...] = ()
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self, include_requests: bool = False) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Args:
            include_requests: If True, include per-request metrics (can be large).
        """
        d: dict[str, Any] = {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "wall_clock_time_s": self.wall_clock_time_s,
            "output_tokens_per_second": self.output_tokens_per_second,
            "mean_latency_s": self.mean_latency_s,
            "timestamp": self.timestamp.isoformat(),
        }
        if include_requests:
            d["requests"] = [r.to_dict() for r in self.requests]
        if self.gpu_snapshots:
            d["gpu_summary"] = self._compute_gpu_summary()
            d["gpu_devices"] = self._group_gpu_snapshots()
        # Include non-None metadata
        if self.batch_hash is not None:
            d["batch_hash"] = self.batch_hash
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
        if self.provider_kind is not None:
            d["provider_kind"] = self.provider_kind
        if self.tags:
            d["tags"] = dict(self.tags)
        return d

    def _compute_gpu_summary(self) -> dict[str, Any] | None:
        """Compute aggregate GPU statistics across all snapshots."""
        if not self.gpu_snapshots:
            return None

        device_ids = set(s.device_id for s in self.gpu_snapshots)
        utilizations = [s.utilization_pct for s in self.gpu_snapshots]
        memory_used = [s.memory_used_mb for s in self.gpu_snapshots]
        power_values = [s.power_watts for s in self.gpu_snapshots if s.power_watts is not None]

        return {
            "device_count": len(device_ids),
            "sample_count": len(self.gpu_snapshots),
            "avg_utilization_pct": sum(utilizations) / len(utilizations),
            "max_utilization_pct": max(utilizations),
            "avg_memory_used_mb": sum(memory_used) / len(memory_used),
            "max_memory_used_mb": max(memory_used),
            "avg_power_watts": sum(power_values) / len(power_values) if power_values else None,
        }

    def _group_gpu_snapshots(self) -> list[dict[str, Any]]:
        """Group GPU snapshots by device, separating static and dynamic fields."""
        from collections import defaultdict

        # Group snapshots by device_id
        by_device: dict[int, list[GPUSnapshot]] = defaultdict(list)
        for snapshot in self.gpu_snapshots:
            by_device[snapshot.device_id].append(snapshot)

        result = []
        for device_id in sorted(by_device.keys()):
            snapshots = by_device[device_id]
            # Use first snapshot for static fields
            first = snapshots[0]
            device_data: dict[str, Any] = {
                "device_id": device_id,
                "name": first.name,
                "memory_total_mb": first.memory_total_mb,
                "samples": [
                    {
                        "utilization_pct": s.utilization_pct,
                        "memory_used_mb": s.memory_used_mb,
                        "temperature_c": s.temperature_c,
                        "power_watts": s.power_watts,
                        "timestamp": s.timestamp.isoformat(),
                    }
                    for s in snapshots
                ],
            }
            result.append(device_data)
        return result


@dataclass(frozen=True)
class GPUSnapshot:
    """GPU utilization snapshot at a point in time."""

    device_id: int
    name: str
    utilization_pct: float
    memory_used_mb: float
    memory_total_mb: float
    temperature_c: float | None = None
    power_watts: float | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "device_id": self.device_id,
            "name": self.name,
            "utilization_pct": self.utilization_pct,
            "memory_used_mb": self.memory_used_mb,
            "memory_total_mb": self.memory_total_mb,
            "temperature_c": self.temperature_c,
            "power_watts": self.power_watts,
            "timestamp": self.timestamp.isoformat(),
        }
