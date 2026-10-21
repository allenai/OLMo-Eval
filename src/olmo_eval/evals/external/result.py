"""Result types for external evaluations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExternalEvalResult:
    """Result from an external evaluation execution.

    Attributes:
        name: Name of the external evaluation.
        metrics: Dictionary of metric names to values.
        metadata: Additional metadata from the evaluation.
        success: Whether the evaluation completed successfully.
        error: Error message if the evaluation failed.
        duration_seconds: Time taken to run the evaluation.
        raw_output: Raw stdout/stderr from the evaluation process.
    """

    name: str
    metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error: str | None = None
    duration_seconds: float | None = None
    raw_output: str | None = None
    predictions: list[dict[str, Any]] | None = None

    @property
    def num_instances(self) -> int:
        """Number of instances the evaluation scored, or 0 when unknown."""
        if self.predictions:
            return len(self.predictions)
        num_tasks = self.metadata.get("num_tasks")
        if isinstance(num_tasks, int) and not isinstance(num_tasks, bool):
            return num_tasks
        return 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result: dict[str, Any] = {
            "name": self.name,
            "metrics": self.metrics,
            "metadata": self.metadata,
            "success": self.success,
        }
        if self.error is not None:
            result["error"] = self.error
        if self.duration_seconds is not None:
            result["duration_seconds"] = self.duration_seconds
        if self.raw_output is not None:
            result["raw_output"] = self.raw_output
        if self.predictions is not None:
            result["predictions"] = self.predictions
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExternalEvalResult:
        """Create from dictionary."""
        return cls(
            name=data["name"],
            metrics=data.get("metrics", {}),
            metadata=data.get("metadata", {}),
            success=data.get("success", True),
            error=data.get("error"),
            duration_seconds=data.get("duration_seconds"),
            raw_output=data.get("raw_output"),
            predictions=data.get("predictions"),
        )

    @classmethod
    def from_error(cls, name: str, error: str) -> ExternalEvalResult:
        """Create a failed result from an error."""
        return cls(
            name=name,
            success=False,
            error=error,
        )
