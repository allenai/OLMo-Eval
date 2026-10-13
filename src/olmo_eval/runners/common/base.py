"""Base evaluation runner interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from olmo_eval.common.constants.infrastructure import BEAKER_RESULT_DIR
from olmo_eval.common.types import SamplingParams
from olmo_eval.evals.tasks.common.base import TaskConfig


@dataclass
class BaseEvalRunner(ABC):
    """Base class for all evaluation runners.

    Provides common interface and shared functionality for all runner types.
    Subclasses must implement validate(), print_config(), and run().
    """

    # Core required fields
    task_specs: list[str]
    output_dir: str = BEAKER_RESULT_DIR

    # Per-task overrides
    task_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Experiment metadata
    experiment_name: str | None = None
    experiment_group: str | None = None

    # Output persistence options
    save_predictions: bool = True
    save_requests: bool = True

    # Instance/response inspection options
    inspect_instance: bool = False
    inspect_formatted: bool = False
    inspect_tokens: bool = False
    inspect_response: bool = False
    inspect_request: bool = False

    @abstractmethod
    def validate(self) -> None:
        """Validate runner configuration.

        Raises:
            ValidationError: If configuration is invalid.
        """
        pass

    @abstractmethod
    def print_config(self) -> None:
        """Print runner configuration to console."""
        pass

    @abstractmethod
    def run(self) -> dict[str, Any]:
        """Execute evaluation and return results.

        Returns:
            Dictionary containing evaluation results.
        """
        pass

    def _validate_task_specs(self) -> list[str]:
        """Validate task specs and return list of errors.

        Returns:
            List of error messages (empty if all valid).
        """
        from olmo_eval.evals.suites import suite_exists
        from olmo_eval.evals.tasks.common import get_base_task_name, task_exists

        errors = []
        for spec in self.task_specs:
            base_spec = get_base_task_name(spec)
            if not task_exists(base_spec) and not suite_exists(base_spec):
                errors.append(f"Unknown task or suite: '{base_spec}'")
        return errors

    def _build_task_overrides(self, spec: str) -> tuple[dict[str, Any], dict[str, Any]]:
        """Build task and sampling overrides for a given task spec.

        Args:
            spec: Task specification string.

        Returns:
            Tuple of (task_overrides, sampling_overrides)
        """
        from dataclasses import fields

        task_ovr: dict[str, Any] = {}
        sampling_ovr: dict[str, Any] = {}

        # Get field names from dataclasses
        task_fields = {f.name for f in fields(TaskConfig)}
        sampling_fields = {f.name for f in fields(SamplingParams)}

        # Apply per-task overrides
        per_task = self.task_overrides.get(spec, {})
        for key, value in per_task.items():
            if key in task_fields:
                task_ovr[key] = value
            elif key in sampling_fields:
                sampling_ovr[key] = value

        return task_ovr, sampling_ovr


__all__ = ["BaseEvalRunner"]
