"""Nested metric types for agent evaluation.

This module provides structured metric containers for different aspects
of agent evaluation including tool use, trajectory, reliability,
execution, and LLM judge evaluation.
"""

from dataclasses import dataclass, field
from typing import Any

from olmo_eval.common.utils import Serializable


@dataclass(frozen=True, slots=True)
class ToolMetrics(Serializable):
    """Metrics for tool calling accuracy."""

    call_accuracy: float = 0.0
    argument_accuracy: float = 0.0
    sequence_accuracy: float = 0.0
    num_tool_calls: int = 0
    num_correct_calls: int = 0
    num_correct_arguments: int = 0


@dataclass(frozen=True, slots=True)
class TrajectoryMetrics(Serializable):
    """Metrics for trajectory evaluation."""

    response_check: float = 0.0  # Did the tool call sequence match?
    state_check: float = 0.0  # Did the final state match?
    efficiency: float = 0.0  # minimal_steps / actual_steps
    total_steps: int = 0
    minimal_steps: int = 0
    combined_score: float = 0.0  # Both response AND state passed


@dataclass(frozen=True, slots=True)
class ReliabilityMetrics(Serializable):
    """Metrics for multi-trial reliability evaluation."""

    num_trials: int = 0
    pass_at_k: float = 0.0  # At least one success in k trials
    pass_pow_k: float = 0.0  # All k trials succeed
    k: int = 1
    success_count: int = 0
    failure_count: int = 0


@dataclass(frozen=True, slots=True)
class ExecutionMetrics(Serializable):
    """Metrics for task execution."""

    total_runs: int = 0
    successful_runs: int = 0
    processing_errors: int = 0
    instruction_errors: int = 0
    timeout_errors: int = 0
    success_rate: float = 0.0


@dataclass(frozen=True, slots=True)
class JudgeMetrics(Serializable):
    """Metrics from LLM-as-judge evaluation."""

    accuracy: float = 0.0
    not_attempted_rate: float = 0.0
    judge_model: str = ""
    grade_distribution: dict[str, int] = field(default_factory=dict)
    num_evaluated: int = 0
    num_correct: int = 0
    num_incorrect: int = 0
    num_not_attempted: int = 0


@dataclass(frozen=True, slots=True)
class AgentMetrics(Serializable):
    """Container for all agent evaluation metrics.

    This is the top-level metrics container for agent evaluation tasks.
    """

    tool: ToolMetrics | None = None
    trajectory: TrajectoryMetrics | None = None
    reliability: ReliabilityMetrics | None = None
    execution: ExecutionMetrics | None = None
    judge: JudgeMetrics | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentMetrics":
        """Create from dictionary.

        Args:
            data: Dictionary with metric data.

        Returns:
            A new AgentMetrics instance.
        """
        tool = ToolMetrics(**data["tool"]) if "tool" in data else None
        trajectory = TrajectoryMetrics(**data["trajectory"]) if "trajectory" in data else None
        reliability = ReliabilityMetrics(**data["reliability"]) if "reliability" in data else None
        execution = ExecutionMetrics(**data["execution"]) if "execution" in data else None

        judge = None
        if "judge" in data:
            judge_data = data["judge"].copy()
            judge = JudgeMetrics(**judge_data)

        return cls(
            tool=tool,
            trajectory=trajectory,
            reliability=reliability,
            execution=execution,
            judge=judge,
        )
