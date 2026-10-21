"""Execution environment and scoring context."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from olmo_eval.inference.base import InferenceProvider
    from olmo_eval.inference.registry import ProviderLookup

    from .process_pool import ProcessPoolManager


@dataclass(frozen=True)
class ExecutionResult:
    """Result from executing code in an environment."""

    success: bool
    output: str = ""
    exit_code: int = 0
    error: str | None = None


@runtime_checkable
class ExecutionEnvironment(Protocol):
    """Protocol for code execution environments."""

    @property
    def is_running(self) -> bool: ...

    async def execute(self, command: str, timeout: float | None = None) -> str: ...

    async def execute_command(
        self, command: str, timeout: float | None = None
    ) -> ExecutionResult: ...

    async def execute_code(
        self, code: str, language: str = "python", timeout: float | None = None
    ) -> ExecutionResult: ...


@runtime_checkable
class StagingExecutionEnvironment(ExecutionEnvironment, Protocol):
    """Execution environment that can place files before running a command.

    Command text reaches the container as a process argument, which the
    operating system caps at a small size. Payloads too large for that limit
    are staged as files instead, in the same environment the command runs in.
    """

    async def execute_with_files(
        self,
        command: str,
        files: Mapping[str, str],
        timeout: float | None = None,
    ) -> ExecutionResult: ...


@dataclass
class ScoringContext:
    """Context passed to scorers during evaluation."""

    execution_env: ExecutionEnvironment | None = None
    scoring_concurrency: int = 8
    inference_pool: ProviderLookup | None = None
    process_pool_manager: ProcessPoolManager | None = None

    @property
    def has_execution_env(self) -> bool:
        return self.execution_env is not None and self.execution_env.is_running

    def get_provider(self, name: str) -> InferenceProvider:
        if self.inference_pool is None:
            raise RuntimeError("No inference pool configured.")
        return self.inference_pool.get(name)

    @property
    def has_process_pool_manager(self) -> bool:
        return self.process_pool_manager is not None
