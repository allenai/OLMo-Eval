"""Terminal-Bench task dataclass."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TerminalBenchTask:
    """A single Terminal-Bench task.

    Attributes:
        task_id: Unique identifier for the task.
        image: Docker image for the task environment.
        working_dir: Working directory inside the container (from Dockerfile WORKDIR).
        instruction: Task instruction content.
        agent_timeout: Wall-clock budget for the agent in seconds.
        verifier_timeout: Timeout for verification in seconds.
        test_files: Mapping of relative paths to file content (as bytes).
        solution_script: Content of the solution script for oracle mode.
        difficulty: Task difficulty level.
        category: Task category.
        cpus: CPU limit for the task container.
        memory_mb: Memory limit for the task container in megabytes.
        storage_mb: Storage allowance for the task container in megabytes.
        allow_internet: Whether the task expects network access.
        solution_files: Mapping of relative paths under the solution directory
            to file content, for oracle runs that need more than the script.
        build_context: Directory holding the Dockerfile to build the task image
            from, for tasks that ship no prebuilt image.
        build_timeout: Seconds allowed for building the task image.
    """

    task_id: str
    image: str
    working_dir: str
    instruction: str
    agent_timeout: float
    verifier_timeout: float
    test_files: dict[str, bytes]
    solution_script: str
    difficulty: str
    category: str
    cpus: int = 1
    memory_mb: int = 2048
    storage_mb: int = 10240
    allow_internet: bool = True
    solution_files: dict[str, bytes] = field(default_factory=dict)
    build_context: str | None = None
    build_timeout: float = 600.0

    @property
    def needs_build(self) -> bool:
        """Whether the task image must be built rather than pulled."""
        return not self.image and self.build_context is not None
