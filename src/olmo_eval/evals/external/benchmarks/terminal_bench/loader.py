"""Terminal-Bench task loader."""

from __future__ import annotations

import logging
import re
import subprocess
import tomllib
from pathlib import Path
from typing import Any

from .task import TerminalBenchTask

logger = logging.getLogger(__name__)

_SIZE_UNITS_MB = {"k": 1 / 1024, "kb": 1 / 1024, "m": 1, "mb": 1, "g": 1024, "gb": 1024}


def parse_size_mb(value: Any, default: int) -> int:
    """Read a size given in megabytes or as a string like ``"2G"``.

    Args:
        value: An integer number of megabytes, a string with a unit suffix, or None.
        default: Value to use when nothing usable is given.

    Returns:
        The size in megabytes.
    """
    if value is None:
        return default
    if isinstance(value, bool):
        return default
    if isinstance(value, int | float):
        return int(value)
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([a-zA-Z]*)\s*", str(value))
    if not match:
        return default
    number = float(match.group(1))
    unit = match.group(2).lower() or "mb"
    factor = _SIZE_UNITS_MB.get(unit)
    if factor is None:
        return default
    return int(number * factor)


def normalize_label(value: Any) -> str:
    """Lower-case a difficulty or category label and unify its separators.

    Task authors write labels inconsistently (``hard``, ``Hard``, ``data_science``,
    ``data-science``), which would otherwise split one group across metrics.
    """
    text = str(value).strip().lower().replace("_", "-").replace(" ", "-")
    return text or "unknown"


class TerminalBenchLoader:
    """Loads Terminal-Bench style tasks from a git repository.

    The class attributes describe the Terminal-Bench 2.1 release. Pass other
    values to load another dataset in the same task format.
    """

    REPO_URL = "https://github.com/harbor-framework/terminal-bench-2-1.git"
    DEFAULT_REF = "7131e4375048a0e408a8fb404b5f499d726b695b"
    DATASET_VERSION = "2.1"

    def __init__(
        self,
        repo_url: str | None = None,
        default_ref: str | None = None,
        dataset_version: str | None = None,
    ) -> None:
        self.repo_url = repo_url or self.REPO_URL
        self.default_ref = default_ref or self.DEFAULT_REF
        self.dataset_version = dataset_version or self.DATASET_VERSION

    def ensure_repo(self, target_dir: Path, ref: str | None = None) -> Path:
        """Clone or update the Terminal-Bench repository.

        Args:
            target_dir: Directory to clone/update the repo in.
            ref: Git ref to checkout (commit, branch, or tag).

        Returns:
            Path to the repository.
        """
        ref = ref or self.default_ref

        if (target_dir / ".git").exists():
            logger.info(f"Updating Terminal-Bench repo at {target_dir}")
            subprocess.run(
                ["git", "-C", str(target_dir), "fetch", "origin"],
                check=True,
                capture_output=True,
            )
            subprocess.run(
                ["git", "-C", str(target_dir), "checkout", ref],
                check=True,
                capture_output=True,
            )
        else:
            logger.info(f"Cloning Terminal-Bench repo to {target_dir}")
            target_dir.parent.mkdir(parents=True, exist_ok=True)
            # Clone without --depth to support checking out specific commits
            subprocess.run(
                ["git", "clone", self.repo_url, str(target_dir)],
                check=True,
                capture_output=True,
            )
            subprocess.run(
                ["git", "-C", str(target_dir), "checkout", ref],
                check=True,
                capture_output=True,
            )
        return target_dir

    @staticmethod
    def tasks_root(repo_dir: Path) -> Path:
        """Return the directory holding the task directories.

        Terminal-Bench 2.1 keeps tasks under ``tasks/``; 2.0 kept them at the
        repository root.
        """
        tasks_dir = repo_dir / "tasks"
        return tasks_dir if tasks_dir.is_dir() else repo_dir

    def load_tasks(
        self,
        repo_dir: Path,
        task_ids: list[str] | None = None,
    ) -> list[TerminalBenchTask]:
        """Load tasks from the Terminal-Bench repository.

        Args:
            repo_dir: Path to the repository.
            task_ids: Optional list of task IDs to load (default: all).

        Returns:
            List of TerminalBenchTask instances.
        """
        tasks = []

        for task_dir in sorted(self.tasks_root(repo_dir).iterdir()):
            if not task_dir.is_dir():
                continue
            if not (task_dir / "task.toml").exists():
                continue
            if task_ids and task_dir.name not in task_ids:
                continue

            try:
                task = self._load_task(task_dir)
                tasks.append(task)
                logger.debug(f"Loaded task: {task.task_id}")
            except Exception as e:
                logger.warning(f"Failed to load task {task_dir.name}: {e}")

        task_ids = [t.task_id for t in tasks]
        logger.info(f"Loaded {len(tasks)} tasks: {task_ids}")
        return tasks

    @staticmethod
    def _read_files(root: Path) -> dict[str, bytes]:
        """Read every file under a directory, keyed by path relative to it."""
        files: dict[str, bytes] = {}
        if root.exists():
            for path in sorted(root.rglob("*")):
                if path.is_file():
                    files[str(path.relative_to(root))] = path.read_bytes()
        return files

    def _load_task(self, task_dir: Path) -> TerminalBenchTask:
        """Load a single task from its directory.

        Args:
            task_dir: Path to the task directory.

        Returns:
            A TerminalBenchTask instance.
        """
        config = tomllib.loads((task_dir / "task.toml").read_text())

        # Parse WORKDIR from Dockerfile
        dockerfile_path = task_dir / "environment" / "Dockerfile"
        workdir = "/app"  # Default
        if dockerfile_path.exists():
            for line in dockerfile_path.read_text().splitlines():
                line = line.strip()
                if line.startswith("WORKDIR"):
                    parts = line.split(None, 1)
                    if len(parts) > 1:
                        workdir = parts[1].strip().strip('"').strip("'")

        instruction_path = task_dir / "instruction.md"
        instruction = instruction_path.read_text() if instruction_path.exists() else ""

        test_files = self._read_files(task_dir / "tests")
        solution_files = self._read_files(task_dir / "solution")
        solution_script = solution_files.get("solve.sh", b"").decode("utf-8", errors="replace")

        env_config = config.get("environment", {})
        agent_config = config.get("agent", {})
        verifier_config = config.get("verifier", {})
        metadata = config.get("metadata", {})

        image = env_config.get("docker_image") or ""
        environment_dir = task_dir / "environment"
        build_context = str(environment_dir) if not image and dockerfile_path.exists() else None
        if not image and build_context is None:
            raise ValueError("task declares neither a docker_image nor a Dockerfile")

        return TerminalBenchTask(
            task_id=task_dir.name,
            image=image,
            working_dir=workdir,
            instruction=instruction,
            agent_timeout=float(agent_config.get("timeout_sec", 900.0)),
            verifier_timeout=float(verifier_config.get("timeout_sec", 900.0)),
            test_files=test_files,
            solution_script=solution_script,
            difficulty=normalize_label(metadata.get("difficulty", "unknown")),
            category=normalize_label(metadata.get("category", "unknown")),
            build_context=build_context,
            build_timeout=float(env_config.get("build_timeout_sec", 600.0)),
            cpus=int(env_config.get("cpus", 1)),
            memory_mb=parse_size_mb(
                env_config.get("memory_mb", env_config.get("memory")), default=2048
            ),
            storage_mb=parse_size_mb(
                env_config.get("storage_mb", env_config.get("storage")), default=10240
            ),
            allow_internet=bool(env_config.get("allow_internet", True)),
            solution_files=solution_files,
        )
