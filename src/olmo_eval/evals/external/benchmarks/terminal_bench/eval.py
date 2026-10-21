"""Terminal-Bench external evaluation."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from olmo_eval.common.types import LMRequest, RequestType, SamplingParams
from olmo_eval.common.types.trajectory import AgentTrajectory
from olmo_eval.evals.external.base import ExternalEval
from olmo_eval.evals.external.network import get_docker_network_args
from olmo_eval.evals.external.result import ExternalEvalResult
from olmo_eval.harness.sandbox.config import ContainerRuntime, SandboxConfig, SandboxMode
from olmo_eval.harness.sandbox.image import build_task_image, get_swerex_image

from .loader import TerminalBenchLoader
from .task import TerminalBenchTask
from .verifier import SOLUTION_DIR, TerminalBenchVerifier

if TYPE_CHECKING:
    from olmo_eval.harness.sandbox import SandboxManager
    from olmo_eval.inference.base import InferenceProvider

logger = logging.getLogger(__name__)


DEFAULT_SCAFFOLD = "vanillux"

#: Scaffolds this eval can drive, with the tools each one gets.
SCAFFOLD_TOOLS: dict[str, tuple[str, ...]] = {
    "vanillux": ("bash",),
    "openai_agents": ("execute_bash_session", "submit"),
}

#: Extra wall-clock time, past a task's agent timeout, before an agent run is
#: cut off from outside. The scaffold stops itself at the timeout between steps;
#: this bounds a step that is already in flight.
AGENT_TIMEOUT_GRACE = 300.0

#: Containers take longer to come up when a task image has to be pulled first.
SANDBOX_STARTUP_TIMEOUT = 180.0

#: Podman flags for task containers. Rootless Podman maps the container's users
#: into a namespace whose default size leaves common system ids such as
#: ``nogroup`` unmapped, which breaks ``apt-get`` in task images and test
#: scripts. A namespace of 65536 ids covers them.
PODMAN_DOCKER_ARGS: tuple[str, ...] = ("--userns=auto:size=65536",)

OPENAI_AGENTS_SYSTEM_PROMPT = """\
You are an AI assistant helping complete tasks in a Linux terminal.
You have access to the following tools:
- execute_bash_session(command): Execute a bash command in a persistent shell session
- submit(): Call when you have completed the task

Focus on completing the task described in the instructions. Work step by step,
checking your progress as you go. When you believe the task is complete, call
the submit() tool.
"""

_TRUE_VALUES = (True, "true", "True", "1", 1)


@dataclass(frozen=True)
class TerminalBenchDataset:
    """Where a Terminal-Bench style dataset lives and how the eval names it.

    Attributes:
        name: The eval's registered name.
        description: One-line description shown by the CLI.
        repo_url: Git repository holding the tasks.
        repo_ref: Commit, tag or branch to evaluate.
        version: Dataset version recorded in results.
        cache_dir: Where a fresh clone is kept between runs.
        image_prefix: Name prefix for task images built from a Dockerfile.
    """

    name: str
    description: str
    repo_url: str
    repo_ref: str
    version: str
    cache_dir: str
    image_prefix: str


TERMINAL_BENCH_DATASET = TerminalBenchDataset(
    name="terminal_bench_2",
    description=(
        f"Evaluates LLM agents on the {TerminalBenchLoader.DATASET_VERSION} release "
        "of Terminal-Bench, 89 diverse terminal tasks"
    ),
    repo_url=TerminalBenchLoader.REPO_URL,
    repo_ref=TerminalBenchLoader.DEFAULT_REF,
    version=TerminalBenchLoader.DATASET_VERSION,
    cache_dir="/tmp/terminal-bench-2-1-cache",
    image_prefix="tb-task",
)


@dataclass
class TerminalBenchArgs:
    """Arguments for the Terminal-Bench evaluation."""

    task_ids: list[str] | None = None
    repo_path: str | None = None
    repo_ref: str = TerminalBenchLoader.DEFAULT_REF
    max_concurrency: int = 1
    max_turns: int = 64
    n_attempts: int = 1
    oracle: bool = False
    sandbox_mode: str = "docker"
    enable_compaction: bool = True
    scaffold: str = DEFAULT_SCAFFOLD
    command_timeout: float = 120.0
    max_format_errors: int = 64
    resource_limits: bool = True
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 16384

    @classmethod
    def from_dict(
        cls, data: dict[str, Any], default_ref: str = TerminalBenchLoader.DEFAULT_REF
    ) -> TerminalBenchArgs:
        task_ids = data.get("task_ids")
        if isinstance(task_ids, str):
            task_ids = [t.strip() for t in task_ids.split(",") if t.strip()]
        return cls(
            task_ids=task_ids,
            repo_path=data.get("repo_path"),
            repo_ref=data.get("repo_ref", default_ref),
            max_concurrency=int(data.get("max_concurrency", 1)),
            max_turns=int(data.get("max_turns", 64)),
            n_attempts=max(1, int(data.get("n_attempts", 1))),
            oracle=data.get("oracle", False) in _TRUE_VALUES,
            sandbox_mode=data.get("sandbox_mode", "docker"),
            enable_compaction=data.get("enable_compaction", True) in _TRUE_VALUES,
            scaffold=data.get("scaffold", DEFAULT_SCAFFOLD),
            command_timeout=float(data.get("command_timeout", 120.0)),
            max_format_errors=int(data.get("max_format_errors", 64)),
            resource_limits=data.get("resource_limits", True) in _TRUE_VALUES,
            temperature=float(data.get("temperature", 0.7)),
            top_p=float(data.get("top_p", 0.95)),
            max_tokens=int(data.get("max_tokens", 16384)),
        )

    @property
    def sampling_params(self) -> SamplingParams:
        return SamplingParams(
            temperature=self.temperature, top_p=self.top_p, max_tokens=self.max_tokens
        )


@dataclass
class TaskResult:
    """Result of one trial: a single attempt at a Terminal-Bench task."""

    task_id: str
    reward: float
    trajectory: AgentTrajectory
    completion_reason: str
    agent_duration: float
    verification_output: str
    verification_exit_code: int
    error: str | None = None
    difficulty: str = "unknown"
    category: str = "unknown"
    attempt: int = 0
    agent_steps: int = 0

    @property
    def errored(self) -> bool:
        """Whether the trial failed for a reason other than the agent's work."""
        return self.error is not None

    @property
    def native_id(self) -> str:
        return self.task_id if self.attempt == 0 else f"{self.task_id}__{self.attempt}"


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimate for a task with ``c`` passes out of ``n`` attempts."""
    if n <= 0 or k <= 0:
        return 0.0
    k = min(k, n)
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def compute_metrics(results: list[TaskResult], n_attempts: int) -> dict[str, float]:
    """Aggregate trial results into the eval's metrics.

    ``pass_rate`` is the mean reward over every trial, counting an errored
    trial as 0. ``pass_rate_adjusted`` leaves errored trials out, so it reads
    as what the agent scored where the harness worked. With several attempts
    per task, ``pass@k`` is the unbiased estimate averaged over tasks, and the
    spread of the per-attempt pass rates gives ``pass_rate_std`` and
    ``pass_rate_sem``.
    """
    rewards = [r.reward for r in results]
    num_trials = len(results)
    num_errors = sum(1 for r in results if r.errored)
    valid = [r.reward for r in results if not r.errored]

    metrics: dict[str, float] = {
        "pass_rate": sum(rewards) / num_trials if num_trials else 0.0,
        "pass_rate_adjusted": sum(valid) / len(valid) if valid else 0.0,
        "error_rate": num_errors / num_trials if num_trials else 0.0,
        "num_tasks": float(len({r.task_id for r in results})),
        "num_trials": float(num_trials),
        "num_passed": float(sum(1 for r in results if r.reward >= 1.0)),
        "num_errors": float(num_errors),
        "num_agent_timeouts": float(sum(1 for r in results if r.completion_reason == "timeout")),
    }

    by_task: dict[str, list[float]] = {}
    for r in results:
        by_task.setdefault(r.task_id, []).append(r.reward)

    if n_attempts > 1 and by_task:
        for k in sorted({1, n_attempts}):
            estimates = [
                pass_at_k(len(task_rewards), sum(1 for v in task_rewards if v >= 1.0), k)
                for task_rewards in by_task.values()
            ]
            metrics[f"pass@{k}"] = sum(estimates) / len(estimates)

        by_attempt: dict[int, list[float]] = {}
        for r in results:
            by_attempt.setdefault(r.attempt, []).append(r.reward)
        attempt_means = [sum(v) / len(v) for v in by_attempt.values() if v]
        if len(attempt_means) > 1:
            mean = sum(attempt_means) / len(attempt_means)
            variance = sum((m - mean) ** 2 for m in attempt_means) / (len(attempt_means) - 1)
            std = math.sqrt(variance)
            metrics["pass_rate_std"] = std
            metrics["pass_rate_sem"] = std / math.sqrt(len(attempt_means))

    by_difficulty: dict[str, list[float]] = {}
    by_category: dict[str, list[float]] = {}
    for r in results:
        by_difficulty.setdefault(r.difficulty, []).append(r.reward)
        by_category.setdefault(r.category, []).append(r.reward)
    for difficulty, values in by_difficulty.items():
        metrics[f"pass_rate_{difficulty}"] = sum(values) / len(values)
    for category, values in by_category.items():
        safe_name = category.replace(" ", "_").lower()
        metrics[f"pass_rate_{safe_name}"] = sum(values) / len(values)

    return metrics


def resource_docker_args(task: TerminalBenchTask) -> tuple[str, ...]:
    """Container flags that apply a task's CPU and memory limits."""
    return (f"--cpus={task.cpus}", f"--memory={task.memory_mb}m")


class TerminalBenchExternalEval(ExternalEval):
    """Terminal-Bench evaluation with per-trial container orchestration.

    Subclasses evaluate other datasets in the same task format by overriding
    ``dataset``.
    """

    dataset: TerminalBenchDataset = TERMINAL_BENCH_DATASET

    @property
    def name(self) -> str:
        return self.dataset.name

    @property
    def description(self) -> str:
        return self.dataset.description

    @property
    def timeout_seconds(self) -> float:
        return 12000.0  # Max timeout across all tasks

    @property
    def arguments(self) -> dict[str, tuple[str, Any | None]]:
        return {
            "task_ids": ("Comma-separated task IDs to run (default: all)", None),
            "repo_path": ("Local repo path (default: clone fresh)", None),
            "repo_ref": ("Git ref to checkout", self.dataset.repo_ref),
            "max_concurrency": ("Max parallel containers", 1),
            "max_turns": ("Max agent steps per trial", 64),
            "n_attempts": ("Attempts per task, for pass@k", 1),
            "oracle": ("Run solve.sh instead of LLM agent", False),
            "sandbox_mode": ("Sandbox mode: docker, modal", "docker"),
            "scaffold": ("Scaffold: vanillux or openai_agents", DEFAULT_SCAFFOLD),
            "command_timeout": ("Seconds a single command may run", 120.0),
            "max_format_errors": ("Replies without a valid tool call before stopping", 64),
            "resource_limits": ("Apply each task's CPU and memory limits", True),
            "temperature": ("Sampling temperature", 0.7),
            "top_p": ("Nucleus sampling threshold", 0.95),
            "max_tokens": ("Max tokens per model reply", 16384),
        }

    @property
    def scaffold(self) -> str | None:
        return DEFAULT_SCAFFOLD

    async def execute(
        self,
        provider: InferenceProvider,
        args: dict[str, Any],
        output_dir: str | None = None,
        container_runtime: str = "podman",
    ) -> ExternalEvalResult:
        """Execute Terminal-Bench evaluation.

        Args:
            provider: Inference provider for LLM calls.
            args: Evaluation-specific arguments.
            output_dir: Directory to write results.
            container_runtime: Container runtime (docker or podman).

        Returns:
            ExternalEvalResult with metrics and per-trial results.
        """
        start_time = time.time()
        tb_args = TerminalBenchArgs.from_dict(args, default_ref=self.dataset.repo_ref)

        # Validate scaffold early to fail fast before spinning up sandboxes
        from olmo_eval.harness.scaffolds import validate_scaffold

        if tb_args.scaffold not in SCAFFOLD_TOOLS:
            supported = ", ".join(sorted(SCAFFOLD_TOOLS))
            raise ValueError(
                f"Unsupported scaffold {tb_args.scaffold!r} for {self.name}; use one of {supported}"
            )
        validate_scaffold(tb_args.scaffold)

        loader = TerminalBenchLoader(
            repo_url=self.dataset.repo_url,
            default_ref=self.dataset.repo_ref,
            dataset_version=self.dataset.version,
        )
        if tb_args.repo_path:
            repo_dir = Path(tb_args.repo_path)
        else:
            # Clone to a cache directory (not output_dir to avoid copying repo to results)
            repo_dir = Path(self.dataset.cache_dir)
            loader.ensure_repo(repo_dir, tb_args.repo_ref)

        tasks = loader.load_tasks(repo_dir, tb_args.task_ids)
        if not tasks:
            return self._error_result(
                "No tasks found", start_time, f"repo_dir={repo_dir}, task_ids={tb_args.task_ids}"
            )

        trials = [(task, attempt) for task in tasks for attempt in range(tb_args.n_attempts)]
        semaphore = asyncio.Semaphore(tb_args.max_concurrency)
        image_locks = {task.task_id: asyncio.Lock() for task in tasks}

        async def run_trial(task: TerminalBenchTask, attempt: int) -> TaskResult:
            async with semaphore:
                return await self._execute_task(
                    task=task,
                    attempt=attempt,
                    provider=provider,
                    container_runtime=container_runtime,
                    tb_args=tb_args,
                    image_lock=image_locks[task.task_id],
                )

        results = await asyncio.gather(
            *[run_trial(task, attempt) for task, attempt in trials], return_exceptions=True
        )

        task_results: list[TaskResult] = []
        for (task, attempt), result in zip(trials, results, strict=True):
            if isinstance(result, BaseException):
                logger.error(f"Task {task.task_id} attempt {attempt} failed: {result}")
                task_results.append(
                    self._failed_result(task, attempt, str(result), agent_duration=0.0)
                )
            else:
                task_results.append(result)

        metrics = compute_metrics(task_results, tb_args.n_attempts)
        predictions = self._build_predictions(task_results)

        result = ExternalEvalResult(
            name=self.name,
            success=True,
            metrics=metrics,
            metadata={
                "model_name": provider.model_name,
                "dataset": self.dataset.name,
                "dataset_version": self.dataset.version,
                "repo_ref": tb_args.repo_ref,
                "scaffold": tb_args.scaffold,
                "oracle_mode": tb_args.oracle,
                "max_turns": tb_args.max_turns,
                "n_attempts": tb_args.n_attempts,
                "resource_limits": tb_args.resource_limits,
            },
            duration_seconds=time.time() - start_time,
            predictions=predictions,
        )

        if output_dir:
            self._save_results(result, output_dir)
            self._save_task_results(task_results, output_dir)

        return result

    @staticmethod
    def _failed_result(
        task: TerminalBenchTask, attempt: int, error: str, agent_duration: float
    ) -> TaskResult:
        return TaskResult(
            task_id=task.task_id,
            reward=0.0,
            trajectory=AgentTrajectory(turns=()),
            completion_reason="error",
            agent_duration=agent_duration,
            verification_output="",
            verification_exit_code=-1,
            error=error,
            difficulty=task.difficulty,
            category=task.category,
            attempt=attempt,
        )

    async def _task_image(
        self, task: TerminalBenchTask, container_runtime: str, lock: asyncio.Lock | None = None
    ) -> str:
        """Return the sandbox image for a task, building it when the task ships none.

        Image work shells out to the container runtime and can take minutes, so
        it runs off the event loop. The lock keeps several attempts of one task
        from building the same image at once; the first builds, the rest reuse.
        """
        async with lock or asyncio.Lock():
            if task.needs_build:
                assert task.build_context is not None
                base = await asyncio.to_thread(
                    build_task_image,
                    Path(task.build_context),
                    container_runtime,
                    f"{self.dataset.image_prefix}-{task.task_id.lower()}",
                    task.build_timeout,
                )
            else:
                base = task.image
            return await asyncio.to_thread(get_swerex_image, base, container_runtime, pristine=True)

    def _sandbox_config(
        self,
        task: TerminalBenchTask,
        container_runtime: str,
        tb_args: TerminalBenchArgs,
        image: str,
    ) -> SandboxConfig:
        """Build the sandbox configuration for one trial of a task."""
        if tb_args.sandbox_mode == "docker":
            mode = SandboxMode.DOCKER
        elif tb_args.sandbox_mode == "modal":
            mode = SandboxMode.MODAL
        else:
            raise ValueError(
                f"Invalid sandbox_mode: {tb_args.sandbox_mode!r}. Must be 'docker' or 'modal'."
            )
        runtime = cast(ContainerRuntime, container_runtime)

        docker_args: tuple[str, ...] = ()
        if mode == SandboxMode.DOCKER:
            docker_args = tuple(get_docker_network_args(runtime))
            if runtime == "podman":
                docker_args += PODMAN_DOCKER_ARGS
            if tb_args.resource_limits:
                docker_args += resource_docker_args(task)
        if not task.allow_internet:
            logger.warning(
                f"Task {task.task_id} asks for no internet access, which this sandbox "
                "cannot enforce; the container keeps network access."
            )

        return SandboxConfig(
            image=image,
            mode=mode,
            container_runtime=runtime if mode == SandboxMode.DOCKER else "docker",
            working_dir=task.working_dir,
            startup_timeout=SANDBOX_STARTUP_TIMEOUT,
            command_timeout=tb_args.command_timeout,
            docker_args=docker_args,
            pristine_image=True,
        )

    async def _execute_task(
        self,
        task: TerminalBenchTask,
        attempt: int,
        provider: InferenceProvider,
        container_runtime: str,
        tb_args: TerminalBenchArgs,
        image_lock: asyncio.Lock | None = None,
    ) -> TaskResult:
        """Run one trial: start a container, run the agent, verify, and tear down."""
        from olmo_eval.harness.sandbox import SandboxManager

        logger.info(f"Executing task: {task.task_id} (attempt {attempt})")
        task_start = time.time()

        try:
            image = await self._task_image(task, container_runtime, image_lock)
        except Exception as e:
            logger.exception(f"Task {task.task_id} attempt {attempt} image failed")
            return self._failed_result(task, attempt, str(e), time.time() - task_start)

        sandbox_config = self._sandbox_config(task, container_runtime, tb_args, image)
        sandbox_manager = SandboxManager([sandbox_config], owner=f"tb-{task.task_id}-{attempt}")

        try:
            await sandbox_manager.start()

            trajectory = AgentTrajectory(turns=())
            completion_reason = "timeout"
            agent_steps = 0
            agent_error: str | None = None
            if tb_args.oracle:
                trajectory, completion_reason = await self._run_oracle(sandbox_manager, task)
            else:
                try:
                    (
                        trajectory,
                        completion_reason,
                        agent_steps,
                        agent_error,
                    ) = await asyncio.wait_for(
                        self._run_agent(sandbox_manager, task, provider, tb_args),
                        timeout=task.agent_timeout + AGENT_TIMEOUT_GRACE,
                    )
                except TimeoutError:
                    # The agent's time is up; what it left in the container still
                    # gets verified, as the reference harness does.
                    logger.warning(
                        f"Task {task.task_id} agent cut off after "
                        f"{task.agent_timeout + AGENT_TIMEOUT_GRACE:.0f}s"
                    )

            agent_duration = time.time() - task_start

            executor = sandbox_manager.get_executor(frozenset())
            verifier = TerminalBenchVerifier()
            await verifier.inject_tests(executor, task.test_files)
            verification = await verifier.run_verification(
                executor, task.verifier_timeout, task.working_dir, task.task_id
            )

            return TaskResult(
                task_id=task.task_id,
                reward=verification.reward,
                trajectory=trajectory,
                completion_reason=completion_reason,
                agent_duration=agent_duration,
                verification_output=verification.test_output,
                verification_exit_code=verification.test_exit_code,
                error=verification.error or agent_error,
                difficulty=task.difficulty,
                category=task.category,
                attempt=attempt,
                agent_steps=agent_steps,
            )

        except Exception as e:
            logger.exception(f"Task {task.task_id} attempt {attempt} failed")
            return self._failed_result(task, attempt, str(e), time.time() - task_start)
        finally:
            await sandbox_manager.stop()

    async def _run_oracle(
        self,
        sandbox_manager: SandboxManager,
        task: TerminalBenchTask,
    ) -> tuple[AgentTrajectory, str]:
        """Run the reference solution.

        The whole solution directory is placed in the container so a script
        that reads companion files beside it works as it does upstream. The
        script runs as a plain command with its output streamed, which keeps
        a solution that takes many minutes from outliving a single request.

        Returns:
            Tuple of (trajectory, completion_reason).
        """
        executor = sandbox_manager.get_executor(frozenset())
        verifier = TerminalBenchVerifier()
        solution_files = dict(task.solution_files)
        if "solve.sh" not in solution_files:
            solution_files["solve.sh"] = task.solution_script.encode()
        await verifier.inject_files(executor, solution_files, SOLUTION_DIR)

        result = await executor.execute_command(
            f"cd {task.working_dir} && bash {SOLUTION_DIR}/solve.sh",
            timeout=task.agent_timeout,
            stream=True,
            log_prefix=f"tb-{task.task_id}-oracle",
        )

        logger.info(f"Oracle exit code: {result.exit_code}")
        completion_reason = "timeout" if result.error == "timeout" else "oracle"
        return AgentTrajectory(turns=()), completion_reason

    async def _run_agent(
        self,
        sandbox_manager: SandboxManager,
        task: TerminalBenchTask,
        provider: InferenceProvider,
        tb_args: TerminalBenchArgs,
    ) -> tuple[AgentTrajectory, str, int, str | None]:
        """Run the LLM agent.

        Returns:
            Tuple of (trajectory, completion_reason, steps, error). The error is
            set when the run ended because the model could not be served, which
            is the harness's failure rather than the agent's.
        """
        from olmo_eval.harness.config import HarnessConfig
        from olmo_eval.harness.scaffolds import get_scaffold
        from olmo_eval.harness.tools import get_tools

        scaffold_name = tb_args.scaffold
        tools = get_tools(SCAFFOLD_TOOLS[scaffold_name])
        system_prompt = OPENAI_AGENTS_SYSTEM_PROMPT if scaffold_name == "openai_agents" else None
        harness_config = HarnessConfig(
            name=f"terminal_bench_{task.task_id}",
            tools=tools,
            system_prompt=system_prompt,
            max_turns=tb_args.max_turns,
            scaffold=scaffold_name,
            scaffold_kwargs={
                "command_timeout": tb_args.command_timeout,
                "max_format_errors": tb_args.max_format_errors,
                "agent_timeout": task.agent_timeout,
            },
        )

        scaffold = get_scaffold(scaffold_name)
        scaffold.set_sandbox_manager(sandbox_manager)

        request = LMRequest(
            request_type=RequestType.CHAT,
            messages=({"role": "user", "content": task.instruction},),
        )

        run_kwargs: dict[str, Any] = {}
        if scaffold_name == "openai_agents":
            run_kwargs["enable_compaction"] = tb_args.enable_compaction

        logger.info(
            f"Starting agent: scaffold={scaffold_name} task_id={task.task_id} "
            f"max_turns={tb_args.max_turns} agent_timeout={task.agent_timeout}"
        )
        harness_result = await scaffold.run(
            provider,
            harness_config,
            request,
            sampling_params=tb_args.sampling_params,
            trace_metadata={"task_id": task.task_id},
            **run_kwargs,
        )

        trajectory = harness_result.trajectory or AgentTrajectory(turns=())
        completion_reason = harness_result.metadata.get("completion_reason")
        if completion_reason is None:
            completion_reason = "max_turns" if harness_result.max_turns_reached else "complete"
        steps = harness_result.metadata.get("steps")
        if steps is None:
            steps = len(trajectory.assistant_turns)
        # A run that used up its turns is the agent's failure; only a run that
        # ended on an error the harness raised is reported as one.
        error = harness_result.error if not harness_result.max_turns_reached else None
        if error:
            logger.warning(f"Agent for task {task.task_id} failed: {error}")
        logger.info(f"Agent completed task {task.task_id}: {completion_reason}")
        return trajectory, str(completion_reason), int(steps), error

    def _build_predictions(self, task_results: list[TaskResult]) -> list[dict[str, Any]]:
        """Build predictions list from task results."""
        predictions = []
        for r in task_results:
            predictions.append(
                {
                    "native_id": r.native_id,
                    "task_id": r.task_id,
                    "attempt": r.attempt,
                    "instance_metrics": {
                        "reward": {"external": r.reward},
                        "agent_duration": {"external": r.agent_duration},
                        "agent_steps": {"external": float(r.agent_steps)},
                    },
                    "completion_reason": r.completion_reason,
                    "verification_exit_code": r.verification_exit_code,
                    "difficulty": r.difficulty,
                    "category": r.category,
                    "error": r.error,
                    "trajectory": r.trajectory.to_dict() if r.trajectory else None,
                }
            )
        return predictions

    def _save_task_results(self, task_results: list[TaskResult], output_dir: str) -> None:
        """Save detailed task results and trajectories to files."""
        import hashlib
        import re

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        results_file = output_path / f"{self.name}_tasks.json"
        data = []
        for result in task_results:
            output = result.verification_output[:10000] if result.verification_output else ""
            data.append(
                {
                    "task_id": result.task_id,
                    "attempt": result.attempt,
                    "reward": result.reward,
                    "completion_reason": result.completion_reason,
                    "agent_duration": result.agent_duration,
                    "agent_steps": result.agent_steps,
                    "verification_exit_code": result.verification_exit_code,
                    "verification_output": output,
                    "difficulty": result.difficulty,
                    "category": result.category,
                    "error": result.error,
                }
            )

        with open(results_file, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Task results saved to {results_file}")

        traces_dir = output_path / "traces"
        traces_dir.mkdir(parents=True, exist_ok=True)

        for result in task_results:
            if not result.trajectory:
                continue

            sanitized_id = re.sub(r"[^\w\-]", "_", result.native_id)
            sanitized_id = re.sub(r"_+", "_", sanitized_id).strip("_").lower()

            hash_input = f"{self.name}:{result.native_id}"
            short_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:6]

            trace_file = traces_dir / f"{self.name}_{sanitized_id}_{short_hash}.jsonl"

            trajectory_data = {
                "task_id": result.task_id,
                "attempt": result.attempt,
                "completion_reason": result.completion_reason,
                "trajectory": result.trajectory.to_dict(),
            }
            with open(trace_file, "w") as f:
                f.write(json.dumps(trajectory_data) + "\n")

        logger.info(f"Trajectories saved to {traces_dir}/")
