"""Tau2-bench external evaluation implementation."""

from __future__ import annotations

import base64
import json
import logging
import math
import shlex
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from olmo_eval.common.openai_credentials import litellm_model_name
from olmo_eval.evals.external.base import SandboxedExternalEval
from olmo_eval.evals.external.result import ExternalEvalResult

if TYPE_CHECKING:
    from olmo_eval.harness.sandbox.executor import SandboxExecutor
    from olmo_eval.inference.base import InferenceProvider

logger = logging.getLogger(__name__)

Tau2Domain = Literal["airline", "retail", "telecom"]
TAU2_REPOSITORY = "https://github.com/pdasigi/tau2-bench-verified.git"
TAU2_COMMIT = "05bfc0d4d2d963b13a00fce833deac0be255315c"
TAU2_RESULTS_FILE = "data/simulations/results.json"
TAU2_SUCCESS_TOLERANCE = 1e-6
TAU2_TERMINATION_REASONS = frozenset(
    {
        "agent_error",
        "agent_stop",
        "max_steps",
        "too_many_errors",
        "user_error",
        "user_stop",
    }
)

# Default max tokens if we can't query the server
DEFAULT_MAX_TOKENS = 32768


def _parse_optional(data: dict, key: str, type_fn: type) -> Any:
    """Parse an optional value from a dict with type conversion."""
    value = data.get(key)
    return type_fn(value) if value is not None else None


def _parse_bool(value: Any, default: bool = False) -> bool:
    """Parse a boolean value from string or bool."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ("true", "1", "yes")
    return bool(value)


@dataclass
class Tau2Args:
    """Arguments for tau2_bench evaluation."""

    # Core settings
    domain: Tau2Domain = "airline"
    num_trials: int = 1
    max_steps: int = 30
    max_concurrency: int = 3

    # Agent LLM settings
    max_tokens: int | None = None
    max_model_len: int | None = None
    temperature: float | None = None

    # User LLM settings
    user_llm: str = "gpt-4o-mini"
    user_temperature: float | None = None

    # Task filtering
    task_split_name: str | None = None
    task_ids: list[str] | None = None
    num_tasks: int | None = None

    # Execution settings
    max_errors: int | None = None
    seed: int | None = None
    log_level: str | None = None
    enforce_communication_protocol: bool = False

    def __post_init__(self) -> None:
        if self.domain not in ("airline", "retail", "telecom"):
            raise ValueError(
                f"domain must be 'airline', 'retail', or 'telecom', got {self.domain!r}"
            )

        for name in ("num_trials", "max_steps", "max_concurrency"):
            value = getattr(self, name)
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")

        for name in ("max_tokens", "max_model_len", "num_tasks"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive when set, got {value}")

        if self.max_errors is not None and self.max_errors < 0:
            raise ValueError(f"max_errors must be nonnegative when set, got {self.max_errors}")
        if self.seed is not None and self.seed < 0:
            raise ValueError(f"seed must be nonnegative when set, got {self.seed}")
        for name in ("temperature", "user_temperature"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be nonnegative when set, got {value}")

        if self.task_ids is not None:
            if not isinstance(self.task_ids, list) or not self.task_ids:
                raise ValueError("task_ids must be a non-empty list when set")
            if any(not isinstance(task_id, str) or not task_id for task_id in self.task_ids):
                raise ValueError("task_ids must contain non-empty strings")
            if len(self.task_ids) != len(set(self.task_ids)):
                raise ValueError("task_ids must not contain duplicates")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Tau2Args:
        # Handle task_ids which can be comma-separated string or list
        task_ids = data.get("task_ids")
        if isinstance(task_ids, str):
            task_ids = [t.strip() for t in task_ids.split(",") if t.strip()]

        return cls(
            domain=data.get("domain", "airline"),
            num_trials=int(data.get("num_trials", 1)),
            max_steps=int(data.get("max_steps", 30)),
            max_concurrency=int(data.get("max_concurrency", 3)),
            max_tokens=_parse_optional(data, "max_tokens", int),
            max_model_len=_parse_optional(data, "max_model_len", int),
            temperature=_parse_optional(data, "temperature", float),
            user_llm=data.get("user_llm", "gpt-4o-mini"),
            user_temperature=_parse_optional(data, "user_temperature", float),
            task_split_name=data.get("task_split_name"),
            task_ids=task_ids,
            num_tasks=_parse_optional(data, "num_tasks", int),
            max_errors=_parse_optional(data, "max_errors", int),
            seed=_parse_optional(data, "seed", int),
            log_level=data.get("log_level"),
            enforce_communication_protocol=_parse_bool(data.get("enforce_communication_protocol")),
        )


class Tau2ExternalEval(SandboxedExternalEval):
    """Tau2-bench evaluation for customer service agent tasks."""

    @property
    def name(self) -> str:
        return "tau2_bench"

    @property
    def description(self) -> str:
        return (
            "Evaluates LLM agents on customer service tasks across airline, retail, and telecom "
            "domains. Measures task completion rate and constraint satisfaction."
        )

    @property
    def sandbox_image(self) -> str:
        # TODO(undfined): Create a custom image with some core utilities (e.g. git, curl, etc)
        return "ghcr.io/astral-sh/uv:python3.11-bookworm"

    @property
    def working_dir(self) -> str:
        return "/workspace"

    @property
    def timeout_seconds(self) -> float:
        return 7200.0

    @property
    def setup_command(self) -> tuple[str, ...]:
        repo = f"{self.working_dir}/tau2-bench"
        return (
            # Using Pradeep's repo as it handles two errors that the original does not:
            # 1) The assistant returning an empty response.
            # 2) Litellm's timeouts errors.
            f"git clone {TAU2_REPOSITORY} {repo}",
            f"cd {repo} && git checkout {TAU2_COMMIT}",
            f"cd {repo} && uv sync",
            f"mkdir -p {self.results_dir}",
        )

    @property
    def required_secrets(self) -> tuple[str, ...]:
        return ("OPENAI_API_KEY",)

    @property
    def arguments(self) -> dict[str, tuple[str, Any | None]]:
        return {
            "domain": ("Task domain: 'airline', 'retail', or 'telecom'", "airline"),
            "num_trials": ("Number of trials per task", 1),
            "max_steps": ("Max agent steps per trial", 30),
            "max_concurrency": ("Max concurrent requests", 3),
            "max_tokens": ("Max tokens for agent LLM responses", None),
            "max_model_len": ("Model context length for litellm registration", None),
            "temperature": ("Temperature for agent LLM responses", None),
            "user_llm": ("LLM for simulated user (requires API key)", "gpt-4o-mini"),
            "user_temperature": ("Temperature for user LLM", None),
            "task_split_name": ("Task split to run (default: 'base')", None),
            "task_ids": ("Comma-separated task IDs to run", None),
            "num_tasks": ("Number of tasks to run (default: all)", None),
            "max_errors": ("Max consecutive tool errors allowed", None),
            "seed": ("Random seed for reproducibility", None),
            "log_level": ("Log level (DEBUG, INFO, WARNING, ERROR)", None),
            "enforce_communication_protocol": ("Enforce communication protocol rules", False),
        }

    async def execute(
        self,
        provider: InferenceProvider,
        args: dict[str, Any],
        output_dir: str | None = None,
        container_runtime: str = "podman",
    ) -> ExternalEvalResult:
        start_time = time.time()
        try:
            tau2_args = Tau2Args.from_dict(args)
        except (TypeError, ValueError) as e:
            return self._error_result(f"Invalid Tau2 arguments: {e}", start_time)
        all_output: list[str] = []

        # Extract URL and model name from provider for sandbox use
        provider_url = getattr(provider, "base_url", None) or "http://localhost:8000/v1"
        model_name = provider.model_name
        # Detect if this is a locally-deployed server (vLLM) vs external API
        is_local = self._is_local_provider(provider, provider_url)

        try:
            from olmo_eval.harness.sandbox.executor import SandboxExecutor
        except ImportError as e:
            return self._error_result(f"SWE-ReX not installed: {e}", start_time)

        sandbox_config = self._create_sandbox_config(container_runtime, output_dir)

        try:
            async with SandboxExecutor(sandbox_config, name=self.name) as executor:
                if err := await self._run_setup(executor, all_output, start_time):
                    return err

                sandbox_url = self._get_provider_url_for_sandbox(provider_url)

                if is_local:
                    if not await self._check_provider_health(executor, sandbox_url):
                        return self._error_result(
                            f"Provider not reachable at {sandbox_url}",
                            start_time,
                            "\n".join(all_output),
                        )
                    # TODO(undfined): Get this from the source model config in the future.
                    # OpenAI-compatible servers do not expose /v1/models endpoint.
                    max_model_len = tau2_args.max_model_len or DEFAULT_MAX_TOKENS
                    await self._setup_litellm_wrapper(
                        executor, model_name, sandbox_url, max_model_len
                    )

                run_cmd = self._build_run_command(model_name, sandbox_url, is_local, tau2_args)
                logger.info(f"[{self.name}] Running: {run_cmd}")

                run_result = await executor.execute_command(
                    run_cmd, timeout=self.timeout_seconds, stream=True, log_prefix=self.name
                )
                all_output.append(f"$ {run_cmd}\n{run_result.output}")
                logger.info(f"[{self.name}] Run exit code: {run_result.exit_code}")

                result = await self._extract_results(
                    executor,
                    "\n".join(all_output),
                    run_result.exit_code,
                    tau2_args,
                    model_name,
                    output_dir,
                )

        except Exception as e:
            logger.exception(f"[{self.name}] Execution failed")
            return self._error_result(str(e), start_time, "\n".join(all_output))

        result.duration_seconds = time.time() - start_time
        if output_dir:
            self._save_results(result, output_dir)

        return result

    def _build_run_command(
        self,
        model_name: str,
        provider_url: str,
        is_local: bool,
        tau2_args: Tau2Args,
    ) -> str:
        """Build the tau2 run command."""
        agent_model = f"hosted_vllm/{model_name}" if is_local else litellm_model_name(model_name)
        repo = f"{self.working_dir}/tau2-bench"

        # Use wrapper for local providers (registers model with litellm)
        if is_local:
            tau2_cmd = f"{repo}/.venv/bin/python {repo}/tau2_wrapper.py run"
        else:
            tau2_cmd = f"{repo}/.venv/bin/tau2 run"

        parts = [f"cd {repo} &&", tau2_cmd, "--agent-llm", shlex.quote(agent_model)]

        # Agent LLM args
        agent_llm_args: dict[str, Any] = {}
        if is_local:
            agent_llm_args["api_base"] = provider_url
        if tau2_args.max_tokens:
            agent_llm_args["max_tokens"] = tau2_args.max_tokens
        if tau2_args.temperature is not None:
            agent_llm_args["temperature"] = tau2_args.temperature
        if agent_llm_args:
            parts.extend(["--agent-llm-args", shlex.quote(json.dumps(agent_llm_args))])

        # User LLM
        parts.extend(["--user-llm", shlex.quote(litellm_model_name(tau2_args.user_llm))])
        if tau2_args.user_temperature is not None:
            user_args = json.dumps({"temperature": tau2_args.user_temperature})
            parts.extend(["--user-llm-args", shlex.quote(user_args)])

        # Core settings
        parts.extend(
            [
                "--domain",
                shlex.quote(tau2_args.domain),
                "--num-trials",
                str(tau2_args.num_trials),
                "--max-steps",
                str(tau2_args.max_steps),
                "--max-concurrency",
                str(tau2_args.max_concurrency),
                "--save-to",
                "results",  # Saves to {repo}/data/simulations/results.json
            ]
        )

        # Optional args
        if tau2_args.task_split_name:
            parts.extend(["--task-split-name", shlex.quote(tau2_args.task_split_name)])
        if tau2_args.task_ids:
            parts.append("--task-ids")
            parts.extend(shlex.quote(task_id) for task_id in tau2_args.task_ids)
        if tau2_args.num_tasks is not None:
            parts.extend(["--num-tasks", str(tau2_args.num_tasks)])
        if tau2_args.max_errors is not None:
            parts.extend(["--max-errors", str(tau2_args.max_errors)])
        if tau2_args.seed is not None:
            parts.extend(["--seed", str(tau2_args.seed)])
        if tau2_args.log_level:
            parts.extend(["--log-level", shlex.quote(tau2_args.log_level)])
        if tau2_args.enforce_communication_protocol:
            parts.append("--enforce-communication-protocol")

        return " ".join(parts)

    async def _setup_litellm_wrapper(
        self, executor: SandboxExecutor, model_name: str, provider_url: str, max_model_len: int
    ) -> None:
        """Create wrapper script that registers the model with litellm."""
        max_tokens = max_model_len
        repo = f"{self.working_dir}/tau2-bench"

        # Use json.dumps to safely escape the model name for Python string literal
        model_key = f"hosted_vllm/{model_name}"
        script = f'''\
#!/usr/bin/env python
"""Wrapper to register local vLLM model with litellm."""
import litellm
import sys

litellm.register_model({{
    {json.dumps(model_key)}: {{
        "max_tokens": {max_tokens},
        "input_cost_per_token": 0.0,
        "output_cost_per_token": 0.0,
    }}
}})

from tau2.cli import main
sys.exit(main())
'''
        encoded = base64.b64encode(script.encode()).decode()
        wrapper_path = shlex.quote(f"{repo}/tau2_wrapper.py")
        await executor.execute_command(
            f"echo '{encoded}' | base64 -d > {wrapper_path}", timeout=30.0
        )
        logger.info(f"[{self.name}] Created litellm wrapper (max_tokens={max_tokens})")

    async def _extract_results(
        self,
        executor: SandboxExecutor,
        raw_output: str,
        exit_code: int,
        tau2_args: Tau2Args,
        model_name: str,
        output_dir: str | None = None,
    ) -> ExternalEvalResult:
        """Extract metrics from tau2-bench results."""
        if exit_code != 0:
            return ExternalEvalResult(
                name=self.name,
                success=False,
                error=f"Tau2 exited with code {exit_code}",
                raw_output=raw_output,
            )

        results_file = f"{self.working_dir}/tau2-bench/{TAU2_RESULTS_FILE}"
        cat_result = await executor.execute_command(
            f"cat {shlex.quote(results_file)}", timeout=30.0
        )
        if not cat_result.success:
            return ExternalEvalResult(
                name=self.name,
                success=False,
                error=f"Tau2 results file is missing or unreadable: {results_file}",
                raw_output=raw_output,
            )

        try:
            data = json.loads(cat_result.output)
        except json.JSONDecodeError as e:
            return ExternalEvalResult(
                name=self.name,
                success=False,
                error=f"Invalid Tau2 results JSON: {e}",
                raw_output=raw_output,
            )

        try:
            simulations_by_task = self._validate_result_inventory(data, tau2_args)
        except ValueError as e:
            return ExternalEvalResult(
                name=self.name,
                success=False,
                error=f"Invalid Tau2 result inventory: {e}",
                raw_output=raw_output,
            )

        metrics = self._compute_pass_k_metrics(simulations_by_task, tau2_args.num_trials)
        predictions = self._build_predictions(data["tasks"], simulations_by_task)
        simulations = data["simulations"]
        successful_simulations = sum(
            self._is_successful(simulation["reward_info"]["reward"]) for simulation in simulations
        )
        termination_counts = dict(
            sorted(Counter(simulation["termination_reason"] for simulation in simulations).items())
        )
        info = data["info"]
        metadata: dict[str, Any] = {
            "benchmark_repository": TAU2_REPOSITORY,
            "benchmark_commit": TAU2_COMMIT,
            "simulations_file": results_file,
            "model_name": model_name,
            "domain": info["environment_info"]["domain_name"],
            "user_llm": tau2_args.user_llm,
            "task_split_name": tau2_args.task_split_name or "base",
            "seed": info["seed"],
            "num_tasks": len(data["tasks"]),
            "num_trials": info["num_trials"],
            "max_steps": info["max_steps"],
            "max_errors": info["max_errors"],
            "max_concurrency": tau2_args.max_concurrency,
            "expected_simulations": len(data["tasks"]) * tau2_args.num_trials,
            "observed_simulations": len(simulations),
            "successful_simulations": successful_simulations,
            "termination_counts": termination_counts,
            "tau2_run_info": {
                "git_commit": info["git_commit"],
                "agent": info["agent_info"],
                "user": info["user_info"],
            },
        }

        if output_dir:
            trajectories_path = Path(output_dir) / "tau2_trajectories.json"
            trajectories_path.parent.mkdir(parents=True, exist_ok=True)
            with trajectories_path.open("w") as f:
                json.dump(data, f, indent=2)
            metadata["trajectories_file"] = str(trajectories_path)
            logger.info(f"[{self.name}] Trajectories saved to {trajectories_path}")

        return ExternalEvalResult(
            name=self.name,
            success=True,
            metrics=metrics,
            metadata=metadata,
            raw_output=raw_output,
            predictions=predictions,
        )

    @staticmethod
    def _is_successful(reward: float) -> bool:
        return (1.0 - TAU2_SUCCESS_TOLERANCE) <= reward <= (1.0 + TAU2_SUCCESS_TOLERANCE)

    def _validate_result_inventory(
        self, data: Any, tau2_args: Tau2Args
    ) -> dict[str, list[dict[str, Any]]]:
        if not isinstance(data, dict):
            raise ValueError("top-level result must be an object")

        tasks = data.get("tasks")
        simulations = data.get("simulations")
        info = data.get("info")
        if not isinstance(tasks, list) or not tasks:
            raise ValueError("tasks must be a non-empty list")
        if not isinstance(simulations, list) or not simulations:
            raise ValueError("simulations must be a non-empty list")
        if not isinstance(info, dict):
            raise ValueError("info must be an object")

        task_ids: list[str] = []
        for index, task in enumerate(tasks):
            if not isinstance(task, dict):
                raise ValueError(f"task at index {index} must be an object")
            task_data = cast(dict[str, Any], task)
            task_id = task_data.get("id")
            if not isinstance(task_id, str) or not task_id:
                raise ValueError(f"task at index {index} has an invalid id")
            task_ids.append(task_id)
        if len(task_ids) != len(set(task_ids)):
            duplicates = sorted(
                task_id for task_id, count in Counter(task_ids).items() if count > 1
            )
            raise ValueError(f"duplicate task IDs: {duplicates}")

        if info.get("git_commit") != TAU2_COMMIT:
            raise ValueError(
                f"benchmark commit mismatch: {info.get('git_commit')!r} != {TAU2_COMMIT!r}"
            )
        if info.get("num_trials") != tau2_args.num_trials:
            raise ValueError(
                f"num_trials mismatch: {info.get('num_trials')!r} != {tau2_args.num_trials}"
            )
        if info.get("max_steps") != tau2_args.max_steps:
            raise ValueError(
                f"max_steps mismatch: {info.get('max_steps')!r} != {tau2_args.max_steps}"
            )
        if tau2_args.max_errors is not None and info.get("max_errors") != tau2_args.max_errors:
            raise ValueError(
                f"max_errors mismatch: {info.get('max_errors')!r} != {tau2_args.max_errors}"
            )
        if tau2_args.seed is not None and info.get("seed") != tau2_args.seed:
            raise ValueError(f"seed mismatch: {info.get('seed')!r} != {tau2_args.seed}")
        environment_info = info.get("environment_info")
        if not isinstance(environment_info, dict):
            raise ValueError("info.environment_info must be an object")
        if environment_info.get("domain_name") != tau2_args.domain:
            raise ValueError(
                f"domain mismatch: {environment_info.get('domain_name')!r} != {tau2_args.domain!r}"
            )
        for name in ("agent_info", "user_info"):
            if not isinstance(info.get(name), dict):
                raise ValueError(f"info.{name} must be an object")

        task_id_set = set(task_ids)
        simulation_ids: set[str] = set()
        observed_slots: set[tuple[str, int]] = set()
        simulations_by_task: dict[str, list[dict[str, Any]]] = {task_id: [] for task_id in task_ids}
        for index, simulation in enumerate(simulations):
            if not isinstance(simulation, dict):
                raise ValueError(f"simulation at index {index} must be an object")
            simulation_data = cast(dict[str, Any], simulation)

            simulation_id = simulation_data.get("id")
            if not isinstance(simulation_id, str) or not simulation_id:
                raise ValueError(f"simulation at index {index} has an invalid id")
            if simulation_id in simulation_ids:
                raise ValueError(f"duplicate simulation ID: {simulation_id!r}")
            simulation_ids.add(simulation_id)

            task_id = simulation_data.get("task_id")
            if task_id not in task_id_set:
                raise ValueError(f"unexpected task ID in simulation: {task_id!r}")
            trial = simulation_data.get("trial")
            if isinstance(trial, bool) or not isinstance(trial, int):
                raise ValueError(f"trial for task {task_id!r} must be an integer")
            if not 0 <= trial < tau2_args.num_trials:
                raise ValueError(
                    f"trial for task {task_id!r} is out of range: {trial}; "
                    f"expected 0..{tau2_args.num_trials - 1}"
                )

            slot = (task_id, trial)
            if slot in observed_slots:
                raise ValueError(f"duplicate task/trial slot: {slot!r}")
            observed_slots.add(slot)

            reward_info = simulation_data.get("reward_info")
            if not isinstance(reward_info, dict):
                raise ValueError(f"reward_info for slot {slot!r} must be an object")
            reward = reward_info.get("reward")
            if isinstance(reward, bool) or not isinstance(reward, int | float):
                raise ValueError(f"reward for slot {slot!r} must be numeric")
            if not math.isfinite(reward):
                raise ValueError(f"reward for slot {slot!r} must be finite")

            termination_reason = simulation_data.get("termination_reason")
            if termination_reason not in TAU2_TERMINATION_REASONS:
                raise ValueError(
                    f"termination_reason for slot {slot!r} is invalid: {termination_reason!r}"
                )

            simulations_by_task[task_id].append(simulation_data)

        expected_slots = {
            (task_id, trial) for task_id in task_ids for trial in range(tau2_args.num_trials)
        }
        missing_slots = sorted(expected_slots - observed_slots)
        if missing_slots:
            preview = missing_slots[:5]
            suffix = "..." if len(missing_slots) > len(preview) else ""
            raise ValueError(f"missing task/trial slots: {preview}{suffix}")

        for task_simulations in simulations_by_task.values():
            task_simulations.sort(key=lambda simulation: simulation["trial"])
        return simulations_by_task

    def _compute_pass_k_metrics(
        self, simulations_by_task: dict[str, list[dict[str, Any]]], num_trials: int
    ) -> dict[str, float]:
        """Compute pass^k metrics from tau2-bench simulations.

        See: https://arxiv.org/abs/2406.12045
        """
        metrics: dict[str, float] = {}
        for k in range(1, num_trials + 1):
            pass_k_values: list[float] = []
            for simulations in simulations_by_task.values():
                successful_trials = sum(
                    self._is_successful(simulation["reward_info"]["reward"])
                    for simulation in simulations
                )
                pass_k_values.append(math.comb(successful_trials, k) / math.comb(num_trials, k))
            metrics[f"pass^{k}"] = sum(pass_k_values) / len(pass_k_values)

        return metrics

    def _build_predictions(
        self,
        tasks: list[dict[str, Any]],
        simulations_by_task: dict[str, list[dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        """Build per-task predictions from tau2-bench simulations."""
        predictions = []
        for task in tasks:
            task_id = task["id"]
            task_sims = simulations_by_task[task_id]

            # Extract per-trial data
            trials = []
            for sim in task_sims:
                reward_info = sim.get("reward_info", {})
                trial_data: dict[str, Any] = {
                    "trial": sim.get("trial"),
                    "reward": reward_info.get("reward", 0),
                    "duration": sim.get("duration"),
                    "termination_reason": sim.get("termination_reason"),
                    "agent_cost": sim.get("agent_cost"),
                    "user_cost": sim.get("user_cost"),
                }
                # Include reward breakdown if present
                if "reward_breakdown" in reward_info:
                    trial_data["reward_breakdown"] = reward_info["reward_breakdown"]
                if "reward_basis" in reward_info:
                    trial_data["reward_basis"] = reward_info["reward_basis"]
                trials.append(trial_data)

            # Compute aggregated metrics
            rewards = [t["reward"] for t in trials]
            success_rate = sum(self._is_successful(reward) for reward in rewards) / len(rewards)
            average_reward = sum(rewards) / len(rewards)
            total_duration = sum(t.get("duration") or 0 for t in trials)
            avg_duration = total_duration / len(trials)
            total_agent_cost = sum(t.get("agent_cost") or 0 for t in trials)
            total_user_cost = sum(t.get("user_cost") or 0 for t in trials)
            total_cost = total_agent_cost + total_user_cost

            # Compute error rate (agent_error or user_error terminations)
            error_reasons = {"agent_error", "user_error", "too_many_errors"}
            error_count = sum(1 for t in trials if t.get("termination_reason") in error_reasons)
            error_rate = error_count / len(trials) if trials else 0.0

            predictions.append(
                {
                    "native_id": task_id,
                    "instance_metrics": {
                        "success_rate": {"external": success_rate},
                        "average_reward": {"external": average_reward},
                        "avg_duration": {"external": avg_duration},
                        "total_cost": {"external": total_cost},
                        "error_rate": {"external": error_rate},
                    },
                    "num_trials": len(trials),
                    "trials": trials,
                }
            )
        return predictions
