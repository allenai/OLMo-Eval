"""SWE-bench Verified external evaluation."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from olmo_eval.common.types import LMRequest, RequestType, SamplingParams
from olmo_eval.common.types.trajectory import AgentTrajectory
from olmo_eval.evals.external.args import as_bool, as_list, as_optional_int
from olmo_eval.evals.external.base import ExternalEval
from olmo_eval.evals.external.network import get_docker_network_args
from olmo_eval.evals.external.result import ExternalEvalResult
from olmo_eval.harness.sandbox.config import (
    Capability,
    ContainerRuntime,
    SandboxConfig,
    SandboxMode,
)

from .grader import (
    TESTBED_DIR,
    GradeResult,
    extract_patch,
    grade_in_sandbox,
    snapshot_worktree,
)
from .loader import DATASET_PATH, DATASET_REVISION, load_instances, select_instances
from .task import IMAGE_SOURCES, SWEBenchInstance

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

#: Agent turns per instance when ``max_turns`` is not given.
DEFAULT_MAX_TURNS: dict[str, int] = {"vanillux": 64, "openai_agents": 100}

#: Extra wall-clock time, past the agent timeout, before a vanillux run is cut
#: off from outside. The scaffold stops itself at the timeout between steps;
#: this bounds a step that is already in flight.
AGENT_TIMEOUT_GRACE = 300.0

#: Scaffold completion reasons renamed to the ones this eval reports.
COMPLETION_REASONS: dict[str, str] = {
    "max_steps": "max_turns",
    "timeout": "agent_timeout",
    "context_overflow": "context_exceeded",
}

SYSTEM_PROMPT = """\
You are a software engineer working in a Linux container with a code repository.
You have access to the following tools:
- execute_bash_session(command): Execute a bash command in a persistent shell session
- submit(): Call when you have finished making your changes

Work step by step and inspect the results of each command. Avoid commands that print \
very large outputs; use tools like grep, head, tail, and sed -n to view parts of files. \
Edit files with tools such as sed, or by writing them with python or a heredoc. \
Commands that need interactive input are not supported.
"""

INSTANCE_PROMPT = """\
<uploaded_files>
{testbed}
</uploaded_files>
I've uploaded a code repository in the directory {testbed}. Consider the following issue \
description:

<issue_description>
{problem_statement}
</issue_description>
{hints}
Implement the necessary changes to the repository so that the requirements in the issue \
are met. Changes to the test files described in the issue have already been taken care \
of, so you do not need to modify any tests. Make minimal changes to non-test files in \
{testbed} to resolve the issue.

Follow these steps:
1. Explore the repository to find the code relevant to the issue.
2. Write a short script that reproduces the problem and run it to confirm the error.
3. Edit the source code to fix the problem.
4. Rerun your script to confirm the fix, and think about edge cases.
5. Call submit() when you are done. Your changes in {testbed} are collected automatically.
"""

HINTS_TEMPLATE = """
<hints>
{hints_text}
</hints>
"""

# Activates the repository's environment for the agent's shell.
SESSION_SETUP = f"source /opt/miniconda3/bin/activate && conda activate testbed && cd {TESTBED_DIR}"


def _swebench_version() -> str | None:
    try:
        from importlib.metadata import version

        return version("swebench")
    except Exception:
        return None


@dataclass
class SWEBenchArgs:
    """Arguments for SWE-bench evaluation."""

    instance_ids: list[str] | None = None
    repos: list[str] | None = None
    limit: int | None = None
    dataset: str = DATASET_PATH
    revision: str | None = DATASET_REVISION
    max_concurrency: int = 4
    max_turns: int = DEFAULT_MAX_TURNS[DEFAULT_SCAFFOLD]
    command_timeout: float = 120.0
    agent_timeout: float = 3600.0
    max_tool_output_chars: int | None = 10000
    eval_timeout: float = 1800.0
    image_timeout: float = 900.0
    stop_on_submit: bool = True
    resume_from: str | None = None
    image_source: str = "ghcr"
    cleanup_images: bool = True
    include_hints: bool = False
    oracle: bool = False
    sandbox_mode: str = "docker"
    enable_compaction: bool = False
    scaffold: str = DEFAULT_SCAFFOLD
    max_format_errors: int = 64
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 16384

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SWEBenchArgs:
        scaffold = str(data.get("scaffold", DEFAULT_SCAFFOLD))
        return cls(
            instance_ids=as_list(data.get("instance_ids")),
            repos=as_list(data.get("repos")),
            limit=as_optional_int(data.get("limit")),
            dataset=data.get("dataset", DATASET_PATH),
            # The pinned revision belongs to the default dataset only.
            revision=data.get(
                "revision",
                DATASET_REVISION if data.get("dataset", DATASET_PATH) == DATASET_PATH else None,
            )
            or None,
            max_concurrency=int(data.get("max_concurrency", 4)),
            max_turns=int(data.get("max_turns", DEFAULT_MAX_TURNS.get(scaffold, 64))),
            command_timeout=float(data.get("command_timeout", 120.0)),
            agent_timeout=float(data.get("agent_timeout", 3600.0)),
            max_tool_output_chars=as_optional_int(data.get("max_tool_output_chars", 10000)),
            eval_timeout=float(data.get("eval_timeout", 1800.0)),
            image_timeout=float(data.get("image_timeout", 900.0)),
            stop_on_submit=as_bool(data.get("stop_on_submit", True)),
            resume_from=data.get("resume_from") or None,
            image_source=str(data.get("image_source", "ghcr")),
            cleanup_images=as_bool(data.get("cleanup_images", True)),
            include_hints=as_bool(data.get("include_hints", False)),
            oracle=as_bool(data.get("oracle", False)),
            sandbox_mode=data.get("sandbox_mode", "docker"),
            enable_compaction=as_bool(data.get("enable_compaction", False)),
            scaffold=scaffold,
            max_format_errors=int(data.get("max_format_errors", 64)),
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
class InstanceResult:
    """Result of running and grading one SWE-bench instance."""

    instance_id: str
    repo: str
    difficulty: str
    patch: str = ""
    grade: GradeResult = field(default_factory=GradeResult)
    trajectory: AgentTrajectory = field(default_factory=lambda: AgentTrajectory(turns=()))
    completion_reason: str = "error"
    agent_duration: float = 0.0
    eval_duration: float = 0.0
    error: str | None = None

    @property
    def resolved(self) -> bool:
        return self.grade.resolved

    def to_record(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict that ``from_record`` can read back."""
        return {
            "instance_id": self.instance_id,
            "repo": self.repo,
            "difficulty": self.difficulty,
            "patch": self.patch,
            "grade": asdict(self.grade),
            "trajectory": self.trajectory.to_dict(),
            "completion_reason": self.completion_reason,
            "agent_duration": self.agent_duration,
            "eval_duration": self.eval_duration,
            "error": self.error,
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> InstanceResult:
        """Rebuild a result from ``to_record`` output."""
        return cls(
            instance_id=record["instance_id"],
            repo=record["repo"],
            difficulty=record["difficulty"],
            patch=record["patch"],
            grade=GradeResult(**record["grade"]),
            trajectory=AgentTrajectory.from_dict(record["trajectory"]),
            completion_reason=record["completion_reason"],
            agent_duration=record["agent_duration"],
            eval_duration=record["eval_duration"],
            error=record["error"],
        )


def load_progress(path: str | Path) -> dict[str, InstanceResult]:
    """Read finished instances from a progress file, keyed by instance ID.

    A truncated last line, as left by a job that was killed mid-write, is skipped.
    """
    finished: dict[str, InstanceResult] = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        try:
            result = InstanceResult.from_record(json.loads(line))
        except (json.JSONDecodeError, KeyError, TypeError):
            logger.warning(f"Skipping unreadable progress record in {path}")
            continue
        finished[result.instance_id] = result
    return finished


def called_tool(trajectory: AgentTrajectory, name: str) -> bool:
    """Whether the agent called the named tool at any point in the trajectory."""
    return any(call.function.name == name for turn in trajectory.turns for call in turn.tool_calls)


def is_context_length_error(error: BaseException) -> bool:
    """Whether an error reports that the conversation outgrew the model's context window."""
    message = str(error).lower()
    return "maximum context length" in message or "context_length_exceeded" in message


def _hints(instance: SWEBenchInstance, include_hints: bool) -> str:
    if include_hints and instance.hints_text.strip():
        return HINTS_TEMPLATE.format(hints_text=instance.hints_text.strip())
    return ""


def build_prompt(instance: SWEBenchInstance, include_hints: bool = False) -> str:
    """Build the user message that describes the issue to the ``openai_agents`` agent."""
    return INSTANCE_PROMPT.format(
        testbed=TESTBED_DIR,
        problem_statement=instance.problem_statement.strip(),
        hints=_hints(instance, include_hints),
    )


def build_task_text(instance: SWEBenchInstance, include_hints: bool = False) -> str:
    """Build the task text for scaffolds that wrap it in their own instructions.

    The issue is given as is, the way Harbor's SWE-bench tasks state it.
    """
    return instance.problem_statement.strip() + _hints(instance, include_hints)


def compute_metrics(results: list[InstanceResult]) -> dict[str, float]:
    """Aggregate per-instance results into evaluation metrics."""
    n = len(results)

    def rate(values: list[bool]) -> float:
        return sum(values) / len(values) if values else 0.0

    metrics: dict[str, float] = {
        "resolve_rate": rate([r.resolved for r in results]),
        "num_resolved": float(sum(r.resolved for r in results)),
        "num_instances": float(n),
        "patch_apply_rate": rate([r.grade.patch_applied for r in results]),
        "tests_ran_rate": rate([r.grade.tests_ran for r in results]),
        "empty_patch_rate": rate([r.grade.empty_patch for r in results]),
        "error_rate": rate([r.error is not None for r in results]),
    }

    groups: dict[str, list[bool]] = {}
    for r in results:
        groups.setdefault(f"resolve_rate_difficulty_{r.difficulty}", []).append(r.resolved)
        groups.setdefault(f"resolve_rate_repo_{r.repo}", []).append(r.resolved)
    for key, values in sorted(groups.items()):
        safe_key = "".join(c if c.isalnum() else "_" for c in key).lower()
        metrics[safe_key] = rate(values)
    return metrics


class SWEBenchVerifiedExternalEval(ExternalEval):
    """SWE-bench Verified with one agent container and one grading container per instance."""

    @property
    def name(self) -> str:
        return "swe_bench_verified"

    @property
    def description(self) -> str:
        return "Evaluates LLM agents on resolving 500 human-validated GitHub issues"

    @property
    def timeout_seconds(self) -> float:
        return 24 * 3600.0

    @property
    def extras(self) -> tuple[str, ...]:
        return ("swebench",)

    @property
    def arguments(self) -> dict[str, tuple[str, Any | None]]:
        return {
            "instance_ids": ("Comma-separated instance IDs to run (default: all)", None),
            "repos": ("Comma-separated repositories to run, e.g. django/django", None),
            "limit": ("Run only the first N selected instances", None),
            "dataset": ("HuggingFace dataset path", DATASET_PATH),
            "revision": ("Dataset revision", DATASET_REVISION),
            "max_concurrency": ("Max instances in flight at once", 4),
            "max_turns": (
                "Max agent turns per instance (default: 64 for vanillux, 100 for openai_agents)",
                64,
            ),
            "command_timeout": ("Timeout in seconds for each agent command", 120.0),
            "agent_timeout": ("Wall-clock budget in seconds for the agent on one instance", 3600.0),
            "max_tool_output_chars": (
                "Truncate each tool output to this many characters (openai_agents)",
                10000,
            ),
            "eval_timeout": ("Timeout in seconds for running the tests", 1800.0),
            "image_timeout": ("Timeout in seconds for pulling and building an image", 900.0),
            "stop_on_submit": ("End the agent's run when it calls submit() (openai_agents)", True),
            "resume_from": (
                "Progress file from an earlier run; instances recorded there are not rerun",
                None,
            ),
            "image_source": (
                "Where to pull instance images: ghcr (no login needed) or dataset (Docker Hub)",
                "ghcr",
            ),
            "cleanup_images": ("Delete each instance's images after grading it", True),
            "include_hints": ("Show the issue's hints to the agent", False),
            "oracle": ("Grade the reference patch instead of running an agent", False),
            "sandbox_mode": ("Sandbox mode: docker, modal", "docker"),
            "enable_compaction": ("Enable context compaction (openai_agents)", False),
            "scaffold": ("Scaffold: vanillux or openai_agents", DEFAULT_SCAFFOLD),
            "max_format_errors": (
                "Replies in a row without a valid tool call before stopping (vanillux)",
                64,
            ),
            "temperature": ("Sampling temperature (vanillux)", 0.7),
            "top_p": ("Nucleus sampling threshold (vanillux)", 0.95),
            "max_tokens": ("Max tokens per model reply (vanillux)", 16384),
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
        """Run the agent on each selected instance and grade its patch."""
        start_time = time.time()
        swe_args = SWEBenchArgs.from_dict(args)

        if not swe_args.oracle:
            from olmo_eval.harness.scaffolds import validate_scaffold

            if swe_args.scaffold not in SCAFFOLD_TOOLS:
                supported = ", ".join(sorted(SCAFFOLD_TOOLS))
                raise ValueError(
                    f"Unsupported scaffold {swe_args.scaffold!r} for {self.name}; "
                    f"use one of {supported}"
                )
            validate_scaffold(swe_args.scaffold)

        instances = select_instances(
            load_instances(swe_args.dataset, swe_args.revision),
            instance_ids=swe_args.instance_ids,
            repos=swe_args.repos,
            limit=swe_args.limit,
        )
        if not instances:
            return self._error_result("No instances selected", start_time)
        if swe_args.image_source not in IMAGE_SOURCES:
            return self._error_result(
                f"Unknown image_source {swe_args.image_source!r}; expected one of {IMAGE_SOURCES}",
                start_time,
            )

        previous: dict[str, InstanceResult] = {}
        if swe_args.resume_from:
            previous = load_progress(swe_args.resume_from)
            logger.info(f"Resuming: {len(previous)} instances already finished")

        progress_path = Path(output_dir) / f"{self.name}_progress.jsonl" if output_dir else None
        if progress_path is not None:
            progress_path.parent.mkdir(parents=True, exist_ok=True)
            # Carry resumed instances over, so this file alone describes the whole run.
            with open(progress_path, "w") as f:
                for inst in instances:
                    if inst.instance_id in previous:
                        f.write(json.dumps(previous[inst.instance_id].to_record()) + "\n")

        semaphore = asyncio.Semaphore(swe_args.max_concurrency)

        async def run(instance: SWEBenchInstance) -> InstanceResult:
            if instance.instance_id in previous:
                return previous[instance.instance_id]
            async with semaphore:
                result = await self._run_instance(instance, provider, swe_args, container_runtime)
            # Recorded as each instance finishes, so a killed job keeps its finished work.
            if progress_path is not None:
                with open(progress_path, "a") as f:
                    f.write(json.dumps(result.to_record()) + "\n")
            return result

        results = await asyncio.gather(*[run(inst) for inst in instances])

        metrics = compute_metrics(results)
        # A run where no instance got through without an error measured nothing.
        all_failed = all(r.error is not None for r in results)
        result = ExternalEvalResult(
            name=self.name,
            success=not all_failed,
            error=f"All {len(results)} instances failed" if all_failed else None,
            metrics=metrics,
            metadata={
                "model_name": provider.model_name,
                "swebench_version": _swebench_version(),
                "num_tasks": len(results),
                **asdict(swe_args),
            },
            duration_seconds=time.time() - start_time,
            predictions=self._build_predictions(results),
        )

        if output_dir:
            self._save_results(result, output_dir)
            self._save_instance_results(results, output_dir)

        return result

    def _sandbox_config(
        self, image: str, mode: SandboxMode, runtime: ContainerRuntime, command_timeout: float
    ) -> SandboxConfig:
        docker_args = tuple(get_docker_network_args(runtime)) if mode == SandboxMode.DOCKER else ()
        return SandboxConfig(
            image=image,
            mode=mode,
            container_runtime=runtime if mode == SandboxMode.DOCKER else "docker",
            working_dir=TESTBED_DIR,
            command_timeout=command_timeout,
            startup_timeout=600.0,
            docker_args=docker_args,
        )

    async def _run_instance(
        self,
        instance: SWEBenchInstance,
        provider: InferenceProvider,
        swe_args: SWEBenchArgs,
        container_runtime: str,
    ) -> InstanceResult:
        """Produce a patch for one instance, then grade it in a fresh container."""
        from olmo_eval.harness.sandbox.image import remove_swerex_image

        result = InstanceResult(
            instance_id=instance.instance_id,
            repo=instance.repo,
            difficulty=instance.difficulty,
        )
        try:
            mode = SandboxMode(swe_args.sandbox_mode)
        except ValueError:
            result.error = f"Invalid sandbox_mode: {swe_args.sandbox_mode!r}"
            return result
        runtime = cast(ContainerRuntime, container_runtime)

        base_image = instance.image_for(swe_args.image_source)
        try:
            await self._solve_and_grade(
                instance, base_image, provider, swe_args, mode, runtime, result
            )
        finally:
            if swe_args.cleanup_images and mode == SandboxMode.DOCKER:
                await asyncio.to_thread(remove_swerex_image, base_image, runtime)
        return result

    async def _solve_and_grade(
        self,
        instance: SWEBenchInstance,
        base_image: str,
        provider: InferenceProvider,
        swe_args: SWEBenchArgs,
        mode: SandboxMode,
        runtime: ContainerRuntime,
        result: InstanceResult,
    ) -> None:
        """Run the agent (or take the gold patch) and grade the patch, filling in ``result``."""
        from olmo_eval.harness.sandbox import SandboxManager
        from olmo_eval.harness.sandbox.image import get_swerex_image

        try:
            # Image builds and pulls block, so keep them off the event loop. The
            # resolver kills its own subprocesses at the timeout; the outer wait is a
            # backstop in case something else in it stalls.
            image = await asyncio.wait_for(
                asyncio.to_thread(
                    get_swerex_image,
                    base_image,
                    runtime,
                    require_registry=mode == SandboxMode.MODAL,
                    timeout=swe_args.image_timeout,
                ),
                timeout=swe_args.image_timeout + 60.0,
            )
        except TimeoutError as e:
            logger.warning(f"[{instance.instance_id}] Image preparation timed out: {e}")
            result.completion_reason = "image_timeout"
            result.error = f"image_timeout: {e}"
            return
        except Exception as e:
            logger.exception(f"[{instance.instance_id}] Failed to prepare image")
            result.error = f"image_error: {e}"
            return

        agent_start = time.time()
        if swe_args.oracle:
            result.patch = instance.gold_patch
            result.completion_reason = "oracle"
        else:
            manager = SandboxManager(
                [self._sandbox_config(image, mode, runtime, swe_args.command_timeout)],
                owner=f"swe-{instance.instance_id}",
            )
            try:
                await manager.start()
                baseline = await snapshot_worktree(manager.get_executor(frozenset()))
                # vanillux stops itself at the agent timeout, so it gets a grace
                # period for a step in flight before being cut off from outside.
                agent_deadline = swe_args.agent_timeout
                if swe_args.scaffold == "vanillux":
                    agent_deadline += AGENT_TIMEOUT_GRACE
                try:
                    await asyncio.wait_for(
                        self._run_agent(manager, instance, provider, swe_args, result),
                        timeout=agent_deadline,
                    )
                except TimeoutError:
                    logger.info(f"[{instance.instance_id}] Agent ran out of time")
                    result.completion_reason = "agent_timeout"
                except Exception as e:
                    if is_context_length_error(e):
                        logger.info(f"[{instance.instance_id}] Agent ran out of context")
                        result.completion_reason = "context_exceeded"
                    else:
                        logger.exception(f"[{instance.instance_id}] Agent run failed")
                        result.error = f"agent_error: {e}"
                # Changes made before an agent failure still count toward the result.
                result.patch = await extract_patch(manager.get_executor(frozenset()), baseline)
            except Exception as e:
                logger.exception(f"[{instance.instance_id}] Agent sandbox failed")
                result.error = result.error or f"sandbox_error: {e}"
            finally:
                await manager.stop()
        result.agent_duration = time.time() - agent_start

        if not result.patch.strip():
            result.grade = GradeResult(empty_patch=True)
            logger.info(
                f"[{instance.instance_id}] empty patch reason={result.completion_reason} "
                f"error={result.error}"
            )
            return

        eval_start = time.time()
        grader = SandboxManager(
            [self._sandbox_config(image, mode, runtime, swe_args.command_timeout)],
            owner=f"swe-{instance.instance_id}-grade",
        )
        try:
            await grader.start()
            result.grade = await grade_in_sandbox(
                grader.get_executor(frozenset()),
                instance,
                result.patch,
                timeout=swe_args.eval_timeout,
            )
        except Exception as e:
            logger.exception(f"[{instance.instance_id}] Grading failed")
            result.error = result.error or f"grading_error: {e}"
        finally:
            await grader.stop()
        result.eval_duration = time.time() - eval_start

        logger.info(
            f"[{instance.instance_id}] resolved={result.resolved} "
            f"applied={result.grade.patch_applied} tests_ran={result.grade.tests_ran} "
            f"reason={result.completion_reason} error={result.error or result.grade.error}"
        )

    async def _run_agent(
        self,
        sandbox_manager: SandboxManager,
        instance: SWEBenchInstance,
        provider: InferenceProvider,
        swe_args: SWEBenchArgs,
        result: InstanceResult,
    ) -> None:
        """Run the agent in the instance container, recording its trajectory on ``result``."""
        from olmo_eval.harness.config import HarnessConfig
        from olmo_eval.harness.scaffolds import get_scaffold
        from olmo_eval.harness.tools import get_tools

        setup = await sandbox_manager.get_executor(Capability.BASH).execute_in_session(
            SESSION_SETUP, timeout=120.0
        )
        if not setup.success:
            raise RuntimeError(f"Session setup failed: {setup.output[-2000:]}")

        scaffold_name = swe_args.scaffold
        scaffold_kwargs: dict[str, Any] = {}
        run_kwargs: dict[str, Any] = {}
        sampling_params: SamplingParams | None = None
        if scaffold_name == "vanillux":
            # vanillux brings its own system and instance prompts.
            system_prompt = None
            content = build_task_text(instance, swe_args.include_hints)
            scaffold_kwargs = {
                "command_timeout": swe_args.command_timeout,
                "max_format_errors": swe_args.max_format_errors,
                "agent_timeout": swe_args.agent_timeout,
            }
            sampling_params = swe_args.sampling_params
        else:
            system_prompt = SYSTEM_PROMPT
            content = build_prompt(instance, swe_args.include_hints)
            run_kwargs = {
                "enable_compaction": swe_args.enable_compaction,
                "max_tool_output_chars": swe_args.max_tool_output_chars,
                "stop_at_tools": ("submit",) if swe_args.stop_on_submit else None,
            }

        harness_config = HarnessConfig(
            name=f"swe_bench_{instance.instance_id}",
            tools=get_tools(SCAFFOLD_TOOLS[scaffold_name]),
            system_prompt=system_prompt,
            max_turns=swe_args.max_turns,
            scaffold=scaffold_name,
            scaffold_kwargs=scaffold_kwargs,
        )
        scaffold = get_scaffold(scaffold_name)
        scaffold.set_sandbox_manager(sandbox_manager)

        request = LMRequest(
            request_type=RequestType.CHAT,
            messages=({"role": "user", "content": content},),
        )
        harness_result = await scaffold.run(
            provider,
            harness_config,
            request,
            sampling_params=sampling_params,
            trace_metadata={"task_id": instance.instance_id},
            **run_kwargs,
        )
        result.trajectory = harness_result.trajectory or AgentTrajectory(turns=())
        reported = harness_result.metadata.get("completion_reason")
        if reported is not None:
            result.completion_reason = COMPLETION_REASONS.get(reported, reported)
        elif harness_result.max_turns_reached:
            result.completion_reason = "max_turns"
        elif called_tool(result.trajectory, "submit"):
            result.completion_reason = "submitted"
        else:
            result.completion_reason = "complete"
        if harness_result.error and not harness_result.max_turns_reached:
            result.error = f"agent_error: {harness_result.error}"

    def _build_predictions(self, results: list[InstanceResult]) -> list[dict[str, Any]]:
        return [
            {
                "native_id": r.instance_id,
                "instance_metrics": {
                    "resolved": {"external": float(r.resolved)},
                    "patch_applied": {"external": float(r.grade.patch_applied)},
                },
                "agent_duration": r.agent_duration,
                "repo": r.repo,
                "difficulty": r.difficulty,
                "completion_reason": r.completion_reason,
                "model_patch": r.patch,
                "error": r.error or r.grade.error,
                "trajectory": r.trajectory.to_dict(),
            }
            for r in results
        ]

    def _save_instance_results(self, results: list[InstanceResult], output_dir: str) -> None:
        """Write per-instance results and predictions in the upstream format."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        details = [
            {
                "instance_id": r.instance_id,
                "repo": r.repo,
                "difficulty": r.difficulty,
                "resolved": r.resolved,
                "patch_applied": r.grade.patch_applied,
                "tests_ran": r.grade.tests_ran,
                "empty_patch": r.grade.empty_patch,
                "completion_reason": r.completion_reason,
                "agent_duration": r.agent_duration,
                "eval_duration": r.eval_duration,
                "error": r.error or r.grade.error,
                "tests_status": r.grade.tests_status,
                "test_output": r.grade.test_output,
            }
            for r in results
        ]
        with open(output_path / f"{self.name}_instances.json", "w") as f:
            json.dump(details, f, indent=2)

        # Predictions in the format the upstream harness accepts, for re-grading.
        with open(output_path / f"{self.name}_predictions.jsonl", "w") as f:
            for r in results:
                record = {
                    "instance_id": r.instance_id,
                    "model_name_or_path": "olmo-eval",
                    "model_patch": r.patch,
                }
                f.write(json.dumps(record) + "\n")
        logger.info(f"Instance results saved to {output_path}")
