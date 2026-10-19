"""AA-Briefcase-Lite external evaluation."""

from __future__ import annotations

import asyncio
import logging
import shlex
import shutil
import statistics
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from olmo_eval.common.types import LMRequest, RequestType, SamplingParams
from olmo_eval.common.types.trajectory import AgentTrajectory
from olmo_eval.evals.external.args import as_list, as_optional_float, as_optional_int
from olmo_eval.evals.external.base import ExternalEval
from olmo_eval.evals.external.result import ExternalEvalResult
from olmo_eval.harness.sandbox.config import ContainerRuntime, SandboxConfig, SandboxMode
from olmo_eval.harness.scaffolds.stirrup import (
    ABANDON_TOOL_NAME,
    CODE_EXEC_TOOL_NAME,
    DIAGNOSTIC_COUNTS,
    FINISH_TOOL_NAME,
    StirrupScaffold,
)

from .judge import (
    DEFAULT_JUDGE_MAX_TOKENS,
    DEFAULT_JUDGE_MODEL,
    DEFAULT_JUDGE_REASONING_EFFORT,
    CheckVerdict,
    ParsedSubmission,
    build_judge_fn,
    grade_check,
)
from .loader import (
    DATASET_REVISION,
    BriefcaseScenario,
    BriefcaseTask,
    download_scenario,
    load_scenario,
    select_tasks,
    stage_sources,
)

if TYPE_CHECKING:
    from olmo_eval.common.scorers.llm_judge import ConfiguredJudgeFn
    from olmo_eval.harness.sandbox import SandboxManager
    from olmo_eval.inference.base import InferenceProvider

logger = logging.getLogger(__name__)

SANDBOX_CONTEXT = Path(__file__).parent / "sandbox"
HOME_DIR = "/home/user"
OUTBOX_DIR = "/mnt/outbox"
SUBMISSION_DIR = "/mnt/submission"
PARSED_DIR = "/mnt/parsed"
PARSER = "/opt/briefcase/parse_submission.py"

TOOLS = (CODE_EXEC_TOOL_NAME, FINISH_TOOL_NAME, ABANDON_TOOL_NAME)

#: Extra wall-clock time, past the agent timeout, before a run is cut off from
#: outside. The scaffold stops itself at the timeout between turns; this bounds a
#: turn that is already in flight.
AGENT_TIMEOUT_GRACE = 300.0
SANDBOX_STARTUP_TIMEOUT = 300.0
PARSE_TIMEOUT = 1800.0


@dataclass
class BriefcaseArgs:
    """Arguments for the AA-Briefcase-Lite evaluation."""

    task_ids: list[str] | None = None
    revision: str = DATASET_REVISION
    n_attempts: int = 1
    max_concurrency: int = 4
    max_turns: int = 500
    command_timeout: float = 1200.0
    agent_timeout: float | None = None
    context_window_tokens: int | None = None
    summarization_cutoff: float = 0.7
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 32768
    judge_model: str = DEFAULT_JUDGE_MODEL
    judge_reasoning_effort: str | None = DEFAULT_JUDGE_REASONING_EFFORT
    judge_max_tokens: int = DEFAULT_JUDGE_MAX_TOKENS
    judge_concurrency: int = 8
    image_timeout: float = 3600.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BriefcaseArgs:
        defaults = cls()
        effort = data.get("judge_reasoning_effort", defaults.judge_reasoning_effort)
        return cls(
            task_ids=as_list(data.get("task_ids")),
            revision=str(data.get("revision", defaults.revision)),
            n_attempts=int(data.get("n_attempts", defaults.n_attempts)),
            max_concurrency=int(data.get("max_concurrency", defaults.max_concurrency)),
            max_turns=int(data.get("max_turns", defaults.max_turns)),
            command_timeout=float(data.get("command_timeout", defaults.command_timeout)),
            agent_timeout=as_optional_float(data.get("agent_timeout")),
            context_window_tokens=as_optional_int(data.get("context_window_tokens")),
            summarization_cutoff=float(
                data.get("summarization_cutoff", defaults.summarization_cutoff)
            ),
            temperature=float(data.get("temperature", defaults.temperature)),
            top_p=float(data.get("top_p", defaults.top_p)),
            max_tokens=int(data.get("max_tokens", defaults.max_tokens)),
            judge_model=str(data.get("judge_model", defaults.judge_model)),
            judge_reasoning_effort=effort if effort not in ("", "none", "None") else None,
            judge_max_tokens=int(data.get("judge_max_tokens", defaults.judge_max_tokens)),
            judge_concurrency=int(data.get("judge_concurrency", defaults.judge_concurrency)),
            image_timeout=float(data.get("image_timeout", defaults.image_timeout)),
        )

    @property
    def sampling_params(self) -> SamplingParams:
        # A long run's history leaves less room for the reply than a fresh one.
        return SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            fit_max_tokens_to_context=True,
        )


@dataclass
class TrialResult:
    """One agent run on one task, and the grading of what it submitted."""

    task_id: str
    attempt: int
    completion_reason: str = "error"
    finish: dict[str, Any] | None = None
    #: Deliverable file names collected from the finish call.
    submitted: list[str] = field(default_factory=list)
    verdicts: list[CheckVerdict] = field(default_factory=list)
    trajectory: AgentTrajectory = field(default_factory=lambda: AgentTrajectory(turns=()))
    agent_steps: int = 0
    summaries: int = 0
    #: How the run spent its turns: dropped turns, replies without a tool call, and
    #: invalid, unknown or refused calls.
    agent_counts: dict[str, int] = field(default_factory=dict)
    agent_duration: float = 0.0
    grading_duration: float = 0.0
    #: Notes the parser left on the submission, e.g. a file it could not read.
    parser_notes: list[str] = field(default_factory=list)
    #: A failure of the harness, sandbox, model server or judge. A trial with an
    #: error is left out of the metrics.
    error: str | None = None

    @property
    def native_id(self) -> str:
        return f"{self.task_id}:{self.attempt}"

    @property
    def graded(self) -> list[CheckVerdict]:
        return [v for v in self.verdicts if v.passed is not None]

    @property
    def pass_rate(self) -> float | None:
        graded = self.graded
        return sum(bool(v.passed) for v in graded) / len(graded) if graded else None


def build_agent_prompts(
    scenario: BriefcaseScenario, task: BriefcaseTask, max_turns: int
) -> tuple[str, str]:
    """Fill the dataset's agent system and task prompts for one task."""
    names = {"{finish_tool_name}": FINISH_TOOL_NAME, "{abandon_task_finish}": ABANDON_TOOL_NAME}
    system = scenario.agent_system_prompt.replace("{max_turns}", str(max_turns))
    instruction = scenario.agent_task_prompt
    for placeholder, value in names.items():
        system = system.replace(placeholder, value)
        instruction = instruction.replace(placeholder, value)
    deliverables = "\n".join(f"- {name}" for name in task.deliverables)
    instruction = (
        instruction.replace("{scenario_overview}", scenario.scenario_overview.strip())
        .replace("{week_overview}", scenario.week_overview.strip())
        .replace("{task}", task.brief.strip())
        .replace("{expected_output_filenames}", deliverables)
    )
    return system, instruction


def deliverable_sources(deliverables: tuple[str, ...], paths: list[str]) -> dict[str, str]:
    """Map each deliverable to the finish path that submits it.

    A path submits a deliverable when its file name matches; the last such path wins.
    """
    sources: dict[str, str] = {}
    for path in paths:
        name = path.rstrip("/").rsplit("/", 1)[-1]
        if name in deliverables:
            sources[name] = path
    return sources


def bind_mount(source: Path, target: str, read_only: bool) -> tuple[str, str]:
    spec = f"type=bind,src={source},dst={target}"
    return ("--mount", f"{spec},readonly" if read_only else spec)


def compute_metrics(trials: list[TrialResult]) -> dict[str, float]:
    """Pass rates over the graded checks of the trials that ran without an error."""
    scored = [t for t in trials if t.error is None]
    verdicts = [v for t in scored for v in t.graded]

    def rate(values: list[CheckVerdict]) -> float:
        return sum(bool(v.passed) for v in values) / len(values) if values else 0.0

    metrics: dict[str, float] = {"rubric_pass_rate": rate(verdicts)}
    for task_id in sorted({t.task_id for t in scored}):
        metrics[f"rubric_pass_rate_{task_id}"] = rate(
            [v for t in scored if t.task_id == task_id for v in t.graded]
        )
    metrics["accuracy_pass_rate"] = rate([v for v in verdicts if v.check_type == "A"])
    metrics["critical_insight_pass_rate"] = rate([v for v in verdicts if v.check_type == "C"])

    attempts = sorted({t.attempt for t in scored})
    if len(attempts) > 1:
        per_attempt = [
            rate([v for t in scored if t.attempt == a for v in t.graded]) for a in attempts
        ]
        metrics["rubric_pass_rate_stderr"] = statistics.stdev(per_attempt) / len(attempts) ** 0.5

    n = len(scored)
    metrics["finish_rate"] = (
        sum(t.completion_reason == "finished" for t in scored) / n if n else 0.0
    )
    metrics["abandon_rate"] = (
        sum(t.completion_reason == "abandoned" for t in scored) / n if n else 0.0
    )
    metrics["num_trials"] = float(len(trials))
    metrics["num_failed_trials"] = float(len(trials) - n)
    metrics["num_graded_checks"] = float(len(verdicts))
    metrics["num_ungraded_checks"] = float(sum(len(t.verdicts) - len(t.graded) for t in scored))
    return metrics


class AABriefcaseLiteExternalEval(ExternalEval):
    """AA-Briefcase-Lite: one sandboxed agent run per task, then binary rubric grading."""

    @property
    def name(self) -> str:
        return "aa_briefcase_lite"

    @property
    def description(self) -> str:
        return (
            "Evaluates LLM agents on the public week of Artificial Analysis's AA-Briefcase: "
            "due-diligence deliverables graded against binary rubric checks"
        )

    @property
    def timeout_seconds(self) -> float:
        return 24 * 3600.0

    @property
    def required_secrets(self) -> tuple[str, ...]:
        return ("OPENAI_API_KEY",)

    @property
    def extras(self) -> tuple[str, ...]:
        # The rubric judge calls the OpenAI API.
        return ("clients",)

    @property
    def arguments(self) -> dict[str, tuple[str, Any | None]]:
        defaults = BriefcaseArgs()
        return {
            "task_ids": ("Comma-separated task IDs to run, e.g. w1_t2 (default: all four)", None),
            "revision": ("Dataset revision", defaults.revision),
            "n_attempts": ("Agent runs per task", defaults.n_attempts),
            "max_concurrency": ("Max agent runs in flight at once", defaults.max_concurrency),
            "max_turns": ("Max model replies per run", defaults.max_turns),
            "command_timeout": ("Timeout in seconds for each command", defaults.command_timeout),
            "agent_timeout": (
                "Wall-clock budget in seconds for one run; one that runs out submits nothing",
                None,
            ),
            "context_window_tokens": (
                "Context window used to decide when to summarize (default: the server's)",
                None,
            ),
            "summarization_cutoff": (
                "Fraction of the context window that triggers a summary",
                defaults.summarization_cutoff,
            ),
            "temperature": ("Sampling temperature", defaults.temperature),
            "top_p": ("Nucleus sampling threshold", defaults.top_p),
            "max_tokens": ("Max tokens per model reply", defaults.max_tokens),
            "judge_model": ("Rubric judge model", defaults.judge_model),
            "judge_reasoning_effort": (
                "Rubric judge reasoning effort",
                defaults.judge_reasoning_effort,
            ),
            "judge_max_tokens": ("Max tokens per judge reply", defaults.judge_max_tokens),
            "judge_concurrency": ("Max judge calls in flight", defaults.judge_concurrency),
            "image_timeout": (
                "Timeout in seconds for building the sandbox image",
                defaults.image_timeout,
            ),
        }

    @property
    def scaffold(self) -> str | None:
        return "stirrup"

    async def execute(
        self,
        provider: InferenceProvider,
        args: dict[str, Any],
        output_dir: str | None = None,
        container_runtime: str = "podman",
    ) -> ExternalEvalResult:
        """Run the agent on each selected task, then grade what it submitted."""
        start_time = time.time()
        bc_args = BriefcaseArgs.from_dict(args)
        runtime = cast(ContainerRuntime, container_runtime)

        root = await asyncio.to_thread(download_scenario, bc_args.revision)
        scenario = load_scenario(root)
        tasks = select_tasks(scenario.tasks, bc_args.task_ids)
        # Fail before any agent runs if the judge cannot be reached.
        self._build_env_vars()

        try:
            image = await asyncio.to_thread(self._sandbox_image, runtime, bc_args.image_timeout)
        except Exception as e:
            logger.exception("Could not prepare the AA-Briefcase-Lite sandbox image")
            return self._error_result(f"image_error: {e}", start_time)

        judge_fn = build_judge_fn(
            bc_args.judge_model, bc_args.judge_reasoning_effort, bc_args.judge_max_tokens
        )
        judge_limiter = asyncio.Semaphore(max(1, bc_args.judge_concurrency))
        trial_limiter = asyncio.Semaphore(max(1, bc_args.max_concurrency))

        with tempfile.TemporaryDirectory(prefix="aa-briefcase-") as scratch:
            artifacts = Path(output_dir) / "aa_briefcase_lite" if output_dir else Path(scratch)

            async def run(task: BriefcaseTask, attempt: int) -> TrialResult:
                async with trial_limiter:
                    return await self._run_trial(
                        task,
                        attempt,
                        scenario,
                        provider,
                        image,
                        runtime,
                        bc_args,
                        judge_fn,
                        judge_limiter,
                        Path(scratch),
                        artifacts / f"{task.task_id}-{attempt}",
                    )

            trials = await asyncio.gather(
                *[run(task, a) for task in tasks for a in range(bc_args.n_attempts)]
            )

        failed = [t for t in trials if t.error is not None]
        error = None
        if failed:
            details = "; ".join(f"{t.native_id}: {t.error}" for t in failed)
            error = f"{len(failed)} of {len(trials)} trials failed: {details}"
        result = ExternalEvalResult(
            name=self.name,
            # A failed trial is not a score, so a run with one does not stand as a result.
            success=not failed,
            error=error,
            metrics=compute_metrics(trials),
            metadata={
                "model_name": provider.model_name,
                "num_tasks": len(trials),
                "image": image,
                "judge_config": dict(judge_fn.judge_config),
                "pairwise_checks_skipped": scenario.pairwise_checks,
                **asdict(bc_args),
            },
            duration_seconds=time.time() - start_time,
            predictions=[self._prediction(t) for t in trials],
        )
        if output_dir:
            self._save_results(result, output_dir)
        return result

    def _sandbox_image(self, runtime: str, timeout: float) -> str:
        from olmo_eval.harness.sandbox.image import build_task_image, get_swerex_image

        base = build_task_image(SANDBOX_CONTEXT, runtime, "aa-briefcase-sandbox", timeout)
        return get_swerex_image(base, runtime, pristine=True, timeout=timeout)

    def _sandbox_config(
        self,
        image: str,
        runtime: ContainerRuntime,
        command_timeout: float,
        mounts: list[tuple[Path, str, bool]],
    ) -> SandboxConfig:
        docker_args: tuple[str, ...] = ()
        for source, target, read_only in mounts:
            docker_args += bind_mount(source, target, read_only)
        return SandboxConfig(
            image=image,
            mode=SandboxMode.DOCKER,
            container_runtime=runtime,
            working_dir=HOME_DIR,
            startup_timeout=SANDBOX_STARTUP_TIMEOUT,
            command_timeout=command_timeout,
            docker_args=docker_args,
            pristine_image=True,
        )

    async def _run_trial(
        self,
        task: BriefcaseTask,
        attempt: int,
        scenario: BriefcaseScenario,
        provider: InferenceProvider,
        image: str,
        runtime: ContainerRuntime,
        bc_args: BriefcaseArgs,
        judge_fn: ConfiguredJudgeFn,
        judge_limiter: asyncio.Semaphore,
        scratch: Path,
        artifacts: Path,
    ) -> TrialResult:
        """Run the agent on one task, collect its deliverables, and grade them."""
        trial = TrialResult(task_id=task.task_id, attempt=attempt)
        sources = scratch / f"{task.task_id}-{attempt}"
        # Containers may run as another user than this process, so the directories
        # they write to are open to all.
        outbox = artifacts / "submission"
        parsed = artifacts / "parsed"
        for directory in (outbox, parsed):
            # A rerun into the same output directory starts from nothing.
            shutil.rmtree(directory, ignore_errors=True)
            directory.mkdir(parents=True)
            directory.chmod(0o777)
        await asyncio.to_thread(stage_sources, scenario.root, task, sources)

        agent_start = time.time()
        try:
            await self._run_agent_container(
                task, scenario, provider, image, runtime, bc_args, sources, outbox, trial
            )
        except Exception as e:
            logger.exception(f"[{trial.native_id}] Agent sandbox failed")
            trial.error = trial.error or f"sandbox_error: {e}"
        trial.agent_duration = time.time() - agent_start
        if trial.error is not None:
            return trial

        grading_start = time.time()
        submission: ParsedSubmission | None = None
        try:
            if trial.submitted:
                submission = await self._parse_submission(image, runtime, outbox, parsed)
                trial.parser_notes = [
                    f"{name}: {block['text']}"
                    for name, blocks in submission.files.items()
                    for block in blocks
                    if block["type"] == "note"
                ]
            trial.verdicts = list(
                await asyncio.gather(
                    *[
                        grade_check(
                            judge_fn, scenario, task.brief, check, submission, judge_limiter
                        )
                        for check in task.checks
                    ]
                )
            )
        except Exception as e:
            logger.exception(f"[{trial.native_id}] Grading failed")
            trial.error = f"grading_error: {e}"
        trial.grading_duration = time.time() - grading_start
        ungraded = [v for v in trial.verdicts if v.passed is None]
        if ungraded:
            trial.error = f"{len(ungraded)} checks could not be graded: {ungraded[0].error}"
        logger.info(
            f"[{trial.native_id}] {trial.completion_reason}, submitted={trial.submitted}, "
            f"pass_rate={trial.pass_rate}, error={trial.error}"
        )
        return trial

    async def _run_agent_container(
        self,
        task: BriefcaseTask,
        scenario: BriefcaseScenario,
        provider: InferenceProvider,
        image: str,
        runtime: ContainerRuntime,
        bc_args: BriefcaseArgs,
        sources: Path,
        outbox: Path,
        trial: TrialResult,
    ) -> None:
        """Run the agent, then copy the deliverables its finish call names to the outbox."""
        from olmo_eval.harness.sandbox import SandboxManager

        scaffold = StirrupScaffold()

        config = self._sandbox_config(
            image,
            runtime,
            bc_args.command_timeout,
            [
                (sources / "shared", f"{HOME_DIR}/shared", True),
                (sources / "week", f"{HOME_DIR}/week", True),
                (outbox, OUTBOX_DIR, False),
            ],
        )
        manager = SandboxManager([config], owner=f"briefcase-{trial.native_id}")
        try:
            await manager.start()
            scaffold.set_sandbox_manager(manager)
            agent = self._run_agent(scaffold, task, scenario, provider, bc_args, trial)
            if bc_args.agent_timeout is None:
                await agent
            else:
                try:
                    await asyncio.wait_for(
                        agent, timeout=bc_args.agent_timeout + AGENT_TIMEOUT_GRACE
                    )
                except TimeoutError:
                    # The scaffold stops itself at the timeout between turns, so a run
                    # still going here is stuck in a model or sandbox call.
                    trial.error = "agent_stuck: the run outlasted its timeout"
                    trial.trajectory = scaffold.trajectory_so_far()
                    return
            if trial.completion_reason == "finished" and trial.finish is not None:
                await self._collect(manager, task, trial)
        finally:
            await manager.stop()

    async def _run_agent(
        self,
        scaffold: StirrupScaffold,
        task: BriefcaseTask,
        scenario: BriefcaseScenario,
        provider: InferenceProvider,
        bc_args: BriefcaseArgs,
        trial: TrialResult,
    ) -> None:
        from olmo_eval.harness.config import HarnessConfig
        from olmo_eval.harness.tools import get_tools

        system_prompt, instruction = build_agent_prompts(scenario, task, bc_args.max_turns)
        scaffold_kwargs: dict[str, Any] = {
            "command_timeout": bc_args.command_timeout,
            "summarization_cutoff": bc_args.summarization_cutoff,
        }
        if bc_args.agent_timeout is not None:
            scaffold_kwargs["agent_timeout"] = bc_args.agent_timeout
        if bc_args.context_window_tokens is not None:
            scaffold_kwargs["context_window_tokens"] = bc_args.context_window_tokens
        harness_config = HarnessConfig(
            name=f"aa_briefcase_{task.task_id}",
            tools=get_tools(TOOLS),
            system_prompt=system_prompt,
            max_turns=bc_args.max_turns,
            scaffold="stirrup",
            scaffold_kwargs=scaffold_kwargs,
        )
        harness_result = await scaffold.run(
            provider,
            harness_config,
            LMRequest(
                request_type=RequestType.CHAT,
                messages=({"role": "user", "content": instruction},),
            ),
            sampling_params=bc_args.sampling_params,
            trace_metadata={"task_id": trial.native_id},
        )
        trial.trajectory = harness_result.trajectory or AgentTrajectory(turns=())
        trial.completion_reason = str(harness_result.metadata.get("completion_reason", "error"))
        trial.finish = harness_result.metadata.get("finish")
        trial.agent_steps = int(harness_result.metadata.get("steps", 0))
        trial.summaries = int(harness_result.metadata.get("summaries", 0))
        trial.agent_counts = {
            key: int(harness_result.metadata.get(key, 0))
            for key in ("dropped_turns", *DIAGNOSTIC_COUNTS)
        }
        # Running out of turns, time or context is the agent's failure; an error is
        # the model server's.
        if harness_result.error:
            trial.error = f"agent_error: {harness_result.error}"

    async def _collect(
        self, manager: SandboxManager, task: BriefcaseTask, trial: TrialResult
    ) -> None:
        """Copy each deliverable the finish call names into the outbox."""
        from olmo_eval.harness.sandbox import Capability

        assert trial.finish is not None
        sources = deliverable_sources(task.deliverables, trial.finish.get("paths", []))
        for name, path in sources.items():
            result = await manager.execute_command(
                f"cp -- {shlex.quote(path)} {shlex.quote(f'{OUTBOX_DIR}/{name}')}",
                120.0,
                capabilities=Capability.BASH,
            )
            if not result.success:
                raise RuntimeError(f"Could not collect {path}: {result.output[-500:]}")
            trial.submitted.append(name)

    async def _parse_submission(
        self, image: str, runtime: ContainerRuntime, outbox: Path, parsed: Path
    ) -> ParsedSubmission:
        """Parse the deliverables in a fresh container the agent never touched."""
        from olmo_eval.harness.sandbox import SandboxManager

        config = self._sandbox_config(
            image,
            runtime,
            PARSE_TIMEOUT,
            [(outbox, SUBMISSION_DIR, True), (parsed, PARSED_DIR, False)],
        )
        manager = SandboxManager([config], owner=f"briefcase-parse-{outbox.parent.name}")
        try:
            await manager.start()
            # Streamed, because parsing a deck or a video can outlast a single request.
            result = await manager.get_executor(frozenset()).execute_command(
                f"python {PARSER} {SUBMISSION_DIR} {PARSED_DIR}",
                PARSE_TIMEOUT,
                stream=True,
                log_prefix=f"briefcase-parse-{outbox.parent.name}",
            )
            if not result.success:
                raise RuntimeError(f"Parsing the submission failed: {result.output[-2000:]}")
        finally:
            await manager.stop()
        return ParsedSubmission.load(parsed)

    def _prediction(self, trial: TrialResult) -> dict[str, Any]:
        pass_rate = trial.pass_rate
        instance_metrics: dict[str, Any] = {}
        if trial.error is None and pass_rate is not None:
            instance_metrics["rubric_pass_rate"] = {"external": pass_rate}
        return {
            "native_id": trial.native_id,
            "task_id": trial.task_id,
            "attempt": trial.attempt,
            "instance_metrics": instance_metrics,
            "completion_reason": trial.completion_reason,
            "finish": trial.finish,
            "submitted": trial.submitted,
            "checks": [v.to_dict() for v in trial.verdicts],
            "parser_notes": trial.parser_notes,
            "agent_steps": trial.agent_steps,
            "summaries": trial.summaries,
            "agent_counts": trial.agent_counts,
            "agent_duration": trial.agent_duration,
            "grading_duration": trial.grading_duration,
            "error": trial.error,
            "trajectory": trial.trajectory.to_dict(),
        }
