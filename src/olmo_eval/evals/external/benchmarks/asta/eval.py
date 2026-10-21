"""ASTA-bench external evaluation implementation."""

from __future__ import annotations

import hashlib
import json
import logging
import shlex
import subprocess
import time
from typing import TYPE_CHECKING, Any

from olmo_eval.common.config import get_infra_config
from olmo_eval.common.openai_credentials import litellm_model_name
from olmo_eval.evals.external.base import SandboxedExternalEval
from olmo_eval.evals.external.benchmarks.asta.args import ASTA_TASKS, AstaArgs
from olmo_eval.evals.external.benchmarks.asta.result_parser import (
    parse_agenteval_json,
    parse_summary_stats_json,
)
from olmo_eval.evals.external.result import ExternalEvalResult

if TYPE_CHECKING:
    from olmo_eval.harness.sandbox.executor import SandboxExecutor
    from olmo_eval.inference.base import InferenceProvider

logger = logging.getLogger(__name__)

ASTA_IMAGE_VERSION = "20260223.2"
ASTA_BENCH_VERSION = "v0.3.1"


def _get_asta_image(container_runtime: str = "docker") -> str:
    """Get or build the ASTA-bench container image.

    Checks local cache first, then registry, then builds locally.
    """
    config = get_infra_config()
    registry = config.swerex_registry

    hash_input = f"asta-bench:{ASTA_BENCH_VERSION}:{ASTA_IMAGE_VERSION}"
    tag_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:12]

    local_image = f"asta-bench-{tag_hash}:latest"

    result = subprocess.run(
        [container_runtime, "image", "inspect", local_image],
        capture_output=True,
    )
    if result.returncode == 0:
        logger.info(f"Using cached ASTA image: {local_image}")
        return local_image

    logger.debug(f"Local image {local_image} not found, checking registry...")

    if registry:
        registry_image = f"{registry}/asta-bench-{tag_hash}:latest"
        result = subprocess.run(
            [container_runtime, "pull", registry_image],
            capture_output=True,
        )
        if result.returncode == 0:
            subprocess.run(
                [container_runtime, "tag", registry_image, local_image],
                capture_output=True,
            )
            logger.info(f"Pulled ASTA image from registry: {registry_image}")
            return local_image
        logger.debug(f"Registry pull failed for {registry_image}")

    logger.info("Building ASTA image locally...")

    dockerfile = f"""\
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends \\
    git curl ca-certificates build-essential && \\
    rm -rf /var/lib/apt/lists/*
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"
WORKDIR /workspace
RUN git clone --recursive --branch {ASTA_BENCH_VERSION} \\
    https://github.com/allenai/asta-bench.git
WORKDIR /workspace/asta-bench
RUN uv sync
RUN uv pip install swe-rex
RUN mkdir -p /workspace/asta-bench/results
ENV PATH="/workspace/asta-bench/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV INSPECT_SANDBOX=local
ENV INSPECT_EVAL_SANDBOX=local
"""

    result = subprocess.run(
        [container_runtime, "build", "-t", local_image, "-"],
        input=dockerfile.encode(),
        capture_output=True,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode() if result.stderr else ""
        raise RuntimeError(f"Failed to build ASTA image: {stderr}")

    logger.info(f"Built ASTA image: {local_image}")

    if registry:
        registry_image = f"{registry}/asta-bench-{tag_hash}:latest"
        logger.info(f"Pushing ASTA image to registry: {registry_image}")
        tag_cmd = [container_runtime, "tag", local_image, registry_image]
        subprocess.run(tag_cmd, capture_output=True)
        push_cmd = [container_runtime, "push", registry_image]
        push_result = subprocess.run(push_cmd, capture_output=True)
        if push_result.returncode == 0:
            logger.info(f"Pushed ASTA image to registry: {registry_image}")
        else:
            stderr = push_result.stderr.decode() if push_result.stderr else ""
            logger.warning(f"Failed to push to registry (using local image): {stderr}")

    return local_image


class AstaExternalEval(SandboxedExternalEval):
    """ASTA-bench evaluation for AI scientist tasks."""

    @property
    def name(self) -> str:
        return "asta_bench"

    @property
    def description(self) -> str:
        return (
            "Evaluates LLM agents on AI scientist tasks including literature search, "
            "code execution, data analysis, and end-to-end discovery. Uses Inspect AI harness."
        )

    def build_sandbox_image(self, container_runtime: str) -> str:
        """Build or fetch the ASTA-bench container image."""
        return _get_asta_image(container_runtime)

    @property
    def working_dir(self) -> str:
        return "/workspace/asta-bench"

    @property
    def timeout_seconds(self) -> float:
        return 14400.0  # 4 hours

    @property
    def setup_command(self) -> tuple[str, ...]:
        return (
            f"cd {self.working_dir} && uv sync",
            f"mkdir -p {self.results_dir}",
        )

    @property
    def required_secrets(self) -> tuple[str, ...]:
        # GOOGLE_API_KEY is required because asta-bench scorers hardcode Google models
        return (
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "GOOGLE_API_KEY",
            "ASTA_TOOL_KEY",
            "HF_TOKEN",
        )

    @property
    def arguments(self) -> dict[str, tuple[str, Any | None]]:
        # Build task list description from ASTA_TASKS
        task_descriptions = []
        for category, tasks in ASTA_TASKS.items():
            task_descriptions.append(f"{category}: {', '.join(tasks)}")
        tasks_help = (
            "Comma-separated task names to run (default: all). "
            f"Available: {'; '.join(task_descriptions)}"
        )
        return {
            "split": ("Dataset split: 'validation' or 'test'", "validation"),
            "tasks": (tasks_help, None),
            "limit": ("Maximum problems per task", None),
            "solver": ("Agent solver type: 'react' or 'basic'", "react"),
            "max_samples": ("Max concurrent problems", 1),
            "max_sandboxes": ("Max parallel sandboxes", 1),
            "max_connections": ("Max model API connections", 8),
            "sandbox_type": ("Sandbox type: 'local' (Beaker) or 'docker'", "local"),
            "temperature": ("Temperature for agent responses", None),
            "max_tokens": ("Max tokens for agent responses", None),
            "extra_args": ("Extra args to pass to inspect eval", None),
            "dump_trajectories": ("Dump agent trajectories to JSON files", True),
        }

    async def execute(
        self,
        provider: InferenceProvider,
        args: dict[str, Any],
        output_dir: str | None = None,
        container_runtime: str = "podman",
    ) -> ExternalEvalResult:
        start_time = time.time()
        asta_args = AstaArgs.from_dict(args)
        all_output: list[str] = []

        # Extract URL and model name from provider
        provider_url = getattr(provider, "base_url", None) or "http://localhost:8000/v1"
        model_name = provider.model_name
        # Detect if this is a locally-deployed server (vLLM) vs external API
        is_local = self._is_local_provider(provider, provider_url)

        try:
            from olmo_eval.harness.sandbox.executor import SandboxExecutor
        except ImportError as e:
            return self._error_result(f"SWE-ReX not installed: {e}", start_time)

        sandbox_config = self._create_sandbox_config_with_env(
            container_runtime, output_dir, asta_args
        )

        result: ExternalEvalResult | None = None

        async with SandboxExecutor(sandbox_config, name=self.name) as executor:
            try:
                if err := await self._run_setup(executor, all_output, start_time):
                    result = err
                elif is_local:
                    provider_url = self._get_provider_url_for_sandbox(provider_url)
                    if not await self._check_provider_health(executor, provider_url):
                        result = self._error_result(
                            f"Provider not reachable at {provider_url}",
                            start_time,
                            "\n".join(all_output),
                        )

                if result is None:
                    run_cmd = self._build_run_command(model_name, provider_url, is_local, asta_args)
                    logger.info(f"[{self.name}] Running: {run_cmd}")

                    run_result = await executor.execute_command(
                        run_cmd, timeout=self.timeout_seconds, stream=True, log_prefix=self.name
                    )
                    all_output.append(f"$ {run_cmd}\n{run_result.output}")
                    logger.info(f"[{self.name}] Run exit code: {run_result.exit_code}")

                    # Run scoring to compile aggregate results
                    # First check if eval_config.json exists (may not for single-task runs)
                    config_check = await executor.execute_command(
                        f"test -f {self.results_dir}/eval_config.json && echo exists",
                        timeout=30.0,
                    )
                    if "exists" not in config_check.output:
                        # Generate config for single-task scoring
                        config_cmd = self._build_config_only_command(asta_args)
                        logger.info(f"[{self.name}] Generating eval config: {config_cmd}")
                        config_result = await executor.execute_command(
                            config_cmd, timeout=120.0, stream=True, log_prefix=f"{self.name}-config"
                        )
                        all_output.append(f"$ {config_cmd}\n{config_result.output}")

                    score_cmd = self._build_score_command()
                    logger.info(f"[{self.name}] Running scoring: {score_cmd}")

                    score_result = await executor.execute_command(
                        score_cmd, timeout=300.0, stream=True, log_prefix=f"{self.name}-score"
                    )
                    all_output.append(f"$ {score_cmd}\n{score_result.output}")
                    logger.info(f"[{self.name}] Score exit code: {score_result.exit_code}")

                    result = await self._extract_results(
                        executor,
                        "\n".join(all_output),
                        run_result.exit_code,
                        output_dir,
                    )

            except Exception as e:
                logger.exception(f"[{self.name}] Execution failed")
                result = self._error_result(str(e), start_time, "\n".join(all_output))

            finally:
                # Always attempt to dump trajectories, even on failure
                if asta_args.dump_trajectories and output_dir:
                    try:
                        await self._dump_trajectories_to_json(executor, output_dir, all_output)
                    except Exception as e:
                        logger.warning(f"[{self.name}] Failed to dump trajectories: {e}")

        if result is None:
            result = self._error_result("No result produced", start_time, "\n".join(all_output))

        result.duration_seconds = time.time() - start_time
        return result

    def _create_sandbox_config_with_env(
        self,
        container_runtime: str,
        output_dir: str | None,
        asta_args: AstaArgs,
    ) -> Any:
        """Create sandbox configuration with ASTA-specific environment variables."""
        import os
        from typing import cast

        from olmo_eval.evals.external.network import get_docker_network_args
        from olmo_eval.harness.sandbox.config import (
            ContainerRuntime,
            SandboxConfig,
            SandboxMode,
        )

        config = get_infra_config()
        env_vars = self._build_env_vars()

        if asta_args.sandbox_type == "local":
            env_vars["INSPECT_SANDBOX"] = "local"
            env_vars["INSPECT_EVAL_SANDBOX"] = "local"

        # Mount shared cache for inspect_evals data (CORE-Bench capsules, etc.)
        # inspect_evals uses ~/.cache/inspect_evals via platformdirs
        volumes: list[tuple[str, str]] = []
        inspect_cache_dir = config.inspect_cache_dir or os.environ.get("INSPECT_CACHE_DIR")
        if inspect_cache_dir:
            # Create directory structure on host with open permissions
            # so the container (running as root) can read/write
            host_cache = os.path.join(inspect_cache_dir, "inspect_evals")
            core_bench_data = os.path.join(host_cache, "CORE-Bench", "data")
            os.makedirs(core_bench_data, exist_ok=True)
            # Ensure all directories in the path are writable
            for path in [host_cache, os.path.dirname(core_bench_data), core_bench_data]:
                os.chmod(path, 0o777)
            # Mount directly to where inspect_evals expects its cache
            container_cache = "/root/.cache/inspect_evals"
            volumes.append((host_cache, container_cache))
            logger.info(f"[{self.name}] Mounting {host_cache} -> {container_cache}")
        else:
            logger.warning(f"[{self.name}] INSPECT_CACHE_DIR not configured")

        log_dir = None
        if output_dir:
            log_dir = os.path.join(output_dir, "logs")

        runtime = cast(ContainerRuntime, container_runtime)
        image = _get_asta_image(container_runtime)

        # Get network args
        docker_args_list = list(get_docker_network_args(runtime=container_runtime))

        return SandboxConfig(
            image=image,
            mode=SandboxMode.DOCKER,
            container_runtime=runtime,
            command_timeout=self.timeout_seconds,
            working_dir=self.working_dir,
            environment=tuple(env_vars.items()),
            volumes=tuple(volumes),
            docker_args=tuple(docker_args_list),
            log_dir=log_dir,
        )

    def _build_run_command(
        self,
        model_name: str,
        provider_url: str,
        is_local: bool,
        asta_args: AstaArgs,
    ) -> str:
        """Build the astabench run command."""
        # Inspect names OpenRouter models "openrouter/<slug>", the same form as LiteLLM.
        model_spec = f"openai-api/vllm/{model_name}" if is_local else litellm_model_name(model_name)

        args = [
            "uv",
            "run",
            "astabench",
            "eval",
            "--model",
            model_spec,
            "--solver",
            asta_args.solver,
            "--max-samples",
            str(asta_args.max_samples),
            "--max-sandboxes",
            str(asta_args.max_sandboxes),
            "--max-connections",
            str(asta_args.max_connections),
            "--display",
            "plain",
            "--log-dir",
            self.results_dir,
            "--split",
            asta_args.split,
        ]

        if asta_args.sandbox_type == "local":
            args.extend(["--sandbox", "local"])

        if asta_args.limit is not None:
            args.extend(["--limit", str(asta_args.limit)])

        if asta_args.temperature is not None:
            args.extend(["-T", f"temperature={asta_args.temperature}"])

        if asta_args.max_tokens is not None:
            args.extend(["-T", f"max_tokens={asta_args.max_tokens}"])

        # Extra args (for task-specific flags like -T with_search_tools=1)
        args.extend(asta_args.extra_args)

        # Task specifications
        for task in asta_args.tasks or []:
            args.append(f"astabench/{task}")

        # Build command with cd and optional env prefix
        # For local vLLM providers, set VLLM_BASE_URL and VLLM_API_KEY
        # This avoids polluting OPENAI_BASE_URL which would affect scorer models
        env_prefix = (
            f"VLLM_BASE_URL={shlex.quote(provider_url)} VLLM_API_KEY=local " if is_local else ""
        )
        return f"cd {self.working_dir} && {env_prefix}{shlex.join(args)}"

    def _build_score_command(self) -> str:
        """Build the astabench score command."""
        return (
            f"cd {self.working_dir} && "
            f"LITELLM_LOCAL_MODEL_COST_MAP=True "
            f"uv run astabench score {self.results_dir}"
        )

    def _build_config_only_command(self, asta_args: AstaArgs) -> str:
        """Build command to generate eval_config.json for single-task runs."""
        args = [
            "uv",
            "run",
            "astabench",
            "eval",
            "--config-only",
            "--log-dir",
            self.results_dir,
            "--split",
            asta_args.split,
        ]
        # Task specifications
        for task in asta_args.tasks or []:
            args.append(f"astabench/{task}")

        return f"cd {self.working_dir} && {shlex.join(args)}"

    async def _dump_trajectories_to_json(
        self,
        executor: SandboxExecutor,
        output_dir: str,
        all_output: list[str],
    ) -> None:
        """Convert .eval log files to JSON for trajectory analysis.

        This runs even on evaluation failure to preserve agent trajectories
        for debugging purposes.
        """
        from pathlib import Path

        # Create trajectories output directory (under asta_results with other task files)
        trajectories_dir = Path(output_dir) / "asta_results" / "trajectories"
        container_trajectories_dir = f"{self.results_dir}/trajectories_json"

        # Find all .eval files in the results directory
        find_cmd = f"find {self.results_dir} -name '*.eval' -type f 2>/dev/null"
        find_result = await executor.execute_command(find_cmd, timeout=60.0)

        if not find_result.success or not find_result.output.strip():
            logger.info(f"[{self.name}] No .eval files found to convert to JSON")
            return

        eval_files = [f.strip() for f in find_result.output.strip().split("\n") if f.strip()]
        logger.info(f"[{self.name}] Found {len(eval_files)} .eval files to convert")

        # Create output directory in container
        mkdir_cmd = f"mkdir -p {container_trajectories_dir}"
        await executor.execute_command(mkdir_cmd, timeout=30.0)

        # Convert each .eval file to JSON using inspect log convert
        converted_count = 0
        for eval_file in eval_files:
            # Use inspect log convert to produce JSON
            convert_cmd = (
                f"cd {self.working_dir} && "
                f"uv run inspect log convert {shlex.quote(eval_file)} "
                f"--to json --output-dir {container_trajectories_dir}"
            )
            logger.debug(f"[{self.name}] Converting: {convert_cmd}")

            convert_result = await executor.execute_command(
                convert_cmd, timeout=120.0, stream=False
            )

            if convert_result.success:
                converted_count += 1
            else:
                logger.warning(
                    f"[{self.name}] Failed to convert {eval_file}: {convert_result.output}"
                )
                all_output.append(f"$ {convert_cmd}\n{convert_result.output}")

        logger.info(f"[{self.name}] Converted {converted_count}/{len(eval_files)} .eval files")

        # Copy converted JSON files to output directory
        list_json_cmd = f"find {container_trajectories_dir} -name '*.json' -type f 2>/dev/null"
        list_result = await executor.execute_command(list_json_cmd, timeout=60.0)

        if not list_result.success or not list_result.output.strip():
            logger.warning(f"[{self.name}] No JSON trajectory files found after conversion")
            return

        json_files = [f.strip() for f in list_result.output.strip().split("\n") if f.strip()]
        trajectories_dir.mkdir(parents=True, exist_ok=True)

        import base64

        for remote_path in json_files:
            # Get filename for local path
            filename = remote_path.split("/")[-1]
            local_path = trajectories_dir / filename

            # Read file content (use base64 for safe transfer)
            read_result = await executor.execute_command(
                f"base64 {shlex.quote(remote_path)}", timeout=120.0
            )

            if read_result.success and read_result.output.strip():
                try:
                    content = base64.b64decode(read_result.output.strip())
                    with open(local_path, "wb") as f:
                        f.write(content)
                    logger.debug(f"[{self.name}] Saved trajectory: {filename}")
                except Exception as e:
                    logger.warning(f"[{self.name}] Failed to save {filename}: {e}")
            else:
                logger.warning(f"[{self.name}] Failed to read {remote_path}")

        logger.info(f"[{self.name}] Saved {len(json_files)} trajectory files to {trajectories_dir}")

    async def _extract_results(
        self,
        executor: SandboxExecutor,
        raw_output: str,
        exit_code: int,
        output_dir: str | None = None,
    ) -> ExternalEvalResult:
        """Extract metrics from astabench score output files.

        Uses summary_stats.json as the primary source for aggregated metrics
        (overall, tag-level, task-level scores) and scores.json for detailed
        per-task metrics and token usage data.
        """
        from pathlib import Path

        metadata: dict[str, Any] = {}
        all_metrics: dict[str, float] = {}

        # Read summary_stats.json first (primary source for aggregated scores)
        summary_path = f"{self.results_dir}/summary_stats.json"
        summary_result = await executor.execute_command(
            f"cat {shlex.quote(summary_path)}", timeout=60.0
        )

        if summary_result.success and summary_result.output.strip():
            try:
                summary_content = json.loads(summary_result.output)
                parsed_summary = parse_summary_stats_json(summary_content)

                # Use summary metrics as primary metrics
                all_metrics.update(parsed_summary["metrics"])
                metadata["overall_score"] = parsed_summary.get("overall_score")
                metadata["tag_scores"] = parsed_summary.get("tag_scores", {})
                metadata["task_scores"] = parsed_summary.get("task_scores", {})

                if output_dir:
                    local_path = Path(output_dir) / "summary_stats.json"
                    local_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(local_path, "w") as f:
                        json.dump(summary_content, f, indent=2)

                logger.info(
                    f"[{self.name}] Parsed summary_stats.json: "
                    f"overall={parsed_summary.get('overall_score')}, "
                    f"{len(parsed_summary.get('task_scores', {}))} tasks"
                )

            except json.JSONDecodeError as e:
                logger.warning(f"[{self.name}] Failed to parse summary_stats.json: {e}")

        # Read scores.json for detailed per-task metrics and token usage
        scores_path = f"{self.results_dir}/scores.json"
        scores_result = await executor.execute_command(
            f"cat {shlex.quote(scores_path)}", timeout=60.0
        )

        if scores_result.success and scores_result.output.strip():
            try:
                scores_content = json.loads(scores_result.output)
                parsed_scores = parse_agenteval_json(scores_content)

                # Add detailed per-task metrics (task_name/metric_name format)
                for metric_name, value in parsed_scores["metrics"].items():
                    # Prefix with "detail/" to distinguish from summary metrics
                    all_metrics[f"detail/{metric_name}"] = value

                # Add token usage to metadata
                metadata["costs"] = parsed_scores.get("costs", {})
                metadata["num_tasks"] = parsed_scores.get("metadata", {}).get("num_tasks", 0)
                metadata["model"] = parsed_scores.get("metadata", {}).get("model", "")
                metadata["solver"] = parsed_scores.get("metadata", {}).get("solver", "")

                if output_dir:
                    local_path = Path(output_dir) / "scores.json"
                    local_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(local_path, "w") as f:
                        json.dump(scores_content, f, indent=2)

                logger.info(
                    f"[{self.name}] Parsed scores.json: "
                    f"{len(parsed_scores['metrics'])} detailed metrics"
                )

            except json.JSONDecodeError as e:
                logger.warning(f"[{self.name}] Failed to parse scores.json: {e}")

        # Check if we got any metrics
        if not all_metrics:
            return ExternalEvalResult(
                name=self.name,
                success=False,
                error="No metrics extracted from summary_stats.json or scores.json",
                raw_output=raw_output,
            )

        # Copy all files from the results directory for debugging
        if output_dir:
            await self._copy_results_directory(executor, output_dir)

        return ExternalEvalResult(
            name=self.name,
            success=exit_code == 0 and bool(all_metrics),
            metrics=all_metrics,
            metadata=metadata,
            raw_output=raw_output,
        )

    async def _copy_results_directory(
        self,
        executor: SandboxExecutor,
        output_dir: str,
    ) -> None:
        """Copy all files from the sandbox results directory to the output directory."""
        import base64
        from pathlib import Path

        # List all files in the results directory (skip .eval files since we dump JSON trajectories)
        list_result = await executor.execute_command(
            f"find {self.results_dir} -type f -name '*.json' 2>/dev/null",
            timeout=60.0,
        )

        if not list_result.success or not list_result.output.strip():
            logger.warning(f"[{self.name}] No result files found in {self.results_dir}")
            return

        files = [f.strip() for f in list_result.output.strip().split("\n") if f.strip()]
        logger.info(f"[{self.name}] Copying {len(files)} result files from sandbox")

        results_subdir = Path(output_dir) / "asta_results"
        results_subdir.mkdir(parents=True, exist_ok=True)

        for remote_path in files:
            # Get relative path from results_dir
            rel_path = remote_path.replace(self.results_dir + "/", "")
            local_path = results_subdir / rel_path
            local_path.parent.mkdir(parents=True, exist_ok=True)

            # Read file content (use base64 for binary-safe transfer)
            read_result = await executor.execute_command(
                f"base64 {shlex.quote(remote_path)}", timeout=120.0
            )

            if read_result.success and read_result.output.strip():
                try:
                    content = base64.b64decode(read_result.output.strip())
                    with open(local_path, "wb") as f:
                        f.write(content)
                    logger.debug(f"[{self.name}] Copied {rel_path}")
                except Exception as e:
                    logger.warning(f"[{self.name}] Failed to copy {rel_path}: {e}")
            else:
                logger.warning(f"[{self.name}] Failed to read {remote_path}")
