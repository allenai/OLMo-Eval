"""SciCode external evaluation.

Implements the SciCode sub-step accuracy benchmark as an ``ExternalEval`` so
that sequential per-sub-step generation runs against the same ``InferenceProvider``
that served the main request — no auxiliary provider required.

Reference: https://scicode-bench.github.io/
Dataset: https://huggingface.co/datasets/SciCode1/SciCode
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from olmo_eval.common.types import LMRequest, RequestType, SamplingParams
from olmo_eval.evals.external.base import ExternalEval
from olmo_eval.evals.external.result import ExternalEvalResult
from olmo_eval.harness.sandbox import SandboxManager
from olmo_eval.harness.sandbox.config import (
    Capability,
    ContainerRuntime,
    SandboxConfig,
    SandboxMode,
)
from olmo_eval.inference.base import InferenceProvider
from olmo_eval.inference.errors import first_output
from olmo_eval.inference.metrics.core.collector import InstrumentedProvider
from olmo_eval.inference.providers.vllm_server import VLLMServerProvider

from . import loader as scicode_loader
from . import prompts as scicode_prompts
from . import verifier as scicode_verifier

logger = logging.getLogger(__name__)


DEFAULT_H5PY_HOST_PATH = ""
DEFAULT_H5PY_CONTAINER_PATH = "/workspace/scicode_test_data.h5"
AI2_H5PY_HOST_PATH = "/weka/oe-adapt-default/finbarrt/scicode/test_data.h5"

_SANDBOX_LOCK_DIR = Path(__file__).parent / "sandbox"
_SANDBOX_DEPS_CONTAINER_DIR = "/opt/scicode-deps"


@dataclass
class SciCodeConfig:
    """Arguments for SciCode external evaluation."""

    split: str = "test"
    problem_ids: list[str] | None = None
    with_background: bool = True
    enable_thinking: bool = False
    max_tokens: int = 16384
    temperature: float = 0.6
    max_concurrency: int = 4
    command_timeout: float = 600.0
    startup_timeout: float = 300.0
    h5py_host_path: str = DEFAULT_H5PY_HOST_PATH
    h5py_container_path: str = DEFAULT_H5PY_CONTAINER_PATH
    sandbox_image: str = "ghcr.io/astral-sh/uv:python3.12-bookworm-slim"


class SciCodeExternalEval(ExternalEval):
    """SciCode multi-step scientific code generation as an external evaluation.

    For each main problem, generates code for each sub-step sequentially via the
    provided ``InferenceProvider`` (hardcoded snippets for problems 13.6, 62.1,
    and 76.3 are injected verbatim without generation). All generated code is
    then concatenated and tested one sub-step at a time in a Python sandbox
    that has ``numpy``, ``scipy``, ``sympy``, ``h5py``, and the SciCode numeric
    reference ``test_data.h5`` available.
    """

    @property
    def name(self) -> str:
        return "scicode"

    @property
    def description(self) -> str:
        return "SciCode: multi-step scientific code generation (65 problems, 288 sub-steps)."

    @property
    def timeout_seconds(self) -> float:
        return 14400.0

    async def execute(
        self,
        provider: InferenceProvider,
        args: dict[str, Any],
        output_dir: str | None = None,
        container_runtime: str = "podman",
    ) -> ExternalEvalResult:
        start_time = time.time()
        sc_args = SciCodeConfig(**args)

        if not sc_args.h5py_host_path:
            hint = (
                f" Ai2 internal users: pass -A h5py_host_path={AI2_H5PY_HOST_PATH}."
                if os.environ.get("BEAKER_TOKEN")
                else ""
            )
            raise ValueError(
                "SciCode requires h5py_host_path to point to test_data.h5 on the host." + hint
            )

        raw_provider = (
            provider.base_provider if isinstance(provider, InstrumentedProvider) else provider
        )
        if not isinstance(raw_provider, VLLMServerProvider):
            raise TypeError(
                f"SciCode requires VLLMServerProvider, got {type(raw_provider).__name__}."
            )
        raw_provider.chat_template_kwargs = (raw_provider.chat_template_kwargs or {}) | {
            "enable_thinking": sc_args.enable_thinking,
        }

        problems = scicode_loader.load_problems(
            split=sc_args.split, problem_ids=sc_args.problem_ids
        )

        sampling_params = SamplingParams(
            max_tokens=sc_args.max_tokens, temperature=sc_args.temperature
        )

        semaphore = asyncio.Semaphore(sc_args.max_concurrency)

        async def run_problem(problem: scicode_loader.SciCodeProblem) -> dict[str, Any]:
            async with semaphore:
                return await self._run_problem(
                    problem=problem,
                    provider=provider,
                    sampling_params=sampling_params,
                    sc_args=sc_args,
                    container_runtime=container_runtime,
                )

        raw_results = await asyncio.gather(
            *[run_problem(p) for p in problems], return_exceptions=True
        )

        problem_results: list[dict[str, Any]] = []
        for problem, raw in zip(problems, raw_results, strict=True):
            if isinstance(raw, BaseException):
                logger.error(
                    "SciCode problem %s failed with exception: %s", problem.problem_id, raw
                )
                hardcoded = scicode_prompts.HARDCODED_SNIPPETS.get(problem.problem_id, {})
                total_scorable = sum(1 for i in range(len(problem.sub_steps)) if i not in hardcoded)
                problem_results.append(
                    {
                        "problem_id": problem.problem_id,
                        "problem_name": problem.problem_name,
                        "step_results": [],
                        "step_codes": {},
                        "step_texts": {},
                        "passed": 0,
                        "total": total_scorable,
                        "all_passed": False,
                        "error": str(raw),
                    }
                )
            else:
                problem_results.append(raw)

        total_sub = sum(pr["total"] for pr in problem_results)
        passed_sub = sum(pr["passed"] for pr in problem_results)
        main_passed = sum(1 for pr in problem_results if pr["all_passed"])

        metrics: dict[str, float] = {
            "sub_step_accuracy": (passed_sub / total_sub) if total_sub else 0.0,
            "main_problem_accuracy": (
                (main_passed / len(problem_results)) if problem_results else 0.0
            ),
            "num_problems": float(len(problem_results)),
            "num_sub_steps": float(total_sub),
            "passed_sub_steps": float(passed_sub),
        }

        result = ExternalEvalResult(
            name=self.name,
            success=True,
            metrics=metrics,
            metadata={
                "model_name": getattr(provider, "model_name", None),
                "split": sc_args.split,
                "with_background": sc_args.with_background,
                "max_tokens": sc_args.max_tokens,
                "temperature": sc_args.temperature,
            },
            duration_seconds=time.time() - start_time,
            predictions=problem_results,
        )

        if output_dir:
            self._save_results(result, output_dir)
        return result

    async def _run_problem(
        self,
        problem: scicode_loader.SciCodeProblem,
        provider: InferenceProvider,
        sampling_params: SamplingParams,
        sc_args: SciCodeConfig,
        container_runtime: str,
    ) -> dict[str, Any]:
        hardcoded = scicode_prompts.HARDCODED_SNIPPETS.get(problem.problem_id, {})
        sub_steps = problem.sub_steps
        # Hardcoded snippets feed generation via previous_llm_code but stay out
        # of step_codes; the verifier receives them separately as hardcoded_prelude.
        previous_llm_code: list[str | None] = [None] * len(sub_steps)
        for idx, snippet in hardcoded.items():
            if idx < len(sub_steps):
                previous_llm_code[idx] = snippet

        step_codes: dict[int, str] = {}
        step_texts: dict[int, str] = {}

        for idx in range(len(sub_steps)):
            if idx in hardcoded:
                continue
            prompt = scicode_prompts.build_step_prompt(
                sub_steps=sub_steps,
                required_dependencies=problem.required_dependencies,
                step_idx=idx,
                previous_llm_code=previous_llm_code,
                with_background=sc_args.with_background,
            )
            request = LMRequest(
                request_type=RequestType.CHAT,
                messages=({"role": "user", "content": prompt},),
            )
            results = await provider.agenerate([request], sampling_params)
            text = first_output(results[0] if results else []).text
            code = scicode_prompts.extract_step_code(text)
            step_texts[idx] = text
            step_codes[idx] = code
            previous_llm_code[idx] = code

        scorable_indices = [i for i in range(len(sub_steps)) if i not in hardcoded]
        total_scorable = len(scorable_indices)
        hardcoded_prelude = "\n\n".join(
            hardcoded[i] for i in sorted(hardcoded) if i < len(sub_steps)
        )
        full_code = "\n\n".join(step_codes[i] for i in sorted(step_codes) if step_codes[i])

        step_results = await self._verify(
            problem=problem,
            scorable_indices=scorable_indices,
            full_code=full_code,
            hardcoded_prelude=hardcoded_prelude,
            sc_args=sc_args,
            container_runtime=container_runtime,
        )

        passed = sum(1 for r in step_results if r)
        return {
            "problem_id": problem.problem_id,
            "problem_name": problem.problem_name,
            "step_results": step_results,
            "step_codes": step_codes,
            "step_texts": step_texts,
            "passed": passed,
            "total": total_scorable,
            "all_passed": total_scorable > 0 and passed == total_scorable,
        }

    async def _verify(
        self,
        problem: scicode_loader.SciCodeProblem,
        scorable_indices: list[int],
        full_code: str,
        hardcoded_prelude: str,
        sc_args: SciCodeConfig,
        container_runtime: str,
    ) -> list[bool]:
        if not scorable_indices:
            return []

        sandbox_config = SandboxConfig(
            image=sc_args.sandbox_image,
            mode=SandboxMode.DOCKER,
            container_runtime=cast(ContainerRuntime, container_runtime),
            capabilities=(Capability.PYTHON | Capability.BASH),
            startup_timeout=sc_args.startup_timeout,
            command_timeout=sc_args.command_timeout,
            inject_swerex=True,
            volumes=(
                (sc_args.h5py_host_path, sc_args.h5py_container_path),
                (
                    str(_SANDBOX_LOCK_DIR / "pyproject.toml"),
                    f"{_SANDBOX_DEPS_CONTAINER_DIR}/pyproject.toml",
                ),
                (
                    str(_SANDBOX_LOCK_DIR / "uv.lock"),
                    f"{_SANDBOX_DEPS_CONTAINER_DIR}/uv.lock",
                ),
            ),
        )
        sandbox_manager = SandboxManager([sandbox_config], owner=f"scicode-{problem.problem_id}")

        results: list[bool] = []
        try:
            await sandbox_manager.start()
            # --active installs into the venv that swerex's image bakes at /root/venv
            # (exported via VIRTUAL_ENV); without it, uv would create a fresh project venv.
            sync_result = await sandbox_manager.execute_code(
                f"uv sync --active --frozen --no-dev --project {_SANDBOX_DEPS_CONTAINER_DIR}",
                language="bash",
                timeout=sc_args.startup_timeout,
            )
            if not sync_result.success:
                raise RuntimeError(f"SciCode sandbox dep install failed: {sync_result.output}")
            for idx in scorable_indices:
                step = problem.sub_steps[idx]
                script = scicode_verifier.build_step_script(
                    step=step,
                    required_dependencies=problem.required_dependencies,
                    full_code=full_code,
                    hardcoded_prelude=hardcoded_prelude,
                    h5py_file=sc_args.h5py_container_path,
                )
                exec_result = await sandbox_manager.execute_code(
                    script, language="python", timeout=sc_args.command_timeout
                )
                results.append(bool(exec_result.success))
        finally:
            await sandbox_manager.stop()
        return results
