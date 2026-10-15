"""DeepScholar-Bench external evaluation implementation.

DeepScholar-Bench (arXiv 2508.20033) evaluates generative research synthesis:
given a paper's context, a system retrieves prior work and writes the
related-work section, scored on organization, nugget coverage, reference
coverage, and citation precision. We run it in two phases inside one sandbox:
generation (the model under test, driven through LOTUS) then eval (an external
judge model scoring the generated sections).

Modeled on the tau2 external eval: a stock uv image, repo cloned at setup, model
under test wired in via a generated LOTUS config rather than a CLI flag.

Repository: https://github.com/guestrin-lab/deepscholar-bench
"""

from __future__ import annotations

import base64
import json
import logging
import os
import shlex
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from olmo_eval.common.openai_credentials import litellm_model_name, openai_endpoint_model_name
from olmo_eval.evals.external.base import SandboxedExternalEval
from olmo_eval.evals.external.benchmarks.deepscholar.args import DeepScholarArgs
from olmo_eval.evals.external.benchmarks.deepscholar.result_parser import (
    compute_geomean,
    parse_aggregate_csv,
    parse_per_query_csv,
)
from olmo_eval.evals.external.result import ExternalEvalResult

if TYPE_CHECKING:
    from olmo_eval.harness.sandbox.executor import SandboxExecutor
    from olmo_eval.inference.base import InferenceProvider

logger = logging.getLogger(__name__)

# Pinned: the search shim monkeypatches upstream internals
# (recursive_search._process_single_lotus_search_task), so track a fixed commit.
DEEPSCHOLAR_REF = "c95413b3b2f3255b461b90d0ce650f685ae2d1ff"
# DeepScholar's requirements file follows LOTUS HEAD, but the runtime shim patches
# LOTUS internals. Pin the exact revision used by the successful full validation.
LOTUS_REF = "136ae4f4a344a2f75d89f811e516dfcb0de30e46"

# Default token budget for upstream's stage LMs (raising the upstream 512 cap that
# truncates structured outputs). Kept well below max_model_len: LOTUS sends this as
# max_completion_tokens, and vLLM rejects prompt_tokens + max_completion_tokens >
# context, so a large budget would starve prompt room on stage prompts that
# aggregate many reference abstracts. Override with -a stage_max_tokens.
DEFAULT_STAGE_MAX_TOKENS = 4096

# Upstream eval/evaluator/coverage_relevance_rate.py reports the mean graded
# relevance (score in {0, 1, 2}) over deduplicated retrieved sources, omitting the
# 1/(2|S|) normalization the paper defines (arXiv:2508.20033: RR = 1/(2|S|) * sum
# Rel), so its values live on a 0-2 scale and can exceed 1.0. Halve them so the
# reported metric and the geomeans match the paper's 0-1 scale. (The upstream
# leaderboard instead clips to 1.0; we follow the paper for geomean fidelity.)
_HALF_SCALE_METRICS = ("coverage_relevance_rate",)

# Mirrors configs/deepscholar_base.yaml; the `lm` block is filled in per run.
_BASE_CONFIG: dict[str, Any] = {
    "queries_file": "dataset/queries.csv",
    "max_search_retries": 3,
    "use_structured_output": True,
    "enable_web_search": True,
    "per_query_max_search_results_count": 10,
    "use_responses_model": None,
    "num_search_steps": 3,
    "num_search_queries_per_step_per_corpus": 2,
    "use_sem_filter": True,
    "use_sem_topk": True,
    "final_max_results_count": 30,
    "categorize_references": True,
    "generate_category_summary": True,
    "generate_insights": True,
}


class DeepScholarExternalEval(SandboxedExternalEval):
    """DeepScholar-Bench evaluation for generative research synthesis."""

    @property
    def name(self) -> str:
        return "deepscholar_bench"

    @property
    def description(self) -> str:
        return (
            "Evaluates generative research synthesis: given a paper's context, the model "
            "retrieves prior work and writes a related-work section, scored on organization, "
            "nugget/reference coverage, and citation precision (geometric mean)."
        )

    @property
    def sandbox_image(self) -> str:
        return "ghcr.io/astral-sh/uv:python3.10-bookworm"

    @property
    def working_dir(self) -> str:
        return "/workspace"

    @property
    def timeout_seconds(self) -> float:
        # Full validation takes roughly two hours, but retrieval and model latency
        # vary substantially. Keep each phase below the 24h Beaker job cap with
        # enough headroom for transient backend slowdowns.
        return 28800.0

    @property
    def _repo(self) -> str:
        return f"{self.working_dir}/deepscholar-bench"

    @property
    def _venv_python(self) -> str:
        return f"{self._repo}/.venv/bin/python"

    @property
    def _gen_dir(self) -> str:
        return f"{self._repo}/outputs/generation"

    @property
    def _eval_dir(self) -> str:
        return f"{self._repo}/outputs/evaluation"

    @property
    def _config_path(self) -> str:
        return f"{self._repo}/olmo_eval_config.yaml"

    @property
    def _shim_path(self) -> str:
        return f"{self._repo}/olmo_eval_search_shim.py"

    @property
    def _organization_evaluator_path(self) -> str:
        return f"{self._repo}/eval/evaluator/organization.py"

    @property
    def _citation_utils_path(self) -> str:
        return f"{self._repo}/eval/utils.py"

    @property
    def setup_command(self) -> tuple[str, ...]:
        repo_url = "https://github.com/guestrin-lab/deepscholar-bench.git"
        # The repo is ~1.3GB (dataset CSVs + baseline outputs), so a full clone is
        # slow. Shallow-fetch just the target ref (works for a branch name or a
        # commit SHA); no submodules exist. checkout FETCH_HEAD lands the snapshot.
        return (
            f"git init {self._repo}",
            f"cd {self._repo} && git remote add origin {repo_url}",
            f"cd {self._repo} && git fetch --depth 1 origin {DEEPSCHOLAR_REF}",
            f"cd {self._repo} && git checkout FETCH_HEAD",
            f"cd {self._repo} && sed -i "
            f"'s|git+https://github.com/lotus-data/lotus.git#egg=|"
            f"git+https://github.com/lotus-data/lotus.git@{LOTUS_REF}#egg=|' requirements.txt "
            f"&& grep -F '@{LOTUS_REF}#egg=' requirements.txt",
            f"cd {self._repo} && uv venv --python 3.10",
            # Target our .venv explicitly: the swe-rex derived image ships an active
            # /root/venv (3.12), which uv would otherwise install into by default.
            f"cd {self._repo} && uv pip install --python {self._venv_python} -r requirements.txt",
            # The eval phase's cite_p scorer calls nltk.sent_tokenize, which needs the
            # punkt_tab tokenizer data (not bundled with the pip install).
            f"{self._venv_python} -m nltk.downloader punkt_tab",
            f"mkdir -p {self._gen_dir} {self._eval_dir}",
        )

    @property
    def required_secrets(self) -> tuple[str, ...]:
        # OPENAI_API_KEY drives the gpt-4o judge. OPENALEX_API_KEY is required so
        # document_importance cannot silently collapse toward zero on anonymous
        # rate limits. Web-search keys remain optional for non-default corpora.
        return ("OPENAI_API_KEY", "OPENALEX_API_KEY")

    def _build_env_vars(self, secrets: tuple[str, ...] | None = None) -> dict[str, str]:
        env = super()._build_env_vars(secrets)
        # Optional: forward web-search keys when present so a non-default corpus
        # (e.g. -a web_corpuses=TAVILY) works without making the key mandatory.
        # OPENALEX_API_KEY / OPENALEX_EMAIL lift the document_importance citation
        # lookup out of OpenAlex's throttled anonymous tier (see the citation patch).
        for optional in (
            "TAVILY_API_KEY",
            "S2_API_KEY",
            "SERPAPI_API_KEY",
            "OPENALEX_API_KEY",
            "OPENALEX_EMAIL",
        ):
            value = os.environ.get(optional)
            if value:
                env[optional] = value
        return env

    def _create_sandbox_config(self, container_runtime: str, output_dir: str | None = None) -> Any:
        # The bare uv base image has no swe-rex; inject_swerex builds a derived
        # image with swe-rex (plus git/curl) preinstalled, avoiding the failing
        # runtime bootstrap. Same approach as the scicode external eval.
        from dataclasses import replace

        config = super()._create_sandbox_config(container_runtime, output_dir)
        volumes = config.volumes
        if output_dir:
            # Persist every completed query immediately. Post-failure copy-back
            # cannot work when the swe-rex control server is the failed component,
            # and generation only writes summary.json after the whole range exits.
            # Binding the two output trees also makes partial work available in the
            # Beaker result dataset even if the sandbox never answers another RPC.
            dest = Path(output_dir).resolve() / "deepscholar_results"
            generation = dest / "generation"
            evaluation = dest / "evaluation"
            generation.mkdir(parents=True, exist_ok=True)
            evaluation.mkdir(parents=True, exist_ok=True)
            # Rootless Podman maps the container user to a subordinate host UID,
            # so host-owned 0755 directories are readable but not writable inside
            # the sandbox. These paths contain benchmark artifacts only.
            generation.chmod(0o777)
            evaluation.chmod(0o777)
            volumes += (
                (str(generation), self._gen_dir),
                (str(evaluation), self._eval_dir),
            )
        return replace(config, inject_swerex=True, volumes=volumes)

    @property
    def arguments(self) -> dict[str, tuple[str, Any | None]]:
        return {
            "limit": ("Number of queries to run (maps to generation --end-idx)", None),
            "start_idx": ("Starting query index", 0),
            "search_mode": ("Search mode: 'agentic' or 'recursive' (default: config value)", None),
            "search_backend": (
                "Retrieval backend: 'arxiv' (default, keyless), 's2' (Semantic Scholar, "
                "needs S2_API_KEY), or 'tavily' (needs TAVILY_API_KEY)",
                "arxiv",
            ),
            "web_corpuses": (
                "Search corpus: ARXIV (keyless), TAVILY, GOOGLE, GOOGLE_SCHOLAR, BING",
                "ARXIV",
            ),
            "search_steps": ("Recursive search rounds (lower values reduce backend load)", None),
            "search_queries_per_step": (
                "Search queries per round (lower values reduce backend load)",
                None,
            ),
            "temperature": ("Generation temperature for the model under test", None),
            "max_tokens": ("Max tokens for the model under test", 10000),
            "stage_max_tokens": (
                "Token budget for upstream stage LMs (default 4096; raises the "
                "upstream 512 cap that truncates taxonomy JSON, kept below context)",
                None,
            ),
            "local_model_prefix": (
                "litellm prefix for local vLLM ('openai' or 'hosted_vllm')",
                "openai",
            ),
            "lm_timeout": (
                "Per-request timeout in seconds passed to LOTUS/litellm",
                240,
            ),
            "judge_model": ("Judge model for the eval phase", "gpt-4o"),
            "evals": (
                "Comma-separated eval metrics (default: all seven), or 'all'",
                "organization,nugget_coverage,coverage_relevance_rate,document_importance,"
                "reference_coverage,cite_p,claim_coverage",
            ),
            "allow_partial_generation": (
                "Score even if some generation queries failed (default: require all)",
                False,
            ),
            "extra_gen_args": ("Extra args appended to the generation command", None),
            "extra_eval_args": ("Extra args appended to the eval command", None),
        }

    async def execute(
        self,
        provider: InferenceProvider,
        args: dict[str, Any],
        output_dir: str | None = None,
        container_runtime: str = "podman",
    ) -> ExternalEvalResult:
        start_time = time.time()
        ds_args = DeepScholarArgs.from_dict(args)
        all_output: list[str] = []

        base_url = getattr(provider, "base_url", None)
        model_name = provider.model_name
        # Decide locality from the provider's actual base_url, not the localhost
        # fallback below: a provider without base_url is an external API model that
        # must keep upstream behavior, and the fallback would otherwise mislabel it
        # as local and fail the sandbox health check before generation. Pass "" (not
        # the fallback) so a provider managing its own server is still detected via
        # _is_local_provider's _server check even without a base_url.
        is_local = self._is_local_provider(provider, base_url or "")
        provider_url = base_url or "http://localhost:8000/v1"

        # Fail early on a missing search key rather than deep inside generation.
        key_for_backend = {"s2": "S2_API_KEY", "tavily": "TAVILY_API_KEY"}.get(
            ds_args.search_backend
        )
        if key_for_backend and not os.environ.get(key_for_backend):
            return self._error_result(
                f"search_backend={ds_args.search_backend} requires {key_for_backend} "
                f"(map it with --secret-env <beaker_secret>:{key_for_backend})",
                start_time,
            )

        # The search shim only intercepts the recursive path, so a non-arxiv backend
        # forces recursive. (Agentic search uses separate LOTUS tools the shim can't
        # reach, and would still hit arXiv.) Otherwise: local providers default to
        # recursive because upstream's agentic path sends `lm.model` verbatim to a raw
        # OpenAI client, which rejects the litellm "openai/<model>" prefix a vLLM
        # server needs; external API models keep the upstream default (agentic).
        if ds_args.search_backend in ("s2", "tavily") or (is_local and ds_args.search_mode is None):
            ds_args.search_mode = "recursive"

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
                if is_local and not await self._check_provider_health(executor, sandbox_url):
                    return self._error_result(
                        f"Provider not reachable at {sandbox_url}",
                        start_time,
                        "\n".join(all_output),
                    )

                await self._write_config(executor, model_name, sandbox_url, is_local, ds_args)
                await self._write_search_shim(executor)
                await self._write_organization_eval_patch(executor)
                await self._patch_citation_lookup(executor)

                n_success, n_total, generation_ok = await self._run_generation(
                    executor, ds_args, all_output, output_dir
                )
                generation_complete = generation_ok and n_success == n_total
                logger.info(
                    f"[{self.name}] Generation "
                    f"{'complete' if generation_complete else 'incomplete'}: "
                    f"{n_success}/{n_total} queries completed"
                )

                # Strict by default: only score a whole run, or a partial one when
                # explicitly opted in. Generation catches per-query exceptions and still
                # exits 0, and a killed/stalled run leaves partial folders, so both cases
                # would otherwise score a smaller subset than requested. Output folders
                # are bind-mounted, so partial artifacts remain available for inspection.
                may_score = n_success > 0 and (
                    generation_complete or ds_args.allow_partial_generation
                )
                if not may_score:
                    reason = (
                        "Generation phase failed"
                        if not generation_ok
                        else f"Generation incomplete: {n_success}/{n_total} queries succeeded"
                    )
                    return self._error_result(
                        f"{reason}; skipping eval "
                        "(pass -a allow_partial_generation=true to score what completed; "
                        "completed folders are available in deepscholar_results/generation)",
                        start_time,
                        "\n".join(all_output),
                    )

                eval_cmd = self._build_eval_command(ds_args)
                logger.info(f"[{self.name}] Eval: {eval_cmd}")
                eval_result = await executor.execute_command(
                    eval_cmd,
                    timeout=self.timeout_seconds,
                    stream=True,
                    log_prefix=f"{self.name}-eval",
                )
                all_output.append(f"$ {eval_cmd}\n{eval_result.output}")
                logger.info(f"[{self.name}] Eval exit code: {eval_result.exit_code}")

                result = await self._extract_results(
                    executor,
                    "\n".join(all_output),
                    eval_result.exit_code,
                    n_success=n_success,
                    n_total=n_total,
                    generation_ok=generation_ok,
                    requested_metrics=tuple(ds_args.evals),
                )

        except Exception as e:
            logger.exception(f"[{self.name}] Execution failed")
            return self._error_result(str(e), start_time, "\n".join(all_output))

        result.duration_seconds = time.time() - start_time
        if output_dir:
            self._save_results(result, output_dir)
        return result

    def _build_lm_config(
        self, model_name: str, sandbox_url: str, is_local: bool, ds_args: DeepScholarArgs
    ) -> dict[str, Any]:
        """Build the LOTUS `lm` block pointing at the model under test."""
        lm: dict[str, Any] = {
            "temperature": ds_args.temperature if ds_args.temperature is not None else 1.0,
            "max_tokens": ds_args.max_tokens,
            "timeout": ds_args.lm_timeout,
        }
        if is_local:
            # litellm routes "<prefix>/<model>" to the OpenAI-compatible vLLM server at api_base.
            lm["model"] = f"{ds_args.local_model_prefix}/{model_name}"
            lm["api_base"] = sandbox_url
            lm["api_key"] = "EMPTY"
        else:
            # Upstream's agentic search sends lm.model verbatim to a raw OpenAI client,
            # so use a name both that client and litellm accept.
            lm["model"] = openai_endpoint_model_name(model_name)
        return lm

    async def _write_config(
        self,
        executor: SandboxExecutor,
        model_name: str,
        sandbox_url: str,
        is_local: bool,
        ds_args: DeepScholarArgs,
    ) -> None:
        """Write the LOTUS config into the sandbox (JSON is valid YAML)."""
        config = dict(_BASE_CONFIG)
        config["lm"] = self._build_lm_config(model_name, sandbox_url, is_local, ds_args)
        # The pinned upstream Configs removes max_completion_tokens before cloning
        # `lm` into its stage LMs. The shim supplies the smaller stage budget to
        # those clones, so configure generation_lm explicitly to preserve the
        # user-facing max_tokens setting for the final related-work generation.
        config["generation_lm"] = dict(config["lm"])
        config["web_corpuses"] = ds_args.web_corpuses
        # Backend selection (see sandbox_search_shim). For "s2" the shim routes the
        # hardwired arXiv search to Semantic Scholar, so disable the extra web pass
        # to avoid duplicate S2 calls. For "tavily" the shim skips arXiv and uses the
        # TAVILY web corpus, so that pass must stay on.
        if ds_args.search_backend == "s2":
            config["enable_web_search"] = False
        elif ds_args.search_backend == "tavily":
            config["enable_web_search"] = True
            config["web_corpuses"] = ["TAVILY"]
        if ds_args.search_mode:
            config["search_mode"] = ds_args.search_mode
        if ds_args.search_steps is not None:
            config["num_search_steps"] = ds_args.search_steps
        if ds_args.search_queries_per_step is not None:
            config["num_search_queries_per_step_per_corpus"] = ds_args.search_queries_per_step
        content = json.dumps(config, indent=2)
        encoded = base64.b64encode(content.encode()).decode()
        await executor.execute_command(
            f"echo '{encoded}' | base64 -d > {shlex.quote(self._config_path)}", timeout=30.0
        )
        logger.info(f"[{self.name}] Wrote LOTUS config (model={config['lm']['model']})")

    async def _write_search_shim(self, executor: SandboxExecutor) -> None:
        """Copy the search-backend shim into the sandbox (see sandbox_search_shim)."""
        source = (Path(__file__).parent / "sandbox_search_shim.py").read_text()
        encoded = base64.b64encode(source.encode()).decode()
        await executor.execute_command(
            f"echo '{encoded}' | base64 -d > {shlex.quote(self._shim_path)}", timeout=30.0
        )
        logger.info(f"[{self.name}] Wrote search shim to {self._shim_path}")

    async def _write_organization_eval_patch(self, executor: SandboxExecutor) -> None:
        """Install the organization scorer compatible with the pinned LOTUS API."""
        source = (Path(__file__).parent / "organization_eval_patch.py").read_text()
        encoded = base64.b64encode(source.encode()).decode()
        await executor.execute_command(
            f"echo '{encoded}' | base64 -d > {shlex.quote(self._organization_evaluator_path)}",
            timeout=30.0,
        )
        logger.info(
            f"[{self.name}] Patched organization evaluator at {self._organization_evaluator_path}"
        )

    async def _patch_citation_lookup(self, executor: SandboxExecutor) -> None:
        """Route the upstream OpenAlex citation lookup through env-provided creds."""
        source = (Path(__file__).parent / "citation_lookup_patch.py").read_text()
        encoded = base64.b64encode(source.encode()).decode()
        await executor.execute_command(
            f"echo '{encoded}' | base64 -d | python3 - {shlex.quote(self._citation_utils_path)}",
            timeout=30.0,
        )
        logger.info(f"[{self.name}] Patched citation lookup at {self._citation_utils_path}")

    def _build_generation_command(self, ds_args: DeepScholarArgs) -> str:
        # Always run through the shim: it applies the stage-LM token-budget fix for
        # every backend, and (for s2/tavily) also reroutes retrieval. It then hands
        # off to deepscholar_base.main unchanged.
        stage_budget = ds_args.stage_max_tokens or DEFAULT_STAGE_MAX_TOKENS
        entry = [
            f"DEEPSCHOLAR_SEARCH_BACKEND={shlex.quote(ds_args.search_backend)}",
            f"DEEPSCHOLAR_STAGE_MAX_TOKENS={stage_budget}",
            self._venv_python,
            self._shim_path,
        ]
        parts = [
            f"cd {self._repo} &&",
            *entry,
            "--output-folder",
            shlex.quote(self._gen_dir),
            "--config-yaml",
            shlex.quote(self._config_path),
            "--start-idx",
            str(ds_args.start_idx),
        ]
        if ds_args.limit is not None:
            parts.extend(["--end-idx", str(ds_args.start_idx + ds_args.limit)])
        # search_mode is carried in the generated config (see _write_config).
        parts.extend(ds_args.extra_gen_args)
        return " ".join(parts)

    async def _run_generation(
        self,
        executor: SandboxExecutor,
        ds_args: DeepScholarArgs,
        all_output: list[str],
        output_dir: str | None,
    ) -> tuple[int, int, bool]:
        """Run generation once and return (generated, requested, command_succeeded)."""
        n_total = await self._requested_query_count(executor, ds_args)
        gen_cmd = self._build_generation_command(ds_args)
        logger.info(f"[{self.name}] Generation: {gen_cmd}")
        gen_result = await executor.execute_command(
            gen_cmd,
            timeout=self.timeout_seconds,
            stream=True,
            log_prefix=f"{self.name}-gen",
        )
        all_output.append(f"$ {gen_cmd}\n{gen_result.output}")
        logger.info(f"[{self.name}] Generation exit code: {gen_result.exit_code}")

        summary_success, summary_total = await self._generation_counts(executor, output_dir)
        completed = await self._completed_query_ids(
            executor,
            output_dir,
            ds_args.start_idx,
            ds_args.start_idx + n_total if n_total else None,
        )
        n_success = len(completed) if completed is not None else summary_success
        if not n_total:
            n_total = summary_total or n_success
        return n_success, n_total, gen_result.success

    async def _requested_query_count(
        self, executor: SandboxExecutor, ds_args: DeepScholarArgs
    ) -> int:
        """Number of dataset rows requested by the generation bounds."""
        dataset_total = await self._total_query_count(executor)
        available = max(dataset_total - ds_args.start_idx, 0) if dataset_total is not None else None
        if ds_args.limit is not None:
            return min(ds_args.limit, available) if available is not None else ds_args.limit
        return available or 0

    async def _total_query_count(self, executor: SandboxExecutor) -> int | None:
        """Count dataset queries (queries.csv if present, else papers CSV).

        Uses pandas rather than `wc -l` because abstract/related-work fields contain
        newlines inside quoted CSV cells.
        """
        script = (
            "import os, pandas as pd; "
            "f = 'dataset/queries.csv' if os.path.exists('dataset/queries.csv') "
            "else 'dataset/papers_with_related_works.csv'; "
            "print(len(pd.read_csv(f)))"
        )
        result = await executor.execute_command(
            f"cd {self._repo} && {self._venv_python} -c {shlex.quote(script)}", timeout=120.0
        )
        if not result.success:
            return None
        for line in reversed(result.output.strip().splitlines()):
            token = line.strip()
            if token.isdigit():
                return int(token)
        return None

    async def _completed_query_ids(
        self,
        executor: SandboxExecutor,
        output_dir: str | None,
        start: int,
        end: int | None,
    ) -> set[int] | None:
        """Query indices with all artifacts required by the upstream parser."""
        required = ("final_report.md", "intro.md", "paper.csv")
        names: list[str]
        if output_dir:
            root = Path(output_dir) / "deepscholar_results" / "generation"
            if root.is_dir():
                names = [
                    child.name
                    for child in root.iterdir()
                    if child.is_dir() and all((child / filename).is_file() for filename in required)
                ]
                indices = {int(name) for name in names if name.isdigit()}
                return {
                    index for index in indices if index >= start and (end is None or index < end)
                }

        script = (
            "from pathlib import Path; "
            f"root=Path({self._gen_dir!r}); "
            f"required={required!r}; "
            "print('\\n'.join(p.name for p in root.iterdir() "
            "if p.is_dir() and all((p / f).is_file() for f in required)))"
        )
        try:
            result = await executor.execute_command(
                f"{self._venv_python} -c {shlex.quote(script)}", timeout=60.0
            )
        except Exception:
            return None
        if not result.success:
            return None
        names = result.output.splitlines()

        indices = {int(name) for name in names if name.isdigit()}
        return {index for index in indices if index >= start and (end is None or index < end)}

    def _build_eval_command(self, ds_args: DeepScholarArgs) -> str:
        parts = [
            f"cd {self._repo} &&",
            self._venv_python,
            "-m",
            "eval.main",
            "--modes",
            "deepscholar_base",
            "--evals",
            *ds_args.evals,
            "--input-folder",
            shlex.quote(self._gen_dir),
            "--output-folder",
            shlex.quote(self._eval_dir),
            "--model-name",
            shlex.quote(litellm_model_name(ds_args.judge_model)),
        ]
        parts.extend(ds_args.extra_eval_args)
        return " ".join(parts)

    async def _generation_counts(
        self, executor: SandboxExecutor, output_dir: str | None
    ) -> tuple[int, int]:
        """Return (successful_queries, total_queries) from generation summary.json."""
        text = ""
        if output_dir:
            summary_path = Path(output_dir) / "deepscholar_results" / "generation" / "summary.json"
            if summary_path.is_file():
                text = summary_path.read_text()
        if not text:
            try:
                cat = await executor.execute_command(
                    f"cat {shlex.quote(self._gen_dir)}/summary.json", timeout=60.0
                )
            except Exception:
                cat = None
            if cat is not None and cat.success:
                text = cat.output
        if not text.strip():
            return (0, 0)
        try:
            summary = json.loads(text)
        except json.JSONDecodeError:
            return (0, 0)
        if not isinstance(summary, list):
            return (0, 0)
        n_success = sum(1 for r in summary if isinstance(r, dict) and r.get("status") == "success")
        return (n_success, len(summary))

    async def _read_dir(
        self, executor: SandboxExecutor, remote_dir: str, pattern: str
    ) -> dict[str, str]:
        """Return {relative_path: file_text} for files matching pattern under remote_dir."""
        find = await executor.execute_command(
            f"find {shlex.quote(remote_dir)} -type f -name {shlex.quote(pattern)} 2>/dev/null",
            timeout=60.0,
        )
        files: dict[str, str] = {}
        if not find.success or not find.output.strip():
            return files
        for remote_path in (p.strip() for p in find.output.strip().split("\n") if p.strip()):
            cat = await executor.execute_command(f"cat {shlex.quote(remote_path)}", timeout=60.0)
            if cat.success:
                rel = remote_path.replace(remote_dir.rstrip("/") + "/", "")
                files[rel] = cat.output
        return files

    async def _extract_results(
        self,
        executor: SandboxExecutor,
        raw_output: str,
        exit_code: int,
        n_success: int = 0,
        n_total: int = 0,
        generation_ok: bool = True,
        requested_metrics: tuple[str, ...] = (),
    ) -> ExternalEvalResult:
        # Canonical layout: evaluation/<metric>/aggregated_results.csv holds the
        # aggregate for each metric on a single `deepscholar_base` row. The metric
        # name is the parent directory.
        agg_files = await self._read_dir(executor, self._eval_dir, "aggregated_results.csv")
        all_metrics: dict[str, float] = {}
        for rel, text in sorted(agg_files.items()):
            metric = rel.split("/")[0]
            value = parse_aggregate_csv(text, metric)
            if value is not None:
                all_metrics[metric] = value
        parsed_metric_names = tuple(all_metrics)

        # Normalize upstream 0-2 metrics to the paper's 0-1 scale before aggregating.
        for metric in _HALF_SCALE_METRICS:
            if metric in all_metrics:
                all_metrics[metric] /= 2

        # Headline geomean over the primary metrics (None if any is missing).
        geomean = compute_geomean(all_metrics)
        if geomean is not None:
            # Keep the canonical geomean first when no fixed denominator is available.
            all_metrics = {"geomean": geomean, **all_metrics}

        # Upstream aggregates only rows its parser successfully scored. Preserve
        # those canonical values above, and also expose a fixed-request denominator
        # where missing generation/parser rows contribute zero.
        query_files = await self._read_dir(executor, self._eval_dir, "deepscholar_base.csv")
        per_metric: dict[str, dict[str, float]] = {}
        for rel, text in sorted(query_files.items()):
            metric = rel.split("/")[0]
            per_metric[metric] = parse_per_query_csv(text, metric)

        # Same 0-2 -> 0-1 normalization for the per-query values feeding the fixed
        # denominator, so coverage_relevance_rate_fixed and geomean_fixed also match.
        for metric in _HALF_SCALE_METRICS:
            if metric in per_metric:
                per_metric[metric] = {k: v / 2 for k, v in per_metric[metric].items()}

        fixed_metrics: dict[str, float] = {}
        if n_total > 0:
            for metric in parsed_metric_names:
                scores = per_metric.setdefault(metric, {})
                fixed_metrics[metric] = sum(scores.values()) / n_total
                all_metrics[f"{metric}_fixed"] = fixed_metrics[metric]
            fixed_geomean = compute_geomean(fixed_metrics)
            if fixed_geomean is not None:
                # Missing generation/parser rows score zero in this value, so use
                # it as the primary metric for comparisons across complete runs.
                all_metrics = {"geomean_fixed": fixed_geomean, **all_metrics}

        score_sets = [
            set(per_metric[metric]) for metric in requested_metrics if metric in per_metric
        ]
        n_scored = (
            len(set.intersection(*score_sets))
            if score_sets and len(score_sets) == len(requested_metrics)
            else 0
        )

        if not all_metrics:
            return ExternalEvalResult(
                name=self.name,
                success=False,
                error="No metrics parsed from eval outputs (check raw_output / copied files)",
                raw_output=raw_output,
            )

        missing_metrics = sorted(set(requested_metrics) - set(parsed_metric_names))
        success = exit_code == 0 and bool(all_metrics) and not missing_metrics
        if success:
            error = None
        elif missing_metrics:
            error = (
                "Missing requested eval metrics: "
                f"{', '.join(missing_metrics)} (eval phase exited {exit_code})"
            )
        else:
            error = f"Eval phase exited {exit_code} (metrics may be partial)"
        return ExternalEvalResult(
            name=self.name,
            success=success,
            error=error,
            metrics=all_metrics,
            metadata={
                "eval_dir": self._eval_dir,
                "ref": DEEPSCHOLAR_REF,
                "queries_requested": n_total,
                "queries_generated": n_success,
                "queries_scored": n_scored,
                "num_tasks": n_total,
                "fixed_metric_denominator": n_total,
                "metric_query_counts": {
                    metric: len(scores) for metric, scores in sorted(per_metric.items())
                },
                # Backward-compatible alias; now counts scorable artifacts rather
                # than any folder containing final_report.md.
                "queries_succeeded": n_success,
                "queries_total": n_total,
                "generation_complete": generation_ok and n_success == n_total,
            },
            raw_output=raw_output,
        )
