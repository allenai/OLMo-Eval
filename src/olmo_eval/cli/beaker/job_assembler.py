"""Job configuration assembly for Beaker launch."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

from olmo_eval.common.constants.infrastructure import (
    BEAKER_RESULT_DIR,
    cluster_has_weka,
)
from olmo_eval.launch.beaker.constants import BEAKER_INFRA_ENV_VARS
from olmo_eval.launch.beaker.mirror import log

if TYPE_CHECKING:
    from olmo_eval.cli.beaker.config_loader import LaunchConfig
    from olmo_eval.cli.beaker.experiment_plan import ExperimentPlan
    from olmo_eval.harness import HarnessConfig
    from olmo_eval.launch import BeakerJobConfig


OLMO_CORE_PACKAGE_WITH_EXTRAS = "ai2-olmo-core[torchao,transformers]"
_VERSION_SELECTOR_RE = re.compile(r"^(?:\d|==|!=|~=|>=|<=|>|<)")


def get_provider_kind(model_spec: str, default_kind: str | None = None) -> str | None:
    """Get the provider kind for a model.

    Args:
        model_spec: Model name or path.
        default_kind: Default provider kind if model is not a preset or has no kind.

    Returns:
        Provider kind string (e.g., "vllm", "vllm_server", "litellm") or None.
    """
    from olmo_eval.common.configs import get_provider_config

    try:
        provider_config = get_provider_config(model_spec)
        # Fall back to default_kind if config has no kind set
        return provider_config.kind or default_kind
    except Exception:
        return default_kind


def resolve_provider_kind(
    model_spec: str,
    harness_config: HarnessConfig | None,
    harness_overrides: list[str] | None = None,
) -> str | None:
    """Resolve the provider kind a job will run with, the same way ``olmo-eval run`` does.

    Install-time choices such as the isolated vLLM venv must follow this resolution, or
    the job installs for one provider and runs another.

    Args:
        model_spec: Model name or path.
        harness_config: Harness preset with CLI overrides applied, or None.
        harness_overrides: CLI override strings forwarded to the run command.

    Returns:
        Provider kind string (e.g., "vllm", "vllm_server", "litellm") or None.
    """
    from olmo_eval.cli.run.config import merge_model_provider, provider_kind_override
    from olmo_eval.common.configs import get_provider_config
    from olmo_eval.harness import HarnessConfig

    try:
        provider_config = get_provider_config(model_spec)
    except Exception:
        fallback = harness_config.provider.kind if harness_config else None
        kind = provider_kind_override(harness_overrides or []) or fallback
        return str(kind) if kind else None

    base = harness_config or HarnessConfig(name="default")
    merged = merge_model_provider(base, provider_config, harness_overrides)
    return str(merged.provider.kind) if merged.provider.kind else None


def get_provider_extras(model_spec: str, default_kind: str | None = None) -> list[str]:
    """Get the pip extras required for a model's provider.

    Args:
        model_spec: Model name or path.
        default_kind: Effective provider kind to use when the caller has already resolved
            harness/provider overrides. When omitted, the model spec is inspected.

    Returns:
        List of pip extras needed for the provider.
    """
    from olmo_eval.common.constants.infrastructure import BACKEND_OPTIONAL_GROUPS

    provider_kind = default_kind or get_provider_kind(model_spec)

    extras: list[str] = []
    if provider_kind:
        provider_extra = BACKEND_OPTIONAL_GROUPS.get(provider_kind)
        if provider_extra:
            extras.append(provider_extra)
        # vllm_server uses OpenAI client to communicate with vLLM's OpenAI-compatible API
        if provider_kind == "vllm_server":
            extras.append("clients")
    return extras


def get_provider_dependencies(model_spec: str) -> list[str]:
    """Get runtime dependencies from a model's provider config.

    Args:
        model_spec: Model name or path.

    Returns:
        List of package specifiers to install.
    """
    from olmo_eval.common.configs import get_provider_config

    try:
        provider_config = get_provider_config(model_spec)
        return list(provider_config.dependencies)
    except Exception:
        return []


def _is_olmo_core_version_selector(package: str) -> bool:
    """Whether a provider.package value is shorthand for an ai2-olmo-core version."""
    return bool(_VERSION_SELECTOR_RE.match(package))


def _format_install_spec(package: str, flags: list[str]) -> str:
    return " ".join([package, *flags]) if flags else package


def _normalize_olmo_core_package(package: str) -> str:
    """Normalize OLMo-core provider package overrides.

    The GitHub repository is named ``OLMo-core``, but the Python distribution is
    ``ai2-olmo-core``. A bare source URL therefore needs a PEP 508 name binding
    so uv replaces the already-installed distribution instead of targeting the
    repository basename.
    """
    from olmo_eval.launch.beaker.launcher import (
        is_direct_source_package,
        normalize_provider_package,
        parse_install_spec,
    )

    pkg_spec, flags = parse_install_spec(package.strip())
    if not pkg_spec:
        return package

    if _is_olmo_core_version_selector(pkg_spec):
        version_spec = pkg_spec if not pkg_spec[0].isdigit() else f"=={pkg_spec}"
        return _format_install_spec(f"{OLMO_CORE_PACKAGE_WITH_EXTRAS}{version_spec}", flags)

    if " @ " not in pkg_spec and is_direct_source_package(pkg_spec):
        normalized_source = normalize_provider_package(pkg_spec)
        return _format_install_spec(
            f"{OLMO_CORE_PACKAGE_WITH_EXTRAS} @ {normalized_source}",
            flags,
        )

    return package


def normalize_provider_package_for_kind(provider_kind: str | None, package: str) -> str:
    """Normalize a provider package override for the effective provider kind."""
    if provider_kind in {"olmo_core", "olmo_core_vlm"}:
        return _normalize_olmo_core_package(package)
    return package


def collect_install_extras(
    *,
    store: bool = False,
    sandbox: bool = False,
    metrics: bool = False,
    collect_gpu: bool = False,
    scaffold_name: str | None = None,
    provider_extras: list[str] | None = None,
) -> list[str]:
    """Collect pip extras needed for a job.

    Args:
        store: Whether storage is enabled.
        sandbox: Whether sandbox is enabled.
        metrics: Whether metrics collection is enabled.
        collect_gpu: Whether GPU metrics collection is enabled.
        scaffold_name: Harness scaffold name (e.g., "openai_agents").
        provider_extras: Provider-specific extras.

    Returns:
        List of pip extras to install.
    """
    extras: list[str] = ["s3"]

    if store:
        extras.append("storage")
    if sandbox:
        extras.append("sandbox")
    if metrics:
        extras.append("postgres")
    if collect_gpu:
        extras.append("gpu")

    if scaffold_name:
        from olmo_eval.harness import get_scaffold_extras

        for extra in get_scaffold_extras(scaffold_name):
            if extra not in extras:
                extras.append(extra)

    for extra in provider_extras or []:
        if extra not in extras:
            extras.append(extra)

    return extras


def assemble_external_eval_job(
    name: str,
    model: str,
    external_evals: list[str],
    cluster: str,
    num_gpus: int,
    workspace: str,
    beaker_image: str,
    priority: str = "normal",
    timeout: str = "24h",
    budget: str | None = None,
    groups: list[str] | None = None,
    tensor_parallel_size: int = 1,
    s3_bucket: str | None = None,
    s3_prefix: str | None = None,
    s3_region: str = "us-east-1",
    store: bool = False,
    env_secrets: list[tuple[str, str]] | None = None,
    inject_aws_credentials: bool = False,
    inject_gcs_credentials: bool = False,
    eval_args: dict[str, str] | None = None,
    provider_kwargs: dict[str, str] | None = None,
    uv_cache_dir: str | None = None,
    beaker_username: str | None = None,
    preemptible: bool | None = True,
    min_runtime: str | None = None,
    retries: int | None = None,
    provider_kind: str | None = None,
    base_url: str | None = None,
    user_env_vars: dict[str, str] | None = None,
    force_download_model: bool = False,
) -> Any:
    """Assemble a BeakerJobConfig for running external evaluations.

    Args:
        name: Experiment name.
        model: Model name or path.
        external_evals: List of external evaluation names.
        cluster: Beaker cluster name.
        num_gpus: Number of GPUs.
        workspace: Beaker workspace.
        beaker_image: Container image to use.
        priority: Job priority.
        timeout: Job timeout.
        budget: Beaker budget.
        groups: Beaker groups.
        tensor_parallel_size: Tensor parallel size for vLLM.
        s3_bucket: S3 bucket for results.
        s3_prefix: S3 prefix for results.
        s3_region: S3 region.
        env_secrets: List of (env_var, secret_name) tuples.
        inject_aws_credentials: Whether to inject AWS credentials.
        inject_gcs_credentials: Whether to inject GCS credentials.
        eval_args: Arguments to pass to external evaluations.

    Returns:
        Configured BeakerJobConfig.
    """
    from olmo_eval.launch import BeakerEnvSecret, BeakerJobConfig

    # Build command
    command: list[str] = ["olmo-eval", "run-external"]
    command.extend(["-m", model])
    for eval_name in external_evals:
        command.extend(["-e", eval_name])
    command.extend(["-O", BEAKER_RESULT_DIR])

    # Pass provider kind and base_url if specified
    if provider_kind:
        command.extend(["--provider", provider_kind])
    if base_url:
        command.extend(["--base-url", base_url])

    if tensor_parallel_size > 1:
        command.extend(["--tp", str(tensor_parallel_size)])
    if force_download_model:
        command.append("--force-download-model")

    # Add eval_args
    if eval_args:
        for key, value in eval_args.items():
            command.extend(["-a", f"{key}={json.dumps(value)}"])

    # Add provider_kwargs
    if provider_kwargs:
        for key, value in provider_kwargs.items():
            command.extend(["-K", f"{key}={json.dumps(value)}"])

    # Add storage options (only when --store is enabled)
    if store:
        command.append("--store")
        if s3_bucket and s3_prefix:
            command.extend(["--s3-bucket", s3_bucket])
            command.extend(["--s3-prefix", s3_prefix])
            if groups:
                command.extend(["--s3-group", groups[0]])
            if s3_region != "us-east-1":
                command.extend(["--s3-region", s3_region])

    # Add experiment metadata
    if groups:
        command.extend(["--experiment-group", groups[0]])
    command.extend(["--experiment-name", name])

    # Environment variables
    env_vars: dict[str, str] = {
        "BEAKER_ALLOW_SUBCONTAINERS": "1",
        "BEAKER_SKIP_DOCKER_SOCKET": "1",
        "BEAKER_WORKSPACE": workspace,
    }
    # Add infrastructure config for olmo-eval
    env_vars.update(BEAKER_INFRA_ENV_VARS)

    if beaker_username:
        env_vars["BEAKER_AUTHOR"] = beaker_username

    if cluster_has_weka(cluster):
        env_vars.update(
            {
                "HF_HOME": "/weka/oe-eval-default/oyvindt/hf-cache",
                "HF_HUB_CACHE": "/weka/oe-eval-default/oyvindt/hf-cache",
                "INSPECT_CACHE_DIR": "/weka/oe-training-default/olmo-eval/inspect-cache",
                "UV_LINK_MODE": "copy",
            }
        )
        if uv_cache_dir:
            env_vars["UV_CACHE_DIR"] = uv_cache_dir

    # Get registry mirror URL
    try:
        from olmo_eval.launch.beaker.mirror import get_registry_mirror_url

        mirror_url = get_registry_mirror_url()
        env_vars["MIRROR_HOSTS"] = mirror_url
        setup_registry_mirror = True
    except Exception:
        setup_registry_mirror = False

    # Build env secrets
    beaker_env_secrets = []
    if env_secrets:
        beaker_env_secrets = [
            BeakerEnvSecret(env_var, secret_name) for env_var, secret_name in env_secrets
        ]

    # Add store defaults if enabled
    if store:
        from olmo_eval.launch.beaker.secrets import get_store_env_defaults

        env_vars.update(get_store_env_defaults())

    # User-supplied env vars win over everything above
    if user_env_vars:
        env_vars.update(user_env_vars)

    # Collect scaffold names from external evals
    # Check eval_args for scaffold override, otherwise use eval's default
    from olmo_eval.evals.external.registry import get_external_eval

    scaffold_names: list[str] = []

    # Check if scaffold is specified in eval_args (overrides eval default)
    args_scaffold = eval_args.get("scaffold") if eval_args else None
    if args_scaffold:
        scaffold_names.append(args_scaffold)
    else:
        # Fall back to each eval's default scaffold
        for eval_name in external_evals:
            eval_instance = get_external_eval(eval_name)
            if eval_instance.scaffold and eval_instance.scaffold not in scaffold_names:
                scaffold_names.append(eval_instance.scaffold)

    # External evals always run vLLM as a server subprocess, so use isolated venv
    # to avoid dependency conflicts with other packages (e.g., openhands)
    vllm_isolated_venv = True

    # Collect extras from all scaffolds
    extras: list[str] = collect_install_extras(
        store=store,
        sandbox=True,
        provider_extras=get_provider_extras(model, default_kind="vllm_server"),
    )
    for scaffold_name in scaffold_names:
        for extra in collect_install_extras(scaffold_name=scaffold_name):
            if extra not in extras:
                extras.append(extra)

    provider_packages = get_provider_dependencies(model) or None

    return BeakerJobConfig(
        name=name,
        command=command,
        cluster=cluster,
        num_gpus=num_gpus,
        priority=priority,
        preemptible=preemptible,
        min_runtime=min_runtime,
        timeout=timeout,
        shared_memory="10GiB",
        retries=retries,
        workspace=workspace,
        budget=budget,
        groups=groups or [],
        beaker_image=beaker_image,
        inject_aws_credentials=inject_aws_credentials,
        inject_gcs_credentials=inject_gcs_credentials,
        env_vars=env_vars,
        env_secrets=beaker_env_secrets,
        enable_sandbox=True,
        setup_registry_mirror=setup_registry_mirror,
        setup_store_secrets=store,
        extras=extras,
        provider_packages=provider_packages,
        vllm_isolated_venv=vllm_isolated_venv,
    )


class JobConfigAssembler:
    """Assembles BeakerJobConfig from experiment plan."""

    def __init__(
        self,
        config: LaunchConfig,
        effective_image: str,
        effective_groups: list[str],
        beaker_username: str,
        common_secrets: list[tuple[str, str]],
        store_secrets: list[tuple[str, str]],
        task_secrets: list[tuple[str, str]],
        inject_aws_credentials: bool,
        inject_gcs_credentials: bool,
        enable_sandbox: bool = False,
        secret_env_overrides: dict[str, str] | None = None,
    ):
        self.config = config
        self.effective_image = effective_image
        self.effective_groups = effective_groups
        self.beaker_username = beaker_username
        self.common_secrets = common_secrets
        self.store_secrets = store_secrets
        self.task_secrets = task_secrets
        self.inject_aws_credentials = inject_aws_credentials
        self.inject_gcs_credentials = inject_gcs_credentials
        self.enable_sandbox = enable_sandbox
        self.secret_env_overrides = secret_env_overrides or {}

    def assemble(self, exp: ExperimentPlan) -> BeakerJobConfig:
        """Assemble a BeakerJobConfig for an experiment."""
        from olmo_eval.launch import BeakerEnvSecret, BeakerJobConfig

        command = self._build_command(exp)

        # Determine scaffold and sandbox requirements from harness preset
        scaffold_name: str | None = None
        sandbox_enabled = False
        metrics_enabled = False
        collect_gpu_enabled = False
        harness_provider_package: str | None = None
        harness_provider_deps: list[str] = []
        if self.config.harness:
            from olmo_eval.harness import get_harness_preset

            preset = get_harness_preset(self.config.harness)
            if self.config.harness_overrides:
                from olmo_eval.cli.run.config import _apply_harness_overrides

                preset = _apply_harness_overrides(preset, self.config.harness_overrides)
            scaffold_name = preset.scaffold
            sandbox_enabled = bool(preset.sandboxes)
            from olmo_eval.inference.metrics import ReporterType

            metrics_enabled = (
                preset.metrics is not None
                and preset.metrics.enabled
                and preset.metrics.has_reporter(ReporterType.DB)
            )
            collect_gpu_enabled = (
                preset.metrics is not None and preset.metrics.enabled and preset.metrics.collect_gpu
            )
            harness_provider_package = preset.provider.package
            harness_provider_deps = list(preset.provider.dependencies)
        else:
            preset = None

        # Install for the provider the job will actually run with
        provider_kind = resolve_provider_kind(exp.model_spec, preset, self.config.harness_overrides)
        vllm_isolated_venv = provider_kind == "vllm_server"
        model_provider_extras = get_provider_extras(exp.model_spec, default_kind=provider_kind)

        # If provider.package is set, it replaces the provider's bundled extra.
        # Keep companion extras like clients for vllm_server.
        provider_extra_override = (
            {
                "olmo_core": "olmo_core",
                "olmo_core_vlm": "olmo_core_vlm",
                "vllm": "vllm",
                "vllm_server": "vllm",
            }.get(provider_kind)
            if provider_kind is not None
            else None
        )
        if harness_provider_package and provider_extra_override:
            provider_extras = [
                extra for extra in model_provider_extras if extra != provider_extra_override
            ]
        else:
            provider_extras = model_provider_extras

        install_extras = collect_install_extras(
            store=self.config.store,
            sandbox=sandbox_enabled,
            metrics=metrics_enabled,
            collect_gpu=collect_gpu_enabled,
            scaffold_name=scaffold_name,
            provider_extras=provider_extras,
        )

        # Collect env vars that have explicit overrides
        overridden_env_vars = set(self.secret_env_overrides.values())

        # Add default secrets, skipping any that are overridden
        env_secrets = [
            BeakerEnvSecret(env_var, secret_name)
            for env_var, secret_name in self.common_secrets
            if env_var not in overridden_env_vars
        ]
        env_secrets.extend(
            BeakerEnvSecret(env_var, secret_name)
            for env_var, secret_name in self.store_secrets
            if env_var not in overridden_env_vars
        )
        env_secrets.extend(
            BeakerEnvSecret(env_var, secret_name)
            for env_var, secret_name in self.task_secrets
            if env_var not in overridden_env_vars
        )
        # Add explicit secret overrides (beaker_secret -> env_var)
        env_secrets.extend(
            BeakerEnvSecret(env_var, beaker_secret)
            for beaker_secret, env_var in self.secret_env_overrides.items()
        )

        job_env_vars: dict[str, str] = {
            "BEAKER_AUTHOR": self.beaker_username,
            "BEAKER_WORKSPACE": self.config.workspace,
        }
        # Add infrastructure config for olmo-eval
        job_env_vars.update(BEAKER_INFRA_ENV_VARS)

        if cluster_has_weka(self.config.cluster):
            job_env_vars.update(
                {
                    "HF_HOME": "/weka/oe-eval-default/oyvindt/hf-cache",
                    "HF_HUB_CACHE": "/weka/oe-eval-default/oyvindt/hf-cache",
                    "INSPECT_CACHE_DIR": "/weka/oe-training-default/olmo-eval/inspect-cache",
                    "UV_LINK_MODE": "copy",
                }
            )
            if self.config.uv_cache_dir:
                job_env_vars["UV_CACHE_DIR"] = self.config.uv_cache_dir

        # Add store defaults if enabled
        if self.config.store:
            from olmo_eval.launch.beaker.secrets import get_store_env_defaults

            job_env_vars.update(get_store_env_defaults())

        # Configure sandbox environment and registry mirror
        setup_registry_mirror = False
        setup_modal_gcp_secret = False
        log.info(f"Sandbox enabled: {self.enable_sandbox}")
        if self.enable_sandbox:
            job_env_vars["BEAKER_ALLOW_SUBCONTAINERS"] = "1"
            job_env_vars["BEAKER_SKIP_DOCKER_SOCKET"] = "1"

            # Get registry mirror URL for faster image pulls (raises if unavailable)
            from olmo_eval.launch.beaker.mirror import get_registry_mirror_url

            mirror_url = get_registry_mirror_url()
            job_env_vars["MIRROR_HOSTS"] = mirror_url
            setup_registry_mirror = True

        # Check for Modal sandbox with GCP registry auth - auto-setup Modal secret
        if self.config.harness and preset and preset.sandboxes:
            from olmo_eval.harness.sandbox.config import SandboxMode

            for sandbox in preset.sandboxes:
                if (
                    sandbox.mode == SandboxMode.MODAL
                    and sandbox.registry_auth
                    and sandbox.registry_auth.provider == "gcp"
                ):
                    # Use configured secret name or default to gcp-service-account-json
                    secret_name = sandbox.registry_auth.secret_name or "gcp-service-account-json"
                    job_env_vars["MODAL_GCP_SECRET_NAME"] = secret_name
                    setup_modal_gcp_secret = True
                    log.info(f"Modal GCP secret setup enabled: {secret_name}")

        # User-supplied env vars win over everything above
        job_env_vars.update(self.config.env_vars)

        # Collect task dependencies and provider dependencies separately
        task_packages = self._extract_task_dependencies(exp.tasks, exp.task_overrides) or None

        # Build provider packages list:
        # 1. provider.package (overrides default extra like vllm)
        # 2. model's provider dependencies
        # 3. provider.dependencies (additional deps)
        provider_packages: list[str] = []
        if harness_provider_package:
            provider_packages.append(
                normalize_provider_package_for_kind(provider_kind, harness_provider_package)
            )
        provider_packages.extend(get_provider_dependencies(exp.model_spec))
        provider_packages.extend(harness_provider_deps)
        provider_packages = provider_packages or None  # type: ignore[ty:invalid-assignment]

        return BeakerJobConfig(
            name=exp.name,
            command=command,
            cluster=self.config.cluster,
            num_gpus=exp.num_gpus,
            priority=exp.priority,
            preemptible=self.config.preemptible,
            min_runtime=self.config.min_runtime,
            timeout=self.config.timeout,
            shared_memory="10GiB",
            retries=self.config.retries,
            workspace=self.config.workspace,
            budget=self.config.budget,
            extras=install_extras,
            groups=self.effective_groups,
            beaker_image=self.effective_image,
            inject_aws_credentials=self.inject_aws_credentials,
            inject_gcs_credentials=self.inject_gcs_credentials,
            env_vars=job_env_vars,
            env_secrets=env_secrets,
            provider_packages=provider_packages,
            task_packages=task_packages,
            enable_sandbox=self.enable_sandbox,
            setup_registry_mirror=setup_registry_mirror,
            setup_store_secrets=self.config.store,
            vllm_isolated_venv=vllm_isolated_venv,
            setup_modal_gcp_secret=setup_modal_gcp_secret,
        )

    def _extract_task_dependencies(
        self, task_specs: list[str], task_overrides: dict[str, list[str]]
    ) -> list[str] | None:
        from olmo_eval.common.configs import expand_tasks
        from olmo_eval.evals.tasks.common import get_task_dependencies, parse_overrides

        # Expand suites to individual tasks before extracting dependencies
        expanded_specs = expand_tasks(task_specs)

        # Get dependencies from registered task configs
        deps = get_task_dependencies(expanded_specs)

        for _task_spec, overrides in task_overrides.items():
            for override_str in overrides:
                parsed = parse_overrides(override_str)
                if "dependencies" in parsed:
                    override_deps = parsed["dependencies"]
                    if isinstance(override_deps, list):
                        deps.extend(override_deps)
                    else:
                        deps.append(override_deps)

        deps = list(dict.fromkeys(deps))
        return deps if deps else None

    def _build_command(self, exp: ExperimentPlan) -> list[str]:
        """Build the olmo-eval run command."""
        command: list[str] = ["olmo-eval", "run"]

        command.extend(["-O", BEAKER_RESULT_DIR])

        command.extend(["-m", exp.model_spec])

        for t in exp.tasks:
            command.extend(["-t", t])
            if t in exp.task_overrides:
                for override in exp.task_overrides[t]:
                    command.extend(["-o", override])

        if exp.parallelism > 1:
            command.extend(["--parallelism", str(exp.parallelism)])

        if self.effective_groups:
            command.extend(["--experiment-group", self.effective_groups[0]])

        command.extend(["--experiment-name", exp.name])

        if self.config.store:
            command.append("--store")
            command.extend(["--s3-bucket", self.config.s3_bucket])
            command.extend(["--s3-prefix", self.config.s3_prefix])
            if self.effective_groups:
                command.extend(["--s3-group", self.effective_groups[0]])
            if self.config.s3_endpoint_url:
                command.extend(["--s3-endpoint-url", self.config.s3_endpoint_url])
            if self.config.s3_region != "us-east-1":
                command.extend(["--s3-region", self.config.s3_region])

        if self.config.debug_requests:
            command.append("--debug-requests")
        if self.config.debug_provider:
            command.append("--debug-provider")
        if self.config.force_download_model:
            command.append("--force-download-model")
        if not self.config.save_predictions:
            command.append("--no-save-predictions")
        if not self.config.save_requests:
            command.append("--no-save-requests")
        # Use --inspect if all inspect flags are enabled, otherwise add individual flags
        all_inspect = (
            self.config.inspect_instance
            and self.config.inspect_formatted
            and self.config.inspect_tokens
            and self.config.inspect_response
            and self.config.inspect_request
        )
        if all_inspect:
            command.append("--inspect")
        else:
            if self.config.inspect_instance:
                command.append("--inspect-instance")
            if self.config.inspect_formatted:
                command.append("--inspect-formatted")
            if self.config.inspect_tokens:
                command.append("--inspect-tokens")
            if self.config.inspect_response:
                command.append("--inspect-response")
            if self.config.inspect_request:
                command.append("--inspect-request")

        if self.config.harness:
            command.extend(["--harness", self.config.harness])

        for override in self.config.harness_overrides:
            command.extend(["-o", override])

        return command
