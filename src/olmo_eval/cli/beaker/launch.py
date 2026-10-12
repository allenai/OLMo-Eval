"""Launch command for Beaker jobs.

This module provides the main 'beaker launch' command for submitting evaluation jobs.
Configuration loading, validation, and job assembly are delegated to:
- config_loader.py: LaunchConfigLoader for loading/merging config
- task_validator.py: TaskValidator for task validation and priority grouping
- credentials.py: CredentialManager for AWS/GCS credential handling
- experiment_builder.py: ExperimentPlanBuilder for building experiment plans
- job_assembler.py: JobConfigAssembler for assembling BeakerJobConfig
"""

from typing import TYPE_CHECKING

import click

if TYPE_CHECKING:
    from olmo_eval.cli.beaker.experiment_plan import ExperimentPlan
from rich.panel import Panel
from rich.pretty import Pretty
from rich.table import Table

from olmo_eval.cli.utils import (
    ConfiguredExternalEval,
    ExperimentSummary,
    ExternalEvalSummary,
    OrderedMultiOption,
    RunnerConfig,
    console,
    extract_priority_from_overrides,
    parse_key_value_args,
    process_ordered_args,
    reconstruct_ordered_args,
)
from olmo_eval.common.constants.infrastructure import BEAKER_RESULT_DIR, BEAKER_UV_CACHE_DIR


@click.command()
@click.option(
    "--config",
    "-f",
    type=click.Path(exists=True),
    help="YAML config file (CLI args override config values)",
)
@click.option(
    "--name", "-n", help="Experiment name (auto-generated from model/tasks if not provided)"
)
@click.option(
    "--model",
    "-m",
    multiple=True,
    cls=OrderedMultiOption,
    save_to="_ordered",
    help="Model name or preset (can specify multiple).",
)
@click.option(
    "--task",
    "-t",
    multiple=True,
    cls=OrderedMultiOption,
    save_to="_ordered",
    help="Task name with optional @priority suffix. Use -o after to add task overrides.",
)
@click.option(
    "--override",
    "-o",
    multiple=True,
    cls=OrderedMultiOption,
    save_to="_ordered",
    help="Override for preceding --task or --harness (e.g., -o limit=100)",
)
@click.option("--cluster", "-c", default=None, help="Cluster alias (h100, a100, aus) or full name")
@click.option(
    "--max-gpus-per-node",
    default=None,
    type=int,
    help="Maximum GPUs per node (default: 8). Models are split across experiments if exceeded.",
)
@click.option(
    "--priority",
    "-p",
    type=click.Choice(["low", "normal", "high", "urgent"]),
    default=None,
    help="Job priority level (low, normal, high, urgent). Can also use @priority suffix on tasks.",
)
@click.option(
    "--preemptible/--no-preemptible",
    default=None,
    help="Allow preemption (deprecated by Beaker; prefer --min-runtime)",
)
@click.option(
    "--min-runtime",
    default=None,
    help=(
        "Minimum runtime before Beaker may preempt the job (e.g., 2h, 30m). "
        "Replaces --preemptible/--no-preemptible."
    ),
)
@click.option("--timeout", "-T", default=None, help="Job timeout (e.g., 24h, 30m)")
@click.option("--retries", "-r", type=int, help="Number of retries on failure")
@click.option("--workspace", "-w", help="Beaker workspace")
@click.option("--budget", "-B", help="Beaker budget")
@click.option("--image", "-I", help="Beaker image (e.g., ai2-tylerm/olmo-eval-cu1261-trc280-amd64)")
@click.option(
    "--group",
    "-g",
    multiple=True,
    help="Add experiments to Beaker group(s) (can specify multiple, creates if needed)",
)
@click.option("--dry-run", "-d", is_flag=True, help="Print spec without launching")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
@click.option(
    "--follow/--no-follow",
    default=True,
    help="Follow logs after launch (default). Use --no-follow to submit and exit immediately.",
)
@click.option(
    "--aws-credentials/--no-aws-credentials",
    default=None,
    help="Inject AWS credentials for S3 model access. Auto-detected from s3:// model paths.",
)
@click.option(
    "--gcp-credentials/--no-gcp-credentials",
    "gcs_credentials",
    default=None,
    help="Inject GCP credentials. Auto-detected from gs:// model paths.",
)
@click.option(
    "--upload/--no-upload",
    "upload",
    default=None,
    help=(
        "Upload results to the olmo-eval dashboard from the job with your Google "
        "credentials (default: on; OLMO_EVAL_UPLOAD=0 turns it off)"
    ),
)
@click.option(
    "--api-url",
    default=None,
    help="Ingest service URL for the job (default: $OLMO_EVAL_API_URL, else production)",
)
@click.option("--tag", "tags", multiple=True, help="Label to attach to uploaded runs (repeatable)")
@click.option(
    "--debug-requests",
    is_flag=True,
    help="Log HTTP requests/responses to inference providers",
)
@click.option(
    "--debug-provider",
    is_flag=True,
    help="Enable verbose provider logging",
)
@click.option(
    "--force-download-model",
    is_flag=True,
    help="Force-refresh Hugging Face model/tokenizer cache before loading",
)
@click.option(
    "--save-predictions/--no-save-predictions",
    "save_predictions",
    default=True,
    help="Save per-instance predictions to JSONL (default: enabled)",
)
@click.option(
    "--save-requests/--no-save-requests",
    "save_requests",
    default=True,
    help="Save per-instance requests to JSONL (default: enabled)",
)
@click.option(
    "--inspect",
    is_flag=True,
    help="Enable all inspection flags (instance, formatted, tokens, request, response)",
)
@click.option(
    "--inspect-instance",
    is_flag=True,
    help="Print the first instance of each task before running evaluation",
)
@click.option(
    "--inspect-formatted",
    is_flag=True,
    help="Show formatted prompt (after template applied) before evaluation",
)
@click.option(
    "--inspect-tokens",
    is_flag=True,
    help="Show token array before evaluation",
)
@click.option(
    "--inspect-response",
    is_flag=True,
    help="Print the first response of each task after model generation",
)
@click.option(
    "--inspect-request",
    is_flag=True,
    help="Print the first request of each task before model generation",
)
@click.option(
    "-H",
    "--harness",
    type=str,
    default=None,
    help="Harness preset name",
)
@click.option(
    "--external-eval",
    "-E",
    "external_evals",
    multiple=True,
    help="External evaluation name(s) to run instead of tasks (can specify multiple)",
)
@click.option(
    "--eval-arg",
    "-A",
    "eval_args",
    multiple=True,
    help="Arguments for external evals (key=value or JSON dict, e.g., -A domain=retail)",
)
@click.option(
    "--provider-kwarg",
    "-K",
    "provider_kwargs",
    multiple=True,
    help="Provider kwargs for external evals (key=value, e.g., -K enable_chunked_prefill=true)",
)
@click.option(
    "--uv-cache-dir",
    default=BEAKER_UV_CACHE_DIR,
    show_default=True,
    help="UV cache directory for package downloads (on Weka shared storage)",
)
@click.option(
    "--hf-token/--no-hf-token",
    "hf_token",
    default=None,
    help=(
        "Inject HF_TOKEN from the Beaker secret <user>_HF_TOKEN, copying your local token "
        "there if the secret is missing. Default: inject an existing secret, and copy the "
        "local token only when a task declares HF_TOKEN or a model or dataset is gated or "
        "private. --no-hf-token never injects it."
    ),
)
@click.option(
    "--secret-env",
    multiple=True,
    help="Map Beaker secret to env var: BEAKER_SECRET:ENV_VAR (e.g., my-openai-key:OPENAI_API_KEY)",
)
@click.option(
    "-e",
    "--env",
    "env_vars",
    multiple=True,
    help=(
        "Forward or set a job env var. Use 'KEY' to forward from the local shell, "
        "'KEY=VALUE' to set directly. Repeatable. Values are plain text in the job "
        "spec - use --secret-env for sensitive values."
    ),
)
@click.option(
    "--gpus",
    "-G",
    type=int,
    default=None,
    help="Number of GPUs. Defaults to 1 for GPU providers, 0 otherwise.",
)
def launch(
    config: str | None,
    name: str | None,
    model: tuple[str, ...],
    task: tuple[str, ...],
    override: tuple[str, ...],
    cluster: str | None,
    max_gpus_per_node: int | None,
    priority: str | None,
    preemptible: bool | None,
    min_runtime: str | None,
    timeout: str | None,
    retries: int | None,
    workspace: str | None,
    budget: str | None,
    image: str | None,
    group: tuple[str, ...],
    dry_run: bool,
    yes: bool,
    follow: bool,
    aws_credentials: bool | None,
    gcs_credentials: bool | None,
    upload: bool | None,
    api_url: str | None,
    tags: tuple[str, ...],
    debug_requests: bool,
    debug_provider: bool,
    force_download_model: bool,
    save_predictions: bool,
    save_requests: bool,
    inspect: bool,
    inspect_instance: bool,
    inspect_formatted: bool,
    inspect_tokens: bool,
    inspect_response: bool,
    inspect_request: bool,
    harness: str | None,
    external_evals: tuple[str, ...],
    eval_args: tuple[str, ...],
    provider_kwargs: tuple[str, ...],
    uv_cache_dir: str,
    hf_token: bool | None,
    secret_env: tuple[str, ...],
    env_vars: tuple[str, ...],
    gpus: int | None,
) -> None:
    """Launch an evaluation job on Beaker.

    Requires beaker-gantry to be installed: pip install 'olmo-eval[beaker]'
    """
    try:
        from olmo_eval.launch import BeakerLauncher, EvalConfig
    except ImportError:
        console.print(
            "[red]beaker-gantry is not installed.[/red]\n"
            "Install with: pip install 'olmo-eval[beaker]'"
        )
        raise SystemExit(1) from None

    # Process ordered args to associate overrides with models/tasks
    import sys

    from olmo_eval.cli.beaker.config_loader import LaunchConfigLoader
    from olmo_eval.cli.beaker.credentials import CredentialManager
    from olmo_eval.cli.beaker.experiment_builder import ExperimentPlanBuilder
    from olmo_eval.cli.beaker.job_assembler import JobConfigAssembler
    from olmo_eval.cli.beaker.task_validator import TaskValidator
    from olmo_eval.common.constants.infrastructure import (
        BEAKER_DEFAULT_IMAGE,
        BEAKER_SANDBOX_IMAGE,
    )

    ordered_args = reconstruct_ordered_args(sys.argv[1:])
    raw_task_overrides, harness_overrides = process_ordered_args(ordered_args)

    # Extract priority from task overrides (e.g., -o priority=urgent after -t)
    # This is done once here and the filtered overrides are used everywhere
    override_priority, task_overrides = extract_priority_from_overrides(raw_task_overrides)

    # Expand --inspect to enable all individual inspect flags
    if inspect:
        inspect_instance = True
        inspect_formatted = True
        inspect_tokens = True
        inspect_response = True
        inspect_request = True

    # Parse secret_env mappings: BEAKER_SECRET:ENV_VAR -> {beaker_secret: env_var}
    secret_env_overrides: dict[str, str] = {}
    for mapping in secret_env:
        if ":" not in mapping:
            console.print(
                f"[red]Error:[/red] Invalid --secret-env format '{mapping}'. "
                "Expected BEAKER_SECRET:ENV_VAR"
            )
            raise SystemExit(1)
        beaker_secret, env_var = mapping.split(":", 1)
        secret_env_overrides[beaker_secret] = env_var

    # Parse --env/-e entries: "KEY" forwards from local shell, "KEY=VALUE" sets directly
    import os as _os

    parsed_env_vars: dict[str, str] = {}
    for entry in env_vars:
        if "=" in entry:
            key, value = entry.split("=", 1)
            if not key:
                console.print(f"[red]Error:[/red] Invalid --env entry '{entry}': empty key")
                raise SystemExit(1)
            parsed_env_vars[key] = value
        else:
            local_value = _os.environ.get(entry)
            if local_value is None:
                console.print(
                    f"[yellow]Warning:[/yellow] --env {entry} is not set in the local "
                    "shell, skipping"
                )
                continue
            parsed_env_vars[entry] = local_value

    from olmo_eval.upload import resolve_upload_config
    from olmo_eval.upload.config import API_URL_ENV, upload_param_hint

    try:
        upload_config = resolve_upload_config(upload, api_url, tags)
    except ValueError as e:
        raise click.BadParameter(str(e), param_hint=upload_param_hint(e)) from None
    if upload_config.notice:
        console.print(f"[yellow]Note:[/yellow] {upload_config.notice}")
    job_api_url = api_url or _os.environ.get(API_URL_ENV) or None

    # Build CLI args dict
    cli_args = {
        "name": name,
        "model": model,
        "task": task,
        "task_overrides": task_overrides,  # Already filtered (priority extracted)
        "cluster": cluster,
        "max_gpus_per_node": max_gpus_per_node,
        "priority": priority,
        "preemptible": preemptible,
        "min_runtime": min_runtime,
        "timeout": timeout,
        "retries": retries,
        "workspace": workspace,
        "budget": budget,
        "image": image,
        "group": group,
        "gpus": gpus,
        "upload": upload_config.enabled,
        "api_url": job_api_url,
        "tags": list(upload_config.tags),
        "debug_requests": debug_requests,
        "debug_provider": debug_provider,
        "force_download_model": force_download_model,
        "save_predictions": save_predictions,
        "save_requests": save_requests,
        "inspect_instance": inspect_instance,
        "inspect_formatted": inspect_formatted,
        "inspect_tokens": inspect_tokens,
        "inspect_response": inspect_response,
        "inspect_request": inspect_request,
        "harness": harness,
        "harness_overrides": harness_overrides,
        "uv_cache_dir": uv_cache_dir,
        "secret_env_overrides": secret_env_overrides,
        "env_vars": parsed_env_vars,
    }

    # Handle external evaluations mode
    if external_evals:
        if task:
            console.print(
                "[red]Error:[/red] Cannot mix --external-eval (-E) with --task (-t). "
                "This will be supported in a future release. "
                "For now, use separate commands for external evals and tasks."
            )
            raise SystemExit(1)

        # Parse eval_args and provider_kwargs with type coercion
        try:
            parsed_eval_args = parse_key_value_args(eval_args, coerce_types=True)
        except ValueError as e:
            console.print(f"[red]Error:[/red] Invalid eval arg: {e}")
            raise SystemExit(1) from None

        try:
            parsed_provider_kwargs = parse_key_value_args(provider_kwargs, coerce_types=True)
        except ValueError as e:
            console.print(f"[red]Error:[/red] Invalid provider kwarg: {e}")
            raise SystemExit(1) from None

        _launch_external_evals(
            external_evals=list(external_evals),
            model=model,
            name=name,
            cluster=cluster,
            priority=priority,
            timeout=timeout,
            workspace=workspace,
            budget=budget,
            image=image,
            group=group,
            dry_run=dry_run,
            yes=yes,
            follow=follow,
            aws_credentials=aws_credentials,
            gcs_credentials=gcs_credentials,
            upload=upload_config.enabled,
            api_url=job_api_url,
            tags=list(upload_config.tags),
            secret_env_overrides=secret_env_overrides,
            hf_token=hf_token,
            user_env_vars=parsed_env_vars,
            eval_args=parsed_eval_args if parsed_eval_args else None,
            provider_kwargs=parsed_provider_kwargs if parsed_provider_kwargs else None,
            uv_cache_dir=uv_cache_dir,
            preemptible=preemptible,
            min_runtime=min_runtime,
            retries=retries,
            gpus=gpus,
            force_download_model=force_download_model,
        )
        return

    # Load configuration
    config_loader = LaunchConfigLoader(config, cli_args)
    launch_config = config_loader.load()

    # Load EvalConfig for resource lookups
    eval_config: EvalConfig | None = None
    if config:
        import contextlib

        with contextlib.suppress(Exception):
            eval_config = EvalConfig.from_yaml(config)

    # Validate tasks and group by priority
    # Use override_priority from -o priority=X if specified, else use config priority
    effective_priority = override_priority or launch_config.priority
    task_validator = TaskValidator(
        launch_config.task_specs,
        default_priority=effective_priority,
    )
    tasks_by_priority, valid_tasks = task_validator.validate_and_group()

    # Create launcher
    launcher = BeakerLauncher(workspace=launch_config.workspace)

    # Set up credentials
    cred_manager = CredentialManager(
        launch_config.model_specs,
        aws_credentials,
        gcs_credentials,
    )
    inject_aws, inject_gcs = cred_manager.detect_and_setup(launcher)
    _require_upload_credentials(launch_config.upload, dry_run)

    if dry_run:
        console.print("[yellow]Dry run mode - not submitting[/yellow]")

    # Auto-generate group if needed
    effective_groups = _auto_generate_group(launch_config.name, list(launch_config.groups))

    cred_manager.display_aws_info(launcher, inject_aws)

    # Determine effective image
    # Check if harness has a sandbox configured - if so, use sandbox image
    # Apply harness overrides first so -o sandbox.mode=docker works correctly
    harness_needs_sandbox = False
    harness_preset = None
    if launch_config.harness:
        from olmo_eval.cli.run.config import _apply_harness_overrides
        from olmo_eval.harness import get_harness_preset

        harness_preset = get_harness_preset(launch_config.harness)
        if launch_config.harness_overrides:
            harness_preset = _apply_harness_overrides(
                harness_preset, launch_config.harness_overrides
            )
        harness_needs_sandbox = bool(harness_preset.sandboxes)

    if image:
        effective_image = image
    elif eval_config and eval_config.beaker_image:
        effective_image = eval_config.beaker_image
    elif harness_needs_sandbox:
        effective_image = BEAKER_SANDBOX_IMAGE
    else:
        effective_image = BEAKER_DEFAULT_IMAGE

    # Handle group creation
    _handle_group_creation(launcher, effective_groups, dry_run, yes)

    # Build experiment plan
    experiment_builder = ExperimentPlanBuilder(launch_config, tasks_by_priority, override_priority)
    experiment_plan, split_models = experiment_builder.build()

    # Get task configs with overrides applied
    # (task_overrides is already filtered - priority extracted above)
    task_configs_by_spec = _get_task_configs(valid_tasks, launch_config.task_overrides)

    # Collect required secrets from tasks
    all_required_secrets: set[str] = set()
    for task_cfg in task_configs_by_spec.values():
        if hasattr(task_cfg, "required_secrets") and task_cfg.required_secrets:
            all_required_secrets.update(task_cfg.required_secrets)

    # Collect required secrets from model presets
    from olmo_eval.common.configs import get_provider_config

    for model_spec in launch_config.model_specs:
        provider_config = get_provider_config(model_spec)
        if provider_config.required_secrets:
            all_required_secrets.update(provider_config.required_secrets)

    # Collect harness-required secrets
    if launch_config.harness:
        from olmo_eval.harness import get_harness_preset

        harness_config = get_harness_preset(launch_config.harness)
        if harness_config.required_secrets:
            all_required_secrets.update(harness_config.required_secrets)
        # Collect sandbox-required secrets
        for sandbox in harness_config.sandboxes:
            if sandbox.required_secrets:
                all_required_secrets.update(sandbox.required_secrets)

    # Ensure secrets
    common_secrets, task_secrets = _prepare_secrets(
        launcher,
        dry_run=dry_run,
        all_required_secrets=all_required_secrets,
        hf_token=hf_token,
        hf_repos=_hf_repos(launch_config.model_specs, task_configs_by_spec.values()),
    )

    # Print summary header
    total_experiments = len(experiment_plan)
    total_expanded_tasks = len(valid_tasks) * len(launch_config.model_specs)
    console.print()
    console.print(
        f"[bold]Launching {total_experiments} experiment(s) "
        f"with {total_expanded_tasks} task(s)[/bold]"
    )
    if split_models:
        console.print(
            "[dim]  Tasks distributed across multiple experiments due to GPU constraints[/dim]"
        )

    # Show experiment matrix if multiple experiments
    if total_experiments > 1:
        _print_experiment_matrix(experiment_plan)

    console.print()

    # Build job configs and summaries
    job_assembler = JobConfigAssembler(
        launch_config,
        effective_image,
        effective_groups,
        launcher.beaker.user_name,
        common_secrets,
        task_secrets,
        inject_aws,
        inject_gcs,
        enable_sandbox=harness_needs_sandbox,
        secret_env_overrides=launch_config.secret_env_overrides,
    )

    job_configs = []
    experiment_summaries = []

    for exp in experiment_plan:
        job_config = job_assembler.assemble(exp)
        job_configs.append(job_config)

        exp_summary = _build_experiment_summary(
            exp,
            job_config,
            task_configs_by_spec,
            launch_config.harness,
            launch_config.harness_overrides,
        )
        experiment_summaries.append(exp_summary)

    # Print experiment summaries
    for exp_summary in experiment_summaries:
        console.print(
            Panel(
                Pretty(exp_summary, expand_all=True),
                title=f"[bold]{exp_summary.name}[/bold]",
                border_style="cyan",
                expand=True,
            )
        )
        console.print()

    # Confirm and launch
    if not dry_run and not yes and not click.confirm("Proceed with launch?", default=True):
        console.print("[yellow]Launch cancelled[/yellow]")
        raise SystemExit(0)

    launched_experiments = _launch_jobs(launcher, job_configs, dry_run)

    # Follow launched experiments
    if launched_experiments and not dry_run:
        _handle_follow(launcher, launched_experiments, follow)


def _handle_group_creation(
    launcher, effective_groups: list[str], dry_run: bool, yes: bool = False
) -> None:
    """Handle checking and creating Beaker groups."""
    from beaker.exceptions import BeakerGroupNotFound

    existing_groups = []
    missing_groups = []

    for grp in effective_groups:
        qualified_name = f"{launcher.beaker.user_name}/{grp}" if "/" not in grp else grp
        try:
            launcher.beaker.group.get(qualified_name)
            existing_groups.append(grp)
        except BeakerGroupNotFound:
            missing_groups.append(grp)

    if dry_run:
        if missing_groups:
            console.print(
                f"[yellow]Note:[/yellow] The following groups would be created: "
                f"{', '.join(missing_groups)}"
            )
    else:
        if missing_groups:
            console.print(
                f"\n[yellow]The following groups do not exist:[/yellow] {', '.join(missing_groups)}"
            )
            confirmed = yes or click.confirm("Would you like to create these groups?", default=True)
            if not confirmed:
                console.print("[red]Aborted.[/red] Cannot launch without required groups.")
                raise SystemExit(1) from None

            workspace_obj = launcher.beaker.workspace.get(launcher._workspace)
            for grp in missing_groups:
                try:
                    beaker_group = launcher.beaker.group.create(name=grp, workspace=workspace_obj)
                    group_url = launcher.get_group_url(beaker_group)
                    console.print(f"[green]  Created {grp}:[/green] {group_url}")
                except Exception as e:
                    console.print(f"[red]Error:[/red] Failed to create group '{grp}': {e}")
                    raise SystemExit(1) from None


def _get_task_configs(
    valid_tasks: list[str], task_overrides: dict[str, list[str]] | None = None
) -> dict:
    """Get task configs with overrides applied.

    Args:
        valid_tasks: List of task specifications.
        task_overrides: Optional dict of task_spec -> override strings from CLI
                       (already filtered - priority extracted).

    Returns:
        Dict mapping task_spec -> TaskConfig (with overrides applied).
    """
    from copy import deepcopy

    from olmo_eval.cli.run.config import _apply_dotlist_overrides
    from olmo_eval.evals.tasks.common import get_task as get_task_instance
    from olmo_eval.evals.tasks.common import parse_task_spec

    task_overrides = task_overrides or {}
    task_configs = {}

    for task_spec in valid_tasks:
        task_name, variants, _overrides = parse_task_spec(task_spec)
        try:
            task_instance = get_task_instance(task_spec)
        except ValueError as e:
            console.print(f"[red]Error loading task '{task_spec}':[/red] {e}")
            raise SystemExit(1) from None

        # Deep copy the config so we can apply overrides directly
        task_cfg = deepcopy(task_instance.config)

        # Apply CLI overrides using dotlist format (supports nested keys and JSON)
        cli_overrides = task_overrides.get(task_spec, [])
        if cli_overrides:
            # Convert config to dict, apply overrides, convert back
            task_dict = task_cfg.to_dict() if hasattr(task_cfg, "to_dict") else vars(task_cfg)
            task_dict = _apply_dotlist_overrides(task_dict, cli_overrides)
            # Update config attributes from dict
            for key, value in task_dict.items():
                if hasattr(task_cfg, key):
                    setattr(task_cfg, key, value)

        task_configs[task_spec] = task_cfg

    return task_configs


def _require_upload_credentials(upload: bool, dry_run: bool) -> None:
    """Stop before launching when uploads are on but no local Google credentials exist.

    The launch reads the uploader key from Secret Manager with the local credentials, so
    they never leave this machine.
    """
    from olmo_eval.upload.auth import NO_CREDENTIALS_MESSAGE, has_local_google_credentials

    if not upload or has_local_google_credentials():
        return
    if dry_run:
        console.print(f"[yellow]Warning:[/yellow] {NO_CREDENTIALS_MESSAGE}")
        return
    console.print(f"[red]Error:[/red] {NO_CREDENTIALS_MESSAGE}")
    raise SystemExit(1)


def _print_experiment_matrix(experiment_plan: list["ExperimentPlan"]) -> None:
    """Print experiment matrix table."""
    matrix_table = Table(show_header=True, title="Experiment Plan")
    matrix_table.add_column("Name", style="cyan")
    matrix_table.add_column("Models", style="blue")
    matrix_table.add_column("Tasks", style="dim")
    matrix_table.add_column("Priority", style="yellow")
    matrix_table.add_column("GPUs", style="green", justify="right")

    for exp in experiment_plan:
        model_display = exp.model_spec

        if len(exp.tasks) <= 3:
            task_display = ", ".join(exp.tasks)
        else:
            task_display = f"{exp.tasks[0]}, ... ({len(exp.tasks)} total)"

        matrix_table.add_row(
            exp.name,
            model_display,
            task_display,
            exp.priority,
            str(exp.num_gpus),
        )

    console.print(matrix_table)


def _build_experiment_summary(
    exp: "ExperimentPlan",
    job_config,
    task_configs_by_spec: dict,
    harness: str | None = None,
    harness_overrides: list[str] | None = None,
) -> ExperimentSummary:
    """Build experiment summary for display."""
    from olmo_eval.common.configs import expand_tasks
    from olmo_eval.runners.asynq.runner import AsyncEvalRunner

    # Expand task specs to match keys in task_configs_by_spec
    expanded_tasks = expand_tasks(exp.tasks)
    exp_task_configs = []
    for task_spec in expanded_tasks:
        base_spec = task_spec.rsplit("@", 1)[0] if "@" in task_spec else task_spec
        if base_spec in task_configs_by_spec:
            exp_task_configs.append(task_configs_by_spec[base_spec])

    exp_runner_class = AsyncEvalRunner

    exp_runner_config = RunnerConfig(
        runner=exp_runner_class,
        output_dir=BEAKER_RESULT_DIR,
    )

    from olmo_eval.cli.run.config import _apply_harness_overrides, merge_model_provider
    from olmo_eval.common.configs import get_provider_config
    from olmo_eval.harness import get_harness_preset

    harness_config = get_harness_preset(harness or "default")
    if harness_overrides:
        harness_config = _apply_harness_overrides(harness_config, harness_overrides)
    provider_config = get_provider_config(exp.model_spec)
    harness_config = merge_model_provider(harness_config, provider_config, harness_overrides)

    return ExperimentSummary(
        name=exp.name,
        tasks=exp_task_configs,
        harness=harness_config,
        runner=exp_runner_config,
        beaker=job_config,
    )


def _handle_follow(launcher, launched_experiments: list[str], follow: bool) -> None:
    """Handle following launched experiments."""
    if len(launched_experiments) > 1:
        console.print(f"\n[bold]Launched {len(launched_experiments)} experiment(s)[/bold]")

    if follow:
        if len(launched_experiments) == 1:
            import sys

            exit_code = launcher.follow_experiment(launched_experiments[0])
            sys.exit(exit_code)
        else:
            console.print(
                "\n[bold]Multiple experiments launched. "
                "Use 'olmo-eval beaker watch -e <id>' to follow:[/bold]"
            )
            for exp_id in launched_experiments:
                url = launcher.get_experiment_url(exp_id)
                console.print(f"  - {url}")


def _auto_generate_group(prefix: str, existing_groups: list[str]) -> list[str]:
    """Auto-generate a group name if none provided.

    Args:
        prefix: Prefix for the auto-generated group name.
        existing_groups: List of existing group names.

    Returns:
        List of group names (original if non-empty, or auto-generated).
    """
    from datetime import datetime

    if existing_groups:
        return existing_groups
    return [f"{prefix}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"]


def _hf_repos(model_specs, task_configs=()) -> list[tuple[str, str]]:
    """Collect the Hugging Face models and datasets a launch reads, for the gating check.

    Values that are not Hugging Face repo ids (paths, s3:// or gs:// URIs, API model
    names) are filtered out later by ``check_hf_access``.
    """
    from olmo_eval.common.configs import get_provider_config
    from olmo_eval.data import DataSource
    from olmo_eval.data.sources import SourceType

    repos: list[tuple[str, str]] = []
    for model_spec in model_specs:
        repos.append((model_spec, "model"))
        try:
            provider_config = get_provider_config(model_spec)
        except Exception:
            continue
        for repo in (provider_config.model, provider_config.tokenizer):
            if repo:
                repos.append((repo, "model"))
    for task_cfg in task_configs:
        for source in (
            getattr(task_cfg, "data_source", None),
            getattr(task_cfg, "fewshot_source", None),
        ):
            if isinstance(source, str):
                try:
                    source = DataSource.from_uri(source)
                except Exception:
                    continue
            if isinstance(source, DataSource) and source.source_type == SourceType.HF:
                repos.append((source.path, "dataset"))
    return repos


def _prepare_secrets(
    launcher,
    *,
    dry_run: bool,
    all_required_secrets: set[str],
    hf_token: bool | None = None,
    hf_repos: list[tuple[str, str]] | None = None,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Prepare common and task secrets for Beaker jobs.

    The Hugging Face token is injected from an existing ``<user>_HF_TOKEN`` secret, and
    the local token is copied there only when the job needs it (see ``plan_hf_token``).
    A dry run neither reads nor writes Beaker secrets.

    Args:
        launcher: BeakerLauncher whose client defaults to the job's workspace.
        dry_run: If True, return the secrets that would be used without touching Beaker.
        all_required_secrets: Set of required secret environment variable names.
        hf_token: ``--hf-token`` (True), ``--no-hf-token`` (False), or auto (None).
        hf_repos: (repo_id, repo_type) pairs checked for gated or private access.

    Returns:
        Tuple of (common_secrets, task_secrets) as lists of (env_var, secret_name) tuples.
    """
    from olmo_eval.launch.beaker.secrets import (
        COMMON_SECRET_NAMES,
        HF_TOKEN_ENV,
        HfAccessCheck,
        check_hf_access,
        ensure_common_secrets,
        ensure_task_secrets,
        get_local_hf_token,
        plan_hf_token,
        secret_exists,
        write_secret,
    )

    beaker_username = launcher.beaker.user_name
    workspace = launcher._workspace
    task_required_secrets = all_required_secrets - COMMON_SECRET_NAMES
    local_hf_token = get_local_hf_token()

    secret_present: bool | None = None
    if not dry_run and hf_token is not False:
        secret_present = secret_exists(launcher.beaker, f"{beaker_username}_{HF_TOKEN_ENV}")
    # Skip the Hugging Face lookups when the answer cannot change the plan.
    if hf_token is None and not secret_present:
        access = check_hf_access(hf_repos or [], token=local_hf_token)
    else:
        access = HfAccessCheck()
    plan = plan_hf_token(
        username=beaker_username,
        mode=hf_token,
        secret_present=secret_present,
        required=HF_TOKEN_ENV in all_required_secrets,
        access=access,
        have_local_token=local_hf_token is not None,
    )
    if plan.error:
        if not dry_run:
            console.print(f"[red]Error:[/red] {plan.error}")
            raise SystemExit(1)
        console.print(f"[yellow]Warning:[/yellow] {plan.error}")
    elif plan.warning:
        console.print(f"[yellow]Warning:[/yellow] {plan.message}")
    elif plan.message:
        console.print(f"[dim]{plan.message}[/dim]")

    common_secrets: list[tuple[str, str]] = []
    if dry_run:
        # Jobs are not built in a dry run, so explain the BEAKER_TOKEN rule here.
        launcher.beaker_token_secret(workspace, dry_run=True)
        task_secrets = [(s, f"{beaker_username}_{s}") for s in sorted(task_required_secrets)]
    else:
        if plan.copy_local and local_hf_token:
            write_secret(launcher.beaker, plan.secret_name, local_hf_token)
        common_secrets = ensure_common_secrets(workspace=workspace)
        try:
            task_secrets = ensure_task_secrets(
                workspace=workspace,
                required_secrets=task_required_secrets,
            )
        except ValueError as e:
            console.print(f"[red]Error:[/red] {e}")
            raise SystemExit(1) from None
    if plan.inject:
        common_secrets.insert(0, (HF_TOKEN_ENV, plan.secret_name))

    return common_secrets, task_secrets


def _launch_jobs(
    launcher,
    job_configs: list,
    dry_run: bool,
) -> list[str]:
    """Launch job configs and return experiment IDs.

    Args:
        launcher: BeakerLauncher instance.
        job_configs: List of BeakerJobConfig to launch.
        dry_run: If True, skip actual launch.

    Returns:
        List of launched experiment IDs.
    """
    launched_experiments: list[str] = []
    for job_config in job_configs:
        if not dry_run:
            experiment = launcher.launch(job_config)
            if experiment:
                console.print(f"[green]Launched:[/green] {launcher.experiment_url(experiment)}")
                launched_experiments.append(experiment.id)
    return launched_experiments


def _launch_external_evals(
    external_evals: list[str],
    model: tuple[str, ...],
    name: str | None,
    cluster: str | None,
    priority: str | None,
    timeout: str | None,
    workspace: str | None,
    budget: str | None,
    image: str | None,
    group: tuple[str, ...],
    dry_run: bool,
    yes: bool,
    follow: bool,
    aws_credentials: bool | None,
    gcs_credentials: bool | None,
    upload: bool = True,
    api_url: str | None = None,
    tags: list[str] | None = None,
    secret_env_overrides: dict[str, str] | None = None,
    hf_token: bool | None = None,
    user_env_vars: dict[str, str] | None = None,
    eval_args: dict[str, str] | None = None,
    provider_kwargs: dict[str, str] | None = None,
    uv_cache_dir: str | None = None,
    preemptible: bool | None = None,
    min_runtime: str | None = None,
    retries: int | None = None,
    gpus: int | None = None,
    force_download_model: bool = False,
) -> None:
    """Launch external evaluation jobs on Beaker.

    This path mirrors the task-based launch flow but for external evaluations
    that run in sandbox containers with subcontainer support.
    """
    from olmo_eval.cli.beaker.config_loader import resolve_preemption
    from olmo_eval.cli.beaker.credentials import CredentialManager
    from olmo_eval.cli.beaker.job_assembler import assemble_external_eval_job
    from olmo_eval.common.configs import get_provider_config
    from olmo_eval.common.constants.infrastructure import (
        BEAKER_DEFAULT_WORKSPACE,
        BEAKER_SANDBOX_IMAGE,
    )
    from olmo_eval.evals.external import get_external_eval, is_external_eval_registered
    from olmo_eval.launch import BeakerLauncher, get_model_short_name, sanitize_beaker_name

    secret_env_overrides = secret_env_overrides or {}

    # Validate external evals exist
    for eval_name in external_evals:
        if not is_external_eval_registered(eval_name):
            from olmo_eval.evals.external import list_external_evals

            available = list_external_evals()
            console.print(
                f"[red]Error:[/red] External eval '{eval_name}' not found. "
                f"Available: {', '.join(available) or '(none)'}"
            )
            raise SystemExit(1)

    # Validate model is provided
    if not model:
        console.print("[red]Error:[/red] --model is required for external evaluations")
        raise SystemExit(1)

    # Validate cluster is provided
    if not cluster:
        console.print(
            "[red]Error:[/red] --cluster is required for external evaluations. "
            "Use -c h100, -c a100, or a full cluster name."
        )
        raise SystemExit(1)

    # Use defaults if not provided
    effective_workspace = workspace or BEAKER_DEFAULT_WORKSPACE
    effective_priority = priority or "normal"
    effective_timeout = timeout or "24h"
    effective_image = image or BEAKER_SANDBOX_IMAGE
    try:
        effective_preemptible, effective_min_runtime = resolve_preemption(preemptible, min_runtime)
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise SystemExit(1) from None

    # Create launcher
    launcher = BeakerLauncher(workspace=effective_workspace)
    beaker_username = launcher.beaker.user_name

    # Use CredentialManager for consistent credential detection (same as task path)
    cred_manager = CredentialManager(
        model_specs=list(model),
        aws_credentials=aws_credentials,
        gcs_credentials=gcs_credentials,
    )
    inject_aws, inject_gcs = cred_manager.detect_and_setup(launcher)
    _require_upload_credentials(upload, dry_run)

    # Auto-generate group if needed (using shared helper)
    effective_groups = _auto_generate_group("external-eval", list(group))

    # Handle group creation
    _handle_group_creation(launcher, effective_groups, dry_run, yes)

    cred_manager.display_aws_info(launcher, inject_aws)

    # Collect required secrets from external evals and models
    all_required_secrets: set[str] = set()
    for eval_name in external_evals:
        eval_instance = get_external_eval(eval_name)
        if eval_instance.required_secrets:
            all_required_secrets.update(eval_instance.required_secrets)

    for model_spec in model:
        try:
            provider_config = get_provider_config(model_spec)
            if provider_config.required_secrets:
                all_required_secrets.update(provider_config.required_secrets)
        except Exception:
            pass

    # Prepare secrets (using shared helper)
    common_secrets, task_secrets = _prepare_secrets(
        launcher,
        dry_run=dry_run,
        all_required_secrets=all_required_secrets,
        hf_token=hf_token,
        hf_repos=_hf_repos(model),
    )

    # Build env secrets list
    env_secrets = common_secrets + task_secrets

    # Add explicit secret overrides
    env_secrets.extend(
        (env_var, beaker_secret) for beaker_secret, env_var in secret_env_overrides.items()
    )

    if dry_run:
        console.print("[yellow]Dry run mode - not submitting[/yellow]")

    # Build jobs and summaries for each model
    from uuid import uuid4

    launch_id = uuid4().hex[:12]
    job_configs = []
    summaries = []
    for model_spec in model:
        # Get provider config for model alias and as base for summary
        try:
            provider_config = get_provider_config(model_spec)
            model_alias = provider_config.alias
        except Exception:
            from olmo_eval.inference.providers.config import ProviderConfig

            provider_config = ProviderConfig(kind="vllm_server", model=model_spec)
            model_alias = None

        # Determine num_gpus: CLI --gpus takes precedence, then default to 1
        num_gpus = gpus if gpus is not None else 1

        # tensor_parallel_size defaults to num_gpus unless explicitly overridden via -K
        tensor_parallel_size = num_gpus
        if provider_kwargs and "tensor_parallel_size" in provider_kwargs:
            tensor_parallel_size = int(provider_kwargs["tensor_parallel_size"])

        # Update provider_config with tensor_parallel_size for display
        display_overrides: dict[str, object] = {"tensor_parallel_size": tensor_parallel_size}
        if force_download_model:
            display_overrides["force_download"] = True
        display_provider_config = provider_config.with_overrides(**display_overrides)

        # Generate experiment name using shared utility
        short_name = get_model_short_name(model_spec, model_alias)
        if name:
            exp_name = name
        elif len(external_evals) <= 2:
            # Short list: include all eval names
            exp_name = sanitize_beaker_name(f"{short_name}-{'-'.join(external_evals)}-external")
        else:
            # Many evals: show first eval + count to keep name reasonable
            exp_name = sanitize_beaker_name(
                f"{short_name}-{external_evals[0]}-and-{len(external_evals) - 1}-more-external"
            )

        job_config = assemble_external_eval_job(
            name=exp_name,
            model=model_spec,
            external_evals=external_evals,
            cluster=cluster,
            num_gpus=num_gpus,
            workspace=effective_workspace,
            beaker_image=effective_image,
            priority=effective_priority,
            timeout=effective_timeout,
            budget=budget,
            groups=effective_groups,
            tensor_parallel_size=tensor_parallel_size,
            upload=upload,
            api_url=api_url,
            tags=tags,
            launch_id=launch_id,
            env_secrets=env_secrets,
            inject_aws_credentials=inject_aws,
            inject_gcs_credentials=inject_gcs,
            eval_args=eval_args,
            provider_kwargs=provider_kwargs,
            uv_cache_dir=uv_cache_dir,
            beaker_username=beaker_username,
            preemptible=effective_preemptible,
            min_runtime=effective_min_runtime,
            retries=retries,
            provider_kind=str(provider_config.kind),
            base_url=provider_config.base_url,
            user_env_vars=user_env_vars,
            force_download_model=force_download_model,
        )
        job_configs.append(job_config)

        # Build summary with updated provider config
        configured_evals = [
            ConfiguredExternalEval.from_eval(
                get_external_eval(e), display_provider_config, eval_args
            )
            for e in external_evals
        ]
        summary = ExternalEvalSummary(
            name=job_config.name,
            evals=configured_evals,
            beaker=job_config,
        )
        summaries.append(summary)

    # Print summaries
    console.print()
    console.print(f"[bold]Launching {len(job_configs)} external evaluation job(s)[/bold]")
    console.print()

    for summary in summaries:
        console.print(
            Panel(
                Pretty(summary, expand_all=True),
                title=f"[bold]{summary.name}[/bold]",
                border_style="cyan",
                expand=True,
            )
        )
        console.print()

    # Confirm and launch (using shared helper)
    if not dry_run and not yes and not click.confirm("Proceed with launch?", default=True):
        console.print("[yellow]Launch cancelled[/yellow]")
        raise SystemExit(0)

    launched_experiments = _launch_jobs(launcher, job_configs, dry_run)

    # Follow launched experiments
    if launched_experiments and not dry_run:
        _handle_follow(launcher, launched_experiments, follow)
