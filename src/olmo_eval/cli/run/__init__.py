"""Run command for olmo-eval CLI.

This module provides the main 'run' command for executing evaluations.
Configuration parsing and runner creation are delegated to:
- config.py: RunConfigBuilder for parsing and validating CLI arguments
- factory.py: RunnerFactory for creating appropriate runners
- options.py: Option decorators for logical grouping
"""

import contextlib
import signal
import threading
from collections.abc import Iterator
from datetime import UTC, datetime

import click

from olmo_eval.cli.run.options import (
    experiment_options,
    harness_options,
    inspect_options,
    output_options,
    parallelism_options,
    upload_options,
)
from olmo_eval.cli.utils import (
    OrderedMultiOption,
    console,
    print_runtime_environment,
    process_ordered_args,
    reconstruct_ordered_args,
)


class Terminated(BaseException):
    """SIGTERM arrived while a run was in progress (for example, Beaker preemption)."""


@contextlib.contextmanager
def _sigterm_raises() -> Iterator[None]:
    """Turn SIGTERM into Terminated so cleanup runs and the run is recorded as failed."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    def handler(signum: int, frame: object) -> None:
        raise Terminated()

    previous = signal.signal(signal.SIGTERM, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


@click.command()
# Core options (inline)
@click.option(
    "--model",
    "-m",
    required=True,
    cls=OrderedMultiOption,
    save_to="_ordered",
    help="Model name or preset.",
)
@click.option(
    "--task",
    "-t",
    multiple=True,
    required=True,
    cls=OrderedMultiOption,
    save_to="_ordered",
    help="Task spec or suite. Use --override after to add task overrides.",
)
@click.option(
    "--override",
    "-o",
    "cli_override",
    multiple=True,
    cls=OrderedMultiOption,
    save_to="_ordered",
    help="Override for preceding --task or --harness (e.g., -o limit=100)",
)
@click.option("--config", "-c", type=click.Path(exists=True), help="YAML config file")
@click.option(
    "--force-download-model",
    is_flag=True,
    help="Force-refresh Hugging Face model/tokenizer cache before loading",
)
# Grouped options via decorators
@harness_options
@parallelism_options
@upload_options
@experiment_options
@output_options
@inspect_options
def run(
    # Core options
    model: str,
    task: tuple[str, ...],
    cli_override: tuple[str, ...],
    config: str | None,
    force_download_model: bool,
    # Harness options
    harness_preset: str | None,
    harness_config: str | None,
    # Parallelism options
    num_gpus: int,
    parallelism: int,
    # Upload options
    upload: bool | None,
    api_url: str | None,
    tags: tuple[str, ...],
    # Experiment options
    experiment_name: str | None,
    experiment_group: str | None,
    # Output options
    output_dir: str,
    save_predictions: bool,
    save_requests: bool,
    dry_run: bool,
    # Inspect options
    debug_requests: bool,
    debug_provider: bool,
    inspect: bool,
    inspect_instance: bool,
    inspect_formatted: bool,
    inspect_tokens: bool,
    inspect_response: bool,
    inspect_request: bool,
) -> None:
    """Run evaluation on specified tasks.

    Use -o/--override after --harness or -t/--task to apply overrides:

        olmo-eval run --harness default -o provider.kind=vllm_server \
            -m llama3.1-8b -t mmlu -o limit=100
    """
    import os
    import sys

    from olmo_eval.cli.run.config import RunConfigBuilder
    from olmo_eval.cli.run.factory import RunnerFactory
    from olmo_eval.common.logging import configure_logging
    from olmo_eval.runners import ValidationError
    from olmo_eval.upload import mark_run_failed, resolve_upload_config
    from olmo_eval.upload.config import upload_param_hint

    try:
        upload_config = resolve_upload_config(upload, api_url, tags)
    except ValueError as e:
        raise click.BadParameter(str(e), param_hint=upload_param_hint(e)) from None

    # Process ordered args to associate overrides with tasks/harness
    ordered_args = reconstruct_ordered_args(sys.argv[1:])
    task_overrides, harness_overrides = process_ordered_args(ordered_args)

    # Configure logging for Beaker job visibility
    configure_logging(level="INFO")
    if upload_config.notice:
        console.print(f"[yellow]Note:[/yellow] {upload_config.notice}")

    # Set debug environment variables
    if debug_requests:
        os.environ["OLMO_EVAL_DEBUG_REQUESTS"] = "1"
        os.environ["VLLM_DEBUG_REQUESTS"] = "1"
    if debug_provider:
        os.environ["OLMO_EVAL_DEBUG_PROVIDER"] = "1"

    # Expand --inspect to enable all individual inspect flags
    if inspect:
        inspect_instance = True
        inspect_formatted = True
        inspect_tokens = True
        inspect_response = True
        inspect_request = True

    # Print runtime environment summary
    print_runtime_environment()

    # Build configuration
    config_builder = RunConfigBuilder(
        model=model,
        task=task,
        output_dir=output_dir,
        num_gpus=num_gpus,
        parallelism=parallelism,
        experiment_name=experiment_name,
        experiment_group=experiment_group,
        save_predictions=save_predictions,
        save_requests=save_requests,
        inspect_instance=inspect_instance,
        inspect_formatted=inspect_formatted,
        inspect_tokens=inspect_tokens,
        inspect_response=inspect_response,
        inspect_request=inspect_request,
        cli_task_overrides=task_overrides,
        harness_preset=harness_preset,
        harness_config_path=harness_config,
        cli_harness_overrides=harness_overrides,
        force_download_model=force_download_model,
    )

    from rich.panel import Panel
    from rich.pretty import Pretty

    # Build configuration
    run_config = config_builder.build()

    # Print the config with resolved tasks
    console.print(
        Panel(
            Pretty(run_config, expand_all=True),
            title="[bold]Run Configuration[/bold]",
            border_style="cyan",
        )
    )

    # Create runner factory
    factory = RunnerFactory(run_config, upload_config)

    # Create and run the appropriate runner
    runner = factory.create()

    try:
        runner.validate()
    except ValidationError as e:
        console.print(f"[red]Validation error:[/red]\n{e}")
        raise SystemExit(1) from None

    if dry_run:
        runner.print_config()
    else:
        started = datetime.now(UTC)
        try:
            with _sigterm_raises():
                runner.run()
        except (KeyboardInterrupt, Terminated) as e:
            reason = "terminated (SIGTERM)" if isinstance(e, Terminated) else "interrupted"
            console.print(f"\n[bold red]Evaluation {reason}[/bold red]")
            mark_run_failed(runner.output_dir, upload_config, error=f"Run {reason}", since=started)
            raise SystemExit(143 if isinstance(e, Terminated) else 130) from None
        except Exception as e:
            console.print(f"\n[bold red]Evaluation failed:[/bold red] {e}")
            console.print_exception()
            mark_run_failed(runner.output_dir, upload_config, error=str(e), since=started)
            raise SystemExit(1) from None
