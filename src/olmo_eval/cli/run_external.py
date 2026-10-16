"""CLI command for running external evaluations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import click

from olmo_eval.cli.utils import ConfiguredExternalEval, console, parse_key_value_args
from olmo_eval.common.constants.infrastructure import BEAKER_RESULT_DIR


@dataclass
class ExternalRunConfig:
    """Configuration for an external evaluation run."""

    evals: list[ConfiguredExternalEval]
    output_dir: str
    container_runtime: str
    server_port: int = 8000
    eval_args: dict[str, Any] = field(default_factory=dict)


@click.command(name="run-external")
@click.option(
    "--model",
    "-m",
    required=True,
    help="Model name or path (HuggingFace ID or local path)",
)
@click.option(
    "--eval",
    "-e",
    "evals",
    multiple=True,
    required=True,
    help="External evaluation name(s) to run (can specify multiple)",
)
@click.option(
    "--output-dir",
    "-O",
    default=BEAKER_RESULT_DIR,
    help="Directory to write results",
)
@click.option(
    "--provider",
    "-p",
    default="vllm_server",
    type=click.Choice(["vllm", "vllm_server", "litellm"]),
    help="Inference provider to use",
)
@click.option(
    "--base-url",
    help="Base URL for the inference provider (if already running)",
)
@click.option(
    "--tensor-parallel-size",
    "--tp",
    type=int,
    default=None,
    help="Tensor parallel size for vLLM (overrides model preset)",
)
@click.option(
    "--port",
    type=int,
    default=8000,
    help="Port for the vLLM server",
)
@click.option(
    "--runtime",
    type=click.Choice(["docker", "podman"]),
    default="podman",
    help="Container runtime to use for sandboxes",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Print configuration without running",
)
@click.option(
    "--force-download-model",
    is_flag=True,
    help="Force-refresh Hugging Face model/tokenizer cache before loading",
)
@click.option(
    "--arg",
    "-a",
    "eval_args",
    multiple=True,
    help="Arguments for external evals (key=value or JSON dict format)",
)
@click.option(
    "--provider-kwarg",
    "-K",
    "provider_kwargs",
    multiple=True,
    help="Provider kwargs (key=value, e.g., -K enable_chunked_prefill=true)",
)
# Dashboard upload options
@click.option(
    "--upload/--no-upload",
    "upload",
    default=None,
    help="Upload results to the olmo-eval dashboard (default: on; OLMO_EVAL_UPLOAD=0 disables)",
)
@click.option("--api-url", default=None, help="Dashboard ingest service URL")
@click.option("--tag", "tags", multiple=True, help="Label to attach to the uploaded run")
# Experiment metadata
@click.option("--experiment-name", help="Human-readable experiment name")
@click.option("--experiment-group", help="Experiment group for grouping related experiments")
# Provider metrics options
@click.option(
    "--metrics/--no-metrics",
    "enable_metrics",
    default=True,
    help="Enable provider metrics collection (default: enabled)",
)
@click.option(
    "--collect-gpu/--no-collect-gpu",
    default=False,
    help="Collect GPU metrics (default: disabled)",
)
@click.option(
    "--vllm-metrics/--no-vllm-metrics",
    "collect_vllm_server",
    default=True,
    help="Collect vLLM server metrics via /metrics endpoint (default: enabled)",
)
@click.option(
    "--vllm-poll-interval",
    type=float,
    default=10.0,
    help="vLLM metrics polling interval in seconds (default: 10.0)",
)
def run_external(
    model: str,
    evals: tuple[str, ...],
    output_dir: str,
    provider: str,
    base_url: str | None,
    tensor_parallel_size: int | None,
    port: int,
    runtime: str,
    dry_run: bool,
    force_download_model: bool,
    eval_args: tuple[str, ...],
    provider_kwargs: tuple[str, ...],
    upload: bool | None,
    api_url: str | None,
    tags: tuple[str, ...],
    experiment_name: str | None,
    experiment_group: str | None,
    enable_metrics: bool,
    collect_gpu: bool,
    collect_vllm_server: bool,
    vllm_poll_interval: float,
) -> None:
    """Run external black-box evaluations.

    External evaluations run inside sandbox containers and communicate with
    the model via an OpenAI-compatible API.

    Examples:

        # Run tau2_bench on a model
        olmo-eval run-external -m meta-llama/Llama-3.1-8B-Instruct -e tau2_bench

        # Run with custom arguments (key=value format)
        olmo-eval run-external -m my-model -e tau2_bench -a domain=retail -a num_trials=10

        # Run with custom arguments (JSON format)
        olmo-eval run-external -m my-model -e tau2_bench -a '{"domain": "retail", "num_trials": 10}'

        # Run with custom output directory
        olmo-eval run-external -m my-model -e tau2_bench -O ./results

        # Run multiple evaluations
        olmo-eval run-external -m my-model -e tau2_bench -e asta_bench
    """
    from olmo_eval.common.logging import configure_logging

    configure_logging(level="INFO")

    # Build provider config
    from olmo_eval.common.configs import get_provider_config

    try:
        provider_config = get_provider_config(model)
    except Exception:
        # Fall back to creating a basic config
        from olmo_eval.inference.providers.config import ProviderConfig

        provider_config = ProviderConfig(
            kind=provider,
            model=model,
        )

    # Parse provider kwargs (key=value format, with type coercion)
    try:
        parsed_provider_kwargs = parse_key_value_args(provider_kwargs, coerce_types=True)
    except ValueError as e:
        console.print(f"[red]Error:[/red] Invalid provider kwarg: {e}")
        raise SystemExit(1) from None

    force_download_override = bool(parsed_provider_kwargs.pop("force_download", False))

    # Apply overrides
    provider_overrides: dict[str, Any] = {
        "kind": provider,
        "base_url": base_url,
        "tensor_parallel_size": tensor_parallel_size,
        **parsed_provider_kwargs,
    }
    if force_download_model or force_download_override:
        provider_overrides["force_download"] = True
    provider_config = provider_config.with_overrides(**provider_overrides)

    # Parse eval_args (supports both key=value and JSON dict format)
    try:
        parsed_args = parse_key_value_args(eval_args, coerce_types=True)
    except ValueError as e:
        console.print(f"[red]Error:[/red] Invalid eval arg: {e}")
        raise SystemExit(1) from None

    from olmo_eval.upload import resolve_upload_config
    from olmo_eval.upload.config import upload_param_hint

    try:
        upload_config = resolve_upload_config(upload, api_url, tags)
    except ValueError as e:
        raise click.BadParameter(str(e), param_hint=upload_param_hint(e)) from None
    if upload_config.notice:
        console.print(f"[yellow]Note:[/yellow] {upload_config.notice}")

    # Build metrics config (matches default harness preset)
    from olmo_eval.inference.metrics import MetricsConfig

    metrics_config: MetricsConfig | None = None
    if enable_metrics:
        metrics_config = MetricsConfig(
            enabled=True,
            collect_gpu=collect_gpu,
            collect_vllm_server=collect_vllm_server,
            vllm_poll_interval_s=vllm_poll_interval,
            output_dir=output_dir,
            provider_kind=provider,
            model_name=provider_config.alias or provider_config.model,
        )

    # Create runner
    from olmo_eval.runners.external import ExternalEvalRunner

    runner = ExternalEvalRunner(
        provider_config=provider_config,
        external_eval_names=list(evals),
        output_dir=output_dir,
        container_runtime=runtime,
        server_port=port,
        eval_args=parsed_args,
        upload_config=upload_config,
        experiment_name=experiment_name,
        experiment_group=experiment_group,
        metrics=metrics_config,
    )

    # Validate
    try:
        runner.validate()
    except ValueError as e:
        console.print(f"[red]Validation error:[/red] {e}")
        raise SystemExit(1) from None

    # Print configuration
    from rich.panel import Panel
    from rich.pretty import Pretty
    from rich.table import Table

    from olmo_eval.evals.external import get_external_eval

    configured_evals = [
        ConfiguredExternalEval.from_eval(get_external_eval(name), provider_config, parsed_args)
        for name in evals
    ]

    run_config = ExternalRunConfig(
        evals=configured_evals,
        output_dir=output_dir,
        container_runtime=runtime,
        server_port=port,
        eval_args=parsed_args,
    )

    console.print(
        Panel(
            Pretty(run_config, expand_all=True),
            title="[bold]Run Configuration[/bold]",
            border_style="cyan",
        )
    )

    if dry_run:
        console.print("\n[yellow]Dry run mode - not executing[/yellow]")
        return

    # Run evaluations
    console.print("\n[bold]Starting external evaluations...[/bold]")

    try:
        results = runner.run()
    except Exception as e:
        console.print(f"\n[bold red]Evaluation failed:[/bold red] {e}")
        console.print_exception()
        raise SystemExit(1) from None

    # Print summary
    console.print("\n[bold]Results Summary:[/bold]")

    results_table = Table()
    results_table.add_column("Evaluation", style="cyan")
    results_table.add_column("Status")
    results_table.add_column("Metrics")

    for name, result in results.items():
        if result.success:
            status = "[green]Success[/green]"
            # Format metrics vertically, one per line
            metrics_lines = []
            for k, v in result.metrics.items():
                if isinstance(v, float):
                    if v == int(v):
                        metrics_lines.append(f"{k}: [bold]{int(v)}[/bold]")
                    else:
                        metrics_lines.append(f"{k}: [bold]{v:.4f}[/bold]")
                else:
                    metrics_lines.append(f"{k}: [bold]{v}[/bold]")
            metrics = "\n".join(metrics_lines)
        else:
            status = "[red]Failed[/red]"
            metrics = result.error or "Unknown error"

        results_table.add_row(name, status, metrics)

    console.print(results_table)

    # Exit with error if any evaluation failed
    if any(not r.success for r in results.values()):
        raise SystemExit(1)
