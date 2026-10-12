"""olmo-eval CLI entry point."""

# Suppress noisy third-party library output BEFORE any imports.
# Must be at the very top to take effect before transformers/datasets load.
import os

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_DATASETS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BAR", "1")

import click
from rich.table import Table

import olmo_eval.evals  # noqa: F401 - triggers suite registration
import olmo_eval.evals.tasks  # noqa: F401 - triggers task registration
from olmo_eval.cli.beaker import beaker
from olmo_eval.cli.results import results
from olmo_eval.cli.run import run
from olmo_eval.cli.run_external import run_external
from olmo_eval.cli.suite import suite
from olmo_eval.cli.task import task
from olmo_eval.cli.utils import console
from olmo_eval.common.constants import get_model_presets
from olmo_eval.evals.suites import get_suite, list_suites
from olmo_eval.evals.tasks.common import list_tasks, list_variants


@click.group()
def main() -> None:
    """olmo-eval command line interface."""
    from olmo_eval.upload.auth import take_upload_credentials_from_env

    take_upload_credentials_from_env()


# Register command groups
main.add_command(run)
main.add_command(beaker)
main.add_command(results)
main.add_command(task)
main.add_command(suite)
main.add_command(run_external)


@main.command()
@click.option("--filter", "-f", default="", help="Filter by name substring")
def tasks(filter: str) -> None:
    """List all available tasks in the registry."""
    task_names = list_tasks()
    variants = list_variants()

    if not task_names:
        console.print("[dim]No tasks registered.[/dim]")
        return

    table = Table(title="Available Tasks")
    table.add_column("Task", style="cyan")
    table.add_column("Variants", style="green")

    for name in task_names:
        if filter.lower() in name.lower():
            task_variants = variants.get(name, [])
            variant_str = ", ".join(task_variants) if task_variants else "-"
            table.add_row(name, variant_str)

    console.print(table)


def _format_value(value: object) -> str:
    """Format a config value for display."""
    from dataclasses import fields, is_dataclass

    if value is None:
        return ""
    if is_dataclass(value):
        parts = []
        for f in fields(value):
            v = getattr(value, f.name)
            formatted = _format_value(v)
            if formatted:
                parts.append(f"{f.name}={formatted}")
        return "\n".join(parts) if parts else ""
    if isinstance(value, dict):
        if not value:
            return ""
        return "\n".join(f"{k}={v}" for k, v in value.items())
    if isinstance(value, (list, tuple)):
        if not value:
            return ""
        formatted_items = [_format_value(v) for v in value]
        return "\n".join(f for f in formatted_items if f)
    if isinstance(value, bool):
        return str(value) if value else ""
    return str(value) if value else ""


@main.command()
@click.option("--filter", "-f", default="", help="Filter by name substring")
def models(filter: str) -> None:
    """List available model presets."""
    from dataclasses import fields

    from rich.panel import Panel

    presets = get_model_presets()
    if not presets:
        console.print("[dim]No model presets registered.[/dim]")
        return

    # Get field names from first config
    first_cfg = next(iter(presets.values()))
    field_names = [f.name for f in fields(first_cfg)]
    max_field_width = max(len(f) for f in field_names)

    for name, cfg in sorted(presets.items()):
        if filter.lower() in name.lower():
            table = Table(show_header=False, box=None, padding=(0, 1))
            table.add_column("Field", style="dim", width=max_field_width)
            table.add_column("Value", overflow="fold")

            for f in fields(cfg):
                value = getattr(cfg, f.name)
                formatted = _format_value(value)
                if formatted:
                    table.add_row(f.name, formatted)

            console.print(Panel(table, title=f"[cyan]{name}[/cyan]", title_align="left"))
            console.print()


@main.command()
@click.option("--filter", "-f", default="", help="Filter by name substring")
def harnesses(filter: str) -> None:
    """List available harness presets."""
    from rich.panel import Panel
    from rich.pretty import Pretty

    from olmo_eval.harness.presets import get_harness_preset, list_harness_presets

    preset_names = list_harness_presets()
    if not preset_names:
        console.print("[dim]No harness presets registered.[/dim]")
        return

    for name in sorted(preset_names):
        if filter.lower() not in name.lower():
            continue

        cfg = get_harness_preset(name)
        console.print(
            Panel(
                Pretty(cfg, expand_all=True),
                title=f"[bold]{name}[/bold]",
                border_style="cyan",
            )
        )
        console.print()


@main.command()
@click.option("--filter", "-f", default="", help="Filter by name substring")
def scaffolds(filter: str) -> None:
    """List available harness scaffolds."""
    from olmo_eval.harness.scaffolds import SCAFFOLD_REGISTRY

    table = Table(title="Harness Scaffolds")
    table.add_column("Scaffold", style="cyan")
    table.add_column("Required Extra", style="yellow")

    for name, cls in sorted(SCAFFOLD_REGISTRY.items()):
        if filter.lower() in name.lower():
            extras = ", ".join(cls.required_extras) if cls.required_extras else "-"
            table.add_row(name, extras)

    console.print(table)


@main.command()
@click.option("--filter", "-f", default="", help="Filter by name substring")
def suites(filter: str) -> None:
    """List available task suites (task groups)."""
    table = Table(title="Task Suites")
    table.add_column("Suite", style="cyan")
    table.add_column("Tasks", style="dim")
    table.add_column("Aggregation", style="yellow")

    for name in list_suites():
        if filter.lower() in name.lower():
            suite = get_suite(name)
            task_count = len(suite.expanded_tasks)
            table.add_row(name, f"{task_count} tasks", suite.aggregation.value)

    console.print(table)


@main.command(name="external-evals")
@click.option("--filter", "-f", default="", help="Filter by name substring")
def external_evals(filter: str) -> None:
    """List available external evaluations with usage information."""
    from rich.panel import Panel

    from olmo_eval.evals.external import get_external_eval, list_external_evals

    eval_names = list_external_evals()
    if not eval_names:
        console.print("[dim]No external evaluations registered.[/dim]")
        return

    for name in eval_names:
        if filter.lower() not in name.lower():
            continue

        eval_instance = get_external_eval(name)

        # Build details table
        details = Table(show_header=False, box=None, padding=(0, 1))
        details.add_column("Field", style="dim", width=16)
        details.add_column("Value", overflow="fold")

        from olmo_eval.evals.external import SandboxedExternalEval

        # Description
        details.add_row("Description", eval_instance.description)

        # Scaffold (if specified)
        if eval_instance.scaffold:
            details.add_row("Scaffold", f"[magenta]{eval_instance.scaffold}[/magenta]")

        # Sandbox info (only for sandboxed evals)
        if isinstance(eval_instance, SandboxedExternalEval):
            details.add_row("Image", f"[blue]{eval_instance.sandbox_image}[/blue]")
            details.add_row("Working Dir", eval_instance.working_dir)

        # Timeout
        timeout = eval_instance.timeout_seconds
        timeout_str = f"{timeout / 3600:.1f}h" if timeout >= 3600 else f"{timeout:.0f}s"
        details.add_row("Timeout", f"[yellow]{timeout_str}[/yellow]")

        # Arguments
        if eval_instance.arguments:
            args_lines = []
            for arg_name, (desc, default) in eval_instance.arguments.items():
                if default is not None:
                    args_lines.append(
                        f"[green]{arg_name}[/green]: {desc} [dim](default: {default})[/dim]"
                    )
                else:
                    args_lines.append(f"[green]{arg_name}[/green]: {desc} [dim](optional)[/dim]")
            details.add_row("Arguments", "\n".join(args_lines))

        # Required secrets
        if eval_instance.required_secrets:
            secrets_str = ", ".join(eval_instance.required_secrets)
            details.add_row("Required Secrets", f"[red]{secrets_str}[/red]")

        # Setup commands - only for sandboxed evals
        if isinstance(eval_instance, SandboxedExternalEval):
            setup_lines = "\n".join(
                f"[dim]{i}.[/dim] {cmd}" for i, cmd in enumerate(eval_instance.setup_command, 1)
            )
            details.add_row("Setup", setup_lines)

        # Run command - only show if non-empty (sandboxed evals with explicit command)
        run_cmd = eval_instance.run_command
        if run_cmd:
            # Break long commands at argument boundaries for readability
            run_cmd = run_cmd.replace(" --", " \\\n    --")
            details.add_row("Run", run_cmd)

        console.print(
            Panel(
                details,
                title=f"[bold cyan]{name}[/bold cyan]",
                title_align="left",
                border_style="cyan",
            )
        )
        console.print()


if __name__ == "__main__":
    main()
