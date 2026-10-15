"""olmo-eval results: upload a local results directory to the dashboard."""

from __future__ import annotations

import functools

import click

from olmo_eval.cli.utils import console

# Plain numbers and paths: no automatic highlighting.
echo = functools.partial(console.print, highlight=False)


@click.group()
def results() -> None:
    """Work with saved evaluation results."""


def _format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024 or unit == "GiB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{size} B"


@results.command()
@click.argument("directory", type=click.Path(exists=True, file_okay=False))
@click.option("--api-url", default=None, help="Dashboard ingest service URL")
@click.option("--tag", "tags", multiple=True, help="Label to attach to the run (repeatable)")
@click.option(
    "--dry-run",
    is_flag=True,
    help="Build and validate every payload without sending anything",
)
def upload(directory: str, api_url: str | None, tags: tuple[str, ...], dry_run: bool) -> None:
    """Upload a results directory (the -O of a run) to the dashboard.

    Re-uploading the same directory updates the same run. For a Beaker job, fetch
    the result dataset first:

        beaker dataset fetch <dataset-id> -o ./results

        olmo-eval results upload ./results
    """
    from olmo_eval.common.logging import configure_logging
    from olmo_eval.upload import resolve_upload_config, upload_results_dir
    from olmo_eval.upload.config import upload_param_hint
    from olmo_eval.upload.manifest import ManifestError
    from olmo_eval.upload.uploader import dry_run as run_dry_run

    configure_logging(level="INFO")
    try:
        config = resolve_upload_config(True, api_url, tags)
    except ValueError as e:
        raise click.BadParameter(str(e), param_hint=upload_param_hint(e)) from None

    if dry_run:
        try:
            report = run_dry_run(directory, config.tags)
        except ManifestError as e:
            echo(f"[red]Error:[/red] {e}")
            raise SystemExit(1) from None
        plan = report.plan
        echo(f"Run: [cyan]{plan.run_id}[/cyan] ({plan.complete['status']})")
        echo(f"Tasks: {len(plan.tasks)}")
        echo(f"Instances: {report.instances}")
        echo(f"Artifacts: {len(plan.artifacts)} ({_format_bytes(plan.artifact_bytes)})")
        echo(f"Inference metrics: {'yes' if plan.inference else 'no'}")
        echo(f"Suites: {len(plan.complete['suites'])}")
        checked = "contract schema" if report.used_schema else "required keys only"
        for warning in plan.warnings:
            echo(f"[yellow]Warning:[/yellow] {warning}")
        if report.degraded_reason:
            echo(
                "[yellow]Warning:[/yellow] Checked required keys only, not the full "
                f"contract schema: {report.degraded_reason}"
            )
        if report.errors:
            echo(f"[red]{len(report.errors)} validation error(s) ({checked}):[/red]")
            for error in report.errors[:50]:
                echo(f"  {error}")
            raise SystemExit(1)
        echo(f"[green]All payloads are valid[/green] ({checked})")
        return

    result = upload_results_dir(directory, config)
    if not result.ok:
        echo(f"[red]Upload failed:[/red] {result.error}")
        raise SystemExit(1)
    echo(f"[green]Uploaded run {result.run_id}[/green]")
    if result.dashboard_url:
        echo(result.dashboard_url)
