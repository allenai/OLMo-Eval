"""Credential management for Beaker launch."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

if TYPE_CHECKING:
    from olmo_eval.launch import BeakerLauncher

console = Console()


class CredentialManager:
    """Manages credential detection and setup for Beaker jobs."""

    def __init__(
        self,
        model_specs: list[str],
        aws_credentials: bool | None,
        gcs_credentials: bool | None,
    ):
        self.model_specs = model_specs
        self.aws_credentials = aws_credentials
        self.gcs_credentials = gcs_credentials

    def detect_and_setup(self, launcher: BeakerLauncher) -> tuple[bool, bool]:
        """Decide which credentials to inject into the job.

        Personal Google credentials are only injected for GCS model access, and only
        service-account keys. Results upload uses the shared uploader service account
        key instead (see ``olmo_eval.launch.beaker.uploader``).
        """
        from olmo_eval.launch.beaker.aws import is_s3_path
        from olmo_eval.launch.beaker.gcs import get_local_gcs_credentials, is_gcs_path

        s3_models = [m for m in self.model_specs if is_s3_path(m)]
        inject_aws = self.aws_credentials
        if inject_aws is None:
            inject_aws = bool(s3_models)

        gcs_models = [m for m in self.model_specs if is_gcs_path(m)]
        inject_gcs = self.gcs_credentials
        if inject_gcs is None:
            inject_gcs = bool(gcs_models)

        if inject_gcs:
            self._display_gcs_info(launcher, get_local_gcs_credentials())

        return inject_aws, inject_gcs

    def _display_gcs_info(self, launcher: Any, local_gcs_creds: Any) -> None:
        beaker_user = launcher.beaker.user_name

        gcs_table = Table(show_header=False, box=None, expand=True)
        gcs_table.add_column("Key", style="blue")
        gcs_table.add_column("Value")

        if local_gcs_creds:
            gcs_table.add_row("Credentials", "[green]found[/green] (service account)")
            if local_gcs_creds.client_email:
                gcs_table.add_row("Service account", local_gcs_creds.client_email)
            if local_gcs_creds.project_id:
                gcs_table.add_row("Project", local_gcs_creds.project_id)
            gcs_table.add_row("Beaker user", beaker_user)
            gcs_table.add_row("Beaker secret", f"{beaker_user}_GOOGLE_CREDENTIALS")
        else:
            gcs_table.add_row(
                "Credentials",
                "[yellow]not found[/yellow] - job may fail if GCS access is required",
            )

        console.print()
        console.print(
            Panel(
                gcs_table,
                title="[bold]GCS Access Configuration[/bold]",
                border_style="magenta",
                expand=True,
            )
        )
        console.print()

    def display_aws_info(self, launcher: Any, inject_aws: bool) -> None:
        from olmo_eval.launch.beaker.aws import get_local_aws_credentials

        if not inject_aws:
            return

        lines = ["[bold]S3 Access:[/bold]"]
        local_creds = get_local_aws_credentials()
        beaker_user = launcher.beaker.user_name
        if local_creds:
            cred_type = "temporary" if local_creds.session_token else "long-term"
            lines.append(f"  Credentials: [green]found[/green] ({cred_type})")
            lines.append(
                f"  Beaker secrets: {beaker_user}_AWS_ACCESS_KEY_ID, "
                f"{beaker_user}_AWS_SECRET_ACCESS_KEY"
            )
        else:
            lines.append(
                "  Credentials: [yellow]not found[/yellow] - job may fail if S3 access is required"
            )

        console.print(
            Panel(
                "\n".join(lines),
                title="[bold]AWS Credentials[/bold]",
                border_style="green",
                expand=True,
            )
        )
        console.print()
