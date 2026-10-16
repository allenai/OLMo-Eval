"""Runner factory for creating evaluation runners."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from olmo_eval.cli.run.config import RunConfig

if TYPE_CHECKING:
    from olmo_eval.upload import UploadConfig


class RunnerFactory:
    """Factory for creating evaluation runners based on configuration."""

    def __init__(self, config: RunConfig, upload_config: UploadConfig | None = None):
        """Initialize the factory.

        Args:
            config: Parsed run configuration.
            upload_config: Dashboard upload settings, or None to skip uploads.
        """
        self.config = config
        self.upload_config = upload_config

    def create(self) -> Any:
        """Create the runner based on configuration.

        Returns:
            Configured AsyncEvalRunner instance.
        """
        from olmo_eval.runners.asynq.runner import AsyncEvalRunner

        # Get attention_backend from provider kwargs if specified
        provider_kwargs = self.config.harness_config.provider.kwargs
        attention_backend = provider_kwargs.get("attention_backend") if provider_kwargs else None

        return AsyncEvalRunner(
            harness_config=self.config.harness_config,
            task_specs=self.config.task_specs,
            output_dir=self.config.output_dir,
            attention_backend=attention_backend,
            task_overrides=self.config.task_overrides,
            upload_config=self.upload_config,
            experiment_name=self.config.experiment_name,
            experiment_group=self.config.experiment_group,
            save_predictions=self.config.save_predictions,
            save_requests=self.config.save_requests,
            inspect_instance=self.config.inspect_instance,
            inspect_formatted=self.config.inspect_formatted,
            inspect_tokens=self.config.inspect_tokens,
            inspect_response=self.config.inspect_response,
            inspect_request=self.config.inspect_request,
        )
