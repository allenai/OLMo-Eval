"""Runner for external black-box evaluations."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

from olmo_eval.common.constants.infrastructure import BEAKER_RESULT_DIR
from olmo_eval.common.types import compute_model_hash, compute_task_hash
from olmo_eval.evals.external import (
    ExternalEvalResult,
    get_external_eval,
    list_external_evals,
)
from olmo_eval.inference.metrics import (
    InstrumentedProvider,
    MetricsConfig,
    VLLMMetricsMonitor,
)
from olmo_eval.inference.providers.config import ProviderConfig
from olmo_eval.runners.processing.utils import generate_experiment_id

if TYPE_CHECKING:
    from olmo_eval.upload import UploadConfig

logger = logging.getLogger(__name__)


@dataclass
class ExternalEvalRunner:
    """Runner for executing external black-box evaluations.

    Runs external evaluations in sandbox containers against a model provider.
    For local providers (vllm_server), starts a vLLM server. For remote
    providers, connects to the existing endpoint.

    Attributes:
        provider_config: Configuration for the inference provider.
        external_eval_names: Names of external evaluations to run.
        output_dir: Directory to write results.
        container_runtime: Container runtime to use (docker or podman).
        server_port: Port for the vLLM server (when using local provider).
        eval_args: Arguments to pass to external evaluations.
        upload_config: Dashboard upload settings; None skips the upload.
        experiment_name: Human-readable experiment name.
        experiment_group: Experiment group for grouping related experiments.
        metrics: Configuration for metrics collection on the provider.
    """

    provider_config: ProviderConfig
    external_eval_names: list[str] = field(default_factory=list)
    output_dir: str = BEAKER_RESULT_DIR
    container_runtime: str = "podman"
    server_port: int = 8000
    eval_args: dict[str, Any] = field(default_factory=dict)
    upload_config: UploadConfig | None = None
    experiment_name: str | None = None
    experiment_group: str | None = None
    metrics: MetricsConfig | None = None

    def validate(self) -> None:
        """Validate runner configuration.

        Raises:
            ValueError: If configuration is invalid.
        """
        if not self.provider_config.model:
            raise ValueError("provider_config.model is required")

        if not self.external_eval_names:
            raise ValueError("At least one external_eval_name is required")

        # Validate that all external evals exist
        available = set(list_external_evals())
        for name in self.external_eval_names:
            if name not in available:
                raise ValueError(
                    f"External eval '{name}' not found. Available: {', '.join(sorted(available))}"
                )

    def run(self) -> dict[str, ExternalEvalResult]:
        """Sync wrapper for async execution."""
        return asyncio.run(self.run_async())

    async def run_async(self) -> dict[str, ExternalEvalResult]:
        """Execute all external evaluations.

        Returns:
            Dictionary mapping evaluation names to results.
        """
        start_time = time.time()
        results: dict[str, ExternalEvalResult] = {}

        # Start a vLLM server only if provider is vllm/vllm_server without existing base_url
        server_process = None
        base_url = self.provider_config.base_url

        needs_server = (
            self.provider_config.kind in ("vllm", "vllm_server")
            and not self.provider_config.base_url
        )
        if needs_server:
            server_process = self._start_server()
            if server_process is None:
                for name in self.external_eval_names:
                    results[name] = ExternalEvalResult.from_error(
                        name, "Failed to start vLLM server"
                    )
                self._save_results(results, time.time() - start_time)
                return results
            base_url = server_process.base_url

        # Create provider once for all evals
        if server_process is not None:
            # Use the provider we already started
            provider = server_process
        else:
            # Create provider from config with the base_url
            provider = self.provider_config.with_overrides(base_url=base_url).create_provider()

        # Wrap with instrumentation if metrics enabled
        if self.metrics is not None and self.metrics.enabled:
            instrumented = InstrumentedProvider(provider)
            if self.metrics.collect_gpu:
                instrumented.enable_gpu_monitoring(interval_s=1.0)
            provider = instrumented

        # Start vLLM server metrics monitoring if enabled and we have a vLLM server
        vllm_monitor: VLLMMetricsMonitor | None = None
        is_vllm_provider = self.provider_config.kind in ("vllm", "vllm_server")
        if (
            self.metrics is not None
            and self.metrics.enabled
            and self.metrics.collect_vllm_server
            and is_vllm_provider
            and base_url
        ):
            vllm_metrics_path = os.path.join(
                self.output_dir, "metrics", "vllm_server_metrics.jsonl"
            )
            vllm_monitor = VLLMMetricsMonitor(
                base_url=base_url,
                output_path=vllm_metrics_path,
                interval_s=self.metrics.vllm_poll_interval_s,
            )
            vllm_monitor.start()

        try:
            for eval_name in self.external_eval_names:
                logger.info(f"Running external evaluation: {eval_name}")

                try:
                    external_eval = get_external_eval(eval_name)
                    result = await external_eval.execute_with_provider(
                        provider=provider,  # type: ignore[ty:invalid-argument-type]
                        args=self.eval_args,
                        output_dir=self.output_dir,
                        container_runtime=self.container_runtime,
                    )
                    results[eval_name] = result

                    if result.success:
                        logger.info(f"[{eval_name}] Completed successfully")
                        for metric, value in result.metrics.items():
                            logger.info(f"  {metric}: {value}")
                    else:
                        logger.error(f"[{eval_name}] Failed: {result.error}")

                except Exception as e:
                    logger.exception(f"[{eval_name}] Unexpected error")
                    results[eval_name] = ExternalEvalResult.from_error(eval_name, str(e))

        finally:
            # Stop vLLM server metrics monitoring
            if vllm_monitor is not None:
                vllm_snapshots = vllm_monitor.stop()
                if vllm_snapshots:
                    logger.info(f"Collected {len(vllm_snapshots)} vLLM server metrics snapshots")

            # Disable GPU monitoring if enabled
            if isinstance(provider, InstrumentedProvider):
                provider.disable_gpu_monitoring()

            # Stop the server if we started it
            if server_process is not None:
                self._stop_server(server_process)

        # Calculate total duration
        total_duration = time.time() - start_time

        # Save combined results
        self._save_results(results, total_duration)

        return results

    def _start_server(self) -> Any | None:
        """Start the vLLM server.

        Returns:
            The provider instance (which manages the server) or None if failed.
        """
        try:
            # Set log_dir to persist vLLM server logs
            log_dir = os.path.join(self.output_dir, "logs")

            # Create provider config for the server without a base_url
            # This causes VLLMServerProvider to start its own server
            # Enable auto tool choice by default for external evals (e.g., tau2_bench)
            server_config = self.provider_config.with_overrides(
                kind="vllm_server",
                base_url=None,
                log_dir=log_dir,
                enable_auto_tool_choice=True,
            )

            # Create the provider - this starts the server automatically
            provider = server_config.create_provider()
            if hasattr(provider, "base_url"):
                logger.info(f"Provider ready at {provider.base_url}")
            return provider

        except Exception as e:
            logger.error(f"Failed to start vLLM server: {e}")

        return None

    def _stop_server(self, server: Any) -> None:
        """Stop the vLLM server.

        Args:
            server: Provider instance that manages the server.
        """
        logger.info("Stopping vLLM server")

        try:
            if hasattr(server, "close"):
                server.close()
        except Exception as e:
            logger.warning(f"Error stopping server: {e}")

    def _save_results(
        self,
        results: dict[str, ExternalEvalResult],
        total_duration: float,
    ) -> None:
        """Write metrics.json, predictions and the manifest, then upload the results.

        Args:
            results: Dictionary of evaluation results.
            total_duration: Total time taken for all evaluations.
        """
        output_path = Path(self.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Generate experiment tracking identifiers
        experiment_id = generate_experiment_id()
        model_config = self.provider_config.to_dict()
        model_hash = compute_model_hash(model_config)
        finished = datetime.now(UTC)
        timestamp = finished.isoformat()

        # Each external eval becomes a "task" with its metrics
        runner_results: dict[str, Any] = {
            "model": self.provider_config.alias or self.provider_config.model,
            "model_path": self.provider_config.model,
            "provider": str(self.provider_config.kind),
            "timestamp": timestamp,
            "model_config": model_config,
            "tasks": {},
        }

        for eval_name, result in results.items():
            # Convert flat metrics to nested format (metric_name: {scorer: value})
            # External evals have simple metrics like "pass^1", use "external" as scorer
            nested_metrics: dict[str, dict[str, float]] = {}
            for metric_name, value in result.metrics.items():
                nested_metrics[metric_name] = {"external": value}

            # Build task config from eval args
            task_config = {"eval_name": eval_name, **self.eval_args}
            task_hash = compute_task_hash(task_config) or "unknown"

            # Determine primary metric (first metric if any exist)
            primary_metric = None
            if result.metrics:
                first_metric = next(iter(result.metrics.keys()))
                primary_metric = f"{first_metric}:external"

            runner_results["tasks"][eval_name] = {
                "metrics": nested_metrics,
                "task_hash": task_hash,
                "config": task_config,
                "num_instances": result.num_instances,
                "primary_metric": primary_metric,
                "duration_seconds": result.duration_seconds,
                "success": result.success,
                "error": result.error,
                "raw_output": result.raw_output,
                "predictions": result.predictions,
            }

        # Write metrics.json in standard runner format
        self._write_metrics_json(runner_results, experiment_id, total_duration)

        self._write_predictions(runner_results)
        self._write_manifest(
            runner_results,
            experiment_id=experiment_id,
            model_hash=model_hash,
            started_at=(finished - timedelta(seconds=total_duration)).isoformat(),
            finished_at=timestamp,
            total_duration=total_duration,
        )
        if self.upload_config is not None and self.upload_config.enabled:
            from olmo_eval.upload import upload_results_dir

            upload_results_dir(self.output_dir, self.upload_config)

    def _write_predictions(self, runner_results: dict[str, Any]) -> None:
        """Write each external eval's predictions so they can be uploaded and browsed."""
        from olmo_eval.runners.io.writers import write_predictions_jsonl

        for eval_name, task_data in runner_results["tasks"].items():
            predictions = task_data.get("predictions")
            if not predictions:
                continue
            try:
                write_predictions_jsonl(
                    self.output_dir,
                    eval_name,
                    predictions,
                    self.provider_config.model,
                    task_hash=task_data.get("task_hash"),
                )
            except (OSError, TypeError, ValueError) as e:
                logger.warning(f"Could not write predictions for {eval_name}: {e}")

    def _write_manifest(
        self,
        runner_results: dict[str, Any],
        experiment_id: str,
        model_hash: str | None,
        started_at: str,
        finished_at: str,
        total_duration: float,
    ) -> None:
        """Write manifest.json describing this run for the uploader."""
        from olmo_eval.upload.manifest import (
            build_manifest,
            build_model_info,
            build_run_info,
            build_task_entry,
            run_status,
            write_manifest,
        )
        from olmo_eval.upload.metadata import metric_meta_from_metrics

        try:
            tasks = {
                name: build_task_entry(
                    name,
                    metric_meta_from_metrics((), list(data["metrics"])),
                    data.get("error") if not data.get("success", True) else None,
                )
                for name, data in runner_results["tasks"].items()
            }
            errors = [
                {"task": name, "error": t["error"]} for name, t in tasks.items() if t["error"]
            ]
            run = build_run_info(
                run_id=experiment_id,
                status=run_status({name: t["error"] for name, t in tasks.items()}),
                started_at=started_at,
                finished_at=finished_at,
                duration_seconds=total_duration,
                experiment_name=self.experiment_name,
                experiment_group=self.experiment_group,
                task_specs=self.external_eval_names,
                output_dir=self.output_dir,
                errors=errors,
                tags=self.upload_config.tags if self.upload_config else (),
            )
            model = build_model_info(
                name=runner_results["model"],
                path=runner_results["model_path"],
                model_hash=model_hash,
                revision=self.provider_config.revision,
                provider_kind=runner_results["provider"],
                provider_config=runner_results["model_config"],
            )
            write_manifest(self.output_dir, build_manifest(run, model, tasks))
        except Exception as e:
            logger.warning(f"Could not write the run manifest: {e}")

    def _write_metrics_json(
        self,
        runner_results: dict[str, Any],
        experiment_id: str,
        total_duration: float,
    ) -> None:
        """Write metrics.json in standard runner format."""
        from olmo_eval.runners.common.models import MetricsOutput, TaskMetricsEntry

        tasks_output: list[dict[str, Any]] = []
        summary: dict[str, Any] = {}

        for task_name, task_data in runner_results["tasks"].items():
            entry = TaskMetricsEntry(
                task=task_name,
                metrics=task_data["metrics"],
                num_instances=task_data.get("num_instances") or 0,
                model=runner_results["model"],
                primary_metric=task_data.get("primary_metric"),
                config=task_data.get("config"),
                duration_seconds=task_data.get("duration_seconds"),
                task_hash=task_data.get("task_hash"),
            )
            tasks_output.append(entry.to_dict())

            # Add to summary
            if task_data.get("primary_metric") and task_data["metrics"]:
                metric_name = task_data["primary_metric"].split(":")[0]
                if metric_name in task_data["metrics"]:
                    score = task_data["metrics"][metric_name].get("external", 0.0)
                    summary[task_name] = {
                        "metric": task_data["primary_metric"],
                        "score": score,
                    }

        metrics_output = MetricsOutput(
            timestamp=runner_results["timestamp"],
            config=runner_results["model_config"],
            tasks=tasks_output,
            summary=summary,
            experiment_id=experiment_id,
            experiment_name=self.experiment_name,
            experiment_group=self.experiment_group,
            experiment_duration_seconds=total_duration,
        )

        metrics_file = Path(self.output_dir) / "metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(metrics_output.to_dict(), f, indent=2)
        logger.info(f"Metrics saved to {metrics_file}")
