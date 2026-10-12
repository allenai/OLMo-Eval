"""Worker processes for evaluation runners."""

from __future__ import annotations

import asyncio
import multiprocessing as mp
import os
import time
from typing import Any

from olmo_eval.common.logging import get_logger
from olmo_eval.inference.gpu_planner import resolve_visible_devices
from olmo_eval.runners.asynq.types import (
    WORKER_FATAL,
    ResultItem,
)

logger = get_logger(__name__)


def inference_worker(
    worker_id: str,
    gpu_ids: list[int],
    item_queue: mp.Queue,
    result_queue: mp.Queue,
    harness_config_dict: dict[str, Any],
    total_instances: int,
    init_queue: mp.Queue | None = None,
    output_dir: str | None = None,
    num_workers: int = 1,
    start_event: Any | None = None,
) -> None:
    """Worker process that initializes a harness and processes items.

    Processes items in streaming chunks for balanced latency and throughput.
    COMPLETION/LOGLIKELIHOOD requests are batched, CHAT requests use async
    concurrency.

    Args:
        worker_id: Unique worker identifier.
        gpu_ids: GPU IDs to use (sets CUDA_VISIBLE_DEVICES).
        item_queue: Queue of QueueItems (None signals shutdown).
        result_queue: Queue to put ResultItems.
        harness_config_dict: Serialized HarnessConfig.
        total_instances: Total number of instances across all workers.
        init_queue: Optional queue for reporting initialization times.
        output_dir: Output directory for persisting logs (e.g., vLLM server logs).
        num_workers: Number of parallel workers sharing the work.
        start_event: Optional event used to hold all ready workers at a start gate.
    """
    import sys

    from olmo_eval.common.logging import configure_logging, configure_worker_logging

    configure_logging()

    worker_logger = configure_worker_logging(worker_id)

    from olmo_eval.harness import Harness, HarnessConfig

    harness_config = HarnessConfig.from_dict(harness_config_dict)
    provider_config = harness_config.provider
    model_name = provider_config.model

    try:
        if gpu_ids:
            os.environ["CUDA_VISIBLE_DEVICES"] = resolve_visible_devices(gpu_ids)

        provider_kind = str(provider_config.kind)
        tokenizer = provider_config.tokenizer
        max_model_len = provider_config.max_model_len
        max_concurrency = provider_config.max_concurrency or harness_config.max_concurrency
        provider_kwargs = dict(provider_config.kwargs) if provider_config.kwargs else {}

        load_format = provider_kwargs.get("load_format")
        extra_loader_config = provider_kwargs.get("model_loader_extra_config")

        from olmo_eval.launch.config import get_model_short_name

        short_name = get_model_short_name(model_name)
        worker_logger.info(f"Initializing provider: {provider_kind}")
        worker_logger.info(f"  Model: {short_name}")
        if tokenizer:
            worker_logger.info(f"  Tokenizer: {tokenizer}")
        if gpu_ids:
            worker_logger.info(f"  GPUs: {gpu_ids}")

        init_start = time.time()

        has_tools = harness_config.has_tools
        enable_auto_tool_choice = has_tools and provider_kind == "vllm_server"

        # Set log_dir for vllm_server provider - matches metrics naming convention
        log_dir = None
        if provider_kind == "vllm_server" and output_dir:
            safe_model = model_name.replace("/", "_").replace("\\", "_")
            log_dir = os.path.join(output_dir, "logs", f"vllm_server_{safe_model}")

        # Only inject vllm-specific kwargs for vllm providers
        vllm_only_overrides: dict[str, Any] = {}
        if provider_kind in ("vllm", "vllm_server"):
            vllm_only_overrides = dict(
                tensor_parallel_size=len(gpu_ids) if gpu_ids else None,
                load_format=load_format,
                model_loader_extra_config=extra_loader_config,
                enable_auto_tool_choice=enable_auto_tool_choice or None,
                log_dir=log_dir,
            )

        harness_config = harness_config.with_provider_overrides(
            max_model_len=max_model_len,
            max_concurrency=max_concurrency,
            tokenizer=tokenizer,
            **vllm_only_overrides,
        )

        # Update metrics config with runtime values (output_dir, provider_kind, model_name)
        if harness_config.metrics is not None and harness_config.metrics.enabled:
            updated_metrics = harness_config.metrics.with_output_dir(
                output_dir or ""
            ).with_metadata(
                provider_kind=provider_kind,
                model_name=model_name,
            )
            harness_config = harness_config.with_metrics(updated_metrics)

        harness = Harness(harness_config)

        # Force provider creation to catch import errors early
        _ = harness.provider

        # Validate scaffold requirements early to fail fast
        if harness_config.scaffold:
            from olmo_eval.harness.scaffolds import validate_scaffold

            validate_scaffold(harness_config.scaffold)

        # Initialize metrics reporters early so setup errors surface before inference
        harness.initialize_reporters()

        init_time = time.time() - init_start
        worker_logger.info(f"Provider ready ({init_time:.1f}s)")

        try:
            # Configure agent trace output if using the openai_agents scaffold
            if harness_config.scaffold == "openai_agents" and output_dir:
                from olmo_eval.harness.scaffolds.tracing import configure_trace_output

                configure_trace_output(output_dir)

            # Initialize scaffold resources (e.g., sandbox manager) before processing
            if harness_config.scaffold:
                asyncio.run(harness.scaffold.initialize(harness_config))

            # Get batching strategy from config
            from olmo_eval.runners.asynq.batching import BatchConfig, get_strategy

            batch_config = harness_config.batching or BatchConfig()
            batch_config.validate_for_provider(provider_kind)
            strategy = get_strategy(batch_config)

            concurrency_str = max_concurrency or "unlimited"
            if batch_config.strategy == "streaming":
                worker_logger.info(
                    f"Inference worker ready (strategy={batch_config.strategy}, "
                    f"max_in_flight={concurrency_str})"
                )
            else:
                worker_logger.info(
                    f"Inference worker ready (strategy={batch_config.strategy}, "
                    f"chunk_size={batch_config.chunk_size})"
                )

            if init_queue is not None:
                init_queue.put((worker_id, init_time))

            if start_event is not None:
                start_event.wait()

            asyncio.run(
                strategy.run(
                    item_queue,
                    harness,
                    result_queue,
                    max_concurrency,
                    worker_logger,
                    total_instances,
                    num_workers,
                )
            )

            worker_logger.info("Processing complete")
        finally:
            # Clean up harness resources (including sandbox manager)
            asyncio.run(harness.cleanup())
            # Clean up provider
            close_fn = getattr(harness.provider, "close", None)
            if callable(close_fn):
                close_fn()

    except Exception as e:
        import traceback

        worker_logger.error(f"Worker process failed: {e}")
        worker_logger.error(traceback.format_exc())
        result_queue.put(
            ResultItem(
                model_name=model_name,
                task_id=WORKER_FATAL,
                instance_idx=-1,
                instance=None,
                request=None,
                outputs=[],
                error=f"Worker process crashed: {e}",
            )
        )
        sys.exit(1)


__all__ = [
    "inference_worker",
]
