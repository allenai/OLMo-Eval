"""Batched strategy - process items in chunks with continuous dispatch."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import multiprocessing as mp

    from olmo_eval.harness import Harness
    from olmo_eval.runners.asynq.types import QueueItem, ResultItem

from .base import BatchingStrategy


def _compute_batch_hash(items: list[QueueItem]) -> str:
    """Compute a short hash for a batch of items."""
    from olmo_eval.inference.metrics.core.stats import compute_batch_hash

    native_ids = [f"{item.task_id}:{item.instance_idx}" for item in items]
    return compute_batch_hash(native_ids)


class BatchedStrategy(BatchingStrategy):
    """Process items in chunks with continuous dispatch within each chunk."""

    async def run(
        self,
        item_queue: mp.Queue[QueueItem | None],
        harness: Harness,
        result_queue: mp.Queue[ResultItem],
        max_concurrency: int | None,
        worker_logger: logging.Logger,
        total_instances: int,
        num_workers: int = 1,
    ) -> None:
        """Execute batched processing."""
        import math

        from olmo_eval.runners.asynq.processing import process_items

        worker_instances = math.ceil(total_instances / num_workers)
        total_batches = math.ceil(worker_instances / self.config.chunk_size)
        current_batch = 0

        while True:
            # Collect batch
            batch, saw_shutdown = await self.collect_batch(item_queue)

            if not batch and saw_shutdown:
                # Empty batch with shutdown signal - we're done
                return

            if batch:
                current_batch += 1
                batch_hash = _compute_batch_hash(batch)
                batch_size = len(batch)

                worker_logger.info(
                    f"Starting batch {current_batch}/{total_batches} ({batch_hash}) "
                    f"with {batch_size} item(s)"
                )
                start_time = time.perf_counter()
                await process_items(
                    batch,
                    harness,
                    result_queue,
                    max_concurrency,
                    worker_logger,
                    clock=self.clock,
                )

                elapsed = time.perf_counter() - start_time
                rate = batch_size / elapsed if elapsed > 0 else 0
                worker_logger.info(
                    f"Processed batch {current_batch}/{total_batches} ({batch_hash}) "
                    f"in {elapsed:.1f}s ({rate:.1f} items/sec)"
                )

            if saw_shutdown:
                # Processed final batch
                return
