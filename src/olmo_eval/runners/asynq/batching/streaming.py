"""Streaming strategy - send items directly to provider."""

from __future__ import annotations

import asyncio
import logging
import queue
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import multiprocessing as mp

    from olmo_eval.harness import Harness
    from olmo_eval.runners.asynq.types import QueueItem, ResultItem

from .base import BatchingStrategy


class StreamingStrategy(BatchingStrategy):
    """Stream items directly to the provider with no explicit batching.

    Items are sent to the provider as soon as they arrive. Used for
    providers like LLM() that handle their own internal batching.
    """

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
        """Execute streaming to provider."""
        import math

        from olmo_eval.common.progress import ProgressLogger
        from olmo_eval.runners.asynq.processing import process_items

        concurrency = max_concurrency or 64
        semaphore = asyncio.Semaphore(concurrency)
        in_flight: set[asyncio.Task[None]] = set()
        failure: BaseException | None = None

        worker_instances = math.ceil(total_instances / num_workers)
        progress = ProgressLogger(
            total=worker_instances,
            desc="Processed",
            logger=worker_logger,
            color="green",
        )

        async def process_single(item: QueueItem) -> None:
            async with semaphore:
                await process_items(
                    [item], harness, result_queue, 1, worker_logger, show_progress=False
                )
                progress.update(1)

        def task_done(task: asyncio.Task[None]) -> None:
            nonlocal failure
            in_flight.discard(task)
            if not task.cancelled():
                error = task.exception()
                if failure is None:
                    failure = error

        async def get_item() -> QueueItem | None:
            """Get next item from queue asynchronously."""
            loop = asyncio.get_event_loop()
            while True:
                try:
                    return await loop.run_in_executor(None, lambda: item_queue.get(timeout=0.1))
                except queue.Empty:
                    pass
                if failure is not None:
                    raise failure
                if not in_flight:
                    try:
                        return await loop.run_in_executor(None, lambda: item_queue.get(timeout=1.0))
                    except queue.Empty:
                        return None
                await asyncio.sleep(0.01)

        try:
            while True:
                item = await get_item()
                if failure is not None:
                    raise failure

                if item is None:
                    break

                task = asyncio.create_task(process_single(item))
                in_flight.add(task)
                task.add_done_callback(task_done)

            if in_flight:
                await asyncio.gather(*in_flight)
            if failure is not None:
                raise failure
        finally:
            for task in in_flight:
                task.cancel()
            if in_flight:
                await asyncio.gather(*in_flight, return_exceptions=True)

            progress.close()
