"""Continuous-feed async dispatch for inference requests."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from olmo_eval.inference.errors import TerminalProviderError

logger = logging.getLogger(__name__)


@dataclass
class DispatchResult[R]:
    """Result from a dispatched task."""

    index: int
    result: R | None = None
    error: Exception | None = None

    @property
    def success(self) -> bool:
        return self.error is None


@dataclass
class DispatchStats:
    """Statistics from a dispatch run."""

    total: int = 0
    completed: int = 0
    failed: int = 0
    retried: int = 0


@dataclass
class ContinuousBatchDispatcher[T, R]:
    """Dispatch async tasks with continuous-feed pattern."""

    process_fn: Callable[[T], Awaitable[R]]
    max_in_flight: int = 256
    max_retries: int = 0
    on_progress: Callable[[int, int], None] | None = None
    on_result: Callable[[int, R | None, Exception | None], None] | None = None

    # Internal state (not part of init)
    _stats: DispatchStats = field(default_factory=DispatchStats, init=False)

    async def run(self, items: list[T]) -> list[R | None]:
        """Dispatch all items, returning None for failed requests."""
        return [result.result for result in await self.run_results(items)]

    async def run_results(self, items: list[T]) -> list[DispatchResult[R]]:
        """Dispatch all items and return results in original order.

        Args:
            items: List of items to process.

        Returns:
            Results and errors in the same order as input.
        """
        if not items:
            return []

        total = len(items)
        self._stats = DispatchStats(total=total)

        # Results array preserves ordering
        results: list[DispatchResult[R]] = [DispatchResult(index=i) for i in range(total)]

        # Queue of (index, item, attempt) tuples
        pending: list[tuple[int, T, int]] = [(i, item, 0) for i, item in enumerate(items)]
        pending_idx = 0  # Next item to dispatch

        # Track in-flight tasks: task -> (index, item, attempt)
        in_flight: dict[asyncio.Task, tuple[int, T, int]] = {}
        first_error: Exception | None = None

        while pending_idx < len(pending) or in_flight:
            # Top up in-flight set to target
            while len(in_flight) < self.max_in_flight and pending_idx < len(pending):
                idx, item, attempt = pending[pending_idx]
                pending_idx += 1

                task = asyncio.create_task(self._safe_process(item))
                in_flight[task] = (idx, item, attempt)

            if not in_flight:
                break

            # Wait for any one to complete
            done, _ = await asyncio.wait(in_flight.keys(), return_when=asyncio.FIRST_COMPLETED)

            for task in done:
                idx, item, attempt = in_flight.pop(task)
                try:
                    result, error = task.result()
                except TerminalProviderError:
                    for sibling in in_flight:
                        sibling.cancel()
                    await asyncio.gather(*in_flight, return_exceptions=True)
                    raise

                if error is not None and attempt < self.max_retries:
                    # Retry: add back to pending
                    pending.append((idx, item, attempt + 1))
                    self._stats.retried += 1
                else:
                    # Record result
                    results[idx] = DispatchResult(index=idx, result=result, error=error)
                    self._stats.completed += 1
                    if error is not None:
                        self._stats.failed += 1
                        if first_error is None:
                            first_error = error

                    # Callbacks
                    if self.on_result is not None:
                        self.on_result(idx, result, error)
                    if self.on_progress is not None:
                        self.on_progress(self._stats.completed, total)

        if first_error is not None and total > 1:
            logger.error(
                "dispatch: %d/%d requests failed; first error: %s: %s",
                self._stats.failed,
                total,
                type(first_error).__name__,
                first_error,
            )
        return results

    async def _safe_process(self, item: T) -> tuple[R | None, Exception | None]:
        """Process item, catching exceptions."""
        try:
            result = await self.process_fn(item)
            return (result, None)
        except TerminalProviderError:
            raise
        except Exception as e:
            logger.warning("dispatch: process_fn raised %s: %s", type(e).__name__, e)
            return (None, e)

    @property
    def stats(self) -> DispatchStats:
        """Get dispatch statistics."""
        return self._stats


async def dispatch_concurrent[T, R](
    items: list[T],
    process_fn: Callable[[T], Awaitable[R]],
    max_in_flight: int = 256,
    max_retries: int = 0,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[R | None]:
    """Convenience function for one-shot dispatch."""
    dispatcher: ContinuousBatchDispatcher[T, R] = ContinuousBatchDispatcher(
        process_fn=process_fn,
        max_in_flight=max_in_flight,
        max_retries=max_retries,
        on_progress=on_progress,
    )
    return await dispatcher.run(items)


async def dispatch_concurrent_results[T, R](
    items: list[T],
    process_fn: Callable[[T], Awaitable[R]],
    max_in_flight: int = 256,
    max_retries: int = 0,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[DispatchResult[R]]:
    """Dispatch requests while preserving each result and exception."""
    dispatcher: ContinuousBatchDispatcher[T, R] = ContinuousBatchDispatcher(
        process_fn=process_fn,
        max_in_flight=max_in_flight,
        max_retries=max_retries,
        on_progress=on_progress,
    )
    return await dispatcher.run_results(items)


__all__ = [
    "ContinuousBatchDispatcher",
    "DispatchResult",
    "DispatchStats",
    "dispatch_concurrent",
    "dispatch_concurrent_results",
]
