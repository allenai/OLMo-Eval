"""Fair-share attribution of wall time across concurrent units of work."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


class ShareClock:
    """Splits wall time evenly across the units of work in flight at each moment.

    A unit calls ``start()`` when it begins and ``stop(token)`` when it ends. While k
    units are in flight, each second of wall time is credited 1/k to each of them, so
    the shares of all units sum to the time at least one unit was in flight. Time with
    nothing in flight is credited to no one.

    The clock keeps the integral of 1/k(t) dt; a unit's share is the integral's growth
    between its start and stop. It has no lock and is meant for units on one event loop.
    """

    def __init__(self, now: Callable[[], float] = time.monotonic) -> None:
        self._now = now
        self._shared = 0.0
        self._active = 0
        self._last = now()

    def _advance(self) -> None:
        t = self._now()
        if self._active:
            self._shared += (t - self._last) / self._active
        self._last = t

    def start(self) -> float:
        """Begin a unit; returns the token to pass to ``stop``."""
        self._advance()
        self._active += 1
        return self._shared

    def stop(self, token: float) -> float:
        """End the unit begun with ``token``; returns its share in seconds."""
        self._advance()
        self._active -= 1
        return self._shared - token

    def exclude(self, fn: Callable[[], T]) -> tuple[T, float]:
        """Run ``fn``, which blocks the event loop, and keep its time out of every share.

        Units in flight make no progress while ``fn`` runs, so its span is credited to
        none of them. Returns ``fn``'s result and its elapsed seconds, for the caller to
        credit where it belongs.
        """
        self._advance()
        started = self._last
        try:
            return fn(), self._now() - started
        finally:
            self._last = self._now()


__all__ = ["ShareClock"]
