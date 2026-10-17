"""Fair-share attribution of wall time across concurrent units of work."""

from __future__ import annotations

import time
from collections.abc import Callable


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


__all__ = ["ShareClock"]
