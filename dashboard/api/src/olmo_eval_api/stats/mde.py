"""Minimum detectable effect at 80% power (spec 5.7)."""

from __future__ import annotations

from collections.abc import Iterable
from statistics import NormalDist, median

_NORMAL = NormalDist()


def z(q: float) -> float:
    return _NORMAL.inv_cdf(q)


def mde_factor(alpha: float) -> float:
    """z(1 - alpha/2) + z(0.8); 2.8016 for alpha = 0.05."""
    return z(1 - alpha / 2) + z(0.8)


def mde80(standard_errors: Iterable[float | None], alpha: float) -> float | None:
    ses = [se for se in standard_errors if se is not None and se == se]
    if not ses:
        return None
    return mde_factor(alpha) * median(ses)
