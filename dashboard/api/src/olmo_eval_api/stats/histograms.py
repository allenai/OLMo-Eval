"""Histograms and box statistics (spec 5.10)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def histogram(values: Sequence[float] | np.ndarray, bins: int, *, binary: bool = False) -> dict:
    """Fixed-width bins over [min, max]; binary metrics get two bins centered on 0 and 1."""
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if binary:
        counts = [int((arr == 0).sum()), int((arr == 1).sum())]
        return {"edges": [-0.5, 0.5, 1.5], "counts": counts}
    if arr.size == 0:
        return {"edges": [], "counts": []}
    lo, hi = float(arr.min()), float(arr.max())
    if lo == hi:
        return {"edges": [lo - 0.5, hi + 0.5], "counts": [int(arr.size)]}
    counts, edges = np.histogram(arr, bins=bins, range=(lo, hi))
    return {"edges": [float(e) for e in edges], "counts": [int(c) for c in counts]}


def box_stats(values: Sequence[float] | np.ndarray) -> dict[str, Any] | None:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    p5, p25, p50, p75, p95 = np.quantile(arr, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {
        "p5": float(p5),
        "p25": float(p25),
        "p50": float(p50),
        "p75": float(p75),
        "p95": float(p95),
        "mean": float(arr.mean()),
        "n": int(arr.size),
    }
