"""Poisson bootstrap with counter-based weights (spec 5.3).

Resample ``b`` gives instance ``i`` the weight ``W[b, i] ~ Poisson(1)`` computed from
``(seed, task_name, b, key_hash_i)`` alone. A pair of subjects therefore gets the same
confidence interval on every page and in every request, whatever other subjects are present,
and one weight matrix over the union of keys of a task serves every pair.
"""

from __future__ import annotations

import math
import zlib
from collections.abc import Iterator

import numpy as np

POISSON1_CDF = np.cumsum([math.exp(-1) / math.factorial(k) for k in range(13)])
# The same CDF as integer thresholds on the 53-bit uniform draw. ``c <= y / 2**53`` holds exactly
# when ``ceil(c * 2**53) <= y``, and the integer search is several times faster.
_CDF_THRESHOLDS = np.array([math.ceil(c * 2.0**53) for c in POISSON1_CDF], dtype=np.uint64)
MAX_WEIGHT_CELLS = 4_000_000  # about 16 MB of float32 per chunk


def poisson_weights(
    key_hashes: np.ndarray, task_name: str, seed: int, b0: int, nb: int
) -> np.ndarray:
    """float32 array (nb, n) of Poisson(1) weights for resamples b0..b0+nb-1."""
    with np.errstate(over="ignore"):
        k = key_hashes.astype(np.int64).view(np.uint64)[None, :]
        b = np.arange(b0, b0 + nb, dtype=np.uint64)[:, None]
        t = np.uint64(zlib.crc32(task_name.encode()))
        x = (
            k
            ^ (np.uint64(seed) * np.uint64(0x9E3779B97F4A7C15))
            ^ (b * np.uint64(0xD1B54A32D192ED03))
            ^ (t << np.uint64(32))
        )
        x = x + np.uint64(0x9E3779B97F4A7C15)  # splitmix64 finalizer
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        x = x ^ (x >> np.uint64(31))
    return np.searchsorted(_CDF_THRESHOLDS, x >> np.uint64(11), side="right").astype(np.float32)


def weight_chunks(
    key_hashes: np.ndarray, task_name: str, seed: int, n_boot: int
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (b0, weights) chunks covering resamples 0..n_boot-1."""
    n = max(int(key_hashes.size), 1)
    step = max(1, MAX_WEIGHT_CELLS // n)
    for b0 in range(0, n_boot, step):
        nb = min(step, n_boot - b0)
        yield b0, poisson_weights(key_hashes, task_name, seed, b0, nb)


def resampled_means(
    key_hashes: np.ndarray,
    diffs: np.ndarray,
    mask: np.ndarray,
    task_name: str,
    seed: int,
    n_boot: int,
) -> np.ndarray:
    """Bootstrap means of each column of ``diffs`` over the rows where ``mask`` is set.

    ``key_hashes`` (n,), ``diffs`` and ``mask`` (n, P). Returns (n_boot, P); resamples whose
    weights sum to zero for a column are NaN.
    """
    d = np.where(mask, diffs, 0.0).astype(np.float64)
    m = mask.astype(np.float64)
    out = np.empty((n_boot, d.shape[1]), dtype=np.float64)
    for b0, w in weight_chunks(key_hashes, task_name, seed, n_boot):
        w64 = w.astype(np.float64)
        num = w64 @ d
        den = w64 @ m
        with np.errstate(invalid="ignore", divide="ignore"):
            out[b0 : b0 + w.shape[0]] = np.where(den > 0, num / den, np.nan)
    return out


def union_keys(*key_arrays: np.ndarray | None) -> np.ndarray:
    arrays = [k for k in key_arrays if k is not None and k.size > 0]
    if not arrays:
        return np.empty(0, dtype=np.int64)
    return np.unique(np.concatenate(arrays))


def align(union: np.ndarray, keys: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Values placed on ``union`` (sorted, unique); NaN where the key is absent."""
    out = np.full(union.shape, np.nan, dtype=np.float64)
    if keys.size == 0 or union.size == 0:
        return out
    idx = np.searchsorted(union, keys)
    ok = (idx < union.size) & (union[np.minimum(idx, union.size - 1)] == keys)
    out[idx[ok]] = values[ok]
    return out
