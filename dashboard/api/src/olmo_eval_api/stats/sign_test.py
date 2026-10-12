"""Exact two-sided sign test (spec 5.6)."""

from __future__ import annotations

import math


def _log_comb(n: int, k: int) -> float:
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def sign_test(wins: int, losses: int) -> float | None:
    """``min(1, 2 * sum_{k<=min(wins, losses)} C(n, k) / 2^n)`` with n = wins + losses.

    None when nothing is contested.
    """
    n = wins + losses
    if n == 0:
        return None
    k_max = min(wins, losses)
    log_terms = [_log_comb(n, k) - n * math.log(2) for k in range(k_max + 1)]
    top = max(log_terms)
    log_sum = top + math.log(sum(math.exp(t - top) for t in log_terms))
    return min(1.0, 2.0 * math.exp(log_sum))
