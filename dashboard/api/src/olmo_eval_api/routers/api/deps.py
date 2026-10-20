"""Dependencies shared by the /api routers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, Query, Request
from sqlalchemy.ext.asyncio import AsyncSession

from olmo_eval_api.auth.iap import DashboardUser, current_user
from olmo_eval_api.db.engine import get_session
from olmo_eval_api.errors import bad_request
from olmo_eval_api.settings import Settings
from olmo_eval_api.storage.base import Storage

SessionDep = Annotated[AsyncSession, Depends(get_session)]
UserDep = Annotated[DashboardUser, Depends(current_user)]
ALPHAS = (0.01, 0.05, 0.1)
MAX_SEED = 2**32 - 1


def get_settings_dep(request: Request) -> Settings:
    return request.app.state.settings


def get_storage_dep(request: Request) -> Storage:
    return request.app.state.storage


SettingsDep = Annotated[Settings, Depends(get_settings_dep)]
StorageDep = Annotated[Storage, Depends(get_storage_dep)]


@dataclass(frozen=True)
class StatsParams:
    alpha: float = 0.05
    n_boot: int = 2000
    seed: int = 0
    threshold: float = 0.5


def check_stats(
    alpha: float | None, n_boot: int | None, seed: int | None, threshold: float | None
) -> StatsParams:
    alpha = 0.05 if alpha is None else alpha
    if not any(abs(alpha - x) < 1e-12 for x in ALPHAS):
        raise bad_request("alpha must be one of 0.01, 0.05, 0.1")
    n_boot = 2000 if n_boot is None else n_boot
    if not 200 <= n_boot <= 10_000:
        raise bad_request("n_boot must be between 200 and 10000")
    seed = seed or 0
    if not 0 <= seed <= MAX_SEED:
        raise bad_request(f"seed must be between 0 and {MAX_SEED}")
    return StatsParams(alpha=alpha, n_boot=n_boot, seed=seed, threshold=check_threshold(threshold))


def check_threshold(threshold: float | None) -> float:
    """Correctness threshold for binary and bounded metrics: a number in [0, 1]."""
    if threshold is None:
        return 0.5
    if not (math.isfinite(threshold) and 0.0 <= threshold <= 1.0):
        raise bad_request("threshold must be a number between 0 and 1")
    return threshold


def stats_params(
    alpha: float | None = None,
    n_boot: int | None = None,
    seed: int | None = None,
    threshold: float | None = None,
) -> StatsParams:
    return check_stats(alpha, n_boot, seed, threshold)


StatsDep = Annotated[StatsParams, Depends(stats_params)]
LimitQuery = Annotated[int | None, Query()]
ThresholdQuery = Annotated[float, Query(ge=0.0, le=1.0, allow_inf_nan=False)]
