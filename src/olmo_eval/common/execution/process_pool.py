"""Process-backed execution helpers for CPU-heavy local scorers."""

from __future__ import annotations

import asyncio
import importlib
import multiprocessing as mp
import os
import pickle
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any

from olmo_eval.common.scorers.base import ProcessScorer
from olmo_eval.common.scorers.execution import format_scoring_error
from olmo_eval.common.types import Instance, LMOutput


def _resolve_attr(module_name: str, qualname: str) -> Any:
    obj = importlib.import_module(module_name)
    for attr in qualname.split("."):
        obj = getattr(obj, attr)
    return obj


@dataclass(frozen=True, slots=True)
class ProcessScoringPoolConfig:
    """Configuration for a named process pool used by CPU-heavy scorers."""

    workers: int
    start_method: str = "spawn"
    max_tasks_per_child: int | None = 256

    def __post_init__(self) -> None:
        if self.workers <= 0:
            raise ValueError("ProcessScoringPoolConfig.workers must be positive")
        if self.start_method not in {"spawn", "fork", "forkserver"}:
            raise ValueError(
                "ProcessScoringPoolConfig.start_method must be one of: spawn, fork, forkserver"
            )
        if self.start_method == "fork" and self.max_tasks_per_child is not None:
            raise ValueError(
                "ProcessScoringPoolConfig.max_tasks_per_child is incompatible with "
                "start_method='fork'; set max_tasks_per_child=None."
            )
        if self.max_tasks_per_child is not None and self.max_tasks_per_child <= 0:
            raise ValueError("ProcessScoringPoolConfig.max_tasks_per_child must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "workers": self.workers,
            "start_method": self.start_method,
            "max_tasks_per_child": self.max_tasks_per_child,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ProcessScoringPoolConfig:
        return cls(
            workers=int(data["workers"]),
            start_method=str(data.get("start_method", "spawn")),
            max_tasks_per_child=data.get("max_tasks_per_child", 256),
        )


@dataclass(frozen=True, slots=True)
class SerializedProcessScorer:
    """Serialized scorer definition that can be reconstructed in a subprocess."""

    module: str
    qualname: str
    params: dict[str, Any]
    scorer_name: str
    pool_name: str


@dataclass(frozen=True, slots=True)
class ProcessOutputScore:
    """Per-output score returned from a process-backed scorer job."""

    score: float
    error: dict[str, str] | None = None


class ProcessScoringConfigError(RuntimeError):
    """Raised when a process-backed scorer cannot be reconstructed safely."""


def serialize_process_scorer(scorer: ProcessScorer) -> SerializedProcessScorer:
    """Serialize a ProcessScorer into a form safe for subprocess reconstruction."""

    scorer_cls = scorer.__class__
    module = scorer_cls.__module__
    qualname = scorer_cls.__qualname__
    if module == "__main__" or "<locals>" in qualname:
        raise ProcessScoringConfigError(
            f"{scorer_cls.__qualname__} must be defined at module scope to use process scoring."
        )
    if not is_dataclass(scorer):
        raise ProcessScoringConfigError(
            f"{scorer_cls.__qualname__} must be a dataclass to use process scoring."
        )

    pool_name = getattr(scorer, "process_pool_name", "cpu")
    if not isinstance(pool_name, str) or not pool_name:
        raise ProcessScoringConfigError(
            f"{scorer_cls.__qualname__}.process_pool_name must be a non-empty string."
        )

    try:
        resolved_cls = _resolve_attr(module, qualname)
    except Exception as exc:
        raise ProcessScoringConfigError(
            f"Could not import process scorer {module}.{qualname}: {exc}"
        ) from exc

    if not isinstance(resolved_cls, type) or not issubclass(resolved_cls, ProcessScorer):
        raise ProcessScoringConfigError(f"{module}.{qualname} is not a ProcessScorer subclass.")

    try:
        params = asdict(scorer)
    except TypeError as exc:
        raise ProcessScoringConfigError(
            f"Could not serialize fields for process scorer {module}.{qualname}: {exc}"
        ) from exc

    try:
        pickle.dumps(params)
    except Exception as exc:
        raise ProcessScoringConfigError(
            f"Fields for process scorer {module}.{qualname} are not pickle-safe: {exc}"
        ) from exc

    return SerializedProcessScorer(
        module=module,
        qualname=qualname,
        params=params,
        scorer_name=scorer.name,
        pool_name=pool_name,
    )


_SCORER_CACHE: dict[tuple[str, str, bytes], ProcessScorer] = {}


def _get_cached_process_scorer(spec: SerializedProcessScorer) -> ProcessScorer:
    cache_key = (spec.module, spec.qualname, pickle.dumps(spec.params))
    scorer = _SCORER_CACHE.get(cache_key)
    if scorer is not None:
        return scorer

    scorer_cls = _resolve_attr(spec.module, spec.qualname)
    scorer = scorer_cls(**spec.params)
    if not isinstance(scorer, ProcessScorer):
        raise TypeError(f"{spec.module}.{spec.qualname} did not reconstruct as a ProcessScorer")
    _SCORER_CACHE[cache_key] = scorer
    return scorer


def _score_outputs_in_process(
    spec: SerializedProcessScorer,
    instance: Instance,
    outputs: list[LMOutput],
) -> list[ProcessOutputScore]:
    scorer = _get_cached_process_scorer(spec)
    scored: list[ProcessOutputScore] = []
    for output in outputs:
        try:
            value = scorer.process_score(instance, output)
            scored.append(ProcessOutputScore(score=float(value)))
        except Exception as exc:
            scored.append(
                ProcessOutputScore(score=0.0, error=format_scoring_error(exc, phase="process"))
            )
    return scored


class ProcessPoolManager:
    """Manage named process pools for CPU-heavy scoring workloads."""

    def __init__(self, pools: Mapping[str, ProcessScoringPoolConfig]):
        self._configs = dict(pools)
        self._executors: dict[str, ProcessPoolExecutor] = {}
        for name, config in self._configs.items():
            ctx = mp.get_context(config.start_method)
            kwargs: dict[str, Any] = {
                "max_workers": config.workers,
                "mp_context": ctx,
            }
            if config.max_tasks_per_child is not None:
                kwargs["max_tasks_per_child"] = config.max_tasks_per_child
            self._executors[name] = ProcessPoolExecutor(**kwargs)

    @property
    def pool_names(self) -> tuple[str, ...]:
        return tuple(self._executors)

    @property
    def max_workers(self) -> int:
        return max((config.workers for config in self._configs.values()), default=0)

    def get_pool(self, name: str) -> ProcessPoolExecutor:
        if name not in self._executors:
            available = ", ".join(sorted(self._executors)) or "none"
            raise ProcessScoringConfigError(
                f"Process scoring pool '{name}' is not configured. Available pools: {available}"
            )
        return self._executors[name]

    async def score_outputs(
        self,
        scorer: ProcessScorer,
        instance: Instance,
        outputs: list[LMOutput],
    ) -> list[ProcessOutputScore]:
        spec = serialize_process_scorer(scorer)
        loop = asyncio.get_running_loop()
        pool = self.get_pool(spec.pool_name)
        return await loop.run_in_executor(
            pool,
            _score_outputs_in_process,
            spec,
            instance,
            list(outputs),
        )

    def shutdown(self, *, wait: bool = True, cancel_futures: bool = True) -> None:
        for executor in self._executors.values():
            executor.shutdown(wait=wait, cancel_futures=cancel_futures)
        self._executors.clear()


def default_process_pool_workers() -> int:
    """Default worker count for auto-created process scoring pools."""

    cpu_count = os.cpu_count() or 4
    return max(1, int(cpu_count * 0.75))
