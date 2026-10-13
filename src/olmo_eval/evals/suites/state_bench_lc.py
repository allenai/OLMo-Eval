import logging

from olmo_eval.evals.suites.registry import AggregationStrategy, make_suite
from olmo_eval.evals.tasks.state_bench_lc import (
    STATE_BENCH_10PCT_TASKS,
    STATE_BENCH_CONFIGS,
    STATE_BENCH_STRATA_BY_CONFIG,
    STATE_BENCH_TASKS,
    STATE_BENCH_TOKEN_STRATA,
    state_bench_10pct_task_name,
    state_bench_configs_for_stratum,
    state_bench_task_name,
)

logger = logging.getLogger(__name__)

_ALL_STRATA_NOTE = (
    " Spans every token stratum up to 1m tokens; strata longer than the model's context"
    " are left-truncated by the provider, so select stratum suites"
    " (e.g. state_bench:tokens_64k) within the model's context instead."
)


def _coverage_note(tasks: tuple[str, ...], total: int) -> str:
    """Describe partial config coverage so a narrowed suite is visible, not silent."""
    if len(tasks) == total:
        return ""
    return f" Covers {len(tasks)} of {total} configs; the rest have no split in this tier."


def _make_suite_if_nonempty(
    name: str,
    tasks: tuple[str, ...],
    total: int,
    description: str,
) -> None:
    """Register a suite, skipping (and logging) one whose configs all lack the split."""
    if not tasks:
        logger.info(
            "Skipping state-bench suite %r: none of the %d candidate configs have a "
            "staged split for this token tier.",
            name,
            total,
        )
        return
    make_suite(
        name=name,
        tasks=tasks,
        aggregation=AggregationStrategy.AVERAGE,
        description=description + _coverage_note(tasks, total),
    )


make_suite(
    name="state_bench",
    tasks=STATE_BENCH_TASKS,
    aggregation=AggregationStrategy.AVERAGE,
    description="Long-context StateBench tasks across all formats and complexity classes."
    + _ALL_STRATA_NOTE,
)

make_suite(
    name="state_bench_10pct",
    tasks=STATE_BENCH_10PCT_TASKS,
    aggregation=AggregationStrategy.AVERAGE,
    description="Deterministic 10% sample of the long-context StateBench tasks." + _ALL_STRATA_NOTE,
)

for _complexity in ("aperiodic", "periodic", "r-trivial"):
    make_suite(
        name=f"state_bench:{_complexity.replace('-', '_')}",
        tasks=tuple(
            state_bench_task_name(config_name, token_stratum)
            for config_name in STATE_BENCH_CONFIGS
            if config_name.endswith(f"--{_complexity}")
            for token_stratum in STATE_BENCH_STRATA_BY_CONFIG[config_name]
        ),
        aggregation=AggregationStrategy.AVERAGE,
        description=f"Long-context StateBench tasks in the {_complexity} complexity class."
        + _ALL_STRATA_NOTE,
    )
    make_suite(
        name=f"state_bench_10pct:{_complexity.replace('-', '_')}",
        tasks=tuple(
            state_bench_10pct_task_name(config_name, token_stratum)
            for config_name in STATE_BENCH_CONFIGS
            if config_name.endswith(f"--{_complexity}")
            for token_stratum in STATE_BENCH_STRATA_BY_CONFIG[config_name]
        ),
        aggregation=AggregationStrategy.AVERAGE,
        description=f"Deterministic 10% StateBench sample in the {_complexity} complexity class."
        + _ALL_STRATA_NOTE,
    )

for _formatter in (
    "cube-painting",
    "integer-code",
    "people-in-rooms",
    "ruler",
    "spreadsheet-cells",
    "status-lights",
):
    make_suite(
        name=f"state_bench:{_formatter.replace('-', '_')}",
        tasks=tuple(
            state_bench_task_name(config_name, token_stratum)
            for config_name in STATE_BENCH_CONFIGS
            if config_name.startswith(f"{_formatter}--")
            for token_stratum in STATE_BENCH_STRATA_BY_CONFIG[config_name]
        ),
        aggregation=AggregationStrategy.AVERAGE,
        description=f"Long-context StateBench tasks using the {_formatter} format."
        + _ALL_STRATA_NOTE,
    )
    make_suite(
        name=f"state_bench_10pct:{_formatter.replace('-', '_')}",
        tasks=tuple(
            state_bench_10pct_task_name(config_name, token_stratum)
            for config_name in STATE_BENCH_CONFIGS
            if config_name.startswith(f"{_formatter}--")
            for token_stratum in STATE_BENCH_STRATA_BY_CONFIG[config_name]
        ),
        aggregation=AggregationStrategy.AVERAGE,
        description=f"Deterministic 10% StateBench sample using the {_formatter} format."
        + _ALL_STRATA_NOTE,
    )
    _formatter_configs = tuple(
        config_name
        for config_name in STATE_BENCH_CONFIGS
        if config_name.startswith(f"{_formatter}--")
    )
    for _token_stratum in STATE_BENCH_TOKEN_STRATA:
        _staged = state_bench_configs_for_stratum(_token_stratum)
        _staged_configs = tuple(
            config_name for config_name in _formatter_configs if config_name in _staged
        )
        _make_suite_if_nonempty(
            name=(f"state_bench:{_formatter.replace('-', '_')}:{_token_stratum}"),
            tasks=tuple(
                state_bench_task_name(config_name, _token_stratum)
                for config_name in _staged_configs
            ),
            total=len(_formatter_configs),
            description=(
                f"Long-context StateBench {_formatter} tasks in the {_token_stratum} context tier."
            ),
        )
        _make_suite_if_nonempty(
            name=(f"state_bench_10pct:{_formatter.replace('-', '_')}:{_token_stratum}"),
            tasks=tuple(
                state_bench_10pct_task_name(config_name, _token_stratum)
                for config_name in _staged_configs
            ),
            total=len(_formatter_configs),
            description=(
                f"Deterministic 10% StateBench {_formatter} sample in the "
                f"{_token_stratum} context tier."
            ),
        )

for _token_stratum in STATE_BENCH_TOKEN_STRATA:
    _staged_configs = state_bench_configs_for_stratum(_token_stratum)
    _make_suite_if_nonempty(
        name=f"state_bench:{_token_stratum}",
        tasks=tuple(
            state_bench_task_name(config_name, _token_stratum) for config_name in _staged_configs
        ),
        total=len(STATE_BENCH_CONFIGS),
        description=f"Long-context StateBench tasks in the {_token_stratum} context tier.",
    )
    _make_suite_if_nonempty(
        name=f"state_bench_10pct:{_token_stratum}",
        tasks=tuple(
            state_bench_10pct_task_name(config_name, _token_stratum)
            for config_name in _staged_configs
        ),
        total=len(STATE_BENCH_CONFIGS),
        description=f"Deterministic 10% StateBench sample in the {_token_stratum} context tier.",
    )
