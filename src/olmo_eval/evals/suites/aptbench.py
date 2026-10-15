"""APTBench suites.

``aptbench`` is the paper's headline setting (software engineering plus deep
research). Per-category suites average their subtasks; composite suites give
each category equal weight. Every suite also exists with each ``ctx<N>k``
prompt-truncation variant appended to its name (e.g. ``aptbench:ctx32k``).
"""

from olmo_eval.evals.suites.registry import AggregationStrategy, Suite, make_suite
from olmo_eval.evals.tasks.aptbench import APTBENCH_CONTEXT_BUDGETS, aptbench_task_names

_CATEGORIES = ("env_setup", "issue_fix", "deepresearch", "tool", "agentic_math")

for _suffix in ("", *(f":ctx{budget}k" for budget in APTBENCH_CONTEXT_BUDGETS)):
    _by_category: dict[str, Suite] = {
        category: make_suite(
            name=f"aptbench:{category}{_suffix}",
            tasks=tuple(f"{task}{_suffix}" for task in aptbench_task_names(category)),
            description=f"APTBench {category} subtasks",
        )
        for category in _CATEGORIES
    }
    make_suite(
        name=f"aptbench:swe{_suffix}",
        tasks=(_by_category["env_setup"], _by_category["issue_fix"]),
        aggregation=AggregationStrategy.AVERAGE_OF_AVERAGES,
        description="APTBench-SWE: environment setup and issue fixing",
    )
    make_suite(
        name=f"aptbench{_suffix}",
        tasks=(_by_category["env_setup"], _by_category["issue_fix"], _by_category["deepresearch"]),
        aggregation=AggregationStrategy.AVERAGE_OF_AVERAGES,
        description="APTBench SWE and deep-research subtasks",
    )
    make_suite(
        name=f"aptbench:all{_suffix}",
        tasks=tuple(_by_category[category] for category in _CATEGORIES),
        aggregation=AggregationStrategy.AVERAGE_OF_AVERAGES,
        description="All APTBench subtasks, including tool use and agentic math",
    )
