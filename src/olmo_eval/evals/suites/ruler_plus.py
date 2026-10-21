"""RULER-plus (ruler-plus) task suites organized by category and context size.

Mirrors suites/ruler.py's structure (per-category suites plus a combined
"all" suite per context size, a flat average over all tasks as in the paper)
for the ruler-plus task family. Like the tasks, these suites are internal-only
(see ruler_plus_loader.py).
"""

from olmo_eval.data.ruler_plus_tasks import CONTEXT_SIZES as RULER_PLUS_CONTEXT_SIZES
from olmo_eval.data.ruler_plus_tasks import RULER_PLUS_TASKS
from olmo_eval.evals.suites.registry import AggregationStrategy, Suite, register

# Task categories (tags)
CATEGORIES = ["niah", "multi_hop_tracing", "aggregation", "qa"]


# Create suites for each (category, context_size) combination
for size in RULER_PLUS_CONTEXT_SIZES:
    all_tasks: list[str] = []

    for category in CATEGORIES:
        # Find all tasks with this category and context size
        tasks = [
            f"ruler_plus_{task_name}"
            for task_name, task_config in RULER_PLUS_TASKS.items()
            if task_config["tag"] == category and task_name.endswith(f"__{size}")
        ]

        if len(tasks) == 0:
            continue

        # Register category-specific suite: ruler_plus_niah__4096
        suite = Suite(
            name=f"ruler_plus_{category}__{size}",
            tasks=tuple(tasks),
            aggregation=AggregationStrategy.AVERAGE,
            description=f"RULER-plus {category} tasks at {size} context length",
        )
        register(suite)
        all_tasks.extend(tasks)

    # Create combined suite: flat average of all tasks, matching the paper
    if len(all_tasks) > 0:
        all_tasks_suite = Suite(
            name=f"ruler_plus_all__{size}",
            tasks=tuple(all_tasks),
            aggregation=AggregationStrategy.AVERAGE,
            description=(
                f"All RULER-plus tasks at {size} context length (flat average of all tasks)"
            ),
        )
        register(all_tasks_suite)
