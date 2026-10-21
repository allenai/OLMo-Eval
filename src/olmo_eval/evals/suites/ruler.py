"""RULER task suites organized by category and context size.

Matches the suite structure from the old framework:
- Per-category suites for each size: ruler_niah__4096, ruler_aggregation__8192, etc.
- Combined suites for each size: ruler_all__4096, ruler_all__8192, etc.
"""

from olmo_eval.data.ruler_tasks import RULER_TASKS
from olmo_eval.evals.suites.registry import AggregationStrategy, Suite, register

# Context sizes to create suites for
CONTEXT_SIZES = [4096, 8192, 16384, 32768, 65536, 131072]

# Task categories (tags)
CATEGORIES = ["niah", "multi_hop_tracing", "aggregation", "qa"]


# Create suites for each (category, context_size) combination
for size in CONTEXT_SIZES:
    all_tasks: list[str] = []

    for category in CATEGORIES:
        # Find all tasks with this category and context size
        tasks = [
            f"ruler_{task_name}"
            for task_name, task_config in RULER_TASKS.items()
            if task_config["tag"] == category and task_name.endswith(f"__{size}")
        ]

        if len(tasks) == 0:
            continue

        # Register category-specific suite: ruler_niah__4096
        suite = Suite(
            name=f"ruler_{category}__{size}",
            tasks=tuple(tasks),
            aggregation=AggregationStrategy.AVERAGE,
            description=f"RULER {category} tasks at {size} context length",
        )
        register(suite)
        all_tasks.extend(tasks)

    # Create combined suite: flat average of all 13 tasks, matching the paper
    # ("The performance at a certain length is the average of all 13 tasks in RULER")
    if len(all_tasks) > 0:
        all_tasks_suite = Suite(
            name=f"ruler_all__{size}",
            tasks=tuple(all_tasks),
            aggregation=AggregationStrategy.AVERAGE,
            description=f"All RULER tasks at {size} context length (flat average of all tasks)",
        )
        register(all_tasks_suite)
        register(
            Suite(
                name=f"ruler_all_chat__{size}",
                tasks=tuple(f"{t}:chat" for t in all_tasks),
                aggregation=AggregationStrategy.AVERAGE,
                description=f"ruler_all__{size}, every task served through the chat template",
            )
        )
