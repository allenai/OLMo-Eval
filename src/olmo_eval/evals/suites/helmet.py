"""HELMET-plus task suites organized by category and context size.

Mirrors the suite structure in `suites/ruler.py`:
- Per-category suites for each size: helmet_recall__4096, helmet_icl__4096, etc.
- Combined suites for each size: helmet_all__4096, etc.
- A judge-free execution split: helmet_nojudge__4096, etc.

Three deliberate differences from `suites/ruler.py`:

1. `helmet_all__*` aggregates the *category* suites rather than flat-averaging
   every task, matching how the HELMET paper reports an overall number. A flat
   average would let a category's task count set its weight -- ICL contributes
   five tasks and re-ranking one, so re-ranking would be worth a fifth of ICL.

2. `helmet_all__*` is only registered where more than one category exists at
   that length. Where a single category outruns the others, a combined suite
   would be a rename of that category's suite whose composition silently
   differs from the same suite at shorter lengths -- exactly the thing that
   turns a length sweep into a misleading trend line.

3. Some HELMET tasks are graded by an LLM judge (`narrativeqa` and both
   summarization tasks), so running them needs a judge configured. Following
   the execution-oriented split in `suites/science.py`, `helmet_all__*` is the
   umbrella that includes them and `helmet_nojudge__*` is the subset that runs
   without one.
"""

from olmo_eval.data.helmet_tasks import CONTEXT_SIZES, HELMET_TASKS
from olmo_eval.evals.suites.registry import AggregationStrategy, Suite, register

# Task categories (tags), in the order HELMET reports them
CATEGORIES = ["recall", "rag", "rerank", "longqa", "summ", "icl", "cite"]


def _tasks_for(category: str, size: int, judged: bool | None = None) -> list[str]:
    """Task names in a category at a size, optionally filtered by judge use.

    Ablations (the ALCE nocite variants) stay registered as tasks but are left
    out, since HELMET's category averages exclude them.
    """
    return [
        f"helmet_{name}"
        for name, config in HELMET_TASKS.items()
        if config["tag"] == category
        and not config.get("ablation")
        and name.endswith(f"__{size}")
        and (judged is None or bool(config.get("judged")) == judged)
    ]


for size in CONTEXT_SIZES:
    category_suites: list[Suite] = []
    nojudge_suites: list[Suite] = []
    extra_suites: list[Suite] = []

    for category in CATEGORIES:
        tasks = _tasks_for(category, size)
        if not tasks:
            continue

        suite = Suite(
            name=f"helmet_{category}__{size}",
            tasks=tuple(tasks),
            aggregation=AggregationStrategy.AVERAGE,
            description=f"HELMET-plus {category} tasks at {size} context length",
        )
        register(suite)
        category_suites.append(suite)

        # judge-free counterpart, used to build helmet_nojudge__*; it only needs
        # to exist as its own suite when the category has a judged task in it
        unjudged = _tasks_for(category, size, judged=False)
        if not unjudged:
            continue
        if len(unjudged) == len(tasks):
            nojudge_suites.append(suite)
        else:
            nojudge_suite = Suite(
                name=f"helmet_{category}_nojudge__{size}",
                tasks=tuple(unjudged),
                aggregation=AggregationStrategy.AVERAGE,
                description=(
                    f"HELMET-plus {category} tasks at {size} context length, "
                    "excluding LLM-judged tasks"
                ),
            )
            nojudge_suites.append(nojudge_suite)
            extra_suites.append(nojudge_suite)

    if len(category_suites) > 1:
        register(
            Suite(
                name=f"helmet_all__{size}",
                tasks=tuple(category_suites),
                aggregation=AggregationStrategy.AVERAGE_OF_AVERAGES,
                description=(
                    f"All HELMET-plus tasks at {size} context length "
                    f"(mean over category averages: {', '.join(CATEGORIES)}). "
                    "Includes LLM-judged tasks; use helmet_nojudge__* to skip them."
                ),
            )
        )

    # only worth registering when it would actually differ from helmet_all__*
    if len(nojudge_suites) > 1 and nojudge_suites != category_suites:
        for suite in extra_suites:
            register(suite)
        register(
            Suite(
                name=f"helmet_nojudge__{size}",
                tasks=tuple(nojudge_suites),
                aggregation=AggregationStrategy.AVERAGE_OF_AVERAGES,
                description=(
                    f"HELMET-plus tasks at {size} context length that need no LLM "
                    "judge (mean over category averages)"
                ),
            )
        )
