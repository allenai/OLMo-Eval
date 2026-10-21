"""Arena-Hard v2.0 suite.

The two categories are judged against different baseline models, so their win
rates are shown side by side rather than averaged.
"""

from olmo_eval.evals.suites.registry import AggregationStrategy, make_suite

ARENA_HARD_V2 = make_suite(
    "arena_hard_v2",
    ("arena_hard_v2:hard_prompt", "arena_hard_v2:creative_writing"),
    aggregation=AggregationStrategy.DISPLAY_ONLY,
    description="Arena-Hard v2.0: hard prompts and creative writing, judged against baselines",
)
