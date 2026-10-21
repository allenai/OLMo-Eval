"""Berkeley Function Calling Leaderboard suites.

The suites reproduce the leaderboard's own summaries for the single-turn
categories. Two kinds of averaging are involved, and BFCL uses them in
different places:

- Its non-live summaries average the categories as equals, which the suites
  below express with ``AVERAGE`` and ``AVERAGE_OF_AVERAGES``.
- Its live summaries weight each category by how many instances it holds. A
  suite cannot express that, so the weighted summaries are tasks that pool
  their categories' instances — ``bfcl_live_ast`` and ``bfcl_live`` — which
  gives the same number.

The executable and REST categories are not implemented, so ``bfcl:non_live``
here is BFCL's non-live overall with those terms left out, and ``bfcl`` is its
single-turn overall on the same footing. Both differ from a published non-live
number, which averages the executable categories in.
"""

from olmo_eval.evals.suites.registry import AggregationStrategy, Suite, register
from olmo_eval.evals.tasks.bfcl.multi_turn import MULTI_TURN_CATEGORIES

#: Suffix appended to every task spec in a suite, one per prompting regime.
REGIMES: tuple[tuple[str, str], ...] = (
    ("", "native function calling"),
    (":prompt", "BFCL's prompting mode"),
    (":base", "completion prompting with exemplars"),
)

CATEGORIES: tuple[str, ...] = (
    "simple",
    "multiple",
    "parallel",
    "parallel_multiple",
    "java",
    "javascript",
    "irrelevance",
    "live_simple",
    "live_multiple",
    "live_parallel",
    "live_parallel_multiple",
    "live_irrelevance",
    "live_relevance",
)


def _register_regime(suffix: str, description: str) -> None:
    """Register the suite family for one prompting regime."""
    simple = register(
        Suite(
            name=f"bfcl:non_live_simple{suffix}",
            tasks=(f"bfcl_simple{suffix}", f"bfcl_java{suffix}", f"bfcl_javascript{suffix}"),
            aggregation=AggregationStrategy.AVERAGE,
            description=f"BFCL non-live simple AST, across languages ({description})",
        )
    )

    ast_terms: tuple[str | Suite, ...] = (
        simple,
        f"bfcl_multiple{suffix}",
        f"bfcl_parallel{suffix}",
        f"bfcl_parallel_multiple{suffix}",
    )
    register(
        Suite(
            name=f"bfcl:non_live_ast{suffix}",
            tasks=ast_terms,
            aggregation=AggregationStrategy.AVERAGE_OF_AVERAGES,
            description=f"BFCL non-live AST summary ({description})",
        )
    )

    non_live = register(
        Suite(
            name=f"bfcl:non_live{suffix}",
            tasks=(*ast_terms, f"bfcl_irrelevance{suffix}"),
            aggregation=AggregationStrategy.AVERAGE_OF_AVERAGES,
            description=f"BFCL non-live AST summary with irrelevance ({description})",
        )
    )

    register(
        Suite(
            name=f"bfcl{suffix}",
            tasks=(non_live, f"bfcl_live{suffix}"),
            aggregation=AggregationStrategy.AVERAGE_OF_AVERAGES,
            description=f"BFCL v3 single-turn overall ({description})",
        )
    )

    register(
        Suite(
            name=f"bfcl:categories{suffix}",
            tasks=tuple(f"bfcl_{category}{suffix}" for category in CATEGORIES),
            aggregation=AggregationStrategy.DISPLAY_ONLY,
            description=f"Every BFCL v3 single-turn category, reported separately ({description})",
        )
    )


def _register_multi_turn(suffix: str, description: str) -> None:
    """Register the multi-turn summary for one regime.

    These run under a harness carrying the bfcl_multi_turn scaffold, so they
    are kept apart from the single-turn suites rather than folded in.
    """
    register(
        Suite(
            name=f"bfcl:multi_turn{suffix}",
            tasks=tuple(
                f"bfcl_multi_turn_{category}{suffix}" for category in MULTI_TURN_CATEGORIES
            ),
            aggregation=AggregationStrategy.AVERAGE,
            description=f"BFCL v3 multi-turn summary ({description})",
        )
    )


for _suffix, _description in REGIMES:
    _register_regime(_suffix, _description)

for _suffix, _description in REGIMES:
    if _suffix == ":base":
        # A pretrained model is not asked to drive a stateful rollout.
        continue
    _register_multi_turn(_suffix, _description)
