from olmo_eval.evals.suites.registry import AggregationStrategy, make_suite
from olmo_eval.evals.tasks.minerva_math import MATH_SUBSETS

make_suite(
    "minerva_math",
    tuple(f"minerva_math_{t}" for t in MATH_SUBSETS),
    aggregation=AggregationStrategy.AVERAGE,
)

make_suite(
    "aime_2022_to_2026",
    (
        "aime_2022:pass_at_32",
        "aime_2023:pass_at_32",
        "aime_2024:pass_at_32",
        "aime_2025:pass_at_32",
        "aime_2026:pass_at_32",
    ),
    aggregation=AggregationStrategy.AVERAGE_OF_AVERAGES,
    description="AIME 2022-2026 combined.",
)

make_suite(
    "math:posttrain:dev",
    (
        "aime_2022:pass_at_32:16k",
        "aime_2023:pass_at_32:16k",
        "aime_2024:pass_at_32:16k",
        "aime_2025:pass_at_32:16k",
        "aime_2026:pass_at_32:16k",
        "hmmt_nov_2025:pass_at_32:16k",
        "hmmt_feb_2026:pass_at_32:16k",
    ),
    aggregation=AggregationStrategy.AVERAGE,
    description=(
        "Dev set for post-training math experiments: AIME 2022-2026 and HMMT Nov 2025 / Feb 2026."
    ),
)

make_suite(
    "omega:dev",
    ("omega_500:hillclimb", "omega_500_out"),
    aggregation=AggregationStrategy.GAP,
    description=(
        "OMEGA dev tier: omega-500 (in-distribution templates) and its matched test_out "
        "companion, reported as both strict scores and the gap (out minus in)."
    ),
)
