"""General post-training dev suite."""

import olmo_eval.evals.suites.ifbench  # noqa: F401 - ensure suite registration
import olmo_eval.evals.suites.math  # noqa: F401 - ensure suite registration
from olmo_eval.evals.suites.registry import AggregationStrategy, get_suite, make_suite

make_suite(
    "general:posttrain:dev",
    (
        get_suite("math:posttrain:dev"),
        "scicode",
        get_suite("ifbench"),
        "gpqa_diamond",
    ),
    aggregation=AggregationStrategy.AVERAGE_OF_AVERAGES,
    description=(
        "Dev suite for general post-training: math:posttrain:dev, SciCode, IFBench, GPQA Diamond."
    ),
)
