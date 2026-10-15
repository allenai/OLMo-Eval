"""Development tiers and guard benchmarks without a cross-domain aggregate."""

from olmo_eval.evals.suites.math import OMEGA_DEV
from olmo_eval.evals.suites.registry import AggregationStrategy, make_suite

HILLCLIMB_DEV_TIERS = ("livecodebench:lite", OMEGA_DEV)
HILLCLIMB_GUARDS = ("ifeval", "ifeval_ood", "gpqa_main:cot")

make_suite(
    "hillclimb:dev",
    (*HILLCLIMB_DEV_TIERS, *HILLCLIMB_GUARDS),
    aggregation=AggregationStrategy.NONE,
    description=(
        "Hill-climb dev pass: LiveCodeBench lite, the OMEGA in/out pair "
        "with its gap, and the IFEval, IFBench and GPQA main guards."
    ),
)
