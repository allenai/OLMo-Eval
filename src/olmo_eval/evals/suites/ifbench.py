"""IFBench instruction-following suite."""

from olmo_eval.evals.suites.registry import make_suite

IFBENCH = make_suite(
    "ifbench",
    (
        "ifeval_mt_wildchat_unused_withRewrite",
        "ifeval_mt_ood_wildchat_unused_withRewrite",
        "ifeval_ood",
    ),
    description="IFBench: OOD + multi-turn instruction following",
)
