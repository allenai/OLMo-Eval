from olmo_eval.evals.suites.registry import AggregationStrategy, make_suite
from olmo_eval.evals.tasks.mmlu_pro import MMLU_PRO_CATEGORIES, category_slug

_SLUGS = tuple(category_slug(category) for category in MMLU_PRO_CATEGORIES)

MMLU_PRO = make_suite(
    "mmlu_pro",
    tuple(f"mmlu_pro_{s}" for s in _SLUGS),
    aggregation=AggregationStrategy.WEIGHTED_AVERAGE,
    description=(
        "5-shot multiple-choice MMLU-Pro, instance-weighted average over category tasks "
        "(matching the micro average the reference harness reports)"
    ),
)

MMLU_PRO_BPB = make_suite(
    "mmlu_pro:bpb",
    tuple(f"mmlu_pro_{s}:rc:bpb" for s in _SLUGS),
    aggregation=AggregationStrategy.WEIGHTED_AVERAGE,
    description=(
        "5-shot cloze MMLU-Pro bits per byte, instance-weighted average over category tasks. "
        "The per-task metric is already an instance average, so weighting by instance count "
        "reproduces the bits per byte of the pooled question set."
    ),
)

MMLU_PRO_COT = make_suite(
    "mmlu_pro:cot",
    tuple(f"mmlu_pro_{s}:cot" for s in _SLUGS),
    aggregation=AggregationStrategy.WEIGHTED_AVERAGE,
    description=(
        "0-shot chain-of-thought MMLU-Pro, instance-weighted average over category tasks. "
        "This deliberately diverges from oe-eval's mmlu_pro:0shot_cot::tulu3, which "
        "macro-averages, so that all three MMLU-Pro suites here roll up the same way as "
        "the benchmark's own pooled reporting."
    ),
)
