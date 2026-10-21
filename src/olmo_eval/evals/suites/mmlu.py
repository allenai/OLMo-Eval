from olmo_eval.evals.suites.registry import AggregationStrategy, Suite, make_suite, register
from olmo_eval.evals.tasks.mmlu import _HUMANITIES, _OTHER, _SOCIAL_SCIENCES, _STEM, MMLU_SUBJECTS


def _task_names(subjects: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(f"mmlu_{s}" for s in subjects)


def _task_names_variant(subjects: tuple[str, ...], variant: str) -> tuple[str, ...]:
    return tuple(f"mmlu_{s}:{variant}" for s in subjects)


MMLU_STEM = make_suite(
    "mmlu:stem",
    _task_names(_STEM),
)

MMLU_HUMANITIES = make_suite(
    "mmlu:humanities",
    _task_names(_HUMANITIES),
)

MMLU_SOCIAL_SCIENCES = make_suite(
    "mmlu:social_sciences",
    _task_names(_SOCIAL_SCIENCES),
)

MMLU_OTHER = make_suite(
    "mmlu:other",
    _task_names(_OTHER),
)

MMLU = register(
    Suite(
        name="mmlu",
        tasks=(MMLU_STEM, MMLU_HUMANITIES, MMLU_SOCIAL_SCIENCES, MMLU_OTHER),
        aggregation=AggregationStrategy.AVERAGE_OF_AVERAGES,
    )
)

MMLU_BPB = make_suite(
    "mmlu:bpb",
    tuple(f"mmlu_{s}:rc:bpb" for s in MMLU_SUBJECTS),
)

MMLU_COT = make_suite(
    "mmlu:cot",
    _task_names_variant(MMLU_SUBJECTS, "cot"),
    aggregation=AggregationStrategy.AVERAGE,
    description="0-shot chain-of-thought MMLU across all 57 subjects",
)
