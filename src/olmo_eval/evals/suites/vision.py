"""Image-QA, dense-caption and multi-image benchmark suites.

Pointing suites live in :mod:`olmo_eval.evals.suites.pointing`, OCR suites in
:mod:`olmo_eval.evals.suites.ocr`.
"""

from olmo_eval.evals.suites.registry import AggregationStrategy, make_suite

# Single-image question answering (the image-QA set Molmo2 reports). ``mmmu_pro`` is a single
# task whose primary metric is the MMMU-Pro Overall = (standard-10 + vision)/2, so it contributes
# one 0-1 entry like the others.
IMAGE_QA_TASKS = (
    "chart_qa",
    "vqa2",
    "doc_qa",
    "info_qa",
    "text_vqa",
    "real_world_qa",
    "mmmu",
    "mmmu_pro",
    "math_vista",
    "countbench_qa",
    "pixmo_count",
    "ai2d",
    "charxiv_descriptive",
    "charxiv_reasoning",
)

make_suite(
    "image_qa",
    IMAGE_QA_TASKS,
    aggregation=AggregationStrategy.AVERAGE,
    description="Single-image question answering (primary metrics are all 0-1).",
)

make_suite(
    "image_qa_caption",
    (*IMAGE_QA_TASKS, "dense_caption"),
    # dense_caption's primary metric is 0-100, so no cross-task average is computed.
    aggregation=AggregationStrategy.DISPLAY_ONLY,
    description="Image-QA plus PixMo-Cap dense caption (GPT judge).",
)

# Multi-image benchmarks — each instance carries a list of images, scored by the mm_olmo
# MuirBenchEval-family protocol (option-letter accuracy).
MULTI_IMAGE_TASKS = (
    "muir_bench",
    "mmiu",
    "blink",
)

make_suite(
    "multi_image",
    MULTI_IMAGE_TASKS,
    aggregation=AggregationStrategy.AVERAGE,
    description="Multi-image question answering (primary metrics are accuracy, 0-1).",
)
