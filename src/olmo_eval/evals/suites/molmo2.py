"""Molmo2 multimodal benchmark suites."""

from olmo_eval.evals.suites.registry import AggregationStrategy, make_suite

# The image-QA benchmarks Molmo2 reports. ``mmmu_pro`` is a single task whose primary metric is the
# MMMU-Pro Overall = (standard-10 + vision)/2, so it contributes one 0-1 entry like the others.
MOLMO2_IMAGE_QA_TASKS = (
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
    "molmo2_imageqa",
    MOLMO2_IMAGE_QA_TASKS,
    aggregation=AggregationStrategy.AVERAGE,
    description="Molmo2's image-QA benchmarks (primary metrics are all 0-1).",
)

make_suite(
    "molmo2_imageqa_caption",
    (*MOLMO2_IMAGE_QA_TASKS, "dense_caption"),
    # dense_caption's primary metric is 0-100, so no cross-task average is computed.
    aggregation=AggregationStrategy.DISPLAY_ONLY,
    description="Molmo2's image-QA benchmarks plus PixMo-Cap dense caption (GPT judge).",
)

# Image-pointing benchmarks — point-in-mask precision/recall/f1, not VQA-style answers.
MOLMO2_POINTING_TASKS = (
    "pixmo_points_eval",
    "sa_co_gold_subset",
)

make_suite(
    "molmo2_pointing",
    MOLMO2_POINTING_TASKS,
    aggregation=AggregationStrategy.AVERAGE,
    description="Molmo2's image-pointing benchmarks (primary metric is f1, 0-1).",
)

# The mm_olmo `_mp` variants: same scoring, but the prompt is built from a bare label by
# the model's own formatter, so it follows the checkpoint (see `vision.scoring.prompts`).
MOLMO2_POINTING_MP_TASKS = (
    "pixmo_points_eval_mp",
    "sa_co_gold_subset_mp",
    "sa_co_gold_point_4k_mp",
)

# `sa_co_gold_point_mp` (the unsampled 166,766-example gold set) is registered but kept out
# of the suite: it is ~6x the 4k variant and measures the same thing.

make_suite(
    "molmo2_pointing_mp",
    MOLMO2_POINTING_MP_TASKS,
    aggregation=AggregationStrategy.AVERAGE,
    description="Molmo2's image-pointing benchmarks with mm_olmo's model-prompt (_mp) inputs.",
)

# Multi-image benchmarks — each instance carries a list of images, scored by the mm_olmo
# MuirBenchEval-family protocol (option-letter accuracy).
MOLMO2_MULTI_IMAGE_TASKS = (
    "muir_bench",
    "mmiu",
    "blink",
)

make_suite(
    "molmo2_multiimage",
    MOLMO2_MULTI_IMAGE_TASKS,
    aggregation=AggregationStrategy.AVERAGE,
    description="Molmo2's multi-image benchmarks (primary metrics are accuracy, 0-1).",
)
