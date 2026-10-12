"""Scoring for the vision benchmarks: parsers, prompt families, and judges."""

from olmo_eval.evals.vision.scoring.count_parsing import (
    WORD_TO_NUM,
    extract_image_points,
    parse_count,
)
from olmo_eval.evals.vision.scoring.math_vista_offline import (
    DEMO_PROMPT,
    create_test_prompt,
    extract_answer_offline,
    extract_answer_quick,
    math_vista_score_from_extraction,
    math_vista_score_offline,
    normalize_extracted_answer,
    safe_equal,
)
from olmo_eval.evals.vision.scoring.multi_image import (
    multi_image_mc_score,
    strip_multi_image_response,
)
from olmo_eval.evals.vision.scoring.multiple_choice import (
    eval_multi_choice,
    eval_open,
    mmmu_score,
    parse_multi_choice_response,
    parse_open_response,
)
from olmo_eval.evals.vision.scoring.prompt_templates import (
    EVAL_LOADER_SEED,
    LONG_CAPTION_TEMPLATES,
    POINT_COUNT_TEMPLATES,
    dense_caption_question,
    format_mc_question,
    pixmo_count_question,
)
from olmo_eval.evals.vision.scoring.vqa_normalization import (
    anls_metric,
    clean_prediction,
    levenshtein,
    preprocess_answer,
    real_world_qa_score,
    relaxed_correctness,
    scifi_relaxed_correctness,
    select_mc_option,
    vqa_score,
)

__all__ = [
    "DEMO_PROMPT",
    "EVAL_LOADER_SEED",
    "LONG_CAPTION_TEMPLATES",
    "POINT_COUNT_TEMPLATES",
    "WORD_TO_NUM",
    "anls_metric",
    "clean_prediction",
    "create_test_prompt",
    "dense_caption_question",
    "eval_multi_choice",
    "eval_open",
    "extract_answer_offline",
    "extract_answer_quick",
    "extract_image_points",
    "format_mc_question",
    "levenshtein",
    "math_vista_score_from_extraction",
    "math_vista_score_offline",
    "mmmu_score",
    "multi_image_mc_score",
    "normalize_extracted_answer",
    "parse_count",
    "parse_multi_choice_response",
    "parse_open_response",
    "pixmo_count_question",
    "preprocess_answer",
    "real_world_qa_score",
    "safe_equal",
    "relaxed_correctness",
    "scifi_relaxed_correctness",
    "select_mc_option",
    "strip_multi_image_response",
    "vqa_score",
]
