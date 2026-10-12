"""Pointing benchmark suites: natural-image pointing and GUI click pointing.

All primary metrics are 0-1 (point-in-mask f1 or success rate, click accuracy), so each suite
averages its tasks.
"""

from olmo_eval.evals.suites.registry import AggregationStrategy, make_suite

# Natural-image pointing: point-in-mask f1 (PixMo-Points, SA-Co) and Point-Bench's official
# success rate.
POINTING_TASKS = (
    "pixmo_points_eval",
    "sa_co_gold_subset",
    "point_bench",
)

make_suite(
    "pointing",
    POINTING_TASKS,
    aggregation=AggregationStrategy.AVERAGE,
    description="Natural-image pointing: PixMo-Points, SA-Co gold subset, Point-Bench.",
)

# The mm_olmo `_mp` variants: same scoring, but the prompt is built from a bare label by
# the model's own formatter, so it follows the checkpoint (see `vision.scoring.prompts`). Under
# the stage-1 family that is `pointing: <label>`, the turn OLMo-core's stage 1 trains on.
# Point-Bench has no bare label (its queries are sentences), so it is asked as
# `pointing: <query>` under that family, mm_olmo's fallback for label-less pointing data.
POINTING_MP_TASKS = (
    "pixmo_points_eval_mp",
    "sa_co_gold_subset_mp",
    "sa_co_gold_point_4k_mp",
    "point_bench",
)

# `sa_co_gold_point_mp` (the unsampled 166,766-example gold set) is registered but kept out
# of the suite: it is ~6x the 4k variant and measures the same thing.

make_suite(
    "pointing_mp",
    POINTING_MP_TASKS,
    aggregation=AggregationStrategy.AVERAGE,
    description=(
        "Natural-image pointing with mm_olmo's model-prompt (_mp) inputs, plus Point-Bench; "
        "the suite for stage-1 checkpoints."
    ),
)

# GUI click pointing: the first predicted point must land on the target element.
GUI_POINTING_TASKS = (
    "screen_spot_v2",
    "screen_spot_pro",
    "os_world_g",
)

make_suite(
    "gui_pointing",
    GUI_POINTING_TASKS,
    aggregation=AggregationStrategy.AVERAGE,
    description="GUI click pointing: ScreenSpot-v2, ScreenSpot-Pro, OSWorld-G.",
)
