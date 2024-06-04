from typing import Dict

from catwalk.task import Task, rc_metrics
from catwalk.tasks.eleuther import EleutherMMLUTask

from .mmlu_std import create_catwalk_mmlu_std_tasks

# Defines TASKS_LM specifically for task variants geared towards LM type models
# Usually will use TASKS from catwalk as fallback


TASKS_STD: Dict[str, Task] = {
    # Standard eval suite tasks, we use EleutherMMLUTask as a workaround for tasks with
    # fixed fewshot prompt
    "arc_easy_std": EleutherMMLUTask("arc_easy_std", ranked_classification=True).add_metrics(
        rc_metrics(primary="acc_per_char")
    ),
    "arc_easy_mc_std": EleutherMMLUTask("arc_easy_mc_std", ranked_classification=True).add_metrics(
        rc_metrics(primary="acc_raw")
    ),
    "arc_challenge_std": EleutherMMLUTask(
        "arc_challenge_std", ranked_classification=True
    ).add_metrics(rc_metrics(primary="acc_uncond")),
    "arc_challenge_mc_std": EleutherMMLUTask(
        "arc_challenge_mc_std", ranked_classification=True
    ).add_metrics(rc_metrics(primary="acc_raw")),
    "boolq_std": EleutherMMLUTask("boolq_std", ranked_classification=True).add_metrics(
        rc_metrics(primary="acc_raw")
    ),
    "boolq_mc_std": EleutherMMLUTask("boolq_mc_std", ranked_classification=True).add_metrics(
        rc_metrics(primary="acc_raw")
    ),
    "hellaswag_std": EleutherMMLUTask("hellaswag_std", ranked_classification=True).add_metrics(
        rc_metrics(primary="acc_per_char")
    ),
    "hellaswag_mc_std": EleutherMMLUTask(
        "hellaswag_mc_std", ranked_classification=True
    ).add_metrics(rc_metrics(primary="acc_raw")),
    **create_catwalk_mmlu_std_tasks(),
    "winogrande_std": EleutherMMLUTask("winogrande_std", ranked_classification=True).add_metrics(
        rc_metrics(primary="acc_raw")
    ),
    "winogrande_mc_std": EleutherMMLUTask(
        "winogrande_mc_std", ranked_classification=True
    ).add_metrics(rc_metrics(primary="acc_raw")),
    "csqa_std": EleutherMMLUTask("csqa_std", ranked_classification=True).add_metrics(
        rc_metrics(primary="acc_uncond")
    ),
    "csqa_mc_std": EleutherMMLUTask("csqa_mc_std", ranked_classification=True).add_metrics(
        rc_metrics(primary="acc_raw")
    ),
    "openbookqa_std": EleutherMMLUTask("openbookqa_std", ranked_classification=True).add_metrics(
        rc_metrics(primary="acc_uncond")
    ),
    "openbookqa_mc_std": EleutherMMLUTask(
        "openbookqa_mc_std", ranked_classification=True
    ).add_metrics(rc_metrics(primary="acc_raw")),
    "piqa_std": EleutherMMLUTask("piqa_std", ranked_classification=True).add_metrics(
        rc_metrics(primary="acc_per_char")
    ),
    "piqa_mc_std": EleutherMMLUTask("piqa_mc_std", ranked_classification=True).add_metrics(
        rc_metrics(primary="acc_raw")
    ),
    "socialiqa_std": EleutherMMLUTask("socialiqa_std", ranked_classification=True).add_metrics(
        rc_metrics(primary="acc_per_char")
    ),
    "socialiqa_mc_std": EleutherMMLUTask(
        "socialiqa_mc_std", ranked_classification=True
    ).add_metrics(rc_metrics(primary="acc_raw")),
}
