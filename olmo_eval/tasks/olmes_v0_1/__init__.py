from catwalk.dependencies.lm_eval.tasks import TASK_REGISTRY

from . import (
    arc_std,
    boolq_std,
    csqa_std,
    hellaswag_std,
    mmlu_std,
    openbookqa_std,
    piqa_std,
    siqa_std,
    winogrande_std,
)

# This directory has new tasks associated with the Standardized Eval Suite
# Update Eleuther TASK_REGISTRY with local tasks

TASK_REGISTRY.update({
    "arc_easy_std": arc_std.ARCEasyStd,
    "arc_easy_mc_std": arc_std.ARCEasyMCStd,
    "arc_challenge_std": arc_std.ARCChallengeStd,
    "arc_challenge_mc_std": arc_std.ARCChallengeMCStd,
    "boolq_std": boolq_std.BoolQStd,
    "boolq_mc_std": boolq_std.BoolQMCStd,
    "hellaswag_std": hellaswag_std.HellaSwagStd,
    "hellaswag_mc_std": hellaswag_std.HellaSwagMCStd,
    **mmlu_std.create_all_tasks(),
    "winogrande_std": winogrande_std.WinograndeStd,
    "winogrande_mc_std": winogrande_std.WinograndeMCStd,
    "csqa_std": csqa_std.CommonsenseQAStd,
    "csqa_mc_std": csqa_std.CommonsenseQAMCStd,
    "openbookqa_std": openbookqa_std.OpenBookQAStd,
    "openbookqa_mc_std": openbookqa_std.OpenBookQAMCStd,
    "piqa_std": piqa_std.PiQAStd,
    "piqa_mc_std": piqa_std.PiQAMCStd,
    "socialiqa_std": siqa_std.SocialIQAStd,
    "socialiqa_mc_std": siqa_std.SocialIQAMCStd
})
