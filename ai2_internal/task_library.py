# Give names to various sets of tasks and configurations to run:

RC20TASKS = "arc_challenge arc_easy boolq copa headqa_en hellaswag logiqa mathqa mrpc openbookqa piqa qnli qqp rte sciq sst wic winogrande wnli wsc"
RC_TASKS_PLUS = RC20TASKS + " social_iqa csqa"
GEN_TASKS = "naturalqs_short_open drop"
SUMMARY_TASKS = "scitldr xsum"
LEGAL_TASKS = "eurlex unfair_tos case_hold:mc"

TASK_SPECS = {
    "rc20_n0_val1000": {"task": RC20TASKS, "limit": 1000, "num-shots": 0},
    "rc_plus_n0_val1000": {"task": RC_TASKS_PLUS, "limit": 1000, "num-shots": 0},
    "summ_n1_val1000": {"task": SUMMARY_TASKS, "limit": 1000, "num-shots": 1,
                             "fewshot-seed": 1234, "num-recorded-inputs": 3},
    "gen_n5_val1000": {"task": GEN_TASKS, "limit": 1000, "num-shots": 5, "fewshot-seed": 1234},
}

