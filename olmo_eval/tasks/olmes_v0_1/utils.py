import re


# Standardized utilities for generating prompts for different types of tasks
def make_mcq_prompt(
    question, choices, question_prefix="Question: ", choices_prefix="", answer_prefix="Answer:"
):
    choice_labels = ["A", "B", "C", "D", "E"]
    # Use prefix space before each answer label to avoid ambiguity in tokenization
    choices_text = "\n".join([f" {label}. {text}" for label, text in zip(choice_labels, choices)])
    prompt = f"{question_prefix}{question}\n{choices_prefix}{choices_text}\n{answer_prefix}"
    return prompt


def make_cloze_prompt(question, question_prefix="Question: ", answer_prefix="Answer:"):
    prompt = f"{question_prefix}{question}\n{answer_prefix}"
    return prompt


GROUP_AGGREGATORS = {"mmlu_mc_std": "mmlu_mc_std_.*", "mmlu_std": "mmlu_std_.*"}


def combine_metrics(tasks):
    # Combine metrics across MCF/CF and aggregate MMLU scores
    task_scores = []
    grouped_task_scores: dict = {}
    for task in tasks:
        task_name = task["task"]
        task_score = task["metrics"]["rc_metrics"]["acc"]
        num_instances = task["num_instances"]
        group_found = False
        for group, regex in GROUP_AGGREGATORS.items():
            if re.match(regex, task_name):
                group_found = True
                grouped_task_scores[group] = grouped_task_scores.get(group, []) + [
                    {"score": task_score, "num_instances": num_instances}
                ]
                break
        if not group_found:
            task_scores.append(
                {"task": task_name, "score": task_score, "num_instances": num_instances}
            )
    for group, scores in grouped_task_scores.items():
        macro_avg = sum(x["score"] for x in scores) / len(scores)
        total_instances = sum(x["num_instances"] for x in scores)
        micro_avg = sum(x["num_instances"] * x["score"] for x in scores) / total_instances
        task_scores.append(
            {
                "task": group,
                "score": macro_avg,
                "score_micro": micro_avg,
                "num_instances": total_instances,
                "num_scores": len(scores),
            }
        )
    combined_rc_mc: dict = {}
    for task_score in task_scores:
        key = task_score["task"].replace("_mc_", "_").replace("_std", "")
        is_mc = "_mc_" in task_score["task"]
        combined_rc_mc[key] = combined_rc_mc.get(key, []) + [
            {"mc": is_mc, "score": task_score["score"]}
        ]
    results = []
    for task, scores in combined_rc_mc.items():
        if len(scores) != 2:
            raise ValueError(f"Did not find 2 scores for {task}!")
        winner = sorted(scores, key=lambda x: -x["score"])[0]
        result = {
            "task": task,
            "score": winner["score"],
            "used_mc": winner["mc"],
            "all_scores": scores,
        }
        results.append(result)
    results.sort(key=lambda x: x["task"])
    return results
