import os.path
from typing import Dict

from catwalk.task import Task, ppl_metrics, rc_metrics
from catwalk.tasks.eleuther import EleutherClassificationTask, EleutherTask
from catwalk.tasks.perplexity_jsonl import PerplexityJsonLTask

TASKS_LM: Dict[str, Task] = {
    "squad2": EleutherTask("squad2", eleuther_metrics=True),
    "drop": EleutherTask("drop", eleuther_metrics=True, model_args={"max_gen_toks": 50}),
    "ppl_custom": PerplexityJsonLTask().add_metrics(
        ppl_metrics(primary="ppl_token")
    ),  # TODO: task_rename (easy to fix), files (needs convention).
    "wikitext": EleutherTask("wikitext").add_metrics(ppl_metrics(primary="ppl_token")),
    "piqa": EleutherTask("piqa", ranked_classification=True).add_metrics(rc_metrics(primary="acc_per_token")),
    "mrpc": EleutherClassificationTask(
        "mrpc", answer_options=["no", "yes"], metrics=rc_metrics(primary="acc_raw")
    ),
    "qnli": EleutherClassificationTask(
        "qnli", answer_options=["yes", "no"], metrics=rc_metrics(primary="acc_raw")
    ),
    "qqp": EleutherClassificationTask("qqp", answer_options=["no", "yes"], metrics=rc_metrics(primary="acc_raw")),
    "sst": EleutherClassificationTask(
        "sst", answer_options=["negative", "positive"], metrics=rc_metrics(primary="acc_raw")
    ),
    "rte": EleutherClassificationTask(
        "rte", answer_options=["True", "False"], metrics=rc_metrics(primary="acc_raw")
    ),
    "wnli": EleutherClassificationTask(
        "wnli", answer_options=["False", "True"], metrics=rc_metrics(primary="acc_raw")
    ),
    "boolq": EleutherClassificationTask(
        "boolq", answer_options=["no", "yes"], metrics=rc_metrics(primary="acc_raw")
    ),
    "copa": EleutherTask("copa", ranked_classification=True).add_metrics(rc_metrics(primary="acc_raw")),
    "wic": EleutherClassificationTask("wic", answer_options=["no", "yes"], metrics=rc_metrics(primary="acc_raw")),
    "wsc": EleutherClassificationTask("wsc", answer_options=["no", "yes"], metrics=rc_metrics(primary="acc_raw")),
    "naturalqs_short_open": EleutherTask(
        "naturalqs_short_open", eleuther_metrics=True, model_args={"max_gen_toks": 50}
    ),
    "sciq": EleutherTask("sciq", ranked_classification=True).add_metrics(rc_metrics(primary="acc_raw")),
    "arc_easy": EleutherTask("arc_easy", ranked_classification=True).add_metrics(rc_metrics(primary="acc_uncond")),
    "arc_easy:mc": EleutherTask("arc_easy:mc", ranked_classification=True).add_metrics(
        rc_metrics(primary="acc_raw")
    ),
    "arc_challenge": EleutherTask("arc_challenge", ranked_classification=True).add_metrics(
        rc_metrics(primary="acc_uncond")
    ),
    "arc_challenge:mc": EleutherTask("arc_challenge:mc", ranked_classification=True).add_metrics(
        rc_metrics(primary="acc_raw")
    ),
    # For logiqa the answer choices are shown, but full answer string, so trying acc_raw here
    "logiqa": EleutherTask("logiqa", ranked_classification=True).add_metrics(rc_metrics(primary="acc_raw")),
    "hellaswag": EleutherTask("hellaswag", ranked_classification=True).add_metrics(
        rc_metrics(primary="acc_per_token")
    ),
    "openbookqa": EleutherTask("openbookqa", ranked_classification=True).add_metrics(
        rc_metrics(primary="acc_uncond")
    ),
    "headqa_en": EleutherTask("headqa_en", ranked_classification=True).add_metrics(
        rc_metrics(primary="acc_uncond")
    ),
    "mathqa": EleutherTask("mathqa", ranked_classification=True).add_metrics(rc_metrics(primary="acc_per_token")),
    "winogrande": EleutherTask("winogrande", ranked_classification=True).add_metrics(
        rc_metrics(primary="acc_per_token")
    ),
}
