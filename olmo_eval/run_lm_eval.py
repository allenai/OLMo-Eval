import argparse
import json
import logging
import os

import torch
from catwalk.dependencies.lm_eval.utils import simple_parse_args_string
from catwalk.models import MODELS
from catwalk.tasks import TASK_SETS
from catwalk.utils import filter_dict_keys, sanitize
from rjsonnet import evaluate_file
from tango.common.logging import initialize_logging

from olmo_eval.steps import (
    ConstructCatwalkModel,
    ConstructTaskDict,
    PredictAndCalculateMetricsStep,
    ProcessOutputs,
    WriteOutputsAsRows,
)

# Catwalk eval script which is focused on LM models referenced on the fly

_parser = argparse.ArgumentParser()
_parser.add_argument("--config-file", type=str, required=False, help="Config file for evaluation")
_parser.add_argument("--model", type=str, required=False, help="Name of model")
_parser.add_argument("--task", type=str, nargs="+")
_parser.add_argument("--task-file", type=str, help="Jsonl file with task specs")
_parser.add_argument("--split", type=str, default="validation")
_parser.add_argument("--batch-size", type=int, default=32)
_parser.add_argument("--max-batch-tokens", type=int, help="Limit batch size to max tokens")
_parser.add_argument(
    "--model-max-length", type=int, help="Max input length the model should accept"
)
_parser.add_argument("--num-shots", type=int, help="Number of examples in prompt")
_parser.add_argument(
    "--fewshot-seed",
    type=int,
    help="Random seed for picking fixed prompt examples, leave out for varied examples",
)
_parser.add_argument("--limit", type=int, help="Max number of instances for a task")
_parser.add_argument(
    "--full-output-file", type=str, default=None, help="Filename for verbose output"
)
_parser.add_argument("--metrics-file", type=str, default=None, help="Filename for metrics output")
_parser.add_argument(
    "--num-recorded-inputs",
    type=int,
    default=0,
    help="Number of sample model inputs in full output, for sanity checks",
)
_parser.add_argument("--model-path", type=str, help="Explicit path to load model from")
_parser.add_argument("--model-class", type=str, help="Custom Python class for loading model")
_parser.add_argument(
    "--random-subsample-seed",
    type=int,
    help="Random seed for subsampling task instances using limit",
)
_parser.add_argument("--gsheet", type=str, help="Name of Google Sheet for writing results")


def main(args: argparse.Namespace):
    initialize_logging(log_level="INFO")
    logger = logging.getLogger()
    if args.config_file and args.model:
        raise ValueError("Cannot specify both --config-file and --model arguments")
    args_dict = vars(args)
    args_raw = {}
    if args.config_file:
        args_raw = json.loads(evaluate_file(args.config_file))
        for arg, value in args_raw.items():
            args_dict[arg] = value
        if not args_dict["model"]:
            if "::" in args_dict:
                args_dict["model"] = args_dict["model_path"]
            else:
                args_dict["model"] = f"lm::pretrained={args_dict['model_path'].replace('/', '-')}"

    # Some shenanigans to map from run_lm_eval argument structure to ConstructCatwalkModel format
    # TODO fix this mess!
    hf_name = args_dict["model"]
    model_name = hf_name
    model_args = {}
    if hf_name not in MODELS:
        prefix_split = hf_name.split("::", 1)
        model_name = prefix_split[-1]
        model_args = simple_parse_args_string(model_name)
        if "pretrained" not in model_args:
            raise ValueError(f"Unknown model {hf_name}")
        hf_name = model_args["pretrained"]
        model_name = hf_name
        del model_args["pretrained"]
        for key in ["revision", "trust_remote_code"]:
            if args_raw.get(key):
                model_args[key] = args_raw[key]
        if args_dict["model_path"]:
            hf_name = args_dict["model_path"]

    model_obj = ConstructCatwalkModel(cache_results=False).run(
        model_path=hf_name, model_class=args_dict["model_class"], **model_args
    )

    task_args = [
        "limit",
        "split",
        "batch_size",
        "model_max_length",
        "max_batch_tokens",
        "num_shots",
        "fewshot_seed",
        "num_recorded_inputs",
        "random_subsample_seed",
    ]

    default_task_args = {k: v for k, v in args_dict.items() if k in task_args and v is not None}
    if "limit" not in default_task_args:
        default_task_args["limit"] = None  # To override weird default in run_catwalk.py

    # TODO: Should be able to remove these next lines until ConstructTaskDict
    tasks = []
    task_names = set()
    if args_dict["task_file"]:
        with open(args_dict["task_file"], "r") as file:
            for line in file:
                line = line.strip()
                if line and not line.startswith("#"):
                    task_spec = json.loads(line)
                    tasks.append(task_spec)
                    task_names.add(task_spec["name"])

    if args_dict["task"]:
        for task in args_dict["task"]:
            if task in TASK_SETS:
                raise ValueError("Task sets not supported!")
            if task in task_names:
                continue
            task_names.add(task)
            tasks.append({"name": task})

    if not tasks:
        raise ValueError("No tasks specified!")

    # Normalize the tasks, check that they exist, etc
    task_dicts = []
    construct_task_step = ConstructTaskDict(cache_results=False)
    for task in tasks:
        all_task_args = default_task_args.copy()
        all_task_args.update(task)
        task_dicts.append(construct_task_step.run(task_name=task["name"], **all_task_args))

    # Initial loading of model done here for early failures and overrides if needed
    if hasattr(model_obj, "_make_model"):
        logger.info("Loading model...")
        model_cached = model_obj._make_model(
            model_obj.pretrained_model_name_or_path,
            device_map="auto" if torch.cuda.device_count() > 0 else None,
            **model_obj.model_kwargs,
        ).eval()
        if not hasattr(model_cached, "tokenizer"):
            _ = model_obj._make_tokenizer()

    # unconditioned_prompt is taken separately from task_dict, so not on this list
    valid_model_args = [
        "split",
        "limit",
        "batch_size",
        "max_batch_tokens",
        "num_shots",
        "model_max_length",
        "fewshot_seed",
        "num_recorded_inputs",
        "random_subsample_seed",
    ]
    logged_output_keys = [
        "task",
        "model",
        "task_options",
        "metrics",
        "num_instances",
        "processing_time_seconds",
    ]

    verbose_output = []
    beaker_env_variables = {k: v for k, v in os.environ.items() if "BEAKER" in k}
    predict_step = PredictAndCalculateMetricsStep(cache_results=False)
    for task_dict in task_dicts:
        task_name = task_dict["name"]
        logger.info(f"Processing task: {task_name}")
        output = predict_step.run(
            model_obj, task_dict, **filter_dict_keys(task_dict, valid_model_args)
        )
        output = ProcessOutputs().run(output)
        output["model"] = args_dict.get("model", "NA")
        if args_dict.get("model_path"):
            output["model_path"] = args_dict["model_path"]
        if beaker_env_variables:
            output["beaker_info"] = beaker_env_variables
        logger.info(
            f"Results from task {task_name}: {filter_dict_keys(output, logged_output_keys)}"
        )
        logger.info(
            f"First instance details for task {task_name}: {output['instance_predictions'][0]}"
        )
        verbose_output.append(output)
        if args_dict["full_output_file"]:
            logger.info(f"Saving full output in {args_dict['full_output_file']}...")
            with open(args_dict["full_output_file"], "w") as file:
                for d in verbose_output:
                    file.write(json.dumps(sanitize(d)) + "\n")

    num_tasks = len(verbose_output)
    if args_dict["gsheet"]:
        try:
            _ = WriteOutputsAsRows(cache_results=False).run(
                [model_name] * num_tasks,
                verbose_output,
                task_dicts,
                simple_pipeline=True,
                gsheet=args_dict["gsheet"],
            )
        except Exception as e:
            logger.warning(f"Something went wrong when writing Google Sheet: {e}")

    if args_dict["metrics_file"]:
        logger.info(f"Saving metrics in {args_dict['metrics_file']}...")
        with open(args_dict["metrics_file"], "w") as file:
            for d in verbose_output:
                del d["instance_predictions"]  # Destructive
            file.write(json.dumps(sanitize({"metrics": verbose_output})))

    metrics_printed = []
    for d in verbose_output:
        metrics_printed.append(
            f" *** {d['task']} ***  (n = {d['num_instances']})  [{d['task_options']}]"
        )
        metrics = {}
        # Code is a bit confused about nestedness of metrics
        for metric_name, metric in d["metrics"].items():
            if isinstance(metric, dict):
                metrics.update(metric)
            else:
                metrics[metric_name] = metric
        for metric_name, metric in metrics.items():
            metrics_printed.append(f"    {metric_name}: {metric}")
        metrics_printed.append("-----------------")
    logger.info("Overall metrics:\n  " + "\n".join(metrics_printed))


if __name__ == "__main__":
    main(_parser.parse_args())
