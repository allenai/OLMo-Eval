import copy
import json
import logging
import math
import os
import time
from collections import defaultdict
from datetime import datetime
from pydoc import locate
from typing import Any, Dict, List, Optional

import pandas as pd
import pytz
from catwalk.dependencies.lm_eval.utils import simple_parse_args_string
from catwalk.model import Model
from catwalk.models import MODELS, add_decoder_only_model
from catwalk.run_lm_eval import get_instances
from catwalk.task import rc_metrics
from catwalk.tasks import TASKS
from catwalk.tasks.tasks_lm import TASKS_LM
from catwalk.utils import guess_instance_id
from tango.step import Step
from tqdm import tqdm

from olmo_eval.tasks.olmes_v0_1.catwalk_tasks import TASKS_STD

try:
    from hf_olmo import *  # noqa: F403
except ImportError:
    pass

logger = logging.getLogger(__name__)

TASKS_LM.update(TASKS_STD)  # Add OLMES-v0.1 tasks


@Step.register("construct-task")
class ConstructTaskDict(Step):
    VERSION = "005"

    def run(
        self,
        task_name: str,
        task_rename: Optional[str] = None,
        eval_data_path: Optional[str] = os.environ.get("EVAL_DATA_PATH"),
        **kwargs,
    ) -> Dict:  # Task:
        task_dict = {"name": task_name}
        try:
            task_obj = TASKS_LM.get(task_name, TASKS.get(task_name))
        except KeyError:
            raise KeyError(f"{task_name} not found")

        # TODO: not clean.
        if hasattr(task_obj, "clone") and "files" in kwargs:
            if eval_data_path:
                files = [os.path.join(eval_data_path, filename) for filename in kwargs["files"]]
            else:
                files = kwargs["files"]
            task_obj = task_obj.clone(
                files=files, detailed_output=kwargs.get("detailed_output", False)
            )
        task_dict["task_obj"] = task_obj

        if task_rename:
            task_dict["name"] = task_rename

        task_dict.update(**kwargs)

        task_dict = self._update_task_metrics(task_dict)
        task_dict = self._update_unconditioned_prompt(task_dict)
        return task_dict

    @classmethod
    def _update_task_metrics(cls, task_dict: Dict) -> Dict:
        task_name = task_dict["name"]
        task_obj = task_dict["task_obj"]
        if "relative_improvement" in task_obj.metrics or "primary_metric" in task_dict:
            kwargs = {}
            if "primary_metric" in task_dict:
                kwargs["primary"] = task_dict["primary_metric"]
                logger.info(f"Overriding metric for {task_name} with rc_metrics ({kwargs})")
            else:
                logger.warning(f"Overriding 'acc' metric for {task_name} with rc_metrics")
            task_obj.metrics = {}
            task_obj.add_metrics(rc_metrics(**kwargs))
        task_dict["task_obj"] = task_obj
        return task_dict

    @classmethod
    def _update_unconditioned_prompt(cls, task_dict: Dict) -> Dict:
        task_name = task_dict["name"]
        task_obj = task_dict["task_obj"]
        if "unconditioned_prompt" not in task_dict:
            if hasattr(task_obj, "inner_task") and hasattr(
                task_obj.inner_task, "unconditioned_prompt"
            ):
                prompt = task_obj.inner_task.unconditioned_prompt()
                logger.info(f"Using unconditioned prompt for {task_name}: '{prompt}'")
                task_dict["unconditioned_prompt"] = prompt
        return task_dict


@Step.register("construct-catwalk-model")
class ConstructCatwalkModel(Step):
    VERSION = "002"
    # CACHEABLE = False

    def run(
        self,
        model_path: str,
        model_class: Optional[str] = None,
        revision: Optional[str] = None,
        trust_remote_code: bool = False,
    ) -> Model:
        if "::" in model_path:
            model = model_path
        else:
            model = f"lm::pretrained={model_path.replace('/', '-')}"

        if model not in MODELS:
            prefix_split = model.split("::", 1)
            model_name = prefix_split[-1]
            # prefix = "" if len(prefix_split) == 1 else prefix_split[0]+"::"
            model_args = simple_parse_args_string(model_name)
            model_args.update({"revision": revision, "trust_remote_code": trust_remote_code})
            if "pretrained" not in model_args:
                raise ValueError(f"Unknown model {model}")
            hf_name = model_args["pretrained"]
            del model_args["pretrained"]
            if model_path:
                hf_name = model_path
            if model_class:
                model_args["model_class"] = locate(model_class)

                # TODO: why do we do this?
                # # Assuming tokenizer will be loaded with model, so fail if trying to load it otherwise
                # model_args['pretrained_tokenizer_name_or_path'] = 'UnknownTokenizer'
                model_args["pretrained_tokenizer_name_or_path"] = model_path

            add_decoder_only_model(model_name, hf_name, **model_args)
        return MODELS[model]


DEFAULT_PREDICTION_KWARGS: Dict[str, Any] = {
    "model_max_length": 2048,
    "max_batch_tokens": 20480,
    "batch_size": 32,
    "limit": 1000,
    "split": "validation",
    "random_subsample_seed": 1234,
}


@Step.register("process-outputs")
class ProcessOutputs(Step):
    VERSION = "002"

    def run(
        self,
        outputs: Dict,
        **kwargs,
    ) -> Dict:
        task_name = outputs["task"]
        new_metrics: defaultdict[str, Dict] = defaultdict(dict)
        if "subdomain" in outputs["instance_predictions"][0]["instance"]:
            sum_logits: Dict[str, float] = {}
            num_tokens: Dict[str, int] = {}
            num_chars: Dict[str, int] = {}
            num_words: Dict[str, int] = {}
            num_bytes: Dict[str, int] = {}
            for instance_prediction in outputs["instance_predictions"]:
                subdomain = instance_prediction["instance"]["subdomain"]
                sum_logits[subdomain] = (
                    sum_logits.get(subdomain, 0)
                    + instance_prediction["prediction"]["model_output"]["sum_logits"]
                )
                num_tokens[subdomain] = (
                    num_tokens.get(subdomain, 0)
                    + instance_prediction["prediction"]["model_output"]["num_tokens"]
                )
                num_chars[subdomain] = (
                    num_chars.get(subdomain, 0)
                    + instance_prediction["prediction"]["model_output"]["num_chars"]
                )
                num_words[subdomain] = (
                    num_words.get(subdomain, 0)
                    + instance_prediction["prediction"]["model_output"]["num_words"]
                )
                num_bytes[subdomain] = (
                    num_bytes.get(subdomain, 0)
                    + instance_prediction["prediction"]["model_output"]["num_bytes"]
                )

            for subdomain in sum_logits:
                new_metrics[f"ppl_token_{task_name}_subdomains"][subdomain] = safe_exp(
                    -sum_logits[subdomain] / max(num_tokens[subdomain], 1)
                )
                new_metrics[f"ppl_char_{task_name}_subdomains"][subdomain] = safe_exp(
                    -sum_logits[subdomain] / max(num_chars[subdomain], 1)
                )
                new_metrics[f"ppl_word_{task_name}_subdomains"][subdomain] = safe_exp(
                    -sum_logits[subdomain] / max(num_words[subdomain], 1)
                )
                new_metrics[f"ppl_byte_{task_name}_subdomains"][subdomain] = safe_exp(
                    -sum_logits[subdomain] / max(num_bytes[subdomain], 1)
                )
                new_metrics[f"bits_per_byte_{task_name}_subdomains"][subdomain] = -sum_logits[
                    subdomain
                ] / (num_bytes[subdomain] * math.log(2))

        outputs["metrics"].update(new_metrics)

        return outputs


@Step.register("predict-and-calculate-metrics")
class PredictAndCalculateMetricsStep(Step):
    VERSION = "003"

    def run(
        self,
        model: Model,
        task_dict: Dict,
        split: Optional[str] = DEFAULT_PREDICTION_KWARGS["split"],
        limit: Optional[int] = DEFAULT_PREDICTION_KWARGS["limit"],
        random_subsample_seed: Optional[int] = DEFAULT_PREDICTION_KWARGS["random_subsample_seed"],
        model_max_length: int = DEFAULT_PREDICTION_KWARGS["model_max_length"],
        max_batch_tokens: int = DEFAULT_PREDICTION_KWARGS["max_batch_tokens"],
        batch_size: int = DEFAULT_PREDICTION_KWARGS["batch_size"],
        **kwargs,
    ) -> Dict:
        """
        :param model: The catwalk model object.
        :param task_dict: The task dict containing the catwalk task object and kwargs.
        :param split: The data split (default="validation").
        :param limit: Number of instances on which to run the predictions (default=1000).
        :param random_subsample_seed: Random seed for subsampling the instances (default=1234).
        :param model_max_length: Max length of the generation (default=2048).
        :param max_batch_tokens: (default=20480).
        :param batch_size: (default=32).
        :param kwargs:
        :return:
        """
        task_name = task_dict["name"]
        task = task_dict["task_obj"]

        start_time = time.time()

        instances = get_instances(task, split, limit, random_subsample_seed)

        predictions = [
            result
            for result in model.predict(
                task,
                instances,
                batch_size=batch_size,
                model_max_length=model_max_length,
                max_batch_tokens=max_batch_tokens,
                unconditioned_prompt=task_dict.get("unconditioned_prompt", None),
                **kwargs,
            )
        ]
        metrics = model.calculate_metrics(
            task, predictions
        )  # this updates the `predictions` object too

        end_time = time.time()

        instance_predictions = self._instance_predictions_map_list(
            instances,
            predictions,
            task_dict.get("keep_instance_fields", None),
            task_dict.get("keep_all_instance_fields_except", None),
        )

        if instance_predictions:
            self.logger.info(
                f"First instance details for task {task_name}: {instance_predictions[0]}"
            )

        task_options = {
            key: val for key, val in task_dict.items() if key not in ["name", "task_obj"]
        }
        model_kwargs = {}
        if hasattr(model, "model_kwargs"):
            model_kwargs.update(model.model_kwargs)
        output = {
            "task": task_dict["name"],
            "task_options": task_options,  # model prediction kwargs,
            "model_kwargs": model_kwargs,
            "metrics": metrics,
            "num_instances": len(instances),
            "processing_time": end_time - start_time,
            "instance_predictions": instance_predictions,
        }

        if hasattr(task, "process_extra_output"):
            output["per_instance"] = output["instance_predictions"]
            del output["instance_predictions"]
            output = task.process_extra_output(output)
            output["instance_predictions"] = output["per_instance"]
            del output["per_instance"]

        return output

    @classmethod
    def _instance_predictions_map_list(
        cls,
        instances,
        predictions,
        keep_instance_fields: Optional[List] = None,
        keep_all_instance_fields_except: Optional[List] = None,
    ) -> List:
        instance_predictions = []

        for idx, (instance, pred) in enumerate(zip(instances, predictions)):
            instance_id = guess_instance_id(instance, idx=idx)  # dict

            if keep_instance_fields or keep_all_instance_fields_except:
                assert (
                    keep_instance_fields is None or keep_all_instance_fields_except is None
                ), "Can't use both keep_instance_fields and keep_all_instance_fields_except"
                for field in instance:
                    if keep_instance_fields and field not in keep_instance_fields:
                        continue
                    if keep_all_instance_fields_except and field in keep_all_instance_fields_except:
                        continue
                    instance_id[field] = instance[field]

            prediction = pred.get("prediction", pred)

            model_input = None
            # Move model_input from prediction if need be
            if "model_input" in pred:
                model_input = pred["model_input"]
                if "model_input" in prediction:
                    del prediction["model_input"]

            instance_pred = {"instance": instance_id, "prediction": prediction}
            if model_input is not None:
                instance_pred["model_input"] = model_input
            instance_predictions.append(instance_pred)

        return instance_predictions


@Step.register("write-outputs-as-rows")
class WriteOutputsAsRows(Step):
    VERSION = "001"

    def run(
        self,
        models: List[str],
        outputs: List[Dict],
        prediction_kwargs: List[Dict],
        simple_pipeline: bool = False,
        gsheet: Optional[str] = None,
    ) -> List:
        tsv_outputs = []
        for idx, d in enumerate(outputs):
            model = models[idx]
            pred_kwargs = copy.deepcopy(DEFAULT_PREDICTION_KWARGS)
            pred_kwargs.update(prediction_kwargs[idx])
            row = {}
            row["date"] = datetime.now(tz=pytz.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            row["model"] = model
            row["model_kwargs"] = d["model_kwargs"]
            row["full_model"] = f"lm::pretrained={model}"
            metrics_dict = list(d["metrics"].values())[0]

            # TODO: Very hacky.
            if "primary_metric" not in metrics_dict:
                primary_metric = "f1"
            else:
                primary_metric = metrics_dict["primary_metric"]

            row["task"] = d["task"]
            row["primary_metric"] = primary_metric
            row["metric"] = metrics_dict[primary_metric]
            row["processing_time"] = d["processing_time"]
            row["num_instances"] = d["num_instances"]
            if not simple_pipeline:
                row["tango_workspace"] = self.workspace.url
                row["tango_step"] = self.unique_id

            row.update(pred_kwargs)
            if simple_pipeline:
                row["all_metrics"] = json.dumps(metrics_dict)
                row["beaker_id"] = d.get("beaker_info", {}).get("BEAKER_EXPERIMENT_ID", "")
                if "name" in row:
                    del row["name"]  # Stored as "task"
                if "task_obj" in row:
                    del row["task_obj"]
                if "num_recorded_inputs" in row:
                    del row["num_recorded_inputs"]

            tsv_outputs.append(row)

        if gsheet:
            write_to_gsheet(gsheet, tsv_outputs)

        return tsv_outputs


@Step.register("write-outputs-as-rows-multiple-metrics")
class WriteOutputsAsRowsMultipleMetrics(Step):
    VERSION = "001"

    def run(
        self,
        models: List[str],
        outputs: List[Dict],
        prediction_kwargs: List[Dict],
        gsheet: Optional[str] = None,
    ) -> Dict[str, List[Dict]]:
        per_metric_type_tsv_outputs: Dict[str, List[Dict]] = {}
        for idx, d in enumerate(outputs):
            model = models[idx]
            pred_kwargs = copy.deepcopy(DEFAULT_PREDICTION_KWARGS)
            pred_kwargs.update(prediction_kwargs[idx])
            tsv_outputs: List[Dict] = []
            for metric_type_name, metrics_dict in d["metrics"].items():
                row = {}
                row["date"] = datetime.now(tz=pytz.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
                row["model"] = model
                row["model_kwargs"] = d["model_kwargs"]
                row["full_model"] = f"lm::pretrained={model}"
                row["task"] = d["task"]
                row["processing_time"] = d["processing_time"]
                row["num_instances"] = d["num_instances"]
                row["tango_workspace"] = self.workspace.url
                row["tango_step"] = self.unique_id
                for metric_name in metrics_dict:
                    row[metric_name] = metrics_dict[metric_name]

                row.update(pred_kwargs)
                per_metric_type_tsv_outputs[metric_type_name] = per_metric_type_tsv_outputs.get(
                    metric_type_name, []
                ) + [row]
            if "extra_output" in d and "token_count_avg_logits_by_domain" in d["extra_output"]:
                for subdomain, token2countNLogit in tqdm(
                    d["extra_output"]["token_count_avg_logits_by_domain"].items(),
                    desc="reading token_count_avg_logits_by_domain",
                ):
                    for token, countNLogit in token2countNLogit.items():
                        row = {}
                        task = d["task"]
                        row["model"] = model
                        row["split"] = pred_kwargs["split"]
                        if "revision" in d["model_kwargs"]:
                            row["revision"] = d["model_kwargs"]["revision"]
                        row["subdomain"] = subdomain
                        row["token"] = token
                        row["count"] = countNLogit[0]
                        row["avg_logits"] = countNLogit[1]
                        if f"{task}_token_count_avg_logits" not in per_metric_type_tsv_outputs:
                            per_metric_type_tsv_outputs[f"{task}_token_count_avg_logits"] = []
                        per_metric_type_tsv_outputs[f"{task}_token_count_avg_logits"].append(row)

        if gsheet:
            for metric_type_name, tsv_outputs in per_metric_type_tsv_outputs.items():
                # skip _token_count_avg_logits because it's too big
                if metric_type_name.endswith("_token_count_avg_logits"):
                    continue
                write_to_gsheet(gsheet, tsv_outputs, sheet_title=metric_type_name)

        return per_metric_type_tsv_outputs


@Step.register("save-write-outputs-as-rows-multiple-metrics-as-file")
class SaveWriteOutputsAsRowsMultipleMetricsAsFile(Step):
    VERSION = "001"

    def run(self, write_outputs: Dict[str, List[Dict]], output_dir: str) -> None:
        import smart_open

        if output_dir is None:
            logger.info("output_file is None, skipping save to file")
            return
        transport_params = None
        if output_dir.startswith("s3://"):
            import boto3

            session = boto3.Session(
                aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
                aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
                aws_session_token=os.environ["AWS_SESSION_TOKEN"],
            )
            client = session.client("s3")
            transport_params = dict(client=client)
        for table_name in write_outputs:
            output_file = os.path.join(output_dir, table_name + ".jsonl.gz")
            with smart_open.open(output_file, "wb", transport_params=transport_params) as f:
                for row in tqdm(write_outputs[table_name], desc=f"writing {table_name} to file"):
                    f.write(json.dumps(row).encode())
                    f.write(b"\n")


def write_to_gsheet(gsheet: str, rows: List[Dict], sheet_title: str = "Sheet1"):
    import pygsheets

    # make rows into dataframe
    new_df = pd.DataFrame(rows)

    client = pygsheets.authorize(service_account_json=os.environ["GDRIVE_SERVICE_ACCOUNT_JSON"])
    sheet = client.open(gsheet)

    # make sheet if doesn't exist
    if sheet_title in [s.title for s in sheet.worksheets()]:
        worksheet = sheet.worksheet_by_title(sheet_title)
    else:
        sheet.add_worksheet(rows=new_df.shape[0], cols=new_df.shape[1], title=sheet_title)
        worksheet = sheet.worksheet_by_title(sheet_title)
    current_df = worksheet.get_as_df()
    new_df = pd.concat([current_df, new_df])
    worksheet.set_dataframe(new_df, (1, 1), nan="")


def safe_exp(x):
    try:
        ans = math.exp(x)
    except OverflowError:
        ans = 1e30
        logger.warning(f"OverflowError when computing math.exp({x})")
    return ans
