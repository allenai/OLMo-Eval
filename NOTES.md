
# Evaluation

TODO: Remove these notes!!

We use tango and catwalk to build the pipeline.
The catwalk code exists [here](https://github.com/allenai/catwalk/tree/olmo-eval).


## Setup

* Create a [GitHub personal access token](https://github.com/settings/tokens/new) with at least the "repo" scope.

```commandline
conda create -n eval-env python=3.10
conda activate eval-env
pip install -e '.[dev]'
```

```commandline
gcloud auth login
gcloud auth application-default login
gcloud config set project ai2-olmo
```

```commandline
export HF_DATASETS_CACHE=<path>  # optional

# optional; if using olmo models with hf_olmo.
export GLOBAL_MODEL_DIR=/net/nfs.cirrascale/allennlp/akshitab/eval_models
 
# optional; if evaluating on perplexity datasets
export EVAL_DATA_PATH=/net/nfs.cirrascale/allennlp/akshitab/eval_data

# optional; if using checkpoints stored on S3
export AWS_ACCESS_KEY_ID=<access key>
export AWS_SECRET_ACCESS_KEY=<secret key>
```

#### If you want results to (also) be written to a Google Sheet

<details>
    <summary>Setting up the service account</summary>

#### ❗ NOTE: This is a one-time thing and has already been done for the ai2-olmo project.
* Authorization set up https://pygsheets.readthedocs.io/en/stable/authorization.html
</details>

* Share the google sheet with `olmo-eval@ai2-olmo.iam.gserviceaccount.com`.
* Create API json key and download from [here](https://console.cloud.google.com/iam-admin/serviceaccounts/details/116966732244811673427/keys?project=ai2-olmo).

```commandline
export GDRIVE_SERVICE_ACCOUNT_JSON=$(cat downloaded_credentials_file.json)
```

* Uncomment `local gsheet=...` in the evaluation config, and set it to the name of the google sheet.

### Run the pipeline

* Login to a machine with the required number of GPUs (cloud machine / beaker interactive session / beaker batch job).
* Update [tango.yml](tango.yml) (the fields that should be updated are marked).
```commandline
tango --settings tango.yml run configs/evaluation_template.jsonnet
```

* If you want to save results to google cloud buckets, so that they are shareable (also configurable in [tango.yml](tango.yml)):

```commandline
tango run configs/evaluation_template.jsonnet \
    -w gs://olmo-evaluation-runs/your-eval-workspace
```

### Accessing outputs of steps

All intermediate and final results will be saved to the specified workspace, and can be accessed as follows:

```python
from tango import Workspace
workspace = Workspace.from_url("gs://olmo-evaluation-runs/your-eval-workspace")
result = workspace.step_result("combine-all-outputs")
```

If you specify `gsheet` in your config, final results will also be appended to the google sheet.

### If you want to run the evaluation steps on beaker individually

This will run each step of the pipeline individually in different beaker experiments. Compute resources required for each step will be provisioned separately.


<details>
    <summary>Creating a beaker image (❗Update each time the catwalk / tango version is updated).</summary>

This is done so that each individual step does not need to install catwalk and tango, and other libraries, which can be slow.

[Reference](https://beaker-docs.apps.allenai.org/interactive/images.html#building-custom-images)

```commandline
beaker session create --gpus 1 --image beaker://ai2/cuda11.5-cudnn8-dev-ubuntu20.04 --bare --save-image
conda create -n eval-env python=3.10
conda activate eval-env
pip install -e '.[dev]'
exit
beaker image rename <image-id> llm_eval_image
```
</details>

* Update [tango-in-beaker.yml](tango-in-beaker.yml) (the fields that should be updated are marked).

```commandline
export GITHUB_TOKEN="<your token>"  # Needed for beaker to clone the repo.
```

```commandline
tango --settings tango-in-beaker.yml run configs/evaluation_template.jsonnet
```


### Running simple pipeline as single beaker job

The `olmo_eval/run_lm_eval.py` script provides a way to run an evaluation as a single beaker
job with associated result set. Arguments can be provided in a config file, an example is found 
in `configs/run_lm_eval_example.jsonnet`, or as direct arguments (see documentation in script). E.g.,

```commandline
python -m olmo_eval.run_lm_eval --config_file configs/run_lm_eval_example.jsonnet
```
or
```commandline
python -m olmo_eval.run_lm_eval --model lm::pretrained=EleutherAI/pythia-160m,revision=step140000 \
    --task arc_challenge arc_easy  --split validation \
    --full_output_file predictions.jsonl --metrics_file metrics.json --model_max_length 2048 \
    --max_batch_tokens 4096 --num_recorded_inputs 3 --num_shots 0 --gsheet OLMo-evals-testing
```

To launch a job in beaker, it's easiest to use [beaker-gantry](https://github.com/allenai/beaker-gantry), e.g.,
```commandline
gantry run --gpus 1 --venv base --workspace ai2/lm-eval --cluster ai2/aristo-cirrascale \
   --beaker-image oyvindt/OLMoEvalLatest \
   --env 'HF_DATASETS_CACHE=/net/nfs.cirrascale/aristo/oyvindt/hf_datasets_cache' -- \
   python olmo_eval/run_lm_eval.py \
   --model lm::pretrained=EleutherAI/pythia-160m,revision=step140000 \
   --task arc_challenge arc_easy boolq  --split validation \
   --full_output_file /results/predictions.jsonl --metrics_file /results/metrics.json \
   --model_max_length 2048 --max_batch_tokens 4096 --num_recorded_inputs 3 \
   --num_shots 0 --gsheet OLMo-evals-testing
```
or reference a config file, either in `nfs.cirrascale` or a beaker dataset (which can be mounted
in the gantry command).

### Troubleshooting

If some error causes the workspace to go into a bad state (i.e., you get errors that say step should not be in completed state, etc.), you can clear the workspace with

```commandline
python scripts/empty_workspace.py olmo-evaluation-runs/your-eval-workspace
```

## Contributing

### Creating an evaluation config

The evaluation pipeline is run as a cross product of models that need to be evaluated, and task sets.

1. Huggingface model names work directly. For olmo model paths, ensure that they are present in a `gs://` or `s3://` location.
2. Copy `configs/evaluation_template.jsonnet` to `configs/experiment_YYYY_MM_DD.jsonnet`
3. Add models and choose relevant task sets from [configs/task_sets](configs/task_sets).

### Adding new task sets

A task set is of the form:

```jsonnet
{
    name: "<Name of the task set>",
    tasks: [
        {
            task_name: "<One of the tasks present in `TASKS_LM` or `TASKS`>",
            task_kwargs: "<task-specific kwargs (See eval_suite for examples)>",
            prediction_kwargs: "<kwargs on how to evaluate the model on this task>"
        }
    ]
}
```

1. Add new task sets under [configs/task_sets](configs/task_sets). Current full sets include:
   * `gen_tasks.libsonnet`
   * `eval_suite_ppl_val_v2_small.libsonnet`
   * `rc20_tasks.libsonnet`
   * `summary_tasks.libsonnet`

2. The list of potential tasks can be seen by running `python evaluation/see_available_tasks.py`. 


### Adding a new dataset to our perplexity eval set

1. Add the new set under our current ppl data at `$EVAL_DATA_PATH`.
2. Add the name of the folder to [eval_suite_ppl_val_v2_small.libsonnet](configs/task_sets/eval_suite_ppl_val_v2_small.libsonnet).

### Adding tasks already present in catwalk

1. See [gen_tasks.libsonnet](configs/task_sets/gen_tasks.libsonnet) for a simple example.

### Adding new tasks to catwalk

(TODO: catwalk needs better documentation on adding new tasks).
1. See examples [here](https://github.com/allenai/catwalk/tree/olmo-eval/catwalk/tasks).
2. Add newly created tasks to [TASKS_LM](https://github.com/allenai/catwalk/blob/olmo-eval/catwalk/tasks/tasks_lm.py)
 or [TASKS](https://github.com/allenai/catwalk/blob/olmo-eval/catwalk/tasks/__init__.py).
