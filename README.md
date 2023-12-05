
# ai2-llm-eval

The `llm_eval` framework is a way to run evaluation pipelines for language models on NLP tasks. 
The codebase is extensible and contains `task_sets` and example configurations, which run a series
of [`tango`](https://github.com/allenai/tango) steps for computating the model outputs and metrics.


Using this pipeline, you can evaluate _m_ models on _t_ task_sets, where each task_set consists of one or more individual tasks.
Using task_sets allows you to compute aggregate metrics for multiple tasks. The optional `google-sheet` integration can be used
for reporting.

The pipeline is built using [ai2-tango](https://github.com/allenai/tango) and [ai2-catwalk](https://github.com/allenai/catwalk).

## Installation

```commandline
conda create -n eval-pipeline python=3.10
conda activate eval-pipeline
pip install -e .
```

## Quickstart


### Config file

The current `task_sets` can be found at [configs/task_sets](configs/task_sets). In this example, we run `gen_tasks` on `EleutherAI/pythia-1b`.

The example config is [here](configs/example_config.jsonnet). It looks like this:


```jsonnet
// Imports

//...

// Models to evaluate

local models = [
    {
        model_path: "EleutherAI/pythia-1b",
        revision: "step140000", //❗Specify checkpoint if needed
        gpus_needed: 1,
        //❗Task sets contain default values for prediction_kwargs. These can be overriden for each model here.
        prediction_kwargs: {
            model_max_length: 2048,
            max_batch_tokens: 20480,
        }
    },

];

// Tasks to evaluate

local task_sets = [
    gen_tasks.task_set,
    // Can specify other task sets here
];


{
    steps: utils.create_pipeline(models, task_sets) // Creates the set of steps to run for each model-task_set combination.
}
```

### Run the pipeline


The configuration can be run as follows:


```commandline
tango --settings tango.yml run configs/evaluation_template.jsonnet --workspace my-eval-workspace
```

This executes all the steps defined in the config, and saves them in a local `tango` workspace called `my-eval-workspace`. If you add a new task_set or model to your config and run the same command again, it will reuse the previous outputs, and only compute the new outputs.

The output should look like this:

TODO: add picture.


### Load pipeline output

```python
from tango import Workspace
workspace = Workspace.from_url("local://my-eval-workspace")
result = workspace.step_result("combine-all-outputs")
```

Load individual task results with per instance outputs

```python
result = workspace.step_result("outputs_pythia-1bstep140000_gen_tasks_drop")
```


