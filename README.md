
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

The current `task_sets` can be found at [configs/task_sets](configs/task_sets). In this example, we run `gen_tasks` on `EleutherAI/pythia-1b`. The example config is [here](configs/example_config.jsonnet).

The configuration can be run as follows:

```commandline
tango --settings tango.yml run configs/evaluation_template.jsonnet --workspace my-eval-workspace
```

This executes all the steps defined in the config, and saves them in a local `tango` workspace called `my-eval-workspace`. If you add a new task_set or model to your config and run the same command again, it will reuse the previous outputs, and only compute the new outputs.

The output should look like this:

<img width="1886" alt="Screen Shot 2023-12-04 at 9 22 35 PM" src="https://github.com/allenai/ai2-llm-eval/assets/6500683/14a74e61-75d8-470c-8bde-12e35c38c44a">

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


## Advanced

1. [Save output to google sheet](ADVANCED.md)

