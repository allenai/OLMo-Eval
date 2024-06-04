# OLMES-v0.1 LLM Evaluation Standard

This directory contains the data for the OLMES-v0.1 evaluation standard.

All the curated fewshot in-context examples (other than the existing standard MMLU ones) can be found
in the [`std_fewshots.py`](std_fewshot.py) file.

Each task is following the syntax as in v0.3 of 
the [Eleuther LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness), and
can easily be adopted to the v0.4 syntax as well.

For tasks which sample 1000 instances from the dataset, the sampling is done
using a random seed of 1234 using the standard Python random library:

```python
Random(1234).sample(all_instances, 1000)
```

The full list of tasks is listed in the [`task_specs_std.jsonl`](task_specs_std.jsonl) file.

To run the full evaluation using catwalk, for a model from the Hugging Face Hub (e.g., Pythia-1B), use the 
following command:

```commandline
python -m olmo_eval.run_lm_eval --model lm::pretrained=EleutherAI/pythia-1b \
    --task-file olmo_eval/tasks/olmes_v0_1/task_specs_std.jsonl \
    --model-max-length 2048 --max-batch-tokens 4096 \
    --metrics-file metrics.json --full-output-file predictions.jsonl \
    --num-recorded-inputs 3 --model-max-length 2048
```

where max-batch-tokens can be used to control the max number of tokens in a batch, depending on 
model size and GPU memory.

To combine the task scores into final OLMES-v0.1 scores, use the following command:

```commandline
python -m olmo_eval.tasks.olmes_v0_1.combine_scores --metrics-file metrics.json --output-file olmes-scores.json
```

which for the above example of the Pythia-1B model should produce the following output:
```
** Combined OLMES-v0.1 scores **
arc_challenge: 31.4  (CF)
arc_easy     : 63.4  (CF)
boolq        : 56.8  (MCF)
csqa         : 50.9  (CF)
hellaswag    : 48.0  (CF)
mmlu         : 31.1  (CF)
openbookqa   : 40.4  (CF)
piqa         : 68.9  (CF)
socialiqa    : 46.4  (CF)
winogrande   : 52.7  (CF)
--------------------------
average      : 49.0
```
