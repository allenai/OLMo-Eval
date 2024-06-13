# OLMES LLM Evaluation Standard (v0.1)

## Introduction

OLMES (Open Language Model Evaluation Standard) is a set of principles and associated tasks, 
for evaluating large language models (LLMs). See our paper [OLMES: A Standard for Language Model Evaluations (Gu et al, 2024)](https://www.semanticscholar.org/paper/OLMES%3A-A-Standard-for-Language-Model-Evaluations-Gu-Tafjord/c689c37c5367abe4790bff402c1d54944ae73b2a) for more details.

The current version includes:

   * Standardized formatting of dataset instances
   * Curated, few-shot in-context examples for each task
   * Evaluate both multiple-choice (MCF) and cloze-form (CF) formulations and use maximum score
   * Standardized probability normalization schemes for CF
   * Prescribed implementations details:
       * Sampling of 1000 instances for each task if more than 1500
       * Use test split if labels are available, otherwise use validation split
       * For MMLU use macro average over tasks
       * Restrict to maximum 2048 tokens per input


The full list of tasks in the v0.1 standard is as follows (more tasks, including generative CoT, to come):

   * ARC-Challenge
   * ARC-Easy
   * BoolQ
   * CommonsenseQA
   * HellaSwag
   * MMLU
   * OpenBookQA
   * PIQA
   * Social IQa
   * WinoGrande

with details for each task provided in the [`task_specs_std.jsonl`](task_specs_std.jsonl) file. Each task has an
associated source file following the v0.3 format in 
the [EleutherAI LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) to facilitate
straightforward integration in other evaluation code bases.

All the curated fewshot in-context examples (other than the existing standard MMLU ones) can be found
in the [`std_fewshots.py`](std_fewshot.py) file.


## Sampling details

For tasks which sample 1000 instances from the dataset, the sampling is done
using a random seed of 1234 using the standard Python random library:

```python
Random(1234).sample(all_instances, 1000)
```


## Example usage

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

## [Citation](https://arxiv.org/abs/2406.08446)

```
@misc{gu2024olmes,
      title={OLMES: A Standard for Language Model Evaluations}, 
      author={Yuling Gu and Oyvind Tafjord and Bailey Kuehl and Dany Haddad and Jesse Dodge and Hannaneh Hajishirzi},
      year={2024},
      eprint={2406.08446},
      archivePrefix={arXiv}
}
```