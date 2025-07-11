# OLMES LLM Evaluation Standard (v0.1)

## [Important Update] New repository! 
Since the first release of OLMES, we have created a new repository for the standard and associated code, along with expansions to other tasks. The new repository includes code to faithfully reproduce the 
evaluation results in research papers such as
   * **OLMo:** Accelerating the Science of Language Models ([Groeneveld et al, 2024](https://www.semanticscholar.org/paper/ac45bbf9940512d9d686cf8cd3a95969bc313570))
   * **OLMES:** A Standard for Language Model Evaluations ([Gu et al, 2024](https://www.semanticscholar.org/paper/c689c37c5367abe4790bff402c1d54944ae73b2a))
   * **TÜLU 3:** Pushing Frontiers in Open Language Model Post-Training ([Lambert et al, 2024](https://www.semanticscholar.org/paper/T/%22ULU-3%3A-Pushing-Frontiers-in-Open-Language-Model-Lambert-Morrison/5ca8f14a7e47e887a60e7473f9666e1f7fc52de7))
   * **OLMo 2:** 2 OLMo 2 Furious ([Team OLMo et al, 2024](https://arxiv.org/abs/2501.00656))

Please try running the OLMES standard from our new repository:
https://github.com/allenai/olmes !

**More on ongoing efforts:**

OLMES has since been used in supporting evaluation for developing OLMoE (a leading 1B mixture-of-expert model), OLMo 2, TÜLU 3, and is actively used in other Ai2 projects, including research on consistent ranking of models, scaling laws, and building newer open-source models. 

This effort toward an open language model evaluation standard doesn’t just end here. As a community, we can take this evaluation standard further to unify evaluation practices in the field. Since our paper, we have added more tasks to OLMES, covering tasks beyond popular multiple-choice question answering, such as generative and reasoning tasks. We also make efforts toward delineating unseen test suites separate from the development suites, studying various formulations of the generative tasks, standardizing prompt formatting and answer extraction for generative tasks, and experimenting with evaluation setups for instruction-tuned models and the more recent reasoning models (e.g., R1).  The implementations are all openly available in our new repository.



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