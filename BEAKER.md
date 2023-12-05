
# Run on beaker

## Option 1: Run full pipeline in an interactive session

* Start an interactive session with the required number of GPUs
* Run as you would run locally, i.e., 

```commandline
tango --settings tango.yml run configs/example_config.jsonnet --workspace gs://my-gs-workspace
```

## Option 2: Run each step as a different beaker job

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

```commandline
tango --settings tango-in-beaker.yml run configs/example_config.jsonnet --workspace gs://my-gs-workspace
```

💡 See [`tango-in-beaker.yml`](tango-in-beaker.yml) for all configurable options.

## Option 3: Run full pipeline in a single beaker job

Note: Use with `llm_eval/run_llm_eval.py`. See details [here](ADVANCED.md#run-without-tango).

Use [beaker-gantry](https://github.com/allenai/beaker-gantry), e.g.,

```commandline
gantry run --gpus 1 --venv base --workspace ai2/lm-eval --cluster ai2/aristo-cirrascale \
   --beaker-image oyvindt/OLMoEvalLatest \
   --env 'HF_DATASETS_CACHE=/net/nfs.cirrascale/aristo/oyvindt/hf_datasets_cache' -- \
   python llm_eval/run_lm_eval.py \
   --model lm::pretrained=EleutherAI/pythia-160m,revision=step140000 \
   --task arc_challenge arc_easy boolq  --split validation \
   --full_output_file /results/predictions.jsonl --metrics_file /results/metrics.json \
   --model_max_length 2048 --max_batch_tokens 4096 --num_recorded_inputs 3 \
   --num_shots 0 --gsheet OLMo-evals-testing
```
or reference a config file, either in `nfs` or a beaker dataset (which can be mounted
in the gantry command).

