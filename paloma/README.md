# Paloma

The Paloma benchmark makes use of this repo to run evaluation inference. This readme will explain everything you need to know to get results on Paloma and make a submission to our benchmark.

Links:

[Data](https://huggingface.co/datasets/allenai/paloma)

## Getting existing results from the benchmark
Paloma is first and foremost a suite of results from the research community organized by comprability. These are formated as *.jsonl.gz files recording the perplexity per domain over our 585 domains as well as additional metrics discussed in our paper. These are files are the same type of results that are output by running the code in this repo for a given model.

We are also building out a website to allow interactive inspection of these multi-dimensional results. Until then please contact us by emailing the first author of Paloma if you would like access to the raw benchmark results.

So far the models evaluated by the benchmark are the 6 baseline 1B parameter models that we release with Paloma as well as `EleutherAI/pythia-160m`, `EleutherAI/pythia-1B`, and `EleutherAI/pythia-6.9b`.

## Setup
Start by following the installation instructions for this repo in this [readme](../README.md).

Then follow the instructions in this [readme](eval_data/README.md) to obtain and set up the evaluation data.

## Running evaluation
After following the setup instructions above, you can make an evaluation configuration based on our template [here](../configs/example_paloma_config.jsonnet). This is designed to work with any model hosted on the HuggingFace hub. Just specify the name of the model on the hub and any revisions (i.e., checkpoints) that you want results over. Read the comments in the configuration with the ❗ symbol for more information about details you may need to fill in.  Finally make sure to set an output directory for `output_dir` where you want the job to output your results. 

Now you can run your evaluation job locally with the following command (from the root of this repo):
```
tango --settings tango.yml run configs/example_paloma_config.jsonnet --workspace my-eval-workspace
```

## Pretraining your model
Note that if you want to make a submission to our benchmark you must choose whether you will opt in to several experimental controls that will allow your submission to be marked for the greatest level of comparability. In this section we detail how you can accomplish these experimental controls.

### Decontaminating your pretraining data
Our decontamination approach is implemented in the Dolma Tooling repo. This will allow you to remove any document from any your pretraining data that is contaminated with respect to the Paloma.

To do this please follow the instructions [here](https://github.com/allenai/dolma/blob/decon-instructions/docs/paloma_decontamination.md) to decontaminate your own pretraining data.

### Fixing the training data order
Our approach for fixing the training data order requires the use of the same training code that we employ to train our 1B parameter baselines. This training code cannot yet be released as it is being developed for a separate, ongoing project. When that code is released we will update our instructions here to enable use of this experimental control. If you wish to use this experimental control before then, please feel free to reach out to the first author of Paloma.

### Fixing the vocabulary
We ask that submissions that do not investigate changes in vocabulary opt in to our standardized vocabulary to enable the greatest level of comprability. That vocabulary is available from the tokenizer hosted on HuggingFace hub as `allenai/gpt-neox-olmo-dolma-v1_5`. 

## Making a submission
At present we are building out an automatic submission process that will soon be available. Until then please reach out to us by emailing the first author of Paloma, if you would like to submit results to the benchmark.