# Paloma

In addition to the dataset hosted here, Paloma introduces guidelines for making perplexity results comparable across models and code that implements these guidelines with specific experimental controls. This page will walk you through how to apply these standards to your experiments.

Whether you are just evaluating an off-the-shelf model or preparing to conduct your own pretraining experiment from scratch, we recommend that you employ as much of our standardized code as possible to ensure the greatest level comparability with existing results.

Links:

[Data](https://huggingface.co/datasets/allenai/paloma)

## Setup
Start by following the installation instructions for this repo in this [readme](../README.md).

Then, download the PALOMA dataset from HF hub:

```commandline
huggingface-cli login
git lfs install
git clone https://huggingface.co/datasets/allenai/paloma
```

Finally, export the path to this data when running the pipeline:

```commandline
export EVAL_DATA_PATH=/path/to/paloma
```

## Running evaluation
After following the setup instructions above, you can make an evaluation configuration based on our template [here](../configs/example_paloma_config.jsonnet). This is designed to work with any model hosted on the HuggingFace hub. Just specify the name of the model on the hub and any revisions (i.e., checkpoints) that you want results over. Read the comments in the configuration with the ❗ symbol for more information about details you may need to fill in.  Finally make sure to set an output directory for `output_dir` where you want the job to output your results. 

Now you can run your evaluation job locally with the following command (from the root of this repo):
```
tango --settings tango.yml run configs/example_paloma_config.jsonnet --workspace my-eval-workspace
```

## Pretraining your model
If you are pretraining from scratch, we recommend you adopt several experimental controls that will allow the greatest level of comparability for your results. In this section we detail how you can accomplish these experimental controls.

### Decontaminating your pretraining data
Our decontamination approach is implemented in the Dolma Tooling repo. This will allow you to remove any document from any your pretraining data that is contaminated with respect to the Paloma.

To do this please follow the instructions [here](https://github.com/allenai/dolma/blob/decon-instructions/docs/paloma_decontamination.md) to decontaminate your own pretraining data.

### Fixing the training data order
Our approach for fixing the training data order requires the use of [the same OLMo training code](https://github.com/allenai/OLMo/tree/1f2f02052d2a5ecba82ff45bbfc731651b1e7d29) that we employ to train our 1B parameter baselines. Contemporary LMs train on instances that are maximum sequence length concatenations of training documents, so we must fix the order of concatenated instances. We do this by fixing the tokenization, maximum sequence length, and random seed, as well as providing dataloading code where order is invariant to number of devices.

### Fixing the vocabulary
If you do not investigate changes in vocabulary we recommend standardized vocabulary to enable the greatest level of comparability. The vocabulary we employ in our baseline models is available from the tokenizer hosted on HuggingFace hub as `allenai/gpt-neox-olmo-dolma-v1_5`. 

## Citation

```bibtex
@article{Magnusson2023PalomaAB,
  title={Paloma: A Benchmark for Evaluating Language Model Fit},
  author={Ian Magnusson and Akshita Bhagia and Valentin Hofmann and Luca Soldaini and A. Jha and Oyvind Tafjord and Dustin Schwenk and Pete Walsh and Yanai Elazar and Kyle Lo and Dirk Groeneveld and Iz Beltagy and Hanna Hajishirzi and Noah A. Smith and Kyle Richardson and Jesse Dodge},
  journal={ArXiv},
  year={2023},
  volume={abs/2312.10523},
  url={https://api.semanticscholar.org/CorpusID:266348815}
}
```
