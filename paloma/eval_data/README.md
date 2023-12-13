# Local evaluation data

This directory is used as a temporary work around until we implement perplexity inference with HF hub datasets.

To use Paloma with this pipeline you will need to first download the data from HF hub (install git lfs first in necessary):
```
huggingface-cli login
git lfs install
git clone https://huggingface.co/datasets/olmo-friends/paloma
```

Then when you run the pipe line you will first need to export the path to this data
```
cd paloma
export EVAL_DATA_PATH=$(pwd)
```
