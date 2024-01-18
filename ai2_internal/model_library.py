# Give names to various models and associated configurations

MODEL_SPECS = {
    "falcon-rw-7b": {"name": "tiiuae/falcon-rw-7b", "remote-code": True, 'max-batch-tokens': 2*2048}, # Example of running a huggingface model
    "falcon-7b": {"name": "tiiuae/falcon-7b", "remote-code": True, 'max-batch-tokens': 2*2048},
    "llama-7b": {"beaker-model": "oyvindt/llama-7b", "max-batch-tokens": 2048*2}, # Example of running model stored as beaker dataset
    "llama-13b": {"beaker-model": "oyvindt/llama-13b", "max-batch-tokens": 2048*2},
    "llama2-7b": {"name":"llama2-7b", "model-path": "/net/nfs.cirrascale/allennlp/yizhongw/hf_llama2_models/7B", "max-batch-tokens": 2048*2}, # Example of using model path (this one needs to be run on cirrascale)
    "mistral-7b-v0.1": {"name": "mistralai/Mistral-7B-v0.1", "max-batch-tokens": 2*2048},
    "mpt-7b": {"name": "mosaicml/mpt-7b", "checkpoints": [None], "overrides": {"max-batch-tokens": 2048*2}},
    "mpt-7b-instruct": {"name": "mosaicml/mpt-7b-instruct", "checkpoints": [None], "overrides": {"max-batch-tokens": 2048*2}},
    "olmo-7b-v1_5-mix-mitch-ish-step412000": {"beaker-model": "oyvindt/olmo-7b-v1_5-mix-mitch-ish-step412000", "max-batch-tokens": 2*2048},
    "olmo-7b-v1_5-mix-mitch-ish-mosaic-step557000": {"beaker-model": "oyvindt/olmo-7b-v1_5-mix-mitch-ish-mosaic-step557000-hf", "max-batch-tokens": 2*2048},
    "olmo-7b-v1_5-mix-mitch-ish-lumi-2T-step456000": {"beaker-model": "oyvindt/olmo-7b-v1_5-mix-mitch-ish-lumi-2T-step456000", "max-batch-tokens": 2*2048},
    "persimmon-8b-base": {"name": "adept/persimmon-8b-base", 'max-batch-tokens': 2*2048},
    "persimmon-8b-chat": {"name": "adept/persimmon-8b-chat", 'max-batch-tokens': 2*2048},
    "pythia-160m-step140000": {"name": "EleutherAI/pythia-160m", "checkpoint": "step140000"},
    "xgen-7b-4k-base": {"name": "Salesforce/xgen-7b-4k-base", "remote-code": True, 'max-batch-tokens': 2*2048},
    "xgen-7b-8k-inst": {"name": "Salesforce/xgen-7b-8k-inst", "remote-code": True, 'max-batch-tokens': 2*2048},
    "zephyr-7b-beta": {"name": "HuggingFaceH4/zephyr-7b-beta", "max-batch-tokens": 1*2048},
}