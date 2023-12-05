import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

MIN_TRANSFORMERS_VERSION = '4.25.1'

# check transformers version
assert transformers.__version__ >= MIN_TRANSFORMERS_VERSION, f'Please upgrade transformers to version {MIN_TRANSFORMERS_VERSION} or higher.'

# init
tokenizer = AutoTokenizer.from_pretrained("togethercomputer/RedPajama-INCITE-7B-Base")
model = AutoModelForCausalLM.from_pretrained("togethercomputer/RedPajama-INCITE-7B-Base", torch_dtype=torch.float16)

# count non-embedding parameters
non_embedding_params = sum([p.numel() for p in model.parameters() if p.dim() > 1])
print(non_embedding_params)