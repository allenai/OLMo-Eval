"""
Safety suites for thinking and instruct models, and wildguard vs openai models

Example commands to run:

For a reasoning model:
olmo-eval beaker launch  \
    --harness default   \
    -o provider.kwargs.tensor_parallel_size=2   \
    -o 'metrics.collect_gpu=true'   \
    -o auxiliary_providers.sr_judge.kind=vllm_server   \
    -o auxiliary_providers.sr_judge.model=google/gemma-2b   \
    -o auxiliary_providers.sr_judge.tokenizer=qylu4156/strongreject-15k-v1   \
    -o auxiliary_providers.sr_judge.kwargs.enable_lora=true   \
    -o auxiliary_providers.sr_judge.kwargs.lora_modules=\
    [strongreject=qylu4156/strongreject-15k-v1] \
    -o auxiliary_providers.sr_judge.kwargs.gpu_memory_utilization=0.2  \
    -o auxiliary_providers.sr_judge.kwargs.add_bos_token=true \
    -o auxiliary_providers.wg_judge.kind=vllm_server \
    -o auxiliary_providers.wg_judge.model=allenai/wildguard \
    -o auxiliary_providers.wg_judge.kwargs.add_bos_token=true \
    -o scoring_concurrency=4   \
    -m allenai/Olmo-3-7B-Think   \
    -t "safety_thinking@high" \
    -w "ai2/WORKSPACE"   \
    -B "ai2/BUDGET"   \
    --cluster h100

For an instruct model:
olmo-eval beaker launch  \
    --harness default   \
    -o 'metrics.collect_gpu=true'   \
    -o auxiliary_providers.sr_judge.kind=vllm_server   \
    -o auxiliary_providers.sr_judge.model=google/gemma-2b   \
    -o auxiliary_providers.sr_judge.tokenizer=qylu4156/strongreject-15k-v1   \
    -o auxiliary_providers.sr_judge.kwargs.enable_lora=true   \
    -o auxiliary_providers.sr_judge.kwargs.lora_modules=\
    [strongreject=qylu4156/strongreject-15k-v1] \
    -o auxiliary_providers.sr_judge.kwargs.gpu_memory_utilization=0.2  \
    -o auxiliary_providers.sr_judge.kwargs.add_bos_token=true \
    -o auxiliary_providers.wg_judge.kind=vllm_server \
    -o auxiliary_providers.wg_judge.model=allenai/wildguard \
    -o auxiliary_providers.wg_judge.kwargs.add_bos_token=true \
    -o scoring_concurrency=4   \
    -m allenai/Olmo-3-7B-Instruct   \
    -t "safety_instruct@high" \
    -w "ai2/WORKSPACE"   \
    -B "ai2/BUDGET"   \
    --cluster h100

For a base model:
olmo-eval beaker launch  \
    --harness default   \
    -o 'metrics.collect_gpu=true'   \
    -o auxiliary_providers.sr_judge.kind=vllm_server   \
    -o auxiliary_providers.sr_judge.model=google/gemma-2b   \
    -o auxiliary_providers.sr_judge.tokenizer=qylu4156/strongreject-15k-v1   \
    -o auxiliary_providers.sr_judge.kwargs.enable_lora=true   \
    -o auxiliary_providers.sr_judge.kwargs.lora_modules=\
    [strongreject=qylu4156/strongreject-15k-v1] \
    -o auxiliary_providers.sr_judge.kwargs.gpu_memory_utilization=0.2  \
    -o auxiliary_providers.sr_judge.kwargs.add_bos_token=true \
    -o auxiliary_providers.wg_judge.kind=vllm_server \
    -o auxiliary_providers.wg_judge.model=allenai/wildguard \
    -o auxiliary_providers.wg_judge.kwargs.add_bos_token=true \
    -o scoring_concurrency=4   \
    -m allenai/Olmo-3-1025-7B   \
    -t "safety_base@high" \
    -w "ai2/WORKSPACE"   \
    -B "ai2/BUDGET"   \
    --cluster h100

On the OpenAI harness:
olmo-eval beaker launch  \
    --harness default   \
    -o 'metrics.collect_gpu=true'   \
    -o scoring_concurrency=4   \
    -m allenai/Olmo-3-7B-Instruct   \
    -t "safety_openai@high" \
    -w "ai2/WORKSPACE"   \
    -B "ai2/BUDGET"   \
    --cluster h100


"""

from olmo_eval.evals.suites.registry import AggregationStrategy, make_suite

SAFETY_TASKS = ["do_anything_now", "harmbench", "wildguardtest", "wildjailbreak", "xstest"]

MCQ_TASKS = ["wmdp", "bbq"]

make_suite(
    "safety_thinking",
    (
        *(f"{task}:wg_judge_thinking" for task in SAFETY_TASKS),
        *(f"{task}:mcq" for task in MCQ_TASKS),
        "strongreject:sr_judge_thinking",
    ),
    aggregation=AggregationStrategy.AVERAGE,
    description="Safety evals for posttrained reasoning models with a wildguard judge",
)

make_suite(
    "safety_instruct",
    (
        *(f"{task}:wg_judge" for task in SAFETY_TASKS),
        *(f"{task}:mcq" for task in MCQ_TASKS),
        "strongreject:sr_judge",
    ),
    aggregation=AggregationStrategy.AVERAGE,
    description="Safety evals for posttrained instruct models with a wildguard judge",
)

make_suite(
    "safety_openai",
    (
        *(f"{task}:openai_judge" for task in SAFETY_TASKS),
        *(f"{task}:mcq" for task in MCQ_TASKS),
        "strongreject:openai_judge",
    ),
    aggregation=AggregationStrategy.AVERAGE,
    description="Safety evals for posttrained instruct models with an openai judge",
)

make_suite(
    "safety_base",
    (
        *(f"{task}:base" for task in SAFETY_TASKS),
        *(f"{task}:base" for task in MCQ_TASKS),
        "strongreject:base",
    ),
    aggregation=AggregationStrategy.AVERAGE,
    description="Safety evals for pretrained models",
)
