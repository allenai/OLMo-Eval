/*--------------------------------------- Configurations -----------------------------------------*/

local utils = import '../configs/utils.libsonnet';

local rc20_tasks = import '../configs/task_sets/rc20_tasks.libsonnet';
local gen_tasks = import '../configs/task_sets/gen_tasks.libsonnet';
local summary_tasks = import '../configs/task_sets/summary_tasks.libsonnet';
local standard_benchmarks = import '../configs/task_sets/standard_benchmarks.libsonnet';
local mmlu = import '../configs/task_sets/mmlu_tasks.libsonnet';

local gsheet = null;

// Models to evaluate

local models = [
    {
        model_path: "sshleifer/tiny-gpt2",
        gpus_needed: 1,
        prediction_kwargs: {
            model_max_length: 128,
            // max_batch_tokens: 32,
            limit: 2,
            //fewshot_seed: 1234, //etc.
        }
    }
];

local task_sets = [
    rc20_tasks.task_set,
    gen_tasks.task_set,
    summary_tasks.task_set,
    standard_benchmarks.task_set,
    mmlu.task_set
];


{
    steps: utils.create_pipeline(models, task_sets, gsheet)
}
