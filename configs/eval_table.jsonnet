/*--------------------------------------- Configurations -----------------------------------------*/

local utils = import 'utils.libsonnet';

local standard = import 'task_sets/standard_benchmarks.libsonnet';
local mmlu = import 'task_sets/mmlu_tasks.libsonnet';


//❗Set gsheet to the name of your google sheet.
// Set it to null if you do not want your results to be uploaded to a google sheet (they will still be saved as an object).
//local gsheet = "auto-gsheet-test";
local gsheet = null;

// Models to evaluate

local models_to_eval = ["tiiuae/falcon-7b", "mosaicml/mpt-7b", "meta-llama/Llama-2-7b-hf", "meta-llama/Llama-2-13b-hf"]

local models = [
    {
        model_path: "tiiuae/falcon-7b",
        gpus_needed: 1,
        prediction_kwargs: {
            model_max_length: 2048,
            max_batch_tokens: 20480,
        }
    },
    {
        model_path: "mosaicml/mpt-7b",
        gpus_needed: 1,
        prediction_kwargs: {
            model_max_length: 2048,
            max_batch_tokens: 20480,
        }
    },
    {
        model_path: "meta-llama/Llama-2-7b-hf",
        gpus_needed: 1,
        prediction_kwargs: {
            model_max_length: 2048,
            max_batch_tokens: 20480,
        }
    },
    {
        model_path: "meta-llama/Llama-2-13b-hf",
        gpus_needed: 2,
        prediction_kwargs: {
            model_max_length: 2048,
            max_batch_tokens: 20480,
        }
    }
];

local task_sets = [
    standard_benchmarks.task_set,
    mmlu_tasks.task_set
];


{
    steps: utils.create_pipeline(models, task_sets, gsheet)
}
