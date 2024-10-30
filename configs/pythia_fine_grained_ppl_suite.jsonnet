/*--------------------------------------- Configurations -----------------------------------------*/

local utils = import 'utils.libsonnet';

local ppl_suite = import 'task_sets/eval_suite_ppl_val_v3.libsonnet';


//❗Set gsheet to the name of your google sheet.
// Set it to null if you do not want your results to be uploaded to a google sheet (they will still be saved as an object).
// local gsheet = "ppl-suite-token-ppls-working-env";
local gsheet = null;

local create_models = function(model_path, revisions, gpus_needed) [
    {
        model_path: model_path,
        revision: rev,
        gpus_needed: gpus_needed,
        prediction_kwargs: {
            model_max_length: 2048,
            max_batch_tokens: 2048,
            limit: null,
            //fewshot_seed: 1234, //etc.
        }
    }
    for rev in revisions
];

local revisions = [
    "step" + std.toString(i * 10000)
    for i in std.range(14, 14)
];


local models = create_models("EleutherAI/pythia-6.9b", revisions, 1); //EleutherAI/pythia-6.9b needs diff batch size

local task_sets = [
    ppl_suite.task_set
];


{
    steps: utils.create_fine_grained_pipeline(models, task_sets, gsheet)
}
