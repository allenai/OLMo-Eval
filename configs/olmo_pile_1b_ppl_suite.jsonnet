/*--------------------------------------- Configurations -----------------------------------------*/

local utils = import 'utils.libsonnet';

// local ppl_suite = import 'task_sets/eval_suite_ppl_test_v3_not_deconned.libsonnet';
local ppl_suite = import 'task_sets/eval_suite_ppl_val_v2_small.libsonnet';


//❗Set gsheet to the name of your google sheet.
// Set it to null if you do not want your results to be uploaded to a google sheet (they will still be saved as an object).
// local gsheet = "ppl-suite-token-ppls-working-env";
// local gsheet = "perplexity-suite-paper";
// local gsheet = null;

local gsheet = "perplexity-suite-paper-pile-debug";
// local output_dir = "s3://ai2-llm/eval-results/perplexity/olmo/1b/olmo-small-pile-150B-mcli-hf-tokenizer-results-new-workspaces-reunsharded-local-beaker/";

local create_models = function(model_path, revisions, gpus_needed) [
    {
        model_path: model_path,
        revision: rev,
        gpus_needed: gpus_needed,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null,
        }
    }
    for rev in revisions
];

// local revisions = [
//     "step" + std.toString(i * 10000)
//     for i in std.range(1, 14)
// ] + ['step143000'];
// step5000-unsharded thru step35000-unsharded
// local revisions = [
//     "step" + std.toString(i * 5000) + "-unsharded"
//     for i in std.range(1, 7)
// ];

local revisions = [
    "step15000-unsharded",
    "step25000-unsharded"
];


local models = create_models("s3://ai2-llm/checkpoints/1b/olmo-small-pile-150B-mcli-hf-tokenizer-reunshard", revisions, 1);
local task_sets = [
    ppl_suite.task_set
];


// {
//     steps: utils.create_fine_grained_pipeline(models, task_sets, gsheet, output_dir)
// }

{
    steps: utils.create_pipeline(models, task_sets, gsheet)
}