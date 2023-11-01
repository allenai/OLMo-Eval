/*--------------------------------------- Configurations -----------------------------------------*/

local utils = import 'utils.libsonnet';

local ppl_suite = import 'task_sets/eval_suite_ppl_test_v3_not_deconned.libsonnet';

local gsheet = "perplexity-suite-paper";

local create_models = function(model_path, revisions, gpus_needed) [
    {
        model_path: model_path,
        revision: rev,
        gpus_needed: gpus_needed,
        prediction_kwargs: {
            model_max_length: 2048,
            max_batch_tokens: 2048,
            limit: null,
        }
    }
    for rev in revisions
];

local revisions = [
    "240b_tokens",
    "280b_tokens",
    "400b_tokens",
    "440b_tokens",
    "500b_tokens",
    "600b_tokens",
    "700b_tokens",
    "720b_tokens",
    "960b_tokens",
    "1t_tokens"
];


local models = create_models("togethercomputer/RedPajama-INCITE-7B-Base", revisions, 1);

local task_sets = [
    ppl_suite.task_set
];


{
    steps: utils.create_fine_grained_pipeline(models, task_sets, gsheet)
}