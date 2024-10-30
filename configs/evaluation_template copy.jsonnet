/*--------------------------------------- Configurations -----------------------------------------*/

local utils = import 'utils.libsonnet';

// local rc20_tasks = import 'task_sets/rc20_tasks.libsonnet';
// local gen_tasks = import 'task_sets/gen_tasks.libsonnet';
// local summary_tasks = import 'task_sets/summary_tasks.libsonnet';
local ppl_suite = import 'task_sets/eval_suite_ppl_val_v3.libsonnet';


//❗Set gsheet to the name of your google sheet.
// Set it to null if you do not want your results to be uploaded to a google sheet (they will still be saved as an object).
//local gsheet = "auto-gsheet-test";
local gsheet = "test-sheet-1";

// Models to evaluate

local models = [
    /*{
        model_path: "s3://ai2-llm/test_fixtures/olmo-1b", //❗Specify olmo unsharded checkpoint path
        gpus_needed: 1,
        trust_remote_code: true
    },*/
    {
        model_path: "sshleifer/tiny-gpt2",
        gpus_needed: 0,
        //❗Task sets contain default values for prediction_kwargs. These can be overriden for each model here.
        prediction_kwargs: {
            model_max_length: 2048,
            max_batch_tokens: 20480,
            limit: 5,
        }
    }
];

local task_sets = [
    // rc20_tasks.task_set,
    // gen_tasks.task_set,
    // summary_tasks.task_set,
    ppl_suite.task_set
];


{
    steps: utils.create_fine_grained_pipeline(models, task_sets, gsheet)
}
