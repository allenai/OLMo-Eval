/*--------------------------------------- Configurations -----------------------------------------*/

local utils = import 'utils.libsonnet';

local gen_tasks = import 'task_sets/gen_tasks.libsonnet';
//❗other task sets can be imported here

//❗Set gsheet to the name of your google sheet.
// Set it to null if you do not want your results to be uploaded to a google sheet (they will still be saved as an object).
//local gsheet = "auto-gsheet-test";
local gsheet = null;

// Models to evaluate

local models = [
    {
        model_path: "EleutherAI/pythia-1b",
        revision: "step140000", //❗Specify checkpoint if needed
        gpus_needed: 1,
        //❗Task sets contain default values for prediction_kwargs. These can be overriden for each model here.
        prediction_kwargs: {
            model_max_length: 2048,
            max_batch_tokens: 20480,
            limit: 2
        }
    }
    //❗other models can be added here
];

local task_sets = [
    gen_tasks.task_set,
    //❗other task sets can be added here
];


{
    steps: utils.create_pipeline(models, task_sets, gsheet)
}
