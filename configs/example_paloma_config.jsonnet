/*--------------------------------------- Configurations -----------------------------------------*/

local utils = import 'utils.libsonnet';

//❗ To run this config you will need to first set up the data following the instructions in ai2-llm-eval/eval_data/README.md
local ppl_suite = import 'task_sets/paloma_hf_release_val.libsonnet';


//❗Set gsheet to the name of your google sheet.
// Set it to null if you do not want your results to be uploaded to a google sheet (they will still be saved as an object).
local gsheet = null;
//❗Set output_dir to a directory where you want to save outputs as jsonl.gz files .
// Set it to null if you do not want your results saved as jsonl.gz files.
local output_dir = "/Users/ianm/projects/paloma-release/ai2-llm-eval/output/";

local create_models = function(model_path, revisions, gpus_needed) [
    {
        model_path: model_path,
        revision: rev,
        gpus_needed: gpus_needed,
        prediction_kwargs: {
            model_max_length: 2048, //❗Ensure that this is set to the actual max len of your model
            limit: 2, //❗ Here we only run 2 examples per task for testing purposes. Set this to null to run all examples.
        }
    }
    for rev in revisions
];

local revisions = [
    "step" + std.toString(i * 10000)
    for i in std.range(14, 14) //❗ Set this to the range of revisions you want to run.
];


local models = create_models("EleutherAI/pythia-160m-seed1", revisions, 1); 
local task_sets = [
    ppl_suite.task_set
];


{
    steps: utils.create_fine_grained_pipeline(models, task_sets, gsheet, output_dir)
}
