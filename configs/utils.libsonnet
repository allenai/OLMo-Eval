/*
    Utility functions for composing jsonnet configurations for the evaluation pipeline.
*/

// task_set1: [task1, task2], task_set2: [task3, task4] --> [task1, task2, task3, task4]
local flatten_task_sets(task_sets) = std.flatMap(
    function(task_set) std.map(
        function(task) {
            task_set: task_set.name,
            task_name: task.task_name,
            prediction_kwargs: task.prediction_kwargs,
            task_kwargs: task.task_kwargs
        },
        task_set.tasks
    ),
    task_sets
);


// models: [model1, model2], task_configs: [task1, task2, task3] --> [(model1, task1), (model1, task2), ... (model2, task3)]
local model_task_cross_product(models, task_configs) = std.flatMap(
    function(task_config) std.map(
        function(model_config)
            // model pred kwargs override task pred kwargs.
            local prediction_kwargs = std.get(task_config, "prediction_kwargs", {}) + std.get(model_config, "prediction_kwargs", {});
            local full_config = model_config + task_config;
            full_config + {"prediction_kwargs": prediction_kwargs},
        models
    ),
    task_configs
);

local basepath(path) =
  // -1 index does not work, so we do this.
  local temp = std.split(path, "/");
  temp[std.length(temp)-1];

local full_model(model_config) = basepath(model_config.model_path) + std.get(model_config, "revision", "");

local contains = function(main_string, sub_string)
    std.length(std.findSubstr(sub_string, main_string)) > 0;

local is_olmo_model = function(model_config)
    contains(std.get(model_config, "model_path"), "olmo");

// Model steps

local model_location_step_name(model_config) = "model_location_" + full_model(model_config);
local model_location_ref(model_config) = {type: "ref", ref: model_location_step_name(model_config)};

local create_model_location_steps(models) = std.foldl(
    function(x, model_config) x + {
        [if is_olmo_model(model_config) then model_location_step_name(model_config)]: {
            type: "get-model-path",
            model_path: model_config.model_path,
            revision: std.get(model_config, "revision"),
            step_resources: {
                gpu_count: 0
            }
        }
    },
    models,
    {}
);

local catwalk_model_step_name(model_config) = "catwalk_model_" + full_model(model_config);
local catwalk_model_ref(model_config) = {type: "ref", ref: catwalk_model_step_name(model_config)};

local create_catwalk_model_steps(models) = std.foldl(
    function(x, model_config) x + {
        [catwalk_model_step_name(model_config)]: {
            type: "construct-catwalk-model",
            model_path: if is_olmo_model(model_config) then model_location_ref(model_config) else model_config.model_path,
            model_class: std.get(model_config, "hf_model_class"),
            revision: std.get(model_config, "revision"),
            trust_remote_code: std.get(model_config, "trust_remote_code", false),
            step_resources: {
                gpu_count: 0
            }
        }
    },
    models,
    {}
);


// Task steps
local task_step_name(config) = "task_" + config.task_set + "_" + config.task_name + std.get(config.task_kwargs, "task_rename", "");
local task_ref(config) = {type: "ref", ref: task_step_name(config)};


local create_task_steps(task_configs) = std.foldl(
    function(x, config) x + {
        [task_step_name(config)]: config.task_kwargs + {
            type: "construct-task",
            task_name: config.task_name,
            step_resources: {
                gpu_count: 0
            }
        }
    },
    task_configs,
    {}
);


// Output steps
local outputs_step_name(config) =
    "outputs_" +
    full_model(config) + "_" +
    config.task_set + "_" +
    config.task_name + std.get(config.task_kwargs, "task_rename", "");

local outputs_ref(config) = {type: "ref", ref: outputs_step_name(config)};
local processed_outputs_ref(config) = {type: "ref", ref: "processed_" + outputs_step_name(config)};

local create_outputs_steps(model_task_configs) = std.foldl(
    function(x, config) x + {
        [outputs_step_name(config)]: {
            type: "predict-and-calculate-metrics",
            model: catwalk_model_ref(config),
            task_dict: {type: "ref", ref: task_step_name(config)},
            step_resources: {
                gpu_count: config.gpus_needed
            }
        } + config.prediction_kwargs,

    },
    model_task_configs,
    {}
);

local create_process_outputs_steps(model_task_configs) = std.foldl(
    function(x, config) x + {
        ["processed_" + outputs_step_name(config)]: {
            type: "process-outputs",
            outputs: outputs_ref(config),
            step_resources: {
                gpu_count: 0
            }
        },
    },
    model_task_configs,
    {}
);

local all_outputs(model_task_configs) = [
    outputs_ref(config)
    for config in model_task_configs
];

local all_processed_outputs(model_task_configs) = [
    processed_outputs_ref(config)
    for config in model_task_configs
];

local all_pred_kwargs(model_task_configs) = [
    config.prediction_kwargs
    for config in model_task_configs
];

local all_models(model_task_configs) = [
    config.model_path
    for config in model_task_configs
];

local create_outputs_as_rows_steps(model_task_configs, gsheet) =
    {
        "combine-all-outputs": {
            type: "write-outputs-as-rows",
            outputs: all_outputs(model_task_configs),
            models: all_models(model_task_configs),
            prediction_kwargs: all_pred_kwargs(model_task_configs),
            gsheet: gsheet,
            step_resources: {
                gpu_count: 0
            }
        }
    };

local create_processed_outputs_as_rows_multiple_metrics_steps(model_task_configs, gsheet) =
    {
        "combine-all-outputs": {
            type: "write-outputs-as-rows-multiple-metrics",
            outputs: all_processed_outputs(model_task_configs),
            models: all_models(model_task_configs),
            prediction_kwargs: all_pred_kwargs(model_task_configs),
            gsheet: gsheet,
            step_resources: {
                gpu_count: 0
            }
        }
    };

local create_save_write_outputs_as_rows_multiple_metrics_as_file_steps(output_dir) =
    {
        "save-to-file": {
            type: "save-write-outputs-as-rows-multiple-metrics-as-file",
            write_outputs: {type: "ref", ref: "combine-all-outputs"},
            output_dir: output_dir,
            step_resources: {
                gpu_count: 0
            }
        }
    };

local create_pipeline(models, task_sets, gsheet) =

    // Model steps
    local model_location_steps = create_model_location_steps(models);
    local catwalk_model_steps = create_catwalk_model_steps(models);

    // Task steps
    local task_configs = flatten_task_sets(task_sets);
    local task_steps = create_task_steps(task_configs);

    // Prediction and metrics
    local model_task_configs = model_task_cross_product(models, task_configs);
    local outputs_steps = create_outputs_steps(model_task_configs);

    // Aggregate results for each task set and model combination
    local combine_all_outputs = create_outputs_as_rows_steps(model_task_configs, gsheet);

    local all_steps =
        model_location_steps +
        catwalk_model_steps +
        task_steps +
        outputs_steps +
        combine_all_outputs;

    all_steps;

local create_fine_grained_pipeline(models, task_sets, gsheet, output_dir = null) =

    // Model steps
    local model_location_steps = create_model_location_steps(models);
    local catwalk_model_steps = create_catwalk_model_steps(models);

    // Task steps
    local task_configs = flatten_task_sets(task_sets);
    local task_steps = create_task_steps(task_configs);

    // Prediction and metrics
    local model_task_configs = model_task_cross_product(models, task_configs);
    local outputs_steps = create_outputs_steps(model_task_configs);

    local processed_outputs_steps = create_process_outputs_steps(model_task_configs);

    // Aggregate results for each task set and model combination
    local combine_all_outputs = create_processed_outputs_as_rows_multiple_metrics_steps(model_task_configs, gsheet);

    local save_to_file = create_save_write_outputs_as_rows_multiple_metrics_as_file_steps(output_dir);

    local all_steps =
        model_location_steps +
        catwalk_model_steps +
        task_steps +
        outputs_steps +
        processed_outputs_steps +
        combine_all_outputs +
        save_to_file;

    all_steps;


{
    create_pipeline: create_pipeline,
    create_fine_grained_pipeline: create_fine_grained_pipeline
}

/*local wandb_log_step = {
    logged_metrics: {
        type: "log-metrics",
            //project: "wandb-eval-test",
            //entity: "ai2-llm",
            model_name: "test-olmo-model",
            task_set: "rc20tasks",
            task: "boolq",
            metrics: {type: "ref", "ref": "metrics_test-olmo-model_rc20tasks_boolq"}

    }
};*/
