import json
import os
import pandas as pd
import random
import re

# Fill these in before use (e.g., in jupyter notebook)
REPO_DIR = ""
DEFAULT_CATWALK_OPTIONS = {}
DEFAULT_GANTRY_OPTIONS = {}


def make_exp_name(model_name, checkpoint=None):
    random_hex = '%010x' % random.randrange(16**10)
    res = "lmeval-" + re.sub(".*/", "", model_name)
    if checkpoint:
        res += "-" + checkpoint
    res+= "-" + random_hex
    # Make conform to valid Beaker experiment names
    res = re.sub("[^-_.a-zA-Z0-9]", "", res)
    return res


def run_gantry_simple(command):
    global REPO_DIR
    current_dir = os.getcwd()
    os.chdir(REPO_DIR)
    stream = os.popen(command)
    output = stream.read()
    stream.close()
    os.chdir(current_dir)
    return {"command": command, "output": output}


def run_catwalk(model_spec, task_spec, exp_spec=None, show_command=False, gantry_extra_args=None):
    global DEFAULT_GANTRY_OPTIONS, DEFAULT_CATWALK_OPTIONS
    exp_spec = exp_spec or {}
    gantry_args = {'venv': 'base', 'workspace': DEFAULT_GANTRY_OPTIONS['workspace'],
                   'env': 'HF_DATASETS_CACHE=' + DEFAULT_GANTRY_OPTIONS['beaker-dataset-cache'],
                   'env-secret': 'GDRIVE_SERVICE_ACCOUNT_JSON=GDRIVE_SERVICE_ACCOUNT_JSON',
                   'install': '"pip install -e .[olmo]"'}
    if gantry_extra_args:
        gantry_args.update(gantry_extra_args)
    catwalk_args = DEFAULT_CATWALK_OPTIONS.copy()
    catwalk_args.update({'full-output-file': '/results/predictions.jsonl',
                         'metrics-file': '/results/metrics.json'})
    model_name = model_spec.get('name')
    if not model_name:
        model_name = model_spec['beaker-model'].split('/')[-1]
    gpus = exp_spec.get("gpus")
    if gpus is None:  # hacky code to guesstimate number of gpus needed
        if "-30b" in model_name:
            gpus = 4
        elif '-40b' in model_name:
            gpus = 6
        elif "-13b" in model_name or "-8b" in model_name:
            gpus = 2
        else:
            gpus = 1
    gantry_args['gpus'] = gpus
    checkpoint = model_spec.get("checkpoint")
    exp_name = make_exp_name(model_name, checkpoint)
    gantry_args['name'] = exp_name
    if 'hostname' not in gantry_args:
        gantry_args['cluster'] = exp_spec.get('cluster', gantry_args.get('cluster', DEFAULT_GANTRY_OPTIONS['cluster']))
    gantry_args['beaker-image'] = exp_spec.get('beaker-image',
                                               gantry_args.get('beaker-image', DEFAULT_GANTRY_OPTIONS['beaker-image']))
    if 'model-max-length' in model_spec:
        catwalk_args['model-max-length'] = model_spec['model-max-length']
    if 'max-batch-tokens' in model_spec:
        catwalk_args['max-batch-tokens'] = model_spec['max-batch-tokens']
    if 'limit' in task_spec:
        catwalk_args['limit'] = task_spec['limit']
    if 'split' in task_spec:
        catwalk_args['split'] = task_spec['split']
    if 'num-recorded-inputs' in task_spec:
        catwalk_args['num-recorded-inputs'] = task_spec['num-recorded-inputs']
    catwalk_args['num-shots'] = task_spec.get('num-shots', 0)
    if 'fewshot_seed' in task_spec:
        catwalk_args['fewshot-seed'] = task_spec['fewshot-seed']
    if 'gsheet' in exp_spec:
        catwalk_args['gsheet'] = exp_spec['gsheet']

    beaker_datasets = []
    task_file = task_spec.get('task-file')
    if task_file:
        task_file = task_file.replace('.jsonl', '')
        catwalk_args['task-file'] = f"/input/{task_file}.jsonl"
        beaker_datasets.append(f'oyvindt/{task_file}:/input')
        if 'ppl-val-v2-small' in task_file:
            beaker_datasets.append('oyvindt/olmo-ppl-val-v2-small:/data/olmo-ppl-val-v2-small')
    else:
        catwalk_args['task'] = task_spec.get('task')
    if "beaker-model" in model_spec:
        beaker_datasets.append(model_spec['beaker-model']+":/model")
        catwalk_args['model-path'] = "/model"
    if "model-path" in model_spec:
        catwalk_args['model-path'] = model_spec['model-path']
    model_full = f"lm::pretrained={model_name}"
    if checkpoint:
        model_full += f",revision={checkpoint}"
    remote_code_default = False
    if not 'olmo' in model_name:
        if "-falcon" in model_name or '-mpt' in model_name:
            remote_code_default = True
    if model_spec.get('remote-code', remote_code_default):
        model_full += ",trust_remote_code=True"
    catwalk_args['model'] = model_full

    template = ["gantry run"]
    for key, value in gantry_args.items():
        template.append(f"--{key}")
        template.append(str(value))
    for beaker_dataset in beaker_datasets:
        template.append("--dataset")
        template.append(beaker_dataset)
    template.append("-- python olmo_eval/run_lm_eval.py")
    for key, value in catwalk_args.items():
        template.append(f"--{key}")
        template.append(str(value))
    command = " ".join(template)
    print(f"Running {exp_name}")
    if show_command:
        print(command)
    run_output = run_gantry_simple(command)
    res = {"exp_name": exp_name, "model": model_name,
           "model_spec": model_spec, "task_spec": task_spec, "exp_spec": exp_spec,
           "gantry_command": run_output['command']}
    return res


def load_gsheet_as_df(gsheet_name, sheet_index=0, auth_file=None):
    import pygsheets
    if auth_file:
        with open(auth_file) as file:
            service_account_json = file.read()
    else:
        service_account_json=os.environ["GDRIVE_SERVICE_ACCOUNT_JSON"]
    client = pygsheets.authorize(service_account_json=service_account_json)
    sheet = client.open(gsheet_name)
    return sheet[sheet_index].get_as_df()

