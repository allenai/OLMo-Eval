
# Advanced settings


## Save output to google sheet

* Enable Google Drive API in your GCE console for your project.
* [Set up a service account](https://pygsheets.readthedocs.io/en/stable/authorization.html)
* Share the google sheet with the service accout email.
* Create and download the API json key for the service account in your google project.
* Uncomment `local gsheet=...` in the evaluation config, and set it to the name of the google sheet.
* Set environment variable to downloaded API key before running the `tango run` command.

```commandline
export GDRIVE_SERVICE_ACCOUNT_JSON=$(cat downloaded_credentials_file.json)
tango --settings tango.yml run configs/evaluation_template.jsonnet --workspace my-eval-workspace
```

## Use a remote workspace

`tango` allows you to use a `google` workspace.

* Enable Google Datastore in your GCE console for your project.

* Login with CLI

```commandline
gcloud auth login
gcloud auth application-default login
gcloud config set project <your-project>
```

* Run with a `gs:` workspace.

```commandline
tango --settings tango.yml run configs/example_config.jsonnet --workspace gs://my-gs-workspace
```

This will create a `tango` workspace in google cloud bucket.

💡 See [`tango.yml`](tango.yml) to set this as the default option.

### Troubleshooting

If some error causes your google workspace to go into a bad state (i.e., you get errors that say step should not be in completed state, etc.), you can clear the workspace with

```commandline
python scripts/empty_workspace.py my-gs-workspace
```


## Run without Tango

The [`olmo_eval/run_lm_eval.py`](olmo_eval/run_lm_eval.py) script provides a way to run an evaluation as a single
job with associated result set. Arguments can be provided in a config file, an example is found 
in `configs/run_lm_eval_example.jsonnet`, or as direct arguments (see documentation in script). E.g.,

```commandline
python -m olmo_eval.run_lm_eval --config_file configs/run_lm_eval_example.jsonnet
```
or
```commandline
python -m olmo_eval.run_lm_eval --model lm::pretrained=EleutherAI/pythia-160m,revision=step140000 \
    --task arc_challenge arc_easy  --split validation \
    --full_output_file predictions.jsonl --metrics_file metrics.json --model_max_length 2048 \
    --max_batch_tokens 4096 --num_recorded_inputs 3 --num_shots 0 --gsheet OLMo-evals-testing
```

