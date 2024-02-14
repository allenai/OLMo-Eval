# AI2 Internal Resources

## Simplified catwalk evaluations

Run catwalk evaluations in beaker without requiring setting up and managing a tango workspace.

### Setup (one time)
**In your desired directory:**

Clone the eval repository.
```
git clone https://github.com/allenai/OLMo-Eval.git
cd OLMo-Eval
git checkout ai2-internal
```

Create conda environment and install requirements (including beaker gantry and jupyter notebook):
```
conda create -n ai2-olmo-eval python=3.10
conda activate ai2-olmo-eval
pip install -e '.[dev]'
pip install beaker-gantry
pip install notebook
```


**To automatically pipe the output to Google Sheets:**

1. Create a developer account (and a new project) then a service account following [these instructions](https://pygsheets.readthedocs.io/en/stable/authorization.html).
If you have a service account that you can use already, skip this step. To check this, go to the [Google Developers Console](https://console.developers.google.com/). 
On the left panel, click "Credentials" and look for a "Service Account" you can use for this purpose. [Tip: Click on the service account, look under "KEYS" and make sure you can "Create new key".]

E.g., if you are at AI2, you are likely to have access to the project with your team name. 
Then, for instance, under the "AI2 Aristo" project, there is a service account 
with the email "olmo-eval-share@ai2-aristo.iam.gserviceaccount.com" and 
name "olmo-eval-share". You can also create a service account under your team's project.

2. Create a Google sheet. Share the Google sheet with the service account email 
(e.g., "olmo-eval-share@ai2-aristo.iam.gserviceaccount.com")

3. Create and download the API json key for the service account. Click on the service account, 
look under "KEYS" and "Create new key". Choose "JSON" as the key type when prompted. 
Save the downloaded json key with desired name, at a location of your choice.

4. Set API key as environmental variable.

```
# NOTE: Replace "downloaded_credentials_file.json" with your downloaded json
export GDRIVE_SERVICE_ACCOUNT_JSON=$(cat ../downloaded_credentials_file.json)

# NOTE: Replace "your-llm-eval-workspace" with your beaker workspace for running this
beaker secret write --workspace ai2/your-llm-eval-workspace GDRIVE_SERVICE_ACCOUNT_JSON $GDRIVE_SERVICE_ACCOUNT_JSON
```

Example
```
export GDRIVE_SERVICE_ACCOUNT_JSON=$(cat ../downloaded_credentials_file.json)

beaker secret write --workspace ai2/yulingg-llm-eval GDRIVE_SERVICE_ACCOUNT_JSON $GDRIVE_SERVICE_ACCOUNT_JSON
```

## Run jupyter notebook
For the rest, just (modify and) run the cells in the notebook
[CatwalkEvaluation.ipynb](CatwalkEvaluation.ipynb)!



