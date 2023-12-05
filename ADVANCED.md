
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