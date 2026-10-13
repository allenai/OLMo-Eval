variable "project_id" {
  description = "GCP project created by Skiff2 for olmo-eval."
  type        = string
  default     = "ai2-skiff2-olmo-eval"
}

variable "region" {
  description = "Region for Cloud SQL and the buckets."
  type        = string
  default     = "us-west1"
}

variable "debug_iam_users" {
  description = "Google accounts that can impersonate the runtime service accounts (api, dashboard) and log in to Cloud SQL with IAM auth."
  type        = list(string)
  default     = ["chrisg@allenai.org"]
}

variable "github_actions_service_account" {
  description = "Skiff2's deploy service account. It reads build assets (fonts) during CI."
  type        = string
  default     = "github-actions@ai2-skiff2-olmo-eval.iam.gserviceaccount.com"
}

variable "uploader_key_readers" {
  description = "Principals that can read the uploader key, i.e. launch Beaker jobs that upload results."
  type        = list(string)
  default     = ["domain:allenai.org"]
}
