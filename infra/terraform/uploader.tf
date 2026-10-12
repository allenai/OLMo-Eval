# Identity for Beaker jobs that upload results. It has no Google Cloud roles: the ingest
# service allowlists it (UPLOADER_SERVICE_ACCOUNT in dashboard/api settings), and
# artifacts reach GCS through URLs the API signs. A leaked key can create new runs, which
# any @allenai.org account can already do with its own credentials. It cannot change or
# delete an existing run: that needs the run's uploader, its author, or the write secret
# kept in the run's results directory (services/ingest.py can_write).
resource "google_service_account" "uploader" {
  account_id   = "olmo-eval-uploader"
  display_name = "olmo-eval results uploader (Beaker jobs)"
}

# A new key is created on the first apply after each rotation period. Jobs launched with
# the old key fail to upload after rotation; their results stay on disk and can be
# re-uploaded with `olmo-eval results upload`.
resource "time_rotating" "uploader_key" {
  rotation_days = 90
}

# The private key is also stored in this configuration's state, which only project
# owners can read. They can create keys for this account anyway.
resource "google_service_account_key" "uploader" {
  service_account_id = google_service_account.uploader.name

  keepers = {
    rotation = time_rotating.uploader_key.id
  }

  lifecycle {
    create_before_destroy = true
  }
}

# olmo-eval beaker launch reads this with the launching user's local credentials and
# copies it into the job's workspace as the olmo_eval_uploader_key Beaker secret.
resource "google_secret_manager_secret" "uploader_key" {
  secret_id = "olmo-eval-uploader-key"

  replication {
    auto {}
  }
}

resource "google_secret_manager_secret_version" "uploader_key" {
  secret      = google_secret_manager_secret.uploader_key.id
  secret_data = base64decode(google_service_account_key.uploader.private_key)
}

resource "google_secret_manager_secret_iam_member" "uploader_key_readers" {
  for_each  = toset(var.uploader_key_readers)
  secret_id = google_secret_manager_secret.uploader_key.id
  role      = "roles/secretmanager.secretAccessor"
  member    = each.value
}

# Readers need no project-level role such as serviceusage.serviceUsageConsumer. Secret Manager
# bills an access to the project that holds the secret and ignores the caller's quota project
# (x-goog-user-project): with user credentials, the access succeeds with no quota project, with
# the user's own quota project, and with a project that does not exist.
