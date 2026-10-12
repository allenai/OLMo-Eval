# Runtime identity for the ingest Cloud Run service. Its database user owns the schema and runs
# the migrations, and it writes and deletes run artifacts in the results bucket.
resource "google_service_account" "api" {
  account_id   = "olmo-eval-api"
  display_name = "olmo-eval dashboard API"
}

# Runtime identity for the ui Cloud Run service and its api sidecar (dashboard mode). It reads
# the database and the results bucket, and writes only the few tables the dashboard edits (see
# scripts/grant_db.py).
resource "google_service_account" "dashboard" {
  account_id   = "olmo-eval-dashboard"
  display_name = "olmo-eval dashboard (read-only UI)"
}

locals {
  api_member       = "serviceAccount:${google_service_account.api.email}"
  dashboard_member = "serviceAccount:${google_service_account.dashboard.email}"

  runtime_project_roles = [
    "roles/cloudsql.client",
    "roles/cloudsql.instanceUser",
    "roles/logging.logWriter",
  ]

  debug_project_roles = [
    "roles/cloudsql.client",
    "roles/cloudsql.instanceUser",
  ]

  debug_project_bindings = {
    for pair in setproduct(var.debug_iam_users, local.debug_project_roles) :
    "${pair[0]} ${pair[1]}" => { user = pair[0], role = pair[1] }
  }
}

# Neither service reads Secret Manager secrets. If a service later needs one, grant
# roles/secretmanager.secretAccessor on that secret only.
resource "google_project_iam_member" "api" {
  for_each = toset(local.runtime_project_roles)
  project  = var.project_id
  role     = each.value
  member   = local.api_member
}

resource "google_project_iam_member" "dashboard" {
  for_each = toset(local.runtime_project_roles)
  project  = var.project_id
  role     = each.value
  member   = local.dashboard_member
}

# Lets ingest sign V4 upload URLs through IAM signBlob as itself.
resource "google_service_account_iam_member" "api_self_token_creator" {
  service_account_id = google_service_account.api.name
  role               = "roles/iam.serviceAccountTokenCreator"
  member             = local.api_member
}

# Lets the dashboard sign V4 download URLs through IAM signBlob as itself.
resource "google_service_account_iam_member" "dashboard_self_token_creator" {
  service_account_id = google_service_account.dashboard.name
  role               = "roles/iam.serviceAccountTokenCreator"
  member             = local.dashboard_member
}

# Ingest lists and reads objects, signs uploads (create and overwrite), and deletes a run's
# prefix. objectUser covers that without objectAdmin's object IAM permissions.
resource "google_storage_bucket_iam_member" "api_results_object_user" {
  bucket = google_storage_bucket.results.name
  role   = "roles/storage.objectUser"
  member = local.api_member
}

resource "google_storage_bucket_iam_member" "dashboard_results_object_viewer" {
  bucket = google_storage_bucket.results.name
  role   = "roles/storage.objectViewer"
  member = local.dashboard_member
}

# Impersonation of the runtime service accounts, for debugging and verification.
resource "google_service_account_iam_member" "debug_token_creator" {
  for_each           = toset(var.debug_iam_users)
  service_account_id = google_service_account.api.name
  role               = "roles/iam.serviceAccountTokenCreator"
  member             = "user:${each.value}"
}

resource "google_service_account_iam_member" "debug_dashboard_token_creator" {
  for_each           = toset(var.debug_iam_users)
  service_account_id = google_service_account.dashboard.name
  role               = "roles/iam.serviceAccountTokenCreator"
  member             = "user:${each.value}"
}

resource "google_project_iam_member" "debug" {
  for_each = local.debug_project_bindings
  project  = var.project_id
  role     = each.value.role
  member   = "user:${each.value.user}"
}

# CI downloads the Telegraf fonts from the build-assets bucket before building the UI image.
resource "google_storage_bucket_iam_member" "github_actions_build_assets_viewer" {
  bucket = google_storage_bucket.build_assets.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${var.github_actions_service_account}"
}
