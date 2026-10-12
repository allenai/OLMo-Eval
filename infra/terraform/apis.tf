# Skiff2 already enables storage, iam, iamcredentials, run, iap and secretmanager.
resource "google_project_service" "sqladmin" {
  service            = "sqladmin.googleapis.com"
  disable_on_destroy = false
}
