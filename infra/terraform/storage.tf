# Every file of every run's output directory, under [dev/]runs/<run_id>/.
resource "google_storage_bucket" "results" {
  name                        = "ai2-skiff2-olmo-eval-results"
  location                    = "US-WEST1"
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"
  force_destroy               = false

  # Non-prod environments write under dev/.
  lifecycle_rule {
    condition {
      age            = 30
      matches_prefix = ["dev/"]
    }
    action {
      type = "Delete"
    }
  }

  lifecycle_rule {
    condition {
      age            = 180
      matches_prefix = ["runs/"]
    }
    action {
      type          = "SetStorageClass"
      storage_class = "NEARLINE"
    }
  }

  lifecycle_rule {
    condition {
      age = 7
    }
    action {
      type = "AbortIncompleteMultipartUpload"
    }
  }
}

# Private files the UI build needs and the public repo must not contain (Telegraf fonts under
# fonts/).
resource "google_storage_bucket" "build_assets" {
  name                        = "ai2-skiff2-olmo-eval-build-assets"
  location                    = "US-WEST1"
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"
}
