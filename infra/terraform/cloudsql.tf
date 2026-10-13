resource "google_sql_database_instance" "main" {
  name                = "olmo-eval-db"
  database_version    = "POSTGRES_17"
  region              = var.region
  deletion_protection = true

  settings {
    # Postgres 16+ defaults to ENTERPRISE_PLUS. Keep the cheaper edition.
    edition                     = "ENTERPRISE"
    tier                        = "db-custom-2-8192"
    availability_type           = "ZONAL"
    disk_type                   = "PD_SSD"
    disk_size                   = 20
    disk_autoresize             = true
    disk_autoresize_limit       = 500
    deletion_protection_enabled = true
    # Only the Cloud SQL connectors may connect.
    connector_enforcement = "REQUIRED"

    ip_configuration {
      # Public IP with no authorized networks.
      ipv4_enabled = true
      ssl_mode     = "ENCRYPTED_ONLY"
    }

    backup_configuration {
      enabled                        = true
      point_in_time_recovery_enabled = true
      start_time                     = "10:00" # UTC, 3 AM PDT
      transaction_log_retention_days = 7

      backup_retention_settings {
        retained_backups = 14
      }
    }

    # Sunday 11:00 UTC, 4 AM PDT.
    maintenance_window {
      day          = 7
      hour         = 11
      update_track = "stable"
    }

    insights_config {
      query_insights_enabled = true
    }

    database_flags {
      name  = "cloudsql.iam_authentication"
      value = "on"
    }

    database_flags {
      name  = "log_min_duration_statement"
      value = "1000"
    }

    user_labels = {
      app = "olmo-eval-dashboard"
    }
  }

  depends_on = [google_project_service.sqladmin]
}

# Used by the prod Skiff environment.
resource "google_sql_database" "prod" {
  name     = "olmo_eval"
  instance = google_sql_database_instance.main.name
}

# Shared by every non-prod Skiff environment (branch deploys).
resource "google_sql_database" "dev" {
  name     = "olmo_eval_dev"
  instance = google_sql_database_instance.main.name
}

# The ingest service's database user. It owns the tables and runs the migrations, through
# CREATE on schema public that scripts/grant_db.py grants. It is not a cloudsqlsuperuser member.
resource "google_sql_user" "api" {
  instance       = google_sql_database_instance.main.name
  name           = trimsuffix(google_service_account.api.email, ".gserviceaccount.com")
  type           = "CLOUD_IAM_SERVICE_ACCOUNT"
  database_roles = []
}

# The dashboard's database user. It gets SELECT on every table and a few writes from the table
# owner (scripts/grant_db.py, and the ingest service after each migration).
resource "google_sql_user" "dashboard" {
  instance = google_sql_database_instance.main.name
  name     = trimsuffix(google_service_account.dashboard.email, ".gserviceaccount.com")
  type     = "CLOUD_IAM_SERVICE_ACCOUNT"
}

resource "google_sql_user" "debug" {
  for_each       = toset(var.debug_iam_users)
  instance       = google_sql_database_instance.main.name
  name           = each.value
  type           = "CLOUD_IAM_USER"
  database_roles = ["cloudsqlsuperuser"]
}
