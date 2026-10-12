output "instance_connection_name" {
  value = google_sql_database_instance.main.connection_name
}

output "api_service_account_email" {
  value = google_service_account.api.email
}

output "api_db_user" {
  value = google_sql_user.api.name
}

output "database_names" {
  value = {
    prod = google_sql_database.prod.name
    dev  = google_sql_database.dev.name
  }
}

output "results_bucket" {
  value = google_storage_bucket.results.name
}

output "build_assets_bucket" {
  value = google_storage_bucket.build_assets.name
}

output "uploader_service_account_email" {
  value = google_service_account.uploader.email
}

output "dashboard_service_account_email" {
  value = google_service_account.dashboard.email
}

output "dashboard_db_user" {
  value = google_sql_user.dashboard.name
}
