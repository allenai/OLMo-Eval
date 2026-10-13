terraform {
  required_version = ">= 1.13"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 7.46"
    }
    time = {
      source  = "hashicorp/time"
      version = "~> 0.13"
    }
  }

  # Shares the Skiff2 state bucket. Skiff uses the terraform/infra and terraform/services
  # prefixes; this configuration owns only terraform/olmo-eval.
  backend "gcs" {
    bucket = "ai2-skiff2-olmo-eval-tf-state"
    prefix = "terraform/olmo-eval"
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}
