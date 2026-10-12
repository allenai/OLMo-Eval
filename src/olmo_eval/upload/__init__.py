"""Upload eval results to the olmo-eval dashboard.

olmo-eval sends results to the dashboard's ingest service (a Cloud Run service)
with the user's Google credentials. Large files go straight to GCS through signed
URLs handed out by the service. Nothing here talks to the database directly.
"""

from olmo_eval.upload.config import UploadConfig, resolve_upload_config
from olmo_eval.upload.uploader import (
    UploadResult,
    mark_run_failed,
    register_run_start,
    upload_results_dir,
)

__all__ = [
    "UploadConfig",
    "UploadResult",
    "mark_run_failed",
    "register_run_start",
    "resolve_upload_config",
    "upload_results_dir",
]
