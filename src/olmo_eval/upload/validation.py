"""Payload checks against the ingest contract.

The full JSON Schema lives in the repository at
``dashboard/contract/ingest-v1.schema.json``. When it and ``jsonschema`` are
available, payloads are validated against it. Otherwise (an installed package
outside a checkout, or no ``jsonschema``) only required keys are checked: this
degraded mode is logged once and reported by ``PayloadValidator.degraded_reason``.

The schema is not packaged with olmo-eval: ``jsonschema`` is a dev-only
dependency, so a packaged copy would not enable full validation by itself, and a
second copy could drift from the contract.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

from olmo_eval.common.logging import get_logger

logger = get_logger("upload.validation")

SCHEMA_RELATIVE_PATH = Path("dashboard") / "contract" / "ingest-v1.schema.json"

_REQUIRED: dict[str, tuple[str, ...]] = {
    "RunUpsertRequest": ("protocol_version", "client", "run", "model"),
    "SignArtifactsRequest": ("artifacts",),
    "TaskResultIn": (
        "task_name",
        "task_hash",
        "primary_metric",
        "metrics",
        "metric_meta",
        "config",
        "num_instances",
        "suites",
        "instance_count",
    ),
    "InstanceBatchRequest": ("batch_index", "instances"),
    "InferenceUploadRequest": ("source_paths", "batches", "series", "gpu_devices"),
    "CompleteRequest": ("status", "suites", "expected_task_results", "expected_artifacts"),
}


def find_contract_schema(start: Path | None = None) -> Path | None:
    """Look for the contract schema above the working directory and the package."""
    import olmo_eval

    candidates = [Path.cwd() if start is None else start, Path(olmo_eval.__file__).resolve()]
    for base in candidates:
        for directory in [base, *base.parents]:
            path = directory / SCHEMA_RELATIVE_PATH
            if path.is_file():
                return path
    return None


@lru_cache(maxsize=4)
def _validator_for(schema_path: str | None, definition: str) -> Any:
    if schema_path is None:
        return None
    try:
        import jsonschema
    except ImportError:
        return None
    schema = json.loads(Path(schema_path).read_text())
    wrapper = {
        "$schema": schema.get("$schema", "http://json-schema.org/draft-07/schema#"),
        "$ref": f"#/definitions/{definition}",
        "definitions": schema["definitions"],
    }
    return jsonschema.Draft7Validator(wrapper, format_checker=jsonschema.FormatChecker())


@lru_cache(maxsize=8)
def _warn_degraded(reason: str) -> None:
    """Log the degraded-mode warning once per reason."""
    logger.warning(f"Upload payloads are checked for required keys only: {reason}")


def _jsonschema_available() -> bool:
    try:
        import jsonschema  # noqa: F401
    except ImportError:
        return False
    return True


class PayloadValidator:
    """Validates payloads by contract definition name."""

    def __init__(self, schema_path: Path | None = None) -> None:
        self.schema_path = schema_path if schema_path is not None else find_contract_schema()
        reason = self.degraded_reason
        if reason:
            _warn_degraded(reason)

    @property
    def degraded_reason(self) -> str | None:
        """Why full schema validation is unavailable, or None when it is used."""
        if self.schema_path is None:
            return (
                f"contract schema {SCHEMA_RELATIVE_PATH.as_posix()} not found above the "
                "working directory or the olmo-eval package (run from a repository checkout "
                "for full validation)"
            )
        if not _jsonschema_available():
            return "jsonschema is not installed (pip install jsonschema for full validation)"
        return None

    @property
    def uses_schema(self) -> bool:
        return (
            self.schema_path is not None
            and _validator_for(str(self.schema_path), "RunUpsertRequest") is not None
        )

    def errors(self, definition: str, payload: Mapping[str, Any]) -> list[str]:
        """Return validation messages (empty when the payload is valid)."""
        validator = _validator_for(str(self.schema_path) if self.schema_path else None, definition)
        if validator is not None:
            messages = []
            for error in validator.iter_errors(payload):
                location = "/".join(str(part) for part in error.absolute_path)
                prefix = f"{definition}/{location}" if location else definition
                messages.append(f"{prefix}: {error.message}")
            return messages
        missing = [key for key in _REQUIRED.get(definition, ()) if key not in payload]
        return [f"{definition}: missing required key {key!r}" for key in missing]
