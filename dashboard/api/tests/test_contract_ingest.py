"""The Pydantic ingest models accept every contract example and reject malformed payloads."""

from __future__ import annotations

import copy
from typing import Any

import pytest
from jsonschema import Draft7Validator
from pydantic import BaseModel, ValidationError

from olmo_eval_api.schemas import ingest as s
from tests.helpers import load_example, load_schema

SCHEMA = load_schema("ingest-v1.schema.json")
INDEX: dict[str, str] = load_example("index.json")
INGEST_EXAMPLES = {k: v for k, v in INDEX.items() if k.startswith("ingest/")}
MODELS: dict[str, type[BaseModel]] = {
    "RunUpsertRequest": s.RunUpsertRequest,
    "RunUpsertResponse": s.RunUpsertResponse,
    "WhoAmIResponse": s.WhoAmIResponse,
    "SignArtifactsRequest": s.SignArtifactsRequest,
    "SignArtifactsResponse": s.SignArtifactsResponse,
    "TaskResultIn": s.TaskResultIn,
    "TaskResultUpsertResponse": s.TaskResultUpsertResponse,
    "InstanceBatchRequest": s.InstanceBatchRequest,
    "InstanceBatchResponse": s.InstanceBatchResponse,
    "InferenceUploadRequest": s.InferenceUploadRequest,
    "InferenceUploadResponse": s.InferenceUploadResponse,
    "CompleteRequest": s.CompleteRequest,
    "CompleteResponse": s.CompleteResponse,
}


def schema_errors(definition: str, payload: Any) -> list[str]:
    schema = {**SCHEMA, "$ref": f"#/definitions/{definition}"}
    return [e.message for e in Draft7Validator(schema).iter_errors(payload)]


@pytest.mark.parametrize(("path", "definition"), sorted(INGEST_EXAMPLES.items()))
def test_examples_validate(path: str, definition: str) -> None:
    payload = load_example(path)
    assert schema_errors(definition, payload) == []
    model = MODELS.get(definition)
    if model is None:
        assert definition == "ErrorResponse"
        return
    parsed = model.model_validate(payload)
    if model.model_config.get("extra") == "forbid":
        # Response models round-trip to JSON that still matches the schema.
        dumped = parsed.model_dump(mode="json")
        assert schema_errors(definition, dumped) == []


def test_every_ingest_definition_is_exercised() -> None:
    covered = set(INGEST_EXAMPLES.values())
    assert set(MODELS) <= covered | {"WhoAmIResponse"}


MALFORMED = [
    ("ingest/run-upsert-start.request.json", s.RunUpsertRequest, ["run", "run_id"], "BAD ID"),
    ("ingest/run-upsert-start.request.json", s.RunUpsertRequest, ["protocol_version"], 2),
    ("ingest/run-upsert-start.request.json", s.RunUpsertRequest, ["run", "status"], "done"),
    ("ingest/run-upsert-start.request.json", s.RunUpsertRequest, ["run", "tags"], ["bad tag"]),
    ("ingest/run-upsert-start.request.json", s.RunUpsertRequest, ["model"], None),
    ("ingest/task-result.request.json", s.TaskResultIn, ["num_instances"], -1),
    ("ingest/task-result.request.json", s.TaskResultIn, ["predictions_path"], "/abs/path"),
    ("ingest/task-result.request.json", s.TaskResultIn, ["requests_path"], "a/../b"),
    ("ingest/task-result.request.json", s.TaskResultIn, ["metrics"], {"acc": 0.5}),
    ("ingest/instances.request.json", s.InstanceBatchRequest, ["instances"], []),
    ("ingest/instances.request.json", s.InstanceBatchRequest, ["instances", 0, "native_id"], ""),
    (
        "ingest/instances.request.json",
        s.InstanceBatchRequest,
        ["instances", 0, "judge_verdict"],
        "x" * 65,
    ),
    (
        "ingest/artifacts-sign.request.json",
        s.SignArtifactsRequest,
        ["artifacts", 0, "md5_b64"],
        "abc",
    ),
    ("ingest/artifacts-sign.request.json", s.SignArtifactsRequest, ["artifacts", 0, "kind"], "zip"),
    (
        "ingest/artifacts-sign.request.json",
        s.SignArtifactsRequest,
        ["artifacts", 0, "size_bytes"],
        6 * 1024**3,
    ),
    ("ingest/complete.request.json", s.CompleteRequest, ["status"], "running"),
    ("ingest/inference.request.json", s.InferenceUploadRequest, ["batches", 0, "seq"], -1),
    (
        "ingest/instances.request.json",
        s.InstanceBatchRequest,
        ["instances", 0, "completion_tokens"],
        2**31,
    ),
    ("ingest/instances.request.json", s.InstanceBatchRequest, ["instances", 0, "doc_id"], 2**31),
    ("ingest/task-result.request.json", s.TaskResultIn, ["instance_count"], 2**31),
    ("ingest/run-upsert-final.request.json", s.RunUpsertRequest, ["run", "launch_id"], "x" * 65),
    (
        "ingest/complete-with-suites.request.json",
        s.CompleteRequest,
        ["suites", 0, "aggregation"],
        "x" * 33,
    ),
]


@pytest.mark.parametrize(("path", "model", "where", "value"), MALFORMED)
def test_malformed_rejected(path: str, model: type[BaseModel], where: list, value: Any) -> None:
    payload = copy.deepcopy(load_example(path))
    target = payload
    for key in where[:-1]:
        target = target[key]
    target[where[-1]] = value
    with pytest.raises(ValidationError):
        model.model_validate(payload)
    assert schema_errors(INDEX[path], payload), "the schema should reject it too"


def test_unknown_fields_ignored() -> None:
    payload = load_example("ingest/run-upsert-start.request.json")
    payload["run"]["future_field"] = 1
    s.RunUpsertRequest.model_validate(payload)
