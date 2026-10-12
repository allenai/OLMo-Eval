"""Pydantic models for the ingest protocol v1 (dashboard/contract/ingest-types.ts).

Request models ignore unknown fields so an older server tolerates new optional fields from a
newer client. The JSON Schema in the contract stays strict and is enforced in tests.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field

RUN_ID_PATTERN = r"^[a-z0-9]{6,32}$"
TAG_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:/+-]{0,63}$"
RELATIVE_PATH_PATTERN = r"^(?!/)(?!.*(^|/)\.\.?(/|$))[^\x00]{1,1024}$"
MAX_ARTIFACT_BYTES = 5 * 1024**3

RunId = Annotated[str, Field(pattern=RUN_ID_PATTERN)]
Tag = Annotated[str, Field(pattern=TAG_PATTERN)]
NonNeg = Annotated[int, Field(ge=0, le=2**63 - 1)]
# Counts and sizes stored in 32-bit columns.
NonNeg32 = Annotated[int, Field(ge=0, le=2**31 - 1)]
NestedMetrics = dict[str, dict[str, float | None]]
FlatMetrics = dict[str, float | None]
RunStatus = Literal["running", "complete", "partial", "failed"]
FinalRunStatus = Literal["complete", "partial", "failed"]
UploadState = Literal["uploading", "complete"]
ArtifactKind = Literal[
    "metrics",
    "manifest",
    "predictions",
    "requests",
    "inference_metrics",
    "logs",
    "traces",
    "other",
]


def _relative_path_ok(value: str) -> str:
    if value.startswith("/") or "\x00" in value or not 1 <= len(value) <= 1024:
        raise ValueError("must be a relative POSIX path")
    if any(part in ("", ".", "..") for part in value.split("/")):
        raise ValueError("must not contain empty, '.' or '..' segments")
    return value


class _In(BaseModel):
    model_config = ConfigDict(extra="ignore")


class _Out(BaseModel):
    model_config = ConfigDict(extra="forbid")


# Pydantic's regex engine does not support the lookaheads in RELATIVE_PATH_PATTERN, so the same
# rule is checked in Python.
RelativePath = Annotated[str, AfterValidator(_relative_path_ok)]


class ClientInfo(_In):
    name: str
    version: str


class GitInfo(_In):
    repo: str | None = None
    commit: str | None = None
    branch: str | None = None
    dirty: bool | None = None


class BeakerInfo(_In):
    experiment_id: str | None = None
    workload_id: str | None = None
    job_id: str | None = None
    task_id: str | None = None
    workspace: str | None = None
    workspace_id: str | None = None
    result_dataset_id: str | None = None
    node_hostname: str | None = None
    cluster: str | None = None
    priority: str | None = None
    image: str | None = None
    budget: str | None = None
    gpu_count: NonNeg | None = None
    cpu_count: float | None = None


class EnvironmentInfo(_In):
    hostname: str | None = None
    platform: str | None = None
    python_version: str | None = None
    packages: dict[str, str] = Field(default_factory=dict)
    cuda_version: str | None = None
    gpu_type: str | None = None
    gpu_count: NonNeg | None = None


class RunError(_In):
    task: str | None
    error: str


class RunInfo(_In):
    run_id: RunId
    launch_id: str | None = Field(max_length=64)
    experiment_name: str | None
    experiment_group: str | None
    status: RunStatus
    started_at: datetime | None
    finished_at: datetime | None
    duration_seconds: float | None
    author: str | None
    tags: list[Tag] = Field(max_length=50)
    olmo_eval_version: str | None
    git: GitInfo
    beaker: BeakerInfo | None
    environment: EnvironmentInfo
    argv: list[str]
    task_specs: list[str]
    output_dir: str | None
    harness_config: dict[str, Any] | None
    provider_init_seconds: dict[str, float] | None
    startup_seconds: float | None = None
    processing_started_at: datetime | None = None
    processing_seconds: float | None = None
    errors: list[RunError]


class ModelInfo(_In):
    name: str = Field(min_length=1)
    path: str
    model_hash: str | None
    revision: str | None
    provider_kind: str
    provider_config: dict[str, Any]
    family: str | None = None
    series: str | None = None
    step: NonNeg | None = None
    tokens_seen: NonNeg | None = None


class RunUpsertRequest(_In):
    protocol_version: Literal[1]
    client: ClientInfo
    run: RunInfo
    model: ModelInfo


class RunUpsertResponse(_Out):
    run_id: str
    created: bool
    status: RunStatus
    upload_state: UploadState
    gcs_prefix: str
    dashboard_url: str
    uploaded_by: str


class WhoAmIResponse(_Out):
    email: str
    principal_type: Literal["user", "service_account"]
    expires_in: NonNeg
    protocol_versions: list[int]


class ArtifactIn(_In):
    path: RelativePath
    size_bytes: int = Field(ge=0, le=MAX_ARTIFACT_BYTES)
    md5_b64: str = Field(pattern=r"^[A-Za-z0-9+/]{22}==$")
    content_type: str
    kind: ArtifactKind
    task_name: str | None = None


class SignArtifactsRequest(_In):
    artifacts: list[ArtifactIn] = Field(min_length=1, max_length=500)


class SignedUpload(_Out):
    path: str
    gs_uri: str
    skip: bool
    url: str | None
    method: Literal["PUT"] = "PUT"
    headers: dict[str, str]
    expires_at: datetime | None


class SignArtifactsResponse(_Out):
    uploads: list[SignedUpload]


class MetricMetaIn(_In):
    higher_is_better: bool | None = None
    display_format: Literal["percent", "raw"] | None = None
    unit: str | None = None


class TaskResultIn(_In):
    task_name: str = Field(min_length=1)
    task_hash: str = Field(min_length=1, max_length=64)
    base_task: str | None
    primary_metric: str | None
    metrics: NestedMetrics
    metric_meta: dict[str, MetricMetaIn]
    config: dict[str, Any]
    num_fewshot: NonNeg32 | None
    limit: NonNeg32 | None
    split: str | None
    num_instances: NonNeg32
    instances_processed: NonNeg32 | None
    instances_failed: NonNeg32 | None
    error: str | None
    error_summary: dict[str, Any] | None
    duration_seconds: float | None
    first_request_at: datetime | None = None
    last_completed_at: datetime | None = None
    prompt_tokens_total: NonNeg | None = None
    completion_tokens_total: NonNeg | None = None
    attributed_inference_seconds: float | None = None
    predictions_path: RelativePath | None
    requests_path: RelativePath | None
    suites: list[str]
    instance_count: NonNeg32


class TaskResultUpsertResponse(_Out):
    task_result_id: int
    created: bool
    instances_cleared: bool
    unchanged: bool = False


class InstanceIn(_In):
    native_id: str = Field(min_length=1, max_length=512)
    doc_id: NonNeg32 | None
    primary_score: float | None
    metrics: FlatMetrics = Field(max_length=200)
    label: str | None = Field(max_length=256)
    finish_reason: str | None
    completion_tokens: NonNeg32 | None
    prompt_tokens: NonNeg32 | None
    num_outputs: NonNeg
    extracted_answer: str | None = Field(max_length=256)
    prompt_preview: str | None = Field(max_length=300)
    output_preview: str | None = Field(max_length=300)
    judge_verdict: str | None = Field(max_length=64)
    has_scoring_error: bool
    has_execution_result: bool
    has_judge_result: bool
    has_trajectory: bool
    pred_offset: NonNeg | None
    pred_length: NonNeg32 | None
    req_offset: NonNeg | None
    req_length: NonNeg32 | None


class InstanceBatchRequest(_In):
    batch_index: NonNeg
    instances: list[InstanceIn] = Field(min_length=1, max_length=2000)


class InstanceBatchResponse(_Out):
    task_result_id: int
    received: NonNeg
    total_stored: NonNeg


class InferenceBatchIn(_In):
    seq: NonNeg32
    timestamp: datetime
    task_name: str | None
    total_requests: NonNeg32
    successful_requests: NonNeg32
    failed_requests: NonNeg32
    total_prompt_tokens: NonNeg
    total_completion_tokens: NonNeg
    wall_clock_time_s: float
    output_tokens_per_second: float
    mean_latency_s: float
    gpu_summary: dict[str, float | None] | None


class SeriesIn(_In):
    name: str
    unit: str | None
    device: str | None
    points: list[tuple[float, float]] = Field(max_length=500)


class LatencyQuantiles(_In):
    p5: float
    p25: float
    p50: float
    p75: float
    p95: float
    mean: float | None = None
    n: NonNeg


class RequestLatencyIn(_In):
    end_to_end_s: LatencyQuantiles | None
    ttft_s: LatencyQuantiles | None
    tpot_s: LatencyQuantiles | None


class GpuDeviceIn(_In):
    device_id: NonNeg
    name: str
    memory_total_mb: float | None


class InferenceUploadRequest(_In):
    source_paths: list[RelativePath]
    batches: list[InferenceBatchIn] = Field(max_length=20000)
    series: list[SeriesIn] = Field(max_length=64)
    request_latency: RequestLatencyIn | None
    gpu_devices: list[GpuDeviceIn]


class InferenceUploadResponse(_Out):
    batches_stored: NonNeg
    series_stored: NonNeg


class SuiteChildIn(_In):
    type: Literal["task", "suite"]
    name: str


class SuiteResultIn(_In):
    name: str = Field(min_length=1)
    aggregation: str = Field(max_length=32)
    parent: str | None
    description: str | None
    children: list[SuiteChildIn]
    metrics: NestedMetrics
    primary_metric: str | None
    score: float | None
    num_tasks: NonNeg32 | None


class CompleteRequest(_In):
    status: FinalRunStatus
    finished_at: datetime | None
    duration_seconds: float | None
    suites: list[SuiteResultIn]
    expected_task_results: NonNeg
    expected_artifacts: NonNeg


class CompleteResponse(_Out):
    run_id: str
    status: FinalRunStatus
    upload_state: UploadState
    dashboard_url: str
    task_results: NonNeg
    instances: NonNeg
    artifacts_missing: list[str]
    warnings: list[str]
