/**
 * Ingest API, protocol version 1.
 *
 * The upload protocol between olmo-eval (the client, src/olmo_eval/upload/) and
 * the ingest service (dashboard/api with API_MODE=ingest). This file is the
 * source of truth. ingest-v1.schema.json is generated from it with
 * `npm --prefix dashboard/ui run contract:generate`; do not edit the schema by hand.
 *
 * Conventions:
 * - JSON bodies, snake_case keys, UTC ISO 8601 timestamps with an offset.
 * - Scores are raw metric values exactly as olmo-eval computed them (accuracy 0.741,
 *   never 74.1). Formatting is the dashboard's job.
 * - Every endpoint is idempotent, so the client may retry any request.
 * - Errors use ErrorResponse with an HTTP status code.
 */

// ---------------------------------------------------------------------------
// Scalars
// ---------------------------------------------------------------------------

/** @asType integer */
export type Integer = number;

/**
 * @asType integer
 * @minimum 0
 * @maximum 9223372036854775807
 */
export type NonNegativeInteger = number;

/**
 * A count or size stored in a 32-bit column.
 * @asType integer
 * @minimum 0
 * @maximum 2147483647
 */
export type NonNegativeInt32 = number;

/** @format date-time */
export type DateTime = string;

/** Arbitrary JSON object (configs, error summaries). */
export type JsonObject = { [key: string]: unknown };

/**
 * Run ID. olmo-eval uses its experiment_id (12 lowercase hex characters).
 * @pattern ^[a-z0-9]{6,32}$
 */
export type RunId = string;

/**
 * Free-form label attached to a run.
 * @pattern ^[A-Za-z0-9][A-Za-z0-9._:/+-]{0,63}$
 */
export type Tag = string;

/**
 * Path of a file relative to the run's output directory, POSIX separators,
 * no leading slash, no "." or ".." segments.
 * @pattern ^(?!/)(?!.*(^|/)\.\.?(/|$))[^\x00]{1,1024}$
 */
export type RelativePath = string;

/** Nested metrics exactly as metrics.json stores them: {metric_name: {scorer_name: value}}. */
export type NestedMetrics = { [metric: string]: { [scorer: string]: number | null } };

/** Flat metrics keyed by "metric_name:scorer_name". */
export type FlatMetrics = { [metricKey: string]: number | null };

export type RunStatus = "running" | "complete" | "partial" | "failed";
export type FinalRunStatus = "complete" | "partial" | "failed";
export type UploadState = "uploading" | "complete";

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

export interface ErrorBody {
    /** Machine-readable code: bad_request, unauthenticated, forbidden, not_found, conflict, payload_too_large, validation_error, internal. */
    code: string;
    message: string;
    request_id: string;
    details?: unknown;
}

/** Body of every non-2xx response. */
export interface ErrorResponse {
    error: ErrorBody;
}

// ---------------------------------------------------------------------------
// GET /v1/whoami
// ---------------------------------------------------------------------------

export interface WhoAmIResponse {
    /** Verified email from the Google access token. */
    email: string;
    principal_type: "user" | "service_account";
    /** Seconds until the presented token expires. */
    expires_in: NonNegativeInteger;
    /** Ingest protocol versions this server accepts. */
    protocol_versions: Integer[];
}

// ---------------------------------------------------------------------------
// PUT /v1/runs/{run_id}
// ---------------------------------------------------------------------------

export interface ClientInfo {
    /** Always "olmo-eval" for the official client. */
    name: string;
    version: string;
}

export interface GitInfo {
    /** "owner/name", e.g. "allenai/olmo-eval" (GITHUB_REPO in Beaker). */
    repo: string | null;
    /** Full SHA when known (GIT_REF in Beaker), else a short SHA. */
    commit: string | null;
    branch: string | null;
    /** True when the local working tree had uncommitted changes. */
    dirty: boolean | null;
}

/** Beaker metadata captured from the job environment. All null outside Beaker. */
export interface BeakerInfo {
    experiment_id: string | null;
    workload_id: string | null;
    job_id: string | null;
    task_id: string | null;
    /** Workspace full name, e.g. "ai2/olmo-eval-debug". */
    workspace: string | null;
    workspace_id: string | null;
    result_dataset_id: string | null;
    node_hostname: string | null;
    /** Cluster, e.g. "ai2/jupiter". From OLMO_EVAL_BEAKER_CLUSTER or derived from node_hostname. */
    cluster: string | null;
    priority: string | null;
    /** Beaker image or Docker image the job ran. */
    image: string | null;
    budget: string | null;
    gpu_count: NonNegativeInteger | null;
    cpu_count: number | null;
}

export interface EnvironmentInfo {
    hostname: string | null;
    platform: string | null;
    python_version: string | null;
    /** Installed distribution versions: "olmo-eval", "vllm", "torch", "transformers", "datasets", "litellm" when present. */
    packages: { [distribution: string]: string };
    cuda_version: string | null;
    /** e.g. "NVIDIA H100 80GB HBM3". */
    gpu_type: string | null;
    gpu_count: NonNegativeInteger | null;
}

export interface RunError {
    /** Task spec the error belongs to, or null for a run-level error. */
    task: string | null;
    error: string;
}

export interface RunInfo {
    run_id: RunId;
    /** Shared by all runs created by one `olmo-eval beaker launch`. Null for local runs. */
    /** @maxLength 64 */
    launch_id: string | null;
    experiment_name: string | null;
    experiment_group: string | null;
    status: RunStatus;
    started_at: DateTime | null;
    finished_at: DateTime | null;
    duration_seconds: number | null;
    /** BEAKER_AUTHOR, else $USER. Display only; the server records the verified uploader separately. */
    author: string | null;
    /** @maxItems 50 */
    tags: Tag[];
    olmo_eval_version: string | null;
    git: GitInfo;
    beaker: BeakerInfo | null;
    environment: EnvironmentInfo;
    /** ["olmo-eval", ...sys.argv[1:]]. */
    argv: string[];
    /** Original -t values as typed, including suite names. */
    task_specs: string[];
    output_dir: string | null;
    /** HarnessConfig.to_dict() (metrics.json "config"). */
    harness_config: JsonObject | null;
    provider_init_seconds: { [worker: string]: number } | null;
    /**
     * Seconds from olmo-eval starting until every inference worker was ready (model download
     * and load, server start). Excludes Beaker queueing and job setup before olmo-eval starts.
     */
    startup_seconds?: number | null;
    /** When every inference worker was ready. Task spans are measured from here. */
    processing_started_at?: DateTime | null;
    /** Seconds from processing_started_at until the last task finished (inference and scoring). */
    processing_seconds?: number | null;
    errors: RunError[];
}

export interface ModelInfo {
    /** Display name: provider alias if set, else provider model. */
    name: string;
    /** provider.model: HF ID, local path, s3://, gs:// or /weka path. */
    path: string;
    /** olmo-eval compute_model_hash (16 hex). */
    model_hash: string | null;
    revision: string | null;
    /** vllm, vllm_server, hf, olmo_core, olmo_core_vlm, litellm, mock, external, ... */
    provider_kind: string;
    /** ProviderConfig.to_dict() plus attention_backend. */
    provider_config: JsonObject;
    /** Optional explicit values. The server derives them from name/path/revision when absent. */
    family?: string | null;
    series?: string | null;
    step?: NonNegativeInteger | null;
    tokens_seen?: NonNegativeInteger | null;
}

/**
 * Create or update a run. Sent once when the run starts (status "running") and
 * again at the end with the final status. Re-sending is safe.
 */
export interface RunUpsertRequest {
    protocol_version: 1;
    client: ClientInfo;
    run: RunInfo;
    model: ModelInfo;
}

export interface RunUpsertResponse {
    run_id: RunId;
    /** True when this request created the run. */
    created: boolean;
    /** Status after the upsert (a final status is never downgraded to "running"). */
    status: RunStatus;
    upload_state: UploadState;
    /** gs:// URI of the run's artifact prefix, ending in "/". */
    gcs_prefix: string;
    dashboard_url: string;
    /** Verified email of the principal that first uploaded the run. */
    uploaded_by: string;
}

// ---------------------------------------------------------------------------
// POST /v1/runs/{run_id}/artifacts:sign
// ---------------------------------------------------------------------------

export type ArtifactKind =
    | "metrics"
    | "manifest"
    | "predictions"
    | "requests"
    | "inference_metrics"
    | "logs"
    | "traces"
    | "other";

export interface ArtifactIn {
    path: RelativePath;
    /**
     * @asType integer
     * @minimum 0
     * @maximum 5368709120
     */
    size_bytes: number;
    /**
     * Base64 MD5 of the file contents (what GCS reports as md5Hash).
     * @pattern ^[A-Za-z0-9+/]{22}==$
     */
    md5_b64: string;
    content_type: string;
    kind: ArtifactKind;
    /** Task spec for predictions/requests files, else null. */
    task_name?: string | null;
}

export interface SignArtifactsRequest {
    /**
     * @minItems 1
     * @maxItems 500
     */
    artifacts: ArtifactIn[];
}

export interface SignedUpload {
    path: RelativePath;
    gs_uri: string;
    /** True when an object with the same size and MD5 already exists; do not upload. */
    skip: boolean;
    /** V4 signed URL, null when skip is true. */
    url: string | null;
    method: "PUT";
    /**
     * Headers the PUT must send exactly (Content-Type, Content-MD5, and
     * x-goog-content-length-range, which makes GCS reject a body of another size).
     */
    headers: { [name: string]: string };
    expires_at: DateTime | null;
}

export interface SignArtifactsResponse {
    uploads: SignedUpload[];
}

// ---------------------------------------------------------------------------
// POST /v1/runs/{run_id}/task-results
// ---------------------------------------------------------------------------

/** Per-metric metadata resolved by olmo-eval from live Metric objects. Keyed by metric name (no scorer). */
export interface MetricMetaIn {
    /** Metric.higher_is_better(), null when unknown. */
    higher_is_better: boolean | null;
    /** Metric.display_format(). */
    display_format: "percent" | "raw" | null;
    /** Metric.metric_unit(). */
    unit: string | null;
}

/**
 * Create or replace one task result. Upsert key: (run_id, task_name, task_hash).
 * Replacing deletes the task result's stored instances; the client then sends all
 * instances again.
 */
export interface TaskResultIn {
    /** Task spec as run (metrics.json tasks[].task) with any "@priority" suffix removed. */
    task_name: string;
    /** metrics.json tasks[].task_hash (16 hex). */
    task_hash: string;
    /** Registered base task name, e.g. "gsm8k" for "gsm8k:cot:olmo3". */
    base_task: string | null;
    /** "metric_name:scorer_name"; tasks[].primary_metric, else summary[task].metric. */
    primary_metric: string | null;
    metrics: NestedMetrics;
    metric_meta: { [metricName: string]: MetricMetaIn };
    /** TaskConfig.to_dict() (metrics.json tasks[].config). */
    config: JsonObject;
    num_fewshot: NonNegativeInt32 | null;
    limit: NonNegativeInt32 | null;
    split: string | null;
    /** Scored and saved instances (tasks[].num_instances). */
    num_instances: NonNegativeInt32;
    instances_processed: NonNegativeInt32 | null;
    instances_failed: NonNegativeInt32 | null;
    error: string | null;
    error_summary: JsonObject | null;
    /**
     * Seconds from the run's processing_started_at until this task finished. Tasks share
     * inference workers, so this is not the task's own cost; see first_request_at and the
     * token totals for that.
     */
    duration_seconds: number | null;
    /** When the task's first request was sent to an inference worker. */
    first_request_at?: DateTime | null;
    /** When the task's last instance finished. */
    last_completed_at?: DateTime | null;
    /** Prompt tokens over every request of the task, including failed instances. */
    prompt_tokens_total?: NonNegativeInteger | null;
    /** Completion tokens over every request of the task, including failed instances. */
    completion_tokens_total?: NonNegativeInteger | null;
    /**
     * Inference seconds attributed to this task from per-batch inference metrics (each batch's
     * wall time split by the task's share of the batch's tokens). Null unless recorded.
     */
    attributed_inference_seconds?: number | null;
    predictions_path: RelativePath | null;
    requests_path: RelativePath | null;
    /** Registered suites that contain this task spec (static membership). */
    suites: string[];
    /** Number of instance rows the client will send for this task result (0 without predictions). */
    instance_count: NonNegativeInt32;
}

export interface TaskResultUpsertResponse {
    task_result_id: Integer;
    created: boolean;
    /** True when previously stored instances were deleted. */
    instances_cleared: boolean;
    /**
     * True when the stored task result already matches this request and its files, so its
     * instances were kept and need not be sent again.
     */
    unchanged: boolean;
}

// ---------------------------------------------------------------------------
// POST /v1/task-results/{task_result_id}/instances
// ---------------------------------------------------------------------------

/**
 * Compact per-instance row extracted by olmo-eval from the predictions and requests
 * JSONL files. Full records stay in GCS and are read by byte range.
 */
export interface InstanceIn {
    /**
     * str(native_id), truncated to 512 characters. Unique within a task result; duplicates get a
     * "#<n>" suffix, added within the 512-character limit.
     * @minLength 1
     * @maxLength 512
     */
    native_id: string;
    doc_id: NonNegativeInt32 | null;
    /** Value of the task's primary metric for this instance, null when absent. */
    primary_score: number | null;
    /**
     * All finite per-instance metric values, flattened to "metric:scorer".
     * @maxProperties 200
     */
    metrics: FlatMetrics;
    /**
     * Gold label, JSON-encoded when not a string, truncated.
     * @maxLength 256
     */
    label: string | null;
    /** "length" if any sample was truncated, else model_output[0].finish_reason. */
    finish_reason: string | null;
    /** Sum over all generated samples (multiple choice: model_output[0]). */
    completion_tokens: NonNegativeInt32 | null;
    /** Prompt tokens for this instance (log-likelihood requests: context plus continuation). */
    prompt_tokens: NonNegativeInt32 | null;
    /** len(model_output). */
    num_outputs: NonNegativeInteger;
    /** @maxLength 256 */
    extracted_answer: string | null;
    /**
     * First 300 characters of the question: request doc.query, else the last user
     * message, else the end of the prompt string.
     * @maxLength 300
     */
    prompt_preview: string | null;
    /**
     * First 300 characters of model_output[0].text (or final_output).
     * @maxLength 300
     */
    output_preview: string | null;
    /**
     * judge_result when it is a string, else judge_result.verdict, truncated.
     * @maxLength 64
     */
    judge_verdict: string | null;
    has_scoring_error: boolean;
    has_execution_result: boolean;
    has_judge_result: boolean;
    has_trajectory: boolean;
    /** Byte offset and length (without the newline) of this instance's line in predictions_path. */
    pred_offset: NonNegativeInteger | null;
    pred_length: NonNegativeInt32 | null;
    /** Byte offset and length of the matching line in requests_path, null when not found. */
    req_offset: NonNegativeInteger | null;
    req_length: NonNegativeInt32 | null;
}

export interface InstanceBatchRequest {
    batch_index: NonNegativeInteger;
    /**
     * @minItems 1
     * @maxItems 2000
     */
    instances: InstanceIn[];
}

export interface InstanceBatchResponse {
    task_result_id: Integer;
    received: NonNegativeInteger;
    /** Instances stored for this task result after this batch. */
    total_stored: NonNegativeInteger;
}

// ---------------------------------------------------------------------------
// PUT /v1/runs/{run_id}/inference
// ---------------------------------------------------------------------------

/** One BatchMetrics line from metrics/*-inference.jsonl without gpu_devices or requests. */
export interface InferenceBatchIn {
    /** Order within the run, starting at 0. */
    seq: NonNegativeInt32;
    timestamp: DateTime;
    /** BatchMetrics.task_name (null today: chunks mix tasks). */
    task_name: string | null;
    total_requests: NonNegativeInt32;
    successful_requests: NonNegativeInt32;
    failed_requests: NonNegativeInt32;
    total_prompt_tokens: NonNegativeInteger;
    total_completion_tokens: NonNegativeInteger;
    wall_clock_time_s: number;
    output_tokens_per_second: number;
    mean_latency_s: number;
    /** BatchMetrics gpu_summary as written. */
    gpu_summary: { [key: string]: number | null } | null;
}

export type SeriesPoint = [number, number];

/** A downsampled time series. x is seconds since the run's started_at. */
export interface SeriesIn {
    /** gpu_utilization_pct, gpu_memory_used_mb, vllm_num_requests_running, vllm_num_requests_waiting, vllm_kv_cache_usage_pct, ... */
    name: string;
    unit: string | null;
    /** GPU device id as a string for per-device series, else null. */
    device: string | null;
    /** @maxItems 500 */
    points: SeriesPoint[];
}

export interface LatencyQuantiles {
    p5: number;
    p25: number;
    p50: number;
    p75: number;
    p95: number;
    /** Arithmetic mean. Optional because older clients sent quantiles only. */
    mean?: number;
    n: NonNegativeInteger;
}

export interface RequestLatencyIn {
    end_to_end_s: LatencyQuantiles | null;
    ttft_s: LatencyQuantiles | null;
    tpot_s: LatencyQuantiles | null;
}

export interface GpuDeviceIn {
    device_id: NonNegativeInteger;
    name: string;
    memory_total_mb: number | null;
}

/** Replaces all inference data for the run. */
export interface InferenceUploadRequest {
    /** Relative paths of the source JSONL files. */
    source_paths: RelativePath[];
    /** @maxItems 20000 */
    batches: InferenceBatchIn[];
    /** @maxItems 64 */
    series: SeriesIn[];
    /** Present only when per-request metrics were recorded. */
    request_latency: RequestLatencyIn | null;
    gpu_devices: GpuDeviceIn[];
}

export interface InferenceUploadResponse {
    batches_stored: NonNegativeInteger;
    series_stored: NonNegativeInteger;
}

// ---------------------------------------------------------------------------
// POST /v1/runs/{run_id}/complete
// ---------------------------------------------------------------------------

export interface SuiteChildIn {
    type: "task" | "suite";
    name: string;
}

/** One suite aggregation from metrics/results "suites", plus its definition. */
export interface SuiteResultIn {
    name: string;
    /** none, average, weighted_average, average_of_averages, display_only. */
    /** @maxLength 32 */
    aggregation: string;
    /** Parent suite name for nested suites, else null. */
    parent: string | null;
    description: string | null;
    /** Direct children from the registered Suite definition (or the contributing tasks when unregistered). */
    children: SuiteChildIn[];
    metrics: NestedMetrics;
    /** "primary_score:average" for olmo-eval suites. */
    primary_metric: string | null;
    /** Suite primary score, null for display_only/none. */
    score: number | null;
    num_tasks: NonNegativeInt32 | null;
}

/**
 * Finish an upload. The server recomputes derived data (stderr, metric kinds,
 * suite stderr, score vectors, run summary), verifies artifacts, and marks the
 * upload complete. Idempotent.
 */
export interface CompleteRequest {
    status: FinalRunStatus;
    finished_at: DateTime | null;
    duration_seconds: number | null;
    suites: SuiteResultIn[];
    expected_task_results: NonNegativeInteger;
    expected_artifacts: NonNegativeInteger;
}

export interface CompleteResponse {
    run_id: RunId;
    status: FinalRunStatus;
    upload_state: UploadState;
    dashboard_url: string;
    task_results: NonNegativeInteger;
    instances: NonNegativeInteger;
    /** Artifact paths that were signed but are missing from GCS. */
    artifacts_missing: RelativePath[];
    warnings: string[];
}
