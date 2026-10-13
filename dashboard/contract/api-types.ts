/**
 * Dashboard read API (served under /api by dashboard/api with API_MODE=dashboard).
 *
 * The UI imports these types; the server's Pydantic response models must produce
 * JSON that validates against api-v1.schema.json, which is generated from this file
 * with `npm --prefix dashboard/ui run contract:generate`. Do not edit the schema by hand.
 *
 * Conventions:
 * - snake_case keys, UTC ISO 8601 timestamps, raw metric values (0.741, not 74.1).
 * - Lists use keyset pagination: `limit` (default 50, max 500) and an opaque `cursor`;
 *   responses carry `next_cursor` (null on the last page) and an exact `total`.
 * - Every response has an X-Request-Id header; errors use ErrorResponse.
 * - Free-text identifiers (task, suite, group, model series, native_id) travel as
 *   query parameters, never as path segments.
 * - Subject keys: "r:<run_id>" for a run, "m:<model_id>" for a model checkpoint whose
 *   task results merge as "latest result per task".
 */

// ---------------------------------------------------------------------------
// Scalars
// ---------------------------------------------------------------------------

/** @asType integer */
export type Integer = number;

/**
 * @asType integer
 * @minimum 0
 */
export type Count = number;

/** @format date-time */
export type DateTime = string;

/** YYYY-MM-DD in UTC. */
export type DateString = string;

export type JsonObject = { [key: string]: unknown };

/** @pattern ^[a-z0-9]{6,32}$ */
export type RunId = string;

/**
 * 12 hex chars: sha256(name + "\u0000" + model_hash)[:12].
 * @pattern ^[0-9a-f]{12}$
 */
export type ModelId = string;

/** @pattern ^(r:[a-z0-9]{6,32}|m:[0-9a-f]{12})$ */
export type SubjectKey = string;

/** Database ID of a task result. */
export type TaskResultId = Integer;

/** Flat metrics keyed by "metric_name:scorer_name". */
export type FlatMetrics = { [metricKey: string]: number | null };

export type RunStatus = "running" | "complete" | "partial" | "failed";
export type UploadState = "uploading" | "complete";
export type MetricKind = "binary" | "bounded" | "unbounded";
export type DisplayFormat = "percent" | "raw";

// ---------------------------------------------------------------------------
// Errors, health, identity
// ---------------------------------------------------------------------------

export interface ErrorBody {
    code: string;
    message: string;
    request_id: string;
    details?: unknown;
}

export interface ErrorResponse {
    error: ErrorBody;
}

/** GET /health and GET /api/health */
export interface HealthResponse {
    status: "ok";
    mode: "dashboard" | "ingest" | "all";
    git_sha: string | null;
    skiff_env: string;
    db: "ok";
}

/** GET /api/me */
export interface MeResponse {
    email: string;
    /** Local part of the email; what `user=me` filters resolve to. */
    username: string;
    /** True when the identity came from DASHBOARD_DEV_USER (local development only). */
    dev_mode: boolean;
}

// ---------------------------------------------------------------------------
// Shared objects
// ---------------------------------------------------------------------------

export interface MetricMeta {
    higher_is_better: boolean | null;
    display_format: DisplayFormat;
    unit: string | null;
    kind: MetricKind;
}

export type DeltaMethod = "paired_bootstrap" | "unpaired" | "insufficient" | "none";

/** Difference "this minus reference" with uncertainty. See spec section 5. */
export interface DeltaStats {
    delta: number | null;
    ci_low: number | null;
    ci_high: number | null;
    p_value: number | null;
    n_shared: Count;
    method: DeltaMethod;
    /** CI excludes zero (paired_bootstrap or unpaired only). */
    significant: boolean;
    /** True when delta is in the better direction, null when direction unknown or delta is 0/null. */
    improved: boolean | null;
    /** Task hashes differ between the two sides. */
    hash_mismatch: boolean;
    alpha: number;
    n_boot: Count | null;
    /** Human-readable reason when method is insufficient or none. */
    note: string | null;
}

/** 2x2 counts; "a" is this run/subject, "b" the reference (baseline). */
export interface Contingency {
    both_right: Count;
    only_a: Count;
    only_b: Count;
    both_wrong: Count;
    n_shared: Count;
    /** only_a - only_b */
    net: Integer;
    threshold: number;
}

export interface ModelRef {
    model_id: ModelId;
    name: string;
    model_hash: string;
    series: string;
    /** Short display label for the series (basename when the series is a path). */
    series_label: string;
    family: string | null;
    step: Count | null;
    tokens_seen: Count | null;
}

export interface RunLinks {
    beaker_experiment: string | null;
    beaker_result_dataset: string | null;
    beaker_workspace: string | null;
    gcs_console: string | null;
    github_commit: string | null;
}

export interface Headline {
    kind: "suite" | "task" | "mean";
    name: string | null;
    score: number | null;
    stderr: number | null;
    display_format: DisplayFormat;
}

export interface RunSummary {
    run_id: RunId;
    launch_id: string | null;
    experiment_name: string | null;
    experiment_group: string | null;
    status: RunStatus;
    upload_state: UploadState;
    /** status is "running" and the run has not been updated for 24 hours. */
    stale: boolean;
    model: ModelRef;
    author: string | null;
    uploaded_by: string;
    tags: string[];
    created_at: DateTime;
    started_at: DateTime | null;
    finished_at: DateTime | null;
    duration_seconds: number | null;
    num_tasks: Count;
    num_failed_tasks: Count;
    num_instances: Count;
    headline: Headline | null;
    beaker_experiment_id: string | null;
    beaker_workspace: string | null;
    git_commit: string | null;
    git_branch: string | null;
    gcs_prefix: string;
    links: RunLinks;
}

export interface SubjectInfo {
    key: SubjectKey;
    kind: "run" | "model";
    label: string;
    model: ModelRef;
    /** Set for run subjects. */
    run: RunSummary | null;
    /** Runs that contribute task results (one for run subjects). */
    run_ids: RunId[];
    latest_at: DateTime | null;
}

export interface Histogram {
    edges: number[];
    counts: Count[];
}

export interface BoxStats {
    p5: number;
    p25: number;
    p50: number;
    p75: number;
    p95: number;
    mean: number;
    n: Count;
}

// ---------------------------------------------------------------------------
// GET /api/search?q=&types=&limit=
// ---------------------------------------------------------------------------

export type SearchResultType = "run" | "model" | "task" | "suite" | "group" | "user" | "id";

export interface SearchResult {
    type: SearchResultType;
    /** Stable key: run_id, model series, task name, suite name, group, username. */
    key: string;
    label: string;
    secondary: string | null;
    /** UI route to open, e.g. "/runs/3f2a1b4c5d6e". */
    href: string;
    /** Subject key when the result can be added to compare or set as baseline. */
    subject: SubjectKey | null;
}

export interface SearchGroup {
    type: SearchResultType;
    items: SearchResult[];
}

export interface SearchResponse {
    query: string;
    groups: SearchGroup[];
}

// ---------------------------------------------------------------------------
// GET /api/runs  and  GET /api/runs/facets
// ---------------------------------------------------------------------------

export interface ScoreColumnMeta {
    /** "task:<task_name>" or "suite:<suite_name>" as requested in `cols`. */
    key: string;
    kind: "task" | "suite";
    name: string;
    display_format: DisplayFormat;
    higher_is_better: boolean | null;
}

export interface ScoreColumnValue {
    score: number | null;
    stderr: number | null;
    n: Count | null;
    task_result_id: TaskResultId | null;
    /** Vs `baseline` when given. Runs-list deltas are always method "unpaired". */
    delta: DeltaStats | null;
}

export interface RunRow extends RunSummary {
    /** Keyed by ScoreColumnMeta.key. */
    scores: { [columnKey: string]: ScoreColumnValue };
}

export interface RunsListResponse {
    items: RunRow[];
    next_cursor: string | null;
    total: Count;
    columns: ScoreColumnMeta[];
    baseline: SubjectInfo | null;
}

export interface FacetValue {
    value: string;
    count: Count;
}

export interface RunsFacetsResponse {
    user: FacetValue[];
    family: FacetValue[];
    group: FacetValue[];
    workspace: FacetValue[];
    tag: FacetValue[];
    status: FacetValue[];
    model: FacetValue[];
}

/** PATCH /api/runs/{run_id} */
export interface RunPatchRequest {
    tags?: string[];
    notes?: string | null;
}

/** POST /api/runs/tags */
export interface BulkTagRequest {
    /** @maxItems 500 */
    run_ids: RunId[];
    add: string[];
    remove: string[];
}

export interface BulkTagResponse {
    updated: Count;
}

// ---------------------------------------------------------------------------
// GET /api/runs/{run_id}
// ---------------------------------------------------------------------------

export interface ModelDetail extends ModelRef {
    path: string;
    revision: string | null;
    provider_kind: string;
    provider_config: JsonObject;
    /** Hash of provider_config without path-like keys; groups checkpoints with the same settings. */
    settings_hash: string;
}

export interface GitDetail {
    repo: string | null;
    commit: string | null;
    branch: string | null;
    dirty: boolean | null;
}

export interface BeakerDetail {
    experiment_id: string | null;
    workload_id: string | null;
    job_id: string | null;
    task_id: string | null;
    workspace: string | null;
    workspace_id: string | null;
    result_dataset_id: string | null;
    node_hostname: string | null;
    cluster: string | null;
    priority: string | null;
    image: string | null;
    budget: string | null;
    gpu_count: Count | null;
    cpu_count: number | null;
}

export interface EnvironmentDetail {
    hostname: string | null;
    platform: string | null;
    python_version: string | null;
    packages: { [distribution: string]: string };
    cuda_version: string | null;
    gpu_type: string | null;
    gpu_count: Count | null;
}

export interface SuiteChild {
    type: "task" | "suite";
    name: string;
}

export interface SuiteDef {
    name: string;
    aggregation: string;
    description: string | null;
    children: SuiteChild[];
    definition_hash: string;
}

export interface RunError {
    task: string | null;
    error: string;
}

export interface RunDetail extends RunSummary {
    model_detail: ModelDetail;
    git: GitDetail;
    beaker: BeakerDetail | null;
    environment: EnvironmentDetail;
    olmo_eval_version: string | null;
    argv: string[];
    /** shlex-joined argv, ready to copy. */
    reproduce_command: string | null;
    task_specs: string[];
    output_dir: string | null;
    notes: string | null;
    errors: RunError[];
    provider_init_seconds: { [worker: string]: number } | null;
    /** See TaskRuntime.startup_seconds. Null for runs that predate runtime recording. */
    startup_seconds: number | null;
    /** When every inference worker was ready. */
    processing_started_at: DateTime | null;
    /** Seconds from processing_started_at until the last task finished. */
    processing_seconds: number | null;
    /** Definitions of the suites stored with this run, outermost first. */
    suites_used: SuiteDef[];
    /** Other runs with the same launch_id. */
    siblings: RunSummary[];
    updated_at: DateTime;
}

// ---------------------------------------------------------------------------
// Task runtime
// ---------------------------------------------------------------------------

/**
 * How inference_seconds was obtained:
 * - measured: the run contained only this task, so its processing time is the task's time.
 * - attributed: from per-batch inference metrics split by the task's share of each batch.
 * - estimated: the run's processing time times the task's share of the run's tokens.
 * - not_recorded: the run predates runtime recording or lacks token counts.
 */
export type RuntimeBasis = "measured" | "attributed" | "estimated" | "not_recorded";

/** Selects which runtime a view shows. */
export type RuntimeMode = "inference" | "with_startup";

export interface TaskRuntime {
    /** Inference time for this task alone. */
    inference_seconds: number | null;
    basis: RuntimeBasis;
    /** The run's startup (model load, server start); excludes Beaker queueing. */
    startup_seconds: number | null;
    /** inference_seconds plus the run's full startup: about what running this task alone takes. */
    with_startup_seconds: number | null;
    /** The task's share of the run's prompt plus completion tokens, 0 to 1. */
    token_share: number | null;
    prompt_tokens_total: Count | null;
    completion_tokens_total: Count | null;
    /** inference_seconds per 1,000 instances. */
    seconds_per_1k_instances: number | null;
    /** Wall-clock span within the run, in seconds after processing started. */
    span_start_s: number | null;
    span_end_s: number | null;
}

// ---------------------------------------------------------------------------
// GET /api/runs/{run_id}/task-results?baseline=&alpha=
// ---------------------------------------------------------------------------

export interface BaselineCell {
    task_result_id: TaskResultId;
    run_id: RunId;
    task_hash: string;
    score: number | null;
    stderr: number | null;
    n: Count;
    metrics: FlatMetrics;
    delta: DeltaStats;
    /** Null for unbounded metrics or when instances do not pair. */
    contingency: Contingency | null;
}

export interface TaskResultRow {
    task_result_id: TaskResultId;
    run_id: RunId;
    task_name: string;
    task_hash: string;
    base_task: string | null;
    /** "metric:scorer" used as the score. */
    primary_metric: string | null;
    /** Keyed by "metric:scorer" for every key in metrics. */
    metric_meta: { [metricKey: string]: MetricMeta };
    score: number | null;
    stderr: number | null;
    /** The corpus score equals the mean of per-instance values (times instance_scale). */
    score_is_mean: boolean;
    instance_scale: number;
    n: Count;
    instances_processed: Count | null;
    instances_failed: Count | null;
    instances_stored: Count;
    metrics: FlatMetrics;
    error: string | null;
    error_summary: JsonObject | null;
    /** Seconds from the run's processing start until this task finished (shared workers). */
    duration_seconds: number | null;
    runtime: TaskRuntime;
    completion_tokens_total: Count | null;
    mean_completion_tokens: number | null;
    /** Share of instances with finish_reason "length". */
    truncation_rate: number | null;
    finish_reason_counts: { [reason: string]: Count };
    num_fewshot: Count | null;
    limit: Count | null;
    split: string | null;
    suites: string[];
    predictions_uri: string | null;
    requests_uri: string | null;
    baseline: BaselineCell | null;
}

export interface RunTaskResultsResponse {
    run_id: RunId;
    baseline: SubjectInfo | null;
    items: TaskResultRow[];
    computed_at: DateTime;
}

// ---------------------------------------------------------------------------
// GET /api/runs/{run_id}/suites?baseline=&alpha=
// ---------------------------------------------------------------------------

export interface SuiteBaselineCell {
    score: number | null;
    stderr: number | null;
    delta: DeltaStats;
}

export interface SuiteResultRow {
    name: string;
    parent: string | null;
    depth: Count;
    aggregation: string;
    description: string | null;
    children: SuiteChild[];
    score: number | null;
    stderr: number | null;
    num_tasks: Count;
    n_instances: Count;
    display_format: DisplayFormat;
    higher_is_better: boolean | null;
    /** Child tasks the run has no result for. */
    tasks_missing: string[];
    baseline: SuiteBaselineCell | null;
}

export interface RunSuitesResponse {
    run_id: RunId;
    baseline: SubjectInfo | null;
    /** Pre-order traversal (parents before children). */
    items: SuiteResultRow[];
    /** Task names in the run that belong to no stored suite. */
    unsuited_tasks: string[];
    computed_at: DateTime;
}

// ---------------------------------------------------------------------------
// GET /api/runs/{run_id}/task-results/{task_result_id}/instances
// ---------------------------------------------------------------------------

export interface InstanceRow {
    native_id: string;
    doc_id: Count | null;
    primary_score: number | null;
    /** Null for unbounded metrics. */
    correct: boolean | null;
    metrics: FlatMetrics;
    label: string | null;
    extracted_answer: string | null;
    finish_reason: string | null;
    completion_tokens: Count | null;
    prompt_tokens: Count | null;
    num_outputs: Count;
    prompt_preview: string | null;
    output_preview: string | null;
    judge_verdict: string | null;
    has_scoring_error: boolean;
    has_execution_result: boolean;
    has_judge_result: boolean;
    has_trajectory: boolean;
    baseline_score: number | null;
    baseline_correct: boolean | null;
    /** primary_score - baseline_score */
    delta: number | null;
}

export interface InstancesResponse {
    task_result_id: TaskResultId;
    baseline_task_result_id: TaskResultId | null;
    kind: MetricKind;
    threshold: number;
    items: InstanceRow[];
    next_cursor: string | null;
    total: Count;
}

// ---------------------------------------------------------------------------
// GET /api/instances/detail?task_result_id=&native_id=
// ---------------------------------------------------------------------------

export interface InstanceDetailResponse {
    task_result_id: TaskResultId;
    run_id: RunId;
    task_name: string;
    native_id: string;
    doc_id: Count | null;
    primary_score: number | null;
    metrics: FlatMetrics;
    /** Full prediction record (one predictions JSONL line), null when unavailable. */
    prediction: JsonObject | null;
    /** Full request record (one requests JSONL line), null when unavailable. */
    request: JsonObject | null;
    predictions_uri: string | null;
    requests_uri: string | null;
    /** Why prediction or request is null (missing offsets, record over 25 MiB, ...). */
    unavailable_reason: string | null;
}

// ---------------------------------------------------------------------------
// GET /api/runs/{run_id}/task-results/{task_result_id}/histograms?baseline=&bins=
// ---------------------------------------------------------------------------

export interface LengthHistograms {
    correct: Histogram;
    incorrect: Histogram;
    all: Histogram;
}

export interface HistogramsResponse {
    task_result_id: TaskResultId;
    metric: string | null;
    kind: MetricKind;
    score: Histogram | null;
    baseline_score: Histogram | null;
    /** Per-instance (this - baseline) over shared instances. */
    delta: Histogram | null;
    completion_tokens: LengthHistograms | null;
    finish_reasons: { [reason: string]: Count };
    /** sampling_params.max_tokens from the task config. */
    max_tokens: Count | null;
}

// ---------------------------------------------------------------------------
// GET /api/runs/{run_id}/inference?baseline=
// ---------------------------------------------------------------------------

export interface InferenceKpis {
    wall_clock_s: number | null;
    total_requests: Count | null;
    failed_requests: Count | null;
    prompt_tokens: Count | null;
    completion_tokens: Count | null;
    output_tokens_per_second: number | null;
    mean_latency_s: number | null;
    ttft_p50_s: number | null;
    ttft_p95_s: number | null;
    gpu_utilization_mean_pct: number | null;
    gpu_memory_max_mb: number | null;
    gpu_hours: number | null;
}

export interface InferenceTaskRow {
    task_name: string;
    task_result_id: TaskResultId;
    instances: Count;
    completion_tokens: Count | null;
    prompt_tokens: Count | null;
    mean_completion_tokens: number | null;
    truncation_rate: number | null;
    duration_seconds: number | null;
    completion_tokens_box: BoxStats | null;
}

export interface SeriesOut {
    name: string;
    unit: string | null;
    device: string | null;
    /** [seconds since run start, value] */
    points: [number, number][];
}

export interface InferenceBatchOut {
    seq: Count;
    t: number;
    task_name: string | null;
    total_requests: Count;
    failed_requests: Count;
    total_prompt_tokens: Count;
    total_completion_tokens: Count;
    wall_clock_time_s: number;
    output_tokens_per_second: number;
    mean_latency_s: number;
}

export interface InferenceResponse {
    run_id: RunId;
    available: boolean;
    kpis: InferenceKpis;
    baseline_kpis: InferenceKpis | null;
    per_task: InferenceTaskRow[];
    /** Distribution of batch mean latency (per task when batches carry task_name). */
    batch_latency: BoxStats | null;
    batch_latency_by_task: { task_name: string; box: BoxStats }[];
    request_latency: {
        end_to_end_s: BoxStats | null;
        ttft_s: BoxStats | null;
        tpot_s: BoxStats | null;
    } | null;
    /** Up to 2,000 batches, downsampled evenly when there are more. */
    batches: InferenceBatchOut[];
    series: SeriesOut[];
    gpu_devices: { device_id: Count; name: string; memory_total_mb: number | null }[];
    started_at: DateTime | null;
}

// ---------------------------------------------------------------------------
// GET /api/runs/{run_id}/configs
// ---------------------------------------------------------------------------

export interface TaskConfigEntry {
    task_result_id: TaskResultId;
    task_name: string;
    task_hash: string;
    config: JsonObject;
}

export interface RunConfigsResponse {
    run_id: RunId;
    model_config: JsonObject;
    harness_config: JsonObject | null;
    environment: EnvironmentDetail;
    argv: string[];
    task_configs: TaskConfigEntry[];
}

// ---------------------------------------------------------------------------
// GET /api/runs/{run_id}/artifacts  and  POST /api/artifacts/sign
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

export interface ArtifactRow {
    path: string;
    kind: ArtifactKind;
    task_name: string | null;
    size_bytes: Count;
    uploaded: boolean;
    updated_at: DateTime;
    gs_uri: string;
    console_url: string;
}

export interface ArtifactsResponse {
    run_id: RunId;
    gcs_prefix: string;
    gcs_console: string;
    items: ArtifactRow[];
    beaker: BeakerDetail | null;
    links: RunLinks;
}

export interface SignDownloadRequest {
    gs_uri: string;
}

export interface SignDownloadResponse {
    url: string;
    expires_at: DateTime;
}

// ---------------------------------------------------------------------------
// GET /api/runs/{run_id}/baseline-suggestions
// ---------------------------------------------------------------------------

export interface BaselineSuggestion {
    subject: SubjectKey;
    reason: "previous_checkpoint" | "same_group" | "previous_run_same_tasks";
    label: string;
    model: ModelRef;
    run: RunSummary | null;
}

export interface BaselineSuggestionsResponse {
    items: BaselineSuggestion[];
}

// ---------------------------------------------------------------------------
// POST /api/subjects/resolve  and  GET /api/subjects/task-result
// ---------------------------------------------------------------------------

export interface ResolveSubjectsRequest {
    /** @maxItems 50 */
    subjects: SubjectKey[];
    /** Restrict model-subject merging to runs in this group. */
    group?: string | null;
}

export interface ResolveSubjectsResponse {
    items: SubjectInfo[];
    missing: SubjectKey[];
}

export interface SubjectTaskResultResponse {
    subject: SubjectKey;
    task_name: string;
    task_result_id: TaskResultId | null;
    run_id: RunId | null;
    task_hash: string | null;
}

// ---------------------------------------------------------------------------
// Compare: shared request fields
// ---------------------------------------------------------------------------

/**
 * scope: "all" (union of tasks), "shared" (tasks every subject has),
 * "suite:<suite_name>", or "task:<task_name>".
 * metric: "primary" or a "metric:scorer" key.
 */
export interface CompareRequestBase {
    /**
     * @minItems 1
     * @maxItems 50
     */
    subjects: SubjectKey[];
    group?: string | null;
    scope: string;
    /**
     * "primary" (each task's primary metric), "metric:scorer", or a runtime metric:
     * "runtime:inference" or "runtime:with_startup" (seconds, lower is better; no
     * instance-level pairing, so pairwise significance is not available for them).
     */
    metric: string;
    alpha?: number;
    n_boot?: Count;
    seed?: Integer;
    shared_only?: boolean;
}

// POST /api/compare/matrix
export interface MatrixRequest extends CompareRequestBase {
    baseline?: SubjectKey | null;
}

export type CellStatus = "ok" | "missing" | "failed" | "partial";

export interface MatrixCell {
    status: CellStatus;
    score: number | null;
    stderr: number | null;
    n: Count | null;
    task_result_id: TaskResultId | null;
    run_id: RunId | null;
    task_hash: string | null;
    /** Vs baseline; null when no baseline or this is the baseline column. */
    delta: DeltaStats | null;
    /** Suite rows: children with no result. */
    children_missing: Count;
}

export interface MatrixRow {
    /** "task:<name>" or "suite:<name>" */
    key: string;
    kind: "task" | "suite";
    name: string;
    depth: Count;
    parent: string | null;
    aggregation: string | null;
    /** Resolved "metric:scorer" for task rows. */
    metric_key: string | null;
    meta: MetricMeta | null;
    /** Aligned with MatrixResponse.subjects. */
    cells: MatrixCell[];
}

export interface Coverage {
    n_subjects: Count;
    n_tasks: Count;
    complete: Count;
    missing: Count;
    failed: Count;
    hash_mismatch: Count;
}

export interface MatrixResponse {
    subjects: SubjectInfo[];
    baseline: SubjectKey | null;
    scope: string;
    metric: string;
    /** Display order: suites pre-order with their tasks; rows for missing tasks are kept. */
    rows: MatrixRow[];
    coverage: Coverage;
    /** Minimum detectable effect at 80% power, raw metric units. */
    mde80: number | null;
    alpha: number;
    computed_at: DateTime;
}

// POST /api/compare/pairwise
export interface PairwiseRequest extends CompareRequestBase {
    /** Instance-level tie margin in raw metric units. */
    margin?: number;
}

export interface PairStats {
    row: SubjectKey;
    col: SubjectKey;
    n_shared: Count;
    n_contested: Count;
    wins: Count;
    losses: Count;
    ties: Count;
    /** wins / (wins + losses); 0.5 when nothing is contested. */
    win_rate: number;
    /** row minus col, raw units. */
    delta: number | null;
    ci_low: number | null;
    ci_high: number | null;
    p_bootstrap: number | null;
    p_sign: number | null;
    /** Share of bootstrap resamples where row is better. */
    prob_row_better: number | null;
    significant: boolean;
    method: DeltaMethod;
}

export interface PairwiseRowSummary {
    subject: SubjectKey;
    mean_win_rate: number;
    ci_low: number | null;
    ci_high: number | null;
    mean_delta: number | null;
}

export interface PairwiseResponse {
    /** Ordered by mean win rate, best first. */
    subjects: SubjectInfo[];
    /** Every ordered pair (row != col). */
    pairs: PairStats[];
    rows: PairwiseRowSummary[];
    tasks_used: string[];
    mde80: number | null;
    alpha: number;
    margin: number;
    higher_is_better: boolean | null;
    display_format: DisplayFormat;
    warnings: string[];
    computed_at: DateTime;
}

// POST /api/compare/contingency
export interface ContingencyRequest {
    a: SubjectKey;
    b: SubjectKey;
    group?: string | null;
    scope: string;
    metric: string;
    /** Correctness threshold for bounded metrics (default 0.5). */
    threshold?: number;
}

export interface ContingencyTaskRow {
    task_name: string;
    kind: MetricKind;
    task_hash_a: string | null;
    task_hash_b: string | null;
    task_result_id_a: TaskResultId | null;
    task_result_id_b: TaskResultId | null;
    contingency: Contingency | null;
    delta: number | null;
}

export interface ContingencyResponse {
    a: SubjectKey;
    b: SubjectKey;
    /** Sorted by net, most negative first. */
    items: ContingencyTaskRow[];
    totals: Contingency;
}

// POST /api/compare/instances
export interface CompareInstancesRequest {
    /**
     * @minItems 1
     * @maxItems 30
     */
    subjects: SubjectKey[];
    group?: string | null;
    task_name: string;
    metric: string;
    threshold?: number;
    filter: "all" | "disagree" | "all_wrong" | "baseline_wrong" | "cell";
    /** Required when filter is "cell"; uses a and b. */
    cell?: "both_right" | "only_a" | "only_b" | "both_wrong" | null;
    a?: SubjectKey | null;
    b?: SubjectKey | null;
    baseline?: SubjectKey | null;
    sort?: "disagreement" | "native_id" | "n_correct";
    include_previews?: boolean;
    cursor?: string | null;
    /** Max 1,000 with previews, 20,000 without. */
    limit?: Count;
}

export interface CompareInstanceRow {
    native_id: string;
    /** Aligned with subjects; null when the subject lacks the instance. */
    scores: (number | null)[];
    correct: (boolean | null)[];
    n_correct: Count;
    n_present: Count;
    prompt_preview: string | null;
    output_previews: (string | null)[] | null;
}

export interface CompareInstancesResponse {
    subjects: { key: SubjectKey; task_result_id: TaskResultId | null; run_id: RunId | null }[];
    kind: MetricKind;
    threshold: number;
    items: CompareInstanceRow[];
    /** Index i = number of instances (after filtering) with i subjects correct. */
    counts_by_n_correct: Count[];
    total: Count;
    next_cursor: string | null;
}

// ---------------------------------------------------------------------------
// POST /api/tasks/distributions
// ---------------------------------------------------------------------------

export interface DistributionKey {
    task_name: string;
    /** Null pools every variant of the task. */
    task_hash: string | null;
    metric: string;
}

export interface DistributionsRequest {
    /** @maxItems 200 */
    items: DistributionKey[];
    per_model?: "latest" | "all";
}

export interface Distribution {
    task_name: string;
    task_hash: string | null;
    metric: string;
    n: Count;
    /** Sorted ascending; all scores when n <= 500, else 500 evenly spaced quantiles. */
    scores: number[];
    is_quantiles: boolean;
    higher_is_better: boolean | null;
    top: { run_id: RunId; model: ModelRef; score: number } | null;
}

export interface DistributionsResponse {
    items: Distribution[];
}

// ---------------------------------------------------------------------------
// Models: GET /api/models, GET /api/models/series, GET /api/models/progression
// ---------------------------------------------------------------------------

export interface VariantCount {
    settings_hash: string;
    n_runs: Count;
}

export interface ModelSeriesRow {
    series: string;
    series_label: string;
    family: string | null;
    n_checkpoints: Count;
    n_runs: Count;
    latest_step: Count | null;
    last_run_at: DateTime | null;
    latest_run: RunSummary | null;
    variants: VariantCount[];
}

export interface ModelsListResponse {
    items: ModelSeriesRow[];
    next_cursor: string | null;
    total: Count;
}

export interface CheckpointRow {
    model: ModelRef;
    settings_hash: string;
    run_ids: RunId[];
    latest_run_at: DateTime | null;
}

export interface VariantInfo {
    settings_hash: string;
    n_runs: Count;
    /** Provider-config keys whose values differ from the most common variant. */
    differs: JsonObject;
}

export interface ModelSeriesDetailResponse {
    series: string;
    series_label: string;
    family: string | null;
    n_runs: Count;
    /** Sorted by step ascending (null steps last). */
    checkpoints: CheckpointRow[];
    variants: VariantInfo[];
}

export interface ProgressionPoint {
    model_id: ModelId;
    run_id: RunId;
    task_result_id: TaskResultId | null;
    step: Count | null;
    tokens_seen: Count | null;
    date: DateTime;
    score: number | null;
    stderr: number | null;
    n: Count | null;
}

export interface ProgressionReference {
    subject: SubjectKey;
    label: string;
    score: number | null;
    stderr: number | null;
}

export interface ProgressionPanel {
    /** "task:<name>" or "suite:<name>" */
    key: string;
    kind: "task" | "suite";
    name: string;
    meta: MetricMeta | null;
    points: ProgressionPoint[];
    references: ProgressionReference[];
}

export interface ProgressionResponse {
    series: string;
    settings_hash: string | null;
    metric: string;
    panels: ProgressionPanel[];
}

// ---------------------------------------------------------------------------
// Tasks and suites
// ---------------------------------------------------------------------------

export interface TaskRow {
    task_name: string;
    base_task: string | null;
    n_variants: Count;
    n_runs: Count;
    n_models: Count;
    suites: string[];
    primary_metric: string | null;
    last_run_at: DateTime | null;
}

export interface TasksListResponse {
    items: TaskRow[];
    next_cursor: string | null;
    total: Count;
}

export interface TaskVariantInfo {
    task_hash: string;
    n_runs: Count;
    primary_metric: string | null;
    metric_meta: { [metricKey: string]: MetricMeta };
    num_fewshot: Count | null;
    limit: Count | null;
    split: string | null;
    /** Median stored instances across runs. */
    n_instances: Count | null;
    first_seen_at: DateTime;
    last_seen_at: DateTime;
    config: JsonObject;
}

export interface TaskDetailResponse {
    task_name: string;
    base_task: string | null;
    suites: string[];
    /** Most runs first. */
    variants: TaskVariantInfo[];
}

export interface LeaderboardRow {
    rank: Count;
    subject: SubjectKey;
    model: ModelRef;
    run_id: RunId;
    task_result_id: TaskResultId | null;
    task_hash: string | null;
    score: number | null;
    stderr: number | null;
    ci_low: number | null;
    ci_high: number | null;
    n: Count | null;
    date: DateTime;
    experiment_group: string | null;
    author: string | null;
    /** 95% CI overlaps the leader's. */
    tied_with_leader: boolean;
    /** Suite leaderboards only: child score by child key ("task:<name>" or "suite:<name>"). */
    child_scores: { [childKey: string]: number | null } | null;
    /** Task leaderboards only; null for suites. */
    runtime: TaskRuntime | null;
    gpu_type: string | null;
    gpu_count: Count | null;
}

export interface LeaderboardResponse {
    kind: "task" | "suite";
    name: string;
    task_hash: string | null;
    metric: string;
    meta: MetricMeta | null;
    /** Suite leaderboards only. */
    children: SuiteChild[] | null;
    items: LeaderboardRow[];
    next_cursor: string | null;
    total: Count;
}

export interface SuiteRow {
    suite_name: string;
    aggregation: string;
    description: string | null;
    n_children: Count;
    n_tasks: Count;
    n_runs: Count;
    last_run_at: DateTime | null;
}

export interface SuitesListResponse {
    items: SuiteRow[];
    next_cursor: string | null;
    total: Count;
}

export interface SuiteTreeNode {
    type: "task" | "suite";
    name: string;
    aggregation: string | null;
    children: SuiteTreeNode[];
}

export interface SuiteDetailResponse {
    suite_name: string;
    aggregation: string;
    description: string | null;
    definition_hash: string;
    /** Number of distinct definitions seen for this name. */
    definitions_seen: Count;
    tree: SuiteTreeNode;
    n_runs: Count;
}

// ---------------------------------------------------------------------------
// Groups
// ---------------------------------------------------------------------------

export interface MiniHeatmap {
    rows: { key: SubjectKey; label: string }[];
    columns: { key: string; label: string }[];
    /** rows x columns */
    values: (number | null)[][];
}

export interface GroupRow {
    name: string;
    n_runs: Count;
    n_models: Count;
    n_tasks: Count;
    users: string[];
    first_run_at: DateTime;
    last_run_at: DateTime;
    heatmap: MiniHeatmap | null;
}

export interface GroupsListResponse {
    items: GroupRow[];
    next_cursor: string | null;
    total: Count;
}

export interface CoverageCell {
    subject: SubjectKey;
    task_name: string;
    status: "ok" | "failed" | "missing";
    task_result_id: TaskResultId | null;
    run_id: RunId | null;
}

export interface GroupDetailResponse {
    name: string;
    n_runs: Count;
    n_models: Count;
    n_tasks: Count;
    users: string[];
    first_run_at: DateTime;
    last_run_at: DateTime;
    /** One model subject per model in the group. */
    subjects: SubjectInfo[];
    tasks: string[];
    coverage: CoverageCell[];
}

// ---------------------------------------------------------------------------
// Home: GET /api/activity, GET /api/stats/summary
// ---------------------------------------------------------------------------

export interface ActivityItem {
    /** launch_id, or "run:<run_id>" for runs without a launch. */
    key: string;
    launch_id: string | null;
    user: string | null;
    created_at: DateTime;
    runs: RunSummary[];
    n_tasks: Count;
    n_failed_tasks: Count;
    /** Worst status among the runs (failed > partial > running > complete). */
    status: RunStatus;
    /** First task error message when any task failed. */
    failure_reason: string | null;
}

export interface ActivityResponse {
    items: ActivityItem[];
    next_cursor: string | null;
}

export interface DailyStat {
    date: DateString;
    runs: Count;
    models: Count;
    instances: Count;
}

export interface StatsSummaryResponse {
    window_days: Count;
    runs: Count;
    models: Count;
    instances: Count;
    daily: DailyStat[];
}

// ---------------------------------------------------------------------------
// Saved views: GET/POST /api/views, PATCH/DELETE /api/views/{view_id}
// ---------------------------------------------------------------------------

export interface SavedView {
    id: string;
    name: string;
    owner_email: string;
    shared: boolean;
    /** UI page the query belongs to, e.g. "runs" or "compare". */
    page: string;
    /** URL query string without the leading "?". */
    query: string;
    created_at: DateTime;
    updated_at: DateTime;
}

export interface SavedViewsResponse {
    mine: SavedView[];
    shared: SavedView[];
}

export interface CreateViewRequest {
    /**
     * @minLength 1
     * @maxLength 120
     */
    name: string;
    page: string;
    /** @maxLength 4000 */
    query: string;
    shared: boolean;
}

export interface UpdateViewRequest {
    name?: string;
    query?: string;
    shared?: boolean;
}

// ---------------------------------------------------------------------------
// GET /api/tasks/{task_name}/runtime?hash=&gpu_type=&family=&group=&user=
// Runtime of every finalized run of a task variant (default: the most-run variant).
// ---------------------------------------------------------------------------

export interface RuntimeStats {
    n: Count;
    median: number | null;
    p90: number | null;
    min: number | null;
    max: number | null;
}

export interface TaskRuntimeModelRow {
    model: ModelRef;
    gpu_type: string | null;
    gpu_count: Count | null;
    runs: Count;
    inference: RuntimeStats;
    with_startup: RuntimeStats;
    seconds_per_1k_instances: RuntimeStats;
    /** Score of the model's latest run in this group. */
    latest_score: number | null;
}

export interface TaskRuntimePoint {
    run_id: RunId;
    task_result_id: TaskResultId;
    model: ModelRef;
    date: DateTime;
    gpu_type: string | null;
    gpu_count: Count | null;
    score: number | null;
    n: Count | null;
    runtime: TaskRuntime;
}

export interface TaskRuntimeResponse {
    task_name: string;
    task_hash: string | null;
    meta: MetricMeta | null;
    /** GPU types present, most runs first (for the filter). */
    gpu_types: string[];
    /** One row per (model, GPU type, GPU count). */
    rows: TaskRuntimeModelRow[];
    /** One point per finalized run with any runtime data, newest first, at most 2,000. */
    points: TaskRuntimePoint[];
    /** Runs of this variant whose runtime is not recorded. */
    not_recorded: Count;
}
