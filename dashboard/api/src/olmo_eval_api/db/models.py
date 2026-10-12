"""SQLModel table definitions.

Postgres holds everything the dashboard queries. Full prediction and request records stay in
GCS and are read by byte range (``instance_results.pred_offset`` / ``pred_length``).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    Double,
    ForeignKey,
    ForeignKeyConstraint,
    Identity,
    Index,
    Integer,
    LargeBinary,
    SmallInteger,
    String,
    Table,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlmodel import Field, SQLModel

metadata = SQLModel.metadata


def _ts(nullable: bool = False, default_now: bool = False) -> Any:
    return Field(
        default=None,
        sa_column=Column(
            DateTime(timezone=True),
            nullable=nullable,
            server_default=func.now() if default_now else None,
        ),
    )


def _jsonb(nullable: bool = False, default: str | None = None) -> Any:
    return Field(
        default=None,
        sa_column=Column(
            JSONB,
            nullable=nullable,
            server_default=text(f"'{default}'::jsonb") if default is not None else None,
        ),
    )


def _text(nullable: bool = True) -> Any:
    return Field(default=None, sa_column=Column(Text, nullable=nullable))


def _str(length: int, nullable: bool = True) -> Any:
    return Field(default=None, sa_column=Column(String(length), nullable=nullable))


def _int(type_: Any = Integer, nullable: bool = True, default: int | None = None) -> Any:
    return Field(
        default=default,
        sa_column=Column(
            type_,
            nullable=nullable,
            server_default=text(str(default)) if default is not None else None,
        ),
    )


def _float(nullable: bool = True) -> Any:
    return Field(default=None, sa_column=Column(Double, nullable=nullable))


def _bool(nullable: bool = False, default: bool | None = None) -> Any:
    return Field(
        default=default,
        sa_column=Column(
            Boolean,
            nullable=nullable,
            server_default=text(str(default).lower()) if default is not None else None,
        ),
    )


def _text_array() -> Any:
    return Field(
        default_factory=list,
        sa_column=Column(ARRAY(Text), nullable=False, server_default=text("'{}'::text[]")),
    )


def _trgm(name: str, column: str) -> Index:
    return Index(name, column, postgresql_using="gin", postgresql_ops={column: "gin_trgm_ops"})


class Model(SQLModel, table=True):
    __tablename__ = "models"
    __table_args__ = (
        UniqueConstraint("name", "model_hash", name="uq_models_name_hash"),
        Index("ix_models_series", "series"),
        Index("ix_models_family", "family"),
        _trgm("ix_models_name_trgm", "name"),
        _trgm("ix_models_series_trgm", "series"),
    )

    model_id: str = Field(sa_column=Column(String(12), primary_key=True))
    name: str = _text(nullable=False)
    model_hash: str = _str(64, nullable=False)
    path: str = _text(nullable=False)
    revision: str | None = _text()
    provider_kind: str = _text(nullable=False)
    provider_config: dict[str, Any] = _jsonb()
    settings_hash: str = _str(12, nullable=False)
    series: str = _text(nullable=False)
    series_label: str = _text(nullable=False)
    family: str | None = _text()
    step: int | None = _int(BigInteger)
    tokens_seen: int | None = _int(BigInteger)
    first_seen_at: datetime = _ts()
    last_seen_at: datetime = _ts()


class Run(SQLModel, table=True):
    __tablename__ = "runs"
    __table_args__ = (
        CheckConstraint(
            "status IN ('running', 'complete', 'partial', 'failed')", name="ck_runs_status"
        ),
        CheckConstraint("upload_state IN ('uploading', 'complete')", name="ck_runs_upload_state"),
        Index("ix_runs_launch_id", "launch_id"),
        Index("ix_runs_model_id", "model_id"),
        Index("ix_runs_experiment_group", "experiment_group"),
        Index("ix_runs_author", "author"),
        Index("ix_runs_uploaded_by", "uploaded_by"),
        Index("ix_runs_tags", "tags", postgresql_using="gin"),
        Index("ix_runs_created", text("created_at DESC"), text("run_id DESC")),
        Index(
            "ix_runs_git_commit", "git_commit", postgresql_ops={"git_commit": "text_pattern_ops"}
        ),
        Index("ix_runs_beaker_experiment_id", "beaker_experiment_id"),
        Index("ix_runs_beaker_job_id", "beaker_job_id"),
        Index("ix_runs_beaker_result_dataset_id", "beaker_result_dataset_id"),
        Index("ix_runs_beaker_workspace", "beaker_workspace"),
        _trgm("ix_runs_search_text_trgm", "search_text"),
    )

    run_id: str = Field(sa_column=Column(String(32), primary_key=True))
    launch_id: str | None = _str(64)
    model_id: str = Field(
        sa_column=Column(String(12), ForeignKey("models.model_id"), nullable=False)
    )
    model_name: str = _text(nullable=False)
    experiment_name: str | None = _text()
    experiment_group: str | None = _text()
    status: str = _str(16, nullable=False)
    upload_state: str = _str(16, nullable=False)
    author: str | None = _text()
    uploaded_by: str = _text(nullable=False)
    # SHA-256 hex of the secret the uploading client keeps in the run's results directory.
    write_secret_hash: str | None = _str(64)
    tags: list[str] = _text_array()
    # The tags the uploading client last sent, so a re-upload does not restore tags that were
    # removed in the dashboard.
    client_tags: list[str] = _text_array()
    notes: str | None = _text()
    created_at: datetime = _ts(default_now=True)
    updated_at: datetime = _ts()
    started_at: datetime | None = _ts(nullable=True)
    finished_at: datetime | None = _ts(nullable=True)
    duration_seconds: float | None = _float()
    olmo_eval_version: str | None = _text()
    git_repo: str | None = _text()
    git_commit: str | None = _text()
    git_branch: str | None = _text()
    git_dirty: bool | None = _bool(nullable=True)
    beaker_experiment_id: str | None = _text()
    beaker_job_id: str | None = _text()
    beaker_result_dataset_id: str | None = _text()
    beaker_workspace: str | None = _text()
    beaker: dict[str, Any] | None = _jsonb(nullable=True)
    environment: dict[str, Any] = _jsonb()
    argv: list[str] = _jsonb()
    task_specs: list[str] = _jsonb()
    output_dir: str | None = _text()
    harness_config: dict[str, Any] | None = _jsonb(nullable=True)
    provider_init_seconds: dict[str, float] | None = _jsonb(nullable=True)
    # Seconds from olmo-eval starting until every inference worker was ready.
    startup_seconds: float | None = _float()
    # When every inference worker was ready; task spans are measured from here.
    processing_started_at: datetime | None = _ts(nullable=True)
    # Seconds from processing_started_at until the last task finished.
    processing_seconds: float | None = _float()
    errors: list[dict[str, Any]] = _jsonb(default="[]")
    client: dict[str, Any] | None = _jsonb(nullable=True)
    gcs_prefix: str = _text(nullable=False)
    num_tasks: int = _int(nullable=False, default=0)
    num_failed_tasks: int = _int(nullable=False, default=0)
    num_instances: int = _int(BigInteger, nullable=False, default=0)
    headline: dict[str, Any] | None = _jsonb(nullable=True)
    search_text: str = _text(nullable=False)


class TaskVariant(SQLModel, table=True):
    __tablename__ = "task_variants"
    __table_args__ = (_trgm("ix_task_variants_task_name_trgm", "task_name"),)

    task_name: str = Field(sa_column=Column(Text, primary_key=True))
    task_hash: str = Field(sa_column=Column(String(64), primary_key=True))
    base_task: str | None = _text()
    config: dict[str, Any] = _jsonb()
    primary_metric: str | None = _text()
    metric_meta: dict[str, Any] = _jsonb(default="{}")
    num_fewshot: int | None = _int()
    task_limit: int | None = _int()
    split: str | None = _text()
    suites: list[str] = _text_array()
    first_seen_at: datetime = _ts()
    last_seen_at: datetime = _ts()


class TaskResult(SQLModel, table=True):
    __tablename__ = "task_results"
    __table_args__ = (
        UniqueConstraint("run_id", "task_name", "task_hash", name="uq_task_results_run_task"),
        ForeignKeyConstraint(
            ["task_name", "task_hash"],
            ["task_variants.task_name", "task_variants.task_hash"],
            name="fk_task_results_variant",
        ),
        Index(
            "ix_task_results_variant_model",
            "task_name",
            "task_hash",
            "model_id",
            text("run_created_at DESC"),
        ),
        Index("ix_task_results_model_task", "model_id", "task_name", text("run_created_at DESC")),
        Index("ix_task_results_task_created", "task_name", text("run_created_at DESC")),
    )

    id: int | None = Field(
        default=None, sa_column=Column(BigInteger, Identity(always=False), primary_key=True)
    )
    run_id: str = Field(
        sa_column=Column(String(32), ForeignKey("runs.run_id", ondelete="CASCADE"), nullable=False)
    )
    task_name: str = _text(nullable=False)
    task_hash: str = _str(64, nullable=False)
    model_id: str = _str(12, nullable=False)
    run_created_at: datetime = _ts()
    primary_metric: str | None = _text()
    score: float | None = _float()
    stderr: float | None = _float()
    score_is_mean: bool = _bool(default=False)
    instance_scale: float = Field(
        default=1.0, sa_column=Column(Double, nullable=False, server_default=text("1"))
    )
    metric_kind: str | None = _str(16)
    higher_is_better: bool | None = _bool(nullable=True)
    display_format: str = Field(
        default="raw", sa_column=Column(String(8), nullable=False, server_default="raw")
    )
    metrics: dict[str, float | None] = _jsonb()
    metric_meta: dict[str, Any] = _jsonb(default="{}")
    num_instances: int = _int(nullable=False)
    instances_processed: int | None = _int()
    instances_failed: int | None = _int()
    instance_count_expected: int = _int(nullable=False)
    instances_stored: int = _int(nullable=False, default=0)
    error: str | None = _text()
    error_summary: dict[str, Any] | None = _jsonb(nullable=True)
    # Seconds from the run's processing start until this task finished (shared workers, so not
    # the task's own cost; see services/runtime.py).
    duration_seconds: float | None = _float()
    first_request_at: datetime | None = _ts(nullable=True)
    last_completed_at: datetime | None = _ts(nullable=True)
    attributed_inference_seconds: float | None = _float()
    # Task-level token totals as the client reported them (every request, including failed
    # instances); null for older clients.
    reported_prompt_tokens_total: int | None = _int(BigInteger)
    reported_completion_tokens_total: int | None = _int(BigInteger)
    # Set when the run completes: the reported total when present, else the sum over stored
    # instances.
    completion_tokens_total: int | None = _int(BigInteger)
    prompt_tokens_total: int | None = _int(BigInteger)
    mean_completion_tokens: float | None = _float()
    truncation_rate: float | None = _float()
    finish_reason_counts: dict[str, int] = _jsonb(default="{}")
    predictions_path: str | None = _text()
    requests_path: str | None = _text()
    created_at: datetime = _ts()
    updated_at: datetime = _ts()
    finalized_at: datetime | None = _ts(nullable=True)
    # Hash of the last upload's request and file checksums; an identical re-upload is skipped.
    content_hash: str | None = _str(64)


class InstanceResult(SQLModel, table=True):
    __tablename__ = "instance_results"
    __table_args__ = (
        Index("ix_instance_results_key_hash", "task_result_id", "key_hash"),
        Index("ix_instance_results_doc_id", "task_result_id", "doc_id"),
        Index("ix_instance_results_score", "task_result_id", "primary_score"),
    )

    task_result_id: int = Field(
        sa_column=Column(
            BigInteger,
            ForeignKey("task_results.id", ondelete="CASCADE"),
            primary_key=True,
        )
    )
    native_id: str = Field(sa_column=Column(Text, primary_key=True))
    key_hash: int = _int(BigInteger, nullable=False)
    doc_id: int | None = _int()
    primary_score: float | None = _float()
    metrics: dict[str, float | None] = _jsonb(default="{}")
    label: str | None = _text()
    extracted_answer: str | None = _text()
    finish_reason: str | None = _str(32)
    completion_tokens: int | None = _int()
    prompt_tokens: int | None = _int()
    num_outputs: int = _int(SmallInteger, nullable=False)
    prompt_preview: str | None = _text()
    output_preview: str | None = _text()
    judge_verdict: str | None = _str(64)
    has_scoring_error: bool = _bool()
    has_execution_result: bool = _bool()
    has_judge_result: bool = _bool()
    has_trajectory: bool = _bool()
    pred_offset: int | None = _int(BigInteger)
    req_offset: int | None = _int(BigInteger)
    pred_length: int | None = _int()
    req_length: int | None = _int()


class TaskResultVector(SQLModel, table=True):
    __tablename__ = "task_result_vectors"

    task_result_id: int = Field(
        sa_column=Column(
            BigInteger,
            ForeignKey("task_results.id", ondelete="CASCADE"),
            primary_key=True,
        )
    )
    metric_key: str = _text(nullable=False)
    n: int = _int(nullable=False)
    key_hashes: bytes = Field(sa_column=Column(LargeBinary, nullable=False))
    scores: bytes = Field(sa_column=Column(LargeBinary, nullable=False))
    built_at: datetime = _ts()


class SuiteResult(SQLModel, table=True):
    __tablename__ = "suite_results"
    __table_args__ = (
        UniqueConstraint("run_id", "suite_name", name="uq_suite_results_run_suite"),
        Index("ix_suite_results_suite_run", "suite_name", "run_id"),
    )

    id: int | None = Field(
        default=None, sa_column=Column(BigInteger, Identity(always=False), primary_key=True)
    )
    run_id: str = Field(
        sa_column=Column(String(32), ForeignKey("runs.run_id", ondelete="CASCADE"), nullable=False)
    )
    suite_name: str = _text(nullable=False)
    parent_suite: str | None = _text()
    aggregation: str = _str(32, nullable=False)
    description: str | None = _text()
    children: list[dict[str, str]] = _jsonb()
    definition_hash: str = _str(16, nullable=False)
    score: float | None = _float()
    stderr: float | None = _float()
    num_tasks: int | None = _int()
    n_instances: int | None = _int(BigInteger)
    metrics: dict[str, float | None] = _jsonb()
    primary_metric: str | None = _text()
    display_format: str = _str(8, nullable=False)
    higher_is_better: bool | None = _bool(nullable=True)
    tasks_missing: list[str] = _jsonb(default="[]")


class SuiteDefRow(SQLModel, table=True):
    __tablename__ = "suite_defs"

    suite_name: str = Field(sa_column=Column(Text, primary_key=True))
    definition_hash: str = Field(sa_column=Column(String(16), primary_key=True))
    aggregation: str = _str(32, nullable=False)
    description: str | None = _text()
    children: list[dict[str, str]] = _jsonb()
    first_seen_at: datetime = _ts()
    last_seen_at: datetime = _ts()


class Artifact(SQLModel, table=True):
    __tablename__ = "artifacts"

    run_id: str = Field(
        sa_column=Column(
            String(32), ForeignKey("runs.run_id", ondelete="CASCADE"), primary_key=True
        )
    )
    path: str = Field(sa_column=Column(Text, primary_key=True))
    kind: str = _str(32, nullable=False)
    task_name: str | None = _text()
    size_bytes: int = _int(BigInteger, nullable=False)
    md5_b64: str = _str(24, nullable=False)
    content_type: str = _text(nullable=False)
    uploaded: bool = _bool(default=False)
    created_at: datetime = _ts()
    updated_at: datetime = _ts()


class InferenceBatch(SQLModel, table=True):
    __tablename__ = "inference_batches"

    run_id: str = Field(
        sa_column=Column(
            String(32), ForeignKey("runs.run_id", ondelete="CASCADE"), primary_key=True
        )
    )
    seq: int = Field(sa_column=Column(Integer, primary_key=True))
    ts: datetime = _ts()
    task_name: str | None = _text()
    total_requests: int = _int(nullable=False)
    successful_requests: int = _int(nullable=False)
    failed_requests: int = _int(nullable=False)
    total_prompt_tokens: int = _int(BigInteger, nullable=False)
    total_completion_tokens: int = _int(BigInteger, nullable=False)
    wall_clock_time_s: float = _float(nullable=False)
    output_tokens_per_second: float = _float(nullable=False)
    mean_latency_s: float = _float(nullable=False)
    gpu_summary: dict[str, Any] | None = _jsonb(nullable=True)


class RunInference(SQLModel, table=True):
    __tablename__ = "run_inference"

    run_id: str = Field(
        sa_column=Column(
            String(32), ForeignKey("runs.run_id", ondelete="CASCADE"), primary_key=True
        )
    )
    source_paths: list[str] = _jsonb()
    series: list[dict[str, Any]] = _jsonb()
    request_latency: dict[str, Any] | None = _jsonb(nullable=True)
    gpu_devices: list[dict[str, Any]] = _jsonb()
    updated_at: datetime = _ts()


class SavedViewRow(SQLModel, table=True):
    __tablename__ = "saved_views"
    __table_args__ = (
        Index("ix_saved_views_owner", "owner_email"),
        Index("ix_saved_views_shared", "shared", postgresql_where=text("shared")),
    )

    id: str = Field(sa_column=Column(String(32), primary_key=True))
    name: str = _text(nullable=False)
    owner_email: str = _text(nullable=False)
    shared: bool = _bool(default=False)
    page: str = _str(32, nullable=False)
    query: str = _text(nullable=False)
    created_at: datetime = _ts()
    updated_at: datetime = _ts()


class StatsCacheRow(SQLModel, table=True):
    __tablename__ = "stats_cache"

    key: str = Field(sa_column=Column(String(64), primary_key=True))
    value: dict[str, Any] = _jsonb()
    created_at: datetime = _ts(default_now=True)


def _table(model: type[SQLModel]) -> Table:
    return SQLModel.metadata.tables[str(model.__tablename__)]


# Typed Table objects for building SQL expressions (``runs_t.c.run_id == x``). Type checkers see
# SQLModel attributes as plain Python values, so expressions use these instead.
models_t = _table(Model)
runs_t = _table(Run)
task_variants_t = _table(TaskVariant)
task_results_t = _table(TaskResult)
instance_results_t = _table(InstanceResult)
task_result_vectors_t = _table(TaskResultVector)
suite_results_t = _table(SuiteResult)
suite_defs_t = _table(SuiteDefRow)
artifacts_t = _table(Artifact)
inference_batches_t = _table(InferenceBatch)
run_inference_t = _table(RunInference)
saved_views_t = _table(SavedViewRow)
stats_cache_t = _table(StatsCacheRow)

ALL_TABLES = [
    "stats_cache",
    "saved_views",
    "run_inference",
    "inference_batches",
    "artifacts",
    "suite_defs",
    "suite_results",
    "task_result_vectors",
    "instance_results",
    "task_results",
    "task_variants",
    "runs",
    "models",
]
