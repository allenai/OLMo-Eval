"""Initial schema.

Revision ID: 0001
Revises:
Create Date: 2026-10-01
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
    # On Cloud SQL, let members of cloudsqlsuperuser (IAM users used for debugging) read the
    # tables the API's IAM user creates. Plain local Postgres has no such role.
    op.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'cloudsqlsuperuser') THEN
                ALTER DEFAULT PRIVILEGES IN SCHEMA public
                    GRANT ALL ON TABLES TO cloudsqlsuperuser;
                ALTER DEFAULT PRIVILEGES IN SCHEMA public
                    GRANT ALL ON SEQUENCES TO cloudsqlsuperuser;
            END IF;
        END
        $$;
        """
    )
    op.create_table(
        "models",
        sa.Column("model_id", sa.String(length=12), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("model_hash", sa.String(length=64), nullable=False),
        sa.Column("path", sa.Text(), nullable=False),
        sa.Column("revision", sa.Text(), nullable=True),
        sa.Column("provider_kind", sa.Text(), nullable=False),
        sa.Column("provider_config", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("settings_hash", sa.String(length=12), nullable=False),
        sa.Column("series", sa.Text(), nullable=False),
        sa.Column("series_label", sa.Text(), nullable=False),
        sa.Column("family", sa.Text(), nullable=True),
        sa.Column("step", sa.BigInteger(), nullable=True),
        sa.Column("tokens_seen", sa.BigInteger(), nullable=True),
        sa.Column("first_seen_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_seen_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("model_id"),
        sa.UniqueConstraint("name", "model_hash", name="uq_models_name_hash"),
    )
    op.create_index("ix_models_family", "models", ["family"], unique=False)
    op.create_index(
        "ix_models_name_trgm",
        "models",
        ["name"],
        unique=False,
        postgresql_using="gin",
        postgresql_ops={"name": "gin_trgm_ops"},
    )
    op.create_index("ix_models_series", "models", ["series"], unique=False)
    op.create_index(
        "ix_models_series_trgm",
        "models",
        ["series"],
        unique=False,
        postgresql_using="gin",
        postgresql_ops={"series": "gin_trgm_ops"},
    )
    op.create_table(
        "saved_views",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("owner_email", sa.Text(), nullable=False),
        sa.Column("shared", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("page", sa.String(length=32), nullable=False),
        sa.Column("query", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_saved_views_owner", "saved_views", ["owner_email"], unique=False)
    op.create_index(
        "ix_saved_views_shared",
        "saved_views",
        ["shared"],
        unique=False,
        postgresql_where=sa.text("shared"),
    )
    op.create_table(
        "stats_cache",
        sa.Column("key", sa.String(length=64), nullable=False),
        sa.Column("value", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("key"),
    )
    op.create_table(
        "suite_defs",
        sa.Column("suite_name", sa.Text(), nullable=False),
        sa.Column("definition_hash", sa.String(length=16), nullable=False),
        sa.Column("aggregation", sa.String(length=32), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("children", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("first_seen_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_seen_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("suite_name", "definition_hash"),
    )
    op.create_table(
        "task_variants",
        sa.Column("task_name", sa.Text(), nullable=False),
        sa.Column("task_hash", sa.String(length=64), nullable=False),
        sa.Column("base_task", sa.Text(), nullable=True),
        sa.Column("config", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("primary_metric", sa.Text(), nullable=True),
        sa.Column(
            "metric_meta",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column("num_fewshot", sa.Integer(), nullable=True),
        sa.Column("task_limit", sa.Integer(), nullable=True),
        sa.Column("split", sa.Text(), nullable=True),
        sa.Column(
            "suites",
            postgresql.ARRAY(sa.Text()),
            server_default=sa.text("'{}'::text[]"),
            nullable=False,
        ),
        sa.Column("first_seen_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_seen_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("task_name", "task_hash"),
    )
    op.create_index(
        "ix_task_variants_task_name_trgm",
        "task_variants",
        ["task_name"],
        unique=False,
        postgresql_using="gin",
        postgresql_ops={"task_name": "gin_trgm_ops"},
    )
    op.create_table(
        "runs",
        sa.Column("run_id", sa.String(length=32), nullable=False),
        sa.Column("launch_id", sa.String(length=64), nullable=True),
        sa.Column("model_id", sa.String(length=12), nullable=False),
        sa.Column("model_name", sa.Text(), nullable=False),
        sa.Column("experiment_name", sa.Text(), nullable=True),
        sa.Column("experiment_group", sa.Text(), nullable=True),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("upload_state", sa.String(length=16), nullable=False),
        sa.Column("author", sa.Text(), nullable=True),
        sa.Column("uploaded_by", sa.Text(), nullable=False),
        sa.Column(
            "tags",
            postgresql.ARRAY(sa.Text()),
            server_default=sa.text("'{}'::text[]"),
            nullable=False,
        ),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("duration_seconds", sa.Double(), nullable=True),
        sa.Column("olmo_eval_version", sa.Text(), nullable=True),
        sa.Column("git_repo", sa.Text(), nullable=True),
        sa.Column("git_commit", sa.Text(), nullable=True),
        sa.Column("git_branch", sa.Text(), nullable=True),
        sa.Column("git_dirty", sa.Boolean(), nullable=True),
        sa.Column("beaker_experiment_id", sa.Text(), nullable=True),
        sa.Column("beaker_job_id", sa.Text(), nullable=True),
        sa.Column("beaker_result_dataset_id", sa.Text(), nullable=True),
        sa.Column("beaker_workspace", sa.Text(), nullable=True),
        sa.Column("beaker", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("environment", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("argv", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("task_specs", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("output_dir", sa.Text(), nullable=True),
        sa.Column("harness_config", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("provider_init_seconds", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column(
            "errors",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
        sa.Column("client", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("gcs_prefix", sa.Text(), nullable=False),
        sa.Column("num_tasks", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("num_failed_tasks", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("num_instances", sa.BigInteger(), server_default=sa.text("0"), nullable=False),
        sa.Column("headline", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("search_text", sa.Text(), nullable=False),
        sa.CheckConstraint(
            "status IN ('running', 'complete', 'partial', 'failed')", name="ck_runs_status"
        ),
        sa.CheckConstraint(
            "upload_state IN ('uploading', 'complete')", name="ck_runs_upload_state"
        ),
        sa.ForeignKeyConstraint(
            ["model_id"],
            ["models.model_id"],
        ),
        sa.PrimaryKeyConstraint("run_id"),
    )
    op.create_index("ix_runs_author", "runs", ["author"], unique=False)
    op.create_index("ix_runs_beaker_experiment_id", "runs", ["beaker_experiment_id"], unique=False)
    op.create_index("ix_runs_beaker_job_id", "runs", ["beaker_job_id"], unique=False)
    op.create_index(
        "ix_runs_beaker_result_dataset_id", "runs", ["beaker_result_dataset_id"], unique=False
    )
    op.create_index("ix_runs_beaker_workspace", "runs", ["beaker_workspace"], unique=False)
    op.create_index(
        "ix_runs_created",
        "runs",
        [sa.literal_column("created_at DESC"), sa.literal_column("run_id DESC")],
        unique=False,
    )
    op.create_index("ix_runs_experiment_group", "runs", ["experiment_group"], unique=False)
    op.create_index(
        "ix_runs_git_commit",
        "runs",
        ["git_commit"],
        unique=False,
        postgresql_ops={"git_commit": "text_pattern_ops"},
    )
    op.create_index("ix_runs_launch_id", "runs", ["launch_id"], unique=False)
    op.create_index("ix_runs_model_id", "runs", ["model_id"], unique=False)
    op.create_index(
        "ix_runs_search_text_trgm",
        "runs",
        ["search_text"],
        unique=False,
        postgresql_using="gin",
        postgresql_ops={"search_text": "gin_trgm_ops"},
    )
    op.create_index("ix_runs_tags", "runs", ["tags"], unique=False, postgresql_using="gin")
    op.create_index("ix_runs_uploaded_by", "runs", ["uploaded_by"], unique=False)
    op.create_table(
        "artifacts",
        sa.Column("run_id", sa.String(length=32), nullable=False),
        sa.Column("path", sa.Text(), nullable=False),
        sa.Column("kind", sa.String(length=32), nullable=False),
        sa.Column("task_name", sa.Text(), nullable=True),
        sa.Column("size_bytes", sa.BigInteger(), nullable=False),
        sa.Column("md5_b64", sa.String(length=24), nullable=False),
        sa.Column("content_type", sa.Text(), nullable=False),
        sa.Column("uploaded", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["run_id"], ["runs.run_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("run_id", "path"),
    )
    op.create_table(
        "inference_batches",
        sa.Column("run_id", sa.String(length=32), nullable=False),
        sa.Column("seq", sa.Integer(), nullable=False),
        sa.Column("ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("task_name", sa.Text(), nullable=True),
        sa.Column("total_requests", sa.Integer(), nullable=False),
        sa.Column("successful_requests", sa.Integer(), nullable=False),
        sa.Column("failed_requests", sa.Integer(), nullable=False),
        sa.Column("total_prompt_tokens", sa.BigInteger(), nullable=False),
        sa.Column("total_completion_tokens", sa.BigInteger(), nullable=False),
        sa.Column("wall_clock_time_s", sa.Double(), nullable=False),
        sa.Column("output_tokens_per_second", sa.Double(), nullable=False),
        sa.Column("mean_latency_s", sa.Double(), nullable=False),
        sa.Column("gpu_summary", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.ForeignKeyConstraint(["run_id"], ["runs.run_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("run_id", "seq"),
    )
    op.create_table(
        "run_inference",
        sa.Column("run_id", sa.String(length=32), nullable=False),
        sa.Column("source_paths", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("series", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("request_latency", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("gpu_devices", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["run_id"], ["runs.run_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("run_id"),
    )
    op.create_table(
        "suite_results",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column("run_id", sa.String(length=32), nullable=False),
        sa.Column("suite_name", sa.Text(), nullable=False),
        sa.Column("parent_suite", sa.Text(), nullable=True),
        sa.Column("aggregation", sa.String(length=32), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("children", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("definition_hash", sa.String(length=16), nullable=False),
        sa.Column("score", sa.Double(), nullable=True),
        sa.Column("stderr", sa.Double(), nullable=True),
        sa.Column("num_tasks", sa.Integer(), nullable=True),
        sa.Column("n_instances", sa.BigInteger(), nullable=True),
        sa.Column("metrics", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("primary_metric", sa.Text(), nullable=True),
        sa.Column("display_format", sa.String(length=8), nullable=False),
        sa.Column("higher_is_better", sa.Boolean(), nullable=True),
        sa.Column(
            "tasks_missing",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["run_id"], ["runs.run_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("run_id", "suite_name", name="uq_suite_results_run_suite"),
    )
    op.create_index(
        "ix_suite_results_suite_run", "suite_results", ["suite_name", "run_id"], unique=False
    )
    op.create_table(
        "task_results",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column("run_id", sa.String(length=32), nullable=False),
        sa.Column("task_name", sa.Text(), nullable=False),
        sa.Column("task_hash", sa.String(length=64), nullable=False),
        sa.Column("model_id", sa.String(length=12), nullable=False),
        sa.Column("run_created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("primary_metric", sa.Text(), nullable=True),
        sa.Column("score", sa.Double(), nullable=True),
        sa.Column("stderr", sa.Double(), nullable=True),
        sa.Column("score_is_mean", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("instance_scale", sa.Double(), server_default=sa.text("1"), nullable=False),
        sa.Column("metric_kind", sa.String(length=16), nullable=True),
        sa.Column("higher_is_better", sa.Boolean(), nullable=True),
        sa.Column("display_format", sa.String(length=8), server_default="raw", nullable=False),
        sa.Column("metrics", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column(
            "metric_meta",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column("num_instances", sa.Integer(), nullable=False),
        sa.Column("instances_processed", sa.Integer(), nullable=True),
        sa.Column("instances_failed", sa.Integer(), nullable=True),
        sa.Column("instance_count_expected", sa.Integer(), nullable=False),
        sa.Column("instances_stored", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("error_summary", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("duration_seconds", sa.Double(), nullable=True),
        sa.Column("completion_tokens_total", sa.BigInteger(), nullable=True),
        sa.Column("prompt_tokens_total", sa.BigInteger(), nullable=True),
        sa.Column("mean_completion_tokens", sa.Double(), nullable=True),
        sa.Column("truncation_rate", sa.Double(), nullable=True),
        sa.Column(
            "finish_reason_counts",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column("predictions_path", sa.Text(), nullable=True),
        sa.Column("requests_path", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finalized_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["run_id"], ["runs.run_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["task_name", "task_hash"],
            ["task_variants.task_name", "task_variants.task_hash"],
            name="fk_task_results_variant",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("run_id", "task_name", "task_hash", name="uq_task_results_run_task"),
    )
    op.create_index(
        "ix_task_results_model_task",
        "task_results",
        ["model_id", "task_name", sa.literal_column("run_created_at DESC")],
        unique=False,
    )
    op.create_index(
        "ix_task_results_task_created",
        "task_results",
        ["task_name", sa.literal_column("run_created_at DESC")],
        unique=False,
    )
    op.create_index(
        "ix_task_results_variant_model",
        "task_results",
        ["task_name", "task_hash", "model_id", sa.literal_column("run_created_at DESC")],
        unique=False,
    )
    op.create_table(
        "instance_results",
        sa.Column("task_result_id", sa.BigInteger(), nullable=False),
        sa.Column("native_id", sa.Text(), nullable=False),
        sa.Column("key_hash", sa.BigInteger(), nullable=False),
        sa.Column("doc_id", sa.Integer(), nullable=True),
        sa.Column("primary_score", sa.Double(), nullable=True),
        sa.Column(
            "metrics",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column("label", sa.Text(), nullable=True),
        sa.Column("extracted_answer", sa.Text(), nullable=True),
        sa.Column("finish_reason", sa.String(length=32), nullable=True),
        sa.Column("completion_tokens", sa.Integer(), nullable=True),
        sa.Column("prompt_tokens", sa.Integer(), nullable=True),
        sa.Column("num_outputs", sa.SmallInteger(), nullable=False),
        sa.Column("prompt_preview", sa.Text(), nullable=True),
        sa.Column("output_preview", sa.Text(), nullable=True),
        sa.Column("judge_verdict", sa.String(length=64), nullable=True),
        sa.Column("has_scoring_error", sa.Boolean(), nullable=False),
        sa.Column("has_execution_result", sa.Boolean(), nullable=False),
        sa.Column("has_judge_result", sa.Boolean(), nullable=False),
        sa.Column("has_trajectory", sa.Boolean(), nullable=False),
        sa.Column("pred_offset", sa.BigInteger(), nullable=True),
        sa.Column("req_offset", sa.BigInteger(), nullable=True),
        sa.Column("pred_length", sa.Integer(), nullable=True),
        sa.Column("req_length", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(["task_result_id"], ["task_results.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("task_result_id", "native_id"),
    )
    op.create_index(
        "ix_instance_results_doc_id", "instance_results", ["task_result_id", "doc_id"], unique=False
    )
    op.create_index(
        "ix_instance_results_key_hash",
        "instance_results",
        ["task_result_id", "key_hash"],
        unique=False,
    )
    op.create_index(
        "ix_instance_results_score",
        "instance_results",
        ["task_result_id", "primary_score"],
        unique=False,
    )
    op.create_table(
        "task_result_vectors",
        sa.Column("task_result_id", sa.BigInteger(), nullable=False),
        sa.Column("metric_key", sa.Text(), nullable=False),
        sa.Column("n", sa.Integer(), nullable=False),
        sa.Column("key_hashes", sa.LargeBinary(), nullable=False),
        sa.Column("scores", sa.LargeBinary(), nullable=False),
        sa.Column("built_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["task_result_id"], ["task_results.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("task_result_id"),
    )


def downgrade() -> None:
    for table in (
        "task_result_vectors",
        "instance_results",
        "task_results",
        "suite_results",
        "run_inference",
        "inference_batches",
        "artifacts",
        "runs",
        "task_variants",
        "suite_defs",
        "stats_cache",
        "saved_views",
        "models",
    ):
        op.drop_table(table)
