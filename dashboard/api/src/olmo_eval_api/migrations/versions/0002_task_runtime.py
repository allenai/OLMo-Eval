"""Task runtime: run startup and processing times, task spans and reported token totals.

Expand-only: every new column is nullable, so the previous release keeps working.

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-02
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TZ = sa.DateTime(timezone=True)


def upgrade() -> None:
    op.add_column("runs", sa.Column("startup_seconds", sa.Double(), nullable=True))
    op.add_column("runs", sa.Column("processing_started_at", _TZ, nullable=True))
    op.add_column("runs", sa.Column("processing_seconds", sa.Double(), nullable=True))
    op.add_column("task_results", sa.Column("first_request_at", _TZ, nullable=True))
    op.add_column("task_results", sa.Column("last_completed_at", _TZ, nullable=True))
    op.add_column(
        "task_results", sa.Column("attributed_inference_seconds", sa.Double(), nullable=True)
    )
    op.add_column(
        "task_results", sa.Column("reported_prompt_tokens_total", sa.BigInteger(), nullable=True)
    )
    op.add_column(
        "task_results",
        sa.Column("reported_completion_tokens_total", sa.BigInteger(), nullable=True),
    )


def downgrade() -> None:
    for column in (
        "reported_completion_tokens_total",
        "reported_prompt_tokens_total",
        "attributed_inference_seconds",
        "last_completed_at",
        "first_request_at",
    ):
        op.drop_column("task_results", column)
    for column in ("processing_seconds", "processing_started_at", "startup_seconds"):
        op.drop_column("runs", column)
