"""Per-generation diagnostics for task results.

Revision ID: 0006
Revises: 0005
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0006"
down_revision: str | None = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("task_results", sa.Column("generation_counts", postgresql.JSONB(), nullable=True))


def downgrade() -> None:
    op.drop_column("task_results", "generation_counts")
