"""Task result content hash: lets an identical re-upload keep the stored instances.

Expand-only: the new column is nullable, so the previous release keeps working.

Revision ID: 0004
Revises: 0003
Create Date: 2026-10-07
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("task_results", sa.Column("content_hash", sa.String(length=64), nullable=True))


def downgrade() -> None:
    op.drop_column("task_results", "content_hash")
