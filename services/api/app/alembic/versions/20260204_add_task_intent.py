"""add task intent column

Revision ID: 20260204_add_task_intent
Revises:
Create Date: 2026-02-04
"""

from alembic import op
import sqlalchemy as sa


revision = "20260204_add_task_intent"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("tasks", sa.Column("intent", sa.String(), nullable=True))


def downgrade() -> None:
    op.drop_column("tasks", "intent")
