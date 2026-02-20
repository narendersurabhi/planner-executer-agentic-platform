"""add task tool_inputs column

Revision ID: 20260204_add_task_tool_inputs
Revises: 20260204_add_task_intent
Create Date: 2026-02-04
"""

from alembic import op
import sqlalchemy as sa


revision = "20260204_add_task_tool_inputs"
down_revision = "20260204_add_task_intent"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("tasks", sa.Column("tool_inputs", sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column("tasks", "tool_inputs")
