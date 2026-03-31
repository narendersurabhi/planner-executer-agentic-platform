"""add durable task results table

Revision ID: 20260320_add_task_results
Revises: 20260317_add_workflow_triggers_and_runs
Create Date: 2026-03-20
"""

from alembic import op
import sqlalchemy as sa


revision = "20260320_add_task_results"
down_revision = "20260317_add_workflow_triggers_and_runs"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "task_results",
        sa.Column("task_id", sa.String(), nullable=False),
        sa.Column("job_id", sa.String(), nullable=True),
        sa.Column("plan_id", sa.String(), nullable=True),
        sa.Column("status", sa.String(), nullable=False, server_default=""),
        sa.Column("result", sa.JSON(), nullable=False),
        sa.Column("latest_error", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("task_id"),
    )
    op.create_index("ix_task_results_job_id", "task_results", ["job_id"], unique=False)
    op.create_index("ix_task_results_plan_id", "task_results", ["plan_id"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_task_results_plan_id", table_name="task_results")
    op.drop_index("ix_task_results_job_id", table_name="task_results")
    op.drop_table("task_results")
