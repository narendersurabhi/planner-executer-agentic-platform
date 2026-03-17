"""add workflow trigger and run tables

Revision ID: 20260317_add_workflow_triggers_and_runs
Revises: 20260317_add_workflow_definitions
Create Date: 2026-03-17
"""

from alembic import op
import sqlalchemy as sa


revision = "20260317_add_workflow_triggers_and_runs"
down_revision = "20260317_add_workflow_definitions"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "workflow_triggers",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("definition_id", sa.String(), nullable=False),
        sa.Column("title", sa.String(), nullable=False),
        sa.Column("trigger_type", sa.String(), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("config", sa.JSON(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=True),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["definition_id"], ["workflow_definitions.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_workflow_triggers_definition_id",
        "workflow_triggers",
        ["definition_id"],
        unique=False,
    )
    op.create_index(
        "ix_workflow_triggers_user_id",
        "workflow_triggers",
        ["user_id"],
        unique=False,
    )

    op.create_table(
        "workflow_runs",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("definition_id", sa.String(), nullable=False),
        sa.Column("version_id", sa.String(), nullable=False),
        sa.Column("trigger_id", sa.String(), nullable=True),
        sa.Column("title", sa.String(), nullable=False),
        sa.Column("goal", sa.String(), nullable=False, server_default=""),
        sa.Column("requested_context", sa.JSON(), nullable=False),
        sa.Column("job_id", sa.String(), nullable=False),
        sa.Column("plan_id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=True),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["definition_id"], ["workflow_definitions.id"]),
        sa.ForeignKeyConstraint(["version_id"], ["workflow_versions.id"]),
        sa.ForeignKeyConstraint(["trigger_id"], ["workflow_triggers.id"]),
        sa.ForeignKeyConstraint(["job_id"], ["jobs.id"]),
        sa.ForeignKeyConstraint(["plan_id"], ["plans.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_workflow_runs_definition_id", "workflow_runs", ["definition_id"], unique=False)
    op.create_index("ix_workflow_runs_version_id", "workflow_runs", ["version_id"], unique=False)
    op.create_index("ix_workflow_runs_trigger_id", "workflow_runs", ["trigger_id"], unique=False)
    op.create_index("ix_workflow_runs_job_id", "workflow_runs", ["job_id"], unique=False)
    op.create_index("ix_workflow_runs_plan_id", "workflow_runs", ["plan_id"], unique=False)
    op.create_index("ix_workflow_runs_user_id", "workflow_runs", ["user_id"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_workflow_runs_user_id", table_name="workflow_runs")
    op.drop_index("ix_workflow_runs_plan_id", table_name="workflow_runs")
    op.drop_index("ix_workflow_runs_job_id", table_name="workflow_runs")
    op.drop_index("ix_workflow_runs_trigger_id", table_name="workflow_runs")
    op.drop_index("ix_workflow_runs_version_id", table_name="workflow_runs")
    op.drop_index("ix_workflow_runs_definition_id", table_name="workflow_runs")
    op.drop_table("workflow_runs")

    op.drop_index("ix_workflow_triggers_user_id", table_name="workflow_triggers")
    op.drop_index("ix_workflow_triggers_definition_id", table_name="workflow_triggers")
    op.drop_table("workflow_triggers")
