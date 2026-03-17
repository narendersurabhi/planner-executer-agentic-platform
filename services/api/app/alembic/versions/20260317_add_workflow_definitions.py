"""add workflow definition and version tables

Revision ID: 20260317_add_workflow_definitions
Revises: 20260204_add_task_tool_inputs
Create Date: 2026-03-17
"""

from alembic import op
import sqlalchemy as sa


revision = "20260317_add_workflow_definitions"
down_revision = "20260204_add_task_tool_inputs"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "workflow_definitions",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("title", sa.String(), nullable=False),
        sa.Column("goal", sa.String(), nullable=False, server_default=""),
        sa.Column("context_json", sa.JSON(), nullable=False),
        sa.Column("draft", sa.JSON(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=True),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_workflow_definitions_user_id",
        "workflow_definitions",
        ["user_id"],
        unique=False,
    )

    op.create_table(
        "workflow_versions",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("definition_id", sa.String(), nullable=False),
        sa.Column("version_number", sa.Integer(), nullable=False),
        sa.Column("title", sa.String(), nullable=False),
        sa.Column("goal", sa.String(), nullable=False, server_default=""),
        sa.Column("context_json", sa.JSON(), nullable=False),
        sa.Column("draft", sa.JSON(), nullable=False),
        sa.Column("compiled_plan", sa.JSON(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=True),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["definition_id"], ["workflow_definitions.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_workflow_versions_definition_id",
        "workflow_versions",
        ["definition_id"],
        unique=False,
    )
    op.create_index(
        "ix_workflow_versions_user_id",
        "workflow_versions",
        ["user_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_workflow_versions_user_id", table_name="workflow_versions")
    op.drop_index("ix_workflow_versions_definition_id", table_name="workflow_versions")
    op.drop_table("workflow_versions")
    op.drop_index("ix_workflow_definitions_user_id", table_name="workflow_definitions")
    op.drop_table("workflow_definitions")
