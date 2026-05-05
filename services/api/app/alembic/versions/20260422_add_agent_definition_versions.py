"""add immutable agent definition versions

Revision ID: 20260422_agent_def_versions
Revises: 20260422_agent_defs
Create Date: 2026-04-22
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260422_agent_def_versions"
down_revision = "20260422_agent_defs"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "agent_definition_versions",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("agent_definition_id", sa.String(), nullable=False),
        sa.Column("version_number", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("agent_capability_id", sa.String(), nullable=False),
        sa.Column("instructions", sa.Text(), nullable=False),
        sa.Column("default_goal", sa.String(), nullable=False, server_default=""),
        sa.Column("default_workspace_path", sa.String(), nullable=True),
        sa.Column(
            "default_constraints",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'[]'"),
        ),
        sa.Column("default_max_steps", sa.Integer(), nullable=True),
        sa.Column("model_config", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column(
            "allowed_capability_ids",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'[]'"),
        ),
        sa.Column("memory_policy", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("guardrail_policy", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("workspace_policy", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column(
            "definition_metadata",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'{}'"),
        ),
        sa.Column(
            "version_metadata",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'{}'"),
        ),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("user_id", sa.String(), nullable=True),
        sa.Column("published_by", sa.String(), nullable=True),
        sa.Column("version_note", sa.Text(), nullable=True),
        sa.Column("definition_created_at", sa.DateTime(), nullable=True),
        sa.Column("definition_updated_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["agent_definition_id"], ["agent_definitions.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "agent_definition_id",
            "version_number",
            name="uq_agent_definition_versions_definition_version",
        ),
    )
    op.create_index(
        "ix_agent_definition_versions_agent_definition_id",
        "agent_definition_versions",
        ["agent_definition_id"],
        unique=False,
    )
    op.create_index(
        "ix_agent_definition_versions_agent_capability_id",
        "agent_definition_versions",
        ["agent_capability_id"],
        unique=False,
    )
    op.create_index(
        "ix_agent_definition_versions_created_at",
        "agent_definition_versions",
        ["created_at"],
        unique=False,
    )
    op.create_index(
        "ix_agent_definition_versions_enabled",
        "agent_definition_versions",
        ["enabled"],
        unique=False,
    )
    op.create_index(
        "ix_agent_definition_versions_name",
        "agent_definition_versions",
        ["name"],
        unique=False,
    )
    op.create_index(
        "ix_agent_definition_versions_published_by",
        "agent_definition_versions",
        ["published_by"],
        unique=False,
    )
    op.create_index(
        "ix_agent_definition_versions_user_id",
        "agent_definition_versions",
        ["user_id"],
        unique=False,
    )
    op.create_index(
        "ix_agent_definition_versions_version_number",
        "agent_definition_versions",
        ["version_number"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_agent_definition_versions_version_number",
        table_name="agent_definition_versions",
    )
    op.drop_index(
        "ix_agent_definition_versions_user_id",
        table_name="agent_definition_versions",
    )
    op.drop_index(
        "ix_agent_definition_versions_published_by",
        table_name="agent_definition_versions",
    )
    op.drop_index("ix_agent_definition_versions_name", table_name="agent_definition_versions")
    op.drop_index("ix_agent_definition_versions_enabled", table_name="agent_definition_versions")
    op.drop_index("ix_agent_definition_versions_created_at", table_name="agent_definition_versions")
    op.drop_index(
        "ix_agent_definition_versions_agent_capability_id",
        table_name="agent_definition_versions",
    )
    op.drop_index(
        "ix_agent_definition_versions_agent_definition_id",
        table_name="agent_definition_versions",
    )
    op.drop_table("agent_definition_versions")
