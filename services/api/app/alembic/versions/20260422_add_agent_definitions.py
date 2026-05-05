"""add agent definition registry table

Revision ID: 20260422_agent_defs
Revises: 20260412_exec_request_leases
Create Date: 2026-04-22
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260422_agent_defs"
down_revision = "20260412_exec_request_leases"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "agent_definitions",
        sa.Column("id", sa.String(), nullable=False),
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
        sa.Column(
            "model_config",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'{}'"),
        ),
        sa.Column(
            "allowed_capability_ids",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'[]'"),
        ),
        sa.Column(
            "memory_policy",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'{}'"),
        ),
        sa.Column(
            "guardrail_policy",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'{}'"),
        ),
        sa.Column(
            "workspace_policy",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'{}'"),
        ),
        sa.Column("metadata", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("user_id", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_agent_definitions_name", "agent_definitions", ["name"], unique=False)
    op.create_index(
        "ix_agent_definitions_agent_capability_id",
        "agent_definitions",
        ["agent_capability_id"],
        unique=False,
    )
    op.create_index(
        "ix_agent_definitions_enabled",
        "agent_definitions",
        ["enabled"],
        unique=False,
    )
    op.create_index(
        "ix_agent_definitions_user_id",
        "agent_definitions",
        ["user_id"],
        unique=False,
    )
    op.create_index(
        "ix_agent_definitions_created_at",
        "agent_definitions",
        ["created_at"],
        unique=False,
    )
    op.create_index(
        "ix_agent_definitions_updated_at",
        "agent_definitions",
        ["updated_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_agent_definitions_updated_at", table_name="agent_definitions")
    op.drop_index("ix_agent_definitions_created_at", table_name="agent_definitions")
    op.drop_index("ix_agent_definitions_user_id", table_name="agent_definitions")
    op.drop_index("ix_agent_definitions_enabled", table_name="agent_definitions")
    op.drop_index("ix_agent_definitions_agent_capability_id", table_name="agent_definitions")
    op.drop_index("ix_agent_definitions_name", table_name="agent_definitions")
    op.drop_table("agent_definitions")
