"""add execution request lease and heartbeat fields

Revision ID: 20260412_add_execution_request_leases
Revises: 20260412_add_canonical_runs_shadow
Create Date: 2026-04-12
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260412_add_execution_request_leases"
down_revision = "20260412_add_canonical_runs_shadow"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("step_attempts", sa.Column("lease_owner", sa.String(), nullable=True))
    op.add_column("step_attempts", sa.Column("lease_expires_at", sa.DateTime(), nullable=True))
    op.add_column("step_attempts", sa.Column("last_heartbeat_at", sa.DateTime(), nullable=True))
    op.add_column(
        "step_attempts",
        sa.Column("heartbeat_count", sa.Integer(), nullable=False, server_default="0"),
    )
    op.create_index(
        "ix_step_attempts_lease_expires_at",
        "step_attempts",
        ["lease_expires_at"],
        unique=False,
    )
    op.create_index(
        "ix_step_attempts_last_heartbeat_at",
        "step_attempts",
        ["last_heartbeat_at"],
        unique=False,
    )

    op.add_column("execution_requests", sa.Column("request_id", sa.String(), nullable=True))
    op.add_column("execution_requests", sa.Column("capability_id", sa.String(), nullable=True))
    op.add_column(
        "execution_requests",
        sa.Column("retry_policy", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
    )
    op.add_column(
        "execution_requests",
        sa.Column("policy_snapshot", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
    )
    op.add_column(
        "execution_requests",
        sa.Column("context_provenance", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
    )
    op.add_column("execution_requests", sa.Column("deadline_at", sa.DateTime(), nullable=True))
    op.add_column(
        "execution_requests", sa.Column("retry_classification", sa.String(), nullable=True)
    )
    op.add_column("execution_requests", sa.Column("lease_owner", sa.String(), nullable=True))
    op.add_column("execution_requests", sa.Column("lease_expires_at", sa.DateTime(), nullable=True))
    op.add_column("execution_requests", sa.Column("last_heartbeat_at", sa.DateTime(), nullable=True))
    op.create_index(
        "ix_execution_requests_request_id",
        "execution_requests",
        ["request_id"],
        unique=False,
    )
    op.create_index(
        "ix_execution_requests_capability_id",
        "execution_requests",
        ["capability_id"],
        unique=False,
    )
    op.create_index(
        "ix_execution_requests_deadline_at",
        "execution_requests",
        ["deadline_at"],
        unique=False,
    )
    op.create_index(
        "ix_execution_requests_lease_expires_at",
        "execution_requests",
        ["lease_expires_at"],
        unique=False,
    )
    op.create_index(
        "ix_execution_requests_last_heartbeat_at",
        "execution_requests",
        ["last_heartbeat_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_execution_requests_last_heartbeat_at", table_name="execution_requests")
    op.drop_index("ix_execution_requests_lease_expires_at", table_name="execution_requests")
    op.drop_index("ix_execution_requests_deadline_at", table_name="execution_requests")
    op.drop_index("ix_execution_requests_capability_id", table_name="execution_requests")
    op.drop_index("ix_execution_requests_request_id", table_name="execution_requests")
    op.drop_column("execution_requests", "last_heartbeat_at")
    op.drop_column("execution_requests", "lease_expires_at")
    op.drop_column("execution_requests", "lease_owner")
    op.drop_column("execution_requests", "retry_classification")
    op.drop_column("execution_requests", "deadline_at")
    op.drop_column("execution_requests", "context_provenance")
    op.drop_column("execution_requests", "policy_snapshot")
    op.drop_column("execution_requests", "retry_policy")
    op.drop_column("execution_requests", "capability_id")
    op.drop_column("execution_requests", "request_id")

    op.drop_index("ix_step_attempts_last_heartbeat_at", table_name="step_attempts")
    op.drop_index("ix_step_attempts_lease_expires_at", table_name="step_attempts")
    op.drop_column("step_attempts", "heartbeat_count")
    op.drop_column("step_attempts", "last_heartbeat_at")
    op.drop_column("step_attempts", "lease_expires_at")
    op.drop_column("step_attempts", "lease_owner")
