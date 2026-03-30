"""add runtime observability tables

Revision ID: 20260320_add_runtime_observability
Revises: 20260320_add_task_results
Create Date: 2026-03-20
"""

from alembic import op
import sqlalchemy as sa


revision = "20260320_add_runtime_observability"
down_revision = "20260320_add_task_results"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "step_attempts",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("run_id", sa.String(), nullable=False),
        sa.Column("job_id", sa.String(), nullable=False),
        sa.Column("step_id", sa.String(), nullable=False),
        sa.Column("attempt_number", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("status", sa.String(), nullable=False, server_default=""),
        sa.Column("worker_id", sa.String(), nullable=True),
        sa.Column("started_at", sa.DateTime(), nullable=False),
        sa.Column("finished_at", sa.DateTime(), nullable=True),
        sa.Column("error_code", sa.String(), nullable=True),
        sa.Column("error_message", sa.String(), nullable=True),
        sa.Column("retry_classification", sa.String(), nullable=True),
        sa.Column("result_summary", sa.JSON(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_step_attempts_run_id", "step_attempts", ["run_id"], unique=False)
    op.create_index("ix_step_attempts_job_id", "step_attempts", ["job_id"], unique=False)
    op.create_index("ix_step_attempts_step_id", "step_attempts", ["step_id"], unique=False)

    op.create_table(
        "invocations",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("run_id", sa.String(), nullable=False),
        sa.Column("job_id", sa.String(), nullable=False),
        sa.Column("step_id", sa.String(), nullable=False),
        sa.Column("step_attempt_id", sa.String(), nullable=False),
        sa.Column("request_id", sa.String(), nullable=True),
        sa.Column("capability_id", sa.String(), nullable=False),
        sa.Column("adapter_id", sa.String(), nullable=True),
        sa.Column("request", sa.JSON(), nullable=False),
        sa.Column("response", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(), nullable=False, server_default=""),
        sa.Column("started_at", sa.DateTime(), nullable=False),
        sa.Column("finished_at", sa.DateTime(), nullable=True),
        sa.Column("error_code", sa.String(), nullable=True),
        sa.Column("error_message", sa.String(), nullable=True),
        sa.ForeignKeyConstraint(["step_attempt_id"], ["step_attempts.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_invocations_run_id", "invocations", ["run_id"], unique=False)
    op.create_index("ix_invocations_job_id", "invocations", ["job_id"], unique=False)
    op.create_index("ix_invocations_step_id", "invocations", ["step_id"], unique=False)
    op.create_index(
        "ix_invocations_step_attempt_id", "invocations", ["step_attempt_id"], unique=False
    )

    op.create_table(
        "run_events",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("run_id", sa.String(), nullable=False),
        sa.Column("job_id", sa.String(), nullable=False),
        sa.Column("step_id", sa.String(), nullable=True),
        sa.Column("step_attempt_id", sa.String(), nullable=True),
        sa.Column("event_type", sa.String(), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("occurred_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["step_attempt_id"], ["step_attempts.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_run_events_run_id", "run_events", ["run_id"], unique=False)
    op.create_index("ix_run_events_job_id", "run_events", ["job_id"], unique=False)
    op.create_index("ix_run_events_step_id", "run_events", ["step_id"], unique=False)
    op.create_index(
        "ix_run_events_step_attempt_id", "run_events", ["step_attempt_id"], unique=False
    )
    op.create_index("ix_run_events_event_type", "run_events", ["event_type"], unique=False)
    op.create_index("ix_run_events_occurred_at", "run_events", ["occurred_at"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_run_events_occurred_at", table_name="run_events")
    op.drop_index("ix_run_events_event_type", table_name="run_events")
    op.drop_index("ix_run_events_step_attempt_id", table_name="run_events")
    op.drop_index("ix_run_events_step_id", table_name="run_events")
    op.drop_index("ix_run_events_job_id", table_name="run_events")
    op.drop_index("ix_run_events_run_id", table_name="run_events")
    op.drop_table("run_events")

    op.drop_index("ix_invocations_step_attempt_id", table_name="invocations")
    op.drop_index("ix_invocations_step_id", table_name="invocations")
    op.drop_index("ix_invocations_job_id", table_name="invocations")
    op.drop_index("ix_invocations_run_id", table_name="invocations")
    op.drop_table("invocations")

    op.drop_index("ix_step_attempts_step_id", table_name="step_attempts")
    op.drop_index("ix_step_attempts_job_id", table_name="step_attempts")
    op.drop_index("ix_step_attempts_run_id", table_name="step_attempts")
    op.drop_table("step_attempts")
