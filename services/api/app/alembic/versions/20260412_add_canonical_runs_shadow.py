"""add canonical run shadow tables

Revision ID: 20260412_add_canonical_runs_shadow
Revises: 20260324_add_feedback_records
Create Date: 2026-04-12
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260412_add_canonical_runs_shadow"
down_revision = "20260324_add_feedback_records"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "runs",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("kind", sa.String(), nullable=False),
        sa.Column("title", sa.String(), nullable=False),
        sa.Column("goal", sa.String(), nullable=False),
        sa.Column(
            "requested_context",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'{}'"),
        ),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("job_id", sa.String(), nullable=False),
        sa.Column("workflow_run_id", sa.String(), nullable=True),
        sa.Column("plan_id", sa.String(), nullable=True),
        sa.Column("source_definition_id", sa.String(), nullable=True),
        sa.Column("source_version_id", sa.String(), nullable=True),
        sa.Column("source_trigger_id", sa.String(), nullable=True),
        sa.Column("user_id", sa.String(), nullable=True),
        sa.Column("run_spec", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("metadata", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["job_id"], ["jobs.id"]),
        sa.ForeignKeyConstraint(["workflow_run_id"], ["workflow_runs.id"]),
        sa.ForeignKeyConstraint(["plan_id"], ["plans.id"]),
        sa.ForeignKeyConstraint(["source_definition_id"], ["workflow_definitions.id"]),
        sa.ForeignKeyConstraint(["source_version_id"], ["workflow_versions.id"]),
        sa.ForeignKeyConstraint(["source_trigger_id"], ["workflow_triggers.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_runs_kind", "runs", ["kind"], unique=False)
    op.create_index("ix_runs_status", "runs", ["status"], unique=False)
    op.create_index("ix_runs_job_id", "runs", ["job_id"], unique=False)
    op.create_index("ix_runs_workflow_run_id", "runs", ["workflow_run_id"], unique=False)
    op.create_index("ix_runs_plan_id", "runs", ["plan_id"], unique=False)
    op.create_index("ix_runs_source_definition_id", "runs", ["source_definition_id"], unique=False)
    op.create_index("ix_runs_source_version_id", "runs", ["source_version_id"], unique=False)
    op.create_index("ix_runs_source_trigger_id", "runs", ["source_trigger_id"], unique=False)
    op.create_index("ix_runs_user_id", "runs", ["user_id"], unique=False)
    op.create_index("ix_runs_created_at", "runs", ["created_at"], unique=False)
    op.create_index("ix_runs_updated_at", "runs", ["updated_at"], unique=False)

    op.create_table(
        "run_steps",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("run_id", sa.String(), nullable=False),
        sa.Column("job_id", sa.String(), nullable=False),
        sa.Column("plan_id", sa.String(), nullable=True),
        sa.Column("spec_step_id", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("description", sa.String(), nullable=False),
        sa.Column("instruction", sa.String(), nullable=False, server_default=""),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("intent", sa.String(), nullable=True),
        sa.Column("capability_request_id", sa.String(), nullable=False),
        sa.Column("execution_request_id", sa.String(), nullable=True),
        sa.Column("capability_id", sa.String(), nullable=False),
        sa.Column(
            "input_bindings",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'{}'"),
        ),
        sa.Column("execution_gate", sa.JSON(), nullable=True),
        sa.Column(
            "retry_policy",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'{}'"),
        ),
        sa.Column(
            "acceptance_policy",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'{}'"),
        ),
        sa.Column(
            "depends_on",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'[]'"),
        ),
        sa.Column("metadata", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["run_id"], ["runs.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_run_steps_run_id", "run_steps", ["run_id"], unique=False)
    op.create_index("ix_run_steps_job_id", "run_steps", ["job_id"], unique=False)
    op.create_index("ix_run_steps_plan_id", "run_steps", ["plan_id"], unique=False)
    op.create_index("ix_run_steps_spec_step_id", "run_steps", ["spec_step_id"], unique=False)
    op.create_index("ix_run_steps_status", "run_steps", ["status"], unique=False)
    op.create_index("ix_run_steps_capability_id", "run_steps", ["capability_id"], unique=False)
    op.create_index("ix_run_steps_updated_at", "run_steps", ["updated_at"], unique=False)

    op.create_table(
        "execution_requests",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("run_id", sa.String(), nullable=False),
        sa.Column("job_id", sa.String(), nullable=False),
        sa.Column("step_id", sa.String(), nullable=False),
        sa.Column("step_attempt_id", sa.String(), nullable=True),
        sa.Column("attempt_number", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("status", sa.String(), nullable=False, server_default=""),
        sa.Column("request", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["step_attempt_id"], ["step_attempts.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_execution_requests_run_id", "execution_requests", ["run_id"], unique=False)
    op.create_index("ix_execution_requests_job_id", "execution_requests", ["job_id"], unique=False)
    op.create_index("ix_execution_requests_step_id", "execution_requests", ["step_id"], unique=False)
    op.create_index(
        "ix_execution_requests_step_attempt_id",
        "execution_requests",
        ["step_attempt_id"],
        unique=False,
    )
    op.create_index("ix_execution_requests_status", "execution_requests", ["status"], unique=False)
    op.create_index(
        "ix_execution_requests_created_at",
        "execution_requests",
        ["created_at"],
        unique=False,
    )
    op.create_index(
        "ix_execution_requests_updated_at",
        "execution_requests",
        ["updated_at"],
        unique=False,
    )

    op.create_table(
        "step_checkpoints",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("run_id", sa.String(), nullable=False),
        sa.Column("job_id", sa.String(), nullable=False),
        sa.Column("step_id", sa.String(), nullable=False),
        sa.Column("step_attempt_id", sa.String(), nullable=True),
        sa.Column("checkpoint_key", sa.String(), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("input_digest", sa.String(), nullable=True),
        sa.Column("replay_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("source", sa.String(), nullable=True),
        sa.Column("outcome", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["step_attempt_id"], ["step_attempts.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_step_checkpoints_run_id", "step_checkpoints", ["run_id"], unique=False)
    op.create_index("ix_step_checkpoints_job_id", "step_checkpoints", ["job_id"], unique=False)
    op.create_index("ix_step_checkpoints_step_id", "step_checkpoints", ["step_id"], unique=False)
    op.create_index(
        "ix_step_checkpoints_step_attempt_id",
        "step_checkpoints",
        ["step_attempt_id"],
        unique=False,
    )
    op.create_index(
        "ix_step_checkpoints_checkpoint_key",
        "step_checkpoints",
        ["checkpoint_key"],
        unique=False,
    )
    op.create_index("ix_step_checkpoints_created_at", "step_checkpoints", ["created_at"], unique=False)
    op.create_index("ix_step_checkpoints_updated_at", "step_checkpoints", ["updated_at"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_step_checkpoints_updated_at", table_name="step_checkpoints")
    op.drop_index("ix_step_checkpoints_created_at", table_name="step_checkpoints")
    op.drop_index("ix_step_checkpoints_checkpoint_key", table_name="step_checkpoints")
    op.drop_index("ix_step_checkpoints_step_attempt_id", table_name="step_checkpoints")
    op.drop_index("ix_step_checkpoints_step_id", table_name="step_checkpoints")
    op.drop_index("ix_step_checkpoints_job_id", table_name="step_checkpoints")
    op.drop_index("ix_step_checkpoints_run_id", table_name="step_checkpoints")
    op.drop_table("step_checkpoints")

    op.drop_index("ix_execution_requests_updated_at", table_name="execution_requests")
    op.drop_index("ix_execution_requests_created_at", table_name="execution_requests")
    op.drop_index("ix_execution_requests_status", table_name="execution_requests")
    op.drop_index("ix_execution_requests_step_attempt_id", table_name="execution_requests")
    op.drop_index("ix_execution_requests_step_id", table_name="execution_requests")
    op.drop_index("ix_execution_requests_job_id", table_name="execution_requests")
    op.drop_index("ix_execution_requests_run_id", table_name="execution_requests")
    op.drop_table("execution_requests")

    op.drop_index("ix_run_steps_updated_at", table_name="run_steps")
    op.drop_index("ix_run_steps_capability_id", table_name="run_steps")
    op.drop_index("ix_run_steps_status", table_name="run_steps")
    op.drop_index("ix_run_steps_spec_step_id", table_name="run_steps")
    op.drop_index("ix_run_steps_plan_id", table_name="run_steps")
    op.drop_index("ix_run_steps_job_id", table_name="run_steps")
    op.drop_index("ix_run_steps_run_id", table_name="run_steps")
    op.drop_table("run_steps")

    op.drop_index("ix_runs_updated_at", table_name="runs")
    op.drop_index("ix_runs_created_at", table_name="runs")
    op.drop_index("ix_runs_user_id", table_name="runs")
    op.drop_index("ix_runs_source_trigger_id", table_name="runs")
    op.drop_index("ix_runs_source_version_id", table_name="runs")
    op.drop_index("ix_runs_source_definition_id", table_name="runs")
    op.drop_index("ix_runs_plan_id", table_name="runs")
    op.drop_index("ix_runs_workflow_run_id", table_name="runs")
    op.drop_index("ix_runs_job_id", table_name="runs")
    op.drop_index("ix_runs_status", table_name="runs")
    op.drop_index("ix_runs_kind", table_name="runs")
    op.drop_table("runs")
