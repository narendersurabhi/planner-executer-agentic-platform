"""add feedback records

Revision ID: 20260324_add_feedback_records
Revises: 20260322_render_id_backfill
Create Date: 2026-03-24
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260324_add_feedback_records"
down_revision = "20260322_render_id_backfill"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "feedback",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("target_type", sa.String(), nullable=False),
        sa.Column("target_id", sa.String(), nullable=False),
        sa.Column("session_id", sa.String(), nullable=True),
        sa.Column("job_id", sa.String(), nullable=True),
        sa.Column("plan_id", sa.String(), nullable=True),
        sa.Column("message_id", sa.String(), nullable=True),
        sa.Column("user_id", sa.String(), nullable=True),
        sa.Column("actor_key", sa.String(), nullable=True),
        sa.Column("sentiment", sa.String(), nullable=False),
        sa.Column("score", sa.Integer(), nullable=True),
        sa.Column("reason_codes", sa.JSON(), nullable=False, server_default=sa.text("'[]'")),
        sa.Column("comment", sa.Text(), nullable=True),
        sa.Column("snapshot", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("metadata", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_feedback_target_type", "feedback", ["target_type"], unique=False)
    op.create_index("ix_feedback_target_id", "feedback", ["target_id"], unique=False)
    op.create_index("ix_feedback_session_id", "feedback", ["session_id"], unique=False)
    op.create_index("ix_feedback_job_id", "feedback", ["job_id"], unique=False)
    op.create_index("ix_feedback_plan_id", "feedback", ["plan_id"], unique=False)
    op.create_index("ix_feedback_message_id", "feedback", ["message_id"], unique=False)
    op.create_index("ix_feedback_user_id", "feedback", ["user_id"], unique=False)
    op.create_index("ix_feedback_actor_key", "feedback", ["actor_key"], unique=False)
    op.create_index("ix_feedback_sentiment", "feedback", ["sentiment"], unique=False)
    op.create_index("ix_feedback_created_at", "feedback", ["created_at"], unique=False)
    op.create_index("ix_feedback_updated_at", "feedback", ["updated_at"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_feedback_updated_at", table_name="feedback")
    op.drop_index("ix_feedback_created_at", table_name="feedback")
    op.drop_index("ix_feedback_sentiment", table_name="feedback")
    op.drop_index("ix_feedback_actor_key", table_name="feedback")
    op.drop_index("ix_feedback_user_id", table_name="feedback")
    op.drop_index("ix_feedback_message_id", table_name="feedback")
    op.drop_index("ix_feedback_plan_id", table_name="feedback")
    op.drop_index("ix_feedback_job_id", table_name="feedback")
    op.drop_index("ix_feedback_session_id", table_name="feedback")
    op.drop_index("ix_feedback_target_id", table_name="feedback")
    op.drop_index("ix_feedback_target_type", table_name="feedback")
    op.drop_table("feedback")
