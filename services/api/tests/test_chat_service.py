import os
from datetime import UTC, datetime, timedelta

os.environ["DATABASE_URL"] = "sqlite:///./test.db"
os.environ["ORCHESTRATOR_ENABLED"] = "false"
os.environ["JOB_RECOVERY_ENABLED"] = "false"
os.environ["POLICY_GATE_ENABLED"] = "false"
os.environ["CAPABILITY_MODE"] = "enabled"

from services.api.app import chat_service  # noqa: E402
from services.api.app.database import Base, SessionLocal, engine  # noqa: E402
from services.api.app.models import ChatSessionRecord  # noqa: E402


Base.metadata.create_all(bind=engine)


def test_merge_session_metadata_for_persistence_preserves_clarification_state_on_conflict() -> None:
    latest_metadata = {
        "_chat_state_version": 2,
        "context_json": {"topic": "Deployment report"},
        "pending_clarification": {
            "schema_version": "clarification_state_v1",
            "state_version": 2,
            "questions": ["What tone should it use?"],
            "known_slot_values": {"audience": "Senior AI engineers"},
            "resolved_slots": {"audience": "Senior AI engineers"},
            "slot_provenance": {"audience": "explicit_user"},
        },
    }
    desired_metadata = {
        "_chat_state_version": 1,
        "context_json": {"path": "artifacts/report.docx"},
        "pending_clarification": {
            "schema_version": "clarification_state_v1",
            "state_version": 1,
            "questions": ["What output path or filename should be used?"],
            "known_slot_values": {"tone": "practical"},
            "resolved_slots": {"tone": "practical"},
            "slot_provenance": {"tone": "clarification_normalized"},
        },
    }

    merged, conflicted = chat_service._merge_session_metadata_for_persistence(
        latest_metadata=latest_metadata,
        desired_metadata=desired_metadata,
        cleared_keys=set(),
        loaded_state_version=1,
    )

    assert conflicted is True
    assert merged["_chat_state_version"] == 3
    assert merged["_chat_state_conflict_count"] == 1
    assert merged["context_json"]["topic"] == "Deployment report"
    assert merged["context_json"]["path"] == "artifacts/report.docx"
    assert merged["pending_clarification"]["known_slot_values"]["audience"] == "Senior AI engineers"
    assert merged["pending_clarification"]["known_slot_values"]["tone"] == "practical"
    assert merged["pending_clarification"]["slot_provenance"]["audience"] == "explicit_user"
    assert (
        merged["pending_clarification"]["slot_provenance"]["tone"]
        == "clarification_normalized"
    )


def test_merge_session_metadata_for_persistence_respects_cleared_pending_state() -> None:
    latest_metadata = {
        "_chat_state_version": 4,
        "draft_goal": "Create a document",
        "pending_clarification": {
            "schema_version": "clarification_state_v1",
            "state_version": 4,
            "questions": ["Who is the target audience?"],
        },
        "context_json": {"topic": "Deployment report"},
    }
    desired_metadata = {
        "_chat_state_version": 4,
        "active_job_id": "job-123",
        "context_json": {"topic": "Deployment report", "tone": "practical"},
    }

    merged, conflicted = chat_service._merge_session_metadata_for_persistence(
        latest_metadata=latest_metadata,
        desired_metadata=desired_metadata,
        cleared_keys={"draft_goal", "pending_clarification"},
        loaded_state_version=4,
    )

    assert conflicted is False
    assert "draft_goal" not in merged
    assert "pending_clarification" not in merged
    assert merged["active_job_id"] == "job-123"
    assert merged["context_json"]["topic"] == "Deployment report"
    assert merged["context_json"]["tone"] == "practical"


def test_persist_chat_session_state_retries_on_updated_at_conflict() -> None:
    runtime = type("_Runtime", (), {"utcnow": lambda self: datetime.now(UTC)})()

    session_id = "chat-session-concurrency"
    created_at = datetime.now(UTC)

    db_create = SessionLocal()
    try:
        record = ChatSessionRecord(
            id=session_id,
            title="New chat",
            metadata_json={
                "_chat_state_version": 1,
                "pending_clarification": {
                    "schema_version": "clarification_state_v1",
                    "state_version": 1,
                    "known_slot_values": {"audience": "Senior AI engineers"},
                    "resolved_slots": {"audience": "Senior AI engineers"},
                    "slot_provenance": {"audience": "explicit_user"},
                },
            },
            created_at=created_at,
            updated_at=created_at,
        )
        db_create.add(record)
        db_create.commit()
    finally:
        db_create.close()

    db_stale = SessionLocal()
    db_latest = SessionLocal()
    try:
        stale_record = (
            db_stale.query(ChatSessionRecord)
            .filter(ChatSessionRecord.id == session_id)
            .first()
        )
        assert stale_record is not None
        loaded_updated_at = stale_record.updated_at
        loaded_state_version = chat_service._session_state_version(stale_record.metadata_json)

        latest_record = (
            db_latest.query(ChatSessionRecord)
            .filter(ChatSessionRecord.id == session_id)
            .first()
        )
        assert latest_record is not None
        latest_record.metadata_json = {
            "_chat_state_version": 2,
            "pending_clarification": {
                "schema_version": "clarification_state_v1",
                "state_version": 2,
                "known_slot_values": {
                    "audience": "Senior AI engineers",
                    "topic": "Kubernetes deployment report",
                },
                "resolved_slots": {
                    "audience": "Senior AI engineers",
                    "topic": "Kubernetes deployment report",
                },
                "slot_provenance": {
                    "audience": "explicit_user",
                    "topic": "explicit_user",
                },
            },
        }
        latest_record.updated_at = created_at + timedelta(seconds=1)
        db_latest.commit()

        merged = chat_service._persist_chat_session_state(
            db=db_stale,
            record=stale_record,
            desired_metadata={
                "pending_clarification": {
                    "schema_version": "clarification_state_v1",
                    "state_version": 1,
                    "known_slot_values": {"tone": "practical"},
                    "resolved_slots": {"tone": "practical"},
                    "slot_provenance": {"tone": "clarification_normalized"},
                },
            },
            desired_title="Deployment chat",
            loaded_updated_at=loaded_updated_at,
            loaded_state_version=loaded_state_version,
            cleared_keys=set(),
            runtime=runtime,
        )
        db_stale.commit()
    finally:
        db_stale.close()
        db_latest.close()

    db_verify = SessionLocal()
    try:
        persisted = (
            db_verify.query(ChatSessionRecord)
            .filter(ChatSessionRecord.id == session_id)
            .first()
        )
        assert persisted is not None
        assert merged["_chat_state_version"] == 3
        assert merged["_chat_state_conflict_count"] == 1
        assert persisted.title == "Deployment chat"
        assert (
            persisted.metadata_json["pending_clarification"]["known_slot_values"]["audience"]
            == "Senior AI engineers"
        )
        assert (
            persisted.metadata_json["pending_clarification"]["known_slot_values"]["topic"]
            == "Kubernetes deployment report"
        )
        assert (
            persisted.metadata_json["pending_clarification"]["known_slot_values"]["tone"]
            == "practical"
        )
    finally:
        cleanup = db_verify.query(ChatSessionRecord).filter(ChatSessionRecord.id == session_id).first()
        if cleanup is not None:
            db_verify.delete(cleanup)
            db_verify.commit()
        db_verify.close()
