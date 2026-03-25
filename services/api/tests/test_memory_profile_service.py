import os

os.environ["DATABASE_URL"] = "sqlite:///./test.db"
os.environ["ORCHESTRATOR_ENABLED"] = "false"
os.environ["JOB_RECOVERY_ENABLED"] = "false"
os.environ["POLICY_GATE_ENABLED"] = "false"
os.environ["CAPABILITY_MODE"] = "enabled"

from services.api.app.database import Base, SessionLocal, engine  # noqa: E402
from services.api.app import memory_profile_service  # noqa: E402


Base.metadata.create_all(bind=engine)


def test_apply_user_profile_updates_from_text_persists_preferences() -> None:
    db = SessionLocal()
    try:
        entry, decisions = memory_profile_service.apply_user_profile_updates_from_text(
            db,
            user_id="user-profile-test",
            content="I prefer markdown and concise responses.",
        )
        assert entry is not None
        assert len(decisions) == 2
        assert entry.payload["preferences"]["preferred_output_format"] == "markdown"
        assert entry.payload["preferences"]["response_verbosity"] == "concise"
    finally:
        db.close()
