import os

from fastapi.testclient import TestClient

os.environ["DATABASE_URL"] = "sqlite:///./test.db"
os.environ["ORCHESTRATOR_ENABLED"] = "false"
os.environ["JOB_RECOVERY_ENABLED"] = "false"
os.environ["POLICY_GATE_ENABLED"] = "false"
os.environ["CAPABILITY_MODE"] = "enabled"

from services.api.app import main  # noqa: E402
from services.api.app.database import Base, engine  # noqa: E402


Base.metadata.create_all(bind=engine)

client = TestClient(main.app)


def test_semantic_memory_write_redacts_sensitive_content() -> None:
    response = client.post(
        "/memory/semantic/write",
        json={
            "user_id": "semantic-redaction-test",
            "subject": "contact",
            "namespace": "preferences",
            "fact": "Email me at narender@example.com and use sk-abcdefghijklmnopqrstuvwxyz.",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["semantic_record"]["fact"].count("[redacted]") == 2
    assert body["entry"]["metadata"]["sensitive"] is True
    assert body["entry"]["metadata"]["indexable"] is False
