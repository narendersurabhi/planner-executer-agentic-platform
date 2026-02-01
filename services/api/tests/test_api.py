import os
from datetime import datetime

import sqlalchemy
from fastapi.testclient import TestClient

os.environ["DATABASE_URL"] = "sqlite:///./test.db"

from services.api.app import main  # noqa: E402
from services.api.app.database import Base, engine


Base.metadata.create_all(bind=engine)

client = TestClient(main.app)


def test_create_job():
    response = client.post(
        "/jobs",
        json={"goal": "demo", "context_json": {}, "priority": 1},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["goal"] == "demo"


def test_event_stream():
    response = client.get("/events/stream?once=true")
    assert response.status_code == 200
