import os
from types import SimpleNamespace

from fastapi.testclient import TestClient

os.environ["DATABASE_URL"] = "sqlite:///./test.db"
os.environ["ORCHESTRATOR_ENABLED"] = "false"
os.environ["JOB_RECOVERY_ENABLED"] = "false"
os.environ["POLICY_GATE_ENABLED"] = "false"

from services.api.app import chat_service, main  # noqa: E402
from services.api.app.database import Base, engine  # noqa: E402


Base.metadata.create_all(bind=engine)

client = TestClient(main.app)


def test_create_chat_session() -> None:
    response = client.post("/chat/sessions", json={"title": "Delivery chat"})

    assert response.status_code == 200
    body = response.json()
    assert body["title"] == "Delivery chat"
    assert body["messages"] == []
    assert body["active_job_id"] is None


def test_chat_turn_can_respond_without_creating_job() -> None:
    session = client.post("/chat/sessions", json={}).json()

    response = client.post(
        f"/chat/sessions/{session['id']}/messages",
        json={"content": "help", "context_json": {}, "priority": 0},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["job"] is None
    assert body["assistant_message"]["action"]["type"] == "respond"
    assert "workflow" in body["assistant_message"]["content"].lower()
    assert "pending_clarification" not in body["session"]["metadata"]


def test_chat_turn_can_request_clarification_for_ambiguous_workflow() -> None:
    session = client.post("/chat/sessions", json={}).json()

    response = client.post(
        f"/chat/sessions/{session['id']}/messages",
        json={"content": "Generate a deployment report", "context_json": {}, "priority": 0},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["job"] is None
    assert body["assistant_message"]["action"]["type"] == "ask_clarification"
    assert body["assistant_message"]["action"]["clarification_questions"]
    assert body["session"]["metadata"]["pending_clarification"]["questions"]


def test_chat_turn_can_submit_job() -> None:
    session = client.post("/chat/sessions", json={}).json()

    response = client.post(
        f"/chat/sessions/{session['id']}/messages",
        json={
            "content": "Render a PDF deployment report",
            "context_json": {"topic": "Deployment report"},
            "priority": 1,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["job"]["goal"] == "Render a PDF deployment report"
    assert body["assistant_message"]["action"]["type"] == "submit_job"
    assert body["assistant_message"]["action"]["job_id"] == body["job"]["id"]
    assert body["session"]["active_job_id"] == body["job"]["id"]
    assert len(body["session"]["messages"]) == 2


def test_chat_turn_uses_pending_clarification_context_to_submit_job() -> None:
    session = client.post("/chat/sessions", json={}).json()
    client.post(
        f"/chat/sessions/{session['id']}/messages",
        json={"content": "Generate a deployment report", "context_json": {}, "priority": 0},
    )

    response = client.post(
        f"/chat/sessions/{session['id']}/messages",
        json={"content": "PDF for leadership, dry run only", "context_json": {}, "priority": 0},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["job"] is not None
    assert body["assistant_message"]["action"]["type"] == "submit_job"
    assert "User clarification:" in body["job"]["goal"]


def test_chat_turn_can_use_llm_router_for_conversational_reply(monkeypatch) -> None:
    class _Router:
        def generate_request_json_object(self, request):
            assert "current_user_message" in request.prompt
            return {
                "route": "respond",
                "assistant_response": "This stays in chat.",
                "intent": "other",
                "risk_level": "read_only",
                "confidence": 0.93,
            }

    monkeypatch.setattr(main, "_chat_router_provider", _Router())
    session = client.post("/chat/sessions", json={}).json()

    response = client.post(
        f"/chat/sessions/{session['id']}/messages",
        json={"content": "Could you explain the architecture?", "context_json": {}, "priority": 0},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["job"] is None
    assert body["assistant_message"]["action"]["type"] == "respond"
    assert body["assistant_message"]["content"] == "This stays in chat."


def test_chat_turn_can_execute_direct_capability(monkeypatch) -> None:
    class _Router:
        def generate_request_json_object(self, request):
            assert "direct_capabilities" in request.prompt
            return {
                "route": "tool_call",
                "assistant_response": "",
                "intent": "io",
                "risk_level": "read_only",
                "confidence": 0.97,
                "capability_id": "filesystem.workspace.list",
                "arguments": {"path": "", "max_files": 2},
            }

    class _DirectExecutor:
        def execute_capability(self, *, capability_id: str, arguments: dict[str, object], trace_id: str):
            assert capability_id == "filesystem.workspace.list"
            assert arguments["max_files"] == 2
            assert trace_id
            return SimpleNamespace(
                capability_id=capability_id,
                tool_name="workspace_list_files",
                output={"entries": [{"path": "/shared/workspace/README.md", "type": "file"}]},
                assistant_response="Entries:\n- /shared/workspace/README.md",
            )

    monkeypatch.setattr(main, "_chat_router_provider", _Router())
    monkeypatch.setattr(main, "_chat_direct_executor", _DirectExecutor())
    session = client.post("/chat/sessions", json={}).json()

    response = client.post(
        f"/chat/sessions/{session['id']}/messages",
        json={"content": "List the workspace files", "context_json": {}, "priority": 0},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["job"] is None
    assert body["assistant_message"]["action"]["type"] == "tool_call"
    assert body["assistant_message"]["action"]["capability_id"] == "filesystem.workspace.list"
    assert "README.md" in body["assistant_message"]["content"]
    assert body["assistant_message"]["metadata"]["tool_output"]["entries"][0]["path"].endswith(
        "README.md"
    )


def test_chat_memory_arguments_inherit_user_id_from_context() -> None:
    enriched = chat_service._enrich_memory_arguments(
        "memory.read",
        {"name": "user_profile", "key": "profile"},
        {"user_id": "narendersurabhi"},
    )

    assert enriched == {
        "name": "user_profile",
        "key": "profile",
        "user_id": "narendersurabhi",
        "scope": "user",
    }
