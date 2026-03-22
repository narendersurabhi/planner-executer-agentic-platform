import os
from datetime import UTC, datetime

from fastapi.testclient import TestClient

os.environ["DATABASE_URL"] = "sqlite:///./test.db"
os.environ["ORCHESTRATOR_ENABLED"] = "false"
os.environ["JOB_RECOVERY_ENABLED"] = "false"
os.environ["POLICY_GATE_ENABLED"] = "false"
os.environ["CAPABILITY_MODE"] = "enabled"

from libs.core import models  # noqa: E402
from libs.core import capability_registry as cap_registry  # noqa: E402
from services.api.app import chat_service, main  # noqa: E402
from services.api.app.database import Base, engine  # noqa: E402


Base.metadata.create_all(bind=engine)

client = TestClient(main.app)
main.CHAT_RESPONSE_MODE = "answer_only"
main.CHAT_ROUTING_MODE = "response_first"
main._chat_router_provider = None
main._chat_response_provider = None
main._chat_pending_correction_provider = None


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


def test_chat_turn_lists_available_capabilities_when_asked(monkeypatch) -> None:
    capability = cap_registry.CapabilitySpec(
        capability_id="github.repo.list",
        description="List repositories for a user or organization.",
        risk_tier="read_only",
        idempotency="read",
        group="github",
        subgroup="repositories",
        enabled=True,
    )
    registry = cap_registry.CapabilityRegistry(capabilities={"github.repo.list": capability})
    monkeypatch.setattr(main.capability_registry, "load_capability_registry", lambda: registry)

    session = client.post("/chat/sessions", json={}).json()

    response = client.post(
        f"/chat/sessions/{session['id']}/messages",
        json={"content": "What capabilities do you have?", "context_json": {}, "priority": 0},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["job"] is None
    assert body["assistant_message"]["action"]["type"] == "respond"
    assert "Available capabilities for this assistant (1):" in body["assistant_message"]["content"]
    assert "github.repo.list" in body["assistant_message"]["content"]
    assert "List repositories for a user or organization." in body["assistant_message"]["content"]


def test_chat_turn_lists_only_allowlisted_capabilities_for_assistant(monkeypatch) -> None:
    allowed = cap_registry.CapabilitySpec(
        capability_id="github.repo.list",
        description="List repositories for a user or organization.",
        risk_tier="read_only",
        idempotency="read",
        group="github",
        subgroup="repositories",
        enabled=True,
    )
    blocked = cap_registry.CapabilitySpec(
        capability_id="github.branch.create",
        description="Create a branch in a repository.",
        risk_tier="bounded_write",
        idempotency="write",
        group="github",
        subgroup="branches",
        enabled=True,
    )
    registry = cap_registry.CapabilityRegistry(
        capabilities={
            "github.repo.list": allowed,
            "github.branch.create": blocked,
        }
    )
    monkeypatch.setattr(main.capability_registry, "load_capability_registry", lambda: registry)
    monkeypatch.setattr(main.capability_registry, "resolve_capability_mode", lambda: "enabled")
    monkeypatch.setattr(
        main.capability_registry,
        "evaluate_capability_allowlist",
        lambda capability_id, service_name=None: cap_registry.CapabilityAllowDecision(
            allowed=capability_id != "github.branch.create",
            reason="allowed" if capability_id != "github.branch.create" else "service_disabled",
        ),
    )

    session = client.post("/chat/sessions", json={}).json()
    response = client.post(
        f"/chat/sessions/{session['id']}/messages",
        json={"content": "What tools are available here?", "context_json": {}, "priority": 0},
    )

    assert response.status_code == 200
    body = response.json()
    assert "github.repo.list" in body["assistant_message"]["content"]
    assert "github.branch.create" not in body["assistant_message"]["content"]


def test_chat_turn_lists_only_scoped_capabilities_when_query_mentions_github(monkeypatch) -> None:
    github_capability = cap_registry.CapabilitySpec(
        capability_id="github.repo.list",
        description="List repositories for a user or organization.",
        risk_tier="read_only",
        idempotency="read",
        group="github",
        subgroup="repositories",
        enabled=True,
    )
    filesystem_capability = cap_registry.CapabilitySpec(
        capability_id="filesystem.workspace.list",
        description="List files under workspace storage.",
        risk_tier="read_only",
        idempotency="read",
        group="filesystem",
        subgroup="workspace",
        enabled=True,
    )
    registry = cap_registry.CapabilityRegistry(
        capabilities={
            "github.repo.list": github_capability,
            "filesystem.workspace.list": filesystem_capability,
        }
    )
    monkeypatch.setattr(main.capability_registry, "load_capability_registry", lambda: registry)
    monkeypatch.setattr(main.capability_registry, "resolve_capability_mode", lambda: "enabled")

    session = client.post("/chat/sessions", json={}).json()
    response = client.post(
        f"/chat/sessions/{session['id']}/messages",
        json={"content": "What can you do related to GitHub?", "context_json": {}, "priority": 0},
    )

    assert response.status_code == 200
    body = response.json()
    assert "Available capabilities related to 'github'" in body["assistant_message"]["content"]
    assert "github.repo.list" in body["assistant_message"]["content"]
    assert "filesystem.workspace.list" not in body["assistant_message"]["content"]


def test_chat_turn_uses_vector_capability_search_for_fuzzy_scope_queries(monkeypatch) -> None:
    github_capability = cap_registry.CapabilitySpec(
        capability_id="github.file.create_or_update",
        description="Create or update a single file in a repository.",
        risk_tier="bounded_write",
        idempotency="safe_write",
        group="github",
        subgroup="files",
        enabled=True,
    )
    filesystem_capability = cap_registry.CapabilitySpec(
        capability_id="filesystem.workspace.list",
        description="List files under workspace storage.",
        risk_tier="read_only",
        idempotency="read",
        group="filesystem",
        subgroup="workspace",
        enabled=True,
    )
    registry = cap_registry.CapabilityRegistry(
        capabilities={
            "github.file.create_or_update": github_capability,
            "filesystem.workspace.list": filesystem_capability,
        }
    )

    def _rag_request(path, *, method="GET", body=None, query=None, timeout_s=20.0):
        del method, query, timeout_s
        if path == "/index/upsert_texts":
            return {"collection_name": "test", "upserted_count": 2, "chunk_ids": ["a", "b"]}
        if path == "/retrieve":
            assert body is not None
            assert body["query"] == "repo changes"
            return {
                "matches": [
                    {
                        "document_id": "github.file.create_or_update",
                        "score": 0.82,
                        "metadata": {"capability_id": "github.file.create_or_update"},
                    }
                ]
            }
        raise AssertionError(f"unexpected RAG path: {path}")

    monkeypatch.setattr(main.capability_registry, "load_capability_registry", lambda: registry)
    monkeypatch.setattr(main.capability_registry, "resolve_capability_mode", lambda: "enabled")
    monkeypatch.setattr(main.capability_search, "search_capabilities", lambda **kwargs: [])
    monkeypatch.setattr(main, "_rag_retriever_request_json", _rag_request)
    monkeypatch.setattr(main, "_chat_capability_vector_synced_namespace", None)

    session = client.post("/chat/sessions", json={}).json()
    response = client.post(
        f"/chat/sessions/{session['id']}/messages",
        json={"content": "What can you do related to repo changes?", "context_json": {}, "priority": 0},
    )

    assert response.status_code == 200
    body = response.json()
    assert "Available capabilities related to 'repo changes'" in body["assistant_message"]["content"]
    assert "github.file.create_or_update" in body["assistant_message"]["content"]
    assert "filesystem.workspace.list" not in body["assistant_message"]["content"]


def test_chat_turn_uses_vector_intent_detection_for_fuzzy_capability_question(monkeypatch) -> None:
    capability = cap_registry.CapabilitySpec(
        capability_id="github.repo.list",
        description="List repositories for a user or organization.",
        risk_tier="read_only",
        idempotency="read",
        group="github",
        subgroup="repositories",
        enabled=True,
    )
    registry = cap_registry.CapabilityRegistry(capabilities={"github.repo.list": capability})

    def _rag_request(path, *, method="GET", body=None, query=None, timeout_s=20.0):
        del method, query, timeout_s
        if path == "/index/upsert_texts":
            return {"collection_name": "test", "upserted_count": 3, "chunk_ids": ["a", "b", "c"]}
        if path == "/retrieve":
            assert body is not None
            assert body["query"] == "What kinds of work can you handle here?"
            return {
                "matches": [
                    {
                        "document_id": "capability_discovery",
                        "score": 0.84,
                        "metadata": {"intent_id": "capability_discovery"},
                    },
                    {
                        "document_id": "general_chat",
                        "score": 0.51,
                        "metadata": {"intent_id": "general_chat"},
                    },
                ]
            }
        raise AssertionError(f"unexpected RAG path: {path}")

    monkeypatch.setattr(main.capability_registry, "load_capability_registry", lambda: registry)
    monkeypatch.setattr(main.capability_registry, "resolve_capability_mode", lambda: "enabled")
    monkeypatch.setattr(main, "_rag_retriever_request_json", _rag_request)
    monkeypatch.setattr(main, "_chat_intent_vector_synced_namespace", None)
    monkeypatch.setattr(main, "CHAT_CAPABILITY_VECTOR_SEARCH_ENABLED", False)

    session = client.post("/chat/sessions", json={}).json()
    response = client.post(
        f"/chat/sessions/{session['id']}/messages",
        json={"content": "What kinds of work can you handle here?", "context_json": {}, "priority": 0},
    )

    assert response.status_code == 200
    body = response.json()
    assert "Available capabilities for this assistant (1):" in body["assistant_message"]["content"]
    assert "github.repo.list" in body["assistant_message"]["content"]


def test_chat_turn_uses_vector_intent_detection_for_fuzzy_github_scope(monkeypatch) -> None:
    github_capability = cap_registry.CapabilitySpec(
        capability_id="github.repo.list",
        description="List repositories for a user or organization.",
        risk_tier="read_only",
        idempotency="read",
        group="github",
        subgroup="repositories",
        enabled=True,
    )
    filesystem_capability = cap_registry.CapabilitySpec(
        capability_id="filesystem.workspace.list",
        description="List files under workspace storage.",
        risk_tier="read_only",
        idempotency="read",
        group="filesystem",
        subgroup="workspace",
        enabled=True,
    )
    registry = cap_registry.CapabilityRegistry(
        capabilities={
            "github.repo.list": github_capability,
            "filesystem.workspace.list": filesystem_capability,
        }
    )

    def _rag_request(path, *, method="GET", body=None, query=None, timeout_s=20.0):
        del method, query, timeout_s
        if path == "/index/upsert_texts":
            return {"collection_name": "test", "upserted_count": 3, "chunk_ids": ["a", "b", "c"]}
        if path == "/retrieve":
            assert body is not None
            assert body["query"] == "What kinds of GitHub work can you handle here?"
            return {
                "matches": [
                    {
                        "document_id": "capability_discovery",
                        "score": 0.86,
                        "metadata": {"intent_id": "capability_discovery"},
                    },
                    {
                        "document_id": "workflow_execution",
                        "score": 0.47,
                        "metadata": {"intent_id": "workflow_execution"},
                    },
                ]
            }
        raise AssertionError(f"unexpected RAG path: {path}")

    monkeypatch.setattr(main.capability_registry, "load_capability_registry", lambda: registry)
    monkeypatch.setattr(main.capability_registry, "resolve_capability_mode", lambda: "enabled")
    monkeypatch.setattr(main, "_rag_retriever_request_json", _rag_request)
    monkeypatch.setattr(main, "_chat_intent_vector_synced_namespace", None)
    monkeypatch.setattr(main, "CHAT_CAPABILITY_VECTOR_SEARCH_ENABLED", False)

    session = client.post("/chat/sessions", json={}).json()
    response = client.post(
        f"/chat/sessions/{session['id']}/messages",
        json={
            "content": "What kinds of GitHub work can you handle here?",
            "context_json": {},
            "priority": 0,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert "Available capabilities related to 'github'" in body["assistant_message"]["content"]
    assert "github.repo.list" in body["assistant_message"]["content"]
    assert "filesystem.workspace.list" not in body["assistant_message"]["content"]


def test_chat_turn_capability_discovery_overrides_llm_chat_reply(monkeypatch) -> None:
    capability = cap_registry.CapabilitySpec(
        capability_id="github.repo.list",
        description="List repositories for a user or organization.",
        risk_tier="read_only",
        idempotency="read",
        group="github",
        subgroup="repositories",
        enabled=True,
    )
    registry = cap_registry.CapabilityRegistry(capabilities={"github.repo.list": capability})

    class _BoundaryResponder:
        def generate_request_json_object(self, request):
            assert request.metadata == {"component": "chat_boundary_decision"}
            return {"decision": "chat_reply", "assistant_response": "Generic chat reply."}

        def generate_request(self, request):
            raise AssertionError("chat generation should not run for deterministic capability discovery")

    monkeypatch.setattr(main.capability_registry, "load_capability_registry", lambda: registry)
    monkeypatch.setattr(main.capability_registry, "resolve_capability_mode", lambda: "enabled")
    monkeypatch.setattr(main, "CHAT_RESPONSE_MODE", "answer_or_handoff")
    monkeypatch.setattr(main, "_chat_response_provider", _BoundaryResponder())
    monkeypatch.setattr(
        main,
        "_chat_router_provider",
        type(
            "_Router",
            (),
            {
                "generate_request_json_object": lambda self, request: (_ for _ in ()).throw(
                    AssertionError("router should not run for chat replies")
                )
            },
        )(),
    )

    session = client.post("/chat/sessions", json={}).json()
    response = client.post(
        f"/chat/sessions/{session['id']}/messages",
        json={"content": "What capabilities do you have?", "context_json": {}, "priority": 0},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["assistant_message"]["action"]["type"] == "respond"
    assert "Available capabilities for this assistant (1):" in body["assistant_message"]["content"]
    assert "Generic chat reply." not in body["assistant_message"]["content"]


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


def test_chat_turn_can_run_published_workflow_by_version_reference() -> None:
    create_response = client.post(
        "/workflows/definitions",
        json={
            "title": "Chat-started workspace listing",
            "goal": "List workspace files",
            "user_id": "narendersurabhi",
            "context_json": {"user_id": "narendersurabhi"},
            "draft": {
                "summary": "Chat-started workspace listing",
                "nodes": [
                    {
                        "id": "n1",
                        "taskName": "ListWorkspace",
                        "capabilityId": "filesystem.workspace.list",
                        "bindings": {},
                    }
                ],
                "edges": [],
            },
        },
    )
    assert create_response.status_code == 200
    definition = create_response.json()

    publish_response = client.post(f"/workflows/definitions/{definition['id']}/publish", json={})
    assert publish_response.status_code == 200
    version = publish_response.json()

    session = client.post("/chat/sessions", json={}).json()
    response = client.post(
        f"/chat/sessions/{session['id']}/messages",
        json={
            "content": "Run the published workflow now",
            "context_json": {
                "workflow_version_id": version["id"],
                "topic": "Release readiness",
            },
            "priority": 2,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["assistant_message"]["action"]["type"] == "run_workflow"
    assert body["assistant_message"]["action"]["workflow_version_id"] == version["id"]
    assert body["job"]["metadata"]["workflow_source"] == "studio"
    assert body["workflow_run"]["version_id"] == version["id"]
    assert body["workflow_run"]["requested_context_json"]["topic"] == "Release readiness"
    assert "workflow_version_id" not in body["workflow_run"]["requested_context_json"]
    assert body["session"]["active_job_id"] == body["job"]["id"]
    assert body["session"]["metadata"]["active_workflow_run_id"] == body["workflow_run"]["id"]
    assert body["assistant_message"]["metadata"]["workflow_run"]["id"] == body["workflow_run"]["id"]


def test_chat_started_workflow_posts_terminal_output_back_into_session(monkeypatch) -> None:
    create_response = client.post(
        "/workflows/definitions",
        json={
            "title": "Chat workflow reply",
            "goal": "Return generated text to chat",
            "user_id": "narendersurabhi",
            "context_json": {"user_id": "narendersurabhi"},
            "draft": {
                "summary": "Chat workflow reply",
                "workflowInterface": {
                    "inputs": [],
                    "variables": [],
                    "outputs": [
                        {
                            "key": "reply",
                            "label": "Reply",
                            "binding": {
                                "kind": "step_output",
                                "sourceNodeId": "n1",
                                "sourcePath": "result.text",
                            },
                        }
                    ],
                },
                "nodes": [
                    {
                        "id": "n1",
                        "taskName": "GenerateReply",
                        "capabilityId": "filesystem.workspace.list",
                        "bindings": {},
                    }
                ],
                "edges": [],
            },
        },
    )
    assert create_response.status_code == 200
    definition = create_response.json()

    publish_response = client.post(f"/workflows/definitions/{definition['id']}/publish", json={})
    assert publish_response.status_code == 200
    version = publish_response.json()

    session = client.post("/chat/sessions", json={}).json()
    response = client.post(
        f"/chat/sessions/{session['id']}/messages",
        json={
            "content": "Run the chat reply workflow",
            "context_json": {"workflow_version_id": version["id"]},
            "priority": 1,
        },
    )

    assert response.status_code == 200
    body = response.json()
    job_id = body["job"]["id"]

    tasks_response = client.get(f"/jobs/{job_id}/tasks")
    assert tasks_response.status_code == 200
    task_id = tasks_response.json()[0]["id"]
    monkeypatch.setattr(
        main,
        "_load_task_output",
        lambda current_task_id: (
            {"result": {"text": "Workflow generated answer."}} if current_task_id == task_id else {}
        ),
    )

    main._handle_task_completed(
        {
            "task_id": task_id,
            "payload": {
                "task_id": task_id,
                "outputs": {"result": {"text": "Workflow generated answer."}},
            },
        }
    )

    session_response = client.get(f"/chat/sessions/{session['id']}")
    assert session_response.status_code == 200
    session_body = session_response.json()
    assert session_body["active_job_id"] is None
    assert len(session_body["messages"]) == 3
    assert session_body["messages"][-1]["role"] == "assistant"
    assert session_body["messages"][-1]["content"] == "Workflow generated answer."
    assert session_body["messages"][-1]["job_id"] == job_id


def test_chat_turn_asks_for_missing_workflow_input_and_uses_followup_reply() -> None:
    create_response = client.post(
        "/workflows/definitions",
        json={
            "title": "Clarified workflow",
            "goal": "List a workspace subdirectory",
            "user_id": "narendersurabhi",
            "context_json": {"user_id": "narendersurabhi"},
            "draft": {
                "summary": "Clarified workflow",
                "workflowInterface": {
                    "inputs": [
                        {
                            "key": "path",
                            "label": "Workspace Path",
                            "valueType": "string",
                            "required": True,
                            "description": "Relative workspace path to inspect.",
                        }
                    ],
                    "variables": [],
                    "outputs": [],
                },
                "nodes": [
                    {
                        "id": "n1",
                        "taskName": "ListWorkspace",
                        "capabilityId": "filesystem.workspace.list",
                        "bindings": {
                            "path": {"kind": "workflow_input", "inputKey": "path"}
                        },
                    }
                ],
                "edges": [],
            },
        },
    )
    assert create_response.status_code == 200
    definition = create_response.json()

    publish_response = client.post(f"/workflows/definitions/{definition['id']}/publish", json={})
    assert publish_response.status_code == 200
    version = publish_response.json()

    session = client.post("/chat/sessions", json={}).json()
    clarification_response = client.post(
        f"/chat/sessions/{session['id']}/messages",
        json={
            "content": "Run the clarified workflow",
            "context_json": {"workflow_version_id": version["id"]},
            "priority": 1,
        },
    )

    assert clarification_response.status_code == 200
    clarification_body = clarification_response.json()
    assert clarification_body["job"] is None
    assert clarification_body["workflow_run"] is None
    assert clarification_body["assistant_message"]["action"]["type"] == "ask_clarification"
    assert "path" in clarification_body["assistant_message"]["content"].lower()
    assert clarification_body["session"]["metadata"]["pending_workflow_input"]["key"] == "path"

    run_response = client.post(
        f"/chat/sessions/{session['id']}/messages",
        json={"content": "reports", "context_json": {}, "priority": 1},
    )

    assert run_response.status_code == 200
    run_body = run_response.json()
    assert run_body["assistant_message"]["action"]["type"] == "run_workflow"
    assert run_body["workflow_run"]["version_id"] == version["id"]
    assert run_body["job"]["context_json"]["workflow"]["inputs"]["path"] == "reports"
    assert run_body["session"]["metadata"]["active_workflow_run_id"] == run_body["workflow_run"]["id"]
    assert "pending_workflow_input" not in run_body["session"]["metadata"]


def test_chat_turn_can_skip_optional_workflow_input_with_use_default_reply() -> None:
    create_response = client.post(
        "/workflows/definitions",
        json={
            "title": "Optional recursive listing",
            "goal": "List workspace files with optional recursion",
            "user_id": "narendersurabhi",
            "context_json": {"user_id": "narendersurabhi"},
            "draft": {
                "summary": "Optional recursive listing",
                "workflowInterface": {
                    "inputs": [
                        {
                            "key": "recursive",
                            "label": "Recursive",
                            "valueType": "boolean",
                            "required": False,
                            "description": "Whether to recurse into subdirectories.",
                        }
                    ],
                    "variables": [],
                    "outputs": [],
                },
                "nodes": [
                    {
                        "id": "n1",
                        "taskName": "ListWorkspace",
                        "capabilityId": "filesystem.workspace.list",
                        "bindings": {
                            "recursive": {"kind": "workflow_input", "inputKey": "recursive"}
                        },
                    }
                ],
                "edges": [],
            },
        },
    )
    assert create_response.status_code == 200
    definition = create_response.json()

    publish_response = client.post(f"/workflows/definitions/{definition['id']}/publish", json={})
    assert publish_response.status_code == 200
    version = publish_response.json()

    session = client.post("/chat/sessions", json={}).json()
    clarification_response = client.post(
        f"/chat/sessions/{session['id']}/messages",
        json={
            "content": "Run the optional workflow",
            "context_json": {
                "workflow_version_id": version["id"],
                "workflow_inputs": {"recursive": "sometimes"},
            },
            "priority": 1,
        },
    )

    assert clarification_response.status_code == 200
    clarification_body = clarification_response.json()
    assert clarification_body["job"] is None
    assert clarification_body["assistant_message"]["action"]["type"] == "ask_clarification"
    assert "use default" in clarification_body["assistant_message"]["content"].lower()
    assert clarification_body["session"]["metadata"]["pending_workflow_input"]["key"] == "recursive"

    run_response = client.post(
        f"/chat/sessions/{session['id']}/messages",
        json={"content": "use default", "context_json": {}, "priority": 1},
    )

    assert run_response.status_code == 200
    run_body = run_response.json()
    assert run_body["assistant_message"]["action"]["type"] == "run_workflow"
    assert run_body["workflow_run"]["version_id"] == version["id"]
    assert run_body["job"]["context_json"]["workflow"]["inputs"]["recursive"] is None
    assert "pending_workflow_input" not in run_body["session"]["metadata"]


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


def test_chat_turn_can_run_published_workflow_by_trigger_reference() -> None:
    create_response = client.post(
        "/workflows/definitions",
        json={
            "title": "Chat-triggered workspace listing",
            "goal": "List workspace files",
            "user_id": "narendersurabhi",
            "context_json": {"user_id": "narendersurabhi"},
            "draft": {
                "summary": "Chat-triggered workspace listing",
                "nodes": [
                    {
                        "id": "n1",
                        "taskName": "ListWorkspace",
                        "capabilityId": "filesystem.workspace.list",
                        "bindings": {},
                    }
                ],
                "edges": [],
            },
        },
    )
    assert create_response.status_code == 200
    definition = create_response.json()

    publish_response = client.post(f"/workflows/definitions/{definition['id']}/publish", json={})
    assert publish_response.status_code == 200
    version = publish_response.json()

    trigger_response = client.post(
        f"/workflows/definitions/{definition['id']}/triggers",
        json={
            "title": "Manual trigger",
            "trigger_type": "manual",
            "enabled": True,
            "config": {"version_mode": "latest_published"},
        },
    )
    assert trigger_response.status_code == 200
    trigger = trigger_response.json()

    session = client.post("/chat/sessions", json={}).json()
    response = client.post(
        f"/chat/sessions/{session['id']}/messages",
        json={
            "content": "Kick off the saved workflow",
            "context_json": {
                "workflow_trigger_id": trigger["id"],
                "workflow_context_json": {"channel": "chat"},
            },
            "priority": 1,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["assistant_message"]["action"]["type"] == "run_workflow"
    assert body["assistant_message"]["action"]["workflow_trigger_id"] == trigger["id"]
    assert body["workflow_run"]["trigger_id"] == trigger["id"]
    assert body["workflow_run"]["version_id"] == version["id"]
    assert body["workflow_run"]["requested_context_json"]["channel"] == "chat"
    assert body["job"]["metadata"]["workflow_trigger_id"] == trigger["id"]


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

    monkeypatch.setattr(main, "CHAT_ROUTING_MODE", "always_router")
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


def test_chat_turn_uses_separate_response_provider_for_conversational_reply(monkeypatch) -> None:
    class _Router:
        def generate_request_json_object(self, request):
            assert "current_user_message" in request.prompt
            return {
                "route": "respond",
                "assistant_response": "Router fallback response.",
                "intent": "other",
                "risk_level": "read_only",
                "confidence": 0.93,
            }

    class _Responder:
        def generate_request(self, request):
            assert request.metadata == {"component": "chat_response"}
            assert "current_user_message" in request.prompt
            return type("_Response", (), {"content": "Response model answer."})()

    monkeypatch.setattr(main, "CHAT_ROUTING_MODE", "response_first")
    monkeypatch.setattr(main, "CHAT_RESPONSE_MODE", "answer_only")
    monkeypatch.setattr(main, "_chat_router_provider", _Router())
    monkeypatch.setattr(main, "_chat_response_provider", _Responder())
    session = client.post("/chat/sessions", json={}).json()

    response = client.post(
        f"/chat/sessions/{session['id']}/messages",
        json={"content": "Explain the planner at a high level.", "context_json": {}, "priority": 0},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["job"] is None
    assert body["assistant_message"]["action"]["type"] == "respond"
    assert body["assistant_message"]["content"] == "Response model answer."


def test_chat_turn_response_first_skips_router_for_conversational_turn(monkeypatch) -> None:
    class _Router:
        def generate_request_json_object(self, request):
            raise AssertionError("router should not run for conversational turns in response_first mode")

    class _PendingCorrectionProvider:
        def generate_request_json_object(self, request):
            raise AssertionError("pending correction classifier should not run for normal conversational turns")

    class _Responder:
        def generate_request(self, request):
            assert request.metadata == {"component": "chat_response"}
            return type("_Response", (), {"content": "Direct response model answer."})()

    monkeypatch.setattr(main, "CHAT_ROUTING_MODE", "response_first")
    monkeypatch.setattr(main, "CHAT_RESPONSE_MODE", "answer_only")
    monkeypatch.setattr(
        main,
        "_normalize_goal_intent",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("intent normalization should not run for conversational fast path")
        ),
    )
    monkeypatch.setattr(main, "_chat_router_provider", _Router())
    monkeypatch.setattr(main, "_chat_pending_correction_provider", _PendingCorrectionProvider())
    monkeypatch.setattr(main, "_chat_response_provider", _Responder())
    session = client.post("/chat/sessions", json={}).json()

    response = client.post(
        f"/chat/sessions/{session['id']}/messages",
        json={"content": "Can you explain the architecture?", "context_json": {}, "priority": 0},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["job"] is None
    assert body["assistant_message"]["action"]["type"] == "respond"
    assert body["assistant_message"]["content"] == "Direct response model answer."


def test_chat_turn_response_first_treats_discussion_request_as_conversational(monkeypatch) -> None:
    class _Router:
        def generate_request_json_object(self, request):
            raise AssertionError("router should not run for discussion-style conversational turns")

    class _PendingCorrectionProvider:
        def generate_request_json_object(self, request):
            raise AssertionError("pending correction classifier should not run for normal discussion turns")

    class _Responder:
        def generate_request(self, request):
            assert request.metadata == {"component": "chat_response"}
            assert "discuss about kubernetes" in request.prompt.lower()
            return type("_Response", (), {"content": "Let's discuss Kubernetes."})()

    monkeypatch.setattr(main, "CHAT_ROUTING_MODE", "response_first")
    monkeypatch.setattr(main, "CHAT_RESPONSE_MODE", "answer_only")
    monkeypatch.setattr(
        main,
        "_normalize_goal_intent",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("intent normalization should not run for discussion-style conversational turns")
        ),
    )
    monkeypatch.setattr(main, "_chat_router_provider", _Router())
    monkeypatch.setattr(main, "_chat_pending_correction_provider", _PendingCorrectionProvider())
    monkeypatch.setattr(main, "_chat_response_provider", _Responder())
    session = client.post("/chat/sessions", json={}).json()

    response = client.post(
        f"/chat/sessions/{session['id']}/messages",
        json={"content": "I want to discuss about Kubernetes with you.", "context_json": {}, "priority": 0},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["job"] is None
    assert body["assistant_message"]["action"]["type"] == "respond"
    assert body["assistant_message"]["content"] == "Let's discuss Kubernetes."


def test_chat_turn_response_first_answer_or_handoff_answers_without_router(monkeypatch) -> None:
    class _Router:
        def generate_request_json_object(self, request):
            raise AssertionError("router should not run when response model answers directly")

    class _Responder:
        def generate_request_json_object(self, request):
            assert request.metadata == {"component": "chat_boundary_decision"}
            assert "current_user_message" in request.prompt
            return {
                "decision": "chat_reply",
                "assistant_response": "Direct answer-or-handoff response.",
            }

    monkeypatch.setattr(main, "CHAT_ROUTING_MODE", "response_first")
    monkeypatch.setattr(main, "CHAT_RESPONSE_MODE", "answer_or_handoff")
    monkeypatch.setattr(
        main,
        "_normalize_goal_intent",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("intent normalization should not run for answer-or-handoff direct responses")
        ),
    )
    monkeypatch.setattr(main, "_chat_router_provider", _Router())
    monkeypatch.setattr(main, "_chat_response_provider", _Responder())
    session = client.post("/chat/sessions", json={}).json()

    response = client.post(
        f"/chat/sessions/{session['id']}/messages",
        json={"content": "Can you explain the architecture?", "context_json": {}, "priority": 0},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["job"] is None
    assert body["assistant_message"]["action"]["type"] == "respond"
    assert body["assistant_message"]["content"] == "Direct answer-or-handoff response."


def test_chat_turn_response_first_answer_or_handoff_keeps_interview_practice_in_chat(
    monkeypatch,
) -> None:
    class _Router:
        def generate_request_json_object(self, request):
            raise AssertionError("router should not run for interview-practice chat turns")

    class _Responder:
        def generate_request_json_object(self, request):
            assert request.metadata == {"component": "chat_boundary_decision"}
            assert "practice interview questions and answers" in request.prompt.lower()
            return {
                "decision": "chat_reply",
                "assistant_response": "Great — let's practice Applied AI Engineer interview questions.",
            }

    monkeypatch.setattr(main, "CHAT_ROUTING_MODE", "response_first")
    monkeypatch.setattr(main, "CHAT_RESPONSE_MODE", "answer_or_handoff")
    monkeypatch.setattr(
        main,
        "_normalize_goal_intent",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("intent normalization should not run for interview-practice chat turns")
        ),
    )
    monkeypatch.setattr(main, "_chat_router_provider", _Router())
    monkeypatch.setattr(main, "_chat_response_provider", _Responder())
    session = client.post("/chat/sessions", json={}).json()

    response = client.post(
        f"/chat/sessions/{session['id']}/messages",
        json={
            "content": (
                "I want to practice interview questions and answers with you. "
                "You ask me a question, I will type the answer. Lets do it for applied AI engineer."
            ),
            "context_json": {},
            "priority": 0,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["job"] is None
    assert body["assistant_message"]["action"]["type"] == "respond"
    assert (
        body["assistant_message"]["content"]
        == "Great — let's practice Applied AI Engineer interview questions."
    )


def test_chat_turn_response_first_answer_or_handoff_can_escalate_to_router(monkeypatch) -> None:
    class _Router:
        def generate_request_json_object(self, request):
            assert request.metadata["component"] == "chat_router"
            return {
                "route": "submit_job",
                "assistant_response": "",
                "intent": "render",
                "risk_level": "bounded_write",
                "confidence": 0.95,
                "output_format": "pdf",
            }

    class _Responder:
        def generate_request_json_object(self, request):
            assert request.metadata == {"component": "chat_boundary_decision"}
            return {"decision": "execution_request", "assistant_response": ""}

    monkeypatch.setattr(main, "CHAT_ROUTING_MODE", "response_first")
    monkeypatch.setattr(main, "CHAT_RESPONSE_MODE", "answer_or_handoff")
    monkeypatch.setattr(main, "_chat_router_provider", _Router())
    monkeypatch.setattr(main, "_chat_response_provider", _Responder())
    session = client.post("/chat/sessions", json={}).json()

    response = client.post(
        f"/chat/sessions/{session['id']}/messages",
        json={"content": "Can you prepare a deployment report?", "context_json": {}, "priority": 0},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["job"] is not None
    assert body["assistant_message"]["action"]["type"] == "submit_job"


def test_chat_route_goal_intent_profile_uses_minimal_pre_submit_slots(monkeypatch) -> None:
    monkeypatch.setattr(
        main,
        "CHAT_PRE_SUBMIT_BLOCKING_SLOTS",
        {"output_format", "target_system", "safety_constraints"},
    )
    profile = main.workflow_contracts.GoalIntentProfile(
        intent="generate",
        source="test",
        confidence=0.42,
        risk_level="bounded_write",
        threshold=0.7,
        low_confidence=True,
        needs_clarification=True,
        requires_blocking_clarification=True,
        questions=["What format?", "What should the system do first?"],
        blocking_slots=["output_format", "intent_action"],
        missing_slots=["output_format", "intent_action"],
        slot_values={"output_format": "", "intent_action": "generate"},
    )

    narrowed = main._chat_route_goal_intent_profile(profile, goal="Create a document")

    assert narrowed.blocking_slots == ["output_format"]
    assert narrowed.missing_slots == ["output_format"]
    assert narrowed.questions == [
        "What output format do you need (for example PDF, DOCX, JSON, or Markdown)?"
    ]


def test_chat_turn_exits_pending_clarification_when_user_requests_chat_only_response(
    monkeypatch,
) -> None:
    session = client.post("/chat/sessions", json={}).json()

    first_response = client.post(
        f"/chat/sessions/{session['id']}/messages",
        json={"content": "Generate a deployment report", "context_json": {}, "priority": 0},
    )
    assert first_response.status_code == 200
    first_body = first_response.json()
    assert first_body["assistant_message"]["action"]["type"] == "ask_clarification"
    assert first_body["session"]["metadata"]["pending_clarification"]["questions"]

    class _Router:
        def generate_request_json_object(self, request):
            raise AssertionError("router should not run when user exits pending clarification to stay in chat")

    class _Responder:
        def generate_request_json_object(self, request):
            assert request.metadata == {"component": "chat_boundary_decision"}
            assert "don't need a document" in request.prompt.lower()
            return {
                "decision": "exit_pending_to_chat",
                "assistant_response": "Here are my thoughts on Kubernetes.",
            }

    monkeypatch.setattr(main, "CHAT_ROUTING_MODE", "response_first")
    monkeypatch.setattr(main, "CHAT_RESPONSE_MODE", "answer_or_handoff")
    monkeypatch.setattr(main, "_chat_router_provider", _Router())
    monkeypatch.setattr(main, "_chat_response_provider", _Responder())
    monkeypatch.setattr(
        main,
        "_normalize_goal_intent",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("intent normalization should not run when user switches back to chat")
        ),
    )

    response = client.post(
        f"/chat/sessions/{session['id']}/messages",
        json={
            "content": "don't need a document. Just put in the chat response",
            "context_json": {},
            "priority": 0,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["job"] is None
    assert body["assistant_message"]["action"]["type"] == "respond"
    assert body["assistant_message"]["content"] == "Here are my thoughts on Kubernetes."
    assert "pending_clarification" not in body["session"]["metadata"]
    assert "draft_goal" not in body["session"]["metadata"]


def test_chat_turn_exits_pending_clarification_for_semantic_chat_only_correction(
    monkeypatch,
) -> None:
    session = client.post("/chat/sessions", json={}).json()

    first_response = client.post(
        f"/chat/sessions/{session['id']}/messages",
        json={"content": "Generate a deployment report", "context_json": {}, "priority": 0},
    )
    assert first_response.status_code == 200

    class _Router:
        def generate_request_json_object(self, request):
            raise AssertionError("router should not run when user semantically redirects back to chat")

    class _Responder:
        def generate_request_json_object(self, request):
            assert request.metadata == {"component": "chat_boundary_decision"}
            assert "skip the workflow and answer here" in request.prompt.lower()
            return {
                "decision": "exit_pending_to_chat",
                "assistant_response": "Here is the direct chat answer.",
            }

    monkeypatch.setattr(main, "CHAT_ROUTING_MODE", "response_first")
    monkeypatch.setattr(main, "CHAT_RESPONSE_MODE", "answer_or_handoff")
    monkeypatch.setattr(main, "_chat_router_provider", _Router())
    monkeypatch.setattr(main, "_chat_response_provider", _Responder())
    monkeypatch.setattr(
        main,
        "_normalize_goal_intent",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("intent normalization should not run when semantic correction redirects to chat")
        ),
    )

    response = client.post(
        f"/chat/sessions/{session['id']}/messages",
        json={
            "content": "Skip the workflow and answer here in chat.",
            "context_json": {},
            "priority": 0,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["job"] is None
    assert body["assistant_message"]["action"]["type"] == "respond"
    assert body["assistant_message"]["content"] == "Here is the direct chat answer."
    assert "pending_clarification" not in body["session"]["metadata"]


def test_chat_turn_answer_or_handoff_classifier_failure_does_not_fallback_to_execution(
    monkeypatch,
) -> None:
    class _Router:
        def generate_request_json_object(self, request):
            raise AssertionError("router should not run when the boundary decision fails")

    monkeypatch.setattr(main, "CHAT_RESPONSE_MODE", "answer_or_handoff")
    monkeypatch.setattr(main, "_chat_router_provider", _Router())
    monkeypatch.setattr(
        main,
        "_chat_response_provider",
        type(
            "_FailingBoundaryProvider",
            (),
            {
                "generate_request_json_object": lambda self, request: (_ for _ in ()).throw(
                    RuntimeError("boundary failed")
                )
            },
        )(),
    )
    monkeypatch.setattr(
        main,
        "_normalize_goal_intent",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("intent normalization should not run when boundary decision fails")
        ),
    )
    session = client.post("/chat/sessions", json={}).json()

    response = client.post(
        f"/chat/sessions/{session['id']}/messages",
        json={"content": "Generate a deployment report", "context_json": {}, "priority": 0},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["job"] is None
    assert body["assistant_message"]["action"]["type"] == "respond"
    assert "workflow" in body["assistant_message"]["content"].lower()


def test_chat_turn_pending_boundary_failure_returns_meta_clarification(monkeypatch) -> None:
    session = client.post("/chat/sessions", json={}).json()
    first_response = client.post(
        f"/chat/sessions/{session['id']}/messages",
        json={"content": "Generate a deployment report", "context_json": {}, "priority": 0},
    )
    assert first_response.status_code == 200

    class _Router:
        def generate_request_json_object(self, request):
            raise AssertionError("router should not run when pending boundary decision fails")

    monkeypatch.setattr(main, "CHAT_RESPONSE_MODE", "answer_or_handoff")
    monkeypatch.setattr(main, "_chat_router_provider", _Router())
    monkeypatch.setattr(
        main,
        "_chat_response_provider",
        type(
            "_FailingBoundaryProvider",
            (),
            {
                "generate_request_json_object": lambda self, request: (_ for _ in ()).throw(
                    RuntimeError("boundary failed")
                )
            },
        )(),
    )
    monkeypatch.setattr(
        main,
        "_normalize_goal_intent",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError(
                "intent normalization should not run when pending boundary decision fails"
            )
        ),
    )

    response = client.post(
        f"/chat/sessions/{session['id']}/messages",
        json={"content": "PDF for leadership", "context_json": {}, "priority": 0},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["job"] is None
    assert body["assistant_message"]["action"]["type"] == "respond"
    assert "continue the current workflow request" in body["assistant_message"]["content"].lower()
    assert body["session"]["metadata"]["pending_clarification"]["questions"]


def test_build_chat_pending_correction_provider_prefers_dedicated_model(monkeypatch) -> None:
    captured: dict[str, str] = {}

    def _resolve_provider(provider_name, **kwargs):
        captured["provider_name"] = provider_name
        captured["model"] = kwargs.get("model") or ""
        return object()

    monkeypatch.setattr(main, "CHAT_PENDING_CORRECTION_MODE", "llm")
    monkeypatch.setattr(main, "LLM_PROVIDER_NAME", "openai")
    monkeypatch.setattr(main, "CHAT_PENDING_CORRECTION_MODEL", "gpt-5-nano")
    monkeypatch.setattr(main, "CHAT_RESPONSE_MODEL", "gpt-4.1-mini")
    monkeypatch.setattr(main, "LLM_MODEL_NAME", "gpt-5-mini")
    monkeypatch.setattr(main, "resolve_provider", _resolve_provider)

    provider = main._build_chat_pending_correction_provider()

    assert provider is not None
    assert captured == {"provider_name": "openai", "model": "gpt-5-nano"}


def test_chat_turn_response_first_still_uses_router_for_execution_turn(monkeypatch) -> None:
    calls = {"router": 0}

    class _Router:
        def generate_request_json_object(self, request):
            calls["router"] += 1
            return {
                "route": "submit_job",
                "assistant_response": "",
                "intent": "render",
                "risk_level": "bounded_write",
                "confidence": 0.95,
            }

    class _Responder:
        def generate_request_json_object(self, request):
            assert request.metadata == {"component": "chat_boundary_decision"}
            return {"decision": "execution_request", "assistant_response": ""}

    normalize_calls = {"count": 0}

    def _normalize_goal_intent(goal, **_kwargs):
        normalize_calls["count"] += 1
        return main.workflow_contracts.NormalizedIntentEnvelope(
            goal=goal,
            profile=main.workflow_contracts.GoalIntentProfile(
                intent="render",
                source="test",
                confidence=0.95,
                risk_level="bounded_write",
                threshold=0.7,
                low_confidence=False,
                needs_clarification=False,
                requires_blocking_clarification=False,
                questions=[],
                blocking_slots=[],
                missing_slots=[],
                slot_values={"intent_action": "render", "risk_level": "bounded_write"},
            ),
            graph=main.workflow_contracts.IntentGraph(segments=[]),
        )

    monkeypatch.setattr(main, "CHAT_ROUTING_MODE", "response_first")
    monkeypatch.setattr(main, "CHAT_RESPONSE_MODE", "answer_or_handoff")
    monkeypatch.setattr(main, "_normalize_goal_intent", _normalize_goal_intent)
    monkeypatch.setattr(main, "_chat_router_provider", _Router())
    monkeypatch.setattr(main, "_chat_response_provider", _Responder())
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
    assert calls["router"] == 1
    assert normalize_calls["count"] >= 1
    assert body["job"]["goal"] == "Render a PDF deployment report"
    assert body["assistant_message"]["action"]["type"] == "submit_job"


def test_chat_turn_tool_call_does_not_use_chat_response_provider(monkeypatch) -> None:
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

    class _Responder:
        def generate_request(self, request):
            raise AssertionError("chat response provider should not run for tool_call")

    def _run_chat_direct_capability(*, db, chat_session_id, goal, capability_id, arguments, context_json, priority):
        del db, chat_session_id, goal, arguments, context_json, priority
        return chat_service.ChatDirectRunResult(
            job=models.Job(
                id="job-tool-call",
                goal="List workspace files",
                context_json={},
                status=models.JobStatus.succeeded,
                created_at=datetime.now(UTC),
                updated_at=datetime.now(UTC),
                priority=0,
                metadata={"workflow_source": "chat_direct"},
            ),
            capability_id=capability_id,
            tool_name="filesystem.workspace.list",
            output={"entries": [{"path": "/shared/workspace/README.md", "type": "file"}]},
            assistant_response="Listed workspace files.",
            error=None,
        )

    monkeypatch.setattr(main, "_chat_router_provider", _Router())
    monkeypatch.setattr(main, "_chat_response_provider", _Responder())
    monkeypatch.setattr(main, "_run_chat_direct_capability", _run_chat_direct_capability)
    session = client.post("/chat/sessions", json={}).json()

    response = client.post(
        f"/chat/sessions/{session['id']}/messages",
        json={"content": "List the workspace files", "context_json": {}, "priority": 0},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["assistant_message"]["action"]["type"] == "tool_call"
    assert body["assistant_message"]["content"] == "Listed workspace files."
    assert body["job"]["id"] == "job-tool-call"


def test_chat_turn_executes_direct_capability_as_synchronous_one_step_run(monkeypatch) -> None:
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

    def _execute_chat_task_sync(task_payload: dict[str, object]) -> models.TaskResult:
        assert task_payload["job_id"]
        assert task_payload["run_id"] == task_payload["job_id"]
        assert task_payload["tool_inputs"]["filesystem.workspace.list"]["max_files"] == 2
        now = datetime.now(UTC)
        output = {"entries": [{"path": "/shared/workspace/README.md", "type": "file"}]}
        return models.TaskResult(
            task_id=str(task_payload["task_id"]),
            status=models.TaskStatus.completed,
            outputs={"filesystem.workspace.list": output},
            artifacts=[],
            tool_calls=[
                models.ToolCall(
                    tool_name="filesystem.workspace.list",
                    input=dict(task_payload["tool_inputs"]["filesystem.workspace.list"]),
                    idempotency_key="chat-direct-sync",
                    trace_id=str(task_payload["trace_id"]),
                    request_id="filesystem.workspace.list",
                    capability_id="filesystem.workspace.list",
                    adapter_id="mcp:filesystem:workspace_list_files",
                    started_at=now,
                    finished_at=now,
                    status="completed",
                    output_or_error=output,
                )
            ],
            started_at=now,
            finished_at=now,
            error=None,
        )

    monkeypatch.setattr(main, "_chat_router_provider", _Router())
    monkeypatch.setattr(main, "_execute_chat_task_sync", _execute_chat_task_sync)
    session = client.post("/chat/sessions", json={}).json()

    response = client.post(
        f"/chat/sessions/{session['id']}/messages",
        json={"content": "List the workspace files", "context_json": {}, "priority": 0},
    )

    assert response.status_code == 200
    body = response.json()
    job_id = body["job"]["id"]
    assert body["job"]["status"] == "succeeded"
    assert body["job"]["metadata"]["workflow_source"] == "chat_direct"
    assert body["job"]["metadata"]["scheduler_mode"] == "postgres_run_spec"
    assert body["job"]["metadata"]["run_kind"] == "chat_direct"
    assert body["job"]["metadata"]["run_spec"]["kind"] == "chat_direct"
    assert body["session"]["active_job_id"] is None
    assert body["assistant_message"]["action"]["type"] == "tool_call"
    assert body["assistant_message"]["action"]["job_id"] == job_id
    assert body["assistant_message"]["action"]["capability_id"] == "filesystem.workspace.list"
    assert body["assistant_message"]["job_id"] == job_id
    assert "README.md" in body["assistant_message"]["content"]
    assert body["assistant_message"]["metadata"]["tool_output"]["entries"][0]["path"].endswith(
        "README.md"
    )

    task_results = client.get(f"/jobs/{job_id}/task_results")
    assert task_results.status_code == 200
    task_result_payload = next(iter(task_results.json().values()))
    assert task_result_payload["outputs"]["filesystem.workspace.list"]["entries"][0]["path"].endswith(
        "README.md"
    )

    attempts_response = client.get(f"/jobs/{job_id}/debugger/attempts")
    assert attempts_response.status_code == 200
    attempts = attempts_response.json()
    assert len(attempts) == 1
    assert attempts[0]["run_id"] == job_id
    assert attempts[0]["status"] == "completed"

    invocations_response = client.get(f"/jobs/{job_id}/debugger/invocations")
    assert invocations_response.status_code == 200
    invocations = invocations_response.json()
    assert len(invocations) == 1
    assert invocations[0]["run_id"] == job_id
    assert invocations[0]["capability_id"] == "filesystem.workspace.list"
    assert invocations[0]["adapter_id"] == "mcp:filesystem:workspace_list_files"

    events_response = client.get(f"/jobs/{job_id}/debugger/events")
    assert events_response.status_code == 200
    event_types = {entry["event_type"] for entry in events_response.json()}
    assert {"task.ready", "task.started", "task.completed"}.issubset(event_types)


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
