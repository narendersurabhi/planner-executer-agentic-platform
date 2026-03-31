import json
import os
import re
from datetime import UTC, datetime

from fastapi.testclient import TestClient

os.environ["DATABASE_URL"] = "sqlite:///./test.db"
os.environ["ORCHESTRATOR_ENABLED"] = "false"
os.environ["JOB_RECOVERY_ENABLED"] = "false"
os.environ["POLICY_GATE_ENABLED"] = "false"
os.environ["CAPABILITY_MODE"] = "enabled"

from libs.core import models  # noqa: E402
from libs.core import capability_registry as cap_registry  # noqa: E402
from services.api.app import chat_service, main, memory_profile_service  # noqa: E402
from services.api.app.database import Base, SessionLocal, engine  # noqa: E402


Base.metadata.create_all(bind=engine)

client = TestClient(main.app)
main.CHAT_RESPONSE_MODE = "answer_only"
main.CHAT_ROUTING_MODE = "response_first"
main._chat_router_provider = None
main._chat_response_provider = None
main._chat_pending_correction_provider = None


def _metric_value(metrics_text: str, metric_name: str, labels: dict[str, str] | None = None) -> float:
    for line in metrics_text.splitlines():
        if not line or line.startswith("#"):
            continue
        if labels:
            if not line.startswith(f"{metric_name}{{"):
                continue
            if any(f'{key}="{value}"' not in line for key, value in labels.items()):
                continue
            return float(line.rsplit(" ", 1)[-1])
        if re.match(rf"^{re.escape(metric_name)}\s+[-+0-9.eE]+$", line):
            return float(line.rsplit(" ", 1)[-1])
    return 0.0


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


def test_chat_capability_vector_sync_cleans_stale_namespaces(monkeypatch) -> None:
    capability = cap_registry.CapabilitySpec(
        capability_id="github.file.create_or_update",
        description="Create or update a single file in a repository.",
        risk_tier="bounded_write",
        idempotency="safe_write",
        group="github",
        subgroup="files",
        enabled=True,
    )
    capabilities = [(capability.capability_id, capability)]
    entries = main._chat_capability_search_entries(capabilities)
    current_namespace = main._chat_capability_vector_namespace(capabilities)
    stale_namespace = f"{main.CHAT_CAPABILITY_VECTOR_NAMESPACE_PREFIX}:legacy"
    calls: list[tuple[str, dict[str, object]]] = []

    def _rag_request(path, *, method="GET", body=None, query=None, timeout_s=20.0):
        del method, query, timeout_s
        calls.append((path, dict(body or {})))
        if path == "/index/upsert_texts":
            return {"collection_name": "test", "upserted_count": 1, "chunk_ids": ["c1"]}
        if path == "/documents/list":
            return {
                "collection_name": "test",
                "truncated": False,
                "scanned_point_count": 2,
                "documents": [
                    {
                        "document_id": "github.file.create_or_update",
                        "namespace": current_namespace,
                        "metadata": {"catalog_type": "assistant_capability"},
                    },
                    {
                        "document_id": "document.docx.generate",
                        "namespace": stale_namespace,
                        "metadata": {"catalog_type": "assistant_capability"},
                    },
                ],
            }
        if path == "/documents/delete":
            return {
                "collection_name": "test",
                "document_id": body["document_id"],
                "deleted_chunk_count": 1,
            }
        raise AssertionError(f"unexpected RAG path: {path}")

    monkeypatch.setattr(main, "_rag_retriever_request_json", _rag_request)
    monkeypatch.setattr(main, "_chat_capability_vector_synced_namespace", None)

    namespace = main._ensure_chat_capability_vector_index(capabilities, entries)

    assert namespace == current_namespace
    assert [path for path, _body in calls] == [
        "/index/upsert_texts",
        "/documents/list",
        "/documents/delete",
    ]
    assert calls[-1][1] == {
        "collection_name": main.CHAT_CAPABILITY_VECTOR_COLLECTION,
        "namespace": stale_namespace,
        "workspace_id": main.CHAT_CAPABILITY_VECTOR_WORKSPACE_ID,
        "document_id": "document.docx.generate",
    }


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
            "context_json": {
                "topic": "Deployment report",
                "path": "artifacts/deployment-report.pdf",
            },
            "priority": 1,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["job"]["goal"] == "Render a PDF deployment report"
    assert body["job"]["metadata"]["workflow_source"] == "chat"
    assert body["job"]["metadata"]["render_path_mode"] == "auto"
    assert body["assistant_message"]["action"]["type"] == "submit_job"
    assert body["assistant_message"]["action"]["job_id"] == body["job"]["id"]
    assert body["session"]["active_job_id"] == body["job"]["id"]
    assert len(body["session"]["messages"]) == 2


def test_chat_turn_does_not_spawn_new_job_for_active_job_confirmation() -> None:
    session = client.post("/chat/sessions", json={}).json()

    first_response = client.post(
        f"/chat/sessions/{session['id']}/messages",
        json={
            "content": "Render a PDF deployment report",
            "context_json": {
                "topic": "Deployment report",
                "path": "artifacts/deployment-report.pdf",
            },
            "priority": 1,
        },
    )
    assert first_response.status_code == 200
    first_body = first_response.json()
    active_job_id = first_body["job"]["id"]

    follow_up = client.post(
        f"/chat/sessions/{session['id']}/messages",
        json={"content": "yes. go ahead", "context_json": {}, "priority": 1},
    )

    assert follow_up.status_code == 200
    body = follow_up.json()
    assert body["job"] is None
    assert body["assistant_message"]["action"]["type"] == "respond"
    assert active_job_id in body["assistant_message"]["content"]
    assert body["session"]["active_job_id"] == active_job_id


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
        json={
            "content": "PDF for leadership, dry run only. Save it as artifacts/deployment-report.pdf",
            "context_json": {"path": "artifacts/deployment-report.pdf"},
            "priority": 0,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["job"] is not None
    assert body["assistant_message"]["action"]["type"] == "submit_job"
    assert "User clarification:" in body["job"]["goal"]


def test_chat_turn_normalizes_document_clarification_before_submit(monkeypatch) -> None:
    def _normalize_goal_intent(goal, **_kwargs):
        pending = "User clarification:" not in goal
        return main.workflow_contracts.NormalizedIntentEnvelope(
            goal=goal,
            profile=main.workflow_contracts.GoalIntentProfile(
                intent="render",
                source="test",
                confidence=0.95,
                risk_level="bounded_write",
                threshold=0.7,
                low_confidence=False,
                needs_clarification=pending,
                requires_blocking_clarification=pending,
                questions=["What output format do you need?"] if pending else [],
                blocking_slots=["output_format"] if pending else [],
                missing_slots=["output_format"] if pending else [],
                slot_values={"intent_action": "render", "risk_level": "bounded_write"},
            ),
            graph=main.workflow_contracts.IntentGraph(
                segments=[
                    main.workflow_contracts.IntentGraphSegment(
                        id="s1",
                        intent="generate",
                        objective="Generate document spec",
                        required_inputs=["instruction", "topic", "audience", "tone"],
                        suggested_capabilities=["document.spec.generate"],
                    )
                ]
            ),
            candidate_capabilities={"s1": ["document.spec.generate"]},
        )

    class _Normalizer:
        def generate_request_json_object(self, request):
            assert request.metadata == {"component": "chat_clarification_normalizer"}
            return {
                "normalized_slots": {
                    "topic": "Deployment report",
                    "audience": "Senior AI engineers",
                    "tone": "practical",
                },
                "field_confidence": {
                    "topic": 0.91,
                    "audience": 0.94,
                    "tone": 0.96,
                },
                "unresolved_fields": [],
            }

    monkeypatch.setattr(main, "_normalize_goal_intent", _normalize_goal_intent)
    monkeypatch.setattr(main, "_chat_clarification_normalizer_provider", _Normalizer())
    monkeypatch.setattr(main, "CHAT_CLARIFICATION_NORMALIZER_ENABLED", True)

    session = client.post("/chat/sessions", json={}).json()
    first = client.post(
        f"/chat/sessions/{session['id']}/messages",
        json={"content": "Create a deployment report", "context_json": {}, "priority": 0},
    )
    assert first.status_code == 200
    assert first.json()["assistant_message"]["action"]["type"] == "ask_clarification"

    response = client.post(
        f"/chat/sessions/{session['id']}/messages",
        json={
            "content": "Make it a PDF for senior AI engineers. Give your best.",
            "context_json": {},
            "priority": 0,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["job"] is not None
    assert body["assistant_message"]["action"]["type"] == "submit_job"
    assert body["job"]["context_json"]["topic"] == "Deployment report"
    assert body["job"]["context_json"]["audience"] == "Senior AI engineers"
    assert body["job"]["context_json"]["tone"] == "practical"
    assert body["job"]["context_json"]["clarification_normalization"]["source"] == (
        "chat_clarification_normalizer"
    )


def test_chat_turn_keeps_document_submit_in_clarification_when_normalizer_leaves_required_field_unresolved(
    monkeypatch,
) -> None:
    def _normalize_goal_intent(goal, **_kwargs):
        pending = "User clarification:" not in goal
        return main.workflow_contracts.NormalizedIntentEnvelope(
            goal=goal,
            profile=main.workflow_contracts.GoalIntentProfile(
                intent="render",
                source="test",
                confidence=0.95,
                risk_level="bounded_write",
                threshold=0.7,
                low_confidence=False,
                needs_clarification=pending,
                requires_blocking_clarification=pending,
                questions=["What output format do you need?"] if pending else [],
                blocking_slots=["output_format"] if pending else [],
                missing_slots=["output_format"] if pending else [],
                slot_values={"intent_action": "render", "risk_level": "bounded_write"},
            ),
            graph=main.workflow_contracts.IntentGraph(
                segments=[
                    main.workflow_contracts.IntentGraphSegment(
                        id="s1",
                        intent="generate",
                        objective="Generate document spec",
                        required_inputs=["instruction", "topic", "audience", "tone"],
                        suggested_capabilities=["document.spec.generate"],
                    )
                ]
            ),
            candidate_capabilities={"s1": ["document.spec.generate"]},
        )

    class _Normalizer:
        def generate_request_json_object(self, request):
            assert request.metadata == {"component": "chat_clarification_normalizer"}
            return {
                "normalized_slots": {
                    "topic": "Deployment report",
                    "audience": "Senior AI engineers",
                },
                "field_confidence": {
                    "topic": 0.91,
                    "audience": 0.94,
                    "tone": 0.25,
                },
                "unresolved_fields": ["tone"],
            }

    monkeypatch.setattr(main, "_normalize_goal_intent", _normalize_goal_intent)
    monkeypatch.setattr(main, "_chat_clarification_normalizer_provider", _Normalizer())
    monkeypatch.setattr(main, "CHAT_CLARIFICATION_NORMALIZER_ENABLED", True)

    session = client.post("/chat/sessions", json={}).json()
    client.post(
        f"/chat/sessions/{session['id']}/messages",
        json={"content": "Create a deployment report", "context_json": {}, "priority": 0},
    )

    response = client.post(
        f"/chat/sessions/{session['id']}/messages",
        json={
            "content": "Make it a PDF for senior AI engineers. Give your best.",
            "context_json": {},
            "priority": 0,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["job"] is None
    assert body["assistant_message"]["action"]["type"] == "ask_clarification"
    assert "what tone should it use" in body["assistant_message"]["content"].lower()
    assert body["session"]["metadata"]["pending_clarification"]["questions"]


def test_chat_turn_blocks_first_submit_when_selected_capability_required_fields_are_missing(
    monkeypatch,
) -> None:
    class _Router:
        def generate_request_json_object(self, request):
            return {
                "route": "submit_job",
                "assistant_response": "",
                "intent": "generate",
                "risk_level": "bounded_write",
                "confidence": 0.97,
            }

    def _normalize_goal_intent(goal, **kwargs):
        context_envelope = kwargs.get("context_envelope")
        missing_fields: list[str] = []
        if context_envelope is not None:
            intent_context = main.context_service.intent_context_view(context_envelope)
            for field in ("audience", "tone"):
                if not str(intent_context.get(field) or "").strip():
                    missing_fields.append(field)

        questions = [
            main.chat_clarification_normalizer.clarification_question_for_field(field, goal=goal)
            for field in missing_fields
        ]
        return main.workflow_contracts.NormalizedIntentEnvelope(
            goal=goal,
            profile=main.workflow_contracts.GoalIntentProfile(
                intent="generate",
                source="test",
                confidence=0.97,
                risk_level="bounded_write",
                threshold=0.7,
                low_confidence=False,
                needs_clarification=bool(missing_fields),
                requires_blocking_clarification=bool(missing_fields),
                questions=questions,
                blocking_slots=list(missing_fields),
                missing_slots=list(missing_fields),
                slot_values={"intent_action": "generate", "risk_level": "bounded_write"},
                clarification_mode="capability_required_inputs" if missing_fields else None,
            ),
            graph=main.workflow_contracts.IntentGraph(
                segments=[
                    main.workflow_contracts.IntentGraphSegment(
                        id="s1",
                        intent="generate",
                        objective="Generate document spec",
                        required_inputs=["instruction", "topic", "audience", "tone"],
                        suggested_capabilities=["document.spec.generate"],
                    )
                ]
            ),
            candidate_capabilities={"s1": ["document.spec.generate"]},
        )

    monkeypatch.setattr(main, "CHAT_ROUTING_MODE", "always_router")
    monkeypatch.setattr(main, "_chat_router_provider", _Router())
    monkeypatch.setattr(main, "_normalize_goal_intent", _normalize_goal_intent)
    monkeypatch.setattr(main, "CHAT_CLARIFICATION_NORMALIZER_ENABLED", False)

    session = client.post("/chat/sessions", json={}).json()
    response = client.post(
        f"/chat/sessions/{session['id']}/messages",
        json={"content": "Create a deployment report", "context_json": {}, "priority": 0},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["job"] is None
    assert body["assistant_message"]["action"]["type"] == "ask_clarification"
    assert "target audience" in body["assistant_message"]["content"].lower()
    assert "tone" in body["assistant_message"]["content"].lower()
    assert body["session"]["metadata"]["pending_clarification"]["questions"]


def test_chat_turn_blocks_submit_when_post_normalization_still_requires_blocking_clarification(
    monkeypatch,
) -> None:
    class _Router:
        def generate_request_json_object(self, request):
            return {
                "route": "submit_job",
                "assistant_response": "",
                "intent": "generate",
                "risk_level": "bounded_write",
                "confidence": 0.97,
            }

    def _normalize_goal_intent(goal, **_kwargs):
        return main.workflow_contracts.NormalizedIntentEnvelope(
            goal=goal,
            profile=main.workflow_contracts.GoalIntentProfile(
                intent="generate",
                source="test",
                confidence=0.97,
                risk_level="bounded_write",
                threshold=0.7,
                low_confidence=False,
                needs_clarification=False,
                requires_blocking_clarification=False,
                questions=[],
                blocking_slots=[],
                missing_slots=[],
                slot_values={"intent_action": "generate", "risk_level": "bounded_write"},
            ),
            graph=main.workflow_contracts.IntentGraph(),
            candidate_capabilities={},
        )

    def _normalize_submit_context(**kwargs):
        return chat_service.ChatSubmitNormalizationResult(
            goal=kwargs["goal"],
            clarification_questions=[],
            requires_blocking_clarification=True,
            goal_intent_profile={
                "intent": "render",
                "source": "test_submit_normalizer",
                "confidence": 0.68,
                "risk_level": "bounded_write",
                "threshold": 0.7,
                "low_confidence": True,
                "needs_clarification": True,
                "requires_blocking_clarification": True,
                "questions": ["What should the system do first (generate, transform, validate, render, or io)?"],
                "blocking_slots": ["intent_action"],
                "missing_slots": ["intent_action"],
                "slot_values": {"intent_action": "render", "risk_level": "bounded_write"},
            },
        )

    monkeypatch.setattr(main, "CHAT_ROUTING_MODE", "always_router")
    monkeypatch.setattr(main, "_chat_router_provider", _Router())
    monkeypatch.setattr(main, "_normalize_goal_intent", _normalize_goal_intent)
    monkeypatch.setattr(main, "_normalize_chat_submit_context", _normalize_submit_context)

    session = client.post("/chat/sessions", json={}).json()
    response = client.post(
        f"/chat/sessions/{session['id']}/messages",
        json={"content": "create a document", "context_json": {}, "priority": 0},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["job"] is None
    assert body["assistant_message"]["action"]["type"] == "ask_clarification"
    assert "what should the system do first" in body["assistant_message"]["content"].lower()
    assert body["assistant_message"]["action"]["goal_intent_profile"][
        "requires_blocking_clarification"
    ]
    assert body["session"]["metadata"]["pending_clarification"]["questions"]


def test_chat_turn_persists_typed_pending_clarification_state_with_active_target(
    monkeypatch,
) -> None:
    def _normalize_goal_intent(goal, **_kwargs):
        return main.workflow_contracts.NormalizedIntentEnvelope(
            goal=goal,
            profile=main.workflow_contracts.GoalIntentProfile(
                intent="generate",
                source="test",
                confidence=0.97,
                risk_level="bounded_write",
                threshold=0.7,
                needs_clarification=True,
                requires_blocking_clarification=True,
                questions=["Who is the target audience?", "What tone should it use?"],
                blocking_slots=["audience", "tone"],
                missing_slots=["audience", "tone"],
                slot_values={"intent_action": "generate", "risk_level": "bounded_write"},
            ),
            graph=main.workflow_contracts.IntentGraph(
                segments=[
                    main.workflow_contracts.IntentGraphSegment(
                        id="s1",
                        intent="generate",
                        objective="Generate document spec",
                        suggested_capabilities=["document.spec.generate"],
                    )
                ]
            ),
            candidate_capabilities={"s1": ["document.spec.generate"]},
        )

    registry = cap_registry.CapabilityRegistry(
        capabilities={
            "document.spec.generate": cap_registry.CapabilitySpec(
                capability_id="document.spec.generate",
                description="Generate a document spec.",
                risk_tier="bounded_write",
                idempotency="write",
                group="documents",
                subgroup="generation",
            )
        }
    )

    monkeypatch.setattr(main, "CHAT_ROUTING_MODE", "response_first")
    monkeypatch.setattr(main, "_normalize_goal_intent", _normalize_goal_intent)
    monkeypatch.setattr(main.capability_registry, "load_capability_registry", lambda: registry)
    monkeypatch.setattr(chat_service.capability_registry, "load_capability_registry", lambda: registry)

    session = client.post("/chat/sessions", json={}).json()
    response = client.post(
        f"/chat/sessions/{session['id']}/messages",
        json={"content": "Create a deployment report", "context_json": {}, "priority": 0},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["assistant_message"]["action"]["type"] == "ask_clarification"
    assert body["assistant_message"]["action"]["clarification_questions"] == [
        "Who is the target audience?"
    ]
    assert body["assistant_message"]["content"] == "Who is the target audience?"
    pending = body["session"]["metadata"]["pending_clarification"]
    assert pending["schema_version"] == "clarification_state_v1"
    assert pending["state_version"] == 1
    assert pending["original_goal"] == "Create a deployment report"
    assert pending["active_family"] == "documents"
    assert pending["active_segment_id"] == "s1"
    assert pending["active_capability_id"] == "document.spec.generate"
    assert pending["execution_frame"]["schema_version"] == "execution_frame_v1"
    assert pending["execution_frame"]["mode"] == "clarification"
    assert pending["execution_frame"]["active_capability_id"] == "document.spec.generate"
    assert pending["questions"] == ["Who is the target audience?"]
    assert pending["pending_questions"] == [
        "Who is the target audience?",
        "What tone should it use?",
    ]
    assert pending["current_question"] == "Who is the target audience?"
    assert pending["current_question_field"] == "audience"


def test_chat_turn_asks_one_submit_normalization_question_at_a_time(monkeypatch) -> None:
    class _Router:
        def generate_request_json_object(self, request):
            return {
                "route": "submit_job",
                "assistant_response": "",
                "intent": "generate",
                "risk_level": "bounded_write",
                "confidence": 0.97,
            }

    def _normalize_goal_intent(goal, **_kwargs):
        return main.workflow_contracts.NormalizedIntentEnvelope(
            goal=goal,
            profile=main.workflow_contracts.GoalIntentProfile(
                intent="generate",
                source="test",
                confidence=0.97,
                risk_level="bounded_write",
                threshold=0.7,
                low_confidence=False,
                needs_clarification=False,
                requires_blocking_clarification=False,
                questions=[],
                blocking_slots=[],
                missing_slots=[],
                slot_values={"intent_action": "generate", "risk_level": "bounded_write"},
            ),
            graph=main.workflow_contracts.IntentGraph(),
            candidate_capabilities={},
        )

    def _normalize_submit_context(**kwargs):
        return chat_service.ChatSubmitNormalizationResult(
            goal=kwargs["goal"],
            clarification_questions=[
                "Who is the target audience?",
                "What tone should it use?",
            ],
            requires_blocking_clarification=True,
            goal_intent_profile={
                "intent": "generate",
                "source": "test_submit_normalizer",
                "confidence": 0.91,
                "risk_level": "bounded_write",
                "threshold": 0.7,
                "low_confidence": False,
                "needs_clarification": True,
                "requires_blocking_clarification": True,
                "questions": [
                    "Who is the target audience?",
                    "What tone should it use?",
                ],
                "blocking_slots": ["audience", "tone"],
                "missing_slots": ["audience", "tone"],
                "slot_values": {"intent_action": "generate", "risk_level": "bounded_write"},
            },
        )

    monkeypatch.setattr(main, "CHAT_ROUTING_MODE", "always_router")
    monkeypatch.setattr(main, "_chat_router_provider", _Router())
    monkeypatch.setattr(main, "_normalize_goal_intent", _normalize_goal_intent)
    monkeypatch.setattr(main, "_normalize_chat_submit_context", _normalize_submit_context)

    session = client.post("/chat/sessions", json={}).json()
    response = client.post(
        f"/chat/sessions/{session['id']}/messages",
        json={"content": "create a deployment report", "context_json": {}, "priority": 0},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["job"] is None
    assert body["assistant_message"]["action"]["type"] == "ask_clarification"
    assert body["assistant_message"]["action"]["clarification_questions"] == [
        "Who is the target audience?"
    ]
    assert body["assistant_message"]["content"] == "Who is the target audience?"
    pending = body["session"]["metadata"]["pending_clarification"]
    assert pending["questions"] == ["Who is the target audience?"]
    assert pending["pending_questions"] == [
        "Who is the target audience?",
        "What tone should it use?",
    ]
    assert pending["current_question_field"] == "audience"


def test_chat_turn_maps_pending_answer_before_routing_and_advances_queue(
    monkeypatch,
) -> None:
    route_calls = {"count": 0}

    def _route_turn(**kwargs):
        route_calls["count"] += 1
        if route_calls["count"] == 1:
            return {
                "type": "ask_clarification",
                "assistant_content": "Who is the target audience?",
                "clarification_questions": [
                    "Who is the target audience?",
                    "What tone should it use?",
                ],
                "goal_intent_profile": {
                    "intent": "generate",
                    "source": "test",
                    "confidence": 0.97,
                    "risk_level": "bounded_write",
                    "threshold": 0.7,
                    "low_confidence": False,
                    "needs_clarification": True,
                    "requires_blocking_clarification": True,
                    "questions": [
                        "Who is the target audience?",
                        "What tone should it use?",
                    ],
                    "blocking_slots": ["audience", "tone"],
                    "missing_slots": ["audience", "tone"],
                    "slot_values": {
                        "intent_action": "generate",
                        "risk_level": "bounded_write",
                    },
                },
                "resolved_goal": kwargs["candidate_goal"],
            }
        assert kwargs["merged_context"]["audience"] == "Senior software engineers"
        pending = kwargs["session_metadata"]["pending_clarification"]
        assert pending["known_slot_values"]["audience"] == "Senior software engineers"
        assert pending["current_question_field"] == "tone"
        assert pending["questions"] == ["What tone should it use?"]
        return {
            "type": "ask_clarification",
            "assistant_content": "What tone should it use?",
            "clarification_questions": ["What tone should it use?"],
            "goal_intent_profile": {
                "intent": "generate",
                "source": "test",
                "confidence": 0.97,
                "risk_level": "bounded_write",
                "threshold": 0.7,
                "low_confidence": False,
                "needs_clarification": True,
                "requires_blocking_clarification": True,
                "questions": ["What tone should it use?"],
                "blocking_slots": ["tone"],
                "missing_slots": ["tone"],
                "slot_values": {
                    "intent_action": "generate",
                    "risk_level": "bounded_write",
                    "audience": "Senior software engineers",
                },
            },
            "resolved_goal": kwargs["candidate_goal"],
        }

    def _normalize_submit_context(**kwargs):
        pending = kwargs["session_metadata"].get("pending_clarification")
        if not pending:
            return None
        return chat_service.ChatSubmitNormalizationResult(
            goal=kwargs["goal"],
            context_json={
                "audience": "Senior software engineers",
                "clarification_normalization": {
                    "source": "chat_clarification_normalizer",
                    "fields": ["audience"],
                    "confidence": {"audience": 0.97},
                },
            },
            clarification_questions=["What tone should it use?"],
            requires_blocking_clarification=True,
            goal_intent_profile={
                "intent": "generate",
                "source": "test_submit_normalizer",
                "confidence": 0.97,
                "risk_level": "bounded_write",
                "threshold": 0.7,
                "low_confidence": False,
                "needs_clarification": True,
                "requires_blocking_clarification": True,
                "questions": ["What tone should it use?"],
                "blocking_slots": ["tone"],
                "missing_slots": ["tone"],
                "slot_values": {
                    "intent_action": "generate",
                    "risk_level": "bounded_write",
                    "audience": "Senior software engineers",
                },
            },
        )

    monkeypatch.setattr(main, "_route_chat_turn", _route_turn)
    monkeypatch.setattr(main, "_normalize_chat_submit_context", _normalize_submit_context)

    session = client.post("/chat/sessions", json={}).json()
    first = client.post(
        f"/chat/sessions/{session['id']}/messages",
        json={"content": "Create a deployment report", "context_json": {}, "priority": 0},
    )
    assert first.status_code == 200
    assert first.json()["assistant_message"]["content"] == "Who is the target audience?"

    second = client.post(
        f"/chat/sessions/{session['id']}/messages",
        json={"content": "Senior software engineers", "context_json": {}, "priority": 0},
    )

    assert second.status_code == 200
    body = second.json()
    assert body["assistant_message"]["action"]["type"] == "ask_clarification"
    assert body["assistant_message"]["content"] == "What tone should it use?"
    pending = body["session"]["metadata"]["pending_clarification"]
    assert pending["known_slot_values"]["audience"] == "Senior software engineers"
    assert pending["current_question_field"] == "tone"
    assert pending["questions"] == ["What tone should it use?"]


def test_chat_turn_restarts_pending_clarification_for_execution_intent_change(
    monkeypatch,
) -> None:
    route_calls = {"count": 0}
    normalization_calls = {"count": 0}

    def _route_turn(**kwargs):
        route_calls["count"] += 1
        if route_calls["count"] == 1:
            return {
                "type": "ask_clarification",
                "assistant_content": "What tone should it use?",
                "clarification_questions": ["What tone should it use?"],
                "goal_intent_profile": {
                    "intent": "generate",
                    "source": "test",
                    "confidence": 0.97,
                    "risk_level": "bounded_write",
                    "threshold": 0.7,
                    "low_confidence": False,
                    "needs_clarification": True,
                    "requires_blocking_clarification": True,
                    "questions": ["What tone should it use?"],
                    "blocking_slots": ["tone"],
                    "missing_slots": ["tone"],
                    "slot_values": {
                        "intent_action": "generate",
                        "risk_level": "bounded_write",
                    },
                },
                "resolved_goal": kwargs["candidate_goal"],
            }
        assert "pending_clarification" not in kwargs["session_metadata"]
        assert kwargs["candidate_goal"] == "Actually make this a PDF checklist instead"
        return {
            "type": "ask_clarification",
            "assistant_content": "Who is the target audience?",
            "clarification_questions": ["Who is the target audience?"],
            "goal_intent_profile": {
                "intent": "generate",
                "source": "test",
                "confidence": 0.97,
                "risk_level": "bounded_write",
                "threshold": 0.7,
                "low_confidence": False,
                "needs_clarification": True,
                "requires_blocking_clarification": True,
                "questions": ["Who is the target audience?"],
                "blocking_slots": ["audience"],
                "missing_slots": ["audience"],
                "slot_values": {
                    "intent_action": "generate",
                    "risk_level": "bounded_write",
                    "output_format": "pdf",
                },
            },
            "resolved_goal": kwargs["candidate_goal"],
        }

    def _normalize_submit_context(**_kwargs):
        normalization_calls["count"] += 1
        return None

    monkeypatch.setattr(main, "_route_chat_turn", _route_turn)
    monkeypatch.setattr(main, "_normalize_chat_submit_context", _normalize_submit_context)

    session = client.post("/chat/sessions", json={}).json()
    first = client.post(
        f"/chat/sessions/{session['id']}/messages",
        json={"content": "Create a deployment report", "context_json": {}, "priority": 0},
    )
    assert first.status_code == 200
    assert first.json()["assistant_message"]["content"] == "What tone should it use?"

    second = client.post(
        f"/chat/sessions/{session['id']}/messages",
        json={
            "content": "Actually make this a PDF checklist instead",
            "context_json": {},
            "priority": 0,
        },
    )

    assert second.status_code == 200
    body = second.json()
    assert normalization_calls["count"] == 0
    assert body["assistant_message"]["action"]["type"] == "ask_clarification"
    assert body["assistant_message"]["content"] == "Who is the target audience?"
    pending = body["session"]["metadata"]["pending_clarification"]
    assert pending["original_goal"] == "Actually make this a PDF checklist instead"
    assert pending["questions"] == ["Who is the target audience?"]


def test_chat_turn_keeps_pending_clarification_for_output_format_redirect_answer(
    monkeypatch,
) -> None:
    route_calls = {"count": 0}
    normalization_calls = {"count": 0}

    def _route_turn(**kwargs):
        route_calls["count"] += 1
        if route_calls["count"] == 1:
            return {
                "type": "ask_clarification",
                "assistant_content": "What output format do you need (for example PDF, DOCX, JSON, or Markdown)?",
                "clarification_questions": [
                    "What output format do you need (for example PDF, DOCX, JSON, or Markdown)?"
                ],
                "goal_intent_profile": {
                    "intent": "generate",
                    "source": "test",
                    "confidence": 0.97,
                    "risk_level": "bounded_write",
                    "threshold": 0.7,
                    "low_confidence": False,
                    "needs_clarification": True,
                    "requires_blocking_clarification": True,
                    "questions": [
                        "What output format do you need (for example PDF, DOCX, JSON, or Markdown)?"
                    ],
                    "blocking_slots": ["output_format"],
                    "missing_slots": ["output_format"],
                    "slot_values": {
                        "intent_action": "generate",
                        "risk_level": "bounded_write",
                    },
                },
                "resolved_goal": kwargs["candidate_goal"],
            }
        assert kwargs["candidate_goal"].endswith(
            "User clarification: Actually make it DOCX instead"
        )
        assert "pending_clarification" in kwargs["session_metadata"]
        return {
            "type": "ask_clarification",
            "assistant_content": "What output path or filename should be used?",
            "clarification_questions": ["What output path or filename should be used?"],
            "goal_intent_profile": {
                "intent": "generate",
                "source": "test",
                "confidence": 0.97,
                "risk_level": "bounded_write",
                "threshold": 0.7,
                "low_confidence": False,
                "needs_clarification": True,
                "requires_blocking_clarification": True,
                "questions": ["What output path or filename should be used?"],
                "blocking_slots": ["path"],
                "missing_slots": ["path"],
                "slot_values": {
                    "intent_action": "generate",
                    "risk_level": "bounded_write",
                    "output_format": "docx",
                },
            },
            "resolved_goal": kwargs["candidate_goal"],
        }

    def _normalize_submit_context(**kwargs):
        normalization_calls["count"] += 1
        return chat_service.ChatSubmitNormalizationResult(
            goal=kwargs["goal"],
            context_json={
                "output_format": "docx",
                "clarification_normalization": {
                    "source": "chat_clarification_normalizer",
                    "fields": ["output_format"],
                    "confidence": {"output_format": 0.96},
                },
            },
            clarification_questions=["What output path or filename should be used?"],
            requires_blocking_clarification=True,
            goal_intent_profile={
                "intent": "generate",
                "source": "test_submit_normalizer",
                "confidence": 0.96,
                "risk_level": "bounded_write",
                "threshold": 0.7,
                "low_confidence": False,
                "needs_clarification": True,
                "requires_blocking_clarification": True,
                "questions": ["What output path or filename should be used?"],
                "blocking_slots": ["path"],
                "missing_slots": ["path"],
                "slot_values": {
                    "intent_action": "generate",
                    "risk_level": "bounded_write",
                    "output_format": "docx",
                },
            },
        )

    monkeypatch.setattr(main, "_route_chat_turn", _route_turn)
    monkeypatch.setattr(main, "_normalize_chat_submit_context", _normalize_submit_context)

    session = client.post("/chat/sessions", json={}).json()
    first = client.post(
        f"/chat/sessions/{session['id']}/messages",
        json={"content": "Create a deployment report", "context_json": {}, "priority": 0},
    )
    assert first.status_code == 200

    second = client.post(
        f"/chat/sessions/{session['id']}/messages",
        json={
            "content": "Actually make it DOCX instead",
            "context_json": {},
            "priority": 0,
        },
    )

    assert second.status_code == 200
    body = second.json()
    assert normalization_calls["count"] == 1
    assert body["assistant_message"]["action"]["type"] == "ask_clarification"
    assert body["assistant_message"]["content"] == "What output path or filename should be used?"
    pending = body["session"]["metadata"]["pending_clarification"]
    assert pending["original_goal"] == "Create a deployment report"
    assert pending["known_slot_values"]["output_format"] == "docx"


def test_chat_turn_scopes_follow_up_clarification_to_active_segment(
    monkeypatch,
) -> None:
    class _Router:
        def generate_request_json_object(self, request):
            return {
                "route": "submit_job",
                "assistant_response": "",
                "intent": "generate",
                "risk_level": "bounded_write",
                "confidence": 0.97,
            }

    def _normalize_goal_intent(goal, **kwargs):
        context_envelope = kwargs.get("context_envelope")
        intent_context = (
            main.context_service.intent_context_view(context_envelope)
            if context_envelope is not None
            else {}
        )
        has_audience = bool(str(intent_context.get("audience") or "").strip())
        missing_fields = ["tone", "query"] if has_audience else ["audience", "tone", "query"]
        questions = [
            main.chat_clarification_normalizer.clarification_question_for_field(field, goal=goal)
            for field in missing_fields
        ]
        return main.workflow_contracts.NormalizedIntentEnvelope(
            goal=goal,
            profile=main.workflow_contracts.GoalIntentProfile(
                intent="generate",
                source="test",
                confidence=0.97,
                risk_level="bounded_write",
                threshold=0.7,
                low_confidence=False,
                needs_clarification=bool(missing_fields),
                requires_blocking_clarification=bool(missing_fields),
                questions=questions,
                blocking_slots=list(missing_fields),
                missing_slots=list(missing_fields),
                slot_values={"intent_action": "generate", "risk_level": "bounded_write"},
                clarification_mode="capability_required_inputs" if missing_fields else None,
            ),
            graph=main.workflow_contracts.IntentGraph(
                segments=[
                    main.workflow_contracts.IntentGraphSegment(
                        id="s1",
                        intent="generate",
                        objective="Generate document spec",
                        required_inputs=["instruction", "audience", "tone"],
                        suggested_capabilities=["document.spec.generate"],
                    ),
                    main.workflow_contracts.IntentGraphSegment(
                        id="s2",
                        intent="io",
                        objective="Search GitHub issues",
                        required_inputs=["query"],
                        suggested_capabilities=["github.issue.search"],
                    ),
                ]
            ),
            candidate_capabilities={
                "s1": ["document.spec.generate"],
                "s2": ["github.issue.search"],
            },
        )

    registry = cap_registry.CapabilityRegistry(
        capabilities={
            "document.spec.generate": cap_registry.CapabilitySpec(
                capability_id="document.spec.generate",
                description="Generate a document spec.",
                risk_tier="bounded_write",
                idempotency="write",
                group="documents",
                subgroup="generation",
                planner_hints={
                    "chat_collectible_fields": ["instruction", "audience", "tone"],
                    "chat_required_fields": ["audience", "tone"],
                },
            ),
            "github.issue.search": cap_registry.CapabilitySpec(
                capability_id="github.issue.search",
                description="Search GitHub issues.",
                risk_tier="read_only",
                idempotency="read",
                group="github",
                subgroup="issues",
                planner_hints={
                    "chat_collectible_fields": ["query"],
                    "chat_required_fields": ["query"],
                },
            ),
        }
    )

    monkeypatch.setattr(main, "_normalize_goal_intent", _normalize_goal_intent)
    monkeypatch.setattr(main.capability_registry, "load_capability_registry", lambda: registry)
    monkeypatch.setattr(chat_service.capability_registry, "load_capability_registry", lambda: registry)
    monkeypatch.setattr(main, "CHAT_CLARIFICATION_NORMALIZER_ENABLED", False)

    session = client.post("/chat/sessions", json={}).json()

    monkeypatch.setattr(main, "CHAT_ROUTING_MODE", "response_first")
    first = client.post(
        f"/chat/sessions/{session['id']}/messages",
        json={"content": "Create a deployment report", "context_json": {}, "priority": 0},
    )

    assert first.status_code == 200
    first_body = first.json()
    assert first_body["assistant_message"]["action"]["type"] == "ask_clarification"
    pending = first_body["session"]["metadata"]["pending_clarification"]
    assert pending["active_segment_id"] == "s1"
    assert pending["active_capability_id"] == "document.spec.generate"

    monkeypatch.setattr(main, "CHAT_ROUTING_MODE", "always_router")
    monkeypatch.setattr(main, "_chat_router_provider", _Router())
    second = client.post(
        f"/chat/sessions/{session['id']}/messages",
        json={
            "content": "Audience is SSE",
            "context_json": {"audience": "SSE"},
            "priority": 0,
        },
    )

    assert second.status_code == 200
    second_body = second.json()
    assert second_body["job"] is None
    assert second_body["assistant_message"]["action"]["type"] == "ask_clarification"
    assert "tone" in second_body["assistant_message"]["content"].lower()
    assert "search query" not in second_body["assistant_message"]["content"].lower()


def test_chat_turn_keeps_submit_in_clarification_when_submit_normalization_throws(
    monkeypatch,
) -> None:
    class _Router:
        def generate_request_json_object(self, request):
            return {
                "route": "submit_job",
                "assistant_response": "Starting the job now.",
                "intent": "generate",
                "risk_level": "bounded_write",
                "confidence": 0.96,
            }

    def _raise_normalization_error(**_kwargs):
        raise RuntimeError("submit normalization blew up")

    def _normalize_goal_intent(goal, **kwargs):
        context_envelope = kwargs.get("context_envelope")
        missing_fields = []
        if context_envelope is not None:
            missing_fields.append("output_format")
        return main.workflow_contracts.NormalizedIntentEnvelope(
            goal=goal,
            profile=main.workflow_contracts.GoalIntentProfile(
                intent="generate",
                source="test",
                confidence=0.96,
                risk_level="bounded_write",
                threshold=0.7,
                low_confidence=False,
                needs_clarification=bool(missing_fields),
                requires_blocking_clarification=bool(missing_fields),
                questions=[
                    "What output format do you need (for example PDF, DOCX, JSON, or Markdown)?"
                ]
                if missing_fields
                else [],
                blocking_slots=list(missing_fields),
                missing_slots=list(missing_fields),
                slot_values={"intent_action": "generate", "risk_level": "bounded_write"},
                clarification_mode="targeted_slot_filling" if missing_fields else None,
            ),
            graph=main.workflow_contracts.IntentGraph(),
            candidate_capabilities={},
        )

    monkeypatch.setattr(main, "CHAT_ROUTING_MODE", "always_router")
    monkeypatch.setattr(main, "_chat_router_provider", _Router())
    monkeypatch.setattr(main, "_normalize_chat_submit_context", _raise_normalization_error)
    monkeypatch.setattr(main, "_normalize_goal_intent", _normalize_goal_intent)

    session = client.post("/chat/sessions", json={}).json()
    response = client.post(
        f"/chat/sessions/{session['id']}/messages",
        json={"content": "Create a deployment report", "context_json": {}, "priority": 0},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["job"] is None
    assert body["assistant_message"]["action"]["type"] == "ask_clarification"
    assert "remaining required details" in body["assistant_message"]["content"].lower()
    assert body["session"]["metadata"]["pending_clarification"]["questions"] == [
        "I still need the remaining required details before I can submit this request."
    ]


def test_chat_turn_preserves_original_goal_and_filename_across_clarification_turns(
    monkeypatch,
) -> None:
    class _Router:
        def generate_request_json_object(self, request):
            return {
                "route": "submit_job",
                "assistant_response": "",
                "intent": "generate",
                "risk_level": "bounded_write",
                "confidence": 0.97,
            }

    def _normalize_goal_intent(goal, **kwargs):
        context_envelope = kwargs.get("context_envelope")
        intent_context = (
            main.context_service.intent_context_view(context_envelope)
            if context_envelope is not None
            else {}
        )
        missing_fields: list[str] = []
        for field in ("topic", "audience", "tone"):
            if not str(intent_context.get(field) or "").strip():
                missing_fields.append(field)
        questions = [
            main.chat_clarification_normalizer.clarification_question_for_field(field, goal=goal)
            for field in missing_fields
        ]
        return main.workflow_contracts.NormalizedIntentEnvelope(
            goal=goal,
            profile=main.workflow_contracts.GoalIntentProfile(
                intent="generate",
                source="test",
                confidence=0.97,
                risk_level="bounded_write",
                threshold=0.7,
                low_confidence=False,
                needs_clarification=bool(missing_fields),
                requires_blocking_clarification=bool(missing_fields),
                questions=questions,
                blocking_slots=list(missing_fields),
                missing_slots=list(missing_fields),
                slot_values={"intent_action": "generate", "risk_level": "bounded_write"},
                clarification_mode="capability_required_inputs" if missing_fields else None,
            ),
            graph=main.workflow_contracts.IntentGraph(
                segments=[
                    main.workflow_contracts.IntentGraphSegment(
                        id="s1",
                        intent="generate",
                        objective="Generate document spec",
                        required_inputs=["instruction", "topic", "audience", "tone"],
                        suggested_capabilities=["document.spec.generate"],
                    ),
                    main.workflow_contracts.IntentGraphSegment(
                        id="s2",
                        intent="render",
                        objective="Render DOCX",
                        required_inputs=["document_spec", "output_filename"],
                        suggested_capabilities=["document.docx.render"],
                    ),
                ]
            ),
            candidate_capabilities={
                "s1": ["document.spec.generate"],
                "s2": ["document.docx.render"],
            },
        )

    class _Normalizer:
        def generate_request_json_object(self, request):
            payload = json.loads(request.prompt)
            capability_id = payload["capability_id"]
            goal = payload["goal_with_clarifications"]
            if capability_id == "document.spec.generate":
                slots = {}
                confidence = {}
                if "How AI workflows are deployed to Kubernetes" in goal:
                    slots["topic"] = "How AI workflows are deployed to Kubernetes"
                    confidence["topic"] = 0.95
                if "SSE" in goal:
                    slots["audience"] = "SSE"
                    confidence["audience"] = 0.96
                if "practical" in goal.lower():
                    slots["tone"] = "practical"
                    confidence["tone"] = 0.98
                unresolved = [
                    field for field in payload["missing_fields"] if field not in slots
                ]
                return {
                    "normalized_slots": slots,
                    "field_confidence": confidence,
                    "unresolved_fields": unresolved,
                }
            if capability_id == "document.docx.render":
                if "Kubernetes AI Deployments.docx" in goal:
                    return {
                        "normalized_slots": {"path": "Kubernetes AI Deployments.docx"},
                        "field_confidence": {"path": 0.97},
                        "unresolved_fields": [],
                    }
            return {
                "normalized_slots": {},
                "field_confidence": {},
                "unresolved_fields": payload["missing_fields"],
            }

    monkeypatch.setattr(main, "CHAT_ROUTING_MODE", "always_router")
    monkeypatch.setattr(main, "_chat_router_provider", _Router())
    monkeypatch.setattr(main, "_normalize_goal_intent", _normalize_goal_intent)
    monkeypatch.setattr(main, "_chat_clarification_normalizer_provider", _Normalizer())
    monkeypatch.setattr(main, "CHAT_CLARIFICATION_NORMALIZER_ENABLED", True)

    session = client.post("/chat/sessions", json={}).json()
    first = client.post(
        f"/chat/sessions/{session['id']}/messages",
        json={"content": "Create a document", "context_json": {}, "priority": 0},
    )

    assert first.status_code == 200
    first_body = first.json()
    assert first_body["assistant_message"]["action"]["type"] == "ask_clarification"
    assert (
        first_body["session"]["metadata"]["pending_clarification"]["original_goal"]
        == "Create a document"
    )

    second = client.post(
        f"/chat/sessions/{session['id']}/messages",
        json={
            "content": (
                '1) Document title and desired filename. Answer: "Kubernetes AI Deployments.docx"\n'
                '2) The full content or an outline. Answer: "How AI workflows are deployed to Kubernetes"\n'
                "3) Intended audience and purpose. Answer: SSE\n"
                "8) Where should I deliver/save the .docx? Answer: save to workspace folder"
            ),
            "context_json": {},
            "priority": 0,
        },
    )

    assert second.status_code == 200
    second_body = second.json()
    assert second_body["job"] is None
    assert second_body["assistant_message"]["action"]["type"] == "ask_clarification"
    assert "tone" in second_body["assistant_message"]["content"].lower()
    assert (
        second_body["session"]["metadata"]["pending_clarification"]["original_goal"]
        == "Create a document"
    )
    assert second_body["session"]["metadata"]["context_json"]["path"] == (
        "Kubernetes AI Deployments.docx"
    )

    third = client.post(
        f"/chat/sessions/{session['id']}/messages",
        json={
            "content": "Practical",
            "context_json": {},
            "priority": 0,
        },
    )

    assert third.status_code == 200
    third_body = third.json()
    assert third_body["job"] is not None
    assert third_body["assistant_message"]["action"]["type"] == "submit_job"
    assert third_body["job"]["context_json"]["topic"] == "How AI workflows are deployed to Kubernetes"
    assert third_body["job"]["context_json"]["audience"] == "SSE"
    assert third_body["job"]["context_json"]["tone"] == "practical"
    assert third_body["job"]["context_json"]["path"] == "Kubernetes AI Deployments.docx"


def test_chat_submit_normalization_uses_prior_chat_turns_to_fill_document_slots(
    monkeypatch,
) -> None:
    calls = {"router": 0}

    class _Router:
        def generate_request_json_object(self, request):
            calls["router"] += 1
            if calls["router"] == 1:
                return {
                    "route": "ask_clarification",
                    "assistant_response": "",
                    "intent": "generate",
                    "risk_level": "bounded_write",
                    "confidence": 0.97,
                    "clarification_questions": [
                        "Who is the target audience?",
                        "What tone should it use?",
                        "What output path or filename should be used?",
                    ],
                    "missing_slots": ["audience", "tone", "path"],
                }
            if calls["router"] == 2:
                return {
                    "route": "ask_clarification",
                    "assistant_response": "",
                    "intent": "generate",
                    "risk_level": "bounded_write",
                    "confidence": 0.97,
                    "clarification_questions": ["What output path or filename should be used?"],
                    "missing_slots": ["path"],
                }
            return {
                "route": "submit_job",
                "assistant_response": "",
                "intent": "generate",
                "risk_level": "bounded_write",
                "confidence": 0.97,
            }

    def _normalize_goal_intent(goal, **kwargs):
        context_envelope = kwargs.get("context_envelope")
        intent_context = (
            main.context_service.intent_context_view(context_envelope)
            if context_envelope is not None
            else {}
        )
        missing_fields: list[str] = []
        for field in ("topic", "audience", "tone", "path"):
            if not str(intent_context.get(field) or "").strip():
                missing_fields.append(field)
        questions = [
            main.chat_clarification_normalizer.clarification_question_for_field(field, goal=goal)
            for field in missing_fields
        ]
        return main.workflow_contracts.NormalizedIntentEnvelope(
            goal=goal,
            profile=main.workflow_contracts.GoalIntentProfile(
                intent="generate",
                source="test",
                confidence=0.97,
                risk_level="bounded_write",
                threshold=0.7,
                low_confidence=False,
                needs_clarification=bool(missing_fields),
                requires_blocking_clarification=bool(missing_fields),
                questions=questions,
                blocking_slots=list(missing_fields),
                missing_slots=list(missing_fields),
                slot_values={"intent_action": "generate", "risk_level": "bounded_write"},
                clarification_mode="capability_required_inputs" if missing_fields else None,
            ),
            graph=main.workflow_contracts.IntentGraph(
                segments=[
                    main.workflow_contracts.IntentGraphSegment(
                        id="s1",
                        intent="generate",
                        objective="Generate document spec",
                        required_inputs=["instruction", "topic", "audience", "tone"],
                        suggested_capabilities=["document.spec.generate"],
                    ),
                    main.workflow_contracts.IntentGraphSegment(
                        id="s2",
                        intent="render",
                        objective="Render DOCX",
                        required_inputs=["document_spec", "output_filename"],
                        suggested_capabilities=["document.docx.render"],
                    ),
                ]
            ),
            candidate_capabilities={
                "s1": ["document.spec.generate"],
                "s2": ["document.docx.render"],
            },
        )

    class _Normalizer:
        def generate_request_json_object(self, request):
            payload = json.loads(request.prompt)
            history = payload["conversation_history"]
            combined = " ".join(entry["content"] for entry in history).lower()
            capability_id = payload["capability_id"]
            if capability_id == "document.spec.generate":
                assert any(
                    "target audience is senior ai engineers" in entry["content"].lower()
                    for entry in history
                )
                assert any("practical tone" in entry["content"].lower() for entry in history)
                slots = {}
                if "kubernetes deployment report" in payload["goal_with_clarifications"].lower():
                    slots["topic"] = "Kubernetes deployment report"
                if "senior ai engineers" in combined:
                    slots["audience"] = "Senior AI engineers"
                if "practical" in combined:
                    slots["tone"] = "practical"
                unresolved = [
                    field for field in payload["missing_fields"] if field not in slots
                ]
                return {
                    "normalized_slots": slots,
                    "field_confidence": {field: 0.99 for field in slots},
                    "unresolved_fields": unresolved,
                }
            if capability_id == "document.docx.render":
                assert "medic.docx" in payload["goal_with_clarifications"].lower()
                return {
                    "normalized_slots": {"path": "medic.docx"},
                    "field_confidence": {"path": 0.99},
                    "unresolved_fields": [],
                }
            return {
                "normalized_slots": {},
                "field_confidence": {},
                "unresolved_fields": payload["missing_fields"],
            }

    monkeypatch.setattr(main, "CHAT_ROUTING_MODE", "always_router")
    monkeypatch.setattr(main, "_chat_router_provider", _Router())
    monkeypatch.setattr(main, "_normalize_goal_intent", _normalize_goal_intent)
    monkeypatch.setattr(main, "_chat_clarification_normalizer_provider", _Normalizer())
    monkeypatch.setattr(main, "CHAT_CLARIFICATION_NORMALIZER_ENABLED", True)

    session = client.post("/chat/sessions", json={}).json()
    first = client.post(
        f"/chat/sessions/{session['id']}/messages",
        json={"content": "Create a Kubernetes deployment report", "context_json": {}, "priority": 0},
    )
    assert first.status_code == 200
    assert first.json()["assistant_message"]["action"]["type"] == "ask_clarification"

    second = client.post(
        f"/chat/sessions/{session['id']}/messages",
        json={
            "content": "Target audience is Senior AI engineers. Use a practical tone.",
            "context_json": {},
            "priority": 0,
        },
    )
    assert second.status_code == 200
    second_body = second.json()
    assert second_body["assistant_message"]["action"]["type"] == "ask_clarification"
    assert "path" not in second_body["session"]["metadata"]["context_json"]
    assert "audience" not in second_body["session"]["metadata"]["context_json"]
    assert "tone" not in second_body["session"]["metadata"]["context_json"]

    third = client.post(
        f"/chat/sessions/{session['id']}/messages",
        json={"content": "medic.docx", "context_json": {}, "priority": 0},
    )

    assert third.status_code == 200
    third_body = third.json()
    assert third_body["job"] is not None
    assert third_body["assistant_message"]["action"]["type"] == "submit_job"
    assert third_body["job"]["context_json"]["topic"] == "Kubernetes deployment report"
    assert third_body["job"]["context_json"]["audience"] == "Senior AI engineers"
    assert third_body["job"]["context_json"]["tone"] == "practical"
    assert third_body["job"]["context_json"]["path"] == "medic.docx"


def test_chat_submit_normalization_prefers_pending_clarification_slot_ledger(
    monkeypatch,
) -> None:
    def _normalize_goal_intent(goal, **kwargs):
        context_envelope = kwargs.get("context_envelope")
        intent_context = (
            main.context_service.intent_context_view(context_envelope)
            if context_envelope is not None
            else {}
        )
        missing_fields = [
            field
            for field in ("topic", "audience", "tone", "path")
            if not str(intent_context.get(field) or "").strip()
        ]
        questions = [
            main.chat_clarification_normalizer.clarification_question_for_field(field, goal=goal)
            for field in missing_fields
        ]
        return main.workflow_contracts.NormalizedIntentEnvelope(
            goal=goal,
            profile=main.workflow_contracts.GoalIntentProfile(
                intent="generate",
                source="test",
                confidence=0.97,
                risk_level="bounded_write",
                threshold=0.7,
                low_confidence=False,
                needs_clarification=bool(missing_fields),
                requires_blocking_clarification=bool(missing_fields),
                questions=questions,
                blocking_slots=list(missing_fields),
                missing_slots=list(missing_fields),
                slot_values={"intent_action": "generate", "risk_level": "bounded_write"},
                clarification_mode="capability_required_inputs" if missing_fields else None,
            ),
            graph=main.workflow_contracts.IntentGraph(
                segments=[
                    main.workflow_contracts.IntentGraphSegment(
                        id="s1",
                        intent="generate",
                        objective="Generate document spec",
                        required_inputs=["instruction", "topic", "audience", "tone"],
                        suggested_capabilities=["document.spec.generate"],
                    ),
                    main.workflow_contracts.IntentGraphSegment(
                        id="s2",
                        intent="render",
                        objective="Render DOCX",
                        required_inputs=["document_spec", "output_filename"],
                        suggested_capabilities=["document.docx.render"],
                    ),
                ]
            ),
            candidate_capabilities={
                "s1": ["document.spec.generate"],
                "s2": ["document.docx.render"],
            },
        )

    monkeypatch.setattr(main, "_normalize_goal_intent", _normalize_goal_intent)
    monkeypatch.setattr(main, "CHAT_CLARIFICATION_NORMALIZER_ENABLED", False)

    pending_state = main.chat_contracts.ClarificationState(
        original_goal="Create a Kubernetes deployment report",
        active_family="documents",
        active_segment_id="s1",
        active_capability_id="document.spec.generate",
        questions=[],
        pending_fields=[],
        required_fields=["topic", "audience", "tone", "path"],
        known_slot_values={
            "topic": "Kubernetes deployment report",
            "audience": "Senior AI engineers",
            "tone": "practical",
            "path": "artifacts/medic.docx",
        },
        resolved_slots={
            "topic": "Kubernetes deployment report",
            "audience": "Senior AI engineers",
            "tone": "practical",
            "path": "artifacts/medic.docx",
        },
        slot_provenance={
            "topic": "explicit_user",
            "audience": "explicit_user",
            "tone": "clarification_normalized",
            "path": "explicit_user",
        },
    ).model_dump(mode="json", exclude_none=True)
    session_metadata = {
        "draft_goal": "Create a Kubernetes deployment report",
        "pending_clarification": pending_state,
    }

    db = SessionLocal()
    try:
        context_envelope = main.context_service.build_chat_context_envelope(
            db=db,
            goal="Create a Kubernetes deployment report\n\nUser clarification: submit it now",
            session_metadata=session_metadata,
            session_context={},
            turn_context={},
            user_id=None,
        )
        result = main._normalize_chat_submit_context(
            db=db,
            goal="Create a Kubernetes deployment report\n\nUser clarification: submit it now",
            content="submit it now",
            session_metadata=session_metadata,
            merged_context={},
            context_envelope=context_envelope,
            user_id=None,
            messages=[],
        )
    finally:
        db.close()

    assert result is None
    submit_view = main.context_service.chat_submit_context_view(context_envelope)
    assert submit_view["topic"] == "Kubernetes deployment report"
    assert submit_view["audience"] == "Senior AI engineers"
    assert submit_view["tone"] == "practical"
    assert submit_view["path"] == "artifacts/medic.docx"


def test_chat_turn_normalizes_using_selected_github_capability_contract(monkeypatch) -> None:
    calls = {"count": 0}

    class _Router:
        def generate_request_json_object(self, request):
            calls["count"] += 1
            if calls["count"] == 1:
                return {
                    "route": "ask_clarification",
                    "assistant_response": "What GitHub issues should I search for?",
                    "intent": "io",
                    "risk_level": "read_only",
                    "confidence": 0.95,
                    "clarification_questions": ["What GitHub issues should I search for?"],
                }
            return {
                "route": "submit_job",
                "assistant_response": "",
                "intent": "io",
                "risk_level": "read_only",
                "confidence": 0.95,
            }

    def _normalize_goal_intent(goal, **_kwargs):
        return main.workflow_contracts.NormalizedIntentEnvelope(
            goal=goal,
            profile=main.workflow_contracts.GoalIntentProfile(
                intent="io",
                source="test",
                confidence=0.95,
                risk_level="read_only",
                threshold=0.7,
                low_confidence=False,
                needs_clarification=False,
                requires_blocking_clarification=False,
                questions=[],
                blocking_slots=[],
                missing_slots=[],
                slot_values={"intent_action": "io", "risk_level": "read_only"},
            ),
            graph=main.workflow_contracts.IntentGraph(
                segments=[
                    main.workflow_contracts.IntentGraphSegment(
                        id="s1",
                        intent="io",
                        objective="Search GitHub issues",
                        required_inputs=["query"],
                        suggested_capabilities=["github.issue.search"],
                    )
                ]
            ),
            candidate_capabilities={"s1": ["github.issue.search"]},
        )

    class _Normalizer:
        def generate_request_json_object(self, request):
            assert request.metadata == {"component": "chat_clarification_normalizer"}
            return {
                "normalized_slots": {"query": "authentication failures in private repos"},
                "field_confidence": {"query": 0.93},
                "unresolved_fields": [],
            }

    monkeypatch.setattr(main, "_normalize_goal_intent", _normalize_goal_intent)
    monkeypatch.setattr(main, "_chat_router_provider", _Router())
    monkeypatch.setattr(main, "_chat_clarification_normalizer_provider", _Normalizer())
    monkeypatch.setattr(main, "CHAT_CLARIFICATION_NORMALIZER_ENABLED", True)

    session = client.post("/chat/sessions", json={}).json()
    first = client.post(
        f"/chat/sessions/{session['id']}/messages",
        json={"content": "Search GitHub issues", "context_json": {}, "priority": 0},
    )
    assert first.status_code == 200
    assert first.json()["assistant_message"]["action"]["type"] == "ask_clarification"

    response = client.post(
        f"/chat/sessions/{session['id']}/messages",
        json={
            "content": "authentication failures in private repos",
            "context_json": {},
            "priority": 0,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["job"] is not None
    assert body["assistant_message"]["action"]["type"] == "submit_job"
    assert body["job"]["context_json"]["query"] == "authentication failures in private repos"
    assert body["job"]["context_json"]["clarification_normalization"]["capability_id"] == (
        "github.issue.search"
    )


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
        json={
            "content": "Can you prepare a deployment report?",
            "context_json": {
                "topic": "Deployment report",
                "path": "artifacts/deployment-report.pdf",
            },
            "priority": 0,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["job"] is not None
    assert body["assistant_message"]["action"]["type"] == "submit_job"


def test_chat_boundary_uses_semantic_capability_evidence_and_persists_decision(
    monkeypatch,
) -> None:
    registry = cap_registry.CapabilityRegistry(
        capabilities={
            "document.spec.generate": cap_registry.CapabilitySpec(
                capability_id="document.spec.generate",
                description="Generate a structured document spec from an instruction.",
                risk_tier="read_only",
                idempotency="read",
                group="documents",
                subgroup="generation",
                enabled=True,
            ),
            "document.docx.render": cap_registry.CapabilitySpec(
                capability_id="document.docx.render",
                description="Render a DOCX document from a DocumentSpec.",
                risk_tier="bounded_write",
                idempotency="safe_write",
                group="documents",
                subgroup="rendering",
                enabled=True,
            ),
            "github.repo.list": cap_registry.CapabilitySpec(
                capability_id="github.repo.list",
                description="List repositories.",
                risk_tier="read_only",
                idempotency="read",
                group="github",
                subgroup="repositories",
                enabled=True,
            ),
        }
    )

    class _Router:
        def generate_request_json_object(self, request):
            assert request.metadata["component"] == "chat_router"
            return {
                "route": "submit_job",
                "assistant_response": "",
                "intent": "generate",
                "risk_level": "bounded_write",
                "confidence": 0.95,
                "output_format": "docx",
            }

    class _Responder:
        def generate_request_json_object(self, request):
            assert request.metadata == {"component": "chat_boundary_decision"}
            payload = json.loads(request.prompt)
            evidence = payload["boundary_evidence"]
            assert evidence["conversation_mode_hint"] == "execution_oriented"
            assert any(
                capability["capability_id"].startswith("document.")
                for capability in evidence["top_capabilities"]
            )
            assert any(family["family"] == "documents" for family in evidence["top_families"])
            assert evidence["intent"] == "generate"
            return {
                "decision": "execution_request",
                "assistant_response": "",
                "reason_code": "semantic_capability_evidence",
            }

    monkeypatch.setattr(main.capability_registry, "load_capability_registry", lambda: registry)
    monkeypatch.setattr(main.capability_registry, "resolve_capability_mode", lambda: "enabled")
    monkeypatch.setattr(
        main.capability_registry,
        "evaluate_capability_allowlist",
        lambda capability_id, service_name=None: cap_registry.CapabilityAllowDecision(
            allowed=True,
            reason="allowed",
        ),
    )
    monkeypatch.setattr(main, "CHAT_CAPABILITY_VECTOR_SEARCH_ENABLED", False)
    monkeypatch.setattr(main, "CHAT_INTENT_VECTOR_SEARCH_ENABLED", False)
    monkeypatch.setattr(main, "_chat_intent_vector_synced_namespace", None)
    monkeypatch.setattr(main, "CHAT_ROUTING_MODE", "response_first")
    monkeypatch.setattr(main, "CHAT_RESPONSE_MODE", "answer_or_handoff")
    monkeypatch.setattr(main, "_chat_router_provider", _Router())
    monkeypatch.setattr(main, "_chat_response_provider", _Responder())

    before_metrics = client.get("/metrics")
    assert before_metrics.status_code == 200
    decision_before = _metric_value(
        before_metrics.text,
        "chat_boundary_decisions_total",
        {
            "decision": "execution_request",
            "conversation_mode_hint": "execution_oriented",
            "pending_clarification": "false",
            "workflow_target_available": "false",
        },
    )
    reason_before = _metric_value(
        before_metrics.text,
        "chat_boundary_reason_total",
        {
            "decision": "execution_request",
            "reason_code": "semantic_capability_evidence",
        },
    )

    session = client.post("/chat/sessions", json={}).json()
    response = client.post(
        f"/chat/sessions/{session['id']}/messages",
        json={"content": "create a document on Kubernetes", "context_json": {}, "priority": 0},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["assistant_message"]["action"]["type"] in {"submit_job", "ask_clarification"}
    assert (
        body["assistant_message"]["metadata"]["boundary_decision"]["reason_code"]
        == "semantic_capability_evidence"
    )
    assert (
        body["assistant_message"]["metadata"]["boundary_decision"]["evidence"]["intent"]
        == "generate"
    )
    assert any(
        family["family"] == "documents"
        for family in body["assistant_message"]["metadata"]["boundary_decision"]["evidence"][
            "top_families"
        ]
    )

    after_metrics = client.get("/metrics")
    assert after_metrics.status_code == 200
    assert (
        _metric_value(
            after_metrics.text,
            "chat_boundary_decisions_total",
            {
                "decision": "execution_request",
                "conversation_mode_hint": "execution_oriented",
                "pending_clarification": "false",
                "workflow_target_available": "false",
            },
        )
        >= decision_before + 1
    )
    assert (
        _metric_value(
            after_metrics.text,
            "chat_boundary_reason_total",
            {
                "decision": "execution_request",
                "reason_code": "semantic_capability_evidence",
            },
        )
        >= reason_before + 1
    )


def test_chat_boundary_capability_evidence_scopes_to_active_family(
    monkeypatch,
) -> None:
    registry = cap_registry.CapabilityRegistry(
        capabilities={
            "document.spec.generate": cap_registry.CapabilitySpec(
                capability_id="document.spec.generate",
                description="Generate a structured document spec from an instruction.",
                risk_tier="bounded_write",
                idempotency="write",
                group="documents",
                subgroup="generation",
                enabled=True,
                planner_hints={"required_inputs": ["instruction", "topic", "audience", "tone"]},
            ),
            "document.docx.render": cap_registry.CapabilitySpec(
                capability_id="document.docx.render",
                description="Render a DOCX document from a DocumentSpec to a path.",
                risk_tier="bounded_write",
                idempotency="safe_write",
                group="documents",
                subgroup="rendering",
                enabled=True,
                planner_hints={"required_inputs": ["document_spec", "path"]},
            ),
            "github.issue.search": cap_registry.CapabilitySpec(
                capability_id="github.issue.search",
                description="Search GitHub issues by query.",
                risk_tier="read_only",
                idempotency="read",
                group="github",
                subgroup="issues",
                enabled=True,
                planner_hints={"required_inputs": ["query"]},
            ),
        }
    )

    monkeypatch.setattr(main.capability_registry, "load_capability_registry", lambda: registry)
    monkeypatch.setattr(main.capability_registry, "resolve_capability_mode", lambda: "enabled")
    monkeypatch.setattr(
        main.capability_registry,
        "evaluate_capability_allowlist",
        lambda capability_id, service_name=None: cap_registry.CapabilityAllowDecision(
            allowed=True,
            reason="allowed",
        ),
    )
    monkeypatch.setattr(main, "CHAT_CAPABILITY_VECTOR_SEARCH_ENABLED", False)

    top_capabilities, top_families = main._chat_boundary_capability_evidence(
        query="search github issues",
        preferred_family="documents",
        preferred_capability_ids=["document.docx.render"],
        pending_fields=["path"],
    )

    assert top_capabilities
    assert all(capability.capability_id.startswith("document.") for capability in top_capabilities)
    assert top_families
    assert top_families[0].family == "documents"
    assert all(
        capability_id.startswith("document.")
        for capability_id in top_families[0].capability_ids
    )


def test_chat_boundary_overrides_chat_reply_when_execution_signal_is_strong(
    monkeypatch,
) -> None:
    calls = {"router": 0}

    class _Router:
        def generate_request_json_object(self, request):
            calls["router"] += 1
            return {
                "route": "submit_job",
                "assistant_response": "",
                "intent": "generate",
                "risk_level": "bounded_write",
                "confidence": 0.95,
                "output_format": "docx",
            }

    class _Responder:
        def generate_request_json_object(self, request):
            payload = json.loads(request.prompt)
            evidence = payload["boundary_evidence"]
            assert evidence["execution_signal_strength"] == "strong"
            assert evidence["top_family_score"] == 1.73
            return {
                "decision": "chat_reply",
                "assistant_response": "Let me explain what I can do.",
                "reason_code": "model_chat_bias",
            }

    monkeypatch.setattr(
        main,
        "_chat_boundary_capability_evidence",
        lambda **_kwargs: (
            [
                main.chat_contracts.ChatBoundaryCapabilityEvidence(
                    capability_id="document.spec.generate",
                    group="documents",
                    subgroup="generation",
                    score=0.91,
                    source="test",
                ),
                main.chat_contracts.ChatBoundaryCapabilityEvidence(
                    capability_id="document.docx.render",
                    group="documents",
                    subgroup="rendering",
                    score=0.82,
                    source="test",
                ),
            ],
            [
                main.chat_contracts.ChatBoundaryFamilyEvidence(
                    family="documents",
                    score=1.73,
                    capability_ids=["document.spec.generate", "document.docx.render"],
                )
            ],
        ),
    )
    monkeypatch.setattr(
        main,
        "_chat_boundary_intent_evidence",
        lambda **_kwargs: main.workflow_contracts.GoalIntentProfile(
            intent="generate",
            risk_level="bounded_write",
        ),
    )
    monkeypatch.setattr(main, "CHAT_ROUTING_MODE", "response_first")
    monkeypatch.setattr(main, "CHAT_RESPONSE_MODE", "answer_or_handoff")
    monkeypatch.setattr(main, "_chat_router_provider", _Router())
    monkeypatch.setattr(main, "_chat_response_provider", _Responder())

    session = client.post("/chat/sessions", json={}).json()
    response = client.post(
        f"/chat/sessions/{session['id']}/messages",
        json={"content": "create a document on Kubernetes", "context_json": {}, "priority": 0},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["assistant_message"]["action"]["type"] in {"submit_job", "ask_clarification"}
    assert body["assistant_message"]["metadata"]["boundary_decision"]["decision"] == "execution_request"
    assert (
        body["assistant_message"]["metadata"]["boundary_decision"]["reason_code"]
        == "execution_signal_override"
    )
    assert (
        body["assistant_message"]["metadata"]["boundary_decision"]["evidence"][
            "execution_signal_strength"
        ]
        == "strong"
    )
    assert calls["router"] == 1


def test_chat_boundary_overrides_chat_reply_to_continue_pending_for_clarification_answer(
    monkeypatch,
) -> None:
    session = client.post("/chat/sessions", json={}).json()
    first_response = client.post(
        f"/chat/sessions/{session['id']}/messages",
        json={"content": "Generate a deployment report", "context_json": {}, "priority": 0},
    )
    assert first_response.status_code == 200
    assert first_response.json()["assistant_message"]["action"]["type"] == "ask_clarification"

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
                "output_format": "docx",
            }

    class _Responder:
        def generate_request_json_object(self, request):
            payload = json.loads(request.prompt)
            evidence = payload["boundary_evidence"]
            assert evidence["pending_clarification"] is True
            assert evidence["likely_clarification_answer"] is True
            assert evidence["conversation_mode_hint"] == "clarification_answer"
            return {
                "decision": "chat_reply",
                "assistant_response": "Sure, let's just talk here.",
                "reason_code": "model_chat_bias",
            }

    monkeypatch.setattr(main, "CHAT_ROUTING_MODE", "response_first")
    monkeypatch.setattr(main, "CHAT_RESPONSE_MODE", "answer_or_handoff")
    monkeypatch.setattr(main, "_chat_router_provider", _Router())
    monkeypatch.setattr(main, "_chat_response_provider", _Responder())

    response = client.post(
        f"/chat/sessions/{session['id']}/messages",
        json={"content": "docx is fine", "context_json": {}, "priority": 0},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["assistant_message"]["action"]["type"] in {"submit_job", "ask_clarification"}
    assert body["assistant_message"]["metadata"]["boundary_decision"]["decision"] == "continue_pending"
    assert (
        body["assistant_message"]["metadata"]["boundary_decision"]["reason_code"]
        == "clarification_answer_override"
    )
    assert calls["router"] == 1


def test_chat_boundary_preserves_pending_clarification_for_execution_oriented_long_answer(
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
    assert first_body["session"]["metadata"]["pending_clarification"]["original_goal"] == (
        "Generate a deployment report"
    )

    calls = {"router": 0}

    class _Router:
        def generate_request_json_object(self, request):
            calls["router"] += 1
            return {
                "route": "ask_clarification",
                "assistant_response": "",
                "intent": "render",
                "risk_level": "bounded_write",
                "confidence": 0.95,
                "clarification_questions": ["What tone should it use?"],
                "missing_slots": ["tone"],
            }

    class _Responder:
        def generate_request_json_object(self, request):
            payload = json.loads(request.prompt)
            evidence = payload["boundary_evidence"]
            assert evidence["pending_clarification"] is True
            assert evidence["likely_clarification_answer"] is False
            assert evidence["conversation_mode_hint"] == "execution_oriented"
            return {
                "decision": "chat_reply",
                "assistant_response": "Let me explain Kubernetes automation instead.",
                "reason_code": "model_chat_bias",
            }

    monkeypatch.setattr(main, "CHAT_ROUTING_MODE", "response_first")
    monkeypatch.setattr(main, "CHAT_RESPONSE_MODE", "answer_or_handoff")
    monkeypatch.setattr(main, "_chat_router_provider", _Router())
    monkeypatch.setattr(main, "_chat_response_provider", _Responder())

    response = client.post(
        f"/chat/sessions/{session['id']}/messages",
        json={
            "content": (
                "Create automation to run in cluster to document how to deploy AI workflows "
                "to Kubernetes. The file name should be Deployment of AI workflows to "
                "Kubernetes.docx"
            ),
            "context_json": {},
            "priority": 0,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["job"] is None
    assert body["assistant_message"]["action"]["type"] == "ask_clarification"
    assert body["assistant_message"]["metadata"]["boundary_decision"]["decision"] == "continue_pending"
    assert (
        body["assistant_message"]["metadata"]["boundary_decision"]["reason_code"]
        == "pending_clarification_state_preservation"
    )
    assert body["session"]["metadata"]["pending_clarification"]["original_goal"] == (
        "Generate a deployment report"
    )
    assert "pending_clarification" in body["session"]["metadata"]
    assert calls["router"] == 1


def test_chat_boundary_coerces_non_pending_meta_clarification_back_to_execution(
    monkeypatch,
) -> None:
    calls = {"router": 0}

    class _Router:
        def generate_request_json_object(self, request):
            calls["router"] += 1
            return {
                "route": "ask_clarification",
                "assistant_response": "",
                "intent": "generate",
                "risk_level": "bounded_write",
                "confidence": 0.95,
                "clarification_questions": [
                    "What output format do you need (for example PDF, DOCX, JSON, or Markdown)?"
                ],
                "missing_slots": ["output_format"],
            }

    class _Responder:
        def generate_request_json_object(self, request):
            payload = json.loads(request.prompt)
            evidence = payload["boundary_evidence"]
            assert evidence["pending_clarification"] is False
            assert evidence["needs_clarification"] is True
            assert evidence["conversation_mode_hint"] == "execution_oriented"
            return {
                "decision": "meta_clarification",
                "assistant_response": "Could you specify the desired output format?",
                "reason_code": "missing_output_format",
            }

    monkeypatch.setattr(main, "CHAT_ROUTING_MODE", "response_first")
    monkeypatch.setattr(main, "CHAT_RESPONSE_MODE", "answer_or_handoff")
    monkeypatch.setattr(main, "_chat_router_provider", _Router())
    monkeypatch.setattr(main, "_chat_response_provider", _Responder())

    session = client.post("/chat/sessions", json={}).json()
    response = client.post(
        f"/chat/sessions/{session['id']}/messages",
        json={"content": "create a document on Kubernetes", "context_json": {}, "priority": 0},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["job"] is None
    assert body["assistant_message"]["action"]["type"] == "ask_clarification"
    assert body["assistant_message"]["metadata"]["boundary_decision"]["decision"] == "execution_request"
    assert (
        body["assistant_message"]["metadata"]["boundary_decision"]["reason_code"]
        == "non_pending_meta_clarification_execution_override"
    )
    assert body["session"]["metadata"]["pending_clarification"]["original_goal"] == (
        "create a document on Kubernetes"
    )
    assert calls["router"] == 1


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


def test_chat_route_goal_intent_profile_preserves_intent_disagreement_question(monkeypatch) -> None:
    monkeypatch.setattr(
        main,
        "CHAT_PRE_SUBMIT_BLOCKING_SLOTS",
        {"output_format", "target_system", "safety_constraints"},
    )
    profile = main.workflow_contracts.GoalIntentProfile(
        intent="io",
        source="llm",
        confidence=0.41,
        risk_level="read_only",
        threshold=0.7,
        low_confidence=True,
        needs_clarification=True,
        requires_blocking_clarification=True,
        questions=["Should I generate new content, or should I fetch, inspect, or list data for this request?"],
        blocking_slots=["intent_action"],
        missing_slots=["intent_action"],
        slot_values={"intent_action": "io"},
        clarification_mode="intent_disagreement",
    )

    narrowed = main._chat_route_goal_intent_profile(profile, goal="Create a release report")

    assert narrowed.blocking_slots == ["intent_action"]
    assert narrowed.missing_slots == ["intent_action"]
    assert narrowed.questions == profile.questions


def test_chat_turn_asks_clarification_for_intent_disagreement(monkeypatch) -> None:
    session = client.post("/chat/sessions", json={}).json()

    monkeypatch.setattr(
        main,
        "_normalize_goal_intent",
        lambda *args, **kwargs: main.workflow_contracts.NormalizedIntentEnvelope(
            goal="Create a release report",
            profile=main.workflow_contracts.GoalIntentProfile(
                intent="io",
                source="llm",
                confidence=0.41,
                risk_level="read_only",
                threshold=0.7,
                low_confidence=True,
                needs_clarification=True,
                requires_blocking_clarification=True,
                questions=[
                    "Should I generate new content, or should I fetch, inspect, or list data for this request?"
                ],
                blocking_slots=["intent_action"],
                missing_slots=["intent_action"],
                slot_values={"intent_action": "io"},
                clarification_mode="intent_disagreement",
            ),
            clarification=main.workflow_contracts.ClarificationState(
                needs_clarification=True,
                requires_blocking_clarification=True,
                missing_inputs=["intent_action"],
                questions=[
                    "Should I generate new content, or should I fetch, inspect, or list data for this request?"
                ],
                blocking_slots=["intent_action"],
                slot_values={"intent_action": "io"},
                clarification_mode="intent_disagreement",
            ),
            trace=main.workflow_contracts.NormalizationTrace(
                assessment_source="llm",
                context_projection="intent",
                disagreement={"reason_code": "capability_intent_conflict"},
            ),
        ),
    )

    response = client.post(
        f"/chat/sessions/{session['id']}/messages",
        json={"content": "Create a release report", "context_json": {}, "priority": 0},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["assistant_message"]["action"]["type"] == "ask_clarification"
    assert body["assistant_message"]["content"] == (
        "Should I generate new content, or should I fetch, inspect, or list data for this request?"
    )
    assert body["session"]["metadata"]["pending_clarification"]["questions"] == [
        "Should I generate new content, or should I fetch, inspect, or list data for this request?"
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


def test_chat_turn_pending_boundary_failure_returns_ask_clarification(monkeypatch) -> None:
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
    assert body["assistant_message"]["action"]["type"] == "ask_clarification"
    assert "continue the current workflow request" in body["assistant_message"]["content"].lower()
    assert body["session"]["metadata"]["pending_clarification"]["questions"]


def test_chat_turn_pending_meta_clarification_uses_ask_clarification_path(
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
            raise AssertionError("router should not run when pending clarification stays ambiguous")

    class _Responder:
        def generate_request_json_object(self, request):
            assert request.metadata == {"component": "chat_boundary_decision"}
            return {
                "decision": "meta_clarification",
                "assistant_response": (
                    "Do you want to continue the current workflow request, or should I answer here in chat instead?"
                ),
                "reason_code": "ambiguous_pending_direction",
            }

    monkeypatch.setattr(main, "CHAT_RESPONSE_MODE", "answer_or_handoff")
    monkeypatch.setattr(main, "CHAT_ROUTING_MODE", "response_first")
    monkeypatch.setattr(main, "_chat_router_provider", _Router())
    monkeypatch.setattr(main, "_chat_response_provider", _Responder())

    response = client.post(
        f"/chat/sessions/{session['id']}/messages",
        json={
            "content": "Can you clarify what you still need from me?",
            "context_json": {},
            "priority": 0,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["job"] is None
    assert body["assistant_message"]["action"]["type"] == "ask_clarification"
    assert "continue the current workflow request" in body["assistant_message"]["content"].lower()
    assert body["session"]["metadata"]["pending_clarification"]["questions"] == [
        "Do you want to continue the current workflow request, or should I answer here in chat instead?"
    ]


def test_chat_turn_router_execution_question_does_not_stay_in_respond(
    monkeypatch,
) -> None:
    class _Router:
        def generate_request_json_object(self, request):
            assert request.metadata["component"] == "chat_router"
            return {
                "route": "respond",
                "assistant_response": "What output format do you need?",
                "intent": "generate",
                "risk_level": "bounded_write",
                "confidence": 0.96,
                "clarification_questions": ["What output format do you need?"],
                "missing_slots": ["output_format"],
            }

    monkeypatch.setattr(main, "CHAT_RESPONSE_MODE", "answer_or_handoff")
    monkeypatch.setattr(main, "CHAT_ROUTING_MODE", "always_router")
    monkeypatch.setattr(main, "_chat_router_provider", _Router())

    session = client.post("/chat/sessions", json={}).json()
    response = client.post(
        f"/chat/sessions/{session['id']}/messages",
        json={"content": "Create a deployment report", "context_json": {}, "priority": 0},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["job"] is None
    assert body["assistant_message"]["action"]["type"] == "ask_clarification"
    assert body["assistant_message"]["content"] == "What output format do you need?"
    assert body["session"]["metadata"]["pending_clarification"]["questions"] == [
        "What output format do you need?"
    ]


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


def test_chat_turn_profile_update_uses_authenticated_user_id_not_context_user_id() -> None:
    headers = {"X-Authenticated-User-Id": "alice"}
    session = client.post("/chat/sessions", json={}, headers=headers).json()

    response = client.post(
        f"/chat/sessions/{session['id']}/messages",
        json={
            "content": "I prefer markdown and concise responses.",
            "context_json": {"user_id": "mallory"},
            "priority": 0,
        },
        headers=headers,
    )

    assert response.status_code == 200
    body = response.json()
    assert body["session"]["metadata"].get("_chat_user_id") is None
    if isinstance(body["session"]["metadata"].get("context_json"), dict):
        assert "user_id" not in body["session"]["metadata"]["context_json"]

    db = SessionLocal()
    try:
        alice_profile = memory_profile_service.load_user_profile(db, "alice")
        mallory_profile = memory_profile_service.load_user_profile(db, "mallory")
    finally:
        db.close()

    assert alice_profile.preferences.preferred_output_format == "markdown"
    assert alice_profile.preferences.response_verbosity == "concise"
    assert mallory_profile.preferences.preferred_output_format is None


def test_chat_route_context_includes_profile_without_persisting_it(monkeypatch) -> None:
    db = SessionLocal()
    try:
        memory_profile_service.write_user_profile(
            db,
            user_id="alice",
            payload={"preferences": {"preferred_output_format": "markdown"}},
        )
    finally:
        db.close()

    def _route_turn(*, content, candidate_goal, session_metadata, merged_context, messages):
        del content, candidate_goal, session_metadata, messages
        assert merged_context["user_profile"]["preferences"]["preferred_output_format"] == "markdown"
        return {
            "type": "respond",
            "assistant_content": "Profile-aware route.",
            "goal_intent_profile": {},
            "resolved_goal": "Profile-aware route.",
        }

    monkeypatch.setattr(main, "_route_chat_turn", _route_turn)
    headers = {"X-Authenticated-User-Id": "alice"}
    session = client.post("/chat/sessions", json={}, headers=headers).json()

    response = client.post(
        f"/chat/sessions/{session['id']}/messages",
        json={"content": "hello", "context_json": {}, "priority": 0},
        headers=headers,
    )

    assert response.status_code == 200
    body = response.json()
    assert body["assistant_message"]["content"] == "Profile-aware route."
    if isinstance(body["session"]["metadata"].get("context_json"), dict):
        assert "user_profile" not in body["session"]["metadata"]["context_json"]


def test_chat_session_rejects_authenticated_user_mismatch() -> None:
    alice_headers = {"X-Authenticated-User-Id": "alice"}
    bob_headers = {"X-Authenticated-User-Id": "bob"}
    session = client.post("/chat/sessions", json={}, headers=alice_headers).json()

    get_response = client.get(f"/chat/sessions/{session['id']}", headers=bob_headers)
    message_response = client.post(
        f"/chat/sessions/{session['id']}/messages",
        json={"content": "hello", "context_json": {}, "priority": 0},
        headers=bob_headers,
    )

    assert get_response.status_code == 404
    assert message_response.status_code == 404
