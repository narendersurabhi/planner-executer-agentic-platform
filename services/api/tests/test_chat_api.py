import os
from datetime import UTC, datetime

from fastapi.testclient import TestClient

os.environ["DATABASE_URL"] = "sqlite:///./test.db"
os.environ["ORCHESTRATOR_ENABLED"] = "false"
os.environ["JOB_RECOVERY_ENABLED"] = "false"
os.environ["POLICY_GATE_ENABLED"] = "false"
os.environ["CAPABILITY_MODE"] = "enabled"

from libs.core import models  # noqa: E402
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

    class _Responder:
        def generate_request(self, request):
            assert request.metadata == {"component": "chat_response"}
            return type("_Response", (), {"content": "Direct response model answer."})()

    monkeypatch.setattr(main, "CHAT_ROUTING_MODE", "response_first")
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
    assert body["assistant_message"]["content"] == "Direct response model answer."


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
        def generate_request(self, request):
            raise AssertionError("response provider should not run for execution turns")

    monkeypatch.setattr(main, "CHAT_ROUTING_MODE", "response_first")
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
