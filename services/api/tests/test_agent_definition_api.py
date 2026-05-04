import os
import uuid

from fastapi.testclient import TestClient

os.environ["DATABASE_URL"] = "sqlite:///./test.db"
os.environ["ORCHESTRATOR_ENABLED"] = "false"
os.environ["JOB_RECOVERY_ENABLED"] = "false"
os.environ["POLICY_GATE_ENABLED"] = "false"
os.environ["CAPABILITY_MODE"] = "enabled"

from libs.core import capability_registry as cap_registry  # noqa: E402
from services.api.app import main  # noqa: E402
from services.api.app.database import Base, engine  # noqa: E402


Base.metadata.create_all(bind=engine)

client = TestClient(main.app)


def _agent_registry() -> cap_registry.CapabilityRegistry:
    return cap_registry.CapabilityRegistry(
        capabilities={
            "codegen.autonomous": cap_registry.CapabilitySpec(
                capability_id="codegen.autonomous",
                description="Plan and implement code changes autonomously.",
                enabled=True,
                risk_tier="high_risk",
                idempotency="unsafe_write",
                group="codegen",
                subgroup="generation",
                tags=("coding-agent", "autonomous"),
                adapters=(
                    cap_registry.CapabilityAdapterSpec(
                        type="tool",
                        server_id="local_worker",
                        tool_name="coding_agent_autonomous",
                    ),
                ),
            ),
            "filesystem.workspace.list": cap_registry.CapabilitySpec(
                capability_id="filesystem.workspace.list",
                description="List files in the workspace.",
                enabled=True,
                risk_tier="read_only",
                idempotency="read",
                group="filesystem",
                subgroup="workspace",
                adapters=(
                    cap_registry.CapabilityAdapterSpec(
                        type="tool",
                        server_id="local_worker",
                        tool_name="filesystem.workspace.list",
                    ),
                ),
            ),
            "document.docx.render": cap_registry.CapabilitySpec(
                capability_id="document.docx.render",
                description="Render a DOCX document.",
                enabled=True,
                risk_tier="bounded_write",
                idempotency="safe_write",
                group="document",
                subgroup="render",
                adapters=(
                    cap_registry.CapabilityAdapterSpec(
                        type="tool",
                        server_id="local_worker",
                        tool_name="document.docx.render",
                    ),
                ),
            ),
        }
    )


def _profile_agent_run_spec(
    *,
    primary_capability_id: str = "codegen.autonomous",
    extra_capability_id: str | None = "filesystem.workspace.list",
) -> dict:
    primary_request = {
        "request_id": primary_capability_id,
        "capability_id": primary_capability_id,
        "execution_request_id": primary_capability_id,
    }
    steps = [
        {
            "step_id": "primary_agent",
            "name": "PrimaryAgent",
            "description": "Run the primary agent",
            "instruction": "",
            "capability_request": primary_request,
            "input_bindings": {
                "goal": "",
                "workspace_path": "",
            },
            "depends_on": [],
        }
    ]
    capability_requests = [primary_request]
    dag_edges = []
    if extra_capability_id:
        extra_request = {
            "request_id": extra_capability_id,
            "capability_id": extra_capability_id,
            "execution_request_id": extra_capability_id,
        }
        steps.append(
            {
                "step_id": "extra_step",
                "name": "ExtraStep",
                "description": "Run an extra capability",
                "instruction": "Run the extra capability.",
                "capability_request": extra_request,
                "input_bindings": {},
                "depends_on": ["primary_agent"],
            }
        )
        capability_requests.append(extra_request)
        dag_edges.append(["primary_agent", "extra_step"])
    return {
        "version": "1",
        "kind": "studio",
        "planner_version": "workbench_v1",
        "tasks_summary": "",
        "steps": steps,
        "dag_edges": dag_edges,
        "capability_requests": capability_requests,
    }


def test_agent_definition_crud_and_soft_delete(monkeypatch) -> None:
    monkeypatch.setattr(main.capability_registry, "load_capability_registry", _agent_registry)
    user_id = f"agent-api-{uuid.uuid4()}"

    create_response = client.post(
        "/agents/definitions",
        json={
            "name": "  Codegen Autonomous  ",
            "description": "Reusable codegen agent",
            "agent_capability_id": "CODEGEN.AUTONOMOUS",
            "instructions": "Plan and implement the requested change.",
            "default_goal": "Implement the workspace task.",
            "default_workspace_path": " workbench-agent ",
            "default_constraints": ["run tests", "run tests", "keep scope tight"],
            "default_max_steps": 6,
            "model_config": {"provider": "openai", "model": "gpt-5.4"},
            "allowed_capability_ids": ["filesystem.workspace.list"],
            "memory_policy": {"mode": "disabled"},
            "guardrail_policy": {"schema": "strict"},
            "workspace_policy": {"write_roots": ["workbench-agent"]},
            "metadata": {"label": "engineering"},
            "user_id": user_id,
        },
    )

    assert create_response.status_code == 200
    created = create_response.json()
    agent_id = created["id"]
    assert created["name"] == "Codegen Autonomous"
    assert created["agent_capability_id"] == "codegen.autonomous"
    assert created["default_workspace_path"] == "workbench-agent"
    assert created["default_constraints"] == ["run tests", "keep scope tight"]
    assert created["model_config"] == {"provider": "openai", "model": "gpt-5.4"}
    assert created["allowed_capability_ids"] == ["filesystem.workspace.list"]
    assert created["enabled"] is True

    list_response = client.get("/agents/definitions", params={"user_id": user_id})
    assert list_response.status_code == 200
    listed = list_response.json()
    assert [item["id"] for item in listed] == [agent_id]

    get_response = client.get(f"/agents/definitions/{agent_id}")
    assert get_response.status_code == 200
    assert get_response.json()["id"] == agent_id

    update_response = client.put(
        f"/agents/definitions/{agent_id}",
        json={
            "name": "Codegen Autonomous v2",
            "default_max_steps": None,
            "allowed_capability_ids": [],
            "model_config": {"provider": "openai", "model": "gpt-5.4-mini"},
        },
    )
    assert update_response.status_code == 200
    updated = update_response.json()
    assert updated["name"] == "Codegen Autonomous v2"
    assert updated["default_max_steps"] is None
    assert updated["allowed_capability_ids"] == []
    assert updated["model_config"] == {"provider": "openai", "model": "gpt-5.4-mini"}

    delete_response = client.delete(f"/agents/definitions/{agent_id}")
    assert delete_response.status_code == 200
    assert delete_response.json() == {"ok": True}

    default_list_response = client.get("/agents/definitions", params={"user_id": user_id})
    assert default_list_response.status_code == 200
    assert default_list_response.json() == []

    disabled_list_response = client.get(
        "/agents/definitions",
        params={"user_id": user_id, "include_disabled": "true"},
    )
    assert disabled_list_response.status_code == 200
    assert [item["id"] for item in disabled_list_response.json()] == [agent_id]
    assert disabled_list_response.json()[0]["enabled"] is False


def test_agent_definition_publish_versions_are_immutable(monkeypatch) -> None:
    monkeypatch.setattr(main.capability_registry, "load_capability_registry", _agent_registry)
    user_id = f"agent-version-{uuid.uuid4()}"

    create_response = client.post(
        "/agents/definitions",
        json={
            "name": "Versioned codegen profile",
            "description": "Before publish",
            "agent_capability_id": "codegen.autonomous",
            "instructions": "Use the published instructions.",
            "default_goal": "Use the published goal.",
            "default_workspace_path": "published-workspace",
            "default_constraints": ["published constraint"],
            "default_max_steps": 3,
            "model_config": {"model": "published"},
            "allowed_capability_ids": ["filesystem.workspace.list"],
            "metadata": {"draft": "before"},
            "user_id": user_id,
        },
    )
    assert create_response.status_code == 200
    agent_id = create_response.json()["id"]

    publish_response = client.post(
        f"/agents/definitions/{agent_id}/versions",
        json={
            "version_note": "Initial publish",
            "published_by": "phase-five-test",
            "metadata": {"channel": "staging"},
        },
    )
    assert publish_response.status_code == 200
    version = publish_response.json()
    version_id = version["id"]
    assert version["agent_definition_id"] == agent_id
    assert version["version_number"] == 1
    assert version["description"] == "Before publish"
    assert version["default_goal"] == "Use the published goal."
    assert version["model_config"] == {"model": "published"}
    assert version["definition_metadata"] == {"draft": "before"}
    assert version["version_metadata"] == {"channel": "staging"}
    assert version["published_by"] == "phase-five-test"
    assert version["version_note"] == "Initial publish"

    update_response = client.put(
        f"/agents/definitions/{agent_id}",
        json={
            "description": "After publish",
            "instructions": "Use changed instructions.",
            "default_goal": "Use changed goal.",
            "default_workspace_path": "changed-workspace",
            "default_constraints": ["changed constraint"],
            "default_max_steps": 8,
            "model_config": {"model": "changed"},
            "metadata": {"draft": "after"},
        },
    )
    assert update_response.status_code == 200

    list_response = client.get(f"/agents/definitions/{agent_id}/versions")
    assert list_response.status_code == 200
    assert [item["version_number"] for item in list_response.json()] == [1]

    get_response = client.get(f"/agents/definitions/{agent_id}/versions/1")
    assert get_response.status_code == 200
    fetched_version = get_response.json()
    assert fetched_version["id"] == version_id
    assert fetched_version["description"] == "Before publish"
    assert fetched_version["default_goal"] == "Use the published goal."
    assert fetched_version["model_config"] == {"model": "published"}


def test_agent_definition_create_rejects_non_agentic_primary_capability(monkeypatch) -> None:
    monkeypatch.setattr(main.capability_registry, "load_capability_registry", _agent_registry)

    response = client.post(
        "/agents/definitions",
        json={
            "name": "Renderer",
            "agent_capability_id": "document.docx.render",
            "instructions": "Render a document.",
        },
    )

    assert response.status_code == 422
    assert "agent_definition_primary_capability_not_agentic" in response.json()["detail"]


def test_agent_definition_create_rejects_unknown_allowed_capability(monkeypatch) -> None:
    monkeypatch.setattr(main.capability_registry, "load_capability_registry", _agent_registry)

    response = client.post(
        "/agents/definitions",
        json={
            "name": "Codegen",
            "agent_capability_id": "codegen.autonomous",
            "instructions": "Plan and implement code.",
            "allowed_capability_ids": ["missing.capability"],
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"] == (
        "agent_definition_allowed_capability_not_found:missing.capability"
    )


def test_agent_definition_update_rejects_blank_required_fields(monkeypatch) -> None:
    monkeypatch.setattr(main.capability_registry, "load_capability_registry", _agent_registry)

    create_response = client.post(
        "/agents/definitions",
        json={
            "name": "Codegen",
            "agent_capability_id": "codegen.autonomous",
            "instructions": "Plan and implement code.",
            "user_id": f"agent-api-{uuid.uuid4()}",
        },
    )
    assert create_response.status_code == 200
    agent_id = create_response.json()["id"]

    response = client.put(
        f"/agents/definitions/{agent_id}",
        json={"name": "  ", "instructions": "  "},
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "agent_definition_name_required"


def test_workbench_agent_run_with_definition_applies_defaults_and_stores_snapshot(
    monkeypatch,
) -> None:
    monkeypatch.setattr(main.capability_registry, "load_capability_registry", _agent_registry)
    user_id = f"agent-run-{uuid.uuid4()}"

    create_response = client.post(
        "/agents/definitions",
        json={
            "name": "Codegen launch profile",
            "agent_capability_id": "codegen.autonomous",
            "instructions": "Use the saved agent profile instructions.",
            "default_goal": "Implement the saved profile goal.",
            "default_workspace_path": "workbench-agent",
            "default_constraints": ["keep the change scoped", "run focused tests"],
            "default_max_steps": 5,
            "model_config": {"provider": "openai", "model": "gpt-5.4"},
            "allowed_capability_ids": ["filesystem.workspace.list"],
            "user_id": user_id,
        },
    )
    assert create_response.status_code == 200
    agent_definition_id = create_response.json()["id"]

    launch_response = client.post(
        "/workbench/agent-runs",
        json={
            "agent_definition_id": agent_definition_id,
            "context_json": {"workspace": "demo"},
            "run_spec": _profile_agent_run_spec(),
        },
    )

    assert launch_response.status_code == 200
    body = launch_response.json()
    run_id = body["run"]["id"]
    first_step = body["run_spec"]["steps"][0]
    assert body["run"]["title"] == "Codegen launch profile"
    assert body["run"]["goal"] == "Implement the saved profile goal."
    assert body["run"]["user_id"] == user_id
    assert first_step["instruction"] == "Use the saved agent profile instructions."
    assert first_step["input_bindings"] == {
        "goal": "Implement the saved profile goal.",
        "workspace_path": "workbench-agent",
        "constraints": "keep the change scoped\nrun focused tests",
        "max_steps": 5,
    }
    assert body["run"]["metadata"]["agent_definition_id"] == agent_definition_id
    snapshot = body["run"]["metadata"]["agent_definition_snapshot"]
    assert snapshot["agent_definition_id"] == agent_definition_id
    assert snapshot["name"] == "Codegen launch profile"
    assert snapshot["model_config"] == {"provider": "openai", "model": "gpt-5.4"}
    assert snapshot["captured_at"]

    update_response = client.put(
        f"/agents/definitions/{agent_definition_id}",
        json={
            "name": "Changed profile name",
            "instructions": "Changed instructions.",
            "model_config": {"provider": "other"},
        },
    )
    assert update_response.status_code == 200

    run_response = client.get(f"/runs/{run_id}")
    assert run_response.status_code == 200
    persisted_snapshot = run_response.json()["metadata"]["agent_definition_snapshot"]
    assert persisted_snapshot["name"] == "Codegen launch profile"
    assert persisted_snapshot["instructions"] == "Use the saved agent profile instructions."
    assert persisted_snapshot["model_config"] == {"provider": "openai", "model": "gpt-5.4"}


def test_workbench_agent_run_with_definition_version_uses_published_snapshot(
    monkeypatch,
) -> None:
    monkeypatch.setattr(main.capability_registry, "load_capability_registry", _agent_registry)
    user_id = f"agent-version-launch-{uuid.uuid4()}"

    create_response = client.post(
        "/agents/definitions",
        json={
            "name": "Published launch profile",
            "agent_capability_id": "codegen.autonomous",
            "instructions": "Use published launch instructions.",
            "default_goal": "Use published launch goal.",
            "default_workspace_path": "published-launch-workspace",
            "default_constraints": ["published launch constraint"],
            "default_max_steps": 4,
            "allowed_capability_ids": ["filesystem.workspace.list"],
            "user_id": user_id,
        },
    )
    assert create_response.status_code == 200
    agent_id = create_response.json()["id"]

    publish_response = client.post(f"/agents/definitions/{agent_id}/versions", json={})
    assert publish_response.status_code == 200
    version_id = publish_response.json()["id"]

    update_response = client.put(
        f"/agents/definitions/{agent_id}",
        json={
            "instructions": "Changed launch instructions.",
            "default_goal": "Changed launch goal.",
            "default_workspace_path": "changed-launch-workspace",
            "default_constraints": ["changed launch constraint"],
            "default_max_steps": 9,
        },
    )
    assert update_response.status_code == 200

    launch_response = client.post(
        "/workbench/agent-runs",
        json={
            "agent_definition_version_id": version_id,
            "context_json": {"workspace": "demo"},
            "run_spec": _profile_agent_run_spec(),
        },
    )

    assert launch_response.status_code == 200
    body = launch_response.json()
    first_step = body["run_spec"]["steps"][0]
    assert body["run"]["title"] == "Published launch profile"
    assert body["run"]["goal"] == "Use published launch goal."
    assert body["run"]["metadata"]["agent_definition_id"] == agent_id
    assert body["run"]["metadata"]["agent_definition_version_id"] == version_id
    assert body["run"]["metadata"]["agent_definition_version_number"] == 1
    snapshot = body["run"]["metadata"]["agent_definition_snapshot"]
    assert snapshot["agent_definition_id"] == agent_id
    assert snapshot["agent_definition_version_id"] == version_id
    assert snapshot["agent_definition_version_number"] == 1
    assert snapshot["instructions"] == "Use published launch instructions."
    assert first_step["instruction"] == "Use published launch instructions."
    assert first_step["input_bindings"] == {
        "goal": "Use published launch goal.",
        "workspace_path": "published-launch-workspace",
        "constraints": "published launch constraint",
        "max_steps": 4,
    }


def test_workbench_agent_run_with_definition_version_rejects_mismatch(
    monkeypatch,
) -> None:
    monkeypatch.setattr(main.capability_registry, "load_capability_registry", _agent_registry)

    first_response = client.post(
        "/agents/definitions",
        json={
            "name": "First versioned profile",
            "agent_capability_id": "codegen.autonomous",
            "instructions": "Use first instructions.",
            "user_id": f"first-{uuid.uuid4()}",
        },
    )
    second_response = client.post(
        "/agents/definitions",
        json={
            "name": "Second versioned profile",
            "agent_capability_id": "codegen.autonomous",
            "instructions": "Use second instructions.",
            "user_id": f"second-{uuid.uuid4()}",
        },
    )
    assert first_response.status_code == 200
    assert second_response.status_code == 200
    first_id = first_response.json()["id"]
    second_id = second_response.json()["id"]
    publish_response = client.post(f"/agents/definitions/{first_id}/versions", json={})
    assert publish_response.status_code == 200
    version_id = publish_response.json()["id"]

    launch_response = client.post(
        "/workbench/agent-runs",
        json={
            "agent_definition_id": second_id,
            "agent_definition_version_id": version_id,
            "context_json": {"workspace": "demo"},
            "run_spec": _profile_agent_run_spec(extra_capability_id=None),
        },
    )

    assert launch_response.status_code == 422
    assert launch_response.json()["detail"]["error"] == "agent_definition_version_mismatch"


def test_workbench_agent_run_with_definition_rejects_primary_mismatch(
    monkeypatch,
) -> None:
    monkeypatch.setattr(main.capability_registry, "load_capability_registry", _agent_registry)
    create_response = client.post(
        "/agents/definitions",
        json={
            "name": "Codegen",
            "agent_capability_id": "codegen.autonomous",
            "instructions": "Plan and implement code.",
        },
    )
    assert create_response.status_code == 200

    response = client.post(
        "/workbench/agent-runs",
        json={
            "agent_definition_id": create_response.json()["id"],
            "run_spec": _profile_agent_run_spec(
                primary_capability_id="filesystem.workspace.list",
                extra_capability_id=None,
            ),
        },
    )

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["error"] == "agent_definition_primary_capability_mismatch"
    assert detail["expected_capability_id"] == "codegen.autonomous"
    assert detail["actual_capability_id"] == "filesystem.workspace.list"


def test_workbench_agent_run_with_definition_rejects_disallowed_extra_capability(
    monkeypatch,
) -> None:
    monkeypatch.setattr(main.capability_registry, "load_capability_registry", _agent_registry)
    create_response = client.post(
        "/agents/definitions",
        json={
            "name": "Codegen",
            "agent_capability_id": "codegen.autonomous",
            "instructions": "Plan and implement code.",
            "default_goal": "Implement code.",
            "default_workspace_path": "workbench-agent",
            "allowed_capability_ids": ["filesystem.workspace.list"],
        },
    )
    assert create_response.status_code == 200

    response = client.post(
        "/workbench/agent-runs",
        json={
            "agent_definition_id": create_response.json()["id"],
            "run_spec": _profile_agent_run_spec(extra_capability_id="document.docx.render"),
        },
    )

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["error"] == "agent_definition_disallowed_capability"
    assert detail["disallowed"] == [
        {"step_id": "extra_step", "capability_id": "document.docx.render"}
    ]


def test_workbench_agent_run_with_definition_rejects_disabled_definition(
    monkeypatch,
) -> None:
    monkeypatch.setattr(main.capability_registry, "load_capability_registry", _agent_registry)
    create_response = client.post(
        "/agents/definitions",
        json={
            "name": "Codegen",
            "agent_capability_id": "codegen.autonomous",
            "instructions": "Plan and implement code.",
        },
    )
    assert create_response.status_code == 200
    agent_definition_id = create_response.json()["id"]
    delete_response = client.delete(f"/agents/definitions/{agent_definition_id}")
    assert delete_response.status_code == 200

    response = client.post(
        "/workbench/agent-runs",
        json={
            "agent_definition_id": agent_definition_id,
            "run_spec": _profile_agent_run_spec(extra_capability_id=None),
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "agent_definition_disabled"
