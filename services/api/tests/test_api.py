import os
import uuid
from datetime import UTC, datetime

import redis
from fastapi.testclient import TestClient

os.environ["DATABASE_URL"] = "sqlite:///./test.db"
os.environ["ORCHESTRATOR_ENABLED"] = "false"
os.environ["JOB_RECOVERY_ENABLED"] = "false"
os.environ["POLICY_GATE_ENABLED"] = "false"

from services.api.app import main, memory_store  # noqa: E402
from services.api.app.database import Base, engine
from services.api.app.database import SessionLocal
from services.api.app.models import EventOutboxRecord, JobRecord, PlanRecord, TaskRecord
from libs.core import events, execution_contracts, models
from libs.core import capability_registry as cap_registry
from libs.core.llm_provider import LLMProvider, LLMRequest, LLMResponse


Base.metadata.create_all(bind=engine)

client = TestClient(main.app)


def _utcnow() -> datetime:
    return datetime.now(UTC)


def test_create_job():
    response = client.post(
        "/jobs",
        json={"goal": "demo", "context_json": {}, "priority": 1},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["goal"] == "demo"


def test_workflow_definition_publish_and_run_bypasses_planner_job_event() -> None:
    create_response = client.post(
        "/workflows/definitions",
        json={
            "title": "Workspace listing",
            "goal": "List workspace files",
            "user_id": "narendersurabhi",
            "context_json": {"user_id": "narendersurabhi"},
            "draft": {
                "summary": "Workspace listing",
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
    assert version["version_number"] == 1
    assert version["compiled_plan"]["tasks"][0]["name"] == "ListWorkspace"

    run_response = client.post(
        f"/workflows/versions/{version['id']}/run",
        json={"priority": 2},
    )
    assert run_response.status_code == 200
    run_body = run_response.json()
    job_id = run_body["job"]["id"]
    assert run_body["workflow_run"]["definition_id"] == definition["id"]
    assert run_body["workflow_run"]["version_id"] == version["id"]
    assert run_body["workflow_run"]["job_id"] == job_id
    assert run_body["job"]["metadata"]["workflow_source"] == "studio"
    assert run_body["job"]["metadata"]["workflow_definition_id"] == definition["id"]
    assert run_body["job"]["metadata"]["workflow_version_id"] == version["id"]
    assert run_body["job"]["metadata"]["workflow_run_id"] == run_body["workflow_run"]["id"]
    assert run_body["plan"]["job_id"] == job_id

    with SessionLocal() as db:
        task_records = db.query(TaskRecord).filter(TaskRecord.job_id == job_id).all()
        assert len(task_records) == 1
        event_records = db.query(EventOutboxRecord).all()
    matching_events = [
        row
        for row in event_records
        if isinstance(row.envelope_json, dict) and row.envelope_json.get("job_id") == job_id
    ]
    event_types = {row.event_type for row in matching_events}
    assert "plan.created" in event_types
    assert "job.created" not in event_types

    runs_response = client.get(f"/workflows/definitions/{definition['id']}/runs")
    assert runs_response.status_code == 200
    runs = runs_response.json()
    assert len(runs) >= 1
    assert runs[0]["job_id"] == job_id
    assert runs[0]["job_status"] in {
        models.JobStatus.queued.value,
        models.JobStatus.running.value,
        models.JobStatus.planning.value,
    }


def test_workflow_trigger_create_invoke_and_list() -> None:
    create_response = client.post(
        "/workflows/definitions",
        json={
            "title": "Triggered workspace listing",
            "goal": "List workspace files",
            "user_id": "narendersurabhi",
            "context_json": {"user_id": "narendersurabhi"},
            "draft": {
                "summary": "Triggered workspace listing",
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
            "title": "Run latest workspace listing",
            "trigger_type": "manual",
            "enabled": True,
            "config": {"version_mode": "latest_published"},
            "metadata": {"source": "test"},
        },
    )
    assert trigger_response.status_code == 200
    trigger = trigger_response.json()
    assert trigger["definition_id"] == definition["id"]
    assert trigger["trigger_type"] == "manual"

    list_triggers_response = client.get(f"/workflows/definitions/{definition['id']}/triggers")
    assert list_triggers_response.status_code == 200
    trigger_items = list_triggers_response.json()
    assert len(trigger_items) >= 1
    assert trigger_items[0]["id"] == trigger["id"]

    invoke_response = client.post(
        f"/workflows/triggers/{trigger['id']}/invoke",
        json={"version_id": version["id"], "priority": 1},
    )
    assert invoke_response.status_code == 200
    invoke_body = invoke_response.json()
    assert invoke_body["workflow_run"]["trigger_id"] == trigger["id"]
    assert invoke_body["job"]["metadata"]["workflow_trigger_id"] == trigger["id"]
    assert invoke_body["workflow_run"]["job_id"] == invoke_body["job"]["id"]

    runs_response = client.get(f"/workflows/definitions/{definition['id']}/runs?limit=5")
    assert runs_response.status_code == 200
    runs = runs_response.json()
    assert len(runs) >= 1
    assert runs[0]["trigger_id"] == trigger["id"]


def test_build_workflow_interface_runtime_context_resolves_bindings() -> None:
    with SessionLocal() as db:
        memory_store.write_memory(
            db,
            models.MemoryWrite(
                name="user_profile",
                scope=models.MemoryScope.user,
                user_id="narendersurabhi",
                key="profile",
                payload={"full_name": "Narender Rao Surabhi"},
                metadata={},
            ),
        )
        workflow_interface, errors, _warnings = main._coerce_workflow_interface(
            {
                "inputs": [
                    {
                        "key": "topic",
                        "valueType": "string",
                        "required": True,
                        "binding": {"kind": "context", "path": "topic"},
                    },
                    {
                        "key": "profile",
                        "valueType": "object",
                        "binding": {
                            "kind": "memory",
                            "scope": "user",
                            "name": "user_profile",
                            "key": "profile",
                        },
                    },
                    {
                        "key": "api_key",
                        "valueType": "string",
                        "binding": {"kind": "secret", "secretName": "TEST_WORKFLOW_SECRET"},
                    },
                ],
                "variables": [
                    {
                        "key": "topic_alias",
                        "binding": {"kind": "workflow_input", "inputKey": "topic"},
                    }
                ],
            }
        )
        assert errors == []
        context, diagnostics = main._build_workflow_interface_runtime_context(
            workflow_interface,
            base_context={"topic": "Release notes", "user_id": "narendersurabhi"},
            db=db,
            preview=False,
        )

    assert diagnostics == []
    assert context["workflow"]["inputs"]["topic"] == "Release notes"
    assert context["workflow"]["inputs"]["profile"] == {"full_name": "Narender Rao Surabhi"}
    assert context["workflow"]["inputs"]["api_key"] == execution_contracts.build_secret_ref(
        "TEST_WORKFLOW_SECRET"
    )
    assert context["workflow"]["variables"]["topic_alias"] == "Release notes"


def test_intent_clarify_endpoint_validates_goal():
    response = client.post("/intent/clarify", json={})
    assert response.status_code == 400
    assert response.json()["detail"] == "goal_required"


def test_intent_clarify_endpoint_returns_assessment():
    response = client.post("/intent/clarify", json={"goal": "Render a PDF status report"})
    assert response.status_code == 200
    body = response.json()
    assessment = body["assessment"]
    assert body["goal"] == "Render a PDF status report"
    assert assessment["intent"] == "render"
    assert assessment["needs_clarification"] is False
    assert assessment["source"] in {"goal_text", "task_text", "explicit", "default"}


def test_intent_decompose_endpoint_returns_graph():
    response = client.post(
        "/intent/decompose",
        json={"goal": "Fetch repositories, summarize insights, and render a PDF report."},
    )
    assert response.status_code == 200
    body = response.json()
    graph = body["intent_graph"]
    assert graph["summary"]["segment_count"] >= 2
    assert graph["segments"][0]["id"] == "s1"
    assert graph["summary"]["schema_version"] == "intent_v2"
    assert isinstance(graph["segments"][0].get("slots"), dict)
    assert "must_have_inputs" in graph["segments"][0]["slots"]


def test_capabilities_search_returns_ranked_matches() -> None:
    response = client.post(
        "/capabilities/search",
        json={"query": "render a pdf report from a document spec", "intent": "render", "limit": 5},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "render a pdf report from a document spec"
    assert body["intent"] == "render"
    assert len(body["items"]) >= 1
    ids = [item["id"] for item in body["items"]]
    assert "document.pdf.generate" in ids
    first = body["items"][0]
    assert isinstance(first["score"], float)
    assert isinstance(first["reason"], str) and first["reason"]


def test_capabilities_search_emits_capability_search_event(monkeypatch) -> None:
    events_seen = []
    monkeypatch.setattr(
        main,
        "_emit_event",
        lambda event_type, payload: events_seen.append((event_type, payload)),
    )

    response = client.post(
        "/capabilities/search",
        json={
            "query": "render a pdf report from a document spec",
            "intent": "render",
            "limit": 5,
            "request_source": "composer",
            "correlation_id": "corr-search-1",
            "job_id": "job-search-1",
        },
    )
    assert response.status_code == 200
    assert events_seen
    event_type, payload = events_seen[-1]
    assert event_type == "plan.capability_search"
    assert payload["request_source"] == "composer"
    assert payload["correlation_id"] == "corr-search-1"
    assert payload["job_id"] == "job-search-1"
    assert payload["result_count"] >= 1
    assert any(item["id"] == "document.pdf.generate" for item in payload["results"])


def test_capabilities_search_requires_query() -> None:
    response = client.post("/capabilities/search", json={"query": ""})
    assert response.status_code == 400
    assert response.json()["detail"] == "query_required"


def test_capabilities_search_validates_limit() -> None:
    response = client.post("/capabilities/search", json={"query": "pdf", "limit": 0})
    assert response.status_code == 400
    assert response.json()["detail"] == "limit_out_of_range"


def test_intent_decompose_endpoint_uses_llm_when_enabled(monkeypatch):
    requests: list[LLMRequest] = []

    class _Provider(LLMProvider):
        def generate_request(self, request: LLMRequest):
            requests.append(request)
            return LLMResponse(
                content=(
                    '{"segments":[{"id":"s1","intent":"generate","objective":"Create summary",'
                    '"confidence":0.92,"depends_on":[],"required_inputs":["instruction"],'
                    '"suggested_capabilities":["llm.text.generate"],'
                    '"slots":{"entity":"summary","artifact_type":"content","output_format":"txt",'
                    '"risk_level":"read_only","must_have_inputs":["instruction"]}}]}'
                )
            )

    monkeypatch.setattr(main, "INTENT_DECOMPOSE_ENABLED", True)
    monkeypatch.setattr(main, "INTENT_DECOMPOSE_MODE", "llm")
    monkeypatch.setattr(main, "_intent_decompose_provider", _Provider())

    response = client.post("/intent/decompose", json={"goal": "Create summary"})
    assert response.status_code == 200
    graph = response.json()["intent_graph"]
    assert graph["source"] == "llm"
    assert graph["summary"]["segment_count"] == 1
    assert graph["segments"][0]["intent"] == "generate"
    assert graph["segments"][0]["source"] == "llm"
    assert graph["segments"][0]["slots"]["entity"] == "summary"
    assert graph["segments"][0]["slots"]["must_have_inputs"] == ["instruction"]
    assert requests
    assert requests[0].metadata is not None
    assert requests[0].metadata["operation"] == "intent_decompose"


def test_intent_decompose_endpoint_falls_back_to_heuristic_on_llm_failure(monkeypatch):
    class _Provider(LLMProvider):
        def generate(self, prompt: str):  # noqa: ARG002
            return LLMResponse(content="not-json")

    monkeypatch.setattr(main, "INTENT_DECOMPOSE_ENABLED", True)
    monkeypatch.setattr(main, "INTENT_DECOMPOSE_MODE", "hybrid")
    monkeypatch.setattr(main, "_intent_decompose_provider", _Provider())

    response = client.post("/intent/decompose", json={"goal": "Validate schema and render PDF"})
    assert response.status_code == 200
    graph = response.json()["intent_graph"]
    assert graph["source"] == "heuristic"
    assert graph["summary"]["segment_count"] >= 1


def test_intent_decompose_endpoint_filters_unknown_llm_capabilities(monkeypatch):
    class _Provider(LLMProvider):
        def generate(self, prompt: str):  # noqa: ARG002
            return LLMResponse(
                content=(
                    '{"segments":[{"id":"s1","intent":"generate","objective":"Create summary",'
                    '"confidence":0.9,"depends_on":[],"required_inputs":["instruction"],'
                    '"suggested_capabilities":["cap.web_fetch","cap.unknown"]}]}'
                )
            )

    monkeypatch.setattr(main, "INTENT_DECOMPOSE_ENABLED", True)
    monkeypatch.setattr(main, "INTENT_DECOMPOSE_MODE", "llm")
    monkeypatch.setattr(main, "_intent_decompose_provider", _Provider())

    response = client.post("/intent/decompose", json={"goal": "Create summary"})
    assert response.status_code == 200
    graph = response.json()["intent_graph"]
    segment = graph["segments"][0]
    suggested = segment["suggested_capabilities"]
    assert "cap.web_fetch" not in suggested
    assert "cap.unknown" not in suggested
    assert len(suggested) >= 1
    rankings = segment.get("suggested_capability_rankings")
    assert isinstance(rankings, list)
    assert len(rankings) == len(suggested)
    assert all(isinstance(entry.get("reason"), str) and entry.get("reason") for entry in rankings)
    assert all(entry.get("source") in {"llm", "fallback_segment", "heuristic"} for entry in rankings)
    catalog_ids = main._intent_catalog_capability_ids()
    assert all(capability_id in catalog_ids for capability_id in suggested)


def test_intent_decompose_endpoint_normalizes_capability_id_casing(monkeypatch):
    class _Provider(LLMProvider):
        def generate(self, prompt: str):  # noqa: ARG002
            return LLMResponse(
                content=(
                    '{"segments":[{"id":"s1","intent":"generate","objective":"Create summary",'
                    '"confidence":0.9,"depends_on":[],"required_inputs":["instruction"],'
                    '"suggested_capabilities":["LLM.TEXT.GENERATE"]}]}'
                )
            )

    monkeypatch.setattr(main, "INTENT_DECOMPOSE_ENABLED", True)
    monkeypatch.setattr(main, "INTENT_DECOMPOSE_MODE", "llm")
    monkeypatch.setattr(main, "_intent_decompose_provider", _Provider())

    response = client.post("/intent/decompose", json={"goal": "Create summary"})
    assert response.status_code == 200
    graph = response.json()["intent_graph"]
    segment = graph["segments"][0]
    assert segment["suggested_capabilities"] == ["llm.text.generate"]
    rankings = segment.get("suggested_capability_rankings")
    assert isinstance(rankings, list)
    assert len(rankings) == 1
    assert rankings[0]["id"] == "llm.text.generate"
    assert rankings[0]["source"] == "llm"
    assert isinstance(rankings[0]["reason"], str) and rankings[0]["reason"]


def test_intent_decompose_endpoint_limits_capability_rankings_to_top_k(monkeypatch):
    class _Provider(LLMProvider):
        def generate(self, prompt: str):  # noqa: ARG002
            return LLMResponse(
                content=(
                    '{"segments":[{"id":"s1","intent":"generate","objective":"Create summary",'
                    '"confidence":0.9,"depends_on":[],"required_inputs":["instruction"],'
                    '"suggested_capabilities":["llm.text.generate","document.spec.generate",'
                    '"document.spec.validate","document.pdf.generate"]}]}'
                )
            )

    monkeypatch.setattr(main, "INTENT_DECOMPOSE_ENABLED", True)
    monkeypatch.setattr(main, "INTENT_DECOMPOSE_MODE", "llm")
    monkeypatch.setattr(main, "INTENT_CAPABILITY_TOP_K", 3)
    monkeypatch.setattr(main, "_intent_decompose_provider", _Provider())

    response = client.post("/intent/decompose", json={"goal": "Create summary"})
    assert response.status_code == 200
    graph = response.json()["intent_graph"]
    segment = graph["segments"][0]
    rankings = segment.get("suggested_capability_rankings")
    assert isinstance(rankings, list)
    assert len(rankings) == 3
    assert len(segment["suggested_capabilities"]) == 3
    assert segment["suggested_capabilities"] == [entry["id"] for entry in rankings]
    assert graph["summary"]["capability_top_k"] == 3


def test_intent_decompose_endpoint_validates_interaction_summaries_contract():
    response = client.post(
        "/intent/decompose",
        json={
            "goal": "Create summary",
            "interaction_summaries": [{"facts": ["captured fact only"]}],
        },
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "interaction_summaries[0].action_required"


def test_intent_decompose_endpoint_filters_unsupported_facts_with_summaries(monkeypatch):
    class _Provider(LLMProvider):
        def generate(self, prompt: str):  # noqa: ARG002
            return LLMResponse(
                content=(
                    '{"segments":[{"id":"s1","intent":"generate","objective":"Build report",'
                    '"objective_facts":["collect relevant sources","include pricing analysis"],'
                    '"confidence":0.9,"depends_on":[],"required_inputs":["instruction"],'
                    '"suggested_capabilities":["llm.text.generate"]}]}'
                )
            )

    monkeypatch.setattr(main, "INTENT_DECOMPOSE_ENABLED", True)
    monkeypatch.setattr(main, "INTENT_DECOMPOSE_MODE", "llm")
    monkeypatch.setattr(main, "_intent_decompose_provider", _Provider())

    response = client.post(
        "/intent/decompose",
        json={
            "goal": "Create summary",
            "interaction_summaries": [
                {
                    "id": "i1",
                    "facts": ["collect relevant sources"],
                    "action": "open search",
                }
            ],
        },
    )
    assert response.status_code == 200
    graph = response.json()["intent_graph"]
    segment = graph["segments"][0]
    assert segment["objective"] == "collect relevant sources"
    assert "include pricing analysis" in segment.get("unsupported_facts", [])
    summary = graph["summary"]
    assert summary["has_interaction_summaries"] is True
    assert summary["fact_candidates"] == 2
    assert summary["fact_supported"] == 1
    assert summary["fact_stripped"] == 1


def test_intent_decompose_endpoint_reports_compaction_metadata(monkeypatch):
    monkeypatch.setattr(main, "INTERACTION_SUMMARY_COMPACTION_ENABLED", True)
    monkeypatch.setattr(main, "INTERACTION_SUMMARY_MAX_ITEMS", 1)
    response = client.post(
        "/intent/decompose",
        json={
            "goal": "Create summary",
            "interaction_summaries": [
                {"id": "i1", "facts": ["first fact"], "action": "open tab"},
                {"id": "i2", "facts": ["second fact"], "action": "click button"},
            ],
        },
    )
    assert response.status_code == 200
    graph = response.json()["intent_graph"]
    summary = graph["summary"]
    compaction = summary["interaction_summary_compaction"]
    assert compaction["input_count"] == 2
    assert compaction["output_count"] == 1
    assert compaction["applied"] is True


def test_create_job_requires_intent_clarification_when_confidence_is_low():
    response = client.post(
        "/jobs?require_clarification=true",
        json={"goal": "help", "context_json": {}, "priority": 0},
    )
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["error"] == "intent_clarification_required"
    assert detail["goal_intent_profile"]["needs_clarification"] is True


def test_create_job_clarification_gate_blocks_only_blocking_slots(monkeypatch):
    monkeypatch.setattr(main, "INTENT_MIN_CONFIDENCE", 0.99)
    monkeypatch.setattr(main, "INTENT_CLARIFICATION_BLOCKING_SLOTS", {"output_format"})
    response = client.post(
        "/jobs?require_clarification=true",
        json={"goal": "Render a PDF deployment report", "context_json": {}, "priority": 0},
    )
    assert response.status_code == 200
    profile = response.json()["metadata"]["goal_intent_profile"]
    assert profile["low_confidence"] is True
    assert profile["missing_slots"] == []
    assert profile["requires_blocking_clarification"] is False


def test_create_job_rejects_invalid_interaction_summaries_in_context():
    response = client.post(
        "/jobs",
        json={
            "goal": "Create report",
            "context_json": {"interaction_summaries": [{"action": "open page"}]},
            "priority": 0,
        },
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "interaction_summaries[0].facts_required"


def test_create_job_persists_raw_and_compact_interaction_summaries(monkeypatch):
    monkeypatch.setattr(main, "INTERACTION_SUMMARY_COMPACTION_ENABLED", True)
    monkeypatch.setattr(main, "INTERACTION_SUMMARY_MAX_ITEMS", 2)
    response = client.post(
        "/jobs",
        json={
            "goal": "Create report",
            "context_json": {
                "interaction_summaries": [
                    {"id": "i1", "facts": ["first fact"], "action": "open"},
                    {"id": "i2", "facts": ["second fact"], "action": "click"},
                    {"id": "i3", "facts": ["third fact"], "action": "submit"},
                ]
            },
            "priority": 0,
        },
    )
    assert response.status_code == 200
    body = response.json()
    job_id = body["id"]
    compacted = body["context_json"]["interaction_summaries"]
    assert len(compacted) <= 2
    assert body["context_json"]["interaction_summaries_ref"]["memory_name"] == "interaction_summaries_compact"

    raw_read = client.get("/memory/read", params={"name": "interaction_summaries", "job_id": job_id})
    assert raw_read.status_code == 200
    raw_entries = raw_read.json()
    assert len(raw_entries) >= 1
    assert len(raw_entries[0]["payload"]["interaction_summaries"]) == 3

    compact_read = client.get(
        "/memory/read",
        params={"name": "interaction_summaries_compact", "job_id": job_id},
    )
    assert compact_read.status_code == 200
    compact_entries = compact_read.json()
    assert len(compact_entries) >= 1
    assert len(compact_entries[0]["payload"]["interaction_summaries"]) <= 2


def test_create_plan_persists_task_intent_profiles_and_surfaces_on_tasks():
    job_response = client.post(
        "/jobs",
        json={"goal": "render a deployment document", "context_json": {}, "priority": 1},
    )
    assert job_response.status_code == 200
    job_id = job_response.json()["id"]

    plan_payload = {
        "planner_version": "test",
        "tasks_summary": "render one file",
        "dag_edges": [],
        "tasks": [
            {
                "name": "RenderDoc",
                "description": "Render output to PDF",
                "instruction": "Render the final document",
                "acceptance_criteria": ["returns output path"],
                "expected_output_schema_ref": "schemas/output_path",
                "deps": [],
                "tool_requests": [],
                "tool_inputs": {},
                "critic_required": False,
            }
        ],
    }
    plan_response = client.post(f"/plans?job_id={job_id}", json=plan_payload)
    assert plan_response.status_code == 200

    with SessionLocal() as db:
        job_record = db.query(JobRecord).filter(JobRecord.id == job_id).first()
        assert job_record is not None
        metadata = job_record.metadata_json or {}
        profiles = metadata.get("task_intent_profiles")
        assert isinstance(profiles, dict)
        assert len(profiles) == 1
        only_profile = next(iter(profiles.values()))
        assert only_profile["intent"] == "render"
        assert only_profile["source"] in {"task_text", "goal_text", "explicit", "default"}
        assert 0.0 <= float(only_profile["confidence"]) <= 1.0

    tasks_response = client.get(f"/jobs/{job_id}/tasks")
    assert tasks_response.status_code == 200
    tasks = tasks_response.json()
    assert len(tasks) == 1
    assert tasks[0]["intent_source"] is not None
    assert isinstance(tasks[0]["intent_confidence"], (int, float))


def test_memory_specs_endpoint_lists_user_scoped_specs() -> None:
    response = client.get("/memory/specs")
    assert response.status_code == 200
    specs = response.json()
    names = {entry["name"] for entry in specs}
    assert "user_profile" in names
    assert "semantic_memory" in names


def test_delete_user_memory_entry() -> None:
    write_response = client.post(
        "/memory/write",
        json={
            "name": "user_profile",
            "scope": "user",
            "user_id": "demo-user",
            "key": "preferences",
            "payload": {"theme": "light"},
            "metadata": {"source": "test"},
        },
    )
    assert write_response.status_code == 200

    delete_response = client.delete(
        "/memory/delete",
        params={
            "name": "user_profile",
            "scope": "user",
            "user_id": "demo-user",
            "key": "preferences",
        },
    )
    assert delete_response.status_code == 200
    deleted = delete_response.json()
    assert deleted["name"] == "user_profile"
    assert deleted["user_id"] == "demo-user"
    assert deleted["key"] == "preferences"

    read_response = client.get(
        "/memory/read",
        params={
            "name": "user_profile",
            "scope": "user",
            "user_id": "demo-user",
            "key": "preferences",
        },
    )
    assert read_response.status_code == 200
    assert read_response.json() == []


def test_create_job_persists_goal_intent_graph():
    response = client.post(
        "/jobs",
        json={"goal": "Fetch data then validate and render PDF", "context_json": {}, "priority": 1},
    )
    assert response.status_code == 200
    metadata = response.json()["metadata"]
    assert "goal_intent_graph" in metadata
    graph = metadata["goal_intent_graph"]
    assert graph["summary"]["segment_count"] >= 2


def test_intent_decompose_uses_semantic_workflow_hints_in_llm_prompt(monkeypatch):
    user_id = f"intent-hints-{uuid.uuid4()}"
    write = client.post(
        "/memory/semantic/write",
        json={
            "user_id": user_id,
            "namespace": "intent_workflows",
            "subject": "pdf_report_flow",
            "fact": "Successful workflow for PDF report generation.",
            "confidence": 0.95,
            "metadata": {"from_test": True},
            "source": "test",
            "key": f"intent_workflow:{uuid.uuid4()}",
            "reasoning": "stored from prior successful run",
            "keywords": ["pdf", "report", "workflow"],
            "aliases": ["render pdf report"],
        },
    )
    assert write.status_code == 200
    with SessionLocal() as db:
        entry = memory_store.read_memory(
            db,
            models.MemoryQuery(
                name="semantic_memory",
                scope=models.MemoryScope.user,
                user_id=user_id,
                key=write.json()["entry"]["key"],
                limit=1,
            ),
        )[0]
        payload = dict(entry.payload)
        payload["intent_workflow"] = {
            "job_id": "job-prev",
            "goal": "Render monthly pdf report",
            "outcome": "succeeded",
            "intent_order": ["generate", "render"],
            "capabilities": ["document.spec.generate", "document.pdf.generate"],
            "confidence": 0.9,
            "threshold": 0.7,
        }
        memory_store.write_memory(
            db,
            models.MemoryWrite(
                name="semantic_memory",
                scope=models.MemoryScope.user,
                user_id=user_id,
                key=entry.key,
                payload=payload,
                metadata=entry.metadata,
            ),
        )

    prompts: list[str] = []

    class _Provider(LLMProvider):
        def generate(self, prompt: str):  # noqa: ARG002
            prompts.append(prompt)
            return LLMResponse(
                content=(
                    '{"segments":[{"id":"s1","intent":"generate","objective":"Create report",'
                    '"confidence":0.91,"depends_on":[],"required_inputs":["instruction"],'
                    '"suggested_capabilities":["document.spec.generate"]}]}'
                )
            )

    monkeypatch.setattr(main, "INTENT_DECOMPOSE_ENABLED", True)
    monkeypatch.setattr(main, "INTENT_DECOMPOSE_MODE", "llm")
    monkeypatch.setattr(main, "_intent_decompose_provider", _Provider())
    monkeypatch.setattr(main, "INTENT_MEMORY_RETRIEVAL_ENABLED", True)
    monkeypatch.setattr(main, "INTENT_MEMORY_RETRIEVAL_LIMIT", 3)

    response = client.post(
        "/intent/decompose",
        json={"goal": "Create a PDF report", "user_id": user_id},
    )
    assert response.status_code == 200
    graph = response.json()["intent_graph"]
    assert graph["summary"]["memory_hints_used"] >= 1
    assert prompts
    assert "Similar successful workflow hints from semantic memory" in prompts[0]


def test_intent_decompose_uses_semantic_capability_hints_in_llm_prompt(monkeypatch):
    prompts: list[str] = []

    class _Provider(LLMProvider):
        def generate(self, prompt: str):  # noqa: ARG002
            prompts.append(prompt)
            return LLMResponse(
                content=(
                    '{"segments":[{"id":"s1","intent":"render","objective":"Render pdf report",'
                    '"confidence":0.9,"depends_on":[],"required_inputs":["document_spec","path"],'
                    '"suggested_capabilities":["document.pdf.generate"]}]}'
                )
            )

    monkeypatch.setattr(main, "INTENT_DECOMPOSE_ENABLED", True)
    monkeypatch.setattr(main, "INTENT_DECOMPOSE_MODE", "llm")
    monkeypatch.setattr(main, "_intent_decompose_provider", _Provider())

    response = client.post(
        "/intent/decompose",
        json={"goal": "Render a PDF report from a document spec"},
    )
    assert response.status_code == 200
    graph = response.json()["intent_graph"]
    assert graph["summary"]["semantic_capability_hints_used"] >= 1
    assert prompts
    assert "Most relevant capabilities for this goal from local semantic search" in prompts[0]
    assert "document.pdf.generate" in prompts[0]


def test_intent_decompose_emits_capability_search_event_for_semantic_hints(monkeypatch):
    events_seen = []
    monkeypatch.setattr(
        main,
        "_emit_event",
        lambda event_type, payload: events_seen.append((event_type, payload)),
    )

    graph = main._decompose_goal_intent("Search semantic memory for user preferences")

    assert graph.summary.semantic_capability_hints_used is not None
    assert graph.summary.semantic_capability_hints_used >= 1
    assert events_seen
    event_type, payload = events_seen[-1]
    assert event_type == "plan.capability_search"
    assert payload["request_source"] == "intent_decompose"
    assert payload["query"] == "Search semantic memory for user preferences"
    assert payload["result_count"] >= 1


def test_refresh_job_status_persists_intent_outcome_memory_and_calibration():
    create = client.post(
        "/jobs",
        json={
            "goal": "Generate and render a PDF report",
            "context_json": {"user_id": f"intent-loop-{uuid.uuid4()}"},
            "priority": 0,
        },
    )
    assert create.status_code == 200
    job = create.json()
    job_id = job["id"]
    user_id = job["context_json"]["user_id"]

    plan = client.post(
        f"/plans?job_id={job_id}",
        json={
            "planner_version": "test",
            "tasks_summary": "single step",
            "dag_edges": [],
            "tasks": [
                {
                    "name": "GenerateSpec",
                    "description": "create spec",
                    "instruction": "create document spec",
                    "acceptance_criteria": ["spec exists"],
                    "expected_output_schema_ref": "schemas/document_spec",
                    "intent": "generate",
                    "deps": [],
                    "tool_requests": ["document.spec.generate"],
                    "tool_inputs": {"document.spec.generate": {"instruction": "Create report spec"}},
                    "critic_required": False,
                }
            ],
        },
    )
    assert plan.status_code == 200
    tasks = client.get(f"/jobs/{job_id}/tasks")
    assert tasks.status_code == 200
    task_id = tasks.json()[0]["id"]

    main._handle_task_completed(
        {
            "task_id": task_id,
            "payload": {
                "task_id": task_id,
                "outputs": {"document.spec.generate": {"document_spec": {"blocks": []}}},
            },
        }
    )

    job_after = client.get(f"/jobs/{job_id}")
    assert job_after.status_code == 200
    metadata = job_after.json()["metadata"]
    assert metadata.get("intent_confidence_outcome_recorded") is True
    assert metadata.get("intent_memory_persisted") is True

    semantic = client.get(
        "/memory/read",
        params={"name": "semantic_memory", "user_id": user_id, "key": f"intent_workflow:{job_id}"},
    )
    assert semantic.status_code == 200
    entries = semantic.json()
    assert len(entries) == 1
    payload = entries[0]["payload"]
    assert payload["namespace"] == "intent_workflows"
    assert payload["intent_workflow"]["outcome"] == "succeeded"


def test_contract_intent_mismatch_triggers_auto_replan_with_recovery_metadata():
    now = _utcnow()
    job_id = str(uuid.uuid4())
    plan_id = str(uuid.uuid4())
    task_id = str(uuid.uuid4())
    with SessionLocal() as db:
        db.add(
            JobRecord(
                id=job_id,
                goal="List github repos",
                context_json={},
                status=models.JobStatus.running.value,
                created_at=now,
                updated_at=now,
                priority=0,
                metadata_json={},
            )
        )
        db.add(
            PlanRecord(
                id=plan_id,
                job_id=job_id,
                planner_version="test",
                created_at=now,
                tasks_summary="one task",
                dag_edges=[],
                policy_decision={},
            )
        )
        db.add(
            TaskRecord(
                id=task_id,
                job_id=job_id,
                plan_id=plan_id,
                name="ListRepos",
                description="list repos",
                instruction="list repos from github",
                acceptance_criteria=["repos listed"],
                expected_output_schema_ref="schemas/unknown",
                status=models.TaskStatus.running.value,
                deps=[],
                attempts=0,
                max_attempts=3,
                rework_count=0,
                max_reworks=2,
                assigned_to=None,
                intent="generate",
                tool_requests=["github.repo.list"],
                tool_inputs={"github.repo.list": {"owner": "octocat"}},
                created_at=now,
                updated_at=now,
                critic_required=False,
            )
        )
        db.commit()

    main._handle_task_failed(
        {
            "task_id": task_id,
            "payload": {
                "task_id": task_id,
                "error": "contract.intent_mismatch:task_intent_mismatch:github.repo.list:generate:allowed=io",
            },
        }
    )

    job_after = client.get(f"/jobs/{job_id}")
    assert job_after.status_code == 200
    body = job_after.json()
    assert body["status"] == "planning"
    metadata = body["metadata"]
    assert metadata.get("replan_reason") == "intent_mismatch_auto_repair"
    recovery = metadata.get("intent_mismatch_recovery")
    assert isinstance(recovery, dict)
    assert recovery.get("failing_capability") == "github.repo.list"
    assert recovery.get("allowed_task_intents") == ["io"]

def test_semantic_memory_write_and_read_round_trip():
    user_id = f"semantic-user-{uuid.uuid4()}"
    response = client.post(
        "/memory/semantic/write",
        json={
            "user_id": user_id,
            "namespace": "user_prefs",
            "subject": "response_style",
            "fact": "User prefers concise answers with concrete next steps.",
            "keywords": ["concise", "actionable"],
            "confidence": 0.92,
            "source": "ui_manual",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["entry"]["name"] == "semantic_memory"
    assert body["entry"]["scope"] == "user"
    assert body["entry"]["user_id"] == user_id
    assert body["semantic_record"]["subject"] == "response_style"

    read_response = client.get(
        "/memory/read",
        params={"name": "semantic_memory", "user_id": user_id, "limit": 10},
    )
    assert read_response.status_code == 200
    entries = read_response.json()
    assert len(entries) >= 1
    assert any(
        isinstance(entry.get("payload"), dict)
        and "concise answers" in str(entry["payload"].get("fact", "")).lower()
        for entry in entries
    )


def test_semantic_memory_search_returns_ranked_match():
    user_id = f"semantic-search-{uuid.uuid4()}"
    write_a = client.post(
        "/memory/semantic/write",
        json={
            "user_id": user_id,
            "namespace": "runtime",
            "subject": "mcp_reliability",
            "fact": "Use retries with bounded backoff when MCP calls timeout.",
            "keywords": ["mcp", "retries", "timeout", "backoff"],
            "confidence": 0.95,
            "source": "api_test",
        },
    )
    assert write_a.status_code == 200
    write_b = client.post(
        "/memory/semantic/write",
        json={
            "user_id": user_id,
            "namespace": "ui",
            "subject": "dashboard_theme",
            "fact": "Use neutral color palettes for operational dashboards.",
            "keywords": ["ui", "color"],
            "confidence": 0.8,
            "source": "api_test",
        },
    )
    assert write_b.status_code == 200

    search = client.post(
        "/memory/semantic/search",
        json={
            "user_id": user_id,
            "query": "mcp timeout retries with backoff",
            "limit": 5,
            "min_score": 0.05,
        },
    )
    assert search.status_code == 200
    body = search.json()
    assert body["count"] >= 1
    top = body["matches"][0]
    assert "retries" in str(top.get("fact", "")).lower()
    assert float(top.get("score", 0)) > 0


def test_emit_event_persists_outbox_when_redis_is_unavailable(monkeypatch):
    job_id = f"job-outbox-down-{uuid.uuid4()}"

    class _RedisDown:
        def xadd(self, stream, payload):
            raise redis.RedisError("redis down")

    monkeypatch.setattr(main, "redis_client", _RedisDown())
    monkeypatch.setattr(main, "EVENT_OUTBOX_ENABLED", True)
    with SessionLocal() as db:
        db.query(EventOutboxRecord).delete()
        db.commit()

    main._emit_event(
        "job.created",
        {
            "id": job_id,
            "job_id": job_id,
            "goal": "outbox fallback",
            "context_json": {},
            "status": models.JobStatus.queued.value,
            "priority": 0,
            "metadata": {},
            "created_at": _utcnow().isoformat(),
            "updated_at": _utcnow().isoformat(),
        },
    )

    with SessionLocal() as db:
        rows = db.query(EventOutboxRecord).all()
        assert len(rows) == 1
        row = rows[0]
        assert row.stream == events.JOB_STREAM
        assert row.event_type == "job.created"
        assert row.published_at is None
        assert (row.attempts or 0) >= 1
        assert row.last_error is not None


def test_dispatch_event_outbox_once_publishes_pending_rows(monkeypatch):
    outbox_id = f"outbox-{uuid.uuid4()}"
    now = _utcnow()
    sent = []

    class _RedisOk:
        def xadd(self, stream, payload):
            sent.append((stream, payload))
            return "1-0"

    monkeypatch.setattr(main, "redis_client", _RedisOk())
    monkeypatch.setattr(main, "EVENT_OUTBOX_ENABLED", True)
    with SessionLocal() as db:
        db.query(EventOutboxRecord).delete()
        db.add(
            EventOutboxRecord(
                id=outbox_id,
                stream=events.TASK_STREAM,
                event_type="task.ready",
                envelope_json={
                    "type": "task.ready",
                    "version": "1",
                    "occurred_at": now.isoformat(),
                    "correlation_id": str(uuid.uuid4()),
                    "job_id": "job-x",
                    "task_id": "task-x",
                    "payload": {"task_id": "task-x"},
                },
                attempts=0,
                last_error=None,
                created_at=now,
                updated_at=now,
                published_at=None,
            )
        )
        db.commit()

    dispatched = main._dispatch_event_outbox_once()
    assert dispatched == 1
    assert len(sent) == 1
    assert sent[0][0] == events.TASK_STREAM

    with SessionLocal() as db:
        row = db.query(EventOutboxRecord).filter(EventOutboxRecord.id == outbox_id).first()
        assert row is not None
        assert row.published_at is not None
        assert row.last_error is None
        assert (row.attempts or 0) >= 1


def test_event_stream():
    response = client.get("/events/stream?once=true")
    assert response.status_code == 200


def test_job_event_outbox_filters_by_job_and_pending_state():
    job_id = f"job-outbox-view-{uuid.uuid4()}"
    other_job_id = f"job-outbox-other-{uuid.uuid4()}"
    now = _utcnow()
    with SessionLocal() as db:
        db.query(EventOutboxRecord).delete()
        db.add(
            EventOutboxRecord(
                id=str(uuid.uuid4()),
                stream=events.TASK_STREAM,
                event_type="task.ready",
                envelope_json={
                    "type": "task.ready",
                    "job_id": job_id,
                    "task_id": "task-1",
                    "occurred_at": now.isoformat(),
                    "correlation_id": str(uuid.uuid4()),
                    "payload": {"task_id": "task-1"},
                },
                attempts=2,
                last_error="redis down",
                created_at=now,
                updated_at=now,
                published_at=None,
            )
        )
        db.add(
            EventOutboxRecord(
                id=str(uuid.uuid4()),
                stream=events.TASK_STREAM,
                event_type="task.ready",
                envelope_json={
                    "type": "task.ready",
                    "job_id": other_job_id,
                    "task_id": "task-2",
                    "occurred_at": now.isoformat(),
                    "correlation_id": str(uuid.uuid4()),
                    "payload": {"task_id": "task-2"},
                },
                attempts=1,
                last_error="redis down",
                created_at=now,
                updated_at=now,
                published_at=None,
            )
        )
        db.add(
            EventOutboxRecord(
                id=str(uuid.uuid4()),
                stream=events.TASK_STREAM,
                event_type="task.ready",
                envelope_json={
                    "type": "task.ready",
                    "job_id": job_id,
                    "task_id": "task-3",
                    "occurred_at": now.isoformat(),
                    "correlation_id": str(uuid.uuid4()),
                    "payload": {"task_id": "task-3"},
                },
                attempts=1,
                last_error=None,
                created_at=now,
                updated_at=now,
                published_at=now,
            )
        )
        db.commit()

    response = client.get(f"/jobs/{job_id}/events/outbox?pending_only=true&limit=10")
    assert response.status_code == 200
    body = response.json()
    assert body["job_id"] == job_id
    assert body["pending_only"] is True
    assert body["count"] == 1
    assert len(body["items"]) == 1
    assert body["items"][0]["job_id"] == job_id
    assert body["items"][0]["published_at"] is None


def test_job_details():
    job_id = f"job-details-{uuid.uuid4()}"
    plan_id = f"plan-details-{uuid.uuid4()}"
    task_id = f"task-details-{uuid.uuid4()}"
    now = _utcnow()
    with SessionLocal() as db:
        db.add(
            JobRecord(
                id=job_id,
                goal="details",
                context_json={},
                status=models.JobStatus.queued.value,
                created_at=now,
                updated_at=now,
                priority=0,
                metadata_json={},
            )
        )
        db.add(
            PlanRecord(
                id=plan_id,
                job_id=job_id,
                planner_version="test",
                created_at=now,
                tasks_summary="one task",
                dag_edges=[],
                policy_decision={},
            )
        )
        db.add(
            TaskRecord(
                id=task_id,
                job_id=job_id,
                plan_id=plan_id,
                name="t1",
                description="desc",
                instruction="do it",
                acceptance_criteria=[],
                expected_output_schema_ref="TaskResult",
                status=models.TaskStatus.pending.value,
                deps=[],
                attempts=0,
                max_attempts=1,
                rework_count=0,
                max_reworks=0,
                assigned_to=None,
                intent=None,
                tool_requests=[],
                tool_inputs={},
                created_at=now,
                updated_at=now,
                critic_required=1,
            )
        )
        db.commit()

    response = client.get(f"/jobs/{job_id}/details")
    assert response.status_code == 200
    data = response.json()
    assert data["job_id"] == job_id
    assert data["plan"]["id"] == plan_id
    assert len(data["tasks"]) == 1
    assert data["tasks"][0]["id"] == task_id
    assert task_id in data["task_results"]


def test_job_debugger_returns_timeline_and_error_classification(monkeypatch):
    job_id = f"job-debug-{uuid.uuid4()}"
    plan_id = f"plan-debug-{uuid.uuid4()}"
    task_id = f"task-debug-{uuid.uuid4()}"
    now = _utcnow()
    with SessionLocal() as db:
        db.add(
            JobRecord(
                id=job_id,
                goal="debug",
                context_json={"job": {"topic": "latency"}},
                status=models.JobStatus.running.value,
                created_at=now,
                updated_at=now,
                priority=0,
                metadata_json={},
            )
        )
        db.add(
            PlanRecord(
                id=plan_id,
                job_id=job_id,
                planner_version="test",
                created_at=now,
                tasks_summary="single debug task",
                dag_edges=[],
                policy_decision={},
            )
        )
        db.add(
            TaskRecord(
                id=task_id,
                job_id=job_id,
                plan_id=plan_id,
                name="GenerateSpec",
                description="desc",
                instruction="do it",
                acceptance_criteria=[],
                expected_output_schema_ref="TaskResult",
                status=models.TaskStatus.failed.value,
                deps=[],
                attempts=2,
                max_attempts=3,
                rework_count=0,
                max_reworks=0,
                assigned_to=None,
                intent=None,
                tool_requests=["document.spec.generate"],
                tool_inputs={"document.spec.generate": {"job": {"topic": "latency"}}},
                created_at=now,
                updated_at=now,
                critic_required=1,
            )
        )
        db.commit()

    monkeypatch.setattr(
        main,
        "_load_task_result",
        lambda _task_id: {"task_id": _task_id, "error": "contract.input_missing:job"},
    )
    monkeypatch.setattr(
        main,
        "_read_task_events_for_job",
        lambda _job_id, _limit: [
            {
                "stream_id": "1-0",
                "type": "task.started",
                "occurred_at": now.isoformat(),
                "job_id": _job_id,
                "task_id": task_id,
                "status": "running",
                "attempts": 2,
                "max_attempts": 3,
                "worker_consumer": "worker-a",
                "run_id": "run-1",
                "error": "",
            }
        ],
    )

    response = client.get(f"/jobs/{job_id}/debugger")
    assert response.status_code == 200
    payload = response.json()
    assert payload["job_id"] == job_id
    assert payload["job_status"] == "running"
    assert payload["plan_id"] == plan_id
    assert payload["timeline_events_scanned"] == 1
    assert len(payload["tasks"]) == 1
    task_payload = payload["tasks"][0]
    assert task_payload["task"]["id"] == task_id
    assert task_payload["tool_inputs_resolved"] is True
    assert task_payload["error"]["category"] == "contract"
    assert task_payload["error"]["retryable"] is False
    assert len(task_payload["timeline"]) == 1


def test_plan_created_enqueues_ready_tasks():
    job_id = f"job-test-plan-{uuid.uuid4()}"
    with SessionLocal() as db:
        db.add(
            JobRecord(
                id=job_id,
                goal="demo",
                context_json={},
                status=models.JobStatus.queued.value,
                created_at=_utcnow(),
                updated_at=_utcnow(),
                priority=0,
                metadata_json={},
            )
        )
        db.commit()
    plan = models.PlanCreate(
        planner_version="test",
        tasks_summary="t1 then t2",
        dag_edges=[["t1", "t2"]],
        tasks=[
            models.TaskCreate(
                name="t1",
                description="first",
                instruction="do first",
                acceptance_criteria=["ok"],
                expected_output_schema_ref="TaskResult",
                deps=[],
                tool_requests=[],
                critic_required=False,
            ),
            models.TaskCreate(
                name="t2",
                description="second",
                instruction="do second",
                acceptance_criteria=["ok"],
                expected_output_schema_ref="TaskResult",
                deps=["t1"],
                tool_requests=[],
                critic_required=False,
            ),
        ],
    )
    payload = plan.model_dump()
    payload["job_id"] = job_id
    envelope = {
        "type": "plan.created",
        "payload": payload,
        "job_id": job_id,
        "correlation_id": "corr",
    }
    events: list[tuple[str, dict]] = []
    original_emit = main._emit_event
    try:
        main._emit_event = lambda event_type, event_payload: events.append(
            (event_type, event_payload)
        )
        main._handle_plan_created(envelope)
    finally:
        main._emit_event = original_emit
    with SessionLocal() as db:
        tasks = db.query(TaskRecord).filter(TaskRecord.job_id == job_id).all()
        by_name = {task.name: task for task in tasks}
        assert by_name["t1"].status == models.TaskStatus.ready.value
        assert by_name["t2"].status == models.TaskStatus.pending.value
    assert any(event_type == "task.ready" for event_type, _ in events)


def test_handle_plan_created_persists_embedded_capability_bindings() -> None:
    job_id = f"job-test-bindings-{uuid.uuid4()}"
    now = _utcnow()
    with SessionLocal() as db:
        db.add(
            JobRecord(
                id=job_id,
                goal="check repo",
                context_json={},
                status=models.JobStatus.queued.value,
                created_at=now,
                updated_at=now,
                priority=0,
                metadata_json={},
            )
        )
        db.commit()
    plan = models.PlanCreate(
        planner_version="test",
        tasks_summary="check repo",
        dag_edges=[],
        tasks=[
            models.TaskCreate(
                name="CheckRepo",
                description="Check repo",
                instruction="Check repo",
                acceptance_criteria=["checked"],
                expected_output_schema_ref="TaskResult",
                deps=[],
                tool_requests=["github.repo.list"],
                tool_inputs={"github.repo.list": {"owner": "narendersurabhi", "repo": "demo"}},
                capability_bindings={
                    "github.repo.list": {
                        "request_id": "github.repo.list",
                        "capability_id": "github.repo.list",
                        "tool_name": "github.repo.list",
                        "adapter_type": "mcp",
                    }
                },
                critic_required=False,
            )
        ],
    )
    envelope = {
        "type": "plan.created",
        "payload": {"job_id": job_id, **plan.model_dump(mode="json")},
        "job_id": job_id,
        "correlation_id": "corr",
    }

    main._handle_plan_created(envelope)

    with SessionLocal() as db:
        task = db.query(TaskRecord).filter(TaskRecord.job_id == job_id).one()
        assert task.tool_inputs["github.repo.list"]["repo"] == "demo"
        assert (
            task.tool_inputs[main.execution_contracts.EXECUTION_BINDINGS_KEY]["github.repo.list"][
                "capability_id"
            ]
            == "github.repo.list"
        )


def test_handle_plan_failed_marks_queued_job_failed_and_exposes_details_error():
    job_id = f"job-plan-failed-{uuid.uuid4()}"
    now = _utcnow()
    with SessionLocal() as db:
        db.add(
            JobRecord(
                id=job_id,
                goal="planner fail test",
                context_json={},
                status=models.JobStatus.queued.value,
                created_at=now,
                updated_at=now,
                priority=0,
                metadata_json={},
            )
        )
        db.commit()

    main._handle_plan_failed(
        {
            "type": "plan.failed",
            "job_id": job_id,
            "payload": {
                "job_id": job_id,
                "error": "Invalid plan generated: tool_intent_mismatch:test",
            },
        }
    )

    with SessionLocal() as db:
        job = db.query(JobRecord).filter(JobRecord.id == job_id).first()
        assert job is not None
        assert job.status == models.JobStatus.failed.value
        assert isinstance(job.metadata_json, dict)
        assert "tool_intent_mismatch" in str(job.metadata_json.get("plan_error", ""))

    response = client.get(f"/jobs/{job_id}/details")
    assert response.status_code == 200
    payload = response.json()
    assert payload["job_status"] == models.JobStatus.failed.value
    assert "tool_intent_mismatch" in str(payload.get("job_error", ""))


def test_handle_task_started_sets_task_running_and_job_running():
    job_id = f"job-task-started-{uuid.uuid4()}"
    plan_id = f"plan-task-started-{uuid.uuid4()}"
    task_id = f"task-task-started-{uuid.uuid4()}"
    now = _utcnow()
    with SessionLocal() as db:
        db.add(
            JobRecord(
                id=job_id,
                goal="task started",
                context_json={},
                status=models.JobStatus.planning.value,
                created_at=now,
                updated_at=now,
                priority=0,
                metadata_json={},
            )
        )
        db.add(
            PlanRecord(
                id=plan_id,
                job_id=job_id,
                planner_version="test",
                created_at=now,
                tasks_summary="single task",
                dag_edges=[],
                policy_decision={},
            )
        )
        db.add(
            TaskRecord(
                id=task_id,
                job_id=job_id,
                plan_id=plan_id,
                name="only-task",
                description="desc",
                instruction="do",
                acceptance_criteria=[],
                expected_output_schema_ref="TaskResult",
                status=models.TaskStatus.ready.value,
                deps=[],
                attempts=1,
                max_attempts=3,
                rework_count=0,
                max_reworks=0,
                assigned_to=None,
                intent=None,
                tool_requests=[],
                tool_inputs={},
                created_at=now,
                updated_at=now,
                critic_required=0,
            )
        )
        db.commit()

    envelope = {
        "type": "task.started",
        "job_id": job_id,
        "task_id": task_id,
        "payload": {
            "task_id": task_id,
            "attempts": 1,
            "max_attempts": 3,
            "worker_consumer": "worker-a",
        },
    }
    main._handle_task_started(envelope)

    with SessionLocal() as db:
        job = db.query(JobRecord).filter(JobRecord.id == job_id).first()
        task = db.query(TaskRecord).filter(TaskRecord.id == task_id).first()
        assert job is not None
        assert task is not None
        assert job.status == models.JobStatus.running.value
        assert task.status == models.TaskStatus.running.value
        assert task.assigned_to == "worker-a"


def test_read_task_dlq_filters_by_job_and_respects_limit(monkeypatch):
    class _RedisStub:
        def xrevrange(self, stream, max_id, min_id, count=0):
            return [
                (
                    "11-0",
                    {
                        "data": '{"message_id":"m-1","job_id":"job-a","task_id":"t-1","error":"timed out","failed_at":"2026-02-14T00:00:00Z"}'
                    },
                ),
                (
                    "10-0",
                    {
                        "data": '{"message_id":"m-2","job_id":"job-b","task_id":"t-2","error":"hard failure"}'
                    },
                ),
                (
                    "9-0",
                    {
                        "data": '{"message_id":"m-3","job_id":"job-a","task_id":"t-3","error":"fatal"}'
                    },
                ),
            ]

    monkeypatch.setattr(main, "redis_client", _RedisStub())
    response = client.get("/jobs/job-a/tasks/dlq?limit=1")
    assert response.status_code == 200
    payload = response.json()
    assert len(payload) == 1
    assert payload[0]["job_id"] == "job-a"
    assert payload[0]["message_id"] == "m-1"


def test_read_task_dlq_returns_503_on_redis_error(monkeypatch):
    class _RedisStub:
        def xrevrange(self, stream, max_id, min_id, count=0):
            raise redis.RedisError("down")

    monkeypatch.setattr(main, "redis_client", _RedisStub())
    response = client.get("/jobs/job-a/tasks/dlq?limit=5")
    assert response.status_code == 503
    assert response.json()["detail"].startswith("redis_error:")


def test_list_capabilities_returns_required_inputs(monkeypatch):
    capability = cap_registry.CapabilitySpec(
        capability_id="github.repo.list",
        description="List repos",
        risk_tier="read_only",
        idempotency="read",
        group="github",
        subgroup="repositories",
        input_schema_ref="github_repo_list_capability_input",
        output_schema_ref=None,
        adapters=(
            cap_registry.CapabilityAdapterSpec(
                type="mcp",
                server_id="github_local",
                tool_name="search_repositories",
            ),
        ),
        tags=("github",),
        enabled=True,
    )
    registry = cap_registry.CapabilityRegistry(capabilities={"github.repo.list": capability})
    monkeypatch.setattr(main.capability_registry, "resolve_capability_mode", lambda: "enabled")
    monkeypatch.setattr(main.capability_registry, "load_capability_registry", lambda: registry)
    monkeypatch.setattr(
        main,
        "_load_schema_from_ref",
        lambda schema_ref: {
            "type": "object",
            "required": ["query"],
            "properties": {"query": {"type": "string"}},
        },
    )
    response = client.get("/capabilities")
    assert response.status_code == 200
    payload = response.json()
    assert payload["mode"] == "enabled"
    assert len(payload["items"]) == 1
    item = payload["items"][0]
    assert item["id"] == "github.repo.list"
    assert item["group"] == "github"
    assert item["subgroup"] == "repositories"
    assert item["required_inputs"] == ["query"]


def test_composer_recommend_capabilities_heuristic(monkeypatch):
    spec_generate = cap_registry.CapabilitySpec(
        capability_id="document.spec.generate",
        description="Generate a document spec",
        risk_tier="read_only",
        idempotency="read",
        input_schema_ref="schema_generate",
        output_schema_ref=None,
        adapters=(),
        enabled=True,
    )
    spec_derive = cap_registry.CapabilitySpec(
        capability_id="document.output.derive",
        description="Derive output path",
        risk_tier="read_only",
        idempotency="read",
        input_schema_ref="schema_derive",
        output_schema_ref=None,
        adapters=(),
        enabled=True,
    )
    registry = cap_registry.CapabilityRegistry(
        capabilities={
            "document.spec.generate": spec_generate,
            "document.output.derive": spec_derive,
        }
    )
    monkeypatch.setattr(main.capability_registry, "load_capability_registry", lambda: registry)

    def _schema_loader(schema_ref):
        if schema_ref == "schema_generate":
            return {
                "type": "object",
                "required": ["topic"],
                "properties": {"topic": {"type": "string"}},
            }
        if schema_ref == "schema_derive":
            return {
                "type": "object",
                "required": ["topic"],
                "properties": {"topic": {"type": "string"}},
            }
        return None

    monkeypatch.setattr(main, "_load_schema_from_ref", _schema_loader)
    response = client.post(
        "/composer/recommend_capabilities",
        json={
            "goal": "Use document.output.derive for output path",
            "context_json": {"topic": "Latency"},
            "draft": {"nodes": [{"id": "n1", "capabilityId": "document.spec.generate"}]},
            "use_llm": False,
            "max_results": 3,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["source"] == "heuristic"
    assert isinstance(body["recommendations"], list)
    assert len(body["recommendations"]) >= 1
    assert body["recommendations"][0]["id"] == "document.output.derive"


def test_composer_recommend_capabilities_uses_llm_when_available(monkeypatch):
    spec_generate = cap_registry.CapabilitySpec(
        capability_id="document.spec.generate",
        description="Generate a document spec",
        risk_tier="read_only",
        idempotency="read",
        input_schema_ref="schema_generate",
        output_schema_ref=None,
        adapters=(),
        enabled=True,
    )
    spec_derive = cap_registry.CapabilitySpec(
        capability_id="document.output.derive",
        description="Derive output path",
        risk_tier="read_only",
        idempotency="read",
        input_schema_ref="schema_derive",
        output_schema_ref=None,
        adapters=(),
        enabled=True,
    )
    registry = cap_registry.CapabilityRegistry(
        capabilities={
            "document.spec.generate": spec_generate,
            "document.output.derive": spec_derive,
        }
    )
    monkeypatch.setattr(main.capability_registry, "load_capability_registry", lambda: registry)
    monkeypatch.setattr(
        main,
        "_load_schema_from_ref",
        lambda _ref: {"type": "object", "required": ["topic"], "properties": {"topic": {"type": "string"}}},
    )

    requests: list[LLMRequest] = []

    class _Provider(LLMProvider):
        def generate_request(self, request: LLMRequest):
            requests.append(request)
            return LLMResponse(
                content='{"recommendations":[{"id":"document.output.derive","reason":"next step","confidence":0.91}]}'
            )

    monkeypatch.setattr(main, "_composer_recommender_provider", _Provider())

    response = client.post(
        "/composer/recommend_capabilities",
        json={
            "goal": "Generate and render document",
            "context_json": {"topic": "Latency"},
            "draft": {"nodes": [{"id": "n1", "capabilityId": "document.spec.generate"}]},
            "use_llm": True,
            "max_results": 3,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["source"] == "llm"
    assert body["recommendations"][0]["id"] == "document.output.derive"
    assert requests
    assert requests[0].metadata is not None
    assert requests[0].metadata["operation"] == "capability_recommendations"


def test_preflight_plan_endpoint_returns_valid_true_for_simple_plan():
    payload = {
        "plan": {
            "planner_version": "ui_chaining_composer_v1",
            "tasks_summary": "simple",
            "dag_edges": [],
            "tasks": [
                {
                    "name": "TransformData",
                    "description": "transform",
                    "instruction": "transform",
                    "acceptance_criteria": ["done"],
                    "expected_output_schema_ref": "",
                    "deps": [],
                    "tool_requests": ["json_transform"],
                    "tool_inputs": {"json_transform": {"input": {"name": "demo"}}},
                    "critic_required": False,
                }
            ],
        },
        "job_context": {},
    }
    response = client.post("/plans/preflight", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["valid"] is True
    assert body["errors"] == {}


def test_preflight_plan_endpoint_returns_reference_error_for_broken_dependency_tool():
    payload = {
        "plan": {
            "planner_version": "ui_chaining_composer_v1",
            "tasks_summary": "broken",
            "dag_edges": [["GenerateSpec", "ValidateSpec"]],
            "tasks": [
                {
                    "name": "GenerateSpec",
                    "description": "generate",
                    "instruction": "generate",
                    "acceptance_criteria": ["done"],
                    "expected_output_schema_ref": "",
                    "deps": [],
                    "tool_requests": ["document.spec.generate"],
                    "tool_inputs": {"document.spec.generate": {"job": {"topic": "demo"}}},
                    "critic_required": False,
                },
                {
                    "name": "ValidateSpec",
                    "description": "validate",
                    "instruction": "validate",
                    "acceptance_criteria": ["done"],
                    "expected_output_schema_ref": "",
                    "deps": ["GenerateSpec"],
                    "tool_requests": ["document.spec.validate"],
                    "tool_inputs": {
                        "document.spec.validate": {
                            "document_spec": {
                                "$from": [
                                    "dependencies_by_name",
                                    "GenerateSpec",
                                    "unknown.tool",
                                    "document_spec",
                                ]
                            }
                        }
                    },
                    "critic_required": False,
                },
            ],
        },
        "job_context": {},
    }
    response = client.post("/plans/preflight", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["valid"] is False
    assert "ValidateSpec" in body["errors"]
    assert "input reference resolution failed" in body["errors"]["ValidateSpec"]


def test_preflight_plan_endpoint_rejects_tool_intent_mismatch():
    payload = {
        "plan": {
            "planner_version": "ui_chaining_composer_v1",
            "tasks_summary": "mismatch",
            "dag_edges": [],
            "tasks": [
                {
                    "name": "IoTask",
                    "description": "Read data",
                    "instruction": "Fetch data only",
                    "acceptance_criteria": ["done"],
                    "expected_output_schema_ref": "",
                    "intent": "io",
                    "deps": [],
                    "tool_requests": ["llm_generate"],
                    "tool_inputs": {"llm_generate": {"text": "hello"}},
                    "critic_required": False,
                }
            ],
        },
        "job_context": {},
    }
    response = client.post("/plans/preflight", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["valid"] is False
    assert "IoTask" in body["errors"]
    assert body["errors"]["IoTask"].startswith("tool_intent_mismatch:llm_generate")


def test_preflight_plan_endpoint_rejects_capability_intent_mismatch(monkeypatch):
    capability = cap_registry.CapabilitySpec(
        capability_id="github.repo.list",
        description="List repos",
        risk_tier="read_only",
        idempotency="read",
        planner_hints={"task_intents": ["io"]},
        adapters=(
            cap_registry.CapabilityAdapterSpec(
                type="mcp",
                server_id="github_local",
                tool_name="search_repositories",
            ),
        ),
        enabled=True,
    )
    registry = cap_registry.CapabilityRegistry(capabilities={"github.repo.list": capability})
    monkeypatch.setattr(main.capability_registry, "resolve_capability_mode", lambda: "enabled")
    monkeypatch.setattr(main.capability_registry, "load_capability_registry", lambda: registry)

    payload = {
        "plan": {
            "planner_version": "ui_chaining_composer_v1",
            "tasks_summary": "capability mismatch",
            "dag_edges": [],
            "tasks": [
                {
                    "name": "BadCapabilityIntent",
                    "description": "Generate report",
                    "instruction": "Generate data",
                    "acceptance_criteria": ["done"],
                    "expected_output_schema_ref": "",
                    "intent": "generate",
                    "deps": [],
                    "tool_requests": ["github.repo.list"],
                    "tool_inputs": {"github.repo.list": {"query": "user:octocat"}},
                    "critic_required": False,
                }
            ],
        },
        "job_context": {},
    }
    response = client.post("/plans/preflight", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["valid"] is False
    assert "BadCapabilityIntent" in body["errors"]
    assert body["errors"]["BadCapabilityIntent"].startswith(
        "task_intent_mismatch:github.repo.list:generate"
    )


def test_preflight_plan_endpoint_returns_intent_segment_slot_diagnostics() -> None:
    payload = {
        "plan": {
            "planner_version": "ui_chaining_composer_v1",
            "tasks_summary": "segment contract",
            "dag_edges": [],
            "tasks": [
                {
                    "name": "GenerateText",
                    "description": "Generate text",
                    "instruction": "Generate the content",
                    "acceptance_criteria": ["done"],
                    "expected_output_schema_ref": "",
                    "intent": "generate",
                    "deps": [],
                    "tool_requests": ["llm_generate"],
                    "tool_inputs": {"llm_generate": {"text": "hello"}},
                    "critic_required": False,
                }
            ],
        },
        "goal_intent_graph": {
            "segments": [
                {
                    "id": "s1",
                    "intent": "generate",
                    "objective": "Generate report text",
                    "required_inputs": ["instruction"],
                    "suggested_capabilities": ["llm.text.generate"],
                    "slots": {
                        "entity": "report",
                        "artifact_type": "content",
                        "output_format": "txt",
                        "risk_level": "read_only",
                        "must_have_inputs": ["path"],
                    },
                }
            ]
        },
        "job_context": {},
    }
    response = client.post("/plans/preflight", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["valid"] is False
    assert body["errors"]["GenerateText"].startswith("intent_segment_invalid:llm_generate")
    diagnostics = body.get("diagnostics", [])
    assert isinstance(diagnostics, list) and diagnostics
    assert diagnostics[0]["code"] == "intent_segment.must_have_inputs_missing"
    assert diagnostics[0]["field"] == "GenerateText"


def test_preflight_plan_endpoint_accepts_intent_segment_tool_inputs_requirement() -> None:
    payload = {
        "plan": {
            "planner_version": "ui_chaining_composer_v1",
            "tasks_summary": "segment contract",
            "dag_edges": [],
            "tasks": [
                {
                    "name": "ImplementRepository",
                    "description": "Implement repository changes",
                    "instruction": "Use codegen.autonomous to implement the goal.",
                    "acceptance_criteria": ["done"],
                    "expected_output_schema_ref": "",
                    "intent": "generate",
                    "deps": [],
                    "tool_requests": ["llm_generate"],
                    "tool_inputs": {"llm_generate": {"text": "implement it"}},
                    "critic_required": False,
                }
            ],
        },
        "goal_intent_graph": {
            "segments": [
                {
                    "id": "s1",
                    "intent": "generate",
                    "objective": "Implement repository",
                    "required_inputs": ["instruction"],
                    "suggested_capabilities": ["llm.text.generate"],
                    "slots": {
                        "entity": "repository",
                        "artifact_type": "code",
                        "output_format": "txt",
                        "risk_level": "read_only",
                        "must_have_inputs": ["tool_inputs"],
                    },
                }
            ]
        },
        "job_context": {},
    }
    response = client.post("/plans/preflight", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["valid"] is True
    assert body["errors"] == {}


def test_preflight_task_intent_uses_goal_text_when_task_text_generic() -> None:
    task = models.TaskCreate(
        name="ValidateSpec",
        description="Step one",
        instruction="Handle this task.",
        acceptance_criteria=["Done"],
        expected_output_schema_ref="",
        deps=[],
        tool_requests=[],
        tool_inputs={},
        critic_required=False,
    )
    inferred = main._preflight_task_intent(
        task,
        goal_text="Validate the generated document against schema constraints.",
    )
    assert inferred == "validate"


def test_build_plan_from_composer_draft_derives_intent_from_goal(monkeypatch) -> None:
    capability = cap_registry.CapabilitySpec(
        capability_id="custom.step",
        description="Custom step",
        risk_tier="read_only",
        idempotency="read",
        planner_hints={},
        adapters=(
            cap_registry.CapabilityAdapterSpec(
                type="local_tool",
                server_id="local_worker",
                tool_name="json_transform",
            ),
        ),
        enabled=True,
    )
    registry = cap_registry.CapabilityRegistry(capabilities={"custom.step": capability})
    monkeypatch.setattr(main.capability_registry, "load_capability_registry", lambda: registry)

    plan, errors, _warnings = main._build_plan_from_composer_draft(
        {"nodes": [{"id": "n1", "capabilityId": "custom.step", "taskName": "CustomStep"}]},
        goal_text="Render the final document as PDF.",
    )
    assert not errors
    assert plan is not None
    assert plan.tasks[0].intent == models.ToolIntent.render


def test_build_plan_from_composer_draft_flags_control_flow_nodes() -> None:
    plan, errors, _warnings = main._build_plan_from_composer_draft(
        {
            "nodes": [
                {
                    "id": "n1",
                    "taskName": "ApprovalGate",
                    "nodeKind": "control",
                    "controlKind": "if_else",
                    "capabilityId": "studio.control.if_else",
                    "controlConfig": {"expression": "context.approved == true"},
                }
            ]
        }
    )

    assert plan is None
    codes = {entry["code"] for entry in errors}
    assert "draft.control_if_else_true_branch_missing" in codes
    assert "draft.control_if_else_false_branch_missing" in codes


def test_build_plan_from_composer_draft_validates_switch_control_cases() -> None:
    plan, errors, _warnings = main._build_plan_from_composer_draft(
        {
            "nodes": [
                {
                    "id": "n1",
                    "taskName": "RouteByStatus",
                    "nodeKind": "control",
                    "controlKind": "switch",
                    "capabilityId": "studio.control.switch",
                    "controlConfig": {
                        "expression": "context.status",
                        "switchCases": [{"id": "c1", "label": "", "match": ""}],
                    },
                }
            ]
        }
    )

    assert plan is None
    codes = {entry["code"] for entry in errors}
    assert "draft.control_switch_case_label_missing" in codes
    assert "draft.control_switch_case_match_missing" in codes


def test_build_plan_from_composer_draft_lowers_if_control_to_execution_gate(monkeypatch) -> None:
    capability = cap_registry.CapabilitySpec(
        capability_id="json_transform",
        description="Transform JSON",
        enabled=True,
        risk_tier="read_only",
        idempotency="read",
        output_schema_ref="schemas/json_object",
        planner_hints={"task_intents": ["transform"]},
        adapters=(
            cap_registry.CapabilityAdapterSpec(
                type="local_tool",
                server_id="local_worker",
                tool_name="json_transform",
            ),
        ),
    )
    registry = cap_registry.CapabilityRegistry(capabilities={"json_transform": capability})
    monkeypatch.setattr(main.capability_registry, "load_capability_registry", lambda: registry)

    plan, errors, _warnings = main._build_plan_from_composer_draft(
        {
            "nodes": [
                {
                    "id": "source",
                    "capabilityId": "json_transform",
                    "taskName": "LoadData",
                },
                {
                    "id": "gate",
                    "taskName": "OnlyIfApproved",
                    "nodeKind": "control",
                    "controlKind": "if",
                    "capabilityId": "studio.control.if",
                    "controlConfig": {"expression": "context.approved == true"},
                },
                {
                    "id": "target",
                    "capabilityId": "json_transform",
                    "taskName": "TransformData",
                },
            ],
            "edges": [
                {"fromNodeId": "source", "toNodeId": "gate"},
                {"fromNodeId": "gate", "toNodeId": "target"},
            ],
        }
    )

    assert errors == []
    assert plan is not None
    assert [task.name for task in plan.tasks] == ["LoadData", "TransformData"]
    assert plan.tasks[1].deps == ["LoadData"]
    assert plan.tasks[1].tool_inputs[main.execution_contracts.EXECUTION_GATE_KEY] == {
        "json_transform": {"expression": "context.approved == true"}
    }


def test_build_plan_from_composer_draft_lowers_if_else_to_branch_gates(monkeypatch) -> None:
    capability = cap_registry.CapabilitySpec(
        capability_id="json_transform",
        description="Transform JSON",
        enabled=True,
        risk_tier="read_only",
        idempotency="read",
        output_schema_ref="schemas/json_object",
        planner_hints={"task_intents": ["transform"]},
        adapters=(
            cap_registry.CapabilityAdapterSpec(
                type="local_tool",
                server_id="local_worker",
                tool_name="json_transform",
            ),
        ),
    )
    registry = cap_registry.CapabilityRegistry(capabilities={"json_transform": capability})
    monkeypatch.setattr(main.capability_registry, "load_capability_registry", lambda: registry)

    plan, errors, _warnings = main._build_plan_from_composer_draft(
        {
            "nodes": [
                {"id": "source", "capabilityId": "json_transform", "taskName": "LoadData"},
                {
                    "id": "gate",
                    "taskName": "ApprovalBranch",
                    "nodeKind": "control",
                    "controlKind": "if_else",
                    "capabilityId": "studio.control.if_else",
                    "controlConfig": {
                        "expression": "context.approved == true",
                        "trueLabel": "approved",
                        "falseLabel": "rejected",
                    },
                },
                {"id": "true-node", "capabilityId": "json_transform", "taskName": "ApprovedPath"},
                {"id": "false-node", "capabilityId": "json_transform", "taskName": "RejectedPath"},
                {"id": "join-node", "capabilityId": "json_transform", "taskName": "JoinedPath"},
            ],
            "edges": [
                {"fromNodeId": "source", "toNodeId": "gate"},
                {"fromNodeId": "gate", "toNodeId": "true-node", "branchLabel": "approved"},
                {"fromNodeId": "gate", "toNodeId": "false-node", "branchLabel": "rejected"},
                {"fromNodeId": "true-node", "toNodeId": "join-node"},
                {"fromNodeId": "false-node", "toNodeId": "join-node"},
            ],
        }
    )

    assert errors == []
    assert plan is not None
    task_by_name = {task.name: task for task in plan.tasks}
    assert task_by_name["ApprovedPath"].tool_inputs[main.execution_contracts.EXECUTION_GATE_KEY] == {
        "json_transform": {"expression": "context.approved == true"}
    }
    assert task_by_name["RejectedPath"].tool_inputs[main.execution_contracts.EXECUTION_GATE_KEY] == {
        "json_transform": {"expression": "context.approved == true", "negate": True}
    }
    assert main.execution_contracts.EXECUTION_GATE_KEY not in task_by_name["JoinedPath"].tool_inputs


def test_build_plan_from_composer_draft_requires_if_else_branch_labels(monkeypatch) -> None:
    capability = cap_registry.CapabilitySpec(
        capability_id="json_transform",
        description="Transform JSON",
        enabled=True,
        risk_tier="read_only",
        idempotency="read",
        output_schema_ref="schemas/json_object",
        planner_hints={"task_intents": ["transform"]},
        adapters=(
            cap_registry.CapabilityAdapterSpec(
                type="local_tool",
                server_id="local_worker",
                tool_name="json_transform",
            ),
        ),
    )
    registry = cap_registry.CapabilityRegistry(capabilities={"json_transform": capability})
    monkeypatch.setattr(main.capability_registry, "load_capability_registry", lambda: registry)

    plan, errors, _warnings = main._build_plan_from_composer_draft(
        {
            "nodes": [
                {
                    "id": "gate",
                    "taskName": "ApprovalBranch",
                    "nodeKind": "control",
                    "controlKind": "if_else",
                    "capabilityId": "studio.control.if_else",
                    "controlConfig": {
                        "expression": "context.approved == true",
                    },
                },
                {"id": "target", "capabilityId": "json_transform", "taskName": "TargetPath"},
            ],
            "edges": [
                {"fromNodeId": "gate", "toNodeId": "target"},
            ],
        }
    )

    assert plan is None
    codes = {entry["code"] for entry in errors}
    assert "draft.control_if_else_true_branch_missing" in codes
    assert "draft.control_if_else_false_branch_missing" in codes


def test_build_plan_from_composer_draft_lowers_parallel_fan_out(monkeypatch) -> None:
    capability = cap_registry.CapabilitySpec(
        capability_id="json_transform",
        description="Transform JSON",
        enabled=True,
        risk_tier="read_only",
        idempotency="read",
        output_schema_ref="schemas/json_object",
        planner_hints={"task_intents": ["transform"]},
        adapters=(
            cap_registry.CapabilityAdapterSpec(
                type="local_tool",
                server_id="local_worker",
                tool_name="json_transform",
            ),
        ),
    )
    registry = cap_registry.CapabilityRegistry(capabilities={"json_transform": capability})
    monkeypatch.setattr(main.capability_registry, "load_capability_registry", lambda: registry)

    plan, errors, _warnings = main._build_plan_from_composer_draft(
        {
            "nodes": [
                {"id": "source", "capabilityId": "json_transform", "taskName": "LoadData"},
                {
                    "id": "parallel",
                    "taskName": "FanOut",
                    "nodeKind": "control",
                    "controlKind": "parallel",
                    "capabilityId": "studio.control.parallel",
                    "controlConfig": {"parallelMode": "fan_out"},
                },
                {"id": "left", "capabilityId": "json_transform", "taskName": "LeftBranch"},
                {"id": "right", "capabilityId": "json_transform", "taskName": "RightBranch"},
            ],
            "edges": [
                {"fromNodeId": "source", "toNodeId": "parallel"},
                {"fromNodeId": "parallel", "toNodeId": "left"},
                {"fromNodeId": "parallel", "toNodeId": "right"},
            ],
        }
    )

    assert errors == []
    assert plan is not None
    task_by_name = {task.name: task for task in plan.tasks}
    assert task_by_name["LeftBranch"].deps == ["LoadData"]
    assert task_by_name["RightBranch"].deps == ["LoadData"]


def test_build_plan_from_composer_draft_lowers_parallel_fan_in(monkeypatch) -> None:
    capability = cap_registry.CapabilitySpec(
        capability_id="json_transform",
        description="Transform JSON",
        enabled=True,
        risk_tier="read_only",
        idempotency="read",
        output_schema_ref="schemas/json_object",
        planner_hints={"task_intents": ["transform"]},
        adapters=(
            cap_registry.CapabilityAdapterSpec(
                type="local_tool",
                server_id="local_worker",
                tool_name="json_transform",
            ),
        ),
    )
    registry = cap_registry.CapabilityRegistry(capabilities={"json_transform": capability})
    monkeypatch.setattr(main.capability_registry, "load_capability_registry", lambda: registry)

    plan, errors, _warnings = main._build_plan_from_composer_draft(
        {
            "nodes": [
                {"id": "left", "capabilityId": "json_transform", "taskName": "LeftBranch"},
                {"id": "right", "capabilityId": "json_transform", "taskName": "RightBranch"},
                {
                    "id": "parallel",
                    "taskName": "JoinBranches",
                    "nodeKind": "control",
                    "controlKind": "parallel",
                    "capabilityId": "studio.control.parallel",
                    "controlConfig": {"parallelMode": "fan_in"},
                },
                {"id": "target", "capabilityId": "json_transform", "taskName": "AfterJoin"},
            ],
            "edges": [
                {"fromNodeId": "left", "toNodeId": "parallel"},
                {"fromNodeId": "right", "toNodeId": "parallel"},
                {"fromNodeId": "parallel", "toNodeId": "target"},
            ],
        }
    )

    assert errors == []
    assert plan is not None
    task_by_name = {task.name: task for task in plan.tasks}
    assert task_by_name["AfterJoin"].deps == ["LeftBranch", "RightBranch"]


def test_build_plan_from_composer_draft_requires_parallel_fan_in_sources(monkeypatch) -> None:
    capability = cap_registry.CapabilitySpec(
        capability_id="json_transform",
        description="Transform JSON",
        enabled=True,
        risk_tier="read_only",
        idempotency="read",
        output_schema_ref="schemas/json_object",
        planner_hints={"task_intents": ["transform"]},
        adapters=(
            cap_registry.CapabilityAdapterSpec(
                type="local_tool",
                server_id="local_worker",
                tool_name="json_transform",
            ),
        ),
    )
    registry = cap_registry.CapabilityRegistry(capabilities={"json_transform": capability})
    monkeypatch.setattr(main.capability_registry, "load_capability_registry", lambda: registry)

    plan, errors, _warnings = main._build_plan_from_composer_draft(
        {
            "nodes": [
                {"id": "left", "capabilityId": "json_transform", "taskName": "LeftBranch"},
                {
                    "id": "parallel",
                    "taskName": "JoinBranches",
                    "nodeKind": "control",
                    "controlKind": "parallel",
                    "capabilityId": "studio.control.parallel",
                    "controlConfig": {"parallelMode": "fan_in"},
                },
                {"id": "target", "capabilityId": "json_transform", "taskName": "AfterJoin"},
            ],
            "edges": [
                {"fromNodeId": "left", "toNodeId": "parallel"},
                {"fromNodeId": "parallel", "toNodeId": "target"},
            ],
        }
    )

    assert plan is None
    codes = {entry["code"] for entry in errors}
    assert "draft.control_parallel_fan_in_sources_missing" in codes


def test_build_plan_from_composer_draft_injects_user_id_for_user_memory_bindings(monkeypatch) -> None:
    capability = cap_registry.CapabilitySpec(
        capability_id="json_transform",
        description="Transform JSON",
        enabled=True,
        risk_tier="read_only",
        idempotency="read",
        output_schema_ref="schemas/json_object",
        planner_hints={"task_intents": ["transform"]},
        adapters=(
            cap_registry.CapabilityAdapterSpec(
                type="local_tool",
                server_id="local_worker",
                tool_name="json_transform",
            ),
        ),
    )
    registry = cap_registry.CapabilityRegistry(capabilities={"json_transform": capability})
    monkeypatch.setattr(main.capability_registry, "load_capability_registry", lambda: registry)

    plan, errors, _warnings = main._build_plan_from_composer_draft(
        {
            "nodes": [
                {
                    "id": "n1",
                    "capabilityId": "json_transform",
                    "taskName": "LoadProfile",
                    "bindings": {
                        "profile": {"kind": "memory", "scope": "user", "name": "user_profile", "key": "profile"}
                    },
                }
            ]
        },
        job_context={"user_id": "narendersurabhi"},
    )

    assert errors == []
    assert plan is not None
    assert plan.tasks[0].tool_inputs["json_transform"]["profile"] == {
        "scope": "user",
        "name": "user_profile",
        "key": "profile",
        "user_id": "narendersurabhi",
    }


def test_build_plan_from_composer_draft_lowers_workflow_inputs_and_variables(monkeypatch) -> None:
    capability = cap_registry.CapabilitySpec(
        capability_id="json_transform",
        description="Transform JSON",
        enabled=True,
        risk_tier="read_only",
        idempotency="read",
        output_schema_ref="schemas/json_object",
        planner_hints={"task_intents": ["transform"]},
        adapters=(
            cap_registry.CapabilityAdapterSpec(
                type="local_tool",
                server_id="local_worker",
                tool_name="json_transform",
            ),
        ),
    )
    registry = cap_registry.CapabilityRegistry(capabilities={"json_transform": capability})
    monkeypatch.setattr(main.capability_registry, "load_capability_registry", lambda: registry)

    plan, errors, _warnings = main._build_plan_from_composer_draft(
        {
            "workflowInterface": {
                "inputs": [
                    {
                        "key": "topic",
                        "valueType": "string",
                        "required": True,
                    }
                ],
                "variables": [
                    {
                        "key": "topic_alias",
                        "binding": {"kind": "workflow_input", "inputKey": "topic"},
                    }
                ],
                "outputs": [],
            },
            "nodes": [
                {
                    "id": "n1",
                    "capabilityId": "json_transform",
                    "taskName": "UseInterface",
                    "bindings": {
                        "topic": {"kind": "workflow_input", "inputKey": "topic"},
                        "alias": {
                            "kind": "workflow_variable",
                            "variableKey": "topic_alias",
                        },
                    },
                }
            ],
        }
    )

    assert errors == []
    assert plan is not None
    assert plan.tasks[0].tool_inputs["json_transform"]["topic"] == {
        "$from": ["job_context", "workflow", "inputs", "topic"]
    }
    assert plan.tasks[0].tool_inputs["json_transform"]["alias"] == {
        "$from": ["job_context", "workflow", "variables", "topic_alias"]
    }


def test_workflow_version_run_accepts_explicit_inputs(monkeypatch) -> None:
    capability = cap_registry.CapabilitySpec(
        capability_id="json_transform",
        description="Transform JSON",
        enabled=True,
        risk_tier="read_only",
        idempotency="read",
        output_schema_ref="schemas/json_object",
        planner_hints={"task_intents": ["transform"]},
        adapters=(
            cap_registry.CapabilityAdapterSpec(
                type="local_tool",
                server_id="local_worker",
                tool_name="json_transform",
            ),
        ),
    )
    registry = cap_registry.CapabilityRegistry(capabilities={"json_transform": capability})
    monkeypatch.setattr(main.capability_registry, "load_capability_registry", lambda: registry)

    create_response = client.post(
        "/workflows/definitions",
        json={
            "title": "Workflow inputs demo",
            "goal": "Use workflow inputs",
            "context_json": {"user_id": "narendersurabhi"},
            "draft": {
                "summary": "Workflow inputs demo",
                "workflowInterface": {
                    "inputs": [
                        {
                            "key": "topic",
                            "valueType": "string",
                            "required": True,
                        }
                    ],
                    "variables": [
                        {
                            "key": "topic_alias",
                            "binding": {"kind": "workflow_input", "inputKey": "topic"},
                        }
                    ],
                    "outputs": [],
                },
                "nodes": [
                    {
                        "id": "n1",
                        "taskName": "Transform",
                        "capabilityId": "json_transform",
                        "bindings": {
                            "input": {
                                "kind": "workflow_variable",
                                "variableKey": "topic_alias",
                            },
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

    run_response = client.post(
        f"/workflows/versions/{version['id']}/run",
        json={"inputs": {"topic": "Quarterly update"}},
    )
    assert run_response.status_code == 200
    run_body = run_response.json()
    assert run_body["job"]["context_json"]["workflow"]["inputs"]["topic"] == "Quarterly update"
    assert (
        run_body["job"]["context_json"]["workflow"]["variables"]["topic_alias"]
        == "Quarterly update"
    )
    assert run_body["workflow_run"]["metadata"]["workflow_input_keys"] == ["topic"]

    with SessionLocal() as db:
        task_record = db.query(TaskRecord).filter(TaskRecord.job_id == run_body["job"]["id"]).first()
        assert task_record is not None
        assert task_record.tool_inputs["json_transform"]["input"] == {
            "$from": ["job_context", "workflow", "variables", "topic_alias"]
        }


def test_task_payload_from_record_flags_unresolved_reference_inputs() -> None:
    now = _utcnow()
    record = TaskRecord(
        id=f"task-ref-{uuid.uuid4()}",
        job_id="job-ref",
        plan_id="plan-ref",
        name="RenderDocument",
        description="Render document",
        instruction="Render",
        acceptance_criteria=["docx created"],
        expected_output_schema_ref="schemas/docx_output",
        status=models.TaskStatus.ready.value,
        deps=["GenerateDocumentSpec"],
        attempts=0,
        max_attempts=3,
        rework_count=0,
        max_reworks=0,
        assigned_to=None,
        intent="render",
        tool_requests=["docx_generate_from_spec"],
        tool_inputs={
            "docx_generate_from_spec": {
                "path": "documents/out.docx",
                "document_spec": {
                    "$from": "dependencies_by_name.GenerateDocumentSpec.llm_generate_document_spec.document_spec"
                },
            }
        },
        created_at=now,
        updated_at=now,
        critic_required=0,
    )
    payload = main._task_payload_from_record(record, correlation_id="corr", context={})
    validation = payload.get("tool_inputs_validation", {})
    assert "docx_generate_from_spec" in validation
    assert "input reference resolution failed" in validation["docx_generate_from_spec"]


def test_task_payload_from_record_resolves_validation_report_alias_reference() -> None:
    now = _utcnow()
    record = TaskRecord(
        id=f"task-improve-{uuid.uuid4()}",
        job_id="job-ref",
        plan_id="plan-ref",
        name="Improve_DocumentSpec",
        description="Improve doc spec",
        instruction="Improve",
        acceptance_criteria=["improved"],
        expected_output_schema_ref="schemas/document_spec",
        status=models.TaskStatus.ready.value,
        deps=["Generate_DocumentSpec", "Validate_DocumentSpec"],
        attempts=0,
        max_attempts=3,
        rework_count=0,
        max_reworks=0,
        assigned_to=None,
        intent="transform",
        tool_requests=["llm_improve_document_spec"],
        tool_inputs={
            "llm_improve_document_spec": {
                "document_spec": {
                    "$from": "dependencies_by_name.Generate_DocumentSpec.llm_generate_document_spec.document_spec"
                },
                "validation_report": {
                    "$from": "dependencies_by_name.Validate_DocumentSpec.document_spec_validate.validation_report"
                },
            }
        },
        created_at=now,
        updated_at=now,
        critic_required=0,
    )
    context = {
        "dependencies_by_name": {
            "Generate_DocumentSpec": {
                "llm_generate_document_spec": {
                    "document_spec": {"blocks": [{"type": "paragraph", "text": "hello"}]}
                }
            },
            "Validate_DocumentSpec": {
                "document_spec_validate": {
                    "valid": True,
                    "errors": [],
                    "warnings": [],
                    "stats": {"block_count": 1},
                }
            },
        },
        "dependencies": {},
    }

    payload = main._task_payload_from_record(record, correlation_id="corr", context=context)
    assert "tool_inputs_validation" not in payload
    resolved = payload["tool_inputs"]["llm_improve_document_spec"]
    assert resolved["document_spec"]["blocks"][0]["text"] == "hello"
    assert resolved["validation_report"]["valid"] is True


def test_task_payload_from_record_resolves_dotted_request_id_reference() -> None:
    now = _utcnow()
    record = TaskRecord(
        id=f"task-validate-{uuid.uuid4()}",
        job_id="job-ref",
        plan_id="plan-ref",
        name="Validate DocumentSpec",
        description="Validate doc spec",
        instruction="Validate",
        acceptance_criteria=["validated"],
        expected_output_schema_ref="schemas/validation_report",
        status=models.TaskStatus.ready.value,
        deps=["Generate DocumentSpec"],
        attempts=0,
        max_attempts=3,
        rework_count=0,
        max_reworks=0,
        assigned_to=None,
        intent="validate",
        tool_requests=["document_spec_validate"],
        tool_inputs={
            "document_spec_validate": {
                "strict": True,
                "document_spec": {
                    "$from": "dependencies_by_name.Generate DocumentSpec.document.spec.generate.document_spec"
                },
            }
        },
        created_at=now,
        updated_at=now,
        critic_required=0,
    )
    context = {
        "dependencies_by_name": {
            "Generate DocumentSpec": {
                "document.spec.generate": {
                    "document_spec": {"blocks": [{"type": "paragraph", "text": "hello"}]}
                }
            }
        },
        "dependencies": {},
    }

    payload = main._task_payload_from_record(record, correlation_id="corr", context=context)
    assert "tool_inputs_validation" not in payload
    resolved = payload["tool_inputs"]["document_spec_validate"]
    assert resolved["document_spec"]["blocks"][0]["text"] == "hello"


def test_task_payload_from_record_resolves_output_path_alias_reference() -> None:
    now = _utcnow()
    record = TaskRecord(
        id=f"task-render-{uuid.uuid4()}",
        job_id="job-ref",
        plan_id="plan-ref",
        name="Render DOCX",
        description="Render DOCX",
        instruction="Render DOCX",
        acceptance_criteria=["docx created"],
        expected_output_schema_ref="schemas/output_path",
        status=models.TaskStatus.ready.value,
        deps=["Derive Output Filename"],
        attempts=0,
        max_attempts=3,
        rework_count=0,
        max_reworks=0,
        assigned_to=None,
        intent="render",
        tool_requests=["docx_generate_from_spec"],
        tool_inputs={
            "docx_generate_from_spec": {
                "document_spec": {"blocks": []},
                "output_path": {
                    "$from": "dependencies_by_name.Derive Output Filename.derive_output_filename.output_path"
                },
            }
        },
        created_at=now,
        updated_at=now,
        critic_required=0,
    )
    context = {
        "dependencies_by_name": {
            "Derive Output Filename": {
                "derive_output_filename": {"path": "artifacts/output.docx"}
            }
        },
        "dependencies": {},
    }

    payload = main._task_payload_from_record(record, correlation_id="corr", context=context)
    assert "tool_inputs_validation" not in payload
    resolved = payload["tool_inputs"]["docx_generate_from_spec"]
    assert resolved["output_path"] == "artifacts/output.docx"


def test_task_payload_from_record_resolves_task_level_path_reference() -> None:
    now = _utcnow()
    record = TaskRecord(
        id=f"task-render-{uuid.uuid4()}",
        job_id="job-ref",
        plan_id="plan-ref",
        name="Render DOCX",
        description="Render DOCX",
        instruction="Render DOCX",
        acceptance_criteria=["docx created"],
        expected_output_schema_ref="schemas/output_path",
        status=models.TaskStatus.ready.value,
        deps=["DeriveResumeOutputPath"],
        attempts=0,
        max_attempts=3,
        rework_count=0,
        max_reworks=0,
        assigned_to=None,
        intent="render",
        tool_requests=["docx_generate_from_spec"],
        tool_inputs={
            "docx_generate_from_spec": {
                "document_spec": {"blocks": []},
                "output_path": {
                    "$from": "dependencies_by_name.DeriveResumeOutputPath.path"
                },
            }
        },
        created_at=now,
        updated_at=now,
        critic_required=0,
    )
    context = {
        "dependencies_by_name": {
            "DeriveResumeOutputPath": {
                "derive_output_filename": {"path": "artifacts/output.docx"}
            }
        },
        "dependencies": {},
    }

    payload = main._task_payload_from_record(record, correlation_id="corr", context=context)
    assert "tool_inputs_validation" not in payload
    resolved = payload["tool_inputs"]["docx_generate_from_spec"]
    assert resolved["output_path"] == "artifacts/output.docx"


def test_task_payload_from_record_includes_intent_segment_profile() -> None:
    now = _utcnow()
    record = TaskRecord(
        id=f"task-segment-{uuid.uuid4()}",
        job_id="job-ref",
        plan_id="plan-ref",
        name="RenderReport",
        description="Render report",
        instruction="Render",
        acceptance_criteria=["done"],
        expected_output_schema_ref="schemas/pdf_output",
        status=models.TaskStatus.ready.value,
        deps=[],
        attempts=0,
        max_attempts=3,
        rework_count=0,
        max_reworks=0,
        assigned_to=None,
        intent=None,
        tool_requests=["document.pdf.generate"],
        tool_inputs={"document.pdf.generate": {"path": "artifacts/report.pdf"}},
        created_at=now,
        updated_at=now,
        critic_required=0,
    )
    intent_profile = {
        "intent": "render",
        "source": "goal_graph",
        "confidence": 0.91,
        "segment": {
            "id": "s3",
            "intent": "render",
            "objective": "Render final PDF report",
            "required_inputs": ["document_spec", "path"],
            "suggested_capabilities": ["document.pdf.generate"],
            "slots": {
                "entity": "report",
                "artifact_type": "document",
                "output_format": "pdf",
                "risk_level": "bounded_write",
                "must_have_inputs": ["document_spec", "path"],
            },
        },
    }

    payload = main._task_payload_from_record(
        record,
        correlation_id="corr",
        context={},
        intent_profile=intent_profile,
    )
    assert payload["intent"] == "render"
    assert payload["intent_source"] == "goal_graph"
    assert payload["intent_confidence"] == 0.91
    assert payload["intent_segment"]["id"] == "s3"
    assert payload["intent_segment"]["slots"]["output_format"] == "pdf"
    assert payload["trace_id"] == "corr"
    parsed = main.execution_contracts.build_task_dispatch_payload(payload)
    assert parsed.correlation_id == "corr"
    assert parsed.trace_id == "corr"


def test_task_payload_from_record_uses_typed_dispatch_contract() -> None:
    now = _utcnow()
    record = TaskRecord(
        id=f"task-dispatch-{uuid.uuid4()}",
        job_id="job-ref",
        plan_id="plan-ref",
        name="GenerateSpec",
        description="Generate doc spec",
        instruction="Generate",
        acceptance_criteria=["done"],
        expected_output_schema_ref="schemas/document_spec",
        status=models.TaskStatus.ready.value,
        deps=[],
        attempts=0,
        max_attempts=0,
        rework_count=0,
        max_reworks=0,
        assigned_to=None,
        intent=None,
        tool_requests=["document.spec.generate"],
        tool_inputs=main.execution_contracts.embed_capability_bindings(
            {"document.spec.generate": {"job": {"topic": "latency"}}},
            {
                "document.spec.generate": {
                    "request_id": "document.spec.generate",
                    "capability_id": "document.spec.generate",
                    "tool_name": "document.spec.generate",
                    "adapter_type": "capability",
                }
            },
            request_ids=["document.spec.generate"],
        ),
        created_at=now,
        updated_at=now,
        critic_required=0,
    )

    payload = main._task_payload_from_record(record, correlation_id="corr", context={})

    dispatch = main.execution_contracts.build_task_dispatch_payload(payload)
    assert dispatch.task_id == record.id
    assert dispatch.plan_id == "plan-ref"
    assert dispatch.correlation_id == "corr"
    assert dispatch.trace_id == "corr"
    assert dispatch.attempts == 1
    assert dispatch.max_attempts == 1
    assert dispatch.tool_requests == ["document.spec.generate"]
    assert main.execution_contracts.EXECUTION_BINDINGS_KEY not in payload["tool_inputs"]
    assert dispatch.capability_bindings["document.spec.generate"].capability_id == (
        "document.spec.generate"
    )
    assert dispatch.tool_inputs_resolved is True


def test_plan_preflight_compiler_accepts_valid_dependency_chain() -> None:
    plan = models.PlanCreate(
        planner_version="test",
        tasks_summary="json chain",
        dag_edges=[["MakeJson", "ReuseJson"]],
        tasks=[
            models.TaskCreate(
                name="MakeJson",
                description="Build json",
                instruction="Build",
                acceptance_criteria=["done"],
                expected_output_schema_ref="schemas/json_object",
                deps=[],
                tool_requests=["json_transform"],
                tool_inputs={"json_transform": {"input": {"name": "demo"}}},
                critic_required=False,
            ),
            models.TaskCreate(
                name="ReuseJson",
                description="Reuse json",
                instruction="Reuse",
                acceptance_criteria=["done"],
                expected_output_schema_ref="schemas/json_object",
                deps=["MakeJson"],
                tool_requests=["json_transform"],
                tool_inputs={
                    "json_transform": {
                        "input": {"$from": "dependencies_by_name.MakeJson.json_transform.result"}
                    }
                },
                critic_required=False,
            ),
        ],
    )
    errors = main._compile_plan_preflight(plan, job_context={})
    assert errors == {}


def test_plan_preflight_compiler_flags_broken_reference_path() -> None:
    plan = models.PlanCreate(
        planner_version="test",
        tasks_summary="broken ref",
        dag_edges=[["MakeJson", "ReuseJson"]],
        tasks=[
            models.TaskCreate(
                name="MakeJson",
                description="Build json",
                instruction="Build",
                acceptance_criteria=["done"],
                expected_output_schema_ref="schemas/json_object",
                deps=[],
                tool_requests=["json_transform"],
                tool_inputs={"json_transform": {"input": {"name": "demo"}}},
                critic_required=False,
            ),
            models.TaskCreate(
                name="ReuseJson",
                description="Reuse json",
                instruction="Reuse",
                acceptance_criteria=["done"],
                expected_output_schema_ref="schemas/json_object",
                deps=["MakeJson"],
                tool_requests=["json_transform"],
                tool_inputs={
                    "json_transform": {
                        "input": {"$from": "dependencies_by_name.MakeJson.missing_tool.result"}
                    }
                },
                critic_required=False,
            ),
        ],
    )
    errors = main._compile_plan_preflight(plan, job_context={})
    assert "ReuseJson" in errors
    assert "input reference resolution failed" in errors["ReuseJson"]


def test_plan_preflight_accepts_reference_path_with_dotted_tool_name() -> None:
    plan = models.PlanCreate(
        planner_version="test",
        tasks_summary="github repo ref",
        dag_edges=[["Verify repository exists", "Decide whether to proceed"]],
        tasks=[
            models.TaskCreate(
                name="Verify repository exists",
                description="Verify repository exists",
                instruction="Check repo",
                acceptance_criteria=["done"],
                expected_output_schema_ref="schemas/github_repo_list_result",
                intent=models.ToolIntent.validate,
                deps=[],
                tool_requests=["github.repo.list"],
                tool_inputs={"github.repo.list": {"query": "repo:demo owner:octocat"}},
                critic_required=False,
            ),
            models.TaskCreate(
                name="Decide whether to proceed",
                description="Decide whether to proceed",
                instruction="Use repository check output",
                acceptance_criteria=["done"],
                expected_output_schema_ref="schemas/validation_report",
                intent=models.ToolIntent.validate,
                deps=["Verify repository exists"],
                tool_requests=["json_transform"],
                tool_inputs={
                    "json_transform": {
                        "input": {
                            "$from": "dependencies_by_name.Verify repository exists.github.repo.list"
                        }
                    }
                },
                critic_required=False,
            ),
        ],
    )
    errors = main._compile_plan_preflight(plan, job_context={})
    assert errors == {}


def test_plan_preflight_ignores_non_matching_intent_segments_when_suggested_capabilities_present() -> None:
    plan = models.PlanCreate(
        planner_version="test",
        tasks_summary="derive path",
        dag_edges=[],
        tasks=[
            models.TaskCreate(
                name="DeriveOutput",
                description="Derive filename",
                instruction="Derive output path",
                acceptance_criteria=["done"],
                expected_output_schema_ref="schemas/output_path",
                intent=models.ToolIntent.transform,
                deps=[],
                tool_requests=["derive_output_filename"],
                tool_inputs={
                    "derive_output_filename": {
                        "candidate_name": "Anjali Surabhi",
                        "target_role_name": "Associate Analyst",
                        "company_name": "Molina Healthcare",
                        "document_type": "document",
                        "output_extension": "docx",
                        "output_dir": "documents",
                        "today": "2026-02-28",
                    }
                },
                critic_required=False,
            )
        ],
    )
    goal_intent_graph = {
        "segments": [
            {
                "id": "s1",
                "intent": "render",
                "objective": "Render docx",
                "required_inputs": ["document_spec", "output_path"],
                "suggested_capabilities": ["document.docx.generate"],
            }
        ]
    }
    errors = main._compile_plan_preflight(
        plan,
        job_context={},
        goal_intent_graph=goal_intent_graph,
    )
    assert errors == {}


def test_plan_preflight_uses_task_instruction_for_intent_segment_contract() -> None:
    plan = models.PlanCreate(
        planner_version="test",
        tasks_summary="abort if missing",
        dag_edges=[["VerifyRepoExists", "AbortIfRepoMissing"]],
        tasks=[
            models.TaskCreate(
                name="VerifyRepoExists",
                description="Verify repository exists",
                instruction="Call github.repo.list to verify the repository exists.",
                acceptance_criteria=["done"],
                expected_output_schema_ref="schemas/github_repo_list_result",
                intent=models.ToolIntent.validate,
                deps=[],
                tool_requests=["github.repo.list"],
                tool_inputs={
                    "github.repo.list": {
                        "query": "repo:scientific-agent-lab owner:narendersurabhhi"
                    }
                },
                critic_required=False,
            ),
            models.TaskCreate(
                name="AbortIfRepoMissing",
                description="Stop if repository missing",
                instruction=(
                    "Inspect the VerifyRepoExists output; if the repository is not found, "
                    "mark the plan aborted and stop execution."
                ),
                acceptance_criteria=["done"],
                expected_output_schema_ref="schemas/validation_report",
                intent=models.ToolIntent.generate,
                deps=["VerifyRepoExists"],
                tool_requests=["llm_generate"],
                tool_inputs={},
                critic_required=True,
            ),
        ],
    )
    goal_intent_graph = {
        "segments": [
            {
                "id": "s1",
                "intent": "generate",
                "objective": "Stop if repository missing",
                "required_inputs": ["instruction"],
                "suggested_capabilities": ["llm.text.generate"],
                "slots": {
                    "entity": "repository",
                    "artifact_type": "validation_report",
                    "output_format": "txt",
                    "risk_level": "read_only",
                    "must_have_inputs": ["instruction"],
                },
            }
        ]
    }

    errors = main._compile_plan_preflight(
        plan,
        job_context={},
        goal_intent_graph=goal_intent_graph,
    )
    assert errors == {}


def test_plan_preflight_uses_synthesized_github_query_for_intent_segment_contract() -> None:
    plan = models.PlanCreate(
        planner_version="test",
        tasks_summary="repo check",
        dag_edges=[],
        tasks=[
            models.TaskCreate(
                name="CheckRepoExists",
                description="Verify repository exists",
                instruction="Call github.repo.list to verify the repository exists.",
                acceptance_criteria=["done"],
                expected_output_schema_ref="schemas/github_repo_list_result",
                intent=models.ToolIntent.validate,
                deps=[],
                tool_requests=["github.repo.list"],
                tool_inputs={"github.repo.list": {}},
                critic_required=False,
            )
        ],
    )
    goal_intent_graph = {
        "segments": [
            {
                "id": "s1",
                "intent": "validate",
                "objective": "Verify repository exists",
                "required_inputs": ["query"],
                "suggested_capabilities": ["github.repo.list"],
                "slots": {
                    "entity": "repository",
                    "artifact_type": "validation_report",
                    "output_format": None,
                    "risk_level": "read_only",
                    "must_have_inputs": ["query"],
                },
            }
        ]
    }

    errors = main._compile_plan_preflight(
        plan,
        job_context={
            "repo_owner": "narendersurabhi",
            "repo_name": "scientific-agent-lab",
        },
        goal_intent_graph=goal_intent_graph,
    )
    assert errors == {}


def test_retry_task_from_dlq_resets_task_and_deletes_stream_entry(monkeypatch):
    job_id = f"job-retry-task-{uuid.uuid4()}"
    plan_id = f"plan-retry-task-{uuid.uuid4()}"
    task_id = f"task-retry-task-{uuid.uuid4()}"
    now = _utcnow()
    with SessionLocal() as db:
        db.add(
            JobRecord(
                id=job_id,
                goal="retry one task",
                context_json={},
                status=models.JobStatus.failed.value,
                created_at=now,
                updated_at=now,
                priority=0,
                metadata_json={},
            )
        )
        db.add(
            PlanRecord(
                id=plan_id,
                job_id=job_id,
                planner_version="test",
                created_at=now,
                tasks_summary="single task",
                dag_edges=[],
                policy_decision={},
            )
        )
        db.add(
            TaskRecord(
                id=task_id,
                job_id=job_id,
                plan_id=plan_id,
                name="only-task",
                description="desc",
                instruction="do",
                acceptance_criteria=[],
                expected_output_schema_ref="TaskResult",
                status=models.TaskStatus.failed.value,
                deps=[],
                attempts=2,
                max_attempts=3,
                rework_count=1,
                max_reworks=2,
                assigned_to=None,
                intent=None,
                tool_requests=[],
                tool_inputs={},
                created_at=now,
                updated_at=now,
                critic_required=0,
            )
        )
        db.commit()

    class _RedisStub:
        def __init__(self):
            self.deleted = []

        def xdel(self, stream, stream_id):
            self.deleted.append((stream, stream_id))
            return 1

    redis_stub = _RedisStub()
    monkeypatch.setattr(main, "redis_client", redis_stub)
    captured = {"called": False}
    monkeypatch.setattr(
        main,
        "_enqueue_ready_tasks",
        lambda *args, **kwargs: captured.__setitem__("called", True),
    )

    response = client.post(
        f"/jobs/{job_id}/tasks/{task_id}/retry",
        json={"stream_id": "99-0"},
    )
    assert response.status_code == 200
    assert captured["called"] is True
    assert redis_stub.deleted == [(events.TASK_DLQ_STREAM, "99-0")]

    with SessionLocal() as db:
        refreshed = db.query(TaskRecord).filter(TaskRecord.id == task_id).first()
        assert refreshed is not None
        assert refreshed.status == models.TaskStatus.pending.value
        assert refreshed.attempts == 0
        assert refreshed.rework_count == 0


def test_retry_task_from_dlq_requires_failed_status():
    job_id = f"job-retry-task-state-{uuid.uuid4()}"
    plan_id = f"plan-retry-task-state-{uuid.uuid4()}"
    task_id = f"task-retry-task-state-{uuid.uuid4()}"
    now = _utcnow()
    with SessionLocal() as db:
        db.add(
            JobRecord(
                id=job_id,
                goal="retry one task state",
                context_json={},
                status=models.JobStatus.running.value,
                created_at=now,
                updated_at=now,
                priority=0,
                metadata_json={},
            )
        )
        db.add(
            PlanRecord(
                id=plan_id,
                job_id=job_id,
                planner_version="test",
                created_at=now,
                tasks_summary="single task",
                dag_edges=[],
                policy_decision={},
            )
        )
        db.add(
            TaskRecord(
                id=task_id,
                job_id=job_id,
                plan_id=plan_id,
                name="only-task",
                description="desc",
                instruction="do",
                acceptance_criteria=[],
                expected_output_schema_ref="TaskResult",
                status=models.TaskStatus.completed.value,
                deps=[],
                attempts=1,
                max_attempts=3,
                rework_count=0,
                max_reworks=2,
                assigned_to=None,
                intent=None,
                tool_requests=[],
                tool_inputs={},
                created_at=now,
                updated_at=now,
                critic_required=0,
            )
        )
        db.commit()

    response = client.post(f"/jobs/{job_id}/tasks/{task_id}/retry", json={"stream_id": "100-0"})
    assert response.status_code == 400
    assert response.json()["detail"] == "Task is not failed"
