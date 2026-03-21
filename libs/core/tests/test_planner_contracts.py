from __future__ import annotations

from datetime import datetime

from libs.core import capability_registry, models, planner_contracts


def _job() -> models.Job:
    return models.Job(
        id="job-1",
        goal="Create a DOCX from markdown",
        context_json={"output_dir": "artifacts", "repo_owner": "narendersurabhi"},
        status=models.JobStatus.queued,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
        priority=0,
        metadata={
            "job_type": "document",
            "goal_intent_graph": {
                "segments": [
                    {
                        "id": "seg-1",
                        "intent": "render",
                        "objective": "Render the document",
                    }
                ]
            },
        },
    )


def test_build_plan_request_extracts_intent_graph_and_capabilities() -> None:
    tool = models.ToolSpec(
        name="docx_generate_from_spec",
        description="Render a DOCX",
        input_schema={},
        output_schema={},
        tool_intent=models.ToolIntent.render,
    )
    capability = capability_registry.CapabilitySpec(
        capability_id="github.repo.list",
        description="Check for a repo",
        group="github",
        subgroup="repo",
        risk_tier="low",
        idempotency="read",
        input_schema_ref="schemas/github.repo.list",
        output_schema_ref="schemas/github.repo.list.output",
        adapters=(
            capability_registry.CapabilityAdapterSpec(
                type="mcp",
                server_id="github_local",
                tool_name="github.repo.list",
                enabled=True,
            ),
        ),
        planner_hints={"task_intents": ["io"]},
    )

    request = planner_contracts.build_plan_request(
        _job(),
        tools=[tool],
        capabilities={"github.repo.list": capability},
        semantic_capability_hints=[{"capability_id": "github.repo.list", "score": 0.9}],
        max_dependency_depth=4,
    )

    assert request.job_id == "job-1"
    assert request.goal_intent_graph is not None
    assert planner_contracts.goal_intent_sequence(request) == ["render"]
    assert planner_contracts.capability_map(request)["github.repo.list"].planner_hints == {
        "task_intents": ["io"]
    }
    assert request.semantic_capability_hints[0]["capability_id"] == "github.repo.list"
    assert request.max_dependency_depth == 4


def test_build_plan_request_prefers_normalized_envelope_graph_when_present() -> None:
    job = _job()
    job.metadata = {
        "goal_intent_graph": {
            "segments": [
                {
                    "id": "legacy-seg",
                    "intent": "generate",
                    "objective": "Legacy graph segment",
                }
            ]
        },
        "normalized_intent_envelope": {
            "goal": job.goal,
            "profile": {"intent": "render", "source": "llm"},
            "graph": {
                "segments": [
                    {
                        "id": "env-seg",
                        "intent": "render",
                        "objective": "Render the document",
                        "suggested_capabilities": ["document.docx.generate"],
                    }
                ]
            },
        },
    }

    request = planner_contracts.build_plan_request(job, tools=[], capabilities={})

    assert request.normalized_intent_envelope is not None
    assert request.goal_intent_graph is not None
    assert request.goal_intent_graph.segments[0].id == "env-seg"
    assert planner_contracts.goal_intent_sequence(request) == ["render"]


def test_build_plan_request_supports_envelope_only_job_metadata() -> None:
    job = _job()
    job.metadata = {
        "normalized_intent_envelope": {
            "goal": job.goal,
            "profile": {"intent": "validate", "source": "llm"},
            "graph": {
                "segments": [
                    {
                        "id": "env-only-seg",
                        "intent": "validate",
                        "objective": "Validate repository state",
                        "suggested_capabilities": ["github.repo.list"],
                    }
                ]
            },
        }
    }

    request = planner_contracts.build_plan_request(job, tools=[], capabilities={})

    assert request.normalized_intent_envelope is not None
    assert request.goal_intent_graph is not None
    assert request.goal_intent_graph.segments[0].id == "env-only-seg"
    assert planner_contracts.goal_intent_sequence(request) == ["validate"]


def test_governance_context_uses_request_metadata() -> None:
    request = planner_contracts.build_plan_request(_job(), tools=[], capabilities={})

    context = planner_contracts.governance_context(request)

    assert context["job_id"] == "job-1"
    assert context["job_type"] == "document"
    assert context["job_context"]["output_dir"] == "artifacts"
