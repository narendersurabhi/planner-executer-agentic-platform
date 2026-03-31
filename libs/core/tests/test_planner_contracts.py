from __future__ import annotations

from datetime import datetime

import pytest

from libs.core import capability_registry, models, planner_contracts, workflow_contracts


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
        name="docx_render_from_spec",
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
                        "suggested_capabilities": ["document.docx.render"],
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


def test_build_plan_request_defaults_render_path_mode_to_explicit() -> None:
    request = planner_contracts.build_plan_request(_job(), tools=[], capabilities={})

    assert request.render_path_mode == planner_contracts.RENDER_PATH_MODE_EXPLICIT


def test_build_plan_request_projects_stage_specific_planner_context() -> None:
    job = _job()
    job.context_json = {
        "title": "Senior AI Engineer cheat sheet",
        "user_profile": {"preferences": {"response_verbosity": "concise"}},
        "interaction_summaries": [
            {"facts": ["thanks"], "action": "thanks"},
            {"facts": ["document tone should be practical"], "action": "set document tone"},
        ],
        "interaction_summaries_ref": {"memory_name": "interaction_summaries_compact"},
        "interaction_summaries_meta": {"count": 2},
    }
    job.metadata = {
        "normalized_intent_envelope": workflow_contracts.dump_normalized_intent_envelope(
            workflow_contracts.NormalizedIntentEnvelope(
                goal=job.goal,
                profile=workflow_contracts.GoalIntentProfile(
                    intent="generate",
                    source="test",
                    missing_slots=["topic", "path"],
                ),
                graph=workflow_contracts.IntentGraph(
                    segments=[
                        workflow_contracts.IntentGraphSegment(
                            id="seg-1",
                            intent="generate",
                            objective="Render the document",
                            suggested_capabilities=["document.docx.render"],
                        )
                    ]
                ),
                candidate_capabilities={"seg-1": ["document.docx.render"]},
                clarification=workflow_contracts.ClarificationState(
                    missing_inputs=["topic", "path"],
                ),
            )
        )
        or {}
    }

    request = planner_contracts.build_plan_request(job, tools=[], capabilities={})

    assert "user_profile" not in request.job_context
    assert "interaction_summaries_ref" not in request.job_context
    assert "interaction_summaries_meta" not in request.job_context
    assert request.job_context["capability_candidates"] == ["document.docx.render"]
    assert request.job_context["missing_inputs"] == ["path"]
    assert request.job_context["interaction_summaries"] == [
        {"facts": ["document tone should be practical"], "action": "set document tone"}
    ]


def test_validate_render_path_requirement_accepts_job_context_reference() -> None:
    error = planner_contracts.validate_render_path_requirement(
        request_id="docx_render_from_spec",
        raw_payload={"path": {"$from": "job_context.path"}},
        resolved_payload={"path": "documents/report.docx"},
        job_context={"path": "documents/report.docx"},
        render_path_mode="explicit",
    )

    assert error is None


def test_validate_render_path_requirement_rejects_dependency_reference() -> None:
    error = planner_contracts.validate_render_path_requirement(
        request_id="docx_render_from_spec",
        raw_payload={
            "path": {
                "$from": "dependencies_by_name.DeriveOutputPath.derive_output_filename.path"
            }
        },
        resolved_payload={"path": "documents/report.docx"},
        job_context={},
        render_path_mode="explicit",
    )

    assert error == "render_path_derived_not_allowed:docx_render_from_spec"


def test_canonicalize_planner_request_ids_rewrites_aliases_and_adapter_tools() -> None:
    capability = capability_registry.CapabilitySpec(
        capability_id="document.docx.render",
        description="Render a DOCX",
        group="documents",
        subgroup="rendering",
        risk_tier="read_only",
        idempotency="read",
        aliases=("document.docx.generate",),
        adapters=(
            capability_registry.CapabilityAdapterSpec(
                type="tool",
                server_id="local_worker",
                tool_name="docx_render_from_spec",
                enabled=True,
            ),
        ),
    )

    canonicalized, rewrites = planner_contracts.canonicalize_planner_request_ids(
        [
            "document.docx.generate",
            "docx_render_from_spec",
            "document.docx.render",
        ],
        capabilities={"document.docx.render": capability},
    )

    assert canonicalized == ["document.docx.render"]
    assert rewrites == {
        "document.docx.generate": "document.docx.render",
        "docx_render_from_spec": "document.docx.render",
    }


def test_compile_task_request_payloads_compiles_capability_requests_to_runtime_tools() -> None:
    capability = capability_registry.CapabilitySpec(
        capability_id="document.spec.generate",
        description="Generate a document spec",
        group="documents",
        subgroup="generation",
        risk_tier="read_only",
        idempotency="read",
        adapters=(
            capability_registry.CapabilityAdapterSpec(
                type="tool",
                server_id="local_worker",
                tool_name="llm_generate_document_spec",
                enabled=True,
            ),
        ),
    )

    compiled = planner_contracts.compile_task_request_payloads(
        capability_requests=["document.spec.generate"],
        tool_inputs={"document.spec.generate": {"instruction": "Generate a document"}},
        capabilities={"document.spec.generate": capability},
    )

    assert compiled.request_ids == ["llm_generate_document_spec"]
    assert compiled.request_id_rewrites == {
        "document.spec.generate": "llm_generate_document_spec"
    }
    assert compiled.tool_inputs == {
        "llm_generate_document_spec": {"instruction": "Generate a document"}
    }
    assert compiled.capability_bindings["llm_generate_document_spec"]["capability_id"] == (
        "document.spec.generate"
    )


def test_validate_planner_request_language_rejects_raw_adapter_tool_name() -> None:
    capability = capability_registry.CapabilitySpec(
        capability_id="document.spec.generate",
        description="Generate a document spec",
        group="documents",
        subgroup="generation",
        risk_tier="read_only",
        idempotency="read",
        enabled=True,
        adapters=(
            capability_registry.CapabilityAdapterSpec(
                type="tool",
                server_id="local_worker",
                tool_name="llm_generate_document_spec",
                enabled=True,
            ),
        ),
    )

    error = planner_contracts.validate_planner_request_language(
        "llm_generate_document_spec",
        capabilities={"document.spec.generate": capability},
        full_capabilities={"document.spec.generate": capability},
        runtime_tool_names=["llm_generate_document_spec"],
    )

    assert error == (
        "planner_request_language_invalid:llm_generate_document_spec:"
        "use_capability_id:document.spec.generate"
    )


def test_validate_planner_request_language_supports_compat_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capability = capability_registry.CapabilitySpec(
        capability_id="document.spec.generate",
        description="Generate a document spec",
        group="documents",
        subgroup="generation",
        risk_tier="read_only",
        idempotency="read",
        enabled=True,
        adapters=(
            capability_registry.CapabilityAdapterSpec(
                type="tool",
                server_id="local_worker",
                tool_name="llm_generate_document_spec",
                enabled=True,
            ),
        ),
    )
    monkeypatch.setenv(
        "PLANNER_CAPABILITY_LANGUAGE_MODE",
        planner_contracts.PLANNER_CAPABILITY_LANGUAGE_MODE_COMPAT,
    )

    error = planner_contracts.validate_planner_request_language(
        "llm_generate_document_spec",
        capabilities={"document.spec.generate": capability},
        full_capabilities={"document.spec.generate": capability},
        runtime_tool_names=["llm_generate_document_spec"],
    )

    assert error is None
