from datetime import datetime
import json
from pathlib import Path

import pytest

from libs.core import models
from services.planner.app.main import (
    _ensure_default_value_markers,
    _ensure_job_inputs,
    _ensure_renderer_required_inputs,
    _ensure_resume_converter_inputs,
    _ensure_resume_docspec_inputs,
    _job_goal_intent_sequence,
    _ensure_renderer_output_extensions,
    _ensure_task_intents,
    _ensure_tool_input_dependencies,
    _llm_prompt,
    _validate_plan,
)


@pytest.fixture(autouse=True)
def _disable_governance(monkeypatch) -> None:
    monkeypatch.setenv("TOOL_GOVERNANCE_ENABLED", "false")


def _job() -> models.Job:
    now = datetime.utcnow()
    return models.Job(
        id="job-1",
        goal="test",
        context_json={"today": "2026-02-21", "topic": "distributed systems"},
        status=models.JobStatus.queued,
        created_at=now,
        updated_at=now,
        priority=0,
        metadata={},
    )


def _job_with_goal_intent_segment(segment: dict[str, object]) -> models.Job:
    job = _job()
    job.metadata = {"goal_intent_graph": {"segments": [segment]}}
    return job


def _plan_with_task(
    tool_name: str,
    tool_input: dict,
    deps: list[str] | None = None,
    *,
    intent: models.ToolIntent | None = models.ToolIntent.generate,
) -> models.PlanCreate:
    return models.PlanCreate(
        planner_version="1",
        tasks_summary="test",
        dag_edges=[],
        tasks=[
            models.TaskCreate(
                name="task-1",
                description="desc",
                instruction="instr",
                acceptance_criteria=["ok"],
                expected_output_schema_ref="schemas/test",
                intent=intent,
                deps=deps or [],
                tool_requests=[tool_name],
                tool_inputs={tool_name: tool_input},
                critic_required=False,
            )
        ],
    )


def _tool(
    name: str,
    input_schema: dict,
    *,
    memory_reads: list[str] | None = None,
    tool_intent: models.ToolIntent = models.ToolIntent.generate,
) -> models.ToolSpec:
    return models.ToolSpec(
        name=name,
        description="test tool",
        input_schema=input_schema,
        output_schema={"type": "object"},
        memory_reads=memory_reads or [],
        tool_intent=tool_intent,
    )


def test_ensure_task_intents_falls_back_to_goal_text() -> None:
    plan = _plan_with_task(
        "json_transform",
        {"input": {"x": 1}},
        intent=None,
    )
    updated = _ensure_task_intents(
        plan,
        [
            _tool(
                "json_transform",
                {"type": "object"},
                tool_intent=models.ToolIntent.transform,
            )
        ],
        goal_text="Validate the transformed payload against schema rules.",
    )
    assert updated.tasks[0].intent == models.ToolIntent.validate


def test_ensure_task_intents_can_use_goal_intent_sequence() -> None:
    plan = models.PlanCreate(
        planner_version="1",
        tasks_summary="test",
        dag_edges=[],
        tasks=[
            models.TaskCreate(
                name="task-1",
                description="Step one",
                instruction="Handle task.",
                acceptance_criteria=["ok"],
                expected_output_schema_ref="schemas/test",
                intent=None,
                deps=[],
                tool_requests=[],
                tool_inputs={},
                critic_required=False,
            ),
            models.TaskCreate(
                name="task-2",
                description="Step two",
                instruction="Handle task.",
                acceptance_criteria=["ok"],
                expected_output_schema_ref="schemas/test",
                intent=None,
                deps=["task-1"],
                tool_requests=[],
                tool_inputs={},
                critic_required=False,
            ),
        ],
    )
    updated = _ensure_task_intents(
        plan,
        [],
        goal_text="do work",
        goal_intent_sequence=["io", "render"],
    )
    assert updated.tasks[0].intent == models.ToolIntent.io
    assert updated.tasks[1].intent == models.ToolIntent.render


def test_ensure_task_intents_repairs_explicit_tool_mismatch() -> None:
    plan = _plan_with_task(
        "llm_generate",
        {"text": "hello"},
        intent=models.ToolIntent.io,
    )
    tools = [
        _tool(
            "llm_generate",
            {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
            tool_intent=models.ToolIntent.generate,
        )
    ]
    updated = _ensure_task_intents(plan, tools, goal_text="create content")
    assert updated.tasks[0].intent == models.ToolIntent.generate
    valid, reason = _validate_plan(updated, tools, _job())
    assert valid, reason


def test_ensure_default_value_markers_replaces_default_marker_with_context_value() -> None:
    plan = models.PlanCreate(
        planner_version="1",
        tasks_summary="derive path",
        dag_edges=[],
        tasks=[
            models.TaskCreate(
                name="DeriveOutputFilename",
                description="derive",
                instruction="derive",
                acceptance_criteria=["ok"],
                expected_output_schema_ref="schemas/output_path",
                intent=models.ToolIntent.transform,
                deps=[],
                tool_requests=["derive_output_filename"],
                tool_inputs={
                    "derive_output_filename": {
                        "candidate_name": {"$default": True},
                        "target_role_name": {"$default": True},
                        "company_name": {"$default": True},
                        "today": {"$default": True},
                        "document_type": "resume",
                        "output_extension": "docx",
                        "output_dir": "documents",
                    }
                },
                critic_required=False,
            )
        ],
    )
    now = datetime.utcnow()
    job = models.Job(
        id="job-default-marker",
        goal="derive output path",
        context_json={
            "candidate_name": "Anjali Surabhi",
            "target_role_name": "Associate Analyst",
            "company_name": "Molina Healthcare",
            "today": "2026-02-28",
        },
        status=models.JobStatus.queued,
        created_at=now,
        updated_at=now,
        priority=0,
        metadata={},
    )
    updated = _ensure_default_value_markers(plan, job)
    payload = updated.tasks[0].tool_inputs["derive_output_filename"]
    assert payload["candidate_name"] == "Anjali Surabhi"
    assert payload["target_role_name"] == "Associate Analyst"
    assert payload["company_name"] == "Molina Healthcare"
    assert payload["today"] == "2026-02-28"


def test_validate_plan_ignores_unmatched_goal_segment_for_task() -> None:
    plan = _plan_with_task(
        "document_spec_validate",
        {"document_spec": {"blocks": [{"type": "heading", "text": "Resume"}]}},
        intent=models.ToolIntent.validate,
    )
    tools = [
        _tool(
            "document_spec_validate",
            {
                "type": "object",
                "properties": {"document_spec": {"type": "object"}},
                "required": ["document_spec"],
            },
            tool_intent=models.ToolIntent.validate,
        )
    ]
    job = _job_with_goal_intent_segment(
        {
            "id": "s1",
            "intent": "transform",
            "objective": "tailor resume",
            "required_inputs": [
                "candidate_name",
                "company_name",
                "document_type",
                "output_dir",
                "output_extension",
                "target_role_name",
                "today",
            ],
            "suggested_capabilities": ["llm_tailor_resume_text"],
        }
    )
    valid, reason = _validate_plan(plan, tools, job)
    assert valid, reason


def test_validate_plan_segment_inputs_use_enriched_job_context() -> None:
    plan = _plan_with_task(
        "llm_generate_resume_doc_spec_from_text",
        {"tailored_resume": "tailored"},
        intent=models.ToolIntent.generate,
    )
    tools = [
        _tool(
            "llm_generate_resume_doc_spec_from_text",
            {
                "type": "object",
                "properties": {"tailored_resume": {"type": "string"}},
                "required": ["tailored_resume"],
            },
            tool_intent=models.ToolIntent.generate,
        )
    ]
    now = datetime.utcnow()
    job = models.Job(
        id="job-segment-context",
        goal="generate resume doc spec",
        context_json={
            "job_description": "JD",
            "candidate_resume": "Resume",
            "company_name": "Acme",
        },
        status=models.JobStatus.queued,
        created_at=now,
        updated_at=now,
        priority=0,
        metadata={
            "goal_intent_graph": {
                "segments": [
                    {
                        "id": "s1",
                        "intent": "generate",
                        "objective": "generate resume doc spec",
                        "required_inputs": [
                            "job_posting_text",
                            "original_resume_text",
                            "company_name",
                            "company_logo_image",
                        ],
                        "suggested_capabilities": [
                            "llm_generate_resume_doc_spec_from_text"
                        ],
                    }
                ]
            }
        },
    )
    valid, reason = _validate_plan(plan, tools, job)
    assert valid, reason


def test_ensure_tool_input_dependencies_adds_missing_deps_from_references() -> None:
    plan = models.PlanCreate(
        planner_version="1",
        tasks_summary="test",
        dag_edges=[],
        tasks=[
            models.TaskCreate(
                name="Generate DocumentSpec",
                description="produce spec",
                instruction="generate spec",
                acceptance_criteria=["ok"],
                expected_output_schema_ref="schemas/document_spec",
                intent=models.ToolIntent.generate,
                deps=[],
                tool_requests=["llm_generate_document_spec"],
                tool_inputs={},
                critic_required=False,
            ),
            models.TaskCreate(
                name="Validate DocumentSpec",
                description="validate spec",
                instruction="validate spec",
                acceptance_criteria=["ok"],
                expected_output_schema_ref="schemas/validation_report",
                intent=models.ToolIntent.validate,
                deps=["Generate DocumentSpec"],
                tool_requests=["document_spec_validate"],
                tool_inputs={},
                critic_required=False,
            ),
            models.TaskCreate(
                name="Fix DocumentSpec",
                description="repair spec",
                instruction="repair spec",
                acceptance_criteria=["ok"],
                expected_output_schema_ref="schemas/document_spec",
                intent=models.ToolIntent.transform,
                deps=["Validate DocumentSpec"],
                tool_requests=["llm_improve_document_spec"],
                tool_inputs={
                    "llm_improve_document_spec": {
                        "document_spec": {
                            "$from": "dependencies_by_name.Generate DocumentSpec.llm_generate_document_spec.document_spec"
                        },
                        "validation_report": {
                            "$from": "dependencies_by_name.Validate DocumentSpec.document_spec_validate.validation_report"
                        },
                    }
                },
                critic_required=False,
            ),
        ],
    )
    updated = _ensure_tool_input_dependencies(plan)
    fix_task = next(task for task in updated.tasks if task.name == "Fix DocumentSpec")
    assert fix_task.deps == ["Validate DocumentSpec", "Generate DocumentSpec"]
    assert ["Generate DocumentSpec", "Fix DocumentSpec"] in updated.dag_edges


def test_job_goal_intent_sequence_reads_job_metadata() -> None:
    job = _job()
    job.metadata = {
        "goal_intent_graph": {
            "segments": [
                {"id": "s1", "intent": "io"},
                {"id": "s2", "intent": "render"},
            ]
        }
    }
    assert _job_goal_intent_sequence(job) == ["io", "render"]


def test_llm_prompt_includes_intent_mismatch_recovery_constraints() -> None:
    job = _job()
    job.metadata = {
        "intent_mismatch_recovery": {
            "failed_task_name": "ListRepos",
            "failing_capability": "github.repo.list",
            "actual_task_intent": "generate",
            "allowed_task_intents": ["io"],
        }
    }
    prompt = _llm_prompt(
        job,
        [
            _tool(
                "github.repo.list",
                {"type": "object", "properties": {"owner": {"type": "string"}}},
                tool_intent=models.ToolIntent.io,
            )
        ],
    )
    assert "Intent mismatch auto-repair context" in prompt
    assert "github.repo.list" in prompt
    assert "allowed_task_intents" in prompt


def test_validate_plan_rejects_missing_root_required_with_anyof() -> None:
    plan = _plan_with_task("combo_tool", {"a": "value"})
    tool = _tool(
        "combo_tool",
        {
            "type": "object",
            "properties": {
                "base": {"type": "string"},
                "a": {"type": "string"},
                "b": {"type": "string"},
            },
            "required": ["base"],
            "anyOf": [{"required": ["a"]}, {"required": ["b"]}],
        },
    )
    valid, reason = _validate_plan(plan, [tool], _job())
    assert not valid
    assert reason.startswith("tool_inputs_invalid:combo_tool:task-1")


def test_validate_plan_enforces_nested_allof_contracts() -> None:
    plan = _plan_with_task("nested_tool", {})
    tool = _tool(
        "nested_tool",
        {
            "type": "object",
            "properties": {"target_role_name": {"type": "string", "minLength": 1}},
            "allOf": [{"required": ["target_role_name"]}],
        },
    )
    valid, reason = _validate_plan(plan, [tool], _job())
    assert not valid
    assert reason.startswith("tool_inputs_invalid:nested_tool:task-1")


def test_validate_plan_allows_dependency_filled_inputs() -> None:
    plan = _plan_with_task("improve_spec", {}, deps=["generate_spec"])
    tool = _tool(
        "improve_spec",
        {
            "type": "object",
            "properties": {
                "document_spec": {"type": "object"},
                "validation_report": {"type": "object"},
            },
            "required": ["document_spec", "validation_report"],
        },
    )
    valid, reason = _validate_plan(plan, [tool], _job())
    assert valid, reason


def test_validate_plan_allows_reference_object_inputs() -> None:
    plan = _plan_with_task(
        "docx_generate_from_spec",
        {
            "path": "documents/out.docx",
            "document_spec": {
                "$from": "dependencies_by_name.generate_spec.llm_generate_document_spec.document_spec"
            },
        },
        deps=["generate_spec"],
    )
    tool = _tool(
        "docx_generate_from_spec",
        {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "document_spec": {"type": "object"},
            },
            "required": ["path", "document_spec"],
        },
    )
    valid, reason = _validate_plan(plan, [tool], _job())
    assert valid, reason


def test_validate_plan_rejects_docx_render_without_output_path() -> None:
    plan = _plan_with_task("docx_render_like", {}, deps=["generate_data"])
    tool = _tool(
        "docx_render_like",
        {
            "type": "object",
            "properties": {
                "data": {"type": "object"},
                "template_id": {"type": "string"},
                "template_path": {"type": "string"},
                "output_path": {"type": "string"},
            },
            "required": ["data", "output_path"],
            "anyOf": [{"required": ["template_id"]}, {"required": ["template_path"]}],
        },
    )
    valid, reason = _validate_plan(plan, [tool], _job())
    assert not valid
    assert reason.startswith("tool_inputs_invalid:docx_render_like:task-1")


def test_validate_plan_accepts_memory_backed_contract_branch() -> None:
    plan = _plan_with_task("derive_output_filename", {})
    tool = _tool(
        "derive_output_filename",
        {
            "type": "object",
            "properties": {"topic": {"type": "string"}, "memory": {"type": "object"}},
            "allOf": [{"anyOf": [{"required": ["topic"]}, {"required": ["memory"]}]}],
        },
        memory_reads=["job_context", "task_outputs"],
    )
    valid, reason = _validate_plan(plan, [tool], _job())
    assert valid, reason


def test_validate_plan_rejects_tool_intent_mismatch() -> None:
    plan = _plan_with_task(
        "llm_generate",
        {"text": "hello"},
        intent=models.ToolIntent.io,
    )
    tool = _tool(
        "llm_generate",
        {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        tool_intent=models.ToolIntent.generate,
    )
    valid, reason = _validate_plan(plan, [tool], _job())
    assert not valid
    assert reason.startswith("tool_intent_mismatch:llm_generate:generate:io")


def test_validate_plan_rejects_missing_intent_segment_must_have_inputs() -> None:
    plan = _plan_with_task(
        "llm_generate",
        {},
        intent=models.ToolIntent.generate,
    )
    tool = _tool("llm_generate", {"type": "object"})
    job = _job_with_goal_intent_segment(
        {
            "id": "s1",
            "intent": "generate",
            "objective": "Generate summary",
            "required_inputs": ["instruction"],
            "suggested_capabilities": ["llm.text.generate"],
            "slots": {
                "entity": "summary",
                "artifact_type": "content",
                "output_format": "txt",
                "risk_level": "read_only",
                "must_have_inputs": ["instruction"],
            },
        }
    )
    valid, reason = _validate_plan(plan, [tool], job)
    assert not valid
    assert reason.startswith("intent_segment_invalid:llm_generate:task-1:must_have_inputs_missing:instruction")


def test_validate_plan_resolves_default_job_marker_for_segment_contract() -> None:
    plan = _plan_with_task(
        "llm_tailor_resume_text",
        {"job": {"$default": True}},
        intent=models.ToolIntent.generate,
    )
    tool = _tool(
        "llm_tailor_resume_text",
        {
            "type": "object",
            "properties": {"job": {"type": "object"}},
            "required": ["job"],
        },
    )
    job = _job_with_goal_intent_segment(
        {
            "id": "s1",
            "intent": "generate",
            "objective": "Tailor resume",
            "required_inputs": [
                "original_resume_text",
                "job_posting_text",
                "company_logo_image",
                "company_name",
            ],
            "suggested_capabilities": ["llm.text.generate"],
            "slots": None,
        }
    )
    job.context_json = {
        **dict(job.context_json or {}),
        "company_name": "Molina Healthcare",
        "job_description": "Analyst role",
        "candidate_resume": "Resume text",
    }
    plan_with_job = _ensure_job_inputs(plan, job, [tool])
    valid, reason = _validate_plan(plan_with_job, [tool], job)
    assert valid, reason


def test_ensure_resume_docspec_inputs_autowires_tailored_resume_reference() -> None:
    plan = models.PlanCreate(
        planner_version="1",
        tasks_summary="resume chain",
        dag_edges=[["Tailor Resume Text", "Generate Resume DocSpec From Text"]],
        tasks=[
            models.TaskCreate(
                name="Tailor Resume Text",
                description="tailor",
                instruction="tailor",
                acceptance_criteria=["ok"],
                expected_output_schema_ref="schemas/tailored_resume",
                intent=models.ToolIntent.generate,
                deps=[],
                tool_requests=["llm_tailor_resume_text"],
                tool_inputs={"llm_tailor_resume_text": {"job": {"$default": True}}},
                critic_required=False,
            ),
            models.TaskCreate(
                name="Generate Resume DocSpec From Text",
                description="doc spec",
                instruction="convert",
                acceptance_criteria=["ok"],
                expected_output_schema_ref="schemas/resume_doc_spec",
                intent=models.ToolIntent.generate,
                deps=["Tailor Resume Text"],
                tool_requests=["llm_generate_resume_doc_spec_from_text"],
                tool_inputs={"llm_generate_resume_doc_spec_from_text": {"target_pages": 1}},
                critic_required=False,
            ),
        ],
    )
    updated = _ensure_resume_docspec_inputs(plan)
    task = next(
        item for item in updated.tasks if item.name == "Generate Resume DocSpec From Text"
    )
    payload = dict(task.tool_inputs["llm_generate_resume_doc_spec_from_text"])
    assert payload["tailored_resume"] == {
        "$from": (
            "dependencies_by_name.Tailor Resume Text."
            "llm_tailor_resume_text.tailored_resume"
        )
    }


def test_ensure_resume_converter_inputs_autowires_resume_doc_spec_reference() -> None:
    plan = models.PlanCreate(
        planner_version="1",
        tasks_summary="resume chain",
        dag_edges=[["Generate Resume DocSpec From Text", "ResumeDocSpec to DocumentSpec"]],
        tasks=[
            models.TaskCreate(
                name="Generate Resume DocSpec From Text",
                description="doc spec",
                instruction="generate",
                acceptance_criteria=["ok"],
                expected_output_schema_ref="schemas/resume_doc_spec",
                intent=models.ToolIntent.generate,
                deps=[],
                tool_requests=["llm_generate_resume_doc_spec_from_text"],
                tool_inputs={"llm_generate_resume_doc_spec_from_text": {"target_pages": 1}},
                critic_required=False,
            ),
            models.TaskCreate(
                name="ResumeDocSpec to DocumentSpec",
                description="convert",
                instruction="convert",
                acceptance_criteria=["ok"],
                expected_output_schema_ref="schemas/document_spec",
                intent=models.ToolIntent.transform,
                deps=["Generate Resume DocSpec From Text"],
                tool_requests=["resume_doc_spec_to_document_spec"],
                tool_inputs={},
                critic_required=False,
            ),
        ],
    )
    updated = _ensure_resume_converter_inputs(plan)
    task = next(item for item in updated.tasks if item.name == "ResumeDocSpec to DocumentSpec")
    payload = dict(task.tool_inputs["resume_doc_spec_to_document_spec"])
    assert payload["resume_doc_spec"] == {
        "$from": (
            "dependencies_by_name.Generate Resume DocSpec From Text."
            "llm_generate_resume_doc_spec_from_text.resume_doc_spec"
        )
    }


def test_ensure_resume_converter_inputs_autowires_when_tasks_are_out_of_order() -> None:
    plan = models.PlanCreate(
        planner_version="1",
        tasks_summary="resume chain",
        dag_edges=[["Generate Resume DocSpec From Text", "ResumeDocSpec to DocumentSpec"]],
        tasks=[
            models.TaskCreate(
                name="ResumeDocSpec to DocumentSpec",
                description="convert",
                instruction="convert",
                acceptance_criteria=["ok"],
                expected_output_schema_ref="schemas/document_spec",
                intent=models.ToolIntent.transform,
                deps=[],
                tool_requests=["resume_doc_spec_to_document_spec"],
                tool_inputs={},
                critic_required=False,
            ),
            models.TaskCreate(
                name="Generate Resume DocSpec From Text",
                description="doc spec",
                instruction="generate",
                acceptance_criteria=["ok"],
                expected_output_schema_ref="schemas/resume_doc_spec",
                intent=models.ToolIntent.generate,
                deps=[],
                tool_requests=["llm_generate_resume_doc_spec_from_text"],
                tool_inputs={"llm_generate_resume_doc_spec_from_text": {"target_pages": 1}},
                critic_required=False,
            ),
        ],
    )
    updated = _ensure_resume_converter_inputs(plan)
    task = next(item for item in updated.tasks if item.name == "ResumeDocSpec to DocumentSpec")
    payload = dict(task.tool_inputs["resume_doc_spec_to_document_spec"])
    assert payload["resume_doc_spec"] == {
        "$from": (
            "dependencies_by_name.Generate Resume DocSpec From Text."
            "llm_generate_resume_doc_spec_from_text.resume_doc_spec"
        )
    }


def test_validate_plan_accepts_enabled_capability_with_valid_inputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    schema_path = tmp_path / "capability_input.schema.json"
    schema_path.write_text(
        json.dumps(
            {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            }
        ),
        encoding="utf-8",
    )
    capability_registry_path = tmp_path / "capability_registry.json"
    capability_registry_path.write_text(
        json.dumps(
            {
                "capabilities": [
                    {
                        "id": "github.repo.list",
                        "description": "List repos",
                        "enabled": True,
                        "input_schema_ref": str(schema_path),
                        "adapters": [
                            {
                                "type": "mcp",
                                "server_id": "github_remote",
                                "tool_name": "github_repo_list",
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("CAPABILITY_MODE", "enabled")
    monkeypatch.setenv("CAPABILITY_REGISTRY_PATH", str(capability_registry_path))
    plan = _plan_with_task("github.repo.list", {"query": "agentic"})
    valid, reason = _validate_plan(plan, [], _job())
    assert valid, reason


def test_validate_plan_rejects_capability_intent_mismatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    schema_path = tmp_path / "capability_input.schema.json"
    schema_path.write_text(
        json.dumps(
            {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            }
        ),
        encoding="utf-8",
    )
    capability_registry_path = tmp_path / "capability_registry.json"
    capability_registry_path.write_text(
        json.dumps(
            {
                "capabilities": [
                    {
                        "id": "github.repo.list",
                        "description": "List repos",
                        "enabled": True,
                        "input_schema_ref": str(schema_path),
                        "planner_hints": {"task_intents": ["io"]},
                        "adapters": [
                            {
                                "type": "mcp",
                                "server_id": "github_remote",
                                "tool_name": "github_repo_list",
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("CAPABILITY_MODE", "enabled")
    monkeypatch.setenv("CAPABILITY_REGISTRY_PATH", str(capability_registry_path))
    plan = _plan_with_task(
        "github.repo.list",
        {"query": "agentic"},
        intent=models.ToolIntent.generate,
    )
    valid, reason = _validate_plan(plan, [], _job())
    assert not valid
    assert reason.startswith("capability_intent_invalid:github.repo.list:task-1")


def test_validate_plan_rejects_capability_with_missing_required_inputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    schema_path = tmp_path / "capability_input.schema.json"
    schema_path.write_text(
        json.dumps(
            {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            }
        ),
        encoding="utf-8",
    )
    capability_registry_path = tmp_path / "capability_registry.json"
    capability_registry_path.write_text(
        json.dumps(
            {
                "capabilities": [
                    {
                        "id": "github.repo.list",
                        "description": "List repos",
                        "enabled": True,
                        "input_schema_ref": str(schema_path),
                        "adapters": [
                            {
                                "type": "mcp",
                                "server_id": "github_remote",
                                "tool_name": "github_repo_list",
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("CAPABILITY_MODE", "enabled")
    monkeypatch.setenv("CAPABILITY_REGISTRY_PATH", str(capability_registry_path))
    plan = _plan_with_task("github.repo.list", {})
    valid, reason = _validate_plan(plan, [], _job())
    assert not valid
    assert reason.startswith("capability_inputs_invalid:github.repo.list:task-1:")


def test_validate_plan_rejects_capability_risk_above_intent_segment_threshold(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    schema_path = tmp_path / "capability_input.schema.json"
    schema_path.write_text(
        json.dumps(
            {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            }
        ),
        encoding="utf-8",
    )
    capability_registry_path = tmp_path / "capability_registry.json"
    capability_registry_path.write_text(
        json.dumps(
            {
                "capabilities": [
                    {
                        "id": "filesystem.workspace.delete",
                        "description": "Delete a workspace path",
                        "enabled": True,
                        "risk_tier": "high_risk",
                        "input_schema_ref": str(schema_path),
                        "planner_hints": {"task_intents": ["generate"]},
                        "adapters": [
                            {
                                "type": "tool",
                                "server_id": "local_worker",
                                "tool_name": "workspace_delete",
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("CAPABILITY_MODE", "enabled")
    monkeypatch.setenv("CAPABILITY_REGISTRY_PATH", str(capability_registry_path))
    plan = _plan_with_task(
        "filesystem.workspace.delete",
        {"path": "workspace/tmp.txt"},
        intent=models.ToolIntent.generate,
    )
    job = _job_with_goal_intent_segment(
        {
            "id": "s1",
            "intent": "generate",
            "objective": "Generate report",
            "required_inputs": ["instruction"],
            "suggested_capabilities": ["llm.text.generate"],
            "slots": {
                "entity": "report",
                "artifact_type": "content",
                "output_format": "txt",
                "risk_level": "read_only",
                "must_have_inputs": [],
            },
        }
    )
    valid, reason = _validate_plan(plan, [], job)
    assert not valid
    assert reason.startswith(
        "intent_segment_invalid:filesystem.workspace.delete:task-1:risk_level_mismatch"
    )


def test_ensure_renderer_output_extensions_sets_pdf_on_derive_task(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    capability_registry_path = tmp_path / "capability_registry.json"
    capability_registry_path.write_text(
        json.dumps(
            {
                "capabilities": [
                    {
                        "id": "document.output.derive",
                        "description": "derive output path",
                        "enabled": True,
                        "planner_hints": {"derives_output_path": True},
                        "adapters": [
                            {
                                "type": "tool",
                                "server_id": "local_worker",
                                "tool_name": "derive_output_path",
                            }
                        ],
                    },
                    {
                        "id": "document.pdf.generate",
                        "description": "render pdf",
                        "enabled": True,
                        "planner_hints": {"required_output_extension": "pdf"},
                        "adapters": [
                            {
                                "type": "tool",
                                "server_id": "local_worker",
                                "tool_name": "pdf_generate_from_spec",
                            }
                        ],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("CAPABILITY_MODE", "enabled")
    monkeypatch.setenv("CAPABILITY_REGISTRY_PATH", str(capability_registry_path))
    plan = models.PlanCreate(
        planner_version="1",
        tasks_summary="doc pipeline",
        dag_edges=[],
        tasks=[
            models.TaskCreate(
                name="GenerateSpec",
                description="gen",
                instruction="gen",
                acceptance_criteria=["ok"],
                expected_output_schema_ref="schemas/DocumentSpec",
                deps=[],
                tool_requests=["document.spec.generate"],
                tool_inputs={"document.spec.generate": {"job": {}}},
                critic_required=False,
            ),
            models.TaskCreate(
                name="DeriveOutputPath",
                description="derive",
                instruction="derive",
                acceptance_criteria=["ok"],
                expected_output_schema_ref="schemas/docx_path",
                deps=["GenerateSpec"],
                tool_requests=["document.output.derive"],
                tool_inputs={
                    "document.output.derive": {
                        "topic": "Latency",
                        "today": "2026-02-24",
                        "output_dir": "documents",
                        "document_type": "document",
                    }
                },
                critic_required=False,
            ),
            models.TaskCreate(
                name="RenderPdf",
                description="render",
                instruction="render",
                acceptance_criteria=["ok"],
                expected_output_schema_ref="schemas/pdf_output",
                deps=["GenerateSpec", "DeriveOutputPath"],
                tool_requests=["document.pdf.generate"],
                tool_inputs={
                    "document.pdf.generate": {
                        "document_spec": {
                            "$from": [
                                "dependencies_by_name",
                                "GenerateSpec",
                                "document.spec.generate",
                                "document_spec",
                            ]
                        },
                        "path": {
                            "$from": [
                                "dependencies_by_name",
                                "DeriveOutputPath",
                                "document.output.derive",
                                "path",
                            ]
                        },
                    }
                },
                critic_required=False,
            ),
        ],
    )
    updated = _ensure_renderer_output_extensions(plan)
    derive_inputs = updated.tasks[1].tool_inputs["document.output.derive"]
    assert derive_inputs["output_extension"] == "pdf"


def test_ensure_renderer_output_extensions_keeps_explicit_extension_when_aligned(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    capability_registry_path = tmp_path / "capability_registry.json"
    capability_registry_path.write_text(
        json.dumps(
            {
                "capabilities": [
                    {
                        "id": "document.output.derive",
                        "description": "derive output path",
                        "enabled": True,
                        "planner_hints": {"derives_output_path": True},
                        "adapters": [
                            {
                                "type": "tool",
                                "server_id": "local_worker",
                                "tool_name": "derive_output_path",
                            }
                        ],
                    },
                    {
                        "id": "document.pdf.generate",
                        "description": "render pdf",
                        "enabled": True,
                        "planner_hints": {"required_output_extension": "pdf"},
                        "adapters": [
                            {
                                "type": "tool",
                                "server_id": "local_worker",
                                "tool_name": "pdf_generate_from_spec",
                            }
                        ],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("CAPABILITY_MODE", "enabled")
    monkeypatch.setenv("CAPABILITY_REGISTRY_PATH", str(capability_registry_path))
    plan = models.PlanCreate(
        planner_version="1",
        tasks_summary="doc pipeline",
        dag_edges=[],
        tasks=[
            models.TaskCreate(
                name="DeriveOutputPath",
                description="derive",
                instruction="derive",
                acceptance_criteria=["ok"],
                expected_output_schema_ref="schemas/docx_path",
                deps=[],
                tool_requests=["document.output.derive"],
                tool_inputs={
                    "document.output.derive": {
                        "topic": "Latency",
                        "today": "2026-02-24",
                        "output_dir": "documents",
                        "document_type": "document",
                        "output_extension": "pdf",
                    }
                },
                critic_required=False,
            ),
            models.TaskCreate(
                name="RenderPdf",
                description="render",
                instruction="render",
                acceptance_criteria=["ok"],
                expected_output_schema_ref="schemas/pdf_output",
                deps=["DeriveOutputPath"],
                tool_requests=["document.pdf.generate"],
                tool_inputs={
                    "document.pdf.generate": {
                        "path": {
                            "$from": [
                                "dependencies_by_name",
                                "DeriveOutputPath",
                                "document.output.derive",
                                "path",
                            ]
                        }
                    }
                },
                critic_required=False,
            ),
        ],
    )
    updated = _ensure_renderer_output_extensions(plan)
    derive_inputs = updated.tasks[0].tool_inputs["document.output.derive"]
    assert derive_inputs["output_extension"] == "pdf"


def test_ensure_renderer_output_extensions_autofills_document_type_for_resume_chain(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    capability_registry_path = tmp_path / "capability_registry.json"
    capability_registry_path.write_text(
        json.dumps(
            {
                "capabilities": [
                    {
                        "id": "document.output.derive",
                        "description": "derive output path",
                        "enabled": True,
                        "planner_hints": {"derives_output_path": True},
                        "adapters": [
                            {
                                "type": "tool",
                                "server_id": "local_worker",
                                "tool_name": "derive_output_path",
                            }
                        ],
                    },
                    {
                        "id": "document.docx.generate",
                        "description": "render docx",
                        "enabled": True,
                        "planner_hints": {"required_output_extension": "docx"},
                        "adapters": [
                            {
                                "type": "tool",
                                "server_id": "local_worker",
                                "tool_name": "docx_generate_from_spec",
                            }
                        ],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("CAPABILITY_MODE", "enabled")
    monkeypatch.setenv("CAPABILITY_REGISTRY_PATH", str(capability_registry_path))
    plan = models.PlanCreate(
        planner_version="1",
        tasks_summary="resume chain",
        dag_edges=[],
        tasks=[
            models.TaskCreate(
                name="Generate Resume DocSpec From Text",
                description="doc spec",
                instruction="generate",
                acceptance_criteria=["ok"],
                expected_output_schema_ref="schemas/resume_doc_spec",
                deps=[],
                tool_requests=["llm_generate_resume_doc_spec_from_text"],
                tool_inputs={"llm_generate_resume_doc_spec_from_text": {"target_pages": 1}},
                critic_required=False,
            ),
            models.TaskCreate(
                name="PathDerive",
                description="derive",
                instruction="derive",
                acceptance_criteria=["ok"],
                expected_output_schema_ref="schemas/docx_path",
                deps=["Generate Resume DocSpec From Text"],
                tool_requests=["document.output.derive"],
                tool_inputs={"document.output.derive": {"topic": "Resume"}},
                critic_required=False,
            ),
            models.TaskCreate(
                name="RenderDocx",
                description="render",
                instruction="render",
                acceptance_criteria=["ok"],
                expected_output_schema_ref="schemas/docx_output",
                deps=["PathDerive"],
                tool_requests=["document.docx.generate"],
                tool_inputs={
                    "document.docx.generate": {
                        "path": {
                            "$from": [
                                "dependencies_by_name",
                                "PathDerive",
                                "document.output.derive",
                                "path",
                            ]
                        }
                    }
                },
                critic_required=False,
            ),
        ],
    )
    updated = _ensure_renderer_output_extensions(plan)
    derive_inputs = updated.tasks[1].tool_inputs["document.output.derive"]
    assert derive_inputs["output_extension"] == "docx"
    assert derive_inputs["document_type"] == "resume"


def test_ensure_renderer_required_inputs_autowires_document_spec_and_path() -> None:
    plan = models.PlanCreate(
        planner_version="1",
        tasks_summary="resume chain",
        dag_edges=[],
        tasks=[
            models.TaskCreate(
                name="ResumeDocSpec to DocumentSpec",
                description="convert",
                instruction="convert",
                acceptance_criteria=["ok"],
                expected_output_schema_ref="schemas/document_spec",
                intent=models.ToolIntent.transform,
                deps=[],
                tool_requests=["resume_doc_spec_to_document_spec"],
                tool_inputs={"resume_doc_spec_to_document_spec": {"resume_doc_spec": {}}},
                critic_required=False,
            ),
            models.TaskCreate(
                name="PathDerive",
                description="derive",
                instruction="derive",
                acceptance_criteria=["ok"],
                expected_output_schema_ref="schemas/docx_path",
                intent=models.ToolIntent.io,
                deps=[],
                tool_requests=["derive_output_filename"],
                tool_inputs={"derive_output_filename": {"document_type": "resume"}},
                critic_required=False,
            ),
            models.TaskCreate(
                name="RenderResumeDocx",
                description="render",
                instruction="render",
                acceptance_criteria=["ok"],
                expected_output_schema_ref="schemas/docx_output",
                intent=models.ToolIntent.transform,
                deps=[],
                tool_requests=["docx_generate_from_spec"],
                tool_inputs={},
                critic_required=False,
            ),
        ],
    )
    updated = _ensure_renderer_required_inputs(plan)
    render_task = next(item for item in updated.tasks if item.name == "RenderResumeDocx")
    payload = dict(render_task.tool_inputs["docx_generate_from_spec"])
    assert payload["document_spec"] == {
        "$from": (
            "dependencies_by_name.ResumeDocSpec to DocumentSpec."
            "resume_doc_spec_to_document_spec.document_spec"
        )
    }
    assert payload["path"] == {
        "$from": "dependencies_by_name.PathDerive.derive_output_filename.path"
    }


def test_ensure_renderer_required_inputs_repairs_invalid_reference_task_names() -> None:
    plan = models.PlanCreate(
        planner_version="1",
        tasks_summary="resume chain",
        dag_edges=[],
        tasks=[
            models.TaskCreate(
                name="Convert ResumeDocSpec",
                description="convert",
                instruction="convert",
                acceptance_criteria=["ok"],
                expected_output_schema_ref="schemas/document_spec",
                intent=models.ToolIntent.transform,
                deps=[],
                tool_requests=["resume_doc_spec_to_document_spec"],
                tool_inputs={"resume_doc_spec_to_document_spec": {"resume_doc_spec": {}}},
                critic_required=False,
            ),
            models.TaskCreate(
                name="PathDerive",
                description="derive",
                instruction="derive",
                acceptance_criteria=["ok"],
                expected_output_schema_ref="schemas/docx_path",
                intent=models.ToolIntent.io,
                deps=[],
                tool_requests=["derive_output_filename"],
                tool_inputs={"derive_output_filename": {"document_type": "resume"}},
                critic_required=False,
            ),
            models.TaskCreate(
                name="RenderResumeDocx",
                description="render",
                instruction="render",
                acceptance_criteria=["ok"],
                expected_output_schema_ref="schemas/docx_output",
                intent=models.ToolIntent.transform,
                deps=[],
                tool_requests=["docx_generate_from_spec"],
                tool_inputs={
                    "docx_generate_from_spec": {
                        "document_spec": {
                            "$from": "dependencies_by_name.Convert ResumeDocSpec To DocumentSpec.resume_doc_spec_to_document_spec.document_spec"
                        },
                        "path": {
                            "$from": "dependencies_by_name.Derive Output Filename.derive_output_filename.path"
                        },
                    }
                },
                critic_required=False,
            ),
        ],
    )
    updated = _ensure_renderer_required_inputs(plan)
    render_task = next(item for item in updated.tasks if item.name == "RenderResumeDocx")
    payload = dict(render_task.tool_inputs["docx_generate_from_spec"])
    assert payload["document_spec"] == {
        "$from": "dependencies_by_name.Convert ResumeDocSpec.resume_doc_spec_to_document_spec.document_spec"
    }
    assert payload["path"] == {
        "$from": "dependencies_by_name.PathDerive.derive_output_filename.path"
    }
