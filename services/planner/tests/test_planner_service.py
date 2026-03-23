from __future__ import annotations

from datetime import UTC, datetime

from libs.core import llm_provider, models, planner_contracts
from services.planner.app import planner_service


def _job() -> models.Job:
    return models.Job(
        id="job-1",
        goal="Generate a plan",
        context_json={"topic": "demo"},
        status=models.JobStatus.queued,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        priority=0,
        metadata={},
    )


class _Provider(llm_provider.LLMProvider):
    def generate_request(self, request: llm_provider.LLMRequest) -> llm_provider.LLMResponse:
        del request
        return llm_provider.LLMResponse(
            content=(
                '{"planner_version":"1.0.0","tasks_summary":"demo","dag_edges":[],"tasks":[]}'
            )
        )


def test_build_plan_request_adds_semantic_hints() -> None:
    runtime = planner_service.PlannerServiceRuntime(
        load_capabilities=lambda: {},
        build_semantic_capability_hints=lambda job, capabilities, limit: [
            {"capability_id": "demo", "score": 0.9, "goal": job.goal, "limit": limit}
        ],
        parse_llm_plan=lambda content: models.PlanCreate.model_validate_json(content),
        ensure_llm_tool=lambda plan: plan,
        ensure_task_intents=lambda plan, request: plan,
        ensure_job_inputs=lambda plan, request: plan,
        ensure_default_value_markers=lambda plan, request: plan,
        ensure_renderer_required_inputs=lambda plan: plan,
        ensure_tool_input_dependencies=lambda plan: plan,
        ensure_renderer_output_extensions=lambda plan: plan,
        ensure_execution_bindings=lambda plan: plan,
        apply_max_depth=lambda plan, max_depth: plan,
    )

    request = planner_service.build_plan_request(
        _job(),
        [],
        config=planner_service.PlannerServiceConfig(mode="llm", semantic_hint_limit=5),
        runtime=runtime,
    )

    assert isinstance(request, planner_contracts.PlanRequest)
    assert request.semantic_capability_hints[0]["capability_id"] == "demo"
    assert request.semantic_capability_hints[0]["limit"] == 5


def test_plan_job_uses_request_boundary_for_llm_path() -> None:
    captured: list[planner_contracts.PlanRequest] = []

    def _ensure_task_intents(
        plan: models.PlanCreate,
        request: planner_contracts.PlanRequest,
    ) -> models.PlanCreate:
        captured.append(request)
        return plan

    runtime = planner_service.PlannerServiceRuntime(
        load_capabilities=lambda: {},
        build_semantic_capability_hints=lambda job, capabilities, limit: [],
        parse_llm_plan=lambda content: models.PlanCreate.model_validate_json(content),
        ensure_llm_tool=lambda plan: plan,
        ensure_task_intents=_ensure_task_intents,
        ensure_job_inputs=lambda plan, request: plan,
        ensure_default_value_markers=lambda plan, request: plan,
        ensure_renderer_required_inputs=lambda plan: plan,
        ensure_tool_input_dependencies=lambda plan: plan,
        ensure_renderer_output_extensions=lambda plan: plan,
        ensure_execution_bindings=lambda plan: plan,
        apply_max_depth=lambda plan, max_depth: plan,
    )

    plan = planner_service.plan_job(
        _job(),
        [],
        provider=_Provider(),
        config=planner_service.PlannerServiceConfig(mode="llm"),
        runtime=runtime,
    )

    assert plan.planner_version == "1.0.0"
    assert captured
    assert captured[0].job_id == "job-1"


def test_build_llm_prompt_uses_request_contract() -> None:
    request = planner_contracts.PlanRequest(
        job_id="job-1",
        goal="Render a DOCX",
        job_payload={"goal": "Render a DOCX"},
        semantic_capability_hints=[{"capability_id": "docx.generate"}],
    )

    prompt = planner_service.build_llm_prompt(request)

    assert "Goal: Render a DOCX" in prompt
    assert "docx.generate" in prompt


def test_validate_plan_request_uses_service_owned_capability_rules() -> None:
    request = planner_contracts.PlanRequest(
        job_id="job-1",
        goal="Check repo",
        capabilities=[
            planner_contracts.PlanRequestCapability(
                capability_id="github.repo.list",
                planner_hints={"task_intents": ["io"]},
            )
        ],
    )
    plan = models.PlanCreate(
        planner_version="1.0.0",
        tasks_summary="repo",
        dag_edges=[],
        tasks=[
            models.TaskCreate(
                name="CheckRepo",
                description="Check repo",
                instruction="Check repo existence.",
                acceptance_criteria=["Repo checked"],
                expected_output_schema_ref="schemas/test",
                intent=models.ToolIntent.generate,
                deps=[],
                tool_requests=["github.repo.list"],
                tool_inputs={"github.repo.list": {"owner": "narendersurabhi", "repo": "demo"}},
                critic_required=False,
            )
        ],
    )

    valid, reason = planner_service.validate_plan_request(
        plan,
        request,
        schema_registry_path="schemas",
    )

    assert not valid
    assert reason.startswith("capability_intent_invalid:github.repo.list:CheckRepo:")


def test_validate_plan_request_prefers_normalized_goal_segment_for_ambiguous_task_text(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        planner_service.tool_registry,
        "evaluate_tool_allowlist",
        lambda *_args, **_kwargs: planner_service.tool_registry.ToolAllowDecision(
            allowed=True,
            reason="allowed_for_test",
        ),
    )
    request = planner_contracts.PlanRequest(
        job_id="job-1",
        goal="Handle repository work",
        tools=[
            models.ToolSpec(
                name="github.repo.list",
                description="List repositories",
                input_schema={},
                output_schema={},
                tool_intent=models.ToolIntent.io,
            )
        ],
        normalized_intent_envelope={
            "goal": "Handle repository work",
            "profile": {"intent": "io", "source": "llm"},
            "graph": {
                "segments": [
                    {
                        "id": "s1",
                        "intent": "io",
                        "objective": "List repositories",
                        "suggested_capabilities": ["github.repo.list"],
                    }
                ]
            },
        },
    )
    plan = models.PlanCreate(
        planner_version="1.0.0",
        tasks_summary="repo",
        dag_edges=[],
        tasks=[
            models.TaskCreate(
                name="CheckRepo",
                description="Handle it",
                instruction="Do the work.",
                acceptance_criteria=["Repo checked"],
                expected_output_schema_ref="schemas/test",
                intent=None,
                deps=[],
                tool_requests=["github.repo.list"],
                tool_inputs={"github.repo.list": {}},
                critic_required=False,
            )
        ],
    )

    valid, reason = planner_service.validate_plan_request(
        plan,
        request,
        schema_registry_path="schemas",
    )

    assert valid, reason


def test_select_goal_intent_segment_for_task_skips_capability_match_with_mismatched_intent() -> None:
    task = models.TaskCreate(
        name="RenderPdf",
        description="Render the final PDF",
        instruction="Render the validated document spec as PDF.",
        acceptance_criteria=["PDF rendered"],
        expected_output_schema_ref="schemas/test",
        intent=models.ToolIntent.render,
        deps=[],
        tool_requests=["document.pdf.render"],
        tool_inputs={"document.pdf.render": {"path": "artifacts/report.pdf"}},
        critic_required=False,
    )

    selected = planner_service.select_goal_intent_segment_for_task(
        task=task,
        task_index=0,
        task_intent="render",
        goal_intent_segments=[
            {
                "id": "s1",
                "intent": "generate",
                "objective": "Render the final PDF",
                "suggested_capabilities": ["document.pdf.render"],
            }
        ],
        total_tasks=1,
        capabilities={
            "document.pdf.render": planner_contracts.PlanRequestCapability(
                capability_id="document.pdf.render",
                planner_hints={"task_intents": ["render"]},
            )
        },
    )

    assert selected is None


def test_build_validation_payload_projects_explicit_document_generation_fields() -> None:
    request = planner_contracts.PlanRequest(
        job_id="job-1",
        goal="Convert markdown to DOCX",
        job_context={
            "topic": "Demo",
            "audience": "General",
            "tone": "neutral",
        },
        job_payload={
            "goal": "Convert markdown to DOCX",
            "status": "queued",
            "metadata": {"llm_provider": "openai"},
            "context_json": {
                "markdown_text": "# Heading\n\nParagraph",
                "topic": "Demo",
                "tone": "neutral",
                "unrelated_blob": {"huge": "payload"},
            },
        },
    )
    task = models.TaskCreate(
        name="Generate DocumentSpec",
        description="Generate document spec",
        instruction="Generate a document spec.",
        acceptance_criteria=["Spec produced"],
        expected_output_schema_ref="schemas/document_spec",
        intent=models.ToolIntent.generate,
        deps=[],
        tool_requests=["llm_generate_document_spec"],
        tool_inputs={"llm_generate_document_spec": {}},
        critic_required=False,
    )
    tool = models.ToolSpec(
        name="llm_generate_document_spec",
        description="Generate a document spec",
        input_schema={
            "type": "object",
            "properties": {
                "instruction": {"type": "string"},
                "topic": {"type": "string"},
                "audience": {"type": "string"},
                "tone": {"type": "string"},
            },
            "required": ["instruction", "topic", "audience", "tone"],
        },
        output_schema={"type": "object"},
        tool_intent=models.ToolIntent.generate,
    )

    payload = planner_service.build_validation_payload(
        task,
        tool,
        request,
        raw_tool_inputs={},
    )

    assert payload["instruction"] == "Convert markdown to DOCX"
    assert payload["topic"] == "Demo"
    assert payload["audience"] == "General"
    assert payload["tone"] == "neutral"


def test_build_validation_payload_projects_markdown_document_generation_fields() -> None:
    request = planner_contracts.PlanRequest(
        job_id="job-1",
        goal="Convert markdown to DOCX",
        job_context={
            "markdown_text": "# Heading\n\nParagraph",
            "topic": "Demo",
            "tone": "neutral",
            "today": "2026-03-16",
            "output_dir": "documents",
        },
        job_payload={
            "goal": "Convert markdown to DOCX",
            "status": "queued",
            "context_json": {
                "markdown_text": "# Heading\n\nParagraph",
                "topic": "Demo",
                "tone": "neutral",
                "today": "2026-03-16",
                "output_dir": "documents",
            },
        },
    )
    task = models.TaskCreate(
        name="Generate DocumentSpec From Markdown",
        description="Generate document spec from markdown",
        instruction="Transform markdown into a document spec.",
        acceptance_criteria=["Spec produced"],
        expected_output_schema_ref="schemas/document_spec",
        intent=models.ToolIntent.transform,
        deps=[],
        tool_requests=["llm_generate_document_spec_from_markdown"],
        tool_inputs={"llm_generate_document_spec_from_markdown": {}},
        critic_required=False,
    )
    tool = models.ToolSpec(
        name="llm_generate_document_spec_from_markdown",
        description="Generate a document spec from markdown",
        input_schema={
            "type": "object",
            "properties": {
                "markdown_text": {"type": "string"},
                "topic": {"type": "string"},
                "tone": {"type": "string"},
                "today": {"type": "string"},
                "output_dir": {"type": "string"},
            },
            "required": ["markdown_text"],
        },
        output_schema={"type": "object"},
        tool_intent=models.ToolIntent.transform,
    )

    payload = planner_service.build_validation_payload(
        task,
        tool,
        request,
        raw_tool_inputs={},
    )

    assert payload["markdown_text"] == "# Heading\n\nParagraph"
    assert payload["topic"] == "Demo"
    assert payload["tone"] == "neutral"
    assert payload["today"] == "2026-03-16"
    assert payload["output_dir"] == "documents"


def test_postprocess_llm_plan_synthesizes_execution_bindings() -> None:
    request = planner_contracts.PlanRequest(job_id="job-1", goal="Check repo")
    plan = models.PlanCreate(
        planner_version="1.0.0",
        tasks_summary="repo",
        dag_edges=[],
        tasks=[
            models.TaskCreate(
                name="CheckRepo",
                description="Check repo",
                instruction="Check repo existence.",
                acceptance_criteria=["Repo checked"],
                expected_output_schema_ref="schemas/test",
                intent=models.ToolIntent.io,
                deps=[],
                tool_requests=["github.repo.list"],
                tool_inputs={"github.repo.list": {"owner": "narendersurabhi", "repo": "demo"}},
                critic_required=False,
            )
        ],
    )
    runtime = planner_service.PlannerServiceRuntime(
        load_capabilities=lambda: {},
        build_semantic_capability_hints=lambda job, capabilities, limit: [],
        parse_llm_plan=lambda content: models.PlanCreate.model_validate_json(content),
        ensure_llm_tool=lambda value: value,
        ensure_task_intents=lambda value, plan_request: value,
        ensure_job_inputs=lambda value, plan_request: value,
        ensure_default_value_markers=lambda value, plan_request: value,
        ensure_renderer_required_inputs=lambda value: value,
        ensure_tool_input_dependencies=lambda value: value,
        ensure_renderer_output_extensions=lambda value: value,
        ensure_execution_bindings=lambda value: value.model_copy(
            update={
                "tasks": [
                    task.model_copy(
                        update={
                            "capability_bindings": {
                                "github.repo.list": {
                                    "request_id": "github.repo.list",
                                    "capability_id": "github.repo.list",
                                    "tool_name": "github.repo.list",
                                    "adapter_type": "mcp",
                                }
                            }
                        }
                    )
                    for task in value.tasks
                ]
            }
        ),
        apply_max_depth=lambda value, max_depth: value,
    )

    result = planner_service.postprocess_llm_plan(plan, request, runtime=runtime)

    assert result.tasks[0].capability_bindings["github.repo.list"]["capability_id"] == (
        "github.repo.list"
    )
