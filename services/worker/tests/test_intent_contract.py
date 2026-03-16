from datetime import UTC, datetime

from libs.core import capability_registry, execution_contracts, intent_contract, models
from services.worker.app import main


def test_execute_task_delegates_through_execution_request_boundary(monkeypatch) -> None:
    request = execution_contracts.TaskExecutionRequest(
        task_id="task-1",
        source_payload={"task_id": "task-1"},
    )
    expected = models.TaskResult(
        task_id="task-1",
        status=models.TaskStatus.completed,
        outputs={},
        artifacts=[],
        tool_calls=[],
        started_at=datetime.now(UTC),
        finished_at=datetime.now(UTC),
    )
    seen: list[execution_contracts.TaskExecutionRequest] = []

    def fake_build(payload: dict, *, default_max_attempts: int) -> execution_contracts.TaskExecutionRequest:
        assert payload == {"task_id": "task-1"}
        assert default_max_attempts == main.WORKER_DEFAULT_MAX_ATTEMPTS
        return request

    def fake_execute(
        built_request: execution_contracts.TaskExecutionRequest,
    ) -> models.TaskResult:
        seen.append(built_request)
        return expected

    monkeypatch.setattr(main.execution_contracts, "build_task_execution_request", fake_build)
    monkeypatch.setattr(main, "execute_task_request", fake_execute)

    result = main.execute_task({"task_id": "task-1"})

    assert result is expected
    assert seen == [request]


def test_execute_task_request_delegates_to_execution_service(monkeypatch) -> None:
    request = execution_contracts.TaskExecutionRequest(
        task_id="task-1",
        source_payload={"task_id": "task-1"},
    )
    expected = models.TaskResult(
        task_id="task-1",
        status=models.TaskStatus.completed,
        outputs={},
        artifacts=[],
        tool_calls=[],
        started_at=datetime.now(UTC),
        finished_at=datetime.now(UTC),
    )

    monkeypatch.setattr(
        main.tool_runtime_adapter,
        "build_worker_tool_runtime",
        lambda **kwargs: object(),
    )
    monkeypatch.setattr(
        main.capability_runtime_adapter,
        "build_worker_capability_runtime",
        lambda **kwargs: object(),
    )

    seen: list[tuple[execution_contracts.TaskExecutionRequest, object, object]] = []

    def fake_execute_task_request(
        built_request: execution_contracts.TaskExecutionRequest,
        *,
        context,
        callbacks,
    ) -> models.TaskResult:
        seen.append((built_request, context.tool_runtime, context.capability_runtime))
        assert callbacks.task_intent_inference is main._task_intent_inference_for_request
        assert callbacks.intent_segment is main._intent_segment_for_request
        return expected

    monkeypatch.setattr(main.execution_service, "execute_task_request", fake_execute_task_request)

    result = main.execute_task_request(request)

    assert result is expected
    assert len(seen) == 1
    assert seen[0][0] is request
    assert seen[0][1] is not None
    assert seen[0][2] is not None


def test_infer_task_intent_uses_payload_hint() -> None:
    payload = {
        "intent": "render",
        "description": "Generate something",
        "instruction": "Generate",
        "acceptance_criteria": ["done"],
    }
    assert main._infer_task_intent(payload) == "render"


def test_infer_task_intent_inference_exposes_source_and_confidence() -> None:
    payload = {
        "description": "Step one",
        "instruction": "Handle task",
        "acceptance_criteria": ["done"],
        "goal": "Validate this output against schema",
    }
    inference = main._infer_task_intent_inference(payload)
    assert inference.intent == "validate"
    assert inference.source == "goal_text"
    assert inference.confidence > 0


def test_intent_mismatch_rejects_generate_tool_for_io_task() -> None:
    mismatch = main._intent_mismatch("io", models.ToolIntent.generate, "llm_generate")
    assert mismatch == "tool_intent_mismatch:llm_generate:generate:io"


def test_capability_intent_mismatch_rejects_disallowed_task_intent() -> None:
    spec = capability_registry.CapabilitySpec(
        capability_id="github.repo.list",
        description="List repositories",
        risk_tier="read_only",
        idempotency="read",
        planner_hints={"task_intents": ["io"]},
    )
    mismatch = main._capability_intent_mismatch("generate", spec)
    assert mismatch == "task_intent_mismatch:github.repo.list:generate:allowed=io"


def test_capability_intent_mismatch_allows_configured_task_intent() -> None:
    spec = capability_registry.CapabilitySpec(
        capability_id="document.spec.generate",
        description="Generate document spec",
        risk_tier="read_only",
        idempotency="read",
        planner_hints={"task_intents": ["generate", "transform"]},
    )
    mismatch = main._capability_intent_mismatch("generate", spec)
    assert mismatch is None


def test_intent_segment_from_payload_prefers_direct_segment() -> None:
    payload = {
        "intent_segment": {
            "id": "s1",
            "intent": "render",
            "objective": "Render final PDF",
            "slots": {
                "entity": "report",
                "artifact_type": "document",
                "output_format": "pdf",
                "risk_level": "bounded_write",
                "must_have_inputs": ["document_spec", "path"],
            },
        }
    }
    segment = main._intent_segment_from_payload(payload)
    assert segment is not None
    assert segment["intent"] == "render"


def test_intent_segment_contract_allows_renderer_without_explicit_path() -> None:
    segment = {
        "id": "s1",
        "intent": "render",
        "objective": "Render final PDF",
        "required_inputs": ["document_spec", "path"],
        "slots": {
            "entity": "report",
            "artifact_type": "document",
            "output_format": "pdf",
            "risk_level": "bounded_write",
            "must_have_inputs": ["document_spec", "path"],
        },
    }
    mismatch = intent_contract.validate_intent_segment_contract(
        segment=segment,
        task_intent="render",
        tool_name="document.pdf.generate",
        payload={"document_spec": {"blocks": []}},
        capability_id="document.pdf.generate",
        capability_risk_tier="bounded_write",
    )
    assert mismatch is None


def test_tool_payload_builds_github_repo_query_from_context_fields() -> None:
    payload = main._tool_payload(
        "github.repo.list",
        "Verify repository exists",
        {
            "job_context": {
                "repo_owner": "narendersurabhi",
                "repo_name": "scientific-agent-lab",
            }
        },
        {"tool_inputs": {"github.repo.list": {}}},
        {"github.repo.list": {}},
    )
    assert payload["query"] == "repo:scientific-agent-lab owner:narendersurabhi"


def test_validate_expected_output_rejects_invalid_render_validation_report() -> None:
    error = main._validate_expected_output(
        {
            "tool_requests": ["docx_generate_from_spec"],
            "tool_inputs": {
                "docx_generate_from_spec": {
                    "validation_report": {
                        "valid": False,
                        "errors": [
                            {
                                "path": "/blocks/0/text",
                                "message": "text/paragraph requires text: string",
                            }
                        ],
                    }
                }
            },
        },
        {},
    )

    assert (
        error
        == "render_validation_failed:docx_generate_from_spec:/blocks/0/text: text/paragraph requires text: string"
    )


def test_validate_expected_output_rejects_render_errors_without_validation_report() -> None:
    error = main._validate_expected_output(
        {
            "tool_requests": ["document.pdf.generate"],
            "tool_inputs": {
                "document.pdf.generate": {
                    "errors": [
                        {
                            "path": "/blocks/1/items",
                            "message": "items must be an array",
                        }
                    ]
                }
            },
        },
        {},
    )

    assert (
        error
        == "render_validation_failed:document.pdf.generate:/blocks/1/items: items must be an array"
    )
