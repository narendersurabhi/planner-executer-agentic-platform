from libs.core import capability_registry, intent_contract, models
from services.worker.app import main


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


def test_intent_segment_contract_detects_missing_required_payload_input() -> None:
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
    assert mismatch == "must_have_inputs_missing:path"
