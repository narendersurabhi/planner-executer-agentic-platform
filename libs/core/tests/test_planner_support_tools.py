from __future__ import annotations

import json

from libs.core import capability_registry, models, planner_support_tools


def _capabilities() -> dict[str, capability_registry.CapabilitySpec]:
    return {
        "document.spec.generate": capability_registry.CapabilitySpec(
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
                ),
            ),
        ),
        "document.docx.render": capability_registry.CapabilitySpec(
            capability_id="document.docx.render",
            description="Render a docx artifact",
            group="documents",
            subgroup="rendering",
            risk_tier="bounded_write",
            idempotency="safe_write",
            adapters=(
                capability_registry.CapabilityAdapterSpec(
                    type="tool",
                    server_id="local_worker",
                    tool_name="docx_render_from_spec",
                ),
            ),
        ),
    }


def test_build_planner_support_tool_specs_exposes_metadata_only_surface() -> None:
    names = [tool.name for tool in planner_support_tools.build_planner_support_tool_specs()]

    assert names == [
        "search_capabilities",
        "get_capability_contract",
        "get_schema",
        "get_workflow_hints",
        "get_memory_hints",
        "finalize_run_spec",
    ]


def test_search_capabilities_support_returns_ranked_matches() -> None:
    result = planner_support_tools.search_capabilities_support(
        query="render docx document",
        capabilities=_capabilities(),
        intent_hint="render",
        limit=2,
    )

    assert result["matches"][0]["id"] == "document.docx.render"


def test_finalize_run_spec_support_compiles_capability_first_plan(monkeypatch) -> None:
    monkeypatch.setattr(
        planner_support_tools.capability_registry,
        "load_capability_registry",
        lambda: capability_registry.CapabilityRegistry(capabilities=_capabilities()),
    )
    plan = models.PlanCreate(
        planner_version="planner_vNext",
        tasks_summary="Generate a document spec",
        dag_edges=[],
        tasks=[
            models.TaskCreate(
                name="GenerateDocumentSpec",
                description="Generate a doc spec",
                instruction="Generate a doc spec",
                acceptance_criteria=["Spec produced"],
                expected_output_schema_ref="schemas/document_spec",
                intent=models.ToolIntent.generate,
                deps=[],
                capability_requests=["document.spec.generate"],
                tool_requests=[],
                tool_inputs={"document.spec.generate": {"instruction": "Generate a doc spec"}},
                critic_required=False,
            )
        ],
    )

    result = planner_support_tools.finalize_run_spec_support(plan=plan)

    assert result["ok"] is True
    assert result["run_spec"]["steps"][0]["capability_request"]["request_id"] == (
        "document.spec.generate"
    )
    assert result["run_spec"]["steps"][0]["capability_request"]["execution_request_id"] == (
        "llm_generate_document_spec"
    )


def test_get_schema_support_summarizes_required_fields(tmp_path) -> None:
    schema_path = tmp_path / "example_schema.json"
    schema_path.write_text(
        json.dumps(
            {
                "type": "object",
                "properties": {"topic": {"type": "string"}, "audience": {"type": "string"}},
                "required": ["topic"],
            }
        ),
        encoding="utf-8",
    )

    result = planner_support_tools.get_schema_support(
        schema_ref="example_schema",
        schema_registry_path=str(tmp_path),
    )

    assert result["summary"]["required_fields"] == ["topic"]
    assert result["summary"]["property_count"] == 2
