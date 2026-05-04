from libs.core import capability_registry, execution_contracts, models, run_specs


def _document_capability_registry() -> capability_registry.CapabilityRegistry:
    return capability_registry.CapabilityRegistry(
        capabilities={
            "document.spec.generate": capability_registry.CapabilitySpec(
                capability_id="document.spec.generate",
                description="Generate a document spec.",
                enabled=True,
                risk_tier="read_only",
                idempotency="read",
                group="documents",
                subgroup="generation",
                adapters=(
                    capability_registry.CapabilityAdapterSpec(
                        type="tool",
                        server_id="local_worker",
                        tool_name="llm_generate_document_spec",
                    ),
                ),
            )
        }
    )


def test_plan_to_run_spec_roundtrips_execution_gate() -> None:
    plan = models.PlanCreate(
        planner_version="ui_chaining_composer_v2",
        tasks_summary="Round-trip studio plan",
        dag_edges=[["LoadData", "TransformData"]],
        tasks=[
            models.TaskCreate(
                name="LoadData",
                description="Load source data",
                instruction="Use capability filesystem.workspace.list.",
                acceptance_criteria=["Completed capability filesystem.workspace.list"],
                expected_output_schema_ref="schemas/json_object",
                intent=models.ToolIntent.io,
                deps=[],
                tool_requests=["filesystem.workspace.list"],
                tool_inputs={"filesystem.workspace.list": {"path": "."}},
                critic_required=False,
            ),
            models.TaskCreate(
                name="TransformData",
                description="Transform the source data",
                instruction="Use capability json_transform.",
                acceptance_criteria=["Completed capability json_transform"],
                expected_output_schema_ref="schemas/json_object",
                intent=models.ToolIntent.transform,
                deps=["LoadData"],
                tool_requests=["json_transform"],
                tool_inputs=execution_contracts.embed_execution_gate(
                    {
                        "json_transform": {
                            "source": {
                                "$from": [
                                    "dependencies_by_name",
                                    "LoadData",
                                    "filesystem.workspace.list",
                                    "items",
                                ]
                            }
                        }
                    },
                    {"expression": "context.approved == true"},
                    request_ids=["json_transform"],
                ),
                critic_required=False,
            ),
        ],
    )

    run_spec = run_specs.plan_to_run_spec(plan, kind=models.RunKind.studio)

    assert run_spec.kind == models.RunKind.studio
    assert [step.name for step in run_spec.steps] == ["LoadData", "TransformData"]
    assert run_spec.steps[1].depends_on == [run_spec.steps[0].step_id]
    assert run_spec.steps[1].execution_gate == {"expression": "context.approved == true"}
    assert run_spec.steps[1].input_bindings == {
        "source": {
            "$from": [
                "dependencies_by_name",
                "LoadData",
                "filesystem.workspace.list",
                "items",
            ]
        }
    }

    round_tripped = run_specs.run_spec_to_plan(run_spec)

    assert round_tripped.model_dump(mode="json") == plan.model_dump(mode="json")


def test_plan_to_run_spec_accepts_capability_requests_without_tool_requests(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        run_specs.capability_registry,
        "load_capability_registry",
        lambda: _document_capability_registry(),
    )
    plan = models.PlanCreate(
        planner_version="planner_v2",
        tasks_summary="Capability-first planner plan",
        dag_edges=[],
        tasks=[
            models.TaskCreate(
                name="GenerateDocumentSpec",
                description="Generate a document spec",
                instruction="Use capability document.spec.generate.",
                acceptance_criteria=["Spec produced"],
                expected_output_schema_ref="schemas/document_spec",
                intent=models.ToolIntent.generate,
                deps=[],
                capability_requests=["document.spec.generate"],
                tool_requests=[],
                tool_inputs={"document.spec.generate": {"instruction": "Generate a document"}},
                critic_required=False,
            )
        ],
    )

    run_spec = run_specs.plan_to_run_spec(plan, kind=models.RunKind.planner)

    assert len(run_spec.steps) == 1
    assert run_spec.steps[0].capability_request.request_id == "document.spec.generate"
    assert run_spec.steps[0].capability_request.capability_id == "document.spec.generate"
    assert run_spec.steps[0].capability_request.execution_request_id == (
        "llm_generate_document_spec"
    )
    assert run_spec.steps[0].input_bindings == {"instruction": "Generate a document"}
    assert run_spec.steps[0].routing_hints["tool_name"] == "llm_generate_document_spec"
    round_tripped = run_specs.run_spec_to_plan(run_spec)
    assert round_tripped.tasks[0].capability_requests == ["document.spec.generate"]
    assert round_tripped.tasks[0].tool_requests == []
    assert round_tripped.tasks[0].tool_inputs == {
        "document.spec.generate": {"instruction": "Generate a document"}
    }


def test_parse_legacy_run_spec_promotes_compiled_request_id_to_execution_request_id() -> None:
    run_spec = run_specs.parse_run_spec(
        {
            "kind": "planner",
            "steps": [
                {
                    "step_id": "generate",
                    "name": "GenerateDocumentSpec",
                    "description": "Generate a document spec",
                    "instruction": "Generate a document spec",
                    "capability_request": {
                        "request_id": "llm_generate_document_spec",
                        "capability_id": "document.spec.generate",
                    },
                }
            ],
        }
    )

    assert run_spec is not None
    assert run_spec.steps[0].capability_request.request_id == "document.spec.generate"
    assert run_spec.steps[0].capability_request.execution_request_id == (
        "llm_generate_document_spec"
    )
