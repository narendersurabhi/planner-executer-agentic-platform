from libs.core import intent_contract


def test_infer_task_intent_for_task_uses_goal_when_task_text_generic() -> None:
    inferred = intent_contract.infer_task_intent_for_task(
        explicit_intent=None,
        description="Step one",
        instruction="Handle this task.",
        acceptance_criteria=["Done"],
        goal_text="Validate the generated document against schema rules.",
    )
    assert inferred == "validate"


def test_infer_task_intent_for_payload_uses_nested_job_goal() -> None:
    inferred = intent_contract.infer_task_intent_for_payload(
        {
            "description": "Step one",
            "instruction": "Handle this task.",
            "acceptance_criteria": ["Done"],
            "job": {"goal": "Render a PDF from the approved document spec."},
        }
    )
    assert inferred == "render"


def test_infer_task_intent_explicit_value_wins_over_goal() -> None:
    inferred = intent_contract.infer_task_intent_for_task(
        explicit_intent="io",
        description="Generate report",
        instruction="Generate report",
        acceptance_criteria=["Done"],
        goal_text="Generate a report document.",
    )
    assert inferred == "io"


def test_infer_task_intent_with_metadata_reports_source_and_confidence() -> None:
    inference = intent_contract.infer_task_intent_for_task_with_metadata(
        explicit_intent=None,
        description="Step one",
        instruction="Handle this task.",
        acceptance_criteria=["Done"],
        goal_text="Render the final output as PDF.",
    )
    assert inference.intent == "render"
    assert inference.source == "goal_text"
    assert inference.confidence > 0


def test_infer_task_intent_with_metadata_avoids_substring_false_positive() -> None:
    inference = intent_contract.infer_task_intent_from_goal_with_metadata(
        "Polish this document spec for release readiness"
    )
    assert inference.intent == "generate"
    assert inference.source == "default"
    assert inference.confidence == 0.4


def test_decompose_goal_intent_produces_ordered_segments() -> None:
    graph = intent_contract.decompose_goal_intent(
        "Fetch GitHub repositories, then summarize findings, then render a PDF report."
    )
    segments = graph["segments"]
    assert len(segments) >= 3
    assert segments[0]["intent"] == "io"
    assert segments[1]["depends_on"] == ["s1"]
    assert segments[-1]["intent"] == "render"
    assert "document.pdf.render" in segments[-1]["suggested_capabilities"]
    assert isinstance(segments[-1]["slots"], dict)
    assert segments[-1]["slots"]["output_format"] == "pdf"
    assert "must_have_inputs" in segments[-1]["slots"]
    assert graph["summary"]["schema_version"] == "intent_v2"


def test_decompose_goal_intent_handles_empty_goal() -> None:
    graph = intent_contract.decompose_goal_intent("")
    assert graph["segments"] == []
    assert graph["summary"]["segment_count"] == 0
    assert graph["summary"]["schema_version"] == "intent_v2"


def test_decompose_goal_intent_render_segment_suggests_renderer_only() -> None:
    graph = intent_contract.decompose_goal_intent("Render a PDF status report.")
    segments = graph["segments"]
    assert len(segments) == 1
    assert segments[0]["intent"] == "render"
    assert segments[0]["suggested_capabilities"] == ["document.pdf.render"]


def test_decompose_goal_intent_splits_comma_separated_action_clauses() -> None:
    graph = intent_contract.decompose_goal_intent(
        "Generate an implementation runbook, validate the document spec, then render a DOCX."
    )
    segments = graph["segments"]
    assert [segment["intent"] for segment in segments[:3]] == ["generate", "validate", "render"]
    assert "document.spec.generate" in segments[0]["suggested_capabilities"]
    assert segments[1]["suggested_capabilities"] == ["document.spec.validate"]
    assert segments[2]["suggested_capabilities"] == ["document.docx.render"]


def test_decompose_goal_intent_prefers_openapi_iterative_capabilities() -> None:
    graph = intent_contract.decompose_goal_intent(
        "Generate an OpenAPI specification for a payment API and iteratively improve it."
    )
    segments = graph["segments"]
    assert len(segments) >= 2
    assert segments[0]["intent"] == "generate"
    assert segments[0]["suggested_capabilities"][0] == "openapi.spec.generate_iterative"
    assert segments[1]["intent"] == "transform"
    assert segments[1]["suggested_capabilities"][0] == "openapi.spec.improve_iterative"


def test_decompose_goal_intent_prefers_github_publish_capabilities() -> None:
    graph = intent_contract.decompose_goal_intent(
        "Generate code changes, push files to GitHub, and create a pull request."
    )
    segments = graph["segments"]
    assert [segment["intent"] for segment in segments[:3]] == ["generate", "io", "io"]
    assert segments[0]["suggested_capabilities"][0] == "codegen.generate"
    assert segments[1]["suggested_capabilities"][0] == "github.files.push"
    assert segments[2]["suggested_capabilities"][0] == "github.pull_request.create"


def test_decompose_goal_intent_prefers_workspace_delete_capability_for_destructive_goal() -> None:
    graph = intent_contract.decompose_goal_intent(
        "Delete the production deployment manifest from the workspace."
    )
    segments = graph["segments"]
    assert len(segments) == 1
    assert segments[0]["suggested_capabilities"][0] == "filesystem.workspace.delete"
    assert "instruction" not in segments[0]["required_inputs"]


def test_derive_segment_missing_inputs_accepts_goal_implied_instruction() -> None:
    missing = intent_contract.derive_segment_missing_inputs(
        goal="Generate a quarterly planning memo draft for leadership.",
        segment={
            "intent": "generate",
            "objective": "Generate a quarterly planning memo draft for leadership",
            "required_inputs": ["instruction"],
            "slots": {"must_have_inputs": ["instruction"]},
        },
        slot_values={"intent_action": "generate"},
        candidate_required_inputs=[],
        low_confidence=False,
    )
    assert missing == []


def test_derive_segment_missing_inputs_accepts_explicit_goal_path() -> None:
    missing = intent_contract.derive_segment_missing_inputs(
        goal="Render the approved document spec to /shared/artifacts/q1-plan.pdf.",
        segment={
            "intent": "render",
            "objective": "Render the approved document spec to /shared/artifacts/q1-plan.pdf",
            "required_inputs": ["path_or_format", "output_format=pdf"],
            "suggested_capabilities": ["document.pdf.render"],
            "slots": {"must_have_inputs": ["path", "output_format"]},
        },
        slot_values={"intent_action": "render", "output_format": "pdf"},
        candidate_required_inputs=[],
        low_confidence=False,
    )
    assert missing == []


def test_validate_intent_segment_contract_requires_renderer_explicit_path() -> None:
    segment = {
        "intent": "render",
        "objective": "Render final PDF",
        "required_inputs": ["document_spec"],
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
        tool_name="document.pdf.render",
        payload={"document_spec": {"blocks": []}},
        capability_id="document.pdf.render",
        capability_risk_tier="bounded_write",
    )
    assert mismatch == "must_have_inputs_missing:path"


def test_validate_intent_segment_contract_accepts_path_for_path_requirement() -> None:
    segment = {
        "intent": "render",
        "objective": "Render final DOCX",
        "required_inputs": ["document_spec"],
        "slots": {
            "entity": "report",
            "artifact_type": "document",
            "output_format": "docx",
            "risk_level": "bounded_write",
            "must_have_inputs": ["document_spec", "path"],
        },
    }
    mismatch = intent_contract.validate_intent_segment_contract(
        segment=segment,
        task_intent="render",
        tool_name="docx_render_from_spec",
        payload={"document_spec": {"blocks": []}, "path": "artifacts/output.docx"},
        capability_id="document.docx.render",
        capability_risk_tier="bounded_write",
    )
    assert mismatch is None


def test_validate_intent_segment_contract_accepts_path_for_query_requirement() -> None:
    segment = {
        "intent": "io",
        "objective": "List a workspace subdirectory",
        "required_inputs": ["source_or_query"],
        "slots": {
            "entity": "workspace",
            "artifact_type": "filesystem",
            "risk_level": "read_only",
            "must_have_inputs": ["query"],
        },
    }
    mismatch = intent_contract.validate_intent_segment_contract(
        segment=segment,
        task_intent="io",
        tool_name="filesystem.workspace.list",
        payload={"path": "reports"},
        capability_id="filesystem.workspace.list",
        capability_risk_tier="read_only",
    )
    assert mismatch is None


def test_validate_intent_segment_contract_allows_filesystem_list_without_query() -> None:
    segment = {
        "intent": "io",
        "objective": "List workspace files",
        "required_inputs": ["source_or_query"],
        "slots": {
            "entity": "workspace",
            "artifact_type": "filesystem",
            "risk_level": "read_only",
            "must_have_inputs": ["query"],
        },
    }
    mismatch = intent_contract.validate_intent_segment_contract(
        segment=segment,
        task_intent="io",
        tool_name="filesystem.workspace.list",
        payload={"recursive": None},
        capability_id="filesystem.workspace.list",
        capability_risk_tier="read_only",
    )
    assert mismatch is None


def test_validate_intent_segment_contract_checks_output_format_hint() -> None:
    segment = {
        "intent": "render",
        "objective": "Render final PDF",
        "slots": {
            "entity": "report",
            "artifact_type": "document",
            "output_format": "pdf",
            "risk_level": "bounded_write",
            "must_have_inputs": ["path"],
        },
    }
    mismatch = intent_contract.validate_intent_segment_contract(
        segment=segment,
        task_intent="render",
        tool_name="document.docx.render",
        payload={"path": "artifacts/output.docx"},
        capability_id="document.docx.render",
        capability_risk_tier="bounded_write",
    )
    assert mismatch == "output_format_mismatch:document.docx.render:expected=pdf:got=docx"


def test_validate_intent_segment_contract_treats_renderer_hint_as_output_format_input() -> None:
    segment = {
        "intent": "render",
        "objective": "Render final DOCX",
        "required_inputs": ["input_data", "path_or_format"],
        "slots": {
            "entity": "report",
            "artifact_type": "document",
            "output_format": "docx",
            "risk_level": "bounded_write",
            "must_have_inputs": ["document_spec", "output_format"],
        },
    }
    mismatch = intent_contract.validate_intent_segment_contract(
        segment=segment,
        task_intent="render",
        tool_name="docx_render_from_spec",
        payload={"document_spec": {"blocks": []}},
        capability_id="document.docx.render",
        capability_risk_tier="bounded_write",
    )
    assert mismatch is None


def test_validate_intent_segment_contract_treats_render_json_as_spec_input_format() -> None:
    segment = {
        "intent": "render",
        "objective": "Render DocumentSpec JSON into a document",
        "required_inputs": ["document_spec_json", "path"],
        "suggested_capabilities": ["document.docx.render"],
        "slots": {
            "entity": "artifact",
            "artifact_type": "document",
            "output_format": "json",
            "risk_level": "bounded_write",
            "must_have_inputs": ["document_spec", "path"],
        },
    }
    mismatch = intent_contract.validate_intent_segment_contract(
        segment=segment,
        task_intent="render",
        tool_name="document.docx.render",
        payload={"document_spec": {"blocks": []}, "path": "Narender.docx"},
        capability_id="document.docx.render",
        capability_risk_tier="bounded_write",
    )
    assert mismatch is None


def test_normalize_intent_segment_slots_uses_render_capability_for_json_spec_input() -> None:
    slots = intent_contract.normalize_intent_segment_slots(
        raw_slots={
            "entity": "artifact",
            "artifact_type": "document",
            "output_format": "json",
            "risk_level": "bounded_write",
            "must_have_inputs": ["document_spec", "path"],
        },
        fallback_slots=None,
        intent="render",
        objective="Render DocumentSpec JSON into a document",
        required_inputs=["document_spec_json", "path"],
        suggested_capabilities=["document.docx.render"],
    )

    assert slots["output_format"] == "docx"


def test_validate_intent_segment_contract_normalizes_render_risk_to_bounded_write() -> None:
    segment = {
        "intent": "render",
        "objective": "Render final DOCX",
        "required_inputs": ["document_spec", "path"],
        "slots": {
            "entity": "report",
            "artifact_type": "document",
            "output_format": "docx",
            "risk_level": "read_only",
            "must_have_inputs": ["document_spec", "path"],
        },
    }
    mismatch = intent_contract.validate_intent_segment_contract(
        segment=segment,
        task_intent="render",
        tool_name="document.docx.render",
        payload={"document_spec": {"blocks": []}, "path": "artifacts/output.docx"},
        capability_id="document.docx.render",
        capability_risk_tier="bounded_write",
    )
    assert mismatch is None


def test_validate_intent_segment_contract_allows_io_segment_for_transform_task() -> None:
    segment = {
        "intent": "io",
        "objective": "Derive output filename",
        "slots": {
            "entity": "document",
            "artifact_type": "document",
            "output_format": "docx",
            "risk_level": "read_only",
            "must_have_inputs": ["path_or_format"],
        },
    }
    mismatch = intent_contract.validate_intent_segment_contract(
        segment=segment,
        task_intent="transform",
        tool_name="derive_output_filename",
        payload={
            "path": "documents/engineer.docx",
            "output_extension": "docx",
            "target_role_name": "Engineer",
        },
        capability_id="derive_output_filename",
        capability_risk_tier="read_only",
    )
    assert mismatch is None


def test_validate_intent_segment_contract_allows_io_segment_for_validate_task() -> None:
    segment = {
        "intent": "io",
        "objective": "Validate document spec",
        "slots": {
            "entity": "document",
            "artifact_type": "validation_report",
            "output_format": None,
            "risk_level": "read_only",
            "must_have_inputs": ["document_spec"],
        },
    }
    mismatch = intent_contract.validate_intent_segment_contract(
        segment=segment,
        task_intent="validate",
        tool_name="document_spec_validate",
        payload={"document_spec": {"blocks": []}},
        capability_id="document.spec.validate",
        capability_risk_tier="read_only",
    )
    assert mismatch is None


def test_validate_intent_segment_contract_allows_placeholder_document_spec_mapping() -> None:
    segment = {
        "intent": "validate",
        "objective": "Validate document spec",
        "slots": {
            "entity": "artifact",
            "artifact_type": "document",
            "output_format": "docx",
            "risk_level": "read_only",
            "must_have_inputs": ["document_spec"],
        },
    }
    mismatch = intent_contract.validate_intent_segment_contract(
        segment=segment,
        task_intent="validate",
        tool_name="document_spec_validate",
        payload={"document_spec": {}},
        capability_id="document.spec.validate",
        capability_risk_tier="read_only",
    )
    assert mismatch is None


def test_validate_intent_segment_contract_treats_document_spec_as_input_data() -> None:
    segment = {
        "intent": "validate",
        "objective": "Validate document spec",
        "required_inputs": ["input_data", "acceptance_criteria_or_schema"],
        "suggested_capabilities": ["document.spec.validate"],
        "slots": {
            "entity": "document_spec",
            "artifact_type": "validation_report",
            "output_format": None,
            "risk_level": "read_only",
            "must_have_inputs": ["input_data"],
        },
    }
    mismatch = intent_contract.validate_intent_segment_contract(
        segment=segment,
        task_intent="validate",
        tool_name="document_spec_validate",
        payload={"document_spec": {}},
        capability_id="document.spec.validate",
        capability_risk_tier="read_only",
    )
    assert mismatch is None


def test_validate_intent_segment_contract_allows_derive_output_filename_without_document_spec() -> None:
    segment = {
        "intent": "transform",
        "objective": "Derive output filename",
        "slots": {
            "entity": "artifact",
            "artifact_type": "document",
            "output_format": "docx",
            "risk_level": "read_only",
            "must_have_inputs": ["document_spec", "output_extension"],
        },
    }
    mismatch = intent_contract.validate_intent_segment_contract(
        segment=segment,
        task_intent="transform",
        tool_name="derive_output_filename",
        payload={
            "target_role_name": "Active Inference Engineer",
            "output_dir": "documents",
            "output_extension": "docx",
        },
        capability_id="derive_output_filename",
        capability_risk_tier="read_only",
    )
    assert mismatch is None


def test_normalize_intent_segment_slots_removes_filename_requirements_for_document_spec_generation() -> None:
    slots = intent_contract.normalize_intent_segment_slots(
        raw_slots={
            "entity": "document",
            "artifact_type": "document_spec",
            "risk_level": "read_only",
            "must_have_inputs": [
                "filename",
                "title",
                "output_path",
                "path",
                "output_format",
                "format",
                "compactness",
            ],
        },
        fallback_slots=None,
        intent="generate",
        objective="Generate document spec",
        required_inputs=[],
        suggested_capabilities=["document.spec.generate"],
    )
    assert slots["must_have_inputs"] == ["title"]


def test_normalize_intent_segment_slots_removes_output_format_fallback_for_document_spec_generation() -> None:
    slots = intent_contract.normalize_intent_segment_slots(
        raw_slots={
            "entity": "document",
            "artifact_type": "document_spec",
            "risk_level": "read_only",
        },
        fallback_slots=None,
        intent="generate",
        objective="Generate a DOCX-ready document spec",
        required_inputs=["instruction", "output_format=docx"],
        suggested_capabilities=["document.spec.generate"],
    )
    assert slots["must_have_inputs"] == ["instruction"]


def test_normalize_intent_segment_slots_removes_format_and_compactness_fallbacks_for_document_spec_generation() -> None:
    slots = intent_contract.normalize_intent_segment_slots(
        raw_slots={
            "entity": "document",
            "artifact_type": "document_spec",
            "risk_level": "read_only",
        },
        fallback_slots=None,
        intent="generate",
        objective="Generate a compact DOCX document spec",
        required_inputs=["instruction", "format=docx", "compactness=compact"],
        suggested_capabilities=["document.spec.generate"],
    )
    assert slots["must_have_inputs"] == ["instruction"]


def test_normalize_intent_segment_slots_removes_length_limit_for_document_spec_generation() -> None:
    slots = intent_contract.normalize_intent_segment_slots(
        raw_slots={
            "entity": "document",
            "artifact_type": "document_spec",
            "risk_level": "read_only",
            "must_have_inputs": ["length_limit", "title"],
        },
        fallback_slots=None,
        intent="generate",
        objective="GenerateDocumentSpec",
        required_inputs=["instruction"],
        suggested_capabilities=["llm_generate_document_spec"],
    )
    assert slots["must_have_inputs"] == ["title"]


def test_normalize_intent_segment_slots_detects_collapsed_documentspec_objective() -> None:
    slots = intent_contract.normalize_intent_segment_slots(
        raw_slots={"risk_level": "read_only"},
        fallback_slots=None,
        intent="generate",
        objective="GenerateDocumentSpec",
        required_inputs=["instruction", "format=docx"],
        suggested_capabilities=["document.spec.generate"],
    )
    assert slots["entity"] == "document_spec"
    assert slots["artifact_type"] == "document_spec"
    assert slots["must_have_inputs"] == ["instruction"]


def test_normalize_intent_segment_slots_overrides_wrong_explicit_artifact_for_documentspec_generation() -> None:
    slots = intent_contract.normalize_intent_segment_slots(
        raw_slots={
            "entity": "document",
            "artifact_type": "document",
            "risk_level": "read_only",
            "must_have_inputs": ["filename", "format", "title"],
        },
        fallback_slots=None,
        intent="generate",
        objective="GenerateDocumentSpec",
        required_inputs=["instruction"],
        suggested_capabilities=["llm_generate_document_spec"],
    )
    assert slots["entity"] == "document_spec"
    assert slots["artifact_type"] == "document_spec"
    assert slots["must_have_inputs"] == ["title"]


def test_normalize_intent_segment_slots_removes_authoring_inputs_for_clarification_step() -> None:
    slots = intent_contract.normalize_intent_segment_slots(
        raw_slots={
            "risk_level": "read_only",
            "must_have_inputs": ["audience", "tone", "topic", "output_format"],
        },
        fallback_slots=None,
        intent="generate",
        objective="Clarify output format",
        required_inputs=["audience", "tone", "topic", "output_format"],
        suggested_capabilities=["llm_generate"],
    )
    assert slots["must_have_inputs"] == []


def test_validate_intent_segment_contract_accepts_title_alias_from_explicit_payload() -> None:
    segment = {
        "intent": "generate",
        "objective": "Generate document spec",
        "slots": {
            "entity": "document",
            "artifact_type": "document_spec",
            "output_format": None,
            "risk_level": "read_only",
            "must_have_inputs": ["title"],
        },
    }
    mismatch = intent_contract.validate_intent_segment_contract(
        segment=segment,
        task_intent="generate",
        tool_name="llm_generate_document_spec",
        payload={"target_role_name": "Platform Engineer"},
        capability_id="document.spec.generate",
        capability_risk_tier="read_only",
    )
    assert mismatch is None


def test_validate_intent_segment_contract_ignores_wrong_explicit_artifact_for_documentspec_generation() -> None:
    segment = {
        "intent": "generate",
        "objective": "GenerateDocumentSpec",
        "required_inputs": ["instruction"],
        "suggested_capabilities": ["llm_generate_document_spec"],
        "slots": {
            "entity": "document",
            "artifact_type": "document",
            "risk_level": "read_only",
            "must_have_inputs": ["filename", "format"],
        },
    }
    mismatch = intent_contract.validate_intent_segment_contract(
        segment=segment,
        task_intent="generate",
        tool_name="llm_generate_document_spec",
        payload={"instruction": "Generate a resume document spec."},
        capability_id="document.spec.generate",
        capability_risk_tier="read_only",
    )
    assert mismatch is None


def test_validate_intent_segment_contract_allows_clarify_output_format_without_authoring_inputs() -> None:
    segment = {
        "intent": "generate",
        "objective": "Clarify output format",
        "required_inputs": ["audience", "tone", "topic", "output_format"],
        "suggested_capabilities": ["llm_generate"],
        "slots": {
            "risk_level": "read_only",
            "must_have_inputs": ["audience", "tone", "topic", "output_format"],
        },
    }
    mismatch = intent_contract.validate_intent_segment_contract(
        segment=segment,
        task_intent="generate",
        tool_name="llm_generate",
        payload={"instruction": "Ask the user which output format they want."},
        capability_id="llm.text.generate",
        capability_risk_tier="read_only",
    )
    assert mismatch is None


def test_validate_intent_segment_contract_does_not_accept_generate_title_from_job_context() -> None:
    segment = {
        "intent": "generate",
        "objective": "Generate document spec",
        "slots": {
            "entity": "document",
            "artifact_type": "document_spec",
            "output_format": None,
            "risk_level": "read_only",
            "must_have_inputs": ["title"],
        },
    }
    mismatch = intent_contract.validate_intent_segment_contract(
        segment=segment,
        task_intent="generate",
        tool_name="llm_generate_document_spec",
        payload={"job": {"context_json": {"target_role_name": "Platform Engineer"}}},
        capability_id="document.spec.generate",
        capability_risk_tier="read_only",
    )
    assert mismatch == "must_have_inputs_missing:title"


def test_validate_intent_segment_contract_accepts_collapsed_documentspec_objective_without_format() -> None:
    segment = {
        "intent": "generate",
        "objective": "GenerateDocumentSpec",
        "required_inputs": ["instruction", "format=docx"],
        "suggested_capabilities": ["document.spec.generate"],
        "slots": {
            "risk_level": "read_only",
        },
    }
    mismatch = intent_contract.validate_intent_segment_contract(
        segment=segment,
        task_intent="generate",
        tool_name="llm_generate_document_spec",
        payload={"instruction": "Generate a compact resume document spec."},
        capability_id="document.spec.generate",
        capability_risk_tier="read_only",
    )
    assert mismatch is None


def test_validate_intent_segment_contract_accepts_target_repo_alias_from_query() -> None:
    segment = {
        "intent": "io",
        "objective": "Verify repository exists on GitHub",
        "slots": {
            "entity": "repository",
            "artifact_type": "content",
            "output_format": None,
            "risk_level": "read_only",
            "must_have_inputs": ["target_repo"],
        },
    }
    mismatch = intent_contract.validate_intent_segment_contract(
        segment=segment,
        task_intent="io",
        tool_name="github.repo.list",
        payload={"query": "repo:owner/demo owner:owner"},
        capability_id="github.repo.list",
        capability_risk_tier="read_only",
    )
    assert mismatch is None


def test_validate_intent_segment_contract_accepts_repo_full_name_from_owner_and_repo() -> None:
    segment = {
        "intent": "io",
        "objective": "Verify repository accessibility",
        "slots": {
            "entity": "repository",
            "artifact_type": "validation_report",
            "output_format": None,
            "risk_level": "read_only",
            "must_have_inputs": ["repo_full_name"],
        },
    }
    mismatch = intent_contract.validate_intent_segment_contract(
        segment=segment,
        task_intent="io",
        tool_name="github.repo.list",
        payload={"owner": "narendersurabhi", "repo": "scientific-agent-lab"},
        capability_id="github.repo.list",
        capability_risk_tier="read_only",
    )
    assert mismatch is None


def test_validate_intent_segment_contract_accepts_repo_full_name_from_query() -> None:
    segment = {
        "intent": "io",
        "objective": "Verify repository accessibility",
        "slots": {
            "entity": "repository",
            "artifact_type": "validation_report",
            "output_format": None,
            "risk_level": "read_only",
            "must_have_inputs": ["repo_full_name"],
        },
    }
    mismatch = intent_contract.validate_intent_segment_contract(
        segment=segment,
        task_intent="io",
        tool_name="github.repo.list",
        payload={"query": "repo:scientific-agent-lab owner:narendersurabhi"},
        capability_id="github.repo.list",
        capability_risk_tier="read_only",
    )
    assert mismatch is None


def test_validate_intent_segment_contract_accepts_title_alias_from_topic() -> None:
    segment = {
        "intent": "generate",
        "objective": "Generate document spec",
        "slots": {
            "entity": "document",
            "artifact_type": "document_spec",
            "output_format": None,
            "risk_level": "read_only",
            "must_have_inputs": ["title"],
        },
    }
    mismatch = intent_contract.validate_intent_segment_contract(
        segment=segment,
        task_intent="generate",
        tool_name="llm_generate_document_spec",
        payload={"topic": "active inference"},
        capability_id="document.spec.generate",
        capability_risk_tier="read_only",
    )
    assert mismatch is None


def test_validate_intent_segment_contract_accepts_main_topic_alias_from_topic() -> None:
    segment = {
        "intent": "generate",
        "objective": "Generate document spec",
        "slots": {
            "entity": "document",
            "artifact_type": "document_spec",
            "output_format": None,
            "risk_level": "read_only",
            "must_have_inputs": ["main_topic"],
        },
    }
    mismatch = intent_contract.validate_intent_segment_contract(
        segment=segment,
        task_intent="generate",
        tool_name="llm_generate_document_spec",
        payload={"topic": "AI lifecycle cheatsheet"},
        capability_id="document.spec.generate",
        capability_risk_tier="read_only",
    )
    assert mismatch is None


def test_validate_intent_segment_contract_ignores_goal_for_document_spec_generation() -> None:
    segment = {
        "intent": "generate",
        "objective": "Generate initial DocumentSpec from goal",
        "required_inputs": ["goal"],
        "suggested_capabilities": ["document.spec.generate"],
        "slots": {
            "entity": "document_spec",
            "artifact_type": "document_spec",
            "output_format": "json",
            "risk_level": "bounded_write",
            "must_have_inputs": ["goal"],
        },
    }
    mismatch = intent_contract.validate_intent_segment_contract(
        segment=segment,
        task_intent="generate",
        tool_name="llm_generate_document_spec",
        payload={"instruction": "Generate the requested document specification."},
        capability_id="document.spec.generate",
        capability_risk_tier="read_only",
    )
    assert mismatch is None


def test_validate_intent_segment_contract_ignores_workspace_path_for_document_spec_generation() -> None:
    segment = {
        "intent": "generate",
        "objective": "Generate a document spec",
        "required_inputs": ["goal", "workspace_path"],
        "slots": {
            "entity": "artifact",
            "artifact_type": "content",
            "output_format": "json",
            "risk_level": "read_only",
            "must_have_inputs": ["goal", "workspace_path"],
        },
    }
    mismatch = intent_contract.validate_intent_segment_contract(
        segment=segment,
        task_intent="generate",
        tool_name="document.spec.generate",
        payload={
            "instruction": "Create a document spec.",
            "topic": "Agent + Capability Workbench",
            "audience": "operators",
            "tone": "practical",
        },
        capability_id="document.spec.generate",
        capability_risk_tier="read_only",
    )
    assert mismatch is None


def test_validate_intent_segment_contract_ignores_final_path_for_document_spec_generation() -> None:
    segment = {
        "intent": "generate",
        "objective": "GenerateDocumentSpec",
        "required_inputs": ["instruction"],
        "suggested_capabilities": ["document.spec.generate"],
        "slots": {
            "entity": "document_spec",
            "artifact_type": "document_spec",
            "output_format": "json",
            "risk_level": "read_only",
            "must_have_inputs": ["instruction"],
        },
    }
    mismatch = intent_contract.validate_intent_segment_contract(
        segment=segment,
        task_intent="generate",
        tool_name="llm_generate_document_spec",
        payload={
            "instruction": "Create a document spec.",
            "path": "Narender.docx",
            "output_path": "Narender.docx",
        },
        capability_id="document.spec.generate",
        capability_risk_tier="read_only",
    )
    assert mismatch is None


def test_validate_intent_segment_contract_accepts_length_with_instruction() -> None:
    segment = {
        "intent": "generate",
        "objective": "Generate document spec",
        "slots": {
            "entity": "document",
            "artifact_type": "document_spec",
            "output_format": None,
            "risk_level": "read_only",
            "must_have_inputs": ["length"],
        },
    }
    mismatch = intent_contract.validate_intent_segment_contract(
        segment=segment,
        task_intent="generate",
        tool_name="llm_generate_document_spec",
        payload={"instruction": "Generate a concise one-page resume style document spec."},
        capability_id="document.spec.generate",
        capability_risk_tier="read_only",
    )
    assert mismatch is None


def test_derive_envelope_clarification_requires_path_for_render_without_explicit_path() -> None:
    clarification = intent_contract.derive_envelope_clarification(
        goal="Render the approved document spec as a PDF.",
        profile={
            "intent": "render",
            "low_confidence": False,
            "slot_values": {"intent_action": "render"},
            "blocking_slots": ["output_format"],
            "missing_slots": [],
        },
        graph={
            "segments": [
                {
                    "id": "s1",
                    "intent": "render",
                    "objective": "Render final artifact",
                    "required_inputs": ["input_data", "path_or_format"],
                    "suggested_capabilities": ["document.pdf.render"],
                    "slots": {
                        "output_format": "pdf",
                        "risk_level": "bounded_write",
                        "must_have_inputs": ["path"],
                    },
                }
            ]
        },
        candidate_required_inputs_by_segment={"s1": ["document_spec", "path"]},
    )

    assert clarification["missing_inputs"] == ["path"]
    assert clarification["questions"] == ["What output path or filename should be used?"]


def test_derive_envelope_clarification_requires_instruction_for_generic_document_request() -> None:
    clarification = intent_contract.derive_envelope_clarification(
        goal="create a document\n\nUser clarification: hello.docx",
        profile={
            "intent": "render",
            "low_confidence": False,
            "slot_values": {"intent_action": "render", "path": "hello.docx", "output_format": "docx"},
            "blocking_slots": [],
            "missing_slots": [],
        },
        graph={
            "segments": [
                {
                    "id": "s1",
                    "intent": "generate",
                    "objective": "Generate a DocumentSpec based on user request",
                    "required_inputs": ["document_instruction"],
                    "suggested_capabilities": ["document.spec.generate"],
                    "slots": {
                        "artifact_type": "document_spec",
                        "must_have_inputs": ["document_instruction"],
                    },
                },
                {
                    "id": "s2",
                    "intent": "render",
                    "objective": "Render the DocumentSpec into a DOCX file",
                    "required_inputs": ["document_spec", "output_path"],
                    "suggested_capabilities": ["document.docx.render"],
                    "slots": {
                        "output_format": "docx",
                        "must_have_inputs": ["document_spec", "path"],
                    },
                },
            ]
        },
        candidate_required_inputs_by_segment={
            "s1": ["instruction"],
            "s2": ["document_spec", "path"],
        },
    )

    assert clarification["missing_inputs"] == ["instruction"]
    assert clarification["questions"] == ["What should this specifically cover?"]


def test_derive_envelope_clarification_treats_specific_goal_as_instruction() -> None:
    clarification = intent_contract.derive_envelope_clarification(
        goal="Create a two-page Kubernetes incident runbook for SREs in a practical tone.",
        profile={
            "intent": "generate",
            "low_confidence": False,
            "slot_values": {"intent_action": "generate"},
            "blocking_slots": [],
            "missing_slots": [],
        },
        graph={
            "segments": [
                {
                    "id": "s1",
                    "intent": "generate",
                    "objective": "Generate a DocumentSpec based on user request",
                    "required_inputs": ["instruction"],
                    "suggested_capabilities": ["document.spec.generate"],
                    "slots": {
                        "artifact_type": "document_spec",
                        "must_have_inputs": ["instruction"],
                    },
                }
            ]
        },
        candidate_required_inputs_by_segment={"s1": ["instruction"]},
    )

    assert clarification["missing_inputs"] == []


def test_derive_envelope_clarification_uses_capability_required_document_fields() -> None:
    clarification = intent_contract.derive_envelope_clarification(
        goal="Create a deployment report",
        profile={
            "intent": "generate",
            "low_confidence": False,
            "slot_values": {
                "intent_action": "generate",
                "instruction": "Create a deployment report",
                "topic": "Deployment report",
            },
            "blocking_slots": [],
            "missing_slots": [],
        },
        graph={
            "segments": [
                {
                    "id": "s1",
                    "intent": "generate",
                    "objective": "Generate document spec",
                    "required_inputs": ["instruction", "topic"],
                    "suggested_capabilities": ["document.spec.generate"],
                    "slots": {
                        "artifact_type": "document_spec",
                        "must_have_inputs": ["instruction", "topic"],
                    },
                }
            ]
        },
        candidate_required_inputs_by_segment={
            "s1": ["instruction", "topic", "audience", "tone"],
        },
    )

    assert clarification["missing_inputs"] == ["audience", "tone"]
    assert clarification["questions"] == [
        "Who is the target audience?",
        "What tone should it use?",
    ]


def test_derive_envelope_clarification_skips_target_system_when_capability_is_specific() -> None:
    clarification = intent_contract.derive_envelope_clarification(
        goal="List GitHub branches and summarize release readiness.",
        profile={
            "intent": "io",
            "low_confidence": False,
            "slot_values": {"intent_action": "io"},
            "blocking_slots": ["target_system"],
            "missing_slots": [],
        },
        graph={
            "segments": [
                {
                    "id": "s1",
                    "intent": "io",
                    "objective": "List GitHub branches",
                    "required_inputs": ["source_or_query"],
                    "suggested_capabilities": ["github.branch.list"],
                    "slots": {
                        "risk_level": "read_only",
                        "must_have_inputs": ["query"],
                    },
                }
            ]
        },
        candidate_required_inputs_by_segment={"s1": ["query"]},
    )

    assert clarification["missing_inputs"] == []


def test_derive_envelope_clarification_detects_intent_disagreement() -> None:
    clarification = intent_contract.derive_envelope_clarification(
        goal="Create a release report",
        profile={
            "intent": "io",
            "low_confidence": True,
            "slot_values": {"intent_action": "io"},
            "blocking_slots": [],
            "missing_slots": [],
        },
        heuristic_profile={
            "intent": "generate",
            "source": "heuristic",
        },
        graph={
            "segments": [
                {
                    "id": "s1",
                    "intent": "generate",
                    "objective": "Generate the release report",
                    "required_inputs": ["instruction"],
                    "suggested_capabilities": ["document.spec.generate"],
                    "slots": {
                        "artifact_type": "document_spec",
                        "must_have_inputs": ["instruction"],
                    },
                }
            ]
        },
        context_capability_candidates=["document.spec.generate"],
        candidate_required_inputs_by_segment={"s1": ["instruction"]},
    )

    assert clarification["missing_inputs"] == ["intent_action"]
    assert clarification["blocking_slots"] == ["intent_action"]
    assert clarification["clarification_mode"] == "intent_disagreement"
    assert clarification["disagreement"]["reason_code"] == "graph_intent_conflict"
    assert clarification["disagreement"]["capability_intents"] == ["generate"]
    assert "generate new content" in clarification["questions"][0]


def test_validate_intent_segment_contract_accepts_document_instruction_alias() -> None:
    segment = {
        "intent": "generate",
        "objective": "Generate document spec",
        "slots": {
            "entity": "document_spec",
            "artifact_type": "document_spec",
            "output_format": None,
            "risk_level": "read_only",
            "must_have_inputs": ["document_instruction"],
        },
    }
    mismatch = intent_contract.validate_intent_segment_contract(
        segment=segment,
        task_intent="generate",
        tool_name="llm_generate_document_spec",
        payload={"instruction": "Generate a Kubernetes incident runbook."},
        capability_id="document.spec.generate",
        capability_risk_tier="read_only",
    )

    assert mismatch is None


def test_validate_intent_segment_contract_accepts_document_goal_aliases() -> None:
    segment = {
        "intent": "generate",
        "objective": "Generate initial DocumentSpec from goal",
        "slots": {
            "entity": "document_spec",
            "artifact_type": "document_spec",
            "output_format": "json",
            "risk_level": "read_only",
            "must_have_inputs": ["document_goal"],
        },
    }
    mismatch = intent_contract.validate_intent_segment_contract(
        segment=segment,
        task_intent="generate",
        tool_name="document.spec.generate_iterative",
        payload={
            "instruction": "cover agentic ai ops at the top tier companies",
            "job": {
                "goal": "cover agentic ai ops at the top tier companies",
                "context_json": {
                    "topic": "agentic ai ops at top tier companies",
                },
            },
        },
        capability_id="document.spec.generate_iterative",
        capability_risk_tier="read_only",
    )

    assert mismatch is None


def test_select_active_execution_target_prefers_segment_with_pending_overlap() -> None:
    target = intent_contract.select_active_execution_target(
        graph={
            "segments": [
                {
                    "id": "s1",
                    "intent": "generate",
                    "required_inputs": ["instruction", "audience", "tone"],
                    "suggested_capabilities": ["document.spec.generate"],
                },
                {
                    "id": "s2",
                    "intent": "io",
                    "required_inputs": ["query"],
                    "suggested_capabilities": ["github.issue.search"],
                },
            ]
        },
        candidate_capabilities={
            "s1": ["document.spec.generate"],
            "s2": ["github.issue.search"],
        },
        known_slot_values={"instruction": "Create a report", "audience": "SSE"},
        pending_fields=["tone", "query"],
        preferred_segment_id="s1",
        preferred_capability_id="document.spec.generate",
    )

    assert target is not None
    assert target.segment_id == "s1"
    assert target.capability_id == "document.spec.generate"
    assert target.unresolved_fields == ("tone",)


def test_select_active_execution_target_shifts_to_render_when_path_is_only_remaining_field() -> None:
    target = intent_contract.select_active_execution_target(
        graph={
            "segments": [
                {
                    "id": "s1",
                    "intent": "generate",
                    "required_inputs": ["instruction", "topic", "audience", "tone"],
                    "suggested_capabilities": ["document.spec.generate"],
                },
                {
                    "id": "s2",
                    "intent": "render",
                    "required_inputs": ["document_spec", "output_filename"],
                    "suggested_capabilities": ["document.docx.render"],
                },
            ]
        },
        candidate_capabilities={
            "s1": ["document.spec.generate"],
            "s2": ["document.docx.render"],
        },
        known_slot_values={
            "instruction": "Create a report",
            "topic": "Kubernetes",
            "audience": "SSE",
            "tone": "practical",
        },
        pending_fields=["path"],
        preferred_segment_id="s1",
        preferred_capability_id="document.spec.generate",
    )

    assert target is not None
    assert target.segment_id == "s2"
    assert target.capability_id == "document.docx.render"
    assert target.unresolved_fields == ("path",)
