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


def test_decompose_goal_intent_produces_ordered_segments() -> None:
    graph = intent_contract.decompose_goal_intent(
        "Fetch GitHub repositories, then summarize findings, then render a PDF report."
    )
    segments = graph["segments"]
    assert len(segments) >= 3
    assert segments[0]["intent"] == "io"
    assert segments[1]["depends_on"] == ["s1"]
    assert segments[-1]["intent"] == "render"
    assert "document.pdf.generate" in segments[-1]["suggested_capabilities"]
    assert isinstance(segments[-1]["slots"], dict)
    assert segments[-1]["slots"]["output_format"] == "pdf"
    assert "must_have_inputs" in segments[-1]["slots"]
    assert graph["summary"]["schema_version"] == "intent_v2"


def test_decompose_goal_intent_handles_empty_goal() -> None:
    graph = intent_contract.decompose_goal_intent("")
    assert graph["segments"] == []
    assert graph["summary"]["segment_count"] == 0
    assert graph["summary"]["schema_version"] == "intent_v2"


def test_decompose_goal_intent_render_segment_does_not_insert_derive_step() -> None:
    graph = intent_contract.decompose_goal_intent("Render a PDF status report.")
    segments = graph["segments"]
    assert len(segments) == 1
    assert segments[0]["intent"] == "render"
    assert segments[0]["suggested_capabilities"] == ["document.pdf.generate"]


def test_validate_intent_segment_contract_allows_renderer_without_explicit_path() -> None:
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
        tool_name="document.pdf.generate",
        payload={"document_spec": {"blocks": []}},
        capability_id="document.pdf.generate",
        capability_risk_tier="bounded_write",
    )
    assert mismatch is None


def test_validate_intent_segment_contract_accepts_output_path_for_path_requirement() -> None:
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
        tool_name="docx_generate_from_spec",
        payload={"document_spec": {"blocks": []}, "output_path": "artifacts/output.docx"},
        capability_id="document.docx.generate",
        capability_risk_tier="bounded_write",
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
        tool_name="document.docx.generate",
        payload={"path": "artifacts/output.docx"},
        capability_id="document.docx.generate",
        capability_risk_tier="bounded_write",
    )
    assert mismatch == "output_format_mismatch:document.docx.generate:expected=pdf:got=docx"


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
        capability_id="document.output.derive",
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


def test_validate_intent_segment_contract_allows_derive_output_without_document_spec() -> None:
    segment = {
        "intent": "transform",
        "objective": "Derive output path",
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
        tool_name="derive_output_path",
        payload={"topic": "active inference", "output_dir": "documents", "output_extension": "docx"},
        capability_id="document.output.derive",
        capability_risk_tier="read_only",
    )
    assert mismatch is None


def test_validate_intent_segment_contract_accepts_filename_aliases_from_payload() -> None:
    segment = {
        "intent": "generate",
        "objective": "Generate document spec",
        "slots": {
            "entity": "document",
            "artifact_type": "document_spec",
            "output_format": None,
            "risk_level": "read_only",
            "must_have_inputs": ["filename"],
        },
    }
    mismatch = intent_contract.validate_intent_segment_contract(
        segment=segment,
        task_intent="generate",
        tool_name="llm_generate_document_spec",
        payload={"output_path": "documents/platform_engineer_resume.docx"},
        capability_id="document.spec.generate",
        capability_risk_tier="read_only",
    )
    assert mismatch is None


def test_validate_intent_segment_contract_accepts_title_alias_from_job_context() -> None:
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


def test_validate_intent_segment_contract_accepts_date_alias_from_today() -> None:
    segment = {
        "intent": "generate",
        "objective": "Generate document spec",
        "slots": {
            "entity": "document",
            "artifact_type": "document_spec",
            "output_format": None,
            "risk_level": "read_only",
            "must_have_inputs": ["date"],
        },
    }
    mismatch = intent_contract.validate_intent_segment_contract(
        segment=segment,
        task_intent="generate",
        tool_name="llm_generate_document_spec",
        payload={"today": "2026-03-08"},
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
