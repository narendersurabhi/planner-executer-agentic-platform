from libs.core import payload_resolver


def test_resolve_tool_payload_drops_final_artifact_fields_for_document_spec_generation() -> None:
    payload = payload_resolver.resolve_tool_payload(
        "llm_generate_document_spec",
        "GenerateDocumentSpec",
        {
            "job_context": {
                "path": "Narender.docx",
                "output_format": "docx",
                "topic": "Agentic AI Ops best practices",
            }
        },
        {"output_path": "Narender.docx"},
        {
            "llm_generate_document_spec": {
                "instruction": "Create a document spec.",
                "topic": "Agentic AI Ops best practices",
                "audience": "agentic ai ops engineers",
                "tone": "practical",
            }
        },
    )

    assert payload == {
        "instruction": "Create a document spec.",
        "topic": "Agentic AI Ops best practices",
        "audience": "agentic ai ops engineers",
        "tone": "practical",
    }


def test_reference_payload_validation_drops_final_artifact_fields_for_document_spec_generation() -> None:
    payload = payload_resolver.normalize_reference_payload_for_validation(
        {
            "instruction": "Create a document spec.",
            "topic": "Agentic AI Ops best practices",
            "audience": "agentic ai ops engineers",
            "tone": "practical",
            "path": "Narender.docx",
            "output_format": "docx",
        },
        tool_name="document.spec.generate",
    )

    assert payload == {
        "instruction": "Create a document spec.",
        "topic": "Agentic AI Ops best practices",
        "audience": "agentic ai ops engineers",
        "tone": "practical",
    }
