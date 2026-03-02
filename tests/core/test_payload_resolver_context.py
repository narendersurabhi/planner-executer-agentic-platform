from __future__ import annotations

from libs.core import tool_registry
from libs.core.payload_resolver import (
    normalize_reference_payload_for_validation,
    resolve_tool_inputs_with_errors,
    resolve_tool_payload,
    validate_tool_inputs,
)


def test_derive_output_filename_uses_job_context_topic_for_validation() -> None:
    payload = resolve_tool_payload(
        "derive_output_filename",
        "Call derive_output_filename for documents output.",
        {"job_context": {"topic": "Latency in Distributed Systems", "today": "2026-02-16"}},
        {},
        {},
    )

    schemas = {spec.name: spec.input_schema for spec in tool_registry.default_registry().list_specs()}
    errors = validate_tool_inputs({"derive_output_filename": payload}, schemas)

    assert payload["topic"] == "Latency in Distributed Systems"
    assert payload["today"] == "2026-02-16"
    assert errors == {}


def test_resolve_tool_inputs_with_dependency_reference_path() -> None:
    context = {
        "dependencies_by_name": {
            "GenerateSpec": {
                "llm_generate_document_spec": {
                    "document_spec": {"title": "Latency Guide", "blocks": []}
                }
            }
        }
    }
    tool_inputs = {
        "docx_generate_from_spec": {
            "path": "documents/latency.docx",
            "document_spec": {
                "$from": "dependencies_by_name.GenerateSpec.llm_generate_document_spec.document_spec"
            },
        }
    }
    resolved, errors = resolve_tool_inputs_with_errors(
        ["docx_generate_from_spec"],
        "Render the document",
        context,
        {},
        tool_inputs,
    )
    assert errors == {}
    assert resolved["docx_generate_from_spec"]["document_spec"]["title"] == "Latency Guide"


def test_resolve_tool_inputs_with_reference_default() -> None:
    tool_inputs = {
        "derive_output_filename": {
            "topic": {"$from": "job_context.topic", "$default": "Fallback Topic"},
            "today": "2026-02-16",
            "output_dir": "documents",
            "document_type": "document",
        }
    }
    resolved, errors = resolve_tool_inputs_with_errors(
        ["derive_output_filename"],
        "derive path",
        {},
        {},
        tool_inputs,
    )
    assert errors == {}
    assert resolved["derive_output_filename"]["topic"] == "Fallback Topic"


def test_resolve_tool_inputs_reports_missing_reference() -> None:
    tool_inputs = {
        "docx_generate_from_spec": {
            "path": "documents/out.docx",
            "document_spec": {"$from": "dependencies_by_name.MissingTask.output.document_spec"},
        }
    }
    resolved, errors = resolve_tool_inputs_with_errors(
        ["docx_generate_from_spec"],
        "Render document",
        {"dependencies_by_name": {}},
        {},
        tool_inputs,
    )
    assert resolved == {}
    assert "docx_generate_from_spec" in errors
    assert "reference resolution failed" in errors["docx_generate_from_spec"]


def test_normalize_reference_payload_for_validation_uses_dependency_default() -> None:
    payload = {
        "document_spec": {"$from": "dependencies_by_name.GenerateSpec.llm_generate_document_spec.document_spec"},
        "path": "documents/out.docx",
    }
    normalized = normalize_reference_payload_for_validation(
        payload,
        dependency_defaults={"document_spec": {}},
    )
    assert normalized["document_spec"] == {}
