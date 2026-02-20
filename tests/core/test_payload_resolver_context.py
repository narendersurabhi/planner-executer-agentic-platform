from __future__ import annotations

from libs.core import tool_registry
from libs.core.payload_resolver import resolve_tool_payload, validate_tool_inputs


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

