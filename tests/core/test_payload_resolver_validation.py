from libs.core.payload_resolver import validate_tool_inputs


def test_validate_tool_inputs_missing_required() -> None:
    schemas = {
        "docx_generate_from_spec": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "document_spec": {"type": "object"}},
            "required": ["path", "document_spec"],
        }
    }
    tool_inputs = {"docx_generate_from_spec": {"path": "resumes/out.docx"}}

    errors = validate_tool_inputs(tool_inputs, schemas)

    assert "docx_generate_from_spec" in errors
    assert "document_spec" in errors["docx_generate_from_spec"]


def test_validate_tool_inputs_ok() -> None:
    schemas = {
        "docx_generate_from_spec": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "document_spec": {"type": "object"}},
            "required": ["path", "document_spec"],
        }
    }
    tool_inputs = {"docx_generate_from_spec": {"path": "resumes/out.docx", "document_spec": {}}}

    errors = validate_tool_inputs(tool_inputs, schemas)

    assert errors == {}
