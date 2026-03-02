from services.worker.app import memory_semantics


def test_select_memory_payload_filters_generic_keys() -> None:
    output = {
        "document_spec": {"doc_type": "generic_document"},
        "extra": "ignore",
    }
    selected = memory_semantics.select_memory_payload("llm_generate_document_spec", output)
    assert selected == {"document_spec": {"doc_type": "generic_document"}}


def test_apply_memory_defaults_prefers_latest_document_spec() -> None:
    payload = {
        "memory": {
            "task_outputs": [
                {"document_spec": {"doc_type": "older"}},
                {"_memory_key": "document_spec:latest", "document_spec": {"doc_type": "latest"}},
            ]
        }
    }
    updated = memory_semantics.apply_memory_defaults("llm_improve_document_spec", payload)
    assert updated["document_spec"]["doc_type"] == "latest"


def test_docx_generate_from_spec_prefers_document_path_alias() -> None:
    payload = {
        "document_spec": {"doc_type": "generic_document"},
        "memory": {
            "task_outputs": [
                {"_memory_key": "docx_path:latest", "path": "documents/latest.docx"},
                {"_memory_key": "docx_path:document:latest", "path": "documents/preferred.docx"},
            ]
        },
    }
    updated = memory_semantics.apply_memory_defaults("docx_generate_from_spec", payload)
    assert updated["path"] == "documents/preferred.docx"


def test_missing_memory_only_inputs_detects_required_document_spec() -> None:
    payload = {"memory": {"task_outputs": []}}
    missing = memory_semantics.missing_memory_only_inputs("llm_improve_document_spec", payload)
    assert missing == ["document_spec"]
