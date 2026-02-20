from services.worker.app import memory_semantics


def test_select_memory_payload_filters_keys() -> None:
    output = {
        "tailored_resume": {
            "schema_version": "1.0",
            "header": {
                "name": "Jane Doe",
                "title": "Engineer",
                "location": "Okemos, MI",
                "phone": "123",
                "email": "jane@example.com",
                "links": {"linkedin": "https://example.com"},
            },
            "summary": "hi",
            "skills": [],
            "experience": [],
            "education": [],
            "certifications": [],
        },
        "alignment_score": 92.5,
        "alignment_summary": "strong match",
        "extra": "ignore",
    }
    selected = memory_semantics.select_memory_payload("llm_improve_tailored_resume_text", output)
    assert selected == {
        "tailored_resume": {
            "schema_version": "1.0",
            "header": {
                "name": "Jane Doe",
                "title": "Engineer",
                "location": "Okemos, MI",
                "phone": "123",
                "email": "jane@example.com",
                "links": {"linkedin": "https://example.com"},
            },
            "summary": "hi",
            "skills": [],
            "experience": [],
            "education": [],
            "certifications": [],
        },
        "alignment_score": 92.5,
        "alignment_summary": "strong match",
    }


def test_apply_memory_defaults_fills_missing_tailored_text() -> None:
    payload = {
        "memory": {
            "task_outputs": [
                {
                    "tailored_resume": {
                        "schema_version": "1.0",
                        "header": {
                            "name": "Jane Doe",
                            "title": "Engineer",
                            "location": "Okemos, MI",
                            "phone": "123",
                            "email": "jane@example.com",
                            "links": {"linkedin": "https://example.com"},
                        },
                        "summary": "from memory",
                        "skills": [],
                        "experience": [],
                        "education": [],
                        "certifications": [],
                    },
                    "alignment_score": 88,
                },
                {
                    "tailored_resume": {
                        "schema_version": "1.0",
                        "header": {
                            "name": "Jane Doe",
                            "title": "Engineer",
                            "location": "Okemos, MI",
                            "phone": "123",
                            "email": "jane@example.com",
                            "links": {"linkedin": "https://example.com"},
                        },
                        "summary": "older",
                        "skills": [],
                        "experience": [],
                        "education": [],
                        "certifications": [],
                    }
                },
            ]
        }
    }
    updated = memory_semantics.apply_memory_defaults("llm_improve_tailored_resume_text", payload)
    assert updated["tailored_resume"]["summary"] == "from memory"


def test_apply_memory_defaults_overrides_manual_when_memory_only() -> None:
    payload = {
        "tailored_resume": {
            "header": {
                "name": "Manual",
                "title": "Role",
                "location": "Somewhere",
                "phone": "123",
                "email": "a@b.com",
                "links": {"linkedin": "https://example.com"},
            },
            "schema_version": "1.0",
            "summary": "manual",
            "skills": [{"group_name": "Tech", "items": ["t"]}],
            "experience": [
                {
                    "company": "Manual",
                    "title": "Role",
                    "location": "Somewhere",
                    "dates": "2020-2021",
                    "bullets": ["Did thing"],
                }
            ],
            "education": [],
            "certifications": [],
        },
        "memory": {
            "task_outputs": [
                {
                    "_memory_key": "tailored_resume:latest",
                    "tailored_resume": {
                        "header": {
                            "name": "Memory",
                            "title": "Role",
                            "location": "Somewhere",
                            "phone": "123",
                            "email": "a@b.com",
                            "links": {"linkedin": "https://example.com"},
                        },
                        "schema_version": "1.0",
                        "summary": "from memory",
                        "skills": [{"group_name": "Tech", "items": ["t"]}],
                        "experience": [
                            {
                                "company": "Memory",
                                "title": "Role",
                                "location": "Somewhere",
                                "dates": "2020-2021",
                                "bullets": ["Did thing"],
                            }
                        ],
                        "education": [],
                        "certifications": [],
                    },
                }
            ]
        },
    }
    updated = memory_semantics.apply_memory_defaults("llm_improve_tailored_resume_text", payload)
    assert updated["tailored_resume"]["summary"] == "from memory"


def test_apply_memory_defaults_prefers_latest_keyed_entry() -> None:
    payload = {
        "memory": {
            "task_outputs": [
                {
                    "tailored_resume": {
                        "schema_version": "1.0",
                        "header": {
                            "name": "Jane Doe",
                            "title": "Engineer",
                            "location": "Okemos, MI",
                            "phone": "123",
                            "email": "jane@example.com",
                            "links": {"linkedin": "https://example.com"},
                        },
                        "summary": "older",
                        "skills": [],
                        "experience": [],
                        "education": [],
                        "certifications": [],
                    }
                },
                {
                    "_memory_key": "tailored_resume:latest",
                    "tailored_resume": {
                        "schema_version": "1.0",
                        "header": {
                            "name": "Jane Doe",
                            "title": "Engineer",
                            "location": "Okemos, MI",
                            "phone": "123",
                            "email": "jane@example.com",
                            "links": {"linkedin": "https://example.com"},
                        },
                        "summary": "preferred",
                        "skills": [],
                        "experience": [],
                        "education": [],
                        "certifications": [],
                    },
                },
            ]
        }
    }
    updated = memory_semantics.apply_memory_defaults("llm_improve_tailored_resume_text", payload)
    assert updated["tailored_resume"]["summary"] == "preferred"


def test_extract_memory_value_prefers_alignment_latest_key() -> None:
    entries = [
        {"alignment_score": 80},
        {"_memory_key": "alignment_score:latest", "alignment_score": 92},
    ]
    value = memory_semantics.extract_memory_value(entries, "alignment_score")
    assert value == 92


def test_select_memory_payload_includes_resume_doc_spec() -> None:
    output = {
        "resume_doc_spec": {"doc_type": "resume"},
        "validation": {"valid": True, "errors": [], "warnings": [], "stats": {"block_count": 1}},
        "extra": "ignore",
    }
    selected = memory_semantics.select_memory_payload(
        "llm_generate_resume_doc_spec_from_text", output
    )
    assert selected == {
        "resume_doc_spec": {"doc_type": "resume"},
        "validation": {"valid": True, "errors": [], "warnings": [], "stats": {"block_count": 1}},
    }


def test_select_memory_payload_includes_cover_letter() -> None:
    output = {
        "cover_letter": {"full_name": "Jane Doe", "body": "Hello"},
        "extra": "ignore",
    }
    selected = memory_semantics.select_memory_payload(
        "llm_generate_cover_letter_from_resume", output
    )
    assert selected == {"cover_letter": {"full_name": "Jane Doe", "body": "Hello"}}


def test_select_memory_payload_includes_coverletter_doc_spec() -> None:
    output = {
        "coverletter_doc_spec": {"doc_type": "cover_letter"},
        "validation": {"valid": True},
        "extra": "ignore",
    }
    selected = memory_semantics.select_memory_payload(
        "llm_generate_coverletter_doc_spec_from_text", output
    )
    assert selected == {"coverletter_doc_spec": {"doc_type": "cover_letter"}}


def test_apply_memory_defaults_fills_resume_doc_spec() -> None:
    payload = {
        "memory": {
            "task_outputs": [
                {"_memory_key": "resume_doc_spec:latest", "resume_doc_spec": {"doc_type": "resume"}}
            ]
        }
    }
    updated = memory_semantics.apply_memory_defaults("resume_doc_spec_validate", payload)
    assert updated["resume_doc_spec"]["doc_type"] == "resume"


def test_coverletter_doc_spec_converter_fills_from_memory() -> None:
    payload = {
        "memory": {
            "task_outputs": [
                {
                    "_memory_key": "coverletter_doc_spec:latest",
                    "coverletter_doc_spec": {"doc_type": "cover_letter"},
                }
            ]
        }
    }
    updated = memory_semantics.apply_memory_defaults(
        "coverletter_doc_spec_to_document_spec",
        payload,
    )
    assert updated["coverletter_doc_spec"]["doc_type"] == "cover_letter"


def test_docx_generate_from_spec_preserves_manual_value() -> None:
    payload = {
        "document_spec": {"doc_type": "manual"},
        "memory": {
            "task_outputs": [
                {"_memory_key": "document_spec:latest", "document_spec": {"doc_type": "memory"}}
            ]
        },
    }
    updated = memory_semantics.apply_memory_defaults("docx_generate_from_spec", payload)
    assert updated["document_spec"]["doc_type"] == "manual"


def test_cover_letter_generate_fills_from_memory() -> None:
    payload = {
        "memory": {
            "task_outputs": [
                {
                    "_memory_key": "cover_letter:latest",
                    "cover_letter": {"full_name": "Jane Doe", "body": "Hello"},
                },
                {"_memory_key": "docx_path:latest", "path": "cover_letters/jane.docx"},
            ]
        },
    }
    updated = memory_semantics.apply_memory_defaults("cover_letter_generate_ats_docx", payload)
    assert updated["cover_letter"]["full_name"] == "Jane Doe"
    assert updated["path"] == "cover_letters/jane.docx"


def test_missing_memory_only_inputs_detects_missing() -> None:
    payload = {"memory": {"task_outputs": []}}
    missing = memory_semantics.missing_memory_only_inputs("docx_generate_from_spec", payload)
    assert missing == []


def test_document_spec_validate_is_not_memory_only() -> None:
    payload = {"memory": {"task_outputs": []}}
    missing = memory_semantics.missing_memory_only_inputs("document_spec_validate", payload)
    assert missing == []


def test_resume_doc_spec_to_document_spec_is_memory_only() -> None:
    payload = {"memory": {"task_outputs": []}}
    missing = memory_semantics.missing_memory_only_inputs(
        "resume_doc_spec_to_document_spec", payload
    )
    assert missing == ["resume_doc_spec"]


def test_coverletter_doc_spec_to_document_spec_is_memory_only() -> None:
    payload = {"memory": {"task_outputs": []}}
    missing = memory_semantics.missing_memory_only_inputs(
        "coverletter_doc_spec_to_document_spec", payload
    )
    assert missing == ["coverletter_doc_spec"]


def test_cover_letter_generate_is_not_memory_only() -> None:
    payload = {"memory": {"task_outputs": []}}
    missing = memory_semantics.missing_memory_only_inputs("cover_letter_generate_ats_docx", payload)
    assert missing == []


def test_coverletter_doc_spec_llm_tool_is_not_memory_only() -> None:
    payload = {"memory": {"task_outputs": []}}
    missing = memory_semantics.missing_memory_only_inputs(
        "llm_generate_coverletter_doc_spec_from_text", payload
    )
    assert missing == []


def test_stable_memory_keys_for_derive_output_filename_adds_type_alias() -> None:
    resume_keys = memory_semantics.stable_memory_keys(
        "derive_output_filename",
        {"path": "resumes/a.docx", "document_type": "resume"},
    )
    assert "docx_path:latest" in resume_keys
    assert "docx_path:resume:latest" in resume_keys

    cover_keys = memory_semantics.stable_memory_keys(
        "derive_output_filename",
        {"path": "resumes/b.docx", "document_type": "cover_letter"},
    )
    assert "docx_path:latest" in cover_keys
    assert "docx_path:cover_letter:latest" in cover_keys


def test_docx_generate_from_spec_uses_resume_specific_path_alias() -> None:
    payload = {
        "document_spec": {"doc_type": "resume_doc_spec_converted"},
        "memory": {
            "task_outputs": [
                {"_memory_key": "docx_path:latest", "path": "resumes/cover.docx"},
                {"_memory_key": "docx_path:resume:latest", "path": "resumes/resume.docx"},
            ]
        },
    }
    updated = memory_semantics.apply_memory_defaults("docx_generate_from_spec", payload)
    assert updated["path"] == "resumes/resume.docx"


def test_docx_generate_from_spec_uses_cover_letter_specific_path_alias() -> None:
    payload = {
        "document_spec": {"doc_type": "coverletter_doc_spec_converted"},
        "memory": {
            "task_outputs": [
                {"_memory_key": "docx_path:latest", "path": "resumes/resume.docx"},
                {"_memory_key": "docx_path:cover_letter:latest", "path": "resumes/cover.docx"},
            ]
        },
    }
    updated = memory_semantics.apply_memory_defaults("docx_generate_from_spec", payload)
    assert updated["path"] == "resumes/cover.docx"
