from libs.core.job_projection import (
    compact_document_job_payload,
    project_document_generation_inputs,
    project_markdown_document_generation_inputs,
    project_job_payload_for_tool,
)


def test_compact_document_job_payload_strips_unrelated_job_metadata() -> None:
    job = {
        "goal": "Create a DOCX from markdown",
        "status": "queued",
        "metadata": {"llm_provider": "openai"},
        "context_json": {
            "markdown_text": "# Title\n\nBody",
            "topic": "demo",
            "tone": "neutral",
            "today": "2026-03-16",
            "output_dir": "documents",
            "unrelated": {"nested": "value"},
        },
    }

    compact = compact_document_job_payload(job)

    assert compact == {
        "goal": "Create a DOCX from markdown",
        "context_json": {
            "markdown_text": "# Title\n\nBody",
            "topic": "demo",
            "tone": "neutral",
            "today": "2026-03-16",
            "output_dir": "documents",
        },
    }


def test_project_job_payload_for_tool_only_compacts_iterative_document_tools() -> None:
    job = {
        "goal": "Create a DOCX from markdown",
        "status": "queued",
        "metadata": {"llm_provider": "openai"},
        "context_json": {"markdown_text": "# Title\n\nBody"},
    }

    projected = project_job_payload_for_tool("llm_generate_document_spec", job)
    untouched = project_job_payload_for_tool("github.repo.list", job)

    assert projected == job
    assert untouched == job


def test_project_document_generation_inputs_flattens_job_payload_into_explicit_fields() -> None:
    job = {
        "goal": "Transform markdown source into a DocumentSpec JSON.",
        "status": "queued",
        "metadata": {"llm_provider": "openai"},
        "context_json": {
            "markdown_text": "# Title\n\nBody",
            "topic": "markdown to docx",
            "audience": "general",
            "tone": "practical",
            "today": "2026-03-16",
            "output_dir": "documents",
            "target_role_name": "Platform Engineer",
            "unrelated": {"nested": "value"},
        },
    }

    projected = project_document_generation_inputs(job)

    assert projected == {
        "instruction": "Transform markdown source into a DocumentSpec JSON.",
        "topic": "markdown to docx",
        "audience": "general",
        "tone": "practical",
        "today": "2026-03-16",
        "output_dir": "documents",
        "target_role_name": "Platform Engineer",
    }


def test_project_markdown_document_generation_inputs_keeps_markdown_source() -> None:
    job = {
        "goal": "Transform markdown source into a DocumentSpec JSON.",
        "status": "queued",
        "context_json": {
            "markdown_text": "# Title\n\nBody",
            "topic": "markdown to docx",
            "audience": "general",
            "tone": "practical",
            "today": "2026-03-16",
            "output_dir": "documents",
        },
    }

    projected = project_markdown_document_generation_inputs(job)

    assert projected == {
        "markdown_text": "# Title\n\nBody",
        "topic": "markdown to docx",
        "audience": "general",
        "tone": "practical",
        "today": "2026-03-16",
        "output_dir": "documents",
    }
