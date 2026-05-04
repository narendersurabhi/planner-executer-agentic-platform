from libs.core.job_projection import (
    compact_document_job_payload,
    merge_document_job_payload,
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


def test_merge_document_job_payload_backfills_context_from_job_context() -> None:
    merged = merge_document_job_payload(
        {},
        job_context={
            "topic": "agentic ai ops",
            "audience": "architects implementing agentic ai platforms",
            "tone": "practical",
            "today": "2026-04-19",
        },
        default_goal="cover agentic ai ops at the top tier companies",
    )

    assert merged == {
        "goal": "cover agentic ai ops at the top tier companies",
        "topic": "agentic ai ops",
        "audience": "architects implementing agentic ai platforms",
        "tone": "practical",
        "today": "2026-04-19",
        "context_json": {
            "topic": "agentic ai ops",
            "audience": "architects implementing agentic ai platforms",
            "tone": "practical",
            "today": "2026-04-19",
        },
    }


def test_project_document_generation_inputs_applies_conservative_defaults() -> None:
    projected = project_document_generation_inputs(
        {
            "goal": "Create a document about agentic AI ops",
            "context_json": {
                "topic": "agentic AI ops at top companies",
            },
        },
        apply_defaults=True,
    )

    assert projected == {
        "instruction": "Create a document about agentic AI ops",
        "topic": "agentic AI ops at top companies",
        "audience": "general professional audience",
        "tone": "practical",
    }


def test_project_document_generation_inputs_defaults_topic_from_instruction_when_missing() -> None:
    projected = project_document_generation_inputs(
        {
            "goal": "Create a document",
            "context_json": {},
        },
        default_goal="Create a document",
        apply_defaults=True,
    )

    assert projected == {
        "instruction": "Create a document",
        "topic": "Create a document",
        "audience": "general professional audience",
        "tone": "practical",
    }
