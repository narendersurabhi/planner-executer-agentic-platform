import pytest

from libs.core.job_projection import (
    project_document_generation_inputs,
    project_markdown_document_generation_inputs,
)
from libs.core.llm_provider import LLMProvider
from libs.framework.tool_runtime import ToolExecutionError
from libs.tools.document_spec_llm import (
    llm_generate_document_spec,
    llm_generate_document_spec_from_markdown,
)


class _ProviderStub(LLMProvider):
    def generate(self, prompt: str):  # pragma: no cover - compatibility only
        raise NotImplementedError

    def generate_json_object(self, prompt: str) -> dict:
        del prompt
        return {"blocks": [{"type": "paragraph", "text": "hello"}]}


def test_project_document_generation_inputs_excludes_markdown_source() -> None:
    job = {
        "goal": "Transform markdown source into a DocumentSpec JSON.",
        "instruction": "stale instruction",
        "status": "queued",
        "metadata": {"llm_provider": "openai"},
        "context_json": {
            "markdown_text": "# Heading\n\nParagraph",
            "topic": "markdown to docx",
            "audience": "general",
            "tone": "practical",
            "today": "2026-03-14",
            "output_dir": "documents",
        },
    }

    projected = project_document_generation_inputs(job)

    assert projected == {
        "instruction": "stale instruction",
        "topic": "markdown to docx",
        "audience": "general",
        "tone": "practical",
        "today": "2026-03-14",
        "output_dir": "documents",
    }


def test_project_markdown_document_generation_inputs_keeps_markdown_source() -> None:
    job = {
        "goal": "Transform markdown source into a DocumentSpec JSON.",
        "context_json": {
            "markdown_text": "# Heading\n\nParagraph",
            "topic": "markdown to docx",
            "audience": "general",
            "tone": "practical",
            "today": "2026-03-14",
            "output_dir": "documents",
        },
    }

    projected = project_markdown_document_generation_inputs(job)

    assert projected == {
        "markdown_text": "# Heading\n\nParagraph",
        "topic": "markdown to docx",
        "audience": "general",
        "tone": "practical",
        "today": "2026-03-14",
        "output_dir": "documents",
    }


def test_project_document_generation_inputs_uses_goal_as_instruction_when_missing() -> None:
    job = {
        "goal": "Generate a report",
        "topic": "Quarterly report",
        "audience": "executives",
        "tone": "neutral",
        "today": "2026-03-16",
        "output_dir": "documents",
    }

    projected = project_document_generation_inputs(job)

    assert projected["instruction"] == "Generate a report"
    assert projected["topic"] == "Quarterly report"
    assert projected["audience"] == "executives"


def test_llm_generate_document_spec_rejects_job_payload() -> None:
    with pytest.raises(ToolExecutionError) as exc:
        llm_generate_document_spec(
            {"job": {"goal": "Generate a document"}},
            _ProviderStub(),
            sanitize_document_spec=lambda value: value,
        )

    assert "job is not supported" in str(exc.value)


def test_llm_generate_document_spec_requires_explicit_fields() -> None:
    with pytest.raises(ToolExecutionError) as exc:
        llm_generate_document_spec(
            {"topic": "Quarterly report"},
            _ProviderStub(),
            sanitize_document_spec=lambda value: value,
        )

    assert "Missing required explicit fields" in str(exc.value)


def test_llm_generate_document_spec_uses_explicit_fields_only() -> None:
    result = llm_generate_document_spec(
        {
            "instruction": "Generate a concise quarterly report document spec.",
            "topic": "Quarterly report",
            "audience": "executives",
            "tone": "neutral",
            "today": "2026-03-16",
            "output_dir": "documents",
        },
        _ProviderStub(),
        sanitize_document_spec=lambda value: value,
    )

    assert result == {"document_spec": {"blocks": [{"type": "paragraph", "text": "hello"}]}}


def test_llm_generate_document_spec_from_markdown_rejects_job_payload() -> None:
    with pytest.raises(ToolExecutionError) as exc:
        llm_generate_document_spec_from_markdown(
            {"job": {"goal": "Generate a document"}},
            _ProviderStub(),
            sanitize_document_spec=lambda value: value,
        )

    assert "job is not supported" in str(exc.value)


def test_llm_generate_document_spec_from_markdown_requires_markdown_text() -> None:
    with pytest.raises(ToolExecutionError) as exc:
        llm_generate_document_spec_from_markdown(
            {"topic": "Quarterly report"},
            _ProviderStub(),
            sanitize_document_spec=lambda value: value,
        )

    assert "Missing required explicit fields" in str(exc.value)


def test_llm_generate_document_spec_from_markdown_uses_markdown_source() -> None:
    result = llm_generate_document_spec_from_markdown(
        {
            "markdown_text": "# Heading\n\nParagraph",
            "topic": "Quarterly report",
            "audience": "executives",
            "tone": "neutral",
            "today": "2026-03-16",
            "output_dir": "documents",
        },
        _ProviderStub(),
        sanitize_document_spec=lambda value: value,
    )

    assert result == {"document_spec": {"blocks": [{"type": "paragraph", "text": "hello"}]}}
