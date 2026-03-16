from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from libs.framework.tool_runtime import Tool, ToolRegistry
from libs.tools.core_ops import CoreOpsHandlers, register_core_ops_tools
from libs.tools.docx_generate_from_spec import register_docx_tools
from libs.tools.document_spec_iterative import register_document_spec_iterative_tools
from libs.tools.document_spec_llm import register_document_spec_llm_tools
from libs.tools.document_spec_validate import register_document_spec_tools
from libs.tools.github_tools import register_github_tools
from libs.tools.llm_tool_groups import register_coding_agent_tools, register_llm_text_tool
from libs.tools.openapi_iterative import register_openapi_iterative_tools
from libs.tools.pdf_generate_from_spec import register_pdf_tools

from .llm_provider import LLMProvider
from .models import RiskLevel, ToolIntent, ToolSpec


@dataclass(frozen=True)
class ToolCatalogHandlers:
    core_ops_handlers: CoreOpsHandlers
    resolve_llm_timeout_s: Callable[[Optional[LLMProvider]], int]
    resolve_coding_agent_timeout_s: Callable[[], int]
    resolve_llm_iterative_timeout_s: Callable[[Optional[LLMProvider]], int]
    llm_generate: Callable[[Dict[str, Any], LLMProvider], Dict[str, Any]]
    coding_agent_generate: Callable[[Dict[str, Any]], Dict[str, Any]]
    coding_agent_autonomous: Callable[[Dict[str, Any], LLMProvider], Dict[str, Any]]
    coding_agent_publish_pr: Callable[[Dict[str, Any]], Dict[str, Any]]
    llm_generate_document_spec: Callable[[Dict[str, Any], LLMProvider], Dict[str, Any]]
    llm_improve_document_spec: Callable[[Dict[str, Any], LLMProvider], Dict[str, Any]]
    sanitize_document_spec: Callable[[dict[str, Any]], dict[str, Any]]


def register_default_tools(
    registry: ToolRegistry,
    *,
    handlers: ToolCatalogHandlers,
    http_fetch_enabled: bool = False,
    llm_enabled: bool = False,
    llm_provider: Optional[LLMProvider] = None,
) -> None:
    registry.register(
        Tool(
            spec=ToolSpec(
                name="json_transform",
                description="Wrap or reshape already-available JSON input",
                usage_guidance=(
                    "Use only when you already have structured JSON and need to pass it forward. "
                    "Provide the JSON as the 'input' field. This tool does NOT generate new content."
                ),
                input_schema={
                    "type": "object",
                    "properties": {"input": {}},
                    "required": ["input"],
                },
                output_schema={
                    "type": "object",
                    "properties": {"result": {}},
                    "required": ["result"],
                },
                timeout_s=5,
                risk_level=RiskLevel.low,
                tool_intent=ToolIntent.transform,
            ),
            handler=lambda payload: {"result": payload.get("input")},
        )
    )

    register_core_ops_tools(
        registry,
        handlers=handlers.core_ops_handlers,
        http_fetch_enabled=http_fetch_enabled,
    )

    register_docx_tools(registry)
    register_pdf_tools(registry)
    register_document_spec_tools(registry)
    register_github_tools(registry)

    if not llm_enabled:
        return
    if llm_provider is None:
        raise ValueError("llm_enabled requires a llm_provider instance")

    llm_timeout_s = handlers.resolve_llm_timeout_s(llm_provider)
    coding_agent_timeout_s = handlers.resolve_coding_agent_timeout_s()
    llm_iterative_timeout_s = handlers.resolve_llm_iterative_timeout_s(llm_provider)

    register_llm_text_tool(
        registry,
        timeout_s=llm_timeout_s,
        handler=lambda payload, provider=llm_provider: handlers.llm_generate(payload, provider),
    )
    register_coding_agent_tools(
        registry,
        timeout_s=coding_agent_timeout_s,
        handler_generate=handlers.coding_agent_generate,
        handler_autonomous=lambda payload, provider=llm_provider: handlers.coding_agent_autonomous(
            payload, provider
        ),
        handler_publish_pr=handlers.coding_agent_publish_pr,
    )
    register_document_spec_llm_tools(
        registry,
        llm_provider,
        timeout_s=llm_timeout_s,
        sanitize_document_spec=handlers.sanitize_document_spec,
    )
    register_document_spec_iterative_tools(
        registry,
        llm_provider,
        timeout_s=llm_iterative_timeout_s,
        generate_document_spec=handlers.llm_generate_document_spec,
        improve_document_spec=handlers.llm_improve_document_spec,
        sanitize_document_spec=handlers.sanitize_document_spec,
    )
    register_openapi_iterative_tools(
        registry,
        llm_provider,
        timeout_s=llm_iterative_timeout_s,
    )
