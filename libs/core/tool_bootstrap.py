from __future__ import annotations

from typing import Optional

from libs.framework.tool_runtime import ToolRegistry

from . import tool_catalog, tool_governance, tool_plugins
from .llm_provider import LLMProvider


def build_tool_registry(
    *,
    handlers: tool_catalog.ToolCatalogHandlers,
    http_fetch_enabled: bool = False,
    llm_enabled: bool = False,
    llm_provider: Optional[LLMProvider] = None,
    service_name: str | None = None,
) -> ToolRegistry:
    registry = ToolRegistry()
    tool_catalog.register_default_tools(
        registry,
        handlers=handlers,
        http_fetch_enabled=http_fetch_enabled,
        llm_enabled=llm_enabled,
        llm_provider=llm_provider,
    )
    tool_plugins.load_configured_plugins(
        registry,
        llm_enabled=llm_enabled,
        llm_provider=llm_provider,
        http_fetch_enabled=http_fetch_enabled,
    )
    tool_governance.filter_registry_tools(registry, service_name)
    return registry


def build_default_registry(
    *,
    http_fetch_enabled: bool = False,
    llm_enabled: bool = False,
    llm_provider: Optional[LLMProvider] = None,
    service_name: str | None = None,
) -> ToolRegistry:
    from . import tool_registry as tool_registry_module

    return build_tool_registry(
        handlers=tool_registry_module._default_catalog_handlers(),
        http_fetch_enabled=http_fetch_enabled,
        llm_enabled=llm_enabled,
        llm_provider=llm_provider,
        service_name=service_name,
    )
