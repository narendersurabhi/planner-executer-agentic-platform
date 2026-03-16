from libs.core import tool_plugins
from libs.framework.tool_runtime import ToolRegistry


def test_call_register_fn_passes_supported_context_kwargs() -> None:
    seen: dict[str, object] = {}

    def register_tools(
        registry,
        *,
        context,
        llm_enabled,
        llm_provider,
        http_fetch_enabled,
    ) -> None:
        seen["registry"] = registry
        seen["context"] = context
        seen["llm_enabled"] = llm_enabled
        seen["llm_provider"] = llm_provider
        seen["http_fetch_enabled"] = http_fetch_enabled

    registry = ToolRegistry()
    provider = object()

    tool_plugins.call_register_fn(
        register_tools,
        registry,
        llm_enabled=True,
        llm_provider=provider,
        http_fetch_enabled=False,
    )

    assert seen["registry"] is registry
    assert seen["context"] == {
        "llm_enabled": True,
        "llm_provider": provider,
        "http_fetch_enabled": False,
    }
    assert seen["llm_enabled"] is True
    assert seen["llm_provider"] is provider
    assert seen["http_fetch_enabled"] is False
