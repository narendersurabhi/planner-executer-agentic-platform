from libs.core import tool_bootstrap
from libs.framework.tool_runtime import ToolRegistry


def test_build_tool_registry_wires_catalog_plugins_and_governance(monkeypatch) -> None:
    calls: list[tuple[str, object]] = []

    monkeypatch.setattr(
        tool_bootstrap.tool_catalog,
        "register_default_tools",
        lambda registry, **kwargs: calls.append(("catalog", registry)),
    )
    monkeypatch.setattr(
        tool_bootstrap.tool_plugins,
        "load_configured_plugins",
        lambda registry, **kwargs: calls.append(("plugins", registry)),
    )
    monkeypatch.setattr(
        tool_bootstrap.tool_governance,
        "filter_registry_tools",
        lambda registry, service_name: calls.append(("governance", service_name)),
    )

    registry = tool_bootstrap.build_tool_registry(
        handlers=object(),
        http_fetch_enabled=False,
        llm_enabled=False,
        llm_provider=None,
        service_name="worker",
    )

    assert isinstance(registry, ToolRegistry)
    assert calls[0][0] == "catalog"
    assert calls[1][0] == "plugins"
    assert calls[2] == ("governance", "worker")
