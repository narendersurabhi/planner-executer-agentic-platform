from __future__ import annotations

import inspect
import logging
import os
from importlib import import_module
from importlib import metadata as importlib_metadata
from typing import Any, Callable, Optional

from libs.framework.tool_runtime import ToolExecutionError, ToolRegistry

from .llm_provider import LLMProvider

LOGGER = logging.getLogger(__name__)


class ToolPluginLoadError(ToolExecutionError):
    pass


def _parse_csv_values(raw: str | None) -> set[str]:
    if not raw:
        return set()
    return {part.strip() for part in raw.split(",") if part.strip()}


def _plugin_fail_fast() -> bool:
    return os.getenv("TOOL_PLUGIN_FAIL_FAST", "true").lower() == "true"


def parse_plugin_spec(spec: str) -> tuple[str, str]:
    value = spec.strip()
    if not value:
        raise ToolPluginLoadError("Empty plugin spec in TOOL_PLUGIN_MODULES")
    if ":" not in value:
        return value, "register_tools"
    module_path, attr_name = value.split(":", 1)
    module_path = module_path.strip()
    attr_name = attr_name.strip() or "register_tools"
    if not module_path:
        raise ToolPluginLoadError(f"Invalid plugin spec: {spec}")
    return module_path, attr_name


def resolve_module_register_fn(module: Any, attr_name: str) -> Callable[..., None]:
    candidates = [attr_name]
    if attr_name == "register_tools":
        candidates.append("register")
    for candidate in candidates:
        register_fn = getattr(module, candidate, None)
        if callable(register_fn):
            return register_fn
    raise ToolPluginLoadError(
        f"Tool plugin module '{module.__name__}' missing callable '{attr_name}'"
    )


def call_register_fn(
    register_fn: Callable[..., None],
    registry: ToolRegistry,
    *,
    llm_enabled: bool,
    llm_provider: Optional[LLMProvider],
    http_fetch_enabled: bool,
) -> None:
    signature = inspect.signature(register_fn)
    kwargs: dict[str, Any] = {}
    if "context" in signature.parameters:
        kwargs["context"] = {
            "llm_enabled": llm_enabled,
            "llm_provider": llm_provider,
            "http_fetch_enabled": http_fetch_enabled,
        }
    if "llm_enabled" in signature.parameters:
        kwargs["llm_enabled"] = llm_enabled
    if "llm_provider" in signature.parameters:
        kwargs["llm_provider"] = llm_provider
    if "http_fetch_enabled" in signature.parameters:
        kwargs["http_fetch_enabled"] = http_fetch_enabled
    if "provider" in signature.parameters and "llm_provider" not in signature.parameters:
        kwargs["provider"] = llm_provider
    try:
        register_fn(registry, **kwargs)
    except TypeError as exc:
        raise ToolPluginLoadError(f"Invalid tool plugin signature for {register_fn}: {exc}") from exc


def _iter_entry_points(group: str) -> list[Any]:
    eps = importlib_metadata.entry_points()
    if hasattr(eps, "select"):
        return list(eps.select(group=group))
    return list(eps.get(group, []))


def load_module_plugins(
    registry: ToolRegistry,
    *,
    llm_enabled: bool,
    llm_provider: Optional[LLMProvider],
    http_fetch_enabled: bool,
) -> None:
    plugin_specs = _parse_csv_values(os.getenv("TOOL_PLUGIN_MODULES"))
    for plugin_spec in sorted(plugin_specs):
        try:
            module_path, attr_name = parse_plugin_spec(plugin_spec)
            module = import_module(module_path)
            register_fn = resolve_module_register_fn(module, attr_name)
            call_register_fn(
                register_fn,
                registry,
                llm_enabled=llm_enabled,
                llm_provider=llm_provider,
                http_fetch_enabled=http_fetch_enabled,
            )
            LOGGER.info("tool_plugin_loaded source=module plugin=%s", plugin_spec)
        except Exception as exc:  # noqa: BLE001
            LOGGER.error(
                "tool_plugin_load_failed source=module plugin=%s error=%s",
                plugin_spec,
                exc,
            )
            if _plugin_fail_fast():
                raise


def load_entrypoint_plugins(
    registry: ToolRegistry,
    *,
    llm_enabled: bool,
    llm_provider: Optional[LLMProvider],
    http_fetch_enabled: bool,
) -> None:
    if os.getenv("TOOL_PLUGIN_DISCOVERY_ENABLED", "false").lower() != "true":
        return
    group = os.getenv("TOOL_PLUGIN_ENTRYPOINT_GROUP", "awe.tools")
    for entry_point in _iter_entry_points(group):
        try:
            loaded = entry_point.load()
            if callable(loaded):
                register_fn = loaded
            else:
                register_fn = resolve_module_register_fn(loaded, "register_tools")
            call_register_fn(
                register_fn,
                registry,
                llm_enabled=llm_enabled,
                llm_provider=llm_provider,
                http_fetch_enabled=http_fetch_enabled,
            )
            LOGGER.info("tool_plugin_loaded source=entry_point plugin=%s", entry_point.name)
        except Exception as exc:  # noqa: BLE001
            LOGGER.error(
                "tool_plugin_load_failed source=entry_point plugin=%s error=%s",
                entry_point.name,
                exc,
            )
            if _plugin_fail_fast():
                raise


def load_configured_plugins(
    registry: ToolRegistry,
    *,
    llm_enabled: bool,
    llm_provider: Optional[LLMProvider],
    http_fetch_enabled: bool,
) -> None:
    load_module_plugins(
        registry,
        llm_enabled=llm_enabled,
        llm_provider=llm_provider,
        http_fetch_enabled=http_fetch_enabled,
    )
    load_entrypoint_plugins(
        registry,
        llm_enabled=llm_enabled,
        llm_provider=llm_provider,
        http_fetch_enabled=http_fetch_enabled,
    )
