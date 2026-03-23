from __future__ import annotations

import os
from dataclasses import dataclass, field

from libs.framework.tool_runtime import ToolRegistry

from . import capability_registry, mcp_gateway, tool_bootstrap
from .llm_provider import MockLLMProvider

SCHEMA_VERSION = "runtime-manifest.v1"


@dataclass(frozen=True)
class RuntimeCapabilityStatus:
    capability_id: str
    available: bool
    reason: str = ""
    details: tuple[str, ...] = ()
    adapter_types: tuple[str, ...] = ()


@dataclass(frozen=True)
class RuntimeManifest:
    service_name: str
    capability_mode: str
    schema_version: str = SCHEMA_VERSION
    runtime_version: str = "phase0"
    capabilities: dict[str, RuntimeCapabilityStatus] = field(default_factory=dict)
    build_errors: tuple[str, ...] = ()


def build_runtime_manifest(service_name: str = "worker") -> RuntimeManifest:
    normalized_service = (service_name or "worker").strip().lower() or "worker"
    capability_mode = capability_registry.resolve_capability_mode()
    build_errors: list[str] = []
    try:
        registry = capability_registry.load_capability_registry()
    except Exception as exc:  # noqa: BLE001
        return RuntimeManifest(
            service_name=normalized_service,
            capability_mode=capability_mode,
            build_errors=(f"capability registry could not be loaded: {exc}",),
        )

    try:
        server_registry = mcp_gateway.load_mcp_server_registry()
    except Exception as exc:  # noqa: BLE001
        server_registry = mcp_gateway.MCPServerRegistry(servers={})
        build_errors.append(f"MCP server registry could not be loaded: {exc}")

    tool_registry, tool_registry_error = _build_tool_registry_for_service(normalized_service)
    if tool_registry_error:
        build_errors.append(tool_registry_error)

    capabilities: dict[str, RuntimeCapabilityStatus] = {}
    for capability_id, spec in registry.capabilities.items():
        adapter_types = tuple(
            adapter.type
            for adapter in spec.adapters
            if getattr(adapter, "enabled", True)
        )
        details: list[str] = []
        reason = ""
        available = True

        if capability_mode == "disabled":
            available = False
            reason = "capability_mode_disabled"
            details.append(
                f"capability mode is disabled for service '{normalized_service}'"
            )
        elif not spec.enabled:
            available = False
            reason = "capability_disabled"
            details.append(f"capability '{capability_id}' is disabled in the registry")
        else:
            allow = capability_registry.evaluate_capability_allowlist(
                capability_id,
                normalized_service,
            )
            if not allow.allowed:
                available = False
                reason = "capability_governance_denied"
                details.append(
                    f"capability governance denies '{capability_id}' for service "
                    f"'{normalized_service}' ({allow.reason})"
                )
            else:
                adapter_errors = _adapter_conformance_errors(
                    spec=spec,
                    service_name=normalized_service,
                    tool_registry=tool_registry,
                    tool_registry_error=tool_registry_error,
                    server_registry=server_registry,
                )
                if adapter_errors:
                    available = False
                    reason = "adapter_unavailable"
                    details.extend(adapter_errors)

        capabilities[capability_id] = RuntimeCapabilityStatus(
            capability_id=capability_id,
            available=available,
            reason=reason,
            details=tuple(details),
            adapter_types=adapter_types,
        )

    return RuntimeManifest(
        service_name=normalized_service,
        capability_mode=capability_mode,
        runtime_version=os.getenv("TOOL_VERSION", "unknown").strip() or "unknown",
        capabilities=capabilities,
        build_errors=tuple(build_errors),
    )


def explain_capability_unavailability(
    manifest: RuntimeManifest,
    capability_id: str,
) -> str | None:
    canonical = capability_registry.canonicalize_capability_id(capability_id)
    status = manifest.capabilities.get(canonical or capability_id)
    if status is None:
        if manifest.build_errors:
            return "; ".join(manifest.build_errors)
        return f"capability '{capability_id}' is not registered"
    if status.available:
        return None
    parts = list(status.details)
    if manifest.build_errors:
        parts.extend(manifest.build_errors)
    if parts:
        return "; ".join(parts)
    if status.reason:
        return status.reason.replace("_", " ")
    return f"capability '{capability_id}' is not available"


def _adapter_conformance_errors(
    *,
    spec: capability_registry.CapabilitySpec,
    service_name: str,
    tool_registry: ToolRegistry | None,
    tool_registry_error: str | None,
    server_registry: mcp_gateway.MCPServerRegistry,
) -> list[str]:
    adapters = [adapter for adapter in spec.adapters if adapter.enabled]
    if not adapters:
        return [f"capability '{spec.capability_id}' has no enabled adapters"]

    errors: list[str] = []
    for adapter in adapters:
        adapter_type = str(adapter.type or "").strip().lower()
        if adapter_type in {"tool", "native_tool", "local_tool"}:
            tool_name = str(adapter.tool_name or "").strip()
            if not tool_name:
                errors.append(
                    f"capability '{spec.capability_id}' has a tool adapter with no tool name"
                )
                continue
            if tool_registry is None:
                reason = tool_registry_error or "tool registry is unavailable"
                errors.append(
                    f"worker tool adapter '{tool_name}' for capability "
                    f"'{spec.capability_id}' is unavailable: {reason}"
                )
                continue
            try:
                tool_registry.get(tool_name)
            except KeyError:
                errors.append(
                    f"worker tool adapter '{tool_name}' for capability "
                    f"'{spec.capability_id}' is not loaded in service '{service_name}'"
                )
                continue
            return []

        if adapter_type == "mcp":
            server = server_registry.get(adapter.server_id)
            if server is None:
                errors.append(
                    f"MCP server '{adapter.server_id}' for capability "
                    f"'{spec.capability_id}' is not configured"
                )
                continue
            if not server.enabled:
                errors.append(
                    f"MCP server '{adapter.server_id}' for capability "
                    f"'{spec.capability_id}' is disabled"
                )
                continue
            if not str(server.base_url or "").strip():
                errors.append(
                    f"MCP server '{adapter.server_id}' for capability "
                    f"'{spec.capability_id}' has no base URL"
                )
                continue
            if server.bearer_env and not str(os.getenv(server.bearer_env, "")).strip():
                errors.append(
                    f"MCP server '{adapter.server_id}' for capability "
                    f"'{spec.capability_id}' requires env var '{server.bearer_env}'"
                )
                continue
            return []

        errors.append(
            f"capability '{spec.capability_id}' uses unsupported adapter type '{adapter.type}'"
        )
    return errors


def _build_tool_registry_for_service(
    service_name: str,
) -> tuple[ToolRegistry | None, str | None]:
    normalized_service = (service_name or "").strip().lower()
    http_fetch_enabled = False
    llm_enabled = False
    if normalized_service == "worker":
        http_fetch_enabled = os.getenv("TOOL_HTTP_FETCH_ENABLED", "false").lower() == "true"
        llm_enabled = os.getenv("WORKER_MODE", "rule_based").strip().lower() == "llm"
    elif normalized_service == "api":
        llm_enabled = True
    elif normalized_service == "planner":
        llm_enabled = True
    provider = MockLLMProvider() if llm_enabled else None
    try:
        return (
            tool_bootstrap.build_default_registry(
                http_fetch_enabled=http_fetch_enabled,
                llm_enabled=llm_enabled,
                llm_provider=provider,
                service_name=normalized_service or None,
            ),
            None,
        )
    except Exception as exc:  # noqa: BLE001
        return None, f"tool registry could not be loaded for service '{normalized_service}': {exc}"
