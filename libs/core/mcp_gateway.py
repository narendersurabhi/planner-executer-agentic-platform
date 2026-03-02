from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

try:  # Optional import at runtime.
    import yaml
except Exception:  # noqa: BLE001
    yaml = None

from libs.core import capability_registry as cap_registry
from libs.core import tracing as core_tracing
from libs.framework.tool_runtime import ToolExecutionError, classify_tool_error
from libs.tools import mcp_client


LOGGER = logging.getLogger(__name__)


class MCPGatewayConfigError(ValueError):
    pass


@dataclass(frozen=True)
class MCPServerSpec:
    server_id: str
    base_url: str
    route_paths: tuple[str, ...] = ()
    timeout_s: float | None = None
    max_retries: int | None = None
    retry_sleep_s: float | None = None
    headers: dict[str, str] = field(default_factory=dict)
    bearer_env: str | None = None
    enabled: bool = True


@dataclass(frozen=True)
class MCPServerRegistry:
    servers: dict[str, MCPServerSpec]

    def get(self, server_id: str) -> MCPServerSpec | None:
        return self.servers.get(server_id)

    def require(self, server_id: str) -> MCPServerSpec:
        spec = self.get(server_id)
        if spec is None:
            raise MCPGatewayConfigError(f"mcp_server_not_found:{server_id}")
        return spec


_SERVER_CACHE_KEY: tuple[str, float] | None = None
_SERVER_CACHE_VALUE: MCPServerRegistry | None = None


def resolve_mcp_servers_config_path() -> Path:
    raw = os.getenv("MCP_SERVERS_CONFIG_PATH", "config/mcp_servers.yaml").strip()
    return Path(raw).expanduser()


def load_mcp_server_registry(path: Path | None = None) -> MCPServerRegistry:
    global _SERVER_CACHE_KEY, _SERVER_CACHE_VALUE

    resolved = (path or resolve_mcp_servers_config_path()).expanduser()
    try:
        mtime = resolved.stat().st_mtime
    except OSError:
        mtime = -1.0
    cache_key = (str(resolved), mtime)
    if _SERVER_CACHE_KEY == cache_key and _SERVER_CACHE_VALUE is not None:
        return _SERVER_CACHE_VALUE
    if mtime < 0:
        registry = MCPServerRegistry(servers={})
        _SERVER_CACHE_KEY = cache_key
        _SERVER_CACHE_VALUE = registry
        return registry
    data = _read_config(resolved)
    registry = _parse_server_registry(data)
    _SERVER_CACHE_KEY = cache_key
    _SERVER_CACHE_VALUE = registry
    return registry


def invoke_capability(
    capability_id: str,
    arguments: dict[str, Any],
    *,
    capability_registry: cap_registry.CapabilityRegistry | None = None,
    server_registry: MCPServerRegistry | None = None,
    post_tool_call: Callable[..., dict[str, Any]] | None = None,
    execute_tool: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    registry = capability_registry or cap_registry.load_capability_registry()
    servers = server_registry or load_mcp_server_registry()
    spec = registry.require(capability_id)
    if not spec.enabled:
        raise ToolExecutionError(f"capability_disabled:{capability_id}")
    adapters = [adapter for adapter in spec.adapters if adapter.enabled]
    if not adapters:
        raise ToolExecutionError(f"capability_adapter_missing:{capability_id}")

    runner = post_tool_call or _post_mcp_tool_call
    errors: list[str] = []
    for adapter in adapters:
        mapped_args = _map_arguments(arguments, adapter.arg_map)
        adapter_type = adapter.type.strip().lower()

        if adapter_type in {"tool", "native_tool"}:
            if execute_tool is None:
                errors.append(f"{adapter.server_id}:tool_executor_missing")
                continue
            try:
                response = execute_tool(adapter.tool_name, mapped_args)
                normalized = response if isinstance(response, dict) else {"result": response}
                return _map_response(normalized, adapter.response_path)
            except ToolExecutionError as exc:
                errors.append(f"{adapter.server_id}:{exc}")
                continue
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{adapter.server_id}:tool_execution_failed:{exc}")
                continue

        if adapter_type != "mcp":
            errors.append(f"{adapter.server_id}:unsupported_adapter_type:{adapter.type}")
            continue

        server = servers.get(adapter.server_id)
        if server is None:
            errors.append(f"{adapter.server_id}:server_not_found")
            continue
        if not server.enabled:
            errors.append(f"{adapter.server_id}:server_disabled")
            continue
        headers = _resolve_server_headers(server)
        route_paths = adapter.route_paths or server.route_paths or ()
        try:
            response = runner(
                server.base_url,
                adapter.tool_name,
                mapped_args,
                route_paths=route_paths or None,
                headers=headers or None,
                timeout_s_override=adapter.timeout_s or server.timeout_s,
                max_retries_override=adapter.max_retries
                if adapter.max_retries is not None
                else server.max_retries,
                retry_sleep_s_override=adapter.retry_sleep_s
                if adapter.retry_sleep_s is not None
                else server.retry_sleep_s,
            )
            return _map_response(response, adapter.response_path)
        except ToolExecutionError as exc:
            errors.append(f"{adapter.server_id}:{exc}")
            continue
    raise ToolExecutionError(f"mcp_gateway_all_adapters_failed:{' | '.join(errors)}")


def _post_mcp_tool_call(
    service_url: str,
    tool_name: str,
    arguments: dict[str, Any],
    *,
    route_paths: tuple[str, ...] | list[str] | None = None,
    headers: dict[str, str] | None = None,
    timeout_s_override: float | None = None,
    max_retries_override: int | None = None,
    retry_sleep_s_override: float | None = None,
) -> dict[str, Any]:
    return mcp_client.post_mcp_tool_call(
        service_url=service_url,
        tool_name=tool_name,
        arguments=arguments,
        route_paths=route_paths,
        headers=headers,
        timeout_s_override=timeout_s_override,
        max_retries_override=max_retries_override,
        retry_sleep_s_override=retry_sleep_s_override,
        call_mcp_tool_sdk=_call_mcp_tool_sdk,
        classify_tool_error=classify_tool_error,
        logger=LOGGER,
        tracing_module=core_tracing,
    )


def _call_mcp_tool_sdk(
    mcp_url: str,
    tool_name: str,
    arguments: dict[str, Any],
    timeout_s: float,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    return mcp_client.call_mcp_tool_sdk(
        mcp_url=mcp_url,
        tool_name=tool_name,
        arguments=arguments,
        timeout_s=timeout_s,
        headers=headers,
        tracing_module=core_tracing,
        logger=LOGGER,
    )


def _map_arguments(arguments: dict[str, Any], arg_map: dict[str, Any]) -> dict[str, Any]:
    if not arg_map:
        return dict(arguments)
    mapped: dict[str, Any] = {}
    for target, source in arg_map.items():
        if isinstance(source, str) and source in arguments:
            mapped[target] = arguments[source]
        else:
            mapped[target] = source
    return mapped


def _map_response(response: dict[str, Any], response_path: str | None) -> dict[str, Any]:
    if not response_path:
        return response
    current: Any = response
    for token in response_path.split("."):
        key = token.strip()
        if not key:
            continue
        if not isinstance(current, dict) or key not in current:
            raise ToolExecutionError(f"mcp_gateway_response_path_missing:{response_path}")
        current = current[key]
    if isinstance(current, dict):
        return current
    return {"result": current}


def _resolve_server_headers(server: MCPServerSpec) -> dict[str, str]:
    headers = dict(server.headers)
    if server.bearer_env:
        token = os.getenv(server.bearer_env, "").strip()
        if token:
            headers.setdefault("Authorization", f"Bearer {token}")
    return headers


def _read_config(path: Path) -> dict[str, Any]:
    content = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise MCPGatewayConfigError("PyYAML is required to load YAML MCP servers config")
        parsed = yaml.safe_load(content) or {}
    else:
        parsed = json.loads(content)
    if not isinstance(parsed, dict):
        raise MCPGatewayConfigError("MCP server config root must be an object")
    return parsed


def _parse_server_registry(raw: dict[str, Any]) -> MCPServerRegistry:
    raw_servers = raw.get("mcp_servers", {})
    if not isinstance(raw_servers, dict):
        raise MCPGatewayConfigError("mcp_servers must be an object")
    parsed: dict[str, MCPServerSpec] = {}
    for key, value in raw_servers.items():
        if not isinstance(value, dict):
            raise MCPGatewayConfigError(f"mcp_servers.{key} must be an object")
        server_id = str(key).strip()
        if not server_id:
            continue
        if server_id in parsed:
            raise MCPGatewayConfigError(f"duplicate_server_id:{server_id}")
        parsed[server_id] = _parse_server(server_id, value)
    return MCPServerRegistry(servers=parsed)


def _parse_server(server_id: str, raw: dict[str, Any]) -> MCPServerSpec:
    base_url = _required_str(raw, "base_url", f"mcp_servers.{server_id}.base_url")
    route_paths = _parse_route_paths(
        raw.get("route_paths", []), f"mcp_servers.{server_id}.route_paths"
    )
    timeout_s = _optional_float(raw.get("timeout_s"), f"mcp_servers.{server_id}.timeout_s")
    max_retries = _optional_int(
        raw.get("max_retries"), f"mcp_servers.{server_id}.max_retries"
    )
    retry_sleep_s = _optional_float(
        raw.get("retry_sleep_s"), f"mcp_servers.{server_id}.retry_sleep_s"
    )
    headers = _parse_headers(raw.get("headers"), f"mcp_servers.{server_id}.headers")
    bearer_env = _optional_str(raw.get("bearer_env"), None)
    enabled = bool(raw.get("enabled", True))
    return MCPServerSpec(
        server_id=server_id,
        base_url=base_url.rstrip("/"),
        route_paths=route_paths,
        timeout_s=timeout_s,
        max_retries=max_retries,
        retry_sleep_s=retry_sleep_s,
        headers=headers,
        bearer_env=bearer_env,
        enabled=enabled,
    )


def _required_str(raw: dict[str, Any], key: str, field_name: str) -> str:
    value = raw.get(key)
    if isinstance(value, str) and value.strip():
        return value.strip()
    raise MCPGatewayConfigError(f"missing_required_field:{field_name}")


def _optional_str(value: Any, default: str | None) -> str | None:
    if value is None:
        return default
    if not isinstance(value, str):
        return default
    stripped = value.strip()
    return stripped or default


def _optional_float(value: Any, field_name: str) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise MCPGatewayConfigError(f"invalid_float:{field_name}") from exc


def _optional_int(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise MCPGatewayConfigError(f"invalid_int:{field_name}") from exc


def _parse_route_paths(value: Any, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise MCPGatewayConfigError(f"{field_name} must be a list")
    seen: set[str] = set()
    paths: list[str] = []
    for idx, item in enumerate(value):
        if not isinstance(item, str):
            raise MCPGatewayConfigError(f"{field_name}[{idx}] must be a string")
        normalized = item.strip()
        if not normalized:
            continue
        if normalized != "/" and not normalized.startswith("/"):
            normalized = f"/{normalized}"
        if normalized in seen:
            continue
        seen.add(normalized)
        paths.append(normalized)
    return tuple(paths)


def _parse_headers(value: Any, field_name: str) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise MCPGatewayConfigError(f"{field_name} must be an object")
    headers: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key.strip():
            raise MCPGatewayConfigError(f"{field_name} has an invalid key")
        if not isinstance(item, str):
            raise MCPGatewayConfigError(f"{field_name}.{key} must be a string")
        headers[key.strip()] = item
    return headers
