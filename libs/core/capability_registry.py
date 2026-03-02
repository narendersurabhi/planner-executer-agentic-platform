from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:  # Optional import at runtime.
    import yaml
except Exception:  # noqa: BLE001
    yaml = None


LOGGER = logging.getLogger(__name__)


class CapabilityRegistryError(ValueError):
    pass


@dataclass(frozen=True)
class CapabilityAdapterSpec:
    type: str
    server_id: str
    tool_name: str
    route_paths: tuple[str, ...] = ()
    timeout_s: float | None = None
    max_retries: int | None = None
    retry_sleep_s: float | None = None
    arg_map: dict[str, Any] = field(default_factory=dict)
    response_path: str | None = None
    enabled: bool = True


@dataclass(frozen=True)
class CapabilityAllowDecision:
    allowed: bool
    reason: str
    mode: str = "enforce"
    violated: bool = False


@dataclass(frozen=True)
class CapabilitySpec:
    capability_id: str
    description: str
    risk_tier: str
    idempotency: str
    group: str | None = None
    subgroup: str | None = None
    input_schema_ref: str | None = None
    output_schema_ref: str | None = None
    adapters: tuple[CapabilityAdapterSpec, ...] = ()
    tags: tuple[str, ...] = ()
    planner_hints: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True


@dataclass(frozen=True)
class CapabilityRegistry:
    capabilities: dict[str, CapabilitySpec]

    def get(self, capability_id: str) -> CapabilitySpec | None:
        return self.capabilities.get(capability_id)

    def require(self, capability_id: str) -> CapabilitySpec:
        spec = self.get(capability_id)
        if spec is None:
            raise CapabilityRegistryError(f"capability_not_found:{capability_id}")
        return spec

    def enabled_capabilities(self) -> dict[str, CapabilitySpec]:
        return {cap_id: spec for cap_id, spec in self.capabilities.items() if spec.enabled}


_CAPABILITY_CACHE_KEY: tuple[str, float] | None = None
_CAPABILITY_CACHE_VALUE: CapabilityRegistry | None = None


def resolve_capability_mode() -> str:
    mode = os.getenv("CAPABILITY_MODE", "disabled").strip().lower()
    if mode in {"disabled", "dry_run", "enabled"}:
        return mode
    return "disabled"


def resolve_capability_registry_path() -> Path:
    raw = os.getenv("CAPABILITY_REGISTRY_PATH", "config/capability_registry.yaml").strip()
    return Path(raw).expanduser()


def _capability_governance_enabled() -> bool:
    return os.getenv("CAPABILITY_GOVERNANCE_ENABLED", "true").strip().lower() == "true"


def _capability_governance_mode() -> str:
    raw = os.getenv("CAPABILITY_GOVERNANCE_MODE", "enforce").strip().lower()
    if raw in {"enforce", "dry_run"}:
        return raw
    return "enforce"


def _parse_csv_values(raw: str | None) -> set[str]:
    if not raw:
        return set()
    return {part.strip() for part in raw.split(",") if part.strip()}


def _normalize_service_name(service_name: str | None) -> str:
    if not service_name:
        return ""
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", service_name.strip())
    return normalized.upper().strip("_")


def _service_env_name(service_name: str, suffix: str) -> str:
    return f"{service_name}_{suffix}"


def _deny_decision(mode: str, reason: str) -> CapabilityAllowDecision:
    if mode == "dry_run":
        return CapabilityAllowDecision(True, f"dry_run:{reason}", mode=mode, violated=True)
    return CapabilityAllowDecision(False, reason, mode=mode, violated=True)


def evaluate_capability_allowlist(
    capability_id: str,
    service_name: str | None = None,
) -> CapabilityAllowDecision:
    if not capability_id:
        return CapabilityAllowDecision(False, "missing_capability_id", mode="enforce", violated=True)
    if not _capability_governance_enabled():
        return CapabilityAllowDecision(True, "governance_disabled", mode="enforce", violated=False)

    mode = _capability_governance_mode()
    disabled_capabilities = _parse_csv_values(os.getenv("DISABLED_CAPABILITIES"))
    enabled_capabilities = _parse_csv_values(os.getenv("ENABLED_CAPABILITIES"))

    normalized_service = _normalize_service_name(service_name)
    service_disabled: set[str] = set()
    service_enabled: set[str] = set()
    if normalized_service:
        service_disabled = _parse_csv_values(
            os.getenv(_service_env_name(normalized_service, "DISABLED_CAPABILITIES"))
        )
        service_enabled = _parse_csv_values(
            os.getenv(_service_env_name(normalized_service, "ENABLED_CAPABILITIES"))
        )

    deny_checks = (
        ("global_disabled", disabled_capabilities),
        ("service_disabled", service_disabled),
    )
    for reason, denied_set in deny_checks:
        if denied_set and capability_id in denied_set:
            return _deny_decision(mode, reason)

    allow_checks = (
        ("not_in_global_enabled", enabled_capabilities),
        ("not_in_service_enabled", service_enabled),
    )
    for reason, allowed_set in allow_checks:
        if allowed_set and capability_id not in allowed_set:
            return _deny_decision(mode, reason)
    return CapabilityAllowDecision(True, "allowed", mode=mode, violated=False)


def load_capability_registry(path: Path | None = None) -> CapabilityRegistry:
    global _CAPABILITY_CACHE_KEY, _CAPABILITY_CACHE_VALUE

    resolved = (path or resolve_capability_registry_path()).expanduser()
    try:
        mtime = resolved.stat().st_mtime
    except OSError:
        mtime = -1.0
    cache_key = (str(resolved), mtime)
    if _CAPABILITY_CACHE_KEY == cache_key and _CAPABILITY_CACHE_VALUE is not None:
        return _CAPABILITY_CACHE_VALUE
    if mtime < 0:
        registry = CapabilityRegistry(capabilities={})
        _CAPABILITY_CACHE_KEY = cache_key
        _CAPABILITY_CACHE_VALUE = registry
        return registry
    data = _read_registry_file(resolved)
    registry = _parse_capability_registry(data)
    _CAPABILITY_CACHE_KEY = cache_key
    _CAPABILITY_CACHE_VALUE = registry
    return registry


def _read_registry_file(path: Path) -> dict[str, Any]:
    content = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise CapabilityRegistryError("PyYAML is required to load YAML capability registry")
        parsed = yaml.safe_load(content) or {}
    else:
        parsed = json.loads(content)
    if not isinstance(parsed, dict):
        raise CapabilityRegistryError("Capability registry root must be an object")
    return parsed


def _parse_capability_registry(data: dict[str, Any]) -> CapabilityRegistry:
    raw_caps = data.get("capabilities", [])
    if not isinstance(raw_caps, list):
        raise CapabilityRegistryError("capabilities must be a list")
    parsed: dict[str, CapabilitySpec] = {}
    for idx, raw in enumerate(raw_caps):
        if not isinstance(raw, dict):
            raise CapabilityRegistryError(f"capabilities[{idx}] must be an object")
        spec = _parse_capability(raw, idx)
        if spec.capability_id in parsed:
            raise CapabilityRegistryError(f"duplicate_capability_id:{spec.capability_id}")
        parsed[spec.capability_id] = spec
    return CapabilityRegistry(capabilities=parsed)


def _parse_capability(raw: dict[str, Any], idx: int) -> CapabilitySpec:
    capability_id = _required_str(raw, ("id", "capability_id"), f"capabilities[{idx}].id")
    description = _required_str(raw, ("description",), f"capabilities[{idx}].description")
    risk_tier = _optional_str(raw.get("risk_tier"), "read_only")
    idempotency = _optional_str(raw.get("idempotency"), "read")
    group = _optional_str(raw.get("group"), None)
    subgroup = _optional_str(raw.get("subgroup"), None)
    input_schema_ref = _optional_str(raw.get("input_schema_ref"), None)
    output_schema_ref = _optional_str(raw.get("output_schema_ref"), None)
    tags = _parse_string_list(raw.get("tags"), field_name=f"capabilities[{idx}].tags")
    planner_hints = raw.get("planner_hints", {})
    if not isinstance(planner_hints, dict):
        raise CapabilityRegistryError(f"capabilities[{idx}].planner_hints must be an object")
    enabled = bool(raw.get("enabled", True))

    adapters_raw = raw.get("adapters", [])
    if not isinstance(adapters_raw, list):
        raise CapabilityRegistryError(f"capabilities[{idx}].adapters must be a list")
    adapters = tuple(
        _parse_adapter(adapter_raw, idx, adapter_idx)
        for adapter_idx, adapter_raw in enumerate(adapters_raw)
    )
    return CapabilitySpec(
        capability_id=capability_id,
        description=description,
        risk_tier=risk_tier,
        idempotency=idempotency,
        group=group,
        subgroup=subgroup,
        input_schema_ref=input_schema_ref,
        output_schema_ref=output_schema_ref,
        adapters=adapters,
        tags=tags,
        planner_hints=dict(planner_hints),
        enabled=enabled,
    )


def _parse_adapter(raw: Any, capability_idx: int, adapter_idx: int) -> CapabilityAdapterSpec:
    field_prefix = f"capabilities[{capability_idx}].adapters[{adapter_idx}]"
    if not isinstance(raw, dict):
        raise CapabilityRegistryError(f"{field_prefix} must be an object")
    adapter_type = _optional_str(raw.get("type"), "mcp")
    server_id = _required_str(raw, ("server_id",), f"{field_prefix}.server_id")
    tool_name = _required_str(raw, ("tool_name",), f"{field_prefix}.tool_name")
    route_paths = _parse_route_paths(raw.get("route_paths"), f"{field_prefix}.route_paths")
    timeout_s = _optional_float(raw.get("timeout_s"), f"{field_prefix}.timeout_s")
    max_retries = _optional_int(raw.get("max_retries"), f"{field_prefix}.max_retries")
    retry_sleep_s = _optional_float(raw.get("retry_sleep_s"), f"{field_prefix}.retry_sleep_s")
    arg_map = raw.get("arg_map", {})
    if not isinstance(arg_map, dict):
        raise CapabilityRegistryError(f"{field_prefix}.arg_map must be an object")
    response_path = _optional_str(raw.get("response_path"), None)
    enabled = bool(raw.get("enabled", True))
    return CapabilityAdapterSpec(
        type=adapter_type,
        server_id=server_id,
        tool_name=tool_name,
        route_paths=route_paths,
        timeout_s=timeout_s,
        max_retries=max_retries,
        retry_sleep_s=retry_sleep_s,
        arg_map=arg_map,
        response_path=response_path,
        enabled=enabled,
    )


def _required_str(raw: dict[str, Any], keys: tuple[str, ...], field_name: str) -> str:
    for key in keys:
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise CapabilityRegistryError(f"missing_required_field:{field_name}")


def _optional_str(value: Any, default: str | None) -> str | None:
    if value is None:
        return default
    if not isinstance(value, str):
        return default
    stripped = value.strip()
    if not stripped:
        return default
    return stripped


def _optional_float(value: Any, field_name: str) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise CapabilityRegistryError(f"invalid_float:{field_name}") from exc


def _optional_int(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise CapabilityRegistryError(f"invalid_int:{field_name}") from exc


def _parse_route_paths(value: Any, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise CapabilityRegistryError(f"{field_name} must be a list")
    paths: list[str] = []
    seen: set[str] = set()
    for idx, item in enumerate(value):
        if not isinstance(item, str):
            raise CapabilityRegistryError(f"{field_name}[{idx}] must be a string")
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


def _parse_string_list(value: Any, *, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise CapabilityRegistryError(f"{field_name} must be a list")
    out: list[str] = []
    for idx, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise CapabilityRegistryError(f"{field_name}[{idx}] must be a non-empty string")
        out.append(item.strip())
    return tuple(out)
