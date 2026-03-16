from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from libs.framework.tool_runtime import Tool, ToolRegistry

from .models import RiskLevel, ToolSpec

try:  # Optional at import-time; policy service already depends on PyYAML.
    import yaml
except Exception:  # noqa: BLE001
    yaml = None

LOGGER = logging.getLogger(__name__)


@dataclass
class ToolAllowDecision:
    allowed: bool
    reason: str
    mode: str = "enforce"
    violated: bool = False


@dataclass
class _Ruleset:
    allow: set[str]
    deny: set[str]


@dataclass
class _GovernanceConfig:
    mode: str
    global_rules: _Ruleset
    service_rules: dict[str, _Ruleset]
    tenant_rules: dict[str, _Ruleset]
    job_type_rules: dict[str, _Ruleset]
    blocked_risk_by_service: dict[str, set[str]]


_GOVERNANCE_CACHE_KEY: tuple[str, float] | None = None
_GOVERNANCE_CACHE_VALUE: _GovernanceConfig | None = None


def _parse_csv_values(raw: str | None) -> set[str]:
    if not raw:
        return set()
    return {part.strip() for part in raw.split(",") if part.strip()}


def _tool_governance_enabled() -> bool:
    return os.getenv("TOOL_GOVERNANCE_ENABLED", "true").lower() == "true"


def _tool_governance_mode_env() -> str | None:
    raw = os.getenv("TOOL_GOVERNANCE_MODE", "").strip().lower()
    if raw in {"enforce", "dry_run"}:
        return raw
    return None


def _normalize_rule_key(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"[^A-Za-z0-9]+", "_", value.strip()).lower().strip("_")


def _resolve_tool_governance_config_path() -> Path:
    raw = os.getenv("TOOL_GOVERNANCE_CONFIG_PATH", "config/tool_governance.yaml").strip()
    return Path(raw).expanduser()


def _ruleset_from_raw(raw: Any) -> _Ruleset:
    if not isinstance(raw, dict):
        return _Ruleset(allow=set(), deny=set())
    allow = (
        {str(item).strip() for item in raw.get("allow", []) if str(item).strip()}
        if isinstance(raw.get("allow"), list)
        else set()
    )
    deny = (
        {str(item).strip() for item in raw.get("deny", []) if str(item).strip()}
        if isinstance(raw.get("deny"), list)
        else set()
    )
    return _Ruleset(allow=allow, deny=deny)


def _rules_map_from_raw(raw: Any) -> dict[str, _Ruleset]:
    if not isinstance(raw, dict):
        return {}
    mapped: dict[str, _Ruleset] = {}
    for key, value in raw.items():
        norm_key = _normalize_rule_key(str(key))
        if not norm_key:
            continue
        mapped[norm_key] = _ruleset_from_raw(value)
    return mapped


def _blocked_risk_by_service_from_raw(raw: Any) -> dict[str, set[str]]:
    if not isinstance(raw, dict):
        return {}
    mapped: dict[str, set[str]] = {}
    for key, value in raw.items():
        norm_key = _normalize_rule_key(str(key))
        if not norm_key:
            continue
        if isinstance(value, list):
            mapped[norm_key] = {
                str(entry).strip().lower() for entry in value if str(entry).strip()
            }
    return mapped


def _default_governance_config() -> _GovernanceConfig:
    mode = _tool_governance_mode_env() or "enforce"
    return _GovernanceConfig(
        mode=mode,
        global_rules=_Ruleset(allow=set(), deny=set()),
        service_rules={},
        tenant_rules={},
        job_type_rules={},
        blocked_risk_by_service={},
    )


def _load_governance_config() -> _GovernanceConfig:
    global _GOVERNANCE_CACHE_KEY, _GOVERNANCE_CACHE_VALUE
    config = _default_governance_config()
    if not _tool_governance_enabled():
        config.mode = "enforce"
        return config
    config_path = _resolve_tool_governance_config_path()
    try:
        mtime = config_path.stat().st_mtime
    except OSError:
        mtime = -1.0
    cache_key = (str(config_path), mtime)
    if _GOVERNANCE_CACHE_KEY == cache_key and _GOVERNANCE_CACHE_VALUE is not None:
        return _GOVERNANCE_CACHE_VALUE
    loaded = config
    if mtime >= 0 and yaml is not None:
        try:
            data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("tool_governance_config_load_failed path=%s error=%s", config_path, exc)
            data = {}
        gov = data.get("tool_governance", {}) if isinstance(data, dict) else {}
        if isinstance(gov, dict):
            mode = _tool_governance_mode_env() or str(gov.get("mode", "")).strip().lower() or "enforce"
            if mode not in {"enforce", "dry_run"}:
                mode = "enforce"
            loaded = _GovernanceConfig(
                mode=mode,
                global_rules=_ruleset_from_raw(gov.get("global", {})),
                service_rules=_rules_map_from_raw(gov.get("services", {})),
                tenant_rules=_rules_map_from_raw(gov.get("tenants", {})),
                job_type_rules=_rules_map_from_raw(gov.get("job_types", {})),
                blocked_risk_by_service=_blocked_risk_by_service_from_raw(
                    (gov.get("risk", {}) or {}).get("blocked_levels_by_service", {})
                    if isinstance(gov.get("risk", {}), dict)
                    else {}
                ),
            )
    _GOVERNANCE_CACHE_KEY = cache_key
    _GOVERNANCE_CACHE_VALUE = loaded
    return loaded


def _normalize_service_name(service_name: str | None) -> str:
    if not service_name:
        return ""
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", service_name.strip())
    return normalized.upper().strip("_")


def _service_env_name(service_name: str, suffix: str) -> str:
    return f"{service_name}_{suffix}"


def _context_lookup(context: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = context.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    nested = context.get("job_context")
    if isinstance(nested, dict):
        for key in keys:
            value = nested.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def _resolve_allowlist_sets(
    service_name: str | None, context: dict[str, Any] | None = None
) -> dict[str, Any]:
    gov = _load_governance_config()
    normalized_service_key = _normalize_rule_key(service_name)
    service_rules = gov.service_rules.get(normalized_service_key, _Ruleset(set(), set()))
    context = context if isinstance(context, dict) else {}
    tenant_key = _normalize_rule_key(
        _context_lookup(context, "tenant_id", "org_id", "organization_id")
    )
    job_type_key = _normalize_rule_key(_context_lookup(context, "job_type", "workflow_type"))
    tenant_rules = gov.tenant_rules.get(tenant_key, _Ruleset(set(), set()))
    job_type_rules = gov.job_type_rules.get(job_type_key, _Ruleset(set(), set()))
    normalized_service = _normalize_service_name(service_name)
    global_enabled = _parse_csv_values(os.getenv("ENABLED_TOOLS"))
    global_disabled = _parse_csv_values(os.getenv("DISABLED_TOOLS"))
    service_enabled = set()
    service_disabled = set()
    if normalized_service:
        service_enabled = _parse_csv_values(
            os.getenv(_service_env_name(normalized_service, "ENABLED_TOOLS"))
        )
        service_disabled = _parse_csv_values(
            os.getenv(_service_env_name(normalized_service, "DISABLED_TOOLS"))
        )
    return {
        "mode": gov.mode,
        "global_enabled": global_enabled,
        "global_disabled": global_disabled,
        "service_enabled": service_enabled,
        "service_disabled": service_disabled,
        "config_global_allow": gov.global_rules.allow,
        "config_global_deny": gov.global_rules.deny,
        "config_service_allow": service_rules.allow,
        "config_service_deny": service_rules.deny,
        "config_tenant_allow": tenant_rules.allow,
        "config_tenant_deny": tenant_rules.deny,
        "config_job_type_allow": job_type_rules.allow,
        "config_job_type_deny": job_type_rules.deny,
        "blocked_risk_levels": gov.blocked_risk_by_service.get(normalized_service_key, set()),
    }


def _deny_decision(mode: str, reason: str) -> ToolAllowDecision:
    if mode == "dry_run":
        return ToolAllowDecision(True, f"dry_run:{reason}", mode=mode, violated=True)
    return ToolAllowDecision(False, reason, mode=mode, violated=True)


def evaluate_tool_allowlist(
    tool_name: str,
    service_name: str | None = None,
    *,
    context: dict[str, Any] | None = None,
    tool_spec: ToolSpec | None = None,
) -> ToolAllowDecision:
    sets = _resolve_allowlist_sets(service_name, context=context)
    mode = str(sets.get("mode") or "enforce")

    deny_checks = (
        ("global_disabled", sets["global_disabled"]),
        ("service_disabled", sets["service_disabled"]),
        ("config_global_deny", sets["config_global_deny"]),
        ("config_service_deny", sets["config_service_deny"]),
        ("config_tenant_deny", sets["config_tenant_deny"]),
        ("config_job_type_deny", sets["config_job_type_deny"]),
    )
    for reason, denied_set in deny_checks:
        if denied_set and tool_name in denied_set:
            return _deny_decision(mode, reason)

    blocked_risk_levels = sets.get("blocked_risk_levels", set())
    if blocked_risk_levels and tool_spec is not None:
        risk_level = (
            tool_spec.risk_level.value
            if isinstance(tool_spec.risk_level, RiskLevel)
            else str(tool_spec.risk_level).lower()
        )
        if risk_level in blocked_risk_levels:
            return _deny_decision(mode, f"risk_blocked:{risk_level}")

    allow_checks = (
        ("not_in_global_enabled", sets["global_enabled"]),
        ("not_in_service_enabled", sets["service_enabled"]),
        ("not_in_config_global_allow", sets["config_global_allow"]),
        ("not_in_config_service_allow", sets["config_service_allow"]),
        ("not_in_config_tenant_allow", sets["config_tenant_allow"]),
        ("not_in_config_job_type_allow", sets["config_job_type_allow"]),
    )
    for reason, allowed_set in allow_checks:
        if allowed_set and tool_name not in allowed_set:
            return _deny_decision(mode, reason)
    return ToolAllowDecision(True, "allowed", mode=mode, violated=False)


def filter_registry_tools(registry: ToolRegistry, service_name: str | None = None) -> None:
    sets = _resolve_allowlist_sets(service_name)
    effective_lists = [
        sets.get("global_enabled", set()),
        sets.get("global_disabled", set()),
        sets.get("service_enabled", set()),
        sets.get("service_disabled", set()),
        sets.get("config_global_allow", set()),
        sets.get("config_global_deny", set()),
        sets.get("config_service_allow", set()),
        sets.get("config_service_deny", set()),
        sets.get("blocked_risk_levels", set()),
    ]
    if not any(effective_lists):
        return
    kept: dict[str, Tool] = {}
    for tool_name in list(registry._tools.keys()):
        tool = registry._tools[tool_name]
        if evaluate_tool_allowlist(tool_name, service_name, tool_spec=tool.spec).allowed:
            kept[tool_name] = registry._tools[tool_name]
    registry._tools = kept
