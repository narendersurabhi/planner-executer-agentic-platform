from __future__ import annotations

import json
import logging
import math
import os
import re
import shutil
import inspect
import time
from datetime import UTC, datetime
from subprocess import CompletedProcess, run
from importlib import import_module
from importlib import metadata as importlib_metadata
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urlparse

from .llm_provider import LLMProvider
from . import prompts, tracing as core_tracing
from libs.core import mcp_gateway
from .models import RiskLevel, ToolIntent, ToolSpec
from libs.framework.tool_runtime import (
    Tool,
    ToolExecutionError,
    ToolRegistry,
    classify_tool_error as _classify_tool_error,
    validate_schema as _validate_schema,
)
from libs.core.memory_client import MemoryClient, MemoryClientError
from libs.tools.docx_generate_from_spec import register_docx_tools
from libs.tools.pdf_generate_from_spec import register_pdf_tools
from libs.tools.document_spec_validate import register_document_spec_tools
from libs.tools.document_spec_iterative import register_document_spec_iterative_tools
from libs.tools.document_spec_llm import (
    llm_generate_document_spec as _llm_generate_document_spec_external,
    llm_improve_document_spec as _llm_improve_document_spec_external,
    register_document_spec_llm_tools,
)
from libs.tools.core_ops import CoreOpsHandlers, register_core_ops_tools
from libs.tools.llm_tool_groups import (
    register_coding_agent_tools,
    register_llm_text_tool,
)
from libs.tools import mcp_client
from libs.tools import coder_tools
from libs.tools.github_tools import register_github_tools
from libs.tools.openapi_iterative import register_openapi_iterative_tools

try:  # Optional at import-time; policy service already depends on PyYAML.
    import yaml
except Exception:  # noqa: BLE001
    yaml = None

LOGGER = logging.getLogger(__name__)


class ToolPluginLoadError(ToolExecutionError):
    pass


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


def _extract_mcp_error_phase(error_text: str) -> str | None:
    return mcp_client.extract_mcp_error_phase(error_text)


def _safe_artifact_path(path: str, default_name: str) -> Path:
    base_dir = Path("/shared/artifacts")
    base_dir.mkdir(parents=True, exist_ok=True)
    candidate = Path(path or default_name)
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        resolved = (base_dir / candidate).resolve()
    if not str(resolved).startswith(str(base_dir.resolve())):
        raise ToolExecutionError("Invalid path outside /shared/artifacts")
    return resolved


def _workspace_root() -> Path:
    env_root = os.getenv("WORKSPACE_DIR")
    if env_root:
        return Path(env_root).resolve()
    return Path(__file__).resolve().parents[2]


def _safe_workspace_path(path: str, default_name: str) -> Path:
    base_dir = _workspace_root()
    base_dir.mkdir(parents=True, exist_ok=True)
    candidate = Path(path or default_name)
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        resolved = (base_dir / candidate).resolve()
    if not str(resolved).startswith(str(base_dir.resolve())):
        raise ToolExecutionError("Invalid path outside workspace")
    return resolved


def _resolve_schema_path(schema_ref: str) -> Path:
    registry_dir = Path(os.getenv("SCHEMA_REGISTRY_PATH", "/app/schemas"))
    template_dir = Path(os.getenv("DOCX_TEMPLATE_DIR", "/shared/templates"))
    name = schema_ref
    if schema_ref.startswith("schema/"):
        name = schema_ref.split("/", 1)[1]
    if not name.endswith(".json"):
        name = f"{name}.json"
    candidate = registry_dir / name
    if candidate.exists():
        return candidate
    return template_dir / name


def _validate_schema_from_registry(schema_ref: str, payload: Dict[str, Any]) -> None:
    schema_path = _resolve_schema_path(schema_ref)
    if not schema_path.exists():
        raise ToolExecutionError(f"schema_not_found:{schema_path}")
    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise ToolExecutionError(f"invalid_schema:{exc}") from exc
    _validate_schema(schema, payload, "input")


def _parse_http_allowlist() -> List[str]:
    raw = os.getenv("TOOL_HTTP_FETCH_ALLOWLIST", "")
    return [entry.strip() for entry in raw.split(",") if entry.strip()]


def _resolve_llm_timeout_s(provider: Optional[LLMProvider]) -> int:
    if provider is not None and hasattr(provider, "timeout_s"):
        try:
            return max(1, int(math.ceil(getattr(provider, "timeout_s"))))
        except (TypeError, ValueError):
            pass
    env_timeout = os.getenv("OPENAI_TIMEOUT_S")
    if env_timeout:
        try:
            return max(1, int(math.ceil(float(env_timeout))))
        except ValueError:
            return 30
    return 30


def _resolve_coding_agent_timeout_s() -> int:
    env_timeout = os.getenv("CODING_AGENT_TIMEOUT_S")
    if env_timeout:
        try:
            return max(1, int(math.ceil(float(env_timeout))))
        except ValueError:
            return 30
    return 30


def _resolve_coder_http_timeout_s() -> int:
    for key in ("CODER_HTTP_TIMEOUT_S", "CODING_AGENT_TIMEOUT_S"):
        env_timeout = os.getenv(key)
        if env_timeout:
            try:
                return max(1, int(math.ceil(float(env_timeout))))
            except ValueError:
                return 30
    return 30


def _resolve_mcp_timeout_s() -> float:
    return mcp_client.resolve_mcp_timeout_s()


def _resolve_mcp_outer_timeout_headroom_s() -> float:
    return mcp_client.resolve_mcp_outer_timeout_headroom_s()


def _resolve_mcp_tool_timeout_s() -> int:
    return mcp_client.resolve_mcp_tool_timeout_s()


def _resolve_mcp_max_retries() -> int:
    return mcp_client.resolve_mcp_max_retries()


def _resolve_mcp_retry_sleep_s() -> float:
    return mcp_client.resolve_mcp_retry_sleep_s()


def _resolve_mcp_first_attempt_reserve_s(timeout_s: float) -> float:
    return mcp_client.resolve_mcp_first_attempt_reserve_s(timeout_s)


def _resolve_mcp_transport_timeout_s(timeout_s: float) -> float:
    return mcp_client.resolve_mcp_transport_timeout_s(timeout_s)


def _resolve_mcp_isolation_mode() -> str:
    return mcp_client.resolve_mcp_isolation_mode()


def _streamable_http_client_kwargs(
    client_factory: Callable[..., Any], timeout_s: float
) -> Dict[str, Any]:
    return mcp_client.streamable_http_client_kwargs(client_factory, timeout_s)


def _is_retryable_mcp_error(message: str) -> bool:
    return mcp_client.is_retryable_mcp_error(message)


def _host_allowed(host: str, allowlist: List[str]) -> bool:
    if not allowlist:
        return False
    if "*" in allowlist:
        return True
    for entry in allowlist:
        if entry.startswith("*."):
            suffix = entry[1:]
            if host.endswith(suffix):
                return True
        elif entry.startswith("."):
            if host.endswith(entry):
                return True
        elif host == entry:
            return True
    return False


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


def _plugin_fail_fast() -> bool:
    return os.getenv("TOOL_PLUGIN_FAIL_FAST", "true").lower() == "true"


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


def _tool_enabled(
    name: str, service_name: str | None = None, tool_spec: ToolSpec | None = None
) -> bool:
    return evaluate_tool_allowlist(name, service_name, tool_spec=tool_spec).allowed


def _filter_registry_tools(registry: ToolRegistry, service_name: str | None = None) -> None:
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
        if _tool_enabled(tool_name, service_name, tool.spec):
            kept[tool_name] = registry._tools[tool_name]
    registry._tools = kept


def _parse_plugin_spec(spec: str) -> tuple[str, str]:
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


def _resolve_module_register_fn(module: Any, attr_name: str) -> Callable[..., None]:
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


def _call_register_fn(
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


def _load_module_plugins(
    registry: ToolRegistry,
    *,
    llm_enabled: bool,
    llm_provider: Optional[LLMProvider],
    http_fetch_enabled: bool,
) -> None:
    plugin_specs = _parse_csv_values(os.getenv("TOOL_PLUGIN_MODULES"))
    for plugin_spec in sorted(plugin_specs):
        try:
            module_path, attr_name = _parse_plugin_spec(plugin_spec)
            module = import_module(module_path)
            register_fn = _resolve_module_register_fn(module, attr_name)
            _call_register_fn(
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


def _iter_entry_points(group: str) -> list[Any]:
    eps = importlib_metadata.entry_points()
    if hasattr(eps, "select"):
        return list(eps.select(group=group))
    return list(eps.get(group, []))


def _load_entrypoint_plugins(
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
                register_fn = _resolve_module_register_fn(loaded, "register_tools")
            _call_register_fn(
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


def default_registry(
    http_fetch_enabled: bool = False,
    llm_enabled: bool = False,
    llm_provider: Optional[LLMProvider] = None,
    service_name: Optional[str] = None,
) -> ToolRegistry:
    registry = ToolRegistry()

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
        handlers=CoreOpsHandlers(
            math_eval=_math_eval,
            text_summarize=_text_summarize,
            file_write_artifact=lambda payload: _write_text_file(payload, default_filename="artifact.txt"),
            file_write_text=lambda payload: _write_text_file(payload, default_filename="output.txt"),
            file_write_code=_file_write_code,
            file_read_text=_file_read_text,
            list_files=_list_files,
            workspace_write_text=lambda payload: _write_workspace_text_file(
                payload, default_filename="output.txt"
            ),
            workspace_write_code=_workspace_write_code,
            workspace_read_text=_workspace_read_text,
            workspace_list_files=_list_workspace_files,
            artifact_mkdir=_artifact_mkdir,
            workspace_mkdir=_workspace_mkdir,
            artifact_delete=_artifact_delete,
            workspace_delete=_workspace_delete,
            artifact_rename=_artifact_rename,
            workspace_rename=_workspace_rename,
            artifact_copy=_artifact_copy,
            workspace_copy=_workspace_copy,
            artifact_move=_artifact_move_to_workspace,
            derive_output_path=_derive_output_path,
            derive_output_filename=_derive_output_filename,
            run_tests=_run_tests,
            search_text=_search_text,
            memory_read=_memory_read,
            memory_write=_memory_write,
            memory_semantic_write=_memory_semantic_write,
            memory_semantic_search=_memory_semantic_search,
            docx_render=_docx_render,
            sleep=_sleep,
            http_fetch=_http_fetch,
        ),
        http_fetch_enabled=http_fetch_enabled,
    )

    register_docx_tools(registry)
    register_pdf_tools(registry)
    register_document_spec_tools(registry)
    register_github_tools(registry)

    if llm_enabled:
        if llm_provider is None:
            raise ValueError("llm_enabled requires a llm_provider instance")
        llm_timeout_s = _resolve_llm_timeout_s(llm_provider)
        coding_agent_timeout_s = _resolve_coding_agent_timeout_s()
        mcp_timeout_s = _resolve_mcp_tool_timeout_s()
        llm_iterative_timeout_s = _resolve_llm_iterative_tool_timeout_s(llm_provider)

        register_llm_text_tool(
            registry,
            timeout_s=llm_timeout_s,
            handler=lambda payload, provider=llm_provider: _llm_generate(payload, provider),
        )
        register_coding_agent_tools(
            registry,
            timeout_s=coding_agent_timeout_s,
            handler_generate=_coding_agent_generate,
            handler_autonomous=lambda payload, provider=llm_provider: _coding_agent_autonomous(
                payload, provider
            ),
            handler_publish_pr=_coding_agent_publish_pr,
        )
        register_document_spec_llm_tools(
            registry,
            llm_provider,
            timeout_s=llm_timeout_s,
            sanitize_document_spec=_sanitize_document_spec,
        )
        register_document_spec_iterative_tools(
            registry,
            llm_provider,
            timeout_s=llm_iterative_timeout_s,
            generate_document_spec=_llm_generate_document_spec,
            improve_document_spec=_llm_improve_document_spec,
            sanitize_document_spec=_sanitize_document_spec,
        )
        register_openapi_iterative_tools(
            registry,
            llm_provider,
            timeout_s=llm_iterative_timeout_s,
        )

    _load_module_plugins(
        registry,
        llm_enabled=llm_enabled,
        llm_provider=llm_provider,
        http_fetch_enabled=http_fetch_enabled,
    )
    _load_entrypoint_plugins(
        registry,
        llm_enabled=llm_enabled,
        llm_provider=llm_provider,
        http_fetch_enabled=http_fetch_enabled,
    )
    _filter_registry_tools(registry, service_name)

    return registry


def _math_eval(payload: Dict[str, Any]) -> Dict[str, Any]:
    expr = payload.get("expr", "0")
    allowed = {"sqrt": math.sqrt, "pow": pow}
    value = eval(expr, {"__builtins__": {}}, allowed)  # noqa: S307
    return {"value": value}


def _text_summarize(payload: Dict[str, Any]) -> Dict[str, Any]:
    text = payload.get("text", "")
    summary = text[:200]
    return {"summary": summary}


def _write_text_file(payload: Dict[str, Any], default_filename: str) -> Dict[str, Any]:
    path = payload.get("path", "")
    content = payload.get("content", "")
    if path and path.endswith("/"):
        raise ToolExecutionError("Missing file name in path")
    candidate = _safe_artifact_path(path, default_filename)
    candidate.parent.mkdir(parents=True, exist_ok=True)
    candidate.write_text(content, encoding="utf-8")
    return {"path": str(candidate)}


def _write_workspace_text_file(payload: Dict[str, Any], default_filename: str) -> Dict[str, Any]:
    path = payload.get("path", "")
    content = payload.get("content", "")
    if path and path.endswith("/"):
        raise ToolExecutionError("Missing file name in path")
    candidate = _safe_workspace_path(path, default_filename)
    candidate.parent.mkdir(parents=True, exist_ok=True)
    candidate.write_text(content, encoding="utf-8")
    return {"path": str(candidate)}


def _file_write_code(payload: Dict[str, Any]) -> Dict[str, Any]:
    path = payload.get("path", "")
    if not path:
        raise ToolExecutionError("Missing file name in path")
    _ensure_code_extension(path)
    return _write_text_file(payload, default_filename="output.txt")


def _workspace_write_code(payload: Dict[str, Any]) -> Dict[str, Any]:
    path = payload.get("path", "")
    if not path:
        raise ToolExecutionError("Missing file name in path")
    _ensure_code_extension(path)
    return _write_workspace_text_file(payload, default_filename="output.txt")


def _ensure_code_extension(path: str) -> None:
    code_extensions = {
        ".py",
        ".js",
        ".ts",
        ".tsx",
        ".jsx",
        ".html",
        ".css",
        ".json",
        ".md",
        ".yml",
        ".yaml",
        ".toml",
        ".sh",
        ".sql",
        ".txt",
    }
    suffix = Path(path).suffix.lower()
    if not suffix or suffix not in code_extensions:
        raise ToolExecutionError("Unsupported code file extension")


def _file_read_text(payload: Dict[str, Any]) -> Dict[str, Any]:
    path = payload.get("path", "")
    if not path:
        raise ToolExecutionError("Missing path")
    candidate = _safe_artifact_path(path, "output.txt")
    if not candidate.exists():
        raise ToolExecutionError("File not found")
    return {"content": candidate.read_text(encoding="utf-8")}


def _workspace_read_text(payload: Dict[str, Any]) -> Dict[str, Any]:
    path = payload.get("path", "")
    if not path:
        raise ToolExecutionError("Missing path")
    candidate = _safe_workspace_path(path, "output.txt")
    if not candidate.exists():
        raise ToolExecutionError("File not found")
    return {"content": candidate.read_text(encoding="utf-8")}


def _list_files(payload: Dict[str, Any]) -> Dict[str, Any]:
    path = payload.get("path", "")
    recursive = bool(payload.get("recursive", False))
    max_files = payload.get("max_files", 200)
    if not isinstance(max_files, int) or max_files < 1:
        max_files = 200
    root = _safe_artifact_path(path, "")
    if not root.exists():
        return {"entries": []}
    if root.is_file():
        return {"entries": [{"path": str(root), "type": "file"}]}
    entries = []
    iterator = root.rglob("*") if recursive else root.glob("*")
    for entry in iterator:
        entry_type = "dir" if entry.is_dir() else "file"
        entries.append({"path": str(entry), "type": entry_type})
        if len(entries) >= max_files:
            break
    return {"entries": entries}


def _list_workspace_files(payload: Dict[str, Any]) -> Dict[str, Any]:
    path = payload.get("path", "")
    recursive = bool(payload.get("recursive", False))
    max_files = payload.get("max_files", 200)
    if not isinstance(max_files, int) or max_files < 1:
        max_files = 200
    root = _safe_workspace_path(path, "")
    if not root.exists():
        return {"entries": []}
    if root.is_file():
        return {"entries": [{"path": str(root), "type": "file"}]}
    entries = []
    iterator = root.rglob("*") if recursive else root.glob("*")
    for entry in iterator:
        entry_type = "dir" if entry.is_dir() else "file"
        entries.append({"path": str(entry), "type": entry_type})
        if len(entries) >= max_files:
            break
    return {"entries": entries}


def _artifact_mkdir(payload: Dict[str, Any]) -> Dict[str, Any]:
    path = payload.get("path", "")
    if not path:
        raise ToolExecutionError("Missing path")
    parents = bool(payload.get("parents", True))
    exist_ok = bool(payload.get("exist_ok", True))
    candidate = _safe_artifact_path(path, "")
    candidate.mkdir(parents=parents, exist_ok=exist_ok)
    return {"path": str(candidate)}


def _workspace_mkdir(payload: Dict[str, Any]) -> Dict[str, Any]:
    path = payload.get("path", "")
    if not path:
        raise ToolExecutionError("Missing path")
    parents = bool(payload.get("parents", True))
    exist_ok = bool(payload.get("exist_ok", True))
    candidate = _safe_workspace_path(path, "")
    candidate.mkdir(parents=parents, exist_ok=exist_ok)
    return {"path": str(candidate)}


def _delete_path(target: Path, *, recursive: bool, missing_ok: bool) -> Dict[str, Any]:
    if not target.exists():
        if missing_ok:
            return {"path": str(target), "deleted": False}
        raise ToolExecutionError("Path not found")
    if target.is_dir():
        if recursive:
            shutil.rmtree(target)
        else:
            target.rmdir()
    else:
        target.unlink()
    return {"path": str(target), "deleted": True}


def _artifact_delete(payload: Dict[str, Any]) -> Dict[str, Any]:
    path = payload.get("path", "")
    if not path:
        raise ToolExecutionError("Missing path")
    recursive = bool(payload.get("recursive", False))
    missing_ok = bool(payload.get("missing_ok", False))
    target = _safe_artifact_path(path, "")
    return _delete_path(target, recursive=recursive, missing_ok=missing_ok)


def _workspace_delete(payload: Dict[str, Any]) -> Dict[str, Any]:
    path = payload.get("path", "")
    if not path:
        raise ToolExecutionError("Missing path")
    recursive = bool(payload.get("recursive", False))
    missing_ok = bool(payload.get("missing_ok", False))
    target = _safe_workspace_path(path, "")
    return _delete_path(target, recursive=recursive, missing_ok=missing_ok)


def _replace_existing_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def _rename_path(
    source: Path,
    destination: Path,
    *,
    overwrite: bool,
) -> Dict[str, Any]:
    if not source.exists():
        raise ToolExecutionError("Source path not found")
    if destination.exists():
        if not overwrite:
            raise ToolExecutionError("Destination already exists")
        _replace_existing_path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    source.replace(destination)
    return {"path": str(destination)}


def _artifact_rename(payload: Dict[str, Any]) -> Dict[str, Any]:
    source_path = payload.get("source_path", "")
    destination_path = payload.get("destination_path", "")
    overwrite = bool(payload.get("overwrite", False))
    if not source_path:
        raise ToolExecutionError("Missing source_path")
    if not destination_path:
        raise ToolExecutionError("Missing destination_path")
    if destination_path.endswith("/"):
        raise ToolExecutionError("Missing file or directory name in destination_path")
    source = _safe_artifact_path(source_path, "")
    destination = _safe_artifact_path(destination_path, "")
    return _rename_path(source, destination, overwrite=overwrite)


def _workspace_rename(payload: Dict[str, Any]) -> Dict[str, Any]:
    source_path = payload.get("source_path", "")
    destination_path = payload.get("destination_path", "")
    overwrite = bool(payload.get("overwrite", False))
    if not source_path:
        raise ToolExecutionError("Missing source_path")
    if not destination_path:
        raise ToolExecutionError("Missing destination_path")
    if destination_path.endswith("/"):
        raise ToolExecutionError("Missing file or directory name in destination_path")
    source = _safe_workspace_path(source_path, "")
    destination = _safe_workspace_path(destination_path, "")
    return _rename_path(source, destination, overwrite=overwrite)


def _copy_path(
    source: Path,
    destination: Path,
    *,
    overwrite: bool,
    recursive: bool,
) -> Dict[str, Any]:
    if not source.exists():
        raise ToolExecutionError("Source path not found")
    if destination.exists():
        if not overwrite:
            raise ToolExecutionError("Destination already exists")
        _replace_existing_path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        if not recursive:
            raise ToolExecutionError("Source is a directory; set recursive=true")
        shutil.copytree(source, destination)
    else:
        shutil.copy2(source, destination)
    return {"path": str(destination)}


def _artifact_copy(payload: Dict[str, Any]) -> Dict[str, Any]:
    source_path = payload.get("source_path", "")
    destination_path = payload.get("destination_path", "")
    overwrite = bool(payload.get("overwrite", False))
    recursive = bool(payload.get("recursive", True))
    if not source_path:
        raise ToolExecutionError("Missing source_path")
    if not destination_path:
        raise ToolExecutionError("Missing destination_path")
    if destination_path.endswith("/"):
        raise ToolExecutionError("Missing file or directory name in destination_path")
    source = _safe_artifact_path(source_path, "")
    destination = _safe_artifact_path(destination_path, "")
    return _copy_path(source, destination, overwrite=overwrite, recursive=recursive)


def _workspace_copy(payload: Dict[str, Any]) -> Dict[str, Any]:
    source_path = payload.get("source_path", "")
    destination_path = payload.get("destination_path", "")
    overwrite = bool(payload.get("overwrite", False))
    recursive = bool(payload.get("recursive", True))
    if not source_path:
        raise ToolExecutionError("Missing source_path")
    if not destination_path:
        raise ToolExecutionError("Missing destination_path")
    if destination_path.endswith("/"):
        raise ToolExecutionError("Missing file or directory name in destination_path")
    source = _safe_workspace_path(source_path, "")
    destination = _safe_workspace_path(destination_path, "")
    return _copy_path(source, destination, overwrite=overwrite, recursive=recursive)


def _artifact_move_to_workspace(payload: Dict[str, Any]) -> Dict[str, Any]:
    source_path = payload.get("source_path", "")
    destination_path = payload.get("destination_path", "")
    overwrite = bool(payload.get("overwrite", False))
    if not source_path:
        raise ToolExecutionError("Missing source_path")
    if not destination_path:
        raise ToolExecutionError("Missing destination_path")
    if destination_path.endswith("/"):
        raise ToolExecutionError("Missing file name in destination_path")
    source = _safe_artifact_path(source_path, "")
    if not source.exists():
        raise ToolExecutionError("Source file not found")
    destination = _safe_workspace_path(destination_path, "")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise ToolExecutionError("Destination already exists")
    shutil.move(str(source), str(destination))
    return {"path": str(destination)}


def _derive_output_path(payload: Dict[str, Any]) -> Dict[str, Any]:
    def pick_str(*values: Any) -> str:
        for value in values:
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    memory_context = _select_job_context_from_memory(payload.get("memory"))
    nested_context = memory_context.get("context_json")
    if not isinstance(nested_context, dict):
        nested_context = {}

    topic = pick_str(
        payload.get("topic"),
        memory_context.get("topic"),
        nested_context.get("topic"),
    )
    date_value = pick_str(
        payload.get("date"),
        payload.get("today"),
        memory_context.get("date"),
        memory_context.get("today"),
        nested_context.get("date"),
        nested_context.get("today"),
    )
    output_dir = (
        pick_str(
            payload.get("output_dir"),
            memory_context.get("output_dir"),
            nested_context.get("output_dir"),
        )
        or "documents"
    )
    document_type = pick_str(
        payload.get("document_type"),
        memory_context.get("document_type"),
        nested_context.get("document_type"),
    )
    extension_hint = pick_str(
        payload.get("output_extension"),
        payload.get("file_extension"),
        payload.get("extension"),
        payload.get("format"),
        memory_context.get("output_extension"),
        memory_context.get("file_extension"),
        memory_context.get("extension"),
        memory_context.get("format"),
        nested_context.get("output_extension"),
        nested_context.get("file_extension"),
        nested_context.get("extension"),
        nested_context.get("format"),
    )

    if not topic:
        raise ToolExecutionError("Missing topic")

    normalized_doc_type = (
        document_type.lower().replace("-", "_")
        if isinstance(document_type, str) and document_type
        else "document"
    )
    known_format_types = {
        "pdf",
        "docx",
        "md",
        "markdown",
        "txt",
        "html",
        "htm",
        "json",
        "yaml",
        "yml",
        "xml",
        "csv",
    }

    def normalize_extension(raw: str) -> str:
        value = raw.strip().lower()
        if value.startswith("."):
            value = value[1:]
        if value == "markdown":
            value = "md"
        if not value:
            return ""
        if not re.fullmatch(r"[a-z0-9]{1,16}", value):
            raise ToolExecutionError("Invalid output_extension")
        return value

    output_extension = ""
    if extension_hint:
        output_extension = normalize_extension(extension_hint)
    elif normalized_doc_type in known_format_types:
        output_extension = normalize_extension(normalized_doc_type)
    if not output_extension:
        output_extension = "docx"

    output_dir = output_dir.strip().strip("/")
    if not output_dir:
        output_dir = "documents"
    if output_dir.startswith("/") or ".." in Path(output_dir).parts:
        raise ToolExecutionError("Invalid output_dir")

    date_source = date_value or datetime.now(UTC).date().isoformat()
    topic_slug = re.sub(r"[^a-z0-9]+", "_", topic.lower())
    topic_slug = re.sub(r"_+", "_", topic_slug).strip("_") or "document"
    date_slug = re.sub(r"[^0-9]+", "_", date_source)
    date_slug = re.sub(r"_+", "_", date_slug).strip("_")
    if not date_slug:
        raise ToolExecutionError("Invalid date")

    filename = f"{topic_slug}_{date_slug}.{output_extension}"
    return {
        "path": f"{output_dir}/{filename}",
        "document_type": normalized_doc_type,
        "output_extension": output_extension,
    }


def _derive_output_filename(payload: Dict[str, Any]) -> Dict[str, Any]:
    def pick_str(*values: Any) -> str:
        for value in values:
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    memory_context = _select_job_context_from_memory(payload.get("memory"))
    nested_context = memory_context.get("context_json")
    if not isinstance(nested_context, dict):
        nested_context = {}

    role_name = pick_str(
        payload.get("target_role_name"),
        payload.get("role_name"),
        payload.get("topic"),
        memory_context.get("target_role_name"),
        memory_context.get("role_name"),
        memory_context.get("topic"),
        nested_context.get("target_role_name"),
        nested_context.get("role_name"),
        nested_context.get("topic"),
    )
    company_name = pick_str(
        payload.get("company_name"),
        payload.get("company"),
        memory_context.get("company_name"),
        memory_context.get("company"),
        nested_context.get("company_name"),
        nested_context.get("company"),
    )
    candidate_name = pick_str(
        payload.get("candidate_name"),
        memory_context.get("candidate_name"),
        nested_context.get("candidate_name"),
    )
    first_name = pick_str(
        payload.get("first_name"),
        memory_context.get("first_name"),
        nested_context.get("first_name"),
    )
    last_name = pick_str(
        payload.get("last_name"),
        memory_context.get("last_name"),
        nested_context.get("last_name"),
    )
    job_description = pick_str(
        payload.get("job_description"),
        memory_context.get("job_description"),
        nested_context.get("job_description"),
    )
    date_value = pick_str(
        payload.get("date"),
        payload.get("today"),
        memory_context.get("date"),
        memory_context.get("today"),
        nested_context.get("date"),
        nested_context.get("today"),
    )
    output_dir = (
        pick_str(
            payload.get("output_dir"),
            memory_context.get("output_dir"),
            nested_context.get("output_dir"),
        )
        or "documents"
    )
    document_type = pick_str(
        payload.get("document_type"),
        memory_context.get("document_type"),
        nested_context.get("document_type"),
    )
    extension_hint = pick_str(
        payload.get("output_extension"),
        payload.get("file_extension"),
        payload.get("extension"),
        payload.get("format"),
        memory_context.get("output_extension"),
        memory_context.get("file_extension"),
        memory_context.get("extension"),
        memory_context.get("format"),
        nested_context.get("output_extension"),
        nested_context.get("file_extension"),
        nested_context.get("extension"),
        nested_context.get("format"),
    )
    normalized_doc_type = document_type.lower().replace("-", "_")
    known_format_types = {
        "pdf",
        "docx",
        "md",
        "markdown",
        "txt",
        "html",
        "htm",
        "json",
        "yaml",
        "yml",
        "xml",
        "csv",
    }

    def normalize_extension(raw: str) -> str:
        value = raw.strip().lower()
        if value.startswith("."):
            value = value[1:]
        if value == "markdown":
            value = "md"
        if not value:
            return ""
        if not re.fullmatch(r"[a-z0-9]{1,16}", value):
            raise ToolExecutionError("Invalid output_extension")
        return value

    output_extension = ""
    if extension_hint:
        output_extension = normalize_extension(extension_hint)
    elif normalized_doc_type in known_format_types:
        output_extension = normalize_extension(normalized_doc_type)
    if not output_extension:
        output_extension = "docx"
    if not isinstance(output_dir, str):
        output_dir = "documents"
    output_dir = output_dir.strip().strip("/")
    if not output_dir:
        output_dir = "documents"
    if output_dir.startswith("/") or ".." in Path(output_dir).parts:
        raise ToolExecutionError("Invalid output_dir")

    def clean_label(value: Any) -> str:
        if not isinstance(value, str):
            return ""
        # Keep output human-readable while stripping filesystem-unsafe characters.
        cleaned = re.sub(r'[<>:"/\\\\|?*]', " ", value)
        cleaned = re.sub(r"[,_;:]+", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" .")
        return cleaned

    def slugify(value: str, pattern: str) -> str:
        cleaned = re.sub(pattern, "_", value.lower())
        cleaned = re.sub(r"_+", "_", cleaned).strip("_")
        return cleaned

    if (not isinstance(role_name, str) or not role_name.strip()) and isinstance(job_description, str):
        role_name = _derive_role_name_from_jd(job_description)

    role_label = clean_label(role_name)
    if not role_label:
        raise ToolExecutionError("Missing target_role_name")
    if not isinstance(date_value, str) or not date_value.strip():
        # Fallback for plans that omit date/today in output-path derivation.
        date_value = datetime.now(UTC).date().isoformat()

    role_slug = slugify(role_label or str(role_name), r"[^a-z0-9]+") or "document"
    date_slug = slugify(date_value, r"[^0-9]+")
    if not date_slug:
        raise ToolExecutionError("Invalid date")
    filename = f"{role_slug}_{date_slug}.{output_extension}"
    return {
        "path": f"{output_dir}/{filename}",
        "document_type": normalized_doc_type or "document",
        "output_extension": output_extension,
    }


def _derive_role_name_from_jd(job_description: str) -> str:
    patterns = (
        r"(?im)^\s*title\s*:\s*(.+)$",
        r"(?im)^\s*role\s*:\s*(.+)$",
        r"(?im)^\s*position\s*:\s*(.+)$",
        r"(?im)\bwe are hiring (?:a|an)\s+([^.\n]+)",
        r"(?im)\bseeking (?:a|an)\s+([^.\n]+)",
    )
    for pattern in patterns:
        match = re.search(pattern, job_description)
        if match:
            return match.group(1).strip(" -:,.")
    first_line = next((line.strip() for line in job_description.splitlines() if line.strip()), "")
    return first_line[:120].strip(" -:,.")


def _select_job_context_from_memory(memory: Any) -> Dict[str, Any]:
    if not isinstance(memory, dict):
        return {}
    direct = memory.get("context_json")
    if isinstance(direct, dict):
        return direct
    entries = memory.get("job_contexts")
    if not isinstance(entries, list):
        entries = memory.get("job_context")
    if isinstance(entries, list):
        for entry in entries:
            if isinstance(entry, dict):
                payload = entry.get("payload")
                if isinstance(payload, dict):
                    context_json = payload.get("context_json")
                    if isinstance(context_json, dict):
                        return context_json
                    return payload
    task_outputs = memory.get("task_outputs")
    if isinstance(task_outputs, list):
        for entry in task_outputs:
            if not isinstance(entry, dict):
                continue
            payload = entry.get("payload")
            if isinstance(payload, dict):
                context_json = payload.get("context_json")
                if isinstance(context_json, dict):
                    return context_json
    return {}


def _run_tests(payload: Dict[str, Any]) -> Dict[str, Any]:
    command = payload.get("command", "")
    args = payload.get("args") or []
    cwd = payload.get("cwd", "")
    if not isinstance(args, list):
        args = []
    allowlist = {"pytest", "python"}
    if command not in allowlist:
        raise ToolExecutionError("Command not allowed")
    cmd = [command, *args]
    if command == "python":
        if len(args) < 2 or args[0] != "-m" or args[1] != "pytest":
            raise ToolExecutionError("Only python -m pytest is allowed")
    run_cwd = _safe_artifact_path(cwd, "")
    if not run_cwd.exists():
        raise ToolExecutionError("Working directory not found")
    result: CompletedProcess[str] = run(
        cmd, cwd=str(run_cwd), capture_output=True, text=True, check=False
    )
    return {"exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def _search_text(payload: Dict[str, Any]) -> Dict[str, Any]:
    query = payload.get("query", "")
    if not isinstance(query, str) or not query:
        raise ToolExecutionError("Missing query")
    path = payload.get("path", "")
    glob = payload.get("glob", "")
    case_sensitive = bool(payload.get("case_sensitive", False))
    use_regex = bool(payload.get("regex", False))
    context_lines = payload.get("context_lines", 0)
    if not isinstance(context_lines, int) or context_lines < 0:
        context_lines = 0
    max_matches = payload.get("max_matches", 200)
    if not isinstance(max_matches, int) or max_matches < 1:
        max_matches = 200
    root = _safe_artifact_path(path, "")
    if not root.exists():
        return {"matches": []}
    pattern = glob or "**/*"
    matches = []
    needle = query if case_sensitive else query.lower()
    regex = None
    if use_regex:
        import re

        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            regex = re.compile(query, flags=flags)
        except re.error as exc:
            raise ToolExecutionError(f"Invalid regex: {exc}") from exc
    for file_path in root.glob(pattern):
        if not file_path.is_file():
            continue
        try:
            with file_path.open("r", encoding="utf-8") as handle:
                lines = handle.readlines()
                for idx, line in enumerate(lines, start=1):
                    hay = line if case_sensitive else line.lower()
                    matched = False
                    if regex is not None:
                        matched = regex.search(line) is not None
                    else:
                        matched = needle in hay
                    if matched:
                        start = max(0, idx - 1 - context_lines)
                        end = min(len(lines), idx - 1 + context_lines + 1)
                        context = [item.rstrip("\n") for item in lines[start:end]]
                        entry = {"path": str(file_path), "line": idx, "text": line.rstrip("\n")}
                        if context_lines:
                            entry["context"] = context
                        matches.append(entry)
                        if len(matches) >= max_matches:
                            return {"matches": matches}
        except OSError:
            continue
    return {"matches": matches}


def _memory_client() -> MemoryClient:
    base_url = os.getenv("MEMORY_API_URL", "http://api:8000").strip() or "http://api:8000"
    timeout_raw = os.getenv("MEMORY_API_TIMEOUT_S", "5.0").strip()
    try:
        timeout_s = float(timeout_raw)
    except ValueError:
        timeout_s = 5.0
    timeout_s = max(0.5, min(timeout_s, 60.0))
    return MemoryClient(base_url, timeout_s=timeout_s)


def _memory_read(payload: Dict[str, Any]) -> Dict[str, Any]:
    name = payload.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ToolExecutionError("Missing name")
    limit = payload.get("limit", 50)
    if not isinstance(limit, int):
        raise ToolExecutionError("limit must be an integer")
    include_expired = bool(payload.get("include_expired", False))
    try:
        entries = _memory_client().read(
            name=name.strip(),
            scope=payload.get("scope") if isinstance(payload.get("scope"), str) else None,
            key=payload.get("key") if isinstance(payload.get("key"), str) else None,
            job_id=payload.get("job_id") if isinstance(payload.get("job_id"), str) else None,
            user_id=payload.get("user_id") if isinstance(payload.get("user_id"), str) else None,
            project_id=payload.get("project_id")
            if isinstance(payload.get("project_id"), str)
            else None,
            limit=max(1, min(limit, 200)),
            include_expired=include_expired,
        )
    except MemoryClientError as exc:
        raise ToolExecutionError(f"memory_read_failed:{exc}") from exc
    first = entries[0] if entries else None
    output: Dict[str, Any] = {"entries": entries, "count": len(entries)}
    if isinstance(first, dict):
        output["entry"] = first
        payload_obj = first.get("payload")
        if isinstance(payload_obj, dict):
            output["payload"] = payload_obj
    return output


def _memory_write(payload: Dict[str, Any]) -> Dict[str, Any]:
    name = payload.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ToolExecutionError("Missing name")
    entry_payload = payload.get("payload")
    if not isinstance(entry_payload, dict):
        raise ToolExecutionError("payload must be an object")

    request: Dict[str, Any] = {"name": name.strip(), "payload": entry_payload}
    for field in ("scope", "key", "job_id", "user_id", "project_id"):
        value = payload.get(field)
        if isinstance(value, str) and value.strip():
            request[field] = value.strip()
    ttl_seconds = payload.get("ttl_seconds")
    if ttl_seconds is not None:
        if not isinstance(ttl_seconds, int) or ttl_seconds <= 0:
            raise ToolExecutionError("ttl_seconds must be a positive integer")
        request["ttl_seconds"] = ttl_seconds
    metadata = payload.get("metadata")
    if metadata is not None:
        if not isinstance(metadata, dict):
            raise ToolExecutionError("metadata must be an object")
        request["metadata"] = metadata
    try:
        written = _memory_client().write(request)
    except MemoryClientError as exc:
        raise ToolExecutionError(f"memory_write_failed:{exc}") from exc
    if not isinstance(written, dict):
        raise ToolExecutionError("memory_write_failed:empty_response")
    return {"entry": written}


def _memory_semantic_write(payload: Dict[str, Any]) -> Dict[str, Any]:
    fact = payload.get("fact")
    if not isinstance(fact, str) or not fact.strip():
        raise ToolExecutionError("fact is required")
    request: Dict[str, Any] = {"fact": fact.strip()}
    for field in ("subject", "namespace", "source", "source_ref", "reasoning", "key", "user_id"):
        value = payload.get(field)
        if isinstance(value, str) and value.strip():
            request[field] = value.strip()
    for list_field in ("aliases", "keywords"):
        value = payload.get(list_field)
        if value is None:
            continue
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise ToolExecutionError(f"{list_field} must be an array of strings")
        request[list_field] = [item.strip() for item in value if item.strip()]
    confidence = payload.get("confidence")
    if confidence is not None:
        if not isinstance(confidence, (int, float)):
            raise ToolExecutionError("confidence must be a number between 0 and 1")
        confidence_f = float(confidence)
        if confidence_f < 0 or confidence_f > 1:
            raise ToolExecutionError("confidence must be a number between 0 and 1")
        request["confidence"] = confidence_f
    metadata = payload.get("metadata")
    if metadata is not None:
        if not isinstance(metadata, dict):
            raise ToolExecutionError("metadata must be an object")
        request["metadata"] = metadata
    try:
        written = _memory_client().semantic_write(request)
    except MemoryClientError as exc:
        raise ToolExecutionError(f"memory_semantic_write_failed:{exc}") from exc
    if not isinstance(written, dict):
        raise ToolExecutionError("memory_semantic_write_failed:empty_response")
    return written


def _memory_semantic_search(payload: Dict[str, Any]) -> Dict[str, Any]:
    query = payload.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ToolExecutionError("query is required")
    request: Dict[str, Any] = {"query": query.strip()}
    for field in ("namespace", "subject", "key", "user_id"):
        value = payload.get(field)
        if isinstance(value, str) and value.strip():
            request[field] = value.strip()
    limit = payload.get("limit")
    if limit is not None:
        if not isinstance(limit, int):
            raise ToolExecutionError("limit must be an integer")
        request["limit"] = max(1, min(limit, 50))
    min_score = payload.get("min_score")
    if min_score is not None:
        if not isinstance(min_score, (int, float)):
            raise ToolExecutionError("min_score must be a number")
        request["min_score"] = max(0.0, float(min_score))
    include_payload = payload.get("include_payload")
    if include_payload is not None:
        request["include_payload"] = bool(include_payload)
    try:
        result = _memory_client().semantic_search(request)
    except MemoryClientError as exc:
        raise ToolExecutionError(f"memory_semantic_search_failed:{exc}") from exc
    if not isinstance(result, dict):
        raise ToolExecutionError("memory_semantic_search_failed:empty_response")
    return result


def _resolve_template_path(template_path: str, template_id: str) -> Path:
    base_dir = Path(os.getenv("DOCX_TEMPLATE_DIR", "/shared/templates"))
    base_dir.mkdir(parents=True, exist_ok=True)
    candidate = Path(template_path or "")
    if not template_path:
        candidate = Path(f"{template_id}.docx")
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        resolved = (base_dir / candidate).resolve()
    if not str(resolved).startswith(str(base_dir.resolve())):
        raise ToolExecutionError("Invalid template path outside template directory")
    return resolved


def _docx_render(payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        from docxtpl import DocxTemplate
    except Exception as exc:  # noqa: BLE001
        raise ToolExecutionError("docxtpl is not installed") from exc
    data = payload.get("data")
    if not isinstance(data, dict):
        raise ToolExecutionError("data must be an object")
    schema_ref = payload.get("schema_ref", "")
    template_id = payload.get("template_id", "")
    template_path = payload.get("template_path", "")
    output_path = payload.get("output_path", "")
    if not output_path:
        raise ToolExecutionError("Missing output_path")
    if not schema_ref and template_id:
        schema_ref = template_id
    if schema_ref:
        _validate_schema_from_registry(schema_ref, data)
    if not template_id and not template_path:
        raise ToolExecutionError("Missing template_id or template_path")
    template_file = _resolve_template_path(template_path, template_id)
    if not template_file.exists():
        raise ToolExecutionError("DOCX template not found")
    candidate = _safe_artifact_path(output_path, f"docx_{template_id or 'output'}.docx")
    candidate.parent.mkdir(parents=True, exist_ok=True)
    doc = DocxTemplate(str(template_file))
    doc.render(data)
    doc.save(str(candidate))
    return {"path": str(candidate)}


def _sleep(payload: Dict[str, Any]) -> Dict[str, Any]:
    seconds = float(payload.get("seconds", 0))
    time.sleep(seconds)
    return {"slept": seconds}


def _http_fetch(payload: Dict[str, Any]) -> Dict[str, Any]:
    import urllib.request

    url = payload.get("url", "")
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ToolExecutionError("Unsupported URL scheme")
    host = parsed.hostname or ""
    allowlist = _parse_http_allowlist()
    if not _host_allowed(host, allowlist):
        raise ToolExecutionError("URL host not in allowlist")
    with urllib.request.urlopen(url, timeout=5) as response:
        body = response.read().decode("utf-8")
    return {"body": body}


def _post_json(url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    return coder_tools.post_json(url, payload, timeout_s=_resolve_coder_http_timeout_s())


def _post_mcp_tool_call(
    service_url: str,
    tool_name: str,
    arguments: Dict[str, Any],
) -> Dict[str, Any]:
    return mcp_client.post_mcp_tool_call(
        service_url,
        tool_name,
        arguments,
        call_mcp_tool_sdk=_call_mcp_tool_sdk,
        classify_tool_error=_classify_tool_error,
        logger=LOGGER,
        tracing_module=core_tracing,
    )


def _call_mcp_tool_sdk(
    mcp_url: str,
    tool_name: str,
    arguments: Dict[str, Any],
    timeout_s: float,
    headers: Dict[str, str] | None = None,
) -> Dict[str, Any]:
    return mcp_client.call_mcp_tool_sdk(
        mcp_url,
        tool_name,
        arguments,
        timeout_s,
        headers=headers,
        tracing_module=core_tracing,
        logger=LOGGER,
    )


def _call_mcp_tool_sdk_inproc(
    mcp_url: str,
    tool_name: str,
    arguments: Dict[str, Any],
    timeout_s: float,
    headers: Dict[str, str] | None = None,
) -> Dict[str, Any]:
    return mcp_client.call_mcp_tool_sdk_inproc(
        mcp_url,
        tool_name,
        arguments,
        timeout_s,
        headers=headers,
        tracing_module=core_tracing,
    )


def _mcp_process_entry(
    queue: Any,
    mcp_url: str,
    tool_name: str,
    arguments: Dict[str, Any],
    timeout_s: float,
    headers: Dict[str, str] | None = None,
) -> None:
    mcp_client.mcp_process_entry(
        queue,
        mcp_url,
        tool_name,
        arguments,
        timeout_s,
        headers,
    )


def _call_mcp_tool_sdk_process(
    mcp_url: str,
    tool_name: str,
    arguments: Dict[str, Any],
    timeout_s: float,
    headers: Dict[str, str] | None = None,
) -> Dict[str, Any]:
    return mcp_client.call_mcp_tool_sdk_process(
        mcp_url,
        tool_name,
        arguments,
        timeout_s,
        headers=headers,
        logger=LOGGER,
        tracing_module=core_tracing,
    )


def _flatten_exception_messages(exc: BaseException) -> List[str]:
    return mcp_client.flatten_exception_messages(exc)


def _extract_mcp_sdk_result(result: Any) -> Dict[str, Any]:
    return mcp_client.extract_mcp_sdk_result(result)


def _normalize_mcp_structured_result(structured: Dict[str, Any]) -> Dict[str, Any]:
    return mcp_client.normalize_mcp_structured_result(structured)


def _extract_mcp_error_detail(result: Any) -> str:
    return mcp_client.extract_mcp_error_detail(result)


def _coding_agent_generate(payload: Dict[str, Any]) -> Dict[str, Any]:
    return coder_tools.coding_agent_generate(
        payload,
        post_mcp_tool_call=_post_mcp_tool_call,
        write_workspace_text_file=_write_workspace_text_file,
    )


def _build_plan_prompt(goal: str, constraints: Optional[str], max_steps: int) -> str:
    return coder_tools.build_plan_prompt(goal, constraints, max_steps)


def _render_plan_markdown(goal: str, steps: list[dict[str, Any]], statuses: list[bool]) -> str:
    return coder_tools.render_plan_markdown(goal, steps, statuses)


def _coding_agent_autonomous(payload: Dict[str, Any], provider: LLMProvider) -> Dict[str, Any]:
    return coder_tools.coding_agent_autonomous(
        payload,
        provider,
        post_mcp_tool_call=_post_mcp_tool_call,
        write_workspace_text_file=_write_workspace_text_file,
        extract_json=_extract_json,
    )


def _coding_agent_publish_pr(payload: Dict[str, Any]) -> Dict[str, Any]:
    return coder_tools.coding_agent_publish_pr(
        payload,
        safe_workspace_path=_safe_workspace_path,
        invoke_capability=mcp_gateway.invoke_capability,
    )


def _llm_generate(payload: Dict[str, Any], provider: LLMProvider) -> Dict[str, Any]:
    prompt = payload.get("text") or payload.get("prompt") or ""
    response = provider.generate(prompt)
    return {"text": response.content}


def _llm_generate_document_spec(payload: Dict[str, Any], provider: LLMProvider) -> Dict[str, Any]:
    return _llm_generate_document_spec_external(
        payload,
        provider,
        sanitize_document_spec=_sanitize_document_spec,
    )


def _sanitize_document_spec(document_spec: Dict[str, Any]) -> Dict[str, Any]:
    sanitized = dict(document_spec)
    blocks = sanitized.get("blocks")
    if isinstance(blocks, list):
        sanitized["blocks"] = _sanitize_document_blocks(blocks)
    return sanitized


def _sanitize_document_blocks(blocks: List[Any]) -> List[Dict[str, Any]]:
    cleaned: List[Dict[str, Any]] = []
    for raw in blocks:
        if not isinstance(raw, dict):
            continue
        block = dict(raw)
        block_type = block.get("type")

        if block_type == "spacer":
            # Spacer maps to an empty paragraph in DOCX; skip to avoid blank lines.
            continue

        if block_type in {"text", "paragraph", "heading", "optional_paragraph"}:
            text = block.get("text")
            if isinstance(text, str) and not text.strip():
                continue

        if block_type == "bullets":
            items = block.get("items")
            if isinstance(items, list):
                filtered_items = []
                for item in items:
                    if isinstance(item, str) and not item.strip():
                        continue
                    filtered_items.append(item)
                if not filtered_items:
                    continue
                block["items"] = filtered_items

        if block_type == "repeat":
            template = block.get("template")
            if isinstance(template, list):
                cleaned_template = _sanitize_document_blocks(template)
                if not cleaned_template:
                    continue
                block["template"] = cleaned_template

        cleaned.append(block)
    return cleaned


def _resolve_llm_iterative_tool_timeout_s(provider: LLMProvider) -> int:
    base = _resolve_llm_timeout_s(provider)
    return min(900, max(60, base * 3))


def _extract_json(text: str) -> str:
    fence_match = re.search(r"```(?:json)?\s*(.+?)```", text, flags=re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text.strip()


def _llm_improve_document_spec(payload: Dict[str, Any], provider: LLMProvider) -> Dict[str, Any]:
    return _llm_improve_document_spec_external(
        payload,
        provider,
        sanitize_document_spec=_sanitize_document_spec,
    )
