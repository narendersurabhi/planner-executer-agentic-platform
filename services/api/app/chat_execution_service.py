from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Any

from libs.core import capability_registry, mcp_gateway, tool_bootstrap, tool_governance
from libs.framework.tool_runtime import ToolExecutionError, ToolRegistry


DEFAULT_CHAT_DIRECT_CAPABILITIES = {
    "github.repo.list",
    "github.user.me",
    "github.issue.search",
    "github.user.search",
    "github.branch.list",
    "filesystem.artifacts.list",
    "filesystem.artifacts.read_text",
    "filesystem.artifacts.search_text",
    "filesystem.workspace.list",
    "filesystem.workspace.read_text",
    "memory.read",
    "memory.semantic.search",
    "rag.retrieve",
}


@dataclass(frozen=True)
class ChatDirectExecutionConfig:
    allowed_capabilities: set[str]
    output_size_cap: int = 50_000
    max_preview_chars: int = 1_500


@dataclass(frozen=True)
class ChatDirectExecutionResult:
    capability_id: str
    tool_name: str
    output: dict[str, Any]
    assistant_response: str


@dataclass(frozen=True)
class ChatDirectExecutor:
    registry: ToolRegistry
    config: ChatDirectExecutionConfig
    service_name: str = "api"

    def execute_capability(
        self,
        *,
        capability_id: str,
        arguments: dict[str, Any],
        trace_id: str,
    ) -> ChatDirectExecutionResult:
        normalized_capability_id = str(capability_id or "").strip()
        if not normalized_capability_id:
            raise ToolExecutionError("chat_direct_missing_capability_id")
        if normalized_capability_id not in self.config.allowed_capabilities:
            raise ToolExecutionError(f"chat_direct_capability_not_allowed:{normalized_capability_id}")

        registry = capability_registry.load_capability_registry()
        spec = registry.require(normalized_capability_id)
        if not spec.enabled:
            raise ToolExecutionError(f"chat_direct_capability_disabled:{normalized_capability_id}")
        if spec.risk_tier != "read_only":
            raise ToolExecutionError(f"chat_direct_capability_not_read_only:{normalized_capability_id}")

        allow_decision = capability_registry.evaluate_capability_allowlist(
            normalized_capability_id,
            self.service_name,
        )
        if not allow_decision.allowed:
            raise ToolExecutionError(
                f"chat_direct_capability_blocked:{normalized_capability_id}:{allow_decision.reason}"
            )

        def _execute_native_tool(tool_name: str, tool_arguments: dict[str, Any]) -> dict[str, Any]:
            tool = self.registry.get(tool_name)
            tool_decision = tool_governance.evaluate_tool_allowlist(
                tool_name,
                self.service_name,
                context={"chat_direct": True},
                tool_spec=tool.spec,
            )
            if not tool_decision.allowed:
                raise ToolExecutionError(f"chat_direct_tool_blocked:{tool_name}:{tool_decision.reason}")
            payload = dict(tool_arguments)
            payload["_registry"] = self.registry
            call = self.registry.execute(
                tool_name,
                payload=payload,
                idempotency_key=str(uuid.uuid4()),
                trace_id=trace_id,
                max_output_bytes=self.config.output_size_cap,
            )
            if call.status != "completed":
                output = call.output_or_error if isinstance(call.output_or_error, dict) else {}
                raise ToolExecutionError(str(output.get("error", "chat_direct_tool_failed")))
            output = call.output_or_error
            if isinstance(output, dict):
                return output
            return {"result": output}

        result = mcp_gateway.invoke_capability(
            normalized_capability_id,
            dict(arguments),
            capability_registry=registry,
            execute_tool=_execute_native_tool,
        )
        output = result if isinstance(result, dict) else {"result": result}
        return ChatDirectExecutionResult(
            capability_id=normalized_capability_id,
            tool_name=_tool_name_for_capability(spec),
            output=output,
            assistant_response=_format_chat_direct_result(
                normalized_capability_id,
                output,
                max_preview_chars=self.config.max_preview_chars,
            ),
        )


def build_chat_direct_executor(
    *,
    service_name: str = "api",
    allowed_capabilities: set[str] | None = None,
    output_size_cap: int = 50_000,
    llm_enabled: bool = False,
    llm_provider_instance: Any | None = None,
) -> ChatDirectExecutor:
    registry = tool_bootstrap.build_default_registry(
        http_fetch_enabled=False,
        llm_enabled=llm_enabled,
        llm_provider=llm_provider_instance,
        service_name=service_name,
    )
    return ChatDirectExecutor(
        registry=registry,
        service_name=service_name,
        config=ChatDirectExecutionConfig(
            allowed_capabilities=set(allowed_capabilities or DEFAULT_CHAT_DIRECT_CAPABILITIES),
            output_size_cap=output_size_cap,
        ),
    )


def _tool_name_for_capability(spec: capability_registry.CapabilitySpec) -> str:
    if not spec.adapters:
        return spec.capability_id
    return spec.adapters[0].tool_name


def tool_name_for_capability(spec: capability_registry.CapabilitySpec) -> str:
    return _tool_name_for_capability(spec)


def _format_chat_direct_result(
    capability_id: str,
    output: dict[str, Any],
    *,
    max_preview_chars: int,
) -> str:
    if capability_id == "github.repo.list":
        items = output.get("items")
        if isinstance(items, list) and items:
            names = [
                str(item.get("full_name") or item.get("name") or "").strip()
                for item in items
                if isinstance(item, dict)
            ]
            names = [name for name in names if name]
            if names:
                return "Repositories:\n" + "\n".join(f"- {name}" for name in names[:10])
    if capability_id == "github.branch.list":
        items = output.get("items")
        if isinstance(items, list) and items:
            names = [
                str(item.get("name") or "").strip()
                for item in items
                if isinstance(item, dict)
            ]
            names = [name for name in names if name]
            if names:
                return "Branches:\n" + "\n".join(f"- {name}" for name in names[:15])
    if capability_id in {"filesystem.workspace.list", "filesystem.artifacts.list"}:
        entries = output.get("entries")
        if isinstance(entries, list):
            paths = [
                str(item.get("path") or "").strip()
                for item in entries
                if isinstance(item, dict)
            ]
            paths = [path for path in paths if path]
            if paths:
                return "Entries:\n" + "\n".join(f"- {path}" for path in paths[:20])
            return "No entries found."
    if capability_id in {"filesystem.workspace.read_text", "filesystem.artifacts.read_text"}:
        content = output.get("content")
        if isinstance(content, str):
            preview = content[:max_preview_chars]
            return preview if len(content) <= max_preview_chars else f"{preview}\n\n[truncated]"
    if capability_id == "filesystem.artifacts.search_text":
        matches = output.get("matches")
        if isinstance(matches, list) and matches:
            lines: list[str] = []
            for match in matches[:10]:
                if not isinstance(match, dict):
                    continue
                path = str(match.get("path") or "").strip()
                line = match.get("line")
                text = str(match.get("text") or "").strip()
                prefix = f"{path}:{line}" if path and line is not None else path or "match"
                lines.append(f"- {prefix} {text}".rstrip())
            if lines:
                return "Matches:\n" + "\n".join(lines)
            return "No matches found."
    if capability_id in {"memory.read", "memory.semantic.search"}:
        entries = output.get("entries") or output.get("matches")
        if isinstance(entries, list) and entries:
            preview = json.dumps(entries[:5], ensure_ascii=True, indent=2)
            return preview[:max_preview_chars]
    if capability_id == "rag.retrieve":
        matches = output.get("matches")
        if isinstance(matches, list) and matches:
            lines: list[str] = []
            for match in matches[:5]:
                if not isinstance(match, dict):
                    continue
                source_uri = str(match.get("source_uri") or "").strip()
                document_id = str(match.get("document_id") or "").strip()
                score = match.get("score")
                text = str(match.get("text") or "").strip()
                label = source_uri or document_id or "match"
                if isinstance(score, (int, float)):
                    prefix = f"- {label} (score {score:.3f})"
                else:
                    prefix = f"- {label}"
                if text:
                    excerpt = text[:180]
                    suffix = "..." if len(text) > 180 else ""
                    lines.append(f"{prefix}: {excerpt}{suffix}")
                else:
                    lines.append(prefix)
            if lines:
                return "Retrieved matches:\n" + "\n".join(lines)
            return "No matches found."

    rendered = json.dumps(output, ensure_ascii=True, indent=2, default=str)
    if len(rendered) > max_preview_chars:
        return rendered[:max_preview_chars] + "\n\n[truncated]"
    return rendered


def format_chat_direct_result(
    capability_id: str,
    output: dict[str, Any],
    *,
    max_preview_chars: int = 1_500,
) -> str:
    return _format_chat_direct_result(
        capability_id,
        output,
        max_preview_chars=max_preview_chars,
    )
