from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from libs.core import llm_provider, models, tool_bootstrap, tool_registry
from libs.framework.tool_runtime import Tool, ToolRegistry


@dataclass(frozen=True)
class WorkerToolRuntime:
    registry: ToolRegistry
    service_name: str = "worker"

    def get_tool(self, tool_name: str) -> Tool:
        return self.registry.get(tool_name)

    def execute_tool(
        self,
        tool_name: str,
        *,
        payload: dict[str, Any],
        idempotency_key: str,
        trace_id: str,
        max_output_bytes: int,
    ) -> models.ToolCall:
        return self.registry.execute(
            tool_name,
            payload=payload,
            idempotency_key=idempotency_key,
            trace_id=trace_id,
            max_output_bytes=max_output_bytes,
        )

    def evaluate_allowlist(
        self,
        tool_name: str,
        *,
        context: dict[str, Any] | None = None,
        tool_spec: models.ToolSpec | None = None,
    ) -> tool_registry.ToolAllowDecision:
        return tool_registry.evaluate_tool_allowlist(
            tool_name,
            self.service_name,
            context=context,
            tool_spec=tool_spec,
        )


def build_worker_tool_runtime(
    *,
    http_fetch_enabled: bool,
    llm_enabled: bool,
    llm_provider_instance: llm_provider.LLMProvider | None,
    service_name: str = "worker",
) -> WorkerToolRuntime:
    registry = tool_bootstrap.build_default_registry(
        http_fetch_enabled=http_fetch_enabled,
        llm_enabled=llm_enabled,
        llm_provider=llm_provider_instance,
        service_name=service_name,
    )
    return WorkerToolRuntime(registry=registry, service_name=service_name)
