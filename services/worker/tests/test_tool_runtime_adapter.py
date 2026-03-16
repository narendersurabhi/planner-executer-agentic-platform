from __future__ import annotations

from libs.core import models
from services.worker.app import tool_runtime_adapter


def test_build_worker_tool_runtime_uses_tool_bootstrap(monkeypatch) -> None:
    registry = object()
    seen: list[tuple[bool, bool, object | None, str]] = []

    def fake_build_default_registry(
        *,
        http_fetch_enabled: bool = False,
        llm_enabled: bool = False,
        llm_provider=None,
        service_name: str | None = None,
    ) -> object:
        seen.append((http_fetch_enabled, llm_enabled, llm_provider, str(service_name)))
        return registry

    monkeypatch.setattr(
        tool_runtime_adapter.tool_bootstrap,
        "build_default_registry",
        fake_build_default_registry,
    )

    runtime = tool_runtime_adapter.build_worker_tool_runtime(
        http_fetch_enabled=True,
        llm_enabled=True,
        llm_provider_instance="provider",
        service_name="worker",
    )

    assert runtime.registry is registry
    assert runtime.service_name == "worker"
    assert seen == [(True, True, "provider", "worker")]


def test_worker_tool_runtime_delegates_allowlist(monkeypatch) -> None:
    runtime = tool_runtime_adapter.WorkerToolRuntime(registry=object(), service_name="worker")
    expected = tool_runtime_adapter.tool_registry.ToolAllowDecision(True, "allowed")
    seen: list[tuple[str, str, dict[str, object] | None, models.ToolSpec | None]] = []

    def fake_evaluate(
        tool_name: str,
        service_name: str | None = None,
        *,
        context=None,
        tool_spec=None,
    ):
        seen.append((tool_name, str(service_name), context, tool_spec))
        return expected

    monkeypatch.setattr(
        tool_runtime_adapter.tool_registry,
        "evaluate_tool_allowlist",
        fake_evaluate,
    )

    decision = runtime.evaluate_allowlist(
        "llm_generate",
        context={"job_id": "job-1"},
        tool_spec=models.ToolSpec(
            name="llm_generate",
            description="generate",
            input_schema={},
            output_schema={},
            tool_intent=models.ToolIntent.generate,
        ),
    )

    assert decision is expected
    assert seen[0][0] == "llm_generate"
    assert seen[0][1] == "worker"
