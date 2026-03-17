from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

from libs.core import capability_registry, models
from services.api.app import chat_execution_service


class _FakeRegistry:
    def get(self, tool_name: str):
        return SimpleNamespace(
            spec=models.ToolSpec(
                name=tool_name,
                description="fake",
                input_schema={},
                output_schema={},
                tool_intent=models.ToolIntent.io,
            )
        )

    def execute(self, tool_name: str, *, payload, idempotency_key: str, trace_id: str, max_output_bytes: int):
        return models.ToolCall(
            tool_name=tool_name,
            input=dict(payload),
            idempotency_key=idempotency_key,
            trace_id=trace_id,
            started_at=datetime.now(UTC),
            finished_at=datetime.now(UTC),
            status="completed",
            output_or_error={"entries": [{"path": "/shared/workspace/README.md", "type": "file"}]},
        )


def test_execute_capability_allows_read_only_native_tool(monkeypatch) -> None:
    spec = capability_registry.CapabilitySpec(
        capability_id="filesystem.workspace.list",
        description="List workspace files",
        risk_tier="read_only",
        idempotency="read",
        adapters=(
            capability_registry.CapabilityAdapterSpec(
                type="tool",
                server_id="local_worker",
                tool_name="workspace_list_files",
            ),
        ),
        enabled=True,
    )
    monkeypatch.setattr(
        chat_execution_service.capability_registry,
        "load_capability_registry",
        lambda: capability_registry.CapabilityRegistry({"filesystem.workspace.list": spec}),
    )
    monkeypatch.setattr(
        chat_execution_service.capability_registry,
        "evaluate_capability_allowlist",
        lambda capability_id, service_name=None: capability_registry.CapabilityAllowDecision(
            allowed=True, reason="allowed"
        ),
    )
    monkeypatch.setattr(
        chat_execution_service.tool_governance,
        "evaluate_tool_allowlist",
        lambda tool_name, service_name=None, context=None, tool_spec=None: SimpleNamespace(
            allowed=True, reason="allowed"
        ),
    )

    executor = chat_execution_service.ChatDirectExecutor(
        registry=_FakeRegistry(),
        config=chat_execution_service.ChatDirectExecutionConfig(
            allowed_capabilities={"filesystem.workspace.list"}
        ),
    )

    result = executor.execute_capability(
        capability_id="filesystem.workspace.list",
        arguments={"path": ""},
        trace_id="trace-1",
    )

    assert result.capability_id == "filesystem.workspace.list"
    assert "README.md" in result.assistant_response


def test_execute_capability_rejects_non_read_only(monkeypatch) -> None:
    spec = capability_registry.CapabilitySpec(
        capability_id="document.docx.generate",
        description="Render DOCX",
        risk_tier="bounded_write",
        idempotency="write",
        enabled=True,
    )
    monkeypatch.setattr(
        chat_execution_service.capability_registry,
        "load_capability_registry",
        lambda: capability_registry.CapabilityRegistry({"document.docx.generate": spec}),
    )

    executor = chat_execution_service.ChatDirectExecutor(
        registry=_FakeRegistry(),
        config=chat_execution_service.ChatDirectExecutionConfig(
            allowed_capabilities={"document.docx.generate"}
        ),
    )

    try:
        executor.execute_capability(
            capability_id="document.docx.generate",
            arguments={},
            trace_id="trace-1",
        )
    except Exception as exc:  # noqa: BLE001
        assert "chat_direct_capability_not_read_only" in str(exc)
    else:
        raise AssertionError("expected direct execution to reject non-read-only capability")


def test_execute_capability_allows_rag_retrieve_mcp(monkeypatch) -> None:
    spec = capability_registry.CapabilitySpec(
        capability_id="rag.retrieve",
        description="Retrieve ranked chunks from a vector-backed RAG index",
        risk_tier="read_only",
        idempotency="read",
        adapters=(
            capability_registry.CapabilityAdapterSpec(
                type="mcp",
                server_id="rag_retriever_qdrant",
                tool_name="retrieve",
            ),
        ),
        enabled=True,
    )
    monkeypatch.setattr(
        chat_execution_service.capability_registry,
        "load_capability_registry",
        lambda: capability_registry.CapabilityRegistry({"rag.retrieve": spec}),
    )
    monkeypatch.setattr(
        chat_execution_service.capability_registry,
        "evaluate_capability_allowlist",
        lambda capability_id, service_name=None: capability_registry.CapabilityAllowDecision(
            allowed=True, reason="allowed"
        ),
    )
    monkeypatch.setattr(
        chat_execution_service.mcp_gateway,
        "invoke_capability",
        lambda capability_id, arguments, capability_registry=None, execute_tool=None: {
            "matches": [
                {
                    "chunk_id": "chunk-1",
                    "document_id": "doc-1",
                    "text": "Kubernetes local deployment syncs config and restarts workloads.",
                    "score": 0.913,
                    "metadata": {"path": "README.md", "section": "Kubernetes"},
                    "source_uri": "README.md#kubernetes",
                }
            ]
        },
    )

    executor = chat_execution_service.ChatDirectExecutor(
        registry=_FakeRegistry(),
        config=chat_execution_service.ChatDirectExecutionConfig(
            allowed_capabilities={"rag.retrieve"}
        ),
    )

    result = executor.execute_capability(
        capability_id="rag.retrieve",
        arguments={"query": "How does Kubernetes local deployment work?", "top_k": 3},
        trace_id="trace-rag-1",
    )

    assert result.capability_id == "rag.retrieve"
    assert result.tool_name == "retrieve"
    assert "Retrieved matches:" in result.assistant_response
    assert "README.md#kubernetes" in result.assistant_response
