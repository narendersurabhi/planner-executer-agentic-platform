import logging

import pytest

from libs.framework.tool_runtime import ToolExecutionError
from libs.tools import mcp_client


class _Span:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None


class _Tracing:
    @staticmethod
    def start_span(name: str, attributes=None):
        del name, attributes
        return _Span()

    @staticmethod
    def set_span_attributes(span, attributes):
        del span, attributes


def _classify(error_text: str) -> str:
    if "timeout" in error_text:
        return "runtime.timeout"
    if "tool_error" in error_text:
        return "contract.tool_error"
    return "runtime.tool_error"


def test_post_mcp_tool_call_uses_primary_route_first(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    def fake_call(url: str, tool_name: str, arguments: dict, timeout_s: float) -> dict:
        del tool_name, arguments
        seen.append(url)
        assert timeout_s > 0
        return {"ok": True}

    monkeypatch.setenv("MCP_TOOL_TIMEOUT_S", "10")
    result = mcp_client.post_mcp_tool_call(
        "http://tailor:8000",
        "improve_iterative",
        {"x": 1},
        call_mcp_tool_sdk=fake_call,
        classify_tool_error=_classify,
        logger=logging.getLogger(__name__),
        tracing_module=_Tracing,
    )
    assert result["ok"] is True
    assert seen == ["http://tailor:8000/mcp/rpc/mcp"]


def test_post_mcp_tool_call_falls_back_to_legacy_route(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    def fake_call(url: str, tool_name: str, arguments: dict, timeout_s: float) -> dict:
        del tool_name, arguments, timeout_s
        seen.append(url)
        if url.endswith("/mcp/rpc/mcp"):
            raise ToolExecutionError("mcp_sdk_error:not_found")
        return {"ok": True}

    monkeypatch.setenv("MCP_TOOL_TIMEOUT_S", "10")
    monkeypatch.setenv("MCP_TOOL_MAX_RETRIES", "0")
    out = mcp_client.post_mcp_tool_call(
        "http://tailor:8000",
        "tailor_resume",
        {},
        call_mcp_tool_sdk=fake_call,
        classify_tool_error=_classify,
        logger=logging.getLogger(__name__),
        tracing_module=_Tracing,
    )
    assert out["ok"] is True
    assert seen == ["http://tailor:8000/mcp/rpc/mcp", "http://tailor:8000/mcp/rpc"]


def test_post_mcp_tool_call_retries_retryable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = {"count": 0}

    def fake_call(url: str, tool_name: str, arguments: dict, timeout_s: float) -> dict:
        del url, tool_name, arguments, timeout_s
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise ToolExecutionError("mcp_sdk_error:Session terminated")
        return {"ok": True}

    monkeypatch.setenv("MCP_TOOL_TIMEOUT_S", "10")
    monkeypatch.setenv("MCP_TOOL_MAX_RETRIES", "1")
    monkeypatch.setenv("MCP_TOOL_RETRY_SLEEP_S", "0")
    out = mcp_client.post_mcp_tool_call(
        "http://tailor:8000",
        "tailor_resume",
        {},
        call_mcp_tool_sdk=fake_call,
        classify_tool_error=_classify,
        logger=logging.getLogger(__name__),
        tracing_module=_Tracing,
    )
    assert out["ok"] is True
    assert attempts["count"] == 2


def test_post_mcp_tool_call_does_not_retry_tool_error(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = {"count": 0}

    def fake_call(url: str, tool_name: str, arguments: dict, timeout_s: float) -> dict:
        del url, tool_name, arguments, timeout_s
        attempts["count"] += 1
        raise ToolExecutionError("mcp_tool_error:bad_input")

    monkeypatch.setenv("MCP_TOOL_TIMEOUT_S", "10")
    monkeypatch.setenv("MCP_TOOL_MAX_RETRIES", "2")
    with pytest.raises(ToolExecutionError) as exc:
        mcp_client.post_mcp_tool_call(
            "http://tailor:8000",
            "tailor_resume",
            {},
            call_mcp_tool_sdk=fake_call,
            classify_tool_error=_classify,
            logger=logging.getLogger(__name__),
            tracing_module=_Tracing,
        )
    assert "mcp_tool_error:bad_input" in str(exc.value)
    assert attempts["count"] == 1


def test_resolve_mcp_timeout_fallback_and_clamp(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MCP_TOOL_TIMEOUT_S", raising=False)
    monkeypatch.delenv("MCP_TIMEOUT_S", raising=False)
    monkeypatch.delenv("TAILOR_OPENAI_TIMEOUT_S", raising=False)
    monkeypatch.delenv("TAILOR_EVAL_OPENAI_TIMEOUT_S", raising=False)
    monkeypatch.setenv("OPENAI_TIMEOUT_S", "600")
    assert mcp_client.resolve_mcp_timeout_s() == 180.0
    monkeypatch.setenv("OPENAI_TIMEOUT_S", "60")
    assert mcp_client.resolve_mcp_timeout_s() == 60.0
