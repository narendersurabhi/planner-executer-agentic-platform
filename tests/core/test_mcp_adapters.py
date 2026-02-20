from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import importlib.util
import json
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest

FastAPI = pytest.importorskip("fastapi").FastAPI
uvicorn = pytest.importorskip("uvicorn")
pytest.importorskip("mcp")
from mcp import ClientSession  # noqa: E402
from mcp.client.streamable_http import streamable_http_client  # noqa: E402


class _FakeLogger:
    def info(self, *_args: Any, **_kwargs: Any) -> None:
        return

    def error(self, *_args: Any, **_kwargs: Any) -> None:
        return


class _FakeLLMResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeCoderProvider:
    def generate(self, _prompt: str) -> _FakeLLMResponse:
        return _FakeLLMResponse('{"files":[{"path":"hello.py","content":"print(\\"hello\\")"}]}')


class _FakeTailorProvider:
    def generate(self, _prompt: str) -> _FakeLLMResponse:
        return _FakeLLMResponse(
            (
                '{"schema_version":"1.0","header":{"name":"A","title":"B","location":"C",'
                '"phone":"D","email":"E","links":{"linkedin":"L","github":"G"}},'
                '"summary":"S","skills":[{"group_name":"Core","items":["Python"]}],'
                '"experience":[{"company":"X","title":"Y","location":"Z","dates":"2020 - Present",'
                '"bullets":["Built services"]}],"education":[{"degree":"BS","school":"Uni",'
                '"location":"Loc","dates":"2000 - 2004"}],"certifications":'
                '[{"name":"Cert","issuer":"Org","year":"2024"}]}'
            )
        )


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _start_server(app: Any) -> tuple[Any, threading.Thread, str]:
    port = _pick_free_port()
    config = uvicorn.Config(app=app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(80):
        if getattr(server, "started", False):
            break
        time.sleep(0.05)
    if not getattr(server, "started", False):
        raise RuntimeError("uvicorn_server_start_failed")
    return server, thread, f"http://127.0.0.1:{port}"


def _stop_server(server: Any, thread: threading.Thread) -> None:
    server.should_exit = True
    thread.join(timeout=5)


def _extract_result(payload: Any) -> dict[str, Any]:
    structured = getattr(payload, "structuredContent", None)
    if isinstance(structured, dict):
        result = structured.get("result")
        if isinstance(result, dict):
            return result
        return structured
    content = getattr(payload, "content", None)
    if isinstance(content, list):
        for item in content:
            text = getattr(item, "text", None)
            if isinstance(text, str) and text.strip():
                try:
                    parsed = json.loads(text)
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict):
                    result = parsed.get("result")
                    if isinstance(result, dict):
                        return result
                    return parsed
    raise RuntimeError("mcp_result_invalid")


def _candidate_mcp_urls(mcp_url: str) -> list[str]:
    base = mcp_url.rstrip("/")
    # FastMCP streamable_http_app route differs by SDK version:
    # older versions expose at the mount root, newer versions under `/mcp`.
    return [f"{base}/mcp", base]


def _mcp_list_tools(mcp_url: str) -> list[str]:
    async def _run(candidate_url: str) -> list[str]:
        async with streamable_http_client(candidate_url) as (
            read_stream,
            write_stream,
            _session_id,
        ):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.list_tools()
                tools = getattr(result, "tools", [])
                names: list[str] = []
                for tool in tools:
                    name = getattr(tool, "name", None)
                    if isinstance(name, str):
                        names.append(name)
                return names

    last_error: BaseException | None = None
    for candidate in _candidate_mcp_urls(mcp_url):
        try:
            return asyncio.run(_run(candidate))
        except BaseException as exc:  # noqa: BLE001
            last_error = exc
            continue
    if last_error is None:
        raise RuntimeError("mcp_tools_list_failed")
    raise RuntimeError(f"mcp_tools_list_failed:{last_error}") from last_error


def _mcp_call_tool(mcp_url: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    async def _run(candidate_url: str) -> dict[str, Any]:
        async with streamable_http_client(candidate_url) as (
            read_stream,
            write_stream,
            _session_id,
        ):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.call_tool(name, arguments)
                return _extract_result(result)

    last_error: BaseException | None = None
    for candidate in _candidate_mcp_urls(mcp_url):
        try:
            return asyncio.run(_run(candidate))
        except BaseException as exc:  # noqa: BLE001
            last_error = exc
            continue
    if last_error is None:
        raise RuntimeError(f"mcp_tool_call_failed:{name}")
    raise RuntimeError(f"mcp_tool_call_failed:{name}:{last_error}") from last_error


def _load_module(module_name: str, module_path: Path, service_root: Path):
    sys.path.insert(0, str(service_root))
    try:
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"unable to load module spec for {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        try:
            sys.path.remove(str(service_root))
        except ValueError:
            pass


def test_coder_mcp_tools_and_call() -> None:
    root = Path(__file__).resolve().parents[2]
    service_root = root / "services" / "coder"
    module = _load_module("coder_mcp_test_module", service_root / "app" / "mcp.py", service_root)

    mcp_app, session_manager = module.create_mcp_asgi_app(_FakeCoderProvider(), _FakeLogger())

    @asynccontextmanager
    async def lifespan(_app: Any):
        async with session_manager.run():
            yield

    app = FastAPI(lifespan=lifespan)
    app.mount(
        "/mcp/rpc",
        mcp_app,
    )
    server, thread, base_url = _start_server(app)
    try:
        tool_names = _mcp_list_tools(f"{base_url}/mcp/rpc")
        assert "generate_code" in tool_names
        payload = _mcp_call_tool(
            f"{base_url}/mcp/rpc",
            "generate_code",
            {"goal": "make hello.py"},
        )
        assert payload["files"][0]["path"] == "hello.py"
    finally:
        _stop_server(server, thread)


def test_tailor_mcp_tools_and_call() -> None:
    root = Path(__file__).resolve().parents[2]
    service_root = root / "services" / "tailor"
    module = _load_module("tailor_mcp_test_module", service_root / "app" / "mcp.py", service_root)

    mcp_app, session_manager = module.create_mcp_asgi_app(_FakeTailorProvider())

    @asynccontextmanager
    async def lifespan(_app: Any):
        async with session_manager.run():
            yield

    app = FastAPI(lifespan=lifespan)
    app.mount("/mcp/rpc", mcp_app)
    server, thread, base_url = _start_server(app)
    try:
        tool_names = _mcp_list_tools(f"{base_url}/mcp/rpc")
        assert "tailor_resume" in tool_names
        payload = _mcp_call_tool(
            f"{base_url}/mcp/rpc",
            "tailor_resume",
            {
                "job": {
                    "context_json": {
                        "candidate_resume": "candidate",
                        "job_description": "job",
                        "target_role_name": "role",
                        "seniority_level": "Senior",
                    }
                }
            },
        )
        assert payload["tailored_resume"]["schema_version"] == "1.0"
    finally:
        _stop_server(server, thread)
