from __future__ import annotations

import asyncio
import inspect
import json
import logging
import math
import multiprocessing as mp
import os
import re
import time
import traceback
from typing import Any, Callable

from libs.framework.tool_runtime import ToolExecutionError


def extract_mcp_error_phase(error_text: str) -> str | None:
    match = re.search(r"(?:^|[;,\s])phase=([a-zA-Z0-9_.:/-]+)", error_text or "")
    if not match:
        return None
    return match.group(1)


def resolve_mcp_timeout_s() -> float:
    for key in ("MCP_TOOL_TIMEOUT_S", "MCP_TIMEOUT_S"):
        env_timeout = os.getenv(key)
        if not env_timeout:
            continue
        try:
            return max(1.0, float(env_timeout))
        except ValueError:
            return 45.0
    for key in ("TAILOR_OPENAI_TIMEOUT_S", "TAILOR_EVAL_OPENAI_TIMEOUT_S", "OPENAI_TIMEOUT_S"):
        env_timeout = os.getenv(key)
        if not env_timeout:
            continue
        try:
            # Avoid hanging indefinitely when global model timeout is very large.
            return min(180.0, max(5.0, float(env_timeout)))
        except ValueError:
            continue
    return 45.0


def resolve_mcp_outer_timeout_headroom_s() -> float:
    env_headroom = os.getenv("MCP_TOOL_OUTER_TIMEOUT_HEADROOM_S")
    if env_headroom:
        try:
            return max(0.0, min(120.0, float(env_headroom)))
        except ValueError:
            return 15.0
    return 15.0


def resolve_mcp_tool_timeout_s() -> int:
    timeout_s = resolve_mcp_timeout_s() + resolve_mcp_outer_timeout_headroom_s()
    return max(1, int(math.ceil(timeout_s)))


def resolve_mcp_max_retries() -> int:
    env_retries = os.getenv("MCP_TOOL_MAX_RETRIES")
    if env_retries:
        try:
            return max(0, int(env_retries))
        except ValueError:
            return 1
    return 1


def resolve_mcp_retry_sleep_s() -> float:
    env_sleep = os.getenv("MCP_TOOL_RETRY_SLEEP_S")
    if env_sleep:
        try:
            return max(0.0, float(env_sleep))
        except ValueError:
            return 0.25
    return 0.25


def resolve_mcp_first_attempt_reserve_s(timeout_s: float) -> float:
    env_reserve = os.getenv("MCP_FIRST_ATTEMPT_RESERVE_S")
    if env_reserve:
        try:
            return max(0.05, float(env_reserve))
        except ValueError:
            return min(30.0, max(5.0, timeout_s * 0.1))
    return min(30.0, max(5.0, timeout_s * 0.1))


def resolve_mcp_transport_timeout_s(timeout_s: float) -> float:
    env_timeout = os.getenv("MCP_TRANSPORT_TIMEOUT_S")
    if env_timeout:
        try:
            return max(1.0, float(env_timeout))
        except ValueError:
            return timeout_s
    return timeout_s


def resolve_mcp_isolation_mode() -> str:
    mode = os.getenv("MCP_TOOL_ISOLATION_MODE", "process").strip().lower()
    return "process" if mode == "process" else "thread"


def streamable_http_client_kwargs(
    client_factory: Callable[..., Any], timeout_s: float
) -> dict[str, Any]:
    """Best-effort timeout kwargs for different MCP SDK versions."""
    transport_timeout = resolve_mcp_transport_timeout_s(timeout_s)
    connect_timeout = min(10.0, transport_timeout)
    try:
        params = inspect.signature(client_factory).parameters
    except (TypeError, ValueError):
        return {}
    candidates: dict[str, Any] = {
        "timeout": transport_timeout,
        "request_timeout": transport_timeout,
        "read_timeout": transport_timeout,
        "write_timeout": transport_timeout,
        "connect_timeout": connect_timeout,
        "sse_read_timeout": transport_timeout,
        "http_timeout": transport_timeout,
    }
    return {name: value for name, value in candidates.items() if name in params}


def is_retryable_mcp_error(message: str) -> bool:
    lower = message.lower()
    return (
        lower.startswith("mcp_sdk_error:")
        or lower.startswith("mcp_sdk_timeout:")
        or "session terminated" in lower
        or "timeout" in lower
    )


def post_mcp_tool_call(
    service_url: str,
    tool_name: str,
    arguments: dict[str, Any],
    *,
    call_mcp_tool_sdk: Callable[[str, str, dict[str, Any], float], dict[str, Any]],
    classify_tool_error: Callable[[str], str],
    logger: logging.Logger,
    tracing_module: Any,
) -> dict[str, Any]:
    base_url = service_url.rstrip("/")
    attempts = ("/mcp/rpc/mcp", "/mcp/rpc")
    timeout_s = resolve_mcp_timeout_s()
    max_retries = resolve_mcp_max_retries()
    retry_sleep_s = resolve_mcp_retry_sleep_s()
    isolation_mode = resolve_mcp_isolation_mode()
    first_attempt_reserve_s = resolve_mcp_first_attempt_reserve_s(timeout_s)
    started_at = time.monotonic()
    deadline = started_at + timeout_s
    total_slots = len(attempts) * (max_retries + 1)
    prompt_version = os.getenv("PROMPT_VERSION", "unknown")
    policy_version = os.getenv("POLICY_VERSION", "unknown")
    tool_version = os.getenv("TOOL_VERSION", "unknown")
    errors: list[str] = []
    common_context = {
        "tool_name": tool_name,
        "service_url": base_url,
        "mcp_timeout_s": timeout_s,
        "max_retries": max_retries,
        "mcp_isolation_mode": isolation_mode,
        "prompt_version": prompt_version,
        "policy_version": policy_version,
        "tool_version": tool_version,
    }

    def _log_mcp_event(level: str, event_name: str, payload: dict[str, Any]) -> None:
        full_payload = {**common_context, **payload}
        message = f"{event_name} payload=%s"
        encoded = json.dumps(full_payload, ensure_ascii=True, sort_keys=True)
        if level == "warning":
            logger.warning(message, encoded, extra=full_payload)
            return
        if level == "error":
            logger.error(message, encoded, extra=full_payload)
            return
        logger.info(message, encoded, extra=full_payload)

    with tracing_module.start_span(
        "tool_registry.mcp_tool_call",
        attributes={
            "mcp.tool_name": tool_name,
            "mcp.service_url": base_url,
            "mcp.timeout_s": timeout_s,
            "mcp.max_retries": max_retries,
        },
    ) as span:

        def _finalize_and_raise(error_text: str, status: str) -> None:
            elapsed_ms = int(max(0.0, time.monotonic() - started_at) * 1000)
            payload = {
                "status": status,
                "error": error_text,
                "error_code": classify_tool_error(error_text),
                "elapsed_ms": elapsed_ms,
                "attempt_errors": list(errors),
            }
            _log_mcp_event("error", "mcp_call_failed", payload)
            tracing_module.set_span_attributes(
                span,
                {
                    "mcp.status": status,
                    "mcp.error": error_text,
                    "mcp.elapsed_s": max(0.0, time.monotonic() - started_at),
                },
            )
            raise ToolExecutionError(error_text)

        for path_idx, mcp_path in enumerate(attempts):
            mcp_url = f"{base_url}{mcp_path}"
            with tracing_module.start_span(
                "tool_registry.mcp_route_attempt",
                attributes={
                    "mcp.tool_name": tool_name,
                    "mcp.path": mcp_path,
                    "mcp.url": mcp_url,
                },
            ) as route_span:
                for attempt in range(max_retries + 1):
                    with tracing_module.start_span(
                        "tool_registry.mcp_route_attempt_try",
                        attributes={
                            "mcp.tool_name": tool_name,
                            "mcp.path": mcp_path,
                            "mcp.url": mcp_url,
                            "mcp.route_attempt": attempt + 1,
                            "mcp.route_attempts_total": max_retries + 1,
                        },
                    ) as attempt_span:
                        slot_idx = path_idx * (max_retries + 1) + attempt
                        remaining_s = deadline - time.monotonic()
                        remaining_slots_after_current = max(0, total_slots - slot_idx - 1)
                        attempt_started_at = time.monotonic()
                        start_payload = {
                            "mcp_url": mcp_url,
                            "route_path": mcp_path,
                            "attempt": attempt + 1,
                            "attempts_total": max_retries + 1,
                            "deadline_remaining_ms": int(max(0.0, remaining_s) * 1000),
                        }
                        _log_mcp_event("info", "mcp_attempt_start", start_payload)
                        if remaining_s <= 0.05:
                            detail = f"mcp_call_timed_out_after_{timeout_s:.1f}s"
                            timeout_error = f"mcp_sdk_timeout:{detail}"
                            _log_mcp_event(
                                "warning",
                                "mcp_attempt_timeout",
                                {
                                    **start_payload,
                                    "error": detail,
                                    "error_code": classify_tool_error(timeout_error),
                                },
                            )
                            tracing_module.set_span_attributes(
                                attempt_span,
                                {
                                    "mcp.route_status": "failed",
                                    "mcp.route_error": timeout_error,
                                },
                            )
                            tracing_module.set_span_attributes(
                                route_span,
                                {"mcp.route_status": "failed", "mcp.route_error": timeout_error},
                            )
                            _finalize_and_raise(timeout_error, "timeout")

                        if slot_idx == 0 and remaining_slots_after_current > 0:
                            reserve_s = min(
                                max(0.05, first_attempt_reserve_s),
                                max(0.0, remaining_s - 0.05),
                            )
                            per_attempt_timeout_s = max(0.05, remaining_s - reserve_s)
                            timeout_allocation = "first_attempt_bias"
                        else:
                            slots_including_current = remaining_slots_after_current + 1
                            per_attempt_timeout_s = max(0.05, remaining_s / slots_including_current)
                            timeout_allocation = "equal_remaining"
                        try:
                            tracing_module.set_span_attributes(
                                attempt_span,
                                {
                                    "mcp.route_timeout_s": per_attempt_timeout_s,
                                    "mcp.deadline_remaining_s": remaining_s,
                                    "mcp.timeout_allocation": timeout_allocation,
                                    "mcp.remaining_slots_after_current": remaining_slots_after_current,
                                },
                            )
                            result = call_mcp_tool_sdk(
                                mcp_url,
                                tool_name,
                                arguments,
                                timeout_s=per_attempt_timeout_s,
                            )
                            elapsed_ms = int(max(0.0, time.monotonic() - attempt_started_at) * 1000)
                            _log_mcp_event(
                                "info",
                                "mcp_attempt_success",
                                {
                                    **start_payload,
                                    "route": "mcp_sdk",
                                    "attempt_timeout_ms": int(per_attempt_timeout_s * 1000),
                                    "attempt_elapsed_ms": elapsed_ms,
                                    "timeout_allocation": timeout_allocation,
                                    "deadline_remaining_ms": int(
                                        max(0.0, deadline - time.monotonic()) * 1000
                                    ),
                                },
                            )
                            tracing_module.set_span_attributes(
                                attempt_span, {"mcp.route_status": "ok"}
                            )
                            tracing_module.set_span_attributes(route_span, {"mcp.route_status": "ok"})
                            tracing_module.set_span_attributes(
                                span,
                                {
                                    "mcp.selected_path": mcp_path,
                                    "mcp.status": "ok",
                                    "mcp.elapsed_s": time.monotonic() - started_at,
                                },
                            )
                            return result
                        except Exception as exc:  # noqa: BLE001
                            if isinstance(exc, ToolExecutionError):
                                error_text = str(exc)
                            else:
                                error_detail = "; ".join(flatten_exception_messages(exc))
                                error_text = (
                                    "mcp_sdk_error:"
                                    f"phase=route_call;error_type={exc.__class__.__name__};{error_detail}"
                                )
                            error_code = classify_tool_error(error_text)
                            error_phase = extract_mcp_error_phase(error_text)
                            if error_text.startswith("mcp_tool_error"):
                                _log_mcp_event(
                                    "warning",
                                    "mcp_attempt_tool_error",
                                    {
                                        **start_payload,
                                        "error": error_text,
                                        "error_code": error_code,
                                        "error_phase": error_phase,
                                    },
                                )
                                tracing_module.set_span_attributes(
                                    attempt_span, {"mcp.route_status": "tool_error"}
                                )
                                tracing_module.set_span_attributes(
                                    route_span, {"mcp.route_status": "tool_error"}
                                )
                                _finalize_and_raise(error_text, "tool_error")
                            retryable = is_retryable_mcp_error(error_text)
                            if retryable and attempt < max_retries:
                                _log_mcp_event(
                                    "info",
                                    "mcp_attempt_retrying",
                                    {
                                        **start_payload,
                                        "error": error_text,
                                        "error_code": error_code,
                                        "error_phase": error_phase,
                                        "retry_sleep_s": retry_sleep_s * float(attempt + 1),
                                        "timeout_allocation": timeout_allocation,
                                        "deadline_remaining_ms": int(
                                            max(0.0, deadline - time.monotonic()) * 1000
                                        ),
                                    },
                                )
                                tracing_module.set_span_attributes(
                                    attempt_span,
                                    {
                                        "mcp.route_status": "retrying",
                                        "mcp.route_error": error_text,
                                    },
                                )
                                tracing_module.set_span_attributes(
                                    route_span,
                                    {
                                        "mcp.route_status": "retrying",
                                        "mcp.route_error": error_text,
                                    },
                                )
                                if retry_sleep_s > 0:
                                    next_remaining_s = deadline - time.monotonic()
                                    if next_remaining_s <= 0:
                                        detail = f"mcp_call_timed_out_after_{timeout_s:.1f}s"
                                        _finalize_and_raise(f"mcp_sdk_timeout:{detail}", "timeout")
                                    sleep_s = min(
                                        retry_sleep_s * float(attempt + 1),
                                        max(0.0, next_remaining_s - 0.05),
                                    )
                                    if sleep_s > 0:
                                        time.sleep(sleep_s)
                                continue
                            attempt_elapsed_ms = int(
                                max(0.0, time.monotonic() - attempt_started_at) * 1000
                            )
                            failed_payload = {
                                **start_payload,
                                "error": error_text,
                                "error_code": error_code,
                                "error_phase": error_phase,
                                "retryable": retryable,
                                "attempt_timeout_ms": int(per_attempt_timeout_s * 1000),
                                "attempt_elapsed_ms": attempt_elapsed_ms,
                                "timeout_allocation": timeout_allocation,
                                "deadline_remaining_ms": int(
                                    max(0.0, deadline - time.monotonic()) * 1000
                                ),
                            }
                            tracing_module.set_span_attributes(
                                attempt_span,
                                {
                                    "mcp.route_status": "failed",
                                    "mcp.route_error": error_text,
                                },
                            )
                            tracing_module.set_span_attributes(
                                route_span,
                                {
                                    "mcp.route_status": "failed",
                                    "mcp.route_error": error_text,
                                },
                            )
                            _log_mcp_event("warning", "mcp_attempt_failed", failed_payload)
                            errors.append(f"{mcp_path}#{attempt + 1}:{error_text}")
                            break
        joined = " | ".join(errors) if errors else "no_route_errors_recorded"
        _finalize_and_raise(f"mcp_sdk_all_routes_failed:{joined}", "all_routes_failed")


def call_mcp_tool_sdk(
    mcp_url: str,
    tool_name: str,
    arguments: dict[str, Any],
    timeout_s: float,
    *,
    tracing_module: Any,
    logger: logging.Logger,
) -> dict[str, Any]:
    mode = resolve_mcp_isolation_mode()
    if mode == "process":
        return call_mcp_tool_sdk_process(
            mcp_url,
            tool_name,
            arguments,
            timeout_s,
            logger=logger,
            tracing_module=tracing_module,
        )
    return call_mcp_tool_sdk_inproc(
        mcp_url,
        tool_name,
        arguments,
        timeout_s,
        tracing_module=tracing_module,
    )


def call_mcp_tool_sdk_inproc(
    mcp_url: str,
    tool_name: str,
    arguments: dict[str, Any],
    timeout_s: float,
    *,
    tracing_module: Any,
) -> dict[str, Any]:
    try:
        from mcp import ClientSession
        from mcp.client.streamable_http import streamable_http_client
    except Exception as exc:  # noqa: BLE001
        raise ToolExecutionError(f"mcp_sdk_unavailable:{exc}") from exc

    streamable_kwargs = streamable_http_client_kwargs(streamable_http_client, timeout_s)
    with tracing_module.start_span(
        "tool_registry.mcp_sdk_call",
        attributes={
            "mcp.tool_name": tool_name,
            "mcp.url": mcp_url,
            "mcp.timeout_s": timeout_s,
        },
    ) as sdk_span:
        sdk_started_at = time.monotonic()
        phase = "stream_open"

        async def _run_call() -> Any:
            nonlocal phase
            async with streamable_http_client(mcp_url, **streamable_kwargs) as (
                read_stream,
                write_stream,
                _session_id,
            ):
                async with ClientSession(read_stream, write_stream) as session:
                    phase = "initialize"
                    with tracing_module.start_span(
                        "tool_registry.mcp_sdk_initialize",
                        attributes={
                            "mcp.tool_name": tool_name,
                            "mcp.url": mcp_url,
                        },
                    ):
                        await session.initialize()
                    phase = "call_tool"
                    with tracing_module.start_span(
                        "tool_registry.mcp_sdk_call_tool",
                        attributes={
                            "mcp.tool_name": tool_name,
                            "mcp.url": mcp_url,
                        },
                    ):
                        return await session.call_tool(tool_name, arguments)

        try:
            result = asyncio.run(asyncio.wait_for(_run_call(), timeout=timeout_s))
        except TimeoutError as exc:
            elapsed_s = time.monotonic() - sdk_started_at
            detail = (
                f"phase={phase};mcp_call_timed_out_after_{timeout_s:.1f}s;elapsed_s={elapsed_s:.3f}"
            )
            tracing_module.set_span_attributes(
                sdk_span,
                {
                    "mcp.error": detail,
                    "mcp.phase": phase,
                    "mcp.elapsed_s": elapsed_s,
                },
            )
            raise ToolExecutionError(f"mcp_sdk_timeout:{detail}") from exc
        except Exception as exc:  # noqa: BLE001
            elapsed_s = time.monotonic() - sdk_started_at
            detail = "; ".join(flatten_exception_messages(exc))
            error_detail = f"phase={phase};error_type={exc.__class__.__name__};{detail}"
            tracing_module.set_span_attributes(
                sdk_span,
                {
                    "mcp.error": error_detail,
                    "mcp.phase": phase,
                    "mcp.elapsed_s": elapsed_s,
                },
            )
            raise ToolExecutionError(f"mcp_sdk_error:{error_detail}") from exc
        tracing_module.set_span_attributes(sdk_span, {"mcp.status": "ok"})
        return extract_mcp_sdk_result(result)


def mcp_process_entry(
    queue: Any,
    mcp_url: str,
    tool_name: str,
    arguments: dict[str, Any],
    timeout_s: float,
) -> None:
    try:
        result = call_mcp_tool_sdk_inproc(
            mcp_url,
            tool_name,
            arguments,
            timeout_s,
            tracing_module=_NoopTracing,
        )
        queue.put({"ok": result})
    except Exception as exc:  # noqa: BLE001
        queue.put(
            {
                "error": str(exc),
                "error_type": exc.__class__.__name__,
                "traceback": traceback.format_exc(),
            }
        )


def call_mcp_tool_sdk_process(
    mcp_url: str,
    tool_name: str,
    arguments: dict[str, Any],
    timeout_s: float,
    *,
    logger: logging.Logger,
    tracing_module: Any,
) -> dict[str, Any]:
    del tracing_module  # Process worker uses no-op tracing.
    ctx = mp.get_context("spawn")
    queue = ctx.Queue(maxsize=1)
    process = ctx.Process(
        target=mcp_process_entry,
        args=(queue, mcp_url, tool_name, arguments, timeout_s),
        daemon=True,
    )
    process.start()
    process.join(timeout=timeout_s + 2.0)
    if process.is_alive():
        process.terminate()
        process.join(timeout=1.0)
        detail = f"phase=process.join;mcp_call_timed_out_after_{timeout_s:.1f}s"
        raise ToolExecutionError(f"mcp_sdk_timeout:{detail}")

    outcome: dict[str, Any] | None = None
    try:
        if not queue.empty():
            outcome = queue.get_nowait()
    except Exception:  # noqa: BLE001
        outcome = None
    finally:
        queue.close()

    if isinstance(outcome, dict):
        if isinstance(outcome.get("ok"), dict):
            return outcome["ok"]
        if isinstance(outcome.get("error"), str):
            detail = outcome["error"]
            error_type = outcome.get("error_type")
            if isinstance(error_type, str) and error_type.strip():
                detail = f"{detail};child_error_type={error_type}"
            child_traceback = outcome.get("traceback")
            if isinstance(child_traceback, str) and child_traceback.strip():
                logger.warning(
                    "mcp_child_process_error_traceback tool=%s url=%s traceback=%s",
                    tool_name,
                    mcp_url,
                    child_traceback,
                )
            raise ToolExecutionError(detail)

    exit_code = process.exitcode
    raise ToolExecutionError(f"mcp_sdk_error:process_exit_code_{exit_code}")


def flatten_exception_messages(exc: BaseException) -> list[str]:
    messages = [str(exc)]
    nested = getattr(exc, "exceptions", None)
    if isinstance(nested, (list, tuple)):
        for child in nested:
            if isinstance(child, BaseException):
                messages.extend(flatten_exception_messages(child))
    deduped: list[str] = []
    seen: set[str] = set()
    for message in messages:
        normalized = message.strip()
        if not normalized:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped or [exc.__class__.__name__]


def extract_mcp_sdk_result(result: Any) -> dict[str, Any]:
    is_error = getattr(result, "isError", False)
    if is_error:
        error_detail = extract_mcp_error_detail(result)
        if error_detail:
            raise ToolExecutionError(f"mcp_tool_error:{error_detail}")
        raise ToolExecutionError("mcp_tool_error")

    structured = getattr(result, "structuredContent", None)
    if isinstance(structured, dict):
        return normalize_mcp_structured_result(structured)

    content = getattr(result, "content", None)
    if isinstance(content, list):
        for item in content:
            text = getattr(item, "text", None)
            if isinstance(text, str) and text.strip():
                try:
                    parsed = json.loads(text)
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict):
                    return parsed

    raise ToolExecutionError("mcp_sdk_result_invalid")


def normalize_mcp_structured_result(structured: dict[str, Any]) -> dict[str, Any]:
    # FastMCP wraps tool outputs as {"result": <tool_output>}.
    # Internal callers expect the raw tool_output dictionary.
    result_value = structured.get("result")
    if isinstance(result_value, dict):
        return result_value
    return structured


def extract_mcp_error_detail(result: Any) -> str:
    structured = getattr(result, "structuredContent", None)
    if isinstance(structured, dict):
        return json.dumps(structured, ensure_ascii=True)
    content = getattr(result, "content", None)
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            text = getattr(item, "text", None)
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
        if parts:
            return " | ".join(parts)
    return ""


class _NoopSpan:
    def __enter__(self) -> "_NoopSpan":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        return None


class _NoopTracing:
    @staticmethod
    def start_span(name: str, attributes: dict[str, Any] | None = None) -> _NoopSpan:
        del name, attributes
        return _NoopSpan()

    @staticmethod
    def set_span_attributes(span: Any, attributes: dict[str, Any]) -> None:
        del span, attributes
        return None
