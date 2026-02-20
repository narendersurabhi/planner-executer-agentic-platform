from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Callable
from typing import Any, Dict

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.server import TransportSecuritySettings

from libs.core import logging as core_logging
from tailor_core import (
    TailorError,
    improve_resume as run_improve_resume,
    improve_resume_iterative as run_improve_resume_iterative,
    tailor_resume as run_tailor_resume,
)

LOGGER = core_logging.get_logger("tailor")


@dataclass
class _IterativeDedupeEntry:
    status: str  # in_progress | done | failed
    started_at: float
    updated_at: float
    event: threading.Event = field(default_factory=threading.Event)
    result: Dict[str, Any] | None = None
    error: str | None = None


_ITERATIVE_DEDUPE_LOCK = threading.Lock()
_ITERATIVE_DEDUPE_CACHE: dict[str, _IterativeDedupeEntry] = {}


def _parse_optional_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _parse_optional_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _resolve_iterative_cache_ttl_s() -> float:
    configured = _parse_optional_float(os.getenv("TAILOR_ITERATIVE_CACHE_TTL_S"))
    if configured is None:
        return 1800.0
    return max(1.0, configured)


def _resolve_iterative_inflight_wait_s() -> float:
    configured = _parse_optional_float(os.getenv("TAILOR_ITERATIVE_INFLIGHT_WAIT_S"))
    if configured is None:
        return 30.0
    return max(0.0, configured)


def _resolve_iterative_cache_max_entries() -> int:
    configured = _parse_optional_int(os.getenv("TAILOR_ITERATIVE_CACHE_MAX_ENTRIES"))
    if configured is None:
        return 256
    return max(16, configured)


def _iterative_request_key(
    *,
    tailored_resume: Dict[str, Any] | None,
    tailored_text: str | None,
    job: Dict[str, Any] | None,
    min_alignment_score: float,
    max_iterations: int,
) -> str:
    payload = {
        "tailored_resume": tailored_resume,
        "tailored_text": tailored_text,
        "job": job or {},
        "min_alignment_score": float(min_alignment_score),
        "max_iterations": int(max_iterations),
    }
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _prune_iterative_dedupe_cache(now: float) -> None:
    ttl_s = _resolve_iterative_cache_ttl_s()
    max_entries = _resolve_iterative_cache_max_entries()
    stale_before = now - ttl_s
    stale_in_progress_before = now - max(ttl_s * 2.0, 900.0)
    stale_keys: list[str] = []
    for key, entry in _ITERATIVE_DEDUPE_CACHE.items():
        if entry.status == "in_progress":
            if entry.started_at < stale_in_progress_before:
                stale_keys.append(key)
            continue
        if entry.updated_at < stale_before:
            stale_keys.append(key)
    for key in stale_keys:
        _ITERATIVE_DEDUPE_CACHE.pop(key, None)
    if len(_ITERATIVE_DEDUPE_CACHE) <= max_entries:
        return
    evictable = sorted(
        (
            (key, entry.updated_at)
            for key, entry in _ITERATIVE_DEDUPE_CACHE.items()
            if entry.status != "in_progress"
        ),
        key=lambda item: item[1],
    )
    overflow = len(_ITERATIVE_DEDUPE_CACHE) - max_entries
    for key, _ in evictable[:overflow]:
        _ITERATIVE_DEDUPE_CACHE.pop(key, None)


def _run_improve_iterative_with_dedupe(
    *,
    key: str,
    job_id: str,
    compute: Callable[[], Dict[str, Any]],
) -> Dict[str, Any]:
    key_prefix = key[:12]
    now = time.monotonic()
    owner_entry: _IterativeDedupeEntry | None = None
    wait_event: threading.Event | None = None
    with _ITERATIVE_DEDUPE_LOCK:
        _prune_iterative_dedupe_cache(now)
        entry = _ITERATIVE_DEDUPE_CACHE.get(key)
        if entry is not None:
            if entry.status == "done" and isinstance(entry.result, dict):
                LOGGER.info(
                    "mcp_improve_iterative_cache_hit",
                    job_id=job_id,
                    key_prefix=key_prefix,
                    age_ms=int(max(0.0, now - entry.updated_at) * 1000),
                )
                return entry.result
            if entry.status == "in_progress":
                wait_event = entry.event
                LOGGER.info(
                    "mcp_improve_iterative_cache_wait",
                    job_id=job_id,
                    key_prefix=key_prefix,
                    inflight_age_ms=int(max(0.0, now - entry.started_at) * 1000),
                    wait_s=float(_resolve_iterative_inflight_wait_s()),
                )
            else:
                _ITERATIVE_DEDUPE_CACHE.pop(key, None)
                entry = None
        if entry is None:
            owner_entry = _IterativeDedupeEntry(status="in_progress", started_at=now, updated_at=now)
            _ITERATIVE_DEDUPE_CACHE[key] = owner_entry
            LOGGER.info(
                "mcp_improve_iterative_cache_miss",
                job_id=job_id,
                key_prefix=key_prefix,
            )

    if owner_entry is None and wait_event is not None:
        wait_s = _resolve_iterative_inflight_wait_s()
        if wait_s > 0:
            wait_event.wait(timeout=wait_s)
        with _ITERATIVE_DEDUPE_LOCK:
            entry = _ITERATIVE_DEDUPE_CACHE.get(key)
            if entry is not None and entry.status == "done" and isinstance(entry.result, dict):
                LOGGER.info(
                    "mcp_improve_iterative_cache_replay",
                    job_id=job_id,
                    key_prefix=key_prefix,
                )
                return entry.result
        LOGGER.info(
            "mcp_improve_iterative_cache_wait_expired",
            job_id=job_id,
            key_prefix=key_prefix,
        )
        return compute()

    try:
        result = compute()
    except Exception as exc:  # noqa: BLE001
        with _ITERATIVE_DEDUPE_LOCK:
            entry = _ITERATIVE_DEDUPE_CACHE.get(key)
            if entry is owner_entry and entry is not None:
                entry.status = "failed"
                entry.error = str(exc)
                entry.updated_at = time.monotonic()
                entry.event.set()
        raise

    with _ITERATIVE_DEDUPE_LOCK:
        entry = _ITERATIVE_DEDUPE_CACHE.get(key)
        if entry is owner_entry and entry is not None:
            entry.status = "done"
            entry.result = result
            entry.updated_at = time.monotonic()
            entry.event.set()
    return result


def _extract_job_id(job: Dict[str, Any] | None) -> str:
    if not isinstance(job, dict):
        return ""
    job_id = job.get("id")
    if isinstance(job_id, str) and job_id.strip():
        return job_id.strip()
    context = job.get("context_json")
    if isinstance(context, dict):
        context_job_id = context.get("id")
        if isinstance(context_job_id, str) and context_job_id.strip():
            return context_job_id.strip()
    return ""


def create_mcp_asgi_app(provider: Any, evaluator: Dict[str, Any] | None = None):
    default_hosts = [
        "tailor",
        "tailor:8000",
        "localhost",
        "localhost:8000",
        "localhost:*",
        "127.0.0.1",
        "127.0.0.1:8000",
        "127.0.0.1:*",
    ]
    raw_allowed_hosts = os.getenv("MCP_ALLOWED_HOSTS", "")
    allowed_hosts = [h.strip() for h in raw_allowed_hosts.split(",") if h.strip()] or default_hosts
    mcp = FastMCP(
        "agentic-tailor",
        transport_security=TransportSecuritySettings(allowed_hosts=allowed_hosts),
    )

    @mcp.tool()
    def tailor_resume(job: Dict[str, Any], memory: Dict[str, Any] | None = None) -> Dict[str, Any]:
        try:
            return {"tailored_resume": run_tailor_resume(job or {}, memory, provider)}
        except TailorError as exc:
            raise RuntimeError(exc.detail) from exc

    @mcp.tool()
    def improve_resume(
        tailored_resume: Dict[str, Any] | None = None,
        tailored_text: str | None = None,
        job: Dict[str, Any] | None = None,
        memory: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        try:
            return run_improve_resume(
                tailored_resume=tailored_resume,
                tailored_text=tailored_text,
                job=job or {},
                memory=memory,
                provider=provider,
                evaluator=evaluator,
            )
        except TailorError as exc:
            raise RuntimeError(exc.detail) from exc

    @mcp.tool()
    def improve_iterative(
        tailored_resume: Dict[str, Any] | None = None,
        tailored_text: str | None = None,
        job: Dict[str, Any] | None = None,
        memory: Dict[str, Any] | None = None,
        min_alignment_score: float = 85,
        max_iterations: int = 2,
    ) -> Dict[str, Any]:
        started_at = time.monotonic()
        job_payload = job or {}
        job_id = _extract_job_id(job_payload)
        request_key = _iterative_request_key(
            tailored_resume=tailored_resume,
            tailored_text=tailored_text,
            job=job_payload,
            min_alignment_score=min_alignment_score,
            max_iterations=max_iterations,
        )
        LOGGER.info(
            "mcp_improve_iterative_started",
            job_id=job_id,
            min_alignment_score=float(min_alignment_score),
            max_iterations=int(max_iterations),
            has_tailored_resume=bool(tailored_resume),
            has_tailored_text=bool((tailored_text or "").strip()),
            request_key_prefix=request_key[:12],
        )
        try:
            result = _run_improve_iterative_with_dedupe(
                key=request_key,
                job_id=job_id,
                compute=lambda: run_improve_resume_iterative(
                    tailored_resume=tailored_resume,
                    tailored_text=tailored_text,
                    job=job_payload,
                    memory=memory,
                    min_alignment_score=min_alignment_score,
                    max_iterations=max_iterations,
                    provider=provider,
                    evaluator=evaluator,
                ),
            )
            LOGGER.info(
                "mcp_improve_iterative_finished",
                job_id=job_id,
                alignment_score=float(result.get("alignment_score", 0.0)),
                iterations=int(result.get("iterations", 0)),
                reached_threshold=bool(result.get("reached_threshold", False)),
                duration_ms=int(max(0.0, time.monotonic() - started_at) * 1000),
            )
            return result
        except TailorError as exc:
            LOGGER.warning(
                "mcp_improve_iterative_failed",
                job_id=job_id,
                error=exc.detail,
                duration_ms=int(max(0.0, time.monotonic() - started_at) * 1000),
            )
            raise RuntimeError(exc.detail) from exc

    mcp_app = mcp.streamable_http_app()
    session_manager = mcp_app.routes[0].app.session_manager
    return mcp_app, session_manager
