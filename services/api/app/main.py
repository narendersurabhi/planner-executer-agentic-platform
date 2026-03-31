from __future__ import annotations

from contextlib import asynccontextmanager
import difflib
import json
import os
import logging
import hashlib
import re
import threading
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request as UrlRequest, urlopen

import redis
from fastapi import Body, Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from prometheus_client import Counter, make_asgi_app
from sqlalchemy.orm import Session

from libs.core import (
    capability_search,
    capability_registry,
    chat_contracts,
    document_store,
    execution_contracts,
    events,
    feedback_eval,
    intent_contract,
    logging as core_logging,
    mcp_gateway,
    models,
    payload_resolver,
    planner_contracts,
    run_specs,
    runtime_manifest,
    state_machine,
    tool_bootstrap,
    workflow_contracts,
)
from libs.core.llm_provider import (
    LLMProvider,
    LLMProviderError,
    LLMRequest,
    MockLLMProvider,
    resolve_provider,
)
from .database import Base, SessionLocal, engine
from .models import (
    ChatMessageRecord,
    ChatSessionRecord,
    EventOutboxRecord,
    InvocationRecord,
    JobRecord,
    PlanRecord,
    RunEventRecord,
    StepAttemptRecord,
    TaskRecord,
    TaskResultRecord,
    WorkflowDefinitionRecord,
    WorkflowRunRecord,
    WorkflowTriggerRecord,
    WorkflowVersionRecord,
)
from . import (
    chat_clarification_normalizer,
    chat_execution_service,
    chat_service,
    context_service,
    dispatch_service,
    feedback_service,
    intent_service,
    memory_promotion_service,
    memory_store,
)

core_logging.configure_logging("api")
logger = logging.getLogger("api.orchestrator")


def _utcnow() -> datetime:
    return datetime.now(UTC)


@asynccontextmanager
async def _app_lifespan(_app: FastAPI):
    _init_db()
    yield


app = FastAPI(title="Agentic Workflow Studio API", lifespan=_app_lifespan)

cors_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "http://localhost:3002,http://localhost:3000").split(
        ","
    )
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


def _parse_confidence_threshold_map(
    raw: str | None,
    *,
    allowed_keys: set[str] | None = None,
) -> dict[str, float]:
    if not raw:
        return {}
    value = raw.strip()
    if not value:
        return {}
    parsed: dict[str, Any] = {}
    try:
        candidate = json.loads(value)
        if isinstance(candidate, Mapping):
            parsed = dict(candidate)
    except Exception:
        parsed = {}
    if not parsed:
        for part in value.split(","):
            piece = part.strip()
            if not piece:
                continue
            if "=" in piece:
                key, threshold = piece.split("=", 1)
            elif ":" in piece:
                key, threshold = piece.split(":", 1)
            else:
                continue
            parsed[key.strip()] = threshold.strip()
    normalized: dict[str, float] = {}
    for raw_key, raw_threshold in parsed.items():
        key = str(raw_key or "").strip().lower()
        if not key:
            continue
        if allowed_keys and key not in allowed_keys:
            continue
        try:
            threshold = float(raw_threshold)
        except (TypeError, ValueError):
            continue
        normalized[key] = max(0.0, min(1.0, threshold))
    return normalized


REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", "/shared/artifacts")
WORKSPACE_DIR = os.getenv("WORKSPACE_DIR", "/shared/workspace")
SCHEMA_REGISTRY_PATH = os.getenv("SCHEMA_REGISTRY_PATH", "/app/schemas")
ORCHESTRATOR_ENABLED = os.getenv("ORCHESTRATOR_ENABLED", "true").lower() == "true"
POLICY_GATE_ENABLED = os.getenv("POLICY_GATE_ENABLED", "false").lower() == "true"
JOB_RECOVERY_ENABLED = os.getenv("JOB_RECOVERY_ENABLED", "true").lower() == "true"
LLM_PROVIDER_NAME = os.getenv("LLM_PROVIDER", "").strip()
LLM_MODEL_NAME = os.getenv("OPENAI_MODEL", "").strip()
CHAT_ROUTER_MODEL = os.getenv("CHAT_ROUTER_MODEL", "").strip()
CHAT_RESPONSE_MODEL = os.getenv("CHAT_RESPONSE_MODEL", "").strip()
CHAT_PENDING_CORRECTION_MODEL = os.getenv("CHAT_PENDING_CORRECTION_MODEL", "").strip()
CHAT_CLARIFICATION_NORMALIZER_ENABLED = (
    os.getenv("CHAT_CLARIFICATION_NORMALIZER_ENABLED", "true").lower() == "true"
)
CHAT_CLARIFICATION_NORMALIZER_MODEL = os.getenv(
    "CHAT_CLARIFICATION_NORMALIZER_MODEL", ""
).strip()
CHAT_CLARIFICATION_NORMALIZER_CONFIDENCE_THRESHOLD = max(
    0.0,
    min(
        1.0,
        float(os.getenv("CHAT_CLARIFICATION_NORMALIZER_CONFIDENCE_THRESHOLD", "0.7")),
    ),
)
CHAT_RESPONSE_MODE = os.getenv("CHAT_RESPONSE_MODE", "answer_or_handoff").strip().lower()
if CHAT_RESPONSE_MODE not in {"answer_only", "answer_or_handoff"}:
    CHAT_RESPONSE_MODE = "answer_or_handoff"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com").strip()
OPENAI_TIMEOUT_S = float(os.getenv("OPENAI_TIMEOUT_S", "30"))
OPENAI_MAX_RETRIES = int(os.getenv("OPENAI_MAX_RETRIES", "0"))
COMPOSER_RECOMMENDER_MODEL = os.getenv("COMPOSER_RECOMMENDER_MODEL", "").strip()
COMPOSER_RECOMMENDER_ENABLED = os.getenv("COMPOSER_RECOMMENDER_ENABLED", "true").lower() == "true"
ORCHESTRATOR_RECOVER_PENDING = os.getenv("ORCHESTRATOR_RECOVER_PENDING", "true").lower() == "true"
ORCHESTRATOR_RECOVER_IDLE_MS = int(os.getenv("ORCHESTRATOR_RECOVER_IDLE_MS", "60000"))
REPLAN_MAX = int(os.getenv("REPLAN_MAX", "1"))
TOOL_INPUT_VALIDATION_ENABLED = os.getenv("TOOL_INPUT_VALIDATION_ENABLED", "true").lower() == "true"
INTENT_CLARIFICATION_ON_CREATE = (
    os.getenv("INTENT_CLARIFICATION_ON_CREATE", "false").lower() == "true"
)
INTENT_MIN_CONFIDENCE = float(os.getenv("INTENT_MIN_CONFIDENCE", "0.70"))
INTENT_MIN_CONFIDENCE_BY_INTENT = _parse_confidence_threshold_map(
    os.getenv("INTENT_MIN_CONFIDENCE_BY_INTENT", ""),
    allowed_keys={member.value for member in models.ToolIntent},
)
INTENT_MIN_CONFIDENCE_BY_RISK = _parse_confidence_threshold_map(
    os.getenv("INTENT_MIN_CONFIDENCE_BY_RISK", ""),
    allowed_keys={"read_only", "bounded_write", "high_risk_write"},
)
INTENT_CLARIFICATION_BLOCKING_SLOTS = {
    entry.strip().lower()
    for entry in os.getenv(
        "INTENT_CLARIFICATION_BLOCKING_SLOTS",
        "intent_action,output_format,target_system,safety_constraints",
    ).split(",")
    if entry.strip()
}
CHAT_PRE_SUBMIT_BLOCKING_SLOTS = {
    entry.strip().lower()
    for entry in os.getenv(
        "CHAT_PRE_SUBMIT_BLOCKING_SLOTS",
        "output_format,target_system,safety_constraints",
    ).split(",")
    if entry.strip()
}
INTENT_ASSESS_ENABLED = os.getenv("INTENT_ASSESS_ENABLED", "true").lower() == "true"
INTENT_ASSESS_MODE = os.getenv("INTENT_ASSESS_MODE", "heuristic").strip().lower()
if INTENT_ASSESS_MODE not in {"heuristic", "llm", "hybrid"}:
    INTENT_ASSESS_MODE = "heuristic"
INTENT_ASSESS_MODEL = os.getenv("INTENT_ASSESS_MODEL", "").strip()
INTENT_DECOMPOSE_ENABLED = os.getenv("INTENT_DECOMPOSE_ENABLED", "true").lower() == "true"
INTENT_DECOMPOSE_MODE = os.getenv("INTENT_DECOMPOSE_MODE", "heuristic").strip().lower()
if INTENT_DECOMPOSE_MODE not in {"heuristic", "llm", "hybrid"}:
    INTENT_DECOMPOSE_MODE = "heuristic"
INTENT_DECOMPOSE_MODEL = os.getenv("INTENT_DECOMPOSE_MODEL", "").strip()
INTENT_CAPABILITY_TOP_K = max(1, min(10, int(os.getenv("INTENT_CAPABILITY_TOP_K", "3"))))
INTENT_MEMORY_RETRIEVAL_ENABLED = (
    os.getenv("INTENT_MEMORY_RETRIEVAL_ENABLED", "true").lower() == "true"
)
INTENT_MEMORY_RETRIEVAL_LIMIT = max(
    1, min(10, int(os.getenv("INTENT_MEMORY_RETRIEVAL_LIMIT", "3")))
)
CHAT_DIRECT_EXECUTION_ENABLED = (
    os.getenv("CHAT_DIRECT_EXECUTION_ENABLED", "true").lower() == "true"
)
CHAT_INTENT_VECTOR_SEARCH_ENABLED = (
    os.getenv("CHAT_INTENT_VECTOR_SEARCH_ENABLED", "true").lower() == "true"
)
CHAT_INTENT_VECTOR_COLLECTION = os.getenv("CHAT_INTENT_VECTOR_COLLECTION", "").strip() or None
CHAT_INTENT_VECTOR_NAMESPACE_PREFIX = (
    os.getenv("CHAT_INTENT_VECTOR_NAMESPACE_PREFIX", "chat_intent_catalog").strip()
    or "chat_intent_catalog"
)
CHAT_INTENT_VECTOR_WORKSPACE_ID = (
    os.getenv("CHAT_INTENT_VECTOR_WORKSPACE_ID", "chat-intent-catalog").strip()
    or "chat-intent-catalog"
)
CHAT_INTENT_VECTOR_TOP_K = max(1, min(8, int(os.getenv("CHAT_INTENT_VECTOR_TOP_K", "3"))))
CHAT_INTENT_VECTOR_TIMEOUT_S = max(
    1.0, float(os.getenv("CHAT_INTENT_VECTOR_TIMEOUT_S", "4.0"))
)
_chat_intent_vector_min_score_raw = os.getenv("CHAT_INTENT_VECTOR_MIN_SCORE", "0.6").strip()
try:
    CHAT_INTENT_VECTOR_MIN_SCORE = float(_chat_intent_vector_min_score_raw or "0.6")
except ValueError:
    CHAT_INTENT_VECTOR_MIN_SCORE = 0.6
try:
    CHAT_INTENT_VECTOR_SCORE_MARGIN = max(
        0.0, float(os.getenv("CHAT_INTENT_VECTOR_SCORE_MARGIN", "0.03"))
    )
except ValueError:
    CHAT_INTENT_VECTOR_SCORE_MARGIN = 0.03
CHAT_CAPABILITY_VECTOR_SEARCH_ENABLED = (
    os.getenv("CHAT_CAPABILITY_VECTOR_SEARCH_ENABLED", "true").lower() == "true"
)
CHAT_CAPABILITY_VECTOR_COLLECTION = os.getenv("CHAT_CAPABILITY_VECTOR_COLLECTION", "").strip() or None
CHAT_CAPABILITY_VECTOR_NAMESPACE_PREFIX = (
    os.getenv("CHAT_CAPABILITY_VECTOR_NAMESPACE_PREFIX", "chat_capability_catalog").strip()
    or "chat_capability_catalog"
)
CHAT_CAPABILITY_VECTOR_WORKSPACE_ID = (
    os.getenv("CHAT_CAPABILITY_VECTOR_WORKSPACE_ID", "chat-capability-catalog").strip()
    or "chat-capability-catalog"
)
CHAT_CAPABILITY_VECTOR_TOP_K = max(
    1, min(50, int(os.getenv("CHAT_CAPABILITY_VECTOR_TOP_K", "12")))
)
CHAT_CAPABILITY_VECTOR_TIMEOUT_S = max(
    1.0, float(os.getenv("CHAT_CAPABILITY_VECTOR_TIMEOUT_S", "5.0"))
)
CHAT_CAPABILITY_VECTOR_CLEANUP_ENABLED = (
    os.getenv("CHAT_CAPABILITY_VECTOR_CLEANUP_ENABLED", "true").lower() == "true"
)
CHAT_CAPABILITY_VECTOR_CLEANUP_LIMIT = max(
    1, min(500, int(os.getenv("CHAT_CAPABILITY_VECTOR_CLEANUP_LIMIT", "500")))
)
_chat_capability_vector_min_score_raw = os.getenv("CHAT_CAPABILITY_VECTOR_MIN_SCORE", "").strip()
try:
    CHAT_CAPABILITY_VECTOR_MIN_SCORE = (
        float(_chat_capability_vector_min_score_raw)
        if _chat_capability_vector_min_score_raw
        else None
    )
except ValueError:
    CHAT_CAPABILITY_VECTOR_MIN_SCORE = None
CHAT_ROUTING_MODE = os.getenv("CHAT_ROUTING_MODE", "response_first").strip().lower()
if CHAT_ROUTING_MODE not in {"always_router", "response_first"}:
    CHAT_ROUTING_MODE = "response_first"
CHAT_PENDING_CORRECTION_MODE = os.getenv("CHAT_PENDING_CORRECTION_MODE", "llm").strip().lower()
if CHAT_PENDING_CORRECTION_MODE not in {"heuristic", "llm", "hybrid"}:
    CHAT_PENDING_CORRECTION_MODE = "llm"
INTENT_VECTOR_SEARCH_ENABLED = (
    os.getenv("INTENT_VECTOR_SEARCH_ENABLED", "true").lower() == "true"
)
INTENT_VECTOR_COLLECTION = os.getenv("INTENT_VECTOR_COLLECTION", "").strip() or None
INTENT_VECTOR_NAMESPACE_PREFIX = (
    os.getenv("INTENT_VECTOR_NAMESPACE_PREFIX", "intent_catalog").strip()
    or "intent_catalog"
)
INTENT_VECTOR_WORKSPACE_ID = (
    os.getenv("INTENT_VECTOR_WORKSPACE_ID", "intent-catalog").strip()
    or "intent-catalog"
)
INTENT_VECTOR_TOP_K = max(1, min(5, int(os.getenv("INTENT_VECTOR_TOP_K", "3"))))
INTENT_VECTOR_TIMEOUT_S = max(1.0, float(os.getenv("INTENT_VECTOR_TIMEOUT_S", "4.0")))
_intent_vector_min_score_raw = os.getenv("INTENT_VECTOR_MIN_SCORE", "0.62").strip()
try:
    INTENT_VECTOR_MIN_SCORE = float(_intent_vector_min_score_raw or "0.62")
except ValueError:
    INTENT_VECTOR_MIN_SCORE = 0.62
_intent_vector_score_margin_raw = os.getenv("INTENT_VECTOR_SCORE_MARGIN", "0.04").strip()
try:
    INTENT_VECTOR_SCORE_MARGIN = max(0.0, float(_intent_vector_score_margin_raw or "0.04"))
except ValueError:
    INTENT_VECTOR_SCORE_MARGIN = 0.04
_intent_vector_override_confidence_raw = os.getenv(
    "INTENT_VECTOR_OVERRIDE_MAX_HEURISTIC_CONFIDENCE",
    "0.72",
).strip()
try:
    INTENT_VECTOR_OVERRIDE_MAX_HEURISTIC_CONFIDENCE = max(
        0.0,
        min(1.0, float(_intent_vector_override_confidence_raw or "0.72")),
    )
except ValueError:
    INTENT_VECTOR_OVERRIDE_MAX_HEURISTIC_CONFIDENCE = 0.72
_intent_vector_trigger_confidence_raw = os.getenv(
    "INTENT_VECTOR_TRIGGER_MAX_HEURISTIC_CONFIDENCE",
    "0.55",
).strip()
try:
    INTENT_VECTOR_TRIGGER_MAX_HEURISTIC_CONFIDENCE = max(
        0.0,
        min(1.0, float(_intent_vector_trigger_confidence_raw or "0.55")),
    )
except ValueError:
    INTENT_VECTOR_TRIGGER_MAX_HEURISTIC_CONFIDENCE = 0.55
_intent_assess_skip_llm_confidence_raw = os.getenv(
    "INTENT_ASSESS_SKIP_LLM_MIN_CONFIDENCE",
    "0.72",
).strip()
try:
    INTENT_ASSESS_SKIP_LLM_MIN_CONFIDENCE = max(
        0.0,
        min(1.0, float(_intent_assess_skip_llm_confidence_raw or "0.72")),
    )
except ValueError:
    INTENT_ASSESS_SKIP_LLM_MIN_CONFIDENCE = 0.72
CHAT_DIRECT_CAPABILITIES = {
    entry.strip()
    for entry in os.getenv(
        "CHAT_DIRECT_CAPABILITIES",
        ",".join(sorted(chat_execution_service.DEFAULT_CHAT_DIRECT_CAPABILITIES)),
    ).split(",")
    if entry.strip()
}
if not CHAT_DIRECT_EXECUTION_ENABLED:
    CHAT_DIRECT_CAPABILITIES = set()
INTENT_MEMORY_PERSIST_ENABLED = (
    os.getenv("INTENT_MEMORY_PERSIST_ENABLED", "true").lower() == "true"
)
INTERACTION_SUMMARY_MEMORY_PERSIST = (
    os.getenv("INTERACTION_SUMMARY_MEMORY_PERSIST", "true").lower() == "true"
)
INTERACTION_SUMMARY_COMPACTION_ENABLED = (
    os.getenv("INTERACTION_SUMMARY_COMPACTION_ENABLED", "true").lower() == "true"
)
INTERACTION_SUMMARY_COMPACT_EVERY_N = max(
    1, int(os.getenv("INTERACTION_SUMMARY_COMPACT_EVERY_N", "10"))
)
INTERACTION_SUMMARY_MAX_ITEMS = max(
    1, int(os.getenv("INTERACTION_SUMMARY_MAX_ITEMS", "40"))
)
INTERACTION_SUMMARY_MAX_FACTS_PER_ITEM = max(
    1, int(os.getenv("INTERACTION_SUMMARY_MAX_FACTS_PER_ITEM", "4"))
)
INTERACTION_SUMMARY_MAX_CHARS_PER_FIELD = max(
    24, int(os.getenv("INTERACTION_SUMMARY_MAX_CHARS_PER_FIELD", "180"))
)
INTERACTION_SUMMARY_MAX_TOKENS = max(
    200, int(os.getenv("INTERACTION_SUMMARY_MAX_TOKENS", "1800"))
)
EVENT_OUTBOX_ENABLED = os.getenv("EVENT_OUTBOX_ENABLED", "true").lower() == "true"
EVENT_OUTBOX_BATCH_SIZE = int(os.getenv("EVENT_OUTBOX_BATCH_SIZE", "200"))
EVENT_OUTBOX_POLL_S = float(os.getenv("EVENT_OUTBOX_POLL_S", "1.0"))
EVENT_OUTBOX_REDIS_RETRIES = int(os.getenv("EVENT_OUTBOX_REDIS_RETRIES", "3"))
EVENT_OUTBOX_REDIS_RETRY_SLEEP_S = float(os.getenv("EVENT_OUTBOX_REDIS_RETRY_SLEEP_S", "0.2"))
SEMANTIC_MEMORY_DEFAULT_USER_ID = os.getenv("SEMANTIC_MEMORY_DEFAULT_USER_ID", "default-user").strip()
RUNTIME_CONFORMANCE_ENABLED = (
    os.getenv("RUNTIME_CONFORMANCE_ENABLED", "true").lower() == "true"
)
STUDIO_RUN_SCHEDULER_ENABLED = (
    os.getenv("STUDIO_RUN_SCHEDULER_ENABLED", "false").lower() == "true"
)
PLANNER_RUN_SCHEDULER_ENABLED = (
    os.getenv("PLANNER_RUN_SCHEDULER_ENABLED", "false").lower() == "true"
)
RUNTIME_CONFORMANCE_SERVICE = (
    os.getenv("RUNTIME_CONFORMANCE_SERVICE", "worker").strip().lower() or "worker"
)
POSTGRES_RUN_SPEC_SCHEDULER_MODE = "postgres_run_spec"
redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
TASK_OUTPUT_KEY_PREFIX = "task_output:"
TASK_RESULT_KEY_PREFIX = "task_result:"
CHAT_DIRECT_SYNC_WORKER_CONSUMER = "api.chat_sync"

_tool_spec_registry = tool_bootstrap.build_default_registry(
    http_fetch_enabled=False,
    llm_enabled=True,
    llm_provider=MockLLMProvider(),
    service_name="api",
)
TOOL_INPUT_SCHEMAS = {spec.name: spec.input_schema for spec in _tool_spec_registry.list_specs()}
TOOL_INTENTS_BY_NAME = {spec.name: spec.tool_intent for spec in _tool_spec_registry.list_specs()}


def _build_composer_recommender_provider() -> LLMProvider | None:
    if not COMPOSER_RECOMMENDER_ENABLED:
        return None
    provider_name = (LLM_PROVIDER_NAME or "").strip().lower()
    if not provider_name or provider_name == "mock":
        return None
    model_name = (COMPOSER_RECOMMENDER_MODEL or LLM_MODEL_NAME or "").strip()
    if not model_name:
        return None
    try:
        return resolve_provider(
            provider_name,
            api_key=OPENAI_API_KEY or None,
            model=model_name,
            base_url=OPENAI_BASE_URL or None,
            timeout_s=max(1.0, OPENAI_TIMEOUT_S),
            max_retries=max(0, OPENAI_MAX_RETRIES),
        )
    except Exception:  # noqa: BLE001
        logger.exception(
            "composer_recommender_provider_init_failed",
            extra={"provider": provider_name, "model": model_name},
        )
        return None


_composer_recommender_provider = _build_composer_recommender_provider()


def _build_intent_assess_provider() -> LLMProvider | None:
    if not INTENT_ASSESS_ENABLED:
        return None
    if INTENT_ASSESS_MODE == "heuristic":
        return None
    provider_name = (LLM_PROVIDER_NAME or "").strip().lower()
    if not provider_name or provider_name == "mock":
        return None
    model_name = (INTENT_ASSESS_MODEL or INTENT_DECOMPOSE_MODEL or LLM_MODEL_NAME or "").strip()
    if not model_name:
        return None
    try:
        return resolve_provider(
            provider_name,
            api_key=OPENAI_API_KEY or None,
            model=model_name,
            base_url=OPENAI_BASE_URL or None,
            timeout_s=max(1.0, OPENAI_TIMEOUT_S),
            max_retries=max(0, OPENAI_MAX_RETRIES),
        )
    except Exception:  # noqa: BLE001
        logger.exception(
            "intent_assess_provider_init_failed",
            extra={"provider": provider_name, "model": model_name},
        )
        return None


_intent_assess_provider = _build_intent_assess_provider()


def _build_intent_decompose_provider() -> LLMProvider | None:
    if not INTENT_DECOMPOSE_ENABLED:
        return None
    if INTENT_DECOMPOSE_MODE == "heuristic":
        return None
    provider_name = (LLM_PROVIDER_NAME or "").strip().lower()
    if not provider_name or provider_name == "mock":
        return None
    model_name = (INTENT_DECOMPOSE_MODEL or LLM_MODEL_NAME or "").strip()
    if not model_name:
        return None
    try:
        return resolve_provider(
            provider_name,
            api_key=OPENAI_API_KEY or None,
            model=model_name,
            base_url=OPENAI_BASE_URL or None,
            timeout_s=max(1.0, OPENAI_TIMEOUT_S),
            max_retries=max(0, OPENAI_MAX_RETRIES),
        )
    except Exception:  # noqa: BLE001
        logger.exception(
            "intent_decompose_provider_init_failed",
            extra={"provider": provider_name, "model": model_name},
        )
        return None


_intent_decompose_provider = _build_intent_decompose_provider()


def _build_chat_router_provider() -> LLMProvider | None:
    provider_name = (LLM_PROVIDER_NAME or "").strip().lower()
    if not provider_name or provider_name == "mock":
        return None
    model_name = (CHAT_ROUTER_MODEL or LLM_MODEL_NAME or "").strip()
    if not model_name:
        return None
    try:
        return resolve_provider(
            provider_name,
            api_key=OPENAI_API_KEY or None,
            model=model_name,
            base_url=OPENAI_BASE_URL or None,
            timeout_s=max(1.0, OPENAI_TIMEOUT_S),
            max_retries=max(0, OPENAI_MAX_RETRIES),
        )
    except Exception:  # noqa: BLE001
        logger.exception(
            "chat_router_provider_init_failed",
            extra={"provider": provider_name, "model": model_name},
        )
        return None


_chat_router_provider = _build_chat_router_provider()


def _build_chat_response_provider() -> LLMProvider | None:
    provider_name = (LLM_PROVIDER_NAME or "").strip().lower()
    if not provider_name or provider_name == "mock":
        return None
    model_name = (CHAT_RESPONSE_MODEL or LLM_MODEL_NAME or "").strip()
    if not model_name:
        return None
    try:
        return resolve_provider(
            provider_name,
            api_key=OPENAI_API_KEY or None,
            model=model_name,
            base_url=OPENAI_BASE_URL or None,
            timeout_s=max(1.0, OPENAI_TIMEOUT_S),
            max_retries=max(0, OPENAI_MAX_RETRIES),
        )
    except Exception:  # noqa: BLE001
        logger.exception(
            "chat_response_provider_init_failed",
            extra={"provider": provider_name, "model": model_name},
        )
        return None


_chat_response_provider = _build_chat_response_provider()


def _build_chat_pending_correction_provider() -> LLMProvider | None:
    if CHAT_PENDING_CORRECTION_MODE == "heuristic":
        return None
    provider_name = (LLM_PROVIDER_NAME or "").strip().lower()
    if not provider_name or provider_name == "mock":
        return None
    model_name = (
        CHAT_PENDING_CORRECTION_MODEL
        or CHAT_RESPONSE_MODEL
        or LLM_MODEL_NAME
        or ""
    ).strip()
    if not model_name:
        return None
    try:
        return resolve_provider(
            provider_name,
            api_key=OPENAI_API_KEY or None,
            model=model_name,
            base_url=OPENAI_BASE_URL or None,
            timeout_s=max(1.0, OPENAI_TIMEOUT_S),
            max_retries=max(0, OPENAI_MAX_RETRIES),
        )
    except Exception:  # noqa: BLE001
        logger.exception(
            "chat_pending_correction_provider_init_failed",
            extra={"provider": provider_name, "model": model_name},
        )
        return None


_chat_pending_correction_provider = _build_chat_pending_correction_provider()


def _build_chat_clarification_normalizer_provider() -> LLMProvider | None:
    if not CHAT_CLARIFICATION_NORMALIZER_ENABLED:
        return None
    provider_name = (LLM_PROVIDER_NAME or "").strip().lower()
    if not provider_name or provider_name == "mock":
        return None
    model_name = (
        CHAT_CLARIFICATION_NORMALIZER_MODEL
        or CHAT_RESPONSE_MODEL
        or LLM_MODEL_NAME
        or ""
    ).strip()
    if not model_name:
        return None
    try:
        return resolve_provider(
            provider_name,
            api_key=OPENAI_API_KEY or None,
            model=model_name,
            base_url=OPENAI_BASE_URL or None,
            timeout_s=max(1.0, OPENAI_TIMEOUT_S),
            max_retries=max(0, OPENAI_MAX_RETRIES),
        )
    except Exception:  # noqa: BLE001
        logger.exception(
            "chat_clarification_normalizer_provider_init_failed",
            extra={"provider": provider_name, "model": model_name},
        )
        return None


_chat_clarification_normalizer_provider = _build_chat_clarification_normalizer_provider()
_CHAT_INTENT_VECTOR_CATALOG: tuple[dict[str, str], ...] = (
    {
        "id": "capability_discovery",
        "label": "Capability discovery",
        "text": (
            "Chat intent: capability discovery.\n"
            "Use when the user asks what the assistant can do, which capabilities or tools are "
            "available, what work is supported, or which operations the assistant can handle.\n"
            "Examples: what can you do; list your capabilities; what tools are available here; "
            "show supported operations; what kinds of work can you handle here; what kinds of "
            "GitHub work can you handle; what repo changes can you help with; what can this "
            "assistant help with."
        ),
    },
    {
        "id": "general_chat",
        "label": "General chat",
        "text": (
            "Chat intent: general chat.\n"
            "Use when the user wants explanation, discussion, brainstorming, or help "
            "understanding a topic rather than a list of available tools.\n"
            "Examples: explain Kubernetes; help me understand RAG; walk me through Docker; "
            "what is intent classification; how does the planner work."
        ),
    },
    {
        "id": "workflow_execution",
        "label": "Workflow execution",
        "text": (
            "Chat intent: workflow execution.\n"
            "Use when the user wants the assistant to perform work, create or update files, run "
            "commands, generate an artifact, deploy something, or submit a workflow/job.\n"
            "Examples: create a workflow; update the README; run the tests; deploy the API; "
            "open a pull request; build a PDF report."
        ),
    },
)
_INTENT_VECTOR_CATALOG: tuple[dict[str, str], ...] = (
    {
        "id": "generate",
        "label": "Generate",
        "text": (
            "Execution intent: generate.\n"
            "Use when the user wants new content or a new artifact created from scratch.\n"
            "Examples: draft release notes; create a workflow; write a new specification; "
            "produce a report; generate documentation; compose a plan."
        ),
    },
    {
        "id": "transform",
        "label": "Transform",
        "text": (
            "Execution intent: transform.\n"
            "Use when the user wants existing content or data revised, polished, cleaned up, "
            "normalized, reworked, reshaped, or refined.\n"
            "Examples: polish this document spec; clean up this payload; normalize the data; "
            "revise the draft; refactor the JSON structure."
        ),
    },
    {
        "id": "validate",
        "label": "Validate",
        "text": (
            "Execution intent: validate.\n"
            "Use when the user wants something checked, verified, linted, tested, reviewed, "
            "or compared against a schema or contract.\n"
            "Examples: validate this spec; sanity-check the payload; verify the schema; review "
            "for errors; lint the configuration."
        ),
    },
    {
        "id": "render",
        "label": "Render",
        "text": (
            "Execution intent: render.\n"
            "Use when the user wants a final formatted artifact like PDF, DOCX, HTML, or another "
            "presentation-ready output produced from structured content.\n"
            "Examples: render a PDF report; generate a DOCX handout; turn the spec into a final "
            "document; export the final artifact."
        ),
    },
    {
        "id": "io",
        "label": "IO",
        "text": (
            "Execution intent: io.\n"
            "Use when the user wants data read, listed, searched, fetched, uploaded, saved, "
            "pushed, or synchronized with a system such as GitHub, Slack, Jira, memory, or the filesystem.\n"
            "Examples: list repositories; fetch an issue; search memory; upload a file; save the "
            "artifact; push changes to GitHub."
        ),
    },
)
_chat_intent_vector_sync_lock = threading.Lock()
_chat_intent_vector_synced_namespace: str | None = None
_intent_vector_sync_lock = threading.Lock()
_intent_vector_synced_namespace: str | None = None
_chat_capability_vector_sync_lock = threading.Lock()
_chat_capability_vector_synced_namespace: str | None = None


def _rag_retriever_request_json(
    path: str,
    *,
    method: str = "GET",
    body: dict[str, Any] | None = None,
    query: Mapping[str, Any] | None = None,
    timeout_s: float = 20.0,
) -> Any:
    server = mcp_gateway.load_mcp_server_registry().get("rag_retriever_qdrant")
    if server is None or not server.enabled:
        raise HTTPException(status_code=503, detail="rag_retriever_service_unavailable")
    url = f"{server.base_url.rstrip('/')}{path}"
    if query:
        normalized_query = {
            key: value
            for key, value in dict(query).items()
            if value is not None and str(value).strip() != ""
        }
        if normalized_query:
            url = f"{url}?{urlencode(normalized_query, doseq=True)}"
    headers: dict[str, str] = {}
    token = (
        os.getenv(server.bearer_env, "").strip()
        if isinstance(server.bearer_env, str) and server.bearer_env
        else ""
    )
    if token:
        headers["Authorization"] = f"Bearer {token}"
    data = None
    if body is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(body).encode("utf-8")
    request = UrlRequest(url, data=data, headers=headers, method=method)
    try:
        with urlopen(request, timeout=timeout_s) as response:
            raw = response.read().decode("utf-8")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8") if exc.fp else str(exc)
        raise HTTPException(status_code=exc.code, detail=f"rag_retriever_error:{detail}") from exc
    except (URLError, TimeoutError) as exc:
        raise HTTPException(status_code=502, detail=f"rag_retriever_unreachable:{exc}") from exc
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=502, detail="rag_retriever_invalid_json") from exc


def _init_db() -> None:
    Base.metadata.create_all(bind=engine)
    if EVENT_OUTBOX_ENABLED:
        _start_event_outbox_dispatcher()
    if ORCHESTRATOR_ENABLED:
        _start_orchestrator()
    if JOB_RECOVERY_ENABLED:
        _recover_jobs()


jobs_created_total = Counter("jobs_created_total", "Jobs created")
orchestrator_loop_errors_total = Counter(
    "orchestrator_loop_errors_total", "Orchestrator loop errors"
)
orchestrator_handle_errors_total = Counter(
    "orchestrator_handle_errors_total", "Orchestrator event handling errors", ["stream"]
)
orchestrator_recovered_events_total = Counter(
    "orchestrator_recovered_events_total", "Orchestrator recovered pending events", ["stream"]
)
intent_assessments_total = Counter(
    "intent_assessments_total",
    "Goal intent assessments produced",
    ["source", "needs_clarification", "requires_blocking_clarification", "fallback_used"],
)
intent_clarification_required_total = Counter(
    "intent_clarification_required_total",
    "Goal intent assessments that require clarification",
    ["source", "fallback_used"],
)
intent_threshold_evaluations_total = Counter(
    "intent_threshold_evaluations_total",
    "Intent confidence threshold evaluations",
    ["intent", "risk_level", "source", "needs_clarification", "threshold_bucket", "fallback_used"],
)
intent_confidence_outcomes_total = Counter(
    "intent_confidence_outcomes_total",
    "Observed outcomes for goal intent confidence calibration",
    ["intent", "risk_level", "outcome", "above_threshold", "confidence_bucket"],
)
intent_decompose_requests_total = Counter(
    "intent_decompose_requests_total",
    "Intent decomposition requests",
    ["mode", "model", "source", "result", "has_summaries", "fallback_used"],
)
intent_decompose_failures_total = Counter(
    "intent_decompose_failures_total",
    "Intent decomposition LLM failures",
    ["mode", "model", "error_type"],
)
intent_decompose_segments_total = Counter(
    "intent_decompose_segments_total",
    "Intent graph segments generated",
    ["source", "model"],
)
intent_fact_candidates_total = Counter(
    "intent_fact_candidates_total",
    "Intent fact candidates observed before filtering",
    ["source", "model"],
)
intent_fact_supported_total = Counter(
    "intent_fact_supported_total",
    "Intent facts retained after filtering",
    ["source", "model"],
)
intent_fact_stripped_total = Counter(
    "intent_fact_stripped_total",
    "Intent facts stripped as unsupported",
    ["source", "model"],
)
intent_capability_suggestions_total = Counter(
    "intent_capability_suggestions_total",
    "Intent capability suggestions before catalog filtering",
    ["source", "model"],
)
intent_capability_suggestions_matched_total = Counter(
    "intent_capability_suggestions_matched_total",
    "Intent capability suggestions that matched the catalog",
    ["source", "model"],
)
intent_segment_rejections_total = Counter(
    "intent_segment_rejections_total",
    "Intent segment contract rejections observed during control-plane validation",
    ["surface", "reason"],
)
intent_memory_hint_candidates_total = Counter(
    "intent_memory_hint_candidates_total",
    "Semantic intent workflow hint candidates scanned",
)
intent_memory_hints_selected_total = Counter(
    "intent_memory_hints_selected_total",
    "Semantic intent workflow hints selected",
)
interaction_summary_compactions_total = Counter(
    "interaction_summary_compactions_total",
    "Interaction summary compactions",
    ["applied"],
)
interaction_summary_tokens_in_total = Counter(
    "interaction_summary_tokens_in_total",
    "Estimated input tokens for interaction summaries before compaction",
)
interaction_summary_tokens_out_total = Counter(
    "interaction_summary_tokens_out_total",
    "Estimated output tokens for interaction summaries after compaction",
)
interaction_summary_memory_persist_total = Counter(
    "interaction_summary_memory_persist_total",
    "Interaction summary memory writes",
    ["status"],
)
capability_search_requests_total = Counter(
    "capability_search_requests_total",
    "Capability search requests",
    ["request_source", "intent", "result"],
)
capability_search_results_total = Counter(
    "capability_search_results_total",
    "Capability search results returned",
    ["request_source", "intent"],
)
planner_capability_selection_total = Counter(
    "planner_capability_selection_total",
    "Capabilities selected by planner-created plans",
    ["capability"],
)
capability_execution_outcomes_total = Counter(
    "capability_execution_outcomes_total",
    "Capability execution outcomes observed from task results",
    ["capability", "status"],
)
feedback_submitted_total = Counter(
    "feedback_submitted_total",
    "Explicit feedback submissions",
    ["target_type", "sentiment"],
)
feedback_reason_total = Counter(
    "feedback_reason_total",
    "Explicit feedback reasons submitted",
    ["target_type", "reason_code"],
)
feedback_summary_requests_total = Counter(
    "feedback_summary_requests_total",
    "Feedback summary endpoint requests",
)
feedback_examples_export_total = Counter(
    "feedback_examples_export_total",
    "Feedback example export requests",
)
chat_boundary_decisions_total = Counter(
    "chat_boundary_decisions_total",
    "Chat boundary decisions emitted by the response-first boundary model",
    ["decision", "conversation_mode_hint", "pending_clarification", "workflow_target_available"],
)
chat_boundary_reason_total = Counter(
    "chat_boundary_reason_total",
    "Chat boundary reason codes emitted by the response-first boundary model",
    ["decision", "reason_code"],
)
chat_boundary_feedback_total = Counter(
    "chat_boundary_feedback_total",
    "Explicit chat-message feedback grouped by prior chat boundary decision",
    ["decision", "sentiment"],
)
chat_clarification_slot_loss_feedback_total = Counter(
    "chat_clarification_slot_loss_feedback_total",
    "Explicit chat-message feedback grouped by observed clarification slot-loss state",
    ["slot_loss_state", "sentiment"],
)
chat_clarification_family_alignment_feedback_total = Counter(
    "chat_clarification_family_alignment_feedback_total",
    "Explicit chat-message feedback grouped by clarification active-family alignment",
    ["alignment", "sentiment"],
)
chat_clarification_mapping_feedback_total = Counter(
    "chat_clarification_mapping_feedback_total",
    "Explicit chat-message feedback grouped by clarification mapping outcomes",
    ["resolved_active_field", "queue_advanced", "restarted", "sentiment"],
)


def _intent_source_label(source: Any) -> str:
    normalized = str(source or "").strip().lower()
    return normalized or "unknown"


def _metrics_label(value: Any, *, default: str = "unknown") -> str:
    normalized = str(value or "").strip().lower()
    return normalized or default


def _intent_assessment_fallback_used(source: Any) -> bool:
    if not INTENT_ASSESS_ENABLED or INTENT_ASSESS_MODE == "heuristic":
        return False
    return _intent_source_label(source) != "llm"


def _intent_decompose_fallback_used(source: Any) -> bool:
    if not INTENT_DECOMPOSE_ENABLED or INTENT_DECOMPOSE_MODE == "heuristic":
        return False
    return _intent_source_label(source) != "llm"


def _intent_segment_contract_reason(detail: str) -> str:
    normalized = str(detail or "").strip()
    if not normalized:
        return "unknown"
    if normalized.startswith("intent_segment_invalid:"):
        remainder = normalized[len("intent_segment_invalid:") :]
        parts = remainder.split(":", 2)
        normalized = parts[2] if len(parts) == 3 else parts[-1]
    reason = normalized.split(":", 1)[0].strip().lower()
    reason = re.sub(r"[^a-z0-9_]+", "_", reason)
    return reason or "unknown"


def _record_intent_segment_rejection(
    *,
    surface: str,
    task_name: str,
    request_id: str,
    detail: str,
) -> None:
    reason = _intent_segment_contract_reason(detail)
    intent_segment_rejections_total.labels(surface=surface, reason=reason).inc()
    logger.warning(
        "intent_segment_rejected",
        extra={
            "surface": surface,
            "task_name": task_name,
            "request_id": request_id,
            "reason": reason,
            "detail": detail,
        },
    )


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _job_from_record(record: JobRecord) -> models.Job:
    return models.Job(
        id=record.id,
        goal=record.goal,
        context_json=record.context_json or {},
        status=record.status,
        created_at=record.created_at,
        updated_at=record.updated_at,
        priority=record.priority or 0,
        metadata=record.metadata_json or {},
    )


def _plan_from_record(record: PlanRecord) -> models.Plan:
    return models.Plan(
        id=record.id,
        job_id=record.job_id,
        planner_version=record.planner_version,
        created_at=record.created_at,
        tasks_summary=record.tasks_summary,
        dag_edges=record.dag_edges or [],
        policy_decision=record.policy_decision or None,
    )


def _workflow_definition_from_record(
    record: WorkflowDefinitionRecord,
) -> models.WorkflowDefinition:
    return models.WorkflowDefinition(
        id=record.id,
        title=record.title,
        goal=record.goal or "",
        context_json=record.context_json or {},
        draft=record.draft_json or {},
        user_id=record.user_id,
        metadata=record.metadata_json or {},
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


def _workflow_version_from_record(record: WorkflowVersionRecord) -> models.WorkflowVersion:
    return models.WorkflowVersion(
        id=record.id,
        definition_id=record.definition_id,
        version_number=record.version_number,
        title=record.title,
        goal=record.goal or "",
        context_json=record.context_json or {},
        draft=record.draft_json or {},
        compiled_plan=record.compiled_plan_json or {},
        run_spec=_workflow_version_run_spec_payload(record),
        user_id=record.user_id,
        metadata=record.metadata_json or {},
        created_at=record.created_at,
    )


def _workflow_version_run_spec(record: WorkflowVersionRecord) -> models.RunSpec | None:
    metadata = record.metadata_json if isinstance(record.metadata_json, dict) else {}
    stored_run_spec = run_specs.parse_run_spec(metadata.get("run_spec"))
    if stored_run_spec is not None:
        return stored_run_spec
    compiled_plan = _parse_plan_payload(record.compiled_plan_json or {})
    if compiled_plan is None:
        return None
    try:
        return run_specs.plan_to_run_spec(compiled_plan, kind=models.RunKind.studio)
    except ValueError:
        return None


def _workflow_version_run_spec_payload(record: WorkflowVersionRecord) -> dict[str, Any]:
    run_spec = _workflow_version_run_spec(record)
    if run_spec is None:
        return {}
    return run_spec.model_dump(mode="json")


def _workflow_version_plan(record: WorkflowVersionRecord) -> models.PlanCreate | None:
    run_spec = _workflow_version_run_spec(record)
    if run_spec is not None:
        try:
            return run_specs.run_spec_to_plan(run_spec)
        except ValueError:
            pass
    return _parse_plan_payload(record.compiled_plan_json or {})


def _scheduler_mode_from_metadata(metadata: Mapping[str, Any] | None) -> str:
    if not isinstance(metadata, Mapping):
        return ""
    return str(metadata.get("scheduler_mode") or "").strip()


def _workflow_run_uses_postgres_scheduler(record: WorkflowRunRecord | None) -> bool:
    if record is None:
        return False
    return _scheduler_mode_from_metadata(record.metadata_json) == POSTGRES_RUN_SPEC_SCHEDULER_MODE


def _job_uses_postgres_scheduler(job: JobRecord | None) -> bool:
    if job is None:
        return False
    return _scheduler_mode_from_metadata(job.metadata_json) == POSTGRES_RUN_SPEC_SCHEDULER_MODE


def _job_workflow_run_id(job: JobRecord | None) -> str | None:
    metadata = job.metadata_json if job and isinstance(job.metadata_json, dict) else {}
    workflow_run_id = str(metadata.get("workflow_run_id") or "").strip()
    return workflow_run_id or None


def _planner_job_run_spec(job: JobRecord | None) -> models.RunSpec | None:
    metadata = job.metadata_json if job and isinstance(job.metadata_json, dict) else {}
    return run_specs.parse_run_spec(metadata.get("run_spec"))


def _workflow_trigger_from_record(
    record: WorkflowTriggerRecord,
) -> models.WorkflowTrigger:
    trigger_type = record.trigger_type
    if trigger_type not in {item.value for item in models.WorkflowTriggerType}:
        trigger_type = models.WorkflowTriggerType.manual.value
    return models.WorkflowTrigger(
        id=record.id,
        definition_id=record.definition_id,
        title=record.title,
        trigger_type=models.WorkflowTriggerType(trigger_type),
        enabled=bool(record.enabled),
        config=record.config_json or {},
        user_id=record.user_id,
        metadata=record.metadata_json or {},
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


def _job_error_from_metadata(metadata: Mapping[str, Any] | None) -> str | None:
    if not isinstance(metadata, Mapping):
        return None
    plan_error = metadata.get("plan_error")
    if isinstance(plan_error, str) and plan_error.strip():
        return plan_error.strip()
    if metadata.get("plan_preflight_errors") is not None:
        return f"plan_preflight_failed: {metadata.get('plan_preflight_errors')}"
    return None


def _workflow_run_from_record(
    record: WorkflowRunRecord,
    *,
    job_record: JobRecord | None = None,
    latest_task_failure: Mapping[str, Any] | None = None,
) -> models.WorkflowRun:
    job_status: models.JobStatus | None = None
    updated_at = record.updated_at
    job_error: str | None = None
    if job_record is not None:
        try:
            job_status = models.JobStatus(job_record.status)
        except ValueError:
            job_status = None
        job_error = _job_error_from_metadata(
            job_record.metadata_json if isinstance(job_record.metadata_json, dict) else {}
        )
        if isinstance(job_record.updated_at, datetime) and job_record.updated_at > updated_at:
            updated_at = job_record.updated_at
    return models.WorkflowRun(
        id=record.id,
        definition_id=record.definition_id,
        version_id=record.version_id,
        trigger_id=record.trigger_id,
        title=record.title,
        goal=record.goal or "",
        requested_context_json=record.requested_context_json or {},
        job_id=record.job_id,
        plan_id=record.plan_id,
        job_status=job_status,
        job_error=job_error,
        latest_task_id=(
            str(latest_task_failure.get("task_id"))
            if isinstance(latest_task_failure, Mapping) and latest_task_failure.get("task_id")
            else None
        ),
        latest_task_name=(
            str(latest_task_failure.get("task_name"))
            if isinstance(latest_task_failure, Mapping) and latest_task_failure.get("task_name")
            else None
        ),
        latest_task_error=(
            str(latest_task_failure.get("error"))
            if isinstance(latest_task_failure, Mapping) and latest_task_failure.get("error")
            else None
        ),
        user_id=record.user_id,
        metadata=record.metadata_json or {},
        created_at=record.created_at,
        updated_at=updated_at,
    )


def _coerce_task_intent_profiles(metadata: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not isinstance(metadata, Mapping):
        return {}
    raw = metadata.get("task_intent_profiles")
    if not isinstance(raw, Mapping):
        return {}
    profiles: dict[str, dict[str, Any]] = {}
    for task_id, payload in raw.items():
        key = str(task_id).strip()
        if not key or not isinstance(payload, Mapping):
            continue
        intent = str(payload.get("intent") or "").strip()
        source = str(payload.get("source") or "").strip()
        confidence_raw = payload.get("confidence")
        try:
            confidence = float(confidence_raw)
        except (TypeError, ValueError):
            confidence = 0.0
        segment = _normalize_task_intent_profile_segment(payload.get("segment"))
        profiles[key] = {
            "intent": intent,
            "source": source,
            "confidence": max(0.0, min(1.0, confidence)),
        }
        if segment is not None:
            profiles[key]["segment"] = segment
    return profiles


def _normalize_task_intent_profile_segment(raw_segment: Any) -> dict[str, Any] | None:
    if not isinstance(raw_segment, Mapping):
        return None
    intent = (
        intent_contract.normalize_task_intent(raw_segment.get("intent"))
        or models.ToolIntent.generate.value
    )
    objective = str(raw_segment.get("objective") or "").strip()
    required_inputs = _coerce_string_list(raw_segment.get("required_inputs"))
    suggested_capabilities = _coerce_string_list(raw_segment.get("suggested_capabilities"))
    slots = intent_contract.normalize_intent_segment_slots(
        raw_slots=raw_segment.get("slots"),
        fallback_slots=None,
        intent=intent,
        objective=objective,
        required_inputs=required_inputs,
        suggested_capabilities=suggested_capabilities,
    )
    segment_id = str(raw_segment.get("id") or "").strip()
    normalized: dict[str, Any] = {
        "id": segment_id or None,
        "intent": intent,
        "objective": objective,
        "required_inputs": required_inputs,
        "suggested_capabilities": suggested_capabilities,
        "slots": slots,
    }
    depends_on = _coerce_string_list(raw_segment.get("depends_on"))
    if depends_on:
        normalized["depends_on"] = depends_on
    return normalized


def _task_from_record(
    record: TaskRecord,
    intent_profile: Mapping[str, Any] | None = None,
) -> models.Task:
    raw_tool_inputs = record.tool_inputs if isinstance(record.tool_inputs, dict) else {}
    capability_bindings = _task_capability_bindings(
        record.tool_requests or [],
        raw_tool_inputs,
    )
    source = None
    confidence = None
    if isinstance(intent_profile, Mapping):
        raw_source = intent_profile.get("source")
        if isinstance(raw_source, str) and raw_source.strip():
            source = raw_source.strip()
        raw_conf = intent_profile.get("confidence")
        if isinstance(raw_conf, (int, float)):
            confidence = max(0.0, min(1.0, float(raw_conf)))
    return models.Task(
        id=record.id,
        job_id=record.job_id,
        plan_id=record.plan_id,
        name=record.name,
        description=record.description,
        instruction=record.instruction,
        acceptance_criteria=record.acceptance_criteria or [],
        expected_output_schema_ref=record.expected_output_schema_ref,
        status=record.status,
        intent=record.intent,
        intent_source=source,
        intent_confidence=confidence,
        deps=record.deps or [],
        attempts=record.attempts or 0,
        max_attempts=record.max_attempts or 0,
        rework_count=record.rework_count or 0,
        max_reworks=record.max_reworks or 0,
        assigned_to=record.assigned_to,
        tool_requests=record.tool_requests or [],
        tool_inputs=execution_contracts.strip_execution_metadata_from_tool_inputs(raw_tool_inputs),
        capability_bindings=capability_bindings,
        created_at=record.created_at,
        updated_at=record.updated_at,
        critic_required=bool(record.critic_required),
    )


def _api_enabled_capabilities() -> Mapping[str, Any]:
    try:
        registry = capability_registry.load_capability_registry()
    except Exception:  # noqa: BLE001
        return {}
    return registry.enabled_capabilities()


def _task_capability_bindings(
    tool_requests: list[str],
    tool_inputs: Mapping[str, Any] | None,
    capability_bindings: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    return execution_contracts.normalize_capability_bindings(
        {
            "tool_inputs": tool_inputs,
            "capability_bindings": capability_bindings,
        },
        request_ids=tool_requests,
        capabilities=_api_enabled_capabilities(),
    )


def _task_record_tool_inputs(
    tool_requests: list[str],
    tool_inputs: Mapping[str, Any] | None,
    capability_bindings: Mapping[str, Any] | None = None,
    execution_gate: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    merged = execution_contracts.strip_execution_metadata_from_tool_inputs(tool_inputs)
    normalized_execution_gate = execution_contracts.normalize_execution_gates(
        {"tool_inputs": tool_inputs},
        request_ids=tool_requests,
    )
    if isinstance(execution_gate, Mapping):
        normalized_execution_gate.update(
            execution_contracts.normalize_execution_gates(
                {"execution_gates": execution_gate},
                request_ids=tool_requests,
            )
        )
    normalized_capability_bindings = execution_contracts.normalize_capability_bindings(
        {"capability_bindings": capability_bindings},
        request_ids=tool_requests,
        capabilities=_api_enabled_capabilities(),
    )
    if normalized_capability_bindings:
        merged[execution_contracts.EXECUTION_BINDINGS_KEY] = normalized_capability_bindings
    if normalized_execution_gate:
        merged[execution_contracts.EXECUTION_GATE_KEY] = normalized_execution_gate
    return merged


def _task_intent_profile_entry(
    task: models.TaskCreate,
    *,
    goal_text: str,
    intent_segment: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    inference = _task_intent_inference_for_task(task, goal_text=goal_text)
    profile = {
        "intent": inference.intent,
        "source": inference.source,
        "confidence": round(float(inference.confidence), 3),
    }
    normalized_segment = _normalize_task_intent_profile_segment(intent_segment)
    if normalized_segment is not None:
        profile["segment"] = normalized_segment
    return profile


def _merge_task_intent_profiles_into_job_metadata(
    job: JobRecord,
    profiles_by_task_id: Mapping[str, Mapping[str, Any]],
) -> None:
    metadata = dict(job.metadata_json) if isinstance(job.metadata_json, dict) else {}
    merged = _coerce_task_intent_profiles(metadata)
    for task_id, profile in profiles_by_task_id.items():
        key = str(task_id).strip()
        if not key:
            continue
        if not isinstance(profile, Mapping):
            continue
        merged_entry = {
            "intent": str(profile.get("intent") or "").strip(),
            "source": str(profile.get("source") or "").strip(),
            "confidence": max(0.0, min(1.0, float(profile.get("confidence") or 0.0))),
        }
        normalized_segment = _normalize_task_intent_profile_segment(profile.get("segment"))
        if normalized_segment is not None:
            merged_entry["segment"] = normalized_segment
        merged[key] = merged_entry
    metadata["task_intent_profiles"] = merged
    job.metadata_json = metadata


def _normalize_risk_level(value: Any) -> str:
    token = str(value or "").strip().lower().replace("-", "_")
    alias_map = {
        "readonly": "read_only",
        "high_risk": "high_risk_write",
        "high": "high_risk_write",
        "medium": "bounded_write",
        "write": "bounded_write",
        "low": "read_only",
    }
    normalized = alias_map.get(token, token)
    if normalized not in {"read_only", "bounded_write", "high_risk_write"}:
        return "read_only"
    return normalized


def _confidence_bucket(value: float) -> str:
    if value < 0.5:
        return "lt_0_50"
    if value < 0.65:
        return "0_50_0_64"
    if value < 0.8:
        return "0_65_0_79"
    if value < 0.9:
        return "0_80_0_89"
    return "ge_0_90"


def _threshold_bucket(value: float) -> str:
    if value < 0.6:
        return "lt_0_60"
    if value < 0.75:
        return "0_60_0_74"
    if value < 0.85:
        return "0_75_0_84"
    return "ge_0_85"


def _infer_goal_risk_level(goal: str, intent: str) -> str:
    return intent_service.assess_goal_intent(
        goal,
        config=_goal_intent_assess_config(),
        runtime=intent_service.GoalIntentRuntime(
            infer_task_intent=lambda _goal: type(
                "_GoalIntentInference",
                (),
                {"intent": intent, "source": "main_wrapper", "confidence": 1.0},
            )(),
        ),
    ).risk_level or "read_only"


def _goal_intent_assess_config() -> intent_service.GoalIntentConfig:
    return intent_service.GoalIntentConfig(
        min_confidence=INTENT_MIN_CONFIDENCE,
        min_confidence_by_intent=dict(INTENT_MIN_CONFIDENCE_BY_INTENT),
        min_confidence_by_risk=dict(INTENT_MIN_CONFIDENCE_BY_RISK),
        clarification_blocking_slots=set(INTENT_CLARIFICATION_BLOCKING_SLOTS),
    )


def _record_goal_intent_assessment_metrics(
    profile: workflow_contracts.GoalIntentProfile,
) -> None:
    source = _intent_source_label(profile.source)
    needs_clarification = bool(profile.needs_clarification)
    requires_blocking_clarification = bool(profile.requires_blocking_clarification)
    fallback_used = str(_intent_assessment_fallback_used(source)).lower()
    intent_assessments_total.labels(
        source=source,
        needs_clarification=str(needs_clarification).lower(),
        requires_blocking_clarification=str(requires_blocking_clarification).lower(),
        fallback_used=fallback_used,
    ).inc()
    intent_threshold_evaluations_total.labels(
        intent=profile.intent or "generate",
        risk_level=profile.risk_level or "read_only",
        source=source,
        needs_clarification=str(needs_clarification).lower(),
        threshold_bucket=_threshold_bucket(float(profile.threshold or 0.0)),
        fallback_used=fallback_used,
    ).inc()
    if requires_blocking_clarification:
        intent_clarification_required_total.labels(
            source=source,
            fallback_used=fallback_used,
        ).inc()


def _resolve_intent_confidence_threshold(intent: str, risk_level: str) -> float:
    return intent_service.resolve_intent_confidence_threshold(
        intent,
        risk_level,
        config=_goal_intent_assess_config(),
    )


def _extract_goal_slot_signals(goal: str, intent: str, risk_level: str) -> dict[str, Any]:
    profile = intent_service.assess_goal_intent(
        goal,
        config=_goal_intent_assess_config(),
        runtime=intent_service.GoalIntentRuntime(
            infer_task_intent=lambda _goal: type(
                "_GoalIntentInference",
                (),
                {"intent": intent, "source": "main_wrapper", "confidence": 1.0},
            )(),
        ),
    )
    return dict(profile.slot_values)


def _blocking_clarification_slots(intent: str, risk_level: str) -> list[str]:
    return intent_service.blocking_clarification_slots(
        intent,
        risk_level,
        config=_goal_intent_assess_config(),
    )


def _slot_question(slot: str, goal: str) -> str:
    return intent_service.slot_question(slot, goal)


def _chat_route_goal_intent_profile(
    profile: workflow_contracts.GoalIntentProfile,
    *,
    goal: str,
) -> workflow_contracts.GoalIntentProfile:
    preserve_intent_disagreement = (
        str(profile.clarification_mode or "").strip().lower() == "intent_disagreement"
    )
    chat_blocking_slots: list[str] = []
    for raw_slot in profile.blocking_slots:
        normalized = intent_contract.normalize_required_input_key(raw_slot)
        if (
            normalized in CHAT_PRE_SUBMIT_BLOCKING_SLOTS
            or (preserve_intent_disagreement and normalized == "intent_action")
        ) and normalized not in chat_blocking_slots:
            chat_blocking_slots.append(normalized)
    chat_missing_slots: list[str] = []
    for raw_slot in profile.missing_slots:
        normalized = intent_contract.normalize_required_input_key(raw_slot)
        if (
            normalized in CHAT_PRE_SUBMIT_BLOCKING_SLOTS
            or (preserve_intent_disagreement and normalized == "intent_action")
        ) and normalized not in chat_missing_slots:
            chat_missing_slots.append(normalized)
    questions = (
        [
            str(question).strip()
            for question in profile.questions
            if isinstance(question, str) and question.strip()
        ]
        if preserve_intent_disagreement and profile.questions
        else [_slot_question(slot_name, goal) for slot_name in chat_missing_slots]
    )
    return profile.model_copy(
        update={
            "needs_clarification": bool(chat_missing_slots),
            "requires_blocking_clarification": bool(chat_missing_slots),
            "questions": questions,
            "blocking_slots": chat_blocking_slots,
            "missing_slots": chat_missing_slots,
        }
    )


def _looks_like_conversational_turn(content: str) -> bool:
    lowered = str(content or "").strip().lower()
    if not lowered:
        return True
    casual_phrases = (
        "hi",
        "hello",
        "hey",
        "help",
        "thanks",
        "thank you",
        "how are you",
        "what can you do",
        "can you help me understand",
        "why ",
        "what is ",
        "what are ",
        "how does ",
        "how do ",
        "can you explain",
        "explain ",
    )
    workflow_tokens = (
        "create ",
        "build ",
        "generate ",
        "render ",
        "deploy ",
        "port forward",
        "open pull request",
        "open pr",
        "check repo",
        "list repos",
        "write file",
        "update file",
        "make a workflow",
        "create a workflow",
        "submit a job",
        "run ",
    )
    if any(token in lowered for token in workflow_tokens):
        return False
    if lowered.endswith("?"):
        return True
    conversational_patterns = (
        r"\b(?:i want to|i'd like to|id like to|let'?s)\s+(?:discuss|talk about|chat about)\b",
        r"\b(?:discuss|talk about|chat about)\b.{0,80}\b(?:with you|together)?\b",
        r"\b(?:tell me about|walk me through|help me understand|teach me about|give me an overview of)\b",
        r"\b(?:i am|i'm|im)\s+(?:curious about|interested in|trying to understand|learning about)\b",
        r"\b(?:thoughts on|opinion on|overview of|basics of|intro to)\b",
        r"\b(?:practice|mock|roleplay|coach me for|quiz me on)\b.{0,120}\b(?:interview|questions|answers)\b",
        r"\b(?:ask me (?:a )?question|ask me questions one by one|i will type the answer)\b",
        r"\b(?:interview practice|mock interview|practice interview questions)\b",
    )
    if any(re.search(pattern, lowered) for pattern in conversational_patterns):
        return True
    return any(lowered.startswith(phrase) or lowered == phrase for phrase in casual_phrases)


def _looks_like_execution_confirmation(content: str) -> bool:
    lowered = str(content or "").strip().lower()
    if not lowered:
        return False
    normalized = re.sub(r"[^a-z0-9]+", " ", lowered).strip()
    if not normalized:
        return False
    exact_matches = {
        "yes",
        "yes go ahead",
        "go ahead",
        "proceed",
        "continue",
        "continue please",
        "please continue",
        "yes proceed",
        "yes continue",
        "sounds good",
        "ok go ahead",
        "okay go ahead",
    }
    return normalized in exact_matches


def _active_job_confirmation_turn_plan(
    *,
    content: str,
    session_metadata: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(session_metadata, Mapping):
        return None
    if session_metadata.get("pending_clarification"):
        return None
    active_job_id = str(session_metadata.get("active_job_id") or "").strip()
    if not active_job_id or not _looks_like_execution_confirmation(content):
        return None
    return _chat_response_turn_plan(
        goal=content.strip(),
        assistant_content=(
            f"Job {active_job_id} is already submitted. "
            "Ask for status, results, or tell me what you want changed."
        ),
        source="chat_active_job_confirmation",
    )


def _looks_like_chat_only_correction(content: str) -> bool:
    provider = _chat_pending_correction_provider
    if provider is None:
        return False
    try:
        parsed = provider.generate_request_json_object(
            LLMRequest(
                prompt=json.dumps(
                    {
                        "user_message": str(content or ""),
                        "question": (
                            "Is the user canceling or redirecting an in-progress workflow/job/document-style "
                            "clarification and asking for a normal chat answer instead?"
                        ),
                        "response_schema": {
                            "chat_only_correction": "boolean",
                            "confidence": "0..1",
                        },
                    },
                    ensure_ascii=True,
                ),
                system_prompt=(
                    "Classify whether the message means: stop the workflow/job/document path and reply normally in chat. "
                    "Return JSON only."
                ),
                metadata={"component": "chat_pending_correction"},
            )
        )
        llm_result = bool(parsed.get("chat_only_correction"))
        confidence_raw = parsed.get("confidence")
        confidence = float(confidence_raw) if isinstance(confidence_raw, (int, float)) else 0.0
        if CHAT_PENDING_CORRECTION_MODE == "hybrid" and confidence < 0.7:
            return False
        return llm_result
    except Exception:  # noqa: BLE001
        logger.exception("chat_pending_correction_classification_failed")
        return False


def _chat_intent_vector_namespace() -> str:
    digest = hashlib.sha256(
        json.dumps(
            list(_CHAT_INTENT_VECTOR_CATALOG),
            ensure_ascii=True,
            sort_keys=True,
            default=str,
        ).encode("utf-8")
    ).hexdigest()[:16]
    return f"{CHAT_INTENT_VECTOR_NAMESPACE_PREFIX}:{digest}"


def _intent_vector_namespace() -> str:
    digest = hashlib.sha256(
        json.dumps(
            list(_INTENT_VECTOR_CATALOG),
            ensure_ascii=True,
            sort_keys=True,
            default=str,
        ).encode("utf-8")
    ).hexdigest()[:16]
    return f"{INTENT_VECTOR_NAMESPACE_PREFIX}:{digest}"


def _vector_intent_confidence(score: float) -> float:
    bounded = max(0.0, min(1.0, float(score or 0.0)))
    return round(min(0.93, max(0.68, 0.45 + (bounded * 0.5))), 3)


def _ensure_chat_intent_vector_index() -> str | None:
    global _chat_intent_vector_synced_namespace

    if not CHAT_INTENT_VECTOR_SEARCH_ENABLED:
        return None
    namespace = _chat_intent_vector_namespace()
    with _chat_intent_vector_sync_lock:
        if _chat_intent_vector_synced_namespace == namespace:
            return namespace
        try:
            upsert_entries = [
                {
                    "document_id": entry["id"],
                    "text": entry["text"],
                    "source_uri": f"chat-intent://{entry['id']}",
                    "metadata": {
                        "intent_id": entry["id"],
                        "intent_label": entry["label"],
                        "catalog_type": "chat_intent",
                    },
                }
                for entry in _CHAT_INTENT_VECTOR_CATALOG
            ]
            _rag_retriever_request_json(
                "/index/upsert_texts",
                method="POST",
                body={
                    "collection_name": CHAT_INTENT_VECTOR_COLLECTION,
                    "ensure_collection": True,
                    "namespace": namespace,
                    "workspace_id": CHAT_INTENT_VECTOR_WORKSPACE_ID,
                    "entries": upsert_entries,
                },
                timeout_s=CHAT_INTENT_VECTOR_TIMEOUT_S,
            )
        except Exception:  # noqa: BLE001
            logger.exception("chat_intent_vector_index_sync_failed")
            return None
        _chat_intent_vector_synced_namespace = namespace
        return namespace


def _ensure_intent_vector_index() -> str | None:
    global _intent_vector_synced_namespace

    if not INTENT_VECTOR_SEARCH_ENABLED:
        return None
    namespace = _intent_vector_namespace()
    with _intent_vector_sync_lock:
        if _intent_vector_synced_namespace == namespace:
            return namespace
        try:
            upsert_entries = [
                {
                    "document_id": entry["id"],
                    "text": entry["text"],
                    "source_uri": f"intent://{entry['id']}",
                    "metadata": {
                        "intent_id": entry["id"],
                        "intent_label": entry["label"],
                        "catalog_type": "goal_intent",
                    },
                }
                for entry in _INTENT_VECTOR_CATALOG
            ]
            _rag_retriever_request_json(
                "/index/upsert_texts",
                method="POST",
                body={
                    "collection_name": INTENT_VECTOR_COLLECTION,
                    "ensure_collection": True,
                    "namespace": namespace,
                    "workspace_id": INTENT_VECTOR_WORKSPACE_ID,
                    "entries": upsert_entries,
                },
                timeout_s=INTENT_VECTOR_TIMEOUT_S,
            )
        except Exception:  # noqa: BLE001
            logger.exception("intent_vector_index_sync_failed")
            return None
        _intent_vector_synced_namespace = namespace
        return namespace


def _vector_chat_intent_matches(query: str) -> list[dict[str, Any]]:
    namespace = _ensure_chat_intent_vector_index()
    if not namespace or not str(query or "").strip():
        return []
    try:
        result = _rag_retriever_request_json(
            "/retrieve",
            method="POST",
            body={
                "query": query,
                "collection_name": CHAT_INTENT_VECTOR_COLLECTION,
                "namespace": namespace,
                "workspace_id": CHAT_INTENT_VECTOR_WORKSPACE_ID,
                "top_k": CHAT_INTENT_VECTOR_TOP_K,
                "min_score": CHAT_INTENT_VECTOR_MIN_SCORE,
                "include_text": False,
                "include_metadata": True,
            },
            timeout_s=CHAT_INTENT_VECTOR_TIMEOUT_S,
        )
    except Exception:  # noqa: BLE001
        logger.exception("chat_intent_vector_search_failed")
        return []

    matches = result.get("matches") if isinstance(result, dict) else None
    if not isinstance(matches, list):
        return []
    allowed_ids = {entry["id"] for entry in _CHAT_INTENT_VECTOR_CATALOG}
    ranked: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for match in matches:
        if not isinstance(match, dict):
            continue
        metadata = match.get("metadata") if isinstance(match.get("metadata"), dict) else {}
        intent_id = str(metadata.get("intent_id") or match.get("document_id") or "").strip()
        if not intent_id or intent_id in seen_ids or intent_id not in allowed_ids:
            continue
        try:
            score = float(match.get("score") or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        ranked.append(
            {
                "id": intent_id,
                "score": score,
                "source": "vector_search",
            }
        )
        seen_ids.add(intent_id)
    ranked.sort(key=lambda item: (-float(item["score"]), str(item["id"])))
    return ranked


def _vector_goal_intent_matches(goal: str) -> list[dict[str, Any]]:
    namespace = _ensure_intent_vector_index()
    if not namespace or not str(goal or "").strip():
        return []
    try:
        result = _rag_retriever_request_json(
            "/retrieve",
            method="POST",
            body={
                "query": goal,
                "collection_name": INTENT_VECTOR_COLLECTION,
                "namespace": namespace,
                "workspace_id": INTENT_VECTOR_WORKSPACE_ID,
                "top_k": INTENT_VECTOR_TOP_K,
                "min_score": INTENT_VECTOR_MIN_SCORE,
                "include_text": False,
                "include_metadata": True,
            },
            timeout_s=INTENT_VECTOR_TIMEOUT_S,
        )
    except Exception:  # noqa: BLE001
        logger.exception("intent_vector_search_failed")
        return []

    matches = result.get("matches") if isinstance(result, dict) else None
    if not isinstance(matches, list):
        return []
    allowed_ids = {entry["id"] for entry in _INTENT_VECTOR_CATALOG}
    ranked: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for match in matches:
        if not isinstance(match, dict):
            continue
        metadata = match.get("metadata") if isinstance(match.get("metadata"), dict) else {}
        intent_id = str(metadata.get("intent_id") or match.get("document_id") or "").strip()
        if not intent_id or intent_id in seen_ids or intent_id not in allowed_ids:
            continue
        try:
            score = float(match.get("score") or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        ranked.append(
            {
                "id": intent_id,
                "score": score,
                "source": "vector_search",
            }
        )
        seen_ids.add(intent_id)
    ranked.sort(key=lambda item: (-float(item["score"]), str(item["id"])))
    return ranked


def _hybrid_goal_intent_inference(
    goal: str,
    heuristic: intent_contract.TaskIntentInference,
) -> intent_contract.TaskIntentInference:
    matches = _vector_goal_intent_matches(goal)
    if not matches:
        return heuristic
    top_match = matches[0]
    top_score = float(top_match.get("score") or 0.0)
    second_score = float(matches[1].get("score") or 0.0) if len(matches) > 1 else 0.0
    if top_score < INTENT_VECTOR_MIN_SCORE:
        return heuristic
    if second_score and (top_score - second_score) < INTENT_VECTOR_SCORE_MARGIN:
        return heuristic
    vector_intent = intent_contract.normalize_task_intent(top_match.get("id"))
    if not vector_intent:
        return heuristic
    if vector_intent == str(heuristic.intent or "").strip().lower():
        return heuristic
    heuristic_confidence = float(getattr(heuristic, "confidence", 0.0) or 0.0)
    heuristic_source = str(getattr(heuristic, "source", "") or "").strip().lower()
    strong_vector_match = top_score >= max(0.8, INTENT_VECTOR_MIN_SCORE + 0.15)
    if heuristic_source != "default":
        if heuristic_confidence > INTENT_VECTOR_OVERRIDE_MAX_HEURISTIC_CONFIDENCE:
            return heuristic
        if not strong_vector_match:
            return heuristic
    return intent_contract.TaskIntentInference(
        intent=vector_intent,
        source="vector",
        confidence=_vector_intent_confidence(top_score),
    )


def _should_run_goal_intent_vector_search(
    heuristic: intent_contract.TaskIntentInference,
) -> bool:
    if not INTENT_VECTOR_SEARCH_ENABLED:
        return False
    heuristic_source = str(getattr(heuristic, "source", "") or "").strip().lower()
    heuristic_confidence = float(getattr(heuristic, "confidence", 0.0) or 0.0)
    if heuristic_source in {"", "default"}:
        return True
    return heuristic_confidence <= INTENT_VECTOR_TRIGGER_MAX_HEURISTIC_CONFIDENCE


def _should_skip_llm_goal_intent_assessment(
    inference: intent_contract.TaskIntentInference,
    *,
    assess_mode: str,
) -> bool:
    if assess_mode != "hybrid":
        return False
    source = str(getattr(inference, "source", "") or "").strip().lower()
    if source in {"", "default"}:
        return False
    confidence = float(getattr(inference, "confidence", 0.0) or 0.0)
    return confidence >= INTENT_ASSESS_SKIP_LLM_MIN_CONFIDENCE


def _classify_chat_request_intent(content: str) -> str | None:
    matches = _vector_chat_intent_matches(content)
    if not matches:
        return None
    top_match = matches[0]
    top_score = float(top_match.get("score") or 0.0)
    second_score = float(matches[1].get("score") or 0.0) if len(matches) > 1 else 0.0
    if top_score < CHAT_INTENT_VECTOR_MIN_SCORE:
        return None
    if second_score and (top_score - second_score) < CHAT_INTENT_VECTOR_SCORE_MARGIN:
        return None
    intent_id = str(top_match.get("id") or "").strip()
    return intent_id or None


def _fallback_chat_response(content: str) -> str:
    lowered = str(content or "").strip().lower()
    capability_catalog_response = _capability_discovery_chat_response(content)
    if capability_catalog_response:
        return capability_catalog_response
    if lowered in {"hi", "hello", "hey"}:
        return "I can chat, answer questions, and create workflows when execution is needed."
    if lowered in {"thanks", "thank you"}:
        return "You can keep chatting here, or ask me to create a workflow when you want work executed."
    return (
        "I can answer questions directly here, and when you want work executed I can turn that into "
        "a workflow and submit a job."
    )


def _is_capability_discovery_request(content: str) -> bool:
    lowered = str(content or "").strip().lower()
    if not lowered:
        return False
    patterns = (
        r"\bwhat can you do\b",
        r"\bwhat (?:tools|actions|operations) (?:do you have|are available|are supported)\b",
        r"\bwhat can (?:this|the) (?:assistant|agent) do\b",
        r"\bwhat capabilities(?: do you have| are available)?\b",
        r"\bavailable capabilities\b",
        r"\bavailable (?:tools|actions|operations)\b",
        r"\blist (?:your |the )?capabilities\b",
        r"\blist (?:your |the )?(?:tools|actions|operations)\b",
        r"\bshow (?:your |the )?capabilities\b",
        r"\bshow (?:your |the )?(?:tools|actions|operations)\b",
        r"\bwhich capabilities\b",
        r"\bwhich (?:tools|actions|operations)\b",
        r"\bsupported (?:capabilities|tools|actions|operations)\b",
    )
    if any(re.search(pattern, lowered) for pattern in patterns):
        return True
    return _classify_chat_request_intent(content) == "capability_discovery"


def _chat_visible_capabilities() -> list[tuple[str, capability_registry.CapabilitySpec]]:
    if capability_registry.resolve_capability_mode() == "disabled":
        return []
    try:
        registry = capability_registry.load_capability_registry()
    except Exception:  # noqa: BLE001
        return []

    visible: list[tuple[str, capability_registry.CapabilitySpec]] = []
    for capability_id, spec in sorted(registry.enabled_capabilities().items()):
        allow_decision = capability_registry.evaluate_capability_allowlist(
            capability_id,
            "api",
        )
        if not allow_decision.allowed:
            continue
        visible.append((capability_id, spec))
    return visible


def _chat_capability_search_entries(
    capabilities: list[tuple[str, capability_registry.CapabilitySpec]],
) -> list[dict[str, Any]]:
    capability_map = {capability_id: spec for capability_id, spec in capabilities}
    return capability_search.build_capability_search_entries(capability_map)


def _chat_capability_vector_namespace(
    capabilities: list[tuple[str, capability_registry.CapabilitySpec]],
) -> str:
    payload: list[dict[str, Any]] = []
    for capability_id, spec in capabilities:
        payload.append(
            {
                "id": capability_id,
                "description": str(spec.description or "").strip(),
                "group": str(spec.group or "").strip(),
                "subgroup": str(spec.subgroup or "").strip(),
                "risk_tier": str(spec.risk_tier or "").strip(),
                "tags": list(spec.tags),
                "aliases": list(spec.aliases),
            }
        )
    digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=True, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]
    return f"{CHAT_CAPABILITY_VECTOR_NAMESPACE_PREFIX}:{digest}"


def _chat_capability_vector_document(
    capability_id: str,
    spec: capability_registry.CapabilitySpec,
    entry: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    description = str(spec.description or "").strip()
    group = str(spec.group or "").strip()
    subgroup = str(spec.subgroup or "").strip()
    tags = [str(tag).strip() for tag in spec.tags if str(tag).strip()]
    aliases = [str(alias).strip() for alias in spec.aliases if str(alias).strip()]
    required_inputs = [
        str(item).strip()
        for item in entry.get("required_inputs", [])
        if str(item).strip()
    ]
    lines = [f"Capability ID: {capability_id}"]
    if description:
        lines.append(f"Description: {description}")
    if group:
        lines.append(f"Group: {group}")
    if subgroup:
        lines.append(f"Subgroup: {subgroup}")
    if tags:
        lines.append(f"Tags: {', '.join(tags)}")
    if aliases:
        lines.append(f"Aliases: {', '.join(aliases)}")
    if required_inputs:
        lines.append(f"Required inputs: {', '.join(required_inputs)}")
    if spec.risk_tier:
        lines.append(f"Risk tier: {spec.risk_tier}")
    return (
        "\n".join(lines),
        {
            "capability_id": capability_id,
            "group": group,
            "subgroup": subgroup,
            "risk_tier": str(spec.risk_tier or "").strip(),
            "tags": tags,
            "aliases": aliases,
            "catalog_type": "assistant_capability",
        },
    )


def _cleanup_stale_chat_capability_vector_namespaces(current_namespace: str) -> None:
    if not CHAT_CAPABILITY_VECTOR_CLEANUP_ENABLED:
        return
    prefix = f"{CHAT_CAPABILITY_VECTOR_NAMESPACE_PREFIX}:"
    if not current_namespace.startswith(prefix):
        return
    try:
        result = _rag_retriever_request_json(
            "/documents/list",
            method="POST",
            body={
                "collection_name": CHAT_CAPABILITY_VECTOR_COLLECTION,
                "workspace_id": CHAT_CAPABILITY_VECTOR_WORKSPACE_ID,
                "limit": CHAT_CAPABILITY_VECTOR_CLEANUP_LIMIT,
            },
            timeout_s=CHAT_CAPABILITY_VECTOR_TIMEOUT_S,
        )
    except Exception:  # noqa: BLE001
        logger.exception("chat_capability_vector_cleanup_list_failed")
        return
    if not isinstance(result, Mapping):
        return
    if bool(result.get("truncated")):
        logger.warning(
            "chat_capability_vector_cleanup_skipped_truncated",
            extra={"limit": CHAT_CAPABILITY_VECTOR_CLEANUP_LIMIT},
        )
        return
    stale_documents_by_namespace: dict[str, set[str]] = {}
    for document in result.get("documents", []) or []:
        if not isinstance(document, Mapping):
            continue
        namespace = str(document.get("namespace") or "").strip()
        if not namespace.startswith(prefix) or namespace == current_namespace:
            continue
        metadata = document.get("metadata")
        if isinstance(metadata, Mapping):
            catalog_type = str(metadata.get("catalog_type") or "").strip()
            if catalog_type and catalog_type != "assistant_capability":
                continue
        document_id = str(document.get("document_id") or "").strip()
        if document_id:
            stale_documents_by_namespace.setdefault(namespace, set()).add(document_id)
    for namespace, document_ids in stale_documents_by_namespace.items():
        for document_id in sorted(document_ids):
            try:
                _rag_retriever_request_json(
                    "/documents/delete",
                    method="POST",
                    body={
                        "collection_name": CHAT_CAPABILITY_VECTOR_COLLECTION,
                        "namespace": namespace,
                        "workspace_id": CHAT_CAPABILITY_VECTOR_WORKSPACE_ID,
                        "document_id": document_id,
                    },
                    timeout_s=CHAT_CAPABILITY_VECTOR_TIMEOUT_S,
                )
            except Exception:  # noqa: BLE001
                logger.exception(
                    "chat_capability_vector_cleanup_delete_failed",
                    extra={"namespace": namespace, "document_id": document_id},
                )


def _ensure_chat_capability_vector_index(
    capabilities: list[tuple[str, capability_registry.CapabilitySpec]],
    entries: list[dict[str, Any]],
) -> str | None:
    global _chat_capability_vector_synced_namespace

    if not CHAT_CAPABILITY_VECTOR_SEARCH_ENABLED or not capabilities or not entries:
        return None
    namespace = _chat_capability_vector_namespace(capabilities)
    with _chat_capability_vector_sync_lock:
        if _chat_capability_vector_synced_namespace == namespace:
            return namespace
        try:
            upsert_entries: list[dict[str, Any]] = []
            entry_by_id = {
                str(entry.get("id") or "").strip(): entry
                for entry in entries
                if str(entry.get("id") or "").strip()
            }
            for capability_id, spec in capabilities:
                entry = entry_by_id.get(capability_id, {})
                text, metadata = _chat_capability_vector_document(capability_id, spec, entry)
                upsert_entries.append(
                    {
                        "document_id": capability_id,
                        "text": text,
                        "source_uri": f"capability://{capability_id}",
                        "metadata": metadata,
                    }
                )
            _rag_retriever_request_json(
                "/index/upsert_texts",
                method="POST",
                body={
                    "collection_name": CHAT_CAPABILITY_VECTOR_COLLECTION,
                    "ensure_collection": True,
                    "namespace": namespace,
                    "workspace_id": CHAT_CAPABILITY_VECTOR_WORKSPACE_ID,
                    "entries": upsert_entries,
                },
                timeout_s=CHAT_CAPABILITY_VECTOR_TIMEOUT_S,
            )
        except Exception:  # noqa: BLE001
            logger.exception("chat_capability_vector_index_sync_failed")
            return None
        _cleanup_stale_chat_capability_vector_namespaces(namespace)
        _chat_capability_vector_synced_namespace = namespace
        return namespace


def _capability_discovery_scope_query(content: str) -> str:
    lowered = str(content or "").strip().lower()
    if not lowered:
        return ""
    normalized = re.sub(r"[^a-z0-9]+", " ", lowered)
    generic_patterns = (
        r"\bwhat can you do\b",
        r"\bwhat can (?:this|the) (?:assistant|agent) do\b",
        r"\bwhat capabilities(?: do you have| are available)?\b",
        r"\bwhat (?:tools|actions|operations) (?:do you have|are available|are supported)\b",
        r"\bavailable (?:capabilities|tools|actions|operations)\b",
        r"\blist (?:your |the )?(?:capabilities|tools|actions|operations)\b",
        r"\bshow (?:your |the )?(?:capabilities|tools|actions|operations)\b",
        r"\bwhich (?:capabilities|tools|actions|operations)\b",
        r"\bsupported (?:capabilities|tools|actions|operations)\b",
        r"\bwhat (?:kind|kinds|type|types|sort|sorts) of\b",
        r"\bwhat can you help with\b",
        r"\bwhat (?:work|tasks|things|stuff) can you (?:handle|support|do)\b",
    )
    for pattern in generic_patterns:
        normalized = re.sub(pattern, " ", normalized)
    filler_pattern = (
        r"\b(?:related to|about|for|here|assistant|agent|this|the|your|available|supported|"
        r"kind|kinds|type|types|sort|sorts|work|tasks|things|stuff|handle|support|help|can|do)\b"
    )
    normalized = re.sub(filler_pattern, " ", normalized)
    tokens = [
        token
        for token in re.sub(r"\s+", " ", normalized).strip().split()
        if token
        and token
        not in {
            "what",
            "kinds",
            "kind",
            "types",
            "type",
            "sorts",
            "sort",
            "work",
            "tasks",
            "things",
            "stuff",
            "can",
            "you",
            "handle",
            "support",
            "help",
            "do",
            "with",
        }
    ]
    return " ".join(tokens)


def _scoped_chat_visible_capabilities(
    content: str,
    capabilities: list[tuple[str, capability_registry.CapabilitySpec]],
) -> tuple[str, list[tuple[str, capability_registry.CapabilitySpec]]]:
    scope_query = _capability_discovery_scope_query(content)
    if not scope_query or not capabilities:
        return "", capabilities

    capability_map = {capability_id: spec for capability_id, spec in capabilities}
    entries = _chat_capability_search_entries(capabilities)
    lexical_matches = capability_search.search_capabilities(
        query=scope_query,
        capability_entries=entries,
        limit=max(1, len(entries)),
        rerank_feedback_rows=[],
    )
    hybrid_matches = _hybrid_chat_capability_matches(
        query=scope_query,
        capabilities=capabilities,
        lexical_matches=lexical_matches,
        entries=entries,
    )
    if not hybrid_matches:
        return scope_query, []

    matched_capabilities: list[tuple[str, capability_registry.CapabilitySpec]] = []
    for match in hybrid_matches:
        capability_id = str(match.get("id") or "").strip()
        spec = capability_map.get(capability_id)
        if spec is None:
            continue
        matched_capabilities.append((capability_id, spec))
    return scope_query, matched_capabilities or capabilities


def _vector_chat_capability_matches(
    *,
    query: str,
    capabilities: list[tuple[str, capability_registry.CapabilitySpec]],
    entries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    namespace = _ensure_chat_capability_vector_index(capabilities, entries)
    if not namespace:
        return []
    try:
        result = _rag_retriever_request_json(
            "/retrieve",
            method="POST",
            body={
                "query": query,
                "collection_name": CHAT_CAPABILITY_VECTOR_COLLECTION,
                "namespace": namespace,
                "workspace_id": CHAT_CAPABILITY_VECTOR_WORKSPACE_ID,
                "top_k": min(max(1, len(capabilities)), CHAT_CAPABILITY_VECTOR_TOP_K),
                "min_score": CHAT_CAPABILITY_VECTOR_MIN_SCORE,
                "include_text": False,
                "include_metadata": True,
            },
            timeout_s=CHAT_CAPABILITY_VECTOR_TIMEOUT_S,
        )
    except Exception:  # noqa: BLE001
        logger.exception("chat_capability_vector_search_failed")
        return []

    matches = result.get("matches") if isinstance(result, dict) else None
    if not isinstance(matches, list):
        return []
    allowed_ids = {capability_id for capability_id, _spec in capabilities}
    vector_matches: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for match in matches:
        if not isinstance(match, dict):
            continue
        metadata = match.get("metadata") if isinstance(match.get("metadata"), dict) else {}
        capability_id = str(
            metadata.get("capability_id")
            or match.get("document_id")
            or ""
        ).strip()
        if not capability_id or capability_id in seen_ids or capability_id not in allowed_ids:
            continue
        try:
            score = float(match.get("score") or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        vector_matches.append(
            {
                "id": capability_id,
                "score": score,
                "reason": "vector similarity",
                "source": "vector_search",
            }
        )
        seen_ids.add(capability_id)
    return vector_matches


def _hybrid_chat_capability_matches(
    *,
    query: str,
    capabilities: list[tuple[str, capability_registry.CapabilitySpec]],
    lexical_matches: list[dict[str, Any]],
    entries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    vector_matches = _vector_chat_capability_matches(
        query=query,
        capabilities=capabilities,
        entries=entries,
    )
    combined: dict[str, dict[str, Any]] = {}

    for match in lexical_matches:
        capability_id = str(match.get("id") or "").strip()
        if not capability_id:
            continue
        combined[capability_id] = {
            "id": capability_id,
            "lexical_score": float(match.get("score") or 0.0),
            "vector_score": 0.0,
            "reason": str(match.get("reason") or "").strip(),
        }

    for match in vector_matches:
        capability_id = str(match.get("id") or "").strip()
        if not capability_id:
            continue
        current = combined.setdefault(
            capability_id,
            {
                "id": capability_id,
                "lexical_score": 0.0,
                "vector_score": 0.0,
                "reason": "",
            },
        )
        current["vector_score"] = max(current["vector_score"], float(match.get("score") or 0.0))
        if current["reason"]:
            current["reason"] = f"{current['reason']}; vector similarity"
        else:
            current["reason"] = "vector similarity"

    ranked: list[dict[str, Any]] = []
    for item in combined.values():
        lexical_score = float(item.get("lexical_score") or 0.0)
        vector_score = float(item.get("vector_score") or 0.0)
        fused_score = lexical_score + (vector_score * 25.0)
        if lexical_score > 0 and vector_score > 0:
            fused_score += 5.0
        ranked.append(
            {
                "id": item["id"],
                "score": round(fused_score, 6),
                "reason": item["reason"] or "hybrid match",
                "source": (
                    "hybrid_search"
                    if lexical_score > 0 and vector_score > 0
                    else "lexical_search"
                    if lexical_score > 0
                    else "vector_search"
                ),
            }
        )
    ranked.sort(key=lambda item: (-float(item["score"]), str(item["id"])))
    return ranked


def _capability_discovery_chat_response(content: str) -> str:
    if not _is_capability_discovery_request(content):
        return ""
    capabilities = _chat_visible_capabilities()
    scope_query, scoped_capabilities = _scoped_chat_visible_capabilities(content, capabilities)

    if scoped_capabilities:
        header = f"Available capabilities for this assistant ({len(scoped_capabilities)}):"
        if scope_query:
            header = (
                f"Available capabilities related to '{scope_query}' for this assistant "
                f"({len(scoped_capabilities)}):"
            )
        lines = [
            header,
        ]
        for capability_id, spec in scoped_capabilities:
            description = str(spec.description or "").strip()
            if description:
                lines.append(f"- {capability_id}: {description}")
            else:
                lines.append(f"- {capability_id}")
        return "\n".join(lines)

    if scope_query:
        return f"No capabilities related to '{scope_query}' are currently available for this assistant."
    return "No capabilities are currently available for this assistant."


def _conversational_chat_fast_path_envelope(
    *,
    goal: str,
    source: str = "chat_conversational_fast_path",
) -> workflow_contracts.NormalizedIntentEnvelope:
    return workflow_contracts.NormalizedIntentEnvelope(
        goal=str(goal or "").strip(),
        profile=workflow_contracts.GoalIntentProfile(
            intent="other",
            source=source,
            confidence=1.0,
            risk_level="read_only",
            threshold=0.0,
            low_confidence=False,
            needs_clarification=False,
            requires_blocking_clarification=False,
            questions=[],
            blocking_slots=[],
            missing_slots=[],
            slot_values={"intent_action": "other", "risk_level": "read_only"},
            clarification_mode=source,
        ),
        graph=workflow_contracts.IntentGraph(
            segments=[],
            source=source,
            overall_confidence=1.0,
        ),
        candidate_capabilities={},
        clarification=workflow_contracts.ClarificationState(
            needs_clarification=False,
            requires_blocking_clarification=False,
            missing_inputs=[],
            questions=[],
            blocking_slots=[],
            slot_values={"intent_action": "other", "risk_level": "read_only"},
            clarification_mode=source,
        ),
        trace=workflow_contracts.NormalizationTrace(
            assessment_source=source,
            assessment_mode="fast_path",
            assessment_fallback_used=False,
            decomposition_source="disabled",
            decomposition_mode="disabled",
            decomposition_fallback_used=False,
        ),
    )


def _plan_derived_normalized_intent_envelope(
    *,
    goal: str,
    plan: models.PlanCreate,
    source: str,
) -> workflow_contracts.NormalizedIntentEnvelope:
    segments: list[workflow_contracts.IntentGraphSegment] = []
    candidate_capabilities: dict[str, list[str]] = {}
    intent_order: list[str] = []
    for index, task in enumerate(plan.tasks):
        request_ids = _task_request_ids_for_preflight(task)
        task_intent = _preflight_task_intent(task, goal_text=goal) or "io"
        objective = (
            str(task.instruction or "").strip()
            or str(task.description or "").strip()
            or str(task.name or "").strip()
            or str(goal or "").strip()
        )
        segment_id = f"s{index + 1}"
        slots = intent_contract.normalize_intent_segment_slots(
            raw_slots={"must_have_inputs": []},
            intent=task_intent,
            objective=objective,
            required_inputs=(),
            suggested_capabilities=request_ids,
            fallback_slots={"must_have_inputs": []},
        )
        segments.append(
            workflow_contracts.IntentGraphSegment(
                id=segment_id,
                intent=task_intent,
                objective=objective,
                source=source,
                confidence=1.0,
                depends_on=[],
                required_inputs=[],
                suggested_capabilities=request_ids,
                slots=workflow_contracts.IntentGraphSlots.model_validate(slots),
            )
        )
        if request_ids:
            candidate_capabilities[segment_id] = list(request_ids)
        intent_order.append(task_intent)

    unique_intents = list(dict.fromkeys(intent_order))
    overall_intent = unique_intents[0] if len(unique_intents) == 1 else "other"
    return workflow_contracts.NormalizedIntentEnvelope(
        goal=str(goal or "").strip(),
        profile=workflow_contracts.GoalIntentProfile(
            intent=overall_intent,
            source=source,
            confidence=1.0,
            risk_level="read_only",
            threshold=0.0,
            low_confidence=False,
            needs_clarification=False,
            requires_blocking_clarification=False,
            questions=[],
            blocking_slots=[],
            missing_slots=[],
            slot_values={
                "intent_action": overall_intent,
                "risk_level": "read_only",
            },
            clarification_mode=source,
        ),
        graph=workflow_contracts.IntentGraph(
            segments=segments,
            summary=workflow_contracts.IntentGraphSummary(
                segment_count=len(segments),
                intent_order=intent_order,
                schema_version="intent_v2",
            ),
            overall_confidence=1.0,
            source=source,
        ),
        candidate_capabilities=candidate_capabilities,
        clarification=workflow_contracts.ClarificationState(
            needs_clarification=False,
            requires_blocking_clarification=False,
            missing_inputs=[],
            questions=[],
            blocking_slots=[],
            slot_values={
                "intent_action": overall_intent,
                "risk_level": "read_only",
            },
            clarification_mode=source,
        ),
        trace=workflow_contracts.NormalizationTrace(
            assessment_source=source,
            assessment_mode="compiled_plan",
            assessment_fallback_used=False,
            decomposition_source=source,
            decomposition_mode="compiled_plan",
            decomposition_fallback_used=False,
        ),
    )


def _chat_response_turn_plan(
    *,
    goal: str,
    assistant_content: str,
    clear_pending_clarification: bool = False,
    source: str = "chat_boundary_decision",
) -> dict[str, Any]:
    resolved_goal = str(goal or "").strip()
    normalized = _conversational_chat_fast_path_envelope(goal=goal, source=source)
    resolved_assistant_content = (
        _capability_discovery_chat_response(resolved_goal)
        or str(assistant_content or "").strip()
        or _fallback_chat_response(resolved_goal)
    )
    return {
        "type": "respond",
        "assistant_content": resolved_assistant_content,
        "clarification_questions": [],
        "goal_intent_profile": workflow_contracts.dump_goal_intent_profile(normalized.profile)
        or {},
        "normalized_intent_envelope": (
            workflow_contracts.dump_normalized_intent_envelope(normalized) or {}
        ),
        "resolved_goal": resolved_goal,
        "clear_pending_clarification": clear_pending_clarification,
        "response_generated": True,
    }


def _chat_clarification_turn_plan(
    *,
    goal: str,
    clarification_questions: Sequence[str],
    session_metadata: Mapping[str, Any] | None = None,
    assistant_content: str = "",
    source: str = "chat_clarification",
) -> dict[str, Any]:
    resolved_goal = str(goal or "").strip()
    normalized_questions: list[str] = []
    for raw_question in clarification_questions:
        if not isinstance(raw_question, str):
            continue
        question = raw_question.strip()
        if question and question not in normalized_questions:
            normalized_questions.append(question)

    pending_state = _pending_clarification_state_from_metadata(session_metadata)
    assessment = (
        dict(pending_state.goal_intent_profile)
        if pending_state is not None and isinstance(pending_state.goal_intent_profile, Mapping)
        else {}
    )
    existing_questions = [
        str(question).strip()
        for question in assessment.get("questions", [])
        if isinstance(question, str) and str(question).strip()
    ]
    if normalized_questions:
        assessment["questions"] = normalized_questions
    elif existing_questions:
        normalized_questions = existing_questions
    assessment["needs_clarification"] = True
    assessment["requires_blocking_clarification"] = True
    assessment["source"] = str(assessment.get("source") or source).strip() or source

    resolved_assistant_content = (
        str(assistant_content or "").strip()
        or "\n".join(normalized_questions)
        or "I still need the remaining required details before I can continue."
    )
    return {
        "type": "ask_clarification",
        "assistant_content": resolved_assistant_content,
        "clarification_questions": normalized_questions,
        "goal_intent_profile": assessment,
        "resolved_goal": resolved_goal,
    }


def _chat_boundary_failure_response(
    *,
    content: str,
    session_metadata: Mapping[str, Any] | None = None,
    pending_clarification: bool,
) -> dict[str, Any]:
    if pending_clarification:
        return _chat_clarification_turn_plan(
            goal=content.strip(),
            clarification_questions=[
                "I can either continue the current workflow request or answer here in chat. Tell me which one you want."
            ],
            session_metadata=session_metadata,
            source="chat_boundary_failure",
        )
    return _chat_response_turn_plan(
        goal=content.strip(),
        assistant_content=_fallback_chat_response(content),
        source="chat_boundary_failure",
    )


def _chat_router_failure_response(
    *,
    content: str,
    session_metadata: Mapping[str, Any] | None = None,
    pending_clarification: bool,
) -> dict[str, Any]:
    if pending_clarification:
        return _chat_clarification_turn_plan(
            goal=content.strip(),
            clarification_questions=[
                "I could not determine how to continue the current workflow request. Reply with the missing detail, or say you want the answer here in chat."
            ],
            session_metadata=session_metadata,
            source="chat_router_failure",
        )
    return _chat_response_turn_plan(
        goal=content.strip(),
        assistant_content=(
            "I could not determine a safe execution route for that request. "
            "Rephrase the task, or ask me to answer here in chat instead."
        ),
        source="chat_router_failure",
    )


def _goal_intent_segments_from_metadata(metadata: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    graph = _goal_intent_graph_from_metadata(metadata)
    if graph is None:
        return []
    segments: list[dict[str, Any]] = []
    for raw_segment in graph.segments:
        normalized = _normalize_task_intent_profile_segment(
            raw_segment.model_dump(mode="json", exclude_none=True)
        )
        if normalized is not None:
            segments.append(normalized)
    return segments


def _candidate_capabilities_from_intent_graph(
    graph: workflow_contracts.IntentGraph | None,
) -> dict[str, list[str]]:
    if graph is None:
        return {}
    candidates: dict[str, list[str]] = {}
    for segment in graph.segments:
        deduped: list[str] = []
        for capability_id in segment.suggested_capabilities:
            normalized = str(capability_id or "").strip()
            if normalized and normalized not in deduped:
                deduped.append(normalized)
        if deduped:
            candidates[segment.id] = deduped
    return candidates


def _normalized_intent_envelope_from_metadata(
    metadata: Mapping[str, Any] | None,
    *,
    goal: str = "",
) -> workflow_contracts.NormalizedIntentEnvelope | None:
    if not isinstance(metadata, Mapping):
        return None
    envelope = workflow_contracts.parse_normalized_intent_envelope(
        metadata.get("normalized_intent_envelope")
    )
    if envelope is not None:
        return envelope
    profile = workflow_contracts.parse_goal_intent_profile(metadata.get("goal_intent_profile"))
    graph = workflow_contracts.parse_intent_graph(metadata.get("goal_intent_graph"))
    if profile is None and graph is None:
        return None
    graph = graph or workflow_contracts.IntentGraph()
    profile = profile or workflow_contracts.GoalIntentProfile()
    return workflow_contracts.NormalizedIntentEnvelope(
        goal=str(goal or "").strip(),
        profile=profile,
        graph=graph,
        candidate_capabilities=_candidate_capabilities_from_intent_graph(graph),
        clarification=workflow_contracts.ClarificationState(
            needs_clarification=bool(profile.needs_clarification),
            requires_blocking_clarification=bool(profile.requires_blocking_clarification),
            missing_inputs=list(profile.missing_slots),
            questions=list(profile.questions),
            blocking_slots=list(profile.blocking_slots),
            slot_values=dict(profile.slot_values),
            clarification_mode=profile.clarification_mode,
        ),
    )


def _goal_intent_graph_from_metadata(
    metadata: Mapping[str, Any] | None,
    *,
    goal: str = "",
) -> workflow_contracts.IntentGraph | None:
    envelope = _normalized_intent_envelope_from_metadata(metadata, goal=goal)
    if envelope is None:
        return None
    if not envelope.graph.segments:
        return None
    return envelope.graph


def _normalization_response_fields(
    metadata: Mapping[str, Any] | None,
    *,
    goal: str = "",
) -> dict[str, Any]:
    envelope = _normalized_intent_envelope_from_metadata(metadata, goal=goal)
    if envelope is None:
        return {
            "goal_intent_profile": {},
            "goal_intent_graph": None,
            "normalized_intent_envelope": {},
            "normalization_trace": {},
            "normalization_clarification": {},
            "normalization_candidate_capabilities": {},
        }
    envelope_json = workflow_contracts.dump_normalized_intent_envelope(envelope) or {}
    return {
        "goal_intent_profile": workflow_contracts.dump_goal_intent_profile(envelope.profile) or {},
        "goal_intent_graph": workflow_contracts.dump_intent_graph(envelope.graph),
        "normalized_intent_envelope": envelope_json,
        "normalization_trace": (
            dict(envelope_json.get("trace"))
            if isinstance(envelope_json.get("trace"), Mapping)
            else {}
        ),
        "normalization_clarification": (
            dict(envelope_json.get("clarification"))
            if isinstance(envelope_json.get("clarification"), Mapping)
            else {}
        ),
        "normalization_candidate_capabilities": {
            str(segment_id): [
                str(capability_id).strip()
                for capability_id in capability_ids
                if str(capability_id).strip()
            ]
            for segment_id, capability_ids in envelope.candidate_capabilities.items()
            if str(segment_id).strip()
        },
    }


def _assess_goal_intent(
    goal: str,
    *,
    mode_override: str | None = None,
) -> workflow_contracts.GoalIntentProfile:
    return intent_service.assess_goal_intent(
        goal,
        config=_goal_intent_assess_config(),
        runtime=intent_service.GoalIntentRuntime(
            infer_task_intent=lambda goal_text: _infer_goal_intent_with_metadata(
                goal_text,
                mode_override=mode_override,
            ),
            record_metrics=_record_goal_intent_assessment_metrics,
        ),
    )


def _normalize_goal_intent(
    goal: str,
    *,
    db: Session | None = None,
    user_id: str | None = None,
    interaction_summaries: list[dict[str, Any]] | None = None,
    context_envelope: workflow_contracts.ContextEnvelope | Mapping[str, Any] | None = None,
    include_decomposition: bool | None = None,
    assessment_mode_override: str | None = None,
) -> workflow_contracts.NormalizedIntentEnvelope:
    decomposition_enabled = (
        INTENT_DECOMPOSE_ENABLED if include_decomposition is None else bool(include_decomposition)
    )
    normalized_assessment_mode = str(assessment_mode_override or "").strip().lower()
    if normalized_assessment_mode not in {"heuristic", "llm", "hybrid"}:
        normalized_assessment_mode = (
            INTENT_ASSESS_MODE if INTENT_ASSESS_ENABLED else "disabled"
        )
    intent_context = (
        context_service.intent_context_view(context_envelope)
        if context_envelope is not None
        else {}
    )
    if interaction_summaries is None:
        raw_summaries = intent_context.get("interaction_summaries")
        if isinstance(raw_summaries, list):
            interaction_summaries = [
                dict(item) for item in raw_summaries if isinstance(item, Mapping)
            ]
    return intent_service.normalize_goal_intent(
        goal,
        db=db,
        user_id=user_id,
        interaction_summaries=interaction_summaries,
        intent_context=intent_context,
        config=intent_service.IntentNormalizeConfig(
            include_decomposition=decomposition_enabled,
            assessment_mode=normalized_assessment_mode,
            assessment_model=(
                (INTENT_ASSESS_MODEL or INTENT_DECOMPOSE_MODEL or LLM_MODEL_NAME or "").strip()
                if normalized_assessment_mode not in {"disabled", "heuristic"}
                else ""
            ),
            decomposition_mode=INTENT_DECOMPOSE_MODE if decomposition_enabled else "disabled",
            decomposition_model=(
                (INTENT_DECOMPOSE_MODEL or LLM_MODEL_NAME or "").strip()
                if decomposition_enabled
                else ""
            ),
        ),
        runtime=intent_service.IntentNormalizeRuntime(
            assess_goal_intent=lambda goal_text: _assess_goal_intent(
                goal_text,
                mode_override=normalized_assessment_mode,
            ),
            assess_goal_intent_heuristic=lambda goal_text: _assess_goal_intent(
                goal_text,
                mode_override="heuristic",
            ),
            decompose_goal_intent=_decompose_goal_intent,
            capability_required_inputs=_capability_required_inputs_for_intent_normalization,
        ),
    )


def _attach_interaction_compaction_to_envelope(
    normalized: workflow_contracts.NormalizedIntentEnvelope,
    compaction: Mapping[str, Any],
) -> workflow_contracts.NormalizedIntentEnvelope:
    return normalized.model_copy(
        update={
            "graph": _attach_interaction_compaction_to_graph(
                normalized.graph,
                compaction,
            )
        }
    )


def _normalized_intent_response_payload(
    normalized: workflow_contracts.NormalizedIntentEnvelope,
    *,
    include_legacy_graph: bool = False,
) -> dict[str, Any]:
    payload = {
        "goal": normalized.goal,
        "assessment": workflow_contracts.dump_goal_intent_profile(normalized.profile) or {},
        "normalized_intent_envelope": (
            workflow_contracts.dump_normalized_intent_envelope(normalized) or {}
        ),
    }
    if include_legacy_graph:
        payload["intent_graph"] = workflow_contracts.dump_intent_graph(normalized.graph) or {}
    return payload


def _route_chat_turn(
    *,
    content: str,
    candidate_goal: str,
    session_metadata: Mapping[str, Any] | None,
    merged_context: Mapping[str, Any] | None,
    messages: Sequence[chat_contracts.ChatMessage] | None,
) -> dict[str, Any]:
    active_job_confirmation = _active_job_confirmation_turn_plan(
        content=content,
        session_metadata=session_metadata,
    )
    if active_job_confirmation is not None:
        return active_job_confirmation
    pending_clarification = bool(
        isinstance(session_metadata, Mapping) and session_metadata.get("pending_clarification")
    )
    if CHAT_RESPONSE_MODE != "answer_or_handoff":
        return _route_chat_turn_legacy(
            content=content,
            candidate_goal=candidate_goal,
            session_metadata=session_metadata,
            merged_context=merged_context,
            messages=messages,
        )
    if CHAT_ROUTING_MODE == "always_router":
        return _route_chat_turn_with_router(
            content=content,
            candidate_goal=candidate_goal,
            session_metadata=session_metadata,
            merged_context=merged_context,
            messages=messages,
        )
    boundary = _generate_chat_boundary_decision(
        content=content,
        candidate_goal=candidate_goal,
        session_metadata=session_metadata,
        merged_context=merged_context,
        messages=messages,
    )
    if boundary is None:
        return _chat_boundary_failure_response(
            content=content,
            session_metadata=session_metadata,
            pending_clarification=pending_clarification,
        )
    boundary = _postprocess_chat_boundary_decision(boundary, content=content)
    _record_chat_boundary_decision_metrics(boundary)
    decision = boundary.decision
    if decision == chat_contracts.ChatBoundaryDecisionType.chat_reply:
        return _attach_chat_boundary_decision(
            _chat_response_turn_plan(
                goal=content.strip(),
                assistant_content=boundary.assistant_response or _fallback_chat_response(content),
            ),
            boundary,
        )
    if decision == chat_contracts.ChatBoundaryDecisionType.exit_pending_to_chat:
        return _attach_chat_boundary_decision(
            _chat_response_turn_plan(
                goal=content.strip(),
                assistant_content=boundary.assistant_response or _fallback_chat_response(content),
                clear_pending_clarification=True,
            ),
            boundary,
        )
    if decision == chat_contracts.ChatBoundaryDecisionType.meta_clarification:
        if pending_clarification:
            return _attach_chat_boundary_decision(
                _chat_clarification_turn_plan(
                    goal=candidate_goal,
                    clarification_questions=[
                        boundary.assistant_response
                        or (
                            "Do you want to continue the current workflow request, or should I answer "
                            "here in chat instead?"
                        )
                    ],
                    session_metadata=session_metadata,
                    source="chat_boundary_meta_clarification",
                ),
                boundary,
            )
        return _attach_chat_boundary_decision(
            _chat_response_turn_plan(
                goal=content.strip(),
                assistant_content=boundary.assistant_response
                or (
                    "Do you want to continue the current workflow request, or should I answer "
                    "here in chat instead?"
                ),
                source="chat_boundary_meta_clarification",
            ),
            boundary,
        )
    if decision in {
        chat_contracts.ChatBoundaryDecisionType.execution_request,
        chat_contracts.ChatBoundaryDecisionType.continue_pending,
    }:
        return _attach_chat_boundary_decision(
            _route_chat_turn_with_router(
                content=content,
                candidate_goal=candidate_goal,
                session_metadata=session_metadata,
                merged_context=merged_context,
                messages=messages,
            ),
            boundary,
        )
    return _chat_boundary_failure_response(
        content=content,
        session_metadata=session_metadata,
        pending_clarification=pending_clarification,
    )


def _attach_chat_boundary_decision(
    turn_plan: Mapping[str, Any],
    boundary: chat_contracts.ChatBoundaryDecision | None,
) -> dict[str, Any]:
    if boundary is None:
        return dict(turn_plan)
    enriched = dict(turn_plan)
    enriched["boundary_decision"] = boundary.model_dump(mode="json", exclude_none=True)
    return enriched


def _route_chat_turn_legacy(
    *,
    content: str,
    candidate_goal: str,
    session_metadata: Mapping[str, Any] | None,
    merged_context: Mapping[str, Any] | None,
    messages: Sequence[chat_contracts.ChatMessage] | None,
) -> dict[str, Any]:
    fallback = _fallback_chat_turn_route(
        content=content,
        candidate_goal=candidate_goal,
        session_metadata=session_metadata,
        merged_context=merged_context,
    )
    pending_clarification = bool(
        isinstance(session_metadata, Mapping) and session_metadata.get("pending_clarification")
    )
    workflow_invocation = chat_service.workflow_invocation_from_context(merged_context)
    exit_pending_to_chat = (
        pending_clarification
        and workflow_invocation is None
        and _looks_like_chat_only_correction(content)
    )
    should_use_router = (
        CHAT_ROUTING_MODE == "always_router"
        or (pending_clarification and not exit_pending_to_chat)
        or workflow_invocation is not None
        or not _looks_like_conversational_turn(content)
    )
    if not should_use_router:
        return _finalize_chat_turn_plan(
            fallback,
            content=content,
            candidate_goal=candidate_goal,
            merged_context=merged_context,
            messages=messages,
        )
    if _chat_router_provider is None:
        return _finalize_chat_turn_plan(
            fallback,
            content=content,
            candidate_goal=candidate_goal,
            merged_context=merged_context,
            messages=messages,
        )
    try:
        prompt = _build_chat_router_prompt(
            content=content,
            candidate_goal=candidate_goal,
            pending_clarification=pending_clarification,
            merged_context=merged_context,
            messages=messages,
        )
        parsed = _chat_router_provider.generate_request_json_object(
            LLMRequest(
                prompt=prompt,
                system_prompt=(
                    "You route chat turns for an agent platform. "
                    "Return JSON only. "
                    "Use route='respond' for normal conversation or explanation when no tools/workflow are needed. "
                    "Use route='tool_call' only for a single safe read-only capability from the allowed catalog as a synchronous one-step run. "
                    "Use route='run_workflow' only when the user wants to invoke a published Studio workflow and the provided context already includes workflow_trigger_id, workflow_version_id, or workflow_definition_id. "
                    "Use route='submit_job' only when the user wants the system to perform work, create artifacts, inspect systems, or run automation. "
                    "Use route='ask_clarification' only when workflow execution is needed but essential details are missing. "
                    "Never choose tool_call for writes, multi-step work, or anything outside the allowed direct catalog."
                ),
                metadata={
                    "component": "chat_router",
                    "pending_clarification": str(pending_clarification).lower(),
                },
            )
        )
        return _finalize_chat_turn_plan(
            _normalize_chat_route(
                parsed,
                content=content,
                candidate_goal=candidate_goal,
                workflow_invocation_available=workflow_invocation is not None,
            ),
            content=content,
            candidate_goal=candidate_goal,
            merged_context=merged_context,
            messages=messages,
        )
    except Exception:  # noqa: BLE001
        logger.exception("chat_router_failed")
        return _finalize_chat_turn_plan(
            fallback,
            content=content,
            candidate_goal=candidate_goal,
            merged_context=merged_context,
            messages=messages,
        )


def _route_chat_turn_with_router(
    *,
    content: str,
    candidate_goal: str,
    session_metadata: Mapping[str, Any] | None,
    merged_context: Mapping[str, Any] | None,
    messages: Sequence[chat_contracts.ChatMessage] | None,
) -> dict[str, Any]:
    pending_clarification = bool(
        isinstance(session_metadata, Mapping) and session_metadata.get("pending_clarification")
    )
    workflow_invocation = chat_service.workflow_invocation_from_context(merged_context)
    if _chat_router_provider is None:
        return _chat_router_failure_response(
            content=content,
            session_metadata=session_metadata,
            pending_clarification=pending_clarification,
        )
    try:
        prompt = _build_chat_router_prompt(
            content=content,
            candidate_goal=candidate_goal,
            pending_clarification=pending_clarification,
            merged_context=merged_context,
            messages=messages,
        )
        parsed = _chat_router_provider.generate_request_json_object(
            LLMRequest(
                prompt=prompt,
                system_prompt=(
                    "You route chat turns for an agent platform. "
                    "Return JSON only. "
                    "Use route='respond' for normal conversation or explanation when no tools/workflow are needed. "
                    "Use route='tool_call' only for a single safe read-only capability from the allowed catalog as a synchronous one-step run. "
                    "Use route='run_workflow' only when the user wants to invoke a published Studio workflow and the provided context already includes workflow_trigger_id, workflow_version_id, or workflow_definition_id. "
                    "Use route='submit_job' only when the user wants the system to perform work, create artifacts, inspect systems, or run automation. "
                    "Use route='ask_clarification' only when workflow execution is needed but essential details are missing. "
                    "Never choose tool_call for writes, multi-step work, or anything outside the allowed direct catalog."
                ),
                metadata={
                    "component": "chat_router",
                    "pending_clarification": str(pending_clarification).lower(),
                },
            )
        )
        return _finalize_chat_turn_plan(
            _normalize_chat_route(
                parsed,
                content=content,
                candidate_goal=candidate_goal,
                workflow_invocation_available=workflow_invocation is not None,
            ),
            content=content,
            candidate_goal=candidate_goal,
            merged_context=merged_context,
            messages=messages,
        )
    except Exception:  # noqa: BLE001
        logger.exception("chat_router_failed")
        return _chat_router_failure_response(
            content=content,
            session_metadata=session_metadata,
            pending_clarification=pending_clarification,
        )


def _fallback_chat_turn_route(
    *,
    content: str,
    candidate_goal: str,
    session_metadata: Mapping[str, Any] | None,
    merged_context: Mapping[str, Any] | None,
) -> dict[str, Any]:
    pending_clarification = bool(
        isinstance(session_metadata, Mapping) and session_metadata.get("pending_clarification")
    )
    workflow_invocation = chat_service.workflow_invocation_from_context(merged_context)
    exit_pending_to_chat = (
        pending_clarification
        and workflow_invocation is None
        and _looks_like_chat_only_correction(content)
    )
    if exit_pending_to_chat:
        normalized = _conversational_chat_fast_path_envelope(goal=content.strip())
        return {
            "type": "respond",
            "assistant_content": _fallback_chat_response(content),
            "clarification_questions": [],
            "goal_intent_profile": workflow_contracts.dump_goal_intent_profile(normalized.profile)
            or {},
            "normalized_intent_envelope": (
                workflow_contracts.dump_normalized_intent_envelope(normalized) or {}
            ),
            "clear_pending_clarification": True,
        }
    if not pending_clarification and workflow_invocation is None and _looks_like_conversational_turn(content):
        normalized = _conversational_chat_fast_path_envelope(goal=candidate_goal)
        return {
            "type": "respond",
            "assistant_content": _fallback_chat_response(content),
            "clarification_questions": [],
            "goal_intent_profile": workflow_contracts.dump_goal_intent_profile(normalized.profile)
            or {},
            "normalized_intent_envelope": (
                workflow_contracts.dump_normalized_intent_envelope(normalized) or {}
            ),
        }
    normalized = _normalize_goal_intent(candidate_goal)
    assessment = _chat_route_goal_intent_profile(normalized.profile, goal=candidate_goal)
    assessment_json = workflow_contracts.dump_goal_intent_profile(assessment) or {}
    normalized_json = workflow_contracts.dump_normalized_intent_envelope(normalized) or {}
    active_target = (
        _active_execution_target_for_chat(
            normalized=normalized,
            session_metadata=session_metadata,
            merged_context=merged_context,
        )
        if pending_clarification
        else None
    )
    if pending_clarification or not _looks_like_conversational_turn(content):
        if workflow_invocation is not None:
            return {
                "type": "run_workflow",
                "assistant_content": "",
                "clarification_questions": [],
                "goal_intent_profile": assessment_json,
                "normalized_intent_envelope": normalized_json,
            }
        if bool(assessment.requires_blocking_clarification):
            scoped_fields = (
                list(active_target.unresolved_fields or active_target.required_fields)
                if active_target is not None
                else []
            )
            questions = _chat_submit_clarification_questions(
                normalized,
                goal=candidate_goal,
                unresolved_fields=scoped_fields,
                scoped_fields=scoped_fields,
            )
            if not questions:
                questions = [
                    str(question).strip()
                    for question in assessment.questions
                    if isinstance(question, str) and question.strip()
                ]
            return {
                "type": "ask_clarification",
                "assistant_content": "\n".join(questions) if questions else "What should I do next?",
                "clarification_questions": questions,
                "goal_intent_profile": assessment_json,
                "normalized_intent_envelope": normalized_json,
            }
        return {
            "type": "submit_job",
            "assistant_content": "",
            "clarification_questions": [],
            "goal_intent_profile": assessment_json,
            "normalized_intent_envelope": normalized_json,
        }
    return {
        "type": "respond",
        "assistant_content": _fallback_chat_response(content),
        "clarification_questions": [],
        "goal_intent_profile": assessment_json,
        "normalized_intent_envelope": normalized_json,
    }


def _build_chat_router_prompt(
    *,
    content: str,
    candidate_goal: str,
    pending_clarification: bool,
    merged_context: Mapping[str, Any] | None,
    messages: Sequence[chat_contracts.ChatMessage] | None,
) -> str:
    workflow_invocation = chat_service.workflow_invocation_from_context(merged_context)
    recent_messages: list[dict[str, str]] = []
    for message in list(messages or [])[-6:]:
        recent_messages.append(
            {
                "role": str(message.role),
                "content": str(message.content or "")[:500],
            }
        )
    direct_capabilities = [
        {
            "id": capability_id,
            "description": _chat_direct_capability_description(capability_id),
        }
        for capability_id in sorted(CHAT_DIRECT_CAPABILITIES)
    ]
    payload = {
        "current_user_message": content,
        "candidate_goal": candidate_goal,
        "pending_clarification": pending_clarification,
        "context_json": dict(merged_context or {}),
        "recent_messages": recent_messages,
        "direct_capabilities": direct_capabilities,
        "response_schema": {
            "route": "respond | tool_call | ask_clarification | submit_job | run_workflow",
            "assistant_response": "string",
            "intent": "generate | transform | validate | render | io | other",
            "risk_level": "read_only | bounded_write | high_risk_write",
            "confidence": "0..1",
            "output_format": "string",
            "target_system": "string",
            "safety_constraints": "string",
            "workflow_reference": "use run_workflow only when workflow_context.target_available is true",
            "capability_id": "string",
            "arguments": {"any": "json object"},
            "clarification_questions": ["string"],
        },
        "workflow_context": {
            "target_available": workflow_invocation is not None,
            "trigger_id": workflow_invocation.trigger_id if workflow_invocation is not None else None,
            "version_id": workflow_invocation.version_id if workflow_invocation is not None else None,
            "definition_id": workflow_invocation.definition_id
            if workflow_invocation is not None
            else None,
            "input_keys": sorted(workflow_invocation.inputs.keys())
            if workflow_invocation is not None
            else [],
        },
    }
    return (
        "Decide whether this turn should stay conversational or become a workflow request.\n"
        "Rules:\n"
        "- respond: answer normally, no workflow/job needed.\n"
        "- tool_call: execute exactly one safe read-only capability from direct_capabilities as a synchronous one-step run.\n"
        "- run_workflow: invoke the referenced published Studio workflow from workflow_context.\n"
        "- ask_clarification: workflow is needed, but essential details are missing.\n"
        "- submit_job: workflow/job should be created now.\n"
        "- If pending_clarification is true, treat the user message as an attempt to complete an existing workflow request.\n"
        "- Ask for safety constraints only when a high-risk write workflow is actually being requested.\n"
        "- Use tool_call only when one direct capability is sufficient and a single synchronous run is enough.\n"
        "- Use run_workflow only when workflow_context.target_available is true.\n"
        "- Prefer run_workflow over submit_job when the user wants to run an already-published Studio workflow.\n"
        "- Return JSON only.\n\n"
        f"{json.dumps(payload, ensure_ascii=True)}"
    )


def _build_chat_response_prompt(
    *,
    content: str,
    candidate_goal: str,
    merged_context: Mapping[str, Any] | None,
    messages: Sequence[chat_contracts.ChatMessage] | None,
) -> str:
    recent_messages: list[dict[str, str]] = []
    for message in list(messages or [])[-8:]:
        recent_messages.append(
            {
                "role": str(message.role),
                "content": str(message.content or "")[:800],
            }
        )
    payload = {
        "current_user_message": content,
        "candidate_goal": candidate_goal,
        "context_json": dict(merged_context or {}),
        "recent_messages": recent_messages,
    }
    return json.dumps(payload, ensure_ascii=True)


def _chat_boundary_query_text(content: str, candidate_goal: str) -> str:
    normalized_goal = str(candidate_goal or "").strip()
    normalized_content = str(content or "").strip()
    if normalized_goal and normalized_goal.lower() != normalized_content.lower():
        return f"{normalized_goal}\n{normalized_content}"
    return normalized_goal or normalized_content


def _chat_boundary_field_hint_tokens(pending_fields: Sequence[str] | None) -> list[str]:
    hint_tokens: list[str] = []
    field_hints = {
        "path": ("path", "filename", "output"),
        "output_format": ("format", "docx", "pdf", "markdown"),
        "query": ("query", "search"),
        "tone": ("tone", "style"),
        "audience": ("audience", "role"),
        "topic": ("topic", "title"),
        "instruction": ("instruction", "details"),
        "target_system": ("target", "system"),
    }
    for raw_field in pending_fields or ():
        field_name = intent_contract.normalize_required_input_key(raw_field)
        if not field_name:
            continue
        for token in field_hints.get(field_name, (field_name,)):
            normalized = str(token or "").strip().lower()
            if normalized and normalized not in hint_tokens:
                hint_tokens.append(normalized)
    return hint_tokens


def _chat_boundary_scoped_query_text(
    *,
    query: str,
    preferred_family: str | None = None,
    preferred_capability_ids: Sequence[str] | None = None,
    pending_fields: Sequence[str] | None = None,
) -> str:
    parts: list[str] = []
    normalized_query = str(query or "").strip()
    if normalized_query:
        parts.append(normalized_query)
    normalized_family = str(preferred_family or "").strip()
    if normalized_family:
        parts.append(normalized_family)
    capability_hints: list[str] = []
    for raw_capability_id in preferred_capability_ids or ():
        capability_id = capability_registry.canonicalize_capability_id(raw_capability_id)
        if not capability_id:
            continue
        hint = capability_id.replace(".", " ").strip().lower()
        if hint and hint not in capability_hints:
            capability_hints.append(hint)
    if capability_hints:
        parts.append(" ".join(capability_hints[:2]))
    field_hints = _chat_boundary_field_hint_tokens(pending_fields)
    if field_hints:
        parts.append(" ".join(field_hints))
    return "\n".join(part for part in parts if part)


def _chat_boundary_score(value: Any) -> float:
    try:
        return max(0.0, float(value or 0.0))
    except (TypeError, ValueError):
        return 0.0


def _chat_boundary_capability_evidence(
    *,
    query: str,
    preferred_family: str | None = None,
    preferred_capability_ids: Sequence[str] = (),
    pending_fields: Sequence[str] = (),
) -> tuple[
    list[chat_contracts.ChatBoundaryCapabilityEvidence],
    list[chat_contracts.ChatBoundaryFamilyEvidence],
]:
    normalized_query = _chat_boundary_scoped_query_text(
        query=query,
        preferred_family=preferred_family,
        preferred_capability_ids=preferred_capability_ids,
        pending_fields=pending_fields,
    )
    if not normalized_query:
        return [], []
    capabilities = _chat_visible_capabilities()
    if not capabilities:
        return [], []
    normalized_family = str(preferred_family or "").strip().lower()
    if normalized_family:
        scoped_capabilities = [
            (capability_id, spec)
            for capability_id, spec in capabilities
            if str(spec.group or "").strip().lower() == normalized_family
            or str(spec.subgroup or "").strip().lower() == normalized_family
            or capability_id.split(".", 1)[0].strip().lower() == normalized_family
        ]
    else:
        scoped_capabilities = list(capabilities)
    search_capabilities_for_evidence = scoped_capabilities or capabilities
    entries = _chat_capability_search_entries(search_capabilities_for_evidence)
    lexical_matches = capability_search.search_capabilities(
        query=normalized_query,
        capability_entries=entries,
        limit=8,
        rerank_feedback_rows=[],
    )
    matches = _hybrid_chat_capability_matches(
        query=normalized_query,
        capabilities=search_capabilities_for_evidence,
        lexical_matches=lexical_matches,
        entries=entries,
    )
    if not matches and search_capabilities_for_evidence is not capabilities:
        entries = _chat_capability_search_entries(capabilities)
        lexical_matches = capability_search.search_capabilities(
            query=normalized_query,
            capability_entries=entries,
            limit=8,
            rerank_feedback_rows=[],
        )
        matches = _hybrid_chat_capability_matches(
            query=normalized_query,
            capabilities=capabilities,
            lexical_matches=lexical_matches,
            entries=entries,
        )
        search_capabilities_for_evidence = capabilities
    if not matches:
        return [], []
    capability_map = {
        capability_id: spec for capability_id, spec in search_capabilities_for_evidence
    }
    top_capabilities: list[chat_contracts.ChatBoundaryCapabilityEvidence] = []
    family_scores: dict[str, float] = {}
    family_capability_ids: dict[str, list[str]] = {}
    for match in matches[:5]:
        capability_id = str(match.get("id") or "").strip()
        spec = capability_map.get(capability_id)
        if spec is None:
            continue
        try:
            score = float(match.get("score") or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        top_capabilities.append(
            chat_contracts.ChatBoundaryCapabilityEvidence(
                capability_id=capability_id,
                group=str(spec.group or "").strip() or None,
                subgroup=str(spec.subgroup or "").strip() or None,
                score=round(score, 3),
                source=str(match.get("source") or "").strip() or None,
                reason=str(match.get("reason") or "").strip() or None,
            )
        )
        family = str(spec.group or spec.subgroup or "").strip() or capability_id.split(".", 1)[0]
        family_scores[family] = family_scores.get(family, 0.0) + score
        family_capability_ids.setdefault(family, [])
        if capability_id not in family_capability_ids[family]:
            family_capability_ids[family].append(capability_id)
    top_families = [
        chat_contracts.ChatBoundaryFamilyEvidence(
            family=family,
            score=round(score, 3),
            capability_ids=list(family_capability_ids.get(family, [])[:3]),
        )
        for family, score in sorted(family_scores.items(), key=lambda item: (-item[1], item[0]))[:3]
    ]
    return top_capabilities, top_families


def _looks_like_pending_clarification_answer(content: str) -> bool:
    lowered = str(content or "").strip().lower()
    if not lowered:
        return False
    if _looks_like_execution_confirmation(content):
        return True
    if lowered.endswith("?"):
        return False
    if _looks_like_conversational_turn(content):
        return False
    normalized = re.sub(r"[^a-z0-9]+", " ", lowered).strip()
    if not normalized:
        return False
    token_count = len(normalized.split())
    return token_count <= 18


def _chat_boundary_execution_signal_strength(
    *,
    top_capabilities: Sequence[chat_contracts.ChatBoundaryCapabilityEvidence],
    top_families: Sequence[chat_contracts.ChatBoundaryFamilyEvidence],
    intent_profile: workflow_contracts.GoalIntentProfile | None,
) -> tuple[float, float, float, str]:
    top_capability_score = (
        _chat_boundary_score(top_capabilities[0].score) if top_capabilities else 0.0
    )
    top_family_score = _chat_boundary_score(top_families[0].score) if top_families else 0.0
    total_family_score = sum(_chat_boundary_score(family.score) for family in top_families)
    family_concentration = (
        round(top_family_score / total_family_score, 3)
        if total_family_score > 0.0
        else 0.0
    )
    intent = str(intent_profile.intent or "").strip().lower() if intent_profile is not None else ""
    has_execution_intent = intent not in {"", "other", "inform", "clarify"}
    strength = "none"
    if top_capability_score >= 0.8 and top_family_score >= 1.1 and family_concentration >= 0.6:
        strength = "strong"
    elif (
        top_capability_score >= 0.65
        and top_family_score >= 0.8
        and family_concentration >= 0.5
        and has_execution_intent
    ):
        strength = "strong"
    elif top_capability_score >= 0.55 and top_family_score >= 0.65:
        strength = "moderate"
    elif top_capability_score > 0.0 or top_family_score > 0.0:
        strength = "weak"
    return (
        round(top_capability_score, 3),
        round(top_family_score, 3),
        family_concentration,
        strength,
    )


def _chat_boundary_intent_evidence(
    *,
    content: str,
    candidate_goal: str,
    pending_clarification: bool,
) -> workflow_contracts.GoalIntentProfile | None:
    if not pending_clarification and _looks_like_conversational_turn(content):
        return None
    try:
        normalized = _normalize_goal_intent(
            candidate_goal,
            include_decomposition=False,
            assessment_mode_override="heuristic",
        )
    except Exception:  # noqa: BLE001
        logger.exception("chat_boundary_intent_evidence_failed")
        return None
    return _chat_route_goal_intent_profile(normalized.profile, goal=candidate_goal)


def _build_chat_boundary_evidence(
    *,
    content: str,
    candidate_goal: str,
    session_metadata: Mapping[str, Any] | None,
    merged_context: Mapping[str, Any] | None,
) -> chat_contracts.ChatBoundaryEvidence:
    pending_state = _pending_clarification_state_from_metadata(session_metadata)
    pending = (
        pending_state.model_dump(mode="json", exclude_none=True)
        if pending_state is not None
        else {}
    )
    preferred_capability_ids: list[str] = []
    if pending_state is not None:
        for raw_capability_id in (
            [pending_state.active_capability_id] + list(pending_state.candidate_capabilities or [])
        ):
            capability_id = capability_registry.canonicalize_capability_id(raw_capability_id)
            if capability_id and capability_id not in preferred_capability_ids:
                preferred_capability_ids.append(capability_id)
    pending_fields = list(
        pending_state.pending_fields or pending_state.required_fields if pending_state is not None else []
    )
    workflow_invocation = chat_service.workflow_invocation_from_context(merged_context)
    top_capabilities, top_families = _chat_boundary_capability_evidence(
        query=_chat_boundary_query_text(content, candidate_goal),
        preferred_family=(pending_state.active_family if pending_state is not None else None),
        preferred_capability_ids=preferred_capability_ids,
        pending_fields=pending_fields,
    )
    intent_profile = _chat_boundary_intent_evidence(
        content=content,
        candidate_goal=candidate_goal,
        pending_clarification=bool(pending),
    )
    likely_clarification_answer = bool(pending) and _looks_like_pending_clarification_answer(content)
    (
        top_capability_score,
        top_family_score,
        family_concentration,
        execution_signal_strength,
    ) = _chat_boundary_execution_signal_strength(
        top_capabilities=top_capabilities,
        top_families=top_families,
        intent_profile=intent_profile,
    )
    if likely_clarification_answer:
        conversation_mode_hint = "clarification_answer"
    elif _looks_like_conversational_turn(content):
        conversation_mode_hint = "conversational"
    else:
        conversation_mode_hint = "execution_oriented"
    missing_inputs = [
        str(field).strip()
        for field in (
            list(intent_profile.missing_slots)
            if intent_profile is not None and intent_profile.missing_slots
            else pending_fields
        )
        if isinstance(field, str) and str(field).strip()
    ]
    return chat_contracts.ChatBoundaryEvidence(
        goal=str(candidate_goal or "").strip(),
        conversation_mode_hint=conversation_mode_hint,
        pending_clarification=bool(pending),
        workflow_target_available=workflow_invocation is not None,
        likely_clarification_answer=likely_clarification_answer,
        intent=str(intent_profile.intent or "").strip() if intent_profile is not None else "",
        risk_level=str(intent_profile.risk_level or "").strip() if intent_profile is not None else "",
        needs_clarification=bool(intent_profile.needs_clarification) if intent_profile is not None else False,
        missing_inputs=missing_inputs,
        active_family=str(pending_state.active_family or "").strip()
        if pending_state is not None
        else "",
        active_capability_id=str(pending_state.active_capability_id or "").strip()
        if pending_state is not None
        else "",
        clarification_resolved_slot_count=(
            len(
                dict(
                    pending_state.known_slot_values
                    or pending_state.resolved_slots
                    or {}
                )
            )
            if pending_state is not None
            else 0
        ),
        clarification_pending_field_count=(
            len(list(pending_fields))
            if pending_state is not None
            else 0
        ),
        clarification_answer_count=(
            len(list(pending_state.answer_history or []))
            if pending_state is not None
            else 0
        ),
        top_capability_score=top_capability_score,
        top_family_score=top_family_score,
        family_concentration=family_concentration,
        execution_signal_strength=execution_signal_strength,
        top_capabilities=top_capabilities,
        top_families=top_families,
    )


def _build_chat_boundary_decision_prompt(
    *,
    content: str,
    candidate_goal: str,
    session_metadata: Mapping[str, Any] | None,
    merged_context: Mapping[str, Any] | None,
    messages: Sequence[chat_contracts.ChatMessage] | None,
    boundary_evidence: chat_contracts.ChatBoundaryEvidence | None = None,
) -> str:
    recent_messages: list[dict[str, str]] = []
    for message in list(messages or [])[-8:]:
        recent_messages.append(
            {
                "role": str(message.role),
                "content": str(message.content or "")[:800],
            }
        )
    pending = (
        dict(session_metadata.get("pending_clarification"))
        if isinstance(session_metadata, Mapping)
        and isinstance(session_metadata.get("pending_clarification"), Mapping)
        else {}
    )
    workflow_invocation = chat_service.workflow_invocation_from_context(merged_context)
    payload = {
        "current_user_message": content,
        "candidate_goal": candidate_goal,
        "pending_clarification": bool(pending),
        "draft_goal": (
            str(session_metadata.get("draft_goal") or "").strip()
            if isinstance(session_metadata, Mapping)
            else ""
        ),
        "pending_questions": [
            str(question).strip()
            for question in pending.get("questions", [])
            if isinstance(question, str) and question.strip()
        ],
        "context_json": dict(merged_context or {}),
        "boundary_evidence": (
            boundary_evidence.model_dump(mode="json", exclude_none=True)
            if boundary_evidence is not None
            else {}
        ),
        "recent_messages": recent_messages,
        "workflow_context": {
            "target_available": workflow_invocation is not None,
            "trigger_id": workflow_invocation.trigger_id if workflow_invocation is not None else None,
            "version_id": workflow_invocation.version_id if workflow_invocation is not None else None,
            "definition_id": workflow_invocation.definition_id
            if workflow_invocation is not None
            else None,
        },
        "response_schema": {
            "decision": (
                "chat_reply | execution_request | continue_pending | "
                "exit_pending_to_chat | meta_clarification"
            ),
            "assistant_response": "string",
            "confidence": "0..1",
            "reason_code": "string",
        },
    }
    return json.dumps(payload, ensure_ascii=True)


def _generate_chat_response(
    *,
    content: str,
    candidate_goal: str,
    merged_context: Mapping[str, Any] | None,
    messages: Sequence[chat_contracts.ChatMessage] | None,
    fallback_response: str,
) -> str:
    if _chat_response_provider is None:
        return fallback_response
    try:
        response = _chat_response_provider.generate_request(
            LLMRequest(
                prompt=_build_chat_response_prompt(
                    content=content,
                    candidate_goal=candidate_goal,
                    merged_context=merged_context,
                    messages=messages,
                ),
                system_prompt=(
                    "You are the conversational assistant for an agent platform. "
                    "Answer directly and stay in chat. "
                    "Do not claim to have executed tools, created jobs, or run workflows unless the system already did so. "
                    "Be concise, technically accurate, and grounded in the provided context."
                ),
                metadata={"component": "chat_response"},
            )
        )
    except Exception:  # noqa: BLE001
        logger.exception("chat_response_generation_failed")
        return fallback_response
    generated = str(response.content or "").strip()
    return generated or fallback_response


def _generate_chat_boundary_decision(
    *,
    content: str,
    candidate_goal: str,
    session_metadata: Mapping[str, Any] | None,
    merged_context: Mapping[str, Any] | None,
    messages: Sequence[chat_contracts.ChatMessage] | None,
    fallback_response: str | None = None,
) -> chat_contracts.ChatBoundaryDecision | None:
    if CHAT_RESPONSE_MODE != "answer_or_handoff" or _chat_response_provider is None:
        return None
    boundary_evidence = _build_chat_boundary_evidence(
        content=content,
        candidate_goal=candidate_goal,
        session_metadata=session_metadata,
        merged_context=merged_context,
    )
    try:
        parsed = _chat_response_provider.generate_request_json_object(
            LLMRequest(
                prompt=_build_chat_boundary_decision_prompt(
                    content=content,
                    candidate_goal=candidate_goal,
                    session_metadata=session_metadata,
                    merged_context=merged_context,
                    messages=messages,
                    boundary_evidence=boundary_evidence,
                ),
                system_prompt=(
                    "You are the front-door boundary decision model for an agent platform. "
                    "Choose exactly one bounded decision and return JSON only. "
                    "Use boundary_evidence as grounding. "
                    "Strong executable capability-family evidence or an execution-oriented intent should push you toward execution_request unless the user is clearly asking for discussion only. "
                    "A conversational hint alone is not enough to override strong executable evidence. "
                    "If boundary_evidence.execution_signal_strength is 'strong' and conversation_mode_hint is not 'conversational', do not choose chat_reply unless the user explicitly asks for discussion, explanation, brainstorming, tutoring, or interview practice only. "
                    "When pending_clarification is false: "
                    "use decision='chat_reply' for normal conversation, explanation, discussion, advice, tutoring, coaching, quizzes, interview practice, roleplay, brainstorming, or any other back-and-forth chat experience. "
                    "Use decision='execution_request' only when the user wants tools, system actions, file changes, workflow execution, job submission, artifact creation, repository or environment inspection, or automation. "
                    "When pending_clarification is true: "
                    "If boundary_evidence.likely_clarification_answer is true, prefer decision='continue_pending'. "
                    "use decision='continue_pending' if the user is answering the existing workflow clarification or wants to continue that request; "
                    "use decision='exit_pending_to_chat' if the user wants to stop the workflow path and just get a normal chat answer; "
                    "use decision='meta_clarification' if it is ambiguous whether they want to continue the pending workflow or return to normal chat. "
                    "For chat_reply, exit_pending_to_chat, and meta_clarification, include assistant_response. "
                    "Do not choose execution_request just because the user wants a structured conversation or repeated turns."
                ),
                metadata={"component": "chat_boundary_decision"},
            )
        )
    except Exception:  # noqa: BLE001
        logger.exception("chat_boundary_decision_failed")
        return None
    try:
        decision = chat_contracts.ChatBoundaryDecision.model_validate(
            {
                "decision": parsed.get("decision") or parsed.get("type"),
                "confidence": parsed.get("confidence"),
                "assistant_response": (
                    str(parsed.get("assistant_response") or "").strip()
                    or str(fallback_response or "").strip()
                ),
                "reason_code": parsed.get("reason_code"),
                "evidence": boundary_evidence.model_dump(mode="json", exclude_none=True),
            }
        )
    except Exception:  # noqa: BLE001
        return None
    return decision


def _postprocess_chat_boundary_decision(
    boundary: chat_contracts.ChatBoundaryDecision,
    *,
    content: str,
) -> chat_contracts.ChatBoundaryDecision:
    evidence = boundary.evidence or chat_contracts.ChatBoundaryEvidence()
    if (
        not evidence.pending_clarification
        and boundary.decision == chat_contracts.ChatBoundaryDecisionType.chat_reply
        and evidence.execution_signal_strength == "strong"
        and evidence.conversation_mode_hint != "conversational"
    ):
        return boundary.model_copy(
            update={
                "decision": chat_contracts.ChatBoundaryDecisionType.execution_request,
                "assistant_response": "",
                "reason_code": "execution_signal_override",
            }
        )
    if (
        not evidence.pending_clarification
        and boundary.decision == chat_contracts.ChatBoundaryDecisionType.meta_clarification
    ):
        if (
            evidence.conversation_mode_hint != "conversational"
            and (
                evidence.needs_clarification
                or evidence.execution_signal_strength in {"moderate", "strong"}
            )
        ):
            return boundary.model_copy(
                update={
                    "decision": chat_contracts.ChatBoundaryDecisionType.execution_request,
                    "assistant_response": "",
                    "reason_code": "non_pending_meta_clarification_execution_override",
                }
            )
        return boundary.model_copy(
            update={
                "decision": chat_contracts.ChatBoundaryDecisionType.chat_reply,
                "reason_code": "non_pending_meta_clarification_chat_override",
            }
        )
    if (
        evidence.pending_clarification
        and evidence.likely_clarification_answer
        and boundary.decision
        in {
            chat_contracts.ChatBoundaryDecisionType.chat_reply,
            chat_contracts.ChatBoundaryDecisionType.meta_clarification,
        }
    ):
        return boundary.model_copy(
            update={
                "decision": chat_contracts.ChatBoundaryDecisionType.continue_pending,
                "assistant_response": "",
                "reason_code": "clarification_answer_override",
            }
        )
    if (
        evidence.pending_clarification
        and boundary.decision == chat_contracts.ChatBoundaryDecisionType.chat_reply
    ):
        if _looks_like_chat_only_correction(content):
            return boundary.model_copy(
                update={
                    "decision": chat_contracts.ChatBoundaryDecisionType.exit_pending_to_chat,
                    "reason_code": "explicit_chat_only_correction",
                }
            )
        if evidence.conversation_mode_hint != "conversational":
            return boundary.model_copy(
                update={
                    "decision": chat_contracts.ChatBoundaryDecisionType.continue_pending,
                    "assistant_response": "",
                    "reason_code": "pending_clarification_state_preservation",
                }
            )
        return boundary.model_copy(
            update={
                "decision": chat_contracts.ChatBoundaryDecisionType.meta_clarification,
                "assistant_response": (
                    boundary.assistant_response
                    or (
                        "Do you want to continue the current workflow request, "
                        "or should I answer here in chat instead?"
                    )
                ),
                "reason_code": "pending_clarification_ambiguous",
            }
        )
    return boundary


def _record_chat_boundary_decision_metrics(
    boundary: chat_contracts.ChatBoundaryDecision,
) -> None:
    evidence = boundary.evidence or chat_contracts.ChatBoundaryEvidence()
    chat_boundary_decisions_total.labels(
        decision=boundary.decision.value,
        conversation_mode_hint=_metrics_label(
            evidence.conversation_mode_hint,
            default="unknown",
        ),
        pending_clarification="true" if evidence.pending_clarification else "false",
        workflow_target_available="true" if evidence.workflow_target_available else "false",
    ).inc()
    chat_boundary_reason_total.labels(
        decision=boundary.decision.value,
        reason_code=_metrics_label(boundary.reason_code, default="none"),
    ).inc()


def _finalize_chat_turn_plan(
    turn_plan: Mapping[str, Any],
    *,
    content: str,
    candidate_goal: str,
    merged_context: Mapping[str, Any] | None,
    messages: Sequence[chat_contracts.ChatMessage] | None,
) -> dict[str, Any]:
    finalized = dict(turn_plan)
    route_type = str(finalized.get("type") or "").strip().lower()
    if route_type != "respond":
        return finalized
    capability_catalog_response = _capability_discovery_chat_response(content)
    if capability_catalog_response:
        finalized["assistant_content"] = capability_catalog_response
        finalized["response_generated"] = True
        return finalized
    if bool(finalized.get("response_generated")):
        return finalized
    fallback_response = str(finalized.get("assistant_content") or "").strip()
    finalized["assistant_content"] = _generate_chat_response(
        content=content,
        candidate_goal=candidate_goal,
        merged_context=merged_context,
        messages=messages,
        fallback_response=fallback_response or _fallback_chat_response(content),
    )
    return finalized


def _normalize_chat_route(
    parsed: Mapping[str, Any],
    *,
    content: str,
    candidate_goal: str,
    workflow_invocation_available: bool,
) -> dict[str, Any]:
    heuristic = _normalize_goal_intent(candidate_goal).profile
    heuristic = _chat_route_goal_intent_profile(heuristic, goal=candidate_goal)
    route = str(parsed.get("route") or parsed.get("type") or "").strip().lower()
    if route not in {"respond", "tool_call", "ask_clarification", "submit_job", "run_workflow"}:
        route = "respond"
    capability_id = str(parsed.get("capability_id") or "").strip()
    if route == "tool_call" and capability_id not in CHAT_DIRECT_CAPABILITIES:
        route = "respond"
        capability_id = ""
    if route == "run_workflow" and not workflow_invocation_available:
        route = "respond"
    arguments = dict(parsed.get("arguments")) if isinstance(parsed.get("arguments"), Mapping) else {}

    intent = str(parsed.get("intent") or heuristic.intent or "").strip().lower()
    if intent not in {"generate", "transform", "validate", "render", "io"}:
        intent = str(heuristic.intent or "")
    risk_level = str(parsed.get("risk_level") or heuristic.risk_level or "").strip().lower()
    if risk_level not in {"read_only", "bounded_write", "high_risk_write"}:
        risk_level = str(heuristic.risk_level or "read_only")
    confidence_raw = parsed.get("confidence")
    confidence = float(heuristic.confidence or 0.0)
    if isinstance(confidence_raw, (int, float)):
        confidence = max(0.0, min(1.0, float(confidence_raw)))
    threshold = (
        float(heuristic.threshold)
        if (
            intent == str(heuristic.intent or "").strip().lower()
            and risk_level == str(heuristic.risk_level or "").strip().lower()
            and heuristic.threshold is not None
        )
        else _resolve_intent_confidence_threshold(intent, risk_level)
    )

    slot_values = dict(heuristic.slot_values or {})
    for key in ("output_format", "target_system", "safety_constraints"):
        value = parsed.get(key)
        if isinstance(value, str) and value.strip():
            slot_values[key] = value.strip()
    slot_values["intent_action"] = intent
    slot_values["risk_level"] = risk_level

    if (
        intent == str(heuristic.intent or "").strip().lower()
        and risk_level == str(heuristic.risk_level or "").strip().lower()
    ):
        blocking_slots = list(heuristic.blocking_slots)
        missing_slots = list(heuristic.missing_slots)
    else:
        blocking_slots = _blocking_clarification_slots(intent, risk_level)
        missing_slots = [
            slot_name
            for slot_name in blocking_slots
            if not str(slot_values.get(slot_name) or "").strip()
        ]
    parsed_missing_slots = parsed.get("missing_slots")
    if isinstance(parsed_missing_slots, list):
        for slot_name in parsed_missing_slots:
            normalized = str(slot_name).strip()
            if normalized and normalized in blocking_slots and normalized not in missing_slots:
                missing_slots.append(normalized)
    low_confidence = confidence < threshold
    if route != "respond" and low_confidence and "intent_action" in blocking_slots and "intent_action" not in missing_slots:
        missing_slots.append("intent_action")
    if route == "tool_call" and (missing_slots or risk_level != "read_only"):
        route = "respond"
    if route == "submit_job" and missing_slots:
        route = "ask_clarification"
    clarification_questions = [
        str(question).strip()
        for question in parsed.get("clarification_questions", [])
        if isinstance(question, str) and question.strip()
    ]
    assistant_response = str(parsed.get("assistant_response") or "").strip()
    if (
        route == "respond"
        and (
            clarification_questions
            or (
                missing_slots
                and not _looks_like_conversational_turn(content)
            )
        )
    ):
        route = "ask_clarification"
    if route == "ask_clarification" and not clarification_questions:
        clarification_questions = [
            str(question).strip()
            for question in heuristic.questions
            if isinstance(question, str) and question.strip()
        ]
        if not clarification_questions:
            clarification_questions = [
                _slot_question(slot_name, candidate_goal) for slot_name in missing_slots
            ]
    if route == "ask_clarification" and not assistant_response:
        assistant_response = "\n".join(clarification_questions)
    if route == "respond" and not assistant_response:
        assistant_response = _fallback_chat_response(content)
    if route in {"respond", "tool_call", "run_workflow"}:
        blocking_slots = []
        missing_slots = []
        clarification_questions = []
    assessment = {
        "intent": intent,
        "source": "llm_chat_router",
        "confidence": round(confidence, 3),
        "risk_level": risk_level,
        "threshold": threshold,
        "low_confidence": low_confidence,
        "needs_clarification": bool(missing_slots),
        "requires_blocking_clarification": bool(missing_slots),
        "questions": clarification_questions,
        "blocking_slots": blocking_slots,
        "missing_slots": missing_slots,
        "slot_values": slot_values,
        "clarification_mode": "llm_targeted_slot_filling",
    }
    return {
        "type": route,
        "assistant_content": assistant_response,
        "capability_id": capability_id,
        "arguments": arguments,
        "clarification_questions": clarification_questions,
        "goal_intent_profile": workflow_contracts.dump_goal_intent_profile(assessment) or {},
        "resolved_goal": candidate_goal,
    }


def _chat_direct_capability_description(capability_id: str) -> str:
    descriptions = {
        "github.repo.list": "List repositories for a user/org or search repositories.",
        "github.user.me": "Return the authenticated GitHub user profile.",
        "github.issue.search": "Search GitHub issues or pull requests.",
        "github.user.search": "Search GitHub users.",
        "github.branch.list": "List branches for a GitHub repository.",
        "filesystem.artifacts.list": "List files under the shared artifacts directory.",
        "filesystem.artifacts.read_text": "Read a text file from the shared artifacts directory.",
        "filesystem.artifacts.search_text": "Search text within files in the shared artifacts directory.",
        "filesystem.workspace.list": "List files under the shared workspace directory.",
        "filesystem.workspace.read_text": "Read a text file from the shared workspace directory.",
        "memory.read": "Read structured memory entries by name/key/job/user/project.",
        "memory.semantic.search": "Search semantic memory for related facts.",
    }
    return descriptions.get(capability_id, capability_id)


def _dispatch_runtime() -> dispatch_service.ApiDispatchRuntime:
    return dispatch_service.ApiDispatchRuntime(
        redis_client=redis_client,
        session_factory=SessionLocal,
        logger=logger,
        config=dispatch_service.ApiDispatchConfig(
            event_outbox_enabled=EVENT_OUTBOX_ENABLED,
            event_outbox_batch_size=EVENT_OUTBOX_BATCH_SIZE,
            event_outbox_poll_s=EVENT_OUTBOX_POLL_S,
            event_outbox_redis_retries=EVENT_OUTBOX_REDIS_RETRIES,
            event_outbox_redis_retry_sleep_s=EVENT_OUTBOX_REDIS_RETRY_SLEEP_S,
            policy_gate_enabled=POLICY_GATE_ENABLED,
            tool_input_validation_enabled=TOOL_INPUT_VALIDATION_ENABLED,
            tool_input_schemas=TOOL_INPUT_SCHEMAS,
        ),
    )


def _candidate_capability_ids_for_envelope(
    envelope: workflow_contracts.NormalizedIntentEnvelope,
    *,
    active_target: intent_contract.ActiveExecutionTarget | None = None,
    preferred_family: str | None = None,
) -> list[str]:
    if active_target is not None and active_target.capability_ids:
        capability_ids: list[str] = []
        seen: set[str] = set()
        preferred_capability = str(active_target.capability_id or "").strip()
        ordered = list(active_target.capability_ids)
        if preferred_capability and preferred_capability in ordered:
            ordered = [preferred_capability] + [
                capability_id
                for capability_id in ordered
                if capability_id != preferred_capability
            ]
        for capability_id in ordered:
            normalized = capability_registry.canonicalize_capability_id(capability_id)
            if normalized and normalized not in seen:
                seen.add(normalized)
                capability_ids.append(normalized)
        if capability_ids:
            return capability_ids

    capability_ids: list[str] = []
    seen: set[str] = set()
    for capability_list in envelope.candidate_capabilities.values():
        for capability_id in capability_list:
            normalized = capability_registry.canonicalize_capability_id(capability_id)
            if normalized and normalized not in seen:
                seen.add(normalized)
                capability_ids.append(normalized)
    for segment in envelope.graph.segments:
        for capability_id in segment.suggested_capabilities:
            normalized = capability_registry.canonicalize_capability_id(capability_id)
            if normalized and normalized not in seen:
                seen.add(normalized)
                capability_ids.append(normalized)
    normalized_family = str(preferred_family or "").strip().lower()
    if normalized_family:
        filtered = [
            capability_id
            for capability_id in capability_ids
            if str(_capability_family_for_id(capability_id) or "").strip().lower() == normalized_family
        ]
        if filtered:
            return filtered
    return capability_ids


def _pending_clarification_state_from_metadata(
    metadata: Mapping[str, Any] | None,
) -> chat_contracts.ClarificationState | None:
    if not isinstance(metadata, Mapping):
        return None
    raw = metadata.get("pending_clarification")
    if not isinstance(raw, Mapping):
        return None
    try:
        return chat_contracts.ClarificationState.model_validate(dict(raw))
    except Exception:  # noqa: BLE001
        return None


def _capability_family_for_id(capability_id: str) -> str | None:
    normalized = capability_registry.canonicalize_capability_id(capability_id)
    if not normalized:
        return None
    try:
        registry = capability_registry.load_capability_registry()
    except Exception:  # noqa: BLE001
        registry = None
    spec = registry.get(normalized) if registry is not None else None
    family = str(spec.group or spec.subgroup or "").strip() if spec is not None else ""
    if family:
        return family
    prefix = normalized.split(".", 1)[0].strip()
    return prefix or None


def _active_execution_target_for_chat(
    *,
    normalized: workflow_contracts.NormalizedIntentEnvelope,
    session_metadata: Mapping[str, Any] | None,
    merged_context: Mapping[str, Any] | None,
) -> intent_contract.ActiveExecutionTarget | None:
    pending_state = _pending_clarification_state_from_metadata(session_metadata)
    if pending_state is None:
        return None
    known_slot_values = dict(pending_state.known_slot_values or {})
    if isinstance(merged_context, Mapping):
        for key, raw_value in merged_context.items():
            if raw_value is None:
                continue
            if isinstance(raw_value, str) and not raw_value.strip():
                continue
            known_slot_values.setdefault(str(key), raw_value)
    return intent_contract.select_active_execution_target(
        graph=workflow_contracts.dump_intent_graph(normalized.graph) or {},
        candidate_capabilities=normalized.candidate_capabilities,
        known_slot_values=known_slot_values,
        pending_fields=tuple(pending_state.pending_fields or pending_state.required_fields),
        preferred_segment_id=pending_state.active_segment_id,
        preferred_capability_id=pending_state.active_capability_id,
    )


def _chat_submit_capability_contracts(
    capability_ids: Sequence[str],
) -> list[dict[str, Any]]:
    try:
        registry = capability_registry.load_capability_registry()
    except Exception:  # noqa: BLE001
        return []
    contracts: list[dict[str, Any]] = []
    for capability_id in capability_ids:
        spec = registry.get(capability_id)
        if spec is None:
            continue
        planner_hints = dict(spec.planner_hints or {})
        contracts.append(
            {
                "capability_id": spec.capability_id,
                "description": spec.description,
                "required_inputs": _capability_required_inputs_for_intent_normalization(
                    spec.capability_id
                ),
                "chat_collectible_fields": planner_hints.get("chat_collectible_fields"),
                "chat_required_fields": planner_hints.get("chat_required_fields"),
                "field_descriptions": planner_hints.get("chat_field_descriptions"),
                "field_examples": planner_hints.get("chat_field_examples"),
            }
        )
    return contracts


def _chat_submit_clarification_questions(
    normalized: workflow_contracts.NormalizedIntentEnvelope,
    *,
    goal: str,
    unresolved_fields: Sequence[str] = (),
    scoped_fields: Sequence[str] = (),
) -> list[str]:
    questions: list[str] = []

    def _append_question(question: Any) -> None:
        if isinstance(question, str):
            normalized_question = question.strip()
            if normalized_question and normalized_question not in questions:
                questions.append(normalized_question)

    fallback_fields: list[str] = []
    scoped_field_set = {
        normalized_field
        for raw_field in scoped_fields
        if (normalized_field := intent_contract.normalize_required_input_key(raw_field))
    }
    for raw_field in unresolved_fields:
        normalized_field = intent_contract.normalize_required_input_key(raw_field)
        if not normalized_field:
            continue
        if scoped_field_set and normalized_field not in scoped_field_set:
            continue
        if normalized_field not in fallback_fields:
            fallback_fields.append(normalized_field)
    for raw_field in normalized.clarification.missing_inputs:
        normalized_field = intent_contract.normalize_required_input_key(raw_field)
        if not normalized_field:
            continue
        if scoped_field_set and normalized_field not in scoped_field_set:
            continue
        if normalized_field not in fallback_fields:
            fallback_fields.append(normalized_field)
    for raw_field in normalized.profile.missing_slots:
        normalized_field = intent_contract.normalize_required_input_key(raw_field)
        if not normalized_field:
            continue
        if scoped_field_set and normalized_field not in scoped_field_set:
            continue
        if normalized_field not in fallback_fields:
            fallback_fields.append(normalized_field)

    if scoped_field_set and fallback_fields:
        for field in fallback_fields:
            _append_question(
                chat_clarification_normalizer.clarification_question_for_field(field, goal=goal)
            )
        return questions

    for raw_question in normalized.clarification.questions:
        _append_question(raw_question)
    for raw_question in normalized.profile.questions:
        _append_question(raw_question)

    if not questions:
        for field in fallback_fields:
            _append_question(
                chat_clarification_normalizer.clarification_question_for_field(field, goal=goal)
            )
    return questions


def _chat_submit_conversation_history(
    messages: Sequence[chat_contracts.ChatMessage] | None,
) -> list[dict[str, str]]:
    if not isinstance(messages, Sequence):
        return []
    history: list[dict[str, str]] = []
    for message in list(messages)[-12:]:
        role_value = getattr(message, "role", "")
        role = str(getattr(role_value, "value", role_value) or "").strip().lower()
        if role not in {"user", "assistant"}:
            continue
        content = str(getattr(message, "content", "") or "").strip()
        if not content:
            continue
        normalized_content = " ".join(content.split())
        if not normalized_content:
            continue
        history.append(
            {
                "role": role,
                "content": normalized_content[:500],
            }
        )
    return history


def _normalize_chat_submit_context(
    *,
    db: Session,
    goal: str,
    content: str,
    session_metadata: Mapping[str, Any] | None,
    merged_context: Mapping[str, Any] | None,
    context_envelope: workflow_contracts.ContextEnvelope | Mapping[str, Any] | None = None,
    user_id: str | None,
    messages: Sequence[chat_contracts.ChatMessage] | None = None,
) -> chat_service.ChatSubmitNormalizationResult | None:
    submit_context = (
        context_service.chat_submit_context_view(context_envelope)
        if context_envelope is not None
        else dict(merged_context) if isinstance(merged_context, Mapping) else {}
    )
    normalized = _normalize_goal_intent(
        goal,
        db=db,
        user_id=user_id,
        context_envelope=context_envelope,
    )
    pending_state = _pending_clarification_state_from_metadata(session_metadata)
    preferred_field = (
        intent_contract.normalize_required_input_key(pending_state.current_question_field)
        if pending_state is not None and pending_state.current_question_field
        else None
    )
    conversation_history = _chat_submit_conversation_history(messages)
    updates: dict[str, str] = {}
    unresolved: list[str] = []
    accepted_confidence: dict[str, float] = {}
    normalized_capability_ids: list[str] = []
    running_context = dict(submit_context)
    active_target = _active_execution_target_for_chat(
        normalized=normalized,
        session_metadata=session_metadata,
        merged_context=running_context,
    )
    active_candidate_ids = _candidate_capability_ids_for_envelope(
        normalized,
        active_target=active_target,
        preferred_family=pending_state.active_family if pending_state is not None else None,
    )
    all_candidate_ids = _candidate_capability_ids_for_envelope(
        normalized,
        preferred_family=pending_state.active_family if pending_state is not None else None,
    )
    candidate_capability_ids: list[str] = []
    for capability_id in list(active_candidate_ids) + list(all_candidate_ids):
        if capability_id and capability_id not in candidate_capability_ids:
            candidate_capability_ids.append(capability_id)
    scoped_fields = (
        list(active_target.unresolved_fields or active_target.required_fields)
        if active_target is not None
        else []
    )
    active_capability_set = set(active_candidate_ids)
    for raw_contract in _chat_submit_capability_contracts(candidate_capability_ids):
        contracts = chat_clarification_normalizer.build_capability_normalization_contracts(
            goal=goal,
            merged_context=running_context,
            capability_contracts=[raw_contract],
        )
        if not contracts:
            continue
        contract = contracts[0]
        contract_updates: dict[str, str] = {}
        contract_unresolved: list[str] = []
        contract_confidence: dict[str, float] = {}
        if CHAT_CLARIFICATION_NORMALIZER_ENABLED:
            contract_updates, contract_unresolved, contract_confidence = (
                chat_clarification_normalizer.normalize_contract_fields_with_llm(
                    goal=goal,
                    contract=contract,
                    provider=_chat_clarification_normalizer_provider,
                    confidence_threshold=CHAT_CLARIFICATION_NORMALIZER_CONFIDENCE_THRESHOLD,
                    conversation_history=conversation_history,
                    preferred_field=preferred_field,
                    latest_answer=content,
                )
            )
        else:
            contract_unresolved = [
                field for field in contract.required_fields if field in contract.missing_fields
            ]
        if contract_updates:
            updates.update(contract_updates)
            running_context.update(contract_updates)
            accepted_confidence.update(contract_confidence)
            if contract.capability_id not in normalized_capability_ids:
                normalized_capability_ids.append(contract.capability_id)
        if active_target is None or contract.capability_id in active_capability_set:
            for field in contract_unresolved:
                normalized_field = intent_contract.normalize_required_input_key(field)
                if normalized_field and normalized_field not in unresolved:
                    unresolved.append(normalized_field)

    context_updates: dict[str, Any] = {}
    if updates:
        context_updates.update(updates)
        context_updates["clarification_normalization"] = {
            "source": "chat_clarification_normalizer",
            "capability_id": normalized_capability_ids[0] if normalized_capability_ids else None,
            "capability_ids": sorted(set(normalized_capability_ids)),
            "fields": sorted(updates.keys()),
            "confidence": accepted_confidence,
        }
    updated_envelope = context_envelope
    if updates and context_envelope is not None:
        updated_envelope = context_service.update_chat_context_envelope(
            context_envelope,
            goal=goal,
            context_json=context_updates,
        )

    refreshed_normalized = (
        _normalize_goal_intent(
            goal,
            db=db,
            user_id=user_id,
            context_envelope=updated_envelope,
        )
        if updated_envelope is not None
        else normalized
    )
    refreshed_assessment = refreshed_normalized.profile.model_dump(mode="json", exclude_none=True)
    requires_blocking_clarification = bool(
        refreshed_normalized.profile.requires_blocking_clarification
        or refreshed_normalized.clarification.requires_blocking_clarification
    )
    clarification_questions = _chat_submit_clarification_questions(
        refreshed_normalized,
        goal=goal,
        unresolved_fields=unresolved,
        scoped_fields=scoped_fields,
    )
    if requires_blocking_clarification and not clarification_questions:
        clarification_questions = [
            str(question).strip()
            for question in (
                list(refreshed_normalized.clarification.questions)
                + list(refreshed_normalized.profile.questions)
            )
            if isinstance(question, str) and question.strip()
        ]
    if requires_blocking_clarification and not clarification_questions:
        clarification_questions = [
            "I still need the remaining required details before I can submit this request."
        ]
    if not context_updates and not clarification_questions:
        return None
    return chat_service.ChatSubmitNormalizationResult(
        goal=goal,
        context_json=context_updates,
        clarification_questions=clarification_questions,
        requires_blocking_clarification=requires_blocking_clarification,
        goal_intent_profile=refreshed_assessment,
    )


def _chat_runtime() -> chat_service.ChatServiceRuntime:
    return chat_service.ChatServiceRuntime(
        route_turn=_route_chat_turn,
        run_direct_capability=_run_chat_direct_capability,
        create_job=lambda job, db: _create_job_internal(
            job,
            db,
            metadata_overrides={
                "workflow_source": "chat",
                "render_path_mode": planner_contracts.RENDER_PATH_MODE_AUTO,
            },
        ),
        run_workflow=_run_chat_workflow,
        inspect_workflow=_inspect_chat_workflow,
        utcnow=_utcnow,
        make_id=lambda: str(uuid.uuid4()),
        normalize_submit_context=_normalize_chat_submit_context,
        is_chat_only_correction=_looks_like_chat_only_correction,
    )


def _chat_direct_capability_spec(
    *,
    capability_id: str,
):
    normalized_capability_id = str(capability_id or "").strip()
    if not CHAT_DIRECT_EXECUTION_ENABLED:
        raise RuntimeError("chat_direct_execution_disabled")
    if not normalized_capability_id:
        raise RuntimeError("chat_direct_missing_capability_id")
    if normalized_capability_id not in CHAT_DIRECT_CAPABILITIES:
        raise RuntimeError(f"chat_direct_capability_not_allowed:{normalized_capability_id}")
    if capability_registry.resolve_capability_mode() == "disabled":
        raise RuntimeError("chat_direct_capabilities_disabled")
    registry = capability_registry.load_capability_registry()
    spec = registry.require(normalized_capability_id)
    if not spec.enabled:
        raise RuntimeError(f"chat_direct_capability_disabled:{normalized_capability_id}")
    if spec.risk_tier != "read_only":
        raise RuntimeError(f"chat_direct_capability_not_read_only:{normalized_capability_id}")
    allow_decision = capability_registry.evaluate_capability_allowlist(
        normalized_capability_id,
        RUNTIME_CONFORMANCE_SERVICE,
    )
    if not allow_decision.allowed:
        raise RuntimeError(
            f"chat_direct_capability_blocked:{normalized_capability_id}:{allow_decision.reason}"
        )
    return spec


def _build_chat_direct_run_plan(
    capability_spec: capability_registry.CapabilitySpec,
    *,
    arguments: Mapping[str, Any] | None,
) -> tuple[models.PlanCreate, models.RunSpec]:
    capability_id = capability_spec.capability_id
    capability_binding = {
        "capability_id": capability_id,
    }
    if capability_spec.adapters:
        adapter = capability_spec.adapters[0]
        capability_binding.update(
            {
                "adapter_type": adapter.type,
                "server_id": adapter.server_id,
                "tool_name": adapter.tool_name,
            }
        )
    task = models.TaskCreate(
        name=f"ChatDirect:{capability_id}",
        description=capability_spec.description,
        instruction=(
            f"Execute capability {capability_id} once for the current chat turn "
            "and return the result."
        ),
        acceptance_criteria=[f"Capability {capability_id} executes successfully."],
        expected_output_schema_ref=capability_spec.output_schema_ref or "",
        intent=models.ToolIntent.io,
        deps=[],
        tool_requests=[capability_id],
        tool_inputs={capability_id: dict(arguments or {})},
        capability_bindings={capability_id: capability_binding},
        critic_required=False,
    )
    plan = models.PlanCreate(
        planner_version="chat_direct_v1",
        tasks_summary=f"Execute {capability_id} as a synchronous chat direct run.",
        dag_edges=[],
        tasks=[task],
    )
    run_spec = run_specs.plan_to_run_spec(
        plan,
        kind=models.RunKind.chat_direct,
        metadata={
            "source": "chat_direct",
            "execution_mode": "synchronous",
        },
    )
    if run_spec.steps:
        run_spec.steps[0].retry_policy.max_attempts = 1
    return plan, run_spec


def _handle_local_runtime_event(
    event_type: str,
    *,
    correlation_id: str,
    job_id: str,
    task_id: str,
    payload: Mapping[str, Any],
    occurred_at: datetime | None = None,
) -> None:
    envelope = models.EventEnvelope(
        type=event_type,
        version="1",
        occurred_at=occurred_at or _utcnow(),
        correlation_id=correlation_id,
        job_id=job_id,
        task_id=task_id,
        payload=dict(payload),
    )
    _handle_event("local_sync", {"data": envelope.model_dump_json()})


def _execute_chat_task_sync(task_payload: dict[str, Any]) -> models.TaskResult:
    from services.worker.app import main as worker_main

    return worker_main.execute_task(task_payload)


def _chat_direct_output_from_task_result(
    task_result: Mapping[str, Any],
    *,
    request_id: str,
) -> dict[str, Any]:
    outputs = task_result.get("outputs")
    if isinstance(outputs, Mapping):
        resolved = outputs.get(request_id)
        if isinstance(resolved, Mapping):
            return dict(resolved)
        if resolved is not None:
            return {"result": resolved}
        if len(outputs) == 1:
            only_value = next(iter(outputs.values()))
            if isinstance(only_value, Mapping):
                return dict(only_value)
            if only_value is not None:
                return {"result": only_value}
    tool_calls = task_result.get("tool_calls")
    if isinstance(tool_calls, list):
        for call in reversed(tool_calls):
            if not isinstance(call, Mapping):
                continue
            output = call.get("output_or_error")
            if isinstance(output, Mapping):
                return dict(output)
            if output is not None:
                return {"result": output}
    return {}


def _run_chat_direct_capability(
    *,
    db: Session,
    chat_session_id: str,
    goal: str,
    capability_id: str,
    arguments: Mapping[str, Any] | None = None,
    context_json: Mapping[str, Any] | None = None,
    priority: int = 0,
) -> chat_service.ChatDirectRunResult:
    capability_spec = _chat_direct_capability_spec(capability_id=capability_id)
    normalized_arguments = dict(arguments) if isinstance(arguments, Mapping) else {}
    normalized_context_json = dict(context_json) if isinstance(context_json, Mapping) else {}
    plan, run_spec = _build_chat_direct_run_plan(
        capability_spec,
        arguments=normalized_arguments,
    )
    preflight_errors = _merge_preflight_errors(
        _compile_plan_preflight(
            plan,
            normalized_context_json,
            goal_text=str(goal or capability_spec.description or capability_spec.capability_id),
            goal_intent_graph=None,
            render_path_mode=planner_contracts.RENDER_PATH_MODE_AUTO,
        ),
        _compile_plan_runtime_conformance_errors(plan),
    )
    if preflight_errors:
        details = "; ".join(
            f"{task_name}: {message}" for task_name, message in sorted(preflight_errors.items())
        )
        raise RuntimeError(f"chat_direct_preflight_failed:{details}")

    direct_envelope = _plan_derived_normalized_intent_envelope(
        goal=str(goal or capability_spec.description or capability_spec.capability_id),
        plan=plan,
        source="chat_direct_compiled_plan",
    )
    job = _create_job_internal(
        models.JobCreate(
            goal=str(goal or capability_spec.description or capability_spec.capability_id),
            context_json=normalized_context_json,
            priority=priority,
        ),
        db,
        emit_job_created_event=False,
        metadata_overrides={
            "goal_intent_profile": (
                workflow_contracts.dump_goal_intent_profile(direct_envelope.profile) or {}
            ),
            "normalized_intent_envelope": (
                workflow_contracts.dump_normalized_intent_envelope(direct_envelope) or {}
            ),
            "goal_intent_graph": workflow_contracts.dump_intent_graph(direct_envelope.graph),
            "workflow_source": "chat_direct",
            "render_path_mode": planner_contracts.RENDER_PATH_MODE_AUTO,
            "scheduler_mode": POSTGRES_RUN_SPEC_SCHEDULER_MODE,
            "run_kind": models.RunKind.chat_direct.value,
            "run_spec": run_spec.model_dump(mode="json"),
            "chat_session_id": chat_session_id,
            "chat_execution_mode": "sync_direct",
        },
    )
    plan_record = _create_plan_internal(
        plan,
        job_id=job.id,
        db=db,
        emit_plan_created_event=False,
    )
    task_records = db.query(TaskRecord).filter(TaskRecord.plan_id == plan_record.id).all()
    if len(task_records) != 1:
        raise RuntimeError("chat_direct_task_count_invalid")
    task_record = task_records[0]
    now = _utcnow()
    task_record.attempts = 1
    task_record.max_attempts = 1
    task_record.max_reworks = 0
    task_record.status = models.TaskStatus.ready.value
    task_record.updated_at = now
    db.commit()

    job_record = db.query(JobRecord).filter(JobRecord.id == job.id).first()
    if job_record is None:
        raise RuntimeError("chat_direct_job_missing")
    _set_job_status(job_record, models.JobStatus.planning)
    job_record.updated_at = _utcnow()
    db.commit()
    tasks = _resolve_task_deps(task_records)
    task_map = {task.id: task for task in tasks}
    id_to_name = {record.id: record.name for record in task_records}
    task_intent_profiles = _coerce_task_intent_profiles(
        job_record.metadata_json if isinstance(job_record.metadata_json, dict) else {}
    )
    correlation_id = str(uuid.uuid4())
    execution_job_context = _execution_job_context(
        job_record.goal if isinstance(job_record.goal, str) else "",
        job_record.context_json if isinstance(job_record.context_json, dict) else {},
        job_record.metadata_json if isinstance(job_record.metadata_json, dict) else {},
    )
    task_payload = _task_payload_from_record(
        task_record,
        correlation_id,
        context=_build_task_context(
            task_record.id,
            task_map,
            id_to_name,
            execution_job_context,
        ),
        goal_text=job_record.goal if isinstance(job_record.goal, str) else "",
        intent_profile=task_intent_profiles.get(task_record.id),
    )
    task_payload["run_id"] = job.id

    attempts = int(task_payload.get("attempts") or 1)
    max_attempts = int(task_payload.get("max_attempts") or 1)
    if task_payload.get("tool_inputs_validation"):
        failed_payload = dict(task_payload)
        failed_payload["error"] = "tool_inputs_invalid"
        _handle_local_runtime_event(
            "task.failed",
            correlation_id=correlation_id,
            job_id=job.id,
            task_id=task_record.id,
            payload=failed_payload,
        )
    else:
        _handle_local_runtime_event(
            "task.ready",
            correlation_id=correlation_id,
            job_id=job.id,
            task_id=task_record.id,
            payload=task_payload,
        )
        started_at = _utcnow()
        _handle_local_runtime_event(
            "task.started",
            correlation_id=correlation_id,
            job_id=job.id,
            task_id=task_record.id,
            payload={
                "task_id": task_record.id,
                "attempts": attempts,
                "max_attempts": max_attempts,
                "worker_consumer": CHAT_DIRECT_SYNC_WORKER_CONSUMER,
                "run_id": job.id,
            },
            occurred_at=started_at,
        )
        try:
            result = _execute_chat_task_sync(dict(task_payload))
            result_payload = result.model_dump(mode="json")
            event_type = (
                "task.failed" if result.status == models.TaskStatus.failed else "task.completed"
            )
        except Exception as exc:  # noqa: BLE001
            error = f"chat_direct_execution_error:{exc}"
            event_type = "task.failed"
            result_payload = {
                "task_id": task_record.id,
                "status": models.TaskStatus.failed.value,
                "outputs": {"tool_error": {"error": error}},
                "artifacts": [],
                "tool_calls": [],
                "started_at": started_at.isoformat(),
                "finished_at": _utcnow().isoformat(),
                "error": error,
            }
        result_payload["attempts"] = attempts
        result_payload["max_attempts"] = max_attempts
        result_payload["worker_consumer"] = CHAT_DIRECT_SYNC_WORKER_CONSUMER
        result_payload["run_id"] = job.id
        _handle_local_runtime_event(
            event_type,
            correlation_id=correlation_id,
            job_id=job.id,
            task_id=task_record.id,
            payload=result_payload,
        )

    db.expire_all()
    refreshed_job = db.query(JobRecord).filter(JobRecord.id == job.id).first()
    if refreshed_job is None:
        raise RuntimeError("chat_direct_job_missing")
    task_result = _load_task_result(task_record.id)
    error = _extract_error_from_task_result(task_result)
    output = (
        {}
        if error
        else _chat_direct_output_from_task_result(
            task_result,
            request_id=capability_spec.capability_id,
        )
    )
    assistant_response = (
        ""
        if error
        else chat_execution_service.format_chat_direct_result(
            capability_spec.capability_id,
            output,
        )
    )
    return chat_service.ChatDirectRunResult(
        job=_job_from_record(refreshed_job),
        capability_id=capability_spec.capability_id,
        tool_name=chat_execution_service.tool_name_for_capability(capability_spec),
        output=output,
        assistant_response=assistant_response,
        error=error,
    )


def _run_chat_workflow(
    *,
    db: Session,
    workflow_trigger_id: str | None = None,
    workflow_version_id: str | None = None,
    workflow_definition_id: str | None = None,
    inputs: Mapping[str, Any] | None = None,
    context_json: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    idempotency_key: str | None = None,
    priority: int = 0,
) -> models.WorkflowRunResult:
    definition, version, trigger = _resolve_chat_workflow_reference(
        db=db,
        workflow_trigger_id=workflow_trigger_id,
        workflow_version_id=workflow_version_id,
        workflow_definition_id=workflow_definition_id,
    )
    payload: dict[str, Any] = {
        "inputs": dict(inputs) if isinstance(inputs, Mapping) else {},
        "context_json": dict(context_json) if isinstance(context_json, Mapping) else {},
        "priority": priority,
        "metadata": dict(metadata) if isinstance(metadata, Mapping) else {},
    }
    normalized_idempotency_key = str(idempotency_key or "").strip()
    if normalized_idempotency_key:
        payload["idempotency_key"] = normalized_idempotency_key

    workflow_interface, _chat_context_json, runtime_inputs = _prepare_chat_workflow_runtime_inputs(
        version=version,
        trigger=trigger,
        context_json=payload["context_json"] if isinstance(payload["context_json"], Mapping) else {},
        inputs=payload["inputs"] if isinstance(payload["inputs"], Mapping) else {},
    )
    if workflow_interface.get("inputs"):
        payload["inputs"] = runtime_inputs

    if trigger is not None:
        return _run_workflow_version_internal(
            version=version,
            definition=definition,
            db=db,
            payload=payload,
            trigger=trigger,
            source="chat_workflow_trigger",
        )
    source = (
        "chat_workflow_version"
        if str(workflow_version_id or "").strip()
        else "chat_workflow_definition"
    )
    return _run_workflow_version_internal(
        version=version,
        definition=definition,
        db=db,
        payload=payload,
        trigger=None,
        source=source,
    )


def _inspect_chat_workflow(
    *,
    db: Session,
    workflow_trigger_id: str | None = None,
    workflow_version_id: str | None = None,
    workflow_definition_id: str | None = None,
    inputs: Mapping[str, Any] | None = None,
    context_json: Mapping[str, Any] | None = None,
) -> chat_service.ChatWorkflowInspection:
    definition, version, trigger = _resolve_chat_workflow_reference(
        db=db,
        workflow_trigger_id=workflow_trigger_id,
        workflow_version_id=workflow_version_id,
        workflow_definition_id=workflow_definition_id,
    )
    workflow_interface, merged_context_json, runtime_inputs = _prepare_chat_workflow_runtime_inputs(
        version=version,
        trigger=trigger,
        context_json=context_json,
        inputs=inputs,
    )
    _preview_context, workflow_context_errors = _build_workflow_interface_runtime_context(
        workflow_interface,
        base_context=merged_context_json,
        db=db,
        explicit_inputs=runtime_inputs,
        preview=False,
    )
    input_defs = {
        str(item.get("key") or "").strip(): item
        for item in workflow_interface.get("inputs", [])
        if isinstance(item, Mapping)
    }
    missing_inputs: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for error in workflow_context_errors:
        if not isinstance(error, Mapping):
            continue
        code = str(error.get("code") or "").strip()
        if code not in {"draft.workflow_input_value_missing", "draft.workflow_input_value_invalid"}:
            continue
        field = str(error.get("field") or "").strip()
        if not field.startswith("workflowInterface.inputs."):
            continue
        key = field.removeprefix("workflowInterface.inputs.").strip()
        if not key or key in seen_keys:
            continue
        seen_keys.add(key)
        definition_input = input_defs.get(key, {})
        missing_inputs.append(
            {
                "key": key,
                "label": str(definition_input.get("label") or key).strip() or key,
                "value_type": str(definition_input.get("value_type") or "string").strip().lower()
                or "string",
                "description": str(definition_input.get("description") or "").strip(),
                "message": str(error.get("message") or "").strip(),
            }
        )
    return chat_service.ChatWorkflowInspection(
        trigger_id=trigger.id if trigger is not None else None,
        version_id=version.id,
        definition_id=definition.id,
        missing_inputs=missing_inputs,
    )


def _resolve_chat_workflow_reference(
    *,
    db: Session,
    workflow_trigger_id: str | None = None,
    workflow_version_id: str | None = None,
    workflow_definition_id: str | None = None,
) -> tuple[WorkflowDefinitionRecord, WorkflowVersionRecord, WorkflowTriggerRecord | None]:
    normalized_trigger_id = str(workflow_trigger_id or "").strip()
    if normalized_trigger_id:
        trigger = (
            db.query(WorkflowTriggerRecord)
            .filter(WorkflowTriggerRecord.id == normalized_trigger_id)
            .first()
        )
        if trigger is None:
            raise HTTPException(status_code=404, detail="workflow_trigger_not_found")
        if not bool(trigger.enabled):
            raise HTTPException(status_code=400, detail="workflow_trigger_disabled")
        definition = (
            db.query(WorkflowDefinitionRecord)
            .filter(WorkflowDefinitionRecord.id == trigger.definition_id)
            .first()
        )
        if definition is None:
            raise HTTPException(status_code=404, detail="workflow_definition_not_found")
        requested_version_id = str(workflow_version_id or "").strip()
        version_query = db.query(WorkflowVersionRecord).filter(
            WorkflowVersionRecord.definition_id == definition.id
        )
        if requested_version_id:
            version_query = version_query.filter(WorkflowVersionRecord.id == requested_version_id)
        version = version_query.order_by(WorkflowVersionRecord.version_number.desc()).first()
        if version is None:
            raise HTTPException(status_code=404, detail="workflow_version_not_found")
        return definition, version, trigger

    normalized_version_id = str(workflow_version_id or "").strip()
    if normalized_version_id:
        version = (
            db.query(WorkflowVersionRecord)
            .filter(WorkflowVersionRecord.id == normalized_version_id)
            .first()
        )
        if version is None:
            raise HTTPException(status_code=404, detail="workflow_version_not_found")
        definition = (
            db.query(WorkflowDefinitionRecord)
            .filter(WorkflowDefinitionRecord.id == version.definition_id)
            .first()
        )
        if definition is None:
            raise HTTPException(status_code=404, detail="workflow_definition_not_found")
        return definition, version, None

    normalized_definition_id = str(workflow_definition_id or "").strip()
    if normalized_definition_id:
        definition = (
            db.query(WorkflowDefinitionRecord)
            .filter(WorkflowDefinitionRecord.id == normalized_definition_id)
            .first()
        )
        if definition is None:
            raise HTTPException(status_code=404, detail="workflow_definition_not_found")
        version = (
            db.query(WorkflowVersionRecord)
            .filter(WorkflowVersionRecord.definition_id == definition.id)
            .order_by(WorkflowVersionRecord.version_number.desc())
            .first()
        )
        if version is None:
            raise HTTPException(status_code=404, detail="workflow_version_not_found")
        return definition, version, None

    raise HTTPException(status_code=400, detail="chat_workflow_reference_missing")


def _prepare_chat_workflow_runtime_inputs(
    *,
    version: WorkflowVersionRecord,
    trigger: WorkflowTriggerRecord | None,
    context_json: Mapping[str, Any] | None,
    inputs: Mapping[str, Any] | None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any], dict[str, Any]]:
    trigger_context = (
        trigger.config_json.get("context_json")
        if isinstance(trigger, WorkflowTriggerRecord) and isinstance(trigger.config_json, dict)
        else None
    )
    trigger_inputs = (
        trigger.config_json.get("inputs")
        if isinstance(trigger, WorkflowTriggerRecord) and isinstance(trigger.config_json, dict)
        else None
    )
    context_envelope, runtime_inputs = context_service.build_workflow_runtime_context_envelope(
        db=None,
        goal=version.goal or "",
        version_context=version.context_json if isinstance(version.context_json, Mapping) else None,
        trigger_context=trigger_context if isinstance(trigger_context, Mapping) else None,
        request_context=context_json,
        trigger_inputs=trigger_inputs if isinstance(trigger_inputs, Mapping) else None,
        explicit_inputs=inputs,
        runtime_metadata={"surface": "chat_workflow_prepare"},
    )
    merged_context_json = context_service.workflow_runtime_context_view(context_envelope)

    workflow_interface, workflow_interface_errors, _workflow_interface_warnings = (
        _coerce_workflow_interface(
            version.draft_json.get("workflowInterface")
            if isinstance(version.draft_json, dict) and "workflowInterface" in version.draft_json
            else version.draft_json.get("workflow_interface")
            if isinstance(version.draft_json, dict)
            else None
        )
    )
    if workflow_interface_errors:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "workflow_interface_invalid",
                "diagnostics": {"errors": workflow_interface_errors, "warnings": []},
            },
        )

    for definition in workflow_interface.get("inputs", []):
        if not isinstance(definition, Mapping):
            continue
        key = str(definition.get("key") or "").strip()
        if not key or key in runtime_inputs:
            continue
        if key in merged_context_json:
            runtime_inputs[key] = merged_context_json[key]

    return workflow_interface, merged_context_json, runtime_inputs


def _execution_job_context(
    goal_text: str,
    job_context: Mapping[str, Any] | None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    normalized_goal = str(goal_text or "").strip()
    envelope = context_service.build_context_envelope(
        db=None,
        goal=normalized_goal,
        context_sources=context_service.collect_context_sources(job_context=job_context),
        normalized_intent_envelope=_normalized_intent_envelope_from_metadata(
            metadata,
            goal=normalized_goal,
        ),
        runtime_metadata={"surface": "execution_context"},
    )
    return context_service.execution_context_view(envelope)


def _preflight_job_context(
    *,
    db: Session | None,
    goal_text: str,
    job_context: Mapping[str, Any] | None,
    metadata: Mapping[str, Any] | None = None,
    surface: str,
) -> dict[str, Any]:
    normalized_goal = str(goal_text or "").strip()
    envelope = context_service.build_preflight_context_envelope(
        db=db,
        goal=normalized_goal,
        provided_job_context=job_context if isinstance(job_context, Mapping) and job_context else None,
        persisted_job_context=None,
        normalized_intent_envelope=_normalized_intent_envelope_from_metadata(
            metadata,
            goal=normalized_goal,
        ),
        runtime_metadata={"surface": surface},
    )
    return context_service.preflight_context_view(envelope)


def _dispatch_callbacks() -> dispatch_service.ApiDispatchCallbacks:
    return dispatch_service.ApiDispatchCallbacks(
        stream_for_event=_stream_for_event,
        resolve_task_deps=_resolve_task_deps,
        build_task_context=_build_task_context,
        project_execution_context=_execution_job_context,
        coerce_task_intent_profiles=_coerce_task_intent_profiles,
        normalize_task_intent_profile_segment=_normalize_task_intent_profile_segment,
        refresh_job_status=_refresh_job_status,
        emit_event=_emit_event,
    )


def _publish_envelope_to_redis(stream: str, envelope_json: str) -> tuple[bool, str | None]:
    return dispatch_service.publish_envelope_to_redis(
        _dispatch_runtime(),
        stream,
        envelope_json,
    )


def _insert_outbox_event(stream: str, event_type: str, envelope_json: str) -> str | None:
    return dispatch_service.insert_outbox_event(
        _dispatch_runtime(),
        stream,
        event_type,
        envelope_json,
    )


def _update_outbox_publish_state(
    outbox_id: str | None, published: bool, error: str | None = None
) -> None:
    dispatch_service.update_outbox_publish_state(
        _dispatch_runtime(),
        outbox_id,
        published,
        error,
    )


def _dispatch_event_outbox_once() -> int:
    return dispatch_service.dispatch_event_outbox_once(_dispatch_runtime())


def _start_event_outbox_dispatcher() -> None:
    dispatch_service.start_event_outbox_dispatcher(_dispatch_runtime())


def _outbox_entry_payload(
    row: EventOutboxRecord,
    include_payload: bool = False,
) -> dict[str, Any]:
    envelope = row.envelope_json if isinstance(row.envelope_json, dict) else {}
    payload: dict[str, Any] = {
        "id": row.id,
        "stream": row.stream,
        "event_type": row.event_type,
        "job_id": envelope.get("job_id"),
        "task_id": envelope.get("task_id"),
        "occurred_at": envelope.get("occurred_at"),
        "correlation_id": envelope.get("correlation_id"),
        "attempts": row.attempts or 0,
        "last_error": row.last_error,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "updated_at": row.updated_at.isoformat() if row.updated_at else None,
        "published_at": row.published_at.isoformat() if row.published_at else None,
    }
    if include_payload:
        payload["payload"] = envelope
    return payload


def _emit_event(event_type: str, payload: dict[str, Any]) -> None:
    dispatch_service.emit_event(
        _dispatch_runtime(),
        _dispatch_callbacks(),
        event_type,
        payload,
    )


def _resolve_download_path(path: str, root_dir: str, label: str) -> str:
    candidate = path.strip()
    if not candidate:
        raise HTTPException(status_code=400, detail="path is required")
    if candidate.startswith("/"):
        raise HTTPException(status_code=400, detail="path must be relative")
    root = Path(root_dir).resolve()
    target = (root / candidate).resolve()
    if not str(target).startswith(str(root)):
        raise HTTPException(status_code=400, detail=f"Invalid {label} path")
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail=f"{label.title()} file not found")
    return str(target)


def _resolve_artifact_path(path: str) -> str:
    return _resolve_download_path(path, ARTIFACTS_DIR, "artifact")


def _resolve_workspace_path(path: str) -> str:
    return _resolve_download_path(path, WORKSPACE_DIR, "workspace")


_SEMANTIC_TOKEN_RE = re.compile(r"[a-z0-9]{2,}")


def _semantic_default_user_id() -> str:
    candidate = SEMANTIC_MEMORY_DEFAULT_USER_ID.strip()
    return candidate or "default-user"


def _semantic_normalize_text(value: Any, *, max_len: int = 1000) -> str:
    if not isinstance(value, str):
        return ""
    normalized = re.sub(r"\s+", " ", value.strip())
    if len(normalized) > max_len:
        normalized = normalized[:max_len].strip()
    return normalized


def _semantic_normalize_list(values: Any, *, max_items: int = 32) -> list[str]:
    if not isinstance(values, list):
        return []
    normalized: list[str] = []
    seen: set[str] = set()
    for item in values:
        text = _semantic_normalize_text(item, max_len=120)
        if not text:
            continue
        lowered = text.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        normalized.append(text)
        if len(normalized) >= max_items:
            break
    return normalized


def _semantic_tokens(*chunks: Any) -> set[str]:
    tokens: set[str] = set()
    for chunk in chunks:
        text = _semantic_normalize_text(chunk, max_len=2000).lower()
        if not text:
            continue
        tokens.update(_SEMANTIC_TOKEN_RE.findall(text))
    return tokens


def _semantic_slug(value: str) -> str:
    normalized = _semantic_normalize_text(value, max_len=80).lower()
    slug = re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")
    return slug or "general"


def _semantic_build_key(namespace: str, subject: str, fact: str) -> str:
    fingerprint = hashlib.sha1(f"{namespace}|{subject}|{fact}".encode("utf-8")).hexdigest()[:12]
    return f"{_semantic_slug(namespace)}:{_semantic_slug(subject)}:{fingerprint}"


def _semantic_entry_to_match(entry: models.MemoryEntry, score: float, include_payload: bool) -> dict[str, Any]:
    payload = entry.payload if isinstance(entry.payload, dict) else {}
    match: dict[str, Any] = {
        "score": round(score, 4),
        "key": entry.key,
        "subject": payload.get("subject"),
        "fact": payload.get("fact"),
        "namespace": payload.get("namespace"),
        "confidence": payload.get("confidence"),
        "keywords": payload.get("keywords") if isinstance(payload.get("keywords"), list) else [],
        "aliases": payload.get("aliases") if isinstance(payload.get("aliases"), list) else [],
        "source": payload.get("source"),
        "source_ref": payload.get("source_ref"),
        "updated_at": entry.updated_at.isoformat() if entry.updated_at else None,
    }
    if include_payload:
        match["entry"] = entry.model_dump()
    return match


def _semantic_user_id_from_context(context: Mapping[str, Any] | None) -> str:
    if isinstance(context, Mapping):
        for key in ("user_id", "semantic_user_id"):
            candidate = _semantic_normalize_text(context.get(key), max_len=120)
            if candidate:
                return candidate
    return _semantic_default_user_id()


def _chat_authenticated_user_id(request: Request) -> str | None:
    state = getattr(request, "state", None)
    for attr in ("authenticated_user_id", "user_id"):
        candidate = _semantic_normalize_text(getattr(state, attr, None), max_len=120)
        if candidate:
            return candidate
    for header_name in ("X-Authenticated-User-Id", "X-User-Id"):
        candidate = _semantic_normalize_text(request.headers.get(header_name), max_len=120)
        if candidate:
            return candidate
    return None


def _feedback_actor_key(request: Request) -> str | None:
    authenticated = _chat_authenticated_user_id(request)
    if authenticated:
        return authenticated
    return _semantic_normalize_text(request.headers.get("X-Feedback-Actor-Id"), max_len=120)


def _retrieve_intent_workflow_hints(
    db: Session | None,
    *,
    goal: str,
    user_id: str,
    limit: int,
) -> list[dict[str, Any]]:
    if not INTENT_MEMORY_RETRIEVAL_ENABLED:
        return []
    if db is None:
        return []
    normalized_goal = _semantic_normalize_text(goal, max_len=1200)
    if not normalized_goal:
        return []
    normalized_user_id = _semantic_normalize_text(user_id, max_len=120) or _semantic_default_user_id()
    query_model = models.MemoryQuery(
        name="semantic_memory",
        scope=models.MemoryScope.user,
        user_id=normalized_user_id,
        limit=max(50, limit * 8),
    )
    try:
        entries = memory_store.read_memory(db, query_model)
    except Exception:  # noqa: BLE001
        logger.exception(
            "intent_memory_hint_read_failed",
            extra={"user_id": normalized_user_id},
        )
        return []
    query_tokens = _semantic_tokens(normalized_goal)
    query_lc = normalized_goal.lower()
    candidates: list[tuple[float, dict[str, Any]]] = []
    candidate_count = 0
    for entry in entries:
        if not isinstance(entry, models.MemoryEntry):
            continue
        payload = entry.payload if isinstance(entry.payload, dict) else {}
        if not payload:
            continue
        namespace = _semantic_normalize_text(payload.get("namespace"), max_len=120).lower()
        workflow = payload.get("intent_workflow")
        if namespace != "intent_workflows" or not isinstance(workflow, Mapping):
            continue
        outcome = _semantic_normalize_text(workflow.get("outcome"), max_len=60).lower()
        if outcome != "succeeded":
            continue
        workflow_goal = _semantic_normalize_text(workflow.get("goal"), max_len=1200)
        intent_order = _coerce_string_list(workflow.get("intent_order"))
        capabilities = _coerce_string_list(workflow.get("capabilities"))
        document_tokens = _semantic_tokens(
            workflow_goal,
            " ".join(intent_order),
            " ".join(capabilities),
            payload.get("fact"),
        )
        overlap = len(query_tokens.intersection(document_tokens))
        score = overlap / max(1, len(query_tokens))
        if query_lc and workflow_goal and query_lc in workflow_goal.lower():
            score += 0.45
        confidence_raw = workflow.get("confidence")
        if isinstance(confidence_raw, (int, float)):
            score += 0.1 * max(0.0, min(1.0, float(confidence_raw)))
        if score <= 0:
            continue
        candidate_count += 1
        candidates.append(
            (
                score,
                {
                    "key": entry.key,
                    "score": round(score, 4),
                    "goal": workflow_goal,
                    "intent_order": intent_order,
                    "capabilities": capabilities[:10],
                    "outcome": outcome,
                    "confidence": workflow.get("confidence"),
                },
            )
        )
    intent_memory_hint_candidates_total.inc(candidate_count)
    if not candidates:
        return []
    candidates.sort(key=lambda item: item[0], reverse=True)
    selected: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for _, hint in candidates:
        key = str(hint.get("key") or "").strip()
        if key and key in seen_keys:
            continue
        if key:
            seen_keys.add(key)
        selected.append(hint)
        if len(selected) >= max(1, limit):
            break
    intent_memory_hints_selected_total.inc(len(selected))
    return selected


def _persist_intent_workflow_memory(
    db: Session,
    *,
    job: JobRecord,
    tasks: list[TaskRecord],
    status: models.JobStatus,
) -> None:
    if not INTENT_MEMORY_PERSIST_ENABLED:
        return
    metadata = dict(job.metadata_json) if isinstance(job.metadata_json, dict) else {}
    if not isinstance(metadata, dict):
        return
    if metadata.get("intent_memory_persisted"):
        return
    normalized_envelope = _normalized_intent_envelope_from_metadata(metadata, goal=job.goal)
    if normalized_envelope is None:
        return
    profile = workflow_contracts.dump_goal_intent_profile(normalized_envelope.profile) or {}
    segments = _goal_intent_segments_from_metadata(metadata)
    if not profile and not segments:
        return
    user_id = _semantic_user_id_from_context(job.context_json if isinstance(job.context_json, Mapping) else None)
    intent_order = [
        str(segment.get("intent") or "").strip()
        for segment in segments
        if isinstance(segment, Mapping) and str(segment.get("intent") or "").strip()
    ]
    if not intent_order and isinstance(profile, Mapping):
        fallback_intent = intent_contract.normalize_task_intent(profile.get("intent"))
        if fallback_intent:
            intent_order = [fallback_intent]
    capabilities_from_graph: list[str] = []
    for segment in segments:
        if not isinstance(segment, Mapping):
            continue
        for capability_id in _coerce_string_list(segment.get("suggested_capabilities")):
            if capability_id not in capabilities_from_graph:
                capabilities_from_graph.append(capability_id)
    capabilities_from_tasks: list[str] = []
    for task in tasks:
        for tool_name in task.tool_requests or []:
            normalized = str(tool_name or "").strip()
            if not normalized:
                continue
            if normalized not in capabilities_from_tasks:
                capabilities_from_tasks.append(normalized)
    selected_capabilities = capabilities_from_graph or capabilities_from_tasks
    confidence = None
    threshold = None
    if isinstance(profile, Mapping):
        confidence_raw = profile.get("confidence")
        threshold_raw = profile.get("threshold")
        if isinstance(confidence_raw, (int, float)):
            confidence = round(max(0.0, min(1.0, float(confidence_raw))), 3)
        if isinstance(threshold_raw, (int, float)):
            threshold = round(max(0.0, min(1.0, float(threshold_raw))), 3)
    outcome = "failed"
    if status == models.JobStatus.succeeded:
        outcome = "succeeded"
    elif status == models.JobStatus.canceled:
        outcome = "canceled"
    fact = (
        f"Goal outcome={outcome}; intents={','.join(intent_order) or 'unknown'}; "
        f"capabilities={','.join(selected_capabilities[:8]) or 'none'}."
    )
    semantic_record = {
        "type": "semantic_fact",
        "namespace": "intent_workflows",
        "subject": f"goal:{_semantic_normalize_text(job.goal, max_len=120)}",
        "fact": fact,
        "aliases": [],
        "keywords": sorted(
            _semantic_tokens(job.goal, " ".join(intent_order), " ".join(selected_capabilities), outcome)
        )[:32],
        "confidence": 0.9 if outcome == "succeeded" else 0.4,
        "source": "job_outcome",
        "source_ref": job.id,
        "reasoning": "Auto-captured workflow outcome for intent decomposition retrieval.",
        "query_text": " ".join(
            part
            for part in [
                job.goal,
                " ".join(intent_order),
                " ".join(selected_capabilities),
                outcome,
            ]
            if part
        ),
        "captured_at": _utcnow().isoformat(),
        "intent_workflow": {
            "job_id": job.id,
            "goal": _semantic_normalize_text(job.goal, max_len=1200),
            "outcome": outcome,
            "intent_order": intent_order,
            "capabilities": selected_capabilities,
            "confidence": confidence,
            "threshold": threshold,
        },
    }
    memory_key = f"intent_workflow:{job.id}"
    try:
        write_request = memory_promotion_service.build_semantic_memory_write(
            user_id=user_id,
            key=memory_key,
            payload=semantic_record,
            metadata={
                "semantic": True,
                "semantic_namespace": "intent_workflows",
                "semantic_subject": semantic_record["subject"],
                "job_id_source": job.id,
                "outcome": outcome,
                "promotion_source": "intent_workflow_outcome",
            },
        )
        if isinstance(write_request.payload, Mapping):
            write_request.metadata["semantic_subject"] = str(
                write_request.payload.get("subject") or write_request.metadata.get("semantic_subject") or ""
            )
        memory_store.write_memory(db, write_request)
        metadata["intent_memory_persisted"] = True
        metadata["intent_memory_persisted_at"] = _utcnow().isoformat()
        metadata["intent_memory_key"] = memory_key
        metadata["intent_memory_user_id"] = user_id
        metadata["intent_memory_indexable"] = bool(write_request.metadata.get("indexable"))
        metadata["intent_memory_sensitive"] = bool(write_request.metadata.get("sensitive"))
        job.metadata_json = metadata
    except Exception:  # noqa: BLE001
        logger.exception("intent_memory_persist_failed", extra={"job_id": job.id})


def _load_schema_from_ref(schema_ref: str) -> dict[str, Any] | None:
    candidate = Path(schema_ref)
    candidates: list[Path] = []
    normalized_name = schema_ref if schema_ref.endswith(".json") else f"{schema_ref}.json"
    if candidate.is_absolute():
        candidates.append(candidate)
    else:
        candidates.append(Path(SCHEMA_REGISTRY_PATH) / normalized_name)
        repo_schema_dir = Path(__file__).resolve().parents[3] / "schemas"
        if repo_schema_dir not in candidates:
            candidates.append(repo_schema_dir / normalized_name)
        cwd_schema_dir = Path.cwd() / "schemas"
        if cwd_schema_dir != repo_schema_dir:
            candidates.append(cwd_schema_dir / normalized_name)
    for resolved_path in candidates:
        if not resolved_path.exists():
            continue
        try:
            parsed = json.loads(resolved_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def _schema_type_label(schema: dict[str, Any]) -> str:
    raw = schema.get("type")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    if isinstance(raw, list):
        normalized = sorted({str(item).strip() for item in raw if str(item).strip()})
        if normalized:
            return "|".join(normalized)
    if isinstance(schema.get("enum"), list):
        return "enum"
    if isinstance(schema.get("properties"), dict):
        return "object"
    if isinstance(schema.get("items"), dict):
        return "array"
    return "unknown"


def _flatten_schema_fields(
    schema: dict[str, Any] | None,
    *,
    max_depth: int = 8,
) -> list[dict[str, Any]]:
    if not isinstance(schema, dict):
        return []
    fields: list[dict[str, Any]] = []
    seen_paths: set[str] = set()

    def walk(node: Any, path: str, required: bool, depth: int) -> None:
        if depth > max_depth or not isinstance(node, dict):
            return
        if path and path not in seen_paths:
            fields.append(
                {
                    "path": path,
                    "required": required,
                    "type": _schema_type_label(node),
                }
            )
            seen_paths.add(path)
        properties = node.get("properties")
        if isinstance(properties, dict):
            required_keys = {
                key for key in node.get("required", []) if isinstance(key, str) and key.strip()
            }
            for key, child in properties.items():
                if not isinstance(key, str) or not key.strip():
                    continue
                child_path = f"{path}.{key}" if path else key
                walk(child, child_path, key in required_keys, depth + 1)
        items = node.get("items")
        if isinstance(items, dict):
            item_path = f"{path}[]" if path else "[]"
            walk(items, item_path, False, depth + 1)

    walk(schema, "", False, 0)
    return fields


def _resolve_capability_schemas(
    spec: capability_registry.CapabilitySpec,
    *,
    include_schemas: bool,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if not include_schemas:
        return None, None
    input_schema = _load_schema_from_ref(spec.input_schema_ref) if spec.input_schema_ref else None
    output_schema = _load_schema_from_ref(spec.output_schema_ref) if spec.output_schema_ref else None
    if input_schema is not None and output_schema is not None:
        return input_schema, output_schema
    for adapter in spec.adapters:
        if not adapter.enabled:
            continue
        try:
            tool = _tool_spec_registry.get(adapter.tool_name)
        except KeyError:
            continue
        if input_schema is None:
            input_schema = dict(tool.spec.input_schema)
        if output_schema is None:
            output_schema = dict(tool.spec.output_schema)
        if input_schema is not None and output_schema is not None:
            break
    return input_schema, output_schema


def _capability_required_inputs_for_intent_normalization(capability_id: str) -> list[str]:
    normalized = str(capability_id or "").strip()
    if not normalized:
        return []
    try:
        registry = capability_registry.load_capability_registry()
    except Exception:  # noqa: BLE001
        return []
    spec = registry.get(normalized)
    if spec is None:
        return []
    input_schema, _ = _resolve_capability_schemas(spec, include_schemas=True)
    if not isinstance(input_schema, dict):
        return []
    required = input_schema.get("required")
    if not isinstance(required, list):
        return []
    return [entry for entry in required if isinstance(entry, str)]


def _coerce_context_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
        except Exception:  # noqa: BLE001
            return {}
        if isinstance(parsed, dict):
            return parsed
    return {}


def _coerce_composer_nodes(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    nodes: list[dict[str, str]] = []
    for index, raw in enumerate(value):
        if not isinstance(raw, dict):
            continue
        node_id = str(raw.get("id") or f"node-{index + 1}").strip()
        capability_id = str(raw.get("capabilityId") or raw.get("capability_id") or "").strip()
        if not capability_id:
            continue
        output_path = str(raw.get("outputPath") or raw.get("output_path") or "").strip()
        nodes.append(
            {
                "id": node_id,
                "capability_id": capability_id,
                "output_path": output_path,
            }
        )
    return nodes


def _infer_capability_output_path(capability_id: str) -> str:
    normalized = (capability_id or "").strip().lower()
    if "spec.validate" in normalized:
        return "validation_report"
    if "spec" in normalized:
        if "openapi" in normalized:
            return "openapi_spec"
        return "document_spec"
    if "docx" in normalized or "pdf" in normalized or "render" in normalized:
        return "path"
    if "json.transform" in normalized:
        return "result"
    if "llm.text.generate" in normalized:
        return "text"
    return "result"


def _collect_recommendation_capabilities(
    *,
    include_disabled: bool = False,
) -> list[dict[str, Any]]:
    registry = capability_registry.load_capability_registry()
    items: list[dict[str, Any]] = []
    for capability_id, spec in sorted(registry.capabilities.items()):
        if not include_disabled and not spec.enabled:
            continue
        input_schema, _ = _resolve_capability_schemas(spec, include_schemas=True)
        required_inputs: list[str] = []
        if isinstance(input_schema, dict):
            required = input_schema.get("required")
            if isinstance(required, list):
                required_inputs = [entry for entry in required if isinstance(entry, str)]
        items.append(
            {
                "id": capability_id,
                "description": spec.description or "",
                "group": spec.group or "",
                "subgroup": spec.subgroup or "",
                "required_inputs": required_inputs,
            }
        )
    return items


def _is_capability_mentioned(goal_text: str, capability_id: str) -> bool:
    goal = (goal_text or "").strip()
    capability = (capability_id or "").strip()
    if not goal or not capability:
        return False
    pattern = re.compile(rf"(^|[^A-Za-z0-9_.-]){re.escape(capability)}([^A-Za-z0-9_.-]|$)")
    return bool(pattern.search(goal))


def _heuristic_capability_recommendations(
    *,
    goal: str,
    context: dict[str, Any],
    capabilities: list[dict[str, Any]],
    draft_nodes: list[dict[str, str]],
    max_results: int = 6,
) -> list[dict[str, Any]]:
    max_count = max(1, min(12, int(max_results)))
    last_node = draft_nodes[-1] if draft_nodes else None
    existing_capability_ids = {node.get("capability_id", "") for node in draft_nodes}
    recommendations: list[dict[str, Any]] = []

    for item in capabilities:
        capability_id = str(item.get("id") or "").strip()
        if not capability_id:
            continue
        required_inputs = [
            entry
            for entry in item.get("required_inputs", [])
            if isinstance(entry, str) and entry.strip()
        ]
        score = 0
        reasons: list[str] = []

        if _is_capability_mentioned(goal, capability_id):
            score += 100
            reasons.append("mentioned in goal")
        if capability_id in existing_capability_ids:
            score -= 6

        context_covered = [
            field
            for field in required_inputs
            if field in context
            and context[field] is not None
            and (not isinstance(context[field], str) or bool(context[field].strip()))
        ]
        if context_covered:
            score += min(24, len(context_covered) * 8)
            reasons.append(f"context covers {len(context_covered)} required input(s)")
        if not required_inputs:
            score += 4

        if last_node:
            last_output = (
                (last_node.get("output_path") or "").strip()
                or _infer_capability_output_path(last_node.get("capability_id", ""))
            )
            if last_output and last_output in required_inputs:
                score += 32
                reasons.append(f"uses previous output '{last_output}'")
        recommendations.append(
            {
                "id": capability_id,
                "score": score,
                "reason": " • ".join(reasons) if reasons else "general match",
                "confidence": round(max(0.05, min(0.99, (score + 20) / 140)), 3),
            }
        )

    recommendations.sort(key=lambda entry: (-int(entry["score"]), str(entry["id"])))
    return recommendations[:max_count]


def _coerce_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    items: list[str] = []
    for entry in value:
        if not isinstance(entry, str):
            continue
        normalized = entry.strip()
        if not normalized or normalized in items:
            continue
        items.append(normalized)
    return items


def _normalize_interaction_summaries(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("interaction_summaries_must_be_array")
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(value):
        if not isinstance(raw, dict):
            raise ValueError(f"interaction_summaries[{index}]_must_be_object")
        facts = _coerce_string_list(raw.get("facts"))
        evidence = _coerce_string_list(raw.get("evidence"))
        speculation = _coerce_string_list(raw.get("speculation"))
        action = ""
        for key in ("action", "user_action", "action_summary", "userAction"):
            candidate = raw.get(key)
            if isinstance(candidate, str) and candidate.strip():
                action = candidate.strip()
                break
        if not facts:
            raise ValueError(f"interaction_summaries[{index}].facts_required")
        if not action:
            raise ValueError(f"interaction_summaries[{index}].action_required")
        entry_id = str(raw.get("id") or f"i{index + 1}").strip() or f"i{index + 1}"
        normalized.append(
            {
                "id": entry_id,
                "facts": facts,
                "action": action,
                "evidence": evidence,
                "speculation": speculation,
            }
        )
    return normalized


def _truncate_summary_field(value: str, max_chars: int) -> str:
    text = value.strip()
    if len(text) <= max_chars:
        return text
    clipped = text[: max_chars - 1].rstrip()
    return f"{clipped}…"


def _estimate_interaction_summaries_tokens(interaction_summaries: list[dict[str, Any]]) -> int:
    if not interaction_summaries:
        return 0
    total_chars = 0
    for item in interaction_summaries:
        if not isinstance(item, dict):
            continue
        action = item.get("action")
        if isinstance(action, str):
            total_chars += len(action)
        for key in ("facts", "evidence", "speculation"):
            values = item.get(key)
            if not isinstance(values, list):
                continue
            total_chars += sum(len(entry) for entry in values if isinstance(entry, str))
    # Lightweight estimate: ~4 chars/token for English-like text.
    return max(1, (total_chars + 3) // 4)


def _compact_interaction_summaries(
    interaction_summaries: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not interaction_summaries:
        return [], {
            "applied": False,
            "reason": "empty",
            "input_count": 0,
            "output_count": 0,
            "input_tokens_est": 0,
            "output_tokens_est": 0,
            "dropped_items": 0,
        }

    input_count = len(interaction_summaries)
    input_tokens = _estimate_interaction_summaries_tokens(interaction_summaries)
    interaction_summary_tokens_in_total.inc(input_tokens)
    if not INTERACTION_SUMMARY_COMPACTION_ENABLED:
        interaction_summary_compactions_total.labels(applied="false").inc()
        interaction_summary_tokens_out_total.inc(input_tokens)
        return list(interaction_summaries), {
            "applied": False,
            "reason": "disabled",
            "input_count": input_count,
            "output_count": input_count,
            "input_tokens_est": input_tokens,
            "output_tokens_est": input_tokens,
            "dropped_items": 0,
        }

    max_items = INTERACTION_SUMMARY_MAX_ITEMS
    max_facts = INTERACTION_SUMMARY_MAX_FACTS_PER_ITEM
    max_chars = INTERACTION_SUMMARY_MAX_CHARS_PER_FIELD

    dropped_items = max(0, input_count - max_items)
    base_items = (
        list(interaction_summaries[-max_items:])
        if input_count > max_items
        else list(interaction_summaries)
    )
    compacted: list[dict[str, Any]] = []
    for raw in base_items:
        facts = [
            _truncate_summary_field(entry, max_chars)
            for entry in _coerce_string_list(raw.get("facts"))[:max_facts]
        ]
        if not facts:
            fallback_action = str(raw.get("action") or "").strip()
            if fallback_action:
                facts = [_truncate_summary_field(fallback_action, max_chars)]
            else:
                facts = ["interaction summary"]
        compacted.append(
            {
                "id": str(raw.get("id") or "").strip() or f"i{len(compacted) + 1}",
                "action": _truncate_summary_field(str(raw.get("action") or "unknown action"), max_chars),
                "facts": facts,
                "evidence": [
                    _truncate_summary_field(entry, max_chars)
                    for entry in _coerce_string_list(raw.get("evidence"))[:2]
                ],
                "speculation": [
                    _truncate_summary_field(entry, max_chars)
                    for entry in _coerce_string_list(raw.get("speculation"))[:1]
                ],
            }
        )

    # Compact every Nth entry to reduce repetitive detail while preserving latest context.
    if INTERACTION_SUMMARY_COMPACT_EVERY_N > 1 and len(compacted) > INTERACTION_SUMMARY_COMPACT_EVERY_N:
        kept: list[dict[str, Any]] = []
        for index, item in enumerate(compacted):
            is_recent_tail = index >= len(compacted) - INTERACTION_SUMMARY_COMPACT_EVERY_N
            if index % INTERACTION_SUMMARY_COMPACT_EVERY_N == 0 or is_recent_tail:
                kept.append(item)
        if kept:
            dropped_items += max(0, len(compacted) - len(kept))
            compacted = kept

    # Enforce token budget with progressive trimming.
    while len(compacted) > 1 and _estimate_interaction_summaries_tokens(compacted) > INTERACTION_SUMMARY_MAX_TOKENS:
        dropped_items += 1
        compacted.pop(0)
    if _estimate_interaction_summaries_tokens(compacted) > INTERACTION_SUMMARY_MAX_TOKENS:
        item = compacted[-1]
        item["facts"] = item["facts"][:1]
        item["evidence"] = []
        item["speculation"] = []
        item["action"] = _truncate_summary_field(str(item.get("action") or "action"), max_chars // 2)

    output_tokens = _estimate_interaction_summaries_tokens(compacted)
    output_count = len(compacted)
    applied = output_count != input_count or output_tokens < input_tokens
    interaction_summary_compactions_total.labels(applied=str(applied).lower()).inc()
    interaction_summary_tokens_out_total.inc(output_tokens)
    return compacted, {
        "applied": applied,
        "reason": "threshold" if applied else "no_change",
        "input_count": input_count,
        "output_count": output_count,
        "input_tokens_est": input_tokens,
        "output_tokens_est": output_tokens,
        "dropped_items": dropped_items,
        "reduction_ratio": round(
            max(0.0, min(1.0, 1.0 - (output_tokens / input_tokens))),
            3,
        )
        if input_tokens > 0
        else 0.0,
    }


def _persist_interaction_summaries_memory(
    db: Session,
    *,
    job_id: str,
    raw_summaries: list[dict[str, Any]],
    compact_summaries: list[dict[str, Any]],
    compaction: Mapping[str, Any],
    source: str,
) -> None:
    if not INTERACTION_SUMMARY_MEMORY_PERSIST or not job_id or not raw_summaries:
        return
    try:
        memory_store.write_memory(
            db,
            models.MemoryWrite(
                name="interaction_summaries",
                job_id=job_id,
                key="raw:latest",
                payload={"interaction_summaries": raw_summaries},
                metadata={
                    "source": source,
                    "count": len(raw_summaries),
                    "tokens_est": int(compaction.get("input_tokens_est") or 0),
                },
            ),
        )
        memory_store.write_memory(
            db,
            models.MemoryWrite(
                name="interaction_summaries_compact",
                job_id=job_id,
                key="latest",
                payload={"interaction_summaries": compact_summaries},
                metadata={
                    "source": source,
                    "count": len(compact_summaries),
                    "tokens_est": int(compaction.get("output_tokens_est") or 0),
                    "compaction": dict(compaction),
                },
            ),
        )
        interaction_summary_memory_persist_total.labels(status="ok").inc()
    except Exception:  # noqa: BLE001
        interaction_summary_memory_persist_total.labels(status="failed").inc()
        logger.exception("interaction_summaries_memory_persist_failed", extra={"job_id": job_id})


def _attach_interaction_compaction_to_graph(
    graph: workflow_contracts.IntentGraph,
    compaction: Mapping[str, Any],
) -> workflow_contracts.IntentGraph:
    return graph.model_copy(
        update={
            "summary": graph.summary.model_copy(
                update={"interaction_summary_compaction": dict(compaction)}
            )
        }
    )


def _interaction_summary_fact_corpus(interaction_summaries: list[dict[str, Any]]) -> list[str]:
    corpus: list[str] = []
    for item in interaction_summaries:
        for key in ("facts", "evidence", "speculation"):
            values = item.get(key)
            if isinstance(values, list):
                for entry in values:
                    if not isinstance(entry, str):
                        continue
                    normalized = entry.strip()
                    if normalized:
                        corpus.append(normalized.lower())
        action = item.get("action")
        if isinstance(action, str) and action.strip():
            corpus.append(action.strip().lower())
    return corpus


def _is_supported_intent_fact(fact: str, corpus: list[str]) -> bool:
    normalized = fact.strip().lower()
    if not normalized:
        return False
    for entry in corpus:
        if not entry:
            continue
        if normalized == entry:
            return True
        if len(normalized) >= 6 and normalized in entry:
            return True
        if len(entry) >= 6 and entry in normalized:
            return True
    return False


def _coerce_confidence(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return round(max(0.0, min(1.0, parsed)), 3)


def _intent_catalog_capability_ids() -> set[str]:
    try:
        registry = capability_registry.load_capability_registry()
        if hasattr(registry, "capabilities") and isinstance(
            getattr(registry, "capabilities"), Mapping
        ):
            return set(getattr(registry, "capabilities").keys())
        if isinstance(registry, Mapping):
            return set(registry.keys())
        return set()
    except Exception:  # noqa: BLE001
        logger.exception("intent_catalog_capability_registry_load_failed")
        return set()


def _intent_catalog_capability_entries() -> list[dict[str, str]]:
    try:
        registry = capability_registry.load_capability_registry()
        return capability_search.build_capability_search_entries(registry.enabled_capabilities())
    except Exception:  # noqa: BLE001
        logger.exception("intent_catalog_capability_registry_load_failed")
        return []


def _semantic_goal_capability_hints(
    *,
    goal: str,
    allowed_capability_catalog: list[dict[str, Any]],
    limit: int = 8,
    intent_hint: str | None = None,
    request_source: str = "intent_decompose",
) -> list[dict[str, Any]]:
    started = time.perf_counter()
    matches = capability_search.search_capabilities(
        query=goal,
        capability_entries=allowed_capability_catalog,
        limit=limit,
        intent_hint=intent_hint,
    )
    latency_ms = round((time.perf_counter() - started) * 1000.0, 3)
    _emit_capability_search_event(
        query=goal,
        intent_hint=intent_hint,
        limit=limit,
        matches=matches,
        request_source=request_source,
        latency_ms=latency_ms,
    )
    return matches


def _emit_capability_search_event(
    *,
    query: str,
    intent_hint: str | None,
    limit: int,
    matches: list[dict[str, Any]],
    request_source: str,
    latency_ms: float,
    correlation_id: str | None = None,
    job_id: str | None = None,
) -> None:
    normalized_source = str(request_source or "unknown").strip() or "unknown"
    normalized_intent = str(intent_hint or "none").strip().lower() or "none"
    capability_search_requests_total.labels(
        request_source=normalized_source,
        intent=normalized_intent,
        result="ok" if matches else "empty",
    ).inc()
    if matches:
        capability_search_results_total.labels(
            request_source=normalized_source,
            intent=normalized_intent,
        ).inc(len(matches))
    _emit_event(
        "plan.capability_search",
        {
            "correlation_id": correlation_id or str(uuid.uuid4()),
            "job_id": job_id,
            "query": query,
            "intent": intent_hint,
            "limit": limit,
            "request_source": normalized_source,
            "latency_ms": latency_ms,
            "results": [
                {
                    "id": str(entry.get("id") or "").strip(),
                    "score": float(entry.get("score") or 0.0),
                    "reason": str(entry.get("reason") or "").strip(),
                    "source": str(entry.get("source") or "semantic_search").strip()
                    or "semantic_search",
                }
                for entry in matches
                if str(entry.get("id") or "").strip()
            ],
            "result_count": len(matches),
        },
    )


_INTENT_CAPABILITY_HINTS: dict[str, tuple[str, ...]] = {
    "io": (
        "list",
        "search",
        "read",
        "write",
        "memory",
        "github",
        "filesystem",
        "repo",
        "file",
    ),
    "generate": ("generate", "create", "compose", "draft", "llm", "spec", "runbook", "codegen"),
    "transform": ("transform", "improve", "repair", "derive", "convert", "json"),
    "validate": ("validate", "schema", "lint", "check"),
    "render": ("render", "pdf", "docx"),
}


def _canonical_capability_id(
    capability_id: str,
    allowed_capability_ids: set[str],
) -> str | None:
    candidate = capability_registry.canonicalize_capability_id(capability_id)
    if not candidate:
        return None
    if not allowed_capability_ids:
        return candidate
    if candidate in allowed_capability_ids:
        return candidate
    lookup = {entry.lower(): entry for entry in allowed_capability_ids}
    return lookup.get(candidate.lower())


def _capability_task_intent_hint(capability_id: str) -> str | None:
    normalized_capability_id = capability_registry.canonicalize_capability_id(capability_id)
    if not normalized_capability_id:
        return None
    try:
        registry = capability_registry.load_capability_registry()
    except Exception:  # noqa: BLE001
        return None
    spec = registry.get(normalized_capability_id)
    if spec is None:
        return None
    planner_hints = spec.planner_hints if isinstance(spec.planner_hints, Mapping) else {}
    raw_task_intents = planner_hints.get("task_intents")
    if isinstance(raw_task_intents, list):
        normalized = [
            intent_contract.normalize_task_intent(item)
            for item in raw_task_intents
            if intent_contract.normalize_task_intent(item)
        ]
        if len(set(normalized)) == 1:
            return normalized[0]
    subgroup = str(spec.subgroup or "").strip().lower()
    tags = {str(tag).strip().lower() for tag in spec.tags if str(tag).strip()}
    if subgroup == "rendering" or "render" in tags:
        return "render"
    return None


def _reconcile_intent_with_capabilities(intent: str, capability_ids: Sequence[str]) -> str:
    hinted = {
        hint
        for capability_id in capability_ids
        for hint in [_capability_task_intent_hint(capability_id)]
        if hint
    }
    if len(hinted) == 1:
        return next(iter(hinted))
    return intent


def _normalize_catalog_capability_ids(
    capability_ids: list[str],
    allowed_capability_ids: set[str],
    *,
    limit: int = 3,
) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for capability_id in capability_ids:
        canonical = _canonical_capability_id(capability_id, allowed_capability_ids)
        if not canonical or canonical in seen:
            continue
        seen.add(canonical)
        normalized.append(canonical)
        if len(normalized) >= max(1, limit):
            break
    return normalized


def _rank_catalog_capability_ids(
    capability_ids: list[str],
    *,
    allowed_capability_ids: set[str],
    source: str,
    reason: str,
    limit: int = 3,
) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, capability_id in enumerate(capability_ids):
        canonical = _canonical_capability_id(capability_id, allowed_capability_ids)
        if not canonical or canonical in seen:
            continue
        seen.add(canonical)
        ranked.append(
            {
                "id": canonical,
                "score": round(max(0.05, 1.0 - (index * 0.05)), 3),
                "reason": reason,
                "source": source,
            }
        )
        if len(ranked) >= max(1, limit):
            break
    return ranked


def _rank_default_capabilities_for_intent_segment(
    *,
    intent: str,
    objective: str,
    allowed_capability_ids: set[str],
    allowed_capability_catalog: list[dict[str, str]],
    limit: int = 3,
) -> list[dict[str, Any]]:
    if not allowed_capability_ids:
        return []
    semantic_ranked = _semantic_goal_capability_hints(
        goal=objective,
        allowed_capability_catalog=allowed_capability_catalog,
        limit=limit,
        intent_hint=intent,
        request_source="intent_decompose",
    )
    semantic_filtered = [
        item
        for item in semantic_ranked
        if str(item.get("id") or "").strip() in allowed_capability_ids
    ]
    if semantic_filtered:
        return semantic_filtered[: max(1, limit)]
    intent_hints = _INTENT_CAPABILITY_HINTS.get(intent, ())
    objective_lower = objective.lower()
    objective_tokens = {
        token
        for token in re.findall(r"[a-z0-9]+", objective_lower)
        if len(token) >= 4
    }
    scored: list[tuple[float, str, list[str]]] = []
    for entry in allowed_capability_catalog:
        capability_id = str(entry.get("id") or "").strip()
        if not capability_id or capability_id not in allowed_capability_ids:
            continue
        search_blob = str(entry.get("search_blob") or capability_id.lower())
        score = 0.0
        reasons: list[str] = []
        if capability_id.lower() in objective_lower:
            score += 120.0
            reasons.append("objective mentions capability id")
        hint_matches = 0
        for hint in intent_hints:
            if hint in search_blob:
                score += 8.0
                hint_matches += 1
        if hint_matches:
            reasons.append(f"matches {hint_matches} intent hint(s)")
        token_matches = 0
        for token in objective_tokens:
            if token in search_blob:
                score += 3.0
                token_matches += 1
        if token_matches:
            reasons.append(f"matches {token_matches} objective token(s)")
        if "pdf" in objective_lower and "pdf" in search_blob:
            score += 25.0
            reasons.append("pdf objective alignment")
        if "docx" in objective_lower and "docx" in search_blob:
            score += 25.0
            reasons.append("docx objective alignment")
        if intent == "render" and any(token in search_blob for token in ("pdf", "docx", "render")):
            score += 10.0
            reasons.append("render intent alignment")
        if intent == "validate" and "validate" in search_blob:
            score += 10.0
            reasons.append("validate intent alignment")
        if intent == "transform" and any(
            token in search_blob for token in ("transform", "improve", "repair", "derive")
        ):
            score += 10.0
            reasons.append("transform intent alignment")
        if intent == "generate" and any(
            token in search_blob for token in ("generate", "create", "llm", "spec")
        ):
            score += 10.0
            reasons.append("generate intent alignment")
        if intent == "io" and any(
            token in search_blob for token in ("list", "search", "read", "write", "memory", "github")
        ):
            score += 10.0
            reasons.append("io intent alignment")
        if score > 0:
            scored.append((score, capability_id, reasons))

    if not scored:
        for capability_id in sorted(allowed_capability_ids):
            if capability_id.startswith(f"{intent}."):
                scored.append((1.0, capability_id, ["intent-prefix fallback"]))
        if not scored:
            return [
                {
                    "id": capability_id,
                    "score": 0.5,
                    "reason": "catalog fallback",
                    "source": "heuristic",
                }
                for capability_id in sorted(allowed_capability_ids)[: max(1, limit)]
            ]

    scored.sort(key=lambda item: (-item[0], item[1]))
    ranked: list[dict[str, Any]] = []
    seen: set[str] = set()
    for score, capability_id, reasons in scored:
        if capability_id in seen:
            continue
        seen.add(capability_id)
        ranked.append(
            {
                "id": capability_id,
                "score": round(float(score), 3),
                "reason": "; ".join(_coerce_string_list(reasons))
                or "heuristic ranking from intent/objective",
                "source": "heuristic",
            }
        )
        if len(ranked) >= max(1, limit):
            break
    return ranked


def _llm_goal_intent_prompt(
    *,
    goal: str,
    allowed_capability_catalog: list[dict[str, str]],
    semantic_goal_capabilities: list[dict[str, Any]],
    capability_top_k: int,
) -> str:
    allowed_intents = ", ".join(member.value for member in models.ToolIntent)
    prompt = (
        "Classify the user's primary execution intent for the goal.\n"
        "Return ONLY JSON with this shape:\n"
        '{ "intent": "generate", "confidence": 0.0, "suggested_capabilities": ["capability.id"], '
        '"reason": "short rationale" }\n'
        f"Allowed intent values: {allowed_intents}.\n"
        "Rules:\n"
        "- choose exactly one primary intent for the next best execution action\n"
        "- confidence must be between 0 and 1\n"
        "- suggested_capabilities must use only exact IDs from the capability lists below\n"
        f"- suggested_capabilities should contain at most {capability_top_k} IDs\n"
        f"Goal:\n{goal}\n"
    )
    if semantic_goal_capabilities:
        prompt += "Most relevant allowed capabilities for this goal:\n"
        prompt += json.dumps(semantic_goal_capabilities[:8], ensure_ascii=True)
        prompt += "\n"
    if allowed_capability_catalog:
        catalog_lines = [
            (
                f"- {entry['id']}"
                f" | group={entry.get('group') or '-'}"
                f" | subgroup={entry.get('subgroup') or '-'}"
                f" | description={(entry.get('description') or '')[:160]}"
            )
            for entry in allowed_capability_catalog[:48]
            if str(entry.get("id") or "").strip()
        ]
        if catalog_lines:
            prompt += "Allowed capability catalog sample:\n"
            prompt += "\n".join(catalog_lines)
            prompt += "\n"
    return prompt


def _llm_infer_goal_intent_with_capabilities(
    *,
    goal: str,
    provider: LLMProvider,
    fallback_inference: intent_contract.TaskIntentInference,
    allowed_capability_ids: set[str],
    allowed_capability_catalog: list[dict[str, str]],
    capability_top_k: int,
    semantic_goal_capabilities: list[dict[str, Any]] | None = None,
) -> intent_contract.TaskIntentInference:
    prompt = _llm_goal_intent_prompt(
        goal=goal,
        allowed_capability_catalog=allowed_capability_catalog,
        semantic_goal_capabilities=semantic_goal_capabilities or [],
        capability_top_k=capability_top_k,
    )
    parsed = provider.generate_request_json_object(
        LLMRequest(
            prompt=prompt,
            metadata={
                "component": "api",
                "operation": "intent_assess",
                "goal_len": len(goal),
                "capability_catalog_size": len(allowed_capability_catalog),
                "capability_top_k": capability_top_k,
                "semantic_goal_capabilities": len(semantic_goal_capabilities or []),
            },
        )
    )
    intent = intent_contract.normalize_task_intent(parsed.get("intent"))
    if not intent:
        raise ValueError("llm_intent_assess_invalid_intent")
    suggested_capabilities = _normalize_catalog_capability_ids(
        _coerce_string_list(parsed.get("suggested_capabilities")),
        allowed_capability_ids,
        limit=capability_top_k,
    )
    intent = _reconcile_intent_with_capabilities(intent, suggested_capabilities)
    confidence_default = max(0.4, min(0.95, float(fallback_inference.confidence or 0.6)))
    confidence = _coerce_confidence(parsed.get("confidence"), confidence_default)
    if semantic_goal_capabilities and suggested_capabilities:
        semantic_ids = {
            str(item.get("id") or "").strip()
            for item in semantic_goal_capabilities
            if str(item.get("id") or "").strip()
        }
        if not any(capability_id in semantic_ids for capability_id in suggested_capabilities):
            confidence = min(confidence, max(0.45, confidence_default))
    return intent_contract.TaskIntentInference(
        intent=intent,
        source="llm",
        confidence=confidence,
    )


def _infer_goal_intent_with_metadata(
    goal: str,
    *,
    mode_override: str | None = None,
) -> intent_contract.TaskIntentInference:
    heuristic = intent_contract.infer_task_intent_from_goal_with_metadata(goal)
    semantic_inference = (
        _hybrid_goal_intent_inference(goal, heuristic)
        if _should_run_goal_intent_vector_search(heuristic)
        else heuristic
    )
    assess_mode = str(mode_override or INTENT_ASSESS_MODE).strip().lower()
    if assess_mode not in {"heuristic", "llm", "hybrid"}:
        assess_mode = "heuristic"
    if not INTENT_ASSESS_ENABLED or assess_mode == "heuristic":
        return semantic_inference
    if _should_skip_llm_goal_intent_assessment(semantic_inference, assess_mode=assess_mode):
        return semantic_inference
    provider = _intent_assess_provider
    if provider is None:
        return semantic_inference
    allowed_capability_catalog = _intent_catalog_capability_entries()
    allowed_capability_ids = {
        str(entry.get("id") or "").strip()
        for entry in allowed_capability_catalog
        if str(entry.get("id") or "").strip()
    } or _intent_catalog_capability_ids()
    semantic_goal_capabilities = _semantic_goal_capability_hints(
        goal=goal,
        allowed_capability_catalog=allowed_capability_catalog,
        limit=max(4, INTENT_CAPABILITY_TOP_K * 2),
        request_source="intent_assess",
    )
    try:
        return _llm_infer_goal_intent_with_capabilities(
            goal=goal,
            provider=provider,
            fallback_inference=semantic_inference,
            allowed_capability_ids=allowed_capability_ids,
            allowed_capability_catalog=allowed_capability_catalog,
            capability_top_k=INTENT_CAPABILITY_TOP_K,
            semantic_goal_capabilities=semantic_goal_capabilities,
        )
    except Exception:  # noqa: BLE001
        logger.exception(
            "intent_assess_llm_failed",
            extra={"goal": goal[:160], "mode": assess_mode},
        )
        return semantic_inference


def _filter_catalog_capability_ids(
    capability_ids: list[str], allowed_capability_ids: set[str]
) -> list[str]:
    if not capability_ids:
        return []
    if not allowed_capability_ids:
        return capability_ids
    return [capability_id for capability_id in capability_ids if capability_id in allowed_capability_ids]


def _normalize_llm_intent_graph(
    *,
    goal: str,
    parsed: dict[str, Any],
    fallback_graph: dict[str, Any],
    allowed_capability_ids: set[str],
    allowed_capability_catalog: list[dict[str, str]],
    capability_top_k: int,
) -> dict[str, Any]:
    raw_segments = parsed.get("segments")
    if not isinstance(raw_segments, list) or not raw_segments:
        raise ValueError("llm_intent_graph_missing_segments")
    allowed_intents = {member.value for member in models.ToolIntent}
    fallback_segments = (
        fallback_graph.get("segments", [])
        if isinstance(fallback_graph.get("segments"), list)
        else []
    )
    normalized_segments: list[dict[str, Any]] = []
    seen_ids: list[str] = []
    capability_suggestions_total = 0
    capability_suggestions_matched = 0
    capability_suggestions_selected = 0
    capability_suggestions_autofilled = 0

    for index, raw_segment in enumerate(raw_segments[:8]):
        if not isinstance(raw_segment, dict):
            continue
        fallback_segment = (
            fallback_segments[index] if index < len(fallback_segments) else {}
        )
        segment_id_raw = str(raw_segment.get("id") or "").strip()
        segment_id = (
            segment_id_raw
            if re.fullmatch(r"s[1-9]\d*", segment_id_raw) and segment_id_raw not in seen_ids
            else f"s{index + 1}"
        )
        intent_raw = str(raw_segment.get("intent") or "").strip().lower()
        fallback_intent = str(fallback_segment.get("intent") or models.ToolIntent.generate.value)
        intent = intent_raw if intent_raw in allowed_intents else fallback_intent
        objective = str(raw_segment.get("objective") or "").strip() or str(
            fallback_segment.get("objective") or goal
        )
        confidence_default = float(fallback_segment.get("confidence") or 0.6)
        confidence = _coerce_confidence(raw_segment.get("confidence"), confidence_default)

        required_inputs = _coerce_string_list(raw_segment.get("required_inputs"))
        if not required_inputs:
            required_inputs = _coerce_string_list(fallback_segment.get("required_inputs"))
        raw_suggested_capabilities = _coerce_string_list(raw_segment.get("suggested_capabilities"))
        capability_suggestions_total += len(raw_suggested_capabilities)
        ranked_capabilities = _rank_catalog_capability_ids(
            raw_suggested_capabilities,
            allowed_capability_ids=allowed_capability_ids,
            source="llm",
            reason="llm suggested and catalog-validated",
            limit=capability_top_k,
        )
        capability_suggestions_matched += len(ranked_capabilities)
        if not ranked_capabilities:
            ranked_capabilities = _rank_catalog_capability_ids(
                _coerce_string_list(fallback_segment.get("suggested_capabilities")),
                allowed_capability_ids=allowed_capability_ids,
                source="fallback_segment",
                reason="fallback segment suggestion from heuristic decomposition",
                limit=capability_top_k,
            )
        if not ranked_capabilities:
            ranked_capabilities = _rank_default_capabilities_for_intent_segment(
                intent=intent,
                objective=objective,
                allowed_capability_ids=allowed_capability_ids,
                allowed_capability_catalog=allowed_capability_catalog,
                limit=capability_top_k,
            )
            capability_suggestions_autofilled += len(ranked_capabilities)
        capability_suggestions_selected += len(ranked_capabilities)
        suggested_capabilities = [
            str(item.get("id") or "").strip()
            for item in ranked_capabilities
            if isinstance(item, Mapping) and str(item.get("id") or "").strip()
        ]
        intent = _reconcile_intent_with_capabilities(intent, suggested_capabilities)
        slots = intent_contract.normalize_intent_segment_slots(
            raw_slots=raw_segment.get("slots"),
            fallback_slots=(
                fallback_segment.get("slots")
                if isinstance(fallback_segment.get("slots"), Mapping)
                else None
            ),
            intent=intent,
            objective=objective,
            required_inputs=required_inputs,
            suggested_capabilities=suggested_capabilities,
        )

        depends_raw = _coerce_string_list(raw_segment.get("depends_on"))
        depends_on = [dep for dep in depends_raw if dep in seen_ids]
        if not depends_on and seen_ids:
            depends_on = [seen_ids[-1]]

        objective_facts = _coerce_string_list(raw_segment.get("objective_facts"))
        if not objective_facts:
            objective_facts = _coerce_string_list(fallback_segment.get("objective_facts"))

        normalized_segments.append(
            {
                "id": segment_id,
                "intent": intent,
                "objective": objective,
                "objective_facts": objective_facts,
                "source": "llm",
                "confidence": confidence,
                "depends_on": depends_on,
                "required_inputs": required_inputs,
                "suggested_capabilities": suggested_capabilities,
                "suggested_capability_rankings": ranked_capabilities,
                "slots": slots,
            }
        )
        seen_ids.append(segment_id)

    if not normalized_segments:
        raise ValueError("llm_intent_graph_empty")

    overall_confidence = round(
        sum(float(segment["confidence"]) for segment in normalized_segments)
        / len(normalized_segments),
        3,
    )
    return {
        "goal": goal,
        "segments": normalized_segments,
        "summary": {
            "segment_count": len(normalized_segments),
            "intent_order": [str(segment["intent"]) for segment in normalized_segments],
            "capability_suggestions_total": capability_suggestions_total,
            "capability_suggestions_matched": capability_suggestions_matched,
            "capability_suggestions_selected": capability_suggestions_selected,
            "capability_suggestions_autofilled": capability_suggestions_autofilled,
            "capability_top_k": capability_top_k,
            "capability_match_rate": round(
                capability_suggestions_matched / capability_suggestions_total, 3
            )
            if capability_suggestions_total > 0
            else 1.0,
            "schema_version": "intent_v2",
        },
        "overall_confidence": overall_confidence,
        "source": "llm",
    }


def _llm_decompose_goal_intent(
    *,
    goal: str,
    provider: LLMProvider,
    fallback_graph: dict[str, Any],
    allowed_capability_ids: set[str],
    allowed_capability_catalog: list[dict[str, str]],
    capability_top_k: int,
    interaction_summaries: list[dict[str, Any]] | None = None,
    workflow_hints: list[dict[str, Any]] | None = None,
    semantic_goal_capabilities: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    allowed_intents = ", ".join(member.value for member in models.ToolIntent)
    catalog_lines = [
        (
            f"- {entry['id']}"
            f" | group={entry.get('group') or '-'}"
            f" | subgroup={entry.get('subgroup') or '-'}"
            f" | description={(entry.get('description') or '')[:160]}"
        )
        for entry in allowed_capability_catalog
    ]
    prompt = (
        "Decompose the goal into an ordered intent graph for an agent workflow.\n"
        "Return ONLY JSON with this shape:\n"
        '{ "segments": [ { "id": "s1", "intent": "generate", "objective": "short objective", '
        '"objective_facts": ["grounded_fact"], "confidence": 0.0, "depends_on": [], '
        '"required_inputs": ["input_name"], '
        '"suggested_capabilities": ["capability.id"], '
        '"slots": { "entity": "artifact", "artifact_type": "document", '
        '"output_format": "pdf", "risk_level": "bounded_write", '
        '"must_have_inputs": ["path","document_spec"] } } ] }\n'
        f"Allowed intent values: {allowed_intents}.\n"
        "Allowed capability IDs are listed below. suggested_capabilities MUST use only these exact IDs.\n"
        "Rules:\n"
        "- include 1 to 8 segments\n"
        "- keep objectives concrete and short\n"
        "- objective_facts must be grounded claims only (no speculation)\n"
        "- use depends_on only for prior segment IDs\n"
        "- confidence must be between 0 and 1\n"
        "- keep intent consistent with suggested_capabilities and capability subgroup semantics\n"
        "- document.docx.render and document.pdf.render MUST use intent=render\n"
        "- slots.output_format must be one of: pdf, docx, md, txt, html, json, csv, xlsx when known\n"
        "- slots.risk_level must be one of: read_only, bounded_write, high_risk_write\n"
        "- slots.must_have_inputs should contain concrete input keys (not prose)\n"
        "- suggested_capabilities must use exact IDs from the allowed capability catalog\n"
        f"- suggested_capabilities per segment should be at most {capability_top_k} IDs\n"
        f"Goal:\n{goal}\n"
    )
    if catalog_lines:
        prompt += "Allowed capability catalog:\n"
        prompt += "\n".join(catalog_lines)
        prompt += "\n"
    if workflow_hints:
        prompt += "Similar successful workflow hints from semantic memory:\n"
        prompt += json.dumps(workflow_hints[:5], ensure_ascii=True)
        prompt += (
            "\nUse these as priors for segment ordering and suggested_capabilities "
            "only when they fit the current goal.\n"
        )
    if semantic_goal_capabilities:
        prompt += "Most relevant capabilities for this goal from local semantic search:\n"
        prompt += json.dumps(semantic_goal_capabilities[:8], ensure_ascii=True)
        prompt += (
            "\nPrefer these capability IDs when they fit the goal and segment objective.\n"
        )
    if interaction_summaries:
        prompt += "Interaction summaries (grounding evidence):\n"
        prompt += json.dumps(interaction_summaries[:32], ensure_ascii=True)
        prompt += "\nOnly include objective_facts that are directly supported by these summaries.\n"
    parsed = provider.generate_request_json_object(
        LLMRequest(
            prompt=prompt,
            metadata={
                "component": "api",
                "operation": "intent_decompose",
                "goal_len": len(goal),
                "capability_catalog_size": len(allowed_capability_catalog),
                "capability_top_k": capability_top_k,
                "workflow_hints": len(workflow_hints or []),
                "semantic_goal_capabilities": len(semantic_goal_capabilities or []),
                "interaction_summaries": len(interaction_summaries or []),
            },
        )
    )
    return _normalize_llm_intent_graph(
        goal=goal,
        parsed=parsed,
        fallback_graph=fallback_graph,
        allowed_capability_ids=allowed_capability_ids,
        allowed_capability_catalog=allowed_capability_catalog,
        capability_top_k=capability_top_k,
    )


def _segment_fact_candidates(segment: Mapping[str, Any]) -> list[str]:
    objective_facts = _coerce_string_list(segment.get("objective_facts"))
    if objective_facts:
        return objective_facts
    objective = str(segment.get("objective") or "").strip()
    if not objective:
        return []
    parts = [
        entry.strip(" ,.-")
        for entry in re.split(r"(?:[;\n]|(?:\bthen\b)|(?:\band\b)|(?:\bwith\b))", objective, flags=re.IGNORECASE)
    ]
    normalized: list[str] = []
    for part in parts:
        if not part:
            continue
        if part not in normalized:
            normalized.append(part)
    if normalized:
        return normalized
    return [objective]


def _fallback_objective_for_intent(intent: str) -> str:
    mapping = {
        "io": "collect required source data",
        "generate": "generate requested content",
        "transform": "transform intermediate output",
        "validate": "validate output quality",
        "render": "render final artifact",
    }
    return mapping.get(intent, "complete workflow step")


def _apply_supported_fact_filter(
    graph: dict[str, Any],
    interaction_summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    segments = graph.get("segments")
    if not isinstance(segments, list):
        return graph
    corpus = _interaction_summary_fact_corpus(interaction_summaries)
    if not corpus:
        return graph

    normalized_segments: list[dict[str, Any]] = []
    fact_candidates = 0
    fact_supported = 0
    fact_stripped = 0
    for raw_segment in segments:
        if not isinstance(raw_segment, dict):
            continue
        segment = dict(raw_segment)
        candidates = _segment_fact_candidates(segment)
        supported = [
            candidate
            for candidate in candidates
            if _is_supported_intent_fact(candidate, corpus)
        ]
        stripped = [candidate for candidate in candidates if candidate not in supported]
        fact_candidates += len(candidates)
        fact_supported += len(supported)
        fact_stripped += len(stripped)

        if supported:
            segment["objective_facts"] = supported
            segment["objective"] = "; ".join(supported)
        else:
            segment["objective_facts"] = []
            segment["objective"] = _fallback_objective_for_intent(
                str(segment.get("intent") or "")
            )
            segment["confidence"] = round(
                max(0.0, min(1.0, float(segment.get("confidence") or 0.0) * 0.7)),
                3,
            )
        if stripped:
            segment["unsupported_facts"] = stripped
        normalized_segments.append(segment)

    summary_raw = graph.get("summary")
    summary = dict(summary_raw) if isinstance(summary_raw, dict) else {}
    summary.update(
        {
            "fact_candidates": fact_candidates,
            "fact_supported": fact_supported,
            "fact_stripped": fact_stripped,
            "fact_support_rate": round(fact_supported / fact_candidates, 3)
            if fact_candidates > 0
            else 1.0,
            "has_interaction_summaries": True,
        }
    )
    return {**graph, "segments": normalized_segments, "summary": summary}


def _annotate_graph_summary_defaults(
    graph: dict[str, Any],
    *,
    has_interaction_summaries: bool,
    allowed_capability_ids: set[str],
) -> dict[str, Any]:
    segments_raw = graph.get("segments")
    segments = segments_raw if isinstance(segments_raw, list) else []
    summary_raw = graph.get("summary")
    summary = dict(summary_raw) if isinstance(summary_raw, dict) else {}

    capability_suggestions_total = 0
    capability_suggestions_matched = 0
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        suggested = _coerce_string_list(segment.get("suggested_capabilities"))
        capability_suggestions_total += len(suggested)
        if allowed_capability_ids:
            capability_suggestions_matched += len(
                [item for item in suggested if item in allowed_capability_ids]
            )
        else:
            capability_suggestions_matched += len(suggested)

    summary.setdefault("segment_count", len(segments))
    summary.setdefault(
        "intent_order",
        [
            str(segment.get("intent") or "").strip()
            for segment in segments
            if isinstance(segment, dict) and str(segment.get("intent") or "").strip()
        ],
    )
    summary.setdefault("fact_candidates", 0)
    summary.setdefault("fact_supported", 0)
    summary.setdefault("fact_stripped", 0)
    summary.setdefault("fact_support_rate", 1.0)
    summary.setdefault("has_interaction_summaries", has_interaction_summaries)
    summary.setdefault("capability_suggestions_total", capability_suggestions_total)
    summary.setdefault("capability_suggestions_matched", capability_suggestions_matched)
    summary.setdefault("capability_suggestions_selected", capability_suggestions_matched)
    summary.setdefault("capability_suggestions_autofilled", 0)
    summary.setdefault("capability_top_k", INTENT_CAPABILITY_TOP_K)
    summary.setdefault("schema_version", "intent_v2")
    summary.setdefault(
        "capability_match_rate",
        round(capability_suggestions_matched / capability_suggestions_total, 3)
        if capability_suggestions_total > 0
        else 1.0,
    )
    return {**graph, "summary": summary}


def _record_intent_decompose_metrics(
    *,
    graph: dict[str, Any],
    result: str,
    has_interaction_summaries: bool,
) -> None:
    source = _intent_source_label(graph.get("source"))
    model_label = (INTENT_DECOMPOSE_MODEL or LLM_MODEL_NAME or "none").strip() or "none"
    mode_label = INTENT_DECOMPOSE_MODE
    intent_decompose_requests_total.labels(
        mode=mode_label,
        model=model_label,
        source=source,
        result=result,
        has_summaries=str(bool(has_interaction_summaries)).lower(),
        fallback_used=str(_intent_decompose_fallback_used(source)).lower(),
    ).inc()
    segments = graph.get("segments")
    if isinstance(segments, list) and segments:
        intent_decompose_segments_total.labels(source=source, model=model_label).inc(len(segments))
    summary = graph.get("summary")
    if isinstance(summary, dict):
        candidates = int(summary.get("fact_candidates") or 0)
        supported = int(summary.get("fact_supported") or 0)
        stripped = int(summary.get("fact_stripped") or 0)
        cap_total = int(summary.get("capability_suggestions_total") or 0)
        cap_matched = int(summary.get("capability_suggestions_matched") or 0)
        if candidates > 0:
            intent_fact_candidates_total.labels(source=source, model=model_label).inc(candidates)
        if supported > 0:
            intent_fact_supported_total.labels(source=source, model=model_label).inc(supported)
        if stripped > 0:
            intent_fact_stripped_total.labels(source=source, model=model_label).inc(stripped)
        if cap_total > 0:
            intent_capability_suggestions_total.labels(source=source, model=model_label).inc(
                cap_total
            )
        if cap_matched > 0:
            intent_capability_suggestions_matched_total.labels(source=source, model=model_label).inc(
                cap_matched
            )


def _on_intent_decompose_llm_failure(exc: Exception) -> None:
    intent_decompose_failures_total.labels(
        mode=INTENT_DECOMPOSE_MODE,
        model=(INTENT_DECOMPOSE_MODEL or LLM_MODEL_NAME or "none").strip() or "none",
        error_type=type(exc).__name__,
    ).inc()
    logger.exception("intent_decompose_llm_failed")


def _decompose_goal_intent(
    goal: str,
    *,
    db: Session | None = None,
    user_id: str | None = None,
    interaction_summaries: list[dict[str, Any]] | None = None,
) -> workflow_contracts.IntentGraph:
    return intent_service.decompose_goal_intent(
        goal,
        db=db,
        user_id=user_id,
        interaction_summaries=interaction_summaries,
        config=intent_service.IntentDecomposeConfig(
            enabled=INTENT_DECOMPOSE_ENABLED,
            mode=INTENT_DECOMPOSE_MODE,
            capability_top_k=INTENT_CAPABILITY_TOP_K,
            memory_retrieval_enabled=INTENT_MEMORY_RETRIEVAL_ENABLED,
            memory_retrieval_limit=INTENT_MEMORY_RETRIEVAL_LIMIT,
        ),
        runtime=intent_service.IntentDecomposeRuntime(
            provider=_intent_decompose_provider,
            heuristic_decompose=intent_contract.decompose_goal_intent,
            capability_entries=_intent_catalog_capability_entries,
            capability_ids=_intent_catalog_capability_ids,
            normalize_user_id=lambda raw_user_id: _semantic_normalize_text(raw_user_id, max_len=120)
            or _semantic_default_user_id(),
            retrieve_workflow_hints=lambda session, goal_text, normalized_user_id, limit: (
                _retrieve_intent_workflow_hints(
                    session,
                    goal=goal_text,
                    user_id=normalized_user_id,
                    limit=limit,
                )
            ),
            semantic_goal_capability_hints=lambda goal_text, allowed_capability_catalog, limit: (
                _semantic_goal_capability_hints(
                    goal=goal_text,
                    allowed_capability_catalog=allowed_capability_catalog,
                    limit=limit,
                )
            ),
            llm_decompose=_llm_decompose_goal_intent,
            annotate_graph_summary_defaults=lambda graph: _annotate_graph_summary_defaults(
                graph,
                has_interaction_summaries=bool(interaction_summaries),
                allowed_capability_ids={
                    str(entry.get("id") or "").strip()
                    for entry in _intent_catalog_capability_entries()
                    if str(entry.get("id") or "").strip()
                }
                or _intent_catalog_capability_ids(),
            ),
            apply_supported_fact_filter=_apply_supported_fact_filter,
            record_metrics=lambda graph, result, has_interaction_summaries: (
                _record_intent_decompose_metrics(
                    graph=graph,
                    result=result,
                    has_interaction_summaries=has_interaction_summaries,
                )
            ),
            on_llm_failure=_on_intent_decompose_llm_failure,
        ),
    )


def _llm_capability_recommendations(
    *,
    goal: str,
    context: dict[str, Any],
    capabilities: list[dict[str, Any]],
    draft_nodes: list[dict[str, str]],
    max_results: int,
    provider: LLMProvider,
) -> list[dict[str, Any]]:
    catalog_lines = []
    for item in capabilities:
        capability_id = str(item.get("id") or "").strip()
        if not capability_id:
            continue
        required_inputs = [entry for entry in item.get("required_inputs", []) if isinstance(entry, str)]
        description = str(item.get("description") or "").strip()
        catalog_lines.append(
            f"- {capability_id} | required={required_inputs} | description={description[:180]}"
        )
    draft_lines = [
        f"- {node.get('capability_id','')} output={node.get('output_path','') or _infer_capability_output_path(node.get('capability_id',''))}"
        for node in draft_nodes
    ]
    prompt = (
        "You are recommending the next capability for a workflow composer.\n"
        "Return ONLY JSON with this shape:\n"
        '{ "recommendations": [ { "id": "capability.id", "reason": "short reason", "confidence": 0.0 } ] }\n'
        f"Select up to {max_results} capability IDs from the allowed catalog. Prefer compatibility with current chain and required inputs.\n"
        f"Goal:\n{goal}\n\n"
        f"Context keys available:\n{sorted(context.keys())}\n\n"
        f"Current draft chain:\n{draft_lines if draft_lines else ['(empty)']}\n\n"
        "Allowed capability catalog:\n"
        + "\n".join(catalog_lines)
    )
    parsed = provider.generate_request_json_object(
        LLMRequest(
            prompt=prompt,
            metadata={
                "component": "api",
                "operation": "capability_recommendations",
                "goal_len": len(goal),
                "context_keys": len(context),
                "capability_count": len(capabilities),
                "draft_node_count": len(draft_nodes),
                "max_results": max_results,
            },
        )
    )
    recs = parsed.get("recommendations")
    if not isinstance(recs, list):
        raise ValueError("llm_recommendation_missing_recommendations")

    allowed = {str(item.get("id") or "").strip() for item in capabilities}
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in recs:
        if not isinstance(raw, dict):
            continue
        capability_id = str(raw.get("id") or "").strip()
        if not capability_id or capability_id not in allowed or capability_id in seen:
            continue
        seen.add(capability_id)
        reason = str(raw.get("reason") or "").strip() or "recommended by llm"
        confidence_raw = raw.get("confidence", 0.65)
        try:
            confidence = float(confidence_raw)
        except Exception:  # noqa: BLE001
            confidence = 0.65
        confidence = max(0.01, min(0.99, confidence))
        normalized.append(
            {
                "id": capability_id,
                "reason": reason,
                "confidence": round(confidence, 3),
                "score": int(round(confidence * 100)),
            }
        )
        if len(normalized) >= max_results:
            break
    if not normalized:
        raise ValueError("llm_recommendation_empty")
    return normalized


def _split_reference_path(path: str) -> list[str]:
    return [segment.strip() for segment in path.split(".") if segment.strip()]


def _composer_default_task_name(capability_id: str, index: int) -> str:
    cleaned = "".join(ch if ch.isalnum() else " " for ch in capability_id).strip()
    if not cleaned:
        return f"Step{index + 1}"
    parts = [segment for segment in cleaned.split() if segment]
    if not parts:
        return f"Step{index + 1}"
    return "".join(segment[:1].upper() + segment[1:] for segment in parts)


_COMPOSER_CONTROL_KINDS = {"if", "if_else", "switch", "parallel"}
_COMPOSER_EXPRESSION_ROOTS = ("context.", "workflow.input.", "workflow.variable.")


def _composer_control_kind(raw_node: Mapping[str, Any]) -> str:
    control_kind = str(raw_node.get("controlKind") or raw_node.get("control_kind") or "").strip().lower()
    if control_kind in _COMPOSER_CONTROL_KINDS:
        return control_kind
    capability_id = str(raw_node.get("capabilityId") or raw_node.get("capability_id") or "").strip().lower()
    if capability_id.startswith("studio.control."):
        suffix = capability_id.rsplit(".", 1)[-1].strip()
        if suffix in _COMPOSER_CONTROL_KINDS:
            return suffix
    return ""


def _composer_is_control_node(raw_node: Mapping[str, Any]) -> bool:
    node_kind = str(raw_node.get("nodeKind") or raw_node.get("node_kind") or "").strip().lower()
    return node_kind == "control" or bool(_composer_control_kind(raw_node))


def _composer_control_config(raw_node: Mapping[str, Any]) -> Mapping[str, Any]:
    value = raw_node.get("controlConfig")
    if not isinstance(value, Mapping):
        value = raw_node.get("control_config")
    if isinstance(value, Mapping):
        return value
    return {}


def _validate_composer_control_node(
    *,
    node_id: str,
    task_name: str,
    control_kind: str,
    control_config: Mapping[str, Any],
    workflow_input_keys: set[str] | None = None,
    workflow_variable_keys: set[str] | None = None,
) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    expression = str(control_config.get("expression") or "").strip()
    parallel_mode = str(control_config.get("parallelMode") or control_config.get("parallel_mode") or "fan_out").strip().lower()
    if control_kind in {"if", "if_else", "switch"} and not expression:
        diagnostics.append(
            {
                "code": "draft.control_expression_missing",
                "node_id": node_id,
                "field": "expression",
                "message": "Control-flow node requires a non-empty expression.",
            }
        )
    if control_kind in {"if", "if_else"} and expression and not _composer_if_expression_supported(
        expression,
        workflow_input_keys=workflow_input_keys,
        workflow_variable_keys=workflow_variable_keys,
    ):
        diagnostics.append(
            {
                "code": "draft.control_expression_unsupported",
                "node_id": node_id,
                "field": "expression",
                "message": (
                    "Conditional control-flow supports only context.*, workflow.input.*, "
                    "and workflow.variable.* expressions, with declared workflow keys."
                ),
            }
        )
    if control_kind == "parallel" and parallel_mode not in {"fan_out", "fan_in"}:
        diagnostics.append(
            {
                "code": "draft.control_parallel_mode_invalid",
                "node_id": node_id,
                "field": "parallelMode",
                "message": "Parallel control-flow node requires parallelMode of fan_out or fan_in.",
            }
        )
    if control_kind == "switch":
        diagnostics.append(
            {
                "code": "draft.control_flow_unsupported",
                "node_id": node_id,
                "message": (
                    f"Control-flow node '{task_name or control_kind}' ({control_kind}) is not compiled into plans yet."
                ),
            }
        )
    if control_kind == "switch":
        raw_cases = control_config.get("switchCases")
        if not isinstance(raw_cases, list):
            raw_cases = control_config.get("switch_cases")
        cases = raw_cases if isinstance(raw_cases, list) else []
        if not cases:
            diagnostics.append(
                {
                    "code": "draft.control_switch_cases_missing",
                    "node_id": node_id,
                    "field": "switchCases",
                    "message": "Switch control-flow node requires at least one case.",
                }
            )
        seen_labels: set[str] = set()
        for index, raw_case in enumerate(cases):
            if not isinstance(raw_case, Mapping):
                diagnostics.append(
                    {
                        "code": "draft.control_switch_case_invalid",
                        "node_id": node_id,
                        "field": f"switchCases[{index}]",
                        "message": "Switch case must be an object.",
                    }
                )
                continue
            label = str(raw_case.get("label") or "").strip()
            match = str(raw_case.get("match") or "").strip()
            if not label:
                diagnostics.append(
                    {
                        "code": "draft.control_switch_case_label_missing",
                        "node_id": node_id,
                        "field": f"switchCases[{index}].label",
                        "message": "Switch case label is required.",
                    }
                )
            elif label in seen_labels:
                diagnostics.append(
                    {
                        "code": "draft.control_switch_case_label_duplicate",
                        "node_id": node_id,
                        "field": f"switchCases[{index}].label",
                        "message": f"Duplicate switch case label '{label}'.",
                    }
                )
            else:
                seen_labels.add(label)
            if not match:
                diagnostics.append(
                    {
                        "code": "draft.control_switch_case_match_missing",
                        "node_id": node_id,
                        "field": f"switchCases[{index}].match",
                        "message": "Switch case match value is required.",
                    }
                )
    return diagnostics

def _composer_expression_operand(
    token: str,
) -> tuple[str, str] | None:
    normalized = token.strip()
    for prefix, root in (
        ("context.", "context"),
        ("workflow.input.", "workflow_input"),
        ("workflow.variable.", "workflow_variable"),
    ):
        if not normalized.startswith(prefix):
            continue
        remainder = normalized[len(prefix) :].strip()
        if not remainder:
            return None
        key = remainder.split(".", 1)[0].strip()
        if not key:
            return None
        return root, key
    return None


def _composer_expression_operand_supported(
    token: str,
    *,
    workflow_input_keys: set[str] | None = None,
    workflow_variable_keys: set[str] | None = None,
) -> bool:
    operand = _composer_expression_operand(token)
    if operand is None:
        return False
    root, key = operand
    if root == "workflow_input":
        return bool(workflow_input_keys) and key in workflow_input_keys
    if root == "workflow_variable":
        return bool(workflow_variable_keys) and key in workflow_variable_keys
    return True


def _composer_if_expression_supported(
    expression: str,
    *,
    workflow_input_keys: set[str] | None = None,
    workflow_variable_keys: set[str] | None = None,
) -> bool:
    normalized = expression.strip()
    if not normalized:
        return False
    if "==" in normalized or "!=" in normalized:
        left, right = normalized.split("==" if "==" in normalized else "!=", 1)
        left = left.strip()
        right = right.strip()
        if not _composer_expression_operand_supported(
            left,
            workflow_input_keys=workflow_input_keys,
            workflow_variable_keys=workflow_variable_keys,
        ):
            return False
        if not right:
            return False
        if right.startswith(_COMPOSER_EXPRESSION_ROOTS):
            return _composer_expression_operand_supported(
                right,
                workflow_input_keys=workflow_input_keys,
                workflow_variable_keys=workflow_variable_keys,
            )
        return True
    return _composer_expression_operand_supported(
        normalized,
        workflow_input_keys=workflow_input_keys,
        workflow_variable_keys=workflow_variable_keys,
    )


_WORKFLOW_INTERFACE_VALUE_TYPES = {"string", "number", "boolean", "object", "array"}


def _get_nested_value(root: Mapping[str, Any] | None, segments: Sequence[str]) -> Any:
    cursor: Any = root
    for segment in segments:
        if not isinstance(cursor, Mapping):
            return None
        cursor = cursor.get(segment)
    return cursor


def _workflow_placeholder_value(value_type: str, *, key: str = "value") -> Any:
    normalized = str(value_type or "string").strip().lower() or "string"
    if normalized == "number":
        return 0
    if normalized == "boolean":
        return False
    if normalized == "object":
        return {}
    if normalized == "array":
        return []
    return f"<{key}>"


def _parse_workflow_typed_value(raw: Any, value_type: str) -> tuple[Any, str | None]:
    normalized = str(value_type or "string").strip().lower() or "string"
    if raw is None:
        return None, None
    secret_ref = execution_contracts.parse_secret_ref(raw)
    if secret_ref is not None:
        return secret_ref.model_dump(), None
    if normalized == "string":
        if isinstance(raw, str):
            return raw, None
        if isinstance(raw, (dict, list)):
            return json.dumps(raw, ensure_ascii=True), None
        return str(raw), None

    parsed = raw
    if isinstance(raw, str):
        trimmed = raw.strip()
        if not trimmed:
            return None, None
        if normalized == "boolean":
            lowered = trimmed.lower()
            if lowered in {"true", "false"}:
                return lowered == "true", None
        try:
            parsed = json.loads(trimmed)
        except Exception:
            if normalized == "number":
                try:
                    return int(trimmed), None
                except ValueError:
                    try:
                        return float(trimmed), None
                    except ValueError:
                        return None, f"Expected a number value, received '{trimmed}'."
            return None, f"Expected valid JSON for {normalized} value."

    if normalized == "number":
        if isinstance(parsed, bool) or not isinstance(parsed, (int, float)):
            return None, "Expected a number value."
        return parsed, None
    if normalized == "boolean":
        if not isinstance(parsed, bool):
            return None, "Expected a boolean value."
        return parsed, None
    if normalized == "object":
        if not isinstance(parsed, Mapping):
            return None, "Expected an object value."
        return dict(parsed), None
    if normalized == "array":
        if not isinstance(parsed, list):
            return None, "Expected an array value."
        return list(parsed), None
    return parsed, None


def _normalize_workflow_binding(
    raw_binding: Any,
    *,
    field: str,
    allowed_kinds: set[str],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    diagnostics: list[dict[str, Any]] = []
    if raw_binding is None:
        return None, diagnostics
    if not isinstance(raw_binding, Mapping):
        diagnostics.append(
            {
                "code": "draft.workflow_binding_invalid",
                "field": field,
                "message": "Workflow binding must be an object.",
            }
        )
        return None, diagnostics
    kind = str(raw_binding.get("kind") or raw_binding.get("mode") or "").strip()
    if kind not in allowed_kinds:
        diagnostics.append(
            {
                "code": "draft.workflow_binding_kind_unknown",
                "field": field,
                "message": f"Unsupported workflow binding kind: {kind or '<empty>'}",
            }
        )
        return None, diagnostics
    if kind == "literal":
        return {"kind": "literal", "value": raw_binding.get("value")}, diagnostics
    if kind == "context":
        path = str(raw_binding.get("path") or raw_binding.get("contextPath") or "").strip()
        if not path:
            diagnostics.append(
                {
                    "code": "draft.workflow_binding_context_path_missing",
                    "field": field,
                    "message": "Workflow context binding requires a path.",
                }
            )
            return None, diagnostics
        return {"kind": "context", "path": path}, diagnostics
    if kind == "memory":
        scope = str(raw_binding.get("scope") or "job").strip() or "job"
        name = str(raw_binding.get("name") or "").strip()
        if not name:
            diagnostics.append(
                {
                    "code": "draft.workflow_binding_memory_name_missing",
                    "field": field,
                    "message": "Workflow memory binding requires a memory name.",
                }
            )
            return None, diagnostics
        payload: dict[str, Any] = {"kind": "memory", "scope": scope, "name": name}
        key = raw_binding.get("key")
        if isinstance(key, str) and key.strip():
            payload["key"] = key.strip()
        user_id = _semantic_normalize_text(
            raw_binding.get("userId") or raw_binding.get("user_id"),
            max_len=120,
        )
        if user_id:
            payload["user_id"] = user_id
        return payload, diagnostics
    if kind == "secret":
        secret_name = str(
            raw_binding.get("secretName") or raw_binding.get("secret_name") or ""
        ).strip()
        if not secret_name:
            diagnostics.append(
                {
                    "code": "draft.workflow_binding_secret_name_missing",
                    "field": field,
                    "message": "Workflow secret binding requires a secret name.",
                }
            )
            return None, diagnostics
        return {"kind": "secret", "secret_name": secret_name}, diagnostics
    if kind == "workflow_input":
        input_key = str(raw_binding.get("inputKey") or raw_binding.get("input_key") or "").strip()
        if not input_key:
            diagnostics.append(
                {
                    "code": "draft.workflow_binding_input_key_missing",
                    "field": field,
                    "message": "Workflow-input binding requires an input key.",
                }
            )
            return None, diagnostics
        return {"kind": "workflow_input", "input_key": input_key}, diagnostics
    if kind == "workflow_variable":
        variable_key = str(
            raw_binding.get("variableKey") or raw_binding.get("variable_key") or ""
        ).strip()
        if not variable_key:
            diagnostics.append(
                {
                    "code": "draft.workflow_binding_variable_key_missing",
                    "field": field,
                    "message": "Workflow-variable binding requires a variable key.",
                }
            )
            return None, diagnostics
        return {"kind": "workflow_variable", "variable_key": variable_key}, diagnostics
    if kind == "step_output":
        source_node_id = str(
            raw_binding.get("sourceNodeId") or raw_binding.get("nodeId") or ""
        ).strip()
        source_path = str(raw_binding.get("sourcePath") or raw_binding.get("path") or "").strip()
        if not source_node_id or not source_path:
            diagnostics.append(
                {
                    "code": "draft.workflow_binding_step_output_missing",
                    "field": field,
                    "message": "Workflow step-output binding requires source node id and source path.",
                }
            )
            return None, diagnostics
        return {
            "kind": "step_output",
            "source_node_id": source_node_id,
            "source_path": source_path,
        }, diagnostics
    return {"kind": kind}, diagnostics


def _coerce_workflow_interface(
    value: Any,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]], list[dict[str, Any]]]:
    normalized: dict[str, list[dict[str, Any]]] = {"inputs": [], "variables": [], "outputs": []}
    diagnostics_errors: list[dict[str, Any]] = []
    diagnostics_warnings: list[dict[str, Any]] = []
    if not isinstance(value, Mapping):
        return normalized, diagnostics_errors, diagnostics_warnings

    raw_inputs = value.get("inputs")
    seen_input_keys: set[str] = set()
    if isinstance(raw_inputs, list):
        for index, raw_input in enumerate(raw_inputs):
            field_prefix = f"workflowInterface.inputs[{index}]"
            if not isinstance(raw_input, Mapping):
                diagnostics_errors.append(
                    {
                        "code": "draft.workflow_input_invalid",
                        "field": field_prefix,
                        "message": "Workflow input definition must be an object.",
                    }
                )
                continue
            key = str(raw_input.get("key") or "").strip()
            if not key:
                diagnostics_errors.append(
                    {
                        "code": "draft.workflow_input_key_missing",
                        "field": f"{field_prefix}.key",
                        "message": "Workflow input key is required.",
                    }
                )
                continue
            if key in seen_input_keys:
                diagnostics_errors.append(
                    {
                        "code": "draft.workflow_input_key_duplicate",
                        "field": f"{field_prefix}.key",
                        "message": f"Duplicate workflow input key '{key}'.",
                    }
                )
                continue
            seen_input_keys.add(key)
            value_type = str(raw_input.get("valueType") or raw_input.get("value_type") or "string").strip().lower() or "string"
            if value_type not in _WORKFLOW_INTERFACE_VALUE_TYPES:
                diagnostics_errors.append(
                    {
                        "code": "draft.workflow_input_value_type_invalid",
                        "field": f"{field_prefix}.valueType",
                        "message": f"Unsupported workflow input value type '{value_type}'.",
                    }
                )
                value_type = "string"
            binding, binding_errors = _normalize_workflow_binding(
                raw_input.get("binding"),
                field=f"{field_prefix}.binding",
                allowed_kinds={"literal", "context", "memory", "secret"},
            )
            diagnostics_errors.extend(binding_errors)
            normalized["inputs"].append(
                {
                    "id": str(raw_input.get("id") or "").strip() or f"workflow-input-{index + 1}",
                    "key": key,
                    "label": str(raw_input.get("label") or key).strip() or key,
                    "value_type": value_type,
                    "required": bool(raw_input.get("required")),
                    "description": str(raw_input.get("description") or "").strip(),
                    "default_value": raw_input.get("defaultValue"),
                    "binding": binding,
                }
            )

    raw_variables = value.get("variables")
    seen_variable_keys: set[str] = set()
    if isinstance(raw_variables, list):
        for index, raw_variable in enumerate(raw_variables):
            field_prefix = f"workflowInterface.variables[{index}]"
            if not isinstance(raw_variable, Mapping):
                diagnostics_errors.append(
                    {
                        "code": "draft.workflow_variable_invalid",
                        "field": field_prefix,
                        "message": "Workflow variable definition must be an object.",
                    }
                )
                continue
            key = str(raw_variable.get("key") or "").strip()
            if not key:
                diagnostics_errors.append(
                    {
                        "code": "draft.workflow_variable_key_missing",
                        "field": f"{field_prefix}.key",
                        "message": "Workflow variable key is required.",
                    }
                )
                continue
            if key in seen_variable_keys:
                diagnostics_errors.append(
                    {
                        "code": "draft.workflow_variable_key_duplicate",
                        "field": f"{field_prefix}.key",
                        "message": f"Duplicate workflow variable key '{key}'.",
                    }
                )
                continue
            seen_variable_keys.add(key)
            binding, binding_errors = _normalize_workflow_binding(
                raw_variable.get("binding"),
                field=f"{field_prefix}.binding",
                allowed_kinds={"literal", "context", "memory", "secret", "workflow_input"},
            )
            diagnostics_errors.extend(binding_errors)
            normalized["variables"].append(
                {
                    "id": str(raw_variable.get("id") or "").strip()
                    or f"workflow-variable-{index + 1}",
                    "key": key,
                    "description": str(raw_variable.get("description") or "").strip(),
                    "binding": binding,
                }
            )

    raw_outputs = value.get("outputs")
    seen_output_keys: set[str] = set()
    if isinstance(raw_outputs, list):
        for index, raw_output in enumerate(raw_outputs):
            field_prefix = f"workflowInterface.outputs[{index}]"
            if not isinstance(raw_output, Mapping):
                diagnostics_errors.append(
                    {
                        "code": "draft.workflow_output_invalid",
                        "field": field_prefix,
                        "message": "Workflow output definition must be an object.",
                    }
                )
                continue
            key = str(raw_output.get("key") or "").strip()
            if not key:
                diagnostics_errors.append(
                    {
                        "code": "draft.workflow_output_key_missing",
                        "field": f"{field_prefix}.key",
                        "message": "Workflow output key is required.",
                    }
                )
                continue
            if key in seen_output_keys:
                diagnostics_errors.append(
                    {
                        "code": "draft.workflow_output_key_duplicate",
                        "field": f"{field_prefix}.key",
                        "message": f"Duplicate workflow output key '{key}'.",
                    }
                )
                continue
            seen_output_keys.add(key)
            binding, binding_errors = _normalize_workflow_binding(
                raw_output.get("binding"),
                field=f"{field_prefix}.binding",
                allowed_kinds={
                    "literal",
                    "context",
                    "workflow_input",
                    "workflow_variable",
                    "step_output",
                },
            )
            diagnostics_errors.extend(binding_errors)
            normalized["outputs"].append(
                {
                    "id": str(raw_output.get("id") or "").strip() or f"workflow-output-{index + 1}",
                    "key": key,
                    "label": str(raw_output.get("label") or key).strip() or key,
                    "description": str(raw_output.get("description") or "").strip(),
                    "binding": binding,
                }
            )

    return normalized, diagnostics_errors, diagnostics_warnings


def _resolve_workflow_memory_binding(
    db: Session | None,
    binding: Mapping[str, Any],
    *,
    runtime_context: Mapping[str, Any] | None,
) -> tuple[Any, str | None]:
    if db is None:
        return None, "Memory bindings require a database session."
    name = str(binding.get("name") or "").strip()
    scope_raw = str(binding.get("scope") or "job").strip() or "job"
    try:
        scope = models.MemoryScope(scope_raw)
    except ValueError:
        return None, f"Unsupported memory scope '{scope_raw}'."
    query = models.MemoryQuery(
        name=name,
        scope=scope,
        key=str(binding.get("key") or "").strip() or None,
        job_id=str(runtime_context.get("job_id") or "").strip() or None
        if isinstance(runtime_context, Mapping)
        else None,
        user_id=_semantic_normalize_text(
            binding.get("user_id") or _semantic_user_id_from_context(runtime_context),
            max_len=120,
        )
        or None,
        project_id=str(runtime_context.get("project_id") or "").strip() or None
        if isinstance(runtime_context, Mapping)
        else None,
        limit=1,
    )
    try:
        entries = memory_store.read_memory(db, query)
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)
    if not entries:
        return None, "Memory entry not found."
    return entries[0].payload, None


def _resolve_workflow_binding_value(
    binding: Mapping[str, Any] | None,
    *,
    db: Session | None,
    runtime_context: Mapping[str, Any] | None,
    resolved_inputs: Mapping[str, Any] | None = None,
    resolved_variables: Mapping[str, Any] | None = None,
    preview: bool = False,
    placeholder_type: str = "string",
    placeholder_key: str = "value",
) -> tuple[Any, str | None]:
    if not isinstance(binding, Mapping):
        return None, None
    kind = str(binding.get("kind") or "").strip()
    if kind == "literal":
        return binding.get("value"), None
    if kind == "context":
        segments = _split_reference_path(str(binding.get("path") or "").strip())
        if not segments:
            return None, "Workflow context binding requires a path."
        value = _get_nested_value(runtime_context, segments)
        if value is None and preview:
            return _workflow_placeholder_value(placeholder_type, key=placeholder_key), None
        return value, None if value is not None else "Context value not found."
    if kind == "memory":
        if preview:
            return _workflow_placeholder_value(placeholder_type, key=placeholder_key), None
        return _resolve_workflow_memory_binding(db, binding, runtime_context=runtime_context)
    if kind == "secret":
        secret_name = str(binding.get("secret_name") or "").strip()
        if preview:
            return _workflow_placeholder_value(placeholder_type, key=placeholder_key), None
        return execution_contracts.build_secret_ref(secret_name), None
    if kind == "workflow_input":
        input_key = str(binding.get("input_key") or "").strip()
        value = resolved_inputs.get(input_key) if isinstance(resolved_inputs, Mapping) else None
        if value is None and preview:
            return _workflow_placeholder_value(placeholder_type, key=input_key or placeholder_key), None
        return value, None if value is not None else f"Workflow input '{input_key}' is not resolved."
    if kind == "workflow_variable":
        variable_key = str(binding.get("variable_key") or "").strip()
        value = (
            resolved_variables.get(variable_key)
            if isinstance(resolved_variables, Mapping)
            else None
        )
        if value is None and preview:
            return _workflow_placeholder_value(
                placeholder_type,
                key=variable_key or placeholder_key,
            ), None
        return (
            value,
            None if value is not None else f"Workflow variable '{variable_key}' is not resolved.",
        )
    if kind == "step_output":
        if preview:
            return _workflow_placeholder_value(placeholder_type, key=placeholder_key), None
        return None, "Workflow outputs from step outputs are only available after execution."
    return None, f"Unsupported workflow binding kind '{kind}'."


def _build_workflow_interface_runtime_context(
    workflow_interface: Mapping[str, Any] | None,
    *,
    base_context: Mapping[str, Any] | None,
    db: Session | None = None,
    explicit_inputs: Mapping[str, Any] | None = None,
    preview: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    runtime_context = _coerce_context_object(base_context)
    diagnostics_errors: list[dict[str, Any]] = []
    workflow_payload = (
        dict(runtime_context.get("workflow"))
        if isinstance(runtime_context.get("workflow"), Mapping)
        else {}
    )
    existing_inputs = (
        dict(workflow_payload.get("inputs"))
        if isinstance(workflow_payload.get("inputs"), Mapping)
        else {}
    )
    existing_variables = (
        dict(workflow_payload.get("variables"))
        if isinstance(workflow_payload.get("variables"), Mapping)
        else {}
    )
    resolved_inputs: dict[str, Any] = dict(existing_inputs)
    input_defs = (
        workflow_interface.get("inputs", [])
        if isinstance(workflow_interface, Mapping)
        else []
    )
    variable_defs = (
        workflow_interface.get("variables", [])
        if isinstance(workflow_interface, Mapping)
        else []
    )
    normalized_explicit_inputs = (
        dict(explicit_inputs) if isinstance(explicit_inputs, Mapping) else {}
    )

    for definition in input_defs:
        if not isinstance(definition, Mapping):
            continue
        key = str(definition.get("key") or "").strip()
        if not key:
            continue
        value_type = str(definition.get("value_type") or "string").strip().lower() or "string"
        candidate = normalized_explicit_inputs.get(key, resolved_inputs.get(key))
        value = None
        error_message: str | None = None
        if key in normalized_explicit_inputs or key in resolved_inputs:
            value, error_message = _parse_workflow_typed_value(candidate, value_type)
        if value is None and error_message is None:
            binding_value, binding_error = _resolve_workflow_binding_value(
                definition.get("binding"),
                db=db,
                runtime_context=runtime_context,
                preview=preview,
                placeholder_type=value_type,
                placeholder_key=key,
            )
            if binding_error is None and binding_value is not None:
                value, error_message = _parse_workflow_typed_value(binding_value, value_type)
            elif binding_error and not preview:
                error_message = binding_error
        if value is None and error_message is None:
            value, error_message = _parse_workflow_typed_value(
                definition.get("default_value"),
                value_type,
            )
        if value is None and preview:
            value = _workflow_placeholder_value(value_type, key=key)
        if value is None and bool(definition.get("required")):
            diagnostics_errors.append(
                {
                    "code": "draft.workflow_input_value_missing",
                    "field": f"workflowInterface.inputs.{key}",
                    "message": error_message or f"Workflow input '{key}' is required.",
                }
            )
        elif error_message is not None and not preview:
            diagnostics_errors.append(
                {
                    "code": "draft.workflow_input_value_invalid",
                    "field": f"workflowInterface.inputs.{key}",
                    "message": f"Workflow input '{key}' is invalid: {error_message}",
                }
            )
        resolved_inputs[key] = value

    resolved_variables: dict[str, Any] = dict(existing_variables)
    for definition in variable_defs:
        if not isinstance(definition, Mapping):
            continue
        key = str(definition.get("key") or "").strip()
        if not key:
            continue
        value, error_message = _resolve_workflow_binding_value(
            definition.get("binding"),
            db=db,
            runtime_context=runtime_context,
            resolved_inputs=resolved_inputs,
            resolved_variables=resolved_variables,
            preview=preview,
            placeholder_key=key,
        )
        if value is None and preview:
            value = _workflow_placeholder_value("string", key=key)
        if error_message is not None and not preview:
            diagnostics_errors.append(
                {
                    "code": "draft.workflow_variable_value_invalid",
                    "field": f"workflowInterface.variables.{key}",
                    "message": f"Workflow variable '{key}' could not be resolved: {error_message}",
                }
            )
        resolved_variables[key] = value

    workflow_payload["inputs"] = resolved_inputs
    workflow_payload["variables"] = resolved_variables
    if isinstance(workflow_interface, Mapping):
        workflow_payload["interface"] = {
            "inputs": list(input_defs),
            "variables": list(variable_defs),
            "outputs": list(workflow_interface.get("outputs", []))
            if isinstance(workflow_interface.get("outputs"), list)
            else [],
        }
    runtime_context["workflow"] = workflow_payload
    return runtime_context, diagnostics_errors


def _build_plan_from_composer_draft(
    draft: dict[str, Any],
    *,
    goal_text: str = "",
    job_context: Mapping[str, Any] | None = None,
) -> tuple[models.PlanCreate | None, list[dict[str, Any]], list[dict[str, Any]]]:
    diagnostics_errors: list[dict[str, Any]] = []
    diagnostics_warnings: list[dict[str, Any]] = []
    workflow_interface, workflow_interface_errors, workflow_interface_warnings = (
        _coerce_workflow_interface(
            draft.get("workflowInterface")
            if "workflowInterface" in draft
            else draft.get("workflow_interface")
        )
    )
    workflow_input_keys = {
        str(definition.get("key") or "").strip()
        for definition in workflow_interface.get("inputs", [])
        if isinstance(definition, Mapping) and str(definition.get("key") or "").strip()
    }
    workflow_variable_keys = {
        str(definition.get("key") or "").strip()
        for definition in workflow_interface.get("variables", [])
        if isinstance(definition, Mapping) and str(definition.get("key") or "").strip()
    }
    diagnostics_errors.extend(workflow_interface_errors)
    diagnostics_warnings.extend(workflow_interface_warnings)
    preview_job_context, workflow_interface_context_errors = _build_workflow_interface_runtime_context(
        workflow_interface,
        base_context=job_context,
        preview=True,
    )
    diagnostics_errors.extend(workflow_interface_context_errors)
    job_context = preview_job_context
    raw_nodes = draft.get("nodes")
    if not isinstance(raw_nodes, list) or not raw_nodes:
        diagnostics_errors.append(
            {"code": "draft.nodes_missing", "message": "Composer draft must include non-empty nodes."}
        )
        return None, diagnostics_errors, diagnostics_warnings

    registry = capability_registry.load_capability_registry()
    canonical_nodes: list[dict[str, Any]] = []
    node_by_id: dict[str, dict[str, Any]] = {}
    used_task_names: set[str] = set()
    tool_intent_values = {member.value for member in models.ToolIntent}
    has_unsupported_control_nodes = False

    for index, raw_node in enumerate(raw_nodes):
        if not isinstance(raw_node, dict):
            diagnostics_errors.append(
                {
                    "code": "draft.node_invalid",
                    "message": f"Node at index {index + 1} must be an object.",
                }
            )
            continue
        node_id = str(raw_node.get("id") or f"node-{index + 1}").strip()
        capability_id = str(
            raw_node.get("capabilityId") or raw_node.get("capability_id") or ""
        ).strip()
        task_name = str(raw_node.get("taskName") or raw_node.get("task_name") or "").strip()
        is_control_node = _composer_is_control_node(raw_node)
        control_kind = _composer_control_kind(raw_node)
        control_config = _composer_control_config(raw_node)
        if not capability_id:
            diagnostics_errors.append(
                {
                    "code": "draft.capability_missing" if not is_control_node else "draft.control_kind_missing",
                    "node_id": node_id,
                    "message": "capabilityId is required." if not is_control_node else "Control node is missing control kind/capabilityId.",
                }
            )
            continue
        capability_spec = None
        if is_control_node:
            parallel_mode = str(control_config.get("parallelMode") or control_config.get("parallel_mode") or "fan_out").strip().lower()
            if control_kind not in {"if", "if_else"} and not (
                control_kind == "parallel" and parallel_mode in {"fan_out", "fan_in"}
            ):
                has_unsupported_control_nodes = True
            diagnostics_errors.extend(
                _validate_composer_control_node(
                    node_id=node_id,
                    task_name=task_name or capability_id,
                    control_kind=control_kind,
                    control_config=control_config,
                    workflow_input_keys=workflow_input_keys,
                    workflow_variable_keys=workflow_variable_keys,
                )
            )
        else:
            capability_spec = registry.get(capability_id)
        if not is_control_node and capability_spec is None:
            diagnostics_errors.append(
                {
                    "code": "draft.capability_unknown",
                    "node_id": node_id,
                    "message": f"Unknown capability: {capability_id}",
                }
            )
            continue
        if not task_name:
            task_name = _composer_default_task_name(capability_id, index)
            diagnostics_warnings.append(
                {
                    "code": "draft.task_name_defaulted",
                    "node_id": node_id,
                    "message": f"Task name was empty. Using '{task_name}'.",
                }
            )
        deduped_task_name = task_name
        suffix = 2
        while deduped_task_name in used_task_names:
            deduped_task_name = f"{task_name}_{suffix}"
            suffix += 1
        if deduped_task_name != task_name:
            diagnostics_warnings.append(
                {
                    "code": "draft.task_name_deduped",
                    "node_id": node_id,
                    "message": f"Task name '{task_name}' duplicated. Using '{deduped_task_name}'.",
                }
            )
        used_task_names.add(deduped_task_name)
        bindings = raw_node.get("bindings")
        if not isinstance(bindings, dict):
            bindings = raw_node.get("inputBindings")
        if not isinstance(bindings, dict):
            bindings = {}
        canonical = {
            "node_id": node_id,
            "task_name": deduped_task_name,
            "capability_id": capability_id,
            "capability_spec": capability_spec,
            "bindings": dict(bindings),
            "order": index,
            "intent": str(raw_node.get("intent") or "").strip(),
            "is_control": is_control_node,
            "control_kind": control_kind,
            "control_config": dict(control_config),
        }
        canonical_nodes.append(canonical)
        node_by_id[node_id] = canonical

    if not canonical_nodes:
        return None, diagnostics_errors, diagnostics_warnings

    deps_by_node_id: dict[str, set[str]] = {node["node_id"]: set() for node in canonical_nodes}
    raw_edges = draft.get("edges")
    if isinstance(raw_edges, list):
        for edge in raw_edges:
            if not isinstance(edge, dict):
                diagnostics_errors.append(
                    {"code": "draft.edge_invalid", "message": "Each edge must be an object."}
                )
                continue
            source_id = str(edge.get("fromNodeId") or edge.get("from") or "").strip()
            target_id = str(edge.get("toNodeId") or edge.get("to") or "").strip()
            if not source_id or not target_id:
                diagnostics_errors.append(
                    {
                        "code": "draft.edge_endpoints_missing",
                        "message": "Edge must include fromNodeId and toNodeId.",
                    }
                )
                continue
            if source_id not in node_by_id:
                diagnostics_errors.append(
                    {
                        "code": "draft.edge_source_unknown",
                        "message": f"Edge source not found: {source_id}",
                    }
                )
                continue
            if target_id not in node_by_id:
                diagnostics_errors.append(
                    {
                        "code": "draft.edge_target_unknown",
                        "message": f"Edge target not found: {target_id}",
                    }
                )
                continue
            if source_id == target_id:
                diagnostics_errors.append(
                    {"code": "draft.edge_self_cycle", "message": f"Self-cycle on node {source_id}."}
                )
                continue
            deps_by_node_id[target_id].add(source_id)

    children_by_node_id: dict[str, set[str]] = {node["node_id"]: set() for node in canonical_nodes}
    edge_branch_labels: dict[tuple[str, str], str] = {}
    for target_id, source_ids in deps_by_node_id.items():
        for source_id in source_ids:
            if source_id in children_by_node_id:
                children_by_node_id[source_id].add(target_id)
    if isinstance(raw_edges, list):
        for edge in raw_edges:
            if not isinstance(edge, dict):
                continue
            source_id = str(edge.get("fromNodeId") or edge.get("from") or "").strip()
            target_id = str(edge.get("toNodeId") or edge.get("to") or "").strip()
            branch_label = str(edge.get("branchLabel") or edge.get("branch_label") or "").strip()
            if source_id and target_id and branch_label:
                edge_branch_labels[(source_id, target_id)] = branch_label

    for node in canonical_nodes:
        if not node.get("is_control") or node.get("control_kind") != "parallel":
            continue
        parallel_mode = str(
            node.get("control_config", {}).get("parallelMode")
            or node.get("control_config", {}).get("parallel_mode")
            or "fan_out"
        ).strip().lower()
        outgoing_count = len(children_by_node_id.get(node["node_id"], set()))
        incoming_count = len(deps_by_node_id.get(node["node_id"], set()))
        if parallel_mode == "fan_out" and outgoing_count < 2:
            diagnostics_errors.append(
                {
                    "code": "draft.control_parallel_fan_out_targets_missing",
                    "node_id": node["node_id"],
                    "field": "edges",
                    "message": "Parallel fan_out control node requires at least two outgoing edges.",
                }
            )
        if parallel_mode == "fan_in" and incoming_count < 2:
            diagnostics_errors.append(
                {
                    "code": "draft.control_parallel_fan_in_sources_missing",
                    "node_id": node["node_id"],
                    "field": "edges",
                    "message": "Parallel fan_in control node requires at least two incoming edges.",
                }
            )
        if parallel_mode == "fan_in" and outgoing_count < 1:
            diagnostics_errors.append(
                {
                    "code": "draft.control_parallel_fan_in_target_missing",
                    "node_id": node["node_id"],
                    "field": "edges",
                    "message": "Parallel fan_in control node requires at least one outgoing edge.",
                }
            )

    def _resolve_non_control_deps(node_id: str, seen: set[str] | None = None) -> set[str]:
        resolved: set[str] = set()
        for source_id in deps_by_node_id.get(node_id, set()):
            if source_id == node_id:
                continue
            source_node = node_by_id.get(source_id)
            if source_node is None:
                continue
            if not source_node.get("is_control"):
                resolved.add(source_id)
                continue
            next_seen = set(seen or set())
            if source_id in next_seen:
                continue
            next_seen.add(source_id)
            resolved.update(_resolve_non_control_deps(source_id, next_seen))
        return resolved

    for node in canonical_nodes:
        node["execution_gate"] = None
    for node in canonical_nodes:
        if not node.get("is_control") or node.get("control_kind") != "if":
            continue
        expression = str(node.get("control_config", {}).get("expression") or "").strip()
        if not expression or not _composer_if_expression_supported(
            expression,
            workflow_input_keys=workflow_input_keys,
            workflow_variable_keys=workflow_variable_keys,
        ):
            continue
        queue = list(children_by_node_id.get(node["node_id"], set()))
        visited: set[str] = set()
        while queue:
            candidate_id = queue.pop(0)
            if candidate_id in visited:
                continue
            visited.add(candidate_id)
            candidate = node_by_id.get(candidate_id)
            if candidate is None:
                continue
            if not candidate.get("is_control"):
                candidate["execution_gate"] = {"expression": expression}
            queue.extend(children_by_node_id.get(candidate_id, set()))

    for node in canonical_nodes:
        if not node.get("is_control") or node.get("control_kind") != "if_else":
            continue
        expression = str(node.get("control_config", {}).get("expression") or "").strip()
        if not expression or not _composer_if_expression_supported(
            expression,
            workflow_input_keys=workflow_input_keys,
            workflow_variable_keys=workflow_variable_keys,
        ):
            continue
        config = node.get("control_config", {})
        true_label = str(config.get("trueLabel") or "true").strip().lower()
        false_label = str(config.get("falseLabel") or "false").strip().lower()
        outgoing = sorted(children_by_node_id.get(node["node_id"], set()))
        true_roots: set[str] = set()
        false_roots: set[str] = set()
        for target_id in outgoing:
            branch_label = edge_branch_labels.get((node["node_id"], target_id), "").strip().lower()
            if branch_label == true_label:
                true_roots.add(target_id)
            elif branch_label == false_label:
                false_roots.add(target_id)
        if not true_roots:
            diagnostics_errors.append(
                {
                    "code": "draft.control_if_else_true_branch_missing",
                    "node_id": node["node_id"],
                    "field": "edges",
                    "message": "If / Else control node requires an outgoing edge labeled for the true branch.",
                }
            )
        if not false_roots:
            diagnostics_errors.append(
                {
                    "code": "draft.control_if_else_false_branch_missing",
                    "node_id": node["node_id"],
                    "field": "edges",
                    "message": "If / Else control node requires an outgoing edge labeled for the false branch.",
                }
            )

        def _descendants(roots: set[str]) -> set[str]:
            seen: set[str] = set()
            queue = list(roots)
            while queue:
                candidate_id = queue.pop(0)
                if candidate_id in seen:
                    continue
                seen.add(candidate_id)
                queue.extend(sorted(children_by_node_id.get(candidate_id, set())))
            return seen

        true_descendants = _descendants(true_roots)
        false_descendants = _descendants(false_roots)
        for candidate_id in sorted(true_descendants - false_descendants):
            candidate = node_by_id.get(candidate_id)
            if candidate is not None and not candidate.get("is_control"):
                candidate["execution_gate"] = {"expression": expression}
        for candidate_id in sorted(false_descendants - true_descendants):
            candidate = node_by_id.get(candidate_id)
            if candidate is not None and not candidate.get("is_control"):
                candidate["execution_gate"] = {"expression": expression, "negate": True}

    tasks_payload: list[dict[str, Any]] = []
    for node in canonical_nodes:
        if node.get("is_control"):
            continue
        node_id = node["node_id"]
        capability_id = node["capability_id"]
        capability_spec = node["capability_spec"]
        bindings = node["bindings"]
        tool_input_payload: dict[str, Any] = {}

        for field, raw_binding in bindings.items():
            if not isinstance(field, str) or not field.strip():
                continue
            field_name = field.strip()
            if not isinstance(raw_binding, dict):
                diagnostics_errors.append(
                    {
                        "code": "draft.binding_invalid",
                        "node_id": node_id,
                        "field": field_name,
                        "message": "Binding must be an object.",
                    }
                )
                continue
            binding_kind = str(raw_binding.get("kind") or raw_binding.get("mode") or "").strip()
            if binding_kind == "literal":
                tool_input_payload[field_name] = raw_binding.get("value")
                continue
            if binding_kind == "context":
                raw_path = str(raw_binding.get("path") or raw_binding.get("contextPath") or "").strip()
                segments = _split_reference_path(raw_path)
                if not segments:
                    diagnostics_errors.append(
                        {
                            "code": "draft.binding_context_path_missing",
                            "node_id": node_id,
                            "field": field_name,
                            "message": "Context binding requires a non-empty path.",
                        }
                    )
                    continue
                tool_input_payload[field_name] = {"$from": ["job_context", *segments]}
                continue
            if binding_kind == "workflow_input":
                input_key = str(
                    raw_binding.get("inputKey") or raw_binding.get("input_key") or ""
                ).strip()
                if not input_key:
                    diagnostics_errors.append(
                        {
                            "code": "draft.binding_workflow_input_key_missing",
                            "node_id": node_id,
                            "field": field_name,
                            "message": "Workflow-input binding requires an input key.",
                        }
                    )
                    continue
                tool_input_payload[field_name] = {
                    "$from": ["job_context", "workflow", "inputs", input_key]
                }
                continue
            if binding_kind == "workflow_variable":
                variable_key = str(
                    raw_binding.get("variableKey") or raw_binding.get("variable_key") or ""
                ).strip()
                if not variable_key:
                    diagnostics_errors.append(
                        {
                            "code": "draft.binding_workflow_variable_key_missing",
                            "node_id": node_id,
                            "field": field_name,
                            "message": "Workflow-variable binding requires a variable key.",
                        }
                    )
                    continue
                tool_input_payload[field_name] = {
                    "$from": ["job_context", "workflow", "variables", variable_key]
                }
                continue
            if binding_kind in {"step_output", "from"}:
                source_id = str(
                    raw_binding.get("nodeId") or raw_binding.get("sourceNodeId") or ""
                ).strip()
                source_path = str(raw_binding.get("path") or raw_binding.get("sourcePath") or "").strip()
                if not source_id or source_id not in node_by_id:
                    diagnostics_errors.append(
                        {
                            "code": "draft.binding_source_missing",
                            "node_id": node_id,
                            "field": field_name,
                            "message": "Step-output binding references an unknown source node.",
                        }
                    )
                    continue
                path_segments = _split_reference_path(source_path)
                if not path_segments:
                    diagnostics_errors.append(
                        {
                            "code": "draft.binding_source_path_missing",
                            "node_id": node_id,
                            "field": field_name,
                            "message": "Step-output binding requires source path.",
                        }
                    )
                    continue
                source_node = node_by_id[source_id]
                deps_by_node_id[node_id].add(source_id)
                reference: dict[str, Any] = {
                    "$from": [
                        "dependencies_by_name",
                        source_node["task_name"],
                        source_node["capability_id"],
                        *path_segments,
                    ]
                }
                if "default" in raw_binding:
                    reference["$default"] = raw_binding.get("default")
                elif "defaultValue" in raw_binding:
                    raw_default = raw_binding.get("defaultValue")
                    if isinstance(raw_default, str):
                        trimmed_default = raw_default.strip()
                        if trimmed_default:
                            try:
                                reference["$default"] = json.loads(trimmed_default)
                            except Exception:
                                reference["$default"] = raw_default
                    elif raw_default is not None:
                        reference["$default"] = raw_default
                tool_input_payload[field_name] = reference
                continue
            if binding_kind == "memory":
                name = str(raw_binding.get("name") or "").strip()
                if not name:
                    diagnostics_errors.append(
                        {
                            "code": "draft.binding_memory_name_missing",
                            "node_id": node_id,
                            "field": field_name,
                            "message": "Memory binding requires name.",
                        }
                    )
                    continue
                memory_scope = str(raw_binding.get("scope") or "job").strip() or "job"
                memory_payload: dict[str, Any] = {
                    "scope": memory_scope,
                    "name": name,
                }
                key = raw_binding.get("key")
                if isinstance(key, str) and key.strip():
                    memory_payload["key"] = key.strip()
                explicit_user_id = _semantic_normalize_text(
                    raw_binding.get("userId") or raw_binding.get("user_id"),
                    max_len=120,
                )
                if explicit_user_id:
                    memory_payload["user_id"] = explicit_user_id
                elif memory_scope == "user":
                    memory_payload["user_id"] = _semantic_user_id_from_context(job_context)
                tool_input_payload[field_name] = memory_payload
                continue
            diagnostics_errors.append(
                {
                    "code": "draft.binding_kind_unknown",
                    "node_id": node_id,
                    "field": field_name,
                    "message": f"Unsupported binding kind: {binding_kind or '<empty>'}",
                }
            )

        deps = [
            node_by_id[dep_id]["task_name"]
            for dep_id in sorted(_resolve_non_control_deps(node_id))
        ]
        embedded_tool_inputs = execution_contracts.embed_execution_gate(
            {capability_id: tool_input_payload},
            node.get("execution_gate"),
            request_ids=[capability_id],
        )
        task_payload: dict[str, Any] = {
            "name": node["task_name"],
            "description": capability_spec.description,
            "instruction": f"Use capability {capability_id}.",
            "acceptance_criteria": [f"Completed capability {capability_id}"],
            "expected_output_schema_ref": capability_spec.output_schema_ref or "",
            "deps": deps,
            "tool_requests": [capability_id],
            "tool_inputs": embedded_tool_inputs,
            "critic_required": False,
        }
        intent_value = node.get("intent")
        if isinstance(intent_value, str) and intent_value in tool_intent_values:
            task_payload["intent"] = intent_value
        else:
            planner_hints = (
                capability_spec.planner_hints
                if isinstance(capability_spec.planner_hints, dict)
                else {}
            )
            hinted_intents = planner_hints.get("task_intents")
            if isinstance(hinted_intents, list):
                first_hint = next(
                    (
                        entry
                        for entry in hinted_intents
                        if isinstance(entry, str) and entry in tool_intent_values
                    ),
                    None,
                )
                if first_hint:
                    task_payload["intent"] = first_hint
            if "intent" not in task_payload:
                inferred_intent = intent_contract.infer_task_intent_for_task_with_metadata(
                    explicit_intent=None,
                    description=str(task_payload.get("description") or ""),
                    instruction=str(task_payload.get("instruction") or ""),
                    acceptance_criteria=task_payload.get("acceptance_criteria"),
                    goal_text=goal_text,
                ).intent
                if inferred_intent in tool_intent_values:
                    task_payload["intent"] = inferred_intent
        tasks_payload.append(task_payload)

    dag_edges: list[list[str]] = []
    for node in canonical_nodes:
        if node.get("is_control"):
            continue
        target_task_name = node["task_name"]
        for source_node_id in sorted(_resolve_non_control_deps(node["node_id"])):
            source_task_name = node_by_id[source_node_id]["task_name"]
            dag_edges.append([source_task_name, target_task_name])

    if diagnostics_errors or has_unsupported_control_nodes:
        return None, diagnostics_errors, diagnostics_warnings

    plan_payload = {
        "planner_version": "ui_chaining_composer_v2",
        "tasks_summary": str(
            draft.get("summary")
            or draft.get("tasks_summary")
            or f"Execute composed chain with {len(tasks_payload)} step(s)."
        ),
        "dag_edges": dag_edges,
        "tasks": tasks_payload,
    }
    plan = _parse_plan_payload(plan_payload)
    if plan is None:
        diagnostics_errors.append(
            {
                "code": "draft.plan_invalid",
                "message": "Composer draft could not be compiled into a valid PlanCreate payload.",
            }
        )
    return plan, diagnostics_errors, diagnostics_warnings


def _plan_created_payload(plan: models.PlanCreate, job_id: str) -> dict[str, Any]:
    payload = plan.model_dump()
    payload["job_id"] = job_id
    return payload


def _stream_for_event(event_type: str) -> str:
    if event_type.startswith("job"):
        return events.JOB_STREAM
    if event_type.startswith("plan"):
        return events.PLAN_STREAM
    if event_type.startswith("task"):
        return events.TASK_STREAM
    if event_type.startswith("policy"):
        return events.POLICY_STREAM
    if event_type.startswith("feedback"):
        return events.FEEDBACK_STREAM
    return events.TASK_STREAM


def _start_orchestrator() -> None:
    thread = threading.Thread(target=_orchestrator_loop, daemon=True)
    thread.start()


def _ensure_orchestrator_groups(
    local_redis: redis.Redis, group: str, stream_keys: list[str]
) -> None:
    for stream in stream_keys:
        try:
            local_redis.xgroup_create(stream, group, id="0-0", mkstream=True)
        except redis.ResponseError as exc:
            # BUSYGROUP means the group already exists.
            if "BUSYGROUP" not in str(exc):
                logger.warning(
                    "orchestrator_group_create_response_error",
                    extra={"stream": stream, "group": group, "error": str(exc)},
                )
        except redis.RedisError as exc:
            logger.warning(
                "orchestrator_group_create_redis_error",
                extra={"stream": stream, "group": group, "error": str(exc)},
            )


def _orchestrator_loop() -> None:
    consumer = str(uuid.uuid4())
    group = "api-orchestrator"
    stream_keys = [
        events.PLAN_STREAM,
        events.TASK_STREAM,
        events.CRITIC_STREAM,
        events.POLICY_STREAM,
    ]
    local_redis = redis.Redis.from_url(REDIS_URL, decode_responses=True)
    last_recovery = 0.0
    _ensure_orchestrator_groups(local_redis, group, stream_keys)
    while True:
        try:
            now = time.time()
            if ORCHESTRATOR_RECOVER_PENDING and now - last_recovery > 30:
                _recover_pending_events(local_redis, group, consumer, stream_keys)
                last_recovery = now
            messages = local_redis.xreadgroup(
                group, consumer, {stream: ">" for stream in stream_keys}, count=10, block=1000
            )
            for stream_name, entries in messages:
                for message_id, data in entries:
                    try:
                        _handle_event(stream_name, data)
                        local_redis.xack(stream_name, group, message_id)
                    except Exception:
                        logger.exception(
                            "orchestrator_handle_event_error",
                            extra={"stream": stream_name, "message_id": message_id},
                        )
                        orchestrator_handle_errors_total.labels(stream=stream_name).inc()
        except redis.ResponseError as exc:
            # Redis can lose groups/streams after restarts. Recreate and continue.
            if "NOGROUP" in str(exc):
                logger.warning("orchestrator_missing_group_recover", extra={"error": str(exc)})
                _ensure_orchestrator_groups(local_redis, group, stream_keys)
                time.sleep(0.2)
                continue
            logger.exception("orchestrator_loop_error")
            orchestrator_loop_errors_total.inc()
            time.sleep(1)
        except Exception:
            logger.exception("orchestrator_loop_error")
            orchestrator_loop_errors_total.inc()
            time.sleep(1)


def _recover_pending_events(
    local_redis: redis.Redis,
    group: str,
    consumer: str,
    stream_keys: list[str],
) -> None:
    for stream in stream_keys:
        try:
            res = local_redis.xautoclaim(
                stream,
                group,
                consumer,
                min_idle_time=ORCHESTRATOR_RECOVER_IDLE_MS,
                start_id="0-0",
                count=50,
            )
            next_id, messages, *rest = res
            if rest:
                logger.info(
                    "orchestrator_recover_deleted_ids",
                    extra={"stream": stream, "deleted_ids": rest[0]},
                )
        except Exception:
            logger.exception("orchestrator_recover_error", extra={"stream": stream})
            continue
        if not messages:
            continue
        for message_id, data in messages:
            try:
                _handle_event(stream, data)
                local_redis.xack(stream, group, message_id)
                orchestrator_recovered_events_total.labels(stream=stream).inc()
            except Exception:
                logger.exception(
                    "orchestrator_recover_handle_error",
                    extra={"stream": stream, "message_id": message_id},
                )


def _handle_event(stream_name: str, data: dict[str, str]) -> None:
    try:
        envelope = json.loads(data.get("data", "{}"))
    except json.JSONDecodeError:
        return
    event_type = envelope.get("type")
    if event_type == "plan.created":
        _handle_plan_created(envelope)
    elif event_type == "plan.failed":
        _handle_plan_failed(envelope)
    elif event_type == "task.completed":
        _handle_task_completed(envelope)
    elif event_type == "task.failed":
        _handle_task_failed(envelope)
    elif event_type == "task.started":
        _handle_task_started(envelope)
    elif event_type == "task.accepted":
        _handle_task_accepted(envelope)
    elif event_type == "task.rework_requested":
        _handle_task_rework(envelope)
    elif event_type == "policy.decision_made":
        _handle_policy_decision(envelope)
    _persist_run_event(envelope)


def _handle_plan_created(envelope: dict) -> None:
    payload = envelope.get("payload")
    if not isinstance(payload, dict):
        return
    job_id = envelope.get("job_id") or payload.get("job_id")
    if not job_id:
        return
    planner_run_spec = run_specs.parse_run_spec(payload.get("run_spec"))
    plan = _parse_plan_payload(payload)
    if plan is None and planner_run_spec is not None:
        try:
            plan = run_specs.run_spec_to_plan(planner_run_spec)
        except ValueError:
            plan = None
    if plan is None:
        return
    use_postgres_scheduler = PLANNER_RUN_SCHEDULER_ENABLED and planner_run_spec is not None
    now = _utcnow()
    with SessionLocal() as db:
        job = db.query(JobRecord).filter(JobRecord.id == job_id).first()
        job_goal = job.goal if job and isinstance(job.goal, str) else ""
        job_metadata = job.metadata_json if job and isinstance(job.metadata_json, dict) else {}
        job_context = _preflight_job_context(
            db=db,
            goal_text=job_goal,
            job_context=job.context_json if job and isinstance(job.context_json, dict) else {},
            metadata=job_metadata,
            surface="plan_created_preflight",
        )
        preflight_errors = _compile_plan_preflight(
            plan,
            job_context,
            goal_text=job_goal,
            normalized_intent_envelope=_normalized_intent_envelope_from_metadata(
                job_metadata,
                goal=job_goal,
            ),
            goal_intent_graph=job_metadata.get("goal_intent_graph") if isinstance(job_metadata, dict) else None,
            render_path_mode=planner_contracts.render_path_mode_from_metadata(
                job_metadata,
            ),
        )
        preflight_errors = _merge_preflight_errors(
            preflight_errors,
            _compile_plan_runtime_conformance_errors(plan),
        )
        if preflight_errors:
            if job:
                _set_job_status(job, models.JobStatus.failed)
                metadata = dict(job.metadata_json) if isinstance(job.metadata_json, dict) else {}
                metadata["plan_preflight_errors"] = preflight_errors
                job.metadata_json = metadata
                job.updated_at = now
                db.commit()
            _emit_event(
                "plan.failed",
                {
                    "job_id": job_id,
                    "error": "plan_preflight_failed",
                    "preflight_errors": preflight_errors,
                    "correlation_id": envelope.get("correlation_id") or str(uuid.uuid4()),
                },
            )
            return
        existing = db.query(PlanRecord).filter(PlanRecord.job_id == job_id).first()
        if existing:
            plan_id = existing.id
            if use_postgres_scheduler and job:
                metadata = dict(job.metadata_json) if isinstance(job.metadata_json, dict) else {}
                metadata["scheduler_mode"] = POSTGRES_RUN_SPEC_SCHEDULER_MODE
                metadata["run_spec"] = planner_run_spec.model_dump(mode="json")
                job.metadata_json = metadata
                job.updated_at = now
            db.commit()
            _dispatch_ready_work_for_job(job_id, plan_id, envelope.get("correlation_id"))
            _refresh_job_status(job_id)
            return
        record = PlanRecord(
            id=str(uuid.uuid4()),
            job_id=job_id,
            planner_version=plan.planner_version,
            created_at=now,
            tasks_summary=plan.tasks_summary,
            dag_edges=plan.dag_edges,
            policy_decision={},
        )
        db.add(record)
        db.flush()
        plan_record_id = record.id
        selected_capabilities: set[str] = set()
        task_intent_profiles: dict[str, dict[str, Any]] = {}
        goal_intent_segments = _goal_intent_segments_from_metadata(
            job.metadata_json if job and isinstance(job.metadata_json, dict) else {}
        )
        for index, task in enumerate(plan.tasks):
            task_id = str(uuid.uuid4())
            task_record = TaskRecord(
                id=task_id,
                job_id=job_id,
                plan_id=plan_record_id,
                name=task.name,
                description=task.description,
                instruction=task.instruction,
                acceptance_criteria=task.acceptance_criteria,
                expected_output_schema_ref=task.expected_output_schema_ref,
                status=models.TaskStatus.pending.value,
                intent=task.intent.value
                if isinstance(task.intent, models.ToolIntent)
                else task.intent,
                deps=task.deps,
                attempts=0,
                max_attempts=3,
                rework_count=0,
                max_reworks=2,
                assigned_to=None,
                tool_requests=task.tool_requests,
                tool_inputs=_task_record_tool_inputs(
                    task.tool_requests,
                    task.tool_inputs,
                    task.capability_bindings,
                ),
                created_at=now,
                updated_at=now,
                critic_required=1 if task.critic_required else 0,
            )
            db.add(task_record)
            for capability_id in task.tool_requests or []:
                normalized_capability = str(capability_id or "").strip()
                if normalized_capability:
                    selected_capabilities.add(normalized_capability)
            task_intent_value = (
                task.intent.value if isinstance(task.intent, models.ToolIntent) else str(task.intent or "")
            )
            normalized_task_intent = (
                intent_contract.normalize_task_intent(task_intent_value)
                or models.ToolIntent.generate.value
            )
            task_intent_profiles[task_id] = _task_intent_profile_entry(
                task,
                goal_text=job_goal,
                intent_segment=_select_goal_intent_segment_for_task(
                    task=task,
                    task_index=index,
                    task_intent=normalized_task_intent,
                    goal_intent_segments=goal_intent_segments,
                    total_tasks=len(plan.tasks),
                ),
            )
        if job:
            _merge_task_intent_profiles_into_job_metadata(job, task_intent_profiles)
            if use_postgres_scheduler:
                metadata = dict(job.metadata_json) if isinstance(job.metadata_json, dict) else {}
                metadata["scheduler_mode"] = POSTGRES_RUN_SPEC_SCHEDULER_MODE
                metadata["run_spec"] = planner_run_spec.model_dump(mode="json")
                job.metadata_json = metadata
            _set_job_status(job, models.JobStatus.planning)
            job.updated_at = now
        db.commit()
    for capability_id in selected_capabilities:
        planner_capability_selection_total.labels(capability=capability_id).inc()
    _dispatch_ready_work_for_job(job_id, plan_record_id, envelope.get("correlation_id"))
    _refresh_job_status(job_id)


def _handle_plan_failed(envelope: dict) -> None:
    payload = envelope.get("payload") or {}
    job_id = envelope.get("job_id") or payload.get("job_id") or payload.get("id")
    if not job_id:
        return
    error_message = payload.get("error")
    now = _utcnow()
    with SessionLocal() as db:
        job = db.query(JobRecord).filter(JobRecord.id == job_id).first()
        if not job:
            return
        _set_job_status(job, models.JobStatus.failed)
        if error_message:
            metadata = dict(job.metadata_json) if isinstance(job.metadata_json, dict) else {}
            metadata["plan_error"] = error_message
            job.metadata_json = metadata
        job.updated_at = now
        db.commit()


def _handle_task_completed(envelope: dict) -> None:
    payload = envelope.get("payload") or {}
    task_id = payload.get("task_id") or envelope.get("task_id")
    if not task_id:
        return
    _store_task_output(task_id, payload.get("outputs", {}))
    _store_task_result(task_id, payload)
    now = _utcnow()
    occurred_at = _parse_event_datetime(envelope.get("occurred_at")) or now
    with SessionLocal() as db:
        task = db.query(TaskRecord).filter(TaskRecord.id == task_id).first()
        if not task:
            return
        job = db.query(JobRecord).filter(JobRecord.id == task.job_id).first()
        attempt = _upsert_step_attempt_finished(
            db,
            task=task,
            job=job,
            payload=payload,
            status=models.TaskStatus.completed.value,
            occurred_at=occurred_at,
        )
        _replace_attempt_invocations(db, attempt=attempt, task=task, payload=payload)
        for capability_id in task.tool_requests or []:
            normalized_capability = str(capability_id or "").strip()
            if normalized_capability:
                capability_execution_outcomes_total.labels(
                    capability=normalized_capability,
                    status="completed",
                ).inc()
        task.status = models.TaskStatus.completed.value
        task.updated_at = now
        db.commit()
        job_id = task.job_id
        plan_id = task.plan_id
    _dispatch_ready_work_for_job(job_id, plan_id, envelope.get("correlation_id"))
    _refresh_job_status(job_id)


def _handle_task_failed(envelope: dict) -> None:
    payload = envelope.get("payload") or {}
    task_id = payload.get("task_id") or envelope.get("task_id")
    if not task_id:
        return
    _store_task_result(task_id, payload)
    error = payload.get("error")
    now = _utcnow()
    occurred_at = _parse_event_datetime(envelope.get("occurred_at")) or now
    with SessionLocal() as db:
        task = db.query(TaskRecord).filter(TaskRecord.id == task_id).first()
        if not task:
            return
        job = db.query(JobRecord).filter(JobRecord.id == task.job_id).first()
        attempt = _upsert_step_attempt_finished(
            db,
            task=task,
            job=job,
            payload=payload,
            status=models.TaskStatus.failed.value,
            occurred_at=occurred_at,
        )
        _replace_attempt_invocations(db, attempt=attempt, task=task, payload=payload)
        for capability_id in task.tool_requests or []:
            normalized_capability = str(capability_id or "").strip()
            if normalized_capability:
                capability_execution_outcomes_total.labels(
                    capability=normalized_capability,
                    status="failed",
                ).inc()
        if isinstance(error, str) and (
            "tool_intent_mismatch" in error or "contract.intent_mismatch" in error
        ):
            mismatch = _intent_mismatch_repair_constraints(error, task)
            replan_done = _replan_job_for_intent_mismatch(
                db,
                task.job_id,
                mismatch=mismatch,
            )
            if replan_done:
                return
        task.status = models.TaskStatus.failed.value
        task.updated_at = now
        if job:
            _set_job_status(job, models.JobStatus.failed)
            job.updated_at = now
        db.commit()


def _intent_mismatch_repair_constraints(error: str, task: TaskRecord | None) -> dict[str, Any]:
    normalized = str(error or "").strip()
    detail = normalized
    prefix = "contract.intent_mismatch:"
    if detail.startswith(prefix):
        detail = detail[len(prefix) :]
    constraints: dict[str, Any] = {
        "error": normalized,
        "detail": detail,
    }
    if isinstance(task, TaskRecord):
        constraints["failed_task_name"] = task.name
        constraints["failed_task_intent"] = str(task.intent or "").strip()
        constraints["failed_tool_requests"] = list(task.tool_requests or [])
    if detail.startswith("tool_intent_mismatch:"):
        # format: tool_intent_mismatch:<tool_name>:<required_task_intent>:<actual_task_intent>
        parts = detail.split(":")
        if len(parts) >= 4:
            constraints["failing_tool"] = parts[1]
            constraints["required_task_intent"] = parts[2]
            constraints["actual_task_intent"] = parts[3]
    elif detail.startswith("task_intent_mismatch:"):
        # format: task_intent_mismatch:<capability_id>:<actual_task_intent>:allowed=<csv>
        parts = detail.split(":")
        if len(parts) >= 4:
            constraints["failing_capability"] = parts[1]
            constraints["actual_task_intent"] = parts[2]
            allowed_part = parts[3]
            if allowed_part.startswith("allowed="):
                allowed = [
                    value.strip()
                    for value in allowed_part[len("allowed=") :].split(",")
                    if value.strip()
                ]
                if allowed:
                    constraints["allowed_task_intents"] = allowed
    elif detail.startswith("segment_intent_mismatch:"):
        # format: segment_intent_mismatch:<tool_name>:segment=<intent>:task=<intent>
        parts = detail.split(":")
        if len(parts) >= 4:
            constraints["failing_tool"] = parts[1]
            constraints["segment_intent"] = parts[2].replace("segment=", "").strip()
            constraints["actual_task_intent"] = parts[3].replace("task=", "").strip()
    return constraints


def _handle_task_started(envelope: dict) -> None:
    payload = envelope.get("payload") or {}
    task_id = payload.get("task_id") or envelope.get("task_id")
    if not task_id:
        return
    now = _utcnow()
    occurred_at = _parse_event_datetime(envelope.get("occurred_at")) or now
    with SessionLocal() as db:
        task = db.query(TaskRecord).filter(TaskRecord.id == task_id).first()
        if not task:
            return
        task.status = models.TaskStatus.running.value
        worker_consumer = payload.get("worker_consumer")
        if isinstance(worker_consumer, str) and worker_consumer.strip():
            task.assigned_to = worker_consumer.strip()
        attempts = payload.get("attempts")
        if isinstance(attempts, int):
            task.attempts = max(0, attempts)
        task.updated_at = now
        job = db.query(JobRecord).filter(JobRecord.id == task.job_id).first()
        _upsert_step_attempt_started(
            db,
            task=task,
            job=job,
            payload=payload,
            occurred_at=occurred_at,
        )
        if job:
            _set_job_status(job, models.JobStatus.running)
            job.updated_at = now
        db.commit()


def _replan_job_for_intent_mismatch(
    db: Session,
    job_id: str,
    *,
    mismatch: Mapping[str, Any] | None = None,
) -> bool:
    job = db.query(JobRecord).filter(JobRecord.id == job_id).first()
    if not job:
        return False
    metadata = dict(job.metadata_json) if isinstance(job.metadata_json, dict) else {}
    count = int(metadata.get("replan_count", 0))
    if count >= REPLAN_MAX:
        return False
    metadata["replan_count"] = count + 1
    metadata["replan_reason"] = "intent_mismatch_auto_repair"
    mismatch_payload = dict(mismatch) if isinstance(mismatch, Mapping) else {}
    if mismatch_payload:
        mismatch_payload.setdefault("attempt", count + 1)
        mismatch_payload.setdefault("created_at", _utcnow().isoformat())
        metadata["intent_mismatch_recovery"] = mismatch_payload
        history = metadata.get("intent_mismatch_recovery_history")
        if not isinstance(history, list):
            history = []
        history.append(mismatch_payload)
        metadata["intent_mismatch_recovery_history"] = history[-10:]
    job.metadata_json = metadata
    job.status = models.JobStatus.planning.value
    job.updated_at = _utcnow()
    db.query(TaskRecord).filter(TaskRecord.job_id == job_id).delete(synchronize_session=False)
    db.query(PlanRecord).filter(PlanRecord.job_id == job_id).delete(synchronize_session=False)
    db.commit()
    _emit_event("job.created", _job_from_record(job).model_dump())
    return True


def _handle_task_accepted(envelope: dict) -> None:
    payload = envelope.get("payload") or {}
    task_id = payload.get("task_id") or envelope.get("task_id")
    if not task_id:
        return
    now = _utcnow()
    with SessionLocal() as db:
        task = db.query(TaskRecord).filter(TaskRecord.id == task_id).first()
        if not task:
            return
        task.status = models.TaskStatus.accepted.value
        task.updated_at = now
        _update_latest_step_attempt_status(
            db,
            task=task,
            status=models.TaskStatus.accepted.value,
        )
        db.commit()
        job_id = task.job_id
        plan_id = task.plan_id
    _dispatch_ready_work_for_job(job_id, plan_id, envelope.get("correlation_id"))
    _refresh_job_status(job_id)


def _handle_task_rework(envelope: dict) -> None:
    payload = envelope.get("payload") or {}
    task_id = payload.get("task_id") or envelope.get("task_id")
    if not task_id:
        return
    now = _utcnow()
    with SessionLocal() as db:
        task = db.query(TaskRecord).filter(TaskRecord.id == task_id).first()
        if not task:
            return
        _update_latest_step_attempt_status(
            db,
            task=task,
            status=models.TaskStatus.rework_requested.value,
        )
        task.rework_count = (task.rework_count or 0) + 1
        if _limit_exceeded(task.rework_count, task.max_reworks):
            task.status = models.TaskStatus.failed.value
            task.updated_at = now
            db.commit()
            _emit_event(
                "task.failed",
                _task_payload_with_error(
                    task, envelope.get("correlation_id"), "max_reworks_exceeded"
                ),
            )
            _refresh_job_status(task.job_id)
            return
        task.status = models.TaskStatus.pending.value
        task.updated_at = now
        db.commit()
        job_id = task.job_id
        plan_id = task.plan_id
    _dispatch_ready_work_for_job(job_id, plan_id, envelope.get("correlation_id"))


def _parse_plan_payload(payload: dict) -> models.PlanCreate | None:
    try:
        return models.PlanCreate.model_validate(payload)
    except Exception:
        return None


def _limit_exceeded(count: int, limit: int | None) -> bool:
    if not limit or limit <= 0:
        return False
    return count > limit


def _enqueue_ready_tasks(job_id: str, plan_id: str, correlation_id: str | None) -> None:
    dispatch_service.enqueue_ready_tasks(
        job_id,
        plan_id,
        correlation_id,
        runtime=_dispatch_runtime(),
        callbacks=_dispatch_callbacks(),
    )


def _latest_step_attempts_for_steps(
    db: Session,
    step_ids: Sequence[str],
) -> dict[str, StepAttemptRecord]:
    normalized_step_ids = [str(step_id).strip() for step_id in step_ids if str(step_id).strip()]
    if not normalized_step_ids:
        return {}
    rows = (
        db.query(StepAttemptRecord)
        .filter(StepAttemptRecord.step_id.in_(normalized_step_ids))
        .order_by(
            StepAttemptRecord.step_id.asc(),
            StepAttemptRecord.attempt_number.desc(),
            StepAttemptRecord.started_at.desc(),
        )
        .all()
    )
    latest: dict[str, StepAttemptRecord] = {}
    for row in rows:
        latest.setdefault(row.step_id, row)
    return latest


def _scheduler_effective_task_status(
    task: TaskRecord,
    latest_attempt: StepAttemptRecord | None,
) -> str:
    task_status = str(task.status or "").strip() or models.TaskStatus.pending.value
    if task_status != models.TaskStatus.pending.value:
        return task_status
    latest_status = str(latest_attempt.status or "").strip() if latest_attempt is not None else ""
    if latest_status in {
        models.TaskStatus.running.value,
        models.TaskStatus.completed.value,
        models.TaskStatus.accepted.value,
    }:
        return latest_status
    return task_status


def _schedule_postgres_run(job_id: str, *, correlation_id: str | None = None) -> None:
    now = _utcnow()
    events_to_emit: list[tuple[str, dict[str, Any]]] = []
    scheduled_job_id: str | None = None
    with SessionLocal() as db:
        job = db.query(JobRecord).filter(JobRecord.id == job_id).first()
        if job is None or not _job_uses_postgres_scheduler(job):
            return
        workflow_run_id = _job_workflow_run_id(job)
        workflow_run = None
        run_spec: models.RunSpec | None = None
        run_label = job.id
        if workflow_run_id:
            workflow_run = (
                db.query(WorkflowRunRecord).filter(WorkflowRunRecord.id == workflow_run_id).first()
            )
            if workflow_run is not None and _workflow_run_uses_postgres_scheduler(workflow_run):
                version = (
                    db.query(WorkflowVersionRecord)
                    .filter(WorkflowVersionRecord.id == workflow_run.version_id)
                    .first()
                )
                if version is not None:
                    run_spec = _workflow_version_run_spec(version)
                    run_label = workflow_run.id
        if run_spec is None:
            run_spec = _planner_job_run_spec(job)
        if run_spec is None:
            logger.warning(
                "postgres_scheduler_run_spec_missing",
                extra={"job_id": job.id, "workflow_run_id": workflow_run_id},
            )
            return
        plan = db.query(PlanRecord).filter(PlanRecord.job_id == job.id).first()
        if plan is None:
            return
        scheduled_job_id = job.id
        task_records = db.query(TaskRecord).filter(TaskRecord.plan_id == plan.id).all()
        if not task_records:
            return
        task_by_name = {record.name: record for record in task_records}
        latest_attempts = _latest_step_attempts_for_steps(db, [record.id for record in task_records])
        tasks = _resolve_task_deps(task_records)
        task_map = {task.id: task for task in tasks}
        id_to_name = {record.id: record.name for record in task_records}
        job_context = _execution_job_context(
            job.goal if isinstance(job.goal, str) else "",
            job.context_json if isinstance(job.context_json, dict) else {},
            job.metadata_json if isinstance(job.metadata_json, dict) else {},
        )
        job_goal = job.goal if isinstance(job.goal, str) else ""
        task_intent_profiles = _coerce_task_intent_profiles(
            job.metadata_json if isinstance(job.metadata_json, dict) else {}
        )
        completed_step_ids: set[str] = set()
        effective_status_by_step_id: dict[str, str] = {}
        record_by_step_id: dict[str, TaskRecord] = {}
        for step in run_spec.steps:
            record = task_by_name.get(step.name)
            if record is None:
                logger.warning(
                    "postgres_scheduler_task_missing",
                    extra={
                        "run_id": run_label,
                        "job_id": job.id,
                        "step_id": step.step_id,
                        "step_name": step.name,
                    },
                )
                continue
            effective_status = _scheduler_effective_task_status(
                record,
                latest_attempts.get(record.id),
            )
            if effective_status in {
                models.TaskStatus.running.value,
                models.TaskStatus.completed.value,
                models.TaskStatus.accepted.value,
            } and record.status != effective_status:
                record.status = effective_status
                record.updated_at = now
            effective_status_by_step_id[step.step_id] = effective_status
            record_by_step_id[step.step_id] = record
            if effective_status in {models.TaskStatus.completed.value, models.TaskStatus.accepted.value}:
                completed_step_ids.add(step.step_id)
        for step in run_spec.steps:
            record = record_by_step_id.get(step.step_id)
            if record is None:
                continue
            if effective_status_by_step_id.get(step.step_id) != models.TaskStatus.pending.value:
                continue
            if any(dependency_id not in completed_step_ids for dependency_id in step.depends_on):
                continue
            max_attempts = int(record.max_attempts or 0)
            if max_attempts <= 0:
                max_attempts = max(1, int(step.retry_policy.max_attempts or 1))
                record.max_attempts = max_attempts
            next_attempt = (record.attempts or 0) + 1
            if _limit_exceeded(next_attempt, max_attempts):
                record.status = models.TaskStatus.failed.value
                record.updated_at = now
                events_to_emit.append(
                    (
                        "task.failed",
                        _task_payload_with_error(
                            record,
                            correlation_id,
                            "max_attempts_exceeded",
                        ),
                    )
                )
                effective_status_by_step_id[step.step_id] = models.TaskStatus.failed.value
                continue
            context = _build_task_context(record.id, task_map, id_to_name, job_context)
            if POLICY_GATE_ENABLED:
                record.status = models.TaskStatus.blocked.value
                record.updated_at = now
                payload = _task_payload_from_record(
                    record,
                    correlation_id,
                    context,
                    goal_text=job_goal,
                    intent_profile=task_intent_profiles.get(record.id),
                )
                events_to_emit.append(("task.policy_check", payload))
                effective_status_by_step_id[step.step_id] = models.TaskStatus.blocked.value
                continue
            record.attempts = next_attempt
            record.status = models.TaskStatus.ready.value
            record.updated_at = now
            payload = _task_payload_from_record(
                record,
                correlation_id,
                context,
                goal_text=job_goal,
                intent_profile=task_intent_profiles.get(record.id),
            )
            if TOOL_INPUT_VALIDATION_ENABLED and payload.get("tool_inputs_validation"):
                record.status = models.TaskStatus.failed.value
                record.updated_at = now
                failed_payload = dict(payload)
                failed_payload["error"] = "tool_inputs_invalid"
                events_to_emit.append(("task.failed", failed_payload))
                effective_status_by_step_id[step.step_id] = models.TaskStatus.failed.value
                continue
            events_to_emit.append(("task.ready", payload))
            effective_status_by_step_id[step.step_id] = models.TaskStatus.ready.value
        db.commit()
    for event_type, event_payload in events_to_emit:
        _emit_event(event_type, event_payload)
    if scheduled_job_id:
        _refresh_job_status(scheduled_job_id)


def _schedule_workflow_run(workflow_run_id: str, *, correlation_id: str | None = None) -> None:
    with SessionLocal() as db:
        workflow_run = (
            db.query(WorkflowRunRecord).filter(WorkflowRunRecord.id == workflow_run_id).first()
        )
        if workflow_run is None:
            return
        job_id = workflow_run.job_id
    _schedule_postgres_run(job_id, correlation_id=correlation_id)


def _dispatch_ready_work_for_job(
    job_id: str,
    plan_id: str | None,
    correlation_id: str | None,
) -> None:
    resolved_plan_id = plan_id
    with SessionLocal() as db:
        job = db.query(JobRecord).filter(JobRecord.id == job_id).first()
        if job is None:
            return
        use_postgres_scheduler = _job_uses_postgres_scheduler(job)
        if resolved_plan_id is None:
            plan = db.query(PlanRecord).filter(PlanRecord.job_id == job_id).first()
            resolved_plan_id = plan.id if plan is not None else None
    if use_postgres_scheduler:
        _schedule_postgres_run(job_id, correlation_id=correlation_id)
        return
    if resolved_plan_id:
        _enqueue_ready_tasks(job_id, resolved_plan_id, correlation_id)


def _task_payload_from_record(
    record: TaskRecord,
    correlation_id: str | None,
    context: dict[str, Any] | None = None,
    *,
    goal_text: str = "",
    intent_profile: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return dispatch_service.task_payload_from_record(
        record,
        correlation_id,
        context=context,
        goal_text=goal_text,
        intent_profile=intent_profile,
        config=_dispatch_runtime().config,
        callbacks=_dispatch_callbacks(),
    )


def _task_payload_with_error(
    record: TaskRecord, correlation_id: str | None, error: str
) -> dict[str, Any]:
    return dispatch_service.task_payload_with_error(
        record,
        correlation_id,
        error,
        config=_dispatch_runtime().config,
        callbacks=_dispatch_callbacks(),
    )


def _handle_policy_decision(envelope: dict) -> None:
    payload = envelope.get("payload")
    if not isinstance(payload, dict):
        return
    task_id = envelope.get("task_id") or payload.get("task_id")
    if not task_id:
        return
    decision = payload.get("decision")
    reasons = payload.get("reasons") or []
    rewrites = payload.get("rewrites")
    correlation_id = envelope.get("correlation_id")
    now = _utcnow()
    events_to_emit: list[tuple[str, dict[str, Any]]] = []
    with SessionLocal() as db:
        task = db.query(TaskRecord).filter(TaskRecord.id == task_id).first()
        if not task:
            return
        if task.status not in {
            models.TaskStatus.pending.value,
            models.TaskStatus.blocked.value,
        }:
            return
        if decision == "deny":
            task.status = models.TaskStatus.failed.value
            task.updated_at = now
            db.commit()
            reason_text = "policy_denied"
            if reasons:
                reason_text = f"policy_denied:{'; '.join(reasons)}"
            events_to_emit.append(
                ("task.failed", _task_payload_with_error(task, correlation_id, reason_text))
            )
        elif decision in {"allow", "rewrite"}:
            if isinstance(rewrites, dict):
                _apply_task_rewrites(task, rewrites)
            next_attempt = (task.attempts or 0) + 1
            if _limit_exceeded(next_attempt, task.max_attempts):
                task.status = models.TaskStatus.failed.value
                task.updated_at = now
                db.commit()
                events_to_emit.append(
                    (
                        "task.failed",
                        _task_payload_with_error(task, correlation_id, "max_attempts_exceeded"),
                    )
                )
            else:
                task.attempts = next_attempt
                task.status = models.TaskStatus.ready.value
                task.updated_at = now
                db.commit()
                task_records = db.query(TaskRecord).filter(TaskRecord.plan_id == task.plan_id).all()
                tasks = _resolve_task_deps(task_records)
                task_map = {entry.id: entry for entry in tasks}
                id_to_name = {record.id: record.name for record in task_records}
                job_record = db.query(JobRecord).filter(JobRecord.id == task.job_id).first()
                job_context = (
                    _execution_job_context(
                        job_record.goal if job_record and isinstance(job_record.goal, str) else "",
                        job_record.context_json
                        if job_record and isinstance(job_record.context_json, dict)
                        else {},
                        job_record.metadata_json
                        if job_record and isinstance(job_record.metadata_json, dict)
                        else {},
                    )
                    if job_record
                    else {}
                )
                job_goal = job_record.goal if job_record and isinstance(job_record.goal, str) else ""
                task_intent_profiles = _coerce_task_intent_profiles(
                    job_record.metadata_json
                    if job_record and isinstance(job_record.metadata_json, dict)
                    else {}
                )
                context = _build_task_context(task.id, task_map, id_to_name, job_context)
                payload = _task_payload_from_record(
                    task,
                    correlation_id,
                    context,
                    goal_text=job_goal,
                    intent_profile=task_intent_profiles.get(task.id),
                )
                if TOOL_INPUT_VALIDATION_ENABLED and payload.get("tool_inputs_validation"):
                    task.status = models.TaskStatus.failed.value
                    task.updated_at = now
                    db.commit()
                    failed_payload = dict(payload)
                    failed_payload["error"] = "tool_inputs_invalid"
                    events_to_emit.append(
                        (
                            "task.failed",
                            failed_payload,
                        )
                    )
                else:
                    events_to_emit.append(("task.ready", payload))
        db.commit()
    for event_type, event_payload in events_to_emit:
        _emit_event(event_type, event_payload)
    job_id = envelope.get("job_id") or payload.get("job_id")
    if job_id:
        _refresh_job_status(job_id)


def _recover_jobs() -> None:
    with SessionLocal() as db:
        jobs = (
            db.query(JobRecord)
            .filter(
                JobRecord.status.in_(
                    [
                        models.JobStatus.queued.value,
                        models.JobStatus.planning.value,
                        models.JobStatus.running.value,
                    ]
                )
            )
            .all()
        )
        for job in jobs:
            plan = db.query(PlanRecord).filter(PlanRecord.job_id == job.id).first()
            if plan:
                _dispatch_ready_work_for_job(job.id, plan.id, None)
                continue
            _emit_event("job.created", _job_from_record(job).model_dump())


def _apply_task_rewrites(task: TaskRecord, rewrites: dict[str, Any]) -> None:
    if "instruction" in rewrites and isinstance(rewrites["instruction"], str):
        task.instruction = rewrites["instruction"]
    if "description" in rewrites and isinstance(rewrites["description"], str):
        task.description = rewrites["description"]
    if "acceptance_criteria" in rewrites and isinstance(rewrites["acceptance_criteria"], list):
        task.acceptance_criteria = rewrites["acceptance_criteria"]
    if "expected_output_schema_ref" in rewrites and isinstance(
        rewrites["expected_output_schema_ref"], str
    ):
        task.expected_output_schema_ref = rewrites["expected_output_schema_ref"]
    execution_rewritten = False
    tool_requests = list(task.tool_requests or [])
    tool_inputs = execution_contracts.strip_execution_metadata_from_tool_inputs(
        task.tool_inputs if isinstance(task.tool_inputs, dict) else {}
    )
    capability_bindings = _task_capability_bindings(
        tool_requests,
        task.tool_inputs if isinstance(task.tool_inputs, dict) else {},
    )
    if "tool_requests" in rewrites and isinstance(rewrites["tool_requests"], list):
        tool_requests = rewrites["tool_requests"]
        task.tool_requests = tool_requests
        execution_rewritten = True
    if "tool_inputs" in rewrites and isinstance(rewrites["tool_inputs"], dict):
        tool_inputs = execution_contracts.strip_execution_metadata_from_tool_inputs(
            rewrites["tool_inputs"]
        )
        execution_rewritten = True
    if "capability_bindings" in rewrites and isinstance(rewrites["capability_bindings"], dict):
        capability_bindings = _task_capability_bindings(
            tool_requests,
            tool_inputs,
            rewrites["capability_bindings"],
        )
        execution_rewritten = True
    elif execution_rewritten:
        capability_bindings = _task_capability_bindings(
            tool_requests,
            tool_inputs,
            capability_bindings,
        )
    if execution_rewritten:
        task.tool_inputs = _task_record_tool_inputs(
            tool_requests,
            tool_inputs,
            capability_bindings,
        )


def _resolve_task_deps(task_records: list[TaskRecord]) -> list[models.Task]:
    name_to_id = {record.name: record.id for record in task_records}
    tasks: list[models.Task] = []
    for record in task_records:
        task = _task_from_record(record)
        resolved = []
        for dep in task.deps:
            if dep in name_to_id:
                resolved.append(name_to_id[dep])
            else:
                resolved.append(dep)
        tasks.append(task.model_copy(update={"deps": resolved}))
    return tasks


def _preflight_placeholder_for_key(key: str) -> Any:
    normalized = key.strip().lower()
    if normalized in {
        "path",
        "output_path",
        "topic",
        "target_role_name",
        "role_name",
        "company_name",
        "company",
        "candidate_name",
        "first_name",
        "last_name",
        "job_description",
        "date",
        "today",
        "document_type",
        "output_extension",
        "file_extension",
        "extension",
        "format",
    }:
        return "preflight"
    if normalized in {"errors", "warnings", "allowed_block_types", "items"}:
        return []
    if normalized in {"valid", "strict"}:
        return True
    if normalized in {"max_iterations", "max_blocks", "max_depth", "target_pages", "page_count"}:
        return 1
    if normalized in {"document_spec", "validation_report", "job", "result", "data"}:
        return {}
    return {
        "document_spec": {},
        "validation_report": {"valid": True, "errors": [], "warnings": []},
        "path": "documents/preflight.pdf",
        "result": {},
        "text": "preflight",
    }


def _put_nested_value(root: dict[str, Any], segments: list[str], value: Any) -> None:
    cursor: dict[str, Any] = root
    for idx, segment in enumerate(segments):
        if idx == len(segments) - 1:
            cursor[segment] = value
            return
        next_value = cursor.get(segment)
        if not isinstance(next_value, dict):
            next_value = {}
            cursor[segment] = next_value
        cursor = next_value


def _collect_reference_paths(value: Any) -> list[list[str]]:
    refs: list[list[str]] = []

    def _walk(node: Any) -> None:
        if isinstance(node, dict):
            from_path = node.get("$from")
            if isinstance(from_path, str) and from_path.strip():
                raw = from_path.strip()
                if raw.startswith("$."):
                    raw = raw[2:]
                if raw.startswith("/"):
                    parts = [part for part in raw.split("/")[1:] if part]
                else:
                    parts = [segment for segment in raw.split(".") if segment]
                if parts:
                    refs.append(parts)
            for child in node.values():
                _walk(child)
        elif isinstance(node, list):
            for child in node:
                _walk(child)

    _walk(value)
    return refs


def _normalize_preflight_reference_payload(
    value: Any,
    *,
    dependency_names: set[str],
    tasks_by_name: Mapping[str, models.TaskCreate],
) -> Any:
    if isinstance(value, dict):
        normalized = dict(value)
        from_path = normalized.get("$from")
        if isinstance(from_path, str) and from_path.strip():
            raw = from_path.strip()
            body = raw[2:] if raw.startswith("$.") else raw
            if body.startswith("/"):
                parts = [part.replace("~1", "/").replace("~0", "~") for part in body.split("/")[1:]]
            else:
                parts = [segment for segment in body.split(".") if segment]
            canonical = _canonicalize_preflight_reference_path(
                parts,
                dependency_names=dependency_names,
                tasks_by_name=tasks_by_name,
            )
            if canonical:
                escaped = [segment.replace("~", "~0").replace("/", "~1") for segment in canonical]
                normalized["$from"] = "/" + "/".join(escaped)
        for key, child in list(normalized.items()):
            if key == "$from":
                continue
            normalized[key] = _normalize_preflight_reference_payload(
                child,
                dependency_names=dependency_names,
                tasks_by_name=tasks_by_name,
            )
        return normalized
    if isinstance(value, list):
        return [
            _normalize_preflight_reference_payload(
                child,
                dependency_names=dependency_names,
                tasks_by_name=tasks_by_name,
            )
            for child in value
        ]
    return value


def _canonicalize_preflight_reference_path(
    path: list[str],
    *,
    dependency_names: set[str],
    tasks_by_name: Mapping[str, models.TaskCreate],
) -> list[str]:
    if len(path) < 3:
        return path

    if path[0] in {"dependencies_by_name", "dependencies"}:
        root = path[0]
        remaining = path[1:]
    else:
        root = ""
        remaining = path

    dep_name: str | None = None
    dep_consumed = 0
    for end in range(len(remaining), 0, -1):
        candidate = ".".join(remaining[:end]).strip()
        if candidate in dependency_names:
            dep_name = candidate
            dep_consumed = end
            break
    if not dep_name:
        return path

    dep_task = tasks_by_name.get(dep_name)
    if dep_task is None:
        return path

    tail = remaining[dep_consumed:]
    if not tail:
        normalized = [dep_name]
        return [root, *normalized] if root else normalized

    tool_requests = _task_request_ids_for_preflight(dep_task)
    tool_name: str | None = None
    tool_consumed = 0
    for end in range(len(tail), 0, -1):
        candidate = ".".join(tail[:end]).strip()
        if candidate in tool_requests:
            tool_name = candidate
            tool_consumed = end
            break

    normalized = [dep_name]
    if tool_name:
        normalized.append(tool_name)
        normalized.extend(tail[tool_consumed:])
    else:
        normalized.extend(tail)
    return [root, *normalized] if root else normalized


def _task_request_ids_for_preflight(task: models.TaskCreate) -> list[str]:
    return planner_contracts.planner_task_request_ids(task)


def _build_preflight_dependency_output(task: models.TaskCreate) -> dict[str, Any]:
    output: dict[str, Any] = {
        "document_spec": {},
        "validation_report": {"valid": True, "errors": [], "warnings": []},
        "path": "documents/preflight.pdf",
    }
    for tool_name in _task_request_ids_for_preflight(task):
        output[tool_name] = {
            "document_spec": {},
            "validation_report": {"valid": True, "errors": [], "warnings": []},
            "path": "documents/preflight.pdf",
            "result": {},
            "text": "preflight",
        }
    return output


def _collect_ancestor_task_names(
    task: models.TaskCreate, tasks_by_name: dict[str, models.TaskCreate]
) -> set[str]:
    ancestors: set[str] = set()
    stack = list(task.deps or [])
    while stack:
        dep_name = stack.pop()
        if dep_name in ancestors:
            continue
        ancestors.add(dep_name)
        dep = tasks_by_name.get(dep_name)
        if dep:
            for child in dep.deps or []:
                if child not in ancestors:
                    stack.append(child)
    return ancestors


def _compile_plan_preflight(
    plan: models.PlanCreate,
    job_context: dict[str, Any] | None,
    *,
    goal_text: str = "",
    normalized_intent_envelope: workflow_contracts.NormalizedIntentEnvelope
    | Mapping[str, Any]
    | None = None,
    goal_intent_graph: Mapping[str, Any] | None = None,
    render_path_mode: str = planner_contracts.RENDER_PATH_MODE_EXPLICIT,
) -> dict[str, str]:
    errors: dict[str, str] = {}
    full_capability_registry: capability_registry.CapabilityRegistry | None = None
    capabilities: dict[str, capability_registry.CapabilitySpec] = {}
    try:
        full_capability_registry = capability_registry.load_capability_registry()
        capabilities = full_capability_registry.enabled_capabilities()
    except Exception:  # noqa: BLE001
        capabilities = {}
    tasks_by_name: dict[str, models.TaskCreate] = {}
    duplicate_names: set[str] = set()
    for task in plan.tasks:
        if task.name in tasks_by_name:
            duplicate_names.add(task.name)
        tasks_by_name[task.name] = task
    if duplicate_names:
        for name in sorted(duplicate_names):
            errors[name] = "duplicate_task_name"
        return errors

    for task in plan.tasks:
        missing_deps = [dep for dep in (task.deps or []) if dep not in tasks_by_name]
        if missing_deps:
            errors[task.name] = f"unknown_dependencies:{', '.join(sorted(missing_deps))}"

    if errors:
        return errors

    goal_intent_segments = _goal_intent_segments_for_preflight(
        normalized_intent_envelope=normalized_intent_envelope,
        goal_intent_graph=goal_intent_graph,
    )
    for task_index, task in enumerate(plan.tasks):
        task_request_ids = _task_request_ids_for_preflight(task)
        for request_id in task_request_ids:
            language_error = planner_contracts.validate_planner_request_language(
                request_id,
                capabilities=capabilities,
                full_capabilities=full_capability_registry,
                runtime_tool_names=sorted(set(TOOL_INPUT_SCHEMAS) | set(TOOL_INTENTS_BY_NAME)),
            )
            if language_error:
                errors[task.name] = language_error
                break
        if task.name in errors:
            continue
        dependency_names = _collect_ancestor_task_names(task, tasks_by_name)
        task_intent = _preflight_task_intent(task, goal_text=goal_text)
        goal_intent_segment = _select_goal_intent_segment_for_task(
            task=task,
            task_index=task_index,
            task_intent=task_intent,
            goal_intent_segments=goal_intent_segments,
            total_tasks=len(plan.tasks),
        )
        for request_id in task_request_ids:
            tool_intent = TOOL_INTENTS_BY_NAME.get(request_id)
            if tool_intent is not None:
                mismatch = intent_contract.validate_tool_intent_compatibility(
                    task_intent,
                    tool_intent,
                    request_id,
                )
                if mismatch:
                    errors[task.name] = mismatch
                    break
            capability_spec = capabilities.get(request_id)
            capability_mismatch = _preflight_capability_intent_mismatch(
                task_intent, request_id, capability_spec
            )
            if capability_mismatch:
                errors[task.name] = capability_mismatch
                break
        if task.name in errors:
            continue
        context: dict[str, Any] = {"dependencies_by_name": {}, "dependencies": {}}
        for dep_name in dependency_names:
            dep_task = tasks_by_name.get(dep_name)
            if not dep_task:
                continue
            stub = _build_preflight_dependency_output(dep_task)
            context["dependencies_by_name"][dep_name] = stub
            context["dependencies"][dep_name] = stub
        if isinstance(job_context, dict) and job_context:
            context["job_context"] = job_context

        normalized_tool_inputs = _normalize_preflight_reference_payload(
            task.tool_inputs or {},
            dependency_names=dependency_names,
            tasks_by_name=tasks_by_name,
        )
        task_payload = {
            "name": task.name,
            "instruction": task.instruction,
            "tool_requests": list(task_request_ids),
            "tool_inputs": normalized_tool_inputs,
        }

        # Seed exact $from paths with typed placeholders to reduce false negatives
        # when dependency outputs are not available yet.
        references = _collect_reference_paths(normalized_tool_inputs)
        reference_error: str | None = None
        for path in references:
            if not path:
                continue
            path = _canonicalize_preflight_reference_path(
                path,
                dependency_names=dependency_names,
                tasks_by_name=tasks_by_name,
            )
            if path[0] in {"dependencies_by_name", "dependencies", "job_context"}:
                root_key = path[0]
                if root_key == "job_context":
                    if len(path) > 1 and "job_context" in context:
                        _put_nested_value(
                            context["job_context"],
                            path[1:],
                            _preflight_placeholder_for_key(path[-1]),
                        )
                    continue
                if len(path) >= 4:
                    dep_name = path[1]
                    dep_task = tasks_by_name.get(dep_name)
                    if dep_name in dependency_names and dep_task:
                        referenced_tool = path[2]
                        if referenced_tool not in set(_task_request_ids_for_preflight(dep_task)):
                            reference_error = (
                                "input reference resolution failed: "
                                f"path '{'.'.join(path)}' references unknown dependency tool "
                                f"'{referenced_tool}' for task '{dep_name}'"
                            )
                            break
                _put_nested_value(
                    context[root_key], path[1:], _preflight_placeholder_for_key(path[-1])
                )
            else:
                if len(path) >= 3:
                    dep_name = path[0]
                    dep_task = tasks_by_name.get(dep_name)
                    if dep_name in dependency_names and dep_task:
                        referenced_tool = path[1]
                        if referenced_tool not in set(_task_request_ids_for_preflight(dep_task)):
                            reference_error = (
                                "input reference resolution failed: "
                                f"path '{'.'.join(path)}' references unknown dependency tool "
                                f"'{referenced_tool}' for task '{dep_name}'"
                            )
                            break
                _put_nested_value(
                    context["dependencies_by_name"],
                    path,
                    _preflight_placeholder_for_key(path[-1]),
                )
                _put_nested_value(
                    context["dependencies"], path, _preflight_placeholder_for_key(path[-1])
                )
        if reference_error:
            errors[task.name] = reference_error
            continue

        resolved_inputs, resolution_errors = payload_resolver.resolve_tool_inputs_with_errors(
            task_payload["tool_requests"],
            task_payload["instruction"],
            context,
            task_payload,
            task_payload["tool_inputs"],
        )
        if resolution_errors:
            first_tool, message = next(iter(resolution_errors.items()))
            errors[task.name] = f"{first_tool}:{message}"
            continue

        request_payload_error: str | None = None
        for request_id in task_request_ids:
            resolved_payload_raw = resolved_inputs.get(request_id, {})
            resolved_payload = (
                resolved_payload_raw
                if isinstance(resolved_payload_raw, Mapping)
                else {}
            )
            request_payload_error = _preflight_request_payload_semantics(
                request_id=request_id,
                payload=resolved_payload,
                raw_payload=(
                    normalized_tool_inputs.get(request_id, {})
                    if isinstance(normalized_tool_inputs.get(request_id), Mapping)
                    else {}
                ),
                job_context=context.get("job_context"),
                render_path_mode=render_path_mode,
            )
            if request_payload_error:
                errors[task.name] = request_payload_error
                break
        if request_payload_error:
            continue

        validation_errors = payload_resolver.validate_tool_inputs(
            resolved_inputs, TOOL_INPUT_SCHEMAS
        )
        if validation_errors:
            first_tool, message = next(iter(validation_errors.items()))
            errors[task.name] = f"{first_tool}:{message}"
            continue

        for request_id in task_request_ids:
            resolved_payload_raw = resolved_inputs.get(request_id, {})
            resolved_payload = (
                resolved_payload_raw
                if isinstance(resolved_payload_raw, Mapping)
                else {}
            )
            segment_payload = dict(resolved_payload)
            segment_payload.setdefault("tool_inputs", task_payload.get("tool_inputs", {}))
            if (
                "instruction" not in segment_payload
                and isinstance(task.instruction, str)
                and task.instruction.strip()
            ):
                segment_payload["instruction"] = task.instruction.strip()
            if request_id == "github.repo.list":
                github_query = _synthesize_preflight_github_repo_query(
                    task_payload=task_payload,
                    segment_payload=segment_payload,
                    job_context=context.get("job_context"),
                )
                if github_query:
                    segment_payload["query"] = github_query
            capability_spec = capabilities.get(request_id)
            segment_contract_error = intent_contract.validate_intent_segment_contract(
                segment=goal_intent_segment,
                task_intent=task_intent,
                tool_name=request_id,
                payload=segment_payload,
                capability_id=request_id if capability_spec is not None else None,
                capability_risk_tier=capability_spec.risk_tier
                if capability_spec is not None
                else None,
            )
            if segment_contract_error:
                _record_intent_segment_rejection(
                    surface="api_preflight",
                    task_name=task.name,
                    request_id=request_id,
                    detail=segment_contract_error,
                )
                errors[task.name] = (
                    f"intent_segment_invalid:{request_id}:{task.name}:{segment_contract_error}"
                )
                break

    return errors


def _merge_preflight_errors(
    primary: Mapping[str, str] | None,
    secondary: Mapping[str, str] | None,
) -> dict[str, str]:
    merged = dict(primary or {})
    for task_name, message in (secondary or {}).items():
        key = str(task_name)
        if key not in merged:
            merged[key] = str(message)
    return merged


def _task_capability_requests_for_runtime_conformance(
    task: models.TaskCreate,
    *,
    known_capabilities: set[str],
) -> list[str]:
    capability_ids: list[str] = []
    seen: set[str] = set()
    for request_id in getattr(task, "capability_requests", None) or []:
        normalized = str(request_id or "").strip()
        if normalized and normalized in known_capabilities and normalized not in seen:
            seen.add(normalized)
            capability_ids.append(normalized)
    for request_id in task.tool_requests or []:
        normalized = str(request_id or "").strip()
        if normalized and normalized in known_capabilities and normalized not in seen:
            seen.add(normalized)
            capability_ids.append(normalized)
    raw_bindings = task.capability_bindings if isinstance(task.capability_bindings, dict) else {}
    for binding in raw_bindings.values():
        if not isinstance(binding, Mapping):
            continue
        capability_id = str(binding.get("capability_id") or "").strip()
        if capability_id and capability_id not in seen:
            seen.add(capability_id)
            capability_ids.append(capability_id)
    return capability_ids


def _compile_plan_runtime_conformance_errors(
    plan: models.PlanCreate,
    *,
    service_name: str = RUNTIME_CONFORMANCE_SERVICE,
) -> dict[str, str]:
    if not RUNTIME_CONFORMANCE_ENABLED:
        return {}
    manifest = runtime_manifest.build_runtime_manifest(service_name)
    known_capabilities = set(manifest.capabilities.keys())
    if not known_capabilities and not manifest.build_errors:
        return {}
    errors: dict[str, str] = {}
    for task in plan.tasks:
        for capability_id in _task_capability_requests_for_runtime_conformance(
            task,
            known_capabilities=known_capabilities,
        ):
            detail = runtime_manifest.explain_capability_unavailability(manifest, capability_id)
            if detail:
                errors[task.name] = (
                    f"runtime conformance failed for capability '{capability_id}' on "
                    f"{service_name}: {detail}"
                )
                break
    return errors


def _preflight_request_payload_semantics(
    *,
    request_id: str,
    payload: Mapping[str, Any],
    raw_payload: Mapping[str, Any] | None = None,
    job_context: Mapping[str, Any] | None = None,
    render_path_mode: str = planner_contracts.RENDER_PATH_MODE_EXPLICIT,
) -> str | None:
    render_path_error = planner_contracts.validate_render_path_requirement(
        request_id=request_id,
        raw_payload=raw_payload,
        resolved_payload=payload,
        job_context=job_context,
        render_path_mode=render_path_mode,
    )
    if render_path_error:
        return render_path_error
    if request_id in {"memory.read", "memory.write"}:
        return _preflight_memory_request_payload(request_id=request_id, payload=payload)
    return None


def _preflight_memory_request_payload(
    *,
    request_id: str,
    payload: Mapping[str, Any],
) -> str | None:
    name = str(payload.get("name") or "").strip()
    if not name:
        return None

    registry = memory_store.MEMORY_REGISTRY
    if not registry.has(name):
        suggestion = difflib.get_close_matches(
            name,
            [spec.name for spec in registry.list()],
            n=1,
            cutoff=0.45,
        )
        suffix = f"; did you mean '{suggestion[0]}'?" if suggestion else ""
        return f"{request_id}:unknown_memory:{name}{suffix}"

    spec = registry.get(name)
    raw_scope = str(payload.get("scope") or "").strip().lower()
    expected_scope = spec.scope.value
    if raw_scope and raw_scope != expected_scope:
        return (
            f"{request_id}:scope_mismatch:{name}:expected_{expected_scope}:got_{raw_scope}"
        )

    resolved_scope = raw_scope or expected_scope
    if resolved_scope in {"request", "session"} and not str(payload.get("job_id") or "").strip():
        return f"{request_id}:job_id_required:{name}:{resolved_scope}"
    if resolved_scope == "user" and not str(payload.get("user_id") or "").strip():
        return f"{request_id}:user_id_required:{name}"
    if resolved_scope == "project" and not str(payload.get("project_id") or "").strip():
        return f"{request_id}:project_id_required:{name}"
    return None


def _task_intent_inference_for_task(
    task: models.TaskCreate,
    *,
    goal_text: str = "",
) -> intent_contract.TaskIntentInference:
    inference = intent_contract.infer_task_intent_for_task_with_metadata(
        explicit_intent=task.intent,
        description=task.description,
        instruction=task.instruction,
        acceptance_criteria=task.acceptance_criteria,
        goal_text=goal_text,
    )
    inferred = inference.intent
    source = inference.source
    confidence = float(inference.confidence)
    if inferred == models.ToolIntent.generate.value and not task.intent:
        unique_tool_intents = {
            TOOL_INTENTS_BY_NAME[request_id]
            for request_id in _task_request_ids_for_preflight(task)
            if request_id in TOOL_INTENTS_BY_NAME
        }
        if len(unique_tool_intents) == 1:
            inferred = next(iter(unique_tool_intents)).value
            source = "tool_intent"
            confidence = 0.9
    return intent_contract.TaskIntentInference(
        intent=inferred,
        source=source,
        confidence=max(0.0, min(1.0, confidence)),
    )


def _preflight_task_intent(task: models.TaskCreate, *, goal_text: str = "") -> str:
    return _task_intent_inference_for_task(task, goal_text=goal_text).intent


def _goal_intent_segments_for_preflight(
    *,
    normalized_intent_envelope: workflow_contracts.NormalizedIntentEnvelope
    | Mapping[str, Any]
    | None = None,
    goal_intent_graph: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    graph: workflow_contracts.IntentGraph | None = None
    envelope = workflow_contracts.parse_normalized_intent_envelope(normalized_intent_envelope)
    if envelope is not None and envelope.graph.segments:
        graph = envelope.graph
    if graph is None:
        graph = workflow_contracts.parse_intent_graph(goal_intent_graph)
    if graph is None:
        return []
    segments: list[dict[str, Any]] = []
    for raw_segment in graph.segments:
        normalized = _normalize_task_intent_profile_segment(
            raw_segment.model_dump(mode="json", exclude_none=True)
        )
        if normalized is not None:
            segments.append(normalized)
    return segments


def _select_goal_intent_segment_for_task(
    *,
    task: models.TaskCreate,
    task_index: int,
    task_intent: str,
    goal_intent_segments: Sequence[dict[str, Any]],
    total_tasks: int,
) -> dict[str, Any] | None:
    if not goal_intent_segments:
        return None
    try:
        capability_map = capability_registry.load_capability_registry().enabled_capabilities()
    except Exception:  # noqa: BLE001
        capability_map = {}
    has_suggested_capabilities = any(
        isinstance(segment.get("suggested_capabilities"), list)
        and bool(segment.get("suggested_capabilities"))
        for segment in goal_intent_segments
    )
    task_requests = {
        str(name).strip().lower()
        for name in _task_request_ids_for_preflight(task)
        if str(name).strip()
    }
    if task_requests:
        matching: list[dict[str, Any]] = []
        for segment in goal_intent_segments:
            suggested = segment.get("suggested_capabilities")
            if not isinstance(suggested, list):
                continue
            suggested_ids = {str(item).strip().lower() for item in suggested if str(item).strip()}
            expanded_ids = set(suggested_ids)
            for capability_id in suggested_ids:
                capability = capability_map.get(capability_id)
                if capability is None:
                    continue
                for adapter in capability.adapters:
                    tool_name = str(adapter.tool_name or "").strip().lower()
                    if tool_name:
                        expanded_ids.add(tool_name)
            if expanded_ids & task_requests:
                matching.append(segment)
        if matching:
            if task_intent:
                for segment in matching:
                    if intent_contract.normalize_task_intent(segment.get("intent")) == task_intent:
                        return segment
                return None
            return matching[0]
    if has_suggested_capabilities:
        if len(goal_intent_segments) == 1:
            only_segment = goal_intent_segments[0]
            if intent_contract.normalize_task_intent(only_segment.get("intent")) == task_intent:
                return only_segment
        return None
    for segment in goal_intent_segments:
        if intent_contract.normalize_task_intent(segment.get("intent")) == task_intent:
            return segment
    if len(goal_intent_segments) == total_tasks and task_index < len(goal_intent_segments):
        return goal_intent_segments[task_index]
    return None


def _preflight_error_diagnostic(task_name: str, message: str) -> dict[str, Any]:
    normalized = str(message or "").strip()
    diagnostic: dict[str, Any] = {
        "severity": "error",
        "code": "preflight_error",
        "field": task_name,
        "message": normalized or "preflight validation failed",
    }
    if not normalized:
        return diagnostic

    if normalized.startswith("intent_segment_invalid:"):
        remainder = normalized[len("intent_segment_invalid:") :]
        tool_name = ""
        detail = remainder
        first_sep = remainder.find(":")
        if first_sep >= 0:
            tool_name = remainder[:first_sep].strip()
            detail = remainder[first_sep + 1 :]
        if detail.startswith(f"{task_name}:"):
            detail = detail[len(task_name) + 1 :]
        if detail.startswith("must_have_inputs_missing:"):
            missing = detail.split(":", 1)[1].strip()
            missing_fields = [entry for entry in missing.split(",") if entry]
            diagnostic["code"] = "intent_segment.must_have_inputs_missing"
            if missing_fields:
                diagnostic["slot_fields"] = missing_fields
            target = tool_name or "tool"
            diagnostic["message"] = (
                f"{target} is missing required intent inputs: {missing or 'unknown'}."
            )
            return diagnostic
        if detail.startswith("output_format_mismatch:"):
            diagnostic["code"] = "intent_segment.output_format_mismatch"
            target = tool_name or "tool"
            diagnostic["message"] = (
                f"{target} output format does not match intent segment constraints."
            )
            return diagnostic
        if detail.startswith("risk_level_mismatch:"):
            diagnostic["code"] = "intent_segment.risk_level_mismatch"
            target = tool_name or "tool"
            diagnostic["message"] = (
                f"{target} risk tier exceeds intent segment risk constraints."
            )
            return diagnostic
        if detail.startswith("segment_intent_mismatch:"):
            diagnostic["code"] = "intent_segment.intent_mismatch"
            target = tool_name or "tool"
            diagnostic["message"] = (
                f"{target} task intent does not match the assigned intent segment."
            )
            return diagnostic
        diagnostic["code"] = "intent_segment.invalid"
        return diagnostic

    if normalized.startswith("tool_intent_mismatch:"):
        diagnostic["code"] = "tool_intent_mismatch"
        return diagnostic
    if normalized.startswith("task_intent_mismatch:"):
        diagnostic["code"] = "task_intent_mismatch"
        return diagnostic
    if normalized.startswith("planner_request_language_invalid:"):
        parts = normalized.split(":")
        requested_id = parts[1] if len(parts) > 1 else ""
        canonical_id = parts[3] if len(parts) > 3 else ""
        diagnostic["code"] = "planner_request_language_invalid"
        diagnostic["message"] = (
            f"Planner tasks must use canonical capability IDs. Use "
            f"'{canonical_id or 'a canonical capability ID'}' instead of "
            f"'{requested_id or 'the raw request ID'}'."
        )
        if requested_id:
            diagnostic["request_id"] = requested_id
        if canonical_id:
            diagnostic["canonical_request_id"] = canonical_id
        return diagnostic
    if normalized.startswith("planner_request_capability_disabled:"):
        capability_id = normalized.split(":", 1)[1].strip()
        diagnostic["code"] = "planner_request_capability_disabled"
        diagnostic["message"] = f"Capability '{capability_id}' is disabled."
        if capability_id:
            diagnostic["capability_id"] = capability_id
        return diagnostic
    if normalized.startswith("planner_request_capability_not_allowed:"):
        capability_id = normalized.split(":", 1)[1].strip()
        diagnostic["code"] = "planner_request_capability_not_allowed"
        diagnostic["message"] = f"Capability '{capability_id}' is not allowed for planning."
        if capability_id:
            diagnostic["capability_id"] = capability_id
        return diagnostic
    if normalized.startswith("planner_request_runtime_tool_not_allowed:"):
        request_id = normalized.split(":", 1)[1].strip()
        diagnostic["code"] = "planner_request_runtime_tool_not_allowed"
        diagnostic["message"] = (
            f"Runtime tool '{request_id}' cannot appear in planner-visible requests."
        )
        if request_id:
            diagnostic["request_id"] = request_id
        return diagnostic
    if normalized.startswith("planner_request_unknown_capability:"):
        request_id = normalized.split(":", 1)[1].strip()
        diagnostic["code"] = "planner_request_unknown_capability"
        diagnostic["message"] = f"Unknown capability '{request_id}'."
        if request_id:
            diagnostic["request_id"] = request_id
        return diagnostic
    if normalized.startswith("render_path_explicit_required:"):
        diagnostic["code"] = "render_path_explicit_required"
        diagnostic["message"] = (
            "Render tasks in raw API and automation flows require a caller-provided path."
        )
        return diagnostic
    if normalized.startswith("render_path_derived_not_allowed:"):
        diagnostic["code"] = "render_path_derived_not_allowed"
        diagnostic["message"] = (
            "Render tasks in raw API and automation flows cannot derive paths from prior task output."
        )
        return diagnostic
    if normalized.startswith("runtime conformance failed for capability"):
        diagnostic["code"] = "runtime_conformance_failed"
        return diagnostic
    if normalized.startswith("capability_inputs_invalid:"):
        diagnostic["code"] = "capability_inputs_invalid"
        return diagnostic
    if normalized.startswith("tool_inputs_invalid:"):
        diagnostic["code"] = "tool_inputs_invalid"
        return diagnostic
    if normalized.startswith("input reference resolution failed:"):
        diagnostic["code"] = "tool_input_reference_invalid"
        return diagnostic
    return diagnostic


def _preflight_error_diagnostics(errors: Mapping[str, str]) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    for task_name, message in errors.items():
        diagnostics.append(_preflight_error_diagnostic(str(task_name), str(message)))
    return diagnostics


def _task_intent_summary(
    tasks: list[models.TaskCreate],
    *,
    goal_text: str = "",
) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for task in tasks:
        inference = _task_intent_inference_for_task(task, goal_text=goal_text)
        summary[task.name] = {
            "intent": inference.intent,
            "source": inference.source,
            "confidence": round(float(inference.confidence), 3),
        }
    return summary


def _preflight_capability_intent_mismatch(
    task_intent: str,
    capability_id: str,
    capability_spec: capability_registry.CapabilitySpec | None,
) -> str | None:
    if capability_spec is None:
        return None
    hints = capability_spec.planner_hints if isinstance(capability_spec.planner_hints, dict) else {}
    raw_allowed = hints.get("task_intents")
    if not isinstance(raw_allowed, list) or not raw_allowed:
        return None
    allowed = {
        normalized
        for item in raw_allowed
        for normalized in [intent_contract.normalize_task_intent(item)]
        if normalized
    }
    if not allowed:
        return None
    if task_intent in allowed:
        return None
    return f"task_intent_mismatch:{capability_id}:{task_intent}:allowed={','.join(sorted(allowed))}"


def _synthesize_preflight_github_repo_query(
    *,
    task_payload: Mapping[str, Any] | None,
    segment_payload: Mapping[str, Any] | None,
    job_context: Mapping[str, Any] | None,
) -> str | None:
    def _read_str(source: Mapping[str, Any] | None, *keys: str) -> str | None:
        if not isinstance(source, Mapping):
            return None
        for key in keys:
            value = source.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    raw_tool_inputs = None
    if isinstance(task_payload, Mapping):
        tool_inputs = task_payload.get("tool_inputs")
        if isinstance(tool_inputs, Mapping):
            entry = tool_inputs.get("github.repo.list")
            if isinstance(entry, Mapping):
                raw_tool_inputs = entry
    owner = (
        _read_str(raw_tool_inputs, "owner", "repo_owner")
        or _read_str(segment_payload, "owner", "repo_owner")
        or _read_str(job_context, "owner", "repo_owner")
    )
    repo = (
        _read_str(raw_tool_inputs, "repo", "repo_name")
        or _read_str(segment_payload, "repo", "repo_name")
        or _read_str(job_context, "repo", "repo_name")
    )
    if owner and repo:
        return f"repo:{repo} owner:{owner}"
    return (
        _read_str(raw_tool_inputs, "query", "github_query")
        or _read_str(segment_payload, "query", "github_query")
        or _read_str(job_context, "query", "github_query")
    )


def _store_task_output(task_id: str, outputs: dict[str, Any]) -> None:
    try:
        redis_client.set(f"{TASK_OUTPUT_KEY_PREFIX}{task_id}", json.dumps(outputs))
    except Exception:
        return


def _load_task_output(task_id: str) -> dict[str, Any]:
    durable_result = _load_task_result_from_postgres(task_id)
    durable_outputs = durable_result.get("outputs")
    if isinstance(durable_outputs, dict):
        return durable_outputs
    try:
        raw = redis_client.get(f"{TASK_OUTPUT_KEY_PREFIX}{task_id}")
        if not raw:
            return {}
        return json.loads(raw)
    except Exception:
        return {}


def _json_safe_result_payload(result: Mapping[str, Any] | dict[str, Any]) -> dict[str, Any]:
    try:
        normalized = json.loads(json.dumps(dict(result), default=str))
    except Exception:
        normalized = dict(result)
    return normalized if isinstance(normalized, dict) else {}


def _persist_task_result_to_postgres(task_id: str, result: Mapping[str, Any]) -> None:
    normalized_result = _json_safe_result_payload(result)
    now = _utcnow()
    try:
        with SessionLocal() as db:
            record = (
                db.query(TaskResultRecord)
                .filter(TaskResultRecord.task_id == task_id)
                .first()
            )
            task = db.query(TaskRecord).filter(TaskRecord.id == task_id).first()
            job_id = (
                task.job_id
                if task is not None
                else str(normalized_result.get("job_id") or "").strip() or None
            )
            plan_id = task.plan_id if task is not None else None
            latest_error = _extract_error_from_task_result(normalized_result)
            status = str(normalized_result.get("status") or "").strip()
            if record is None:
                db.add(
                    TaskResultRecord(
                        task_id=task_id,
                        job_id=job_id,
                        plan_id=plan_id,
                        status=status,
                        result_json=normalized_result,
                        latest_error=latest_error,
                        created_at=now,
                        updated_at=now,
                    )
                )
            else:
                record.job_id = job_id
                record.plan_id = plan_id
                record.status = status
                record.result_json = normalized_result
                record.latest_error = latest_error
                record.updated_at = now
            db.commit()
    except Exception:
        return


def _load_task_result_from_postgres(task_id: str) -> dict[str, Any]:
    try:
        with SessionLocal() as db:
            record = (
                db.query(TaskResultRecord)
                .filter(TaskResultRecord.task_id == task_id)
                .first()
            )
            if record is None or not isinstance(record.result_json, dict):
                return {}
            return dict(record.result_json)
    except Exception:
        return {}


def _store_task_result(task_id: str, result: dict[str, Any]) -> None:
    normalized_result = _json_safe_result_payload(result)
    _persist_task_result_to_postgres(task_id, normalized_result)
    try:
        redis_client.set(
            f"{TASK_RESULT_KEY_PREFIX}{task_id}",
            json.dumps(normalized_result),
        )
    except Exception:
        return


def _load_task_result(task_id: str) -> dict[str, Any]:
    durable_result = _load_task_result_from_postgres(task_id)
    if durable_result:
        return durable_result
    try:
        raw = redis_client.get(f"{TASK_RESULT_KEY_PREFIX}{task_id}")
        if not raw:
            return {}
        return json.loads(raw)
    except Exception:
        return {}


def _alias_dependency_output_keys(task: models.Task | TaskRecord | None, output: Any) -> dict[str, Any]:
    if not isinstance(output, Mapping):
        return {}
    aliased = dict(output)
    if task is None:
        return aliased
    tool_requests = [
        str(request_id).strip()
        for request_id in (task.tool_requests or [])
        if str(request_id).strip()
    ]
    if not tool_requests:
        return aliased
    raw_tool_inputs = task.tool_inputs if isinstance(task.tool_inputs, Mapping) else {}
    bindings = _task_capability_bindings(tool_requests, raw_tool_inputs)
    for request_id, binding in bindings.items():
        aliases = {
            request_id,
            str(binding.get("capability_id") or "").strip(),
            str(binding.get("tool_name") or "").strip(),
        }
        aliases = {alias for alias in aliases if alias}
        if not aliases:
            continue
        value = next((aliased.get(alias) for alias in aliases if alias in aliased), None)
        if value is None:
            continue
        for alias in aliases:
            aliased.setdefault(alias, value)
    return aliased


def _build_task_context(
    task_id: str,
    task_map: dict[str, models.Task],
    id_to_name: dict[str, str],
    job_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    task = task_map.get(task_id)
    if not task:
        return {}
    deps = task.deps or []
    visited: set[str] = set()
    stack: list[str] = list(deps)
    while stack:
        dep_id = stack.pop()
        if dep_id in visited:
            continue
        visited.add(dep_id)
        dep_task = task_map.get(dep_id)
        if dep_task and dep_task.deps:
            for child in dep_task.deps:
                if child not in visited:
                    stack.append(child)
    outputs_by_id = {
        dep_id: _alias_dependency_output_keys(task_map.get(dep_id), _load_task_output(dep_id))
        for dep_id in visited
    }
    outputs_by_name = {
        id_to_name.get(dep_id, dep_id): output for dep_id, output in outputs_by_id.items()
    }
    context: dict[str, Any] = {
        "dependencies": outputs_by_id,
        "dependencies_by_name": outputs_by_name,
    }
    if isinstance(job_context, dict) and job_context:
        context["job_context"] = job_context
    return context


def _record_intent_confidence_outcome(job: JobRecord, status: models.JobStatus) -> None:
    metadata = dict(job.metadata_json) if isinstance(job.metadata_json, dict) else {}
    if not isinstance(metadata, dict):
        return
    if metadata.get("intent_confidence_outcome_recorded"):
        return
    normalized_envelope = _normalized_intent_envelope_from_metadata(metadata, goal=job.goal)
    if normalized_envelope is None:
        return
    profile = normalized_envelope.profile
    intent = intent_contract.normalize_task_intent(profile.intent) or "generate"
    risk_level = _normalize_risk_level(profile.risk_level)
    try:
        confidence = float(profile.confidence)
    except (TypeError, ValueError):
        return
    threshold_raw = profile.threshold
    try:
        threshold = float(threshold_raw)
    except (TypeError, ValueError):
        threshold = _resolve_intent_confidence_threshold(intent, risk_level)
    threshold = max(0.0, min(1.0, threshold))
    bounded_confidence = max(0.0, min(1.0, confidence))
    outcome = "failed"
    if status == models.JobStatus.succeeded:
        outcome = "succeeded"
    elif status == models.JobStatus.canceled:
        outcome = "canceled"
    above_threshold = bounded_confidence >= threshold
    intent_confidence_outcomes_total.labels(
        intent=intent,
        risk_level=risk_level,
        outcome=outcome,
        above_threshold=str(bool(above_threshold)).lower(),
        confidence_bucket=_confidence_bucket(bounded_confidence),
    ).inc()
    metadata["intent_confidence_outcome_recorded"] = True
    metadata["intent_confidence_outcome"] = {
        "outcome": outcome,
        "intent": intent,
        "risk_level": risk_level,
        "confidence": round(bounded_confidence, 3),
        "threshold": round(threshold, 3),
        "above_threshold": above_threshold,
        "recorded_at": _utcnow().isoformat(),
    }
    job.metadata_json = metadata


def _refresh_job_status(job_id: str) -> None:
    with SessionLocal() as db:
        job = db.query(JobRecord).filter(JobRecord.id == job_id).first()
        if not job:
            return
        tasks = db.query(TaskRecord).filter(TaskRecord.job_id == job_id).all()
        if not tasks:
            return
        statuses = {task.status for task in tasks}
        now = _utcnow()
        if models.TaskStatus.failed.value in statuses:
            next_status = models.JobStatus.failed
        elif statuses.issubset(
            {models.TaskStatus.completed.value, models.TaskStatus.accepted.value}
        ):
            next_status = models.JobStatus.succeeded
        elif (
            models.TaskStatus.ready.value in statuses or models.TaskStatus.running.value in statuses
        ):
            next_status = models.JobStatus.running
        else:
            next_status = models.JobStatus.planning
        _set_job_status(job, next_status)
        if next_status in {
            models.JobStatus.succeeded,
            models.JobStatus.failed,
            models.JobStatus.canceled,
        }:
            _record_intent_confidence_outcome(job, next_status)
            _persist_intent_workflow_memory(
                db,
                job=job,
                tasks=tasks,
                status=next_status,
            )
            _deliver_chat_workflow_terminal_message(
                db,
                job=job,
                tasks=tasks,
                status=next_status,
            )
        job.updated_at = now
        db.commit()


def _set_job_status(job: JobRecord, status: models.JobStatus) -> None:
    current = models.JobStatus(job.status)
    if state_machine.validate_job_transition(current, status):
        job.status = status.value


def _deliver_chat_workflow_terminal_message(
    db: Session,
    *,
    job: JobRecord,
    tasks: Sequence[TaskRecord],
    status: models.JobStatus,
) -> None:
    if status not in {
        models.JobStatus.succeeded,
        models.JobStatus.failed,
        models.JobStatus.canceled,
    }:
        return
    metadata = dict(job.metadata_json) if isinstance(job.metadata_json, dict) else {}
    if metadata.get("chat_workflow_terminal_message_id"):
        return
    workflow_run = (
        db.query(WorkflowRunRecord)
        .filter(WorkflowRunRecord.job_id == job.id)
        .order_by(WorkflowRunRecord.created_at.desc())
        .first()
    )
    if workflow_run is None:
        return
    run_metadata = dict(workflow_run.metadata_json or {})
    chat_session_id = str(run_metadata.get("chat_session_id") or "").strip()
    if not chat_session_id:
        return
    session = db.query(ChatSessionRecord).filter(ChatSessionRecord.id == chat_session_id).first()
    if session is None:
        return
    version = (
        db.query(WorkflowVersionRecord)
        .filter(WorkflowVersionRecord.id == workflow_run.version_id)
        .first()
    )
    content = _build_chat_workflow_terminal_message(
        job=job,
        tasks=tasks,
        workflow_run=workflow_run,
        workflow_version=version,
        status=status,
    )
    if not content:
        return
    now = _utcnow()
    workflow_run_model = _workflow_run_from_record(workflow_run, job_record=job)
    message = ChatMessageRecord(
        id=str(uuid.uuid4()),
        session_id=session.id,
        role=chat_contracts.ChatRole.assistant.value,
        content=content,
        metadata_json={
            "workflow_run": workflow_run_model.model_dump(mode="json", exclude_none=True),
            "workflow_delivery": {
                "source": "workflow_terminal_message",
                "job_status": status.value,
            },
        },
        action_json=None,
        job_id=job.id,
        created_at=now,
    )
    db.add(message)
    session_metadata = dict(session.metadata_json or {})
    if str(session_metadata.get("active_job_id") or "").strip() == job.id:
        session_metadata.pop("active_job_id", None)
    if str(session_metadata.get("active_workflow_run_id") or "").strip() == workflow_run.id:
        session_metadata.pop("active_workflow_run_id", None)
    session.metadata_json = session_metadata
    session.updated_at = now
    metadata["chat_workflow_terminal_message_at"] = now.isoformat()
    metadata["chat_workflow_terminal_message_id"] = message.id
    job.metadata_json = metadata


def _build_chat_workflow_terminal_message(
    *,
    job: JobRecord,
    tasks: Sequence[TaskRecord],
    workflow_run: WorkflowRunRecord,
    workflow_version: WorkflowVersionRecord | None,
    status: models.JobStatus,
) -> str:
    workflow_title = str(workflow_run.title or workflow_run.goal or "Workflow").strip() or "Workflow"
    if status == models.JobStatus.succeeded:
        payload = _resolve_chat_workflow_success_payload(
            job=job,
            tasks=tasks,
            workflow_version=workflow_version,
        )
        rendered = _render_chat_workflow_payload(payload)
        if rendered:
            return rendered
        return f"Workflow `{workflow_title}` completed."
    if status == models.JobStatus.canceled:
        return f"Workflow `{workflow_title}` was canceled."
    failure_detail = _chat_workflow_failure_detail(tasks)
    if failure_detail:
        return f"Workflow `{workflow_title}` failed.\n\n{failure_detail}"
    return f"Workflow `{workflow_title}` failed."


def _resolve_chat_workflow_success_payload(
    *,
    job: JobRecord,
    tasks: Sequence[TaskRecord],
    workflow_version: WorkflowVersionRecord | None,
) -> Any:
    workflow_outputs = _resolve_chat_workflow_interface_outputs(
        job=job,
        tasks=tasks,
        workflow_version=workflow_version,
    )
    if workflow_outputs:
        if len(workflow_outputs) == 1:
            return next(iter(workflow_outputs.values()))
        return workflow_outputs
    ordered_tasks = sorted(
        tasks,
        key=lambda task: (task.created_at or datetime.min.replace(tzinfo=UTC), task.id),
    )
    for task in reversed(ordered_tasks):
        task_result = _load_task_result(task.id)
        preferred = _extract_chat_workflow_preferred_payload(task_result.get("outputs"))
        if preferred is not None:
            return preferred
        tool_calls = task_result.get("tool_calls")
        if isinstance(tool_calls, list):
            for call in reversed(tool_calls):
                if not isinstance(call, Mapping):
                    continue
                preferred = _extract_chat_workflow_preferred_payload(call.get("output_or_error"))
                if preferred is not None:
                    return preferred
    return None


def _resolve_chat_workflow_interface_outputs(
    *,
    job: JobRecord,
    tasks: Sequence[TaskRecord],
    workflow_version: WorkflowVersionRecord | None,
) -> dict[str, Any]:
    if workflow_version is None or not isinstance(workflow_version.draft_json, dict):
        return {}
    workflow_interface, workflow_interface_errors, _workflow_interface_warnings = (
        _coerce_workflow_interface(
            workflow_version.draft_json.get("workflowInterface")
            if "workflowInterface" in workflow_version.draft_json
            else workflow_version.draft_json.get("workflow_interface")
        )
    )
    if workflow_interface_errors:
        return {}
    output_defs = workflow_interface.get("outputs", [])
    if not isinstance(output_defs, list) or not output_defs:
        return {}
    job_context = job.context_json if isinstance(job.context_json, dict) else {}
    workflow_payload = (
        dict(job_context.get("workflow"))
        if isinstance(job_context.get("workflow"), Mapping)
        else {}
    )
    task_outputs_by_name = {
        task.name: _load_task_output(task.id)
        for task in sorted(tasks, key=lambda item: (item.created_at, item.id))
    }
    task_names_by_node_id = _workflow_task_names_by_node_id(workflow_version.draft_json)
    resolved: dict[str, Any] = {}
    for output_def in output_defs:
        if not isinstance(output_def, Mapping):
            continue
        key = str(output_def.get("key") or "").strip()
        if not key:
            continue
        value = _resolve_chat_workflow_output_binding(
            output_def.get("binding"),
            job_context=job_context,
            workflow_payload=workflow_payload,
            task_outputs_by_name=task_outputs_by_name,
            task_names_by_node_id=task_names_by_node_id,
        )
        if value is not None:
            resolved[key] = value
    return resolved


def _resolve_chat_workflow_output_binding(
    binding: Any,
    *,
    job_context: Mapping[str, Any],
    workflow_payload: Mapping[str, Any],
    task_outputs_by_name: Mapping[str, Any],
    task_names_by_node_id: Mapping[str, str],
) -> Any:
    if not isinstance(binding, Mapping):
        return None
    kind = str(binding.get("kind") or "").strip()
    if kind == "literal":
        return binding.get("value")
    if kind == "context":
        return _get_nested_value(job_context, _split_reference_path(str(binding.get("path") or "")))
    if kind == "workflow_input":
        key = str(binding.get("input_key") or "").strip()
        inputs = workflow_payload.get("inputs")
        return inputs.get(key) if isinstance(inputs, Mapping) else None
    if kind == "workflow_variable":
        key = str(binding.get("variable_key") or "").strip()
        variables = workflow_payload.get("variables")
        return variables.get(key) if isinstance(variables, Mapping) else None
    if kind == "step_output":
        source_node_id = str(binding.get("source_node_id") or "").strip()
        source_path = str(binding.get("source_path") or "").strip()
        task_name = task_names_by_node_id.get(source_node_id)
        if not task_name or not source_path:
            return None
        task_output = task_outputs_by_name.get(task_name)
        if not isinstance(task_output, Mapping):
            return None
        return _get_nested_value(task_output, _split_reference_path(source_path))
    return None


def _workflow_task_names_by_node_id(draft_json: Mapping[str, Any]) -> dict[str, str]:
    raw_nodes = draft_json.get("nodes")
    if not isinstance(raw_nodes, list):
        return {}
    task_names: dict[str, str] = {}
    for index, raw_node in enumerate(raw_nodes):
        if not isinstance(raw_node, Mapping):
            continue
        node_id = str(raw_node.get("id") or "").strip()
        if not node_id:
            continue
        capability_id = str(
            raw_node.get("capabilityId") or raw_node.get("capability_id") or ""
        ).strip()
        task_name = str(raw_node.get("taskName") or raw_node.get("task_name") or "").strip()
        if not task_name:
            task_name = _composer_default_task_name(capability_id, index)
        task_names[node_id] = task_name
    return task_names


def _extract_chat_workflow_preferred_payload(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip()
        return normalized or None
    if isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, Mapping):
        for key in ("text", "content", "markdown", "message", "summary", "result"):
            if key in value:
                preferred = _extract_chat_workflow_preferred_payload(value.get(key))
                if preferred is not None:
                    return preferred
        if len(value) == 1:
            only_value = next(iter(value.values()))
            preferred = _extract_chat_workflow_preferred_payload(only_value)
            if preferred is not None:
                return preferred
        return value
    if isinstance(value, list):
        if not value:
            return None
        if len(value) == 1:
            preferred = _extract_chat_workflow_preferred_payload(value[0])
            if preferred is not None:
                return preferred
        return value
    return str(value)


def _render_chat_workflow_payload(value: Any, *, max_chars: int = 4000) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return _truncate_chat_workflow_text(value.strip(), max_chars=max_chars)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, Mapping):
        preferred = _extract_chat_workflow_preferred_payload(value)
        if preferred is not None and preferred is not value:
            return _render_chat_workflow_payload(preferred, max_chars=max_chars)
        items = value.get("items")
        if isinstance(items, list) and items:
            lines: list[str] = []
            for item in items[:10]:
                if isinstance(item, Mapping):
                    label = (
                        str(
                            item.get("text")
                            or item.get("title")
                            or item.get("name")
                            or item.get("path")
                            or item.get("id")
                            or ""
                        ).strip()
                    )
                    if not label:
                        label = json.dumps(dict(item), ensure_ascii=True)
                else:
                    label = str(item).strip()
                if label:
                    lines.append(f"- {label}")
            if lines:
                return _truncate_chat_workflow_text("\n".join(lines), max_chars=max_chars)
        rendered = json.dumps(dict(value), ensure_ascii=True, indent=2)
        return _truncate_chat_workflow_text(rendered, max_chars=max_chars)
    if isinstance(value, list):
        lines = [
            _render_chat_workflow_payload(item, max_chars=max_chars // 2).strip()
            for item in value[:10]
        ]
        lines = [line for line in lines if line]
        if not lines:
            return ""
        if all("\n" not in line for line in lines):
            return _truncate_chat_workflow_text(
                "\n".join(f"- {line}" for line in lines),
                max_chars=max_chars,
            )
        rendered = json.dumps(list(value[:10]), ensure_ascii=True, indent=2)
        return _truncate_chat_workflow_text(rendered, max_chars=max_chars)
    return _truncate_chat_workflow_text(str(value), max_chars=max_chars)


def _truncate_chat_workflow_text(value: str, *, max_chars: int) -> str:
    normalized = str(value or "").strip()
    if len(normalized) <= max_chars:
        return normalized
    return f"{normalized[:max_chars].rstrip()}\n\n[truncated]"


def _chat_workflow_failure_detail(tasks: Sequence[TaskRecord]) -> str:
    ordered_tasks = sorted(
        tasks,
        key=lambda task: (task.updated_at or task.created_at, task.id),
    )
    for task in reversed(ordered_tasks):
        task_result = _load_task_result(task.id)
        error = str(task_result.get("error") or "").strip()
        if error:
            return f"Task `{task.name}` failed: {error}"
        outputs = task_result.get("outputs")
        if isinstance(outputs, Mapping):
            tool_error = outputs.get("tool_error")
            if isinstance(tool_error, Mapping):
                message = str(tool_error.get("error") or "").strip()
                if message:
                    return f"Task `{task.name}` failed: {message}"
    return ""


def _parse_task_dlq_entry(stream_id: str, record: Mapping[str, str]) -> models.TaskDlqEntry | None:
    raw = record.get("data")
    if not raw:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    envelope = payload.get("envelope")
    envelope_dict = envelope if isinstance(envelope, dict) else {}
    task_payload = payload.get("task_payload")
    task_payload_dict = task_payload if isinstance(task_payload, dict) else {}
    job_id = payload.get("job_id")
    if not isinstance(job_id, str) or not job_id:
        maybe_job = envelope_dict.get("job_id")
        job_id = maybe_job if isinstance(maybe_job, str) else None
    task_id = payload.get("task_id")
    if not isinstance(task_id, str) or not task_id:
        maybe_task = task_payload_dict.get("task_id")
        task_id = maybe_task if isinstance(maybe_task, str) else None
    error = payload.get("error")
    return models.TaskDlqEntry(
        stream_id=stream_id,
        message_id=str(payload.get("message_id") or stream_id),
        failed_at=payload.get("failed_at") if isinstance(payload.get("failed_at"), str) else None,
        error=error if isinstance(error, str) and error else "unknown_error",
        worker_consumer=payload.get("worker_consumer")
        if isinstance(payload.get("worker_consumer"), str)
        else None,
        job_id=job_id,
        task_id=task_id,
        envelope=envelope_dict,
        task_payload=task_payload_dict,
    )


def _read_task_dlq(job_id: str, limit: int) -> list[models.TaskDlqEntry]:
    scan_count = min(max(limit * 5, limit), 500)
    rows = redis_client.xrevrange(events.TASK_DLQ_STREAM, "+", "-", count=scan_count)
    entries: list[models.TaskDlqEntry] = []
    for stream_id, record in rows:
        parsed = _parse_task_dlq_entry(stream_id, record)
        if parsed is None:
            continue
        if parsed.job_id != job_id:
            continue
        entries.append(parsed)
        if len(entries) >= limit:
            break
    return entries


_TRANSIENT_ERROR_TOKENS = (
    "timed out",
    "timeout",
    "connection refused",
    "remote end closed",
    "session terminated",
    "temporary",
    "service unavailable",
    "too many requests",
    "rate limit",
    "502",
    "503",
    "504",
)

_CONTRACT_ERROR_CODES = {
    "contract.input_invalid",
    "contract.output_invalid",
    "contract.schema_not_found",
    "contract.schema_invalid",
    "contract.tool_not_found",
    "contract.input_missing",
    "contract.intent_mismatch",
    "unknown_tool",
    "memory_only_inputs_missing",
    "tool_intent_mismatch",
}


def _classify_task_error(error: str | None) -> dict[str, Any]:
    if not error:
        return {
            "category": "none",
            "code": "none",
            "retryable": False,
            "message": "",
            "hint": "",
        }
    normalized = error.strip().lower()
    code = normalized.split(":", 1)[0] if normalized else "unknown"
    if code in _CONTRACT_ERROR_CODES:
        return {
            "category": "contract",
            "code": code,
            "retryable": False,
            "message": error,
            "hint": "Fix capability inputs/schema or chaining references before retrying.",
        }
    if "path must be a non-empty string" in normalized:
        return {
            "category": "contract",
            "code": "contract.input_invalid",
            "retryable": False,
            "message": error,
            "hint": "Set a non-empty 'path' input and bind it explicitly from job context, workflow input, or a previous step.",
        }
    if "timeout" in normalized or "timed out" in normalized:
        return {
            "category": "timeout",
            "code": code,
            "retryable": True,
            "message": error,
            "hint": "Retry is safe. If repeated, increase timeout or reduce task scope.",
        }
    if any(token in normalized for token in _TRANSIENT_ERROR_TOKENS):
        return {
            "category": "transient",
            "code": code,
            "retryable": True,
            "message": error,
            "hint": "Retry is recommended. Check dependent service health if it repeats.",
        }
    if "policy" in normalized:
        return {
            "category": "policy",
            "code": code,
            "retryable": False,
            "message": error,
            "hint": "Policy denied this action. Update policy or task intent/tool selection.",
        }
    return {
        "category": "runtime",
        "code": code,
        "retryable": False,
        "message": error,
        "hint": "Inspect tool output/logs and task inputs for root cause.",
    }


def _extract_error_from_task_result(result: dict[str, Any]) -> str | None:
    value = result.get("error")
    if isinstance(value, str) and value:
        return value
    outputs = result.get("outputs")
    if isinstance(outputs, dict):
        tool_error = outputs.get("tool_error")
        if isinstance(tool_error, dict):
            message = tool_error.get("error")
            if isinstance(message, str) and message:
                return message
    return None


def _durable_run_id(job: JobRecord | None, job_id: str) -> str:
    metadata = job.metadata_json if job and isinstance(job.metadata_json, dict) else {}
    workflow_run_id = str(metadata.get("workflow_run_id") or "").strip()
    return workflow_run_id or job_id


def _payload_attempt_number(payload: Mapping[str, Any] | None) -> int | None:
    if not isinstance(payload, Mapping):
        return None
    raw_attempt = payload.get("attempts")
    if isinstance(raw_attempt, bool):
        return None
    try:
        attempt_number = int(raw_attempt)
    except (TypeError, ValueError):
        return None
    return attempt_number if attempt_number > 0 else None


def _parse_event_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)
    if isinstance(value, str) and value.strip():
        normalized = value.strip()
        if normalized.endswith("Z"):
            normalized = normalized[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)
    return None


def _step_attempt_record_id(step_id: str, attempt_number: int) -> str:
    return hashlib.sha1(f"step_attempt:{step_id}:{attempt_number}".encode("utf-8")).hexdigest()


def _run_event_record_id(
    *,
    event_type: str,
    correlation_id: str,
    job_id: str,
    step_id: str | None,
    occurred_at: datetime,
    payload: Mapping[str, Any],
) -> str:
    material = json.dumps(
        {
            "event_type": event_type,
            "correlation_id": correlation_id,
            "job_id": job_id,
            "step_id": step_id,
            "occurred_at": occurred_at.isoformat(),
            "payload": dict(payload),
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha1(material.encode("utf-8")).hexdigest()


def _invocation_record_id(
    step_attempt_id: str,
    index: int,
    request_id: str | None,
    tool_call: Mapping[str, Any],
) -> str:
    material = json.dumps(
        {
            "step_attempt_id": step_attempt_id,
            "index": index,
            "request_id": request_id,
            "tool_name": tool_call.get("tool_name"),
            "idempotency_key": tool_call.get("idempotency_key"),
            "trace_id": tool_call.get("trace_id"),
            "started_at": tool_call.get("started_at"),
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha1(material.encode("utf-8")).hexdigest()


def _task_error_from_payload(payload: Mapping[str, Any] | None) -> str | None:
    if not isinstance(payload, Mapping):
        return None
    error = payload.get("error")
    if isinstance(error, str) and error:
        return error
    outputs = payload.get("outputs")
    if isinstance(outputs, Mapping):
        tool_error = outputs.get("tool_error")
        if isinstance(tool_error, Mapping):
            message = tool_error.get("error")
            if isinstance(message, str) and message:
                return message
        validation_error = outputs.get("validation_error")
        if isinstance(validation_error, Mapping):
            message = validation_error.get("error")
            if isinstance(message, str) and message:
                return message
    return None


def _task_error_code_from_payload(
    payload: Mapping[str, Any] | None,
    error_message: str | None,
) -> str | None:
    if isinstance(payload, Mapping):
        outputs = payload.get("outputs")
        if isinstance(outputs, Mapping):
            tool_error = outputs.get("tool_error")
            if isinstance(tool_error, Mapping):
                raw_code = tool_error.get("error_code")
                if isinstance(raw_code, str) and raw_code:
                    return raw_code
            validation_error = outputs.get("validation_error")
            if isinstance(validation_error, Mapping):
                raw_code = validation_error.get("error_code")
                if isinstance(raw_code, str) and raw_code:
                    return raw_code
    if isinstance(error_message, str) and error_message:
        return error_message.split(":", 1)[0]
    return None


def _retry_classification_for_error(error_message: str | None) -> str | None:
    if not error_message:
        return "succeeded"
    classification = _classify_task_error(error_message)
    if classification.get("retryable"):
        return "retryable"
    return f"terminal_{classification.get('category', 'runtime')}"


def _build_step_attempt_result_summary(
    payload: Mapping[str, Any] | None,
    error_message: str | None,
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {}
    outputs = payload.get("outputs")
    artifacts = payload.get("artifacts")
    tool_calls = payload.get("tool_calls")
    summary: dict[str, Any] = {
        "status": str(payload.get("status") or ""),
        "output_keys": sorted(outputs.keys()) if isinstance(outputs, Mapping) else [],
        "artifact_count": len(artifacts) if isinstance(artifacts, list) else 0,
        "tool_call_count": len(tool_calls) if isinstance(tool_calls, list) else 0,
    }
    if error_message:
        classification = _classify_task_error(error_message)
        summary["error_category"] = classification.get("category")
        summary["error_code"] = classification.get("code")
    if isinstance(artifacts, list):
        for artifact in artifacts:
            if not isinstance(artifact, Mapping):
                continue
            if artifact.get("type") != "run_scorecard":
                continue
            run_scorecard = artifact.get("summary")
            if isinstance(run_scorecard, Mapping):
                summary["run_scorecard"] = dict(run_scorecard)
                break
    return summary


def _latest_step_attempt_record(db: Session, step_id: str) -> StepAttemptRecord | None:
    return (
        db.query(StepAttemptRecord)
        .filter(StepAttemptRecord.step_id == step_id)
        .order_by(StepAttemptRecord.attempt_number.desc(), StepAttemptRecord.started_at.desc())
        .first()
    )


def _upsert_step_attempt_started(
    db: Session,
    *,
    task: TaskRecord,
    job: JobRecord | None,
    payload: Mapping[str, Any],
    occurred_at: datetime,
) -> StepAttemptRecord:
    attempt_number = _payload_attempt_number(payload) or max(int(task.attempts or 0), 1)
    attempt_id = _step_attempt_record_id(task.id, attempt_number)
    record = db.query(StepAttemptRecord).filter(StepAttemptRecord.id == attempt_id).first()
    if record is None:
        record = StepAttemptRecord(
            id=attempt_id,
            run_id=_durable_run_id(job, task.job_id),
            job_id=task.job_id,
            step_id=task.id,
            attempt_number=attempt_number,
            status=models.TaskStatus.running.value,
            worker_id=payload.get("worker_consumer")
            if isinstance(payload.get("worker_consumer"), str)
            else None,
            started_at=occurred_at,
            finished_at=None,
            error_code=None,
            error_message=None,
            retry_classification=None,
            result_summary_json={},
        )
        db.add(record)
        return record
    record.status = models.TaskStatus.running.value
    record.worker_id = (
        payload.get("worker_consumer")
        if isinstance(payload.get("worker_consumer"), str)
        else record.worker_id
    )
    if record.started_at is None or occurred_at < record.started_at:
        record.started_at = occurred_at
    return record


def _upsert_step_attempt_finished(
    db: Session,
    *,
    task: TaskRecord,
    job: JobRecord | None,
    payload: Mapping[str, Any],
    status: str,
    occurred_at: datetime,
) -> StepAttemptRecord:
    latest_attempt = _latest_step_attempt_record(db, task.id)
    attempt_number = _payload_attempt_number(payload)
    if attempt_number is None and latest_attempt is not None:
        attempt_number = latest_attempt.attempt_number
    if attempt_number is None:
        attempt_number = max(int(task.attempts or 0), 1)
    attempt_id = _step_attempt_record_id(task.id, attempt_number)
    record = db.query(StepAttemptRecord).filter(StepAttemptRecord.id == attempt_id).first()
    started_at = _parse_event_datetime(payload.get("started_at")) or occurred_at
    if record is None:
        record = StepAttemptRecord(
            id=attempt_id,
            run_id=_durable_run_id(job, task.job_id),
            job_id=task.job_id,
            step_id=task.id,
            attempt_number=attempt_number,
            status=status,
            worker_id=payload.get("worker_consumer")
            if isinstance(payload.get("worker_consumer"), str)
            else None,
            started_at=started_at,
            finished_at=None,
            error_code=None,
            error_message=None,
            retry_classification=None,
            result_summary_json={},
        )
        db.add(record)
    error_message = _task_error_from_payload(payload)
    record.status = status
    record.worker_id = (
        payload.get("worker_consumer")
        if isinstance(payload.get("worker_consumer"), str)
        else record.worker_id
    )
    record.started_at = record.started_at or started_at
    record.finished_at = _parse_event_datetime(payload.get("finished_at")) or occurred_at
    record.error_message = error_message
    record.error_code = _task_error_code_from_payload(payload, error_message)
    record.retry_classification = _retry_classification_for_error(error_message)
    record.result_summary_json = _build_step_attempt_result_summary(payload, error_message)
    return record


def _update_latest_step_attempt_status(
    db: Session,
    *,
    task: TaskRecord,
    status: str,
) -> StepAttemptRecord | None:
    record = _latest_step_attempt_record(db, task.id)
    if record is None:
        return None
    record.status = status
    if status in {
        models.TaskStatus.accepted.value,
        models.TaskStatus.rework_requested.value,
    }:
        record.finished_at = record.finished_at or _utcnow()
    return record


def _binding_adapter_id(binding: Mapping[str, Any] | None) -> str | None:
    if not isinstance(binding, Mapping):
        return None
    parts = [
        str(binding.get("adapter_type") or "").strip(),
        str(binding.get("server_id") or "").strip(),
        str(binding.get("tool_name") or "").strip(),
    ]
    normalized = [part for part in parts if part]
    return ":".join(normalized) if normalized else None


def _replace_attempt_invocations(
    db: Session,
    *,
    attempt: StepAttemptRecord,
    task: TaskRecord,
    payload: Mapping[str, Any],
) -> None:
    db.query(InvocationRecord).filter(
        InvocationRecord.step_attempt_id == attempt.id
    ).delete(synchronize_session=False)
    tool_calls = payload.get("tool_calls")
    if not isinstance(tool_calls, list):
        return
    bindings = _task_capability_bindings(
        task.tool_requests or [],
        task.tool_inputs if isinstance(task.tool_inputs, dict) else {},
    )
    for index, raw_call in enumerate(tool_calls):
        if not isinstance(raw_call, Mapping):
            continue
        request_id = raw_call.get("request_id")
        if not isinstance(request_id, str) or not request_id.strip():
            request_id = task.tool_requests[index] if index < len(task.tool_requests or []) else None
        binding = bindings.get(request_id) if isinstance(request_id, str) else None
        tool_name = str(raw_call.get("tool_name") or "").strip()
        capability_id = str(raw_call.get("capability_id") or "").strip()
        if not capability_id and isinstance(binding, Mapping):
            capability_id = str(binding.get("capability_id") or "").strip()
        capability_id = capability_id or tool_name or str(request_id or "")
        adapter_id = raw_call.get("adapter_id")
        if not isinstance(adapter_id, str) or not adapter_id.strip():
            adapter_id = _binding_adapter_id(binding)
        request_payload = raw_call.get("input")
        response_payload = raw_call.get("output_or_error")
        error_message = None
        error_code = None
        if isinstance(response_payload, Mapping):
            maybe_error = response_payload.get("error")
            if isinstance(maybe_error, str) and maybe_error:
                error_message = maybe_error
            maybe_code = response_payload.get("error_code")
            if isinstance(maybe_code, str) and maybe_code:
                error_code = maybe_code
        if error_code is None and error_message:
            error_code = error_message.split(":", 1)[0]
        db.add(
            InvocationRecord(
                id=_invocation_record_id(attempt.id, index, request_id, raw_call),
                run_id=attempt.run_id,
                job_id=attempt.job_id,
                step_id=attempt.step_id,
                step_attempt_id=attempt.id,
                request_id=request_id if isinstance(request_id, str) else None,
                capability_id=capability_id,
                adapter_id=adapter_id.strip() if isinstance(adapter_id, str) and adapter_id.strip() else None,
                request_json=dict(request_payload) if isinstance(request_payload, Mapping) else {},
                response_json=dict(response_payload) if isinstance(response_payload, Mapping) else {},
                status=str(raw_call.get("status") or ""),
                started_at=_parse_event_datetime(raw_call.get("started_at")) or attempt.started_at,
                finished_at=_parse_event_datetime(raw_call.get("finished_at")),
                error_code=error_code,
                error_message=error_message,
            )
        )


def _resolve_step_attempt_id_for_event(
    db: Session,
    *,
    step_id: str | None,
    payload: Mapping[str, Any],
) -> str | None:
    if not isinstance(step_id, str) or not step_id:
        return None
    attempt_number = _payload_attempt_number(payload)
    if attempt_number is not None:
        attempt_id = _step_attempt_record_id(step_id, attempt_number)
        record = db.query(StepAttemptRecord).filter(StepAttemptRecord.id == attempt_id).first()
        if record is not None:
            return record.id
    latest_attempt = _latest_step_attempt_record(db, step_id)
    return latest_attempt.id if latest_attempt is not None else None


def _persist_run_event(envelope: Mapping[str, Any]) -> None:
    event_type = str(envelope.get("type") or "").strip()
    correlation_id = str(envelope.get("correlation_id") or "").strip()
    payload = envelope.get("payload")
    payload_dict = dict(payload) if isinstance(payload, Mapping) else {}
    step_id = envelope.get("task_id")
    if not isinstance(step_id, str) or not step_id:
        raw_task_id = payload_dict.get("task_id")
        step_id = raw_task_id if isinstance(raw_task_id, str) and raw_task_id else None
    job_id = envelope.get("job_id")
    if not isinstance(job_id, str) or not job_id:
        raw_job_id = payload_dict.get("job_id")
        job_id = raw_job_id if isinstance(raw_job_id, str) and raw_job_id else None
    if not event_type or not correlation_id:
        return
    with SessionLocal() as db:
        task = None
        if isinstance(step_id, str) and step_id:
            task = db.query(TaskRecord).filter(TaskRecord.id == step_id).first()
            if job_id is None and task is not None:
                job_id = task.job_id
        if not isinstance(job_id, str) or not job_id:
            return
        job = db.query(JobRecord).filter(JobRecord.id == job_id).first()
        occurred_at = _parse_event_datetime(envelope.get("occurred_at")) or _utcnow()
        step_attempt_id = _resolve_step_attempt_id_for_event(
            db,
            step_id=step_id,
            payload=payload_dict,
        )
        record_id = _run_event_record_id(
            event_type=event_type,
            correlation_id=correlation_id,
            job_id=job_id,
            step_id=step_id,
            occurred_at=occurred_at,
            payload=payload_dict,
        )
        record = db.query(RunEventRecord).filter(RunEventRecord.id == record_id).first()
        if record is None:
            record = RunEventRecord(
                id=record_id,
                run_id=_durable_run_id(job, job_id),
                job_id=job_id,
                step_id=step_id,
                step_attempt_id=step_attempt_id,
                event_type=event_type,
                payload_json=payload_dict,
                occurred_at=occurred_at,
            )
            db.add(record)
        else:
            record.step_attempt_id = step_attempt_id
            record.payload_json = payload_dict
            record.occurred_at = occurred_at
        db.commit()


def _step_attempt_from_record(record: StepAttemptRecord) -> models.StepAttempt:
    return models.StepAttempt(
        id=record.id,
        run_id=record.run_id,
        job_id=record.job_id,
        step_id=record.step_id,
        attempt_number=record.attempt_number,
        status=record.status,
        worker_id=record.worker_id,
        started_at=record.started_at,
        finished_at=record.finished_at,
        error_code=record.error_code,
        error_message=record.error_message,
        retry_classification=record.retry_classification,
        result_summary=record.result_summary_json or {},
    )


def _invocation_from_record(record: InvocationRecord) -> models.Invocation:
    return models.Invocation(
        id=record.id,
        run_id=record.run_id,
        job_id=record.job_id,
        step_id=record.step_id,
        step_attempt_id=record.step_attempt_id,
        request_id=record.request_id,
        capability_id=record.capability_id,
        adapter_id=record.adapter_id,
        request=record.request_json or {},
        response=record.response_json or {},
        status=record.status,
        started_at=record.started_at,
        finished_at=record.finished_at,
        error_code=record.error_code,
        error_message=record.error_message,
    )


def _run_event_from_record(record: RunEventRecord) -> models.RunEvent:
    return models.RunEvent(
        id=record.id,
        run_id=record.run_id,
        job_id=record.job_id,
        step_id=record.step_id,
        step_attempt_id=record.step_attempt_id,
        event_type=record.event_type,
        payload=record.payload_json or {},
        occurred_at=record.occurred_at,
    )


def _debugger_timeline_entry_from_run_event(record: RunEventRecord) -> dict[str, Any]:
    payload = record.payload_json if isinstance(record.payload_json, dict) else {}
    status = payload.get("status")
    status_text = status if isinstance(status, str) and status else record.event_type.replace("task.", "")
    return {
        "stream_id": record.id,
        "type": record.event_type,
        "occurred_at": record.occurred_at.isoformat(),
        "job_id": record.job_id,
        "task_id": record.step_id,
        "status": status_text,
        "attempts": payload.get("attempts"),
        "max_attempts": payload.get("max_attempts"),
        "worker_consumer": payload.get("worker_consumer")
        if isinstance(payload.get("worker_consumer"), str)
        else None,
        "run_id": payload.get("run_id") if isinstance(payload.get("run_id"), str) else record.run_id,
        "error": _task_error_from_payload(payload) or "",
    }


def _latest_task_failures_for_jobs(
    db: Session,
    job_ids: Sequence[str],
) -> dict[str, dict[str, Any]]:
    normalized_job_ids = [job_id for job_id in job_ids if isinstance(job_id, str) and job_id]
    if not normalized_job_ids:
        return {}
    task_records = (
        db.query(TaskRecord)
        .filter(TaskRecord.job_id.in_(normalized_job_ids))
        .all()
    )
    task_name_by_id = {
        record.id: record.name
        for record in task_records
        if isinstance(record.id, str) and record.id
    }
    result_records = (
        db.query(TaskResultRecord)
        .filter(TaskResultRecord.job_id.in_(normalized_job_ids))
        .order_by(TaskResultRecord.updated_at.desc())
        .all()
    )
    latest_by_job: dict[str, dict[str, Any]] = {}
    for record in result_records:
        if not isinstance(record.job_id, str) or not record.job_id:
            continue
        error = str(record.latest_error or "").strip()
        if not error:
            continue
        if record.job_id in latest_by_job:
            continue
        latest_by_job[record.job_id] = {
            "task_id": record.task_id,
            "task_name": task_name_by_id.get(record.task_id),
            "error": error,
            "updated_at": record.updated_at.isoformat()
            if isinstance(record.updated_at, datetime)
            else None,
        }
    return latest_by_job


def _parse_task_stream_event(stream_id: str, record: Mapping[str, str]) -> dict[str, Any] | None:
    raw = record.get("data")
    if not raw:
        return None
    try:
        envelope = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(envelope, dict):
        return None
    payload = envelope.get("payload")
    payload_dict = payload if isinstance(payload, dict) else {}
    task_id = envelope.get("task_id")
    if not isinstance(task_id, str) or not task_id:
        maybe_task = payload_dict.get("task_id")
        task_id = maybe_task if isinstance(maybe_task, str) else None
    event_type = envelope.get("type")
    if not isinstance(event_type, str) or not event_type:
        return None
    occurred_at = envelope.get("occurred_at")
    if not isinstance(occurred_at, str):
        occurred_at = _utcnow().isoformat()
    status = payload_dict.get("status")
    status_text = status if isinstance(status, str) and status else event_type.replace("task.", "")
    error = payload_dict.get("error")
    return {
        "stream_id": stream_id,
        "type": event_type,
        "occurred_at": occurred_at,
        "job_id": envelope.get("job_id") if isinstance(envelope.get("job_id"), str) else None,
        "task_id": task_id,
        "status": status_text,
        "attempts": payload_dict.get("attempts"),
        "max_attempts": payload_dict.get("max_attempts"),
        "worker_consumer": payload_dict.get("worker_consumer"),
        "run_id": payload_dict.get("run_id"),
        "error": error if isinstance(error, str) else "",
    }


def _read_task_events_for_job(job_id: str, limit: int) -> list[dict[str, Any]]:
    scan_count = min(max(limit * 8, 400), 5000)
    try:
        rows = redis_client.xrevrange(events.TASK_STREAM, "+", "-", count=scan_count)
    except redis.RedisError:
        return []
    entries: list[dict[str, Any]] = []
    for stream_id, record in rows:
        parsed = _parse_task_stream_event(stream_id, record)
        if parsed is None:
            continue
        if parsed.get("job_id") != job_id:
            continue
        entries.append(parsed)
        if len(entries) >= limit:
            break
    entries.reverse()
    return entries


def _list_step_attempts_for_job(db: Session, job_id: str) -> list[models.StepAttempt]:
    rows = (
        db.query(StepAttemptRecord)
        .filter(StepAttemptRecord.job_id == job_id)
        .order_by(StepAttemptRecord.started_at.asc(), StepAttemptRecord.attempt_number.asc())
        .all()
    )
    return [_step_attempt_from_record(row) for row in rows]


def _list_invocations_for_job(db: Session, job_id: str) -> list[models.Invocation]:
    rows = (
        db.query(InvocationRecord)
        .filter(InvocationRecord.job_id == job_id)
        .order_by(InvocationRecord.started_at.asc(), InvocationRecord.capability_id.asc())
        .all()
    )
    return [_invocation_from_record(row) for row in rows]


def _list_run_events_for_job(
    db: Session,
    job_id: str,
    *,
    limit: int,
) -> list[models.RunEvent]:
    rows = (
        db.query(RunEventRecord)
        .filter(RunEventRecord.job_id == job_id)
        .order_by(RunEventRecord.occurred_at.asc())
        .limit(limit)
        .all()
    )
    return [_run_event_from_record(row) for row in rows]


def _debugger_timeline_for_job(job_id: str, *, limit: int, db: Session) -> list[dict[str, Any]]:
    durable_rows = (
        db.query(RunEventRecord)
        .filter(RunEventRecord.job_id == job_id)
        .order_by(RunEventRecord.occurred_at.asc())
        .limit(limit)
        .all()
    )
    if durable_rows:
        return [_debugger_timeline_entry_from_run_event(row) for row in durable_rows]
    return _read_task_events_for_job(job_id, limit)


@app.post("/chat/sessions", response_model=chat_contracts.ChatSession)
def create_chat_session(
    request: chat_contracts.ChatSessionCreate,
    raw_request: Request,
    db: Session = Depends(get_db),
) -> chat_contracts.ChatSession:
    return chat_service.create_session(
        db,
        request,
        runtime=_chat_runtime(),
        user_id=_chat_authenticated_user_id(raw_request),
    )


@app.get("/chat/sessions/{session_id}", response_model=chat_contracts.ChatSession)
def get_chat_session(
    session_id: str,
    raw_request: Request,
    db: Session = Depends(get_db),
) -> chat_contracts.ChatSession:
    session = chat_service.get_session(
        db,
        session_id,
        user_id=_chat_authenticated_user_id(raw_request),
    )
    if session is None:
        raise HTTPException(status_code=404, detail="chat_session_not_found")
    return session


@app.post(
    "/chat/sessions/{session_id}/messages",
    response_model=chat_contracts.ChatTurnResponse,
)
def create_chat_message(
    session_id: str,
    request: chat_contracts.ChatTurnRequest,
    raw_request: Request,
    db: Session = Depends(get_db),
) -> chat_contracts.ChatTurnResponse:
    try:
        return chat_service.handle_turn(
            db,
            session_id,
            request,
            runtime=_chat_runtime(),
            user_id=_chat_authenticated_user_id(raw_request),
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="chat_session_not_found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _raise_feedback_http_error(exc: ValueError) -> None:
    detail = str(exc)
    status_code = 404 if detail == "feedback_target_not_found" else 400
    raise HTTPException(status_code=status_code, detail=detail) from exc


@app.post("/feedback", response_model=models.Feedback)
def submit_feedback(
    request: models.FeedbackCreate,
    raw_request: Request,
    db: Session = Depends(get_db),
) -> models.Feedback:
    try:
        feedback = feedback_service.submit_feedback(
            db,
            request,
            actor_key=_feedback_actor_key(raw_request),
            user_id=_chat_authenticated_user_id(raw_request),
        )
    except ValueError as exc:
        _raise_feedback_http_error(exc)
    feedback_submitted_total.labels(
        target_type=feedback.target_type.value,
        sentiment=feedback.sentiment.value,
    ).inc()
    for reason_code in feedback.reason_codes:
        feedback_reason_total.labels(
            target_type=feedback.target_type.value,
            reason_code=reason_code,
        ).inc()
    dimensions = (
        dict(feedback.metadata.get("dimensions") or {})
        if isinstance(feedback.metadata, Mapping)
        else {}
    )
    boundary_decision = _metrics_label(dimensions.get("boundary_decision"), default="none")
    if feedback.target_type == models.FeedbackTargetType.chat_message and boundary_decision != "none":
        chat_boundary_feedback_total.labels(
            decision=boundary_decision,
            sentiment=feedback.sentiment.value,
        ).inc()
    if feedback.target_type == models.FeedbackTargetType.chat_message:
        slot_loss_state = _metrics_label(
            dimensions.get("clarification_slot_loss_state"),
            default="none",
        )
        if slot_loss_state != "none":
            chat_clarification_slot_loss_feedback_total.labels(
                slot_loss_state=slot_loss_state,
                sentiment=feedback.sentiment.value,
            ).inc()
        family_alignment = _metrics_label(
            dimensions.get("clarification_family_alignment"),
            default="unknown",
        )
        if family_alignment != "unknown":
            chat_clarification_family_alignment_feedback_total.labels(
                alignment=family_alignment,
                sentiment=feedback.sentiment.value,
            ).inc()
        chat_clarification_mapping_feedback_total.labels(
            resolved_active_field=_metrics_label(
                dimensions.get("clarification_mapping_resolved_active_field"),
                default="unknown",
            ),
            queue_advanced=_metrics_label(
                dimensions.get("clarification_mapping_queue_advanced"),
                default="unknown",
            ),
            restarted=_metrics_label(
                dimensions.get("clarification_mapping_restarted"),
                default="unknown",
            ),
            sentiment=feedback.sentiment.value,
        ).inc()
    _emit_event("feedback.submitted", feedback.model_dump(mode="json"))
    return feedback


@app.get("/feedback", response_model=models.FeedbackListResponse)
def get_feedback(
    target_type: models.FeedbackTargetType | None = Query(default=None),
    target_id: str | None = Query(default=None),
    session_id: str | None = Query(default=None),
    job_id: str | None = Query(default=None),
    plan_id: str | None = Query(default=None),
    message_id: str | None = Query(default=None),
    actor_key: str | None = Query(default=None),
    limit: int = Query(default=200, ge=1, le=500),
    db: Session = Depends(get_db),
) -> models.FeedbackListResponse:
    return feedback_service.list_feedback_response(
        db,
        target_type=target_type,
        target_id=target_id,
        session_id=session_id,
        job_id=job_id,
        plan_id=plan_id,
        message_id=message_id,
        actor_key=actor_key,
        limit=limit,
    )


@app.get("/feedback/summary", response_model=models.FeedbackSummaryResponse)
def get_feedback_summary(
    target_type: models.FeedbackTargetType | None = Query(default=None),
    sentiment: models.FeedbackSentiment | None = Query(default=None),
    workflow_source: str | None = Query(default=None),
    llm_model: str | None = Query(default=None),
    planner_version: str | None = Query(default=None),
    since: datetime | None = Query(default=None),
    until: datetime | None = Query(default=None),
    limit: int = Query(default=500, ge=1, le=5000),
    db: Session = Depends(get_db),
) -> models.FeedbackSummaryResponse:
    feedback_summary_requests_total.inc()
    return feedback_service.summary_feedback_response(
        db,
        models.FeedbackSummaryRequest(
            target_type=target_type,
            sentiment=sentiment,
            workflow_source=workflow_source,
            llm_model=llm_model,
            planner_version=planner_version,
            since=since,
            until=until,
            limit=limit,
        ),
    )


@app.get("/feedback/examples", response_model=None)
def get_feedback_examples(
    target_type: models.FeedbackTargetType | None = Query(default=None),
    sentiment: list[models.FeedbackSentiment] | None = Query(default=None),
    reason_code: str | None = Query(default=None),
    workflow_source: str | None = Query(default=None),
    llm_model: str | None = Query(default=None),
    planner_version: str | None = Query(default=None),
    since: datetime | None = Query(default=None),
    until: datetime | None = Query(default=None),
    limit: int = Query(default=200, ge=1, le=5000),
    format: models.FeedbackExampleFormat = Query(default=models.FeedbackExampleFormat.json),
    db: Session = Depends(get_db),
) -> Any:
    feedback_examples_export_total.inc()
    export = feedback_service.export_feedback_examples(
        db,
        target_type=target_type,
        sentiments=sentiment,
        reason_code=reason_code,
        workflow_source=workflow_source,
        llm_model=llm_model,
        planner_version=planner_version,
        since=since,
        until=until,
        limit=limit,
    )
    if format == models.FeedbackExampleFormat.jsonl:
        payload = feedback_eval.dumps_feedback_eval_rows_jsonl(
            [item.model_dump(mode="json") for item in export.items]
        )
        return StreamingResponse(
            iter([payload.encode("utf-8")]),
            media_type="application/x-ndjson",
        )
    return export


@app.get("/feedback/chat-boundary/review", response_model=models.ChatBoundaryReviewQueueResponse)
def get_chat_boundary_review_queue(
    review_label: str | None = Query(default=None),
    boundary_decision: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    db: Session = Depends(get_db),
) -> models.ChatBoundaryReviewQueueResponse:
    return feedback_service.chat_boundary_review_queue(
        db,
        review_label=review_label,
        boundary_decision=boundary_decision,
        limit=limit,
    )


@app.get(
    "/feedback/chat-clarification/review",
    response_model=models.ChatClarificationReviewQueueResponse,
)
def get_chat_clarification_review_queue(
    review_label: str | None = Query(default=None),
    active_family: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    db: Session = Depends(get_db),
) -> models.ChatClarificationReviewQueueResponse:
    return feedback_service.chat_clarification_review_queue(
        db,
        review_label=review_label,
        active_family=active_family,
        limit=limit,
    )


@app.get("/feedback/intent/review", response_model=models.IntentReviewQueueResponse)
def get_intent_review_queue(
    review_label: str | None = Query(default=None),
    intent: str | None = Query(default=None),
    intent_source: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    db: Session = Depends(get_db),
) -> models.IntentReviewQueueResponse:
    return feedback_service.intent_review_queue(
        db,
        review_label=review_label,
        intent=intent,
        intent_source=intent_source,
        limit=limit,
    )


@app.get("/feedback/intent/tuning-report", response_model=models.IntentTuningReportResponse)
def get_intent_tuning_report(
    review_label: str | None = Query(default=None),
    intent: str | None = Query(default=None),
    intent_source: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    db: Session = Depends(get_db),
) -> models.IntentTuningReportResponse:
    return feedback_service.intent_tuning_report(
        db,
        review_label=review_label,
        intent=intent,
        intent_source=intent_source,
        limit=limit,
    )


@app.get(
    "/feedback/intent/tuning-candidates",
    response_model=models.IntentTuningCandidateExportResponse,
)
def get_intent_tuning_candidates(
    review_label: str | None = Query(default=None),
    intent: str | None = Query(default=None),
    intent_source: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    db: Session = Depends(get_db),
) -> models.IntentTuningCandidateExportResponse:
    return feedback_service.intent_tuning_candidates(
        db,
        review_label=review_label,
        intent=intent,
        intent_source=intent_source,
        limit=limit,
    )


@app.get("/jobs/{job_id}/feedback", response_model=models.FeedbackListResponse)
def get_job_feedback(
    job_id: str,
    limit: int = Query(default=200, ge=1, le=500),
    db: Session = Depends(get_db),
) -> models.FeedbackListResponse:
    job = db.query(JobRecord).filter(JobRecord.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return feedback_service.list_feedback_response(db, job_id=job_id, limit=limit)


@app.get("/chat/sessions/{session_id}/feedback", response_model=models.FeedbackListResponse)
def get_chat_session_feedback(
    session_id: str,
    raw_request: Request,
    limit: int = Query(default=200, ge=1, le=500),
    db: Session = Depends(get_db),
) -> models.FeedbackListResponse:
    session = chat_service.get_session(
        db,
        session_id,
        user_id=_chat_authenticated_user_id(raw_request),
    )
    if session is None:
        raise HTTPException(status_code=404, detail="chat_session_not_found")
    return feedback_service.list_feedback_response(db, session_id=session_id, limit=limit)


def _create_job_internal(
    job: models.JobCreate,
    db: Session,
    *,
    require_clarification: bool = False,
    emit_job_created_event: bool = True,
    metadata_overrides: Mapping[str, Any] | None = None,
) -> models.Job:
    job_id = str(uuid.uuid4())
    now = _utcnow()
    context_json_for_job = (
        dict(job.context_json) if isinstance(job.context_json, dict) else {}
    )
    interaction_summaries_raw: list[dict[str, Any]] = []
    interaction_summaries_compact: list[dict[str, Any]] = []
    interaction_compaction: dict[str, Any] = {}
    if isinstance(job.context_json, dict) and "interaction_summaries" in job.context_json:
        try:
            interaction_summaries_raw = _normalize_interaction_summaries(
                job.context_json.get("interaction_summaries")
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        (
            interaction_summaries_compact,
            interaction_compaction,
        ) = _compact_interaction_summaries(interaction_summaries_raw)
        context_json_for_job["interaction_summaries"] = interaction_summaries_compact
        context_json_for_job["interaction_summaries_ref"] = {
            "memory_name": "interaction_summaries_compact",
            "key": "latest",
            "job_id": job_id,
            "raw_memory_name": "interaction_summaries",
            "raw_key": "raw:latest",
        }
        context_json_for_job["interaction_summaries_meta"] = interaction_compaction

    metadata: Dict[str, Any] = {}
    if job.idempotency_key:
        metadata["idempotency_key"] = job.idempotency_key
    metadata["render_path_mode"] = planner_contracts.RENDER_PATH_MODE_EXPLICIT
    if LLM_PROVIDER_NAME:
        metadata["llm_provider"] = LLM_PROVIDER_NAME
    if LLM_MODEL_NAME:
        metadata["llm_model"] = LLM_MODEL_NAME
    semantic_user_id = _semantic_user_id_from_context(
        context_json_for_job if isinstance(context_json_for_job, Mapping) else None
    )
    intent_context_envelope = context_service.build_context_envelope(
        db=db,
        goal=job.goal,
        context_sources=context_service.collect_context_sources(
            job_context=context_json_for_job,
        ),
        user_id=semantic_user_id,
        runtime_metadata={"surface": "job_create_intent"},
    )
    normalized_intent_envelope = _normalize_goal_intent(
        job.goal,
        db=db,
        user_id=semantic_user_id,
        interaction_summaries=interaction_summaries_compact or interaction_summaries_raw,
        context_envelope=intent_context_envelope,
    )
    if interaction_summaries_raw and INTENT_DECOMPOSE_ENABLED:
        normalized_intent_envelope = normalized_intent_envelope.model_copy(
            update={
                "graph": _attach_interaction_compaction_to_graph(
                    normalized_intent_envelope.graph,
                    interaction_compaction,
                )
            }
        )
    goal_assessment = normalized_intent_envelope.profile
    goal_assessment_json = workflow_contracts.dump_goal_intent_profile(goal_assessment) or {}
    gate_on_create = bool(require_clarification or INTENT_CLARIFICATION_ON_CREATE)
    if gate_on_create and bool(goal_assessment.requires_blocking_clarification):
        raise HTTPException(
            status_code=422,
            detail={
                "error": "intent_clarification_required",
                "goal_intent_profile": goal_assessment_json,
                "normalized_intent_envelope": (
                    workflow_contracts.dump_normalized_intent_envelope(normalized_intent_envelope)
                    or {}
                ),
            },
        )
    metadata["semantic_user_id"] = semantic_user_id
    metadata["goal_intent_profile"] = goal_assessment_json
    metadata["normalized_intent_envelope"] = (
        workflow_contracts.dump_normalized_intent_envelope(normalized_intent_envelope) or {}
    )
    if INTENT_DECOMPOSE_ENABLED:
        goal_intent_graph = normalized_intent_envelope.graph
        metadata["goal_intent_graph"] = goal_intent_graph.model_dump(mode="json", exclude_none=True)
    if isinstance(metadata_overrides, Mapping):
        metadata.update(dict(metadata_overrides))
    record = JobRecord(
        id=job_id,
        goal=job.goal,
        context_json=context_json_for_job,
        status=models.JobStatus.queued.value,
        created_at=now,
        updated_at=now,
        priority=job.priority,
        metadata_json=metadata,
    )
    db.add(record)
    db.commit()
    jobs_created_total.inc()
    if interaction_summaries_raw:
        _persist_interaction_summaries_memory(
            db,
            job_id=job_id,
            raw_summaries=interaction_summaries_raw,
            compact_summaries=interaction_summaries_compact or interaction_summaries_raw,
            compaction=interaction_compaction,
            source="job_create",
        )
    if isinstance(context_json_for_job, dict) and context_json_for_job:
        try:
            memory_store.write_memory(
                db,
                models.MemoryWrite(
                    name="job_context",
                    job_id=job_id,
                    payload=context_json_for_job,
                    metadata={"source": "job_create"},
                ),
            )
        except (KeyError, ValueError):
            pass
        _seed_task_output_memory(db, job_id, context_json_for_job)
    if emit_job_created_event:
        _emit_event("job.created", _job_from_record(record).model_dump())
    return _job_from_record(record)


@app.post("/jobs", response_model=models.Job)
def create_job(
    job: models.JobCreate,
    require_clarification: bool = Query(False),
    db: Session = Depends(get_db),
) -> models.Job:
    return _create_job_internal(job, db, require_clarification=require_clarification)


def _seed_task_output_memory(db: Session, job_id: str, context_json: Dict[str, Any]) -> None:
    seed_map = {
        "document_spec": "document_spec:latest",
        "path": "docx_path:latest",
    }
    for field, key in seed_map.items():
        value = context_json.get(field)
        if isinstance(value, str):
            if not value.strip():
                continue
        elif isinstance(value, dict):
            if not value:
                continue
        elif value is None:
            continue
        try:
            memory_store.write_memory(
                db,
                models.MemoryWrite(
                    name="task_outputs",
                    job_id=job_id,
                    key=key,
                    payload={field: value},
                    metadata={"source": "job_create"},
                ),
            )
        except (KeyError, ValueError):
            continue


@app.get("/jobs", response_model=List[models.Job])
def list_jobs(db: Session = Depends(get_db)) -> List[models.Job]:
    jobs = db.query(JobRecord).all()
    return [_job_from_record(job) for job in jobs]


@app.get("/capabilities")
def list_capabilities(
    include_disabled: bool = Query(False),
    with_schemas: bool = Query(True),
) -> dict[str, Any]:
    mode = capability_registry.resolve_capability_mode()
    try:
        registry = capability_registry.load_capability_registry()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"capability_registry_load_failed:{exc}") from exc

    items: list[dict[str, Any]] = []
    for capability_id, spec in sorted(registry.capabilities.items()):
        if not include_disabled and not spec.enabled:
            continue
        input_schema, output_schema = _resolve_capability_schemas(spec, include_schemas=with_schemas)
        required_inputs: list[str] = []
        if isinstance(input_schema, dict):
            required = input_schema.get("required")
            if isinstance(required, list):
                required_inputs = [entry for entry in required if isinstance(entry, str)]
        input_fields = _flatten_schema_fields(input_schema) if with_schemas else []
        output_fields = _flatten_schema_fields(output_schema) if with_schemas else []
        items.append(
            {
                "id": capability_id,
                "description": spec.description,
                "enabled": spec.enabled,
                "risk_tier": spec.risk_tier,
                "idempotency": spec.idempotency,
                "group": spec.group,
                "subgroup": spec.subgroup,
                "tags": list(spec.tags),
                "input_schema_ref": spec.input_schema_ref,
                "output_schema_ref": spec.output_schema_ref,
                "input_schema": input_schema,
                "output_schema": output_schema,
                "required_inputs": required_inputs,
                "input_fields": input_fields,
                "output_fields": output_fields,
                "planner_hints": spec.planner_hints if isinstance(spec.planner_hints, dict) else {},
                "adapters": [
                    {
                        "type": adapter.type,
                        "server_id": adapter.server_id,
                        "tool_name": adapter.tool_name,
                    }
                    for adapter in spec.adapters
                    if adapter.enabled
                ],
            }
        )
    return {"mode": mode, "items": items}


@app.post("/capabilities/search")
def search_capabilities(payload: dict[str, Any] = Body(default_factory=dict)) -> dict[str, Any]:
    query = str(payload.get("query") or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="query_required")
    try:
        limit = int(payload.get("limit", 8))
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="limit_invalid")
    if limit < 1 or limit > 50:
        raise HTTPException(status_code=400, detail="limit_out_of_range")
    intent_hint = str(payload.get("intent") or "").strip().lower() or None
    request_source = str(payload.get("request_source") or "api").strip().lower() or "api"
    correlation_id = str(payload.get("correlation_id") or "").strip() or None
    job_id = str(payload.get("job_id") or "").strip() or None
    mode = capability_registry.resolve_capability_mode()
    try:
        registry = capability_registry.load_capability_registry()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"capability_registry_load_failed:{exc}") from exc

    entries = capability_search.build_capability_search_entries(registry.enabled_capabilities())
    started = time.perf_counter()
    matches = capability_search.search_capabilities(
        query=query,
        capability_entries=entries,
        limit=limit,
        intent_hint=intent_hint,
    )
    latency_ms = round((time.perf_counter() - started) * 1000.0, 3)
    _emit_capability_search_event(
        query=query,
        intent_hint=intent_hint,
        limit=limit,
        matches=matches,
        request_source=request_source,
        latency_ms=latency_ms,
        correlation_id=correlation_id,
        job_id=job_id,
    )
    details_by_id = {str(entry.get("id") or ""): entry for entry in entries}
    items: list[dict[str, Any]] = []
    for match in matches:
        capability_id = str(match.get("id") or "").strip()
        if not capability_id:
            continue
        entry = details_by_id.get(capability_id, {})
        items.append(
            {
                "id": capability_id,
                "score": float(match.get("score") or 0.0),
                "reason": str(match.get("reason") or "").strip() or "semantic match",
                "source": str(match.get("source") or "semantic_search"),
                "description": str(entry.get("description") or "").strip(),
                "group": str(entry.get("group") or "").strip(),
                "subgroup": str(entry.get("subgroup") or "").strip(),
                "tags": [tag for tag in entry.get("tags", []) if isinstance(tag, str)],
            }
        )
    return {
        "mode": mode,
        "query": query,
        "intent": intent_hint,
        "limit": limit,
        "items": items,
    }


@app.get("/jobs/{job_id}", response_model=models.Job)
def get_job(job_id: str, db: Session = Depends(get_db)) -> models.Job:
    job = db.query(JobRecord).filter(JobRecord.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return _job_from_record(job)


@app.get("/jobs/{job_id}/plan", response_model=models.Plan)
def get_plan(job_id: str, db: Session = Depends(get_db)) -> models.Plan:
    plan = db.query(PlanRecord).filter(PlanRecord.job_id == job_id).first()
    if not plan:
        raise HTTPException(status_code=404, detail="Plan not found")
    return _plan_from_record(plan)


@app.get("/jobs/{job_id}/tasks", response_model=List[models.Task])
def get_tasks(job_id: str, db: Session = Depends(get_db)) -> List[models.Task]:
    job = db.query(JobRecord).filter(JobRecord.id == job_id).first()
    tasks = db.query(TaskRecord).filter(TaskRecord.job_id == job_id).all()
    profiles = _coerce_task_intent_profiles(
        job.metadata_json if job and isinstance(job.metadata_json, dict) else {}
    )
    return [_task_from_record(task, profiles.get(task.id)) for task in tasks]


@app.get("/jobs/{job_id}/task_results")
def get_task_results(job_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    tasks = db.query(TaskRecord).filter(TaskRecord.job_id == job_id).all()
    return {task.id: _load_task_result(task.id) for task in tasks}


@app.get("/jobs/{job_id}/details", response_model=models.JobDetails)
def get_job_details(job_id: str, db: Session = Depends(get_db)) -> models.JobDetails:
    job = db.query(JobRecord).filter(JobRecord.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    plan_record = db.query(PlanRecord).filter(PlanRecord.job_id == job_id).first()
    tasks = db.query(TaskRecord).filter(TaskRecord.job_id == job_id).all()
    metadata = job.metadata_json if isinstance(job.metadata_json, dict) else {}
    profiles = _coerce_task_intent_profiles(metadata)
    normalization_fields = _normalization_response_fields(metadata, goal=job.goal or "")
    return models.JobDetails(
        job_id=job_id,
        job_status=models.JobStatus(job.status),
        job_error=_job_error_from_metadata(metadata),
        plan=_plan_from_record(plan_record) if plan_record else None,
        tasks=[_task_from_record(task, profiles.get(task.id)) for task in tasks],
        task_results={task.id: _load_task_result(task.id) for task in tasks},
        goal_intent_profile=normalization_fields["goal_intent_profile"],
        goal_intent_graph=normalization_fields["goal_intent_graph"],
        normalized_intent_envelope=normalization_fields["normalized_intent_envelope"],
        normalization_trace=normalization_fields["normalization_trace"],
        normalization_clarification=normalization_fields["normalization_clarification"],
        normalization_candidate_capabilities=normalization_fields["normalization_candidate_capabilities"],
    )


@app.get("/jobs/{job_id}/debugger")
def get_job_debugger(
    job_id: str,
    limit: int = Query(400, ge=50, le=2000),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    job = db.query(JobRecord).filter(JobRecord.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    plan_record = db.query(PlanRecord).filter(PlanRecord.job_id == job_id).first()
    task_records = (
        db.query(TaskRecord)
        .filter(TaskRecord.job_id == job_id)
        .order_by(TaskRecord.created_at.asc(), TaskRecord.name.asc())
        .all()
    )
    task_models = _resolve_task_deps(task_records)
    task_map = {task.id: task for task in task_models}
    id_to_name = {record.id: record.name for record in task_records}
    metadata = job.metadata_json if isinstance(job.metadata_json, dict) else {}
    job_context = _execution_job_context(
        job.goal if isinstance(job.goal, str) else "",
        job.context_json if isinstance(job.context_json, dict) else {},
        metadata,
    )
    task_intent_profiles = _coerce_task_intent_profiles(metadata)
    normalization_fields = _normalization_response_fields(metadata, goal=job.goal or "")

    timeline = _debugger_timeline_for_job(job_id, limit=limit, db=db)
    timeline_by_task: dict[str, list[dict[str, Any]]] = {}
    for entry in timeline:
        task_id = entry.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            continue
        timeline_by_task.setdefault(task_id, []).append(entry)
    step_attempts = _list_step_attempts_for_job(db, job_id)
    invocations = _list_invocations_for_job(db, job_id)
    invocations_by_attempt: dict[str, list[dict[str, Any]]] = {}
    for invocation in invocations:
        invocations_by_attempt.setdefault(invocation.step_attempt_id, []).append(
            invocation.model_dump(mode="json")
        )
    attempts_by_task: dict[str, list[dict[str, Any]]] = {}
    for attempt in step_attempts:
        attempt_payload = attempt.model_dump(mode="json")
        attempt_payload["invocations"] = invocations_by_attempt.get(attempt.id, [])
        attempts_by_task.setdefault(attempt.step_id, []).append(attempt_payload)

    tasks_payload: list[dict[str, Any]] = []
    for record in task_records:
        context = _build_task_context(record.id, task_map, id_to_name, job_context)
        hydrated_payload = _task_payload_from_record(
            record,
            correlation_id=None,
            context=context,
            goal_text=job.goal if isinstance(job.goal, str) else "",
            intent_profile=task_intent_profiles.get(record.id),
        )
        task_result = _load_task_result(record.id)
        timeline_entries = timeline_by_task.get(record.id, [])
        latest_error = _extract_error_from_task_result(task_result)
        if not latest_error:
            for entry in reversed(timeline_entries):
                error = entry.get("error")
                if isinstance(error, str) and error:
                    latest_error = error
                    break
        tasks_payload.append(
            {
                "task": _task_from_record(
                    record,
                    task_intent_profiles.get(record.id),
                ).model_dump(mode="json"),
                "resolved_tool_inputs": hydrated_payload.get("tool_inputs", {}),
                "tool_inputs_validation": hydrated_payload.get("tool_inputs_validation", {}),
                "tool_inputs_resolved": bool(hydrated_payload.get("tool_inputs_resolved")),
                "context_keys": sorted(context.keys()),
                "timeline": timeline_entries,
                "attempts": attempts_by_task.get(record.id, []),
                "latest_result": task_result,
                "error": _classify_task_error(latest_error),
            }
        )

    return {
        "job_id": job_id,
        "job_status": job.status,
        "plan_id": plan_record.id if plan_record else None,
        "generated_at": _utcnow().isoformat(),
        "timeline_events_scanned": len(timeline),
        "goal_intent_profile": normalization_fields["goal_intent_profile"],
        "goal_intent_graph": normalization_fields["goal_intent_graph"],
        "normalized_intent_envelope": normalization_fields["normalized_intent_envelope"],
        "normalization_trace": normalization_fields["normalization_trace"],
        "normalization_clarification": normalization_fields["normalization_clarification"],
        "normalization_candidate_capabilities": normalization_fields["normalization_candidate_capabilities"],
        "tasks": tasks_payload,
    }


@app.get("/jobs/{job_id}/debugger/attempts", response_model=List[models.StepAttempt])
def get_job_debugger_attempts(
    job_id: str,
    db: Session = Depends(get_db),
) -> List[models.StepAttempt]:
    job = db.query(JobRecord).filter(JobRecord.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return _list_step_attempts_for_job(db, job_id)


@app.get("/jobs/{job_id}/debugger/invocations", response_model=List[models.Invocation])
def get_job_debugger_invocations(
    job_id: str,
    db: Session = Depends(get_db),
) -> List[models.Invocation]:
    job = db.query(JobRecord).filter(JobRecord.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return _list_invocations_for_job(db, job_id)


@app.get("/jobs/{job_id}/debugger/events", response_model=List[models.RunEvent])
def get_job_debugger_events(
    job_id: str,
    limit: int = Query(400, ge=50, le=2000),
    db: Session = Depends(get_db),
) -> List[models.RunEvent]:
    job = db.query(JobRecord).filter(JobRecord.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return _list_run_events_for_job(db, job_id, limit=limit)


@app.get("/jobs/{job_id}/events/outbox")
def get_job_event_outbox(
    job_id: str,
    pending_only: bool = Query(True),
    limit: int = Query(100, ge=1, le=1000),
    include_payload: bool = Query(False),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    base_query = db.query(EventOutboxRecord)
    if pending_only:
        base_query = base_query.filter(EventOutboxRecord.published_at.is_(None))
    rows = (
        base_query.order_by(EventOutboxRecord.created_at.desc())
        .limit(max(limit * 20, limit))
        .all()
    )
    entries: list[dict[str, Any]] = []
    for row in rows:
        envelope = row.envelope_json if isinstance(row.envelope_json, dict) else {}
        if envelope.get("job_id") != job_id:
            continue
        entries.append(_outbox_entry_payload(row, include_payload=include_payload))
        if len(entries) >= limit:
            break
    return {
        "job_id": job_id,
        "pending_only": pending_only,
        "count": len(entries),
        "items": entries,
    }


@app.get("/tasks/{task_id}", response_model=models.Task)
def get_task(task_id: str, db: Session = Depends(get_db)) -> models.Task:
    task = db.query(TaskRecord).filter(TaskRecord.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    job = db.query(JobRecord).filter(JobRecord.id == task.job_id).first()
    profiles = _coerce_task_intent_profiles(
        job.metadata_json if job and isinstance(job.metadata_json, dict) else {}
    )
    return _task_from_record(task, profiles.get(task.id))


@app.post("/jobs/{job_id}/cancel", response_model=models.Job)
def cancel_job(job_id: str, db: Session = Depends(get_db)) -> models.Job:
    job = db.query(JobRecord).filter(JobRecord.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if not state_machine.validate_job_transition(
        models.JobStatus(job.status), models.JobStatus.canceled
    ):
        raise HTTPException(status_code=400, detail="Invalid state transition")
    job.status = models.JobStatus.canceled.value
    job.updated_at = _utcnow()
    db.commit()
    _emit_event("job.canceled", {"job_id": job_id, "correlation_id": str(uuid.uuid4())})
    return _job_from_record(job)


@app.post("/jobs/{job_id}/resume", response_model=models.Job)
def resume_job(job_id: str, db: Session = Depends(get_db)) -> models.Job:
    job = db.query(JobRecord).filter(JobRecord.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if models.JobStatus(job.status) != models.JobStatus.canceled:
        raise HTTPException(status_code=400, detail="Job is not canceled")
    _set_job_status(job, models.JobStatus.planning)
    job.updated_at = _utcnow()
    db.commit()
    plan = db.query(PlanRecord).filter(PlanRecord.job_id == job_id).first()
    if plan:
        _dispatch_ready_work_for_job(job_id, plan.id, None)
    return _job_from_record(job)


@app.post("/jobs/{job_id}/retry", response_model=models.Job)
def retry_job(job_id: str, db: Session = Depends(get_db)) -> models.Job:
    job = db.query(JobRecord).filter(JobRecord.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if models.JobStatus(job.status) not in {models.JobStatus.failed, models.JobStatus.canceled}:
        raise HTTPException(status_code=400, detail="Job is not retryable")
    now = _utcnow()
    tasks = db.query(TaskRecord).filter(TaskRecord.job_id == job_id).all()
    for task in tasks:
        task.status = models.TaskStatus.pending.value
        task.attempts = 0
        task.rework_count = 0
        task.updated_at = now
    _set_job_status(job, models.JobStatus.planning)
    job.updated_at = now
    db.commit()
    plan = db.query(PlanRecord).filter(PlanRecord.job_id == job_id).first()
    if plan:
        _dispatch_ready_work_for_job(job_id, plan.id, None)
    return _job_from_record(job)


@app.post("/jobs/{job_id}/retry_failed", response_model=models.Job)
def retry_failed_tasks(job_id: str, db: Session = Depends(get_db)) -> models.Job:
    job = db.query(JobRecord).filter(JobRecord.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    now = _utcnow()
    failed_tasks = (
        db.query(TaskRecord)
        .filter(
            TaskRecord.job_id == job_id,
            TaskRecord.status == models.TaskStatus.failed.value,
        )
        .all()
    )
    if not failed_tasks:
        raise HTTPException(status_code=400, detail="No failed tasks to retry")
    for task in failed_tasks:
        task.status = models.TaskStatus.pending.value
        task.attempts = 0
        task.rework_count = 0
        task.updated_at = now
    _set_job_status(job, models.JobStatus.planning)
    job.updated_at = now
    db.commit()
    plan = db.query(PlanRecord).filter(PlanRecord.job_id == job_id).first()
    if plan:
        _dispatch_ready_work_for_job(job_id, plan.id, None)
    return _job_from_record(job)


@app.post("/jobs/{job_id}/tasks/{task_id}/retry", response_model=models.Job)
def retry_task(
    job_id: str,
    task_id: str,
    payload: dict[str, Any] = Body(default_factory=dict),
    db: Session = Depends(get_db),
) -> models.Job:
    job = db.query(JobRecord).filter(JobRecord.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    task = (
        db.query(TaskRecord).filter(TaskRecord.id == task_id, TaskRecord.job_id == job_id).first()
    )
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    if task.status != models.TaskStatus.failed.value:
        raise HTTPException(status_code=400, detail="Task is not failed")
    now = _utcnow()
    task.status = models.TaskStatus.pending.value
    task.attempts = 0
    task.rework_count = 0
    task.updated_at = now
    _set_job_status(job, models.JobStatus.planning)
    job.updated_at = now
    db.commit()
    stream_id = payload.get("stream_id")
    if isinstance(stream_id, str) and stream_id:
        try:
            redis_client.xdel(events.TASK_DLQ_STREAM, stream_id)
        except redis.RedisError:
            logger.warning(
                "task_dlq_delete_failed",
                extra={"job_id": job_id, "task_id": task_id, "stream_id": stream_id},
            )
    plan = db.query(PlanRecord).filter(PlanRecord.id == task.plan_id).first()
    if not plan:
        plan = db.query(PlanRecord).filter(PlanRecord.job_id == job_id).first()
    if plan:
        _dispatch_ready_work_for_job(job_id, plan.id, None)
    return _job_from_record(job)


@app.post("/jobs/{job_id}/replan", response_model=models.Job)
def replan_job(job_id: str, db: Session = Depends(get_db)) -> models.Job:
    job = db.query(JobRecord).filter(JobRecord.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    metadata = dict(job.metadata_json) if isinstance(job.metadata_json, dict) else {}
    metadata["replan_count"] = int(metadata.get("replan_count", 0)) + 1
    metadata["replan_reason"] = "manual"
    job.metadata_json = metadata
    job.status = models.JobStatus.planning.value
    job.updated_at = _utcnow()
    db.query(TaskRecord).filter(TaskRecord.job_id == job_id).delete(synchronize_session=False)
    db.query(PlanRecord).filter(PlanRecord.job_id == job_id).delete(synchronize_session=False)
    db.commit()
    _emit_event("job.created", _job_from_record(job).model_dump())
    return _job_from_record(job)


@app.post("/jobs/{job_id}/clear")
def clear_job(job_id: str, db: Session = Depends(get_db)) -> dict[str, str]:
    job = db.query(JobRecord).filter(JobRecord.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    db.query(TaskRecord).filter(TaskRecord.job_id == job_id).delete()
    db.query(PlanRecord).filter(PlanRecord.job_id == job_id).delete()
    db.query(JobRecord).filter(JobRecord.id == job_id).delete()
    db.commit()
    return {"status": "cleared"}


@app.get("/artifacts/download")
def download_artifact(path: str = Query(..., description="Path relative to /shared/artifacts")):
    filename = Path(path).name
    try:
        resolved = _resolve_artifact_path(path)
        filename = Path(resolved).name
        return FileResponse(resolved, filename=filename, media_type="application/octet-stream")
    except HTTPException as exc:
        if exc.status_code != 404 or not document_store.is_s3_enabled():
            raise
    try:
        payload = document_store.download_artifact_bytes(path)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=404, detail=f"Artifact not found in object store: {exc}"
        ) from exc
    return StreamingResponse(
        iter([payload]),
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/workspace/download")
def download_workspace_file(
    path: str = Query(..., description="Path relative to /shared/workspace"),
):
    resolved = _resolve_workspace_path(path)
    filename = Path(resolved).name
    return FileResponse(resolved, filename=filename, media_type="application/octet-stream")


@app.get("/events/stream")
def stream_events(request: Request, once: bool = False):
    def event_generator():
        if once:
            yield "data: {}\n\n"
            return
        last_ids = {
            events.JOB_STREAM: "0-0",
            events.PLAN_STREAM: "0-0",
            events.TASK_STREAM: "0-0",
            events.CRITIC_STREAM: "0-0",
            events.POLICY_STREAM: "0-0",
            events.FEEDBACK_STREAM: "0-0",
        }
        while True:
            if request.client is None:
                break
            results = redis_client.xread(last_ids, block=1000, count=10)
            for stream_name, messages in results:
                for message_id, data in messages:
                    last_ids[stream_name] = message_id
                    payload = data.get("data")
                    yield f"data: {payload}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/memory/write", response_model=models.MemoryEntry)
def write_memory(entry: models.MemoryWrite, db: Session = Depends(get_db)) -> models.MemoryEntry:
    try:
        return memory_store.write_memory(db, entry)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        detail = str(exc)
        if "memory_conflict" in detail:
            raise HTTPException(status_code=409, detail=detail) from exc
        raise HTTPException(status_code=400, detail=detail) from exc


@app.get("/memory/read", response_model=List[models.MemoryEntry])
def read_memory(
    name: str = Query(...),
    scope: models.MemoryScope | None = Query(None),
    key: str | None = Query(None),
    job_id: str | None = Query(None),
    user_id: str | None = Query(None),
    project_id: str | None = Query(None),
    limit: int = Query(50, ge=1, le=200),
    include_expired: bool = Query(False),
    db: Session = Depends(get_db),
) -> List[models.MemoryEntry]:
    query = models.MemoryQuery(
        name=name,
        scope=scope,
        key=key,
        job_id=job_id,
        user_id=user_id,
        project_id=project_id,
        limit=limit,
        include_expired=include_expired,
    )
    try:
        return memory_store.read_memory(db, query)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/memory/specs", response_model=List[models.MemorySpec])
def list_memory_specs() -> List[models.MemorySpec]:
    return memory_store.list_memory_specs()


@app.delete("/memory/delete", response_model=models.MemoryEntry)
def delete_memory(
    name: str = Query(...),
    scope: models.MemoryScope | None = Query(None),
    key: str | None = Query(None),
    job_id: str | None = Query(None),
    user_id: str | None = Query(None),
    project_id: str | None = Query(None),
    db: Session = Depends(get_db),
) -> models.MemoryEntry:
    try:
        return memory_store.delete_memory(
            db,
            name=name,
            scope=scope,
            key=key,
            job_id=job_id,
            user_id=user_id,
            project_id=project_id,
        )
    except KeyError as exc:
        detail = str(exc)
        status_code = 404 if "memory_not_found" in detail or "unknown_memory" in detail else 400
        raise HTTPException(status_code=status_code, detail=detail) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/memory/semantic/write")
def write_semantic_memory(
    payload: dict[str, Any],
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="invalid_semantic_payload")

    fact = _semantic_normalize_text(payload.get("fact"), max_len=2400)
    if not fact:
        raise HTTPException(status_code=400, detail="fact_required")
    subject = _semantic_normalize_text(payload.get("subject"), max_len=240) or "general"
    namespace = _semantic_normalize_text(payload.get("namespace"), max_len=120) or "general"
    aliases = _semantic_normalize_list(payload.get("aliases"), max_items=24)
    keywords = _semantic_normalize_list(payload.get("keywords"), max_items=48)
    if not keywords:
        keywords = sorted(_semantic_tokens(subject, fact))[:24]

    confidence_raw = payload.get("confidence", 0.8)
    if not isinstance(confidence_raw, (int, float)):
        raise HTTPException(status_code=400, detail="confidence_must_be_number")
    confidence = float(confidence_raw)
    if confidence < 0 or confidence > 1:
        raise HTTPException(status_code=400, detail="confidence_out_of_range")

    source = _semantic_normalize_text(payload.get("source"), max_len=120) or "manual"
    source_ref = _semantic_normalize_text(payload.get("source_ref"), max_len=300)
    reasoning = _semantic_normalize_text(payload.get("reasoning"), max_len=1200)

    key = _semantic_normalize_text(payload.get("key"), max_len=200)
    if not key:
        key = _semantic_build_key(namespace, subject, fact)
    user_id = _semantic_normalize_text(payload.get("user_id"), max_len=120) or _semantic_default_user_id()
    job_id = _semantic_normalize_text(payload.get("job_id"), max_len=120)

    metadata_raw = payload.get("metadata")
    if metadata_raw is not None and not isinstance(metadata_raw, dict):
        raise HTTPException(status_code=400, detail="metadata_must_be_object")
    metadata = dict(metadata_raw or {})
    metadata.setdefault("semantic", True)
    metadata.setdefault("semantic_namespace", namespace)
    metadata.setdefault("semantic_subject", subject)
    if job_id:
        metadata.setdefault("job_id_source", job_id)

    ttl_seconds = payload.get("ttl_seconds")
    if ttl_seconds is not None:
        if not isinstance(ttl_seconds, int) or ttl_seconds <= 0:
            raise HTTPException(status_code=400, detail="ttl_seconds_must_be_positive_integer")

    semantic_record = {
        "type": "semantic_fact",
        "namespace": namespace,
        "subject": subject,
        "fact": fact,
        "aliases": aliases,
        "keywords": keywords,
        "confidence": confidence,
        "source": source,
        "source_ref": source_ref,
        "reasoning": reasoning,
        "query_text": " ".join(
            part for part in [namespace, subject, fact, " ".join(keywords), " ".join(aliases)] if part
        ),
        "captured_at": _utcnow().isoformat(),
    }
    write_request = memory_promotion_service.build_semantic_memory_write(
        user_id=user_id,
        key=key,
        payload=semantic_record,
        metadata=metadata,
        ttl_seconds=ttl_seconds,
    )
    if isinstance(write_request.payload, Mapping):
        write_request.metadata["semantic_subject"] = str(
            write_request.payload.get("subject") or write_request.metadata.get("semantic_subject") or ""
        )
    try:
        entry = memory_store.write_memory(db, write_request)
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"entry": entry.model_dump(), "semantic_record": write_request.payload}


@app.post("/memory/semantic/search")
def search_semantic_memory(
    payload: dict[str, Any],
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="invalid_semantic_payload")
    query = _semantic_normalize_text(payload.get("query"), max_len=600)
    if not query:
        raise HTTPException(status_code=400, detail="query_required")

    namespace_filter = _semantic_normalize_text(payload.get("namespace"), max_len=120).lower()
    subject_filter = _semantic_normalize_text(payload.get("subject"), max_len=240).lower()
    key_filter = _semantic_normalize_text(payload.get("key"), max_len=200)
    user_id = _semantic_normalize_text(payload.get("user_id"), max_len=120) or _semantic_default_user_id()
    include_payload = bool(payload.get("include_payload", True))

    limit_raw = payload.get("limit", 10)
    if not isinstance(limit_raw, int):
        raise HTTPException(status_code=400, detail="limit_must_be_integer")
    limit = max(1, min(limit_raw, 50))

    min_score_raw = payload.get("min_score", 0.01)
    if not isinstance(min_score_raw, (int, float)):
        raise HTTPException(status_code=400, detail="min_score_must_be_number")
    min_score = max(0.0, float(min_score_raw))
    include_expired = bool(payload.get("include_expired", False))

    if key_filter:
        query_model = models.MemoryQuery(
            name="semantic_memory",
            scope=models.MemoryScope.user,
            key=key_filter,
            user_id=user_id,
            limit=1,
            include_expired=include_expired,
        )
        try:
            keyed_entries = memory_store.read_memory(db, query_model)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if keyed_entries:
            entry = keyed_entries[0]
            return {
                "query": query,
                "count": 1,
                "matches": [_semantic_entry_to_match(entry, 1.0, include_payload)],
                "user_id": user_id,
            }
        return {"query": query, "count": 0, "matches": [], "user_id": user_id}

    base_query = models.MemoryQuery(
        name="semantic_memory",
        scope=models.MemoryScope.user,
        user_id=user_id,
        limit=max(limit * 5, 50),
        include_expired=include_expired,
    )
    try:
        entries = memory_store.read_memory(db, base_query)
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    query_tokens = _semantic_tokens(query)
    query_lc = query.lower()
    ranked: list[tuple[float, models.MemoryEntry]] = []
    for entry in entries:
        payload_obj = entry.payload if isinstance(entry.payload, dict) else {}
        fact = _semantic_normalize_text(payload_obj.get("fact"), max_len=2400)
        if not fact:
            continue
        subject = _semantic_normalize_text(payload_obj.get("subject"), max_len=240)
        namespace = _semantic_normalize_text(payload_obj.get("namespace"), max_len=120)
        if namespace_filter and namespace.lower() != namespace_filter:
            continue
        if subject_filter and subject_filter not in subject.lower():
            continue
        aliases = _semantic_normalize_list(payload_obj.get("aliases"), max_items=24)
        keywords = _semantic_normalize_list(payload_obj.get("keywords"), max_items=48)
        doc_tokens = _semantic_tokens(subject, fact, " ".join(aliases), " ".join(keywords), namespace)
        overlap = len(query_tokens.intersection(doc_tokens))
        score = overlap / max(1, len(query_tokens))
        fact_lc = fact.lower()
        subject_lc = subject.lower()
        namespace_lc = namespace.lower()
        if query_lc in fact_lc:
            score += 0.6
        if query_lc in subject_lc:
            score += 0.45
        if query_lc in namespace_lc:
            score += 0.2
        confidence = payload_obj.get("confidence")
        if isinstance(confidence, (int, float)):
            bounded_confidence = min(max(float(confidence), 0.0), 1.0)
            score += 0.1 * bounded_confidence
        if score < min_score:
            continue
        ranked.append((score, entry))
    ranked.sort(
        key=lambda item: (
            item[0],
            item[1].updated_at.isoformat() if item[1].updated_at else "",
        ),
        reverse=True,
    )
    matches = [_semantic_entry_to_match(entry, score, include_payload) for score, entry in ranked[:limit]]
    return {
        "query": query,
        "count": len(matches),
        "matches": matches,
        "user_id": user_id,
    }


@app.get("/rag/documents")
def list_rag_documents(
    collection_name: str | None = Query(None),
    namespace: str | None = Query(None),
    tenant_id: str | None = Query(None),
    user_id: str | None = Query(None),
    workspace_id: str | None = Query(None),
    query: str | None = Query(None),
    limit: int = Query(100, ge=1, le=500),
) -> dict[str, Any]:
    result = _rag_retriever_request_json(
        "/documents/list",
        method="POST",
        body={
            "collection_name": collection_name,
            "namespace": namespace,
            "tenant_id": tenant_id,
            "user_id": user_id,
            "workspace_id": workspace_id,
            "query": query,
            "limit": limit,
        },
    )
    if not isinstance(result, dict):
        raise HTTPException(status_code=502, detail="rag_retriever_invalid_list_response")
    return result


@app.get("/rag/documents/chunks")
def get_rag_document_chunks(
    document_id: str = Query(..., min_length=1),
    collection_name: str | None = Query(None),
    namespace: str | None = Query(None),
    tenant_id: str | None = Query(None),
    user_id: str | None = Query(None),
    workspace_id: str | None = Query(None),
    limit: int = Query(500, ge=1, le=2000),
) -> dict[str, Any]:
    result = _rag_retriever_request_json(
        "/documents/chunks",
        method="POST",
        body={
            "document_id": document_id,
            "collection_name": collection_name,
            "namespace": namespace,
            "tenant_id": tenant_id,
            "user_id": user_id,
            "workspace_id": workspace_id,
            "limit": limit,
        },
    )
    if not isinstance(result, dict):
        raise HTTPException(status_code=502, detail="rag_retriever_invalid_chunks_response")
    return result


@app.delete("/rag/documents")
def delete_rag_document(
    document_id: str = Query(..., min_length=1),
    collection_name: str | None = Query(None),
    namespace: str | None = Query(None),
    tenant_id: str | None = Query(None),
    user_id: str | None = Query(None),
    workspace_id: str | None = Query(None),
) -> dict[str, Any]:
    result = _rag_retriever_request_json(
        "/documents/delete",
        method="POST",
        body={
            "document_id": document_id,
            "collection_name": collection_name,
            "namespace": namespace,
            "tenant_id": tenant_id,
            "user_id": user_id,
            "workspace_id": workspace_id,
        },
    )
    if not isinstance(result, dict):
        raise HTTPException(status_code=502, detail="rag_retriever_invalid_delete_response")
    return result


@app.post("/rag/index")
def index_rag_document(payload: dict[str, Any] = Body(default_factory=dict)) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="invalid_rag_index_payload")
    mode = str(payload.get("mode") or "").strip().lower()
    if mode not in {"markdown", "text", "workspace_file", "workspace_directory"}:
        raise HTTPException(status_code=400, detail="unsupported_rag_index_mode")
    collection_name = payload.get("collection_name")
    ensure_collection = bool(payload.get("ensure_collection", True))
    namespace = payload.get("namespace")
    tenant_id = payload.get("tenant_id")
    user_id = payload.get("user_id")
    workspace_id = payload.get("workspace_id")
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    result: Any
    if mode == "markdown":
        markdown_text = str(payload.get("markdown_text") or payload.get("content") or "").strip()
        if not markdown_text:
            raise HTTPException(status_code=400, detail="markdown_text_required")
        result = _rag_retriever_request_json(
            "/index/markdown",
            method="POST",
            body={
                "markdown_text": markdown_text,
                "collection_name": collection_name,
                "ensure_collection": ensure_collection,
                "document_id": payload.get("document_id"),
                "source_uri": payload.get("source_uri"),
                "namespace": namespace,
                "tenant_id": tenant_id,
                "user_id": user_id,
                "workspace_id": workspace_id,
                "chunk_size_chars": payload.get("chunk_size_chars"),
                "chunk_overlap_chars": payload.get("chunk_overlap_chars"),
                "max_chunks": payload.get("max_chunks"),
                "metadata": metadata,
            },
        )
    elif mode == "text":
        text = str(payload.get("text") or payload.get("content") or "").strip()
        if not text:
            raise HTTPException(status_code=400, detail="text_required")
        document_id = str(
            payload.get("document_id")
            or payload.get("source_uri")
            or f"manual/{uuid.uuid4()}"
        ).strip()
        result = _rag_retriever_request_json(
            "/index/upsert_texts",
            method="POST",
            body={
                "collection_name": collection_name,
                "ensure_collection": ensure_collection,
                "namespace": namespace,
                "tenant_id": tenant_id,
                "user_id": user_id,
                "workspace_id": workspace_id,
                "entries": [
                    {
                        "document_id": document_id,
                        "text": text,
                        "source_uri": payload.get("source_uri"),
                        "metadata": metadata,
                    }
                ],
            },
        )
    elif mode == "workspace_file":
        path = str(payload.get("path") or "").strip()
        if not path:
            raise HTTPException(status_code=400, detail="path_required")
        result = _rag_retriever_request_json(
            "/index/workspace_file",
            method="POST",
            body={
                "path": path,
                "collection_name": collection_name,
                "ensure_collection": ensure_collection,
                "document_id": payload.get("document_id"),
                "source_uri": payload.get("source_uri"),
                "namespace": namespace,
                "tenant_id": tenant_id,
                "user_id": user_id,
                "workspace_id": workspace_id,
                "chunk_size_chars": payload.get("chunk_size_chars"),
                "chunk_overlap_chars": payload.get("chunk_overlap_chars"),
                "max_chunks": payload.get("max_chunks"),
                "metadata": metadata,
            },
        )
    else:
        directory_path = str(payload.get("directory_path") or "").strip()
        if not directory_path:
            raise HTTPException(status_code=400, detail="directory_path_required")
        result = _rag_retriever_request_json(
            "/index/workspace_directory",
            method="POST",
            body={
                "directory_path": directory_path,
                "collection_name": collection_name,
                "ensure_collection": ensure_collection,
                "namespace": namespace,
                "tenant_id": tenant_id,
                "user_id": user_id,
                "workspace_id": workspace_id,
                "recursive": bool(payload.get("recursive", True)),
                "extensions": payload.get("extensions"),
                "max_files": payload.get("max_files"),
                "chunk_size_chars": payload.get("chunk_size_chars"),
                "chunk_overlap_chars": payload.get("chunk_overlap_chars"),
                "max_chunks_per_file": payload.get("max_chunks_per_file"),
                "metadata": metadata,
            },
        )
    if not isinstance(result, dict):
        raise HTTPException(status_code=502, detail="rag_retriever_invalid_index_response")
    return result


@app.put("/rag/documents")
def replace_rag_document(
    document_id: str = Query(..., min_length=1),
    payload: dict[str, Any] = Body(default_factory=dict),
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="invalid_rag_replace_payload")
    delete_result = _rag_retriever_request_json(
        "/documents/delete",
        method="POST",
        body={
            "document_id": document_id,
            "collection_name": payload.get("collection_name"),
            "namespace": payload.get("namespace"),
            "tenant_id": payload.get("tenant_id"),
            "user_id": payload.get("user_id"),
            "workspace_id": payload.get("workspace_id"),
        },
    )
    if not isinstance(delete_result, dict):
        raise HTTPException(status_code=502, detail="rag_retriever_invalid_delete_response")
    normalized_payload = dict(payload)
    normalized_payload["document_id"] = document_id
    normalized_payload.setdefault("source_uri", document_id)
    index_result = index_rag_document(normalized_payload)
    return {"deleted": delete_result, "indexed": index_result}


@app.get("/jobs/{job_id}/tasks/dlq", response_model=List[models.TaskDlqEntry])
def read_task_dlq(job_id: str, limit: int = Query(25, ge=1, le=200)) -> List[models.TaskDlqEntry]:
    try:
        return _read_task_dlq(job_id, limit)
    except redis.RedisError as exc:
        raise HTTPException(status_code=503, detail=f"redis_error:{exc}") from exc


@app.post("/plans/preflight")
def preflight_plan(
    payload: dict[str, Any],
    job_id: str | None = None,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="invalid_preflight_payload")
    raw_plan = payload.get("plan", payload)
    if not isinstance(raw_plan, dict):
        raise HTTPException(status_code=400, detail="invalid_preflight_plan_payload")
    plan = _parse_plan_payload(raw_plan)
    if plan is None:
        raise HTTPException(status_code=400, detail="invalid_preflight_plan_payload")

    provided_job_context = payload.get("job_context")
    job_context: dict[str, Any] = (
        provided_job_context if isinstance(provided_job_context, dict) else {}
    )
    job: JobRecord | None = None
    if isinstance(job_id, str) and job_id.strip():
        job = db.query(JobRecord).filter(JobRecord.id == job_id).first()
    if not job_context and job and isinstance(job.context_json, dict):
        job_context = job.context_json
    goal_text = str(payload.get("goal") or "").strip()
    if not goal_text and job and isinstance(job.goal, str):
        goal_text = job.goal

    provided_envelope = _normalized_intent_envelope_from_metadata(payload, goal=goal_text)
    provided_graph = payload.get("goal_intent_graph") or payload.get("intent_graph")
    if job and isinstance(job.metadata_json, dict):
        if provided_envelope is None:
            provided_envelope = _normalized_intent_envelope_from_metadata(
                job.metadata_json,
                goal=goal_text or (job.goal if isinstance(job.goal, str) else ""),
            )
        if not isinstance(provided_graph, Mapping):
            provided_graph = job.metadata_json.get("goal_intent_graph")
    preflight_envelope = context_service.build_preflight_context_envelope(
        db=db,
        goal=goal_text,
        provided_job_context=job_context if job_context else None,
        persisted_job_context=job.context_json if job and isinstance(job.context_json, dict) else None,
        normalized_intent_envelope=provided_envelope,
        runtime_metadata={"surface": "plans_preflight"},
    )
    preflight_errors = _compile_plan_preflight(
        plan,
        context_service.preflight_context_view(preflight_envelope),
        goal_text=goal_text,
        normalized_intent_envelope=provided_envelope,
        goal_intent_graph=provided_graph if isinstance(provided_graph, Mapping) else None,
        render_path_mode=planner_contracts.RENDER_PATH_MODE_EXPLICIT,
    )
    diagnostics = _preflight_error_diagnostics(preflight_errors)
    return {
        "valid": not bool(preflight_errors),
        "errors": preflight_errors,
        "diagnostics": diagnostics,
        "intent_inference": _task_intent_summary(plan.tasks, goal_text=goal_text),
    }


@app.post("/intent/clarify")
def clarify_intent(
    payload: dict[str, Any],
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="invalid_intent_payload")
    goal = str(payload.get("goal") or "").strip()
    if not goal:
        raise HTTPException(status_code=400, detail="goal_required")
    explicit_user_id = _semantic_normalize_text(payload.get("user_id"), max_len=120)
    context_obj = _coerce_context_object(
        payload.get("context_json") or payload.get("job_context")
    )
    semantic_user_id = explicit_user_id or _semantic_user_id_from_context(context_obj)
    context_envelope = context_service.build_context_envelope(
        db=db,
        goal=goal,
        context_sources=context_service.collect_context_sources(intent_request_context=context_obj),
        user_id=semantic_user_id,
        runtime_metadata={"surface": "intent_clarify"},
    )
    normalized = _normalize_goal_intent(
        goal,
        db=db,
        user_id=semantic_user_id,
        context_envelope=context_envelope,
    )
    return _normalized_intent_response_payload(normalized)


@app.post("/intent/decompose")
def decompose_intent(
    payload: dict[str, Any],
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="invalid_intent_payload")
    goal = str(payload.get("goal") or "").strip()
    if not goal:
        raise HTTPException(status_code=400, detail="goal_required")
    try:
        interaction_summaries = _normalize_interaction_summaries(
            payload.get("interaction_summaries")
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    compacted_summaries, compaction = _compact_interaction_summaries(interaction_summaries)
    explicit_user_id = _semantic_normalize_text(payload.get("user_id"), max_len=120)
    context_obj = _coerce_context_object(
        payload.get("context_json") or payload.get("job_context")
    )
    semantic_user_id = explicit_user_id or _semantic_user_id_from_context(context_obj)
    context_envelope = context_service.build_context_envelope(
        db=db,
        goal=goal,
        context_sources=context_service.collect_context_sources(intent_request_context=context_obj),
        user_id=semantic_user_id,
        runtime_metadata={"surface": "intent_decompose"},
    )
    normalized = _normalize_goal_intent(
        goal,
        db=db,
        user_id=semantic_user_id,
        interaction_summaries=compacted_summaries or interaction_summaries,
        context_envelope=context_envelope,
        include_decomposition=True,
        assessment_mode_override="heuristic",
    )
    if interaction_summaries:
        normalized = _attach_interaction_compaction_to_envelope(normalized, compaction)
    return _normalized_intent_response_payload(normalized, include_legacy_graph=True)


def _workflow_title_fallback(goal: str, title: str) -> str:
    candidate = str(title or "").strip() or str(goal or "").strip()
    return candidate[:120] or "Workflow Studio draft"


def _compile_workflow_definition_version(
    definition: WorkflowDefinitionRecord,
) -> tuple[models.PlanCreate, models.RunSpec, dict[str, Any]]:
    raw_draft = definition.draft_json if isinstance(definition.draft_json, dict) else {}
    goal_text = str(definition.goal or "").strip()
    job_context = dict(definition.context_json) if isinstance(definition.context_json, dict) else {}
    compile_envelope = context_service.build_preflight_context_envelope(
        db=None,
        goal=goal_text,
        provided_job_context=job_context,
        persisted_job_context=None,
        runtime_metadata={"surface": "workflow_definition_compile"},
    )
    compile_job_context = context_service.preflight_context_view(compile_envelope)
    workflow_interface, workflow_interface_errors, workflow_interface_warnings = (
        _coerce_workflow_interface(
            raw_draft.get("workflowInterface")
            if "workflowInterface" in raw_draft
            else raw_draft.get("workflow_interface")
        )
    )
    if workflow_interface_errors:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "workflow_compile_failed",
                "diagnostics": {
                    "errors": workflow_interface_errors,
                    "warnings": workflow_interface_warnings,
                },
            },
        )
    preview_job_context, workflow_context_errors = _build_workflow_interface_runtime_context(
        workflow_interface,
        base_context=compile_job_context,
        preview=True,
    )
    if workflow_context_errors:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "workflow_compile_failed",
                "diagnostics": {
                    "errors": workflow_context_errors,
                    "warnings": workflow_interface_warnings,
                },
            },
        )
    plan, diagnostics_errors, diagnostics_warnings = _build_plan_from_composer_draft(
        raw_draft,
        goal_text=goal_text,
        job_context=compile_job_context,
    )
    if plan is None:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "workflow_compile_failed",
                "diagnostics": {
                    "errors": diagnostics_errors,
                    "warnings": diagnostics_warnings,
                },
            },
        )
    preflight_errors = _compile_plan_preflight(
        plan,
        preview_job_context,
        goal_text=goal_text,
        goal_intent_graph=None,
        render_path_mode=planner_contracts.RENDER_PATH_MODE_AUTO,
    )
    preflight_errors = _merge_preflight_errors(
        preflight_errors,
        _compile_plan_runtime_conformance_errors(plan),
    )
    try:
        run_spec = run_specs.plan_to_run_spec(plan, kind=models.RunKind.studio)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "workflow_run_spec_compile_failed",
                "diagnostics": {
                    "errors": [
                        {
                            "code": "workflow.run_spec_compile_failed",
                            "message": str(exc),
                        }
                    ],
                    "warnings": diagnostics_warnings,
                },
            },
        ) from exc
    if preflight_errors:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "workflow_preflight_failed",
                "preflight_errors": preflight_errors,
                "diagnostics": {
                    "errors": diagnostics_errors,
                    "warnings": diagnostics_warnings,
                },
            },
        )
    return plan, run_spec, {
        "workflow_interface": workflow_interface,
        "diagnostics": {
            "errors": diagnostics_errors,
            "warnings": diagnostics_warnings,
        },
        "preflight_errors": preflight_errors,
    }


@app.post("/workflows/definitions", response_model=models.WorkflowDefinition)
def create_workflow_definition(
    payload: models.WorkflowDefinitionCreate,
    db: Session = Depends(get_db),
) -> models.WorkflowDefinition:
    now = _utcnow()
    normalized_user_id = _semantic_normalize_text(payload.user_id, max_len=120) or _semantic_user_id_from_context(
        payload.context_json
    )
    record = WorkflowDefinitionRecord(
        id=str(uuid.uuid4()),
        title=_workflow_title_fallback(payload.goal, payload.title),
        goal=str(payload.goal or "").strip(),
        context_json=dict(payload.context_json) if isinstance(payload.context_json, dict) else {},
        draft_json=dict(payload.draft) if isinstance(payload.draft, dict) else {},
        user_id=normalized_user_id or None,
        metadata_json=dict(payload.metadata) if isinstance(payload.metadata, dict) else {},
        created_at=now,
        updated_at=now,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return _workflow_definition_from_record(record)


@app.get("/workflows/definitions", response_model=List[models.WorkflowDefinition])
def list_workflow_definitions(
    user_id: str | None = Query(None),
    db: Session = Depends(get_db),
) -> List[models.WorkflowDefinition]:
    query = db.query(WorkflowDefinitionRecord)
    normalized_user_id = _semantic_normalize_text(user_id, max_len=120)
    if normalized_user_id:
        query = query.filter(WorkflowDefinitionRecord.user_id == normalized_user_id)
    records = query.order_by(WorkflowDefinitionRecord.updated_at.desc()).all()
    return [_workflow_definition_from_record(record) for record in records]


@app.get("/workflows/definitions/{definition_id}", response_model=models.WorkflowDefinition)
def get_workflow_definition(
    definition_id: str,
    db: Session = Depends(get_db),
) -> models.WorkflowDefinition:
    record = (
        db.query(WorkflowDefinitionRecord)
        .filter(WorkflowDefinitionRecord.id == definition_id)
        .first()
    )
    if record is None:
        raise HTTPException(status_code=404, detail="workflow_definition_not_found")
    return _workflow_definition_from_record(record)


@app.put("/workflows/definitions/{definition_id}", response_model=models.WorkflowDefinition)
def update_workflow_definition(
    definition_id: str,
    payload: models.WorkflowDefinitionUpdate,
    db: Session = Depends(get_db),
) -> models.WorkflowDefinition:
    record = (
        db.query(WorkflowDefinitionRecord)
        .filter(WorkflowDefinitionRecord.id == definition_id)
        .first()
    )
    if record is None:
        raise HTTPException(status_code=404, detail="workflow_definition_not_found")
    next_goal = str(payload.goal if payload.goal is not None else record.goal or "").strip()
    next_title = (
        str(payload.title).strip()
        if payload.title is not None
        else str(record.title or "").strip()
    )
    record.title = _workflow_title_fallback(next_goal, next_title)
    record.goal = next_goal
    if payload.context_json is not None and isinstance(payload.context_json, dict):
        record.context_json = dict(payload.context_json)
    if payload.draft is not None and isinstance(payload.draft, dict):
        record.draft_json = dict(payload.draft)
    if payload.user_id is not None:
        record.user_id = _semantic_normalize_text(payload.user_id, max_len=120) or None
    if payload.metadata is not None and isinstance(payload.metadata, dict):
        record.metadata_json = dict(payload.metadata)
    record.updated_at = _utcnow()
    db.commit()
    db.refresh(record)
    return _workflow_definition_from_record(record)


@app.delete("/workflows/definitions/{definition_id}")
def delete_workflow_definition(
    definition_id: str,
    db: Session = Depends(get_db),
) -> dict[str, bool]:
    record = (
        db.query(WorkflowDefinitionRecord)
        .filter(WorkflowDefinitionRecord.id == definition_id)
        .first()
    )
    if record is None:
        raise HTTPException(status_code=404, detail="workflow_definition_not_found")
    db.query(WorkflowRunRecord).filter(
        WorkflowRunRecord.definition_id == definition_id
    ).delete(synchronize_session=False)
    db.query(WorkflowTriggerRecord).filter(
        WorkflowTriggerRecord.definition_id == definition_id
    ).delete(synchronize_session=False)
    db.query(WorkflowVersionRecord).filter(
        WorkflowVersionRecord.definition_id == definition_id
    ).delete(synchronize_session=False)
    db.delete(record)
    db.commit()
    return {"ok": True}


@app.get(
    "/workflows/definitions/{definition_id}/versions",
    response_model=List[models.WorkflowVersion],
)
def list_workflow_versions(
    definition_id: str,
    db: Session = Depends(get_db),
) -> List[models.WorkflowVersion]:
    definition = (
        db.query(WorkflowDefinitionRecord)
        .filter(WorkflowDefinitionRecord.id == definition_id)
        .first()
    )
    if definition is None:
        raise HTTPException(status_code=404, detail="workflow_definition_not_found")
    records = (
        db.query(WorkflowVersionRecord)
        .filter(WorkflowVersionRecord.definition_id == definition_id)
        .order_by(WorkflowVersionRecord.version_number.desc())
        .all()
    )
    return [_workflow_version_from_record(record) for record in records]


@app.post(
    "/workflows/definitions/{definition_id}/publish",
    response_model=models.WorkflowVersion,
)
def publish_workflow_definition(
    definition_id: str,
    payload: dict[str, Any] = Body(default_factory=dict),
    db: Session = Depends(get_db),
) -> models.WorkflowVersion:
    definition = (
        db.query(WorkflowDefinitionRecord)
        .filter(WorkflowDefinitionRecord.id == definition_id)
        .first()
    )
    if definition is None:
        raise HTTPException(status_code=404, detail="workflow_definition_not_found")
    compiled_plan, run_spec, publish_meta = _compile_workflow_definition_version(definition)
    latest = (
        db.query(WorkflowVersionRecord)
        .filter(WorkflowVersionRecord.definition_id == definition_id)
        .order_by(WorkflowVersionRecord.version_number.desc())
        .first()
    )
    next_version_number = 1 if latest is None else int(latest.version_number) + 1
    merged_metadata = dict(definition.metadata_json or {})
    if isinstance(payload, dict) and isinstance(payload.get("metadata"), dict):
        merged_metadata.update(dict(payload["metadata"]))
    merged_metadata["publish"] = publish_meta
    merged_metadata["run_spec"] = run_spec.model_dump(mode="json")
    record = WorkflowVersionRecord(
        id=str(uuid.uuid4()),
        definition_id=definition.id,
        version_number=next_version_number,
        title=definition.title,
        goal=definition.goal,
        context_json=dict(definition.context_json or {}),
        draft_json=dict(definition.draft_json or {}),
        compiled_plan_json=compiled_plan.model_dump(mode="json"),
        user_id=definition.user_id,
        metadata_json=merged_metadata,
        created_at=_utcnow(),
    )
    definition.updated_at = _utcnow()
    db.add(record)
    db.commit()
    db.refresh(record)
    return _workflow_version_from_record(record)


@app.post(
    "/workflows/definitions/{definition_id}/triggers",
    response_model=models.WorkflowTrigger,
)
def create_workflow_trigger(
    definition_id: str,
    payload: models.WorkflowTriggerCreate,
    db: Session = Depends(get_db),
) -> models.WorkflowTrigger:
    definition = (
        db.query(WorkflowDefinitionRecord)
        .filter(WorkflowDefinitionRecord.id == definition_id)
        .first()
    )
    if definition is None:
        raise HTTPException(status_code=404, detail="workflow_definition_not_found")
    now = _utcnow()
    trigger_title = str(payload.title or "").strip() or f"{definition.title} trigger"
    record = WorkflowTriggerRecord(
        id=str(uuid.uuid4()),
        definition_id=definition.id,
        title=trigger_title,
        trigger_type=payload.trigger_type.value,
        enabled=bool(payload.enabled),
        config_json=dict(payload.config) if isinstance(payload.config, dict) else {},
        user_id=_semantic_normalize_text(
            payload.user_id if payload.user_id is not None else definition.user_id,
            max_len=120,
        )
        or None,
        metadata_json=dict(payload.metadata) if isinstance(payload.metadata, dict) else {},
        created_at=now,
        updated_at=now,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return _workflow_trigger_from_record(record)


@app.get(
    "/workflows/definitions/{definition_id}/triggers",
    response_model=List[models.WorkflowTrigger],
)
def list_workflow_triggers(
    definition_id: str,
    db: Session = Depends(get_db),
) -> List[models.WorkflowTrigger]:
    definition = (
        db.query(WorkflowDefinitionRecord)
        .filter(WorkflowDefinitionRecord.id == definition_id)
        .first()
    )
    if definition is None:
        raise HTTPException(status_code=404, detail="workflow_definition_not_found")
    records = (
        db.query(WorkflowTriggerRecord)
        .filter(WorkflowTriggerRecord.definition_id == definition_id)
        .order_by(WorkflowTriggerRecord.updated_at.desc())
        .all()
    )
    return [_workflow_trigger_from_record(record) for record in records]


@app.put("/workflows/triggers/{trigger_id}", response_model=models.WorkflowTrigger)
def update_workflow_trigger(
    trigger_id: str,
    payload: models.WorkflowTriggerUpdate,
    db: Session = Depends(get_db),
) -> models.WorkflowTrigger:
    record = (
        db.query(WorkflowTriggerRecord)
        .filter(WorkflowTriggerRecord.id == trigger_id)
        .first()
    )
    if record is None:
        raise HTTPException(status_code=404, detail="workflow_trigger_not_found")
    if payload.title is not None:
        record.title = str(payload.title).strip() or record.title
    if payload.enabled is not None:
        record.enabled = bool(payload.enabled)
    if payload.config is not None and isinstance(payload.config, dict):
        record.config_json = dict(payload.config)
    if payload.user_id is not None:
        record.user_id = _semantic_normalize_text(payload.user_id, max_len=120) or None
    if payload.metadata is not None and isinstance(payload.metadata, dict):
        record.metadata_json = dict(payload.metadata)
    record.updated_at = _utcnow()
    db.commit()
    db.refresh(record)
    return _workflow_trigger_from_record(record)


@app.get(
    "/workflows/definitions/{definition_id}/runs",
    response_model=List[models.WorkflowRun],
)
def list_workflow_runs(
    definition_id: str,
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
) -> List[models.WorkflowRun]:
    definition = (
        db.query(WorkflowDefinitionRecord)
        .filter(WorkflowDefinitionRecord.id == definition_id)
        .first()
    )
    if definition is None:
        raise HTTPException(status_code=404, detail="workflow_definition_not_found")
    records = (
        db.query(WorkflowRunRecord)
        .filter(WorkflowRunRecord.definition_id == definition_id)
        .order_by(WorkflowRunRecord.created_at.desc())
        .limit(limit)
        .all()
    )
    job_ids = [record.job_id for record in records if isinstance(record.job_id, str)]
    job_map = (
        {
            job.id: job
            for job in db.query(JobRecord).filter(JobRecord.id.in_(job_ids)).all()
        }
        if job_ids
        else {}
    )
    latest_failures = _latest_task_failures_for_jobs(db, job_ids)
    return [
        _workflow_run_from_record(
            record,
            job_record=job_map.get(record.job_id),
            latest_task_failure=latest_failures.get(record.job_id),
        )
        for record in records
    ]


def _run_workflow_version_internal(
    *,
    version: WorkflowVersionRecord,
    definition: WorkflowDefinitionRecord,
    db: Session,
    payload: dict[str, Any],
    trigger: WorkflowTriggerRecord | None = None,
    source: str,
) -> models.WorkflowRunResult:
    compiled_plan = _workflow_version_plan(version)
    if compiled_plan is None:
        raise HTTPException(status_code=400, detail="workflow_version_plan_invalid")
    run_spec = _workflow_version_run_spec(version)
    use_postgres_scheduler = STUDIO_RUN_SCHEDULER_ENABLED and run_spec is not None
    trigger_context = (
        trigger.config_json.get("context_json")
        if isinstance(trigger, WorkflowTriggerRecord) and isinstance(trigger.config_json, dict)
        else None
    )
    trigger_inputs = (
        trigger.config_json.get("inputs")
        if isinstance(trigger, WorkflowTriggerRecord) and isinstance(trigger.config_json, dict)
        else None
    )
    request_context = payload.get("context_json") if isinstance(payload, dict) else None
    workflow_interface, workflow_interface_errors, _workflow_interface_warnings = (
        _coerce_workflow_interface(
            version.draft_json.get("workflowInterface")
            if isinstance(version.draft_json, dict) and "workflowInterface" in version.draft_json
            else version.draft_json.get("workflow_interface")
            if isinstance(version.draft_json, dict)
            else None
        )
    )
    if workflow_interface_errors:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "workflow_interface_invalid",
                "diagnostics": {"errors": workflow_interface_errors, "warnings": []},
            },
        )
    context_envelope, runtime_inputs = context_service.build_workflow_runtime_context_envelope(
        db=db,
        goal=version.goal or definition.goal or definition.title,
        version_context=version.context_json if isinstance(version.context_json, Mapping) else None,
        trigger_context=trigger_context if isinstance(trigger_context, Mapping) else None,
        request_context=request_context if isinstance(request_context, Mapping) else None,
        trigger_inputs=trigger_inputs if isinstance(trigger_inputs, Mapping) else None,
        explicit_inputs=payload.get("inputs") if isinstance(payload, dict) and isinstance(payload.get("inputs"), dict) else None,
        runtime_metadata={"surface": source},
    )
    context_json = context_service.workflow_runtime_context_view(context_envelope)
    context_json, workflow_context_errors = _build_workflow_interface_runtime_context(
        workflow_interface,
        base_context=context_json,
        db=db,
        explicit_inputs=runtime_inputs,
        preview=False,
    )
    if workflow_context_errors:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "workflow_inputs_invalid",
                "diagnostics": {"errors": workflow_context_errors, "warnings": []},
            },
        )
    run_preflight_errors = _merge_preflight_errors(
        _compile_plan_preflight(
            compiled_plan,
            context_json,
            goal_text=version.goal or definition.goal or definition.title,
            goal_intent_graph=None,
            render_path_mode=planner_contracts.RENDER_PATH_MODE_AUTO,
        ),
        _compile_plan_runtime_conformance_errors(compiled_plan),
    )
    if run_preflight_errors:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "workflow_preflight_failed",
                "preflight_errors": run_preflight_errors,
                "diagnostics": {
                    "errors": _preflight_error_diagnostics(run_preflight_errors),
                    "warnings": [],
                },
            },
        )
    workflow_envelope = _plan_derived_normalized_intent_envelope(
        goal=version.goal or definition.goal or definition.title,
        plan=compiled_plan,
        source="workflow_compiled_plan",
    )
    priority_raw = payload.get("priority", 0) if isinstance(payload, dict) else 0
    try:
        priority = int(priority_raw)
    except (TypeError, ValueError):
        priority = 0
    idempotency_key = (
        str(payload.get("idempotency_key") or "").strip()
        if isinstance(payload, dict)
        else ""
    )
    metadata_overrides = {
        "workflow_source": "studio",
        "render_path_mode": planner_contracts.RENDER_PATH_MODE_AUTO,
        "workflow_definition_id": definition.id,
        "workflow_version_id": version.id,
        "goal_intent_profile": (
            workflow_contracts.dump_goal_intent_profile(workflow_envelope.profile) or {}
        ),
        "normalized_intent_envelope": (
            workflow_contracts.dump_normalized_intent_envelope(workflow_envelope) or {}
        ),
        "goal_intent_graph": workflow_contracts.dump_intent_graph(workflow_envelope.graph),
    }
    if use_postgres_scheduler:
        metadata_overrides["scheduler_mode"] = POSTGRES_RUN_SPEC_SCHEDULER_MODE
    if trigger is not None:
        metadata_overrides["workflow_trigger_id"] = trigger.id
    job = _create_job_internal(
        models.JobCreate(
            goal=version.goal or definition.goal or definition.title,
            context_json=context_json,
            priority=priority,
            idempotency_key=idempotency_key or None,
        ),
        db,
        emit_job_created_event=False,
        metadata_overrides=metadata_overrides,
    )
    plan_record = _create_plan_internal(
        compiled_plan,
        job_id=job.id,
        db=db,
        emit_plan_created_event=not use_postgres_scheduler,
    )
    now = _utcnow()
    run_metadata = {"source": source}
    if use_postgres_scheduler:
        run_metadata["scheduler_mode"] = POSTGRES_RUN_SPEC_SCHEDULER_MODE
    if isinstance(payload, dict) and isinstance(payload.get("metadata"), dict):
        run_metadata.update(dict(payload["metadata"]))
    if trigger is not None:
        run_metadata["trigger_type"] = trigger.trigger_type
    if runtime_inputs:
        run_metadata["workflow_input_keys"] = sorted(str(key) for key in runtime_inputs.keys())
    run_record = WorkflowRunRecord(
        id=str(uuid.uuid4()),
        definition_id=definition.id,
        version_id=version.id,
        trigger_id=trigger.id if trigger is not None else None,
        title=version.title or definition.title,
        goal=version.goal or definition.goal or definition.title,
        requested_context_json=context_json,
        job_id=job.id,
        plan_id=plan_record.id,
        user_id=version.user_id or definition.user_id,
        metadata_json=run_metadata,
        created_at=now,
        updated_at=now,
    )
    db.add(run_record)
    job_record = db.query(JobRecord).filter(JobRecord.id == job.id).first()
    if job_record is not None:
        job_metadata = dict(job_record.metadata_json or {})
        job_metadata["workflow_run_id"] = run_record.id
        if trigger is not None:
            job_metadata["workflow_trigger_id"] = trigger.id
        job_record.metadata_json = job_metadata
        job_record.updated_at = now
    db.commit()
    db.refresh(run_record)
    if use_postgres_scheduler:
        _schedule_workflow_run(run_record.id, correlation_id=str(uuid.uuid4()))
    if job_record is not None:
        db.refresh(job_record)
        job = _job_from_record(job_record)
    return models.WorkflowRunResult(
        workflow_definition=_workflow_definition_from_record(definition),
        workflow_version=_workflow_version_from_record(version),
        workflow_run=_workflow_run_from_record(run_record, job_record=job_record),
        job=job,
        plan=plan_record,
    )


@app.post(
    "/workflows/versions/{version_id}/run",
    response_model=models.WorkflowRunResult,
)
def run_workflow_version(
    version_id: str,
    payload: dict[str, Any] = Body(default_factory=dict),
    db: Session = Depends(get_db),
) -> models.WorkflowRunResult:
    version = (
        db.query(WorkflowVersionRecord)
        .filter(WorkflowVersionRecord.id == version_id)
        .first()
    )
    if version is None:
        raise HTTPException(status_code=404, detail="workflow_version_not_found")
    definition = (
        db.query(WorkflowDefinitionRecord)
        .filter(WorkflowDefinitionRecord.id == version.definition_id)
        .first()
    )
    if definition is None:
        raise HTTPException(status_code=404, detail="workflow_definition_not_found")
    return _run_workflow_version_internal(
        version=version,
        definition=definition,
        db=db,
        payload=payload if isinstance(payload, dict) else {},
        trigger=None,
        source="workflow_version_run",
    )


@app.post(
    "/workflows/triggers/{trigger_id}/invoke",
    response_model=models.WorkflowRunResult,
)
def invoke_workflow_trigger(
    trigger_id: str,
    payload: dict[str, Any] = Body(default_factory=dict),
    db: Session = Depends(get_db),
) -> models.WorkflowRunResult:
    trigger = (
        db.query(WorkflowTriggerRecord)
        .filter(WorkflowTriggerRecord.id == trigger_id)
        .first()
    )
    if trigger is None:
        raise HTTPException(status_code=404, detail="workflow_trigger_not_found")
    if not bool(trigger.enabled):
        raise HTTPException(status_code=400, detail="workflow_trigger_disabled")
    definition = (
        db.query(WorkflowDefinitionRecord)
        .filter(WorkflowDefinitionRecord.id == trigger.definition_id)
        .first()
    )
    if definition is None:
        raise HTTPException(status_code=404, detail="workflow_definition_not_found")
    requested_version_id = str(payload.get("version_id") or "").strip() if isinstance(payload, dict) else ""
    version_query = db.query(WorkflowVersionRecord).filter(
        WorkflowVersionRecord.definition_id == definition.id
    )
    if requested_version_id:
        version_query = version_query.filter(WorkflowVersionRecord.id == requested_version_id)
    version = version_query.order_by(WorkflowVersionRecord.version_number.desc()).first()
    if version is None:
        raise HTTPException(status_code=404, detail="workflow_version_not_found")
    return _run_workflow_version_internal(
        version=version,
        definition=definition,
        db=db,
        payload=payload if isinstance(payload, dict) else {},
        trigger=trigger,
        source="workflow_trigger_invoke",
    )


@app.post("/composer/compile")
def compile_composer_draft(
    payload: dict[str, Any],
    job_id: str | None = None,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="invalid_composer_payload")
    raw_draft = payload.get("draft", payload)
    if not isinstance(raw_draft, dict):
        raise HTTPException(status_code=400, detail="invalid_composer_draft")

    job_context = _coerce_context_object(payload.get("job_context"))
    if not job_context:
        job_context = _coerce_context_object(
            raw_draft.get("contextJson") if "contextJson" in raw_draft else raw_draft.get("context_json")
        )
    job: JobRecord | None = None
    if isinstance(job_id, str) and job_id.strip():
        job = db.query(JobRecord).filter(JobRecord.id == job_id).first()
    if not job_context and job and isinstance(job.context_json, dict):
        job_context = job.context_json
    goal_text = str(payload.get("goal") or "").strip()
    if not goal_text:
        goal_text = str(raw_draft.get("goal") or raw_draft.get("job_goal") or "").strip()
    if not goal_text and job and isinstance(job.goal, str):
        goal_text = job.goal
    provided_envelope = _normalized_intent_envelope_from_metadata(payload, goal=goal_text)
    if provided_envelope is None and isinstance(raw_draft, Mapping):
        provided_envelope = _normalized_intent_envelope_from_metadata(raw_draft, goal=goal_text)
    provided_graph = payload.get("goal_intent_graph") or payload.get("intent_graph")
    if not isinstance(provided_graph, Mapping):
        provided_graph = raw_draft.get("goal_intent_graph") or raw_draft.get("intent_graph")
    if job and isinstance(job.metadata_json, dict):
        if provided_envelope is None:
            provided_envelope = _normalized_intent_envelope_from_metadata(
                job.metadata_json,
                goal=goal_text or (job.goal if isinstance(job.goal, str) else ""),
            )
        if not isinstance(provided_graph, Mapping):
            provided_graph = job.metadata_json.get("goal_intent_graph")
    compile_envelope = context_service.build_preflight_context_envelope(
        db=db,
        goal=goal_text,
        provided_job_context=job_context if job_context else None,
        persisted_job_context=job.context_json if job and isinstance(job.context_json, dict) else None,
        normalized_intent_envelope=provided_envelope,
        runtime_metadata={"surface": "composer_compile"},
    )
    compile_job_context = context_service.preflight_context_view(compile_envelope)

    workflow_interface, workflow_interface_errors, workflow_interface_warnings = (
        _coerce_workflow_interface(
            raw_draft.get("workflowInterface")
            if "workflowInterface" in raw_draft
            else raw_draft.get("workflow_interface")
        )
    )
    preview_job_context, workflow_context_errors = _build_workflow_interface_runtime_context(
        workflow_interface,
        base_context=compile_job_context,
        explicit_inputs=payload.get("inputs") if isinstance(payload.get("inputs"), dict) else None,
        preview=True,
    )
    plan, diagnostics_errors, diagnostics_warnings = _build_plan_from_composer_draft(
        raw_draft,
        goal_text=goal_text,
        job_context=compile_job_context,
    )
    diagnostics_errors.extend(workflow_interface_errors)
    diagnostics_errors.extend(workflow_context_errors)
    diagnostics_warnings.extend(workflow_interface_warnings)
    preflight_errors: dict[str, str] = {}
    run_spec: models.RunSpec | None = None
    if plan is not None:
        try:
            run_spec = run_specs.plan_to_run_spec(plan, kind=models.RunKind.studio)
        except ValueError as exc:
            diagnostics_errors.append(
                {
                    "code": "draft.run_spec_compile_failed",
                    "message": str(exc),
                }
            )
        preflight_errors = _compile_plan_preflight(
            plan,
            preview_job_context,
            goal_text=goal_text,
            normalized_intent_envelope=provided_envelope,
            goal_intent_graph=provided_graph if isinstance(provided_graph, Mapping) else None,
            render_path_mode=planner_contracts.RENDER_PATH_MODE_AUTO,
        )
        for diagnostic in _preflight_error_diagnostics(preflight_errors):
            diagnostics_errors.append(
                {
                    "code": diagnostic.get("code") or "preflight_error",
                    "field": diagnostic.get("field"),
                    "message": diagnostic.get("message") or "",
                }
            )

    valid = plan is not None and not diagnostics_errors
    return {
        "valid": valid,
        "diagnostics": {
            "valid": not diagnostics_errors,
            "errors": diagnostics_errors,
            "warnings": diagnostics_warnings,
        },
        "plan": plan.model_dump() if plan is not None else None,
        "run_spec": run_spec.model_dump(mode="json") if run_spec is not None else None,
        "preflight_errors": preflight_errors,
        "intent_inference": _task_intent_summary(plan.tasks, goal_text=goal_text)
        if plan is not None
        else {},
    }


@app.post("/composer/recommend_capabilities")
def recommend_composer_capabilities(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="invalid_recommendation_payload")
    goal = str(payload.get("goal") or "").strip()
    context = _coerce_context_object(payload.get("context_json") or payload.get("job_context"))
    raw_draft = payload.get("draft", payload.get("composer_draft"))
    draft_nodes = _coerce_composer_nodes(raw_draft.get("nodes") if isinstance(raw_draft, dict) else [])
    include_disabled = bool(payload.get("include_disabled", False))
    max_results_raw = payload.get("max_results", 6)
    try:
        max_results = max(1, min(12, int(max_results_raw)))
    except Exception:  # noqa: BLE001
        max_results = 6

    try:
        capabilities = _collect_recommendation_capabilities(include_disabled=include_disabled)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=500, detail=f"capability_registry_load_failed:{exc}"
        ) from exc
    heuristic_recommendations = _heuristic_capability_recommendations(
        goal=goal,
        context=context,
        capabilities=capabilities,
        draft_nodes=draft_nodes,
        max_results=max_results,
    )

    use_llm = bool(payload.get("use_llm", False))
    if not use_llm:
        return {
            "source": "heuristic",
            "recommendations": heuristic_recommendations,
            "model": None,
        }

    if _composer_recommender_provider is None:
        return {
            "source": "heuristic",
            "recommendations": heuristic_recommendations,
            "model": None,
            "warning": "llm_recommender_unavailable",
        }

    try:
        llm_recommendations = _llm_capability_recommendations(
            goal=goal,
            context=context,
            capabilities=capabilities,
            draft_nodes=draft_nodes,
            max_results=max_results,
            provider=_composer_recommender_provider,
        )
        return {
            "source": "llm",
            "recommendations": llm_recommendations,
            "model": COMPOSER_RECOMMENDER_MODEL or LLM_MODEL_NAME or None,
        }
    except (LLMProviderError, ValueError) as exc:
        return {
            "source": "llm_fallback",
            "recommendations": heuristic_recommendations,
            "model": COMPOSER_RECOMMENDER_MODEL or LLM_MODEL_NAME or None,
            "warning": str(exc),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "source": "llm_fallback",
            "recommendations": heuristic_recommendations,
            "model": COMPOSER_RECOMMENDER_MODEL or LLM_MODEL_NAME or None,
            "warning": f"llm_recommendation_failed:{exc}",
        }


def _create_plan_internal(
    plan: models.PlanCreate,
    *,
    job_id: str,
    db: Session,
    emit_plan_created_event: bool = True,
) -> models.Plan:
    job = db.query(JobRecord).filter(JobRecord.id == job_id).first()
    goal_text = job.goal if job and isinstance(job.goal, str) else ""
    metadata = job.metadata_json if job and isinstance(job.metadata_json, dict) else {}
    job_context = _preflight_job_context(
        db=db,
        goal_text=goal_text,
        job_context=job.context_json if job and isinstance(job.context_json, dict) else {},
        metadata=metadata,
        surface="create_plan_internal_preflight",
    )
    preflight_errors = _compile_plan_preflight(
        plan,
        job_context,
        goal_text=goal_text,
        normalized_intent_envelope=_normalized_intent_envelope_from_metadata(
            metadata,
            goal=goal_text,
        ),
        goal_intent_graph=metadata.get("goal_intent_graph") if isinstance(metadata, dict) else None,
        render_path_mode=planner_contracts.render_path_mode_from_metadata(
            metadata,
        ),
    )
    if preflight_errors:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "plan_preflight_failed",
                "preflight_errors": preflight_errors,
            },
        )
    plan_id = str(uuid.uuid4())
    now = _utcnow()
    record = PlanRecord(
        id=plan_id,
        job_id=job_id,
        planner_version=plan.planner_version,
        created_at=now,
        tasks_summary=plan.tasks_summary,
        dag_edges=plan.dag_edges,
        policy_decision={},
    )
    db.add(record)
    task_intent_profiles: dict[str, dict[str, Any]] = {}
    goal_intent_segments = _goal_intent_segments_from_metadata(
        job.metadata_json if job and isinstance(job.metadata_json, dict) else {}
    )
    for index, task in enumerate(plan.tasks):
        task_id = str(uuid.uuid4())
        task_record = TaskRecord(
            id=task_id,
            job_id=job_id,
            plan_id=plan_id,
            name=task.name,
            description=task.description,
            instruction=task.instruction,
            acceptance_criteria=task.acceptance_criteria,
            expected_output_schema_ref=task.expected_output_schema_ref,
            status=models.TaskStatus.pending.value,
            intent=task.intent.value
            if isinstance(task.intent, models.ToolIntent)
            else task.intent,
            deps=task.deps,
            attempts=0,
            max_attempts=3,
            rework_count=0,
            max_reworks=2,
            assigned_to=None,
            tool_requests=task.tool_requests,
            tool_inputs=_task_record_tool_inputs(
                task.tool_requests,
                task.tool_inputs,
                task.capability_bindings,
            ),
            created_at=now,
            updated_at=now,
            critic_required=1 if task.critic_required else 0,
        )
        db.add(task_record)
        task_intent_value = (
            task.intent.value if isinstance(task.intent, models.ToolIntent) else str(task.intent or "")
        )
        normalized_task_intent = (
            intent_contract.normalize_task_intent(task_intent_value)
            or models.ToolIntent.generate.value
        )
        task_intent_profiles[task_id] = _task_intent_profile_entry(
            task,
            goal_text=goal_text,
            intent_segment=_select_goal_intent_segment_for_task(
                task=task,
                task_index=index,
                task_intent=normalized_task_intent,
                goal_intent_segments=goal_intent_segments,
                total_tasks=len(plan.tasks),
            ),
        )
    if job:
        _merge_task_intent_profiles_into_job_metadata(job, task_intent_profiles)
        job.updated_at = now
    db.commit()
    if emit_plan_created_event:
        payload = _plan_created_payload(plan, job_id)
        payload["correlation_id"] = str(uuid.uuid4())
        _emit_event("plan.created", payload)
    return _plan_from_record(record)


@app.post("/plans", response_model=models.Plan)
def create_plan(plan: models.PlanCreate, job_id: str, db: Session = Depends(get_db)) -> models.Plan:
    return _create_plan_internal(plan, job_id=job_id, db=db)
