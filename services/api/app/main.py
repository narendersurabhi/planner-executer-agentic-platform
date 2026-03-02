from __future__ import annotations

import json
import os
import logging
import hashlib
import re
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Mapping, Sequence

import redis
from fastapi import Body, Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from prometheus_client import Counter, make_asgi_app
from sqlalchemy.orm import Session

from libs.core import (
    capability_search,
    capability_registry,
    document_store,
    events,
    intent_contract,
    logging as core_logging,
    models,
    orchestrator,
    payload_resolver,
    state_machine,
    tool_registry,
)
from libs.core.llm_provider import LLMProvider, LLMProviderError, MockLLMProvider, resolve_provider
from .database import Base, SessionLocal, engine
from .models import EventOutboxRecord, JobRecord, PlanRecord, TaskRecord
from . import memory_store

core_logging.configure_logging("api")
logger = logging.getLogger("api.orchestrator")

app = FastAPI(title="Agentic Planner Executor API")

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
redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
TASK_OUTPUT_KEY_PREFIX = "task_output:"
TASK_RESULT_KEY_PREFIX = "task_result:"

_tool_spec_registry = tool_registry.default_registry(
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


@app.on_event("startup")
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
    ["needs_clarification"],
)
intent_clarification_required_total = Counter(
    "intent_clarification_required_total",
    "Goal intent assessments that require clarification",
)
intent_threshold_evaluations_total = Counter(
    "intent_threshold_evaluations_total",
    "Intent confidence threshold evaluations",
    ["intent", "risk_level", "needs_clarification", "threshold_bucket"],
)
intent_confidence_outcomes_total = Counter(
    "intent_confidence_outcomes_total",
    "Observed outcomes for goal intent confidence calibration",
    ["intent", "risk_level", "outcome", "above_threshold", "confidence_bucket"],
)
intent_decompose_requests_total = Counter(
    "intent_decompose_requests_total",
    "Intent decomposition requests",
    ["mode", "model", "source", "result", "has_summaries"],
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
        tool_inputs=record.tool_inputs or {},
        created_at=record.created_at,
        updated_at=record.updated_at,
        critic_required=bool(record.critic_required),
    )


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
    lowered = str(goal or "").lower()
    high_risk_tokens = (
        "delete",
        "destroy",
        "remove",
        "drop",
        "shutdown",
        "infra",
        "prod",
        "production",
        "payment",
        "billing",
        "credential",
        "secret",
    )
    if any(token in lowered for token in high_risk_tokens):
        return "high_risk_write"
    bounded_tokens = (
        "create",
        "update",
        "write",
        "publish",
        "push",
        "commit",
        "open pr",
        "pull request",
        "render",
        "generate",
    )
    if intent in {"render", "transform", "generate"} and any(
        token in lowered for token in bounded_tokens
    ):
        return "bounded_write"
    if intent == "io" and any(token in lowered for token in ("write", "upload", "save", "push")):
        return "bounded_write"
    return "read_only"


def _resolve_intent_confidence_threshold(intent: str, risk_level: str) -> float:
    base = max(0.0, min(1.0, float(INTENT_MIN_CONFIDENCE)))
    intent_override = INTENT_MIN_CONFIDENCE_BY_INTENT.get(intent)
    risk_override = INTENT_MIN_CONFIDENCE_BY_RISK.get(risk_level)
    candidates = [base]
    if isinstance(intent_override, float):
        candidates.append(intent_override)
    if isinstance(risk_override, float):
        candidates.append(risk_override)
    return round(max(candidates), 3)


def _extract_goal_slot_signals(goal: str, intent: str, risk_level: str) -> dict[str, Any]:
    lowered = str(goal or "").lower()
    output_format = ""
    for token, normalized in (
        ("pdf", "pdf"),
        ("docx", "docx"),
        ("markdown", "md"),
        (".md", "md"),
        ("json", "json"),
        ("csv", "csv"),
        ("xlsx", "xlsx"),
        ("excel", "xlsx"),
        ("html", "html"),
        ("text", "txt"),
        ("txt", "txt"),
    ):
        if token in lowered:
            output_format = normalized
            break
    target_system = ""
    target_candidates = (
        "github",
        "gitlab",
        "jira",
        "slack",
        "notion",
        "confluence",
        "filesystem",
        "workspace",
        "artifacts",
        "gmail",
    )
    for token in target_candidates:
        if token in lowered:
            target_system = token
            break
    safety_constraints = ""
    if any(
        token in lowered
        for token in (
            "read only",
            "read-only",
            "no write",
            "without write",
            "do not delete",
            "safe mode",
            "dry run",
        )
    ):
        safety_constraints = "present"
    return {
        "intent_action": intent or "",
        "output_format": output_format,
        "target_system": target_system,
        "safety_constraints": safety_constraints,
        "risk_level": risk_level,
    }


def _blocking_clarification_slots(intent: str, risk_level: str) -> list[str]:
    slots: list[str] = []
    if "intent_action" in INTENT_CLARIFICATION_BLOCKING_SLOTS:
        slots.append("intent_action")
    if intent in {"render", "generate"} and "output_format" in INTENT_CLARIFICATION_BLOCKING_SLOTS:
        slots.append("output_format")
    if intent == "io" and "target_system" in INTENT_CLARIFICATION_BLOCKING_SLOTS:
        slots.append("target_system")
    if risk_level == "high_risk_write" and "safety_constraints" in INTENT_CLARIFICATION_BLOCKING_SLOTS:
        slots.append("safety_constraints")
    return slots


def _slot_question(slot: str, goal: str) -> str:
    if slot == "intent_action":
        return "What should the system do first (generate, transform, validate, render, or io)?"
    if slot == "output_format":
        return "What output format do you need (for example PDF, DOCX, JSON, or Markdown)?"
    if slot == "target_system":
        return "Which target system should this use (for example GitHub, Jira, Slack, filesystem)?"
    if slot == "safety_constraints":
        return "What safety constraints must be enforced (for example read-only, no deletes, dry-run)?"
    return f"Provide clarification for slot '{slot}' for goal: '{goal[:120]}'."


def _goal_intent_segments_from_metadata(metadata: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(metadata, Mapping):
        return []
    graph = metadata.get("goal_intent_graph")
    if not isinstance(graph, Mapping):
        return []
    raw_segments = graph.get("segments")
    if not isinstance(raw_segments, list):
        return []
    segments: list[dict[str, Any]] = []
    for raw_segment in raw_segments:
        normalized = _normalize_task_intent_profile_segment(raw_segment)
        if normalized is not None:
            segments.append(normalized)
    return segments


def _assess_goal_intent(goal: str) -> dict[str, Any]:
    inference = intent_contract.infer_task_intent_from_goal_with_metadata(goal)
    intent = inference.intent
    risk_level = _infer_goal_risk_level(goal, intent)
    threshold = _resolve_intent_confidence_threshold(intent, risk_level)
    confidence = round(float(inference.confidence), 3)
    slot_values = _extract_goal_slot_signals(goal, intent, risk_level)
    blocking_slots = _blocking_clarification_slots(intent, risk_level)
    missing_slots = [
        slot_name
        for slot_name in blocking_slots
        if not str(slot_values.get(slot_name) or "").strip()
    ]
    low_confidence = confidence < threshold
    # Low-confidence intent inference is treated as missing intent_action for slot-filling.
    if low_confidence and "intent_action" in blocking_slots and "intent_action" not in missing_slots:
        missing_slots.append("intent_action")
    requires_blocking_clarification = bool(missing_slots)
    needs_clarification = bool(missing_slots)
    intent_assessments_total.labels(
        needs_clarification=str(bool(needs_clarification)).lower()
    ).inc()
    intent_threshold_evaluations_total.labels(
        intent=intent,
        risk_level=risk_level,
        needs_clarification=str(bool(needs_clarification)).lower(),
        threshold_bucket=_threshold_bucket(threshold),
    ).inc()
    if requires_blocking_clarification:
        intent_clarification_required_total.inc()
    questions: list[str] = []
    for slot_name in missing_slots:
        questions.append(_slot_question(slot_name, goal))
    return {
        "intent": intent,
        "source": inference.source,
        "confidence": confidence,
        "risk_level": risk_level,
        "threshold": threshold,
        "low_confidence": low_confidence,
        "needs_clarification": needs_clarification,
        "requires_blocking_clarification": requires_blocking_clarification,
        "questions": questions,
        "blocking_slots": blocking_slots,
        "missing_slots": missing_slots,
        "slot_values": slot_values,
        "clarification_mode": "targeted_slot_filling",
    }


def _publish_envelope_to_redis(stream: str, envelope_json: str) -> tuple[bool, str | None]:
    retries = max(1, EVENT_OUTBOX_REDIS_RETRIES)
    sleep_s = max(0.0, EVENT_OUTBOX_REDIS_RETRY_SLEEP_S)
    for attempt in range(1, retries + 1):
        try:
            redis_client.xadd(stream, {"data": envelope_json})
            return True, None
        except redis.RedisError as exc:
            if attempt >= retries:
                return False, str(exc)
            if sleep_s > 0:
                time.sleep(sleep_s)
    return False, "unknown_redis_publish_error"


def _insert_outbox_event(stream: str, event_type: str, envelope_json: str) -> str | None:
    outbox_id = str(uuid.uuid4())
    now = datetime.utcnow()
    try:
        with SessionLocal() as db:
            db.add(
                EventOutboxRecord(
                    id=outbox_id,
                    stream=stream,
                    event_type=event_type,
                    envelope_json=json.loads(envelope_json),
                    attempts=0,
                    last_error=None,
                    created_at=now,
                    updated_at=now,
                    published_at=None,
                )
            )
            db.commit()
        return outbox_id
    except Exception:
        logger.exception("event_outbox_insert_failed", extra={"event_type": event_type, "stream": stream})
        return None


def _update_outbox_publish_state(
    outbox_id: str | None, published: bool, error: str | None = None
) -> None:
    if not outbox_id:
        return
    now = datetime.utcnow()
    try:
        with SessionLocal() as db:
            row = db.query(EventOutboxRecord).filter(EventOutboxRecord.id == outbox_id).first()
            if not row:
                return
            row.attempts = (row.attempts or 0) + 1
            row.updated_at = now
            row.last_error = None if published else (error or "redis_publish_failed")
            if published:
                row.published_at = now
            db.commit()
    except Exception:
        logger.exception(
            "event_outbox_update_failed", extra={"outbox_id": outbox_id, "published": published}
        )


def _dispatch_event_outbox_once() -> int:
    if not EVENT_OUTBOX_ENABLED:
        return 0
    dispatched = 0
    try:
        with SessionLocal() as db:
            pending = (
                db.query(EventOutboxRecord)
                .filter(EventOutboxRecord.published_at.is_(None))
                .order_by(EventOutboxRecord.created_at.asc())
                .limit(max(1, EVENT_OUTBOX_BATCH_SIZE))
                .all()
            )
            if not pending:
                return 0
            for row in pending:
                envelope_json = json.dumps(row.envelope_json)
                published, error = _publish_envelope_to_redis(row.stream, envelope_json)
                row.attempts = (row.attempts or 0) + 1
                row.updated_at = datetime.utcnow()
                if published:
                    row.published_at = row.updated_at
                    row.last_error = None
                    dispatched += 1
                else:
                    row.last_error = error or "redis_publish_failed"
            db.commit()
    except Exception:
        logger.exception("event_outbox_dispatch_failed")
    return dispatched


def _start_event_outbox_dispatcher() -> None:
    def _loop() -> None:
        while True:
            _dispatch_event_outbox_once()
            time.sleep(max(0.1, EVENT_OUTBOX_POLL_S))

    thread = threading.Thread(target=_loop, daemon=True, name="event-outbox-dispatcher")
    thread.start()


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
    envelope = models.EventEnvelope(
        type=event_type,
        version="1",
        occurred_at=datetime.utcnow(),
        correlation_id=payload.get("correlation_id", str(uuid.uuid4())),
        job_id=payload.get("job_id") or payload.get("id"),
        task_id=payload.get("task_id"),
        payload=payload,
    )
    stream = _stream_for_event(event_type)
    envelope_json = envelope.model_dump_json()
    if not EVENT_OUTBOX_ENABLED:
        published, error = _publish_envelope_to_redis(stream, envelope_json)
        if not published:
            logger.warning(
                "event_emit_failed",
                extra={"event_type": event_type, "stream": stream, "error": error or "redis_publish_failed"},
            )
        return
    outbox_id = _insert_outbox_event(stream, event_type, envelope_json)
    published, error = _publish_envelope_to_redis(stream, envelope_json)
    if outbox_id is None and not published:
        # Last-chance persistence when the initial outbox insert fails.
        outbox_id = _insert_outbox_event(stream, event_type, envelope_json)
    _update_outbox_publish_state(outbox_id, published, error)
    if not published:
        logger.warning(
            "event_emit_deferred_to_outbox",
            extra={
                "event_type": event_type,
                "stream": stream,
                "outbox_id": outbox_id,
                "error": error or "redis_publish_failed",
            },
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
    profile = metadata.get("goal_intent_profile")
    graph = metadata.get("goal_intent_graph")
    segments = (
        _goal_intent_segments_from_metadata(metadata)
        if isinstance(graph, Mapping)
        else []
    )
    if not isinstance(profile, Mapping) and not segments:
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
        "captured_at": datetime.utcnow().isoformat(),
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
        memory_store.write_memory(
            db,
            models.MemoryWrite(
                name="semantic_memory",
                scope=models.MemoryScope.user,
                user_id=user_id,
                key=memory_key,
                payload=semantic_record,
                metadata={
                    "semantic": True,
                    "semantic_namespace": "intent_workflows",
                    "semantic_subject": semantic_record["subject"],
                    "job_id_source": job.id,
                    "outcome": outcome,
                },
            ),
        )
        metadata["intent_memory_persisted"] = True
        metadata["intent_memory_persisted_at"] = datetime.utcnow().isoformat()
        metadata["intent_memory_key"] = memory_key
        metadata["intent_memory_user_id"] = user_id
        job.metadata_json = metadata
    except Exception:  # noqa: BLE001
        logger.exception("intent_memory_persist_failed", extra={"job_id": job.id})


def _load_schema_from_ref(schema_ref: str) -> dict[str, Any] | None:
    candidate = Path(schema_ref)
    if not candidate.is_absolute():
        candidate = Path(SCHEMA_REGISTRY_PATH) / (
            schema_ref if schema_ref.endswith(".json") else f"{schema_ref}.json"
        )
    if not candidate.exists():
        return None
    try:
        parsed = json.loads(candidate.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    return parsed if isinstance(parsed, dict) else None


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
    if "output.derive" in normalized:
        return "path"
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
            last_capability_id = (last_node.get("capability_id") or "").strip()
            if last_output and last_output in required_inputs:
                score += 32
                reasons.append(f"uses previous output '{last_output}'")
            if capability_id == "document.output.derive" and (
                last_capability_id.startswith("document.spec.")
                or last_capability_id.startswith("document.runbook.")
            ):
                score += 30
                reasons.append("common next step after document spec generation")
            if capability_id in {"document.docx.generate", "document.pdf.generate"} and (
                last_capability_id == "document.output.derive"
            ):
                score += 36
                reasons.append("render after derive output path")

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


def _extract_json_object_from_text(text: str) -> dict[str, Any] | None:
    raw = (text or "").strip()
    if not raw:
        return None
    if raw.startswith("```"):
        lines = [line for line in raw.splitlines() if not line.strip().startswith("```")]
        raw = "\n".join(lines).strip()
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except Exception:  # noqa: BLE001
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        return None
    snippet = raw[start : end + 1]
    try:
        parsed = json.loads(snippet)
    except Exception:  # noqa: BLE001
        return None
    return parsed if isinstance(parsed, dict) else None


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
    graph: dict[str, Any], compaction: Mapping[str, Any]
) -> dict[str, Any]:
    summary = graph.get("summary")
    summary_dict = dict(summary) if isinstance(summary, dict) else {}
    summary_dict["interaction_summary_compaction"] = dict(compaction)
    return {**graph, "summary": summary_dict}


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
) -> list[dict[str, Any]]:
    return capability_search.search_capabilities(
        query=goal,
        capability_entries=allowed_capability_catalog,
        limit=limit,
        intent_hint=intent_hint,
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
    candidate = str(capability_id or "").strip()
    if not candidate:
        return None
    if not allowed_capability_ids:
        return candidate
    if candidate in allowed_capability_ids:
        return candidate
    lookup = {entry.lower(): entry for entry in allowed_capability_ids}
    return lookup.get(candidate.lower())


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
    response = provider.generate(prompt)
    raw_content: Any
    if isinstance(response, Mapping):
        raw_content = response
    elif hasattr(response, "content"):
        raw_content = getattr(response, "content")
    else:
        raw_content = response
    parsed: dict[str, Any] | None = None
    if isinstance(raw_content, dict):
        parsed = raw_content
    elif isinstance(raw_content, bytes):
        parsed = _extract_json_object_from_text(raw_content.decode("utf-8", errors="ignore"))
    elif isinstance(raw_content, str):
        parsed = _extract_json_object_from_text(raw_content)
    else:
        parsed = _extract_json_object_from_text(str(raw_content))
    if not isinstance(parsed, dict):
        raise ValueError("llm_intent_graph_parse_failed")
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
    source = str(graph.get("source") or "unknown")
    model_label = (INTENT_DECOMPOSE_MODEL or LLM_MODEL_NAME or "none").strip() or "none"
    mode_label = INTENT_DECOMPOSE_MODE
    intent_decompose_requests_total.labels(
        mode=mode_label,
        model=model_label,
        source=source,
        result=result,
        has_summaries=str(bool(has_interaction_summaries)).lower(),
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


def _decompose_goal_intent(
    goal: str,
    *,
    db: Session | None = None,
    user_id: str | None = None,
    interaction_summaries: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    fallback_graph = intent_contract.decompose_goal_intent(goal)
    if "source" not in fallback_graph:
        fallback_graph = {**fallback_graph, "source": "heuristic"}
    allowed_capability_catalog = _intent_catalog_capability_entries()
    allowed_capability_ids = {
        str(entry.get("id") or "").strip()
        for entry in allowed_capability_catalog
        if str(entry.get("id") or "").strip()
    }
    if not allowed_capability_ids:
        allowed_capability_ids = _intent_catalog_capability_ids()
    normalized_user_id = _semantic_normalize_text(user_id, max_len=120) or _semantic_default_user_id()
    workflow_hints = _retrieve_intent_workflow_hints(
        db,
        goal=goal,
        user_id=normalized_user_id,
        limit=INTENT_MEMORY_RETRIEVAL_LIMIT,
    )
    semantic_goal_capabilities = _semantic_goal_capability_hints(
        goal=goal,
        allowed_capability_catalog=allowed_capability_catalog,
        limit=max(4, INTENT_CAPABILITY_TOP_K * 2),
    )
    has_interaction_summaries = bool(interaction_summaries)
    result = "heuristic"
    graph = fallback_graph
    if not INTENT_DECOMPOSE_ENABLED:
        result = "disabled"
    elif INTENT_DECOMPOSE_MODE == "heuristic":
        result = "heuristic"
    elif _intent_decompose_provider is None:
        result = "provider_unavailable"
    else:
        try:
            graph = _llm_decompose_goal_intent(
                goal=goal,
                provider=_intent_decompose_provider,
                fallback_graph=fallback_graph,
                allowed_capability_ids=allowed_capability_ids,
                allowed_capability_catalog=allowed_capability_catalog,
                capability_top_k=INTENT_CAPABILITY_TOP_K,
                interaction_summaries=interaction_summaries,
                workflow_hints=workflow_hints,
                semantic_goal_capabilities=semantic_goal_capabilities,
            )
            result = "llm"
        except (LLMProviderError, ValueError) as exc:
            intent_decompose_failures_total.labels(
                mode=INTENT_DECOMPOSE_MODE,
                model=(INTENT_DECOMPOSE_MODEL or LLM_MODEL_NAME or "none").strip() or "none",
                error_type=type(exc).__name__,
            ).inc()
            logger.exception("intent_decompose_llm_failed")
            graph = fallback_graph
            result = "llm_failed_fallback"
        except Exception as exc:  # noqa: BLE001
            intent_decompose_failures_total.labels(
                mode=INTENT_DECOMPOSE_MODE,
                model=(INTENT_DECOMPOSE_MODEL or LLM_MODEL_NAME or "none").strip() or "none",
                error_type=type(exc).__name__,
            ).inc()
            logger.exception("intent_decompose_llm_failed")
            graph = fallback_graph
            result = "llm_failed_fallback"
    graph = _annotate_graph_summary_defaults(
        graph,
        has_interaction_summaries=has_interaction_summaries,
        allowed_capability_ids=allowed_capability_ids,
    )
    summary_raw = graph.get("summary")
    summary = dict(summary_raw) if isinstance(summary_raw, dict) else {}
    summary["memory_hints_used"] = len(workflow_hints)
    summary["memory_retrieval_enabled"] = bool(INTENT_MEMORY_RETRIEVAL_ENABLED)
    summary["semantic_capability_hints_used"] = len(semantic_goal_capabilities)
    graph = {**graph, "summary": summary}
    if interaction_summaries:
        graph = _apply_supported_fact_filter(graph, interaction_summaries)
    _record_intent_decompose_metrics(
        graph=graph,
        result=result,
        has_interaction_summaries=has_interaction_summaries,
    )
    return graph


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
    content = provider.generate(prompt).content
    parsed = _extract_json_object_from_text(content)
    if not isinstance(parsed, dict):
        raise ValueError("llm_recommendation_parse_failed")
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


def _build_plan_from_composer_draft(
    draft: dict[str, Any],
    *,
    goal_text: str = "",
) -> tuple[models.PlanCreate | None, list[dict[str, Any]], list[dict[str, Any]]]:
    diagnostics_errors: list[dict[str, Any]] = []
    diagnostics_warnings: list[dict[str, Any]] = []
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
        if not capability_id:
            diagnostics_errors.append(
                {
                    "code": "draft.capability_missing",
                    "node_id": node_id,
                    "message": "capabilityId is required.",
                }
            )
            continue
        capability_spec = registry.get(capability_id)
        if capability_spec is None:
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

    tasks_payload: list[dict[str, Any]] = []
    for node in canonical_nodes:
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
                memory_payload: dict[str, Any] = {
                    "scope": str(raw_binding.get("scope") or "job"),
                    "name": name,
                }
                key = raw_binding.get("key")
                if isinstance(key, str) and key.strip():
                    memory_payload["key"] = key.strip()
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

        deps = [node_by_id[dep_id]["task_name"] for dep_id in deps_by_node_id.get(node_id, set())]
        task_payload: dict[str, Any] = {
            "name": node["task_name"],
            "description": capability_spec.description,
            "instruction": f"Use capability {capability_id}.",
            "acceptance_criteria": [f"Completed capability {capability_id}"],
            "expected_output_schema_ref": capability_spec.output_schema_ref or "",
            "deps": deps,
            "tool_requests": [capability_id],
            "tool_inputs": {capability_id: tool_input_payload},
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
        target_task_name = node["task_name"]
        for source_node_id in deps_by_node_id.get(node["node_id"], set()):
            source_task_name = node_by_id[source_node_id]["task_name"]
            dag_edges.append([source_task_name, target_task_name])

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


def _handle_plan_created(envelope: dict) -> None:
    payload = envelope.get("payload")
    if not isinstance(payload, dict):
        return
    job_id = envelope.get("job_id") or payload.get("job_id")
    if not job_id:
        return
    plan = _parse_plan_payload(payload)
    if plan is None:
        return
    now = datetime.utcnow()
    with SessionLocal() as db:
        job = db.query(JobRecord).filter(JobRecord.id == job_id).first()
        job_context = job.context_json if job and isinstance(job.context_json, dict) else {}
        job_goal = job.goal if job and isinstance(job.goal, str) else ""
        preflight_errors = _compile_plan_preflight(
            plan,
            job_context,
            goal_text=job_goal,
            goal_intent_graph=job.metadata_json.get("goal_intent_graph")
            if job and isinstance(job.metadata_json, dict)
            else None,
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
            db.commit()
            _enqueue_ready_tasks(job_id, plan_id, envelope.get("correlation_id"))
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
                tool_inputs=task.tool_inputs,
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
            _set_job_status(job, models.JobStatus.planning)
            job.updated_at = now
        db.commit()
    _enqueue_ready_tasks(job_id, plan_record_id, envelope.get("correlation_id"))
    _refresh_job_status(job_id)


def _handle_plan_failed(envelope: dict) -> None:
    payload = envelope.get("payload") or {}
    job_id = envelope.get("job_id") or payload.get("job_id") or payload.get("id")
    if not job_id:
        return
    error_message = payload.get("error")
    now = datetime.utcnow()
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
    now = datetime.utcnow()
    with SessionLocal() as db:
        task = db.query(TaskRecord).filter(TaskRecord.id == task_id).first()
        if not task:
            return
        task.status = models.TaskStatus.completed.value
        task.updated_at = now
        db.commit()
        job_id = task.job_id
        plan_id = task.plan_id
    _enqueue_ready_tasks(job_id, plan_id, envelope.get("correlation_id"))
    _refresh_job_status(job_id)


def _handle_task_failed(envelope: dict) -> None:
    payload = envelope.get("payload") or {}
    task_id = payload.get("task_id") or envelope.get("task_id")
    if not task_id:
        return
    _store_task_result(task_id, payload)
    error = payload.get("error")
    now = datetime.utcnow()
    with SessionLocal() as db:
        task = db.query(TaskRecord).filter(TaskRecord.id == task_id).first()
        if not task:
            return
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
        job = db.query(JobRecord).filter(JobRecord.id == task.job_id).first()
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
    now = datetime.utcnow()
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
        mismatch_payload.setdefault("created_at", datetime.utcnow().isoformat())
        metadata["intent_mismatch_recovery"] = mismatch_payload
        history = metadata.get("intent_mismatch_recovery_history")
        if not isinstance(history, list):
            history = []
        history.append(mismatch_payload)
        metadata["intent_mismatch_recovery_history"] = history[-10:]
    job.metadata_json = metadata
    job.status = models.JobStatus.planning.value
    job.updated_at = datetime.utcnow()
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
    now = datetime.utcnow()
    with SessionLocal() as db:
        task = db.query(TaskRecord).filter(TaskRecord.id == task_id).first()
        if not task:
            return
        task.status = models.TaskStatus.accepted.value
        task.updated_at = now
        db.commit()
        job_id = task.job_id
        plan_id = task.plan_id
    _enqueue_ready_tasks(job_id, plan_id, envelope.get("correlation_id"))
    _refresh_job_status(job_id)


def _handle_task_rework(envelope: dict) -> None:
    payload = envelope.get("payload") or {}
    task_id = payload.get("task_id") or envelope.get("task_id")
    if not task_id:
        return
    now = datetime.utcnow()
    with SessionLocal() as db:
        task = db.query(TaskRecord).filter(TaskRecord.id == task_id).first()
        if not task:
            return
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
    _enqueue_ready_tasks(job_id, plan_id, envelope.get("correlation_id"))


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
    now = datetime.utcnow()
    events: list[tuple[str, dict[str, Any]]] = []
    with SessionLocal() as db:
        task_records = db.query(TaskRecord).filter(TaskRecord.plan_id == plan_id).all()
        if not task_records:
            return
        job_record = db.query(JobRecord).filter(JobRecord.id == job_id).first()
        job_context = (
            job_record.context_json
            if job_record and isinstance(job_record.context_json, dict)
            else {}
        )
        job_goal = job_record.goal if job_record and isinstance(job_record.goal, str) else ""
        task_intent_profiles = _coerce_task_intent_profiles(
            job_record.metadata_json if job_record and isinstance(job_record.metadata_json, dict) else {}
        )
        tasks = _resolve_task_deps(task_records)
        task_map = {task.id: task for task in tasks}
        id_to_name = {record.id: record.name for record in task_records}
        ready_ids = set(orchestrator.ready_tasks(tasks))
        if not ready_ids:
            return
        for record in task_records:
            if record.id in ready_ids and record.status == models.TaskStatus.pending.value:
                if POLICY_GATE_ENABLED:
                    record.status = models.TaskStatus.blocked.value
                    record.updated_at = now
                    context = _build_task_context(record.id, task_map, id_to_name, job_context)
                    payload = _task_payload_from_record(
                        record,
                        correlation_id,
                        context,
                        goal_text=job_goal,
                        intent_profile=task_intent_profiles.get(record.id),
                    )
                    events.append(("task.policy_check", payload))
                    continue
                next_attempt = (record.attempts or 0) + 1
                if _limit_exceeded(next_attempt, record.max_attempts):
                    record.status = models.TaskStatus.failed.value
                    record.updated_at = now
                    events.append(
                        (
                            "task.failed",
                            _task_payload_with_error(
                                record, correlation_id, "max_attempts_exceeded"
                            ),
                        )
                    )
                    continue
                record.attempts = next_attempt
                record.status = models.TaskStatus.ready.value
                record.updated_at = now
                context = _build_task_context(record.id, task_map, id_to_name, job_context)
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
                    events.append(
                        (
                            "task.failed",
                            failed_payload,
                        )
                    )
                    continue
                events.append(("task.ready", payload))
        db.commit()
    for event_type, payload in events:
        _emit_event(event_type, payload)
    _refresh_job_status(job_id)


def _task_payload_from_record(
    record: TaskRecord,
    correlation_id: str | None,
    context: dict[str, Any] | None = None,
    *,
    goal_text: str = "",
    intent_profile: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "task_id": record.id,
        "id": record.id,
        "job_id": record.job_id,
        "plan_id": record.plan_id,
        "name": record.name,
        "description": record.description,
        "instruction": record.instruction,
        "acceptance_criteria": record.acceptance_criteria or [],
        "expected_output_schema_ref": record.expected_output_schema_ref,
        "status": record.status,
        "deps": record.deps or [],
        "attempts": record.attempts or 0,
        "max_attempts": record.max_attempts or 0,
        "rework_count": record.rework_count or 0,
        "max_reworks": record.max_reworks or 0,
        "assigned_to": record.assigned_to,
        "tool_requests": record.tool_requests or [],
        "tool_inputs": record.tool_inputs or {},
        "critic_required": bool(record.critic_required),
        "intent": record.intent,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
        "correlation_id": correlation_id or str(uuid.uuid4()),
    }
    normalized_profile_segment = _normalize_task_intent_profile_segment(
        intent_profile.get("segment") if isinstance(intent_profile, Mapping) else None
    )
    if normalized_profile_segment is not None:
        payload["intent_segment"] = normalized_profile_segment
    profile_intent = (
        intent_contract.normalize_task_intent(intent_profile.get("intent"))
        if isinstance(intent_profile, Mapping)
        else None
    )
    profile_source = (
        str(intent_profile.get("source") or "").strip()
        if isinstance(intent_profile, Mapping)
        else ""
    )
    profile_confidence = None
    if isinstance(intent_profile, Mapping):
        raw_confidence = intent_profile.get("confidence")
        if isinstance(raw_confidence, (int, float)):
            profile_confidence = max(0.0, min(1.0, float(raw_confidence)))
    if profile_intent and not payload.get("intent"):
        payload["intent"] = profile_intent
    if profile_intent and profile_source and profile_confidence is not None:
        payload["intent_source"] = profile_source
        payload["intent_confidence"] = round(profile_confidence, 3)
    else:
        inference_payload = dict(payload)
        if goal_text:
            inference_payload["goal"] = goal_text
        intent_inference = intent_contract.infer_task_intent_for_payload_with_metadata(
            inference_payload
        )
        payload["intent_source"] = intent_inference.source
        payload["intent_confidence"] = round(float(intent_inference.confidence), 3)
        if not payload.get("intent"):
            payload["intent"] = intent_inference.intent
    ctx = context or {}
    if context:
        payload["context"] = context
    resolved_inputs, resolution_errors = payload_resolver.resolve_tool_inputs_with_errors(
        payload["tool_requests"],
        payload["instruction"],
        ctx,
        payload,
        payload.get("tool_inputs", {}),
    )
    validation_errors: dict[str, str] = {}
    if resolved_inputs:
        payload["tool_inputs"] = resolved_inputs
        payload["tool_inputs_resolved"] = True
        validation_errors.update(
            payload_resolver.validate_tool_inputs(resolved_inputs, TOOL_INPUT_SCHEMAS)
        )
    if resolution_errors:
        validation_errors.update(resolution_errors)
    if validation_errors:
        payload["tool_inputs_validation"] = validation_errors
    return payload


def _task_payload_with_error(
    record: TaskRecord, correlation_id: str | None, error: str
) -> dict[str, Any]:
    payload = _task_payload_from_record(record, correlation_id)
    payload["error"] = error
    return payload


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
    now = datetime.utcnow()
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
                    job_record.context_json
                    if job_record and isinstance(job_record.context_json, dict)
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
                _enqueue_ready_tasks(job.id, plan.id, None)
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
    if "tool_requests" in rewrites and isinstance(rewrites["tool_requests"], list):
        task.tool_requests = rewrites["tool_requests"]
    if "tool_inputs" in rewrites and isinstance(rewrites["tool_inputs"], dict):
        task.tool_inputs = rewrites["tool_inputs"]


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


def _build_preflight_dependency_output(task: models.TaskCreate) -> dict[str, Any]:
    output: dict[str, Any] = {
        "document_spec": {},
        "validation_report": {"valid": True, "errors": [], "warnings": []},
        "path": "documents/preflight.pdf",
    }
    for tool_name in task.tool_requests:
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
    goal_intent_graph: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    errors: dict[str, str] = {}
    capabilities: dict[str, capability_registry.CapabilitySpec] = {}
    try:
        if capability_registry.resolve_capability_mode() != "disabled":
            capabilities = capability_registry.load_capability_registry().enabled_capabilities()
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
        goal_intent_graph=goal_intent_graph
    )
    for task_index, task in enumerate(plan.tasks):
        dependency_names = _collect_ancestor_task_names(task, tasks_by_name)
        task_intent = _preflight_task_intent(task, goal_text=goal_text)
        goal_intent_segment = _select_goal_intent_segment_for_task(
            task=task,
            task_index=task_index,
            task_intent=task_intent,
            goal_intent_segments=goal_intent_segments,
            total_tasks=len(plan.tasks),
        )
        for request_id in task.tool_requests or []:
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

        task_payload = {
            "name": task.name,
            "instruction": task.instruction,
            "tool_requests": list(task.tool_requests or []),
            "tool_inputs": task.tool_inputs or {},
        }

        # Seed exact $from paths with typed placeholders to reduce false negatives
        # when dependency outputs are not available yet.
        references = _collect_reference_paths(task.tool_inputs or {})
        reference_error: str | None = None
        for path in references:
            if not path:
                continue
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
                        if referenced_tool not in set(dep_task.tool_requests or []):
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
                        if referenced_tool not in set(dep_task.tool_requests or []):
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

        validation_errors = payload_resolver.validate_tool_inputs(
            resolved_inputs, TOOL_INPUT_SCHEMAS
        )
        if validation_errors:
            first_tool, message = next(iter(validation_errors.items()))
            errors[task.name] = f"{first_tool}:{message}"
            continue

        for request_id in task.tool_requests or []:
            resolved_payload_raw = resolved_inputs.get(request_id, {})
            resolved_payload = (
                resolved_payload_raw
                if isinstance(resolved_payload_raw, Mapping)
                else {}
            )
            capability_spec = capabilities.get(request_id)
            segment_contract_error = intent_contract.validate_intent_segment_contract(
                segment=goal_intent_segment,
                task_intent=task_intent,
                tool_name=request_id,
                payload=resolved_payload,
                capability_id=request_id if capability_spec is not None else None,
                capability_risk_tier=capability_spec.risk_tier
                if capability_spec is not None
                else None,
            )
            if segment_contract_error:
                errors[task.name] = (
                    f"intent_segment_invalid:{request_id}:{task.name}:{segment_contract_error}"
                )
                break

    return errors


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
            for request_id in (task.tool_requests or [])
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
    goal_intent_graph: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    if not isinstance(goal_intent_graph, Mapping):
        return []
    raw_segments = goal_intent_graph.get("segments")
    if not isinstance(raw_segments, list):
        return []
    segments: list[dict[str, Any]] = []
    for raw_segment in raw_segments:
        normalized = _normalize_task_intent_profile_segment(raw_segment)
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
        for name in (task.tool_requests or [])
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
            for segment in matching:
                if intent_contract.normalize_task_intent(segment.get("intent")) == task_intent:
                    return segment
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


def _store_task_output(task_id: str, outputs: dict[str, Any]) -> None:
    try:
        redis_client.set(f"{TASK_OUTPUT_KEY_PREFIX}{task_id}", json.dumps(outputs))
    except Exception:
        return


def _load_task_output(task_id: str) -> dict[str, Any]:
    try:
        raw = redis_client.get(f"{TASK_OUTPUT_KEY_PREFIX}{task_id}")
        if not raw:
            return {}
        return json.loads(raw)
    except Exception:
        return {}


def _store_task_result(task_id: str, result: dict[str, Any]) -> None:
    try:
        redis_client.set(f"{TASK_RESULT_KEY_PREFIX}{task_id}", json.dumps(result))
    except Exception:
        return


def _load_task_result(task_id: str) -> dict[str, Any]:
    try:
        raw = redis_client.get(f"{TASK_RESULT_KEY_PREFIX}{task_id}")
        if not raw:
            return {}
        return json.loads(raw)
    except Exception:
        return {}


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
    outputs_by_id = {dep_id: _load_task_output(dep_id) for dep_id in visited}
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
    profile = metadata.get("goal_intent_profile")
    if not isinstance(profile, Mapping):
        return
    intent = intent_contract.normalize_task_intent(profile.get("intent")) or "generate"
    risk_level = _normalize_risk_level(profile.get("risk_level"))
    try:
        confidence = float(profile.get("confidence"))
    except (TypeError, ValueError):
        return
    threshold_raw = profile.get("threshold")
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
        "recorded_at": datetime.utcnow().isoformat(),
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
        now = datetime.utcnow()
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
        job.updated_at = now
        db.commit()


def _set_job_status(job: JobRecord, status: models.JobStatus) -> None:
    current = models.JobStatus(job.status)
    if state_machine.validate_job_transition(current, status):
        job.status = status.value


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
            "hint": "Set a non-empty 'path' input. Recommended: add document.output.derive and bind its path output to the renderer path input.",
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
        occurred_at = datetime.utcnow().isoformat()
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


@app.post("/jobs", response_model=models.Job)
def create_job(
    job: models.JobCreate,
    require_clarification: bool = Query(False),
    db: Session = Depends(get_db),
) -> models.Job:
    job_id = str(uuid.uuid4())
    now = datetime.utcnow()
    goal_assessment = _assess_goal_intent(job.goal)
    gate_on_create = bool(require_clarification or INTENT_CLARIFICATION_ON_CREATE)
    if gate_on_create and bool(goal_assessment.get("requires_blocking_clarification")):
        raise HTTPException(
            status_code=422,
            detail={
                "error": "intent_clarification_required",
                "goal_intent_profile": goal_assessment,
            },
        )
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
    if LLM_PROVIDER_NAME:
        metadata["llm_provider"] = LLM_PROVIDER_NAME
    if LLM_MODEL_NAME:
        metadata["llm_model"] = LLM_MODEL_NAME
    semantic_user_id = _semantic_user_id_from_context(
        context_json_for_job if isinstance(context_json_for_job, Mapping) else None
    )
    metadata["semantic_user_id"] = semantic_user_id
    metadata["goal_intent_profile"] = goal_assessment
    if INTENT_DECOMPOSE_ENABLED:
        goal_intent_graph = _decompose_goal_intent(
            job.goal,
            db=db,
            user_id=semantic_user_id,
            interaction_summaries=interaction_summaries_compact or interaction_summaries_raw,
        )
        if interaction_summaries_raw:
            goal_intent_graph = _attach_interaction_compaction_to_graph(
                goal_intent_graph,
                interaction_compaction,
            )
        metadata["goal_intent_graph"] = goal_intent_graph
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
    _emit_event("job.created", _job_from_record(record).model_dump())
    return _job_from_record(record)


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
    job_error: str | None = None
    if isinstance(metadata.get("plan_error"), str) and str(metadata.get("plan_error")).strip():
        job_error = str(metadata.get("plan_error")).strip()
    elif metadata.get("plan_preflight_errors") is not None:
        job_error = f"plan_preflight_failed: {metadata.get('plan_preflight_errors')}"
    return models.JobDetails(
        job_id=job_id,
        job_status=models.JobStatus(job.status),
        job_error=job_error,
        plan=_plan_from_record(plan_record) if plan_record else None,
        tasks=[_task_from_record(task, profiles.get(task.id)) for task in tasks],
        task_results={task.id: _load_task_result(task.id) for task in tasks},
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
    job_context = job.context_json if isinstance(job.context_json, dict) else {}
    task_intent_profiles = _coerce_task_intent_profiles(
        job.metadata_json if isinstance(job.metadata_json, dict) else {}
    )

    timeline = _read_task_events_for_job(job_id, limit)
    timeline_by_task: dict[str, list[dict[str, Any]]] = {}
    for entry in timeline:
        task_id = entry.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            continue
        timeline_by_task.setdefault(task_id, []).append(entry)

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
                "latest_result": task_result,
                "error": _classify_task_error(latest_error),
            }
        )

    return {
        "job_id": job_id,
        "job_status": job.status,
        "plan_id": plan_record.id if plan_record else None,
        "generated_at": datetime.utcnow().isoformat(),
        "timeline_events_scanned": len(timeline),
        "tasks": tasks_payload,
    }


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
    job.updated_at = datetime.utcnow()
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
    job.updated_at = datetime.utcnow()
    db.commit()
    plan = db.query(PlanRecord).filter(PlanRecord.job_id == job_id).first()
    if plan:
        _enqueue_ready_tasks(job_id, plan.id, None)
    return _job_from_record(job)


@app.post("/jobs/{job_id}/retry", response_model=models.Job)
def retry_job(job_id: str, db: Session = Depends(get_db)) -> models.Job:
    job = db.query(JobRecord).filter(JobRecord.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if models.JobStatus(job.status) not in {models.JobStatus.failed, models.JobStatus.canceled}:
        raise HTTPException(status_code=400, detail="Job is not retryable")
    now = datetime.utcnow()
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
        _enqueue_ready_tasks(job_id, plan.id, None)
    return _job_from_record(job)


@app.post("/jobs/{job_id}/retry_failed", response_model=models.Job)
def retry_failed_tasks(job_id: str, db: Session = Depends(get_db)) -> models.Job:
    job = db.query(JobRecord).filter(JobRecord.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    now = datetime.utcnow()
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
        _enqueue_ready_tasks(job_id, plan.id, None)
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
    now = datetime.utcnow()
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
        _enqueue_ready_tasks(job_id, plan.id, None)
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
    job.updated_at = datetime.utcnow()
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
        "captured_at": datetime.utcnow().isoformat(),
    }
    write_request = models.MemoryWrite(
        name="semantic_memory",
        scope=models.MemoryScope.user,
        key=key,
        user_id=user_id,
        payload=semantic_record,
        metadata=metadata,
        ttl_seconds=ttl_seconds,
    )
    try:
        entry = memory_store.write_memory(db, write_request)
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"entry": entry.model_dump(), "semantic_record": semantic_record}


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
    if not job_context and isinstance(job_id, str) and job_id.strip():
        job = db.query(JobRecord).filter(JobRecord.id == job_id).first()
        if job and isinstance(job.context_json, dict):
            job_context = job.context_json
    goal_text = str(payload.get("goal") or "").strip()
    if not goal_text and isinstance(job_id, str) and job_id.strip():
        job = db.query(JobRecord).filter(JobRecord.id == job_id).first()
        if job and isinstance(job.goal, str):
            goal_text = job.goal

    provided_graph = payload.get("goal_intent_graph") or payload.get("intent_graph")
    if not isinstance(provided_graph, Mapping) and isinstance(job_id, str) and job_id.strip():
        job = db.query(JobRecord).filter(JobRecord.id == job_id).first()
        if job and isinstance(job.metadata_json, dict):
            provided_graph = job.metadata_json.get("goal_intent_graph")
    preflight_errors = _compile_plan_preflight(
        plan,
        job_context,
        goal_text=goal_text,
        goal_intent_graph=provided_graph if isinstance(provided_graph, Mapping) else None,
    )
    diagnostics = _preflight_error_diagnostics(preflight_errors)
    return {
        "valid": not bool(preflight_errors),
        "errors": preflight_errors,
        "diagnostics": diagnostics,
        "intent_inference": _task_intent_summary(plan.tasks, goal_text=goal_text),
    }


@app.post("/intent/clarify")
def clarify_intent(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="invalid_intent_payload")
    goal = str(payload.get("goal") or "").strip()
    if not goal:
        raise HTTPException(status_code=400, detail="goal_required")
    assessment = _assess_goal_intent(goal)
    return {
        "goal": goal,
        "assessment": assessment,
    }


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
    graph = _decompose_goal_intent(
        goal,
        db=db,
        user_id=semantic_user_id,
        interaction_summaries=compacted_summaries or interaction_summaries,
    )
    if interaction_summaries:
        graph = _attach_interaction_compaction_to_graph(graph, compaction)
    return {
        "goal": goal,
        "intent_graph": graph,
    }


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
    if not job_context and isinstance(job_id, str) and job_id.strip():
        job = db.query(JobRecord).filter(JobRecord.id == job_id).first()
        if job and isinstance(job.context_json, dict):
            job_context = job.context_json
    goal_text = str(payload.get("goal") or "").strip()
    if not goal_text:
        goal_text = str(raw_draft.get("goal") or raw_draft.get("job_goal") or "").strip()
    if not goal_text and isinstance(job_id, str) and job_id.strip():
        job = db.query(JobRecord).filter(JobRecord.id == job_id).first()
        if job and isinstance(job.goal, str):
            goal_text = job.goal
    provided_graph = payload.get("goal_intent_graph") or payload.get("intent_graph")
    if not isinstance(provided_graph, Mapping):
        provided_graph = raw_draft.get("goal_intent_graph") or raw_draft.get("intent_graph")
    if not isinstance(provided_graph, Mapping) and isinstance(job_id, str) and job_id.strip():
        job = db.query(JobRecord).filter(JobRecord.id == job_id).first()
        if job and isinstance(job.metadata_json, dict):
            provided_graph = job.metadata_json.get("goal_intent_graph")

    plan, diagnostics_errors, diagnostics_warnings = _build_plan_from_composer_draft(
        raw_draft,
        goal_text=goal_text,
    )
    preflight_errors: dict[str, str] = {}
    if plan is not None:
        preflight_errors = _compile_plan_preflight(
            plan,
            job_context,
            goal_text=goal_text,
            goal_intent_graph=provided_graph if isinstance(provided_graph, Mapping) else None,
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


@app.post("/plans", response_model=models.Plan)
def create_plan(plan: models.PlanCreate, job_id: str, db: Session = Depends(get_db)) -> models.Plan:
    job = db.query(JobRecord).filter(JobRecord.id == job_id).first()
    job_context = job.context_json if job and isinstance(job.context_json, dict) else {}
    goal_text = job.goal if job and isinstance(job.goal, str) else ""
    preflight_errors = _compile_plan_preflight(
        plan,
        job_context,
        goal_text=goal_text,
        goal_intent_graph=job.metadata_json.get("goal_intent_graph")
        if job and isinstance(job.metadata_json, dict)
        else None,
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
    now = datetime.utcnow()
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
            tool_inputs=task.tool_inputs,
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
    payload = _plan_created_payload(plan, job_id)
    payload["correlation_id"] = str(uuid.uuid4())
    _emit_event("plan.created", payload)
    return _plan_from_record(record)
