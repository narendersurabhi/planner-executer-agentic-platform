from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable, Mapping
import re
from typing import Any

from . import models


TASK_INTENT_VALUES = tuple(intent.value for intent in models.ToolIntent)

_KEYWORD_MAP: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("validate", ("validate", "verify", "check", "lint", "schema")),
    ("render", ("render", "rendering", "docx", "pdf")),
    ("transform", ("transform", "reshape", "wrap", "convert", "derive", "summarize", "repair")),
    ("generate", ("generate", "create", "draft", "write", "compose", "produce", "build")),
    ("io", ("read", "fetch", "list", "search", "load", "save", "download", "upload")),
)

_ALLOWED_TOOL_INTENTS_BY_TASK_INTENT: dict[str, set[models.ToolIntent]] = {
    # generate tasks can compose and finish artifacts with render/io tools.
    "generate": {models.ToolIntent.generate, models.ToolIntent.render, models.ToolIntent.io},
    # transform tasks can reshape data and optionally use io utilities.
    "transform": {models.ToolIntent.transform, models.ToolIntent.io},
    # validate tasks can validate and normalize inputs before asserting quality.
    "validate": {
        models.ToolIntent.validate,
        models.ToolIntent.transform,
        models.ToolIntent.io,
    },
    # render tasks can format outputs and do lightweight data shaping.
    "render": {models.ToolIntent.render, models.ToolIntent.transform, models.ToolIntent.io},
    # io tasks should stay side-effect/data movement focused.
    "io": {models.ToolIntent.io},
}


@dataclass(frozen=True)
class TaskIntentInference:
    intent: str
    source: str
    confidence: float


@dataclass(frozen=True)
class GoalIntentSegment:
    id: str
    intent: str
    objective: str
    source: str
    confidence: float
    depends_on: tuple[str, ...]
    required_inputs: tuple[str, ...]
    suggested_capabilities: tuple[str, ...]
    slots: dict[str, Any]


_INTENT_SOURCE_EXPLICIT = "explicit"
_INTENT_SOURCE_TASK_TEXT = "task_text"
_INTENT_SOURCE_GOAL_TEXT = "goal_text"
_INTENT_SOURCE_DEFAULT = "default"
_INTENT_SOURCE_GOAL_FALLBACK = "goal_fallback"

_SUGGESTED_CAPABILITIES_BY_INTENT: dict[str, tuple[str, ...]] = {
    "io": ("memory.read", "github.repo.list", "filesystem.artifacts.search_text"),
    "generate": ("llm.text.generate", "document.spec.generate"),
    "transform": ("utility.json.transform", "document.spec.improve"),
    "validate": ("document.spec.validate",),
    "render": ("document.docx.generate", "document.pdf.generate"),
}

_REQUIRED_INPUTS_BY_INTENT: dict[str, tuple[str, ...]] = {
    "io": ("source_or_query",),
    "generate": ("instruction",),
    "transform": ("input_data",),
    "validate": ("input_data", "acceptance_criteria_or_schema"),
    "render": ("input_data", "path_or_format"),
}

_SLOT_OUTPUT_FORMAT_MAP: dict[str, str] = {
    "pdf": "pdf",
    ".pdf": "pdf",
    "docx": "docx",
    ".docx": "docx",
    "md": "md",
    ".md": "md",
    "markdown": "md",
    "txt": "txt",
    ".txt": "txt",
    "text": "txt",
    "plain_text": "txt",
    "html": "html",
    ".html": "html",
    "json": "json",
    ".json": "json",
    "csv": "csv",
    ".csv": "csv",
    "xlsx": "xlsx",
    ".xlsx": "xlsx",
}

_SLOT_RISK_LEVEL_MAP: dict[str, str] = {
    "read_only": "read_only",
    "readonly": "read_only",
    "low": "read_only",
    "bounded_write": "bounded_write",
    "write": "bounded_write",
    "medium": "bounded_write",
    "high_risk": "high_risk_write",
    "high_risk_write": "high_risk_write",
    "high": "high_risk_write",
}

_SLOT_RISK_RANK: dict[str, int] = {
    "read_only": 1,
    "bounded_write": 2,
    "high_risk": 3,
    "high_risk_write": 3,
}


def _coerce_string_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, Iterable) or isinstance(value, (str, bytes)):
        return ()
    items: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        normalized = item.strip()
        if normalized and normalized not in items:
            items.append(normalized)
    return tuple(items)


def _normalize_slot_token(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    if not normalized:
        return None
    normalized = normalized.replace("-", "_").replace(" ", "_")
    return re.sub(r"[^a-z0-9_]", "", normalized) or None


def _normalize_output_format(value: Any) -> str | None:
    token = _normalize_slot_token(value)
    if not token:
        return None
    return _SLOT_OUTPUT_FORMAT_MAP.get(token)


def _normalize_risk_level(value: Any) -> str | None:
    token = _normalize_slot_token(value)
    if not token:
        return None
    return _SLOT_RISK_LEVEL_MAP.get(token)


def _extract_required_input_key(value: str) -> str:
    token = value.strip().lower()
    if "=" in token:
        token = token.split("=", 1)[0].strip()
    token = token.replace("-", "_").replace(" ", "_")
    token = re.sub(r"[^a-z0-9_]", "", token)
    alias_map = {
        "path_or_format": "path",
        "source_or_query": "query",
    }
    return alias_map.get(token, token)


def _infer_segment_output_format(
    objective: str,
    suggested_capabilities: Iterable[str],
    required_inputs: Iterable[str],
) -> str | None:
    lowered = objective.lower()
    capability_blob = " ".join(str(item).lower() for item in suggested_capabilities)
    required_blob = " ".join(str(item).lower() for item in required_inputs)
    search_blob = f"{lowered} {capability_blob} {required_blob}"
    if "pdf" in search_blob:
        return "pdf"
    if "docx" in search_blob:
        return "docx"
    if "markdown" in search_blob or ".md" in search_blob or " md " in f" {search_blob} ":
        return "md"
    if "html" in search_blob:
        return "html"
    if "json" in search_blob:
        return "json"
    if "csv" in search_blob:
        return "csv"
    if "xlsx" in search_blob or "excel" in search_blob:
        return "xlsx"
    if "text" in search_blob or "txt" in search_blob:
        return "txt"
    return None


def _infer_segment_entity(objective: str, artifact_type: str) -> str:
    lowered = objective.lower()
    if "runbook" in lowered:
        return "runbook"
    if "repo" in lowered or "github" in lowered:
        return "repository"
    if "document spec" in lowered or "document_spec" in lowered:
        return "document_spec"
    return artifact_type


def _infer_segment_artifact_type(intent: str, objective: str, output_format: str | None) -> str:
    lowered = objective.lower()
    if "document spec" in lowered or "document_spec" in lowered or " spec" in f" {lowered} ":
        return "document_spec"
    if "validation" in lowered or "schema" in lowered or intent == "validate":
        return "validation_report"
    if "repo" in lowered or "github" in lowered:
        return "repository_data"
    if output_format in {"pdf", "docx", "md", "txt", "html"}:
        return "document"
    if output_format in {"json", "csv", "xlsx"}:
        return "structured_data"
    if intent == "render":
        return "rendered_artifact"
    if intent == "transform":
        return "intermediate_output"
    return "content"


def _infer_segment_risk_level(intent: str, objective: str, suggested_capabilities: Iterable[str]) -> str:
    lowered = objective.lower()
    capability_blob = " ".join(str(item).lower() for item in suggested_capabilities)
    search_blob = f"{lowered} {capability_blob}"
    high_risk_tokens = (
        "delete",
        "destroy",
        "infra",
        "production",
        "deploy",
        "payment",
        "billing",
        "shutdown",
    )
    if any(token in search_blob for token in high_risk_tokens):
        return "high_risk_write"
    if intent in {"render", "io"} or any(
        token in search_blob for token in ("write", "create", "update", "push", "publish")
    ):
        return "bounded_write"
    return "read_only"


def _slot_value(raw_slots: Any, key: str) -> Any:
    if isinstance(raw_slots, Mapping):
        return raw_slots.get(key)
    return None


def _normalize_must_have_inputs(
    value: Any,
    *,
    fallback_inputs: Iterable[str],
    output_format: str | None,  # noqa: ARG001
) -> tuple[str, ...]:
    has_explicit_value = isinstance(value, Iterable) and not isinstance(value, (str, bytes))
    explicit = _coerce_string_tuple(value)
    normalized: list[str] = []
    if has_explicit_value:
        candidates = explicit
    else:
        candidates = tuple(
            item
            for item in (
                _extract_required_input_key(raw)
                for raw in fallback_inputs
            )
            if item and "_or_" not in item and len(item) <= 48
        )
    for candidate in candidates:
        normalized_key = _extract_required_input_key(candidate)
        if not normalized_key or normalized_key in normalized:
            continue
        normalized.append(normalized_key)
    return tuple(normalized)


def normalize_intent_segment_slots(
    *,
    raw_slots: Any,
    intent: str,
    objective: str,
    required_inputs: Iterable[str],
    suggested_capabilities: Iterable[str],
    fallback_slots: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    fallback_slots = fallback_slots if isinstance(fallback_slots, Mapping) else {}
    output_format = (
        _normalize_output_format(_slot_value(raw_slots, "output_format"))
        or _normalize_output_format(fallback_slots.get("output_format"))
        or _infer_segment_output_format(objective, suggested_capabilities, required_inputs)
    )
    artifact_type = (
        _normalize_slot_token(_slot_value(raw_slots, "artifact_type"))
        or _normalize_slot_token(fallback_slots.get("artifact_type"))
        or _infer_segment_artifact_type(intent, objective, output_format)
    )
    entity = (
        _normalize_slot_token(_slot_value(raw_slots, "entity"))
        or _normalize_slot_token(fallback_slots.get("entity"))
        or _infer_segment_entity(objective, artifact_type)
    )
    risk_level = (
        _normalize_risk_level(_slot_value(raw_slots, "risk_level"))
        or _normalize_risk_level(fallback_slots.get("risk_level"))
        or _infer_segment_risk_level(intent, objective, suggested_capabilities)
    )
    must_have_inputs = _normalize_must_have_inputs(
        _slot_value(raw_slots, "must_have_inputs")
        if _slot_value(raw_slots, "must_have_inputs") is not None
        else fallback_slots.get("must_have_inputs"),
        fallback_inputs=required_inputs,
        output_format=output_format,
    )
    return {
        "entity": entity or "artifact",
        "artifact_type": artifact_type or "content",
        "output_format": output_format,
        "risk_level": risk_level or "read_only",
        "must_have_inputs": list(must_have_inputs),
    }


def _tool_output_format_hint(tool_name: str) -> str | None:
    lowered = tool_name.lower()
    if "pdf" in lowered:
        return "pdf"
    if "docx" in lowered:
        return "docx"
    if "markdown" in lowered or "_md" in lowered:
        return "md"
    if "json" in lowered:
        return "json"
    if "csv" in lowered:
        return "csv"
    return None


def _coerce_payload_mapping(payload: Any) -> Mapping[str, Any]:
    if isinstance(payload, Mapping):
        return payload
    return {}


def _value_present(value: Any, *, key: str) -> bool:
    if key == "document_spec":
        # Planner-time validation may normalize dependency references to {} placeholders.
        # Treat any mapping as present so intent checks don't reject valid dependency wiring.
        return isinstance(value, Mapping)
    if isinstance(value, str):
        return bool(value.strip())
    return value is not None


def _job_context_value(payload_map: Mapping[str, Any], key: str) -> Any:
    job = payload_map.get("job")
    if not isinstance(job, Mapping):
        return None
    if key in job:
        return job.get(key)
    context_json = job.get("context_json")
    if isinstance(context_json, Mapping) and key in context_json:
        return context_json.get(key)
    return None


def _payload_has_required_input(
    payload_map: Mapping[str, Any], key: str, *, tool_name: str | None = None
) -> bool:
    allow_job_context = tool_name not in {
        "llm_generate_document_spec",
        "document.spec.generate",
        "llm_generate_document_spec_from_markdown",
        "document.spec.generate_from_markdown",
    }
    if key == "repo_full_name":
        owner = (
            payload_map.get("owner")
            or payload_map.get("repo_owner")
            or (allow_job_context and _job_context_value(payload_map, "owner"))
            or (allow_job_context and _job_context_value(payload_map, "repo_owner"))
        )
        repo = (
            payload_map.get("repo")
            or payload_map.get("repo_name")
            or (allow_job_context and _job_context_value(payload_map, "repo"))
            or (allow_job_context and _job_context_value(payload_map, "repo_name"))
        )
        if _value_present(owner, key="owner") and _value_present(repo, key="repo"):
            return True
    if key == "length":
        for candidate in ("length", "target_pages", "page_count", "max_words", "word_count"):
            if candidate in payload_map and _value_present(payload_map.get(candidate), key=candidate):
                return True
            value = _job_context_value(payload_map, candidate) if allow_job_context else None
            if _value_present(value, key=candidate):
                return True
        # For document generation, a concrete instruction is often sufficient to infer length.
        if _value_present(payload_map.get("instruction"), key="instruction"):
            return True
    if key in payload_map and _value_present(payload_map.get(key), key=key):
        return True
    aliases: dict[str, tuple[str, ...]] = {
        "input_data": (
            "document_spec",
            "content",
            "text",
            "markdown_text",
            "json",
            "data",
        ),
        "path": ("output_path",),
        "output_path": ("path",),
        "date": ("today",),
        "filename": ("file_name", "output_filename", "path", "output_path"),
        "github_query": ("query",),
        "target_repo": ("query", "github_query", "repo", "repo_name"),
        "repo_full_name": ("query", "github_query", "target_repo"),
        "title": (
            "document_title",
            "topic",
            "subject",
            "target_role_name",
            "role_name",
            "job_title",
            "role",
        ),
        "job_posting_text": ("job_description",),
        "company_logo_image": ("company_logo", "company_logo_url", "logo_url"),
    }
    for candidate in (key,) + aliases.get(key, ()):
        if candidate in payload_map and _value_present(payload_map.get(candidate), key=candidate):
            return True
        value = _job_context_value(payload_map, candidate) if allow_job_context else None
        if _value_present(value, key=key):
            return True
    return False


def _risk_rank(value: Any) -> int:
    normalized = _normalize_risk_level(value)
    if normalized:
        return _SLOT_RISK_RANK.get(normalized, 0)
    token = _normalize_slot_token(value)
    if not token:
        return 0
    return _SLOT_RISK_RANK.get(token, 0)


def _segment_task_intent_compatible(segment_intent: str, task_intent: str) -> bool:
    if segment_intent == task_intent:
        return True
    # Some intent decomposition steps classify path/filename derivation as io.
    # Planner tasks using derive/validation utilities can be transform/validate.
    if segment_intent == "io" and task_intent in {"transform", "validate"}:
        return True
    return False


def validate_intent_segment_contract(
    *,
    segment: Mapping[str, Any] | None,
    task_intent: str,
    tool_name: str,
    payload: Mapping[str, Any] | None = None,
    capability_id: str | None = None,
    capability_risk_tier: str | None = None,
) -> str | None:
    if not isinstance(segment, Mapping):
        return None
    segment_intent = normalize_task_intent(segment.get("intent"))
    if segment_intent and not _segment_task_intent_compatible(segment_intent, task_intent):
        return (
            f"segment_intent_mismatch:{tool_name}:"
            f"segment={segment_intent}:task={task_intent}"
        )
    required_inputs = _coerce_string_tuple(segment.get("required_inputs"))
    suggested_capabilities = _coerce_string_tuple(segment.get("suggested_capabilities"))
    slots = normalize_intent_segment_slots(
        raw_slots=segment.get("slots"),
        fallback_slots=None,
        intent=segment_intent or task_intent,
        objective=str(segment.get("objective") or ""),
        required_inputs=required_inputs,
        suggested_capabilities=suggested_capabilities,
    )
    payload_map = _coerce_payload_mapping(payload)
    missing_inputs: list[str] = []
    for key in slots.get("must_have_inputs", []):
        if not isinstance(key, str) or not key:
            continue
        if key == "output_format" and _tool_output_format_hint(capability_id or tool_name):
            continue
        if (
            key in {"path", "output_path"}
            and _capability_auto_derives_output_path(capability_id or tool_name)
        ):
            continue
        if key == "document_spec" and _capability_derives_output_path(capability_id or tool_name):
            # Output-path derivation capabilities don't need the full document spec payload.
            continue
        if "_or_" in key:
            continue
        if _payload_has_required_input(payload_map, key, tool_name=tool_name):
            continue
        missing_inputs.append(key)
    if missing_inputs:
        return f"must_have_inputs_missing:{','.join(sorted(set(missing_inputs)))}"

    output_format = slots.get("output_format")
    if isinstance(output_format, str) and output_format:
        resolved_format = _normalize_output_format(output_format)
        if resolved_format:
            raw_path = payload_map.get("path")
            if not isinstance(raw_path, str):
                raw_path = payload_map.get("output_path")
            if isinstance(raw_path, str) and raw_path.strip():
                path = raw_path.strip().lower()
                if "." in path:
                    extension = path.rsplit(".", 1)[-1]
                    normalized_ext = _normalize_output_format(extension)
                    if normalized_ext and normalized_ext != resolved_format:
                        return (
                            f"output_format_mismatch:{tool_name}:"
                            f"expected={resolved_format}:got={normalized_ext}"
                        )
            hinted = _tool_output_format_hint(capability_id or tool_name)
            if hinted and hinted != resolved_format:
                return (
                    f"output_format_mismatch:{tool_name}:"
                    f"expected={resolved_format}:hinted={hinted}"
                )

    slot_rank = _risk_rank(slots.get("risk_level"))
    capability_rank = _risk_rank(capability_risk_tier)
    if slot_rank > 0 and capability_rank > slot_rank:
        expected = _normalize_risk_level(slots.get("risk_level")) or str(slots.get("risk_level") or "")
        actual = _normalize_risk_level(capability_risk_tier) or str(capability_risk_tier or "")
        return f"risk_level_mismatch:{tool_name}:expected<={expected}:actual={actual}"
    return None


def _capability_auto_derives_output_path(capability_id_or_tool_name: str) -> bool:
    normalized = str(capability_id_or_tool_name or "").strip().lower()
    return normalized in {
        "document.docx.generate",
        "document.pdf.generate",
        "docx_generate_from_spec",
        "pdf_generate_from_spec",
    }


def _capability_derives_output_path(capability_id_or_tool_name: str) -> bool:
    normalized = str(capability_id_or_tool_name or "").strip().lower()
    return normalized in {
        "document.output.derive",
        "derive_output_path",
        "derive_output_filename",
    }


def _infer_intent_from_text_with_source(
    text: str,
    *,
    source: str,
) -> TaskIntentInference:
    normalized = text.lower()
    for intent, keywords in _KEYWORD_MAP:
        if any(keyword in normalized for keyword in keywords):
            confidence = 0.82 if source == _INTENT_SOURCE_TASK_TEXT else 0.72
            return TaskIntentInference(intent=intent, source=source, confidence=confidence)
    return TaskIntentInference(
        intent=models.ToolIntent.generate.value,
        source=_INTENT_SOURCE_DEFAULT,
        confidence=0.4,
    )


def normalize_task_intent(value: Any) -> str | None:
    if isinstance(value, models.ToolIntent):
        return value.value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in TASK_INTENT_VALUES:
            return normalized
    return None


def infer_task_intent_from_text(
    *,
    description: str = "",
    instruction: str = "",
    acceptance_criteria: Iterable[str] | None = None,
) -> str:
    parts: list[str] = []
    if description:
        parts.append(description)
    if instruction:
        parts.append(instruction)
    if acceptance_criteria:
        parts.extend([item for item in acceptance_criteria if isinstance(item, str)])
    text = " ".join(parts).lower()
    for intent, keywords in _KEYWORD_MAP:
        if any(keyword in text for keyword in keywords):
            return intent
    return "generate"


def infer_task_intent_from_text_with_metadata(
    *,
    description: str = "",
    instruction: str = "",
    acceptance_criteria: Iterable[str] | None = None,
) -> TaskIntentInference:
    parts: list[str] = []
    if description:
        parts.append(description)
    if instruction:
        parts.append(instruction)
    if acceptance_criteria:
        parts.extend([item for item in acceptance_criteria if isinstance(item, str)])
    text = " ".join(parts)
    return _infer_intent_from_text_with_source(text, source=_INTENT_SOURCE_TASK_TEXT)


def infer_task_intent_from_goal(goal_text: str) -> str:
    goal = goal_text.strip() if isinstance(goal_text, str) else ""
    if not goal:
        return models.ToolIntent.generate.value
    return infer_task_intent_from_text(description=goal)


def infer_task_intent_from_goal_with_metadata(goal_text: str) -> TaskIntentInference:
    goal = goal_text.strip() if isinstance(goal_text, str) else ""
    if not goal:
        return TaskIntentInference(
            intent=models.ToolIntent.generate.value,
            source=_INTENT_SOURCE_DEFAULT,
            confidence=0.4,
        )
    return _infer_intent_from_text_with_source(goal, source=_INTENT_SOURCE_GOAL_TEXT)


def _split_goal_clauses(goal_text: str) -> list[str]:
    goal = re.sub(r"\s+", " ", goal_text.strip())
    if not goal:
        return []
    normalized = goal.replace("->", " then ").replace("=>", " then ")
    parts = re.split(
        r"(?:\b(?:and then|then|after that|next|finally)\b|[;\n])",
        normalized,
        flags=re.IGNORECASE,
    )
    clauses: list[str] = []
    for part in parts:
        chunk = part.strip()
        if not chunk:
            continue
        sub_parts = re.split(r"\band\b", chunk, flags=re.IGNORECASE)
        if len(sub_parts) == 1:
            candidate = chunk.strip(" ,.-")
            if candidate:
                clauses.append(candidate)
            continue
        for sub in sub_parts:
            candidate = sub.strip(" ,.-")
            if candidate:
                clauses.append(candidate)
    if not clauses:
        return [goal]
    merged: list[str] = []
    for clause in clauses:
        if len(clause) < 4 and merged:
            merged[-1] = f"{merged[-1]} {clause}".strip()
            continue
        merged.append(clause)
    return merged[:8]


def _required_inputs_for_clause(intent: str, clause: str) -> tuple[str, ...]:
    base = list(_REQUIRED_INPUTS_BY_INTENT.get(intent, ()))
    lowered = clause.lower()
    if intent == "render":
        if "pdf" in lowered:
            base.append("output_format=pdf")
        elif "docx" in lowered:
            base.append("output_format=docx")
    if intent == "io" and any(token in lowered for token in ("github", "repo", "repository")):
        base.append("query")
    deduped: list[str] = []
    for item in base:
        if item not in deduped:
            deduped.append(item)
    return tuple(deduped)


def _suggested_capabilities_for_clause(intent: str, clause: str) -> tuple[str, ...]:
    suggestions = list(_SUGGESTED_CAPABILITIES_BY_INTENT.get(intent, ()))
    lowered = clause.lower()
    if intent == "render":
        if "pdf" in lowered:
            suggestions = ["document.pdf.generate"]
        elif "docx" in lowered:
            suggestions = ["document.docx.generate"]
    if intent == "io" and any(token in lowered for token in ("github", "repo", "repository")):
        suggestions.insert(0, "github.repo.list")
    deduped: list[str] = []
    for item in suggestions:
        if item not in deduped:
            deduped.append(item)
    return tuple(deduped)


def decompose_goal_intent(goal_text: str) -> dict[str, Any]:
    goal = goal_text.strip() if isinstance(goal_text, str) else ""
    if not goal:
        return {
            "goal": "",
            "segments": [],
            "summary": {"segment_count": 0, "intent_order": [], "schema_version": "intent_v2"},
            "overall_confidence": 0.0,
        }
    clauses = _split_goal_clauses(goal)
    overall = infer_task_intent_from_goal_with_metadata(goal)
    segments: list[GoalIntentSegment] = []
    for index, clause in enumerate(clauses):
        inferred = infer_task_intent_from_goal_with_metadata(clause)
        if inferred.source == _INTENT_SOURCE_DEFAULT and overall.source != _INTENT_SOURCE_DEFAULT:
            inferred = TaskIntentInference(
                intent=overall.intent,
                source=_INTENT_SOURCE_GOAL_FALLBACK,
                confidence=max(0.45, min(0.95, overall.confidence * 0.85)),
            )
        segment_id = f"s{index + 1}"
        depends_on = (f"s{index}",) if index > 0 else ()
        required_inputs = _required_inputs_for_clause(inferred.intent, clause)
        suggested_capabilities = _suggested_capabilities_for_clause(inferred.intent, clause)
        slots = normalize_intent_segment_slots(
            raw_slots={},
            intent=inferred.intent,
            objective=clause,
            required_inputs=required_inputs,
            suggested_capabilities=suggested_capabilities,
            fallback_slots=None,
        )
        segments.append(
            GoalIntentSegment(
                id=segment_id,
                intent=inferred.intent,
                objective=clause,
                source=inferred.source,
                confidence=round(max(0.0, min(1.0, float(inferred.confidence))), 3),
                depends_on=depends_on,
                required_inputs=required_inputs,
                suggested_capabilities=suggested_capabilities,
                slots=slots,
            )
        )
    average_confidence = sum(segment.confidence for segment in segments) / max(1, len(segments))
    return {
        "goal": goal,
        "segments": [
            {
                "id": segment.id,
                "intent": segment.intent,
                "objective": segment.objective,
                "source": segment.source,
                "confidence": segment.confidence,
                "depends_on": list(segment.depends_on),
                "required_inputs": list(segment.required_inputs),
                "suggested_capabilities": list(segment.suggested_capabilities),
                "slots": dict(segment.slots),
            }
            for segment in segments
        ],
        "summary": {
            "segment_count": len(segments),
            "intent_order": [segment.intent for segment in segments],
            "schema_version": "intent_v2",
        },
        "overall_confidence": round(average_confidence, 3),
    }


def _extract_goal_text(payload: Mapping[str, Any]) -> str:
    goal = payload.get("goal")
    if isinstance(goal, str) and goal.strip():
        return goal
    job = payload.get("job")
    if isinstance(job, Mapping):
        nested_goal = job.get("goal")
        if isinstance(nested_goal, str) and nested_goal.strip():
            return nested_goal
    return ""


def infer_task_intent_for_payload(payload: Mapping[str, Any]) -> str:
    return infer_task_intent_for_payload_with_metadata(payload).intent


def infer_task_intent_for_payload_with_metadata(payload: Mapping[str, Any]) -> TaskIntentInference:
    explicit = normalize_task_intent(payload.get("intent")) or normalize_task_intent(
        payload.get("task_intent")
    )
    if explicit:
        return TaskIntentInference(
            intent=explicit,
            source=_INTENT_SOURCE_EXPLICIT,
            confidence=1.0,
        )
    description = payload.get("description")
    instruction = payload.get("instruction")
    criteria = payload.get("acceptance_criteria")
    inferred = infer_task_intent_from_text_with_metadata(
        description=description if isinstance(description, str) else "",
        instruction=instruction if isinstance(instruction, str) else "",
        acceptance_criteria=criteria if isinstance(criteria, list) else [],
    )
    if inferred.intent != models.ToolIntent.generate.value:
        return inferred
    goal_inferred = infer_task_intent_from_goal_with_metadata(_extract_goal_text(payload))
    if goal_inferred.source != _INTENT_SOURCE_DEFAULT:
        return goal_inferred
    return inferred


def infer_task_intent_for_task(
    *,
    explicit_intent: Any,
    description: str,
    instruction: str,
    acceptance_criteria: Iterable[str] | None,
    goal_text: str = "",
) -> str:
    return infer_task_intent_for_task_with_metadata(
        explicit_intent=explicit_intent,
        description=description,
        instruction=instruction,
        acceptance_criteria=acceptance_criteria,
        goal_text=goal_text,
    ).intent


def infer_task_intent_for_task_with_metadata(
    *,
    explicit_intent: Any,
    description: str,
    instruction: str,
    acceptance_criteria: Iterable[str] | None,
    goal_text: str = "",
) -> TaskIntentInference:
    normalized = normalize_task_intent(explicit_intent)
    if normalized:
        return TaskIntentInference(
            intent=normalized,
            source=_INTENT_SOURCE_EXPLICIT,
            confidence=1.0,
        )
    inferred = infer_task_intent_from_text_with_metadata(
        description=description,
        instruction=instruction,
        acceptance_criteria=acceptance_criteria,
    )
    if inferred.intent != models.ToolIntent.generate.value:
        return inferred
    goal_inferred = infer_task_intent_from_goal_with_metadata(goal_text)
    if goal_inferred.source != _INTENT_SOURCE_DEFAULT:
        return goal_inferred
    return inferred


def validate_tool_intent_compatibility(
    task_intent: str,
    tool_intent: models.ToolIntent,
    tool_name: str,
) -> str | None:
    allowed = _ALLOWED_TOOL_INTENTS_BY_TASK_INTENT.get(task_intent)
    if not allowed:
        return f"invalid_task_intent:{task_intent}"
    if tool_intent in allowed:
        return None
    return f"tool_intent_mismatch:{tool_name}:{tool_intent.value}:{task_intent}"
