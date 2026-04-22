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
    (
        "transform",
        (
            "transform",
            "reshape",
            "wrap",
            "convert",
            "derive",
            "summarize",
            "repair",
            "improve",
            "iterate",
            "iteratively",
        ),
    ),
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


@dataclass(frozen=True)
class ActiveExecutionTarget:
    segment_id: str | None
    capability_id: str | None
    capability_ids: tuple[str, ...]
    required_fields: tuple[str, ...]
    unresolved_fields: tuple[str, ...]


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
    "render": ("document.docx.render", "document.pdf.render"),
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
    "word": "docx",
    "word_document": "docx",
    "microsoft_word": "docx",
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

_CLARIFICATION_REQUIRED_INPUT_KEYS: set[str] = {
    "instruction",
    "topic",
    "main_topic",
    "audience",
    "tone",
    "query",
    "path",
    "output_path",
    "filename",
    "output_format",
    "target_system",
    "safety_constraints",
    "intent_action",
}

_NON_COLLECTIBLE_REQUIRED_INPUT_KEYS: set[str] = {
    "document_spec",
    "document_spec_data",
    "input_data",
    "job",
    "tool_inputs",
    "memory",
    "task_outputs",
}

_GENERIC_CAPABILITY_SYSTEMS: set[str] = {
    "document",
    "llm",
    "utility",
    "openapi",
    "codegen",
}

_INTENT_ACTION_PHRASES: dict[str, str] = {
    "generate": "generate new content",
    "transform": "transform or improve existing content",
    "validate": "validate or check something",
    "render": "render a final artifact",
    "io": "fetch, inspect, or list data",
}

_GENERIC_INSTRUCTION_TOKENS: set[str] = {
    "a",
    "an",
    "the",
    "this",
    "that",
    "these",
    "those",
    "please",
    "create",
    "generate",
    "write",
    "draft",
    "compose",
    "produce",
    "build",
    "make",
    "render",
    "save",
    "convert",
    "document",
    "doc",
    "docx",
    "pdf",
    "file",
    "artifact",
    "content",
    "output",
    "page",
    "pages",
}

_CLAUSE_SPLIT_ACTION_HINTS: tuple[str, ...] = (
    "read",
    "fetch",
    "list",
    "search",
    "load",
    "save",
    "download",
    "upload",
    "generate",
    "create",
    "draft",
    "write",
    "compose",
    "produce",
    "build",
    "transform",
    "reshape",
    "wrap",
    "convert",
    "derive",
    "summarize",
    "repair",
    "improve",
    "validate",
    "verify",
    "check",
    "lint",
    "render",
    "push",
    "publish",
    "open",
)


def _contains_any(text: str, tokens: Iterable[str]) -> bool:
    lowered = text.lower()
    for raw_token in tokens:
        token = str(raw_token or "").strip().lower()
        if not token:
            continue
        pattern = r"(?<![a-z0-9])" + re.escape(token).replace(r"\ ", r"\s+") + r"(?![a-z0-9])"
        if re.search(pattern, lowered):
            return True
    return False


def _looks_like_workspace_delete_request(text: str) -> bool:
    lowered = str(text or "").lower()
    return (
        "workspace" in lowered
        and _contains_any(lowered, ("delete", "remove", "destroy", "drop", "wipe", "purge"))
        and _contains_any(lowered, ("manifest", "file", "files", "directory", "folder", "workspace"))
    )


def _prepend_unique(items: list[str], *candidates: str) -> list[str]:
    ordered = list(items)
    for candidate in reversed(candidates):
        if candidate and candidate not in ordered:
            ordered.insert(0, candidate)
    return ordered


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
        "artifact_name": "path",
        "content_instructions": "instruction",
        "document_content_instructions": "instruction",
        "document_instruction": "instruction",
        "document_path": "path",
        "document_spec_data": "document_spec",
        "final_document_path": "path",
        "path_or_format": "path",
        "source_or_query": "query",
        "output_path": "path",
        "filename": "path",
        "file_name": "path",
        "output_filename": "path",
    }
    return alias_map.get(token, token)


def normalize_required_input_key(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return _extract_required_input_key(value)


def select_active_execution_target(
    *,
    graph: Mapping[str, Any] | None,
    candidate_capabilities: Mapping[str, Any] | None = None,
    known_slot_values: Mapping[str, Any] | None = None,
    pending_fields: Iterable[Any] = (),
    preferred_segment_id: str | None = None,
    preferred_capability_id: str | None = None,
) -> ActiveExecutionTarget | None:
    graph_map = graph if isinstance(graph, Mapping) else {}
    raw_segments = graph_map.get("segments")
    if not isinstance(raw_segments, list):
        return None

    candidate_map = candidate_capabilities if isinstance(candidate_capabilities, Mapping) else {}
    normalized_pending_fields = {
        normalized
        for value in pending_fields
        if (normalized := normalize_required_input_key(value))
    }
    known_fields = {
        normalized
        for raw_key, raw_value in (known_slot_values or {}).items()
        if raw_value is not None
        and (not isinstance(raw_value, str) or raw_value.strip())
        and (normalized := normalize_required_input_key(raw_key))
    }
    preferred_segment = str(preferred_segment_id or "").strip()
    preferred_capability = str(preferred_capability_id or "").strip()

    best_target: ActiveExecutionTarget | None = None
    best_score: tuple[int, int, int, int, int, int] | None = None
    fallback_target: ActiveExecutionTarget | None = None

    for index, raw_segment in enumerate(raw_segments):
        if not isinstance(raw_segment, Mapping):
            continue
        segment_id = str(raw_segment.get("id") or "").strip()
        if not segment_id:
            continue
        required_fields = _segment_required_fields(raw_segment)
        unresolved_required_fields = tuple(
            field for field in required_fields if field and field not in known_fields
        )
        overlapping_unresolved_fields = tuple(
            field
            for field in unresolved_required_fields
            if not normalized_pending_fields or field in normalized_pending_fields
        )
        unresolved_fields = overlapping_unresolved_fields or unresolved_required_fields
        capability_ids = _segment_capability_ids(
            raw_segment,
            candidate_capabilities=candidate_map.get(segment_id),
        )
        capability_match = int(bool(preferred_capability and preferred_capability in capability_ids))
        segment_match = int(bool(preferred_segment and segment_id == preferred_segment))
        overlap = len(normalized_pending_fields.intersection(unresolved_required_fields))
        has_relevant_unresolved = int(bool(unresolved_required_fields))

        selected_capability = (
            preferred_capability
            if capability_match
            else (capability_ids[0] if capability_ids else None)
        )
        target = ActiveExecutionTarget(
            segment_id=segment_id,
            capability_id=selected_capability,
            capability_ids=capability_ids,
            required_fields=required_fields,
            unresolved_fields=unresolved_fields,
        )
        if segment_match and fallback_target is None:
            fallback_target = target

        score = (
            overlap,
            has_relevant_unresolved,
            capability_match,
            segment_match,
            len(unresolved_required_fields),
            -index,
        )
        if best_score is None or score > best_score:
            best_score = score
            best_target = target

    if best_target is None:
        return fallback_target
    if best_score is not None and best_score[1] == 0 and fallback_target is not None:
        return fallback_target
    return best_target


def _segment_required_fields(segment: Mapping[str, Any]) -> tuple[str, ...]:
    normalized: list[str] = []
    for raw_value in segment.get("required_inputs", []) or []:
        key = normalize_required_input_key(raw_value)
        if key and key not in normalized:
            normalized.append(key)
    slots = segment.get("slots")
    if isinstance(slots, Mapping):
        for raw_value in slots.get("must_have_inputs", []) or []:
            key = normalize_required_input_key(raw_value)
            if key and key not in normalized:
                normalized.append(key)
    return tuple(normalized)


def _segment_capability_ids(
    segment: Mapping[str, Any],
    *,
    candidate_capabilities: Any = None,
) -> tuple[str, ...]:
    normalized: list[str] = []
    for source in (candidate_capabilities, segment.get("suggested_capabilities")):
        for raw_value in _coerce_string_tuple(source):
            if raw_value and raw_value not in normalized:
                normalized.append(raw_value)
    return tuple(normalized)


def capability_intent_hints(capability_ids: Iterable[str]) -> list[str]:
    hints: list[str] = []
    seen: set[str] = set()
    for raw_capability_id in capability_ids:
        capability_id = str(raw_capability_id or "").strip().lower()
        if not capability_id:
            continue
        hint = _capability_intent_hint(capability_id)
        if hint and hint not in seen:
            seen.add(hint)
            hints.append(hint)
    return hints


def detect_intent_disagreement(
    *,
    goal: str,
    profile: Mapping[str, Any] | None,
    heuristic_profile: Mapping[str, Any] | None = None,
    graph: Mapping[str, Any] | None = None,
    context_capability_candidates: Iterable[str] = (),
    missing_inputs: Iterable[str] = (),
) -> dict[str, Any]:
    del goal
    profile_map = profile if isinstance(profile, Mapping) else {}
    heuristic_map = heuristic_profile if isinstance(heuristic_profile, Mapping) else {}
    graph_map = graph if isinstance(graph, Mapping) else {}

    profile_intent = normalize_task_intent(profile_map.get("intent"))
    heuristic_intent = normalize_task_intent(heuristic_map.get("intent"))
    profile_low_confidence = bool(profile_map.get("low_confidence"))
    graph_source = str(graph_map.get("source") or "").strip().lower()
    try:
        graph_confidence = float(graph_map.get("overall_confidence") or 0.0)
    except (TypeError, ValueError):
        graph_confidence = 0.0
    trusted_graph = bool(
        profile_low_confidence
        or graph_source not in {"", "heuristic"}
        or graph_confidence >= 0.75
    )
    graph_intents: list[str] = []
    graph_capabilities: list[str] = []
    segments = graph_map.get("segments") if isinstance(graph_map.get("segments"), list) else []
    if trusted_graph:
        for raw_segment in segments:
            if not isinstance(raw_segment, Mapping):
                continue
            segment_intent = normalize_task_intent(raw_segment.get("intent"))
            if segment_intent and segment_intent not in graph_intents:
                graph_intents.append(segment_intent)
            for capability_id in _coerce_string_tuple(raw_segment.get("suggested_capabilities")):
                if capability_id not in graph_capabilities:
                    graph_capabilities.append(capability_id)
    capability_intents = capability_intent_hints(
        list(context_capability_candidates) + graph_capabilities
    )
    normalized_missing_inputs = [
        normalize_required_input_key(value)
        for value in missing_inputs
        if normalize_required_input_key(value)
    ]

    reason_code: str | None = None
    evidence_intent: str | None = None
    if len(graph_intents) == 1 and profile_intent and graph_intents[0] != profile_intent:
        reason_code = "graph_intent_conflict"
        evidence_intent = graph_intents[0]
    elif (
        len(capability_intents) == 1
        and profile_intent
        and capability_intents[0] != profile_intent
        and (profile_low_confidence or capability_intents[0] == heuristic_intent)
    ):
        reason_code = "capability_intent_conflict"
        evidence_intent = capability_intents[0]
    elif heuristic_intent and profile_intent and heuristic_intent != profile_intent and profile_low_confidence:
        reason_code = "heuristic_intent_conflict"
        evidence_intent = heuristic_intent
    elif profile_intent == "io" and any(
        key in {"path", "output_format"} for key in normalized_missing_inputs
    ):
        reason_code = "required_inputs_conflict"
        evidence_intent = (
            graph_intents[0]
            if len(graph_intents) == 1
            else capability_intents[0]
            if len(capability_intents) == 1
            else "render"
        )
    elif profile_intent in {"generate", "render"} and "target_system" in normalized_missing_inputs:
        reason_code = "required_inputs_conflict"
        evidence_intent = (
            graph_intents[0]
            if len(graph_intents) == 1
            else capability_intents[0]
            if len(capability_intents) == 1
            else "io"
        )

    if not reason_code:
        return {}

    ordered_intents: list[str] = []
    for candidate_intent in (
        evidence_intent,
        heuristic_intent,
        profile_intent,
    ):
        if candidate_intent and candidate_intent not in ordered_intents:
            ordered_intents.append(candidate_intent)
    question = _intent_disagreement_question(ordered_intents)
    return {
        "detected": True,
        "reason_code": reason_code,
        "profile_intent": profile_intent,
        "heuristic_intent": heuristic_intent,
        "graph_intents": graph_intents,
        "capability_intents": capability_intents,
        "question": question,
        "blocking_slots": ["intent_action"],
        "missing_inputs": ["intent_action"],
        "clarification_mode": "intent_disagreement",
    }


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


def _objective_mentions_document_spec(objective: str) -> bool:
    lowered = objective.lower()
    if "document spec" in lowered or "document_spec" in lowered:
        return True
    dense = re.sub(r"[^a-z0-9]+", "", lowered)
    return "documentspec" in dense


def _objective_is_clarification_step(objective: str) -> bool:
    lowered = objective.lower().strip()
    if not lowered:
        return False
    if "clarify" in lowered or "clarification" in lowered:
        return True
    dense = re.sub(r"[^a-z0-9]+", "", lowered)
    return "clarify" in dense or "clarification" in dense


def _infer_segment_entity(objective: str, artifact_type: str) -> str:
    lowered = objective.lower()
    if "runbook" in lowered:
        return "runbook"
    if "repo" in lowered or "github" in lowered:
        return "repository"
    if _objective_mentions_document_spec(objective):
        return "document_spec"
    return artifact_type


def _infer_segment_artifact_type(intent: str, objective: str, output_format: str | None) -> str:
    lowered = objective.lower()
    if _objective_mentions_document_spec(objective) or " spec" in f" {lowered} ":
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
    objective_is_document_spec_generation = (
        intent == "generate" and _objective_mentions_document_spec(objective)
    )
    objective_is_clarification = _objective_is_clarification_step(objective)
    output_format = (
        _normalize_output_format(_slot_value(raw_slots, "output_format"))
        or _normalize_output_format(fallback_slots.get("output_format"))
        or _infer_segment_output_format(objective, suggested_capabilities, required_inputs)
    )
    if intent == "render" and output_format == "json":
        render_hint = _first_output_format_hint(suggested_capabilities)
        if render_hint in {"docx", "pdf", "md", "txt", "html"}:
            output_format = render_hint
    artifact_type = (
        _normalize_slot_token(_slot_value(raw_slots, "artifact_type"))
        or _normalize_slot_token(fallback_slots.get("artifact_type"))
        or _infer_segment_artifact_type(intent, objective, output_format)
    )
    if objective_is_document_spec_generation:
        artifact_type = "document_spec"
    entity = (
        _normalize_slot_token(_slot_value(raw_slots, "entity"))
        or _normalize_slot_token(fallback_slots.get("entity"))
        or _infer_segment_entity(objective, artifact_type)
    )
    if objective_is_document_spec_generation:
        entity = "document_spec"
    risk_level = (
        _normalize_risk_level(_slot_value(raw_slots, "risk_level"))
        or _normalize_risk_level(fallback_slots.get("risk_level"))
        or _infer_segment_risk_level(intent, objective, suggested_capabilities)
    )
    if intent == "render" and _risk_rank(risk_level) < _risk_rank("bounded_write"):
        risk_level = "bounded_write"
    must_have_inputs = _normalize_must_have_inputs(
        _slot_value(raw_slots, "must_have_inputs")
        if _slot_value(raw_slots, "must_have_inputs") is not None
        else fallback_slots.get("must_have_inputs"),
        fallback_inputs=required_inputs,
        output_format=output_format,
    )
    if intent == "generate" and artifact_type == "document_spec":
        must_have_inputs = tuple(
            key
            for key in must_have_inputs
            if key
            not in {
                "goal",
                "filename",
                "path",
                "output_path",
                "output_format",
                "format",
                "compactness",
                "length_limit",
            }
        )
    if objective_is_clarification:
        must_have_inputs = tuple(
            key
            for key in must_have_inputs
            if key
            not in {
                "title",
                "topic",
                "audience",
                "tone",
                "output_format",
                "compactness",
                "length_limit",
            }
        )
    return {
        "entity": entity or "artifact",
        "artifact_type": artifact_type or "content",
        "output_format": output_format,
        "risk_level": risk_level or "read_only",
        "must_have_inputs": list(must_have_inputs),
    }


def required_input_question(required_input: str, goal: str) -> str:
    key = normalize_required_input_key(required_input)
    if key == "instruction":
        return "What should this specifically cover?"
    if key in {"topic", "main_topic", "title"}:
        return "What is the main topic or title?"
    if key == "audience":
        return "Who is the target audience?"
    if key == "tone":
        return "What tone should it use?"
    if key in {"path", "output_path", "filename"}:
        return "What output path or filename should be used?"
    if key == "output_format":
        return "What output format do you need (for example PDF, DOCX, JSON, or Markdown)?"
    if key == "target_system":
        return "Which target system should this use (for example GitHub, Jira, Slack, filesystem)?"
    if key == "safety_constraints":
        return "What safety constraints must be enforced (for example read-only, no deletes, dry-run)?"
    if key == "intent_action":
        return "What should the system do first (generate, transform, validate, render, or io)?"
    return f"Provide clarification for input '{key}' for goal: '{goal[:120]}'."


def _goal_mentions_explicit_path(goal: str, objective: str) -> bool:
    blob = f"{goal} {objective}".strip()
    if not blob:
        return False
    if re.search(r"(?:^|[\s'\"`])(?:/|\.?/)[^\s]+(?:\.[A-Za-z0-9]{2,8})", blob):
        return True
    return bool(re.search(r"\b[\w./-]+\.(?:pdf|docx|md|txt|html|json|csv|xlsx)\b", blob))


def _goal_implies_concrete_instruction(goal: str) -> bool:
    raw = str(goal or "").strip()
    if not raw:
        return False
    primary = re.split(r"\n+\s*user clarification:\s*", raw, maxsplit=1, flags=re.IGNORECASE)[0]
    primary = re.sub(r"\b[\w./-]+\.(?:pdf|docx|md|txt|html|json|csv|xlsx)\b", " ", primary, flags=re.IGNORECASE)
    primary = re.sub(r"(?:^|[\s'\"`])(?:/|\.?/)[^\s]+", " ", primary)
    tokens = re.findall(r"[a-z0-9]+", primary.lower())
    informative = [
        token
        for token in tokens
        if token not in _GENERIC_INSTRUCTION_TOKENS and not token.isdigit()
    ]
    return bool(informative)


def _segment_candidate_format_hints(capability_ids: Iterable[str]) -> set[str]:
    hints: set[str] = set()
    for capability_id in capability_ids:
        hinted = _tool_output_format_hint(str(capability_id or ""))
        if hinted:
            hints.add(hinted)
    return hints


def _segment_candidate_system_hints(capability_ids: Iterable[str]) -> set[str]:
    systems: set[str] = set()
    for capability_id in capability_ids:
        normalized = str(capability_id or "").strip().lower()
        if "." not in normalized:
            continue
        system = normalized.split(".", 1)[0].strip()
        if system and system not in _GENERIC_CAPABILITY_SYSTEMS:
            systems.add(system)
    return systems


def derive_segment_missing_inputs(
    *,
    goal: str,
    segment: Mapping[str, Any],
    slot_values: Mapping[str, Any] | None = None,
    candidate_required_inputs: Iterable[str] = (),
    low_confidence: bool = False,
) -> list[str]:
    if not isinstance(segment, Mapping):
        return []
    slot_values = slot_values if isinstance(slot_values, Mapping) else {}
    segment_slots = dict(segment.get("slots")) if isinstance(segment.get("slots"), Mapping) else {}
    objective = str(segment.get("objective") or "").strip()
    candidate_capabilities = _coerce_string_tuple(segment.get("suggested_capabilities"))
    candidate_formats = _segment_candidate_format_hints(candidate_capabilities)
    candidate_systems = _segment_candidate_system_hints(candidate_capabilities)
    raw_required_inputs = list(_coerce_string_tuple(segment.get("required_inputs")))
    raw_must_have_inputs = list(_coerce_string_tuple(segment_slots.get("must_have_inputs")))
    combined_required_inputs: list[str] = []
    direct_clarification_candidates: set[str] = set(_CLARIFICATION_REQUIRED_INPUT_KEYS)
    required_input_sources: list[str] = raw_required_inputs + raw_must_have_inputs
    if not low_confidence and not raw_required_inputs and not raw_must_have_inputs:
        required_input_sources.extend(
            [str(item) for item in candidate_required_inputs if isinstance(item, str)]
        )
    for raw_value in required_input_sources:
        normalized = normalize_required_input_key(raw_value)
        if normalized and normalized not in combined_required_inputs:
            combined_required_inputs.append(normalized)
        if normalized and normalized not in _NON_COLLECTIBLE_REQUIRED_INPUT_KEYS:
            direct_clarification_candidates.add(normalized)

    def _mapping_has_value(mapping: Mapping[str, Any], *keys: str) -> bool:
        for key in keys:
            if key in mapping and _value_present(mapping.get(key), key=key):
                return True
        return False

    path_present = (
        _mapping_has_value(segment_slots, "path", "output_path", "filename", "file_name", "output_filename")
        or _mapping_has_value(slot_values, "path", "output_path", "filename", "file_name", "output_filename")
        or _goal_mentions_explicit_path(goal, objective)
    )
    output_format_present = (
        _mapping_has_value(segment_slots, "output_format", "format")
        or _mapping_has_value(slot_values, "output_format", "format")
        or path_present
        or len(candidate_formats) == 1
    )
    target_system_present = (
        _mapping_has_value(segment_slots, "target_system")
        or _mapping_has_value(slot_values, "target_system")
        or len(candidate_systems) == 1
    )
    safety_present = _mapping_has_value(segment_slots, "safety_constraints") or _mapping_has_value(
        slot_values, "safety_constraints"
    )
    intent_action_present = bool(
        normalize_task_intent(slot_values.get("intent_action"))
        or normalize_task_intent(segment.get("intent"))
    )
    instruction_present = (
        _mapping_has_value(
            segment_slots,
            "instruction",
            "document_instruction",
            "document_content_instructions",
            "content_instructions",
        )
        or _mapping_has_value(
            slot_values,
            "instruction",
            "document_instruction",
            "document_content_instructions",
            "content_instructions",
        )
        or _goal_implies_concrete_instruction(goal)
    )

    missing_inputs: list[str] = []
    for key in combined_required_inputs:
        if key not in direct_clarification_candidates:
            continue
        if key == "instruction":
            if not instruction_present:
                missing_inputs.append("instruction")
            continue
        if key in {"path", "output_path", "filename"}:
            if not path_present:
                missing_inputs.append("path")
            continue
        if key == "output_format":
            if not output_format_present:
                missing_inputs.append("output_format")
            continue
        if key == "target_system":
            if not target_system_present:
                missing_inputs.append("target_system")
            continue
        if key == "safety_constraints":
            if not safety_present:
                missing_inputs.append("safety_constraints")
            continue
        if key == "intent_action":
            if not intent_action_present:
                missing_inputs.append("intent_action")
            continue
        if not _mapping_has_value(segment_slots, key) and not _mapping_has_value(slot_values, key):
            missing_inputs.append(key)

    segment_intent = normalize_task_intent(segment.get("intent")) or ""
    if (
        segment_intent == "render"
        and not output_format_present
        and len(candidate_formats) > 1
        and "output_format" not in missing_inputs
    ):
        missing_inputs.append("output_format")
    if (
        low_confidence
        and not intent_action_present
        and "intent_action" not in missing_inputs
    ):
        missing_inputs.append("intent_action")
    return list(dict.fromkeys(missing_inputs))


def derive_envelope_clarification(
    *,
    goal: str,
    profile: Mapping[str, Any] | None,
    heuristic_profile: Mapping[str, Any] | None = None,
    graph: Mapping[str, Any] | None,
    context_capability_candidates: Iterable[str] = (),
    candidate_required_inputs_by_segment: Mapping[str, Iterable[str]] | None = None,
) -> dict[str, Any]:
    profile_map = profile if isinstance(profile, Mapping) else {}
    graph_map = graph if isinstance(graph, Mapping) else {}
    slot_values = (
        dict(profile_map.get("slot_values"))
        if isinstance(profile_map.get("slot_values"), Mapping)
        else {}
    )
    segments = graph_map.get("segments") if isinstance(graph_map.get("segments"), list) else []
    low_confidence = bool(profile_map.get("low_confidence"))
    profile_intent = normalize_task_intent(profile_map.get("intent")) or ""
    prioritized_segment_ids: set[str] = set()
    if profile_intent and len(segments) > 1:
        prioritized_segment_ids = {
            str(segment.get("id") or "").strip()
            for segment in segments
            if isinstance(segment, Mapping)
            and normalize_task_intent(segment.get("intent")) == profile_intent
            and str(segment.get("id") or "").strip()
        }
    candidate_required_inputs_by_segment = (
        candidate_required_inputs_by_segment
        if isinstance(candidate_required_inputs_by_segment, Mapping)
        else {}
    )

    missing_inputs: list[str] = []
    for raw_segment in segments:
        if not isinstance(raw_segment, Mapping):
            continue
        segment_id = str(raw_segment.get("id") or "").strip()
        if prioritized_segment_ids and segment_id not in prioritized_segment_ids:
            continue
        segment_missing = derive_segment_missing_inputs(
            goal=goal,
            segment=raw_segment,
            slot_values=slot_values,
            candidate_required_inputs=candidate_required_inputs_by_segment.get(segment_id, ()),
            low_confidence=low_confidence,
        )
        for key in segment_missing:
            if key not in missing_inputs:
                missing_inputs.append(key)

    if not missing_inputs:
        for raw_key in _coerce_string_tuple(profile_map.get("missing_slots")):
            normalized = normalize_required_input_key(raw_key)
            if normalized and normalized not in missing_inputs:
                missing_inputs.append(normalized)

    disagreement = detect_intent_disagreement(
        goal=goal,
        profile=profile_map,
        heuristic_profile=heuristic_profile,
        graph=graph_map,
        context_capability_candidates=context_capability_candidates,
        missing_inputs=missing_inputs,
    )
    if disagreement:
        disagreement_missing = [
            normalize_required_input_key(value)
            for value in disagreement.get("missing_inputs", [])
            if normalize_required_input_key(value)
        ]
        missing_inputs = _prepend_unique(missing_inputs, *disagreement_missing)

    questions = [required_input_question(key, goal) for key in missing_inputs]
    clarification_mode = "capability_required_inputs"
    if disagreement:
        disagreement_question = str(disagreement.get("question") or "").strip()
        remaining_questions = [
            question
            for question, missing_input in zip(questions, missing_inputs)
            if normalize_required_input_key(missing_input) != "intent_action"
        ]
        questions = [disagreement_question] if disagreement_question else []
        questions.extend(remaining_questions)
        clarification_mode = str(disagreement.get("clarification_mode") or "").strip() or clarification_mode
    return {
        "needs_clarification": bool(missing_inputs),
        "requires_blocking_clarification": bool(missing_inputs),
        "missing_inputs": missing_inputs,
        "questions": questions,
        "blocking_slots": list(missing_inputs),
        "slot_values": dict(slot_values),
        "clarification_mode": clarification_mode,
        "disagreement": disagreement,
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


def _first_output_format_hint(values: Iterable[Any]) -> str | None:
    for value in values:
        hinted = _tool_output_format_hint(str(value or ""))
        if hinted:
            return hinted
    return None


def _is_document_spec_generation_identifier(value: Any) -> bool:
    return str(value or "").strip().lower() in {
        "llm_generate_document_spec",
        "document.spec.generate",
        "llm_iterative_improve_document_spec",
        "document.spec.generate_iterative",
        "llm_generate_document_spec_from_markdown",
        "document.spec.generate_from_markdown",
    }


def _ignored_document_spec_generation_required_inputs(
    tool_name: str | None,
    capability_id: str | None,
) -> set[str]:
    if not (
        _is_document_spec_generation_identifier(tool_name)
        or _is_document_spec_generation_identifier(capability_id)
    ):
        return set()
    return {
        "goal",
        "workspace_path",
        "filename",
        "path",
        "output_path",
        "output_format",
        "format",
        "compactness",
        "length_limit",
    }


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
    normalized_tool_name = str(tool_name or "").strip().lower()
    allow_job_context = not _is_document_spec_generation_identifier(normalized_tool_name)
    if key == "query" and normalized_tool_name in {
        "filesystem.workspace.list",
        "filesystem.artifacts.list",
        "list_files",
    }:
        return True
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
        "artifact_name": ("path", "output_path", "filename", "file_name", "output_filename"),
        "content_instructions": ("instruction",),
        "document_goal": ("instruction", "goal", "topic", "main_topic"),
        "document_content_instructions": ("instruction",),
        "document_instruction": ("instruction",),
        "document_path": ("path", "output_path"),
        "document_spec_data": ("document_spec",),
        "final_document_path": ("path", "output_path"),
        "input_data": (
            "document_spec",
            "content",
            "text",
            "markdown_text",
            "json",
            "data",
        ),
        "query": ("path", "source"),
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
            "main_topic",
            "subject",
            "target_role_name",
            "role_name",
            "job_title",
            "role",
        ),
        "topic": ("main_topic", "title", "subject", "document_title"),
        "main_topic": ("topic", "title", "subject", "document_title"),
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


def _capability_intent_hint(capability_id: str) -> str | None:
    normalized = str(capability_id or "").strip().lower()
    if not normalized:
        return None
    if any(token in normalized for token in (".validate", "validate.", "schema.validate")):
        return "validate"
    if any(token in normalized for token in (".render", ".render_", "render.")):
        return "render"
    if any(
        token in normalized
        for token in (
            ".transform",
            ".improve",
            ".repair",
            "derive_output_filename",
            ".rerank",
        )
    ):
        return "transform"
    if any(
        token in normalized
        for token in (
            ".search",
            ".list",
            ".read",
            ".retrieve",
            ".download",
            ".load",
            ".me",
        )
    ):
        return "io"
    if any(
        token in normalized
        for token in (
            ".generate",
            ".create",
            ".compose",
            "generate_iterative",
        )
    ):
        return "generate"
    return None


def _intent_disagreement_question(intents: Iterable[str]) -> str:
    normalized = [
        intent
        for intent in (normalize_task_intent(value) for value in intents)
        if intent
    ]
    ordered: list[str] = []
    for intent in normalized:
        if intent not in ordered:
            ordered.append(intent)
    if len(ordered) >= 2:
        primary = _INTENT_ACTION_PHRASES.get(ordered[0], ordered[0])
        secondary = _INTENT_ACTION_PHRASES.get(ordered[1], ordered[1])
        return f"Should I {primary}, or should I {secondary} for this request?"
    return required_input_question("intent_action", "")


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
    # Generated artifacts often need small transform helpers, such as deriving an
    # output filename, before the final render/write step.
    if segment_intent == "generate" and task_intent == "transform":
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
    ignored_required_inputs = _ignored_document_spec_generation_required_inputs(
        tool_name,
        capability_id,
    )
    if _capability_derives_output_path(capability_id or tool_name):
        ignored_required_inputs.update({"goal", "workspace_path"})
    for key in slots.get("must_have_inputs", []):
        if not isinstance(key, str) or not key:
            continue
        if key in ignored_required_inputs:
            continue
        if key == "output_format" and _tool_output_format_hint(capability_id or tool_name):
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
    ignore_final_artifact_format = (
        slots.get("artifact_type") == "document_spec"
        and (
            _is_document_spec_generation_identifier(tool_name)
            or _is_document_spec_generation_identifier(capability_id)
        )
    )
    if isinstance(output_format, str) and output_format:
        resolved_format = _normalize_output_format(output_format)
        render_hint = _tool_output_format_hint(capability_id or tool_name)
        if (
            normalize_task_intent(task_intent) == "render"
            and resolved_format == "json"
            and render_hint in {"docx", "pdf", "md", "txt", "html"}
        ):
            resolved_format = render_hint
        if resolved_format and not ignore_final_artifact_format:
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
            hinted = render_hint
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


def _capability_derives_output_path(capability_id_or_tool_name: str) -> bool:
    normalized = str(capability_id_or_tool_name or "").strip().lower()
    return normalized in {
        "derive_output_filename",
    }


def _infer_intent_from_text_with_source(
    text: str,
    *,
    source: str,
) -> TaskIntentInference:
    normalized = text.lower()
    if "pull request" in normalized or re.search(r"\bpr\b", normalized):
        if _contains_any(normalized, ("create", "open", "submit", "raise")):
            confidence = 0.9 if source == _INTENT_SOURCE_TASK_TEXT else 0.82
            return TaskIntentInference(intent="io", source=source, confidence=confidence)
    if _contains_any(normalized, ("push", "publish", "upload")) and _contains_any(
        normalized, ("github", "repo", "repository", "file", "files", "branch")
    ):
        confidence = 0.88 if source == _INTENT_SOURCE_TASK_TEXT else 0.8
        return TaskIntentInference(intent="io", source=source, confidence=confidence)
    if "openapi" in normalized and _contains_any(normalized, ("improve", "repair", "iterate")):
        confidence = 0.86 if source == _INTENT_SOURCE_TASK_TEXT else 0.78
        return TaskIntentInference(intent="transform", source=source, confidence=confidence)
    if "openapi" in normalized and _contains_any(normalized, ("generate", "create", "build")):
        confidence = 0.86 if source == _INTENT_SOURCE_TASK_TEXT else 0.78
        return TaskIntentInference(intent="generate", source=source, confidence=confidence)
    for intent, keywords in _KEYWORD_MAP:
        if _contains_any(normalized, keywords):
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
        if _contains_any(text, keywords):
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
    action_pattern = "|".join(re.escape(token) for token in _CLAUSE_SPLIT_ACTION_HINTS)
    normalized = re.sub(
        rf",\s*(?=(?:{action_pattern})\b)",
        " then ",
        normalized,
        flags=re.IGNORECASE,
    )
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
    if intent == "generate" and _looks_like_workspace_delete_request(lowered):
        base = []
    if intent == "render":
        if "pdf" in lowered:
            base.append("output_format=pdf")
        elif "docx" in lowered:
            base.append("output_format=docx")
    if intent == "io" and any(token in lowered for token in ("github", "repo", "repository")):
        base.append("query")
    if intent == "io" and any(token in lowered for token in ("pull request", "branch", "files")):
        base.append("target_repo")
    if intent == "generate" and "openapi" in lowered:
        base.append("instruction")
    if intent == "transform" and "openapi" in lowered:
        base.append("input_data")
    deduped: list[str] = []
    for item in base:
        if item not in deduped:
            deduped.append(item)
    return tuple(deduped)


def _suggested_capabilities_for_clause(
    intent: str,
    clause: str,
    *,
    goal_context: str = "",
) -> tuple[str, ...]:
    suggestions = list(_SUGGESTED_CAPABILITIES_BY_INTENT.get(intent, ()))
    lowered = clause.lower()
    context_lowered = f"{goal_context} {clause}".lower().strip()
    if intent == "io":
        if "semantic memory" in context_lowered:
            suggestions = _prepend_unique(suggestions, "memory.semantic.search", "memory.read")
        elif any(token in context_lowered for token in ("memory", "notes", "preferences", "context")):
            suggestions = _prepend_unique(suggestions, "memory.read")
        if "workspace" in context_lowered and any(token in lowered for token in ("search", "find", "grep")):
            suggestions = _prepend_unique(suggestions, "filesystem.workspace.search_text")
        if "artifact" in context_lowered and any(token in lowered for token in ("search", "find", "error")):
            suggestions = _prepend_unique(suggestions, "filesystem.artifacts.search_text")
        if "branch" in lowered and "github" in context_lowered:
            suggestions = _prepend_unique(suggestions, "github.branch.list", "github.repo.list")
        elif any(token in context_lowered for token in ("github", "repo", "repository")):
            suggestions = _prepend_unique(suggestions, "github.repo.list")
        if any(token in lowered for token in ("push", "publish")) and any(
            token in context_lowered for token in ("github", "repo", "repository", "files")
        ):
            suggestions = _prepend_unique(suggestions, "github.files.push")
        if "pull request" in lowered or re.search(r"\bpr\b", lowered):
            suggestions = _prepend_unique(suggestions, "github.pull_request.create")
    if intent == "render":
        if "pdf" in lowered:
            suggestions = ["document.pdf.render"]
        elif "docx" in lowered:
            suggestions = ["document.docx.render"]
    if intent == "generate":
        if _looks_like_workspace_delete_request(context_lowered):
            suggestions = _prepend_unique(suggestions, "filesystem.workspace.delete")
        if "openapi" in context_lowered:
            suggestions = _prepend_unique(suggestions, "openapi.spec.generate_iterative")
        if any(token in lowered for token in ("code changes", "codegen", "code change", "source code")):
            suggestions = _prepend_unique(suggestions, "codegen.generate")
        if "markdown" in lowered:
            suggestions = _prepend_unique(suggestions, "document.spec.generate_from_markdown")
        if any(token in lowered for token in ("runbook", "document", "report", "brief", "release notes")):
            suggestions = _prepend_unique(suggestions, "document.spec.generate", "llm.text.generate")
    if intent == "transform":
        if "openapi" in context_lowered:
            suggestions = _prepend_unique(
                suggestions,
                "openapi.spec.improve_iterative",
                "utility.json.transform",
            )
        if any(token in lowered for token in ("repair", "invalid")) and any(
            token in lowered for token in ("document", "schema", "spec", "field")
        ):
            suggestions = _prepend_unique(
                suggestions,
                "document.spec.repair",
                "utility.json.transform",
            )
        if (
            "openapi" not in lowered
            and "improve" in lowered
            and any(token in lowered for token in ("document", "spec"))
        ):
            suggestions = _prepend_unique(
                suggestions,
                "document.spec.improve",
                "utility.json.transform",
            )
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
        suggested_capabilities = _suggested_capabilities_for_clause(
            inferred.intent,
            clause,
            goal_context=goal,
        )
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
