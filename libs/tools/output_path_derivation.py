from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping


class OutputPathDerivationError(ValueError):
    pass


def resolve_or_derive_output_path(
    payload: Mapping[str, Any],
    *,
    extension: str,
    default_output_dir: str = "documents",
) -> str:
    provided = payload.get("path")
    if isinstance(provided, str) and provided.strip():
        return provided.strip()
    if provided is not None and not (isinstance(provided, str) and not provided.strip()):
        raise OutputPathDerivationError("path must be a non-empty string")

    memory_context = _select_job_context_from_memory(payload.get("memory"))
    nested_context = memory_context.get("context_json")
    if not isinstance(nested_context, Mapping):
        nested_context = {}

    topic = _pick_str(
        payload.get("topic"),
        memory_context.get("topic"),
        nested_context.get("topic"),
        _topic_from_document_spec(payload.get("document_spec")),
    )
    if not topic:
        topic = "document"

    output_dir = (
        _pick_str(
            payload.get("output_dir"),
            memory_context.get("output_dir"),
            nested_context.get("output_dir"),
        )
        or default_output_dir
    )
    output_dir = output_dir.strip().strip("/")
    if not output_dir:
        output_dir = default_output_dir
    if output_dir.startswith("/") or ".." in Path(output_dir).parts:
        raise OutputPathDerivationError("Invalid output_dir")

    date_value = _pick_str(
        payload.get("today"),
        payload.get("date"),
        memory_context.get("today"),
        memory_context.get("date"),
        nested_context.get("today"),
        nested_context.get("date"),
    )
    if not date_value:
        date_value = datetime.now(UTC).date().isoformat()

    topic_slug = re.sub(r"[^a-z0-9]+", "_", topic.lower())
    topic_slug = re.sub(r"_+", "_", topic_slug).strip("_") or "document"
    date_slug = re.sub(r"[^0-9]+", "_", date_value)
    date_slug = re.sub(r"_+", "_", date_slug).strip("_")
    if not date_slug:
        raise OutputPathDerivationError("Invalid date")

    normalized_extension = extension.strip().lower().lstrip(".")
    if not normalized_extension:
        raise OutputPathDerivationError("Invalid extension")
    if not re.fullmatch(r"[a-z0-9]{1,16}", normalized_extension):
        raise OutputPathDerivationError("Invalid extension")

    return f"{output_dir}/{topic_slug}_{date_slug}.{normalized_extension}"


def _pick_str(*values: Any) -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _topic_from_document_spec(document_spec: Any) -> str:
    if not isinstance(document_spec, Mapping):
        return ""

    tokens = document_spec.get("tokens")
    if isinstance(tokens, Mapping):
        for key in ("topic", "title", "subject", "document_title", "name"):
            value = tokens.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    for key in ("topic", "title", "subject", "document_title", "name"):
        value = document_spec.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    blocks = document_spec.get("blocks")
    if isinstance(blocks, list):
        for block in blocks:
            if not isinstance(block, Mapping):
                continue
            block_type = block.get("type")
            if block_type not in {"heading", "paragraph", "text"}:
                continue
            text = block.get("text")
            if isinstance(text, str) and text.strip():
                return text.strip()

    return ""


def _select_job_context_from_memory(memory: Any) -> dict[str, Any]:
    if not isinstance(memory, Mapping):
        return {}
    entries = memory.get("job_context")
    if not isinstance(entries, list) or not entries:
        return {}

    normalized: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        payload = entry.get("payload")
        if isinstance(payload, Mapping):
            normalized_entry = dict(payload)
            for meta_key in ("_memory_key", "_memory_updated_at"):
                if meta_key in entry:
                    normalized_entry[meta_key] = entry[meta_key]
            normalized.append(normalized_entry)
        else:
            normalized.append(dict(entry))
    if not normalized:
        return {}

    def score(entry: Mapping[str, Any]) -> tuple[int, int, str]:
        topic = entry.get("topic")
        output_dir = entry.get("output_dir")
        updated_at = str(entry.get("_memory_updated_at") or "")
        topic_score = len(topic.strip()) if isinstance(topic, str) else 0
        output_dir_score = len(output_dir.strip()) if isinstance(output_dir, str) else 0
        return (topic_score, output_dir_score, updated_at)

    best = max(normalized, key=score)
    return {k: v for k, v in best.items() if isinstance(k, str) and not k.startswith("_")}
