from __future__ import annotations

import os
import re
from pathlib import Path
from collections.abc import Iterable, Mapping
from typing import Any

from . import capability_registry, capability_reranker


_STOPWORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "into",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}

_INTENT_TOKENS: dict[str, tuple[str, ...]] = {
    "generate": ("generate", "create", "draft", "compose", "write", "produce"),
    "transform": ("transform", "derive", "convert", "improve", "repair"),
    "validate": ("validate", "check", "lint", "schema", "verify"),
    "render": ("render", "docx", "pdf", "format"),
    "io": ("read", "write", "search", "list", "fetch", "memory", "file", "github"),
}

_DEFAULT_RERANK_FEEDBACK_PATH = Path(
    os.getenv("CAPABILITY_SEARCH_FEEDBACK_PATH", "artifacts/evals/capability_search_feedback.jsonl")
)


def _tokenize(value: str) -> list[str]:
    if not value:
        return []
    lowered = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", value).lower()
    normalized = re.sub(r"[^a-z0-9]+", " ", lowered)
    tokens: list[str] = []
    for token in normalized.split():
        if len(token) < 2 or token in _STOPWORDS:
            continue
        tokens.append(token)
    return tokens


def _coerce_tags(value: Any) -> tuple[str, ...]:
    if not isinstance(value, Iterable) or isinstance(value, (str, bytes)):
        return ()
    tags: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        normalized = item.strip()
        if normalized and normalized not in tags:
            tags.append(normalized)
    return tuple(tags)


def build_capability_search_entry(
    capability_id: str,
    spec: capability_registry.CapabilitySpec,
) -> dict[str, Any]:
    group = str(spec.group or "").strip()
    subgroup = str(spec.subgroup or "").strip()
    description = str(spec.description or "").strip()
    tags = _coerce_tags(spec.tags)
    required_inputs = tuple(
        str(item).strip()
        for item in (
            spec.planner_hints.get("required_inputs", [])
            if isinstance(spec.planner_hints, Mapping)
            else []
        )
        if str(item).strip()
    )
    field_blob = " ".join(
        part
        for part in (
            capability_id,
            description,
            group,
            subgroup,
            " ".join(tags),
            " ".join(required_inputs),
        )
        if part
    )
    return {
        "id": capability_id,
        "description": description,
        "group": group,
        "subgroup": subgroup,
        "tags": list(tags),
        "required_inputs": list(required_inputs),
        "search_blob": field_blob.lower(),
        "tokens": sorted(set(_tokenize(field_blob))),
    }


def build_capability_search_entries(
    capabilities: Mapping[str, capability_registry.CapabilitySpec],
) -> list[dict[str, Any]]:
    entries = [
        build_capability_search_entry(capability_id, spec)
        for capability_id, spec in sorted(capabilities.items())
    ]
    return entries


def search_capabilities(
    *,
    query: str,
    capability_entries: Iterable[Mapping[str, Any]],
    limit: int = 8,
    intent_hint: str | None = None,
    rerank_feedback_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    query_tokens = sorted(set(_tokenize(query)))
    if not query_tokens:
        return []
    lowered_query = query.lower()
    intent_tokens = _INTENT_TOKENS.get(str(intent_hint or "").strip().lower(), ())
    results: list[dict[str, Any]] = []
    for entry in capability_entries:
        capability_id = str(entry.get("id") or "").strip()
        if not capability_id:
            continue
        entry_tokens = {
            token
            for token in entry.get("tokens", [])
            if isinstance(token, str) and token.strip()
        }
        if not entry_tokens:
            entry_tokens = set(_tokenize(str(entry.get("search_blob") or capability_id)))
        overlap = sorted(token for token in query_tokens if token in entry_tokens)
        score = float(len(overlap) * 10)
        reasons: list[str] = []
        if overlap:
            reasons.append(f"token overlap: {', '.join(overlap[:4])}")
        capability_lower = capability_id.lower()
        if capability_lower in lowered_query:
            score += 50.0
            reasons.append("goal mentions capability id")
        if any(token in capability_lower for token in query_tokens):
            score += 8.0
        search_blob = str(entry.get("search_blob") or capability_lower)
        if intent_tokens:
            matched_intent = [token for token in intent_tokens if token in search_blob]
            if matched_intent:
                score += float(len(matched_intent) * 4)
                reasons.append(f"intent-aligned: {', '.join(matched_intent[:3])}")
        tags = [tag for tag in entry.get("tags", []) if isinstance(tag, str)]
        tag_matches = [tag for tag in tags if tag.lower() in query_tokens]
        if tag_matches:
            score += float(len(tag_matches) * 6)
            reasons.append(f"tag match: {', '.join(tag_matches[:3])}")
        if score <= 0:
            continue
        results.append(
            {
                "id": capability_id,
                "score": round(score, 3),
                "reason": "; ".join(reasons) or "semantic match",
                "source": "semantic_search",
            }
        )
    results.sort(key=lambda item: (-float(item["score"]), str(item["id"])))
    if rerank_feedback_rows is None and os.getenv("CAPABILITY_SEARCH_RERANK_ENABLED", "true").lower() == "true":
        rerank_feedback_rows = capability_reranker.load_feedback_rows(_DEFAULT_RERANK_FEEDBACK_PATH)
    return capability_reranker.rerank_capability_results(
        query=query,
        intent_hint=intent_hint,
        results=results,
        feedback_rows=rerank_feedback_rows,
        limit=max(1, limit),
    )
