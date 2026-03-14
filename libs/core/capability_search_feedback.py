from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


def _normalize_str(value: Any) -> str:
    return str(value or "").strip()


def _normalize_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    items: list[str] = []
    seen: set[str] = set()
    for entry in value:
        text = _normalize_str(entry)
        if not text or text in seen:
            continue
        seen.add(text)
        items.append(text)
    return items


def _event_payload(event: Mapping[str, Any]) -> dict[str, Any]:
    payload = event.get("payload")
    return payload if isinstance(payload, dict) else {}


def build_feedback_rows(envelopes: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, Any]] = {}

    def ensure_row(job_id: str, correlation_id: str) -> dict[str, Any]:
        key = (job_id, correlation_id)
        row = grouped.get(key)
        if row is None:
            row = {
                "job_id": job_id,
                "correlation_id": correlation_id,
                "search_queries": [],
                "retrieved_capabilities": [],
                "selected_capabilities": [],
                "executed_capabilities": [],
                "execution_succeeded": False,
                "planner_override": [],
                "retrieved_selected": [],
                "retrieved_executed": [],
                "hard_negative_ids": [],
            }
            grouped[key] = row
        return row

    for envelope in envelopes:
        event_type = _normalize_str(envelope.get("type"))
        correlation_id = _normalize_str(envelope.get("correlation_id")) or "unknown"
        job_id = _normalize_str(envelope.get("job_id"))
        payload = _event_payload(envelope)
        if not job_id:
            job_id = _normalize_str(payload.get("job_id"))
        if not job_id:
            continue
        row = ensure_row(job_id, correlation_id)
        if event_type == "plan.capability_search":
            query = _normalize_str(payload.get("query") or payload.get("goal"))
            if query and query not in row["search_queries"]:
                row["search_queries"].append(query)
            for result in payload.get("results", []):
                if not isinstance(result, Mapping):
                    continue
                capability_id = _normalize_str(result.get("id"))
                if capability_id and capability_id not in row["retrieved_capabilities"]:
                    row["retrieved_capabilities"].append(capability_id)
        elif event_type in {"plan.capability_selection", "plan.created"}:
            selected = _normalize_str_list(payload.get("selected_capabilities"))
            if not selected and event_type == "plan.created":
                tasks = payload.get("tasks")
                if isinstance(tasks, list):
                    for task in tasks:
                        if not isinstance(task, Mapping):
                            continue
                        for tool_name in task.get("tool_requests", []) or []:
                            normalized = _normalize_str(tool_name)
                            if normalized and normalized not in selected:
                                selected.append(normalized)
            for capability_id in selected:
                if capability_id not in row["selected_capabilities"]:
                    row["selected_capabilities"].append(capability_id)
        elif event_type in {"task.completed", "task.failed"}:
            tool_calls = payload.get("tool_calls")
            call_names: list[str] = []
            if isinstance(tool_calls, list):
                for call in tool_calls:
                    if not isinstance(call, Mapping):
                        continue
                    tool_name = _normalize_str(call.get("tool_name"))
                    if tool_name:
                        call_names.append(tool_name)
            if not call_names:
                outputs = payload.get("outputs")
                if isinstance(outputs, Mapping):
                    tool_error = outputs.get("tool_error")
                    if isinstance(tool_error, Mapping):
                        tool_name = _normalize_str(tool_error.get("tool_name"))
                        if tool_name:
                            call_names.append(tool_name)
            for capability_id in call_names:
                row["executed_capabilities"].append(
                    {"id": capability_id, "status": "completed" if event_type == "task.completed" else "failed"}
                )
            if event_type == "task.completed":
                row["execution_succeeded"] = True

    rows: list[dict[str, Any]] = []
    for row in grouped.values():
        retrieved = list(row["retrieved_capabilities"])
        selected = list(row["selected_capabilities"])
        executed = [
            entry["id"]
            for entry in row["executed_capabilities"]
            if isinstance(entry, Mapping) and _normalize_str(entry.get("id"))
        ]
        row["retrieved_selected"] = [cap for cap in selected if cap in retrieved]
        row["retrieved_executed"] = [cap for cap in executed if cap in retrieved]
        row["planner_override"] = [cap for cap in selected if cap not in retrieved]
        row["hard_negative_ids"] = [
            cap for cap in retrieved if cap not in row["retrieved_selected"] and cap not in row["retrieved_executed"]
        ]
        rows.append(row)
    rows.sort(key=lambda item: (item["job_id"], item["correlation_id"]))
    return rows
