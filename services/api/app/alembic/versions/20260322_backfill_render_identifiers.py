"""backfill legacy document render capability and tool identifiers

Revision ID: 20260322_backfill_render_identifiers
Revises: 20260320_add_runtime_observability
Create Date: 2026-03-22
"""

from __future__ import annotations

from typing import Any

from alembic import op
import sqlalchemy as sa


revision = "20260322_backfill_render_identifiers"
down_revision = "20260320_add_runtime_observability"
branch_labels = None
depends_on = None


_STRING_REPLACEMENTS = {
    "document.docx.generate": "document.docx.render",
    "document.pdf.generate": "document.pdf.render",
    "docx_generate_from_spec": "docx_render_from_spec",
    "pdf_generate_from_spec": "pdf_render_from_spec",
}

_URI_PREFIXES = ("capability://", "tool://")

_TABLE_UPDATES = (
    {"name": "jobs", "pk": ("id",), "json": ("context_json", "metadata"), "text": ()},
    {"name": "chat_sessions", "pk": ("id",), "json": ("metadata",), "text": ()},
    {"name": "chat_messages", "pk": ("id",), "json": ("metadata", "action"), "text": ()},
    {
        "name": "workflow_definitions",
        "pk": ("id",),
        "json": ("context_json", "draft", "metadata"),
        "text": (),
    },
    {
        "name": "workflow_versions",
        "pk": ("id",),
        "json": ("context_json", "draft", "compiled_plan", "metadata"),
        "text": (),
    },
    {"name": "workflow_triggers", "pk": ("id",), "json": ("config", "metadata"), "text": ()},
    {
        "name": "workflow_runs",
        "pk": ("id",),
        "json": ("requested_context", "metadata"),
        "text": (),
    },
    {"name": "plans", "pk": ("id",), "json": ("policy_decision",), "text": ()},
    {"name": "tasks", "pk": ("id",), "json": ("tool_requests", "tool_inputs"), "text": ()},
    {"name": "task_results", "pk": ("task_id",), "json": ("result",), "text": ()},
    {"name": "step_attempts", "pk": ("id",), "json": ("result_summary",), "text": ()},
    {
        "name": "invocations",
        "pk": ("id",),
        "json": ("request", "response"),
        "text": ("capability_id",),
    },
    {"name": "run_events", "pk": ("id",), "json": ("payload",), "text": ()},
    {"name": "event_outbox", "pk": ("id",), "json": ("envelope_json",), "text": ()},
    {"name": "memory", "pk": ("id",), "json": ("payload", "metadata"), "text": ()},
)


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    for table_config in _TABLE_UPDATES:
        table_name = table_config["name"]
        if not inspector.has_table(table_name):
            continue
        _backfill_table(
            bind,
            table_name=table_name,
            pk_columns=table_config["pk"],
            json_columns=table_config["json"],
            text_columns=table_config["text"],
        )


def downgrade() -> None:
    # Data backfill only. Downgrade is intentionally a no-op.
    return None


def _backfill_table(
    bind: sa.engine.Connection,
    *,
    table_name: str,
    pk_columns: tuple[str, ...],
    json_columns: tuple[str, ...],
    text_columns: tuple[str, ...],
) -> None:
    metadata = sa.MetaData()
    table = sa.Table(table_name, metadata, autoload_with=bind)
    selected_columns = [table.c[column_name] for column_name in (*pk_columns, *json_columns, *text_columns)]
    rows = bind.execute(sa.select(*selected_columns)).mappings()
    for row in rows:
        update_values: dict[str, Any] = {}
        for column_name in json_columns:
            original = row.get(column_name)
            updated = _canonicalize_value(original)
            if updated != original:
                update_values[column_name] = updated
        for column_name in text_columns:
            original = row.get(column_name)
            updated = _canonicalize_string(original)
            if updated != original:
                update_values[column_name] = updated
        if not update_values:
            continue
        where_clause = sa.and_(*(table.c[column_name] == row[column_name] for column_name in pk_columns))
        bind.execute(table.update().where(where_clause).values(**update_values))


def _canonicalize_value(value: Any) -> Any:
    if isinstance(value, str):
        return _canonicalize_string(value)
    if isinstance(value, list):
        return [_canonicalize_value(item) for item in value]
    if isinstance(value, dict):
        updated: dict[Any, Any] = {}
        for key, nested_value in value.items():
            updated_key = _canonicalize_string(key) if isinstance(key, str) else key
            updated[updated_key] = _canonicalize_value(nested_value)
        return updated
    return value


def _canonicalize_string(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    direct = _STRING_REPLACEMENTS.get(value)
    if direct is not None:
        return direct
    for prefix in _URI_PREFIXES:
        if value.startswith(prefix):
            suffix = value[len(prefix) :]
            replaced = _STRING_REPLACEMENTS.get(suffix)
            if replaced is not None:
                return f"{prefix}{replaced}"
    return value
