#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import redis

from libs.core import capability_search_feedback, events


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build capability-search feedback rows.")
    parser.add_argument(
        "--source",
        choices=("redis", "outbox", "auto"),
        default="auto",
        help="Load events from Redis streams, SQL outbox, or auto-detect.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/evals/capability_search_feedback.jsonl"),
        help="Path to write joined feedback rows.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5000,
        help="Maximum number of events to scan from each source.",
    )
    return parser


def _load_outbox_events(limit: int) -> list[dict[str, Any]]:
    from services.api.app.database import SessionLocal
    from services.api.app.models import EventOutboxRecord

    with SessionLocal() as db:
        rows = (
            db.query(EventOutboxRecord)
            .filter(
                EventOutboxRecord.event_type.in_(
                    [
                        "plan.capability_search",
                        "plan.capability_selection",
                        "plan.created",
                        "task.completed",
                        "task.failed",
                    ]
                )
            )
            .order_by(EventOutboxRecord.created_at.asc())
            .limit(max(1, limit))
            .all()
        )
    events_out: list[dict[str, Any]] = []
    for row in rows:
        envelope = row.envelope_json if isinstance(row.envelope_json, dict) else {}
        if envelope:
            events_out.append(envelope)
    return events_out


def _load_redis_events(limit: int) -> list[dict[str, Any]]:
    redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")
    client = redis.Redis.from_url(redis_url, decode_responses=True)
    stream_names = [events.PLAN_STREAM, events.TASK_STREAM]
    items: list[dict[str, Any]] = []
    for stream_name in stream_names:
        for _message_id, payload in client.xrange(stream_name, count=max(1, limit)):
            raw = payload.get("data")
            if not isinstance(raw, str):
                continue
            try:
                envelope = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(envelope, dict):
                items.append(envelope)
    return items


def main() -> int:
    args = _build_arg_parser().parse_args()
    envelopes: list[dict[str, Any]] = []
    source_used = args.source
    try:
        if args.source in {"redis", "auto"}:
            try:
                envelopes = _load_redis_events(args.limit)
                if envelopes:
                    source_used = "redis"
            except Exception:
                if args.source == "redis":
                    raise
        if not envelopes and args.source in {"outbox", "auto"}:
            envelopes = _load_outbox_events(args.limit)
            source_used = "outbox"
    except Exception as exc:  # noqa: BLE001
        print(f"failed_to_build_capability_search_feedback:{exc}", file=sys.stderr)
        return 2

    rows = capability_search_feedback.build_feedback_rows(envelopes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )
    print(f"wrote feedback rows: {args.output} ({len(rows)} rows, source={source_used})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
