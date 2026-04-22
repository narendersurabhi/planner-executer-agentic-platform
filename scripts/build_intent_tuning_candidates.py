#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from libs.core import intent_tuning


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export reviewed intent tuning candidates into eval-ready artifacts."
    )
    parser.add_argument(
        "--jsonl-output",
        type=Path,
        default=Path("artifacts/evals/intent_tuning_candidates.jsonl"),
        help="Path to write the flattened JSONL tuning bundle.",
    )
    parser.add_argument(
        "--yaml-output",
        type=Path,
        default=Path("artifacts/evals/intent_tuning_candidates.yaml"),
        help="Path to write YAML gold-case stubs.",
    )
    parser.add_argument(
        "--review-label",
        type=str,
        default=None,
        help="Optional review label filter.",
    )
    parser.add_argument(
        "--intent",
        type=str,
        default=None,
        help="Optional normalized intent filter.",
    )
    parser.add_argument(
        "--intent-source",
        type=str,
        default=None,
        help="Optional intent assessment source filter.",
    )
    parser.add_argument(
        "--tuning-focus",
        type=str,
        default=None,
        help="Optional tuning focus filter.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Maximum number of reviewed candidates to export.",
    )
    parser.add_argument(
        "--description",
        type=str,
        default="Intent tuning candidates exported from reviewed feedback.",
        help="Description to embed in the YAML bundle.",
    )
    parser.add_argument(
        "--database-url",
        type=str,
        default=None,
        help="Optional database URL override for loading feedback rows.",
    )
    return parser


def main() -> int:
    args = _build_arg_parser().parse_args()
    if args.database_url:
        os.environ["DATABASE_URL"] = str(args.database_url).strip()
    from services.api.app import feedback_service
    from services.api.app.database import SessionLocal

    try:
        with SessionLocal() as db:
            export = feedback_service.intent_tuning_candidates(
                db,
                review_label=args.review_label,
                intent=args.intent,
                intent_source=args.intent_source,
                limit=max(1, int(args.limit)),
            )
    except Exception as exc:  # noqa: BLE001
        print(f"failed_to_build_intent_tuning_candidates:{exc}", file=sys.stderr)
        return 2

    items = [item.model_dump(mode="json") for item in export.items]
    if args.tuning_focus:
        normalized_focus = str(args.tuning_focus or "").strip()
        items = [item for item in items if str(item.get("tuning_focus") or "").strip() == normalized_focus]

    jsonl_payload = intent_tuning.dumps_intent_tuning_rows_jsonl(items)
    yaml_payload = intent_tuning.dumps_intent_tuning_gold_yaml(
        items,
        description=args.description,
    )

    args.jsonl_output.parent.mkdir(parents=True, exist_ok=True)
    args.yaml_output.parent.mkdir(parents=True, exist_ok=True)
    args.jsonl_output.write_text(
        jsonl_payload + ("\n" if jsonl_payload else ""),
        encoding="utf-8",
    )
    args.yaml_output.write_text(yaml_payload, encoding="utf-8")
    print(
        "wrote intent tuning candidates: "
        f"{args.jsonl_output} and {args.yaml_output} ({len(items)} items)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
