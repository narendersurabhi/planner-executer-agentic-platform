#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from libs.core import chat_routing_feedback, chat_routing_training, models
from services.api.app import feedback_service
from services.api.app.database import SessionLocal


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build chat-routing feedback rows and reranker data.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/evals/chat_routing_feedback.jsonl"),
        help="Path to write chat-routing feedback rows.",
    )
    parser.add_argument(
        "--training-output",
        type=Path,
        default=Path("training/chat_routing_reranker_train.jsonl"),
        help="Path to write reranker training examples.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5000,
        help="Maximum number of chat-message feedback examples to export.",
    )
    return parser


def main() -> int:
    args = _build_arg_parser().parse_args()
    with SessionLocal() as db:
        export = feedback_service.export_feedback_examples(
            db,
            target_type=models.FeedbackTargetType.chat_message,
            sentiments=[
                models.FeedbackSentiment.positive,
                models.FeedbackSentiment.partial,
                models.FeedbackSentiment.negative,
            ],
            limit=max(1, args.limit),
        )
    items = [item.model_dump(mode="json") for item in export.items]
    feedback_rows = chat_routing_feedback.build_feedback_rows(items)
    training_rows = chat_routing_training.build_reranker_training_examples(feedback_rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in feedback_rows)
        + ("\n" if feedback_rows else ""),
        encoding="utf-8",
    )
    args.training_output.parent.mkdir(parents=True, exist_ok=True)
    args.training_output.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in training_rows)
        + ("\n" if training_rows else ""),
        encoding="utf-8",
    )
    print(
        "wrote chat-routing feedback rows: "
        f"{args.output} ({len(feedback_rows)} rows) and {args.training_output} "
        f"({len(training_rows)} examples)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
