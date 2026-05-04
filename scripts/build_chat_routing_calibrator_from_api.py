#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from libs.core import chat_routing_calibrator, chat_routing_feedback, chat_routing_training


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build chat-routing calibrator artifacts from feedback examples exported over the API."
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("CHAT_ROUTING_FEEDBACK_BASE_URL", "http://127.0.0.1:18000"),
        help="Base URL for the API environment.",
    )
    parser.add_argument(
        "--bearer-token",
        default=os.getenv("CHAT_ROUTING_FEEDBACK_BEARER_TOKEN") or None,
        help="Optional bearer token for the feedback export endpoint.",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=float(os.getenv("CHAT_ROUTING_FEEDBACK_TIMEOUT_S", "30.0")),
        help="HTTP timeout in seconds.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=int(os.getenv("CHAT_ROUTING_FEEDBACK_LIMIT", "5000")),
        help="Maximum number of chat-message feedback examples to export.",
    )
    parser.add_argument(
        "--min-examples",
        type=int,
        default=int(os.getenv("CHAT_ROUTING_CALIBRATOR_MIN_EXAMPLES", "10")),
        help="Minimum number of reranker training examples required to write a model.",
    )
    parser.add_argument(
        "--feedback-output",
        type=Path,
        default=Path("artifacts/evals/chat_routing_feedback.jsonl"),
        help="Path to write chat-routing feedback rows.",
    )
    parser.add_argument(
        "--training-output",
        type=Path,
        default=Path("training/chat_routing_reranker_train.jsonl"),
        help="Path to write chat-routing reranker training rows.",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=Path("artifacts/evals/chat_routing_calibrator.json"),
        help="Path to write the trained calibrator model.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("artifacts/evals/chat_routing_calibrator_report.json"),
        help="Path to write the build summary report.",
    )
    return parser


def _normalize_str(value: Any) -> str:
    return str(value or "").strip()


def _request_examples(
    *,
    base_url: str,
    bearer_token: str | None,
    timeout_s: float,
    limit: int,
) -> list[dict[str, Any]]:
    query = urlencode(
        [
            ("target_type", "chat_message"),
            ("sentiment", "positive"),
            ("sentiment", "partial"),
            ("sentiment", "negative"),
            ("limit", str(max(1, limit))),
            ("format", "jsonl"),
        ]
    )
    headers = {"Accept": "application/x-ndjson"}
    if bearer_token:
        headers["Authorization"] = f"Bearer {bearer_token}"
    request = Request(
        f"{base_url.rstrip('/')}/feedback/examples?{query}",
        headers=headers,
        method="GET",
    )
    with urlopen(request, timeout=timeout_s) as response:
        payload = response.read().decode("utf-8")
    examples: list[dict[str, Any]] = []
    for raw_line in payload.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        item = json.loads(line)
        if isinstance(item, dict):
            examples.append(item)
    return examples


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )


def main() -> int:
    args = _build_arg_parser().parse_args()
    try:
        examples = _request_examples(
            base_url=args.base_url,
            bearer_token=args.bearer_token,
            timeout_s=max(1.0, float(args.timeout_s)),
            limit=max(1, int(args.limit)),
        )
    except Exception as exc:  # noqa: BLE001
        print(f"failed_to_fetch_chat_routing_feedback_examples:{exc}", file=sys.stderr)
        return 2

    feedback_rows = chat_routing_feedback.build_feedback_rows(examples)
    training_rows = chat_routing_training.build_reranker_training_examples(feedback_rows)
    model = chat_routing_calibrator.train_model(training_rows)

    _write_jsonl(args.feedback_output, feedback_rows)
    _write_jsonl(args.training_output, training_rows)

    model_written = False
    model_path = None
    if len(training_rows) >= max(1, int(args.min_examples)):
        model_path = str(chat_routing_calibrator.save_model(model, args.model_output))
        model_written = True

    summary = {
        "base_url": _normalize_str(args.base_url),
        "fetched_examples": len(examples),
        "feedback_rows": len(feedback_rows),
        "training_examples": len(training_rows),
        "min_examples_required": max(1, int(args.min_examples)),
        "model_written": model_written,
        "model_path": model_path,
        "model_version": _normalize_str(model.get("model_version")) or None,
        "model_accuracy": float(model.get("accuracy") or 0.0),
        "positive_examples": int(model.get("positive_examples") or 0),
        "negative_examples": int(model.get("negative_examples") or 0),
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(
        "wrote chat-routing calibrator bundle: "
        f"feedback_rows={len(feedback_rows)} training_examples={len(training_rows)} "
        f"model_written={model_written}"
    )
    if not model_written:
        print(
            "insufficient_chat_routing_training_examples:"
            f" required>={max(1, int(args.min_examples))} actual={len(training_rows)}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
