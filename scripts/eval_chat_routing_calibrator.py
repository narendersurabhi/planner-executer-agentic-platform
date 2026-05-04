#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from libs.core import chat_routing_calibrator, chat_routing_calibrator_eval


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate chat-routing calibrator disagreements against feedback rows."
    )
    parser.add_argument(
        "--feedback",
        type=Path,
        default=Path("artifacts/evals/chat_routing_feedback.jsonl"),
        help="Path to chat-routing feedback rows.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("artifacts/evals/chat_routing_calibrator.json"),
        help="Path to trained chat-routing calibrator model.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/evals/chat_routing_calibrator_replay_report.json"),
        help="Path to write replay report.",
    )
    parser.add_argument(
        "--min-probability",
        type=float,
        default=float(os.getenv("CHAT_ROUTING_CALIBRATOR_MIN_PROBABILITY", "0.65")),
        help="Minimum probability for live override simulation.",
    )
    parser.add_argument(
        "--min-margin",
        type=float,
        default=float(os.getenv("CHAT_ROUTING_CALIBRATOR_MIN_MARGIN", "0.08")),
        help="Minimum probability margin for live override simulation.",
    )
    return parser


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def main() -> int:
    args = _build_arg_parser().parse_args()
    rows = _load_jsonl(args.feedback)
    model = chat_routing_calibrator.load_model(args.model)
    if not rows:
        print("missing_chat_routing_feedback_rows", file=sys.stderr)
        return 2
    if model is None:
        print("missing_chat_routing_calibrator_model", file=sys.stderr)
        return 2
    report = chat_routing_calibrator_eval.evaluate_feedback_rows(
        rows,
        model=model,
        min_probability=max(0.0, min(1.0, float(args.min_probability))),
        min_margin=max(0.0, min(1.0, float(args.min_margin))),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = report.get("summary") or {}
    print("Chat Routing Calibrator Replay Summary")
    print(f"  rows: {int(summary.get('row_count', 0))}")
    print(f"  disagreement_rate: {float(summary.get('disagreement_rate', 0.0)):.3f}")
    print(f"  shadow_better_rate: {float(summary.get('shadow_better_rate', 0.0)):.3f}")
    print(f"  shadow_worse_rate: {float(summary.get('shadow_worse_rate', 0.0)):.3f}")
    print(f"  live_override_rate: {float(summary.get('live_override_rate', 0.0)):.3f}")
    print(f"  live_better_rate: {float(summary.get('live_better_rate', 0.0)):.3f}")
    print(f"  live_worse_rate: {float(summary.get('live_worse_rate', 0.0)):.3f}")
    print(f"wrote report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
