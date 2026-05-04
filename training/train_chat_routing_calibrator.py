#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from libs.core import chat_routing_calibrator


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the chat-routing calibrator from reranker rows.")
    parser.add_argument(
        "--training-data",
        type=Path,
        default=Path("training/chat_routing_reranker_train.jsonl"),
        help="Path to chat-routing reranker training rows.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/evals/chat_routing_calibrator.json"),
        help="Path to write the trained calibrator model.",
    )
    parser.add_argument("--epochs", type=int, default=120, help="Number of SGD epochs.")
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.18,
        help="Learning rate for the logistic calibrator.",
    )
    return parser


def main() -> int:
    args = _build_arg_parser().parse_args()
    rows: list[dict] = []
    if args.training_data.exists():
        for raw_line in args.training_data.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    model = chat_routing_calibrator.train_model(
        rows,
        epochs=max(1, args.epochs),
        learning_rate=max(0.001, float(args.learning_rate)),
    )
    if int(model.get("training_examples") or 0) <= 0:
        print("no chat-routing training examples available", file=sys.stderr)
        return 2
    target = chat_routing_calibrator.save_model(model, args.output)
    print(
        "wrote chat-routing calibrator: "
        f"{target} ({model['training_examples']} examples, accuracy={model['accuracy']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
