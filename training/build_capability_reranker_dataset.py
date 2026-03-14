#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from libs.core import capability_search_training


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build capability reranker dataset from feedback rows.")
    parser.add_argument(
        "--feedback",
        type=Path,
        default=Path("artifacts/evals/capability_search_feedback.jsonl"),
        help="Path to feedback JSONL rows.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("training/capability_reranker_train.jsonl"),
        help="Path to write reranker training examples.",
    )
    return parser


def main() -> int:
    args = _build_arg_parser().parse_args()
    rows: list[dict] = []
    if args.feedback.exists():
        for raw_line in args.feedback.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    examples = capability_search_training.build_reranker_training_examples(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in examples) + ("\n" if examples else ""),
        encoding="utf-8",
    )
    print(f"wrote reranker dataset: {args.output} ({len(examples)} examples)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
