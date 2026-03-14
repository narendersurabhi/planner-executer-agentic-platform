#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from libs.core import capability_search_feedback_eval


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate capability-search feedback rows.")
    parser.add_argument(
        "--feedback",
        type=Path,
        default=Path("artifacts/evals/capability_search_feedback.jsonl"),
        help="Path to capability-search feedback JSONL rows.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/evals/capability_search_feedback_report.json"),
        help="Path to write feedback eval report.",
    )
    parser.add_argument("--min-planner-selection-hit-rate", type=float, default=None)
    parser.add_argument("--min-execution-success-rate", type=float, default=None)
    return parser


def _load_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def main() -> int:
    args = _build_arg_parser().parse_args()
    rows = _load_rows(args.feedback)
    report = capability_search_feedback_eval.evaluate_feedback_rows(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Capability Search Feedback Eval Summary")
    for key in (
        "row_count",
        "planner_selection_hit_rate",
        "execution_success_rate",
        "planner_override_rate",
        "hard_negative_ratio",
    ):
        print(f"  {key}: {report.get(key)}")
    failures: list[str] = []
    if (
        args.min_planner_selection_hit_rate is not None
        and float(report.get("planner_selection_hit_rate", 0.0))
        < float(args.min_planner_selection_hit_rate)
    ):
        failures.append("planner_selection_hit_rate_below_threshold")
    if (
        args.min_execution_success_rate is not None
        and float(report.get("execution_success_rate", 0.0))
        < float(args.min_execution_success_rate)
    ):
        failures.append("execution_success_rate_below_threshold")
    if failures:
        for failure in failures:
            print(failure, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
