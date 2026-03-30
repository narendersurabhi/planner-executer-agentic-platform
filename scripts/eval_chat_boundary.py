#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

from libs.core import chat_boundary_eval


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate chat boundary policy behavior against a gold set."
    )
    parser.add_argument(
        "--gold",
        type=Path,
        default=Path("eval/chat_boundary_gold.yaml"),
        help="Path to chat-boundary gold cases (YAML).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/evals/chat_boundary_report.json"),
        help="Path to write JSON report.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print per-case rows.")
    parser.add_argument("--min-accuracy", type=float, default=None)
    parser.add_argument("--max-false-chat-reply-rate", type=float, default=None)
    parser.add_argument("--min-pending-continuation-rate", type=float, default=None)
    parser.add_argument("--max-active-family-drift-rate", type=float, default=None)
    return parser


def _print_summary(summary: Mapping[str, Any]) -> None:
    print("Chat Boundary Eval Summary")
    print(f"  cases: {int(summary.get('case_count', 0))}")
    print(f"  accuracy: {float(summary.get('accuracy', 0.0)):.3f}")
    print(f"  false_chat_reply_rate: {float(summary.get('false_chat_reply_rate', 0.0)):.3f}")
    print(
        f"  execution_escalation_rate: {float(summary.get('execution_escalation_rate', 0.0)):.3f}"
    )
    print(
        "  pending_continuation_rate: "
        f"{float(summary.get('pending_continuation_rate', 0.0)):.3f}"
    )
    print(
        f"  active_family_drift_rate: {float(summary.get('active_family_drift_rate', 0.0)):.3f}"
    )


def _print_case_rows(case_results: list[Mapping[str, Any]]) -> None:
    print("\nPer-case")
    for case in case_results:
        print(
            "  "
            + f"{case.get('case_id', 'unknown')}: "
            + f"expected={case.get('expected_decision', 'unknown')} "
            + f"predicted={case.get('predicted_decision', 'unknown')} "
            + f"match={bool(case.get('match'))} "
            + f"family={case.get('top_family') or '-'} "
            + f"active_family={case.get('active_family') or '-'} "
            + f"alignment={case.get('family_alignment') or '-'}"
        )


def _check_thresholds(report: Mapping[str, Any], args: argparse.Namespace) -> list[str]:
    summary = report.get("summary", {})
    failures: list[str] = []
    accuracy = float(summary.get("accuracy", 0.0))
    false_chat_reply_rate = float(summary.get("false_chat_reply_rate", 0.0))
    pending_continuation_rate = float(summary.get("pending_continuation_rate", 0.0))
    active_family_drift_rate = float(summary.get("active_family_drift_rate", 0.0))
    if args.min_accuracy is not None and accuracy < args.min_accuracy:
        failures.append(f"accuracy_below_threshold:{accuracy:.3f}<{float(args.min_accuracy):.3f}")
    if (
        args.max_false_chat_reply_rate is not None
        and false_chat_reply_rate > args.max_false_chat_reply_rate
    ):
        failures.append(
            "false_chat_reply_rate_above_threshold:"
            + f"{false_chat_reply_rate:.3f}>{float(args.max_false_chat_reply_rate):.3f}"
        )
    if (
        args.min_pending_continuation_rate is not None
        and pending_continuation_rate < args.min_pending_continuation_rate
    ):
        failures.append(
            "pending_continuation_rate_below_threshold:"
            + f"{pending_continuation_rate:.3f}<{float(args.min_pending_continuation_rate):.3f}"
        )
    if (
        args.max_active_family_drift_rate is not None
        and active_family_drift_rate > args.max_active_family_drift_rate
    ):
        failures.append(
            "active_family_drift_rate_above_threshold:"
            + f"{active_family_drift_rate:.3f}>{float(args.max_active_family_drift_rate):.3f}"
        )
    return failures


def main() -> int:
    args = _build_arg_parser().parse_args()
    try:
        cases = chat_boundary_eval.load_chat_boundary_eval_cases(args.gold)
    except Exception as exc:  # noqa: BLE001
        print(f"failed_to_load_chat_boundary_eval_cases:{exc}", file=sys.stderr)
        return 2

    report = chat_boundary_eval.evaluate_chat_boundary_cases(
        cases,
        predict=chat_boundary_eval.predict_chat_boundary_from_evidence,
    )
    _print_summary(report.get("summary", {}))
    if args.verbose:
        raw_cases = report.get("cases", [])
        if isinstance(raw_cases, list):
            _print_case_rows([entry for entry in raw_cases if isinstance(entry, Mapping)])

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"wrote report: {args.output}")

    failures = _check_thresholds(report, args)
    if failures:
        print("threshold check failed:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
