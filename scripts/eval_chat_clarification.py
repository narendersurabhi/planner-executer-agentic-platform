#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

from libs.core import chat_clarification_eval


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate chat clarification mapping behavior against a gold set."
    )
    parser.add_argument(
        "--gold",
        type=Path,
        default=Path("eval/chat_clarification_mapping_gold.yaml"),
        help="Path to chat clarification gold cases (YAML).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/evals/chat_clarification_report.json"),
        help="Path to write JSON report.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print per-case rows.")
    parser.add_argument("--min-overall-accuracy", type=float, default=None)
    parser.add_argument("--min-restart-decision-accuracy", type=float, default=None)
    parser.add_argument("--min-resolved-active-field-accuracy", type=float, default=None)
    parser.add_argument("--min-queue-advance-accuracy", type=float, default=None)
    parser.add_argument("--max-wrong-field-assignment-rate", type=float, default=None)
    return parser


def _print_summary(summary: Mapping[str, Any]) -> None:
    print("Chat Clarification Eval Summary")
    print(f"  cases: {int(summary.get('case_count', 0))}")
    print(f"  overall_accuracy: {float(summary.get('overall_accuracy', 0.0)):.3f}")
    print(
        "  restart_decision_accuracy: "
        f"{float(summary.get('restart_decision_accuracy', 0.0)):.3f}"
    )
    print(
        "  resolved_active_field_accuracy: "
        f"{float(summary.get('resolved_active_field_accuracy', 0.0)):.3f}"
    )
    print(
        f"  queue_advance_accuracy: {float(summary.get('queue_advance_accuracy', 0.0)):.3f}"
    )
    print(
        "  wrong_field_assignment_rate: "
        f"{float(summary.get('wrong_field_assignment_rate', 0.0)):.3f}"
    )


def _print_case_rows(case_results: list[Mapping[str, Any]]) -> None:
    print("\nPer-case")
    for case in case_results:
        print(
            "  "
            + f"{case.get('case_id', 'unknown')}: "
            + f"restart={case.get('predicted_restarted')} "
            + f"resolved_active={case.get('predicted_resolved_active_field')} "
            + f"queue_advanced={case.get('predicted_queue_advanced')} "
            + f"overall_match={case.get('overall_match')}"
        )


def _check_thresholds(report: Mapping[str, Any], args: argparse.Namespace) -> list[str]:
    summary = report.get("summary", {})
    failures: list[str] = []
    overall_accuracy = float(summary.get("overall_accuracy", 0.0))
    restart_decision_accuracy = float(summary.get("restart_decision_accuracy", 0.0))
    resolved_active_field_accuracy = float(
        summary.get("resolved_active_field_accuracy", 0.0)
    )
    queue_advance_accuracy = float(summary.get("queue_advance_accuracy", 0.0))
    wrong_field_assignment_rate = float(summary.get("wrong_field_assignment_rate", 0.0))
    if args.min_overall_accuracy is not None and overall_accuracy < args.min_overall_accuracy:
        failures.append(
            f"overall_accuracy_below_threshold:{overall_accuracy:.3f}"
            f"<{float(args.min_overall_accuracy):.3f}"
        )
    if (
        args.min_restart_decision_accuracy is not None
        and restart_decision_accuracy < args.min_restart_decision_accuracy
    ):
        failures.append(
            "restart_decision_accuracy_below_threshold:"
            + f"{restart_decision_accuracy:.3f}"
            + f"<{float(args.min_restart_decision_accuracy):.3f}"
        )
    if (
        args.min_resolved_active_field_accuracy is not None
        and resolved_active_field_accuracy < args.min_resolved_active_field_accuracy
    ):
        failures.append(
            "resolved_active_field_accuracy_below_threshold:"
            + f"{resolved_active_field_accuracy:.3f}"
            + f"<{float(args.min_resolved_active_field_accuracy):.3f}"
        )
    if (
        args.min_queue_advance_accuracy is not None
        and queue_advance_accuracy < args.min_queue_advance_accuracy
    ):
        failures.append(
            "queue_advance_accuracy_below_threshold:"
            + f"{queue_advance_accuracy:.3f}"
            + f"<{float(args.min_queue_advance_accuracy):.3f}"
        )
    if (
        args.max_wrong_field_assignment_rate is not None
        and wrong_field_assignment_rate > args.max_wrong_field_assignment_rate
    ):
        failures.append(
            "wrong_field_assignment_rate_above_threshold:"
            + f"{wrong_field_assignment_rate:.3f}"
            + f">{float(args.max_wrong_field_assignment_rate):.3f}"
        )
    return failures


def main() -> int:
    args = _build_arg_parser().parse_args()
    try:
        cases = chat_clarification_eval.load_chat_clarification_eval_cases(args.gold)
    except Exception as exc:  # noqa: BLE001
        print(f"failed_to_load_chat_clarification_eval_cases:{exc}", file=sys.stderr)
        return 2

    report = chat_clarification_eval.evaluate_chat_clarification_cases(cases)
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
