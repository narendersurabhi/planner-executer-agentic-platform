#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

from libs.core import capability_registry, intent_contract
from libs.core.intent_eval import evaluate_intent_cases, load_intent_eval_cases


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate intent decomposition against a gold set.")
    parser.add_argument(
        "--gold",
        type=Path,
        default=Path("eval/intent_gold.yaml"),
        help="Path to gold intent evaluation cases (YAML).",
    )
    parser.add_argument(
        "--mode",
        choices=("heuristic", "runtime"),
        default="heuristic",
        help="Use deterministic heuristic decomposition or runtime API decomposition.",
    )
    parser.add_argument(
        "--runtime-intent-mode",
        choices=("heuristic", "llm", "hybrid"),
        default=None,
        help="Override runtime INTENT_DECOMPOSE_MODE when --mode=runtime.",
    )
    parser.add_argument("--top-k", type=int, default=3, help="Top-k for segment capability hit-rate.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write full JSON report.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-case metrics.",
    )
    parser.add_argument("--min-intent-f1", type=float, default=None)
    parser.add_argument("--min-capability-f1", type=float, default=None)
    parser.add_argument("--min-segment-hit-rate", type=float, default=None)
    return parser


def _build_decomposer(
    mode: str,
    runtime_intent_mode: str | None,
) -> tuple[str, Any]:
    if mode == "heuristic":
        return "heuristic", intent_contract.decompose_goal_intent

    from services.api.app import main as api_main

    if runtime_intent_mode:
        api_main.INTENT_DECOMPOSE_MODE = runtime_intent_mode
        if runtime_intent_mode == "heuristic":
            api_main._intent_decompose_provider = None
    return "runtime", api_main._decompose_goal_intent


def _print_summary(summary: Mapping[str, Any]) -> None:
    intent = summary.get("intent_micro", {})
    capability = summary.get("capability_micro", {})
    segment = summary.get("segment_topk", {})
    invalid = summary.get("invalid_capabilities", {})
    ordered = summary.get("intent_ordered_proxy", {})
    print("Intent Eval Summary")
    print(f"  cases: {summary.get('case_count', 0)}")
    print(
        "  intent micro: "
        f"p={float(intent.get('precision', 0.0)):.3f} "
        f"r={float(intent.get('recall', 0.0)):.3f} "
        f"f1={float(intent.get('f1', 0.0)):.3f}"
    )
    print(
        "  capability micro: "
        f"p={float(capability.get('precision', 0.0)):.3f} "
        f"r={float(capability.get('recall', 0.0)):.3f} "
        f"f1={float(capability.get('f1', 0.0)):.3f}"
    )
    print(
        "  segment top-k: "
        f"hit_rate={float(segment.get('hit_rate', 0.0)):.3f} "
        f"(hits={int(segment.get('hits', 0))}/{int(segment.get('slots', 0))}, "
        f"k={int(segment.get('top_k', 0))})"
    )
    print(
        "  invalid capability ids: "
        f"{int(invalid.get('count', 0))} "
        f"(rate={float(invalid.get('rate', 0.0)):.3f})"
    )
    print(
        "  intent ordered proxy: "
        f"hit_rate={float(ordered.get('hit_rate', 0.0)):.3f} "
        f"(hits={int(ordered.get('hits', 0))}/{int(ordered.get('total', 0))})"
    )


def _print_case_rows(case_results: list[Mapping[str, Any]]) -> None:
    print("\nPer-case")
    for case in case_results:
        case_id = str(case.get("case_id") or "unknown")
        intent = case.get("intent", {})
        capability = case.get("capabilities", {})
        segment = case.get("segment_topk", {})
        invalid = capability.get("invalid", [])
        invalid_suffix = ""
        if isinstance(invalid, list) and invalid:
            invalid_suffix = f" invalid={','.join(str(item) for item in invalid)}"
        print(
            "  "
            + f"{case_id}: "
            + f"intent_f1={float(intent.get('f1', 0.0)):.3f} "
            + f"cap_f1={float(capability.get('f1', 0.0)):.3f} "
            + f"topk_hit={float(segment.get('hit_rate', 0.0)):.3f}"
            + invalid_suffix
        )


def _check_thresholds(report: Mapping[str, Any], args: argparse.Namespace) -> list[str]:
    failures: list[str] = []
    summary = report.get("summary", {})
    intent_f1 = float(summary.get("intent_micro", {}).get("f1", 0.0))
    capability_f1 = float(summary.get("capability_micro", {}).get("f1", 0.0))
    segment_hit_rate = float(summary.get("segment_topk", {}).get("hit_rate", 0.0))
    if args.min_intent_f1 is not None and intent_f1 < args.min_intent_f1:
        failures.append(
            f"intent_f1_below_threshold:{intent_f1:.3f}<{float(args.min_intent_f1):.3f}"
        )
    if args.min_capability_f1 is not None and capability_f1 < args.min_capability_f1:
        failures.append(
            f"capability_f1_below_threshold:{capability_f1:.3f}<{float(args.min_capability_f1):.3f}"
        )
    if args.min_segment_hit_rate is not None and segment_hit_rate < args.min_segment_hit_rate:
        failures.append(
            "segment_hit_rate_below_threshold:"
            + f"{segment_hit_rate:.3f}<{float(args.min_segment_hit_rate):.3f}"
        )
    return failures


def main() -> int:
    args = _build_arg_parser().parse_args()
    try:
        cases = load_intent_eval_cases(args.gold)
    except Exception as exc:  # noqa: BLE001
        print(f"failed_to_load_gold_cases:{exc}", file=sys.stderr)
        return 2

    _, decompose_goal = _build_decomposer(args.mode, args.runtime_intent_mode)
    allowed_capability_ids = set(capability_registry.load_capability_registry().enabled_capabilities())
    report = evaluate_intent_cases(
        cases,
        decompose_goal=decompose_goal,
        top_k=max(1, int(args.top_k)),
        allowed_capability_ids=allowed_capability_ids,
    )

    _print_summary(report.get("summary", {}))
    if args.verbose:
        case_results = report.get("cases", [])
        if isinstance(case_results, list):
            _print_case_rows([row for row in case_results if isinstance(row, Mapping)])

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
