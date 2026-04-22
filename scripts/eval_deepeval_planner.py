#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping

from libs.core.deepeval_agents import build_planner_gold_bundle
from libs.core.deepeval_support import (
    load_settings_from_env,
    run_deepeval_cases,
    write_report,
)
from scripts.deepeval_eval_helpers import build_heuristic_goal_normalizer


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run DeepEval-backed planner replay from intent normalization gold cases."
    )
    parser.add_argument(
        "--normalization-gold",
        type=Path,
        default=Path("eval/intent_normalization_gold.yaml"),
        help="Path to planner normalization gold cases.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/evals/deepeval_planner_report.json"),
        help="Path to write the normalized DeepEval report.",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=None,
        help="Minimum overall score required for success.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print failing case ids.")
    return parser


def _print_summary(report: Mapping[str, Any]) -> None:
    print("DeepEval Planner Summary")
    print(f"  cases: {int(report.get('case_count', 0))}")
    print(f"  pass_rate: {float(report.get('pass_rate', 0.0)):.3f}")
    print(f"  overall_score: {float(report.get('overall_score', 0.0)):.3f}")
    threshold = report.get("threshold")
    if threshold is not None:
        print(f"  threshold: {float(threshold):.3f}")
        print(f"  threshold_passed: {bool(report.get('threshold_passed'))}")
    deepeval_meta = report.get("deepeval", {})
    print(f"  judge_used: {bool(deepeval_meta.get('judge_used'))}")


def main() -> int:
    args = _build_arg_parser().parse_args()
    try:
        settings = load_settings_from_env()
        bundle = build_planner_gold_bundle(
            normalization_gold=args.normalization_gold,
            normalize_goal_intent=build_heuristic_goal_normalizer(),
        )
        report = run_deepeval_cases(
            surface=bundle.surface,
            dataset_id=bundle.dataset_id,
            cases=bundle.cases,
            metric_specs=bundle.metric_specs,
            settings=settings,
            threshold=(
                float(args.min_score)
                if args.min_score is not None
                else float(settings.min_planner_score)
            ),
            deterministic_summary=bundle.deterministic_summary,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"deepeval_planner_eval_failed:{exc}", file=sys.stderr)
        return 2

    report = dict(report)
    report["report_path"] = str(args.output)
    write_report(args.output, report)
    _print_summary(report)
    if args.verbose:
        failing_case_ids = list(report.get("failing_case_ids") or [])
        if failing_case_ids:
            print("  failing_case_ids:")
            for case_id in failing_case_ids:
                print(f"    - {case_id}")
    return 0 if bool(report.get("threshold_passed")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
