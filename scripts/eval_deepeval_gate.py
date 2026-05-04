#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from libs.core.deepeval_agents import build_chat_gold_bundle, build_planner_gold_bundle
from libs.core.deepeval_support import (
    build_gate_report,
    load_settings_from_env,
    run_deepeval_cases,
    write_report,
)
from scripts.deepeval_eval_helpers import build_heuristic_goal_normalizer


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the combined DeepEval gate for chat and planner."
    )
    parser.add_argument(
        "--chat-output",
        type=Path,
        default=Path("artifacts/evals/deepeval_chat_report.json"),
        help="Path to write the chat report.",
    )
    parser.add_argument(
        "--planner-output",
        type=Path,
        default=Path("artifacts/evals/deepeval_planner_report.json"),
        help="Path to write the planner report.",
    )
    parser.add_argument(
        "--gate-output",
        type=Path,
        default=Path("artifacts/evals/deepeval_gate_report.json"),
        help="Path to write the combined gate report.",
    )
    return parser


def main() -> int:
    args = _build_arg_parser().parse_args()
    try:
        settings = load_settings_from_env()
        chat_bundle = build_chat_gold_bundle(
            boundary_gold=Path("eval/chat_boundary_gold.yaml"),
            clarification_gold=Path("eval/chat_clarification_mapping_gold.yaml"),
        )
        planner_bundle = build_planner_gold_bundle(
            normalization_gold=Path("eval/intent_normalization_gold.yaml"),
            normalize_goal_intent=build_heuristic_goal_normalizer(),
        )
        chat_report = run_deepeval_cases(
            surface=chat_bundle.surface,
            dataset_id=chat_bundle.dataset_id,
            cases=chat_bundle.cases,
            metric_specs=chat_bundle.metric_specs,
            settings=settings,
            threshold=float(settings.min_chat_score),
            deterministic_summary=chat_bundle.deterministic_summary,
        )
        planner_report = run_deepeval_cases(
            surface=planner_bundle.surface,
            dataset_id=planner_bundle.dataset_id,
            cases=planner_bundle.cases,
            metric_specs=planner_bundle.metric_specs,
            settings=settings,
            threshold=float(settings.min_planner_score),
            deterministic_summary=planner_bundle.deterministic_summary,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"deepeval_gate_failed:{exc}", file=sys.stderr)
        return 2

    chat_report = dict(chat_report)
    planner_report = dict(planner_report)
    chat_report["report_path"] = str(args.chat_output)
    planner_report["report_path"] = str(args.planner_output)
    gate_report = build_gate_report(chat_report=chat_report, planner_report=planner_report)
    gate_report["report_path"] = str(args.gate_output)

    write_report(args.chat_output, chat_report)
    write_report(args.planner_output, planner_report)
    write_report(args.gate_output, gate_report)

    print("DeepEval Gate Summary")
    print(
        f"  chat_score={float(chat_report.get('overall_score', 0.0)):.3f} "
        f"chat_threshold_passed={bool(chat_report.get('threshold_passed'))}"
    )
    print(
        f"  planner_score={float(planner_report.get('overall_score', 0.0)):.3f} "
        f"planner_threshold_passed={bool(planner_report.get('threshold_passed'))}"
    )
    print(f"  overall_passed={bool(gate_report.get('threshold_passed'))}")
    return 0 if bool(gate_report.get("threshold_passed")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
