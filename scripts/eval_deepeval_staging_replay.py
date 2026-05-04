#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from libs.core.deepeval_agents import (
    build_chat_feedback_bundle,
    build_planner_feedback_bundle,
)
from libs.core.deepeval_support import (
    build_gate_report,
    load_settings_from_env,
    run_deepeval_cases,
    write_report,
)
from scripts.deepeval_eval_helpers import request_feedback_examples


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay staging chat and planner feedback slices through the DeepEval harness."
    )
    parser.add_argument(
        "--base-url",
        default=(
            os.getenv("DEEPEVAL_STAGING_BASE_URL")
            or os.getenv("CHAT_BOUNDARY_LIVE_BASE_URL")
            or "http://127.0.0.1:18000"
        ),
        help="Base URL for the staging API.",
    )
    parser.add_argument(
        "--bearer-token",
        default=(
            os.getenv("DEEPEVAL_STAGING_BEARER_TOKEN")
            or os.getenv("CHAT_BOUNDARY_LIVE_BEARER_TOKEN")
            or None
        ),
        help="Optional bearer token for the staging API.",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=float(os.getenv("DEEPEVAL_STAGING_TIMEOUT_S", "30.0")),
        help="HTTP timeout in seconds.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=int(os.getenv("DEEPEVAL_STAGING_LIMIT", "100")),
        help="Maximum number of feedback examples to fetch per target type.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/evals/deepeval_staging_replay_report.json"),
        help="Path to write the combined replay report.",
    )
    return parser


def main() -> int:
    args = _build_arg_parser().parse_args()
    try:
        settings = load_settings_from_env()
        chat_examples = request_feedback_examples(
            base_url=args.base_url,
            bearer_token=args.bearer_token,
            timeout_s=max(1.0, float(args.timeout_s)),
            target_type="chat_message",
            limit=max(1, int(args.limit)),
        )
        planner_examples = request_feedback_examples(
            base_url=args.base_url,
            bearer_token=args.bearer_token,
            timeout_s=max(1.0, float(args.timeout_s)),
            target_type="plan",
            limit=max(1, int(args.limit)),
        )
        chat_bundle = build_chat_feedback_bundle(
            examples=chat_examples,
            dataset_id="staging_feedback_chat_message",
        )
        planner_bundle = build_planner_feedback_bundle(
            examples=planner_examples,
            dataset_id="staging_feedback_plan",
        )
        chat_report = run_deepeval_cases(
            surface=chat_bundle.surface,
            dataset_id=chat_bundle.dataset_id,
            cases=chat_bundle.cases,
            metric_specs=chat_bundle.metric_specs,
            settings=settings,
            threshold=float(settings.min_chat_score) if chat_bundle.cases else None,
            deterministic_summary=chat_bundle.deterministic_summary,
        )
        planner_report = run_deepeval_cases(
            surface=planner_bundle.surface,
            dataset_id=planner_bundle.dataset_id,
            cases=planner_bundle.cases,
            metric_specs=planner_bundle.metric_specs,
            settings=settings,
            threshold=float(settings.min_planner_score) if planner_bundle.cases else None,
            deterministic_summary=planner_bundle.deterministic_summary,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"deepeval_staging_replay_failed:{exc}", file=sys.stderr)
        return 2

    gate_report = build_gate_report(chat_report=chat_report, planner_report=planner_report)
    gate_report["base_url"] = args.base_url
    gate_report["fetched_examples"] = {
        "chat_message": len(chat_examples),
        "plan": len(planner_examples),
    }
    write_report(args.output, gate_report)

    print("DeepEval Staging Replay Summary")
    print(
        f"  chat_examples={len(chat_examples)} "
        f"chat_score={float(chat_report.get('overall_score', 0.0)):.3f}"
    )
    print(
        f"  planner_examples={len(planner_examples)} "
        f"planner_score={float(planner_report.get('overall_score', 0.0)):.3f}"
    )
    print(f"  overall_passed={bool(gate_report.get('threshold_passed'))}")
    return 0 if bool(gate_report.get("threshold_passed")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
