#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

from libs.core import capability_registry, capability_search, capability_search_eval


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate capability search against a gold set.")
    parser.add_argument(
        "--gold",
        type=Path,
        default=Path("eval/capability_search_gold.jsonl"),
        help="Path to capability search gold cases (JSONL).",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Top-k retrieval cutoff.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/evals/capability_search_report.json"),
        help="Path to write JSON report.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print per-case rows.")
    parser.add_argument("--min-hit-rate-at-3", type=float, default=None)
    parser.add_argument("--min-mrr", type=float, default=None)
    parser.add_argument("--min-ndcg", type=float, default=None)
    return parser


def _print_summary(summary: Mapping[str, Any]) -> None:
    print("Capability Search Eval Summary")
    print(f"  cases: {int(summary.get('case_count', 0))}")
    print(f"  hit_rate@1: {float(summary.get('hit_rate_at_1', 0.0)):.3f}")
    print(f"  hit_rate@3: {float(summary.get('hit_rate_at_3', 0.0)):.3f}")
    print(f"  recall@5: {float(summary.get('recall_at_5', 0.0)):.3f}")
    print(f"  mrr: {float(summary.get('mrr', 0.0)):.3f}")
    print(f"  ndcg: {float(summary.get('ndcg', 0.0)):.3f}")
    print(f"  must_have_hit_rate: {float(summary.get('must_have_hit_rate', 0.0)):.3f}")


def _print_case_rows(case_results: list[Mapping[str, Any]]) -> None:
    print("\nPer-case")
    for case in case_results:
        metrics = case.get("metrics", {})
        retrieved = case.get("retrieved", [])
        top_ids = [str(entry.get("id") or "") for entry in retrieved[:3] if isinstance(entry, Mapping)]
        print(
            "  "
            + f"{case.get('case_id', 'unknown')}: "
            + f"hit@3={float(metrics.get('hit_rate_at_3', 0.0)):.3f} "
            + f"mrr={float(metrics.get('mrr', 0.0)):.3f} "
            + f"ndcg={float(metrics.get('ndcg', 0.0)):.3f} "
            + f"top3={','.join(top_ids)}"
        )


def _check_thresholds(report: Mapping[str, Any], args: argparse.Namespace) -> list[str]:
    summary = report.get("summary", {})
    failures: list[str] = []
    hit_rate_at_3 = float(summary.get("hit_rate_at_3", 0.0))
    mrr = float(summary.get("mrr", 0.0))
    ndcg = float(summary.get("ndcg", 0.0))
    if args.min_hit_rate_at_3 is not None and hit_rate_at_3 < args.min_hit_rate_at_3:
        failures.append(
            f"hit_rate_at_3_below_threshold:{hit_rate_at_3:.3f}<{float(args.min_hit_rate_at_3):.3f}"
        )
    if args.min_mrr is not None and mrr < args.min_mrr:
        failures.append(f"mrr_below_threshold:{mrr:.3f}<{float(args.min_mrr):.3f}")
    if args.min_ndcg is not None and ndcg < args.min_ndcg:
        failures.append(f"ndcg_below_threshold:{ndcg:.3f}<{float(args.min_ndcg):.3f}")
    return failures


def main() -> int:
    args = _build_arg_parser().parse_args()
    try:
        cases = capability_search_eval.load_capability_search_eval_cases(args.gold)
        registry = capability_registry.load_capability_registry()
        entries = capability_search.build_capability_search_entries(registry.enabled_capabilities())
    except Exception as exc:  # noqa: BLE001
        print(f"failed_to_prepare_capability_search_eval:{exc}", file=sys.stderr)
        return 2

    report = capability_search_eval.evaluate_capability_search_cases(
        cases,
        capability_entries=entries,
        limit=max(1, int(args.top_k)),
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
