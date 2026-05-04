#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import yaml


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run live chat-boundary regression cases against an API environment."
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("CHAT_BOUNDARY_LIVE_BASE_URL", "http://127.0.0.1:18000"),
        help="Base URL for the API environment under test.",
    )
    parser.add_argument(
        "--gold",
        type=Path,
        default=Path(
            os.getenv("CHAT_BOUNDARY_LIVE_GOLD", "eval/chat_boundary_live_regression.yaml")
        ),
        help="Path to live chat-boundary regression cases (YAML).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            os.getenv(
                "CHAT_BOUNDARY_LIVE_OUTPUT",
                "artifacts/evals/chat_boundary_live_report.json",
            )
        ),
        help="Path to write JSON report.",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=float(os.getenv("CHAT_BOUNDARY_LIVE_TIMEOUT_S", "30.0")),
        help="HTTP timeout in seconds.",
    )
    parser.add_argument(
        "--bearer-token",
        default=os.getenv("CHAT_BOUNDARY_LIVE_BEARER_TOKEN") or None,
        help="Optional bearer token for environments that require auth.",
    )
    parser.add_argument(
        "--min-pass-rate",
        type=float,
        default=float(os.getenv("CHAT_BOUNDARY_LIVE_MIN_PASS_RATE", "1.0")),
        help="Minimum pass rate required for success.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print per-case rows.")
    return parser


def _normalize_string(value: Any) -> str:
    return str(value or "").strip()


def _load_cases(path: Path) -> list[dict[str, Any]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, Mapping):
        raise ValueError("chat_boundary_live_regression_invalid_root")
    raw_cases = raw.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("chat_boundary_live_regression_missing_cases")
    cases: list[dict[str, Any]] = []
    for index, raw_case in enumerate(raw_cases, start=1):
        if not isinstance(raw_case, Mapping):
            continue
        case_id = _normalize_string(raw_case.get("id") or f"case_{index}")
        content = _normalize_string(raw_case.get("content"))
        if not case_id or not content:
            raise ValueError(f"chat_boundary_live_regression_invalid_case:index={index}")
        cases.append(
            {
                "id": case_id,
                "content": content,
                "context_json": dict(raw_case.get("context_json") or {}),
                "priority": int(raw_case.get("priority") or 0),
                "expected_action_types": [
                    _normalize_string(item)
                    for item in raw_case.get("expected_action_types") or []
                    if _normalize_string(item)
                ],
                "forbidden_action_types": [
                    _normalize_string(item)
                    for item in raw_case.get("forbidden_action_types") or []
                    if _normalize_string(item)
                ],
                "forbidden_intent_sources": [
                    _normalize_string(item)
                    for item in raw_case.get("forbidden_intent_sources") or []
                    if _normalize_string(item)
                ],
                "notes": _normalize_string(raw_case.get("notes")),
            }
        )
    if not cases:
        raise ValueError("chat_boundary_live_regression_empty")
    return cases


def _request_json(
    base_url: str,
    path: str,
    *,
    method: str = "GET",
    payload: Mapping[str, Any] | None = None,
    timeout_s: float,
    bearer_token: str | None,
) -> Any:
    headers = {"Content-Type": "application/json"}
    if bearer_token:
        headers["Authorization"] = f"Bearer {bearer_token}"
    request = Request(
        f"{base_url.rstrip('/')}{path}",
        data=json.dumps(payload).encode("utf-8") if payload is not None else None,
        headers=headers,
        method=method,
    )
    with urlopen(request, timeout=timeout_s) as response:
        return json.loads(response.read().decode("utf-8"))


def _run_case(
    base_url: str,
    case: Mapping[str, Any],
    *,
    timeout_s: float,
    bearer_token: str | None,
) -> dict[str, Any]:
    session = _request_json(
        base_url,
        "/chat/sessions",
        method="POST",
        payload={"title": f"live-regression:{case['id']}"},
        timeout_s=timeout_s,
        bearer_token=bearer_token,
    )
    response = _request_json(
        base_url,
        f"/chat/sessions/{session['id']}/messages",
        method="POST",
        payload={
            "content": case["content"],
            "context_json": case["context_json"],
            "priority": case["priority"],
        },
        timeout_s=timeout_s,
        bearer_token=bearer_token,
    )
    assistant_message = response.get("assistant_message") or {}
    action = assistant_message.get("action") or {}
    metadata = assistant_message.get("metadata") or {}
    intent_profile = metadata.get("goal_intent_profile") or {}
    action_type = _normalize_string(action.get("type"))
    intent_source = _normalize_string(intent_profile.get("source"))
    expected_action_types = set(case.get("expected_action_types") or [])
    forbidden_action_types = set(case.get("forbidden_action_types") or [])
    forbidden_intent_sources = set(case.get("forbidden_intent_sources") or [])
    matches_expected = not expected_action_types or action_type in expected_action_types
    violates_forbidden_action = bool(action_type and action_type in forbidden_action_types)
    violates_forbidden_source = bool(intent_source and intent_source in forbidden_intent_sources)
    passed = matches_expected and not violates_forbidden_action and not violates_forbidden_source
    return {
        "case_id": case["id"],
        "content": case["content"],
        "action_type": action_type,
        "intent_source": intent_source,
        "passed": passed,
        "expected_action_types": sorted(expected_action_types),
        "forbidden_action_types": sorted(forbidden_action_types),
        "forbidden_intent_sources": sorted(forbidden_intent_sources),
        "job_id": action.get("job_id"),
        "workflow_run_id": action.get("workflow_run_id"),
        "notes": case.get("notes") or None,
    }


def _run_cases(
    base_url: str,
    cases: list[dict[str, Any]],
    *,
    timeout_s: float,
    bearer_token: str | None,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for case in cases:
        try:
            results.append(
                _run_case(
                    base_url,
                    case,
                    timeout_s=timeout_s,
                    bearer_token=bearer_token,
                )
            )
        except (HTTPError, URLError, TimeoutError, ValueError) as exc:
            results.append(
                {
                    "case_id": case["id"],
                    "content": case["content"],
                    "action_type": "",
                    "intent_source": "",
                    "passed": False,
                    "error": str(exc),
                    "expected_action_types": case.get("expected_action_types") or [],
                    "forbidden_action_types": case.get("forbidden_action_types") or [],
                    "forbidden_intent_sources": case.get("forbidden_intent_sources") or [],
                    "notes": case.get("notes") or None,
                }
            )
    passed_count = sum(1 for result in results if bool(result.get("passed")))
    case_count = len(results)
    return {
        "summary": {
            "case_count": case_count,
            "passed_count": passed_count,
            "pass_rate": (passed_count / case_count) if case_count else 0.0,
        },
        "cases": results,
    }


def _print_summary(summary: Mapping[str, Any]) -> None:
    print("Chat Boundary Live Regression Summary")
    print(f"  cases: {int(summary.get('case_count', 0))}")
    print(f"  passed: {int(summary.get('passed_count', 0))}")
    print(f"  pass_rate: {float(summary.get('pass_rate', 0.0)):.3f}")


def _print_case_rows(case_results: list[Mapping[str, Any]]) -> None:
    print("\nPer-case")
    for case in case_results:
        print(
            "  "
            + f"{case.get('case_id', 'unknown')}: "
            + f"passed={bool(case.get('passed'))} "
            + f"action={case.get('action_type') or '-'} "
            + f"intent_source={case.get('intent_source') or '-'}"
        )


def main() -> int:
    args = _build_arg_parser().parse_args()
    try:
        cases = _load_cases(args.gold)
    except Exception as exc:  # noqa: BLE001
        print(f"failed_to_load_chat_boundary_live_regression_cases:{exc}", file=sys.stderr)
        return 2

    report = _run_cases(
        args.base_url,
        cases,
        timeout_s=args.timeout_s,
        bearer_token=args.bearer_token,
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

    if float((report.get("summary") or {}).get("pass_rate", 0.0)) < float(args.min_pass_rate):
        print(
            "chat_boundary_live_regression_threshold_failed:"
            f"pass_rate={float((report.get('summary') or {}).get('pass_rate', 0.0)):.3f}:"
            f"min_pass_rate={float(args.min_pass_rate):.3f}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
