#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

from libs.core import capability_registry, intent_contract, workflow_contracts
from libs.core.intent_eval import evaluate_intent_cases, load_intent_eval_cases
from services.api.app import intent_service


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate intent decomposition against a gold set.")
    parser.add_argument(
        "--gold",
        type=Path,
        default=Path("eval/intent_gold.yaml"),
        help="Path to gold intent evaluation cases (YAML).",
    )
    parser.add_argument(
        "--normalization-gold",
        type=Path,
        default=Path("eval/intent_normalization_gold.yaml"),
        help="Optional path to normalization gold cases (YAML).",
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
    parser.add_argument("--min-normalization-intent-accuracy", type=float, default=None)
    parser.add_argument("--min-normalization-capability-alignment", type=float, default=None)
    parser.add_argument("--min-missing-input-precision", type=float, default=None)
    parser.add_argument("--min-disagreement-clarification-rate", type=float, default=None)
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


def _load_required_inputs_lookup() -> Any:
    registry = capability_registry.load_capability_registry()
    schema_dir = Path("schemas")
    required_inputs_by_capability: dict[str, list[str]] = {}
    for capability_id, spec in registry.enabled_capabilities().items():
        required_inputs: list[str] = []
        if spec.input_schema_ref:
            schema_path = schema_dir / f"{spec.input_schema_ref}.json"
            if schema_path.exists():
                try:
                    schema = json.loads(schema_path.read_text(encoding="utf-8"))
                except Exception:  # noqa: BLE001
                    schema = {}
                raw_required = schema.get("required") if isinstance(schema, Mapping) else None
                if isinstance(raw_required, list):
                    required_inputs = [entry for entry in raw_required if isinstance(entry, str)]
        required_inputs_by_capability[capability_id] = required_inputs

    def _lookup(capability_id: str) -> list[str]:
        canonical = registry.canonicalize_id(capability_id) or str(capability_id or "").strip()
        return list(required_inputs_by_capability.get(canonical, ()))

    return _lookup


def _goal_intent_assess_config() -> intent_service.GoalIntentConfig:
    return intent_service.GoalIntentConfig(
        min_confidence=0.70,
        min_confidence_by_intent={},
        min_confidence_by_risk={},
        clarification_blocking_slots={
            "intent_action",
            "output_format",
            "target_system",
            "safety_constraints",
        },
    )


def _build_normalizer(
    mode: str,
    runtime_intent_mode: str | None,
) -> tuple[str, Any]:
    required_inputs_lookup = _load_required_inputs_lookup()
    if mode == "heuristic":
        def _assess(goal: str) -> workflow_contracts.GoalIntentProfile:
            return intent_service.assess_goal_intent(
                goal,
                config=_goal_intent_assess_config(),
                runtime=intent_service.GoalIntentRuntime(
                    infer_task_intent=intent_contract.infer_task_intent_from_goal_with_metadata,
                    record_metrics=None,
                ),
            )

        def _decompose(goal: str, **_: Any) -> workflow_contracts.IntentGraph:
            return workflow_contracts.IntentGraph.model_validate(
                intent_contract.decompose_goal_intent(goal)
            )

        def _normalize(goal: str, intent_context: Mapping[str, Any] | None) -> dict[str, Any]:
            envelope = intent_service.normalize_goal_intent(
                goal,
                intent_context=dict(intent_context or {}),
                config=intent_service.IntentNormalizeConfig(
                    include_decomposition=True,
                    assessment_mode="heuristic",
                    assessment_model="",
                    decomposition_mode="heuristic",
                    decomposition_model="",
                ),
                runtime=intent_service.IntentNormalizeRuntime(
                    assess_goal_intent=_assess,
                    assess_goal_intent_heuristic=_assess,
                    decompose_goal_intent=_decompose,
                    capability_required_inputs=required_inputs_lookup,
                ),
            )
            return envelope.model_dump(mode="json", exclude_none=True)

        return "heuristic", _normalize

    from services.api.app import main as api_main

    if runtime_intent_mode:
        api_main.INTENT_DECOMPOSE_MODE = runtime_intent_mode
        api_main.INTENT_ASSESS_MODE = runtime_intent_mode
        if runtime_intent_mode == "heuristic":
            api_main._intent_decompose_provider = None
            api_main._intent_provider = None

    def _normalize_runtime(goal: str, intent_context: Mapping[str, Any] | None) -> dict[str, Any]:
        envelope = intent_service.normalize_goal_intent(
            goal,
            intent_context=dict(intent_context or {}),
            config=intent_service.IntentNormalizeConfig(
                include_decomposition=True,
                assessment_mode=api_main.INTENT_ASSESS_MODE if api_main.INTENT_ASSESS_ENABLED else "disabled",
                assessment_model=(
                    (api_main.INTENT_ASSESS_MODEL or api_main.INTENT_DECOMPOSE_MODEL or api_main.LLM_MODEL_NAME or "").strip()
                    if api_main.INTENT_ASSESS_ENABLED and api_main.INTENT_ASSESS_MODE not in {"disabled", "heuristic"}
                    else ""
                ),
                decomposition_mode=api_main.INTENT_DECOMPOSE_MODE if api_main.INTENT_DECOMPOSE_ENABLED else "disabled",
                decomposition_model=(
                    (api_main.INTENT_DECOMPOSE_MODEL or api_main.LLM_MODEL_NAME or "").strip()
                    if api_main.INTENT_DECOMPOSE_ENABLED
                    else ""
                ),
            ),
            runtime=intent_service.IntentNormalizeRuntime(
                assess_goal_intent=lambda goal_text: api_main._assess_goal_intent(
                    goal_text,
                    mode_override=api_main.INTENT_ASSESS_MODE if api_main.INTENT_ASSESS_ENABLED else "disabled",
                ),
                assess_goal_intent_heuristic=lambda goal_text: api_main._assess_goal_intent(
                    goal_text,
                    mode_override="heuristic",
                ),
                decompose_goal_intent=api_main._decompose_goal_intent,
                capability_required_inputs=required_inputs_lookup,
            ),
        )
        return envelope.model_dump(mode="json", exclude_none=True)

    return "runtime", _normalize_runtime


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
    normalization = summary.get("normalization", {})
    if isinstance(normalization, Mapping) and normalization:
        missing_inputs = normalization.get("missing_inputs", {})
        print("  normalization:")
        print(
            "    profile intent accuracy: "
            f"{float(normalization.get('profile_intent_accuracy', 0.0)):.3f} "
            f"(hits={int(normalization.get('profile_intent_hits', 0))}/"
            f"{int(normalization.get('profile_intent_total', 0))})"
        )
        print(
            "    capability alignment: "
            f"{float(normalization.get('capability_alignment_rate', 0.0)):.3f} "
            f"(hits={int(normalization.get('capability_alignment_hits', 0))}/"
            f"{int(normalization.get('capability_alignment_total', 0))})"
        )
        print(
            "    missing inputs: "
            f"p={float(missing_inputs.get('precision', 0.0)):.3f} "
            f"r={float(missing_inputs.get('recall', 0.0)):.3f} "
            f"f1={float(missing_inputs.get('f1', 0.0)):.3f}"
        )
        print(
            "    disagreement clarification rate: "
            f"{float(normalization.get('disagreement_clarification_rate', 0.0)):.3f} "
            f"(hits={int(normalization.get('disagreement_clarification_hits', 0))}/"
            f"{int(normalization.get('disagreement_clarification_total', 0))})"
        )


def _print_case_rows(case_results: list[Mapping[str, Any]]) -> None:
    print("\nPer-case")
    for case in case_results:
        case_id = str(case.get("case_id") or "unknown")
        intent = case.get("intent", {})
        capability = case.get("capabilities", {})
        segment = case.get("segment_topk", {})
        normalization = case.get("normalization", {})
        invalid = capability.get("invalid", [])
        invalid_suffix = ""
        if isinstance(invalid, list) and invalid:
            invalid_suffix = f" invalid={','.join(str(item) for item in invalid)}"
        normalization_suffix = ""
        if isinstance(normalization, Mapping) and normalization:
            profile_intent = normalization.get("profile_intent", {})
            missing_inputs = normalization.get("missing_inputs", {})
            actual_profile_intent = str(profile_intent.get("actual") or "").strip()
            normalization_suffix = (
                f" profile_intent={actual_profile_intent or '-'}"
                f" missing_precision={float(missing_inputs.get('precision', 0.0)):.3f}"
            )
        print(
            "  "
            + f"{case_id}: "
            + f"intent_f1={float(intent.get('f1', 0.0)):.3f} "
            + f"cap_f1={float(capability.get('f1', 0.0)):.3f} "
            + f"topk_hit={float(segment.get('hit_rate', 0.0)):.3f}"
            + normalization_suffix
            + invalid_suffix
        )


def _check_thresholds(report: Mapping[str, Any], args: argparse.Namespace) -> list[str]:
    failures: list[str] = []
    summary = report.get("summary", {})
    intent_f1 = float(summary.get("intent_micro", {}).get("f1", 0.0))
    capability_f1 = float(summary.get("capability_micro", {}).get("f1", 0.0))
    segment_hit_rate = float(summary.get("segment_topk", {}).get("hit_rate", 0.0))
    normalization = summary.get("normalization", {})
    normalization_intent_accuracy = float(
        normalization.get("profile_intent_accuracy", 0.0)
    ) if isinstance(normalization, Mapping) else 0.0
    normalization_capability_alignment = float(
        normalization.get("capability_alignment_rate", 0.0)
    ) if isinstance(normalization, Mapping) else 0.0
    missing_input_precision = float(
        normalization.get("missing_inputs", {}).get("precision", 0.0)
    ) if isinstance(normalization, Mapping) else 0.0
    disagreement_clarification_rate = float(
        normalization.get("disagreement_clarification_rate", 0.0)
    ) if isinstance(normalization, Mapping) else 0.0
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
    if (
        args.min_normalization_intent_accuracy is not None
        and normalization_intent_accuracy < args.min_normalization_intent_accuracy
    ):
        failures.append(
            "normalization_intent_accuracy_below_threshold:"
            + f"{normalization_intent_accuracy:.3f}<"
            + f"{float(args.min_normalization_intent_accuracy):.3f}"
        )
    if (
        args.min_normalization_capability_alignment is not None
        and normalization_capability_alignment < args.min_normalization_capability_alignment
    ):
        failures.append(
            "normalization_capability_alignment_below_threshold:"
            + f"{normalization_capability_alignment:.3f}<"
            + f"{float(args.min_normalization_capability_alignment):.3f}"
        )
    if (
        args.min_missing_input_precision is not None
        and missing_input_precision < args.min_missing_input_precision
    ):
        failures.append(
            "missing_input_precision_below_threshold:"
            + f"{missing_input_precision:.3f}<"
            + f"{float(args.min_missing_input_precision):.3f}"
        )
    if (
        args.min_disagreement_clarification_rate is not None
        and disagreement_clarification_rate < args.min_disagreement_clarification_rate
    ):
        failures.append(
            "disagreement_clarification_rate_below_threshold:"
            + f"{disagreement_clarification_rate:.3f}<"
            + f"{float(args.min_disagreement_clarification_rate):.3f}"
        )
    return failures


def main() -> int:
    args = _build_arg_parser().parse_args()
    try:
        cases = load_intent_eval_cases(args.gold)
    except Exception as exc:  # noqa: BLE001
        print(f"failed_to_load_gold_cases:{exc}", file=sys.stderr)
        return 2
    try:
        normalization_cases = load_intent_eval_cases(args.normalization_gold)
    except Exception as exc:  # noqa: BLE001
        print(f"failed_to_load_normalization_gold_cases:{exc}", file=sys.stderr)
        return 2

    _, decompose_goal = _build_decomposer(args.mode, args.runtime_intent_mode)
    _, normalize_goal_intent = _build_normalizer(args.mode, args.runtime_intent_mode)
    allowed_capability_ids = set(capability_registry.load_capability_registry().enabled_capabilities())
    report = evaluate_intent_cases(
        cases,
        decompose_goal=decompose_goal,
        top_k=max(1, int(args.top_k)),
        allowed_capability_ids=allowed_capability_ids,
    )
    normalization_report = evaluate_intent_cases(
        normalization_cases,
        decompose_goal=decompose_goal,
        normalize_goal_intent=normalize_goal_intent,
        top_k=max(1, int(args.top_k)),
        allowed_capability_ids=allowed_capability_ids,
    )
    if isinstance(normalization_report.get("summary"), Mapping):
        report["summary"]["normalization"] = normalization_report["summary"].get("normalization", {})
    if isinstance(normalization_report.get("cases"), list):
        report["normalization_cases"] = normalization_report["cases"]

    _print_summary(report.get("summary", {}))
    if args.verbose:
        case_results = report.get("cases", [])
        if isinstance(case_results, list):
            _print_case_rows([row for row in case_results if isinstance(row, Mapping)])
        normalization_case_results = report.get("normalization_cases", [])
        if isinstance(normalization_case_results, list) and normalization_case_results:
            print("\nNormalization Cases")
            _print_case_rows(
                [row for row in normalization_case_results if isinstance(row, Mapping)]
            )

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
