from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import json
import os
import time
from typing import Any, Mapping, Sequence

from . import llm_provider

DEEPEVAL_REPORT_SCHEMA_VERSION = "deepeval_eval_report_v1"
DEEPEVAL_GATE_SCHEMA_VERSION = "deepeval_gate_report_v1"
_VALID_MODES = {"off", "local", "upload"}
_JUDGE_PROVIDERS_REQUIRING_MODEL = {"openai", "gemini", "openai_compatible"}


def _normalize_string(value: Any) -> str:
    return str(value or "").strip()


def _read_bool_env(name: str, default: bool) -> bool:
    raw = _normalize_string(os.getenv(name))
    if not raw:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def _read_float_env(name: str, default: float) -> float:
    raw = _normalize_string(os.getenv(name))
    if not raw:
        return default
    return float(raw)


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _safe_mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _score_value(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return None


def dumps_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


@dataclass(frozen=True)
class DeepEvalSettings:
    enabled: bool = False
    mode: str = "local"
    judge_provider: str = "mock"
    judge_model: str | None = None
    judge_api_key: str | None = None
    judge_base_url: str | None = None
    upload_results: bool = False
    experiment_name: str = "awe-agent-release-gate"
    min_chat_score: float = 0.95
    min_planner_score: float = 0.90


@dataclass(frozen=True)
class DeepEvalMetricSpec:
    name: str
    criteria: str
    threshold: float = 0.5


@dataclass(frozen=True)
class DeepEvalCase:
    case_id: str
    surface: str
    source_dataset_id: str
    input_text: str
    actual_output: str
    expected_output: str
    context: tuple[str, ...] = ()
    local_scores: Mapping[str, Any] = field(default_factory=dict)
    local_pass: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)


def load_settings_from_env() -> DeepEvalSettings:
    mode = _normalize_string(os.getenv("DEEPEVAL_MODE", "local")).lower() or "local"
    if mode not in _VALID_MODES:
        raise ValueError(
            "deepeval_mode_invalid:"
            f" expected one of {sorted(_VALID_MODES)} got={mode or '<empty>'}"
        )
    return DeepEvalSettings(
        enabled=_read_bool_env("DEEPEVAL_ENABLED", False),
        mode=mode,
        judge_provider=_normalize_string(os.getenv("DEEPEVAL_JUDGE_PROVIDER", "mock")).lower()
        or "mock",
        judge_model=_normalize_string(os.getenv("DEEPEVAL_JUDGE_MODEL")) or None,
        judge_api_key=_normalize_string(os.getenv("DEEPEVAL_JUDGE_API_KEY")) or None,
        judge_base_url=_normalize_string(os.getenv("DEEPEVAL_JUDGE_BASE_URL")) or None,
        upload_results=_read_bool_env("DEEPEVAL_UPLOAD_RESULTS", False),
        experiment_name=_normalize_string(
            os.getenv("DEEPEVAL_EXPERIMENT_NAME", "awe-agent-release-gate")
        )
        or "awe-agent-release-gate",
        min_chat_score=_read_float_env("DEEPEVAL_MIN_CHAT_SCORE", 0.95),
        min_planner_score=_read_float_env("DEEPEVAL_MIN_PLANNER_SCORE", 0.90),
    )


def judge_requested(settings: DeepEvalSettings) -> bool:
    return (
        bool(settings.enabled)
        and settings.mode != "off"
        and settings.judge_provider not in {"", "mock"}
    )


def validate_settings(settings: DeepEvalSettings) -> None:
    if settings.mode not in _VALID_MODES:
        raise ValueError(
            "deepeval_mode_invalid:"
            f" expected one of {sorted(_VALID_MODES)} got={settings.mode}"
        )
    if not judge_requested(settings):
        return
    if settings.judge_provider not in _JUDGE_PROVIDERS_REQUIRING_MODEL:
        raise ValueError(
            "deepeval_judge_provider_invalid:"
            f" provider={settings.judge_provider} supported={sorted(_JUDGE_PROVIDERS_REQUIRING_MODEL | {'mock'})}"
        )
    if not settings.judge_model:
        raise ValueError(
            "deepeval_judge_model_missing:"
            " set DEEPEVAL_JUDGE_MODEL when DEEPEVAL_ENABLED=true"
        )
    if not settings.judge_api_key:
        raise ValueError(
            "deepeval_judge_api_key_missing:"
            " set DEEPEVAL_JUDGE_API_KEY when DEEPEVAL_ENABLED=true"
        )
    if settings.judge_provider == "openai_compatible" and not settings.judge_base_url:
        raise ValueError(
            "deepeval_judge_base_url_missing:"
            " set DEEPEVAL_JUDGE_BASE_URL when DEEPEVAL_JUDGE_PROVIDER=openai_compatible"
        )


def _build_deterministic_metrics_summary(cases: Sequence[DeepEvalCase]) -> list[dict[str, Any]]:
    values_by_metric: dict[str, list[float]] = defaultdict(list)
    for case in cases:
        for key, value in case.local_scores.items():
            score = _score_value(value)
            if score is None:
                continue
            values_by_metric[str(key)].append(score)
    summaries: list[dict[str, Any]] = []
    for key in sorted(values_by_metric):
        values = values_by_metric[key]
        summaries.append(
            {
                "name": key,
                "count": len(values),
                "average_score": round(_safe_mean(values), 4),
                "min_score": round(min(values), 4),
                "max_score": round(max(values), 4),
            }
        )
    return summaries


def _local_case_score(case: DeepEvalCase) -> float:
    primary_score = _score_value(case.metadata.get("primary_local_score"))
    if primary_score is not None:
        return primary_score
    values = [
        score
        for score in (_score_value(value) for value in case.local_scores.values())
        if score is not None
    ]
    if values:
        return _safe_mean(values)
    return 1.0 if case.local_pass else 0.0


def _fallback_case_results(cases: Sequence[DeepEvalCase]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for case in cases:
        local_score = _local_case_score(case)
        results.append(
            {
                "case_id": case.case_id,
                "surface": case.surface,
                "source_dataset_id": case.source_dataset_id,
                "passed": bool(case.local_pass),
                "overall_score": round(local_score, 4),
                "judge_score": None,
                "local_score": round(local_score, 4),
                "judge_metrics": [],
                "local_scores": {
                    str(key): score
                    for key, score in (
                        (metric_name, _score_value(metric_value))
                        for metric_name, metric_value in case.local_scores.items()
                    )
                    if score is not None
                },
                "metadata": dict(case.metadata),
            }
        )
    return results


def _judge_metric_summaries(case_results: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    metrics: dict[str, dict[str, Any]] = {}
    for case in case_results:
        for metric in case.get("judge_metrics") or []:
            if not isinstance(metric, Mapping):
                continue
            name = _normalize_string(metric.get("name"))
            if not name:
                continue
            bucket = metrics.setdefault(
                name,
                {
                    "name": name,
                    "scores": [],
                    "success_count": 0,
                    "count": 0,
                    "error_count": 0,
                    "threshold": metric.get("threshold"),
                    "models": set(),
                    "total_cost": 0.0,
                },
            )
            score = _score_value(metric.get("score"))
            if score is not None:
                bucket["scores"].append(score)
            bucket["count"] += 1
            if bool(metric.get("success")):
                bucket["success_count"] += 1
            if _normalize_string(metric.get("error")):
                bucket["error_count"] += 1
            model_name = _normalize_string(metric.get("evaluation_model"))
            if model_name:
                bucket["models"].add(model_name)
            bucket["total_cost"] += float(metric.get("evaluation_cost") or 0.0)
    summaries: list[dict[str, Any]] = []
    for name in sorted(metrics):
        bucket = metrics[name]
        scores = bucket["scores"]
        summaries.append(
            {
                "name": name,
                "count": int(bucket["count"]),
                "average_score": round(_safe_mean(scores), 4) if scores else None,
                "pass_rate": round(
                    _safe_div(bucket["success_count"], bucket["count"]),
                    4,
                ),
                "error_count": int(bucket["error_count"]),
                "threshold": bucket["threshold"],
                "evaluation_models": sorted(bucket["models"]),
                "total_cost": round(float(bucket["total_cost"]), 6),
            }
        )
    return summaries


def _normalize_vendor_results(
    cases: Sequence[DeepEvalCase],
    evaluation_result: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    case_lookup = {case.case_id: case for case in cases}
    vendor_lookup: dict[str, Any] = {}
    raw_test_results = getattr(evaluation_result, "test_results", None)
    if isinstance(raw_test_results, list):
        for item in raw_test_results:
            name = _normalize_string(getattr(item, "name", None))
            if name:
                vendor_lookup[name] = item

    normalized_cases: list[dict[str, Any]] = []
    total_cost = 0.0
    for case in cases:
        local_score = _local_case_score(case)
        vendor_result = vendor_lookup.get(case.case_id)
        if vendor_result is None:
            normalized_cases.append(
                {
                    "case_id": case.case_id,
                    "surface": case.surface,
                    "source_dataset_id": case.source_dataset_id,
                    "passed": bool(case.local_pass),
                    "overall_score": round(local_score, 4),
                    "judge_score": None,
                    "local_score": round(local_score, 4),
                    "judge_metrics": [],
                    "local_scores": {
                        str(key): score
                        for key, score in (
                            (metric_name, _score_value(metric_value))
                            for metric_name, metric_value in case.local_scores.items()
                        )
                        if score is not None
                    },
                    "metadata": dict(case.metadata),
                    "error": "deepeval_result_missing",
                }
            )
            continue

        metric_rows: list[dict[str, Any]] = []
        metric_scores: list[float] = []
        raw_metrics = getattr(vendor_result, "metrics_data", None)
        if isinstance(raw_metrics, list):
            for metric in raw_metrics:
                score = _score_value(getattr(metric, "score", None))
                if score is not None:
                    metric_scores.append(score)
                evaluation_cost = float(getattr(metric, "evaluation_cost", 0.0) or 0.0)
                total_cost += evaluation_cost
                metric_rows.append(
                    {
                        "name": _normalize_string(getattr(metric, "name", None)),
                        "score": score,
                        "success": bool(getattr(metric, "success", False)),
                        "threshold": getattr(metric, "threshold", None),
                        "reason": _normalize_string(getattr(metric, "reason", None)) or None,
                        "error": _normalize_string(getattr(metric, "error", None)) or None,
                        "evaluation_cost": evaluation_cost,
                        "evaluation_model": _normalize_string(
                            getattr(metric, "evaluation_model", None)
                        )
                        or None,
                    }
                )
        judge_score = _safe_mean(metric_scores) if metric_scores else None
        passed = bool(getattr(vendor_result, "success", False))
        overall_score = judge_score if judge_score is not None else local_score
        normalized_cases.append(
            {
                "case_id": case.case_id,
                "surface": case.surface,
                "source_dataset_id": case.source_dataset_id,
                "passed": passed,
                "overall_score": round(overall_score, 4),
                "judge_score": round(judge_score, 4) if judge_score is not None else None,
                "local_score": round(local_score, 4),
                "judge_metrics": metric_rows,
                "local_scores": {
                    str(key): score
                    for key, score in (
                        (metric_name, _score_value(metric_value))
                        for metric_name, metric_value in case.local_scores.items()
                    )
                    if score is not None
                },
                "metadata": dict(case.metadata),
            }
        )
    deepeval_meta = {
        "test_run_id": getattr(evaluation_result, "test_run_id", None),
        "confident_link": getattr(evaluation_result, "confident_link", None),
        "total_cost": round(total_cost, 6),
    }
    return normalized_cases, deepeval_meta


def _build_report(
    *,
    surface: str,
    dataset_id: str,
    cases: Sequence[DeepEvalCase],
    case_results: Sequence[Mapping[str, Any]],
    threshold: float | None,
    settings: DeepEvalSettings,
    deterministic_summary: Mapping[str, Any] | None,
    judge_used: bool,
    deepeval_meta: Mapping[str, Any] | None,
    wall_time_s: float,
) -> dict[str, Any]:
    case_count = len(case_results)
    pass_count = sum(1 for case in case_results if bool(case.get("passed")))
    overall_scores = [
        float(case.get("overall_score") or 0.0)
        for case in case_results
        if case.get("overall_score") is not None
    ]
    overall_score = _safe_mean(overall_scores)
    failing_case_ids = [
        str(case.get("case_id") or "")
        for case in case_results
        if not bool(case.get("passed")) and _normalize_string(case.get("case_id"))
    ]
    report = {
        "schema_version": DEEPEVAL_REPORT_SCHEMA_VERSION,
        "surface": surface,
        "source_dataset_id": dataset_id,
        "source_dataset_ids": sorted({case.source_dataset_id for case in cases}),
        "case_count": case_count,
        "pass_count": pass_count,
        "pass_rate": round(_safe_div(pass_count, case_count), 4),
        "overall_score": round(overall_score, 4),
        "threshold": threshold,
        "threshold_passed": (
            True if threshold is None else round(overall_score, 4) >= float(threshold)
        ),
        "failing_case_ids": failing_case_ids,
        "judge_metrics_summary": _judge_metric_summaries(case_results),
        "deterministic_metrics_summary": _build_deterministic_metrics_summary(cases),
        "deterministic_summary": dict(deterministic_summary or {}),
        "cost_summary": {
            "total_evaluation_cost": round(
                float((deepeval_meta or {}).get("total_cost") or 0.0),
                6,
            ),
            "average_evaluation_cost": round(
                _safe_div(
                    float((deepeval_meta or {}).get("total_cost") or 0.0),
                    case_count,
                ),
                6,
            ),
        },
        "latency_summary": {
            "wall_time_s": round(wall_time_s, 4),
            "average_case_wall_time_s": round(_safe_div(wall_time_s, case_count), 4),
        },
        "deepeval": {
            "enabled": bool(settings.enabled),
            "mode": settings.mode,
            "judge_requested": judge_requested(settings),
            "judge_used": judge_used,
            "judge_provider": settings.judge_provider,
            "judge_model": settings.judge_model,
            "upload_results": bool(settings.upload_results),
            "experiment_name": settings.experiment_name,
            "test_run_id": (deepeval_meta or {}).get("test_run_id"),
            "confident_link": (deepeval_meta or {}).get("confident_link"),
        },
        "cases": list(case_results),
    }
    return report


def run_deepeval_cases(
    *,
    surface: str,
    dataset_id: str,
    cases: Sequence[DeepEvalCase],
    metric_specs: Sequence[DeepEvalMetricSpec],
    settings: DeepEvalSettings,
    threshold: float | None,
    deterministic_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    validate_settings(settings)
    started_at = time.monotonic()
    if not judge_requested(settings):
        case_results = _fallback_case_results(cases)
        return _build_report(
            surface=surface,
            dataset_id=dataset_id,
            cases=cases,
            case_results=case_results,
            threshold=threshold,
            settings=settings,
            deterministic_summary=deterministic_summary,
            judge_used=False,
            deepeval_meta={},
            wall_time_s=time.monotonic() - started_at,
        )

    try:
        from deepeval import evaluate
        from deepeval.evaluate.configs import DisplayConfig, ErrorConfig
        from deepeval.metrics import GEval
        from deepeval.models.base_model import DeepEvalBaseLLM
        from deepeval.test_case import LLMTestCase, LLMTestCaseParams
    except ImportError as exc:  # pragma: no cover - requires optional dependency
        raise RuntimeError(
            "deepeval_package_required:"
            " install with `uv run --with deepeval ...` or disable DEEPEVAL_ENABLED"
        ) from exc

    class _ProviderJudgeModel(DeepEvalBaseLLM):  # pragma: no cover - exercised via optional dependency
        def __init__(self, current_settings: DeepEvalSettings) -> None:
            self._settings = current_settings
            self._provider: llm_provider.LLMProvider | None = None

        def load_model(self) -> llm_provider.LLMProvider:
            if self._provider is None:
                self._provider = llm_provider.resolve_provider(
                    self._settings.judge_provider,
                    api_key=self._settings.judge_api_key,
                    model=self._settings.judge_model,
                    base_url=self._settings.judge_base_url,
                    max_output_tokens=1024,
                    timeout_s=60.0,
                    max_retries=1,
                )
            return self._provider

        def generate(self, prompt: str, schema: Any | None = None) -> Any:
            provider = self.load_model()
            text = provider.generate_request(
                llm_provider.LLMRequest(
                    prompt=prompt,
                    metadata={"component": "deepeval_judge"},
                )
            ).content
            if schema is not None:
                validator = getattr(schema, "model_validate_json", None)
                if callable(validator):
                    try:
                        return validator(text)
                    except Exception:  # noqa: BLE001
                        return text
            return text

        async def a_generate(self, prompt: str, schema: Any | None = None) -> Any:
            return self.generate(prompt, schema=schema)

        def get_model_name(self) -> str:
            return self._settings.judge_model or f"{self._settings.judge_provider}:unknown"

    test_cases = [
        LLMTestCase(
            name=case.case_id,
            input=case.input_text,
            actual_output=case.actual_output,
            expected_output=case.expected_output,
            context=list(case.context),
            additional_metadata={
                "surface": case.surface,
                "source_dataset_id": case.source_dataset_id,
                **dict(case.metadata),
            },
            tags=[surface, case.surface, case.source_dataset_id],
        )
        for case in cases
    ]
    judge_model = _ProviderJudgeModel(settings)
    metrics = [
        GEval(
            name=spec.name,
            criteria=spec.criteria,
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.EXPECTED_OUTPUT,
                LLMTestCaseParams.CONTEXT,
            ],
            model=judge_model,
            threshold=spec.threshold,
            async_mode=False,
            verbose_mode=False,
        )
        for spec in metric_specs
    ]
    evaluation_result = evaluate(
        test_cases=test_cases,
        metrics=metrics,
        identifier=settings.experiment_name,
        display_config=DisplayConfig(show_indicator=False, print_results=False),
        error_config=ErrorConfig(ignore_errors=False, skip_on_missing_params=False),
    )
    case_results, deepeval_meta = _normalize_vendor_results(cases, evaluation_result)
    return _build_report(
        surface=surface,
        dataset_id=dataset_id,
        cases=cases,
        case_results=case_results,
        threshold=threshold,
        settings=settings,
        deterministic_summary=deterministic_summary,
        judge_used=True,
        deepeval_meta=deepeval_meta,
        wall_time_s=time.monotonic() - started_at,
    )


def build_gate_report(
    *,
    chat_report: Mapping[str, Any],
    planner_report: Mapping[str, Any],
) -> dict[str, Any]:
    chat_score = float(chat_report.get("overall_score") or 0.0)
    planner_score = float(planner_report.get("overall_score") or 0.0)
    chat_threshold = chat_report.get("threshold")
    planner_threshold = planner_report.get("threshold")
    return {
        "schema_version": DEEPEVAL_GATE_SCHEMA_VERSION,
        "chat": {
            "overall_score": round(chat_score, 4),
            "threshold": chat_threshold,
            "threshold_passed": bool(chat_report.get("threshold_passed")),
            "pass_rate": float(chat_report.get("pass_rate") or 0.0),
            "failing_case_ids": list(chat_report.get("failing_case_ids") or []),
            "report_path": chat_report.get("report_path"),
        },
        "planner": {
            "overall_score": round(planner_score, 4),
            "threshold": planner_threshold,
            "threshold_passed": bool(planner_report.get("threshold_passed")),
            "pass_rate": float(planner_report.get("pass_rate") or 0.0),
            "failing_case_ids": list(planner_report.get("failing_case_ids") or []),
            "report_path": planner_report.get("report_path"),
        },
        "threshold_passed": bool(chat_report.get("threshold_passed"))
        and bool(planner_report.get("threshold_passed")),
        "overall_score": round(_safe_mean([chat_score, planner_score]), 4),
    }


def write_report(path: os.PathLike[str] | str, report: Mapping[str, Any]) -> None:
    from pathlib import Path

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
