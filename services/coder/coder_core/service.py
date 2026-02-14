from __future__ import annotations

import json
import os
import time
from typing import Any

from libs.core import llm_provider

from .errors import CoderError
from .models import CodeGenRequest, CodeGenResponse


def _parse_optional_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _parse_optional_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _parse_optional_int_with_default(value: str | None, default: int) -> int:
    parsed = _parse_optional_int(value)
    if parsed is None:
        return default
    return parsed


def _parse_optional_float_with_default(value: str | None, default: float) -> float:
    parsed = _parse_optional_float(value)
    if parsed is None:
        return default
    return parsed


def create_provider_from_env() -> Any:
    return llm_provider.resolve_provider(
        os.getenv("LLM_PROVIDER", "mock"),
        api_key=os.getenv("OPENAI_API_KEY", ""),
        model=os.getenv("OPENAI_MODEL", ""),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com"),
        temperature=_parse_optional_float(os.getenv("OPENAI_TEMPERATURE")),
        max_output_tokens=_parse_optional_int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS")),
        timeout_s=_parse_optional_float(os.getenv("OPENAI_TIMEOUT_S")),
        max_retries=_parse_optional_int(os.getenv("OPENAI_MAX_RETRIES")),
    )


def build_prompt(request: CodeGenRequest) -> str:
    files_hint = ""
    if request.files:
        files_hint = (
            "You MUST output exactly these relative file paths:\n- "
            + "\n- ".join(request.files)
            + "\n"
        )
    constraints = f"Constraints:\n{request.constraints}\n" if request.constraints else ""
    return (
        "You are a coding agent. Return ONLY JSON (no prose, no markdown).\n"
        "Output must be a single JSON object with this shape:\n"
        '{"files":[{"path":"relative/path.ext","content":"..."}]}\n'
        "Rules:\n"
        "- Paths must be relative (no leading slash), no '..' segments.\n"
        "- Keep code concise and runnable.\n"
        "- Do not include backticks or markdown.\n"
        f"{files_hint}"
        f"{constraints}"
        f"Goal:\n{request.goal}\n"
    )


def extract_json(text: str) -> dict:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in LLM response")
    candidate = stripped[start : end + 1]
    return json.loads(candidate)


def preview(text: str, limit: str | None) -> str:
    if not limit:
        return ""
    try:
        max_len = int(limit)
    except ValueError:
        return ""
    if max_len <= 0:
        return ""
    return text[:max_len]


def generate_code(request: CodeGenRequest, provider: Any, logger: Any) -> CodeGenResponse:
    prompt = build_prompt(request)
    logger.info(
        "codegen_request",
        extra={
            "provider": os.getenv("LLM_PROVIDER", "mock"),
            "model": os.getenv("OPENAI_MODEL", ""),
            "goal_len": len(request.goal),
            "files_count": len(request.files or []),
            "constraints_len": len(request.constraints or ""),
            "prompt_len": len(prompt),
        },
    )
    prompt_preview = preview(prompt, os.getenv("CODER_LOG_PROMPT_PREVIEW_CHARS"))
    if prompt_preview:
        logger.info("codegen_prompt_preview", extra={"preview": prompt_preview})

    start = time.perf_counter()
    max_retries = _parse_optional_int_with_default(os.getenv("CODER_LLM_MAX_RETRIES"), 2)
    retry_sleep = _parse_optional_float_with_default(os.getenv("CODER_LLM_RETRY_SLEEP_S"), 1.5)

    attempt = 0
    response = None
    while attempt <= max_retries:
        try:
            response = provider.generate(prompt)
            break
        except Exception as exc:  # noqa: BLE001
            attempt += 1
            logger.error(
                "codegen_llm_error",
                extra={"error": str(exc), "attempt": attempt, "max_retries": max_retries},
            )
            if attempt > max_retries:
                raise CoderError(str(exc), status_code=502) from exc
            time.sleep(retry_sleep)

    duration_ms = int((time.perf_counter() - start) * 1000)
    response_preview = preview(response.content, os.getenv("CODER_LOG_RESPONSE_PREVIEW_CHARS"))
    logger.info(
        "codegen_llm_response",
        extra={
            "duration_ms": duration_ms,
            "response_len": len(response.content or ""),
        },
    )
    if os.getenv("CODER_LOG_RESPONSE_FULL", "").lower() == "true":
        logger.info("codegen_response_full", extra={"response": response.content})
    if response_preview:
        logger.info("codegen_response_preview", extra={"preview": response_preview})

    try:
        data = extract_json(response.content)
    except Exception as exc:  # noqa: BLE001
        logger.error("codegen_parse_error", extra={"error": str(exc)})
        raise CoderError(f"invalid_json:{exc}") from exc
    if not isinstance(data, dict) or "files" not in data:
        raise CoderError("missing_files")
    try:
        parsed = CodeGenResponse.model_validate(data)
    except Exception as exc:  # noqa: BLE001
        logger.error("codegen_schema_error", extra={"error": str(exc)})
        raise CoderError(f"invalid_schema:{exc}") from exc
    logger.info("codegen_success", extra={"files_count": len(parsed.files)})
    return parsed
