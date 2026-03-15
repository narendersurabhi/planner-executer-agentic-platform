from __future__ import annotations

from dataclasses import dataclass
import json
import os
import time
from typing import Any

from libs.core import llm_provider

from .errors import CoderError
from .models import CodeGenRequest, CodeGenResponse


@dataclass(frozen=True)
class ProviderConfig:
    provider_name: str
    api_key: str
    model: str
    base_url: str
    temperature: float | None
    max_output_tokens: int | None
    timeout_s: float | None
    max_retries: int | None

    @classmethod
    def from_env(cls) -> ProviderConfig:
        return cls(
            provider_name=os.getenv("LLM_PROVIDER", "mock"),
            api_key=os.getenv("OPENAI_API_KEY", ""),
            model=os.getenv("OPENAI_MODEL", ""),
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com"),
            temperature=_parse_optional_float(os.getenv("OPENAI_TEMPERATURE")),
            max_output_tokens=_parse_optional_int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS")),
            timeout_s=_parse_optional_float(os.getenv("OPENAI_TIMEOUT_S")),
            max_retries=_parse_optional_int(os.getenv("OPENAI_MAX_RETRIES")),
        )


@dataclass(frozen=True)
class CoderServiceConfig:
    provider_name: str
    model: str
    llm_max_retries: int
    llm_retry_sleep_s: float
    log_prompt_preview_chars: str | None
    log_response_preview_chars: str | None
    log_response_full: bool

    @classmethod
    def from_env(cls) -> CoderServiceConfig:
        provider = ProviderConfig.from_env()
        return cls(
            provider_name=provider.provider_name,
            model=provider.model,
            llm_max_retries=_parse_optional_int_with_default(
                os.getenv("CODER_LLM_MAX_RETRIES"), 2
            ),
            llm_retry_sleep_s=_parse_optional_float_with_default(
                os.getenv("CODER_LLM_RETRY_SLEEP_S"), 1.5
            ),
            log_prompt_preview_chars=os.getenv("CODER_LOG_PROMPT_PREVIEW_CHARS"),
            log_response_preview_chars=os.getenv("CODER_LOG_RESPONSE_PREVIEW_CHARS"),
            log_response_full=os.getenv("CODER_LOG_RESPONSE_FULL", "").lower() == "true",
        )


class PromptBuilder:
    def build(self, request: CodeGenRequest) -> str:
        files_hint = ""
        if request.files:
            files_hint = (
                "You MUST output exactly these relative file paths:\n- "
                + "\n- ".join(request.files)
                + "\n"
            )
        constraints = (
            f"Constraints:\n{request.constraints}\n" if request.constraints else ""
        )
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


def build_llm_request(
    request: CodeGenRequest, config: ProviderConfig | None = None
) -> llm_provider.LLMRequest:
    provider_config = config or ProviderConfig.from_env()
    return llm_provider.LLMRequest(
        prompt=build_prompt(request),
        temperature=provider_config.temperature,
        max_output_tokens=provider_config.max_output_tokens,
        metadata={
            "component": "coder",
            "goal_len": len(request.goal),
            "files_count": len(request.files or []),
            "constraints_len": len(request.constraints or ""),
            "model": provider_config.model,
        },
    )


class ResponseParser:
    def parse(self, text: str) -> CodeGenResponse:
        data = self.extract_json_object(text)
        if not isinstance(data, dict) or "files" not in data:
            raise CoderError("missing_files")
        try:
            return CodeGenResponse.model_validate(data)
        except Exception as exc:  # noqa: BLE001
            raise CoderError(f"invalid_schema:{exc}") from exc

    def extract_json_object(self, text: str) -> dict[str, Any]:
        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = stripped.strip("`")
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise CoderError("invalid_json:No JSON object found in LLM response")
        candidate = stripped[start : end + 1]
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError as exc:
            raise CoderError(f"invalid_json:{exc}") from exc
        if not isinstance(data, dict):
            raise CoderError("invalid_json:Top-level JSON value must be an object")
        return data


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
    config = ProviderConfig.from_env()
    return llm_provider.resolve_provider(
        config.provider_name,
        api_key=config.api_key,
        model=config.model,
        base_url=config.base_url,
        temperature=config.temperature,
        max_output_tokens=config.max_output_tokens,
        timeout_s=config.timeout_s,
        max_retries=config.max_retries,
    )


def build_prompt(request: CodeGenRequest) -> str:
    return PromptBuilder().build(request)


def extract_json(text: str) -> dict:
    return ResponseParser().extract_json_object(text)


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


def _call_provider_with_retry(
    llm_request: llm_provider.LLMRequest, provider: Any, logger: Any, config: CoderServiceConfig
) -> Any:
    logger.info(
        "codegen_request",
        extra={
            "provider": config.provider_name,
            "model": config.model,
            "prompt_len": len(llm_request.prompt),
            "metadata": llm_request.metadata or {},
        },
    )
    prompt_preview = preview(llm_request.prompt, config.log_prompt_preview_chars)
    if prompt_preview:
        logger.info("codegen_prompt_preview", extra={"preview": prompt_preview})

    start = time.perf_counter()
    attempt = 0
    response = None
    while attempt <= config.llm_max_retries:
        try:
            response = provider.generate_request(llm_request)
            break
        except Exception as exc:  # noqa: BLE001
            attempt += 1
            logger.error(
                "codegen_llm_error",
                extra={
                    "error": str(exc),
                    "attempt": attempt,
                    "max_retries": config.llm_max_retries,
                },
            )
            if attempt > config.llm_max_retries:
                raise CoderError(str(exc), status_code=502) from exc
            time.sleep(config.llm_retry_sleep_s)

    duration_ms = int((time.perf_counter() - start) * 1000)
    response_preview = preview(response.content, config.log_response_preview_chars)
    logger.info(
        "codegen_llm_response",
        extra={
            "duration_ms": duration_ms,
            "response_len": len(response.content or ""),
        },
    )
    if config.log_response_full:
        logger.info("codegen_response_full", extra={"response": response.content})
    if response_preview:
        logger.info("codegen_response_preview", extra={"preview": response_preview})
    return response


def generate_code(
    request: CodeGenRequest,
    provider: Any,
    logger: Any,
    config: CoderServiceConfig | None = None,
) -> CodeGenResponse:
    service_config = config or CoderServiceConfig.from_env()
    provider_config = ProviderConfig.from_env()
    llm_request = build_llm_request(request, provider_config)
    logger.info(
        "codegen_request_context",
        extra={
            "goal_len": len(request.goal),
            "files_count": len(request.files or []),
            "constraints_len": len(request.constraints or ""),
        },
    )
    response = _call_provider_with_retry(llm_request, provider, logger, service_config)

    try:
        parsed = ResponseParser().parse(response.content)
    except CoderError as exc:
        event = "codegen_schema_error" if exc.detail.startswith("invalid_schema:") else "codegen_parse_error"
        logger.error(event, extra={"error": exc.detail})
        raise
    logger.info("codegen_success", extra={"files_count": len(parsed.files)})
    return parsed
