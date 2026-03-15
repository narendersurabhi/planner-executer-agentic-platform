from __future__ import annotations

from dataclasses import dataclass

from coder_core.errors import CoderError
from coder_core.models import CodeGenRequest
from coder_core.service import (
    CoderServiceConfig,
    ProviderConfig,
    ResponseParser,
    build_llm_request,
    build_prompt,
    generate_code,
)
from libs.core.llm_provider import LLMRequest, LLMResponse


@dataclass
class _ProviderStub:
    responses: list[str]
    failures: int = 0

    def __post_init__(self) -> None:
        self.calls = 0
        self.requests: list[LLMRequest] = []

    def generate_request(self, request: LLMRequest) -> LLMResponse:
        call_index = self.calls
        self.calls += 1
        self.requests.append(request)
        if call_index < self.failures:
            raise RuntimeError("temporary upstream failure")
        return LLMResponse(content=self.responses[call_index - self.failures])


class _LoggerStub:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def info(self, event: str, extra: dict | None = None) -> None:
        self.events.append((event, extra or {}))

    def error(self, event: str, extra: dict | None = None) -> None:
        self.events.append((event, extra or {}))


def test_build_prompt_includes_requested_files_and_constraints() -> None:
    request = CodeGenRequest(
        goal="Build a demo service",
        files=["app.py", "README.md"],
        constraints="Keep it short",
    )

    prompt = build_prompt(request)

    assert "You MUST output exactly these relative file paths" in prompt
    assert "- app.py" in prompt
    assert "- README.md" in prompt
    assert "Constraints:\nKeep it short" in prompt


def test_response_parser_rejects_non_object_json() -> None:
    parser = ResponseParser()

    try:
        parser.extract_json_object('["not-an-object"]')
    except CoderError as exc:
        assert exc.detail == "invalid_json:No JSON object found in LLM response"
    else:  # pragma: no cover
        raise AssertionError("Expected CoderError")


def test_build_llm_request_carries_prompt_and_metadata() -> None:
    request = CodeGenRequest(goal="Build app", files=["app.py"], constraints="Keep it short")

    llm_request = build_llm_request(
        request,
        ProviderConfig(
            provider_name="openai",
            api_key="test-key",
            model="gpt-5-mini",
            base_url="https://api.openai.com",
            temperature=None,
            max_output_tokens=512,
            timeout_s=30.0,
            max_retries=1,
        ),
    )

    assert "Goal:\nBuild app" in llm_request.prompt
    assert llm_request.max_output_tokens == 512
    assert llm_request.metadata == {
        "component": "coder",
        "goal_len": 9,
        "files_count": 1,
        "constraints_len": 13,
        "model": "gpt-5-mini",
    }


def test_generate_code_uses_service_config_retry_and_parses_response() -> None:
    provider = _ProviderStub(
        responses=['{"files":[{"path":"app.py","content":"print(1)"}]}'],
        failures=1,
    )
    logger = _LoggerStub()
    config = CoderServiceConfig(
        provider_name="openai",
        model="gpt-5-mini",
        llm_max_retries=1,
        llm_retry_sleep_s=0.0,
        log_prompt_preview_chars=None,
        log_response_preview_chars=None,
        log_response_full=False,
    )

    result = generate_code(CodeGenRequest(goal="Build app"), provider, logger, config)

    assert provider.calls == 2
    assert result.model_dump() == {"files": [{"path": "app.py", "content": "print(1)"}]}
    assert provider.requests[0].metadata is not None
    assert provider.requests[0].metadata["component"] == "coder"
    assert any(event == "codegen_llm_error" for event, _ in logger.events)
    assert any(event == "codegen_success" for event, _ in logger.events)


def test_generate_code_raises_coder_error_for_invalid_schema() -> None:
    provider = _ProviderStub(responses=['{"files":[{"path":"app.py"}]}'])
    logger = _LoggerStub()
    config = CoderServiceConfig(
        provider_name="mock",
        model="",
        llm_max_retries=0,
        llm_retry_sleep_s=0.0,
        log_prompt_preview_chars=None,
        log_response_preview_chars=None,
        log_response_full=False,
    )

    try:
        generate_code(CodeGenRequest(goal="Build app"), provider, logger, config)
    except CoderError as exc:
        assert exc.detail.startswith("invalid_schema:")
    else:  # pragma: no cover
        raise AssertionError("Expected CoderError")
