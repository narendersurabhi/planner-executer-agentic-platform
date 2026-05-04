from __future__ import annotations

import io
import json
from urllib.error import HTTPError

from urllib.error import URLError

from libs.core.llm_provider import (
    LLMProvider,
    LLMProviderError,
    LLMRequest,
    LLMResponse,
    OpenAIChatCompletionsProvider,
    OpenAIProvider,
    parse_json_object,
)
from libs.core import llm_provider as llm_provider_module


class _FakeHTTPResponse:
    def __init__(self, payload: dict) -> None:
        self._raw = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._raw

    def __enter__(self) -> "_FakeHTTPResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


def _success_payload() -> dict:
    return {
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": '{"ok":true}'}],
            }
        ]
    }


def test_openai_provider_omits_temperature_for_gpt5(monkeypatch) -> None:
    captured_payloads: list[dict] = []

    def _fake_urlopen(request, timeout=0):  # type: ignore[no-untyped-def]
        body = json.loads(request.data.decode("utf-8"))
        captured_payloads.append(body)
        return _FakeHTTPResponse(_success_payload())

    monkeypatch.setattr(llm_provider_module, "urlopen", _fake_urlopen)

    provider = OpenAIProvider(
        api_key="test-key",
        model="gpt-5-mini",
        temperature=0.7,
    )
    response = provider.generate("hello")
    assert response.content == '{"ok":true}'
    assert len(captured_payloads) == 1
    assert "temperature" not in captured_payloads[0]


def test_openai_provider_retries_without_temperature_on_unsupported_error(monkeypatch) -> None:
    captured_payloads: list[dict] = []
    state = {"count": 0}

    def _fake_urlopen(request, timeout=0):  # type: ignore[no-untyped-def]
        body = json.loads(request.data.decode("utf-8"))
        captured_payloads.append(body)
        if state["count"] == 0:
            state["count"] += 1
            error_body = (
                b'{"error":{"message":"Unsupported parameter: \'temperature\' is not supported with this model."}}'
            )
            raise HTTPError(
                url="https://api.openai.com/v1/responses",
                code=400,
                msg="Bad Request",
                hdrs=None,
                fp=io.BytesIO(error_body),
            )
        return _FakeHTTPResponse(_success_payload())

    monkeypatch.setattr(llm_provider_module, "urlopen", _fake_urlopen)

    provider = OpenAIProvider(
        api_key="test-key",
        model="gpt-4.1-mini",
        temperature=0.2,
        max_retries=0,
    )
    response = provider.generate("hello")
    assert response.content == '{"ok":true}'
    assert len(captured_payloads) == 2
    assert captured_payloads[0]["temperature"] == 0.2
    assert "temperature" not in captured_payloads[1]


def test_openai_provider_builds_request_payload_from_llm_request() -> None:
    provider = OpenAIProvider(
        api_key="test-key",
        model="gpt-4.1-mini",
        temperature=0.4,
        max_output_tokens=256,
    )

    payload = provider._build_payload(  # type: ignore[attr-defined]
        LLMRequest(
            prompt="hello",
            system_prompt="system message",
            temperature=0.1,
            max_output_tokens=42,
            metadata={"component": "coder", "goal_len": 9},
        )
    )

    assert payload == {
        "model": "gpt-4.1-mini",
        "input": "hello",
        "instructions": "system message",
        "temperature": 0.1,
        "max_output_tokens": 42,
        "metadata": {"component": "coder", "goal_len": "9"},
    }


def test_openai_chat_completions_provider_builds_messages_payload() -> None:
    provider = OpenAIChatCompletionsProvider(
        api_key="test-key",
        model="gemini-2.5-flash",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        temperature=0.3,
        max_output_tokens=128,
        provider_label="Gemini",
    )

    payload = provider._build_payload(  # type: ignore[attr-defined]
        LLMRequest(
            prompt="hello",
            system_prompt="system message",
            temperature=0.1,
            max_output_tokens=42,
            metadata={"component": "planner"},
        )
    )

    assert payload == {
        "model": "gemini-2.5-flash",
        "messages": [
            {"role": "system", "content": "system message"},
            {"role": "user", "content": "hello"},
        ],
        "temperature": 0.1,
        "max_tokens": 42,
    }


def test_openai_chat_completions_provider_keeps_metadata_for_non_gemini_base_url() -> None:
    provider = OpenAIChatCompletionsProvider(
        api_key="test-key",
        model="gpt-4.1-mini",
        base_url="https://api.openai.com/v1",
        temperature=0.3,
        max_output_tokens=128,
    )

    payload = provider._build_payload(  # type: ignore[attr-defined]
        LLMRequest(
            prompt="hello",
            system_prompt="system message",
            temperature=0.1,
            max_output_tokens=42,
            metadata={"component": "planner"},
        )
    )

    assert payload == {
        "model": "gpt-4.1-mini",
        "messages": [
            {"role": "system", "content": "system message"},
            {"role": "user", "content": "hello"},
        ],
        "temperature": 0.1,
        "max_tokens": 42,
        "metadata": {"component": "planner"},
    }


def test_resolve_provider_supports_gemini_env(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-flash")
    monkeypatch.delenv("GEMINI_BASE_URL", raising=False)

    provider = llm_provider_module.resolve_provider(
        "gemini",
        api_key="openai-key",
        model="gpt-5-mini",
        base_url="https://api.openai.com",
    )

    assert isinstance(provider, OpenAIChatCompletionsProvider)
    assert provider.api_key == "gemini-key"
    assert provider.model == "gemini-2.5-flash"
    assert provider.base_url == "https://generativelanguage.googleapis.com/v1beta/openai"


def test_resolve_provider_supports_openai_compatible_endpoint() -> None:
    provider = llm_provider_module.resolve_provider(
        "openai_compatible",
        api_key="custom-key",
        model="fine-tuned-model",
        base_url="https://llm.example.test/v1",
    )

    assert isinstance(provider, OpenAIChatCompletionsProvider)
    assert provider.api_key == "custom-key"
    assert provider.model == "fine-tuned-model"
    assert provider.base_url == "https://llm.example.test/v1"


def test_openai_provider_retries_retryable_connection_error(monkeypatch) -> None:
    state = {"count": 0}
    sleeps: list[int] = []

    def _fake_urlopen(request, timeout=0):  # type: ignore[no-untyped-def]
        del request, timeout
        if state["count"] == 0:
            state["count"] += 1
            raise URLError("temporary network failure")
        return _FakeHTTPResponse(_success_payload())

    monkeypatch.setattr(llm_provider_module, "urlopen", _fake_urlopen)
    monkeypatch.setattr(llm_provider_module.time, "sleep", lambda seconds: sleeps.append(seconds))

    provider = OpenAIProvider(
        api_key="test-key",
        model="gpt-4.1-mini",
        max_retries=1,
    )

    response = provider.generate("hello")

    assert response.content == '{"ok":true}'
    assert sleeps == [1]


def test_openai_provider_raises_for_empty_output(monkeypatch) -> None:
    def _fake_urlopen(request, timeout=0):  # type: ignore[no-untyped-def]
        del request, timeout
        return _FakeHTTPResponse({"output": []})

    monkeypatch.setattr(llm_provider_module, "urlopen", _fake_urlopen)

    provider = OpenAIProvider(api_key="test-key", model="gpt-4.1-mini")

    try:
        provider.generate("hello")
    except LLMProviderError as exc:
        assert str(exc) == "OpenAI API returned empty output"
    else:  # pragma: no cover
        raise AssertionError("Expected LLMProviderError")


def test_parse_json_object_extracts_fenced_json_object() -> None:
    payload = parse_json_object('```json\n{"ok": true, "count": 2}\n```')

    assert payload == {"ok": True, "count": 2}


def test_parse_json_object_accepts_double_encoded_json_string() -> None:
    payload = parse_json_object('"{\\"ok\\": true, \\"count\\": 2}"')

    assert payload == {"ok": True, "count": 2}


def test_parse_json_object_accepts_singleton_list_with_object() -> None:
    payload = parse_json_object('[{"ok": true}]')

    assert payload == {"ok": True}


def test_generate_json_object_uses_generate_compatibility_path() -> None:
    class _LegacyProvider(LLMProvider):
        def generate(self, prompt: str) -> LLMResponse:
            assert prompt == "hello"
            return LLMResponse(content='{"ok": true}')

    payload = _LegacyProvider().generate_json_object("hello")

    assert payload == {"ok": True}
