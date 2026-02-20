from __future__ import annotations

import io
import json
from urllib.error import HTTPError

from libs.core.llm_provider import OpenAIProvider
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
