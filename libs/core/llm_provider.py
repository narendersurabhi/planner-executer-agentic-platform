from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import time


@dataclass
class LLMResponse:
    content: str


class LLMProviderError(Exception):
    pass


class LLMProvider:
    def generate(self, prompt: str) -> LLMResponse:  # pragma: no cover - interface
        raise NotImplementedError


class MockLLMProvider(LLMProvider):
    def generate(self, prompt: str) -> LLMResponse:
        return LLMResponse(content="Mock response")


class OpenAIProvider(LLMProvider):
    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://api.openai.com",
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        timeout_s: float = 30.0,
        max_retries: int = 0,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.timeout_s = timeout_s
        self.max_retries = max_retries

    def generate(self, prompt: str) -> LLMResponse:
        payload: Dict[str, Any] = {"model": self.model, "input": prompt}
        if self.temperature is not None and _model_supports_temperature(self.model):
            payload["temperature"] = self.temperature
        if self.max_output_tokens is not None:
            payload["max_output_tokens"] = self.max_output_tokens
        attempts = self.max_retries + 1
        retried_without_temperature = False
        attempt = 0
        while attempt < attempts:
            request = Request(
                f"{self.base_url}/v1/responses",
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            try:
                with urlopen(request, timeout=self.timeout_s) as response:
                    body = response.read().decode("utf-8")
                data = json.loads(body)
                text = _extract_output_text(data)
                if not text:
                    raise LLMProviderError("OpenAI API returned empty output")
                return LLMResponse(content=text)
            except HTTPError as exc:
                detail = exc.read().decode("utf-8") if exc.fp else str(exc)
                if (
                    "temperature" in payload
                    and not retried_without_temperature
                    and _is_unsupported_temperature_error(detail)
                ):
                    payload.pop("temperature", None)
                    retried_without_temperature = True
                    continue
                if exc.code in {429, 500, 502, 503, 504} and attempt < attempts - 1:
                    time.sleep(min(2**attempt, 8))
                    attempt += 1
                    continue
                raise LLMProviderError(f"OpenAI API error: {detail}") from exc
            except (URLError, TimeoutError) as exc:
                if attempt < attempts - 1:
                    time.sleep(min(2**attempt, 8))
                    attempt += 1
                    continue
                raise LLMProviderError(f"OpenAI API connection error: {exc}") from exc
        raise LLMProviderError("OpenAI API request failed after retries")


def resolve_provider(
    provider_name: str,
    *,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: Optional[float] = None,
    max_output_tokens: Optional[int] = None,
    timeout_s: Optional[float] = None,
    max_retries: Optional[int] = None,
) -> LLMProvider:
    name = (provider_name or "mock").lower()
    if name == "openai":
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
        if not model:
            raise ValueError("OPENAI_MODEL is required when LLM_PROVIDER=openai")
        return OpenAIProvider(
            api_key=api_key,
            model=model,
            base_url=base_url or "https://api.openai.com",
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            timeout_s=timeout_s or 30.0,
            max_retries=max_retries or 0,
        )
    return MockLLMProvider()


def _extract_output_text(response: Dict[str, Any]) -> str:
    parts: list[str] = []
    for item in response.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                parts.append(content.get("text", ""))
    return "".join(parts).strip()


def _model_supports_temperature(model: str) -> bool:
    normalized = (model or "").strip().lower()
    # GPT-5 responses currently reject temperature.
    return not normalized.startswith("gpt-5")


def _is_unsupported_temperature_error(detail: str) -> bool:
    lowered = (detail or "").lower()
    return "unsupported parameter" in lowered and "temperature" in lowered
