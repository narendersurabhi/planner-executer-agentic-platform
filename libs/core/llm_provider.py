from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LLMResponse:
    content: str


class LLMProvider:
    def generate(self, prompt: str) -> LLMResponse:  # pragma: no cover - interface
        raise NotImplementedError


class MockLLMProvider(LLMProvider):
    def generate(self, prompt: str) -> LLMResponse:
        return LLMResponse(content="Mock response")


class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    def generate(self, prompt: str) -> LLMResponse:
        return LLMResponse(content="OpenAI provider stub")
