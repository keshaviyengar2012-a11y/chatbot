"""Thin wrapper around the Anthropic Messages API.

Reads ANTHROPIC_API_KEY from the environment. The model can be overridden with TUTOR_MODEL.
"""

from __future__ import annotations

import os
from typing import Protocol, Sequence

DEFAULT_MODEL = "claude-sonnet-5-5"


class LLM(Protocol):
    def complete(self, system: str, messages: Sequence[dict]) -> str: ...


class AnthropicLLM:
    def __init__(self, model: str | None = None, max_tokens: int = 1024, client=None) -> None:
        self.model = model or os.environ.get("TUTOR_MODEL", DEFAULT_MODEL)
        self.max_tokens = max_tokens
        if client is None:
            import anthropic

            client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        self._client = client

    def complete(self, system: str, messages: Sequence[dict]) -> str:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=list(messages),
        )
        return "".join(block.text for block in response.content if block.type == "text").strip()
