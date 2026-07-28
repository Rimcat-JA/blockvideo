"""LLM provider abstraction.

The interface is intentionally minimal so we can plug in OpenAI-compatible
APIs (incl. Responses API endpoints shaped for chat completions) or any
other backend (Gemini, Anthropic, local llama.cpp, ...). The application
must never depend on a vendor SDK directly — call through this abstraction.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any


@dataclass
class LLMMessage:
    """One role/content pair sent to an LLM provider."""

    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass
class LLMRequest:
    """Vendor-neutral chat request including optional JSON/schema hints."""

    messages: list[LLMMessage]
    response_format: dict[str, Any] | None = None  # JSON schema hint
    temperature: float = 0.2
    max_tokens: int | None = None


@dataclass
class LLMResponse:
    """Normalized provider response with optional raw vendor payload."""

    content: str
    raw: dict[str, Any] | None = None


class LLMProvider(abc.ABC):
    """Common LLM interface.

    Implementations must be safe about logging: never include API keys or
    full script text in exception messages. Use the ``secret_store`` summary
    for diagnostics.
    """

    name: str = "abstract"

    @abc.abstractmethod
    async def chat(self, request: LLMRequest) -> LLMResponse:
        """Send a chat request and return normalized text content."""
        ...

    async def chat_json(self, request: LLMRequest) -> dict[str, Any]:
        """Request and parse JSON, tolerating models that wrap it in prose.

        Parse the response as-is first. The salvage heuristics below are for
        models that wrap JSON in a markdown fence or add a preamble, but they
        are destructive when applied to *valid* JSON whose string values
        happen to contain ``` (a script quoting a code block) or braces — the
        fence regex then matches inside the payload and slices out a fragment.
        Trying a straight parse first keeps well-formed responses intact.
        """
        import json
        import re

        response = await self.chat(request)
        text = response.content.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Salvage: strip a surrounding ```json fence, if present.
        fence = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL)
        if fence:
            candidate = fence.group(1)
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

        # Salvage: take the widest {...} region.
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
        return json.loads(text)


class ProviderError(RuntimeError):
    """Raised when an upstream LLM fails.

    Exception messages are sanitized: stack frames may show technical detail
    but should never carry the API key. Use ``safe=True`` for user-facing
    messages.
    """

    def __init__(self, message: str, *, safe: bool = True, original: Exception | None = None) -> None:
        """Create an error with a user-safe message and optional original cause."""
        super().__init__(message)
        self.safe = safe
        self.original = original


async def safe_chat_json(provider: LLMProvider, request: LLMRequest) -> dict[str, Any]:
    """Run chat_json and convert ProviderError into a sanitized message."""
    try:
        return await provider.chat_json(request)
    except ProviderError:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        raise ProviderError(str(exc), safe=True, original=exc) from exc
