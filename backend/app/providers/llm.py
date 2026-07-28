"""Vendor-neutral language-model request and response abstractions.

The interface is intentionally minimal so we can plug in OpenAI-compatible
APIs (incl. Responses API endpoints shaped for chat completions) or any
other backend (Gemini, Anthropic, local llama.cpp, ...). The application
must never depend on a vendor SDK directly — call through this abstraction.
Imports:
    ``abc`` marks the provider interface as abstract.
    ``dataclass`` defines transport objects with predictable fields.
    ``Any`` represents vendor JSON payloads without coupling to an SDK.

The application depends on this module rather than a vendor SDK.  Real HTTP
providers and deterministic test providers implement the same ``chat`` method.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any


@dataclass
class LLMMessage:
    """One ordered role/content item in a chat request.

    Attributes:
        role: Provider role, normally ``system``, ``user``, or ``assistant``.
        content: UTF-8 text sent to the provider.

    """

    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass
class LLMRequest:
    """Vendor-neutral chat request and generation controls.

    Attributes:
        messages: Ordered conversation items.
        response_format: Optional provider JSON/schema hint.
        temperature: Sampling temperature forwarded to compatible providers.
        max_tokens: Optional output-token limit; providers translate syntax
            when an endpoint requires a different token field.

    """

    messages: list[LLMMessage]
    response_format: dict[str, Any] | None = None  # JSON schema hint
    temperature: float = 0.2
    max_tokens: int | None = None


@dataclass
class LLMResponse:
    """Normalized response text plus an optional raw JSON payload.

    Attributes:
        content: Text consumed by pipeline parsers.
        raw: Original provider mapping for debugging or provider-specific use;
            it may be ``None`` for minimal implementations.

    """

    content: str
    raw: dict[str, Any] | None = None


class LLMProvider(abc.ABC):
    """Common LLM interface.

    Implementations must be safe about logging: never include API keys or
    full script text in exception messages. Use the ``secret_store`` summary
    for diagnostics.

    Attributes:
        name: Stable provider identifier used in diagnostics and tests.

    """

    name: str = "abstract"

    @abc.abstractmethod
    async def chat(self, request: LLMRequest) -> LLMResponse:
        """Send one chat request and return normalized text.

        Args:
            request: Vendor-neutral messages and generation controls.

        Returns:
            A normalized ``LLMResponse``.

        Raises:
            ProviderError: Implementations should convert upstream or protocol
                failures into a safe provider exception.

        """
        ...

    async def chat_json(self, request: LLMRequest) -> dict[str, Any]:
        """Request and parse JSON, tolerating models that wrap it in prose.

        Args:
            request: Request whose response should contain a JSON object.

        Returns:
            The parsed JSON mapping returned by ``chat`` or recovered from a
            surrounding markdown fence/wider object region.

        Raises:
            json.JSONDecodeError: If the response is not recoverable JSON.

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

    Attributes:
        safe: Whether ``str(error)`` is suitable for a user-facing response.
        original: Optional underlying exception retained for server-side
            diagnostics and exception chaining.

    """

    def __init__(self, message: str, *, safe: bool = True, original: Exception | None = None) -> None:
        """Create an error with a user-safe message and optional original cause."""
        super().__init__(message)
        self.safe = safe
        self.original = original


async def safe_chat_json(provider: LLMProvider, request: LLMRequest) -> dict[str, Any]:
    """Run ``chat_json`` and normalize unexpected failures.

    Args:
        provider: LLM implementation to call.
        request: JSON-oriented request passed to that provider.

    Returns:
        The parsed JSON mapping from the provider.

    Raises:
        ProviderError: Re-raises an existing provider error or wraps any other
            exception as a safe provider error while retaining its cause.

    """
    try:
        return await provider.chat_json(request)
    except ProviderError:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        raise ProviderError(str(exc), safe=True, original=exc) from exc
