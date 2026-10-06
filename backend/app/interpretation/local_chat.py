"""Loopback-only Chat Completions transport for D16 development inference."""
from __future__ import annotations

import asyncio
from typing import Any, Literal
from urllib.parse import urlsplit

import httpx

from app.interpretation.errors import InterpretationError
from app.interpretation.parser import strict_json
from app.interpretation.transport import ModelMessage

MAX_HTTP_RESPONSE_BYTES = 131_072


def local_base_url(value: str) -> str:
    """No remote host, embedded credential, query, proxy or redirect fallback."""
    try:
        parts = urlsplit(value)
        if (parts.scheme != "http" or parts.hostname not in {"127.0.0.1", "::1", "localhost"}
                or parts.username is not None or parts.password is not None
                or parts.query or parts.fragment or parts.path.rstrip("/") != "/v1"
                or parts.port is None or not 1 <= parts.port <= 65535):
            raise ValueError("invalid local URL")
        host = "[::1]" if parts.hostname == "::1" else "127.0.0.1"
        return f"http://{host}:{parts.port}/v1"
    except (ValueError, TypeError, AttributeError):
        raise InterpretationError("configuration_error") from None


class LocalChatAdapter:
    """Owns an HTTP client. No SDK, paid API, retry, tool execution or JSON repair."""

    def __init__(
        self, base_url: str, model: str, *, timeout_seconds: float = 120.0,
        max_tokens: int = 768, reasoning_effort: Literal["none"] | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = local_base_url(base_url)
        if (not isinstance(model, str) or not model.strip() or len(model) > 200
                or type(max_tokens) is not int or not 64 <= max_tokens <= 2048
                or type(timeout_seconds) not in {int, float}
                or not 1 <= timeout_seconds <= 180
                or reasoning_effort not in (None, "none")):
            raise InterpretationError("configuration_error")
        self.model = model
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds
        self.reasoning_effort = reasoning_effort
        self._client = httpx.AsyncClient(
            timeout=timeout_seconds, trust_env=False, follow_redirects=False,
            transport=transport,
        )

    async def complete(
        self, messages: tuple[ModelMessage, ...], schema: dict[str, Any],
    ) -> str:
        """Return only a completed assistant text; never retry uncertain requests."""
        payload = {
            "model": self.model,
            "messages": [{"role": message.role, "content": message.content} for message in messages],
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "blockvideo_proposal", "strict": True, "schema": schema,
            }},
            "temperature": 0, "max_tokens": self.max_tokens, "stream": False,
        }
        # Explicit endpoint capability, verified on the selected LM Studio model.
        # None preserves the server default; there is no compatibility retry.
        if self.reasoning_effort is not None:
            payload["reasoning_effort"] = self.reasoning_effort
        try:
            # httpx's read timeout alone resets for every chunk. Also bound
            # the complete request so a slow-drip response cannot wait forever.
            async with asyncio.timeout(self.timeout_seconds):
                async with self._client.stream(
                    "POST", f"{self.base_url}/chat/completions", json=payload,
                ) as response:
                    if response.status_code != 200:
                        raise InterpretationError("http_error", http_status=response.status_code)
                    data = bytearray()
                    async for chunk in response.aiter_bytes():
                        data.extend(chunk)
                        if len(data) > MAX_HTTP_RESPONSE_BYTES:
                            raise InterpretationError("response_too_large")
        except (httpx.TimeoutException, TimeoutError):
            raise InterpretationError("timeout") from None
        except httpx.HTTPError:
            raise InterpretationError("connection_failed") from None
        try:
            body = strict_json(data.decode("utf-8"))
            choices = body["choices"]
            if not isinstance(choices, list) or len(choices) != 1:
                raise ValueError("expected exactly one choice")
            choice = choices[0]
            message = choice["message"]
            if message.get("refusal"):
                raise InterpretationError("refused")
            if message.get("tool_calls") or message.get("function_call"):
                raise InterpretationError("invalid_response")
            if choice["finish_reason"] != "stop":
                raise InterpretationError("incomplete_response")
            if message.get("role") != "assistant" or not isinstance(message["content"], str):
                raise ValueError("expected assistant text")
            # Some local servers silently answer with a loaded model when the
            # requested identifier is absent. Never execute that substitution.
            if body.get("model") != self.model:
                raise InterpretationError("model_mismatch")
            return message["content"]
        except (ValueError, TypeError, KeyError, IndexError, AttributeError, RecursionError) as exc:
            if isinstance(exc, InterpretationError):
                raise
            raise InterpretationError("invalid_response") from None

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> LocalChatAdapter:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.aclose()
