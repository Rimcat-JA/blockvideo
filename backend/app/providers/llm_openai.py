"""HTTP implementation for OpenAI-compatible chat-completions APIs.

Works with any endpoint that implements the chat.completions shape. We do
not depend on the OpenAI SDK so that local proxies (Ollama, LM Studio,
vLLM, ...) work out of the box.
Imports:
    ``Any`` types decoded provider JSON.
    ``httpx`` performs asynchronous HTTP requests.
    ``redact`` protects short upstream error previews.
    LLM transport/error classes define the application-facing contract.
"""
from __future__ import annotations

from typing import Any

import httpx

from app.core.security import redact
from app.providers.llm import LLMProvider, LLMRequest, LLMResponse, ProviderError


class OpenAICompatibleProvider(LLMProvider):
    """Async client for APIs implementing ``/chat/completions``.

    Attributes:
        name: Stable factory identifier.
        _api_key: Raw credential retained only for outgoing authorization.
        _base_url, _model: Endpoint and model selected for requests.
        _extra_headers: Optional attribution/provider headers.
        _use_max_completion_tokens: Latched compatibility flag after a
            ``max_tokens`` rejection.
        _client: Injected or internally owned ``httpx.AsyncClient``.

    """

    name = "openai_compatible"

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        *,
        timeout: float = 300.0,
        client: httpx.AsyncClient | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        """Configure credentials, endpoint, model, and HTTP ownership.

        Args:
            api_key: Bearer credential used for the endpoint.
            base_url: OpenAI-compatible API base URL.
            model: Model identifier forwarded in every payload.
            timeout: Timeout for an internally created HTTP client.
            client: Optional injected client for tests or shared lifecycle.
            extra_headers: Optional additional request headers.

        Raises:
            ProviderError: If the key, base URL, or model is empty.

        """
        if not api_key:
            raise ProviderError("LLM API key is required", safe=True)
        if not base_url:
            raise ProviderError("LLM base URL is required", safe=True)
        if not model:
            raise ProviderError("LLM model name is required", safe=True)
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._extra_headers = dict(extra_headers or {})
        # Flipped on the first request the endpoint rejects for using
        # ``max_tokens`` (OpenAI's GPT-5 family). See ``_rejects_max_tokens``.
        self._use_max_completion_tokens = False
        self._client = client or httpx.AsyncClient(timeout=timeout)

    def _build_payload(self, request: LLMRequest) -> tuple[dict[str, Any], str | None]:
        """Translate an ``LLMRequest`` into an API JSON payload.

        Args:
            request: Vendor-neutral messages and generation controls.

        Returns:
            ``(payload, token_key)`` where ``token_key`` records whether this
            request used ``max_tokens`` or ``max_completion_tokens``.

        """
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            "temperature": request.temperature,
            "stream": False,
        }
        token_key: str | None = None
        if request.max_tokens is not None:
            token_key = (
                "max_completion_tokens" if self._use_max_completion_tokens else "max_tokens"
            )
            payload[token_key] = request.max_tokens
        if request.response_format is not None:
            payload["response_format"] = request.response_format
        return payload, token_key

    @staticmethod
    def _rejects_max_tokens(status: int, body: str) -> bool:
        """Return whether the endpoint wants ``max_completion_tokens`` instead.

        Args:
            status: HTTP status code returned by the endpoint.
            body: Short response body inspected for both token-field names.

        Returns:
            ``True`` only for the observed compatibility error shape.

        OpenAI's GPT-5 family dropped ``max_tokens``; other OpenAI-compatible
        endpoints still require it. Rather than hard-coding a model list that
        goes stale, detect the rejection once and remember the answer.

        """
        return status == 400 and "max_tokens" in body and "max_completion_tokens" in body

    async def chat(self, request: LLMRequest) -> LLMResponse:
        """POST a chat request and normalize the first choice's content.

        Args:
            request: Chat messages and optional JSON/token controls.

        Returns:
            ``LLMResponse`` containing the provider content and raw JSON.

        Raises:
            ProviderError: For transport errors, HTTP errors, or malformed
                response shapes.  Error previews are redacted and truncated.

        """
        url = f"{self._base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            **self._extra_headers,
        }

        async def _post(payload: dict[str, Any]) -> httpx.Response:
            try:
                return await self._client.post(url, json=payload, headers=headers)
            except httpx.HTTPError as exc:
                raise ProviderError(
                    f"LLM接続に失敗しました: {exc.__class__.__name__}", safe=True, original=exc
                ) from exc

        payload, token_key = self._build_payload(request)
        response = await _post(payload)
        # Decide on what *this* request sent, not on the current flag: the
        # provider is shared across concurrent calls, so a sibling coroutine
        # may have latched the flag between this request and its response.
        if token_key == "max_tokens" and self._rejects_max_tokens(
            response.status_code, response.text or ""
        ):
            # Latch so subsequent calls skip the probe.
            self._use_max_completion_tokens = True
            retry_payload = dict(payload)
            retry_payload.pop("max_tokens", None)
            retry_payload["max_completion_tokens"] = request.max_tokens
            response = await _post(retry_payload)

        if response.status_code >= 400:
            # never include the API key in the error message
            text_preview = response.text[:200] if response.text else ""
            redacted = redact(text_preview)
            raise ProviderError(
                f"LLMリクエストエラー (status {response.status_code}): {redacted}",
                safe=True,
            )
        data = response.json()
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError("LLMレスポンス形式が不正です", safe=True, original=exc) from exc
        if not isinstance(content, str):
            content = str(content)
        return LLMResponse(content=content, raw=data)

    async def aclose(self) -> None:
        """Close the underlying HTTP client when this provider owns it.

        Injected clients are also closed by the current implementation; callers
        should therefore treat a provider as owning its client after passing it
        to the constructor.
        """
        if self._client is not None:
            await self._client.aclose()
