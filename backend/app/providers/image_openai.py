"""HTTP implementation for the OpenAI Images generations endpoint.

Imports:
    ``base64`` decodes the API's ``b64_json`` image payload.
    ``Path`` identifies the PNG destination.
    ``httpx`` performs asynchronous HTTP requests.
    Security/provider modules redact errors and expose the shared interface.
"""
from __future__ import annotations

import base64
from pathlib import Path

import httpx

from app.core.security import redact
from app.providers.image import ImageProvider
from app.providers.llm import ProviderError


class OpenAIImageProvider(ImageProvider):
    """Async Images API client that writes decoded bytes to disk.

    Attributes:
        name: Stable factory identifier.
        _api_key, _model, _base_url: Request configuration.
        _client: HTTP client used for image requests.

    """

    name = "openai"

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-image-1",
        *,
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 120.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        """Configure the Images endpoint and HTTP client.

        Args:
            api_key: Bearer credential required by the API.
            model: Image model identifier.
            base_url: API base URL, normally ending in ``/v1``.
            timeout: Timeout for an internally created client.
            client: Optional injected asynchronous client.

        Raises:
            ProviderError: If ``api_key`` is empty.

        """
        if not api_key:
            raise ProviderError("Image API key is required", safe=True)
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._client = client or httpx.AsyncClient(timeout=timeout)

    async def generate_image(
        self, prompt: str, width: int, height: int, output_path: Path
    ) -> Path:
        """Request one base64 image and persist the decoded bytes.

        Args:
            prompt: Text description submitted to the image model.
            width: Desired output width used to choose an API aspect ratio.
            height: Desired output height used to choose an API aspect ratio.
            output_path: Destination path, created with parent directories.

        Returns:
            The supplied output path after writing the response bytes.

        Raises:
            ProviderError: On transport, HTTP, or malformed base64-response
                errors.  Messages contain only redacted short previews.

        """
        url = f"{self._base_url}/images/generations"
        payload = {
            "model": self._model,
            "prompt": prompt,
            "size": self._size_for_aspect(width, height),
            "n": 1,
            "response_format": "b64_json",
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        try:
            response = await self._client.post(url, json=payload, headers=headers)
        except httpx.HTTPError as exc:
            raise ProviderError(
                f"画像API接続に失敗しました: {exc.__class__.__name__}",
                safe=True,
                original=exc,
            ) from exc
        if response.status_code >= 400:
            text_preview = response.text[:200] if response.text else ""
            raise ProviderError(
                f"画像APIエラー (status {response.status_code}): {redact(text_preview)}",
                safe=True,
            )
        try:
            data = response.json()
            b64 = data["data"][0]["b64_json"]
            raw = base64.b64decode(b64)
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ProviderError("画像APIレスポンスが不正です", safe=True, original=exc) from exc
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(raw)
        return output_path

    @staticmethod
    def _size_for_aspect(width: int, height: int) -> str:
        """Map dimensions to one of the API's supported aspect sizes."""
        if width >= height:
            return "1536x1024"
        return "1024x1536"

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        if self._client is not None:
            await self._client.aclose()
