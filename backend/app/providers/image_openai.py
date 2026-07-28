"""OpenAI Images API provider."""
from __future__ import annotations

import base64
from pathlib import Path

import httpx

from app.core.security import redact
from app.providers.image import ImageProvider
from app.providers.llm import ProviderError


class OpenAIImageProvider(ImageProvider):
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
        if not api_key:
            raise ProviderError("Image API key is required", safe=True)
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._client = client or httpx.AsyncClient(timeout=timeout)

    async def generate_image(
        self, prompt: str, width: int, height: int, output_path: Path
    ) -> Path:
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
        # OpenAI image API supports 1024x1024, 1024x1536, 1536x1024.
        if width >= height:
            return "1536x1024"
        return "1024x1536"

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()