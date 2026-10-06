"""Local embeddings only: bounded vectors out, no tools or filesystem access."""
from __future__ import annotations

import asyncio
from urllib.parse import urlsplit

import httpx

from app.operations.limits import MAX_INDEX_DOCUMENTS
from app.retrieval.contracts import EmbeddingProfile
from app.retrieval.serialization import RetrievalError, decode
from app.retrieval.vectors import normalize_vector


def local_url(value: str) -> str:
    try:
        parts = urlsplit(value)
        if (parts.scheme != "http" or parts.hostname not in {"127.0.0.1", "localhost", "::1"}
                or parts.username is not None or parts.password is not None
                or parts.query or parts.fragment or parts.path.rstrip("/") != "/v1"
                or parts.port is None or not 1 <= parts.port <= 65535):
            raise ValueError()
        host = "[::1]" if parts.hostname == "::1" else "127.0.0.1"
        return f"http://{host}:{parts.port}/v1"
    except (ValueError, TypeError, AttributeError):
        raise RetrievalError("invalid_endpoint") from None


class LocalEmbeddingAdapter:
    def __init__(self, profile: EmbeddingProfile, base_url: str, *, timeout_seconds: float = 60,
                 transport: httpx.AsyncBaseTransport | None = None) -> None:
        if type(timeout_seconds) not in (int, float) or not 1 <= timeout_seconds <= 120:
            raise RetrievalError("invalid_timeout")
        self.profile = EmbeddingProfile.model_validate(profile.model_dump())
        self.base_url = local_url(base_url)
        self.timeout_seconds = timeout_seconds
        self.calls = 0
        self._client = httpx.AsyncClient(timeout=timeout_seconds, trust_env=False,
                                         follow_redirects=False, transport=transport)

    async def embed_documents(self, texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
        """Sequential batches; a failure never returns or publishes partial vectors."""
        if not texts or len(texts) > MAX_INDEX_DOCUMENTS or any(
            not isinstance(t, str) or not t.strip() or len(t.encode("utf-8")) > 1200 for t in texts
        ):
            raise RetrievalError("invalid_embedding_input")
        result: list[tuple[float, ...]] = []
        for offset in range(0, len(texts), 8):
            batch = tuple(self.profile.document_prefix + t for t in texts[offset:offset + 8])
            result.extend(await self._batch(batch))
        return tuple(result)

    async def embed_query(self, text: str) -> tuple[float, ...]:
        """Use the model's query prefix; never silently truncate an instruction."""
        if not isinstance(text, str) or not text.strip() or len(text.encode("utf-8")) > 1200:
            raise RetrievalError("query_too_long_or_empty")
        return (await self._batch((self.profile.query_prefix + text,)))[0]

    async def _batch(self, texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
        self.calls += 1
        try:
            async with asyncio.timeout(self.timeout_seconds):
                async with self._client.stream("POST", f"{self.base_url}/embeddings", json={
                    "model": self.profile.model, "input": list(texts), "encoding_format": "float",
                }) as response:
                    if response.status_code != 200:
                        raise RetrievalError("embedding_http_error")
                    data = bytearray()
                    async for chunk in response.aiter_bytes():
                        data.extend(chunk)
                        if len(data) > 2_000_000:
                            raise RetrievalError("embedding_response_too_large")
        except (TimeoutError, httpx.TimeoutException):
            raise RetrievalError("embedding_timeout") from None
        except httpx.HTTPError:
            raise RetrievalError("embedding_connection_failed") from None
        try:
            body = decode(bytes(data))
            if body.get("model") != self.profile.model:
                raise RetrievalError("embedding_model_mismatch")
            rows = body["data"]
            if not isinstance(rows, list) or len(rows) != len(texts):
                raise RetrievalError("embedding_count_mismatch")
            ordered: dict[int, tuple[float, ...]] = {}
            for row in rows:
                index = row["index"]
                if type(index) is not int or not 0 <= index < len(texts) or index in ordered:
                    raise RetrievalError("embedding_index_mismatch")
                ordered[index] = normalize_vector(row["embedding"], self.profile.dimensions)
            return tuple(ordered[i] for i in range(len(texts)))
        except (KeyError, TypeError, AttributeError, OverflowError):
            raise RetrievalError("invalid_embedding_response") from None

    async def __aenter__(self) -> LocalEmbeddingAdapter:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self._client.aclose()
