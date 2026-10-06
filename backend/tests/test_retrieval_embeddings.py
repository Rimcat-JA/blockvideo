"""Local transport validates identity, row order and finite vectors before use."""
from __future__ import annotations

import json

import httpx
import pytest

from app.retrieval.contracts import EmbeddingProfile
from app.retrieval.embeddings import LocalEmbeddingAdapter, local_url
from app.retrieval.serialization import RetrievalError


def profile() -> EmbeddingProfile:
    return EmbeddingProfile(model="test-embedding", weights_sha256="a" * 64, dimensions=2,
                            document_prefix="search_document: ", query_prefix="search_query: ")


def body() -> dict:
    return {"model": "test-embedding", "data": [{"index": 0, "embedding": [3.0, 4.0]}]}


async def test_embeddings_ordered_normalized_prefixed_and_batched() -> None:
    sent = []

    def respond(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        sent.append(payload)
        return httpx.Response(200, json={"model": "test-embedding", "data": [
            {"index": i, "embedding": [i + 1, 1]} for i in reversed(range(len(payload["input"])))
        ]})

    async with LocalEmbeddingAdapter(profile(), "http://127.0.0.1:1234/v1",
                                      transport=httpx.MockTransport(respond)) as adapter:
        vectors = await adapter.embed_documents(tuple(f"日本語の合成例{i}" for i in range(9)))
        assert adapter.calls == 2
    assert len(vectors) == 9 and vectors[0] == vectors[8]
    assert vectors[0][0] == vectors[0][1] and vectors[1][0] > vectors[1][1]
    assert all(text.startswith("search_document: ") for p in sent for text in p["input"])
    assert all(set(p) == {"model", "input", "encoding_format"} for p in sent)


@pytest.mark.parametrize("mutate,code", [
    (lambda b: b.update(model="wrong"), "embedding_model_mismatch"),
    (lambda b: b.pop("model"), "embedding_model_mismatch"),
    (lambda b: b.update(data=[]), "embedding_count_mismatch"),
    (lambda b: b["data"][0].update(index=True), "embedding_index_mismatch"),
    (lambda b: b["data"][0].update(index=3), "embedding_index_mismatch"),
    (lambda b: b["data"][0].update(embedding=[0, 0]), "invalid_vector"),
    (lambda b: b["data"][0].update(embedding=[True, 1]), "invalid_vector"),
    (lambda b: b["data"][0].update(embedding=["3", "4"]), "invalid_vector"),
    (lambda b: b["data"][0].update(embedding=[1, 2, 3]), "invalid_vector"),
])
async def test_invalid_embedding_response_never_yields_vectors(mutate, code) -> None:
    raw = body()
    mutate(raw)
    async with LocalEmbeddingAdapter(profile(), "http://localhost:1234/v1",
        transport=httpx.MockTransport(lambda _r: httpx.Response(200, json=raw))) as adapter:
        with pytest.raises(RetrievalError, match=code):
            await adapter.embed_documents(("合成例",))
        assert adapter.calls == 1


@pytest.mark.parametrize("status", [302, 400, 429, 500])
async def test_http_failure_is_bounded_and_not_retried(status) -> None:
    async with LocalEmbeddingAdapter(profile(), "http://localhost:1234/v1",
        transport=httpx.MockTransport(lambda _r: httpx.Response(status, text="private echoed input",
                                                               headers={"Location": "https://example.com"}))) as adapter:
        with pytest.raises(RetrievalError) as exc:
            await adapter.embed_documents(("合成例",))
        assert str(exc.value) == "embedding_http_error" and adapter.calls == 1


@pytest.mark.parametrize("content", [
    b'{"model":"test-embedding","model":"wrong","data":[]}',
    b'{"model":"test-embedding","data":[{"index":0,"embedding":[NaN,1]}]}',
    b'{"model":"test-embedding","data":[{"index":0,"embedding":[1e999,1]}]}',
    b'null', b'{', b'x' * 2_000_001,
], ids=["duplicate-key", "nan", "overflow", "null", "incomplete", "oversized"])
async def test_invalid_json_nonfinite_or_oversized_response(content) -> None:
    async with LocalEmbeddingAdapter(profile(), "http://localhost:1234/v1",
        transport=httpx.MockTransport(lambda _r: httpx.Response(200, content=content))) as adapter:
        with pytest.raises(RetrievalError):
            await adapter.embed_documents(("合成例",))


@pytest.mark.parametrize("url", ["https://example.com/v1", "http://127.0.0.1:1234/v1?key=secret",
    "http://user:secret@localhost:1234/v1", "http://localhost:1234/other", "http://127.0.0.1/v1"])
def test_remote_credential_or_ambiguous_endpoint_rejected(url) -> None:
    with pytest.raises(RetrievalError, match="invalid_endpoint"):
        local_url(url)


async def test_later_batch_failure_does_not_return_partial_success() -> None:
    calls = 0

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise httpx.ReadTimeout("private echoed input", request=request)
        return httpx.Response(200, json={"model": "test-embedding", "data": [
            {"index": i, "embedding": [3, 4]} for i in range(8)]})

    async with LocalEmbeddingAdapter(profile(), "http://localhost:1234/v1",
                                      transport=httpx.MockTransport(respond)) as adapter:
        with pytest.raises(RetrievalError, match="embedding_timeout"):
            await adapter.embed_documents(tuple("合成" for _ in range(9)))
        assert calls == 2
