"""Offline E5 asset identity, query/document prefixes and masked pooling."""
from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.retrieval.contracts import EmbeddingProfile
from app.retrieval.onnx_embeddings import OnnxEmbeddingAdapter, file_hash, file_signature
from app.retrieval.serialization import RetrievalError


@pytest.fixture
def profile() -> EmbeddingProfile:
    return EmbeddingProfile.model_validate_json((Path(__file__).parents[1] / "app/retrieval/e5-profile.json").read_text())


def test_wrong_asset_bytes_are_rejected_before_runtime_loading(tmp_path: Path, profile: EmbeddingProfile) -> None:
    (tmp_path / "model.onnx").write_bytes(b"not-the-model")
    (tmp_path / "tokenizer.json").write_bytes(b"{}")
    with pytest.raises(RetrievalError, match="embedding_assets_mismatch"):
        OnnxEmbeddingAdapter(profile, tmp_path)


@pytest.mark.parametrize("change", [{"dimensions": 768}, {"query_prefix": "wrong"}, {"tokenizer_sha256": None}], ids=["dimensions", "prefix", "tokenizer"])
def test_wrong_profile_rejected(tmp_path: Path, profile: EmbeddingProfile, change: dict) -> None:
    with pytest.raises(RetrievalError, match="embedding_profile_mismatch"):
        OnnxEmbeddingAdapter(profile.model_copy(update=change), tmp_path)


def test_tokenizer_hash_is_also_verified(tmp_path: Path, profile: EmbeddingProfile) -> None:
    (tmp_path / "model.onnx").write_bytes(b"test")
    (tmp_path / "tokenizer.json").write_bytes(b"changed")
    profile = profile.model_copy(update={"weights_sha256": file_hash(tmp_path / "model.onnx")})
    with pytest.raises(RetrievalError, match="embedding_assets_mismatch"):
        OnnxEmbeddingAdapter(profile, tmp_path)


@pytest.fixture
def encoder(tmp_path: Path, profile: EmbeddingProfile):
    np = pytest.importorskip("numpy")
    result = OnnxEmbeddingAdapter.__new__(OnnxEmbeddingAdapter)
    result.profile = profile
    result._np, result._lock, result.calls = np, threading.Lock(), 0
    result._weights, result._tokenizer_path = tmp_path / "model", tmp_path / "tokenizer"
    result._weights.write_bytes(b"fixed")
    result._tokenizer_path.write_bytes(b"fixed")
    result._signatures = (file_signature(result._weights), file_signature(result._tokenizer_path))
    seen = []
    def encode(texts: list[str]):
        seen.extend(texts)
        return [SimpleNamespace(ids=[0, 42, 1], attention_mask=[1, 1, 0]) for _ in texts]
    result._tokenizer = SimpleNamespace(encode_batch=encode)
    def infer(_output, inputs):
        size = len(inputs["input_ids"])
        hidden = np.zeros((size, 3, 384))
        hidden[:, 0, 0] = 2
        hidden[:, 1, 1] = 2
        hidden[:, 2, :] = 999  # padding must be ignored
        return [hidden]
    result._session = SimpleNamespace(run=infer, get_inputs=lambda: [SimpleNamespace(name=n) for n in ("input_ids", "attention_mask")])
    return result, seen


async def test_mean_pool_mask_normalization_and_prefixes(encoder) -> None:
    adapter, seen = encoder
    query = await adapter.embed_query("字幕を大きく")
    documents = await adapter.embed_documents(("説明", "例文"))
    assert seen == ["query: 字幕を大きく", "passage: 説明", "passage: 例文"]
    assert query[:2] == pytest.approx((2 ** -0.5, 2 ** -0.5))
    assert sum(query[2:]) == 0 and documents == (query, query)
    assert adapter.calls == 2


async def test_changed_assets_stop_even_with_cached_runtime(encoder) -> None:
    adapter, _ = encoder
    adapter._weights.write_bytes(b"different bytes")
    with pytest.raises(RetrievalError, match="embedding_assets_changed"):
        await adapter.embed_query("test")


async def test_token_limit_has_no_silent_truncation(encoder) -> None:
    adapter, _ = encoder
    adapter._tokenizer.encode_batch = lambda _texts: [SimpleNamespace(ids=[1] * 513)]
    with pytest.raises(RetrievalError, match="embedding_token_limit"):
        await adapter.embed_query("test")


async def test_unexpected_tensor_shape_rejected(encoder) -> None:
    adapter, _ = encoder
    adapter._session.run = lambda *_: [adapter._np.zeros((1, 384))]
    with pytest.raises(RetrievalError, match="invalid_embedding_response"):
        await adapter.embed_query("test")


@pytest.mark.parametrize("value", [None, "", "a" * 1201], ids=["not-string", "empty", "too-long"])
async def test_invalid_query_never_reaches_encoder(encoder, value: object) -> None:
    adapter, seen = encoder
    with pytest.raises(RetrievalError, match="query_too_long_or_empty"):
        await adapter.embed_query(value)
    assert not seen


async def test_document_batch_limit_matches_the_index_limit(encoder) -> None:
    from app.operations.limits import MAX_INDEX_DOCUMENTS

    adapter, _ = encoder
    # A 1,000-operation catalog has more than 4,096 documents; the index accepts them.
    vectors = await adapter.embed_documents(tuple(f"文書{i}" for i in range(5000)))
    assert len(vectors) == 5000
    with pytest.raises(RetrievalError, match="invalid_embedding_input"):
        await adapter.embed_documents(tuple("x" for _ in range(MAX_INDEX_DOCUMENTS + 1)))
