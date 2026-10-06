"""Lazy local configuration; constructing this object does not load an index/model."""
from __future__ import annotations

import asyncio
import threading
from functools import lru_cache
from pathlib import Path

from pydantic import ValidationError

from app.retrieval.contracts import EmbeddingProfile
from app.retrieval.embeddings import LocalEmbeddingAdapter
from app.retrieval.onnx_embeddings import OnnxEmbeddingAdapter
from app.retrieval.reader import load_index
from app.retrieval.serialization import RetrievalError, read_bytes
from app.retrieval.sources import load_sources
from app.semantic_interpretation.service import SemanticInterpreter

_ENCODER_LOCK = threading.Lock()


@lru_cache(maxsize=1)
def _cached_encoder(profile: EmbeddingProfile, directory: Path) -> OnnxEmbeddingAdapter:
    return OnnxEmbeddingAdapter(profile, directory)


def _encoder(profile: EmbeddingProfile, directory: Path) -> OnnxEmbeddingAdapter:
    # Serialize first load too; functools alone permits concurrent cache misses.
    with _ENCODER_LOCK:
        return _cached_encoder(profile, directory)


def configured_semantic(index_path: Path, profile_path: Path, assets_path: Path,
                        base_url: str, *, allow_all_tools: bool = True) -> SemanticInterpreter:
    @lru_cache(maxsize=1)
    def profile() -> EmbeddingProfile:
        try:
            return EmbeddingProfile.model_validate_json(read_bytes(profile_path, 16_000))
        except ValidationError:
            raise RetrievalError("invalid_profile") from None

    class Encoder:
        async def embed_query(self, text: str) -> tuple[float, ...]:
            value = profile()
            if value.transport == "local-onnx-e5-v1":
                adapter = await asyncio.to_thread(_encoder, value, assets_path)
                return await adapter.embed_query(text)
            async with LocalEmbeddingAdapter(value, base_url) as http_adapter:
                return await http_adapter.embed_query(text)

    return SemanticInterpreter(lambda source: load_index(index_path, source, profile()),
        load_sources, Encoder(), allow_all_tools=allow_all_tools)
