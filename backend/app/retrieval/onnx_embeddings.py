"""Optional offline E5 encoder: pinned data, CPU, masked mean pooling, no remote code."""
from __future__ import annotations

import asyncio
import hashlib
import threading
from pathlib import Path

from app.operations.limits import MAX_INDEX_DOCUMENTS
from app.retrieval.contracts import EmbeddingProfile
from app.retrieval.serialization import RetrievalError
from app.retrieval.vectors import normalize_vector


def file_hash(path: Path) -> str:
    try:
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()
    except OSError:
        raise RetrievalError("embedding_assets_unavailable") from None


def file_signature(path: Path) -> tuple[int, int]:
    try:
        value = path.stat()
        return value.st_size, value.st_mtime_ns
    except OSError:
        raise RetrievalError("embedding_assets_unavailable") from None


class OnnxEmbeddingAdapter:
    """One model instance, two CPU threads; optional packages imported only here."""

    def __init__(self, profile: EmbeddingProfile, directory: Path) -> None:
        if (profile.transport != "local-onnx-e5-v1" or not profile.tokenizer_sha256
                or profile.dimensions != 384 or profile.document_prefix != "passage: "
                or profile.query_prefix != "query: "):
            raise RetrievalError("embedding_profile_mismatch")
        self.profile = profile
        self._weights, self._tokenizer_path = directory / "model.onnx", directory / "tokenizer.json"
        if (file_hash(self._weights) != profile.weights_sha256
                or file_hash(self._tokenizer_path) != profile.tokenizer_sha256):
            raise RetrievalError("embedding_assets_mismatch")
        self._signatures = (file_signature(self._weights), file_signature(self._tokenizer_path))
        try:
            import numpy as np
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except ImportError:
            raise RetrievalError("embedding_runtime_unavailable") from None
        self._np = np
        try:
            self._tokenizer = Tokenizer.from_file(str(self._tokenizer_path))
            self._tokenizer.no_truncation()
            self._tokenizer.enable_padding(pad_id=1, pad_token="<pad>")
            options = ort.SessionOptions()
            options.intra_op_num_threads = 2
            options.inter_op_num_threads = 1
            options.log_severity_level = 3
            self._session = ort.InferenceSession(str(self._weights), sess_options=options,
                                                 providers=["CPUExecutionProvider"])
        except Exception:
            raise RetrievalError("embedding_runtime_failed") from None
        self._lock = threading.Lock()
        self.calls = 0

    def _encode(self, texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
        with self._lock:
            if self._signatures != (file_signature(self._weights), file_signature(self._tokenizer_path)):
                raise RetrievalError("embedding_assets_changed")
            self.calls += 1
            try:
                encodings = self._tokenizer.encode_batch(list(texts))
                if any(len(e.ids) > 512 for e in encodings):
                    raise RetrievalError("embedding_token_limit")
                np = self._np
                ids = np.asarray([e.ids for e in encodings], dtype=np.int64)
                mask = np.asarray([e.attention_mask for e in encodings], dtype=np.int64)
                supplied = {"input_ids": ids, "attention_mask": mask,
                            "token_type_ids": np.zeros_like(ids)}
                inputs = {item.name: supplied[item.name] for item in self._session.get_inputs()}
                hidden = self._session.run(None, inputs)[0]
                if hidden.shape != (*ids.shape, self.profile.dimensions):
                    raise RetrievalError("invalid_embedding_response")
                masked = np.where(mask[..., None].astype(bool), hidden, 0.0)
                pooled = masked.sum(axis=1) / mask.sum(axis=1)[:, None]
                return tuple(normalize_vector(v.tolist(), self.profile.dimensions) for v in pooled)
            except RetrievalError:
                raise
            except Exception:
                raise RetrievalError("embedding_runtime_failed") from None

    async def embed_documents(self, texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
        if not texts or len(texts) > MAX_INDEX_DOCUMENTS or any(not isinstance(t, str) or not t.strip() or len(t.encode()) > 1200 for t in texts):
            raise RetrievalError("invalid_embedding_input")
        result: list[tuple[float, ...]] = []
        for offset in range(0, len(texts), 8):
            batch = tuple(self.profile.document_prefix + t for t in texts[offset:offset + 8])
            result.extend(await asyncio.to_thread(self._encode, batch))
        return tuple(result)

    async def embed_query(self, text: str) -> tuple[float, ...]:
        if not isinstance(text, str) or not text.strip() or len(text.encode()) > 1200:
            raise RetrievalError("query_too_long_or_empty")
        return (await asyncio.to_thread(self._encode, (self.profile.query_prefix + text,)))[0]

    async def __aenter__(self) -> OnnxEmbeddingAdapter:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        pass  # No sockets/network lifetime. Application may reuse this CPU session.
