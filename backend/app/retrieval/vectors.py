"""Finite full-dimensional vectors with an explicit, shared normalization rule."""
from __future__ import annotations

import math

from app.retrieval.serialization import RetrievalError


def normalize_vector(value: object, dimensions: int) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != dimensions:
        raise RetrievalError("invalid_vector")
    if any(type(x) not in (int, float) or not math.isfinite(x) for x in value):
        raise RetrievalError("invalid_vector")
    norm = math.hypot(*value)
    if not math.isfinite(norm) or norm < 1e-12:
        raise RetrievalError("invalid_vector")
    return tuple(float(x / norm) for x in value)


def validate_vectors(vectors: tuple[tuple[float, ...], ...], count: int, dimensions: int) -> None:
    if len(vectors) != count:
        raise RetrievalError("vector_count_mismatch")
    for vector in vectors:
        normalize_vector(vector, dimensions)
        if not math.isclose(math.hypot(*vector), 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise RetrievalError("vector_not_normalized")
