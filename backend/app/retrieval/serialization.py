"""Bounded JSON and fixed public failures; never echo model or file contents."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class RetrievalError(ValueError):
    """Fixed reason code suitable for a diagnostic, without private data."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate key")
        result[key] = value
    return result


def _constant(_value: str) -> None:
    raise ValueError("nonfinite constant")


def decode(data: bytes) -> Any:
    try:
        return json.loads(data.decode("utf-8"), object_pairs_hook=_unique,
                          parse_constant=_constant)
    except (ValueError, UnicodeError, RecursionError):
        raise RetrievalError("invalid_json") from None


def read_bytes(path: Path, limit: int) -> bytes:
    try:
        with path.open("rb") as stream:
            data = stream.read(limit + 1)
    except OSError:
        raise RetrievalError("file_unavailable") from None
    if len(data) > limit:
        raise RetrievalError("file_too_large")
    return data
