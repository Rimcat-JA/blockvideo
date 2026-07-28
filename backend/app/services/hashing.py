"""Content hashing utilities.

A short, deterministic hash of the inputs that affect a particular pipeline
stage. Used to skip work that is already up-to-date.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any


def hash_value(*parts: Any, **kwargs: Any) -> str:
    """Return a stable SHA-256 over arbitrary picklable inputs."""
    parts = (*parts, *sorted(kwargs.items()))
    h = hashlib.sha256()
    for part in parts:
        if part is None:
            h.update(b"\x00")
        elif isinstance(part, str):
            h.update(b"str\x00")
            h.update(part.encode("utf-8"))
        elif isinstance(part, bytes):
            h.update(b"byt\x00")
            h.update(part)
        elif isinstance(part, bool):
            h.update(b"bool\x00")
            h.update(b"1" if part else b"0")
        elif isinstance(part, (int, float)):
            h.update(b"num\x00")
            h.update(str(part).encode("utf-8"))
        elif isinstance(part, dict):
            h.update(b"dic\x00")
            h.update(json.dumps(part, sort_keys=True, ensure_ascii=False).encode("utf-8"))
        elif isinstance(part, (list, tuple)):
            h.update(b"lst\x00")
            h.update(json.dumps(list(part), ensure_ascii=False).encode("utf-8"))
        else:
            h.update(b"rep\x00")
            h.update(repr(part).encode("utf-8"))
    return h.hexdigest()


def short_hash(*parts: Any) -> str:
    return hash_value(*parts)[:16]
