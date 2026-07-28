"""Deterministic content hashes used for stage cache keys.

A short, deterministic hash of the inputs that affect a particular pipeline
stage. Used to skip work that is already up-to-date.
Imports:
    ``hashlib`` provides SHA-256.
    ``json`` serializes dictionaries and sequences in a predictable form.
    ``Any`` permits the small set of supported input types.

The serializer is deliberately local and explicit.  Strings, bytes, booleans,
numbers, dictionaries, lists/tuples, ``None``, and a final ``repr`` fallback
are tagged before hashing so common values of different types do not collide
merely because their textual forms match.  The fallback is only deterministic
when the object's ``repr`` is deterministic.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any


def hash_value(*parts: Any, **kwargs: Any) -> str:
    """Return a tagged SHA-256 digest for supported values.

    Args:
        *parts: Ordered values contributing to the digest.
        **kwargs: Named values appended in sorted-key order, making keyword
            ordering irrelevant to the result.

    Returns:
        A 64-character hexadecimal SHA-256 digest.  JSON-compatible mappings
        and sequences are serialized deterministically; unsupported objects
        use their UTF-8 ``repr`` representation.

    Notes:
        This is an input/cache fingerprint, not a password hash or a universal
        serialization format.  Objects whose ``repr`` includes an address may
        therefore produce process-specific hashes.

    """
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
    """Return the first 16 hexadecimal characters of ``hash_value``.

    Args:
        *parts: Ordered values passed to ``hash_value``.

    Returns:
        A compact digest suitable for filenames, logs, and database cache keys.

    """
    return hash_value(*parts)[:16]
