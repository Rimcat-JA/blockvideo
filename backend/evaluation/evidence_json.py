"""Bounded canonical durable JSON validation, independent of evidence schemas."""
from __future__ import annotations

import json
import math
from typing import TypeVar

from pydantic import BaseModel, TypeAdapter

from evaluation.tool_attestation import canonical_json_bytes

TModel = TypeVar("TModel", bound=BaseModel)
TValue = TypeVar("TValue")
_MAX_DEPTH = 32
_MAX_TOKENS = 8_000_000
_MAX_STRING_BYTES = 8192
_WHITESPACE = b" \t\r\n"
_DELIMITERS = b'{}[],:" \t\r\n'


def _check_lexical_limits(raw: bytes) -> None:
    depth = 0
    tokens = 0
    position = 0
    while position < len(raw):
        char = raw[position]
        if char in _WHITESPACE:
            position += 1
            continue
        tokens += 1
        if tokens > _MAX_TOKENS:
            raise ValueError("evidence token limit exceeded")
        start = position
        position += 1
        if char == 34:
            while position < len(raw):
                char = raw[position]
                position += 1
                if char == 92:
                    position += 1
                elif char == 34:
                    break
                if position - start > _MAX_STRING_BYTES:
                    raise ValueError("evidence string limit exceeded")
            else:
                raise ValueError("invalid evidence JSON")
            if position - start > _MAX_STRING_BYTES:
                raise ValueError("evidence string limit exceeded")
        elif char in b"{[":
            depth += 1
            if depth > _MAX_DEPTH:
                raise ValueError("evidence structural limit exceeded")
        elif char in b"}]":
            depth -= 1
            if depth < 0:
                raise ValueError("invalid evidence JSON")
        elif char not in b",:":
            while position < len(raw) and raw[position] not in _DELIMITERS:
                position += 1
    if depth:
        raise ValueError("invalid evidence JSON")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("invalid evidence JSON")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise ValueError("invalid evidence JSON")


def _check_values(value: object) -> None:
    if isinstance(value, str):
        value.encode("utf-8", errors="strict")
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("invalid evidence JSON")
    elif isinstance(value, dict):
        for key, item in value.items():
            _check_values(key)
            _check_values(item)
    elif isinstance(value, list):
        for item in value:
            _check_values(item)


def _check_canonical_raw(raw: bytes, maximum: int) -> None:
    """Shared pre-model checks: size, lexical limits, duplicates, nonfinite, canonical LF bytes."""
    if type(raw) is not bytes or type(maximum) is not int or maximum < 1:
        raise ValueError("invalid evidence JSON")
    if len(raw) > maximum:
        raise ValueError("evidence size limit exceeded")
    _check_lexical_limits(raw)
    try:
        text = raw.decode("utf-8", errors="strict")
        decoded = json.loads(
            text, object_pairs_hook=_unique_object, parse_constant=_reject_constant,
        )
        _check_values(decoded)
        canonical = canonical_json_bytes(decoded) + b"\n"
    except (ValueError, TypeError, UnicodeError, OverflowError, RecursionError):
        raise ValueError("invalid evidence JSON") from None
    if canonical != raw:
        raise ValueError("evidence is not canonical")


def parse_canonical_typed(
    raw: bytes, adapter: TypeAdapter[TValue], *, maximum: int,
) -> TValue:
    """Strict canonical parse for non-model evidence such as tuples of models.

    Shares every check with `parse_canonical_model`; the strictly validated value
    must serialize back to exactly the same canonical LF-terminated bytes.
    """
    _check_canonical_raw(raw, maximum)
    try:
        value = adapter.validate_json(raw, strict=True)
        canonical = canonical_json_bytes(adapter.dump_python(value, mode="json")) + b"\n"
    except (ValueError, TypeError, UnicodeError, OverflowError, RecursionError):
        raise ValueError("evidence model mismatch") from None
    if canonical != raw:
        raise ValueError("evidence model mismatch")
    return value


def parse_canonical_model(
    raw: bytes, model_type: type[TModel], *, maximum: int,
) -> TModel:
    """Reject noncanonical or unsafe evidence without including input in errors."""
    _check_canonical_raw(raw, maximum)
    try:
        model = model_type.model_validate_json(raw, strict=True)
        canonical = canonical_json_bytes(model) + b"\n"
    except (ValueError, TypeError, UnicodeError, OverflowError, RecursionError):
        raise ValueError("evidence model mismatch") from None
    if canonical != raw:
        raise ValueError("evidence model mismatch")
    return model
