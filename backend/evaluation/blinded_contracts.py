"""Strict blinded-evaluation protocol and approval contracts."""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import stat
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from evaluation.contracts import Case

MAX_PROTOCOL_CASES = 65_535
MAX_PROTOCOL_BYTES = 64 * 1024 * 1024
_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_CASE_ID_PATTERN = re.compile(r"^D24-[DH]\d{3}$")
_CASE_DOMAIN = b"blockvideo-case-v1\0"
_CATEGORY_DOMAIN = b"blockvideo-category-v1\0"
_REPARSE_POINT = 0x400

OpaqueToken = Annotated[str, Field(pattern=_SHA256_PATTERN)]
Sha256 = Annotated[str, Field(pattern=_SHA256_PATTERN)]
ProtocolCount = Annotated[int, Field(strict=True, ge=1, le=MAX_PROTOCOL_CASES)]
ResultCount = Annotated[int, Field(strict=True, ge=0, le=MAX_PROTOCOL_CASES)]


def _canonical_json_length(value: object) -> int:
    return len(
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    )


def maximum_protocol_serialized_bytes() -> int:
    """Return a proven canonical-byte upper bound for the maximum protocol topology."""
    token = "0" * 64
    scalar_shape = {
        "schema_version": 1,
        "candidate_id": "\U0010ffff" * 128,
        "modes": ["all_tools", "stateful"],
        "per_call_deadline_seconds": 180,
        "maximum_model_calls": 4,
        "isolation": "fresh_case_state_under_source_group",
        "corpus_sha256": token,
        "human_approval_sha256": token,
        "independent_approval_sha256": token,
        "freeze_sha256": token,
        "d36_trial_tool_sha256": token,
        "d37_evaluator_tool_sha256": token,
        "model_configuration_sha256": token,
        "stateful_index_sha256": token,
        "category_count": MAX_PROTOCOL_CASES,
        "category_tokens": [],
        "case_count": MAX_PROTOCOL_CASES,
        "case_tokens": [],
        "case_categories": [],
    }
    token_item_length = _canonical_json_length(token)
    binding_item_length = _canonical_json_length(
        {"case_token": token, "category_token": token}
    )
    topology_growth = (
        2 * (MAX_PROTOCOL_CASES * (token_item_length + 1) - 1)
        + MAX_PROTOCOL_CASES * (binding_item_length + 1)
        - 1
    )
    return _canonical_json_length(scalar_shape) + topology_growth + 1


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class CaseCategoryBinding(_StrictModel):
    case_token: OpaqueToken
    category_token: OpaqueToken


def _identifier_bytes(value: str, *, case_id: bool) -> bytes:
    if not value or "\0" in value or (case_id and _CASE_ID_PATTERN.fullmatch(value) is None):
        raise ValueError("invalid D24 case ID" if case_id else "invalid category ID")
    try:
        return value.encode("ascii")
    except UnicodeEncodeError:
        raise ValueError("token identifiers must be ASCII") from None


def _opaque_token(key: bytes | bytearray, domain: bytes, identifier: bytes) -> str:
    if len(key) != 32:
        raise ValueError("token key must contain exactly 32 raw bytes")
    return hmac.new(key, domain + identifier, hashlib.sha256).hexdigest()


def opaque_case_token(key: bytes, case_id: str) -> OpaqueToken:
    return _opaque_token(key, _CASE_DOMAIN, _identifier_bytes(case_id, case_id=True))


def opaque_category_token(key: bytes, category_id: str) -> OpaqueToken:
    return _opaque_token(key, _CATEGORY_DOMAIN, _identifier_bytes(category_id, case_id=False))


def _new_token_key_buffer() -> bytearray:
    return bytearray()


@contextmanager
def token_key(path: Path) -> Iterator[bytearray]:
    """Read one exact external key without following links and erase its buffer."""
    key = _new_token_key_buffer()
    descriptor: int | None = None
    try:
        before = path.lstat()
        if (
            stat.S_ISLNK(before.st_mode)
            or getattr(before, "st_file_attributes", 0) & _REPARSE_POINT
            or not stat.S_ISREG(before.st_mode)
        ):
            raise ValueError("token key must be a regular non-link file")
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        try:
            opened = os.fstat(descriptor)
            if (
                not stat.S_ISREG(opened.st_mode)
                or getattr(opened, "st_file_attributes", 0) & _REPARSE_POINT
                or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
            ):
                raise ValueError("token key identity changed while opening")
            key.extend(os.read(descriptor, 33))
            final = os.fstat(descriptor)
        finally:
            current = descriptor
            descriptor = None
            os.close(current)
        after = path.lstat()
        if (
            stat.S_ISLNK(after.st_mode)
            or getattr(after, "st_file_attributes", 0) & _REPARSE_POINT
            or not stat.S_ISREG(after.st_mode)
        ):
            raise ValueError("token key identity changed after reading")
        identities = (
            (before.st_dev, before.st_ino, before.st_size),
            (opened.st_dev, opened.st_ino, opened.st_size),
            (final.st_dev, final.st_ino, final.st_size),
            (after.st_dev, after.st_ino, after.st_size),
        )
        if len(set(identities)) != 1 or len(key) != 32:
            raise ValueError("token key must contain exactly 32 raw bytes")
        yield key
    finally:
        try:
            if descriptor is not None:
                os.close(descriptor)
        finally:
            descriptor = None
            for index in range(len(key)):
                key[index] = 0


def case_category_bindings(
    cases: Sequence[Case], key: bytes | bytearray
) -> tuple[CaseCategoryBinding, ...]:
    if not 1 <= len(cases) <= MAX_PROTOCOL_CASES:
        raise ValueError("protocol corpus size is outside the supported range")
    bindings = []
    for case in cases:
        if case.split != "held_out" or not case.tags:
            raise ValueError("protocol cases must be held-out D24 cases with tags")
        bindings.append(
            CaseCategoryBinding(
                case_token=opaque_case_token(key, case.case_id),
                category_token=opaque_category_token(key, case.tags[0]),
            )
        )
    bindings.sort(key=lambda item: (item.case_token, item.category_token))
    if len({item.case_token for item in bindings}) != len(bindings):
        raise ValueError("protocol case tokens must be unique")
    return tuple(bindings)


class EvaluationProtocol(_StrictModel):
    schema_version: Literal[1]
    candidate_id: Annotated[str, Field(min_length=1, max_length=128)]
    modes: tuple[Literal["all_tools"], Literal["stateful"]]
    per_call_deadline_seconds: Literal[180]
    maximum_model_calls: Literal[4]
    isolation: Literal["fresh_case_state_under_source_group"]
    corpus_sha256: Sha256
    human_approval_sha256: Sha256
    independent_approval_sha256: Sha256
    freeze_sha256: Sha256
    d36_trial_tool_sha256: Sha256
    d37_evaluator_tool_sha256: Sha256
    model_configuration_sha256: Sha256
    stateful_index_sha256: Sha256
    category_count: ProtocolCount
    category_tokens: Annotated[
        tuple[OpaqueToken, ...], Field(min_length=1, max_length=MAX_PROTOCOL_CASES)
    ]
    case_count: ProtocolCount
    case_tokens: Annotated[
        tuple[OpaqueToken, ...], Field(min_length=1, max_length=MAX_PROTOCOL_CASES)
    ]
    case_categories: Annotated[
        tuple[CaseCategoryBinding, ...], Field(min_length=1, max_length=MAX_PROTOCOL_CASES)
    ]

    @field_validator("schema_version", "per_call_deadline_seconds", "maximum_model_calls", mode="before")
    @classmethod
    def validate_literal_integers(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("protocol literal must be an integer")
        return value

    @model_validator(mode="after")
    def validate_topology(self) -> Self:
        if self.modes != ("all_tools", "stateful"):
            raise ValueError("evaluation modes must use the exact canonical order")
        if self.case_tokens != tuple(sorted(set(self.case_tokens))):
            raise ValueError("protocol case tokens must be unique and sorted")
        if self.category_tokens != tuple(sorted(set(self.category_tokens))):
            raise ValueError("protocol category tokens must be unique and sorted")
        if self.case_count != len(self.case_tokens):
            raise ValueError("protocol case count does not match its tokens")
        if self.category_count != len(self.category_tokens):
            raise ValueError("protocol category count does not match its tokens")
        if self.category_count > self.case_count:
            raise ValueError("protocol category count cannot exceed case count")
        binding_pairs = tuple(
            (binding.case_token, binding.category_token) for binding in self.case_categories
        )
        if binding_pairs != tuple(sorted(set(binding_pairs))):
            raise ValueError("case/category bindings must be unique and sorted")
        bound_cases = tuple(binding.case_token for binding in self.case_categories)
        if len(bound_cases) != len(set(bound_cases)) or tuple(sorted(bound_cases)) != self.case_tokens:
            raise ValueError("every protocol case must have exactly one category binding")
        if tuple(sorted({binding.category_token for binding in self.case_categories})) != (
            self.category_tokens
        ):
            raise ValueError("binding categories must equal the protocol category set")
        return self
