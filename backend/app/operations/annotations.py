"""Japanese annotations that make catalog operations findable and distinguishable.

Annotations are retrieval and selection aids only: they never add arguments,
capabilities or execution power. Utterances, synonyms and scenarios are embedded
as index documents; distinctions (what an operation is NOT for) are shown to the
model beside the candidate, because negative sentences match badly in embeddings.
They are written from the specification and development material, never from
held-out evaluation cases.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.operations.schema_validation import CatalogError

DEFAULT_ANNOTATIONS = Path(__file__).with_name("operation_annotations.json")


class OperationAnnotation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    utterances: tuple[str, ...] = Field(default=(), max_length=64)
    synonyms: tuple[str, ...] = Field(default=(), max_length=64)
    scenarios: tuple[str, ...] = Field(default=(), max_length=16)
    distinctions: tuple[str, ...] = Field(default=(), max_length=16)

    # Distinctions document how an operation differs from look-alikes for catalog
    # authors and reviewers; they are not indexed (they name the other operations).
    def searchable(self) -> tuple[str, ...]:
        return (*self.utterances, *self.synonyms, *self.scenarios)


class OperationAnnotations(BaseModel):
    """Keys are an operation ID (all versions) or ``<operation_id>@<version>``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    format_version: int
    operations: dict[str, OperationAnnotation]

    def for_operation(self, operation_id: str, operation_version: int) -> OperationAnnotation:
        shared = self.operations.get(operation_id)
        versioned = self.operations.get(f"{operation_id}@{operation_version}")
        parts = [item for item in (shared, versioned) if item is not None]
        if not parts:
            return OperationAnnotation()
        return OperationAnnotation(**{
            field: tuple(text for item in parts for text in getattr(item, field))
            for field in OperationAnnotation.model_fields
        })

    def require_known_operations(self, operations: set[tuple[str, int]]) -> None:
        """An annotation for an operation the catalog does not have is a typo."""
        ids = {operation_id for operation_id, _ in operations}
        unknown = sorted(key for key in self.operations
                         if key.split("@", 1)[0] not in ids
                         or ("@" in key and (key.split("@", 1)[0], _version(key)) not in operations))
        if unknown:
            raise CatalogError(f"annotations for unknown operations: {unknown}")


def _version(key: str) -> int:
    try:
        return int(key.split("@", 1)[1])
    except ValueError:
        return -1


def parse_annotations(raw: bytes) -> OperationAnnotations:
    try:
        annotations = OperationAnnotations.model_validate(json.loads(raw.decode("utf-8")))
    except (UnicodeDecodeError, json.JSONDecodeError, ValidationError) as exc:
        raise CatalogError(f"invalid operation annotations: {exc}") from exc
    if annotations.format_version != 1:
        raise CatalogError("unsupported annotations format_version")
    for key, item in annotations.operations.items():
        if any(not text.strip() for field in OperationAnnotation.model_fields for text in getattr(item, field)):
            raise CatalogError(f"empty annotation text for {key}")
    return annotations


@lru_cache(maxsize=4)
def load_annotations(path: Path = DEFAULT_ANNOTATIONS) -> OperationAnnotations:
    try:
        return parse_annotations(path.read_bytes())
    except OSError as exc:
        raise CatalogError(f"operation annotations unavailable: {exc}") from exc
