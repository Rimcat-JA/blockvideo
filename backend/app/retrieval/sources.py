"""Derive bounded public documents from canonical definitions and host bindings."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from app.operations.annotations import OperationAnnotations, parse_annotations
from app.operations.catalog import CatalogError, OperationCatalog, load_catalog
from app.operations.limits import MAX_INDEX_DOCUMENTS
from app.retrieval.contracts import CatalogScope, IndexDocument
from app.retrieval.serialization import RetrievalError, canonical, decode, digest, read_bytes

DEFAULT_CATALOG = Path(__file__).parents[1] / "operations/definitions.json"
DEFAULT_SCOPE = Path(__file__).parents[1] / "operations/search_scope.json"


@dataclass(frozen=True)
class IndexSources:
    catalog_sha256: str
    scope_sha256: str
    catalog_semantic_sha256: str
    app_id: str
    operation_count: int
    documents: tuple[IndexDocument, ...]
    annotations_sha256: str | None = None


def _input_lines(schema: dict[str, Any], path: str = "arguments") -> list[str]:
    """All constraints survive extraction; only nested branches become own lines."""
    own = {k: v for k, v in schema.items() if k not in {"properties", "items"}}
    lines = [f"{path}: {canonical(own).decode('utf-8')}"]
    for name, child in sorted(schema.get("properties", {}).items()):
        lines.extend(_input_lines(child, f"{path}.{name}"))
    if "items" in schema:
        lines.extend(_input_lines(schema["items"], f"{path}[]"))
    return lines


def _chunks(text: str) -> list[str]:
    """Bound UTF-8 bytes as well as characters; never split a Unicode character."""
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for char in text:
        width = len(char.encode("utf-8"))
        if size + width > 1200:
            chunks.append("".join(current))
            current, size = [], 0
        current.append(char)
        size += width
    if current:
        chunks.append("".join(current))
    return chunks


def extract_documents(catalog: OperationCatalog, scope: CatalogScope,
                      annotations: OperationAnnotations | None = None) -> tuple[IndexDocument, ...]:
    """No callable/policy names, project state, evaluation cases or model examples."""
    definitions = sorted(catalog.definitions, key=lambda d: (d.operation_id, d.operation_version))
    keys = {(d.operation_id, d.operation_version) for d in definitions}
    bindings = {b.key: b for b in scope.bindings}
    if len(bindings) != len(scope.bindings) or set(bindings) != keys:
        raise RetrievalError("scope_catalog_mismatch")
    documents: list[IndexDocument] = []
    for definition in definitions:
        binding = bindings[(definition.operation_id, definition.operation_version)]
        if len(set(binding.required_capabilities)) != len(binding.required_capabilities):
            raise RetrievalError("duplicate_capability")
        definition_hash = digest(canonical(definition.model_dump(mode="json")))
        sections = [("description", [definition.description]), ("example", definition.examples),
                    ("input", _input_lines(definition.input_schema))]
        if annotations is not None:
            # Japanese utterances, synonyms and scenarios: each its own document, so the
            # per-operation maximum score matches whichever phrasing is closest.
            sections.append(("annotation", list(annotations.for_operation(
                definition.operation_id, definition.operation_version).searchable())))
        for kind, texts in sections:
            ordinal = 0
            for text in texts:
                for chunk in _chunks(text):
                    documents.append(IndexDocument(
                        operation_id=definition.operation_id, operation_version=definition.operation_version,
                        document_id=f"{definition.operation_id}@{definition.operation_version}:{kind}:{ordinal}",
                        app_id=scope.app_id, required_capabilities=tuple(sorted(binding.required_capabilities)),
                        kind=kind, ordinal=ordinal, text=chunk,
                        definition_sha256=definition_hash,
                    ))
                    ordinal += 1
    if not documents or len(documents) > MAX_INDEX_DOCUMENTS:
        raise RetrievalError("document_limit")
    return tuple(documents)


def load_sources(catalog_path: Path = DEFAULT_CATALOG, scope_path: Path = DEFAULT_SCOPE) -> IndexSources:
    """Read a coherent bounded snapshot; full source bytes invalidate old indexes.

    Annotations are read from ``operation_annotations.json`` beside the catalog when
    that file exists; a catalog without one keeps the annotation-free index format.
    """
    catalog_bytes = read_bytes(catalog_path, 2_000_000)
    scope_bytes = read_bytes(scope_path, 200_000)
    annotations_path = catalog_path.with_name("operation_annotations.json")
    annotations_bytes = read_bytes(annotations_path, 1_000_000) if annotations_path.is_file() else None
    try:
        decode(catalog_bytes)  # Reject duplicate keys before the legacy loader.
        catalog = load_catalog(catalog_path)
        scope_raw = decode(scope_bytes)
        if not isinstance(scope_raw, dict) or type(scope_raw.get("format_version")) is not int:
            raise RetrievalError("invalid_source")
        scope = CatalogScope.model_validate(scope_raw)
        annotations = parse_annotations(annotations_bytes) if annotations_bytes is not None else None
        if annotations is not None:
            annotations.require_known_operations(
                {(d.operation_id, d.operation_version) for d in catalog.definitions})
    except (CatalogError, ValidationError, TypeError):
        raise RetrievalError("invalid_source") from None
    if (read_bytes(catalog_path, 2_000_000) != catalog_bytes or read_bytes(scope_path, 200_000) != scope_bytes
            or (annotations_bytes is not None and read_bytes(annotations_path, 1_000_000) != annotations_bytes)):
        raise RetrievalError("source_changed")
    docs = extract_documents(catalog, scope, annotations)
    ordered = sorted(catalog.definitions, key=lambda d: (d.operation_id, d.operation_version))
    return IndexSources(digest(catalog_bytes), digest(scope_bytes),
                        digest(canonical([d.model_dump(mode="json") for d in ordered])),
                        scope.app_id, len(ordered), docs,
                        digest(annotations_bytes) if annotations_bytes is not None else None)
