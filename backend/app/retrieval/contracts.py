"""Immutable index metadata and state-independent scope contracts."""
from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.operations.limits import MAX_CATALOG_OPERATIONS, MAX_INDEX_DOCUMENTS

Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Identifier = Annotated[str, Field(pattern=r"^[a-z][a-z0-9.-]{0,127}$")]
Version = Annotated[int, Field(ge=1, strict=True)]
Scalar = Annotated[float, Field(strict=True, allow_inf_nan=False)]


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class OperationRef(FrozenModel):
    operation_id: Identifier
    operation_version: Version

    @property
    def key(self) -> tuple[str, int]:
        return self.operation_id, self.operation_version


class ScopeBinding(OperationRef):
    required_capabilities: tuple[Identifier, ...] = Field(min_length=1)


class CatalogScope(FrozenModel):
    format_version: Literal[1]
    app_id: Identifier
    bindings: tuple[ScopeBinding, ...] = Field(min_length=1, max_length=MAX_CATALOG_OPERATIONS)


class SearchScope(FrozenModel):
    """Host-provided installed capabilities and exact versions, never live state."""

    app_id: Identifier
    capabilities: tuple[Identifier, ...]
    operations: tuple[OperationRef, ...]


class EmbeddingProfile(FrozenModel):
    model: str = Field(min_length=1, max_length=200)
    weights_sha256: Digest
    dimensions: int = Field(ge=1, le=4096, strict=True)
    document_prefix: str = Field(max_length=80)
    query_prefix: str = Field(max_length=80)
    normalization: Literal["l2-full-v1"] = "l2-full-v1"
    transport: Literal["local-openai-embeddings-v1", "local-onnx-e5-v1"] = "local-openai-embeddings-v1"
    tokenizer_sha256: Digest | None = None
    source_revision: str | None = Field(default=None, max_length=64)


class IndexDocument(OperationRef):
    document_id: str = Field(min_length=1, max_length=200)
    app_id: Identifier
    required_capabilities: tuple[Identifier, ...]
    kind: Literal["description", "example", "input", "annotation"]
    ordinal: int = Field(ge=0, strict=True)
    text: str = Field(min_length=1, max_length=1200)
    definition_sha256: Digest


class IndexBundle(FrozenModel):
    documents: tuple[IndexDocument, ...] = Field(min_length=1, max_length=MAX_INDEX_DOCUMENTS)
    vectors: tuple[tuple[Scalar, ...], ...] = Field(min_length=1, max_length=MAX_INDEX_DOCUMENTS)


class IndexManifest(FrozenModel):
    format_version: Literal[1] = 1
    builder_version: Literal["operation-index-v1"] = "operation-index-v1"
    extraction_version: Literal["public-metadata-chunks-v1", "public-metadata-annotations-v2"] = "public-metadata-chunks-v1"
    created_at: str = Field(min_length=1, max_length=80)
    app_id: Identifier
    catalog_sha256: Digest
    scope_sha256: Digest
    catalog_semantic_sha256: Digest
    # Bound when the catalog carries operation annotations (index documents of kind "annotation").
    annotations_sha256: Digest | None = None
    profile: EmbeddingProfile
    operation_count: int = Field(ge=1, le=MAX_CATALOG_OPERATIONS, strict=True)
    document_count: int = Field(ge=1, le=MAX_INDEX_DOCUMENTS, strict=True)
    documents_sha256: Digest
    vectors_sha256: Digest
    bundle_sha256: Digest
    # No file name is accepted from an untrusted manifest. Derive it from digest.
