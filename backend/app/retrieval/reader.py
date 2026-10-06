"""Read-only verified index snapshot; no ranking, readiness or execution."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from app.operations.limits import MAX_INDEX_BUNDLE_BYTES
from app.retrieval.contracts import EmbeddingProfile, IndexBundle, IndexDocument, IndexManifest, SearchScope
from app.retrieval.serialization import RetrievalError, canonical, decode, digest, read_bytes
from app.retrieval.sources import IndexSources
from app.retrieval.vectors import validate_vectors


def check_sources(manifest: IndexManifest, sources: IndexSources) -> None:
    if (manifest.catalog_sha256 != sources.catalog_sha256
            or manifest.scope_sha256 != sources.scope_sha256
            or manifest.catalog_semantic_sha256 != sources.catalog_semantic_sha256
            or manifest.annotations_sha256 != sources.annotations_sha256
            or manifest.app_id != sources.app_id
            or manifest.operation_count != sources.operation_count):
        raise RetrievalError("stale_index")


@dataclass(frozen=True)
class VerifiedIndex:
    manifest: IndexManifest
    bundle: IndexBundle

    def eligible_documents(self, scope: SearchScope, current_sources: IndexSources) -> tuple[IndexDocument, ...]:
        """Recheck current source fingerprints before every later candidate lookup."""
        check_sources(self.manifest, current_sources)
        scope = SearchScope.model_validate(scope.model_dump())
        if scope.app_id != self.manifest.app_id:
            return ()
        requested = {ref.key for ref in scope.operations}
        known = {doc.key for doc in self.bundle.documents}
        if len(requested) != len(scope.operations) or requested - known:
            raise RetrievalError("unknown_or_duplicate_scope_operation")
        capabilities = set(scope.capabilities)
        return tuple(doc for doc in self.bundle.documents
                     if doc.key in requested and set(doc.required_capabilities) <= capabilities)


# Verified indexes by (directory, manifest bytes, bundle file identity, sources, profile).
# A large bundle is hashed and validated once; any change to the manifest, the bundle
# file (size, mtime, inode), the catalog sources or the profile verifies it again.
_VERIFIED: dict[tuple[object, ...], VerifiedIndex] = {}
_VERIFIED_LIMIT = 4


def load_index(directory: Path, sources: IndexSources, profile: EmbeddingProfile) -> VerifiedIndex:
    """A manifest is read once; its digest names one immutable bundle."""
    try:
        manifest_bytes = read_bytes(directory / "manifest.json", 64_000)
        raw = decode(manifest_bytes)
        if not isinstance(raw, dict) or type(raw.get("format_version")) is not int:
            raise RetrievalError("invalid_manifest")
        manifest = IndexManifest.model_validate(raw)
        check_sources(manifest, sources)
        if manifest.profile != profile:
            raise RetrievalError("embedding_profile_mismatch")
        bundle_path = directory / f"bundle-{manifest.bundle_sha256}.json"
        try:
            stat = bundle_path.stat()
        except OSError:
            raise RetrievalError("file_unavailable") from None
        key = (str(directory.resolve()), digest(manifest_bytes), stat.st_size, stat.st_mtime_ns, stat.st_ino,
               sources.catalog_sha256, sources.scope_sha256, sources.catalog_semantic_sha256,
               sources.annotations_sha256, canonical(profile.model_dump(mode="json")))
        cached = _VERIFIED.get(key)
        if cached is not None:
            return cached
        payload = read_bytes(bundle_path, MAX_INDEX_BUNDLE_BYTES)
        if digest(payload) != manifest.bundle_sha256:
            raise RetrievalError("bundle_hash_mismatch")
        bundle = IndexBundle.model_validate(decode(payload))
        documents_json = [d.model_dump(mode="json") for d in bundle.documents]
        if (len(bundle.documents) != manifest.document_count
                or digest(canonical(documents_json)) != manifest.documents_sha256
                or digest(canonical(bundle.vectors)) != manifest.vectors_sha256):
            raise RetrievalError("content_hash_mismatch")
        # Re-derive public text and exact versions. A self-consistent forged
        # manifest with a wrong ID/version/schema/scope must still fail.
        if bundle.documents != sources.documents:
            raise RetrievalError("documents_source_mismatch")
        validate_vectors(bundle.vectors, len(bundle.documents), profile.dimensions)
        verified = VerifiedIndex(manifest, bundle)
        if len(_VERIFIED) >= _VERIFIED_LIMIT:
            _VERIFIED.pop(next(iter(_VERIFIED)))
        _VERIFIED[key] = verified
        return verified
    except (ValidationError, TypeError, UnicodeError, RecursionError, OverflowError):
        raise RetrievalError("invalid_index") from None
