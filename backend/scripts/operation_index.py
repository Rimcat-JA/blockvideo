"""Build/check an operation index from canonical sources; never execute operations."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path

from pydantic import ValidationError

from app.retrieval.builder import publish_index
from app.retrieval.contracts import EmbeddingProfile, OperationRef, SearchScope
from app.retrieval.embeddings import LocalEmbeddingAdapter
from app.retrieval.onnx_embeddings import OnnxEmbeddingAdapter
from app.retrieval.reader import load_index
from app.retrieval.serialization import RetrievalError, decode, read_bytes
from app.retrieval.sources import DEFAULT_CATALOG, DEFAULT_SCOPE, load_sources

DEFAULT_PROFILE = Path(__file__).resolve().parents[1] / "app/retrieval/nomic-profile.json"


def weights_digest(path: Path) -> str:
    try:
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()
    except OSError:
        raise RetrievalError("weights_unavailable") from None


async def build(args: argparse.Namespace, profile: EmbeddingProfile) -> dict[str, object]:
    sources = load_sources(args.catalog, args.scope)
    if weights_digest(args.weights) != profile.weights_sha256:
        raise RetrievalError("weights_mismatch")
    encoder = (OnnxEmbeddingAdapter(profile, args.weights.parent)
               if profile.transport == "local-onnx-e5-v1" else LocalEmbeddingAdapter(profile, args.base_url))
    async with encoder as adapter:
        vectors = await adapter.embed_documents(tuple(d.text for d in sources.documents))
        calls = adapter.calls
    # No incomplete or mixed-source build can replace the published manifest.
    if load_sources(args.catalog, args.scope) != sources:
        raise RetrievalError("source_changed")
    if weights_digest(args.weights) != profile.weights_sha256:
        raise RetrievalError("weights_changed")
    manifest = publish_index(args.index, sources, profile, vectors)
    load_index(args.index, sources, profile)
    return {"status": "built", "operation_count": manifest.operation_count,
            "document_count": manifest.document_count, "dimensions": profile.dimensions,
            "embedding_calls": calls, "bundle_sha256": manifest.bundle_sha256}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("command", choices=("build", "validate", "select"))
    result.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    result.add_argument("--scope", type=Path, default=DEFAULT_SCOPE)
    result.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    result.add_argument("--index", type=Path, required=True)
    result.add_argument("--weights", type=Path, help="Build only: local file matching the profile SHA-256")
    result.add_argument("--base-url", default="http://127.0.0.1:1234/v1")
    result.add_argument("--app", help="Select only: exact host application ID")
    result.add_argument("--capability", action="append", default=[])
    result.add_argument("--operation", action="append", default=[], help="Select only: exact ID@version")
    return result


def main() -> int:
    arg_parser = parser()
    args = arg_parser.parse_args()
    if args.command == "build" and args.weights is None:
        arg_parser.error("build requires --weights")
    if args.command == "select" and args.app is None:
        arg_parser.error("select requires --app")
    try:
        profile = EmbeddingProfile.model_validate(decode(read_bytes(args.profile, 16_000)))
        if args.command == "build":
            output = asyncio.run(build(args, profile))
        else:
            sources = load_sources(args.catalog, args.scope)
            index = load_index(args.index, sources, profile)
            output = {"status": "valid", "operation_count": index.manifest.operation_count,
                      "document_count": index.manifest.document_count,
                      "bundle_sha256": index.manifest.bundle_sha256, "embedding_calls": 0}
            if args.command == "select":
                refs = []
                for raw in args.operation:
                    operation_id, version = raw.rsplit("@", 1)
                    refs.append(OperationRef(operation_id=operation_id, operation_version=int(version)))
                selected = index.eligible_documents(SearchScope(
                    app_id=args.app, capabilities=tuple(args.capability), operations=tuple(refs)), sources)
                output = {"status": "scoped", "document_count": len(selected), "embedding_calls": 0,
                          "operations": [{"operation_id": key[0], "operation_version": key[1]}
                                         for key in sorted({d.key for d in selected})]}
        print(json.dumps(output, ensure_ascii=False))
        return 0
    except RetrievalError as exc:
        print(json.dumps({"status": "error", "reason_code": exc.code}))
    except (ValidationError, ValueError, TypeError, OSError):
        print(json.dumps({"status": "error", "reason_code": "invalid_configuration"}))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
