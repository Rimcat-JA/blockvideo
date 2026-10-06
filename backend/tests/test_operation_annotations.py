"""Japanese operation annotations: coverage, index binding and hybrid ranking."""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from app.operations.annotations import load_annotations, parse_annotations
from app.operations.catalog import CatalogError, load_catalog
from app.retrieval.builder import publish_index
from app.retrieval.contracts import EmbeddingProfile, OperationRef, SearchScope
from app.retrieval.ranking import lexical_similarity, rank_operations
from app.retrieval.reader import load_index
from app.retrieval.serialization import RetrievalError
from app.retrieval.sources import DEFAULT_CATALOG, DEFAULT_SCOPE, load_sources

CATALOG = load_catalog(DEFAULT_CATALOG)
OPERATIONS = {(item.operation_id, item.operation_version) for item in CATALOG.definitions}
PROFILE = EmbeddingProfile(model="synthetic", weights_sha256="a" * 64, dimensions=2,
                           document_prefix="document: ", query_prefix="query: ")


def test_every_operation_has_japanese_utterances_and_notes() -> None:
    annotations = load_annotations()
    annotations.require_known_operations(OPERATIONS)
    for operation_id, version in OPERATIONS:
        item = annotations.for_operation(operation_id, version)
        assert len(item.utterances) >= 2, operation_id
        assert item.distinctions, operation_id


def test_annotations_for_unknown_operations_are_rejected() -> None:
    raw = b'{"format_version": 1, "operations": {"project.missing.op": {"utterances": ["x"]}}}'
    with pytest.raises(CatalogError, match="unknown operations"):
        parse_annotations(raw).require_known_operations(OPERATIONS)
    with pytest.raises(CatalogError, match="empty annotation"):
        parse_annotations(b'{"format_version": 1, "operations": {"project.status.get": {"utterances": [" "]}}}')


def test_annotations_become_index_documents_bound_by_hash(tmp_path: Path) -> None:
    sources = load_sources()
    kinds = {document.kind for document in sources.documents}
    assert "annotation" in kinds and sources.annotations_sha256 is not None
    manifest = publish_index(tmp_path, sources, PROFILE, tuple((1.0, 0.0) for _ in sources.documents))
    assert manifest.annotations_sha256 == sources.annotations_sha256
    assert manifest.extraction_version == "public-metadata-annotations-v2"
    load_index(tmp_path, sources, PROFILE)

    # A changed annotation file makes the published index stale.
    catalog_dir = tmp_path / "catalog"
    catalog_dir.mkdir()
    for name in ("definitions.json", "search_scope.json", "operation_annotations.json"):
        shutil.copy(DEFAULT_CATALOG.with_name(name), catalog_dir / name)
    annotations = catalog_dir / "operation_annotations.json"
    annotations.write_text(annotations.read_text(encoding="utf-8").replace("動画の状態を確認して", "状態を見せて"),
                           encoding="utf-8")
    changed = load_sources(catalog_dir / "definitions.json", catalog_dir / "search_scope.json")
    assert changed.annotations_sha256 != sources.annotations_sha256
    with pytest.raises(RetrievalError):
        load_index(tmp_path, changed, PROFILE)


def test_catalog_without_annotations_keeps_the_annotation_free_format(tmp_path: Path) -> None:
    shutil.copy(DEFAULT_CATALOG, tmp_path / "definitions.json")
    shutil.copy(DEFAULT_SCOPE, tmp_path / "search_scope.json")
    sources = load_sources(tmp_path / "definitions.json", tmp_path / "search_scope.json")
    assert sources.annotations_sha256 is None
    assert "annotation" not in {document.kind for document in sources.documents}


def test_lexical_similarity_is_bounded_and_prefers_shared_words() -> None:
    assert lexical_similarity("動画を作り直して", "動画を作り直して") == 1.0
    assert lexical_similarity("", "動画") == 0.0
    assert lexical_similarity("字幕を大きくして", "字幕を少し大きくして") > lexical_similarity("字幕を大きくして", "生成を止めて")


def test_hybrid_ranking_finds_the_annotated_operation_when_vectors_cannot(tmp_path: Path) -> None:
    sources = load_sources()
    # Identical vectors: cosine cannot separate operations, so only the lexical match ranks.
    publish_index(tmp_path, sources, PROFILE, tuple((1.0, 0.0) for _ in sources.documents))
    index = load_index(tmp_path, sources, PROFILE)
    scope = SearchScope(app_id="blockvideo",
        capabilities=tuple(sorted({c for d in sources.documents for c in d.required_capabilities})),
        operations=tuple(OperationRef(operation_id=o, operation_version=v) for o, v in sorted(OPERATIONS)))
    expectations = {
        "ジョブ3をキャンセルして": "project.generation.cancel",
        "以前の設定に戻して": "project.settings.restore",
        "動画を書き出して": "project.generation.start",
        "進み具合を教えて": "project.status.get",
    }
    for text, operation_id in expectations.items():
        ranking = rank_operations(index, (1.0, 0.0), scope, sources, query_text=text)
        assert ranking[0].operation_id == operation_id, (text, ranking[:3])
        assert -1 <= ranking[0].score <= 1
    plain = rank_operations(index, (1.0, 0.0), scope, sources)
    assert {item.score for item in plain} == {1.0}


def test_verified_index_is_reused_until_the_bundle_file_changes(tmp_path: Path) -> None:
    sources = load_sources()
    manifest = publish_index(tmp_path, sources, PROFILE, tuple((1.0, 0.0) for _ in sources.documents))
    first = load_index(tmp_path, sources, PROFILE)
    assert load_index(tmp_path, sources, PROFILE) is first
    bundle = tmp_path / f"bundle-{manifest.bundle_sha256}.json"
    bundle.write_bytes(bundle.read_bytes().replace(b"1.0", b"0.5", 1))
    with pytest.raises(RetrievalError):
        load_index(tmp_path, sources, PROFILE)
