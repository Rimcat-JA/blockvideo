"""D26 source/version integrity, atomic publication and scope isolation."""
from __future__ import annotations

import argparse
import ast
import json
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.operations.contracts import OperationRequest
from app.retrieval import builder
from app.retrieval.contracts import EmbeddingProfile, OperationRef, SearchScope
from app.retrieval.reader import load_index
from app.retrieval.serialization import RetrievalError, canonical, digest
from app.retrieval.sources import DEFAULT_CATALOG, DEFAULT_SCOPE, load_sources
from scripts import operation_index


@pytest.fixture
def sources_files(tmp_path: Path) -> tuple[Path, Path]:
    catalog, scope = tmp_path / "definitions.json", tmp_path / "scope.json"
    catalog.write_bytes(DEFAULT_CATALOG.read_bytes())
    scope.write_bytes(DEFAULT_SCOPE.read_bytes())
    return catalog, scope


@pytest.fixture
def profile() -> EmbeddingProfile:
    return EmbeddingProfile(model="synthetic", weights_sha256="a" * 64, dimensions=2,
                            document_prefix="document: ", query_prefix="query: ")


def write_json(path: Path, value: object) -> None:
    path.write_bytes(canonical(value))


def alter_catalog(paths: tuple[Path, Path], change: str) -> None:
    raw = json.loads(paths[0].read_bytes())
    adjust = next(item for item in raw["operations"] if item["operation_id"] == "project.subtitle-font-size.adjust")
    if change == "description":
        adjust["description"] += " Updated canonical explanation."
    elif change == "example":
        adjust["examples"].append("字幕サイズを明示量で変える")
    elif change == "schema":
        adjust["input_schema"]["properties"]["delta"]["minimum"] = -100
    elif change == "version":
        adjust["operation_version"] = 3
        scope = json.loads(paths[1].read_bytes())
        for binding in scope["bindings"]:
            if binding["operation_id"] == adjust["operation_id"]:
                binding["operation_version"] = 3
        write_json(paths[1], scope)
    write_json(paths[0], raw)


def build_index(tmp_path: Path, paths: tuple[Path, Path], profile: EmbeddingProfile):
    sources = load_sources(*paths)
    root = tmp_path / "index"
    builder.publish_index(root, sources, profile, tuple((1.0, 0.0) for _ in sources.documents))
    return root, sources


def full_scope(sources) -> SearchScope:
    return SearchScope(app_id="blockvideo",
        capabilities=tuple(sorted({c for d in sources.documents for c in d.required_capabilities})),
        operations=tuple(OperationRef(operation_id=k[0], operation_version=k[1])
                         for k in sorted({d.key for d in sources.documents})))


def test_documents_reproducible_complete_and_public(sources_files) -> None:
    sources = load_sources(*sources_files)
    assert sources == load_sources(*sources_files)
    assert sources.operation_count == 10
    assert len({d.document_id for d in sources.documents}) == len(sources.documents)
    assert {d.kind for d in sources.documents} == {"description", "example", "input"}
    raw = canonical([d.model_dump(mode="json") for d in sources.documents]).decode()
    assert "voicevox_speed_scale" in raw and "pronunciation_overrides" in raw
    assert "handler_key" not in raw and "precondition_key" not in raw
    assert "project_editable" not in raw


def test_unicode_long_description_is_losslessly_chunked(sources_files) -> None:
    raw = json.loads(sources_files[0].read_bytes())
    text = "日本語😀" * 2000
    raw["operations"][0]["description"] = text
    key = raw["operations"][0]["operation_id"]
    write_json(sources_files[0], raw)
    docs = [d for d in load_sources(*sources_files).documents if d.operation_id == key and d.kind == "description"]
    assert "".join(d.text for d in docs) == text
    assert all(len(d.text.encode()) <= 1200 for d in docs)


@pytest.mark.parametrize("change", ["description", "example", "schema", "version"])
def test_old_index_and_cached_reader_reject_changed_source(tmp_path, sources_files, profile, change) -> None:
    root, before = build_index(tmp_path, sources_files, profile)
    cached = load_index(root, before, profile)
    alter_catalog(sources_files, change)
    current = load_sources(*sources_files)
    with pytest.raises(RetrievalError, match="stale_index"):
        load_index(root, current, profile)
    with pytest.raises(RetrievalError, match="stale_index"):
        cached.eligible_documents(full_scope(current), current)
    manifest = builder.publish_index(root, current, profile, tuple((0.0, 1.0) for _ in current.documents))
    rebuilt = load_index(root, current, profile)
    assert rebuilt.bundle.documents == current.documents
    assert manifest.catalog_sha256 != cached.manifest.catalog_sha256
    assert len(list(root.glob("bundle-*.json"))) == 2  # old complete bundle retained


@pytest.mark.parametrize("change", ["missing", "unknown", "duplicate", "capability_duplicate"])
def test_scope_requires_complete_exact_source_bindings(sources_files, change) -> None:
    scope = json.loads(sources_files[1].read_bytes())
    if change == "missing":
        scope["bindings"].pop()
    elif change == "unknown":
        scope["bindings"][0]["operation_id"] = "project.unknown"
    elif change == "duplicate":
        scope["bindings"].append(scope["bindings"][0])
    else:
        scope["bindings"][0]["required_capabilities"] *= 2
    write_json(sources_files[1], scope)
    with pytest.raises(RetrievalError):
        load_sources(*sources_files)


def test_scope_only_filters_installed_capabilities_and_exact_versions(tmp_path, sources_files, profile) -> None:
    root, sources = build_index(tmp_path, sources_files, profile)
    index = load_index(root, sources, profile)
    scope = full_scope(sources)
    assert index.eligible_documents(scope, sources) == sources.documents
    assert index.eligible_documents(scope.model_copy(update={"app_id": "another-app"}), sources) == ()
    assert index.eligible_documents(scope.model_copy(update={"capabilities": ()}), sources) == ()
    v1 = OperationRef(operation_id="project.settings.update", operation_version=1)
    v2 = OperationRef(operation_id="project.settings.update", operation_version=2)
    selected = index.eligible_documents(SearchScope(app_id="blockvideo", capabilities=("settings.write",),
                                                   operations=(v1, v2)), sources)
    assert {d.key for d in selected} == {v1.key}
    selected = index.eligible_documents(SearchScope(app_id="blockvideo",
        capabilities=("settings.write", "settings.relative-subtitle"), operations=(v2,)), sources)
    assert {d.key for d in selected} == {v2.key}
    with pytest.raises(ValidationError):
        SearchScope.model_validate({**scope.model_dump(), "job_running": True})
    # Live job state has no input path. A supported edit remains eligible even
    # when the separate execution layer would report project_editable blocked.
    assert any(d.operation_id == "project.subtitle-font-size.set" for d in index.eligible_documents(scope, sources))
    with pytest.raises(ValidationError):
        OperationRequest.model_validate(selected[0].model_dump())


@pytest.mark.parametrize("refs", [
    (OperationRef(operation_id="project.unknown", operation_version=1),),
    (OperationRef(operation_id="project.status.get", operation_version=99),),
    (OperationRef(operation_id="project.status.get", operation_version=1),) * 2,
])
def test_unknown_or_duplicate_requested_version_cannot_select_other_version(tmp_path, sources_files, profile, refs) -> None:
    root, sources = build_index(tmp_path, sources_files, profile)
    with pytest.raises(RetrievalError, match="unknown_or_duplicate_scope_operation"):
        load_index(root, sources, profile).eligible_documents(full_scope(sources).model_copy(update={"operations": refs}), sources)


@pytest.mark.parametrize("change", [
    {"model": "another-model"}, {"weights_sha256": "b" * 64}, {"dimensions": 3},
    {"document_prefix": "different: "}, {"query_prefix": "different: "},
])
def test_embedding_profile_is_part_of_identity(tmp_path, sources_files, profile, change) -> None:
    root, sources = build_index(tmp_path, sources_files, profile)
    with pytest.raises(RetrievalError, match="embedding_profile_mismatch"):
        load_index(root, sources, profile.model_copy(update=change))


@pytest.mark.parametrize("change", ["id", "version", "text", "app", "capability", "missing", "duplicate", "vector"])
def test_self_consistent_tampered_bundle_still_checked_against_source(tmp_path, sources_files, profile, change) -> None:
    root, sources = build_index(tmp_path, sources_files, profile)
    manifest = json.loads((root / "manifest.json").read_bytes())
    bundle = json.loads((root / f'bundle-{manifest["bundle_sha256"]}.json').read_bytes())
    doc = bundle["documents"][0]
    if change == "id":
        doc["operation_id"] = "project.unknown"
    elif change == "version":
        doc["operation_version"] = 99
    elif change == "text":
        doc["text"] = "forged new instructions"
    elif change == "app":
        doc["app_id"] = "another-app"
    elif change == "capability":
        doc["required_capabilities"] = []
    elif change == "missing":
        bundle["documents"].pop()
        bundle["vectors"].pop()
    elif change == "duplicate":
        bundle["documents"][1] = doc
    elif change == "vector":
        bundle["vectors"][0] = [0.0, 0.0]
    payload = canonical(bundle)
    manifest.update(bundle_sha256=digest(payload), document_count=len(bundle["documents"]),
        documents_sha256=digest(canonical(bundle["documents"])), vectors_sha256=digest(canonical(bundle["vectors"])))
    (root / f'bundle-{manifest["bundle_sha256"]}.json').write_bytes(payload)
    write_json(root / "manifest.json", manifest)
    with pytest.raises(RetrievalError):
        load_index(root, sources, profile)


@pytest.mark.parametrize("change", ["partial_manifest", "missing_bundle", "corrupt_bundle", "old_format", "filename"])
def test_corrupt_or_partial_index_fails_closed(tmp_path, sources_files, profile, change) -> None:
    root, sources = build_index(tmp_path, sources_files, profile)
    manifest = json.loads((root / "manifest.json").read_bytes())
    bundle_path = root / f'bundle-{manifest["bundle_sha256"]}.json'
    if change == "partial_manifest":
        (root / "manifest.json").write_text('{"format_version":', encoding="utf-8")
    elif change == "missing_bundle":
        bundle_path.unlink()
    elif change == "corrupt_bundle":
        bundle_path.write_bytes(b"{}")
    else:
        manifest.update({"format_version": 99} if change == "old_format" else {"bundle_file": "../secret.json"})
        write_json(root / "manifest.json", manifest)
    with pytest.raises(RetrievalError):
        load_index(root, sources, profile)


def test_failed_publication_preserves_prior_manifest(tmp_path, sources_files, profile, monkeypatch) -> None:
    root, sources = build_index(tmp_path, sources_files, profile)
    before = (root / "manifest.json").read_bytes()
    original = builder._atomic_write

    def fail_manifest(path: Path, data: bytes) -> None:
        if path.name == "manifest.json":
            raise OSError("synthetic failure")
        original(path, data)

    monkeypatch.setattr(builder, "_atomic_write", fail_manifest)
    with pytest.raises(RetrievalError, match="index_write_failed"):
        builder.publish_index(root, sources, profile, tuple((0.0, 1.0) for _ in sources.documents))
    assert (root / "manifest.json").read_bytes() == before
    assert load_index(root, sources, profile).bundle.vectors[0] == (1.0, 0.0)


async def test_source_change_during_embedding_does_not_publish(tmp_path, sources_files, profile, monkeypatch) -> None:
    root, _ = build_index(tmp_path, sources_files, profile)
    before = (root / "manifest.json").read_bytes()

    class Adapter:
        calls = 1

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            pass

        async def embed_documents(self, texts):
            alter_catalog(sources_files, "description")
            return tuple((1.0, 0.0) for _ in texts)

    monkeypatch.setattr(operation_index, "LocalEmbeddingAdapter", lambda *_args: Adapter())
    monkeypatch.setattr(operation_index, "weights_digest", lambda _path: profile.weights_sha256)
    args = argparse.Namespace(catalog=sources_files[0], scope=sources_files[1], index=root,
                              weights=tmp_path / "unused", base_url="http://127.0.0.1:1234/v1")
    with pytest.raises(RetrievalError, match="source_changed"):
        await operation_index.build(args, profile)
    assert (root / "manifest.json").read_bytes() == before


def test_manifest_source_scope_hash_is_checked(tmp_path, sources_files, profile) -> None:
    root, sources = build_index(tmp_path, sources_files, profile)
    with pytest.raises(RetrievalError, match="stale_index"):
        load_index(root, replace(sources, scope_sha256="f" * 64), profile)


def test_retrieval_dependency_boundary_has_no_execution_or_model_writer() -> None:
    app = DEFAULT_CATALOG.parents[1]
    forbidden = ("app.db", "app.models", "app.workers", "app.language_operations", "app.interpretation",
                 "app.operations.bootstrap", "app.operations.service", "app.operations.handlers", "evaluation")
    for file in (app / "retrieval").glob("*.py"):
        tree = ast.parse(file.read_text(encoding="utf-8"))
        imports = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module]
        imports += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
        assert not any(name.startswith(forbidden) for name in imports), file
    for file in (app / "interpretation").glob("*.py"):
        assert "app.retrieval.builder" not in file.read_text(encoding="utf-8")
