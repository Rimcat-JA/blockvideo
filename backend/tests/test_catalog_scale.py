"""Scale: a 1,000-operation synthetic catalog is indexed, searched and narrowed."""
from __future__ import annotations

import json
import time
from pathlib import Path

from app.retrieval.builder import publish_index
from app.retrieval.contracts import EmbeddingProfile, OperationRef, SearchScope
from app.retrieval.ranking import rank_operations
from app.retrieval.reader import load_index
from app.retrieval.sources import DEFAULT_CATALOG, load_sources

COUNT = 1000
PROFILE = EmbeddingProfile(model="synthetic", weights_sha256="a" * 64, dimensions=2,
                           document_prefix="document: ", query_prefix="query: ")
SUBJECTS = ("在庫", "請求書", "会議室", "配送", "勤怠", "見積", "問い合わせ", "契約", "備品", "研修")
VERBS = ("確認", "登録", "更新", "取消", "集計", "承認", "差し戻し", "通知", "検索", "出力")
PLACES = ("札幌", "仙台", "東京", "横浜", "名古屋", "京都", "大阪", "神戸", "広島", "福岡")


def _phrase(index: int) -> str:
    # Unique wording per operation from distinct words (place x subject x verb).
    # Character bigrams separate words well but not near-identical numbers.
    return f"{PLACES[index // 100]}支店の{SUBJECTS[index % 10]}を{VERBS[(index // 10) % 10]}して"


def _write_catalog(root: Path) -> None:
    template = next(item for item in json.loads(DEFAULT_CATALOG.read_text(encoding="utf-8"))["operations"]
                    if item["operation_id"] == "project.status.get")
    operations, bindings, annotations = [], [], {}
    for index in range(COUNT):
        operation_id = f"project.synthetic-{index:04d}.run"
        operations.append({**template, "operation_id": operation_id,
                           "description": f"Synthetic operation {index}.", "examples": [f"Synthetic {index}"]})
        bindings.append({"operation_id": operation_id, "operation_version": 1,
                         "required_capabilities": ["project.status.read"]})
        annotations[operation_id] = {"utterances": [_phrase(index)],
                                     "distinctions": ["合成の規模試験用の操作。"]}
    (root / "definitions.json").write_text(json.dumps({"operations": operations}, ensure_ascii=False), encoding="utf-8")
    (root / "search_scope.json").write_text(json.dumps({"format_version": 1, "app_id": "blockvideo",
                                                        "bindings": bindings}), encoding="utf-8")
    (root / "operation_annotations.json").write_text(json.dumps(
        {"format_version": 1, "operations": annotations}, ensure_ascii=False), encoding="utf-8")


def test_thousand_operation_catalog_is_searchable_and_narrowed(tmp_path: Path) -> None:
    catalog_dir = tmp_path / "catalog"
    catalog_dir.mkdir()
    _write_catalog(catalog_dir)
    sources = load_sources(catalog_dir / "definitions.json", catalog_dir / "search_scope.json")
    assert sources.operation_count == COUNT
    publish_index(tmp_path / "index", sources, PROFILE, tuple((1.0, 0.0) for _ in sources.documents))
    index = load_index(tmp_path / "index", sources, PROFILE)
    scope = SearchScope(app_id="blockvideo", capabilities=("project.status.read",),
        operations=tuple(OperationRef(operation_id=f"project.synthetic-{i:04d}.run", operation_version=1)
                         for i in range(COUNT)))
    started = time.perf_counter()
    for index_number in (0, 137, 512, 999):
        ranking = rank_operations(index, (1.0, 0.0), scope, sources, query_text=_phrase(index_number))
        assert ranking[0].operation_id == f"project.synthetic-{index_number:04d}.run"
        # Only the first stage (5 candidates) would reach the model, never the whole catalog.
        assert len(ranking[:5]) == 5 and len(ranking) == COUNT
    assert time.perf_counter() - started < 30
