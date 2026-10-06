"""D27 candidate isolation, bounded costs, integrity and durable effect boundary."""
from __future__ import annotations

import asyncio
import ast
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.api.routes_language import language_service
from app.interpretation.contracts import CandidateRef, InterpretationInput, MinimalState
from app.interpretation.errors import InterpretationError
from app.language_operations.contracts import LanguageResponse
from app.language_operations.observability import diagnostic_event
from app.language_operations.service import LanguageOperationService
from app.main import create_app
from app.operations.bootstrap import operation_service
from app.operations.catalog import OperationCatalog
from app.retrieval.builder import publish_index
from app.retrieval.contracts import EmbeddingProfile, OperationRef, SearchScope
from app.retrieval.ranking import rank_operations
from app.retrieval.reader import load_index
from app.retrieval.serialization import RetrievalError
from app.retrieval.sources import DEFAULT_CATALOG, load_sources
from app.semantic_interpretation.runtime import configured_semantic
from app.semantic_interpretation.service import SemanticInterpreter

CATALOG = OperationCatalog(definitions=tuple(operation_service.list_definitions()))
REFS = tuple(CandidateRef(operation_id=d.operation_id, operation_version=d.operation_version) for d in CATALOG.definitions)
UNSUPPORTED = {"kind": "unsupported", "reason": "候補にありません。"}
QUESTION = {"kind": "clarification", "question": "対象と値を教えてください。", "missing_fields": ["intent"]}
SET = {"kind": "operation", "operation_id": "project.subtitle-font-size.set", "operation_version": 1, "arguments": {"value": 56}}


class Encoder:
    calls = 0
    failure = False

    async def embed_query(self, text: str) -> tuple[float, ...]:
        self.calls += 1
        if self.failure:
            raise RetrievalError("private-path-must-not-leak")
        return (1.0, 0.0)


class Replies:
    def __init__(self, replies: list[Any]) -> None:
        self.replies, self.calls = replies, []
        self.on_call = lambda: None

    async def complete(self, messages: Any, schema: Any) -> str:
        self.calls.append(json.loads(messages[1].content))
        self.on_call()
        value = self.replies.pop(0)
        if isinstance(value, Exception):
            raise value
        return value if isinstance(value, str) else json.dumps({"result": value}, ensure_ascii=False)


def build_setup(tmp_path: Path) -> tuple[Any, Any, Any, Any, Any]:
    sources = load_sources()
    profile = EmbeddingProfile(model="synthetic", weights_sha256="a" * 64, dimensions=2,
        document_prefix="document: ", query_prefix="query: ")
    publish_index(tmp_path, sources, profile, tuple((1.0, 0.0) for _ in sources.documents))
    encoder = Encoder()
    scope = SearchScope(app_id="blockvideo",
        capabilities=tuple(sorted({c for d in sources.documents for c in d.required_capabilities})),
        operations=tuple(OperationRef(operation_id=r.operation_id, operation_version=r.operation_version) for r in REFS))
    runner = SemanticInterpreter(lambda s: load_index(tmp_path, s, profile), lambda: sources, encoder, scope=scope)
    return runner, encoder, scope, sources, profile


@pytest.fixture
def setup(tmp_path: Path):
    return build_setup(tmp_path)


async def run(runner: SemanticInterpreter, adapter: Replies, text: str = "字幕を56pxにして"):
    return await runner.preview(CATALOG, adapter, InterpretationInput(text=text, candidates=REFS,
        state=MinimalState(selected_project_id=1, revision=1, subtitle_font_size=48)))


@pytest.mark.parametrize("first", [UNSUPPORTED, QUESTION, "broken", SET], ids=["missing", "question", "json", "out-of-scope"])
async def test_narrow_failure_expands_once_then_full_and_only_final_is_proposal(setup, first) -> None:
    runner, encoder, *_ = setup
    adapter = Replies([first, UNSUPPORTED, SET])
    # No lexical overlap with any annotation, so uniform vectors decide the stages.
    result = await run(runner, adapter, text="zq")
    assert result.interpretation.status == "proposed"
    assert result.interpretation.proposal.arguments == {"value": 56}
    assert [len(c["candidates"]) for c in adapter.calls] == [5, 8, 10]
    assert encoder.calls == result.trace.embedding_calls == 1
    assert result.trace.chat_calls == 3
    assert result.trace.expansion_count == result.trace.all_tools_count == 1
    assert all(s.request_bytes > 0 and s.response_bytes > 0 for s in result.trace.stages)
    assert not result.interpretation.executed


async def test_full_scope_allows_one_repair_and_stops_at_four_calls(setup) -> None:
    runner, *_ = setup
    adapter = Replies([UNSUPPORTED, UNSUPPORTED, "invalid", "invalid"])
    result = await run(runner, adapter)
    assert result.interpretation.status == "error"
    assert result.trace.chat_calls == 4
    assert result.interpretation.attempts == 2


async def test_unsupported_is_only_final_after_all_supported_candidates(setup) -> None:
    runner, *_ = setup
    result = await run(runner, Replies([UNSUPPORTED] * 3), "動画をメールして")
    assert result.interpretation.status == "unsupported" and result.trace.chat_calls == 3
    assert [s.result for s in result.trace.stages] == ["unsupported"] * 3


async def test_no_all_tools_returns_question_not_unsupported(setup) -> None:
    runner, *_ = setup
    runner._allow_all = False
    result = await run(runner, Replies([UNSUPPORTED] * 2))
    assert result.interpretation.status == "needs_input" and result.trace.chat_calls == 2
    assert result.trace.reason == "fallback_unavailable"
    assert result.trace.all_tools_count == 0


@pytest.mark.parametrize("replies,status,text", [([UNSUPPORTED] * 3, "unsupported", "動画をメールして"),
                                                  ([QUESTION, UNSUPPORTED, UNSUPPORTED], "needs_input", "動画をメールして"),
                                                  ([UNSUPPORTED, UNSUPPORTED, SET], "proposed", "字幕を56pxにして")])
async def test_catalog_too_large_for_all_tools_trusts_only_unanimous_unsupported(setup, monkeypatch,
                                                                                replies, status, text) -> None:
    runner, *_ = setup
    # Pretend the 9-operation catalog cannot be shown at once (as with 1,000 operations).
    monkeypatch.setattr("app.semantic_interpretation.service.MAX_PROMPT_CANDIDATES", 5)
    result = await run(runner, Replies(list(replies)), text)
    assert result.interpretation.status == status and result.trace.all_tools_count == 0
    assert [stage.name for stage in result.trace.stages] == ["initial", "expanded", "wide"]
    assert result.trace.reason == {"unsupported": "unsupported_without_full_scope",
                                   "needs_input": "fallback_unavailable",
                                   "proposed": "candidate_insufficient"}[status]


@pytest.mark.parametrize("reason", ["connection_failed", "timeout", "http_error", "model_mismatch"])
async def test_model_transport_failure_never_retried_as_search_expansion(setup, reason: str) -> None:
    runner, *_ = setup
    result = await run(runner, Replies([InterpretationError(reason)]))
    assert result.trace.chat_calls == 1 and result.trace.expansion_count == 0
    assert result.interpretation.failure.reason_code == reason


async def test_embedding_failure_falls_back_without_a_similarity_or_leaked_exception(setup) -> None:
    runner, encoder, *_ = setup
    encoder.failure = True
    result = await run(runner, Replies([SET]))
    assert result.interpretation.status == "proposed" and result.trace.all_tools_count == 1
    assert result.trace.reason == "search_unavailable" and not result.trace.ranking
    assert "private-path" not in result.model_dump_json()


async def test_long_query_is_not_truncated_and_no_encoder_called(setup) -> None:
    runner, encoder, *_ = setup
    adapter = Replies([QUESTION])
    text = "長い依頼" * 150
    result = await run(runner, adapter, text)
    assert result.trace.embedding_calls == encoder.calls == 0
    assert result.trace.all_tools_count == 1 and adapter.calls[0]["request"] == text


@pytest.mark.parametrize("changed", ["app", "caps"], ids=["other-app", "no-capabilities"])
async def test_empty_scope_does_not_call_model_or_assert_unsupported(setup, changed: str) -> None:
    runner, encoder, scope, *_ = setup
    runner._scope = scope.model_copy(update={"app_id": "another-app"} if changed == "app" else {"capabilities": ()})
    adapter = Replies([])
    result = await run(runner, adapter)
    assert result.interpretation.status == "needs_input" and result.trace.reason == "no_candidates"
    assert not adapter.calls and encoder.calls == 0


async def test_fallback_never_widens_capabilities_and_preserves_blocked_state(setup) -> None:
    runner, _, scope, *_ = setup
    runner._scope = scope.model_copy(update={"capabilities": ("generation.start",)})
    adapter = Replies([UNSUPPORTED])
    request = InterpretationInput(text="生成して", candidates=REFS, state=MinimalState(status="generating"))
    result = await runner.preview(CATALOG, adapter, request)
    assert result.interpretation.status == "unsupported"
    assert len(adapter.calls[0]["candidates"]) == 1
    assert adapter.calls[0]["candidates"][0]["operation_id"] == "project.generation.start"
    assert adapter.calls[0]["state"]["status"] == "generating"


async def test_source_changed_during_inference_discards_otherwise_valid_proposal(setup) -> None:
    runner, _, _, sources, _ = setup
    adapter = Replies([{"kind": "operation", "operation_id": "project.generation.start", "operation_version": 1, "arguments": {"kind": "full"}}])
    adapter.on_call = lambda: setattr(runner, "_sources", lambda: replace(sources, scope_sha256="f" * 64))
    result = await run(runner, adapter)
    assert result.interpretation.proposal is None and result.trace.reason == "integrity_failure"
    assert result.trace.chat_calls == 1


async def test_generation_after_save_cannot_escape_host_scope(setup) -> None:
    runner, _, scope, *_ = setup
    runner._scope = scope.model_copy(update={"capabilities": ("settings.write",)})
    result = await run(runner, Replies([{**SET, "generate_after_save": True}]))
    assert result.interpretation.status == "needs_input" and result.interpretation.proposal.kind == "clarification"
    assert result.trace.reason == "fallback_unavailable"


async def test_prompt_order_is_canonical_even_when_ranking_is_different(setup, monkeypatch) -> None:
    from app.semantic_interpretation import service
    runner, _, scope, sources, _ = setup
    ranking = rank_operations(runner._load(sources), (1.0, 0.0), scope, sources)
    monkeypatch.setattr(service, "rank_operations", lambda *_args, **_kwargs: tuple(reversed(ranking)))
    adapter = Replies([SET])
    result = await run(runner, adapter)
    actual = [(r["operation_id"], r["operation_version"]) for r in adapter.calls[0]["candidates"]]
    assert actual == sorted(r.key for r in tuple(reversed(ranking))[:5])
    assert result.trace.ranking[0].key == ranking[-1].key


async def test_stale_index_never_falls_back_to_model(setup) -> None:
    runner, encoder, _, sources, _ = setup
    runner._sources = lambda: replace(sources, scope_sha256="f" * 64)
    result = await run(runner, Replies([]))
    assert result.trace.reason == "integrity_failure" and not result.interpretation.proposal
    assert encoder.calls == result.trace.chat_calls == 0


async def test_shared_deadline_includes_encoder_and_has_no_post_timeout_inference(setup) -> None:
    runner, encoder, *_ = setup
    async def slow(_text: str) -> tuple[float, ...]:
        await asyncio.sleep(1)
        return (1.0, 0.0)
    encoder.embed_query = slow
    runner._timeout = 0.01
    result = await run(runner, Replies([]))
    assert result.trace.reason == "deadline" and result.trace.chat_calls == 0


def test_ranking_deduplicates_versions_and_keeps_stable_ties(setup) -> None:
    runner, _, scope, sources, _ = setup
    index = runner._load(sources)
    ranked = rank_operations(index, (4.0, 0.0), scope, sources)
    assert len(ranked) == 10 and all(c.score == 1.0 for c in ranked)
    assert [r.key for r in ranked] == sorted({d.key for d in sources.documents})
    assert len([r for r in ranked if r.operation_id == "project.settings.update"]) == 2
    assert [r.score for r in rank_operations(index, (-1.0, 0.0), scope, sources)] == [-1.0] * 10
    with pytest.raises(RetrievalError):
        rank_operations(index, (float("nan"), 0.0), scope, sources)


def test_lazy_runtime_construction_does_not_touch_missing_files(tmp_path: Path) -> None:
    runner = configured_semantic(tmp_path / "absent", tmp_path / "profile", tmp_path / "assets", "http://invalid")
    assert runner is not None


def test_search_trace_survives_save_reload_replay_and_contains_no_log_text(setup, temp_storage: Path) -> None:
    runner, encoder, *_ = setup
    adapter = Replies([UNSUPPORTED, UNSUPPORTED, SET])
    service = LanguageOperationService(operation_service, adapter, semantic=runner)
    app = create_app()
    app.dependency_overrides[language_service] = lambda: service
    with TestClient(app) as client:
        project = client.post("/api/projects", json={"title": "D27 synthetic", "source_script": "非公開テスト文字列",
            "subtitle_font_size": 48, "use_fake_providers": True}).json()
        payload = {"request_id": "PRIVATE-d27", "text": "字幕を56pxにして", "target": {"project_id": project["id"]}, "base_revision": 1}
        first = client.post("/api/language/requests", json=payload).json()
        assert first["status"] == "completed" and first["result"]["revision"] == 2
        encoder.failure = True
        replay = client.post("/api/language/requests", json=payload).json()
        lookup = client.get("/api/language/requests/PRIVATE-d27").json()
        assert replay["diagnostics"] == lookup["diagnostics"] == first["diagnostics"]
        assert len(adapter.calls) == 3 and encoder.calls == 1
        assert first["mode"] == replay["mode"] == "semantic"
        event = diagnostic_event(LanguageResponse.model_validate(first), "prepared")
        logged = json.dumps(event, ensure_ascii=False)
        assert "PRIVATE-d27" not in logged and "字幕を" not in logged and "非公開" not in logged
        assert event["retrieval"]["chat_calls"] == 3


def test_semantic_layer_cannot_import_execution_or_index_writer() -> None:
    folder = DEFAULT_CATALOG.parents[1] / "semantic_interpretation"
    forbidden = ("app.db", "app.models", "app.workers", "app.language_operations", "app.operations.service",
        "app.operations.handlers", "app.operations.bootstrap", "app.retrieval.builder", "evaluation")
    for file in folder.glob("*.py"):
        tree = ast.parse(file.read_text(encoding="utf-8"))
        imports = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module]
        imports += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
        assert not any(name.startswith(forbidden) for name in imports), file


def test_confirmation_and_busy_validation_remain_in_common_core(setup, temp_storage: Path) -> None:
    from sqlalchemy import func, select
    from app.db import get_session_factory
    from app.models.job import GenerationJob
    runner, encoder, *_ = setup
    generate = {"kind": "operation", "operation_id": "project.generation.start", "operation_version": 1, "arguments": {"kind": "full"}}
    adapter = Replies([generate, UNSUPPORTED, UNSUPPORTED, SET])
    app = create_app()
    app.dependency_overrides[language_service] = lambda: LanguageOperationService(operation_service, adapter, semantic=runner)
    client = TestClient(app)  # no dispatcher; pending jobs can be inspected deterministically
    try:
        project = client.post("/api/projects", json={"title": "D27", "source_script": "合成台本。", "use_fake_providers": True}).json()
        payload = {"request_id": "gen", "text": "動画を作り直して", "target": {"project_id": project["id"]}, "base_revision": 1}
        prepared = client.post("/api/language/requests", json=payload).json()
        assert prepared["status"] == "ready" and prepared["requires_confirmation"] and not prepared["executed"]
        def jobs() -> int:
            with get_session_factory()() as db:
                return db.scalar(select(func.count()).select_from(GenerationJob))
        assert jobs() == 0
        confirm = {"confirmation_token": prepared["confirmation_token"], "confirm_generation": False}
        assert client.post("/api/language/requests/gen/execute", json=confirm).status_code == 409
        confirm["confirm_generation"] = True
        first = client.post("/api/language/requests/gen/execute", json=confirm).json()
        again = client.post("/api/language/requests/gen/execute", json=confirm).json()
        assert first["result"] == again["result"] and jobs() == 1 and encoder.calls == 1
        blocked = client.post("/api/language/requests", json={**payload, "request_id": "busy", "text": "字幕を56pxにして"}).json()
        assert blocked["status"] == "blocked" and not blocked["executed"]
        current = client.get(f'/api/projects/{project["id"]}').json()
        assert current["revision"] == 1 and current["subtitle_font_size"] == 48
        assert jobs() == 1
    finally:
        client.close()
