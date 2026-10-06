"""D30 normal HTTP/runtime wiring with deterministic boundary adapters only."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from app.api import routes_language
from app.core.config import get_settings
from app.db import get_session_factory
from app.main import create_app
from app.models.job import GenerationJob, JobStatus
from app.models.project import Project
from app.models.operation_request import OperationReceipt
from app.operations.bootstrap import operation_service
from app.operations.contracts import OperationRequest, OperationTarget
from app.semantic_interpretation import runtime
from app.workers.operation_dispatcher import mark_interrupted_operation_jobs
from tests.test_semantic_interpretation import QUESTION, SET, UNSUPPORTED, Replies, setup as setup

START = {"kind": "operation", "operation_id": "project.generation.start", "operation_version": 1,
         "arguments": {"kind": "full"}}
SPEED = {"kind": "operation", "operation_id": "project.settings.update", "operation_version": 1,
         "arguments": {"voicevox_speed_scale": 1.2}}


@pytest.fixture(params=["all_tools", "stateful"])
def host(request: Any, setup: Any, temp_storage: Path, tmp_path: Path, monkeypatch: Any):
    _, encoder, _, _, profile = setup
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(profile.model_dump_json(), encoding="utf-8")
    settings = get_settings()
    settings.language_model = "fixed-test-adapter"
    settings.language_retrieval_index = tmp_path if request.param == "stateful" else None
    settings.language_retrieval_profile = profile_path
    settings.language_retrieval_readiness = True
    replies = Replies([])

    async def complete(self: Any, messages: Any, schema: Any) -> str:
        return await replies.complete(messages, schema)

    class Embeddings:
        def __init__(self, *args: Any) -> None:
            pass

        async def __aenter__(self) -> Any:
            return encoder

        async def __aexit__(self, *args: Any) -> None:
            pass

    monkeypatch.setattr(routes_language._LocalAdapter, "complete", complete)
    monkeypatch.setattr(runtime, "LocalEmbeddingAdapter", Embeddings)
    # No service override: exercise actual route -> configured runtime -> common core.
    client = TestClient(create_app())  # dispatcher omitted to inspect queued work
    project = client.post('/api/projects', json={"title": "D30 synthetic", "source_script": "結合試験の合成原稿です。",
                                                "use_fake_providers": True}).json()
    yield client, replies, encoder, settings, project, request.param, profile_path
    client.close()


def payload(project: dict[str, Any], request_id: str, text: str, revision: int = 1) -> dict[str, Any]:
    return {"request_id": request_id, "text": text, "base_revision": revision, "target": {"project_id": project["id"]}}


def counts() -> tuple[int, int, int]:
    with get_session_factory()() as db:
        return (db.scalar(select(func.count()).select_from(GenerationJob)),
                db.scalar(select(func.count()).select_from(OperationReceipt)),
                db.scalar(select(func.count()).select_from(GenerationJob).where(GenerationJob.cancel_requested.is_(True))))


def test_normal_wiring_save_and_replay_survive_mode_and_connection_change(host: Any) -> None:
    client, replies, encoder, settings, project, mode, _ = host
    assert client.get('/api/language/connection').json()["operation_mode"] == mode
    replies.replies = [SET] * 3
    request = payload(project, "saved", "字幕を56pxにして")
    saved = client.post('/api/language/requests', json=request).json()
    assert saved["status"] == "completed" and saved["result"]["revision"] == 2
    assert counts() == (0, 1, 0)
    assert saved["mode"] == ("semantic" if mode == "stateful" else "all_tools")
    assert bool(encoder.calls) == (mode == "stateful")
    assert all("execution_state" not in call for call in replies.calls)  # no D29 experiment prompt
    if mode == "stateful":
        assert all(s["candidate_state"] for s in saved["diagnostics"]["retrieval"]["stages"])
    previous = len(replies.calls), encoder.calls
    settings.language_retrieval_index = None if mode == "stateful" else Path("missing-index")
    settings.language_model = None
    replay = client.post('/api/language/requests', json=request).json()
    assert replay["result"] == saved["result"] and replay["diagnostics"] == saved["diagnostics"]
    assert (len(replies.calls), encoder.calls) == previous and counts() == (0, 1, 0)
    assert client.post('/api/language/requests', json={**request, "text": "字幕を60pxにして"}).status_code == 409


def test_explicit_generation_busy_refusal_and_safe_recovery_without_model(host: Any) -> None:
    client, replies, encoder, settings, project, mode, _ = host
    replies.replies = [START] * 3
    request = payload(project, "generate", "現在の設定で動画を生成して")
    ready = client.post('/api/language/requests', json=request).json()
    assert ready["requires_confirmation"] and ready["status"] == "ready" and counts()[0] == 0
    confirm = {"confirmation_token": ready["confirmation_token"], "confirm_generation": True}
    result = client.post('/api/language/requests/generate/execute', json=confirm).json()
    duplicate = client.post('/api/language/requests/generate/execute', json=confirm).json()
    assert duplicate == result and counts()[0] == 1
    before = counts()
    replies.replies = [SPEED] * 3
    blocked = client.post('/api/language/requests', json=payload(project, "busy", "話速を1.2倍にして")).json()
    assert blocked["status"] == "blocked" and blocked["failure"]["reason_code"] == "project_busy"
    assert counts() == before
    if mode == "stateful":
        rows = blocked["diagnostics"]["retrieval"]["stages"][-1]["candidate_state"]["candidates"]
        assert any(c["operation_id"] == SPEED["operation_id"] and c["readiness"] == "blocked" for c in rows)
    with get_session_factory()() as db:
        job = db.scalar(select(GenerationJob))
        job_id = job.id
        job.status = JobStatus.running  # interrupted local-work fixture
        db.commit()
    settings.language_model = None
    settings.language_retrieval_index = Path("missing-index")
    assert mark_interrupted_operation_jobs() == 1
    assert mark_interrupted_operation_jobs() == 0
    with get_session_factory()() as db:
        assert db.get(GenerationJob, job_id).status == JobStatus.pending
        assert db.get(Project, project["id"]).voicevox_speed_scale == 1.0
    calls = len(replies.calls), encoder.calls
    replay = client.post('/api/language/requests', json=request).json()
    assert replay["result"]["job_id"] == result["result"]["job_id"]
    assert (len(replies.calls), encoder.calls) == calls and counts() == before


@pytest.mark.parametrize("reply,text,status", [(QUESTION, "字幕を大きくして", "needs_input"),
                                               (UNSUPPORTED, "動画をメールで送って", "unsupported")])
def test_missing_and_unsupported_do_not_execute(host: Any, reply: dict, text: str, status: str) -> None:
    client, replies, _, _, project, *_ = host
    replies.replies = [reply] * 3
    result = client.post('/api/language/requests', json=payload(project, "question", text)).json()
    assert result["status"] == status and not result["executed"] and counts() == (0, 0, 0)


def test_index_profile_mismatch_fails_closed_but_all_tools_ignores_index(host: Any) -> None:
    client, replies, encoder, _, project, mode, profile_path = host
    profile_path.write_text('{}', encoding='utf-8')
    replies.replies = [SET] * 3
    result = client.post('/api/language/requests', json=payload(project, "bad-index", "字幕を56pxにして")).json()
    if mode == "stateful":
        assert result["status"] == "error" and result["diagnostics"]["retrieval"]["reason"] == "integrity_failure"
        assert not replies.calls and not encoder.calls and counts() == (0, 0, 0)
    else:
        assert result["status"] == "completed" and not encoder.calls


def test_unknown_operation_never_reaches_handler_after_fallback(host: Any) -> None:
    client, replies, _, _, project, *_ = host
    replies.replies = [{"kind": "operation", "operation_id": "unknown.execute", "operation_version": 1,
                        "arguments": {}}] * 4
    response = client.post('/api/language/requests', json=payload(project, "unknown", "字幕を56pxにして")).json()
    assert response["status"] == "error" and counts() == (0, 0, 0)
    assert len(replies.calls) <= 4


def test_state_becoming_stale_during_inference_is_rechecked(host: Any) -> None:
    client, replies, _, _, project, *_ = host
    replies.replies = [SPEED] * 3
    def race() -> None:
        if len(replies.calls) == 1:
            with get_session_factory()() as db:
                operation_service.execute(db, OperationRequest(operation_id="project.subtitle-font-size.set",
                    target=OperationTarget(project_id=project["id"]), arguments={"value": 60},
                    request_id="concurrent-save", base_revision=1))
    replies.on_call = race
    response = client.post('/api/language/requests', json=payload(project, "race", "話速を1.2倍にして")).json()
    assert response["status"] == "blocked"
    assert response["failure"]["reason_code"] == "stale_state"
    current = client.get(f'/api/projects/{project["id"]}').json()
    assert current["subtitle_font_size"] == 60 and current["voicevox_speed_scale"] == 1.0
