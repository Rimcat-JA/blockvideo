"""D28 provisional states, retained blocked candidates and immutable effect boundary."""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sqlalchemy import func, select

from app.api.routes_language import language_service
from app.db import get_session_factory
from app.interpretation.contracts import CandidateRef, InterpretationInput, MinimalState
from app.language_operations.candidate_state import current_candidates
from app.language_operations.contracts import LanguageResponse
from app.language_operations.service import LanguageOperationService
from app.main import create_app
from app.models.job import GenerationJob, JobStatus
from app.models.operation_request import OperationReceipt
from app.models.project import Project, ProjectStatus
from app.operations.bootstrap import operation_service
from app.operations.candidate_readiness import candidate_snapshot
from app.operations.contracts import CandidateReadiness, CandidateReadinessSnapshot, OperationRequest, OperationTarget
from evaluation.readiness_filter import hard_filter_candidates
from tests.test_semantic_interpretation import CATALOG, REFS, QUESTION, Replies, setup as setup

SPEED = {"kind": "operation", "operation_id": "project.settings.update", "operation_version": 1,
         "arguments": {"voicevox_speed_scale": 1.2}}


def create(client: TestClient) -> dict[str, Any]:
    return client.post("/api/projects", json={"title": "D28 synthetic", "source_script": "合成の検証原稿です。",
                                             "use_fake_providers": True}).json()


def start(project: dict[str, Any]) -> int:
    with get_session_factory()() as db:
        result = operation_service.execute(db, OperationRequest(operation_id="project.generation.start",
            target=OperationTarget(project_id=project["id"]), arguments={"kind": "full"},
            request_id=f"start-{project['id']}", base_revision=project["revision"]))
        return result.job_id


def finish_fixture(project_id: int, job_id: int) -> None:
    """A terminal cancellation fixture has no completed-video requirement."""
    with get_session_factory()() as db:
        job = db.get(GenerationJob, job_id)
        job.status = JobStatus.cancelled
        project = db.get(Project, project_id)
        project.status = ProjectStatus.cancelled
        db.commit()


def counts() -> tuple[int, int, int]:
    with get_session_factory()() as db:
        return (db.scalar(select(func.count()).select_from(GenerationJob)),
                db.scalar(select(func.count()).select_from(OperationReceipt)),
                db.scalar(select(func.count()).select_from(GenerationJob).where(GenerationJob.cancel_requested.is_(True))))


@pytest.fixture
def harness(setup: Any, temp_storage: Path):
    runner, encoder, *_ = setup
    adapter = Replies([])
    app = create_app()
    app.dependency_overrides[language_service] = lambda: LanguageOperationService(
        operation_service, adapter, semantic=runner, readiness_annotations=True)
    client = TestClient(app)  # no dispatcher: inspect real durable pending jobs
    yield client, adapter, runner, encoder
    client.close()


@pytest.mark.parametrize("target", [None, 999], ids=["missing-target", "missing-project"])
def test_target_hints_do_not_guess_values(harness: Any, target: int | None) -> None:
    with get_session_factory()() as db:
        result = candidate_snapshot(db, CATALOG.definitions, OperationTarget(project_id=target))
        assert {r.reason_code for r in result.candidates} == {"target_required" if target is None else "target_not_found"}
        assert all(not r.arguments_checked and r.phase == "candidate_preview" for r in result.candidates)
    assert counts() == (0, 0, 0)


def test_argument_checks_are_unknown_even_when_target_is_editable(harness: Any) -> None:
    client, *_ = harness
    project = create(client)
    with get_session_factory()() as db:
        snapshot = candidate_snapshot(db, CATALOG.definitions, OperationTarget(project_id=project["id"]))
    for row in snapshot.candidates:
        if row.operation_id == "project.status.get":
            assert row.readiness == "ready"
        else:
            assert row.readiness == "needs_input" and row.reason_code == "arguments_unchecked"
            assert row.missing_fields
    assert counts() == (0, 0, 0)


def test_identical_intent_keeps_same_rank_and_blocks_without_cancel_or_queue(harness: Any) -> None:
    client, adapter, _, encoder = harness
    editable, busy = create(client), create(client)
    job_id = start(busy)
    adapter.replies = [SPEED, SPEED]
    def send(project: dict[str, Any], request_id: str) -> dict[str, Any]:
        return client.post("/api/language/requests", json={"request_id": request_id,
            "text": "話速を1.2倍にして", "target": {"project_id": project["id"]}, "base_revision": 1}).json()
    saved = send(editable, "editable")
    before = counts()
    blocked = send(busy, "busy")
    assert saved["status"] == "completed" and saved["result"]["resolved_arguments"] == {"voicevox_speed_scale": 1.2}
    assert blocked["status"] == "blocked" and blocked["failure"]["reason_code"] == "project_busy"
    assert blocked["prepared_request"]["operation_id"] == SPEED["operation_id"]
    assert counts() == before and blocked["result"] is None
    ranks = [r["diagnostics"]["retrieval"]["ranking"] for r in (saved, blocked)]
    assert ranks[0] == ranks[1]
    hints = [next(c["readiness_hint"] for c in call["candidates"] if c["operation_id"] == SPEED["operation_id"])
             for call in adapter.calls]
    assert hints[0]["reason_code"] == "arguments_unchecked"
    assert hints[1]["reason_code"] == "project_busy"
    assert len(adapter.calls) == encoder.calls == 2
    with get_session_factory()() as db:
        assert db.get(Project, busy["id"]).voicevox_speed_scale == 1.0
        assert db.get(GenerationJob, job_id).status == JobStatus.pending
    # Refresh does not alter the stored blocked receipt, call a model or read an index.
    assert client.get('/api/language/requests/busy/candidate-readiness').status_code == 200
    finish_fixture(busy["id"], job_id)
    encoder.failure = True
    refreshed = client.get('/api/language/requests/busy/candidate-readiness').json()
    assert all(c["reason_code"] != "project_busy" for c in refreshed["candidates"])
    replay = send(busy, "busy")
    assert replay["status"] == "blocked" and replay["diagnostics"] == blocked["diagnostics"]
    assert len(adapter.calls) == encoder.calls == 2
    assert counts() == before
    encoder.failure = False
    adapter.replies = [SPEED]
    assert send(busy, "after-finish")["status"] == "completed"


def test_all_fallback_stages_keep_blocked_and_refresh_same_revision(harness: Any) -> None:
    client, adapter, *_ = harness
    project = create(client)
    job_id = start(project)
    adapter.replies = [QUESTION, QUESTION, SPEED]
    adapter.on_call = lambda: finish_fixture(project["id"], job_id) if len(adapter.calls) == 1 else None
    response = client.post("/api/language/requests", json={"request_id": "changing", "text": "話速を1.2倍にして",
        "target": {"project_id": project["id"]}, "base_revision": 1, "review_all": True}).json()
    stages = response["diagnostics"]["retrieval"]["stages"]
    assert [len(s["candidates"]) for s in stages] == [5, 8, 10]
    settings = [next(c for c in s["candidate_state"]["candidates"]
                     if c["operation_id"] == "project.settings.update" and c["operation_version"] == 1) for s in stages]
    assert [c["revision"] for c in settings] == [1, 1, 1]
    assert [c["reason_code"] for c in settings] == ["project_busy", "arguments_unchecked", "arguments_unchecked"]
    assert response["status"] == "ready" and not response["executed"]
    assert LanguageResponse.model_validate(response).diagnostics.retrieval.stages[0].candidate_state


def test_busy_recheck_after_model_and_after_confirmation(harness: Any) -> None:
    client, adapter, *_ = harness
    first = create(client)
    adapter.replies = [SPEED]
    adapter.on_call = lambda: start(first)
    blocked = client.post("/api/language/requests", json={"request_id": "race-model", "text": "話速を1.2倍にして",
        "target": {"project_id": first["id"]}, "base_revision": 1}).json()
    assert blocked["status"] == "blocked"
    hint = next(c for c in blocked["diagnostics"]["retrieval"]["stages"][0]["candidate_state"]["candidates"]
                if c["operation_id"] == "project.settings.update")
    assert hint["reason_code"] == "arguments_unchecked"
    second = create(client)
    adapter.on_call = lambda: None
    adapter.replies = [SPEED]
    ready = client.post("/api/language/requests", json={"request_id": "race-confirm", "text": "話速を1.2倍にして",
        "target": {"project_id": second["id"]}, "base_revision": 1, "review_all": True}).json()
    assert ready["status"] == "ready"
    start(second)
    before = counts()
    result = client.post('/api/language/requests/race-confirm/execute', json={
        "confirmation_token": ready["confirmation_token"], "confirm_generation": False}).json()
    assert result["status"] == "blocked" and counts() == before
    assert client.get(f'/api/projects/{second["id"]}').json()["voicevox_speed_scale"] == 1.0


@pytest.mark.parametrize("text", ["話速を変えて", "キャンセルして", "以前の設定に戻して"])
def test_unknown_arguments_or_references_are_questions_not_guessed_operations(harness: Any, text: str) -> None:
    client, adapter, *_ = harness
    project = create(client)
    adapter.replies = [QUESTION] * 3
    response = client.post('/api/language/requests', json={"request_id": "missing", "text": text,
        "target": {"project_id": project["id"]}, "base_revision": 1}).json()
    assert response["status"] == "needs_input" and not response["executed"]
    assert counts() == (0, 0, 0)


@pytest.mark.parametrize("defect", ["missing", "duplicate", "other-version", "other-target"])
def test_interpreter_rejects_mismatched_candidate_annotations(defect: str) -> None:
    ref = CandidateRef(operation_id="project.status.get", operation_version=1)
    row = CandidateReadiness(operation_id=ref.operation_id, operation_version=1, readiness="ready", project_id=1)
    rows = () if defect == "missing" else (row, row) if defect == "duplicate" else (
        row.model_copy(update={"operation_version": 2} if defect == "other-version" else {"project_id": 2}),)
    with pytest.raises(ValidationError):
        InterpretationInput(text="状態", candidates=(ref,), state=MinimalState(selected_project_id=1),
                            candidate_state=CandidateReadinessSnapshot(observed_at=1.0, candidates=rows))


def test_refresh_handles_deleted_target_and_does_not_require_model(harness: Any) -> None:
    client, adapter, *_ = harness
    project = create(client)
    adapter.replies = [SPEED]
    client.post('/api/language/requests', json={"request_id": "deleted", "text": "話速を1.2倍にして",
        "target": {"project_id": project["id"]}, "base_revision": 1})
    assert client.delete(f'/api/projects/{project["id"]}').status_code == 204
    response = client.get('/api/language/requests/deleted/candidate-readiness')
    assert response.status_code == 200
    assert {c["reason_code"] for c in response.json()["candidates"]} == {"target_not_found"}
    assert len(adapter.calls) == 1
    assert client.get('/api/language/requests/absent/candidate-readiness').status_code == 404


def test_hard_filter_is_experiment_only_and_keeps_unknown_arguments(harness: Any) -> None:
    client, *_ = harness
    project = create(client)
    with get_session_factory()() as db:
        idle = current_candidates(db, operation_service, REFS, project["id"])
    assert hard_filter_candidates(REFS, idle) == REFS
    start(project)
    with get_session_factory()() as db:
        busy = current_candidates(db, operation_service, REFS, project["id"])
    selected = hard_filter_candidates(REFS, busy)
    assert {r.operation_id for r in selected} == {"project.status.get", "project.generation.cancel"}
    with pytest.raises(ValueError):
        hard_filter_candidates(REFS[:1], busy)
    blocked = next(c for c in busy.candidates if c.readiness == "blocked")
    ref = CandidateRef(operation_id=blocked.operation_id, operation_version=blocked.operation_version)
    assert hard_filter_candidates((ref,), busy.model_copy(update={"candidates": (blocked,)})) == ()
    for file in (Path(__file__).parents[1] / 'app').rglob('*.py'):
        tree = ast.parse(file.read_text(encoding='utf-8'))
        imports = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module]
        assert not any(name.startswith('evaluation') for name in imports), file


async def test_hard_filter_experimental_scope_cannot_reappear_in_fallback(harness: Any) -> None:
    client, _, runner, *_ = harness
    project = create(client)
    start(project)
    with get_session_factory()() as db:
        snapshot = current_candidates(db, operation_service, REFS, project["id"])
    refs = hard_filter_candidates(REFS, snapshot)
    adapter = Replies([QUESTION])
    result = await runner.preview(CATALOG, adapter, InterpretationInput(text="話速を1.2倍にして", candidates=refs,
        state=MinimalState(selected_project_id=project["id"])))
    assert result.interpretation.status == "needs_input"
    assert all(c.operation_id in {r.operation_id for r in refs} for stage in result.trace.stages for c in stage.candidates)
    assert result.trace.chat_calls == 1 and counts() == (1, 1, 0)
