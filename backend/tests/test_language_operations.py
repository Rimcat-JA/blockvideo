"""D17 API equivalence, All Tools, confirmation and durable language replay."""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from app.api.routes_language import language_service
from app.db import get_session_factory
from app.interpretation.transport import ModelMessage
from app.language_operations.contracts import LanguageInput
from app.language_operations.service import LanguageOperationService
from app.main import create_app
from app.models.job import GenerationJob, JobStatus
from app.models.language_request import LanguageRequestRecord
from app.models.operation_request import OperationReceipt
from app.operations.bootstrap import operation_service


class ReplyAdapter:
    def __init__(self) -> None:
        self.response = json.dumps({"result": {"kind": "operation",
            "operation_id": "project.subtitle-font-size.set", "operation_version": 1,
            "arguments": {"value": 56}}})
        self.calls: list[dict[str, Any]] = []

    async def complete(self, messages: tuple[ModelMessage, ...], schema: dict[str, Any]) -> str:
        self.calls.append({"prompt": json.loads(messages[1].content), "schema": schema})
        return self.response

    def operation(self, operation_id: str, arguments: dict[str, Any]) -> None:
        self.response = json.dumps({"result": {"kind": "operation", "operation_id": operation_id,
                                             "operation_version": 1, "arguments": arguments}})


@pytest.fixture()
def harness(temp_storage: Path):
    adapter = ReplyAdapter()
    service = LanguageOperationService(operation_service, adapter)
    app = create_app()
    app.dependency_overrides[language_service] = lambda: service
    client = TestClient(app, raise_server_exceptions=True)
    try:
        # Deliberately do not enter lifespan: assertions inspect queued jobs.
        yield client, adapter, service
    finally:
        client.close()


def create(client: TestClient) -> int:
    response = client.post("/api/projects", json={"title": "Synthetic D17", "source_script": "送信しない合成原稿。",
                                               "subtitle_font_size": 48, "use_fake_providers": True})
    assert response.status_code == 201, response.text
    return response.json()["id"]


def language_input(project_id: int | None, *, request_id: str = "nl-test", text: str = "字幕を56pxにして", **extra: Any) -> dict:
    return {"request_id": request_id, "text": text, "target": {"selected_project_id": project_id}, **extra}


def count(model: type) -> int:
    with get_session_factory()() as db:
        return db.scalar(select(func.count()).select_from(model))


def submit(client: TestClient, payload: dict, *, prepare: bool = False) -> dict:
    response = client.post("/api/language/requests" + ("/prepare" if prepare else ""), json=payload)
    assert response.status_code == 200, response.text
    return response.json()


def confirm(client: TestClient, response: dict, *, generation: bool = False):
    return client.post(f"/api/language/requests/{response['request_id']}/execute", json={
        "confirmation_token": response["confirmation_token"], "confirm_generation": generation,
    })


@pytest.mark.parametrize("operation_id,arguments,field,value", [
    ("project.subtitle-font-size.set", {"value": 56}, "subtitle_font_size", 56),
    ("project.subtitle-font-size.adjust", {"delta": 2}, "subtitle_font_size", 50),
    ("project.settings.update", {"voicevox_speed_scale": 1.3, "subtitle_font_size": 52}, "voicevox_speed_scale", 1.3),
])
def test_language_and_typed_settings_have_identical_persisted_effects(harness, operation_id: str, arguments: dict,
                                                                   field: str, value: Any) -> None:
    client, adapter, _ = harness
    first, second = create(client), create(client)
    adapter.operation(operation_id, arguments)
    text = {"project.subtitle-font-size.set": "字幕を56pxにして",
            "project.subtitle-font-size.adjust": "字幕を少し大きくして",
            "project.settings.update": "字幕を52pxにして、速度を1.3倍に"}[operation_id]
    language = submit(client, language_input(first, text=text))
    typed = client.post("/api/operations/execute", json={"operation_id": operation_id, "arguments": arguments,
                        "target": {"project_id": second}, "request_id": "typed", "base_revision": 1}).json()
    assert language["status"] == "completed" and language["executed"] is True
    assert language["result"]["resolved_arguments"] == typed["resolved_arguments"]
    assert language["result"]["revision"] == typed["revision"] == 2
    assert client.get(f"/api/projects/{first}").json()[field] == client.get(f"/api/projects/{second}").json()[field] == value
    assert count(GenerationJob) == 0
    prompt = adapter.calls[0]["prompt"]
    assert len(prompt["candidates"]) == 10
    assert {item["operation_id"] for item in prompt["candidates"]} == {item.operation_id for item in operation_service.list_definitions()}
    assert "source_script" not in json.dumps(prompt)
    assert "title" not in prompt["state"]


def test_status_reads_core_result_without_mutating_project(harness) -> None:
    client, adapter, _ = harness
    project_id = create(client)
    before = client.get(f"/api/projects/{project_id}").json()
    adapter.operation("project.status.get", {})
    result = submit(client, language_input(project_id, text="状態を教えて"))
    assert result["status"] == "completed" and result["result"]["changed"] is False
    assert result["result"]["data"]["subtitle_font_size"] == 48
    assert client.get(f"/api/projects/{project_id}").json() == before
    assert count(GenerationJob) == 0


def test_prepare_is_read_only_and_confirm_cannot_edit_frozen_operation(harness) -> None:
    client, _, _ = harness
    project_id = create(client)
    result = submit(client, language_input(project_id), prepare=True)
    assert result["status"] == "ready" and not result["executed"]
    assert client.get(f"/api/projects/{project_id}").json()["subtitle_font_size"] == 48
    assert count(OperationReceipt) == 0
    endpoint = f"/api/language/requests/{result['request_id']}/execute"
    assert client.post(endpoint, json={"confirmation_token": result["confirmation_token"], "arguments": {"value": 100}}).status_code == 422
    assert client.post(endpoint, json={"confirmation_token": "0" * 64}).status_code == 409
    assert confirm(client, result).json()["result"]["resolved_arguments"] == {"value": 56}


@pytest.mark.parametrize("via_config", [False, True])
def test_review_all_does_not_automatically_save(harness, via_config: bool) -> None:
    client, _, service = harness
    project_id = create(client)
    service.review_all = via_config
    result = submit(client, language_input(project_id, review_all=not via_config))
    assert result["requires_confirmation"] and not result["executed"]
    assert client.get(f"/api/projects/{project_id}").json()["revision"] == 1
    assert confirm(client, result).json()["status"] == "completed"


def test_hallucinated_generation_from_settings_text_requires_explicit_confirmation(harness) -> None:
    client, adapter, _ = harness
    project_id = create(client)
    adapter.operation("project.generation.start", {"kind": "full"})
    result = submit(client, language_input(project_id, text="字幕を56pxにして。生成はしないで。"))
    assert result["requires_confirmation"] and not result["executed"]
    assert count(GenerationJob) == 0
    assert confirm(client, result).json()["detail"]["reason_code"] == "generation_confirmation_required"
    assert count(GenerationJob) == 0


def test_generation_after_confirmation_matches_typed_job_and_replays_once(harness) -> None:
    client, adapter, _ = harness
    first, second = create(client), create(client)
    adapter.operation("project.generation.start", {"kind": "full"})
    result = submit(client, language_input(first, text="動画を作り直して"))
    completed = confirm(client, result, generation=True).json()
    typed = client.post("/api/operations/execute", json={"operation_id": "project.generation.start", "arguments": {"kind": "full"},
        "target": {"project_id": second}, "request_id": "typed-gen", "base_revision": 1}).json()
    assert completed["result"]["generation_requested"] and completed["result"]["job_id"]
    assert count(GenerationJob) == 2
    with get_session_factory()() as db:
        job1 = db.get(GenerationJob, completed["result"]["job_id"])
        job2 = db.get(GenerationJob, typed["job_id"])
        assert job1.kind == job2.kind == "full"
        assert job1.input_revision == job2.input_revision == 1
        assert job1.input_snapshot["project"] == job2.input_snapshot["project"]
    assert confirm(client, result, generation=True).json() == completed
    assert count(GenerationJob) == 2
    assert len(adapter.calls) == 1


def test_same_id_replay_precedes_model_current_state_and_restart(harness) -> None:
    client, adapter, _ = harness
    project_id = create(client)
    adapter.operation("project.subtitle-font-size.adjust", {"delta": 2})
    payload = language_input(project_id, text="字幕を少し大きくして")
    first = submit(client, payload)
    adapter.response = "INVALID DIFFERENT RESPONSE"
    # New orchestration instance simulates process-local state loss.
    client.app.dependency_overrides[language_service] = lambda: LanguageOperationService(operation_service)
    assert submit(client, payload) == first
    assert client.get(f"/api/language/requests/{payload['request_id']}").json() == first
    assert client.get(f"/api/projects/{project_id}").json()["subtitle_font_size"] == 50
    assert count(OperationReceipt) == 1


def test_foreign_core_receipt_cannot_be_reported_as_language_success(harness) -> None:
    client, adapter, _ = harness
    project_id = create(client)
    body = language_input(project_id)
    prepared = submit(client, body, prepare=True)
    # A separate typed client accidentally reuses the public tracing ID.
    response = client.post("/api/operations/execute", json={
        "operation_id": "project.status.get", "arguments": {}, "target": {"project_id": project_id},
        "request_id": prepared["core_request_id"], "base_revision": 1,
    })
    assert response.status_code == 200
    for response in (client.get("/api/language/requests/nl-test"),
                     client.post("/api/language/requests", json=body), confirm(client, prepared)):
        assert response.status_code == 409
        assert response.json()["detail"]["reason_code"] == "core_request_conflict"
    assert len(adapter.calls) == 1
    assert client.get(f"/api/projects/{project_id}").json()["subtitle_font_size"] == 48
    assert count(OperationReceipt) == 1 and count(GenerationJob) == 0


def test_id_content_conflict_does_not_reinterpret(harness) -> None:
    client, adapter, _ = harness
    project_id = create(client)
    submit(client, language_input(project_id))
    response = client.post("/api/language/requests", json=language_input(project_id, text="字幕を80pxに"))
    assert response.status_code == 409 and response.json()["detail"]["reason_code"] == "request_id_conflict"
    assert len(adapter.calls) == 1


@pytest.mark.parametrize("payload,code", [
    ({"target": {}}, "target_required"),
    ({"target": {"project_id": 99999}}, "target_not_found"),
    ({"text": "プロジェクト99999の字幕を56pxに"}, "target_conflict"),
    ({"text": "project #99999 の字幕を56pxに"}, "target_conflict"),
    ({"base_revision": 3}, "stale_state"),
])
def test_target_and_revision_rejections_precede_inference(harness, payload: dict, code: str) -> None:
    client, adapter, _ = harness
    project_id = create(client)
    result = submit(client, {**language_input(project_id), **payload})
    assert not result["executed"] and result["failure"]["reason_code"] == code
    assert not adapter.calls and count(OperationReceipt) == 0


@pytest.mark.parametrize("response,reason", [
    ("not JSON", "invalid_json"),
    ('{"result":{"kind":"operation","operation_id":"os.system","operation_version":1,"arguments":{}}}', "candidate_not_offered"),
    ('{"result":{"kind":"operation","operation_id":"project.subtitle-font-size.set","operation_version":1,"arguments":{"value":56},"generation_requested":true}}', "invalid_output"),
])
def test_unvalidated_proposals_do_not_reach_execution(harness, response: str, reason: str) -> None:
    client, adapter, _ = harness
    project_id = create(client)
    adapter.response = response
    result = submit(client, language_input(project_id))
    assert result["status"] == "error" and result["failure"]["reason_code"] == reason
    assert count(OperationReceipt) == count(GenerationJob) == 0
    assert client.get(f"/api/projects/{project_id}").json()["revision"] == 1


def test_stale_after_prepare_is_rejected_at_actual_execution(harness) -> None:
    client, _, _ = harness
    project_id = create(client)
    result = submit(client, language_input(project_id), prepare=True)
    assert client.patch(f"/api/projects/{project_id}", json={"subtitle_font_size": 70}).status_code == 200
    execution = confirm(client, result).json()
    assert execution["status"] == "blocked" and execution["failure"]["reason_code"] == "stale_state"
    assert count(OperationReceipt) == 0
    assert client.get(f"/api/projects/{project_id}").json()["subtitle_font_size"] == 70


@pytest.mark.parametrize("kind,fields,status", [
    ("clarification", {"question": "何pxにしますか？", "missing_fields": ["arguments"]}, "needs_input"),
    ("unsupported", {"reason": "未対応です。"}, "unsupported"),
])
def test_questions_and_unsupported_are_not_execution_results(harness, kind: str, fields: dict, status: str) -> None:
    client, adapter, _ = harness
    project_id = create(client)
    adapter.response = json.dumps({"result": {"kind": kind, **fields}})
    response = submit(client, language_input(project_id))
    assert response["status"] == status and response["result"] is None and not response["executed"]
    assert count(OperationReceipt) == count(GenerationJob) == 0


async def test_state_change_during_model_call_blocks_save(temp_storage: Path) -> None:
    client = TestClient(create_app())
    project_id = create(client)

    class ChangeAdapter(ReplyAdapter):
        async def complete(self, messages: tuple[ModelMessage, ...], schema: dict[str, Any]) -> str:
            client.patch(f"/api/projects/{project_id}", json={"subtitle_font_size": 70})
            return self.response

    with get_session_factory()() as db:
        result = await LanguageOperationService(operation_service, ChangeAdapter()).submit(db, LanguageInput(
            request_id="change-during", text="字幕56px", target={"project_id": project_id},
        ))
    assert result.status == "blocked" and result.failure.reason_code == "stale_state"
    assert client.get(f"/api/projects/{project_id}").json()["subtitle_font_size"] == 70


async def test_concurrent_same_id_has_one_interpretation(temp_storage: Path) -> None:
    client = TestClient(create_app())
    project_id = create(client)
    entered, release = asyncio.Event(), asyncio.Event()

    class WaitingAdapter(ReplyAdapter):
        async def complete(self, messages: tuple[ModelMessage, ...], schema: dict[str, Any]) -> str:
            self.calls.append({})
            entered.set()
            await release.wait()
            return self.response

    adapter = WaitingAdapter()
    service = LanguageOperationService(operation_service, adapter)
    request = LanguageInput(request_id="same", text="字幕56px", target={"project_id": project_id})
    with get_session_factory()() as db1, get_session_factory()() as db2:
        first = asyncio.create_task(service.submit(db1, request))
        await entered.wait()
        second = await service.submit(db2, request)
        assert second.status == "interpreting"
        release.set()
        done = await first
    assert done.status == "completed" and len(adapter.calls) == 1
    assert count(OperationReceipt) == 1


def test_receipt_recovers_crash_before_language_acknowledgement(harness) -> None:
    client, adapter, service = harness
    project_id = create(client)
    result = submit(client, language_input(project_id), prepare=True)
    with get_session_factory()() as db:
        from app.operations.contracts import OperationRequest
        committed = operation_service.execute(db, OperationRequest.model_validate(result["prepared_request"]))
    # Durable language record is still ready, as if process died immediately after core commit.
    with get_session_factory()() as db:
        assert db.get(LanguageRequestRecord, result["request_id"]).status == "ready"
    recovered = client.get(f"/api/language/requests/{result['request_id']}").json()
    assert recovered["status"] == "completed" and recovered["result"] == committed.model_dump(mode="json")
    assert submit(client, language_input(project_id)) == recovered
    assert len(adapter.calls) == 1 and count(OperationReceipt) == 1


async def test_expired_claim_is_not_replayed_and_late_result_cannot_execute(temp_storage: Path) -> None:
    client = TestClient(create_app())
    project_id = create(client)

    class ExpiringAdapter(ReplyAdapter):
        async def complete(self, messages: tuple[ModelMessage, ...], schema: dict[str, Any]) -> str:
            with get_session_factory()() as other:
                row = other.get(LanguageRequestRecord, "expired")
                row.lease_until = 0
                other.commit()
            return self.response

    with get_session_factory()() as db:
        service = LanguageOperationService(operation_service, ExpiringAdapter())
        request = LanguageInput(request_id="expired", text="字幕56px", target={"project_id": project_id})
        result = await service.submit(db, request)
        assert result.status == "error" and result.failure.reason_code == "interpretation_interrupted"
        assert (await service.submit(db, request)) == result
    assert count(OperationReceipt) == 0
    assert client.get(f"/api/projects/{project_id}").json()["subtitle_font_size"] == 48


def test_raw_utterance_is_not_stored_in_ledger(harness) -> None:
    client, _, _ = harness
    project_id = create(client)
    submit(client, language_input(project_id, text="合成の非公開入力_MARKER 字幕56px"))
    with get_session_factory()() as db:
        row = db.get(LanguageRequestRecord, "nl-test")
        assert len(row.input_fingerprint) == 64
        assert "非公開入力_MARKER" not in json.dumps(row.response_json, ensure_ascii=False)


def test_invalid_unicode_is_rejected_without_model_or_state_change(harness) -> None:
    client, adapter, _ = harness
    project_id = create(client)
    # Escape invalid Unicode into the HTTP JSON, instead of encoding it as UTF-8.
    body = json.dumps(language_input(project_id, text="bad \ud800"))
    response = client.post("/api/language/requests", content=body, headers={"Content-Type": "application/json"})
    assert response.status_code == 422
    result = response.json()
    assert result["detail"]["reason_code"] == "invalid_request" and "bad" not in response.text
    assert not adapter.calls and count(OperationReceipt) == 0 and count(GenerationJob) == 0
    assert client.get(f"/api/projects/{project_id}").json()["revision"] == 1
    assert client.post("/api/language/requests", content=body,
                       headers={"Content-Type": "application/json"}).json() == result


def typed(client: TestClient, project_id: int, operation_id: str, arguments: dict,
          *, request_id: str, revision: int = 1) -> dict:
    response = client.post("/api/operations/execute", json={"operation_id": operation_id,
        "arguments": arguments, "target": {"project_id": project_id},
        "request_id": request_id, "base_revision": revision})
    assert response.status_code == 200, response.text
    return response.json()


def test_restore_matches_typed_and_never_generates(harness) -> None:
    client, adapter, _ = harness
    first, second = create(client), create(client)
    for project_id in (first, second):
        assert client.patch(f"/api/projects/{project_id}", json={"subtitle_font_size": 70}).status_code == 200
    adapter.operation("project.settings.restore", {"revision": 1})
    result = submit(client, language_input(first, text="設定をrevision 1に戻して"))["result"]
    expected = typed(client, second, "project.settings.restore", {"revision": 1}, request_id="restore-typed", revision=2)
    assert result["revision"] == expected["revision"] == 3
    assert result["data"]["settings"] == expected["data"]["settings"]
    assert result["data"]["settings"]["subtitle_font_size"] == 48
    assert count(GenerationJob) == 0


@pytest.mark.parametrize("retry", [False, True])
def test_cancel_and_retry_match_typed_controls(harness, retry: bool) -> None:
    client, adapter, _ = harness
    first, second = create(client), create(client)
    jobs = [typed(client, project_id, "project.generation.start", {}, request_id=f"start-{project_id}")["job_id"]
            for project_id in (first, second)]
    operation = "project.generation.retry" if retry else "project.generation.cancel"
    if retry:
        with get_session_factory()() as db:
            for job_id in jobs:
                db.get(GenerationJob, job_id).status = JobStatus.failed
            db.commit()
    adapter.operation(operation, {"job_id": jobs[0]})
    result = submit(client, language_input(first, text=f"ジョブ{jobs[0]}を{'再試行' if retry else 'キャンセル'}して"))
    if retry:
        assert result["requires_confirmation"] and count(GenerationJob) == 2
        result = confirm(client, result, generation=True).json()
    expected = typed(client, second, operation, {"job_id": jobs[1]}, request_id="typed-control")
    assert result["status"] == "completed" and result["result"]["revision"] == expected["revision"] == 1
    with get_session_factory()() as db:
        if retry:
            first_job = db.get(GenerationJob, result["result"]["job_id"])
            second_job = db.get(GenerationJob, expected["job_id"])
            assert (first_job.parent_job_id, second_job.parent_job_id) == tuple(jobs)
            assert first_job.input_snapshot["project"] == second_job.input_snapshot["project"]
        else:
            assert all(db.get(GenerationJob, job_id).status == JobStatus.cancelled for job_id in jobs)
            assert all(db.get(GenerationJob, job_id).cancel_requested for job_id in jobs)
    assert count(GenerationJob) == (4 if retry else 2)


@pytest.mark.parametrize("operation,arguments,text", [
    ("project.settings.restore", {"revision": 1}, "以前の設定に戻して"),
    ("project.settings.restore", {"revision": 1}, "revision 2に戻して"),
    ("project.generation.cancel", {"job_id": 1}, "生成をキャンセルして"),
    ("project.generation.cancel", {"job_id": 1}, "ジョブ2をキャンセルして"),
    ("project.generation.retry", {"job_id": 1}, "失敗した生成を再試行して"),
    ("project.generation.cancel", {"job_id": 1}, "ジョブ1とジョブ2をキャンセルして"),
])
def test_guessed_or_conflicting_job_and_history_ids_never_execute(harness, operation: str, arguments: dict, text: str) -> None:
    client, adapter, _ = harness
    project_id = create(client)
    adapter.operation(operation, arguments)
    result = submit(client, language_input(project_id, text=text))
    assert result["status"] == "needs_input" and not result["executed"]
    assert result["clarification"]["missing_fields"] == ["arguments"]
    assert result["interpretation"]["proposal"]["arguments"] == arguments
    assert result["prepared_request"] is None
    assert count(OperationReceipt) == 0 and count(GenerationJob) == 0
    assert client.get(f"/api/projects/{project_id}").json()["revision"] == 1


def test_other_project_job_cannot_be_cancelled(harness) -> None:
    client, adapter, _ = harness
    first, second = create(client), create(client)
    job = typed(client, second, "project.generation.start", {}, request_id="other-job")["job_id"]
    adapter.operation("project.generation.cancel", {"job_id": job})
    result = submit(client, language_input(first, text=f"ジョブ{job}をキャンセル"))
    assert result["status"] == "blocked" and result["failure"]["reason_code"] == "job_not_found"
    with get_session_factory()() as db:
        assert not db.get(GenerationJob, job).cancel_requested


def test_busy_generation_blocks_settings(harness) -> None:
    client, _, _ = harness
    project_id = create(client)
    typed(client, project_id, "project.generation.start", {}, request_id="busy-job")
    result = submit(client, language_input(project_id))
    assert result["status"] == "blocked" and result["failure"]["reason_code"] == "project_busy"
    assert client.get(f"/api/projects/{project_id}").json()["subtitle_font_size"] == 48


def test_unknown_outcome_cannot_start_again_after_confirmation(harness) -> None:
    client, adapter, _ = harness
    project_id = create(client)
    job_id = typed(client, project_id, "project.generation.start", {}, request_id="unknown-job")["job_id"]
    with get_session_factory()() as db:
        db.get(GenerationJob, job_id).status = JobStatus.unknown
        db.commit()
    adapter.operation("project.generation.start", {})
    result = submit(client, language_input(project_id, text="動画を作り直して"))
    confirmed = confirm(client, result, generation=True).json()
    assert confirmed["status"] == "blocked"
    assert confirmed["failure"]["reason_code"] == "external_outcome_unknown"
    assert count(GenerationJob) == 1


def test_replay_still_works_after_project_deleted_and_model_config_invalid(harness, monkeypatch: pytest.MonkeyPatch) -> None:
    client, _, _ = harness
    project_id = create(client)
    payload = language_input(project_id)
    original = submit(client, payload)
    assert client.delete(f"/api/projects/{project_id}").status_code == 204
    client.app.dependency_overrides.clear()
    from app.core.config import get_settings
    settings = get_settings()
    monkeypatch.setattr(settings, "language_model", "nonexistent")
    monkeypatch.setattr(settings, "language_base_url", "https://remote.invalid/v1")
    assert submit(client, payload) == original


def test_failed_provider_does_not_leak_exception_and_same_id_never_retries(harness) -> None:
    client, adapter, _ = harness
    project_id = create(client)
    async def fail(*_args):
        adapter.calls.append({})
        raise RuntimeError("private source and token")
    adapter.complete = fail
    original = submit(client, language_input(project_id))
    assert original["failure"]["reason_code"] == "interpretation_failed"
    assert "private" not in json.dumps(original)
    assert submit(client, language_input(project_id)) == original
    assert len(adapter.calls) == 1


@pytest.mark.parametrize("phase", ["after_prepare", "after_core_commit"])
def test_actual_process_exit_resumes_without_reinterpretation(harness, phase: str) -> None:
    client, adapter, _ = harness
    project_id = create(client)
    worker = '''
import asyncio, json, os
from app.db import init_db, get_session_factory
from app.language_operations.contracts import LanguageInput
from app.language_operations.service import LanguageOperationService
from app.operations.bootstrap import operation_service
class Adapter:
    async def complete(self, *args):
        return json.dumps({"result":{"kind":"operation","operation_id":"project.subtitle-font-size.adjust","operation_version":1,"arguments":{"delta":2}}})
async def main():
    init_db()
    with get_session_factory()() as db:
        service = LanguageOperationService(operation_service, Adapter())
        result = await service.prepare(db, LanguageInput(request_id="process",text="字幕を少し大きくして",target={"project_id":int(os.environ["D17_PROJECT"])}))
        if os.environ["D17_PHASE"] == "after_core_commit":
            operation_service.execute(db, result.prepared_request)
    os._exit(31)
asyncio.run(main())
'''
    env = {**os.environ, "D17_PROJECT": str(project_id), "D17_PHASE": phase}
    process = subprocess.run([sys.executable, "-c", worker], cwd=Path(__file__).resolve().parents[1], env=env,
                             capture_output=True, text=True, timeout=30)
    assert process.returncode == 31, process.stderr
    # A fresh request-scoped service must use the stored proposal/receipt, never this adapter.
    adapter.response = "INVALID"
    result = submit(client, language_input(project_id, request_id="process", text="字幕を少し大きくして"))
    assert result["status"] == "completed" and result["result"]["resolved_arguments"] == {"value": 50}
    assert client.get(f"/api/projects/{project_id}").json()["subtitle_font_size"] == 50
    assert not adapter.calls and count(OperationReceipt) == 1
