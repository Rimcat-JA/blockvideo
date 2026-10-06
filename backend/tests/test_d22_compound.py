"""Atomic settings, explicit follow-up generation, and independent durable receipts."""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest

from app.db import get_session_factory
from app.language_operations.contracts import LanguageExecution
from app.language_operations.service import LanguageOperationService
from app.models.job import GenerationJob
from app.models.operation_request import OperationReceipt
from app.operations.bootstrap import operation_service
from app.operations.contracts import OperationRequest
from tests.test_generation_plan import ready_project
from tests.test_language_operations import confirm, count, create, harness as harness, language_input, submit


def compound(adapter: Any, *, settings: dict | None = None, delta: int | None = 2, generate: bool = True) -> None:
    adapter.response = json.dumps({"result": {
        "kind": "operation", "operation_id": "project.settings.update", "operation_version": 2,
        "arguments": {"settings": settings if settings is not None else {
            "pronunciation_overrides": [{"surface": "API", "reading": "エーピーアイ", "accent": None}],
            "voicevox_speed_scale": 1.2}, "subtitle_font_size_delta": delta},
        "generate_after_save": generate,
    }})


def request(project: int, **extra: Any) -> dict:
    return language_input(project, text="字幕を少し大きくして、速度は1.2倍、APIをエーピーアイと読んで。作り直して", **extra)


def test_atomic_save_then_explicit_single_union_job_and_replay(harness: Any) -> None:
    client, adapter, _ = harness
    with get_session_factory()() as db:
        project = ready_project(db)
        project_id, font = project.id, project.subtitle_font_size
    compound(adapter)
    saved = submit(client, request(project_id))
    assert saved["status"] == "ready" and saved["executed"] and saved["requires_confirmation"]
    assert saved["result"]["revision"] == 2
    assert saved["result"]["resolved_arguments"]["subtitle_font_size"] == font + 2
    assert count(GenerationJob) == 0 and count(OperationReceipt) == 1
    assert confirm(client, saved).status_code == 409
    assert submit(client, request(project_id)) == saved
    assert client.get(f"/api/language/requests/{saved['request_id']}").json() == saved
    assert len(adapter.calls) == 1
    done = confirm(client, saved, generation=True).json()
    assert done["status"] == "completed" and done["generation_result"]["job_id"]
    with get_session_factory()() as db:
        job = db.get(GenerationJob, done["generation_result"]["job_id"])
        assert job.input_revision == 2
        assert job.plan_json["stages"] == ["audio", "render"]
        assert job.input_snapshot["project"]["subtitle_font_size"] == font + 2
    assert confirm(client, saved, generation=True).json() == done
    assert count(GenerationJob) == 1 and count(OperationReceipt) == 2


@pytest.mark.parametrize("settings,delta", [
    ({"subtitle_font_size": 64, "voicevox_speed_scale": 3}, None),
    ({"subtitle_font_size": 64}, 2),
    ({"voicevox_speed_scale": 1.2}, 104),
    ({"subtitle_font_size": 64, "pronunciation_overrides": [{"surface": "API", "reading": "エーピーアイ", "accent": 99}]}, None),
    ({}, None),
])
def test_one_bad_setting_rolls_back_everything(harness: Any, settings: dict, delta: int | None) -> None:
    client, adapter, _ = harness
    project = create(client)
    before = client.get(f"/api/projects/{project}").json()
    compound(adapter, settings=settings, delta=delta)
    result = submit(client, request(project))
    assert result["status"] in {"error", "blocked", "needs_input"}
    assert client.get(f"/api/projects/{project}").json() == before
    assert count(OperationReceipt) == count(GenerationJob) == 0


def test_review_all_has_two_distinct_confirmations(harness: Any) -> None:
    client, adapter, service = harness
    service.review_all = True
    project = create(client)
    compound(adapter)
    prepared = submit(client, request(project))
    assert not prepared["executed"] and count(OperationReceipt) == 0
    saved = confirm(client, prepared, generation=True).json()
    assert saved["status"] == "ready" and saved["executed"]
    assert saved["confirmation_token"] != prepared["confirmation_token"]
    assert count(GenerationJob) == 0
    # A lost save response or double click cannot confirm the later generation.
    assert confirm(client, prepared, generation=True).json() == saved
    assert count(GenerationJob) == 0
    assert confirm(client, saved, generation=True).json()["status"] == "completed"
    assert count(GenerationJob) == 1


@pytest.mark.parametrize("after_generation", [False, True])
def test_each_receipt_recovers_without_acknowledgement_or_model(harness: Any, after_generation: bool) -> None:
    client, adapter, _ = harness
    project = create(client)
    compound(adapter)
    prepared = submit(client, request(project), prepare=True)
    with get_session_factory()() as db:
        operation_service.execute(db, OperationRequest.model_validate(prepared["prepared_request"]))
    recovered = client.get("/api/language/requests/nl-test").json()
    assert recovered["status"] == "ready" and recovered["result"]["revision"] == 2
    if after_generation:
        with get_session_factory()() as db:
            operation_service.execute(db, OperationRequest.model_validate(recovered["generation_request"]))
    with get_session_factory()() as db:
        restarted = LanguageOperationService(operation_service).get(db, "nl-test")
    assert restarted.status == ("completed" if after_generation else "ready")
    assert count(GenerationJob) == int(after_generation)
    assert len(adapter.calls) == 1


def test_simultaneous_confirmation_creates_one_job(harness: Any) -> None:
    client, adapter, service = harness
    project = create(client)
    compound(adapter)
    saved = submit(client, request(project))
    def execute() -> dict:
        with get_session_factory()() as db:
            return service.execute(db, "nl-test", LanguageExecution(
                confirmation_token=saved["confirmation_token"], confirm_generation=True)).model_dump(mode="json")
    with ThreadPoolExecutor(max_workers=2) as pool:
        first, second = list(pool.map(lambda _: execute(), range(2)))
    assert first == second
    assert count(GenerationJob) == 1 and count(OperationReceipt) == 2


@pytest.mark.parametrize("action", ["change", "correction", "dismiss"])
def test_stale_or_superseded_generation_keeps_saved_settings(harness: Any, action: str) -> None:
    client, adapter, _ = harness
    project = create(client)
    compound(adapter)
    saved = submit(client, request(project))
    if action == "change":
        client.patch(f"/api/projects/{project}", json={"subtitle_font_size": 80})
    else:
        adapter.operation("project.subtitle-font-size.adjust", {"delta": -2})
        result = submit(client, language_input(project, request_id="followup", text="違う、少し小さく", base_revision=2,
            continuation={"parent_request_id": "nl-test", "relation": action}))
        assert result["status"] == ("dismissed" if action == "dismiss" else "completed")
        if action == "correction":
            assert adapter.calls[-1]["prompt"]["dialogue"][-1]["settings_saved"] is True
    failed = confirm(client, saved, generation=True)
    assert failed.status_code == 409 or failed.json()["status"] == "blocked"
    observed = client.get("/api/language/requests/nl-test").json()
    assert observed["result"] == saved["result"] and observed["executed"]
    assert observed["status"] == "blocked"
    assert count(GenerationJob) == 0
    assert client.get(f"/api/projects/{project}").json()["voicevox_speed_scale"] == 1.2


def test_missing_argument_then_short_answer_saves_all_once(harness: Any) -> None:
    client, adapter, _ = harness
    project = create(client)
    adapter.response = json.dumps({"result": {"kind": "clarification", "question": "字幕は何pxにしますか？", "missing_fields": ["arguments"]}})
    asked = submit(client, language_input(project, text="字幕を大きくして、APIをエーピーアイと読んで。作り直して"))
    assert asked["status"] == "needs_input" and count(OperationReceipt) == 0
    compound(adapter, settings={"subtitle_font_size": 64, "pronunciation_overrides": [{"surface": "API", "reading": "エーピーアイ", "accent": None}]}, delta=None)
    answered = submit(client, language_input(project, request_id="answer", text="64px", base_revision=1,
        continuation={"parent_request_id": "nl-test", "relation": "answer"}))
    assert answered["status"] == "ready" and answered["result"]["revision"] == 2
    assert set(answered["result"]["data"]["changed_fields"]) == {"subtitle_font_size", "pronunciation_overrides"}
    assert count(GenerationJob) == 0


def test_negated_generation_saves_only_settings(harness: Any) -> None:
    client, adapter, _ = harness
    project = create(client)
    compound(adapter, settings={"voicevox_speed_scale": 1.2}, generate=False)
    result = submit(client, language_input(project, text="字幕を少し大きく、速度1.2倍。生成はしないで"))
    assert result["status"] == "completed" and result["generation_request"] is None
    assert result["result"]["revision"] == 2 and count(GenerationJob) == 0
