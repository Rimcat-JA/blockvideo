"""D33 canonical durable-state snapshots and request/core crash boundaries."""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import subprocess
import sys
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any

import httpx
import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import get_session_factory
from app.language_operations import repository
from app.interpretation.transport import ModelMessage
from app.language_operations.contracts import LanguageExecution, LanguageInput
from app.language_operations.service import LanguageOperationService
from app.models.artifact import GenerationArtifact
from app.models.external_call import ExternalCall
from app.models.job import GenerationJob, JobStatus
from app.models.language_request import LanguageRequestRecord
from app.models.language_turn import LanguageTurn
from app.models.operation_request import OperationReceipt
from app.models.project import Project, ProjectStatus
from app.models.settings_revision import SettingsRevision
from app.operations.bootstrap import operation_service
from app.operations.contracts import OperationRequest
from app.services import artifact_store
from app.services.external_calls import ExternalOutcomeUnknown, job_call_context, journaled_post
from app.services.generation_snapshots import (
    GenerationCancelled,
    StaleGenerationInput,
    capture_inputs,
    fingerprint_inputs,
)
from app.services.job_records import create_pending_job
from app.services.paths import project_dir, relpath_for_db
from app.services.settings_history import configuration, record_settings
from app.services.transactions import atomic_write
from app.workers import operation_dispatcher
from app.workers.job_runner import JobRegistry
from app.workers.operation_dispatcher import (
    dispatch_pending_operation_jobs,
    mark_interrupted_operation_jobs,
)


def _canonical(value: Any) -> Any:
    if isinstance(value, bytes):
        return {"sha256": hashlib.sha256(value).hexdigest()}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: _canonical(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    return value


def _record(row: Any) -> dict[str, Any]:
    return {
        column.name: _canonical(getattr(row, column.name))
        for column in row.__table__.columns
    }


def recovery_snapshot(project_id: int) -> dict[str, object]:
    """Reopen SQLite and return the canonical D33 durable-state view."""
    with get_session_factory()() as db:
        project = db.get(Project, project_id)
        assert project is not None
        job_ids = select(GenerationJob.id).where(
            GenerationJob.project_id == project_id
        )
        return {
            "project": {
                "id": project.id,
                "revision": project.revision,
                "settings": _canonical(configuration(project)),
                "status": project.status.value,
                "progress": project.progress,
                "current_stage": project.current_stage,
                "current_artifact_id": project.current_artifact_id,
                "output_video_path": project.output_video_path,
                "output_subtitle_path": project.output_subtitle_path,
                "error_message": project.error_message,
            },
            "settings_revisions": [
                _record(row)
                for row in db.scalars(
                    select(SettingsRevision)
                    .where(SettingsRevision.project_id == project_id)
                    .order_by(SettingsRevision.revision, SettingsRevision.id)
                )
            ],
            "language_requests": [
                _record(row)
                for row in db.scalars(
                    select(LanguageRequestRecord)
                    .where(LanguageRequestRecord.project_id == project_id)
                    .order_by(LanguageRequestRecord.request_id)
                )
            ],
            "language_turns": [
                _record(row)
                for row in db.scalars(
                    select(LanguageTurn)
                    .join(
                        LanguageRequestRecord,
                        LanguageRequestRecord.request_id == LanguageTurn.request_id,
                    )
                    .where(LanguageRequestRecord.project_id == project_id)
                    .order_by(LanguageTurn.request_id)
                )
            ],
            "jobs": [
                _record(row)
                for row in db.scalars(
                    select(GenerationJob)
                    .where(GenerationJob.project_id == project_id)
                    .order_by(GenerationJob.id)
                )
            ],
            "receipts": [
                _record(row)
                for row in db.scalars(
                    select(OperationReceipt)
                    .where(OperationReceipt.project_id == project_id)
                    .order_by(OperationReceipt.request_id)
                )
            ],
            "external_calls": [
                _record(row)
                for row in db.scalars(
                    select(ExternalCall)
                    .where(ExternalCall.job_id.in_(job_ids))
                    .order_by(ExternalCall.id)
                )
            ],
            "artifacts": [
                _record(row)
                for row in db.scalars(
                    select(GenerationArtifact)
                    .where(GenerationArtifact.project_id == project_id)
                    .order_by(GenerationArtifact.id)
                )
            ],
        }


def restart_twice() -> tuple[int, int]:
    """Run startup reconciliation twice and expose both transition counts."""
    return mark_interrupted_operation_jobs(), mark_interrupted_operation_jobs()


def _assert_publication_recovery_is_stable(
    project_id: int,
    job_id: int,
    prior_artifact_id: int,
    expected_status: str,
) -> None:
    immediate = recovery_snapshot(project_id)
    assert any(
        row["id"] == prior_artifact_id for row in immediate["artifacts"]
    )
    artifact_rows = immediate["artifacts"]
    external_calls = immediate["external_calls"]
    pointers = {
        key: immediate["project"][key]
        for key in (
            "current_artifact_id",
            "output_video_path",
            "output_subtitle_path",
        )
    }

    assert mark_interrupted_operation_jobs() == 1
    after_first = recovery_snapshot(project_id)
    job = next(row for row in after_first["jobs"] if row["id"] == job_id)
    assert job["status"] == expected_status
    assert after_first["artifacts"] == artifact_rows
    assert after_first["external_calls"] == external_calls
    assert {
        key: after_first["project"][key] for key in pointers
    } == pointers

    assert mark_interrupted_operation_jobs() == 0
    assert recovery_snapshot(project_id) == after_first


class _ReplyAdapter:
    def __init__(self, operation_id: str, arguments: dict[str, Any]) -> None:
        self.operation_id = operation_id
        self.arguments = arguments
        self.calls = 0

    async def complete(
        self, messages: tuple[ModelMessage, ...], schema: dict[str, Any]
    ) -> str:
        del messages, schema
        self.calls += 1
        return json.dumps(
            {
                "result": {
                    "kind": "operation",
                    "operation_id": self.operation_id,
                    "operation_version": 1,
                    "arguments": self.arguments,
                }
            }
        )


def _seed_canonical_state() -> tuple[int, int]:
    with get_session_factory()() as db:
        project = Project(
            title="D33 synthetic",
            source_script="合成された回復テストです。",
            subtitle_font_size=48,
            use_fake_providers=True,
        )
        db.add(project)
        db.flush()
        record_settings(db, project)
        job = GenerationJob(
            project_id=project.id,
            status=JobStatus.completed,
            current_stage="completed",
            progress=1.0,
            stage_progress=1.0,
            input_revision=project.revision,
            input_snapshot={"project": {"id": project.id, "revision": project.revision}},
            input_fingerprint="a" * 64,
        )
        db.add(job)
        db.flush()
        db.add(
            ExternalCall(
                job_id=job.id,
                fingerprint="c" * 64,
                provider="synthetic",
                endpoint="https://synthetic.invalid/d33",
                remote_side_effect=True,
                status="succeeded",
                response_status=200,
                response_body=b'{"ok":true}',
                response_content_type="application/json",
            )
        )
        artifact = GenerationArtifact(
            project_id=project.id,
            job_id=job.id,
            revision=project.revision,
            input_fingerprint=job.input_fingerprint,
            video_path=f"projects/{project.id}/history/prior.mp4",
            manifest_json={"video": {"sha256": "b" * 64}},
        )
        db.add(artifact)
        db.flush()
        project.current_artifact_id = artifact.id
        project.status = ProjectStatus.completed
        project.output_video_path = artifact.video_path
        db.commit()
        project_id = project.id
        artifact_id = artifact.id

    request = OperationRequest(
        request_id="d33-prior-receipt",
        operation_id="project.status.get",
        target={"project_id": project_id},
        arguments={},
    )
    with get_session_factory()() as db:
        operation_service.execute(db, request)

    with get_session_factory()() as db:
        response, owner, state = repository.claim(
            db,
            LanguageInput(
                request_id="d33-prior-language",
                text="状態を確認して",
                target={"project_id": project_id},
            ),
        )
        assert response.status == "interpreting"
        assert owner is not None
        assert state is not None

    return project_id, artifact_id


def _contains_type(value: Any, expected_type: type) -> bool:
    if isinstance(value, expected_type):
        return True
    if isinstance(value, dict):
        return any(_contains_type(item, expected_type) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_type(item, expected_type) for item in value)
    return False


def _language_request(project_id: int, request_id: str, text: str) -> LanguageInput:
    return LanguageInput(
        request_id=request_id,
        text=text,
        target={"project_id": project_id},
    )


def _prepare_generation(
    project_id: int, request_id: str
) -> tuple[LanguageOperationService, LanguageInput, Any]:
    request = _language_request(project_id, request_id, "動画を生成して")
    service = LanguageOperationService(
        operation_service,
        _ReplyAdapter("project.generation.start", {"kind": "full"}),
    )
    with get_session_factory()() as db:
        response = asyncio.run(service.prepare(db, request))
    assert response.status == "ready"
    assert response.requires_confirmation
    assert response.confirmation_token is not None
    return service, request, response


def _effect_state(snapshot: dict[str, object]) -> dict[str, object]:
    return {
        key: snapshot[key]
        for key in (
            "project",
            "settings_revisions",
            "jobs",
            "receipts",
            "external_calls",
            "artifacts",
        )
    }


def _receipt(snapshot: dict[str, object], request_id: str) -> dict[str, Any]:
    matching = [
        row for row in snapshot["receipts"] if row["request_id"] == request_id
    ]
    assert len(matching) == 1
    return matching[0]


def _assert_exact_receipt_result(
    snapshot: dict[str, object], request_id: str, result: Any
) -> dict[str, Any]:
    receipt = _receipt(snapshot, request_id)
    assert receipt["result_json"] == result.model_dump(mode="json")
    return receipt


def _assert_atomic_settings_effect(
    before: dict[str, object],
    after: dict[str, object],
    request_id: str,
    result: Any,
) -> None:
    assert after["project"] == {
        **before["project"],
        "revision": before["project"]["revision"] + 1,
        "settings": {**before["project"]["settings"], "subtitle_font_size": 56},
    }
    assert after["jobs"] == before["jobs"]
    assert after["external_calls"] == before["external_calls"]
    assert after["artifacts"] == before["artifacts"]
    assert after["language_requests"] == before["language_requests"]
    assert after["language_turns"] == before["language_turns"]
    assert len(after["settings_revisions"]) == len(before["settings_revisions"]) + 1
    assert all(row in after["settings_revisions"] for row in before["settings_revisions"])
    new_history = [
        row for row in after["settings_revisions"]
        if row not in before["settings_revisions"]
    ]
    assert len(new_history) == 1
    assert new_history[0]["revision"] == result.revision
    assert new_history[0]["settings_json"] == after["project"]["settings"]
    assert new_history[0]["changed_fields"] == ["subtitle_font_size"]
    assert len(after["receipts"]) == len(before["receipts"]) + 1
    assert all(row in after["receipts"] for row in before["receipts"])
    new_receipts = [
        row for row in after["receipts"] if row not in before["receipts"]
    ]
    assert len(new_receipts) == 1
    receipt = _assert_exact_receipt_result(after, request_id, result)
    assert receipt["result_revision"] == result.revision
    assert receipt["job_id"] is None


def _assert_atomic_generation_effect(
    before: dict[str, object],
    after: dict[str, object],
    request_id: str,
    result: Any,
) -> None:
    assert after["project"]["revision"] == before["project"]["revision"]
    assert after["project"]["settings"] == before["project"]["settings"]
    assert after["project"]["status"] == "generating"
    assert (
        after["project"]["current_artifact_id"]
        == before["project"]["current_artifact_id"]
    )
    assert after["settings_revisions"] == before["settings_revisions"]
    assert [row["request_id"] for row in after["language_requests"]] == [
        row["request_id"] for row in before["language_requests"]
    ]
    assert after["language_turns"] == before["language_turns"]
    assert after["external_calls"] == before["external_calls"]
    assert after["artifacts"] == before["artifacts"]
    assert len(after["receipts"]) == len(before["receipts"]) + 1
    assert len(after["jobs"]) == len(before["jobs"]) + 1
    assert all(row in after["receipts"] for row in before["receipts"])
    assert all(row in after["jobs"] for row in before["jobs"])
    new_receipts = [
        row for row in after["receipts"] if row not in before["receipts"]
    ]
    new_jobs = [row for row in after["jobs"] if row not in before["jobs"]]
    assert len(new_receipts) == len(new_jobs) == 1
    receipt = _assert_exact_receipt_result(after, request_id, result)
    job = new_jobs[0]
    assert receipt["job_id"] == job["id"] == result.job_id
    assert job["project_id"] == receipt["project_id"]
    assert [row["job_id"] for row in new_receipts] == [row["id"] for row in new_jobs]


def test_canonical_snapshot_reopens_complete_stable_state(temp_storage) -> None:
    project_id, artifact_id = _seed_canonical_state()

    first = recovery_snapshot(project_id)
    second = recovery_snapshot(project_id)

    assert first == second
    assert first["project"]["id"] == project_id
    assert first["project"]["revision"] == 1
    assert first["project"]["status"] == "completed"
    assert first["project"]["progress"] == 0.0
    assert first["project"]["current_stage"] is None
    assert first["project"]["current_artifact_id"] == artifact_id
    assert first["project"]["output_video_path"] == f"projects/{project_id}/history/prior.mp4"
    assert first["project"]["output_subtitle_path"] is None
    assert first["project"]["error_message"] is None
    assert first["project"]["settings"] == {
        "visual_focus_enabled": True,
        "subtitle_mode": "sentence",
        "narration_pacing_mode": "adaptive",
        "pronunciation_overrides": [],
        "title": "D33 synthetic",
        "voicevox_url": "http://127.0.0.1:50021",
        "voicevox_speaker_id": 1,
        "voicevox_speed_scale": 1.0,
        "voicevox_pitch_scale": 0.0,
        "voicevox_intonation_scale": 1.0,
        "voicevox_volume_scale": 1.0,
        "subtitle_enabled": True,
        "subtitle_font_size": 48,
        "subtitle_position": "bottom",
        "subtitle_text_color": "#FFFFFF",
        "subtitle_outline_color": "#000000",
        "subtitle_background": True,
        "subtitle_max_chars_per_line": 36,
        "pre_margin_seconds": 0.15,
        "post_margin_seconds": 1.5,
        "min_display_seconds": 2.0,
        "narration_sentence_pause_seconds": 1.5,
        "max_slides_per_block": 1,
    }
    assert [row["revision"] for row in first["settings_revisions"]] == [1]
    assert [row["request_id"] for row in first["language_requests"]] == [
        "d33-prior-language"
    ]
    assert first["language_requests"][0]["status"] == "interpreting"
    assert first["language_turns"] == [
        {
            "request_id": "d33-prior-language",
            "parent_request_id": None,
            "relation": None,
            "text": "状態を確認して",
            "successor_request_id": None,
        }
    ]
    assert [row["request_id"] for row in first["receipts"]] == [
        "d33-prior-receipt"
    ]
    assert [row["id"] for row in first["artifacts"]] == [artifact_id]
    assert first["external_calls"][0]["response_body"] == {
        "sha256": "4062edaf750fb8074e7e83e0c9028c94e32468a8b6f1614774328ef045150f93"
    }
    assert not _contains_type(first, bytes)
    assert not _contains_type(first, Project)
    json.dumps(first, sort_keys=True, allow_nan=False)


@pytest.mark.parametrize("boundary", ["before", "after"])
def test_request_claim_crash_expires_owner_or_executes_once(
    temp_storage, monkeypatch: pytest.MonkeyPatch, boundary: str
) -> None:
    project_id, artifact_id = _seed_canonical_state()
    adapter = _ReplyAdapter("project.subtitle-font-size.set", {"value": 56})
    service = LanguageOperationService(operation_service, adapter)
    request = _language_request(project_id, f"d33-claim-{boundary}", "字幕を56pxにして")
    original_claim = repository.claim
    before = recovery_snapshot(project_id)

    def crashing_claim(db, incoming):
        if boundary == "before":
            raise OSError("synthetic D33 boundary")
        original_claim(db, incoming)
        raise OSError("synthetic D33 boundary")

    monkeypatch.setattr(repository, "claim", crashing_claim)
    with get_session_factory()() as db, pytest.raises(
        OSError, match="synthetic D33 boundary"
    ):
        asyncio.run(service.submit(db, request))
    monkeypatch.setattr(repository, "claim", original_claim)

    crashed = recovery_snapshot(project_id)
    assert crashed["project"]["current_artifact_id"] == artifact_id
    assert restart_twice() == (0, 0)
    assert recovery_snapshot(project_id) == crashed

    if boundary == "after":
        assert _effect_state(crashed) == _effect_state(before)
        claimed = next(
            row for row in crashed["language_requests"]
            if row["request_id"] == request.request_id
        )
        assert claimed["status"] == "interpreting"
        with get_session_factory()() as db:
            record = db.get(LanguageRequestRecord, request.request_id)
            assert record is not None
            record.lease_until = 0.0
            db.commit()
        with get_session_factory()() as db:
            reconciled_response = service.get(db, request.request_id)
        assert reconciled_response.status == "error"
        assert reconciled_response.failure is not None
        assert reconciled_response.failure.reason_code == "interpretation_interrupted"
        assert reconciled_response.result is None
        assert not reconciled_response.executed
        reconciled = recovery_snapshot(project_id)
        stored = next(
            row for row in reconciled["language_requests"]
            if row["request_id"] == request.request_id
        )
        assert stored["response_json"] == reconciled_response.model_dump(mode="json")
        assert restart_twice() == (0, 0)
        assert recovery_snapshot(project_id) == reconciled
        with get_session_factory()() as db:
            second = asyncio.run(service.submit(db, request))
        assert second == reconciled_response
        assert recovery_snapshot(project_id) == reconciled
        assert restart_twice() == (0, 0)
        assert recovery_snapshot(project_id) == reconciled
        assert _effect_state(reconciled) == _effect_state(before)
        assert adapter.calls == 0
        return

    assert crashed == before
    with get_session_factory()() as db:
        first = asyncio.run(service.submit(db, request))
    assert first.status == "completed"
    assert first.result is not None
    assert first.result.resolved_arguments == {"value": 56}
    final = recovery_snapshot(project_id)
    _assert_exact_receipt_result(final, first.core_request_id, first.result)
    assert final["project"]["revision"] == 2
    assert [row["revision"] for row in final["settings_revisions"]] == [1, 2]
    assert restart_twice() == (0, 0)
    assert recovery_snapshot(project_id) == final
    with get_session_factory()() as db:
        replay = asyncio.run(service.submit(db, request))
    assert replay == first
    assert recovery_snapshot(project_id) == final
    assert restart_twice() == (0, 0)
    assert recovery_snapshot(project_id) == final
    assert adapter.calls == 1


@pytest.mark.parametrize("boundary", ["before", "after"])
def test_settings_core_commit_is_atomic_with_history_and_receipt(
    temp_storage, monkeypatch: pytest.MonkeyPatch, boundary: str
) -> None:
    project_id, _ = _seed_canonical_state()
    request = OperationRequest(
        request_id=f"d33-settings-commit-{boundary}",
        operation_id="project.subtitle-font-size.set",
        target={"project_id": project_id},
        arguments={"value": 56},
        base_revision=1,
    )
    before = recovery_snapshot(project_id)

    with get_session_factory()() as db:
        original_commit = db.commit

        def crashing_commit() -> None:
            if boundary == "after":
                original_commit()
            raise OSError("synthetic D33 boundary")

        monkeypatch.setattr(db, "commit", crashing_commit)
        with pytest.raises(OSError, match="synthetic D33 boundary"):
            operation_service.execute(db, request)

    crashed = recovery_snapshot(project_id)
    if boundary == "before":
        assert crashed == before
    else:
        with get_session_factory()() as db:
            persisted = operation_service.get_result(db, request.request_id)
        _assert_atomic_settings_effect(before, crashed, request.request_id, persisted)
    assert restart_twice() == (0, 0)
    assert recovery_snapshot(project_id) == crashed

    with get_session_factory()() as db:
        replay = operation_service.execute(db, request)
    final = recovery_snapshot(project_id)
    _assert_atomic_settings_effect(before, final, request.request_id, replay)
    assert restart_twice() == (0, 0)
    assert recovery_snapshot(project_id) == final
    with get_session_factory()() as db:
        assert operation_service.execute(db, request) == replay
    assert recovery_snapshot(project_id) == final
    assert restart_twice() == (0, 0)
    assert recovery_snapshot(project_id) == final


@pytest.mark.parametrize("boundary", ["before", "after"])
def test_core_receipt_crash_rolls_back_then_generation_confirmation_executes_once(
    temp_storage, monkeypatch: pytest.MonkeyPatch, boundary: str
) -> None:
    from app.operations import service as core_service_module

    project_id, artifact_id = _seed_canonical_state()
    service, request, prepared = _prepare_generation(
        project_id, f"d33-receipt-{boundary}"
    )
    confirmation = LanguageExecution(
        confirmation_token=prepared.confirmation_token,
        confirm_generation=True,
    )
    before = recovery_snapshot(project_id)
    original_save_receipt = core_service_module.save_receipt

    def crashing_save_receipt(*args, **kwargs):
        if boundary == "before":
            raise OSError("synthetic D33 boundary")
        original_save_receipt(*args, **kwargs)
        raise OSError("synthetic D33 boundary")

    monkeypatch.setattr(core_service_module, "save_receipt", crashing_save_receipt)
    with get_session_factory()() as db, pytest.raises(
        OSError, match="synthetic D33 boundary"
    ):
        service.execute(db, request.request_id, confirmation)
    monkeypatch.setattr(core_service_module, "save_receipt", original_save_receipt)

    crashed = recovery_snapshot(project_id)
    assert crashed == before
    assert crashed["project"]["current_artifact_id"] == artifact_id
    assert restart_twice() == (0, 0)
    assert recovery_snapshot(project_id) == crashed

    with get_session_factory()() as db:
        completed = service.execute(db, request.request_id, confirmation)
    assert completed.status == "completed"
    assert completed.result is not None
    final = recovery_snapshot(project_id)
    _assert_atomic_generation_effect(
        before, final, prepared.core_request_id, completed.result
    )
    assert restart_twice() == (0, 0)
    assert recovery_snapshot(project_id) == final
    with get_session_factory()() as db:
        replay = service.execute(db, request.request_id, confirmation)
    assert replay == completed
    assert recovery_snapshot(project_id) == final
    assert restart_twice() == (0, 0)
    assert recovery_snapshot(project_id) == final


_PROCESS_DEATH_PROGRAM = r'''
import asyncio
import os
import sys
from pathlib import Path

import httpx
from sqlalchemy import select

from app.db import get_session_factory
from app.models.block import Block
from app.models.job import GenerationJob, JobStatus
from app.models.project import Project
from app.services import artifact_store, external_calls
from app.services.external_calls import job_call_context, journaled_post
from app.services.generation_snapshots import capture_inputs, fingerprint_inputs
from app.services.job_records import create_pending_job
from app.services.transactions import atomic_write
from app.workers.job_runner import JobRegistry

scenario = sys.argv[1]
project_id = int(sys.argv[2])
marker_path = Path(sys.argv[3])
extra = Path(sys.argv[4]) if len(sys.argv) > 4 else None


def marker(value: str) -> None:
    with marker_path.open("a", encoding="utf-8") as stream:
        stream.write(value + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def newest_job() -> GenerationJob:
    with get_session_factory()() as db:
        return db.scalar(
            select(GenerationJob)
            .where(GenerationJob.project_id == project_id)
            .order_by(GenerationJob.id.desc())
        )


def create_running_job() -> int:
    with get_session_factory()() as db, atomic_write(db):
        job = create_pending_job(db, project_id)
        job.status = JobStatus.running
        job_id = job.id
    return job_id


async def provider_death() -> None:
    job_id = newest_job().id
    original_claim = external_calls._claim
    original_finish = external_calls._finish

    def claim(*args: object, **kwargs: object):
        result = original_claim(*args, **kwargs)
        marker("claimed")
        if scenario == "provider_claimed":
            os._exit(71)
        return result

    def send(request: httpx.Request) -> httpx.Response:
        marker("sent")
        if scenario == "provider_sent":
            os._exit(72)
        return httpx.Response(200, json={"synthetic": "saved"})

    def finish(*args: object, **kwargs: object) -> None:
        original_finish(*args, **kwargs)
        marker("response_saved")
        os._exit(73)

    external_calls._claim = claim
    if scenario == "provider_finished":
        external_calls._finish = finish
    async with httpx.AsyncClient(transport=httpx.MockTransport(send)) as client:
        with job_call_context(job_id):
            await journaled_post(
                client,
                "https://synthetic.invalid/d33-process-death",
                provider="synthetic",
                json={"fixed": True},
            )


def checkpoint_death() -> None:
    job_id = create_running_job()
    if scenario == "checkpoint_resume":
        with get_session_factory()() as db, atomic_write(db):
            project = db.get(Project, project_id)
            db.add(Block(
                project_id=project_id,
                index=0,
                source_text="合成チェックポイント。",
                tts_text="合成チェックポイント。",
            ))
        with get_session_factory()() as db, atomic_write(db):
            project = db.get(Project, project_id)
            job = db.get(GenerationJob, job_id)
            checkpoint = capture_inputs(project)
            job.plan_json = {
                **(job.plan_json or {}),
                "resume_inputs": checkpoint,
                "resume_fingerprint": fingerprint_inputs(checkpoint),
            }
    elif scenario == "checkpoint_missing":
        with get_session_factory()() as db, atomic_write(db):
            job = db.get(GenerationJob, job_id)
            job.plan_json = {
                **(job.plan_json or {}),
                "resume_inputs": job.input_snapshot,
            }
    elif scenario == "checkpoint_altered":
        with get_session_factory()() as db, atomic_write(db):
            job = db.get(GenerationJob, job_id)
            job.plan_json = {
                **(job.plan_json or {}),
                "resume_inputs": {**job.input_snapshot, "project_id": 9999},
                "resume_fingerprint": fingerprint_inputs(job.input_snapshot),
            }
    elif scenario == "checkpoint_revision":
        with get_session_factory()() as db, atomic_write(db):
            db.get(Project, project_id).revision += 1
    elif scenario == "checkpoint_cancelled":
        assert JobRegistry().request_cancel(job_id)
    marker("checkpoint_saved")
    os._exit({
        "checkpoint_initial": 81,
        "checkpoint_resume": 82,
        "checkpoint_missing": 83,
        "checkpoint_altered": 84,
        "checkpoint_revision": 85,
        "checkpoint_cancelled": 86,
    }[scenario])


async def shutdown_death() -> None:
    from app.main import app
    from app.workers import operation_dispatcher
    from app.workers.job_runner import job_registry

    observed_running = asyncio.Event()

    async def local_generation(job_id: int, cancel_check) -> None:
        del cancel_check
        with get_session_factory()() as db, atomic_write(db):
            job = db.get(GenerationJob, job_id)
            assert job is not None
            assert job.status == JobStatus.running
            job.current_stage = "shutdown-observed"
            project = db.get(Project, job.project_id)
            project.current_stage = "shutdown-observed"
        marker("durable_running")
        observed_running.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            marker("task_cancelled")
            raise

    operation_dispatcher.run_generation_job = local_generation
    shutdown_registry = job_registry.shutdown

    async def observed_shutdown() -> None:
        marker("registry_shutdown_started")
        await shutdown_registry()
        marker("registry_shutdown_finished")

    job_registry.shutdown = observed_shutdown
    async with app.router.lifespan_context(app):
        await asyncio.wait_for(observed_running.wait(), timeout=10)
    marker("lifespan_exit")


async def artifact_death() -> None:
    job = newest_job()
    candidate = extra

    async def validate(path: Path) -> dict[str, object]:
        return {
            **artifact_store.file_identity(path),
            "duration_ms": 1000,
            "width": 320,
            "height": 180,
        }

    artifact_store.validate_video = validate
    original_replace = Path.replace

    def replace(path: Path, target: Path) -> Path:
        if scenario == "artifact_before_rename":
            os._exit(91)
        result = original_replace(path, target)
        marker("artifact_renamed")
        if scenario == "artifact_after_rename":
            os._exit(92)
        return result

    Path.replace = replace
    marker("checkpoint_saved")
    await artifact_store.publish_artifact(
        job.id,
        candidate,
        None,
        settled_inputs=job.input_snapshot,
        materials=[],
        cancel_check=lambda: False,
    )
    os._exit(93)


if scenario.startswith("provider_"):
    asyncio.run(provider_death())
elif scenario.startswith("checkpoint_"):
    checkpoint_death()
elif scenario == "shutdown":
    asyncio.run(shutdown_death())
elif scenario.startswith("artifact_"):
    asyncio.run(artifact_death())
else:
    raise AssertionError(scenario)
'''


def _run_process_death(
    scenario: str,
    project_id: int,
    marker_path: Path,
    extra: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        "-c",
        _PROCESS_DEATH_PROGRAM,
        scenario,
        str(project_id),
        str(marker_path),
    ]
    if extra is not None:
        command.append(str(extra))
    return subprocess.run(
        command,
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
        timeout=30,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        check=False,
    )


def _new_running_job(project_id: int) -> int:
    with get_session_factory()() as db, atomic_write(db):
        job = create_pending_job(db, project_id)
        job.status = JobStatus.running
        return job.id


def _latest_job_id(project_id: int) -> int:
    with get_session_factory()() as db:
        job_id = db.scalar(
            select(GenerationJob.id)
            .where(GenerationJob.project_id == project_id)
            .order_by(GenerationJob.id.desc())
        )
    assert job_id is not None
    return job_id


def _marker_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines() if path.exists() else []


def _materialize_prior_artifact(project_id: int, artifact_id: int) -> Path:
    prior = project_dir(project_id) / "history" / "prior.mp4"
    prior.parent.mkdir(parents=True, exist_ok=True)
    prior.write_bytes(b"prior-success")
    identity = artifact_store.file_identity(prior)
    with get_session_factory()() as db:
        artifact = db.get(GenerationArtifact, artifact_id)
        artifact.video_path = identity["path"]
        artifact.manifest_json = {"schema_version": 1, "video": identity}
        project = db.get(Project, project_id)
        project.output_video_path = identity["path"]
        db.commit()
    return prior


@pytest.mark.parametrize(
    ("scenario", "exit_code", "expected_markers", "expected_status"),
    [
        ("provider_claimed", 71, ["claimed"], "unknown"),
        ("provider_sent", 72, ["claimed", "sent"], "unknown"),
        (
            "provider_finished",
            73,
            ["claimed", "sent", "response_saved"],
            "succeeded",
        ),
    ],
)
def test_provider_process_death_never_repeats_remote_send(
    temp_storage: Path,
    scenario: str,
    exit_code: int,
    expected_markers: list[str],
    expected_status: str,
) -> None:
    project_id, prior_artifact_id = _seed_canonical_state()
    job_id = _new_running_job(project_id)
    marker_path = temp_storage / f"{scenario}.markers"

    result = _run_process_death(scenario, project_id, marker_path)

    assert result.returncode == exit_code, result.stderr
    assert _marker_lines(marker_path) == expected_markers
    crashed = recovery_snapshot(project_id)
    calls = [row for row in crashed["external_calls"] if row["job_id"] == job_id]
    assert len(calls) == 1
    assert calls[0]["status"] == (
        "in_flight" if expected_status == "unknown" else "succeeded"
    )
    first, second = restart_twice()
    assert (first, second) == (1, 0)
    recovered = recovery_snapshot(project_id)
    calls = [row for row in recovered["external_calls"] if row["job_id"] == job_id]
    assert len(calls) == 1
    assert calls[0]["status"] == expected_status
    assert recovered["project"]["current_artifact_id"] == prior_artifact_id
    stable = recovery_snapshot(project_id)
    assert stable == recovered

    sends = 0

    def send(request: httpx.Request) -> httpx.Response:
        nonlocal sends
        sends += 1
        return httpx.Response(200, json={"unexpected": True})

    async def replay() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(send)) as client:
            with job_call_context(job_id):
                if expected_status == "unknown":
                    with pytest.raises(ExternalOutcomeUnknown):
                        await journaled_post(
                            client,
                            "https://synthetic.invalid/d33-process-death",
                            provider="synthetic",
                            json={"fixed": True},
                        )
                else:
                    response = await journaled_post(
                        client,
                        "https://synthetic.invalid/d33-process-death",
                        provider="synthetic",
                        json={"fixed": True},
                    )
                    assert response.json() == {"synthetic": "saved"}

    asyncio.run(replay())
    assert sends == 0
    assert _marker_lines(marker_path).count("sent") == int(scenario != "provider_claimed")
    assert restart_twice() == (0, 0)
    assert recovery_snapshot(project_id) == stable


@pytest.mark.parametrize(
    ("scenario", "exit_code", "expected_status"),
    [
        ("checkpoint_initial", 81, "pending"),
        ("checkpoint_resume", 82, "pending"),
        ("checkpoint_missing", 83, "failed"),
        ("checkpoint_altered", 84, "failed"),
        ("checkpoint_revision", 85, "failed"),
        ("checkpoint_cancelled", 86, "cancelled"),
    ],
)
def test_checkpoint_and_cancellation_process_death_is_stable(
    temp_storage: Path,
    scenario: str,
    exit_code: int,
    expected_status: str,
) -> None:
    project_id, prior_artifact_id = _seed_canonical_state()
    marker_path = temp_storage / f"{scenario}.markers"

    result = _run_process_death(scenario, project_id, marker_path)

    assert result.returncode == exit_code, result.stderr
    assert _marker_lines(marker_path) == ["checkpoint_saved"]
    job_id = _latest_job_id(project_id)
    crashed = recovery_snapshot(project_id)
    job = next(row for row in crashed["jobs"] if row["id"] == job_id)
    assert job["status"] == "running"
    assert job["input_fingerprint"] == fingerprint_inputs(job["input_snapshot"])
    if scenario == "checkpoint_resume":
        assert job["plan_json"]["resume_fingerprint"] == fingerprint_inputs(
            job["plan_json"]["resume_inputs"]
        )
    if scenario == "checkpoint_cancelled":
        assert job["cancel_requested"] is True
    assert restart_twice() == (1, 0)
    recovered = recovery_snapshot(project_id)
    job = next(row for row in recovered["jobs"] if row["id"] == job_id)
    assert job["status"] == expected_status
    assert recovered["project"]["current_artifact_id"] == prior_artifact_id
    if expected_status in {"failed", "cancelled"}:
        class RejectingRegistry:
            def is_running(self, target: int) -> bool:
                pytest.fail(f"terminal job {target} reached dispatch")

            def submit(self, target: int, factory: Any) -> None:
                del factory
                pytest.fail(f"terminal job {target} reached provider construction")

        assert dispatch_pending_operation_jobs(registry=RejectingRegistry()) == 0
    assert recovery_snapshot(project_id) == recovered
    assert restart_twice() == (0, 0)
    assert recovery_snapshot(project_id) == recovered


@pytest.mark.asyncio
async def test_immediate_shutdown_drains_unstarted_task_and_reopens_registry(
    temp_storage: Path,
) -> None:
    del temp_storage
    project_id, _prior_artifact_id = _seed_canonical_state()
    with get_session_factory()() as db, atomic_write(db):
        job_id = create_pending_job(db, project_id).id
    registry = JobRegistry()
    callback_started = False

    async def work(_cancel_check: Any) -> None:
        nonlocal callback_started
        callback_started = True

    task = registry.submit(job_id, work)
    await registry.shutdown()

    assert task.cancelled()
    assert not callback_started
    assert registry._tasks == {}
    assert registry._cancel_flags == {}
    assert not registry.is_running(job_id)
    registry.start()
    assert registry.accepting


def test_shutdown_cancellation_reconciles_running_job_without_remote_duplicate(
    temp_storage: Path,
) -> None:
    project_id, prior_artifact_id = _seed_canonical_state()
    job_id = _new_running_job(project_id)
    marker_path = temp_storage / "shutdown.markers"

    result = _run_process_death("shutdown", project_id, marker_path)

    assert result.returncode == 0, result.stderr
    assert _marker_lines(marker_path) == [
        "durable_running",
        "registry_shutdown_started",
        "task_cancelled",
        "registry_shutdown_finished",
        "lifespan_exit",
    ]
    crashed = recovery_snapshot(project_id)
    job = next(row for row in crashed["jobs"] if row["id"] == job_id)
    assert job["status"] == "running"
    assert job["current_stage"] == "shutdown-observed"
    assert crashed["project"]["current_stage"] == "shutdown-observed"
    assert [
        row for row in crashed["external_calls"] if row["job_id"] == job_id
    ] == []

    assert restart_twice() == (1, 0)
    recovered = recovery_snapshot(project_id)
    job = next(row for row in recovered["jobs"] if row["id"] == job_id)
    assert job["status"] == "pending"
    assert job["recovery_message"] == "再起動後、安全な保存地点から処理を再開します。"
    assert recovered["project"]["current_artifact_id"] == prior_artifact_id
    assert [
        row for row in recovered["external_calls"] if row["job_id"] == job_id
    ] == []
    assert recovery_snapshot(project_id) == recovered
    assert restart_twice() == (0, 0)
    assert recovery_snapshot(project_id) == recovered


@pytest.mark.parametrize(
    ("scenario", "exit_code"),
    [
        ("artifact_before_rename", 91),
        ("artifact_after_rename", 92),
        ("artifact_published", 93),
    ],
)
def test_artifact_publication_process_death_preserves_history_and_identity(
    temp_storage: Path,
    monkeypatch: pytest.MonkeyPatch,
    scenario: str,
    exit_code: int,
) -> None:
    project_id, prior_artifact_id = _seed_canonical_state()
    prior_path = _materialize_prior_artifact(project_id, prior_artifact_id)
    job_id = _new_running_job(project_id)
    candidate = project_dir(project_id) / "history" / f"job-{job_id:08d}" / "video.pending.mp4"
    candidate.parent.mkdir(parents=True, exist_ok=True)
    candidate.write_bytes(b"new-verified-video")
    marker_path = temp_storage / f"{scenario}.markers"

    result = _run_process_death(scenario, project_id, marker_path, candidate)

    assert result.returncode == exit_code, result.stderr
    markers = _marker_lines(marker_path)
    assert markers[0] == "checkpoint_saved"
    assert markers.count("artifact_renamed") == int(scenario != "artifact_before_rename")
    crashed = recovery_snapshot(project_id)
    final = candidate.with_name("video.mp4")
    assert candidate.exists() == (scenario == "artifact_before_rename")
    assert final.exists() == (scenario != "artifact_before_rename")
    new_artifacts = [row for row in crashed["artifacts"] if row["job_id"] == job_id]
    if scenario == "artifact_published":
        assert len(new_artifacts) == 1
        assert crashed["project"]["current_artifact_id"] == new_artifacts[0]["id"]
        assert restart_twice() == (0, 0)
    else:
        assert new_artifacts == []
        assert crashed["project"]["current_artifact_id"] == prior_artifact_id
        assert restart_twice() == (1, 0)
    recovered = recovery_snapshot(project_id)
    assert any(row["id"] == prior_artifact_id for row in recovered["artifacts"])
    assert prior_path.read_bytes() == b"prior-success"
    assert restart_twice() == (0, 0)
    assert recovery_snapshot(project_id) == recovered

    if scenario == "artifact_published":
        async def reject_validation(path: Path) -> dict[str, object]:
            pytest.fail(f"committed replay revalidated missing candidate {path}")

        monkeypatch.setattr(artifact_store, "validate_video", reject_validation)
        artifact = asyncio.run(
            artifact_store.publish_artifact(
                job_id,
                candidate,
                None,
                settled_inputs=next(
                    row for row in recovered["jobs"] if row["id"] == job_id
                )["input_snapshot"],
                materials=[],
                cancel_check=lambda: pytest.fail(
                    "committed replay reconsidered cancellation"
                ),
            )
        )
        assert artifact.id == new_artifacts[0]["id"]
        replayed = recovery_snapshot(project_id)
        assert replayed == recovered
        assert len(
            [row for row in replayed["artifacts"] if row["job_id"] == job_id]
        ) == 1


def test_resumed_artifact_after_rename_completes_once_and_replays_stably(
    temp_storage: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_id, prior_artifact_id = _seed_canonical_state()
    prior_path = _materialize_prior_artifact(project_id, prior_artifact_id)
    job_id = _new_running_job(project_id)
    with get_session_factory()() as db, atomic_write(db):
        checkpointed_job = db.get(GenerationJob, job_id)
        checkpointed_job.plan_json = {
            **(checkpointed_job.plan_json or {}),
            "resume_inputs": checkpointed_job.input_snapshot,
            "resume_fingerprint": checkpointed_job.input_fingerprint,
        }
    directory = project_dir(project_id) / "history" / f"job-{job_id:08d}"
    candidate = directory / "video.pending.mp4"
    candidate.parent.mkdir(parents=True, exist_ok=True)
    candidate.write_bytes(b"abandoned-verified-video")
    marker_path = temp_storage / "resumed-artifact-after-rename.markers"

    result = _run_process_death(
        "artifact_after_rename", project_id, marker_path, candidate
    )

    assert result.returncode == 92, result.stderr
    assert _marker_lines(marker_path) == ["checkpoint_saved", "artifact_renamed"]
    abandoned = directory / "video.mp4"
    assert abandoned.read_bytes() == b"abandoned-verified-video"
    assert restart_twice() == (1, 0)
    pending = recovery_snapshot(project_id)
    job = next(row for row in pending["jobs"] if row["id"] == job_id)
    assert job["status"] == "pending"
    assert pending["project"]["current_artifact_id"] == prior_artifact_id

    candidate.write_bytes(b"new-verified-candidate")

    async def validate(path: Path) -> dict[str, object]:
        return {
            **artifact_store.file_identity(path),
            "duration_ms": 1000,
            "width": 320,
            "height": 180,
        }

    monkeypatch.setattr(artifact_store, "validate_video", validate)
    callback_count = 0

    async def complete_locally(target_job_id: int, cancel_check: Any) -> None:
        nonlocal callback_count
        callback_count += 1
        assert target_job_id == job_id
        with get_session_factory()() as db:
            resumed_job = db.get(GenerationJob, job_id)
            project = db.get(Project, project_id)
            assert resumed_job is not None
            assert project is not None
            assert resumed_job.status == JobStatus.running
            assert resumed_job.input_revision == project.revision == 1
            assert resumed_job.input_fingerprint == fingerprint_inputs(
                resumed_job.input_snapshot
            )
            assert resumed_job.input_fingerprint == fingerprint_inputs(
                capture_inputs(project)
            )
            assert resumed_job.plan_json["resume_fingerprint"] == fingerprint_inputs(
                resumed_job.plan_json["resume_inputs"]
            )
            assert resumed_job.plan_json["resume_fingerprint"] == (
                resumed_job.input_fingerprint
            )
            settled_inputs = resumed_job.plan_json["resume_inputs"]
        await artifact_store.publish_artifact(
            job_id,
            candidate,
            None,
            settled_inputs=settled_inputs,
            materials=[],
            cancel_check=cancel_check,
        )

    monkeypatch.setattr(operation_dispatcher, "run_generation_job", complete_locally)

    async def dispatch_and_wait() -> None:
        registry = JobRegistry()
        assert dispatch_pending_operation_jobs(registry=registry) == 1
        assert dispatch_pending_operation_jobs(registry=registry) == 0
        tasks = list(registry._tasks.values())
        assert len(tasks) == 1
        await asyncio.gather(*tasks)

    asyncio.run(dispatch_and_wait())

    assert callback_count == 1
    completed = recovery_snapshot(project_id)
    job = next(row for row in completed["jobs"] if row["id"] == job_id)
    artifacts = [row for row in completed["artifacts"] if row["job_id"] == job_id]
    assert job["status"] == "completed"
    assert len(artifacts) == 1
    assert completed["project"]["current_artifact_id"] == artifacts[0]["id"]
    assert completed["project"]["output_video_path"] == artifacts[0]["video_path"]
    assert abandoned.read_bytes() == b"new-verified-candidate"
    assert prior_path.read_bytes() == b"prior-success"
    assert any(
        row["id"] == prior_artifact_id for row in completed["artifacts"]
    )
    assert [row for row in completed["external_calls"] if row["job_id"] == job_id] == []

    candidate.write_bytes(b"must-not-replace-current")

    async def reject_validation(path: Path) -> dict[str, object]:
        pytest.fail(f"completed replay revalidated candidate {path}")

    monkeypatch.setattr(artifact_store, "validate_video", reject_validation)
    replayed = asyncio.run(
        artifact_store.publish_artifact(
            job_id,
            candidate,
            None,
            settled_inputs=job["input_snapshot"],
            materials=[],
            cancel_check=lambda: pytest.fail(
                "completed replay reconsidered cancellation"
            ),
        )
    )
    assert replayed.id == artifacts[0]["id"]
    assert abandoned.read_bytes() == b"new-verified-candidate"
    assert recovery_snapshot(project_id) == completed
    assert restart_twice() == (0, 0)
    assert recovery_snapshot(project_id) == completed


def test_resumed_publication_does_not_replace_referenced_job_history_video(
    temp_storage: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_id, prior_artifact_id = _seed_canonical_state()
    job_id = _new_running_job(project_id)
    directory = project_dir(project_id) / "history" / f"job-{job_id:08d}"
    candidate = directory / "video.pending.mp4"
    final = directory / "video.mp4"
    directory.mkdir(parents=True, exist_ok=True)
    candidate.write_bytes(b"new-candidate")
    final.write_bytes(b"referenced-current")
    final_identity = artifact_store.file_identity(final)
    with get_session_factory()() as db:
        prior = db.get(GenerationArtifact, prior_artifact_id)
        project = db.get(Project, project_id)
        prior.video_path = final_identity["path"]
        prior.manifest_json = {"schema_version": 1, "video": final_identity}
        project.output_video_path = final_identity["path"]
        db.commit()

    async def validate(path: Path) -> dict[str, object]:
        return {
            **artifact_store.file_identity(path),
            "duration_ms": 1000,
            "width": 320,
            "height": 180,
        }

    monkeypatch.setattr(artifact_store, "validate_video", validate)
    with get_session_factory()() as db:
        settled_inputs = db.get(GenerationJob, job_id).input_snapshot
    with pytest.raises(RuntimeError, match="確定動画"):
        asyncio.run(
            artifact_store.publish_artifact(
                job_id,
                candidate,
                None,
                settled_inputs=settled_inputs,
                materials=[],
                cancel_check=lambda: False,
            )
        )
    assert final.read_bytes() == b"referenced-current"
    snapshot = recovery_snapshot(project_id)
    assert snapshot["project"]["current_artifact_id"] == prior_artifact_id
    assert [row for row in snapshot["artifacts"] if row["job_id"] == job_id] == []
    _assert_publication_recovery_is_stable(
        project_id, job_id, prior_artifact_id, "pending"
    )


@pytest.mark.parametrize(
    ("blocker", "error_type"),
    [
        ("cancelled", GenerationCancelled),
        ("stale_revision", StaleGenerationInput),
        ("stale_job_input", StaleGenerationInput),
        ("stale_project_input", StaleGenerationInput),
        ("unresolved_call", ExternalOutcomeUnknown),
    ],
)
def test_publication_eligibility_failure_leaves_candidate_and_destination_unchanged(
    temp_storage: Path,
    monkeypatch: pytest.MonkeyPatch,
    blocker: str,
    error_type: type[Exception],
) -> None:
    project_id, prior_artifact_id = _seed_canonical_state()
    job_id = _new_running_job(project_id)
    directory = project_dir(project_id) / "history" / f"job-{job_id:08d}"
    candidate = directory / "video.pending.mp4"
    final = directory / "video.mp4"
    directory.mkdir(parents=True, exist_ok=True)
    candidate_bytes = f"candidate-{blocker}".encode()
    candidate.write_bytes(candidate_bytes)

    async def validate(path: Path) -> dict[str, object]:
        return {
            **artifact_store.file_identity(path),
            "duration_ms": 1000,
            "width": 320,
            "height": 180,
        }

    monkeypatch.setattr(artifact_store, "validate_video", validate)
    with get_session_factory()() as db, atomic_write(db):
        job = db.get(GenerationJob, job_id)
        project = db.get(Project, project_id)
        settled_inputs = job.input_snapshot
        if blocker == "cancelled":
            job.cancel_requested = True
        elif blocker == "stale_revision":
            project.revision += 1
        elif blocker == "stale_job_input":
            job.input_fingerprint = "0" * 64
        elif blocker == "stale_project_input":
            project.source_script = "publication must not adopt this changed input"
        else:
            db.add(
                ExternalCall(
                    job_id=job_id,
                    fingerprint="f" * 64,
                    provider="synthetic",
                    endpoint="https://synthetic.invalid/d33-publication",
                    remote_side_effect=True,
                    status="unknown",
                )
            )

    with pytest.raises(error_type):
        asyncio.run(
            artifact_store.publish_artifact(
                job_id,
                candidate,
                None,
                settled_inputs=settled_inputs,
                materials=[],
                cancel_check=lambda: False,
            )
        )

    assert candidate.read_bytes() == candidate_bytes
    assert not final.exists()
    assert [
        row
        for row in recovery_snapshot(project_id)["artifacts"]
        if row["job_id"] == job_id
    ] == []
    expected_status = {
        "cancelled": "cancelled",
        "unresolved_call": "unknown",
    }.get(blocker, "failed")
    _assert_publication_recovery_is_stable(
        project_id, job_id, prior_artifact_id, expected_status
    )


@pytest.mark.parametrize("change", ["mutate", "delete"])
def test_subtitle_identity_change_at_artifact_boundary_rolls_back_publication(
    temp_storage: Path,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    project_id, prior_artifact_id = _seed_canonical_state()
    prior_path = _materialize_prior_artifact(project_id, prior_artifact_id)
    job_id = _new_running_job(project_id)
    directory = project_dir(project_id) / "history" / f"job-{job_id:08d}"
    candidate = directory / "video.pending.mp4"
    final = directory / "video.mp4"
    subtitle = directory / "subtitles.ass"
    directory.mkdir(parents=True, exist_ok=True)
    candidate_bytes = b"subtitle-boundary-video"
    candidate.write_bytes(candidate_bytes)
    subtitle.write_bytes(b"verified-subtitles")
    before = recovery_snapshot(project_id)

    async def validate(path: Path) -> dict[str, object]:
        return {
            **artifact_store.file_identity(path),
            "duration_ms": 1000,
            "width": 320,
            "height": 180,
        }

    monkeypatch.setattr(artifact_store, "validate_video", validate)
    original_replace = Path.replace

    def change_subtitle_after_rename(path: Path, target: Path) -> Path:
        result = original_replace(path, target)
        if path == candidate:
            if change == "mutate":
                subtitle.write_bytes(b"changed-after-initial-verification")
            else:
                subtitle.unlink()
        return result

    monkeypatch.setattr(Path, "replace", change_subtitle_after_rename)
    with get_session_factory()() as db:
        settled_inputs = db.get(GenerationJob, job_id).input_snapshot

    with pytest.raises(StaleGenerationInput, match="字幕"):
        asyncio.run(
            artifact_store.publish_artifact(
                job_id,
                candidate,
                subtitle,
                settled_inputs=settled_inputs,
                materials=[],
                cancel_check=lambda: False,
            )
        )

    assert not candidate.exists()
    assert final.read_bytes() == candidate_bytes
    assert not (directory / "manifest.json").exists()
    assert recovery_snapshot(project_id) == before
    assert prior_path.read_bytes() == b"prior-success"
    _assert_publication_recovery_is_stable(
        project_id, job_id, prior_artifact_id, "pending"
    )
    assert prior_path.read_bytes() == b"prior-success"


@pytest.mark.parametrize(
    "reference_kind",
    ["artifact_path", "same_job", "project_output", "project_current"],
)
def test_identical_destination_reference_never_removes_or_replaces_files(
    temp_storage: Path,
    monkeypatch: pytest.MonkeyPatch,
    reference_kind: str,
) -> None:
    project_id, prior_artifact_id = _seed_canonical_state()
    job_id = _new_running_job(project_id)
    directory = project_dir(project_id) / "history" / f"job-{job_id:08d}"
    candidate = directory / "video.pending.mp4"
    final = directory / "video.mp4"
    directory.mkdir(parents=True, exist_ok=True)
    identical_bytes = b"identical-but-referenced"
    candidate.write_bytes(identical_bytes)
    final.write_bytes(identical_bytes)
    final_path = relpath_for_db(final)

    with get_session_factory()() as db, atomic_write(db):
        project = db.get(Project, project_id)
        prior = db.get(GenerationArtifact, prior_artifact_id)
        if reference_kind == "artifact_path":
            prior.video_path = final_path
        elif reference_kind == "same_job":
            prior.job_id = job_id
        elif reference_kind == "project_output":
            project.output_video_path = final_path
        else:
            prior.video_path = final_path
            project.current_artifact_id = prior.id
        settled_inputs = db.get(GenerationJob, job_id).input_snapshot

    async def validate(path: Path) -> dict[str, object]:
        return {
            **artifact_store.file_identity(path),
            "duration_ms": 1000,
            "width": 320,
            "height": 180,
        }

    monkeypatch.setattr(artifact_store, "validate_video", validate)

    def publication() -> GenerationArtifact:
        return asyncio.run(
            artifact_store.publish_artifact(
                job_id,
                candidate,
                None,
                settled_inputs=settled_inputs,
                materials=[],
                cancel_check=lambda: False,
            )
        )

    if reference_kind == "same_job":
        assert publication().id == prior_artifact_id
    else:
        with pytest.raises(RuntimeError, match="確定動画"):
            publication()

    assert candidate.read_bytes() == identical_bytes
    assert final.read_bytes() == identical_bytes
    snapshot = recovery_snapshot(project_id)
    assert len(snapshot["artifacts"]) == 1
    assert snapshot["project"]["current_artifact_id"] == prior_artifact_id
    _assert_publication_recovery_is_stable(
        project_id, job_id, prior_artifact_id, "pending"
    )


@pytest.mark.parametrize("boundary", ["flush", "commit"])
def test_artifact_database_failure_leaves_only_unreferenced_video_orphan(
    temp_storage: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
) -> None:
    project_id, prior_artifact_id = _seed_canonical_state()
    job_id = _new_running_job(project_id)
    directory = project_dir(project_id) / "history" / f"job-{job_id:08d}"
    candidate = directory / "video.pending.mp4"
    final = directory / "video.mp4"
    directory.mkdir(parents=True, exist_ok=True)
    candidate_bytes = f"database-{boundary}-failure".encode()
    candidate.write_bytes(candidate_bytes)
    before = recovery_snapshot(project_id)

    async def validate(path: Path) -> dict[str, object]:
        return {
            **artifact_store.file_identity(path),
            "duration_ms": 1000,
            "width": 320,
            "height": 180,
        }

    monkeypatch.setattr(artifact_store, "validate_video", validate)
    original_flush = Session.flush
    original_commit = Session.commit
    if boundary == "flush":
        def fail_artifact_flush(self: Session, objects: Any = None) -> None:
            if any(isinstance(value, GenerationArtifact) for value in self.new):
                raise OSError("synthetic artifact flush failure")
            original_flush(self, objects)

        monkeypatch.setattr(Session, "flush", fail_artifact_flush)
    else:
        def fail_artifact_commit(self: Session) -> None:
            raise OSError("synthetic artifact commit failure")

        monkeypatch.setattr(Session, "commit", fail_artifact_commit)

    with get_session_factory()() as db:
        settled_inputs = db.get(GenerationJob, job_id).input_snapshot

    with pytest.raises(OSError, match=f"synthetic artifact {boundary} failure"):
        asyncio.run(
            artifact_store.publish_artifact(
                job_id,
                candidate,
                None,
                settled_inputs=settled_inputs,
                materials=[],
                cancel_check=lambda: False,
            )
        )

    assert not candidate.exists()
    assert final.read_bytes() == candidate_bytes
    assert set(directory.iterdir()) == {final}
    assert not (directory / "manifest.json").exists()
    assert recovery_snapshot(project_id) == before
    monkeypatch.setattr(Session, "flush", original_flush)
    monkeypatch.setattr(Session, "commit", original_commit)
    _assert_publication_recovery_is_stable(
        project_id, job_id, prior_artifact_id, "pending"
    )


@pytest.mark.parametrize("boundary", ["before", "after"])
def test_core_commit_crash_has_no_effect_or_complete_generation_receipt(
    temp_storage, monkeypatch: pytest.MonkeyPatch, boundary: str
) -> None:
    project_id, artifact_id = _seed_canonical_state()
    service, request, prepared = _prepare_generation(
        project_id, f"d33-core-commit-{boundary}"
    )
    confirmation = LanguageExecution(
        confirmation_token=prepared.confirmation_token,
        confirm_generation=True,
    )
    before = recovery_snapshot(project_id)

    with get_session_factory()() as db:
        original_commit = db.commit
        commit_count = 0

        def crashing_commit() -> None:
            nonlocal commit_count
            commit_count += 1
            if commit_count != 2:
                original_commit()
                return
            if boundary == "after":
                original_commit()
            raise OSError("synthetic D33 boundary")

        monkeypatch.setattr(db, "commit", crashing_commit)
        with pytest.raises(OSError, match="synthetic D33 boundary"):
            service.execute(db, request.request_id, confirmation)
        assert commit_count == 2

    crashed = recovery_snapshot(project_id)
    core_request_id = prepared.core_request_id
    if boundary == "before":
        assert crashed == before
    else:
        with get_session_factory()() as db:
            persisted = operation_service.get_result(db, core_request_id)
        _assert_atomic_generation_effect(
            before, crashed, core_request_id, persisted
        )
    assert crashed["project"]["current_artifact_id"] == artifact_id
    assert restart_twice() == (0, 0)
    assert recovery_snapshot(project_id) == crashed

    with get_session_factory()() as db:
        completed = service.execute(db, request.request_id, confirmation)
    assert completed.status == "completed"
    assert completed.result is not None
    final = recovery_snapshot(project_id)
    _assert_atomic_generation_effect(
        before, final, core_request_id, completed.result
    )
    assert restart_twice() == (0, 0)
    assert recovery_snapshot(project_id) == final
    with get_session_factory()() as db:
        replay = service.execute(db, request.request_id, confirmation)
    assert replay == completed
    assert recovery_snapshot(project_id) == final
    assert restart_twice() == (0, 0)
    assert recovery_snapshot(project_id) == final
