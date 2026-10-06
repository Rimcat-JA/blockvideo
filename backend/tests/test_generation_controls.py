"""D12–D15 acceptance of distinct save/generate/cancel/retry/historical-restore intents."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from typing import Any
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from app.db import get_session_factory
from app.main import create_app
from app.models.external_call import ExternalCall
from app.models.job import GenerationJob, JobStatus
from app.models.operation_request import OperationReceipt
from app.models.project import Project
from app.models.settings_revision import SettingsRevision
from app.operations.bootstrap import build_operation_service
from app.operations.contracts import OperationRequest, OperationResult
from app.services.settings_history import configuration


def make_project() -> int:
    with get_session_factory()() as db:
        project = Project(title="D15 synthetic", source_script="検証用の合成台本です。",
                          subtitle_font_size=48, use_fake_providers=True)
        db.add(project)
        db.commit()
        return project.id


def request(project_id: int, operation: str, arguments: dict[str, Any] | None = None, *,
            revision: int | None = None, request_id: str | None = None,
            generate: bool = False) -> OperationRequest:
    if revision is None:
        with get_session_factory()() as db:
            revision = db.get(Project, project_id).revision
    return OperationRequest(operation_id=f"project.{operation}", target={"project_id": project_id},
                            arguments=arguments or {}, base_revision=revision,
                            request_id=request_id or str(uuid4()), generation_requested=generate)


def execute(value: OperationRequest) -> OperationResult:
    with get_session_factory()() as db:
        return build_operation_service().execute(db, value)


def post(client: TestClient, value: OperationRequest):
    return client.post("/api/operations/execute", json=value.model_dump(mode="json"))


def finish(job_id: int, status: JobStatus = JobStatus.failed) -> None:
    with get_session_factory()() as db:
        db.get(GenerationJob, job_id).status = status
        db.commit()


@pytest.mark.parametrize("generate", [False, True])
def test_multiple_settings_save_one_revision_and_at_most_one_explicit_job(temp_storage, generate: bool) -> None:
    project_id = make_project()
    value = request(project_id, "settings.update", {
        "subtitle_font_size": 56, "voicevox_speed_scale": 1.2,
        "pronunciation_overrides": [{"surface": "API", "reading": "エーピーアイ"}],
    }, generate=generate)
    result = execute(value)
    assert result.revision == 2
    assert result.generation_requested is generate
    assert (result.job_id is not None) is generate
    assert execute(value) == result
    with get_session_factory()() as db:
        project = db.get(Project, project_id)
        assert project.subtitle_font_size == 56
        assert project.voicevox_speed_scale == 1.2
        assert project.pronunciation_overrides[0]["reading"] == "エーピーアイ"
        assert db.scalar(select(func.count()).select_from(GenerationJob)) == int(generate)
        assert db.scalar(select(func.count()).select_from(OperationReceipt)) == 1
        assert list(db.scalars(select(SettingsRevision.revision).order_by(SettingsRevision.revision))) == [1, 2]
        if generate:
            job = db.get(GenerationJob, result.job_id)
            assert job.input_revision == 2
            assert job.input_snapshot["project"]["subtitle_font_size"] == 56
            assert job.plan_json["stages"] == ["split", "plan", "image", "audio", "render"]


def test_combined_validation_failure_rolls_back_all_fields_and_job(temp_storage) -> None:
    project_id = make_project()
    client = TestClient(create_app())
    response = post(client, request(project_id, "settings.update", {
        "subtitle_font_size": 56, "voicevox_speed_scale": 9.0,
    }, generate=True))
    assert response.status_code == 422
    with get_session_factory()() as db:
        project = db.get(Project, project_id)
        assert (project.subtitle_font_size, project.revision) == (48, 1)
        assert db.scalar(select(func.count()).select_from(GenerationJob)) == 0
        assert db.scalar(select(func.count()).select_from(OperationReceipt)) == 0


def test_restore_any_older_settings_creates_new_revision_and_preserves_source(temp_storage) -> None:
    project_id = make_project()
    execute(request(project_id, "settings.update", {"subtitle_font_size": 56}))
    execute(request(project_id, "settings.update", {"subtitle_font_size": 64, "voicevox_speed_scale": 1.3}))
    with get_session_factory()() as db:
        baseline = db.scalar(select(SettingsRevision).where(SettingsRevision.revision == 1)).settings_json
        db.get(Project, project_id).source_script = "後から変更した合成台本。"
        db.commit()
    restore = request(project_id, "settings.restore", {"revision": 1})
    result = execute(restore)
    assert result.revision == 4
    assert result.data["restored_from_revision"] == 1
    assert not result.generation_requested and result.job_id is None
    assert execute(restore) == result
    with get_session_factory()() as db:
        project = db.get(Project, project_id)
        assert configuration(project) == baseline
        assert project.source_script == "後から変更した合成台本。"
        assert db.scalar(select(func.count()).select_from(GenerationJob)) == 0
        row = db.scalar(select(SettingsRevision).where(SettingsRevision.revision == 4))
        assert row.restored_from_revision == 1


def test_noop_restore_is_recorded_as_a_new_revision(temp_storage) -> None:
    project_id = make_project()
    execute(request(project_id, "settings.update", {"subtitle_font_size": 56}))
    result = execute(request(project_id, "settings.restore", {"revision": 2}))
    assert result.revision == 3
    with get_session_factory()() as db:
        row = db.scalar(select(SettingsRevision).where(SettingsRevision.revision == 3))
        assert row is not None and row.restored_from_revision == 2
        assert db.scalar(select(func.count()).select_from(GenerationJob)) == 0


def test_stale_restore_cannot_overwrite_later_settings(temp_storage) -> None:
    project_id = make_project()
    execute(request(project_id, "settings.update", {"subtitle_font_size": 56}))
    stale = request(project_id, "settings.restore", {"revision": 1})
    execute(request(project_id, "settings.update", {"subtitle_font_size": 72}))
    response = post(TestClient(create_app()), stale)
    assert response.status_code == 409
    assert response.json()["detail"]["reason_code"] == "stale_state"
    with get_session_factory()() as db:
        project = db.get(Project, project_id)
        assert (project.subtitle_font_size, project.revision) == (72, 3)
        assert db.get(OperationReceipt, stale.request_id) is None


def test_restore_while_generating_rejects_without_cancelling_job(temp_storage) -> None:
    project_id = make_project()
    execute(request(project_id, "settings.update", {"subtitle_font_size": 56}))
    generated = execute(request(project_id, "generation.start"))
    response = post(TestClient(create_app()), request(project_id, "settings.restore", {"revision": 1}))
    assert response.status_code == 409
    assert response.json()["detail"]["reason_code"] == "project_busy"
    with get_session_factory()() as db:
        job = db.get(GenerationJob, generated.job_id)
        assert job.status == JobStatus.pending and not job.cancel_requested
        assert db.get(Project, project_id).subtitle_font_size == 56


@pytest.mark.parametrize("source_status", [JobStatus.failed, JobStatus.cancelled])
def test_retry_uses_current_settings_and_links_original_job(temp_storage, source_status: JobStatus) -> None:
    project_id = make_project()
    original = execute(request(project_id, "generation.start"))
    finish(original.job_id, source_status)
    execute(request(project_id, "settings.update", {"subtitle_font_size": 68, "voicevox_speed_scale": 1.4}))
    retry = request(project_id, "generation.retry", {"job_id": original.job_id})
    result = execute(retry)
    assert result.job_id != original.job_id
    assert result.data["uses_current_settings"] is True
    assert execute(retry) == result
    with get_session_factory()() as db:
        old = db.get(GenerationJob, original.job_id)
        current = db.get(GenerationJob, result.job_id)
        assert old.input_snapshot["project"]["subtitle_font_size"] == 48
        assert old.status == source_status
        assert current.parent_job_id == old.id
        assert current.input_revision == 2
        assert current.input_snapshot["project"]["subtitle_font_size"] == 68
        assert current.input_snapshot["project"]["voicevox_speed_scale"] == 1.4
        assert db.scalar(select(func.count()).select_from(GenerationJob)) == 2


def test_concurrent_retry_resends_create_one_new_job_and_changed_id_body_conflicts(temp_storage) -> None:
    project_id = make_project()
    original = execute(request(project_id, "generation.start"))
    finish(original.job_id)
    retry = request(project_id, "generation.retry", {"job_id": original.job_id})
    barrier = Barrier(4)

    def worker(_: int) -> dict:
        barrier.wait(timeout=10)
        return execute(retry).model_dump(mode="json")

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(worker, range(4)))
    assert all(result == results[0] for result in results)
    conflict = retry.model_copy(update={"arguments": {"job_id": results[0]["job_id"]}})
    response = post(TestClient(create_app()), conflict)
    assert response.status_code == 409
    assert response.json()["detail"]["reason_code"] == "request_id_conflict"
    with get_session_factory()() as db:
        assert db.scalar(select(func.count()).select_from(GenerationJob)) == 2


@pytest.mark.parametrize("uncertainty", ["unknown_job", "unknown_call", "in_flight_call"])
@pytest.mark.parametrize("operation", ["generation.start", "generation.retry"])
def test_uncertain_remote_work_blocks_new_generation_and_retry(temp_storage, uncertainty: str, operation: str) -> None:
    project_id = make_project()
    original = execute(request(project_id, "generation.start"))
    finish(original.job_id, JobStatus.unknown if uncertainty == "unknown_job" else JobStatus.failed)
    if uncertainty != "unknown_job":
        with get_session_factory()() as db:
            db.add(ExternalCall(job_id=original.job_id, fingerprint="synthetic-unknown", provider="synthetic",
                                endpoint="https://synthetic.invalid", remote_side_effect=True,
                                status="unknown" if uncertainty == "unknown_call" else "in_flight"))
            db.commit()
    value = request(project_id, operation, {"job_id": original.job_id} if operation.endswith("retry") else {})
    response = post(TestClient(create_app()), value)
    assert response.status_code == 409
    assert response.json()["detail"]["reason_code"] == "external_outcome_unknown"
    with get_session_factory()() as db:
        assert db.scalar(select(func.count()).select_from(GenerationJob)) == 1
        assert db.get(OperationReceipt, value.request_id) is None


def test_local_unknown_call_does_not_block_retry(temp_storage) -> None:
    project_id = make_project()
    failed = execute(request(project_id, "generation.start"))
    finish(failed.job_id)
    with get_session_factory()() as db:
        db.add(ExternalCall(
            job_id=failed.job_id,
            fingerprint="local-unknown",
            provider="synthetic-local",
            endpoint="http://localhost/jobs",
            remote_side_effect=False,
            status="unknown",
        ))
        db.commit()

    result = execute(request(project_id, "generation.retry", {"job_id": failed.job_id}))

    assert result.job_id is not None
    with get_session_factory()() as db:
        assert db.get(GenerationJob, result.job_id).parent_job_id == failed.job_id


def test_cancelled_job_with_persisted_cancel_request_can_retry(temp_storage) -> None:
    project_id = make_project()
    cancelled = execute(request(project_id, "generation.start"))
    with get_session_factory()() as db:
        job = db.get(GenerationJob, cancelled.job_id)
        job.status = JobStatus.cancelled
        job.cancel_requested = True
        db.commit()

    result = execute(request(project_id, "generation.retry", {"job_id": cancelled.job_id}))

    assert result.job_id is not None
    with get_session_factory()() as db:
        assert db.get(GenerationJob, result.job_id).parent_job_id == cancelled.job_id


def test_other_job_unknown_call_also_blocks_retry_of_a_known_failure(temp_storage) -> None:
    project_id = make_project()
    failed = execute(request(project_id, "generation.start"))
    finish(failed.job_id)
    other = execute(request(project_id, "generation.start"))
    finish(other.job_id)
    with get_session_factory()() as db:
        db.add(ExternalCall(job_id=other.job_id, fingerprint="other-unknown", provider="synthetic",
                            endpoint="https://synthetic.invalid", remote_side_effect=True, status="unknown"))
        db.commit()
    response = post(TestClient(create_app()), request(project_id, "generation.retry", {"job_id": failed.job_id}))
    assert response.status_code == 409
    assert response.json()["detail"]["reason_code"] == "external_outcome_unknown"
    with get_session_factory()() as db:
        assert db.scalar(select(func.count()).select_from(GenerationJob)) == 2


@pytest.mark.parametrize("initial", [JobStatus.pending, JobStatus.running, JobStatus.completed])
def test_cancel_is_idempotent_preserves_settings_and_respects_completed_work(temp_storage, initial: JobStatus) -> None:
    project_id = make_project()
    generated = execute(request(project_id, "settings.update", {"subtitle_font_size": 62}, generate=True))
    finish(generated.job_id, initial)
    cancel = request(project_id, "generation.cancel", {"job_id": generated.job_id})
    result = execute(cancel)
    assert execute(cancel) == result
    again = execute(request(project_id, "generation.cancel", {"job_id": generated.job_id}))
    assert again.changed is False
    with get_session_factory()() as db:
        job = db.get(GenerationJob, generated.job_id)
        assert job.status == (JobStatus.cancelled if initial == JobStatus.pending else initial)
        assert job.cancel_requested is (initial != JobStatus.completed)
        assert db.scalar(select(func.count()).select_from(GenerationJob)) == 1
        project = db.get(Project, project_id)
        assert (project.subtitle_font_size, project.revision) == (62, 2)


def test_job_controls_cannot_target_another_project(temp_storage) -> None:
    owner = make_project()
    wrong = make_project()
    generated = execute(request(owner, "generation.start"))
    finish(generated.job_id)
    for operation in ["generation.cancel", "generation.retry"]:
        response = post(TestClient(create_app()), request(wrong, operation, {"job_id": generated.job_id}))
        assert response.status_code == 404
        assert response.json()["detail"]["reason_code"] == "job_not_found"


def test_generation_failure_does_not_partially_save_combined_settings(temp_storage) -> None:
    project_id = make_project()
    original = execute(request(project_id, "generation.start"))
    finish(original.job_id, JobStatus.unknown)
    value = request(project_id, "settings.update", {"subtitle_font_size": 78}, generate=True)
    response = post(TestClient(create_app()), value)
    assert response.status_code == 409
    with get_session_factory()() as db:
        project = db.get(Project, project_id)
        assert (project.subtitle_font_size, project.revision) == (48, 1)
        assert db.get(OperationReceipt, value.request_id) is None
        assert list(db.scalars(select(SettingsRevision.revision))) == [1]


def test_project_deletion_cannot_attach_old_settings_to_a_new_project(temp_storage) -> None:
    old_project = make_project()
    execute(request(old_project, "settings.update", {"subtitle_font_size": 56}))
    client = TestClient(create_app())
    assert client.delete(f"/api/projects/{old_project}").status_code == 204
    created = client.post("/api/projects", json={"title": "New synthetic owner", "source_script": "新しい合成台本。",
                                                "subtitle_font_size": 88, "use_fake_providers": True})
    assert created.status_code == 201
    project_id = created.json()["id"]
    execute(request(project_id, "settings.update", {"subtitle_font_size": 92}))
    execute(request(project_id, "settings.restore", {"revision": 1}))
    with get_session_factory()() as db:
        project = db.get(Project, project_id)
        assert project.title == "New synthetic owner"
        assert project.subtitle_font_size == 88


def test_history_exposes_revision_settings_versions_and_safe_job_metadata(temp_storage) -> None:
    project_id = make_project()
    execute(request(project_id, "settings.update", {"subtitle_font_size": 58}))
    generated = execute(request(project_id, "generation.start"))
    response = TestClient(create_app()).get(f"/api/projects/{project_id}/history")
    assert response.status_code == 200
    body = response.json()
    assert body["revision"] == 2
    assert body["output_state"] == "none"
    assert body["current_artifact_id"] is None
    assert body["artifacts"] == []
    versions = {version["revision"]: version for version in body["settings_versions"]}
    assert versions[1]["settings"]["subtitle_font_size"] == 48
    assert versions[2]["settings"]["subtitle_font_size"] == 58
    assert versions[2]["changed_fields"] == ["subtitle_font_size"]
    assert body["jobs"][0]["id"] == generated.job_id
    assert body["jobs"][0]["input_revision"] == 2
    assert body["jobs"][0]["cancel_requested"] is False
    assert "input_snapshot" not in body["jobs"][0]
    assert "検証用の合成台本" not in response.text
