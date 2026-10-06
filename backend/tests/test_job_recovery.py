"""D14 restart decisions use persisted snapshots and remote-call outcomes."""
from __future__ import annotations

from typing import Any

import pytest
from sqlalchemy import select

from app.db import get_session_factory
from app.models.artifact import GenerationArtifact
from app.models.block import Block
from app.models.external_call import ExternalCall
from app.models.job import GenerationJob, JobStatus
from app.models.operation_request import OperationReceipt
from app.models.project import Project, ProjectStatus
from app.services.generation_snapshots import capture_inputs, fingerprint_inputs
from app.services.job_records import create_pending_job
from app.services.transactions import atomic_write
from app.workers import operation_dispatcher as dispatcher


def persisted_job(*, status: JobStatus = JobStatus.running) -> tuple[int, int]:
    with get_session_factory()() as db:
        project = Project(title="D14 synthetic", source_script="安全な復旧を検証する合成台本です。",
                          use_fake_providers=True)
        db.add(project)
        db.commit()
        project_id = project.id
    with get_session_factory()() as db, atomic_write(db):
        job = create_pending_job(db, project_id)
        job.status = status
        job_id = job.id
    return project_id, job_id


def interrupted_call(job_id: int, *, remote: bool = True) -> None:
    with get_session_factory()() as db:
        db.add(ExternalCall(job_id=job_id, fingerprint=f"synthetic-{job_id}", provider="synthetic",
                            endpoint="https://synthetic.invalid", remote_side_effect=remote,
                            status="in_flight"))
        db.commit()


def test_safe_running_snapshot_becomes_pending_once_without_receipt(temp_storage) -> None:
    project_id, job_id = persisted_job()
    with get_session_factory()() as db:
        assert db.scalar(select(OperationReceipt)) is None
    assert dispatcher.mark_interrupted_operation_jobs() == 1
    assert dispatcher.mark_interrupted_operation_jobs() == 0
    with get_session_factory()() as db:
        job = db.get(GenerationJob, job_id)
        assert job.status == JobStatus.pending
        assert job.finished_at is None
        assert job.recovery_message
        assert job.error_message is None
        assert db.get(Project, project_id).revision == 1


def test_completed_external_result_does_not_prevent_safe_resume(temp_storage) -> None:
    _, job_id = persisted_job()
    with get_session_factory()() as db:
        db.add(ExternalCall(job_id=job_id, fingerprint="saved-result", provider="synthetic",
                            endpoint="https://synthetic.invalid", remote_side_effect=True,
                            status="succeeded", response_status=200, response_body=b'{"saved":true}'))
        db.commit()
    assert dispatcher.mark_interrupted_operation_jobs() == 1
    with get_session_factory()() as db:
        assert db.get(GenerationJob, job_id).status == JobStatus.pending
        assert db.scalar(select(ExternalCall)).response_body == b'{"saved":true}'


@pytest.mark.parametrize("cancel_requested", [False, True])
def test_remote_inflight_becomes_unknown_even_if_cancellation_was_requested(temp_storage, cancel_requested: bool) -> None:
    _, job_id = persisted_job()
    interrupted_call(job_id)
    with get_session_factory()() as db:
        db.get(GenerationJob, job_id).cancel_requested = cancel_requested
        db.commit()
    assert dispatcher.mark_interrupted_operation_jobs() == 1
    with get_session_factory()() as db:
        job = db.get(GenerationJob, job_id)
        assert job.status == JobStatus.unknown
        assert job.finished_at is not None
        assert "未確定" in job.recovery_message
        assert db.scalar(select(ExternalCall)).status == "unknown"
    assert dispatcher.dispatch_pending_operation_jobs() == 0


def test_local_computation_is_retryable_after_restart(temp_storage) -> None:
    _, job_id = persisted_job()
    interrupted_call(job_id, remote=False)
    assert dispatcher.mark_interrupted_operation_jobs() == 1
    with get_session_factory()() as db:
        assert db.get(GenerationJob, job_id).status == JobStatus.pending
        assert db.scalar(select(ExternalCall)).status == "failed"


def test_persisted_cancellation_completes_without_running_more_work(temp_storage) -> None:
    project_id, job_id = persisted_job()
    with get_session_factory()() as db:
        db.get(GenerationJob, job_id).cancel_requested = True
        db.commit()
    assert dispatcher.mark_interrupted_operation_jobs() == 1
    with get_session_factory()() as db:
        job = db.get(GenerationJob, job_id)
        assert job.status == JobStatus.cancelled
        assert job.finished_at is not None
        project = db.get(Project, project_id)
        assert project.status == ProjectStatus.cancelled
        assert project.revision == 1
    assert dispatcher.dispatch_pending_operation_jobs() == 0


@pytest.mark.parametrize("corruption", ["missing_snapshot", "stale_revision", "changed_input", "corrupt_snapshot"])
def test_unverifiable_or_stale_inputs_fail_without_automatic_provider_work(temp_storage, corruption: str) -> None:
    project_id, job_id = persisted_job()
    with get_session_factory()() as db:
        project = db.get(Project, project_id)
        job = db.get(GenerationJob, job_id)
        if corruption == "missing_snapshot":
            job.input_snapshot = None
        elif corruption == "stale_revision":
            project.revision += 1
        elif corruption == "changed_input":
            project.source_script = "別の入力に変わった合成台本。"
        else:
            job.input_snapshot = {**job.input_snapshot, "project_id": 9999}
        db.commit()
    assert dispatcher.mark_interrupted_operation_jobs() == 1
    with get_session_factory()() as db:
        assert db.get(GenerationJob, job_id).status == JobStatus.failed
    assert dispatcher.dispatch_pending_operation_jobs() == 0


def test_generated_input_checkpoint_allows_resume_after_split(temp_storage) -> None:
    project_id, job_id = persisted_job()
    with get_session_factory()() as db:
        project = db.get(Project, project_id)
        db.add(Block(project_id=project_id, index=0, source_text="合成ブロック。", tts_text="合成ブロック。"))
        db.commit()
        db.expire_all()
        checkpoint = capture_inputs(project)
        job = db.get(GenerationJob, job_id)
        assert job.input_snapshot["blocks"] == []
        job.plan_json = {**job.plan_json, "resume_inputs": checkpoint,
                         "resume_fingerprint": fingerprint_inputs(checkpoint)}
        db.commit()
    assert dispatcher.mark_interrupted_operation_jobs() == 1
    with get_session_factory()() as db:
        assert db.get(GenerationJob, job_id).status == JobStatus.pending
        assert db.get(Project, project_id).revision == 1


@pytest.mark.parametrize("corruption", ["changed_snapshot", "missing_snapshot", "missing_fingerprint", "changed_settings"])
def test_corrupt_checkpoint_is_not_adopted_as_a_new_input_baseline(temp_storage, corruption: str) -> None:
    project_id, job_id = persisted_job()
    with get_session_factory()() as db:
        project = db.get(Project, project_id)
        job = db.get(GenerationJob, job_id)
        checkpoint = capture_inputs(project)
        resume: dict[str, Any] = {"resume_inputs": checkpoint, "resume_fingerprint": fingerprint_inputs(checkpoint)}
        if corruption == "changed_snapshot":
            resume["resume_inputs"] = {"corrupt": True}
        elif corruption == "missing_snapshot":
            resume.pop("resume_inputs")
        elif corruption == "missing_fingerprint":
            resume.pop("resume_fingerprint")
        else:
            project.subtitle_font_size += 2
            checkpoint = capture_inputs(project)
            resume = {"resume_inputs": checkpoint, "resume_fingerprint": fingerprint_inputs(checkpoint)}
        job.plan_json = {**job.plan_json, **resume}
        db.commit()
    assert dispatcher.mark_interrupted_operation_jobs() == 1
    with get_session_factory()() as db:
        assert db.get(GenerationJob, job_id).status == JobStatus.failed


def test_completed_artifact_and_job_remain_completed_after_restart(temp_storage) -> None:
    project_id, job_id = persisted_job(status=JobStatus.completed)
    with get_session_factory()() as db:
        artifact = GenerationArtifact(project_id=project_id, job_id=job_id, revision=1,
                                      video_path="synthetic/history/already-completed.mp4", manifest_json={})
        db.add(artifact)
        db.commit()
        artifact_id = artifact.id
        project = db.get(Project, project_id)
        project.current_artifact_id = artifact_id
        project.status = ProjectStatus.completed
        db.commit()
    assert dispatcher.mark_interrupted_operation_jobs() == 0
    assert dispatcher.dispatch_pending_operation_jobs() == 0
    with get_session_factory()() as db:
        assert db.get(GenerationJob, job_id).status == JobStatus.completed
        assert db.get(Project, project_id).current_artifact_id == artifact_id
        assert db.get(GenerationArtifact, artifact_id).job_id == job_id


def test_dispatcher_picks_up_all_snapshot_jobs_and_skips_legacy_untracked_rows(temp_storage, monkeypatch) -> None:
    _, job_id = persisted_job(status=JobStatus.pending)
    with get_session_factory()() as db:
        project = Project(title="Legacy synthetic", source_script="合成。", use_fake_providers=True)
        db.add(project)
        db.flush()
        db.add(GenerationJob(project_id=project.id, status=JobStatus.pending))
        db.commit()
    submissions: list[tuple[int, Any]] = []

    class Registry:
        def is_running(self, target: int) -> bool:
            return any(existing == target for existing, _ in submissions)

        def submit(self, target: int, factory: Any) -> None:
            submissions.append((target, factory))

    monkeypatch.setattr(dispatcher, "job_registry", Registry())
    assert dispatcher.dispatch_pending_operation_jobs() == 1
    assert dispatcher.dispatch_pending_operation_jobs() == 0
    assert [target for target, _ in submissions] == [job_id]
