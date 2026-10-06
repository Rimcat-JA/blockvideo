"""Explicit generation, cancellation, current-settings retry and history restores."""
from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.artifact import GenerationArtifact
from app.models.job import GenerationJob, JobStatus
from app.models.project import Project
from app.models.settings_revision import SettingsRevision
from app.operations.contracts import OperationResult
from app.operations.errors import OperationError
from app.services.artifact_store import artifact_file_path, artifact_is_available
from app.services.external_calls import has_unresolved_calls
from app.services.job_records import create_pending_job
from app.services.project_settings import apply_project_settings
from app.services.settings_history import configuration, validate_settings


def _result(project: Project, operation_id: str, *, changed: bool = False,
            data: dict[str, Any] | None = None, job: GenerationJob | None = None) -> OperationResult:
    return OperationResult(operation_id=operation_id, project_id=project.id, changed=changed,
                           revision=project.revision, state_revision=str(project.revision), data=data or {},
                           job_id=job.id if job else None, generation_requested=job is not None)


def update_settings(db: Session, project: Project, arguments: dict[str, Any]) -> OperationResult:
    try:
        updates = validate_settings(project, arguments)
    except ValueError as exc:
        raise OperationError("invalid_arguments", "設定値が正しくありません") from exc
    changed = apply_project_settings(project, updates)
    db.flush()
    return _result(project, "project.settings.update", changed=bool(changed),
                   data={"changed_fields": sorted(changed), "settings": configuration(project)})


def start_generation(db: Session, project: Project, arguments: dict[str, Any]) -> OperationResult:
    kind = arguments.get("kind", "full")
    block_index = arguments.get("block_index")
    if kind in {"full", "rerender"} and block_index is not None:
        raise OperationError("invalid_arguments", "この生成方法にはブロック番号を指定できません")
    try:
        job = create_pending_job(db, project.id, kind=kind, block_index=block_index)
    except ValueError as exc:
        raise OperationError("invalid_arguments", str(exc)) from exc
    return _result(project, "project.generation.start", job=job, data={"plan": job.plan_json})


def _target_job(db: Session, project: Project, job_id: int) -> GenerationJob:
    job = db.get(GenerationJob, job_id)
    if job is None or job.project_id != project.id:
        raise OperationError("job_not_found", "対象の生成処理が見つかりません")
    return job


def cancel_generation(db: Session, project: Project, arguments: dict[str, Any]) -> OperationResult:
    from app.services.job_control import cancel_job

    job = _target_job(db, project, arguments["job_id"])
    changed = cancel_job(db, job)
    return _result(project, "project.generation.cancel", changed=changed,
                   data={"job_id": job.id, "status": job.status.value,
                         "cancel_requested": job.cancel_requested})


def retry_generation(db: Session, project: Project, arguments: dict[str, Any]) -> OperationResult:
    previous = _target_job(db, project, arguments["job_id"])
    if has_unresolved_calls(db, previous.id) or previous.status == JobStatus.unknown:
        raise OperationError("external_outcome_unknown", "外部処理の結果が未確定です。自動的に再送できません")
    if previous.status not in {JobStatus.failed, JobStatus.cancelled}:
        raise OperationError("job_not_retryable", "失敗またはキャンセルされた処理を指定してください")
    job = create_pending_job(db, project.id, kind=previous.kind,
                             block_index=previous.block_index, parent_job_id=previous.id)
    return _result(project, "project.generation.retry", job=job,
                   data={"parent_job_id": previous.id, "uses_current_settings": True, "plan": job.plan_json})


def restore_settings(db: Session, project: Project, arguments: dict[str, Any]) -> OperationResult:
    saved = db.scalar(select(SettingsRevision).where(
        SettingsRevision.project_id == project.id, SettingsRevision.revision == arguments["revision"]))
    if saved is None:
        raise OperationError("settings_revision_not_found", "その版の設定は保存されていません")
    updates = validate_settings(project, saved.settings_json)
    changed = apply_project_settings(project, updates, restored_from_revision=saved.revision)
    db.flush()
    return _result(project, "project.settings.restore", changed=bool(changed),
                   data={"restored_from_revision": saved.revision, "settings": configuration(project),
                         "changed_fields": sorted(changed)})


def _subtitle_is_available(artifact: GenerationArtifact) -> bool:
    if artifact.subtitle_path is None:
        return True
    try:
        artifact_file_path(artifact, "subtitle")
        return True
    except (OSError, ValueError):
        return False


def restore_artifact(db: Session, project: Project, arguments: dict[str, Any]) -> OperationResult:
    """Point the project's current video at an earlier verified artifact; settings stay as they are."""
    artifact = db.get(GenerationArtifact, arguments["artifact_id"])
    if artifact is None or artifact.project_id != project.id:
        raise OperationError("artifact_not_found", "その番号の完成動画は出力履歴にありません")
    if not artifact_is_available(artifact) or not _subtitle_is_available(artifact):
        raise OperationError("artifact_unavailable", "その完成動画のファイルが欠損または変更されています")
    changed = project.current_artifact_id != artifact.id
    project.current_artifact_id = artifact.id
    project.output_video_path = artifact.video_path
    project.output_subtitle_path = artifact.subtitle_path
    db.flush()
    return _result(project, "project.artifact.restore", changed=changed,
                   data={"artifact_id": artifact.id, "artifact_revision": artifact.revision,
                         "settings_revision": project.revision})
