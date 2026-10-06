"""Common HTTP-boundary safety helpers.

Imports:
    ``Path`` resolves stored artifact values.
    ``HTTPException`` communicates invalid user-controlled paths as HTTP 400.
    ``get_settings`` supplies the trusted storage root.
"""
from __future__ import annotations

from pathlib import Path

from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.core.config import get_settings


def ensure_project_idle(project_id: int, db: Session) -> None:
    """Reject writes while a durable or process-local generation is active."""
    from app.services.job_records import has_active_project_job
    from app.workers.job_runner import job_registry

    if has_active_project_job(db, project_id, job_registry.is_running):
        raise HTTPException(status_code=409, detail="このプロジェクトは生成中です。完了後に再実行してください。")


def ensure_project_deletable(project_id: int, db: Session) -> None:
    """Reject deletion while local or externally uncertain work may still finish."""
    from sqlalchemy import select

    from app.models.external_call import ExternalCall
    from app.models.job import GenerationJob, JobStatus

    ensure_project_idle(project_id, db)
    unknown_job = db.scalar(
        select(GenerationJob.id)
        .where(
            GenerationJob.project_id == project_id,
            GenerationJob.status == JobStatus.unknown,
        )
        .limit(1)
    )
    unresolved_call = db.scalar(
        select(ExternalCall.id)
        .join(GenerationJob, GenerationJob.id == ExternalCall.job_id)
        .where(
            GenerationJob.project_id == project_id,
            ExternalCall.remote_side_effect.is_(True),
            ExternalCall.status.in_(["in_flight", "unknown"]),
        )
        .limit(1)
    )
    if unknown_job is not None or unresolved_call is not None:
        raise HTTPException(
            status_code=409,
            detail="外部処理の結果が未確定です。確認できるまで削除できません。",
        )


def delete_project_external_calls(project_id: int, db: Session) -> None:
    """Delete resolved journal rows before their owning jobs cascade."""
    from sqlalchemy import delete, select

    from app.models.external_call import ExternalCall
    from app.models.job import GenerationJob

    job_ids = select(GenerationJob.id).where(GenerationJob.project_id == project_id)
    db.execute(delete(ExternalCall).where(ExternalCall.job_id.in_(job_ids)))


def ensure_render_assets_ready(project) -> None:
    """Reject stale media before enqueueing a render-only job."""
    from app.services.invalidation import stale_media_message

    message = stale_media_message(project.blocks)
    if message:
        raise HTTPException(status_code=409, detail=message)


def validate_artifact_path(rel: str) -> Path:
    """Resolve a stored artifact path and enforce storage-root containment.

    Args:
        rel: Database value expected to be relative to configured storage.

    Returns:
        Normalized absolute path under ``settings.storage_root``.

    Raises:
        HTTPException: Status 400 when the resolved path escapes the storage
            root.  The caller still decides whether a missing file is 404.

    """
    settings = get_settings()
    abs_path = (settings.storage_root / rel).resolve()
    storage_root = settings.storage_root.resolve()
    if not str(abs_path).startswith(str(storage_root)):
        raise HTTPException(status_code=400, detail="不正なパスです")
    return abs_path
