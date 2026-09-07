"""Common HTTP-boundary safety helpers.

Imports:
    ``Path`` resolves stored artifact values.
    ``HTTPException`` communicates invalid user-controlled paths as HTTP 400.
    ``get_settings`` supplies the trusted storage root.
"""
from __future__ import annotations

from pathlib import Path

from fastapi import HTTPException

from app.core.config import get_settings


def ensure_project_idle(project_id: int, db) -> None:
    """Reject concurrent writes to one project's output in this local process."""
    from sqlalchemy import select
    from app.models.job import GenerationJob
    from app.workers.job_runner import job_registry

    ids = db.scalars(select(GenerationJob.id).where(
        GenerationJob.project_id == project_id,
        GenerationJob.status.in_(["pending", "running"]),
    ))
    if any(job_registry.is_running(job_id) for job_id in ids):
        raise HTTPException(status_code=409, detail="このプロジェクトは生成中です。完了後に再実行してください。")


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
