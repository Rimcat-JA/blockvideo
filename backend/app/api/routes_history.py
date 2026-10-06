"""Read-only project history and verified, project-owned artifact downloads."""
from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import get_db
from app.models.artifact import GenerationArtifact
from app.models.job import GenerationJob
from app.models.project import Project
from app.models.settings_revision import SettingsRevision
from app.services.artifact_store import artifact_file_path, artifact_to_summary, file_identity
from app.services.job_views import build_recovery_contexts, job_summary, utc_timestamp
from app.services.settings_history import settings_version_summary

router = APIRouter(prefix="/projects", tags=["history"])


def _project(db: Session, project_id: int) -> Project:
    project = db.get(Project, project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="project not found")
    return project


def _output_state(project: Project, artifacts: list[dict[str, Any]]) -> str:
    if project.current_artifact_id is not None:
        current = next((artifact for artifact in artifacts if artifact["id"] == project.current_artifact_id), None)
        if current is None or not current["available"]:
            return "missing"
        return "current" if current["is_current"] else "stale"
    if artifacts:
        return "stale" if any(artifact["available"] for artifact in artifacts) else "missing"
    if project.output_video_path:
        try:
            file_identity(project.output_video_path)
            return "stale"
        except (OSError, ValueError):
            return "missing"
    return "none"


@router.get("/{project_id}/history")
def project_history(project_id: int, db: Session = Depends(get_db)) -> dict[str, Any]:
    """Read recorded history only; polling never manufactures a settings version."""
    project = _project(db, project_id)
    artifacts = list(db.scalars(select(GenerationArtifact).where(
        GenerationArtifact.project_id == project_id,
    ).order_by(GenerationArtifact.created_at.desc(), GenerationArtifact.id.desc())))
    summaries = [
        {**artifact_to_summary(artifact, project), "created_at": utc_timestamp(artifact.created_at)}
        for artifact in artifacts
    ]
    versions = db.scalars(select(SettingsRevision).where(
        SettingsRevision.project_id == project_id,
    ).order_by(SettingsRevision.revision.desc())).all()
    jobs = db.scalars(select(GenerationJob).where(
        GenerationJob.project_id == project_id,
    ).order_by(GenerationJob.id.desc()).limit(100)).all()
    recovery_contexts = build_recovery_contexts(db, [project_id])
    recovery_context = recovery_contexts.get(project_id)
    return {
        "revision": project.revision,
        "output_state": _output_state(project, summaries),
        "current_artifact_id": project.current_artifact_id,
        "artifacts": summaries,
        "settings_versions": [
            {**settings_version_summary(version), "created_at": utc_timestamp(version.created_at)}
            for version in versions
        ],
        "jobs": [job_summary(job, recovery_context).model_dump() for job in jobs],
    }


@router.get("/{project_id}/history/artifacts/{artifact_id}/{kind}")
def download_history_artifact(
    project_id: int, artifact_id: int, kind: Literal["video", "subtitle"],
    db: Session = Depends(get_db),
) -> FileResponse:
    """Check ownership, storage containment and recorded content before serving."""
    _project(db, project_id)
    artifact = db.get(GenerationArtifact, artifact_id)
    if artifact is None or artifact.project_id != project_id:
        raise HTTPException(status_code=404, detail="artifact not found")
    try:
        path = artifact_file_path(artifact, kind)
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=404, detail="artifact file is missing or no longer matches its recorded content") from exc
    is_video = kind == "video"
    return FileResponse(
        path,
        media_type="video/mp4" if is_video else "text/x-ssa",
        filename=f"blockvideo_{project_id}_version_{artifact.id}.{'mp4' if is_video else 'ass'}",
    )
