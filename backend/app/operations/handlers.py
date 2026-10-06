"""BlockVideo-specific handlers exposed only through the operation registry."""
from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from app.models.project import Project
from app.operations.contracts import OperationResult
from app.operations.readiness import project_state_revision
from app.services.project_settings import apply_project_settings


def set_subtitle_font_size(
    db: Session, project: Project, arguments: dict[str, Any]
) -> OperationResult:
    """Apply a resolved absolute size; the service owns the transaction."""
    changed = apply_project_settings(project, {"subtitle_font_size": arguments["value"]})
    db.flush()
    return OperationResult(
        operation_id="project.subtitle-font-size.set",
        project_id=project.id,
        changed=bool(changed),
        state_revision=project_state_revision(project),
        revision=project.revision,
        data={"subtitle_font_size": project.subtitle_font_size},
    )


def get_project_status(
    _db: Session, project: Project, _arguments: dict[str, Any]
) -> OperationResult:
    """Return current non-secret project state without writing."""
    return OperationResult(
        operation_id="project.status.get",
        project_id=project.id,
        changed=False,
        state_revision=project_state_revision(project),
        revision=project.revision,
        data={
            "status": project.status.value,
            "progress": project.progress,
            "current_stage": project.current_stage,
            "block_count": len(project.blocks),
            "output_video_path": project.output_video_path,
            "error_message": project.error_message,
            "subtitle_font_size": project.subtitle_font_size,
        },
    )
