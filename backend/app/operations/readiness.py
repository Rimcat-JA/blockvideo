"""Resolve project targets and evaluate current execution readiness."""
from __future__ import annotations

from sqlalchemy.orm import Session

from app.models.project import Project
from app.operations.contracts import (
    OperationDefinition,
    OperationTarget,
    Readiness,
    ReadinessResult,
)
from app.services.job_liveness import is_running
from app.services.job_records import has_active_project_job


def project_state_revision(project: Project) -> str:
    """Keep the string token field compatible while using a durable revision."""
    return str(project.revision)


def resolve_project(db: Session, target: OperationTarget) -> Project | None:
    """Load the single unambiguous target project."""
    project_id = target.resolved_id
    return db.get(Project, project_id) if project_id is not None else None


def has_live_job(db: Session, project_id: int) -> bool:
    """Return whether durable intent or a live process task blocks mutation."""
    return has_active_project_job(db, project_id, is_running)


def evaluate_readiness(
    db: Session,
    definition: OperationDefinition,
    target: OperationTarget,
) -> ReadinessResult:
    """Evaluate target existence and live-state constraints for an operation."""
    project_id = target.resolved_id
    if project_id is None:
        return ReadinessResult(
            operation_id=definition.operation_id,
            readiness=Readiness.needs_input,
            reason_code="target_required",
            missing_fields=["project_id"],
        )
    project = resolve_project(db, target)
    if project is None:
        return ReadinessResult(
            operation_id=definition.operation_id,
            readiness=Readiness.unsupported,
            reason_code="target_not_found",
            project_id=project_id,
        )
    if definition.precondition_key == "project_editable" and has_live_job(db, project_id):
        return ReadinessResult(
            operation_id=definition.operation_id,
            readiness=Readiness.blocked,
            reason_code="project_busy",
            project_id=project_id,
            state_revision=project_state_revision(project),
            revision=project.revision,
        )
    return ReadinessResult(
        operation_id=definition.operation_id,
        readiness=Readiness.ready,
        project_id=project_id,
        state_revision=project_state_revision(project),
        revision=project.revision,
    )
