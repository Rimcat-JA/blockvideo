"""Allocate project identities under the caller's SQLite writer transaction."""
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.artifact import GenerationArtifact
from app.models.operation_request import OperationReceipt
from app.models.project import Project
from app.models.project_identity import ProjectIdentity
from app.models.settings_revision import SettingsRevision


def reserve_project_id(db: Session, project_id: int) -> None:
    """Keep legacy project IDs reserved after explicit deletion too."""
    if db.get(ProjectIdentity, project_id) is None:
        db.add(ProjectIdentity(id=project_id))
        db.flush()


def allocate_project_id(db: Session) -> int:
    """Never recycle a directory or a public identity referenced by durable records."""
    columns = [Project.id, ProjectIdentity.id, OperationReceipt.project_id,
               GenerationArtifact.project_id, SettingsRevision.project_id]
    project_id = max(db.scalar(select(func.max(column))) or 0 for column in columns) + 1
    reserve_project_id(db, project_id)
    return project_id
