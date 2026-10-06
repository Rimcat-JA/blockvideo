"""Save and restore non-secret configuration snapshots inside the caller's transaction."""
from __future__ import annotations

from copy import deepcopy
from datetime import timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.project import Project
from app.models.settings_revision import SettingsRevision
from app.schemas import ProjectCreate, ProjectPatch

SETTING_FIELDS = tuple(ProjectPatch.model_fields)


def configuration(project: Project) -> dict[str, Any]:
    return {name: deepcopy(getattr(project, name)) for name in SETTING_FIELDS}


def validate_settings(project: Project, updates: dict[str, Any]) -> dict[str, Any]:
    """Reuse complete project validation without accepting credentials or content edits."""
    patch = ProjectPatch.model_validate(updates, strict=True).model_dump(exclude_unset=True)
    if any(value is None for value in patch.values()):
        raise ValueError("設定を空にすることはできません")
    complete = ProjectCreate.model_validate({**configuration(project), **patch,
                                            "source_script": project.source_script,
                                            "use_fake_providers": project.use_fake_providers}, strict=True)
    normalized = complete.model_dump()
    return {field: normalized[field] for field in patch}


def record_settings(db: Session, project: Project, changed_fields: set[str] | None = None,
                    restored_from_revision: int | None = None) -> SettingsRevision:
    """Retain the first recorded configuration for each monotonically increasing revision."""
    existing = db.scalar(select(SettingsRevision).where(
        SettingsRevision.project_id == project.id, SettingsRevision.revision == project.revision))
    if existing is not None:
        return existing
    # autoflush is disabled: also consider a baseline just added in this transaction.
    for pending in db.new:
        if isinstance(pending, SettingsRevision) and (pending.project_id, pending.revision) == (project.id, project.revision):
            return pending
    row = SettingsRevision(project_id=project.id, revision=project.revision,
                           settings_json=configuration(project), changed_fields=sorted(changed_fields or []),
                           restored_from_revision=restored_from_revision)
    db.add(row)
    return row


def settings_version_summary(row: SettingsRevision) -> dict[str, Any]:
    created = row.created_at if row.created_at.tzinfo else row.created_at.replace(tzinfo=timezone.utc)
    return {"revision": row.revision, "created_at": created.isoformat(),
            "restored_from_revision": row.restored_from_revision,
            "changed_fields": row.changed_fields, "settings": row.settings_json}
