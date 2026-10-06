"""Shared mutation rules for persisted project settings."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from sqlalchemy.orm import object_session

from app.models.project import Project
from app.services.invalidation import invalidate_project_settings
from app.services.settings_history import record_settings


def apply_project_settings(project: Project, updates: Mapping[str, Any], *,
                           restored_from_revision: int | None = None) -> set[str]:
    """Apply validated settings and invalidate only affected pipeline stages.

    The caller owns commit and rollback. Unknown attributes fail before any
    assignment so a malformed internal caller cannot partially mutate a row.
    """
    unknown = [field for field in updates if not hasattr(project, field)]
    if unknown:
        raise ValueError(f"unknown project settings: {sorted(unknown)}")
    changed = {field for field, value in updates.items() if getattr(project, field) != value}
    db = object_session(project)
    if db is not None:
        record_settings(db, project)
    for field in changed:
        setattr(project, field, updates[field])
    if changed:
        project.revision += 1
    # A restore is an explicit recorded action even when values already match.
    elif restored_from_revision is not None:
        project.revision += 1
    if db is not None:
        record_settings(db, project, changed, restored_from_revision)
    invalidate_project_settings(project, changed)
    return changed
