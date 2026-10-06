"""Non-secret project configuration snapshots for explicit historical restore."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import DateTime, Integer, JSON, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


class SettingsRevision(Base):
    """A saved configuration; restoration creates a new project revision."""

    __tablename__ = "settings_revisions"
    __table_args__ = (UniqueConstraint("project_id", "revision"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    project_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    revision: Mapped[int] = mapped_column(Integer, nullable=False)
    settings_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    changed_fields: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    restored_from_revision: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
