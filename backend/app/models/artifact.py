"""Immutable successful final-video history, separate from mutable working media."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import JSON, DateTime, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


class GenerationArtifact(Base):
    """A verified final video; legacy imports deliberately have unknown revision."""

    __tablename__ = "generation_artifacts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    project_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    job_id: Mapped[int | None] = mapped_column(Integer, unique=True, nullable=True)
    revision: Mapped[int | None] = mapped_column(Integer, nullable=True)
    input_fingerprint: Mapped[str | None] = mapped_column(String(64), nullable=True)
    video_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    subtitle_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    manifest_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False,
    )
