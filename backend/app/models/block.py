"""Block ORM model — one row per script chunk."""
from __future__ import annotations

import enum
from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, Enum, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


class BlockStatus(str, enum.Enum):
    """Status values for an individual pipeline stage on a block."""

    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"
    skipped = "skipped"


class VisualType(str, enum.Enum):
    """Visual renderer selected for a block's slide plan."""

    ai_image = "ai_image"
    code_slide = "code_slide"
    # Author-drawn slide content, rendered verbatim.
    verbatim_slide = "verbatim_slide"
    diagram = "diagram"
    # Structured SICP-style diagrams drawn from a spec rather than Mermaid.
    pointer_diagram = "pointer_diagram"
    env_diagram = "env_diagram"
    formula = "formula"
    comparison = "comparison"
    title_slide = "title_slide"
    text_slide = "text_slide"


class Block(Base):
    """Persisted script block with visual, audio, and render artifacts."""

    __tablename__ = "blocks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    project_id: Mapped[int] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True
    )
    index: Mapped[int] = mapped_column(Integer, nullable=False)

    source_text: Mapped[str] = mapped_column(Text, nullable=False)
    tts_text: Mapped[str] = mapped_column(Text, nullable=False)

    visual_type: Mapped[VisualType] = mapped_column(
        Enum(VisualType, native_enum=False, length=32),
        default=VisualType.text_slide,
        nullable=False,
    )
    visual_plan_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    image_prompt: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Artifact paths (relative to project storage dir).
    image_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    audio_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    video_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    display_duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)

    status_split: Mapped[BlockStatus] = mapped_column(
        Enum(BlockStatus, native_enum=False, length=16),
        default=BlockStatus.completed,
        nullable=False,
    )
    status_visual_plan: Mapped[BlockStatus] = mapped_column(
        Enum(BlockStatus, native_enum=False, length=16),
        default=BlockStatus.pending,
        nullable=False,
    )
    status_image: Mapped[BlockStatus] = mapped_column(
        Enum(BlockStatus, native_enum=False, length=16),
        default=BlockStatus.pending,
        nullable=False,
    )
    status_audio: Mapped[BlockStatus] = mapped_column(
        Enum(BlockStatus, native_enum=False, length=16),
        default=BlockStatus.pending,
        nullable=False,
    )
    status_render: Mapped[BlockStatus] = mapped_column(
        Enum(BlockStatus, native_enum=False, length=16),
        default=BlockStatus.pending,
        nullable=False,
    )

    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    # content_hash: aggregated hash used for cache invalidation across stages.
    content_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    project: Mapped["Project"] = relationship("Project", back_populates="blocks")  # noqa: F821

    def to_summary(self) -> dict:
        """Return block state with API-relative artifact URLs."""
        return {
            "id": self.id,
            "index": self.index,
            "source_text": self.source_text,
            "tts_text": self.tts_text,
            "visual_type": self.visual_type.value if self.visual_type else None,
            "visual_plan": self.visual_plan_json,
            "image_prompt": self.image_prompt,
            "image_url": self._artifact_url("image"),
            "audio_url": self._artifact_url("audio"),
            "video_url": self._artifact_url("video"),
            "duration_ms": self.duration_ms,
            "display_duration_ms": self.display_duration_ms,
            "status_split": self.status_split.value,
            "status_visual_plan": self.status_visual_plan.value,
            "status_image": self.status_image.value,
            "status_audio": self.status_audio.value,
            "status_render": self.status_render.value,
            "error_message": self.error_message,
        }

    def _artifact_url(self, kind: str) -> str | None:
        """Build the API URL for one available block artifact kind."""
        if kind == "image" and self.image_path:
            return f"/api/projects/{self.project_id}/artifacts/image/{self.index}"
        if kind == "audio" and self.audio_path:
            return f"/api/projects/{self.project_id}/artifacts/audio/{self.index}"
        if kind == "video" and self.video_path:
            return f"/api/projects/{self.project_id}/artifacts/video/{self.index}"
        return None
