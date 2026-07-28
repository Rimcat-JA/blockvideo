"""ORM representation of one script chunk and its generated artifacts.

Imports:
    ``enum`` supplies persisted status and visual-type values.
    ``datetime`` records UTC timestamps.
    SQLAlchemy types declare the project foreign key, JSON plan, paths, and
    per-stage status columns.

Each block belongs to one project and progresses through split, visual-plan,
image, audio, and render stages.  Artifact paths are storage-relative strings,
not arbitrary paths supplied directly to HTTP responses.
"""
from __future__ import annotations

import enum
from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, Enum, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


class BlockStatus(str, enum.Enum):
    """Status values for an individual block pipeline stage."""

    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"
    skipped = "skipped"


class VisualType(str, enum.Enum):
    """Renderer selector persisted with a block's visual plan.

    The values distinguish remote ``ai_image`` output from locally rendered
    code, diagram, formula, comparison, title, and text slides.  Structured
    ``pointer_diagram`` and ``env_diagram`` values are drawn by PIL.
    """

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
    """Persisted script chunk, plan, artifacts, and per-stage statuses.

    Attributes:
        id, project_id, index: Database identity, owning project, and source
            order.
        source_text, tts_text: Display/source text and narration text.
        visual_type, visual_plan_json, image_prompt: Planned visual metadata.
        image_path, audio_path, video_path: Storage-relative artifact paths.
        duration_ms, display_duration_ms: Audio duration and duration including
            viewer-reading margins.
        status_split, status_visual_plan, status_image, status_audio,
            status_render: Independent stage state values.
        error_message: Last block-level failure message.
        content_hash: Short cache/invalidation hash for visual-plan inputs.
        created_at, updated_at: UTC persistence timestamps.
        project: Parent relationship used by SQLAlchemy.

    """

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

    # Artifact paths relative to the configured storage root.
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

    # Aggregated input hash used for visual-stage cache invalidation.
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
        """Return block state with API-relative artifact URLs.

        Returns:
            A JSON-compatible mapping containing source/narration text, visual
            metadata, artifact URLs when paths exist, durations, every stage
            status, and the latest error message.

        """
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
        """Build the API URL for one available artifact kind.

        Args:
            kind: One of ``image``, ``audio``, or ``video``.

        Returns:
            The project/block artifact route when the corresponding database
            path is present; otherwise ``None``.  Unknown kinds also return
            ``None`` rather than constructing an unvalidated URL.

        """
        if kind == "image" and self.image_path:
            return f"/api/projects/{self.project_id}/artifacts/image/{self.index}"
        if kind == "audio" and self.audio_path:
            return f"/api/projects/{self.project_id}/artifacts/audio/{self.index}"
        if kind == "video" and self.video_path:
            return f"/api/projects/{self.project_id}/artifacts/video/{self.index}"
        return None
