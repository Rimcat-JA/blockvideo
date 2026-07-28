"""ORM representation of a video project and its persisted configuration.

Imports:
    ``enum`` defines string-valued lifecycle states.
    ``datetime`` records UTC creation/update timestamps.
    SQLAlchemy types declare SQLite columns and parent relationships.

The project row stores user-editable settings and aggregate progress.  Raw API
keys are deliberately absent; they live in the process-local secret store.
"""
from __future__ import annotations

import enum
from datetime import datetime, timezone

from sqlalchemy import DateTime, Enum, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


class ProjectStatus(str, enum.Enum):
    """String values exposed while a project moves through the pipeline.

    ``pending`` is the initial state; ``splitting``, ``planning``,
    ``generating``, and ``rendering`` identify active stages; terminal states
    are ``completed``, ``failed``, and ``cancelled``.
    """

    pending = "pending"
    splitting = "splitting"
    planning = "planning"
    generating = "generating"
    rendering = "rendering"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


class Project(Base):
    """Persisted request, settings, progress, relationships, and output paths.

    Attributes:
        id: Database primary key and storage-directory identifier.
        title, source_script, global_visual_style: User input and project-wide
            visual context.
        status, progress, current_stage: Aggregate pipeline state shown by the
            frontend while a worker runs.
        llm_* and image_*: Non-secret provider names, endpoints, and models.
        voicevox_*: Voice engine endpoint and synthesis tuning.
        subtitle_*: Caption visibility, layout, colors, and wrapping settings.
        narration_sentence_pause_seconds: Extra pause inserted between spoken
            sentences by VOICEVOX planning.
        max_slides_per_block: Upper bound on visual slides displayed for one
            block.
        pre_margin_seconds, post_margin_seconds, min_display_seconds: Timing
            margins used to derive display duration from audio duration.
        output_*: Storage-relative paths to final video and project subtitles.
        use_fake_providers: Select deterministic offline providers.
        error_message: Truncated user-visible failure information.
        created_at, updated_at: UTC persistence timestamps.
        blocks, jobs: Cascading child relationships ordered or queried by the
            associated pipeline endpoints.

    """

    __tablename__ = "projects"

    # Identity and original user content.
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    source_script: Mapped[str] = mapped_column(Text, nullable=False)
    global_visual_style: Mapped[str | None] = mapped_column(Text, nullable=True)

    status: Mapped[ProjectStatus] = mapped_column(
        Enum(ProjectStatus, native_enum=False, length=32),
        default=ProjectStatus.pending,
        nullable=False,
    )
    progress: Mapped[float] = mapped_column(default=0.0, nullable=False)
    current_stage: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # Provider configuration (non-secret; raw API keys are never ORM fields).
    llm_provider: Mapped[str] = mapped_column(String(32), default="openai_compatible", nullable=False)
    llm_base_url: Mapped[str | None] = mapped_column(String(512), nullable=True)
    llm_model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    image_provider: Mapped[str] = mapped_column(String(32), default="openai", nullable=False)
    image_model: Mapped[str | None] = mapped_column(String(128), nullable=True)

    # VOICEVOX config copied into each project's editable settings.
    voicevox_url: Mapped[str] = mapped_column(
        String(512), default="http://127.0.0.1:50021", nullable=False
    )
    voicevox_speaker_id: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    voicevox_speed_scale: Mapped[float] = mapped_column(default=1.0, nullable=False)
    voicevox_pitch_scale: Mapped[float] = mapped_column(default=0.0, nullable=False)
    voicevox_intonation_scale: Mapped[float] = mapped_column(default=1.0, nullable=False)
    voicevox_volume_scale: Mapped[float] = mapped_column(default=1.0, nullable=False)

    # Subtitle appearance and wrapping configuration.
    subtitle_enabled: Mapped[bool] = mapped_column(default=True, nullable=False)
    subtitle_font_size: Mapped[int] = mapped_column(default=48, nullable=False)
    subtitle_position: Mapped[str] = mapped_column(default="bottom", nullable=False)
    subtitle_text_color: Mapped[str] = mapped_column(default="#FFFFFF", nullable=False)
    subtitle_outline_color: Mapped[str] = mapped_column(default="#000000", nullable=False)
    subtitle_background: Mapped[bool] = mapped_column(default=True, nullable=False)
    subtitle_max_chars_per_line: Mapped[int] = mapped_column(default=36, nullable=False)

    # Pacing. The silence held at each 。 — VOICEVOX's own gap is about 0.4s,
    # which reads as rushed. Lives on the project rather than in server config
    # because it is a taste the viewer feels directly and wants to tune per
    # video. 0 keeps VOICEVOX's default.
    narration_sentence_pause_seconds: Mapped[float] = mapped_column(
        default=1.5, nullable=False, server_default="1.5"
    )
    # How many slides one block may show. A block's script often holds more
    # visuals than there is time to take in — six listings under forty seconds
    # of narration is six seconds each, which is not long enough to read code.
    # The cap is applied when the slides are *drawn*, so raising it needs the
    # image stage re-run, not just a re-render.
    max_slides_per_block: Mapped[int] = mapped_column(
        default=1, nullable=False, server_default="1"
    )

    # Display time margins
    pre_margin_seconds: Mapped[float] = mapped_column(default=0.15, nullable=False)
    # Trailing hold after the narration ends, before the next block. Gives the
    # viewer a beat to read the slide instead of cutting on the last syllable.
    post_margin_seconds: Mapped[float] = mapped_column(default=1.5, nullable=False)
    min_display_seconds: Mapped[float] = mapped_column(default=2.0, nullable=False)

    # Storage-relative artifact paths; API routes validate before serving them.
    output_video_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    output_subtitle_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)

    # Provider-mode override (allows Fake for demos)
    use_fake_providers: Mapped[bool] = mapped_column(default=False, nullable=False)

    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    blocks: Mapped[list["Block"]] = relationship(  # noqa: F821
        "Block", back_populates="project", cascade="all, delete-orphan", order_by="Block.index"
    )
    jobs: Mapped[list["GenerationJob"]] = relationship(  # noqa: F821
        "GenerationJob", back_populates="project", cascade="all, delete-orphan"
    )

    def to_summary(self) -> dict:
        """Return the compact state view used by internal callers.

        Returns:
            A JSON-compatible mapping containing identity, lifecycle progress,
            timestamps, block count, final video path, and error text.  It does
            not include source text, provider settings, or secret material.

        """
        return {
            "id": self.id,
            "title": self.title,
            "status": self.status.value,
            "progress": self.progress,
            "current_stage": self.current_stage,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "block_count": len(self.blocks) if self.blocks else 0,
            "output_video_path": self.output_video_path,
            "error_message": self.error_message,
        }
