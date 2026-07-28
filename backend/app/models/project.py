"""Project ORM model."""
from __future__ import annotations

import enum
from datetime import datetime, timezone

from sqlalchemy import DateTime, Enum, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


class ProjectStatus(str, enum.Enum):
    pending = "pending"
    splitting = "splitting"
    planning = "planning"
    generating = "generating"
    rendering = "rendering"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


class Project(Base):
    __tablename__ = "projects"

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

    # Provider configuration (non-secret)
    llm_provider: Mapped[str] = mapped_column(String(32), default="openai_compatible", nullable=False)
    llm_base_url: Mapped[str | None] = mapped_column(String(512), nullable=True)
    llm_model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    image_provider: Mapped[str] = mapped_column(String(32), default="openai", nullable=False)
    image_model: Mapped[str | None] = mapped_column(String(128), nullable=True)

    # VOICEVOX config (non-secret)
    voicevox_url: Mapped[str] = mapped_column(
        String(512), default="http://127.0.0.1:50021", nullable=False
    )
    voicevox_speaker_id: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    voicevox_speed_scale: Mapped[float] = mapped_column(default=1.0, nullable=False)
    voicevox_pitch_scale: Mapped[float] = mapped_column(default=0.0, nullable=False)
    voicevox_intonation_scale: Mapped[float] = mapped_column(default=1.0, nullable=False)
    voicevox_volume_scale: Mapped[float] = mapped_column(default=1.0, nullable=False)

    # Subtitle config
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

    # Output paths (relative to storage root for safety)
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