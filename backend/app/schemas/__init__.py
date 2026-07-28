"""Pydantic request, patch, and response schemas for the HTTP API.

Imports:
    Pydantic base/config/field helpers enforce type, length, enum-like, and
    unknown-field rules at the HTTP boundary.
    ``Any`` describes renderer-specific JSON plan fragments.

The schemas intentionally contain no ORM or network behavior.  Route handlers
construct them from database rows; worker/provider code consumes the validated
request values.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class HealthResponse(BaseModel):
    """Backend health flags exposed to the frontend and smoke scripts.

    Attributes:
        status: Simple health marker, currently ``"ok"``.
        version: Backend package version.
        ffmpeg_available, ffprobe_available: Executable discovery results.

    """

    status: str
    version: str
    ffmpeg_available: bool
    ffprobe_available: bool


class VoicevoxSpeaker(BaseModel):
    """Small speaker representation returned by the legacy endpoint.

    Attributes:
        speaker_id: Numeric VOICEVOX speaker identifier.
        name: Display name.
        styles: Raw style dictionaries from the engine.

    """

    speaker_id: int
    name: str
    styles: list[dict[str, Any]] = Field(default_factory=list)


class VoicevoxSpeakersResponse(BaseModel):
    """Envelope for the legacy VOICEVOX speaker-list response.

    Attributes:
        url: Engine URL queried.
        speakers: Normalized speaker entries.

    """

    url: str
    speakers: list[VoicevoxSpeaker]


class ProviderConfig(BaseModel):
    """Optional BYOK provider values accepted during project creation.

    Attributes:
        llm_api_key, image_api_key: Raw request credentials transferred to the
            process-local secret store and never returned in response schemas.
        llm_base_url, llm_model, image_base_url, image_model: Non-secret
            provider endpoint/model overrides.

    """

    model_config = ConfigDict(extra="forbid")

    llm_api_key: str | None = Field(default=None)
    llm_base_url: str | None = Field(default=None)
    llm_model: str | None = Field(default=None)
    image_api_key: str | None = Field(default=None)
    image_base_url: str | None = Field(default=None)
    image_model: str | None = Field(default=None)


class ProjectCreate(BaseModel):
    """Full project-creation payload including rendering and pacing settings.

    Attributes:
        title, source_script: Required project identity and source content.
        voicevox_*: Engine endpoint, speaker, and prosody controls.
        subtitle_*: Caption visibility, layout, colors, and wrapping controls.
        pre_margin_seconds, post_margin_seconds, min_display_seconds:
            Viewer-reading timing margins.
        narration_sentence_pause_seconds: Sentence breath inserted by planning.
        max_slides_per_block: Visual-slide cap.
        use_fake_providers: Select deterministic offline providers.
        providers: Optional in-memory BYOK configuration.

    """

    model_config = ConfigDict(extra="forbid")

    # Required project input.
    title: str = Field(min_length=1, max_length=255)
    source_script: str = Field(min_length=1, max_length=100_000)
    # VOICEVOX endpoint and synthesis controls.
    voicevox_url: str = Field(default="http://127.0.0.1:50021")
    voicevox_speaker_id: int = Field(default=1, ge=0)
    voicevox_speed_scale: float = Field(default=1.0, ge=0.5, le=2.0)
    voicevox_pitch_scale: float = Field(default=0.0, ge=-1.0, le=1.0)
    voicevox_intonation_scale: float = Field(default=1.0, ge=0.0, le=2.0)
    voicevox_volume_scale: float = Field(default=1.0, ge=0.0, le=2.0)
    # Subtitle presentation controls.
    subtitle_enabled: bool = Field(default=True)
    subtitle_font_size: int = Field(default=48, ge=16, le=120)
    subtitle_position: str = Field(default="bottom")
    subtitle_text_color: str = Field(default="#FFFFFF")
    subtitle_outline_color: str = Field(default="#000000")
    subtitle_background: bool = Field(default=True)
    subtitle_max_chars_per_line: int = Field(default=36, ge=8, le=120)
    # Display timing and visual-count controls.
    pre_margin_seconds: float = Field(default=0.15, ge=0.0, le=5.0)
    post_margin_seconds: float = Field(default=1.5, ge=0.0, le=5.0)
    min_display_seconds: float = Field(default=2.0, ge=0.5, le=10.0)
    narration_sentence_pause_seconds: float = Field(default=1.5, ge=0.0, le=5.0)
    max_slides_per_block: int = Field(default=1, ge=1, le=9)
    # Provider selection and optional request-scoped credentials.
    use_fake_providers: bool = Field(default=False)

    providers: ProviderConfig = Field(default_factory=ProviderConfig)

    @field_validator("voicevox_url")
    @classmethod
    def _validate_url(cls, value: str) -> str:
        """Reject endpoints that are not explicit HTTP(S) URLs.

        Args:
            value: Candidate VOICEVOX base URL.

        Returns:
            The unchanged URL when it begins with ``http://`` or ``https://``.

        Raises:
            ValueError: If the scheme is not explicitly HTTP(S).

        """
        if not (value.startswith("http://") or value.startswith("https://")):
            raise ValueError("VOICEVOX URL must start with http:// or https://")
        return value

    @field_validator("subtitle_position")
    @classmethod
    def _validate_position(cls, value: str) -> str:
        """Restrict subtitle placement to top, middle, or bottom.

        Args:
            value: Requested subtitle alignment.

        Returns:
            The unchanged supported value.

        Raises:
            ValueError: For any other alignment string.

        """
        if value not in {"top", "middle", "bottom"}:
            raise ValueError("subtitle_position must be top|middle|bottom")
        return value


class QuickCreate(BaseModel):
    """Paste-a-script-and-go payload.

    Attributes:
        source_script: Required script to persist and process.
        title: Optional explicit title; absent values use an LLM/baseline title.
        voicevox_url, voicevox_speaker_id: Minimal engine override.
        use_fake_providers: Offline deterministic provider switch.
        Optional pacing fields: ``None`` means retain model defaults.

    Everything except the script uses project defaults; the title is derived
    from the script by the LLM. Optional overrides exist so the UI can still
    pass a VOICEVOX endpoint/speaker without reopening the full form.

    """

    model_config = ConfigDict(extra="forbid")

    source_script: str = Field(min_length=1, max_length=100_000)
    title: str | None = Field(default=None, max_length=255)
    voicevox_url: str = Field(default="http://127.0.0.1:50021")
    voicevox_speaker_id: int = Field(default=1, ge=0)
    use_fake_providers: bool = Field(default=False)

    # Pacing. ``None`` means "leave the project default alone", so the UI can
    # send only what the user actually touched.
    voicevox_speed_scale: float | None = Field(default=None, ge=0.5, le=2.0)
    narration_sentence_pause_seconds: float | None = Field(default=None, ge=0.0, le=5.0)
    post_margin_seconds: float | None = Field(default=None, ge=0.0, le=5.0)
    subtitle_font_size: int | None = Field(default=None, ge=16, le=120)
    max_slides_per_block: int | None = Field(default=None, ge=1, le=9)

    @field_validator("voicevox_url")
    @classmethod
    def _validate_url(cls, value: str) -> str:
        """Validate the optional VOICEVOX endpoint used by quick generation.

        Args:
            value: Candidate HTTP(S) URL.

        Returns:
            The unchanged value when its scheme is accepted.

        Raises:
            ValueError: For non-HTTP(S) values.

        """
        if not (value.startswith("http://") or value.startswith("https://")):
            raise ValueError("VOICEVOX URL must start with http:// or https://")
        return value


class ProjectPatch(BaseModel):
    """Partial project settings update applied by the PATCH route.

    Every field is optional so ``model_dump(exclude_unset=True)`` can apply
    only values the client actually sent.  Extra keys are rejected.
    """

    model_config = ConfigDict(extra="forbid")

    title: str | None = None
    voicevox_url: str | None = None
    voicevox_speaker_id: int | None = Field(default=None, ge=0)
    voicevox_speed_scale: float | None = Field(default=None, ge=0.5, le=2.0)
    voicevox_pitch_scale: float | None = Field(default=None, ge=-1.0, le=1.0)
    voicevox_intonation_scale: float | None = Field(default=None, ge=0.0, le=2.0)
    voicevox_volume_scale: float | None = Field(default=None, ge=0.0, le=2.0)
    subtitle_enabled: bool | None = None
    subtitle_font_size: int | None = Field(default=None, ge=16, le=120)
    subtitle_position: str | None = None
    subtitle_text_color: str | None = None
    subtitle_outline_color: str | None = None
    subtitle_background: bool | None = None
    subtitle_max_chars_per_line: int | None = Field(default=None, ge=8, le=120)
    pre_margin_seconds: float | None = Field(default=None, ge=0.0, le=5.0)
    post_margin_seconds: float | None = Field(default=None, ge=0.0, le=5.0)
    min_display_seconds: float | None = Field(default=None, ge=0.5, le=10.0)


class ProjectSummary(BaseModel):
    """Compact project representation used by list responses.

    Attributes:
        id, title: Project identity.
        status, progress, current_stage: Aggregate pipeline state.
        block_count: Number of persisted blocks.
        output_video_path, error_message: Output/error metadata.
        created_at, updated_at: ISO-8601 timestamp strings.

    """

    id: int
    title: str
    status: str
    progress: float
    current_stage: str | None
    block_count: int
    output_video_path: str | None
    error_message: str | None
    created_at: str | None
    updated_at: str | None


class ProjectDetail(ProjectSummary):
    """Project summary plus source, pacing, subtitle, and output settings.

    Raw provider credentials are intentionally not part of this response.
    """

    source_script: str
    global_visual_style: str | None
    voicevox_url: str
    voicevox_speaker_id: int
    voicevox_speed_scale: float
    voicevox_pitch_scale: float
    voicevox_intonation_scale: float
    voicevox_volume_scale: float
    subtitle_enabled: bool
    subtitle_font_size: int
    subtitle_position: str
    subtitle_text_color: str
    subtitle_outline_color: str
    subtitle_background: bool
    subtitle_max_chars_per_line: int
    pre_margin_seconds: float
    post_margin_seconds: float
    min_display_seconds: float
    narration_sentence_pause_seconds: float = 1.5
    max_slides_per_block: int = 1
    use_fake_providers: bool
    output_subtitle_path: str | None


class BlockSummary(BaseModel):
    """Block state and API URLs for generated artifacts.

    Attributes:
        id, project_id, index: Identity and source order.
        source_text, tts_text: Display and narration text.
        visual_type, visual_plan, image_prompt: Visual metadata.
        image_url, audio_url, video_url: API artifact routes when available.
        duration_ms, display_duration_ms: Audio/display timing.
        status_*: Per-stage status values.
        error_message: Latest block error.

    """

    id: int
    project_id: int
    index: int
    source_text: str
    tts_text: str
    visual_type: str
    visual_plan: dict[str, Any] | None
    image_prompt: str | None
    image_url: str | None
    audio_url: str | None
    video_url: str | None
    duration_ms: int | None
    display_duration_ms: int | None
    status_split: str
    status_visual_plan: str
    status_image: str
    status_audio: str
    status_render: str
    error_message: str | None


class BlockPatch(BaseModel):
    """Editable block text and visual-plan fields for the PATCH route.

    ``visual_plan`` is an untyped JSON mapping at this boundary because its
    exact shape depends on the selected visual type; the renderer performs the
    corresponding normalization later.
    """

    model_config = ConfigDict(extra="forbid")

    source_text: str | None = Field(default=None, min_length=1, max_length=4000)
    tts_text: str | None = Field(default=None, min_length=1, max_length=4000)
    visual_plan: dict[str, Any] | None = None


class JobSummary(BaseModel):
    """Public progress representation of a background generation job.

    Attributes:
        id, project_id: Job and owning project identity.
        current_stage, status: Lifecycle labels.
        progress, stage_progress: Overall/current-stage fractions.
        started_at, finished_at: Optional ISO timestamps.
        error_message: Truncated failure detail.

    """

    id: int
    project_id: int
    current_stage: str
    status: str
    progress: float
    stage_progress: float
    started_at: str | None
    finished_at: str | None
    error_message: str | None


class GenerateAllResponse(BaseModel):
    """Response returned when generation or rerender is queued.

    Attributes:
        job: Newly created job summary.
        message: Short queue operation label.

    """

    job: JobSummary
    message: str = "queued"


class QuickCreateResponse(BaseModel):
    """Paste-and-go result containing the project and queued job.

    Attributes:
        project: Created project detail.
        job: Newly queued full-pipeline job.
        message: Short queue operation label.

    """

    project: ProjectDetail
    job: JobSummary
    message: str = "queued"


class SpeakerStyle(BaseModel):
    """One VOICEVOX style belonging to a speaker.

    Attributes:
        id: Numeric style identifier.
        name: Optional display name supplied by the engine.

    """

    id: int
    name: str | None = None


class SpeakerInfo(BaseModel):
    """Normalized VOICEVOX speaker and style information for the UI.

    Attributes:
        speaker_id, name: Speaker identity/display name.
        styles: Normalized style entries.

    """

    speaker_id: int
    name: str
    styles: list[SpeakerStyle] = Field(default_factory=list)


class SpeakersEnvelope(BaseModel):
    """Speaker-discovery response with the queried base URL.

    Attributes:
        url: Normalized engine endpoint used for discovery.
        speakers: Available normalized speaker/style records.

    """

    url: str
    speakers: list[SpeakerInfo]
