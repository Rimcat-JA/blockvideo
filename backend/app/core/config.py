"""Environment-backed settings and executable/path resolution.

Imports:
    ``os`` reads optional executable overrides.
    ``lru_cache`` keeps one ``Settings`` instance per process.
    ``Path`` represents storage, sample, and executable configuration paths.
    ``BaseSettings`` loads typed values from environment variables and the
    backend ``.env`` file.

The module-level roots are derived from this file's location.  Settings hold
non-secret defaults and optional environment-level provider credentials;
project-specific BYOK secrets belong in ``app.core.security.SecretStore``.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


# ``backend/`` directory containing the application package and ``.env``.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Customer project root containing storage and samples.
REPO_ROOT = PROJECT_ROOT.parent
# Default runtime artifact/database directory.
STORAGE_ROOT = REPO_ROOT / "storage"
# Sample scripts used by the demo and manual checks.
SAMPLES_ROOT = REPO_ROOT / "samples"


class Settings(BaseSettings):
    """Typed application settings loaded from environment and ``.env``.

    Most values are safe defaults for a local instance.  API keys are optional
    environment fallbacks and are never included in project JSON or logs;
    request/project BYOK values are kept separately by ``SecretStore``.

    Attributes:
        app_name, app_version, environment: Identity and runtime mode.
        storage_root, database_url: Local persistence locations.
        api_host, api_port, cors_origins: HTTP listener and browser origins.
        llm_* and image_*: Optional provider fallback configuration.
        voicevox_*: Default VOICEVOX synthesis controls.
        output_* and crossfade_seconds: Video frame/encoding defaults.
        splitter_* and narration_*: LLM splitting and timing concurrency.
        ffmpeg_path, ffprobe_path: Optional executable overrides.
        mermaid_*: Optional Mermaid CLI renderer configuration.
        subtitle_band_height: Pixels reserved for burned-in captions.

    """

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "BlockVideo"
    app_version: str = "0.1.0"
    environment: str = "development"

    # Storage layout
    storage_root: Path = STORAGE_ROOT
    database_url: str = f"sqlite:///{(STORAGE_ROOT / 'blockvideo.db').as_posix()}"

    # HTTP
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]

    # LLM fallbacks (env-var only)
    llm_api_key: str | None = None
    llm_base_url: str | None = None
    llm_model: str | None = None
    # Optional cheaper/faster model for the high-volume per-block planning
    # stage. Falls back to ``llm_model`` when unset.
    llm_model_planner: str | None = None
    # Bounded concurrency for the per-block visual-plan stage.
    visual_plan_concurrency: int = 6
    # OpenRouter attribution headers (optional, ignored by other providers).
    openrouter_referer: str | None = "http://localhost:5173"
    openrouter_title: str | None = "BlockVideo"

    # Image provider fallbacks (env-var only)
    image_api_key: str | None = None
    image_base_url: str | None = None
    image_model: str | None = None

    # VOICEVOX defaults
    voicevox_url: str = "http://127.0.0.1:50021"
    voicevox_speaker_id: int = 1
    voicevox_speed_scale: float = 1.0
    voicevox_pitch_scale: float = 0.0
    voicevox_intonation_scale: float = 1.0
    voicevox_volume_scale: float = 1.0
    voicevox_pre_phoneme_length: float = 0.15
    voicevox_post_phoneme_length: float = 0.35
    voicevox_min_display_seconds: float = 2.0

    # Renderer defaults
    output_width: int = 1920
    output_height: int = 1080
    output_fps: int = 30
    crossfade_seconds: float = 0.0

    # LLM splitting behavior
    splitter_min_chars: int = 60
    splitter_max_chars: int = 130
    splitter_target_chars: int = 100
    splitter_max_attempts: int = 3
    # Long scripts are cut into segments before being sent to the LLM: the
    # split prompt echoes the text back, so one call over a whole script needs
    # output tokens proportional to its length and stalls (or truncates).
    splitter_segment_chars: int = 1200
    splitter_concurrency: int = 4
    # Scripts written as documents lean on their code listings to complete
    # sentences; since code is shown rather than spoken, those blocks need
    # their narration rewritten. One extra LLM call per affected block.
    narration_repair_enabled: bool = True
    narration_repair_concurrency: int = 4

    # Narration timing. The block is planned one sentence at a time so that
    # captions and slide changes can be timed against the voice VOICEVOX is
    # actually going to produce; the same split is where the gap at each 。 is
    # widened into a breath. VOICEVOX's own gap there is ~0.4s, which reads as
    # rushed when sentences are short. This is the default for a *new* project;
    # the value used at synthesis is the project's own, editable from the UI.
    narration_sentence_pause_seconds: float = 1.5
    narration_query_concurrency: int = 4

    # ffmpeg path override
    ffmpeg_path: str | None = None
    ffprobe_path: str | None = None

    # Mermaid diagram rendering (optional). When mmdc is not available we
    # fall back to the built-in PIL diagram renderer, so these are optional.
    mermaid_mmdc_path: str | None = None
    mermaid_puppeteer_config: Path | None = None
    mermaid_render_timeout: float = 90.0

    # Subtitle burn-in band layout: the slide is scaled into the upper
    # (height - band) region and the lower band hosts burned-in subtitles.
    subtitle_band_height: int = 200


@lru_cache
def get_settings() -> Settings:
    """Load the cached settings and ensure the runtime directories exist.

    Returns:
        The process-wide ``Settings`` instance.  Environment variables are
        read only when the cache is first populated.

    Side Effects:
        Creates ``storage_root`` and its ``projects`` child when absent.

    """
    settings = Settings()
    settings.storage_root.mkdir(parents=True, exist_ok=True)
    (settings.storage_root / "projects").mkdir(parents=True, exist_ok=True)
    return settings


def reset_settings_cache() -> None:
    """Clear the settings cache so the next call rereads configuration.

    This function is primarily a test seam for changing environment variables
    between cases; it does not delete storage directories.
    """
    get_settings.cache_clear()


def resolve_ffmpeg(settings: Settings | None = None) -> str:
    """Resolve the FFmpeg executable from settings or ``FFMPEG_PATH``.

    Args:
        settings: Optional already-loaded settings object.

    Returns:
        Explicit ``Settings.ffmpeg_path``, then the ``FFMPEG_PATH`` environment
        value, otherwise the bare ``"ffmpeg"`` name for PATH lookup.

    """
    settings = settings or get_settings()
    if settings.ffmpeg_path:
        return settings.ffmpeg_path
    found = os.environ.get("FFMPEG_PATH")
    if found:
        return found
    # default to PATH lookup
    return "ffmpeg"


def resolve_ffprobe(settings: Settings | None = None) -> str:
    """Resolve FFprobe using settings, ``FFPROBE_PATH``, then PATH lookup.

    Args:
        settings: Optional already-loaded settings object.

    Returns:
        The configured executable name or path, defaulting to ``"ffprobe"``.

    """
    settings = settings or get_settings()
    if settings.ffprobe_path:
        return settings.ffprobe_path
    found = os.environ.get("FFPROBE_PATH")
    if found:
        return found
    return "ffprobe"


def resolve_mmdc(settings: Settings | None = None) -> str:
    """Resolve the mermaid-cli (mmdc) executable path.

    Args:
        settings: Optional already-loaded settings object.

    Returns:
        Explicit ``mermaid_mmdc_path`` -> ``MERMAID_MMDC_PATH`` -> the bare
        ``"mmdc"`` executable name.  The renderer itself falls back to
        ``npx --yes mmdc`` when that bare name is not on PATH.

    """
    settings = settings or get_settings()
    if settings.mermaid_mmdc_path:
        return str(settings.mermaid_mmdc_path)
    found = os.environ.get("MERMAID_MMDC_PATH")
    if found:
        return found
    return "mmdc"
