"""Map logical project artifacts to safe storage paths.

Imports:
    ``Path`` performs platform-independent path composition.
    ``get_settings`` supplies the configured storage root.

All public path functions are pure path computations except
``ensure_project_layout``.  Database-facing helpers store paths relative to
``storage_root`` where possible; API handlers must still validate them before
serving a file.
"""
from __future__ import annotations

from pathlib import Path

from app.core.config import get_settings


def project_dir(project_id: int) -> Path:
    """Return the storage directory for one project.

    Args:
        project_id: Numeric project identifier used as a zero-padded folder.

    Returns:
        ``<storage_root>/projects/<project_id:04d>``.  The directory is not
        created by this function.

    """
    return get_settings().storage_root / "projects" / f"{project_id:04d}"


def project_block_dir(project_id: int, index: int) -> Path:
    """Return one block's artifact directory without creating it.

    Args:
        project_id: Owning project identifier.
        index: Zero-based script-block index.

    Returns:
        The zero-padded ``blocks/<index>`` path under ``project_dir``.

    """
    return project_dir(project_id) / "blocks" / f"{index:04d}"


def block_image_path(project_id: int, index: int, slot: int = 0) -> Path:
    """Path for a block's slide image.

    ``slot`` 0 is the block's primary slide (``image.png``, the one the UI
    shows). Blocks whose script held several visuals get additional slides
    numbered from 1, shown in sequence during the block.

    Args:
        project_id: Owning project identifier.
        index: Zero-based block index.
        slot: Ordered slide number; ``0`` is the primary image.

    Returns:
        The canonical PNG path for the requested slide slot.  No filesystem
        operation occurs.

    """
    name = "image.png" if slot == 0 else f"image_{slot}.png"
    return project_block_dir(project_id, index) / name


def block_audio_path(project_id: int, index: int) -> Path:
    """Return the canonical WAV path for one block.

    Args:
        project_id: Owning project identifier.
        index: Zero-based block index.

    Returns:
        The path ending in ``audio.wav`` under the block directory.

    """
    return project_block_dir(project_id, index) / "audio.wav"


def block_narration_path(project_id: int, index: int) -> Path:
    """Sentence timings measured from VOICEVOX when the audio was made.

    Written by the audio stage, read by the render stage to time captions and
    slide changes against the actual voice.

    Args:
        project_id: Owning project identifier.
        index: Zero-based block index.

    Returns:
        The path ending in ``narration.json``.  The file is written by audio
        generation and read by subtitle/render stages.

    """
    return project_block_dir(project_id, index) / "narration.json"


def block_video_path(project_id: int, index: int) -> Path:
    """Return the canonical per-block MP4 path.

    Args:
        project_id: Owning project identifier.
        index: Zero-based block index.

    Returns:
        The path ending in ``video.mp4``.

    """
    return project_block_dir(project_id, index) / "video.mp4"


def block_subtitle_path(project_id: int, index: int) -> Path:
    """Return the per-block ASS path used for burn-in.

    The file's timestamps begin at zero and end at the block's own duration;
    FFmpeg burns it into the block clip before project concatenation.
    """
    return project_block_dir(project_id, index) / "subtitle.ass"


def project_subtitle_path(project_id: int) -> Path:
    """Return the whole-project ASS path with absolute timeline timestamps."""
    return project_dir(project_id) / "subtitles.ass"


def project_json_path(project_id: int) -> Path:
    """Return the human-readable project metadata JSON path."""
    return project_dir(project_id) / "project.json"


def timeline_json_path(project_id: int) -> Path:
    """Return the JSON path containing only absolute timeline rows."""
    return project_dir(project_id) / "timeline.json"


def output_video_path(project_id: int) -> Path:
    """Return the final concatenated project MP4 path."""
    return project_dir(project_id) / "output" / "video.mp4"


def concat_list_path(project_id: int) -> Path:
    """Return the temporary FFmpeg concat-demuxer list path."""
    return project_dir(project_id) / "concat.list"


def ensure_project_layout(project_id: int) -> None:
    """Create the project, block, output, and log directories if absent.

    Args:
        project_id: Project whose storage tree should be initialized.

    Side Effects:
        Creates ``projects/<id>/blocks``, ``output``, and ``logs`` with
        ``exist_ok=True``.  Existing files and directories are untouched.

    """
    pd = project_dir(project_id)
    (pd / "blocks").mkdir(parents=True, exist_ok=True)
    (pd / "output").mkdir(parents=True, exist_ok=True)
    (pd / "logs").mkdir(parents=True, exist_ok=True)


def relpath_for_db(absolute: Path) -> str:
    """Convert an artifact path to a storage-relative database value.

    Args:
        absolute: Path intended to point inside the configured storage root.

    Returns:
        POSIX-style path relative to storage when containment succeeds;
        otherwise the resolved path's POSIX string as a compatibility fallback.

    """
    settings = get_settings()
    try:
        return absolute.resolve().relative_to(settings.storage_root.resolve()).as_posix()
    except ValueError:
        return absolute.as_posix()


def abspath_from_db(rel: str) -> Path:
    """Resolve a database path value against the configured storage root.

    Args:
        rel: Stored relative or path-like artifact value.

    Returns:
        A normalized absolute ``Path``.  Callers serving files should also use
        ``app.api.utils.validate_artifact_path`` for an HTTP-safe containment
        check.

    """
    settings = get_settings()
    return (settings.storage_root / rel).resolve()
