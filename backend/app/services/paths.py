"""Project layout helpers — maps logical artifacts to filesystem paths."""
from __future__ import annotations

from pathlib import Path

from app.core.config import get_settings


def project_dir(project_id: int) -> Path:
    return get_settings().storage_root / "projects" / f"{project_id:04d}"


def project_block_dir(project_id: int, index: int) -> Path:
    return project_dir(project_id) / "blocks" / f"{index:04d}"


def block_image_path(project_id: int, index: int, slot: int = 0) -> Path:
    """Path for a block's slide image.

    ``slot`` 0 is the block's primary slide (``image.png``, the one the UI
    shows). Blocks whose script held several visuals get additional slides
    numbered from 1, shown in sequence during the block.
    """
    name = "image.png" if slot == 0 else f"image_{slot}.png"
    return project_block_dir(project_id, index) / name


def block_audio_path(project_id: int, index: int) -> Path:
    return project_block_dir(project_id, index) / "audio.wav"


def block_narration_path(project_id: int, index: int) -> Path:
    """Sentence timings measured from VOICEVOX when the audio was made.

    Written by the audio stage, read by the render stage to time captions and
    slide changes against the actual voice.
    """
    return project_block_dir(project_id, index) / "narration.json"


def block_video_path(project_id: int, index: int) -> Path:
    return project_block_dir(project_id, index) / "video.mp4"


def block_subtitle_path(project_id: int, index: int) -> Path:
    """Per-block .ass used for burn-in (start=0, end=block duration)."""
    return project_block_dir(project_id, index) / "subtitle.ass"


def project_subtitle_path(project_id: int) -> Path:
    """Whole-project .ass with absolute timeline (for external players)."""
    return project_dir(project_id) / "subtitles.ass"


def project_json_path(project_id: int) -> Path:
    return project_dir(project_id) / "project.json"


def timeline_json_path(project_id: int) -> Path:
    return project_dir(project_id) / "timeline.json"


def output_video_path(project_id: int) -> Path:
    return project_dir(project_id) / "output" / "video.mp4"


def concat_list_path(project_id: int) -> Path:
    return project_dir(project_id) / "concat.list"


def ensure_project_layout(project_id: int) -> None:
    pd = project_dir(project_id)
    (pd / "blocks").mkdir(parents=True, exist_ok=True)
    (pd / "output").mkdir(parents=True, exist_ok=True)
    (pd / "logs").mkdir(parents=True, exist_ok=True)


def relpath_for_db(absolute: Path) -> str:
    settings = get_settings()
    try:
        return absolute.resolve().relative_to(settings.storage_root.resolve()).as_posix()
    except ValueError:
        return absolute.as_posix()


def abspath_from_db(rel: str) -> Path:
    settings = get_settings()
    return (settings.storage_root / rel).resolve()