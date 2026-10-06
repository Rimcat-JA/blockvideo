"""Declare downstream effects and combine required work into one ordered plan."""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from sqlalchemy.orm import object_session
from sqlalchemy import select

from app.core.config import get_settings
from app.models.artifact import GenerationArtifact
from app.models.block import BlockStatus
from app.models.job import GenerationJob
from app.models.project import Project
from app.services.paths import block_narration_path


STAGE_ORDER = ("split", "plan", "image", "audio", "render")
RENDER_SETTINGS = {
    "visual_focus_enabled", "subtitle_mode", "subtitle_font_size", "subtitle_position",
    "subtitle_text_color", "subtitle_outline_color", "subtitle_background",
    "subtitle_max_chars_per_line", "pre_margin_seconds", "post_margin_seconds",
    "min_display_seconds",
}
AUDIO_SETTINGS = {"narration_pacing_mode", "narration_sentence_pause_seconds",
                  "pronunciation_overrides", "max_slides_per_block"}
IMAGE_SETTINGS = {"max_slides_per_block", "subtitle_enabled"}


def dependencies_for_settings(fields: Iterable[str]) -> set[str]:
    """Existing BlockVideo setting semantics expressed as a reusable declaration."""
    changed = set(fields)
    stages: set[str] = set()
    if changed & AUDIO_SETTINGS or any(name.startswith("voicevox_") for name in changed):
        stages.add("audio")
    if changed & IMAGE_SETTINGS:
        stages.add("image")
    if stages or changed & RENDER_SETTINGS:
        stages.add("render")
    return stages


def _exists(path: str | None) -> bool:
    if not path:
        return False
    root = get_settings().storage_root.resolve()
    candidate = (root / path).resolve()
    return candidate.is_relative_to(root) and candidate.is_file() and candidate.stat().st_size > 0


def _changed_materials(project: Project) -> set[str]:
    """Detect altered formerly verified media; old untracked projects have no baseline."""
    from app.services.artifact_store import file_identity

    db = object_session(project)
    artifact_id = getattr(project, "current_artifact_id", None)
    artifact = db.get(GenerationArtifact, artifact_id) if db and artifact_id else None
    baseline: dict[str, dict[str, Any]] = {}
    if artifact is not None:
        for item in artifact.manifest_json.get("materials", []) + artifact.manifest_json.get("block_videos", []):
            baseline[item["path"]] = item
    # Successful intermediate jobs and safe restart checkpoints may be newer
    # than the last final video. Their verified files supersede older hashes.
    if db:
        recent = list(db.scalars(select(GenerationJob).where(GenerationJob.project_id == project.id)
                                .order_by(GenerationJob.id.desc()).limit(100)))
        for job in reversed(recent):
            for item in (job.plan_json or {}).get("stage_materials", []):
                baseline[item["path"]] = item
    changed: set[str] = set()
    for item in baseline.values():
        try:
            if file_identity(Path(item["path"])) != item:
                changed.add(item["path"])
        except (OSError, ValueError):
            changed.add(item["path"])
    return changed


def build_generation_plan(
    project: Project, kind: str = "full", block_index: int | None = None,
) -> dict[str, Any]:
    """Plan only stale/missing stages; a fresh full request assembles one final video."""
    if kind == "render":
        kind = "rerender"
    if kind not in {"full", "rerender", "block_visual", "block_audio"}:
        raise ValueError(f"unsupported generation kind: {kind}")
    blocks = sorted(project.blocks, key=lambda row: row.index)
    selected = blocks if block_index is None else [row for row in blocks if row.index == block_index]
    if kind.startswith("block_") and (block_index is None or not selected):
        raise ValueError("target block does not exist")
    reasons: list[str] = []
    by_block: dict[str, list[str]] = {}
    stages: set[str] = set()
    changed_materials = _changed_materials(project)
    if not blocks and kind == "full":
        stages.update(STAGE_ORDER)
        reasons.append("initial_split")
    for block in selected:
        work: set[str] = set()
        if kind == "block_audio":
            work.add("audio")
        elif kind == "block_visual":
            if block.status_visual_plan != BlockStatus.completed or block.visual_plan_json is None:
                work.add("plan")
            work.add("image")
        elif kind == "rerender":
            work.add("render")
        else:
            if block.status_visual_plan != BlockStatus.completed or block.visual_plan_json is None:
                work.add("plan")
            if block.status_image != BlockStatus.completed or not _exists(block.image_path):
                work.add("image")
            if block.status_audio != BlockStatus.completed or not _exists(block.audio_path):
                work.add("audio")
            image_dir = (get_settings().storage_root / block.image_path).parent if block.image_path else None
            for path in changed_materials:
                material = get_settings().storage_root / path
                if image_dir is not None and material.parent == image_dir and material.name.startswith("image"):
                    work.add("image")
                if path == block.audio_path or material == block_narration_path(project.id, block.index):
                    work.add("audio")
                if path == block.video_path:
                    work.add("render")
            if "plan" in work:
                work.add("image")
                if project.narration_pacing_mode == "adaptive":
                    work.add("audio")
            if work or block.status_render != BlockStatus.completed or not _exists(block.video_path):
                work.add("render")
        by_block[str(block.index)] = [stage for stage in STAGE_ORDER if stage in work]
        stages.update(work)
    if kind in {"full", "rerender"}:
        stages.add("render")
    return {
        "schema_version": 1, "kind": kind, "block_index": block_index,
        "stages": [stage for stage in STAGE_ORDER if stage in stages],
        "blocks": by_block, "reasons": reasons,
    }
