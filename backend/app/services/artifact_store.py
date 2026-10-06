"""Verify immutable final videos and publish history under the job's writer lock."""
from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from sqlalchemy import select

from app.core.config import get_settings, resolve_ffprobe
from app.db import get_session_factory
from app.models.artifact import GenerationArtifact
from app.models.block import BlockStatus
from app.models.job import GenerationJob, JobStatus
from app.models.project import Project, ProjectStatus
from app.services.generation_snapshots import (
    GenerationCancelled, StaleGenerationInput, capture_inputs, fingerprint_inputs,
)
from app.services.paths import block_image_path, block_narration_path, project_dir, relpath_for_db
from app.services.transactions import atomic_write


def safe_storage_path(path: Path | str) -> Path:
    """Resolve only files within storage, including proper path component boundaries."""
    root = get_settings().storage_root.resolve()
    candidate = (root / path).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError("artifact path is outside storage")
    return candidate


def file_identity(path: Path | str) -> dict[str, Any]:
    """Measure real file content, never infer validity merely from a saved path."""
    candidate = safe_storage_path(path)
    if not candidate.is_file() or candidate.stat().st_size <= 0:
        raise FileNotFoundError("artifact file is missing or empty")
    digest = hashlib.sha256()
    with candidate.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": relpath_for_db(candidate), "size": candidate.stat().st_size,
            "sha256": digest.hexdigest()}


def collect_render_materials(project: Project) -> list[dict[str, Any]]:
    """Freeze the actual image/audio/timing inputs consumed by the render stage."""
    files: set[Path] = set()
    for block in project.blocks:
        if not block.image_path or not block.audio_path:
            raise FileNotFoundError("render inputs are missing")
        files.update({safe_storage_path(block.image_path), safe_storage_path(block.audio_path)})
        for slot in range(1, 9):
            extra = block_image_path(project.id, block.index, slot)
            if not extra.exists():
                break
            files.add(extra)
        timing = block_narration_path(project.id, block.index)
        if timing.exists():
            files.add(timing)
    return [file_identity(path) for path in sorted(files)]


def collect_completed_materials(project: Project) -> list[dict[str, Any]]:
    """Checkpoint verified stage files, including intermediate-only job results."""
    paths: set[Path] = set()
    for block in project.blocks:
        if block.status_image == BlockStatus.completed and block.image_path:
            paths.add(safe_storage_path(block.image_path))
            for slot in range(1, 9):
                extra = block_image_path(project.id, block.index, slot)
                if not extra.exists():
                    break
                paths.add(extra)
        if block.status_audio == BlockStatus.completed and block.audio_path:
            paths.add(safe_storage_path(block.audio_path))
            timing = block_narration_path(project.id, block.index)
            if timing.exists():
                paths.add(timing)
        if block.status_render == BlockStatus.completed and block.video_path:
            paths.add(safe_storage_path(block.video_path))
    # Missing completed files are repaired by the planner. Once a present file
    # is recorded here, any later disappearance fails the runner's guard.
    return [file_identity(path) for path in sorted(paths) if path.is_file() and path.stat().st_size > 0]


def verify_materials(materials: list[dict[str, Any]]) -> None:
    """Fail closed if a renderer's referenced material disappeared or changed."""
    for expected in materials:
        try:
            if file_identity(expected["path"]) != expected:
                raise StaleGenerationInput("生成中に参照素材が変更されました")
        except (OSError, ValueError) as exc:
            raise StaleGenerationInput("生成中に参照素材が欠損しました") from exc


async def validate_video(path: Path) -> dict[str, Any]:
    """Probe a real nonempty video stream and duration before declaring success."""
    identity = file_identity(path)
    proc = await asyncio.create_subprocess_exec(
        resolve_ffprobe(), "-v", "error", "-show_entries",
        "format=duration:stream=codec_type,width,height", "-of", "json", str(path),
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    stdout, _stderr = await proc.communicate()
    if proc.returncode:
        raise RuntimeError("完成動画の検証に失敗しました")
    payload = json.loads(stdout.decode("utf-8"))
    duration = float(payload.get("format", {}).get("duration", 0))
    videos = [stream for stream in payload.get("streams", [])
              if stream.get("codec_type") == "video" and stream.get("width", 0) > 0
              and stream.get("height", 0) > 0]
    if not videos or not 0 < duration < float("inf"):
        raise RuntimeError("完成動画に有効な映像または再生時間がありません")
    if file_identity(path) != identity:
        raise StaleGenerationInput("検証中に完成動画が変更されました")
    return {**identity, "duration_ms": round(duration * 1000),
            "width": videos[0]["width"], "height": videos[0]["height"]}


def artifact_file_path(artifact: GenerationArtifact, kind: str = "video") -> Path:
    """Return an existing verified file for a history download endpoint."""
    if kind not in {"video", "subtitle"}:
        raise ValueError("unsupported artifact kind")
    value = artifact.video_path if kind == "video" else artifact.subtitle_path
    if not value:
        raise FileNotFoundError("artifact does not have this file")
    path = safe_storage_path(value)
    identity = file_identity(path)
    manifest = artifact.manifest_json
    expected = manifest.get(kind) if isinstance(manifest, dict) else None
    if (not isinstance(expected, dict) or not isinstance(expected.get("path"), str)
            or type(expected.get("size")) is not int or expected["size"] <= 0
            or not isinstance(expected.get("sha256"), str) or len(expected["sha256"]) != 64
            or any(char not in "0123456789abcdef" for char in expected["sha256"])):
        raise FileNotFoundError("artifact has no valid recorded content checksum")
    if any(identity[key] != expected.get(key) for key in ("path", "size", "sha256")):
        raise FileNotFoundError("artifact file no longer matches its verified content")
    return path


def artifact_is_available(artifact: GenerationArtifact) -> bool:
    """Check history media without an external process on each page refresh."""
    try:
        artifact_file_path(artifact)
        return True
    except (OSError, ValueError):
        return False


def artifact_to_summary(artifact: GenerationArtifact, project: Project | None = None) -> dict[str, Any]:
    """Expose history identity, availability and whether settings still match."""
    available = artifact_is_available(artifact)
    current = bool(project is not None and artifact.id == project.current_artifact_id
                   and artifact.revision == project.revision
                   and artifact.input_fingerprint == fingerprint_inputs(capture_inputs(project))
                   and all(block.status_render == BlockStatus.completed for block in project.blocks)
                   and available)
    prefix = f"/api/projects/{artifact.project_id}/history/artifacts/{artifact.id}"
    return {
        "id": artifact.id, "project_id": artifact.project_id, "job_id": artifact.job_id,
        "revision": artifact.revision, "created_at": artifact.created_at.isoformat(),
        "available": available, "is_current": current,
        "video_url": f"{prefix}/video", "subtitle_url": f"{prefix}/subtitle" if artifact.subtitle_path else None,
    }


async def preserve_legacy_video(project_id: int) -> None:
    """Import one pre-history video without claiming an unrecorded settings revision."""
    with get_session_factory()() as db:
        project = db.get(Project, project_id)
        if project is None or project.current_artifact_id or not project.output_video_path:
            return
        original_path = project.output_video_path
        source = safe_storage_path(original_path)
        subtitle_source = project.output_subtitle_path
    try:
        video = await validate_video(source)
    except (OSError, ValueError, RuntimeError):
        # An unverifiable legacy file remains untouched, but is not labelled a success.
        return
    directory = project_dir(project_id) / "history" / f"legacy-{uuid4().hex}"
    directory.mkdir(parents=True, exist_ok=False)
    destination = directory / "video.mp4"
    shutil.copy2(source, destination)
    if file_identity(destination)["sha256"] != video["sha256"]:
        raise StaleGenerationInput("既存動画が履歴へのコピー中に変更されました")
    video = {**video, **file_identity(destination)}
    subtitle: dict[str, Any] | None = None
    if subtitle_source:
        try:
            old_subtitle = safe_storage_path(subtitle_source)
            file_identity(old_subtitle)
            shutil.copy2(old_subtitle, directory / "subtitles.ass")
            subtitle = file_identity(directory / "subtitles.ass")
        except (OSError, ValueError):
            pass
    with get_session_factory()() as db, atomic_write(db):
        project = db.get(Project, project_id)
        if project is None or project.current_artifact_id or project.output_video_path != original_path:
            return
        # A copy racing with a file replacement must not receive a false success label.
        if file_identity(source)["sha256"] != video["sha256"]:
            raise StaleGenerationInput("既存動画が履歴への保存中に変更されました")
        artifact = GenerationArtifact(
            project_id=project_id, video_path=video["path"],
            subtitle_path=subtitle["path"] if subtitle else None,
            manifest_json={"schema_version": 1, "legacy": True, "video": video, "subtitle": subtitle},
        )
        db.add(artifact)
        db.flush()
        project.current_artifact_id = artifact.id
        project.output_video_path = artifact.video_path
        project.output_subtitle_path = artifact.subtitle_path


def _destination_is_referenced(db: Any, job_id: int, final_path: str) -> bool:
    artifact_reference = db.scalar(
        select(GenerationArtifact.id).where(
            (GenerationArtifact.video_path == final_path)
            | (GenerationArtifact.job_id == job_id)
        ).limit(1)
    )
    current_artifact_ids = select(GenerationArtifact.id).where(
        GenerationArtifact.video_path == final_path
    )
    project_reference = db.scalar(
        select(Project.id).where(
            (Project.output_video_path == final_path)
            | (Project.current_artifact_id.in_(current_artifact_ids))
        ).limit(1)
    )
    return artifact_reference is not None or project_reference is not None


async def publish_artifact(
    job_id: int, candidate: Path, subtitle: Path | None, *,
    settled_inputs: dict[str, Any], materials: list[dict[str, Any]],
    cancel_check: Callable[[], bool],
    block_videos: list[dict[str, Any]] | None = None,
) -> GenerationArtifact:
    """Publish immutable success history; promote only the exact current input state.

    Completion and cancellation compete for the same SQLite writer boundary.
    A new verified candidate may replace only the same pending job's exact,
    unreferenced history filename after a rename-before-commit crash. Referenced
    successes are never replaced or deleted.
    """
    final = candidate.with_name("video.mp4")
    with get_session_factory()() as db:
        existing = db.scalar(
            select(GenerationArtifact).where(GenerationArtifact.job_id == job_id)
        )
        if existing is not None:
            if final.exists() and not _destination_is_referenced(
                db, job_id, relpath_for_db(final)
            ):
                raise RuntimeError("このジョブの確定動画の参照を確認できません")
            db.expunge(existing)
            return existing
    if cancel_check():
        raise GenerationCancelled("ユーザーによりキャンセルされました")
    video = await validate_video(candidate)
    if any(file_identity(candidate)[key] != video[key] for key in ("path", "size", "sha256")):
        raise StaleGenerationInput("検証後に完成動画が変更されました")
    verify_materials(materials)
    verify_materials(block_videos or [])
    subtitle_identity = file_identity(subtitle) if subtitle else None
    settled_fingerprint = fingerprint_inputs(settled_inputs)
    with get_session_factory()() as db, atomic_write(db):
        job = db.get(GenerationJob, job_id)
        if job is None:
            raise StaleGenerationInput("生成ジョブが見つかりません")
        existing = db.scalar(select(GenerationArtifact).where(GenerationArtifact.job_id == job_id))
        if existing is not None:
            db.expunge(existing)
            return existing
        if job.cancel_requested or cancel_check() or job.status == JobStatus.cancelled:
            raise GenerationCancelled("ユーザーによりキャンセルされました")
        if job.status != JobStatus.running:
            raise StaleGenerationInput("生成ジョブは実行中ではありません")
        from app.services.external_calls import ExternalOutcomeUnknown, has_unresolved_calls

        if has_unresolved_calls(db, job.id):
            raise ExternalOutcomeUnknown("外部呼び出しの結果が未確定のため公開できません")
        project = db.get(Project, job.project_id)
        if project is None:
            raise StaleGenerationInput("対象プロジェクトが見つかりません")
        if project.revision != job.input_revision:
            raise StaleGenerationInput("生成開始後にプロジェクトの版が変更されました")
        if job.input_fingerprint != settled_fingerprint:
            raise StaleGenerationInput("生成ジョブの入力と確定入力が一致しません")
        current_fingerprint = fingerprint_inputs(capture_inputs(project))
        if current_fingerprint != settled_fingerprint:
            raise StaleGenerationInput("生成開始後に参照入力が変更されました")
        verify_materials(materials)
        verify_materials(block_videos or [])
        expected = (
            project_dir(job.project_id)
            / "history"
            / f"job-{job.id:08d}"
            / "video.mp4"
        ).resolve()
        if final.resolve() != expected or candidate.parent.resolve() != expected.parent:
            raise RuntimeError("このジョブの確定動画を公開できません")
        final_path = relpath_for_db(final)
        if _destination_is_referenced(db, job.id, final_path):
            raise RuntimeError("このジョブの確定動画が既に参照されています")
        if file_identity(candidate) != {
            key: video[key] for key in ("path", "size", "sha256")
        }:
            raise StaleGenerationInput("検証後に完成動画が変更されました")
        if subtitle is not None:
            try:
                if file_identity(subtitle) != subtitle_identity:
                    raise StaleGenerationInput("検証後に字幕ファイルが変更されました")
            except (OSError, ValueError) as exc:
                raise StaleGenerationInput("検証後に字幕ファイルが欠損しました") from exc
        candidate.replace(final)
        final_identity = file_identity(final)
        if any(final_identity[key] != video[key] for key in ("size", "sha256")):
            raise StaleGenerationInput("確定中に完成動画が変更されました")
        video = {**video, **final_identity}
        manifest = {
            "schema_version": 1, "job_id": job.id, "revision": job.input_revision,
            "requested_input_fingerprint": job.input_fingerprint,
            "settled_inputs": settled_inputs, "input_fingerprint": settled_fingerprint,
            "materials": materials, "video": video, "subtitle": subtitle_identity,
            "block_videos": block_videos or [],
            "metadata": [file_identity(path) for path in
                         (final.parent / "project.json", final.parent / "timeline.json") if path.exists()],
        }
        if subtitle is not None:
            try:
                if file_identity(subtitle) != subtitle_identity:
                    raise StaleGenerationInput("確定中に字幕ファイルが変更されました")
            except (OSError, ValueError) as exc:
                raise StaleGenerationInput("確定中に字幕ファイルが欠損しました") from exc
        artifact = GenerationArtifact(
            project_id=job.project_id, job_id=job.id, revision=job.input_revision,
            input_fingerprint=settled_fingerprint, video_path=video["path"],
            subtitle_path=subtitle_identity["path"] if subtitle_identity else None,
            manifest_json=manifest,
        )
        db.add(artifact)
        db.flush()
        project.current_artifact_id = artifact.id
        project.output_video_path = artifact.video_path
        project.output_subtitle_path = artifact.subtitle_path
        project.status = ProjectStatus.completed
        project.progress = 1.0
        project.current_stage = "done"
        project.error_message = None
        job.status = JobStatus.completed
        job.progress = 1.0
        job.current_stage = "done"
        job.finished_at = datetime.now(timezone.utc)
        job.error_message = None
        db.flush()
        db.expunge(artifact)
        return artifact
