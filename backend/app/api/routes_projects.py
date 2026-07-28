"""Project CRUD and pipeline endpoints."""
from __future__ import annotations


from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.utils import validate_artifact_path
from app.core.logging import log
from app.core.security import SecretBundle, secret_store
from app.db import get_db
from app.models.block import Block
from app.models.job import GenerationJob
from app.models.project import Project
from app.schemas import (
    BlockSummary,
    GenerateAllResponse,
    JobSummary,
    ProjectCreate,
    ProjectDetail,
    ProjectPatch,
    ProjectSummary,
    QuickCreate,
    QuickCreateResponse,
)
from app.services.paths import ensure_project_layout, project_dir
from app.services.provider_factory import build_providers_for_project
from app.services.visual_planner import generate_title
from app.workers.job_runner import (
    enqueue_full_pipeline,
    enqueue_rerender,
    job_registry,
)


router = APIRouter(prefix="/projects")


def _project_summary(project: Project) -> ProjectSummary:
    return ProjectSummary(
        id=project.id,
        title=project.title,
        status=project.status.value,
        progress=project.progress,
        current_stage=project.current_stage,
        block_count=len(project.blocks),
        output_video_path=project.output_video_path,
        error_message=project.error_message,
        created_at=project.created_at.isoformat() if project.created_at else None,
        updated_at=project.updated_at.isoformat() if project.updated_at else None,
    )


def _project_detail(project: Project) -> ProjectDetail:
    return ProjectDetail(
        id=project.id,
        title=project.title,
        status=project.status.value,
        progress=project.progress,
        current_stage=project.current_stage,
        block_count=len(project.blocks),
        output_video_path=project.output_video_path,
        error_message=project.error_message,
        created_at=project.created_at.isoformat() if project.created_at else None,
        updated_at=project.updated_at.isoformat() if project.updated_at else None,
        source_script=project.source_script,
        global_visual_style=project.global_visual_style,
        voicevox_url=project.voicevox_url,
        voicevox_speaker_id=project.voicevox_speaker_id,
        voicevox_speed_scale=project.voicevox_speed_scale,
        voicevox_pitch_scale=project.voicevox_pitch_scale,
        voicevox_intonation_scale=project.voicevox_intonation_scale,
        voicevox_volume_scale=project.voicevox_volume_scale,
        subtitle_enabled=project.subtitle_enabled,
        subtitle_font_size=project.subtitle_font_size,
        subtitle_position=project.subtitle_position,
        subtitle_text_color=project.subtitle_text_color,
        subtitle_outline_color=project.subtitle_outline_color,
        subtitle_background=project.subtitle_background,
        subtitle_max_chars_per_line=project.subtitle_max_chars_per_line,
        pre_margin_seconds=project.pre_margin_seconds,
        post_margin_seconds=project.post_margin_seconds,
        min_display_seconds=project.min_display_seconds,
        narration_sentence_pause_seconds=project.narration_sentence_pause_seconds,
        max_slides_per_block=project.max_slides_per_block,
        use_fake_providers=project.use_fake_providers,
        output_subtitle_path=project.output_subtitle_path,
    )


def _block_summary(block: Block) -> BlockSummary:
    project_id = block.project_id
    return BlockSummary(
        id=block.id,
        project_id=project_id,
        index=block.index,
        source_text=block.source_text,
        tts_text=block.tts_text,
        visual_type=block.visual_type.value,
        visual_plan=block.visual_plan_json,
        image_prompt=block.image_prompt,
        image_url=f"/api/projects/{project_id}/artifacts/image/{block.index}" if block.image_path else None,
        audio_url=f"/api/projects/{project_id}/artifacts/audio/{block.index}" if block.audio_path else None,
        video_url=f"/api/projects/{project_id}/artifacts/video/{block.index}" if block.video_path else None,
        duration_ms=block.duration_ms,
        display_duration_ms=block.display_duration_ms,
        status_split=block.status_split.value,
        status_visual_plan=block.status_visual_plan.value,
        status_image=block.status_image.value,
        status_audio=block.status_audio.value,
        status_render=block.status_render.value,
        error_message=block.error_message,
    )


def _job_summary(job: GenerationJob) -> JobSummary:
    return JobSummary(
        id=job.id,
        project_id=job.project_id,
        current_stage=job.current_stage,
        status=job.status.value,
        progress=job.progress,
        stage_progress=job.stage_progress,
        started_at=job.started_at.isoformat() if job.started_at else None,
        finished_at=job.finished_at.isoformat() if job.finished_at else None,
        error_message=job.error_message,
    )


def _provisional_title(script: str) -> str:
    """Cheap title used until (or instead of) the LLM-generated one."""
    for line in script.splitlines():
        stripped = line.strip().lstrip("#＃■・-— ").strip()
        if stripped:
            return stripped[:60]
    return "無題の台本"


@router.post("/quick", response_model=QuickCreateResponse, status_code=201)
async def quick_create(
    payload: QuickCreate, db: Session = Depends(get_db)
) -> QuickCreateResponse:
    """Paste a script, get a video.

    Creates the project with defaults, names it from the script, and starts
    the full pipeline in one call. Title generation is best-effort: a failed
    or slow title must never stop the video from being produced, so we fall
    back to the script's first line.
    """
    script = payload.source_script.strip()
    if not script:
        raise HTTPException(status_code=422, detail="source_script is empty")

    project = Project(
        title=(payload.title or _provisional_title(script)),
        source_script=script,
        voicevox_url=payload.voicevox_url,
        voicevox_speaker_id=payload.voicevox_speaker_id,
        use_fake_providers=payload.use_fake_providers,
    )
    # Only the pacing the caller actually set; the rest keeps the model default.
    for field in (
        "voicevox_speed_scale",
        "narration_sentence_pause_seconds",
        "post_margin_seconds",
        "subtitle_font_size",
        "max_slides_per_block",
    ):
        value = getattr(payload, field)
        if value is not None:
            setattr(project, field, value)
    db.add(project)
    db.commit()
    db.refresh(project)
    ensure_project_layout(project.id)

    if not payload.title:
        try:
            bundle = build_providers_for_project(project)
            project.title = await generate_title(bundle.llm, script=script)
            db.commit()
            db.refresh(project)
        except Exception as exc:  # noqa: BLE001 - never block generation
            log.warning(
                "タイトル自動生成に失敗、暫定タイトルを使用します: {err}",
                err=exc.__class__.__name__,
            )

    job = await enqueue_full_pipeline(project.id)
    fresh_job = db.get(GenerationJob, job.id) or job
    db.refresh(project)
    return QuickCreateResponse(
        project=_project_detail(project), job=_job_summary(fresh_job)
    )


@router.post("", response_model=ProjectDetail, status_code=201)
def create_project(payload: ProjectCreate, db: Session = Depends(get_db)) -> ProjectDetail:
    project = Project(
        title=payload.title,
        source_script=payload.source_script,
        voicevox_url=payload.voicevox_url,
        voicevox_speaker_id=payload.voicevox_speaker_id,
        voicevox_speed_scale=payload.voicevox_speed_scale,
        voicevox_pitch_scale=payload.voicevox_pitch_scale,
        voicevox_intonation_scale=payload.voicevox_intonation_scale,
        voicevox_volume_scale=payload.voicevox_volume_scale,
        subtitle_enabled=payload.subtitle_enabled,
        subtitle_font_size=payload.subtitle_font_size,
        subtitle_position=payload.subtitle_position,
        subtitle_text_color=payload.subtitle_text_color,
        subtitle_outline_color=payload.subtitle_outline_color,
        subtitle_background=payload.subtitle_background,
        subtitle_max_chars_per_line=payload.subtitle_max_chars_per_line,
        pre_margin_seconds=payload.pre_margin_seconds,
        post_margin_seconds=payload.post_margin_seconds,
        narration_sentence_pause_seconds=payload.narration_sentence_pause_seconds,
        max_slides_per_block=payload.max_slides_per_block,
        min_display_seconds=payload.min_display_seconds,
        use_fake_providers=payload.use_fake_providers,
    )
    db.add(project)
    db.flush()
    if payload.providers.model_dump(exclude_none=True):
        secret_store.set(
            project.id,
            SecretBundle(
                llm_api_key=payload.providers.llm_api_key,
                llm_base_url=payload.providers.llm_base_url,
                llm_model=payload.providers.llm_model,
                image_api_key=payload.providers.image_api_key,
                image_base_url=payload.providers.image_base_url,
                image_model=payload.providers.image_model,
            ),
        )
    db.commit()
    db.refresh(project)
    ensure_project_layout(project.id)
    return _project_detail(project)


@router.get("", response_model=list[ProjectSummary])
def list_projects(db: Session = Depends(get_db)) -> list[ProjectSummary]:
    projects = db.execute(select(Project).order_by(Project.created_at.desc())).scalars().all()
    return [_project_summary(p) for p in projects]


@router.get("/{project_id}", response_model=ProjectDetail)
def get_project(project_id: int, db: Session = Depends(get_db)) -> ProjectDetail:
    project = db.get(Project, project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="project not found")
    return _project_detail(project)


@router.delete("/{project_id}", status_code=204)
def delete_project(project_id: int, db: Session = Depends(get_db)) -> Response:
    project = db.get(Project, project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="project not found")
    secret_store.drop(project_id)
    db.delete(project)
    db.commit()
    # Best-effort filesystem cleanup; ignore failures.
    import shutil

    try:

        pd = project_dir(project_id)
        if pd.exists():
            shutil.rmtree(pd, ignore_errors=True)
    except Exception as exc:  # pragma: no cover
        log.warning("storage cleanup failed project_id={pid} err={err}", pid=project_id, err=exc.__class__.__name__)
    return Response(status_code=204)


@router.patch("/{project_id}", response_model=ProjectDetail)
def patch_project(
    project_id: int, payload: ProjectPatch, db: Session = Depends(get_db)
) -> ProjectDetail:
    project = db.get(Project, project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="project not found")
    for field, value in payload.model_dump(exclude_unset=True).items():
        setattr(project, field, value)
    db.commit()
    db.refresh(project)
    return _project_detail(project)


@router.get("/{project_id}/blocks", response_model=list[BlockSummary])
def list_blocks(project_id: int, db: Session = Depends(get_db)) -> list[BlockSummary]:
    project = db.get(Project, project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="project not found")
    return [_block_summary(b) for b in sorted(project.blocks, key=lambda b: b.index)]


@router.get("/{project_id}/jobs", response_model=list[JobSummary])
def list_jobs(project_id: int, db: Session = Depends(get_db)) -> list[JobSummary]:
    project = db.get(Project, project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="project not found")
    jobs = db.execute(
        select(GenerationJob)
        .where(GenerationJob.project_id == project_id)
        .order_by(GenerationJob.id.desc())
        .limit(20)
    ).scalars().all()
    return [_job_summary(j) for j in jobs]


@router.post("/{project_id}/generate-all", response_model=GenerateAllResponse, status_code=202)
async def generate_all(project_id: int, db: Session = Depends(get_db)) -> GenerateAllResponse:
    project = db.get(Project, project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="project not found")
    ensure_project_layout(project_id)
    job = await enqueue_full_pipeline(project_id)
    # enqueue_* closes its own session; re-fetch from the request session.
    fresh = db.get(GenerationJob, job.id)
    summary_source = fresh if fresh is not None else job
    return GenerateAllResponse(job=_job_summary(summary_source), message="queued")


@router.post("/{project_id}/cancel", status_code=200)
def cancel_project(project_id: int, db: Session = Depends(get_db)) -> dict[str, int]:
    """Cancel any active job for the project."""
    cancelled: list[int] = []
    jobs = db.execute(
        select(GenerationJob).where(
            GenerationJob.project_id == project_id,
            GenerationJob.status.in_(["pending", "running"]),
        )
    ).scalars().all()
    for job in jobs:
        job.cancel_requested = True
        if job_registry.request_cancel(job.id):
            cancelled.append(job.id)
    db.commit()
    return {"cancelled": len(cancelled)}


@router.post("/{project_id}/rerender", response_model=GenerateAllResponse, status_code=202)
async def rerender(project_id: int, db: Session = Depends(get_db)) -> GenerateAllResponse:
    project = db.get(Project, project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="project not found")
    if not project.blocks:
        raise HTTPException(status_code=400, detail="まだブロックが生成されていません")
    job = await enqueue_rerender(project_id)
    fresh = db.get(GenerationJob, job.id)
    summary_source = fresh if fresh is not None else job
    return GenerateAllResponse(job=_job_summary(summary_source), message="rerender queued")


@router.get("/{project_id}/artifacts/image/{block_index}")
def artifact_image(project_id: int, block_index: int, db: Session = Depends(get_db)):
    project = db.get(Project, project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="project not found")
    block = next((b for b in project.blocks if b.index == block_index), None)
    if block is None or not block.image_path:
        raise HTTPException(status_code=404, detail="block image not available")
    path = validate_artifact_path(block.image_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="artifact missing on disk")
    return FileResponse(path, media_type="image/png")


@router.get("/{project_id}/artifacts/audio/{block_index}")
def artifact_audio(project_id: int, block_index: int, db: Session = Depends(get_db)):
    project = db.get(Project, project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="project not found")
    block = next((b for b in project.blocks if b.index == block_index), None)
    if block is None or not block.audio_path:
        raise HTTPException(status_code=404, detail="block audio not available")
    path = validate_artifact_path(block.audio_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="artifact missing on disk")
    return FileResponse(path, media_type="audio/wav")


@router.get("/{project_id}/artifacts/video/{block_index}")
def artifact_video_block(project_id: int, block_index: int, db: Session = Depends(get_db)):
    project = db.get(Project, project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="project not found")
    block = next((b for b in project.blocks if b.index == block_index), None)
    if block is None or not block.video_path:
        raise HTTPException(status_code=404, detail="block video not available")
    path = validate_artifact_path(block.video_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="artifact missing on disk")
    return FileResponse(path, media_type="video/mp4")


@router.get("/{project_id}/download")
def download_video(project_id: int, db: Session = Depends(get_db)):
    project = db.get(Project, project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="project not found")
    if not project.output_video_path:
        raise HTTPException(status_code=404, detail="完成動画がまだ生成されていません")
    path = validate_artifact_path(project.output_video_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="動画ファイルが見つかりません")
    return FileResponse(
        path,
        media_type="video/mp4",
        filename=f"blockvideo_{project_id}.mp4",
    )
