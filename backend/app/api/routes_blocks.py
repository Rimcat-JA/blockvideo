"""Block lookup, editing, and targeted regeneration endpoints.

Imports:
    FastAPI/SQLAlchemy types define the HTTP and database boundaries.
    ``_block_summary`` reuses project-route serialization.
    Worker enqueue functions schedule visual, audio, or project-render work.
    Pydantic schemas define editable fields and queued-job responses.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.routes_projects import _block_summary
from app.api.utils import ensure_project_idle, ensure_render_assets_ready
from app.db import get_db
from app.models.block import Block, VisualType
from app.models.project import Project
from app.models.job import GenerationJob
from app.models.block import BlockStatus
from app.schemas import BlockPatch, BlockSummary, GenerateAllResponse, JobSummary
from app.workers.job_runner import (
    enqueue_block_audio_rerun,
    enqueue_block_visual_rerun,
    enqueue_rerender,
)


# Mounted below the application-level ``/api`` prefix.
router = APIRouter(prefix="/blocks")


@router.get("/{block_id}", response_model=BlockSummary)
def get_block(block_id: int, db: Session = Depends(get_db)) -> BlockSummary:
    """Return one block's current state and artifact URLs.

    Args:
        block_id: Database primary key.
        db: Request-scoped SQLAlchemy session.

    Returns:
        ``BlockSummary`` for the row.

    Raises:
        HTTPException: Status 404 when the block does not exist.

    """
    block = db.get(Block, block_id)
    if block is None:
        raise HTTPException(status_code=404, detail="block not found")
    return _block_summary(block)


@router.patch("/{block_id}", response_model=BlockSummary)
def patch_block(block_id: int, payload: BlockPatch, db: Session = Depends(get_db)) -> BlockSummary:
    """Apply supplied editable fields and return the refreshed block.

    Args:
        block_id: Database primary key.
        payload: Validated source/TTS/visual-plan patch.
        db: Request-scoped SQLAlchemy session.

    Returns:
        Updated ``BlockSummary``.

    Raises:
        HTTPException: Status 404 when the block does not exist.

    Side Effects:
        Mutates and commits the block row; it does not automatically enqueue a
        regeneration job.

    """
    block = db.get(Block, block_id)
    if block is None:
        raise HTTPException(status_code=404, detail="block not found")
    updates = payload.model_dump(exclude_unset=True)
    # Validate the entire request before applying even its first text edit.
    for field in ("source_text", "tts_text"):
        if field in updates and updates[field] is None:
            raise HTTPException(status_code=422, detail=f"{field} cannot be null")

    plan = updates.get("visual_plan")
    visual_type = VisualType.text_slide
    image_prompt = None
    if plan is not None:
        try:
            previous_type = block.visual_type.value if block.visual_type else VisualType.text_slide.value
            visual_type = VisualType(plan.get("visual_type", previous_type))
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=422, detail="unsupported visual_plan.visual_type") from exc
        prompt = plan.get("image_prompt")
        if prompt is not None and not isinstance(prompt, str):
            raise HTTPException(status_code=422, detail="visual_plan.image_prompt must be a string or null")
        plan = {**plan, "visual_type": visual_type.value}
        if visual_type == VisualType.ai_image:
            image_prompt = prompt

    ensure_project_idle(block.project_id, db)
    for field in ("source_text", "tts_text"):
        if field not in updates or getattr(block, field) == updates[field]:
            continue
        setattr(block, field, updates[field])
        block.status_render = BlockStatus.pending
        if field == "tts_text":
            block.status_audio = BlockStatus.pending
        else:
            block.status_visual_plan = BlockStatus.pending
            block.status_image = BlockStatus.pending
            if block.project.narration_pacing_mode == "adaptive":
                block.status_audio = BlockStatus.pending

    if "visual_plan" in updates:
        plan_changed = (
            plan is None
            or block.visual_plan_json != plan
            or block.visual_type != visual_type
            or block.image_prompt != image_prompt
        )
        if plan_changed:
            block.visual_plan_json = plan
            block.visual_type = visual_type
            block.image_prompt = image_prompt
            block.status_image = BlockStatus.pending
            block.status_render = BlockStatus.pending
            # Visible focus terms affect adaptive pauses, even if speech text
            # itself did not change. Fixed-mode audio remains reusable.
            if block.project.narration_pacing_mode == "adaptive":
                block.status_audio = BlockStatus.pending
        # An explicit plan is ready to render, including when source_text was
        # edited in this request. Null deliberately asks the planner to rebuild.
        block.status_visual_plan = BlockStatus.completed if plan is not None else BlockStatus.pending
    db.commit()
    db.refresh(block)
    return _block_summary(block)


@router.post("/{block_id}/regenerate-visual", response_model=GenerateAllResponse, status_code=202)
async def regenerate_visual(block_id: int, db: Session = Depends(get_db)) -> GenerateAllResponse:
    """Queue visual regeneration for one project/block pair.

    Args:
        block_id: Database primary key.
        db: Request-scoped session used to verify ownership and refresh the job.

    Returns:
        ``GenerateAllResponse`` with an HTTP-202 queued job.

    Raises:
        HTTPException: Status 404 when the block or owning project is absent.

    """
    block = db.get(Block, block_id)
    if block is None:
        raise HTTPException(status_code=404, detail="block not found")
    project = db.get(Project, block.project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="project not found")
    ensure_project_idle(block.project_id, db)
    job = await enqueue_block_visual_rerun(block.project_id, block.index)
    job = db.get(GenerationJob, job.id) or job
    return GenerateAllResponse(
        job=JobSummary(
            id=job.id,
            project_id=job.project_id,
            current_stage=job.current_stage,
            status=job.status.value,
            progress=job.progress,
            stage_progress=job.stage_progress,
            started_at=job.started_at.isoformat() if job.started_at else None,
            finished_at=job.finished_at.isoformat() if job.finished_at else None,
            error_message=job.error_message,
        ),
        message="visual regeneration queued",
    )


@router.post("/{block_id}/regenerate-audio", response_model=GenerateAllResponse, status_code=202)
async def regenerate_audio(block_id: int, db: Session = Depends(get_db)) -> GenerateAllResponse:
    """Queue audio regeneration for one project/block pair.

    Args:
        block_id: Database primary key.
        db: Request-scoped session used to verify ownership and refresh the job.

    Returns:
        ``GenerateAllResponse`` with an HTTP-202 queued job.

    Raises:
        HTTPException: Status 404 when the block or owning project is absent.

    """
    block = db.get(Block, block_id)
    if block is None:
        raise HTTPException(status_code=404, detail="block not found")
    project = db.get(Project, block.project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="project not found")
    ensure_project_idle(block.project_id, db)
    job = await enqueue_block_audio_rerun(block.project_id, block.index)
    job = db.get(GenerationJob, job.id) or job
    return GenerateAllResponse(
        job=JobSummary(
            id=job.id,
            project_id=job.project_id,
            current_stage=job.current_stage,
            status=job.status.value,
            progress=job.progress,
            stage_progress=job.stage_progress,
            started_at=job.started_at.isoformat() if job.started_at else None,
            finished_at=job.finished_at.isoformat() if job.finished_at else None,
            error_message=job.error_message,
        ),
        message="audio regeneration queued",
    )


@router.post("/{block_id}/rerender", response_model=GenerateAllResponse, status_code=202)
async def rerender_block(block_id: int, db: Session = Depends(get_db)) -> GenerateAllResponse:
    """Queue a project-wide render requested from one block.

    Args:
        block_id: Database primary key used to find the owning project.
        db: Request-scoped session.

    Returns:
        ``GenerateAllResponse`` with an HTTP-202 project rerender job.

    Raises:
        HTTPException: Status 404 when the block or project is absent.

    The final MP4 is project-wide, so this endpoint intentionally schedules a
    complete render rather than encoding only one block.

    """
    block = db.get(Block, block_id)
    if block is None:
        raise HTTPException(status_code=404, detail="block not found")
    project = db.get(Project, block.project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="project not found")
    # For MVP, rerendering a single block requires rerunning the whole
    # concat — schedule it as a full re-render job.
    ensure_project_idle(block.project_id, db)
    ensure_render_assets_ready(project)
    job = await enqueue_rerender(block.project_id)
    job = db.get(GenerationJob, job.id) or job
    return GenerateAllResponse(
        job=JobSummary(
            id=job.id,
            project_id=job.project_id,
            current_stage=job.current_stage,
            status=job.status.value,
            progress=job.progress,
            stage_progress=job.stage_progress,
            started_at=job.started_at.isoformat() if job.started_at else None,
            finished_at=job.finished_at.isoformat() if job.finished_at else None,
            error_message=job.error_message,
        ),
        message="block rerender queued (via project rerender)",
    )
