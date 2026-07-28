"""Block-level endpoints — rerun visual / audio / render for one block."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.routes_projects import _block_summary
from app.db import get_db
from app.models.block import Block
from app.models.project import Project
from app.schemas import BlockPatch, BlockSummary, GenerateAllResponse, JobSummary
from app.workers.job_runner import (
    enqueue_block_audio_rerun,
    enqueue_block_visual_rerun,
    enqueue_rerender,
)


router = APIRouter(prefix="/blocks")


@router.get("/{block_id}", response_model=BlockSummary)
def get_block(block_id: int, db: Session = Depends(get_db)) -> BlockSummary:
    """Return one block's current state and artifact URLs."""
    block = db.get(Block, block_id)
    if block is None:
        raise HTTPException(status_code=404, detail="block not found")
    return _block_summary(block)


@router.patch("/{block_id}", response_model=BlockSummary)
def patch_block(block_id: int, payload: BlockPatch, db: Session = Depends(get_db)) -> BlockSummary:
    """Apply editable block fields and return the refreshed block."""
    block = db.get(Block, block_id)
    if block is None:
        raise HTTPException(status_code=404, detail="block not found")
    for field, value in payload.model_dump(exclude_unset=True).items():
        setattr(block, field, value)
    db.commit()
    db.refresh(block)
    return _block_summary(block)


@router.post("/{block_id}/regenerate-visual", response_model=GenerateAllResponse, status_code=202)
async def regenerate_visual(block_id: int, db: Session = Depends(get_db)) -> GenerateAllResponse:
    """Queue visual regeneration for the block's project/index pair."""
    block = db.get(Block, block_id)
    if block is None:
        raise HTTPException(status_code=404, detail="block not found")
    project = db.get(Project, block.project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="project not found")
    job = await enqueue_block_visual_rerun(block.project_id, block.index)
    db.refresh(job)
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
    """Queue audio regeneration for the block's project/index pair."""
    block = db.get(Block, block_id)
    if block is None:
        raise HTTPException(status_code=404, detail="block not found")
    project = db.get(Project, block.project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="project not found")
    job = await enqueue_block_audio_rerun(block.project_id, block.index)
    db.refresh(job)
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
    """Queue a project render because final concat is project-wide."""
    block = db.get(Block, block_id)
    if block is None:
        raise HTTPException(status_code=404, detail="block not found")
    project = db.get(Project, block.project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="project not found")
    # For MVP, rerendering a single block requires rerunning the whole
    # concat — schedule it as a full re-render job.
    job = await enqueue_rerender(block.project_id)
    db.refresh(job)
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
