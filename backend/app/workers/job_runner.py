"""Background job runner.

For the MVP we use an asyncio task tracker with cancellation tokens,
rather than Redis/RQ. The architecture is deliberately abstracted so
that an RQ/ARQ swap-in only requires re-implementing this module.
"""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select

from app.core.logging import log
from app.db import get_session_factory
from app.models.job import GenerationJob, JobStatus
from app.services.pipeline import (
    rerender_project,
    rerun_block_audio,
    rerun_block_visual,
    run_full_pipeline,
)


class JobRegistry:
    """Track process-local asyncio tasks and their cancellation events."""

    def __init__(self) -> None:
        """Create empty task and cancellation registries."""
        self._tasks: dict[int, asyncio.Task] = {}
        self._cancel_flags: dict[int, asyncio.Event] = {}

    def submit(
        self,
        job_id: int,
        coro_factory: Callable[[], Awaitable[Any]],
    ) -> asyncio.Task:
        """Start a job task that owns status commits and final cleanup."""
        cancel = asyncio.Event()
        self._cancel_flags[job_id] = cancel

        async def _runner() -> None:
            factory = get_session_factory()
            db = factory()
            try:
                job = db.execute(
                    select(GenerationJob).where(GenerationJob.id == job_id)
                ).scalar_one_or_none()
                if job is None:
                    return
                job.status = JobStatus.running
                job.started_at = datetime.now(timezone.utc)
                db.commit()
                try:
                    def cancel_check() -> bool:
                        return cancel.is_set() or self._is_cancel_requested(job_id, db)
                    await coro_factory(cancel_check)
                    job.status = JobStatus.completed
                    job.finished_at = datetime.now(timezone.utc)
                    job.progress = 1.0
                    job.error_message = None
                except Exception as exc:
                    job.status = JobStatus.cancelled if cancel.is_set() else JobStatus.failed
                    job.finished_at = datetime.now(timezone.utc)
                    job.error_message = f"{exc.__class__.__name__}: {str(exc)[:300]}"
                    log.error(
                        "job failed id={job_id} error={error}",
                        job_id=job_id,
                        error=exc.__class__.__name__,
                    )
                finally:
                    db.commit()
            finally:
                db.close()
                self._tasks.pop(job_id, None)
                self._cancel_flags.pop(job_id, None)

        task = asyncio.create_task(_runner())
        self._tasks[job_id] = task
        return task

    def request_cancel(self, job_id: int) -> bool:
        """Signal a live task or mark a not-yet-running job for cancellation."""
        ev = self._cancel_flags.get(job_id)
        if ev is not None:
            ev.set()
            return True
        # job is not currently running in this process; mark the row as
        # cancelled so the next worker pickup stops.
        factory = get_session_factory()
        db = factory()
        try:
            job = db.execute(
                select(GenerationJob).where(GenerationJob.id == job_id)
            ).scalar_one_or_none()
            if job is None:
                return False
            job.cancel_requested = True
            db.commit()
            return True
        finally:
            db.close()

    def is_running(self, job_id: int) -> bool:
        """Return whether this process currently tracks the job's task."""
        return job_id in self._tasks

    def _is_cancel_requested(self, job_id: int, db) -> bool:
        """Read the durable cancellation flag for a job."""
        job = db.execute(
            select(GenerationJob).where(GenerationJob.id == job_id)
        ).scalar_one_or_none()
        return bool(job and job.cancel_requested)


# Process-wide singleton.
job_registry = JobRegistry()


async def enqueue_full_pipeline(project_id: int) -> GenerationJob:
    """Create and submit a full split-to-MP4 generation job."""
    factory = get_session_factory()
    db = factory()
    try:
        job = GenerationJob(
            project_id=project_id,
            current_stage="queued",
            status=JobStatus.pending,
        )
        db.add(job)
        db.commit()
        db.refresh(job)
        job_registry.submit(
            job.id,
            lambda cancel: run_full_pipeline(project_id, cancel_check=cancel),
        )
        return job
    finally:
        db.close()


async def enqueue_rerender(project_id: int) -> GenerationJob:
    """Create and submit a render-only project job."""
    factory = get_session_factory()
    db = factory()
    try:
        job = GenerationJob(
            project_id=project_id,
            current_stage="rerender",
            status=JobStatus.pending,
        )
        db.add(job)
        db.commit()
        db.refresh(job)
        job_registry.submit(job.id, lambda cancel: rerender_project(project_id))
        return job
    finally:
        db.close()


async def enqueue_block_visual_rerun(project_id: int, block_index: int) -> GenerationJob:
    """Create and submit a one-block visual regeneration job."""
    factory = get_session_factory()
    db = factory()
    try:
        job = GenerationJob(
            project_id=project_id,
            current_stage=f"block_visual:{block_index}",
            status=JobStatus.pending,
        )
        db.add(job)
        db.commit()
        db.refresh(job)
        job_registry.submit(
            job.id, lambda cancel: rerun_block_visual(project_id, block_index)
        )
        return job
    finally:
        db.close()


async def enqueue_block_audio_rerun(project_id: int, block_index: int) -> GenerationJob:
    """Create and submit a one-block audio regeneration job."""
    factory = get_session_factory()
    db = factory()
    try:
        job = GenerationJob(
            project_id=project_id,
            current_stage=f"block_audio:{block_index}",
            status=JobStatus.pending,
        )
        db.add(job)
        db.commit()
        db.refresh(job)
        job_registry.submit(
            job.id, lambda cancel: rerun_block_audio(project_id, block_index)
        )
        return job
    finally:
        db.close()
