"""Process-local asynchronous job runner.

For the MVP we use an asyncio task tracker with cancellation tokens,
rather than Redis/RQ. The architecture is deliberately abstracted so
that an RQ/ARQ swap-in only requires re-implementing this module.

Imports:
    ``asyncio`` owns task/event lifecycle.
    Collections/types describe coroutine factories and callbacks.
    ``datetime`` stamps durable job transitions in UTC.
    SQLAlchemy queries and commits job rows.
    Pipeline entry points perform the actual media work.
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
    """Track process-local tasks and their cooperative cancellation events.

    Attributes:
        _tasks: Live ``job_id -> asyncio.Task`` mapping.
        _cancel_flags: Live ``job_id -> asyncio.Event`` mapping checked by
            pipeline callbacks.

    The registry is not a durable queue.  Durable ``GenerationJob`` rows retain
    state, but tasks disappear when the process exits.

    """

    def __init__(self) -> None:
        """Create empty task and cancellation registries."""
        self._tasks: dict[int, asyncio.Task] = {}
        self._cancel_flags: dict[int, asyncio.Event] = {}

    def submit(
        self,
        job_id: int,
        coro_factory: Callable[[], Awaitable[Any]],
    ) -> asyncio.Task:
        """Start a task that owns status commits and cleanup.

        Args:
            job_id: Persisted generation-job primary key.
            coro_factory: Callable receiving a cancellation predicate and
                returning the pipeline coroutine to await.

        Returns:
            The newly scheduled ``asyncio.Task``.

        Side Effects:
            Adds live task/cancellation entries, marks the job running, awaits
            the pipeline, commits completed/failed/cancelled state, closes its
            private session, and removes registry entries.

        """
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
        """Signal a live task or persist cancellation for a pending job.

        Args:
            job_id: Job primary key to cancel.

        Returns:
            ``True`` when a live event was signaled or a row was found and
            marked; ``False`` when no such job exists.

        Side Effects:
            Sets an in-memory event for live tasks, or commits the durable
            ``cancel_requested`` flag for jobs not currently tracked here.

        """
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
        """Return whether this process currently tracks a live task.

        Args:
            job_id: Job primary key.

        Returns:
            ``True`` only while ``submit`` has a task in ``_tasks``.

        """
        return job_id in self._tasks

    def _is_cancel_requested(self, job_id: int, db) -> bool:
        """Read the durable cancellation flag for a job.

        Args:
            job_id: Job primary key.
            db: Session owned by the running task.

        Returns:
            Boolean value of the row's ``cancel_requested`` field, or ``False``
            when the row no longer exists.

        """
        job = db.execute(
            select(GenerationJob).where(GenerationJob.id == job_id)
        ).scalar_one_or_none()
        return bool(job and job.cancel_requested)


# Process-wide singleton used by route handlers to signal live tasks.
job_registry = JobRegistry()


async def enqueue_full_pipeline(project_id: int) -> GenerationJob:
    """Create and submit a full split-to-MP4 generation job.

    Args:
        project_id: Existing project database identifier.

    Returns:
        Newly committed pending ``GenerationJob`` row.

    Side Effects:
        Inserts a job row, schedules ``run_full_pipeline``, and closes the
        enqueueing session.  The returned ORM object is detached afterward.

    """
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
    """Create and submit a render-only project job.

    Args:
        project_id: Existing project database identifier.

    Returns:
        Newly committed pending rerender job.

    Side Effects:
        Inserts the job and schedules ``rerender_project`` in the process-local
        registry.

    """
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
    """Create and submit a one-block visual regeneration job.

    Args:
        project_id: Owning project identifier.
        block_index: Zero-based block index.

    Returns:
        Newly committed pending job whose current stage identifies the block.

    """
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
    """Create and submit a one-block audio regeneration job.

    Args:
        project_id: Owning project identifier.
        block_index: Zero-based block index.

    Returns:
        Newly committed pending job whose current stage identifies the block.

    """
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
