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

from app.core.logging import log
from app.db import get_session_factory
from app.models.job import GenerationJob, JobStatus
from app.models.project import Project, ProjectStatus
from app.services.pipeline import run_generation_job
from app.services.external_calls import ExternalOutcomeUnknown, has_unresolved_calls, job_call_context
from app.services.generation_snapshots import GenerationCancelled
from app.services.job_control import cancel_job

from app.services.transactions import atomic_write
from app.services.job_liveness import clear_running, is_running, mark_running
from app.services.job_records import ProjectBusyError, create_pending_job, has_active_project_job


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
        """Create an accepting task and cancellation registry."""
        self._tasks: dict[int, asyncio.Task] = {}
        self._cancel_flags: dict[int, asyncio.Event] = {}
        self._accepting = True

    @property
    def accepting(self) -> bool:
        """Return whether this lifecycle currently accepts submissions."""
        return self._accepting

    def start(self) -> None:
        """Open a fully drained registry for a new application lifespan."""
        if self._tasks or self._cancel_flags:
            raise RuntimeError("job registry cannot reopen before shutdown completes")
        self._accepting = True

    def close(self) -> None:
        """Stop admission synchronously before dispatcher or task draining."""
        self._accepting = False

    def submit(
        self,
        job_id: int,
        coro_factory: Callable[[Callable[[], bool]], Awaitable[Any]],
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
        if not self._accepting:
            raise RuntimeError("job registry is closing")
        existing = self._tasks.get(job_id)
        if existing is not None:
            return existing
        cancel = asyncio.Event()
        self._cancel_flags[job_id] = cancel

        async def _runner() -> None:
            try:
                with get_session_factory()() as db, atomic_write(db):
                    job = db.get(GenerationJob, job_id)
                    if job is None or job.status != JobStatus.pending:
                        return
                    if job.cancel_requested:
                        job.status = JobStatus.cancelled
                        job.finished_at = datetime.now(timezone.utc)
                        return
                    job.status = JobStatus.running
                    job.started_at = datetime.now(timezone.utc)
                outcome, message = JobStatus.completed, None
                try:
                    def cancel_check() -> bool:
                        with get_session_factory()() as check_db:
                            return cancel.is_set() or self._is_cancel_requested(job_id, check_db)
                    with job_call_context(job_id):
                        await coro_factory(cancel_check)
                except ExternalOutcomeUnknown:
                    outcome = JobStatus.unknown
                    message = "外部処理の結果が未確定です。重複実行を避けるため再送していません。"
                except GenerationCancelled:
                    outcome = JobStatus.cancelled
                except asyncio.CancelledError:
                    # Leave the durable running checkpoint for startup reconciliation.
                    raise
                except Exception as exc:
                    outcome = JobStatus.failed
                    message = f"生成処理に失敗しました ({exc.__class__.__name__})"
                    log.error("job failed id={job_id} error={error}", job_id=job_id, error=exc.__class__.__name__)
                with get_session_factory()() as db, atomic_write(db):
                    job = db.get(GenerationJob, job_id)
                    if job is None or job.status == JobStatus.completed:
                        # Publication has already committed success; a later cancel loses.
                        return
                    if has_unresolved_calls(db, job_id):
                        outcome = JobStatus.unknown
                        message = "外部処理の結果が未確定です。自動再送は停止しています。"
                    elif job.cancel_requested or cancel.is_set():
                        outcome = JobStatus.cancelled
                    job.status = outcome
                    job.finished_at = datetime.now(timezone.utc)
                    job.error_message = message
                    if outcome == JobStatus.completed:
                        job.progress = 1.0
                    project = db.get(Project, job.project_id)
                    if project is not None and (job.input_revision is None or project.revision == job.input_revision):
                        project.status = (ProjectStatus.completed if outcome == JobStatus.completed else
                                          ProjectStatus.cancelled if outcome == JobStatus.cancelled else ProjectStatus.failed)
                        project.current_stage = outcome.value
                        project.error_message = message
            finally:
                self._tasks.pop(job_id, None)
                self._cancel_flags.pop(job_id, None)
                clear_running(job_id)

        task = asyncio.create_task(_runner())
        self._tasks[job_id] = task
        mark_running(job_id)
        return task

    async def shutdown(self) -> None:
        """Close admission, then cancel, await, and remove every accepted task."""
        self.close()
        while self._tasks:
            captured = tuple(self._tasks.items())
            tasks = tuple(task for _job_id, task in captured)
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            for job_id, task in captured:
                if self._tasks.get(job_id) is task:
                    self._tasks.pop(job_id, None)
                self._cancel_flags.pop(job_id, None)
                clear_running(job_id)
        for job_id in tuple(self._cancel_flags):
            self._cancel_flags.pop(job_id, None)
            clear_running(job_id)

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
        with get_session_factory()() as db, atomic_write(db):
            job = db.get(GenerationJob, job_id)
            if job is None:
                return False
            changed = cancel_job(db, job)
        if changed:
            ev = self._cancel_flags.get(job_id)
            if ev is not None:
                ev.set()
        return changed

    def is_running(self, job_id: int) -> bool:
        """Return whether this process currently tracks a live task.

        Args:
            job_id: Job primary key.

        Returns:
            ``True`` only while ``submit`` has a task in ``_tasks``.

        """
        return is_running(job_id)

    def _is_cancel_requested(self, job_id: int, db) -> bool:
        """Read the durable cancellation flag for a job.

        Args:
            job_id: Job primary key.
            db: Session owned by the running task.

        Returns:
            Boolean value of the row's ``cancel_requested`` field, or ``False``
            when the row no longer exists.

        """
        job = db.get(GenerationJob, job_id, populate_existing=True)
        return bool(job and job.cancel_requested)


# Process-wide singleton used by route handlers to signal live tasks.
job_registry = JobRegistry()


def _enqueue(
    project_id: int, stage: str,
) -> GenerationJob:
    """Serialize legacy job creation with durable operation settings/intent writes."""
    with get_session_factory()() as db:
        with atomic_write(db):
            if has_active_project_job(db, project_id, job_registry.is_running):
                raise ProjectBusyError("このプロジェクトは生成中です。完了後に再実行してください。")
            job = create_pending_job(db, project_id, stage)
        db.refresh(job)
        job_registry.submit(job.id, lambda cancel: run_generation_job(job.id, cancel))
        return job


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
    return _enqueue(project_id, "queued")


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
    return _enqueue(project_id, "rerender")


async def enqueue_block_visual_rerun(project_id: int, block_index: int) -> GenerationJob:
    """Create and submit a one-block visual regeneration job.

    Args:
        project_id: Owning project identifier.
        block_index: Zero-based block index.

    Returns:
        Newly committed pending job whose current stage identifies the block.

    """
    return _enqueue(project_id, f"block_visual:{block_index}")


async def enqueue_block_audio_rerun(project_id: int, block_index: int) -> GenerationJob:
    """Create and submit a one-block audio regeneration job.

    Args:
        project_id: Owning project identifier.
        block_index: Zero-based block index.

    Returns:
        Newly committed pending job whose current stage identifies the block.

    """
    return _enqueue(project_id, f"block_audio:{block_index}")
