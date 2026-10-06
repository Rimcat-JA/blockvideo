"""Deliver committed operation generation intents to the single-server worker."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from sqlalchemy import select

from app.core.logging import log
from app.db import get_session_factory
from app.models.job import GenerationJob, JobStatus
from app.models.project import Project, ProjectStatus
from app.services.pipeline import run_generation_job
from app.services.external_calls import has_unresolved_calls, mark_interrupted_calls
from app.services.generation_snapshots import FROZEN_SNAPSHOT_KEYS, capture_inputs, fingerprint_inputs
from app.services.transactions import atomic_write
from app.workers.job_runner import JobRegistry, job_registry


def dispatch_pending_operation_jobs(*, registry: JobRegistry | None = None) -> int:
    """Submit persisted pending jobs; workers perform the atomic running claim."""
    target_registry = registry if registry is not None else job_registry
    with get_session_factory()() as db:
        pending = list(db.execute(
            select(GenerationJob.id, GenerationJob.project_id)
            .where(GenerationJob.status == JobStatus.pending, GenerationJob.input_revision.is_not(None))
            .order_by(GenerationJob.id).limit(100)
        ))
    submitted = 0
    for job_id, project_id in pending:
        if target_registry.is_running(job_id):
            continue
        target_registry.submit(
            job_id,
            lambda cancel, jid=job_id: run_generation_job(jid, cancel),
        )
        submitted += 1
    return submitted


def _can_resume(job: GenerationJob, project: Project | None) -> bool:
    """Require intact initial/checkpoint snapshots matching current frozen inputs."""
    if project is None or job.input_revision != project.revision:
        return False
    if not isinstance(job.input_snapshot, dict) or not job.input_fingerprint:
        return False
    if job.plan_json is not None and not isinstance(job.plan_json, dict):
        return False
    try:
        if fingerprint_inputs(job.input_snapshot) != job.input_fingerprint:
            return False
        plan = job.plan_json or {}
        expected = job.input_fingerprint
        if "resume_inputs" in plan or "resume_fingerprint" in plan:
            checkpoint = plan.get("resume_inputs")
            expected = plan.get("resume_fingerprint")
            if not isinstance(checkpoint, dict) or not expected:
                return False
            if fingerprint_inputs(checkpoint) != expected:
                return False
            # Split/planning may establish blocks and a generated style. They
            # never adopt changed settings or runtime configuration silently.
            for key in FROZEN_SNAPSHOT_KEYS:
                if checkpoint.get(key) != job.input_snapshot.get(key):
                    return False
        return expected == fingerprint_inputs(capture_inputs(project))
    except (TypeError, ValueError):
        return False


def mark_interrupted_operation_jobs() -> int:
    """Recover only verified safe work at single-server startup.

    Remote in-flight work remains unknown. Cancellation completes without new
    work. Intact matching checkpoints return to pending; other jobs fail visibly.
    """
    with get_session_factory()() as db, atomic_write(db):
        mark_interrupted_calls(db)
        db.flush()
        jobs = list(db.scalars(select(GenerationJob).where(GenerationJob.status == JobStatus.running)))
        for job in jobs:
            project = db.get(Project, job.project_id)
            if has_unresolved_calls(db, job.id):
                job.status = JobStatus.unknown
                message = "再起動前の外部処理の結果が未確定です。自動再送していません。"
            elif job.cancel_requested:
                job.status = JobStatus.cancelled
                message = "再起動後にキャンセル要求を確認しました。"
            elif _can_resume(job, project):
                job.status = JobStatus.pending
                message = "再起動後、安全な保存地点から処理を再開します。"
                job.finished_at = None
            else:
                job.status = JobStatus.failed
                message = "停止した生成の入力を確認できません。現在の設定で新しく生成してください。"
            job.recovery_message = message
            job.error_message = None if job.status == JobStatus.pending else message
            if job.status != JobStatus.pending:
                job.finished_at = datetime.now(timezone.utc)
            if project is not None and project.revision == job.input_revision:
                project.status = {
                    JobStatus.pending: ProjectStatus.generating,
                    JobStatus.cancelled: ProjectStatus.cancelled,
                }.get(job.status, ProjectStatus.failed)
                project.current_stage = job.status.value
                project.error_message = job.error_message
        return len(jobs)



async def run_operation_dispatcher() -> None:
    """Poll the transactional outbox; transient submission failures stay pending."""
    while True:
        try:
            dispatch_pending_operation_jobs()
        except Exception as exc:
            log.error("operation dispatch deferred error={error}", error=exc.__class__.__name__)
        await asyncio.sleep(1.0)
