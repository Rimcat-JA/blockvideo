"""Transaction-local durable cancellation; publication uses the same writer lock."""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.models.job import GenerationJob, JobStatus
from app.models.project import Project, ProjectStatus


def cancel_job(db: Session, job: GenerationJob) -> bool:
    """Cancellation cannot undo a completed publication or resolve unknown remote work."""
    if job.status not in {JobStatus.pending, JobStatus.running} or job.cancel_requested:
        return False
    job.cancel_requested = True
    if job.status == JobStatus.pending:
        job.status = JobStatus.cancelled
        job.finished_at = datetime.now(timezone.utc)
        project = db.get(Project, job.project_id)
        if project is not None and (job.input_revision is None or job.input_revision == project.revision):
            project.status = ProjectStatus.cancelled
            project.current_stage = "cancelled"
            project.error_message = None
    db.flush()
    return True
