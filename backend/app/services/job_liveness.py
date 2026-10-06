"""Process-local generation-job liveness without worker or pipeline imports."""
from __future__ import annotations

_running_job_ids: set[int] = set()


def mark_running(job_id: int) -> None:
    """Record that the process owns a live task for a durable job."""
    _running_job_ids.add(job_id)


def clear_running(job_id: int) -> None:
    """Remove a completed or abandoned process-local task marker."""
    _running_job_ids.discard(job_id)


def is_running(job_id: int) -> bool:
    """Return whether this process currently owns the job's task."""
    return job_id in _running_job_ids
