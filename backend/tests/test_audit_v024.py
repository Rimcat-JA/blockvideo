"""Regressions for the v0.2.4 audit fixes."""
from __future__ import annotations

from pathlib import Path

import pytest

from app.db import get_session_factory
from app.models.job import GenerationJob, JobStatus
from app.models.project import Project, ProjectStatus
from app.workers.operation_dispatcher import mark_interrupted_operation_jobs
from tests.test_job_recovery import persisted_job


@pytest.mark.parametrize("status", [JobStatus.running, JobStatus.pending])
def test_jobs_left_in_progress_by_a_v0_app_release_their_project(
    temp_storage: Path, status: JobStatus
) -> None:
    project_id, job_id = persisted_job(status=status)
    with get_session_factory()() as db:
        job = db.get(GenerationJob, job_id)
        # A v0 job has no frozen inputs at all.
        job.input_revision = None
        job.input_snapshot = None
        job.input_fingerprint = None
        db.get(Project, project_id).status = ProjectStatus.rendering
        db.commit()

    assert mark_interrupted_operation_jobs() == 1
    assert mark_interrupted_operation_jobs() == 0

    with get_session_factory()() as db:
        assert db.get(GenerationJob, job_id).status == JobStatus.failed
        assert db.get(Project, project_id).status == ProjectStatus.failed
