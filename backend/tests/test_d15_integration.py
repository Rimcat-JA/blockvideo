"""Final integration regressions across migration, provider identity and entry points."""
from __future__ import annotations

import pytest
from pathlib import Path
from typing import Any
from sqlalchemy import inspect, text
from sqlalchemy import func, select
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.core.security import SecretBundle, secret_store
from app.db import Base, get_engine, get_session_factory
from app.migrations.lease import acquire_database_lease
from app.migrations.runner import migrate_database
from app.models.job import GenerationJob, JobStatus
from app.models.project import Project
from app.main import create_app
from app.services.generation_snapshots import capture_inputs, fingerprint_inputs
from app.workers.operation_dispatcher import mark_interrupted_operation_jobs
from tests.test_generation_controls import execute, request
from tests.test_generation_plan import ready_project
from tests.test_job_recovery import persisted_job


@pytest.mark.parametrize("with_checkpoint", [False, True])
def test_lost_project_provider_identity_never_resumes_on_global_provider(
    temp_storage: Path, monkeypatch: pytest.MonkeyPatch, with_checkpoint: bool,
) -> None:
    settings = get_settings()
    monkeypatch.setattr(settings, "llm_base_url", "https://global.invalid/v1")
    monkeypatch.setattr(settings, "llm_model", "global-synthetic")
    monkeypatch.setattr(settings, "llm_api_key", "synthetic-global-key")
    project_id, job_id = persisted_job()
    secret_store.set(project_id, SecretBundle(llm_api_key="synthetic-project-key",
                                            llm_base_url="https://project.invalid/v1", llm_model="project-synthetic"))
    try:
        with get_session_factory()() as db:
            project = db.get(Project, project_id)
            project.use_fake_providers = False
            job = db.get(GenerationJob, job_id)
            job.input_snapshot = capture_inputs(project)
            job.input_fingerprint = fingerprint_inputs(job.input_snapshot)
            assert "synthetic-project-key" not in str(job.input_snapshot)
            secret_store.drop(project_id)
            if with_checkpoint:
                checkpoint = capture_inputs(project)
                job.plan_json = {**job.plan_json, "resume_inputs": checkpoint,
                                 "resume_fingerprint": fingerprint_inputs(checkpoint)}
            db.commit()
        assert mark_interrupted_operation_jobs() == 1
        with get_session_factory()() as db:
            assert db.get(GenerationJob, job_id).status == JobStatus.failed
    finally:
        secret_store.drop(project_id)


@pytest.mark.parametrize("kind, expected", [("block_audio", ["audio"]), ("block_visual", ["image"]),
                                            ("rerender", ["render"])])
def test_shared_generation_entry_preserves_kind_and_block_index(temp_storage: Path, kind: str, expected: list[str]) -> None:
    with get_session_factory()() as db:
        project = ready_project(db)
        project_id = project.id
    args = {"kind": kind}
    if kind.startswith("block_"):
        args["block_index"] = 0
    result = execute(request(project_id, "generation.start", args))
    with get_session_factory()() as db:
        job = db.get(GenerationJob, result.job_id)
        assert job.kind == kind
        assert job.block_index == args.get("block_index")
        assert job.plan_json["stages"] == expected


def test_additive_d12_d15_migration_preserves_existing_project_and_job(temp_storage: Path) -> None:
    project_id, job_id = persisted_job()
    added = ["kind", "block_index", "input_revision", "input_snapshot", "input_fingerprint",
             "plan_json", "parent_job_id", "recovery_message"]
    engine = get_engine()
    with engine.begin() as connection:
        for table in ["generation_artifacts", "settings_revisions", "external_calls", "project_identities"]:
            connection.execute(text(f"DROP TABLE {table}"))
        connection.execute(text("ALTER TABLE projects DROP COLUMN current_artifact_id"))
        for name in added:
            connection.execute(text(f"ALTER TABLE generation_jobs DROP COLUMN {name}"))
    engine.dispose()
    database_url = get_settings().database_url
    lease = acquire_database_lease(database_url)
    try:
        assert migrate_database(database_url, Base.metadata, lease=lease).status == "migrated"
        assert migrate_database(database_url, Base.metadata, lease=lease).status == "current"
    finally:
        lease.release()
    assert {"generation_artifacts", "settings_revisions", "external_calls", "project_identities"} <= set(inspect(get_engine()).get_table_names())
    with get_session_factory()() as db:
        project, job = db.get(Project, project_id), db.get(GenerationJob, job_id)
        assert project.revision == 1 and project.current_artifact_id is None
        assert job.kind == "full" and job.input_snapshot is None
    mark_interrupted_operation_jobs()
    with get_session_factory()() as db:
        assert db.get(GenerationJob, job_id).status == JobStatus.failed


def test_quick_project_and_generation_intent_commit_together(temp_storage: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.api import routes_projects

    original = routes_projects.create_pending_job

    def interrupted(*args: Any, **kwargs: Any) -> None:
        original(*args, **kwargs)
        raise RuntimeError("synthetic interruption before commit")

    monkeypatch.setattr(routes_projects, "create_pending_job", interrupted)
    response = TestClient(create_app(), raise_server_exceptions=False).post(
        "/api/projects/quick", json={"source_script": "合成テストです。", "use_fake_providers": True})
    assert response.status_code == 500
    with get_session_factory()() as db:
        assert db.scalar(select(func.count()).select_from(Project)) == 0
        assert db.scalar(select(func.count()).select_from(GenerationJob)) == 0
    monkeypatch.setattr(routes_projects, "create_pending_job", original)
    response = TestClient(create_app()).post(
        "/api/projects/quick", json={"source_script": "合成テストです。", "use_fake_providers": True})
    assert response.status_code == 201
    with get_session_factory()() as db:
        job = db.get(GenerationJob, response.json()["job"]["id"])
        assert job.status == JobStatus.pending and job.input_snapshot
