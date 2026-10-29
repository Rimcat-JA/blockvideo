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


def test_a_legacy_subtitle_colour_does_not_block_later_saves(temp_storage: Path) -> None:
    from fastapi.testclient import TestClient

    from app.main import create_app

    client = TestClient(create_app())
    project = client.post("/api/projects", json={
        "title": "legacy colour", "source_script": "台本です。", "use_fake_providers": True,
    }).json()
    with get_session_factory()() as db:
        db.get(Project, project["id"]).subtitle_outline_color = "navy-ish"
        db.commit()

    response = client.patch(f"/api/projects/{project['id']}", json={"title": "renamed"})

    assert response.status_code == 200, response.text
    assert response.json()["title"] == "renamed"
    with get_session_factory()() as db:
        assert db.get(Project, project["id"]).subtitle_outline_color == "#000000"


@pytest.mark.parametrize("path, extra", [
    ("/api/projects", {"title": "t", "source_script": "台本です。"}),
    ("/api/projects/quick", {"source_script": "台本です。"}),
])
def test_an_out_of_range_speaker_id_is_a_validation_error(
    temp_storage: Path, path: str, extra: dict
) -> None:
    from fastapi.testclient import TestClient

    from app.main import create_app

    client = TestClient(create_app())
    response = client.post(path, json={**extra, "use_fake_providers": True,
                                       "voicevox_speaker_id": 2**63})
    assert response.status_code == 422


def test_restore_script_lists_and_restores_the_latest_backup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import sqlite3
    from contextlib import closing

    from app.core import config
    from app.db import Base, register_models
    from app.migrations.lease import acquire_database_lease
    from app.migrations.runner import migrate_database
    from scripts import restore_backup
    from tests.fixtures.migrations.build_fixtures import build_fixture

    register_models()
    database = build_fixture("d30_v0", tmp_path / "blockvideo.db", Base.metadata)
    url = f"sqlite:///{database.as_posix()}"
    lease = acquire_database_lease(url)
    try:
        migrate_database(url, Base.metadata, lease=lease)
    finally:
        lease.release()
    monkeypatch.setenv("DATABASE_URL", url)
    config.reset_settings_cache()

    assert restore_backup.main([]) == 0
    assert ".backups" in capsys.readouterr().out
    assert restore_backup.main(["--latest"]) == 0
    with closing(sqlite3.connect(database)) as restored:
        assert restored.execute("PRAGMA user_version").fetchone() == (0,)

    held = acquire_database_lease(url)
    try:
        assert restore_backup.main(["--latest"]) == 1
        assert "停止" in capsys.readouterr().err
    finally:
        held.release()
    config.reset_settings_cache()
