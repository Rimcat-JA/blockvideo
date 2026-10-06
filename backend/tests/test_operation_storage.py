"""Persistent schema upgrade and SQLite transaction boundary checks."""
from __future__ import annotations

import pytest
from sqlalchemy import inspect, text

from app.core.config import get_settings
from app.db import Base, get_engine, get_session_factory
from app.migrations.lease import acquire_database_lease
from app.migrations.runner import migrate_database
from app.models.project import Project
from app.operations.bootstrap import build_operation_service
from app.operations.errors import OperationError
from app.services.transactions import atomic_write
from tests.test_operation_durability import adjustment, execute, make_project


def test_upgrade_database_without_d11_preserves_existing_rows(temp_storage) -> None:
    project_id = make_project()
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE operation_requests"))
        conn.execute(text("ALTER TABLE projects DROP COLUMN revision"))
    engine.dispose()
    database_url = get_settings().database_url
    lease = acquire_database_lease(database_url)
    try:
        assert migrate_database(database_url, Base.metadata, lease=lease).status == "migrated"
        assert migrate_database(database_url, Base.metadata, lease=lease).status == "current"
    finally:
        lease.release()
    assert "operation_requests" in inspect(get_engine()).get_table_names()
    with get_session_factory()() as db:
        project = db.get(Project, project_id)
        assert project.revision == 1
        assert project.subtitle_font_size == 48
        assert project.source_script == "合成テストです。"
    request = adjustment(project_id)
    result = execute(request)
    assert execute(request) == result


def test_cached_project_is_refreshed_inside_write_boundary(temp_storage) -> None:
    project_id = make_project()
    with get_session_factory()() as stale_db:
        cached = stale_db.get(Project, project_id)
        assert cached.revision == 1
        execute(adjustment(project_id))
        with pytest.raises(OperationError) as error:
            build_operation_service().execute(stale_db, adjustment(project_id, "stale"))
        assert error.value.reason_code == "stale_state"


def test_write_boundary_does_not_silently_discard_caller_changes(temp_storage) -> None:
    project_id = make_project()
    with get_session_factory()() as db:
        db.get(Project, project_id).title = "unsaved title"
        with pytest.raises(ValueError, match="pending changes"):
            execute_service = build_operation_service()
            execute_service.execute(db, adjustment(project_id))
        assert db.get(Project, project_id).title == "unsaved title"


def test_busy_database_fails_without_unguarded_write(temp_storage) -> None:
    project_id = make_project()
    with get_session_factory()() as owner, atomic_write(owner):
        with get_session_factory()() as contender:
            contender.execute(text("PRAGMA busy_timeout = 1"))
            with pytest.raises(OperationError) as error:
                build_operation_service().execute(contender, adjustment(project_id))
            assert error.value.reason_code == "database_busy"
    # The same ID is still available after the lock is released.
    assert execute(adjustment(project_id)).revision == 2
