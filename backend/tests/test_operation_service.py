"""Readiness, dispatch, and representative operation persistence tests."""
from __future__ import annotations

from sqlalchemy import select

from app.db import get_session_factory
from app.models.block import Block, BlockStatus
from app.models.project import Project
from app.operations.bootstrap import build_operation_service
from app.operations.contracts import (
    OperationRequest,
    OperationTarget,
    Readiness,
    ReadinessResult,
)
from app.operations.service import OperationError


def _project() -> Project:
    return Project(
        title="Plan C sample",
        source_script="操作コアの合成サンプルです。",
        subtitle_font_size=48,
        use_fake_providers=True,
    )


def _request(project_id: int | None, value: object = 56, revision: str | None = None) -> OperationRequest:
    return OperationRequest(
        operation_id="project.subtitle-font-size.set",
        target=OperationTarget(project_id=project_id),
        arguments={"value": value},
        observed_state_revision=revision,
    )


def test_missing_target_needs_input(temp_storage) -> None:
    service = build_operation_service()
    with get_session_factory()() as db:
        result = service.readiness(db, _request(None))
    assert result.readiness == Readiness.needs_input
    assert result.reason_code == "target_required"
    assert result.missing_fields == ["project_id"]


def test_nonexistent_target_is_unsupported(temp_storage) -> None:
    service = build_operation_service()
    with get_session_factory()() as db:
        result = service.readiness(db, _request(999))
    assert result.readiness == Readiness.unsupported
    assert result.reason_code == "target_not_found"


def test_busy_project_is_blocked(temp_storage, monkeypatch) -> None:
    service = build_operation_service()
    with get_session_factory()() as db:
        project = _project()
        db.add(project)
        db.commit()
        monkeypatch.setattr("app.operations.readiness.has_live_job", lambda _db, _id: True)
        result = service.readiness(db, _request(project.id))
    assert result.readiness == Readiness.blocked
    assert result.reason_code == "project_busy"


def test_stale_observation_never_writes(temp_storage) -> None:
    service = build_operation_service()
    with get_session_factory()() as db:
        project = _project()
        db.add(project)
        db.commit()
        project_id = project.id
        try:
            service.execute(db, _request(project_id, revision="stale"))
        except OperationError as exc:
            assert exc.reason_code == "stale_state"
        else:
            raise AssertionError("stale execution was accepted")
        db.refresh(project)
        assert project.subtitle_font_size == 48


def test_invalid_value_never_writes(temp_storage) -> None:
    service = build_operation_service()
    with get_session_factory()() as db:
        project = _project()
        db.add(project)
        db.commit()
        try:
            service.execute(db, _request(project.id, value=True))
        except OperationError as exc:
            assert exc.reason_code == "invalid_arguments"
        else:
            raise AssertionError("invalid value was accepted")
        db.refresh(project)
        assert project.subtitle_font_size == 48


def test_state_is_rechecked_immediately_before_dispatch(temp_storage, monkeypatch) -> None:
    service = build_operation_service()
    with get_session_factory()() as db:
        project = _project()
        db.add(project)
        db.commit()
        calls = 0

        def changing_readiness(_db, definition, _target):
            nonlocal calls
            calls += 1
            return ReadinessResult(
                operation_id=definition.operation_id,
                readiness=Readiness.ready if calls == 1 else Readiness.blocked,
                reason_code=None if calls == 1 else "project_busy",
                project_id=project.id,
                state_revision=project.updated_at.isoformat(),
            )

        monkeypatch.setattr("app.operations.service.evaluate_readiness", changing_readiness)
        try:
            service.execute(db, _request(project.id))
        except OperationError as exc:
            assert exc.reason_code == "project_busy"
        else:
            raise AssertionError("state change before dispatch was ignored")
        db.refresh(project)
        assert calls == 2
        assert project.subtitle_font_size == 48


def test_subtitle_size_persists_and_invalidates_render(temp_storage) -> None:
    service = build_operation_service()
    factory = get_session_factory()
    with factory() as db:
        project = _project()
        block = Block(
            project=project,
            index=0,
            source_text="本文",
            tts_text="本文",
            status_render=BlockStatus.completed,
        )
        db.add_all([project, block])
        db.commit()
        project_id = project.id
        result = service.execute(db, _request(project_id, value=56))
        assert result.changed is True
        assert result.data == {"subtitle_font_size": 56}
        assert block.status_render == BlockStatus.pending
    with factory() as reloaded:
        saved = reloaded.get(Project, project_id)
        assert saved is not None
        assert saved.subtitle_font_size == 56


def test_setting_same_value_reports_unchanged(temp_storage) -> None:
    service = build_operation_service()
    with get_session_factory()() as db:
        project = _project()
        db.add(project)
        db.commit()
        result = service.execute(db, _request(project.id, value=48))
    assert result.changed is False


def test_status_operation_is_read_only(temp_storage) -> None:
    service = build_operation_service()
    factory = get_session_factory()
    with factory() as db:
        project = _project()
        db.add(project)
        db.commit()
        project_id = project.id
        updated_at = project.updated_at
        request = OperationRequest(
            operation_id="project.status.get",
            target=OperationTarget(project_id=project_id),
            arguments={},
        )
        result = service.execute(db, request)
        assert result.changed is False
        assert result.data["status"] == "pending"
        assert result.data["subtitle_font_size"] == 48
    with factory() as reloaded:
        saved = reloaded.scalar(select(Project).where(Project.id == project_id))
        assert saved is not None
        assert saved.updated_at == updated_at
