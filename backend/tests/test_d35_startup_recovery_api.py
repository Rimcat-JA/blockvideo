"""D34 startup lifecycle, degraded API, and cross-process lease regressions."""
from __future__ import annotations

import asyncio
import multiprocessing
import os
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
import pytest
from sqlalchemy import event

from app.core.startup_status import (
    StartupStatus,
    StartupUnavailableError,
    get_startup_status,
    reset_startup_status_for_tests,
    set_startup_status,
)
from app.db import Base, get_db, get_session_factory, reset_db_for_tests
from app.main import create_app
from app.migrations import MigrationError, MigrationResult
from app.migrations.backup import sha256_file
from app.migrations.runner import restore_database_backup
from app.models.external_call import ExternalCall
from app.models.job import GenerationJob, JobStatus
from app.models.project import Project
from app.services.job_records import ProjectBusyError, UnresolvedExternalWorkError, create_pending_job
from app.services.job_views import build_recovery_contexts, job_summary
from app.services.transactions import begin_write
from tests.fixtures.migrations.build_fixtures import build_fixture


class RecordingLease:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.release_count = 0

    def release(self) -> None:
        self.events.append("release")
        self.release_count += 1


class RecordingRegistry:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.accepting = False

    def start(self) -> None:
        self.events.append("registry_start")
        self.accepting = True

    def close(self) -> None:
        self.events.append("registry_close")
        self.accepting = False

    async def shutdown(self) -> None:
        self.events.append("registry_shutdown")


@pytest.fixture(autouse=True)
def reset_startup_status() -> None:
    reset_startup_status_for_tests()
    yield
    reset_startup_status_for_tests()


def _patch_successful_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[str], RecordingLease, RecordingRegistry]:
    from app import main as main_module

    events: list[str] = []
    lease = RecordingLease(events)
    registry = RecordingRegistry(events)

    def register_models() -> None:
        events.append("register")

    def acquire(database_url: str) -> RecordingLease:
        assert database_url
        events.append("acquire")
        return lease

    def migrate(database_url: str, metadata: Any, *, lease: RecordingLease) -> MigrationResult:
        assert database_url
        assert lease is not None
        assert {"projects", "generation_jobs", "language_requests"} <= set(metadata.tables)
        events.append("migrate")
        return MigrationResult("current", 1, 1, False, None)

    def init_db() -> None:
        events.append("init")

    def recover() -> int:
        events.append("recover")
        return 0

    async def dispatch() -> None:
        events.append("dispatch")
        await asyncio.Event().wait()

    def shutdown_db() -> None:
        assert get_startup_status() == StartupStatus(
            status="starting",
            reason_code=None,
            message="終了処理中です。",
            schema_version=None,
            backup_available=False,
        )
        events.append("shutdown_db")

    monkeypatch.setattr(main_module, "register_models", register_models)
    monkeypatch.setattr(main_module, "acquire_database_lease", acquire)
    monkeypatch.setattr(main_module, "migrate_database", migrate)
    monkeypatch.setattr(main_module, "init_db", init_db)
    monkeypatch.setattr(main_module, "shutdown_db", shutdown_db)
    monkeypatch.setattr(main_module, "mark_interrupted_operation_jobs", recover)
    monkeypatch.setattr(main_module, "run_operation_dispatcher", dispatch)
    monkeypatch.setattr(main_module.operation_dispatcher, "job_registry", registry)
    return events, lease, registry


def test_job_summary_derives_exact_recovery_codes_from_persisted_state_and_journal(
    temp_storage: Path,
) -> None:
    with get_session_factory()() as db:
        jobs: dict[str, GenerationJob] = {}
        for name, status, cancel_requested in [
            ("pending", JobStatus.pending, False),
            ("running", JobStatus.running, False),
            ("cancel_requested", JobStatus.running, True),
            ("failed_retryable", JobStatus.failed, False),
            ("failed_unknown_call", JobStatus.failed, False),
            ("failed_local_unknown_call", JobStatus.failed, False),
            ("unknown", JobStatus.unknown, False),
            ("unknown_cancel_requested", JobStatus.unknown, True),
            ("cancelled", JobStatus.cancelled, True),
            ("completed", JobStatus.completed, False),
            ("detached_failed", JobStatus.failed, False),
            ("detached_cancelled", JobStatus.cancelled, True),
        ]:
            project = Project(title=f"recovery-{name}", source_script="synthetic")
            db.add(project)
            db.flush()
            job = GenerationJob(
                project_id=project.id,
                current_stage="synthetic",
                status=status,
                cancel_requested=cancel_requested,
            )
            db.add(job)
            jobs[name] = job
        db.flush()
        db.add_all([
            ExternalCall(
                job_id=jobs["failed_unknown_call"].id,
                fingerprint="f" * 64,
                provider="synthetic",
                endpoint="https://provider.invalid/jobs",
                remote_side_effect=True,
                status="unknown",
            ),
            ExternalCall(
                job_id=jobs["failed_local_unknown_call"].id,
                fingerprint="l" * 64,
                provider="synthetic-local",
                endpoint="http://localhost/jobs",
                remote_side_effect=False,
                status="unknown",
            ),
        ])
        blocked_project = Project(title="project-blocker", source_script="synthetic")
        db.add(blocked_project)
        db.flush()
        blocked_target = GenerationJob(
            project_id=blocked_project.id,
            current_stage="synthetic",
            status=JobStatus.cancelled,
            cancel_requested=True,
        )
        unknown_sibling = GenerationJob(
            project_id=blocked_project.id,
            current_stage="synthetic",
            status=JobStatus.unknown,
        )
        db.add_all([blocked_target, unknown_sibling])
        jobs["cancelled_other_unknown_job"] = blocked_target
        db.commit()

        detached_failed = jobs.pop("detached_failed")
        detached_cancelled = jobs.pop("detached_cancelled")
        db.refresh(detached_failed)
        db.refresh(detached_cancelled)
        db.expunge(detached_failed)
        db.expunge(detached_cancelled)
        contexts = build_recovery_contexts(db, [job.project_id for job in jobs.values()])
        summaries = {
            name: job_summary(job, contexts.get(job.project_id))
            for name, job in jobs.items()
        }
        summaries["detached_failed"] = job_summary(detached_failed)
        summaries["detached_cancelled"] = job_summary(detached_cancelled)

    expected = {
        "pending": ("wait", "wait", False, None),
        "running": ("wait", "wait", False, None),
        "cancel_requested": ("wait", "wait", False, None),
        "failed_retryable": ("safe_retry", "retry_current", True, None),
        "failed_unknown_call": (
            "external_outcome_unknown",
            "check_provider",
            False,
            "以前の外部処理の結果が未確定です。外部サービス側の履歴を確認できるまで再実行できません。",
        ),
        "failed_local_unknown_call": ("safe_retry", "retry_current", True, None),
        "unknown": (
            "external_outcome_unknown",
            "check_provider",
            False,
            "外部処理の結果が未確定です。このアプリでは結果を照会できないため、外部サービス側の履歴を確認してください。",
        ),
        "unknown_cancel_requested": (
            "external_outcome_unknown",
            "check_provider",
            False,
            "外部処理の結果が未確定です。このアプリでは結果を照会できないため、外部サービス側の履歴を確認してください。",
        ),
        "cancelled": ("safe_retry", "retry_current", True, None),
        "cancelled_other_unknown_job": (
            "external_outcome_unknown",
            "check_provider",
            False,
            "以前の外部処理の結果が未確定です。外部サービス側の履歴を確認できるまで再実行できません。",
        ),
        "completed": ("completed", "none", False, None),
        "detached_failed": (
            "refresh_required",
            "refresh",
            False,
            "現在の状態を再取得してから再実行してください。",
        ),
        "detached_cancelled": (
            "refresh_required",
            "refresh",
            False,
            "現在の状態を再取得してから再実行してください。",
        ),
    }
    assert {
        name: (
            summary.recovery_code,
            summary.recommended_action,
            summary.retryable,
            summary.retry_blocked_reason,
        )
        for name, summary in summaries.items()
    } == expected


@pytest.mark.parametrize(
    ("route_name", "enqueue_name"),
    [
        ("regenerate_visual", "enqueue_block_visual_rerun"),
        ("regenerate_audio", "enqueue_block_audio_rerun"),
        ("rerender_block", "enqueue_rerender"),
    ],
)
def test_block_regeneration_responses_include_required_recovery_contract(
    temp_storage: Path,
    monkeypatch: pytest.MonkeyPatch,
    route_name: str,
    enqueue_name: str,
) -> None:
    from app.api import routes_blocks
    from app.models.block import Block

    with get_session_factory()() as db:
        project = Project(title="synthetic", source_script="synthetic")
        db.add(project)
        db.flush()
        block = Block(
            project_id=project.id,
            index=0,
            source_text="synthetic",
            tts_text="synthetic",
        )
        job = GenerationJob(
            project_id=project.id,
            current_stage="queued",
            status=JobStatus.pending,
            progress=0.0,
            stage_progress=0.0,
            cancel_requested=False,
        )
        db.add_all([block, job])
        db.commit()

        async def enqueue(*_args: Any) -> GenerationJob:
            return job

        monkeypatch.setattr(routes_blocks, "ensure_project_idle", lambda *_args: None)
        monkeypatch.setattr(routes_blocks, "ensure_render_assets_ready", lambda *_args: None)
        monkeypatch.setattr(routes_blocks, enqueue_name, enqueue)

        response = asyncio.run(getattr(routes_blocks, route_name)(block.id, db))

        assert response.job.recovery_code == "wait"
        assert response.job.recommended_action == "wait"
        assert response.job.retryable is False


@pytest.mark.parametrize("blocker", ("busy", "unresolved_remote"))
def test_project_generation_recovery_sees_blockers_older_than_history_window(
    temp_storage: Path,
    blocker: str,
) -> None:
    from app.api.routes_history import project_history
    from app.api.routes_projects import get_project

    with get_session_factory()() as db:
        project = Project(title=f"old-{blocker}", source_script="synthetic")
        db.add(project)
        db.flush()
        old_job = GenerationJob(
            project_id=project.id,
            current_stage="synthetic",
            status=JobStatus.running if blocker == "busy" else JobStatus.failed,
            input_snapshot={"source_script": "synthetic"} if blocker == "busy" else None,
        )
        db.add(old_job)
        db.flush()
        if blocker == "unresolved_remote":
            db.add(ExternalCall(
                job_id=old_job.id,
                fingerprint="o" * 64,
                provider="synthetic",
                endpoint="https://provider.invalid/jobs",
                remote_side_effect=True,
                status="unknown",
            ))
        db.add_all([
            GenerationJob(
                project_id=project.id,
                current_stage="synthetic",
                status=JobStatus.completed,
            )
            for _ in range(101)
        ])
        db.commit()

        history = project_history(project.id, db)
        assert len(history["jobs"]) == 100
        assert old_job.id not in {job["id"] for job in history["jobs"]}

        detail = get_project(project.id, db)
        expected = (
            ("busy", "wait")
            if blocker == "busy"
            else ("external_outcome_unknown", "check_provider")
        )
        assert (
            detail.generation_recovery.code,
            detail.generation_recovery.recommended_action,
        ) == expected

        begin_write(db)
        with pytest.raises(
            ProjectBusyError if blocker == "busy" else UnresolvedExternalWorkError
        ):
            create_pending_job(db, project.id)
        db.rollback()


@pytest.mark.parametrize(
    ("terminal_status", "active_status"),
    [
        (JobStatus.failed, JobStatus.running),
        (JobStatus.cancelled, JobStatus.pending),
    ],
)
def test_history_blocks_terminal_retry_guidance_while_sibling_job_is_active(
    temp_storage: Path,
    terminal_status: JobStatus,
    active_status: JobStatus,
) -> None:
    from app.api.routes_history import project_history

    with get_session_factory()() as db:
        project = Project(title="active-sibling", source_script="synthetic")
        db.add(project)
        db.flush()
        terminal = GenerationJob(
            project_id=project.id,
            current_stage="synthetic",
            status=terminal_status,
        )
        active = GenerationJob(
            project_id=project.id,
            current_stage="synthetic",
            status=active_status,
        )
        db.add_all([terminal, active])
        db.commit()

        history = project_history(project.id, db)
        terminal_summary = next(job for job in history["jobs"] if job["id"] == terminal.id)

        assert terminal_summary["recovery_code"] == "wait"
        assert terminal_summary["recommended_action"] == "wait"
        assert terminal_summary["retryable"] is False

        begin_write(db)
        with pytest.raises(ProjectBusyError):
            create_pending_job(db, project.id)
        db.rollback()


def test_project_generation_recovery_is_ready_without_durable_blockers(
    temp_storage: Path,
) -> None:
    from app.api.routes_projects import get_project

    with get_session_factory()() as db:
        project = Project(title="ready", source_script="synthetic")
        db.add(project)
        db.commit()

        detail = get_project(project.id, db)

        assert detail.generation_recovery.model_dump() == {
            "code": "ready",
            "recommended_action": "generate",
        }


def test_job_list_and_history_build_one_recovery_context_query_each(
    temp_storage: Path,
) -> None:
    from app.api.routes_history import project_history
    from app.api.routes_projects import list_jobs

    with get_session_factory()() as db:
        project = Project(title="query-count", source_script="synthetic")
        db.add(project)
        db.flush()
        db.add_all([
            GenerationJob(
                project_id=project.id,
                current_stage="synthetic",
                status=JobStatus.failed,
            )
            for _ in range(100)
        ])
        db.commit()
        project_id = project.id

    def select_count(call: Any) -> tuple[int, int]:
        with get_session_factory()() as db:
            statements: list[str] = []

            def record(_conn: Any, _cursor: Any, statement: str, *_args: Any) -> None:
                if statement.lstrip().upper().startswith("SELECT"):
                    statements.append(statement)

            engine = db.get_bind()
            event.listen(engine, "before_cursor_execute", record)
            try:
                result = call(db)
            finally:
                event.remove(engine, "before_cursor_execute", record)
            return len(result if isinstance(result, list) else result["jobs"]), len(statements)

    assert select_count(lambda db: list_jobs(project_id, db)) == (20, 3)
    assert select_count(lambda db: project_history(project_id, db)) == (100, 5)


def test_lifespan_registers_leases_migrates_then_initializes_before_database_users(
    temp_storage: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events, lease, _registry = _patch_successful_lifecycle(monkeypatch)
    application = create_app()

    async def exercise() -> None:
        async with application.router.lifespan_context(application):
            await asyncio.sleep(0)
            assert get_startup_status() == StartupStatus(
                status="ready",
                reason_code=None,
                message="起動が完了しました。",
                schema_version=1,
                backup_available=False,
            )
            assert events == [
                "register",
                "acquire",
                "migrate",
                "init",
                "recover",
                "registry_start",
                "dispatch",
            ]
            assert lease.release_count == 0

    asyncio.run(exercise())

    assert events[-4:] == [
        "registry_close",
        "registry_shutdown",
        "shutdown_db",
        "release",
    ]
    assert lease.release_count == 1
    with pytest.raises(StartupUnavailableError):
        next(get_db())


@pytest.mark.parametrize(
    "reason_code",
    [
        "database_lease_unavailable",
        "schema_too_new",
        "unsupported_database",
        "unsupported_legacy_schema",
        "backup_failed",
        "backup_invalid",
        "migration_failed",
        "migration_verification_failed",
    ],
)
def test_migration_failure_keeps_status_routes_readable_and_database_routes_fixed_503(
    temp_storage: Path,
    monkeypatch: pytest.MonkeyPatch,
    reason_code: str,
) -> None:
    from app import main as main_module

    events: list[str] = []
    lease = RecordingLease(events)
    monkeypatch.setattr(main_module, "register_models", lambda: events.append("register"))
    monkeypatch.setattr(
        main_module,
        "acquire_database_lease",
        lambda _database_url: events.append("acquire") or lease,
    )

    def fail_migration(*_args: Any, **_kwargs: Any) -> None:
        events.append("migrate")
        raise MigrationError(reason_code)  # type: ignore[arg-type]

    monkeypatch.setattr(main_module, "migrate_database", fail_migration)
    monkeypatch.setattr(main_module, "init_db", lambda: pytest.fail("init must not run"))
    monkeypatch.setattr(main_module, "shutdown_db", lambda: events.append("shutdown_db"))
    monkeypatch.setattr(
        main_module,
        "mark_interrupted_operation_jobs",
        lambda: pytest.fail("recovery must not run"),
    )
    monkeypatch.setattr(
        main_module,
        "run_operation_dispatcher",
        lambda: pytest.fail("dispatcher must not run"),
    )

    with TestClient(create_app()) as client:
        assert client.get("/api/startup").json() == {
            "status": "migration_failed",
            "reason_code": reason_code,
            "message": "データベースの移行に失敗しました。管理者に確認してください。",
            "schema_version": None,
            "backup_available": False,
        }
        assert client.get("/api/health").json()["status"] == "degraded"
        response = client.get("/api/projects")
        assert response.status_code == 503
        assert response.headers["Retry-After"] == "5"
        assert response.json() == {
            "detail": {
                "reason_code": "startup_unavailable",
                "message": "データベースを利用できません。起動状態を確認してください。",
            }
        }
        assert lease.release_count == 0

    assert events == ["register", "acquire", "migrate", "shutdown_db", "release"]
    assert get_startup_status() == StartupStatus(
        status="starting",
        reason_code=None,
        message="終了処理中です。",
        schema_version=None,
        backup_available=False,
    )
    assert lease.release_count == 1


def test_migration_failure_propagates_verified_backup_availability(
    temp_storage: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app import main as main_module

    events: list[str] = []
    lease = RecordingLease(events)
    monkeypatch.setattr(main_module, "register_models", lambda: events.append("register"))
    monkeypatch.setattr(
        main_module,
        "acquire_database_lease",
        lambda _database_url: events.append("acquire") or lease,
    )

    def fail_after_backup(*_args: Any, **_kwargs: Any) -> None:
        events.append("migrate")
        raise MigrationError("migration_failed", backup_available=True)

    monkeypatch.setattr(main_module, "migrate_database", fail_after_backup)
    monkeypatch.setattr(main_module, "init_db", lambda: pytest.fail("init must not run"))
    monkeypatch.setattr(main_module, "shutdown_db", lambda: events.append("shutdown_db"))

    with TestClient(create_app()) as client:
        assert client.get("/api/startup").json() == {
            "status": "migration_failed",
            "reason_code": "migration_failed",
            "message": "データベースの移行に失敗しました。管理者に確認してください。",
            "schema_version": None,
            "backup_available": True,
        }


def test_post_acquisition_startup_error_releases_lease_once(
    temp_storage: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app import main as main_module

    events, lease, _registry = _patch_successful_lifecycle(monkeypatch)

    def fail_init() -> None:
        events.append("init")
        raise RuntimeError("synthetic init failure")

    monkeypatch.setattr(main_module, "init_db", fail_init)

    with pytest.raises(RuntimeError, match="synthetic init failure"):
        with TestClient(create_app()):
            pass

    assert events == [
        "register",
        "acquire",
        "migrate",
        "init",
        "shutdown_db",
        "release",
    ]
    assert lease.release_count == 1


def test_registry_shutdown_error_still_releases_lease_once(
    temp_storage: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events, lease, registry = _patch_successful_lifecycle(monkeypatch)

    async def fail_shutdown() -> None:
        events.append("registry_shutdown")
        raise RuntimeError("synthetic shutdown failure")

    registry.shutdown = fail_shutdown  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="synthetic shutdown failure"):
        with TestClient(create_app()):
            pass

    assert events[-4:] == [
        "registry_close",
        "registry_shutdown",
        "shutdown_db",
        "release",
    ]
    assert lease.release_count == 1


def test_lease_acquisition_failure_aborts_without_database_users_or_release(
    temp_storage: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app import main as main_module

    events: list[str] = []
    monkeypatch.setattr(main_module, "register_models", lambda: events.append("register"))

    def fail_acquire(_database_url: str) -> None:
        events.append("acquire")
        raise MigrationError("database_lease_unavailable")

    monkeypatch.setattr(main_module, "acquire_database_lease", fail_acquire)
    monkeypatch.setattr(
        main_module, "migrate_database", lambda *_args, **_kwargs: pytest.fail("no migration")
    )
    monkeypatch.setattr(main_module, "init_db", lambda: pytest.fail("no init"))
    monkeypatch.setattr(
        main_module,
        "mark_interrupted_operation_jobs",
        lambda: pytest.fail("no recovery"),
    )
    monkeypatch.setattr(
        main_module,
        "run_operation_dispatcher",
        lambda: pytest.fail("no dispatcher"),
    )

    with pytest.raises(MigrationError, match="database_lease_unavailable"):
        with TestClient(create_app()):
            pass

    assert events == ["register", "acquire"]


def test_offline_restore_between_lifespans_reopens_restored_database_inode(
    temp_storage: Path,
) -> None:
    database = temp_storage / "blockvideo.db"
    reset_db_for_tests()
    database.unlink(missing_ok=True)
    build_fixture("d30_v0", database, Base.metadata)
    application = create_app()

    with TestClient(application):
        with get_session_factory()() as db:
            project = db.get(Project, 101)
            assert project is not None
            assert project.title == "fixture-project"
            project.title = "state-from-replaced-inode"
            db.commit()

    with pytest.raises(StartupUnavailableError):
        next(get_db())

    backup = next(
        path
        for path in (database.parent / ".backups").iterdir()
        if not path.name.endswith(".metadata.json")
    )
    backup_sha256 = sha256_file(backup)
    restore_database_backup(
        f"sqlite:///{database.as_posix()}",
        backup,
        backup_sha256,
        Base.metadata,
    )
    assert sha256_file(database) == backup_sha256

    with TestClient(application):
        with get_session_factory()() as db:
            project = db.get(Project, 101)
            assert project is not None
            assert project.title == "fixture-project"


def _hold_application_lifespan(
    database_url: str,
    storage_root: str,
    ready: multiprocessing.Queue,
    stop: multiprocessing.synchronize.Event,
) -> None:
    os.environ["DATABASE_URL"] = database_url
    os.environ["STORAGE_ROOT"] = storage_root
    from app.core.config import reset_settings_cache
    from app.db import reset_db_for_tests

    reset_settings_cache()
    reset_db_for_tests()
    try:
        with TestClient(create_app()):
            ready.put(None)
            stop.wait(timeout=30)
    except Exception as exc:
        ready.put((exc.__class__.__name__, str(exc)))
        raise


def test_second_application_process_is_excluded_until_first_shutdown(
    temp_storage: Path,
) -> None:
    database = temp_storage / "process-exclusion.db"
    database_url = f"sqlite:///{database.as_posix()}"
    context = multiprocessing.get_context("spawn")
    ready = context.Queue()
    stop = context.Event()
    process = context.Process(
        target=_hold_application_lifespan,
        args=(database_url, str(temp_storage), ready, stop),
    )
    process.start()
    try:
        assert ready.get(timeout=30) is None
        os.environ["DATABASE_URL"] = database_url
        from app.core.config import reset_settings_cache
        from app.db import reset_db_for_tests

        reset_settings_cache()
        reset_db_for_tests()
        with pytest.raises(MigrationError, match="database_lease_unavailable"):
            with TestClient(create_app()):
                pass
    finally:
        stop.set()
        process.join(timeout=30)
        if process.is_alive():
            process.terminate()
            process.join(timeout=10)

    assert process.exitcode == 0
    set_startup_status(
        StartupStatus(
            status="starting",
            reason_code=None,
            message="起動処理中です。",
            schema_version=None,
            backup_available=False,
        )
    )
    with TestClient(create_app()) as client:
        assert client.get("/api/startup").json()["status"] == "ready"
