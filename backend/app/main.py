"""FastAPI application factory and process startup lifecycle.

Imports:
    ``asynccontextmanager`` defines startup/shutdown lifecycle scope.
    FastAPI/CORS/JSONResponse build the HTTP application boundary.
    Version, settings, logging, database, and route modules supply the app's
    identity, configuration, startup services, and endpoints.

``create_app`` is the testable factory.  The module-level ``app`` is the ASGI
object used by Uvicorn and other deployment runners.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, suppress
import uuid

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import __version__
from app.api.routes_blocks import router as blocks_router
from app.api.routes_health import router as health_router
from app.api.routes_history import router as history_router
from app.api.routes_language import router as language_router
from app.api.routes_language_connection import router as language_connection_router
from app.api.routes_operations import router as operations_router
from app.api.routes_projects import router as projects_router
from app.api.routes_startup import router as startup_router
from app.core.config import get_settings
from app.core.logging import configure_logging, log
from app.core.access_logging import protect_access_logs
from app.core.startup_status import (
    StartupStatus,
    StartupUnavailableError,
    set_startup_status,
)
from app.db import Base, init_db, register_models, shutdown_db
from app.migrations import MigrationError, acquire_database_lease, migrate_database
from app.services.transactions import WriteBusyError
from app.services.job_records import ProjectBusyError, UnresolvedExternalWorkError
from app.workers import operation_dispatcher
from app.workers.operation_dispatcher import mark_interrupted_operation_jobs, run_operation_dispatcher


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize process-wide services for the ASGI application lifetime.

    Args:
        app: FastAPI instance entering its lifespan.  The argument is required
            by the framework and is not otherwise inspected.

    Side Effects:
        Configures Loguru, loads settings, logs startup identity, and creates
        or updates local database tables, marks interrupted durable jobs, and
        runs the pending-operation dispatcher until shutdown.

    """
    configure_logging()
    protect_access_logs()
    settings = get_settings()
    log.info(
        "starting BlockVideo version={version} env={env}",
        version=__version__,
        env=settings.environment,
    )
    set_startup_status(
        StartupStatus(
            status="starting",
            reason_code=None,
            message="起動処理中です。",
            schema_version=None,
            backup_available=False,
        )
    )
    register_models()
    lease = acquire_database_lease(settings.database_url)
    dispatcher: asyncio.Task[None] | None = None
    registry_started = False
    try:
        try:
            migration = migrate_database(
                settings.database_url, Base.metadata, lease=lease
            )
        except MigrationError as exc:
            log.error(
                "database migration failed error_class={error_class} reason_code={reason_code}",
                error_class=exc.__class__.__name__,
                reason_code=exc.reason_code,
            )
            set_startup_status(
                StartupStatus(
                    status="migration_failed",
                    reason_code=exc.reason_code,
                    message="データベースの移行に失敗しました。管理者に確認してください。",
                    schema_version=None,
                    backup_available=exc.backup_available,
                )
            )
            yield
            return

        init_db()
        set_startup_status(
            StartupStatus(
                status="ready",
                reason_code=None,
                message="起動が完了しました。",
                schema_version=migration.to_version,
                backup_available=migration.backup_created,
            )
        )
        mark_interrupted_operation_jobs()
        operation_dispatcher.job_registry.start()
        registry_started = True
        dispatcher = asyncio.create_task(run_operation_dispatcher())
        yield
    finally:
        try:
            if registry_started:
                operation_dispatcher.job_registry.close()
            try:
                if dispatcher is not None:
                    dispatcher.cancel()
                    with suppress(asyncio.CancelledError):
                        await dispatcher
            finally:
                if registry_started:
                    await operation_dispatcher.job_registry.shutdown()
        finally:
            set_startup_status(
                StartupStatus(
                    status="starting",
                    reason_code=None,
                    message="終了処理中です。",
                    schema_version=None,
                    backup_available=False,
                )
            )
            try:
                shutdown_db()
            finally:
                lease.release()


def create_app() -> FastAPI:
    """Build and configure the FastAPI application.

    Returns:
        A new FastAPI instance with permissive local-development CORS, health,
        project, and block routers under ``/api``, plus a final unexpected-error
        handler that returns a short JSON response.

    Side Effects:
        Reads cached settings while constructing middleware configuration; the
        database itself is initialized later by ``lifespan``.

    """
    settings = get_settings()
    app = FastAPI(
        title="BlockVideo API",
        version=__version__,
        lifespan=lifespan,
        # Do not include docs URLs in production-style defaults; keep them for MVP.
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(health_router, prefix="/api")
    app.include_router(startup_router, prefix="/api")
    app.include_router(projects_router, prefix="/api")
    app.include_router(blocks_router, prefix="/api")
    app.include_router(operations_router, prefix="/api")
    app.include_router(history_router, prefix="/api")
    app.include_router(language_router, prefix="/api")
    app.include_router(language_connection_router, prefix="/api")

    @app.exception_handler(UnresolvedExternalWorkError)
    async def _external_unknown(_request, exc: UnresolvedExternalWorkError) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(ProjectBusyError)
    async def _project_busy(_request, exc: ProjectBusyError) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(WriteBusyError)
    async def _write_busy(_request, exc: WriteBusyError) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": str(exc)},
                            headers={"Retry-After": "1"})

    @app.exception_handler(StartupUnavailableError)
    async def _startup_unavailable(
        _request, _exc: StartupUnavailableError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=503,
            content={
                "detail": {
                    "reason_code": "startup_unavailable",
                    "message": "データベースを利用できません。起動状態を確認してください。",
                }
            },
            headers={"Retry-After": "5"},
        )

    @app.exception_handler(Exception)
    async def _unhandled(_request, exc: Exception) -> JSONResponse:
        """Convert an unexpected exception into a fixed, correlatable response."""
        correlation_id = uuid.uuid4().hex[:16]
        route = _request.scope.get("route")
        path = getattr(route, "path", "unknown")
        log.error(
            "unhandled exception correlation_id={correlation_id} path={path} error_class={error_class}",
            correlation_id=correlation_id,
            path=path,
            error_class=exc.__class__.__name__,
        )
        return JSONResponse(
            status_code=500,
            content={
                "detail": {
                    "reason_code": "internal_error",
                    "message": "処理に失敗しました。再読み込み後も続く場合は記録番号を確認してください。",
                    "correlation_id": correlation_id,
                },
            },
        )

    return app


# ASGI entry point imported by Uvicorn/Gunicorn-style runners.
app = create_app()
