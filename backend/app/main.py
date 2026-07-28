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

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import __version__
from app.api.routes_blocks import router as blocks_router
from app.api.routes_health import router as health_router
from app.api.routes_projects import router as projects_router
from app.core.config import get_settings
from app.core.logging import configure_logging, log
from app.db import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize process-wide services for the ASGI application lifetime.

    Args:
        app: FastAPI instance entering its lifespan.  The argument is required
            by the framework and is not otherwise inspected.

    Side Effects:
        Configures Loguru, loads settings, logs startup identity, and creates
        or updates local database tables before yielding control.  No explicit
        shutdown work is currently required after the yield.

    """
    configure_logging()
    settings = get_settings()
    log.info(
        "starting BlockVideo version={version} env={env}",
        version=__version__,
        env=settings.environment,
    )
    init_db()
    yield


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
    app.include_router(projects_router, prefix="/api")
    app.include_router(blocks_router, prefix="/api")

    @app.exception_handler(Exception)
    async def _unhandled(_request, exc: Exception):  # pragma: no cover
        """Convert an unexpected exception into a bounded JSON 500 response.

        Args:
            _request: FastAPI request object retained for handler signature.
            exc: Unhandled exception raised by route/dependency code.

        Returns:
            ``JSONResponse`` containing the exception class and first 200
            characters of its string representation.

        """
        log.error("unhandled exception: {error}", error=str(exc))
        return JSONResponse(
            status_code=500,
            content={"detail": f"{exc.__class__.__name__}: {str(exc)[:200]}"},
        )

    return app


# ASGI entry point imported by Uvicorn/Gunicorn-style runners.
app = create_app()
