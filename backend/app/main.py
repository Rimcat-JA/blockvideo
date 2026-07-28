"""BlockVideo FastAPI application."""
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
        log.error("unhandled exception: {error}", error=str(exc))
        return JSONResponse(
            status_code=500,
            content={"detail": f"{exc.__class__.__name__}: {str(exc)[:200]}"},
        )

    return app


app = create_app()