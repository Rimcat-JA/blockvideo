"""Process-wide SQLAlchemy 2.x synchronous setup for local SQLite storage.

Imports:
    ``Iterator`` types the request-scoped dependency generator.
    ``Path`` creates SQLite parent directories.
    SQLAlchemy engine/session classes provide the database and ORM base.
    ``get_settings`` supplies the configured database URL.

The module lazily creates one engine and one session factory per process.  The
cache is intentional for the single-user local application; tests can call
``reset_db_for_tests`` to dispose it and drop the current schema.
"""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import get_settings
from app.core.startup_status import StartupUnavailableError, get_startup_status


class Base(DeclarativeBase):
    """Declarative base from which all BlockVideo ORM models inherit."""

    pass


# Lazy process-wide engine; initialized on the first database access.
_engine = None
# Lazy factory bound to ``_engine``; reset alongside the engine in tests.
_SessionLocal: sessionmaker[Session] | None = None


def _make_engine_url(database_url: str) -> str:
    """Prepare a database URL and create a local SQLite parent directory.

    Args:
        database_url: SQLAlchemy URL.  Only ``sqlite:///`` URLs receive local
            filesystem preparation; other dialects pass through unchanged.

    Returns:
        The same URL string, suitable for ``create_engine``.

    Side Effects:
        Creates the parent directory for a file-backed SQLite database.

    """
    # SQLAlchemy needs forward slashes even on Windows; we already converted.
    if database_url.startswith("sqlite:///"):
        path = database_url[len("sqlite:///") :]
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    return database_url


def get_engine():
    """Return the lazily constructed process-wide SQLAlchemy engine.

    Returns:
        The cached SQLAlchemy engine, creating it from ``Settings.database_url``
        on the first call.  SQLite connections disable same-thread checking
        because FastAPI dependency execution can cross worker threads.

    """
    global _engine
    if _engine is None:
        settings = get_settings()
        url = _make_engine_url(settings.database_url)
        _engine = create_engine(
            url,
            connect_args={"check_same_thread": False, "timeout": 30},
            future=True,
        )
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    """Return the cached ``Session`` factory bound to ``get_engine()``.

    Returns:
        A SQLAlchemy ``sessionmaker`` configured with no autocommit and no
        autoflush.  Each caller must close the session it creates.

    """
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=get_engine(), future=True
        )
    return _SessionLocal


def register_models() -> None:
    """Import every ORM model module without opening or altering a database."""
    from app.models import artifact as _artifact  # noqa: F401
    from app.models import block as _block  # noqa: F401
    from app.models import external_call as _external_call  # noqa: F401
    from app.models import job as _job  # noqa: F401
    from app.models import language_request as _language_request  # noqa: F401
    from app.models import language_turn as _language_turn  # noqa: F401
    from app.models import operation_request as _operation_request  # noqa: F401
    from app.models import project as _project  # noqa: F401
    from app.models import project_identity as _project_identity  # noqa: F401
    from app.models import settings_revision as _settings_revision  # noqa: F401


def init_db() -> None:
    """Create the registered current schema after migration approval."""
    Base.metadata.create_all(bind=get_engine())


def shutdown_db() -> None:
    """Dispose the application engine pool and clear cached database factories.

    This production shutdown seam performs no schema mutation. The next
    application lifespan constructs a new engine bound to the current database
    file after acquiring its lease.
    """
    global _engine, _SessionLocal
    try:
        if _engine is not None:
            _engine.dispose()
    finally:
        _engine = None
        _SessionLocal = None


def get_db() -> Iterator[Session]:
    """Yield one request-scoped session and close it after use.

    Yields:
        A new SQLAlchemy ``Session`` bound to the cached application engine.

    Side Effects:
        Always closes the yielded session in the generator's ``finally`` block;
        transaction commit/rollback remains the caller's responsibility.

    """
    if get_startup_status().status != "ready":
        raise StartupUnavailableError()
    factory = get_session_factory()
    db = factory()
    try:
        yield db
    finally:
        db.close()


def reset_db_for_tests() -> None:
    """Drop the current schema, dispose the engine, and clear both caches.

    This is a test-only reset seam.  The next database access constructs a new
    engine and session factory from the then-current settings.

    Side Effects:
        Drops every table registered in ``Base.metadata`` and disposes open
        connections held by the cached engine.
    """
    global _engine, _SessionLocal
    if _engine is not None:
        Base.metadata.drop_all(bind=_engine)
        _engine.dispose()
    _engine = None
    _SessionLocal = None
