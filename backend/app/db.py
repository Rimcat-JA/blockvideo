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
            connect_args={"check_same_thread": False},
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


def init_db() -> None:
    """Register models, create missing tables, and apply additive updates.

    Side Effects:
        Imports concrete model modules, creates missing tables, and adds model
        columns absent from an existing SQLite database.  Existing data is not
        migrated beyond these additive column operations.
    """
    # Import models so they register with Base.
    from app.models import block as _block  # noqa: F401
    from app.models import job as _job  # noqa: F401
    from app.models import project as _project  # noqa: F401

    engine = get_engine()
    Base.metadata.create_all(bind=engine)
    _add_missing_columns(engine)


def _add_missing_columns(engine) -> None:
    """Add columns that exist on the models but not yet in the database.

    ``create_all`` only creates missing *tables*, so a new field on an
    existing model is silently absent and every query against it fails. There
    is no migration tool here by design (single-user, local SQLite), and the
    only schema changes this project makes are additive, which SQLite's
    ``ALTER TABLE ADD COLUMN`` handles directly. Anything else — dropping,
    renaming, retyping — is deliberately not attempted.

    Args:
        engine: SQLAlchemy engine whose tables should be compared with
            ``Base.metadata``.

    Raises:
        RuntimeError: If a missing non-nullable column has no server default,
            because SQLite cannot add it while preserving existing rows.

    Side Effects:
        Executes ``ALTER TABLE ... ADD COLUMN`` statements for missing,
        additive columns.  Tables and existing columns are not removed or
        retyped.

    """
    from sqlalchemy import inspect, text

    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())
    with engine.begin() as connection:
        for table in Base.metadata.sorted_tables:
            if table.name not in existing_tables:
                continue
            present = {c["name"] for c in inspector.get_columns(table.name)}
            for column in table.columns:
                if column.name in present:
                    continue
                ddl = f"{column.name} {column.type.compile(engine.dialect)}"
                default = column.server_default
                if default is not None:
                    ddl += f" DEFAULT {default.arg}"
                elif not column.nullable:
                    # ADD COLUMN cannot leave existing rows without a value.
                    raise RuntimeError(
                        f"{table.name}.{column.name} is NOT NULL without a "
                        "server_default; add one so existing rows can be filled"
                    )
                connection.execute(text(f"ALTER TABLE {table.name} ADD COLUMN {ddl}"))


def get_db() -> Iterator[Session]:
    """Yield one request-scoped session and close it after use.

    Yields:
        A new SQLAlchemy ``Session`` bound to the cached application engine.

    Side Effects:
        Always closes the yielded session in the generator's ``finally`` block;
        transaction commit/rollback remains the caller's responsibility.

    """
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
