"""Database setup using SQLAlchemy 2.x sync engine + SQLite."""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import get_settings


class Base(DeclarativeBase):
    """Declarative base shared by every SQLAlchemy model."""

    pass


_engine = None
_SessionLocal: sessionmaker[Session] | None = None


def _make_engine_url(database_url: str) -> str:
    """Prepare a SQLite URL and create its parent directory when necessary."""
    # SQLAlchemy needs forward slashes even on Windows; we already converted.
    if database_url.startswith("sqlite:///"):
        path = database_url[len("sqlite:///") :]
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    return database_url


def get_engine():
    """Return the lazily constructed process-wide SQLAlchemy engine."""
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
    """Return the cached session factory bound to the application engine."""
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=get_engine(), future=True
        )
    return _SessionLocal


def init_db() -> None:
    """Create model tables and apply the project's additive SQLite updates."""
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
    """Yield one request-scoped session and close it after the request."""
    factory = get_session_factory()
    db = factory()
    try:
        yield db
    finally:
        db.close()


def reset_db_for_tests() -> None:
    """Drop all tables and clear cached engine/session."""
    global _engine, _SessionLocal
    if _engine is not None:
        Base.metadata.drop_all(bind=_engine)
        _engine.dispose()
    _engine = None
    _SessionLocal = None
