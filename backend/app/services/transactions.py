"""Short SQLite writer transactions shared by settings and operation entry points."""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session


class WriteBusyError(RuntimeError):
    """The database writer is busy; retry the same request, not a new intent."""


def begin_write(db: Session) -> None:
    """Own a fresh writer transaction, rejecting pending ORM changes.

    SQLite's writer lock covers read/resolve/write across processes. Ending a
    clean read transaction also expires identity-map rows before revision checks.
    Callers must not have flushed uncommitted writes before entering this boundary.
    They must commit or roll back; network/media work never belongs inside.
    """
    if db.new or db.dirty or db.deleted:
        raise ValueError("write boundary requires a session without pending changes")
    db.rollback()
    db.expire_all()
    try:
        db.execute(text("BEGIN IMMEDIATE"))
    except OperationalError as exc:
        db.rollback()
        code = getattr(exc.orig, "sqlite_errorcode", 0)
        if code & 0xFF in {5, 6}:
            raise WriteBusyError("database busy; retry the same request ID") from exc
        raise


@contextmanager
def atomic_write(db: Session) -> Iterator[None]:
    """Commit all effects together, including receipts and pending jobs."""
    begin_write(db)
    try:
        yield
        db.commit()
    except BaseException:
        db.rollback()
        raise
