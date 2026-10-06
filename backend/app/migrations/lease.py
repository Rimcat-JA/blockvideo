"""Non-blocking exclusive lease for one file-backed SQLite database."""
from __future__ import annotations

import os
import stat
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy.engine import make_url

from app.migrations.contracts import MigrationError


def database_path_from_url(database_url: str) -> Path:
    """Return the canonical path for a supported file-backed SQLite URL."""
    try:
        url = make_url(database_url)
    except Exception as exc:
        raise MigrationError("unsupported_database") from exc
    if url.drivername != "sqlite" or not url.database or url.database == ":memory:":
        raise MigrationError("unsupported_database")
    return Path(url.database).expanduser().resolve()


def _descriptor_bytes(descriptor: int) -> bytes:
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while chunk := os.read(descriptor, 4096):
        chunks.append(chunk)
    return b"".join(chunks)


def _path_identity_and_bytes(path: Path) -> tuple[tuple[int, int], bytes]:
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        path_stat = os.fstat(descriptor)
        if not stat.S_ISREG(path_stat.st_mode):
            raise OSError("lease path is not a regular file")
        return (path_stat.st_dev, path_stat.st_ino), _descriptor_bytes(descriptor)
    finally:
        os.close(descriptor)


@dataclass
class DatabaseLease:
    database_path: Path
    lock_path: Path
    _descriptor: int | None = field(repr=False)
    _identity: tuple[int, int] = field(repr=False)
    _token: str = field(repr=False)
    _payload: bytes = field(repr=False)

    def _assert_owned_lock(self) -> None:
        descriptor = self._descriptor
        if descriptor is None:
            raise MigrationError("database_lease_unavailable")
        try:
            descriptor_stat = os.fstat(descriptor)
            descriptor_identity = (descriptor_stat.st_dev, descriptor_stat.st_ino)
            path_identity, path_bytes = _path_identity_and_bytes(self.lock_path)
            if (
                not stat.S_ISREG(descriptor_stat.st_mode)
                or descriptor_identity != self._identity
                or path_identity != self._identity
                or _descriptor_bytes(descriptor) != self._payload
                or path_bytes != self._payload
            ):
                raise MigrationError("database_lease_unavailable")
        except MigrationError:
            raise
        except OSError as exc:
            raise MigrationError("database_lease_unavailable") from exc

    def assert_held_for(self, database_url: str) -> None:
        """Require the original path, inode, descriptor, and canonical token bytes."""
        if self._descriptor is None:
            raise MigrationError("database_lease_unavailable")
        try:
            requested_path = database_path_from_url(database_url)
        except MigrationError as exc:
            raise MigrationError("database_lease_unavailable") from exc
        if requested_path != self.database_path:
            raise MigrationError("database_lease_unavailable")
        self._assert_owned_lock()

    def release(self) -> None:
        """Atomically isolate and delete only the unchanged owned lease payload."""
        descriptor = self._descriptor
        if descriptor is None:
            return
        try:
            self._assert_owned_lock()
        except MigrationError:
            self._descriptor = None
            os.close(descriptor)
            raise

        owned_identity = self._identity
        self._descriptor = None
        os.close(descriptor)
        tombstone = self.lock_path.with_name(
            f"{self.lock_path.name}.release-{self._token}"
        )
        try:
            os.replace(self.lock_path, tombstone)
        except FileNotFoundError:
            return
        try:
            moved_identity, moved_bytes = _path_identity_and_bytes(tombstone)
        except OSError as exc:
            raise MigrationError("database_lease_unavailable") from exc
        if moved_identity == owned_identity and moved_bytes == self._payload:
            tombstone.unlink()
            return

        try:
            os.link(tombstone, self.lock_path)
        except OSError as exc:
            raise MigrationError("database_lease_unavailable") from exc
        tombstone.unlink()
        if moved_identity == owned_identity:
            raise MigrationError("database_lease_unavailable")


def acquire_database_lease(database_url: str) -> DatabaseLease:
    """Acquire the sibling migration lock once without waiting or database I/O."""
    database_path = database_path_from_url(database_url)
    lock_path = database_path.with_name(f"{database_path.name}.migration.lock")
    flags = os.O_CREAT | os.O_EXCL | os.O_RDWR | getattr(os, "O_BINARY", 0)
    try:
        database_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as exc:
        raise MigrationError("database_lease_unavailable") from exc
    timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    token = uuid.uuid4().hex
    payload = f"pid={os.getpid()}\nutc={timestamp}\ntoken={token}\n".encode("ascii")
    stat_result = os.fstat(descriptor)
    identity = (stat_result.st_dev, stat_result.st_ino)
    try:
        if os.write(descriptor, payload) != len(payload):
            raise OSError("incomplete lease write")
        os.fsync(descriptor)
        return DatabaseLease(
            database_path=database_path,
            lock_path=lock_path,
            _descriptor=descriptor,
            _identity=identity,
            _token=token,
            _payload=payload,
        )
    except Exception as exc:
        os.close(descriptor)
        try:
            current = lock_path.lstat()
            if (current.st_dev, current.st_ino) == identity:
                lock_path.unlink()
        except OSError:
            pass
        raise MigrationError("database_lease_unavailable") from exc
