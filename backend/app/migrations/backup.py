"""Verified SQLite backup, metadata, and durable publication operations."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import stat
import uuid
from collections.abc import Callable
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import MetaData

from app.migrations.contracts import MigrationError, TableIdentity
from app.migrations.schema import critical_identity_snapshot

_FREE_SPACE_GUARD_BYTES = 16 * 1024 * 1024
_METADATA_SCHEMA_VERSION = 1
_METADATA_KEYS = frozenset(
    {
        "metadata_schema_version",
        "target_database_path_sha256",
        "source_schema_version",
        "source_critical_identities",
        "backup_sha256",
    }
)
_IDENTITY_KEYS = frozenset(
    {"table", "row_count", "primary_key_columns", "primary_key_sha256"}
)


@dataclass(frozen=True)
class VerifiedBackup:
    path: Path
    sha256: str


@dataclass(frozen=True)
class BackupMetadata:
    target_database_path_sha256: str
    source_schema_version: int
    source_critical_identities: dict[str, TableIdentity]
    backup_sha256: str

    def canonical_bytes(self) -> bytes:
        payload = {
            "backup_sha256": self.backup_sha256,
            "metadata_schema_version": _METADATA_SCHEMA_VERSION,
            "source_critical_identities": {
                table: {
                    "primary_key_columns": list(identity.primary_key_columns),
                    "primary_key_sha256": identity.primary_key_sha256,
                    "row_count": identity.row_count,
                    "table": identity.table,
                }
                for table, identity in self.source_critical_identities.items()
            },
            "source_schema_version": self.source_schema_version,
            "target_database_path_sha256": self.target_database_path_sha256,
        }
        return json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")


def sha256_file(path: Path) -> str:
    """Return the lowercase SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def target_database_path_sha256(database_path: Path) -> str:
    """Bind metadata to the normalized canonical target database path."""
    canonical = os.path.normcase(str(database_path.expanduser().resolve()))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def backup_metadata_path(backup_path: Path) -> Path:
    """Return the canonical sidecar path for one published backup."""
    return backup_path.with_name(f"{backup_path.name}.metadata.json")


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _parse_identity(table: str, value: object) -> TableIdentity:
    if not isinstance(value, dict) or set(value) != _IDENTITY_KEYS:
        raise MigrationError("backup_invalid")
    row_count = value["row_count"]
    columns = value["primary_key_columns"]
    if (
        value["table"] != table
        or not isinstance(row_count, int)
        or isinstance(row_count, bool)
        or row_count < 0
        or not isinstance(columns, list)
        or not columns
        or any(not isinstance(column, str) or not column for column in columns)
        or not _is_sha256(value["primary_key_sha256"])
    ):
        raise MigrationError("backup_invalid")
    return TableIdentity(
        table=table,
        row_count=row_count,
        primary_key_columns=tuple(columns),
        primary_key_sha256=value["primary_key_sha256"],
    )


def parse_backup_metadata(raw: bytes) -> BackupMetadata:
    """Parse only exact canonical version-1 backup metadata."""
    try:
        payload: Any = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MigrationError("backup_invalid") from exc
    if not isinstance(payload, dict) or set(payload) != _METADATA_KEYS:
        raise MigrationError("backup_invalid")
    identities_payload = payload["source_critical_identities"]
    if (
        payload["metadata_schema_version"] != _METADATA_SCHEMA_VERSION
        or not _is_sha256(payload["target_database_path_sha256"])
        or payload["source_schema_version"] not in (0, 1)
        or not isinstance(identities_payload, dict)
        or not _is_sha256(payload["backup_sha256"])
    ):
        raise MigrationError("backup_invalid")
    identities = {
        table: _parse_identity(table, identity)
        for table, identity in identities_payload.items()
        if isinstance(table, str) and table
    }
    if len(identities) != len(identities_payload):
        raise MigrationError("backup_invalid")
    metadata = BackupMetadata(
        target_database_path_sha256=payload["target_database_path_sha256"],
        source_schema_version=payload["source_schema_version"],
        source_critical_identities=identities,
        backup_sha256=payload["backup_sha256"],
    )
    if metadata.canonical_bytes() != raw:
        raise MigrationError("backup_invalid")
    return metadata


def fsync_directory(path: Path) -> None:
    """Synchronize directory entries where the platform permits directory fsync."""
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return
    try:
        try:
            os.fsync(descriptor)
        except OSError:
            pass
    finally:
        os.close(descriptor)


def _integrity_is_ok(connection: sqlite3.Connection) -> bool:
    return connection.execute("PRAGMA integrity_check").fetchall() == [("ok",)]


def _atomic_replace(source: Path, destination: Path) -> None:
    os.replace(source, destination)


def _validated_backup_root(database_path: Path) -> Path:
    backup_root = database_path.parent / ".backups"
    backup_root.mkdir(parents=True, exist_ok=True)
    root_stat = backup_root.lstat()
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    file_attributes = getattr(root_stat, "st_file_attributes", 0)
    expected_root = database_path.parent.resolve() / ".backups"
    if (
        stat.S_ISLNK(root_stat.st_mode)
        or file_attributes & reparse_flag
        or not stat.S_ISDIR(root_stat.st_mode)
        or backup_root.resolve(strict=True) != expected_root
    ):
        raise MigrationError("backup_failed")
    return backup_root


def create_verified_backup(
    database_path: Path,
    source: sqlite3.Connection,
    metadata: MetaData,
    expected_identities: dict[str, TableIdentity],
    source_schema_version: int,
    *,
    before_publish: Callable[[], None],
) -> VerifiedBackup:
    """Verify temp bytes before and after durable atomic backup publication."""
    temporary_path: Path | None = None
    metadata_temporary_path: Path | None = None
    final_path: Path | None = None
    final_metadata_path: Path | None = None
    published = False
    backup_root: Path | None = None
    try:
        database_size = database_path.stat().st_size
        free_bytes = shutil.disk_usage(database_path.parent).free
        if free_bytes <= database_size + _FREE_SPACE_GUARD_BYTES:
            raise MigrationError("backup_failed")

        backup_root = _validated_backup_root(database_path)
        fsync_directory(database_path.parent)
        unique = uuid.uuid4().hex
        temporary_path = backup_root / f".{database_path.name}.{unique}.tmp"
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        final_path = backup_root / f"{database_path.name}.v0.{timestamp}.{unique}.sqlite3"
        final_metadata_path = backup_metadata_path(final_path)
        metadata_temporary_path = backup_root / f".{final_metadata_path.name}.{unique}.tmp"

        with closing(sqlite3.connect(temporary_path)) as destination:
            with destination:
                source.backup(destination)
                if not _integrity_is_ok(destination):
                    raise MigrationError("backup_invalid")
                try:
                    backup_identities = critical_identity_snapshot(destination, metadata)
                except MigrationError as exc:
                    raise MigrationError("backup_invalid") from exc
                if backup_identities != expected_identities:
                    raise MigrationError("backup_invalid")

        with temporary_path.open("r+b") as stream:
            stream.flush()
            os.fsync(stream.fileno())
        pre_publish_sha256 = sha256_file(temporary_path)
        before_publish()
        _atomic_replace(temporary_path, final_path)
        fsync_directory(backup_root)
        if sha256_file(final_path) != pre_publish_sha256:
            raise MigrationError("backup_invalid")

        backup_metadata = BackupMetadata(
            target_database_path_sha256=target_database_path_sha256(database_path),
            source_schema_version=source_schema_version,
            source_critical_identities=expected_identities,
            backup_sha256=pre_publish_sha256,
        )
        with metadata_temporary_path.open("xb") as stream:
            stream.write(backup_metadata.canonical_bytes())
            stream.flush()
            os.fsync(stream.fileno())
        before_publish()
        _atomic_replace(metadata_temporary_path, final_metadata_path)
        fsync_directory(backup_root)
        if final_metadata_path.read_bytes() != backup_metadata.canonical_bytes():
            raise MigrationError("backup_invalid")

        published = True
        return VerifiedBackup(path=final_path, sha256=pre_publish_sha256)
    except MigrationError:
        raise
    except (OSError, sqlite3.Error) as exc:
        raise MigrationError("backup_failed") from exc
    finally:
        for path in (temporary_path, metadata_temporary_path):
            if path is not None:
                path.unlink(missing_ok=True)
        if not published:
            for path in (final_metadata_path, final_path):
                if path is not None:
                    path.unlink(missing_ok=True)
            if backup_root is not None:
                fsync_directory(backup_root)
