"""Stable result, identity, and failure contracts for SQLite migrations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias


MigrationReasonCode: TypeAlias = Literal[
    "database_lease_unavailable",
    "schema_too_new",
    "unsupported_database",
    "unsupported_legacy_schema",
    "backup_failed",
    "backup_invalid",
    "migration_failed",
    "migration_verification_failed",
]


@dataclass(frozen=True)
class TableIdentity:
    table: str
    row_count: int
    primary_key_columns: tuple[str, ...]
    primary_key_sha256: str


@dataclass(frozen=True)
class MigrationResult:
    status: Literal["created", "current", "migrated"]
    from_version: int
    to_version: int
    backup_created: bool
    backup_sha256: str | None


class MigrationError(RuntimeError):
    """Bounded migration failure carrying a stable machine-readable reason."""

    reason_code: MigrationReasonCode
    backup_available: bool

    def __init__(
        self,
        reason_code: MigrationReasonCode,
        *,
        backup_available: bool = False,
    ) -> None:
        self.reason_code = reason_code
        self.backup_available = backup_available
        super().__init__(reason_code)
