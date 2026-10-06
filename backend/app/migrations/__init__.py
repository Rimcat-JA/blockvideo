"""Dependency-isolated SQLite migration interfaces."""
from app.migrations.backup import BackupMetadata, backup_metadata_path, sha256_file
from app.migrations.contracts import MigrationError, MigrationResult, TableIdentity
from app.migrations.lease import DatabaseLease, acquire_database_lease
from app.migrations.runner import migrate_database, restore_database_backup
from app.migrations.schema import (
    apply_v0_to_v1,
    classify_v0,
    critical_identity_snapshot,
    sqlite_affinity,
    validate_critical_references,
    validate_schema_compatibility,
)

__all__ = [
    "BackupMetadata",
    "DatabaseLease",
    "MigrationError",
    "MigrationResult",
    "TableIdentity",
    "acquire_database_lease",
    "apply_v0_to_v1",
    "backup_metadata_path",
    "classify_v0",
    "critical_identity_snapshot",
    "migrate_database",
    "restore_database_backup",
    "sha256_file",
    "sqlite_affinity",
    "validate_critical_references",
    "validate_schema_compatibility",
]
