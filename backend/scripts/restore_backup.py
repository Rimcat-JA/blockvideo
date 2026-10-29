"""List or restore the database backups the migration runner publishes.

Run from backend/ with the app stopped:

    uv run python -m scripts.restore_backup            # list backups, newest first
    uv run python -m scripts.restore_backup --latest   # restore the newest one
    uv run python -m scripts.restore_backup <path>     # restore a specific one

The expected SHA-256 comes from the backup's own metadata file, and the
restore refuses anything that does not verify against it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.core.config import get_settings
from app.db import Base, register_models
from app.migrations.backup import backup_metadata_path, parse_backup_metadata
from app.migrations.contracts import MigrationError
from app.migrations.lease import database_path_from_url
from app.migrations.runner import restore_database_backup

_REASONS = {
    "database_lease_unavailable": "アプリ（バックエンド）が動いています。停止してから実行してください。",
    "backup_invalid": "バックアップを検証できませんでした（破損・改変・別のDB用の可能性があります）。",
}


def published_backups(database_path: Path) -> list[Path]:
    """Backups that have a metadata file, newest first."""
    root = database_path.parent / ".backups"
    if not root.is_dir():
        return []
    backups = [
        path for path in root.glob(f"{database_path.name}.v*.sqlite3")
        if backup_metadata_path(path).is_file()
    ]
    return sorted(backups, key=lambda path: path.name, reverse=True)


def restore(database_url: str, backup: Path) -> None:
    metadata = parse_backup_metadata(backup_metadata_path(backup).read_bytes())
    register_models()
    restore_database_backup(database_url, backup, metadata.backup_sha256, Base.metadata)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    target = parser.add_mutually_exclusive_group()
    target.add_argument("backup", nargs="?", type=Path, help="restore this backup file")
    target.add_argument("--latest", action="store_true", help="restore the newest backup")
    args = parser.parse_args(argv)

    database_url = get_settings().database_url
    database_path = database_path_from_url(database_url)
    backups = published_backups(database_path)
    if args.backup is None and not args.latest:
        if not backups:
            print(f"バックアップはありません: {database_path.parent / '.backups'}")
        for backup in backups:
            print(backup)
        return 0

    backup = args.backup.resolve() if args.backup is not None else (backups[0] if backups else None)
    if backup is None:
        print("復元できるバックアップがありません。", file=sys.stderr)
        return 1
    try:
        restore(database_url, backup)
    except (MigrationError, OSError) as exc:
        reason = getattr(exc, "reason_code", None) or "backup_invalid"
        print(f"復元できませんでした: {_REASONS.get(reason, reason)}", file=sys.stderr)
        return 1
    print(f"{database_path} を {backup.name} から復元しました。アプリを再起動してください。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
