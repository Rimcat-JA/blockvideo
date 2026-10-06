"""SQLite v0-to-v1 classification, additive DDL, and identity validation."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterable
from typing import Literal, TypeAlias

from sqlalchemy import MetaData, Table, UniqueConstraint, create_engine
from sqlalchemy.dialects.sqlite import dialect as sqlite_dialect
from sqlalchemy.pool import StaticPool
from sqlalchemy.schema import CreateColumn, CreateIndex, CreateTable

from app.migrations.contracts import MigrationError, TableIdentity


SQLiteAffinity: TypeAlias = Literal["INTEGER", "TEXT", "BLOB", "REAL", "NUMERIC"]

_SQLITE_ASCII_FOLD = str.maketrans(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz"
)
_SQLITE_ASCII_UPPER = str.maketrans(
    "abcdefghijklmnopqrstuvwxyz", "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
)

CRITICAL_TABLES = (
    "projects",
    "blocks",
    "generation_jobs",
    "operation_requests",
    "external_calls",
    "generation_artifacts",
    "settings_revisions",
    "language_requests",
    "language_turns",
)


def sqlite_affinity(declared_type: str) -> SQLiteAffinity:
    """Derive affinity using SQLite's ordered declared-type rules."""
    normalized = declared_type.translate(_SQLITE_ASCII_UPPER)
    if "INT" in normalized:
        return "INTEGER"
    if any(token in normalized for token in ("CHAR", "CLOB", "TEXT")):
        return "TEXT"
    if "BLOB" in normalized or not normalized:
        return "BLOB"
    if any(token in normalized for token in ("REAL", "FLOA", "DOUB")):
        return "REAL"
    return "NUMERIC"


def _quote(identifier: str) -> str:
    return sqlite_dialect().identifier_preparer.quote(identifier)


def _normalized_identifier(identifier: str) -> str:
    return identifier.translate(_SQLITE_ASCII_FOLD)


def _identifier_map(identifiers: Iterable[str]) -> dict[str, str]:
    indexed: dict[str, str] = {}
    for identifier in identifiers:
        normalized = _normalized_identifier(identifier)
        if normalized in indexed:
            raise MigrationError("unsupported_legacy_schema")
        indexed[normalized] = identifier
    return indexed


def _validate_metadata_identifiers(metadata: MetaData) -> None:
    _identifier_map(table.name for table in metadata.tables.values())
    for table in metadata.tables.values():
        _identifier_map(column.name for column in table.columns)


def _table_names(connection: sqlite3.Connection) -> set[str]:
    return {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
    }


def _table_name_map(connection: sqlite3.Connection) -> dict[str, str]:
    return _identifier_map(_table_names(connection))


def _table_info(
    connection: sqlite3.Connection, table: str
) -> dict[str, tuple[str, int]]:
    info = {
        str(row[1]): (str(row[2] or ""), int(row[5]))
        for row in connection.execute(f"PRAGMA table_info({_quote(table)})")
    }
    _identifier_map(info)
    return info


IndexColumnSet: TypeAlias = frozenset[str]
IndexRequirements: TypeAlias = tuple[
    dict[IndexColumnSet, tuple[str, ...]], dict[IndexColumnSet, tuple[str, ...]]
]


def _metadata_index_requirements(table: Table) -> IndexRequirements:
    ordinary: dict[IndexColumnSet, tuple[str, ...]] = {}
    unique: dict[IndexColumnSet, tuple[str, ...]] = {}

    for index in table.indexes:
        columns = tuple(column.name for column in index.columns)
        if not columns or len(columns) != len(index.expressions):
            raise MigrationError("unsupported_legacy_schema")
        canonical = frozenset(_normalized_identifier(column) for column in columns)
        target = unique if index.unique else ordinary
        target.setdefault(canonical, columns)

    for constraint in table.constraints:
        if not isinstance(constraint, UniqueConstraint):
            continue
        columns = tuple(column.name for column in constraint.columns)
        if not columns:
            raise MigrationError("unsupported_legacy_schema")
        canonical = frozenset(_normalized_identifier(column) for column in columns)
        unique.setdefault(canonical, columns)

    for column in table.columns:
        if column.unique:
            columns = (column.name,)
            unique.setdefault(
                frozenset({_normalized_identifier(column.name)}), columns
            )
    return ordinary, unique


def _observed_index_sets(
    connection: sqlite3.Connection, table: str
) -> tuple[set[IndexColumnSet], set[IndexColumnSet]]:
    indexed: set[IndexColumnSet] = set()
    unique: set[IndexColumnSet] = set()
    try:
        rows = connection.execute(f"PRAGMA index_list({_quote(table)})").fetchall()
        for row in rows:
            if len(row) > 4 and int(row[4]):
                continue
            index_name = str(row[1])
            info = connection.execute(
                f"PRAGMA index_info({_quote(index_name)})"
            ).fetchall()
            if not info or any(item[2] is None for item in info):
                continue
            columns = frozenset(
                _normalized_identifier(str(item[2])) for item in info
            )
            indexed.add(columns)
            if int(row[2]):
                unique.add(columns)
    except sqlite3.Error as exc:
        raise MigrationError("unsupported_legacy_schema") from exc
    return indexed, unique


def _validate_required_indexes(
    connection: sqlite3.Connection, metadata: MetaData
) -> None:
    observed_tables = _table_name_map(connection)
    for table in metadata.sorted_tables:
        actual_table = observed_tables.get(_normalized_identifier(table.name))
        if actual_table is None:
            raise MigrationError("unsupported_legacy_schema")
        required_ordinary, required_unique = _metadata_index_requirements(table)
        indexed, unique = _observed_index_sets(connection, actual_table)
        if not set(required_ordinary) <= indexed or not set(required_unique) <= unique:
            raise MigrationError("unsupported_legacy_schema")


def _scratch_schema(metadata: MetaData) -> dict[str, dict[str, tuple[str, int]]]:
    _validate_metadata_identifiers(metadata)
    scratch = sqlite3.connect(":memory:")
    engine = create_engine(
        "sqlite://",
        creator=lambda: scratch,
        poolclass=StaticPool,
    )
    try:
        metadata.create_all(engine)
        return {
            table: _table_info(scratch, table)
            for table in _table_names(scratch)
        }
    except MigrationError:
        raise
    except Exception as exc:
        raise MigrationError("unsupported_legacy_schema") from exc
    finally:
        engine.dispose()


def validate_schema_compatibility(
    connection: sqlite3.Connection, metadata: MetaData, *, version: int
) -> None:
    """Validate version-aware table, column, and affinity compatibility."""
    if version not in (0, 1):
        raise MigrationError("unsupported_legacy_schema")

    expected = _scratch_schema(metadata)
    expected_tables = _identifier_map(expected)
    observed_tables = _table_name_map(connection)
    for table in metadata.sorted_tables:
        normalized_table = _normalized_identifier(table.name)
        expected_table = expected_tables[normalized_table]
        actual_table = observed_tables.get(normalized_table)
        if actual_table is None:
            if version == 1:
                raise MigrationError("unsupported_legacy_schema")
            continue

        observed_columns = _table_info(connection, actual_table)
        observed_names = _identifier_map(observed_columns)
        expected_columns = expected[expected_table]
        expected_names = _identifier_map(expected_columns)
        for column in table.columns:
            normalized_column = _normalized_identifier(column.name)
            actual_column = observed_names.get(normalized_column)
            if actual_column is None:
                if version == 1 or (
                    not column.nullable and column.server_default is None
                ):
                    raise MigrationError("unsupported_legacy_schema")
                continue
            expected_column = expected_names[normalized_column]
            if sqlite_affinity(observed_columns[actual_column][0]) != sqlite_affinity(
                expected_columns[expected_column][0]
            ):
                raise MigrationError("unsupported_legacy_schema")

    if version == 1:
        _validate_required_indexes(connection, metadata)


def classify_v0(connection: sqlite3.Connection, metadata: MetaData) -> None:
    """Validate structural compatibility for an additive version-0 schema."""
    validate_schema_compatibility(connection, metadata, version=0)


def _missing_columns(
    connection: sqlite3.Connection, metadata: MetaData
) -> list[tuple[str, object]]:
    existing_tables = _table_name_map(connection)
    missing: list[tuple[str, object]] = []
    for table in metadata.sorted_tables:
        actual_table = existing_tables.get(_normalized_identifier(table.name))
        if actual_table is None:
            continue
        present = _identifier_map(_table_info(connection, actual_table))
        for column in table.columns:
            if _normalized_identifier(column.name) in present:
                continue
            if not column.nullable and column.server_default is None:
                raise MigrationError("unsupported_legacy_schema")
            missing.append((actual_table, column))
    return missing


def _create_missing_tables(
    connection: sqlite3.Connection,
    metadata: MetaData,
    existing_tables: dict[str, str],
) -> None:
    dialect = sqlite_dialect()
    for table in metadata.sorted_tables:
        if _normalized_identifier(table.name) in existing_tables:
            continue
        connection.execute(str(CreateTable(table).compile(dialect=dialect)))
        for index in sorted(table.indexes, key=lambda item: item.name or ""):
            connection.execute(str(CreateIndex(index).compile(dialect=dialect)))


def _deterministic_index_name(
    table: str,
    columns: tuple[str, ...],
    *,
    unique: bool,
    existing_names: set[str],
) -> str:
    kind = "uq" if unique else "ix"
    descriptor = json.dumps(
        [kind, table, *columns], ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    digest = hashlib.sha256(descriptor).hexdigest()[:12]
    base = f"d34_{kind}_{table}_{'_'.join(columns)}_{digest}"
    candidate = base
    suffix = 1
    while _normalized_identifier(candidate) in existing_names:
        candidate = f"{base}_{suffix}"
        suffix += 1
    existing_names.add(_normalized_identifier(candidate))
    return candidate


def _create_missing_indexes(
    connection: sqlite3.Connection, metadata: MetaData
) -> None:
    observed_tables = _table_name_map(connection)
    existing_names = {
        _normalized_identifier(str(row[0]))
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        )
    }
    for table in metadata.sorted_tables:
        actual_table = observed_tables[_normalized_identifier(table.name)]
        observed_columns = _identifier_map(_table_info(connection, actual_table))
        required_ordinary, required_unique = _metadata_index_requirements(table)
        indexed, unique = _observed_index_sets(connection, actual_table)
        requirements = (
            (False, required_ordinary, indexed),
            (True, required_unique, unique),
        )
        for is_unique, required, present in requirements:
            for column_set, columns in sorted(
                required.items(), key=lambda item: item[1]
            ):
                if column_set in present:
                    continue
                actual_columns = tuple(
                    observed_columns[_normalized_identifier(column)]
                    for column in columns
                )
                index_name = _deterministic_index_name(
                    table.name,
                    columns,
                    unique=is_unique,
                    existing_names=existing_names,
                )
                qualifier = "UNIQUE " if is_unique else ""
                connection.execute(
                    f"CREATE {qualifier}INDEX {_quote(index_name)} ON "
                    f"{_quote(actual_table)} "
                    f"({', '.join(_quote(column) for column in actual_columns)})"
                )
                indexed.add(column_set)
                if is_unique:
                    unique.add(column_set)


def apply_v0_to_v1(connection: sqlite3.Connection, metadata: MetaData) -> None:
    """Validate and apply only additive current-schema operations."""
    version_row = connection.execute("PRAGMA user_version").fetchone()
    version = int(version_row[0]) if version_row else 0
    if version > 1:
        raise MigrationError("schema_too_new")
    if version < 0:
        raise MigrationError("unsupported_legacy_schema")
    validate_schema_compatibility(connection, metadata, version=version)
    if version == 1:
        return

    missing_columns = _missing_columns(connection, metadata)
    existing_tables = _table_name_map(connection)
    dialect = sqlite_dialect()
    if connection.in_transaction:
        raise MigrationError("migration_failed")
    try:
        connection.execute("BEGIN IMMEDIATE")
        _create_missing_tables(connection, metadata, existing_tables)
        for table_name, column in missing_columns:
            column_ddl = str(CreateColumn(column).compile(dialect=dialect))
            connection.execute(
                f"ALTER TABLE {_quote(table_name)} ADD COLUMN {column_ddl}"
            )
        _create_missing_indexes(connection, metadata)
        connection.execute("PRAGMA user_version=1")
        connection.commit()
    except MigrationError:
        connection.rollback()
        raise
    except sqlite3.Error as exc:
        connection.rollback()
        raise MigrationError("migration_failed") from exc


def _typed_primary_key(value: object) -> dict[str, str]:
    if isinstance(value, int) and not isinstance(value, bool):
        return {"type": "integer", "value": str(value)}
    if isinstance(value, str):
        return {"type": "text", "value": value}
    raise MigrationError("migration_verification_failed")


def _identity_digest(
    table: str, primary_key_columns: tuple[str, ...], rows: Iterable[tuple[object, ...]]
) -> str:
    encoded_rows = []
    for row in rows:
        encoded = [_typed_primary_key(value) for value in row]
        encoded_bytes = json.dumps(
            encoded, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        encoded_rows.append((encoded_bytes, encoded))
    encoded_rows.sort(key=lambda item: item[0])
    payload = {
        "primary_key_columns": list(primary_key_columns),
        "rows": [item[1] for item in encoded_rows],
        "table": table,
    }
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _primary_key_columns(
    info: dict[str, tuple[str, int]],
) -> tuple[str, ...]:
    positioned = sorted(
        ((position, column) for column, (_, position) in info.items() if position > 0)
    )
    if not positioned or [position for position, _ in positioned] != list(
        range(1, len(positioned) + 1)
    ):
        raise MigrationError("unsupported_legacy_schema")
    return tuple(column for _, column in positioned)


def critical_identity_snapshot(
    connection: sqlite3.Connection,
    metadata: MetaData,
) -> dict[str, TableIdentity]:
    """Capture identities using exact ordered primary keys from current metadata."""
    expected = _scratch_schema(metadata)
    expected_tables = _identifier_map(expected)
    observed_tables = _table_name_map(connection)
    snapshot: dict[str, TableIdentity] = {}
    for table in CRITICAL_TABLES:
        normalized_table = _normalized_identifier(table)
        expected_table = expected_tables.get(normalized_table)
        if expected_table is None:
            raise MigrationError("unsupported_legacy_schema")
        actual_table = observed_tables.get(normalized_table)
        if actual_table is None:
            continue

        expected_info = expected[expected_table]
        observed_info = _table_info(connection, actual_table)
        expected_primary_key = _primary_key_columns(expected_info)
        observed_primary_key = _primary_key_columns(observed_info)
        if tuple(map(_normalized_identifier, observed_primary_key)) != tuple(
            map(_normalized_identifier, expected_primary_key)
        ):
            raise MigrationError("unsupported_legacy_schema")

        observed_columns = _identifier_map(observed_info)
        try:
            selected_columns = tuple(
                observed_columns[_normalized_identifier(column)]
                for column in expected_primary_key
            )
        except KeyError as exc:
            raise MigrationError("unsupported_legacy_schema") from exc
        quoted_columns = ", ".join(_quote(column) for column in selected_columns)
        try:
            rows = connection.execute(
                f"SELECT {quoted_columns} FROM {_quote(actual_table)}"
            ).fetchall()
            row_count_row = connection.execute(
                f"SELECT COUNT(*) FROM {_quote(actual_table)}"
            ).fetchone()
        except sqlite3.Error as exc:
            raise MigrationError("unsupported_legacy_schema") from exc
        if row_count_row is None:
            raise MigrationError("unsupported_legacy_schema")
        row_count = int(row_count_row[0])
        canonical_primary_key = tuple(expected_primary_key)
        snapshot[table] = TableIdentity(
            table=table,
            row_count=row_count,
            primary_key_columns=canonical_primary_key,
            primary_key_sha256=_identity_digest(
                table, canonical_primary_key, rows
            ),
        )
    return snapshot


SchemaIdentifiers: TypeAlias = dict[str, tuple[str, dict[str, str]]]


def _schema_identifiers(connection: sqlite3.Connection) -> SchemaIdentifiers:
    schemas: SchemaIdentifiers = {}
    for actual_table in _table_names(connection):
        canonical_table = _normalized_identifier(actual_table)
        columns = _identifier_map(_table_info(connection, actual_table))
        schemas[canonical_table] = (actual_table, columns)
    return schemas


def _has_columns(
    schemas: SchemaIdentifiers, table: str, columns: set[str]
) -> bool:
    schema = schemas.get(_normalized_identifier(table))
    if schema is None:
        return False
    return {_normalized_identifier(column) for column in columns} <= schema[1].keys()


def _require_no_rows(connection: sqlite3.Connection, query: str) -> None:
    if connection.execute(query).fetchone() is not None:
        raise MigrationError("migration_verification_failed")


def validate_critical_references(connection: sqlite3.Connection) -> None:
    """Validate declared and D34 semantic ownership references when present."""
    if connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
        raise MigrationError("migration_verification_failed")
    schemas = _schema_identifiers(connection)

    def table(name: str, alias: str) -> str:
        return f"{_quote(schemas[_normalized_identifier(name)][0])} {alias}"

    def column(table_name: str, name: str, alias: str) -> str:
        columns = schemas[_normalized_identifier(table_name)][1]
        return f"{alias}.{_quote(columns[_normalized_identifier(name)])}"

    blocks_requirements = {
        ("blocks", frozenset({"project_id"})),
        ("projects", frozenset({"id"})),
    }
    if all(_has_columns(schemas, name, set(columns)) for name, columns in blocks_requirements):
        _require_no_rows(
            connection,
            f"SELECT 1 FROM {table('blocks', 'b')} LEFT JOIN {table('projects', 'p')} "
            f"ON {column('projects', 'id', 'p')}={column('blocks', 'project_id', 'b')} "
            f"WHERE {column('projects', 'id', 'p')} IS NULL LIMIT 1",
        )

    jobs_requirements = {
        ("generation_jobs", frozenset({"id", "project_id", "parent_job_id"})),
        ("projects", frozenset({"id"})),
    }
    if all(_has_columns(schemas, name, set(columns)) for name, columns in jobs_requirements):
        _require_no_rows(
            connection,
            f"SELECT 1 FROM {table('generation_jobs', 'j')} "
            f"LEFT JOIN {table('projects', 'p')} ON {column('projects', 'id', 'p')}="
            f"{column('generation_jobs', 'project_id', 'j')} "
            f"LEFT JOIN {table('generation_jobs', 'parent')} ON "
            f"{column('generation_jobs', 'id', 'parent')}="
            f"{column('generation_jobs', 'parent_job_id', 'j')} WHERE "
            f"{column('projects', 'id', 'p')} IS NULL OR "
            f"({column('generation_jobs', 'parent_job_id', 'j')} IS NOT NULL AND "
            f"({column('generation_jobs', 'id', 'parent')} IS NULL OR "
            f"{column('generation_jobs', 'project_id', 'parent')}<>"
            f"{column('generation_jobs', 'project_id', 'j')})) LIMIT 1",
        )

    calls_requirements = {
        ("external_calls", frozenset({"job_id"})),
        ("generation_jobs", frozenset({"id"})),
    }
    if all(_has_columns(schemas, name, set(columns)) for name, columns in calls_requirements):
        _require_no_rows(
            connection,
            f"SELECT 1 FROM {table('external_calls', 'c')} "
            f"LEFT JOIN {table('generation_jobs', 'j')} ON "
            f"{column('generation_jobs', 'id', 'j')}={column('external_calls', 'job_id', 'c')} "
            f"WHERE {column('generation_jobs', 'id', 'j')} IS NULL LIMIT 1",
        )

    artifacts_requirements = {
        ("generation_artifacts", frozenset({"project_id", "job_id"})),
        ("projects", frozenset({"id"})),
        ("generation_jobs", frozenset({"id", "project_id"})),
    }
    if all(_has_columns(schemas, name, set(columns)) for name, columns in artifacts_requirements):
        _require_no_rows(
            connection,
            f"SELECT 1 FROM {table('generation_artifacts', 'a')} "
            f"LEFT JOIN {table('projects', 'p')} ON {column('projects', 'id', 'p')}="
            f"{column('generation_artifacts', 'project_id', 'a')} "
            f"LEFT JOIN {table('generation_jobs', 'j')} ON {column('generation_jobs', 'id', 'j')}="
            f"{column('generation_artifacts', 'job_id', 'a')} WHERE "
            f"{column('projects', 'id', 'p')} IS NULL OR "
            f"({column('generation_artifacts', 'job_id', 'a')} IS NOT NULL AND "
            f"({column('generation_jobs', 'id', 'j')} IS NULL OR "
            f"{column('generation_jobs', 'project_id', 'j')}<>"
            f"{column('generation_artifacts', 'project_id', 'a')})) LIMIT 1",
        )

    revisions_requirements = {
        ("settings_revisions", frozenset({"id", "project_id", "revision", "restored_from_revision"})),
        ("projects", frozenset({"id"})),
    }
    if all(_has_columns(schemas, name, set(columns)) for name, columns in revisions_requirements):
        _require_no_rows(
            connection,
            f"SELECT 1 FROM {table('settings_revisions', 's')} "
            f"LEFT JOIN {table('projects', 'p')} ON {column('projects', 'id', 'p')}="
            f"{column('settings_revisions', 'project_id', 's')} "
            f"LEFT JOIN {table('settings_revisions', 'source')} ON "
            f"{column('settings_revisions', 'project_id', 'source')}="
            f"{column('settings_revisions', 'project_id', 's')} AND "
            f"{column('settings_revisions', 'revision', 'source')}="
            f"{column('settings_revisions', 'restored_from_revision', 's')} WHERE "
            f"{column('projects', 'id', 'p')} IS NULL OR "
            f"({column('settings_revisions', 'restored_from_revision', 's')} IS NOT NULL AND "
            f"{column('settings_revisions', 'id', 'source')} IS NULL) LIMIT 1",
        )

    current_artifact_requirements = {
        ("projects", frozenset({"id", "current_artifact_id"})),
        ("generation_artifacts", frozenset({"id", "project_id"})),
    }
    if all(_has_columns(schemas, name, set(columns)) for name, columns in current_artifact_requirements):
        _require_no_rows(
            connection,
            f"SELECT 1 FROM {table('projects', 'p')} "
            f"LEFT JOIN {table('generation_artifacts', 'a')} ON "
            f"{column('generation_artifacts', 'id', 'a')}="
            f"{column('projects', 'current_artifact_id', 'p')} WHERE "
            f"{column('projects', 'current_artifact_id', 'p')} IS NOT NULL AND "
            f"({column('generation_artifacts', 'id', 'a')} IS NULL OR "
            f"{column('generation_artifacts', 'project_id', 'a')}<>"
            f"{column('projects', 'id', 'p')}) LIMIT 1",
        )

    receipt_requirements = {
        ("operation_requests", frozenset({"project_id", "job_id"})),
        ("generation_jobs", frozenset({"id", "project_id"})),
    }
    if all(_has_columns(schemas, name, set(columns)) for name, columns in receipt_requirements):
        _require_no_rows(
            connection,
            f"SELECT 1 FROM {table('operation_requests', 'r')} "
            f"JOIN {table('generation_jobs', 'j')} ON {column('generation_jobs', 'id', 'j')}="
            f"{column('operation_requests', 'job_id', 'r')} WHERE "
            f"{column('operation_requests', 'project_id', 'r')} IS NULL OR "
            f"{column('generation_jobs', 'project_id', 'j')}<>"
            f"{column('operation_requests', 'project_id', 'r')} LIMIT 1",
        )

    language_requirements = {
        ("language_requests", frozenset({"project_id", "core_request_id"})),
        ("operation_requests", frozenset({"request_id", "project_id"})),
    }
    if all(_has_columns(schemas, name, set(columns)) for name, columns in language_requirements):
        _require_no_rows(
            connection,
            f"SELECT 1 FROM {table('language_requests', 'l')} "
            f"LEFT JOIN {table('operation_requests', 'r')} ON "
            f"{column('operation_requests', 'request_id', 'r')}="
            f"{column('language_requests', 'core_request_id', 'l')} WHERE "
            f"{column('operation_requests', 'request_id', 'r')} IS NOT NULL AND "
            f"({column('language_requests', 'project_id', 'l')} IS NULL OR "
            f"{column('operation_requests', 'project_id', 'r')} IS NULL OR "
            f"{column('operation_requests', 'project_id', 'r')}<>"
            f"{column('language_requests', 'project_id', 'l')}) LIMIT 1",
        )

    turn_columns = {"request_id", "parent_request_id", "successor_request_id"}
    if _has_columns(schemas, "language_turns", turn_columns) and _has_columns(
        schemas, "language_requests", {"request_id"}
    ):
        def turn_request(alias: str) -> str:
            return column("language_turns", "request_id", alias)

        def request_id(alias: str) -> str:
            return column("language_requests", "request_id", alias)

        def parent(alias: str) -> str:
            return column("language_turns", "parent_request_id", alias)

        def successor(alias: str) -> str:
            return column("language_turns", "successor_request_id", alias)

        _require_no_rows(
            connection,
            f"SELECT 1 FROM {table('language_turns', 't')} "
            f"LEFT JOIN {table('language_requests', 'own')} ON {request_id('own')}={turn_request('t')} "
            f"LEFT JOIN {table('language_requests', 'parent_request')} ON "
            f"{request_id('parent_request')}={parent('t')} "
            f"LEFT JOIN {table('language_turns', 'parent_turn')} ON "
            f"{turn_request('parent_turn')}={parent('t')} "
            f"LEFT JOIN {table('language_requests', 'successor_request')} ON "
            f"{request_id('successor_request')}={successor('t')} "
            f"LEFT JOIN {table('language_turns', 'successor_turn')} ON "
            f"{turn_request('successor_turn')}={successor('t')} WHERE "
            f"{request_id('own')} IS NULL OR ({parent('t')} IS NOT NULL AND "
            f"({request_id('parent_request')} IS NULL OR {turn_request('parent_turn')} IS NULL OR "
            f"{successor('parent_turn')} IS NULL OR {successor('parent_turn')}<>{turn_request('t')})) OR "
            f"({successor('t')} IS NOT NULL AND ({request_id('successor_request')} IS NULL OR "
            f"{turn_request('successor_turn')} IS NULL OR {parent('successor_turn')} IS NULL OR "
            f"{parent('successor_turn')}<>{turn_request('t')})) LIMIT 1",
        )
