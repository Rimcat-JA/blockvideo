"""D34 versioned SQLite migration and historical compatibility tests."""
from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
import stat
import subprocess
import sys
import time
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import (
    Boolean,
    Column,
    Float,
    Integer,
    LargeBinary,
    MetaData,
    String,
    Table,
    create_engine,
)

from app.migrations.backup import backup_metadata_path, sha256_file
from app.migrations.contracts import MigrationError, MigrationResult, TableIdentity
from app.migrations.lease import acquire_database_lease
from app.migrations.runner import migrate_database, restore_database_backup
from app.migrations.schema import (
    apply_v0_to_v1,
    classify_v0,
    critical_identity_snapshot,
    sqlite_affinity,
    validate_critical_references,
    validate_schema_compatibility,
)
from app.db import Base, init_db, register_models, reset_db_for_tests
from tests.fixtures.migrations.build_fixtures import (
    FIXTURE_BUILDERS,
    build_affinity_collision,
    build_altered_legacy_primary_key,
    build_fixture,
    build_matching_affinity_aliases,
)


EXPECTED_TABLES = {
    "blocks",
    "external_calls",
    "generation_artifacts",
    "generation_jobs",
    "language_requests",
    "language_turns",
    "operation_requests",
    "project_identities",
    "projects",
    "settings_revisions",
}

_LEASE_HOLDER = r"""
from pathlib import Path
import sys
import time
from app.migrations.lease import acquire_database_lease

lease = acquire_database_lease(sys.argv[1])
Path(sys.argv[2]).write_text("ready", encoding="ascii")
try:
    deadline = time.monotonic() + 30
    while not Path(sys.argv[3]).exists():
        if time.monotonic() >= deadline:
            raise SystemExit("lease holder timed out")
        time.sleep(0.01)
finally:
    lease.release()
"""


AFFINITY_COLLISIONS = (
    ("projects", "id", "VARCHAR(32)", "101"),
    ("projects", "title", "INT", "'coercible text'"),
    ("external_calls", "response_body", "TEXT", "x'31'"),
    ("projects", "progress", "DECIMAL(8,2)", "1.5"),
    ("projects", "subtitle_enabled", "REAL", "1"),
)


@pytest.fixture(scope="module", autouse=True)
def registered_models() -> None:
    register_models()


@pytest.fixture()
def historical_db(tmp_path: Path, request: pytest.FixtureRequest) -> Path:
    return build_fixture(request.param, tmp_path / f"{request.param}.db", Base.metadata)


def _schema(connection: sqlite3.Connection) -> dict[str, dict[str, str]]:
    tables = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
    }
    return {
        table: {
            row[1]: sqlite_affinity(row[2])
            for row in connection.execute(f'PRAGMA table_info("{table}")')
        }
        for table in tables
    }


def _dump(connection: sqlite3.Connection) -> tuple[str, ...]:
    return tuple(connection.iterdump())


def _index_sets(
    connection: sqlite3.Connection, table: str
) -> tuple[set[frozenset[str]], set[frozenset[str]]]:
    indexed: set[frozenset[str]] = set()
    unique: set[frozenset[str]] = set()
    for row in connection.execute(f'PRAGMA index_list("{table}")'):
        if int(row[4]):
            continue
        columns = frozenset(
            str(info[2]).lower()
            for info in connection.execute(f'PRAGMA index_info("{row[1]}")')
            if info[2] is not None
        )
        indexed.add(columns)
        if int(row[2]):
            unique.add(columns)
    return indexed, unique


def _url(path: Path) -> str:
    return f"sqlite:///{path.as_posix()}"


def _published_backups(root: Path) -> tuple[Path, ...]:
    return tuple(
        path
        for path in root.iterdir()
        if not path.name.endswith(".metadata.json")
    )


def _migrate(path: Path) -> MigrationResult:
    url = _url(path)
    lease = acquire_database_lease(url)
    try:
        return migrate_database(url, Base.metadata, lease=lease)
    finally:
        lease.release()


def _start_lease_holder(
    database: Path, ready: Path, release: Path
) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [sys.executable, "-c", _LEASE_HOLDER, _url(database), str(ready), str(release)],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )


def _wait_for(path: Path, process: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + 30
    while not path.exists():
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            raise AssertionError(f"lease holder exited: {stdout}\n{stderr}")
        if time.monotonic() >= deadline:
            raise AssertionError("lease holder did not become ready")
        time.sleep(0.01)


def test_classify_sqlite_affinity_uses_ordered_sqlite_rules() -> None:
    assert sqlite_affinity("UNSIGNED BIG INT") == "INTEGER"
    assert sqlite_affinity("VARCHAR(255)") == "TEXT"
    assert sqlite_affinity("CLOB") == "TEXT"
    assert sqlite_affinity("") == "BLOB"
    assert sqlite_affinity("DOUBLE PRECISION") == "REAL"
    assert sqlite_affinity("FLOAT") == "REAL"
    assert sqlite_affinity("BOOLEAN") == "NUMERIC"
    assert sqlite_affinity("DECIMAL(8,2)") == "NUMERIC"
    assert sqlite_affinity("BLOBBER") == "BLOB"


def test_sqlite_affinity_does_not_unicode_fold_dotless_i_to_ascii_int() -> None:
    declared_type = "\u0131nt"
    with sqlite3.connect(":memory:") as connection:
        storage_class, value = connection.execute(
            f'SELECT typeof(CAST(7.5 AS "{declared_type}")), '
            f'CAST(7.5 AS "{declared_type}")'
        ).fetchone()

    assert (storage_class, value) == ("real", 7.5)
    assert sqlite_affinity(declared_type) == "NUMERIC"


@pytest.mark.parametrize("fixture_name", ("upstream_v0", "d30_v0", "partially_additive_v0"))
def test_classify_supported_v0_uses_scratch_metadata(
    tmp_path: Path, fixture_name: str
) -> None:
    path = build_fixture(fixture_name, tmp_path / f"{fixture_name}.db", Base.metadata)
    with sqlite3.connect(path) as connection:
        classify_v0(connection, Base.metadata)


@pytest.mark.parametrize(
    ("table", "column", "declared_type", "value_sql"), AFFINITY_COLLISIONS
)
def test_classify_rejects_each_known_affinity_name_collision(
    tmp_path: Path,
    table: str,
    column: str,
    declared_type: str,
    value_sql: str,
) -> None:
    path = tmp_path / f"{table}-{column}.db"
    build_affinity_collision(
        path,
        Base.metadata,
        table=table,
        column=column,
        declared_type=declared_type,
        value_sql=value_sql,
    )
    before = path.read_bytes()

    with sqlite3.connect(path) as connection:
        with pytest.raises(MigrationError) as exc_info:
            classify_v0(connection, Base.metadata)

    assert exc_info.value.reason_code == "unsupported_legacy_schema"
    assert path.read_bytes() == before


def test_classify_accepts_declared_type_aliases_with_equal_affinity(tmp_path: Path) -> None:
    path = tmp_path / "affinity-aliases.db"
    build_matching_affinity_aliases(path, Base.metadata)
    metadata = MetaData()
    Table(
        "projects",
        metadata,
        Column("id", Integer),
        Column("title", String),
        Column("progress", Float),
        Column("subtitle_enabled", Boolean),
    )
    Table("external_calls", metadata, Column("response_body", LargeBinary))
    with sqlite3.connect(path) as connection:
        classify_v0(connection, metadata)


@pytest.mark.parametrize("fixture_name", ("upstream_v0", "d30_v0", "partially_additive_v0"))
def test_schema_compatibility_accepts_supported_v0(
    tmp_path: Path, fixture_name: str
) -> None:
    path = build_fixture(fixture_name, tmp_path / f"{fixture_name}.db", Base.metadata)
    with sqlite3.connect(path) as connection:
        validate_schema_compatibility(connection, Base.metadata, version=0)


def test_schema_compatibility_rejects_v0_table_missing_required_column(
    tmp_path: Path,
) -> None:
    path = build_fixture("current_v1", tmp_path / "missing-required.db", Base.metadata)
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA user_version=0")
        connection.execute('ALTER TABLE projects DROP COLUMN "title"')
        with pytest.raises(MigrationError) as exc_info:
            validate_schema_compatibility(connection, Base.metadata, version=0)

    assert exc_info.value.reason_code == "unsupported_legacy_schema"


def test_schema_compatibility_rejects_v1_missing_required_unique_index(
    tmp_path: Path,
) -> None:
    path = build_fixture(
        "partial_language_parent_v0", tmp_path / "missing-unique.db", Base.metadata
    )
    with sqlite3.connect(path) as connection:
        connection.execute(
            'ALTER TABLE language_turns ADD COLUMN "parent_request_id" VARCHAR(128)'
        )
        connection.execute("PRAGMA user_version=1")
        with pytest.raises(MigrationError) as exc_info:
            validate_schema_compatibility(connection, Base.metadata, version=1)

    assert exc_info.value.reason_code == "unsupported_legacy_schema"


def test_schema_compatibility_rejects_v1_missing_required_ordinary_index(
    tmp_path: Path,
) -> None:
    path = build_fixture("current_v1", tmp_path / "missing-index.db", Base.metadata)
    with sqlite3.connect(path) as connection:
        connection.execute("DROP INDEX ix_blocks_project_id")
        with pytest.raises(MigrationError) as exc_info:
            validate_schema_compatibility(connection, Base.metadata, version=1)

    assert exc_info.value.reason_code == "unsupported_legacy_schema"


def test_schema_compatibility_matches_index_columns_case_insensitively(
    tmp_path: Path,
) -> None:
    path = build_fixture(
        "partial_language_parent_v0", tmp_path / "mixed-case-index.db", Base.metadata
    )
    with sqlite3.connect(path) as connection:
        connection.execute(
            'ALTER TABLE language_turns ADD COLUMN "Parent_Request_ID" VARCHAR(128)'
        )
        connection.execute(
            'CREATE UNIQUE INDEX "Mixed_Case_Parent" '
            'ON language_turns ("Parent_Request_ID")'
        )
        connection.execute("PRAGMA user_version=1")
        validate_schema_compatibility(connection, Base.metadata, version=1)


def test_distinct_unknown_unicode_identifiers_are_preserved(tmp_path: Path) -> None:
    path = build_fixture("d30_v0", tmp_path / "unicode-identifiers.db", Base.metadata)
    with sqlite3.connect(path) as connection:
        connection.executescript(
            'CREATE TABLE "Ä" ("Ö" TEXT);'
            'CREATE TABLE "ä" ("ö" TEXT);'
            'ALTER TABLE projects ADD COLUMN "Ü" TEXT;'
            'ALTER TABLE projects ADD COLUMN "ü" TEXT;'
        )
        classify_v0(connection, Base.metadata)
        apply_v0_to_v1(connection, Base.metadata)
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        project_columns = {
            row[1] for row in connection.execute('PRAGMA table_info("projects")')
        }

    assert {"Ä", "ä"} <= tables
    assert {"Ü", "ü"} <= project_columns


def test_case_insensitive_known_identifiers_are_classified_and_extended(
    tmp_path: Path,
) -> None:
    path = build_fixture("d30_v0", tmp_path / "mixed-case.db", Base.metadata)
    with sqlite3.connect(path) as connection:
        connection.execute('ALTER TABLE projects RENAME TO temporary_projects')
        connection.execute('ALTER TABLE temporary_projects RENAME TO "Projects"')
        connection.execute('ALTER TABLE "Projects" RENAME COLUMN id TO temporary_id')
        connection.execute('ALTER TABLE "Projects" RENAME COLUMN temporary_id TO "ID"')
        connection.execute('ALTER TABLE "Projects" DROP COLUMN current_artifact_id')
        connection.execute("UPDATE \"Projects\" SET title='mixed' WHERE \"ID\"=101")
        connection.commit()

        classify_v0(connection, Base.metadata)
        apply_v0_to_v1(connection, Base.metadata)
        project_tables = [
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND lower(name)='projects'"
            )
        ]
        columns = {
            row[1].casefold()
            for row in connection.execute('PRAGMA table_info("Projects")')
        }
        assert project_tables == ["Projects"]
        assert {column.name.casefold() for column in Base.metadata.tables["projects"].columns} <= columns
        assert connection.execute('SELECT "ID", "Title" FROM "Projects"').fetchone() == (
            101,
            "mixed",
        )


@pytest.mark.parametrize("duplicate_kind", ("table", "column"))
def test_case_colliding_known_metadata_identifiers_are_rejected_before_ddl(
    duplicate_kind: str,
) -> None:
    metadata = MetaData()
    if duplicate_kind == "table":
        Table("projects", metadata, Column("id", Integer, primary_key=True))
        Table("PROJECTS", metadata, Column("other_id", Integer, primary_key=True))
    else:
        Table(
            "projects",
            metadata,
            Column("id", Integer, primary_key=True),
            Column("ID", Integer),
        )

    with sqlite3.connect(":memory:") as connection:
        with pytest.raises(MigrationError) as exc_info:
            apply_v0_to_v1(connection, metadata)
        assert not _schema(connection)
    assert exc_info.value.reason_code == "unsupported_legacy_schema"


@pytest.mark.parametrize("fixture_name", tuple(FIXTURE_BUILDERS))
def test_schema_version_fixtures_are_deterministic(
    tmp_path: Path, fixture_name: str
) -> None:
    first = build_fixture(fixture_name, tmp_path / "first.db", Base.metadata)
    second = build_fixture(fixture_name, tmp_path / "second.db", Base.metadata)
    with sqlite3.connect(first) as first_connection, sqlite3.connect(second) as second_connection:
        assert _dump(first_connection) == _dump(second_connection)


@pytest.mark.parametrize("fixture_name", ("upstream_v0", "d30_v0"))
def test_historical_fixture_ddl_is_independent_of_current_metadata(
    tmp_path: Path, fixture_name: str
) -> None:
    path = build_fixture(fixture_name, tmp_path / f"{fixture_name}.db", MetaData())
    with sqlite3.connect(path) as connection:
        assert "projects" in _schema(connection)
        assert connection.execute("SELECT title FROM projects WHERE id=101").fetchone() == (
            "fixture-project",
        )


@pytest.mark.parametrize(
    "historical_db",
    ("empty_v0", "upstream_v0", "d30_v0", "partially_additive_v0"),
    indirect=True,
)
def test_additive_v0_to_v1_matches_scratch_schema_and_preserves_unknown_data(
    historical_db: Path,
) -> None:
    scratch = historical_db.with_name("scratch.db")
    engine = create_engine(f"sqlite:///{scratch.as_posix()}")
    try:
        Base.metadata.create_all(engine)
    finally:
        engine.dispose()

    with sqlite3.connect(historical_db) as connection:
        before_tables = _schema(connection)
        legacy_rows = (
            connection.execute("SELECT * FROM legacy_notes").fetchall()
            if "legacy_notes" in before_tables
            else None
        )
        apply_v0_to_v1(connection, Base.metadata)
        migrated = _schema(connection)
        assert connection.execute("PRAGMA user_version").fetchone() == (1,)

    with sqlite3.connect(scratch) as connection:
        expected = _schema(connection)

    assert set(expected) <= set(migrated)
    for table, columns in expected.items():
        assert columns.items() <= migrated[table].items()
    for table, columns in before_tables.items():
        assert columns.items() <= migrated[table].items()
    if legacy_rows is not None:
        with sqlite3.connect(historical_db) as connection:
            assert connection.execute("SELECT * FROM legacy_notes").fetchall() == legacy_rows
            assert connection.execute(
                "SELECT legacy_marker FROM projects WHERE id = 101"
            ).fetchone() == ("keep-upstream",)
    if "partial_extra" in before_tables.get("projects", {}):
        with sqlite3.connect(historical_db) as connection:
            assert connection.execute(
                "SELECT partial_extra FROM projects WHERE id = 101"
            ).fetchone() == ("keep-partial",)


@pytest.mark.parametrize(
    ("fixture_name", "table", "column_set", "first_insert", "duplicate_insert"),
    (
        (
            "partial_artifact_job_v0",
            "generation_artifacts",
            frozenset({"job_id"}),
            "INSERT INTO generation_artifacts "
            "(id, project_id, job_id, video_path, manifest_json, created_at) "
            "VALUES (1, 101, 900, 'one.mp4', '{}', '2026-01-02 03:04:05')",
            "INSERT INTO generation_artifacts "
            "(id, project_id, job_id, video_path, manifest_json, created_at) "
            "VALUES (2, 101, 900, 'two.mp4', '{}', '2026-01-02 03:04:05')",
        ),
        (
            "partial_operation_job_v0",
            "operation_requests",
            frozenset({"job_id"}),
            "INSERT INTO operation_requests VALUES "
            "('one', '{}', 'project.status.get', 1, 101, 1, 1, '{}', 0, 900, "
            "'receipt:one', '{}', '2026-01-02 03:04:05')",
            "INSERT INTO operation_requests VALUES "
            "('two', '{}', 'project.status.get', 1, 101, 1, 1, '{}', 0, 900, "
            "'receipt:two', '{}', '2026-01-02 03:04:05')",
        ),
        (
            "partial_language_parent_v0",
            "language_turns",
            frozenset({"parent_request_id"}),
            "INSERT INTO language_turns "
            "(request_id, parent_request_id, text) VALUES ('one', 'parent', 'one')",
            "INSERT INTO language_turns "
            "(request_id, parent_request_id, text) VALUES ('two', 'parent', 'two')",
        ),
    ),
)
def test_partial_v0_migration_restores_unique_indexes_and_enforcement(
    tmp_path: Path,
    fixture_name: str,
    table: str,
    column_set: frozenset[str],
    first_insert: str,
    duplicate_insert: str,
) -> None:
    path = build_fixture(fixture_name, tmp_path / f"{fixture_name}.db", Base.metadata)

    result = _migrate(path)

    assert result.status == "migrated"
    with sqlite3.connect(path) as connection:
        validate_schema_compatibility(connection, Base.metadata, version=1)
        indexed, unique = _index_sets(connection, table)
        assert column_set in indexed
        assert column_set in unique
        connection.execute(first_insert)
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(duplicate_insert)


def test_additive_index_reconciliation_preserves_existing_weaker_and_extra_indexes(
    tmp_path: Path,
) -> None:
    path = build_fixture(
        "partial_artifact_job_v0", tmp_path / "preserve-indexes.db", Base.metadata
    )
    with sqlite3.connect(path) as connection:
        connection.execute(
            'ALTER TABLE generation_artifacts ADD COLUMN "job_id" INTEGER'
        )
        connection.execute(
            "CREATE INDEX legacy_artifact_job_lookup "
            "ON generation_artifacts (job_id)"
        )
        connection.execute(
            "CREATE INDEX legacy_artifact_video_lookup "
            "ON generation_artifacts (video_path)"
        )

    _migrate(path)

    with sqlite3.connect(path) as connection:
        index_names = {
            str(row[1])
            for row in connection.execute(
                'PRAGMA index_list("generation_artifacts")'
            )
        }
        indexed, unique = _index_sets(connection, "generation_artifacts")
    assert "legacy_artifact_job_lookup" in index_names
    assert "legacy_artifact_video_lookup" in index_names
    assert frozenset({"job_id"}) in indexed
    assert frozenset({"job_id"}) in unique
    assert frozenset({"video_path"}) in indexed


def test_schema_version_v1_is_classified_without_mutation(tmp_path: Path) -> None:
    path = build_fixture("current_v1", tmp_path / "current.db", Base.metadata)
    with sqlite3.connect(path) as connection:
        before = _dump(connection)
        apply_v0_to_v1(connection, Base.metadata)
        assert _dump(connection) == before
        assert connection.execute("PRAGMA user_version").fetchone() == (1,)


def test_schema_version_v2_fails_without_mutation(tmp_path: Path) -> None:
    path = build_fixture("newer_v2", tmp_path / "newer.db", Base.metadata)
    before = path.read_bytes()
    with sqlite3.connect(path) as connection:
        with pytest.raises(MigrationError) as exc_info:
            apply_v0_to_v1(connection, Base.metadata)
    assert exc_info.value.reason_code == "schema_too_new"
    assert path.read_bytes() == before


def test_additive_affinity_failure_precedes_ddl(tmp_path: Path) -> None:
    path = tmp_path / "collision.db"
    build_affinity_collision(
        path,
        Base.metadata,
        table="projects",
        column="id",
        declared_type="TEXT",
        value_sql="101",
    )
    with sqlite3.connect(path) as connection:
        before = _dump(connection)
        with pytest.raises(MigrationError) as exc_info:
            apply_v0_to_v1(connection, Base.metadata)
        assert _dump(connection) == before
        assert connection.execute("PRAGMA user_version").fetchone() == (0,)
    assert exc_info.value.reason_code == "unsupported_legacy_schema"


def test_additive_register_models_has_no_database_side_effect(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    database = tmp_path / "not-created.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{database.as_posix()}")
    table_names = tuple(Base.metadata.tables)
    register_models()
    register_models()
    assert tuple(Base.metadata.tables) == table_names
    assert set(table_names) == EXPECTED_TABLES
    assert not database.exists()


def test_additive_init_db_only_creates_registered_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "init.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{database.as_posix()}")
    from app.core import config

    config.reset_settings_cache()
    reset_db_for_tests()
    try:
        init_db()
        with sqlite3.connect(database) as connection:
            assert set(_schema(connection)) == EXPECTED_TABLES
            assert connection.execute("PRAGMA user_version").fetchone() == (0,)
    finally:
        reset_db_for_tests()
        config.reset_settings_cache()


def test_critical_identity_uses_current_metadata_pk_order(tmp_path: Path) -> None:
    path = build_fixture("d30_v0", tmp_path / "identity.db", Base.metadata)
    with sqlite3.connect(path) as connection:
        snapshot = critical_identity_snapshot(connection, Base.metadata)
    assert snapshot["projects"].primary_key_columns == ("id",)


def test_critical_identity_hashes_text_keys_with_canonical_byte_sorting(
    tmp_path: Path,
) -> None:
    path = build_fixture("current_v1", tmp_path / "text-identities.db", Base.metadata)
    with sqlite3.connect(path) as connection:
        connection.executemany(
            "INSERT INTO language_requests ("
            "request_id, input_fingerprint, core_request_id, status, owner_token, "
            "lease_until, created_at, response_json"
            ") VALUES (?, 'fingerprint', ?, 'completed', 'owner', 0, 0, '{}')",
            (("ä", "core-a"), ("z", "core-z")),
        )
        snapshot = critical_identity_snapshot(connection, Base.metadata)

    assert snapshot["language_requests"].primary_key_sha256 == (
        "ec2cdc2bed0f0d7b3dd202ad7808c074c495a08bd07d05d42356fa7b85c70153"
    )


def test_critical_identity_rejects_altered_legacy_primary_key(tmp_path: Path) -> None:
    path = tmp_path / "altered-pk.db"
    build_altered_legacy_primary_key(path, Base.metadata)
    with sqlite3.connect(path) as connection:
        with pytest.raises(MigrationError) as exc_info:
            critical_identity_snapshot(connection, Base.metadata)
    assert exc_info.value.reason_code == "unsupported_legacy_schema"


def test_critical_identity_rejects_missing_expected_primary_key(tmp_path: Path) -> None:
    path = tmp_path / "missing-pk.db"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE projects (title TEXT PRIMARY KEY)")
        with pytest.raises(MigrationError) as exc_info:
            critical_identity_snapshot(connection, Base.metadata)
    assert exc_info.value.reason_code == "unsupported_legacy_schema"


MIXED_CASE_MALFORMED_REFERENCE_SCHEMAS = (
    (
        'CREATE TABLE "Blocks" ("Project_ID" INTEGER);'
        'CREATE TABLE "Projects" ("ID" INTEGER);'
        'INSERT INTO "Blocks" VALUES (1);',
    ),
    (
        'CREATE TABLE "Generation_Jobs" ('
        '"ID" INTEGER, "Project_ID" INTEGER, "Parent_Job_ID" INTEGER);'
        'CREATE TABLE "Projects" ("ID" INTEGER);'
        'INSERT INTO "Generation_Jobs" VALUES (1, 1, NULL);',
    ),
    (
        'CREATE TABLE "External_Calls" ("Job_ID" INTEGER);'
        'CREATE TABLE "Generation_Jobs" ("ID" INTEGER);'
        'INSERT INTO "External_Calls" VALUES (1);',
    ),
    (
        'CREATE TABLE "Generation_Artifacts" ("Project_ID" INTEGER, "Job_ID" INTEGER);'
        'CREATE TABLE "Projects" ("ID" INTEGER);'
        'CREATE TABLE "Generation_Jobs" ("ID" INTEGER, "Project_ID" INTEGER);'
        'INSERT INTO "Generation_Artifacts" VALUES (1, NULL);',
    ),
    (
        'CREATE TABLE "Settings_Revisions" ('
        '"ID" INTEGER, "Project_ID" INTEGER, "Revision" INTEGER, '
        '"Restored_From_Revision" INTEGER);'
        'CREATE TABLE "Projects" ("ID" INTEGER);'
        'INSERT INTO "Projects" VALUES (1);'
        'INSERT INTO "Settings_Revisions" VALUES (1, 1, 2, 1);',
    ),
    (
        'CREATE TABLE "Projects" ("ID" INTEGER, "Current_Artifact_ID" INTEGER);'
        'CREATE TABLE "Generation_Artifacts" ("ID" INTEGER, "Project_ID" INTEGER);'
        'INSERT INTO "Projects" VALUES (1, 1);'
        'INSERT INTO "Generation_Artifacts" VALUES (1, 2);',
    ),
    (
        'CREATE TABLE "Operation_Requests" ("Project_ID" INTEGER, "Job_ID" INTEGER);'
        'CREATE TABLE "Generation_Jobs" ("ID" INTEGER, "Project_ID" INTEGER);'
        'INSERT INTO "Operation_Requests" VALUES (1, 1);'
        'INSERT INTO "Generation_Jobs" VALUES (1, 2);',
    ),
    (
        'CREATE TABLE "Language_Requests" ('
        '"Request_ID" TEXT, "Project_ID" INTEGER, "Core_Request_ID" TEXT);'
        'CREATE TABLE "Operation_Requests" ("Request_ID" TEXT, "Project_ID" INTEGER);'
        "INSERT INTO \"Language_Requests\" VALUES ('language', 1, 'core');"
        "INSERT INTO \"Operation_Requests\" VALUES ('core', 2);",
    ),
    (
        'CREATE TABLE "Language_Requests" ("Request_ID" TEXT);'
        'CREATE TABLE "Language_Turns" ('
        '"Request_ID" TEXT, "Parent_Request_ID" TEXT, "Successor_Request_ID" TEXT);'
        "INSERT INTO \"Language_Requests\" VALUES ('parent');"
        "INSERT INTO \"Language_Requests\" VALUES ('child');"
        "INSERT INTO \"Language_Turns\" VALUES ('parent', NULL, NULL);"
        "INSERT INTO \"Language_Turns\" VALUES ('child', 'parent', NULL);",
    ),
)


@pytest.mark.parametrize("schema_sql", MIXED_CASE_MALFORMED_REFERENCE_SCHEMAS)
def test_mixed_case_known_identifiers_execute_every_semantic_reference_check(
    schema_sql: tuple[str],
) -> None:
    with sqlite3.connect(":memory:") as connection:
        connection.executescript(schema_sql[0])
        with pytest.raises(MigrationError) as exc_info:
            validate_critical_references(connection)
    assert exc_info.value.reason_code == "migration_verification_failed"


@pytest.mark.parametrize(
    ("language_project_id", "receipt_project_id"),
    ((None, 1), (2, 1), (1, None)),
)
def test_language_request_with_core_receipt_requires_matching_project(
    language_project_id: int | None,
    receipt_project_id: int | None,
) -> None:
    with sqlite3.connect(":memory:") as connection:
        connection.executescript(
            "CREATE TABLE operation_requests (request_id TEXT PRIMARY KEY, project_id INTEGER);"
            "CREATE TABLE language_requests ("
            "request_id TEXT PRIMARY KEY, core_request_id TEXT, project_id INTEGER);"
        )
        connection.execute(
            "INSERT INTO operation_requests VALUES ('core', ?)",
            (receipt_project_id,),
        )
        connection.execute(
            "INSERT INTO language_requests VALUES ('language', 'core', ?)",
            (language_project_id,),
        )
        with pytest.raises(MigrationError) as exc_info:
            validate_critical_references(connection)
    assert exc_info.value.reason_code == "migration_verification_failed"


def test_receipt_with_existing_job_rejects_null_project_owner() -> None:
    with sqlite3.connect(":memory:") as connection:
        connection.executescript(
            "CREATE TABLE operation_requests (project_id INTEGER, job_id INTEGER);"
            "CREATE TABLE generation_jobs (id INTEGER PRIMARY KEY, project_id INTEGER NOT NULL);"
            "INSERT INTO operation_requests VALUES (NULL, 1);"
            "INSERT INTO generation_jobs VALUES (1, 1);"
        )
        with pytest.raises(MigrationError) as exc_info:
            validate_critical_references(connection)
    assert exc_info.value.reason_code == "migration_verification_failed"


def test_language_request_may_reference_deleted_project_when_receipt_ownership_matches() -> None:
    with sqlite3.connect(":memory:") as connection:
        connection.executescript(
            "CREATE TABLE operation_requests (request_id TEXT PRIMARY KEY, project_id INTEGER);"
            "CREATE TABLE language_requests ("
            "request_id TEXT PRIMARY KEY, core_request_id TEXT, project_id INTEGER);"
            "INSERT INTO operation_requests VALUES ('core', 1);"
            "INSERT INTO language_requests VALUES ('language', 'core', 1);"
        )
        validate_critical_references(connection)


@pytest.mark.parametrize(
    ("parent_request_id", "successor_request_id"),
    (("parent", None), (None, "successor")),
)
def test_language_turn_reciprocal_link_rejects_null_other_side(
    parent_request_id: str | None, successor_request_id: str | None
) -> None:
    with sqlite3.connect(":memory:") as connection:
        connection.executescript(
            "CREATE TABLE language_requests (request_id TEXT PRIMARY KEY);"
            "CREATE TABLE language_turns (request_id TEXT PRIMARY KEY, "
            "parent_request_id TEXT, successor_request_id TEXT);"
            "INSERT INTO language_requests VALUES ('current');"
            "INSERT INTO language_requests VALUES ('parent');"
            "INSERT INTO language_requests VALUES ('successor');"
            "INSERT INTO language_turns VALUES ('parent', NULL, NULL);"
            "INSERT INTO language_turns VALUES ('successor', NULL, NULL);"
        )
        connection.execute(
            "INSERT INTO language_turns VALUES ('current', ?, ?)",
            (parent_request_id, successor_request_id),
        )
        with pytest.raises(MigrationError) as exc_info:
            validate_critical_references(connection)
    assert exc_info.value.reason_code == "migration_verification_failed"


def test_schema_version_lease_creates_nested_database_parent(tmp_path: Path) -> None:
    database = tmp_path / "nested" / "database" / "lease.db"
    lease = acquire_database_lease(f"sqlite:///{database.as_posix()}")
    try:
        assert database.parent.is_dir()
        assert lease.lock_path.exists()
        assert not database.exists()
    finally:
        lease.release()


def test_schema_version_lease_is_bound_to_one_database_and_release_is_idempotent(
    tmp_path: Path,
) -> None:
    database = tmp_path / "lease.db"
    url = f"sqlite:///{database.as_posix()}"
    lease = acquire_database_lease(url)
    assert lease.lock_path.exists()
    assert not database.exists()
    lease.assert_held_for(url)

    with pytest.raises(MigrationError) as exc_info:
        lease.assert_held_for(f"sqlite:///{(tmp_path / 'other.db').as_posix()}")
    assert exc_info.value.reason_code == "database_lease_unavailable"
    assert not (tmp_path / "other.db").exists()

    lease.release()
    lease.release()
    assert not lease.lock_path.exists()
    with pytest.raises(MigrationError) as released_error:
        lease.assert_held_for(url)
    assert released_error.value.reason_code == "database_lease_unavailable"


def test_lease_file_is_exclusive_bounded_and_never_removed_as_stale(
    tmp_path: Path,
) -> None:
    database = tmp_path / "lease.db"
    url = _url(database)
    lease = acquire_database_lease(url)
    try:
        payload = lease.lock_path.read_text(encoding="ascii")
        assert re.fullmatch(
            r"pid=\d+\nutc=\d{4}-\d\d-\d\dT[^\n]+Z\ntoken=[0-9a-f]{32}\n",
            payload,
        )
        with pytest.raises(MigrationError) as exc_info:
            acquire_database_lease(url)
        assert exc_info.value.reason_code == "database_lease_unavailable"
        assert lease.lock_path.read_text(encoding="ascii") == payload
    finally:
        lease.release()

    lease.lock_path.write_text("stale", encoding="ascii")
    with pytest.raises(MigrationError) as stale_error:
        acquire_database_lease(url)
    assert stale_error.value.reason_code == "database_lease_unavailable"
    assert lease.lock_path.read_text(encoding="ascii") == "stale"


def test_lease_assert_rejects_in_place_payload_tamper(tmp_path: Path) -> None:
    database = tmp_path / "assert-tamper.db"
    url = _url(database)
    lease = acquire_database_lease(url)
    original_identity = lease.lock_path.stat().st_ino
    lease.lock_path.write_bytes(b"mutated-in-place")
    assert lease.lock_path.stat().st_ino == original_identity

    try:
        with pytest.raises(MigrationError) as exc_info:
            lease.assert_held_for(url)
        assert exc_info.value.reason_code == "database_lease_unavailable"
    finally:
        try:
            lease.release()
        except MigrationError:
            lease.lock_path.unlink(missing_ok=True)


def test_lease_release_rejects_in_place_payload_tamper(tmp_path: Path) -> None:
    database = tmp_path / "release-tamper.db"
    lease = acquire_database_lease(_url(database))
    original_identity = lease.lock_path.stat().st_ino
    tampered = b"mutated-in-place"
    lease.lock_path.write_bytes(tampered)
    assert lease.lock_path.stat().st_ino == original_identity

    with pytest.raises(MigrationError) as exc_info:
        lease.release()

    assert exc_info.value.reason_code == "database_lease_unavailable"
    assert lease.lock_path.read_bytes() == tampered
    lease.lock_path.unlink()


def test_spawned_lease_contention_is_immediate_and_precedes_database_io(
    tmp_path: Path,
) -> None:
    database = tmp_path / "contended.db"
    ready = tmp_path / "ready"
    release = tmp_path / "release"
    process = _start_lease_holder(database, ready, release)
    try:
        _wait_for(ready, process)
        assert not database.exists()
        started = time.monotonic()
        with pytest.raises(MigrationError) as exc_info:
            acquire_database_lease(_url(database))
        assert time.monotonic() - started < 1.0
        assert exc_info.value.reason_code == "database_lease_unavailable"
        assert not database.exists()
    finally:
        release.write_text("release", encoding="ascii")
        stdout, stderr = process.communicate(timeout=30)
        assert process.returncode == 0, f"{stdout}\n{stderr}"

    lease = acquire_database_lease(_url(database))
    lease.release()


def test_runner_creates_empty_database_at_v1_without_backup(tmp_path: Path) -> None:
    database = tmp_path / "empty.db"
    result = _migrate(database)

    assert result.status == "created"
    assert result.from_version == 0
    assert result.to_version == 1
    assert result.backup_created is False
    assert result.backup_sha256 is None
    assert not (tmp_path / ".backups").exists()
    with sqlite3.connect(database) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (1,)
        assert set(_schema(connection)) == EXPECTED_TABLES


def test_runner_keeps_current_database_bytes_and_reports_current(tmp_path: Path) -> None:
    database = build_fixture("current_v1", tmp_path / "current.db", Base.metadata)
    before = database.read_bytes()

    result = _migrate(database)

    assert result.status == "current"
    assert result.from_version == result.to_version == 1
    assert result.backup_created is False
    assert database.read_bytes() == before


def test_runner_backs_up_and_preserves_critical_identities(tmp_path: Path) -> None:
    database = build_fixture("d30_v0", tmp_path / "legacy.db", Base.metadata)
    with sqlite3.connect(database) as connection:
        before = critical_identity_snapshot(connection, Base.metadata)
    assert before["projects"].primary_key_sha256 == (
        "a4d7550358c994173c884a06d6b049e2d13b44b790a9c88a68b9883777ccb63a"
    )

    result = _migrate(database)

    backups = _published_backups(tmp_path / ".backups")
    assert result.status == "migrated"
    assert result.backup_created is True
    assert len(backups) == 1
    assert result.backup_sha256 == sha256_file(backups[0])
    with sqlite3.connect(backups[0]) as backup, sqlite3.connect(database) as migrated:
        assert backup.execute("PRAGMA integrity_check").fetchall() == [("ok",)]
        assert critical_identity_snapshot(backup, Base.metadata) == before
        after = critical_identity_snapshot(migrated, Base.metadata)
        assert migrated.execute("PRAGMA integrity_check").fetchall() == [("ok",)]
        assert migrated.execute("PRAGMA foreign_key_check").fetchall() == []
        assert migrated.execute("PRAGMA user_version").fetchone() == (1,)
    for table, identity in before.items():
        assert after[table] == identity
    for table in set(after) - set(before):
        assert after[table].row_count == 0


def test_runner_requires_free_space_strictly_above_database_plus_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.migrations import backup

    database = build_fixture("d30_v0", tmp_path / "legacy.db", Base.metadata)
    before = database.read_bytes()
    minimum = len(before) + 16 * 1024 * 1024
    monkeypatch.setattr(
        backup.shutil,
        "disk_usage",
        lambda _path: type("Usage", (), {"total": minimum, "used": 0, "free": minimum})(),
    )

    with pytest.raises(MigrationError) as exc_info:
        _migrate(database)

    assert exc_info.value.reason_code == "backup_failed"
    assert database.read_bytes() == before
    assert not (tmp_path / ".backups").exists()


def test_runner_rejects_missing_released_and_mismatched_lease_before_database_io(
    tmp_path: Path,
) -> None:
    database = tmp_path / "target.db"
    other = tmp_path / "other.db"
    url = _url(database)
    with pytest.raises(MigrationError) as missing:
        migrate_database(url, Base.metadata, lease=None)
    assert missing.value.reason_code == "database_lease_unavailable"
    assert not database.exists()

    other_lease = acquire_database_lease(_url(other))
    try:
        with pytest.raises(MigrationError) as mismatch:
            migrate_database(url, Base.metadata, lease=other_lease)
        assert mismatch.value.reason_code == "database_lease_unavailable"
        assert not database.exists()
    finally:
        other_lease.release()

    released = acquire_database_lease(url)
    released.release()
    with pytest.raises(MigrationError) as released_error:
        migrate_database(url, Base.metadata, lease=released)
    assert released_error.value.reason_code == "database_lease_unavailable"
    assert not database.exists()


def test_backup_rejects_non_directory_publication_root(tmp_path: Path) -> None:
    database = build_fixture("d30_v0", tmp_path / "legacy.db", Base.metadata)
    before = database.read_bytes()
    backup_root = tmp_path / ".backups"
    backup_root.write_bytes(b"not-a-directory")

    with pytest.raises(MigrationError) as exc_info:
        _migrate(database)

    assert exc_info.value.reason_code == "backup_failed"
    assert database.read_bytes() == before
    assert backup_root.read_bytes() == b"not-a-directory"


def test_backup_rejects_redirected_symlink_root_before_writing(tmp_path: Path) -> None:
    database = build_fixture("d30_v0", tmp_path / "legacy.db", Base.metadata)
    before = database.read_bytes()
    redirected = tmp_path / "redirected"
    redirected.mkdir()
    backup_root = tmp_path / ".backups"
    try:
        backup_root.symlink_to(redirected, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlink creation is unavailable")

    with pytest.raises(MigrationError) as exc_info:
        _migrate(database)

    assert exc_info.value.reason_code == "backup_failed"
    assert database.read_bytes() == before
    assert tuple(redirected.iterdir()) == ()


def test_backup_lease_loss_before_backup_publication_leaves_no_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.migrations import backup as backup_module

    database = build_fixture("d30_v0", tmp_path / "legacy.db", Base.metadata)
    before = database.read_bytes()
    url = _url(database)
    lease = acquire_database_lease(url)
    original_hash = backup_module.sha256_file
    original_replace = backup_module._atomic_replace
    tampered = False
    publications = 0

    def hash_then_lose_lease(path: Path) -> str:
        nonlocal tampered
        digest = original_hash(path)
        if not tampered and path.suffix == ".tmp":
            lease.lock_path.write_bytes(b"lease-lost-before-backup-publication")
            tampered = True
        return digest

    def record_publication(source: Path, destination: Path) -> None:
        nonlocal publications
        publications += 1
        original_replace(source, destination)

    monkeypatch.setattr(backup_module, "sha256_file", hash_then_lose_lease)
    monkeypatch.setattr(backup_module, "_atomic_replace", record_publication)
    try:
        with pytest.raises(MigrationError) as exc_info:
            migrate_database(url, Base.metadata, lease=lease)
        assert exc_info.value.reason_code == "database_lease_unavailable"
        assert publications == 0
        assert database.read_bytes() == before
        assert tuple((tmp_path / ".backups").iterdir()) == ()
    finally:
        try:
            lease.release()
        except MigrationError:
            lease.lock_path.unlink(missing_ok=True)


def test_backup_lease_loss_before_metadata_publication_removes_published_backup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.migrations import backup as backup_module

    database = build_fixture("d30_v0", tmp_path / "legacy.db", Base.metadata)
    before = database.read_bytes()
    url = _url(database)
    lease = acquire_database_lease(url)
    original_replace = backup_module._atomic_replace
    publications = 0

    def publish_backup_then_lose_lease(source: Path, destination: Path) -> None:
        nonlocal publications
        original_replace(source, destination)
        publications += 1
        if publications == 1:
            lease.lock_path.write_bytes(b"lease-lost-before-metadata-publication")

    monkeypatch.setattr(backup_module, "_atomic_replace", publish_backup_then_lose_lease)
    try:
        with pytest.raises(MigrationError) as exc_info:
            migrate_database(url, Base.metadata, lease=lease)
        assert exc_info.value.reason_code == "database_lease_unavailable"
        assert publications == 1
        assert database.read_bytes() == before
        assert tuple((tmp_path / ".backups").iterdir()) == ()
    finally:
        try:
            lease.release()
        except MigrationError:
            lease.lock_path.unlink(missing_ok=True)


def test_runner_rejects_invalid_backup_integrity_without_publishing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.migrations import backup

    database = build_fixture("d30_v0", tmp_path / "legacy.db", Base.metadata)
    before = database.read_bytes()
    monkeypatch.setattr(backup, "_integrity_is_ok", lambda _connection: False)

    with pytest.raises(MigrationError) as exc_info:
        _migrate(database)

    assert exc_info.value.reason_code == "backup_invalid"
    assert exc_info.value.backup_available is False
    assert database.read_bytes() == before
    assert tuple((tmp_path / ".backups").iterdir()) == ()


def test_runner_removes_unpublished_backup_when_hashing_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.migrations import backup

    database = build_fixture("d30_v0", tmp_path / "legacy.db", Base.metadata)
    before = database.read_bytes()

    def fail_hash(_path: Path) -> str:
        raise OSError("synthetic hash failure")

    monkeypatch.setattr(backup, "sha256_file", fail_hash)
    with pytest.raises(MigrationError) as exc_info:
        _migrate(database)

    assert exc_info.value.reason_code == "backup_failed"
    assert exc_info.value.backup_available is False
    assert database.read_bytes() == before
    assert tuple((tmp_path / ".backups").iterdir()) == ()


def test_runner_retains_verified_backup_when_ddl_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.migrations import runner

    database = build_fixture("d30_v0", tmp_path / "legacy.db", Base.metadata)
    before = database.read_bytes()

    def fail_ddl(*_args: object, **_kwargs: object) -> None:
        raise MigrationError("migration_failed")

    monkeypatch.setattr(runner, "apply_v0_to_v1", fail_ddl)
    with pytest.raises(MigrationError) as exc_info:
        _migrate(database)

    assert exc_info.value.reason_code == "migration_failed"
    assert exc_info.value.backup_available is True
    assert database.read_bytes() == before
    backups = _published_backups(tmp_path / ".backups")
    assert len(backups) == 1
    assert sha256_file(backups[0])


@pytest.mark.parametrize("failure", ("identity", "reference"))
def test_runner_surfaces_post_migration_verification_failures_with_backup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    from app.migrations import runner

    database = build_fixture("d30_v0", tmp_path / f"{failure}.db", Base.metadata)
    if failure == "identity":
        original = runner.critical_identity_snapshot
        calls = 0

        def fail_post_identity(
            connection: sqlite3.Connection, metadata: MetaData
        ) -> dict[str, TableIdentity]:
            nonlocal calls
            calls += 1
            if calls >= 2:
                raise MigrationError("migration_verification_failed")
            return original(connection, metadata)

        monkeypatch.setattr(runner, "critical_identity_snapshot", fail_post_identity)
    else:
        original_references = runner.validate_critical_references
        calls = 0

        def fail_post_references(connection: sqlite3.Connection) -> None:
            nonlocal calls
            calls += 1
            if calls >= 2:
                raise MigrationError("migration_verification_failed")
            original_references(connection)

        monkeypatch.setattr(runner, "validate_critical_references", fail_post_references)

    with pytest.raises(MigrationError) as exc_info:
        _migrate(database)

    assert exc_info.value.reason_code == "migration_verification_failed"
    assert exc_info.value.backup_available is True
    backups = _published_backups(tmp_path / ".backups")
    assert len(backups) == 1
    with closing(sqlite3.connect(database)) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (1,)
    monkeypatch.undo()
    restore_database_backup(
        _url(database), backups[0], sha256_file(backups[0]), Base.metadata
    )
    assert database.read_bytes() == backups[0].read_bytes()


def test_runner_preserves_intentional_receipt_non_foreign_keys(tmp_path: Path) -> None:
    database = build_fixture("current_v1", tmp_path / "receipts.db", Base.metadata)
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA user_version=0")
        connection.execute(
            "INSERT INTO operation_requests VALUES ("
            "'deleted-owner', '{}', 'project.status.get', 1, 999, 1, 1, '{}', "
            "0, 999, 'receipt:deleted-owner', '{}', '2026-01-02 03:04:05'"
            ")"
        )

    result = _migrate(database)

    assert result.status == "migrated"
    with sqlite3.connect(database) as connection:
        assert connection.execute(
            "SELECT project_id, job_id FROM operation_requests "
            "WHERE request_id='deleted-owner'"
        ).fetchone() == (999, 999)


def test_restore_rejects_modified_backup_and_preserves_target(tmp_path: Path) -> None:
    database = build_fixture("d30_v0", tmp_path / "legacy.db", Base.metadata)
    result = _migrate(database)
    migrated = database.read_bytes()
    backup = _published_backups(tmp_path / ".backups")[0]
    backup.write_bytes(backup.read_bytes() + b"modified")

    with pytest.raises(MigrationError) as exc_info:
        restore_database_backup(
            _url(database), backup, result.backup_sha256 or "", Base.metadata
        )

    assert exc_info.value.reason_code == "backup_invalid"
    assert database.read_bytes() == migrated
    assert backup.read_bytes().endswith(b"modified")
    assert not database.with_name(f"{database.name}.migration.lock").exists()


def test_restore_rejects_hash_matching_corrupt_backup(tmp_path: Path) -> None:
    database = build_fixture("current_v1", tmp_path / "target.db", Base.metadata)
    before = database.read_bytes()
    corrupt = tmp_path / "corrupt.sqlite3"
    corrupt.write_bytes(b"not a sqlite database")

    with pytest.raises(MigrationError) as exc_info:
        restore_database_backup(
            _url(database), corrupt, sha256_file(corrupt), Base.metadata
        )

    assert exc_info.value.reason_code == "backup_invalid"
    assert database.read_bytes() == before
    assert corrupt.read_bytes() == b"not a sqlite database"


def test_restore_is_excluded_by_live_process_then_succeeds_offline(
    tmp_path: Path,
) -> None:
    database = build_fixture("d30_v0", tmp_path / "legacy.db", Base.metadata)
    result = _migrate(database)
    backup = _published_backups(tmp_path / ".backups")[0]
    backup_bytes = backup.read_bytes()
    migrated_bytes = database.read_bytes()
    ready = tmp_path / "ready"
    release = tmp_path / "release"
    process = _start_lease_holder(database, ready, release)
    try:
        _wait_for(ready, process)
        started = time.monotonic()
        with pytest.raises(MigrationError) as exc_info:
            restore_database_backup(
                _url(database), backup, result.backup_sha256 or "", Base.metadata
            )
        assert time.monotonic() - started < 1.0
        assert exc_info.value.reason_code == "database_lease_unavailable"
        assert database.read_bytes() == migrated_bytes
        assert backup.read_bytes() == backup_bytes
    finally:
        release.write_text("release", encoding="ascii")
        stdout, stderr = process.communicate(timeout=30)
        assert process.returncode == 0, f"{stdout}\n{stderr}"

    restore_database_backup(
        _url(database), backup, result.backup_sha256 or "", Base.metadata
    )
    assert database.read_bytes() == backup_bytes
    assert backup.read_bytes() == backup_bytes
    with sqlite3.connect(database) as connection:
        assert connection.execute("PRAGMA integrity_check").fetchall() == [("ok",)]
        assert connection.execute("PRAGMA user_version").fetchone() == (0,)


def _canonical_metadata(metadata: dict[str, object]) -> bytes:
    return json.dumps(
        metadata, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _backup_for_restore(tmp_path: Path) -> tuple[Path, Path, MigrationResult]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    database = build_fixture("d30_v0", tmp_path / "legacy.db", Base.metadata)
    result = _migrate(database)
    backup = next(
        path
        for path in (tmp_path / ".backups").iterdir()
        if not path.name.endswith(".metadata.json")
    )
    return database, backup, result


def test_backup_publishes_canonical_target_bound_metadata_and_stable_hash(
    tmp_path: Path,
) -> None:
    database, backup, result = _backup_for_restore(tmp_path)
    metadata_path = backup_metadata_path(backup)
    metadata_bytes = metadata_path.read_bytes()
    metadata = json.loads(metadata_bytes)

    assert metadata_bytes == _canonical_metadata(metadata)
    assert metadata == {
        "backup_sha256": result.backup_sha256,
        "metadata_schema_version": 1,
        "source_critical_identities": metadata["source_critical_identities"],
        "source_schema_version": 0,
        "target_database_path_sha256": metadata["target_database_path_sha256"],
    }
    assert metadata["source_critical_identities"]["projects"]["row_count"] == 1
    assert sha256_file(backup) == result.backup_sha256
    assert database.resolve() != backup.resolve()


def test_backup_rejects_bytes_changed_during_atomic_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.migrations import backup as backup_module

    database = build_fixture("d30_v0", tmp_path / "legacy.db", Base.metadata)
    original_replace = backup_module._atomic_replace

    def replace_then_corrupt(source: Path, destination: Path) -> None:
        original_replace(source, destination)
        if not str(destination).endswith(".metadata.json"):
            with Path(destination).open("ab") as stream:
                stream.write(b"changed-after-publication")

    monkeypatch.setattr(backup_module, "_atomic_replace", replace_then_corrupt)

    with pytest.raises(MigrationError) as exc_info:
        _migrate(database)

    assert exc_info.value.reason_code == "backup_invalid"
    assert tuple((tmp_path / ".backups").iterdir()) == ()


def test_restore_rejects_backup_outside_exact_target_backup_directory(
    tmp_path: Path,
) -> None:
    database, backup, result = _backup_for_restore(tmp_path)
    outside = tmp_path / "outside.sqlite3"
    shutil.copyfile(backup, outside)
    shutil.copyfile(backup_metadata_path(backup), backup_metadata_path(outside))
    target_before = database.read_bytes()

    with pytest.raises(MigrationError) as exc_info:
        restore_database_backup(
            _url(database), outside, result.backup_sha256 or "", Base.metadata
        )

    assert exc_info.value.reason_code == "backup_invalid"
    assert database.read_bytes() == target_before


@pytest.mark.parametrize("reparse_target", ("root", "backup"))
def test_restore_rejects_mocked_windows_reparse_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reparse_target: str,
) -> None:
    database, backup, result = _backup_for_restore(tmp_path)
    target_before = database.read_bytes()
    selected_path = backup.parent if reparse_target == "root" else backup
    original_lstat = Path.lstat
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)

    def lstat_with_reparse(path: Path) -> os.stat_result:
        observed = original_lstat(path)
        if path != selected_path:
            return observed
        return SimpleNamespace(
            st_mode=observed.st_mode,
            st_dev=observed.st_dev,
            st_ino=observed.st_ino,
            st_size=observed.st_size,
            st_mtime_ns=observed.st_mtime_ns,
            st_ctime_ns=observed.st_ctime_ns,
            st_file_attributes=(
                getattr(observed, "st_file_attributes", 0) | reparse_flag
            ),
        )

    monkeypatch.setattr(Path, "lstat", lstat_with_reparse)

    with pytest.raises(MigrationError) as exc_info:
        restore_database_backup(
            _url(database), backup, result.backup_sha256 or "", Base.metadata
        )

    assert exc_info.value.reason_code == "backup_invalid"
    assert database.read_bytes() == target_before


def test_restore_rejects_redirected_symlink_root_when_supported(tmp_path: Path) -> None:
    database, backup, result = _backup_for_restore(tmp_path)
    target_before = database.read_bytes()
    backup_root = backup.parent
    redirected = tmp_path / "redirected"
    backup_root.rename(redirected)
    try:
        backup_root.symlink_to(redirected, target_is_directory=True)
    except OSError:
        redirected.rename(backup_root)
        pytest.skip("directory symlink creation is unavailable")

    with pytest.raises(MigrationError) as exc_info:
        restore_database_backup(
            _url(database), backup, result.backup_sha256 or "", Base.metadata
        )

    assert exc_info.value.reason_code == "backup_invalid"
    assert database.read_bytes() == target_before


def test_restore_rejects_backup_bound_to_another_target(tmp_path: Path) -> None:
    source, backup, result = _backup_for_restore(tmp_path / "source")
    target_root = tmp_path / "target"
    target_root.mkdir()
    target = build_fixture("current_v1", target_root / "target.db", Base.metadata)
    target_backup_root = target.parent / ".backups"
    target_backup_root.mkdir()
    copied_backup = target_backup_root / backup.name
    shutil.copyfile(backup, copied_backup)
    shutil.copyfile(backup_metadata_path(backup), backup_metadata_path(copied_backup))
    target_before = target.read_bytes()

    with pytest.raises(MigrationError) as exc_info:
        restore_database_backup(
            _url(target), copied_backup, result.backup_sha256 or "", Base.metadata
        )

    assert exc_info.value.reason_code == "backup_invalid"
    assert target.read_bytes() == target_before
    assert source.exists()


def test_restore_rejects_symlink_and_special_backup_paths(tmp_path: Path) -> None:
    database, backup, result = _backup_for_restore(tmp_path)
    target_before = database.read_bytes()
    special = backup.parent / "special.sqlite3"
    special.mkdir()
    backup_metadata_path(special).write_bytes(backup_metadata_path(backup).read_bytes())
    with pytest.raises(MigrationError) as special_error:
        restore_database_backup(
            _url(database), special, result.backup_sha256 or "", Base.metadata
        )
    assert special_error.value.reason_code == "backup_invalid"

    symlink = backup.parent / "linked.sqlite3"
    try:
        symlink.symlink_to(backup)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    shutil.copyfile(backup_metadata_path(backup), backup_metadata_path(symlink))
    with pytest.raises(MigrationError) as symlink_error:
        restore_database_backup(
            _url(database), symlink, result.backup_sha256 or "", Base.metadata
        )
    assert symlink_error.value.reason_code == "backup_invalid"
    assert database.read_bytes() == target_before


def _replace_backup_with_incompatible_v1(
    backup: Path, *, mutation_sql: str
) -> str:
    source = backup.with_name("source-v1.db")
    build_fixture("current_v1", source, Base.metadata)
    with sqlite3.connect(source) as connection:
        connection.execute(mutation_sql)
        identities = critical_identity_snapshot(connection, Base.metadata)
    shutil.copyfile(source, backup)

    backup_hash = sha256_file(backup)
    metadata_path = backup_metadata_path(backup)
    metadata = json.loads(metadata_path.read_bytes())
    metadata["source_schema_version"] = 1
    metadata["source_critical_identities"] = {
        table: {
            "primary_key_columns": list(identity.primary_key_columns),
            "primary_key_sha256": identity.primary_key_sha256,
            "row_count": identity.row_count,
            "table": identity.table,
        }
        for table, identity in identities.items()
    }
    metadata["backup_sha256"] = backup_hash
    metadata_path.write_bytes(_canonical_metadata(metadata))
    return backup_hash


@pytest.mark.parametrize(
    "mutation_sql",
    (
        "DROP TABLE project_identities",
        'ALTER TABLE projects DROP COLUMN "current_artifact_id"',
    ),
    ids=("missing-table", "missing-column"),
)
def test_restore_rejects_valid_v1_backup_with_incomplete_current_schema_before_mutation(
    tmp_path: Path, mutation_sql: str
) -> None:
    database, backup, _result = _backup_for_restore(tmp_path)
    expected_hash = _replace_backup_with_incompatible_v1(
        backup, mutation_sql=mutation_sql
    )
    target_before = database.read_bytes()
    stale_sidecar = Path(f"{database}-journal")
    stale_sidecar.write_bytes(b"stale")

    with pytest.raises(MigrationError) as exc_info:
        restore_database_backup(_url(database), backup, expected_hash, Base.metadata)

    assert exc_info.value.reason_code == "backup_invalid"
    assert database.read_bytes() == target_before
    assert stale_sidecar.read_bytes() == b"stale"


def test_restore_rejects_incompatible_metadata_schema(tmp_path: Path) -> None:
    database, backup, result = _backup_for_restore(tmp_path)
    metadata_path = backup_metadata_path(backup)
    metadata = json.loads(metadata_path.read_bytes())
    metadata["source_schema_version"] = 2
    metadata_path.write_bytes(_canonical_metadata(metadata))
    target_before = database.read_bytes()

    with pytest.raises(MigrationError) as exc_info:
        restore_database_backup(
            _url(database), backup, result.backup_sha256 or "", Base.metadata
        )

    assert exc_info.value.reason_code == "backup_invalid"
    assert database.read_bytes() == target_before


def test_restore_rejects_noncanonical_metadata_bytes(tmp_path: Path) -> None:
    database, backup, result = _backup_for_restore(tmp_path)
    metadata_path = backup_metadata_path(backup)
    metadata_path.write_bytes(metadata_path.read_bytes() + b"\n")
    target_before = database.read_bytes()

    with pytest.raises(MigrationError) as exc_info:
        restore_database_backup(
            _url(database), backup, result.backup_sha256 or "", Base.metadata
        )

    assert exc_info.value.reason_code == "backup_invalid"
    assert database.read_bytes() == target_before


def test_restore_rejects_metadata_identity_mismatch(tmp_path: Path) -> None:
    database, backup, result = _backup_for_restore(tmp_path)
    metadata_path = backup_metadata_path(backup)
    metadata = json.loads(metadata_path.read_bytes())
    metadata["source_critical_identities"]["projects"]["row_count"] = 99
    metadata_path.write_bytes(_canonical_metadata(metadata))
    target_before = database.read_bytes()

    with pytest.raises(MigrationError) as exc_info:
        restore_database_backup(
            _url(database), backup, result.backup_sha256 or "", Base.metadata
        )

    assert exc_info.value.reason_code == "backup_invalid"
    assert database.read_bytes() == target_before


def test_restore_rejects_reference_incompatible_backup(tmp_path: Path) -> None:
    database, backup, _result = _backup_for_restore(tmp_path)
    with sqlite3.connect(backup) as connection:
        connection.execute("PRAGMA foreign_keys=OFF")
        connection.execute("UPDATE projects SET current_artifact_id=999")
        connection.commit()
    new_hash = sha256_file(backup)
    metadata_path = backup_metadata_path(backup)
    metadata = json.loads(metadata_path.read_bytes())
    metadata["backup_sha256"] = new_hash
    metadata_path.write_bytes(_canonical_metadata(metadata))
    target_before = database.read_bytes()

    with pytest.raises(MigrationError) as exc_info:
        restore_database_backup(_url(database), backup, new_hash, Base.metadata)

    assert exc_info.value.reason_code == "backup_invalid"
    assert database.read_bytes() == target_before


def test_restore_removes_stale_sqlite_sidecars_before_atomic_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database, backup, result = _backup_for_restore(tmp_path)
    for suffix in ("-wal", "-shm", "-journal"):
        Path(f"{database}{suffix}").write_bytes(b"stale")
    from app.migrations.lease import DatabaseLease

    original_assert = DatabaseLease.assert_held_for
    held_checks: list[bool] = []

    def record_assert(self: DatabaseLease, database_url: str) -> None:
        original_assert(self, database_url)
        held_checks.append(self.lock_path.exists())

    monkeypatch.setattr(DatabaseLease, "assert_held_for", record_assert)
    restore_database_backup(
        _url(database), backup, result.backup_sha256 or "", Base.metadata
    )

    assert len(held_checks) >= 5
    assert all(held_checks)
    assert all(
        not Path(f"{database}{suffix}").exists()
        for suffix in ("-wal", "-shm", "-journal")
    )
    assert sha256_file(database) == result.backup_sha256
    with sqlite3.connect(database) as connection:
        assert connection.execute("PRAGMA integrity_check").fetchall() == [("ok",)]


def test_lease_release_restores_replacement_moved_by_forced_race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.migrations import lease as lease_module

    database = tmp_path / "race.db"
    lease = acquire_database_lease(_url(database))
    replacement = b"pid=999\nutc=2026-01-01T00:00:00Z\ntoken=replacement\n"
    original_replace = lease_module.os.replace

    def replace_after_intruder(source: Path, destination: Path) -> None:
        Path(source).unlink()
        Path(source).write_bytes(replacement)
        original_replace(source, destination)

    monkeypatch.setattr(lease_module.os, "replace", replace_after_intruder)
    lease.release()

    assert lease.lock_path.read_bytes() == replacement
    assert not tuple(tmp_path.glob("*.release-*"))


def test_lease_release_collision_never_deletes_other_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.migrations import lease as lease_module

    database = tmp_path / "collision.db"
    lease = acquire_database_lease(_url(database))
    replacement = b"pid=999\nutc=2026-01-01T00:00:00Z\ntoken=replacement\n"
    intruder = b"pid=1000\nutc=2026-01-01T00:00:01Z\ntoken=intruder\n"
    original_replace = lease_module.os.replace
    original_link = lease_module.os.link

    def replace_after_intruder(source: Path, destination: Path) -> None:
        Path(source).unlink()
        Path(source).write_bytes(replacement)
        original_replace(source, destination)

    def collide_restore(source: Path, destination: Path) -> None:
        Path(destination).write_bytes(intruder)
        original_link(source, destination)

    monkeypatch.setattr(lease_module.os, "replace", replace_after_intruder)
    monkeypatch.setattr(lease_module.os, "link", collide_restore)

    with pytest.raises(MigrationError) as exc_info:
        lease.release()

    assert exc_info.value.reason_code == "database_lease_unavailable"
    assert lease.lock_path.read_bytes() == intruder
    tombstones = tuple(tmp_path.glob("*.release-*"))
    assert len(tombstones) == 1
    assert tombstones[0].read_bytes() == replacement
