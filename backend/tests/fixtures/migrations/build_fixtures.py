"""Deterministic synthetic SQLite ancestry fixtures for D34 migration tests."""
from __future__ import annotations

import sqlite3
from collections.abc import Callable
from pathlib import Path

from sqlalchemy import MetaData, create_engine


FixtureBuilder = Callable[[Path, MetaData], None]


_FIXTURE_ROOT = Path(__file__).parent


def _create_frozen(path: Path, fixture_name: str) -> None:
    ddl = (_FIXTURE_ROOT / f"{fixture_name}.sql").read_text(encoding="utf-8")
    with sqlite3.connect(path) as connection:
        connection.executescript(ddl)
        connection.execute("PRAGMA user_version=0")


def _create_current(path: Path, metadata: MetaData, *, version: int) -> None:
    engine = create_engine(f"sqlite:///{path.as_posix()}")
    try:
        metadata.create_all(engine)
        with engine.begin() as connection:
            connection.exec_driver_sql(f"PRAGMA user_version={version}")
    finally:
        engine.dispose()


def _seed_project(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        INSERT INTO projects (
            id, revision, title, source_script, status, progress,
            llm_provider, image_provider, voicevox_url, voicevox_speaker_id,
            voicevox_speed_scale, voicevox_pitch_scale,
            voicevox_intonation_scale, voicevox_volume_scale,
            subtitle_enabled, subtitle_font_size, subtitle_position,
            subtitle_text_color, subtitle_outline_color, subtitle_background,
            subtitle_max_chars_per_line, visual_focus_enabled, subtitle_mode,
            narration_pacing_mode, pronunciation_overrides,
            narration_sentence_pause_seconds, max_slides_per_block,
            pre_margin_seconds, post_margin_seconds, min_display_seconds,
            use_fake_providers, created_at, updated_at
        ) VALUES (
            101, 1, 'fixture-project', 'fixture-script', 'pending', 0.0,
            'openai_compatible', 'openai', 'http://127.0.0.1:50021', 1,
            1.0, 0.0, 1.0, 1.0,
            1, 48, 'bottom', '#FFFFFF', '#000000', 1,
            36, 0, 'packed', 'fixed', '[]',
            1.5, 1, 0.15, 1.5, 2.0,
            1, '2026-01-02 03:04:05', '2026-01-02 03:04:05'
        )
        """
    )


def build_empty_v0(path: Path, metadata: MetaData) -> None:
    del metadata
    sqlite3.connect(path).close()


def build_upstream_v0(path: Path, metadata: MetaData) -> None:
    del metadata
    _create_frozen(path, "upstream_v0")


def build_d30_v0(path: Path, metadata: MetaData) -> None:
    del metadata
    _create_frozen(path, "d30_v0")


def build_partially_additive_v0(path: Path, metadata: MetaData) -> None:
    _create_current(path, metadata, version=0)
    with sqlite3.connect(path) as connection:
        _seed_project(connection)
        connection.execute("DROP TABLE language_turns")
        connection.execute('ALTER TABLE projects DROP COLUMN "current_artifact_id"')
        connection.execute('ALTER TABLE projects ADD COLUMN "partial_extra" TEXT')
        connection.execute(
            "UPDATE projects SET partial_extra = 'keep-partial' WHERE id = 101"
        )


def _quote(identifier: str) -> str:
    escaped = identifier.replace('"', '""')
    return f'"{escaped}"'


def _rebuild_without_column(
    connection: sqlite3.Connection, table: str, omitted_column: str
) -> None:
    columns = [
        row
        for row in connection.execute(f"PRAGMA table_info({_quote(table)})")
        if str(row[1]).lower() != omitted_column.lower()
    ]
    definitions = []
    for _, name, declared_type, not_null, default, primary_key in columns:
        definition = f"{_quote(str(name))} {declared_type}"
        if primary_key:
            definition += " PRIMARY KEY"
        if not_null:
            definition += " NOT NULL"
        if default is not None:
            definition += f" DEFAULT {default}"
        definitions.append(definition)
    replacement = f"__partial_{table}"
    selected = ", ".join(_quote(str(row[1])) for row in columns)
    connection.execute(
        f"CREATE TABLE {_quote(replacement)} ({', '.join(definitions)})"
    )
    connection.execute(
        f"INSERT INTO {_quote(replacement)} ({selected}) "
        f"SELECT {selected} FROM {_quote(table)}"
    )
    connection.execute(f"DROP TABLE {_quote(table)}")
    connection.execute(
        f"ALTER TABLE {_quote(replacement)} RENAME TO {_quote(table)}"
    )


def _build_partial_unique_column_v0(
    path: Path, metadata: MetaData, *, table: str, column: str
) -> None:
    _create_current(path, metadata, version=0)
    with sqlite3.connect(path) as connection:
        _seed_project(connection)
        _rebuild_without_column(connection, table, column)


def build_partial_artifact_job_v0(path: Path, metadata: MetaData) -> None:
    _build_partial_unique_column_v0(
        path, metadata, table="generation_artifacts", column="job_id"
    )


def build_partial_operation_job_v0(path: Path, metadata: MetaData) -> None:
    _build_partial_unique_column_v0(
        path, metadata, table="operation_requests", column="job_id"
    )


def build_partial_language_parent_v0(path: Path, metadata: MetaData) -> None:
    _build_partial_unique_column_v0(
        path, metadata, table="language_turns", column="parent_request_id"
    )


def build_current_v1(path: Path, metadata: MetaData) -> None:
    _create_current(path, metadata, version=1)
    with sqlite3.connect(path) as connection:
        _seed_project(connection)


def build_newer_v2(path: Path, metadata: MetaData) -> None:
    _create_current(path, metadata, version=2)
    with sqlite3.connect(path) as connection:
        _seed_project(connection)


def build_altered_legacy_primary_key(path: Path, metadata: MetaData) -> None:
    del metadata
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE projects (id INTEGER NOT NULL, title TEXT PRIMARY KEY)"
        )
        connection.execute("INSERT INTO projects VALUES (101, 'fixture-project')")


def build_affinity_collision(
    path: Path,
    metadata: MetaData,
    *,
    table: str,
    column: str,
    declared_type: str,
    value_sql: str,
) -> None:
    del metadata
    with sqlite3.connect(path) as connection:
        connection.execute(
            f'CREATE TABLE "{table}" ("{column}" {declared_type})'
        )
        connection.execute(
            f'INSERT INTO "{table}" ("{column}") VALUES ({value_sql})'
        )


def build_matching_affinity_aliases(path: Path, metadata: MetaData) -> None:
    del metadata
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE projects (
                id INT8,
                title CLOB,
                progress DOUBLE PRECISION,
                subtitle_enabled BOOLEAN
            )
            """
        )
        connection.execute(
            "INSERT INTO projects VALUES (101, 'alias', 1.5, 1)"
        )
        connection.execute("CREATE TABLE external_calls (response_body)")
        connection.execute("INSERT INTO external_calls VALUES (x'31')")


FIXTURE_BUILDERS: dict[str, FixtureBuilder] = {
    "empty_v0": build_empty_v0,
    "upstream_v0": build_upstream_v0,
    "d30_v0": build_d30_v0,
    "partially_additive_v0": build_partially_additive_v0,
    "partial_artifact_job_v0": build_partial_artifact_job_v0,
    "partial_operation_job_v0": build_partial_operation_job_v0,
    "partial_language_parent_v0": build_partial_language_parent_v0,
    "current_v1": build_current_v1,
    "newer_v2": build_newer_v2,
}


def build_fixture(name: str, path: Path, metadata: MetaData) -> Path:
    """Build one named fixture at a caller-owned empty path."""
    FIXTURE_BUILDERS[name](path, metadata)
    return path
