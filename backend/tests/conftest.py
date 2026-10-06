"""Pytest fixtures and configuration."""
from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture()
def temp_storage(monkeypatch, tmp_path: Path) -> Path:
    storage = tmp_path / "storage"
    storage.mkdir(parents=True, exist_ok=True)
    db = storage / "blockvideo.db"
    monkeypatch.setenv("STORAGE_ROOT", str(storage))
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db.as_posix()}")
    # Force FFMPEG_PATH / FFPROBE_PATH empty so tests can detect missing binary.
    monkeypatch.delenv("FFMPEG_PATH", raising=False)
    monkeypatch.delenv("FFPROBE_PATH", raising=False)
    from app.core import config

    config.reset_settings_cache()
    from app.core.startup_status import StartupStatus, set_startup_status
    from app.db import init_db, register_models, reset_db_for_tests

    reset_db_for_tests()
    register_models()
    init_db()
    set_startup_status(
        StartupStatus(
            status="ready",
            reason_code=None,
            message="起動が完了しました。",
            schema_version=1,
            backup_available=False,
        )
    )
    yield storage
    # teardown
    shutil.rmtree(storage, ignore_errors=True)
    config.reset_settings_cache()
    reset_db_for_tests()
    from app.core.startup_status import reset_startup_status_for_tests

    reset_startup_status_for_tests()


@pytest.fixture()
def ffmpeg_available() -> bool:
    from app.services.ffmpeg_runner import ffmpeg_available, ffprobe_available

    return ffmpeg_available() and ffprobe_available()


@pytest.fixture()
def skip_if_no_ffmpeg(ffmpeg_available):
    if not ffmpeg_available:
        pytest.skip("ffmpeg/ffprobe not available on PATH")