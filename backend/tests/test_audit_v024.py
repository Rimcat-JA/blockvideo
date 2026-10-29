"""Regressions for the v0.2.4 audit fixes."""
from __future__ import annotations

from pathlib import Path

import pytest

from app.db import get_session_factory
from app.models.job import GenerationJob, JobStatus
from app.models.project import Project, ProjectStatus
from app.workers.operation_dispatcher import mark_interrupted_operation_jobs
from tests.test_job_recovery import persisted_job


@pytest.mark.parametrize("status", [JobStatus.running, JobStatus.pending])
def test_jobs_left_in_progress_by_a_v0_app_release_their_project(
    temp_storage: Path, status: JobStatus
) -> None:
    project_id, job_id = persisted_job(status=status)
    with get_session_factory()() as db:
        job = db.get(GenerationJob, job_id)
        # A v0 job has no frozen inputs at all.
        job.input_revision = None
        job.input_snapshot = None
        job.input_fingerprint = None
        db.get(Project, project_id).status = ProjectStatus.rendering
        db.commit()

    assert mark_interrupted_operation_jobs() == 1
    assert mark_interrupted_operation_jobs() == 0

    with get_session_factory()() as db:
        assert db.get(GenerationJob, job_id).status == JobStatus.failed
        assert db.get(Project, project_id).status == ProjectStatus.failed


def test_a_legacy_subtitle_colour_does_not_block_later_saves(temp_storage: Path) -> None:
    from fastapi.testclient import TestClient

    from app.main import create_app

    client = TestClient(create_app())
    project = client.post("/api/projects", json={
        "title": "legacy colour", "source_script": "台本です。", "use_fake_providers": True,
    }).json()
    with get_session_factory()() as db:
        db.get(Project, project["id"]).subtitle_outline_color = "navy-ish"
        db.commit()

    response = client.patch(f"/api/projects/{project['id']}", json={"title": "renamed"})

    assert response.status_code == 200, response.text
    assert response.json()["title"] == "renamed"
    with get_session_factory()() as db:
        assert db.get(Project, project["id"]).subtitle_outline_color == "#000000"


@pytest.mark.parametrize("path, extra", [
    ("/api/projects", {"title": "t", "source_script": "台本です。"}),
    ("/api/projects/quick", {"source_script": "台本です。"}),
])
def test_an_out_of_range_speaker_id_is_a_validation_error(
    temp_storage: Path, path: str, extra: dict
) -> None:
    from fastapi.testclient import TestClient

    from app.main import create_app

    client = TestClient(create_app())
    response = client.post(path, json={**extra, "use_fake_providers": True,
                                       "voicevox_speaker_id": 2**63})
    assert response.status_code == 422


def test_restore_script_lists_and_restores_the_latest_backup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import sqlite3
    from contextlib import closing

    from app.core import config
    from app.db import Base, register_models
    from app.migrations.lease import acquire_database_lease
    from app.migrations.runner import migrate_database
    from scripts import restore_backup
    from tests.fixtures.migrations.build_fixtures import build_fixture

    register_models()
    database = build_fixture("d30_v0", tmp_path / "blockvideo.db", Base.metadata)
    url = f"sqlite:///{database.as_posix()}"
    lease = acquire_database_lease(url)
    try:
        migrate_database(url, Base.metadata, lease=lease)
    finally:
        lease.release()
    monkeypatch.setenv("DATABASE_URL", url)
    config.reset_settings_cache()

    assert restore_backup.main([]) == 0
    assert ".backups" in capsys.readouterr().out
    assert restore_backup.main(["--latest"]) == 0
    with closing(sqlite3.connect(database)) as restored:
        assert restored.execute("PRAGMA user_version").fetchone() == (0,)

    held = acquire_database_lease(url)
    try:
        assert restore_backup.main(["--latest"]) == 1
        assert "停止" in capsys.readouterr().err
    finally:
        held.release()
    config.reset_settings_cache()


def test_a_listing_too_large_to_read_is_clipped_before_allocation(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.services import diagram_renderer

    sizes: list[tuple[int, int]] = []
    original = diagram_renderer._blank_layer

    def record(w: int, h: int):
        sizes.append((w, h))
        return original(min(w, 64), min(h, 64))

    monkeypatch.setattr(diagram_renderer, "_blank_layer", record)
    lines = ["x" * 4000] + ["y"] * 4000
    diagram_renderer.render_grid_content(lines, 1920, 880)

    (width, height), = sizes
    # The unclipped layer was 28004 x 52013 RGBA (5.4 GiB).
    assert width * height * 4 < 200 * 1024 * 1024


def test_file_identity_reuses_the_digest_of_an_unchanged_old_file(
    temp_storage: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os
    import time

    from app.services import artifact_store

    path = temp_storage / "video.mp4"
    path.write_bytes(b"a" * 1024)
    old = time.time() - 60
    os.utime(path, (old, old))
    first = artifact_store.file_identity(path)

    hashed: list[bytes] = []
    real_sha256 = artifact_store.hashlib.sha256

    def counting(*args):
        hashed.append(b"")
        return real_sha256(*args)

    monkeypatch.setattr(artifact_store.hashlib, "sha256", counting)
    assert artifact_store.file_identity(path) == first
    assert hashed == []

    path.write_bytes(b"b" * 1024)  # same size, new content and mtime
    assert artifact_store.file_identity(path)["sha256"] != first["sha256"]
    assert len(hashed) == 1


def test_a_recently_written_file_is_always_rehashed(temp_storage: Path) -> None:
    from app.services import artifact_store

    path = temp_storage / "slide.png"
    path.write_bytes(b"a" * 64)
    first = artifact_store.file_identity(path)
    stat = path.stat()
    path.write_bytes(b"b" * 64)
    import os
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))  # same tick, same size

    assert artifact_store.file_identity(path)["sha256"] != first["sha256"]


def test_unpublished_history_output_is_removed_but_published_output_is_kept(temp_storage: Path) -> None:
    from app.services.artifact_store import discard_unpublished_output
    from app.services.paths import project_dir

    project_id, job_id = persisted_job()
    directory = project_dir(project_id) / "history" / f"job-{job_id:08d}"
    directory.mkdir(parents=True)
    (directory / "video.pending.mp4").write_bytes(b"x")

    discard_unpublished_output(job_id, directory)

    assert not directory.exists()


def test_mermaid_never_falls_back_to_npx(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from app.services import mermaid_renderer

    monkeypatch.setattr(mermaid_renderer, "resolve_mmdc", lambda: "mmdc")
    monkeypatch.setattr(mermaid_renderer.shutil, "which", lambda name: "C:/npx" if name == "npx" else None)
    with pytest.raises(RuntimeError, match="mmdc"):
        mermaid_renderer.render_mermaid_to_png_sync("graph TD; A-->B", tmp_path / "out.png", width=100, height=100)


def test_block_patch_caps_listing_lengths_like_a_generated_plan(temp_storage: Path) -> None:
    from fastapi.testclient import TestClient

    from app.main import create_app
    from app.models.block import Block

    client = TestClient(create_app())
    project = client.post("/api/projects", json={
        "title": "caps", "source_script": "台本です。", "use_fake_providers": True,
    }).json()
    with get_session_factory()() as db:
        block = Block(project_id=project["id"], index=0, source_text="本文。", tts_text="本文。")
        db.add(block)
        db.commit()
        block_id = block.id

    too_long = client.patch(f"/api/blocks/{block_id}", json={
        "visual_plan": {"visual_type": "code_slide", "code": "x" * 8001}})
    within = client.patch(f"/api/blocks/{block_id}", json={
        "visual_plan": {"visual_type": "code_slide", "code": "x" * 8000}})

    assert too_long.status_code == 422
    assert within.status_code == 200, within.text


def test_segment_script_keeps_runs_of_sentence_enders() -> None:
    from app.services.splitter import segment_script

    text = ("えっ？！本当に？ はい。。。" + chr(10)) * 200
    pieces = segment_script(text, max_chars=1200)
    assert len(pieces) > 1
    joined = "".join(pieces)
    for ender in "。？！":
        assert joined.count(ender) == text.count(ender)


def test_deterministic_splitters_keep_runs_of_enders() -> None:
    from app.core.config import get_settings
    from app.providers.llm_fake import FakeLLMProvider
    from app.services.splitter import _deterministic_split

    text = "えっ？！本当に!? はい。。。"
    blocks = _deterministic_split(text, get_settings())
    assert "".join(b.source_text for b in blocks).replace(" ", "") == text.replace(" ", "")
    fake = FakeLLMProvider()._split_text_into_blocks(text, max_chars=200)
    assert "".join(fake).replace(" ", "") == text.replace(" ", "")


@pytest.mark.asyncio
async def test_a_short_script_that_is_only_code_still_gets_a_spoken_cue() -> None:
    from app.core import config
    from app.providers.llm_fake import FakeLLMProvider
    from app.services.splitter import split_script

    config.reset_settings_cache()
    script = chr(10).join(["```slide", "┌──┐", "│A │", "└──┘", "```"])
    result = await split_script(script, FakeLLMProvider(), config.get_settings())

    assert result.blocks
    assert all(block.tts_text.strip() for block in result.blocks)


@pytest.mark.parametrize("requested, at_least", [(80, 80), (96, 96), (120, 100)])
def test_large_subtitle_fonts_are_honoured_with_shorter_cues(requested: int, at_least: int) -> None:
    from app.services.subtitles import build_band_cues

    text = "これは字幕の大きさを確かめるための少し長めのナレーションです。二つ目の文も続けて読み上げます。"
    cues, font = build_band_cues(text, duration_ms=8000, band_height=200,
                                 base_font_size=requested, base_max_chars=36)
    assert font >= at_least  # used to be 66px for all three
    assert "".join(cue.text.replace(chr(10), "") for cue in cues).replace(" ", "") == text
