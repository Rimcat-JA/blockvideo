"""D12 dependencies are checked by observed stage sets, not implementation copies."""
from __future__ import annotations

import pytest

from app.db import get_session_factory
from app.models.block import Block, BlockStatus, VisualType
from app.models.project import Project
from app.services.generation_plan import build_generation_plan, dependencies_for_settings
from app.services.generation_snapshots import capture_inputs, fingerprint_inputs
from app.services.paths import block_audio_path, block_image_path, block_video_path, relpath_for_db
from app.services.project_settings import apply_project_settings


def ready_project(db):
    project = Project(title="synthetic", source_script="元の台本。", use_fake_providers=True)
    db.add(project)
    db.flush()
    block = Block(
        project_id=project.id, index=0, source_text="編集した台本。", tts_text="編集した読み上げ。",
        visual_type=VisualType.text_slide,
        visual_plan_json={"visual_type": "text_slide", "heading": "合成テスト"},
        status_split=BlockStatus.completed, status_visual_plan=BlockStatus.completed,
        status_image=BlockStatus.completed, status_audio=BlockStatus.completed,
        status_render=BlockStatus.completed, duration_ms=1000, display_duration_ms=2000,
    )
    for field, path in (("image_path", block_image_path(project.id, 0)),
                        ("audio_path", block_audio_path(project.id, 0)),
                        ("video_path", block_video_path(project.id, 0))):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic-fixture")
        setattr(block, field, relpath_for_db(path))
    db.add(block)
    db.commit()
    db.refresh(project)
    return project


@pytest.mark.parametrize(("updates", "stages"), [
    ({"subtitle_font_size": 58}, ["render"]),
    ({"subtitle_mode": "packed"}, ["render"]),
    ({"voicevox_speed_scale": 1.2}, ["audio", "render"]),
    ({"voicevox_speaker_id": 2}, ["audio", "render"]),
    ({"pronunciation_overrides": [{"surface": "API", "reading": "エーピーアイ"}]}, ["audio", "render"]),
    ({"subtitle_enabled": False}, ["image", "render"]),
    ({"max_slides_per_block": 2}, ["image", "audio", "render"]),
    ({"subtitle_font_size": 58, "voicevox_speed_scale": 1.2}, ["audio", "render"]),
])
def test_setting_changes_plan_only_required_stages(temp_storage, updates, stages):
    with get_session_factory()() as db:
        project = ready_project(db)
        apply_project_settings(project, updates)
        assert build_generation_plan(project)["stages"] == stages
        assert dependencies_for_settings(updates) == set(stages)


def test_fresh_request_with_unchanged_settings_still_renders_once(temp_storage):
    with get_session_factory()() as db:
        project = ready_project(db)
        before = capture_inputs(project)
        assert build_generation_plan(project)["stages"] == ["render"]
        assert build_generation_plan(project)["blocks"] == {"0": []}
        assert capture_inputs(project) == before


def test_missing_media_is_regenerated_and_split_does_not_erase_edits(temp_storage):
    with get_session_factory()() as db:
        project = ready_project(db)
        block_audio_path(project.id, 0).unlink()
        assert build_generation_plan(project)["stages"] == ["audio", "render"]
        assert project.blocks[0].tts_text == "編集した読み上げ。"


def test_initial_job_plans_every_required_stage(temp_storage):
    with get_session_factory()() as db:
        project = Project(title="new", source_script="合成テスト。", use_fake_providers=True)
        db.add(project)
        db.commit()
        assert build_generation_plan(project)["stages"] == ["split", "plan", "image", "audio", "render"]


def test_snapshot_changes_for_block_edits_but_not_progress(temp_storage):
    with get_session_factory()() as db:
        project = ready_project(db)
        original = fingerprint_inputs(capture_inputs(project))
        project.progress = .8
        project.output_video_path = "old.mp4"
        assert fingerprint_inputs(capture_inputs(project)) == original
        project.blocks[0].tts_text = "違う入力。"
        assert fingerprint_inputs(capture_inputs(project)) != original


def test_snapshot_is_detached_from_nested_json_and_contains_no_secret(temp_storage):
    with get_session_factory()() as db:
        project = ready_project(db)
        snapshot = capture_inputs(project)
        project.blocks[0].visual_plan_json["heading"] = "modified"
        assert snapshot["blocks"][0]["visual_plan"]["heading"] == "合成テスト"
        assert "api_key" not in str(snapshot)
