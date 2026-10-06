"""D13 file integrity, stale completion, cancellation, and real fake-provider output."""
from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path

import pytest
from sqlalchemy import select

from app.db import get_session_factory
from app.models.artifact import GenerationArtifact
from app.models.job import GenerationJob, JobStatus
from app.models.project import Project
from app.services import artifact_store as store
from app.services.generation_plan import build_generation_plan
from app.services.generation_snapshots import (
    GenerationCancelled, StaleGenerationInput, capture_inputs, fingerprint_inputs,
)
from app.services.paths import project_dir, relpath_for_db


def make_job(db, project=None):
    if project is None:
        project = Project(title="合成テスト", source_script="これは短い合成テストです。", use_fake_providers=True)
        db.add(project)
        db.flush()
    snapshot = capture_inputs(project)
    job = GenerationJob(project_id=project.id, status=JobStatus.running, kind="full",
                        input_revision=project.revision, input_snapshot=snapshot,
                        input_fingerprint=fingerprint_inputs(snapshot), plan_json=build_generation_plan(project))
    db.add(job)
    db.commit()
    return project, job


def candidate_for(project_id: int, job_id: int, contents: bytes = b"verified-by-test-probe") -> Path:
    path = project_dir(project_id) / "history" / f"job-{job_id:08d}" / "video.pending.mp4"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(contents)
    return path


@pytest.fixture
def accept_synthetic_probe(monkeypatch):
    async def probe(path):
        return {**store.file_identity(path), "duration_ms": 1000, "width": 320, "height": 180}
    monkeypatch.setattr(store, "validate_video", probe)


@pytest.mark.asyncio
async def test_all_successes_same_revision_have_distinct_immutable_files(temp_storage, accept_synthetic_probe):
    with get_session_factory()() as db:
        project, job = make_job(db)
        project_id, first_id = project.id, job.id
        snapshot = capture_inputs(project)
    first = await store.publish_artifact(first_id, candidate_for(project_id, first_id), None,
                                         settled_inputs=snapshot, materials=[], cancel_check=lambda: False)
    with get_session_factory()() as db:
        project = db.get(Project, project_id)
        _, job = make_job(db, project)
        second_id = job.id
    second = await store.publish_artifact(second_id, candidate_for(project_id, second_id, b"second"), None,
                                          settled_inputs=snapshot, materials=[], cancel_check=lambda: False)
    assert first.id != second.id
    assert first.video_path != second.video_path
    assert store.artifact_file_path(first).read_bytes() == b"verified-by-test-probe"
    with get_session_factory()() as db:
        project = db.get(Project, project_id)
        assert project.current_artifact_id == second.id
        assert store.artifact_to_summary(second, project)["is_current"]
        assert not store.artifact_to_summary(first, project)["is_current"]
        assert len(list(db.scalars(select(GenerationArtifact)))) == 2


@pytest.mark.asyncio
async def test_old_revision_completion_cannot_publish_history_or_replace_current(
    temp_storage, accept_synthetic_probe
):
    with get_session_factory()() as db:
        project, job = make_job(db)
        project_id, old_job_id, old_snapshot = project.id, job.id, capture_inputs(project)
        project.subtitle_font_size += 2
        project.revision += 1
        db.commit()
        _, new_job = make_job(db, project)
        new_id, current_snapshot = new_job.id, capture_inputs(project)
    current = await store.publish_artifact(new_id, candidate_for(project_id, new_id, b"new"), None,
                                           settled_inputs=current_snapshot, materials=[], cancel_check=lambda: False)
    stale_candidate = candidate_for(project_id, old_job_id, b"old")
    with pytest.raises(StaleGenerationInput):
        await store.publish_artifact(old_job_id, stale_candidate, None,
                                     settled_inputs=old_snapshot, materials=[], cancel_check=lambda: False)
    assert stale_candidate.read_bytes() == b"old"
    assert not stale_candidate.with_name("video.mp4").exists()
    with get_session_factory()() as db:
        project = db.get(Project, project_id)
        assert project.current_artifact_id == current.id
        assert list(db.scalars(select(GenerationArtifact))) == [db.get(GenerationArtifact, current.id)]


@pytest.mark.asyncio
async def test_same_revision_wrong_input_is_rejected(temp_storage, accept_synthetic_probe):
    with get_session_factory()() as db:
        project, job = make_job(db)
        snapshot, project_id, job_id = capture_inputs(project), project.id, job.id
        project.source_script = "差し替えられた入力。"
        db.commit()
    with pytest.raises(StaleGenerationInput):
        await store.publish_artifact(job_id, candidate_for(project_id, job_id), None,
                                     settled_inputs=snapshot, materials=[], cancel_check=lambda: False)
    with get_session_factory()() as db:
        assert list(db.scalars(select(GenerationArtifact))) == []


@pytest.mark.asyncio
async def test_cancel_at_publication_wins_without_history(temp_storage, accept_synthetic_probe):
    with get_session_factory()() as db:
        project, job = make_job(db)
        snapshot, project_id, job_id = capture_inputs(project), project.id, job.id
        job.cancel_requested = True
        db.commit()
    with pytest.raises(GenerationCancelled):
        await store.publish_artifact(job_id, candidate_for(project_id, job_id), None,
                                     settled_inputs=snapshot, materials=[], cancel_check=lambda: False)
    with get_session_factory()() as db:
        assert list(db.scalars(select(GenerationArtifact))) == []
        assert db.get(Project, project_id).output_video_path is None


@pytest.mark.asyncio
async def test_swapped_material_blocks_publication(temp_storage, accept_synthetic_probe):
    with get_session_factory()() as db:
        project, job = make_job(db)
        snapshot, project_id, job_id = capture_inputs(project), project.id, job.id
    material = project_dir(project_id) / "audio.wav"
    material.parent.mkdir(parents=True, exist_ok=True)
    material.write_bytes(b"first audio")
    manifest = [store.file_identity(material)]
    material.write_bytes(b"other audio")
    with pytest.raises(StaleGenerationInput):
        await store.publish_artifact(job_id, candidate_for(project_id, job_id), None,
                                     settled_inputs=snapshot, materials=manifest, cancel_check=lambda: False)


@pytest.mark.asyncio
async def test_deleted_or_corrupt_history_is_unavailable(temp_storage, accept_synthetic_probe):
    with get_session_factory()() as db:
        project, job = make_job(db)
        snapshot, project_id, job_id = capture_inputs(project), project.id, job.id
    artifact = await store.publish_artifact(job_id, candidate_for(project_id, job_id), None,
                                            settled_inputs=snapshot, materials=[], cancel_check=lambda: False)
    path = store.artifact_file_path(artifact)
    path.write_bytes(b"corrupted")
    assert not store.artifact_is_available(artifact)
    path.unlink()
    assert not store.artifact_is_available(artifact)


@pytest.mark.asyncio
async def test_import_legacy_video_keeps_unknown_revision_and_never_overwrites(temp_storage, accept_synthetic_probe):
    with get_session_factory()() as db:
        project, _job = make_job(db)
        project_id = project.id
        original = project_dir(project_id) / "output" / "video.mp4"
        original.parent.mkdir(parents=True, exist_ok=True)
        original.write_bytes(b"legacy video")
        project.output_video_path = relpath_for_db(original)
        db.commit()
    await store.preserve_legacy_video(project_id)
    await store.preserve_legacy_video(project_id)
    with get_session_factory()() as db:
        artifacts = list(db.scalars(select(GenerationArtifact)))
        assert len(artifacts) == 1
        assert artifacts[0].revision is None
        assert store.artifact_file_path(artifacts[0]).read_bytes() == b"legacy video"
        assert original.read_bytes() == b"legacy video"


@pytest.mark.asyncio
async def test_real_fake_pipeline_and_subtitle_only_regeneration(temp_storage, skip_if_no_ffmpeg, monkeypatch):
    from app.core.config import get_settings
    from app.models.block import Block
    from app.services import pipeline
    from app.services.project_settings import apply_project_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "output_width", 640)
    monkeypatch.setattr(settings, "output_height", 360)
    monkeypatch.setattr(settings, "output_fps", 12)
    monkeypatch.setattr(settings, "subtitle_band_height", 100)
    with get_session_factory()() as db:
        project, job = make_job(db)
        project_id, first_id = project.id, job.id
    await pipeline.run_generation_job(first_id, lambda: False)
    with get_session_factory()() as db:
        first = db.scalar(select(GenerationArtifact).where(GenerationArtifact.job_id == first_id))
        assert first and store.artifact_is_available(first)
        project = db.get(Project, project_id)
        old_text = db.scalar(select(Block).where(Block.project_id == project_id)).tts_text
        apply_project_settings(project, {"subtitle_font_size": 56})
        db.commit()
        _, second_job = make_job(db, project)
        second_id = second_job.id
        assert second_job.plan_json["stages"] == ["render"]

    async def forbidden(*args, **kwargs):
        pytest.fail("subtitle-only change must not invoke upstream stages")
    for name in ("run_split_stage", "run_visual_plan_stage", "run_image_stage", "run_audio_stage"):
        monkeypatch.setattr(pipeline, name, forbidden)
    await pipeline.run_generation_job(second_id, lambda: False)
    with get_session_factory()() as db:
        artifacts = list(db.scalars(select(GenerationArtifact)))
        assert len(artifacts) == 2
        assert all(store.artifact_is_available(artifact) for artifact in artifacts)
        assert db.scalar(select(Block).where(Block.project_id == project_id)).tts_text == old_text


def test_storage_sibling_prefix_is_rejected(temp_storage):
    with pytest.raises(ValueError):
        store.safe_storage_path(temp_storage.with_name(temp_storage.name + "-outside") / "video.mp4")


@pytest.mark.asyncio
async def test_cancellation_after_probe_still_prevents_publish(temp_storage, monkeypatch):
    with get_session_factory()() as db:
        project, job = make_job(db)
        snapshot, project_id, job_id = capture_inputs(project), project.id, job.id

    async def cancelling_probe(path):
        with get_session_factory()() as db:
            db.get(GenerationJob, job_id).cancel_requested = True
            db.commit()
        await asyncio.sleep(0)
        return {**store.file_identity(path), "duration_ms": 1000}
    monkeypatch.setattr(store, "validate_video", cancelling_probe)
    with pytest.raises(GenerationCancelled):
        await store.publish_artifact(job_id, candidate_for(project_id, job_id), None,
                                     settled_inputs=snapshot, materials=[], cancel_check=lambda: False)


@pytest.mark.asyncio
async def test_database_manifest_is_complete_without_artifact_manifest_file(
    temp_storage, accept_synthetic_probe
):
    with get_session_factory()() as db:
        project, job = make_job(db)
        project_id, first_id, snapshot = project.id, job.id, capture_inputs(project)
    first = await store.publish_artifact(first_id, candidate_for(project_id, first_id), None,
                                         settled_inputs=snapshot, materials=[], cancel_check=lambda: False)
    original = store.artifact_file_path(first).read_bytes()
    with get_session_factory()() as db:
        _, job = make_job(db, db.get(Project, project_id))
        second_id = job.id
    second = await store.publish_artifact(
        second_id,
        candidate_for(project_id, second_id, b"new"),
        None,
        settled_inputs=snapshot,
        materials=[],
        cancel_check=lambda: False,
    )

    assert store.artifact_file_path(first).read_bytes() == original
    assert store.artifact_file_path(second).read_bytes() == b"new"
    first_manifest = (
        project_dir(project_id) / "history" / f"job-{first_id:08d}" / "manifest.json"
    )
    second_directory = project_dir(project_id) / "history" / f"job-{second_id:08d}"
    assert not first_manifest.exists()
    assert not (second_directory / "manifest.json").exists()
    with get_session_factory()() as db:
        artifacts = list(db.scalars(select(GenerationArtifact).order_by(GenerationArtifact.id)))
        assert len(artifacts) == 2
        persisted = artifacts[1]
        assert db.get(Project, project_id).current_artifact_id == persisted.id == second.id
        assert set(persisted.manifest_json) == {
            "schema_version",
            "job_id",
            "revision",
            "requested_input_fingerprint",
            "settled_inputs",
            "input_fingerprint",
            "materials",
            "video",
            "subtitle",
            "block_videos",
            "metadata",
        }
        assert persisted.manifest_json["schema_version"] == 1
        assert persisted.manifest_json["job_id"] == second_id
        assert persisted.manifest_json["revision"] == 1
        assert persisted.manifest_json["requested_input_fingerprint"] == fingerprint_inputs(snapshot)
        assert persisted.manifest_json["settled_inputs"] == snapshot
        assert persisted.manifest_json["input_fingerprint"] == fingerprint_inputs(snapshot)
        assert persisted.manifest_json["materials"] == []
        assert persisted.manifest_json["subtitle"] is None
        assert persisted.manifest_json["block_videos"] == []
        assert persisted.manifest_json["metadata"] == []
        assert persisted.manifest_json["video"] == {
            "path": relpath_for_db(second_directory / "video.mp4"),
            "size": 3,
            "sha256": hashlib.sha256(b"new").hexdigest(),
            "duration_ms": 1000,
            "width": 320,
            "height": 180,
        }


@pytest.mark.asyncio
async def test_unknown_remote_call_prevents_final_publication(temp_storage, accept_synthetic_probe):
    from app.models.external_call import ExternalCall
    from app.services.external_calls import ExternalOutcomeUnknown

    with get_session_factory()() as db:
        project, job = make_job(db)
        project_id, job_id, snapshot = project.id, job.id, capture_inputs(project)
        db.add(ExternalCall(job_id=job_id, fingerprint="a" * 64, provider="test",
                            endpoint="https://example.invalid/generate", remote_side_effect=True,
                            status="unknown"))
        db.commit()
    with pytest.raises(ExternalOutcomeUnknown):
        await store.publish_artifact(job_id, candidate_for(project_id, job_id), None,
                                     settled_inputs=snapshot, materials=[], cancel_check=lambda: False)
    with get_session_factory()() as db:
        assert list(db.scalars(select(GenerationArtifact))) == []


@pytest.mark.asyncio
async def test_unknown_plan_result_cannot_be_hidden_as_partial_failure(temp_storage, monkeypatch):
    from app.core.config import get_settings
    from app.models.block import Block
    from app.services import pipeline
    from app.services.external_calls import ExternalOutcomeUnknown
    from app.services.provider_factory import build_providers_for_project, build_voicevox_settings

    async def unknown(*args, **kwargs):
        raise ExternalOutcomeUnknown("injected unknown outcome")
    monkeypatch.setattr(pipeline, "generate_visual_plan", unknown)
    with get_session_factory()() as db:
        project, _job = make_job(db)
        project.global_visual_style = "synthetic"
        db.add(Block(project_id=project.id, index=0, source_text="合成テスト。", tts_text="合成テスト。"))
        db.commit()
        db.refresh(project)
        ctx = pipeline.StageContext(project, get_settings(), build_providers_for_project(project),
                                    build_voicevox_settings(project))
        with pytest.raises(ExternalOutcomeUnknown):
            await pipeline.run_visual_plan_stage(ctx, db)


@pytest.mark.asyncio
async def test_empty_video_is_not_a_success(temp_storage):
    with get_session_factory()() as db:
        project, job = make_job(db)
        project_id, job_id = project.id, job.id
    path = candidate_for(project_id, job_id, b"")
    with pytest.raises(FileNotFoundError):
        await store.validate_video(path)


@pytest.mark.asyncio
async def test_cancelled_job_never_starts_provider_work(temp_storage, monkeypatch):
    from app.services import pipeline

    with get_session_factory()() as db:
        _project, job = make_job(db)
        job_id = job.id
        job.cancel_requested = True
        db.commit()
    monkeypatch.setattr(pipeline, "_providers_for_stages", lambda *_args: pytest.fail("provider constructed"))
    with pytest.raises(GenerationCancelled):
        await pipeline.run_generation_job(job_id, lambda: False)


@pytest.mark.parametrize("manifest", [None, [], {}, {"video": {}}, {"video": {"sha256": "bad"}}])
def test_history_without_valid_manifest_cannot_be_served(temp_storage, manifest):
    path = temp_storage / "unverified.mp4"
    path.write_bytes(b"not verified")
    artifact = GenerationArtifact(project_id=1, video_path=relpath_for_db(path), manifest_json=manifest)
    assert not store.artifact_is_available(artifact)
    with pytest.raises(FileNotFoundError):
        store.artifact_file_path(artifact)
