"""Partial rerun semantics.

We don't spin up an HTTP server here; we exercise ``pipeline`` directly to
confirm that re-running one stage does not invalidate artifacts from
earlier stages.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.models.block import BlockStatus, VisualType
from app.services.hashing import hash_value, short_hash


def _make_project(db_session, fake: bool = True):
    from app.models.project import Project

    p = Project(
        title="demo",
        source_script="",
        use_fake_providers=fake,
        voicevox_url="http://127.0.0.1:50021",
        voicevox_speaker_id=1,
    )
    db_session.add(p)
    db_session.flush()
    return p


def _make_ctx(project, bundle):
    from app.core.config import get_settings

    return SimpleNamespace(
        project=project,
        settings=get_settings(),
        bundle=bundle,
        voicevox_settings=None,
        progress_cb=None,
        is_cancelled=lambda: False,
    )


@pytest.mark.asyncio
async def test_partial_rerun_only_touches_image_status(temp_storage) -> None:
    """Re-running the image stage should reset only status_image on the target."""
    from app.db import get_session_factory
    from app.models.block import Block
    from app.services import pipeline as pipe
    from app.services.paths import block_image_path, ensure_project_layout
    from app.services.provider_factory import build_providers_for_project

    factory = get_session_factory()
    db = factory()
    try:
        project = _make_project(db)
        blocks: list[Block] = []
        for i in range(3):
            b = Block(
                project_id=project.id,
                index=i,
                source_text=f"サンプルテキスト{i}。",
                tts_text=f"サンプルテキスト{i}。",
                visual_type=VisualType.title_slide,
                status_split=BlockStatus.completed,
                status_visual_plan=BlockStatus.completed,
                status_image=BlockStatus.completed,
                status_audio=BlockStatus.completed,
                status_render=BlockStatus.completed,
                content_hash=short_hash(i),
            )
            blocks.append(b)
            db.add(b)
        db.commit()
        target = blocks[1]
        target.status_image = BlockStatus.failed
        db.commit()

        called: list[int] = []

        async def fake_render(ctx, block, style, db):
            called.append(block.index)
            ensure_project_layout(ctx.project.id)
            output = block_image_path(ctx.project.id, block.index)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b"x")
            block.image_path = "fake"
            block.status_image = BlockStatus.completed
            db.commit()

        pipe._render_block_image = fake_render  # type: ignore

        ensure_project_layout(project.id)
        bundle = build_providers_for_project(project)
        ctx = _make_ctx(project, bundle)
        await pipe.run_image_stage(ctx, db)  # type: ignore
        db.commit()
        db.refresh(target)
        assert target.status_image == BlockStatus.completed
        assert called == [1]
        assert blocks[0].status_image == BlockStatus.completed
        assert blocks[2].status_image == BlockStatus.completed
    finally:
        db.close()


@pytest.mark.asyncio
async def test_hash_reflects_audio_settings_change(temp_storage) -> None:
    """content_hash must change when audio-relevant settings change."""
    base = hash_value("block-1", "短めのソースです。", "読み上げテキスト", "code_slide")
    changed_speed = hash_value(
        "block-1", "短めのソースです。", "読み上げテキスト", "code_slide", speed=1.2
    )
    assert base != changed_speed


@pytest.mark.asyncio
async def test_repeated_image_rerun_is_noop_when_complete(temp_storage) -> None:
    from app.db import get_session_factory
    from app.models.block import Block, BlockStatus, VisualType
    from app.models.project import Project
    from app.services import pipeline as pipe
    from app.services.paths import ensure_project_layout
    from app.services.provider_factory import build_providers_for_project

    factory = get_session_factory()
    db = factory()
    try:
        project = Project(
            title="t",
            source_script="x。",
            use_fake_providers=True,
            voicevox_url="http://127.0.0.1:50021",
            voicevox_speaker_id=1,
        )
        db.add(project)
        db.flush()
        b = Block(
            project_id=project.id,
            index=0,
            source_text="x。",
            tts_text="x。",
            visual_type=VisualType.text_slide,
            status_split=BlockStatus.completed,
            status_visual_plan=BlockStatus.completed,
            status_image=BlockStatus.completed,
            status_audio=BlockStatus.completed,
            status_render=BlockStatus.completed,
        )
        db.add(b)
        db.commit()

        called: list[int] = []

        async def fake_render(ctx, block, style, db):
            called.append(block.index)

        pipe._render_block_image = fake_render  # type: ignore
        ensure_project_layout(project.id)
        bundle = build_providers_for_project(project)
        ctx = _make_ctx(project, bundle)
        await pipe.run_image_stage(ctx, db)  # type: ignore
        assert called == []  # already complete
    finally:
        db.close()