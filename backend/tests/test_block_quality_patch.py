"""Manual visual edits keep the renderer selector and adaptive audio coherent."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.db import get_db, get_session_factory
from app.main import create_app
from app.models.block import Block, BlockStatus, VisualType
from app.models.project import Project


AI_PLAN = {"visual_type": "ai_image", "heading": "以前の図", "image_prompt": "old picture"}
SLIDE_PLAN = {"visual_type": "verbatim_slide", "heading": "キーと値", "verbatim": "key -> value"}


@pytest.fixture()
def block_client(temp_storage):
    with get_session_factory()() as session:
        project = Project(title="編集テスト", source_script="keyを説明します。", use_fake_providers=True)
        session.add(project)
        session.flush()
        block = Block(
            project_id=project.id, index=0,
            source_text="keyを説明します。", tts_text="keyを説明します。",
            visual_type=VisualType.ai_image, visual_plan_json=AI_PLAN,
            image_prompt="old picture", video_path="last-completed.mp4",
            status_visual_plan=BlockStatus.completed,
            status_image=BlockStatus.completed, status_audio=BlockStatus.completed,
            status_render=BlockStatus.completed,
        )
        session.add(block)
        session.commit()
        block_id = block.id
    return TestClient(create_app()), block_id


def test_ai_to_local_plan_updates_selector_and_invalidates_adaptive_audio(block_client):
    client, block_id = block_client
    response = client.patch(f"/api/blocks/{block_id}", json={"visual_plan": SLIDE_PLAN})
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["visual_type"] == "verbatim_slide"
    assert data["visual_plan"] == SLIDE_PLAN
    assert data["image_prompt"] is None
    assert data["status_visual_plan"] == "completed"
    assert data["status_image"] == data["status_audio"] == data["status_render"] == "pending"
    with get_session_factory()() as session:
        block = session.get(Block, block_id)
        assert block.visual_type == VisualType.verbatim_slide
        assert block.video_path == "last-completed.mp4"


@pytest.mark.asyncio
async def test_patched_local_plan_never_calls_remote_image_provider(block_client, monkeypatch):
    from app.services import pipeline

    client, block_id = block_client
    assert client.patch(f"/api/blocks/{block_id}", json={"visual_plan": SLIDE_PLAN}).status_code == 200
    rendered = []

    def render_local(plan, output, **kwargs):
        rendered.append(plan["visual_type"])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"local-render-placeholder")

    monkeypatch.setattr(pipeline.image_renderer, "render_visual_plan", render_local)
    remote_image = SimpleNamespace(generate_image=AsyncMock())
    with get_session_factory()() as session:
        block = session.get(Block, block_id)
        context = SimpleNamespace(project=block.project, settings=get_settings(), bundle=SimpleNamespace(image=remote_image))
        await pipeline._render_block_image(context, block, "", session)
    assert rendered == ["verbatim_slide"]
    remote_image.generate_image.assert_not_awaited()


def test_local_to_ai_plan_copies_prompt_and_keeps_fixed_audio(block_client):
    client, block_id = block_client
    with get_session_factory()() as session:
        block = session.get(Block, block_id)
        block.visual_type = VisualType.verbatim_slide
        block.visual_plan_json = SLIDE_PLAN
        block.image_prompt = None
        block.project.narration_pacing_mode = "fixed"
        session.commit()
    response = client.patch(f"/api/blocks/{block_id}", json={"visual_plan": AI_PLAN})
    assert response.status_code == 200, response.text
    assert response.json()["visual_type"] == "ai_image"
    assert response.json()["image_prompt"] == "old picture"
    assert response.json()["status_audio"] == "completed"
    assert response.json()["status_image"] == response.json()["status_render"] == "pending"


@pytest.mark.parametrize("plan, expected_type", [(SLIDE_PLAN, "verbatim_slide"), ({"heading": "新しい図"}, "text_slide")])
def test_unplanned_block_accepts_visual_patch_without_previous_type(block_client, plan, expected_type):
    client, block_id = block_client
    with get_session_factory()() as session:
        block = session.get(Block, block_id)
        # A not-yet-flushed planning state may have no selector. Keep the API
        # robust before SQLAlchemy has applied its insertion default.
        block.visual_type = None

        def override_db():
            yield session

        client.app.dependency_overrides[get_db] = override_db
        try:
            response = client.patch(f"/api/blocks/{block_id}", json={"visual_plan": plan})
        finally:
            client.app.dependency_overrides.clear()
    assert response.status_code == 200, response.text
    assert response.json()["visual_type"] == expected_type


@pytest.mark.parametrize("invalid_type", ["unknown_renderer", None, ["verbatim_slide"]])
def test_invalid_visual_type_rejects_all_edits_before_writing(block_client, invalid_type):
    client, block_id = block_client
    before = client.get(f"/api/blocks/{block_id}").json()
    response = client.patch(f"/api/blocks/{block_id}", json={
        "source_text": "更新した本文。", "tts_text": "更新した音声。",
        "visual_plan": {"visual_type": invalid_type},
    })
    assert response.status_code == 422, response.text
    assert client.get(f"/api/blocks/{block_id}").json() == before


@pytest.mark.parametrize("field", ["source_text", "tts_text"])
def test_null_text_rejected_without_saving_valid_plan_edit(block_client, field):
    client, block_id = block_client
    before = client.get(f"/api/blocks/{block_id}").json()
    response = client.patch(f"/api/blocks/{block_id}", json={field: None, "visual_plan": SLIDE_PLAN})
    assert response.status_code == 422, response.text
    assert client.get(f"/api/blocks/{block_id}").json() == before


def test_unchanged_plan_does_not_invalidate_completed_stages(block_client):
    client, block_id = block_client
    before = client.get(f"/api/blocks/{block_id}").json()
    response = client.patch(f"/api/blocks/{block_id}", json={"visual_plan": AI_PLAN})
    assert response.status_code == 200, response.text
    assert response.json() == before


def test_explicit_plan_takes_precedence_over_source_replanning(block_client):
    client, block_id = block_client
    response = client.patch(f"/api/blocks/{block_id}", json={
        "source_text": "新しい本文。", "visual_plan": SLIDE_PLAN,
    })
    assert response.status_code == 200, response.text
    assert response.json()["status_visual_plan"] == "completed"
    assert response.json()["visual_plan"] == SLIDE_PLAN


def test_null_plan_requests_replanning_and_clears_stale_metadata(block_client):
    client, block_id = block_client
    response = client.patch(f"/api/blocks/{block_id}", json={"visual_plan": None})
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["visual_plan"] is None
    assert data["visual_type"] == "text_slide"
    assert data["image_prompt"] is None
    assert data["status_visual_plan"] == data["status_image"] == data["status_audio"] == "pending"


def test_missing_selector_inherits_current_type_and_invalid_prompt_is_rejected(block_client):
    client, block_id = block_client
    response = client.patch(f"/api/blocks/{block_id}", json={"visual_plan": {"heading": "新しい図", "image_prompt": "new picture"}})
    assert response.status_code == 200, response.text
    assert response.json()["visual_plan"]["visual_type"] == "ai_image"
    assert response.json()["image_prompt"] == "new picture"
    before = response.json()
    response = client.patch(f"/api/blocks/{block_id}", json={"source_text": "更新本文。", "visual_plan": {"visual_type": "ai_image", "image_prompt": 42}})
    assert response.status_code == 422, response.text
    assert client.get(f"/api/blocks/{block_id}").json() == before
