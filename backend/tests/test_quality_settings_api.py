"""Quality settings survive API round trips and upgrade legacy databases safely."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.db import _add_missing_columns, get_engine, get_session_factory
from app.main import create_app
from app.models.job import GenerationJob
from app.models.project import Project


@pytest.fixture()
def client(temp_storage):
    return TestClient(create_app())


INPUT = {"title": "読みやすい動画", "source_script": "APIを説明します。図の矢印を見てください。", "use_fake_providers": True}
OVERRIDES = [{"surface": "API", "reading": "エーピーアイ", "accent": 0}]
QUALITY = {
    "visual_focus_enabled": False,
    "subtitle_mode": "packed",
    "narration_pacing_mode": "fixed",
    "pronunciation_overrides": OVERRIDES,
    "narration_sentence_pause_seconds": 0.9,
    "max_slides_per_block": 3,
}


def test_quality_defaults_and_patch_round_trip(client):
    created = client.post("/api/projects", json=INPUT)
    assert created.status_code == 201, created.text
    project = created.json()
    assert project["visual_focus_enabled"] is True
    assert project["subtitle_mode"] == "sentence"
    assert project["narration_pacing_mode"] == "adaptive"
    assert project["pronunciation_overrides"] == []
    patched = client.patch(f"/api/projects/{project['id']}", json=QUALITY)
    assert patched.status_code == 200, patched.text
    retrieved = client.get(f"/api/projects/{project['id']}").json()
    assert {key: retrieved[key] for key in QUALITY} == QUALITY
    # Empty PATCH is legal and keeps saved settings.
    assert client.patch(f"/api/projects/{project['id']}", json={}).json() == retrieved


def test_detailed_create_preserves_quality_settings(client):
    response = client.post("/api/projects", json={**INPUT, **QUALITY})
    assert response.status_code == 201, response.text
    assert {key: response.json()[key] for key in QUALITY} == QUALITY


def test_quick_create_preserves_quality_settings(client, monkeypatch):
    async def enqueue(project_id):
        with get_session_factory()() as session:
            job = GenerationJob(project_id=project_id)
            session.add(job)
            session.commit()
            session.refresh(job)
            session.expunge(job)
            return job

    monkeypatch.setattr("app.api.routes_projects.enqueue_full_pipeline", enqueue)
    response = client.post("/api/projects/quick", json={**INPUT, **QUALITY})
    assert response.status_code == 201, response.text
    project = response.json()["project"]
    assert {key: project[key] for key in QUALITY} == QUALITY


@pytest.mark.parametrize("field", list(QUALITY))
def test_patch_rejects_explicit_null_quality_settings(client, field):
    project_id = client.post("/api/projects", json=INPUT).json()["id"]
    response = client.patch(f"/api/projects/{project_id}", json={field: None})
    assert response.status_code == 422, response.text


@pytest.mark.parametrize("entry", [
    {"surface": "", "reading": "エー"},
    {"surface": "  ", "reading": "エー"},
    {"surface": "A。", "reading": "エー"},
    {"surface": "A", "reading": "えー"},
    {"surface": "A", "reading": "ーエ"},
    {"surface": "A", "reading": "エー", "accent": -1},
    {"surface": "A", "reading": "キャット", "accent": 4},
    {"surface": "A", "reading": "エー", "accent": True},
])
def test_invalid_pronunciations_rejected_before_creation(client, entry):
    response = client.post("/api/projects", json={**INPUT, "pronunciation_overrides": [entry]})
    assert response.status_code == 422, response.text


def test_duplicate_and_excessive_pronunciations_rejected(client):
    for entries in [OVERRIDES * 2, [{"surface": str(i), "reading": "エー"} for i in range(101)]]:
        response = client.post("/api/projects", json={**INPUT, "pronunciation_overrides": entries})
        assert response.status_code == 422, response.text


def test_legacy_database_keeps_previous_quality_modes(temp_storage):
    factory = get_session_factory()
    with factory() as session:
        legacy = Project(title="以前の動画", source_script="本文", narration_sentence_pause_seconds=2.3)
        session.add(legacy)
        session.commit()
        legacy_id = legacy.id
    engine = get_engine()
    # Recreate the previous schema using only the four newly introduced columns.
    with engine.begin() as connection:
        for name in ("visual_focus_enabled", "subtitle_mode", "narration_pacing_mode", "pronunciation_overrides"):
            connection.exec_driver_sql(f"ALTER TABLE projects DROP COLUMN {name}")
    engine.dispose()  # Simulate opening the legacy DB in a fresh application process.
    _add_missing_columns(engine)
    _add_missing_columns(engine)  # Startup is idempotent.
    with factory() as session:
        legacy = session.get(Project, legacy_id)
        assert legacy.visual_focus_enabled is False
        assert legacy.subtitle_mode == "packed"
        assert legacy.narration_pacing_mode == "fixed"
        assert legacy.pronunciation_overrides == []
        assert legacy.narration_sentence_pause_seconds == 2.3
        fresh = Project(title="新しい動画", source_script="本文")
        session.add(fresh)
        session.commit()
        assert fresh.visual_focus_enabled is True
        assert fresh.subtitle_mode == "sentence"
        assert fresh.narration_pacing_mode == "adaptive"
        assert fresh.pronunciation_overrides == []
