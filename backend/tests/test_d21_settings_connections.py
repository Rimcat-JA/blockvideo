"""D21: the connected settings retain specialist rules across entry points."""
from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from app.db import get_session_factory
from app.models.job import GenerationJob
from app.models.project import Project
from app.operations.bootstrap import operation_service
from app.schemas import ProjectPatch
from app.services.generation_plan import build_generation_plan
from tests.test_generation_plan import ready_project
from tests.test_language_operations import harness as harness, language_input


def send_settings(client: TestClient, adapter: Any, entry: str, project_id: int,
                  changes: dict[str, Any], text: str) -> Any:
    if entry == "patch":
        return client.patch(f"/api/projects/{project_id}", json=changes)
    if entry == "language":
        adapter.operation("project.settings.update", changes)
        return client.post("/api/language/requests", json=language_input(
            project_id, request_id=f"d21-{entry}-{project_id}", text=text, base_revision=1))
    return client.post("/api/operations/execute", json={
        "request_id": f"d21-{entry}-{project_id}", "operation_id": "project.settings.update",
        "operation_version": 1, "target": {"project_id": project_id}, "base_revision": 1,
        "arguments": changes,
    })


@pytest.mark.parametrize("changes,text,stages", [
    ({"subtitle_mode": "packed"}, "字幕を複数の文でまとめて", ["render"]),
    ({"voicevox_speed_scale": 1.25}, "話す速さを1.25倍にして", ["audio", "render"]),
    ({"voicevox_speaker_id": 2}, "話者IDを2にして", ["audio", "render"]),
    ({"pronunciation_overrides": [{"surface": "API", "reading": "エーピーアイ", "accent": None}]},
     "APIはエーピーアイと読んで", ["audio", "render"]),
    ({"narration_pacing_mode": "fixed"}, "文末の間を一定にして", ["audio", "render"]),
])
def test_saved_settings_and_stage_effects_match_across_entries(
    harness: Any, changes: dict[str, Any], text: str, stages: list[str],
) -> None:
    client, adapter, _ = harness
    with get_session_factory()() as db:
        ids = [ready_project(db).id for _ in range(4)]
    for entry, project_id in zip(("core", "language", "patch"), ids[:3], strict=True):
        response = send_settings(client, adapter, entry, project_id, changes, text)
        assert response.status_code == 200, response.text
        if entry == "language":
            assert response.json()["status"] == "completed"
        saved = client.get(f"/api/projects/{project_id}").json()
        assert saved["revision"] == 2
        assert all(saved[field] == value for field, value in changes.items())
        with get_session_factory()() as db:
            project = db.get(Project, project_id)
            assert build_generation_plan(project)["stages"] == stages
            assert project.source_script == "元の台本。"
            assert project.blocks[0].source_text == "編集した台本。"
            assert project.blocks[0].tts_text == "編集した読み上げ。"
            assert db.scalar(select(func.count()).select_from(GenerationJob)) == 0
    untouched = client.get(f"/api/projects/{ids[-1]}").json()
    assert untouched["revision"] == 1 and untouched["subtitle_mode"] == "sentence"
    assert untouched["voicevox_speed_scale"] == 1.0 and untouched["voicevox_speaker_id"] == 1
    assert untouched["pronunciation_overrides"] == []


@pytest.mark.parametrize("changes,text", [
    ({"voicevox_speed_scale": 3.0}, "話速を3倍にして"),
    ({"voicevox_speaker_id": -1}, "話者IDを-1にして"),
    ({"subtitle_mode": "word"}, "字幕を単語ごとにして"),
    ({"pronunciation_overrides": [{"surface": "API", "reading": "ャピ", "accent": None}]},
     "APIはャピと読んで"),
    ({"pronunciation_overrides": [{"surface": "API", "reading": "エーピーアイ", "accent": 99}]},
     "APIはエーピーアイと読んで、アクセントは99"),
])
def test_invalid_specialist_input_never_partially_saves(
    harness: Any, changes: dict[str, Any], text: str,
) -> None:
    client, adapter, _ = harness
    with get_session_factory()() as db:
        ids = [ready_project(db).id for _ in range(3)]
    for entry, project_id in zip(("core", "language", "patch"), ids, strict=True):
        before = client.get(f"/api/projects/{project_id}").json()
        # A valid unrelated field must not survive a rejected compound settings object.
        response = send_settings(client, adapter, entry, project_id, {"subtitle_font_size": 72, **changes}, text)
        if entry == "language":
            assert response.status_code == 200
            assert response.json()["status"] in {"error", "needs_input", "blocked"}
        else:
            assert response.status_code in {400, 422}, response.text
        assert client.get(f"/api/projects/{project_id}").json() == before
    with get_session_factory()() as db:
        assert db.scalar(select(func.count()).select_from(GenerationJob)) == 0


def test_catalog_exposes_every_patchable_setting_without_extra_handlers() -> None:
    definition = next(item for item in operation_service.list_definitions()
                      if item.operation_id == "project.settings.update")
    assert set(definition.input_schema["properties"]) == set(ProjectPatch.model_fields)
    # Settings stay one operation; the ninth ID is the later artifact restore.
    assert len({item.operation_id for item in operation_service.list_definitions()}) == 9
