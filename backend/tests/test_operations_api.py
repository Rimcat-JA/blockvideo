"""Thin HTTP entry tests for the common operation core."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.models.block import Block, BlockStatus
from app.models.project import Project
from app.db import get_session_factory


@pytest.fixture()
def client(temp_storage):
    return TestClient(create_app())


def _create(client: TestClient) -> int:
    response = client.post(
        "/api/projects",
        json={
            "title": "Plan C sample",
            "source_script": "操作コアの合成サンプルです。",
            "subtitle_font_size": 48,
            "use_fake_providers": True,
        },
    )
    assert response.status_code == 201, response.text
    return response.json()["id"]


def _request(project_id: int | None, value: object = 56) -> dict:
    return {
        "operation_id": "project.subtitle-font-size.set",
        "operation_version": 1,
        "target": {"project_id": project_id},
        "arguments": {"value": value},
    }


def test_lists_definitions_from_the_catalog(client: TestClient) -> None:
    response = client.get("/api/operations")
    assert response.status_code == 200
    assert [item["operation_id"] for item in response.json()] == [
        "project.artifact.restore",
        "project.generation.cancel",
        "project.generation.retry",
        "project.generation.start",
        "project.settings.restore",
        "project.settings.update",
        "project.settings.update",  # v2 with atomic relative settings.
        "project.status.get",
        "project.subtitle-font-size.adjust",
        "project.subtitle-font-size.set",
    ]


def test_readiness_reports_missing_target_as_409_without_executing(client: TestClient) -> None:
    response = client.post("/api/operations/readiness", json=_request(None))
    assert response.status_code == 409
    assert response.json()["detail"]["readiness"] == "needs_input"
    assert response.json()["detail"]["missing_fields"] == ["project_id"]


def test_conflicting_target_ids_are_422(client: TestClient) -> None:
    response = client.post(
        "/api/operations/readiness",
        json={**_request(1), "target": {"project_id": 1, "selected_project_id": 2}},
    )
    assert response.status_code == 422


def test_execute_persists_subtitle_size(client: TestClient) -> None:
    project_id = _create(client)
    response = client.post("/api/operations/execute", json=_request(project_id, 56))
    assert response.status_code == 200, response.text
    assert response.json()["data"] == {"subtitle_font_size": 56}
    assert client.get(f"/api/projects/{project_id}").json()["subtitle_font_size"] == 56


def test_status_execution_is_structured_and_read_only(client: TestClient) -> None:
    project_id = _create(client)
    response = client.post(
        "/api/operations/execute",
        json={
            "operation_id": "project.status.get",
            "target": {"selected_project_id": project_id},
            "arguments": {},
        },
    )
    assert response.status_code == 200
    assert response.json()["changed"] is False
    assert response.json()["data"]["status"] == "pending"


def test_unknown_operation_is_404(client: TestClient) -> None:
    response = client.post(
        "/api/operations/execute",
        json={"operation_id": "project.unknown", "target": {}, "arguments": {}},
    )
    assert response.status_code == 404
    assert response.json()["detail"]["reason_code"] == "operation_not_found"


def test_invalid_arguments_are_422_and_do_not_write(client: TestClient) -> None:
    project_id = _create(client)
    response = client.post("/api/operations/execute", json=_request(project_id, True))
    assert response.status_code == 422
    assert response.json()["detail"]["reason_code"] == "invalid_arguments"
    assert client.get(f"/api/projects/{project_id}").json()["subtitle_font_size"] == 48


def test_non_ready_execution_is_machine_readable_409(client: TestClient) -> None:
    response = client.post("/api/operations/execute", json=_request(None))
    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["reason_code"] == "target_required"
    assert detail["readiness"] == "needs_input"
    assert detail["missing_fields"] == ["project_id"]


def test_nonexistent_target_execution_is_409(client: TestClient) -> None:
    response = client.post("/api/operations/execute", json=_request(999))
    assert response.status_code == 409
    assert response.json()["detail"]["reason_code"] == "target_not_found"


def test_stale_execution_is_409_and_preserves_value(client: TestClient) -> None:
    project_id = _create(client)
    request = {**_request(project_id), "observed_state_revision": "stale"}
    response = client.post("/api/operations/execute", json=request)
    assert response.status_code == 409
    assert response.json()["detail"]["reason_code"] == "stale_state"
    assert client.get(f"/api/projects/{project_id}").json()["subtitle_font_size"] == 48


def test_busy_execution_is_409_and_preserves_value(client: TestClient, monkeypatch) -> None:
    project_id = _create(client)
    monkeypatch.setattr("app.operations.readiness.has_live_job", lambda _db, _id: True)
    response = client.post("/api/operations/execute", json=_request(project_id))
    assert response.status_code == 409
    assert response.json()["detail"]["reason_code"] == "project_busy"
    assert client.get(f"/api/projects/{project_id}").json()["subtitle_font_size"] == 48


def test_patch_and_operation_share_render_invalidation(client: TestClient) -> None:
    project_id = _create(client)
    factory = get_session_factory()
    with factory() as db:
        project = db.get(Project, project_id)
        assert project is not None
        db.add(Block(
            project=project,
            index=0,
            source_text="本文",
            tts_text="本文",
            status_render=BlockStatus.completed,
        ))
        db.commit()
    patched = client.patch(f"/api/projects/{project_id}", json={"subtitle_font_size": 54})
    assert patched.status_code == 200
    with factory() as db:
        project = db.get(Project, project_id)
        assert project is not None
        assert project.blocks[0].status_render == BlockStatus.pending
        project.blocks[0].status_render = BlockStatus.completed
        db.commit()
    executed = client.post("/api/operations/execute", json=_request(project_id, 58))
    assert executed.status_code == 200
    with factory() as db:
        project = db.get(Project, project_id)
        assert project is not None
        assert project.blocks[0].status_render == BlockStatus.pending
