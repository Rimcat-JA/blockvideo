"""HTTP API smoke tests."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.core import config
from app.main import create_app


@pytest.fixture()
def client(temp_storage):
    config.reset_settings_cache()
    app = create_app()
    return TestClient(app)


def test_health_endpoint(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "version" in data


def test_create_and_get_project_with_fake_providers(client):
    payload = {
        "title": "テスト動画",
        "source_script": "これはテスト用の日本語台本です。短めの文章で構成しています。",
        "use_fake_providers": True,
        "voicevox_url": "http://127.0.0.1:50021",
        "voicevox_speaker_id": 1,
    }
    r = client.post("/api/projects", json=payload)
    assert r.status_code == 201, r.text
    project = r.json()
    assert project["id"]
    pid = project["id"]
    r2 = client.get(f"/api/projects/{pid}")
    assert r2.status_code == 200
    r3 = client.get(f"/api/projects/{pid}/blocks")
    assert r3.status_code == 200
    assert isinstance(r3.json(), list)


def test_validation_rejects_bad_url(client):
    payload = {
        "title": "t",
        "source_script": "本文",
        "voicevox_url": "not-a-url",
    }
    r = client.post("/api/projects", json=payload)
    assert r.status_code == 422


def test_download_404_before_video_exists(client):
    payload = {
        "title": "t",
        "source_script": "本文",
        "use_fake_providers": True,
    }
    pid = client.post("/api/projects", json=payload).json()["id"]
    r = client.get(f"/api/projects/{pid}/download")
    assert r.status_code == 404


def test_secret_summary_masks_keys(client):
    payload = {
        "title": "t",
        "source_script": "本文",
        "use_fake_providers": False,
        "providers": {
            "llm_api_key": "sk-proj-abcdefghijklmnop",
            "llm_base_url": "https://api.openai.com/v1",
            "llm_model": "gpt-4o-mini",
        },
    }
    # Note: validation should still accept this (provider key is opaque text).
    # The endpoint just stores the secret bundle in memory. We don't expose
    # it back via the GET endpoint.
    r = client.post("/api/projects", json=payload)
    assert r.status_code == 201
    pid = r.json()["id"]
    r2 = client.get(f"/api/projects/{pid}")
    assert r2.status_code == 200
    body = r2.text
    assert "sk-proj-abcdefghijklmnop" not in body