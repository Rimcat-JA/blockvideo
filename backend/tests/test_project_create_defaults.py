"""New projects take omitted VOICEVOX and pacing values from the server settings."""
from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.core import config
from app.core.security import SecretBundle, secret_store
from app.main import create_app

INPUT = {"title": "既定値", "source_script": "APIを説明します。図を見てください。", "use_fake_providers": True}


@pytest.fixture()
def client(temp_storage: Any, monkeypatch: Any) -> TestClient:
    monkeypatch.setenv("VOICEVOX_URL", "http://host.docker.internal:50021")
    monkeypatch.setenv("VOICEVOX_SPEAKER_ID", "3")
    monkeypatch.setenv("NARRATION_SENTENCE_PAUSE_SECONDS", "0.8")
    config.reset_settings_cache()
    return TestClient(create_app())


def test_detailed_create_uses_server_voicevox_defaults(client: TestClient) -> None:
    project = client.post("/api/projects", json=INPUT).json()
    assert project["voicevox_url"] == "http://host.docker.internal:50021"
    assert project["voicevox_speaker_id"] == 3
    assert project["narration_sentence_pause_seconds"] == 0.8


def test_explicit_values_still_win_and_speaker_zero_is_kept(client: TestClient) -> None:
    body = {**INPUT, "voicevox_url": "http://192.168.1.20:50021", "voicevox_speaker_id": 0,
            "narration_sentence_pause_seconds": 0.0}
    project = client.post("/api/projects", json=body).json()
    assert project["voicevox_url"] == "http://192.168.1.20:50021"
    assert project["voicevox_speaker_id"] == 0
    assert project["narration_sentence_pause_seconds"] == 0.0


def test_quick_create_uses_server_voicevox_defaults(client: TestClient) -> None:
    response = client.post("/api/projects/quick", json={"source_script": INPUT["source_script"],
                                                        "use_fake_providers": True})
    assert response.status_code == 201, response.text
    project = response.json()["project"]
    assert project["voicevox_url"] == "http://host.docker.internal:50021"
    assert project["voicevox_speaker_id"] == 3


def test_new_project_never_inherits_leftover_byok_keys(client: TestClient) -> None:
    # A failed create could have left keys under the id the next project receives.
    secret_store.set(1, SecretBundle(llm_api_key="sk-leftover", llm_base_url="https://a.example/v1"))
    project = client.post("/api/projects", json=INPUT).json()
    assert project["id"] == 1
    assert secret_store.get(1) is None


def test_out_of_range_server_defaults_fall_back_and_settings_stay_editable(temp_storage: Any, monkeypatch: Any) -> None:
    monkeypatch.setenv("VOICEVOX_SPEED_SCALE", "9.0")
    monkeypatch.setenv("NARRATION_SENTENCE_PAUSE_SECONDS", "12")
    config.reset_settings_cache()
    client = TestClient(create_app())
    project = client.post("/api/projects", json=INPUT).json()
    assert project["voicevox_speed_scale"] == 1.0
    assert project["narration_sentence_pause_seconds"] == 1.5
    saved = client.patch(f"/api/projects/{project['id']}",
                         json={"subtitle_font_size": 50})
    assert saved.status_code == 200, saved.text


def test_quick_create_uses_server_voice_scales(temp_storage: Any, monkeypatch: Any) -> None:
    monkeypatch.setenv("VOICEVOX_PITCH_SCALE", "0.1")
    monkeypatch.setenv("VOICEVOX_VOLUME_SCALE", "1.5")
    config.reset_settings_cache()
    client = TestClient(create_app())
    project = client.post("/api/projects/quick", json={"source_script": INPUT["source_script"],
                                                        "use_fake_providers": True}).json()["project"]
    assert project["voicevox_pitch_scale"] == 0.1
    assert project["voicevox_volume_scale"] == 1.5


def test_defaults_endpoint_reports_the_server_defaults(client: TestClient) -> None:
    defaults = client.get("/api/projects/defaults").json()
    assert defaults["voicevox_speaker_id"] == 3
    assert defaults["narration_sentence_pause_seconds"] == 0.8
    assert defaults["voicevox_speed_scale"] == 1.0
