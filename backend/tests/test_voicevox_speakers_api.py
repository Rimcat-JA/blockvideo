"""The speaker list endpoint converts provider speakers into the response schema."""
from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from app.api import routes_health
from app.main import create_app
from app.providers.voicevox import Speaker


class _Client:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url

    async def speakers(self) -> list[Speaker]:
        return [Speaker(speaker_id=0, name="四国めたん",
                        styles=[{"name": "ノーマル", "id": 2, "type": "talk"}, {"name": "あまあま", "id": 0}])]

    async def aclose(self) -> None:
        return None


def test_speakers_endpoint_returns_normalized_speakers(temp_storage: Any, monkeypatch: Any) -> None:
    monkeypatch.setattr(routes_health, "VoicevoxClient", _Client)
    response = TestClient(create_app()).get("/api/voicevox/speakers")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["speakers"] == [{"speaker_id": 0, "name": "四国めたん",
                                 "styles": [{"id": 2, "name": "ノーマル"}, {"id": 0, "name": "あまあま"}]}]
