"""Real-shaped VOICEVOX /speakers responses and non-VOICEVOX answers."""
from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from app.api import routes_health
from app.main import create_app
from app.providers.llm import ProviderError
from app.providers.voicevox import VoicevoxClient

_REAL_SPEAKERS = [
    {"name": "四国めたん", "speaker_uuid": "7ffcb7ce", "version": "0.14",
     "styles": [{"name": "ノーマル", "id": 2, "type": "talk"}, {"name": "あまあま", "id": 0, "type": "talk"}]},
    {"name": "ずんだもん", "speaker_uuid": "388f246b", "version": "0.14",
     "styles": [{"name": "ノーマル", "id": 3, "type": "talk"}]},
]


def _client(handler: Any) -> VoicevoxClient:
    client = VoicevoxClient("http://voicevox.test")
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return client


def _speakers(handler: Any) -> list[Any]:
    async def run() -> list[Any]:
        client = _client(handler)
        try:
            return await client.speakers()
        finally:
            await client.aclose()
    return asyncio.run(run())


def test_speaker_id_is_the_first_talk_style() -> None:
    speakers = _speakers(lambda request: httpx.Response(200, json=_REAL_SPEAKERS))
    assert [(s.name, s.speaker_id) for s in speakers] == [("四国めたん", 2), ("ずんだもん", 3)]


@pytest.mark.parametrize("response", [
    httpx.Response(200, text="<!doctype html><html></html>", headers={"content-type": "text/html"}),
    httpx.Response(200, json={"error": "not voicevox"}),
])
def test_non_voicevox_answer_is_a_provider_error(response: httpx.Response) -> None:
    with pytest.raises(ProviderError):
        _speakers(lambda request: response)


def test_speakers_route_answers_503_for_a_non_voicevox_url(temp_storage: Any, monkeypatch: Any) -> None:
    def make(base_url: str) -> VoicevoxClient:
        return _client(lambda request: httpx.Response(200, text="<html></html>"))
    monkeypatch.setattr(routes_health, "VoicevoxClient", make)
    response = TestClient(create_app()).get("/api/voicevox/speakers", params={"url": "http://127.0.0.1:5173"})
    assert response.status_code == 503, response.text
