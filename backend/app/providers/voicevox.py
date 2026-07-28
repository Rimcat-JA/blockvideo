"""VOICEVOX Engine client.

Talks to a running VOICEVOX Engine over HTTP. The engine is a separate
service that the user is expected to launch themselves; we never bake
secrets into requests because the engine does not need them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import httpx

from app.providers.llm import ProviderError


@dataclass
class VoicevoxSettings:
    base_url: str = "http://127.0.0.1:50021"
    speaker_id: int = 1
    speed_scale: float = 1.0
    pitch_scale: float = 0.0
    intonation_scale: float = 1.0
    volume_scale: float = 1.0
    pre_phoneme_length: float = 0.1
    post_phoneme_length: float = 0.1


@dataclass
class Speaker:
    speaker_id: int
    name: str
    styles: list[dict[str, Any]] = field(default_factory=list)


class VoicevoxClient:
    def __init__(self, base_url: str, *, timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=timeout)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def health(self) -> bool:
        try:
            r = await self._client.get(f"{self.base_url}/version")
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    async def speakers(self) -> list[Speaker]:
        try:
            r = await self._client.get(f"{self.base_url}/speakers")
        except httpx.HTTPError as exc:
            raise ProviderError(
                f"VOICEVOX Engineに接続できません ({self.base_url})",
                safe=True,
                original=exc,
            ) from exc
        if r.status_code >= 400:
            raise ProviderError(
                f"VOICEVOX話者一覧取得に失敗しました (status {r.status_code})",
                safe=True,
            )
        data = r.json()
        return [
            Speaker(
                speaker_id=int(item.get("speaker_id", 0)),
                name=str(item.get("name", "")),
                styles=list(item.get("styles") or []),
            )
            for item in data
        ]

    async def audio_query(self, text: str, speaker_id: int) -> dict[str, Any]:
        url = f"{self.base_url}/audio_query"
        params = {"text": text, "speaker": speaker_id}
        try:
            r = await self._client.post(url, params=params)
        except httpx.HTTPError as exc:
            raise ProviderError(
                "VOICEVOX audio_queryに失敗しました (Engine起動を確認してください)",
                safe=True,
                original=exc,
            ) from exc
        if r.status_code == 404:
            raise ProviderError(
                f"VOICEVOX話者ID {speaker_id} が見つかりません",
                safe=True,
            )
        if r.status_code >= 400:
            raise ProviderError(
                f"VOICEVOX audio_query失敗 (status {r.status_code})",
                safe=True,
            )
        return r.json()

    async def synthesis(self, query: dict[str, Any], speaker_id: int) -> bytes:
        url = f"{self.base_url}/synthesis"
        params = {"speaker": speaker_id}
        try:
            r = await self._client.post(
                url, params=params, json=query, headers={"Content-Type": "application/json"}
            )
        except httpx.HTTPError as exc:
            raise ProviderError(
                "VOICEVOX synthesisに失敗しました",
                safe=True,
                original=exc,
            ) from exc
        if r.status_code == 404:
            raise ProviderError(
                f"VOICEVOX話者ID {speaker_id} が見つかりません",
                safe=True,
            )
        if r.status_code >= 400:
            raise ProviderError(
                f"VOICEVOX synthesis失敗 (status {r.status_code})",
                safe=True,
            )
        return r.content

    def apply_settings(self, query: dict[str, Any], settings: VoicevoxSettings) -> dict[str, Any]:
        """Mutate a query dict with the user-configured tuning."""
        query.setdefault("speedScale", settings.speed_scale)
        query["speedScale"] = settings.speed_scale
        query["pitchScale"] = settings.pitch_scale
        query["intonationScale"] = settings.intonation_scale
        query["volumeScale"] = settings.volume_scale
        query["prePhonemeLength"] = settings.pre_phoneme_length
        query["postPhonemeLength"] = settings.post_phoneme_length
        return query

    async def synthesize_text(
        self, text: str, settings: VoicevoxSettings
    ) -> tuple[bytes, dict[str, Any]]:
        query = await self.audio_query(text, settings.speaker_id)
        self.apply_settings(query, settings)
        audio = await self.synthesis(query, settings.speaker_id)
        return audio, query


class FakeVoicevoxClient:
    """In-memory VOICEVOX replacement used by tests and demo mode.

    Generates a deterministic silent-ish WAV whose length scales with the
    number of characters in the input (≈50 ms per char, clamped). This is
    enough to exercise the rest of the pipeline without needing a live
    engine.
    """

    def __init__(self) -> None:
        self.speaker_id = 1
        self.calls: list[tuple[str, int]] = []

    async def health(self) -> bool:
        return True

    async def speakers(self) -> list[Speaker]:
        return [
            Speaker(speaker_id=1, name="Fake四国めたん", styles=[{"id": 1, "name": "ノーマル"}]),
            Speaker(speaker_id=2, name="Fake春日部つむぎ", styles=[{"id": 2, "name": "ノーマル"}]),
        ]

    async def synthesize_text(self, text: str, settings: VoicevoxSettings) -> tuple[bytes, dict]:
        self.calls.append((text, settings.speaker_id))
        import math
        import struct
        import wave

        chars = max(1, len(text))
        # 50ms per char, clamped between 0.5s and 8s
        seconds = max(0.5, min(8.0, chars * 0.05))
        sample_rate = 24000
        n_samples = int(seconds * sample_rate)
        # 16-bit mono PCM, very low amplitude noise so it's not silent.
        amplitude = 600
        period = max(1, int(sample_rate / 220.0))
        buf = bytearray()
        for i in range(n_samples):
            sample = int(amplitude * math.sin(2 * math.pi * (i % period) / period))
            buf.extend(struct.pack("<h", sample))
        import io
        bio = io.BytesIO()
        with wave.open(bio, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sample_rate)
            w.writeframes(bytes(buf))
        return bio.getvalue(), {"fake": True, "text": text}