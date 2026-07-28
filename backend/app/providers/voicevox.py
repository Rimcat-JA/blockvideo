"""VOICEVOX Engine HTTP client and deterministic offline substitute.

Talks to a running VOICEVOX Engine over HTTP. The engine is a separate
service that the user is expected to launch themselves; we never bake
secrets into requests because the engine does not need them.
Imports:
    ``dataclass`` defines synthesis settings and normalized speaker records.
    ``Any`` describes VOICEVOX's JSON query payloads.
    ``httpx`` performs asynchronous requests to a running engine.
    ``ProviderError`` converts transport/protocol failures into safe errors.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import httpx

from app.providers.llm import ProviderError


@dataclass
class VoicevoxSettings:
    """VOICEVOX synthesis parameters copied from project settings.

    Attributes:
        base_url: Running engine root URL.
        speaker_id: VOICEVOX speaker/style identifier.
        speed_scale, pitch_scale, intonation_scale, volume_scale: Engine
            prosody controls forwarded into the query.
        pre_phoneme_length, post_phoneme_length: Silence around synthesized
            audio, in seconds.

    """

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
    """Normalized speaker metadata returned to API clients.

    Attributes:
        speaker_id: Numeric VOICEVOX speaker identifier.
        name: Human-readable speaker name.
        styles: Raw style dictionaries supplied by the engine.

    """

    speaker_id: int
    name: str
    styles: list[dict[str, Any]] = field(default_factory=list)


class VoicevoxClient:
    """Async HTTP client for a running VOICEVOX Engine instance.

    Attributes:
        base_url: Normalized engine URL without a trailing slash.
        _client: Owned ``httpx.AsyncClient`` connection pool.

    """

    def __init__(self, base_url: str, *, timeout: float = 30.0) -> None:
        """Create a client pointed at one VOICEVOX base URL.

        Args:
            base_url: Engine URL; trailing slashes are removed.
            timeout: Per-request HTTP timeout in seconds.

        """
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=timeout)

    async def aclose(self) -> None:
        """Close the HTTP connection pool owned by this client."""
        await self._client.aclose()

    async def health(self) -> bool:
        """Return whether the engine answers ``GET /version`` with 200.

        Returns:
            ``False`` for network errors or non-200 status codes.

        """
        try:
            r = await self._client.get(f"{self.base_url}/version")
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    async def speakers(self) -> list[Speaker]:
        """Fetch and normalize all available speakers and styles.

        Returns:
            A list of normalized ``Speaker`` objects.

        Raises:
            ProviderError: When the engine cannot be reached or returns an
                HTTP error status.

        """
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
        """Ask VOICEVOX for a synthesis query for one text fragment.

        Args:
            text: Narration fragment submitted as the ``text`` query parameter.
            speaker_id: Speaker/style identifier.

        Returns:
            Raw VOICEVOX audio-query JSON mapping.

        Raises:
            ProviderError: For transport errors, unknown speakers, or other
                HTTP error responses.

        """
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
        """Synthesize a prepared query and return WAV response bytes.

        Args:
            query: VOICEVOX audio-query mapping, optionally tuned by
                ``apply_settings``.
            speaker_id: Speaker/style identifier accepted by the engine.

        Returns:
            Raw WAV bytes from ``POST /synthesis``.

        Raises:
            ProviderError: For transport errors, unknown speakers, or HTTP
                failures.

        """
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
        """Apply project tuning values in place and return ``query``.

        Args:
            query: Mutable VOICEVOX query mapping.
            settings: User/project synthesis controls.

        Returns:
            The same mapping after all supported controls are overwritten.

        Side Effects:
            Mutates the supplied dictionary; it does not copy the query.

        """
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
        """Build, tune, and synthesize one complete text fragment.

        Returns:
            ``(wav_bytes, tuned_query)`` so callers can persist audio and,
            when needed, inspect the exact query used by the engine.

        """
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

    Attributes:
        speaker_id: Default fake speaker ID used by the fixture.
        calls: ``(text, speaker_id)`` records for test assertions.

    """

    def __init__(self) -> None:
        """Create an offline client with a call log and default speaker."""
        self.speaker_id = 1
        self.calls: list[tuple[str, int]] = []

    async def health(self) -> bool:
        """Report healthy because no external engine is required."""
        return True

    async def speakers(self) -> list[Speaker]:
        """Return the deterministic fake speaker catalog."""
        return [
            Speaker(speaker_id=1, name="Fake四国めたん", styles=[{"id": 1, "name": "ノーマル"}]),
            Speaker(speaker_id=2, name="Fake春日部つむぎ", styles=[{"id": 2, "name": "ノーマル"}]),
        ]

    async def synthesize_text(self, text: str, settings: VoicevoxSettings) -> tuple[bytes, dict]:
        """Return deterministic low-amplitude WAV bytes sized by text length.

        Args:
            text: Text whose character count controls fake duration.
            settings: Settings object; only ``speaker_id`` is recorded by the
                fake implementation.

        Returns:
            ``(wav_bytes, metadata)`` with a deterministic mono 24 kHz WAV.

        """
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
