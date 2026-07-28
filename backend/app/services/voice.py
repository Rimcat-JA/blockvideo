"""Audio generation helpers — wrapping VOICEVOX client with caching."""
from __future__ import annotations

import asyncio
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from app.core.logging import log
from app.core.config import resolve_ffprobe
from app.providers.llm import ProviderError
from app.providers.voicevox import FakeVoicevoxClient, VoicevoxClient, VoicevoxSettings
from app.services.narration import (
    NarrationPlan,
    SentenceSpan,
    build_narration_plan,
    rescale_spans,
)


@dataclass
class AudioResult:
    path: Path
    duration_ms: int
    # When VOICEVOX planned the narration sentence by sentence, where each
    # sentence lands in the audio. Empty when the plan could not be built.
    spans: list[SentenceSpan] = field(default_factory=list)


async def ffprobe_duration_ms(path: Path) -> int:
    ffprobe = resolve_ffprobe()
    if not shutil.which(ffprobe) and not Path(ffprobe).exists():
        raise ProviderError(
            f"ffprobeが見つかりません ({ffprobe})。FFmpegをインストールしてください。",
            safe=True,
        )
    proc = await asyncio.create_subprocess_exec(
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise ProviderError(
            f"ffprobe失敗 (rc={proc.returncode}): {stderr.decode('utf-8', 'replace')[:200]}",
            safe=True,
        )
    payload = json.loads(stdout.decode("utf-8") or "{}")
    duration = float(payload.get("format", {}).get("duration", 0.0))
    return int(round(duration * 1000))


async def _with_retries(make_audio, *, max_attempts: int) -> bytes:
    """Run a synthesis coroutine, retrying transient VOICEVOX failures."""
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return await make_audio()
        except ProviderError as exc:
            last_error = exc
            # Don't retry 404-style permanent errors.
            if "話者ID" in str(exc):
                raise
            log.warning(
                "VOICEVOX合成失敗 attempt={attempt} error={error}",
                attempt=attempt,
                error=exc.__class__.__name__,
            )
            await asyncio.sleep(min(2.0, 0.5 * (2 ** (attempt - 1))))
    if last_error is not None:
        raise last_error
    raise ProviderError("VOICEVOX合成に失敗しました", safe=True)


async def synthesize_audio(
    text: str,
    settings: VoicevoxSettings,
    output_path: Path,
    *,
    client: VoicevoxClient | FakeVoicevoxClient | None = None,
    max_attempts: int = 3,
) -> AudioResult:
    """Synthesize one block of narration. Retries on transient errors."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    close_client = False
    if client is None:
        client = VoicevoxClient(settings.base_url)
        close_client = True
    try:
        async def once() -> bytes:
            audio, _query = await client.synthesize_text(text, settings)
            return audio

        audio = await _with_retries(once, max_attempts=max_attempts)
        output_path.write_bytes(audio)
        duration = await ffprobe_duration_ms(output_path)
        return AudioResult(path=output_path, duration_ms=duration)
    finally:
        if close_client and isinstance(client, VoicevoxClient):
            await client.aclose()


async def synthesize_block(
    text: str,
    settings: VoicevoxSettings,
    output_path: Path,
    *,
    client: VoicevoxClient | FakeVoicevoxClient | None = None,
    sentence_pause_seconds: float = 0.0,
    plan_concurrency: int = 4,
    max_attempts: int = 3,
) -> AudioResult:
    """Synthesize a block, measuring where each sentence lands in the audio.

    Plans the narration sentence by sentence (see ``narration``) so that the
    caption and slide changes can be timed against the real voice instead of
    against character counts, and so the gap at each ``。`` can be widened
    into a breath. Falls back to plain single-shot synthesis — identical to
    the behaviour before this existed — whenever the plan cannot be built.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    close_client = False
    if client is None:
        client = VoicevoxClient(settings.base_url)
        close_client = True
    try:
        plan: NarrationPlan | None = None
        if isinstance(client, VoicevoxClient):
            plan = await build_narration_plan(
                text,
                settings,
                client,
                sentence_pause_seconds=sentence_pause_seconds,
                concurrency=plan_concurrency,
            )
        if plan is None:
            return await synthesize_audio(
                text, settings, output_path, client=client, max_attempts=max_attempts
            )

        async def once() -> bytes:
            return await client.synthesis(plan.query, settings.speaker_id)

        audio = await _with_retries(once, max_attempts=max_attempts)
        output_path.write_bytes(audio)
        duration = await ffprobe_duration_ms(output_path)
        return AudioResult(
            path=output_path,
            duration_ms=duration,
            spans=rescale_spans(
                plan.spans, predicted_ms=plan.predicted_ms, actual_ms=duration
            ),
        )
    finally:
        if close_client and isinstance(client, VoicevoxClient):
            await client.aclose()


def compute_display_duration_ms(
    audio_duration_ms: int,
    *,
    pre_seconds: float = 0.15,
    post_seconds: float = 0.35,
    min_seconds: float = 2.0,
) -> int:
    total = audio_duration_ms + int(pre_seconds * 1000) + int(post_seconds * 1000)
    floor = int(min_seconds * 1000)
    return max(total, floor)


def ffprobe_is_available() -> bool:
    ffprobe = resolve_ffprobe()
    return bool(shutil.which(ffprobe)) or Path(ffprobe).exists()


async def measure_audio_ms(path: Path) -> int:
    return await ffprobe_duration_ms(path)