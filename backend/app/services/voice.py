"""Audio generation helpers wrapping VOICEVOX with retries and timing files.

Imports:
    ``asyncio`` runs FFprobe and retry sleeps without blocking the event loop.
    ``json`` parses FFprobe output; ``shutil`` checks executable availability.
    Dataclasses/paths represent audio results and destinations.
    Provider/narration modules provide the client, error, and sentence-timing
    boundaries.
"""
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
    """Audio artifact path, measured duration, and optional sentence spans.

    Attributes:
        path: Written WAV artifact.
        duration_ms: FFprobe-measured audio duration.
        spans: Sentence timings when per-sentence planning succeeded; empty for
            fallback single-shot/fake synthesis.

    """

    path: Path
    duration_ms: int
    # When VOICEVOX planned the narration sentence by sentence, where each
    # sentence lands in the audio. Empty when the plan could not be built.
    spans: list[SentenceSpan] = field(default_factory=list)


async def ffprobe_duration_ms(path: Path) -> int:
    """Run FFprobe and return an audio file's duration in milliseconds.

    Args:
        path: Audio file to inspect.

    Returns:
        Rounded ``format.duration`` in milliseconds.

    Raises:
        ProviderError: If FFprobe is unavailable or exits non-zero.

    """
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
    """Run a synthesis coroutine with bounded exponential backoff.

    Args:
        make_audio: Zero-argument awaitable factory that performs one request.
        max_attempts: Number of attempts, including the first try.

    Returns:
        The first successful WAV byte sequence.

    Raises:
        ProviderError: Re-raises permanent speaker-ID errors or the last
            transient provider error after all attempts.

    """
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
    """Synthesize one block with simple whole-text fallback behavior.

    Args:
        text: Narration text sent to VOICEVOX.
        settings: Voice/speaker controls.
        output_path: WAV destination.
        client: Optional caller-owned live or fake client.
        max_attempts: Maximum transient synthesis attempts.

    Returns:
        ``AudioResult`` with the written path and FFprobe duration; spans are
        empty because this path does not build a sentence plan.

    Side Effects:
        Creates parent directories, writes a WAV, and closes an internally
        created live client.

    """
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

    Args:
        text: Narration text for one block.
        settings: Voice/speaker controls.
        output_path: WAV destination.
        client: Optional live/fake client; fake clients use simple synthesis.
        sentence_pause_seconds: Extra pause between measured sentences.
        plan_concurrency: Maximum concurrent VOICEVOX query requests.
        max_attempts: Maximum transient synthesis attempts.

    Returns:
        ``AudioResult`` with measured duration and rescaled spans when planning
        succeeds; otherwise the same result shape from ``synthesize_audio``.

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
    """Add reading margins to audio duration while enforcing a floor.

    Args:
        audio_duration_ms: Measured narration duration.
        pre_seconds: Lead-in hold before/around narration.
        post_seconds: Trailing hold after narration.
        min_seconds: Minimum total display duration.

    Returns:
        Maximum of the margin-adjusted duration and the configured minimum,
        expressed in milliseconds.

    """
    total = audio_duration_ms + int(pre_seconds * 1000) + int(post_seconds * 1000)
    floor = int(min_seconds * 1000)
    return max(total, floor)


def ffprobe_is_available() -> bool:
    """Return whether configured FFprobe is discoverable."""
    ffprobe = resolve_ffprobe()
    return bool(shutil.which(ffprobe)) or Path(ffprobe).exists()


async def measure_audio_ms(path: Path) -> int:
    """Measure one audio file through the shared FFprobe helper.

    This alias keeps callers independent of the implementation name used by
    the synthesis module.
    """
    return await ffprobe_duration_ms(path)
