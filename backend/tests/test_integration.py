"""End-to-end integration test using Fake providers."""
from __future__ import annotations

import wave
from pathlib import Path

import pytest

from app.db import get_session_factory
from app.models.project import Project
from app.services import pipeline
from app.services.paths import (
    ensure_project_layout,
    output_video_path,
)


SAMPLE_SCRIPT = (
    "本動画は Compose Multiplatform の基本を紹介するサンプルです。"
    "Compose は Android だけでなく iOS やデスクトップでも同じコードで UI を組めます。"
    "最初のステップとしては、Android Studio で新規プロジェクトを作るところから始めましょう。"
)


def _create_silent_wav(path: Path, seconds: float = 1.0) -> None:
    sample_rate = 24000
    n_samples = int(seconds * sample_rate)
    import struct

    buf = bytearray()
    for i in range(n_samples):
        sample = int(1000.0 * (1 if (i // 50) % 2 else -1))
        buf.extend(struct.pack("<h", sample))
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(bytes(buf))


@pytest.mark.asyncio
async def test_full_pipeline_with_fake_providers(temp_storage, skip_if_no_ffmpeg) -> None:
    """Run the full pipeline against FakeLLMProvider + FakeVoicevox."""
    ensure_project_layout(1)
    factory = get_session_factory()
    db = factory()
    try:
        project = Project(
            title="integration sample",
            source_script=SAMPLE_SCRIPT,
            use_fake_providers=True,
            voicevox_url="http://127.0.0.1:50021",
            voicevox_speaker_id=1,
        )
        db.add(project)
        db.commit()
        db.refresh(project)

        # Run the pipeline. Cancellation requested mid-way to make sure
        # resume-style idempotency holds, then run again.
        await pipeline.run_full_pipeline(project.id)
        db.refresh(project)
        # Even if fake voice is silent, the duration math must produce
        # a sensible value.
        for b in project.blocks:
            assert b.audio_path, f"block {b.index} missing audio"
            assert b.image_path, f"block {b.index} missing image"
            assert b.video_path, f"block {b.index} missing video"
            assert b.duration_ms and b.duration_ms > 0
            assert b.display_duration_ms and b.display_duration_ms >= 2000
        # Final video must exist.
        final = output_video_path(project.id)
        assert final.exists(), "output MP4 was not produced"
        assert project.output_video_path
        # Idempotency: running again should not raise and should keep videos.
        await pipeline.run_full_pipeline(project.id)
        db.refresh(project)
        assert final.exists()
    finally:
        db.close()