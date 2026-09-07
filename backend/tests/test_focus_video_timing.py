"""Decode real output to catch cumulative rounding drift between focus frames."""
from __future__ import annotations

import shutil
import subprocess
import wave

import pytest
from PIL import Image

from app.core.config import resolve_ffmpeg
from app.services.ffmpeg_runner import build_block_video_args


def test_encoded_focus_changes_follow_cumulative_sentence_times(tmp_path) -> None:
    ffmpeg = resolve_ffmpeg()
    if not shutil.which(ffmpeg):
        pytest.skip("FFmpeg is required for the encoded timing regression")
    fps, count, interval_ms = 30, 12, 113
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    images = []
    for index, color in enumerate(colors):
        path = tmp_path / f"color-{index}.png"
        Image.new("RGB", (64, 64), color).save(path)
        images.append(path)
    audio = tmp_path / "silence.wav"
    with wave.open(str(audio), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24000)
        wav.writeframes(b"\0\0" * 48000)
    output = tmp_path / "focus.mp4"
    args = build_block_video_args(
        slides=[(images[i % 3], interval_ms) for i in range(count)],
        audio=audio, duration_ms=count * interval_ms, output=output,
        ffmpeg=ffmpeg, width=64, height=64, fps=fps,
    )
    encoded = subprocess.run(args, capture_output=True, timeout=60)
    assert encoded.returncode == 0, encoded.stderr.decode("utf-8", "replace")
    decoded = subprocess.run(
        [ffmpeg, "-v", "error", "-i", str(output), "-an", "-vf", "scale=1:1",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True, timeout=60, check=True,
    ).stdout
    actual = [max(range(3), key=lambda c: decoded[i + c]) for i in range(0, len(decoded), 3)]
    expected = []
    for index in range(count):
        rounding = 999 if index == count - 1 else 500
        boundary = ((index + 1) * interval_ms * fps + rounding) // 1000
        expected.extend([index % 3] * (boundary - len(expected)))
    assert actual == expected
