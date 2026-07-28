"""Tests for utility calculations, FFmpeg argv builder, and cache logic."""
from __future__ import annotations

from pathlib import Path

import pytest

from app.services.hashing import hash_value
from app.services.voice import compute_display_duration_ms, ffprobe_duration_ms
from app.services.subtitles import (
    build_subtitle_cues,
    ms_to_ass_time,
    render_ass,
)
from app.services.ffmpeg_runner import build_block_video_args, build_concat_args
from app.services.mermaid_renderer import mmdc_available


def test_display_duration_includes_margins_and_floor() -> None:
    # 1s audio + pre + post should be > audio alone
    d = compute_display_duration_ms(1000, pre_seconds=0.15, post_seconds=0.35, min_seconds=2.0)
    assert d == 2000  # floor wins
    d2 = compute_display_duration_ms(3000)
    assert d2 >= 3500


def test_ms_to_ass_time_format() -> None:
    assert ms_to_ass_time(0) == "0:00:00.00"
    assert ms_to_ass_time(61_500) == "0:01:01.50"
    assert ms_to_ass_time(3_661_500) == "1:01:01.50"


def test_block_args_is_list_no_shell(tmp_path: Path) -> None:
    image = tmp_path / "in.png"
    audio = tmp_path / "in.wav"
    output = tmp_path / "out.mp4"
    args = build_block_video_args(
        image=image, audio=audio, duration_ms=2000, output=output, ffmpeg="ffmpeg"
    )
    assert args[0] == "ffmpeg"
    assert all(isinstance(a, str) for a in args)
    # scaled vf present, full-frame layout (no subtitle)
    vf = args[args.index("-vf") + 1]
    assert "pad=1920:1080" in vf
    assert "subtitles=" not in vf


def test_block_args_with_subtitle_uses_band_layout(tmp_path: Path) -> None:
    image = tmp_path / "in.png"
    audio = tmp_path / "in.wav"
    sub = tmp_path / "sub.ass"
    sub.write_text("[Script Info]\n", encoding="utf-8")
    output = tmp_path / "out.mp4"
    args = build_block_video_args(
        image=image,
        audio=audio,
        duration_ms=2000,
        output=output,
        ffmpeg="ffmpeg",
        subtitle_path=sub,
        subtitle_band_height=200,
    )
    vf = args[args.index("-vf") + 1]
    # slide region = 1080 - 200 = 880
    assert "scale=1920:880" in vf
    assert "pad=1920:880" in vf
    assert "pad=1920:1080:0:0:black" in vf
    assert "subtitles=" in vf
    # argv is still all strings (no shell)
    assert all(isinstance(a, str) for a in args)


def test_block_args_without_subtitle_full_frame(tmp_path: Path) -> None:
    image = tmp_path / "in.png"
    audio = tmp_path / "in.wav"
    output = tmp_path / "out.mp4"
    args = build_block_video_args(
        image=image,
        audio=audio,
        duration_ms=2000,
        output=output,
        ffmpeg="ffmpeg",
        subtitle_path=None,
        subtitle_band_height=200,
    )
    vf = args[args.index("-vf") + 1]
    assert "pad=1920:1080" in vf
    assert "scale=1920:880" not in vf
    assert "subtitles=" not in vf


def test_concat_args_default_uses_concat_demuxer(tmp_path: Path) -> None:
    list_file = tmp_path / "list.txt"
    out = tmp_path / "out.mp4"
    args = build_concat_args(list_file=list_file, output=out, ffmpeg="ffmpeg", crossfade_seconds=0.0)
    assert "-f" in args
    assert "concat" in args


def test_subtitle_wrap_japanese(tmp_path: Path) -> None:
    cues = build_subtitle_cues(
        [(0, 0, 2000, "これはテスト用の長い日本語の文章です。途中で折り返さなければなりません。")],
        max_chars_per_line=18,
    )
    assert cues and len(cues[0].text.split("\\N")) >= 2
    target = tmp_path / "subs.ass"
    n = render_ass(cues, target)
    assert n == 1
    text = target.read_text(encoding="utf-8")
    assert "[Script Info]" in text
    assert "Dialogue:" in text


def test_block_hash_distinguishes_tts_changes() -> None:
    a = hash_value("block-1", "短めのソースです。", "読み上げテキスト", "code_slide", "大学風")
    b = hash_value("block-1", "短めのソースです。", "別の読み上げテキスト", "code_slide", "大学風")
    assert a != b


@pytest.mark.asyncio
async def test_ffprobe_duration_ms_missing_file(tmp_path: Path) -> None:
    from app.providers.llm import ProviderError

    with pytest.raises(ProviderError):
        await ffprobe_duration_ms(tmp_path / "nope.wav")


@pytest.mark.asyncio
async def test_mermaid_render_skips_when_mmdc_absent(tmp_path: Path) -> None:
    """When mmdc isn't installed, the renderer raises RuntimeError (so the
    caller falls back to PIL) rather than silently succeeding."""
    if mmdc_available():
        pytest.skip("mmdc present; fallback path not exercised here")
    from app.services.mermaid_renderer import render_mermaid_to_png

    out = tmp_path / "diagram.png"
    with pytest.raises(RuntimeError):
        await render_mermaid_to_png(
            "flowchart TD\n A --> B",
            out,
            width=1920,
            height=1080,
        )


def test_block_clip_runs_for_the_full_display_duration() -> None:
    """The trailing hold between blocks must survive into the encoded clip.

    ``-shortest`` truncates at the shortest input (the narration audio), which
    silently discarded the gap; the audio has to be padded with silence
    instead so ``-t`` governs the clip length.
    """
    from pathlib import Path

    from app.services.ffmpeg_runner import build_block_video_args

    args = build_block_video_args(
        image=Path("i.png"),
        audio=Path("a.wav"),
        duration_ms=13_900,
        output=Path("o.mp4"),
        ffmpeg="ffmpeg",
    )
    assert "-shortest" not in args, "-shortest would cut the trailing hold"
    assert "apad" in args, "audio must be padded with silence to reach -t"
    assert args[args.index("-t") + 1] == "13.900"
