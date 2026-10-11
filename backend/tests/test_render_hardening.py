"""Rendering edge cases: colours, quoted paths, odd sizes, backslashes, slide cuts, caption width."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.core.config import Settings
from app.schemas import ProjectPatch
from app.services import subtitles
from app.services.ffmpeg_runner import _escape_sub_path, write_concat_list
from app.services.pipeline import MIN_SLIDE_MS, _snap_to_boundaries


@pytest.mark.parametrize(("given", "stored"), [
    ("#fff", "#FFFFFF"), ("ff0000", "#FF0000"), ("red", "#FF0000"), ("赤", "#FF0000"), ("黄色", "#FFFF00"),
])
def test_subtitle_colours_are_normalised(given: str, stored: str) -> None:
    assert ProjectPatch(subtitle_text_color=given).subtitle_text_color == stored


def test_unknown_subtitle_colour_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ProjectPatch(subtitle_outline_color="とても赤い")


def test_stored_bad_colours_render_with_fallbacks() -> None:
    assert subtitles._rgb_to_ass("red") == "&H000000FF"
    assert subtitles._rgb_to_ass("nonsense") == "&H00FFFFFF"
    assert subtitles._rgb_to_ass("nonsense", fallback="&H00000000") == "&H00000000"


def test_odd_frame_and_band_sizes_are_rounded_down_to_even() -> None:
    settings = Settings(_env_file=None, output_width=1281, output_height=721, subtitle_band_height=175)
    assert (settings.output_width, settings.output_height, settings.subtitle_band_height) == (1280, 720, 174)


def test_source_backslashes_never_become_ass_overrides(tmp_path: Path) -> None:
    fit = subtitles.fit_text_to_band("改行は\\Nで表します。", band_height=200, base_font_size=48, base_max_chars=36)
    assert "\\N" not in fit.text  # the source \N is neutralised
    assert subtitles.plain_text(fit.text) == "改行は\\Nで表します。"
    cues = subtitles.build_subtitle_cues([(0, 0, 1000, "C:\\new\\Name")], max_chars_per_line=36)
    out = tmp_path / "x.vtt"
    subtitles.render_vtt(cues, out)
    assert "C:\\new\\Name" in out.read_text(encoding="utf-8")


def test_slide_cuts_stay_ordered_when_an_early_cut_snaps_late() -> None:
    cuts = _snap_to_boundaries([3000, 3200, 9000], [5000], total_ms=20000)
    assert cuts == sorted(cuts)
    assert all(b - a >= 1 for a, b in zip([0, *cuts], cuts))
    assert cuts[1] - cuts[0] >= MIN_SLIDE_MS


def test_small_subtitle_sizes_are_honoured() -> None:
    fit = subtitles.fit_text_to_band("短い字幕です。", band_height=200, base_font_size=16, base_max_chars=36)
    assert fit.font_size == 16


def test_large_subtitle_lines_fit_inside_the_frame() -> None:
    text = "キーが一致するレコードが見つかるまでリストの先頭から一つずつ順番に調べていきます。"
    fit = subtitles.fit_text_to_band(text, band_height=400, base_font_size=96, base_max_chars=36)
    widest = max(len(line) for line in fit.lines) * fit.font_size
    assert widest <= 1920 - 2 * subtitles.SIDE_MARGIN


def test_default_caption_layout_is_unchanged() -> None:
    text = "テーブルは、キーを渡すと対応する値を取り出せるデータ構造です。"
    fit = subtitles.fit_text_to_band(text, band_height=200, base_font_size=48, base_max_chars=36)
    assert fit.font_size == 48 and len(fit.lines) == 1


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_paths_with_an_apostrophe_reach_ffmpeg_intact(tmp_path: Path) -> None:
    root = tmp_path / "O'Brien's videos"
    root.mkdir()
    clip = root / "a.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=64x64:d=0.5",
                    "-pix_fmt", "yuv420p", str(clip)], check=True)
    ass = root / "s.ass"
    subtitles.render_ass([subtitles.SubtitleCue(0, 400, "テスト")], ass, width=64, height=64)
    out = root / "sub.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(clip), "-vf", f"subtitles={_escape_sub_path(ass)}",
                    str(out)], check=True)
    listing = root / "list.txt"
    write_concat_list([out, out], listing)
    joined = root / "joined.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(listing),
                    "-c", "copy", str(joined)], check=True)
    assert joined.stat().st_size > 0
