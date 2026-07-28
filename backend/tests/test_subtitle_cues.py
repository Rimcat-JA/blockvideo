"""A long block's caption advances in steps instead of shrinking to fit.

The slide is stationary for the whole block, but the narration underneath it
can be far longer than the band can hold at a readable size. Splitting the
caption into sequential cues keeps the font large.
"""
from __future__ import annotations

import pytest

from app.services.subtitles import (
    LINE_HEIGHT_RATIO,
    build_band_cues,
    chunk_narration,
)


BAND = 200
FONT = 48
CHARS = 36

SHORT = "短い一文です。"
LONG = (
    "リスト構造の基本を確認します。コンスセルはカーとシーディーアールの二つの部分を持ちます。"
    "リスト、エー、ビー、シー、は三つのセルがシーディーアールで連なります。"
    "最後のセルのシーディーアールは空リストのニルを指して終端します。"
    "ここでセットシーディーアールを使い、最後のセルを先頭へ向けると循環リストになります。"
)


def _cues(text, duration_ms=24_000):
    return build_band_cues(
        text,
        duration_ms=duration_ms,
        band_height=BAND,
        base_font_size=FONT,
        base_max_chars=CHARS,
    )


def test_short_narration_stays_a_single_cue() -> None:
    cues, font = _cues(SHORT, duration_ms=3_000)
    assert len(cues) == 1
    assert font == FONT
    assert cues[0].start_ms == 0 and cues[0].end_ms == 3_000


def test_long_narration_is_split_into_several_cues() -> None:
    cues, _ = _cues(LONG)
    assert len(cues) > 1


def test_splitting_avoids_shrinking_the_font() -> None:
    """The whole point: chunking instead of scaling down."""
    cues, font = _cues(LONG)
    assert font == FONT
    for cue in cues:
        lines = cue.text.split("\\N")
        height = len(lines) * font * LINE_HEIGHT_RATIO
        assert (cue.margin_v or 0) + height <= BAND


def test_cues_are_contiguous_and_cover_the_narration() -> None:
    cues, _ = _cues(LONG)
    assert cues[0].start_ms == 0
    assert cues[-1].end_ms == 24_000
    for prev, nxt in zip(cues, cues[1:]):
        assert prev.end_ms == nxt.start_ms, "captions must not gap or overlap"
        assert prev.end_ms > prev.start_ms


def test_no_narration_text_is_lost() -> None:
    cues, _ = _cues(LONG)
    joined = "".join(c.text.replace("\\N", "") for c in cues)
    assert joined == LONG


def test_chunks_break_on_sentence_boundaries() -> None:
    chunks = chunk_narration(LONG, chars_per_cue=72)
    for chunk in chunks[:-1]:
        assert chunk.rstrip().endswith(("。", "、", "！", "？")), chunk


def test_a_sentence_longer_than_a_cue_is_broken_up() -> None:
    runon = "これは、" + "とても長い説明が、" * 12 + "続きます。"
    chunks = chunk_narration(runon, chars_per_cue=72)
    assert len(chunks) > 1
    assert all(len(c) <= 72 for c in chunks)
    assert "".join(chunks) == runon


def test_text_without_any_punctuation_is_still_bounded() -> None:
    runon = "あ" * 400
    chunks = chunk_narration(runon, chars_per_cue=72)
    assert all(len(c) <= 72 for c in chunks)
    assert "".join(chunks) == runon


def test_cues_are_not_flashed_too_briefly() -> None:
    """Short trailing fragments get merged rather than blinking past."""
    text = "とても長い前半の説明がここに入ります。" * 4 + "はい。"
    cues, _ = _cues(text, duration_ms=20_000)
    durations = [c.end_ms - c.start_ms for c in cues]
    assert min(durations) >= 900, durations


@pytest.mark.parametrize("bad", ["", "   ", None])
def test_empty_narration_produces_no_cues(bad) -> None:
    cues, font = _cues(bad, duration_ms=5_000)
    assert cues == []
    assert font == FONT


def test_zero_duration_is_safe() -> None:
    cues, _ = _cues(LONG, duration_ms=0)
    assert cues == []
