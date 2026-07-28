"""Subtitles must never spill out of the burned-in band onto the slide.

The band is a fixed strip at the bottom of the frame; anything taller than it
overlaps the diagram above, which is the bug these tests pin down.
"""
from __future__ import annotations

import pytest

from app.services.subtitles import LINE_HEIGHT_RATIO, fit_text_to_band


BAND = 200
BASE_FONT = 48
BASE_CHARS = 36


def _height(fit) -> float:
    return len(fit.lines) * fit.font_size * LINE_HEIGHT_RATIO


def _fits(fit, band: int = BAND) -> bool:
    """Text block, offset by its bottom margin, stays inside the band."""
    return fit.margin_v + _height(fit) <= band


SHORT = "リスト構造の基本です。"
TWO_LINES = (
    "リスト a、b、c は三つのセルがシーディーアールで連なり、"
    "最後のセルのシーディーアールは空リストのニルを指します。"
)
LONG = "このように、矢印をたどると以前のセルへ戻ってくる構造を循環構造と呼びます。" * 3
VERY_LONG = "このように、矢印をたどると以前のセルへ戻ってくる構造を循環構造と呼びます。" * 8


@pytest.mark.parametrize("text", [SHORT, TWO_LINES, LONG, VERY_LONG])
def test_text_always_fits_inside_the_band(text) -> None:
    fit = fit_text_to_band(
        text, band_height=BAND, base_font_size=BASE_FONT, base_max_chars=BASE_CHARS
    )
    assert _fits(fit), (
        f"{len(fit.lines)} lines at {fit.font_size}px "
        f"= {_height(fit):.0f}px + margin {fit.margin_v} exceeds band {BAND}"
    )


def test_short_text_keeps_the_configured_font_size() -> None:
    fit = fit_text_to_band(
        SHORT, band_height=BAND, base_font_size=BASE_FONT, base_max_chars=BASE_CHARS
    )
    assert fit.font_size == BASE_FONT
    assert len(fit.lines) == 1


def test_long_text_shrinks_rather_than_overflowing() -> None:
    fit = fit_text_to_band(
        VERY_LONG, band_height=BAND, base_font_size=BASE_FONT, base_max_chars=BASE_CHARS
    )
    assert fit.font_size < BASE_FONT
    assert _fits(fit)


def test_no_narration_is_dropped() -> None:
    """Fitting must re-wrap, never truncate."""
    fit = fit_text_to_band(
        VERY_LONG, band_height=BAND, base_font_size=BASE_FONT, base_max_chars=BASE_CHARS
    )
    joined = "".join(fit.lines)
    assert len(joined) == len(VERY_LONG.strip())


def test_margin_centres_the_block_in_the_band() -> None:
    fit = fit_text_to_band(
        SHORT, band_height=BAND, base_font_size=BASE_FONT, base_max_chars=BASE_CHARS
    )
    above = BAND - fit.margin_v - _height(fit)
    assert abs(above - fit.margin_v) <= 2


@pytest.mark.parametrize("band", [120, 160, 200, 260, 320])
def test_fits_across_band_heights(band) -> None:
    fit = fit_text_to_band(
        LONG, band_height=band, base_font_size=BASE_FONT, base_max_chars=BASE_CHARS
    )
    assert _fits(fit, band)


def test_taller_band_allows_a_larger_font() -> None:
    small = fit_text_to_band(
        LONG, band_height=160, base_font_size=BASE_FONT, base_max_chars=BASE_CHARS
    )
    large = fit_text_to_band(
        LONG, band_height=320, base_font_size=BASE_FONT, base_max_chars=BASE_CHARS
    )
    assert large.font_size >= small.font_size


def test_empty_text_is_handled() -> None:
    fit = fit_text_to_band(
        "", band_height=BAND, base_font_size=BASE_FONT, base_max_chars=BASE_CHARS
    )
    assert _fits(fit)
    assert fit.text == ""


def test_real_worst_case_block_texts_fit() -> None:
    """A cue built from raw markdown (the deterministic-split fallback path)
    is the longest thing the band ever has to hold."""
    raw = (
        "ところが、`set-cdr!`で最後のセルを以前のセルへつなぐと、```text 先頭 → 次 → 次 ↑ "
        "└────────┘ ``` という輪ができます。このように、矢印をたどると以前のセルへ"
        "戻ってくる構造を**循環構造**と呼びます。13."
    )
    fit = fit_text_to_band(
        raw, band_height=BAND, base_font_size=BASE_FONT, base_max_chars=BASE_CHARS
    )
    assert _fits(fit)
