"""Captions and slides are timed against the voice, not the character count.

Japanese speech duration follows moras, not characters — 「登録」 is two
characters and four moras, ``insert!`` is seven and five. Measured over a
real 28-block project, character-proportional cue timing drifted 890 ms on
average and 6.2 s at worst. These tests pin the machinery that replaces it:
sentence spans measured from VOICEVOX, the offset→time mapping built from
them, and the slide changes that snap onto caption boundaries.
"""
from __future__ import annotations

import pytest

from app.services.narration import (
    SentenceSpan,
    char_time_fn,
    read_spans,
    split_sentences_with_offsets,
    write_spans,
)
from app.services.pipeline import MIN_SLIDE_MS, _snap_to_boundaries
from app.services.subtitles import _hard_split, build_band_cues


def _spans(*items: tuple[str, int, int, int, int]) -> list[SentenceSpan]:
    return [
        SentenceSpan(text=t, char_start=cs, char_end=ce, start_ms=s, end_ms=e)
        for t, cs, ce, s, e in items
    ]


class TestSentenceSplitting:
    def test_pieces_tile_the_text_exactly(self) -> None:
        """Offsets are used to look up times, so nothing may be lost."""
        text = "これはテストです。次の文です！最後は？"
        pieces = split_sentences_with_offsets(text)
        assert "".join(p.text for p in pieces) == text
        assert [(p.char_start, p.char_end) for p in pieces] == [(0, 9), (9, 15), (15, 19)]

    def test_text_without_an_ender_is_one_piece(self) -> None:
        pieces = split_sentences_with_offsets("句点のない文")
        assert len(pieces) == 1
        assert pieces[0].char_end == 6

    def test_trailing_whitespace_folds_into_the_last_sentence(self) -> None:
        """An empty tail would make VOICEVOX return a query with no phrases."""
        pieces = split_sentences_with_offsets("本文です。 ")
        assert len(pieces) == 1
        assert pieces[0].char_end == 6

    def test_empty_text_yields_nothing(self) -> None:
        assert split_sentences_with_offsets("") == []


class TestCharTime:
    def test_sentence_boundaries_are_exact(self) -> None:
        at = char_time_fn(_spans(("あい。", 0, 3, 0, 4000), ("うえ。", 3, 6, 4000, 5000)), total_ms=5000)
        assert at is not None
        assert at(0) == 0
        assert at(3) == 4000
        assert at(6) == 5000

    def test_inside_a_sentence_it_interpolates(self) -> None:
        at = char_time_fn(_spans(("あいうえ。", 0, 5, 0, 5000)), total_ms=5000)
        assert at is not None
        # 4 kana at weight 1 and 。 at 0.3: あ ends 1/4.3 of the way through.
        assert at(1) == pytest.approx(5000 / 4.3, abs=1)

    def test_interpolation_inside_a_sentence_weighs_moras(self) -> None:
        """「登録」 is 2 characters and 4 moras; a character split misplaces it."""
        at = char_time_fn(_spans(("登録をあ。", 0, 5, 0, 10_000)), total_ms=10_000)
        assert at is not None
        by_chars = 10_000 * 2 / 5
        assert at(2) > by_chars + 1_000

    def test_a_kanji_dense_sentence_gets_its_real_share(self) -> None:
        """Two sentences of equal length that take very different times."""
        spans = _spans(("登録と更新。", 0, 6, 0, 9000), ("あのあのあの。", 6, 13, 9000, 12000))
        at = char_time_fn(spans, total_ms=12000)
        assert at is not None
        # Character share would put this boundary at 12000*6/13 = 5538 ms.
        assert at(6) == 9000

    def test_no_usable_spans_returns_none(self) -> None:
        assert char_time_fn([], total_ms=1000) is None
        assert char_time_fn(_spans(("", 0, 0, 0, 0)), total_ms=1000) is None


class TestBandCuesUseMeasuredTiming:
    TEXT = "登録と更新を担当します。それはあのあのあのあのあのあのあのあの。"

    def _cues(self, char_time):
        cues, _font = build_band_cues(
            self.TEXT,
            duration_ms=12000,
            band_height=200,
            base_font_size=48,
            base_max_chars=16,
            char_time=char_time,
        )
        return cues

    def test_without_a_mapping_it_falls_back_to_character_share(self) -> None:
        cues = self._cues(None)
        assert len(cues) == 2
        assert cues[0].end_ms == pytest.approx(12000 * 12 / 32, abs=60)

    def test_with_a_mapping_the_cue_changes_on_the_voice(self) -> None:
        spans = _spans(("登録と更新を担当します。", 0, 12, 0, 9000), ("それは…。", 12, 32, 9000, 12000))
        cues = self._cues(char_time_fn(spans, total_ms=12000))
        assert len(cues) == 2
        assert cues[0].end_ms == 9000
        assert cues[-1].end_ms == 12000

    def test_cues_stay_contiguous_and_ordered(self) -> None:
        spans = _spans(("登録と更新を担当します。", 0, 12, 0, 9000), ("それは…。", 12, 32, 9000, 12000))
        cues = self._cues(char_time_fn(spans, total_ms=12000))
        assert cues[0].start_ms == 0
        for a, b in zip(cues, cues[1:]):
            assert a.end_ms == b.start_ms
            assert a.end_ms > a.start_ms


class TestSlideSnapping:
    def test_changes_move_onto_boundaries(self) -> None:
        cuts = _snap_to_boundaries([10_000, 20_000], [4_000, 9_100, 15_000, 21_300], total_ms=30_000)
        assert cuts == [9_100, 21_300]

    def test_a_boundary_is_used_once(self) -> None:
        cuts = _snap_to_boundaries([10_000, 11_000], [10_500, 14_000], total_ms=30_000)
        assert cuts == [10_500, 14_000]

    def test_cuts_stay_in_order(self) -> None:
        cuts = _snap_to_boundaries([8_000, 16_000, 24_000], [1_000, 30_000], total_ms=32_000)
        assert cuts == sorted(cuts)

    def test_without_boundaries_the_even_split_is_kept(self) -> None:
        assert _snap_to_boundaries([10_000, 20_000], [], total_ms=30_000) == [10_000, 20_000]

    def test_a_boundary_that_would_starve_a_slide_is_refused(self) -> None:
        """Snapping must not create a slide too short to register."""
        cuts = _snap_to_boundaries([10_000], [10_000 + MIN_SLIDE_MS, 29_500], total_ms=30_000)
        assert cuts[0] <= 30_000 - MIN_SLIDE_MS


class TestSpanPersistence:
    def test_round_trip(self, tmp_path) -> None:
        path = tmp_path / "narration.json"
        spans = _spans(("あい。", 0, 3, 0, 4000), ("うえ。", 3, 6, 4000, 5000))
        write_spans(path, spans, duration_ms=5000)
        assert read_spans(path) == spans

    def test_a_missing_file_is_not_an_error(self, tmp_path) -> None:
        """Projects made before this existed have no file and fall back."""
        assert read_spans(tmp_path / "nope.json") == []

    def test_corrupt_json_falls_back_rather_than_raising(self, tmp_path) -> None:
        path = tmp_path / "narration.json"
        path.write_text("{not json", encoding="utf-8")
        assert read_spans(path) == []


class TestWordSafeSplitting:
    """An over-long run has to be cut somewhere; not in the middle of a word."""

    def test_a_cut_steps_back_to_a_hiragana_boundary(self) -> None:
        text = "見出しのシーディーアールを書き換えて新レコードを差し込む構造です。"
        pieces = _hard_split(text, 22)
        assert pieces[0] == "見出しのシーディーアールを書き換えて"
        assert pieces[1] == "新レコードを差し込む構造です。"

    def test_nothing_is_lost_or_reordered(self) -> None:
        text = "見出しのシーディーアールを書き換えて新レコードを差し込む構造です。"
        assert "".join(_hard_split(text, 22)) == text

    def test_every_piece_respects_the_limit(self) -> None:
        text = "あ" * 13 + "登録更新担当" * 6
        assert all(len(p) <= 20 for p in _hard_split(text, 20))

    def test_a_piece_never_starts_with_a_small_kana(self) -> None:
        text = "きっかけをしっかりとちゃんとつかむしゅんかんがひつようです。"
        assert all(p[0] not in "ぁぃぅぇぉっゃゅょー" for p in _hard_split(text, 8))

    def test_text_with_no_hiragana_falls_back_to_a_blunt_cut(self) -> None:
        text = "ABCDEFGHIJKLMNOP"
        assert _hard_split(text, 6) == ["ABCDEF", "GHIJKL", "MNOP"]

    def test_short_text_is_left_alone(self) -> None:
        assert _hard_split("みじかい。", 30) == ["みじかい。"]
