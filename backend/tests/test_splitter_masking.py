"""Unit tests for code masking, boundary realignment and block merging.

These cover the three behaviours that make the splitter robust on scripts
containing fenced code and ASCII art. No network access.
"""
from __future__ import annotations

import re

from app.services.splitter import (
    mask_code_blocks,
    merge_small_blocks,
    realign_to_source,
    segment_script,
    unmask_code_blocks,
)
from app.services.stage_schemas import SplitBlock


SCRIPT_WITH_CODE = (
    "通常のリストでは、cdrをたどると空リストに到達します。\n"
    "```scheme\n(define (make-cycle x)\n  (set-cdr! (last-pair x) x)\n  x)\n```\n"
    "ところが、set-cdr! を使うと輪ができます。\n"
    "```text\n先頭 → 次 → 次\n ↑        │\n └────────┘\n```\n"
    "これを循環構造と呼びます。\n"
)


def _squash(text: str) -> str:
    return re.sub(r"\s+", "", text)


def test_mask_roundtrip_is_lossless() -> None:
    masked, blocks = mask_code_blocks(SCRIPT_WITH_CODE)
    assert len(blocks) == 2
    assert "```" not in masked, "fenced code must not survive masking"
    assert "〔コード1〕" in masked and "〔コード2〕" in masked
    assert unmask_code_blocks(masked, blocks) == SCRIPT_WITH_CODE


def test_mask_hides_ascii_art_from_the_model() -> None:
    masked, _ = mask_code_blocks(SCRIPT_WITH_CODE)
    # The box-drawing characters are what break JSON string escaping.
    for glyph in ("│", "→", "└", "─"):
        assert glyph not in masked


def test_unmask_drops_unknown_tokens() -> None:
    assert unmask_code_blocks("前〔コード9〕後", []) == "前後"


def test_segment_never_splits_inside_a_fence() -> None:
    masked, blocks = mask_code_blocks(SCRIPT_WITH_CODE)
    segments = segment_script(masked, max_chars=40)
    assert len(segments) > 1
    for seg in segments:
        # A fence would have to appear as a partial token to be split.
        assert "```" not in seg
    assert _squash("".join(segments)) == _squash(masked)


def test_realign_recovers_from_whitespace_drift() -> None:
    """The model dropping a space must not fail the split."""
    original = "ここで set-cdr! を使い、輪を作ります。そして終わります。"
    # Model echoes it back without the space around the ASCII token.
    blocks = [
        SplitBlock(index=0, source_text="ここでset-cdr!を使い、輪を作ります。", tts_text="a"),
        SplitBlock(index=1, source_text="そして終わります。", tts_text="b"),
    ]
    realigned = realign_to_source(original, blocks)
    assert realigned is not None
    # source_text is taken from the original, so the space is back.
    assert realigned[0].source_text == "ここで set-cdr! を使い、輪を作ります。"
    assert _squash("".join(b.source_text for b in realigned)) == _squash(original)


def test_realign_rejects_dropped_content() -> None:
    original = "一つ目の文です。二つ目の文です。三つ目の文です。"
    blocks = [SplitBlock(index=0, source_text="一つ目の文です。", tts_text="a")]
    assert realign_to_source(original, blocks) is None


def test_realign_rejects_invented_content() -> None:
    original = "一つ目の文です。"
    blocks = [
        SplitBlock(index=0, source_text="一つ目の文です。", tts_text="a"),
        SplitBlock(index=1, source_text="存在しない文です。", tts_text="b"),
    ]
    assert realign_to_source(original, blocks) is None


def test_merge_folds_short_blocks_without_losing_text() -> None:
    blocks = [
        SplitBlock(index=0, source_text="あ" * 10, tts_text="あ" * 10),
        SplitBlock(index=1, source_text="い" * 10, tts_text="い" * 10),
        SplitBlock(index=2, source_text="う" * 90, tts_text="う" * 90),
    ]
    merged = merge_small_blocks(blocks, min_chars=60, max_chars=130)
    assert len(merged) < len(blocks)
    assert [b.index for b in merged] == list(range(len(merged)))
    assert _squash("".join(b.source_text for b in merged)) == _squash(
        "".join(b.source_text for b in blocks)
    )


def test_merge_keeps_already_long_blocks_separate() -> None:
    blocks = [
        SplitBlock(index=0, source_text="あ" * 100, tts_text="あ" * 100),
        SplitBlock(index=1, source_text="い" * 100, tts_text="い" * 100),
    ]
    merged = merge_small_blocks(blocks, min_chars=60, max_chars=130)
    assert len(merged) == 2


def test_merge_measures_narration_not_slide_text() -> None:
    """A block whose source is a big code listing but whose narration is
    short should still merge — duration follows narration, not slide size."""
    blocks = [
        SplitBlock(index=0, source_text="解説文。", tts_text="解説文。"),
        SplitBlock(index=1, source_text="```\n" + "x" * 500 + "\n```", tts_text=""),
    ]
    merged = merge_small_blocks(blocks, min_chars=60, max_chars=130)
    assert len(merged) == 1
    assert "x" * 500 in merged[0].source_text


def test_merge_handles_empty_input() -> None:
    assert merge_small_blocks([], min_chars=60, max_chars=130) == []
