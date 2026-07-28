"""Code, ASCII diagrams and Markdown belong on the slide, never in the audio.

VOICEVOX pronounces whatever it is given, so anything left in ``tts_text``
is read aloud one symbol at a time — and, since the burned-in subtitle is
generated from the narration, it also bloats the subtitle.
"""
from __future__ import annotations

import pytest

from app.services.splitter import sanitize_for_narration


def test_fenced_code_is_removed_but_surrounding_prose_survives() -> None:
    text = (
        "実装はこうです。\n"
        "```scheme\n(define (make-cycle x) (set-cdr! (last-pair x) x) x)\n```\n"
        "これで循環します。"
    )
    out = sanitize_for_narration(text)
    assert "define" not in out
    assert "```" not in out
    assert "実装はこうです。" in out
    assert "これで循環します。" in out


def test_ascii_diagram_lines_are_dropped() -> None:
    text = "先頭 → 次 → 次\n ↑        │\n └────────┘\nという輪ができます。"
    out = sanitize_for_narration(text)
    for glyph in ("│", "└", "─", "↑", "→"):
        assert glyph not in out
    assert "という輪ができます。" in out


@pytest.mark.parametrize(
    "text,expected_fragment",
    [
        ("**循環構造**と呼びます。", "循環構造と呼びます。"),
        ("`set-cdr!`を使います。", "set-cdr!を使います。"),
        ("## 見出し\n本文です。", "本文です。"),
        ("> 引用文です。", "引用文です。"),
        ("- 箇条書きです。", "箇条書きです。"),
        ("1. 番号付きです。", "番号付きです。"),
    ],
)
def test_markdown_markers_are_stripped(text, expected_fragment) -> None:
    out = sanitize_for_narration(text)
    assert expected_fragment in out
    for marker in ("**", "##", "> ", "`"):
        assert marker not in out


def test_horizontal_rule_is_removed() -> None:
    out = sanitize_for_narration("前の話。\n---\n次の話。")
    assert "---" not in out
    assert "前の話。" in out and "次の話。" in out


def test_code_placeholder_tokens_are_removed() -> None:
    for token in ("〔コード1〕", "コード12", "〔 コード 3 〕"):
        out = sanitize_for_narration(f"ここで{token}を見てください。")
        assert "コード" not in out.replace("ここで", "").replace("を見てください。", "")


def test_whitespace_is_collapsed() -> None:
    assert sanitize_for_narration("あ　　い\n\n\nう") == "あ い う"


def test_empty_and_none_are_safe() -> None:
    assert sanitize_for_narration("") == ""
    assert sanitize_for_narration(None) == ""  # type: ignore[arg-type]


def test_plain_japanese_is_untouched() -> None:
    text = "リストの最後のセルのcdrは、空リストnilを指します。"
    assert sanitize_for_narration(text) == text


@pytest.mark.parametrize(
    "text",
    [
        "（ノートに「キー」と大きく書く）花凛さん、13.7では、",
        "こうです。（尻尾を軽くぱたぱたさせる）花凛さん、",
        "（耳をぴんと立てて、矢印を輪にした図を描く）はじめます。",
    ],
)
def test_stage_directions_are_not_narrated(text) -> None:
    out = sanitize_for_narration(text)
    assert "（" not in out and "）" not in out


@pytest.mark.parametrize(
    "text",
    [
        "リスト（エー、ビー、シー）は三つのセルです。",
        "テーブル（表）を実装します。",
        "第9章（データ主導）の話です。",
    ],
)
def test_ordinary_parentheticals_survive(text) -> None:
    """A blanket paren strip would eat meaningful asides."""
    assert sanitize_for_narration(text) == text


def test_horizontal_rule_is_removed_after_lines_are_collapsed() -> None:
    """Block text reaches the sanitiser with newlines already collapsed."""
    assert sanitize_for_narration("前の話。 --- 次の話。") == "前の話。 次の話。"


@pytest.mark.parametrize("literal,spoken", [("#f", "偽"), ("#t", "真")])
def test_scheme_booleans_are_spoken_as_words(literal, spoken) -> None:
    out = sanitize_for_narration(f"`{literal}`を返します。")
    assert out == f"{spoken}を返します。"


def test_hash_is_not_stripped_off_scheme_literals() -> None:
    """The old blanket ``#`` strip turned ``#f`` into the letter "f"."""
    assert "f を返します" not in sanitize_for_narration("`#f`を返します。")


@pytest.mark.parametrize(
    "heading,expected",
    [
        ("## 1．テーブルの基本構造", "テーブルの基本構造"),
        ("# 10．二次元テーブル", "二次元テーブル"),
        ("### 3. レコードを探す", "レコードを探す"),
    ],
)
def test_section_numbers_are_dropped_from_headings(heading, expected) -> None:
    assert sanitize_for_narration(heading) == expected


def test_numbers_in_body_text_are_kept() -> None:
    text = "13.7では、可変ペアを使います。"
    assert sanitize_for_narration(text) == text


def test_a_block_that_is_only_code_becomes_empty() -> None:
    """The splitter fills these with a spoken cue after merging."""
    assert sanitize_for_narration("```text\n┌───┐\n│ a │\n└───┘\n```") == ""
