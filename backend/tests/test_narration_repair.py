"""Narration that lost its meaning with the code is detected and rewritten.

Scripts written as documents let the code complete the sentence — "一つの
データを、`キー . 値` というペアで表します". Code is shown, not spoken, so
that narration reads as "一つのデータを、というペアで表します". These tests
pin the detector; the rewrite itself needs a provider and is exercised
end-to-end.
"""
from __future__ import annotations

import pytest

from app.services.splitter import narration_has_gaps


FENCED = "一つのデータを、\n```text\nキー . 値\n```\nというペアで表します。"


@pytest.mark.parametrize(
    "narration",
    [
        "一次元テーブルでは、一つのデータを、 というペアで表します。",
        "たとえば、 は、 というレコードです。",
        "結果は、 です。",
        "これを省略したものが、 です。",
        "たとえば、 なら、 は、最初のレコードです。",
        "は、 現在調べている先頭レコードのキーを取り出しています。",
    ],
)
def test_broken_narration_is_detected(narration) -> None:
    assert narration_has_gaps(narration, FENCED)


@pytest.mark.parametrize(
    "narration",
    [
        "一次元テーブルでは、一つのデータをキーと値の組で表します。",
        "この手続きは、レコードのリストを先頭から順番に調べます。",
        "空のテーブルは、見出しだけを持ちます。",
        "セル1のcarは値a、cdrはセル2を指します。",
        "リスト、エー、ビー、シーは三つのセルで構成されます。",
    ],
)
def test_healthy_narration_is_not_flagged(narration) -> None:
    assert not narration_has_gaps(narration, FENCED)


@pytest.mark.parametrize(
    "narration",
    [
        "ここでいうテーブルは、行列ではなく、 キーを渡すと値を取り出せる構造です。",
        "複数のレコードをリストにすると、 となります。",
        "本来なら、 として、変数の束縛を変更する必要があります。",
    ],
)
def test_whitespace_left_by_a_removed_fence_is_detected(narration) -> None:
    """Sanitising a fence leaves a space; Japanese prose has none after 、."""
    assert narration_has_gaps(narration, FENCED)


@pytest.mark.parametrize(
    "narration",
    [
        "というレコードです。複数のレコードをリストにします。",
        "となります。ところが、変数tableは古いセルを指しています。",
    ],
)
def test_block_opening_mid_sentence_is_detected(narration) -> None:
    assert narration_has_gaps(narration, FENCED)


def test_comma_followed_by_という_without_a_gap_is_not_flagged() -> None:
    """「探す、という二段階」 is ordinary Japanese, not a hole."""
    text = "次に第2キーでレコードを探す、という二段階になります。"
    assert not narration_has_gaps(text, FENCED)


def test_a_block_without_code_is_never_flagged() -> None:
    """No fence means nothing was removed, so there is no hole to repair."""
    assert not narration_has_gaps("たとえば、 は、 そうです。", "ただの文章です。")


def test_empty_inputs_are_safe() -> None:
    assert not narration_has_gaps("", FENCED)
    assert not narration_has_gaps("なにか。", "")
    assert not narration_has_gaps(None, None)  # type: ignore[arg-type]
