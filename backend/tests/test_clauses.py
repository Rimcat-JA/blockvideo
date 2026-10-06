"""Clause splitting keeps one statement's negation or direction from colouring another."""
from __future__ import annotations

import pytest

from app.language_operations.clauses import clauses


@pytest.mark.parametrize(("text", "expected"), [
    ("話速を1.2倍にして動画は生成しないで", ["話速を1.2倍にして", "動画は生成しないで"]),
    ("話速を1.2倍にしないで音量を0.8倍にして", ["話速を1.2倍にしないで", "音量を0.8倍にして"]),
    ("字幕を60pxにしたうえで動画は生成しないで", ["字幕を60pxにしたうえで", "動画は生成しないで"]),
    ("以前の版に戻してから動画を作り直して", ["以前の版に戻してから", "動画を作り直して"]),
    ("ジョブ7は止めていただかなくていい", ["ジョブ7は止めていただかなくていい"]),
    ("字幕を64pxにしてください", ["字幕を64pxにしてください"]),
    ("話速を1.2倍から1.5倍にして", ["話速を1.2倍から1.5倍にして"]),
    ("動画3については戻さないで、状態だけ教えて", ["動画3については戻さないで", "状態だけ教えて"]),
    ("「話速を1.2倍にして」とは言っていません。音量だけ0.8倍にして",
     ["「話速を1.2倍にして」とは言っていません", "音量だけ0.8倍にして"]),
    ("音量を0.8倍にしてほしいので動画1に戻して", ["音量を0.8倍にしてほしいので", "動画1に戻して"]),
])
def test_clauses(text: str, expected: list[str]) -> None:
    assert clauses(text) == expected
