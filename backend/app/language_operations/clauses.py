"""Split a Japanese request into clauses so each statement is read on its own.

"話速を1.2倍にして動画は生成しないで" holds two statements: a request about speed
and a negation about generation. Guards that look for numbers, directions or
negations read one clause at a time so one statement never colours another.
Quoted text (「…」) is never split: "「話速を1.2倍にして」とは言っていません" is one
negated statement.
"""
from __future__ import annotations

import re
import unicodedata

# A te-form request ends a clause unless it continues into a polite negation
# ("止めていただかなくていい"), a negated te-form ("言っていません", "変えてない") or a
# request ending ("してください"). "について" / "において" are not te-form requests.
_CONTINUES = r"いただかな|もらわな|くれな|ほしくな|ください|ほしい|から|も|いな|いませ|な"
_TE_END = "".join(f"(?<={ending})" if index == 0 else f"|(?<={ending})" for index, ending in enumerate(
    ("して", "って", "んで", "いて", "えて", "けて", "せて", "めて", "べて", "れて", "きて", "みて", "ちて", "ないで")))
_BOUNDARY = (r"[、。,!！?？\n]|\.(?!\d)|(?<=てから)|(?<=うえで)|(?<=上で)|(?<=ので)"
             rf"|(?:{_TE_END})(?<!ついて)(?<!おいて)(?!{_CONTINUES})")
_QUOTE = r"「[^」]*」|『[^』]*』|\"[^\"]*\""


def normalized(text: str) -> str:
    return unicodedata.normalize("NFKC", text).casefold()


def clauses(text: str) -> list[str]:
    """Non-empty clauses of normalized text, in order; quotes stay inside their clause."""
    value = normalized(text)
    quoted = [match.span() for match in re.finditer(_QUOTE, value)]
    parts, start = [], 0
    for match in re.finditer(_BOUNDARY, value):
        if any(left < match.start() < right for left, right in quoted):
            continue
        parts.append(value[start:match.start()])
        start = match.end()
    parts.append(value[start:])
    return [part.strip() for part in parts if part.strip()]
