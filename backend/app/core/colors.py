"""Normalise subtitle colour text to ``#RRGGBB``.

Imports:
    ``re`` matches CSS-style hex colours.

Subtitle colours arrive from the API, the settings editor and language
operations ("字幕を赤にして"), so common English and Japanese colour names are
accepted alongside ``#RGB`` / ``#RRGGBB``.
"""
from __future__ import annotations

import re

_HEX = re.compile(r"#?(?:[0-9A-Fa-f]{3}|[0-9A-Fa-f]{6})")

# Colour names a person or a model is likely to write for captions.
NAMED_COLORS: dict[str, str] = {
    "white": "#FFFFFF", "白": "#FFFFFF", "しろ": "#FFFFFF", "ホワイト": "#FFFFFF",
    "black": "#000000", "黒": "#000000", "くろ": "#000000", "ブラック": "#000000",
    "red": "#FF0000", "赤": "#FF0000", "あか": "#FF0000", "レッド": "#FF0000",
    "yellow": "#FFFF00", "黄": "#FFFF00", "黄色": "#FFFF00", "きいろ": "#FFFF00", "イエロー": "#FFFF00",
    "blue": "#0000FF", "青": "#0000FF", "あお": "#0000FF", "ブルー": "#0000FF",
    "green": "#00FF00", "緑": "#00FF00", "みどり": "#00FF00", "グリーン": "#00FF00",
    "orange": "#FFA500", "オレンジ": "#FFA500", "橙": "#FFA500",
    "gray": "#808080", "grey": "#808080", "灰色": "#808080", "グレー": "#808080",
    "cyan": "#00FFFF", "水色": "#00FFFF", "シアン": "#00FFFF",
    "magenta": "#FF00FF", "マゼンタ": "#FF00FF",
    "pink": "#FFC0CB", "ピンク": "#FFC0CB",
    "purple": "#800080", "紫": "#800080", "むらさき": "#800080", "パープル": "#800080",
}


def normalize_color(value: str) -> str | None:
    """Return ``#RRGGBB`` for a hex colour or a known name, else ``None``.

    Args:
        value: Colour text such as ``#fff``, ``FF0000``, ``red`` or ``赤``.

    Returns:
        The upper-case ``#RRGGBB`` form, or ``None`` when it is not a colour.

    """
    text = value.strip()
    if _HEX.fullmatch(text):
        digits = text.lstrip("#")
        if len(digits) == 3:
            digits = "".join(char * 2 for char in digits)
        return f"#{digits.upper()}"
    return NAMED_COLORS.get(text.lower()) or NAMED_COLORS.get(text.removesuffix("色"))
