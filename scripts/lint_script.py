"""Check an authored script before feeding it to BlockVideo.

Reports the mistakes that reach the screen silently: a ```slide box whose
inner lines do not match its border (the renderer draws exactly what is
written, so a ragged box stays ragged), a sentence too long for one caption,
and identifiers spelled in katakana.

    uv run python ../scripts/lint_script.py ../script.txt

Imports:
    ``io`` wraps stdout with UTF-8 for Japanese diagnostics.
    ``re`` removes fences and separates paragraphs/sentences.
    ``sys`` handles the command-line entry point and backend import path.
    ``Path`` identifies the source script.
"""
from __future__ import annotations

import io
import re
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.services.subtitles import _split_sentences  # noqa: E402
from app.services.visual_planner import (  # noqa: E402
    _SLIDE_FENCE_RX,
    slide_alignment_issues,
)

# Shared preflight limits and pronunciation markers.
MAX_SENTENCE_CHARS = 72
KATAKANA_IDENTIFIERS = ("アソック", "クッダー", "コンス", "セット・クッダー", "ラムダ")


def main(path: Path) -> int:
    """Validate authored slides, narration lengths, and pronunciations.

    Args:
        path: UTF-8 script file to inspect.

    Returns:
        ``0`` when no problems are found, otherwise ``1``.

    Side Effects:
        Reads the script and prints one diagnostic per detected issue plus a
        summary of slide/paragraph counts.

    """
    text = path.read_text(encoding="utf-8")
    problems = 0

    # Slides are located by scanning the fences directly. Splitting the script
    # into blocks first does not work: hand-drawn art contains blank lines, so
    # a paragraph split lands in the middle of a drawing.
    slides = list(_SLIDE_FENCE_RX.finditer(text))
    for n, match in enumerate(slides, 1):
        heading = (match.group(1) or "").strip() or "(見出しなし)"
        for issue in slide_alignment_issues(match.group(2)):
            print(f"スライド{n}「{heading}」 枠がずれています — {issue}")
            problems += 1

    narration = re.sub(r"```[\s\S]*?```", "", text)
    paragraphs = [p for p in re.split(r"\n\s*\n", narration) if p.strip()]

    for sentence in _split_sentences(narration):
        body = " ".join(sentence.split())
        if len(body) > MAX_SENTENCE_CHARS:
            print(f"長すぎる文 ({len(body)}文字 / 上限{MAX_SENTENCE_CHARS}): {body[:40]}…")
            problems += 1

    for katakana in KATAKANA_IDENTIFIERS:
        count = narration.count(katakana)
        if count:
            print(f"カタカナ表記 「{katakana}」×{count} — 英字のまま書いてください")
            problems += 1

    missing = len(paragraphs) - len(slides)
    print()
    print(f"スライド指定      : {len(slides)}")
    print(f"ナレーション段落  : {len(paragraphs)}")
    if missing > 0:
        print(f"スライドなしの段落: {missing}  (この数だけモデルが図を設計します)")
    print(f"問題              : {problems}")
    return 1 if problems else 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main(Path(sys.argv[1])))
