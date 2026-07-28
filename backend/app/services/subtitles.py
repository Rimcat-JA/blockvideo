"""Subtitle (ASS) writer.

Generates an Advanced SubStation Alpha file with one cue per block. Source
text is wrapped to ``subtitle_max_chars_per_line`` after breaking at
natural Japanese punctuation if possible.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass
class SubtitleCue:
    start_ms: int
    end_ms: int
    text: str
    # Per-cue vertical margin. ASS Dialogue rows override the style's MarginV,
    # which lets cues of different line counts each sit centred in the band.
    margin_v: int | None = None


# Colors -> ASS &HBBGGRR& form.
def _rgb_to_ass(hex_color: str) -> str:
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    if len(h) != 6:
        return "&H00FFFFFF"
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"&H00{b:02X}{g:02X}{r:02X}"


_BREAK_CHARS = "。、！？!?」』"

# Characters that must not start a line: small kana, the long vowel mark and
# closing punctuation all continue the word before them.
_NO_LEAD = "ぁぃぅぇぉっゃゅょゎァィゥェォッャュョヮーヽヾ、。，．！？!?)）」』】〕・"


def _is_hiragana(ch: str) -> bool:
    return "ぁ" <= ch <= "ゟ"


def _hard_split(text: str, limit: int) -> list[str]:
    """Cut an over-long run into pieces, avoiding the middle of a word.

    Slicing every ``limit`` characters produced captions like 「新レコー」 /
    「ドを差し込む」. Japanese has no spaces to break on, but a bunsetsu
    almost always ends in hiragana (a particle or inflection), so stepping
    back to the nearest hiragana gives a break a reader accepts. Falls back
    to the blunt cut when there is no such position nearby.
    """
    pieces: list[str] = []
    rest = text
    while len(rest) > limit:
        cut = limit
        for i in range(limit, max(2, limit - 12), -1):
            if _is_hiragana(rest[i - 1]) and rest[i] not in _NO_LEAD:
                cut = i
                break
        pieces.append(rest[:cut])
        rest = rest[cut:]
    if rest:
        pieces.append(rest)
    return pieces


def _wrap_japanese(text: str, max_chars: int) -> list[str]:
    """Wrap a Japanese string into lines of <= ``max_chars``.

    Greedy packing over punctuation-delimited segments: a line takes as many
    whole segments as fit, so lines are filled rather than broken at every
    ``。``. The previous behaviour emitted one line per sentence and could
    only merge pairs, which meant shrinking the font barely reduced the line
    count — and the subtitle still overflowed its band.
    """
    text = text.strip()
    if not text:
        return [""]
    if len(text) <= max_chars:
        return [text]

    # 1) split into segments that end just after a break character.
    segments: list[str] = []
    buf = ""
    for ch in text:
        buf += ch
        if ch in _BREAK_CHARS:
            segments.append(buf)
            buf = ""
    if buf:
        segments.append(buf)

    # 2) any segment longer than a full line is hard-split.
    pieces: list[str] = []
    for seg in segments:
        if len(seg) <= max_chars:
            pieces.append(seg)
        else:
            pieces.extend(_hard_split(seg, max_chars))

    # 3) greedily pack pieces into lines.
    lines: list[str] = []
    cur = ""
    for piece in pieces:
        if not cur:
            cur = piece
        elif len(cur) + len(piece) <= max_chars:
            cur += piece
        else:
            lines.append(cur)
            cur = piece
    if cur:
        lines.append(cur)
    return lines


# libass draws a line roughly 1.25x the nominal font size tall (glyphs plus
# leading). Used to predict how many wrapped lines fit inside the band.
LINE_HEIGHT_RATIO = 1.25


@dataclass
class BandFit:
    """A subtitle laid out to fit inside the burned-in band."""

    lines: list[str]
    font_size: int
    margin_v: int

    @property
    def text(self) -> str:
        return "\\N".join(self.lines)


def fit_text_to_band(
    text: str,
    *,
    band_height: int,
    base_font_size: int,
    base_max_chars: int,
    min_font_size: int = 24,
    pad: int = 16,
) -> BandFit:
    """Lay out *text* so it is guaranteed to fit inside the subtitle band.

    The band is a fixed strip at the bottom of the frame; the slide occupies
    everything above it. A cue that needs more lines than the band can hold
    spills upward over the slide, which is what made subtitles overlap the
    diagrams. Rather than clipping or dropping narration, shrink the font
    until the wrapped text fits — a smaller font also fits more characters
    per line, so line count drops faster than size does.

    Returns the wrapped lines, the font size to use, and the vertical margin
    that centres the block inside the band.
    """
    body = (text or "").strip()
    usable = max(1, band_height - pad * 2)

    best: BandFit | None = None
    size = max(min_font_size, base_font_size)
    while size >= min_font_size:
        # Narrower glyphs at smaller sizes mean more characters per line.
        max_chars = max(8, int(base_max_chars * base_font_size / size))
        lines = _wrap_japanese(body, max_chars) if body else [""]
        text_height = len(lines) * size * LINE_HEIGHT_RATIO
        if text_height <= usable:
            margin_v = max(pad, int((band_height - text_height) / 2))
            return BandFit(lines=lines, font_size=size, margin_v=margin_v)
        best = BandFit(lines=lines, font_size=size, margin_v=pad)
        size -= 2

    # Nothing fits even at the floor; use the smallest layout we produced and
    # pin it to the bottom so it intrudes on the slide as little as possible.
    return best or BandFit(lines=[body], font_size=min_font_size, margin_v=pad)


_SENTENCE_END = "。！？!?"
_SOFT_BREAK = "、，,；;："


def _split_sentences(text: str) -> list[str]:
    """Split on sentence enders, keeping the punctuation with its sentence."""
    out: list[str] = []
    buf = ""
    for ch in text:
        buf += ch
        if ch in _SENTENCE_END:
            out.append(buf)
            buf = ""
    if buf.strip():
        out.append(buf)
    return [s for s in out if s.strip()]


def _split_soft(segment: str, limit: int) -> list[str]:
    """Break one over-long sentence at commas, then hard-split if needed."""
    parts: list[str] = []
    buf = ""
    for ch in segment:
        buf += ch
        if ch in _SOFT_BREAK and len(buf) >= limit // 2:
            parts.append(buf)
            buf = ""
    if buf:
        parts.append(buf)

    out: list[str] = []
    for part in parts:
        if len(part) <= limit:
            out.append(part)
        else:
            out.extend(_hard_split(part, limit))
    return out


def chunk_narration(text: str, *, chars_per_cue: int) -> list[str]:
    """Group narration into subtitle-sized chunks at natural boundaries.

    A block's slide stays on screen for the whole narration, but the caption
    should not: a long block shown as a single cue has to shrink to a size
    that is hard to read. Chunking lets the caption advance underneath a
    stationary slide, one or two sentences at a time.
    """
    body = (text or "").strip()
    if not body:
        return []
    if len(body) <= chars_per_cue:
        return [body]

    pieces: list[str] = []
    for sentence in _split_sentences(body):
        if len(sentence) <= chars_per_cue:
            pieces.append(sentence)
        else:
            pieces.extend(_split_soft(sentence, chars_per_cue))

    chunks: list[str] = []
    cur = ""
    for piece in pieces:
        if not cur:
            cur = piece
        elif len(cur) + len(piece) <= chars_per_cue:
            cur += piece
        else:
            chunks.append(cur)
            cur = piece
    if cur:
        chunks.append(cur)
    return chunks


def build_band_cues(
    text: str,
    *,
    duration_ms: int,
    band_height: int,
    base_font_size: int,
    base_max_chars: int,
    min_cue_ms: int = 1200,
    char_time: Callable[[int], float] | None = None,
) -> tuple[list[SubtitleCue], int]:
    """Lay a block's narration out as a sequence of band-sized cues.

    Returns the cues and the font size to use for all of them.

    ``char_time`` maps a character offset in the narration to the moment it
    is spoken, measured from VOICEVOX (see ``narration.char_time_fn``). Given
    it, cue changes land on the voice. Without it we fall back to spreading
    ``duration_ms`` across the characters, which is off by 890 ms on average
    for Japanese — characters and moras are not proportional — and is kept
    only for projects whose audio predates the measurement.

    A single font is used for the whole block: cues change while the slide
    does not, so a caption that resized between sentences would draw the eye
    for no reason.
    """
    body = (text or "").strip()
    if not body or duration_ms <= 0:
        return [], base_font_size

    # Start from the theoretical capacity, then shrink the chunk size until
    # the chunks actually fit at full size. Line packing breaks on punctuation
    # and leaves slack, so the theoretical figure is optimistic: assuming it
    # left a 68-character cue spilling onto a third line and shrinking the
    # font, which is exactly what chunking is meant to avoid.
    usable = max(1, band_height - 32)
    max_lines = max(1, int(usable // (base_font_size * LINE_HEIGHT_RATIO)))
    chars_per_cue = max(base_max_chars, max_lines * base_max_chars)

    step = max(4, base_max_chars // 4)
    chunks = chunk_narration(body, chars_per_cue=chars_per_cue)
    while chars_per_cue > base_max_chars:
        if all(
            fit_text_to_band(
                chunk,
                band_height=band_height,
                base_font_size=base_font_size,
                base_max_chars=base_max_chars,
            ).font_size == base_font_size
            for chunk in chunks
        ):
            break
        chars_per_cue = max(base_max_chars, chars_per_cue - step)
        chunks = chunk_narration(body, chars_per_cue=chars_per_cue)

    if not chunks:
        return [], base_font_size

    # Where each chunk boundary falls in time. With a measured mapping this is
    # the voice; without one it degrades to the old character share.
    total_chars = sum(len(c) for c in chunks)

    def at(offset: int) -> float:
        if char_time is not None:
            return char_time(offset)
        return duration_ms * offset / max(1, total_chars)

    # Merge chunks that would flash by faster than they can be read.
    merged: list[str] = []
    cursor_chars = 0
    for chunk in chunks:
        span = at(cursor_chars + len(chunk)) - at(cursor_chars)
        if merged and span < min_cue_ms and len(merged[-1]) + len(chunk) <= chars_per_cue * 2:
            merged[-1] += chunk
        else:
            merged.append(chunk)
        cursor_chars += len(chunk)
    chunks = merged
    total_chars = sum(len(c) for c in chunks)

    fits = [
        fit_text_to_band(
            chunk,
            band_height=band_height,
            base_font_size=base_font_size,
            base_max_chars=base_max_chars,
        )
        for chunk in chunks
    ]
    font_size = min(f.font_size for f in fits)
    if font_size != base_font_size:
        # Re-fit everything at the shared size so margins stay consistent.
        fits = [
            fit_text_to_band(
                chunk,
                band_height=band_height,
                base_font_size=font_size,
                base_max_chars=int(base_max_chars * base_font_size / font_size),
            )
            for chunk in chunks
        ]

    cues: list[SubtitleCue] = []
    cursor = 0
    cursor_chars = 0
    for i, (chunk, fit) in enumerate(zip(chunks, fits)):
        cursor_chars += len(chunk)
        if i == len(chunks) - 1:
            end = duration_ms
        else:
            end = min(duration_ms, max(cursor + 1, round(at(cursor_chars))))
        cues.append(
            SubtitleCue(
                start_ms=cursor,
                end_ms=max(cursor + 1, end),
                text=fit.text,
                margin_v=fit.margin_v,
            )
        )
        cursor = cues[-1].end_ms
    return cues, font_size


def _escape_ass(text: str) -> str:
    """Escape characters that have special meaning in ASS."""
    # Newlines inside a cue become \N
    text = text.replace("\r", "").replace("\n", "\\N")
    return text.replace("{", "(").replace("}", ")")


def ms_to_ass_time(ms: int) -> str:
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, x = divmod(rem, 1_000)
    return f"{h:d}:{m:02d}:{s:02d}.{x // 10:02d}"


def build_subtitle_cues(
    blocks: list[tuple[int, int, int, str]],  # (idx, start_ms, end_ms, source_text)
    *,
    max_chars_per_line: int = 36,
) -> list[SubtitleCue]:
    cues: list[SubtitleCue] = []
    for idx, start_ms, end_ms, text in blocks:
        wrapped = _wrap_japanese(text or "", max_chars_per_line)
        joined = "\\N".join(wrapped)
        cues.append(SubtitleCue(start_ms=start_ms, end_ms=end_ms, text=joined))
    return cues


def render_ass(
    cues: list[SubtitleCue],
    output_path: Path,
    *,
    width: int = 1920,
    height: int = 1080,
    font_size: int = 48,
    position: str = "bottom",
    text_color: str = "#FFFFFF",
    outline_color: str = "#000000",
    background: bool = True,
    font_name: str = "Noto Sans CJK JP",
    margin_v_override: int | None = None,
) -> int:
    """Write an .ass file and return the number of cues written.

    ``margin_v_override`` forces the vertical margin (distance from the
    bottom edge for bottom-aligned subtitles). Used by per-block burn-in
    so subtitles land inside the burned-in band rather than overlapping the
    slide. When ``None``, the margin is derived from ``position``.
    """
    if margin_v_override is not None:
        margin_v = margin_v_override
    else:
        margin_v = {"top": 60, "middle": 0, "bottom": 80}.get(position, 80)
    alignment = {"top": 8, "middle": 5, "bottom": 2}.get(position, 2)
    primary = _rgb_to_ass(text_color)
    outline = _rgb_to_ass(outline_color)
    back = "&H80000000" if background else "&H00000000"
    body = [
        "[Script Info]",
        "; Generated by BlockVideo",
        "Title: BlockVideo Subtitles",
        "ScriptType: v4.00+",
        "Collisions: Normal",
        f"PlayResX: {width}",
        f"PlayResY: {height}",
        "WrapStyle: 0",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        (
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
            "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
            "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
            "Alignment, MarginL, MarginR, MarginV, Encoding"
        ),
        (
            f"Style: Default,{font_name},{font_size},{primary},{primary},{outline},{back},"
            f"0,0,0,0,100,100,0,0,1,2,1,{alignment},80,80,{margin_v},1"
        ),
        "",
        "[Events]",
        (
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, "
            "Effect, Text"
        ),
    ]
    for c in cues:
        # MarginL, MarginR, MarginV on the Dialogue row override the style; 0
        # means "inherit". Per-cue MarginV keeps a 1-line and a 3-line caption
        # each centred in the band.
        cue_margin = c.margin_v if c.margin_v is not None else 0
        body.append(
            "Dialogue: 0,"
            f"{ms_to_ass_time(c.start_ms)},{ms_to_ass_time(c.end_ms)},Default,,"
            f"0,0,{cue_margin},,"
            f"{_escape_ass(c.text)}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(body), encoding="utf-8")
    return len(cues)


def build_vtt_time(ms: int) -> str:
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, x = divmod(rem, 1_000)
    return f"{h:02d}:{m:02d}:{s:02d}.{x:03d}"


def render_vtt(cues: list[SubtitleCue], output_path: Path) -> int:
    body = ["WEBVTT", ""]
    for i, c in enumerate(cues, 1):
        body.append(str(i))
        body.append(f"{build_vtt_time(c.start_ms)} --> {build_vtt_time(c.end_ms)}")
        body.append(c.text.replace("\\N", "\n"))
        body.append("")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(body), encoding="utf-8")
    return len(cues)
