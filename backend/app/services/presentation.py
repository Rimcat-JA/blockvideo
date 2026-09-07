"""Sentence-aligned captions and locally rendered focus variants.

Display text remains independent of pronunciation overrides. Recorded sentence
times are reused only when they still describe that exact display text.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

from app.services import narration, subtitles


def display_spans(
    text: str, duration_ms: int, measured: Sequence[narration.SentenceSpan] = (),
) -> list[narration.SentenceSpan]:
    """Use valid measured spans, or proportionally time the original sentences.

    The fallback supports fake providers and projects created before timing
    metadata existed. It is an estimate, never a claimed audio measurement.
    """
    pieces = narration.split_sentences_with_offsets(text)
    if duration_ms <= 0 or not pieces:
        return []
    valid = len(measured) == len(pieces)
    previous = 0
    for span, piece in zip(measured, pieces):
        valid = valid and (
            span.text == piece.text
            and (span.char_start, span.char_end) == (piece.char_start, piece.char_end)
            and previous <= span.start_ms < span.end_ms <= duration_ms
        )
        previous = span.end_ms
    if valid:
        return list(measured)
    weights = [max(1, len(piece.text.strip())) for piece in pieces]
    total = sum(weights)
    cursor = 0
    elapsed = 0
    result = []
    for index, (piece, weight) in enumerate(zip(pieces, weights)):
        elapsed += weight
        end = duration_ms if index == len(pieces) - 1 else round(duration_ms * elapsed / total)
        if end <= cursor:
            continue
        result.append(narration.SentenceSpan(
            text=piece.text, char_start=piece.char_start, char_end=piece.char_end,
            start_ms=cursor, end_ms=end,
        ))
        cursor = end
    return result


def sentence_cues(
    text: str, *, duration_ms: int, measured: Sequence[narration.SentenceSpan],
    band_height: int, font_size: int, max_chars: int,
) -> tuple[list[subtitles.SubtitleCue], int]:
    """Fit each sentence separately; never reveal the next sentence early.

    Long sentences are split into readable chunks within their own measured
    interval. A shared font avoids size jumps at caption boundaries.
    """
    all_cues: list[subtitles.SubtitleCue] = []
    selected_size = font_size
    spans = display_spans(text, duration_ms, measured)
    for span in spans:
        cues, size = subtitles.build_band_cues(
            span.text, duration_ms=span.end_ms - span.start_ms,
            band_height=band_height, base_font_size=font_size,
            base_max_chars=max_chars, min_cue_ms=0,
        )
        selected_size = min(selected_size, size)
        all_cues.extend(subtitles.SubtitleCue(
            start_ms=cue.start_ms + span.start_ms,
            end_ms=cue.end_ms + span.start_ms,
            text=cue.text, margin_v=cue.margin_v,
        ) for cue in cues)
    if selected_size != font_size:
        for cue in all_cues:
            fit = subtitles.fit_text_to_band(
                cue.text.replace('\\N', ''), band_height=band_height,
                base_font_size=selected_size,
                base_max_chars=max(1, int(max_chars * font_size / selected_size)),
            )
            cue.text, cue.margin_v = fit.text, fit.margin_v
    return all_cues, selected_size


def focus_slides(
    *, plan: dict, text: str, measured: Sequence[narration.SentenceSpan],
    audio_ms: int, display_ms: int, primary: Path, directory: Path,
    width: int, height: int,
) -> list[tuple[Path, int]]:
    """Render exact visible matches and hold them for each spoken sentence.

    Variants are keyed by literal focus terms and reused within the block.
    Very short intervals are coalesced to avoid flashing and FFmpeg's minimum
    image interval. Unsupported visuals retain their primary image. Supported
    variants share a freshly rendered neutral baseline so rerendering an old
    project never jumps between the old and current slide layouts.
    """
    from app.services.image_renderer import render_visual_plan
    from app.services.visual_focus import focus_terms
    from app.services.hashing import short_hash

    spans = display_spans(text, audio_ms, measured)
    focused_spans = [(span, focus_terms(plan, span.text)) for span in spans]
    if not any(terms for _, terms in focused_spans):
        return [(primary, display_ms)]
    neutral = directory / f"focus_base_{short_hash(plan, width, height)}.png"
    render_visual_plan(
        plan, neutral, width=width, height=height,
        fallback_summary=text, focus_text=None,
    )
    events: list[tuple[int, Path]] = [(0, neutral)]
    cache: dict[tuple[str, ...], Path] = {}
    for span, terms in focused_spans:
        path = neutral
        if terms:
            if terms not in cache and len(cache) < 32:
                path = directory / f"focus_{short_hash(plan, terms, width, height)}.png"
                render_visual_plan(
                    plan, path, width=width, height=height,
                    fallback_summary=text, focus_text=span.text,
                )
                cache[terms] = path
            # At the variant limit an unseen label returns to the neutral
            # slide, rather than leaving an unrelated earlier label lit up.
            path = cache.get(terms, neutral)
        at = min(display_ms, max(0, span.start_ms))
        if at < 100:
            events[0] = (0, path)
        elif path != events[-1][1] and at - events[-1][0] >= 100:
            if len(events) >= 31:
                # Each interval is one FFmpeg input. Bound the command length
                # on Windows even when a long block alternates two labels.
                if events[-1][1] != neutral:
                    events.append((at, neutral))
                break
            events.append((at, path))
    result: list[tuple[Path, int]] = []
    for index, (start, path) in enumerate(events):
        end = events[index + 1][0] if index + 1 < len(events) else display_ms
        if end - start < 100 and result:
            previous_path, previous_ms = result[-1]
            result[-1] = (previous_path, previous_ms + end - start)
        elif end > start:
            result.append((path, end - start))
    return result or [(neutral, display_ms)]
