"""Sentence-level narration planning against VOICEVOX.

Everything downstream of the voice — when a caption changes, when a slide
changes — needs to know *when* each sentence is spoken. Estimating that from
character counts is badly wrong for Japanese: 「登録」 is two characters and
four moras, ``insert!`` is seven characters and five, ``1`` is one character
and two. Measured over a real project the resulting captions drifted 890 ms
on average and 6.2 s at worst.

VOICEVOX already knows the answer. ``/audio_query`` returns per-mora
``consonant_length`` / ``vowel_length`` before anything is synthesised, and
those add up to the real audio within 0.12%. So we query **per sentence**,
concatenate the resulting accent phrases into one query, and synthesise that
— the accent phrases come back identical to a whole-text query (verified
mora-for-mora on three real blocks), so the voice is unchanged, but we now
know where every sentence boundary falls.

Splitting the query per sentence also gives us the one place to lengthen the
gap at each ``。`` into a breath.

Imports:
    ``asyncio`` bounds concurrent ``/audio_query`` requests.
    ``copy`` prevents mutations to provider-returned query objects.
    ``json`` persists measured spans beside each audio file.
    Dataclasses/types describe spans and callback signatures.
    ``VoicevoxClient`` and settings provide the live synthesis boundary.
"""
from __future__ import annotations

import asyncio
import copy
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from app.core.logging import log
from app.providers.voicevox import VoicevoxClient, VoicevoxSettings
from app.services.pronunciation import Pronunciation, normalize_overrides, pronunciation_query

# Same enders the subtitle chunker breaks on, so cue boundaries and sentence
# spans line up exactly instead of nearly.
SENTENCE_END_CHARS = "。！？!?"


@dataclass
class SentenceSpan:
    """One sentence located in both source text and synthesized audio.

    Attributes:
        text: Exact source sentence for captions, before speech-only replacements.
        char_start, char_end: Half-open character offsets in the block text.
        start_ms, end_ms: Predicted or rescaled audio interval in milliseconds.

    """

    text: str
    char_start: int
    char_end: int
    start_ms: int
    end_ms: int

    def to_dict(self) -> dict[str, Any]:
        """Serialize the span into JSON-compatible primitive values."""
        return {
            "text": self.text,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SentenceSpan":
        """Reconstruct a span with defensive primitive conversions.

        Args:
            data: Mapping read from a narration JSON file.

        Returns:
            A ``SentenceSpan`` with missing values defaulted to empty/zero.

        """
        return cls(
            text=str(data.get("text", "")),
            char_start=int(data.get("char_start", 0)),
            char_end=int(data.get("char_end", 0)),
            start_ms=int(data.get("start_ms", 0)),
            end_ms=int(data.get("end_ms", 0)),
        )


@dataclass
class NarrationPlan:
    """A tuned query plus the predicted location of each sentence.

    Attributes:
        query: Combined VOICEVOX query ready for ``synthesis``.
        spans: Sentence offsets and predicted time ranges.
        predicted_ms: Total duration estimated from mora lengths and margins.

    """

    query: dict[str, Any]
    spans: list[SentenceSpan]
    predicted_ms: int


@dataclass
class _Piece:
    """Internal sentence fragment with source-text character offsets.

    It is kept separate from ``SentenceSpan`` because it exists before a live
    engine supplies timing information.
    """

    text: str
    char_start: int
    char_end: int


def _is_identifier_suffix(text: str, index: int) -> bool:
    """Keep Scheme/Ruby-style ``insert!``/``empty?`` inside their sentence.

    The identifier must continue directly into another identifier character or
    a Japanese particle. A capitalized English exclamation such as ``Hello!``
    keeps its ordinary boundary, including before a Japanese greeting.
    """
    if text[index] not in "!?" or index == 0 or index + 1 >= len(text):
        return False
    identifier = re.search(r"[A-Za-z_][A-Za-z0-9_]*$", text[:index])
    if identifier is None:
        return False
    next_text = text[index + 1:]
    if re.match(r"[A-Za-z0-9_]", next_text):
        return True
    word = identifier.group()
    if word[0].isupper() and not word.isupper():
        return False
    # Closing quotes are common around a function name: 「insert!」を使う。
    return bool(re.match(r"[」』`\"')\]]*(?:[をがはにへとも]|で|から|まで)", next_text))


def split_sentences_with_offsets(text: str) -> list[_Piece]:
    """Split *text* into sentences, keeping each one's character offsets.

    The offsets are what let a caption's character position be converted into
    a time, so the pieces must tile the string exactly — nothing dropped,
    nothing overlapping.

    Args:
        text: Block narration to split at ``SENTENCE_END_CHARS``.

    Returns:
        Ordered pieces whose half-open offsets tile the input text, including
        trailing non-sentence text and folded whitespace-only tails.

    """
    pieces: list[_Piece] = []
    start = 0
    for i, ch in enumerate(text):
        if ch in SENTENCE_END_CHARS and not _is_identifier_suffix(text, i):
            pieces.append(_Piece(text[start : i + 1], start, i + 1))
            start = i + 1
    if start < len(text):
        pieces.append(_Piece(text[start:], start, len(text)))
    # A piece with nothing to say (stray whitespace after the final 。) would
    # make VOICEVOX return an empty query; fold it into its predecessor.
    merged: list[_Piece] = []
    for piece in pieces:
        if not piece.text.strip() and merged:
            merged[-1] = _Piece(
                merged[-1].text + piece.text, merged[-1].char_start, piece.char_end
            )
        else:
            merged.append(piece)
    return merged


def _phrase_seconds(phrase: dict[str, Any]) -> float:
    """Sum voiced and pause mora durations in one accent phrase.

    Args:
        phrase: VOICEVOX accent-phrase mapping.

    Returns:
        Duration in seconds, treating missing/null lengths as zero.

    """
    total = 0.0
    for mora in phrase.get("moras") or []:
        total += (mora.get("consonant_length") or 0.0) + (mora.get("vowel_length") or 0.0)
    pause = phrase.get("pause_mora")
    if pause:
        total += (pause.get("consonant_length") or 0.0) + (pause.get("vowel_length") or 0.0)
    return total


def _breath_mora(seconds: float) -> dict[str, Any]:
    """Build a silent pause mora lasting ``seconds``.

    Args:
        seconds: Desired non-negative pause length.

    Returns:
        VOICEVOX-compatible pause-mora mapping.

    """
    return {
        "text": "、",
        "consonant": None,
        "consonant_length": None,
        "vowel": "pau",
        "vowel_length": max(0.0, seconds),
        "pitch": 0.0,
    }


def sentence_pause_for(
    text: str,
    *,
    pacing_mode: str,
    fixed_seconds: float,
    next_text: str = "",
    focus_terms: Sequence[str] = (),
    next_focus_terms: Sequence[str] = (),
) -> float:
    """Choose a wall-clock reading hold after a sentence, before the next one.

    Adaptive defaults are 0.6 s for explanation, 1.0 s for a visual change,
    and 1.5 s for a dense sentence or an explicitly signposted key point.
    Fixed mode retains the existing project setting exactly.
    """
    if pacing_mode == "fixed":
        return max(0.0, fixed_seconds)
    if pacing_mode != "adaptive":
        raise ValueError(f"Unknown narration pacing mode: {pacing_mode}")
    compact = re.sub(r"\s", "", text)
    if (
        len(compact) >= 60
        or (len(compact) >= 30 and compact.count("、") >= 3)
        or len(set(focus_terms)) >= 3
        or any(marker in compact for marker in ("重要なのは", "ポイントは", "覚えて", "要点は", "まとめると"))
    ):
        return 1.5
    if (
        any(marker in compact for marker in ("図", "矢印", "注目", "見て", "切り替", "次に", "続いて"))
        or next_text.lstrip().startswith(("次に", "続いて", "では、", "一方", "最後に"))
        or (bool(focus_terms or next_focus_terms) and set(focus_terms) != set(next_focus_terms))
    ):
        return 1.0
    return 0.6


async def build_narration_plan(
    text: str,
    settings: VoicevoxSettings,
    client: VoicevoxClient,
    *,
    sentence_pause_seconds: float,
    concurrency: int = 4,
    pacing_mode: str = "fixed",
    pronunciation_overrides: Sequence[Mapping[str, Any] | Pronunciation] | None = None,
    focus_terms: Sequence[Sequence[str]] | None = None,
) -> NarrationPlan | None:
    """Query VOICEVOX per sentence and assemble one query for the block.

    Args:
        text: Block narration text.
        settings: Speaker and query tuning values.
        client: Live VOICEVOX client; fake clients intentionally skip planning.
        sentence_pause_seconds: Wall-clock gap between sentences in fixed mode.
        concurrency: Maximum simultaneous audio-query requests.
        pacing_mode: ``adaptive`` for sentence-specific holds, otherwise ``fixed``.
        pronunciation_overrides: Project-local readings and optional accent indices.
        focus_terms: Diagram focus labels for each source sentence, in order.

    Returns:
        A combined ``NarrationPlan`` or ``None`` when the plan cannot be built
        (no live engine, empty text, or unusable provider responses).

    Returns ``None`` when the plan cannot be built (no live engine, empty
    text, an engine that answered with nothing usable). Callers fall back to
    synthesising the whole text in one go, which is what happened before this
    module existed.

    """
    body = text or ""
    if not body.strip():
        return None
    overrides = normalize_overrides(pronunciation_overrides)
    if pacing_mode not in {"fixed", "adaptive"}:
        raise ValueError(f"Unknown narration pacing mode: {pacing_mode}")
    if not isinstance(client, VoicevoxClient):
        # Fake client (tests, demo mode) has no audio_query.
        return None

    pieces = split_sentences_with_offsets(body)
    if not pieces:
        return None

    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def query_one(piece: _Piece) -> dict[str, Any]:
        async with semaphore:
            return await pronunciation_query(piece.text, settings.speaker_id, client, overrides)

    try:
        queries = await asyncio.gather(*(query_one(p) for p in pieces))
    except Exception as exc:  # engine hiccup — fall back rather than fail the block
        log.warning(
            "narration plan failed, falling back to single-shot synthesis: {error}",
            error=f"{exc.__class__.__name__}: {exc}",
        )
        return None

    merged = copy.deepcopy(queries[0])
    client.apply_settings(merged, settings)
    speed = max(0.1, float(merged.get("speedScale") or 1.0))
    # Newer engines expose global pause overrides. Clear them so per-sentence
    # holds remain accurate and keep the value in wall-clock seconds at any speed.
    merged["pauseLength"] = None
    merged["pauseLengthScale"] = 1.0

    def terms(index: int) -> Sequence[str]:
        return focus_terms[index] if focus_terms and index < len(focus_terms) else ()

    phrases: list[dict[str, Any]] = []
    ranges: list[tuple[int, int]] = []
    for i, query in enumerate(queries):
        own = copy.deepcopy(query.get("accent_phrases") or [])
        if own and i < len(queries) - 1:
            # A sentence queried on its own carries no trailing pause, so this
            # sets the gap rather than adding to one.
            pause_seconds = sentence_pause_for(
                pieces[i].text,
                pacing_mode=pacing_mode,
                fixed_seconds=sentence_pause_seconds,
                next_text=pieces[i + 1].text,
                focus_terms=terms(i),
                next_focus_terms=terms(i + 1),
            )
            own[-1]["pause_mora"] = _breath_mora(pause_seconds * speed)
        start = len(phrases)
        phrases.extend(own)
        ranges.append((start, len(phrases)))
    if not phrases:
        return None
    merged["accent_phrases"] = phrases

    pre = float(merged.get("prePhonemeLength") or 0.0)
    post = float(merged.get("postPhonemeLength") or 0.0)
    cumulative = [0.0]
    for phrase in phrases:
        cumulative.append(cumulative[-1] + _phrase_seconds(phrase))

    def at_ms(phrase_index: int) -> int:
        return int(round((pre + cumulative[phrase_index]) / speed * 1000))

    spans = [
        SentenceSpan(
            text=piece.text,
            char_start=piece.char_start,
            char_end=piece.char_end,
            start_ms=at_ms(lo),
            end_ms=at_ms(hi),
        )
        for piece, (lo, hi) in zip(pieces, ranges)
    ]
    predicted_ms = int(round((pre + cumulative[-1] + post) / speed * 1000))
    return NarrationPlan(query=merged, spans=spans, predicted_ms=predicted_ms)


def rescale_spans(
    spans: Sequence[SentenceSpan], *, predicted_ms: int, actual_ms: int
) -> list[SentenceSpan]:
    """Stretch spans onto the duration the encoder actually produced.

    VOICEVOX's own prediction lands within a few tenths of a percent, but the
    subtitle file has to agree with the audio file exactly or the last cue
    ends early.

    Args:
        spans: Predicted sentence spans.
        predicted_ms: Duration used to calculate those spans.
        actual_ms: Measured duration of the written audio file.

    Returns:
        New spans scaled by ``actual_ms / predicted_ms``; the input sequence is
        not mutated.  Invalid/non-positive durations return a shallow list copy.

    """
    if predicted_ms <= 0 or actual_ms <= 0:
        return list(spans)
    factor = actual_ms / predicted_ms
    return [
        SentenceSpan(
            text=s.text,
            char_start=s.char_start,
            char_end=s.char_end,
            start_ms=int(round(s.start_ms * factor)),
            end_ms=min(actual_ms, int(round(s.end_ms * factor))),
        )
        for s in spans
    ]


def write_spans(path: Path, spans: Sequence[SentenceSpan], *, duration_ms: int) -> None:
    """Persist the spans beside the block's audio.

    They are produced by the audio stage and consumed by the render stage,
    which may run much later (``rerender`` re-runs render alone). Keeping them
    as a file next to ``audio.wav`` matches the rest of the block layout and
    needs no schema change.

    Args:
        path: JSON destination beside the block audio.
        spans: Sentence timing records to serialize.
        duration_ms: Audio duration stored alongside the spans.

    Side Effects:
        Creates the parent directory and overwrites ``path`` with UTF-8 JSON.

    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"duration_ms": int(duration_ms), "spans": [s.to_dict() for s in spans]}
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def read_spans(path: Path) -> list[SentenceSpan]:
    """Load spans written by the audio stage; empty when unavailable.

    Projects generated before this existed simply have no file, and the
    caller falls back to character-proportional timing.

    Args:
        path: JSON file written by ``write_spans``.

    Returns:
        Parsed spans, or an empty list when the file is absent, unreadable, or
        contains invalid JSON.  The duration metadata is intentionally ignored.

    """
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return [SentenceSpan.from_dict(item) for item in payload.get("spans") or []]


# Rough moras per character, used only to place a caption break that falls
# inside a sentence — every sentence boundary itself is measured. Kanji carry
# two or three moras ("登録" → トーロク), Latin words are read as katakana and
# run longer than their letters, a digit is usually two ("1" → イチ).
_MORA_WEIGHTS = (
    (lambda c: c in "ぁぃぅぇぉっゃゅょァィゥェォッャュョ", 0.0),
    (lambda c: "ぁ" <= c <= "ゟ" or "゠" <= c <= "ヿ", 1.0),
    (lambda c: c.isdigit(), 2.0),
    (lambda c: c.isascii() and c.isalpha(), 0.6),
    (lambda c: c.isspace(), 0.0),
    (lambda c: c in "、。，．・「」『』（）()！？!?…ー", 0.3),
    (lambda c: "一" <= c <= "鿿", 1.9),
)


def _mora_weight(text: str) -> float:
    """Estimate Japanese speaking weight for an intra-sentence offset.

    Args:
        text: Text fragment whose approximate mora weight is required.

    Returns:
        Positive weighted length used only for proportional interpolation.

    """
    total = 0.0
    for ch in text:
        for matches, weight in _MORA_WEIGHTS:
            if matches(ch):
                total += weight
                break
        else:
            total += 1.0
    return total


def char_time_fn(
    spans: Sequence[SentenceSpan], *, total_ms: int
) -> Callable[[int], float] | None:
    """Build a character-offset-to-time interpolation function.

    Piecewise linear over the measured sentences: exact at every sentence
    boundary. Inside a sentence — where a caption had to break because the
    sentence alone overflows the band — there is nothing measured to anchor
    to, so the position is estimated by mora weight rather than by character
    count, which is the same mistake at smaller scale.

    Args:
        spans: Measured sentence spans.
        total_ms: Full audio duration used for the final clamp.

    Returns:
        A callable accepting a character offset and returning milliseconds, or
        ``None`` when no usable span exists.

    """
    usable = [s for s in spans if s.char_end > s.char_start]
    if not usable:
        return None

    def at(offset: int) -> float:
        if offset <= usable[0].char_start:
            return float(usable[0].start_ms)
        for span in usable:
            if offset >= span.char_end:
                continue
            local = offset - span.char_start
            whole = _mora_weight(span.text)
            ratio = (_mora_weight(span.text[:local]) / whole) if whole else 0.0
            return span.start_ms + ratio * (span.end_ms - span.start_ms)
        return float(min(total_ms, usable[-1].end_ms) if total_ms else usable[-1].end_ms)

    return at
