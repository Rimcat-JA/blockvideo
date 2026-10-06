"""Project-local readings and accent edits, without changing the engine dictionary.

Matching is literal and longest-first. Latin identifiers and Japanese compounds
are bounded so an override for ``API`` cannot accidentally rewrite ``APIs``.
The replacement text is used only for speech; source caption offsets never move.
"""
from __future__ import annotations

import copy
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

from app.core.provider_errors import ProviderError

if TYPE_CHECKING:
    from app.providers.voicevox import VoicevoxClient


@dataclass(frozen=True)
class Pronunciation:
    surface: str
    reading: str
    accent: int | None = None


def reading_mora_count(reading: str) -> int:
    """Count katakana moras; small vowels/ya-yu-yo join the preceding mora."""
    return sum(ch not in "ァィゥェォャュョヮ" for ch in reading)


def normalize_overrides(
    overrides: Sequence[Mapping[str, Any] | Pronunciation] | None,
) -> list[Pronunciation]:
    """Validate nonempty, unique surfaces, katakana readings and accent bounds."""
    normalized: list[Pronunciation] = []
    seen: set[str] = set()
    for item in overrides or []:
        if isinstance(item, Pronunciation):
            value = item
        else:
            value = Pronunciation(
                surface=str(item.get("surface", "")).strip(),
                reading=str(item.get("reading", "")).strip(),
                accent=item.get("accent"),
            )
        if not value.surface or any(ch in value.surface for ch in "。！？\n\r"):
            raise ProviderError("読み方の表記には空文字や文の区切りを指定できません", safe=True)
        if value.surface in seen:
            raise ProviderError(f"読み方の表記が重複しています: {value.surface}", safe=True)
        if (
            not re.fullmatch(r"[ァ-ヴー]+", value.reading)
            or value.reading[0] in "ァィゥェォャュョヮー"
        ):
            raise ProviderError("読み方は全角カタカナで指定してください", safe=True)
        count = reading_mora_count(value.reading)
        if value.accent is not None and (
            isinstance(value.accent, bool)
            or not isinstance(value.accent, int)
            or not 0 <= value.accent <= count
        ):
            raise ProviderError(
                f"アクセントは0（平板）から読みのモーラ数{count}までで指定してください",
                safe=True,
            )
        seen.add(value.surface)
        normalized.append(value)
    return normalized


def _word_group(ch: str) -> str | None:
    if ch.isascii() and (ch.isalnum() or ch == "_"):
        return "latin"
    if "ァ" <= ch <= "ヿ" or ch == "ー":
        return "katakana"
    if "一" <= ch <= "鿿":
        return "kanji"
    return None


def pronunciation_segments(
    text: str, overrides: Sequence[Pronunciation]
) -> list[tuple[str, Pronunciation | None]]:
    """Return untouched fragments and matched readings, without cascading edits."""
    if not overrides:
        return [(text, None)]
    by_surface = {item.surface: item for item in overrides}
    groups = {"latin": "A-Za-z0-9_", "katakana": "ァ-ヿー", "kanji": "一-鿿"}

    def bounded(surface: str) -> str:
        left, right = _word_group(surface[0]), _word_group(surface[-1])
        if surface[-1] in "!?" and len(surface) > 1 and _word_group(surface[-2]) == "latin":
            right = "latin"
        return (
            (f"(?<![{groups[left]}])" if left else "")
            + re.escape(surface)
            + (f"(?![{groups[right]}])" if right else "")
        )

    # Bound each alternative before choosing the longest match. If "API key"
    # is rejected inside "API keys", the valid shorter "API" still matches.
    pattern = re.compile("|".join(bounded(key) for key in sorted(by_surface, key=len, reverse=True)))
    segments: list[tuple[str, Pronunciation | None]] = []
    cursor = 0
    for match in pattern.finditer(text):
        surface = match.group()
        if cursor < match.start():
            segments.append((text[cursor:match.start()], None))
        item = by_surface[surface]
        segments.append((item.reading, item))
        cursor = match.end()
    if cursor < len(text):
        segments.append((text[cursor:], None))
    return segments


def apply_readings(text: str, overrides: Sequence[Pronunciation]) -> str:
    return "".join(fragment for fragment, _ in pronunciation_segments(text, overrides))


async def pronunciation_query(
    text: str,
    speaker_id: int,
    client: VoicevoxClient,
    overrides: Sequence[Pronunciation],
) -> dict[str, Any]:
    """Build a reading-aware query and recalculate pitch after explicit accent edits.

    Reading-only sentences retain the engine's complete-sentence analysis. When
    an accent is explicitly set, matched terms form their own accent phrases;
    ``/mora_data`` recalculates lengths and pitches in the assembled context.
    The engine's ``update_pitch`` uses ``accent - 1`` as its final high index:
    flat 0 and the final mora select the same point. Use the final mora to also
    support one-mora readings. This controls a phrase rather than applying the
    global dictionary's part-of-speech/particle accent rules.
    """
    segments = pronunciation_segments(text, overrides)
    spoken = "".join(fragment for fragment, _ in segments)
    if not any(item is not None and item.accent is not None for _, item in segments):
        return await client.audio_query(spoken, speaker_id)

    base: dict[str, Any] | None = None
    phrases: list[dict[str, Any]] = []
    for fragment, item in segments:
        if not fragment.strip():
            continue
        # Punctuation on its own has no moras. Preserve its pause/question
        # behavior on the preceding phrase instead of losing it in an empty query.
        if all(ch.isspace() or ch in "、。，．！？!?・…「」『』（）()" for ch in fragment):
            if phrases:
                if any(ch in fragment for ch in "！？!?"):
                    phrases[-1]["is_interrogative"] = any(ch in fragment for ch in "？?")
                if any(ch in fragment for ch in "、，。．"):
                    phrases[-1]["pause_mora"] = {
                        "text": "、", "consonant": None, "consonant_length": None,
                        "vowel": "pau", "vowel_length": 0.3, "pitch": 0.0,
                    }
            continue
        query = await client.audio_query(fragment, speaker_id)
        if base is None:
            base = copy.deepcopy(query)
        own = copy.deepcopy(query.get("accent_phrases") or [])
        if item is not None and item.accent is not None:
            moras = [mora for phrase in own for mora in phrase.get("moras") or []]
            if not moras or item.accent > len(moras):
                raise ProviderError(f"読み方のアクセントを適用できません: {item.surface}", safe=True)
            own = [{
                "moras": moras,
                "accent": item.accent or len(moras),
                "pause_mora": None,
                "is_interrogative": False,
            }]
        phrases.extend(own)
    if base is None or not phrases:
        raise ProviderError("読み方を適用した音声クエリが空です", safe=True)
    base["accent_phrases"] = await client.mora_data(phrases, speaker_id)
    # This metadata is read-only to the engine; the fragments' old kana would
    # misrepresent the complete sentence if left here.
    base["kana"] = None
    return base
