"""Conservative, local alignment between narration and visible slide text.

Only literal labels that occur both on the slide and in the sentence can be
highlighted. No LLM call, translation, guessed identifier, or hidden node ID is
used. The sorted result also serves as a small cache key for rendered variants.
"""
from __future__ import annotations

import re
from typing import Any


_IDENTIFIER = r"[A-Za-z_][A-Za-z0-9_]*(?:[-!?][A-Za-z0-9_]*)*"
_JAPANESE = r"[\u3041-\u3096\u30a1-\u30fa\u30fc\u3400-\u9fff]{2,}"
_TOKEN = re.compile(f"{_IDENTIFIER}|{_JAPANESE}")
_ASCII_WORD = re.compile(r"[A-Za-z0-9_!?-]")
_QUOTED = re.compile(r"[「『`\"]([^「」『』`\"\n]{1,64})[」』`\"]")


def literal_ranges(text: str, term: str) -> tuple[tuple[int, int], ...]:
    """Return exact occurrences, excluding substrings of ASCII identifiers."""
    if not term:
        return ()
    ranges = []
    for match in re.finditer(re.escape(term), text):
        start, end = match.span()
        if _ASCII_WORD.fullmatch(term[0]) and start and _ASCII_WORD.fullmatch(text[start - 1]):
            continue
        if _ASCII_WORD.fullmatch(term[-1]) and end < len(text) and _ASCII_WORD.fullmatch(text[end]):
            continue
        ranges.append((start, end))
    return tuple(ranges)


def matching_terms(visible_text: str, sentence: str | None) -> tuple[str, ...]:
    """Select visible identifiers/labels mentioned literally by one sentence.

    Quoted narration also supports labels embedded in longer Japanese text.
    Numbers and isolated Japanese syllables are excluded because ordinary
    prose would otherwise highlight unrelated diagram cells accidentally.
    """
    if not sentence or not visible_text:
        return ()
    candidates = {match.group() for match in _TOKEN.finditer(visible_text)}
    candidates.update(match.group(1) for match in _QUOTED.finditer(sentence))
    terms = [term for term in candidates if literal_ranges(sentence, term)
             and literal_ranges(visible_text, term) and not term.isspace()]
    # Longest labels win over nested matches; sorted output makes cache keys
    # independent of the order in which the narrator mentions the labels.
    selected: list[str] = []
    for term in sorted(terms, key=lambda value: (-len(value), value)):
        if not any(literal_ranges(longer, term) for longer in selected):
            selected.append(term)
    return tuple(sorted(selected))


def focus_terms(plan: dict[str, Any], sentence: str) -> tuple[str, ...]:
    """Return a deterministic focus key for a supported visible plan body.

    Unsupported image/graph formats keep their static image. Headings alone
    never trigger a variant: the cue must refer to content the viewer can find.
    """
    field = {"verbatim_slide": "verbatim", "code_slide": "code",
             "text_slide": "visual_summary"}.get(plan.get("visual_type") or "text_slide")
    return matching_terms(str(plan.get(field) or ""), sentence) if field else ()
