"""Tell a guessed (missing) value from one that contradicts what the user wrote.

Unattended (YOLO) requests may guess values the request leaves out, but never
override or drop a number or reference the user stated. Normal requests use the
same check as a clarifying guard. Every statement is read within its own clause
(see ``clauses``), so a negation or direction in one clause never applies to
another. This module only reports a conflict; it never fills or converts arguments.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from app.interpretation.contracts import OperationProposal
from app.language_operations.clauses import clauses, normalized
from app.operations.policies import OperationPolicies, load_policies, settings_base, settings_values

_PX = r"(?<![\d.])([0-9]+)\s*(?:px|ピクセル)(?![a-z])"
_NUMBER = r"(?<![\d.])[-+]?\d+(?:\.\d+)?(?![\d.])"
_UP = r"大き|上げ|増や|拡大"
_DOWN = r"小さ|下げ|減ら|縮小"
_SUBTITLE = r"字幕|フォント|文字の大きさ|文字サイズ"
_NEGATION = r"ないで|なくて|しない|ません"
# A clause that only checks a result ("64pxになったか確認して") states no new value.
_CHECK_ONLY = r"なったか|なっているか|なったこと"
# "いや" / "やっぱり" corrects the value stated just before.
_CORRECTION = r"^(?:いや|やっぱり|訂正して|訂正)\s*$|^(?:いや|やっぱり)"
# A number carried over from the next clause must not count something else ("もう1回").
_COUNTED = r"\s*(?:回|件|個|つ|人|分|版|日|時|本|枚目)"
# "動画3ではなく動画4に" names 3 only to exclude it.
_EXCLUDED = r"\d+\s*(?:番)?\s*(?:では|じゃ)なく(?:て)?"


@dataclass
class _Stated:
    """What the user wrote, clause by clause."""

    pixels: set[int] = field(default_factory=set)
    targets: set[int] = field(default_factory=set)
    negated_pixels: set[int] = field(default_factory=set)
    up: bool = False
    down: bool = False
    settings: dict[str, list[float]] = field(default_factory=dict)
    negated: dict[str, set[float]] = field(default_factory=dict)


def _record(stated: _Stated, name: str, value: float, negated: bool) -> None:
    if negated:
        stated.negated.setdefault(name, set()).add(value)
    else:
        stated.settings.setdefault(name, []).append(value)


def _read(texts: list[str], policies: OperationPolicies) -> _Stated:
    stated = _Stated()
    # References ("動画1", "ジョブ7", "第2版") are never setting values.
    references = [pattern for kind in policies.references.values() for pattern in kind.patterns]
    pending: list[str] = []  # a keyword whose number comes in the next clause ("音量を調整して0.8倍にして")
    subtitle_context = False  # "字幕を2px変えて小さくして": the direction follows in the next clause
    correcting = False
    for clause in (part for text in texts for part in clauses(text)):
        negated = bool(re.search(_NEGATION, clause))
        if re.search(_CORRECTION, clause):
            correcting = True
            if re.fullmatch(_CORRECTION, clause):
                continue
        if re.search(_CHECK_ONLY, clause):
            pending, subtitle_context = [], False
            continue
        plain = re.sub(_PX, " ", clause)
        for pattern in references:
            plain = re.sub(pattern, " ", plain, flags=re.IGNORECASE)
        hits: list[tuple[int, int, str, str | None]] = []
        for name, patterns in policies.setting_keywords.items():
            for pattern in patterns:
                hits.extend((match.start(), match.end(), name, match.group(1) if match.groups() else None)
                            for match in re.finditer(pattern, plain))
        hits.sort()
        # Pixel sizes count only where the clause is about subtitles, or is just a size ("64px").
        about_subtitle = bool(re.search(_SUBTITLE, clause)) or (subtitle_context and not hits)
        if about_subtitle or re.fullmatch(r"\s*" + _PX + r"\s*(?:で|に)?\s*(?:お願いします|して)?\s*", clause):
            pixels = {int(value) for value in re.findall(_PX, clause)}
            if negated:
                stated.negated_pixels |= pixels
            else:
                stated.pixels |= pixels
                stated.targets |= {int(value) for value in re.findall(_PX + r"\s*(?:に|へ|で)", clause)}
                stated.up |= bool(re.search(_UP, clause))
                stated.down |= bool(re.search(_DOWN, clause))
        subtitle_context = about_subtitle
        if not hits:
            numbers = [match.group() for match in re.finditer(_NUMBER, plain)
                       if not re.match(_COUNTED, plain[match.end():])]
            if numbers and len(pending) == 1:
                _record(stated, pending[0], float(numbers[-1]), negated)
            pending = []
            continue
        pending = []
        before = {name: len(values) for name, values in stated.settings.items()} if correcting else {}
        for index, (_, end, name, captured) in enumerate(hits):
            if captured is not None:
                value = float(captured)
            else:
                limit = hits[index + 1][0] if index + 1 < len(hits) else len(plain)
                numbers = re.findall(_NUMBER, plain[end:limit])
                if not numbers:
                    pending = [name] if index + 1 == len(hits) else pending
                    continue
                value = float(numbers[-1])  # "1.2倍から1.5倍に" asks for the last one
            _record(stated, name, value, negated)
        if correcting:
            # "話速は1.2倍、いや、話速は1.5倍にして": a corrected value replaces the earlier one;
            # a correction without a new number keeps the earlier value.
            for name, values in stated.settings.items():
                if len(values) > before.get(name, 0):
                    stated.settings[name] = values[before.get(name, 0):]
            correcting = False
    return stated


def _references(text: str, kind_patterns: tuple[str, ...]) -> set[int]:
    chosen = re.sub(_EXCLUDED, " ", normalized(text))
    return {int(value) for pattern in kind_patterns for value in re.findall(pattern, chosen, re.IGNORECASE)}


def _reference_conflict(texts: list[str], proposal: OperationProposal, policies: OperationPolicies) -> bool:
    binding = policies.get(proposal.operation_id).reference
    if binding is None:
        return False
    patterns = policies.references[binding.kind].patterns
    # An answer's own reference settles an ambiguity the earlier request left open.
    stated = next((found for found in (_references(text, patterns) for text in texts) if found), set())
    return bool(stated) and stated != {proposal.arguments.get(binding.argument)}


def _subtitle_conflict(stated: _Stated, values: dict[str, Any], current_subtitle: int | None) -> bool:
    absolute, delta = values.get("subtitle_font_size"), values.get("subtitle_font_size_delta")
    reached = absolute if absolute is not None else (
        current_subtitle + delta if delta is not None and current_subtitle is not None else None)
    if reached is not None and reached in stated.negated_pixels:
        return True  # "字幕は64pxにしないで" also forbids reaching 64px by a delta
    if absolute is not None and ((stated.targets and absolute not in stated.targets)
                                 or (stated.pixels and absolute not in stated.pixels
                                     and not (stated.up or stated.down))):
        return True
    if delta is not None:
        if (delta > 0 and stated.down and not stated.up) or (delta < 0 and stated.up and not stated.down):
            return True
        if stated.pixels and (abs(delta) not in stated.pixels or not (stated.up or stated.down)
                              or bool(stated.targets)):
            return True
    return False


def _settings_conflict(stated: _Stated, proposal: OperationProposal, policies: OperationPolicies,
                       require_all: bool, current_subtitle: int | None) -> bool:
    view = policies.settings_view(proposal.operation_id, proposal.operation_version)
    if view is None:
        return False
    values = settings_values(view, proposal.arguments)
    if _subtitle_conflict(stated, values, current_subtitle):
        return True
    if require_all and stated.pixels and values.get("subtitle_font_size") is None \
            and values.get("subtitle_font_size_delta") is None:
        return True
    proposed = settings_base(view, proposal.arguments) or {}
    for name, current in proposed.items():
        if type(current) in {int, float} and float(current) in stated.negated.get(name, set()):
            return True  # sets exactly what the user said not to
    for name, allowed in stated.settings.items():
        current = proposed.get(name)
        if current is None:
            if require_all:
                return True  # a stated value that is dropped is never a guess
        elif type(current) not in {int, float} or float(current) not in set(allowed):
            return True
    return False


def explicit_conflict(texts: list[str], proposal: OperationProposal, *, require_all: bool = True,
                      current_subtitle: int | None = None) -> bool:
    """True when the proposal contradicts (or, for a whole request, drops) a stated value or reference.

    ``texts`` lists the current request first, then any unsaved request it answers.
    """
    policies = load_policies()
    return (_reference_conflict(texts, proposal, policies)
            or _settings_conflict(_read(texts, policies), proposal, policies, require_all, current_subtitle))


def plan_drops_stated(texts: list[str], steps: list[OperationProposal]) -> bool:
    """True when the plan leaves out a stated setting value or subtitle size.

    A plan that restores a saved settings version may reach stated values through
    the restore, so such plans are not checked; restoring a video changes no settings.
    """
    policies = load_policies()
    if any(policies.get(step.operation_id).reference is not None
           and policies.get(step.operation_id).reference.kind == "revision" for step in steps):
        return False
    stated = _read(texts, policies)
    carried: dict[str, set[float]] = {}
    subtitle = False
    for step in steps:
        view = policies.settings_view(step.operation_id, step.operation_version)
        if view is None:
            continue
        values = settings_values(view, step.arguments)
        subtitle |= values.get("subtitle_font_size") is not None or values.get("subtitle_font_size_delta") is not None
        for name, value in (settings_base(view, step.arguments) or {}).items():
            if type(value) in {int, float}:
                carried.setdefault(name, set()).add(float(value))
    if stated.pixels and not subtitle:
        return True
    # Every stated value must appear: "話速を1.2倍に…話速を1.5倍に" needs both steps.
    return any(not set(allowed) <= carried.get(name, set()) for name, allowed in stated.settings.items())
