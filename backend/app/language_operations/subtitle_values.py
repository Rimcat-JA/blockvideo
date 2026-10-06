"""Conservative value evidence checks, never an intent router or value resolver."""
from __future__ import annotations

import re
import unicodedata
from typing import Literal

from app.interpretation.contracts import ClarificationProposal, DialogueContextTurn, OperationProposal
from app.operations.policies import load_policies, settings_values

_PX = r"(?<![\d.])([0-9]+)\s*(?:px|ピクセル)(?![a-z])"
_UP = r"大き|上げ|増や|増加|拡大"
_DOWN = r"小さ|下げ|減ら|減少|縮小"
_FONT = r"字幕|文字|フォント"


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text).casefold()


def _absolute_values(text: str) -> set[int]:
    return {int(value) for value in re.findall(_PX, text)}


def _relative_is_supplied(text: str, delta: int) -> bool:
    """Require the quantity and direction together, not a number in another clause."""
    direction = _UP if delta > 0 else _DOWN
    opposite = _DOWN if delta > 0 else _UP
    if delta == 0:
        return False
    for clause in re.split(r"[、,。!！?？\n]", text):
        if not re.search(direction, clause) or re.search(opposite, clause):
            continue
        # Other settings' amounts cannot justify a subtitle change.
        if not re.search(_FONT, clause) and re.search(r"速度|音量|声|話速", clause):
            continue
        if abs(delta) == 2 and re.search(r"少し(?:だけ|だけを|もう少し)?[^\d]{0,12}(?:" + direction + ")", clause):
            return True
        if re.search(rf"(?<![\d.]){abs(delta)}\s*(?:px|ピクセル)?\s*(?:だけ|ほど|分)?\s*(?:{direction})", clause):
            return True
    return False


def subtitle_question(
    text: str, turns: tuple[DialogueContextTurn, ...], relation: Literal["answer", "correction", "dismiss"] | None,
    proposal: OperationProposal,
) -> ClarificationProposal | None:
    """Unsupported phrasing asks, rather than guessing; saved text never supplies a new value."""
    view = load_policies().settings_view(proposal.operation_id, proposal.operation_version)
    if view is None:
        return None
    values = settings_values(view, proposal.arguments)
    absolute = values.get("subtitle_font_size")
    delta = values.get("subtitle_font_size_delta")
    supplied = _normalize(text)
    pending: list[str] = []
    if relation == "answer":
        for turn in reversed(turns):
            if turn.settings_saved or turn.status == "completed":
                break
            pending.append(_normalize(turn.text))
    previous = "\n".join(reversed(pending))
    values = _absolute_values(supplied)
    # A short answer to a size question may omit the px suffix.
    if relation == "answer" and turns and re.search(r"px|ピクセル|サイズ|大きさ", _normalize(turns[-1].question or "")):
        short = re.fullmatch(r"([0-9]+)(?:で|に)?(?:お願いします|して)?[。.!！]?", supplied.strip())
        if short:
            values.add(int(short[1]))
    if not values:
        values = _absolute_values(previous)
    evidence = supplied + "\n" + previous
    if (relation == "answer" and previous and re.search(_FONT, previous)
            and re.fullmatch(r"(?:少し|[0-9]+\s*(?:px|ピクセル))(?:だけ)?[。!！]?", supplied.strip())):
        evidence += "\n" + supplied + previous
    invalid = absolute is not None and (values != {absolute} or delta is not None)
    invalid |= delta is not None and not _relative_is_supplied(evidence, delta)
    # Do not partially save an answer that drops a pending, explicit subtitle value.
    missing = absolute is None and delta is None and bool(values) and bool(re.search(_FONT, evidence))
    if invalid or missing:
        return ClarificationProposal(kind="clarification", missing_fields=["arguments"],
            question="字幕サイズの希望値（例：64px）か増減量（例：2px小さく）を指定してください。ほかの変更もまとめて保存するため、まだ設定は変更していません。")
    return None
