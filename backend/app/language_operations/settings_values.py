"""Reject invented numeric settings and dropped explicit speech-speed requests."""
from __future__ import annotations

import re
import unicodedata
from typing import Any, Literal

from app.interpretation.contracts import ClarificationProposal, DialogueContextTurn, OperationProposal
from app.operations.policies import load_policies, settings_base


def settings_value_question(
    text: str, turns: tuple[DialogueContextTurn, ...], relation: Literal["answer", "correction", "dismiss"] | None,
    proposal: OperationProposal,
) -> ClarificationProposal | None:
    """Only veto a proposal; never populate, convert or execute model arguments."""
    view = load_policies().settings_view(proposal.operation_id, proposal.operation_version)
    if view is None:
        return None
    texts = [text]
    if relation == "answer":
        for turn in reversed(turns):
            if turn.settings_saved or turn.status == "completed":
                break
            texts.append(turn.text)
    supplied = unicodedata.normalize("NFKC", "\n".join(texts)).casefold()
    numbers = {float(value) for value in re.findall(r"(?<![\d.])[-+]?\d+(?:\.\d+)?(?![\d.])", supplied)}
    settings: dict[str, Any] = settings_base(view, proposal.arguments) or {}
    invalid = any(type(value) in {int, float} and key != "subtitle_font_size" and value not in numbers
                  for key, value in settings.items())
    for clause in re.split(r"[、,。!！?？\n]", supplied):
        if (re.search(r"速度|話速|話す速さ|読み上げ.{0,3}速さ|スピード", clause)
                and not re.search(r"変えない|変更しない|そのまま|今のまま", clause)
                and re.search(r"\d", clause) and "voicevox_speed_scale" not in settings):
            invalid = True
    if invalid:
        return ClarificationProposal(kind="clarification", missing_fields=["arguments", "intent"],
            question="指定された数値と提案された設定が一致しません。「新しい依頼」で設定名と値をまとめて指定してください。まだ設定は変更していません。")
    return None
