"""Bind supplied readings and merge language additions without losing saved entries."""
from __future__ import annotations

import re
import unicodedata
from typing import Any

from sqlalchemy.orm import Session

from app.interpretation.contracts import ClarificationProposal, OperationProposal
from app.models.project import Project
from app.operations.policies import load_policies, settings_base


def normalized(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    return "".join(chr(ord(char) + 0x60) if "ぁ" <= char <= "ゖ" else char for char in text)


def reading_question(texts: list[str], proposal: OperationProposal) -> ClarificationProposal | None:
    settings = settings_base(load_policies().settings_view(proposal.operation_id, proposal.operation_version),
                             proposal.arguments)
    if settings is None:
        return None
    if "pronunciation_overrides" in settings and not settings["pronunciation_overrides"]:
        # There is no removal operation: an empty list would erase every saved reading.
        return ClarificationProposal(kind="clarification", question="読み方の登録をまとめて消すことはできません。登録・変更する表記と読み方を教えてください。", missing_fields=["arguments"])
    supplied = normalized("\n".join(texts))
    for item in settings.get("pronunciation_overrides", []):
        surface, reading = item["surface"], item["reading"]
        if normalized(surface) not in supplied or normalized(reading) not in supplied:
            return ClarificationProposal(kind="clarification", question="登録する表記と読み方を教えてください。質問に表記がある場合は読み方だけで構いません。", missing_fields=["arguments"])
        accent = item.get("accent")
        # ``normalized`` turns hiragana into katakana, so は/を arrive as ハ/ヲ.
        if accent is not None and not re.search(rf"アクセント\s*(?:ハ|ヲ)?\s*[:：]?\s*{accent}(?!\d)", supplied):
            return ClarificationProposal(kind="clarification", question="アクセントの値を指定するか、自動と指定してください。", missing_fields=["arguments"])
    return None


def merged_arguments(db: Session, project_id: int, proposal: OperationProposal) -> dict[str, Any]:
    arguments = dict(proposal.arguments)
    view = load_policies().settings_view(proposal.operation_id, proposal.operation_version)
    nested = view is not None and bool(view.settings_at)
    settings = dict(arguments[view.settings_at]) if nested else arguments
    if "pronunciation_overrides" in settings and not settings["pronunciation_overrides"]:
        # Never let an empty proposal (e.g. an unattended YOLO run) wipe saved readings.
        settings.pop("pronunciation_overrides")
    additions = settings.get("pronunciation_overrides") if view is not None and view.settings_at is not None else None
    if additions:
        project = db.get(Project, project_id)
        existing = project.pronunciation_overrides if project else []
        by_surface = {item["surface"]: dict(item) for item in existing or []}
        by_surface.update({item["surface"]: dict(item) for item in additions})
        settings["pronunciation_overrides"] = list(by_surface.values())
    if nested:
        arguments["settings"] = settings
    return arguments
