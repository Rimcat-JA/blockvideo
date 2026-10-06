"""Resolve application context without allowing the model to select a target."""
from __future__ import annotations

import re

from sqlalchemy.orm import Session

from app.interpretation.contracts import MinimalState
from app.language_operations.contracts import LanguageError, LanguageInput
from app.models.project import Project

_PROJECT_REFERENCE = re.compile(r"(?:プロジェクト|project)\s*(?:ID\s*[:：]?\s*|[#＃第])?\s*(\d+)", re.IGNORECASE)


def resolve_context(db: Session, request: LanguageInput) -> MinimalState:
    """Only explicit/selected IDs are supported; names are not searched."""
    target_id = request.target.resolved_id
    if target_id is None:
        raise LanguageError("target_required", "操作するプロジェクトを選択してください。")
    mentioned_ids = {int(match) for match in _PROJECT_REFERENCE.findall(request.text)}
    if mentioned_ids and mentioned_ids != {target_id}:
        raise LanguageError("target_conflict", "文章のプロジェクト番号と選択中の対象が一致しません。")
    project = db.get(Project, target_id)
    if project is None:
        raise LanguageError("target_not_found", "対象のプロジェクトが見つかりません。", 404)
    if request.base_revision is not None and request.base_revision != project.revision:
        raise LanguageError("stale_state", "設定が変更されています。最新の状態で新しい要求を送ってください。")
    return MinimalState(selected_project_id=project.id, revision=project.revision,
                        subtitle_font_size=project.subtitle_font_size, status=project.status.value)
