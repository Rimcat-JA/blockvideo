"""Invalidate dependent stage states while keeping the last completed video."""
from __future__ import annotations

from collections.abc import Iterable

from app.models.block import Block, BlockStatus
from app.models.project import Project
from app.services.generation_plan import dependencies_for_settings


def stale_media_message(blocks: Iterable[Block]) -> str | None:
    """Explain why render-only cannot safely use the project's existing media."""
    stale = [str(block.index) for block in blocks if (
        block.status_audio != BlockStatus.completed or block.status_image != BlockStatus.completed
    )]
    if not stale:
        return None
    return (
        f"ブロック {', '.join(stale[:10])} の音声・画像が最新ではありません。"
        "先に「再生成」（初回は「生成開始」）を実行してから、レンダリングしてください。"
    )


def invalidate_project_settings(project: Project, changed_fields: set[str]) -> None:
    """Mark only stages affected by changed settings pending; never delete files."""
    stages = dependencies_for_settings(changed_fields)
    for block in project.blocks:
        if "audio" in stages:
            block.status_audio = BlockStatus.pending
        if "image" in stages:
            block.status_image = BlockStatus.pending
        if "render" in stages:
            block.status_render = BlockStatus.pending
