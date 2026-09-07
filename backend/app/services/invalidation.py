"""Invalidate dependent stage states while keeping the last completed video."""
from __future__ import annotations

from collections.abc import Iterable

from app.models.block import Block, BlockStatus
from app.models.project import Project


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
    audio = any(field.startswith('voicevox_') for field in changed_fields) or bool(
        changed_fields & {'narration_pacing_mode', 'narration_sentence_pause_seconds',
                          'pronunciation_overrides', 'max_slides_per_block'}
    )
    image = bool(changed_fields & {'max_slides_per_block', 'subtitle_enabled'})
    render = audio or image or bool(changed_fields & {
        'visual_focus_enabled', 'subtitle_mode', 'subtitle_font_size',
        'subtitle_position', 'subtitle_text_color', 'subtitle_outline_color',
        'subtitle_background', 'subtitle_max_chars_per_line', 'pre_margin_seconds',
        'post_margin_seconds', 'min_display_seconds',
    })
    for block in project.blocks:
        if audio:
            block.status_audio = BlockStatus.pending
        if image:
            block.status_image = BlockStatus.pending
        if render:
            block.status_render = BlockStatus.pending
