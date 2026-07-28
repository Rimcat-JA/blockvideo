"""End-to-end pipeline orchestrator.

Each stage is independent and idempotent: it inspects ``content_hash`` and
the artifact status to decide whether to skip or rerun.
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.logging import log
from app.models.block import Block, BlockStatus, VisualType
from app.models.project import Project, ProjectStatus
from app.providers.voicevox import VoicevoxSettings
from app.services import ffmpeg_runner, image_renderer, subtitles
from app.services.hashing import short_hash
from app.services.paths import (
    block_audio_path,
    block_image_path,
    block_narration_path,
    block_subtitle_path,
    block_video_path,
    concat_list_path,
    ensure_project_layout,
    output_video_path,
    project_dir,
    project_json_path,
    project_subtitle_path,
    relpath_for_db,
    timeline_json_path,
)
from app.services.provider_factory import (
    ProviderBundle,
    build_providers_for_project,
    build_voicevox_settings,
)
from app.services.splitter import (
    normalize_kept,
    repair_narration_gaps,
    split_script,
)
from app.services import narration
from app.services.voice import compute_display_duration_ms, synthesize_block
from app.services.visual_planner import (
    authored_plan,
    build_slide_sequence,
    extract_authored_slide,
    generate_global_style,
    slide_alignment_issues,
    generate_visual_plan,
)


ProgressCallback = Callable[[str, float, str | None], Awaitable[None]]


@dataclass
class StageContext:
    """Shared project, provider, settings, and cancellation state for stages."""

    project: Project
    settings: Settings
    bundle: ProviderBundle
    voicevox_settings: VoicevoxSettings
    progress_cb: ProgressCallback | None = None
    is_cancelled: Callable[[], bool] = lambda: False

    async def report(self, stage: str, progress: float, message: str | None = None) -> None:
        """Forward stage progress to the optional worker callback."""
        if self.progress_cb:
            await self.progress_cb(stage, progress, message)


async def ensure_global_style(ctx: StageContext, db: Session) -> str:
    """Load a persisted global style or generate and save it once."""
    if ctx.project.global_visual_style:
        return ctx.project.global_visual_style
    style = await generate_global_style(ctx.bundle.llm, project_title=ctx.project.title)
    ctx.project.global_visual_style = style
    db.add(ctx.project)
    db.commit()
    db.refresh(ctx.project)
    return style


async def run_split_stage(ctx: StageContext, db: Session) -> list[Block]:
    """Split the source script, repair narration, and synchronize block rows."""
    script = normalize_kept(ctx.project.source_script)
    result = await split_script(script, ctx.bundle.llm, ctx.settings)
    log.info(
        "split result blocks={count} fallback={fallback} attempts={attempts}",
        count=len(result.blocks),
        fallback=result.used_fallback,
        attempts=result.attempts,
    )
    if result.issues:
        # Without these the deterministic fallback looks like a clean success
        # while silently producing mid-word splits.
        log.warning("split issues: {issues}", issues=" | ".join(result.issues))

    if ctx.settings.narration_repair_enabled and not ctx.bundle.use_fake:
        repaired = await repair_narration_gaps(
            result.blocks, ctx.bundle.planner, ctx.settings
        )
        if repaired:
            log.info(
                "ナレーション修復: {n}/{total} ブロック",
                n=repaired, total=len(result.blocks),
            )

    # wipe blocks that don't correspond to the new split (cache invalidation).
    existing = {b.index: b for b in ctx.project.blocks}
    new_blocks: list[Block] = []
    for i, sb in enumerate(result.blocks):
        block = existing.get(i)
        if block is None:
            block = Block(
                project_id=ctx.project.id,
                index=i,
                source_text=sb.source_text,
                tts_text=sb.tts_text,
            )
            db.add(block)
        else:
            block.source_text = sb.source_text
            block.tts_text = sb.tts_text
        block.status_split = (
            BlockStatus.completed if not result.used_fallback else BlockStatus.completed
        )
        new_blocks.append(block)
    # delete blocks beyond the new count (cache invalidation by deletion)
    for i, block in list(existing.items()):
        if i >= len(result.blocks):
            db.delete(block)
    db.commit()
    for b in new_blocks:
        db.refresh(b)
    return new_blocks


async def run_visual_plan_stage(ctx: StageContext, db: Session) -> int:
    """Plan every block's visual.

    One LLM call per block, so this dominates wall-clock on a long script.
    The calls are independent, so they run concurrently under a bounded
    semaphore; results are applied to the session afterwards, in index order,
    because the SQLAlchemy Session is not safe to touch from concurrent tasks.
    """
    import asyncio

    style = await ensure_global_style(ctx, db)
    blocks = sorted(ctx.project.blocks, key=lambda b: b.index)
    done = sum(1 for b in blocks if b.status_visual_plan == BlockStatus.completed)
    todo = [b for b in blocks if b.status_visual_plan != BlockStatus.completed]
    if not todo:
        return done

    limit = max(1, int(getattr(ctx.settings, "visual_plan_concurrency", 6)))
    sem = asyncio.Semaphore(limit)
    provider = ctx.bundle.planner

    async def plan_one(block: Block):
        if ctx.is_cancelled():
            return block, None
        authored = extract_authored_slide(block.source_text)
        if authored:
            # The author drew this slide; there is nothing to design and no
            # call to spend. It is drawn exactly as written, so a box the
            # script left ragged reaches the screen ragged — say which line.
            ragged = slide_alignment_issues(authored[1])
            if ragged:
                log.warning(
                    "block={idx} スライドの枠が揃っていません: {issues}",
                    idx=block.index, issues=" / ".join(ragged[:3]),
                )
            return block, authored_plan(*authored)
        async with sem:
            if ctx.is_cancelled():
                return block, None
            try:
                plan = await generate_visual_plan(
                    provider,
                    block_index=block.index,
                    tts_text=block.tts_text,
                    source_text=block.source_text,
                    global_style=style,
                )
                return block, plan
            except Exception as exc:  # noqa: BLE001 - recorded per block below
                return block, exc

    results = await asyncio.gather(*(plan_one(b) for b in todo))

    # The planner is meant to be the exception, not the rule: a script that
    # draws its own slides never reaches it, and every block that does is one
    # where a model had to invent the picture. Say so, with the block numbers,
    # so a script missing its slides is visible rather than merely expensive.
    fell_back = [
        b.index for b, outcome in results
        if outcome is not None and not isinstance(outcome, Exception)
        and outcome.visual_type != VisualType.verbatim_slide
    ]
    if fell_back:
        log.warning(
            "台本にスライド指定が無く、モデルが図を設計したブロック: {n}/{total} {idx}",
            n=len(fell_back), total=len(todo),
            idx=fell_back[:20],
        )
    else:
        log.info("全 {n} ブロックが台本のスライド指定を使用 (LLM呼び出しなし)", n=len(todo))

    for block, outcome in results:
        if outcome is None:
            continue
        if isinstance(outcome, Exception):
            block.status_visual_plan = BlockStatus.failed
            block.error_message = f"{outcome.__class__.__name__}: {str(outcome)[:200]}"
            continue
        plan = outcome
        block.visual_type = VisualType(plan.visual_type.value)
        block.visual_plan_json = plan.model_dump()
        if plan.visual_type == VisualType.ai_image:
            block.image_prompt = plan.image_prompt
        block.content_hash = short_hash(
            block.source_text,
            block.tts_text,
            plan.model_dump_json(),
            style,
        )
        block.status_visual_plan = BlockStatus.completed
        block.error_message = None
        done += 1
    db.commit()

    planned = sum(1 for b in blocks if b.status_visual_plan == BlockStatus.completed)
    if planned == 0 and not ctx.is_cancelled():
        # Every block fell back to a bare text slide. Downstream stages would
        # happily render and encode that into a finished-looking but useless
        # video, so fail loudly instead of shipping it.
        first_error = next(
            (b.error_message for b in blocks if b.error_message), "原因不明"
        )
        raise RuntimeError(f"画面構成の生成が全ブロックで失敗しました: {first_error}")
    if planned < len(blocks):
        log.warning(
            "visual plan 部分失敗 planned={planned}/{total}",
            planned=planned, total=len(blocks),
        )
    return done


async def run_image_stage(ctx: StageContext, db: Session) -> int:
    """Render every pending block image while honoring cancellation."""
    count = 0
    style = ctx.project.global_visual_style or ""
    for block in sorted(ctx.project.blocks, key=lambda b: b.index):
        if ctx.is_cancelled():
            break
        if block.status_image == BlockStatus.completed:
            count += 1
            continue
        await _render_block_image(ctx, block, style, db)
        count += 1
    return count


async def _render_block_image(
    ctx: StageContext, block: Block, style: str, db: Session
) -> None:
    """Render one block's primary and additional slide images."""
    ensure_project_layout(ctx.project.id)
    plan = block.visual_plan_json or {}
    output = block_image_path(ctx.project.id, block.index)
    settings = ctx.settings
    block.status_image = BlockStatus.running
    try:
        if block.visual_type == VisualType.ai_image and ctx.bundle.image is not None:
            prompt = (
                f"{plan.get('heading') or '解説画像'}. "
                f"{plan.get('visual_summary') or block.tts_text[:200]}. "
                f"Style: {style}. No text in the image. Abstract symbolic."
            )
            await ctx.bundle.image.generate_image(
                prompt, settings.output_width, settings.output_height, output
            )
        else:
            # Render at the size of the *slide region*, not the whole frame.
            # With subtitles on, ffmpeg fits the slide into
            # height - subtitle_band_height; a full-frame 16:9 image would be
            # letterboxed inside that wider region, shrinking the diagram and
            # adding side bars for nothing.
            slide_height = settings.output_height
            if ctx.project.subtitle_enabled and settings.subtitle_band_height > 0:
                slide_height = max(120, settings.output_height - settings.subtitle_band_height)
            # Render only as many slides as the project will actually show.
            # The cap used to be applied at render time, which meant every
            # extra listing was still drawn and then discarded.
            sequence = build_slide_sequence(
                plan,
                block.source_text,
                max_slides=ctx.project.max_slides_per_block,
            )
            for slot, slide_plan in enumerate(sequence):
                image_renderer.render_visual_plan(
                    slide_plan,
                    block_image_path(ctx.project.id, block.index, slot),
                    width=settings.output_width,
                    height=slide_height,
                    fallback_summary=block.tts_text,
                )
            # Any slides left over from a previous, longer sequence would
            # otherwise be picked up by the render stage.
            for stale in range(len(sequence), len(sequence) + 8):
                block_image_path(ctx.project.id, block.index, stale).unlink(
                    missing_ok=True
                )
        block.image_path = relpath_for_db(output)
        block.status_image = BlockStatus.completed
        block.error_message = None
        db.commit()
    except Exception as exc:
        block.status_image = BlockStatus.failed
        block.error_message = f"{exc.__class__.__name__}: {str(exc)[:200]}"
        db.commit()
        raise


async def run_audio_stage(ctx: StageContext, db: Session) -> int:
    """Synthesize every pending block and persist durations and spans."""
    count = 0
    ensure_project_layout(ctx.project.id)
    for block in sorted(ctx.project.blocks, key=lambda b: b.index):
        if ctx.is_cancelled():
            break
        if block.status_audio == BlockStatus.completed and block.audio_path:
            count += 1
            continue
        await _render_block_audio(ctx, block, db)
        count += 1
    return count


async def _render_block_audio(
    ctx: StageContext, block: Block, db: Session
) -> None:
    """Synthesize one block and save its audio timing metadata."""
    output = block_audio_path(ctx.project.id, block.index)
    block.status_audio = BlockStatus.running
    block.error_message = None
    try:
        audio_result = await synthesize_block(
            block.tts_text,
            ctx.voicevox_settings,
            output,
            client=ctx.bundle.voicevox,
            sentence_pause_seconds=ctx.project.narration_sentence_pause_seconds,
            plan_concurrency=ctx.settings.narration_query_concurrency,
        )
        # Sentence timings are what the render stage times captions and slide
        # changes against, and rerender runs long after this — persist them.
        narration.write_spans(
            block_narration_path(ctx.project.id, block.index),
            audio_result.spans,
            duration_ms=audio_result.duration_ms,
        )
        block.audio_path = relpath_for_db(audio_result.path)
        block.duration_ms = audio_result.duration_ms
        block.display_duration_ms = compute_display_duration_ms(
            audio_result.duration_ms,
            pre_seconds=ctx.project.pre_margin_seconds,
            post_seconds=ctx.project.post_margin_seconds,
            min_seconds=ctx.project.min_display_seconds,
        )
        block.status_audio = BlockStatus.completed
        db.commit()
    except Exception as exc:
        block.status_audio = BlockStatus.failed
        block.error_message = f"{exc.__class__.__name__}: {str(exc)[:200]}"
        db.commit()
        raise


async def run_render_stage(ctx: StageContext, db: Session) -> Path:
    """Encode block videos, concatenate them, and write project metadata."""
    ensure_project_layout(ctx.project.id)
    settings = ctx.settings

    blocks = sorted(ctx.project.blocks, key=lambda b: b.index)
    if not blocks:
        raise RuntimeError("レンダリング対象のブロックがありません")

    # Render per-block videos if any are missing.
    settings = ctx.settings
    storage_root = settings.storage_root.resolve()
    per_block_videos: list[Path] = []
    cues: list[subtitles.SubtitleCue] = []
    timeline: list[dict[str, Any]] = []
    cursor_ms = 0
    for block in blocks:
        if ctx.is_cancelled():
            break
        db.refresh(block)
        if not block.image_path or not block.audio_path or not block.display_duration_ms:
            raise RuntimeError(
                f"ブロック {block.index} の素材が揃っていません "
                f"(image={block.image_path!r}, audio={block.audio_path!r}, dur={block.display_duration_ms})"
            )
        image = (storage_root / block.image_path).resolve()
        audio = (storage_root / block.audio_path).resolve()
        if not image.exists() or not audio.exists():
            raise RuntimeError(
                f"ブロック {block.index} の素材ファイルが見つかりません "
                f"(image={image}, audio={audio})"
            )
        output = block_video_path(ctx.project.id, block.index)
        if not output.exists():
            # Per-block burn-in subtitle (.ass with start=0). Subtitles land
            # in the lower band so they never overlap the slide.
            spans = narration.read_spans(
                block_narration_path(ctx.project.id, block.index)
            )
            block_ass: Path | None = None
            # A slide may change at the end of any sentence — that is where
            # the voice pauses. Caption changes are a subset of those, but
            # they matter more (a slide turning mid-caption is the visible
            # kind of mismatch), so both are offered and the snapper picks
            # whichever is nearest.
            boundaries = [s.end_ms for s in spans[:-1]]
            if ctx.project.subtitle_enabled:
                block_ass = block_subtitle_path(ctx.project.id, block.index)
                # The cue tracks the narration, not the whole block: the
                # trailing hold is meant to be a clean beat on the slide, so
                # the subtitle clears once there is nothing left being said.
                boundaries += _write_block_ass(
                    block_ass,
                    text=block.tts_text,
                    duration_ms=block.duration_ms or block.display_duration_ms or 0,
                    settings=settings,
                    project=ctx.project,
                    spans=spans,
                )
            slides = _block_slides(
                ctx.project.id,
                block.index,
                image,
                block.display_duration_ms,
                boundaries_ms=sorted(set(boundaries)),
                max_slides=ctx.project.max_slides_per_block,
            )
            args = ffmpeg_runner.build_block_video_args(
                slides=slides,
                audio=audio,
                duration_ms=block.display_duration_ms,
                output=output,
                ffmpeg=settings.ffmpeg_path or "ffmpeg",
                width=settings.output_width,
                height=settings.output_height,
                fps=settings.output_fps,
                subtitle_path=block_ass,
                subtitle_band_height=settings.subtitle_band_height
                if ctx.project.subtitle_enabled
                else 0,
            )
            await ffmpeg_runner.run_ffmpeg(
                args,
                log_path=project_dir(ctx.project.id) / "logs" / f"block_{block.index:04d}.log",
            )
            block.video_path = relpath_for_db(output)
            db.commit()
        per_block_videos.append(output)
        # timeline / subtitle cues (whole-project .ass uses tts_text)
        start = cursor_ms
        end = cursor_ms + (block.duration_ms or 0)
        cues.append(
            subtitles.SubtitleCue(
                start_ms=start,
                end_ms=end,
                text=block.tts_text,
            )
        )
        timeline.append(
            {
                "index": block.index,
                "start_ms": start,
                "end_ms": end,
                "duration_ms": block.duration_ms,
                "display_duration_ms": block.display_duration_ms,
                "image_path": block.image_path,
                "audio_path": block.audio_path,
                "video_path": relpath_for_db(output),
                "source_text": block.source_text,
                "tts_text": block.tts_text,
                "visual_type": block.visual_type.value if block.visual_type else None,
            }
        )
        cursor_ms = end
        block.status_render = BlockStatus.completed
        db.commit()

    if ctx.is_cancelled():
        raise RuntimeError("ユーザーによりキャンセルされました")

    # concat
    list_file = concat_list_path(ctx.project.id)
    ffmpeg_runner.write_concat_list(per_block_videos, list_file)
    final = output_video_path(ctx.project.id)
    args = ffmpeg_runner.build_concat_args(
        list_file=list_file,
        output=final,
        ffmpeg=settings.ffmpeg_path or "ffmpeg",
        crossfade_seconds=settings.crossfade_seconds,
    )
    await ffmpeg_runner.run_ffmpeg(
        args,
        log_path=project_dir(ctx.project.id) / "logs" / "concat.log",
    )
    ctx.project.output_video_path = relpath_for_db(final)

    # subtitles (whole-project .ass for external players; absolute timeline)
    ass_path = project_subtitle_path(ctx.project.id)
    if ctx.project.subtitle_enabled:
        subtitles.render_ass(
            cues,
            ass_path,
            width=settings.output_width,
            height=settings.output_height,
            font_size=ctx.project.subtitle_font_size,
            position=ctx.project.subtitle_position,
            text_color=ctx.project.subtitle_text_color,
            outline_color=ctx.project.subtitle_outline_color,
            background=ctx.project.subtitle_background,
        )
    ctx.project.output_subtitle_path = relpath_for_db(ass_path) if ass_path.exists() else None

    # write timeline + project.json
    project_json_path(ctx.project.id).write_text(
        _project_json_payload(ctx.project, timeline),
        encoding="utf-8",
    )
    timeline_json_path(ctx.project.id).write_text(
        _timeline_json_payload(timeline),
        encoding="utf-8",
    )
    db.commit()
    return final


MIN_SLIDE_MS = 2000


def _snap_to_boundaries(
    ideal_ms: list[int], boundaries: list[int], *, total_ms: int
) -> list[int]:
    """Move each slide change onto the nearest caption boundary.

    Splitting a block's running time evenly puts most changes in the middle
    of a sentence — measured on a real project, 56 of 68 changes landed more
    than a second away from any boundary, which reads as the slide moving on
    before the narrator has. Boundaries that are already taken, or that would
    leave a slide shorter than ``MIN_SLIDE_MS``, are skipped; when none is
    usable the ideal point is kept rather than dropping the slide.
    """
    cuts: list[int] = []
    previous = 0
    for i, ideal in enumerate(ideal_ms):
        remaining = len(ideal_ms) - i - 1
        # Leave room for the changes still to come, and for the final slide.
        latest = total_ms - MIN_SLIDE_MS * (remaining + 1)
        usable = [
            b for b in boundaries
            if b not in cuts and previous + MIN_SLIDE_MS <= b <= latest
        ]
        cut = min(usable, key=lambda b: abs(b - ideal)) if usable else ideal
        cuts.append(cut)
        previous = cut
    return cuts


def _block_slides(
    project_id: int,
    index: int,
    primary: Path,
    duration_ms: int,
    *,
    boundaries_ms: list[int] | None = None,
    max_slides: int = 1,
) -> list[tuple[Path, int]]:
    """Return the ordered (image, duration) pairs a block should display.

    Extra slides are whatever ``image_1.png``, ``image_2.png`` … the image
    stage produced for this block. The block's running time is shared equally
    between them, but never below ``MIN_SLIDE_MS`` and never across more than
    ``max_slides`` of them — a block would otherwise flash six listings past
    at six seconds each, too fast to read. Each change is then pulled onto the
    nearest caption boundary in ``boundaries_ms`` so slides turn between
    sentences rather than during them.

    Slides past the cap are dropped from the end, keeping the planner's chosen
    visual first and the rest in the order the script introduces them, which
    is the order the narration talks about them.
    """
    images = [primary]
    for slot in range(1, 9):
        path = block_image_path(project_id, index, slot)
        if not path.exists():
            break
        images.append(path)

    total = max(1, duration_ms)
    affordable = max(1, total // MIN_SLIDE_MS)
    allowed = min(affordable, max(1, max_slides))
    images = images[: max(1, min(len(images), allowed))]
    if len(images) == 1:
        return [(images[0], total)]

    ideal = [total * (i + 1) // len(images) for i in range(len(images) - 1)]
    cuts = _snap_to_boundaries(ideal, sorted(boundaries_ms or []), total_ms=total)

    slides: list[tuple[Path, int]] = []
    previous = 0
    for image, cut in zip(images, cuts):
        slides.append((image, cut - previous))
        previous = cut
    slides.append((images[-1], total - previous))
    return slides


def _write_block_ass(
    ass_path: Path,
    *,
    text: str,
    duration_ms: int,
    settings: Settings,
    project: Project,
    spans: list[narration.SentenceSpan] | None = None,
) -> list[int]:
    """Write the burn-in .ass for one block and return its cue boundaries.

    The narration is split into band-sized cues that advance underneath the
    (stationary) slide, rather than one cue holding the whole block — a long
    block shown at once has to shrink to an unreadable size. ``spans`` are the
    sentence timings VOICEVOX measured when the audio was synthesised; with
    them the cues change on the voice rather than on a character count.

    The returned boundaries are where a slide may change without cutting into
    a sentence.
    """
    cues, font_size = subtitles.build_band_cues(
        text,
        duration_ms=max(1, duration_ms),
        band_height=settings.subtitle_band_height,
        base_font_size=project.subtitle_font_size,
        base_max_chars=project.subtitle_max_chars_per_line,
        char_time=narration.char_time_fn(spans, total_ms=duration_ms) if spans else None,
    )
    if not cues:
        cues = [subtitles.SubtitleCue(start_ms=0, end_ms=max(1, duration_ms), text="")]
    subtitles.render_ass(
        cues,
        ass_path,
        width=settings.output_width,
        height=settings.output_height,
        font_size=font_size,
        position=project.subtitle_position,
        text_color=project.subtitle_text_color,
        outline_color=project.subtitle_outline_color,
        background=project.subtitle_background,
    )
    return [c.end_ms for c in cues[:-1]]


def _project_json_payload(project: Project, timeline: list[dict[str, Any]]) -> str:
    """Serialize project settings and its timeline as human-readable JSON."""
    import json

    payload = {
        "id": project.id,
        "title": project.title,
        "global_visual_style": project.global_visual_style,
        "output_video_path": project.output_video_path,
        "output_subtitle_path": project.output_subtitle_path,
        "blocks": timeline,
        "subtitle": {
            "enabled": project.subtitle_enabled,
            "font_size": project.subtitle_font_size,
            "position": project.subtitle_position,
            "text_color": project.subtitle_text_color,
            "outline_color": project.subtitle_outline_color,
            "background": project.subtitle_background,
            "max_chars_per_line": project.subtitle_max_chars_per_line,
        },
        "voicevox": {
            "url": project.voicevox_url,
            "speaker_id": project.voicevox_speaker_id,
            "speed_scale": project.voicevox_speed_scale,
            "pitch_scale": project.voicevox_pitch_scale,
            "intonation_scale": project.voicevox_intonation_scale,
            "volume_scale": project.voicevox_volume_scale,
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _timeline_json_payload(timeline: list[dict[str, Any]]) -> str:
    """Serialize only timeline rows for consumers that do not need settings."""
    import json

    return json.dumps(timeline, ensure_ascii=False, indent=2)


# ------------------ Public entry points ---------------------------------------


async def run_full_pipeline(
    project_id: int,
    *,
    progress_cb: ProgressCallback | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> None:
    """Run every stage for ``project_id``. Idempotent across stages."""
    from app.db import get_session_factory

    factory = get_session_factory()
    db = factory()
    try:
        project = db.execute(
            select(Project).where(Project.id == project_id)
        ).scalar_one_or_none()
        if project is None:
            raise RuntimeError(f"project {project_id} not found")
        bundle = build_providers_for_project(project)
        voicevox_settings = build_voicevox_settings(project)
        ctx = StageContext(
            project=project,
            settings=get_settings(),
            bundle=bundle,
            voicevox_settings=voicevox_settings,
            progress_cb=progress_cb,
            is_cancelled=cancel_check or (lambda: False),
        )

        async def report(stage: str, progress: float, message: str | None = None) -> None:
            project.current_stage = stage
            project.progress = progress
            db.commit()
            if progress_cb:
                await progress_cb(stage, progress, message)

        try:
            project.status = ProjectStatus.splitting
            await report("split", 0.05)
            await run_split_stage(ctx, db)

            project.status = ProjectStatus.planning
            await report("plan", 0.25)
            await run_visual_plan_stage(ctx, db)

            project.status = ProjectStatus.generating
            await report("image", 0.5)
            await run_image_stage(ctx, db)

            await report("audio", 0.7)
            await run_audio_stage(ctx, db)

            project.status = ProjectStatus.rendering
            await report("render", 0.85)
            await run_render_stage(ctx, db)

            project.status = ProjectStatus.completed
            project.error_message = None
            await report("done", 1.0)
        except Exception as exc:
            if ctx.is_cancelled():
                project.status = ProjectStatus.cancelled
                project.error_message = "ユーザーによりキャンセルされました"
            else:
                project.status = ProjectStatus.failed
                project.error_message = f"{exc.__class__.__name__}: {str(exc)[:300]}"
            raise
        finally:
            db.commit()
    finally:
        db.close()


async def rerun_block_visual(
    project_id: int, block_index: int, *, progress_cb: ProgressCallback | None = None
) -> None:
    """Re-render one block's image while preserving its audio and plan."""
    from app.db import get_session_factory

    factory = get_session_factory()
    db = factory()
    try:
        project = db.execute(
            select(Project).where(Project.id == project_id)
        ).scalar_one_or_none()
        if project is None:
            raise RuntimeError("project not found")
        block = next((b for b in project.blocks if b.index == block_index), None)
        if block is None:
            raise RuntimeError("block not found")
        bundle = build_providers_for_project(project)
        ctx = StageContext(
            project=project,
            settings=get_settings(),
            bundle=bundle,
            voicevox_settings=build_voicevox_settings(project),
            progress_cb=progress_cb,
        )
        # Reset status to pending so we re-run.
        block.status_image = BlockStatus.pending
        # refresh the block in DB
        db.commit()
        ensure_project_layout(project_id)
        style = project.global_visual_style or ""
        block.status_image = BlockStatus.running
        try:
            await _render_block_image(ctx, block, style, db)
        finally:
            db.commit()
    finally:
        db.close()


async def rerun_block_audio(
    project_id: int, block_index: int, *, progress_cb: ProgressCallback | None = None
) -> None:
    """Re-synthesize one block's audio and refresh its timing metadata."""
    from app.db import get_session_factory

    factory = get_session_factory()
    db = factory()
    try:
        project = db.execute(
            select(Project).where(Project.id == project_id)
        ).scalar_one_or_none()
        if project is None:
            raise RuntimeError("project not found")
        block = next((b for b in project.blocks if b.index == block_index), None)
        if block is None:
            raise RuntimeError("block not found")
        bundle = build_providers_for_project(project)
        ctx = StageContext(
            project=project,
            settings=get_settings(),
            bundle=bundle,
            voicevox_settings=build_voicevox_settings(project),
            progress_cb=progress_cb,
        )
        block.status_audio = BlockStatus.pending
        db.commit()
        block.status_audio = BlockStatus.running
        try:
            await _render_block_audio(ctx, block, db)
        finally:
            db.commit()
    finally:
        db.close()


async def rerender_project(
    project_id: int, *, progress_cb: ProgressCallback | None = None
) -> None:
    """Delete stale video artifacts and rebuild only the project render stage."""
    from app.db import get_session_factory

    factory = get_session_factory()
    db = factory()
    try:
        project = db.execute(
            select(Project).where(Project.id == project_id)
        ).scalar_one_or_none()
        if project is None:
            raise RuntimeError("project not found")
        # mark every block render as pending
        for b in project.blocks:
            b.status_render = BlockStatus.pending
        # Remove block videos + final video to force re-render. ``video_path``
        # is stored relative to the storage root (``projects/0007/...``), so it
        # must be joined to that, not to the project directory — joining to the
        # project dir produced a path that never existed, and the silent
        # missing_ok unlink meant rerender only ever re-concatenated the stale
        # block videos instead of rebuilding them.
        storage_root = get_settings().storage_root
        for b in project.blocks:
            if not b.video_path:
                continue
            vp = (storage_root / b.video_path).resolve()
            vp.unlink(missing_ok=True)
        final = output_video_path(project_id)
        if final.exists():
            final.unlink(missing_ok=True)
        project.output_video_path = None
        db.commit()
        bundle = build_providers_for_project(project)
        ctx = StageContext(
            project=project,
            settings=get_settings(),
            bundle=bundle,
            voicevox_settings=build_voicevox_settings(project),
            progress_cb=progress_cb,
        )
        await run_render_stage(ctx, db)
        project.status = ProjectStatus.completed
        db.commit()
    finally:
        db.close()
