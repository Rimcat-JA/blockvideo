"""End-to-end split-to-video pipeline orchestration.

Each stage owns one type of artifact and commits its status/results before the
next stage runs.  Existing artifacts/statuses are used as skip signals where
implemented; the full pipeline itself is rerunnable, but this module does not
provide a universal content-hash cache for every stage.

Imports:
    SQLAlchemy loads and persists projects/blocks.
    Configuration/logging/model modules provide runtime settings and state.
    Provider/path/splitter/narration/render modules implement each stage.
    ``Any`` describes timeline JSON payloads.
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
from app.services import ffmpeg_runner, image_renderer, subtitles, presentation
from app.services.hashing import short_hash
from app.services.paths import (
    block_audio_path,
    block_image_path,
    block_focus_dir,
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


# Callback shape: ``stage``, overall progress fraction, optional user message.
ProgressCallback = Callable[[str, float, str | None], Awaitable[None]]


@dataclass
class StageContext:
    """Shared runtime dependencies passed to stage functions.

    Attributes:
        project: SQLAlchemy project row and its loaded block relationship.
        settings: Global runtime/rendering settings.
        bundle: Real or fake LLM/image/VOICEVOX clients.
        voicevox_settings: Project-specific synthesis controls.
        progress_cb: Optional worker callback for stage progress.
        is_cancelled: Pollable cancellation predicate, false by default.

    """

    project: Project
    settings: Settings
    bundle: ProviderBundle
    voicevox_settings: VoicevoxSettings
    progress_cb: ProgressCallback | None = None
    is_cancelled: Callable[[], bool] = lambda: False
    block_indices: frozenset[int] | None = None
    input_guard: Callable[[], None] | None = None
    output_directory: Path | None = None
    accept_generated_inputs: Callable[[], None] | None = None

    async def report(self, stage: str, progress: float, message: str | None = None) -> None:
        """Forward stage progress to the optional worker callback.

        Args:
            stage: Stable stage name such as ``split`` or ``render``.
            progress: Overall pipeline progress fraction.
            message: Optional human-readable progress detail.

        Side Effects:
            Invokes ``progress_cb`` when configured; otherwise does nothing.

        """
        if self.progress_cb:
            await self.progress_cb(stage, progress, message)


@dataclass
class RenderContext:
    """Local rendering needs persisted media and settings, without API clients."""

    project: Project
    settings: Settings
    is_cancelled: Callable[[], bool] = lambda: False
    input_guard: Callable[[], None] | None = None
    output_directory: Path | None = None


def _check_boundary(ctx: StageContext | RenderContext) -> None:
    """Stop cooperatively before consuming changed inputs or publishing effects."""
    from app.services.generation_snapshots import GenerationCancelled

    if getattr(ctx, "is_cancelled", lambda: False)():
        raise GenerationCancelled("ユーザーによりキャンセルされました")
    guard = getattr(ctx, "input_guard", None)
    if guard:
        guard()


def _selected_blocks(ctx: StageContext) -> list[Block]:
    indices = getattr(ctx, "block_indices", None)
    return [block for block in sorted(ctx.project.blocks, key=lambda row: row.index)
            if indices is None or block.index in indices]


async def ensure_global_style(ctx: StageContext, db: Session) -> str:
    """Load a saved project style or generate and persist one.

    Args:
        ctx: Stage dependencies and project row.
        db: Active SQLAlchemy session.

    Returns:
        Existing or newly generated global visual-style text.

    Side Effects:
        Calls the LLM only when the project has no style, then commits and
        refreshes the project row.

    """
    if ctx.project.global_visual_style:
        return ctx.project.global_visual_style
    style = await generate_global_style(ctx.bundle.llm, project_title=ctx.project.title)
    _check_boundary(ctx)
    ctx.project.global_visual_style = style
    db.add(ctx.project)
    db.commit()
    db.refresh(ctx.project)
    if getattr(ctx, "accept_generated_inputs", None):
        ctx.accept_generated_inputs()
    return style


async def run_split_stage(ctx: StageContext, db: Session) -> list[Block]:
    """Split source text, repair narration, and synchronize block rows.

    Args:
        ctx: Project, provider, settings, and cancellation dependencies.
        db: Active session used to create/update/delete ``Block`` rows.

    Returns:
        Blocks corresponding exactly to the current split result, in source
        order.

    Side Effects:
        May call the LLM, delete rows beyond the new split count, update source
        and TTS text, and commit the synchronized block set.

    """
    _check_boundary(ctx)
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

    _check_boundary(ctx)
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
            if block.source_text != sb.source_text:
                block.status_visual_plan = BlockStatus.pending
                block.status_image = BlockStatus.pending
                block.status_render = BlockStatus.pending
            if block.tts_text != sb.tts_text:
                block.status_audio = BlockStatus.pending
                block.status_render = BlockStatus.pending
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

    Args:
        ctx: Project, planner provider, settings, and cancellation state.
        db: Active session used after concurrent planning completes.

    Returns:
        Number of visual plans completed during this invocation, including
        blocks already marked complete before the call.

    Raises:
        RuntimeError: If every block fails to obtain a usable plan while the
            stage is not cancelled.

    """
    import asyncio

    _check_boundary(ctx)
    style = await ensure_global_style(ctx, db)
    blocks = _selected_blocks(ctx)
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
                from app.services.external_calls import ExternalOutcomeUnknown

                if isinstance(exc, ExternalOutcomeUnknown):
                    raise
                return block, exc

    results = await asyncio.gather(*(plan_one(b) for b in todo))
    _check_boundary(ctx)

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
        block.image_prompt = plan.image_prompt if plan.visual_type == VisualType.ai_image else None
        block.content_hash = short_hash(
            block.source_text,
            block.tts_text,
            plan.model_dump_json(),
            style,
        )
        block.status_visual_plan = BlockStatus.completed
        block.status_image = BlockStatus.pending
        block.status_render = BlockStatus.pending
        if ctx.project.narration_pacing_mode == "adaptive":
            block.status_audio = BlockStatus.pending
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
    """Render every pending block image while honoring cancellation.

    Args:
        ctx: Project, render settings, providers, and cancellation predicate.
        db: Session used by the per-block renderer to commit statuses.

    Returns:
        Number of blocks already or newly marked image-complete.

    """
    count = 0
    style = ctx.project.global_visual_style or ""
    for block in _selected_blocks(ctx):
        _check_boundary(ctx)
        if block.status_image == BlockStatus.completed:
            count += 1
            continue
        await _render_block_image(ctx, block, style, db)
        count += 1
    return count


async def _render_block_image(
    ctx: StageContext, block: Block, style: str, db: Session
) -> None:
    """Render one block's primary and additional slide images.

    Args:
        ctx: Project/runtime dependencies.
        block: Block row whose plan and status are updated.
        style: Persisted global visual style used for remote image prompts.
        db: Session committed after success or failure.

    Side Effects:
        Creates/removes block image files, updates image status/path/error, and
        commits the block.  Errors are recorded and re-raised.

    """
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
            _check_boundary(ctx)
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
        _check_boundary(ctx)
        block.image_path = relpath_for_db(output)
        block.status_image = BlockStatus.completed
        block.status_render = BlockStatus.pending
        block.error_message = None
        db.commit()
    except Exception as exc:
        block.status_image = BlockStatus.failed
        block.error_message = f"{exc.__class__.__name__}: {str(exc)[:200]}"
        db.commit()
        raise


async def run_audio_stage(ctx: StageContext, db: Session) -> int:
    """Synthesize every pending block and persist durations/spans.

    Args:
        ctx: Project/runtime dependencies and VOICEVOX settings.
        db: Session used to commit each block's audio result.

    Returns:
        Number of blocks already or newly marked audio-complete.

    """
    count = 0
    ensure_project_layout(ctx.project.id)
    for block in _selected_blocks(ctx):
        _check_boundary(ctx)
        if block.status_audio == BlockStatus.completed and block.audio_path:
            count += 1
            continue
        await _render_block_audio(ctx, block, db)
        count += 1
    return count


async def _render_block_audio(
    ctx: StageContext, block: Block, db: Session
) -> None:
    """Synthesize one block and save its timing metadata.

    Args:
        ctx: Project/runtime dependencies.
        block: Block row whose audio fields/status are updated.
        db: Session committed after success or failure.

    Side Effects:
        Writes WAV and narration JSON files, computes display duration, updates
        database state, and re-raises failures after recording them.

    """
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
            pacing_mode=ctx.project.narration_pacing_mode,
            pronunciation_overrides=ctx.project.pronunciation_overrides,
            focus_terms=_narration_focus_terms(block),
        )
        _check_boundary(ctx)
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
        block.status_render = BlockStatus.pending
        db.commit()
    except Exception as exc:
        block.status_audio = BlockStatus.failed
        block.error_message = f"{exc.__class__.__name__}: {str(exc)[:200]}"
        db.commit()
        raise


def _narration_focus_terms(block: Block) -> list[tuple[str, ...]]:
    """Match visible labels per sentence for adaptive pauses and emphasis."""
    from app.services.visual_focus import focus_terms

    return [focus_terms(block.visual_plan_json or {}, piece.text)
            for piece in narration.split_sentences_with_offsets(block.tts_text)]


async def run_render_stage(ctx: StageContext | RenderContext, db: Session) -> Path:
    """Encode blocks, concatenate the project video, and write metadata.

    Args:
        ctx: Project, settings, providers, and cancellation state.
        db: Session used to persist block/project render paths and statuses.

    Returns:
        Final project MP4 path.

    Raises:
        RuntimeError: If there are no blocks, required artifacts are absent, or
            cancellation occurs before concatenation.
        ProviderError: If FFmpeg/FFprobe execution fails.

    Side Effects:
        Writes per-block ASS/MP4 files, concat list, final MP4, project ASS,
        timeline JSON, project JSON, and corresponding database fields.

    """
    _check_boundary(ctx)
    ensure_project_layout(ctx.project.id)
    managed_directory = getattr(ctx, "output_directory", None)
    if managed_directory:
        managed_directory.mkdir(parents=True, exist_ok=True)
    settings = ctx.settings

    blocks = sorted(ctx.project.blocks, key=lambda b: b.index)
    if not blocks:
        raise RuntimeError("レンダリング対象のブロックがありません")
    # Validate every block before replacing even the first block MP4. Existing
    # paths may still point to the last successful media after settings changed.
    from app.services.invalidation import stale_media_message

    stale = stale_media_message(blocks)
    if stale:
        raise RuntimeError(stale)

    # Render per-block videos if any are missing.
    settings = ctx.settings
    storage_root = settings.storage_root.resolve()
    per_block_videos: list[Path] = []
    cues: list[subtitles.SubtitleCue] = []
    external_font_size = ctx.project.subtitle_font_size
    timeline: list[dict[str, Any]] = []
    cursor_ms = 0
    for block in blocks:
        _check_boundary(ctx)
        db.refresh(block)
        if block.duration_ms:
            block.display_duration_ms = compute_display_duration_ms(
                block.duration_ms, pre_seconds=ctx.project.pre_margin_seconds,
                post_seconds=ctx.project.post_margin_seconds,
                min_seconds=ctx.project.min_display_seconds,
            )
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
        if not output.exists() or block.status_render != BlockStatus.completed:
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
            if ctx.project.visual_focus_enabled and len(slides) == 1:
                slide_height = settings.output_height
                if ctx.project.subtitle_enabled:
                    slide_height = max(120, slide_height - settings.subtitle_band_height)
                slides = presentation.focus_slides(
                    plan=block.visual_plan_json or {}, text=block.tts_text,
                    measured=spans, audio_ms=block.duration_ms or 0,
                    display_ms=block.display_duration_ms, primary=image,
                    directory=block_focus_dir(ctx.project.id, block.index),
                    width=settings.output_width, height=slide_height,
                )
            candidate = output.with_name("video.pending.mp4")
            args = ffmpeg_runner.build_block_video_args(
                slides=slides,
                audio=audio,
                duration_ms=block.display_duration_ms,
                output=candidate,
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
            _check_boundary(ctx)
            candidate.replace(output)
            block.video_path = relpath_for_db(output)
            db.commit()
        per_block_videos.append(output)
        # timeline / subtitle cues (whole-project .ass uses tts_text)
        start = cursor_ms
        end = cursor_ms + (block.duration_ms or 0)
        block_cues, block_font_size = _make_block_cues(
            text=block.tts_text, duration_ms=block.duration_ms or 0,
            settings=settings, project=ctx.project,
            spans=narration.read_spans(block_narration_path(ctx.project.id, block.index)),
        )
        external_font_size = min(external_font_size, block_font_size)
        cues.extend(subtitles.SubtitleCue(
            start_ms=start + cue.start_ms, end_ms=start + cue.end_ms,
            text=cue.text, margin_v=cue.margin_v,
        ) for cue in block_cues)
        timeline.append(
            {
                "index": block.index,
                "start_ms": start,
                "end_ms": start + block.display_duration_ms,
                "audio_end_ms": end,
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
        cursor_ms += block.display_duration_ms
        block.status_render = BlockStatus.completed
        db.commit()

    _check_boundary(ctx)

    # concat
    list_file = concat_list_path(ctx.project.id)
    ffmpeg_runner.write_concat_list(per_block_videos, list_file)
    final = (managed_directory / "video.mp4") if managed_directory else output_video_path(ctx.project.id)
    candidate_final = final.with_name("video.pending.mp4")
    args = ffmpeg_runner.build_concat_args(
        list_file=list_file,
        output=candidate_final,
        ffmpeg=settings.ffmpeg_path or "ffmpeg",
        crossfade_seconds=settings.crossfade_seconds,
    )
    await ffmpeg_runner.run_ffmpeg(
        args,
        log_path=project_dir(ctx.project.id) / "logs" / "concat.log",
    )
    _check_boundary(ctx)

    # subtitles (whole-project .ass for external players; absolute timeline)
    ass_path = (managed_directory / "subtitles.ass") if managed_directory else project_subtitle_path(ctx.project.id)
    if ctx.project.subtitle_enabled:
        subtitles.render_ass(
            cues,
            ass_path,
            width=settings.output_width,
            height=settings.output_height,
            font_size=external_font_size,
            position=ctx.project.subtitle_position,
            text_color=ctx.project.subtitle_text_color,
            outline_color=ctx.project.subtitle_outline_color,
            background=ctx.project.subtitle_background,
        )
    subtitle_path = relpath_for_db(ass_path) if ctx.project.subtitle_enabled and ass_path.exists() else None

    # write timeline + project.json
    metadata_path = (managed_directory / "project.json") if managed_directory else project_json_path(ctx.project.id)
    metadata_text = _project_json_payload(ctx.project, timeline)
    if managed_directory:
        import json

        metadata = json.loads(metadata_text)
        metadata["output_video_path"] = relpath_for_db(final)
        metadata["output_subtitle_path"] = subtitle_path
        metadata_text = json.dumps(metadata, ensure_ascii=False, indent=2)
    metadata_path.write_text(metadata_text, encoding="utf-8")
    timeline_path = (managed_directory / "timeline.json") if managed_directory else timeline_json_path(ctx.project.id)
    timeline_path.write_text(
        _timeline_json_payload(timeline),
        encoding="utf-8",
    )
    _check_boundary(ctx)
    if managed_directory:
        # The history publisher verifies media and owns the only current-pointer
        # and terminal-job commit. Working metadata never publishes by itself.
        db.commit()
        return candidate_final
    candidate_final.replace(final)
    ctx.project.output_video_path = relpath_for_db(final)
    ctx.project.output_subtitle_path = subtitle_path
    db.commit()
    return final


# Minimum readable lifetime for each slide in a multi-slide block.
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

    Args:
        ideal_ms: Desired ordered cut positions.
        boundaries: Candidate narration/caption boundaries.
        total_ms: Total block duration.

    Returns:
        Ordered cuts snapped to unused candidate boundaries when they preserve
        the minimum slide duration; otherwise the ideal cuts.

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

    Args:
        project_id: Owning project identifier.
        index: Block index.
        primary: Primary image path produced by the image stage.
        duration_ms: Total block display duration.
        boundaries_ms: Optional sentence/caption boundaries for snapping.
        max_slides: Project-level slide cap.

    Returns:
        Ordered ``(image_path, duration_ms)`` pairs whose durations sum to the
        block duration.  Missing extra images stop discovery.

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


def _make_block_cues(
    *, text: str, duration_ms: int, settings: Settings, project: Project,
    spans: list[narration.SentenceSpan] | None = None,
) -> tuple[list[subtitles.SubtitleCue], int]:
    """Share identical caption timing between burn-in and external subtitles."""
    if getattr(project, "subtitle_mode", "packed") == "sentence":
        return presentation.sentence_cues(
            text, duration_ms=duration_ms, measured=spans or [],
            band_height=settings.subtitle_band_height,
            font_size=project.subtitle_font_size,
            max_chars=project.subtitle_max_chars_per_line,
        )
    return subtitles.build_band_cues(
        text, duration_ms=max(1, duration_ms),
        band_height=settings.subtitle_band_height,
        base_font_size=project.subtitle_font_size,
        base_max_chars=project.subtitle_max_chars_per_line,
        char_time=narration.char_time_fn(spans, total_ms=duration_ms) if spans else None,
    )


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

    Args:
        ass_path: Per-block ASS destination.
        text: Narration text.
        duration_ms: Audio/narration interval for cue generation.
        settings: Global output/band settings.
        project: Project subtitle preferences.
        spans: Optional measured sentence spans.

    Returns:
        End times of every cue except the final cue; these are safe slide-change
        boundaries within the block.

    Side Effects:
        Writes a per-block ASS file and may create its parent directory.

    """
    cues, font_size = _make_block_cues(
        text=text, duration_ms=duration_ms, settings=settings,
        project=project, spans=spans,
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
    """Serialize project settings and timeline as formatted JSON text.

    Args:
        project: Persisted project settings and output paths.
        timeline: Already-built per-block absolute timeline rows.

    Returns:
        UTF-8-compatible JSON string with project, subtitle, VOICEVOX, and
        block timeline data.

    """
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
        "presentation": {
            "visual_focus_enabled": project.visual_focus_enabled,
            "subtitle_mode": project.subtitle_mode,
            "narration_pacing_mode": project.narration_pacing_mode,
            "pronunciation_overrides": project.pronunciation_overrides,
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _timeline_json_payload(timeline: list[dict[str, Any]]) -> str:
    """Serialize only timeline rows as formatted JSON text.

    Args:
        timeline: Per-block absolute timeline rows.

    Returns:
        JSON string without project/provider settings.

    """
    import json

    return json.dumps(timeline, ensure_ascii=False, indent=2)


# ------------------ Public entry points ---------------------------------------


async def run_full_pipeline(
    project_id: int,
    *,
    progress_cb: ProgressCallback | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> None:
    """Run split, plan, image, audio, and render stages for one project.

    Args:
        project_id: Existing project database identifier.
        progress_cb: Optional callback receiving stage/progress updates.
        cancel_check: Optional predicate polled by stages and worker code.

    Raises:
        RuntimeError: If the project does not exist or a stage fails.
        ProviderError: If configured external providers/media tools fail.

    Side Effects:
        Creates provider clients, mutates project/block/job-related database
        state, writes all media/metadata artifacts, and closes its session.
        Failures mark the project failed or cancelled before re-raising.

    """
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
    """Re-render one block's image while preserving audio and plan.

    Args:
        project_id: Existing project identifier.
        block_index: Zero-based block index to render.
        progress_cb: Optional callback retained for API symmetry.

    Raises:
        RuntimeError: If the project or block does not exist, or rendering
            fails.

    """
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
    """Re-synthesize one block and refresh timing metadata.

    Args:
        project_id: Existing project identifier.
        block_index: Zero-based block index to synthesize.
        progress_cb: Optional callback retained for API symmetry.

    Raises:
        RuntimeError: If the project/block is missing or synthesis fails.

    """
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
    """Rebuild local video output, keeping the last successful MP4 on failure.

    Args:
        project_id: Existing project identifier.
        progress_cb: Optional callback retained for stage-reporting symmetry.

    Raises:
        RuntimeError: If the project does not exist or required render inputs
            are unavailable.

    Side Effects:
        Marks block render states pending and atomically replaces successful
        outputs. No API keys or live synthesis providers are required.

    """
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
        project.status = ProjectStatus.rendering
        project.error_message = None
        db.commit()
        try:
            if progress_cb:
                await progress_cb("render", 0.0, None)
            await run_render_stage(RenderContext(project, get_settings()), db)
            project.status = ProjectStatus.completed
            project.progress = 1.0
            project.current_stage = "done"
        except Exception as exc:
            project.status = ProjectStatus.failed
            project.error_message = f"{exc.__class__.__name__}: {str(exc)[:300]}"
            raise
        finally:
            db.commit()
    finally:
        db.close()


def _providers_for_stages(project: Project, stages: list[str]) -> ProviderBundle:
    """Only initialize credentials for providers this execution can actually use."""
    if project.use_fake_providers or "split" in stages or "plan" in stages:
        return build_providers_for_project(project)
    from app.core.security import secret_store
    from app.providers.image_openai import OpenAIImageProvider
    from app.providers.llm_fake import FakeLLMProvider
    from app.providers.voicevox import VoicevoxClient

    settings = get_settings()
    secrets = secret_store.get(project.id)
    image_provider = None
    if "image" in stages and any(block.visual_type == VisualType.ai_image for block in project.blocks):
        key = (secrets.image_api_key if secrets else None) or settings.image_api_key
        if key:
            image_provider = OpenAIImageProvider(
                api_key=key,
                model=(secrets.image_model if secrets else None) or settings.image_model or "gpt-image-1",
                base_url=(secrets.image_base_url if secrets else None) or "https://api.openai.com/v1",
            )
    # The unused LLM is an offline placeholder; no planning stage can reach it.
    return ProviderBundle(llm=FakeLLMProvider(), image=image_provider,
                          voicevox=VoicevoxClient(project.voicevox_url), use_fake=False)


async def run_generation_job(job_id: int, cancel_check: Callable[[], bool]) -> None:
    """Execute one durable plan against checked inputs and publish immutable history.

    This is the managed production entry point. Legacy direct stage/pipeline
    functions retain their signatures for callers and regression tests.
    """
    from app.db import get_session_factory
    from app.models.job import GenerationJob, JobStatus
    from app.services.artifact_store import (
        collect_completed_materials, collect_render_materials, file_identity,
        preserve_legacy_video, publish_artifact, verify_materials,
    )
    from app.services.generation_plan import build_generation_plan
    from app.services.generation_snapshots import (
        FROZEN_SNAPSHOT_KEYS, GenerationCancelled, StaleGenerationInput, capture_inputs, fingerprint_inputs,
    )

    factory = get_session_factory()
    bundle: ProviderBundle | None = None
    with factory() as db:
        job = db.get(GenerationJob, job_id)
        if job is None or job.status != JobStatus.running:
            raise StaleGenerationInput("生成ジョブは実行中ではありません")
        project = db.get(Project, job.project_id)
        if project is None or not job.input_snapshot or not job.input_fingerprint:
            raise StaleGenerationInput("保存された生成入力が見つかりません")
        if fingerprint_inputs(job.input_snapshot) != job.input_fingerprint:
            raise StaleGenerationInput("保存された生成入力が破損しています")
        expected_revision = job.input_revision
        expected = (job.plan_json or {}).get("resume_inputs", job.input_snapshot)
        if not isinstance(expected, dict) or any(expected.get(key) != job.input_snapshot.get(key)
                                                for key in FROZEN_SNAPSHOT_KEYS):
            raise StaleGenerationInput("再開地点の設定または接続先が変更されています")
        expected_fingerprint = fingerprint_inputs(expected)
        if "resume_inputs" in (job.plan_json or {}) and job.plan_json.get("resume_fingerprint") != expected_fingerprint:
            raise StaleGenerationInput("保存された再開入力が破損しています")
        if project.revision != expected_revision or fingerprint_inputs(capture_inputs(project)) != expected_fingerprint:
            raise StaleGenerationInput("生成要求後に設定または入力が変更されました")
        expected_materials: list[dict[str, Any]] = []
        mutable_paths: set[str] = set()

        def guard() -> None:
            if cancel_check():
                raise GenerationCancelled("ユーザーによりキャンセルされました")
            with factory() as check_db:
                current_job = check_db.get(GenerationJob, job_id)
                current = check_db.get(Project, project.id)
                if current_job is None or current_job.cancel_requested:
                    raise GenerationCancelled("ユーザーによりキャンセルされました")
                if current is None or current.revision != expected_revision:
                    raise StaleGenerationInput("生成中に設定の版が変更されました")
                if fingerprint_inputs(capture_inputs(current)) != expected_fingerprint:
                    raise StaleGenerationInput("生成中に参照入力が変更されました")
            verify_materials([item for item in expected_materials if item["path"] not in mutable_paths])

        def checkpoint() -> None:
            nonlocal expected_fingerprint, expected_materials
            db.expire_all()
            snapshot = capture_inputs(project)
            # User settings must never be adopted from an out-of-band write.
            if project.revision != expected_revision or any(snapshot.get(key) != job.input_snapshot.get(key)
                                                            for key in FROZEN_SNAPSHOT_KEYS):
                raise StaleGenerationInput("生成中に設定が変更されました")
            expected_fingerprint = fingerprint_inputs(snapshot)
            expected_materials = collect_completed_materials(project)
            job.input_snapshot = snapshot
            job.input_fingerprint = expected_fingerprint
            job.plan_json = {**(job.plan_json or {}), "resume_inputs": snapshot,
                             "resume_fingerprint": expected_fingerprint,
                             "stage_materials": expected_materials}
            db.commit()

        guard()
        await preserve_legacy_video(project.id)
        guard()
        db.expire_all()
        plan = build_generation_plan(project, job.kind, job.block_index)
        expected_materials = collect_completed_materials(project)
        stages: list[str] = plan["stages"]
        if any(stage != "render" for stage in stages):
            bundle = _providers_for_stages(project, stages)
        context = StageContext(
            project=project, settings=get_settings(), bundle=bundle,  # type: ignore[arg-type]
            voicevox_settings=build_voicevox_settings(project), is_cancelled=cancel_check,
            input_guard=guard, accept_generated_inputs=checkpoint,
        )
        output_directory = project_dir(project.id) / "history" / f"job-{job_id:08d}"
        try:
            for number, stage in enumerate(stages):
                guard()
                mutable_paths.clear()
                job.current_stage = stage
                job.progress = number / max(1, len(stages))
                project.current_stage = stage
                project.progress = job.progress
                project.error_message = None
                project.status = {
                    "split": ProjectStatus.splitting, "plan": ProjectStatus.planning,
                    "image": ProjectStatus.generating, "audio": ProjectStatus.generating,
                    "render": ProjectStatus.rendering,
                }[stage]
                db.commit()
                if stage == "split":
                    await run_split_stage(context, db)
                    checkpoint()
                    # Newly split blocks did not exist when the request was accepted.
                    plan = build_generation_plan(project, job.kind, job.block_index)
                    continue
                selected = frozenset(int(index) for index, tasks in plan["blocks"].items() if stage in tasks)
                context.block_indices = selected
                for block in project.blocks:
                    if block.index not in selected:
                        continue
                    if stage == "plan":
                        block.status_visual_plan = BlockStatus.pending
                    elif stage == "image":
                        block.status_image = BlockStatus.pending
                        mutable_paths.update(relpath_for_db(block_image_path(project.id, block.index, slot))
                                             for slot in range(9))
                    elif stage == "audio":
                        block.status_audio = BlockStatus.pending
                        mutable_paths.update({relpath_for_db(block_audio_path(project.id, block.index)),
                                              relpath_for_db(block_narration_path(project.id, block.index))})
                    elif stage == "render":
                        block.status_render = BlockStatus.pending
                        mutable_paths.add(relpath_for_db(block_video_path(project.id, block.index)))
                db.commit()
                if stage == "plan":
                    await run_visual_plan_stage(context, db)
                    if any(block.status_visual_plan != BlockStatus.completed
                           for block in project.blocks if block.index in selected):
                        raise RuntimeError("画面構成が未完成のブロックがあります")
                elif stage == "image":
                    await run_image_stage(context, db)
                elif stage == "audio":
                    await run_audio_stage(context, db)
                elif stage == "render":
                    settled = capture_inputs(project)
                    materials = collect_render_materials(project)
                    reused_videos = [file_identity(block.video_path) for block in project.blocks
                                     if block.status_render == BlockStatus.completed and block.video_path]
                    render_context = RenderContext(
                        project, get_settings(), is_cancelled=cancel_check,
                        input_guard=guard, output_directory=output_directory,
                    )
                    candidate = await run_render_stage(render_context, db)
                    guard()
                    if collect_render_materials(project) != materials:
                        raise StaleGenerationInput("生成中に参照素材の一覧または内容が変更されました")
                    subtitle = output_directory / "subtitles.ass" if project.subtitle_enabled else None
                    await publish_artifact(job_id, candidate, subtitle,
                                           settled_inputs=settled, materials=materials + reused_videos,
                                           cancel_check=cancel_check,
                                           block_videos=[file_identity(block.video_path)
                                                         for block in project.blocks if block.video_path])
                    return
                checkpoint()
                mutable_paths.clear()
                guard()
            # Targeted intermediate operations have no final-video publication.
            guard()
            project.current_stage = "done"
            project.progress = 1.0
            project.status = ProjectStatus.completed
            db.commit()
        finally:
            if bundle is not None:
                closed: set[int] = set()
                for provider in (bundle.llm, bundle.image, bundle.voicevox, bundle.llm_planner):
                    if provider is not None and id(provider) not in closed:
                        closed.add(id(provider))
                        close = getattr(provider, "aclose", None)
                        if close:
                            await close()
