"""Caption, media replacement and quality-setting integration regressions."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.db import get_session_factory
from app.models.block import Block, BlockStatus, VisualType
from app.models.project import Project, ProjectStatus
from app.services import pipeline, presentation
from app.services.narration import SentenceSpan, write_spans
from app.services.paths import (
    block_audio_path, block_image_path, block_narration_path, block_video_path,
    ensure_project_layout, output_video_path, project_subtitle_path,
    relpath_for_db, timeline_json_path,
)


def test_sentence_captions_do_not_reveal_future_sentence():
    text = 'まずキーを探します。見つかったら値を返します。'
    boundary = text.index('見つかったら')
    spans = [SentenceSpan(text[:boundary], 0, boundary, 100, 3000),
             SentenceSpan(text[boundary:], boundary, len(text), 5000, 7500)]
    cues, size = presentation.sentence_cues(
        text, duration_ms=8000, measured=spans,
        band_height=200, font_size=48, max_chars=36,
    )
    assert size == 48
    assert [(cue.start_ms, cue.end_ms) for cue in cues] == [(100, 3000), (5000, 7500)]
    assert '見つかったら' not in cues[0].text
    assert ''.join(cue.text.replace('\\N', '') for cue in cues) == text


def test_sentence_fallback_rejects_stale_timings_and_keeps_display_text():
    text = 'lookupで検索します。値を返します。'
    stale = [SentenceSpan('ルックアップ', 0, 7, 0, 5000)]
    result = presentation.display_spans(text, 5000, stale)
    assert ''.join(span.text for span in result) == text
    assert result[0].text.startswith('lookup')
    assert result[-1].end_ms == 5000
    assert all(a.end_ms == b.start_ms for a, b in zip(result, result[1:]))


def test_focus_variants_follow_sentence_starts_and_reuse_terms(tmp_path, monkeypatch):
    from app.services import image_renderer
    text = 'tableを見ます。keyを探します。keyを取り出します。'
    pieces = presentation.display_spans(text, 6000)
    primary = tmp_path / 'image.png'
    primary.write_bytes(b'primary')
    rendered = []

    def render(plan, path, **kwargs):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'focused')
        rendered.append(kwargs['focus_text'])

    monkeypatch.setattr(image_renderer, 'render_visual_plan', render)
    sequence = presentation.focus_slides(
        plan={'visual_type': 'verbatim_slide', 'verbatim': 'table -> key'},
        text=text, measured=pieces, audio_ms=6000, display_ms=7000,
        primary=primary, directory=tmp_path / 'focus', width=1920, height=880,
    )
    assert len(rendered) == 3  # Current-layout baseline plus two reused focus variants.
    assert rendered[0] is None
    assert len(sequence) == 2
    assert sequence[0][1] == pieces[1].start_ms
    assert sum(duration for _, duration in sequence) == 7000


def test_focus_rerender_uses_current_neutral_layout_without_replacing_primary(tmp_path):
    from PIL import Image, ImageChops
    from app.services.image_renderer import render_visual_plan

    primary = tmp_path / 'image.png'
    Image.new('RGB', (640, 360), 'red').save(primary)
    original = primary.read_bytes()
    plan = {'visual_type': 'verbatim_slide', 'heading': '図', 'verbatim': 'key -> value'}
    text = '説明します。keyを見ます。説明を続けます。'
    sequence = presentation.focus_slides(
        plan=plan, text=text, measured=(), audio_ms=6000, display_ms=7000,
        primary=primary, directory=tmp_path / 'focus', width=640, height=360,
    )
    assert len(sequence) == 3
    assert sequence[0][0] == sequence[-1][0] != primary
    assert primary.read_bytes() == original
    expected = tmp_path / 'expected-neutral.png'
    render_visual_plan(plan, expected, width=640, height=360)
    with Image.open(expected) as current, Image.open(sequence[0][0]) as neutral:
        assert ImageChops.difference(current, neutral).getbbox() is None
    assert sum(duration for _, duration in sequence) == 7000


@pytest.mark.parametrize('plan,text', [
    ({'visual_type': 'diagram', 'diagram': 'key -> value'}, 'keyを見ます。'),
    ({'visual_type': 'verbatim_slide', 'verbatim': 'key -> value'}, '説明します。'),
])
def test_unsupported_or_unmatched_focus_keeps_primary_without_rendering(tmp_path, monkeypatch, plan, text):
    from app.services import image_renderer

    primary = tmp_path / 'image.png'
    monkeypatch.setattr(image_renderer, 'render_visual_plan',
                        lambda *args, **kwargs: pytest.fail('static plans must not be redrawn'))
    sequence = presentation.focus_slides(
        plan=plan, text=text, measured=(), audio_ms=2000, display_ms=3000,
        primary=primary, directory=tmp_path / 'focus', width=640, height=360,
    )
    assert sequence == [(primary, 3000)]


def test_focus_long_alternation_bounds_ffmpeg_inputs_and_finishes_neutral(tmp_path, monkeypatch):
    from app.services import image_renderer

    rendered = []
    monkeypatch.setattr(image_renderer, 'render_visual_plan',
                        lambda plan, path, **kwargs: rendered.append((path, kwargs['focus_text'])))
    text = ''.join('keyです。' if i % 2 else 'valueです。' for i in range(100))
    sequence = presentation.focus_slides(
        plan={'visual_type': 'verbatim_slide', 'verbatim': 'key value'}, text=text,
        measured=(), audio_ms=100000, display_ms=101000,
        primary=tmp_path / 'image.png', directory=tmp_path / 'focus', width=640, height=360,
    )
    neutral = rendered[0][0]
    assert rendered[0][1] is None
    assert len(rendered) == 3  # Repeated labels reuse two files even across 100 sentences.
    assert len(sequence) <= 32
    assert sequence[-1][0] == neutral
    assert sum(duration for _, duration in sequence) == 101000


def test_focus_distinct_variant_limit_clears_earlier_highlight(tmp_path, monkeypatch):
    from app.services import image_renderer

    rendered = []
    monkeypatch.setattr(image_renderer, 'render_visual_plan',
                        lambda plan, path, **kwargs: rendered.append((path, kwargs['focus_text'])))
    sentences = [f'label_{i}です。' for i in range(40)]
    text = ''.join(sentences)
    spans = []
    cursor = 0
    for i, sentence in enumerate(sentences):
        spans.append(SentenceSpan(sentence, cursor, cursor + len(sentence), i * 25, (i + 1) * 25))
        cursor += len(sentence)
    sequence = presentation.focus_slides(
        plan={'visual_type': 'verbatim_slide', 'verbatim': ' '.join(f'label_{i}' for i in range(40))},
        text=text, measured=spans, audio_ms=1000, display_ms=2000,
        primary=tmp_path / 'image.png', directory=tmp_path / 'focus', width=640, height=360,
    )
    assert len(rendered) == 33  # One baseline and at most 32 distinct highlighted files.
    assert len(sequence) < 32  # Hit the variant limit before the input/event limit.
    assert sequence[-1][0] == rendered[0][0]
    assert sequence[-1][1] >= 1000
    assert sum(duration for _, duration in sequence) == 2000


def make_completed_project(db):
    project = Project(title='quality', source_script='tableを見ます。',
                      use_fake_providers=True, status=ProjectStatus.completed,
                      visual_focus_enabled=False, pre_margin_seconds=0,
                      post_margin_seconds=1, subtitle_mode='sentence')
    db.add(project)
    db.flush()
    ensure_project_layout(project.id)
    for index in range(2):
        text = 'tableを見ます。値を取り出します。'
        block = Block(project_id=project.id, index=index, source_text=text,
                      tts_text=text, visual_type=VisualType.verbatim_slide,
                      visual_plan_json={'visual_type':'verbatim_slide', 'verbatim':'table -> value'},
                      status_render=BlockStatus.completed, status_audio=BlockStatus.completed,
                      status_image=BlockStatus.completed, duration_ms=2000,
                      display_duration_ms=3000)
        db.add(block)
        for path in [block_audio_path(project.id,index),block_image_path(project.id,index),
                     block_video_path(project.id,index)]:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'old')
        block.audio_path = relpath_for_db(block_audio_path(project.id,index))
        block.image_path = relpath_for_db(block_image_path(project.id,index))
        block.video_path = relpath_for_db(block_video_path(project.id,index))
        spans = presentation.display_spans(text, 2000)
        write_spans(block_narration_path(project.id,index), spans, duration_ms=2000)
    final = output_video_path(project.id)
    final.write_bytes(b'completed video')
    project.output_video_path = relpath_for_db(final)
    db.commit()
    return project


@pytest.mark.asyncio
async def test_rerender_failure_retains_completed_video_without_api_keys(temp_storage, monkeypatch):
    with get_session_factory()() as db:
        project = make_completed_project(db)
        project.use_fake_providers = False  # No BYOK or environment key in this test.
        db.commit()
        project_id = project.id

    async def failing_ffmpeg(args, **kwargs):
        Path(args[-1]).write_bytes(b'partial output')
        raise RuntimeError('encoding failure')

    monkeypatch.setattr(pipeline.ffmpeg_runner, 'run_ffmpeg', failing_ffmpeg)
    monkeypatch.setattr(pipeline, 'build_providers_for_project',
                        lambda *_: pytest.fail('render-only must not build LLM providers'))
    with pytest.raises(RuntimeError, match='encoding failure'):
        await pipeline.rerender_project(project_id)
    assert output_video_path(project_id).read_bytes() == b'completed video'
    assert block_video_path(project_id,0).read_bytes() == b'old'
    with get_session_factory()() as db:
        project = db.get(Project, project_id)
        assert project.status == ProjectStatus.failed
        assert project.output_video_path


@pytest.mark.asyncio
async def test_external_captions_include_display_hold(temp_storage, monkeypatch):
    async def successful_ffmpeg(args, **kwargs):
        Path(args[-1]).write_bytes(b'new video')

    monkeypatch.setattr(pipeline.ffmpeg_runner, 'run_ffmpeg', successful_ffmpeg)
    with get_session_factory()() as db:
        project = make_completed_project(db)
        project_id = project.id
    await pipeline.rerender_project(project_id)
    timeline = json.loads(timeline_json_path(project_id).read_text(encoding='utf-8'))
    assert [entry['start_ms'] for entry in timeline] == [0,3000]
    assert [entry['end_ms'] for entry in timeline] == [3000,6000]
    assert [entry['audio_end_ms'] for entry in timeline] == [2000,5000]
    ass = project_subtitle_path(project_id).read_text(encoding='utf-8-sig')
    assert '0:00:03.00' in ass
    assert output_video_path(project_id).read_bytes() == b'new video'


@pytest.mark.parametrize('stale_status', ['status_audio', 'status_image'])
@pytest.mark.asyncio
async def test_render_checks_all_media_before_replacing_any_completed_video(
    temp_storage, monkeypatch, stale_status,
):
    with get_session_factory()() as db:
        project = make_completed_project(db)
        # The first block is ready: detecting only while iterating would already
        # replace its MP4 before discovering the second block's stale asset.
        second = next(block for block in project.blocks if block.index == 1)
        setattr(second, stale_status, BlockStatus.pending)
        db.commit()
        project_id = project.id

    async def unexpected_encoding(*args, **kwargs):
        pytest.fail('No block may be encoded until every block has current media')

    monkeypatch.setattr(pipeline.ffmpeg_runner, 'run_ffmpeg', unexpected_encoding)
    with pytest.raises(RuntimeError, match='先に「再生成」'):
        await pipeline.rerender_project(project_id)

    assert output_video_path(project_id).read_bytes() == b'completed video'
    assert block_video_path(project_id, 0).read_bytes() == b'old'
    assert block_video_path(project_id, 1).read_bytes() == b'old'
    with get_session_factory()() as db:
        project = db.get(Project, project_id)
        assert project.output_video_path
        assert project.status == ProjectStatus.failed
        second = next(block for block in project.blocks if block.index == 1)
        assert getattr(second, stale_status) == BlockStatus.pending


@pytest.mark.parametrize('endpoint', ['projects', 'blocks'])
@pytest.mark.parametrize('update, stale_status', [
    ({'pronunciation_overrides': [{'surface': 'table', 'reading': 'テーブル'}]}, 'status_audio'),
    ({'subtitle_enabled': False}, 'status_image'),
])
def test_rerender_api_explains_stale_media_without_enqueueing(
    temp_storage, monkeypatch, endpoint, update, stale_status,
):
    from fastapi.testclient import TestClient
    from sqlalchemy import func, select

    from app.api import routes_blocks, routes_projects
    from app.main import create_app
    from app.models.job import GenerationJob

    with get_session_factory()() as db:
        project = make_completed_project(db)
        project_id = project.id
        block_id = next(block.id for block in project.blocks if block.index == 0)

    async def unexpected_enqueue(*args, **kwargs):
        pytest.fail('Stale media must be rejected before starting a job')

    monkeypatch.setattr(routes_projects, 'enqueue_rerender', unexpected_enqueue)
    monkeypatch.setattr(routes_blocks, 'enqueue_rerender', unexpected_enqueue)
    client = TestClient(create_app())
    changed = client.patch(f'/api/projects/{project_id}', json=update)
    assert changed.status_code == 200, changed.text
    with get_session_factory()() as db:
        project = db.get(Project, project_id)
        assert all(getattr(block, stale_status) == BlockStatus.pending for block in project.blocks)

    identifier = project_id if endpoint == 'projects' else block_id
    response = client.post(f'/api/{endpoint}/{identifier}/rerender')
    assert response.status_code == 409, response.text
    assert '音声・画像が最新ではありません' in response.json()['detail']
    assert '生成開始' in response.json()['detail']
    assert '再生成' in response.json()['detail']
    assert output_video_path(project_id).read_bytes() == b'completed video'
    with get_session_factory()() as db:
        assert db.scalar(select(func.count()).select_from(GenerationJob)) == 0
        project = db.get(Project, project_id)
        assert project.status == ProjectStatus.completed
