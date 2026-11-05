"""MakeupPipeline with fake tracker / segmenter / effects (+ optional real models)."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from makeup_cam import effects as effects_pkg
from makeup_cam.config import MakeupSettings
from makeup_cam.effects.base import Effect, FrameContext
from makeup_cam.models import MODEL_SPECS
from makeup_cam.pipeline import FADE_FRAMES, MakeupPipeline
from makeup_cam.segmentation import SegMasks

H, W = 48, 64
FACE = np.random.default_rng(0).uniform(10, 40, (478, 2)).astype(np.float32)


def sample_frame() -> np.ndarray:
    frame = np.zeros((H, W, 3), np.uint8)
    frame[:, : W // 2] = (40, 80, 120)  # asymmetric, so mirroring is visible
    frame[:, W // 2 :] = (100, 100, 100)
    return frame


class FakeTracker:
    """Returns the scripted landmarks (None = no face) one call after another."""

    def __init__(self, script: list[np.ndarray | None] | None = None, default: np.ndarray | None = FACE) -> None:
        self.script = list(script or [])
        self.default = default
        self.smoothing: list[float] = []
        self.closed = 0
        self.fail: Exception | None = None

    def detect(self, frame: np.ndarray, ts_ms: int) -> tuple[np.ndarray | None, np.ndarray | None]:
        if self.fail:
            raise self.fail
        landmarks = self.script.pop(0) if self.script else self.default
        if landmarks is None:
            return None, None
        return landmarks.copy(), np.zeros(len(landmarks), np.float32)

    def set_smoothing(self, smoothing: float) -> None:
        self.smoothing.append(smoothing)

    def close(self) -> None:
        self.closed += 1


class FakeSegmenter:
    def __init__(self, masks: SegMasks | None = None) -> None:
        self.masks = masks or SegMasks(*(np.full((H, W), 0.5, np.float32) for _ in range(3)))
        self.updates: list[int] = []
        self.intervals: list[int] = []
        self.closed = 0
        self.fail: Exception | None = None

    def update(self, frame: np.ndarray, ts_ms: int, frame_index: int) -> SegMasks:
        if self.fail:
            raise self.fail
        self.updates.append(frame_index)
        return self.masks

    def set_interval(self, interval: int) -> None:
        self.intervals.append(interval)

    def close(self) -> None:
        self.closed += 1


class FakeEffect(Effect):
    """Brightens the frame by ``delta * ctx.intensity`` and records what it saw."""

    name = "lips"

    def __init__(self, delta: float = 0.1, fail: Exception | None = None, result: Any = None) -> None:
        self.delta = delta
        self.fail = fail
        self.result = result
        self.calls: list[FrameContext] = []
        self.intensities: list[float] = []
        self.resets = 0
        self.log: list[str] | None = None

    def apply(self, frame: np.ndarray, ctx: FrameContext, settings: Any) -> np.ndarray:
        self.calls.append(ctx)
        self.intensities.append(ctx.intensity)
        if self.log is not None:
            self.log.append(self.name)
        if self.fail:
            raise self.fail
        if self.result is not None:
            return self.result
        frame += self.delta * ctx.intensity
        return frame

    def reset(self) -> None:
        self.resets += 1


def effect(
    name: str, order: int = 50, *, needs_face: bool = True, needs_hair_mask: bool = False, **kwargs: Any
) -> FakeEffect:
    cls = type(
        f"Fake_{name}",
        (FakeEffect,),
        {"name": name, "order": order, "needs_face": needs_face, "needs_hair_mask": needs_hair_mask},
    )
    return cls(**kwargs)


def hair_effect(**kwargs: Any) -> FakeEffect:
    return effect("hair", 10, needs_face=False, needs_hair_mask=True, **kwargs)


def make_pipeline(
    effects: list[FakeEffect], settings: MakeupSettings | None = None, **kwargs: Any
) -> tuple[MakeupPipeline, FakeTracker, FakeSegmenter]:
    tracker = kwargs.pop("tracker", FakeTracker())
    segmenter = kwargs.pop("segmenter", FakeSegmenter())
    pipeline = MakeupPipeline(settings, tracker=tracker, segmenter=segmenter, effects=list(effects), **kwargs)
    return pipeline, tracker, segmenter


def settings_with(**general: Any) -> MakeupSettings:
    settings = MakeupSettings()
    settings.general.mirror_preview = False
    for key, value in general.items():
        setattr(settings.general, key, value)
    return settings


def run(pipeline: MakeupPipeline, frames: int, start_ms: int = 0) -> list[Any]:
    return [pipeline.process(sample_frame(), start_ms + 33 * i) for i in range(frames)]


# ---------------------------------------------------------------------------


def test_effects_run_in_order_and_change_the_output() -> None:
    order: list[str] = []
    late, early = effect("lips", 70), effect("brows", 30)
    late.log = early.log = order
    pipeline, _, _ = make_pipeline([late, early], settings_with())
    result = run(pipeline, FADE_FRAMES)[-1]
    assert order[-2:] == ["brows", "lips"]
    expected = np.clip(sample_frame() / 255.0 + 0.2, 0, 1) * 255
    np.testing.assert_allclose(result.output, expected, atol=1)
    assert result.face_found and result.output.dtype == np.uint8


def test_failing_effect_is_isolated_and_logged_once(caplog: pytest.LogCaptureFixture) -> None:
    broken, healthy = effect("lips", 70, fail=RuntimeError("boom")), effect("blush", 60)
    pipeline, _, _ = make_pipeline([broken, healthy], settings_with())
    with caplog.at_level(logging.ERROR, logger="makeup_cam.pipeline"):
        results = run(pipeline, 4)
    assert len(healthy.calls) == 4 and len(broken.calls) == 4
    assert all(r.output.shape == (H, W, 3) for r in results)
    assert sum("boom" in record.getMessage() for record in caplog.records) == 1


def test_effect_returning_a_bad_frame_is_skipped(caplog: pytest.LogCaptureFixture) -> None:
    bad = effect("lips", result=np.zeros((2, 2, 3), np.float32))
    pipeline, _, _ = make_pipeline([bad], settings_with())
    with caplog.at_level(logging.ERROR, logger="makeup_cam.pipeline"):
        result = pipeline.process(sample_frame(), 0)
    assert result.output.shape == (H, W, 3)
    np.testing.assert_array_equal(result.output, sample_frame())
    assert any("float32" in record.getMessage() for record in caplog.records)


def test_bypass_passes_frames_through_and_fades_back_in() -> None:
    lips = effect("lips")
    pipeline, _, _ = make_pipeline([lips], settings_with())
    run(pipeline, FADE_FRAMES)
    pipeline.bypass = True
    result = pipeline.process(sample_frame(), 1000)
    assert pipeline.bypass
    np.testing.assert_array_equal(result.output, sample_frame())
    assert result.face_found  # tracking keeps running for the status bar
    calls = len(lips.calls)
    pipeline.bypass = False
    pipeline.process(sample_frame(), 1033)
    assert len(lips.calls) == calls + 1
    assert lips.intensities[-1] == pytest.approx(1 / FADE_FRAMES)


@pytest.mark.parametrize("mirror_output", [False, True])
@pytest.mark.parametrize("mirror_preview", [False, True])
def test_mirroring(mirror_output: bool, mirror_preview: bool) -> None:
    pipeline, _, _ = make_pipeline([], settings_with(mirror_output=mirror_output, mirror_preview=mirror_preview))
    result = pipeline.process(sample_frame(), 0)
    frame, flipped = sample_frame(), cv2.flip(sample_frame(), 1)
    np.testing.assert_array_equal(result.output, flipped if mirror_output else frame)
    np.testing.assert_array_equal(result.preview, flipped if mirror_preview else frame)
    np.testing.assert_array_equal(result.original_preview, flipped if mirror_preview else frame)


def test_landmark_dots_only_on_the_preview() -> None:
    pipeline, _, _ = make_pipeline([], settings_with(show_landmarks=True))
    result = pipeline.process(sample_frame(), 0)
    np.testing.assert_array_equal(result.output, sample_frame())
    assert not np.array_equal(result.preview, sample_frame())
    assert result.landmarks is not None and result.landmarks.shape == (478, 2)


def test_fade_in_and_out_with_the_last_landmarks() -> None:
    moved = FACE + 3.0
    script: list[np.ndarray | None] = [None] + [FACE] * (FADE_FRAMES - 1) + [moved] + [None] * (FADE_FRAMES + 2)
    lips = effect("lips")
    pipeline, _, _ = make_pipeline([lips], settings_with(), tracker=FakeTracker(script))
    results = run(pipeline, len(script))
    steps = [i / FADE_FRAMES for i in range(1, FADE_FRAMES + 1)]
    expected = steps + steps[-2::-1]
    np.testing.assert_allclose(lips.intensities, expected)
    for ctx in lips.calls[FADE_FRAMES:]:
        assert ctx.landmarks is not None and np.array_equal(ctx.landmarks, moved)
    assert [r.face_found for r in results[-3:]] == [False] * 3
    np.testing.assert_array_equal(results[-1].output, sample_frame())
    assert lips.resets >= 1  # temporal state dropped once the face is gone


def test_landmarks_older_than_half_a_second_are_never_used() -> None:
    lips = effect("lips")
    pipeline, _, _ = make_pipeline([lips], settings_with(), tracker=FakeTracker([FACE] * FADE_FRAMES + [None]))
    run(pipeline, FADE_FRAMES)
    calls = len(lips.calls)
    result = pipeline.process(sample_frame(), 33 * FADE_FRAMES + 600)
    assert len(lips.calls) == calls
    np.testing.assert_array_equal(result.output, sample_frame())


def test_no_face_skips_face_effects_but_not_hair() -> None:
    lips, hair = effect("lips"), hair_effect()
    settings = settings_with(intensity=0.8)
    pipeline, _, _ = make_pipeline([lips, hair], settings, tracker=FakeTracker(default=None))
    result = pipeline.process(sample_frame(), 0)
    assert not result.face_found and result.landmarks is None
    assert lips.calls == []
    assert hair.intensities == [pytest.approx(0.8)]
    assert hair.calls[0].hair_mask is not None and hair.calls[0].landmarks is None


def test_without_any_effect_to_run_the_frame_is_untouched() -> None:
    pipeline, _, segmenter = make_pipeline([effect("lips")], settings_with(), tracker=FakeTracker(default=None))
    result = pipeline.process(sample_frame(), 0)
    np.testing.assert_array_equal(result.output, sample_frame())
    assert segmenter.updates == []


@pytest.mark.parametrize(
    ("names", "disabled", "expect_segmentation"),
    [
        (["lips"], [], False),
        (["lips", "hair"], [], True),
        (["lips", "hair"], ["hair"], False),
        (["skin"], [], True),
        (["skin", "hair"], ["skin", "hair"], False),
    ],
)
def test_segmentation_runs_only_for_hair_or_skin(
    names: list[str], disabled: list[str], expect_segmentation: bool
) -> None:
    effects = [hair_effect() if name == "hair" else effect(name) for name in names]
    settings = settings_with()
    for name in disabled:
        getattr(settings, name).enabled = False
    pipeline, _, segmenter = make_pipeline(effects, settings)
    pipeline.process(sample_frame(), 0)
    assert bool(segmenter.updates) == expect_segmentation


def test_hair_effect_waits_for_a_mask() -> None:
    hair = hair_effect()
    pipeline, _, _ = make_pipeline([hair], settings_with(), segmenter=FakeSegmenter(SegMasks()))
    pipeline.process(sample_frame(), 0)
    assert hair.calls == []


def test_disabled_sections_and_zero_intensity_skip_effects() -> None:
    lips, blush = effect("lips"), effect("blush")
    settings = settings_with()
    settings.lips.enabled = False
    pipeline, _, _ = make_pipeline([lips, blush], settings)
    pipeline.process(sample_frame(), 0)
    assert lips.calls == [] and len(blush.calls) == 1
    pipeline.set_settings(settings_with(intensity=0.0))
    pipeline.process(sample_frame(), 33)
    assert len(blush.calls) == 1


def test_effect_without_settings_section_is_reported_and_skipped(caplog: pytest.LogCaptureFixture) -> None:
    orphan = effect("hairstyle")
    pipeline, _, _ = make_pipeline([orphan], settings_with())
    with caplog.at_level(logging.ERROR, logger="makeup_cam.pipeline"):
        run(pipeline, 2)
    assert orphan.calls == []
    assert sum("hairstyle" in record.getMessage() for record in caplog.records) == 1


def test_settings_are_copied_in_and_out() -> None:
    settings = settings_with(intensity=0.7)
    pipeline, _, _ = make_pipeline([], settings)
    settings.general.intensity = 0.1
    assert pipeline.get_settings().general.intensity == pytest.approx(0.7)
    pipeline.get_settings().general.intensity = 0.2
    assert pipeline.get_settings().general.intensity == pytest.approx(0.7)
    pipeline.set_settings(settings)
    assert pipeline.get_settings().general.intensity == pytest.approx(0.1)


def test_model_settings_are_forwarded_when_they_change() -> None:
    pipeline, tracker, segmenter = make_pipeline([], settings_with(landmark_smoothing=0.3, segmentation_interval=3))
    run(pipeline, 3)
    pipeline.set_settings(settings_with(landmark_smoothing=0.9, segmentation_interval=5))
    run(pipeline, 2, start_ms=1000)
    assert tracker.smoothing == [0.3, 0.9]
    assert segmenter.intervals == [3, 5]


def test_settings_can_change_while_processing() -> None:
    pipeline, _, _ = make_pipeline([effect("lips"), hair_effect()], settings_with())
    stop = threading.Event()
    changes: list[int] = []

    def toggle() -> None:
        while not stop.wait(0.0005):
            i = len(changes)
            pipeline.set_settings(settings_with(intensity=(i % 10) / 10, mirror_output=bool(i % 2)))
            pipeline.bypass = i % 3 == 0
            changes.append(i)

    thread = threading.Thread(target=toggle)
    thread.start()
    try:
        results = []
        while len(changes) < 20:
            results += run(pipeline, 10)
    finally:
        stop.set()
        thread.join()
    assert all(r.output.shape == (H, W, 3) for r in results)


def test_timings_cover_every_stage() -> None:
    pipeline, _, _ = make_pipeline([effect("lips"), hair_effect()], settings_with())
    timings = pipeline.process(sample_frame(), 0).timings_ms
    assert {"landmarks", "segmentation", "lips", "hair", "total"} <= set(timings)
    assert all(value >= 0.0 for value in timings.values())
    assert timings["total"] >= timings["landmarks"]


def test_tracker_and_segmenter_failures_are_isolated(caplog: pytest.LogCaptureFixture) -> None:
    tracker, segmenter = FakeTracker(), FakeSegmenter()
    tracker.fail = RuntimeError("tracker down")
    segmenter.fail = RuntimeError("segmenter down")
    hair = hair_effect()
    pipeline, _, _ = make_pipeline([hair], settings_with(), tracker=tracker, segmenter=segmenter)
    with caplog.at_level(logging.ERROR, logger="makeup_cam.pipeline"):
        results = run(pipeline, 3)
    assert not any(r.face_found for r in results)
    assert hair.calls == []  # no mask -> hair skipped, nothing crashed
    messages = [record.getMessage() for record in caplog.records]
    assert sum("tracker down" in m for m in messages) == 1
    assert sum("segmenter down" in m for m in messages) == 1


def test_frame_size_change_resets_effects() -> None:
    lips = effect("lips")
    pipeline, _, _ = make_pipeline([lips], settings_with())
    pipeline.process(sample_frame(), 0)
    resets = lips.resets
    pipeline.process(np.zeros((H * 2, W * 2, 3), np.uint8), 33)
    assert lips.resets > resets
    assert lips.intensities[-1] == pytest.approx(1 / FADE_FRAMES)  # fade restarts


def test_close_releases_models_once_and_then_passes_through() -> None:
    lips = effect("lips")
    pipeline, tracker, segmenter = make_pipeline([lips], settings_with())
    pipeline.close()
    pipeline.close()
    assert tracker.closed == 1 and segmenter.closed == 1
    result = pipeline.process(sample_frame(), 0)
    np.testing.assert_array_equal(result.output, sample_frame())
    assert lips.calls == [] and not result.face_found


def test_invalid_frames_are_rejected() -> None:
    pipeline, _, _ = make_pipeline([], settings_with())
    with pytest.raises(ValueError):
        pipeline.process(np.zeros((H, W), np.uint8))
    with pytest.raises(ValueError):
        pipeline.process(np.zeros((H, W, 3), np.float32))


def test_builtin_effects_load_lazily_only_without_explicit_effects(monkeypatch: pytest.MonkeyPatch) -> None:
    loads: list[int] = []
    blush_cls = type(effect("blush"))

    def fake_loader() -> list[type[Effect]]:
        loads.append(1)
        return [blush_cls]

    monkeypatch.setattr(effects_pkg, "load_builtin_effects", fake_loader)
    make_pipeline([])
    assert loads == []
    pipeline = MakeupPipeline(settings_with(), tracker=FakeTracker(), segmenter=FakeSegmenter())
    assert loads == [1]
    run(pipeline, 1)  # the instantiated default effect runs


# ---------------------------------------------------------------------------
# Real models (skipped when models / scikit-image are unavailable)
# ---------------------------------------------------------------------------


def test_real_models_on_astronaut_portrait(models_dir: Path) -> None:
    if not all((models_dir / spec.filename).is_file() for spec in MODEL_SPECS.values()):
        pytest.skip(f"models not found in {models_dir}")
    data = pytest.importorskip("skimage.data")
    frame = cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR)
    pipeline = MakeupPipeline(settings_with(), model_dir=models_dir, effects=[])
    try:
        results = [pipeline.process(frame, 33 * i) for i in range(3)]
    finally:
        pipeline.close()
    result = results[-1]
    assert result.face_found and result.landmarks is not None
    landmarks = result.landmarks
    assert np.all((landmarks >= 0) & (landmarks < (frame.shape[1], frame.shape[0])))
    assert landmarks[[33, 263], 1].mean() < landmarks[1, 1] < landmarks[13, 1]
    assert 40 < np.linalg.norm(landmarks[33] - landmarks[263]) < 90
    np.testing.assert_array_equal(result.output, frame)  # no effects -> untouched
