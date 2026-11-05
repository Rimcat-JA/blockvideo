"""MakeupPipeline with fake tracker / segmenter / effects (+ optional real models)."""

from __future__ import annotations

import itertools
import logging
import os
import sys
import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from makeup_cam import background as plates
from makeup_cam import effects as effects_pkg
from makeup_cam import pipeline as pipeline_module
from makeup_cam.config import LipSettings, MakeupSettings
from makeup_cam.effects.base import Effect, FrameContext
from makeup_cam.models import MODEL_SPECS
from makeup_cam.pipeline import FADE_FRAMES, SETTINGS_GLIDE_FRAMES, MakeupPipeline
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


@pytest.mark.parametrize("name", ["contour", "reshape"])
def test_optional_mask_users_get_masks_without_skin_or_hair(name: str) -> None:
    pipeline, _, segmenter = make_pipeline([effect(name)], settings_with())
    pipeline.process(sample_frame(), 0)
    assert segmenter.updates


def test_builtin_effects_are_registered_in_their_order() -> None:
    effects = effects_pkg.load_builtin_effects()
    assert [(cls.order, cls.name) for cls in effects if cls.__module__.startswith("makeup_cam.effects.")] == [
        (5, "reshape"),
        (8, "hairshape"),
        (10, "hair"),
        (20, "skin"),
        (25, "contour"),
        (30, "brows"),
        (40, "eyeshadow"),
        (45, "tearbag"),
        (50, "eyeliner"),
        (60, "blush"),
        (70, "lips"),
        (75, "teeth"),
    ]


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
# Background plate
# ---------------------------------------------------------------------------


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A private user data folder: the user's real plate is never read or written."""
    folder = tmp_path / "home"
    monkeypatch.setenv("AI_MAKEUP_CAM_HOME", str(folder))
    return folder


def room_frame(value: int = 200) -> np.ndarray:
    return np.full((H, W, 3), value, np.uint8)


def empty_room(_: np.ndarray) -> float:
    return 0.0


def run_once(pipeline: MakeupPipeline, lips: FakeEffect, ts: int = 0, frame: np.ndarray | None = None) -> FrameContext:
    pipeline.process(sample_frame() if frame is None else frame, ts)
    return lips.calls[-1]


def test_without_a_plate_effects_get_none(home: Path) -> None:
    lips = effect("lips")
    pipeline, _, _ = make_pipeline([lips], settings_with())
    assert run_once(pipeline, lips).background is None
    assert not pipeline.has_background


def test_a_saved_plate_is_used_as_float32_and_only_read(home: Path) -> None:
    plates.save_plate(room_frame(51))
    lips = effect("lips")
    pipeline, _, _ = make_pipeline([lips], settings_with())
    plate = run_once(pipeline, lips).background
    assert plate is not None and plate.dtype == np.float32 and plate.shape == (H, W, 3)
    np.testing.assert_allclose(plate, 0.2, atol=1e-6)
    assert not plate.flags.writeable
    assert pipeline.has_background
    assert run_once(pipeline, lips, 33).background is plate  # converted once, not per frame


def test_a_plate_is_invalid_when_the_resolution_changes(home: Path) -> None:
    plates.save_plate(room_frame())
    lips = effect("lips")
    pipeline, _, _ = make_pipeline([lips], settings_with())
    assert run_once(pipeline, lips).background is not None
    bigger = np.zeros((H * 2, W * 2, 3), np.uint8)
    assert run_once(pipeline, lips, 33, bigger).background is None
    assert not pipeline.has_background
    assert run_once(pipeline, lips, 66).background is not None  # back at its own resolution


def test_capture_background_saves_and_starts_using_the_plate(home: Path) -> None:
    lips = effect("lips")
    pipeline, _, _ = make_pipeline([lips], settings_with(), person_probe=empty_room)
    run_once(pipeline, lips)
    result = pipeline.capture_background([room_frame()] * plates.CAPTURE_FRAMES)
    assert result.ok and result.message == plates.SUCCESS_MESSAGE
    assert pipeline.has_background
    assert plates.plate_path(W, H).is_file()
    np.testing.assert_allclose(run_once(pipeline, lips, 33).background, 200 / 255, atol=1e-6)


def test_a_rejected_capture_keeps_the_previous_plate(home: Path) -> None:
    plates.save_plate(room_frame(100))
    lips = effect("lips")
    pipeline, _, _ = make_pipeline([lips], settings_with(), person_probe=lambda _: 0.5)
    before = run_once(pipeline, lips).background
    result = pipeline.capture_background([room_frame()] * plates.CAPTURE_FRAMES)
    assert not result.ok and "人" in result.message
    assert run_once(pipeline, lips, 33).background is before
    np.testing.assert_array_equal(plates.load_plate(W, H), room_frame(100))


def test_a_capture_at_another_resolution_is_saved_but_not_used(home: Path) -> None:
    lips = effect("lips")
    pipeline, _, _ = make_pipeline([lips], settings_with(), person_probe=empty_room)
    run_once(pipeline, lips)
    other = np.full((H * 2, W * 2, 3), 90, np.uint8)
    assert pipeline.capture_background([other] * plates.CAPTURE_FRAMES).ok
    assert run_once(pipeline, lips, 33).background is None
    assert plates.plate_path(W * 2, H * 2).is_file()


def test_clear_background(home: Path) -> None:
    plates.save_plate(room_frame())
    lips = effect("lips")
    pipeline, _, _ = make_pipeline([lips], settings_with())
    run_once(pipeline, lips)
    result = pipeline.clear_background()
    assert result.ok and result.message == plates.CLEARED_MESSAGE
    assert not pipeline.has_background and not plates.plate_path(W, H).exists()
    assert run_once(pipeline, lips, 33).background is None


def test_a_plate_that_cannot_be_saved_is_used_until_the_app_closes(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(_: np.ndarray) -> Path:
        raise OSError("disk full")

    monkeypatch.setattr(plates, "save_plate", refuse)
    lips = effect("lips")
    pipeline, _, _ = make_pipeline([lips], settings_with(), person_probe=empty_room)
    run_once(pipeline, lips)
    result = pipeline.capture_background([room_frame()] * plates.CAPTURE_FRAMES)
    assert result.ok and result.message == plates.SUCCESS_NOT_SAVED_MESSAGE
    assert run_once(pipeline, lips, 33).background is not None


def test_a_plate_file_that_cannot_be_read_only_means_no_plate(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(width: int, height: int) -> np.ndarray:
        raise RuntimeError("cannot read")

    monkeypatch.setattr(plates, "load_plate", broken)
    lips = effect("lips")
    pipeline, _, _ = make_pipeline([lips], settings_with())
    assert run_once(pipeline, lips).background is None


def test_capture_uses_its_own_model_and_closes_it(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    closed: list[bool] = []

    class FakeProbe:
        def __init__(self, path: Path) -> None:
            self.path = path

        def __call__(self, frame: np.ndarray) -> float:
            return 0.0

        def close(self) -> None:
            closed.append(True)

    monkeypatch.setattr(pipeline_module, "ensure_model", lambda name, directory=None: Path("model.tflite"))
    monkeypatch.setattr(plates, "MediaPipePersonProbe", FakeProbe)
    pipeline, _, _ = make_pipeline([], settings_with())
    assert pipeline.capture_background([room_frame()] * plates.CAPTURE_FRAMES).ok
    assert closed == [True]


def test_capture_never_raises_when_the_model_is_unavailable(
    home: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def fail(name: str, directory: Path | None = None) -> Path:
        raise RuntimeError("no model")

    monkeypatch.setattr(pipeline_module, "ensure_model", fail)
    pipeline, _, _ = make_pipeline([], settings_with())
    with caplog.at_level(logging.ERROR, logger="makeup_cam.pipeline"):
        result = pipeline.capture_background([room_frame()] * plates.CAPTURE_FRAMES)
    assert not result.ok and result.message == plates.CHECK_FAILED_MESSAGE
    assert any("no model" in record.getMessage() for record in caplog.records)
    assert not pipeline.has_background


def test_capturing_and_clearing_while_processing(home: Path) -> None:
    lips = effect("lips")
    pipeline, _, _ = make_pipeline([lips], settings_with(), person_probe=empty_room)
    stop = threading.Event()
    rounds: list[int] = []

    def churn() -> None:
        while not stop.is_set():
            pipeline.capture_background([room_frame()] * plates.MIN_FRAMES)
            pipeline.clear_background()
            rounds.append(1)

    thread = threading.Thread(target=churn)
    thread.start()
    try:
        results = []
        while len(rounds) < 5:
            results += run(pipeline, 10, start_ms=33 * len(results))
    finally:
        stop.set()
        thread.join()
    assert all(r.output.shape == (H, W, 3) for r in results)
    assert all(ctx.background is None or ctx.background.shape == (H, W, 3) for ctx in lips.calls)


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


def test_deleting_during_a_capture_wins(home: Path) -> None:
    """The room's picture is gone once the user was told so, even if a capture was still running."""
    calls: list[int] = []

    def clearing_probe(_: np.ndarray) -> float:
        if not calls:
            assert pipeline.clear_background().ok
        calls.append(1)
        return 0.0

    pipeline = MakeupPipeline(
        tracker=FakeTracker(), segmenter=FakeSegmenter(), effects=[effect("lips", 1)], person_probe=clearing_probe
    )
    result = pipeline.capture_background([room_frame()] * plates.CAPTURE_FRAMES)
    assert not result.ok and result.message == plates.CAPTURE_CANCELLED_MESSAGE
    assert result.plate is None
    assert not pipeline.has_background
    assert not home.exists() or list(home.iterdir()) == []


def test_deleting_during_the_countdown_cancels_a_capture_started_afterwards(home: Path) -> None:
    pipeline = MakeupPipeline(
        tracker=FakeTracker(), segmenter=FakeSegmenter(), effects=[effect("lips", 1)], person_probe=empty_room
    )
    requested = pipeline.background_generation
    pipeline.clear_background()
    assert not pipeline.capture_background([room_frame()] * plates.CAPTURE_FRAMES, requested).ok
    assert not pipeline.has_background and (not home.exists() or list(home.iterdir()) == [])
    # A new request after the deletion is fine, with or without naming when it was made.
    assert pipeline.capture_background([room_frame()] * plates.CAPTURE_FRAMES, pipeline.background_generation).ok
    assert pipeline.capture_background([room_frame()] * plates.CAPTURE_FRAMES).ok


def test_a_plate_that_cannot_be_saved_survives_a_change_of_resolution(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """It is promised for as long as the app runs, and an older file must not silently come back."""
    plates.save_plate(room_frame(40))
    lips = effect("lips", 1)
    pipeline = MakeupPipeline(tracker=FakeTracker(), segmenter=FakeSegmenter(), effects=[lips], person_probe=empty_room)
    run_once(pipeline, lips, 0)

    def refuse(_: np.ndarray) -> Path:
        raise OSError("read-only")

    monkeypatch.setattr(plates, "save_plate", refuse)
    result = pipeline.capture_background([room_frame(200)] * plates.CAPTURE_FRAMES)
    assert result.message == plates.SUCCESS_NOT_SAVED_MESSAGE
    smaller = room_frame()[: H // 2, : W // 2]
    run_once(pipeline, lips, 33, smaller)
    assert not pipeline.has_background
    ctx = run_once(pipeline, lips, 66)
    assert ctx.background is not None and abs(float(ctx.background.mean()) - 200 / 255) < 1e-3
    # Deleting also forgets it.
    pipeline.clear_background()
    run_once(pipeline, lips, 99, smaller)
    assert run_once(pipeline, lips, 132).background is None


# ---------------------------------------------------------------------------
# Speed: reused float buffer, tracking ahead, thread settings
# ---------------------------------------------------------------------------


def every_value_frame() -> np.ndarray:
    values = np.arange(256, dtype=np.uint8).reshape(16, 16)
    return np.dstack([values, values[::-1], values.T])


def test_float_table_matches_the_formula_and_survives_the_round_trip() -> None:
    values = np.arange(256, dtype=np.uint8)
    expected = np.multiply(values, np.float32(1.0 / 255.0), dtype=np.float32)
    assert pipeline_module._TO_FLOAT.dtype == np.float32
    np.testing.assert_array_equal(pipeline_module._TO_FLOAT.ravel(), expected)
    keep = effect("lips", delta=0.0)
    pipeline, _, _ = make_pipeline([keep], settings_with())
    frame = every_value_frame()
    np.testing.assert_array_equal(pipeline.process(frame, 0).output, frame)


def test_effects_get_float32_values_of_the_current_frame() -> None:
    seen: list[np.ndarray] = []

    class Recorder(FakeEffect):
        def apply(self, frame: np.ndarray, ctx: FrameContext, settings: Any) -> np.ndarray:
            assert frame.dtype == np.float32
            seen.append(frame.copy())
            return frame

    pipeline, _, _ = make_pipeline([Recorder()], settings_with())
    first = every_value_frame()
    second = first[::-1].copy()
    for index, frame in enumerate((first, second)):
        pipeline.process(frame, index * 33)
        np.testing.assert_array_equal(seen[index], np.multiply(frame, np.float32(1.0 / 255.0), dtype=np.float32))


def test_what_an_effect_did_never_reaches_the_next_frame() -> None:
    """The buffer is reused: toggled off, partly written and failing effects must leave no trace."""

    class Scribbler(FakeEffect):
        def apply(self, frame: np.ndarray, ctx: FrameContext, settings: Any) -> np.ndarray:
            frame[:] = 0.75
            if self.fail:
                raise self.fail
            return frame

    scribble = Scribbler()
    keep = effect("blush", 60, delta=0.0)
    settings = settings_with()
    pipeline, _, _ = make_pipeline([scribble, keep], settings)
    frame = sample_frame()
    assert not np.array_equal(pipeline.process(frame, 0).output, frame)

    settings.lips.enabled = False
    pipeline.set_settings(settings)
    np.testing.assert_array_equal(pipeline.process(frame, 33).output, frame)

    settings.lips.enabled = True
    pipeline.set_settings(settings)
    scribble.fail = RuntimeError("after writing")
    partly = pipeline.process(frame, 66).output
    assert not np.array_equal(partly, frame)  # as ever: an in-place effect may have written before failing
    settings.lips.enabled = False
    pipeline.set_settings(settings)
    np.testing.assert_array_equal(pipeline.process(frame, 99).output, frame)


def test_results_stay_valid_while_later_frames_are_processed() -> None:
    pipeline, _, _ = make_pipeline([effect("lips", delta=0.1)], settings_with())
    frame = sample_frame()
    first = pipeline.process(frame, 0)
    kept = first.output.copy()
    assert not np.shares_memory(first.output, pipeline._float_frame)
    pipeline.process(frame // 2, 33)
    np.testing.assert_array_equal(first.output, kept)


def test_an_effect_may_return_a_new_array() -> None:
    new = np.full((H, W, 3), 0.25, np.float32)
    effects = [effect("lips", needs_face=False, result=new), effect("blush", 60, needs_face=False, delta=0.25)]
    pipeline, _, _ = make_pipeline(effects, settings_with())
    for ts in (0, 33, 66):
        output = pipeline.process(sample_frame(), ts).output
        assert np.all(output == 128)
        new[:] = 0.25  # the second effect wrote into it
    assert pipeline._float_frame is not None and not np.shares_memory(new, pipeline._float_frame)


def test_the_buffer_follows_the_frame_size() -> None:
    lips = effect("lips", delta=0.0)
    pipeline, _, _ = make_pipeline([lips], settings_with())
    small = sample_frame()
    large = cv2.resize(small, (W * 2, H * 3), interpolation=cv2.INTER_NEAREST)
    for index, frame in enumerate((small, large, small, small, large)):
        result = pipeline.process(frame, index * 33)
        np.testing.assert_array_equal(result.output, frame)
        assert pipeline._float_frame is not None and pipeline._float_frame.shape == frame.shape
    buffer = pipeline._float_frame
    pipeline.process(large, 200)
    assert pipeline._float_frame is buffer


def test_no_effect_needs_no_buffer() -> None:
    settings = settings_with()
    settings.lips.enabled = False
    pipeline, _, _ = make_pipeline([effect("lips")], settings)
    frame = sample_frame()
    assert pipeline.process(frame, 0).output is frame
    assert pipeline._float_frame is None


def test_the_mirrored_previews_can_be_displayed() -> None:
    settings = settings_with(mirror_preview=True)
    pipeline, _, _ = make_pipeline([effect("lips", delta=0.1)], settings)
    frame = sample_frame()
    frame.setflags(write=False)
    result = pipeline.process(frame, 0)
    np.testing.assert_array_equal(np.ascontiguousarray(result.preview), cv2.flip(result.output, 1))
    np.testing.assert_array_equal(np.ascontiguousarray(result.original_preview), cv2.flip(frame, 1))
    assert cv2.resize(result.original_preview, (W // 2, H // 2)).shape == (H // 2, W // 2, 3)  # a view works too


@pytest.fixture
def cv2_threads() -> Any:
    before = cv2.getNumThreads()
    yield
    cv2.setNumThreads(before)


def test_opencv_threads_are_limited_but_never_raised(cv2_threads: None) -> None:
    cv2.setNumThreads(pipeline_module.CV2_THREADS + 4)
    make_pipeline([])
    assert cv2.getNumThreads() == pipeline_module.CV2_THREADS
    cv2.setNumThreads(1)
    make_pipeline([])
    assert cv2.getNumThreads() == 1


def _process_affinity() -> int:
    import ctypes

    kernel32 = ctypes.WinDLL("kernel32")
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    kernel32.GetProcessAffinityMask.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_size_t),
    ]
    allowed, system = ctypes.c_size_t(), ctypes.c_size_t()
    assert kernel32.GetProcessAffinityMask(kernel32.GetCurrentProcess(), ctypes.byref(allowed), ctypes.byref(system))
    return allowed.value


@pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
def test_one_hardware_thread_per_core_stays_within_what_is_allowed() -> None:
    import ctypes

    kernel32 = ctypes.WinDLL("kernel32")
    first = pipeline_module._first_thread_of_each_core(kernel32)
    assert first != 0
    assert bin(first).count("1") <= (os.cpu_count() or 1)
    pipeline_module._use_one_thread_per_core()
    allowed = _process_affinity()
    assert allowed != 0 and allowed & first == allowed
    pipeline_module._use_one_thread_per_core()  # a second pipeline changes nothing
    assert _process_affinity() == allowed


def test_the_affinity_is_never_a_reason_to_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*_: Any, **__: Any) -> Any:
        raise OSError("no such library")

    monkeypatch.setattr(pipeline_module.ctypes, "WinDLL", refuse, raising=False)
    pipeline_module._use_one_thread_per_core()


class ThreadCheckingTracker(FakeTracker):
    """Landmarks depend on the frame's first pixel; fails the test if two calls overlap."""

    def __init__(self) -> None:
        super().__init__()
        self.running = threading.Lock()
        self.overlapped = False
        self.calls: list[tuple[int, int, str]] = []
        self.gate = threading.Event()
        self.gate.set()

    def _enter(self) -> None:
        if not self.running.acquire(blocking=False):
            self.overlapped = True
            self.running.acquire()

    def detect(self, frame: np.ndarray, ts_ms: int) -> tuple[np.ndarray | None, np.ndarray | None]:
        self._enter()
        try:
            self.gate.wait(5.0)
            self.calls.append((id(frame), ts_ms, threading.current_thread().name))
            if self.fail:
                raise self.fail
            return FACE + float(frame[0, 0, 0]), np.zeros(len(FACE), np.float32)
        finally:
            self.running.release()

    def set_smoothing(self, smoothing: float) -> None:
        self._enter()
        try:
            self.smoothing.append(smoothing)
            self.calls.append((0, -1, threading.current_thread().name))
        finally:
            self.running.release()


def value_frame(value: int) -> np.ndarray:
    return np.full((H, W, 3), value, np.uint8)


def test_a_prepared_frame_is_tracked_once_on_its_own_thread() -> None:
    tracker = ThreadCheckingTracker()
    lips = effect("lips", delta=0.0)
    pipeline, _, _ = make_pipeline([lips], settings_with(), tracker=tracker)
    frame = value_frame(7)
    pipeline.prepare(frame, 100)
    result = pipeline.process(frame, 100)
    pipeline.close()
    detections = [call for call in tracker.calls if call[1] >= 0]
    assert detections == [(id(frame), 100, detections[0][2])]
    assert detections[0][2].startswith("tracker")
    np.testing.assert_array_equal(result.landmarks, FACE + 7.0)
    assert not tracker.overlapped


@pytest.mark.parametrize("fail_at", [None, 3])
def test_preparing_changes_nothing_but_the_thread(fail_at: int | None, caplog: pytest.LogCaptureFixture) -> None:
    def run_all(prepare: bool) -> list[tuple[np.ndarray, np.ndarray | None, list[float]]]:
        tracker = ThreadCheckingTracker()
        lips = effect("lips", delta=0.05)
        pipeline, _, _ = make_pipeline([lips], settings_with(), tracker=tracker)
        outcome = []
        for index in range(8):
            frame = value_frame(10 * index)
            tracker.fail = RuntimeError("no model") if index == fail_at else None
            if prepare:
                pipeline.prepare(frame, index * 33)
            result = pipeline.process(frame, index * 33)
            outcome.append((result.output.copy(), result.landmarks, list(lips.intensities)))
        pipeline.close()
        return outcome

    for plain, prepared in zip(run_all(False), run_all(True), strict=True):
        np.testing.assert_array_equal(plain[0], prepared[0])
        np.testing.assert_array_equal(plain[1], prepared[1])
        assert plain[2] == prepared[2]


def test_frames_the_renderer_skips_are_forgotten() -> None:
    tracker = ThreadCheckingTracker()
    pipeline, _, _ = make_pipeline([], settings_with(), tracker=tracker)
    frames = [value_frame(value) for value in (1, 2, 3)]
    futures = []
    for index, frame in enumerate(frames):
        pipeline.prepare(frame, index)
        futures.append(pipeline._track_ahead._jobs[-1][2])
        futures[-1].result(5.0)
    result = pipeline.process(frames[2], 2)
    np.testing.assert_array_equal(result.landmarks, FACE + 3.0)
    assert not pipeline._track_ahead._jobs
    # An older frame that shows up late is simply tracked again, in turn.
    assert pipeline.process(frames[0], 0).landmarks is not None
    assert len([call for call in tracker.calls if call[1] >= 0]) == 4
    pipeline.close()
    assert not tracker.overlapped


def test_a_busy_tracker_is_not_queued_behind() -> None:
    tracker = ThreadCheckingTracker()
    pipeline, _, _ = make_pipeline([], settings_with(), tracker=tracker)
    pipeline.process(value_frame(0), 1)  # the first process() also forwards the settings, which would wait
    tracker.gate.clear()
    first, second = value_frame(1), value_frame(2)
    pipeline.prepare(first, 10)
    pipeline.prepare(second, 33)  # the first is still running: not started now
    assert len(pipeline._track_ahead._jobs) == 1
    # Tracked when asked for, after the running one, never at the same time.
    results: list[Any] = []
    renderer = threading.Thread(target=lambda: results.append(pipeline.process(second, 33)))
    renderer.start()
    renderer.join(0.05)
    assert renderer.is_alive()
    tracker.gate.set()
    renderer.join(5.0)
    np.testing.assert_array_equal(results[0].landmarks, FACE + 2.0)
    pipeline.close()
    assert not tracker.overlapped
    assert [call[1] for call in tracker.calls if call[1] >= 0] == [1, 10, 33]


def test_another_frame_or_timestamp_is_not_taken_for_the_announced_one() -> None:
    tracker = ThreadCheckingTracker()
    pipeline, _, _ = make_pipeline([], settings_with(), tracker=tracker)
    announced = value_frame(5)
    pipeline.prepare(announced, 10)
    pipeline._track_ahead._jobs[-1][2].result(5.0)
    np.testing.assert_array_equal(pipeline.process(value_frame(9), 10).landmarks, FACE + 9.0)
    np.testing.assert_array_equal(pipeline.process(announced, 11).landmarks, FACE + 5.0)
    pipeline.close()
    assert len([call for call in tracker.calls if call[1] >= 0]) == 3


def test_smoothing_changes_wait_for_the_running_detection() -> None:
    tracker = ThreadCheckingTracker()
    settings = settings_with(landmark_smoothing=0.2)
    pipeline, _, _ = make_pipeline([], settings, tracker=tracker)
    frame = value_frame(1)
    pipeline.prepare(frame, 0)
    pipeline.process(frame, 0)
    tracker.gate.clear()
    pipeline.prepare(frame, 33)  # still running when process() forwards the new value
    settings.general.landmark_smoothing = 0.8
    pipeline.set_settings(settings)
    renderer = threading.Thread(target=pipeline.process, args=(frame, 33))
    renderer.start()
    renderer.join(0.05)
    assert renderer.is_alive() and tracker.smoothing == [0.2]
    tracker.gate.set()
    renderer.join(5.0)
    pipeline.close()
    assert tracker.smoothing == [0.2, 0.8]
    assert not tracker.overlapped


def test_fade_out_works_with_prepared_frames() -> None:
    tracker = FakeTracker(script=[FACE] * 8 + [None] * 8)
    lips = effect("lips", delta=0.0)
    pipeline, _, _ = make_pipeline([lips], settings_with(), tracker=tracker)
    for index in range(16):
        frame = sample_frame()
        pipeline.prepare(frame, index * 33)
        pipeline.process(frame, index * 33)
    pipeline.close()
    levels = [round(value * FADE_FRAMES) for value in lips.intensities]
    assert levels[:6] == [1, 2, 3, 4, 5, 6]
    assert levels[-5:] == [5, 4, 3, 2, 1]
    assert lips.resets == 2  # the first frame size, and the lost face


def test_prepare_never_raises_and_does_nothing_after_close(caplog: pytest.LogCaptureFixture) -> None:
    tracker = ThreadCheckingTracker()
    pipeline, _, _ = make_pipeline([], settings_with(), tracker=tracker)
    pipeline.prepare("not a frame", 0)  # type: ignore[arg-type]
    pipeline.prepare(np.zeros((4, 4), np.uint8), 0)
    pipeline.prepare(np.zeros((4, 4, 3), np.float32), 0)
    assert not pipeline._track_ahead._jobs
    pipeline.close()
    pipeline.prepare(value_frame(1), 0)
    assert tracker.calls == []
    assert pipeline.process(value_frame(1), 0).landmarks is None


def test_close_waits_for_a_running_detection() -> None:
    tracker = ThreadCheckingTracker()
    tracker.gate.clear()
    pipeline, _, _ = make_pipeline([], settings_with(), tracker=tracker)
    pipeline.prepare(value_frame(1), 0)
    closer = threading.Thread(target=pipeline.close)
    closer.start()
    closer.join(0.05)
    assert closer.is_alive() and tracker.closed == 0
    tracker.gate.set()
    closer.join(5.0)
    assert not closer.is_alive() and tracker.closed == 1
    assert len(tracker.calls) == 1


def test_prepare_and_process_from_two_threads_at_full_speed() -> None:
    tracker = ThreadCheckingTracker()
    lips = effect("lips", delta=0.0)
    pipeline, _, _ = make_pipeline([lips], settings_with(), tracker=tracker)
    latest: list[tuple[np.ndarray, int]] = []
    stop = threading.Event()

    def camera() -> None:
        ts = 0
        while not stop.is_set():
            frame = value_frame(ts % 251)
            pipeline.prepare(frame, ts)
            latest.append((frame, ts))
            ts += 1

    thread = threading.Thread(target=camera)
    thread.start()
    processed = 0
    try:
        while processed < 200:
            if latest:
                frame, ts = latest[-1]
                result = pipeline.process(frame, ts)
                np.testing.assert_array_equal(result.landmarks, FACE + float(frame[0, 0, 0]))
                processed += 1
    finally:
        stop.set()
        thread.join()
        pipeline.close()
    assert not tracker.overlapped


# ---------------------------------------------------------------------------
# A lost face leaves nothing behind
# ---------------------------------------------------------------------------


class CoastingTracker(FakeTracker):
    """After a lost face, says where the picture moved it (None: cannot tell)."""

    def __init__(self, script: list[np.ndarray | None], coasts: list[np.ndarray | None]) -> None:
        super().__init__(script)
        self.coasts = coasts

    def coast(self) -> tuple[np.ndarray, np.ndarray] | None:
        where = self.coasts.pop(0)
        return None if where is None else (where.copy(), np.zeros(len(where), np.float32))


def levels(lips: FakeEffect) -> list[int]:
    return [round(value * FADE_FRAMES) for value in lips.intensities]


def test_makeup_follows_the_face_while_it_fades_after_a_loss() -> None:
    moved = [FACE + (0.0, 2.0 * i) for i in range(1, 5)]
    tracker = CoastingTracker([FACE] * FADE_FRAMES + [None] * 4, list(moved))
    lips = effect("lips")
    pipeline, _, _ = make_pipeline([lips], settings_with(), tracker=tracker)
    results = run(pipeline, FADE_FRAMES + 4)
    assert levels(lips)[FADE_FRAMES:] == [5, 4, 3, 2]
    for ctx, where in zip(lips.calls[FADE_FRAMES:], moved, strict=True):
        assert np.array_equal(ctx.landmarks, where)
    assert [r.face_found for r in results[-4:]] == [False] * 4  # only the model says a face is there
    assert results[-1].landmarks is None


def test_a_fast_face_with_no_known_motion_moves_on_and_fades_quickly() -> None:
    speed = np.float32([0.0, 60.0])
    path = [FACE + speed * i for i in range(FADE_FRAMES + 2)]
    tracker = FakeTracker([*path, None, None, None, None])
    lips = effect("lips")
    pipeline, _, _ = make_pipeline([lips], settings_with(), tracker=tracker)
    run(pipeline, FADE_FRAMES + 2)
    lips.calls.clear()
    lips.intensities.clear()
    results = run(pipeline, 4)
    assert levels(lips) == [4, 2]  # no pop, but gone in a moment
    np.testing.assert_allclose(lips.calls[0].landmarks, path[-1] + speed)  # not where the face used to be
    np.testing.assert_allclose(lips.calls[1].landmarks, path[-1] + speed * 2)
    assert np.array_equal(results[-1].output, sample_frame())


def test_a_still_face_with_no_known_motion_still_fades_gently() -> None:
    tracker = FakeTracker([FACE] * FADE_FRAMES + [None] * 3)
    lips = effect("lips")
    pipeline, _, _ = make_pipeline([lips], settings_with(), tracker=tracker)
    run(pipeline, FADE_FRAMES + 3)
    assert levels(lips)[FADE_FRAMES:] == [5, 4, 3]


def test_a_face_found_on_and_off_does_not_make_the_makeup_pulse() -> None:
    from makeup_cam.pipeline import STEADY_FRAMES

    on_off: list[np.ndarray | None] = [FACE, FACE, None] * 3 + [FACE, None, FACE, None]
    script: list[np.ndarray | None] = [FACE] * FADE_FRAMES + [None] * (FADE_FRAMES + 1) + on_off + [FACE] * (STEADY_FRAMES + 2)
    lips = effect("lips")
    pipeline, _, _ = make_pipeline([lips], settings_with(), tracker=FakeTracker(script))
    results = run(pipeline, len(script))
    start = FADE_FRAMES + FADE_FRAMES + 1
    for result in results[start : start + len(on_off)]:
        assert np.array_equal(result.output, sample_frame())  # nothing while the detections come and go
    assert levels(lips)[-3:] == [1, 2, 3][: 3]  # it rises once the face stays
    assert len(lips.intensities) >= 3


def test_the_first_face_fades_in_without_waiting() -> None:
    lips = effect("lips")
    pipeline, _, _ = make_pipeline([lips], settings_with())
    run(pipeline, 3)
    assert levels(lips) == [1, 2, 3]


def test_a_prepared_frame_is_tracked_with_the_settings_of_its_time() -> None:
    tracker = ThreadCheckingTracker()
    order: list[object] = []
    detect, set_smoothing = tracker.detect, tracker.set_smoothing
    tracker.detect = lambda frame, ts: (order.append(ts), detect(frame, ts))[1]  # type: ignore[method-assign]
    tracker.set_smoothing = lambda value: (order.append(value), set_smoothing(value))[1]  # type: ignore[method-assign]
    settings = settings_with(landmark_smoothing=0.2)
    pipeline, _, _ = make_pipeline([], settings, tracker=tracker)
    first, second = value_frame(1), value_frame(2)
    pipeline.prepare(first, 0)
    pipeline.process(first, 0)
    settings.general.landmark_smoothing = 0.8
    pipeline.set_settings(settings)
    pipeline.prepare(second, 33)
    pipeline.process(second, 33)
    pipeline.close()
    assert order.index(0.8) < order.index(33)  # no frame late, as without prepare


# ---------------------------------------------------------------------------
# Choosing a preset glides
# ---------------------------------------------------------------------------


def test_glide_interpolates_numbers_and_colors_and_ends_exactly() -> None:
    from makeup_cam.pipeline import SETTINGS_GLIDE_FRAMES

    old, new = settings_with(), settings_with()
    old.lips.opacity, new.lips.opacity = 0.2, 0.6
    old.lips.color, new.lips.color = "#000000", "#FFFFFF"
    old.lips.finish, new.lips.finish = "matte", "gloss"
    seen: list[LipSettings] = []
    lips = effect("lips")
    apply = lips.apply

    def recording(frame: np.ndarray, ctx: FrameContext, settings: Any) -> np.ndarray:
        seen.append(settings)
        return apply(frame, ctx, settings)

    lips.apply = recording  # type: ignore[method-assign]
    pipeline, _, _ = make_pipeline([lips], old)
    run(pipeline, 1)
    pipeline.glide_to_settings(new)
    run(pipeline, SETTINGS_GLIDE_FRAMES + 1)
    steps = [s.opacity for s in seen[1:]]
    assert all(a < b for a, b in itertools.pairwise(steps[:SETTINGS_GLIDE_FRAMES]))
    assert steps[0] == pytest.approx(0.2 + 0.4 / SETTINGS_GLIDE_FRAMES)
    assert seen[1].finish == "matte" and seen[SETTINGS_GLIDE_FRAMES - 1].finish == "gloss"
    assert seen[-1] == new.lips
    assert seen[-1].color == "#FFFFFF" and seen[1].color not in ("#000000", "#FFFFFF")


def test_glide_fades_a_section_only_one_look_has() -> None:
    from makeup_cam.pipeline import SETTINGS_GLIDE_FRAMES

    off, on = settings_with(), settings_with()
    off.lips.enabled = False
    lips = effect("lips", delta=0.0)
    pipeline, _, _ = make_pipeline([lips], off)
    run(pipeline, FADE_FRAMES)
    pipeline.glide_to_settings(on)
    run(pipeline, SETTINGS_GLIDE_FRAMES)
    expected = [i / SETTINGS_GLIDE_FRAMES for i in range(1, SETTINGS_GLIDE_FRAMES)] + [1.0]
    np.testing.assert_allclose(lips.intensities[-SETTINGS_GLIDE_FRAMES:], expected)

    lips.intensities.clear()
    pipeline.glide_to_settings(off)
    results = run(pipeline, SETTINGS_GLIDE_FRAMES)
    np.testing.assert_allclose(lips.intensities, [1.0 - i / SETTINGS_GLIDE_FRAMES for i in range(1, SETTINGS_GLIDE_FRAMES)])
    assert np.array_equal(results[-1].output, sample_frame())  # exactly the new look at the end


def glide_watcher(start: float) -> tuple[Any, list[float], MakeupSettings]:
    """A pipeline at lips opacity ``start``, the values the lips are drawn with, and its settings."""
    old = settings_with()
    old.lips.opacity = start
    seen: list[float] = []
    lips = effect("lips")
    apply = lips.apply
    lips.apply = lambda frame, ctx, settings: (seen.append(settings.opacity), apply(frame, ctx, settings))[1]  # type: ignore[method-assign]
    pipeline, _, _ = make_pipeline([lips], old)
    return pipeline, seen, old


def test_set_settings_is_immediate() -> None:
    pipeline, seen, new = glide_watcher(0.2)
    new.lips.opacity = 0.6
    pipeline.set_settings(new)
    run(pipeline, 1)
    assert seen == [0.6]


def test_a_slider_during_a_glide_goes_on_from_where_the_picture_is() -> None:
    pipeline, seen, new = glide_watcher(0.2)
    new.lips.opacity = 0.6
    pipeline.glide_to_settings(new)
    run(pipeline, 3)
    other = new.copy()
    other.lips.opacity = 0.7
    pipeline.set_settings(other)
    run(pipeline, SETTINGS_GLIDE_FRAMES + 1)
    steps = np.abs(np.diff(seen))
    assert steps.max() <= (0.7 - 0.2) / SETTINGS_GLIDE_FRAMES + 1e-6, seen
    assert seen[-1] == 0.7
    assert pipeline.get_settings().lips.opacity == 0.7


def test_a_second_preset_during_a_glide_goes_on_from_where_the_picture_is() -> None:
    pipeline, seen, first = glide_watcher(0.2)
    first.lips.opacity = 1.0
    pipeline.glide_to_settings(first)
    run(pipeline, 4)  # half way: 0.6
    second = first.copy()
    second.lips.opacity = 0.2
    pipeline.glide_to_settings(second)
    run(pipeline, SETTINGS_GLIDE_FRAMES + 1)
    assert max(seen) <= 0.6 + 1e-6, seen
    assert np.abs(np.diff(seen)).max() <= (1.0 - 0.2) / SETTINGS_GLIDE_FRAMES + 1e-6, seen
    assert seen[-1] == 0.2


def test_a_section_fading_out_comes_back_from_where_it_was_when_the_choice_changes() -> None:
    on, off = settings_with(), settings_with()
    off.lips.enabled = False
    intensities: list[float] = []
    lips = effect("lips")
    apply = lips.apply
    lips.apply = lambda frame, ctx, settings: (intensities.append(ctx.intensity), apply(frame, ctx, settings))[1]  # type: ignore[method-assign]
    pipeline, _, _ = make_pipeline([lips], on)
    run(pipeline, FADE_FRAMES + 2)  # the fade-in of the face is over
    intensities.clear()
    pipeline.glide_to_settings(off)
    run(pipeline, 4)
    pipeline.glide_to_settings(on)
    run(pipeline, SETTINGS_GLIDE_FRAMES + 1)
    assert np.abs(np.diff(intensities)).max() <= 1.0 / SETTINGS_GLIDE_FRAMES + 1e-6, intensities
    assert min(intensities) >= 0.5 - 1.0 / SETTINGS_GLIDE_FRAMES - 1e-6, intensities
    assert intensities[-1] == 1.0


# ---------------------------------------------------------------------------
# Masks while the face moves fast
# ---------------------------------------------------------------------------


def test_masks_come_every_frame_while_the_face_moves_fast() -> None:
    moved = [FACE + np.float32([0.0, 20.0 * k]) for k in range(1, 4)]
    script = [FACE] * 6 + moved + [moved[-1]] * (pipeline_module._FAST_FOR_MASKS_FRAMES + 3)
    settings = settings_with(segmentation_interval=2)
    pipeline, _, segmenter = make_pipeline([], settings, tracker=FakeTracker(script))
    run(pipeline, 6)
    assert segmenter.intervals == [2]
    run(pipeline, 3, 200)
    assert segmenter.intervals[-1] == 1
    run(pipeline, pipeline_module._FAST_FOR_MASKS_FRAMES + 3, 300)
    assert segmenter.intervals == [2, 1, 2]
    pipeline.close()
