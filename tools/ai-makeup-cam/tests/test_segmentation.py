"""Segmenter scheduling, smoothing and threading (fake model + optional real model)."""

from __future__ import annotations

import itertools
import threading
import time
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import pytest

from makeup_cam.models import MODEL_SPECS, SELFIE_MULTICLASS
from makeup_cam.segmentation import BODY_SKIN, FACE_SKIN, HAIR, SegMasks, Segmenter

FRAME = np.zeros((720, 1280, 3), np.uint8)


class FakeSegModel:
    """Returns constant per-class masks; ``level`` scales them (callable per call)."""

    def __init__(self, level: Callable[[int], float] = lambda call: 1.0) -> None:
        self.level = level
        self.inputs: list[tuple[tuple[int, ...], int]] = []
        self.last_rgb: np.ndarray | None = None
        self.gate = threading.Event()
        self.gate.set()
        self.fail_next = False
        self.closed = 0

    def segment(self, rgb: np.ndarray, ts_ms: int) -> list[np.ndarray]:
        self.gate.wait(5.0)
        call = len(self.inputs)
        self.inputs.append((rgb.shape, ts_ms))
        self.last_rgb = rgb.copy()
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("model exploded")
        scale = self.level(call)
        return [np.full(rgb.shape[:2], scale * c / 10.0, np.float32) for c in range(6)]

    def close(self) -> None:
        self.closed += 1


def make_segmenter(model: FakeSegModel, interval: int = 1, background: bool = False) -> Segmenter:
    return Segmenter(None, interval, model=model, background=background)


def hair_value(masks: SegMasks) -> float:
    assert masks.hair is not None
    return float(masks.hair[0, 0])


def wait_for_masks(segmenter: Segmenter, start_index: int, timeout_s: float = 5.0) -> SegMasks:
    deadline = time.monotonic() + timeout_s
    for index in itertools.count(start_index):
        masks = segmenter.update(FRAME, 33 * index, index)
        if masks.hair is not None:
            return masks
        assert time.monotonic() < deadline, "background run did not finish"
        time.sleep(0.005)
    raise AssertionError("unreachable")


def test_masks_are_full_frame_read_only_and_mapped_by_class() -> None:
    segmenter = make_segmenter(FakeSegModel())
    masks = segmenter.update(FRAME, 0, 0)
    for mask, cls in ((masks.hair, HAIR), (masks.face_skin, FACE_SKIN), (masks.body_skin, BODY_SKIN)):
        assert mask is not None and mask.shape == (720, 1280) and mask.dtype == np.float32
        assert not mask.flags.writeable
        np.testing.assert_allclose(mask, cls / 10.0, atol=1e-6)


def test_model_gets_a_downscaled_rgb_copy() -> None:
    model = FakeSegModel()
    frame = FRAME.copy()
    frame[..., 0] = 255  # blue in BGR
    make_segmenter(model).update(frame, 0, 0)
    rgb = model.last_rgb
    assert rgb is not None and rgb.shape == (216, 384, 3)
    assert rgb[..., 2].min() == 255 and rgb[..., 0].max() == 0


def test_runs_every_interval_frames_and_reuses_masks_between() -> None:
    model = FakeSegModel()
    segmenter = make_segmenter(model, interval=3)
    results = [segmenter.update(FRAME, 33 * i, i) for i in range(7)]
    assert len(model.inputs) == 3  # frames 0, 3, 6
    assert results[1] is results[0] and results[2] is results[0]
    assert results[3] is not results[0]


def test_interval_change_and_clamping() -> None:
    segmenter = make_segmenter(FakeSegModel(), interval=4)
    segmenter.set_interval(0)
    assert segmenter.interval == 1


def test_ema_blends_consecutive_runs() -> None:
    segmenter = make_segmenter(FakeSegModel(lambda call: 10.0 if call == 0 else 0.0))
    values = [hair_value(segmenter.update(FRAME, 33 * i, i)) for i in range(3)]
    np.testing.assert_allclose(values, [1.0, 0.5, 0.25], atol=1e-6)


def test_masks_from_before_an_idle_pause_are_dropped() -> None:
    segmenter = make_segmenter(FakeSegModel(lambda call: 10.0 if call == 0 else 0.0), interval=5)
    segmenter.update(FRAME, 0, 0)
    assert hair_value(segmenter.update(FRAME, 2000, 1)) == 0.0  # fresh run, nothing blended


def test_a_run_in_flight_during_an_idle_pause_is_discarded() -> None:
    model = FakeSegModel(lambda call: 10.0 if call == 0 else 0.0)
    model.gate.clear()
    segmenter = make_segmenter(model, background=True)
    try:
        segmenter.update(FRAME, 0, 0)
        model.gate.set()
        time.sleep(0.05)  # the old run finishes while nobody asks for masks
        assert segmenter.update(FRAME, 5000, 1).hair is None
        assert hair_value(wait_for_masks(segmenter, 2)) == 0.0
    finally:
        segmenter.close()


def test_reset_clears_masks_and_history() -> None:
    segmenter = make_segmenter(FakeSegModel(lambda call: 10.0 if call == 0 else 0.0), interval=5)
    segmenter.update(FRAME, 0, 0)
    segmenter.reset()
    assert hair_value(segmenter.update(FRAME, 33, 1)) == 0.0  # runs at once, nothing blended


def test_frame_size_change_resets() -> None:
    segmenter = make_segmenter(FakeSegModel(lambda call: 10.0 if call == 0 else 0.0), interval=5)
    segmenter.update(FRAME, 0, 0)
    masks = segmenter.update(np.zeros((480, 640, 3), np.uint8), 33, 1)
    assert masks.hair is not None and masks.hair.shape == (480, 640)
    assert hair_value(masks) == 0.0


def test_frame_counter_restart_triggers_a_run() -> None:
    model = FakeSegModel()
    segmenter = make_segmenter(model, interval=4)
    segmenter.update(FRAME, 0, 10)
    segmenter.update(FRAME, 33, 0)
    assert len(model.inputs) == 2


def test_timestamps_strictly_increase() -> None:
    model = FakeSegModel()
    segmenter = make_segmenter(model)
    for i, ts in enumerate((100, 100, 40, 300)):
        segmenter.update(FRAME, ts, i)
    stamps = [ts for _, ts in model.inputs]
    assert all(b > a for a, b in itertools.pairwise(stamps))


def test_background_run_does_not_block_and_delivers_later() -> None:
    model = FakeSegModel()
    model.gate.clear()
    segmenter = make_segmenter(model, background=True)
    try:
        started = time.perf_counter()
        assert segmenter.update(FRAME, 0, 0).hair is None
        assert time.perf_counter() - started < 1.0
        model.gate.set()
        assert hair_value(wait_for_masks(segmenter, 1)) == pytest.approx(0.1)
    finally:
        segmenter.close()


def test_background_failure_is_raised_once_then_recovers() -> None:
    model = FakeSegModel()
    model.fail_next = True
    segmenter = make_segmenter(model, background=True)
    try:
        segmenter.update(FRAME, 0, 0)
        deadline = time.monotonic() + 5.0
        with pytest.raises(RuntimeError, match="exploded"):
            for index in itertools.count(1):
                segmenter.update(FRAME, 33 * index, index)
                assert time.monotonic() < deadline
                time.sleep(0.005)
        assert wait_for_masks(segmenter, 100).hair is not None
    finally:
        segmenter.close()


def test_reset_discards_a_run_in_flight() -> None:
    model = FakeSegModel(lambda call: 10.0 if call == 0 else 0.0)
    model.gate.clear()
    segmenter = make_segmenter(model, background=True)
    try:
        segmenter.update(FRAME, 0, 0)
        segmenter.reset()
        model.gate.set()
        assert hair_value(wait_for_masks(segmenter, 1)) == 0.0  # the pre-reset result never shows
    finally:
        segmenter.close()


def test_close_closes_the_model_once() -> None:
    model = FakeSegModel()
    segmenter = make_segmenter(model, background=True)
    segmenter.close()
    segmenter.close()
    assert model.closed == 1


# ---------------------------------------------------------------------------
# Real MediaPipe model (skipped when models / scikit-image are unavailable)
# ---------------------------------------------------------------------------


def test_real_model_finds_hair_and_face_skin(models_dir: Path) -> None:
    path = models_dir / MODEL_SPECS[SELFIE_MULTICLASS].filename
    if not path.is_file():
        pytest.skip(f"{path.name} not found in {models_dir}")
    data = pytest.importorskip("skimage.data")
    frame = cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR)
    segmenter = Segmenter(path, interval=1, background=False)
    try:
        masks = segmenter.update(frame, 0, 0)
    finally:
        segmenter.close()
    assert masks.hair is not None and masks.face_skin is not None
    assert masks.hair.shape == frame.shape[:2]
    assert masks.hair[30:50, 200:240].mean() > 0.5  # top of the head
    assert masks.face_skin[120:140, 205:245].mean() > 0.5  # cheeks / nose
    assert masks.hair[400:, :].mean() < 0.1  # suit and helmet are not hair
