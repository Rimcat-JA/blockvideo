"""One Euro filter, smoothing mapping and FaceTracker (fake model + optional real model)."""

from __future__ import annotations

import itertools
from pathlib import Path

import cv2
import numpy as np
import pytest

from makeup_cam.models import FACE_LANDMARKER, MODEL_SPECS
from makeup_cam.tracking import (
    EYE_OUTER_CORNERS,
    NUM_LANDMARKS,
    DetectionWindow,
    FaceTracker,
    OneEuroFilter,
    follow_face,
    search_windows,
    smoothing_params,
)

FPS = 30.0

# ---------------------------------------------------------------------------
# One Euro filter
# ---------------------------------------------------------------------------


def run_filter(f: OneEuroFilter, samples: np.ndarray) -> np.ndarray:
    return np.array([f(np.atleast_1d(x), i / FPS) for i, x in enumerate(samples)])


def test_first_sample_passes_through() -> None:
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_array_equal(OneEuroFilter(min_cutoff=0.1)(x, 0.0), x)


def test_stationary_signal_converges_without_overshoot() -> None:
    samples = np.r_[0.0, np.full(90, 10.0)]
    out = run_filter(OneEuroFilter(min_cutoff=1.0, beta=0.0), samples)[:, 0]
    assert abs(out[-1] - 10.0) < 1e-3
    assert np.all(np.diff(out) >= 0) and out.max() <= 10.0


def test_higher_beta_follows_a_step_faster() -> None:
    samples = np.r_[np.zeros(10), np.full(60, 100.0)]

    def frames_to_reach_90(beta: float) -> int:
        out = run_filter(OneEuroFilter(min_cutoff=0.5, beta=beta), samples)[:, 0]
        return int(np.argmax(out >= 90.0)) - 10

    slow, fast = frames_to_reach_90(0.0), frames_to_reach_90(0.05)
    assert 0 < fast < slow


def test_jitter_is_reduced_more_with_lower_cutoff() -> None:
    noisy = 50.0 + np.random.default_rng(0).normal(0.0, 1.0, 300)
    stds = [run_filter(OneEuroFilter(min_cutoff=c, beta=0.01), noisy)[30:, 0].std() for c in (3.0, 0.5)]
    assert stds[0] < 0.7 * noisy.std()
    assert stds[1] < 0.6 * stds[0]


def test_points_get_independent_cutoffs() -> None:
    """A moving point must not make a still point jittery (and vice versa)."""
    rng = np.random.default_rng(1)
    pair = OneEuroFilter(min_cutoff=0.5, beta=0.05)
    alone = OneEuroFilter(min_cutoff=0.5, beta=0.05)
    for i in range(60):
        still = np.array([5.0, 5.0]) + rng.normal(0.0, 0.5, 2)
        moving = np.array([10.0 * i, 0.0])
        both = pair(np.stack([still, moving]), i / FPS)
        np.testing.assert_allclose(both[0], alone(still[None], i / FPS)[0])


def test_repeated_timestamp_is_finite() -> None:
    f = OneEuroFilter(min_cutoff=1.0, beta=1.0)
    f(np.zeros(3), 1.0)
    assert np.all(np.isfinite(f(np.ones(3), 1.0)))


def test_reset_forgets_history() -> None:
    f = OneEuroFilter(min_cutoff=0.1)
    f(np.zeros(2), 0.0)
    f.reset()
    np.testing.assert_array_equal(f(np.full(2, 7.0), 0.1), [7.0, 7.0])


def test_smoothing_mapping() -> None:
    assert smoothing_params(0.0) is None
    assert smoothing_params(-1.0) is None
    params = [smoothing_params(s) for s in (0.1, 0.5, 1.0)]
    cutoffs = [p.min_cutoff for p in params if p]
    betas = [p.beta for p in params if p]
    assert cutoffs == sorted(cutoffs, reverse=True) and betas == sorted(betas, reverse=True)
    assert smoothing_params(5.0) == smoothing_params(1.0)


# ---------------------------------------------------------------------------
# Detection windows
# ---------------------------------------------------------------------------


def test_search_windows_cover_the_frame_center_first() -> None:
    wide = search_windows(720, 1280)
    assert wide[0] == DetectionWindow(280, 0, 720)
    covered = np.zeros(1280, bool)
    for w in wide:
        covered[w.x0 : w.x0 + w.size] = True
    assert covered.all()
    assert search_windows(480, 480) == [DetectionWindow(0, 0, 480)]
    assert {w.x0 for w in search_windows(1280, 720)} == {0}  # portrait slides vertically


def test_window_pans_in_small_steps_toward_the_face() -> None:
    window = DetectionWindow(280, 0, 720)
    centered = np.array([[600.0, 300.0], [680.0, 420.0]])
    assert follow_face(window, centered, 720, 1280) == window  # inside the dead zone
    left = centered - (300.0, 0.0)
    moved = follow_face(window, left, 720, 1280)
    assert moved.x0 == 280 - round(0.02 * 720) and moved.y0 == 0


# ---------------------------------------------------------------------------
# FaceTracker with a fake model
# ---------------------------------------------------------------------------

# Landmark layout of the fake model: offsets in face widths from the face center.
TEMPLATE = np.random.default_rng(2).uniform(-0.45, 0.45, (NUM_LANDMARKS, 2))
TEMPLATE[EYE_OUTER_CORNERS[0]] = (-0.3, -0.1)
TEMPLATE[EYE_OUTER_CORNERS[1]] = (0.3, -0.1)


class BlobFaceModel:
    """Finds the bright square "face" in the RGB crop and returns template landmarks.

    Like the real detector it misses faces cut by the crop border.
    """

    def __init__(self) -> None:
        self.inputs: list[tuple[tuple[int, ...], int]] = []
        self.closed = 0

    def detect(self, rgb: np.ndarray, ts_ms: int) -> np.ndarray | None:
        self.inputs.append((rgb.shape, ts_ms))
        ys, xs = np.nonzero(rgb[..., 2] > 128)  # the blob is blue in BGR -> channel 2 in RGB
        height, width = rgb.shape[:2]
        if xs.size == 0 or xs.min() == 0 or ys.min() == 0 or xs.max() == width - 1 or ys.max() == height - 1:
            return None
        size = xs.max() - xs.min() + 1
        center = np.array([(xs.min() + xs.max()) / 2.0, (ys.min() + ys.max()) / 2.0])
        points = (center + TEMPLATE * size) / (width, height)
        return np.c_[points, np.full(NUM_LANDMARKS, 0.01)].astype(np.float32)

    def close(self) -> None:
        self.closed += 1


def blob_frame(cx: float, cy: float, size: int = 160, shape: tuple[int, int] = (720, 1280)) -> np.ndarray:
    frame = np.zeros((*shape, 3), np.uint8)
    x0, y0 = round(cx - size / 2), round(cy - size / 2)
    frame[y0 : y0 + size, x0 : x0 + size] = (255, 0, 0)
    return frame


def expected_landmarks(cx: float, cy: float, size: int = 160) -> np.ndarray:
    return np.array([cx - 0.5, cy - 0.5]) + TEMPLATE * size


def make_tracker(smoothing: float = 0.0) -> tuple[FaceTracker, BlobFaceModel]:
    model = BlobFaceModel()
    return FaceTracker(None, smoothing, model=model), model


def test_detects_on_a_downscaled_square_window_and_maps_back() -> None:
    tracker, model = make_tracker()
    landmarks, z = tracker.detect(blob_frame(640, 360), 0)
    assert model.inputs[0][0] == (640, 640, 3)
    assert landmarks is not None and z is not None
    assert landmarks.shape == (NUM_LANDMARKS, 2) and landmarks.dtype == np.float32
    np.testing.assert_allclose(landmarks, expected_landmarks(640, 360), atol=1.5)
    np.testing.assert_allclose(z, 0.01 * 720, rtol=1e-5)  # z in window-width pixels


def test_small_frames_are_not_upscaled() -> None:
    tracker, model = make_tracker()
    tracker.detect(blob_frame(320, 240, 100, (480, 640)), 0)
    assert model.inputs[0][0] == (480, 480, 3)


def test_off_center_face_is_found_by_searching_windows() -> None:
    tracker, _ = make_tracker()
    frame = blob_frame(150, 360)
    results = [tracker.detect(frame, 33 * i)[0] for i in range(3)]
    found = [r for r in results if r is not None]
    assert found, "the face must be found within one search cycle"
    np.testing.assert_allclose(found[0], expected_landmarks(150, 360), atol=1.0)


def test_window_follows_a_moving_face_without_losing_it() -> None:
    tracker, _ = make_tracker()
    xs = np.r_[np.linspace(640, 1150, 60), np.linspace(1150, 120, 100)]
    lost = 0
    for i, x in enumerate(xs):
        landmarks, _ = tracker.detect(blob_frame(x, 360), 33 * i)
        if landmarks is None:
            lost += 1
        else:
            np.testing.assert_allclose(landmarks, expected_landmarks(x, 360), atol=1.5)
    assert lost == 0


def test_timestamps_fed_to_the_model_strictly_increase() -> None:
    tracker, model = make_tracker()
    frame = blob_frame(640, 360)
    for ts in (100, 100, 50, 200, 200):
        tracker.detect(frame, ts)
    stamps = [ts for _, ts in model.inputs]
    assert stamps[0] == 100 and all(b > a for a, b in itertools.pairwise(stamps))


def test_lost_face_returns_none_and_resets_smoothing() -> None:
    tracker, _ = make_tracker(smoothing=1.0)
    tracker.detect(blob_frame(600, 360), 0)
    assert tracker.detect(np.zeros((720, 1280, 3), np.uint8), 33) == (None, None)
    landmarks, _ = tracker.detect(blob_frame(680, 360), 66)
    assert landmarks is not None
    np.testing.assert_allclose(landmarks, expected_landmarks(680, 360), atol=1.5)  # no lag from 600


def test_smoothing_reduces_jitter() -> None:
    offsets = np.random.default_rng(3).integers(-2, 3, 60)

    def jitter(smoothing: float) -> float:
        tracker, _ = make_tracker(smoothing)
        xs = []
        for i, dx in enumerate(offsets):
            landmarks, _ = tracker.detect(blob_frame(640 + dx, 360), 33 * i)
            assert landmarks is not None
            xs.append(landmarks[:, 0].mean())
        return float(np.std(xs[10:]))

    raw, smooth = jitter(0.0), jitter(0.8)
    assert smooth < 0.5 * raw
    tracker, _ = make_tracker(0.3)
    tracker.set_smoothing(2.0)
    assert tracker.smoothing == 1.0


def test_close_closes_the_model_once() -> None:
    tracker, model = make_tracker()
    tracker.close()
    tracker.close()
    assert model.closed == 1


def test_model_path_or_model_is_required() -> None:
    with pytest.raises(ValueError):
        FaceTracker(None)


# ---------------------------------------------------------------------------
# Real MediaPipe model (skipped when models / scikit-image are unavailable)
# ---------------------------------------------------------------------------


@pytest.fixture
def landmarker_path(models_dir: Path) -> Path:
    path = models_dir / MODEL_SPECS[FACE_LANDMARKER].filename
    if not path.is_file():
        pytest.skip(f"{path.name} not found in {models_dir}")
    return path


def astronaut_bgr() -> np.ndarray:
    data = pytest.importorskip("skimage.data")
    return cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR)


def assert_plausible_face(landmarks: np.ndarray, frame_shape: tuple[int, ...]) -> None:
    height, width = frame_shape[:2]
    assert landmarks.shape == (NUM_LANDMARKS, 2)
    assert np.all((landmarks >= 0) & (landmarks < (width, height)))
    eye_y = landmarks[[33, 263], 1].mean()
    assert eye_y < landmarks[1, 1] < landmarks[13, 1]  # eyes above nose tip above lips
    assert landmarks[263, 0] > landmarks[33, 0]  # subject's left eye is on the image's right


def test_real_model_on_astronaut_portrait(landmarker_path: Path) -> None:
    tracker = FaceTracker(landmarker_path, smoothing=0.5)
    try:
        frame = astronaut_bgr()
        landmarks, z = tracker.detect(frame, 0)
        assert landmarks is not None and z is not None
        assert_plausible_face(landmarks, frame.shape)
        assert 40 < np.linalg.norm(landmarks[33] - landmarks[263]) < 90
        assert np.all(np.isfinite(z))
    finally:
        tracker.close()


def test_real_model_finds_a_dim_face_in_a_wide_frame(landmarker_path: Path, face_frame_720p: np.ndarray) -> None:
    """Regression: the letterboxed full 16:9 frame misses this face; the square window does not."""
    tracker = FaceTracker(landmarker_path, smoothing=0.5)
    try:
        dim = np.clip((face_frame_720p / 255.0) ** 1.6 * 255.0 * (0.85, 0.95, 1.05), 0, 255).astype(np.uint8)
        found = [tracker.detect(dim, 33 * i)[0] for i in range(3)]
        assert all(landmarks is not None for landmarks in found)
        last = found[-1]
        assert last is not None
        assert_plausible_face(last, dim.shape)
    finally:
        tracker.close()
