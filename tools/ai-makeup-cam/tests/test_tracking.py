"""One Euro filter, smoothing mapping and FaceTracker (fake model + optional real model)."""

from __future__ import annotations

import itertools
from pathlib import Path

import cv2
import numpy as np
import pytest

from makeup_cam.models import FACE_LANDMARKER, MODEL_SPECS
from makeup_cam.tracking import (
    _WindowBuffers,
    _MAX_COAST_FRAMES,
    EYE_OUTER_CORNERS,
    NUM_LANDMARKS,
    DetectionWindow,
    FaceMotion,
    FaceTracker,
    OneEuroFilter,
    follow_face,
    search_windows,
    smoothing_params,
    window_rgb,
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


# ---------------------------------------------------------------------------
# Reused image buffers
# ---------------------------------------------------------------------------


def test_window_rgb_gives_the_same_picture_with_buffers() -> None:
    frame = np.random.default_rng(4).integers(0, 255, (720, 1280, 3), np.uint8)
    untouched = frame.copy()
    buffers = _WindowBuffers()
    for window, side in (
        (DetectionWindow(280, 0, 720), 640),  # downscaled
        (DetectionWindow(0, 0, 720), 640),
        (DetectionWindow(100, 50, 300), 640),  # not downscaled: must not write into the frame
        (DetectionWindow(280, 0, 720), 480),  # another size
        (DetectionWindow(280, 0, 720), 640),  # and back
    ):
        np.testing.assert_array_equal(window_rgb(frame, window, side, buffers), window_rgb(frame, window, side))
    np.testing.assert_array_equal(frame, untouched)


def test_window_buffers_are_reused_while_the_size_stays() -> None:
    frame = np.zeros((720, 1280, 3), np.uint8)
    buffers = _WindowBuffers()
    window = DetectionWindow(0, 0, 720)
    first = window_rgb(frame, window, 640, buffers)
    assert window_rgb(frame, window, 640, buffers) is first or np.shares_memory(
        window_rgb(frame, window, 640, buffers), first
    )
    assert window_rgb(frame, window, 320, buffers).shape == (320, 320, 3)
    assert all(array.shape == (320, 320, 3) for array in buffers.for_side(320))


def test_tracker_follows_frame_size_changes_with_its_buffers() -> None:
    tracker, model = make_tracker()
    sizes = (((720, 1280), 640, 360), ((480, 640), 320, 240), ((1080, 1920), 960, 540), ((720, 1280), 640, 360))
    for shape, cx, cy in sizes:
        landmarks, _ = tracker.detect(blob_frame(cx, cy, shape=shape), len(model.inputs) * 33)
        assert landmarks is not None
        np.testing.assert_allclose(landmarks, expected_landmarks(cx, cy), atol=2.5)
    assert [size[0][0] for size in model.inputs] == [640, 480, 640, 640]


# ---------------------------------------------------------------------------
# Image motion: the landmarks follow the picture, not the model's lag
# ---------------------------------------------------------------------------


def textured_frame(dx: float = 0.0, dy: float = 0.0, shape: tuple[int, int] = (720, 720)) -> np.ndarray:
    """A picture full of corners, moved by a sub-pixel amount. Square: the detection window then stays put."""
    noise = np.random.default_rng(5).integers(0, 256, (shape[0] // 8, shape[1] // 8), np.uint8)
    big = cv2.GaussianBlur(cv2.resize(noise, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST), (0, 0), 2.0)
    moved = cv2.warpAffine(big, np.float32([[1, 0, dx], [0, 1, dy]]), (shape[1], shape[0]), borderMode=cv2.BORDER_REFLECT)
    return cv2.cvtColor(moved, cv2.COLOR_GRAY2BGR)


class ScriptedPoints:
    """A landmark model that answers with the given normalized points, one call after another."""

    def __init__(self, script: list[np.ndarray | None]) -> None:
        self.script = script
        self.calls = 0

    def detect(self, rgb: np.ndarray, ts_ms: int) -> np.ndarray | None:
        answer = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        return answer

    def close(self) -> None:
        pass


@pytest.fixture
def face(face_landmarks_720p: np.ndarray) -> np.ndarray:
    """The test face's landmarks, in the middle of a square frame."""
    return face_landmarks_720p - (280.0, 0.0)


def points_at(landmarks: np.ndarray, dy: float) -> np.ndarray:
    """What the model answers for a face moved by ``dy``: normalized (478, 3)."""
    points = (landmarks + (0.0, dy)) / 720.0
    return np.c_[points, np.zeros(NUM_LANDMARKS)].astype(np.float32)


def test_motion_of_a_textured_picture_is_measured(face: np.ndarray) -> None:
    motion = FaceMotion()
    motion.update(textured_frame())
    motion.update(textured_frame(6.0, -4.0))
    fit = motion.measure(face)
    assert fit is not None
    moved = face @ fit[:2, :2].T + fit[:2, 2]
    np.testing.assert_allclose(moved - face, np.broadcast_to([6.0, -4.0], moved.shape), atol=0.25)


def test_no_motion_is_told_without_texture_or_a_previous_frame(face: np.ndarray) -> None:
    motion = FaceMotion()
    assert motion.measure(face) is None
    motion.update(np.zeros((720, 720, 3), np.uint8))
    assert motion.measure(face) is None
    motion.update(np.zeros((720, 720, 3), np.uint8))
    assert motion.measure(face) is None


def test_a_face_outside_the_picture_has_no_motion(face: np.ndarray) -> None:
    motion = FaceMotion()
    motion.update(textured_frame())
    motion.update(textured_frame(3.0, 3.0))
    assert motion.measure(face + (0.0, 2000.0)) is None


def test_a_still_picture_reports_no_motion_at_all(face: np.ndarray) -> None:
    motion = FaceMotion()
    motion.update(textured_frame())
    motion.update(textured_frame())
    fit = motion.measure(face)
    assert fit is not None
    assert np.array_equal(fit, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def test_the_landmarks_keep_up_with_the_picture_when_the_model_lags(face: np.ndarray) -> None:
    """The filter alone trails a nod by several pixels; with the image motion it hardly does."""
    frames = [12.0 * np.sin(2 * np.pi * i / 30) for i in range(45)]
    model_positions = [0.0, *frames[:-1]]  # the model answers with where the face was a frame ago
    script = [points_at(face, dy) for dy in model_positions]
    tracker = FaceTracker(None, 0.5, model=ScriptedPoints(script))
    errors, filter_only = [], []
    plain = OneEuroFilter(*_params_of(0.5))
    for i, dy in enumerate(frames):
        landmarks, _ = tracker.detect(textured_frame(0.0, dy), 33 * i)
        assert landmarks is not None
        errors.append(abs(float(landmarks[33, 1] - face[33, 1]) - dy))
        filtered = plain(np.array([face[33, 1] + model_positions[i]]), i * 33 / 1000.0)
        filter_only.append(abs(float(filtered[0] - face[33, 1]) - dy))
    assert max(errors[5:]) < 0.5 * max(filter_only[5:])
    assert max(errors[5:]) < 2.0


def _params_of(smoothing: float) -> tuple[float, float, float]:
    params = smoothing_params(smoothing)
    assert params is not None
    return params.min_cutoff, params.beta / 120.0, params.d_cutoff


def test_the_face_is_carried_along_while_the_model_finds_nothing(face: np.ndarray) -> None:
    script: list[np.ndarray | None] = [points_at(face, 0.0)] * 3 + [None] * 4
    tracker = FaceTracker(None, 0.5, model=ScriptedPoints(script))
    for i in range(3):
        assert tracker.detect(textured_frame(), 33 * i)[0] is not None
    assert tracker.coast() is None  # the model found it: nothing to carry
    for i in range(1, 5):
        assert tracker.detect(textured_frame(0.0, 5.0 * i), 99 + 33 * i) == (None, None)
        coast = tracker.coast()
        assert coast is not None
        np.testing.assert_allclose(coast[0][33], face[33] + (0.0, 5.0 * i), atol=0.6)
        assert coast[1].shape == (NUM_LANDMARKS,)


def test_coasting_ends_when_the_picture_gives_no_motion_and_after_a_while(face: np.ndarray) -> None:
    tracker = FaceTracker(None, 0.5, model=ScriptedPoints([points_at(face, 0.0), None]))
    tracker.detect(textured_frame(), 0)
    tracker.detect(np.zeros((720, 720, 3), np.uint8), 33)
    assert tracker.coast() is None

    tracker = FaceTracker(None, 0.5, model=ScriptedPoints([points_at(face, 0.0), None]))
    tracker.detect(textured_frame(), 0)
    for i in range(_MAX_COAST_FRAMES):
        tracker.detect(textured_frame(), 33 * (i + 1))
        assert tracker.coast() is not None
    tracker.detect(textured_frame(), 33 * (_MAX_COAST_FRAMES + 1))
    assert tracker.coast() is None


def test_a_one_frame_dropout_causes_no_jump(face: np.ndarray) -> None:
    good = points_at(face, 0.0)
    tracker = FaceTracker(None, 1.0, model=ScriptedPoints([good, good, None, good]))
    frame = textured_frame()
    before, _ = tracker.detect(frame, 0)
    before, _ = tracker.detect(frame, 33)
    tracker.detect(frame, 66)
    after, _ = tracker.detect(frame, 99)
    assert before is not None and after is not None
    np.testing.assert_allclose(after, before, atol=0.01)


def test_a_face_found_far_from_where_it_went_starts_the_filter_anew(face: np.ndarray) -> None:
    near, far = points_at(face, 0.0), points_at(face, 120.0)
    tracker = FaceTracker(None, 1.0, model=ScriptedPoints([near, None, far]))
    frame = textured_frame()
    tracker.detect(frame, 0)
    tracker.detect(frame, 33)
    landmarks, _ = tracker.detect(frame, 66)
    assert landmarks is not None
    np.testing.assert_allclose(landmarks[33], face[33] + (0.0, 120.0), atol=0.01)  # no lag from before


def test_real_model_keeps_up_with_a_nod(landmarker_path: Path, face_frame_720p: np.ndarray) -> None:
    """The eyeliner's problem: a soft nod used to leave the lash line over 6 px behind the eye."""
    still = FaceTracker(landmarker_path, smoothing=0.5)
    moving = FaceTracker(landmarker_path, smoothing=0.5)
    try:
        for i in range(20):
            rest, _ = still.detect(face_frame_720p, 33 * i)
            moving.detect(face_frame_720p, 33 * i)
        assert rest is not None
        worst = 0.0
        for i in range(60):
            dy = 25.0 * np.sin(2 * np.pi * 1.2 * i / 30)
            moved = cv2.warpAffine(face_frame_720p, np.float32([[1, 0, 0], [0, 1, dy]]), (1280, 720), borderMode=cv2.BORDER_REPLICATE)
            landmarks, _ = moving.detect(moved, 33 * (20 + i))
            assert landmarks is not None
            if i >= 8:
                worst = max(worst, float(np.abs(landmarks[[159, 386], 1] - rest[[159, 386], 1] - dy).max()))
        assert worst < 5.0
    finally:
        still.close()
        moving.close()


def test_slack_keeps_small_differences_from_counting_as_speed() -> None:
    def after_step(slack: float, size: float) -> float:
        f = OneEuroFilter(1.0, 1.0, 8.0)
        f.slack = slack
        f(np.zeros((1, 2)), 0.0)
        return float(f(np.full((1, 2), size), 1.0 / FPS)[0, 0])

    assert after_step(0.0, 2.0) > 1.5 * after_step(2.0, 2.0)  # calm within the slack
    assert after_step(2.0, 40.0) > 0.8 * after_step(0.0, 40.0) * 0.9  # a real change still gets through
