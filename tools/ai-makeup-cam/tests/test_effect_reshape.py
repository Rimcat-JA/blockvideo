"""ReshapeEffect on real landmarks: exact no-ops, what each slider does, the cap,
landing accuracy, masks, background protection, the plate, the teeth pin, jitter,
pose fade, odd geometry, and the pipeline contract (later effects stay aligned)."""

from __future__ import annotations

from dataclasses import fields

import cv2
import numpy as np
import pytest

from makeup_cam import regions
from makeup_cam.config import MakeupSettings, ReshapeSettings
from makeup_cam.effects import reshape as reshape_module
from makeup_cam.effects.base import Effect, FrameContext, registered_effects
from makeup_cam.effects.eyes import EyelinerEffect
from makeup_cam.effects.lips import LipEffect
from makeup_cam.effects.reshape import ReshapeEffect
from makeup_cam.pipeline import MakeupPipeline
from makeup_cam.segmentation import SegMasks

SLIDERS = [f.name for f in fields(ReshapeSettings) if f.name not in ("enabled", "protect_background")]
ALL_UP = {name: 1.0 for name in SLIDERS}
ALL_DOWN = {name: (0.0 if name == "mouth_corner" else -1.0) for name in SLIDERS}
MODERATE = {name: 0.4 for name in SLIDERS}


@pytest.fixture
def frame(face_frame_720p: np.ndarray) -> np.ndarray:
    return face_frame_720p.astype(np.float32) / np.float32(255.0)


def run(frame, landmarks, fake_ctx, intensity=1.0, masks=(None, None, None), background=None, effect=None, **values):
    ctx = fake_ctx(frame, landmarks, hair_mask=masks[0], face_skin_mask=masks[1], body_skin_mask=masks[2], intensity=intensity)
    ctx.background = background
    out = (effect or ReshapeEffect()).apply(frame.copy(), ctx, ReshapeSettings(**values))
    return out, ctx


def ramp(height: int, width: int) -> np.ndarray:
    """A frame whose pixels hold their own position, to read the warp back from the output."""
    xs, ys = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    return np.dstack([xs / width, ys / height, np.zeros_like(xs)])


def source_of(out: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per pixel, the pixel index it was taken from (see ``ramp``)."""
    height, width = out.shape[:2]
    return out[..., 0] * width, out[..., 1] * height


def displacement(height, width, landmarks, fake_ctx, **kwargs) -> tuple[np.ndarray, FrameContext]:
    """(H, W) length of the backward field at each pixel, and the context after the warp."""
    frame = ramp(height, width)
    out, ctx = run(frame, landmarks, fake_ctx, **kwargs)
    sx, sy = source_of(out)
    xs, ys = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    return np.hypot(sx - xs, sy - ys), ctx


def landing_error(landmarks, fake_ctx, height=720, width=1280, **kwargs) -> np.ndarray:
    """For each landmark: distance between where its pixel really went and ctx.landmarks."""
    frame = ramp(height, width)
    out, ctx = run(frame, landmarks, fake_ctx, **kwargs)
    q = np.asarray(ctx.landmarks, np.float32)
    sample = cv2.remap(out, (q[:, 0] - 0.5)[None, :], (q[:, 1] - 0.5)[None, :], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)[0]
    decoded = np.stack([sample[:, 0] * width + 0.5, sample[:, 1] * height + 0.5], axis=1)
    return np.linalg.norm(decoded - landmarks, axis=1)


def full_roi(frame: np.ndarray) -> regions.Roi:
    return regions.Roi(0, 0, frame.shape[1], frame.shape[0])


def face_axes(landmarks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    right = regions.eye_center(landmarks, "right")
    left = regions.eye_center(landmarks, "left")
    ex = (left - right) / np.linalg.norm(left - right)
    return ex, np.array([-ex[1], ex[0]], np.float32)


def height_of(landmarks: np.ndarray, reference: np.ndarray, index) -> float:
    """Mean position of the points along the face's down axis (axes of ``reference``)."""
    _, ey = face_axes(reference)
    return float((np.asarray(landmarks)[index] @ ey).mean())


def distance(landmarks: np.ndarray, a: int, b: int) -> float:
    return float(np.linalg.norm(np.asarray(landmarks)[a] - np.asarray(landmarks)[b]))


def test_registered_under_its_settings_section() -> None:
    assert ReshapeEffect in registered_effects()
    assert (ReshapeEffect.name, ReshapeEffect.order, ReshapeEffect.needs_face) == ("reshape", 5, True)
    assert isinstance(getattr(MakeupSettings(), ReshapeEffect.name), ReshapeSettings)


@pytest.mark.parametrize(
    ("intensity", "values", "use_landmarks"),
    [
        (1.0, {}, True),
        (1.0, {"protect_background": 0.0}, True),
        (1.0, {"protect_background": 1.0}, True),
        (0.0, ALL_UP, True),
        (1.0, ALL_UP, False),
    ],
    ids=["sliders-0", "sliders-0-protect-0", "sliders-0-protect-1", "intensity-0", "no-face"],
)
def test_exact_noop_returns_the_frame_and_leaves_the_context_alone(
    frame, face_landmarks_720p, fake_ctx, intensity, values, use_landmarks
) -> None:
    landmarks = face_landmarks_720p if use_landmarks else None
    mask = np.ones(frame.shape[:2], np.float32)
    ctx = fake_ctx(frame, landmarks, hair_mask=mask, face_skin_mask=mask, body_skin_mask=mask, intensity=intensity)
    before = (ctx.landmarks, ctx.hair_mask, ctx.face_skin_mask, ctx.body_skin_mask)
    work = frame.copy()
    out = ReshapeEffect().apply(work, ctx, ReshapeSettings(**values))
    assert out is work
    assert np.array_equal(out, frame)
    after = (ctx.landmarks, ctx.hair_mask, ctx.face_skin_mask, ctx.body_skin_mask)
    assert all(a is b for a, b in zip(before, after))
    assert ctx.cache == {}


@pytest.mark.parametrize("values", [ALL_UP, ALL_DOWN, MODERATE], ids=["up", "down", "moderate"])
def test_output_is_a_valid_frame(frame, face_landmarks_720p, fake_ctx, values) -> None:
    out, ctx = run(frame, face_landmarks_720p, fake_ctx, **values)
    assert out.dtype == np.float32 and out.shape == frame.shape
    assert np.isfinite(out).all()
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert not np.array_equal(out, frame)
    assert ctx.landmarks.shape == face_landmarks_720p.shape and ctx.landmarks.dtype == np.float32


def outer_jaw_width(landmarks):
    return distance(landmarks, 58, 288)


def angle_width(landmarks):
    return distance(landmarks, 172, 397)


def temple_width(landmarks):
    return distance(landmarks, 127, 356)


def eye_width(landmarks):
    return distance(landmarks, regions.LEFT_EYE_INNER_CORNER, regions.LEFT_EYE_OUTER_CORNER)


def wing_width(landmarks):
    return distance(landmarks, 64, 294)


def mouth_size(landmarks):
    return distance(landmarks, regions.MOUTH_LEFT_CORNER, regions.MOUTH_RIGHT_CORNER)


@pytest.mark.parametrize(
    ("slider", "measure", "low", "high"),
    [
        ("face_slim", outer_jaw_width, 0.875, 0.915),
        ("jaw", angle_width, 0.915, 0.955),
        ("eye_size", eye_width, 1.105, 1.145),
        ("nose_width", wing_width, 0.85, 0.89),
        ("mouth_width", mouth_size, 0.905, 0.95),
    ],
)
def test_width_sliders_change_their_feature_by_a_subtle_amount(
    frame, face_landmarks_720p, fake_ctx, slider, measure, low, high
) -> None:
    original = measure(face_landmarks_720p)
    up = measure(run(frame, face_landmarks_720p, fake_ctx, **{slider: 1.0})[1].landmarks) / original
    down = measure(run(frame, face_landmarks_720p, fake_ctx, **{slider: -1.0})[1].landmarks) / original
    half = measure(run(frame, face_landmarks_720p, fake_ctx, **{slider: 0.5})[1].landmarks) / original
    assert low <= up <= high
    assert (up - 1.0) * (down - 1.0) < 0.0  # the other direction goes the other way
    assert min(1.0, up) < half < max(1.0, up)  # and half the slider is about half the change


def test_face_slim_keeps_the_temples_and_the_inner_face(frame, face_landmarks_720p, fake_ctx) -> None:
    moved = run(frame, face_landmarks_720p, fake_ctx, face_slim=1.0, jaw=1.0)[1].landmarks
    assert temple_width(moved) / temple_width(face_landmarks_720p) > 0.97
    inner = regions.LIPS_INNER + regions.LEFT_EYE + regions.RIGHT_EYE + [regions.NOSE_TIP]
    assert np.linalg.norm(moved[inner] - face_landmarks_720p[inner], axis=1).max() < 1.0


def test_chin_length_moves_the_chin_and_not_the_lips(frame, face_landmarks_720p, fake_ctx) -> None:
    lm = face_landmarks_720p
    scale = regions.face_scale(lm)
    longer = run(frame, lm, fake_ctx, chin_length=1.0)[1].landmarks
    shorter = run(frame, lm, fake_ctx, chin_length=-1.0)[1].landmarks
    chin = [152]
    assert 0.05 * scale < height_of(longer, lm, chin) - height_of(lm, lm, chin) < 0.12 * scale
    assert -0.12 * scale < height_of(shorter, lm, chin) - height_of(lm, lm, chin) < -0.05 * scale
    assert np.linalg.norm(longer[regions.LIPS_OUTER] - lm[regions.LIPS_OUTER], axis=1).max() < 0.02 * scale


def test_eye_size_scales_the_iris_with_the_eye_and_keeps_it_round(frame, face_landmarks_720p, fake_ctx) -> None:
    lm = face_landmarks_720p
    moved = run(frame, lm, fake_ctx, eye_size=1.0)[1].landmarks
    for iris in (regions.LEFT_IRIS, regions.RIGHT_IRIS):
        center = lm[iris].mean(axis=0)
        new_center = moved[iris].mean(axis=0)
        ratio = np.linalg.norm(moved[iris] - new_center, axis=1) / np.linalg.norm(lm[iris] - center, axis=1)
        assert ratio.min() > 1.08 and ratio.max() < 1.17
        assert ratio.max() - ratio.min() < 0.02


def test_nose_length_lifts_the_tip_and_leaves_the_lips(frame, face_landmarks_720p, fake_ctx) -> None:
    lm = face_landmarks_720p
    scale = regions.face_scale(lm)
    moved = run(frame, lm, fake_ctx, nose_length=1.0)[1].landmarks
    assert height_of(moved, lm, [regions.NOSE_TIP]) < height_of(lm, lm, [regions.NOSE_TIP]) - 0.025 * scale
    assert np.linalg.norm(moved[regions.LIPS_OUTER] - lm[regions.LIPS_OUTER], axis=1).max() < 0.012 * scale


def test_nose_width_keeps_the_nose_on_its_center_line(frame, face_landmarks_720p, fake_ctx) -> None:
    lm = face_landmarks_720p
    moved = run(frame, lm, fake_ctx, nose_width=1.0)[1].landmarks
    assert np.linalg.norm(moved[regions.NOSE_TIP] - lm[regions.NOSE_TIP]) < 0.5
    assert np.linalg.norm(moved[regions.LIPS_TOP] - lm[regions.LIPS_TOP]) < 0.5


def test_mouth_corner_lifts_the_corners_and_not_the_middle(frame, face_landmarks_720p, fake_ctx) -> None:
    lm = face_landmarks_720p
    scale = regions.face_scale(lm)
    moved = run(frame, lm, fake_ctx, mouth_corner=1.0)[1].landmarks
    corners = [regions.MOUTH_LEFT_CORNER, regions.MOUTH_RIGHT_CORNER]
    assert height_of(moved, lm, corners) < height_of(lm, lm, corners) - 0.03 * scale
    assert np.linalg.norm(moved[regions.LIPS_INNER_TOP] - lm[regions.LIPS_INNER_TOP]) < 0.012 * scale
    # The slider has no minus direction: a negative value does nothing.
    out, ctx = run(frame, lm, fake_ctx, mouth_corner=-1.0)
    assert np.array_equal(out, frame) and ctx.landmarks is not None


def test_lip_fullness_grows_the_outer_lines_only(frame, face_landmarks_720p, fake_ctx) -> None:
    lm = face_landmarks_720p
    fuller = run(frame, lm, fake_ctx, lip_fullness=1.0)[1].landmarks
    thinner = run(frame, lm, fake_ctx, lip_fullness=-1.0)[1].landmarks

    def thickness(points) -> float:
        return distance(points, regions.LIPS_TOP, regions.LIPS_INNER_TOP) + distance(points, regions.LIPS_INNER_BOTTOM, regions.LIPS_BOTTOM)

    original = thickness(lm)
    assert 1.1 * original < thickness(fuller) < 1.4 * original
    assert 0.6 * original < thickness(thinner) < 0.9 * original
    inner = [regions.LIPS_INNER_TOP, regions.LIPS_INNER_BOTTOM]
    assert np.linalg.norm(fuller[inner] - lm[inner], axis=1).max() < 0.5


def test_brow_lift_moves_the_brows_and_not_the_eyes(frame, face_landmarks_720p, fake_ctx) -> None:
    lm = face_landmarks_720p
    scale = regions.face_scale(lm)
    moved = run(frame, lm, fake_ctx, brow_lift=1.0)[1].landmarks
    lowered = run(frame, lm, fake_ctx, brow_lift=-1.0)[1].landmarks
    brows = regions.LEFT_BROW + regions.RIGHT_BROW
    assert height_of(moved, lm, brows) < height_of(lm, lm, brows) - 0.03 * scale
    assert height_of(lowered, lm, brows) > height_of(lm, lm, brows) + 0.03 * scale
    eyes = regions.LEFT_EYE + regions.RIGHT_EYE
    assert np.linalg.norm(moved[eyes] - lm[eyes], axis=1).max() < 0.012 * scale


@pytest.mark.parametrize("values", [ALL_UP, ALL_DOWN] + [{name: 1.0} for name in SLIDERS], ids=lambda v: str(len(v)) + next(iter(v)))
def test_displacement_stays_below_the_cap(face_landmarks_720p, fake_ctx, values) -> None:
    length, _ = displacement(720, 1280, face_landmarks_720p, fake_ctx, protect_background=0.0, **values)
    face_width = float(np.ptp(face_landmarks_720p[regions.FACE_OVAL][:, 0]))
    # Each pixel's source, measured through the (slightly inexact) cubic ramp.
    assert length.max() <= 0.082 * face_width
    if len(values) == len(SLIDERS):
        assert length.max() >= 0.05 * face_width  # everything on really reaches the cap region


@pytest.mark.parametrize("values", [ALL_UP, ALL_DOWN], ids=["up", "down"])
def test_field_never_folds_over(face_landmarks_720p, values) -> None:
    face = reshape_module._face_frame(face_landmarks_720p)
    amounts = reshape_module._Amounts.of(ReshapeSettings(**values), 1.0)
    step = 4
    roi = regions.roi_around(face.landmarks[regions.FACE_OVAL], (720, 1280), reshape_module._ROI_PAD, 8.0)
    grid = reshape_module._forward_field(face, roi, step, amounts, (720, 1280))
    fx, fy = reshape_module._to_frame_axes(grid)
    gxx, gxy = np.gradient(fx, step, axis=1), np.gradient(fx, step, axis=0)
    gyx, gyy = np.gradient(fy, step, axis=1), np.gradient(fy, step, axis=0)
    jacobian = (1.0 + gxx) * (1.0 + gyy) - gxy * gyx
    assert jacobian.min() > 0.2


def smooth_bump_grid(landmarks: np.ndarray, peak_px: float) -> reshape_module._Grid:
    """A grid with a wide smooth field of the given peak length."""
    face = reshape_module._face_frame(landmarks)
    roi = regions.roi_around(face.landmarks[regions.FACE_OVAL], (720, 1280), reshape_module._ROI_PAD, 8.0)
    grid = reshape_module._Grid(face, roi, 4)
    grid.dx = np.float32(peak_px) * np.exp(-(grid.X**2 + grid.Y**2) / (2.0 * (1.2 * face.scale) ** 2)).astype(np.float32)
    grid.dy = np.zeros_like(grid.dx)
    return grid


def test_limit_caps_a_long_field_softly(face_landmarks_720p) -> None:
    width = reshape_module._face_frame(face_landmarks_720p).width
    grid = smooth_bump_grid(face_landmarks_720p, 3.0 * 0.08 * width)
    reshape_module._limit(grid)
    assert np.hypot(grid.dx, grid.dy).max() <= 0.08 * width + 1e-3
    # No plateau: the peak still rises above what its surroundings reach.
    assert np.hypot(grid.dx, grid.dy).max() > 0.9 * 0.08 * width
    # A field well below the cap is left alone.
    small = smooth_bump_grid(face_landmarks_720p, 0.5 * 0.08 * width)
    before = small.dx.copy()
    reshape_module._limit(small)
    assert np.array_equal(small.dx, before)


def test_forward_field_goes_through_the_limit(face_landmarks_720p, monkeypatch) -> None:
    face = reshape_module._face_frame(face_landmarks_720p)
    roi = regions.roi_around(face.landmarks[regions.FACE_OVAL], (720, 1280), reshape_module._ROI_PAD, 8.0)
    amounts = reshape_module._Amounts.of(ReshapeSettings(**ALL_UP), 1.0)
    limited = reshape_module._forward_field(face, roi, 3, amounts, (720, 1280))
    monkeypatch.setattr(reshape_module, "_limit", lambda grid: None)
    raw = reshape_module._forward_field(face, roi, 3, amounts, (720, 1280))
    assert np.hypot(limited.dx, limited.dy).max() < 0.97 * np.hypot(raw.dx, raw.dy).max()  # it does bite here


def test_limit_scales_down_a_field_that_would_fold(face_landmarks_720p) -> None:
    grid = smooth_bump_grid(face_landmarks_720p, 1.0)
    grid.dx = np.zeros_like(grid.dx)
    grid.dx[grid.ny // 2, grid.nx // 2] = 3.5 * grid.step  # far steeper than any window
    reshape_module._limit(grid)
    assert reshape_module._max_gradient(grid.dx, grid.dy, grid.step) <= 0.85 + 1e-5


@pytest.mark.parametrize(("values", "limit"), [({"eye_size": 1.0}, 0.003), ({"lip_fullness": 1.0}, 0.025), (ALL_UP, 0.0003)], ids=["eye", "lips", "up"])
def test_field_is_low_frequency(face_landmarks_720p, values, limit) -> None:
    """Almost no energy at wavelengths below a quarter of the eye distance (the window kinks)."""
    face = reshape_module._face_frame(face_landmarks_720p)
    roi = regions.roi_around(face.landmarks[regions.FACE_OVAL], (720, 1280), reshape_module._ROI_PAD, 8.0)
    amounts = reshape_module._Amounts.of(ReshapeSettings(**values), 1.0)
    step = 3
    fx, fy = reshape_module._to_frame_axes(reshape_module._forward_field(face, roi, step, amounts, (720, 1280)))
    high = total = 0.0
    for field in (fx, fy):
        power = np.abs(np.fft.fft2(field)) ** 2
        frequency = np.hypot(np.fft.fftfreq(field.shape[0], step)[:, None], np.fft.fftfreq(field.shape[1], step)[None, :])
        high += float(power[frequency > 1.0 / (0.25 * face.scale)].sum())
        total += float(power.sum())
    assert high / total < limit


@pytest.mark.parametrize(("raw", "same_as"), [(5.0, 1.0), (-5.0, -1.0)])
def test_slider_values_beyond_their_range_are_clamped(frame, face_landmarks_720p, fake_ctx, raw, same_as) -> None:
    out, _ = run(frame, face_landmarks_720p, fake_ctx, face_slim=raw)
    expected, _ = run(frame, face_landmarks_720p, fake_ctx, face_slim=same_as)
    assert np.array_equal(out, expected)


@pytest.mark.parametrize("values", [MODERATE, ALL_UP, ALL_DOWN], ids=["moderate", "up", "down"])
def test_landmarks_land_where_the_context_says(face_landmarks_720p, fake_ctx, values) -> None:
    errors = landing_error(face_landmarks_720p, fake_ctx, **values)
    assert np.median(errors) < 0.1
    assert np.percentile(errors, 99) < 0.8


def test_landing_with_mixed_signs_on_a_wide_open_mouth(face_landmarks_720p, fake_ctx) -> None:
    """The strongest compression there is: the chin up and the lower lip down, where
    the plain fixed-point search for the landing point used to stop 7 px short."""
    lm = face_landmarks_720p.copy()
    face = reshape_module._face_frame(lm)
    mouth_y = lm[[regions.MOUTH_LEFT_CORNER, regions.MOUTH_RIGHT_CORNER], 1].mean()
    lm[lm[:, 1] > mouth_y, 1] += 0.5 * face.scale  # opens the jaw: lips apart, everything below moves down
    values = {
        "face_slim": -1.0, "jaw": 1.0, "chin_length": -1.0, "eye_size": -1.0, "nose_width": 1.0,
        "nose_length": -1.0, "mouth_width": -1.0, "mouth_corner": 1.0, "lip_fullness": 1.0, "brow_lift": 1.0,
    }  # fmt: skip
    assert landing_error(lm, fake_ctx, **values).max() < 0.5


def test_landing_accuracy_is_sub_pixel_with_a_rolled_head(face_landmarks_720p, fake_ctx) -> None:
    center = face_landmarks_720p.mean(axis=0)
    angle = np.radians(10.0)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]], np.float32)
    rolled = (face_landmarks_720p - center) @ rotation.T + center
    assert np.percentile(landing_error(rolled, fake_ctx, **MODERATE), 99) < 0.5


def test_masks_warp_with_the_same_map(frame, face_landmarks_720p, fake_ctx) -> None:
    roi = full_roi(frame)
    lips = regions.lips_mask(face_landmarks_720p, roi)
    oval = regions.face_oval_mask(face_landmarks_720p, roi)
    body = np.zeros(frame.shape[:2], np.float32)
    for mask in (lips, oval, body):
        mask.setflags(write=False)  # the pipeline hands out read-only masks
    _, ctx = run(frame, face_landmarks_720p, fake_ctx, masks=(lips, oval, body), **ALL_UP)

    def iou(a, b) -> float:
        a, b = a > 0.5, b > 0.5
        return float((a & b).sum() / (a | b).sum())

    assert iou(ctx.hair_mask, regions.lips_mask(ctx.landmarks, roi)) > 0.93
    assert iou(ctx.face_skin_mask, regions.face_oval_mask(ctx.landmarks, roi)) > 0.97
    assert ctx.hair_mask is not lips and ctx.face_skin_mask is not oval  # the inputs stay as they were
    assert not ctx.hair_mask.flags.writeable  # and the copies are read-only like them
    assert iou(lips, regions.lips_mask(face_landmarks_720p, roi)) == 1.0
    assert np.array_equal(ctx.body_skin_mask, body)


def test_landmarks_and_z_contract(frame, face_landmarks_720p, fake_ctx) -> None:
    original = face_landmarks_720p.copy()
    face_landmarks_720p.setflags(write=False)  # read-only like the pipeline's
    z = np.zeros(len(original), np.float32)
    ctx = fake_ctx(frame, face_landmarks_720p)
    ctx.landmarks_z = z
    ReshapeEffect().apply(frame.copy(), ctx, ReshapeSettings(**ALL_UP))
    assert ctx.landmarks_z is z
    assert ctx.landmarks is not face_landmarks_720p
    assert np.array_equal(face_landmarks_720p, original)
    assert not np.array_equal(ctx.landmarks, original)


def test_only_the_area_around_the_face_changes(frame, face_landmarks_720p, fake_ctx) -> None:
    out, _ = run(frame, face_landmarks_720p, fake_ctx, **ALL_UP)
    scale = regions.face_scale(face_landmarks_720p)
    near = regions.roi_around(face_landmarks_720p[regions.FACE_OVAL], frame.shape, pad=0.0, margin_px=1.6 * scale)
    outside = np.ones(frame.shape[:2], bool)
    outside[near.slices] = False
    assert np.array_equal(out[outside], frame[outside])


def test_an_eye_only_warp_stays_near_the_eyes(frame, face_landmarks_720p, fake_ctx) -> None:
    out, _ = run(frame, face_landmarks_720p, fake_ctx, eye_size=1.0)
    changed = np.abs(out - frame).max(axis=2) > 0
    scale = regions.face_scale(face_landmarks_720p)
    ys = np.flatnonzero(changed.any(axis=1))
    eyes = face_landmarks_720p[regions.LEFT_EYE + regions.RIGHT_EYE]
    assert ys[0] > eyes[:, 1].min() - 0.8 * scale
    assert ys[-1] < eyes[:, 1].max() + 0.8 * scale


def test_warp_strength_follows_intensity(face_landmarks_720p, fake_ctx, frame) -> None:
    full = run(frame, face_landmarks_720p, fake_ctx, face_slim=1.0)[1].landmarks
    half = run(frame, face_landmarks_720p, fake_ctx, face_slim=1.0, intensity=0.5)[1].landmarks
    a = np.linalg.norm(full - face_landmarks_720p, axis=1).max()
    b = np.linalg.norm(half - face_landmarks_720p, axis=1).max()
    assert 0.4 * a < b < 0.6 * a


def open_mouth(landmarks: np.ndarray, gap: float) -> np.ndarray:
    """Landmarks with the jaw dropped by ``gap`` px, the corners pulled outward: a big smile."""
    moved = landmarks.copy()
    _, ey = face_axes(landmarks)
    below = regions.LIPS_LOWER_INNER[1:-1] + regions.LIPS_LOWER_OUTER[1:-1] + [152, 148, 176, 149, 377, 400, 378]
    moved[below] += ey * gap
    return moved


@pytest.mark.parametrize(
    "values",
    [{"mouth_width": 1.0}, {"mouth_width": -1.0}, {"mouth_corner": 1.0}, {"lip_fullness": 1.0}, ALL_UP, ALL_DOWN],
    ids=["width+", "width-", "corner", "fullness", "all+", "all-"],
)
def test_teeth_are_pinned(face_landmarks_720p, fake_ctx, values) -> None:
    """Inside a big open mouth (well inside the inner lip line) nothing moves."""
    scale = regions.face_scale(face_landmarks_720p)
    landmarks = open_mouth(face_landmarks_720p, 0.25 * scale)
    length, _ = displacement(720, 1280, landmarks, fake_ctx, **values)
    inner = regions.expand_polygon(landmarks[regions.LIPS_INNER], -0.1 * scale)
    teeth = regions.polygon_mask(inner, regions.Roi(0, 0, 1280, 720)) > 0.99
    assert teeth.sum() > 200
    assert length[teeth].max() < 0.2
    assert np.percentile(length[teeth], 90) < 0.05


def test_teeth_stay_unchanged_on_the_real_smile(frame, face_landmarks_720p, fake_ctx) -> None:
    """The portrait smiles with open lips: mouth sliders must not disturb the teeth."""
    out, _ = run(frame, face_landmarks_720p, fake_ctx, mouth_width=1.0, mouth_corner=1.0)
    inner = regions.expand_polygon(face_landmarks_720p[regions.LIPS_INNER], -3.0)
    teeth = regions.polygon_mask(inner, full_roi(frame)) > 0.99
    assert teeth.sum() > 50
    assert np.abs(out - frame).max(axis=2)[teeth].mean() < 0.012


# --- background protection ------------------------------------------------


def person_masks(landmarks: np.ndarray, shape, extra_px: float = 0.0):
    """A person mask: only the face oval (a person with tied-back hair)."""
    roi = regions.Roi(0, 0, shape[1], shape[0])
    oval = regions.face_oval_mask(landmarks, roi, expand_px=extra_px, feather_px=2.0)
    return None, oval, None


def test_protect_background_fades_the_field_outside_the_person(face_landmarks_720p, fake_ctx) -> None:
    lm = face_landmarks_720p
    scale = regions.face_scale(lm)
    masks = person_masks(lm, (720, 1280))
    far = regions.face_oval_mask(lm, regions.Roi(0, 0, 1280, 720), expand_px=0.45 * scale) < 0.01
    free = displacement(720, 1280, lm, fake_ctx, masks=masks, protect_background=0.0, face_slim=1.0, jaw=1.0)[0]
    default = displacement(720, 1280, lm, fake_ctx, masks=masks, protect_background=0.8, face_slim=1.0, jaw=1.0)[0]
    full = displacement(720, 1280, lm, fake_ctx, masks=masks, protect_background=1.0, face_slim=1.0, jaw=1.0)[0]
    # The neck column stays "person" on purpose, so look beside and above the face.
    far[int(lm[152, 1]) :, :] = False
    assert free[far].max() > 0.5
    assert default[far].max() < 0.3 * free[far].max()
    assert full[far].max() < 0.25


def test_a_fuller_face_is_not_held_back_by_the_protection(face_landmarks_720p, fake_ctx) -> None:
    """The contour moves outward onto what used to be background: where a pixel comes
    from counts as the person as well as where it is now."""
    lm = face_landmarks_720p
    masks = person_masks(lm, (720, 1280))
    values = {"face_slim": -1.0, "jaw": -1.0}
    free = run(np.zeros((720, 1280, 3), np.float32), lm, fake_ctx, masks=masks, protect_background=0.0, **values)[1].landmarks
    protected = run(np.zeros((720, 1280, 3), np.float32), lm, fake_ctx, masks=masks, protect_background=1.0, **values)[1].landmarks
    assert outer_jaw_width(protected) - outer_jaw_width(lm) > 0.85 * (outer_jaw_width(free) - outer_jaw_width(lm))


def test_hair_pushed_outward_is_not_held_back_by_the_protection(face_landmarks_720p, fake_ctx) -> None:
    """Just outside the hair mask the output shows hair that a fuller face pushed there,
    so the field has to stay on although that spot used to be background."""
    lm = face_landmarks_720p
    scale = regions.face_scale(lm)
    roi = regions.Roi(0, 0, 1280, 720)
    hair = regions.face_oval_mask(lm, roi, expand_px=0.25 * scale, feather_px=2.0)
    ring = (hair < 0.01) & (regions.face_oval_mask(lm, roi, expand_px=0.4 * scale) > 0.99)
    ring[: int(lm[61, 1]), :] = False
    ring[int(lm[152, 1]) :, :] = False
    values = {"face_slim": -1.0, "jaw": -1.0}
    free = displacement(720, 1280, lm, fake_ctx, masks=(hair, None, None), protect_background=0.0, **values)[0]
    protected = displacement(720, 1280, lm, fake_ctx, masks=(hair, None, None), protect_background=1.0, **values)[0]
    assert protected[ring].mean() > 0.4 * free[ring].mean()


def test_the_person_masks_add_up(face_landmarks_720p) -> None:
    """They are class probabilities: on the seam between two classes each is about one
    half, and a maximum of them would let the background protection through there."""
    face = reshape_module._face_frame(face_landmarks_720p)
    roi = regions.roi_around(face.landmarks[regions.FACE_OVAL], (720, 1280), reshape_module._ROI_PAD, 8.0)
    half = np.full((720, 1280), 0.5, np.float32)
    ctx = FrameContext(height=720, width=1280, landmarks=face_landmarks_720p, face_skin_mask=half, body_skin_mask=half)
    person = reshape_module._person_on_grid(ctx, reshape_module._Grid(face, roi, 4))
    assert person.min() == pytest.approx(1.0)


def test_hair_next_to_the_face_moves_with_the_face(face_landmarks_720p, fake_ctx) -> None:
    lm = face_landmarks_720p
    scale = regions.face_scale(lm)
    roi = regions.Roi(0, 0, 1280, 720)
    hair = regions.face_oval_mask(lm, roi, expand_px=0.6 * scale, feather_px=2.0)
    ring = (hair > 0.99) & (regions.face_oval_mask(lm, roi, expand_px=0.25 * scale) < 0.01)
    ring[: int(lm[1, 1]), :] = False
    with_hair = displacement(720, 1280, lm, fake_ctx, masks=(hair, None, None), protect_background=1.0, face_slim=1.0)[0]
    without = displacement(720, 1280, lm, fake_ctx, masks=person_masks(lm, (720, 1280)), protect_background=1.0, face_slim=1.0)[0]
    assert with_hair[ring].mean() > 1.5 * without[ring].mean()


def lined_room(height: int, width: int, spacing: int = 30) -> np.ndarray:
    room = np.full((height, width, 3), 0.75, np.float32)
    room[:, spacing // 2 :: spacing] = 0.1
    room[spacing // 2 :: spacing, :] = 0.1
    return room


def with_person(room: np.ndarray, landmarks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``room`` with a flat skin-colored head drawn over it, and the person mask."""
    person = regions.face_oval_mask(landmarks, full_roi(room), feather_px=1.0)
    frame = room + (np.float32([0.55, 0.65, 0.8]) - room) * person[..., None]
    return frame, person


def line_shift(out: np.ndarray, reference: np.ndarray, x: int, rows: np.ndarray) -> float:
    """Largest sideways shift of the dark vertical line at ``x`` over ``rows``, in pixels."""
    worst = 0.0
    for y in rows:
        window = np.arange(x - 12, x + 13)
        a = np.maximum(0.5 - out[y, window, 1], 0.0)
        b = np.maximum(0.5 - reference[y, window, 1], 0.0)
        if a.sum() > 0 and b.sum() > 0:
            worst = max(worst, abs(float((a * window).sum() / a.sum() - (b * window).sum() / b.sum())))
    return worst


def test_straight_lines_behind_the_face_stay_straight(face_landmarks_720p, fake_ctx) -> None:
    lm = face_landmarks_720p
    scale = regions.face_scale(lm)
    left = float(lm[regions.FACE_OVAL][:, 0].min())
    # Vertical lines behind the face at the level of the jaw: one just outside
    # the narrow band next to the silhouette, one far away.
    lines = [int(left - 0.38 * scale), int(left - 1.5 * scale)]
    room = np.full((720, 1280, 3), 0.75, np.float32)
    room[:, lines] = 0.1
    frame, person = with_person(room, lm)
    masks = (None, person, None)
    rows = np.arange(int(lm[61, 1]), int(lm[152, 1]))
    free, _ = run(frame, lm, fake_ctx, masks=masks, protect_background=0.0, face_slim=1.0, jaw=1.0)
    protected, _ = run(frame, lm, fake_ctx, masks=masks, protect_background=1.0, face_slim=1.0, jaw=1.0)
    assert line_shift(free, frame, lines[0], rows) > 0.3  # the test can see bending
    assert line_shift(protected, frame, lines[0], rows) < 0.05
    assert line_shift(free, frame, lines[1], rows) == 0.0
    assert line_shift(protected, frame, lines[1], rows) == 0.0


def test_plate_fills_what_slimming_uncovers(face_landmarks_720p, fake_ctx) -> None:
    lm = face_landmarks_720p
    room = lined_room(720, 1280)
    frame, person = with_person(room, lm)
    masks = (None, person, None)
    with_plate, ctx = run(frame, lm, fake_ctx, masks=masks, background=room, protect_background=1.0, face_slim=1.0, jaw=1.0)
    plain, _ = run(frame, lm, fake_ctx, masks=masks, protect_background=1.0, face_slim=1.0, jaw=1.0)
    roi = full_roi(frame)
    # Pixels well outside the warped head are exactly the room again...
    outside = regions.face_oval_mask(ctx.landmarks, roi, expand_px=4.0) < 0.001
    assert np.abs(with_plate - room).max(axis=2)[outside].max() < 0.1
    assert np.abs(plain - room).max(axis=2)[outside].max() > 0.2
    # ...and the head itself is not touched by the plate (no lines show through).
    inside = regions.face_oval_mask(ctx.landmarks, roi, expand_px=-4.0) > 0.999
    assert np.abs(with_plate - plain).max(axis=2)[inside].max() < 0.01


def test_plate_is_ignored_when_it_no_longer_matches_the_room(face_landmarks_720p, fake_ctx) -> None:
    lm = face_landmarks_720p
    room = lined_room(720, 1280)
    frame, person = with_person(room, lm)
    blocks = np.indices((720, 1280)).sum(axis=0) // 64 % 2
    other_room = (0.2 + 0.6 * blocks.astype(np.float32))[..., None].repeat(3, axis=2)
    with_plate, _ = run(frame, lm, fake_ctx, masks=(None, person, None), background=other_room, protect_background=1.0, face_slim=1.0)
    plain, _ = run(frame, lm, fake_ctx, masks=(None, person, None), protect_background=1.0, face_slim=1.0)
    assert np.array_equal(with_plate, plain)


def test_plate_follows_a_change_of_exposure(face_landmarks_720p, fake_ctx) -> None:
    lm = face_landmarks_720p
    room = lined_room(720, 1280)
    frame, person = with_person(room, lm)
    out, ctx = run(frame, lm, fake_ctx, masks=(None, person, None), background=room * np.float32(0.85), protect_background=1.0, face_slim=1.0)
    outside = regions.face_oval_mask(ctx.landmarks, full_roi(frame), expand_px=4.0) < 0.001
    assert np.abs(out - room).max(axis=2)[outside].max() < 0.1


def dressed(room: np.ndarray, landmarks: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``with_person`` plus clothes: a top in no mask at all (the model has no class
    for clothes) from just below the jaw on. Returns frame, person mask, clothes mask."""
    frame, person = with_person(room, landmarks)
    jaw_y = float(landmarks[list(reshape_module._JAW_POINTS), 1].mean())
    clothes = np.zeros(frame.shape[:2], np.float32)
    clothes[int(jaw_y + 0.2 * regions.face_scale(landmarks)) :, :] = 1.0
    clothes *= 1.0 - person
    return frame + (np.float32([0.5, 0.2, 0.15]) - frame) * clothes[..., None], person, clothes


def outside_of_the_head(ctx: FrameContext, frame: np.ndarray) -> np.ndarray:
    """Room well away from the warped head. With no neck drawn this includes the room
    right below the chin: the column kept for the neck elsewhere must not bend it."""
    return regions.face_oval_mask(ctx.landmarks, full_roi(frame), expand_px=4.0) < 0.001


def test_plate_statistics_do_not_count_the_clothes_as_room(face_landmarks_720p, fake_ctx) -> None:
    """A pixel-perfect plate is fully trusted, and needs no exposure correction."""
    lm = face_landmarks_720p
    room = lined_room(720, 1280)
    frame, person, _ = dressed(room, lm)
    effect = ReshapeEffect()
    run(frame, lm, fake_ctx, masks=(None, person, None), background=room, effect=effect, face_slim=0.4, jaw=0.3)
    assert effect._plate_trust == 1.0
    assert np.abs(effect._plate_gain - 1.0).max() < 0.01
    assert effect._plate_mismatch < 0.01


@pytest.mark.parametrize("values", [{"face_slim": 0.4, "jaw": 0.3}, {"face_slim": 1.0, "jaw": 1.0}, MODERATE], ids=["slim-.4", "slim-1", "moderate"])
def test_plate_shows_no_double_lines_at_the_default_protection(face_landmarks_720p, fake_ctx, values) -> None:
    """All or nothing: the room next to the head is the straight room, with no ghost
    of the bent one in it (a mix by the protection amount would leave 20 % of it)."""
    lm = face_landmarks_720p
    room = lined_room(720, 1280)
    frame, person = with_person(room, lm)
    assert ReshapeSettings().protect_background == 0.8
    out, ctx = run(frame, lm, fake_ctx, masks=(None, person, None), background=room, **values)
    # 20 % of a line of contrast 0.65 would be 0.13. What may remain is the cubic's ring of
    # a line moved by less than the plate bothers with (under 0.3 px, not a second line).
    assert np.abs(out - room).max(axis=2)[outside_of_the_head(ctx, frame)].max() < 0.07
    plain, _ = run(frame, lm, fake_ctx, masks=(None, person, None), **values)
    assert np.abs(plain - room).max(axis=2)[outside_of_the_head(ctx, frame)].max() > 0.2  # bending would be seen


def test_plate_does_not_paint_over_the_clothes(face_landmarks_720p, fake_ctx) -> None:
    lm = face_landmarks_720p
    room = lined_room(720, 1280)
    frame, person, clothes = dressed(room, lm)
    values = {"face_slim": 1.0, "jaw": 1.0}
    with_plate, ctx = run(frame, lm, fake_ctx, masks=(None, person, None), background=room, **values)
    plain, _ = run(frame, lm, fake_ctx, masks=(None, person, None), **values)
    assert not np.array_equal(with_plate, plain)  # the plate is in use above the clothes
    inside = cv2.erode(clothes, np.ones((9, 9), np.uint8)) > 0.5
    assert np.abs(with_plate - plain).max(axis=2)[inside].max() < 0.01


def test_plate_output_stays_in_range(face_landmarks_720p, fake_ctx) -> None:
    """A room brighter than the plate with bright details: the exposure correction clips."""
    lm = face_landmarks_720p
    plate = np.full((720, 1280, 3), 0.62, np.float32)
    plate[:, ::24] = 0.97
    frame, person = with_person(np.clip(plate * 1.25, 0.0, 1.0), lm)
    effect = ReshapeEffect()
    out, _ = run(frame, lm, fake_ctx, masks=(None, person, None), background=plate, effect=effect, face_slim=1.0, jaw=1.0)
    assert effect._plate_trust == 1.0 and effect._plate_gain.min() > 1.2  # really corrected
    assert out.max() <= 1.0 and out.min() >= 0.0


def plate_samples(difference: float, n: int = 200) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    room = np.full((n, 3), 0.5, np.float32)
    room[::2] = 0.3  # some contrast, so a difference cannot pass for a change of exposure
    live = room + np.where(np.arange(n)[:, None] % 2 == 0, difference, -difference).astype(np.float32)
    return live, room, np.zeros(n, bool)


def test_plate_switches_with_a_fade_and_does_not_come_back_at_once() -> None:
    effect = ReshapeEffect()
    plate = np.zeros((4, 4, 3), np.float32)
    assert effect._plate_level(plate, *plate_samples(0.0)) == 1.0
    going = [effect._plate_level(plate, *plate_samples(0.3)) for _ in range(40)]
    assert going[-1] == 0.0
    assert all(0.0 <= a - b <= 0.1 for a, b in zip([1.0, *going], going))  # a fade, no jump
    # Good again, but only after a while: flipping back and forth would be the worst.
    back = [effect._plate_level(plate, *plate_samples(0.0)) for _ in range(120)]
    assert back[0] == 0.0 and back[40] == 0.0
    assert back[-1] == 1.0
    assert all(0.0 <= b - a <= 0.1 for a, b in zip(back, back[1:]))
    # A flicker never gets it back.
    effect = ReshapeEffect()
    effect._plate_level(plate, *plate_samples(0.3))
    assert effect._plate_trust == 0.0
    levels = [effect._plate_level(plate, *plate_samples(0.3 if index % 8 == 0 else 0.0)) for index in range(100)]
    assert max(levels) == 0.0


def test_plate_of_a_moved_camera_is_noticed_on_a_mostly_plain_wall() -> None:
    """Most samples are the same wall either way: the few on edges decide."""
    effect = ReshapeEffect()
    plate = np.zeros((4, 4, 3), np.float32)
    live, room, _ = plate_samples(0.0)
    on_edge = np.zeros(len(live), bool)
    on_edge[:12] = True
    live[:12] = room[:12] + 0.25
    assert effect._plate_level(plate, live, room, on_edge) == 0.0
    assert effect._plate_on is False


def test_plate_of_a_camera_moved_later_is_noticed_on_a_mostly_plain_wall() -> None:
    effect = ReshapeEffect()
    plate = np.zeros((4, 4, 3), np.float32)
    live, room, on_edge = plate_samples(0.0)
    on_edge[:12] = True
    assert effect._plate_level(plate, live, room, on_edge) == 1.0
    live = live.copy()
    live[:12] = room[:12] + 0.25
    for _ in range(40):
        level = effect._plate_level(plate, live, room, on_edge)
    assert level == 0.0 and effect._plate_on is False


def test_plate_is_left_out_while_no_room_is_in_view(face_landmarks_720p, fake_ctx) -> None:
    """Nothing to compare it with: no statistics, no plate, and nothing breaks."""
    lm = face_landmarks_720p
    room = lined_room(720, 1280)
    frame, _ = with_person(room, lm)
    everything = np.ones((720, 1280), np.float32)
    effect = ReshapeEffect()
    with_plate, _ = run(frame, lm, fake_ctx, masks=(None, everything, None), background=room, effect=effect, face_slim=1.0)
    plain, _ = run(frame, lm, fake_ctx, masks=(None, everything, None), face_slim=1.0)
    assert effect._plate_trust is None
    assert np.array_equal(with_plate, plain)


def test_plate_statistics_do_not_count_the_hair_as_room(face_landmarks_720p, fake_ctx) -> None:
    lm = face_landmarks_720p
    room = lined_room(720, 1280)
    hair = regions.face_oval_mask(lm, full_roi(room), expand_px=0.5 * regions.face_scale(lm), feather_px=1.0)
    frame = room + (np.float32([0.08, 0.08, 0.1]) - room) * hair[..., None]
    skin = regions.face_oval_mask(lm, full_roi(room), feather_px=1.0)
    frame = frame + (np.float32([0.55, 0.65, 0.8]) - frame) * skin[..., None]
    effect = ReshapeEffect()
    run(frame, lm, fake_ctx, masks=(hair, None, None), background=room, effect=effect, face_slim=0.4)
    assert effect._plate_trust == 1.0
    assert np.abs(effect._plate_gain - 1.0).max() < 0.01


def test_plate_of_a_slightly_moved_camera_is_not_used(face_landmarks_720p, fake_ctx) -> None:
    lm = face_landmarks_720p
    room = lined_room(720, 1280, spacing=60)
    frame, person = with_person(room, lm)
    moved = np.roll(room, (5, 9), axis=(0, 1))
    with_plate, _ = run(frame, lm, fake_ctx, masks=(None, person, None), background=moved, face_slim=1.0)
    plain, _ = run(frame, lm, fake_ctx, masks=(None, person, None), face_slim=1.0)
    assert np.array_equal(with_plate, plain)


def test_a_new_plate_or_a_reset_starts_over(face_landmarks_720p, fake_ctx) -> None:
    lm = face_landmarks_720p
    room = lined_room(720, 1280)
    frame, person = with_person(room, lm)
    blocks = np.indices((720, 1280)).sum(axis=0) // 64 % 2
    other = (0.2 + 0.6 * blocks.astype(np.float32))[..., None].repeat(3, axis=2)
    masks = (None, person, None)
    fresh, _ = run(frame, lm, fake_ctx, masks=masks, background=room, face_slim=1.0)
    effect = ReshapeEffect()
    run(frame, lm, fake_ctx, masks=masks, background=other, effect=effect, face_slim=1.0)
    assert effect._plate_trust == 0.0
    # A new plate object counts at once, whatever the old one's state was.
    out, _ = run(frame, lm, fake_ctx, masks=masks, background=room, effect=effect, face_slim=1.0)
    assert np.array_equal(out, fresh)
    run(frame, lm, fake_ctx, masks=masks, background=other, effect=effect, face_slim=1.0)
    effect.reset()
    assert effect._plate is None and effect._plate_trust is None and effect._plate_gain is None
    assert effect._plate_on is False and effect._plate_mismatch == 0.0 and effect._plate_good_frames == 0
    out, _ = run(frame, lm, fake_ctx, masks=masks, background=room, effect=effect, face_slim=1.0)
    assert np.array_equal(out, fresh)


@pytest.mark.parametrize(
    "kwargs",
    [{"protect_background": 0.0}, {"no_masks": True}],
    ids=["protection-off", "no-masks"],
)
def test_plate_needs_the_masks_and_the_protection(face_landmarks_720p, fake_ctx, kwargs) -> None:
    lm = face_landmarks_720p
    room = lined_room(720, 1280)
    frame, person = with_person(room, lm)
    protect = kwargs.get("protect_background", 0.8)
    masks = (None, None, None) if kwargs.get("no_masks") else (None, person, None)
    effect = ReshapeEffect()
    with_plate, _ = run(frame, lm, fake_ctx, masks=masks, background=room, effect=effect, protect_background=protect, face_slim=1.0)
    plain, _ = run(frame, lm, fake_ctx, masks=masks, protect_background=protect, face_slim=1.0)
    assert np.array_equal(with_plate, plain)
    assert effect._plate is None


def test_plate_of_another_size_is_ignored(face_landmarks_720p, fake_ctx, frame) -> None:
    person = regions.face_oval_mask(face_landmarks_720p, full_roi(frame))
    wrong = np.zeros((10, 10, 3), np.float32)
    with_plate, _ = run(frame, face_landmarks_720p, fake_ctx, masks=(None, person, None), background=wrong, face_slim=1.0)
    plain, _ = run(frame, face_landmarks_720p, fake_ctx, masks=(None, person, None), face_slim=1.0)
    assert np.array_equal(with_plate, plain)


# --- stability ------------------------------------------------------------


def backward_field(landmarks: np.ndarray, values: dict, step: int = 5, roi=None) -> tuple[np.ndarray, np.ndarray]:
    face = reshape_module._face_frame(np.asarray(landmarks, np.float32))
    amounts = reshape_module._Amounts.of(ReshapeSettings(**values), 1.0)
    grid = reshape_module._forward_field(face, roi, step, amounts, (720, 1280))
    fx, fy = reshape_module._to_frame_axes(grid)
    return reshape_module._invert(fx, fy, step)


@pytest.mark.parametrize(("values", "limit"), [(MODERATE, 0.35), ({"face_slim": 1.0, "jaw": 1.0}, 0.35)], ids=["moderate", "contour"])
def test_landmark_jitter_does_not_shake_the_field(face_landmarks_720p, values, limit) -> None:
    """White noise on every landmark barely reaches the field: gain well below 1."""
    rng = np.random.default_rng(3)
    sigma = 0.5
    roi = regions.roi_around(face_landmarks_720p[regions.FACE_OVAL], (720, 1280), reshape_module._ROI_PAD, 10.0)
    fields_x, fields_y = [], []
    for _ in range(16):
        noisy = face_landmarks_720p + rng.normal(0.0, sigma, face_landmarks_720p.shape).astype(np.float32)
        bx, by = backward_field(noisy, values, roi=roi)
        fields_x.append(bx)
        fields_y.append(by)
    variation = np.sqrt(np.var(fields_x, axis=0) + np.var(fields_y, axis=0))
    assert np.percentile(variation, 99) < limit * sigma * 2.0
    assert np.median(variation) < limit * sigma


def test_grid_step_does_not_flip_when_the_face_scale_hovers(frame, face_landmarks_720p) -> None:
    effect = ReshapeEffect()
    first = effect._grid_step(110.0)
    assert all(effect._grid_step(scale) == first for scale in (109.0, 111.0, 108.0, 112.0))
    assert effect._grid_step(250.0) != first
    effect.reset()
    assert effect._grid_step(110.0) == first


def test_same_input_gives_the_same_output(frame, face_landmarks_720p, fake_ctx) -> None:
    effect = ReshapeEffect()
    a, _ = run(frame, face_landmarks_720p, fake_ctx, effect=effect, **MODERATE)
    b, _ = run(frame, face_landmarks_720p, fake_ctx, effect=effect, **MODERATE)
    assert np.array_equal(a, b)


# --- pose fade --------------------------------------------------------------


def deep_bow(landmarks: np.ndarray) -> np.ndarray:
    """The projection of a deep bow: the nose tip drops toward the chin, the face shortens."""
    moved = landmarks.copy()
    chin = landmarks[152]
    moved[regions.NOSE_TIP] = landmarks[regions.NOSE_TIP] + 0.85 * (chin - landmarks[regions.NOSE_TIP])
    return moved


def turned_away(landmarks: np.ndarray) -> np.ndarray:
    moved = landmarks.copy()
    moved[regions.NOSE_TIP, 0] = landmarks[regions.FACE_OVAL][:, 0].max() - 2.0
    return moved


def test_pose_inside_the_envelope_is_not_faded(face_landmarks_720p) -> None:
    assert reshape_module._pose_level(reshape_module._face_frame(face_landmarks_720p)) == 1.0


def rotated(landmarks: np.ndarray, yaw: float = 0.0, pitch: float = 0.0, roll: float = 0.0) -> np.ndarray:
    """The projection of the head turned in 3D, on a depth model of an ellipsoid and a
    nose, with the nose as far in front of the outline as the real tracker puts it."""
    oval = landmarks[regions.FACE_OVAL]
    center, width = oval.mean(axis=0), float(np.ptp(oval[:, 0]))
    relative = (landmarks - center) / np.float32([width / 2.0, float(np.ptp(oval[:, 1])) / 2.0])
    ball = -np.sqrt(np.maximum(1.0 - (relative**2).sum(axis=1), 0.0)) * 0.31 * width
    nose = -0.2 * width * np.exp(-((landmarks - landmarks[regions.NOSE_TIP]) ** 2).sum(axis=1) / (2.0 * (0.13 * width) ** 2))
    points = np.column_stack([landmarks - center, ball + nose + 0.45 * width])
    y, p, r = np.radians([yaw, pitch, roll])
    turn = np.array([[np.cos(y), 0.0, np.sin(y)], [0.0, 1.0, 0.0], [-np.sin(y), 0.0, np.cos(y)]])
    nod = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(p), -np.sin(p)], [0.0, np.sin(p), np.cos(p)]])
    tilt = np.array([[np.cos(r), -np.sin(r), 0.0], [np.sin(r), np.cos(r), 0.0], [0.0, 0.0, 1.0]])
    return ((points @ (tilt @ nod @ turn).T)[:, :2] + center).astype(np.float32)


@pytest.mark.parametrize(
    "angles",
    [(20, 0, 0), (-20, 0, 0), (0, 15, 0), (0, -15, 0), (0, 0, 10), (0, 0, -10), (20, 15, 10), (-20, 15, -10), (20, -15, 10), (-20, -15, -10)],
    ids=str,
)
def test_pose_all_through_the_interview_envelope_is_not_faded(face_landmarks_720p, angles) -> None:
    level = reshape_module._pose_level(reshape_module._face_frame(rotated(face_landmarks_720p, *angles)))
    assert level > 0.95


@pytest.mark.parametrize("angles", [(75, 0, 0), (-75, 0, 0), (0, 70, 0)], ids=str)
def test_pose_far_outside_the_envelope_is_faded(face_landmarks_720p, angles) -> None:
    level = reshape_module._pose_level(reshape_module._face_frame(rotated(face_landmarks_720p, *angles)))
    assert level == 0.0


@pytest.mark.parametrize("pose", [deep_bow, turned_away], ids=["bow", "turn"])
def test_pose_beyond_the_envelope_fades_the_warp_out(frame, face_landmarks_720p, fake_ctx, pose) -> None:
    landmarks = pose(face_landmarks_720p)
    assert reshape_module._pose_level(reshape_module._face_frame(landmarks)) == 0.0
    # A fresh effect starts faded: the frame comes back untouched.
    work = frame.copy()
    ctx = fake_ctx(frame, landmarks)
    assert ReshapeEffect().apply(work, ctx, ReshapeSettings(**ALL_UP)) is work
    assert np.array_equal(work, frame)


def test_pose_fade_is_gradual_and_reset_clears_it(frame, face_landmarks_720p, fake_ctx) -> None:
    effect = ReshapeEffect()
    settings = ReshapeSettings(face_slim=1.0)
    bow = deep_bow(face_landmarks_720p)
    effect.apply(frame.copy(), fake_ctx(frame, face_landmarks_720p), settings)
    moves = []
    for _ in range(40):
        ctx = fake_ctx(frame, bow)
        effect.apply(frame.copy(), ctx, settings)
        moves.append(float(np.linalg.norm(ctx.landmarks - bow, axis=1).max()))
    assert moves[0] > 0.0
    assert all(b <= a + 1e-3 for a, b in zip(moves, moves[1:]))
    assert moves[0] - moves[1] < 0.3 * moves[0]
    assert moves[-1] == 0.0
    assert effect._step is not None
    effect.reset()
    assert effect._pose is None and effect._step is None


# --- odd geometry ---------------------------------------------------------


@pytest.mark.parametrize("values", [ALL_UP, ALL_DOWN], ids=["up", "down"])
def test_face_partly_out_of_the_frame(frame, face_landmarks_720p, fake_ctx, values) -> None:
    for shift in ([0.0, 360.0], [0.0, -300.0], [-700.0, 0.0], [700.0, 0.0]):
        moved = face_landmarks_720p + np.float32(shift)
        out, ctx = run(frame, moved, fake_ctx, **values)
        assert np.isfinite(out).all() and out.min() >= 0.0 and out.max() <= 1.0
        assert np.isfinite(ctx.landmarks).all()


def cut_by_the_frame(landmarks: np.ndarray, side: str) -> np.ndarray:
    oval = landmarks[regions.FACE_OVAL]
    shift = {
        "left": [-40.0 - oval[:, 0].min(), 0.0],
        "right": [1320.0 - oval[:, 0].max(), 0.0],
        "top": [0.0, -40.0 - oval[:, 1].min()],
        "chin-at-edge": [0.0, 718.0 - landmarks[152, 1]],
    }[side]
    return landmarks + np.float32(shift)


@pytest.mark.parametrize("values", [MODERATE, ALL_UP, ALL_DOWN, {"face_slim": 1.0, "jaw": 1.0}, {"chin_length": -1.0}], ids=["moderate", "up", "down", "slim", "chin"])
@pytest.mark.parametrize("side", ["left", "right", "top", "chin-at-edge"])
def test_no_smear_where_the_frame_cuts_the_face(face_landmarks_720p, fake_ctx, side, values) -> None:
    """Nothing beyond the edges is read, so no outermost pixels get stretched into a
    band: the picture stays an unbroken, unfolded map right up to the frame's edge."""
    moved = cut_by_the_frame(face_landmarks_720p, side)
    out, _ = run(ramp(720, 1280), moved, fake_ctx, protect_background=0.0, **values)
    sx, sy = source_of(out)
    assert np.diff(sx, axis=1).min() > 0.1
    assert np.diff(sy, axis=0).min() > 0.1
    # At the very edge the field has faded out.
    assert np.abs(sx[:, 0]).max() < 0.25 and np.abs(sx[:, -1] - 1279.0).max() < 0.25
    assert np.abs(sy[0, :]).max() < 0.25 and np.abs(sy[-1, :] - 719.0).max() < 0.25


@pytest.mark.parametrize("side", ["left", "right", "top", "chin-at-edge"])
def test_a_flat_picture_stays_flat_at_the_frame_edges(face_landmarks_720p, fake_ctx, side) -> None:
    """No black border comes in, neither into the picture nor into the masks."""
    moved = cut_by_the_frame(face_landmarks_720p, side)
    flat = np.full((720, 1280, 3), 0.8, np.float32)
    mask = np.ones((720, 1280), np.float32)
    out, ctx = run(flat, moved, fake_ctx, masks=(mask, None, None), protect_background=0.0, **ALL_UP)
    assert np.abs(out - 0.8).max() < 1e-4
    assert ctx.hair_mask.min() > 1.0 - 1e-4


def test_face_completely_out_of_the_frame_is_a_noop(frame, face_landmarks_720p, fake_ctx) -> None:
    out, _ = run(frame, face_landmarks_720p + np.float32([0.0, 3000.0]), fake_ctx, **ALL_UP)
    assert np.array_equal(out, frame)


@pytest.mark.parametrize("zoom", [0.35, 3.0], ids=["far", "close"])
def test_far_and_close_faces(frame, face_landmarks_720p, fake_ctx, zoom) -> None:
    center = face_landmarks_720p.mean(axis=0)
    target = np.float32([640.0, 360.0])
    scaled = (face_landmarks_720p - center) * zoom + target
    out, ctx = run(frame, scaled, fake_ctx, **ALL_UP)
    assert np.isfinite(out).all() and out.min() >= 0.0 and out.max() <= 1.0
    face_width = float(np.ptp(scaled[regions.FACE_OVAL][:, 0]))
    assert np.linalg.norm(ctx.landmarks - scaled, axis=1).max() <= 0.085 * face_width


@pytest.mark.parametrize(
    "bad",
    [
        lambda lm: np.full_like(lm, np.nan),
        lambda lm: np.where(np.arange(len(lm))[:, None] == 10, np.inf, lm),
        lambda lm: np.full_like(lm, 100.0),
        lambda lm: (lm - lm.mean(axis=0)) * 0.05 + 300.0,
    ],
    ids=["nan", "inf", "collapsed", "tiny"],
)
def test_unusable_landmarks_are_a_noop(frame, face_landmarks_720p, fake_ctx, bad) -> None:
    landmarks = np.asarray(bad(face_landmarks_720p), np.float32)
    work = frame.copy()
    ctx = fake_ctx(frame, landmarks)
    out = ReshapeEffect().apply(work, ctx, ReshapeSettings(**ALL_UP))
    assert out is work and np.array_equal(out, frame)
    assert ctx.landmarks is not None


def test_closed_mouth(frame, face_landmarks_720p, fake_ctx) -> None:
    closed = face_landmarks_720p.copy()
    seam = (closed[regions.LIPS_UPPER_INNER] + closed[regions.LIPS_LOWER_INNER]) / 2
    closed[regions.LIPS_UPPER_INNER] = seam
    closed[regions.LIPS_LOWER_INNER] = seam
    out, ctx = run(frame, closed, fake_ctx, **ALL_UP)
    assert np.isfinite(out).all()
    assert mouth_size(ctx.landmarks) < mouth_size(closed)


def test_a_failure_leaves_frame_and_context_untouched(frame, face_landmarks_720p, fake_ctx, monkeypatch) -> None:
    def fail(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(reshape_module, "_warped_points", fail)
    mask = np.ones(frame.shape[:2], np.float32)
    ctx = fake_ctx(frame, face_landmarks_720p, hair_mask=mask)
    work = frame.copy()
    with pytest.raises(RuntimeError):
        ReshapeEffect().apply(work, ctx, ReshapeSettings(**ALL_UP))
    assert np.array_equal(work, frame)
    assert ctx.landmarks is not None and np.array_equal(ctx.landmarks, face_landmarks_720p)
    assert ctx.hair_mask is mask


# --- pipeline ---------------------------------------------------------------


class FakeTracker:
    def __init__(self, landmarks: np.ndarray | None) -> None:
        self.landmarks = landmarks

    def detect(self, frame_bgr, ts_ms):
        if self.landmarks is None:
            return None, None
        return self.landmarks.copy(), np.zeros(len(self.landmarks), np.float32)

    def set_smoothing(self, smoothing: float) -> None:
        pass

    def close(self) -> None:
        pass


class FakeSegmenter:
    def update(self, frame_bgr, ts_ms, frame_index) -> SegMasks:
        return SegMasks()

    def set_interval(self, interval: int) -> None:
        pass

    def close(self) -> None:
        pass


class Spy(Effect):
    """Runs after everything else and records the context it sees."""

    name = "general"  # any existing section works; never registered
    order = 1000

    def __init__(self) -> None:
        self.seen: list[np.ndarray | None] = []

    def apply(self, frame, ctx, settings):
        self.seen.append(None if ctx.landmarks is None else np.array(ctx.landmarks))
        return frame


def settings_with(values: dict | None = None) -> MakeupSettings:
    settings = MakeupSettings()
    settings.general.mirror_preview = False
    for name, value in (values or {}).items():
        setattr(settings.reshape, name, value)
    return settings


def run_pipeline(frame_u8: np.ndarray, landmarks: np.ndarray, effects: list[Effect], values: dict | None, frames: int = 8):
    pipeline = MakeupPipeline(settings_with(values), tracker=FakeTracker(landmarks), segmenter=FakeSegmenter(), effects=effects)
    try:
        result = None
        for index in range(frames):
            result = pipeline.process(frame_u8, index * 33)
        return result
    finally:
        pipeline.close()


def test_pipeline_with_sliders_at_zero_is_identical_to_no_reshape(face_frame_720p, face_landmarks_720p) -> None:
    plain = run_pipeline(face_frame_720p, face_landmarks_720p, [LipEffect(), EyelinerEffect()], None)
    with_reshape = run_pipeline(face_frame_720p, face_landmarks_720p, [ReshapeEffect(), LipEffect(), EyelinerEffect()], None)
    assert np.array_equal(plain.output, with_reshape.output)
    assert plain.timings_ms.get("reshape") is None and with_reshape.timings_ms.get("reshape") is not None


def test_later_effects_see_the_warped_landmarks(face_frame_720p, face_landmarks_720p) -> None:
    spy = Spy()
    tracker_points = face_landmarks_720p.copy()
    run_pipeline(face_frame_720p, tracker_points, [ReshapeEffect(), spy], {"face_slim": 1.0, "mouth_width": 1.0})
    assert np.linalg.norm(spy.seen[-1] - face_landmarks_720p, axis=1).max() > 3.0
    assert np.array_equal(tracker_points, face_landmarks_720p)  # the tracker's copy stays as it was


def test_lipstick_follows_the_reshaped_lips(face_frame_720p, face_landmarks_720p) -> None:
    values = {"mouth_width": 1.0, "lip_fullness": 1.0, "mouth_corner": 1.0}
    shaped = run_pipeline(face_frame_720p, face_landmarks_720p, [ReshapeEffect()], values)
    spy = Spy()
    made_up = run_pipeline(face_frame_720p, face_landmarks_720p, [ReshapeEffect(), LipEffect(), spy], values)
    delta = np.abs(made_up.output.astype(np.int16) - shaped.output.astype(np.int16)).max(axis=2) > 8
    landmarks = spy.seen[-1]
    roi = regions.Roi(0, 0, face_frame_720p.shape[1], face_frame_720p.shape[0])
    warped_lips = regions.lips_mask(landmarks, roi, outline=0.1) > 0.5
    original_lips = regions.lips_mask(face_landmarks_720p, roi, outline=0.1) > 0.5
    inside_warped = (delta & warped_lips).sum() / delta.sum()
    inside_original = (delta & original_lips).sum() / delta.sum()
    assert inside_warped > 0.95
    assert inside_warped > inside_original + 0.02


def test_eyeliner_follows_the_reshaped_eyes(face_frame_720p, face_landmarks_720p) -> None:
    values = {"eye_size": 1.0, "brow_lift": 1.0}
    shaped = run_pipeline(face_frame_720p, face_landmarks_720p, [ReshapeEffect()], values)
    spy = Spy()
    made_up = run_pipeline(face_frame_720p, face_landmarks_720p, [ReshapeEffect(), EyelinerEffect(), spy], values)
    delta = np.abs(made_up.output.astype(np.int16) - shaped.output.astype(np.int16)).max(axis=2) > 8
    assert delta.any()
    # The liner never enters the (warped) eye opening: the crisp cut-out still lines up.
    roi = regions.Roi(0, 0, face_frame_720p.shape[1], face_frame_720p.shape[0])
    landmarks = spy.seen[-1]
    opening = regions.eye_opening_mask(landmarks, "left", roi, expand_px=-1.5) > 0.99
    opening |= regions.eye_opening_mask(landmarks, "right", roi, expand_px=-1.5) > 0.99
    assert opening.sum() > 100
    assert delta[opening].sum() == 0


def test_pipeline_fades_the_warp_when_the_face_is_lost(face_frame_720p, face_landmarks_720p) -> None:
    """A bow: the warp fades out over the pipeline's fade and ends exactly at the input."""
    tracker = FakeTracker(face_landmarks_720p)
    effect = ReshapeEffect()
    pipeline = MakeupPipeline(settings_with({"face_slim": 1.0}), tracker=tracker, segmenter=FakeSegmenter(), effects=[effect])
    try:
        for index in range(8):
            pipeline.process(face_frame_720p, index * 33)
        tracker.landmarks = None
        differences = []
        for index in range(8, 18):
            result = pipeline.process(face_frame_720p, index * 33)
            differences.append(int(np.abs(result.output.astype(np.int16) - face_frame_720p.astype(np.int16)).sum()))
    finally:
        pipeline.close()
    assert differences[-1] == 0
    assert all(b <= a * 1.05 for a, b in zip(differences, differences[1:]))
    assert max(a - b for a, b in zip(differences, differences[1:])) < 0.6 * differences[0]


def test_a_one_frame_tracking_dropout_does_not_twitch(face_frame_720p, face_landmarks_720p) -> None:
    """Geometry would show the pipeline's one step of fade as a jump of the contour."""
    tracker = FakeTracker(face_landmarks_720p)
    pipeline = MakeupPipeline(settings_with({"face_slim": 1.0, "jaw": 1.0}), tracker=tracker, segmenter=FakeSegmenter(), effects=[ReshapeEffect()])
    try:
        for index in range(8):
            before = pipeline.process(face_frame_720p, index * 33).output.copy()
        tracker.landmarks = None
        during = pipeline.process(face_frame_720p, 8 * 33).output.copy()
        tracker.landmarks = face_landmarks_720p
        after = pipeline.process(face_frame_720p, 9 * 33).output.copy()
    finally:
        pipeline.close()
    assert not np.array_equal(before, face_frame_720p)
    assert np.array_equal(during, before)
    assert np.array_equal(after, before)


def test_held_intensity_follows_a_real_fade_and_the_master_slider() -> None:
    fade = [1.0, 5 / 6, 4 / 6, 3 / 6, 2 / 6, 1 / 6]
    effect = ReshapeEffect()
    held = [effect._held_intensity(value) for value in fade]
    assert held[:2] == [1.0, 1.0]  # the first step is forgiven
    assert all(a > b for a, b in zip(held[1:], held[2:]))  # the rest goes down steadily
    assert max(a - b for a, b in zip(held[1:], held[2:])) < 0.25  # no jump inside the fade...
    assert held[-1] < 0.25  # ...nor from its last value to zero
    # Up again: right away.
    assert effect._held_intensity(1.0) == 1.0
    # The master slider lowered: gets there in a few frames, without jumping.
    settled = [effect._held_intensity(0.5) for _ in range(8)]
    assert settled[0] < 1.0 and settled[-1] == pytest.approx(0.5, abs=0.01)
    assert settled[1] > settled[-1] + 0.01
    assert all(a >= b for a, b in zip(settled, settled[1:]))
    effect.reset()
    assert effect._intensity is None and effect._asked_intensity is None


def test_real_models_end_to_end(face_frame_720p, models_dir) -> None:
    """Real tracker and segmenter: with reshape on, the output differs in the face only,
    and the warped landmarks are still on the face."""
    from makeup_cam.models import FACE_LANDMARKER, ensure_model
    from makeup_cam.tracking import FaceTracker

    spy = Spy()
    tracker = FaceTracker(ensure_model(FACE_LANDMARKER, directory=models_dir), 0.5)
    pipeline = MakeupPipeline(
        settings_with({"face_slim": 0.5, "eye_size": 0.5}), tracker=tracker, segmenter=FakeSegmenter(), effects=[ReshapeEffect(), spy]
    )
    try:
        for index in range(25):
            result = pipeline.process(face_frame_720p, index * 33)
    finally:
        pipeline.close()
    assert result.face_found and spy.seen[-1] is not None
    diff = np.abs(result.output.astype(np.int16) - face_frame_720p.astype(np.int16)).max(axis=2)
    assert diff.max() > 5
    ys, xs = np.nonzero(diff)
    box = regions.roi_around(spy.seen[-1][regions.FACE_OVAL], face_frame_720p.shape, pad=0.5)
    assert box.x0 <= xs.min() and xs.max() < box.x1
    assert box.y0 <= ys.min() and ys.max() < box.y1


def test_warped_masks_are_new_objects_in_reused_memory() -> None:
    """The hair effect tells masks apart by identity, so reusing the memory must not reuse the object."""
    module = reshape_module
    roi = regions.Roi(10, 10, 30, 30)
    area = module._Area(roi, (slice(0, 1), slice(0, 1)), 8)
    map_y, map_x = np.mgrid[10.0:30.0, 11.0:31.0].astype(np.float32)
    memory = [np.empty((0, 0), np.float32)]
    first_mask = np.random.default_rng(0).uniform(0.0, 1.0, (60, 80)).astype(np.float32)
    first_mask.setflags(write=False)
    first = module._warp_mask(first_mask, area, map_x, map_y, memory, 0)
    expected = first_mask.copy()
    expected[roi.slices] = cv2.remap(first_mask, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    np.testing.assert_array_equal(first, expected)
    assert not first.flags.writeable
    second = module._warp_mask(first_mask[::-1].copy(), area, map_x, map_y, memory, 0)
    assert second is not first and np.shares_memory(second, first)
    larger = module._warp_mask(np.zeros((70, 90), np.float32), area, map_x, map_y, memory, 0)
    assert larger.shape == (70, 90)
    assert module._warp_mask(None, area, map_x, map_y, memory, 0) is None


# ---------------------------------------------------------------------------
# No plate: slimming gives way to lines next to the jaw
# ---------------------------------------------------------------------------


def jaw_width_change(frame: np.ndarray, out: np.ndarray, lm: np.ndarray) -> float:
    """How much of the frame changed: the larger the slimming, the larger this is."""
    return float(np.abs(out - frame).mean())


def run_frames(frame: np.ndarray, lm: np.ndarray, fake_ctx, masks, effect: ReshapeEffect, count: int, **sliders) -> np.ndarray:
    out = frame
    for _ in range(count):
        out, _ = run(frame, lm, fake_ctx, masks=masks, effect=effect, **sliders)
    return out


def test_lines_next_to_the_jaw_slowly_take_back_the_slimming(face_landmarks_720p, fake_ctx) -> None:
    lm = face_landmarks_720p
    frame, person = with_person(lined_room(720, 1280), lm)
    masks = (None, person, None)
    effect = ReshapeEffect()
    first = jaw_width_change(frame, run_frames(frame, lm, fake_ctx, masks, effect, 1, face_slim=1.0), lm)
    previous = effect._slim_level
    assert previous == 1.0
    steps = []
    for _ in range(90):
        run(frame, lm, fake_ctx, masks=masks, effect=effect, face_slim=1.0)
        steps.append(previous - effect._slim_level)
        previous = effect._slim_level
    assert effect._slim_level == pytest.approx(reshape_module._BUSY_ROOM_FLOOR, abs=0.01)
    assert max(steps) <= reshape_module._BUSY_ROOM_FALL + 1e-6  # slowly: the face never jumps
    last = jaw_width_change(frame, run_frames(frame, lm, fake_ctx, masks, effect, 1, face_slim=1.0), lm)
    assert last < 0.6 * first


def test_a_plain_wall_keeps_all_of_the_slimming(face_landmarks_720p, fake_ctx) -> None:
    lm = face_landmarks_720p
    wall = np.full((720, 1280, 3), 0.7, np.float32)
    frame, person = with_person(wall, lm)
    effect = ReshapeEffect()
    run_frames(frame, lm, fake_ctx, (None, person, None), effect, 60, face_slim=1.0)
    assert effect._slim_level == 1.0


def test_a_plate_in_use_keeps_all_of_the_slimming(face_landmarks_720p, fake_ctx) -> None:
    lm = face_landmarks_720p
    room = lined_room(720, 1280)
    frame, person = with_person(room, lm)
    effect = ReshapeEffect()
    for _ in range(60):
        run(frame, lm, fake_ctx, masks=(None, person, None), background=room, effect=effect, face_slim=1.0)
    assert effect._plate_on
    assert effect._slim_level == 1.0


def test_the_room_check_leaves_the_other_sliders_alone(face_landmarks_720p, fake_ctx) -> None:
    lm = face_landmarks_720p
    frame, person = with_person(lined_room(720, 1280), lm)
    masks = (None, person, None)
    effect = ReshapeEffect()
    run_frames(frame, lm, fake_ctx, masks, effect, 30, eye_size=1.0)
    assert effect._slim_level == 1.0
    fresh, _ = run(frame, lm, fake_ctx, masks=masks, eye_size=1.0)
    out, _ = run(frame, lm, fake_ctx, masks=masks, effect=effect, eye_size=1.0)
    assert np.array_equal(out, fresh)
