"""BlushEffect on the astronaut portrait (real landmarks): no-ops, locality,
protected features, placement (offsets, head roll, head yaw), segmentation
mask handling and the blend formula."""

from __future__ import annotations

import math
from dataclasses import fields

import numpy as np
import pytest

from makeup_cam import blend, regions
from makeup_cam.config import BlushSettings, MakeupSettings, hex_to_bgr
from makeup_cam.effects import blush as blush_module
from makeup_cam.effects.base import registered_effects
from makeup_cam.effects.blush import BlushEffect

SHAPES = [value for value, _ in next(f for f in fields(BlushSettings) if f.name == "shape").metadata["choices"]]
INVISIBLE = 1.0 / 255.0  # below one 8-bit output step
GRAY = 0.6


@pytest.fixture
def frame(face_frame_720p: np.ndarray) -> np.ndarray:
    return face_frame_720p.astype(np.float32) / np.float32(255.0)


@pytest.fixture
def gray_frame() -> np.ndarray:
    """A flat frame: the change is proportional to the blush mask (for geometry checks)."""
    return np.full((720, 1280, 3), GRAY, np.float32)


def apply_blush(
    frame: np.ndarray, landmarks: np.ndarray | None, fake_ctx, intensity: float = 1.0, face_skin_mask=None, **values
) -> np.ndarray:
    ctx = fake_ctx(frame, landmarks, intensity=intensity, face_skin_mask=face_skin_mask)
    return BlushEffect().apply(frame.copy(), ctx, BlushSettings(**values))


def change(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    return np.abs(after - before).max(axis=2)


def full_roi(frame: np.ndarray) -> regions.Roi:
    return regions.Roi(0, 0, frame.shape[1], frame.shape[0])


def centroid(weights: np.ndarray) -> np.ndarray:
    ys, xs = np.mgrid[: weights.shape[0], : weights.shape[1]]
    total = weights.sum()
    return np.array([(xs * weights).sum() / total, (ys * weights).sum() / total])


def cheek_centroids(diff: np.ndarray, landmarks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Change-weighted centroids of the image-left and image-right blush."""
    split = int(round(float(landmarks[regions.NOSE_TIP][0])))
    left, right = diff.copy(), diff.copy()
    left[:, split:] = 0
    right[:, :split] = 0
    return centroid(left), centroid(right)


def test_registered_under_its_settings_section() -> None:
    assert BlushEffect in registered_effects()
    assert (BlushEffect.name, BlushEffect.order, BlushEffect.needs_face) == ("blush", 60, True)
    assert isinstance(getattr(MakeupSettings(), BlushEffect.name), BlushSettings)


@pytest.mark.parametrize(
    ("intensity", "values", "use_landmarks"),
    [
        (1.0, {"opacity": 0.0, "size": 1.6, "softness": 1.0}, True),
        (0.0, {"opacity": 1.0}, True),
        (1.0, {"opacity": 1.0}, False),
    ],
    ids=["opacity-0", "intensity-0", "no-face"],
)
def test_exact_noop(frame, face_landmarks_720p, fake_ctx, intensity, values, use_landmarks) -> None:
    landmarks = face_landmarks_720p if use_landmarks else None
    out = apply_blush(frame, landmarks, fake_ctx, intensity=intensity, **values)
    assert out.dtype == np.float32
    assert np.array_equal(out, frame)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize(
    "extremes",
    [
        {},
        {"opacity": 1.0, "size": 1.6, "offset_x": 0.5, "offset_y": -0.5, "softness": 1.0},
        {"opacity": 1.0, "size": 0.5, "offset_x": -0.5, "offset_y": 0.5, "softness": 0.0},
    ],
    ids=["default", "big-out-up-soft", "small-in-down-hard"],
)
def test_every_shape_gives_a_valid_frame(frame, face_landmarks_720p, fake_ctx, shape, extremes) -> None:
    out = apply_blush(frame, face_landmarks_720p, fake_ctx, shape=shape, **extremes)
    assert out.dtype == np.float32 and out.shape == frame.shape
    assert np.isfinite(out).all()
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert change(frame, out).max() > 0.01


def test_default_blush_sits_on_the_cheeks_only(frame, face_landmarks_720p, fake_ctx) -> None:
    out = apply_blush(frame, face_landmarks_720p, fake_ctx)
    diff = change(frame, out)
    roi = full_roi(frame)
    for side in regions.SIDES:
        x, y = np.round(regions.cheek_center(face_landmarks_720p, side)).astype(int)
        assert diff[y - 3 : y + 4, x - 3 : x + 4].mean() > 0.01
        eye = regions.eye_opening_mask(face_landmarks_720p, side, roi) > 0
        assert diff[eye].max() < INVISIBLE
    lips = regions.polygon_mask(face_landmarks_720p[regions.LIPS_OUTER], roi) > 0
    ridge = face_landmarks_720p[regions.NOSE_BRIDGE + [regions.NOSE_TIP]]
    nose = regions.polyline_mask(ridge, 0.2 * regions.face_scale(face_landmarks_720p), roi) > 0
    outside_face = regions.face_oval_mask(face_landmarks_720p, roi) == 0
    for protected in (lips, nose, outside_face):
        assert diff[protected].max() < INVISIBLE
    # The flat gray margins of the frame are background: bit-identical.
    assert np.array_equal(out[:, :280], frame[:, :280]) and np.array_equal(out[:, 1000:], frame[:, 1000:])


def test_strong_blush_still_spares_eyes_and_lips(frame, face_landmarks_720p, fake_ctx) -> None:
    values = {"opacity": 1.0, "size": 1.6, "softness": 1.0, "shape": "round"}
    diff = change(frame, apply_blush(frame, face_landmarks_720p, fake_ctx, **values))
    roi = full_roi(frame)
    lips = regions.lips_mask(face_landmarks_720p, roi) > 0
    eyes = [regions.eye_opening_mask(face_landmarks_720p, side, roi) > 0 for side in regions.SIDES]
    for protected in (lips, *eyes):
        assert diff[protected].max() < 2 * INVISIBLE
    assert diff.max() > 0.1


def test_strength_grows_with_intensity_and_opacity(frame, face_landmarks_720p, fake_ctx) -> None:
    def strength(intensity: float, opacity: float) -> float:
        return float(change(frame, apply_blush(frame, face_landmarks_720p, fake_ctx, intensity, opacity=opacity)).sum())

    by_intensity = [strength(i, 0.5) for i in (0.25, 0.5, 1.0)]
    by_opacity = [strength(1.0, o) for o in (0.2, 0.5, 1.0)]
    assert by_intensity == sorted(by_intensity) and len(set(by_intensity)) == 3 and by_intensity[0] > 0
    assert by_opacity == sorted(by_opacity) and len(set(by_opacity)) == 3 and by_opacity[0] > 0


def test_blush_adds_the_color_to_the_skin(frame, face_landmarks_720p, fake_ctx) -> None:
    x, y = np.round(regions.cheek_center(face_landmarks_720p, "right")).astype(int)
    patch = (slice(y - 3, y + 4), slice(x - 3, x + 4))
    before = blend.bgr_to_lab(frame[patch]).mean(axis=(0, 1))
    pink = blend.bgr_to_lab(apply_blush(frame, face_landmarks_720p, fake_ctx, opacity=0.6)[patch]).mean(axis=(0, 1))
    blue = apply_blush(frame, face_landmarks_720p, fake_ctx, opacity=0.6, color="#6080E0")
    assert pink[1] > before[1] + 3  # redder
    assert blend.bgr_to_lab(blue[patch]).mean(axis=(0, 1))[2] < before[2] - 3  # bluer


def test_offsets_move_both_blushes_outward_and_down(gray_frame, face_landmarks_720p, fake_ctx) -> None:
    def centroids(**values) -> tuple[np.ndarray, np.ndarray]:
        out = apply_blush(gray_frame, face_landmarks_720p, fake_ctx, opacity=1.0, **values)
        return cheek_centroids(change(gray_frame, out), face_landmarks_720p)

    (left0, right0), (left_out, right_out), (left_down, right_down) = (
        centroids(),
        centroids(offset_x=0.5),
        centroids(offset_y=0.5),
    )
    scale = regions.face_scale(face_landmarks_720p)
    assert left_out[0] < left0[0] - 0.05 * scale and right_out[0] > right0[0] + 0.05 * scale
    assert left_down[1] > left0[1] + 0.05 * scale and right_down[1] > right0[1] + 0.05 * scale


@pytest.mark.parametrize("roll_deg", [-15.0, 15.0])
def test_blush_follows_head_roll(gray_frame, face_landmarks_720p, fake_ctx, roll_deg) -> None:
    pivot = face_landmarks_720p[regions.NOSE_TIP]
    angle = math.radians(roll_deg)
    rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]], np.float32)
    rolled = (face_landmarks_720p - pivot) @ rotation.T + pivot
    out = apply_blush(gray_frame, rolled, fake_ctx, opacity=1.0, shape="lifted")
    left, right = cheek_centroids(change(gray_frame, out), rolled)
    measured = math.degrees(math.atan2(right[1] - left[1], right[0] - left[0]))
    eyes = regions.eye_center(rolled, "left") - regions.eye_center(rolled, "right")
    assert measured == pytest.approx(math.degrees(math.atan2(eyes[1], eyes[0])), abs=3.0)


def test_far_cheek_blush_narrows_when_the_head_turns(gray_frame, face_landmarks_720p, fake_ctx) -> None:
    """Squeeze the image-right half of the face horizontally, like a head turned
    to that side: that blush must become narrower than the near one."""
    center_x = float(face_landmarks_720p[regions.NOSE_TIP][0])
    turned = face_landmarks_720p.copy()
    far = turned[:, 0] > center_x
    turned[far, 0] = center_x + (turned[far, 0] - center_x) * 0.55
    diff = change(gray_frame, apply_blush(gray_frame, turned, fake_ctx, opacity=1.0))
    columns = (diff > 0.02).any(axis=0)
    split = int(round(center_x))
    near_width, far_width = columns[:split].sum(), columns[split:].sum()
    assert 0 < far_width < 0.8 * near_width


def test_face_skin_mask_removes_blush_from_non_skin(frame, face_landmarks_720p, fake_ctx) -> None:
    split = int(round(float(face_landmarks_720p[regions.NOSE_TIP][0])))
    skin = np.ones(frame.shape[:2], np.float32)
    skin[:, split:] = 0.0  # e.g. hair over the image-right cheek
    diff = change(frame, apply_blush(frame, face_landmarks_720p, fake_ctx, face_skin_mask=skin))
    assert diff[:, : split - 10].max() > 0.01
    assert diff[:, split + 10 :].max() == 0.0


def test_a_face_skin_mask_that_misses_the_face_is_ignored(frame, face_landmarks_720p, fake_ctx) -> None:
    without = apply_blush(frame, face_landmarks_720p, fake_ctx)
    empty = np.zeros(frame.shape[:2], np.float32)
    assert np.array_equal(apply_blush(frame, face_landmarks_720p, fake_ctx, face_skin_mask=empty), without)


def test_blush_beyond_the_frame_border(frame, face_landmarks_720p, fake_ctx) -> None:
    edge = face_landmarks_720p + np.float32([-float(face_landmarks_720p[regions.NOSE_TIP][0]), 0.0])
    out = apply_blush(frame, edge, fake_ctx, opacity=1.0, size=1.6)
    assert out.min() >= 0.0 and out.max() <= 1.0
    gone = face_landmarks_720p + np.float32([5000.0, 0.0])
    assert np.array_equal(apply_blush(frame, gone, fake_ctx, opacity=1.0), frame)


def test_flush_equals_the_soft_light_multiply_mix(frame) -> None:
    base = np.ascontiguousarray(frame[100:180, 400:520])
    weight = np.linspace(0.0, 1.0, base.shape[0] * base.shape[1], dtype=np.float32).reshape(base.shape[:2])
    color = hex_to_bgr("#E8878C")
    share = blush_module._MULTIPLY_SHARE
    tinted = blend.mix(blend.blend_soft_light(base, color), blend.blend_multiply(base, color), share)
    expected = blend.mix(base, tinted, weight)
    result = blush_module._flush(base, color, weight)
    np.testing.assert_allclose(result, expected, atol=1e-6)
    assert np.array_equal(result[weight == 0], base[weight == 0])


def test_unknown_shape_is_rejected(frame, face_landmarks_720p, fake_ctx) -> None:
    with pytest.raises(ValueError, match="shape"):
        apply_blush(frame, face_landmarks_720p, fake_ctx, shape="heart")
