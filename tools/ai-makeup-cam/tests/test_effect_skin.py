"""Tests for the skin retouching effect (makeup_cam/effects/skin.py)."""

from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np
import pytest

from makeup_cam import regions
from makeup_cam.config import SkinSettings, section_fields
from makeup_cam.effects.base import FrameContext, registered_effects
from makeup_cam.effects.skin import SkinEffect

SKIN_BGR = (0.55, 0.65, 0.80)
BACKGROUND = 0.3
SPOT_BGR = (0.45, 0.52, 0.68)
SLIDERS = ("smoothing", "blemish", "redness", "under_eye", "tone_strength", "brightness")


def only(**amounts: float) -> SkinSettings:
    """Settings with every slider at 0 except the given ones."""
    return replace(SkinSettings(), **{**dict.fromkeys(SLIDERS, 0.0), **amounts})


def strong() -> SkinSettings:
    return replace(
        SkinSettings(),
        smoothing=1.0,
        texture=0.2,
        blemish=1.0,
        redness=1.0,
        under_eye=1.0,
        tone_strength=0.8,
        brightness=0.25,
    )


def scale_of(landmarks: np.ndarray) -> float:
    return regions.face_scale(landmarks)


def synthetic_face(landmarks: np.ndarray, noise: float = 0.0, seed: int = 0) -> np.ndarray:
    """Flat skin-colored face oval on a gray background (float32 BGR), optional texture noise."""
    frame = np.full((720, 1280, 3), BACKGROUND, np.float32)
    oval = regions.face_oval_mask(landmarks, regions.Roi(0, 0, 1280, 720), expand_px=0.3 * scale_of(landmarks))
    frame += (np.array(SKIN_BGR, np.float32) - frame) * oval[..., None]
    if noise:
        grain = np.random.default_rng(seed).standard_normal(frame.shape).astype(np.float32) * noise
        frame = np.clip(frame + grain * oval[..., None], 0.0, 1.0)
    return frame


def as_float(frame_u8: np.ndarray) -> np.ndarray:
    return frame_u8.astype(np.float32) / 255.0


def apply(frame: np.ndarray, ctx: FrameContext, settings: SkinSettings) -> np.ndarray:
    """Apply on a copy (the effect may edit its input in place)."""
    return SkinEffect().apply(frame.copy(), ctx, settings)


def face_roi(landmarks: np.ndarray, margin: float = 0.0) -> regions.Roi:
    roi = regions.roi_around(landmarks[regions.FACE_OVAL], (720, 1280), pad=0.0, margin_px=margin * scale_of(landmarks))
    assert roi is not None
    return roi


def lab_at(image: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """CIE Lab of the pixel at ``xy``."""
    x, y = int(xy[0]), int(xy[1])
    return cv2.cvtColor(image[y : y + 1, x : x + 1], cv2.COLOR_BGR2Lab)[0, 0]


def luma_at(image: np.ndarray, xy: np.ndarray) -> float:
    return float(lab_at(image, xy)[0])


def mean_change(before: np.ndarray, after: np.ndarray, roi: regions.Roi) -> float:
    return float(np.abs(after[roi.slices] - before[roi.slices]).mean())


# --------------------------------------------------------------------------
# Contract.
# --------------------------------------------------------------------------


def test_effect_contract():
    assert SkinEffect.name == "skin"
    assert SkinEffect.order == 20
    assert SkinEffect.needs_face and not SkinEffect.needs_hair_mask
    assert SkinEffect in registered_effects()


def test_all_sliders_zero_is_exact_noop(fake_ctx, face_landmarks_720p):
    frame = synthetic_face(face_landmarks_720p, noise=0.02)
    original = frame.copy()
    settings = replace(only(), texture=0.7, tone_color="#123456")
    out = SkinEffect().apply(frame, fake_ctx(frame, face_landmarks_720p), settings)
    assert out is frame
    np.testing.assert_array_equal(out, original)


def test_zero_intensity_is_exact_noop(fake_ctx, face_landmarks_720p):
    frame = synthetic_face(face_landmarks_720p, noise=0.02)
    original = frame.copy()
    out = SkinEffect().apply(frame, fake_ctx(frame, face_landmarks_720p, intensity=0.0), strong())
    np.testing.assert_array_equal(out, original)


def test_without_landmarks_is_noop(fake_ctx):
    frame = np.random.default_rng(1).random((720, 1280, 3), dtype=np.float32)
    original = frame.copy()
    out = SkinEffect().apply(frame, fake_ctx(frame, None), strong())
    np.testing.assert_array_equal(out, original)


@pytest.mark.parametrize("case", ["tiny", "outside"])
def test_unusable_face_is_noop(fake_ctx, face_landmarks_720p, case):
    center = face_landmarks_720p.mean(axis=0)
    if case == "tiny":  # face scale far below the minimum
        landmarks = center + (face_landmarks_720p - center) * 0.05
    else:  # entirely outside the frame
        landmarks = face_landmarks_720p + np.array([2000.0, 0.0], np.float32)
    frame = synthetic_face(face_landmarks_720p, noise=0.02)
    original = frame.copy()
    out = SkinEffect().apply(frame, fake_ctx(frame, landmarks), strong())
    np.testing.assert_array_equal(out, original)


# --------------------------------------------------------------------------
# Output validity and locality.
# --------------------------------------------------------------------------


def test_output_is_float32_in_range(fake_ctx, face_frame_720p, face_landmarks_720p):
    frame = as_float(face_frame_720p)
    out = apply(frame, fake_ctx(frame, face_landmarks_720p), strong())
    assert out.dtype == np.float32
    assert out.shape == frame.shape
    assert np.isfinite(out).all()
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_pixels_away_from_the_face_are_unchanged(fake_ctx, face_frame_720p, face_landmarks_720p):
    frame = as_float(face_frame_720p)
    out = apply(frame, fake_ctx(frame, face_landmarks_720p), strong())
    outside = np.ones(frame.shape[:2], bool)
    outside[face_roi(face_landmarks_720p, margin=0.2).slices] = False
    np.testing.assert_array_equal(out[outside], frame[outside])
    for y, x in [(0, 0), (0, 1279), (719, 0), (719, 1279)]:  # background corners
        np.testing.assert_array_equal(out[y, x], frame[y, x])
    assert mean_change(frame, out, face_roi(face_landmarks_720p)) > 0.005


def test_eyes_and_mouth_opening_are_untouched(fake_ctx, face_frame_720p, face_landmarks_720p):
    frame = as_float(face_frame_720p)
    out = apply(frame, fake_ctx(frame, face_landmarks_720p), strong())
    lm = face_landmarks_720p
    mouth = (lm[regions.LIPS_INNER_TOP] + lm[regions.LIPS_INNER_BOTTOM]) / 2  # teeth
    for x, y in [lm[regions.LEFT_IRIS_CENTER], lm[regions.RIGHT_IRIS_CENTER], mouth]:
        window = (slice(int(y) - 1, int(y) + 2), slice(int(x) - 1, int(x) + 2))
        np.testing.assert_array_equal(out[window], frame[window])


def test_intensity_scales_monotonically(fake_ctx, face_frame_720p, face_landmarks_720p):
    frame = as_float(face_frame_720p)
    roi = face_roi(face_landmarks_720p)
    changes = [
        mean_change(frame, apply(frame, fake_ctx(frame, face_landmarks_720p, intensity=k), SkinSettings()), roi)
        for k in (0.25, 0.5, 0.75, 1.0)
    ]
    assert changes[0] > 0
    assert all(a < b for a, b in zip(changes, changes[1:]))


@pytest.mark.parametrize("slider", SLIDERS)
def test_each_slider_alone_changes_only_the_face(fake_ctx, face_landmarks_720p, slider):
    frame = synthetic_face(face_landmarks_720p, noise=0.03)  # the grain has red-ish and dark pixels
    # A spot for the blemish cover.
    cheek = regions.cheek_center(face_landmarks_720p, "right")
    cv2.circle(frame, (int(cheek[0]), int(cheek[1])), 3, SPOT_BGR, -1, cv2.LINE_AA)
    out = apply(frame, fake_ctx(frame, face_landmarks_720p), only(**{slider: 0.3 if slider == "brightness" else 1.0}))
    assert mean_change(frame, out, face_roi(face_landmarks_720p)) > 1e-4
    outside = np.ones(frame.shape[:2], bool)
    outside[face_roi(face_landmarks_720p, margin=0.2).slices] = False
    np.testing.assert_array_equal(out[outside], frame[outside])


def _slider_extremes() -> list[tuple[str, float]]:
    cases = []
    for name, meta, _ in section_fields(SkinSettings):
        if meta["kind"] == "float":
            cases += [(name, meta["min"]), (name, meta["max"])]
    return cases


@pytest.mark.parametrize(("name", "value"), _slider_extremes())
def test_every_slider_extreme_gives_valid_output(fake_ctx, face_frame_720p, face_landmarks_720p, name, value):
    frame = as_float(face_frame_720p)
    out = apply(frame, fake_ctx(frame, face_landmarks_720p), replace(strong(), **{name: value}))
    assert out.dtype == np.float32 and out.shape == frame.shape
    assert np.isfinite(out).all() and out.min() >= 0.0 and out.max() <= 1.0


@pytest.mark.parametrize("tone_color", ["#FFFFFF", "#000000", "#F1D3C0", "#8D5A3C", "#3060FF"])
def test_any_foundation_color_gives_valid_output(fake_ctx, face_frame_720p, face_landmarks_720p, tone_color):
    frame = as_float(face_frame_720p)
    settings = replace(strong(), tone_strength=1.0, tone_color=tone_color)
    out = apply(frame, fake_ctx(frame, face_landmarks_720p), settings)
    assert np.isfinite(out).all() and out.min() >= 0.0 and out.max() <= 1.0


# --------------------------------------------------------------------------
# What each correction does.
# --------------------------------------------------------------------------


def test_blemish_cover_fills_spots_but_keeps_lines(fake_ctx, face_landmarks_720p):
    lm = face_landmarks_720p
    clean = synthetic_face(lm)
    frame = clean.copy()
    radius = max(1, round(0.04 * scale_of(lm)))
    spot = regions.cheek_center(lm, "right")
    cv2.circle(frame, (int(spot[0]), int(spot[1])), radius, SPOT_BGR, -1, cv2.LINE_AA)
    line = regions.cheek_center(lm, "left")  # a crease-like line of the same darkness
    half = int(0.3 * scale_of(lm))
    x, y = int(line[0]), int(line[1])
    cv2.line(frame, (x - half, y), (x + half, y), SPOT_BGR, radius, cv2.LINE_AA)
    out = apply(frame, fake_ctx(frame, lm), only(blemish=1.0))

    def deficit(image: np.ndarray, xy: np.ndarray) -> float:
        return luma_at(clean, xy) - luma_at(image, xy)

    assert deficit(out, spot) < 0.5 * deficit(frame, spot)
    assert deficit(out, line) > 0.9 * deficit(frame, line)


def test_redness_pulls_red_patches_toward_the_skin_tone(fake_ctx, face_landmarks_720p):
    lm = face_landmarks_720p
    plain = synthetic_face(lm)
    frame = plain.copy()
    cheek = regions.cheek_center(lm, "right")
    cv2.circle(frame, (int(cheek[0]), int(cheek[1])), int(0.25 * scale_of(lm)), (0.50, 0.55, 0.85), -1, cv2.LINE_AA)
    frame = cv2.GaussianBlur(frame, (0, 0), 0.05 * scale_of(lm))
    out = apply(frame, fake_ctx(frame, lm), only(redness=1.0))
    skin_a = lab_at(plain, cheek)[1]
    before, after = lab_at(frame, cheek)[1] - skin_a, lab_at(out, cheek)[1] - skin_a
    assert before > 10
    assert after < 0.3 * before


def test_under_eye_concealer_lifts_dark_circles(fake_ctx, face_landmarks_720p):
    lm = face_landmarks_720p
    frame = synthetic_face(lm)
    indices = regions.side_indices("right")
    center = ((lm[indices.under_eye_top] + lm[indices.under_eye_bottom]) / 2).mean(axis=0)
    s = scale_of(lm)
    cv2.ellipse(frame, (int(center[0]), int(center[1])), (int(0.3 * s), int(0.08 * s)), 0, 0, 360, (0.45, 0.5, 0.6), -1)
    frame = cv2.GaussianBlur(frame, (0, 0), 0.03 * s)
    out = apply(frame, fake_ctx(frame, lm), only(under_eye=1.0))
    before, after = lab_at(frame, center), lab_at(out, center)
    assert after[0] > before[0] + 2  # lighter
    assert after[2] > before[2]  # warmer (less blue)


@pytest.mark.parametrize("brightness", [0.2, -0.2])
def test_brightness_lifts_or_lowers_the_skin(fake_ctx, face_landmarks_720p, brightness):
    lm = face_landmarks_720p
    frame = synthetic_face(lm)
    cheek = regions.cheek_center(lm, "left")
    out = apply(frame, fake_ctx(frame, lm), only(brightness=brightness))
    change = luma_at(out, cheek) - luma_at(frame, cheek)
    assert change * np.sign(brightness) > 3


def test_foundation_moves_the_skin_chroma_toward_the_shade(fake_ctx, face_landmarks_720p):
    lm = face_landmarks_720p
    frame = synthetic_face(lm)
    cheek = regions.cheek_center(lm, "left")
    shade = "#C08A60"
    target = cv2.cvtColor(np.array([[[0x60, 0x8A, 0xC0]]], np.float32) / 255.0, cv2.COLOR_BGR2Lab)[0, 0]
    out = apply(frame, fake_ctx(frame, lm), replace(only(tone_strength=1.0), tone_color=shade))
    distance_before = np.linalg.norm(lab_at(frame, cheek)[1:] - target[1:])
    distance_after = np.linalg.norm(lab_at(out, cheek)[1:] - target[1:])
    assert distance_after < 0.6 * distance_before


def test_smoothing_reduces_texture_and_texture_slider_keeps_some(fake_ctx, face_landmarks_720p):
    lm = face_landmarks_720p
    frame = synthetic_face(lm, noise=0.03)
    cheek = regions.cheek_center(lm, "left")
    half = int(0.15 * scale_of(lm))
    x, y = int(cheek[0]), int(cheek[1])
    window = (slice(y - half, y + half), slice(x - half, x + half))

    def grain(image: np.ndarray) -> float:
        return float(image[window].std(axis=(0, 1)).mean())

    smooth = apply(frame, fake_ctx(frame, lm), replace(only(smoothing=1.0), texture=0.0))
    textured = apply(frame, fake_ctx(frame, lm), replace(only(smoothing=1.0), texture=0.6))
    assert grain(smooth) < 0.5 * grain(frame)
    assert grain(smooth) < grain(textured) < grain(frame)


# --------------------------------------------------------------------------
# Segmentation masks.
# --------------------------------------------------------------------------


def _hair_over_cheek(landmarks: np.ndarray) -> tuple[np.ndarray, tuple[slice, slice]]:
    """A hair mask covering the image-left part of the face (a lock over one cheek),
    and a window on that cheek well inside the hair."""
    s = scale_of(landmarks)
    cheek = regions.cheek_center(landmarks, "right")  # subject's right = image left
    hair = np.zeros((720, 1280), np.float32)
    hair[:, : int(cheek[0] + 0.4 * s)] = 1.0
    half = int(0.08 * s)
    window = (slice(int(cheek[1]) - half, int(cheek[1]) + half), slice(int(cheek[0]) - half, int(cheek[0]) + half))
    return hair, window


@pytest.mark.parametrize("mask_kind", ["face_skin", "hair"])
def test_segmentation_keeps_hair_untouched(fake_ctx, face_landmarks_720p, mask_kind):
    lm = face_landmarks_720p
    frame = synthetic_face(lm, noise=0.03)
    hair, window = _hair_over_cheek(lm)
    masks = {"face_skin_mask": 1.0 - hair} if mask_kind == "face_skin" else {"hair_mask": hair}
    with_mask = apply(frame, fake_ctx(frame, lm, **masks), only(smoothing=1.0))
    without = apply(frame, fake_ctx(frame, lm), only(smoothing=1.0))
    np.testing.assert_array_equal(with_mask[window], frame[window])
    assert np.abs(without[window] - frame[window]).max() > 0.005


def test_face_skin_mask_that_misses_the_face_is_ignored(fake_ctx, face_landmarks_720p):
    lm = face_landmarks_720p
    frame = synthetic_face(lm, noise=0.03)
    empty = np.zeros(frame.shape[:2], np.float32)
    with_empty = apply(frame, fake_ctx(frame, lm, face_skin_mask=empty), SkinSettings())
    without = apply(frame, fake_ctx(frame, lm), SkinSettings())
    np.testing.assert_array_equal(with_empty, without)


@pytest.mark.parametrize("masks", ["none", "hair", "face_skin", "both", "body_skin"])
def test_runs_with_any_combination_of_masks(fake_ctx, face_frame_720p, face_landmarks_720p, masks):
    frame = as_float(face_frame_720p)
    full = np.ones(frame.shape[:2], np.float32)
    kwargs = {
        "none": {},
        "hair": {"hair_mask": 0.0 * full},
        "face_skin": {"face_skin_mask": full},
        "both": {"hair_mask": 0.0 * full, "face_skin_mask": full},
        "body_skin": {"body_skin_mask": full},
    }[masks]
    out = apply(frame, fake_ctx(frame, face_landmarks_720p, **kwargs), SkinSettings())
    assert mean_change(frame, out, face_roi(face_landmarks_720p)) > 1e-3


def test_read_only_inputs_are_not_modified(fake_ctx, face_frame_720p, face_landmarks_720p):
    """The pipeline hands out read-only landmarks and masks."""
    frame = as_float(face_frame_720p)
    landmarks = face_landmarks_720p.copy()
    face_skin = np.ones(frame.shape[:2], np.float32)
    for array in (landmarks, face_skin):
        array.setflags(write=False)
    ctx = fake_ctx(frame, landmarks, face_skin_mask=face_skin)
    apply(frame, ctx, strong())
    np.testing.assert_array_equal(landmarks, face_landmarks_720p)
    assert (face_skin == 1.0).all()


# --------------------------------------------------------------------------
# Real face.
# --------------------------------------------------------------------------


def test_real_face_defaults_are_subtle(fake_ctx, face_frame_720p, face_landmarks_720p):
    """Natural look: a gentle, smooth change confined to the face."""
    frame = as_float(face_frame_720p)
    out = apply(frame, fake_ctx(frame, face_landmarks_720p), SkinSettings())
    roi = face_roi(face_landmarks_720p)
    change = np.abs(out - frame)
    assert 0.002 < float(change[roi.slices].mean()) < 0.03
    assert float(change.max()) < 0.2
    outside = np.ones(frame.shape[:2], bool)
    outside[face_roi(face_landmarks_720p, margin=0.2).slices] = False
    assert not change[outside].any()


def test_face_partly_outside_the_frame(fake_ctx, face_frame_720p, face_landmarks_720p):
    frame = as_float(face_frame_720p)
    shift = np.array([0.0, -face_landmarks_720p[:, 1].mean()], np.float32)  # upper half above the frame
    out = apply(frame, fake_ctx(frame, face_landmarks_720p + shift), strong())
    assert np.isfinite(out).all() and out.min() >= 0.0 and out.max() <= 1.0
