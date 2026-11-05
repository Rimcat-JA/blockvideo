"""Tests for makeup_cam.blend: blend-mode formulas on known values, exact
no-op behavior, value ranges and dtype preservation."""

from __future__ import annotations

import numpy as np
import pytest

from makeup_cam import blend as B

MODES = ["soft_light", "multiply", "normal", "color", "overlay", "screen"]
SKIN_BGR = (0.62, 0.70, 0.86)
LIP_BGR = (0.38, 0.36, 0.76)


def solid(value: float | tuple[float, float, float], h: int = 4, w: int = 5) -> np.ndarray:
    return np.broadcast_to(np.asarray(value, np.float32), (h, w, 3)).copy()


@pytest.fixture
def image() -> np.ndarray:
    return np.random.default_rng(7).random((24, 32, 3), dtype=np.float32)


@pytest.fixture
def ramp_mask() -> np.ndarray:
    return np.tile(np.linspace(0.0, 1.0, 32, dtype=np.float32), (24, 1))


# --------------------------------------------------------------------------
# mix
# --------------------------------------------------------------------------


def test_mix_formula_and_exact_ends(image: np.ndarray, ramp_mask: np.ndarray) -> None:
    other = np.ones_like(image) - image
    out = B.mix(image, other, ramp_mask)
    m = ramp_mask[..., None]
    np.testing.assert_allclose(out, image * (1 - m) + other * m, atol=1e-6)
    np.testing.assert_array_equal(out[:, 0], image[:, 0])  # mask 0: bit-exact base
    np.testing.assert_allclose(out[:, -1], other[:, -1], atol=1e-6)
    assert out.dtype == np.float32


def test_mix_accepts_colors_and_scalar_masks(image: np.ndarray) -> None:
    out = B.mix(image, (0.0, 0.5, 1.0), 0.25)
    np.testing.assert_allclose(out, image * 0.75 + np.float32([0.0, 0.5, 1.0]) * 0.25, atol=1e-6)
    assert out.dtype == np.float32
    np.testing.assert_array_equal(B.mix(image, (1.0, 1.0, 1.0), np.zeros(image.shape[:2], np.float32)), image)


# --------------------------------------------------------------------------
# Blend formulas on known values
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("fn", "base", "color", "expected"),
    [
        (B.blend_multiply, 0.5, 0.5, 0.25),
        (B.blend_multiply, 0.8, 1.0, 0.8),  # white is neutral
        (B.blend_screen, 0.5, 0.5, 0.75),
        (B.blend_screen, 0.3, 0.0, 0.3),  # black is neutral
        (B.blend_overlay, 0.25, 0.8, 0.4),  # 2 * b * c in the shadows
        (B.blend_overlay, 0.75, 0.8, 0.9),  # 1 - 2 (1 - b)(1 - c) in the highlights
        (B.blend_overlay, 0.3, 0.5, 0.3),  # 50% gray is neutral
        (B.blend_soft_light, 0.25, 1.0, 0.4375),  # (1 - 2c) b^2 + 2 c b
        (B.blend_soft_light, 0.25, 0.0, 0.0625),
        (B.blend_soft_light, 0.6, 0.5, 0.6),  # 50% gray is neutral
        (B.blend_soft_light, 0.0, 0.9, 0.0),
        (B.blend_soft_light, 1.0, 0.1, 1.0),
    ],
)
def test_blend_formulas(fn, base: float, color: float, expected: float) -> None:
    out = fn(solid(base), (color, color, color))
    np.testing.assert_allclose(out, expected, atol=1e-6)


@pytest.mark.parametrize("fn", [B.blend_multiply, B.blend_screen, B.blend_overlay, B.blend_soft_light, B.blend_color])
def test_blend_outputs_stay_in_range_and_keep_dtype(fn, image: np.ndarray) -> None:
    colors = [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0), LIP_BGR, np.random.default_rng(3).random(image.shape, dtype=np.float32)]
    for color in colors:
        out = fn(image, color)
        assert out.shape == image.shape and out.dtype == np.float32
        assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0
    assert fn(image.astype(np.float64), LIP_BGR).dtype == np.float64


def hue_deg(lab: np.ndarray) -> float:
    return float(np.degrees(np.arctan2(lab[2], lab[1])))


def test_blend_color_keeps_lightness_and_takes_hue() -> None:
    lip_lab = B.bgr_to_lab(solid(LIP_BGR))[0, 0]
    for base_bgr in (SKIN_BGR, (0.30, 0.32, 0.55), (0.20, 0.30, 0.25)):
        base = solid(base_bgr)
        out_lab = B.bgr_to_lab(B.blend_color(base, LIP_BGR))[0, 0]
        assert out_lab[0] == pytest.approx(B.bgr_to_lab(base)[0, 0, 0], abs=0.6)
        assert hue_deg(out_lab) == pytest.approx(hue_deg(lip_lab), abs=2.0)


def test_blend_color_reaches_full_chroma_at_the_colors_lightness() -> None:
    lip_lab = B.bgr_to_lab(solid(LIP_BGR))[0, 0]
    gray = B.lab_to_bgr(np.array([[[lip_lab[0], 0.0, 0.0]]], np.float32))
    np.testing.assert_allclose(B.blend_color(gray, LIP_BGR)[0, 0], LIP_BGR, atol=6e-3)


def test_blend_color_with_gray_desaturates_and_keeps_extremes() -> None:
    lab = B.bgr_to_lab(B.blend_color(solid(SKIN_BGR), (0.5, 0.5, 0.5)))[0, 0]
    assert abs(lab[1]) < 0.5 and abs(lab[2]) < 0.5
    extremes = np.array([[[0, 0, 0], [1, 1, 1], [1, 0, 0], [0, 0, 1]]], np.float32)
    out = B.blend_color(extremes, LIP_BGR)
    assert np.isfinite(out).all() and float(out.min()) >= 0.0 and float(out.max()) <= 1.0
    np.testing.assert_allclose(out[0, :2], extremes[0, :2], atol=0.01)  # black and white stay put


# --------------------------------------------------------------------------
# apply_color
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
def test_apply_color_is_an_exact_noop_at_zero(mode: str, image: np.ndarray) -> None:
    assert B.apply_color(image, LIP_BGR, np.ones(image.shape[:2], np.float32), 0.0, mode) is image
    out = B.apply_color(image, LIP_BGR, np.zeros(image.shape[:2], np.float32), 1.0, mode)
    np.testing.assert_array_equal(out, image)


@pytest.mark.parametrize("mode", MODES)
def test_apply_color_masks_and_keeps_dtype(mode: str, image: np.ndarray, ramp_mask: np.ndarray) -> None:
    out = B.apply_color(image, LIP_BGR, ramp_mask, 0.8, mode)
    assert out.dtype == np.float32 and out.shape == image.shape
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0
    np.testing.assert_array_equal(out[:, 0], image[:, 0])
    full = B.apply_color(image, LIP_BGR, None, 0.8, mode)
    np.testing.assert_allclose(out[:, -1], full[:, -1], atol=1e-6)


def test_apply_color_normal_mode_and_opacity(image: np.ndarray) -> None:
    painted = B.apply_color(image, LIP_BGR, None, 1.0, "normal")
    np.testing.assert_allclose(painted, solid(LIP_BGR, *image.shape[:2]), atol=1e-6)
    half = B.apply_color(image, LIP_BGR, None, 0.5, "normal")
    np.testing.assert_allclose(half, (image + np.float32(LIP_BGR)) / 2, atol=1e-6)
    np.testing.assert_array_equal(B.apply_color(image, LIP_BGR, None, 3.0, "normal"), painted)  # clamped


def test_apply_color_rejects_unknown_modes(image: np.ndarray) -> None:
    with pytest.raises(ValueError):
        B.apply_color(image, LIP_BGR, None, 0.5, "hue")  # type: ignore[arg-type]


def test_apply_color_works_on_frame_views(ramp_mask: np.ndarray) -> None:
    frame = np.random.default_rng(1).random((60, 80, 3), dtype=np.float32)
    view = frame[10:34, 20:52]  # non-contiguous ROI view, as effects use
    out = B.apply_color(view, LIP_BGR, ramp_mask, 0.6, "color")
    np.testing.assert_array_equal(out[:, 0], view[:, 0])
    assert out.shape == view.shape


# --------------------------------------------------------------------------
# Color spaces and tone
# --------------------------------------------------------------------------


def test_lab_conversion_reference_values_and_roundtrip(image: np.ndarray) -> None:
    np.testing.assert_allclose(B.bgr_to_lab(solid(1.0))[0, 0], (100.0, 0.0, 0.0), atol=0.01)
    np.testing.assert_allclose(B.bgr_to_lab(solid(0.0))[0, 0], (0.0, 0.0, 0.0), atol=0.01)
    lab = B.bgr_to_lab(image)
    assert lab.dtype == np.float32
    np.testing.assert_allclose(B.lab_to_bgr(lab), image, atol=5e-3)


def test_lab_to_bgr_clips_out_of_gamut() -> None:
    out = B.lab_to_bgr(np.array([[[50.0, 120.0, -120.0], [5.0, -100.0, 100.0]]], np.float32))
    assert out.dtype == np.float32
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


def test_luminance_weights() -> None:
    channels = np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]], np.float32)  # pure B, G, R
    np.testing.assert_allclose(B.luminance(channels)[0], (0.114, 0.587, 0.299), atol=1e-3)
    assert B.luminance(solid(0.5)).shape == (4, 5)


def test_brightness_contrast_formula() -> None:
    img = np.array([[[0.3, 0.3, 0.3], [0.7, 0.7, 0.7]]], np.float32)
    assert B.adjust_brightness_contrast(img, 0.0, 0.0) is img
    np.testing.assert_allclose(B.adjust_brightness_contrast(img, 0.1, 0.0), img + 0.1, atol=1e-6)
    stretched = B.adjust_brightness_contrast(img, 0.0, 0.5, pivot=0.5)
    np.testing.assert_allclose(stretched[0, :, 0], (0.2, 0.8), atol=1e-6)
    auto = B.adjust_brightness_contrast(img, 0.0, 0.5)  # default pivot: mean luminance 0.5
    np.testing.assert_allclose(auto, stretched, atol=1e-5)
    assert float(B.adjust_brightness_contrast(img, 0.9, 1.0).max()) <= 1.0
    assert B.adjust_brightness_contrast(img, 0.0, 0.5).dtype == np.float32


def test_brightness_contrast_mask(image: np.ndarray, ramp_mask: np.ndarray) -> None:
    out = B.adjust_brightness_contrast(image, 0.2, 0.3, mask=ramp_mask)
    np.testing.assert_array_equal(out[:, 0], image[:, 0])
    pivot = float((B.luminance(image) * ramp_mask).sum() / ramp_mask.sum())  # mask-weighted mean
    full = B.adjust_brightness_contrast(image, 0.2, 0.3, pivot=pivot)
    np.testing.assert_allclose(out[:, -1], full[:, -1], atol=1e-6)
    assert out.dtype == np.float32
