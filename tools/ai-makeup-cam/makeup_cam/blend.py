"""Color blending on float images.

Images are BGR float arrays (normally float32) with values in [0, 1];
results keep the input dtype. Masks are HxW (or scalars) and broadcast over
the color channels. Colors are ``(b, g, r)`` tuples in [0, 1] (see
``config.hex_to_bgr``) or arrays broadcastable to the image.

The ``blend_*`` functions only combine pixels with a color; restrict them to
a region and strength with :func:`mix`, or use :func:`apply_color` which
does both. With opacity 0 or a zero mask the base image comes back
unchanged, bit for bit.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Sequence
from typing import Literal

import cv2
import numpy as np

Color = Sequence[float] | np.ndarray
BlendMode = Literal["soft_light", "multiply", "normal", "color", "overlay", "screen"]


def _like(value: Color | float, base: np.ndarray) -> np.ndarray:
    """``value`` as an array of ``base``'s dtype (keeps float32 math float32)."""
    return np.asarray(value, dtype=base.dtype)


def _channel_mask(mask: np.ndarray | float, base: np.ndarray) -> np.ndarray:
    """HxW mask -> HxWx1 so it broadcasts over the channels of ``base``."""
    m = _like(mask, base)
    return m[..., None] if m.ndim == base.ndim - 1 else m


def mix(base: np.ndarray, other: Color | np.ndarray, mask: np.ndarray | float) -> np.ndarray:
    """``base * (1 - mask) + other * mask`` (exactly ``base`` where mask is 0)."""
    out = np.subtract(_like(other, base), base)
    out *= _channel_mask(mask, base)
    out += base
    return out


# --------------------------------------------------------------------------
# Blend modes: f(base, color), no masking. In-place steps keep them cheap.
# --------------------------------------------------------------------------


def blend_multiply(base: np.ndarray, color: Color) -> np.ndarray:
    """Darkens: ``base * color`` (white is neutral)."""
    return base * _like(color, base)


def blend_screen(base: np.ndarray, color: Color) -> np.ndarray:
    """Lightens: ``1 - (1 - base)(1 - color)`` (black is neutral)."""
    out = 1 - base
    out *= 1 - _like(color, base)
    np.subtract(1, out, out=out)
    return out


def blend_overlay(base: np.ndarray, color: Color) -> np.ndarray:
    """Multiply in the base's shadows, screen in its highlights (contrasty)."""
    c = _like(color, base)
    shadows = base * (2 * c)
    highlights = 1 - base
    highlights *= 2 * (1 - c)
    np.subtract(1, highlights, out=highlights)
    return np.where(base < 0.5, shadows, highlights)


def blend_soft_light(base: np.ndarray, color: Color) -> np.ndarray:
    """Gentle tint that keeps shading (Pegtop soft light, 50% gray is neutral).

    ``(1 - 2c) * b^2 + 2c * b`` = ``b * (b + 2c * (1 - b))``: continuous and
    branch-free, very close to the W3C soft-light curve.
    """
    out = 1 - base
    out *= 2 * _like(color, base)
    out += base
    out *= base
    return out


def blend_color(base: np.ndarray, color: Color) -> np.ndarray:
    """Keep the base's lightness (Lab L), take the color's hue and chroma (Lab a, b).

    The most natural way to recolor lips or hair: highlights, shadows and
    texture survive, only the color changes. Where the color's chroma cannot
    be displayed at a pixel's lightness (toward black and white) it is
    reduced to the largest displayable chroma of the same hue, so shadows
    stay dark and highlights stay clean instead of clipping to another hue.
    (A per-pixel color image is simply clipped.)
    """
    lab = bgr_to_lab(base)
    target = bgr_to_lab(_color_image(color))
    if target.shape[:2] == (1, 1):
        a, b = (round(float(v), 3) for v in target[0, 0, 1:])
        lightness_index = cv2.convertScaleAbs(cv2.extractChannel(lab, 0), alpha=255.0 / 100.0)
        fraction = cv2.LUT(lightness_index, _displayable_chroma_lut(a, b))
        lab[..., 1] = a * fraction
        lab[..., 2] = b * fraction
    else:
        lab[..., 1:] = target[..., 1:]
    return lab_to_bgr(lab).astype(base.dtype, copy=False)


def _color_image(color: Color) -> np.ndarray:
    """A color tuple as a 1x1x3 image (arrays with a spatial shape pass through)."""
    c = np.asarray(color, dtype=np.float32)
    return c.reshape(1, 1, 3) if c.ndim == 1 else c


# sRGB (D65) constants matching OpenCV's Lab conversion.
_D65_WHITE = np.array([0.950456, 1.0, 1.088754])
_XYZ_TO_LINEAR_BGR = np.array(
    [
        [0.055648, -0.204043, 1.057311],
        [-0.969256, 1.875991, 0.041556],
        [3.240479, -1.537150, -0.498535],
    ]
)


def _lab_to_linear_bgr(lab: np.ndarray) -> np.ndarray:
    """Unclipped Lab -> linear BGR (cv2 clips, which hides out-of-gamut colors)."""
    fy = (lab[..., 0] + 16.0) / 116.0
    f = np.stack([fy + lab[..., 1] / 500.0, fy, fy - lab[..., 2] / 200.0], axis=-1)
    xyz = np.where(f > 6 / 29, f**3, 3 * (6 / 29) ** 2 * (f - 4 / 29)) * _D65_WHITE
    return xyz @ _XYZ_TO_LINEAR_BGR.T


@functools.lru_cache(maxsize=64)
def _displayable_chroma_lut(a: float, b: float) -> np.ndarray:
    """256-entry float32 table over lightness (index = L * 2.55): the largest
    fraction (0..1) of the chroma ``(a, b)`` inside the sRGB gamut."""
    lightness = np.arange(256) * (100.0 / 255.0)
    low = np.zeros(256)
    high = np.ones(256)
    for _ in range(20):  # bisection; the gamut is star-shaped around the gray axis
        mid = (low + high) / 2
        rgb = _lab_to_linear_bgr(np.stack([lightness, mid * a, mid * b], axis=-1))
        inside = np.all((rgb >= -1e-6) & (rgb <= 1 + 1e-6), axis=-1)
        low = np.where(inside, mid, low)
        high = np.where(inside, high, mid)
    return low.astype(np.float32)


_MODES: dict[str, Callable[[np.ndarray, Color], np.ndarray]] = {
    "soft_light": blend_soft_light,
    "multiply": blend_multiply,
    "normal": lambda base, color: np.broadcast_to(_like(color, base), base.shape),
    "color": blend_color,
    "overlay": blend_overlay,
    "screen": blend_screen,
}


def apply_color(
    base: np.ndarray,
    color_bgr: Color,
    mask: np.ndarray | None,
    opacity: float,
    mode: BlendMode = "soft_light",
) -> np.ndarray:
    """Blend ``color_bgr`` into ``base`` with ``mode``, weighted by ``mask * opacity``.

    ``mask`` None means the whole image. Opacity is clamped to [0, 1]; at 0
    the input array itself is returned. Modes: ``"soft_light"`` (subtle tint),
    ``"multiply"`` (darken), ``"normal"`` (paint over), ``"color"`` (recolor,
    keeps lightness), ``"overlay"``, ``"screen"`` (lighten).
    """
    try:
        blend = _MODES[mode]
    except KeyError:
        raise ValueError(f"unknown blend mode {mode!r}; expected one of {sorted(_MODES)}") from None
    opacity = min(max(float(opacity), 0.0), 1.0)
    if opacity == 0.0:
        return base
    weight = opacity if mask is None else _like(mask, base) * _like(opacity, base)
    return mix(base, blend(base, color_bgr), weight)


# --------------------------------------------------------------------------
# Color spaces and tone.
# --------------------------------------------------------------------------


def bgr_to_lab(img: np.ndarray) -> np.ndarray:
    """BGR [0, 1] -> float32 CIE Lab (sRGB, D65): L in 0..100, a/b about -128..127."""
    return cv2.cvtColor(np.asarray(img, dtype=np.float32), cv2.COLOR_BGR2Lab)


def lab_to_bgr(lab: np.ndarray) -> np.ndarray:
    """float32 Lab -> BGR, clipped to [0, 1] (out-of-gamut colors are clamped)."""
    bgr = cv2.cvtColor(np.asarray(lab, dtype=np.float32), cv2.COLOR_Lab2BGR)
    return np.clip(bgr, 0.0, 1.0, out=bgr)


def luminance(img: np.ndarray) -> np.ndarray:
    """Luma ``0.299 R + 0.587 G + 0.114 B`` as float32 HxW."""
    return cv2.cvtColor(np.asarray(img, dtype=np.float32), cv2.COLOR_BGR2GRAY)


def adjust_brightness_contrast(
    img: np.ndarray,
    brightness: float,
    contrast: float,
    mask: np.ndarray | None = None,
    pivot: float | None = None,
) -> np.ndarray:
    """``(img - pivot) * (1 + contrast) + pivot + brightness``, clipped, blended by ``mask``.

    ``brightness`` is an offset in [0, 1] units (0.1 = +10% of full scale);
    ``contrast`` is relative (0 = unchanged, 0.5 = 50% more, -0.5 = half).
    ``pivot`` is the level contrast stretches around; by default the
    (mask-weighted) mean luminance, so the average tone stays put. With both
    amounts 0 the input array itself is returned.
    """
    if brightness == 0 and contrast == 0:
        return img
    if pivot is None:
        pivot = _mean_luminance(img, mask)
    adjusted = img - _like(pivot, img)
    adjusted *= _like(1.0 + contrast, img)
    adjusted += _like(pivot + brightness, img)
    np.clip(adjusted, 0.0, 1.0, out=adjusted)
    return adjusted if mask is None else mix(img, adjusted, mask)


def _mean_luminance(img: np.ndarray, mask: np.ndarray | None) -> float:
    """Mean luminance of ``img`` weighted by ``mask`` (0.5 for an empty mask)."""
    luma = luminance(img)
    if mask is None:
        return float(luma.mean())
    total = float(np.sum(mask))
    return float(np.sum(luma * mask) / total) if total > 0 else 0.5
