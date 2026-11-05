"""Eyebrows: fill sparse brows and the gaps between hairs, like brow powder.

Area: a smooth brow outline through the brow landmarks (Catmull-Rom curves
along the upper and lower edges, so a crisp setting shows no polygon
corners), made thicker or thinner around its center line with
``thickness``, tapering toward the tail like a real brow, and feathered by
``softness``.

Hair-aware fill: every pixel gets a "hairness" (0 = skin, 1 = a dark hair)
from its luminance between the local skin level (interpolated from the skin
around the brow, so side light or a dim room does not count as hair) and the
darkest hairs of the brow. Blurred, it is the local hair density, which
weights the fill: color goes where the brow really is (onto and between its
hairs) and only lightly onto bare skin inside the outline. When the
landmarks drift from the real brow (e.g. while raising the eyebrows) the
color stays on the hairs instead of drawing a second brow next to them; a
floor keeps sparse tails filled.

Color is multiply with the brow color (pigment over skin) under that smooth
weight. Multiply scales neighboring pixels by the same factor, so the skin
between hairs darkens much more (in absolute terms) than the already dark
hairs: gaps fill in, every hair stays visible, and a hair never ends up
lighter than the skin around it (no flat paint, no inverted texture, even at
full opacity with a very dark color). The color also takes on the scene's
lighting.
"""

from __future__ import annotations

import numpy as np

from .. import blend, regions
from ..config import BrowSettings, hex_to_bgr
from .base import Effect, FrameContext, register_effect

# All lengths below are fractions of the face scale (inter-ocular distance).

# Feather sigma of the brow outline: base + softness slider x range, and
# never below this many pixels (keeps small faces anti-aliased).
_SOFT_BASE = 0.012
_SOFT_RANGE = 0.05
_MIN_FEATHER_PX = 0.5
# Outline: curve samples per landmark segment; half-thickness at the tail
# (before its pointed end) relative to the rest, reached from this fraction
# of the brow length.
_CURVE_SAMPLES = 4
_TAIL_WIDTH = 0.25
_TAPER_START = 0.5
# ROI padding as a fraction of the brow length (covers a thicker brow and
# leaves skin around it for the skin level).
_ROI_PAD = 0.2
# A Gaussian feather is negligible beyond this many sigmas.
_FEATHER_REACH = 3.5
# Mask levels: inside the brow ("core") and clearly outside it (the ring).
_CORE_LEVEL = 0.5
_RING_LEVEL = 0.02
# Skin level: the luminance of the skin in the ring, interpolated across the
# brow with a normalized blur of this sigma. Ring pixels darker than this
# fraction of the ring's bright skin (this percentile) are hair, lashes or
# shadow and are left out.
_SKIN_SIGMA = 0.12
_SKIN_PERCENTILE = 75.0
_SKIN_MIN_FRACTION = 0.85
# Hair level: this luminance percentile of the brow core.
_HAIR_PERCENTILE = 10.0
# Least hair-to-skin contrast assumed, relative to the skin level (keeps
# blond brows and noise from being read as dense dark hair).
_MIN_CONTRAST = 0.12
# Local hair density: hairness blurred with this sigma, normalized by this
# percentile of the density in the core.
_DENSITY_SIGMA = 0.035
_DENSITY_PERCENTILE = 90.0
# Fill where no hair is found (relative to where hairs are dense).
_PRESENCE_FLOOR = 0.3
# Fewer pixels than this in the core or the surrounding skin: no statistics,
# fill evenly at the floor level (tiny or mostly hidden brows).
_MIN_SAMPLES = 12


@register_effect
class BrowEffect(Effect):
    """Brow powder that fills gaps between the hairs and keeps their texture."""

    name = "brows"
    order = 30

    def apply(self, frame: np.ndarray, ctx: FrameContext, settings: BrowSettings) -> np.ndarray:
        strength = _clamp01(settings.opacity) * ctx.intensity
        if strength <= 0.0 or ctx.landmarks is None:
            return frame
        landmarks = ctx.landmarks
        scale = regions.face_scale(landmarks)
        feather_px = max(scale * (_SOFT_BASE + _SOFT_RANGE * _clamp01(settings.softness)), _MIN_FEATHER_PX)
        color = hex_to_bgr(settings.color)
        for side in regions.SIDES:
            outline = _brow_outline(landmarks, side, settings.thickness)
            roi = regions.roi_around(outline, frame.shape, _ROI_PAD, _FEATHER_REACH * feather_px + 2.0)
            if roi is None:
                continue
            area = regions.polygon_mask(outline, roi, feather_px)
            if not area.any():
                continue
            view = frame[roi.slices]
            weight = area * _presence(blend.luminance(view), area, scale) * np.float32(strength)
            frame[roi.slices] = blend.mix(view, blend.blend_multiply(view, color), weight)
        return frame


def _clamp01(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


def _smooth_curve(points: np.ndarray) -> np.ndarray:
    """Uniform Catmull-Rom curve through ``points`` (``_CURVE_SAMPLES`` per segment, ends kept)."""
    p = np.asarray(points, np.float32)
    padded = np.vstack([2 * p[0] - p[1], p, 2 * p[-1] - p[-2]])
    s = np.linspace(0.0, 1.0, _CURVE_SAMPLES, endpoint=False, dtype=np.float32)[:, None]
    s2, s3 = s * s, s * s * s
    basis = ((-s3 + 2 * s2 - s) / 2, (3 * s3 - 5 * s2 + 2) / 2, (-3 * s3 + 4 * s2 + s) / 2, (s3 - s2) / 2)
    segments = [sum(b * padded[i + k] for k, b in enumerate(basis)) for i in range(len(p) - 1)]
    return np.vstack([*segments, p[-1:]])


def _brow_outline(landmarks: np.ndarray, side: regions.Side, thickness: float) -> np.ndarray:
    """Closed brow outline: smooth upper and lower edges, scaled around the center
    line by ``1 + thickness`` (clamped at 0), tapering to a pointed tail."""
    s = regions.side_indices(side)
    upper = _smooth_curve(landmarks[s.brow_upper])
    lower = _smooth_curve(landmarks[s.brow_lower])
    center = (upper + lower) / 2
    t = np.linspace(0.0, 1.0, len(center), dtype=np.float32)
    ramp = np.clip((t - _TAPER_START) / (1.0 - _TAPER_START), 0.0, 1.0)
    taper = 1.0 - (1.0 - _TAIL_WIDTH) * ramp * ramp * (3.0 - 2.0 * ramp)
    half = (upper - lower) / 2 * (max(1.0 + float(thickness), 0.0) * taper)[:, None]
    # The tail ends in a point a tail half-width past the last center point.
    direction = center[-1] - center[-2]
    direction /= max(float(np.linalg.norm(direction)), 1e-6)
    tip = center[-1] + direction * np.float32(np.linalg.norm(half[-1]))
    return np.vstack([center + half, tip, (center - half)[::-1]])


def _presence(luma: np.ndarray, area: np.ndarray, scale: float) -> np.ndarray | float:
    """Smooth fill weight (0..1): the local hair density, with a floor for bare skin."""
    core = area > _CORE_LEVEL
    skin = _skin_around(luma, area < _RING_LEVEL)
    if core.sum() < _MIN_SAMPLES or skin.sum() < _MIN_SAMPLES:
        return _PRESENCE_FLOOR
    density = regions.feather(_hairness(luma, core, skin, scale), _DENSITY_SIGMA * scale)
    full = max(float(np.percentile(density[core], _DENSITY_PERCENTILE)), 1e-3)
    return _PRESENCE_FLOOR + (1.0 - _PRESENCE_FLOOR) * np.clip(density * np.float32(1.0 / full), 0.0, 1.0)


def _skin_around(luma: np.ndarray, ring: np.ndarray) -> np.ndarray:
    """The bright (skin) pixels of the ring around the brow, as a bool mask."""
    if not ring.any():
        return ring
    bright = float(np.percentile(luma[ring], _SKIN_PERCENTILE))
    return ring & (luma >= np.float32(_SKIN_MIN_FRACTION * bright))


def _hairness(luma: np.ndarray, core: np.ndarray, skin: np.ndarray, scale: float) -> np.ndarray:
    """0 for skin-bright pixels, 1 for pixels as dark as the brow's darkest hairs."""
    weight = skin.astype(np.float32)
    sigma = _SKIN_SIGMA * scale
    skin_level = regions.feather(luma * weight, sigma) / np.maximum(regions.feather(weight, sigma), 1e-3)
    hair_level = np.float32(np.percentile(luma[core], _HAIR_PERCENTILE))
    contrast = np.maximum(skin_level - hair_level, np.float32(_MIN_CONTRAST) * skin_level)
    return np.clip((skin_level - luma) / np.maximum(contrast, 1e-3), 0.0, 1.0)
