"""Eye makeup: eyeshadow on the upper lid and eyeliner along the lash line.

Both effects work eye by eye inside a small ROI and size everything by the
face scale (inter-ocular distance), so they look the same near and far. Each
eye gets a face-aligned frame (``_EyeAxes``): "along" runs from the inner to
the outer corner, "up" points toward the brow, so the makeup turns with a
tilted head.

Eyeshadow
    Region: the lid between the upper lash line and the lower edge of the
    brow, from a little past the inner corner to past the outer corner. The
    lash line is lifted by a small guard and the eye opening is cut out with
    a crisp edge, so the eyeball is never tinted; when the eye closes, the
    lash line moves down and the shadow covers the closed lid. Past the
    outer corner the shadow fades out softly, above a soft line rising from
    the corner at a small angle (a lifted, never drooping, outer corner).
    Density uses lid coordinates: ``t`` = distance to the lash line over
    (distance to the lash line + distance to the brow), divided by
    ``height`` (0 at the lash line, 1 at the chosen height). Its iso-lines
    follow the lash line near the eye and the brow near the top, so the
    shadow fans out toward the brow tail like a hand-blended one. The
    density is full near the lash line and fades out around ``t = 1``
    (``softness`` widens the fade), is a little deeper at the lash line
    than at the top, and is deeper toward the outer corner than at the
    inner corner.
    Color is a mix of multiply and soft light relative to the lid
    underneath: the lid keeps its texture and shading, and the shadow takes
    on the scene's light. Shimmer adds a soft sheen on the center of the
    mobile lid, where a real lid catches the light: a screen with a light
    tint of the shadow, scaled by the scene's light level and by how lit
    each lid pixel already is. It is anchored to the landmarks, so it never
    flickers.

Eyeliner
    A stroke along the upper lash line (inner -> outer corner), tapered from
    hairline-thin at the inner corner to full width toward the outer corner,
    then continued by a wing that rises at a fixed angle from the eye line
    and tapers to a point (its length grows with ``wing``). The stroke is
    lifted by most of its half-width so it sits on the lash line rather than
    over the eye, and the eye opening is cut out with a crisp edge: the liner
    never reaches the eyeball, and with the eye closed it follows the lid.
    ``lower`` adds a softer, thinner line along the outer two thirds of the
    lower lash line. Ink is mostly "darken" (paint never lightens the lashes)
    with some multiply, which deepens the lash line under the stroke.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .. import blend, regions
from ..config import EyelinerSettings, EyeshadowSettings, hex_to_bgr
from .base import Effect, FrameContext, register_effect

Bgr = tuple[float, float, float]

# All lengths below are fractions of the face scale (inter-ocular distance)
# unless noted otherwise.

# Eyes narrower than this (in pixels) are skipped: nothing visible to paint.
_MIN_EYE_WIDTH_PX = 4.0
# A Gaussian feather is negligible beyond this many sigmas.
_FEATHER_REACH = 3.5
# Fixed-point bits for sub-pixel accurate cv2 drawing.
_DRAW_SHIFT = 4

# ---- eyeshadow -------------------------------------------------------------

# Region: lash-line guard, reach past the inner corner, feather of the
# region's edges. Past the outer corner the region's own edge lies outside
# the soft bounds below (reach along a direction this far below the eye
# line), so only they shape the shadow there.
_LASH_GUARD = 0.01
_INNER_REACH = 0.08
_REGION_FEATHER = 0.012
_OUTER_REACH = 0.45
_OUTER_EDGE_DEG = -10.0
# Soft bounds past the outer corner: fade-out span along the eye (u, in eye
# widths), and a lower edge rising from the corner at this angle, with this
# half-width (x face scale).
_OUTER_FADE = (1.0, 1.5)
_OUTER_LIFT_DEG = 12.0
_OUTER_EDGE_SOFTNESS = 0.035
# Fade over the lid coordinate t: from (1 - softness) x _FADE_START to
# 1 + softness x _FADE_OVERSHOOT, and never over fewer than _MIN_FADE_PX.
_FADE_START = 0.8
_FADE_OVERSHOOT = 0.35
_MIN_FADE_PX = 2.0
# Density at t >= 1 relative to the lash line.
_TOP_DENSITY = 0.55
# Along the eye (u: 0 = inner corner, 1 = outer corner): fade-in span past the
# inner corner, density at the inner corner, span of the ramp to full density.
_INNER_FADE = (-0.2, 0.05)
_INNER_DENSITY = 0.6
_CORNER_RAMP = (0.1, 0.85)
# Share of multiply in the multiply / soft-light mix.
_SHADOW_MULTIPLY_SHARE = 0.55
# Shimmer reaches full strength at this opacity (a sheer wash already shines).
_FULL_SHIMMER_OPACITY = 0.3
# Sheen spot: center and Gaussian sigma along the eye (u), rise and fall
# spans over the lid coordinate (t).
_SHEEN_ALONG = (0.5, 0.22)
_SHEEN_RISE = (0.05, 0.35)
_SHEEN_FALL = (0.55, 1.0)
# Sheen color: white tinted by this share of the shadow color; strength at
# full shimmer; light level = this luminance percentile of the eye ROI.
_SHEEN_TINT = 0.3
_SHEEN_STRENGTH = 0.8
_LIGHT_PERCENTILE = 95.0
_LIGHT_RANGE = (0.25, 1.0)

# ---- eyeliner --------------------------------------------------------------

# Full stroke width at the outer corner: base + thickness slider x range.
_LINER_WIDTH_BASE = 0.01
_LINER_WIDTH_RANGE = 0.05
# Feather sigma: base + softness slider x range, and never below this many
# pixels (a hairline stroke on a small face stays continuous, not dashed).
_LINER_SOFT_BASE = 0.002
_LINER_SOFT_RANGE = 0.016
_LINER_MIN_FEATHER_PX = 0.4
# Samples along the lash line; width profile from the inner corner (fraction
# of full width) reaching full width at this fraction of the line.
_LINER_SAMPLES = 16
_LINER_INNER_WIDTH = 0.2
_LINER_FULL_AT = 0.85
# The stroke center is lifted off the lash line by this x its half-width.
_LINER_LIFT = 0.8
# Wing: length = base + wing slider x range; angle above the eye line; width
# at the tip relative to full width.
_WING_BASE = 0.02
_WING_RANGE = 0.2
_WING_ANGLE_DEG = 20.0
_WING_SAMPLES = 5
_WING_TIP_WIDTH = 0.12
# Lower line: starts at this fraction of the lower lash line (inner -> outer)
# and is full width from the second one; width and feather relative to the
# upper line.
_LOWER_SPAN = (1.0 / 3.0, 0.85)
_LOWER_WIDTH = 0.6
_LOWER_SOFTEN = 1.6
# Share of multiply in the darken / multiply ink.
_INK_MULTIPLY_SHARE = 0.3


@dataclass(frozen=True)
class _EyeAxes:
    """A face-aligned frame of one eye (frame pixel units)."""

    inner: np.ndarray  # inner corner (x, y)
    outer: np.ndarray  # outer corner (x, y)
    along: np.ndarray  # unit vector, inner -> outer corner
    up: np.ndarray  # unit vector perpendicular to ``along``, toward the brow
    width: float  # corner-to-corner distance

    def direction(self, angle_deg: float) -> np.ndarray:
        """Unit vector ``angle_deg`` above ``along`` (toward ``up``)."""
        angle = math.radians(angle_deg)
        return (self.along * math.cos(angle) + self.up * math.sin(angle)).astype(np.float32)

    def along_coordinate(self, roi: regions.Roi) -> np.ndarray:
        """u per ROI pixel: position along the eye, 0 at the inner and 1 at the outer corner."""
        return _project(roi, self.inner, self.along) * np.float32(1.0 / self.width)


@register_effect
class EyeshadowEffect(Effect):
    """A soft wash of color on the upper lids, deepest at the lash line."""

    name = "eyeshadow"
    order = 40

    def apply(self, frame: np.ndarray, ctx: FrameContext, settings: EyeshadowSettings) -> np.ndarray:
        opacity = _clamp01(settings.opacity)
        strength = opacity * ctx.intensity
        height = _clamp01(settings.height)
        if strength <= 0.0 or height <= 0.0 or ctx.landmarks is None:
            return frame
        landmarks = ctx.landmarks
        scale = regions.face_scale(landmarks)
        softness = _clamp01(settings.softness)
        color = hex_to_bgr(settings.color)
        sheen = _clamp01(settings.shimmer) * min(opacity / _FULL_SHIMMER_OPACITY, 1.0) * ctx.intensity
        for side in regions.SIDES:
            axes = _eye_axes(landmarks, side)
            if axes.width < _MIN_EYE_WIDTH_PX:
                continue
            outline = _lid_outline(landmarks, side, axes, scale)
            margin = _FEATHER_REACH * _REGION_FEATHER * scale + 2.0
            roi = regions.roi_around(outline, frame.shape, pad=0.0, margin_px=margin)
            if roi is None:
                continue
            maps = _shadow_maps(landmarks, side, roi, axes, outline, height, softness, scale)
            view = _tint(frame[roi.slices], color, maps.density * np.float32(strength))
            if sheen > 0.0:
                view = _add_sheen(view, color, maps.sheen_spot(), sheen)
            frame[roi.slices] = view
        return frame


@register_effect
class EyelinerEffect(Effect):
    """A tapered line along the upper lash line with an optional wing and lower line."""

    name = "eyeliner"
    order = 50

    def apply(self, frame: np.ndarray, ctx: FrameContext, settings: EyelinerSettings) -> np.ndarray:
        strength = _clamp01(settings.opacity) * ctx.intensity
        if strength <= 0.0 or ctx.landmarks is None:
            return frame
        landmarks = ctx.landmarks
        scale = regions.face_scale(landmarks)
        width_px = scale * (_LINER_WIDTH_BASE + _LINER_WIDTH_RANGE * _clamp01(settings.thickness))
        softness = _clamp01(settings.softness)
        feather_px = max(scale * (_LINER_SOFT_BASE + _LINER_SOFT_RANGE * softness), _LINER_MIN_FEATHER_PX)
        wing_px = scale * (_WING_BASE + _WING_RANGE * _clamp01(settings.wing))
        lower = _clamp01(settings.lower)
        color = hex_to_bgr(settings.color)
        for side in regions.SIDES:
            axes = _eye_axes(landmarks, side)
            if axes.width < _MIN_EYE_WIDTH_PX:
                continue
            line, profile = _upper_liner(landmarks, side, axes, width_px, wing_px)
            reach = _LOWER_SOFTEN * _FEATHER_REACH * feather_px + width_px + 2.0
            zone = np.vstack([line, landmarks[regions.side_indices(side).eye]])
            roi = regions.roi_around(zone, frame.shape, pad=0.0, margin_px=reach)
            if roi is None:
                continue
            mask = regions.polyline_mask(line, width_px, roi, feather_px, profile)
            if lower > 0.0:
                np.maximum(mask, _lower_liner_mask(landmarks, side, roi, axes, width_px, feather_px) * lower, out=mask)
            mask *= 1.0 - regions.eye_opening_mask(landmarks, side, roi)
            frame[roi.slices] = _ink(frame[roi.slices], color, mask * np.float32(strength))
        return frame


# --------------------------------------------------------------------------
# Shared geometry.
# --------------------------------------------------------------------------


def _clamp01(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


def _smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - np.float32(edge0)) * np.float32(1.0 / (edge1 - edge0)), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _eye_axes(landmarks: np.ndarray, side: regions.Side) -> _EyeAxes:
    s = regions.side_indices(side)
    inner = landmarks[s.eye_upper[0]].astype(np.float32)
    outer = landmarks[s.eye_upper[-1]].astype(np.float32)
    along = outer - inner
    width = float(np.linalg.norm(along))
    along /= max(width, 1e-6)
    up = np.array([along[1], -along[0]], np.float32)
    brow = landmarks[s.brow].astype(np.float32).mean(axis=0)
    if float(np.dot(brow - (inner + outer) / 2, up)) < 0:
        up = -up
    return _EyeAxes(inner, outer, along, up, width)


def _project(roi: regions.Roi, origin: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Signed distance (px) of each ROI pixel center from ``origin`` along a unit ``direction``."""
    xs = np.arange(roi.x0, roi.x1, dtype=np.float32)[None, :] + np.float32(0.5 - origin[0])
    ys = np.arange(roi.y0, roi.y1, dtype=np.float32)[:, None] + np.float32(0.5 - origin[1])
    return xs * np.float32(direction[0]) + ys * np.float32(direction[1])


def _normals(points: np.ndarray, toward: np.ndarray) -> np.ndarray:
    """Unit normals of a polyline at its vertices, each pointing to the side of ``toward``."""
    tangents = np.gradient(points, axis=0)
    tangents /= np.maximum(np.linalg.norm(tangents, axis=1, keepdims=True), 1e-6)
    normals = np.stack([tangents[:, 1], -tangents[:, 0]], axis=1)
    normals *= np.where(normals @ toward < 0, -1.0, 1.0)[:, None]
    return normals.astype(np.float32)


# --------------------------------------------------------------------------
# Eyeshadow.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _ShadowMaps:
    """Per-pixel eyeshadow maps over one eye's ROI."""

    region: np.ndarray  # where the shadow may go (0..1)
    u: np.ndarray  # position along the eye (0 = inner corner, 1 = outer corner)
    t: np.ndarray  # lid coordinate (0 = lash line, 1 = the chosen height)
    density: np.ndarray  # shadow strength (0..1)

    def sheen_spot(self) -> np.ndarray:
        """Where the shimmer catches the light (0..1)."""
        return self.region * _sheen_profile(self.u, self.t)


def _shadow_maps(
    landmarks: np.ndarray,
    side: regions.Side,
    roi: regions.Roi,
    axes: _EyeAxes,
    outline: np.ndarray,
    height: float,
    softness: float,
    scale: float,
) -> _ShadowMaps:
    """Region, lid coordinates and density of the shadow (see the module docstring)."""
    region = regions.polygon_mask(outline, roi, _REGION_FEATHER * scale)
    region *= 1.0 - regions.eye_opening_mask(landmarks, side, roi)
    u = axes.along_coordinate(roi)
    t = _lid_coordinate(landmarks, side, roi, height)
    min_fade = _MIN_FADE_PX / max(height * _lid_depth(landmarks, side), 1.0)
    profile = _fade_profile(t, softness, min_fade) * _along_profile(u)
    return _ShadowMaps(region, u, t, region * _outer_bound(roi, axes, scale) * profile)


def _lid_depth(landmarks: np.ndarray, side: regions.Side) -> float:
    """Mean distance from the upper lash line to the lower edge of the brow (px)."""
    s = regions.side_indices(side)
    lash = regions.resample_polyline(landmarks[s.eye_upper], 8)
    brow = regions.resample_polyline(landmarks[s.brow_lower], 8)
    return float(np.linalg.norm(brow - lash, axis=1).mean())


def _lid_outline(landmarks: np.ndarray, side: regions.Side, axes: _EyeAxes, scale: float) -> np.ndarray:
    """Closed outline of where the shadow may go (see the module docstring)."""
    s = regions.side_indices(side)
    lash = landmarks[s.eye_upper].astype(np.float32) + axes.up * np.float32(_LASH_GUARD * scale)
    brow = landmarks[s.brow_lower].astype(np.float32)
    inner = axes.inner - axes.along * np.float32(_INNER_REACH * scale)
    outer = axes.outer + axes.direction(_OUTER_EDGE_DEG) * np.float32(_OUTER_REACH * scale)
    return np.vstack([brow, outer, lash[::-1], inner])


def _outer_bound(roi: regions.Roi, axes: _EyeAxes, scale: float) -> np.ndarray:
    """Soft limit past the outer corner: a fade along the eye, above a line rising from the corner."""
    fade = 1.0 - _smoothstep(*_OUTER_FADE, axes.along_coordinate(roi))
    rise = axes.direction(_OUTER_LIFT_DEG)
    normal = axes.up - rise * np.float32(np.dot(axes.up, rise))  # perpendicular to the edge, upward
    normal /= np.linalg.norm(normal)
    half = _OUTER_EDGE_SOFTNESS * scale
    # The edge runs half its width below the corner, so the corner keeps full density.
    above = _project(roi, axes.outer - normal * np.float32(half), normal)
    return fade * _smoothstep(0.0, 2.0 * half, above)


def _distance_to(points: np.ndarray, roi: regions.Roi) -> np.ndarray:
    """Distance (px) from each ROI pixel center to an open polyline."""
    seeds = np.full((roi.h, roi.w), 255, np.uint8)
    local = (points.astype(np.float32) - np.float32(roi.offset) - np.float32(0.5)) * np.float32(1 << _DRAW_SHIFT)
    cv2.polylines(seeds, [np.round(local).astype(np.int32)], False, 0, 1, cv2.LINE_8, _DRAW_SHIFT)
    return cv2.distanceTransform(seeds, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)


def _lid_coordinate(landmarks: np.ndarray, side: regions.Side, roi: regions.Roi, height: float) -> np.ndarray:
    """t per ROI pixel: 0 on the upper lash line, 1 at ``height`` of the way to the brow."""
    s = regions.side_indices(side)
    to_lash = _distance_to(landmarks[s.eye_upper], roi)
    to_brow = _distance_to(landmarks[s.brow_lower], roi)
    return to_lash / np.maximum((to_lash + to_brow) * np.float32(height), 1e-3)


def _fade_profile(t: np.ndarray, softness: float, min_fade: float) -> np.ndarray:
    """Density over the lid coordinate: full near the lashes, fading out around t = 1."""
    start = (1.0 - softness) * _FADE_START
    end = max(1.0 + softness * _FADE_OVERSHOOT, start + min_fade)
    fade = 1.0 - _smoothstep(start, end, t)
    return fade * (1.0 - np.float32(1.0 - _TOP_DENSITY) * _smoothstep(0.0, 1.0, t))


def _along_profile(u: np.ndarray) -> np.ndarray:
    """Density along the eye: fades in at the inner corner, deepest toward the outer corner."""
    ramp = np.float32(_INNER_DENSITY) + np.float32(1.0 - _INNER_DENSITY) * _smoothstep(*_CORNER_RAMP, u)
    return _smoothstep(*_INNER_FADE, u) * ramp


def _tint(base: np.ndarray, color_bgr: Bgr, weight: np.ndarray) -> np.ndarray:
    """Multiply / soft-light mix of the shadow color, weighted per pixel.

    Soft light ``b (b + 2c (1 - b))`` and multiply ``b c`` mixed by ``k``
    differ from ``b`` by ``b (A b + B)`` with ``A = (1 - k)(1 - 2c)`` and
    ``B = (2 - k) c - 1``; computed with OpenCV arithmetic (much faster than
    NumPy on small arrays). Exactly ``base`` where the weight is 0.
    """
    k = _SHADOW_MULTIPLY_SHARE
    a = tuple((1 - k) * (1 - 2 * c) for c in color_bgr) + (0.0,)
    b = tuple((2 - k) * c - 1 for c in color_bgr) + (0.0,)
    delta = cv2.multiply(cv2.add(cv2.multiply(base, a), b), base)
    delta = cv2.multiply(delta, cv2.cvtColor(weight, cv2.COLOR_GRAY2BGR))
    return cv2.add(delta, base)


def _sheen_profile(u: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Where the lid catches the light: its center, a third of the way up."""
    center, sigma = _SHEEN_ALONG
    across = np.exp(np.float32(-0.5 / sigma**2) * np.square(u - np.float32(center)))
    return across * _smoothstep(*_SHEEN_RISE, t) * (1.0 - _smoothstep(*_SHEEN_FALL, t))


def _add_sheen(image: np.ndarray, color_bgr: Bgr, spot: np.ndarray, level: float) -> np.ndarray:
    """Screen a light tint of the shadow onto the lid, stronger where it is already lit."""
    luma = blend.luminance(image)
    light = float(np.clip(np.percentile(luma, _LIGHT_PERCENTILE), *_LIGHT_RANGE))
    sheen_color = (1.0 - _SHEEN_TINT) + _SHEEN_TINT * np.asarray(color_bgr, np.float32)
    lit = np.clip(luma * np.float32(1.0 / light), 0.0, 1.0)
    weight = spot * lit * np.float32(_SHEEN_STRENGTH * level)
    return blend.mix(image, blend.blend_screen(image, sheen_color * np.float32(light)), weight)


# --------------------------------------------------------------------------
# Eyeliner.
# --------------------------------------------------------------------------


def _upper_liner(
    landmarks: np.ndarray, side: regions.Side, axes: _EyeAxes, width_px: float, wing_px: float
) -> tuple[np.ndarray, np.ndarray]:
    """Center line and width profile of the upper liner plus its wing (inner -> wing tip)."""
    lash = regions.resample_polyline(landmarks[regions.side_indices(side).eye_upper], _LINER_SAMPLES)
    t = np.linspace(0.0, 1.0, _LINER_SAMPLES, dtype=np.float32)
    profile = _LINER_INNER_WIDTH + (1.0 - _LINER_INNER_WIDTH) * _smoothstep(0.0, _LINER_FULL_AT, t)
    lift = profile * np.float32(_LINER_LIFT * width_px / 2.0)
    line = lash + _normals(lash, axes.up) * lift[:, None]
    tip = axes.outer + axes.direction(_WING_ANGLE_DEG) * np.float32(wing_px)
    steps = np.linspace(0.0, 1.0, _WING_SAMPLES + 1, dtype=np.float32)[1:, None]
    wing = line[-1] + (tip - line[-1]) * steps
    wing_profile = 1.0 + (_WING_TIP_WIDTH - 1.0) * steps[:, 0]
    return np.vstack([line, wing]), np.concatenate([profile, wing_profile])


def _lower_liner_mask(
    landmarks: np.ndarray, side: regions.Side, roi: regions.Roi, axes: _EyeAxes, width_px: float, feather_px: float
) -> np.ndarray:
    """The softer, thinner lower line along the outer part of the lower lash line."""
    lash = regions.resample_polyline(landmarks[regions.side_indices(side).eye_lower], _LINER_SAMPLES)
    t = np.linspace(0.0, 1.0, _LINER_SAMPLES, dtype=np.float32)
    keep = t >= _LOWER_SPAN[0]
    profile = _smoothstep(*_LOWER_SPAN, t[keep])
    width = _LOWER_WIDTH * width_px
    lift = profile * np.float32(_LINER_LIFT * width / 2.0)
    line = lash[keep] + _normals(lash, -axes.up)[keep] * lift[:, None]
    return regions.polyline_mask(line, width, roi, _LOWER_SOFTEN * feather_px, profile)


def _ink(base: np.ndarray, color_bgr: Bgr, weight: np.ndarray) -> np.ndarray:
    """Liner pigment: mostly darken (never lightens dark lashes), some multiply."""
    color = np.asarray(color_bgr, np.float32)
    m = np.float32(_INK_MULTIPLY_SHARE)
    inked = np.minimum(base, color)
    inked *= 1 - m
    inked += base * (color * m)
    return blend.mix(base, inked, weight)
