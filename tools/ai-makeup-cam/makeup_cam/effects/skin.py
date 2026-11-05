"""Skin retouching: smoothing, blemish cover, redness, concealer, foundation tone, brightness.

Everything happens inside a ROI around the face oval. The analysis runs on a
downscaled working copy whose face scale is about ``_WORK_SCALE`` pixels
whatever the distance to the camera, so cost and look do not depend on face
size. Every step is expressed as a smooth per-pixel, per-channel affine model
on that grid,

    out = patch * (1 - shade) + lift,

which is upsampled and applied to the full-resolution patch in a few array
operations. Full-resolution texture and edges therefore survive, and pixels
outside the skin mask come back bit for bit.

1. Skin mask: the face oval (shrunk a little at the jaw and hairline) times
   the segmentation face-skin mask when available, minus the eyes, brows,
   mouth and nostrils, all generously feathered.
2. Smoothing: a fast guided filter (He & Sun, 2015). Per-channel self-guided
   coefficients flatten blotches and texture but keep real edges (nose
   contour, smile lines); ``texture`` keeps that share of the fine detail
   (pores) so the skin never looks plastic.
3. Blemish cover: compact spots darker in the green channel (dark and red
   spots both are) are found with a scale-normalized Hessian blob detector,
   which ignores lines (smile lines, creases), and a scale test, which ignores
   larger shading (the nose wing crease, eye corners). They are filled with
   the surrounding skin color.
4. Color (CIE Lab): redness above the face's typical a* is reduced, an
   under-eye concealer lifts and warms dark circles, a translucent
   foundation evens the chroma out toward ``tone_color``, and brightness
   lifts L by a share of the headroom. Color reaches right up to the
   features but skips pixels much darker than the skin (brow hairs, lashes).
"""

from __future__ import annotations

import functools
from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np

from ..blend import bgr_to_lab, lab_to_bgr, luminance, mix
from ..config import SkinSettings, hex_to_bgr
from ..regions import FACE_OVAL, LIPS_OUTER, SIDES, Roi, Side, expand_polygon, face_scale, roi_around, side_indices
from .base import Effect, FrameContext, register_effect

# All lengths below are fractions of the face scale (inter-ocular distance).

# Working-copy resolution: the face scale in working-copy pixels (never upscaled).
_WORK_SCALE = 40.0
# Faces smaller than this (frame pixels), or with less than this much of the
# face (in face scales) inside the frame, are left alone: too few pixels.
_MIN_FACE_SCALE = 10.0
_MIN_VISIBLE = 0.25
# Context around the face oval for the filters.
_ROI_MARGIN = 0.1

# Skin area.
_OVAL_SHRINK = 0.07  # keep off the jaw line and the hairline
_OVAL_FEATHER = 0.1
_SEGMENTATION_FEATHER = 0.05
_SEGMENTATION_RAMP = (0.25, 0.75)  # face-skin confidence -> weight
_HAIR_RAMP = (0.3, 0.8)  # hair confidence -> exclusion (used without a face-skin mask)
_MIN_SEGMENTATION_COVERAGE = 0.35  # ignore a face-skin mask that misses the face


@dataclass(frozen=True)
class _Guard:
    """How far skin edits keep away from the eyes, brows, mouth and nostrils.

    The union of the features is feathered outward, so every feature keeps
    full protection up to its (grown) outline and the fade lies in the skin.
    """

    eye_expand: float
    brow_thicken: float | None  # brow height grows by this fraction; None: brows unprotected
    mouth_expand: float
    nostril_margin: float
    feather: float


# Texture edits (smoothing, blemishes) stay well off lashes, brow hairs and the lip line.
_RETOUCH_GUARD = _Guard(eye_expand=0.06, brow_thicken=0.8, mouth_expand=0.04, nostril_margin=0.03, feather=0.05)
# Color edits (redness, foundation, brightness, concealer) reach right up to the
# features, like real foundation, so no ring of untouched skin shows around
# them. They also skip pixels much darker than the skin (brow hairs, lashes,
# nostril shadows), so the skin between brow hairs is covered too.
_COLOR_GUARD = _Guard(eye_expand=0.02, brow_thicken=None, mouth_expand=0.0, nostril_margin=0.0, feather=0.04)
_SKIN_LIGHTNESS_RAMP = (0.35, 0.7)  # L / the face's median L -> how skin-like a pixel is

# Nostril openings (subject's right; the left side is the mirrored mesh points).
_RIGHT_NOSTRIL = [75, 79, 237, 239, 238, 20, 60, 99, 240]
_LEFT_NOSTRIL = [305, 309, 457, 459, 458, 250, 290, 328, 460]

# Smoothing (fast guided filter).
_SMOOTH_RADIUS = 0.07
_SMOOTH_EPS = 0.08  # edge threshold, relative to the skin's luminance level
_PORE_SIGMA = 0.02  # detail finer than this is "texture" (pores)

# Blemishes.
_BLEMISH_SIGMA = 0.035  # detector scale (about the radius of a typical spot)
_BLEMISH_RAMP = (0.003, 0.012)  # blob response (relative to the skin level) -> spot weight
_BLEMISH_GROW = 0.03  # feather that also covers the rim of each spot
_BLEMISH_FILL = 0.08  # reach of the surrounding skin that fills a spot

# Color (Lab units: L 0..100, a/b about -128..127).
_REDNESS_MARGIN = 1.5  # a* above the face's median that still counts as normal
# Foundation is translucent and lit like the skin: at full strength the
# face's chroma evens out completely, but its average only moves this share
# of the way to the shade, so the face never parts from the neck in color.
_TONE_COVERAGE = 0.6
_TONE_LIGHTNESS = 0.15  # share of the foundation's lightness difference that is applied
_BRIGHTNESS_GAIN = 1.5  # lightness change per unit brightness, as a share of the headroom
_UNDER_EYE_DEPTH = 0.9
_UNDER_EYE_FEATHER = 0.07
_LASH_GUARD = 0.03  # keep the concealer just off the lower lash line
_LASH_FEATHER = 0.035
_UNDER_EYE_FILL = 0.7  # share of the darkness (vs. the face's median L) filled at 100%
_UNDER_EYE_LIFT = 2.0  # extra L at 100%
_UNDER_EYE_WARM_FILL = 0.6  # share of the missing b* (bluish cast) restored at 100%
_UNDER_EYE_WARM = 1.5  # extra b* (peach) at 100%
_FALLBACK_SKIN_LAB = (60.0, 15.0, 18.0)

# Fixed-point fractional bits for anti-aliased cv2 drawing.
_DRAW_SHIFT = 4


@dataclass(frozen=True)
class _Amounts:
    """Effective strengths for this frame (sliders x master intensity)."""

    smoothing: float
    texture: float
    blemish: float
    redness: float
    under_eye: float
    tone: float
    brightness: float

    @classmethod
    def from_settings(cls, settings: SkinSettings, intensity: float) -> _Amounts:
        k = min(max(float(intensity), 0.0), 1.0)
        return cls(
            smoothing=_unit(settings.smoothing) * k,
            texture=_unit(settings.texture),
            blemish=_unit(settings.blemish) * k,
            redness=_unit(settings.redness) * k,
            under_eye=_unit(settings.under_eye) * k,
            tone=_unit(settings.tone_strength) * k,
            brightness=float(settings.brightness) * k,
        )

    @property
    def active(self) -> bool:
        return self.smoothing > 0 or self.blemish > 0 or self.color_active

    @property
    def color_active(self) -> bool:
        return self.redness > 0 or self.under_eye > 0 or self.tone > 0 or self.brightness != 0


@dataclass(frozen=True)
class _Workspace:
    """The face ROI (frame pixels) and its downscaled working copy."""

    roi: Roi
    small_size: tuple[int, int]  # (width, height) of the working copy
    landmarks: np.ndarray  # (478, 2) in working-copy pixel coordinates
    scale: float  # face scale in working-copy pixels
    full_scale: float  # face scale in frame pixels

    @property
    def full_size(self) -> tuple[int, int]:
        return self.roi.w, self.roi.h


@dataclass(frozen=True)
class _SkinMasks:
    """Working-copy masks."""

    area: np.ndarray  # face oval x segmentation (eyes, brows, mouth included)
    skin: np.ndarray  # area minus the features (``_RETOUCH_GUARD``): smoothing, blemishes
    color: np.ndarray  # area minus the features (``_COLOR_GUARD``): color corrections


class _Model:
    """Per-pixel retouch ``out = patch * (1 - shade) + lift`` on the working-copy grid.

    Starts as the identity (shade and lift zero). Storing ``shade`` rather
    than ``1 - shade`` keeps untouched pixels exactly zero, so they stay
    exact through the upsampling.
    """

    def __init__(self, height: int, width: int) -> None:
        self.shade = np.zeros((height, width, 3), np.float32)
        self.lift = np.zeros((height, width, 3), np.float32)

    def mix_toward(self, weight: np.ndarray, offset: np.ndarray, gain: np.ndarray | None = None) -> None:
        """Blend the current result toward ``gain * patch + offset`` (gain None: 0) by ``weight``."""
        self.shade = mix(self.shade, 1.0 if gain is None else 1.0 - gain, weight)
        self.lift = mix(self.lift, offset, weight)

    def render_into(self, target: np.ndarray, patch: np.ndarray) -> None:
        """Write the model applied to the full-resolution ``patch`` into ``target`` (same shape)."""
        size = (patch.shape[1], patch.shape[0])
        out = _resize(self.shade, size) * patch
        np.subtract(patch, out, out=out)
        out += _resize(self.lift, size)
        np.clip(out, 0.0, 1.0, out=target)


@register_effect
class SkinEffect(Effect):
    """Natural skin retouching (see the module docstring)."""

    name = "skin"
    order = 20

    def apply(self, frame: np.ndarray, ctx: FrameContext, settings: SkinSettings) -> np.ndarray:
        amounts = _Amounts.from_settings(settings, ctx.intensity)
        if not amounts.active or ctx.landmarks is None:
            return frame
        ws = _workspace(ctx.landmarks, frame.shape)
        if ws is None:
            return frame
        patch = frame[ws.roi.slices]
        small = _resize(patch, ws.small_size, cv2.INTER_AREA)
        masks = _skin_masks(ws, ctx)
        model = _retouch_model(small, masks, ws, amounts, settings.tone_color)
        model.render_into(patch, patch)
        return frame


def _retouch_model(
    small: np.ndarray, masks: _SkinMasks, ws: _Workspace, amounts: _Amounts, tone_color: str
) -> _Model:
    model = _Model(*small.shape[:2])
    level = _skin_level(small, masks.skin)
    if amounts.smoothing > 0:
        gain, offset = _smoothing_terms(small, ws.scale, level, amounts.texture)
        model.mix_toward(masks.skin * np.float32(amounts.smoothing), offset, gain)
    if amounts.blemish > 0:
        spots, clean = _blemishes(small, ws.scale, level)
        model.mix_toward(spots * masks.skin * np.float32(amounts.blemish), clean)
    if amounts.color_active:
        model.lift += _color_delta(small, masks, ws, amounts, tone_color)
    return model


# --------------------------------------------------------------------------
# Geometry and masks (working-copy coordinates).
# --------------------------------------------------------------------------


def _workspace(landmarks: np.ndarray, frame_shape: tuple[int, ...]) -> _Workspace | None:
    scale = face_scale(landmarks)
    if not np.isfinite(scale) or scale < _MIN_FACE_SCALE:
        return None
    roi = roi_around(landmarks[FACE_OVAL], frame_shape, pad=0.0, margin_px=_ROI_MARGIN * scale)
    if roi is None or min(roi.w, roi.h) < _MIN_VISIBLE * scale:
        return None
    factor = min(1.0, _WORK_SCALE / scale)
    width, height = max(1, round(roi.w * factor)), max(1, round(roi.h * factor))
    ratio = np.array([width / roi.w, height / roi.h], np.float32)
    local = (np.asarray(landmarks, np.float32) - np.array(roi.offset, np.float32)) * ratio
    return _Workspace(roi, (width, height), local, scale * float(ratio.mean()), scale)


def _skin_masks(ws: _Workspace, ctx: FrameContext) -> _SkinMasks:
    lm, s = ws.landmarks, ws.scale
    oval = expand_polygon(lm[FACE_OVAL], -_OVAL_SHRINK * s)
    area = _shapes_mask(ws.small_size, [oval], sigma=_OVAL_FEATHER * s)
    segmentation = _segmentation_weight(ctx, ws, area)
    if segmentation is not None:
        area *= segmentation
    return _SkinMasks(
        area=area,
        skin=area * (1.0 - _features_mask(ws, _RETOUCH_GUARD)),
        color=area * (1.0 - _features_mask(ws, _COLOR_GUARD)),
    )


def _features_mask(ws: _Workspace, guard: _Guard) -> np.ndarray:
    """Eyes, brows, mouth and nostrils grown by ``guard`` (1 inside, fading outward)."""
    lm, s = ws.landmarks, ws.scale
    polygons = [expand_polygon(lm[side_indices(side).eye], guard.eye_expand * s) for side in SIDES]
    if guard.brow_thicken is not None:
        polygons += [_brow_outline(lm, side, guard.brow_thicken) for side in SIDES]
    polygons.append(expand_polygon(lm[LIPS_OUTER], guard.mouth_expand * s))
    ellipses = [_nostril_ellipse(lm, side, guard.nostril_margin * s) for side in SIDES]
    return _solid_inside(_shapes_mask(ws.small_size, polygons, ellipses, sigma=guard.feather * s))


def _brow_outline(landmarks: np.ndarray, side: Side, thicken: float) -> np.ndarray:
    """One brow, ``thicken`` taller around its center line (as ``regions.brow_mask``)."""
    s = side_indices(side)
    upper, lower = landmarks[s.brow_upper], landmarks[s.brow_lower]
    center = (upper + lower) / 2
    half = (upper - lower) / 2 * (1.0 + thicken)
    return np.vstack([center + half, (center - half)[::-1]])


def _nostril_ellipse(landmarks: np.ndarray, side: Side, margin_px: float) -> tuple[np.ndarray, np.ndarray, float]:
    """``(center, half_axes, angle_deg)`` around one nostril (its mesh points form a thin crescent)."""
    points = landmarks[_LEFT_NOSTRIL if side == "left" else _RIGHT_NOSTRIL].astype(np.float32)
    center, size, angle = cv2.minAreaRect(points)
    return np.array(center), np.array(size) / 2 + margin_px, angle


def _under_eye_zone(ws: _Workspace, area: np.ndarray) -> np.ndarray:
    """Both under-eye concealer zones (as ``regions.under_eye_mask``), kept off the lash line."""
    lm, s = ws.landmarks, ws.scale
    zones, eyes = [], []
    for side in SIDES:
        indices = side_indices(side)
        top = lm[indices.under_eye_top]
        bottom = top + _UNDER_EYE_DEPTH * (lm[indices.under_eye_bottom] - top)
        zones.append(np.vstack([top, bottom[::-1]]))
        eyes.append(expand_polygon(lm[indices.eye], _LASH_GUARD * s))
    zone = _shapes_mask(ws.small_size, zones, sigma=_UNDER_EYE_FEATHER * s)
    lashes = _solid_inside(_shapes_mask(ws.small_size, eyes, sigma=_LASH_FEATHER * s))
    return zone * (1.0 - lashes) * area


def _segmentation_weight(ctx: FrameContext, ws: _Workspace, oval: np.ndarray) -> np.ndarray | None:
    """Soft weight from the segmentation masks (keeps hair and background out), or None."""
    if ctx.face_skin_mask is not None:
        weight = _ramp(_crop_small(ctx.face_skin_mask, ws), *_SEGMENTATION_RAMP)
        coverage = float(np.sum(weight * oval)) / max(float(np.sum(oval)), 1e-6)
        if coverage < _MIN_SEGMENTATION_COVERAGE:
            return None  # the model missed the face (e.g. very dim light): trust the landmarks
    elif ctx.hair_mask is not None:
        weight = 1.0 - _ramp(_crop_small(ctx.hair_mask, ws), *_HAIR_RAMP)
    else:
        return None
    return _blur(weight, _SEGMENTATION_FEATHER * ws.scale)


def _crop_small(mask: np.ndarray, ws: _Workspace) -> np.ndarray:
    # The masks are smooth (upsampled from the model's low resolution): bilinear is enough.
    return _resize(np.asarray(mask[ws.roi.slices], dtype=np.float32), ws.small_size)


def _shapes_mask(
    size: tuple[int, int],
    polygons: Sequence[np.ndarray],
    ellipses: Sequence[tuple[np.ndarray, np.ndarray, float]] = (),
    sigma: float = 0.0,
) -> np.ndarray:
    """Union of anti-aliased shapes on one (height, width) float32 mask, blurred by ``sigma``.

    Coordinates are continuous (pixel ``i`` covers ``[i, i + 1)``, like
    ``regions``). Drawing every shape on one canvas with a single blur is
    several times cheaper than one ``regions`` mask per shape, and the
    supersampling there would be lost in these wide feathers anyway.
    """
    canvas = np.zeros((size[1], size[0]), np.uint8)
    for polygon in polygons:
        cv2.fillPoly(canvas, [_fixed_point(polygon)], 255, cv2.LINE_AA, _DRAW_SHIFT)
    for center, half_axes, angle in ellipses:
        cx, cy = _fixed_point(center).ravel()
        axes = tuple(int(v) for v in np.round(half_axes * (1 << _DRAW_SHIFT)))
        cv2.ellipse(canvas, (int(cx), int(cy)), axes, angle, 0, 360, 255, -1, cv2.LINE_AA, _DRAW_SHIFT)
    return _blur(canvas.astype(np.float32) * np.float32(1.0 / 255.0), sigma)


def _solid_inside(feathered: np.ndarray) -> np.ndarray:
    """A feathered shape is 0.5 on its outline; doubling makes it solid (exactly 1)
    inside and moves the whole fade outside the outline."""
    return np.minimum(feathered * np.float32(2.0), 1.0)


def _fixed_point(points: np.ndarray) -> np.ndarray:
    """Continuous coordinates -> int32 fixed-point pixel-center coordinates for cv2 drawing."""
    return np.round((np.asarray(points, np.float64) - 0.5) * (1 << _DRAW_SHIFT)).astype(np.int32)


# --------------------------------------------------------------------------
# Retouching terms (working copy).
# --------------------------------------------------------------------------


def _skin_level(small: np.ndarray, skin: np.ndarray) -> float:
    """Typical skin luminance (for thresholds that should follow the exposure)."""
    total = float(np.sum(skin))
    if total <= 1e-6:
        return 0.5
    return max(float(np.sum(luminance(small) * skin)) / total, 0.05)


def _smoothing_terms(small: np.ndarray, scale: float, level: float, texture: float) -> tuple[np.ndarray, np.ndarray]:
    """``(gain, offset)`` of the smoothed skin ``gain * patch + offset``.

    The guided filter gives ``gain * patch + offset``. ``texture`` adds back
    that share of the fine detail ``patch - blurred`` wherever the filter
    flattened the skin (``1 - gain``); the blurred image lives on the working
    grid, so the detail includes everything finer than it at full resolution.
    """
    radius = max(1, round(_SMOOTH_RADIUS * scale))
    gain, offset = _guided_coefficients(small, radius, (_SMOOTH_EPS * level) ** 2)
    if texture <= 0:
        return gain, offset
    kept = (1.0 - gain) * np.float32(texture)
    gain += kept
    offset -= kept * _blur(small, _PORE_SIGMA * scale)
    return gain, offset


def _guided_coefficients(image: np.ndarray, radius: int, eps: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel self-guided filter: ``filtered = gain * image + offset``."""
    ksize = (2 * radius + 1, 2 * radius + 1)
    mean = cv2.boxFilter(image, -1, ksize, borderType=cv2.BORDER_REFLECT)
    variance = cv2.boxFilter(image * image, -1, ksize, borderType=cv2.BORDER_REFLECT) - mean * mean
    gain = variance / (variance + np.float32(eps))
    offset = mean - gain * mean
    gain = cv2.boxFilter(gain, -1, ksize, borderType=cv2.BORDER_REFLECT)
    offset = cv2.boxFilter(offset, -1, ksize, borderType=cv2.BORDER_REFLECT)
    return gain, offset


def _blemishes(small: np.ndarray, scale: float, level: float) -> tuple[np.ndarray, np.ndarray]:
    """Spot weights (dark or red specks) and the surrounding skin color to fill them with."""
    sigma = _BLEMISH_SIGMA * scale
    green = cv2.extractChannel(small, 1)
    # Spots answer more at their own scale than at twice it; larger dark shading does not.
    response = _dark_blob_response(green, sigma) - np.maximum(_dark_blob_response(green, 2.0 * sigma), 0.0)
    spots = _ramp(response * np.float32(1.0 / level), *_BLEMISH_RAMP)
    spots = np.minimum(_blur(spots, _BLEMISH_GROW * scale) * np.float32(2.0), 1.0)
    # Average of the nearby skin, leaving the spots themselves out (normalized convolution).
    keep = 1.0 - np.float32(0.95) * spots
    fill_sigma = _BLEMISH_FILL * scale
    clean = _blur(small * keep[..., None], fill_sigma) / _blur(keep, fill_sigma)[..., None]
    return spots, clean


def _dark_blob_response(image: np.ndarray, sigma: float) -> np.ndarray:
    """Scale-normalized smaller Hessian eigenvalue at ``sigma``: > 0 at dark blobs, ~0 along lines and edges."""
    image = cv2.GaussianBlur(image, (0, 0), sigma)
    # 3x3 Sobel second derivatives are 4x the true ones.
    dxx = cv2.Sobel(image, cv2.CV_32F, 2, 0, ksize=3)
    dyy = cv2.Sobel(image, cv2.CV_32F, 0, 2, ksize=3)
    dxy = cv2.Sobel(image, cv2.CV_32F, 1, 1, ksize=3)
    half_trace = (dxx + dyy) * np.float32(0.5)
    half_diff = (dxx - dyy) * np.float32(0.5)
    smaller = half_trace - np.sqrt(half_diff * half_diff + dxy * dxy)
    return smaller * np.float32(sigma * sigma / 4.0)


def _color_delta(
    small: np.ndarray, masks: _SkinMasks, ws: _Workspace, amounts: _Amounts, tone_color: str
) -> np.ndarray:
    """BGR offset for redness, concealer, foundation tone and brightness (0 outside the masks)."""
    lab = bgr_to_lab(small)
    reference = _skin_reference(lab, masks.skin)
    skin_like = _ramp(lab[..., 0] * np.float32(1.0 / max(float(reference[0]), 1.0)), *_SKIN_LIGHTNESS_RAMP)
    color = masks.color * skin_like
    edited = lab.copy()
    if amounts.redness > 0:
        _reduce_redness(edited, color * np.float32(amounts.redness), reference)
    if amounts.under_eye > 0:
        zone = _under_eye_zone(ws, masks.area) * skin_like
        _conceal_under_eyes(edited, zone * np.float32(amounts.under_eye), reference)
    if amounts.tone > 0:
        _apply_foundation(edited, color * np.float32(amounts.tone), reference, _tone_lab(tone_color))
    if amounts.brightness != 0:
        _brighten(edited, color, amounts.brightness)
    return lab_to_bgr(edited) - lab_to_bgr(lab)


def _reduce_redness(lab: np.ndarray, weight: np.ndarray, reference: np.ndarray) -> None:
    a = lab[..., 1]
    excess = np.maximum(a - np.float32(reference[1] + _REDNESS_MARGIN), 0.0)
    a -= excess * weight


def _conceal_under_eyes(lab: np.ndarray, weight: np.ndarray, reference: np.ndarray) -> None:
    """Fill the darkness toward the face's typical lightness and cancel the bluish cast."""
    lightness, b = lab[..., 0], lab[..., 2]
    darkness = np.maximum(np.float32(reference[0]) - lightness, 0.0)
    lightness += weight * (darkness * np.float32(_UNDER_EYE_FILL) + np.float32(_UNDER_EYE_LIFT))
    blue_cast = np.maximum(np.float32(reference[2]) - b, 0.0)
    b += weight * (blue_cast * np.float32(_UNDER_EYE_WARM_FILL) + np.float32(_UNDER_EYE_WARM))


def _apply_foundation(lab: np.ndarray, weight: np.ndarray, reference: np.ndarray, tone: np.ndarray) -> None:
    """Even the chroma out toward the foundation shade; lightness only shifts slightly (shading stays)."""
    target = reference + (tone - reference) * np.float32(_TONE_COVERAGE)
    lab[..., 1] += (target[1] - lab[..., 1]) * weight
    lab[..., 2] += (target[2] - lab[..., 2]) * weight
    lab[..., 0] += weight * np.float32(_TONE_LIGHTNESS * (tone[0] - reference[0]))


def _brighten(lab: np.ndarray, skin: np.ndarray, brightness: float) -> None:
    """Lift (or lower) L by a share of the headroom, so highlights never clip."""
    lightness = lab[..., 0]
    headroom = (100.0 - lightness) if brightness > 0 else lightness
    lightness += skin * headroom * np.float32(brightness * _BRIGHTNESS_GAIN)


def _skin_reference(lab: np.ndarray, skin: np.ndarray) -> np.ndarray:
    """Median Lab of clearly-skin pixels (the face's typical color), sampled on every other pixel."""
    sampled_lab, sampled_skin = lab[::2, ::2], skin[::2, ::2]
    for threshold in (0.5, 0.0):
        core = sampled_skin > threshold
        if np.any(core):
            return np.median(sampled_lab[core], axis=0).astype(np.float32)
    return np.array(_FALLBACK_SKIN_LAB, np.float32)


@functools.lru_cache(maxsize=16)
def _tone_lab(tone_color: str) -> np.ndarray:
    lab = bgr_to_lab(np.array(hex_to_bgr(tone_color), np.float32).reshape(1, 1, 3))[0, 0]
    lab.setflags(write=False)
    return lab


# --------------------------------------------------------------------------
# Small helpers.
# --------------------------------------------------------------------------


def _unit(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


def _resize(image: np.ndarray, size: tuple[int, int], interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
    """Resize to ``size`` = (width, height); returns the input when the size already matches."""
    if (image.shape[1], image.shape[0]) == size:
        return image
    return cv2.resize(image, size, interpolation=interpolation)


def _blur(image: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return image
    return cv2.GaussianBlur(image, (0, 0), sigma, borderType=cv2.BORDER_REPLICATE)


def _ramp(values: np.ndarray, low: float, high: float) -> np.ndarray:
    """Smoothstep from 0 at ``low`` to 1 at ``high`` (float32)."""
    t = np.clip((values - np.float32(low)) * np.float32(1.0 / (high - low)), 0.0, 1.0).astype(np.float32, copy=False)
    return t * t * (3.0 - 2.0 * t)
