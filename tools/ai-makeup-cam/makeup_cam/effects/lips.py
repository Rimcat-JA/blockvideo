"""Lipstick: recolors the lips and sets their finish (matte / satin / gloss).

Color: the Lab "color" blend gives the lips the lipstick's hue and chroma
while keeping their own shading and texture, so they still look like lips.
The lips' average lightness also moves most of the way toward the lipstick's
lightness, so a pale nude lightens and a deep berry darkens instead of every
shade taking the natural lip lightness. The chosen color means "as seen in
neutral light": it is shifted by the scene's color cast, measured on the skin
around the mouth, so a nude stays a nude under a warm lamp instead of turning
lavender.

Finish works on the lips' own highlights (their brightest areas, found per
frame). Matte is part of the pigment: an opaque matte lipstick flattens the
highlights, a sheer one barely does. Satin and gloss add shine on top: satin
brightens the highlights a little, gloss brightens and spreads them and adds
a soft reflection on the center of the lower lip. Shine reaches full strength
already at a sheer opacity (a clear gloss shines as much as a tinted one) and
is screened with the scene's light level, so a dim room never gets
studio-white highlights.

The mask (``regions.lips_mask``) feathers only the outer lip line; the edge
toward the teeth stays crisp, so color never touches the teeth. Close-up
faces (face scale above ``_MAX_WORK_SCALE``) are made up on a downscaled copy
of the lip ROI and the upsampled change is added back, which keeps the cost
bounded; the full-resolution mouth opening still masks that change.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .. import blend, regions
from ..config import LipSettings, hex_to_bgr
from .base import Effect, FrameContext, register_effect

Bgr = tuple[float, float, float]

# All lengths below are fractions of the face scale (inter-ocular distance)
# unless noted otherwise.

# Outer lip line feather: base + feather slider x range.
_FEATHER_BASE = 0.006
_FEATHER_RANGE = 0.035
# ROI padding as a fraction of the mouth width: covers overlining and leaves
# a ring of skin around the lips for the color-cast estimate.
_ROI_PAD = 0.08
# Larger faces (face scale in pixels) are made up at a reduced resolution.
_MAX_WORK_SCALE = 90.0

# Fraction of the gap between the lips' average lightness and the lipstick's
# lightness that an opaque lipstick closes (1 = the lips take its lightness).
_LIGHTNESS_FOLLOW = 0.7

# Scene color cast: the skin around the mouth (outside the lips grown by
# _SKIN_GAP) is compared with a typical skin chroma in neutral light; that
# fraction of the difference, limited in size, shifts the lipstick color.
_SKIN_GAP = 0.03
_REFERENCE_SKIN_AB = np.array([11.0, 18.0], np.float32)
_CAST_ADAPTATION = 0.75
_MAX_CAST = 20.0
_CAST_STEP = 0.5  # quantization (Lab units): steady colors, cached gamut tables
_MIN_SKIN_SAMPLES = 12

# "Core": lip pixels well inside the lip line, where highlights are measured
# (keeps the skin at the border and the teeth out of the statistics).
_CORE_INSET = 0.02
_MIN_CORE_SAMPLES = 8
# Highlights start at the first percentile of the core's smoothed luminance
# and reach full strength at the second one.
_HIGHLIGHT_PERCENTILES = (80.0, 99.0)
_HIGHLIGHT_BLUR = 0.015
# Light level used for added shine: this percentile of the ROI's luminance.
_LIGHT_PERCENTILE = 99.0
_LIGHT_RANGE = (0.3, 1.0)
# Statistics (medians, percentiles) use every n-th pixel (per axis).
_STATS_STEP = 2

# Shine reaches full strength at this opacity (a sheer layer already shines).
_FULL_SHINE_OPACITY = 0.3

# Lower-lip reflection (gloss), relative to the lower lip: center position
# from its top edge, semi-axes (x mouth width, x lower-lip thickness), feather.
_REFLECTION_POSITION = 0.4
_REFLECTION_AXES = (0.16, 0.25)
_REFLECTION_FEATHER = 0.2


@dataclass(frozen=True)
class _Finish:
    """How a finish treats the lips' highlights."""

    flatten: float = 0.0  # matte: fraction of the lightness above the median removed
    shine: float = 0.0  # satin / gloss: highlight brightening (x gloss slider)
    spread: float = 0.0  # extra blur of the shine (wet look), x face scale
    reflection: float = 0.0  # lower-lip reflection strength (x gloss slider)


_FINISHES: dict[str, _Finish] = {
    "matte": _Finish(flatten=0.6),
    "satin": _Finish(shine=0.35),
    "gloss": _Finish(shine=0.8, spread=0.02, reflection=0.6),
}


@register_effect
class LipEffect(Effect):
    """Lipstick color plus matte / satin / gloss finish."""

    name = "lips"
    order = 70

    def apply(self, frame: np.ndarray, ctx: FrameContext, settings: LipSettings) -> np.ndarray:
        if _clamp01(settings.opacity) * ctx.intensity <= 0.0 or ctx.landmarks is None:
            return frame
        finish = _finish(settings.finish)
        landmarks = ctx.landmarks
        scale = regions.face_scale(landmarks)
        feather_px = scale * _feather(settings)
        roi = regions.roi_around(landmarks[regions.LIPS_OUTER], frame.shape, _ROI_PAD, 4.0 * feather_px + 2.0)
        if roi is None:
            return frame
        view = frame[roi.slices]
        if scale <= _MAX_WORK_SCALE:
            _make_up(view, landmarks, roi, scale, finish, settings, ctx.intensity)
        else:
            _make_up_downscaled(view, landmarks, roi, scale, finish, settings, ctx.intensity)
        return frame


def _make_up(
    image: np.ndarray,
    landmarks: np.ndarray,
    roi: regions.Roi,
    scale: float,
    finish: _Finish,
    settings: LipSettings,
    intensity: float,
) -> None:
    """Apply the lipstick in place to ``image``, the pixels of ``roi`` (landmarks
    and ``scale`` in the same pixel coordinates)."""
    opacity = _clamp01(settings.opacity)
    paint = regions.lips_mask(landmarks, roi, outline=settings.outline, feather_px=scale * _feather(settings))
    core = _lip_core(landmarks, roi, scale)
    # Shine level: gloss slider x product coverage x master intensity.
    level = _clamp01(settings.gloss) * min(opacity / _FULL_SHINE_OPACITY, 1.0) * intensity
    highlights = _highlights(blend.luminance(image), core, scale) if finish.shine * level > 0.0 else None
    lab = blend.bgr_to_lab(image)
    color = _scene_color(hex_to_bgr(settings.color), lab, _skin_around_lips(landmarks, roi, scale))
    _apply_lightness(lab[..., 0], color, core, finish.flatten)
    pigment = blend.blend_color(blend.lab_to_bgr(lab), color)
    out = blend.mix(image, pigment, paint * np.float32(opacity * intensity))
    if highlights is not None:
        out = _add_shine(out, highlights, core, finish, level, landmarks, roi, scale)
    image[...] = out


def _make_up_downscaled(
    view: np.ndarray,
    landmarks: np.ndarray,
    roi: regions.Roi,
    scale: float,
    finish: _Finish,
    settings: LipSettings,
    intensity: float,
) -> None:
    """Make up a copy of ``view`` downscaled to a face scale of about
    ``_MAX_WORK_SCALE``, then add the upsampled change to ``view``."""
    size = (max(1, round(roi.w * _MAX_WORK_SCALE / scale)), max(1, round(roi.h * _MAX_WORK_SCALE / scale)))
    ratio = np.array([size[0] / roi.w, size[1] / roi.h], np.float32)
    small = cv2.resize(view, size, interpolation=cv2.INTER_AREA)
    made_up = small.copy()
    local = (landmarks - np.asarray(roi.offset, np.float32)) * ratio
    _make_up(made_up, local, regions.Roi(0, 0, *size), scale * float(ratio[0]), finish, settings, intensity)
    change = cv2.resize(made_up - small, (roi.w, roi.h), interpolation=cv2.INTER_LINEAR)
    mouth_opening = regions.polygon_mask(landmarks[regions.LIPS_INNER], roi)
    view += change * (1.0 - mouth_opening)[..., None]
    np.clip(view, 0.0, 1.0, out=view)  # the upsampled change can overshoot on fine texture


def _clamp01(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


def _feather(settings: LipSettings) -> float:
    """Outer lip line feather sigma, x face scale."""
    return _FEATHER_BASE + _FEATHER_RANGE * _clamp01(settings.feather)


def _finish(name: str) -> _Finish:
    try:
        return _FINISHES[name]
    except KeyError:
        raise ValueError(f"unknown lip finish {name!r}; expected one of {sorted(_FINISHES)}") from None


def _lip_core(landmarks: np.ndarray, roi: regions.Roi, scale: float) -> np.ndarray:
    """Lip interior weight: 0 at the lip lines (outer and toward the teeth),
    rising smoothly to 1 about ``2 * _CORE_INSET`` inside (a soft erosion)."""
    hard = regions.lips_mask(landmarks, roi)
    blurred = regions.feather(hard, _CORE_INSET * scale)
    return np.clip(blurred * 2.0 - 1.0, 0.0, 1.0) * hard


def _skin_around_lips(landmarks: np.ndarray, roi: regions.Roi, scale: float) -> np.ndarray:
    """Weight of the skin around the mouth inside the ROI (lips and mouth excluded)."""
    grown = regions.expand_polygon(landmarks[regions.LIPS_OUTER], _SKIN_GAP * scale)
    return 1.0 - regions.polygon_mask(grown, roi)


def _samples(values: np.ndarray, weight: np.ndarray | None = None) -> np.ndarray:
    """Every ``_STATS_STEP``-th pixel per axis of ``values`` (where ``weight`` > 0.5)."""
    sub = values[::_STATS_STEP, ::_STATS_STEP]
    return sub if weight is None else sub[weight[::_STATS_STEP, ::_STATS_STEP] > 0.5]


def _color_lab(color_bgr: Bgr) -> np.ndarray:
    return blend.bgr_to_lab(np.asarray(color_bgr, np.float32).reshape(1, 1, 3))[0, 0]


def _scene_color(color_bgr: Bgr, lab: np.ndarray, skin: np.ndarray) -> Bgr:
    """The lipstick color as it looks in the scene's light (see module docstring)."""
    pixels = _samples(lab, skin)
    if len(pixels) < _MIN_SKIN_SAMPLES:
        return color_bgr
    lit = pixels[pixels[:, 0] >= np.median(pixels[:, 0])]  # skip shadows, stubble, nostrils
    cast = (np.median(lit[:, 1:], axis=0) - _REFERENCE_SKIN_AB) * _CAST_ADAPTATION
    length = float(np.linalg.norm(cast))
    if length > _MAX_CAST:
        cast *= _MAX_CAST / length
    cast = np.round(cast / _CAST_STEP) * _CAST_STEP
    if not cast.any():
        return color_bgr
    shifted = _color_lab(color_bgr)
    shifted[1:] += cast
    b, g, r = blend.lab_to_bgr(shifted.reshape(1, 1, 3))[0, 0]
    return float(b), float(g), float(r)


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float | None:
    total = float(weights.sum())
    return float((values * weights).sum() / total) if total > 1e-6 else None


def _apply_lightness(lightness: np.ndarray, color_bgr: Bgr, core: np.ndarray, flatten: float) -> None:
    """In place on Lab L: shift toward the lipstick's lightness; matte (``flatten``
    > 0) compresses everything brighter than the lips' median, pixel by pixel."""
    lip_lightness = _weighted_mean(lightness, core)
    if lip_lightness is None:
        return
    if flatten > 0:
        inner = _samples(lightness, core)
        median = float(np.median(inner)) if inner.size else lip_lightness
        lightness -= core * np.float32(flatten) * np.maximum(lightness - median, 0.0)
    lightness += np.float32(_LIGHTNESS_FOLLOW * (float(_color_lab(color_bgr)[0]) - lip_lightness))
    np.clip(lightness, 0.0, 100.0, out=lightness)


def _highlights(luma: np.ndarray, core: np.ndarray, scale: float) -> np.ndarray | None:
    """0..1 map of the lips' brightest areas (None if the lips are too small).

    Measured on the lips' luminance smoothed over the lips only (normalized
    convolution), so highlights follow the lips' shape rather than their
    texture, and bright teeth or skin never leak in.
    """
    if len(_samples(core, core)) < _MIN_CORE_SAMPLES:
        return None
    sigma = _HIGHLIGHT_BLUR * scale
    smooth = regions.feather(luma * core, sigma) / np.maximum(regions.feather(core, sigma), 1e-3)
    start, full = np.percentile(_samples(smooth, core), _HIGHLIGHT_PERCENTILES)
    ramp = np.clip((smooth - start) / max(float(full - start), 1e-4), 0.0, 1.0)
    return ramp * ramp * (3.0 - 2.0 * ramp) * core  # smoothstep: no visible threshold


def _add_shine(
    image: np.ndarray,
    highlights: np.ndarray,
    core: np.ndarray,
    finish: _Finish,
    level: float,
    landmarks: np.ndarray,
    roi: regions.Roi,
    scale: float,
) -> np.ndarray:
    """Screen the scene's light level onto the highlights (and the gloss reflection)."""
    shine = highlights
    if finish.spread > 0:
        shine = np.maximum(shine, regions.feather(shine, finish.spread * scale) * core)
    shine = shine * np.float32(finish.shine * level)
    if finish.reflection > 0:
        reflection = _lower_lip_reflection(landmarks, roi) * core
        shine = np.maximum(shine, reflection * np.float32(finish.reflection * level))
    light = float(np.clip(np.percentile(blend.luminance(_samples(image)), _LIGHT_PERCENTILE), *_LIGHT_RANGE))
    return blend.mix(image, blend.blend_screen(image, (light, light, light)), np.minimum(shine, 1.0))


def _lower_lip_reflection(landmarks: np.ndarray, roi: regions.Roi) -> np.ndarray:
    """Soft oval on the front of the lower lip, where gloss catches the light."""
    top = landmarks[regions.LIPS_INNER_BOTTOM].astype(np.float32)
    bottom = landmarks[regions.LIPS_BOTTOM].astype(np.float32)
    left = landmarks[regions.MOUTH_LEFT_CORNER].astype(np.float32)
    right = landmarks[regions.MOUTH_RIGHT_CORNER].astype(np.float32)
    thickness = float(np.linalg.norm(bottom - top))
    width = float(np.linalg.norm(left - right))
    center = top + _REFLECTION_POSITION * (bottom - top)
    angle = math.degrees(math.atan2(float(left[1] - right[1]), float(left[0] - right[0])))
    axes = (_REFLECTION_AXES[0] * width, _REFLECTION_AXES[1] * thickness)
    return regions.ellipse_mask(center, axes, angle, roi, feather_px=_REFLECTION_FEATHER * thickness)
