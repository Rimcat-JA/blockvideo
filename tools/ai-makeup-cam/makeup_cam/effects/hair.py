"""Hair: sleekness (まとまり), gloss (ツヤ), color, brightness and contrast.

Region
------
The segmenter's hair confidence (low resolution, upsampled and temporally
smoothed) is turned into a soft weight:

* Only hair attached to the head is used. The mask is split into blobs on a
  coarse grid (cheap); blobs away from the face are dropped as background
  false positives (without a face, the largest blobs are kept). Everything
  happens inside the bounding box of what is kept.
* Face skin never gets hair effects: the face-skin confidence is subtracted,
  and the inner face from the landmarks (eyes, brows, nose, mouth, inner
  cheeks) is protected, which also holds while the mask lags behind a
  moving head.
* Low confidences are cut with a smooth threshold, then the effects fade
  in from the edge over a width set by ``edge_softness`` (inside the hair
  only, never spilling onto the background).

Three grids keep the cost independent of the distance to the camera: the
analysis (masks, statistics, strand directions, gloss zone) runs on a
working copy with a face scale of about ``_WORK_SCALE`` pixels; the
per-pixel look is computed on a render copy with a face scale of about
``_RENDER_SCALE`` pixels (the ROI itself for small faces); and the frame only
receives two maps, ``out = in * scale + offset``, so its own texture and
edges survive.

Effects
-------
* Smoothness: frizz is luminance texture across the strands. The strand
  direction is estimated with a structure tensor and the luminance is
  blurred *along* the strands (steered line kernels), which combs the hair
  instead of melting it like an isotropic blur would; where no direction
  dominates (curls, noise) a small isotropic blur takes over. The change is
  added in the hair's own color, and the finest strand detail is attenuated
  a little. Only the hair's interior is touched, so the silhouette stays
  crisp.
* Gloss: brightens the hair's own highlights (areas brighter than most of
  the hair, measured per frame on hair pixels only), most strongly on the
  "angel ring": an arc a little inside the head's silhouette over the top of
  the head, where a faint sheen appears even on dark hair. The brighter
  strands gain contrast and a sheen of the scene's light (partly in the
  hair's hue) is screened on, modulated by the strand texture, so the shine
  follows the strands instead of forming a flat stripe, stays off the
  outermost strands (no halo) and never turns studio white in a dim room.
* Color: hair dye is subtractive, so the color is a per-channel gain that
  takes the hair's average color to the chosen color while keeping every
  strand's relative light and shade. Darkening follows the target almost
  fully (e.g. toning light hair down to dark brown for an interview);
  lightening is limited like a real dye, so dark hair only takes a subtle
  tint. The chosen color means "in neutral light": it is shifted by the
  scene's color cast, measured on the forehead skin, so a dark brown stays
  warm under a warm lamp instead of turning ashy.
* Brightness / contrast: a tone curve on the hair's luminance. Contrast is
  an S-curve pivoting at the hair's mean luminance that never clips;
  brightening is an exposure change that rolls off toward white. The chroma
  follows a brightening only partly, so lifted shadows keep the hair's hue
  instead of turning orange.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass

import cv2
import numpy as np

from .. import blend, regions
from ..config import HairSettings, hex_to_bgr
from .base import Effect, FrameContext, register_effect

Bgr = tuple[float, float, float]

# Lengths are fractions of the face scale (inter-ocular distance) unless noted.

# ---- region and grids ---------------------------------------------------------
_COARSE_STRIDE = 8  # grid step (frame px) of the blob search
_BLOB_THRESHOLD = 0.3  # hair confidence that counts as a blob
_MIN_BLOB_CELLS = 2  # smaller blobs are noise
_HEAD_REACH = 1.0  # blobs must come this close to the face oval
_SECONDARY_BLOB_SHARE = 0.25  # without a face: keep blobs this big vs. the largest
_ROI_MARGIN = 0.3  # around the hair blobs
_HAIR_WIDTH_PER_SCALE = 3.3  # hair width / face scale, to guess the scale without a face
_MIN_SCALE = 8.0  # frame px
# Copies are downscaled by integer factors chosen for a face scale of about
# _WORK_SCALE / _RENDER_SCALE pixels and at most about _WORK_PIXELS /
# _RENDER_PIXELS pixels (long hair), never upscaled. A factor only changes
# when the ideal one moves this far beyond rounding, so a head hovering at a
# threshold does not make the look flicker between two resolutions.
_WORK_SCALE = 36.0
_WORK_PIXELS = 30_000
_RENDER_SCALE = 48.0
_RENDER_PIXELS = 100_000
_FACTOR_HYSTERESIS = 0.15

_CONFIDENCE_RAMP = (0.2, 0.7)  # hair confidence (minus face skin) -> weight
_EDGE_FEATHER = (0.01, 0.06)  # edge feather at edge_softness 0 and 1
_INTERIOR_SIGMA = 0.07  # soft erosion: smoothing and gloss stay off the edges
# Inner face protection: the face oval below the brows, shrunk at the sides
# (hair framing the face is kept), closed over the slightly raised brows.
_INNER_FACE_SIDES = slice(6, 31)  # FACE_OVAL from one temple around the chin to the other
_INNER_FACE_SHRINK = 0.25
_BROW_RAISE = 0.12
_BROW_GUARD = 0.1  # each brow, grown by this, is protected as well (its tail lies outside)
_INNER_FACE_FEATHER = 0.05  # outward only: the protected shapes stay solid
_FOREHEAD, _CHIN = 10, 152

# ---- statistics -----------------------------------------------------------
_MIN_INTERIOR_CELLS = 20  # working-copy pixels needed to measure the hair
_STATS_EMA = 0.3  # share of a new frame's statistics (steady colors)
_STATS_STEP = 2  # percentiles use every n-th pixel per axis
_LOCK_SIGMA = 0.06  # lock-scale smoothing of the luminance for the highlight map
# Smoothed luminance percentiles: highlight ramp start on the angel ring,
# start elsewhere, and full strength.
_HIGHLIGHT_PERCENTILES = (25.0, 55.0, 97.0)
_LIGHT_PERCENTILE = 99.0  # the ROI's brightest pixels give the light level
_LIGHT_LEVEL_RANGE = (0.35, 1.0)
# Scene color cast: forehead skin chromaticity (BGR / luma) vs. a typical skin
# in neutral light (Lab 65, 11, 18); this share of the ratio, limited, is used.
_REFERENCE_SKIN = np.array([0.795, 0.945, 1.187], np.float32)
_SKIN_CONFIDENCE = 0.6
_CAST_ADAPTATION = 0.75
_CAST_RANGE = (0.8, 1.25)

# ---- smoothness -------------------------------------------------------------
_TENSOR_SIGMA = 0.06  # orientation neighborhood
_STRAND_LENGTH = 0.12  # half-length of the line kernels
_MAX_STRAND_HALF_LENGTH = 7  # px: diagonal kernels up to 11x11 (filter2D turns to a slow DFT above)
_DIAGONAL = math.sqrt(0.5)
# Directions of the line kernels (x, y; y down): 0, 45, 90 and 135 degrees.
_LINE_DIRECTIONS = ((1.0, 0.0), (_DIAGONAL, _DIAGONAL), (0.0, 1.0), (-_DIAGONAL, _DIAGONAL))
_ISO_SIGMA = 0.03  # blur where no direction dominates
_COHERENCE_RAMP = (0.15, 0.5)
_SMOOTH_GAIN = 0.8  # share of the combing applied at 100%
_FINE_DETAIL_CUT = 0.4  # share of the strand-scale detail removed at 100%

# ---- color and tone --------------------------------------------------------
_LUMA_WEIGHTS = np.array([0.114, 0.587, 0.299], np.float32)  # B, G, R
_DARKEN_FOLLOW = 0.9  # exponent on the luminance ratio when darkening
_LIGHTEN_FOLLOW = 0.25  # ... and when lightening (dye cannot lighten much)
_MAX_LIGHTEN = 1.35
_CHROMA_GAIN_RANGE = (0.4, 2.5)
# On hair darker than the target, the chroma gains are raised to the power
# (hair / target luminance) ** this: a red on black hair is a dark tint.
_DARK_HAIR_CHROMA = 1.5
_BRIGHTNESS_STOPS = 1.2  # exposure change (stops) per unit brightness
_CONTRAST_GAIN = 1.0  # S-curve exponent change per unit contrast
_TONE_GAIN_RANGE = (0.25, 4.0)  # limits of the tone curve's luminance gain (deep shadows)

# ---- gloss ------------------------------------------------------------------
_STRAND_SIGMA = 0.03  # detail finer than this is strand texture
_DETAIL_FLOOR = 0.03  # luminance floor when strand detail is taken relative to a level
_GLOSS_EDGE_CLEARANCE = (0.5, 0.95)  # interior weight -> gloss weight (off the outermost strands)
_RING_DEPTH = 0.25  # depth of the angel ring's center below the head's silhouette
_RING_HALF_WIDTH = 0.2
_RING_HEIGHT = (0.9, 1.5)  # height above the eyes -> ring weight (upper head only)
_RING_FACE_GROW = 0.2  # the face oval is grown by this to reach the hairline
_RING_BASE = 0.4  # gloss off the ring, relative to on it
_RING_FLOOR = 0.4  # highlight level assumed on the ring even where the hair is dark
_EYES_BELOW_HAIR_TOP = 2.1  # to place the ring without a face
_SHEEN = 0.45  # screen weight at 100%
_SHEEN_HAIR_TINT = 0.5  # share of the hair's own hue in the sheen (the rest is the light's)
_STRAND_BOOST = 1.2  # contrast gain of the brighter strands at 100%
# Sheen per strand (flat, slope, most): flat + slope x relative strand detail,
# clipped to [0, most], so the sheen sits on the brighter strands and skips
# the dark gaps.
_STRAND_MODULATION = (0.6, 6.0, 2.2)


@dataclass(frozen=True)
class _Levels:
    """Slider values times the master intensity."""

    smooth: float
    gloss: float
    dye: float
    brightness: float
    contrast: float

    @classmethod
    def from_settings(cls, settings: HairSettings, intensity: float) -> _Levels:
        k = max(float(intensity), 0.0)
        return cls(
            smooth=_clamp(settings.smoothness, 0.0, 1.0) * k,
            gloss=_clamp(settings.gloss, 0.0, 1.0) * k,
            dye=_clamp(settings.color_strength, 0.0, 1.0) * k,
            brightness=_clamp(settings.brightness, -0.5, 0.5) * k,
            contrast=_clamp(settings.contrast, -0.5, 0.5) * k,
        )

    @property
    def active(self) -> bool:
        return any((self.smooth, self.gloss, self.dye, self.brightness, self.contrast))


@dataclass(frozen=True)
class _Stats:
    """Measurements of the hair and the scene (temporally smoothed)."""

    color: np.ndarray  # mean BGR of the hair interior
    highlights: np.ndarray  # lock-luminance percentiles (see _HIGHLIGHT_PERCENTILES)
    light: float  # the scene's light level
    cast: np.ndarray  # per-channel color cast of the light (1 = neutral)

    def blended(self, previous: _Stats | None) -> _Stats:
        if previous is None:
            return self

        def ema(old: np.ndarray | float, new: np.ndarray | float) -> np.ndarray | float:
            return old + _STATS_EMA * (new - old)

        return _Stats(
            ema(previous.color, self.color),
            ema(previous.highlights, self.highlights),
            ema(previous.light, self.light),
            ema(previous.cast, self.cast),
        )


@dataclass(frozen=True)
class _Area:
    """Where the hair is: the ROI, the face scale and the two copies' grids.

    The working and render copies are the ROI downscaled by integer factors;
    the ROI size is a multiple of both, which keeps box-filter downscaling
    exact and fast and all grids aligned.
    """

    roi: regions.Roi
    scale: float  # face scale, frame px
    work_factor: int  # frame px per working-copy px
    render_factor: int  # frame px per render-copy px
    landmarks: np.ndarray | None  # in working-copy coordinates

    @property
    def work_size(self) -> tuple[int, int]:
        return self.roi.w // self.work_factor, self.roi.h // self.work_factor

    @property
    def render_size(self) -> tuple[int, int]:
        return self.roi.w // self.render_factor, self.roi.h // self.render_factor

    @property
    def work_scale(self) -> float:
        return self.scale / self.work_factor

    def to_work(self, image: np.ndarray) -> np.ndarray:
        """ROI-sized array -> new working-copy array (box-filtered)."""
        return _shrink(image, self.work_factor, self.work_size)

    def to_render(self, image: np.ndarray) -> np.ndarray:
        """ROI-sized array -> new render-copy array (box-filtered)."""
        return _shrink(image, self.render_factor, self.render_size)

    def sample(self, image: np.ndarray) -> np.ndarray:
        """ROI-sized array -> working-copy-sized subsample (for averages only)."""
        return np.ascontiguousarray(image[:: self.work_factor, :: self.work_factor])

    def work_to_render(self, image: np.ndarray) -> np.ndarray:
        """Working-copy array -> render-copy array (the same array if the grids match)."""
        return _enlarge(image, self.work_factor == self.render_factor, self.render_size)

    def render_to_full(self, image: np.ndarray) -> np.ndarray:
        """Render-copy array -> ROI-sized array (the same array if not downscaled)."""
        return _enlarge(image, self.render_factor == 1, (self.roi.w, self.roi.h))


@dataclass(frozen=True)
class _Weights:
    """Working-copy hair weights."""

    hair: np.ndarray  # full coverage up to the (feathered) edge
    interior: np.ndarray  # fades out before the edges


@dataclass(frozen=True)
class _Dye:
    """Per-channel dye gains and where (and how strongly) they apply (render copy)."""

    gains: np.ndarray
    weight: np.ndarray  # hair weight x color strength

    def channel(self, index: int) -> np.ndarray:
        """Gain map of one channel: 1 off the hair, the dye gain on it."""
        return 1.0 + self.weight * np.float32(self.gains[index] - 1.0)


@register_effect
class HairEffect(Effect):
    """Hair gloss, sleekness, color, brightness and contrast."""

    name = "hair"
    order = 10
    needs_face = False
    needs_hair_mask = True

    def __init__(self) -> None:
        self._stats: _Stats | None = None
        self._factors: tuple[int, int] | None = None  # (work, render) of the last frame
        # The segmenter hands out the very same (read-only) mask array until
        # its next run, so the blob search runs once per mask.
        self._blobs: tuple[np.ndarray, np.ndarray] | None = None  # (mask, blob boxes)

    def reset(self) -> None:
        self._stats = None
        self._factors = None
        self._blobs = None

    def apply(self, frame: np.ndarray, ctx: FrameContext, settings: HairSettings) -> np.ndarray:
        levels = _Levels.from_settings(settings, ctx.intensity)
        if not levels.active or ctx.hair_mask is None:
            return frame
        if self._blobs is None or self._blobs[0] is not ctx.hair_mask:
            self._blobs = (ctx.hair_mask, _hair_blobs(ctx.hair_mask))
        area = _hair_area(self._blobs[1], ctx.landmarks, frame.shape, self._factors)
        if area is None:
            return frame
        self._factors = (area.work_factor, area.render_factor)
        view = frame[area.roi.slices]
        luma = blend.luminance(view)
        work_luma = area.to_work(luma)
        face = None if ctx.face_skin_mask is None else area.to_work(ctx.face_skin_mask[area.roi.slices])
        weights = _hair_weights(area.to_work(ctx.hair_mask[area.roi.slices]), face, area, settings.edge_softness)
        lock_luma, hair_luma = _hair_luminance(work_luma, weights.hair, area.work_scale)
        skin = None if face is None else face > _SKIN_CONFIDENCE
        stats = _measure(area.sample(view), work_luma, lock_luma, weights.interior, skin)
        if stats is None:
            return frame
        self._stats = stats = stats.blended(self._stats)
        target = _scene_color(hex_to_bgr(settings.color), stats.cast)
        scale_map, offset_map = _look(area.to_render(luma), hair_luma, lock_luma, weights, area, stats, levels, target)
        view *= area.render_to_full(scale_map)
        if offset_map is not None:
            view += area.render_to_full(offset_map)
        np.clip(view, 0.0, 1.0, out=view)
        return frame


# --------------------------------------------------------------------------
# Region
# --------------------------------------------------------------------------


def _hair_area(
    blobs: np.ndarray,
    landmarks: np.ndarray | None,
    shape: tuple[int, ...],
    previous_factors: tuple[int, int] | None,
) -> _Area | None:
    """ROI around the hair blobs attached to the head, or None when there are none."""
    box = _head_box(blobs, landmarks)
    if box is None:
        return None
    x0, y0, x1, y1 = box
    scale = regions.face_scale(landmarks) if landmarks is not None else (x1 - x0) / _HAIR_WIDTH_PER_SCALE
    scale = max(scale, _MIN_SCALE)
    roi = regions.roi_around(np.array([[x0, y0], [x1, y1]], np.float32), shape, pad=0.0, margin_px=_ROI_MARGIN * scale)
    if roi is None:
        return None
    pixels = roi.w * roi.h
    work_previous, render_previous = previous_factors or (None, None)
    work_factor = _grid_factor(max(scale / _WORK_SCALE, math.sqrt(pixels / _WORK_PIXELS)), work_previous)
    render_factor = _grid_factor(max(scale / _RENDER_SCALE, math.sqrt(pixels / _RENDER_PIXELS)), render_previous)
    step = math.lcm(work_factor, render_factor)
    if roi.w < step or roi.h < step:
        return None
    roi = regions.Roi(roi.x0, roi.y0, roi.x1 - roi.w % step, roi.y1 - roi.h % step)
    local = None
    if landmarks is not None:
        local = (landmarks - np.asarray(roi.offset, np.float32)) * np.float32(1.0 / work_factor)
    return _Area(roi, scale, work_factor, render_factor, local)


def _grid_factor(ideal: float, previous: int | None) -> int:
    """Integer downscale factor for an ``ideal`` (fractional) one, kept at
    ``previous`` while ``ideal`` stays within the hysteresis band around it."""
    if previous is not None and abs(ideal - previous) < 0.5 + _FACTOR_HYSTERESIS:
        return previous
    return max(1, round(ideal))


def _hair_blobs(hair_mask: np.ndarray) -> np.ndarray:
    """Boxes of the confident hair blobs on a coarse grid: (N, 5) int rows of
    x, y, w, h, area in grid cells (``_COARSE_STRIDE`` frame px)."""
    step = _COARSE_STRIDE
    coarse = (hair_mask[::step, ::step] > _BLOB_THRESHOLD).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(coarse, connectivity=8)
    boxes = stats[1:count]  # label 0 is the background
    return boxes[boxes[:, cv2.CC_STAT_AREA] >= _MIN_BLOB_CELLS]


def _head_box(blobs: np.ndarray, landmarks: np.ndarray | None) -> tuple[int, int, int, int] | None:
    """Frame-px bounding box ``(x0, y0, x1, y1)`` of the blobs that belong to the head."""
    if len(blobs) and landmarks is not None:
        blobs = blobs[_near_face(blobs, landmarks)]
    elif len(blobs):
        sizes = blobs[:, cv2.CC_STAT_AREA]
        blobs = blobs[sizes >= _SECONDARY_BLOB_SHARE * sizes.max()]
    if len(blobs) == 0:
        return None
    x0, y0 = blobs[:, 0].min(), blobs[:, 1].min()
    x1, y1 = (blobs[:, 0] + blobs[:, 2]).max(), (blobs[:, 1] + blobs[:, 3]).max()
    step = _COARSE_STRIDE
    return int(x0) * step, int(y0) * step, int(x1) * step, int(y1) * step


def _near_face(boxes: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """Which blob boxes (coarse grid) touch the face oval's box grown by ``_HEAD_REACH``."""
    oval = landmarks[regions.FACE_OVAL]
    reach = _HEAD_REACH * regions.face_scale(landmarks)
    lo = (oval.min(axis=0) - reach) / _COARSE_STRIDE
    hi = (oval.max(axis=0) + reach) / _COARSE_STRIDE
    bx0, by0 = boxes[:, 0], boxes[:, 1]
    bx1, by1 = bx0 + boxes[:, 2], by0 + boxes[:, 3]
    return (bx0 <= hi[0]) & (bx1 >= lo[0]) & (by0 <= hi[1]) & (by1 >= lo[1])


def _hair_weights(hair: np.ndarray, face: np.ndarray | None, area: _Area, edge_softness: float) -> _Weights:
    """Hair weights from the working-copy hair and face-skin confidences (``hair`` is modified).

    The edge feather only fades the effects in toward the inside: it never
    spreads them beyond the segmented hair (a dye would leave a dark halo on
    a light wall). Face skin is subtracted before the confidence ramp (a
    sharper hairline) and again at the end, together with the inner face.
    """
    if face is not None:
        hair *= 1.0 - face
    weight = _smoothstep(hair, *_CONFIDENCE_RAMP)
    softness = _clamp(edge_softness, 0.0, 1.0)
    edge = _EDGE_FEATHER[0] + (_EDGE_FEATHER[1] - _EDGE_FEATHER[0]) * softness
    weight = np.minimum(regions.feather(weight, edge * area.work_scale), weight)
    if face is not None:
        weight *= 1.0 - face
    if area.landmarks is not None:
        weight *= 1.0 - _inner_face(area)
    interior = regions.feather(weight, _INTERIOR_SIGMA * area.work_scale)
    interior = np.clip(interior * 2.0 - 1.0, 0.0, 1.0) * weight
    return _Weights(weight, interior)


def _inner_face(area: _Area) -> np.ndarray:
    """Soft mask of the inner face and the brows (working copy), which never count as hair."""
    lm = area.landmarks
    scale = area.work_scale
    sides = regions.expand_polygon(lm[regions.FACE_OVAL[_INNER_FACE_SIDES]], -_INNER_FACE_SHRINK * scale)
    brow_tops = np.vstack([lm[regions.RIGHT_BROW_UPPER[::-1]], lm[regions.LEFT_BROW_UPPER]])
    brow_tops = brow_tops + _face_up(lm) * np.float32(_BROW_RAISE * scale)
    shapes = [np.vstack([sides, brow_tops])]
    for brow in (regions.RIGHT_BROW, regions.LEFT_BROW):
        shapes.append(regions.expand_polygon(lm[brow], _BROW_GUARD * scale))
    return _soft_polygons(shapes, area.work_size, _INNER_FACE_FEATHER * scale)


def _soft_polygons(shapes: list[np.ndarray], size: tuple[int, int], sigma: float) -> np.ndarray:
    """Union of filled polygons on a ``size`` (w, h) grid, solid inside and
    feathered outward by ``sigma``.

    A cheap stand-in for ``regions.polygon_mask``: the shapes are blurred by
    more than a pixel, so cv2's anti-aliased fill is accurate enough and the
    supersampled rasterizer would only cost time.
    """
    shift = 4
    canvas = np.zeros((size[1], size[0]), np.uint8)
    for points in shapes:
        fixed = np.round((points - 0.5) * (1 << shift)).astype(np.int32)
        cv2.fillPoly(canvas, [fixed], 255, cv2.LINE_AA, shift)
    mask = regions.feather(canvas.astype(np.float32) * np.float32(1.0 / 255.0), sigma)
    return np.minimum(mask * 2.0, 1.0)


def _face_up(landmarks: np.ndarray) -> np.ndarray:
    """Unit vector from the chin toward the forehead."""
    up = landmarks[_FOREHEAD] - landmarks[_CHIN]
    return (up / max(float(np.linalg.norm(up)), 1e-6)).astype(np.float32)


def _hair_luminance(work_luma: np.ndarray, weight: np.ndarray, scale: float) -> tuple[np.ndarray, np.ndarray]:
    """The hair's luminance without its surroundings (working copy): ``(lock, filled)``.

    ``lock`` is the luminance smoothed over hair pixels only (normalized
    convolution), so a bright background or forehead never makes the hair's
    edges look like highlights. ``filled`` is the luminance with everything
    outside the hair replaced by ``lock``, so filters near the edges see
    only hair.
    """
    sigma = _LOCK_SIGMA * scale
    lock = regions.feather(work_luma * weight, sigma) / np.maximum(regions.feather(weight, sigma), 1e-3)
    filled = work_luma - lock
    filled *= weight
    filled += lock
    return lock, filled


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------


def _measure(
    colors: np.ndarray,
    work_luma: np.ndarray,
    lock_luma: np.ndarray,
    interior: np.ndarray,
    skin: np.ndarray | None,
) -> _Stats | None:
    """Hair and scene statistics; ``colors`` is a working-copy-sized sample of
    the ROI, ``skin`` marks confident face skin (the forehead) if known."""
    inside = interior > 0.5
    if int(np.count_nonzero(inside)) < _MIN_INTERIOR_CELLS:
        return None
    color = np.array(cv2.mean(colors, mask=inside.astype(np.uint8))[:3], np.float32)
    step = _STATS_STEP
    highlights = np.percentile(lock_luma[::step, ::step][inside[::step, ::step]], _HIGHLIGHT_PERCENTILES)
    light = float(np.percentile(work_luma[::step, ::step], _LIGHT_PERCENTILE))
    return _Stats(
        color=color,
        highlights=highlights.astype(np.float32),
        light=_clamp(light, *_LIGHT_LEVEL_RANGE),
        cast=_color_cast(colors, skin),
    )


def _color_cast(colors: np.ndarray, skin: np.ndarray | None) -> np.ndarray:
    """Per-channel tint of the light, from the skin's chromaticity (ones if unknown)."""
    neutral = np.ones(3, np.float32)
    if skin is None or int(np.count_nonzero(skin)) < _MIN_INTERIOR_CELLS:
        return neutral
    color = np.array(cv2.mean(colors, mask=skin.astype(np.uint8))[:3], np.float32)
    luma = float(np.dot(_LUMA_WEIGHTS, color))
    if luma <= 1e-3:
        return neutral
    ratio = (color / luma / _REFERENCE_SKIN) ** _CAST_ADAPTATION
    return np.clip(ratio, *_CAST_RANGE).astype(np.float32)


def _scene_color(color_bgr: Bgr, cast: np.ndarray) -> np.ndarray:
    """``color_bgr`` as it would look in the scene's light, at the same luminance."""
    color = np.asarray(color_bgr, np.float32)
    tinted = color * cast
    scale = float(np.dot(_LUMA_WEIGHTS, color)) / max(float(np.dot(_LUMA_WEIGHTS, tinted)), 1e-6)
    return np.clip(tinted * scale, 0.0, 1.0)


# --------------------------------------------------------------------------
# The look, on the render copy
# --------------------------------------------------------------------------


def _look(
    luma: np.ndarray,
    hair_luma: np.ndarray,
    lock_luma: np.ndarray,
    weights: _Weights,
    area: _Area,
    stats: _Stats,
    levels: _Levels,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Render-copy maps ``(scale, offset)`` (HxWx3) with ``out = in * scale + offset``.

    ``luma`` is the render copy's luminance, ``hair_luma`` the working copy's
    filled hair luminance (see ``_hair_luminance``). In order: the smoothing
    is added (in the hair's own color); a luminance gain applies the finer
    smoothing and the strand contrast; the dye gains color the hair; the
    tone curve changes the luminance (the chroma follows a brightening only
    partly, so lifted shadows do not turn orange); the sheen is screened on
    top.
    """
    weight = area.work_to_render(weights.hair)
    gain = np.ones_like(luma)  # luminance gain (keeps the hue)
    added = None  # smoothing: luminance added in the hair's color
    if levels.smooth > 0:
        work_added = _comb(hair_luma, area.work_scale) - hair_luma
        work_added *= weights.interior * np.float32(levels.smooth * _SMOOTH_GAIN)
        added = area.work_to_render(work_added)
        luma = luma + added
        hair_luma = hair_luma + work_added
    sheen = None
    if levels.smooth > 0 or levels.gloss > 0:
        detail, level = _strand_detail(luma, hair_luma, area)
        if levels.smooth > 0:
            cut = area.work_to_render(weights.interior * np.float32(levels.smooth * _FINE_DETAIL_CUT))
            gain -= cut * detail / np.maximum(luma, _DETAIL_FLOOR)
        if levels.gloss > 0:
            zone = area.work_to_render(_gloss_zone(lock_luma, weights, area, stats) * np.float32(levels.gloss))
            boost, sheen = _gloss(detail / (level + _DETAIL_FLOOR), zone, stats.light)
            gain *= boost
    dye = _Dye(_dye_gains(stats.color, target), weight * np.float32(levels.dye)) if levels.dye > 0 else None
    tone_offset = None
    if levels.brightness or levels.contrast:
        tone_scale, tone_offset = _tone(luma * gain, weight, dye, stats.color, levels)
        gain *= tone_scale
    added_color = None if added is None else _chromaticity(stats.color)
    return _channel_maps(gain, dye, added, added_color, tone_offset, sheen, _sheen_tint(stats))


def _channel_maps(
    gain: np.ndarray,
    dye: _Dye | None,
    added: np.ndarray | None,
    added_color: np.ndarray | None,
    tone_offset: np.ndarray | None,
    sheen: np.ndarray | None,
    sheen_tint: np.ndarray,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Per-channel ``(scale, offset)`` maps of
    ``out = ((in + added * added_color) * gain * dye + tone_offset) * (1 - light) + light``
    with ``light = sheen * sheen_tint``; terms that are None are skipped."""
    scales, offsets = [], []
    for channel in range(3):
        scale = gain if dye is None else gain * dye.channel(channel)
        offset = None
        if added is not None and added_color is not None:
            offset = added * (scale * np.float32(added_color[channel]))
        if tone_offset is not None:
            offset = tone_offset if offset is None else offset + tone_offset
        if sheen is not None:
            light = sheen * np.float32(sheen_tint[channel])
            keep = 1.0 - light
            scale = scale * keep
            offset = light if offset is None else offset * keep + light
        scales.append(scale)
        offsets.append(offset)
    return cv2.merge(scales), (None if offsets[0] is None else cv2.merge(offsets))


def _strand_detail(luma: np.ndarray, hair_luma: np.ndarray, area: _Area) -> tuple[np.ndarray, np.ndarray]:
    """Strand-scale luminance detail of ``luma`` (render copy) and the hair's
    local level it is measured against: ``(luma - level, level)``."""
    base = area.work_to_render(regions.feather(hair_luma, _STRAND_SIGMA * area.work_scale))
    return luma - base, base


def _sheen_tint(stats: _Stats) -> np.ndarray:
    """Per-channel color of the sheen (max 1): mostly the light's, partly the hair's hue."""
    light = np.minimum(stats.cast, 1.0)
    hair = stats.color / max(float(stats.color.max()), 1e-3)
    return (1.0 - _SHEEN_HAIR_TINT) * light + _SHEEN_HAIR_TINT * hair * light


def _chromaticity(color: np.ndarray) -> np.ndarray:
    """``color`` divided by its luminance: per-channel factors of a neutral luminance change."""
    return color / max(float(np.dot(_LUMA_WEIGHTS, color)), 1e-3)


# ---- smoothness -------------------------------------------------------------


def _comb(luma: np.ndarray, scale: float) -> np.ndarray:
    """Blur ``luma`` along the local strand direction (isotropic where there is none).

    The direction comes from the structure tensor. Four line kernels
    (``_LINE_DIRECTIONS``) are steered by the squared positive and negative
    parts of the doubled-angle direction vector; they sum to one, so each
    pixel blends the two kernels nearest to its direction.
    """
    gx = cv2.Sobel(luma, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(luma, cv2.CV_32F, 0, 1, ksize=3)
    tensor = cv2.GaussianBlur(cv2.merge([gx * gx, gy * gy, gx * gy]), (0, 0), _TENSOR_SIGMA * scale)
    jxx, jyy, jxy = cv2.split(tensor)
    # The doubled-angle vector of the gradient is (jxx - jyy, 2 jxy); strands
    # run perpendicular to the gradient, so theirs is its negative.
    c2 = jyy - jxx
    s2 = -2.0 * jxy
    norm = cv2.magnitude(c2, s2)
    coherence = norm / (jxx + jyy + 1e-6)
    norm += 1e-9
    c2 /= norm
    s2 /= norm
    half_length = min(_MAX_STRAND_HALF_LENGTH, max(1, round(_STRAND_LENGTH * scale)))
    horizontal, diagonal, vertical, antidiagonal = (
        cv2.filter2D(luma, -1, kernel, borderType=cv2.BORDER_REPLICATE) for kernel in _line_kernels(half_length)
    )
    c_pos = np.maximum(c2, 0.0)
    s_pos = np.maximum(s2, 0.0)
    c_neg = c2 - c_pos
    s_neg = s2 - s_pos
    along = c_pos * c_pos * horizontal
    along += c_neg * c_neg * vertical
    along += s_pos * s_pos * diagonal
    along += s_neg * s_neg * antidiagonal
    around = cv2.GaussianBlur(luma, (0, 0), max(_ISO_SIGMA * scale, 0.5))
    along -= around
    along *= _smoothstep(coherence, *_COHERENCE_RAMP)
    along += around
    return along


@functools.lru_cache(maxsize=8)
def _line_kernels(half_length: int) -> tuple[np.ndarray, ...]:
    """Anti-aliased line kernels with a Gaussian profile along ``_LINE_DIRECTIONS``.

    Each kernel is only as large as its line (1 x n for the horizontal one,
    a smaller square for the diagonals), which keeps filtering cheap.
    """
    t = np.linspace(-half_length, half_length, 16 * half_length + 1)
    profile = np.exp(-0.5 * (t / (0.5 * half_length)) ** 2)
    kernels = []
    for dx, dy in _LINE_DIRECTIONS:
        reach_x, reach_y = math.ceil(half_length * abs(dx) - 1e-9), math.ceil(half_length * abs(dy) - 1e-9)
        x, y = reach_x + t * dx, reach_y + t * dy
        x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
        fx, fy = x - x0, y - y0
        kernel = np.zeros((2 * reach_y + 2, 2 * reach_x + 2), np.float64)
        for ox, oy, w in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
            np.add.at(kernel, (y0 + oy, x0 + ox), w * profile)
        kernel = kernel[: 2 * reach_y + 1, : 2 * reach_x + 1]  # the extra row/column only got zero weights
        kernels.append((kernel / kernel.sum()).astype(np.float32))
    return tuple(kernels)


# ---- color and tone ---------------------------------------------------------


def _dye_gains(hair_bgr: np.ndarray, target_bgr: np.ndarray) -> np.ndarray:
    """Per-channel gains taking the hair's mean color toward ``target_bgr``.

    Like a real dye (without bleach), the color shows fully on hair at least
    as light as the target, and only as a tint on darker hair: both the
    lightening and the chroma change shrink with the hair's darkness.
    """
    target = np.maximum(np.asarray(target_bgr, np.float32), 1e-3)
    hair = np.maximum(hair_bgr, 1e-3)
    hair_luma = float(np.dot(_LUMA_WEIGHTS, hair))
    target_luma = float(np.dot(_LUMA_WEIGHTS, target))
    ratio = target_luma / hair_luma
    chroma = np.clip((target / target_luma) / (hair / hair_luma), *_CHROMA_GAIN_RANGE)
    if ratio > 1.0:
        chroma **= ratio**-_DARK_HAIR_CHROMA
        lightness = min(ratio**_LIGHTEN_FOLLOW, _MAX_LIGHTEN)
    else:
        lightness = ratio**_DARKEN_FOLLOW
    return (chroma * lightness).astype(np.float32)


def _tone(
    luma: np.ndarray,
    weight: np.ndarray,
    dye: _Dye | None,
    hair_bgr: np.ndarray,
    levels: _Levels,
) -> tuple[np.ndarray, np.ndarray]:
    """The brightness / contrast curve on the dyed hair as ``(scale, luminance offset)``.

    The curve's luminance gain ``g`` scales the chroma by ``min(g, sqrt(g))``
    and adds the rest as neutral luminance.
    """
    hair_luma = max(float(np.dot(_LUMA_WEIGHTS, hair_bgr)), 1e-6)
    pivot, dyed = hair_luma, luma
    if dye is not None:
        luma_change = float(np.dot(_LUMA_WEIGHTS, dye.gains * hair_bgr)) / hair_luma - 1.0
        pivot = hair_luma * (1.0 + levels.dye * luma_change)
        dyed = luma * (1.0 + dye.weight * np.float32(luma_change))
    lut = _tone_lut(pivot, levels.brightness, levels.contrast)
    gain = cv2.LUT(cv2.convertScaleAbs(dyed, alpha=255.0), lut)
    gain -= 1.0
    gain *= weight
    gain += 1.0
    scale = np.minimum(gain, np.sqrt(gain))
    return scale, (gain - scale) * dyed


def _tone_lut(pivot: float, brightness: float, contrast: float) -> np.ndarray:
    """256-entry float32 gain table over luminance: contrast S-curve, then brightness."""
    y = np.arange(256, dtype=np.float64) / 255.0
    p = min(max(pivot, 0.02), 0.98)
    e = 1.0 + _CONTRAST_GAIN * contrast
    curved = np.where(y < p, p * (y / p) ** e, 1.0 - (1.0 - p) * ((1.0 - y) / (1.0 - p)) ** e)
    g = 2.0 ** (_BRIGHTNESS_STOPS * brightness)
    toned = 1.0 - (1.0 - curved) ** g if g > 1.0 else curved * g
    gain = toned / np.maximum(y, 1.0 / 255.0)
    return np.clip(gain, *_TONE_GAIN_RANGE).astype(np.float32)


# ---- gloss ------------------------------------------------------------------


def _gloss(detail: np.ndarray, zone: np.ndarray, light: float) -> tuple[np.ndarray, np.ndarray]:
    """``(gain, sheen)`` of the gloss: the brighter strands (relative strand
    ``detail`` > 0) gain contrast, and the sheen sits on them."""
    gain = 1.0 + zone * np.clip(detail, 0.0, 1.0) * np.float32(_STRAND_BOOST)
    flat, slope, most = _STRAND_MODULATION
    sheen = zone * np.clip(flat + slope * detail, 0.0, most) * np.float32(_SHEEN * light)
    return gain, sheen


def _gloss_zone(lock_luma: np.ndarray, weights: _Weights, area: _Area, stats: _Stats) -> np.ndarray:
    """Where gloss goes (working copy): the hair's own highlights, emphasized on
    the angel ring (where moderately bright hair counts as well and a faint
    sheen is added anyway), kept off the outermost strands, whose pixels mix
    with the background."""
    ring = _angel_ring(weights.hair, area)
    in_ring, elsewhere, full = (float(v) for v in stats.highlights)
    start = elsewhere + (in_ring - elsewhere) * ring
    highlights = np.clip((lock_luma - start) / np.maximum(full - start, 1e-4), 0.0, 1.0)
    highlights *= highlights * (3.0 - 2.0 * highlights)  # smoothstep
    highlights = np.maximum(highlights, ring * np.float32(_RING_FLOOR))
    return _smoothstep(weights.interior, *_GLOSS_EDGE_CLEARANCE) * highlights * (_RING_BASE + (1.0 - _RING_BASE) * ring)


def _angel_ring(hair: np.ndarray, area: _Area) -> np.ndarray:
    """Soft arc following the top of the head (working copy), where the "angel
    ring" of glossy hair forms: about ``_RING_DEPTH`` inside the head's outer
    silhouette, over the upper head.

    The face (oval grown up to the hairline) is filled in, so the depth is
    measured from the silhouette only, not from the hairline.
    """
    scale = area.work_scale
    head = (hair > 0.5).astype(np.uint8)
    if area.landmarks is not None:
        oval = regions.expand_polygon(area.landmarks[regions.FACE_OVAL], _RING_FACE_GROW * scale)
        cv2.fillPoly(head, [np.round(oval - 0.5).astype(np.int32)], 1)
    depth = cv2.distanceTransform(head, cv2.DIST_L2, 3) * np.float32(1.0 / scale)
    ring = np.clip(1.0 - ((depth - _RING_DEPTH) * np.float32(1.0 / _RING_HALF_WIDTH)) ** 2, 0.0, 1.0)
    ring *= ring
    return ring * _smoothstep(_height_above_eyes(head, area), *_RING_HEIGHT)


def _height_above_eyes(solid: np.ndarray, area: _Area) -> np.ndarray:
    """Height of each working-copy pixel above the eyes along the face's up axis
    (face scales); without a face, estimated from the top of the hair."""
    scale = area.work_scale
    h, w = solid.shape
    xs = np.arange(w, dtype=np.float32)[None, :]
    ys = np.arange(h, dtype=np.float32)[:, None]
    if area.landmarks is not None:
        lm = area.landmarks
        up = _face_up(lm)
        eyes = (regions.eye_center(lm, "left") + regions.eye_center(lm, "right")) / 2
        return ((xs - eyes[0]) * up[0] + (ys - eyes[1]) * up[1]) * np.float32(1.0 / scale)
    rows = np.flatnonzero(solid.any(axis=1))
    top = float(rows[0]) if len(rows) else 0.0
    return np.broadcast_to((top + _EYES_BELOW_HAIR_TOP * scale - ys) * np.float32(1.0 / scale), (h, w))


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _shrink(image: np.ndarray, factor: int, size: tuple[int, int]) -> np.ndarray:
    """Box-filter ``image`` down by an integer ``factor`` to ``size`` (w, h); always a new array."""
    if factor == 1:
        return np.array(image, dtype=np.float32)
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def _enlarge(image: np.ndarray, same_grid: bool, size: tuple[int, int]) -> np.ndarray:
    """Bilinear upsampling to ``size`` (w, h); ``image`` itself on the same grid."""
    if same_grid:
        return image
    return cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)


def _clamp(value: float, lo: float, hi: float) -> float:
    """``value`` as a float limited to [lo, hi]."""
    return min(max(float(value), lo), hi)


def _smoothstep(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """0 below ``lo``, 1 above ``hi``, a smooth S-curve in between."""
    t = np.clip((x - np.float32(lo)) * np.float32(1.0 / max(hi - lo, 1e-6)), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)
