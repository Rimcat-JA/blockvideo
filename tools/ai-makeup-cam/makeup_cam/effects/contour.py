"""Contour (shading + highlight) and tear bags (涙袋).

Both effects paint soft strokes anchored on face-mesh vertices: every stroke
is a polyline whose vertices are interpolated between two landmarks, so it
follows head roll, yaw and expressions like the skin under it, and every
width is a fraction of the face scale (inter-ocular distance).

ContourEffect ("contour", order 25, right after the skin retouch)
    Shading darkens the jawline (a band just inside the lower face outline,
    from below the ear to the chin, fading toward the chin center), the
    cheek hollow (a diagonal comma under the cheekbone from near the ear
    toward the mouth corner, ending well before it), the sides of the nose
    bridge (from the brow heads down, fading toward the tip) and the
    forehead along the top of the face outline (stronger at the temples).
    Highlight lightens the nose bridge, the top of the cheekbones (a "C"
    around the outer eye up to the temple), the center of the chin and the
    cupid's bow. Each area has its own weight slider; ``shading`` and
    ``highlight`` scale all of them. On a turned head the strokes of the
    far side narrow with that side of the face.

    Everything except the final per-pixel blend is computed on a coarse grid
    (face scale about ``_GRID_SCALE`` grid pixels): the masks are smooth,
    and so are the per-pixel color factors derived from them. Only
    ``out = b + b * gain + add`` runs at full resolution, so texture and
    pores stay intact. The allowed area is the face outline (inset so
    nothing spills past the jaw into the background or the neck) times the
    face-skin segmentation mask when it is reliable (hair, bangs or a hand
    in front of the face are left alone). The forehead band is only drawn
    with a segmentation mask that tells skin from hair, because the face
    mesh continues under bangs, and the parts of it that would lie on hair
    are dropped rather than slid onto the skin below. The eyes, brows and
    nostrils are kept out on the grid with outlines grown by more than a
    grid pixel; the lips, right next to the cupid's bow, at full resolution.

TearbagEffect ("tearbag", order 45, after the eyeshadow)
    A soft highlight band right under the lower lash line, from the inner
    corner (fading in over the inner third) to ``width`` of the way to the
    outer corner, plus a faint thin shadow along the lower edge of the bag.
    Both hang off the lower lash landmarks, so they follow blinks, gaze and
    smiles; the eye opening is cut out with a crisp, slightly grown edge, so
    the eyeball is never painted. When the person smiles (mouth widening,
    smoothed over a few frames) the bag is drawn fuller and brighter, as a
    real tear bag bulges.

Color (shared)
    Shading multiplies the skin by a per-channel filter derived from the
    shading color *relative to a reference skin tone*: on that reference
    the shaded skin takes exactly the chosen color, and every other skin
    gets the same relative darkening. The shadow therefore keeps the skin's
    own undertone: on deep skin it becomes a deeper brown, never a grey or
    ashy veil. The filter's tint is limited and may never cool the skin
    (cooling is what reads as ashy).
    Highlight lifts all channels of a pixel by the same factor, which keeps
    the skin's hue and saturation (a lit deep skin glows instead of turning
    chalky), softly limited by the brightest channel's headroom so nothing
    clips. Very light skin has little headroom, so there a little of the
    highlight color is also screened on top: the pale sheen a pearl
    highlighter gives light skin.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .. import regions
from ..config import ContourSettings, TearbagSettings, hex_to_bgr
from .base import Effect, FrameContext, register_effect

Bgr = tuple[float, float, float]

# All lengths below are fractions of the face scale (inter-ocular distance)
# unless noted otherwise.

# Faces smaller than this (pixels) are skipped: too few pixels to shape.
_MIN_FACE_SCALE = 12.0
# Fixed-point fractional bits for sub-pixel accurate cv2 drawing.
_DRAW_SHIFT = 4
_DRAW_ONE = 1 << _DRAW_SHIFT
# Stroke joints bending by more than ~15 degrees get a round fill.
_JOINT_COSINE = 0.966

# --------------------------------------------------------------------------
# Color.
# --------------------------------------------------------------------------

# Skin tone the shading color is defined on (a light-medium skin, BGR). On
# this skin, full shading turns the skin into the shading color.
_REFERENCE_SKIN: Bgr = hex_to_bgr("#E6C3AD")
# Luma weights in BGR order.
_LUMA_BGR = np.array([0.114, 0.587, 0.299], np.float32)
# The shading filter's luma never drops below this, and its per-channel tint
# (filter / its luma) stays within this range.
_MIN_SHADE_FACTOR = 0.3
_TINT_RANGE = (0.9, 1.12)
# Highlight: relative lift at full weight (x highlight color), and the screen
# share added on very light skin (brightest channel ramp -> 0..1).
_LIGHT_GAIN = 0.6
_SHEEN = 0.25
_SHEEN_RAMP = (0.75, 1.0)
_MIN_SHEEN_BASE = 0.05


def _shading_filter(color_bgr: Bgr) -> np.ndarray:
    """Per-channel multiplier (BGR, each in (0, 1]) applied at full shading.

    ``color / reference skin``, split into a luma factor and a tint. The
    tint is clipped to ``_TINT_RANGE`` and may not be cooler than neutral
    (blue gain above red gain); a cool color keeps only its darkening.
    """
    ratio = np.asarray(color_bgr, np.float32) / np.asarray(_REFERENCE_SKIN, np.float32)
    luma = float(np.dot(_LUMA_BGR, ratio))
    tint = np.clip(ratio / max(luma, 1e-6), *_TINT_RANGE)
    if tint[0] > tint[2]:  # cooler than the skin: would read as grey on deep skin
        tint[0] = tint[2] = (tint[0] + tint[2]) / 2
    tint /= float(np.dot(_LUMA_BGR, tint))
    return np.minimum(tint * min(max(luma, _MIN_SHADE_FACTOR), 1.0), 1.0).astype(np.float32)


@dataclass(frozen=True)
class _Light:
    """How a highlight color lifts the skin (see ``_tone_factors``)."""

    gain: np.ndarray  # per-channel relative lift at full weight (BGR)
    color: np.ndarray  # sheen color (BGR)

    @classmethod
    def of(cls, color_bgr: Bgr, gain: float = _LIGHT_GAIN) -> _Light:
        color = np.clip(np.asarray(color_bgr, np.float32), 0.0, 1.0)
        return cls((color * np.float32(gain)).astype(np.float32), color)


def _tone_gain(
    image: np.ndarray,
    shade: np.ndarray | None,
    light: np.ndarray | None,
    shade_filter: np.ndarray,
    highlight: _Light,
) -> np.ndarray:
    """Per-pixel, per-channel ``gain`` so that ``b * (1 + gain)`` applies shading, then highlight.

    ``shade`` / ``light`` are HxW weights in [0, 1] at the resolution of
    ``image`` (the frame or a downscaled copy of it, used for the
    highlight's headroom and sheen). Shading multiplies by
    ``1 - w * (1 - filter)``. Highlight multiplies by
    ``1 + w * gain * room`` with ``room = (1 - M) / (1 - M + gain * M)`` and
    ``M`` the brightest channel (so ``M`` stays below 1), then screens
    ``s = w * sheen(M)`` of its color on top, written as a gain:
    ``b (1 - s) + s c = b (1 - s + s c / b)`` (exact at full resolution; on
    a downscaled copy the local mean stands in for ``b``). Where both
    weights are 0 the gain is exactly 0.
    """
    if shade is not None:
        factor = np.float32(1.0) - shade[..., None] * (np.float32(1.0) - shade_filter)
    else:
        factor = np.ones(image.shape, np.float32)
    if light is not None:
        brightest = _brightest(image)
        room = np.float32(1.0) - brightest
        room /= np.maximum(room + np.float32(highlight.gain.max()) * brightest, np.float32(1e-6))
        factor *= (light * room)[..., None] * highlight.gain + np.float32(1.0)
        sheen = light * _ramp(brightest, *_SHEEN_RAMP)
        if float(sheen.max()) > 0.0:
            ratio = highlight.color / np.maximum(image, np.float32(_MIN_SHEEN_BASE)) - np.float32(1.0)
            ratio *= (sheen * np.float32(_SHEEN))[..., None]
            factor *= ratio + np.float32(1.0)
    factor -= np.float32(1.0)
    return factor


def _brightest(image: np.ndarray) -> np.ndarray:
    """Per-pixel maximum over the color channels (cv2 is much faster than ndarray.max here)."""
    b, g, r = cv2.split(image)
    return cv2.max(cv2.max(b, g), r)


def _apply_gain(view: np.ndarray, gain: np.ndarray) -> None:
    """``view = min(view * (1 + gain), 1)`` in place (bit-exact where the gain is 0)."""
    out = cv2.add(view, cv2.multiply(view, gain))
    np.minimum(out, np.float32(1.0), out=view)


# --------------------------------------------------------------------------
# Stroke drawing (shared): polylines with per-vertex half-width and level.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _Path:
    """An open polyline in canvas pixels with a half-width and a level per vertex."""

    points: np.ndarray  # (N, 2)
    radii: np.ndarray  # (N,) half-widths in pixels
    levels: np.ndarray  # (N,) 0..1


def _fixed(points: np.ndarray) -> np.ndarray:
    """Continuous coordinates (pixel i covers [i, i + 1)) -> cv2 fixed point."""
    return np.round((np.asarray(points, np.float64) - 0.5) * _DRAW_ONE).astype(np.int32)


def _draw_paths(canvas: np.ndarray, paths: list[_Path]) -> None:
    """Draw ``paths`` onto a uint8 canvas; overlapping paths keep the maximum.

    Each segment is a convex quad between its vertices' half-widths, filled
    with the mean level of its two vertices, with round caps and round
    joints at bends; the feathering afterwards hides the steps between
    segments. The geometry of all paths is prepared in one go (this runs
    every frame, so Python overhead matters).
    """
    paths = [p for p in paths if len(p.points) >= 2]
    if not paths:
        return
    pts = np.concatenate([np.asarray(p.points, np.float64) for p in paths])
    radii = np.maximum(np.concatenate([np.asarray(p.radii, np.float64) for p in paths]), 0.0)
    vertex_levels = np.concatenate([np.asarray(p.levels, np.float64) for p in paths])
    direction = np.diff(pts, axis=0)
    length = np.hypot(direction[:, 0], direction[:, 1])
    normal = np.stack([-direction[:, 1], direction[:, 0]], axis=1) / np.maximum(length, 1e-9)[:, None]
    r0, r1 = radii[:-1, None], radii[1:, None]
    quads = _fixed(np.stack([pts[:-1] + normal * r0, pts[1:] + normal * r1, pts[1:] - normal * r1, pts[:-1] - normal * r0], axis=1))
    centers = _fixed(pts).tolist()
    radii_fixed = np.round(radii * _DRAW_ONE).astype(np.int64).tolist()
    levels = np.round(127.5 * (vertex_levels[:-1] + vertex_levels[1:])).astype(np.int64).tolist()
    # Joints need a round fill only where the path bends noticeably (the
    # quads of a nearly straight run already meet); path ends always get one.
    cosine = np.sum(direction[:-1] * direction[1:], axis=1) / np.maximum(length[:-1] * length[1:], 1e-12)
    turns = np.concatenate([[True], cosine < _JOINT_COSINE, [True]])
    ends = np.cumsum([len(p.points) for p in paths])
    turns[ends[:-1] - 1] = True
    turns[ends[:-1]] = True
    turns = turns.tolist()
    single = len(paths) == 1
    scratch = canvas if single else np.zeros_like(canvas)
    begin = 0
    for end in ends.tolist():
        drawn = False
        last = -1
        for i in range(begin, end - 1):
            level = levels[i]
            if level <= 0 or radii_fixed[i] + radii_fixed[i + 1] <= 0:
                continue
            if last != i and radii_fixed[i] > 0:  # round cap where a run starts
                cv2.circle(scratch, centers[i], radii_fixed[i], level, -1, cv2.LINE_AA, _DRAW_SHIFT)
            if length[i] > 1e-6:
                cv2.fillConvexPoly(scratch, quads[i], level, cv2.LINE_AA, _DRAW_SHIFT)
            if turns[i + 1] and radii_fixed[i + 1] > 0:  # round joint at a bend, cap at the end
                cv2.circle(scratch, centers[i + 1], radii_fixed[i + 1], level, -1, cv2.LINE_AA, _DRAW_SHIFT)
            drawn = True
            last = i + 1
        if drawn and not single:
            np.maximum(canvas, scratch, out=canvas)
            scratch[...] = 0
        begin = end


def _blur(mask: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0.05:
        return mask
    return cv2.GaussianBlur(mask, (0, 0), sigma, borderType=cv2.BORDER_REPLICATE)


def _to_unit(canvas: np.ndarray) -> np.ndarray:
    return canvas.astype(np.float32) * np.float32(1.0 / 255.0)


def _smoothstep(x: np.ndarray | float) -> np.ndarray:
    t = np.clip(np.asarray(x, np.float32), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _ramp(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip((values - np.float32(lo)) * np.float32(1.0 / (hi - lo)), 0.0, 1.0)


def _clamp01(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


def _filled(size: tuple[int, int], polygons: list[np.ndarray]) -> np.ndarray:
    """Union of anti-aliased filled polygons (canvas coordinates) as float32 0..1."""
    canvas = np.zeros((size[1], size[0]), np.uint8)
    cv2.fillPoly(canvas, [_fixed(p) for p in polygons], 255, cv2.LINE_AA, _DRAW_SHIFT)
    return _to_unit(canvas)


def _solid_shapes(size: tuple[int, int], polygons: list[np.ndarray], sigma: float) -> np.ndarray:
    """Union of filled polygons, 1 inside and fading outward over about 2 sigma."""
    return np.minimum(_blur(_filled(size, polygons), sigma) * np.float32(2.0), 1.0)


# --------------------------------------------------------------------------
# Contour geometry.
# --------------------------------------------------------------------------

# Left/right mirror pairs of the face mesh (subject's right index -> left index).
_MIRROR_PAIRS: dict[int, int] = {
    # face outline
    93: 323, 132: 361, 58: 288, 172: 397, 136: 365, 150: 379, 149: 378, 176: 400, 148: 377,
    234: 454, 127: 356, 162: 389, 21: 251, 54: 284, 103: 332, 67: 297, 109: 338,
    # ring just inside the outline
    137: 366, 177: 401, 215: 435, 138: 367, 135: 364, 169: 394, 170: 395, 140: 369, 171: 396,
    227: 447, 139: 368, 71: 301, 68: 298, 104: 333, 69: 299, 108: 337,
    # cheeks and temples
    147: 376, 187: 411, 207: 427, 216: 436, 123: 352, 116: 345, 117: 346, 118: 347, 119: 348,
    143: 372, 34: 264, 156: 383, 124: 353, 35: 265, 111: 340,
    # nose
    193: 417, 122: 351, 196: 419, 3: 248, 51: 281, 188: 412, 174: 399, 236: 456, 134: 363, 55: 285,
    # chin and lips
    208: 428, 37: 267, 39: 269, 167: 393, 165: 391,
}
_MIRROR: dict[int, int] = {**_MIRROR_PAIRS, **{v: k for k, v in _MIRROR_PAIRS.items()}}

# A stroke vertex: (landmark a, landmark b, t, half-width x face scale, level)
# at lm[a] + t * (lm[b] - lm[a]). Indices are for the subject's right side
# (mirrored for the left); midline points mirror onto themselves.
_Vertex = tuple[int, int, float, float, float]


@dataclass(frozen=True)
class _Stroke:
    area: str  # ContourSettings field with this area's weight
    tone: str  # "shade" or "light"
    blur: str  # softness class, see _BLUR
    vertices: tuple[_Vertex, ...]
    mirrored: bool = True  # drawn on both sides (False: midline stroke, drawn once)
    needs_mask: bool = False  # only with a segmentation mask (hair may cover it)


_STROKES: tuple[_Stroke, ...] = (
    # Jawline: inside the outline from below the ear to the chin center.
    _Stroke("jawline", "shade", "broad", (
        (93, 137, 0.35, 0.07, 0.35),
        (132, 177, 0.4, 0.08, 0.85),
        (58, 215, 0.4, 0.08, 1.0),
        (172, 138, 0.4, 0.08, 1.0),
        (136, 135, 0.4, 0.075, 0.85),
        (150, 169, 0.4, 0.07, 0.55),
        (149, 170, 0.4, 0.06, 0.3),
        (176, 140, 0.4, 0.05, 0.12),
        (148, 171, 0.4, 0.04, 0.0),
    )),
    # Cheek hollow: under the cheekbone, from near the ear toward the mouth corner.
    _Stroke("cheek_hollow", "shade", "broad", (
        (93, 137, 0.6, 0.085, 0.55),
        (137, 147, 0.45, 0.085, 1.0),
        (147, 187, 0.6, 0.07, 0.85),
        (187, 207, 0.5, 0.05, 0.4),
        (207, 216, 0.1, 0.03, 0.0),
    )),
    # Nose shadow: along the side of the bridge, from the brow head down.
    _Stroke("nose_shadow", "shade", "fine", (
        (55, 193, 0.35, 0.025, 0.0),
        (193, 122, 0.1, 0.03, 0.75),
        (122, 188, 0.35, 0.035, 1.0),
        (196, 174, 0.35, 0.035, 0.85),
        (3, 236, 0.35, 0.035, 0.5),
        (51, 134, 0.35, 0.03, 0.0),
    )),
    # Forehead: along the top of the outline, temple to center.
    _Stroke("forehead", "shade", "broad", (
        (162, 139, 0.35, 0.09, 0.5),
        (21, 71, 0.35, 0.11, 1.0),
        (54, 68, 0.35, 0.12, 1.0),
        (103, 104, 0.3, 0.12, 0.75),
        (67, 69, 0.3, 0.12, 0.45),
        (109, 108, 0.3, 0.12, 0.3),
        (10, 151, 0.3, 0.12, 0.25),
    ), needs_mask=True),
    # Nose bridge highlight: from between the brows to just above the tip.
    _Stroke("nose_bridge", "light", "fine", (
        (168, 8, 0.5, 0.025, 0.0),
        (168, 6, 0.0, 0.03, 0.6),
        (6, 197, 0.0, 0.035, 1.0),
        (197, 195, 0.0, 0.035, 1.0),
        (195, 5, 0.0, 0.035, 0.85),
        (5, 4, 0.0, 0.03, 0.5),
        (4, 5, 0.3, 0.02, 0.0),
    ), mirrored=False),
    # Cheekbone highlight: a "C" from the top of the cheekbone up to the temple.
    _Stroke("cheekbone", "light", "medium", (
        (118, 117, 0.0, 0.04, 0.0),
        (117, 118, 0.3, 0.06, 0.7),
        (116, 117, 0.2, 0.065, 1.0),
        (34, 116, 0.35, 0.06, 0.9),
        (143, 34, 0.5, 0.05, 0.55),
        (156, 139, 0.4, 0.04, 0.0),
    )),
    # Chin: a short capsule across the center of the chin.
    _Stroke("chin", "light", "medium", (
        (199, 208, 0.4, 0.08, 1.0),
        (199, 428, 0.4, 0.08, 1.0),
    ), mirrored=False),
    # Cupid's bow: a thin line just above the peaks of the upper lip.
    _Stroke("cupids_bow", "light", "fine", (
        (39, 165, 0.3, 0.015, 0.0),
        (37, 167, 0.35, 0.02, 1.0),
        (0, 164, 0.35, 0.02, 0.6),
        (267, 393, 0.35, 0.02, 1.0),
        (269, 391, 0.3, 0.015, 0.0),
    ), mirrored=False),
)

# Feather sigma per softness class: base + softness slider x range.
_BLUR: dict[str, tuple[float, float]] = {
    "broad": (0.04, 0.09),
    "medium": (0.035, 0.07),
    "fine": (0.015, 0.035),
}
# Each straight stroke segment between two vertices is split into this many
# pieces per softness class (the feather hides the steps in width and level
# between them; broad strokes are blurred over more than a segment anyway).
_SUBDIVIDE: dict[str, int] = {"broad": 1, "medium": 1, "fine": 2}


@dataclass(frozen=True)
class _StrokeSide:
    """One stroke on one side, precomputed for fast evaluation."""

    stroke: _Stroke
    side: regions.Side
    start: np.ndarray  # (N,) landmark indices
    end: np.ndarray  # (N,) landmark indices
    t: np.ndarray  # (N, 1) interpolation from start to end
    dense: np.ndarray  # (M, N) linear map from the N vertices to the M drawn points
    radii: np.ndarray  # (M,) half-widths x face scale
    levels: np.ndarray  # (M,)

    @classmethod
    def build(cls, stroke: _Stroke, side: regions.Side) -> _StrokeSide:
        rows = np.array(stroke.vertices, np.float64)
        index = (lambda i: _MIRROR.get(i, i)) if side == "left" else (lambda i: i)
        n, pieces = len(rows), _SUBDIVIDE[stroke.blur]
        dense = np.zeros(((n - 1) * pieces + 1, n), np.float32)
        for k in range(n - 1):
            for j in range(pieces):
                dense[k * pieces + j, k] = 1.0 - j / pieces
                dense[k * pieces + j, k + 1] = j / pieces
        dense[-1, -1] = 1.0
        return cls(
            stroke,
            side,
            np.array([index(int(i)) for i in rows[:, 0]]),
            np.array([index(int(i)) for i in rows[:, 1]]),
            rows[:, 2:3].astype(np.float32),
            dense,
            (dense @ rows[:, 3].astype(np.float32)).astype(np.float32),
            (dense @ rows[:, 4].astype(np.float32)).astype(np.float32),
        )

    def points(self, landmarks: np.ndarray) -> np.ndarray:
        start = landmarks[self.start]
        return self.dense @ (start + self.t * (landmarks[self.end] - start))


_STROKE_SIDES: tuple[_StrokeSide, ...] = tuple(
    _StrokeSide.build(stroke, side)
    for stroke in _STROKES
    for side in (regions.SIDES if stroke.mirrored else ("right",))
)

# Coarse grid: face scale in grid pixels (never finer than the frame), and
# the blur (grid pixels) that smooths frame data sampled onto it.
_GRID_SCALE = 32.0
_DOWNSAMPLE_BLUR = 0.8
# Context around the face outline for the ROI.
_ROI_MARGIN = 0.05
# Allowed area: the face outline shrunk by this much and feathered, so the
# shading fades out right at the jaw instead of passing it.
_OVAL_INSET = 0.035
_OVAL_FEATHER = 0.025
# Face-skin confidence -> weight; a mask covering less than this share of
# the face outline is ignored (stale, or the model missed the face).
_SKIN_RAMP = (0.25, 0.75)
_HAIR_RAMP = (0.3, 0.8)
_MIN_SKIN_COVERAGE = 0.35
_MASK_FEATHER = 0.03
# Turned head: a side's strokes narrow with how wide that cheek looks
# (cheek apple to outline, relative to a frontal face; as the blush does).
_FRONTAL_CHEEK_WIDTH = 0.44
_SQUEEZE_RANGE = (0.55, 1.0)
_CHEEK_CONTOUR: dict[regions.Side, list[int]] = {"right": [234, 93], "left": [454, 323]}

# Protected features. On the grid: outlines grown by the larger of the
# value (x face scale) and _GUARD_GRID_PX grid pixels, so the bilinear
# upsampling cannot carry color onto the feature itself.
_EYE_GUARD = 0.04
_BROW_GUARD = 0.5  # brow height grows by this fraction
_NOSTRIL_GUARD = 0.02
_GUARD_GRID_PX = 2.0
_GUARD_FEATHER = 0.015
# Lips (full resolution): outline growth and feather.
_LIP_GUARD = 0.012
_LIP_FEATHER = 0.012
_RIGHT_NOSTRIL = [75, 79, 237, 239, 238, 20, 60, 99, 240]
_LEFT_NOSTRIL = [305, 309, 457, 459, 458, 250, 290, 328, 460]


@dataclass(frozen=True)
class _Grid:
    """A coarse grid over ``roi``; frame point p maps to ``(p - offset) * ratio``."""

    roi: regions.Roi
    width: int
    height: int

    @classmethod
    def over(cls, roi: regions.Roi, scale: float) -> _Grid:
        factor = max(1.0, scale / _GRID_SCALE)
        return cls(roi, max(1, round(roi.w / factor)), max(1, round(roi.h / factor)))

    @property
    def size(self) -> tuple[int, int]:
        return self.width, self.height

    @property
    def ratio(self) -> np.ndarray:
        return np.array([self.width / self.roi.w, self.height / self.roi.h], np.float32)

    @property
    def area(self) -> regions.Roi:
        return regions.Roi(0, 0, self.width, self.height)

    def points(self, frame_points: np.ndarray) -> np.ndarray:
        return (np.asarray(frame_points, np.float32) - np.asarray(self.roi.offset, np.float32)) * self.ratio

    def downsample(self, image: np.ndarray) -> np.ndarray:
        """A ROI-sized image -> grid resolution.

        Bilinear samples, then a light blur on the grid: as smooth as an area
        average for these purposes (local skin brightness, smooth
        segmentation masks) at a fraction of the cost.
        """
        if (self.width, self.height) == (self.roi.w, self.roi.h):
            return np.asarray(image, np.float32)
        small = cv2.resize(np.asarray(image, np.float32), self.size, interpolation=cv2.INTER_LINEAR)
        return _blur(small, _DOWNSAMPLE_BLUR)

    def upsample(self, image: np.ndarray) -> np.ndarray:
        """A grid image -> ROI resolution (bilinear; exact zeros stay zero)."""
        if (self.width, self.height) == (self.roi.w, self.roi.h):
            return image
        return cv2.resize(image, (self.roi.w, self.roi.h), interpolation=cv2.INTER_LINEAR)


def _cheek_squeeze(landmarks: np.ndarray, side: regions.Side, scale: float) -> float:
    """How wide this side of the face looks relative to a frontal face (< 1: turned away)."""
    across = regions.eye_center(landmarks, "left") - regions.eye_center(landmarks, "right")
    across = across / max(float(np.linalg.norm(across)), 1e-6)
    outward = across if side == "left" else -across
    contour = landmarks[_CHEEK_CONTOUR[side]].astype(np.float32).mean(axis=0)
    width = float(np.dot(contour - regions.cheek_center(landmarks, side), outward))
    return float(np.clip(width / (_FRONTAL_CHEEK_WIDTH * scale), *_SQUEEZE_RANGE))


def _segmentation_weight(grid: _Grid, ctx: FrameContext, oval: np.ndarray) -> np.ndarray | None:
    """Skin weight on the grid from the segmentation masks, or None when unavailable/unreliable."""
    if ctx.face_skin_mask is not None:
        weight = _ramp(grid.downsample(ctx.face_skin_mask[grid.roi.slices]), *_SKIN_RAMP)
        coverage = float(np.sum(weight * oval)) / max(float(np.sum(oval)), 1e-6)
        if coverage >= _MIN_SKIN_COVERAGE:
            return weight
    if ctx.hair_mask is not None:
        return 1.0 - _ramp(grid.downsample(ctx.hair_mask[grid.roi.slices]), *_HAIR_RAMP)
    return None


def _sample(mask: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Bilinear samples of ``mask`` at continuous ``points`` (N, 2) (edges replicated)."""
    coords = (np.asarray(points, np.float32) - np.float32(0.5)).reshape(1, -1, 2)
    values = cv2.remap(mask, coords[..., 0], coords[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return values.reshape(-1)


@register_effect
class ContourEffect(Effect):
    """Shading and highlight that sculpt the face (see the module docstring)."""

    name = "contour"
    order = 25

    def apply(self, frame: np.ndarray, ctx: FrameContext, settings: ContourSettings) -> np.ndarray:
        amounts = {
            "shade": _clamp01(settings.shading) * ctx.intensity,
            "light": _clamp01(settings.highlight) * ctx.intensity,
        }
        active = [s for s in _STROKE_SIDES if amounts[s.stroke.tone] * _clamp01(getattr(settings, s.stroke.area)) > 0.0]
        if not active or ctx.landmarks is None:
            return frame
        landmarks = ctx.landmarks
        scale = regions.face_scale(landmarks)
        if not math.isfinite(scale) or scale < _MIN_FACE_SCALE:
            return frame
        roi = regions.roi_around(landmarks[regions.FACE_OVAL], frame.shape, pad=0.0, margin_px=_ROI_MARGIN * scale)
        if roi is None or min(roi.w, roi.h) < 0.25 * scale:
            return frame
        grid = _Grid.over(roi, scale)
        weights = _contour_weights(grid, landmarks, scale, active, settings, amounts, ctx)
        if weights is None:
            return frame
        view = frame[roi.slices]
        shade, light = weights
        gain = _tone_gain(
            grid.downsample(view) if light is not None else view[: grid.height, : grid.width],
            shade,
            light,
            _shading_filter(hex_to_bgr(settings.shading_color)),
            _Light.of(hex_to_bgr(settings.highlight_color)),
        )
        gain = grid.upsample(gain)
        _protect_lips(landmarks, roi, scale, gain)
        _apply_gain(view, gain)
        return frame


def _contour_weights(
    grid: _Grid,
    landmarks: np.ndarray,
    scale: float,
    active: list[_StrokeSide],
    settings: ContourSettings,
    amounts: dict[str, float],
    ctx: FrameContext,
) -> tuple[np.ndarray | None, np.ndarray | None] | None:
    """(shade, light) weights on the grid (None for a tone with nothing drawn)."""
    unit = scale * float(grid.ratio.mean())  # face scale in grid pixels
    points = grid.points(landmarks)
    inset = regions.expand_polygon(points[regions.FACE_OVAL], -_OVAL_INSET * unit)
    oval = _blur(_filled(grid.size, [inset]), _OVAL_FEATHER * unit)
    skin = _segmentation_weight(grid, ctx, oval)
    if skin is not None:
        skin = _blur(skin, _MASK_FEATHER * unit)
    squeeze = {side: _cheek_squeeze(landmarks, side, scale) for side in regions.SIDES}
    softness = _clamp01(settings.softness)
    sigmas = {name: (base + span * softness) * unit for name, (base, span) in _BLUR.items()}
    paths: dict[tuple[str, str], list[_Path]] = {}
    for item in active:
        stroke = item.stroke
        if stroke.needs_mask and skin is None:
            continue  # cannot tell skin from hair: leave it out
        path_points = item.points(points)
        levels = item.levels * np.float32(_clamp01(getattr(settings, stroke.area)))
        if stroke.needs_mask:
            # Drop the parts of the stroke that lie on hair instead of letting
            # their feather slide onto the skin below the bangs.
            levels = levels * _sample(skin, path_points)
        width = unit * (squeeze[item.side] if stroke.mirrored else 1.0)
        paths.setdefault((stroke.tone, stroke.blur), []).append(_Path(path_points, item.radii * np.float32(width), levels))
    if not paths:
        return None
    allowed = oval if skin is None else oval * skin
    allowed *= np.float32(1.0) - _feature_guard(grid, points, unit)
    tones: dict[str, np.ndarray] = {}
    for (tone, blur_class), group in paths.items():
        canvas = np.zeros((grid.height, grid.width), np.uint8)
        _draw_paths(canvas, group)
        soft = _blur(_to_unit(canvas), sigmas[blur_class])
        tones[tone] = soft if tone not in tones else np.maximum(tones[tone], soft)
    out: list[np.ndarray | None] = []
    for tone in ("shade", "light"):
        weight = tones.get(tone)
        if weight is not None:
            weight = weight * allowed * np.float32(amounts[tone])
            if not float(weight.max()) > 0.0:
                weight = None
        out.append(weight)
    if out[0] is None and out[1] is None:
        return None
    return out[0], out[1]


def _feature_guard(grid: _Grid, points: np.ndarray, unit: float) -> np.ndarray:
    """Eyes, brows and nostrils (grid coordinates) as a 0..1 keep-out mask."""
    polygons = []
    for side in regions.SIDES:
        s = regions.side_indices(side)
        polygons.append(regions.expand_polygon(points[s.eye], max(_EYE_GUARD * unit, _GUARD_GRID_PX)))
        upper, lower = points[s.brow_upper], points[s.brow_lower]
        center, half = (upper + lower) / 2, (upper - lower) / 2 * (1.0 + _BROW_GUARD)
        polygons.append(regions.expand_polygon(np.vstack([center + half, (center - half)[::-1]]), _GUARD_GRID_PX))
        nostril = points[_LEFT_NOSTRIL if side == "left" else _RIGHT_NOSTRIL]
        hull = cv2.convexHull(nostril.astype(np.float32)).reshape(-1, 2)
        polygons.append(regions.expand_polygon(hull, max(_NOSTRIL_GUARD * unit, _GUARD_GRID_PX)))
    return _solid_shapes(grid.size, polygons, max(_GUARD_FEATHER * unit, 0.5))


def _protect_lips(landmarks: np.ndarray, roi: regions.Roi, scale: float, gain: np.ndarray) -> None:
    """Zero the gain (in place, ROI-sized) on the lips, at full resolution."""
    feather_px = _LIP_FEATHER * scale
    outline = regions.expand_polygon(landmarks[regions.LIPS_OUTER], _LIP_GUARD * scale)
    area = regions.roi_around(outline, (roi.y1, roi.x1), pad=0.0, margin_px=3.0 * feather_px + 2.0)
    if area is None:
        return
    area = regions.Roi(max(area.x0, roi.x0), max(area.y0, roi.y0), area.x1, area.y1)
    if area.w <= 0 or area.h <= 0:
        return
    keep = np.float32(1.0) - _solid_shapes((area.w, area.h), [outline - np.asarray(area.offset, np.float32)], feather_px)
    keep = cv2.merge([keep, keep, keep])
    local = (slice(area.y0 - roi.y0, area.y1 - roi.y0), slice(area.x0 - roi.x0, area.x1 - roi.x0))
    gain[local] = cv2.multiply(gain[local], keep)


# --------------------------------------------------------------------------
# Tear bags.
# --------------------------------------------------------------------------

# Band under the lower lashes: center and half-height below the lash line.
_BAG_CENTER = 0.03
_BAG_HALF = 0.024
# Shadow line along the lower edge of the bag.
_SHADOW_CENTER = 0.066
_SHADOW_HALF = 0.006
# Feather sigmas: base + softness slider x range.
_BAG_FEATHER = (0.006, 0.014)
_SHADOW_FEATHER = (0.008, 0.016)
# Along the lid (0 = inner corner, 1 = outer corner): the highlight fades in
# over the inner third, the shadow starts later; both fade out over this
# share of their length.
_BAG_FADE_IN = (0.02, 0.32)
_SHADOW_FADE_IN = (0.15, 0.45)
_FADE_OUT = 0.3
# Highlight (lift gain and pearl color) and shadow color; the shadow's
# strength relative to the highlight.
_BAG_LIGHT_GAIN = 0.9
_BAG_LIGHT: Bgr = hex_to_bgr("#FFF2EA")
_BAG_SHADOW: Bgr = hex_to_bgr("#9E7A68")
_SHADOW_GAIN = 0.6
# The eye opening is cut out grown by this much (crisp), so the eyeball stays clean.
_EYE_CUT = 0.012
# Smiling: mouth width / face scale from the first to the second value maps
# to 0..1 (smoothed over frames); at 1 the bag is this much fuller.
_SMILE_RAMP = (0.98, 1.12)
_SMILE_SMOOTHING = 0.35
_SMILE_SIZE = 0.3
_SMILE_LIGHT = 0.25
_SMILE_SHADOW = 0.4
_LID_SAMPLES = 20
_MIN_EYE_WIDTH_PX = 6.0


@register_effect
class TearbagEffect(Effect):
    """Tear-bag (涙袋) highlight and shadow under the lower lashes."""

    name = "tearbag"
    order = 45

    def __init__(self) -> None:
        self._smile: float | None = None

    def reset(self) -> None:
        self._smile = None

    def apply(self, frame: np.ndarray, ctx: FrameContext, settings: TearbagSettings) -> np.ndarray:
        amount = _clamp01(settings.strength) * ctx.intensity
        reach = _clamp01(settings.width)
        if amount <= 0.0 or reach <= 0.0 or ctx.landmarks is None:
            return frame
        landmarks = ctx.landmarks
        scale = regions.face_scale(landmarks)
        if not math.isfinite(scale) or scale < _MIN_FACE_SCALE:
            return frame
        smile = self._update_smile(landmarks, scale)
        light = _Light.of(_BAG_LIGHT, _BAG_LIGHT_GAIN)
        shade_filter = _shading_filter(_BAG_SHADOW)
        for side in regions.SIDES:
            _tear_bag(frame, landmarks, side, scale, settings, amount, reach, smile, light, shade_filter)
        return frame

    def _update_smile(self, landmarks: np.ndarray, scale: float) -> float:
        width = float(np.linalg.norm(landmarks[regions.MOUTH_LEFT_CORNER] - landmarks[regions.MOUTH_RIGHT_CORNER]))
        target = float(_smoothstep((width / scale - _SMILE_RAMP[0]) / (_SMILE_RAMP[1] - _SMILE_RAMP[0])))
        if self._smile is None or not math.isfinite(self._smile):
            self._smile = target
        else:
            self._smile += _SMILE_SMOOTHING * (target - self._smile)
        return self._smile


def _face_down(landmarks: np.ndarray) -> np.ndarray:
    """Unit vector perpendicular to the eye-to-eye line, pointing toward the chin."""
    across = regions.eye_center(landmarks, "left") - regions.eye_center(landmarks, "right")
    across = across / max(float(np.linalg.norm(across)), 1e-6)
    return np.array([-across[1], across[0]], np.float32)


def _lid_profile(u: np.ndarray, fade_in: tuple[float, float], reach: float) -> np.ndarray:
    """Level along the lid: fades in over ``fade_in`` and out toward ``reach``."""
    rise = _smoothstep((u - fade_in[0]) / (fade_in[1] - fade_in[0]))
    fall = _smoothstep((reach - u) / max(_FADE_OUT * reach, 1e-3))
    return rise * fall


def _tear_bag(
    frame: np.ndarray,
    landmarks: np.ndarray,
    side: regions.Side,
    scale: float,
    settings: TearbagSettings,
    amount: float,
    reach: float,
    smile: float,
    light: _Light,
    shade_filter: np.ndarray,
) -> None:
    s = regions.side_indices(side)
    lash = landmarks[s.eye_lower].astype(np.float32)
    inner, outer = lash[0], lash[-1]
    eye_width = float(np.linalg.norm(outer - inner))
    if not math.isfinite(eye_width) or eye_width < _MIN_EYE_WIDTH_PX:
        return
    # "Down": perpendicular to the eye line, on the chin side of the face
    # (robust while blinking, when the lids meet).
    along = (outer - inner) / eye_width
    down = np.array([-along[1], along[0]], np.float32)
    if float(np.dot(down, _face_down(landmarks))) < 0:
        down = -down
    lid = regions.resample_polyline(lash, _LID_SAMPLES)
    u = np.linspace(0.0, 1.0, _LID_SAMPLES, dtype=np.float32)
    grow = 1.0 + _SMILE_SIZE * smile
    softness = _clamp01(settings.softness)
    bag_sigma = (_BAG_FEATHER[0] + _BAG_FEATHER[1] * softness) * scale
    shadow_sigma = (_SHADOW_FEATHER[0] + _SHADOW_FEATHER[1] * softness) * scale
    bag = _Path(
        lid + down * np.float32(_BAG_CENTER * grow * scale),
        np.full(_LID_SAMPLES, _BAG_HALF * grow * scale, np.float32),
        _lid_profile(u, _BAG_FADE_IN, reach),
    )
    shadow_level = min(amount * _clamp01(settings.shadow) * _SHADOW_GAIN * (1.0 + _SMILE_SHADOW * smile), 1.0)
    shadow = _Path(
        lid + down * np.float32(_SHADOW_CENTER * grow * scale),
        np.full(_LID_SAMPLES, _SHADOW_HALF * scale, np.float32),
        _lid_profile(u, _SHADOW_FADE_IN, reach),
    )
    margin = (_SHADOW_CENTER * grow + _SHADOW_HALF) * scale + 3.5 * max(bag_sigma, shadow_sigma) + 2.0
    roi = regions.roi_around(np.vstack([lid, shadow.points]), frame.shape, pad=0.0, margin_px=margin)
    if roi is None:
        return
    offset = np.asarray(roi.offset, np.float32)
    keep = np.float32(1.0) - regions.eye_opening_mask(landmarks, side, roi, expand_px=_EYE_CUT * scale)
    light_level = np.float32(min(amount * (1.0 + _SMILE_LIGHT * smile), 1.0))
    light_weight = _band(roi, bag, offset, bag_sigma) * keep * light_level
    shade_weight = None
    if shadow_level > 0.0:
        shade_weight = _band(roi, shadow, offset, shadow_sigma) * keep * np.float32(shadow_level)
    view = frame[roi.slices]
    _apply_gain(view, _tone_gain(view, shade_weight, light_weight, shade_filter, light))


def _band(roi: regions.Roi, path: _Path, offset: np.ndarray, sigma: float) -> np.ndarray:
    """The feathered stroke of ``path`` (frame coordinates) as a ROI-sized float32 mask."""
    canvas = np.zeros((roi.h, roi.w), np.uint8)
    _draw_paths(canvas, [_Path(path.points - offset, path.radii, path.levels)])
    return _blur(_to_unit(canvas), sigma)
