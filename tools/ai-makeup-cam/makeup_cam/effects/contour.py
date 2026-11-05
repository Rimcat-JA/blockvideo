"""Contour (shading + highlight) and tear bags (涙袋).

Both effects paint soft strokes anchored on face-mesh vertices: every stroke
is a polyline whose vertices are interpolated between two landmarks, so it
follows head roll, yaw and expressions like the skin under it, and every
width is a fraction of the face scale (inter-ocular distance).

ContourEffect ("contour", order 25, right after the skin retouch)
    Shading darkens the jawline (a broad band hugging the lower face outline,
    from below the ear toward the chin, leaving the chin center alone), the
    cheek hollow (a broad comma under the cheekbone, from near the ear toward
    the mouth corner, ending well before it), the side walls of the nose (from
    the brow heads down, gone by the middle of the nose) and the skin along
    the real hairline. Highlight lightens the nose bridge with a broad glow
    above it (the T zone), the top of the cheekbones (a "C" around the outer
    eye toward the temple), the center of the chin and the cupid's bow. Each
    area has its own weight slider; ``shading`` and ``highlight`` scale all of
    them. On a turned head the strokes of the far side narrow with that side
    of the face.

    What makes this read as shape rather than as painted stripes: every stroke
    is about as wide as its feather, the weight has a long tail (a Gaussian
    plus a wider Gaussian, like blended powder), and neighboring strokes merge
    into one gradient toward the sides of the face.

    Everything except the final per-pixel blend is computed on a coarse grid
    (face scale about ``_GRID_SCALE`` grid pixels): the masks are smooth, and
    so are the per-pixel color factors derived from them. Only
    ``out = b + b * gain`` runs at full resolution, so texture and pores stay
    intact. The allowed area is the face outline times the face-skin
    segmentation mask (hair, bangs or a hand in front of the face are left
    alone). A mask that covers little of the face means the mask and the
    tracker disagree, so the whole effect fades out with that coverage instead
    of guessing. Without a skin mask the outline is inset, so nothing spills
    past the jaw. The eyes and brows are kept out on the grid with outlines
    grown by more than a grid pixel; the lips and nostrils, with the cupid's bow
    between them, on a finer mask.

    The hairline shadow does not use the mesh at all, because the mesh ends in
    the middle of a high forehead and continues under bangs: it is a blur of
    the "not skin" area, kept on the skin. So it hugs the actual hairline or
    bangs, and it is left out without a segmentation mask.

TearbagEffect ("tearbag", order 45, after the eyeshadow)
    A soft highlight band right under the lower lash line, brightest under the
    iris and fading toward both corners (``width`` says how far toward the
    outer one), plus a faint soft shadow along the lower edge of the bag. Both
    hang off the lower lash landmarks, so they follow blinks, gaze and smiles.
    When the person smiles (mouth widening, smoothed over a few frames) the bag
    is drawn fuller and brighter, as a real tear bag bulges; eyes that stay
    narrowed bunch the lid by themselves, so the shadow line fades with the
    eye opening (smoothed, so a blink does not pulse it).

    The eye opening is cut out with a grown outline that fades outward. The
    landmarks still trail the real eye a little when the head moves (up to about
    3 % of the face scale in a fast nod; the tracker removes most of the lag),
    and a band hanging off them would light up the white of the eye. The lid's
    own step between frames predicts that lag well, so while the lid moves
    down the band, its shadow and the lower edge of the cut all move ahead of
    the landmarks by that much (moving up, the band is merely a little low on
    the skin, so nothing is needed).

    The color comes from the skin under the bag, so the bag is that skin lit
    up, never a pale crescent on top of it. That skin is measured outside the
    cut, ignoring whatever is much darker or lighter than the typical pixel
    there (lashes in a blink, the eye white), and smoothed over frames.

Color (shared, see ``_tone_gain``)
    Everything is a smooth per-channel gain computed from the local skin color
    (the frame downscaled to the grid; the bag uses one measured color), so the
    texture survives and the result depends only on the skin the stroke lies on.
    Shading multiplies the skin by a filter derived from the shading color
    *relative to a reference skin tone*: on that reference the shaded skin
    takes exactly the chosen color, and every other skin gets the same
    relative darkening. Skin more saturated than the reference is deepened
    further along its own chroma, and the filter may never cool the skin, so
    a shadow is the skin's own darker tone: a deeper brown on deep skin, a warm
    tan on pale skin, never a grey or ashy veil. A channel that is clipped (often
    red on overexposed light skin) loses less, for the same reason.
    Highlight lifts all channels of the skin by the same factor, which keeps
    its hue and saturation (a lit deep skin glows instead of turning chalky).
    The lift is softly limited to a share of the brightest channel's headroom,
    so it cannot clip. What the headroom denies (very light skin) partly goes
    into closing the gap between the color channels, by a bounded share: the
    pale sheen a pearl highlighter gives light skin, which never reaches flat
    white. On such skin the shape comes mostly from the surrounding shading.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .. import blend, regions
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
# Sub-pixels per pixel along each axis for polygon coverage.
_COVERAGE_SAMPLES = 4
# Stroke joints bending by more than ~15 degrees get a round fill.
_JOINT_COSINE = 0.966
_GROW_KERNEL = np.ones((3, 3), np.uint8)

# --------------------------------------------------------------------------
# Color.
# --------------------------------------------------------------------------

# Skin tone the shading color is defined on (a light-medium skin, BGR). On
# this skin, full shading turns the skin into the shading color.
_REFERENCE_SKIN: Bgr = hex_to_bgr("#E6C3AD")
_REFERENCE_CHROMA = np.asarray(_REFERENCE_SKIN, np.float32) / max(_REFERENCE_SKIN)
# Luma weights in BGR order.
_LUMA_BGR = np.array([0.114, 0.587, 0.299], np.float32)
# The shading filter's luma never drops below this, and its per-channel tint
# (filter / its luma) stays within this range.
_MIN_SHADE_FACTOR = 0.3
_TINT_RANGE = (0.85, 1.2)
# A channel near the top of the range is clipped, its real value unknown: over
# this ramp it keeps up to this share of what shading would take from it. Lowering
# it like the others would turn blown-out skin grey instead of a warm tan.
_CLIPPED_RAMP = (0.9, 1.0)
_CLIPPED_KEEP = 0.6
# Below this the local color is too dark to divide by.
_MIN_BASE = 0.05


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
    """How a highlight lifts the skin (see ``_tone_gain``)."""

    tint: np.ndarray  # highlight color with its brightest channel at 1 (BGR)
    lift: float  # wanted relative lift of the brightest channel at full weight
    room: float  # share of the headroom the lift may use (soft limit)
    pearl: float  # share of the channel gap closed at full weight, where no headroom is left

    @classmethod
    def of(cls, color_bgr: Bgr, lift: float, room: float, pearl: float) -> _Light:
        color = np.clip(np.asarray(color_bgr, np.float32), 0.0, 1.0)
        return cls((color / max(float(color.max()), 1e-3)).astype(np.float32), lift, room, pearl)


def _tone_gain(
    local: np.ndarray,
    shade: np.ndarray | None,
    light: np.ndarray | None,
    shade_filter: np.ndarray,
    highlight: _Light,
) -> np.ndarray:
    """Per-pixel, per-channel ``gain`` so that ``b * (1 + gain)`` applies shading, then highlight.

    ``local`` is the smooth local skin color, as an image of the weights' size
    or as one color (a 1x1x3 image); ``shade`` / ``light`` are HxW weights in
    [0, 1]. Where both weights are 0 the gain is exactly 0.

    Shading multiplies by ``1 - w * (1 - filter * deepen)``, with ``deepen``
    the square root of the skin's chroma (color / brightest channel) relative
    to the reference skin's, limited to 1 (skin paler than the reference is
    not cooled); clipped channels lose less. Highlight wants ``dM = w * lift * M`` on the brightest
    channel ``M`` and gets the soft minimum of that and ``room * (1 - M)``.
    The rest of the wanted lift (0..1, ``denied``) closes ``w * pearl *
    denied`` of the gap between each channel and the lifted brightest channel
    (times the tint), so the channels stay apart.
    """
    brightest = _brightest(local)
    base = np.maximum(brightest, np.float32(_MIN_BASE))
    if shade is not None:
        chroma = local / cv2.merge([base, base, base])
        deepen = cv2.sqrt(np.minimum(chroma / _REFERENCE_CHROMA, np.float32(1.0)))
        deepen[..., 1] = cv2.sqrt(deepen[..., 1])  # green moves less: dark skin should not turn red
        loss = np.float32(1.0) - shade_filter * deepen
        loss *= np.float32(1.0) - _ramp(local, *_CLIPPED_RAMP) * np.float32(_CLIPPED_KEEP)
        factor = np.float32(1.0) - shade[..., None] * loss
    else:
        factor = None
    if light is not None:
        want = light * np.float32(highlight.lift) * base
        room = (np.float32(1.0) - np.minimum(brightest, np.float32(1.0))) * np.float32(highlight.room)
        total = np.maximum(want + room, np.float32(1e-6))
        lift = want * room / total
        lit = (lift / base)[..., None] * highlight.tint + np.float32(1.0)
        pearl = (light * want / total * np.float32(highlight.pearl))[..., None]
        if float(pearl.max()) > 0.0:
            target = (brightest + lift)[..., None] * highlight.tint
            lit += pearl * (target / np.maximum(local, np.float32(_MIN_BASE)) - lit)
        factor = lit if factor is None else factor * lit
    if factor is None:
        return np.zeros(local.shape, np.float32)
    factor -= np.float32(1.0)
    return factor


def _brightest(image: np.ndarray) -> np.ndarray:
    """Per-pixel maximum over the color channels (cv2 is much faster than ndarray.max here)."""
    b, g, r = cv2.split(image)
    return cv2.max(cv2.max(b, g), r)


def _apply_gain(view: np.ndarray, gain: np.ndarray) -> None:
    """``view = min(view * (1 + gain), 1)`` in place (bit-exact where the gain is 0)."""
    cv2.threshold(cv2.add(view, cv2.multiply(view, gain)), 1.0, 1.0, cv2.THRESH_TRUNC, dst=view)


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
    """Union of filled polygons (canvas coordinates) as float32 0..1 coverage.

    Supersampled: under sub-pixel motion cv2's LINE_AA fill flips whole runs of
    pixels along a nearly straight edge, which shimmers. Each polygon is filled
    on its own (cv2 would leave the overlap of two polygons empty).
    """
    canvas = np.zeros((size[1] * _COVERAGE_SAMPLES, size[0] * _COVERAGE_SAMPLES), np.uint8)
    for polygon in polygons:
        cv2.fillPoly(canvas, [_fixed(np.asarray(polygon, np.float64) * _COVERAGE_SAMPLES)], 255, cv2.LINE_8, _DRAW_SHIFT)
    return _to_unit(cv2.resize(canvas, size, interpolation=cv2.INTER_AREA))


def _solid_shapes(size: tuple[int, int], polygons: list[np.ndarray], sigma: float, grow: bool = False) -> np.ndarray:
    """Union of filled polygons, exactly 1 inside and fading outward over about 2 sigma.

    ``grow`` adds a true margin of one pixel all around (``expand_polygon``
    grows thin shapes like the brows very little across their length).
    """
    filled = _filled(size, polygons)
    if grow:
        filled = cv2.dilate(filled, _GROW_KERNEL)
    return np.maximum(filled, np.minimum(_blur(filled, sigma) * np.float32(2.0), 1.0))


def _face_down(landmarks: np.ndarray) -> np.ndarray:
    """Unit vector perpendicular to the eye-to-eye line, pointing toward the chin."""
    across = regions.eye_center(landmarks, "left") - regions.eye_center(landmarks, "right")
    across = across / max(float(np.linalg.norm(across)), 1e-6)
    return np.array([-across[1], across[0]], np.float32)


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
    227: 447, 139: 368, 213: 433, 192: 416,
    # cheeks and temples
    147: 376, 187: 411, 207: 427, 216: 436, 123: 352, 116: 345, 117: 346, 118: 347, 119: 348,
    143: 372, 34: 264, 156: 383, 124: 353, 35: 265, 111: 340, 50: 280, 101: 330, 113: 342, 226: 446,
    # nose
    193: 417, 122: 351, 196: 419, 3: 248, 51: 281, 188: 412, 174: 399, 236: 456, 134: 363, 55: 285,
    45: 275, 189: 413, 245: 465,
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


_STROKES: tuple[_Stroke, ...] = (
    # Jawline: hugging the outline from below the ear toward the chin. The
    # outer half of its feather is cut by the outline, so it is darkest at the
    # edge of the face and fades inward.
    _Stroke("jawline", "shade", "broad", (
        (93, 147, 0.2, 0.06, 0.3),
        (132, 213, 0.25, 0.085, 0.8),
        (58, 138, 0.25, 0.1, 1.0),
        (172, 135, 0.25, 0.1, 1.0),
        (136, 169, 0.25, 0.09, 0.85),
        (150, 170, 0.25, 0.08, 0.55),
        (149, 140, 0.25, 0.065, 0.28),
        (176, 171, 0.25, 0.05, 0.08),
        (148, 171, 0.25, 0.04, 0.0),
    )),
    # Cheek hollow: under the cheekbone, from near the ear toward the mouth
    # corner, ending well before it (a shadow beside the mouth reads as dirt).
    _Stroke("cheek_hollow", "shade", "broad", (
        (137, 147, 0.3, 0.09, 0.65),
        (147, 187, 0.4, 0.1, 1.0),
        (187, 207, 0.25, 0.09, 0.8),
        (187, 207, 0.8, 0.06, 0.35),
        (207, 216, 0.3, 0.04, 0.0),
    )),
    # Nose shadow: on the side wall of the nose, deepest at its root, gone by
    # the middle of the nose.
    _Stroke("nose_shadow", "shade", "nose", (
        (55, 193, 0.5, 0.025, 0.0),
        (193, 189, 0.3, 0.035, 0.65),
        (122, 245, 0.35, 0.04, 0.8),
        (196, 188, 0.5, 0.04, 0.7),
        (3, 174, 0.5, 0.04, 0.4),
        (51, 236, 0.4, 0.03, 0.0),
    )),
    # Nose bridge highlight: from between the brows to just above the tip.
    _Stroke("nose_bridge", "light", "fine", (
        (8, 168, 0.5, 0.05, 0.3),
        (168, 6, 0.5, 0.055, 0.8),
        (6, 197, 0.5, 0.06, 1.0),
        (197, 195, 0.5, 0.06, 1.0),
        (195, 5, 0.5, 0.055, 0.8),
        (5, 4, 0.5, 0.045, 0.4),
        (4, 1, 0.5, 0.03, 0.0),
    ), mirrored=False),
    # T zone: the bridge highlight spreading out as a glow on the forehead.
    _Stroke("nose_bridge", "light", "medium", (
        (8, 9, 0.5, 0.09, 0.4),
        (9, 151, 0.5, 0.14, 0.45),
        (151, 10, 0.3, 0.16, 0.0),
    ), mirrored=False),
    # Cheekbone highlight: a "C" from the top of the cheekbone toward the temple.
    _Stroke("cheekbone", "light", "medium", (
        (119, 118, 0.5, 0.045, 0.0),
        (118, 117, 0.5, 0.065, 0.75),
        (117, 111, 0.5, 0.075, 1.0),
        (111, 143, 0.6, 0.075, 1.0),
        (143, 156, 0.4, 0.06, 0.7),
        (156, 139, 0.2, 0.045, 0.0),
    )),
    # Chin: a short capsule across the center of the chin.
    _Stroke("chin", "light", "medium", (
        (199, 208, 0.45, 0.085, 1.0),
        (199, 428, 0.45, 0.085, 1.0),
    ), mirrored=False),
    # Cupid's bow: a thin line just above the peaks of the upper lip.
    _Stroke("cupids_bow", "light", "fine", (
        (39, 165, 0.3, 0.015, 0.0),
        (37, 167, 0.35, 0.025, 1.0),
        (0, 164, 0.35, 0.025, 0.6),
        (267, 393, 0.35, 0.025, 1.0),
        (269, 391, 0.3, 0.015, 0.0),
    ), mirrored=False),
)

# Feather sigma per softness class: base + softness slider x range.
_BLUR: dict[str, tuple[float, float]] = {
    "broad": (0.05, 0.09),
    "medium": (0.04, 0.06),
    "nose": (0.04, 0.055),
    "fine": (0.025, 0.04),
}
# A stroke's feather lowers its peak; this brings the peak of a typical stroke
# back to about its level (the result is limited to 1).
_STROKE_GAIN = 1.9
# The long tail of blended powder: this share of the weight is blurred further
# by this many times the feather sigma (not around the nose, where the tails
# would meet in the middle).
_TAIL_SHARE = 0.35
_TAIL_SIGMA = 1.8
# Layering powder saturates: a weight w becomes w / (1 + _WEIGHT_KNEE * w), so
# small weights keep their value and full strength never looks painted on.
_WEIGHT_KNEE = 0.45
# Each straight stroke segment between two vertices is split into this many
# pieces per softness class (the feather hides the steps in width and level
# between them; broad strokes are blurred over more than a segment anyway).
_SUBDIVIDE: dict[str, int] = {"broad": 1, "medium": 1, "nose": 2, "fine": 2}


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

# The top of the nose bridge stays free of the nose shadow, with or without a
# highlight on it: the shadow belongs on the side walls, and on skin without
# headroom for a highlight a shaded bridge reads as a grey stripe.
_BRIDGE_KEEP = _StrokeSide.build(
    _Stroke("nose_shadow", "shade", "nose", (
        (8, 168, 0.5, 0.03, 1.0),
        (6, 197, 0.0, 0.04, 1.0),
        (195, 5, 0.0, 0.04, 1.0),
        (4, 1, 0.5, 0.03, 1.0),
    ), mirrored=False),
    "right",
)
_BRIDGE_KEEP_FEATHER = 0.045

# Coarse grid: face scale in grid pixels (never finer than the frame), and
# the blur (grid pixels) that smooths frame data sampled onto it.
_GRID_SCALE = 24.0
_DOWNSAMPLE_BLUR = 0.8
# Context around the face outline for the ROI.
_ROI_MARGIN = 0.1
# Allowed area: without a skin mask the outline is shrunk by this much and
# feathered, so the shading fades out right at the jaw instead of passing it;
# with one, the mask finds the real edge of the face near the outline.
_OVAL_INSET = 0.05
_OVAL_FEATHER = 0.025
_OVAL_GROW_MASKED = 0.05
_OVAL_FEATHER_MASKED = 0.05
# Face-skin confidence -> weight. Share of the face outline this weight covers
# -> trust: below the first value the mask and the tracker disagree too much
# (a stale mask, hair in front of a bowing face, a tracker on the wrong thing)
# and the effect is off; it is at full strength from the second value on. Real
# faces cover at least 0.75, so this is never at work in normal use.
_SKIN_RAMP = (0.25, 0.75)
_HAIR_RAMP = (0.3, 0.8)
_SKIN_COVERAGE_RAMP = (0.35, 0.6)
# Face-skin confidence -> "clearly not skin", what the hairline shadow hugs. Skin
# the model is merely less sure about (a highlight, dim light) does not count.
_NOT_SKIN_RAMP = (0.25, 0.5)
_MASK_FEATHER = 0.03
# Turned head: a side's strokes narrow with how wide that cheek looks
# (cheek apple to outline, relative to a frontal face; as the blush does).
_FRONTAL_CHEEK_WIDTH = 0.44
_SQUEEZE_RANGE = (0.55, 1.0)
_CHEEK_CONTOUR: dict[regions.Side, list[int]] = {"right": [234, 93], "left": [454, 323]}

# Hairline shadow. Its zone: the upper face outline raised by this share of its
# height above the eyes (the hairline is usually above the end of the mesh),
# closed along the brows raised by _HAIRLINE_ABOVE_BROWS, feathered. The
# shadow's reach from the hairline into the skin is base + softness slider x range.
_HAIRLINE_ARC: list[int] = [127, 162, 21, 54, 103, 67, 109, 10, 338, 297, 332, 284, 251, 389, 356]
_HAIRLINE_RAISE = 0.6
_HAIRLINE_ABOVE_BROWS = 0.12
_HAIRLINE_ZONE_FEATHER = 0.1
_HAIRLINE_REACH = (0.1, 0.12)
_HAIRLINE_GAIN = 0.7

# Protected features. On the grid: outlines grown by the larger of the
# value (x face scale) and _GUARD_GRID_PX grid pixels, so the bilinear
# upsampling cannot carry color onto the feature itself.
_EYE_GUARD = 0.04
_BROW_GUARD = 0.5  # brow height grows by this fraction
_GUARD_GRID_PX = 2.5
_GUARD_FEATHER = 0.035
# Lips and nostrils: outline growth, feather and mask resolution (pixels per
# face scale; the coarse grid's margins would swallow the cupid's bow).
_LIP_GUARD = 0.012
_NOSTRIL_GUARD = 0.02
_FINE_FEATHER = 0.02
_FINE_GRID_SCALE = 96.0
_RIGHT_NOSTRIL = [75, 79, 237, 239, 238, 20, 60, 99, 240]
_LEFT_NOSTRIL = [305, 309, 457, 459, 458, 250, 290, 328, 460]

# Highlight tone at full weight (see _tone_gain).
_CONTOUR_LIFT = 0.5
_CONTOUR_ROOM = 0.6
_CONTOUR_PEARL = 0.06


@dataclass(frozen=True)
class _Grid:
    """A coarse grid over ``roi``; frame point p maps to ``(p - offset) * ratio``."""

    roi: regions.Roi
    width: int
    height: int

    @classmethod
    def over(cls, roi: regions.Roi, scale: float, grid_scale: float = _GRID_SCALE) -> _Grid:
        factor = max(1.0, scale / grid_scale)
        return cls(roi, max(1, round(roi.w / factor)), max(1, round(roi.h / factor)))

    @property
    def size(self) -> tuple[int, int]:
        return self.width, self.height

    @property
    def ratio(self) -> np.ndarray:
        return np.array([self.width / self.roi.w, self.height / self.roi.h], np.float32)

    @property
    def coarse(self) -> bool:
        return (self.width, self.height) != (self.roi.w, self.roi.h)

    def points(self, frame_points: np.ndarray) -> np.ndarray:
        return (np.asarray(frame_points, np.float32) - np.asarray(self.roi.offset, np.float32)) * self.ratio

    def pixel_centers(self) -> tuple[np.ndarray, np.ndarray]:
        """Frame coordinates of the grid's pixel centers: x per column, y per row."""
        ratio = self.ratio
        xs = (np.arange(self.width, dtype=np.float32) + np.float32(0.5)) / ratio[0] + np.float32(self.roi.x0)
        ys = (np.arange(self.height, dtype=np.float32) + np.float32(0.5)) / ratio[1] + np.float32(self.roi.y0)
        return xs, ys

    def downsample(self, image: np.ndarray) -> np.ndarray:
        """A ROI-sized image -> grid resolution.

        Bilinear samples, then a light blur on the grid: as smooth as an area
        average for these purposes (local skin color, smooth segmentation
        masks) at a fraction of the cost.
        """
        if not self.coarse:
            return np.asarray(image, np.float32)
        small = cv2.resize(np.asarray(image, np.float32), self.size, interpolation=cv2.INTER_LINEAR)
        return _blur(small, _DOWNSAMPLE_BLUR)

    def upsample(self, image: np.ndarray) -> np.ndarray:
        """A grid image -> ROI resolution (bilinear; exact zeros stay zero)."""
        if not self.coarse:
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


def _skin_trust(weight: np.ndarray, oval: np.ndarray) -> float:
    """0..1: how much of the face outline the skin weight covers (see ``_SKIN_COVERAGE_RAMP``)."""
    coverage = float(np.sum(weight * oval)) / max(float(np.sum(oval)), 1e-6)
    return float(_smoothstep((coverage - _SKIN_COVERAGE_RAMP[0]) / (_SKIN_COVERAGE_RAMP[1] - _SKIN_COVERAGE_RAMP[0])))


def _forehead_zone(landmarks: np.ndarray, scale: float) -> np.ndarray:
    """Closed polygon (frame coordinates) where the hairline shadow may lie."""
    center = (regions.eye_center(landmarks, "left") + regions.eye_center(landmarks, "right")) / 2
    up = -_face_down(landmarks)
    arc = landmarks[_HAIRLINE_ARC].astype(np.float32)
    height = np.maximum((arc - center) @ up, np.float32(0.0))
    arc = arc + up * (height * np.float32(_HAIRLINE_RAISE))[:, None]
    raised = up * np.float32(_HAIRLINE_ABOVE_BROWS * scale)
    left = landmarks[regions.LEFT_BROW_UPPER[::-1]].astype(np.float32) + raised
    right = landmarks[regions.RIGHT_BROW_UPPER].astype(np.float32) + raised
    return np.vstack([arc, left, right])


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
        hairline = amounts["shade"] * _clamp01(settings.forehead)
        if ctx.landmarks is None or not (active or (hairline > 0.0 and ctx.face_skin_mask is not None)):
            return frame
        landmarks = ctx.landmarks
        scale = regions.face_scale(landmarks)
        if not math.isfinite(scale) or scale < _MIN_FACE_SCALE:
            return frame
        zone = _forehead_zone(landmarks, scale) if hairline > 0.0 and ctx.face_skin_mask is not None else None
        outline = landmarks[regions.FACE_OVAL]
        extent = outline if zone is None else np.vstack([outline, zone])
        roi = regions.roi_around(extent, frame.shape, pad=0.0, margin_px=_ROI_MARGIN * scale)
        if roi is None or min(roi.w, roi.h) < 0.25 * scale:
            return frame
        grid = _Grid.over(roi, scale)
        weights = _contour_weights(grid, landmarks, scale, active, zone, settings, amounts, hairline, ctx)
        if weights is None:
            return frame
        view = frame[roi.slices]
        shade, light = weights
        gain = _tone_gain(
            grid.downsample(view),
            shade,
            light,
            _shading_filter(hex_to_bgr(settings.shading_color)),
            _Light.of(hex_to_bgr(settings.highlight_color), _CONTOUR_LIFT, _CONTOUR_ROOM, _CONTOUR_PEARL),
        )
        gain = grid.upsample(gain)
        _protect_lips_and_nostrils(landmarks, roi, scale, gain)
        _apply_gain(view, gain)
        return frame


def _contour_weights(
    grid: _Grid,
    landmarks: np.ndarray,
    scale: float,
    active: list[_StrokeSide],
    zone: np.ndarray | None,
    settings: ContourSettings,
    amounts: dict[str, float],
    hairline: float,
    ctx: FrameContext,
) -> tuple[np.ndarray | None, np.ndarray | None] | None:
    """(shade, light) weights on the grid (None for a tone with nothing drawn)."""
    unit = scale * float(grid.ratio.mean())  # face scale in grid pixels
    points = grid.points(landmarks)
    outline = points[regions.FACE_OVAL]
    inset = _filled(grid.size, [regions.expand_polygon(outline, -_OVAL_INSET * unit)])
    skin = not_skin = None
    trust = 1.0
    if ctx.face_skin_mask is not None:
        confidence = grid.downsample(ctx.face_skin_mask[grid.roi.slices])
        skin = _ramp(confidence, *_SKIN_RAMP)
        trust = _skin_trust(skin, inset)
        if trust <= 0.0:
            return None
        skin = _blur(skin, _MASK_FEATHER * unit)
        if zone is not None:
            not_skin = np.float32(1.0) - _ramp(confidence, *_NOT_SKIN_RAMP)
        grown = regions.expand_polygon(outline, _OVAL_GROW_MASKED * unit)
        area = _blur(_filled(grid.size, [grown]), _OVAL_FEATHER_MASKED * unit) * skin
    else:
        # Nothing tells where the skin ends: stay inside the outline.
        area = _blur(inset, _OVAL_FEATHER * unit)
        if ctx.hair_mask is not None:
            not_hair = np.float32(1.0) - _ramp(grid.downsample(ctx.hair_mask[grid.roi.slices]), *_HAIR_RAMP)
            area = area * _blur(not_hair, _MASK_FEATHER * unit)
    squeeze = {side: _cheek_squeeze(landmarks, side, scale) for side in regions.SIDES}
    softness = _clamp01(settings.softness)
    sigmas = {name: (base + span * softness) * unit for name, (base, span) in _BLUR.items()}
    paths: dict[tuple[str, str], list[_Path]] = {}
    for item in active:
        stroke = item.stroke
        levels = item.levels * np.float32(_clamp01(getattr(settings, stroke.area)))
        width = unit * (squeeze[item.side] if stroke.mirrored else 1.0)
        paths.setdefault((stroke.tone, stroke.blur), []).append(_Path(item.points(points), item.radii * np.float32(width), levels))
    guard = np.float32(1.0) - _feature_guard(grid, points, unit)
    allowed = area * guard * np.float32(trust)
    tones: dict[str, np.ndarray] = {}
    for (tone, blur_class), group in paths.items():
        canvas = np.zeros((grid.height, grid.width), np.uint8)
        _draw_paths(canvas, group)
        sigma = sigmas[blur_class]
        soft = _blur(_to_unit(canvas), sigma)
        if blur_class == "nose":
            soft = soft * (np.float32(1.0) - _bridge_area(grid, points, unit))
        elif blur_class != "fine":
            soft = soft * np.float32(1.0 - _TAIL_SHARE) + _blur(soft, _TAIL_SIGMA * sigma) * np.float32(_TAIL_SHARE)
        soft = np.minimum(soft * np.float32(_STROKE_GAIN), np.float32(1.0))
        tones[tone] = _union(tones[tone], soft) if tone in tones else soft
    weights: dict[str, np.ndarray | None] = {}
    for tone in ("shade", "light"):
        weight = tones.get(tone)
        weights[tone] = None if weight is None else _saturate(weight * allowed * np.float32(amounts[tone]))
    if zone is not None:
        shadow = _saturate(_hairline_shadow(grid, zone, skin, not_skin, unit, softness) * guard * np.float32(hairline * trust))
        weights["shade"] = shadow if weights["shade"] is None else _union(weights["shade"], shadow)
    for tone, weight in weights.items():
        if weight is not None and not float(weight.max()) > 0.0:
            weights[tone] = None
    if weights["shade"] is None and weights["light"] is None:
        return None
    return weights["shade"], weights["light"]


def _saturate(weight: np.ndarray) -> np.ndarray:
    """Compress heavy weights (see ``_WEIGHT_KNEE``); exact zeros stay zero."""
    return weight / (np.float32(1.0) + np.float32(_WEIGHT_KNEE) * weight)


def _union(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Soft union of two weights: no crease where they overlap."""
    return a + b - a * b


def _bridge_area(grid: _Grid, points: np.ndarray, unit: float) -> np.ndarray:
    """The top of the nose bridge as a soft 0..1 mask on the grid."""
    canvas = np.zeros((grid.height, grid.width), np.uint8)
    path = _Path(_BRIDGE_KEEP.points(points), _BRIDGE_KEEP.radii * np.float32(unit), _BRIDGE_KEEP.levels)
    _draw_paths(canvas, [path])
    return _blur(_to_unit(canvas), _BRIDGE_KEEP_FEATHER * unit)


def _hairline_shadow(grid: _Grid, zone: np.ndarray, skin: np.ndarray, not_skin: np.ndarray, unit: float, softness: float) -> np.ndarray:
    """Shadow on the skin along the real hairline (or bangs), 1 at the hairline, fading inward."""
    reach = (_HAIRLINE_REACH[0] + _HAIRLINE_REACH[1] * softness) * unit
    near_hair = np.minimum(_blur(not_skin, reach) * np.float32(2.0), np.float32(1.0))
    area = _blur(_filled(grid.size, [grid.points(zone)]), _HAIRLINE_ZONE_FEATHER * unit)
    return near_hair * area * skin * np.float32(_HAIRLINE_GAIN)


def _feature_guard(grid: _Grid, points: np.ndarray, unit: float) -> np.ndarray:
    """Eyes and brows (grid coordinates) as a 0..1 keep-out mask."""
    polygons = []
    for side in regions.SIDES:
        s = regions.side_indices(side)
        polygons.append(regions.expand_polygon(points[s.eye], max(_EYE_GUARD * unit, _GUARD_GRID_PX)))
        upper, lower = points[s.brow_upper], points[s.brow_lower]
        center, half = (upper + lower) / 2, (upper - lower) / 2 * (1.0 + _BROW_GUARD)
        polygons.append(regions.expand_polygon(np.vstack([center + half, (center - half)[::-1]]), _GUARD_GRID_PX))
    return _solid_shapes(grid.size, polygons, max(_GUARD_FEATHER * unit, 0.5), grow=True)


def _protect_lips_and_nostrils(landmarks: np.ndarray, roi: regions.Roi, scale: float, gain: np.ndarray) -> None:
    """Zero the gain (in place, ROI-sized) there, at full resolution up to ``_FINE_GRID_SCALE``."""
    feather_px = _FINE_FEATHER * scale
    outlines = [regions.expand_polygon(landmarks[regions.LIPS_OUTER], _LIP_GUARD * scale)]
    for nostril in (_RIGHT_NOSTRIL, _LEFT_NOSTRIL):
        hull = cv2.convexHull(landmarks[nostril].astype(np.float32)).reshape(-1, 2)
        outlines.append(regions.expand_polygon(hull, _NOSTRIL_GUARD * scale))
    area = regions.roi_around(np.vstack(outlines), (roi.y1, roi.x1), pad=0.0, margin_px=3.0 * feather_px + 2.0)
    if area is None:
        return
    area = regions.Roi(max(area.x0, roi.x0), max(area.y0, roi.y0), area.x1, area.y1)
    if area.w <= 0 or area.h <= 0:
        return
    grid = _Grid.over(area, scale, _FINE_GRID_SCALE)
    shape = _solid_shapes(grid.size, [grid.points(o) for o in outlines], feather_px * float(grid.ratio.mean()), grow=grid.coarse)
    keep = grid.upsample(np.float32(1.0) - shape)
    keep = cv2.merge([keep, keep, keep])
    local = (slice(area.y0 - roi.y0, area.y1 - roi.y0), slice(area.x0 - roi.x0, area.x1 - roi.x0))
    gain[local] = cv2.multiply(gain[local], keep)


# --------------------------------------------------------------------------
# Tear bags.
# --------------------------------------------------------------------------

# Band under the lower lashes: center and half-height below the lash line.
_BAG_CENTER = 0.042
_BAG_HALF = 0.03
# Shadow along the lower edge of the bag.
_SHADOW_CENTER = 0.092
_SHADOW_HALF = 0.01
# Feather sigmas: base + softness slider x range.
_BAG_FEATHER = (0.008, 0.016)
_SHADOW_FEATHER = (0.012, 0.018)
# Along the lid (0 = inner corner, 1 = outer corner; the iris sits around 0.5):
# the highlight fades in up to the edge of the iris, the shadow starts later;
# both fade out over this share of their length. The width slider moves the end
# of the bag between these points; it fades away entirely below _REACH_FADE.
# Nothing goes further than _LID_END past the outer corner.
_BAG_FADE_IN = (0.02, 0.34)
_SHADOW_FADE_IN = (0.12, 0.42)
_FADE_OUT = 0.35
_REACH_RANGE = (0.7, 1.3)
_REACH_FADE = 0.15
_LID_END = 0.12
# Highlight tone (see _tone_gain) and shadow color; the shadow's strength
# relative to the highlight.
_BAG_LIGHT = _Light.of(hex_to_bgr("#FFF2EA"), lift=0.6, room=0.9, pearl=0.06)
_BAG_SHADOW: Bgr = hex_to_bgr("#9E7A68")
_SHADOW_GAIN = 0.5
# Narrowed eyes: eye height / width from the first to the second value maps
# the shadow from _NARROW_SHADOW of its strength to all of it. Smoothed per
# frame, so a blink hardly moves it and eyes that stay narrowed do.
_OPENNESS_RAMP = (0.12, 0.28)
_NARROW_SHADOW = 0.3
_NARROW_SMOOTHING = 0.08
# Everything above this depth below the lower lash line is left alone (the eye
# opening, the lashes' roots); the bag reaches its full level over the feather.
_EYE_CUT = 0.016
_EYE_CUT_FEATHER = 0.03
# How far the real lower lid is ahead of the smoothed landmarks while it moves
# down: this many times its step between frames, once the step passes the dead
# zone (tracking jitter), up to the maximum. A larger step is a jump (a face
# found again), not motion, and starts the bag afresh.
_LID_LAG_PER_STEP = 0.5
_LID_LAG_DEAD_ZONE = 0.002
_LID_LAG_MAX = 0.08
_LID_JUMP = 0.25
_LID_MIDDLE = slice(2, 7)
# The skin under the bag is sampled on a band below the lashes that does not
# change with a smile. Pixels whose brightness relative to the typical pixel
# there (this percentile: lashes are the common outlier) is outside these ramps
# do not count. The result is smoothed per frame, slower while the eye is
# closing: real light changes take about half a second.
_SKIN_SAMPLE_CENTER = 0.06
_SKIN_SAMPLE_HALF = 0.025
_SKIN_SAMPLE_FEATHER = 0.005
_SKIN_TYPICAL_PERCENTILE = 65.0
_SKIN_DARK_RAMP = (0.75, 0.9)
_SKIN_BRIGHT_RAMP = (1.15, 1.3)
_SKIN_SMOOTHING = 0.1
_SKIN_CLOSED_EYE = 0.1
# Smiling: mouth width / face scale from the first to the second value maps
# to 0..1 (smoothed over frames); at 1 the bag is this much fuller.
_SMILE_RAMP = (0.98, 1.12)
_SMILE_SMOOTHING = 0.35
_SMILE_SIZE = 0.3
_SMILE_LIGHT = 0.25
_SMILE_SHADOW = 0.4
_MIN_EYE_WIDTH_PX = 6.0
# The masks are functions of the position relative to the lid, evaluated on a
# grid of about this many pixels per face scale: no rasterized edge that could
# flip between frames, and a cost that hardly grows with the face.
_BAG_GRID_SCALE = 96.0
# A smoothstep this many sigmas wide has the slope of a Gaussian-feathered edge.
_SOFT_EDGE_WIDTH = 3.76


@dataclass
class _EyeMemory:
    """What the bag of one eye keeps from frame to frame."""

    lid: np.ndarray  # middle of the lower lash line (frame pixels)
    narrow: float  # smoothed shadow factor for narrowed eyes
    skin: np.ndarray | None = None  # smoothed color under the bag (BGR)


@register_effect
class TearbagEffect(Effect):
    """Tear-bag (涙袋) highlight and shadow under the lower lashes."""

    name = "tearbag"
    order = 45

    def __init__(self) -> None:
        self._smile: float | None = None
        self._eyes: dict[regions.Side, _EyeMemory] = {}

    def reset(self) -> None:
        self._smile = None
        self._eyes.clear()

    def apply(self, frame: np.ndarray, ctx: FrameContext, settings: TearbagSettings) -> np.ndarray:
        amount = _clamp01(settings.strength) * ctx.intensity
        width = _clamp01(settings.width)
        if amount <= 0.0 or width <= 0.0 or ctx.landmarks is None:
            self.reset()  # whatever was remembered is out of date when the bag comes back
            return frame
        landmarks = ctx.landmarks
        scale = regions.face_scale(landmarks)
        if not math.isfinite(scale) or scale < _MIN_FACE_SCALE:
            self.reset()
            return frame
        smile = self._update_smile(landmarks, scale)
        amount *= min(width / _REACH_FADE, 1.0)
        shade_filter = _shading_filter(_BAG_SHADOW)
        for side in regions.SIDES:
            memory = _tear_bag(frame, landmarks, side, scale, settings, amount, width, smile, shade_filter, self._eyes.get(side))
            if memory is None:
                self._eyes.pop(side, None)
            else:
                self._eyes[side] = memory
        return frame

    def _update_smile(self, landmarks: np.ndarray, scale: float) -> float:
        width = float(np.linalg.norm(landmarks[regions.MOUTH_LEFT_CORNER] - landmarks[regions.MOUTH_RIGHT_CORNER]))
        target = float(_smoothstep((width / scale - _SMILE_RAMP[0]) / (_SMILE_RAMP[1] - _SMILE_RAMP[0])))
        if self._smile is None or not math.isfinite(self._smile):
            self._smile = target
        else:
            self._smile += _SMILE_SMOOTHING * (target - self._smile)
        return self._smile


def _lid_profile(u: np.ndarray, fade_in: tuple[float, float], reach: float) -> np.ndarray:
    """Level along the lid: fades in over ``fade_in`` and out toward ``reach``."""
    rise = _smoothstep((u - fade_in[0]) / (fade_in[1] - fade_in[0]))
    fall = _smoothstep((reach - u) / max(_FADE_OUT * reach, 1e-3))
    return rise * fall


def _eye_openness(landmarks: np.ndarray, side: regions.Side, eye_width: float) -> float:
    """Height of the eye opening relative to its width (from the lash-line midpoints)."""
    s = regions.side_indices(side)
    upper = landmarks[s.eye_upper[3:6]].astype(np.float32).mean(axis=0)
    lower = landmarks[s.eye_lower[3:6]].astype(np.float32).mean(axis=0)
    return float(np.linalg.norm(upper - lower)) / max(eye_width, 1e-6)


def _lid_lag(step_down: float, scale: float) -> float:
    """How far (pixels, along "down") the real lower lid is ahead of the smoothed one."""
    return min(max(step_down / scale - _LID_LAG_DEAD_ZONE, 0.0) * _LID_LAG_PER_STEP, _LID_LAG_MAX) * scale


def _lid_coordinates(grid: _Grid, lash: np.ndarray, along: np.ndarray, down: np.ndarray, eye_width: float) -> tuple[np.ndarray, np.ndarray]:
    """For every grid pixel: position along the lid (0 inner corner, 1 outer corner) and depth below the lash line (frame pixels)."""
    xs, ys = grid.pixel_centers()
    dx, dy = (xs - lash[0, 0])[None, :], (ys - lash[0, 1])[:, None]
    u = (dx * along[0] + dy * along[1]) * np.float32(1.0 / eye_width)
    depth = dx * down[0] + dy * down[1]
    relative = lash - lash[0]
    lid_u = np.maximum.accumulate(relative @ along) / eye_width  # np.interp needs it sorted
    depth -= np.interp(u, lid_u, relative @ down).astype(np.float32)
    return u, depth


def _soft_band(depth: np.ndarray, top: float, bottom: float, sigma: float) -> np.ndarray:
    """1 between two depths, with smooth edges of about a Gaussian feather of ``sigma``; exactly 0 away from them."""
    width = max(_SOFT_EDGE_WIDTH * sigma, 1e-3)
    return _smoothstep((depth - top) / width + 0.5) - _smoothstep((depth - bottom) / width + 0.5)


def _measure_skin(small: np.ndarray, weight: np.ndarray) -> np.ndarray | None:
    """Mean color (BGR) under ``weight``, leaving out what is far from the typical brightness there."""
    peak = float(weight.max())
    if peak <= 1e-3:
        return None
    luma = blend.luminance(small)
    typical = max(float(np.percentile(luma[weight > 0.5 * peak], _SKIN_TYPICAL_PERCENTILE)), _MIN_BASE)
    relative = luma / np.float32(typical)
    inlier = _ramp(relative, *_SKIN_DARK_RAMP) * (np.float32(1.0) - _ramp(relative, *_SKIN_BRIGHT_RAMP)) * weight
    total = float(inlier.sum())
    if total <= 1e-3:
        return None
    return (small * inlier[..., None]).sum(axis=(0, 1), dtype=np.float32) / np.float32(total)


def _tear_bag(
    frame: np.ndarray,
    landmarks: np.ndarray,
    side: regions.Side,
    scale: float,
    settings: TearbagSettings,
    amount: float,
    width: float,
    smile: float,
    shade_filter: np.ndarray,
    memory: _EyeMemory | None,
) -> _EyeMemory | None:
    """Paints the bag of one eye; returns what to remember for the next frame."""
    s = regions.side_indices(side)
    lash = landmarks[s.eye_lower].astype(np.float32)
    inner, outer = lash[0], lash[-1]
    eye_width = float(np.linalg.norm(outer - inner))
    if not np.isfinite(lash).all() or eye_width < _MIN_EYE_WIDTH_PX:
        return None
    # "Down": perpendicular to the eye line, on the chin side of the face
    # (robust while blinking, when the lids meet).
    along = (outer - inner) / eye_width
    down = np.array([-along[1], along[0]], np.float32)
    if float(np.dot(down, _face_down(landmarks))) < 0:
        down = -down
    middle = lash[_LID_MIDDLE].mean(axis=0)
    if memory is not None and float(np.linalg.norm(middle - memory.lid)) > _LID_JUMP * scale:
        memory = None
    lag = 0.0 if memory is None else _lid_lag(float(np.dot(middle - memory.lid, down)), scale)
    eye_open = float(_ramp(np.float32(_eye_openness(landmarks, side, eye_width)), *_OPENNESS_RAMP))
    narrow = eye_open if memory is None else memory.narrow + _NARROW_SMOOTHING * (eye_open - memory.narrow)
    remembered = _EyeMemory(middle, narrow, None if memory is None else memory.skin)

    grow = 1.0 + _SMILE_SIZE * smile
    softness = _clamp01(settings.softness)
    reach = _REACH_RANGE[0] + (_REACH_RANGE[1] - _REACH_RANGE[0]) * width
    bag_sigma = (_BAG_FEATHER[0] + _BAG_FEATHER[1] * softness) * scale
    shadow_sigma = (_SHADOW_FEATHER[0] + _SHADOW_FEATHER[1] * softness) * scale
    bag_center, shadow_center = _BAG_CENTER * grow * scale, _SHADOW_CENTER * grow * scale
    bag_half, shadow_half = _BAG_HALF * grow * scale, _SHADOW_HALF * scale
    lowest = shadow_center + shadow_half + _SOFT_EDGE_WIDTH * shadow_sigma / 2.0 + lag
    highest = _EYE_CUT * scale + lag
    extent = np.vstack([lash + down * np.float32(highest), lash + down * np.float32(lowest), outer + along * np.float32(_LID_END * eye_width)])
    roi = regions.roi_around(extent, frame.shape, pad=0.0, margin_px=2.0)
    if roi is None:
        return remembered
    grid = _Grid.over(roi, scale, _BAG_GRID_SCALE)
    u, depth = _lid_coordinates(grid, lash, along, down, eye_width)
    depth -= np.float32(lag)
    keep = _smoothstep((depth - np.float32(_EYE_CUT * scale)) * np.float32(1.0 / (_EYE_CUT_FEATHER * scale)))
    keep = keep * _smoothstep((np.float32(1.0 + _LID_END) - u) * np.float32(1.0 / _LID_END))
    level = _lid_profile(u, _BAG_FADE_IN, reach) * keep

    view = frame[roi.slices]
    sample = (_SKIN_SAMPLE_CENTER - _SKIN_SAMPLE_HALF) * scale, (_SKIN_SAMPLE_CENTER + _SKIN_SAMPLE_HALF) * scale
    measured = _measure_skin(grid.downsample(view), level * _soft_band(depth, *sample, _SKIN_SAMPLE_FEATHER * scale))
    if measured is not None:
        if remembered.skin is None:
            remembered.skin = measured
        else:
            pace = _SKIN_SMOOTHING * max(eye_open, _SKIN_CLOSED_EYE)
            remembered.skin = remembered.skin + np.float32(pace) * (measured - remembered.skin)
    if remembered.skin is None:
        return remembered

    light_level = np.float32(min(amount * (1.0 + _SMILE_LIGHT * smile), 1.0))
    light_weight = level * _soft_band(depth, bag_center - bag_half, bag_center + bag_half, bag_sigma) * light_level
    shadow_level = min(
        amount
        * _clamp01(settings.shadow)
        * _SHADOW_GAIN
        * (1.0 + _SMILE_SHADOW * smile)
        * (_NARROW_SHADOW + (1.0 - _NARROW_SHADOW) * narrow),
        1.0,
    )
    shade_weight = None
    if shadow_level > 0.0:
        shade_weight = (
            _lid_profile(u, _SHADOW_FADE_IN, reach)
            * keep
            * _soft_band(depth, shadow_center - shadow_half, shadow_center + shadow_half, shadow_sigma)
            * np.float32(shadow_level)
        )
    gain = _tone_gain(remembered.skin.reshape(1, 1, 3), shade_weight, light_weight, shade_filter, _BAG_LIGHT)
    _apply_gain(view, grid.upsample(gain))
    return remembered
