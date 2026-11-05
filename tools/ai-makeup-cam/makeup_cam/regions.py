"""Face regions: landmark index sets, regions of interest and soft masks.

Landmarks are the 478 points of the MediaPipe Face Landmarker (468 mesh
points + 10 iris points) as an ``(478, 2)`` float array of full-frame pixel
coordinates of the *unmirrored* frame. Coordinates are continuous: pixel
``(i, j)`` covers ``[i, i + 1) x [j, j + 1)``, which is exactly what
``normalized * frame_size`` gives.

Naming: ``LEFT_*`` / ``RIGHT_*`` mean the **subject's** left / right, so
``LEFT_*`` points appear on the right half of an unmirrored image. Open
lines around an eye are ordered from the inner corner (nose side) to the
outer corner; brow lines likewise run from the inner end to the tail.

Masks are float32 arrays of the size of a :class:`Roi`, values in [0, 1].
Shapes are rasterized with 4x4 supersampling (accurate anti-aliased
coverage, unbiased edges) and optionally feathered with a Gaussian. Large
feathers are computed at reduced resolution, which keeps every helper well
below a millisecond for typical face sizes.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

Side = Literal["left", "right"]
SIDES: tuple[Side, Side] = ("left", "right")

# --------------------------------------------------------------------------
# Landmark indices (MediaPipe Face Landmarker, 478-point topology).
# Lists are shared constants: index with them, never mutate them.
# --------------------------------------------------------------------------

# Face outline, clockwise in the image, starting at the top of the forehead.
FACE_OVAL: list[int] = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
    152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
]

# Lip contours as open lines from the subject's right mouth corner (image
# left) to the left corner.
LIPS_UPPER_OUTER: list[int] = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
LIPS_LOWER_OUTER: list[int] = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
LIPS_UPPER_INNER: list[int] = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
LIPS_LOWER_INNER: list[int] = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
# Closed polygons: outer lip line, and the mouth opening (inner lip line).
LIPS_OUTER: list[int] = LIPS_UPPER_OUTER + LIPS_LOWER_OUTER[-2:0:-1]
LIPS_INNER: list[int] = LIPS_UPPER_INNER + LIPS_LOWER_INNER[-2:0:-1]
# Single points on the vertical center line of the mouth.
LIPS_TOP = 0  # top of the upper lip (cupid's bow center)
LIPS_INNER_TOP = 13  # bottom edge of the upper lip
LIPS_INNER_BOTTOM = 14  # top edge of the lower lip
LIPS_BOTTOM = 17  # bottom of the lower lip
MOUTH_RIGHT_CORNER = 61
MOUTH_LEFT_CORNER = 291

# Eyes: lash lines (inner corner -> outer corner) and the closed opening.
RIGHT_EYE_UPPER: list[int] = [133, 173, 157, 158, 159, 160, 161, 246, 33]
RIGHT_EYE_LOWER: list[int] = [133, 155, 154, 153, 145, 144, 163, 7, 33]
LEFT_EYE_UPPER: list[int] = [362, 398, 384, 385, 386, 387, 388, 466, 263]
LEFT_EYE_LOWER: list[int] = [362, 382, 381, 380, 374, 373, 390, 249, 263]
RIGHT_EYE: list[int] = RIGHT_EYE_UPPER + RIGHT_EYE_LOWER[-2:0:-1]
LEFT_EYE: list[int] = LEFT_EYE_UPPER + LEFT_EYE_LOWER[-2:0:-1]
RIGHT_EYE_INNER_CORNER = 133
RIGHT_EYE_OUTER_CORNER = 33
LEFT_EYE_INNER_CORNER = 362
LEFT_EYE_OUTER_CORNER = 263

# Iris: closed 4-point ring around the iris, plus its center point.
RIGHT_IRIS: list[int] = [469, 470, 471, 472]
LEFT_IRIS: list[int] = [474, 475, 476, 477]
RIGHT_IRIS_CENTER = 468
LEFT_IRIS_CENTER = 473

# Upper-lid crease, a few pixels above the upper lash line (inner -> outer).
RIGHT_EYE_CREASE: list[int] = [190, 56, 28, 27, 29, 30, 247]
LEFT_EYE_CREASE: list[int] = [414, 286, 258, 257, 259, 260, 467]

# Under-eye area: the line just below the lower lashes and the line along the
# top of the cheekbone (inner -> outer, index-wise paired).
RIGHT_UNDER_EYE_TOP: list[int] = [243, 112, 26, 22, 23, 24, 110, 25, 130]
RIGHT_UNDER_EYE_BOTTOM: list[int] = [245, 128, 121, 120, 119, 118, 117, 111, 143]
LEFT_UNDER_EYE_TOP: list[int] = [463, 341, 256, 252, 253, 254, 339, 255, 359]
LEFT_UNDER_EYE_BOTTOM: list[int] = [465, 357, 350, 349, 348, 347, 346, 340, 372]

# Brows: upper and lower edges (inner end -> tail, index-wise paired) and the
# closed outline.
RIGHT_BROW_UPPER: list[int] = [107, 66, 105, 63, 70]
RIGHT_BROW_LOWER: list[int] = [55, 65, 52, 53, 46]
LEFT_BROW_UPPER: list[int] = [336, 296, 334, 293, 300]
LEFT_BROW_LOWER: list[int] = [285, 295, 282, 283, 276]
RIGHT_BROW: list[int] = RIGHT_BROW_UPPER + RIGHT_BROW_LOWER[::-1]
LEFT_BROW: list[int] = LEFT_BROW_UPPER + LEFT_BROW_LOWER[::-1]

# Apple of the cheek (average these points for the blush center).
RIGHT_CHEEK_CENTER: list[int] = [50, 101, 36, 205]
LEFT_CHEEK_CENTER: list[int] = [280, 330, 266, 425]

# Nose.
NOSE_TIP = 1
NOSE_BOTTOM = 2  # subnasale, where the nose meets the upper lip
NOSE_BRIDGE: list[int] = [168, 6, 197, 195, 5, 4]  # between the eyes -> tip
# Closed outline around each nose wing (ala), for redness correction.
RIGHT_NOSE_WING: list[int] = [129, 49, 131, 220, 237, 79, 75, 240, 98, 64]
LEFT_NOSE_WING: list[int] = [358, 279, 360, 440, 457, 309, 305, 460, 327, 294]


@dataclass(frozen=True)
class SideIndices:
    """All per-side index sets of one side of the face (see module constants)."""

    eye: list[int]
    eye_upper: list[int]
    eye_lower: list[int]
    eye_crease: list[int]
    iris: list[int]
    iris_center: int
    brow: list[int]
    brow_upper: list[int]
    brow_lower: list[int]
    under_eye_top: list[int]
    under_eye_bottom: list[int]
    cheek_center: list[int]
    nose_wing: list[int]


_SIDE_INDICES: dict[str, SideIndices] = {
    "left": SideIndices(
        eye=LEFT_EYE,
        eye_upper=LEFT_EYE_UPPER,
        eye_lower=LEFT_EYE_LOWER,
        eye_crease=LEFT_EYE_CREASE,
        iris=LEFT_IRIS,
        iris_center=LEFT_IRIS_CENTER,
        brow=LEFT_BROW,
        brow_upper=LEFT_BROW_UPPER,
        brow_lower=LEFT_BROW_LOWER,
        under_eye_top=LEFT_UNDER_EYE_TOP,
        under_eye_bottom=LEFT_UNDER_EYE_BOTTOM,
        cheek_center=LEFT_CHEEK_CENTER,
        nose_wing=LEFT_NOSE_WING,
    ),
    "right": SideIndices(
        eye=RIGHT_EYE,
        eye_upper=RIGHT_EYE_UPPER,
        eye_lower=RIGHT_EYE_LOWER,
        eye_crease=RIGHT_EYE_CREASE,
        iris=RIGHT_IRIS,
        iris_center=RIGHT_IRIS_CENTER,
        brow=RIGHT_BROW,
        brow_upper=RIGHT_BROW_UPPER,
        brow_lower=RIGHT_BROW_LOWER,
        under_eye_top=RIGHT_UNDER_EYE_TOP,
        under_eye_bottom=RIGHT_UNDER_EYE_BOTTOM,
        cheek_center=RIGHT_CHEEK_CENTER,
        nose_wing=RIGHT_NOSE_WING,
    ),
}


def side_indices(side: Side) -> SideIndices:
    """Index sets for the subject's ``"left"`` or ``"right"`` side."""
    try:
        return _SIDE_INDICES[side]
    except KeyError:
        raise ValueError(f"side must be 'left' or 'right', got {side!r}") from None


# --------------------------------------------------------------------------
# Regions of interest and measurements.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Roi:
    """Axis-aligned pixel rectangle ``[x0, x1) x [y0, y1)`` inside a frame."""

    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def w(self) -> int:
        return self.x1 - self.x0

    @property
    def h(self) -> int:
        return self.y1 - self.y0

    @property
    def slices(self) -> tuple[slice, slice]:
        """Index a frame with ``frame[roi.slices]`` to get the ROI view."""
        return slice(self.y0, self.y1), slice(self.x0, self.x1)

    @property
    def offset(self) -> tuple[int, int]:
        """Top-left corner ``(x0, y0)``; subtract it to get ROI-local coordinates."""
        return self.x0, self.y0


def roi_around(
    points: np.ndarray,
    shape: tuple[int, ...],
    pad: float = 0.25,
    margin_px: float = 0.0,
) -> Roi | None:
    """Bounding box of ``points`` grown by ``pad`` x its larger side plus ``margin_px``.

    ``shape`` is the frame shape ``(H, W, ...)``. The box is clipped to the
    frame. Returns None for no (finite) points or an empty clipped box. Use
    ``margin_px`` of about 3x the feather sigma so feathered masks are not
    cut off at the ROI border.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) == 0:
        return None
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    grow = pad * float((hi - lo).max()) + margin_px
    height, width = shape[0], shape[1]
    x0 = max(0, math.floor(lo[0] - grow))
    y0 = max(0, math.floor(lo[1] - grow))
    x1 = min(width, math.ceil(hi[0] + grow))
    y1 = min(height, math.ceil(hi[1] + grow))
    if x1 <= x0 or y1 <= y0:
        return None
    return Roi(x0, y0, x1, y1)


def eye_center(landmarks: np.ndarray, side: Side) -> np.ndarray:
    """Midpoint of the two corners of one eye (stable under gaze changes)."""
    s = side_indices(side)
    return landmarks[[s.eye_upper[0], s.eye_upper[-1]]].astype(np.float32).mean(axis=0)


def cheek_center(landmarks: np.ndarray, side: Side) -> np.ndarray:
    """Center of the apple of the cheek, ``(x, y)`` float32."""
    return landmarks[side_indices(side).cheek_center].astype(np.float32).mean(axis=0)


# Distance from the eye line to the base of the nose, as a fraction of the
# inter-ocular distance. Measured ~0.73-0.75 on frontal faces; the margin
# keeps the inter-ocular distance in charge unless the head is turned
# sideways by more than ~20 degrees.
_EYES_TO_NOSE_BASE_PER_IOD = 0.8


def face_scale(landmarks: np.ndarray) -> float:
    """Face size in pixels: the inter-ocular distance (eye center to eye center).

    The eye centers are the midpoints of the eye corners, so the value does
    not change with gaze. When the head turns sideways the horizontal
    distance shrinks; the vertical eyes-to-nose-base distance (rescaled to
    the same unit) then takes over, so sizes derived from this value stay
    steady. Roughly 60 px for a small face at 720p, 100-130 px for a
    typical webcam framing.
    """
    right = eye_center(landmarks, "right")
    left = eye_center(landmarks, "left")
    inter_ocular = float(np.linalg.norm(left - right))
    eyes_to_nose = float(np.linalg.norm(landmarks[NOSE_BOTTOM] - (left + right) / 2))
    return max(inter_ocular, eyes_to_nose / _EYES_TO_NOSE_BASE_PER_IOD)


# --------------------------------------------------------------------------
# Geometry on point lists.
# --------------------------------------------------------------------------


def expand_polygon(points: np.ndarray, amount_px: float) -> np.ndarray:
    """Move every vertex ``amount_px`` away from the centroid (negative shrinks).

    Vertices never cross the centroid when shrinking. Returns float32 (N, 2).
    """
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if amount_px == 0 or len(pts) == 0:
        return pts.copy()
    centroid = pts.mean(axis=0)
    offsets = pts - centroid
    dist = np.linalg.norm(offsets, axis=1, keepdims=True)
    scale = np.maximum(dist + amount_px, 0.0) / np.maximum(dist, 1e-6)
    return (centroid + offsets * scale).astype(np.float32)


def resample_polyline(points: np.ndarray, count: int) -> np.ndarray:
    """``count`` points evenly spaced by arc length along an open polyline.

    The first and last points are kept. Returns float32 (count, 2).
    """
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    if arc[-1] <= 0:
        return np.repeat(pts[:1], count, axis=0)
    targets = np.linspace(0.0, arc[-1], count)
    xs = np.interp(targets, arc, pts[:, 0])
    ys = np.interp(targets, arc, pts[:, 1])
    return np.stack([xs, ys], axis=1).astype(np.float32)


# --------------------------------------------------------------------------
# Rasterization.
# --------------------------------------------------------------------------

_SUPERSAMPLE = 4  # sub-pixels per pixel along each axis
_SHIFT = 4  # fractional bits of the fixed-point coordinates cv2 draws with
_MAX_DOWNSCALE = 8
_MIN_LOWRES_SIGMA = 3.0  # blur sigma (in low-res pixels) that keeps the upsampling invisible
_FEATHER_REACH = 4.0  # a Gaussian counts as zero beyond this many sigmas


def _feather_downscale(sigma_px: float) -> int:
    """Resolution divisor for blurring with ``sigma_px`` (1 = full resolution)."""
    return int(min(max(sigma_px // _MIN_LOWRES_SIGMA, 1), _MAX_DOWNSCALE))


def _twice_signed_area(pts: np.ndarray, edges: np.ndarray) -> float:
    return float(np.dot(pts[:, 0], edges[:, 1]) - np.dot(pts[:, 1], edges[:, 0]))


def _inset_polygon(points: np.ndarray, delta: float) -> np.ndarray | None:
    """Move every edge of a closed polygon ``delta`` inward (mitered corners).

    Returns None when nothing is left (the polygon is thinner than
    ``2 * delta`` everywhere, e.g. the opening of a closed mouth).
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(pts) < 3:
        return None
    edges = np.concatenate((pts[1:], pts[:1])) - pts  # edge i: vertex i -> i + 1
    twice_area = _twice_signed_area(pts, edges)
    if twice_area == 0:
        return None
    # Unit normals pointing into the polygon (left of each edge when the
    # signed area is positive); zero for zero-length edges.
    lengths = np.maximum(np.hypot(edges[:, 0], edges[:, 1]), 1e-12)
    inward = edges[:, ::-1] * (np.array([-1.0, 1.0]) * (np.sign(twice_area) / lengths)[:, None])
    # Vertex i joins edges i - 1 and i: offset delta / cos(half the turn
    # angle) along the bisector, limited to 2 * delta at spikes.
    bisector = inward + np.concatenate((inward[-1:], inward[:-1]))
    norm = np.hypot(bisector[:, 0], bisector[:, 1])
    scale = 2.0 * delta / np.maximum(norm * np.maximum(norm, 1.0), 1e-12)
    inset = pts + bisector * scale[:, None]
    inset_edges = np.concatenate((inset[1:], inset[:1])) - inset
    return inset if _twice_signed_area(inset, inset_edges) * twice_area > 0 else None


def _segment_quads(points: np.ndarray, radii: np.ndarray) -> np.ndarray:
    """(M, 4, 2) convex quads around each polyline segment, half-width ``radii``
    at its vertices (zero-length or zero-width segments are dropped)."""
    a, b = points[:-1], points[1:]
    ra, rb = radii[:-1, None], radii[1:, None]
    direction = b - a
    length = np.linalg.norm(direction, axis=1, keepdims=True)
    keep = (length[:, 0] > 1e-6) & (ra[:, 0] + rb[:, 0] > 0)
    normal = np.divide(np.stack([-direction[:, 1], direction[:, 0]], axis=1), length, out=np.zeros_like(direction), where=length > 1e-6)
    quads = np.stack([a + normal * ra, b + normal * rb, b - normal * rb, a - normal * ra], axis=1)
    return quads[keep]


class _Raster:
    """Supersampled 8-bit canvas covering ``roi`` at 1/``downscale`` resolution.

    Shapes are scan-converted without anti-aliasing on a grid of 4x4
    sub-pixels per (low-res) pixel; :meth:`coverage` box-filters that down to
    accurate anti-aliased coverage. cv2 fills every sub-pixel whose center
    is within half a sub-pixel of a shape, so shapes are drawn shrunk by that
    much (``edge_bias_px``) to keep their edges unbiased.
    """

    def __init__(self, roi: Roi, downscale: int = 1) -> None:
        self.roi = roi
        self.h = -(-roi.h // downscale)
        self.w = -(-roi.w // downscale)
        self._scale = _SUPERSAMPLE / downscale
        self.edge_bias_px = 0.5 / self._scale
        self.canvas = np.zeros((self.h * _SUPERSAMPLE, self.w * _SUPERSAMPLE), np.uint8)

    def _fixed(self, points: np.ndarray) -> np.ndarray:
        """Frame coordinates -> int32 fixed-point canvas coordinates (pixel-center based)."""
        local = (np.asarray(points, dtype=np.float64) - self.roi.offset) * self._scale - 0.5
        return np.round(local * (1 << _SHIFT)).astype(np.int32)

    def _length(self, px: float) -> int:
        return int(round(px * self._scale * (1 << _SHIFT)))

    def polygon(self, points: np.ndarray) -> None:
        inset = _inset_polygon(points, self.edge_bias_px)
        if inset is not None:
            cv2.fillPoly(self.canvas, [self._fixed(inset)], 255, cv2.LINE_8, _SHIFT)

    def stroke(self, points: np.ndarray, radii: np.ndarray) -> None:
        """Polyline with per-vertex half-widths, round joins and caps."""
        radii = np.maximum(radii - self.edge_bias_px, 0.0)
        for (x, y), radius in zip(self._fixed(points), radii):
            if radius > 0:
                cv2.circle(self.canvas, (int(x), int(y)), self._length(radius), 255, -1, cv2.LINE_8, _SHIFT)
        for quad in self._fixed(_segment_quads(points, radii)):
            cv2.fillConvexPoly(self.canvas, quad, 255, cv2.LINE_8, _SHIFT)

    def ellipse(self, center: np.ndarray, axes_px: Sequence[float], angle_deg: float) -> None:
        x, y = self._fixed(center)
        axes = tuple(self._length(max(a - self.edge_bias_px, 0.0)) for a in axes_px)
        cv2.ellipse(self.canvas, (int(x), int(y)), axes, angle_deg, 0, 360, 255, -1, cv2.LINE_8, _SHIFT)

    def coverage(self) -> np.ndarray:
        """Fraction of each (low-res) pixel covered, float32 (h, w) in [0, 1]."""
        small = cv2.resize(self.canvas, (self.w, self.h), interpolation=cv2.INTER_AREA)
        return small.astype(np.float32) * np.float32(1.0 / 255.0)


def _render(
    roi: Roi,
    feather_px: float,
    extent: tuple[np.ndarray, float],
    draw: Callable[[_Raster], None],
) -> np.ndarray:
    """Rasterize what ``draw`` paints, feather it and return a ROI-sized mask.

    ``extent`` is ``(points, reach)``: everything drawn lies within ``reach``
    pixels of the points. Only that area plus the feather's reach is
    processed; the rest of the mask stays zero.
    """
    mask = np.zeros((roi.h, roi.w), np.float32)
    points, reach = extent
    margin = reach + _FEATHER_REACH * max(feather_px, 0.0) + 1
    # Bounds clipped to [0, roi.x1) x [0, roi.y1); then cut to the ROI's start.
    bounds = roi_around(points, (roi.y1, roi.x1), pad=0.0, margin_px=margin)
    if bounds is None or bounds.x1 <= roi.x0 or bounds.y1 <= roi.y0:
        return mask  # nothing reaches into the ROI
    area = Roi(max(bounds.x0, roi.x0), max(bounds.y0, roi.y0), bounds.x1, bounds.y1)
    downscale = _feather_downscale(feather_px) if feather_px > 0 else 1
    raster = _Raster(area, downscale)
    draw(raster)
    shape = raster.coverage()
    if feather_px > 0:
        shape = cv2.GaussianBlur(shape, (0, 0), feather_px / downscale, borderType=cv2.BORDER_REPLICATE)
    if downscale > 1:
        shape = cv2.resize(shape, (raster.w * downscale, raster.h * downscale), interpolation=cv2.INTER_LINEAR)
    local_y, local_x = area.y0 - roi.y0, area.x0 - roi.x0
    mask[local_y : local_y + area.h, local_x : local_x + area.w] = shape[: area.h, : area.w]
    return np.clip(mask, 0.0, 1.0, out=mask)


def feather(mask: np.ndarray, sigma_px: float) -> np.ndarray:
    """Gaussian-blur a float mask by ``sigma_px`` pixels (``sigma_px <= 0``: unchanged).

    Large sigmas are blurred at reduced resolution (same result to the eye,
    a fraction of the cost). Returns float32 of the same shape in [0, 1].
    """
    if sigma_px <= 0:
        return mask
    src = np.asarray(mask, dtype=np.float32)
    h, w = src.shape[:2]
    downscale = _feather_downscale(sigma_px)
    if downscale == 1:
        out = cv2.GaussianBlur(src, (0, 0), sigma_px, borderType=cv2.BORDER_REPLICATE)
    else:
        small_size = (max(1, round(w / downscale)), max(1, round(h / downscale)))
        small = cv2.resize(src, small_size, interpolation=cv2.INTER_AREA)
        sigma_small = sigma_px * small_size[0] / w
        small = cv2.GaussianBlur(small, (0, 0), sigma_small, borderType=cv2.BORDER_REPLICATE)
        out = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.clip(out, 0.0, 1.0, out=out)


# --------------------------------------------------------------------------
# Generic shape masks (points in full-frame pixel coordinates).
# --------------------------------------------------------------------------


def polygon_mask(points: np.ndarray, roi: Roi, feather_px: float = 0.0) -> np.ndarray:
    """Anti-aliased filled polygon as a ROI-sized float32 mask in [0, 1].

    Coverage is exact to a few percent; slivers thinner than the sampling
    grid (1/4 px, coarser under large feathers) vanish.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    return _render(roi, feather_px, (pts, 0.0), lambda raster: raster.polygon(pts))


def polyline_mask(
    points: np.ndarray,
    thickness_px: float,
    roi: Roi,
    feather_px: float = 0.0,
    thickness_profile: Sequence[float] | np.ndarray | None = None,
) -> np.ndarray:
    """A stroked open polyline with round joins and caps.

    ``thickness_px`` is the full stroke width. ``thickness_profile`` gives
    one width multiplier per vertex (e.g. tapering ``[0.3, 1, 1, 0.1]`` for
    an eyeliner); widths are interpolated linearly along each segment.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    profile = np.ones(len(pts)) if thickness_profile is None else np.asarray(thickness_profile, np.float64)
    if len(profile) != len(pts):
        raise ValueError("thickness_profile needs one value per point")
    if thickness_px <= 0 or len(pts) == 0:
        return np.zeros((roi.h, roi.w), np.float32)
    radii = np.maximum(profile, 0.0) * (thickness_px / 2.0)
    return _render(roi, feather_px, (pts, float(radii.max())), lambda raster: raster.stroke(pts, radii))


def ellipse_mask(
    center_xy: Sequence[float] | np.ndarray,
    axes_xy: Sequence[float],
    angle_deg: float,
    roi: Roi,
    feather_px: float = 0.0,
) -> np.ndarray:
    """Filled ellipse; ``axes_xy`` are the semi-axes (half widths) in pixels,
    ``angle_deg`` rotates it clockwise in image coordinates (cv2 convention)."""
    if min(axes_xy) <= 0:
        return np.zeros((roi.h, roi.w), np.float32)
    center = np.asarray(center_xy, dtype=np.float64).reshape(2)
    extent = (center.reshape(1, 2), float(max(axes_xy)))
    return _render(roi, feather_px, extent, lambda raster: raster.ellipse(center, axes_xy, angle_deg))


# --------------------------------------------------------------------------
# Face-region masks. All return float32 (roi.h, roi.w) masks in [0, 1].
# --------------------------------------------------------------------------


def lips_mask(landmarks: np.ndarray, roi: Roi, outline: float = 0.0, feather_px: float = 0.0) -> np.ndarray:
    """Both lips, without the mouth opening (teeth / tongue / dark gap).

    ``outline`` grows (+, overlining) or shrinks (-) the outer lip line by
    that fraction of the combined lip thickness (upper + lower lip). The
    feather softens only the outer lip line; the edge toward the teeth stays
    crisp so color never spills onto the teeth. With the mouth closed the
    opening has no area and nothing is removed.
    """
    thickness = float(
        np.linalg.norm(landmarks[LIPS_TOP] - landmarks[LIPS_INNER_TOP])
        + np.linalg.norm(landmarks[LIPS_INNER_BOTTOM] - landmarks[LIPS_BOTTOM])
    )
    outer = expand_polygon(landmarks[LIPS_OUTER], outline * thickness)
    lips = polygon_mask(outer, roi, feather_px)
    opening = polygon_mask(landmarks[LIPS_INNER], roi)
    return lips * (1.0 - opening)


def eye_opening_mask(
    landmarks: np.ndarray, side: Side, roi: Roi, expand_px: float = 0.0, feather_px: float = 0.0
) -> np.ndarray:
    """The visible eye (white + iris) between the lash lines of one eye.

    ``expand_px`` grows (+) or shrinks (-) the outline, e.g. to protect the
    lash line from skin effects.
    """
    outline = expand_polygon(landmarks[side_indices(side).eye], expand_px)
    return polygon_mask(outline, roi, feather_px)


def brow_mask(
    landmarks: np.ndarray, side: Side, roi: Roi, thickness: float = 0.0, feather_px: float = 0.0
) -> np.ndarray:
    """One eyebrow.

    ``thickness`` changes the brow height around its center line: +0.5 is
    50% thicker, -0.5 half as thick (clamped at -1). The length is kept.
    """
    s = side_indices(side)
    upper = landmarks[s.brow_upper].astype(np.float32)
    lower = landmarks[s.brow_lower].astype(np.float32)
    center = (upper + lower) / 2
    half = (upper - lower) / 2 * max(1.0 + thickness, 0.0)
    outline = np.vstack([center + half, (center - half)[::-1]])
    return polygon_mask(outline, roi, feather_px)


def face_oval_mask(landmarks: np.ndarray, roi: Roi, expand_px: float = 0.0, feather_px: float = 0.0) -> np.ndarray:
    """The face inside the jaw line and the top of the forehead (includes eyes,
    brows and lips; subtract their masks where needed)."""
    return polygon_mask(expand_polygon(landmarks[FACE_OVAL], expand_px), roi, feather_px)


def under_eye_mask(
    landmarks: np.ndarray, side: Side, roi: Roi, depth: float = 1.0, feather_px: float = 0.0
) -> np.ndarray:
    """The area below one eye where dark circles show (concealer zone).

    Spans from just below the lower lashes down toward the top of the
    cheekbone; ``depth`` (0..1+) is how far down, as a fraction of that
    distance. The eye opening itself is always excluded.
    """
    if depth <= 0:
        return np.zeros((roi.h, roi.w), np.float32)
    s = side_indices(side)
    top = landmarks[s.under_eye_top].astype(np.float32)
    bottom = top + depth * (landmarks[s.under_eye_bottom] - top)
    area = polygon_mask(np.vstack([top, bottom[::-1]]), roi, feather_px)
    return area * (1.0 - eye_opening_mask(landmarks, side, roi))


_LID_SAMPLES = 16


def eyelid_mask(
    landmarks: np.ndarray, side: Side, roi: Roi, height: float, feather_px: float = 0.0
) -> np.ndarray:
    """The upper eyelid from the lash line up toward the brow (eyeshadow zone).

    ``height`` (0..1) is how far up, as a fraction of the lash-to-brow
    distance (0 = nothing, 1 = up to the lower edge of the brow). The upper
    boundary follows the brow, so it fans out toward the brow tail. The
    feather softens the upper boundary; the eye opening is cut out with a
    crisp edge so the lash line keeps full strength.
    """
    if height <= 0:
        return np.zeros((roi.h, roi.w), np.float32)
    s = side_indices(side)
    lash = resample_polyline(landmarks[s.eye_upper], _LID_SAMPLES)
    brow = resample_polyline(landmarks[s.brow_lower], _LID_SAMPLES)
    top = lash + height * (brow - lash)
    # Close the shape along the lower lash line so the feather does not fade
    # the lid right above the lashes; the opening is removed afterwards.
    outline = np.vstack([top, landmarks[s.eye_lower][::-1].astype(np.float32)])
    lid = polygon_mask(outline, roi, feather_px)
    return lid * (1.0 - eye_opening_mask(landmarks, side, roi))
