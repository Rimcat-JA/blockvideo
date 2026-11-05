"""Blush: a soft flush of color on the apples of the cheeks.

Placement: each cheek gets one ellipse in a face-aligned frame. "Outward"
(toward the ear) and "down" follow the eye line, so the blush turns with a
tilted head, and both blushes move outward together with ``offset_x``. The
center is the apple of the cheek (``regions.cheek_center``) moved by the
shape's own offset and the position sliders. Horizontal sizes on each side
are scaled by how wide that cheek looks (apple to face contour, relative to
a frontal face), so with a turned head the far blush narrows with the cheek.
Shapes: round (a circle on the apple), oval (wide, along the cheekbone) and
lifted (narrower, rotated up toward the temple).

The blush stays on the cheeks: the face oval (shrunk) bounds it, the
face-skin segmentation mask removes hair or hands over the cheek when it is
available, and keep-out zones (as soft as the blush itself) protect the eyes
and the center of the face (nose, the area between nose and mouth, lips).

The mask is smooth, so it is rendered on a coarse grid (face scale about
``_GRID_SCALE`` pixels) and upsampled; only the color blend runs at full
resolution.

Color is a mix of multiply and soft light, both relative to the skin
underneath: the cheek keeps its texture and shading, the flush reads as color
in the skin rather than paint on top, and it takes on the scene's lighting (a
warm lamp warms the blush too).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .. import regions
from ..config import BlushSettings, hex_to_bgr
from .base import Effect, FrameContext, register_effect

# All lengths below are fractions of the face scale (inter-ocular distance).

# Softness: Gaussian feather sigma = base + softness slider x range.
_SOFT_BASE = 0.05
_SOFT_RANGE = 0.15
# The position sliders (-0.5..0.5) move the blush by this x slider value.
_OFFSET_RANGE = 0.5
# Distance from the apple of the cheek to the face contour on a frontal face
# (measured 0.42-0.46); a narrower-looking cheek means the head is turned.
_FRONTAL_CHEEK_WIDTH = 0.44
_SQUEEZE_RANGE = (0.3, 1.25)
# Face contour points at the height of the cheekbone, per side.
_CHEEK_CONTOUR: dict[regions.Side, list[int]] = {"right": [234, 93], "left": [454, 323]}

# Where the blush may go. Guard edges are at least _GUARD_FEATHER soft, and
# as soft as _GUARD_SOFTNESS x the blush's own feather. The eye and lip
# guards reach at least _GUARD_REACH guard sigmas beyond the feature, so the
# eyes and lips stay clean however soft the blush is; the face oval is
# shrunk by at least _OVAL_REACH sigmas, so little color passes the contour.
_GUARD_FEATHER = 0.05
_GUARD_SOFTNESS = 0.5
_GUARD_REACH = 2.5
_OVAL_INSET = 0.06
_OVAL_REACH = 1.5
_EYE_GUARD = 0.15
_LIP_GUARD = 0.06
_CENTER_GUARD = 0.05
# Nose (lower bridge and both wings) plus the lips: their convex hull is the
# center of the face, including the skin between nose and mouth.
_CENTER_POINTS: list[int] = (
    regions.NOSE_BRIDGE[2:] + regions.RIGHT_NOSE_WING + regions.LEFT_NOSE_WING + regions.LIPS_OUTER
)
# Face-skin confidence at which a pixel counts as fully skin; a mask that
# covers less than _MIN_SKIN_COVERAGE of the blush is ignored (stale or wrong).
_SKIN_FULL_CONFIDENCE = 0.5
_MIN_SKIN_COVERAGE = 0.35

# Coarse mask grid: face scale in grid pixels (never finer than the frame).
_GRID_SCALE = 24.0
# The feathered mask is negligible beyond this many sigmas.
_FEATHER_REACH = 3.0

# Color: share of multiply in the multiply / soft-light mix.
_MULTIPLY_SHARE = 0.6


@dataclass(frozen=True)
class _Shape:
    """A blush shape in the face-aligned frame of one cheek."""

    axes: tuple[float, float]  # semi-axes along / across the shape, x face scale
    lift_deg: float  # rotation raising the outer end toward the temple
    shift: tuple[float, float]  # center offset (outward, down) from the apple, x face scale


_SHAPES: dict[str, _Shape] = {
    "round": _Shape(axes=(0.26, 0.26), lift_deg=0.0, shift=(0.0, 0.0)),
    "oval": _Shape(axes=(0.38, 0.24), lift_deg=10.0, shift=(0.05, 0.0)),
    "lifted": _Shape(axes=(0.42, 0.19), lift_deg=28.0, shift=(0.12, -0.06)),
}


@dataclass(frozen=True)
class _Ellipse:
    center: np.ndarray  # (x, y) frame pixels
    axes: tuple[float, float]  # semi-axes in pixels
    angle_deg: float  # cv2 convention (clockwise in image coordinates)


@dataclass(frozen=True)
class _Grid:
    """A coarse pixel grid over ``roi``: one grid pixel is ``factor`` x ``factor`` frame pixels."""

    roi: regions.Roi
    factor: int

    @property
    def area(self) -> regions.Roi:
        """The whole grid as a ROI in grid coordinates."""
        return regions.Roi(0, 0, -(-self.roi.w // self.factor), -(-self.roi.h // self.factor))

    def points(self, frame_points: np.ndarray) -> np.ndarray:
        """Frame pixel coordinates -> grid coordinates."""
        offset = np.asarray(self.roi.offset, np.float32)
        return (np.asarray(frame_points, np.float32) - offset) / np.float32(self.factor)

    def downsample(self, frame_mask: np.ndarray) -> np.ndarray:
        """A full-frame mask -> grid resolution (area average)."""
        area = self.area
        return cv2.resize(frame_mask[self.roi.slices], (area.w, area.h), interpolation=cv2.INTER_AREA)

    def upsample(self, mask: np.ndarray) -> np.ndarray:
        """A grid mask -> ROI resolution (bilinear, pixel centers aligned)."""
        if self.factor == 1:
            return mask
        size = (mask.shape[1] * self.factor, mask.shape[0] * self.factor)
        return cv2.resize(mask, size, interpolation=cv2.INTER_LINEAR)[: self.roi.h, : self.roi.w]


@register_effect
class BlushEffect(Effect):
    """Cheek blush placed on the apples of the cheeks."""

    name = "blush"
    order = 60

    def apply(self, frame: np.ndarray, ctx: FrameContext, settings: BlushSettings) -> np.ndarray:
        opacity = _clamp01(settings.opacity) * ctx.intensity
        if opacity <= 0.0 or ctx.landmarks is None:
            return frame
        shape = _shape(settings.shape)
        landmarks = ctx.landmarks
        scale = regions.face_scale(landmarks)
        feather_px = scale * (_SOFT_BASE + _SOFT_RANGE * _clamp01(settings.softness))
        ellipses = [_cheek_ellipse(landmarks, side, shape, settings, scale) for side in regions.SIDES]
        roi = _blush_roi(ellipses, frame.shape, feather_px)
        if roi is None:
            return frame
        grid = _Grid(roi, max(1, int(scale // _GRID_SCALE)))
        mask = _blush_mask(grid, ellipses, feather_px)
        mask *= _allowed_area(grid, landmarks, scale, feather_px, ctx.face_skin_mask, mask)
        base = frame[roi.slices]
        frame[roi.slices] = _flush(base, hex_to_bgr(settings.color), grid.upsample(mask) * np.float32(opacity))
        return frame


def _clamp01(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


def _shape(name: str) -> _Shape:
    try:
        return _SHAPES[name]
    except KeyError:
        raise ValueError(f"unknown blush shape {name!r}; expected one of {sorted(_SHAPES)}") from None


def _face_axes(landmarks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unit vectors (across, down): across runs from the subject's right eye to
    the left eye (image left to right), down is perpendicular toward the chin."""
    across = regions.eye_center(landmarks, "left") - regions.eye_center(landmarks, "right")
    across /= max(float(np.linalg.norm(across)), 1e-6)
    return across, np.array([-across[1], across[0]], np.float32)


def _cheek_squeeze(landmarks: np.ndarray, side: regions.Side, outward: np.ndarray, scale: float) -> float:
    """How wide this cheek looks relative to a frontal face (< 1: turned away)."""
    contour = landmarks[_CHEEK_CONTOUR[side]].astype(np.float32).mean(axis=0)
    width = float(np.dot(contour - regions.cheek_center(landmarks, side), outward))
    return float(np.clip(width / (_FRONTAL_CHEEK_WIDTH * scale), *_SQUEEZE_RANGE))


def _cheek_ellipse(
    landmarks: np.ndarray, side: regions.Side, shape: _Shape, settings: BlushSettings, scale: float
) -> _Ellipse:
    across, down = _face_axes(landmarks)
    outward = across if side == "left" else -across
    squeeze = _cheek_squeeze(landmarks, side, outward, scale)
    # Linear map from the cheek frame (outward, down) in face-scale units to frame pixels.
    to_frame = np.column_stack([outward * squeeze, down]) * scale
    shift = np.array(shape.shift) + _OFFSET_RANGE * np.array([settings.offset_x, settings.offset_y])
    center = regions.cheek_center(landmarks, side) + to_frame @ shift
    lift = math.radians(shape.lift_deg)
    # Columns: the shape's long axis (outward and up) and short axis, scaled by the semi-axes.
    axes = np.array([[math.cos(lift), math.sin(lift)], [-math.sin(lift), math.cos(lift)]]) * (
        np.array(shape.axes) * settings.size
    )
    # The image of a circle under a linear map is an ellipse: its semi-axes are
    # the singular values, its directions the left singular vectors.
    directions, semi_axes, _ = np.linalg.svd(to_frame @ axes)
    angle = math.degrees(math.atan2(directions[1, 0], directions[0, 0]))
    return _Ellipse(center.astype(np.float32), (float(semi_axes[0]), float(semi_axes[1])), angle)


def _blush_roi(ellipses: list[_Ellipse], shape: tuple[int, ...], feather_px: float) -> regions.Roi | None:
    centers = np.array([e.center for e in ellipses])
    reach = max(max(e.axes) for e in ellipses) + _FEATHER_REACH * feather_px
    return regions.roi_around(centers, shape, pad=0.0, margin_px=reach)


def _blush_mask(grid: _Grid, ellipses: list[_Ellipse], feather_px: float) -> np.ndarray:
    """Both feathered ellipses on the grid."""
    f = grid.factor
    mask = np.zeros((grid.area.h, grid.area.w), np.float32)
    for e in ellipses:
        axes = (e.axes[0] / f, e.axes[1] / f)
        ellipse = regions.ellipse_mask(grid.points(e.center), axes, e.angle_deg, grid.area, feather_px / f)
        np.maximum(mask, ellipse, out=mask)
    return mask


def _allowed_area(
    grid: _Grid,
    landmarks: np.ndarray,
    scale: float,
    feather_px: float,
    face_skin_mask: np.ndarray | None,
    blush: np.ndarray,
) -> np.ndarray:
    """Where blush may go (on the grid): inside the face, on skin, away from the
    eyes, the lips and the center of the face."""
    points = grid.points(landmarks)
    unit = scale / grid.factor  # face scale in grid pixels
    soft = max(_GUARD_FEATHER * unit, _GUARD_SOFTNESS * feather_px / grid.factor)
    inset = max(_OVAL_INSET * unit, _OVAL_REACH * soft)
    allowed = regions.face_oval_mask(points, grid.area, expand_px=-inset, feather_px=soft)
    if face_skin_mask is not None:
        allowed *= _skin_weight(grid.downsample(face_skin_mask), blush)
    eye_guard = max(_EYE_GUARD * unit, _GUARD_REACH * soft)
    lips = regions.expand_polygon(points[regions.LIPS_OUTER], max(_LIP_GUARD * unit, _GUARD_REACH * soft))
    center = regions.expand_polygon(cv2.convexHull(points[_CENTER_POINTS]).reshape(-1, 2), _CENTER_GUARD * unit)
    keep_out = [
        *(regions.eye_opening_mask(points, side, grid.area, eye_guard, soft) for side in regions.SIDES),
        regions.polygon_mask(lips, grid.area, soft),
        regions.polygon_mask(center, grid.area, soft),
    ]
    for zone in keep_out:
        allowed *= 1.0 - zone
    return allowed


def _skin_weight(face_skin: np.ndarray, blush: np.ndarray) -> np.ndarray | float:
    """Face-skin confidence as a 0..1 weight, or 1 when the mask misses the cheeks."""
    weight = np.clip(face_skin * (1.0 / _SKIN_FULL_CONFIDENCE), 0.0, 1.0)
    total = float(blush.sum())
    if total <= 0 or float((weight * blush).sum()) < _MIN_SKIN_COVERAGE * total:
        return 1.0
    return weight


def _flush(base: np.ndarray, color_bgr: tuple[float, float, float], weight: np.ndarray) -> np.ndarray:
    """Multiply / soft-light mix of the blush color, weighted per pixel.

    Equal to ``mix(base, mix(blend_soft_light(base, c), blend_multiply(base, c),
    k), weight)`` in closed form: soft light ``b * (b + 2c(1 - b))`` and
    multiply ``b * c`` mixed by ``k`` differ from ``b`` by ``b * (A b + B)``
    with ``A = (1 - k)(1 - 2c)`` and ``B = (2 - k) c - 1`` per channel.
    ``A b + B`` is one ``cv2.transform``; with OpenCV arithmetic this is about
    3x faster than NumPy. Exactly ``base`` where the weight is 0.
    """
    k = _MULTIPLY_SHARE
    color = np.asarray(color_bgr, np.float32)
    affine = np.hstack([np.diag((1 - k) * (1 - 2 * color)), ((2 - k) * color - 1)[:, None]]).astype(np.float32)
    delta = cv2.multiply(cv2.transform(base, affine), base)
    delta = cv2.multiply(delta, cv2.cvtColor(weight, cv2.COLOR_GRAY2BGR))
    return cv2.add(delta, base)
