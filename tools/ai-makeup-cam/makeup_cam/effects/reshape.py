"""Face shape correction: a small, smooth geometric warp of the real pixels.

The warp is one displacement field, built fresh for every frame from the
landmarks in five steps:

1. *Forward field.* Each slider adds a smooth displacement ``D`` on a coarse
   grid over the head: where the pixel now at ``x`` should go (``x + D(x)``).
   The shapes are closed-form windows placed and sized by groups of landmarks
   (see the ``_add_*`` functions), so the field is low-frequency by
   construction: no window follows a single jittering landmark, and there is
   no interpolation matrix that could ring or amplify noise.
2. *Teeth pin.* ``D`` is forced to zero inside the mouth opening, so whatever
   the sliders do, the teeth stay exactly where they are: the lips and the
   mouth corners slide over them and never squeeze them.
3. *Backward map.* ``D`` is inverted on the grid by fixed-point iteration,
   ``b(p) = -D(p + b(p))``, which gives ``out(p) = in(p + b(p))`` with
   ``out(x + D(x)) = in(x)``: every point lands where ``D`` sends it.
4. *Background protection.* ``b`` is faded out where the output pixel and its
   source are both outside the person (segmentation masks when available, the
   face oval and the neck otherwise), according to ``protect_background``.
5. *Apply.* The grid is upsampled and ``cv2.remap`` runs on the bounding box
   of the non-zero field. Masks are warped with the same map, and
   ``ctx.landmarks`` is replaced by the warped positions (found by inverting
   the final map), so every later effect (lips, liner, ...) stays aligned with
   the reshaped face.

Empty-room plate (``ctx.background``): the warped person is cut out and put over
a room that does not bend at all, the unwarped frame where the room was visible
anyway and the plate where the person used to be. This is all or nothing on
purpose: a partial mix of the bent and the straight room would show every line
twice. While the plate does not match the room any more (camera moved, light
changed) the warp falls back to bending the room next to the person.

Without a plate, slimming the face has to stretch the room next to the jaw. On a plain
wall that cannot be seen, but a door frame or a shelf right next to the jaw would bend. So
while there are such lines next to the jaw, slimming (only that) fades to ``_BUSY_ROOM_FLOOR``:
slowly, so the face never visibly changes shape, and more slowly back when they are gone.

Limits: displacements are capped at ``_MAX_FIELD`` of the face width with all
sliders at their ends, and fade out near the frame's edges, so the warp never
reads past them. Poses beyond the interview envelope (a deep bow, a turn past
about 40 degrees) fade the warp out, because the landmarks that drive it become
unreliable there. A one-frame tracking dropout does not change the warp's
strength at all: the pipeline's fade works for colors, but on geometry it would
be a visible twitch. With every slider at 0, or at intensity 0, the frame object
comes back untouched and the context is not modified.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import cv2
import numpy as np

from .. import regions
from ..config import ReshapeSettings
from .base import Effect, FrameContext, register_effect

# All lengths are fractions of the face scale (inter-ocular distance, "IOD")
# unless noted otherwise. A frontal face is about 2.2 IOD wide.

# Hard limit of the displacement length with everything on, x face width. Soft
# above _MAX_FIELD * _CAP_KNEE, so the limit cannot be seen as a plateau.
_MAX_FIELD = 0.08
_CAP_KNEE = 0.7
# The field never compresses or stretches by more than this (x its own length
# per pixel): a guard against odd geometry, far from what normal faces reach.
_MAX_GRADIENT = 0.85
# The field fades to zero over this distance from the frame's edges (x face
# width): more than 3 x _MAX_FIELD, so even the longest field never reads past
# the edge (it would smear the outermost pixels) nor gets close to folding there.
_EDGE_FADE = 0.25
# One step of the pipeline's fade (pipeline.FADE_FRAMES = 6) in the warp's
# intensity is forgiven: the intensity drop of a one-frame dropout changes nothing.
_DROPOUT_HOLD = 6.0 / 5.0
# Above the intensity that the pipeline asks for (master slider lowered): closes this fraction per frame.
_INTENSITY_RELEASE = 0.5

# Coarse grid spacing (x face scale, then limited in pixels).
_GRID_STEP = 0.05
_GRID_STEP_PX = (2, 12)
# The step only changes when it is off by more than this ratio (hysteresis:
# a face scale near a rounding boundary must not make the grid flip per frame).
_GRID_STEP_SLACK = 1.2
# ROI: the face oval grown by this fraction of its larger side.
_ROI_PAD = 0.34
# Fixed-point iterations inverting the field on the grid.
_INVERSION_STEPS = 6
# Landmarks through the final map: at most this many Newton steps, stopping
# below this error (pixels).
_LANDMARK_NEWTON_STEPS = 3
_LANDMARK_TOLERANCE_PX = 0.01
# Light blur of the assembled field: removes the kinks where windows start and end.
_FIELD_BLUR = 0.035
# Teeth pin: feather of the mouth opening, and a gain that makes the pin
# complete a little inside the inner lip line.
_PIN_FEATHER = 0.05
_PIN_GAIN = 1.5
# Field lengths below this (pixels) are treated as zero when bounding the warp.
_ACTIVE_FIELD_PX = 0.02
# Smaller faces (pixels of face scale) are left alone.
_MIN_SCALE_PX = 12.0
# Cubic keeps about 97 % of the high-frequency energy where linear keeps 87 %,
# at about the same cost; it can overshoot, so the result is clipped.
_INTERPOLATION = cv2.INTER_CUBIC

# face_slim / jaw: contour moved toward the oval's center at slider 1.0, as a
# fraction of the contour's horizontal distance from that center.
_SLIM = 0.12
_JAW = 0.07
# Radial profile: u = distance from the oval's center / oval radius there.
_CONTOUR_INNER_U = 0.5  # the field starts here...
_CONTOUR_OUTER_U = 1.55  # ...peaks at the contour (u = 1) and ends here
# Slimming starts at the eye line and is complete at this fraction of the
# eye line -> chin distance (the temples stay, the cheeks and jaw move).
_SLIM_RISE = (0.05, 0.62)
# The jaw window around the gonial angle, half height.
_JAW_RADIUS = 0.55
_JAW_POINTS = (172, 397)
# The contour field keeps away from the mouth (r = elliptical mouth radius).
_MOUTH_GUARD_R = (0.85, 1.6)

_CHIN = 0.085
_CHIN_POINT = 152
_CHIN_WINDOW_X = 1.7  # half width of the moving part: wide, so the jaw line follows the chin without a notch
_CHIN_FLOOR = 0.8  # the field ends this far below the chin
_CHIN_START = 0.05  # starts this fraction of the lower lip -> chin distance below the lip

# eye_size: scale change of the eye region at slider 1.0; plateau and outer
# radius of the window as multiples of the eye's width. Toward the nose the
# window ends on the eye-to-eye midline, so the two eyes never pull the nose
# bridge from both sides.
_EYE_SCALE = 0.13
_EYE_RADIUS = (1.3, 1.05)
_EYE_PLATEAU_R2 = 0.2
_EYE_MIDLINE_MARGIN = 0.92

# nose_width: scale change around the nose's center line; window radii in
# multiples of the nose half width, up in IOD, down limited by the upper lip.
_NOSE_NARROW = 0.2
_NOSE_RADIUS_X = 2.6
_NOSE_RADIUS_UP = 0.55
_NOSE_WINGS = (64, 294)
_NOSE_LIP_GAP = 0.9
# nose_length: displacement of the nose tip and base.
_NOSE_SHORTEN = 0.045
_NOSE_LENGTH_RADIUS_X = 1.9

# mouth_width: corners moved toward the center (x half mouth width). Only the
# outer part of the mouth moves (from _MOUTH_INNER_U to the corner at u = 1),
# then the field fades out into the cheek by _MOUTH_OUTER_U.
_MOUTH_NARROW = 0.11
_MOUTH_INNER_U = 0.35
_MOUTH_OUTER_U = 1.9
_MOUTH_WINDOW_Y = 0.3  # window half height beyond the lips
# mouth_corner: lift at slider 1.0; window radii (toward the center x half
# mouth width, toward the cheek and vertically in IOD).
_CORNER_LIFT = 0.055
_CORNER_RADIUS_IN = 0.6
_CORNER_RADIUS_OUT = 0.6
_CORNER_RADIUS_Y = 0.36

# lip_fullness: outer lip line moved outward, x lip thickness.
_LIP_GROW = 0.28
_LIP_FADE = 0.2  # the field ends this far (IOD) beyond the outer lip line
_LIP_MIN_THICKNESS = 0.02

# brow_lift: displacement of the brow; window radii (x half brow length, up in
# IOD, down as a fraction of the brow -> eye center distance).
_BROW_LIFT = 0.06
_BROW_RADIUS_X = 1.5
_BROW_RADIUS_UP = 0.6
_BROW_RADIUS_DOWN = 0.82

# Background protection: inside the oval (u below these) is always "person";
# the transition is further softened by _PROTECT_BLUR.
_PROTECT_OVAL_U = (1.05, 1.2)
_PROTECT_MASK_RAMP = (0.15, 0.55)
_PROTECT_BLUR = 0.1
_PROTECT_NECK_X = 0.95

# Background plate. The warped person is cut out at its mask's middle; where the
# person was, the plate takes over from the mask's first traces, so no ghost of
# the old outline stays behind. Below these field lengths (pixels) nothing moves
# visibly, so the picture stays as it is, with the real soft edge of the person.
_PLATE_OVAL_U = (0.95, 1.0)
_PLATE_MOVED_RAMP = (0.2, 0.8)
_PLATE_SOURCE_RAMP = (0.03, 0.3)
_PLATE_ACTIVE_PX = (0.05, 0.3)
# A pixel also counts as the person when the mean difference of its channels from
# the plate is above this ramp, plus the usual difference in the room (camera noise).
_PLATE_FOREGROUND = (0.05, 0.12)
_CHANNEL_MEAN = np.float32([[1.0, 1.0, 1.0]]) / 3.0
_NEIGHBORS = np.ones((3, 3), np.uint8)
# Does the plate still match? Live and plate pixels are compared on grid nodes (about
# _PLATE_STATS_PX apart, closer on edges) that surely show the room: below this person
# mask, outside this oval radius, and above the jaw, because the masks have no class
# for clothes. Exposure drift up to _PLATE_GAIN_RANGE is corrected. The plate is
# dropped above the first difference (smoothed over time) and used again after
# some time below the second, each time with a short fade. The difference is the
# larger of a high percentile of all samples and of those on an edge (_PLATE_EDGE: mean
# difference from the neighborhood), which notice a moved camera in a mostly plain room.
_PLATE_ROOM_BELOW = 0.1
_PLATE_ROOM_OVAL_U = 1.05
_PLATE_STATS_PX = 12.0
_PLATE_MIN_SAMPLES = 16
_PLATE_GAIN_RANGE = (0.75, 1.3)
_PLATE_MISMATCH_PERCENTILE = 75.0
_PLATE_EDGE = 0.08
_PLATE_MIN_EDGE_SAMPLES = 8
_PLATE_MISMATCH_OFF = 0.09
_PLATE_MISMATCH_ON = 0.06
_PLATE_RETURN_FRAMES = 60
_PLATE_ADAPT = 0.1
_PLATE_FADE_FRAMES = 15

# Room next to the jaw, without a plate: the nodes that surely show the room between these oval
# radii, from the eye line to the chin. Pixels there on an edge (difference from their neighborhood above
# _BUSY_ROOM_EDGE) as a share of them: slimming is at full strength below the first value and at the
# floor above the second. Measured every _BUSY_ROOM_EVERY frames; changes by these steps per frame.
_BUSY_ROOM_OVAL_U = (1.05, 1.5)
_BUSY_ROOM_EDGE = 0.035
_BUSY_ROOM_SHARE = (0.01, 0.03)
_BUSY_ROOM_FLOOR = 0.25
_BUSY_ROOM_EVERY = 8
_BUSY_ROOM_MIN_NODES = 16
_BUSY_ROOM_FALL = 1.0 / 45.0
_BUSY_ROOM_RISE = 1.0 / 150.0
_SLIM_SETTLED_INTENSITY = 0.5  # below this the warp is still fading in: the level may jump

# Pose fade: fully on up to the first value, off from the second. Yaw: left/right
# asymmetry of the nose in the face outline. Pitch: nose tip position between the
# eye line (0) and the chin (1). Calibrated on real 3D landmarks rotated in software:
# no fade at yaw 20 + pitch 15 + roll 10 for any test face; the fade starts at about
# yaw 40, 20 degrees down and 45 degrees up, and ends at about yaw 50-65, 40 degrees
# down and 55 degrees up.
_YAW_FADE = (0.66, 0.95)
_PITCH_FADE_DOWN = (0.5, 0.72)
_PITCH_FADE_UP = (0.08, -0.1)
_POSE_ADAPT = 0.2


@dataclass(frozen=True)
class _Amounts:
    """Slider values (with the master intensity applied)."""

    face_slim: float
    jaw: float
    chin_length: float
    eye_size: float
    nose_width: float
    nose_length: float
    mouth_width: float
    mouth_corner: float
    lip_fullness: float
    brow_lift: float

    @classmethod
    def of(cls, settings: ReshapeSettings, intensity: float) -> "_Amounts":
        level = max(float(intensity), 0.0)

        def value(raw: float) -> float:
            return min(max(float(raw), -1.0), 1.0) * level

        return cls(
            value(settings.face_slim),
            value(settings.jaw),
            value(settings.chin_length),
            value(settings.eye_size),
            value(settings.nose_width),
            value(settings.nose_length),
            value(settings.mouth_width),
            max(value(settings.mouth_corner), 0.0),
            value(settings.lip_fullness),
            value(settings.brow_lift),
        )

    def any(self) -> bool:
        return any(getattr(self, name) != 0.0 for name in self.__dataclass_fields__)

    def slims(self) -> bool:
        return self.face_slim != 0.0 or self.jaw != 0.0

    def scaled(self, factor: float) -> "_Amounts":
        return _Amounts(*(getattr(self, name) * factor for name in self.__dataclass_fields__))


@dataclass(frozen=True)
class _Face:
    """Landmarks in a face-aligned frame: X toward the subject's left eye
    (image right), Y toward the chin, origin at the center of the face oval."""

    landmarks: np.ndarray
    origin: np.ndarray
    ex: np.ndarray
    ey: np.ndarray
    x: np.ndarray  # (N,) face-frame coordinates of the landmarks
    y: np.ndarray
    scale: float  # inter-ocular distance (regions.face_scale)
    width: float  # face width (outermost oval points)


def _face_frame(landmarks: np.ndarray) -> _Face | None:
    """The face frame, or None for unusable landmarks (non-finite, too small)."""
    if not np.isfinite(landmarks).all():
        return None
    scale = float(regions.face_scale(landmarks))
    if scale < _MIN_SCALE_PX:
        return None
    axis = regions.eye_center(landmarks, "left") - regions.eye_center(landmarks, "right")
    length = float(np.hypot(axis[0], axis[1]))
    if length < 1e-3:
        return None
    ex = axis / length
    ey = np.array([-ex[1], ex[0]], np.float32)
    origin = landmarks[regions.FACE_OVAL].mean(axis=0).astype(np.float32)
    relative = landmarks - origin
    x = relative @ ex
    y = relative @ ey
    width = float(x[regions.FACE_OVAL].max() - x[regions.FACE_OVAL].min())
    if width < 1e-3:
        return None
    return _Face(landmarks, origin, ex, ey, x, y, scale, width)


def _smoothstep(lo: float, hi: float, value: np.ndarray) -> np.ndarray:
    t = np.clip((value - lo) / (hi - lo), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _bump(r2: np.ndarray) -> np.ndarray:
    """Compact bell over the squared radius: 1 at 0, zero (with zero slope) at 1."""
    t = np.clip(1.0 - r2, 0.0, 1.0)
    return t * t


def _ramp_peak(lo: float, peak: float, hi: float, value: np.ndarray) -> np.ndarray:
    """0 below ``lo``, smoothly up to 1 at ``peak``, smoothly down to 0 at ``hi``."""
    return _smoothstep(lo, peak, value) * (1.0 - _smoothstep(peak, hi, value))


class _Grid:
    """Coarse grid over the ROI. Node ``k`` sits at ``x0 + (k + 0.5) * step`` in
    continuous frame coordinates, which is what ``cv2.resize`` by ``step`` expects."""

    def __init__(self, face: _Face, roi: regions.Roi, step: int) -> None:
        self.face = face
        self.roi = roi
        self.step = step
        self.nx = -(-roi.w // step)
        self.ny = -(-roi.h // step)
        xs = (roi.x0 + (np.arange(self.nx, dtype=np.float32) + 0.5) * step) - face.origin[0]
        ys = (roi.y0 + (np.arange(self.ny, dtype=np.float32) + 0.5) * step) - face.origin[1]
        self.xs = xs
        self.ys = ys
        self.X = xs[None, :] * face.ex[0] + ys[:, None] * face.ex[1]
        self.Y = xs[None, :] * face.ey[0] + ys[:, None] * face.ey[1]
        self.dx = np.zeros((self.ny, self.nx), np.float32)
        self.dy = np.zeros((self.ny, self.nx), np.float32)
        self._u: np.ndarray | None = None
        self._radius: np.ndarray | None = None

    def sample(self, image: np.ndarray) -> np.ndarray:
        """The pixels of a full-frame picture at the nodes."""
        roi, step = self.roi, self.step
        ys = np.minimum(roi.y0 + step // 2 + step * np.arange(self.ny), roi.y1 - 1)
        xs = np.minimum(roi.x0 + step // 2 + step * np.arange(self.nx), roi.x1 - 1)
        return image[np.ix_(ys, xs)]

    def patch(self, x: float, y: float, reach: float) -> tuple[slice, slice]:
        """Slices of the nodes within ``reach`` px (per axis) of face-frame point (x, y)."""
        face = self.face
        px = x * face.ex[0] + y * face.ey[0]
        py = x * face.ex[1] + y * face.ey[1]
        i0 = int(np.searchsorted(self.xs, px - reach))
        i1 = int(np.searchsorted(self.xs, px + reach, side="right"))
        j0 = int(np.searchsorted(self.ys, py - reach))
        j1 = int(np.searchsorted(self.ys, py + reach, side="right"))
        return slice(j0, j1), slice(i0, i1)

    @property
    def u(self) -> np.ndarray:
        """Distance from the oval's center / oval radius in that direction, per node."""
        if self._u is None:
            face = self.face
            ox, oy = face.x[regions.FACE_OVAL], face.y[regions.FACE_OVAL]
            angles = np.arctan2(oy, ox)
            order = np.argsort(angles)
            angles = angles[order]
            radii = np.hypot(ox, oy)[order]
            radii = (np.roll(radii, 1) + 2.0 * radii + np.roll(radii, -1)) / 4.0
            table_angles = np.concatenate(([angles[-1] - 2.0 * np.pi], angles, [angles[0] + 2.0 * np.pi]))
            table_radii = np.concatenate(([radii[-1]], radii, [radii[0]]))
            radius = np.interp(np.arctan2(self.Y, self.X), table_angles, table_radii).astype(np.float32)
            self._u = np.hypot(self.X, self.Y) / np.maximum(radius, 1e-3)
        return self._u


# --------------------------------------------------------------------------
# Forward field, one function per slider group. Everything is in the face
# frame and adds to grid.dx / grid.dy.
# --------------------------------------------------------------------------


def _mouth_box(face: _Face) -> tuple[float, float, float, float]:
    """Mouth center (x, y) and half width / half height of the outer lip line."""
    x, y = face.x, face.y
    right, left = regions.MOUTH_RIGHT_CORNER, regions.MOUTH_LEFT_CORNER
    half_width = max(abs(float(x[left] - x[right])) / 2.0, 1.0)
    center_x = float(x[left] + x[right]) / 2.0
    top = float(min(y[regions.LIPS_TOP], y[left], y[right]))
    bottom = float(max(y[regions.LIPS_BOTTOM], y[left], y[right]))
    return center_x, (top + bottom) / 2.0, half_width, max((bottom - top) / 2.0, 1.0)


def _add_contour(grid: _Grid, amounts: _Amounts) -> None:
    """face_slim and jaw: the lower contour moves toward the oval's center."""
    if amounts.face_slim == 0.0 and amounts.jaw == 0.0:
        return
    face = grid.face
    eye_y = float(face.y[[regions.RIGHT_EYE_INNER_CORNER, regions.RIGHT_EYE_OUTER_CORNER,
                          regions.LEFT_EYE_INNER_CORNER, regions.LEFT_EYE_OUTER_CORNER]].mean())
    chin_y = float(face.y[_CHIN_POINT])
    span = max(chin_y - eye_y, 1.0)
    level = np.zeros_like(grid.Y)
    if amounts.face_slim != 0.0:
        level += (amounts.face_slim * _SLIM) * _smoothstep(
            eye_y + _SLIM_RISE[0] * span, eye_y + _SLIM_RISE[1] * span, grid.Y
        )
    if amounts.jaw != 0.0:
        jaw_y = float(face.y[list(_JAW_POINTS)].mean())
        radius = _JAW_RADIUS * face.scale
        level += (amounts.jaw * _JAW) * _bump(((grid.Y - jaw_y) / radius) ** 2)
    u = grid.u
    profile = _ramp_peak(_CONTOUR_INNER_U, 1.0, _CONTOUR_OUTER_U, u)
    # X / u is the horizontal position of the contour along this node's ray.
    move = level * profile * grid.X / np.maximum(u, 0.25)
    mx, my, half_w, half_h = _mouth_box(face)
    mouth_r = np.sqrt(((grid.X - mx) / half_w) ** 2 + ((grid.Y - my) / (half_h + 0.1 * face.scale)) ** 2)
    move *= _smoothstep(_MOUTH_GUARD_R[0], _MOUTH_GUARD_R[1], mouth_r)
    grid.dx -= move


def _add_chin(grid: _Grid, amount: float) -> None:
    """chin_length: the chin moves down (+) or up (-); the lips stay."""
    if amount == 0.0:
        return
    face = grid.face
    chin_x, chin_y = float(face.x[_CHIN_POINT]), float(face.y[_CHIN_POINT])
    lip_y = float(face.y[regions.LIPS_BOTTOM])
    start = lip_y + _CHIN_START * max(chin_y - lip_y, 0.0)
    floor = chin_y + _CHIN_FLOOR * face.scale
    profile = _ramp_peak(start, chin_y, max(floor, chin_y + 1.0), grid.Y)
    window = _bump(((grid.X - chin_x) / (_CHIN_WINDOW_X * face.scale)) ** 2)
    grid.dy += (amount * _CHIN * face.scale) * profile * window


def _add_eye(grid: _Grid, side: regions.Side, amount: float) -> None:
    """eye_size: scale around the middle of the eye corners, fading out toward
    the brow and the cheek. The iris scales with the eye and stays round."""
    face = grid.face
    s = regions.side_indices(side)
    inner, outer = s.eye_upper[0], s.eye_upper[-1]
    cx = float(face.x[inner] + face.x[outer]) / 2.0
    cy = float(face.y[inner] + face.y[outer]) / 2.0
    eye_width = max(float(np.hypot(face.x[inner] - face.x[outer], face.y[inner] - face.y[outer])), 1.0)
    rx, ry = _EYE_RADIUS[0] * eye_width, _EYE_RADIUS[1] * eye_width
    midline = float(face.x[[regions.RIGHT_EYE_INNER_CORNER, regions.LEFT_EYE_INNER_CORNER]].mean())
    to_nose = max(_EYE_MIDLINE_MARGIN * abs(cx - midline), 1.0)
    area = grid.patch(cx, cy, max(rx, ry) * 1.42)
    rel_x = grid.X[area] - cx
    rel_y = grid.Y[area] - cy
    toward_nose = rel_x * math.copysign(1.0, midline - cx) > 0.0
    r2 = (rel_x / np.where(toward_nose, min(to_nose, rx), rx)) ** 2 + (rel_y / ry) ** 2
    window = 1.0 - _smoothstep(_EYE_PLATEAU_R2, 1.0, r2)
    scale = amount * _EYE_SCALE
    grid.dx[area] += scale * rel_x * window
    grid.dy[area] += scale * rel_y * window


def _add_nose(grid: _Grid, amounts: _Amounts) -> None:
    """nose_width: squeeze the wings toward the nose's center line. nose_length:
    lift the tip and the base. The window ends before the upper lip."""
    if amounts.nose_width == 0.0 and amounts.nose_length == 0.0:
        return
    face = grid.face
    right, left = _NOSE_WINGS
    cx = float(face.x[right] + face.x[left]) / 2.0
    half = max(abs(float(face.x[left] - face.x[right])) / 2.0, 1.0)
    tip_y = float(face.y[regions.NOSE_TIP] + face.y[regions.NOSE_BOTTOM]) / 2.0
    up = _NOSE_RADIUS_UP * face.scale
    down = max(_NOSE_LIP_GAP * (float(face.y[regions.LIPS_TOP]) - tip_y), 1.0)
    reach = max(_NOSE_RADIUS_X * half, up, down) * 1.42
    area = grid.patch(cx, tip_y, reach)
    rel_x = grid.X[area] - cx
    rel_y = grid.Y[area] - tip_y
    y_term = (rel_y / np.where(rel_y < 0.0, up, down)) ** 2
    if amounts.nose_width != 0.0:
        window = _bump((rel_x / (_NOSE_RADIUS_X * half)) ** 2 + y_term)
        grid.dx[area] -= (amounts.nose_width * _NOSE_NARROW) * rel_x * window
    if amounts.nose_length != 0.0:
        window = _bump((rel_x / (_NOSE_LENGTH_RADIUS_X * half)) ** 2 + y_term)
        grid.dy[area] -= (amounts.nose_length * _NOSE_SHORTEN * face.scale) * window


def _add_mouth_width(grid: _Grid, amount: float) -> None:
    """mouth_width: the corners move toward the center, the middle stays."""
    if amount == 0.0:
        return
    face = grid.face
    mx, my, half_w, half_h = _mouth_box(face)
    reach_y = half_h + _MOUTH_WINDOW_Y * face.scale
    area = grid.patch(mx, my, max(_MOUTH_OUTER_U * half_w, reach_y) * 1.42)
    u = (grid.X[area] - mx) / half_w
    s = np.abs(u)
    window = _bump(((grid.Y[area] - my) / reach_y) ** 2)
    grid.dx[area] -= (amount * _MOUTH_NARROW * half_w) * np.sign(u) * _ramp_peak(_MOUTH_INNER_U, 1.0, _MOUTH_OUTER_U, s) * window


def _add_mouth_corners(grid: _Grid, amount: float) -> None:
    """mouth_corner: lift each corner; the middle of the mouth stays."""
    if amount == 0.0:
        return
    face = grid.face
    center_x, _, half_w, _ = _mouth_box(face)
    lift = amount * _CORNER_LIFT * face.scale
    inward = _CORNER_RADIUS_IN * half_w
    outward = _CORNER_RADIUS_OUT * face.scale
    ry = _CORNER_RADIUS_Y * face.scale
    for corner in (regions.MOUTH_LEFT_CORNER, regions.MOUTH_RIGHT_CORNER):
        cx, cy = float(face.x[corner]), float(face.y[corner])
        area = grid.patch(cx, cy, max(inward, outward, ry) * 1.42)
        rel_x = (grid.X[area] - cx) * math.copysign(1.0, cx - center_x)  # + = toward the cheek
        r2 = (rel_x / np.where(rel_x < 0.0, inward, outward)) ** 2 + ((grid.Y[area] - cy) / ry) ** 2
        grid.dy[area] -= lift * _bump(r2)


def _column(points_x: np.ndarray, points_y: np.ndarray, x: np.ndarray) -> np.ndarray:
    order = np.argsort(points_x)
    return np.interp(x, points_x[order], points_y[order]).astype(np.float32)


def _add_lips(grid: _Grid, amount: float) -> None:
    """lip_fullness: the outer lip lines move outward, the lines toward the
    teeth stay. The amount follows each lip's own thickness."""
    if amount == 0.0:
        return
    face = grid.face
    mx, my, half_w, half_h = _mouth_box(face)
    fade = _LIP_FADE * face.scale
    area = grid.patch(mx, my, max(half_w, half_h + fade * 2.0) * 1.42)
    X, Y = grid.X[area], grid.Y[area]
    taper = _bump(((X - mx) / half_w) ** 2)
    min_thickness = _LIP_MIN_THICKNESS * face.scale
    for inner_line, outer_line, sign in (
        (regions.LIPS_UPPER_INNER, regions.LIPS_UPPER_OUTER, -1.0),
        (regions.LIPS_LOWER_INNER, regions.LIPS_LOWER_OUTER, 1.0),
    ):
        inner = _column(face.x[inner_line], face.y[inner_line], X)
        outer = _column(face.x[outer_line], face.y[outer_line], X)
        thickness = np.maximum((outer - inner) * sign, min_thickness)
        depth = (Y - inner) * sign  # distance from the line toward the teeth, outward
        inside = np.clip(depth / thickness, 0.0, 1.0)
        beyond = 1.0 - _smoothstep(0.0, fade, depth - thickness)
        grid.dy[area] += (sign * amount * _LIP_GROW) * thickness * inside * beyond * taper


def _add_brow(grid: _Grid, side: regions.Side, amount: float) -> None:
    """brow_lift: the brow moves up (+), the forehead above it and the lid below
    it take up the difference; the eye stays."""
    face = grid.face
    s = regions.side_indices(side)
    cx = float(face.x[s.brow].mean())
    cy = float(face.y[s.brow].mean())
    half_len = max(abs(float(face.x[s.brow_upper[0]] - face.x[s.brow_upper[-1]])) / 2.0, 1.0)
    eye_y = float(face.y[s.eye_upper[0]] + face.y[s.eye_upper[-1]]) / 2.0
    down = max(_BROW_RADIUS_DOWN * (eye_y - cy), 1.0)
    up = _BROW_RADIUS_UP * face.scale
    rx = _BROW_RADIUS_X * half_len
    area = grid.patch(cx, cy, max(rx, up, down) * 1.42)
    rel_y = grid.Y[area] - cy
    r2 = ((grid.X[area] - cx) / rx) ** 2 + (rel_y / np.where(rel_y < 0.0, up, down)) ** 2
    grid.dy[area] -= (amount * _BROW_LIFT * face.scale) * _bump(r2)


def _pin_teeth(grid: _Grid) -> None:
    """No displacement inside the mouth opening (see the module docstring)."""
    face = grid.face
    local = (face.landmarks[regions.LIPS_INNER] - np.float32(grid.roi.offset)) / grid.step
    pin = regions.polygon_mask(local, regions.Roi(0, 0, grid.nx, grid.ny), feather_px=_PIN_FEATHER * face.scale / grid.step)
    keep = 1.0 - np.minimum(pin * _PIN_GAIN, 1.0)
    grid.dx *= keep
    grid.dy *= keep


def _limit(grid: _Grid) -> None:
    """Soft-limit the displacement length to ``_MAX_FIELD`` of the face width, and
    scale the whole field down if it would ever get close to folding over."""
    cap = _MAX_FIELD * grid.face.width
    knee = _CAP_KNEE * cap
    length = np.hypot(grid.dx, grid.dy)
    longest = float(length.max())
    if longest > knee:
        limited = knee + (cap - knee) * np.tanh((np.maximum(length, knee) - knee) / (cap - knee))
        factor = np.where(length > knee, limited / np.maximum(length, 1e-6), 1.0)
        grid.dx *= factor
        grid.dy *= factor
    gradient = _max_gradient(grid.dx, grid.dy, grid.step)
    if gradient > _MAX_GRADIENT:
        factor = _MAX_GRADIENT / gradient
        grid.dx *= factor
        grid.dy *= factor


def _max_gradient(dx: np.ndarray, dy: np.ndarray, step: float) -> float:
    if dx.shape[0] < 2 or dx.shape[1] < 2:
        return 0.0
    best = 0.0
    for channel in (dx, dy):
        best = max(best, float(np.abs(np.diff(channel, axis=0)).max()), float(np.abs(np.diff(channel, axis=1)).max()))
    return best / step


def _fade_at_frame_edges(grid: _Grid, frame_shape: tuple[int, ...]) -> None:
    """No displacement at the frame's edges: there is nothing beyond them to read
    from, so reading past an edge would smear its outermost pixels."""
    height, width = frame_shape[0], frame_shape[1]
    roi, step = grid.roi, grid.step
    margin = _EDGE_FADE * grid.face.width
    if min(roi.x0, roi.y0, width - roi.x1, height - roi.y1) >= margin:
        return
    xs = roi.x0 + (np.arange(grid.nx, dtype=np.float32) + 0.5) * step
    ys = roi.y0 + (np.arange(grid.ny, dtype=np.float32) + 0.5) * step
    fade_x = _smoothstep(0.0, margin, np.minimum(xs, width - xs))
    fade_y = _smoothstep(0.0, margin, np.minimum(ys, height - ys))
    fade = fade_y[:, None] * fade_x[None, :]
    grid.dx *= fade
    grid.dy *= fade


def _forward_field(face: _Face, roi: regions.Roi, step: int, amounts: _Amounts, frame_shape: tuple[int, ...]) -> _Grid:
    """Grid with the forward displacement (face-frame axes) of all sliders."""
    grid = _Grid(face, roi, step)
    _add_contour(grid, amounts)
    _add_chin(grid, amounts.chin_length)
    if amounts.eye_size != 0.0:
        _add_eye(grid, "left", amounts.eye_size)
        _add_eye(grid, "right", amounts.eye_size)
    _add_nose(grid, amounts)
    _add_mouth_width(grid, amounts.mouth_width)
    _add_mouth_corners(grid, amounts.mouth_corner)
    _add_lips(grid, amounts.lip_fullness)
    if amounts.brow_lift != 0.0:
        _add_brow(grid, "left", amounts.brow_lift)
        _add_brow(grid, "right", amounts.brow_lift)
    sigma = _FIELD_BLUR * face.scale / step
    grid.dx = cv2.GaussianBlur(grid.dx, (0, 0), sigma, borderType=cv2.BORDER_REPLICATE)
    grid.dy = cv2.GaussianBlur(grid.dy, (0, 0), sigma, borderType=cv2.BORDER_REPLICATE)
    _pin_teeth(grid)
    _fade_at_frame_edges(grid, frame_shape)
    _limit(grid)
    return grid


# --------------------------------------------------------------------------
# From the forward field to the map.
# --------------------------------------------------------------------------


def _sample_field(fx: np.ndarray, fy: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Bilinear lookup of a grid field at grid-index positions (zero outside)."""
    sx = cv2.remap(fx, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    sy = cv2.remap(fy, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    return sx, sy


def _invert(fx: np.ndarray, fy: np.ndarray, step: int) -> tuple[np.ndarray, np.ndarray]:
    """Backward field ``b`` with ``b(p) = -D(p + b(p))`` on the grid (fixed point)."""
    ny, nx = fx.shape
    base_x = np.arange(nx, dtype=np.float32)[None, :]
    base_y = np.arange(ny, dtype=np.float32)[:, None]
    bx, by = -fx, -fy
    for _ in range(_INVERSION_STEPS):
        sx, sy = _sample_field(fx, fy, base_x + bx / step, base_y + by / step)
        bx, by = -sx, -sy
    return bx, by


def _person_on_grid(ctx: FrameContext, grid: _Grid) -> np.ndarray | None:
    """How likely each node shows the person, or None without masks. The masks are
    class probabilities, so they add up (their maximum dips along the seams
    between the classes, e.g. along the jaw between face and neck)."""
    masks = [m for m in (ctx.hair_mask, ctx.face_skin_mask, ctx.body_skin_mask) if m is not None]
    if not masks:
        return None
    roi = grid.roi
    size = (grid.nx, grid.ny)
    person = np.zeros((grid.ny, grid.nx), np.float32)
    for mask in masks:
        person += cv2.resize(mask[roi.slices], size, interpolation=cv2.INTER_LINEAR)
    return np.minimum(person, 1.0)


def _core_person(grid: _Grid, oval_u: tuple[float, float]) -> np.ndarray:
    """The part of the person known from the landmarks alone: the face oval
    (soft edge at ``oval_u``) and the neck column below the chin."""
    face = grid.face
    oval = 1.0 - _smoothstep(oval_u[0], oval_u[1], grid.u)
    chin_y = float(face.y[_CHIN_POINT])
    neck = _smoothstep(chin_y - 0.1 * face.scale, chin_y + 0.1 * face.scale, grid.Y)
    off_center = np.abs(grid.X - float(face.x[_CHIN_POINT])) / face.scale
    neck = neck * (1.0 - _smoothstep(_PROTECT_NECK_X * 0.7, _PROTECT_NECK_X, off_center))
    return np.maximum(oval, neck)


def _moved(values: np.ndarray, bx: np.ndarray, by: np.ndarray, step: int) -> np.ndarray:
    """``values`` (grid resolution) warped by the backward field."""
    ny, nx = values.shape
    base_x = np.arange(nx, dtype=np.float32)[None, :]
    base_y = np.arange(ny, dtype=np.float32)[:, None]
    return cv2.remap(values, base_x + bx / step, base_y + by / step, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _protection(grid: _Grid, person: np.ndarray | None, bx: np.ndarray, by: np.ndarray, protect: float) -> np.ndarray:
    """Field multiplier per node: 1 on the person, ``1 - protect`` on the background."""
    keep = _core_person(grid, _PROTECT_OVAL_U)
    # Both where a pixel is now and where it comes from count as the person
    # (a fuller face moves onto what used to be background).
    keep = np.maximum(keep, _moved(keep, bx, by, grid.step))
    if person is not None:
        on_person = _smoothstep(_PROTECT_MASK_RAMP[0], _PROTECT_MASK_RAMP[1], person)
        keep = np.maximum(keep, np.maximum(on_person, _moved(on_person, bx, by, grid.step)))
    sigma = _PROTECT_BLUR * grid.face.scale / grid.step
    keep = cv2.GaussianBlur(keep, (0, 0), max(sigma, 0.5), borderType=cv2.BORDER_REPLICATE)
    return 1.0 - protect * (1.0 - keep)


def _warped_points(points: np.ndarray, grid: _Grid, fx: np.ndarray, fy: np.ndarray, bx: np.ndarray, by: np.ndarray) -> np.ndarray:
    """Where each point lands: the ``q`` with ``q + b(q) = point``. Newton steps
    from the forward field, which is already exact unless the background protection
    weakened the field there; a plain fixed-point iteration would run away where
    the map compresses (the backward field is steeper than 1 there)."""
    roi, step = grid.roi, grid.step
    offset = np.float32(roi.offset)

    def lookup(fa: np.ndarray, fb: np.ndarray, at: np.ndarray) -> np.ndarray:
        local = (at - offset) / step - 0.5
        sx, sy = _sample_field(fa, fb, local[:, 0][None, :], local[:, 1][None, :])
        return np.stack([sx[0], sy[0]], axis=1)

    points = np.asarray(points, np.float32)
    targets = points + lookup(fx, fy, points)
    along_x = np.float32([step, 0.0])
    along_y = np.float32([0.0, step])
    for _ in range(_LANDMARK_NEWTON_STEPS):
        here = lookup(bx, by, targets)
        error = targets + here - points
        if float(np.abs(error).max()) < _LANDMARK_TOLERANCE_PX:
            break
        # The Jacobian of q + b(q), from differences over one grid cell.
        ddx = (lookup(bx, by, targets + along_x) - here) / step
        ddy = (lookup(bx, by, targets + along_y) - here) / step
        a, b = 1.0 + ddx[:, 0], ddy[:, 0]
        c, d = ddx[:, 1], 1.0 + ddy[:, 1]
        det = np.maximum(a * d - b * c, 0.05)  # positive: the map does not fold
        targets = targets - np.stack([d * error[:, 0] - b * error[:, 1], a * error[:, 1] - c * error[:, 0]], axis=1) / det[:, None]
    return targets.astype(np.float32)


def _pose_level(face: _Face) -> float:
    """1 inside the interview pose envelope, fading to 0 well beyond it."""
    x, y = face.x, face.y
    oval_x = x[regions.FACE_OVAL]
    to_right = float(x[regions.NOSE_TIP] - oval_x.min())
    to_left = float(oval_x.max() - x[regions.NOSE_TIP])
    asymmetry = abs(to_left - to_right) / max(to_left + to_right, 1.0)
    level = 1.0 - float(_smoothstep(_YAW_FADE[0], _YAW_FADE[1], np.float32(asymmetry)))
    eye_y = float(y[[regions.RIGHT_EYE_INNER_CORNER, regions.RIGHT_EYE_OUTER_CORNER,
                     regions.LEFT_EYE_INNER_CORNER, regions.LEFT_EYE_OUTER_CORNER]].mean())
    span = float(y[_CHIN_POINT]) - eye_y
    if span <= 0.0:
        return 0.0
    nose = (float(y[regions.NOSE_TIP]) - eye_y) / span
    level *= 1.0 - float(_smoothstep(_PITCH_FADE_DOWN[0], _PITCH_FADE_DOWN[1], np.float32(nose)))
    level *= float(_smoothstep(_PITCH_FADE_UP[1], _PITCH_FADE_UP[0], np.float32(nose)))
    return level


@dataclass(frozen=True)
class _Area:
    """The part of the ROI with a non-zero field, in pixels and in grid nodes."""

    roi: regions.Roi
    nodes: tuple[slice, slice]
    step: int

    def upsample(self, values: np.ndarray) -> np.ndarray:
        """A grid-resolution array at the full resolution of ``roi``."""
        small = values[self.nodes]
        size = (small.shape[1] * self.step, small.shape[0] * self.step)
        return cv2.resize(small, size, interpolation=cv2.INTER_LINEAR)[: self.roi.h, : self.roi.w]


@dataclass(frozen=True)
class _Warp:
    """The final map: the backward field on the grid, and the part of it that moves anything."""

    grid: _Grid
    area: _Area
    bx: np.ndarray
    by: np.ndarray


def _bounding_area(grid: _Grid, active: np.ndarray, columns: tuple[int, int] | None = None) -> _Area | None:
    """Bounding box (grown by one node) of the active nodes, optionally of a range of columns only."""
    first = 0 if columns is None else columns[0]
    part = active if columns is None else active[:, columns[0] : columns[1]]
    rows = np.flatnonzero(part.any(axis=1))
    cols = np.flatnonzero(part.any(axis=0))
    if rows.size == 0:
        return None
    roi, step = grid.roi, grid.step
    j0, j1 = max(int(rows[0]) - 1, 0), min(int(rows[-1]) + 2, grid.ny)
    i0, i1 = max(first + int(cols[0]) - 1, 0), min(first + int(cols[-1]) + 2, grid.nx)
    pixels = regions.Roi(
        roi.x0 + i0 * step, roi.y0 + j0 * step, min(roi.x0 + i1 * step, roi.x1), min(roi.y0 + j1 * step, roi.y1)
    )
    return _Area(pixels, (slice(j0, j1), slice(i0, i1)), step)


def _column_runs(active: np.ndarray) -> list[tuple[int, int]]:
    """Ranges of neighboring columns with active nodes."""
    used = np.concatenate(([False], active.any(axis=0), [False]))
    edges = np.flatnonzero(used[1:] != used[:-1])
    return [(int(start), int(end)) for start, end in zip(edges[::2], edges[1::2])]


def _warp_mask(
    mask: np.ndarray | None, area: _Area, map_x: np.ndarray, map_y: np.ndarray, memory: list[np.ndarray], slot: int
) -> np.ndarray | None:
    """``mask`` with ``area`` warped. The result lives in ``memory`` (a frame's worth of fresh memory costs more
    than copying into it) and lasts until the next frame; it is a new array object each time, because the
    hair effect recognizes a mask it has already searched by its identity."""
    if mask is None:
        return None
    if memory[slot].shape != mask.shape:
        memory[slot] = np.empty(mask.shape, mask.dtype)
    warped = memory[slot].view()
    np.copyto(warped, mask)
    warped[area.roi.slices] = cv2.remap(mask, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    warped.setflags(write=False)
    return warped


def _person_ramp(masks: list[np.ndarray | None], roi: regions.Roi, ramp: tuple[float, float]) -> np.ndarray:
    """Smooth step of the summed person masks in ``roi``. Computed at half
    resolution, which is plenty for masks this soft."""
    total = sum(m[roi.y0 : roi.y1 : 2, roi.x0 : roi.x1 : 2] for m in masks if m is not None)
    return cv2.resize(_smoothstep(ramp[0], ramp[1], total), (roi.w, roi.h), interpolation=cv2.INTER_LINEAR)


def _room_samples(live: np.ndarray, room: np.ndarray, grid: _Grid, person: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Live and plate colors, (N, 3) each, where the frame surely shows the room, and
    (N,) which of them sit on an edge of either picture. ``live`` and ``room`` are
    the two pictures on the grid's nodes. Of the plain room a thinned-out set is
    plenty; every node on an edge counts, because thin lines are rarely hit at all."""
    face = grid.face
    stride = max(round(_PLATE_STATS_PX / grid.step), 1)
    jaw_y = float(face.y[list(_JAW_POINTS)].mean())
    where = (person < _PLATE_ROOM_BELOW) & (grid.u > _PLATE_ROOM_OVAL_U) & (grid.Y < jaw_y)
    edges = cv2.max(_contrast(live), _contrast(room)) > _PLATE_EDGE
    chosen = edges.copy()
    chosen[::stride, ::stride] = True
    chosen &= where
    return live[chosen], room[chosen], edges[chosen]


def _contrast(picture: np.ndarray) -> np.ndarray:
    """Mean difference of the channels from the picture's own 3 x 3 neighborhood."""
    return cv2.transform(cv2.absdiff(picture, cv2.blur(picture, (3, 3))), _CHANNEL_MEAN)


def _fill_from_plate(
    warped: np.ndarray,
    frame: np.ndarray,
    plate: np.ndarray,
    person: np.ndarray,
    warped_masks: list[np.ndarray | None],
    warp: _Warp,
    live: np.ndarray,
    room: np.ndarray,
    gain: np.ndarray,
    noise: float,
    level: float,
) -> None:
    """Replace the bent room around the warped person (``warped``, the picture of
    ``warp.area``) by the straight one: live pixels where the room was visible
    anyway, the plate where the person used to be (see the module docstring).
    ``live`` and ``room`` are the frame and the plate on the grid's nodes, ``person``
    its person mask. Only the outline of the warped person needs a finer picture
    than the grid. ``noise`` is the usual difference between frame and plate in the
    room. ``level`` below 1 only occurs for a moment, while the plate is switched on
    or off."""
    grid, area = warp.grid, warp.area
    bx, by, step = warp.bx, warp.by, grid.step
    # The model does not count everything on the person as hair or skin (lips,
    # glasses, clothes, holes in its masks), so whatever the frame shows that the
    # plate does not is the person as well. What is not caught that way looks like
    # the room anyway, except for the face, which the landmarks vouch for. Not the
    # neck column that _core_person adds: it is wider than most necks, and the room
    # seen through it next to the neck would stay bent.
    core = 1.0 - _smoothstep(_PLATE_OVAL_U[0], _PLATE_OVAL_U[1], grid.u)
    gain_matrix = np.diag(gain)
    difference = cv2.transform(cv2.absdiff(live, cv2.transform(room, gain_matrix)), _CHANNEL_MEAN)
    not_room = _smoothstep(_PLATE_FOREGROUND[0] + noise, _PLATE_FOREGROUND[1] + noise, difference)
    # Closed: a part of the room with the color of the clothes (a dark line behind
    # a dark top) must not let the plate through.
    not_room = cv2.morphologyEx(not_room, cv2.MORPH_CLOSE, _NEIGHBORS)
    moving = _smoothstep(_PLATE_ACTIVE_PX[0], _PLATE_ACTIVE_PX[1], np.hypot(bx, by))
    weights = (1.0 - _moved(cv2.max(core, not_room), bx, by, step)) * moving * np.float32(level)
    # Grown by one node: what the grid does not see of the old outline must not stay behind.
    sources = cv2.dilate(cv2.max(_smoothstep(_PLATE_SOURCE_RAMP[0], _PLATE_SOURCE_RAMP[1], person), core), _NEIGHBORS)
    active = (weights > 0.0) & (_moved(person, bx, by, step) < _PLATE_MOVED_RAMP[1])
    # Usually one box on each side of the face: its middle has no room to fill.
    for columns in _column_runs(active):
        box = _bounding_area(grid, active, columns)
        roi = box.roi
        weight = cv2.multiply(1.0 - _person_ramp(warped_masks, roi, _PLATE_MOVED_RAMP), box.upsample(weights))
        source = box.upsample(sources)
        behind = cv2.transform(plate[roi.slices], gain_matrix)
        cv2.threshold(behind, 1.0, 1.0, cv2.THRESH_TRUNC, dst=behind)
        behind = cv2.blendLinear(frame[roi.slices], behind, 1.0 - source, source)
        y0, x0 = roi.y0 - area.roi.y0, roi.x0 - area.roi.x0
        part = warped[y0 : y0 + roi.h, x0 : x0 + roi.w]
        part[...] = cv2.blendLinear(part, behind, 1.0 - weight, weight)


@register_effect
class ReshapeEffect(Effect):
    """Geometric face shape correction; see the module docstring."""

    name = "reshape"
    order = 5

    def __init__(self) -> None:
        self._mask_memory = [np.empty((0, 0), np.float32) for _ in range(3)]  # see _warp_mask
        self._step: int | None = None
        self._pose: float | None = None
        self._asked_intensity: float | None = None
        self._intensity: float | None = None
        self._forget_plate()
        self._forget_room()

    def reset(self) -> None:
        self._step = None
        self._pose = None
        self._asked_intensity = None
        self._intensity = None
        self._forget_plate()
        self._forget_room()

    def _forget_room(self) -> None:
        self._slim_level: float | None = None  # 0..1, how much of the slimming the room next to the jaw allows
        self._slim_wanted = 1.0
        self._room_frames = 0

    def _forget_plate(self) -> None:
        self._plate: np.ndarray | None = None  # the plate the statistics below belong to
        self._plate_trust: float | None = None  # 0..1, how much of the plate is used
        self._plate_gain: np.ndarray | None = None
        self._plate_mismatch = 0.0
        self._plate_edge_mismatch = 0.0
        self._plate_good_frames = 0
        self._plate_on = False

    def apply(self, frame: np.ndarray, ctx: FrameContext, settings: ReshapeSettings) -> np.ndarray:
        amounts = _Amounts.of(settings, ctx.intensity)
        if ctx.landmarks is None or not amounts.any():
            return frame
        face = _face_frame(np.asarray(ctx.landmarks, np.float32))
        if face is None:
            return frame
        amounts = amounts.scaled(self._held_intensity(ctx.intensity) / ctx.intensity * self._update_pose(face))
        if not amounts.any():
            return frame
        if ctx.background is not None and ctx.background is not self._plate:
            self._forget_room()  # a new plate: whatever was seen without it does not count
        amounts = self._slim_for_the_room(amounts)
        step = self._grid_step(face.scale)
        roi = regions.roi_around(face.landmarks[regions.FACE_OVAL], frame.shape, _ROI_PAD, 2.0 * step)
        if roi is None:
            return frame
        grid = _forward_field(face, roi, step, amounts, frame.shape)
        fx, fy = _to_frame_axes(grid)
        bx, by = _invert(fx, fy, step)
        protect = min(max(float(settings.protect_background), 0.0), 1.0)
        person = None
        if protect > 0.0:
            person = _person_on_grid(ctx, grid)
            keep = _protection(grid, person, bx, by, protect)
            bx = bx * keep
            by = by * keep
        area = _bounding_area(grid, np.hypot(bx, by) > _ACTIVE_FIELD_PX)
        if area is None:
            return frame
        pixels = area.roi
        map_x = area.upsample(bx) + np.arange(pixels.x0, pixels.x1, dtype=np.float32)[None, :]
        map_y = area.upsample(by) + np.arange(pixels.y0, pixels.y1, dtype=np.float32)[:, None]
        warped = cv2.remap(frame, map_x, map_y, _INTERPOLATION, borderMode=cv2.BORDER_REPLICATE)
        np.clip(warped, 0.0, 1.0, out=warped)
        warped_masks = [
            _warp_mask(m, area, map_x, map_y, self._mask_memory, slot)
            for slot, m in enumerate((ctx.hair_mask, ctx.face_skin_mask, ctx.body_skin_mask))
        ]
        plate = ctx.background
        if person is None or plate is None or plate.shape != frame.shape:
            self._forget_plate()
        else:
            live_nodes, room_nodes = grid.sample(frame), grid.sample(plate)
            level = self._plate_level(plate, *_room_samples(live_nodes, room_nodes, grid, person))
            if level > 0.0:
                warp = _Warp(grid, area, bx, by)
                _fill_from_plate(warped, frame, plate, person, warped_masks, warp, live_nodes, room_nodes, self._plate_gain, self._plate_mismatch, level)
        if person is not None and amounts.slims():
            self._watch_the_room(frame, grid, person)
        targets = _warped_points(face.landmarks, grid, fx, fy, bx, by)
        targets.setflags(write=False)
        # Everything succeeded: commit the frame and the context together.
        frame[pixels.slices] = warped
        ctx.landmarks = targets
        ctx.hair_mask, ctx.face_skin_mask, ctx.body_skin_mask = warped_masks
        return frame

    def _slim_for_the_room(self, amounts: _Amounts) -> _Amounts:
        """``amounts`` with the slimming that the room next to the jaw allows (see the module docstring)."""
        if self._slim_level is None:
            self._slim_level = 1.0
        elif self._intensity is not None and self._intensity <= _SLIM_SETTLED_INTENSITY:
            self._slim_level = self._slim_wanted  # still fading in: no change can be seen
        else:
            change = self._slim_wanted - self._slim_level
            self._slim_level += max(min(change, _BUSY_ROOM_RISE), -_BUSY_ROOM_FALL)
        if self._slim_level >= 1.0:
            return amounts
        return dataclasses.replace(amounts, face_slim=amounts.face_slim * self._slim_level, jaw=amounts.jaw * self._slim_level)

    def _watch_the_room(self, frame: np.ndarray, grid: _Grid, person: np.ndarray) -> None:
        """Update what ``_slim_for_the_room`` aims at. A plate in use bends nothing."""
        if self._plate_on:
            self._slim_wanted = 1.0
            return
        self._room_frames = (self._room_frames + 1) % _BUSY_ROOM_EVERY
        if self._room_frames != 1:
            return
        share = _busy_room_share(frame, grid, person)
        if share is not None:
            busy = float(_smoothstep(_BUSY_ROOM_SHARE[0], _BUSY_ROOM_SHARE[1], np.float32(share)))
            self._slim_wanted = 1.0 - busy * (1.0 - _BUSY_ROOM_FLOOR)

    def _held_intensity(self, intensity: float) -> float:
        """The intensity to warp with. The pipeline fades the effects when the face
        is lost: one step down, and one step up again when the tracker finds it in
        the next frame. Colors hide that; geometry would twitch for one frame. So
        the first step down is held, and the rest of a real fade goes down to
        zero in one frame less (no jump at its end)."""
        asked, held = self._asked_intensity, self._intensity
        if asked is None or held is None or intensity >= held:
            held = intensity
        elif intensity < asked:
            held = min(held, intensity * _DROPOUT_HOLD)
        else:  # no longer falling, but below what is held: the master slider was lowered
            held += _INTENSITY_RELEASE * (intensity - held)
        self._asked_intensity = intensity
        self._intensity = held
        return held

    def _grid_step(self, scale: float) -> int:
        wanted = _GRID_STEP * scale
        step = self._step
        if step is None or not step / _GRID_STEP_SLACK <= wanted <= step * _GRID_STEP_SLACK:
            step = int(np.clip(round(wanted), *_GRID_STEP_PX))
            self._step = step
        return step

    def _update_pose(self, face: _Face) -> float:
        level = _pose_level(face)
        if self._pose is None:
            self._pose = level
        else:
            self._pose += _POSE_ADAPT * (level - self._pose)
            if level == 0.0 and self._pose < 0.02:
                self._pose = 0.0
        return self._pose

    def _plate_level(self, plate: np.ndarray, live_px: np.ndarray, room_px: np.ndarray, on_edge: np.ndarray) -> float:
        """How much of the plate to use: 1 or 0, in between only while switching.
        Also keeps its exposure gain up to date. Both follow per-frame statistics of
        the room, so they are smoothed over time; the first measurement counts at
        once (the warp is still fading in then, so no jump can be seen)."""
        if plate is not self._plate:
            self._forget_plate()
            self._plate = plate
        if len(live_px) < _PLATE_MIN_SAMPLES:
            return self._plate_trust if self._plate_trust is not None else 0.0
        # Medians and a percentile: what slips through as "room" must not decide
        # anything (stray hair, a shadow of the person, a passing object).
        gain = np.median(live_px / np.maximum(room_px, 0.02), axis=0)
        gain = np.clip(gain, *_PLATE_GAIN_RANGE).astype(np.float32)
        difference = np.abs(live_px - room_px * gain)
        mismatch = float(np.percentile(difference, _PLATE_MISMATCH_PERCENTILE))
        # Plain walls say nothing about a moved camera, the edges do. Only frames
        # with enough samples on edges count: the others have no opinion on it. Not the
        # median: most nodes "on" a thin line are its neighbors, which match anyway.
        edges = float(np.percentile(difference[on_edge], _PLATE_MISMATCH_PERCENTILE)) if int(on_edge.sum()) >= _PLATE_MIN_EDGE_SAMPLES else None
        if self._plate_trust is None or self._plate_gain is None:
            self._plate_gain, self._plate_mismatch = gain, mismatch
            self._plate_edge_mismatch = edges or 0.0
            self._plate_on = max(self._plate_mismatch, self._plate_edge_mismatch) < _PLATE_MISMATCH_OFF
            self._plate_trust = 1.0 if self._plate_on else 0.0
            return self._plate_trust
        self._plate_gain = self._plate_gain + _PLATE_ADAPT * (gain - self._plate_gain)
        self._plate_mismatch += _PLATE_ADAPT * (mismatch - self._plate_mismatch)
        if edges is not None:
            self._plate_edge_mismatch += _PLATE_ADAPT * (edges - self._plate_edge_mismatch)
        worst = max(self._plate_mismatch, self._plate_edge_mismatch)
        if worst > _PLATE_MISMATCH_OFF:
            self._plate_on = False
        # Coming back takes a while: switching back and forth would be the worst of all.
        self._plate_good_frames = self._plate_good_frames + 1 if worst < _PLATE_MISMATCH_ON else 0
        if self._plate_good_frames >= _PLATE_RETURN_FRAMES:
            self._plate_on = True
        target = 1.0 if self._plate_on else 0.0
        self._plate_trust += max(min(target - self._plate_trust, 1.0 / _PLATE_FADE_FRAMES), -1.0 / _PLATE_FADE_FRAMES)
        return self._plate_trust


def _busy_room_share(frame: np.ndarray, grid: _Grid, person: np.ndarray) -> float | None:
    """How much of the room next to the jaw lies on an edge (None: too little of it can be seen)."""
    face = grid.face
    eye_y = float(face.y[[regions.RIGHT_EYE_OUTER_CORNER, regions.LEFT_EYE_OUTER_CORNER]].mean())
    zone = (
        (cv2.dilate(person, _NEIGHBORS) < _PLATE_ROOM_BELOW)
        & (grid.u > _BUSY_ROOM_OVAL_U[0])
        & (grid.u < _BUSY_ROOM_OVAL_U[1])
        & (grid.Y > eye_y)
        & (grid.Y < float(face.y[_CHIN_POINT]))
    )
    if int(zone.sum()) < _BUSY_ROOM_MIN_NODES:
        return None
    # Only the rows that matter: this takes up to a millisecond on a large face.
    rows = np.flatnonzero(zone.any(axis=1))
    first, last = int(rows[0]), int(rows[-1]) + 1
    roi, step = grid.roi, grid.step
    top, bottom = roi.y0 + first * step, min(roi.y0 + last * step, roi.y1)
    green = frame[top:bottom, roi.x0 : roi.x1, 1]
    edges = cv2.absdiff(green, cv2.blur(green, (3, 3))) > _BUSY_ROOM_EDGE
    size = (grid.nx * step, (last - first) * step)
    pixels = cv2.resize(np.uint8(zone[first:last]), size, interpolation=cv2.INTER_NEAREST)[: bottom - top, : roi.w] > 0
    return float((edges & pixels).sum()) / max(int(pixels.sum()), 1)


def _to_frame_axes(grid: _Grid) -> tuple[np.ndarray, np.ndarray]:
    """The forward field with frame (x, y) components instead of face-frame ones."""
    ex, ey = grid.face.ex, grid.face.ey
    return grid.dx * ex[0] + grid.dy * ey[0], grid.dx * ex[1] + grid.dy * ey[1]
