"""Face landmark tracking: MediaPipe Face Landmarker + One Euro smoothing.

Detection window
----------------
MediaPipe's face detector letterboxes the whole input into a small square,
so on a wide 16:9 frame a face at normal webcam distance becomes tiny and is
often missed in dim light. ``FaceTracker`` therefore never feeds the full
wide frame: it feeds a square window (side = the frame's short side,
downscaled to ``detect_long_side``) that follows the face. While no face is
tracked it cycles the window through center / start / end positions, which
together cover the whole frame. On the test portraits this finds dim and
small faces that the full frame misses and lowers landmark jitter about 4x.

Smoothing
---------
Landmarks are smoothed with a One Euro filter whose speed term is measured in
inter-eye distances per second rather than pixels per second, so the same
slider value feels the same near and far from the camera. See ``smoothing_params``
for the 0..1 slider mapping.

Image motion
------------
A filter on the model's output alone always trails a moving head, and the model
itself lags a little behind the picture too, because it starts from where the
face was. A thin eyeliner shows even a few pixels of that as a second line. So
the tracker also measures how the rigid part of the face moved between the last
two frames, from the picture itself (``FaceMotion``, sparse optical flow on
textured landmarks such as the eye corners), and moves the filter's state by
that motion before the new sample goes in. The filter then only has to deal with
what the head motion does not explain: noise, which it removes as before, and the
expressions, which it follows as before. Where the picture gives no reliable
motion (too few points agree) the filter works exactly as it would without this.

The same motion carries the last landmarks across frames in which the model finds
no face (``FaceTracker.coast``), so makeup fading out after a lost face follows the
face instead of staying behind on the wall.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

NUM_LANDMARKS = 478
# Outer eye corners (subject's right, subject's left): their distance is the
# face scale that normalizes the filter's speed term.
EYE_OUTER_CORNERS = (33, 263)

DEFAULT_DETECT_LONG_SIDE = 640
# The detection window pans toward the face once the face center is more than
# _PAN_DEAD_ZONE (fraction of the window side) off-center, by at most
# _PAN_MAX_STEP per frame: MediaPipe's own tracking expects the face near where
# it was in the previous input, and a sudden window jump makes it lose the face.
_PAN_DEAD_ZONE = 0.05
_PAN_MAX_STEP = 0.02
# Smallest time step the filter assumes (guards repeated timestamps).
_MIN_DT_S = 1e-3

# Landmarks whose image motion is the head's: textured, and hardly moved by expressions
# (eye corners, the outer eye, brow ends, the nose, the nostril wings, the upper cheeks).
_MOTION_POINTS = np.array(
    [33, 133, 362, 263, 130, 243, 463, 359, 226, 446, 70, 300, 107, 336, 105, 334]
    + [168, 6, 197, 195, 5, 4, 1, 98, 327, 64, 294, 116, 345, 50, 280]
)
_MOTION_MIN_POINTS = 10  # fewer agreeing points than this: no motion is reported
_MOTION_MARGIN = 0.7  # the picture used reaches this far around the face, x face scale
_MOTION_WORK_SCALE = 110.0  # px of face scale the optical flow works at (bigger faces use every n-th pixel)
_MOTION_WINDOW = (15, 15)
_MOTION_LEVELS = 3
_MOTION_CRITERIA = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 8, 0.03)
_MOTION_AGREEMENT_PX = 0.8  # a point agrees with the motion within this distance (at the working size)
_MOTION_MAX_LINEAR = 0.12  # largest believable change of the linear part per frame (scale, rotation, shear)
# Largest point displacement (px at the working size) at which the measured motion counts for nothing and
# in full: the flow's own noise must not shake a still face, and so slow a head does not lag anyway.
_MOTION_STILL_PX = (0.3, 1.0)
_MOTION_FULL_AFFINE_PX = 3.0  # above this (px at the working size) the motion is measured with a full affine map
# How far the model's output trails the picture, in frames of the head's own motion
# (measured on nods of 75 to 380 px/s; it starts from where the face was).
_MODEL_LAG_FRAMES = 0.4
_MODEL_MAX_LAG = 0.012  # and never more than this, x face scale: it does not grow with the speed of a bow
_MOTION_SLACK = 0.5  # see FaceTracker.detect, x the head's motion in a frame
# A sample this far from the carried-along state (x face scale, median over the points) starts the filter anew.
_RESTART_JUMP = 0.2
# How long the last landmarks are carried along after the model lost the face, in frames.
_MAX_COAST_FRAMES = 15


# --------------------------------------------------------------------------
# One Euro filter
# --------------------------------------------------------------------------


def _smoothing_factor(cutoff_hz: np.ndarray | float, dt_s: float) -> np.ndarray | float:
    """Exponential smoothing factor of a first-order low-pass at ``cutoff_hz``."""
    r = 2.0 * math.pi * np.asarray(cutoff_hz) * dt_s
    return r / (r + 1.0)


class OneEuroFilter:
    """One Euro filter (Casiez et al., CHI 2012), vectorized.

    Filters every element of an array independently in time. The adaptive
    cutoff uses the speed of each *point*: the norm over the last axis for
    arrays with ndim >= 2 (an (N, 2) array is N points with one cutoff each),
    or the absolute value for 1-D arrays.

    ``min_cutoff`` (Hz): cutoff at rest; lower = less jitter, more lag.
    ``beta``: cutoff increase per unit of speed; higher = less lag in motion.
    ``d_cutoff`` (Hz): cutoff of the low-pass applied to the speed estimate.
    ``slack``: differences below this are no reason to react faster (known errors of the input).
    The parameters are plain attributes and may be changed between calls.
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.slack = 0.0  # a sample this far from the state (in x's units) does not count as speed
        self._x: np.ndarray | None = None
        self._dx: np.ndarray | None = None
        self._t_s = 0.0

    def reset(self) -> None:
        """Forget the history; the next sample passes through unchanged."""
        self._x = None
        self._dx = None

    def move(self, motion: np.ndarray) -> None:
        """Apply a 2x3 image motion to the filtered x, y (the first two columns of an (N, >=2) state)."""
        if self._x is not None:
            self._x[:, :2] = self._x[:, :2] @ motion[:2, :2].T + motion[:2, 2]

    def __call__(self, x: np.ndarray, t_s: float) -> np.ndarray:
        """Filter sample ``x`` taken at time ``t_s`` (seconds); returns a new array."""
        x = np.asarray(x, dtype=np.float64)
        if self._x is None or self._dx is None or self._x.shape != x.shape:
            self._x = x.copy()
            self._dx = np.zeros_like(x)
            self._t_s = t_s
            return self._x.copy()
        dt = max(t_s - self._t_s, _MIN_DT_S)
        self._t_s = t_s
        self._dx += _smoothing_factor(self.d_cutoff, dt) * ((x - self._x) / dt - self._dx)
        speed = np.linalg.norm(self._dx, axis=-1, keepdims=True) if x.ndim >= 2 else np.abs(self._dx)
        if self.slack > 0.0:
            speed = np.maximum(speed - self.slack / dt, 0.0)
        cutoff = self.min_cutoff + self.beta * speed
        self._x += _smoothing_factor(cutoff, dt) * (x - self._x)
        return self._x.copy()


@dataclass(frozen=True)
class SmoothingParams:
    min_cutoff: float  # Hz
    beta: float  # Hz per (inter-eye distance / second)
    d_cutoff: float  # Hz


# Slider endpoints, interpolated geometrically (equal slider steps feel even).
_MIN_CUTOFF_RANGE = (6.0, 0.3)
_BETA_RANGE = (20.0, 4.0)
# The speed estimate must react within a frame or two: a mouth opens faster than the filter would otherwise follow.
_D_CUTOFF = 8.0


def smoothing_params(smoothing: float) -> SmoothingParams | None:
    """Map the 0..1 "顔追跡のなめらかさ" slider to One Euro parameters.

    * 0 -> ``None``: no filtering, raw landmarks (fastest response).
    * (0, 1] -> ``min_cutoff`` falls geometrically from 6 Hz to 0.3 Hz
      (less jitter at rest) and ``beta`` from 20 to 4 (steadier during
      motion, at the cost of some lag). 0.5 gives about 1.3 Hz / 8.9.

    ``beta`` is per inter-eye distance per second: at 0.5 a head moving one
    inter-eye distance per second raises the cutoff by about 8.9 Hz. The
    endpoints were tuned on synthetic head motion with 0.7 px landmark noise:
    at 0.5 jitter at rest halves while the lag during a fast head sway
    (~500 px/s at 720p) stays around 1 px.
    """
    s = min(max(float(smoothing), 0.0), 1.0)
    if s == 0.0:
        return None

    def geometric(lo_hi: tuple[float, float]) -> float:
        return lo_hi[0] * (lo_hi[1] / lo_hi[0]) ** s

    return SmoothingParams(geometric(_MIN_CUTOFF_RANGE), geometric(_BETA_RANGE), _D_CUTOFF)


# --------------------------------------------------------------------------
# MediaPipe adapter
# --------------------------------------------------------------------------


class LandmarkModel(Protocol):
    """What ``FaceTracker`` needs from a landmark model (fakes implement this)."""

    def detect(self, rgb: np.ndarray, ts_ms: int) -> np.ndarray | None:
        """(478, 3) normalized x, y, z of the single face in ``rgb``, or None.

        ``rgb`` is reused for the next frame: do not keep it after returning.
        """

    def close(self) -> None: ...


class MediaPipeFaceLandmarker:
    """Thin adapter over MediaPipe's Face Landmarker in VIDEO mode.

    The detector threshold defaults below MediaPipe's 0.5 so that small or dim
    faces are found; the landmark model's own face-presence check (kept at
    0.5) still rejects non-faces.
    """

    def __init__(self, model_path: str | Path, min_detection_confidence: float = 0.3) -> None:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        # The model is passed as bytes: MediaPipe cannot open paths containing
        # non-ASCII characters on Windows (e.g. a Japanese user name).
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_buffer=Path(model_path).read_bytes()),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=min_detection_confidence,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._image = mp.Image
        self._srgb = mp.ImageFormat.SRGB

    def detect(self, rgb: np.ndarray, ts_ms: int) -> np.ndarray | None:
        result = self._landmarker.detect_for_video(self._image(image_format=self._srgb, data=rgb), ts_ms)
        if not result.face_landmarks:
            return None
        return np.array([(p.x, p.y, p.z) for p in result.face_landmarks[0]], dtype=np.float32)

    def close(self) -> None:
        self._landmarker.close()


# --------------------------------------------------------------------------
# Detection window
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DetectionWindow:
    """Square region of the frame fed to the landmark model."""

    x0: int
    y0: int
    size: int


def search_windows(height: int, width: int) -> list[DetectionWindow]:
    """Square windows (center first) that together cover the whole frame."""
    size = min(height, width)
    slack_x, slack_y = width - size, height - size
    fractions = (0.5, 0.0, 1.0) if slack_x or slack_y else (0.5,)
    return [DetectionWindow(round(slack_x * f), round(slack_y * f), size) for f in fractions]


def follow_face(window: DetectionWindow, points: np.ndarray, height: int, width: int) -> DetectionWindow:
    """Return the window panned a small step toward the face center (if it is off-center)."""
    center = (points[:, :2].min(axis=0) + points[:, :2].max(axis=0)) / 2.0
    origin = np.array([window.x0, window.y0])
    target = np.clip(np.round(center - window.size / 2.0), 0, [width - window.size, height - window.size])
    offset = target - origin
    if np.abs(offset).max() <= _PAN_DEAD_ZONE * window.size:
        return window
    max_step = max(1, round(_PAN_MAX_STEP * window.size))
    x0, y0 = (origin + np.clip(offset, -max_step, max_step)).astype(int)
    return DetectionWindow(int(x0), int(y0), window.size)


def window_rgb(
    frame_bgr: np.ndarray, window: DetectionWindow, long_side: int, buffers: _WindowBuffers | None = None
) -> np.ndarray:
    """Contiguous RGB crop of ``window``, downscaled to at most ``long_side``.

    With ``buffers`` the result lives in them until the next call.
    """
    crop = frame_bgr[window.y0 : window.y0 + window.size, window.x0 : window.x0 + window.size]
    side = min(window.size, long_side)
    scaled, rgb = (None, None) if buffers is None else buffers.for_side(side)
    if window.size > long_side:
        crop = cv2.resize(crop, (long_side, long_side), dst=scaled, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB, dst=rgb)


class _WindowBuffers:
    """Memory for ``window_rgb``, kept from frame to frame: an image of this size is
    served fresh by the OS on every allocation, which costs as much as the conversion."""

    def __init__(self) -> None:
        self._side = 0
        self._arrays: tuple[np.ndarray, np.ndarray] | None = None

    def for_side(self, side: int) -> tuple[np.ndarray, np.ndarray]:
        if self._arrays is None or side != self._side:
            self._side = side
            self._arrays = (np.empty((side, side, 3), np.uint8), np.empty((side, side, 3), np.uint8))
        return self._arrays


def face_scale_px(points: np.ndarray) -> float:
    """Distance between the outer eye corners in pixels (at least 1)."""
    a, b = EYE_OUTER_CORNERS
    return max(float(np.linalg.norm(points[a, :2] - points[b, :2])), 1.0)


# --------------------------------------------------------------------------
# Image motion
# --------------------------------------------------------------------------


class FaceMotion:
    """Motion of the rigid part of the face between the last two frames, measured in the picture."""

    def __init__(self) -> None:
        self._gray: np.ndarray | None = None
        self._previous: np.ndarray | None = None

    def reset(self) -> None:
        self._gray = None
        self._previous = None

    def update(self, frame_bgr: np.ndarray) -> None:
        """The new frame: the one before it becomes the reference. Buffers are reused."""
        spare = self._previous if self._previous is not None and self._previous.shape == frame_bgr.shape[:2] else None
        self._previous = self._gray
        self._gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY, dst=spare)

    def measure(self, points: np.ndarray) -> np.ndarray | None:
        """2x3 map from where ``points`` (N, 2+ of the last frame, px) were to where they are now.

        None when the picture does not tell: too few textured points that agree, or an
        unbelievable result (a hand in front of the face, the face leaving the frame).
        """
        if self._previous is None or self._gray is None or self._previous.shape != self._gray.shape:
            return None
        height, width = self._gray.shape
        scale = face_scale_px(points)
        low = np.floor(points[:, :2].min(axis=0) - _MOTION_MARGIN * scale).astype(int)
        high = np.ceil(points[:, :2].max(axis=0) + _MOTION_MARGIN * scale).astype(int)
        x0, y0 = max(int(low[0]), 0), max(int(low[1]), 0)
        x1, y1 = min(int(high[0]), width), min(int(high[1]), height)
        step = max(1, round(scale / _MOTION_WORK_SCALE))
        if (x1 - x0) // step < 2 * _MOTION_WINDOW[0] or (y1 - y0) // step < 2 * _MOTION_WINDOW[1]:
            return None
        before = np.ascontiguousarray(self._previous[y0:y1:step, x0:x1:step])
        after = np.ascontiguousarray(self._gray[y0:y1:step, x0:x1:step])
        start = (points[_MOTION_POINTS, :2].astype(np.float32) - (x0, y0)) / step
        inside = (start >= 0).all(axis=1) & (start < (before.shape[1], before.shape[0])).all(axis=1)
        if int(inside.sum()) < _MOTION_MIN_POINTS:
            return None
        start = start[inside].reshape(-1, 1, 2).astype(np.float32)
        end, status, _ = cv2.calcOpticalFlowPyrLK(
            before, after, start, None, winSize=_MOTION_WINDOW, maxLevel=_MOTION_LEVELS, criteria=_MOTION_CRITERIA
        )
        if end is None:
            return None
        tracked = status.ravel() == 1
        if int(tracked.sum()) < _MOTION_MIN_POINTS:
            return None
        return _fit_motion(start[tracked].reshape(-1, 2), end[tracked].reshape(-1, 2), (x0, y0), step)


def _fit(estimate: Callable[..., tuple[np.ndarray | None, np.ndarray | None]], start: np.ndarray, end: np.ndarray) -> np.ndarray | None:
    fit, inliers = estimate(
        start, end, method=cv2.RANSAC, ransacReprojThreshold=_MOTION_AGREEMENT_PX, maxIters=64, refineIters=4
    )
    return fit if fit is not None and inliers is not None and int(inliers.sum()) >= _MOTION_MIN_POINTS else None


def _fit_motion(start: np.ndarray, end: np.ndarray, origin: tuple[int, int], step: int) -> np.ndarray | None:
    """The map most of these point pairs agree on, in full-frame pixels (see ``measure``)."""
    # Shift, turn and scale only for small motion, where the flow's noise would show: a full affine map
    # shakes more. A fast bow foreshortens the face, and there a full affine map is much closer.
    fit = _fit(cv2.estimateAffinePartial2D, start, end)
    if fit is None:
        return None
    largest = float(np.abs(start @ fit[:2, :2].T + fit[:2, 2] - start).max())
    if largest > _MOTION_FULL_AFFINE_PX:
        full = _fit(cv2.estimateAffine2D, start, end)
        fit = fit if full is None else full
    linear = fit[:2, :2]
    if np.abs(linear - np.eye(2)).max() > _MOTION_MAX_LINEAR:
        return None
    low, high = _MOTION_STILL_PX
    trust = min(max((largest - low) / (high - low), 0.0), 1.0)
    trust = trust * trust * (3.0 - 2.0 * trust)
    identity = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    # The crop works in (frame - origin) / step: only the translation changes in frame pixels.
    offset = np.array(origin, np.float64)
    translation = fit[:2, 2] * step + offset - linear @ offset
    return identity + trust * (np.column_stack([linear, translation]) - identity)


# --------------------------------------------------------------------------
# Tracker
# --------------------------------------------------------------------------


class FaceTracker:
    """Single-face landmark tracker with temporal smoothing.

    ``detect`` returns full-frame pixel coordinates of the unmirrored frame.
    Pass ``model`` to inject a ``LandmarkModel`` (tests); otherwise the
    MediaPipe model at ``model_path`` is loaded.
    """

    def __init__(
        self,
        model_path: str | Path | None,
        smoothing: float = 0.5,
        *,
        detect_long_side: int = DEFAULT_DETECT_LONG_SIDE,
        model: LandmarkModel | None = None,
    ) -> None:
        if model is None:
            if model_path is None:
                raise ValueError("model_path or model is required")
            model = MediaPipeFaceLandmarker(model_path)
        self._model = model
        self._detect_long_side = detect_long_side
        self._buffers = _WindowBuffers()
        self._filter = OneEuroFilter()
        self._params: SmoothingParams | None = None
        self._smoothing = 0.0
        self.set_smoothing(smoothing)
        self._last_ts = -1
        self._frame_shape: tuple[int, int] | None = None
        self._window: DetectionWindow | None = None
        self._tracking = False
        self._search_queue: deque[DetectionWindow] = deque()
        self._motion = FaceMotion()
        self._last_points: np.ndarray | None = None  # (478, 3) where the face was in the last frame
        self._coasted = 0  # frames the last points have been carried along without the model
        self._closed = False

    @property
    def smoothing(self) -> float:
        return self._smoothing

    def set_smoothing(self, smoothing: float) -> None:
        """0 = raw landmarks (fast response) ... 1 = very stable (see ``smoothing_params``)."""
        self._smoothing = min(max(float(smoothing), 0.0), 1.0)
        self._params = smoothing_params(self._smoothing)
        if self._params is None:
            self._filter.reset()
        else:
            self._filter.min_cutoff = self._params.min_cutoff
            self._filter.d_cutoff = self._params.d_cutoff

    def detect(self, frame_bgr: np.ndarray, ts_ms: int) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Landmarks ((478, 2) float32 px) and depth ((478,) float32 px) or (None, None)."""
        height, width = frame_bgr.shape[:2]
        if self._frame_shape != (height, width):
            self.reset()
            self._frame_shape = (height, width)
        window = self._next_window(height, width)
        ts = self._next_timestamp(ts_ms)
        raw = self._model.detect(window_rgb(frame_bgr, window, self._detect_long_side, self._buffers), ts)
        self._motion.update(frame_bgr)
        motion = None if self._last_points is None else self._motion.measure(self._last_points)
        if raw is None:
            self._on_lost(window)
            self._coast_with(motion)
            return None, None
        points = raw.astype(np.float64) * window.size + (window.x0, window.y0, 0.0)
        self._on_found(window, points, height, width)
        self._filter.slack = 0.0
        if motion is not None:
            step = self._last_points[:, :2] @ motion[:2, :2].T + motion[:2, 2] - self._last_points[:, :2]
            scale = face_scale_px(points)
            moved = float(np.median(np.abs(step).max(axis=1))) / scale
            points[:, :2] += step * min(_MODEL_LAG_FRAMES, _MODEL_MAX_LAG / max(moved, 1e-6))
            self._filter.move(motion)
            # While the head moves, the model is off by about as much as the head moved in a frame, and
            # the picture knows better. What goes beyond that is an expression, and is followed as ever.
            self._filter.slack = moved * scale * _MOTION_SLACK
        if self._coasted and self._restarts_filter(points):
            self._filter.reset()
        points = self._smooth(points, ts)
        self._last_points = points
        self._coasted = 0
        landmarks = np.ascontiguousarray(points[:, :2], dtype=np.float32)
        return landmarks, points[:, 2].astype(np.float32)

    def coast(self) -> tuple[np.ndarray, np.ndarray] | None:
        """After ``detect`` found no face: where the face went according to the picture alone.

        Landmarks and depth like those of ``detect``, or None when that cannot be told.
        """
        if self._coasted == 0 or self._last_points is None:
            return None
        points = self._last_points
        return np.ascontiguousarray(points[:, :2], dtype=np.float32), points[:, 2].astype(np.float32)

    def reset(self) -> None:
        """Drop smoothing history and the detection window (keeps the timestamp guard)."""
        self._filter.reset()
        self._window = None
        self._tracking = False
        self._search_queue.clear()
        self._motion.reset()
        self._last_points = None
        self._coasted = 0

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._model.close()

    # ---- internals --------------------------------------------------------

    def _next_timestamp(self, ts_ms: int) -> int:
        # MediaPipe VIDEO mode rejects timestamps that do not strictly increase.
        self._last_ts = max(int(ts_ms), self._last_ts + 1)
        return self._last_ts

    def _next_window(self, height: int, width: int) -> DetectionWindow:
        if self._window is None:
            if not self._search_queue:
                self._search_queue.extend(search_windows(height, width))
            self._window = self._search_queue.popleft()
        return self._window

    def _restarts_filter(self, points: np.ndarray) -> bool:
        """True if a face found again is not where the picture says the old one went."""
        distance = np.median(np.linalg.norm(points[:, :2] - self._last_points[:, :2], axis=1))
        return bool(distance > _RESTART_JUMP * face_scale_px(points))

    def _coast_with(self, motion: np.ndarray | None) -> None:
        """Carry the last points (and the filter) along, or give them up."""
        if motion is None or self._coasted >= _MAX_COAST_FRAMES or self._last_points is None:
            self._filter.reset()
            self._last_points = None
            self._coasted = 0
            return
        points = self._last_points.copy()
        points[:, :2] = points[:, :2] @ motion[:2, :2].T + motion[:2, 2]
        self._last_points = points
        self._coasted += 1
        self._filter.move(motion)

    def _on_lost(self, window: DetectionWindow) -> None:
        self._window = None
        if self._tracking:
            # A face tracked until now is most likely still there: retry the
            # same window once before searching the rest of the frame.
            self._search_queue = deque([window])
        self._tracking = False

    def _on_found(self, window: DetectionWindow, points: np.ndarray, height: int, width: int) -> None:
        self._tracking = True
        self._search_queue.clear()
        self._window = follow_face(window, points, height, width)

    def _smooth(self, points: np.ndarray, ts_ms: int) -> np.ndarray:
        if self._params is None:
            return points
        self._filter.beta = self._params.beta / face_scale_px(points)
        return self._filter(points, ts_ms / 1000.0)
