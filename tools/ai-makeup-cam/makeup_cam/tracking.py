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
"""

from __future__ import annotations

import math
from collections import deque
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
    The parameters are plain attributes and may be changed between calls.
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x: np.ndarray | None = None
        self._dx: np.ndarray | None = None
        self._t_s = 0.0

    def reset(self) -> None:
        """Forget the history; the next sample passes through unchanged."""
        self._x = None
        self._dx = None

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
_D_CUTOFF = 1.5


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
        """(478, 3) normalized x, y, z of the single face in ``rgb``, or None."""

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


def window_rgb(frame_bgr: np.ndarray, window: DetectionWindow, long_side: int) -> np.ndarray:
    """Contiguous RGB crop of ``window``, downscaled to at most ``long_side``."""
    crop = frame_bgr[window.y0 : window.y0 + window.size, window.x0 : window.x0 + window.size]
    if window.size > long_side:
        crop = cv2.resize(crop, (long_side, long_side), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)


def face_scale_px(points: np.ndarray) -> float:
    """Distance between the outer eye corners in pixels (at least 1)."""
    a, b = EYE_OUTER_CORNERS
    return max(float(np.linalg.norm(points[a, :2] - points[b, :2])), 1.0)


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
        self._filter = OneEuroFilter()
        self._params: SmoothingParams | None = None
        self._smoothing = 0.0
        self.set_smoothing(smoothing)
        self._last_ts = -1
        self._frame_shape: tuple[int, int] | None = None
        self._window: DetectionWindow | None = None
        self._tracking = False
        self._search_queue: deque[DetectionWindow] = deque()
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
        raw = self._model.detect(window_rgb(frame_bgr, window, self._detect_long_side), ts)
        if raw is None:
            self._on_lost(window)
            return None, None
        points = raw.astype(np.float64) * window.size + (window.x0, window.y0, 0.0)
        self._on_found(window, points, height, width)
        points = self._smooth(points, ts)
        landmarks = np.ascontiguousarray(points[:, :2], dtype=np.float32)
        return landmarks, points[:, 2].astype(np.float32)

    def reset(self) -> None:
        """Drop smoothing history and the detection window (keeps the timestamp guard)."""
        self._filter.reset()
        self._window = None
        self._tracking = False
        self._search_queue.clear()

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

    def _on_lost(self, window: DetectionWindow) -> None:
        self._filter.reset()
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
