"""Per-frame processing: tracking -> segmentation -> effects -> output frames.

``MakeupPipeline.process`` runs on the camera worker thread; settings and
bypass may be changed from the GUI thread at any time.

Never-crash policy: the tracker, the segmenter and every effect are isolated.
An exception is logged once per source and message, and that step is skipped
for the frame, so a bug degrades the makeup instead of breaking the call.

Fading: face-dependent effects fade in over ``FADE_FRAMES`` frames when a
face appears, and fade out when it is lost, so makeup never pops on or off. The
fade scales each face effect's ``ctx.intensity``: every effect multiplies its
opacity by it and is an exact no-op at 0, so this works like blending the
processed and original frame per effect, without copying frames. Effects
that do not need a face (hair) are not faded, because they do not depend on
it. Constructing the pipeline without ``tracker`` / ``segmenter`` loads (and
if needed downloads) the models, which may raise ``ModelDownloadError``.

A fade-out must not leave makeup behind where the face was. During it the makeup
follows the face by image motion (``FaceTracker.coast``). When that is unknown the
last landmarks are kept (never older than ``MAX_LANDMARK_AGE_MS``), which is right
for a face that sits still. A face that was moving fast keeps its last velocity
and fades twice as fast: staying behind, its makeup would land on the wall. A face
that comes back must also be found
``STEADY_FRAMES`` times in a row before the makeup grows again, because after a bow
it is found on and off, and the model's first answers are poor.

The hair masks are made every ``segmentation_interval`` frames, so on the frames in between
they are a frame old. While the face moves fast that is far enough for the hair effects to color
the wall where the hair just was, so the masks are then made as often as the model can.

Changing the look: ``set_settings`` takes effect on the next frame, which is what a slider
needs. ``glide_to_settings`` (a preset was chosen) moves there over ``SETTINGS_GLIDE_FRAMES``
frames: numbers and colors are interpolated, and a section that only one of the two looks has
fades with ``ctx.intensity``, so the look, the face shape included, never jumps. Choices (a lip
finish, say) change half way. At the end the settings are exactly the new ones. A change that comes
while the picture is still on its way, by either call, starts a new way from where the picture is.

Background plate: the pipeline holds one float32 picture of the empty room for
the current resolution and hands it to effects as ``ctx.background``. It is
loaded from disk when the resolution changes (never more often), replaced by
``capture_background`` and dropped by ``clear_background``; see ``background.py``.
Deleting wins over a capture that is still running, whichever finishes first.

Tracking ahead: when the camera thread announces each new frame with ``prepare``,
its landmarks are computed on their own thread right away, while the frame before
it is still being rendered, and ``process`` picks them up. Every frame is still
drawn with its own landmarks, so this adds no lag and changes no pixel; the camera
thread is only spared the wait. ``process`` works the same without it.
"""

from __future__ import annotations

import copy
import ctypes
import dataclasses
import logging
import sys
import threading
import time
from collections import deque
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

from . import background as plates
from . import regions
from .config import SECTIONS, GeneralSettings, MakeupSettings
from .effects.base import Effect, FrameContext
from .models import FACE_LANDMARKER, SELFIE_MULTICLASS, ensure_model
from .segmentation import SegMasks, Segmenter
from .tracking import FaceTracker

logger = logging.getLogger(__name__)

FADE_FRAMES = 6
MAX_LANDMARK_AGE_MS = 500.0
# Detections in a row before the fade rises (see the module docstring).
STEADY_FRAMES = 3
# Faster than this (x face scale per frame, in each of the last two frames), the last
# landmarks of a lost face are worthless: about 1.5 face scales per second at 30 fps.
_FAST_FACE_STEP = 0.05
SETTINGS_GLIDE_FRAMES = 8
# Masks as often as possible (see the module docstring): the face moved faster than this
# (x face scale per frame: about 1 face scale per second at 30 fps), and for this many frames after that.
_FAST_FOR_MASKS_STEP = 0.03
_FAST_FOR_MASKS_FRAMES = 10
# Effects that use segmentation masks when available without requiring them:
# skin and contour limit their area with the face-skin mask (the hairline shadow
# needs it), and reshape protects the hair and uses the plate only with masks.
# hairshape is not listed: needs_hair_mask already requests them.
OPTIONAL_MASK_USERS = frozenset({"skin", "contour", "reshape"})

_RELATION_PROCESSOR_CORE = 0
_MAX_LOGGED_ERRORS = 256
_MAX_TRACK_AHEAD = 4  # announced frames kept for process() (the renderer skips some when it falls behind)
_MESH_COLOR = (0, 255, 0)
_IRIS_COLOR = (0, 255, 255)
_FIRST_IRIS_INDEX = 468

# OpenCV's default of one worker per logical CPU almost doubles the CPU this process uses (3.2
# against 1.9 logical CPUs at 8 cores / 16 threads) and makes no frame faster: most calls here
# work on small areas, and the idle workers spin. That CPU belongs to the video call.
CV2_THREADS = 4
# uint8 -> float32 in [0, 1]. A table gives the same values as ``value * float32(1 / 255)``, and
# with a reused destination it takes a ninth of the time: fresh memory of this size is page-faulted.
_TO_FLOAT = np.multiply(np.arange(256, dtype=np.uint8), np.float32(1.0 / 255.0), dtype=np.float32).reshape(256, 1)


@dataclass
class ProcessResult:
    output: np.ndarray  # uint8 BGR for the virtual camera (mirrored iff general.mirror_output)
    preview: np.ndarray  # uint8 BGR for the GUI (mirrored iff mirror_preview; dots iff show_landmarks)
    original_preview: np.ndarray  # unprocessed frame, mirrored like preview (then a view, not contiguous)
    face_found: bool
    # "landmarks" (the wait for them), "segmentation", "to_float", each effect run, "clip", "to_uint8",
    # "compose", "total"
    timings_ms: dict[str, float]
    # Landmarks detected in this frame ((478, 2) px of the unmirrored frame).
    landmarks: np.ndarray | None = None


class Tracker(Protocol):
    def detect(self, frame_bgr: np.ndarray, ts_ms: int) -> tuple[np.ndarray | None, np.ndarray | None]: ...
    def set_smoothing(self, smoothing: float) -> None: ...
    def close(self) -> None: ...


class MaskSource(Protocol):
    def update(self, frame_bgr: np.ndarray, ts_ms: int, frame_index: int) -> SegMasks: ...
    def set_interval(self, interval: int) -> None: ...
    def close(self) -> None: ...


class _SettingsGlide:
    """The way from one look to another; see the module docstring."""

    def __init__(self, old: MakeupSettings, new: MakeupSettings, old_extra: dict[str, float] | None = None) -> None:
        self._old = old
        self._old_extra = {} if old_extra is None else old_extra
        self._new = new
        self._frame = 0
        self.now = (old, self._old_extra)  # what the last frame was drawn with

    def next(self) -> tuple[MakeupSettings, dict[str, float]] | None:
        """The settings of the next frame, and the sections' extra intensity. None: finished."""
        self._frame += 1
        if self._frame >= SETTINGS_GLIDE_FRAMES:
            return None
        t = self._frame / SETTINGS_GLIDE_FRAMES
        now = self._new.copy()
        now.general.intensity += (self._old.general.intensity - self._new.general.intensity) * (1.0 - t)
        extra: dict[str, float] = {}
        for name in SECTIONS:
            if name == "general":
                continue
            old, new = getattr(self._old, name), getattr(self._new, name)
            old_on, new_on = getattr(old, "enabled", True), getattr(new, "enabled", True)
            if old_on and new_on:
                _blend_section(getattr(now, name), old, new, t)
            elif old_on:
                setattr(now, name, copy.copy(old))
            if old_on or new_on:
                start = self._old_extra.get(name, 1.0) if old_on else 0.0
                level = start + ((1.0 if new_on else 0.0) - start) * t
                if level != 1.0:
                    extra[name] = level
        self.now = (now, extra)
        return now, extra


def _blend_section(out: Any, old: Any, new: Any, t: float) -> None:
    for field in dataclasses.fields(new):
        a, b = getattr(old, field.name), getattr(new, field.name)
        if isinstance(b, float) and isinstance(a, float):
            setattr(out, field.name, a + (b - a) * t)
        elif isinstance(b, str) and b.startswith("#") and isinstance(a, str) and a.startswith("#"):
            setattr(out, field.name, _blend_hex(a, b, t))
        elif t < 0.5:
            setattr(out, field.name, a)


def _blend_hex(a: str, b: str, t: float) -> str:
    channels = (round(int(a[i : i + 2], 16) * (1.0 - t) + int(b[i : i + 2], 16) * t) for i in (1, 3, 5))
    return "#" + "".join(f"{value:02X}" for value in channels)


@dataclass(frozen=True)
class _Tracked:
    """What the tracker said about one frame."""

    landmarks: np.ndarray | None
    z: np.ndarray | None
    coast: tuple[np.ndarray, np.ndarray] | None = None  # no face found: where the picture says it went


class _TrackAhead:
    """Runs the tracker on announced frames, before ``process`` gets to them.

    Nothing starts until the first frame is announced, and until then the tracker is simply
    called by whoever asks. From then on every use of the tracker goes through one worker
    thread, as the model must not run twice at a time.
    """

    def __init__(self, tracker: Tracker) -> None:
        self._tracker = tracker
        self._lock = threading.Lock()
        self._pool: ThreadPoolExecutor | None = None
        self._closed = False
        self._smoothing: float | None = None  # the value the tracker has (or will have, next in the queue)
        self._jobs: deque[tuple[np.ndarray, int, Future[_Tracked]]] = deque(maxlen=_MAX_TRACK_AHEAD)

    def announce(self, frame: np.ndarray, ts_ms: int, smoothing: float) -> None:
        """Start on this frame unless the last one is still running (it is then tracked when asked for).

        ``smoothing`` is what the frame is to be tracked with: the same as when ``process`` tracks it.
        """
        with self._lock:
            if self._closed or (self._jobs and not self._jobs[-1][2].done()):
                return
            if self._pool is None:
                self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tracker")
            if self._smoothing is not None and smoothing != self._smoothing:
                self._smoothing = smoothing
                self._pool.submit(self._tracker.set_smoothing, smoothing)
            self._jobs.append((frame, ts_ms, self._pool.submit(self._run, frame, ts_ms)))

    def detect(self, frame: np.ndarray, ts_ms: int) -> _Tracked:
        """Landmarks of this frame: from its job, or tracked now. Raises what the tracker raises."""
        with self._lock:
            future = self._take(frame, ts_ms)
            if future is None:
                if self._pool is None:
                    return self._run(frame, ts_ms)  # under the lock: a job cannot start meanwhile
                future = self._pool.submit(self._run, frame, ts_ms)
        return future.result()

    def _run(self, frame: np.ndarray, ts_ms: int) -> _Tracked:
        landmarks, z = self._tracker.detect(frame, ts_ms)
        if landmarks is not None:
            return _Tracked(landmarks, z)
        coast = getattr(self._tracker, "coast", None)  # not every tracker can tell
        return _Tracked(None, None, None if coast is None else coast())

    def set_smoothing(self, smoothing: float) -> None:
        with self._lock:
            if smoothing == self._smoothing:
                return
            self._smoothing = smoothing
            if self._pool is None:
                self._tracker.set_smoothing(smoothing)
                return
            future = self._pool.submit(self._tracker.set_smoothing, smoothing)
        future.result()

    def stop(self) -> None:
        """Wait for what is running; frames announced later are ignored."""
        with self._lock:
            self._closed = True
            pool, self._pool = self._pool, None
            self._jobs.clear()
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)

    def _take(self, frame: np.ndarray, ts_ms: int) -> Future[_Tracked] | None:
        """The job of this frame; it and all older ones are forgotten (frames the renderer skipped)."""
        for index, (announced, announced_ts, future) in enumerate(self._jobs):
            if announced is frame and announced_ts == ts_ms:
                for _ in range(index + 1):
                    self._jobs.popleft()
                return future
        return None


@dataclass
class _Face:
    """Landmarks makeup is drawn with, and the fade level (0..FADE_FRAMES)."""

    landmarks: np.ndarray | None = None
    z: np.ndarray | None = None
    seen_ms: float = 0.0
    level: int = 0
    steady: int = 0  # detections in a row
    step: float = 0.0  # x face scale: how far it moved between the last two detections
    velocity: np.ndarray | None = None  # set while it moved fast in both of the last two steps: px per frame


@dataclass(frozen=True)
class _Step:
    effect: Effect
    settings: Any
    intensity: float


class MakeupPipeline:
    """Turns camera frames into made-up frames; see the module docstring."""

    def __init__(
        self,
        settings: MakeupSettings | None = None,
        *,
        tracker: Tracker | None = None,
        segmenter: MaskSource | None = None,
        model_dir: Path | None = None,
        effects: list[Effect] | None = None,
        person_probe: plates.PersonProbe | None = None,
    ) -> None:
        self._settings = (settings or MakeupSettings()).copy()
        self._bypass = False
        self._glide: _SettingsGlide | None = None
        self._settings_lock = threading.Lock()
        self._process_lock = threading.Lock()
        _limit_cv2_threads()
        _use_one_thread_per_core()
        general = self._settings.general
        self._effects = _builtin_effects() if effects is None else sorted(effects, key=_effect_sort_key)
        if tracker is None:
            tracker = FaceTracker(ensure_model(FACE_LANDMARKER, directory=model_dir), general.landmark_smoothing)
        try:
            if segmenter is None:
                model = ensure_model(SELFIE_MULTICLASS, directory=model_dir)
                segmenter = Segmenter(model, general.segmentation_interval)
        except BaseException:
            tracker.close()
            raise
        self._tracker: Tracker = tracker
        self._track_ahead = _TrackAhead(tracker)
        self._segmenter: MaskSource = segmenter
        self._model_dir = model_dir
        self._person_probe = person_probe  # None: load the real model for each capture
        self._plate: np.ndarray | None = None
        self._plate_lock = threading.Lock()  # never held by process() except for an assignment
        self._plate_clears = 0  # counts clear_background: a capture begun before the last one is void
        self._plate_commits = 0  # counts plates put to use: an older one read from disk must not replace them
        self._unsaved_plates: dict[tuple[int, int], np.ndarray] = {}  # plates that could not be written, by (H, W)
        self._applied_model_settings: tuple[float, int] | None = None
        self._fast_for_masks = 0  # frames left with masks as often as possible
        self._face = _Face()
        self._face_lost_since_start = False  # a face was lost since the frame size or bypass last changed
        self._frame_shape: tuple[int, int] | None = None
        self._frame_index = 0
        self._float_frame: np.ndarray | None = None  # reused from frame to frame; holds nothing between them
        self._logged: set[tuple[str, str]] = set()
        self._closed = False

    # ---- thread-safe controls -------------------------------------------

    def set_settings(self, settings: MakeupSettings) -> None:
        new = settings.copy()
        with self._settings_lock:
            if self._glide is not None:
                start, extra = self._glide.now
                self._glide = _SettingsGlide(start, new, extra)
            self._settings = new

    def glide_to_settings(self, settings: MakeupSettings) -> None:
        """Like ``set_settings``, but without a jump in the picture (see the module docstring)."""
        new = settings.copy()
        with self._settings_lock:
            start, extra = (self._settings, None) if self._glide is None else self._glide.now
            self._glide = _SettingsGlide(start, new, extra)
            self._settings = new

    def get_settings(self) -> MakeupSettings:
        with self._settings_lock:
            return self._settings.copy()

    @property
    def bypass(self) -> bool:
        """True = frames pass through untouched (GUI "メイクOFF")."""
        with self._settings_lock:
            return self._bypass

    @bypass.setter
    def bypass(self, value: bool) -> None:
        with self._settings_lock:
            self._bypass = bool(value)

    @property
    def has_background(self) -> bool:
        """True while a background plate for the current resolution is in use."""
        return self._plate is not None

    @property
    def background_generation(self) -> int:
        """Changes with every ``clear_background``. Read it when the user asks for a capture
        (before the countdown) and pass it to ``capture_background``."""
        with self._plate_lock:
            return self._plate_clears

    def capture_background(self, frames: Sequence[np.ndarray], generation: int | None = None) -> plates.PlateResult:
        """Build, save and start using a plate from raw uint8 BGR camera frames.

        ``frames`` come from a ``background.PlateCollector``. Takes up to about
        a second (median, segmentation model): call it on a worker thread, not
        on a camera thread. Never raises; the result's ``message`` says what
        happened, in Japanese. A rejected capture keeps the previous plate.

        If the background is deleted after ``generation`` was read (by default:
        after this call began), nothing is saved or used and the result is not ok.
        """
        started = self.background_generation if generation is None else generation
        result = self._capture_plate(frames)
        if result.plate is None:
            return result
        with self._plate_lock:
            if self._plate_clears != started:
                return plates.PlateResult(False, plates.CAPTURE_CANCELLED_MESSAGE)
            return self._commit_plate(result)

    def clear_background(self) -> plates.PlateResult:
        """Stop using the plate and delete every saved one. Never raises."""
        with self._plate_lock:
            self._plate_clears += 1
            self._plate = None
            self._unsaved_plates.clear()
            return plates.clear_plates()

    # ---- processing -------------------------------------------------------

    def prepare(self, frame_bgr: np.ndarray, timestamp_ms: int) -> None:
        """Announce a new camera frame that will (probably) reach ``process`` with this very array
        and timestamp: its landmarks start right away, on their own thread. Returns at once and
        never raises; call it from the camera thread. Do not write into the frame afterwards."""
        try:
            if _is_frame(frame_bgr):
                with self._settings_lock:
                    smoothing = self._settings.general.landmark_smoothing
                self._track_ahead.announce(frame_bgr, int(timestamp_ms), smoothing)
        except Exception as exc:
            self._log_once("tracker", exc)

    def process(self, frame_bgr: np.ndarray, timestamp_ms: int | None = None) -> ProcessResult:
        """Process one uint8 BGR frame (HxWx3). ``timestamp_ms`` defaults to a monotonic clock."""
        frame = _validated_frame(frame_bgr)
        start = time.perf_counter()
        ts_ms = int(time.monotonic() * 1000.0) if timestamp_ms is None else int(timestamp_ms)
        with self._settings_lock:
            settings, bypass = self._settings, self._bypass
            step = None if self._glide is None else self._glide.next()
            if step is None:
                self._glide = None
        extra = None
        if step is not None:
            settings, extra = step
        timings = {"landmarks": 0.0, "segmentation": 0.0}
        with self._process_lock:
            if self._closed:
                rendered, landmarks = frame, None
            else:
                rendered, landmarks = self._process_locked(frame, ts_ms, settings, bypass, timings, extra)
        compose_start = time.perf_counter()
        result = _compose(frame, rendered, landmarks, settings.general, timings)
        timings["compose"] = _ms_since(compose_start)
        timings["total"] = _ms_since(start)
        return result

    def close(self) -> None:
        """Release the models; later ``process`` calls pass frames through."""
        with self._process_lock:
            if self._closed:
                return
            self._closed = True
            self._track_ahead.stop()
            for name, resource in (("tracker", self._tracker), ("segmenter", self._segmenter)):
                try:
                    resource.close()
                except Exception as exc:  # closing must never raise into the GUI
                    self._log_once(name, exc)

    # ---- internals --------------------------------------------------------

    def _process_locked(
        self,
        frame: np.ndarray,
        ts_ms: int,
        settings: MakeupSettings,
        bypass: bool,
        timings: dict[str, float],
        extra: dict[str, float] | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        self._frame_index += 1
        self._on_frame_shape(frame.shape[:2])
        self._apply_model_settings(settings.general)
        tracked = self._track(frame, ts_ms, timings)
        landmarks = tracked.landmarks
        if bypass:
            self._forget_face()
            self._face_lost_since_start = False
            return frame, landmarks
        fade = self._update_face(tracked, ts_ms)
        steps = self._plan(settings, fade, extra or {})
        if not steps:
            return frame, landmarks
        return self._render(frame, ts_ms, steps, timings), landmarks

    def _on_frame_shape(self, shape: tuple[int, int]) -> None:
        if shape != self._frame_shape:
            self._frame_shape = shape
            self._face = _Face()
            self._face_lost_since_start = False
            self._reset_effects()
            self._load_plate(shape)

    def _load_plate(self, shape: tuple[int, int]) -> None:
        """Use the plate of this resolution, if any (a plate of another size is invalid)."""
        with self._plate_lock:
            self._plate = self._unsaved_plates.get(shape)
            if self._plate is not None:
                return
            state = (self._plate_clears, self._plate_commits)
        try:
            plate = plates.load_plate(shape[1], shape[0])  # slow: outside the lock
        except Exception as exc:
            self._log_once("background", exc)
            return
        if plate is None:
            return
        image = plates.plate_to_float(plate)
        with self._plate_lock:
            if state == (self._plate_clears, self._plate_commits) and shape == self._frame_shape:
                self._plate = image

    def _commit_plate(self, result: plates.PlateResult) -> plates.PlateResult:
        """Save a captured plate and put it to use; the caller holds the plate lock."""
        shape = result.plate.shape[:2]
        image = plates.plate_to_float(result.plate)
        self._plate_commits += 1
        try:
            plates.save_plate(result.plate)
            self._unsaved_plates.pop(shape, None)
        except (OSError, ValueError, cv2.error) as exc:
            self._log_once("background", exc)
            # Older files on the disk must not come back when the resolution changes and returns.
            self._unsaved_plates[shape] = image
            result = plates.PlateResult(True, plates.SUCCESS_NOT_SAVED_MESSAGE, result.plate, result.person_fraction)
        if self._frame_shape is None or self._frame_shape == shape:
            self._plate = image
        return result

    def _capture_plate(self, frames: Sequence[np.ndarray]) -> plates.PlateResult:
        probe, owned = self._person_probe, None
        try:
            if probe is None:
                probe = owned = plates.MediaPipePersonProbe(ensure_model(SELFIE_MULTICLASS, directory=self._model_dir))
            return plates.capture(frames, probe)
        except Exception as exc:
            self._log_once("background", exc)
            return plates.PlateResult(False, plates.CHECK_FAILED_MESSAGE)
        finally:
            if owned is not None:
                owned.close()

    def _apply_model_settings(self, general: GeneralSettings) -> None:
        if self._face.step > _FAST_FOR_MASKS_STEP:
            self._fast_for_masks = _FAST_FOR_MASKS_FRAMES
        elif self._fast_for_masks > 0:
            self._fast_for_masks -= 1
        wanted = (general.landmark_smoothing, 1 if self._fast_for_masks > 0 else general.segmentation_interval)
        if wanted != self._applied_model_settings:
            self._applied_model_settings = wanted
            self._track_ahead.set_smoothing(wanted[0])
            self._segmenter.set_interval(wanted[1])

    def _track(self, frame: np.ndarray, ts_ms: int, timings: dict[str, float]) -> _Tracked:
        start = time.perf_counter()
        try:
            tracked = self._track_ahead.detect(frame, ts_ms)
        except Exception as exc:
            self._log_once("tracker", exc)
            tracked = _Tracked(None, None)
        timings["landmarks"] = _ms_since(start)
        for array in (tracked.landmarks, tracked.z, *(tracked.coast or ())):
            if array is not None:
                array.setflags(write=False)  # kept for the fade-out; effects must not edit it
        return tracked

    def _update_face(self, tracked: _Tracked, ts_ms: int) -> float:
        """Advance the fade and choose the landmarks to draw with; returns the fade (0..1)."""
        face = self._face
        if tracked.landmarks is None and face.landmarks is not None:
            self._face_lost_since_start = True
        if tracked.landmarks is not None:
            step, moved = _face_step(face.landmarks, tracked.landmarks)
            steady = face.steady + 1
            # A single lost frame is forgiven at once: no face was really gone.
            may_rise = steady >= STEADY_FRAMES or not self._face_lost_since_start or face.level >= FADE_FRAMES - 1
            level = min(face.level + 1, FADE_FRAMES) if may_rise else face.level
            velocity = moved if step > _FAST_FACE_STEP and face.step > _FAST_FACE_STEP else None
            self._face = _Face(tracked.landmarks, tracked.z, ts_ms, level, steady, step, velocity)
        elif face.landmarks is not None and face.level > 1 and ts_ms - face.seen_ms <= MAX_LANDMARK_AGE_MS:
            if tracked.coast is not None:
                face.landmarks, face.z = tracked.coast
            elif face.velocity is not None:
                face.landmarks = face.landmarks + face.velocity
                face.level = max(face.level - 1, 1)  # one more below
            face.level -= 1
            face.steady = 0
        else:
            self._forget_face()
        return self._face.level / FADE_FRAMES

    def _forget_face(self) -> None:
        if self._face.landmarks is not None:
            self._reset_effects()
        self._face = _Face()

    def _reset_effects(self) -> None:
        for effect in self._effects:
            try:
                effect.reset()
            except Exception as exc:
                self._log_once(effect.name, exc)

    def _plan(self, settings: MakeupSettings, fade: float, extra: dict[str, float]) -> list[_Step]:
        """Effects to run this frame, with their intensity."""
        has_face = self._face.landmarks is not None
        steps = []
        for effect in self._effects:
            section = getattr(settings, effect.name, None)
            if section is None:
                self._log_once(effect.name, LookupError(f"no settings section named {effect.name!r}"))
                continue
            if not getattr(section, "enabled", True) or (effect.needs_face and not has_face):
                continue
            intensity = settings.general.intensity * (fade if effect.needs_face else 1.0) * extra.get(effect.name, 1.0)
            if intensity > 0.0:
                steps.append(_Step(effect, section, intensity))
        return steps

    def _render(self, frame: np.ndarray, ts_ms: int, steps: list[_Step], timings: dict[str, float]) -> np.ndarray:
        masks = SegMasks()
        if any(_uses_masks(step.effect) for step in steps):
            masks = self._segment(frame, ts_ms, timings)
        height, width = frame.shape[:2]
        plate = self._plate  # another thread may replace it: read it once
        ctx = FrameContext(
            height=height,
            width=width,
            landmarks=self._face.landmarks,
            landmarks_z=self._face.z,
            hair_mask=masks.hair,
            face_skin_mask=masks.face_skin,
            body_skin_mask=masks.body_skin,
            background=plate if plate is not None and plate.shape[:2] == (height, width) else None,
        )
        start = time.perf_counter()
        image = self._to_float(frame)
        timings["to_float"] = _ms_since(start)
        for step in steps:
            if step.effect.needs_hair_mask and ctx.hair_mask is None:
                continue
            ctx.intensity = step.intensity
            image = self._run_effect(step, image, ctx, timings)
        start = time.perf_counter()
        # Only the low end: the conversion below stops at 255 by itself, but takes absolute values. Half the time of np.clip.
        cv2.max(image, 0.0, dst=image)
        timings["clip"] = _ms_since(start)
        start = time.perf_counter()
        rendered = cv2.convertScaleAbs(image, alpha=255.0)
        timings["to_uint8"] = _ms_since(start)
        return rendered

    def _to_float(self, frame: np.ndarray) -> np.ndarray:
        """The frame as float32 in [0, 1], in the buffer of the last frame (the process lock is held)."""
        if self._float_frame is None or self._float_frame.shape != frame.shape:
            self._float_frame = np.empty(frame.shape, np.float32)
        return cv2.LUT(frame, _TO_FLOAT, dst=self._float_frame)

    def _segment(self, frame: np.ndarray, ts_ms: int, timings: dict[str, float]) -> SegMasks:
        start = time.perf_counter()
        try:
            masks = self._segmenter.update(frame, ts_ms, self._frame_index)
        except Exception as exc:
            self._log_once("segmenter", exc)
            masks = SegMasks()
        timings["segmentation"] = _ms_since(start)
        return masks

    def _run_effect(self, step: _Step, image: np.ndarray, ctx: FrameContext, timings: dict[str, float]) -> np.ndarray:
        start = time.perf_counter()
        try:
            result = step.effect.apply(image, ctx, step.settings)
            if not (isinstance(result, np.ndarray) and result.dtype == np.float32 and result.shape == image.shape):
                raise TypeError(f"apply() must return a float32 array of shape {image.shape}")
            image = result
        except Exception as exc:
            self._log_once(step.effect.name, exc)
        timings[step.effect.name] = _ms_since(start)
        return image

    def _log_once(self, source: str, exc: BaseException) -> None:
        key = (source, f"{type(exc).__name__}: {exc}")
        if key in self._logged or len(self._logged) >= _MAX_LOGGED_ERRORS:
            return
        self._logged.add(key)
        logger.error("%s failed and is skipped for this frame: %s", source, key[1], exc_info=exc)


def _limit_cv2_threads() -> None:
    """Applies to the whole process, as OpenCV has no other setting. A lower value is kept."""
    if cv2.getNumThreads() > CV2_THREADS:
        cv2.setNumThreads(CV2_THREADS)


def _use_one_thread_per_core() -> None:
    """Keep this process off the second hardware thread of each CPU core.

    The segmentation model runs all the time next to the renderer. Where they share a core,
    each gets about half of it: every step of a frame then takes up to twice as long, about one
    frame in four. Off the shared threads the median frame takes 9 % and the slowest 5 % take
    10 % less time, and the model 10 % less, with no less room for other programs (these threads
    are theirs). Windows only; nothing happens where the layout is unknown. Never raises.
    """
    if sys.platform != "win32":
        return
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        process = kernel32.GetCurrentProcess
        process.restype = ctypes.c_void_p
        allowed, system = ctypes.c_size_t(), ctypes.c_size_t()
        kernel32.GetProcessAffinityMask.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_size_t),
        ]
        if not kernel32.GetProcessAffinityMask(process(), ctypes.byref(allowed), ctypes.byref(system)):
            return
        wanted = allowed.value & _first_thread_of_each_core(kernel32)
        if wanted and wanted != allowed.value:
            kernel32.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            kernel32.SetProcessAffinityMask(process(), wanted)
    except (OSError, AttributeError, ValueError) as exc:
        logger.debug("could not set the CPU affinity: %s", exc)


class _LogicalProcessorInfo(ctypes.Structure):
    """SYSTEM_LOGICAL_PROCESSOR_INFORMATION."""

    _fields_ = [("mask", ctypes.c_size_t), ("relationship", ctypes.c_int), ("detail", ctypes.c_ulonglong * 2)]


def _first_thread_of_each_core(kernel32: ctypes.WinDLL) -> int:
    """Affinity mask with the lowest hardware thread of every core (0 if unknown)."""
    size = ctypes.c_ulong(0)
    kernel32.GetLogicalProcessorInformation(None, ctypes.byref(size))
    count = size.value // ctypes.sizeof(_LogicalProcessorInfo)
    if count == 0:
        return 0
    entries = (_LogicalProcessorInfo * count)()
    if not kernel32.GetLogicalProcessorInformation(entries, ctypes.byref(size)):
        return 0
    mask = 0
    for entry in entries:
        if entry.relationship == _RELATION_PROCESSOR_CORE and entry.mask:
            mask |= entry.mask & -entry.mask
    return mask


def _builtin_effects() -> list[Effect]:
    from .effects import load_builtin_effects  # imported lazily: effect modules are heavy

    return [effect_cls() for effect_cls in load_builtin_effects()]


def _effect_sort_key(effect: Effect) -> tuple[int, str]:
    return (effect.order, effect.name)


def _uses_masks(effect: Effect) -> bool:
    return effect.needs_hair_mask or effect.name in OPTIONAL_MASK_USERS


def _face_step(before: np.ndarray | None, after: np.ndarray) -> tuple[float, np.ndarray | None]:
    """How far a face moved between two sets of landmarks: x face scale, and the shift in pixels."""
    if before is None or before.shape != after.shape:
        return 0.0, None
    shift = np.median(after - before, axis=0)
    return float(np.linalg.norm(shift)) / max(regions.face_scale(after), 1.0), shift


def _is_frame(frame: Any) -> bool:
    return isinstance(frame, np.ndarray) and frame.dtype == np.uint8 and frame.ndim == 3 and frame.shape[2] == 3


def _validated_frame(frame: np.ndarray) -> np.ndarray:
    if not _is_frame(frame):
        raise ValueError("frame must be a uint8 BGR array of shape (H, W, 3)")
    return np.ascontiguousarray(frame)


def _compose(
    frame: np.ndarray,
    rendered: np.ndarray,
    landmarks: np.ndarray | None,
    general: GeneralSettings,
    timings: dict[str, float],
) -> ProcessResult:
    """Mirror and annotate the frames. The arrays may share memory: treat them as read-only.

    The original is mirrored as a view (negative strides, no copy): it is only shown while comparing,
    and then costs its 2.5 ms on the GUI thread. The preview, shown all the time, gets a real flip (0.4 ms).
    """
    preview = rendered
    if general.show_landmarks and landmarks is not None:
        preview = draw_landmarks(rendered, landmarks)
    return ProcessResult(
        output=cv2.flip(rendered, 1) if general.mirror_output else rendered,
        preview=cv2.flip(preview, 1) if general.mirror_preview else preview,
        original_preview=frame[:, ::-1] if general.mirror_preview else frame,
        face_found=landmarks is not None,
        timings_ms=timings,
        landmarks=landmarks,
    )


def draw_landmarks(frame: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """Copy of ``frame`` with landmark dots (mesh green, iris yellow)."""
    out = frame.copy()
    radius = max(1, min(frame.shape[:2]) // 400)
    for index, (x, y) in enumerate(np.round(landmarks).astype(int)):
        color = _IRIS_COLOR if index >= _FIRST_IRIS_INDEX else _MESH_COLOR
        cv2.circle(out, (int(x), int(y)), radius, color, -1, lineType=cv2.LINE_AA)
    return out


def _ms_since(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0
