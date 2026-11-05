"""Per-frame processing: tracking -> segmentation -> effects -> output frames.

``MakeupPipeline.process`` runs on the camera worker thread; settings and
bypass may be changed from the GUI thread at any time.

Never-crash policy: the tracker, the segmenter and every effect are isolated.
An exception is logged once per source and message, and that step is skipped
for the frame, so a bug degrades the makeup instead of breaking the call.

Fading: face-dependent effects fade in over ``FADE_FRAMES`` frames when a
face appears, and fade out when it is lost, using the last landmarks (never
older than ``MAX_LANDMARK_AGE_MS``), so makeup never pops on or off. The fade
scales each face effect's ``ctx.intensity``: every effect multiplies its
opacity by it and is an exact no-op at 0, so this works like blending the
processed and original frame per effect, without copying frames. Effects
that do not need a face (hair) are not faded, because they do not depend on
it. Constructing the pipeline without ``tracker`` / ``segmenter`` loads (and
if needed downloads) the models, which may raise ``ModelDownloadError``.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

from .config import GeneralSettings, MakeupSettings
from .effects.base import Effect, FrameContext
from .models import FACE_LANDMARKER, SELFIE_MULTICLASS, ensure_model
from .segmentation import SegMasks, Segmenter
from .tracking import FaceTracker

logger = logging.getLogger(__name__)

FADE_FRAMES = 6
MAX_LANDMARK_AGE_MS = 500.0
# Effects that use segmentation masks when available without requiring them
# (the skin effect refines its area with the face-skin mask).
OPTIONAL_MASK_USERS = frozenset({"skin"})

_MAX_LOGGED_ERRORS = 256
_MESH_COLOR = (0, 255, 0)
_IRIS_COLOR = (0, 255, 255)
_FIRST_IRIS_INDEX = 468


@dataclass
class ProcessResult:
    output: np.ndarray  # uint8 BGR for the virtual camera (mirrored iff general.mirror_output)
    preview: np.ndarray  # uint8 BGR for the GUI (mirrored iff mirror_preview; dots iff show_landmarks)
    original_preview: np.ndarray  # unprocessed frame, mirrored like preview
    face_found: bool
    timings_ms: dict[str, float]  # "landmarks", "segmentation", each effect run, "total"
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


@dataclass
class _Face:
    """Landmarks makeup is drawn with, and the fade level (0..FADE_FRAMES)."""

    landmarks: np.ndarray | None = None
    z: np.ndarray | None = None
    seen_ms: float = 0.0
    level: int = 0


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
    ) -> None:
        self._settings = (settings or MakeupSettings()).copy()
        self._bypass = False
        self._settings_lock = threading.Lock()
        self._process_lock = threading.Lock()
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
        self._segmenter: MaskSource = segmenter
        self._applied_model_settings: tuple[float, int] | None = None
        self._face = _Face()
        self._frame_shape: tuple[int, int] | None = None
        self._frame_index = 0
        self._logged: set[tuple[str, str]] = set()
        self._closed = False

    # ---- thread-safe controls -------------------------------------------

    def set_settings(self, settings: MakeupSettings) -> None:
        copy = settings.copy()
        with self._settings_lock:
            self._settings = copy

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

    # ---- processing -------------------------------------------------------

    def process(self, frame_bgr: np.ndarray, timestamp_ms: int | None = None) -> ProcessResult:
        """Process one uint8 BGR frame (HxWx3). ``timestamp_ms`` defaults to a monotonic clock."""
        frame = _validated_frame(frame_bgr)
        start = time.perf_counter()
        ts_ms = int(time.monotonic() * 1000.0) if timestamp_ms is None else int(timestamp_ms)
        with self._settings_lock:
            settings, bypass = self._settings, self._bypass
        timings = {"landmarks": 0.0, "segmentation": 0.0}
        with self._process_lock:
            if self._closed:
                rendered, landmarks = frame, None
            else:
                rendered, landmarks = self._process_locked(frame, ts_ms, settings, bypass, timings)
        result = _compose(frame, rendered, landmarks, settings.general, timings)
        timings["total"] = _ms_since(start)
        return result

    def close(self) -> None:
        """Release the models; later ``process`` calls pass frames through."""
        with self._process_lock:
            if self._closed:
                return
            self._closed = True
            for name, resource in (("tracker", self._tracker), ("segmenter", self._segmenter)):
                try:
                    resource.close()
                except Exception as exc:  # closing must never raise into the GUI
                    self._log_once(name, exc)

    # ---- internals --------------------------------------------------------

    def _process_locked(
        self, frame: np.ndarray, ts_ms: int, settings: MakeupSettings, bypass: bool, timings: dict[str, float]
    ) -> tuple[np.ndarray, np.ndarray | None]:
        self._frame_index += 1
        self._on_frame_shape(frame.shape[:2])
        self._apply_model_settings(settings.general)
        landmarks, z = self._track(frame, ts_ms, timings)
        if bypass:
            self._forget_face()
            return frame, landmarks
        fade = self._update_face(landmarks, z, ts_ms)
        steps = self._plan(settings, fade)
        if not steps:
            return frame, landmarks
        return self._render(frame, ts_ms, steps, timings), landmarks

    def _on_frame_shape(self, shape: tuple[int, int]) -> None:
        if shape != self._frame_shape:
            self._frame_shape = shape
            self._face = _Face()
            self._reset_effects()

    def _apply_model_settings(self, general: GeneralSettings) -> None:
        wanted = (general.landmark_smoothing, general.segmentation_interval)
        if wanted != self._applied_model_settings:
            self._applied_model_settings = wanted
            self._tracker.set_smoothing(wanted[0])
            self._segmenter.set_interval(wanted[1])

    def _track(
        self, frame: np.ndarray, ts_ms: int, timings: dict[str, float]
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        start = time.perf_counter()
        try:
            landmarks, z = self._tracker.detect(frame, ts_ms)
        except Exception as exc:
            self._log_once("tracker", exc)
            landmarks, z = None, None
        timings["landmarks"] = _ms_since(start)
        for array in (landmarks, z):
            if array is not None:
                array.setflags(write=False)  # kept for the fade-out; effects must not edit it
        return landmarks, z

    def _update_face(self, landmarks: np.ndarray | None, z: np.ndarray | None, ts_ms: int) -> float:
        """Advance the fade and choose the landmarks to draw with; returns the fade (0..1)."""
        face = self._face
        if landmarks is not None:
            self._face = _Face(landmarks, z, ts_ms, min(face.level + 1, FADE_FRAMES))
        elif face.landmarks is not None and face.level > 1 and ts_ms - face.seen_ms <= MAX_LANDMARK_AGE_MS:
            face.level -= 1
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

    def _plan(self, settings: MakeupSettings, fade: float) -> list[_Step]:
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
            intensity = settings.general.intensity * (fade if effect.needs_face else 1.0)
            if intensity > 0.0:
                steps.append(_Step(effect, section, intensity))
        return steps

    def _render(self, frame: np.ndarray, ts_ms: int, steps: list[_Step], timings: dict[str, float]) -> np.ndarray:
        masks = SegMasks()
        if any(_uses_masks(step.effect) for step in steps):
            masks = self._segment(frame, ts_ms, timings)
        height, width = frame.shape[:2]
        ctx = FrameContext(
            height=height,
            width=width,
            landmarks=self._face.landmarks,
            landmarks_z=self._face.z,
            hair_mask=masks.hair,
            face_skin_mask=masks.face_skin,
            body_skin_mask=masks.body_skin,
        )
        image = np.multiply(frame, np.float32(1.0 / 255.0), dtype=np.float32)
        for step in steps:
            if step.effect.needs_hair_mask and ctx.hair_mask is None:
                continue
            ctx.intensity = step.intensity
            image = self._run_effect(step, image, ctx, timings)
        np.clip(image, 0.0, 1.0, out=image)
        return cv2.convertScaleAbs(image, alpha=255.0)

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


def _builtin_effects() -> list[Effect]:
    from .effects import load_builtin_effects  # imported lazily: effect modules are heavy

    return [effect_cls() for effect_cls in load_builtin_effects()]


def _effect_sort_key(effect: Effect) -> tuple[int, str]:
    return (effect.order, effect.name)


def _uses_masks(effect: Effect) -> bool:
    return effect.needs_hair_mask or effect.name in OPTIONAL_MASK_USERS


def _validated_frame(frame: np.ndarray) -> np.ndarray:
    if not (isinstance(frame, np.ndarray) and frame.dtype == np.uint8 and frame.ndim == 3 and frame.shape[2] == 3):
        raise ValueError("frame must be a uint8 BGR array of shape (H, W, 3)")
    return np.ascontiguousarray(frame)


def _compose(
    frame: np.ndarray,
    rendered: np.ndarray,
    landmarks: np.ndarray | None,
    general: GeneralSettings,
    timings: dict[str, float],
) -> ProcessResult:
    """Mirror and annotate the frames. The arrays may share memory: treat them as read-only."""
    preview = rendered
    if general.show_landmarks and landmarks is not None:
        preview = draw_landmarks(rendered, landmarks)
    return ProcessResult(
        output=cv2.flip(rendered, 1) if general.mirror_output else rendered,
        preview=cv2.flip(preview, 1) if general.mirror_preview else preview,
        original_preview=cv2.flip(frame, 1) if general.mirror_preview else frame,
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
