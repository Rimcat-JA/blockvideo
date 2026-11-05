"""UI-agnostic worker: camera -> makeup pipeline -> virtual camera.

Two daemon threads per ``start()``:

- capture thread: owns the camera. Opens it (retrying every
  ``retry_interval_s``), reads frames as fast as the camera delivers them and
  keeps only the newest in a one-slot mailbox. Slow driver calls (opening a
  camera can take seconds on Windows) block only this thread.
- output thread: owns the virtual camera. Takes the newest frame, runs the
  pipeline and sends the result to the virtual camera and to ``on_frame``.
  Stale frames are dropped, so latency never builds up when processing is
  slow. When no frame has arrived for ``camera_timeout_s`` it sends a neutral
  placeholder at the target fps instead, so the call never shows a frozen
  picture.

Commands from other threads only update locked fields; the thread that owns
the device applies them. Settings and bypass go straight to the pipeline,
which is thread-safe. No exception kills either thread: failures are logged
once and reported in ``RunnerStatus``.
"""

from __future__ import annotations

import dataclasses
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from typing import Generic, Protocol, TypeVar

import cv2
import numpy as np

from .camera import CameraError, CameraSource, VirtualCameraError, VirtualCameraOutput
from .config import MakeupSettings
from .pipeline import MakeupPipeline, ProcessResult

logger = logging.getLogger(__name__)

RETRY_INTERVAL_S = 2.0
CAMERA_TIMEOUT_S = 1.0
STATUS_INTERVAL_S = 0.5
# OpenCV's built-in fonts cannot draw Japanese, so the placeholder text is ASCII.
PLACEHOLDER_TEXT = "Camera reconnecting..."

CAMERA_LOST_MESSAGE = "カメラの映像が届いていません。再接続を試みています…"
CAMERA_READ_FAILED_MESSAGE = "カメラの読み取りでエラーが発生しました。再接続を試みています…"
PROCESSING_FAILED_MESSAGE = (
    "メイク処理でエラーが発生したため、加工なしの映像を送っています（詳細はログを確認してください）。"
)

_EMA_ALPHA = 0.1
_POLL_S = 0.05
_MISS_WAIT_S = 0.01
_MAX_LOGGED_ERRORS = 256
_PLACEHOLDER_BG = (38, 38, 38)
_PLACEHOLDER_FG = (150, 150, 150)

T = TypeVar("T")


class FrameSource(Protocol):
    """What the runner needs from a camera (``CameraSource`` or a test fake)."""

    def open(self) -> None: ...
    def read(self) -> np.ndarray | None: ...
    def close(self) -> None: ...


class FrameSink(Protocol):
    """What the runner needs from a virtual camera (``VirtualCameraOutput`` or a fake)."""

    device: str

    def send(self, frame_bgr: np.ndarray) -> None: ...
    def close(self) -> None: ...


CameraFactory = Callable[..., FrameSource]  # called as (index=, width=, height=, fps=)
VcamFactory = Callable[..., FrameSink]  # called as (width=, height=, fps=)


@dataclass(frozen=True)
class RunnerStatus:
    fps: float = 0.0
    camera_ok: bool = False
    camera_error: str | None = None
    virtual_camera: str | None = None  # device name while the virtual camera runs
    virtual_camera_error: str | None = None
    face_found: bool = False
    processing_ms: float = 0.0
    # Set while pipeline.process() fails and raw frames are sent instead.
    processing_error: str | None = None


# --------------------------------------------------------------------------
# Small thread-safe building blocks
# --------------------------------------------------------------------------


class _Locked(Generic[T]):
    """A value shared between threads (store immutable values only)."""

    def __init__(self, value: T) -> None:
        self._lock = threading.Lock()
        self._value = value

    def get(self) -> T:
        with self._lock:
            return self._value

    def set(self, value: T) -> None:
        with self._lock:
            self._value = value

    def update(self, change: Callable[[T], T]) -> T:
        with self._lock:
            self._value = change(self._value)
            return self._value


@dataclass(frozen=True)
class _Captured:
    frame: np.ndarray
    timestamp_ms: int
    seq: int


class _FrameSlot:
    """One-slot mailbox: the capture thread overwrites, the output thread takes the newest."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._latest: _Captured | None = None
        self._closed = False

    def put(self, frame: np.ndarray, timestamp_ms: int) -> None:
        with self._cond:
            seq = 1 if self._latest is None else self._latest.seq + 1
            self._latest = _Captured(frame, timestamp_ms, seq)
            self._cond.notify_all()

    def take_newer(self, seq: int, timeout: float) -> _Captured | None:
        """The newest frame if it is newer than ``seq``; waits up to ``timeout`` seconds."""
        with self._cond:
            self._cond.wait_for(lambda: self._closed or self._has_newer(seq), timeout)
            return self._latest if self._has_newer(seq) and not self._closed else None

    def close(self) -> None:
        """Wake the waiting output thread for good (used by stop())."""
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    def _has_newer(self, seq: int) -> bool:
        return self._latest is not None and self._latest.seq > seq


class _ErrorLog:
    """Logs each distinct failure once; most would otherwise repeat every frame."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._seen: set[tuple[str, str]] = set()

    def report(self, source: str, exc: BaseException) -> None:
        key = (source, f"{type(exc).__name__}: {exc}")
        with self._lock:
            if key in self._seen or len(self._seen) >= _MAX_LOGGED_ERRORS:
                return
            self._seen.add(key)
        if isinstance(exc, (CameraError, VirtualCameraError)):
            logger.warning("%s: %s", source, exc)  # expected (no camera, no OBS...): no traceback
        else:
            logger.error("%s failed: %s", source, key[1], exc_info=exc)


class _Ema:
    def __init__(self, alpha: float = _EMA_ALPHA) -> None:
        self._alpha = alpha
        self.value: float | None = None

    def update(self, sample: float) -> None:
        self.value = sample if self.value is None else self.value + self._alpha * (sample - self.value)

    def reset(self) -> None:
        self.value = None


class _Stats:
    """Smoothed rate of processed camera frames and pipeline time."""

    def __init__(self) -> None:
        self._interval_s = _Ema()
        self._processing_ms = _Ema()
        self._last_at: float | None = None

    @property
    def fps(self) -> float:
        interval = self._interval_s.value
        return 1.0 / interval if interval else 0.0

    @property
    def processing_ms(self) -> float:
        return self._processing_ms.value or 0.0

    def record(self, now: float, processing_ms: float) -> None:
        if self._last_at is not None:
            self._interval_s.update(now - self._last_at)
        self._last_at = now
        self._processing_ms.update(processing_ms)

    def reset(self) -> None:
        self._interval_s.reset()
        self._processing_ms.reset()
        self._last_at = None


# --------------------------------------------------------------------------
# Requests and wiring shared by the runner and its threads
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _CameraRequest:
    index: int
    width: int
    height: int
    fps: int
    generation: int  # bumped by every set_camera(), so re-selecting a camera reconnects it


@dataclass(frozen=True)
class _VcamRequest:
    enabled: bool
    width: int
    height: int
    fps: int


@dataclass(frozen=True)
class _Wiring:
    """Everything the two worker threads share with the runner."""

    pipeline: MakeupPipeline
    camera_factory: CameraFactory
    vcam_factory: VcamFactory
    on_frame: Callable[[ProcessResult], None] | None
    on_status: Callable[[RunnerStatus], None] | None
    camera_request: _Locked[_CameraRequest]
    vcam_request: _Locked[_VcamRequest]
    camera_error: _Locked[str | None]  # why the camera is not open (set by the capture thread)
    status: _Locked[RunnerStatus]
    errors: _ErrorLog
    retry_interval_s: float
    camera_timeout_s: float


# --------------------------------------------------------------------------
# Capture thread
# --------------------------------------------------------------------------


class _CaptureWorker:
    """Owns the camera: opens it (retrying), applies switches, publishes frames."""

    def __init__(self, wiring: _Wiring, stop: threading.Event, slot: _FrameSlot) -> None:
        self._w = wiring
        self._stop = stop
        self._slot = slot
        self._camera: FrameSource | None = None
        self._applied: _CameraRequest | None = None
        self._next_open_at = 0.0

    def run(self) -> None:
        try:
            while not self._stop.is_set():
                try:
                    self._step()
                except Exception as exc:  # last line of defence: keep the thread alive
                    self._w.errors.report("capture", exc)
                    self._stop.wait(_POLL_S)
        finally:
            self._close()

    def _step(self) -> None:
        request = self._w.camera_request.get()
        if request != self._applied:
            self._close()
            self._applied, self._next_open_at = request, 0.0
            self._w.camera_error.set(None)
        if self._camera is None:
            self._open_when_due(request)
        else:
            self._read(self._camera)

    def _open_when_due(self, request: _CameraRequest) -> None:
        if time.monotonic() < self._next_open_at:
            self._stop.wait(_POLL_S)  # short waits keep set_camera() and stop() responsive
            return
        camera: FrameSource | None = None
        try:
            camera = self._w.camera_factory(
                index=request.index, width=request.width, height=request.height, fps=request.fps
            )
            camera.open()
        except Exception as exc:
            if camera is not None:
                _close_quietly(camera, "camera", self._w.errors)
            self._failed(exc, _describe(exc, CameraError, "カメラを開けませんでした"))
            return
        self._camera = camera
        self._w.camera_error.set(None)

    def _read(self, camera: FrameSource) -> None:
        try:
            frame = camera.read()
        except Exception as exc:  # a broken camera object: recreate it
            self._close()
            self._failed(exc, CAMERA_READ_FAILED_MESSAGE)
            return
        if frame is None:
            self._stop.wait(_MISS_WAIT_S)  # do not spin on a failing device
        else:
            self._slot.put(frame, int(time.monotonic() * 1000.0))

    def _failed(self, exc: Exception, message: str) -> None:
        self._w.errors.report("camera", exc)
        self._w.camera_error.set(message)
        self._next_open_at = time.monotonic() + self._w.retry_interval_s

    def _close(self) -> None:
        camera, self._camera = self._camera, None
        if camera is not None:
            _close_quietly(camera, "camera", self._w.errors)


# --------------------------------------------------------------------------
# Output thread
# --------------------------------------------------------------------------


class _VcamLink:
    """The virtual camera on the output thread: opened on demand, retried, reopened on failure."""

    def __init__(self, wiring: _Wiring) -> None:
        self._w = wiring
        self._request = wiring.vcam_request.get()
        self._vcam: FrameSink | None = None
        self._device: str | None = None
        self._next_open_at = 0.0
        self.error: str | None = None

    @property
    def device(self) -> str | None:
        return self._device

    @property
    def size(self) -> tuple[int, int]:
        return self._request.width, self._request.height

    def sync(self, now: float) -> None:
        """Apply the latest request and (re)open the device when due."""
        request = self._w.vcam_request.get()
        if request != self._request:
            self.close()
            self._request, self._next_open_at, self.error = request, now, None
        if request.enabled and self._vcam is None and now >= self._next_open_at:
            self._open(now)

    def send(self, frame: np.ndarray, now: float) -> None:
        if self._vcam is None:
            return
        try:
            self._vcam.send(frame)
        except Exception as exc:
            self.close()
            self._failed(exc, now, "仮想カメラへの映像の送信に失敗しました")

    def close(self) -> None:
        vcam, self._vcam, self._device = self._vcam, None, None
        if vcam is not None:
            _close_quietly(vcam, "virtual camera", self._w.errors)

    def _open(self, now: float) -> None:
        request = self._request
        try:
            vcam = self._w.vcam_factory(width=request.width, height=request.height, fps=request.fps)
        except Exception as exc:
            self._failed(exc, now, "仮想カメラを開始できませんでした")
            return
        self._vcam, self._device, self.error = vcam, str(vcam.device), None

    def _failed(self, exc: Exception, now: float, fallback: str) -> None:
        self._w.errors.report("virtual camera", exc)
        self.error = _describe(exc, VirtualCameraError, fallback)
        self._next_open_at = now + self._w.retry_interval_s


class _OutputWorker:
    """Runs the pipeline on the newest frame and feeds the virtual camera and callbacks."""

    def __init__(self, wiring: _Wiring, stop: threading.Event, slot: _FrameSlot) -> None:
        self._w = wiring
        self._stop = stop
        self._slot = slot
        self._vcam = _VcamLink(wiring)
        self._stats = _Stats()
        self._interval_s = 1.0 / max(1, wiring.vcam_request.get().fps)
        self._seq = 0
        # Counting from start-up makes the placeholder appear only after the timeout.
        self._last_frame_at = time.monotonic()
        self._next_placeholder_at = 0.0
        self._face_found = False
        self._processing_error: str | None = None
        self._emitted: RunnerStatus | None = None
        self._emitted_at = 0.0

    def run(self) -> None:
        try:
            while not self._stop.is_set():
                try:
                    self._step()
                except Exception as exc:  # last line of defence: keep the thread alive
                    self._w.errors.report("output", exc)
                    self._stop.wait(_POLL_S)
        finally:
            self._vcam.close()
            self._publish(RunnerStatus(), time.monotonic(), force=True)

    def _step(self) -> None:
        self._vcam.sync(time.monotonic())
        captured = self._slot.take_newer(self._seq, self._interval_s)
        now = time.monotonic()
        if captured is not None:
            self._seq, self._last_frame_at = captured.seq, now
            self._deliver(self._process(captured, now), now)
        elif self._camera_lost(now):
            self._stats.reset()
            self._face_found = False
            if now >= self._next_placeholder_at:
                self._next_placeholder_at = now + self._interval_s
                width, height = self._vcam.size
                self._deliver(_still_result(placeholder_frame(width, height)), now)
        self._publish(self._current_status(now), now)

    def _process(self, captured: _Captured, now: float) -> ProcessResult:
        start = time.perf_counter()
        try:
            result = self._w.pipeline.process(captured.frame, captured.timestamp_ms)
        except Exception as exc:
            self._w.errors.report("pipeline", exc)
            self._processing_error = PROCESSING_FAILED_MESSAGE
            result = _still_result(captured.frame)
        else:
            self._processing_error = None
        self._face_found = result.face_found
        self._stats.record(now, (time.perf_counter() - start) * 1000.0)
        return result

    def _deliver(self, result: ProcessResult, now: float) -> None:
        self._vcam.send(result.output, now)
        _call_safely(self._w.on_frame, result, "on_frame", self._w.errors)

    def _camera_lost(self, now: float) -> bool:
        return now - self._last_frame_at > self._w.camera_timeout_s

    def _current_status(self, now: float) -> RunnerStatus:
        lost = self._camera_lost(now)
        camera_ok = self._seq > 0 and not lost
        camera_error = None
        if not camera_ok:
            camera_error = self._w.camera_error.get() or (CAMERA_LOST_MESSAGE if lost else None)
        return RunnerStatus(
            fps=self._stats.fps,
            camera_ok=camera_ok,
            camera_error=camera_error,
            virtual_camera=self._vcam.device,
            virtual_camera_error=self._vcam.error,
            face_found=self._face_found,
            processing_ms=self._stats.processing_ms,
            processing_error=self._processing_error,
        )

    def _publish(self, status: RunnerStatus, now: float, force: bool = False) -> None:
        """Store the status; notify on any state change, and at most every STATUS_INTERVAL_S otherwise."""
        self._w.status.set(status)
        unchanged = self._emitted is not None and _same_state(status, self._emitted)
        if unchanged and not force and now - self._emitted_at < STATUS_INTERVAL_S:
            return
        self._emitted, self._emitted_at = status, now
        _call_safely(self._w.on_status, status, "on_status", self._w.errors)


# --------------------------------------------------------------------------
# Public runner
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _Session:
    """The threads of one start() ... stop() run; fresh per run, so a restart never sees stale state."""

    stop_event: threading.Event
    slot: _FrameSlot
    threads: tuple[threading.Thread, ...]

    @classmethod
    def launch(cls, wiring: _Wiring) -> _Session:
        stop_event, slot = threading.Event(), _FrameSlot()
        threads = (
            threading.Thread(
                target=_CaptureWorker(wiring, stop_event, slot).run, name="makeup-cam-capture", daemon=True
            ),
            threading.Thread(target=_OutputWorker(wiring, stop_event, slot).run, name="makeup-cam-output", daemon=True),
        )
        for thread in threads:
            thread.start()
        return cls(stop_event, slot, threads)

    def shutdown(self, timeout: float) -> None:
        self.stop_event.set()
        self.slot.close()
        deadline = time.monotonic() + timeout
        # A callback may call stop() on a worker thread, which cannot join itself.
        others = [thread for thread in self.threads if thread is not threading.current_thread()]
        for thread in others:
            thread.join(max(0.0, deadline - time.monotonic()))
        stuck = [thread.name for thread in others if thread.is_alive()]
        if stuck:
            # Typically blocked inside a driver call; the thread closes its device when that returns.
            logger.warning("still running %.1f s after stop: %s", timeout, ", ".join(stuck))


class CameraRunner:
    """Runs camera -> pipeline -> virtual camera in the background (see the module docstring).

    ``on_frame(ProcessResult)`` and ``on_status(RunnerStatus)`` are called on
    the output thread: GUIs must hand them to their own thread without
    blocking (e.g. a queued Qt signal). ``on_status`` fires on every state
    change and at least every ``STATUS_INTERVAL_S`` while running. The runner
    does not close the pipeline; its owner does.
    """

    def __init__(
        self,
        pipeline: MakeupPipeline,
        *,
        camera_index: int = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        virtual_camera: bool = True,
        on_frame: Callable[[ProcessResult], None] | None = None,
        on_status: Callable[[RunnerStatus], None] | None = None,
        camera_factory: CameraFactory = CameraSource,
        vcam_factory: VcamFactory = VirtualCameraOutput,
        retry_interval_s: float = RETRY_INTERVAL_S,
        camera_timeout_s: float = CAMERA_TIMEOUT_S,
    ) -> None:
        self._pipeline = pipeline
        self._wiring = _Wiring(
            pipeline=pipeline,
            camera_factory=camera_factory,
            vcam_factory=vcam_factory,
            on_frame=on_frame,
            on_status=on_status,
            camera_request=_Locked(_CameraRequest(camera_index, width, height, fps, generation=0)),
            vcam_request=_Locked(_VcamRequest(virtual_camera, width, height, fps)),
            camera_error=_Locked(None),
            status=_Locked(RunnerStatus()),
            errors=_ErrorLog(),
            retry_interval_s=retry_interval_s,
            camera_timeout_s=camera_timeout_s,
        )
        self._lifecycle = threading.Lock()
        self._session: _Session | None = None

    @property
    def status(self) -> RunnerStatus:
        return self._wiring.status.get()

    @property
    def camera_index(self) -> int:
        """The selected camera (e.g. for ``list_cameras(in_use={runner.camera_index})``)."""
        return self._wiring.camera_request.get().index

    @property
    def is_running(self) -> bool:
        with self._lifecycle:
            return self._session is not None

    def start(self) -> None:
        """Start the worker threads; does nothing if already running."""
        with self._lifecycle:
            if self._session is None:
                self._session = _Session.launch(self._wiring)

    def stop(self, timeout: float = 2.0) -> None:
        """Stop the threads and close the camera and virtual camera; safe to call repeatedly."""
        with self._lifecycle:
            session, self._session = self._session, None
            if session is not None:
                session.shutdown(timeout)

    def set_settings(self, settings: MakeupSettings) -> None:
        self._pipeline.set_settings(settings)

    def set_bypass(self, bypass: bool) -> None:
        self._pipeline.bypass = bypass

    def set_camera(self, index: int, width: int | None = None, height: int | None = None) -> None:
        """Switch to camera ``index`` (reconnects even if unchanged).

        A new size also restarts the virtual camera at that size.
        """
        request = self._wiring.camera_request.update(
            lambda old: _CameraRequest(
                index,
                old.width if width is None else width,
                old.height if height is None else height,
                old.fps,
                old.generation + 1,
            )
        )
        self._wiring.vcam_request.update(
            lambda old: dataclasses.replace(old, width=request.width, height=request.height)
        )

    def set_virtual_camera(self, enabled: bool) -> None:
        """Start or stop the virtual camera output (applied on the output thread)."""
        self._wiring.vcam_request.update(lambda old: dataclasses.replace(old, enabled=bool(enabled)))


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


@lru_cache(maxsize=4)
def placeholder_frame(width: int, height: int) -> np.ndarray:
    """Read-only neutral frame sent while the camera is unavailable."""
    frame = np.full((height, width, 3), _PLACEHOLDER_BG, np.uint8)
    unit = min(width, height) / 100.0
    _draw_camera_off_icon(frame, (width // 2, round(height * 0.42)), unit)
    _draw_centered_text(frame, PLACEHOLDER_TEXT, round(height * 0.42 + unit * 26), unit)
    frame.setflags(write=False)
    return frame


def _draw_camera_off_icon(frame: np.ndarray, center: tuple[int, int], unit: float) -> None:
    """A crossed-out camera outline centered on ``center``; ``unit`` = 1% of the short side."""
    cx, cy = center
    thickness = max(1, round(unit * 0.9))

    def at(dx: float, dy: float) -> tuple[int, int]:
        return round(cx + dx * unit), round(cy + dy * unit)

    cv2.rectangle(frame, at(-15, -8), at(15, 12), _PLACEHOLDER_FG, thickness, cv2.LINE_AA)  # body
    cv2.rectangle(frame, at(-6, -12), at(6, -8), _PLACEHOLDER_FG, thickness, cv2.LINE_AA)  # viewfinder
    cv2.circle(frame, at(0, 2), round(unit * 6), _PLACEHOLDER_FG, thickness, cv2.LINE_AA)  # lens
    cv2.line(frame, at(-18, -14), at(18, 17), _PLACEHOLDER_FG, thickness, cv2.LINE_AA)  # slash


def _draw_centered_text(frame: np.ndarray, text: str, baseline_y: int, unit: float) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = unit * 0.14
    thickness = max(1, round(unit * 0.3))
    (text_width, _), _ = cv2.getTextSize(text, font, scale, thickness)
    origin = ((frame.shape[1] - text_width) // 2, baseline_y)
    cv2.putText(frame, text, origin, font, scale, _PLACEHOLDER_FG, thickness, cv2.LINE_AA)


def _still_result(frame: np.ndarray) -> ProcessResult:
    """A result that shows ``frame`` as is (placeholder, or raw frame when processing failed)."""
    return ProcessResult(output=frame, preview=frame, original_preview=frame, face_found=False, timings_ms={})


def _same_state(a: RunnerStatus, b: RunnerStatus) -> bool:
    """Equal apart from the continuously changing measurements."""
    return dataclasses.replace(a, fps=0.0, processing_ms=0.0) == dataclasses.replace(b, fps=0.0, processing_ms=0.0)


def _describe(exc: Exception, friendly: type[Exception], fallback: str) -> str:
    """The Japanese message of our own errors, or ``fallback`` plus the raw error."""
    return str(exc) if isinstance(exc, friendly) else f"{fallback}（{exc}）"


def _call_safely(callback: Callable[[T], None] | None, value: T, name: str, errors: _ErrorLog) -> None:
    if callback is None:
        return
    try:
        callback(value)
    except Exception as exc:
        errors.report(name, exc)


def _close_quietly(device: FrameSource | FrameSink, name: str, errors: _ErrorLog) -> None:
    try:
        device.close()
    except Exception as exc:
        errors.report(name, exc)
