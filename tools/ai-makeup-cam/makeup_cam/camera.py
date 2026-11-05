"""Webcam input and virtual camera output.

Platform notes:

- Windows: DirectShow first (opens fast and honours the MJPG hint most USB
  webcams need for 1280x720 at 30 fps), Media Foundation as the fallback.
  Device names are not available without extra dependencies.
- macOS: AVFoundation. The first open triggers the camera permission prompt
  for the app that started Python (Terminal, iTerm, VS Code, ...).
- Linux: V4L2. Device names are read from sysfs, which also lets us skip
  virtual cameras (selecting our own output would be a feedback loop).

The virtual camera goes through pyvirtualcam: OBS Virtual Camera on Windows
and macOS, v4l2loopback on Linux. Failures become ``VirtualCameraError``
whose message tells the user, in Japanese, what to install or allow.
"""

from __future__ import annotations

import logging
import os
import re
import sys
import threading
import time
from collections.abc import Callable, Collection
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Media Foundation's GPU colour transforms make opening some webcams take
# 10+ seconds. OpenCV reads this when the first MSMF capture is created.
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")

# Card label suggested for our v4l2loopback device (see the Linux hint).
VIRTUAL_CAMERA_LABEL = "AI Makeup Cam"
# Input devices whose names match this are virtual cameras and never listed.
_VIRTUAL_CAMERA_NAME = re.compile(
    r"obs[ -]?virtual|virtual ?cam|v4l2loopback|dummy video device|unity video capture|"
    + re.escape(VIRTUAL_CAMERA_LABEL),
    re.IGNORECASE,
)

LIST_TIME_BUDGET_S = 6.0
REOPEN_AFTER_FAILURES = 15
REOPEN_INTERVAL_S = 2.0
_MAX_CONSECUTIVE_MISSES = 2
_WARMUP_READS = 5
_V4L2_SYSFS = Path("/sys/class/video4linux")
_V4L2_NODE = re.compile(r"video(\d+)")
# Backends where an MJPG request is honoured and needed for 720p30 over USB 2.
_MJPG_BACKENDS = frozenset({cv2.CAP_DSHOW, cv2.CAP_V4L2})
_BACKEND_NAMES = {
    cv2.CAP_DSHOW: "DirectShow",
    cv2.CAP_MSMF: "Media Foundation",
    cv2.CAP_AVFOUNDATION: "AVFoundation",
    cv2.CAP_V4L2: "V4L2",
    cv2.CAP_ANY: "auto",
}

_CAMERA_HINTS = {
    "windows": (
        "ほかのアプリ（Zoom・Teams・ブラウザ・OBS など）がカメラを使っていないか確認してください。"
        "また「設定 → プライバシーとセキュリティ → カメラ」で"
        "「デスクトップ アプリがカメラにアクセスできるようにする」をオンにしてください。"
    ),
    "macos": (
        "「システム設定 → プライバシーとセキュリティ → カメラ」で、"
        "このアプリを起動したターミナル（または Python・VS Code）のカメラ利用を許可してください。"
        "初回は許可を求めるダイアログが表示されます。許可した後も映らない場合は、ターミナルを再起動してください。"
    ),
    "linux": (
        "カメラが接続されているか、/dev/video* を開く権限（video グループ）があるか確認してください。"
        "ほかのアプリが使用中の場合も開けません。"
    ),
    "other": "カメラが接続されているか、ほかのアプリが使用していないか確認してください。",
}

_VIRTUAL_CAMERA_HINTS = {
    "windows": (
        "OBS Studio（バージョン 26 以降）をインストールしてください。"
        "インストールすると「OBS Virtual Camera」が使えるようになります。"
        "インストール後は Zoom・Teams・ブラウザを再起動してください。"
        "OBS Studio で「仮想カメラ開始」を押している場合は停止してください（同時には使えません）。"
    ),
    "macos": (
        "OBS Studio（バージョン 30 以降）を「アプリケーション」フォルダにインストールし、"
        "OBS を一度起動して「仮想カメラ開始」を1回クリックしてください"
        "（仮想カメラのシステム拡張がインストールされます）。"
        "「システム設定 → 一般 → ログイン項目と機能拡張」（または「プライバシーとセキュリティ」）で"
        "拡張機能を許可したら、"
        "OBS の仮想カメラを停止して OBS を終了し、このアプリと Zoom などを再起動してください。"
    ),
    "linux": (
        "v4l2loopback を読み込んでください: "
        f'sudo modprobe v4l2loopback devices=1 exclusive_caps=1 card_label="{VIRTUAL_CAMERA_LABEL}"'
        "（未インストールの場合は sudo apt install v4l2loopback-dkms）。"
    ),
    "other": "このOSの仮想カメラには対応していない可能性があります。",
}

_PYVIRTUALCAM_MISSING = (
    "仮想カメラ用のライブラリ pyvirtualcam を読み込めませんでした。"
    "pip install pyvirtualcam を実行してから、アプリを再起動してください。"
)


class CameraError(RuntimeError):
    """A camera could not be opened. The message is Japanese and actionable."""


class VirtualCameraError(RuntimeError):
    """The virtual camera could not be started or fed. Japanese, actionable message."""


class VideoCaptureLike(Protocol):
    """The subset of ``cv2.VideoCapture`` used here (injectable for tests)."""

    def isOpened(self) -> bool: ...
    def read(self) -> tuple[bool, np.ndarray | None]: ...
    def set(self, prop_id: int, value: float) -> bool: ...
    def get(self, prop_id: int) -> float: ...
    def release(self) -> None: ...


CaptureFactory = Callable[[int, int], VideoCaptureLike]


@dataclass(frozen=True)
class CameraInfo:
    index: int
    name: str
    # Default resolution reported by the driver; 0 when unknown.
    width: int
    height: int


# --------------------------------------------------------------------------
# Platform helpers
# --------------------------------------------------------------------------


def host_os() -> str:
    """The host OS family: "windows", "macos", "linux" or "other"."""
    if sys.platform.startswith("win"):
        return "windows"
    if sys.platform == "darwin":
        return "macos"
    if sys.platform.startswith("linux"):
        return "linux"
    return "other"


def capture_backends() -> tuple[int, ...]:
    """OpenCV capture APIs to try for this OS, best first."""
    backends = {
        "windows": (cv2.CAP_DSHOW, cv2.CAP_MSMF),
        "macos": (cv2.CAP_AVFOUNDATION,),
        "linux": (cv2.CAP_V4L2,),
    }
    return backends.get(host_os(), (cv2.CAP_ANY,))


def default_camera_name(index: int) -> str:
    return f"カメラ {index}"


def camera_open_error_message(index: int) -> str:
    return f"カメラ {index} を開けませんでした。{_CAMERA_HINTS[host_os()]}"


def virtual_camera_error_message(detail: str) -> str:
    return f"仮想カメラを開始できませんでした。{_VIRTUAL_CAMERA_HINTS[host_os()]}\n（詳細: {detail}）"


class _QuietOpenCvLogs:
    """Reentrant, thread-safe context that silences OpenCV's console log.

    Probing devices makes OpenCV print a warning for every absent index or
    unsupported backend; here those failures are expected.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._depth = 0
        self._saved_level: int | None = None

    def __enter__(self) -> None:
        with self._lock:
            api = _opencv_logging()
            if self._depth == 0 and api is not None:
                self._saved_level = api.getLogLevel()
                api.setLogLevel(api.LOG_LEVEL_SILENT)
            self._depth += 1

    def __exit__(self, *exc_info: object) -> None:
        with self._lock:
            self._depth -= 1
            api = _opencv_logging()
            if self._depth == 0 and api is not None and self._saved_level is not None:
                api.setLogLevel(self._saved_level)
                self._saved_level = None


def _opencv_logging() -> Any:
    return getattr(getattr(cv2, "utils", None), "logging", None)


quiet_opencv_logs = _QuietOpenCvLogs()


# --------------------------------------------------------------------------
# Listing cameras
# --------------------------------------------------------------------------


def list_cameras(
    max_index: int = 8,
    *,
    in_use: Collection[int] = (),
    time_budget_s: float = LIST_TIME_BUDGET_S,
    capture_factory: CaptureFactory = cv2.VideoCapture,
) -> list[CameraInfo]:
    """Find connected cameras quickly (call it off the GUI thread).

    Each candidate index is opened once with the OS's preferred backend; no
    frames are read. When the OS gives device names (Linux), exactly those
    devices are probed and virtual cameras are skipped. Otherwise indices are
    probed in order until two consecutive misses after index 0. No new probe
    starts once ``time_budget_s`` has elapsed. ``in_use`` are indices this app
    already has open: they are listed without probing (a busy device may
    refuse a second open), with width/height 0.
    """
    names = _device_names()
    api = capture_backends()[0]
    blind = not names
    candidates = range(max_index) if blind else sorted(i for i in names if i < max_index)
    deadline = time.monotonic() + time_budget_s
    found: list[CameraInfo] = []
    misses = 0
    with quiet_opencv_logs:
        for index in candidates:
            name = names.get(index) or default_camera_name(index)
            if _VIRTUAL_CAMERA_NAME.search(name):
                continue
            busy = index in in_use
            if not busy and time.monotonic() > deadline:
                break
            info = CameraInfo(index, name, 0, 0) if busy else _probe(capture_factory, index, api, name)
            if info is not None:
                found.append(info)
                misses = 0
            elif blind and index > 0:
                misses += 1
                if misses >= _MAX_CONSECUTIVE_MISSES:
                    break
    return found


def _device_names() -> dict[int, str]:
    """{index: name} when the OS exposes names cheaply, else {}."""
    return _v4l2_device_names(_V4L2_SYSFS) if host_os() == "linux" else {}


def _v4l2_device_names(root: Path) -> dict[int, str]:
    names: dict[int, str] = {}
    for node in root.glob("video*"):
        match = _V4L2_NODE.fullmatch(node.name)
        if match is None:
            continue
        index = int(match.group(1))
        try:
            name = (node / "name").read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            name = ""
        names[index] = name or default_camera_name(index)
    return names


def _probe(factory: CaptureFactory, index: int, api: int, name: str) -> CameraInfo | None:
    try:
        cap = factory(index, api)
    except cv2.error:
        return None
    try:
        if not cap.isOpened():
            return None
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return CameraInfo(index, name, width, height)
    except cv2.error:
        return None
    finally:
        cap.release()


# --------------------------------------------------------------------------
# Camera input
# --------------------------------------------------------------------------


class CameraSource:
    """One webcam, opened with the best working backend for the OS.

    ``open`` raises ``CameraError``; ``read`` never raises and returns None
    when no frame is available. After ``REOPEN_AFTER_FAILURES`` consecutive
    failed reads the device is released and reopened (at most once per
    ``reopen_interval_s``), so a camera that was unplugged and plugged back
    in, or whose driver hiccuped, recovers by itself.
    """

    def __init__(
        self,
        index: int = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        *,
        reopen_interval_s: float = REOPEN_INTERVAL_S,
        capture_factory: CaptureFactory = cv2.VideoCapture,
    ) -> None:
        self.index = index
        self.width = width
        self.height = height
        self.fps = fps
        # What the camera actually delivers (valid once opened).
        self.actual_width = 0
        self.actual_height = 0
        self.actual_fps = 0.0
        self._reopen_interval_s = reopen_interval_s
        self._capture_factory = capture_factory
        self._cap: VideoCaptureLike | None = None
        self._api: int | None = None
        self._active = False  # between a successful open() and close()
        self._failures = 0
        self._next_reopen_at = 0.0

    @property
    def is_open(self) -> bool:
        return self._cap is not None

    @property
    def backend(self) -> str | None:
        """Name of the OpenCV backend in use, e.g. "DirectShow"."""
        return None if self._api is None else _BACKEND_NAMES.get(self._api, str(self._api))

    def open(self) -> None:
        """Open the camera; raises ``CameraError`` if no backend delivers frames."""
        if self._cap is not None:
            return
        if not self._open_device():
            raise CameraError(camera_open_error_message(self.index))
        self._active = True

    def read(self) -> np.ndarray | None:
        """The next uint8 BGR frame, or None (reopening the device when needed)."""
        cap = self._device()
        if cap is None:
            return None
        try:
            ok, frame = cap.read()
        except cv2.error:
            ok, frame = False, None
        if ok and frame is not None and frame.size:
            self._failures = 0
            return _to_bgr(frame)
        self._on_read_failure()
        return None

    def close(self) -> None:
        self._active = False
        self._release()

    # ---- internals --------------------------------------------------------

    def _open_device(self) -> bool:
        with quiet_opencv_logs:
            for api in capture_backends():
                cap = self._try_backend(api)
                if cap is not None:
                    self._cap, self._api, self._failures = cap, api, 0
                    logger.info(
                        "camera %d opened via %s: %dx%d @ %.0f fps",
                        self.index,
                        self.backend,
                        self.actual_width,
                        self.actual_height,
                        self.actual_fps,
                    )
                    return True
        return False

    def _try_backend(self, api: int) -> VideoCaptureLike | None:
        try:
            cap = self._capture_factory(self.index, api)
        except cv2.error:
            return None
        try:
            frame = self._start(cap, api)
        except cv2.error as exc:
            logger.debug("camera %d: backend %s failed: %s", self.index, _BACKEND_NAMES.get(api, api), exc)
            frame = None
        if frame is None:
            cap.release()
            return None
        self.actual_height, self.actual_width = frame.shape[:2]
        self.actual_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        return cap

    def _start(self, cap: VideoCaptureLike, api: int) -> np.ndarray | None:
        """Configure an opened capture and return its first frame (None if it gives none)."""
        if not cap.isOpened():
            return None
        if api in _MJPG_BACKENDS:
            # Must precede the size: most USB 2 webcams reach 1280x720@30 only as MJPG.
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        for _ in range(_WARMUP_READS):  # some cameras return empty frames right after opening
            ok, frame = cap.read()
            if ok and frame is not None and frame.size:
                return frame
        return None

    def _device(self) -> VideoCaptureLike | None:
        """The open capture; after a failure it is reopened at most once per interval."""
        if self._active and self._cap is None and time.monotonic() >= self._next_reopen_at:
            self._next_reopen_at = time.monotonic() + self._reopen_interval_s
            self._open_device()
        return self._cap if self._active else None

    def _on_read_failure(self) -> None:
        self._failures += 1
        if self._failures >= REOPEN_AFTER_FAILURES:
            logger.warning("camera %d stopped delivering frames; reopening it", self.index)
            self._release()

    def _release(self) -> None:
        cap, self._cap = self._cap, None
        if cap is not None:
            cap.release()


def _to_bgr(frame: np.ndarray) -> np.ndarray:
    """Some backends deliver gray or BGRA frames; the pipeline wants 3-channel BGR."""
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame


# --------------------------------------------------------------------------
# Virtual camera output
# --------------------------------------------------------------------------


class VirtualCameraOutput:
    """Sends uint8 BGR frames to the OS virtual camera.

    Frames of another size are scaled to fit (aspect ratio kept, black bars),
    so the device format negotiated by Zoom / Teams / Meet never changes.
    Pacing is the caller's job (``sleep_until_next_frame`` would block it).
    """

    def __init__(self, width: int, height: int, fps: float) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self._camera: Any = _start_virtual_camera(width, height, fps)
        self.device: str = str(self._camera.device)
        logger.info("virtual camera started: %s (%dx%d @ %s fps)", self.device, width, height, fps)

    def send(self, frame_bgr: np.ndarray) -> None:
        if self._camera is None:
            raise VirtualCameraError("仮想カメラは停止しています。")
        try:
            self._camera.send(fit_frame(frame_bgr, self.width, self.height))
        except Exception as exc:
            raise VirtualCameraError(f"仮想カメラへの映像の送信に失敗しました（{exc}）") from exc

    def close(self) -> None:
        camera, self._camera = self._camera, None
        if camera is not None:
            camera.close()


def _start_virtual_camera(width: int, height: int, fps: float) -> Any:
    try:
        import pyvirtualcam  # lazy: a broken native backend must not break camera preview
    except (ImportError, OSError) as exc:
        raise VirtualCameraError(_PYVIRTUALCAM_MISSING) from exc
    try:
        return pyvirtualcam.Camera(width, height, fps, fmt=pyvirtualcam.PixelFormat.BGR, print_fps=False)
    except Exception as exc:
        raise VirtualCameraError(virtual_camera_error_message(str(exc))) from exc


def fit_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """``frame`` scaled to fit ``width`` x ``height`` keeping its aspect ratio, centered on black."""
    src_h, src_w = frame.shape[:2]
    if (src_w, src_h) == (width, height):
        return frame
    scale = min(width / src_w, height / src_h)
    new_w = min(width, max(1, round(src_w * scale)))
    new_h = min(height, max(1, round(src_h * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(frame, (new_w, new_h), interpolation=interpolation)
    if (new_w, new_h) == (width, height):
        return resized
    canvas = np.zeros((height, width, 3), np.uint8)
    x0, y0 = (width - new_w) // 2, (height - new_h) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas
