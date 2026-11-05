"""Webcam input and virtual camera output.

Platform notes:

- Windows: DirectShow first (opens fast and honours the MJPG hint most USB
  webcams need for 1280x720 at 30 fps), Media Foundation as the fallback.
  Device names come from raw COM calls through ctypes. A camera index always
  means a DirectShow index: Media Foundation numbers devices differently, so
  the fallback finds the device by name (see ``CameraSource``). Plugging or
  unplugging a camera renumbers the others, so a device name, when known, beats
  an index everywhere.
- macOS: AVFoundation. The first open triggers the camera permission prompt
  for the app that started Python (Terminal, iTerm, VS Code, ...).
- Linux: V4L2. Device names are read from sysfs, which also lets us skip
  virtual cameras (selecting our own output would be a feedback loop).

The virtual camera goes through pyvirtualcam: OBS Virtual Camera on Windows
and macOS, v4l2loopback on Linux. Failures become ``VirtualCameraError``
whose message tells the user, in Japanese, what to install or allow.

A device whose name says it is a virtual camera is never opened as the input,
because it would feed the app's own output back into itself. On Windows, while
the names cannot be read, nothing is opened at all. macOS has no device names
here, so that guard works on Windows and Linux only.

OBS Virtual Camera stretches what it receives to the format the meeting app
negotiated, and it shows its own logo while the output restarts. So the app runs
the output at one 16:9 size (``virtual_camera_size``) for the whole session and
fits every frame into it, whatever the capture size is.

``VideoFileSource`` plays a video file in place of the webcam, for testing the
whole app on a PC without a camera.
"""

from __future__ import annotations

import ctypes
import logging
import math
import os
import re
import sys
import threading
import time
import uuid
from collections.abc import Callable, Collection
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Media Foundation's GPU colour transforms make opening some webcams take
# 10+ seconds. OpenCV reads this when the first MSMF capture is created.
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")


def _silence_ffmpeg_plugin() -> None:
    """Keep the FFmpeg plugin from filling the console with English log lines.

    The plugin has its own log state, which ``cv2.utils.logging`` does not reach:
    on Windows it is a separate DLL on another C runtime, so it reads these
    settings from msvcrt's copy of the environment, which ``os.environ`` does not
    update. They are read once, by the first capture. A value the user set wins.
    """
    settings = {"OPENCV_FFMPEG_LOGLEVEL": "-8"}  # AV_LOG_QUIET
    if sys.platform == "win32":
        settings["OPENCV_LOG_LEVEL"] = "SILENT"  # only the plugin reads this copy: cv2 itself has its own C runtime
    for key, value in settings.items():
        if key in os.environ:
            continue
        if sys.platform != "win32":
            os.environ[key] = value
            continue
        try:
            ctypes.cdll.msvcrt._putenv(f"{key}={value}".encode())
        except (OSError, AttributeError) as exc:  # only log lines are at stake
            logger.debug("could not set %s: %s", key, exc)


_silence_ffmpeg_plugin()

# Card label suggested for our v4l2loopback device (see the Linux hint).
VIRTUAL_CAMERA_LABEL = "AI Makeup Cam"
# Input devices whose names match this are virtual cameras: never listed, never opened.
_VIRTUAL_CAMERA_NAME = re.compile(
    r"obs[ -]?virtual|\bobs[ -]?cam|virtual ?(web)?cam|\bvcam\b|v4l2loopback|dummy video device|unity video capture|"
    + re.escape(VIRTUAL_CAMERA_LABEL),
    re.IGNORECASE,
)

LIST_TIME_BUDGET_S = 6.0
REOPEN_AFTER_FAILURES = 15
REOPEN_INTERVAL_S = 2.0
_MAX_CONSECUTIVE_MISSES = 2
_WARMUP_READS = 5
# A video file with a missing or absurd frame rate plays at the requested fps instead. Below the
# lower limit each frame is delivered several times, which stays clear of the runner's one-second
# "camera lost" timeout and keeps the file in real time.
_VIDEO_FILE_FPS_RANGE = (2.0, 240.0)
_WAIT_SLICE_S = 0.05  # how soon a read() waiting for its frame notices close()
# The virtual camera runs at 16:9 in steps of 16x9 pixels: at least 1280x720, at most 1920x1080.
_OUTPUT_UNITS_RANGE = (80, 120)
_V4L2_SYSFS = Path("/sys/class/video4linux")
_V4L2_NODE = re.compile(r"video(\d+)")
# Backends where an MJPG request is honoured and needed for 720p30 over USB 2.
_MJPG_BACKENDS = frozenset({cv2.CAP_DSHOW, cv2.CAP_V4L2})
_BACKEND_NAMES = {
    cv2.CAP_DSHOW: "DirectShow",
    cv2.CAP_MSMF: "Media Foundation",
    cv2.CAP_AVFOUNDATION: "AVFoundation",
    cv2.CAP_V4L2: "V4L2",
    cv2.CAP_FFMPEG: "FFmpeg",
    cv2.CAP_ANY: "auto",
}

CAMERA_LIST_FAILED_MESSAGE = (
    "カメラの一覧を取得できないため、カメラを開かずに待っています"
    "（仮想カメラを開いて映像がループするのを防ぐためです）。"
    "しばらくしても直らない場合は、PC を再起動してください。"
)

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

_VIRTUAL_CAMERA_BUSY_HINT = (
    "OBS Virtual Camera はインストールされていますが、いまは使用中です。"
    "OBS Studio で「仮想カメラ開始」を押している場合は停止してください。"
    "直らないときは、Zoom・Teams・ブラウザのカメラを一度オフにしてからオンに戻してください。"
    "空きしだい自動で接続します。"
)

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


def camera_missing_message(name: str) -> str:
    return f"「{name}」が見つかりません。接続を確認してください。つなぎ直すと自動で再開します。"


def virtual_camera_as_input_message(name: str) -> str:
    return (
        f"「{name}」は仮想カメラなので、カメラの入力には使えません"
        "（このアプリの出力が入力に戻って、映像がループしてしまいます）。"
        "実際の Web カメラを接続して選んでください。"
    )


def is_virtual_camera_name(name: str) -> bool:
    return _VIRTUAL_CAMERA_NAME.search(name) is not None


def _real_name(name: str | None, index: int) -> str | None:
    """``name``, unless it is only the placeholder for ``index``: that says nothing about the device."""
    return None if name == default_camera_name(index) else name


def virtual_camera_error_message(detail: str) -> str:
    # pyvirtualcam says this when OBS is installed and its queue is taken.
    busy = host_os() == "windows" and "could not be started" in detail
    hint = _VIRTUAL_CAMERA_BUSY_HINT if busy else _VIRTUAL_CAMERA_HINTS[host_os()]
    return f"仮想カメラを開始できませんでした。{hint}\n（詳細: {detail}）"


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
    in_use: Collection[int] | Collection[str] = (),
    time_budget_s: float = LIST_TIME_BUDGET_S,
    capture_factory: CaptureFactory = cv2.VideoCapture,
) -> list[CameraInfo]:
    """Find connected cameras quickly (call it off the GUI thread).

    Each candidate index is opened once with the OS's preferred backend; no
    frames are read. When the OS gives device names (Windows, Linux), exactly those
    devices are probed and virtual cameras are skipped. Otherwise indices are
    probed in order until two consecutive misses after index 0. No new probe
    starts once ``time_budget_s`` has elapsed. ``in_use`` says what this app
    already has open, as device names or else indices: those are listed without
    probing, with width/height 0. A busy device may refuse a second open, and on
    DirectShow opening and closing a live device stops it. When Windows cannot
    say which devices there are, nothing is opened and the list is empty.
    """
    names = _device_names()
    if names is None:
        logger.info("the camera devices could not be listed")
        return []
    api = capture_backends()[0]
    blind = not names
    candidates = range(max_index) if blind else sorted(i for i in names if i < max_index)
    # One device each: on Linux a webcam's metadata node has its name too, and cannot be opened.
    names_in_use = {item for item in in_use if isinstance(item, str)}
    deadline = time.monotonic() + time_budget_s
    found: list[CameraInfo] = []
    misses = 0
    with quiet_opencv_logs:
        for index in candidates:
            name = names.get(index) or default_camera_name(index)
            if is_virtual_camera_name(name):
                continue
            busy = name in names_in_use or index in in_use
            names_in_use.discard(name)
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


def resolve_camera(index: int | None, name: str | None = None) -> tuple[int, str | None]:
    """The camera to start with, and its name if known, from what was remembered.

    Plugging or unplugging a camera renumbers the others, so a remembered name
    beats a remembered index. A virtual camera is never chosen while a real one
    is there. Without any device names the index is all there is to go on. With
    no real camera the index stays as is: opening it then fails with a clear
    message, and the app keeps running. While Windows cannot list its devices
    the remembered choice stands, name included, for ``CameraSource`` to find.
    """
    fallback = 0 if index is None else index
    name = _real_name(name, fallback)  # a placeholder was saved when no real name was available
    names = _device_names()
    if names is None:
        return fallback, name
    if not names:
        return fallback, None
    real = {i: names[i] for i in sorted(names) if not is_virtual_camera_name(names[i])}
    if name:
        if real.get(fallback) == name:
            return fallback, name
        for candidate, candidate_name in real.items():
            if candidate_name == name:
                return candidate, candidate_name
    elif index is not None and index in real:
        return index, real[index]
    if real:
        first = next(iter(real))
        return first, real[first]
    return fallback, None


def _device_names() -> dict[int, str] | None:
    """{index: name} when the OS exposes names cheaply, else {}; None when Windows could not say.

    On Windows these are DirectShow's devices and indices (``cv2.CAP_DSHOW``);
    {} there means that the listing worked and found none.
    """
    os_name = host_os()
    if os_name == "linux":
        return _v4l2_device_names(_V4L2_SYSFS)
    return _dshow_device_names() if os_name == "windows" else {}


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
# Windows device names (COM through ctypes; no dependency beyond the OS)
# --------------------------------------------------------------------------

_COINIT_MULTITHREADED = 0
_CLSCTX_INPROC_SERVER = 1
_S_OK = 0
_RPC_E_CHANGED_MODE = 0x80010106 - (1 << 32)  # HRESULTs are signed here
_VT_BSTR = 8
_MF_VERSION = 0x00020070
_MFSTARTUP_LITE = 1

_CLSID_SYSTEM_DEVICE_ENUM = "62BE5D10-60EB-11D0-BD3B-00A0C911CE86"
_IID_ICREATE_DEV_ENUM = "29840822-5B84-11D0-BD3B-00A0C911CE86"
_IID_IPROPERTY_BAG = "55272A00-42CB-11CE-8135-00AA004BB851"
_CATEGORY_VIDEO_INPUT = "860BB310-5D01-11D0-BD3B-00A0C911CE86"
_MF_SOURCE_TYPE = "C60AC5FE-252A-478F-A0EF-BC8FA5F7CAD3"
_MF_SOURCE_TYPE_VIDCAP = "8AC3587A-4AE7-42D8-99E0-0A6013EEF90F"
_MF_FRIENDLY_NAME = "60D0E559-52F8-4FA2-BBCE-ACDB34A8EC01"

# Vtable positions from the Windows SDK headers.
_RELEASE_SLOT = 2
_CREATE_CLASS_ENUMERATOR_SLOT = 3  # ICreateDevEnum
_ENUM_NEXT_SLOT = 3  # IEnumMoniker
_BIND_TO_STORAGE_SLOT = 9  # IMoniker
_PROPERTY_BAG_READ_SLOT = 3  # IPropertyBag
_GET_ALLOCATED_STRING_SLOT = 13  # IMFAttributes, and IMFActivate which derives from it
_SET_GUID_SLOT = 24  # IMFAttributes

_MAX_DEVICES = 64
# A driver that hangs while being listed must not freeze the caller.
_ENUMERATION_TIMEOUT_S = 3.0
# Normal use needs at most three at once; the limit only stops hung threads piling up.
_enumeration_slots = threading.BoundedSemaphore(8)


class _ListingFailed(OSError):
    """Windows could not list the devices, which is not the same as there being none."""


class _Variant(ctypes.Structure):
    """The start of a VARIANT, padded to its size on both 32 and 64 bit."""

    _fields_ = (
        ("vt", ctypes.c_ushort),
        ("reserved", ctypes.c_ushort * 3),
        ("value", ctypes.c_void_p),
        ("padding", ctypes.c_void_p),
    )


def _guid(text: str) -> ctypes.Array[ctypes.c_char]:
    return ctypes.create_string_buffer(uuid.UUID(text).bytes_le, 16)


class _ComPtr:
    """An owned COM interface pointer, released when its ``with`` block ends."""

    def __init__(self, address: int | None = None) -> None:
        self.pointer = ctypes.c_void_p(address)

    def __enter__(self) -> _ComPtr:
        return self

    def __exit__(self, *exc_info: object) -> None:
        pointer, self.pointer = self.pointer, ctypes.c_void_p()
        if pointer:
            _vtable_function(pointer, _RELEASE_SLOT, ctypes.c_ulong)()

    def method(self, slot: int, *argtypes: Any) -> Callable[..., int]:
        """The interface method at vtable position ``slot``; calling it returns the HRESULT."""
        return _vtable_function(self.pointer, slot, ctypes.c_long, *argtypes)


def _vtable_function(pointer: ctypes.c_void_p, slot: int, restype: Any, *argtypes: Any) -> Callable[..., Any]:
    vtable = ctypes.cast(pointer, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
    prototype = ctypes.WINFUNCTYPE(restype, ctypes.c_void_p, *argtypes)  # type: ignore[attr-defined]
    return partial(prototype(vtable[slot]), pointer)


def _dshow_device_names() -> dict[int, str] | None:
    """{cv2.CAP_DSHOW index: friendly name}; None on any failure; {} when not on Windows."""
    return _enumerate_on_own_thread(_read_dshow_names)


def _msmf_device_names() -> dict[int, str] | None:
    """{cv2.CAP_MSMF index: friendly name}; None on any failure; {} when not on Windows."""
    return _enumerate_on_own_thread(_read_msmf_names)


def _enumerate_on_own_thread(enumerate_devices: Callable[[], dict[int, str]]) -> dict[int, str] | None:
    """Run a device enumeration on a short-lived thread; never raises.

    Its own thread leaves the COM state of the caller (possibly Qt's GUI
    thread) untouched, and a hung driver blocks only that thread. Returns None
    when the devices could not be listed: an error, a hung driver, or too many
    hung drivers already.
    """
    if sys.platform != "win32":
        return {}
    if not _enumeration_slots.acquire(blocking=False):
        return None
    outcome: list[dict[int, str]] = []

    def work() -> None:
        try:
            outcome.append(_with_com(enumerate_devices))
        except Exception as exc:  # e.g. mf.dll is absent on Windows N editions
            logger.debug("listing devices failed: %s", exc)
        finally:
            _enumeration_slots.release()

    thread = threading.Thread(target=work, name="camera-names", daemon=True)
    try:
        thread.start()
    except RuntimeError:
        _enumeration_slots.release()
        return None
    thread.join(_ENUMERATION_TIMEOUT_S)
    if thread.is_alive():
        logger.warning("listing camera devices took over %.0f s; the names stay unknown", _ENUMERATION_TIMEOUT_S)
        return None
    return outcome[0] if outcome else None


def _with_com(work: Callable[[], dict[int, str]]) -> dict[int, str]:
    """Call ``work`` with COM initialised, undoing only an initialisation that succeeded here."""
    ole32 = ctypes.WinDLL("ole32")  # type: ignore[attr-defined]
    ole32.CoInitializeEx.argtypes = (ctypes.c_void_p, ctypes.c_ulong)
    ole32.CoInitializeEx.restype = ctypes.c_long
    result = ole32.CoInitializeEx(None, _COINIT_MULTITHREADED)
    if result < 0 and result != _RPC_E_CHANGED_MODE:  # CHANGED_MODE: usable, owned by someone else
        raise _ListingFailed(f"CoInitializeEx: {result:#x}")
    try:
        return work()
    finally:
        if result >= 0:  # S_OK and S_FALSE both must be balanced
            ole32.CoUninitialize()


def _read_dshow_names(category: str = _CATEGORY_VIDEO_INPUT) -> dict[int, str]:
    ole32 = ctypes.WinDLL("ole32")  # type: ignore[attr-defined]
    ole32.CoCreateInstance.argtypes = (
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_ulong,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
    )
    ole32.CoCreateInstance.restype = ctypes.c_long
    with _ComPtr() as device_enum:
        result = ole32.CoCreateInstance(
            _guid(_CLSID_SYSTEM_DEVICE_ENUM),
            None,
            _CLSCTX_INPROC_SERVER,
            _guid(_IID_ICREATE_DEV_ENUM),
            ctypes.byref(device_enum.pointer),
        )
        if result < 0:
            raise _ListingFailed(f"CoCreateInstance: {result:#x}")
        return _read_category_names(device_enum, category)


def _read_category_names(device_enum: _ComPtr, category: str) -> dict[int, str]:
    """Names in enumeration order, the order OpenCV counts in.

    OpenCV's ``videoInput::getDevice`` counts every moniker, so one whose name
    cannot be read still takes up an index here.
    """
    names: dict[int, str] = {}
    create_enumerator = device_enum.method(
        _CREATE_CLASS_ENUMERATOR_SLOT, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.c_ulong
    )
    with _ComPtr() as monikers:
        result = create_enumerator(_guid(category), ctypes.byref(monikers.pointer), 0)
        if result < 0:
            raise _ListingFailed(f"CreateClassEnumerator: {result:#x}")
        if result != _S_OK or not monikers.pointer:  # S_FALSE with no enumerator: the category is empty
            return names
        next_moniker = monikers.method(
            _ENUM_NEXT_SLOT, ctypes.c_ulong, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_ulong)
        )
        for index in range(_MAX_DEVICES):
            with _ComPtr() as moniker:
                fetched = ctypes.c_ulong()
                result = next_moniker(1, ctypes.byref(moniker.pointer), ctypes.byref(fetched))
                if result < 0:
                    raise _ListingFailed(f"IEnumMoniker::Next: {result:#x}")
                if result != _S_OK or not moniker.pointer:  # S_FALSE: the end
                    break
                names[index] = _moniker_friendly_name(moniker) or default_camera_name(index)
    return names


def _moniker_friendly_name(moniker: _ComPtr) -> str | None:
    bind_to_storage = moniker.method(
        _BIND_TO_STORAGE_SLOT, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)
    )
    with _ComPtr() as bag:
        if bind_to_storage(None, None, _guid(_IID_IPROPERTY_BAG), ctypes.byref(bag.pointer)) < 0 or not bag.pointer:
            return None
        read = bag.method(_PROPERTY_BAG_READ_SLOT, ctypes.c_wchar_p, ctypes.POINTER(_Variant), ctypes.c_void_p)
        oleaut32 = ctypes.WinDLL("oleaut32")  # type: ignore[attr-defined]
        oleaut32.VariantInit.argtypes = oleaut32.VariantClear.argtypes = (ctypes.POINTER(_Variant),)
        oleaut32.VariantInit.restype = None
        oleaut32.VariantClear.restype = ctypes.c_long
        variant = _Variant()
        oleaut32.VariantInit(ctypes.byref(variant))
        try:
            if read("FriendlyName", ctypes.byref(variant), None) < 0 or variant.vt != _VT_BSTR or not variant.value:
                return None
            return ctypes.wstring_at(variant.value).strip() or None
        finally:
            oleaut32.VariantClear(ctypes.byref(variant))


def _read_msmf_names() -> dict[int, str]:
    mfplat = ctypes.WinDLL("mfplat")  # type: ignore[attr-defined]
    mf = ctypes.WinDLL("mf")  # type: ignore[attr-defined]
    mfplat.MFStartup.argtypes = (ctypes.c_ulong, ctypes.c_ulong)
    mfplat.MFStartup.restype = ctypes.c_long
    mfplat.MFShutdown.restype = ctypes.c_long
    mfplat.MFCreateAttributes.argtypes = (ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint32)
    mfplat.MFCreateAttributes.restype = ctypes.c_long
    mf.MFEnumDeviceSources.argtypes = (
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p)),
        ctypes.POINTER(ctypes.c_uint32),
    )
    mf.MFEnumDeviceSources.restype = ctypes.c_long
    if mfplat.MFStartup(_MF_VERSION, _MFSTARTUP_LITE) < 0:
        raise _ListingFailed("MFStartup")
    try:
        with _ComPtr() as attributes:
            if mfplat.MFCreateAttributes(ctypes.byref(attributes.pointer), 1) < 0:
                raise _ListingFailed("MFCreateAttributes")
            set_guid = attributes.method(_SET_GUID_SLOT, ctypes.c_void_p, ctypes.c_void_p)
            if set_guid(_guid(_MF_SOURCE_TYPE), _guid(_MF_SOURCE_TYPE_VIDCAP)) < 0:
                raise _ListingFailed("IMFAttributes::SetGUID")
            devices = ctypes.POINTER(ctypes.c_void_p)()
            count = ctypes.c_uint32()
            if mf.MFEnumDeviceSources(attributes.pointer, ctypes.byref(devices), ctypes.byref(count)) < 0:
                raise _ListingFailed("MFEnumDeviceSources")
            names: dict[int, str] = {}
            try:
                for index in range(count.value):
                    with _ComPtr(devices[index]) as activate:
                        names[index] = _activate_friendly_name(activate) or default_camera_name(index)
            finally:
                _co_task_mem_free(devices)
            return names
    finally:
        mfplat.MFShutdown()


def _activate_friendly_name(activate: _ComPtr) -> str | None:
    get_string = activate.method(
        _GET_ALLOCATED_STRING_SLOT,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_uint32),
    )
    text = ctypes.c_void_p()
    length = ctypes.c_uint32()
    if get_string(_guid(_MF_FRIENDLY_NAME), ctypes.byref(text), ctypes.byref(length)) < 0 or not text:
        return None
    try:
        return ctypes.wstring_at(text).strip() or None
    finally:
        _co_task_mem_free(text)


def _co_task_mem_free(memory: Any) -> None:
    ole32 = ctypes.WinDLL("ole32")  # type: ignore[attr-defined]
    ole32.CoTaskMemFree.argtypes = (ctypes.c_void_p,)
    ole32.CoTaskMemFree.restype = None
    ole32.CoTaskMemFree(memory)


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

    Where the OS gives device names, every (re)open checks them first: a
    virtual camera is refused, and the device is followed by name when plugging
    or unplugging another camera has renumbered it. ``name`` says which device
    is meant from the start; without it, the first open decides. An index without
    a device is not tried at all, and on Windows nothing is opened while the
    names cannot be read. There the DirectShow index is not valid for Media
    Foundation, so that fallback opens the device with the same name, or nothing.
    """

    def __init__(
        self,
        index: int = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        *,
        name: str | None = None,
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
        self._requested_name = _real_name(name, index)
        self._device_name = self._requested_name  # else set by the first successful open
        self._refusal: str | None = None  # why the last open attempt did not touch the device

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
            raise CameraError(self._refusal or camera_open_error_message(self.index))
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
        self._device_name = self._requested_name
        self._release()

    # ---- internals --------------------------------------------------------

    def _open_device(self) -> bool:
        names = _device_names()
        self._refusal = None
        if names is None:
            self._refusal = CAMERA_LIST_FAILED_MESSAGE
            logger.info("camera %d: not opened, the device names are unknown", self.index)
            return False
        if not self._device_allowed(names):
            return False
        with quiet_opencv_logs:
            for api in capture_backends():
                index = self._backend_index(api, names)
                if index is None:
                    logger.info("camera %d: no matching device for %s", self.index, _BACKEND_NAMES.get(api, api))
                    continue
                cap = self._try_backend(index, api)
                if cap is not None:
                    self._cap, self._api, self._failures = cap, api, 0
                    self._device_name = _real_name(names.get(self.index), self.index) or self._device_name
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

    def _device_allowed(self, names: dict[int, str]) -> bool:
        """Follow a renumbered device; rule out virtual and absent ones (only when names are known)."""
        if not names:
            return True
        pinned = self._device_name
        if pinned is not None and names.get(self.index) != pinned:
            moved = [index for index in sorted(names) if names[index] == pinned]
            if not moved:
                self._refusal = camera_missing_message(pinned)
                return False  # unplugged; retried later
            logger.info("camera %r moved from index %d to %d", pinned, self.index, moved[0])
            self.index = moved[0]
        name = names.get(self.index)
        if name is None:
            return False
        if is_virtual_camera_name(name):
            self._refusal = virtual_camera_as_input_message(name)
            logger.info("camera %d is the virtual camera %r; not opening it", self.index, name)
            return False
        return True

    def _backend_index(self, api: int, names: dict[int, str]) -> int | None:
        """This device's index for ``api``, or None if that backend must not be tried."""
        if api != cv2.CAP_MSMF:
            return self.index
        msmf_names = _msmf_device_names()
        if msmf_names is None:
            return None
        if not names:  # nothing to compare with, but a virtual camera can still be ruled out
            other = msmf_names.get(self.index)
            return None if other is not None and is_virtual_camera_name(other) else self.index
        matches = [index for index, name in msmf_names.items() if name == names[self.index]]
        return matches[0] if len(matches) == 1 else None

    def _try_backend(self, index: int, api: int) -> VideoCaptureLike | None:
        try:
            cap = self._capture_factory(index, api)
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


class VideoFileSource:
    """A video file standing in for the webcam (testing without a camera).

    Same interface as ``CameraSource``. Frames come at the file's own frame
    rate in real time, scaled and padded to the requested size like a webcam
    image, and the file loops at its end. A very slow file delivers each frame
    several times, and a still image the same frame for good, so the runner never
    takes it for a lost camera. ``open`` raises ``CameraError``; ``read`` never
    raises and returns None when no frame is available. ``close`` may be called
    while another thread is in ``read``.
    """

    def __init__(
        self,
        path: str | Path,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        *,
        reopen_interval_s: float = REOPEN_INTERVAL_S,
        capture_factory: Callable[[str, int], VideoCaptureLike] = cv2.VideoCapture,
    ) -> None:
        self.path = Path(path)
        self.width = width
        self.height = height
        self.fps = fps
        # What read() delivers (valid once opened).
        self.actual_width = 0
        self.actual_height = 0
        self.actual_fps = 0.0
        self._reopen_interval_s = reopen_interval_s
        self._capture_factory = capture_factory
        self._cap: VideoCaptureLike | None = None
        self._api: int | None = None
        self._active = False  # between a successful open() and close()
        self._lock = threading.Lock()  # serialises the decoder between read() and close()
        self._pending: np.ndarray | None = None  # read by open(), delivered by the first read()
        self._repeat_each = 1  # how many times each frame of the file is delivered
        self._repeated: np.ndarray | None = None
        self._repeats_left = 0
        self._next_frame_at = 0.0
        self._next_reopen_at = 0.0

    @property
    def is_open(self) -> bool:
        return self._cap is not None

    @property
    def backend(self) -> str | None:
        """Name of the OpenCV backend decoding the file, e.g. "FFmpeg"."""
        return None if self._api is None else _BACKEND_NAMES.get(self._api, str(self._api))

    def open(self) -> None:
        """Open the file; raises ``CameraError`` if it gives no frame."""
        if self._cap is not None:
            return
        if not self.path.is_file():
            raise CameraError(f"動画ファイルが見つかりません: {self.path}")
        self._pending = self._start()
        if self._pending is None or self._cap is None:
            raise CameraError(f"動画ファイルを読み込めませんでした（動画ではないか、壊れています）: {self.path}")
        self._set_pace(float(self._cap.get(cv2.CAP_PROP_FPS) or 0.0), self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.actual_width, self.actual_height = self.width, self.height
        self._next_frame_at = time.monotonic()
        self._active = True
        logger.info("video file %s opened: %.2f fps", self.path.name, self.actual_fps)

    def read(self) -> np.ndarray | None:
        """The next uint8 BGR frame at the requested size, once it is due; None if there is none."""
        try:
            if not self._active:
                return None
            self._wait_until_due()
            with self._lock:
                frame = self._next_frame() if self._active else None
            return None if frame is None else fit_frame(_to_bgr(frame), self.width, self.height)
        except Exception as exc:  # cv2.error, or anything else from a damaged file
            logger.debug("reading %s failed: %s", self.path.name, exc)
            return None

    def close(self) -> None:
        self._active = False
        with self._lock:
            self._pending = self._repeated = None
            self._repeats_left = 0
            self._release()

    # ---- internals --------------------------------------------------------

    def _set_pace(self, file_fps: float, frame_count: float) -> None:
        low, high = _VIDEO_FILE_FPS_RANGE
        self._repeat_each = 1
        self.actual_fps = float(self.fps)
        if low <= file_fps <= high:
            self.actual_fps = file_fps
        elif 0.0 < file_fps < low:
            self._repeat_each = math.ceil(low / file_fps)
            self.actual_fps = file_fps * self._repeat_each
        elif frame_count == 1.0:  # no frame rate and one frame: a still image, which needs no reopening
            self._repeat_each = sys.maxsize

    def _start(self) -> np.ndarray | None:
        """(Re)open the file and return its first frame; None (and nothing open) if it gives none."""
        self._release()
        with quiet_opencv_logs:
            for api in (cv2.CAP_FFMPEG, cv2.CAP_ANY):
                frame = self._try_decoder(api)
                if frame is not None:
                    return frame
        return None

    def _try_decoder(self, api: int) -> np.ndarray | None:
        try:
            cap = self._capture_factory(str(self.path), api)
        except cv2.error:
            return None
        try:
            ok, frame = cap.read() if cap.isOpened() and self._has_picture(cap, api) else (False, None)
        except cv2.error:
            ok, frame = False, None
        if ok and frame is not None and frame.size:
            self._cap, self._api = cap, api
            return frame
        cap.release()
        return None

    @staticmethod
    def _has_picture(cap: VideoCaptureLike, api: int) -> bool:
        """False for a file without a video track, which must not be read.

        FFmpeg refuses to open those. Media Foundation (behind ``CAP_ANY``) opens
        them with a size of 0x0, and its ``read`` then never returns.
        """
        if api == cv2.CAP_FFMPEG:
            return True
        return cap.get(cv2.CAP_PROP_FRAME_WIDTH) > 0 and cap.get(cv2.CAP_PROP_FRAME_HEIGHT) > 0

    def _wait_until_due(self) -> None:
        interval = 1.0 / self.actual_fps
        now = time.monotonic()
        while now < self._next_frame_at and self._active:
            time.sleep(min(self._next_frame_at - now, _WAIT_SLICE_S))
            now = time.monotonic()
        # Exact schedule while on time; after a stall continue from now, without a burst of frames.
        behind = now - self._next_frame_at >= interval
        self._next_frame_at = (now if behind else self._next_frame_at) + interval

    def _next_frame(self) -> np.ndarray | None:
        if self._repeats_left > 0 and self._repeated is not None:
            self._repeats_left -= 1
            return self._repeated.copy()  # the receiver owns what it gets
        frame, self._pending = self._pending, None
        if frame is None:
            frame = self._decode_next()
        if frame is not None and self._repeat_each > 1:
            self._repeated, self._repeats_left = frame, self._repeat_each - 1
            return frame.copy()
        return frame

    def _decode_next(self) -> np.ndarray | None:
        if self._cap is not None:
            try:
                ok, frame = self._cap.read()
            except cv2.error:
                ok, frame = False, None
            if ok and frame is not None and frame.size:
                return frame
            self._release()  # the end of the file, or a damaged tail
            self._next_reopen_at = 0.0
        if time.monotonic() < self._next_reopen_at:
            return None
        # Reopening costs a few ms and, unlike seeking, is exact for every container and codec.
        frame = self._start()
        if frame is None:
            self._next_reopen_at = time.monotonic() + self._reopen_interval_s
        return frame

    def _release(self) -> None:
        cap, self._cap = self._cap, None
        if cap is not None:
            cap.release()


def video_file_camera_factory(path: str | Path) -> Callable[..., VideoFileSource]:
    """A ``CameraRunner`` ``camera_factory`` that plays ``path`` whichever camera index is asked for."""

    def make(
        *, index: int = 0, width: int = 1280, height: int = 720, fps: int = 30, name: str | None = None
    ) -> VideoFileSource:
        return VideoFileSource(path, width, height, fps)

    return make


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


def virtual_camera_size(width: int, height: int) -> tuple[int, int]:
    """The size to run the virtual camera at for a capture of ``width`` x ``height``.

    Always 16:9, the format meeting apps ask for: OBS Virtual Camera stretches its
    picture to the app's format, so 640x480 sent to an app at 1280x720 shows the
    face a third too wide. Never below 1280x720, which spares the picture OBS's
    coarse upscaling. Frames of other shapes go in with black bars (``fit_frame``).
    """
    low, high = _OUTPUT_UNITS_RANGE
    units = min(high, max(low, math.ceil(width / 16), math.ceil(height / 9)))
    units += units % 2  # both sides must be even for the video format OBS uses
    return 16 * units, 9 * units


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
