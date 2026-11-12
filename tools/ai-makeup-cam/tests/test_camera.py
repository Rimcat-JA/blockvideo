"""Camera input / virtual camera output, with fake OpenCV captures and a fake pyvirtualcam."""

from __future__ import annotations

import collections
import ctypes
import os
import subprocess
import sys
import threading
import time
import types
import wave
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

import cv2
import numpy as np
import pytest

from makeup_cam import camera
from makeup_cam.camera import (
    REOPEN_AFTER_FAILURES,
    CameraError,
    CameraInfo,
    CameraSource,
    VideoFileSource,
    VirtualCameraError,
    VirtualCameraOutput,
    fit_frame,
    is_virtual_camera_name,
    list_cameras,
    quiet_opencv_logs,
    resolve_camera,
    video_file_camera_factory,
)

MJPG = cv2.VideoWriter.fourcc(*"MJPG")
windows_only = pytest.mark.skipif(sys.platform != "win32", reason="Windows COM")


class FakeDevices:
    """Stands in for ``cv2.VideoCapture``: a factory of scripted fake captures."""

    def __init__(
        self,
        indices: set[int] | None = None,
        *,
        apis: set[int] | None = None,
        silent_apis: frozenset[int] = frozenset(),
        frame_shape: tuple[int, ...] = (720, 1280, 3),
        open_delay_s: float = 0.0,
    ) -> None:
        self.indices = {0} if indices is None else indices
        self.apis = apis  # None = every backend opens
        self.silent_apis = silent_apis  # open but never deliver frames
        self.frame_shape = frame_shape
        self.open_delay_s = open_delay_s
        self.read_results: collections.deque[bool] = collections.deque()  # scripted, then success
        self.created: list[FakeCapture] = []

    def __call__(self, index: int, api: int) -> FakeCapture:
        time.sleep(self.open_delay_s)
        capture = FakeCapture(self, index, api)
        self.created.append(capture)
        return capture

    @property
    def probed(self) -> list[int]:
        return [capture.index for capture in self.created]


class FakeCapture:
    def __init__(self, devices: FakeDevices, index: int, api: int) -> None:
        self.devices = devices
        self.index = index
        self.api = api
        self.opened = index in devices.indices and (devices.apis is None or api in devices.apis)
        self.sets: list[tuple[int, float]] = []
        self.released = False

    def isOpened(self) -> bool:
        return self.opened and not self.released

    def set(self, prop_id: int, value: float) -> bool:
        self.sets.append((prop_id, value))
        return True

    def get(self, prop_id: int) -> float:
        return {cv2.CAP_PROP_FRAME_WIDTH: 640.0, cv2.CAP_PROP_FRAME_HEIGHT: 480.0, cv2.CAP_PROP_FPS: 30.0}.get(
            prop_id, 0.0
        )

    def read(self) -> tuple[bool, np.ndarray | None]:
        if not self.isOpened() or self.api in self.devices.silent_apis:
            return False, None
        scripted = self.devices.read_results
        if scripted and not scripted.popleft():
            return False, None
        return True, np.full(self.devices.frame_shape, 100, np.uint8)

    def release(self) -> None:
        self.released = True


@pytest.fixture
def on_os(monkeypatch: pytest.MonkeyPatch):
    def set_os(name: str) -> None:
        monkeypatch.setattr(camera, "host_os", lambda: name)

    return set_os


@pytest.fixture(autouse=True)
def no_real_devices(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The devices of the PC running the tests (OBS Virtual Camera, say) must not change the results."""
    monkeypatch.setattr(camera, "_V4L2_SYSFS", tmp_path / "no-such-sysfs")
    monkeypatch.setattr(camera, "_dshow_device_names", dict)
    monkeypatch.setattr(camera, "_msmf_device_names", dict)


@pytest.fixture
def windows_devices(monkeypatch: pytest.MonkeyPatch, on_os) -> Callable[..., None]:
    """Scripts what Windows lists: DirectShow names, and Media Foundation names."""

    def script(dshow: dict[int, str], msmf: dict[int, str] | None = None) -> None:
        on_os("windows")
        monkeypatch.setattr(camera, "_dshow_device_names", lambda: dict(dshow))
        monkeypatch.setattr(camera, "_msmf_device_names", lambda: dict(msmf or {}))

    return script


# ---- CameraSource ---------------------------------------------------------


def test_windows_requests_mjpg_before_the_size(on_os) -> None:
    on_os("windows")
    devices = FakeDevices()
    source = CameraSource(0, 1280, 720, 30, capture_factory=devices)

    source.open()

    capture = devices.created[0]
    assert capture.api == cv2.CAP_DSHOW
    assert capture.sets[:4] == [
        (cv2.CAP_PROP_FOURCC, MJPG),
        (cv2.CAP_PROP_FRAME_WIDTH, 1280),
        (cv2.CAP_PROP_FRAME_HEIGHT, 720),
        (cv2.CAP_PROP_FPS, 30),
    ]
    assert source.is_open and source.backend == "DirectShow"
    assert (source.actual_width, source.actual_height, source.actual_fps) == (1280, 720, 30.0)


@pytest.mark.parametrize(
    "devices",
    [
        FakeDevices(apis={cv2.CAP_MSMF}),  # DirectShow cannot open it
        FakeDevices(silent_apis=frozenset({cv2.CAP_DSHOW})),  # DirectShow opens but gives no frames
    ],
)
def test_windows_falls_back_to_media_foundation(on_os, devices: FakeDevices) -> None:
    on_os("windows")
    source = CameraSource(0, capture_factory=devices)

    source.open()

    dshow, msmf = devices.created
    assert dshow.released and not msmf.released
    assert source.backend == "Media Foundation"
    assert (cv2.CAP_PROP_FOURCC, MJPG) not in msmf.sets
    assert source.read() is not None


@pytest.mark.parametrize(
    ("os_name", "hint"),
    [
        ("windows", "デスクトップ アプリがカメラにアクセスできる"),
        ("macos", "プライバシーとセキュリティ → カメラ"),
        ("linux", "/dev/video"),
    ],
)
def test_open_failure_explains_what_to_do(on_os, os_name: str, hint: str) -> None:
    on_os(os_name)
    devices = FakeDevices(indices=set())
    source = CameraSource(3, capture_factory=devices)

    with pytest.raises(CameraError) as error:
        source.open()

    assert "カメラ 3 を開けませんでした" in str(error.value)
    assert hint in str(error.value)
    assert devices.created and all(capture.released for capture in devices.created)
    assert not source.is_open and source.read() is None


@pytest.mark.parametrize("shape", [(48, 64), (48, 64, 4)])
def test_read_returns_three_channel_bgr(on_os, shape: tuple[int, ...]) -> None:
    on_os("linux")
    source = CameraSource(capture_factory=FakeDevices(frame_shape=shape))
    source.open()

    frame = source.read()

    assert frame is not None and frame.shape == (48, 64, 3) and frame.dtype == np.uint8


def test_reopens_after_repeated_read_failures(on_os) -> None:
    on_os("linux")
    devices = FakeDevices()
    source = CameraSource(capture_factory=devices)
    source.open()
    devices.read_results.extend([False] * REOPEN_AFTER_FAILURES)

    results = [source.read() for _ in range(REOPEN_AFTER_FAILURES)]

    assert results == [None] * REOPEN_AFTER_FAILURES
    assert devices.created[0].released and not source.is_open
    assert source.read() is not None  # reopened on the next read
    assert len(devices.created) == 2 and source.is_open


def test_reopen_attempts_are_rate_limited(on_os) -> None:
    on_os("linux")
    devices = FakeDevices()
    source = CameraSource(capture_factory=devices, reopen_interval_s=0.2)
    source.open()
    devices.indices.clear()  # unplugged
    devices.read_results.extend([False] * REOPEN_AFTER_FAILURES)

    for _ in range(REOPEN_AFTER_FAILURES + 10):
        assert source.read() is None
    assert len(devices.created) == 2  # the original + one reopen attempt

    devices.indices.add(0)  # plugged back in
    time.sleep(0.25)
    assert source.read() is not None
    assert len(devices.created) == 3


def test_close_is_idempotent_and_stops_reopening(on_os) -> None:
    on_os("linux")
    devices = FakeDevices()
    source = CameraSource(capture_factory=devices)
    source.open()

    source.close()
    source.close()

    assert devices.created[0].released
    assert source.read() is None
    assert len(devices.created) == 1


# ---- never open a virtual camera, never the wrong device --------------------


@pytest.mark.parametrize(
    "name",
    [
        "OBS Virtual Camera",
        "OBS-Camera",
        "OBS Camera 2",
        "Unity Video Capture",
        "XSplit VCam",
        "ManyCam Virtual Webcam",
        "Streamlabs Virtual Webcam",
        "e2eSoft VirtualCam",
        "AI Makeup Cam",
    ],
)
def test_virtual_camera_names_are_recognised(name: str) -> None:
    assert is_virtual_camera_name(name)


@pytest.mark.parametrize(
    "name",
    [
        "HD Pro Webcam C920",
        "Logi C270 HD WebCam",
        "OBSBOT Tiny 2",
        "USB2.0 UVC Camera",
        "UVCam",
        "HP TrueVision HD Camera",
        "カメラ 0",
    ],
)
def test_real_camera_names_are_not_mistaken_for_virtual_cameras(name: str) -> None:
    assert not is_virtual_camera_name(name)


@pytest.mark.parametrize("os_name", ["windows", "linux"])
def test_a_virtual_camera_is_never_opened_as_the_input(monkeypatch: pytest.MonkeyPatch, on_os, os_name: str) -> None:
    on_os(os_name)
    monkeypatch.setattr(camera, "_device_names", lambda: {0: "OBS Virtual Camera"})
    devices = FakeDevices()
    source = CameraSource(0, capture_factory=devices)

    with pytest.raises(CameraError) as error:
        source.open()

    assert "OBS Virtual Camera" in str(error.value)
    assert "仮想カメラ" in str(error.value) and "Web カメラ" in str(error.value)
    assert devices.created == []  # no backend at all was tried
    assert not source.is_open and source.read() is None


def test_an_index_without_a_device_is_not_tried_when_names_are_known(windows_devices) -> None:
    windows_devices({0: "USB Camera"}, {0: "USB Camera", 1: "Other Camera"})
    devices = FakeDevices(indices={0, 1})
    source = CameraSource(1, capture_factory=devices)

    with pytest.raises(CameraError, match="カメラ 1 を開けませんでした"):
        source.open()

    assert devices.created == []


def test_reopen_after_unplugging_refuses_the_virtual_camera_that_took_the_index(
    windows_devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    windows_devices({0: "USB Camera", 1: "OBS Virtual Camera"})
    devices = FakeDevices(indices={0, 1})
    source = CameraSource(0, capture_factory=devices, reopen_interval_s=0.0)
    source.open()
    assert source.backend == "DirectShow"

    windows_devices({0: "OBS Virtual Camera"})  # the webcam is gone; OBS is now device 0
    devices.read_results.extend([False] * REOPEN_AFTER_FAILURES)
    for _ in range(REOPEN_AFTER_FAILURES + 5):
        assert source.read() is None
    assert len(devices.created) == 1  # OBS was never opened

    windows_devices({0: "USB Camera", 1: "OBS Virtual Camera"})  # plugged back in
    assert source.read() is not None
    assert [capture.index for capture in devices.created] == [0, 0]


def test_reopen_follows_the_device_by_name_when_it_was_renumbered(windows_devices) -> None:
    windows_devices({0: "Built-in", 1: "USB Camera", 2: "OBS Virtual Camera"})
    devices = FakeDevices(indices={0, 1, 2})
    source = CameraSource(1, capture_factory=devices, reopen_interval_s=0.0)
    source.open()

    windows_devices({0: "USB Camera", 1: "OBS Virtual Camera"})  # the built-in camera went away
    devices.read_results.extend([False] * REOPEN_AFTER_FAILURES)
    for _ in range(REOPEN_AFTER_FAILURES):
        source.read()

    assert source.read() is not None
    assert source.index == 0
    assert [capture.index for capture in devices.created] == [1, 0]


def test_a_named_camera_is_found_under_its_new_index_at_the_first_open(windows_devices) -> None:
    windows_devices({0: "USB Camera", 1: "OBS Virtual Camera"})
    devices = FakeDevices(indices={0, 1})
    source = CameraSource(1, capture_factory=devices, name="USB Camera")  # chosen when it was still index 1

    source.open()

    assert [capture.index for capture in devices.created] == [0]
    assert source.index == 0


def test_a_named_camera_never_opens_another_device_that_took_its_index(windows_devices) -> None:
    windows_devices({0: "USB Camera", 1: "Other Camera"})
    devices = FakeDevices(indices={0, 1})
    source = CameraSource(1, capture_factory=devices, name="USB Camera")

    source.open()

    assert [capture.index for capture in devices.created] == [0]


def test_a_named_camera_that_is_gone_is_reported_by_name_and_found_when_back(windows_devices) -> None:
    windows_devices({0: "Other Camera"})
    devices = FakeDevices(indices={0, 1})
    source = CameraSource(1, capture_factory=devices, name="USB Camera")

    with pytest.raises(CameraError, match="「USB Camera」が見つかりません"):
        source.open()
    assert devices.created == []

    windows_devices({0: "Other Camera", 1: "USB Camera"})
    source.open()
    assert [capture.index for capture in devices.created] == [1]


def test_a_placeholder_name_does_not_pin_a_device(windows_devices) -> None:
    windows_devices({0: "USB Camera", 1: "カメラ 1"})  # a device whose name could not be read
    devices = FakeDevices(indices={0, 1})
    source = CameraSource(1, capture_factory=devices, name="カメラ 1")

    source.open()

    assert [capture.index for capture in devices.created] == [1]


def test_a_named_camera_keeps_its_name_when_reopened_after_close(windows_devices) -> None:
    windows_devices({0: "USB Camera", 1: "Other Camera"})
    devices = FakeDevices(indices={0, 1})
    source = CameraSource(1, capture_factory=devices, name="USB Camera")
    source.open()
    source.close()

    windows_devices({0: "Other Camera", 1: "USB Camera"})
    source.open()

    assert [capture.index for capture in devices.created] == [0, 1]


# ---- when Windows cannot list the devices, nothing is opened ------------------


def test_nothing_is_opened_while_windows_cannot_list_the_devices(
    windows_devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    windows_devices({})
    monkeypatch.setattr(camera, "_dshow_device_names", lambda: None)  # OBS may well be device 0: nobody knows
    devices = FakeDevices(indices={0})
    source = CameraSource(0, capture_factory=devices)

    with pytest.raises(CameraError, match="一覧を取得できない"):
        source.open()

    assert devices.created == []
    assert not source.is_open and source.read() is None


def test_a_camera_waiting_for_the_device_list_opens_when_the_list_is_back(
    windows_devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    windows_devices({0: "USB Camera"})
    devices = FakeDevices(indices={0})
    source = CameraSource(0, capture_factory=devices, reopen_interval_s=0.0)
    source.open()

    monkeypatch.setattr(camera, "_dshow_device_names", lambda: None)
    devices.read_results.extend([False] * REOPEN_AFTER_FAILURES)
    for _ in range(REOPEN_AFTER_FAILURES + 5):
        assert source.read() is None
    assert len(devices.created) == 1  # nothing was reopened blindly

    windows_devices({0: "USB Camera"})
    assert source.read() is not None
    assert len(devices.created) == 2


def test_listing_cameras_opens_nothing_while_windows_cannot_list_the_devices(
    windows_devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    windows_devices({})
    monkeypatch.setattr(camera, "_dshow_device_names", lambda: None)
    devices = FakeDevices(indices={0, 1})

    assert list_cameras(in_use={0}, capture_factory=devices) == []
    assert devices.created == []


def test_media_foundation_is_not_tried_when_it_cannot_list_its_devices(
    windows_devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    for dshow in ({}, {0: "USB Camera"}):
        windows_devices(dshow)
        monkeypatch.setattr(camera, "_msmf_device_names", lambda: None)
        devices = FakeDevices(indices={0}, apis={cv2.CAP_MSMF})
        with pytest.raises(CameraError):
            CameraSource(0, capture_factory=devices).open()
        assert all(capture.api == cv2.CAP_DSHOW for capture in devices.created)


def test_media_foundation_fallback_opens_the_same_device_under_its_own_index(windows_devices) -> None:
    # OBS has no Media Foundation device, so the two numberings differ.
    windows_devices(
        {0: "USB Camera A", 1: "OBS Virtual Camera", 2: "USB Camera B"},
        {0: "USB Camera B", 1: "USB Camera A"},
    )
    devices = FakeDevices(indices={0, 1, 2}, silent_apis=frozenset({cv2.CAP_DSHOW}))
    source = CameraSource(2, capture_factory=devices)

    source.open()

    assert [(capture.index, capture.api) for capture in devices.created] == [(2, cv2.CAP_DSHOW), (0, cv2.CAP_MSMF)]
    assert source.backend == "Media Foundation"
    assert source.index == 2


@pytest.mark.parametrize(
    "msmf",
    [
        {},  # could not be listed
        {0: "USB Camera B", 1: "Other Camera"},  # our device is not there, index 0 is another camera
        {0: "USB Camera A", 1: "USB Camera A"},  # two identical cameras: cannot tell which is which
    ],
)
def test_media_foundation_fallback_never_guesses(windows_devices, msmf: dict[int, str]) -> None:
    windows_devices({0: "USB Camera A", 1: "USB Camera C"}, msmf)
    devices = FakeDevices(indices={0, 1, 2}, silent_apis=frozenset({cv2.CAP_DSHOW}))
    source = CameraSource(0, capture_factory=devices)

    with pytest.raises(CameraError, match="カメラ 0 を開けませんでした"):
        source.open()

    assert [(capture.index, capture.api) for capture in devices.created] == [(0, cv2.CAP_DSHOW)]
    assert all(capture.released for capture in devices.created)


def test_media_foundation_without_directshow_names_skips_a_virtual_camera(windows_devices) -> None:
    windows_devices({}, {0: "OBS Virtual Camera", 1: "USB Camera"})
    devices = FakeDevices(indices={0, 1}, apis={cv2.CAP_MSMF})
    source = CameraSource(0, capture_factory=devices)

    with pytest.raises(CameraError):
        source.open()
    assert all(capture.api == cv2.CAP_DSHOW for capture in devices.created)

    source = CameraSource(1, capture_factory=devices)
    source.open()
    assert source.backend == "Media Foundation"


def test_list_cameras_uses_directshow_names_and_never_opens_virtual_cameras(windows_devices) -> None:
    windows_devices({0: "OBS Virtual Camera", 1: "HD Pro Webcam C920", 2: "Unplugged Meanwhile"})
    devices = FakeDevices(indices={0, 1})

    cameras = list_cameras(in_use={0}, capture_factory=devices)

    assert cameras == [CameraInfo(1, "HD Pro Webcam C920", 640, 480)]
    assert devices.probed == [1, 2]
    assert {capture.api for capture in devices.created} == {cv2.CAP_DSHOW}


def test_list_cameras_leaves_the_device_in_use_alone_wherever_it_is_now(windows_devices) -> None:
    windows_devices({0: "USB Camera", 1: "Other Camera", 2: "OBS Virtual Camera"})  # USB Camera was index 1 before
    devices = FakeDevices(indices={0, 1, 2})

    cameras = list_cameras(in_use=("USB Camera",), capture_factory=devices)

    assert cameras == [CameraInfo(0, "USB Camera", 0, 0), CameraInfo(1, "Other Camera", 640, 480)]
    assert devices.probed == [1]


def test_a_name_in_use_stands_for_one_device_only(on_os, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    on_os("linux")
    for node in ("video0", "video1"):  # a webcam and its metadata node, which cannot be opened, share a name
        (tmp_path / node).mkdir()
        (tmp_path / node / "name").write_text("Integrated Camera\n", encoding="utf-8")
    monkeypatch.setattr(camera, "_V4L2_SYSFS", tmp_path)
    devices = FakeDevices(indices={0})

    cameras = list_cameras(in_use=("Integrated Camera",), capture_factory=devices)

    assert cameras == [CameraInfo(0, "Integrated Camera", 0, 0)]
    assert devices.probed == [1]


REAL = {0: "OBS Virtual Camera", 1: "Built-in", 2: "USB Camera"}


@pytest.mark.parametrize(
    ("names", "index", "name", "expected"),
    [
        ({}, 3, "USB Camera", (3, None)),  # no names: only the index to go on
        ({}, None, None, (0, None)),
        (REAL, None, None, (1, "Built-in")),  # default: the first real camera
        (REAL, 2, "USB Camera", (2, "USB Camera")),
        ({0: "USB Camera", 1: "OBS Virtual Camera"}, 2, "USB Camera", (0, "USB Camera")),  # renumbered
        (REAL, 0, "OBS Virtual Camera", (1, "Built-in")),  # never a virtual camera, even if remembered
        (REAL, 2, "Unplugged Camera", (1, "Built-in")),  # the remembered camera is gone
        (REAL, 2, None, (2, "USB Camera")),  # remembered before names existed
        (REAL, 2, "カメラ 2", (2, "USB Camera")),  # a placeholder name is no name
        (REAL, 0, None, (1, "Built-in")),  # the remembered index is a virtual camera now
        ({0: "OBS Virtual Camera"}, None, None, (0, None)),  # nothing real: refused when opened
        ({0: "OBS Virtual Camera"}, 4, "USB Camera", (4, None)),
        (None, 3, "USB Camera", (3, "USB Camera")),  # the list is unknown for now: the choice stands, name included
        (None, None, None, (0, None)),
        (None, 3, "カメラ 3", (3, None)),
    ],
)
def test_resolve_camera(
    monkeypatch: pytest.MonkeyPatch,
    names: dict[int, str] | None,
    index: int | None,
    name: str | None,
    expected: tuple[int, str | None],
) -> None:
    monkeypatch.setattr(camera, "_device_names", lambda: names)
    assert resolve_camera(index, name) == expected


# ---- Windows device names ---------------------------------------------------


def test_device_names_on_windows_are_the_directshow_names(windows_devices) -> None:
    windows_devices({0: "USB Camera"}, {5: "Not this one"})
    assert camera._device_names() == {0: "USB Camera"}


def test_listing_windows_devices_is_a_no_op_elsewhere(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.undo()
    monkeypatch.setattr(sys, "platform", "darwin")
    for name in ("WinDLL", "windll", "WINFUNCTYPE"):
        monkeypatch.delattr(ctypes, name, raising=False)
    assert camera._dshow_device_names() == {}
    assert camera._msmf_device_names() == {}


def test_importing_the_module_does_not_touch_windows_only_ctypes() -> None:
    code = (
        "import ctypes, cv2, numpy\n"
        "for name in ('WinDLL', 'windll', 'WINFUNCTYPE', 'oledll', 'OleDLL'):\n"
        "    if hasattr(ctypes, name):\n"
        "        delattr(ctypes, name)\n"
        "import makeup_cam.camera\n"
        "print('imported')\n"
    )
    root = Path(camera.__file__).parent.parent
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=root, timeout=120)
    assert done.returncode == 0, done.stderr
    assert "imported" in done.stdout


def test_a_failing_enumeration_gives_no_names_and_does_not_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    def explode() -> dict[int, str]:
        raise OSError("mf.dll is not on Windows N")

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(camera, "_with_com", lambda work: work())
    assert camera._enumerate_on_own_thread(explode) is None


def test_a_hung_driver_does_not_freeze_the_caller(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(camera, "_with_com", lambda work: work())
    monkeypatch.setattr(camera, "_ENUMERATION_TIMEOUT_S", 0.05)
    release = threading.Event()

    def hang() -> dict[int, str]:
        release.wait(5.0)
        return {0: "too late"}

    started = time.monotonic()
    assert camera._enumerate_on_own_thread(hang) is None
    assert time.monotonic() - started < 1.0
    release.set()


def test_a_listing_with_no_free_slot_counts_as_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(camera, "_enumeration_slots", threading.BoundedSemaphore(1))
    camera._enumeration_slots.acquire()  # hung drivers hold every slot

    assert camera._enumerate_on_own_thread(dict) is None


def test_a_listing_that_finds_nothing_is_not_a_failed_one(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(camera, "_with_com", lambda work: work())

    assert camera._enumerate_on_own_thread(dict) == {}


@windows_only
def test_real_directshow_listing_with_real_monikers() -> None:
    # The PC may have no camera; these system categories always have real named monikers.
    legacy_filters = "083863F1-70DE-11D0-BD40-00A0C911CE86"
    names = camera._enumerate_on_own_thread(lambda: camera._read_dshow_names(legacy_filters))

    assert names and sorted(names) == list(range(len(names)))
    assert all(isinstance(name, str) and name.strip() for name in names.values())
    assert any(name != camera.default_camera_name(index) for index, name in names.items())


@windows_only
def test_real_video_input_listings_never_raise() -> None:
    from makeup_cam.camera import _enumerate_on_own_thread, _read_dshow_names, _read_msmf_names

    for names in (_enumerate_on_own_thread(_read_dshow_names), _enumerate_on_own_thread(_read_msmf_names)):
        assert isinstance(names, dict)
        assert all(isinstance(index, int) and isinstance(name, str) for index, name in names.items())


@windows_only
def test_real_media_foundation_string_slots_read_the_friendly_name() -> None:
    def work() -> dict[int, str]:
        mfplat = ctypes.WinDLL("mfplat")
        mfplat.MFCreateAttributes.argtypes = (ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint32)
        mfplat.MFCreateAttributes.restype = ctypes.c_long
        with camera._ComPtr() as attributes:
            assert mfplat.MFCreateAttributes(ctypes.byref(attributes.pointer), 1) == 0
            set_string = attributes.method(25, ctypes.c_void_p, ctypes.c_wchar_p)  # IMFAttributes::SetString
            assert set_string(camera._guid(camera._MF_FRIENDLY_NAME), "テスト カメラ") == 0
            return {0: camera._activate_friendly_name(attributes) or ""}

    assert camera._enumerate_on_own_thread(work) == {0: "テスト カメラ"}


@windows_only
def test_com_initialised_differently_by_the_caller_is_left_alone() -> None:
    from makeup_cam.camera import _read_dshow_names, _with_com

    ole32 = ctypes.WinDLL("ole32")
    ole32.CoInitializeEx.argtypes = (ctypes.c_void_p, ctypes.c_ulong)
    ole32.CoInitializeEx.restype = ctypes.c_long
    apartment_threaded = 2
    outcome: dict[str, Any] = {}

    def on_gui_like_thread() -> None:
        outcome["first"] = ole32.CoInitializeEx(None, apartment_threaded)
        outcome["names"] = _with_com(_read_dshow_names)  # asks for MTA: RPC_E_CHANGED_MODE
        outcome["after"] = ole32.CoInitializeEx(None, apartment_threaded)  # S_FALSE: still initialised
        ole32.CoUninitialize()
        ole32.CoUninitialize()

    thread = threading.Thread(target=on_gui_like_thread)
    thread.start()
    thread.join(10.0)

    assert outcome["first"] == 0 and outcome["after"] == 1
    assert isinstance(outcome["names"], dict)


class FakeCom:
    """A COM object made of ctypes callbacks: an actual vtable, with its reference count."""

    def __init__(self, methods: dict[int, tuple[Any, Callable[..., int]]], slots: int = 12) -> None:
        self.refs = 1
        self._keep: list[Any] = []
        vtable = (ctypes.c_void_p * slots)()

        def release(_this: Any) -> int:
            self.refs -= 1
            return self.refs

        methods = {camera._RELEASE_SLOT: (ctypes.WINFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p), release), **methods}
        for slot, (prototype, function) in methods.items():
            callback = prototype(function)
            self._keep.append(callback)
            vtable[slot] = ctypes.cast(callback, ctypes.c_void_p).value
        self._vtable = vtable
        self._object = ctypes.c_void_p(ctypes.addressof(vtable))
        self.address = ctypes.addressof(self._object)


def com_method(*argtypes: Any) -> Any:
    return ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, *argtypes)


def fake_property_bag(value: str | None, read_result: int = 0) -> FakeCom:
    oleaut32 = ctypes.WinDLL("oleaut32")
    oleaut32.SysAllocString.argtypes = (ctypes.c_wchar_p,)
    oleaut32.SysAllocString.restype = ctypes.c_void_p

    def read(_this: Any, name: str, variant: Any, _log: Any) -> int:
        assert name == "FriendlyName"
        if value is not None:
            variant.contents.vt = camera._VT_BSTR
            variant.contents.value = oleaut32.SysAllocString(value)
        return read_result

    prototype = com_method(ctypes.c_wchar_p, ctypes.POINTER(camera._Variant), ctypes.c_void_p)
    return FakeCom({camera._PROPERTY_BAG_READ_SLOT: (prototype, read)})


def fake_moniker(bag: FakeCom | None) -> FakeCom:
    def bind(_this: Any, _ctx: Any, _left: Any, _iid: Any, out: Any) -> int:
        if bag is None:
            return -2147467259  # E_FAIL
        out[0] = bag.address
        return 0

    prototype = com_method(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p))
    return FakeCom({camera._BIND_TO_STORAGE_SLOT: (prototype, bind)})


def fake_device_enum(monikers: list[FakeCom]) -> tuple[FakeCom, FakeCom]:
    pending = iter(monikers)

    def next_moniker(_this: Any, count: int, out: Any, fetched: Any) -> int:
        assert count == 1
        moniker = next(pending, None)
        if moniker is None:
            fetched[0] = 0
            return 1  # S_FALSE: the end
        out[0] = moniker.address
        fetched[0] = 1
        return 0

    prototype = com_method(ctypes.c_ulong, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_ulong))
    enumerator = FakeCom({camera._ENUM_NEXT_SLOT: (prototype, next_moniker)})

    def create(_this: Any, _category: Any, out: Any, flags: int) -> int:
        assert flags == 0
        out[0] = enumerator.address
        return 0

    prototype = com_method(ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.c_ulong)
    return FakeCom({camera._CREATE_CLASS_ENUMERATOR_SLOT: (prototype, create)}), enumerator


@windows_only
def test_listing_counts_every_device_and_releases_every_reference() -> None:
    bags = [fake_property_bag("Camera A"), fake_property_bag(None, read_result=-2147467259), fake_property_bag("  ")]
    monikers = [fake_moniker(bags[0]), fake_moniker(None), fake_moniker(bags[1]), fake_moniker(bags[2])]
    monikers.append(fake_moniker(fake_property_bag("Camera E")))
    device_enum, enumerator = fake_device_enum(monikers)

    with camera._ComPtr(device_enum.address) as owned:
        names = camera._read_category_names(owned, camera._CATEGORY_VIDEO_INPUT)

    assert names == {
        0: "Camera A",
        1: "カメラ 1",  # could not bind: still takes up an index, as in OpenCV
        2: "カメラ 2",  # no name
        3: "カメラ 3",  # blank name
        4: "Camera E",
    }
    assert device_enum.refs == enumerator.refs == 0
    assert all(moniker.refs == 0 for moniker in monikers)
    assert all(bag.refs == 0 for bag in bags)


def fake_activate(name: str | None) -> FakeCom:
    ole32 = ctypes.WinDLL("ole32")
    ole32.CoTaskMemAlloc.argtypes = (ctypes.c_size_t,)
    ole32.CoTaskMemAlloc.restype = ctypes.c_void_p

    def get_string(_this: Any, _key: Any, out: Any, length: Any) -> int:
        if name is None:
            return -1072875802  # MF_E_ATTRIBUTENOTFOUND
        memory = ole32.CoTaskMemAlloc(2 * (len(name) + 1))
        ctypes.memmove(memory, ctypes.create_unicode_buffer(name), 2 * (len(name) + 1))
        out[0] = memory
        length[0] = len(name)
        return 0

    prototype = com_method(ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_uint32))
    return FakeCom({camera._GET_ALLOCATED_STRING_SLOT: (prototype, get_string)}, slots=16)


@windows_only
def test_media_foundation_listing_releases_every_device_and_frees_the_array(monkeypatch: pytest.MonkeyPatch) -> None:
    activates = [fake_activate("USB Camera"), fake_activate(None), fake_activate("Other")]
    ole32 = ctypes.WinDLL("ole32")
    ole32.CoTaskMemAlloc.argtypes = (ctypes.c_size_t,)
    ole32.CoTaskMemAlloc.restype = ctypes.c_void_p
    freed: list[Any] = []

    def enumerate_sources(_attributes: Any, devices: Any, count: Any) -> int:
        array = ctypes.cast(
            ole32.CoTaskMemAlloc(len(activates) * ctypes.sizeof(ctypes.c_void_p)), ctypes.POINTER(ctypes.c_void_p)
        )
        for index, activate in enumerate(activates):
            array[index] = activate.address
        devices._obj.contents = array.contents
        count._obj.value = len(activates)
        return 0

    real_loader = ctypes.WinDLL
    real_free = camera._co_task_mem_free
    fake_mf = types.SimpleNamespace(MFEnumDeviceSources=enumerate_sources)
    monkeypatch.setattr(ctypes, "WinDLL", lambda name: fake_mf if name == "mf" else real_loader(name))
    monkeypatch.setattr(camera, "_co_task_mem_free", lambda memory: (freed.append(memory), real_free(memory))[1])

    names = camera._enumerate_on_own_thread(camera._read_msmf_names)

    assert names == {0: "USB Camera", 1: "カメラ 1", 2: "Other"}
    assert all(activate.refs == 0 for activate in activates)
    assert len(freed) == 3  # two strings and the array


@windows_only
def test_an_empty_category_is_handled() -> None:
    def create(_this: Any, _category: Any, _out: Any, _flags: int) -> int:
        return 1  # S_FALSE and no enumerator: no such devices

    prototype = com_method(ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.c_ulong)
    device_enum = FakeCom({camera._CREATE_CLASS_ENUMERATOR_SLOT: (prototype, create)})

    with camera._ComPtr(device_enum.address) as owned:
        assert camera._read_category_names(owned, camera._CATEGORY_VIDEO_INPUT) == {}
    assert device_enum.refs == 0


E_FAIL = -2147467259


@windows_only
def test_a_category_that_cannot_be_listed_is_a_failure_not_an_empty_list() -> None:
    def create(_this: Any, _category: Any, _out: Any, _flags: int) -> int:
        return E_FAIL

    prototype = com_method(ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.c_ulong)
    device_enum = FakeCom({camera._CREATE_CLASS_ENUMERATOR_SLOT: (prototype, create)})

    with camera._ComPtr(device_enum.address) as owned, pytest.raises(camera._ListingFailed):
        camera._read_category_names(owned, camera._CATEGORY_VIDEO_INPUT)
    assert device_enum.refs == 0


@windows_only
def test_an_enumerator_that_fails_midway_is_a_failure_and_releases_everything() -> None:
    monikers = [fake_moniker(fake_property_bag("Camera A"))]
    pending = iter(monikers)

    def next_moniker(_this: Any, _count: int, out: Any, fetched: Any) -> int:
        moniker = next(pending, None)
        if moniker is None:
            return E_FAIL
        out[0] = moniker.address
        fetched[0] = 1
        return 0

    prototype = com_method(ctypes.c_ulong, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_ulong))
    enumerator = FakeCom({camera._ENUM_NEXT_SLOT: (prototype, next_moniker)})

    def create(_this: Any, _category: Any, out: Any, _flags: int) -> int:
        out[0] = enumerator.address
        return 0

    prototype = com_method(ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.c_ulong)
    device_enum = FakeCom({camera._CREATE_CLASS_ENUMERATOR_SLOT: (prototype, create)})

    with camera._ComPtr(device_enum.address) as owned, pytest.raises(camera._ListingFailed):
        camera._read_category_names(owned, camera._CATEGORY_VIDEO_INPUT)
    assert device_enum.refs == enumerator.refs == monikers[0].refs == 0


# ---- list_cameras -----------------------------------------------------------


def test_list_cameras_stops_after_two_misses_past_index_zero(on_os) -> None:
    on_os("windows")
    devices = FakeDevices(indices={0, 1, 3})

    cameras = list_cameras(capture_factory=devices)

    assert cameras == [
        CameraInfo(0, "カメラ 0", 640, 480),
        CameraInfo(1, "カメラ 1", 640, 480),
        CameraInfo(3, "カメラ 3", 640, 480),
    ]
    assert devices.probed == [0, 1, 2, 3, 4, 5]
    assert {capture.api for capture in devices.created} == {cv2.CAP_DSHOW}
    assert all(capture.released for capture in devices.created)


def test_list_cameras_without_cameras_gives_up_quickly(on_os) -> None:
    on_os("macos")
    devices = FakeDevices(indices=set())

    assert list_cameras(capture_factory=devices) == []
    assert devices.probed == [0, 1, 2]
    assert {capture.api for capture in devices.created} == {cv2.CAP_AVFOUNDATION}


def test_list_cameras_uses_linux_names_and_skips_virtual_cameras(
    on_os, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    on_os("linux")
    nodes = {
        "video0": "Integrated Camera: Integrated C",
        "video1": "Integrated Camera: Integrated C",  # metadata node: cannot be opened
        "video2": "Dummy video device (0x0000)",  # v4l2loopback default name
        "video4": "OBS Virtual Camera",
        "video6": "HD Pro Webcam C920",
        "video10": "Beyond max_index",
    }
    for node, name in nodes.items():
        (tmp_path / node).mkdir()
        (tmp_path / node / "name").write_text(name + "\n", encoding="utf-8")
    monkeypatch.setattr(camera, "_V4L2_SYSFS", tmp_path)
    devices = FakeDevices(indices={0, 2, 4, 6, 10})

    cameras = list_cameras(capture_factory=devices)

    assert [(info.index, info.name) for info in cameras] == [
        (0, "Integrated Camera: Integrated C"),
        (6, "HD Pro Webcam C920"),
    ]
    assert devices.probed == [0, 1, 6]  # virtual cameras are never even opened
    assert {capture.api for capture in devices.created} == {cv2.CAP_V4L2}


def test_list_cameras_does_not_probe_cameras_in_use(on_os) -> None:
    on_os("windows")
    devices = FakeDevices(indices={1})

    cameras = list_cameras(in_use={0}, capture_factory=devices)

    assert cameras == [CameraInfo(0, "カメラ 0", 0, 0), CameraInfo(1, "カメラ 1", 640, 480)]
    assert 0 not in devices.probed


def test_list_cameras_respects_the_time_budget(on_os) -> None:
    on_os("windows")
    devices = FakeDevices(indices=set(range(8)), open_delay_s=0.05)

    cameras = list_cameras(time_budget_s=0.08, capture_factory=devices)

    assert 1 <= len(devices.probed) <= 3
    assert len(cameras) == len(devices.probed)


def test_quiet_opencv_logs_is_reentrant_and_restores_the_level() -> None:
    log = cv2.utils.logging
    previous = log.getLogLevel()
    log.setLogLevel(log.LOG_LEVEL_INFO)
    try:
        with quiet_opencv_logs:
            with quiet_opencv_logs:
                assert log.getLogLevel() == log.LOG_LEVEL_SILENT
            assert log.getLogLevel() == log.LOG_LEVEL_SILENT
        assert log.getLogLevel() == log.LOG_LEVEL_INFO
    finally:
        log.setLogLevel(previous)


# ---- VideoFileSource --------------------------------------------------------


def write_video(path: Path, frames: int = 6, size: tuple[int, int] = (64, 48), fps: float = 20.0) -> Path:
    """A tiny video whose frames get brighter, so the loop point can be seen in the pixels."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), fps, size)
    assert writer.isOpened()
    for index in range(frames):
        writer.write(np.full((size[1], size[0], 3), 40 + 30 * index, np.uint8))
    writer.release()
    return path


@pytest.fixture
def clip(tmp_path: Path) -> Path:
    return write_video(tmp_path / "clip.mp4")


def test_video_file_delivers_webcam_sized_frames_with_black_bars(clip: Path) -> None:
    source = VideoFileSource(clip, 320, 120, 30)
    assert not source.is_open and source.backend is None

    source.open()
    frame = source.read()

    assert source.is_open and source.backend == "FFmpeg"
    assert (source.actual_width, source.actual_height) == (320, 120)
    assert source.actual_fps == pytest.approx(20.0, abs=0.5)
    assert frame is not None and frame.shape == (120, 320, 3) and frame.dtype == np.uint8
    assert (frame[:, :80] == 0).all() and (frame[:, 240:] == 0).all()  # 4:3 content, 160 wide, centered
    assert frame[60, 160, 0] > 20
    source.close()


def test_video_file_loops_at_the_end(clip: Path) -> None:
    source = VideoFileSource(clip, 64, 48)
    source.open()

    levels = []
    for _ in range(14):
        frame = source.read()
        assert frame is not None
        levels.append(int(frame[24, 32, 0]))
    source.close()

    assert levels[0] == levels[6] == levels[12]  # 6 frames per round
    assert levels[5] > levels[0] + 100  # brighter towards the end, then back to the start


def test_video_file_is_paced_to_its_own_frame_rate(clip: Path) -> None:
    source = VideoFileSource(clip, 64, 48, fps=30)
    source.open()

    started = time.monotonic()
    for _ in range(11):  # the first frame is due at once, then 10 intervals of 50 ms
        assert source.read() is not None
    elapsed = time.monotonic() - started
    source.close()

    assert elapsed >= 0.45
    assert elapsed < 3.0  # generous: other work may be running on this machine


def test_video_file_stall_is_not_followed_by_a_burst(clip: Path) -> None:
    source = VideoFileSource(clip, 64, 48)
    source.open()
    source.read()

    time.sleep(0.3)  # 6 frame intervals pass while the reader is busy
    source.read()
    started = time.monotonic()
    source.read()
    source.read()
    elapsed = time.monotonic() - started
    source.close()

    assert elapsed >= 0.07  # two regular intervals, no catching up


def test_video_file_without_a_usable_frame_rate_uses_the_requested_one(clip: Path) -> None:
    class NoRate:
        def __init__(self, path: str, api: int) -> None:
            self._cap = cv2.VideoCapture(path, api)

        def isOpened(self) -> bool:
            return self._cap.isOpened()

        def read(self) -> tuple[bool, np.ndarray | None]:
            return self._cap.read()

        def set(self, prop_id: int, value: float) -> bool:
            return False

        def get(self, prop_id: int) -> float:
            return 0.0

        def release(self) -> None:
            self._cap.release()

    source = VideoFileSource(clip, 64, 48, fps=25, capture_factory=NoRate)
    source.open()
    assert source.actual_fps == 25.0
    source.close()


def test_video_file_with_a_japanese_path(tmp_path: Path) -> None:
    folder = tmp_path / "動画フォルダ"
    folder.mkdir()
    source = VideoFileSource(write_video(tmp_path / "ascii.mp4").rename(folder / "面接の練習.mp4"), 64, 48)
    source.open()
    assert source.read() is not None
    source.close()


def test_missing_video_file_is_a_japanese_error(tmp_path: Path) -> None:
    with pytest.raises(CameraError, match="動画ファイルが見つかりません"):
        VideoFileSource(tmp_path / "none.mp4").open()


@pytest.mark.parametrize("content", [b"", b"this is not a video at all" * 100])
def test_unreadable_video_file_is_a_japanese_error(tmp_path: Path, content: bytes) -> None:
    path = tmp_path / "broken.mp4"
    path.write_bytes(content)
    source = VideoFileSource(path)

    with pytest.raises(CameraError, match="動画ファイルを読み込めませんでした"):
        source.open()

    assert not source.is_open and source.read() is None


def test_video_file_read_never_raises_and_recovers(clip: Path) -> None:
    class Flaky:
        """Works for one frame, then raises on read; later cannot be opened at all for a while."""

        fail = False
        opens = 0

        def __init__(self, path: str, api: int) -> None:
            Flaky.opens += 1
            if Flaky.fail:
                raise cv2.error("cannot open")
            self._reads = 0

        def isOpened(self) -> bool:
            return True

        def read(self) -> tuple[bool, np.ndarray | None]:
            self._reads += 1
            if self._reads > 1:
                raise cv2.error("decoder crashed")
            return True, np.full((48, 64, 3), 9, np.uint8)

        def set(self, prop_id: int, value: float) -> bool:
            return True

        def get(self, prop_id: int) -> float:
            return 240.0

        def release(self) -> None:
            pass

    source = VideoFileSource(clip, 64, 48, reopen_interval_s=60.0, capture_factory=Flaky)
    source.open()
    assert source.read() is not None
    assert source.read() is not None  # the decoder crashed: reopened at once

    Flaky.fail = True
    assert source.read() is None
    opens = Flaky.opens
    assert source.read() is None  # within the interval: not retried
    assert Flaky.opens == opens
    Flaky.fail = False
    source._next_reopen_at = 0.0  # the interval has passed
    assert source.read() is not None


def test_video_file_close_is_idempotent_and_read_after_close_is_none(clip: Path) -> None:
    source = VideoFileSource(clip, 64, 48)
    source.open()
    source.open()

    source.close()
    source.close()

    assert not source.is_open and source.read() is None


def test_video_file_factory_ignores_the_camera_index(clip: Path) -> None:
    source = video_file_camera_factory(clip)(index=7, width=320, height=240, fps=15)
    assert isinstance(source, VideoFileSource)
    assert (source.path, source.width, source.height, source.fps) == (clip, 320, 240, 15)
    assert isinstance(video_file_camera_factory(clip)(index=0, name="USB Camera"), VideoFileSource)


def write_audio_only_file(path: Path) -> Path:
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(8000)
        audio.writeframes(b"\0\0" * 8000)
    return path


class NoVideoTrack:
    """What Media Foundation does with an audio file: it opens at 0x0, and read() never returns."""

    opened: ClassVar[list[NoVideoTrack]] = []

    def __init__(self, path: str, api: int) -> None:
        self.api = api
        self.released = False
        NoVideoTrack.opened.append(self)

    def isOpened(self) -> bool:
        return self.api != cv2.CAP_FFMPEG

    def read(self) -> tuple[bool, np.ndarray | None]:
        raise AssertionError("read() on a file without a video track would never return")

    def get(self, prop_id: int) -> float:
        return 0.0

    def set(self, prop_id: int, value: float) -> bool:
        return False

    def release(self) -> None:
        self.released = True


def test_file_without_a_video_track_is_refused_without_reading_it(tmp_path: Path) -> None:
    NoVideoTrack.opened.clear()
    path = tmp_path / "audio.mp4"
    path.write_bytes(b"not read by the fake")
    source = VideoFileSource(path, capture_factory=NoVideoTrack)

    with pytest.raises(CameraError, match="動画ファイルを読み込めませんでした"):
        source.open()

    assert NoVideoTrack.opened and all(capture.released for capture in NoVideoTrack.opened)
    assert not source.is_open


def test_real_audio_only_file_is_refused_and_never_hangs(tmp_path: Path) -> None:
    path = write_audio_only_file(tmp_path / "音声.wav")
    outcome: list[object] = []

    def open_it() -> None:
        try:
            VideoFileSource(path).open()
            outcome.append("opened")
        except CameraError as exc:
            outcome.append(exc)

    worker = threading.Thread(target=open_it, daemon=True)
    worker.start()
    worker.join(30.0)

    assert not worker.is_alive()  # the bug was a read() inside Media Foundation that never came back
    assert isinstance(outcome[0], CameraError)


def test_the_ffmpeg_plugin_does_not_write_to_the_console(tmp_path: Path) -> None:
    garbage = tmp_path / "text.mp4"
    garbage.write_bytes(b"this is not a video at all" * 100)
    still = tmp_path / "still.png"
    assert cv2.imwrite(str(still), np.full((48, 64, 3), 90, np.uint8))
    code = (
        "import sys\n"
        "from makeup_cam.camera import CameraError, VideoFileSource\n"
        "try:\n"
        "    VideoFileSource(sys.argv[1]).open()\n"
        "except CameraError:\n"
        "    print('refused')\n"
        "source = VideoFileSource(sys.argv[2], 64, 48)\n"
        "source.open()\n"
        "print('read', [source.read() is not None for _ in range(3)])\n"
        "source.close()\n"
    )
    env = {key: value for key, value in os.environ.items() if not key.startswith("OPENCV_")}
    env["PYTHONUTF8"] = "1"
    done = subprocess.run(
        [sys.executable, "-c", code, str(garbage), str(still)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=Path(camera.__file__).parent.parent,
        env=env,
        timeout=120,
    )

    assert done.returncode == 0, done.stderr
    assert "refused" in done.stdout and "read [True, True, True]" in done.stdout
    for noise in ("moov atom", "cap_ffmpeg", "Could not find decoder", "VIDEOIO"):
        assert noise not in done.stderr


def test_close_while_a_read_waits_for_its_frame_stays_closed(clip: Path) -> None:
    source = VideoFileSource(clip, 64, 48)
    source.open()
    assert source.read() is not None
    source._next_frame_at = time.monotonic() + 30.0  # as if the file had a frame only every 30 s
    results: list[np.ndarray | None] = []
    reader = threading.Thread(target=lambda: results.append(source.read()), daemon=True)
    reader.start()
    time.sleep(0.1)

    source.close()
    reader.join(5.0)

    assert not reader.is_alive()  # close wakes the waiting read
    assert results == [None]
    assert not source.is_open  # the bug: the read reopened the file after the close
    assert source.read() is None and not source.is_open


def test_close_waits_for_a_decode_in_progress(clip: Path) -> None:
    source = VideoFileSource(clip, 64, 48, fps=30)
    source.open()
    in_decoder = threading.Event()
    let_go = threading.Event()
    real_next_frame = source._next_frame

    def slow_next_frame() -> np.ndarray | None:
        in_decoder.set()
        let_go.wait(5.0)
        return real_next_frame()

    source._next_frame = slow_next_frame  # type: ignore[method-assign]
    reader = threading.Thread(target=source.read)
    reader.start()
    assert in_decoder.wait(5.0)
    closer = threading.Thread(target=source.close)
    closer.start()
    time.sleep(0.05)
    assert closer.is_alive()  # OpenCV must not release a capture that is being read

    let_go.set()
    reader.join(5.0)
    closer.join(5.0)
    assert not source.is_open


@pytest.mark.parametrize(
    ("file_fps", "frame_count", "actual_fps", "repeat_each"),
    [
        (30.0, 90.0, 30.0, 1),
        (2.0, 6.0, 2.0, 1),  # the slowest rate delivered as it is
        (240.0, 9.0, 240.0, 1),
        (1.5, 6.0, 3.0, 2),  # real time: 1.5 fps x 2
        (1.0, 3.0, 2.0, 2),
        (0.7, 3.0, 2.1, 3),
        (0.2, 3.0, 2.0, 10),
        (0.0, 90.0, 25.0, 1),  # no usable rate: the requested one
        (-1.0, 90.0, 25.0, 1),
        (1000.0, 90.0, 25.0, 1),
        (-1.0, 1.0, 25.0, sys.maxsize),  # a still image
    ],
)
def test_video_file_pace(
    tmp_path: Path, file_fps: float, frame_count: float, actual_fps: float, repeat_each: int
) -> None:
    source = VideoFileSource(tmp_path / "unused.mp4", fps=25)
    source._set_pace(file_fps, frame_count)
    assert source.actual_fps == pytest.approx(actual_fps)
    assert source._repeat_each == repeat_each


def test_slow_video_file_repeats_its_frames_to_stay_in_real_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 20 fps counts as slow here, which keeps the test short: delivered at 40 fps, each frame twice.
    monkeypatch.setattr(camera, "_VIDEO_FILE_FPS_RANGE", (40.0, 240.0))
    source = VideoFileSource(write_video(tmp_path / "clip.mp4", frames=3, fps=20.0), 64, 48)
    source.open()
    assert source.actual_fps == pytest.approx(40.0)

    started = time.monotonic()
    frames = [source.read() for _ in range(7)]
    elapsed = time.monotonic() - started
    source.close()

    levels = [int(frame[24, 32, 0]) for frame in frames]
    assert levels[0] == levels[1] < levels[2] == levels[3] < levels[4] == levels[5]
    assert levels[6] == levels[0]  # looped
    assert frames[0] is not frames[1] and frames[0].base is not frames[1]  # every delivery is the receiver's own
    assert elapsed >= 0.14  # 6 intervals of 25 ms


def test_still_image_needs_no_reopening_and_is_not_changed_by_its_receiver(tmp_path: Path) -> None:
    path = tmp_path / "still.png"
    assert cv2.imwrite(str(path), np.full((48, 64, 3), 90, np.uint8))
    opens: list[int] = []

    def counting_factory(name: str, api: int) -> cv2.VideoCapture:
        opens.append(api)
        return cv2.VideoCapture(name, api)

    source = VideoFileSource(path, 64, 48, fps=100, capture_factory=counting_factory)
    source.open()
    opens_at_start = len(opens)

    first = source.read()
    assert first is not None
    first[:] = 0
    frames = [source.read() for _ in range(5)]
    source.close()

    assert len(opens) == opens_at_start
    assert all(frame is not None and frame[24, 32, 0] == 90 for frame in frames)


# ---- VirtualCameraOutput ----------------------------------------------------


class FakePyVirtualCam:
    """Stands in for ``pyvirtualcam.Camera``."""

    instances: ClassVar[list[FakePyVirtualCam]] = []
    fail_with: ClassVar[Exception | None] = None
    fail_send: ClassVar[Exception | None] = None

    def __init__(self, width: int, height: int, fps: float, **kwargs: object) -> None:
        if FakePyVirtualCam.fail_with is not None:
            raise FakePyVirtualCam.fail_with
        self.args = (width, height, fps)
        self.kwargs = kwargs
        self.device = "OBS Virtual Camera"
        self.sent: list[np.ndarray] = []
        self.closed = 0
        FakePyVirtualCam.instances.append(self)

    def send(self, frame: np.ndarray) -> None:
        if FakePyVirtualCam.fail_send is not None:
            raise FakePyVirtualCam.fail_send
        self.sent.append(frame)

    def close(self) -> None:
        self.closed += 1


@pytest.fixture
def fake_pyvirtualcam(monkeypatch: pytest.MonkeyPatch) -> type[FakePyVirtualCam]:
    module = types.ModuleType("pyvirtualcam")
    module.Camera = FakePyVirtualCam  # type: ignore[attr-defined]
    module.PixelFormat = types.SimpleNamespace(BGR="BGR")  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pyvirtualcam", module)
    monkeypatch.setattr(FakePyVirtualCam, "instances", [])
    monkeypatch.setattr(FakePyVirtualCam, "fail_with", None)
    monkeypatch.setattr(FakePyVirtualCam, "fail_send", None)
    return FakePyVirtualCam


def test_virtual_camera_sends_bgr_frames(fake_pyvirtualcam: type[FakePyVirtualCam]) -> None:
    output = VirtualCameraOutput(1280, 720, 30)
    frame = np.full((720, 1280, 3), 7, np.uint8)

    output.send(frame)

    backend = fake_pyvirtualcam.instances[0]
    assert backend.args == (1280, 720, 30)
    assert backend.kwargs == {"fmt": "BGR", "print_fps": False}
    assert output.device == "OBS Virtual Camera"
    assert backend.sent[0] is frame


def test_virtual_camera_fits_other_sizes_keeping_the_aspect_ratio(
    fake_pyvirtualcam: type[FakePyVirtualCam],
) -> None:
    output = VirtualCameraOutput(1280, 720, 30)

    output.send(np.full((480, 640, 3), 200, np.uint8))

    sent = fake_pyvirtualcam.instances[0].sent[0]
    assert sent.shape == (720, 1280, 3)
    assert (sent[:, 160:1120] == 200).all()  # 4:3 content scaled to 960x720, centered
    assert (sent[:, :160] == 0).all() and (sent[:, 1120:] == 0).all()


def test_virtual_camera_close_is_idempotent(fake_pyvirtualcam: type[FakePyVirtualCam]) -> None:
    output = VirtualCameraOutput(640, 480, 30)

    output.close()
    output.close()

    assert fake_pyvirtualcam.instances[0].closed == 1
    with pytest.raises(VirtualCameraError):
        output.send(np.zeros((480, 640, 3), np.uint8))


@pytest.mark.parametrize(
    ("os_name", "hints"),
    [
        ("windows", ["OBS Studio（バージョン 26 以降）", "再起動"]),
        ("macos", ["OBS Studio（バージョン 30 以降）", "仮想カメラ開始", "システム設定"]),
        ("linux", ["sudo modprobe v4l2loopback", "exclusive_caps=1"]),
    ],
)
def test_virtual_camera_failure_explains_what_to_do(
    fake_pyvirtualcam: type[FakePyVirtualCam], on_os, os_name: str, hints: list[str]
) -> None:
    on_os(os_name)
    fake_pyvirtualcam.fail_with = RuntimeError("'obs' backend: OBS Virtual Camera device not found")

    with pytest.raises(VirtualCameraError) as error:
        VirtualCameraOutput(1280, 720, 30)

    message = str(error.value)
    assert message.startswith("仮想カメラを開始できませんでした。")
    assert all(hint in message for hint in hints)
    assert "OBS Virtual Camera device not found" in message


def test_a_busy_obs_is_not_reported_as_a_missing_install(fake_pyvirtualcam: type[FakePyVirtualCam], on_os) -> None:
    on_os("windows")
    fake_pyvirtualcam.fail_with = RuntimeError("'obs' backend: virtual camera output could not be started")

    with pytest.raises(VirtualCameraError) as error:
        VirtualCameraOutput(1280, 720, 30)

    message = str(error.value)
    assert message.startswith("仮想カメラを開始できませんでした。")
    assert "使用中" in message and "仮想カメラ開始" in message
    assert "インストールしてください" not in message


@pytest.mark.parametrize(
    ("capture", "output"),
    [
        ((1280, 720), (1280, 720)),
        ((960, 540), (1280, 720)),
        ((640, 480), (1280, 720)),  # 4:3 goes in with black bars: OBS would stretch it to a 16:9 app
        ((640, 360), (1280, 720)),
        ((1920, 1080), (1920, 1080)),
        ((1600, 1200), (1920, 1080)),
        ((3840, 2160), (1920, 1080)),  # meeting apps go no higher
        ((1366, 768), (1376, 774)),
    ],
)
def test_virtual_camera_size_is_16_to_9_and_at_least_720p(capture: tuple[int, int], output: tuple[int, int]) -> None:
    width, height = camera.virtual_camera_size(*capture)
    assert (width, height) == output
    assert width * 9 == height * 16
    assert width % 2 == 0 and height % 2 == 0


def test_virtual_camera_send_failure_is_a_virtual_camera_error(
    fake_pyvirtualcam: type[FakePyVirtualCam],
) -> None:
    output = VirtualCameraOutput(64, 48, 30)
    fake_pyvirtualcam.fail_send = RuntimeError("queue closed")

    with pytest.raises(VirtualCameraError, match="送信に失敗"):
        output.send(np.zeros((48, 64, 3), np.uint8))


def test_missing_pyvirtualcam_is_reported_in_japanese(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "pyvirtualcam", None)

    with pytest.raises(VirtualCameraError, match="pip install pyvirtualcam"):
        VirtualCameraOutput(1280, 720, 30)


def test_fit_frame_letterboxes_wide_frames() -> None:
    frame = np.full((1080, 1920, 3), 90, np.uint8)

    fitted = fit_frame(frame, 640, 480)

    assert fitted.shape == (480, 640, 3)
    assert (fitted[60:420] == 90).all()  # 16:9 content is 640x360, centered vertically
    assert (fitted[:60] == 0).all() and (fitted[420:] == 0).all()
    assert fit_frame(fitted, 640, 480) is fitted
