"""Camera input / virtual camera output, with fake OpenCV captures and a fake pyvirtualcam."""

from __future__ import annotations

import collections
import sys
import time
import types
from pathlib import Path
from typing import ClassVar

import cv2
import numpy as np
import pytest

from makeup_cam import camera
from makeup_cam.camera import (
    REOPEN_AFTER_FAILURES,
    CameraError,
    CameraInfo,
    CameraSource,
    VirtualCameraError,
    VirtualCameraOutput,
    fit_frame,
    list_cameras,
    quiet_opencv_logs,
)

MJPG = cv2.VideoWriter.fourcc(*"MJPG")


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
