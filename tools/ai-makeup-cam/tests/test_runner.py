"""CameraRunner with a fake camera, fake virtual camera and fake pipeline (no hardware)."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator

import numpy as np
import pytest

from makeup_cam.camera import CameraError, VirtualCameraError
from makeup_cam.config import MakeupSettings
from makeup_cam.pipeline import ProcessResult
from makeup_cam.runner import (
    CAMERA_LOST_MESSAGE,
    CAMERA_READ_FAILED_MESSAGE,
    PROCESSING_FAILED_MESSAGE,
    CameraRunner,
    RunnerStatus,
    placeholder_frame,
)

WIDTH, HEIGHT, FPS = 64, 48, 50
CAMERA_OPEN_ERROR = "カメラ 0 を開けませんでした。（テスト）"
VCAM_OPEN_ERROR = "仮想カメラを開始できませんでした。（テスト）"


def wait_until(condition: Callable[[], bool], timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.005)
    return condition()


def camera_value(index: int) -> int:
    """Pixel value of every frame of fake camera ``index``."""
    return 40 + 10 * index


# ---- fakes -----------------------------------------------------------------


class FakeCamera:
    def __init__(self, hub: CameraHub, index: int, width: int, height: int, fps: int) -> None:
        self.hub = hub
        self.index = index
        self.size = (width, height)
        self.fps = fps
        self.opened = False
        self.closed = False
        self.reads = 0

    def open(self) -> None:
        if self.hub.open_fails.is_set():
            raise CameraError(CAMERA_OPEN_ERROR)
        self.opened = True

    def read(self) -> np.ndarray | None:
        time.sleep(0.004)  # a camera delivers frames at its own pace
        if self.hub.read_raises.is_set():
            raise RuntimeError("driver crashed")
        if self.hub.outage.is_set():
            return None
        self.reads += 1
        width, height = self.size
        return np.full((height, width, 3), camera_value(self.index), np.uint8)

    def close(self) -> None:
        self.closed = True


class CameraHub:
    """``camera_factory`` for the runner; scripts failures shared by all its cameras."""

    def __init__(self) -> None:
        self.open_fails = threading.Event()  # set: open() raises
        self.outage = threading.Event()  # set: read() returns None
        self.read_raises = threading.Event()  # set: read() raises
        self.cameras: list[FakeCamera] = []

    def __call__(self, *, index: int, width: int, height: int, fps: int) -> FakeCamera:
        camera = FakeCamera(self, index, width, height, fps)
        self.cameras.append(camera)
        return camera

    @property
    def opened(self) -> list[FakeCamera]:
        return [camera for camera in self.cameras if camera.opened]


class FakeVcam:
    device = "Fake Virtual Camera"

    def __init__(self, hub: VcamHub, width: int, height: int, fps: int) -> None:
        self.hub = hub
        self.size = (width, height)
        self.fps = fps
        self.frames: list[np.ndarray] = []
        self.closed = False

    def send(self, frame_bgr: np.ndarray) -> None:
        if self.hub.send_fails.is_set():
            raise RuntimeError("pipe broken")
        self.frames.append(frame_bgr)
        self.hub.sent.append(frame_bgr)

    def close(self) -> None:
        self.closed = True


class VcamHub:
    """``vcam_factory`` for the runner."""

    def __init__(self) -> None:
        self.open_fails = threading.Event()
        self.send_fails = threading.Event()
        self.attempts = 0
        self.vcams: list[FakeVcam] = []
        self.sent: list[np.ndarray] = []

    def __call__(self, *, width: int, height: int, fps: int) -> FakeVcam:
        self.attempts += 1
        if self.open_fails.is_set():
            raise VirtualCameraError(VCAM_OPEN_ERROR)
        vcam = FakeVcam(self, width, height, fps)
        self.vcams.append(vcam)
        return vcam


class FakePipeline:
    """Inverts frames, so processed output is distinguishable from raw frames."""

    def __init__(self, delay_s: float = 0.0) -> None:
        self.delay_s = delay_s
        self.fails = threading.Event()
        self.calls = 0
        self.settings: MakeupSettings | None = None
        self.bypass = False

    def process(self, frame_bgr: np.ndarray, timestamp_ms: int | None = None) -> ProcessResult:
        assert timestamp_ms is not None
        self.calls += 1
        time.sleep(self.delay_s)
        if self.fails.is_set():
            raise RuntimeError("effect exploded")
        output = 255 - frame_bgr
        return ProcessResult(
            output=output, preview=output, original_preview=frame_bgr, face_found=True, timings_ms={"total": 1.0}
        )

    def set_settings(self, settings: MakeupSettings) -> None:
        self.settings = settings


class Harness:
    """A runner wired to fakes, recording callbacks."""

    def __init__(self, *, pipeline: FakePipeline | None = None, **kwargs) -> None:
        self.pipeline = pipeline or FakePipeline()
        self.cameras = CameraHub()
        self.vcams = VcamHub()
        self.results: list[ProcessResult] = []
        self.statuses: list[RunnerStatus] = []
        options = dict(
            width=WIDTH,
            height=HEIGHT,
            fps=FPS,
            on_frame=self.results.append,
            on_status=self.statuses.append,
            camera_factory=self.cameras,
            vcam_factory=self.vcams,
            retry_interval_s=0.05,
            camera_timeout_s=0.15,
        )
        options.update(kwargs)
        self.runner = CameraRunner(self.pipeline, **options)

    def sent_since(self, start: int) -> list[np.ndarray]:
        return self.vcams.sent[start:]

    def wait_for_sent(self, predicate: Callable[[np.ndarray], bool], count: int = 1) -> bool:
        start = len(self.vcams.sent)
        return wait_until(lambda: sum(map(predicate, self.sent_since(start))) >= count)


def processed(index: int = 0) -> Callable[[np.ndarray], bool]:
    return lambda frame: frame[0, 0, 0] == 255 - camera_value(index)


def raw(index: int = 0) -> Callable[[np.ndarray], bool]:
    return lambda frame: frame[0, 0, 0] == camera_value(index)


def is_placeholder(frame: np.ndarray) -> bool:
    return np.array_equal(frame, placeholder_frame(frame.shape[1], frame.shape[0]))


@pytest.fixture
def make_harness() -> Iterator[Callable[..., Harness]]:
    harnesses: list[Harness] = []

    def make(**kwargs) -> Harness:
        harness = Harness(**kwargs)
        harnesses.append(harness)
        return harness

    yield make
    for harness in harnesses:
        harness.runner.stop()


# ---- tests -----------------------------------------------------------------


def test_frames_flow_to_virtual_camera_and_on_frame(make_harness) -> None:
    h = make_harness()
    h.runner.start()

    assert h.wait_for_sent(processed(), count=5)
    assert wait_until(lambda: len(h.results) >= 5)
    assert all(isinstance(result, ProcessResult) for result in h.results)
    assert wait_until(lambda: h.runner.status.fps > 0)

    status = h.runner.status
    assert status.camera_ok and status.camera_error is None
    assert status.virtual_camera == "Fake Virtual Camera" and status.virtual_camera_error is None
    assert status.face_found and status.processing_ms > 0 and status.processing_error is None
    assert h.statuses and h.statuses[-1].camera_ok
    camera = h.cameras.cameras[0]
    assert (camera.index, camera.size, camera.fps) == (0, (WIDTH, HEIGHT), FPS)
    assert h.vcams.vcams[0].size == (WIDTH, HEIGHT)


def test_slow_processing_uses_the_newest_frame(make_harness) -> None:
    h = make_harness(pipeline=FakePipeline(delay_s=0.03))
    h.runner.start()

    assert wait_until(lambda: h.pipeline.calls >= 5)

    camera = h.cameras.cameras[0]
    assert camera.reads > 2 * h.pipeline.calls  # stale frames were dropped, not queued


def test_camera_outage_sends_placeholder_then_recovers(make_harness) -> None:
    h = make_harness()
    h.runner.start()
    assert h.wait_for_sent(processed())

    h.cameras.outage.set()

    assert h.wait_for_sent(is_placeholder, count=5)  # the call never shows a frozen picture
    assert wait_until(lambda: h.runner.status.camera_error == CAMERA_LOST_MESSAGE)
    status = h.runner.status
    assert not status.camera_ok and not status.face_found and status.fps == 0.0
    assert is_placeholder(h.results[-1].preview)
    start = len(h.vcams.sent)
    time.sleep(0.2)
    assert len(h.sent_since(start)) >= 4  # still fed at roughly the target rate (50 fps)

    h.cameras.outage.clear()

    assert h.wait_for_sent(processed())
    assert wait_until(lambda: h.runner.status.camera_ok)
    assert h.runner.status.camera_error is None


def test_camera_open_failures_are_retried(make_harness) -> None:
    h = make_harness()
    h.cameras.open_fails.set()
    h.runner.start()

    assert wait_until(lambda: h.runner.status.camera_error == CAMERA_OPEN_ERROR)
    assert h.wait_for_sent(is_placeholder)
    assert wait_until(lambda: len(h.cameras.cameras) >= 3)  # retried every retry_interval_s

    h.cameras.open_fails.clear()

    assert h.wait_for_sent(processed())
    assert len(h.cameras.opened) == 1
    assert all(camera.closed for camera in h.cameras.cameras[:-1])
    assert wait_until(lambda: h.runner.status.camera_ok and h.runner.status.camera_error is None)


def test_camera_that_raises_is_recreated(make_harness) -> None:
    h = make_harness()
    h.runner.start()
    assert h.wait_for_sent(processed())

    h.cameras.read_raises.set()

    assert wait_until(lambda: h.runner.status.camera_error == CAMERA_READ_FAILED_MESSAGE)
    assert h.cameras.cameras[0].closed
    h.cameras.read_raises.clear()
    assert h.wait_for_sent(processed())
    assert len(h.cameras.cameras) >= 2


def test_virtual_camera_failure_is_reported_and_processing_continues(make_harness) -> None:
    h = make_harness()
    h.vcams.open_fails.set()
    h.runner.start()

    assert wait_until(lambda: h.runner.status.virtual_camera_error == VCAM_OPEN_ERROR)
    assert wait_until(lambda: len(h.results) >= 5)
    assert wait_until(lambda: h.vcams.attempts >= 3)  # retried
    assert h.runner.status.virtual_camera is None and h.runner.status.camera_ok

    h.vcams.open_fails.clear()

    assert h.wait_for_sent(processed())
    assert wait_until(lambda: h.runner.status.virtual_camera == "Fake Virtual Camera")
    assert h.runner.status.virtual_camera_error is None


def test_virtual_camera_send_failure_reopens_it(make_harness) -> None:
    h = make_harness()
    h.runner.start()
    assert h.wait_for_sent(processed())

    h.vcams.send_fails.set()

    assert wait_until(lambda: "送信に失敗" in (h.runner.status.virtual_camera_error or ""))
    assert h.vcams.vcams[0].closed
    h.vcams.send_fails.clear()
    assert wait_until(lambda: len(h.vcams.vcams) >= 2 and len(h.vcams.vcams[-1].frames) > 0)
    assert wait_until(lambda: h.runner.status.virtual_camera_error is None)


def test_set_virtual_camera_toggles_the_output(make_harness) -> None:
    h = make_harness(virtual_camera=False)
    h.runner.start()
    assert wait_until(lambda: len(h.results) >= 3)
    assert h.vcams.attempts == 0 and h.runner.status.virtual_camera is None

    h.runner.set_virtual_camera(True)
    assert h.wait_for_sent(processed())
    assert wait_until(lambda: h.runner.status.virtual_camera == "Fake Virtual Camera")

    h.runner.set_virtual_camera(False)
    assert wait_until(lambda: h.vcams.vcams[0].closed and h.runner.status.virtual_camera is None)
    count = len(h.results)
    assert wait_until(lambda: len(h.results) > count + 3)  # preview keeps running
    assert len(h.vcams.vcams) == 1


def test_set_camera_switches_device_and_output_size(make_harness) -> None:
    h = make_harness()
    h.runner.start()
    assert h.wait_for_sent(processed(0))

    h.runner.set_camera(1, 32, 24)

    assert h.wait_for_sent(processed(1))
    first, second = h.cameras.opened[:2]
    assert first.closed and (second.index, second.size) == (1, (32, 24))
    assert h.runner.camera_index == 1
    assert h.vcams.vcams[0].closed and h.vcams.vcams[-1].size == (32, 24)

    vcam_count = len(h.vcams.vcams)
    h.runner.set_camera(1)  # same camera and size: reconnect the camera only

    assert wait_until(lambda: len(h.cameras.opened) == 3)
    assert h.wait_for_sent(processed(1))
    assert len(h.vcams.vcams) == vcam_count


def test_pipeline_exception_does_not_kill_the_loop(make_harness) -> None:
    h = make_harness()
    h.runner.start()
    assert h.wait_for_sent(processed())

    h.pipeline.fails.set()

    assert h.wait_for_sent(raw(), count=3)  # raw frames keep the call alive
    assert wait_until(lambda: h.runner.status.processing_error == PROCESSING_FAILED_MESSAGE)
    h.pipeline.fails.clear()
    assert h.wait_for_sent(processed())
    assert wait_until(lambda: h.runner.status.processing_error is None)


def test_callback_exceptions_do_not_kill_the_loop(make_harness) -> None:
    def explode(_: object) -> None:
        raise RuntimeError("GUI bug")

    h = make_harness(on_frame=explode, on_status=explode)
    h.runner.start()

    assert h.wait_for_sent(processed(), count=10)


def test_settings_and_bypass_are_forwarded_to_the_pipeline(make_harness) -> None:
    h = make_harness()
    settings = MakeupSettings()

    h.runner.set_settings(settings)
    h.runner.set_bypass(True)

    assert h.pipeline.settings is settings
    assert h.pipeline.bypass is True


def test_start_and_stop_are_idempotent_and_timely(make_harness) -> None:
    h = make_harness()
    h.runner.stop()  # never started: no-op

    h.runner.start()
    h.runner.start()
    assert h.wait_for_sent(processed())
    assert len(h.cameras.cameras) == 1 and h.runner.is_running

    started = time.monotonic()
    h.runner.stop()
    assert time.monotonic() - started < 1.0
    h.runner.stop()

    assert not h.runner.is_running
    assert h.cameras.cameras[0].closed and h.vcams.vcams[0].closed
    assert h.runner.status == RunnerStatus()
    assert h.statuses[-1] == RunnerStatus()
    sent = len(h.vcams.sent)
    time.sleep(0.05)
    assert len(h.vcams.sent) == sent

    h.runner.start()  # restartable
    assert h.wait_for_sent(processed())


def test_placeholder_frame_is_cached_and_read_only() -> None:
    frame = placeholder_frame(320, 180)

    assert frame.shape == (180, 320, 3) and frame.dtype == np.uint8
    assert not frame.flags.writeable
    assert placeholder_frame(320, 180) is frame
    assert frame.max() > frame.min()  # something is drawn on it
