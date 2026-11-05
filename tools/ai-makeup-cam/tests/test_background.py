"""background.py: capture, the person check, saving per resolution, clearing, the collector.

Plates are synthetic images. ``AI_MAKEUP_CAM_HOME`` points to a temporary folder, so
the real user's plate is never touched and nothing else is written.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

import cv2
import numpy as np
import pytest

from makeup_cam import background as plates
from makeup_cam import segmentation
from makeup_cam.config import user_data_dir
from makeup_cam.models import MODEL_SPECS, SELFIE_MULTICLASS

H, W = 36, 64


@pytest.fixture(autouse=True)
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    folder = tmp_path / "home"
    monkeypatch.setenv("AI_MAKEUP_CAM_HOME", str(folder))
    return folder


def room(value: int = 120) -> np.ndarray:
    frame = np.full((H, W, 3), value, np.uint8)
    frame[:, W // 2 :] = (value // 2, value, value)  # not flat, so a saved picture is recognizable
    return frame


def empty_room(_: np.ndarray) -> float:
    return 0.0


def frames(count: int = plates.CAPTURE_FRAMES) -> list[np.ndarray]:
    return [room() for _ in range(count)]


# ---------------------------------------------------------------------------
# Median
# ---------------------------------------------------------------------------


def test_median_removes_what_is_only_in_a_few_frames() -> None:
    stack = frames(7)
    stack[0][5:15, 5:15] = 0
    stack[3][5:15, 5:15] = 255
    stack[6][20:30, 40:60] = 255
    np.testing.assert_array_equal(plates.median_frame(stack), room())


def test_median_of_an_even_count_is_a_real_value() -> None:
    values = [10, 20, 30, 200]
    median = plates.median_frame([np.full((2, 2, 3), v, np.uint8) for v in values])
    assert median.dtype == np.uint8
    assert median[0, 0, 0] == 20


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------


def test_capture_of_an_empty_room() -> None:
    result = plates.capture(frames(), empty_room)
    assert result.ok and result.message == plates.SUCCESS_MESSAGE
    assert "髪型" in result.message and "顔の形" in result.message  # both effects use it
    assert result.plate is not None
    np.testing.assert_array_equal(result.plate, room())


def test_capture_rejects_a_person_with_a_japanese_message() -> None:
    result = plates.capture(frames(), lambda _: 0.25)
    assert not result.ok and result.plate is None
    assert result.person_fraction == pytest.approx(0.25)
    assert "25%" in result.message and "人" in result.message


@pytest.mark.parametrize(
    ("fraction", "accepted"),
    [
        (0.0, True),
        (plates.MAX_PERSON_FRACTION, True),
        (plates.MAX_PERSON_FRACTION + 0.001, False),
        (float("nan"), False),
    ],
)
def test_capture_threshold(fraction: float, accepted: bool) -> None:
    assert plates.capture(frames(), lambda _: fraction).ok is accepted


def test_a_person_in_a_single_checked_frame_is_enough_to_reject() -> None:
    """The median of someone walking through looks empty; the frames themselves do not."""
    stack = frames()
    stack[-1] = np.zeros((H, W, 3), np.uint8)

    def probe(frame: np.ndarray) -> float:
        return 0.5 if frame.max() == 0 else 0.0

    assert not plates.capture(stack, probe).ok


def test_a_failing_check_does_not_raise_and_does_not_accept(caplog: pytest.LogCaptureFixture) -> None:
    def broken(_: np.ndarray) -> float:
        raise RuntimeError("model broke")

    with caplog.at_level(logging.ERROR, logger="makeup_cam.background"):
        result = plates.capture(frames(), broken)
    assert not result.ok and result.message == plates.CHECK_FAILED_MESSAGE
    assert any("person check" in record.getMessage() for record in caplog.records)


@pytest.mark.parametrize("count", [0, plates.MIN_FRAMES - 1])
def test_too_few_frames_are_rejected(count: int) -> None:
    result = plates.capture(frames(count), empty_room)
    assert not result.ok and result.message == plates.TOO_FEW_FRAMES_MESSAGE


def test_frames_that_are_broken_or_of_another_size_do_not_count() -> None:
    other_size = np.zeros((H // 2, W // 2, 3), np.uint8)
    wrong_type = np.zeros((H, W, 3), np.float32)
    stack = [None, wrong_type, *frames(plates.MIN_FRAMES), other_size]
    result = plates.capture(stack, empty_room)  # type: ignore[arg-type]
    assert result.ok
    assert not plates.capture([None, wrong_type, other_size], empty_room).ok  # type: ignore[list-item]


def test_a_black_picture_is_rejected() -> None:
    result = plates.capture([np.zeros((H, W, 3), np.uint8)] * plates.CAPTURE_FRAMES, empty_room)
    assert not result.ok and result.message == plates.TOO_DARK_MESSAGE


# ---------------------------------------------------------------------------
# Saving and loading
# ---------------------------------------------------------------------------


def test_save_and_load_round_trip_exactly(home: Path) -> None:
    plate = np.random.default_rng(0).integers(0, 256, (H, W, 3), np.uint8)
    path = plates.save_plate(plate)
    assert path == home / f"background_{W}x{H}.png" == plates.plate_path(W, H)
    np.testing.assert_array_equal(plates.load_plate(W, H), plate)


def test_the_plate_is_written_only_to_the_user_data_folder(
    home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    elsewhere = tmp_path / "working-directory"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    plates.save_plate(room())
    assert user_data_dir() == home
    assert [p.name for p in home.iterdir()] == [f"background_{W}x{H}.png"]  # no temporary file left
    assert list(elsewhere.iterdir()) == []


def test_each_resolution_has_its_own_plate() -> None:
    small = cv2.resize(room(), (W // 2, H // 2), interpolation=cv2.INTER_NEAREST)
    plates.save_plate(room())
    plates.save_plate(small)
    np.testing.assert_array_equal(plates.load_plate(W, H), room())
    np.testing.assert_array_equal(plates.load_plate(W // 2, H // 2), small)
    assert plates.load_plate(W * 2, H * 2) is None


def test_load_refuses_a_file_of_another_size(home: Path) -> None:
    home.mkdir()
    ok, encoded = cv2.imencode(".png", room())
    assert ok
    plates.plate_path(W + 1, H).write_bytes(encoded.tobytes())
    assert plates.load_plate(W + 1, H) is None


@pytest.mark.parametrize("content", [b"", b"this is not a picture"])
def test_load_refuses_a_damaged_file(home: Path, content: bytes, caplog: pytest.LogCaptureFixture) -> None:
    home.mkdir()
    plates.plate_path(W, H).write_bytes(content)
    with caplog.at_level(logging.WARNING, logger="makeup_cam.background"):
        assert plates.load_plate(W, H) is None
    assert any("invalid" in record.getMessage() for record in caplog.records)


def test_a_folder_with_a_japanese_name_works(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AI_MAKEUP_CAM_HOME", str(tmp_path / "ユーザー" / "背景"))
    plates.save_plate(room())
    np.testing.assert_array_equal(plates.load_plate(W, H), room())


def test_save_rejects_anything_but_a_bgr_picture() -> None:
    with pytest.raises(ValueError):
        plates.save_plate(np.zeros((H, W), np.uint8))
    with pytest.raises(ValueError):
        plates.save_plate(np.zeros((H, W, 3), np.float32))


def test_plate_to_float() -> None:
    image = plates.plate_to_float(np.array([[[0, 51, 255]]], np.uint8))
    assert image.dtype == np.float32 and not image.flags.writeable
    np.testing.assert_allclose(image[0, 0], [0.0, 0.2, 1.0], atol=1e-6)


# ---------------------------------------------------------------------------
# Clearing
# ---------------------------------------------------------------------------


def test_clear_removes_every_resolution_and_nothing_else(home: Path) -> None:
    plates.save_plate(room())
    plates.save_plate(cv2.resize(room(), (W // 2, H // 2), interpolation=cv2.INTER_NEAREST))
    (home / f"background_{W}x{H}.png.tmp").write_bytes(b"left by a crash")
    (home / "background_notes.png").write_bytes(b"not ours")
    (home / "settings.json").write_text("{}", encoding="utf-8")
    result = plates.clear_plates()
    assert result.ok and result.message == plates.CLEARED_MESSAGE
    assert sorted(p.name for p in home.iterdir()) == ["background_notes.png", "settings.json"]
    assert plates.load_plate(W, H) is None


def test_clear_without_any_plate_is_fine(home: Path) -> None:
    assert not home.exists()
    assert plates.clear_plates().ok


def test_clear_reports_a_file_it_cannot_delete(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plates.save_plate(room())

    def refuse(self: Path, missing_ok: bool = False) -> None:
        raise PermissionError("in use")

    monkeypatch.setattr(Path, "unlink", refuse)
    result = plates.clear_plates()
    assert not result.ok
    assert str(home) in result.message and "削除" in result.message


# ---------------------------------------------------------------------------
# Collecting the frames
# ---------------------------------------------------------------------------


class Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def test_collector_waits_for_the_countdown_then_keeps_spaced_copies() -> None:
    clock = Clock()
    collector = plates.PlateCollector(countdown_s=3.0, count=3, spacing_s=0.1, clock=clock)
    frame = room()
    assert collector.seconds_left == 3
    assert collector.message == plates.COUNTDOWN_MESSAGE.format(seconds=3)
    assert collector.message == "3秒後に背景を撮影します。カメラの前から離れてください"
    collector.feed(frame)
    clock.now += 2.5
    assert collector.seconds_left == 1 and collector.frames == []
    clock.now += 0.5
    assert collector.seconds_left == 0 and collector.message == plates.CAPTURING_MESSAGE
    collector.feed(frame)
    collector.feed(frame)  # too soon after the previous one
    assert len(collector.frames) == 1 and not collector.done
    frame[:] = 0  # the kept frame must be a copy: cameras reuse their buffers
    assert collector.frames[0].max() > 0
    clock.now += 0.1
    collector.feed(frame)
    clock.now += 0.1
    collector.feed(frame)
    assert len(collector.frames) == 3 and collector.done
    clock.now += 0.1
    collector.feed(frame)
    assert len(collector.frames) == 3


def test_collector_gives_up_when_the_camera_stops_delivering() -> None:
    clock = Clock()
    collector = plates.PlateCollector(countdown_s=1.0, count=15, timeout_s=2.0, clock=clock)
    clock.now += 2.9
    assert not collector.done
    clock.now += 0.2
    assert collector.done and len(collector.frames) == 0
    result = plates.capture(collector.frames, empty_room)
    assert not result.ok and result.message == plates.TOO_FEW_FRAMES_MESSAGE


def test_the_default_capture_takes_about_a_second() -> None:
    assert plates.COUNTDOWN_SECONDS == 3.0
    assert plates.CAPTURE_FRAMES == 15
    assert 0.5 < plates.CAPTURE_FRAMES * plates.CAPTURE_SPACING_S < 1.5
    assert plates.MAX_PERSON_FRACTION == pytest.approx(0.03)


def test_messages_for_the_user_are_japanese() -> None:
    names = [name for name in vars(plates) if name.endswith("_MESSAGE")]
    assert len(names) >= 9
    for name in names:
        assert re.search(r"[ぁ-んァ-ヶ一-龠]", getattr(plates, name)), name


# ---------------------------------------------------------------------------
# The model's classes
# ---------------------------------------------------------------------------


def test_person_check_counts_people_but_not_other_things(monkeypatch: pytest.MonkeyPatch) -> None:
    """The model's "others" class also fires on chairs and such in an empty room."""
    closed: list[bool] = []
    timestamps: list[int] = []

    class FakeSegmenter:
        def __init__(self, path: Path) -> None:
            self.path = path

        def segment(self, rgb: np.ndarray, ts_ms: int) -> list[np.ndarray]:
            timestamps.append(ts_ms)
            h, w = rgb.shape[:2]
            masks = [np.zeros((h, w), np.float32) for _ in range(6)]
            masks[0][:] = 1.0  # background
            masks[0][: h // 2] = 0.0
            masks[segmentation.OTHERS][: h // 4] = 1.0  # a quarter: a chair, say
            masks[segmentation.CLOTHES][h // 4 : h // 2] = 0.3  # someone, split over two classes
            masks[segmentation.HAIR][h // 4 : h // 2] = 0.3
            return masks

        def close(self) -> None:
            closed.append(True)

    monkeypatch.setattr(plates, "MediaPipeSegmenter", FakeSegmenter)
    probe = plates.MediaPipePersonProbe(Path("model"))
    assert probe(np.zeros((40, 40, 3), np.uint8)) == pytest.approx(0.25)
    assert probe(np.zeros((40, 40, 3), np.uint8)) == pytest.approx(0.25)
    assert timestamps[0] < timestamps[1]  # VIDEO mode needs increasing timestamps
    probe.close()
    assert closed == [True]


# ---------------------------------------------------------------------------
# The real model (skipped when the models are unavailable)
# ---------------------------------------------------------------------------


def test_real_person_check(models_dir: Path) -> None:
    model = models_dir / MODEL_SPECS[SELFIE_MULTICLASS].filename
    if not model.is_file():
        pytest.skip(f"model not found in {models_dir}")
    data = pytest.importorskip("skimage.data")
    portrait = cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR)
    # An empty room: a plain wall, a darker floor line, and a soft light falloff.
    yy, xx = np.mgrid[0:720, 0:1280].astype(np.float32)
    wall = np.dstack([0.72 + 0.1 * xx / 1280, 0.76 + 0.08 * xx / 1280, 0.8 + 0.05 * yy / 720])
    wall[600:] *= 0.6
    wall = (wall * 255).astype(np.uint8)
    probe = plates.MediaPipePersonProbe(model)
    try:
        assert probe(portrait) > 3 * plates.MAX_PERSON_FRACTION
        assert probe(wall) <= plates.MAX_PERSON_FRACTION
        assert plates.capture([wall] * plates.MIN_FRAMES, probe).ok
        assert not plates.capture([portrait] * plates.MIN_FRAMES, probe).ok
    finally:
        probe.close()
    assert os.path.exists(model)  # the model is only read


def test_a_failed_save_leaves_no_picture_of_the_room_behind(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(_: object, __: object) -> None:
        raise PermissionError("the file is read-only")

    monkeypatch.setattr(plates.os, "replace", refuse)
    with pytest.raises(OSError):
        plates.save_plate(room())
    assert list(home.iterdir()) == []
