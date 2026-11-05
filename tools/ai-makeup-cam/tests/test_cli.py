"""The command line: argument parsing, each mode with fakes, and --image with the real models."""

from __future__ import annotations

import io
import json
import os
import signal
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from makeup_cam import __main__ as cli
from makeup_cam import camera, config, runner
from makeup_cam.camera import CameraInfo
from makeup_cam.config import MakeupSettings
from makeup_cam.models import ModelDownloadError
from makeup_cam.presets import find_preset
from makeup_cam.runner import RunnerStatus


@pytest.fixture(autouse=True)
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the user's own presets, session and devices out of these tests."""
    monkeypatch.setenv("AI_MAKEUP_CAM_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(camera, "_device_names", dict)


# ---- argument parsing --------------------------------------------------------


def test_no_options_means_the_window_with_nothing_overridden() -> None:
    args = cli.parse_args([])
    assert not (args.headless or args.list_cameras or args.download_models)
    assert args.image is None
    # gui.run_gui treats None as "not given" and falls back to remembered values.
    assert (args.camera, args.width, args.height, args.fps, args.preset) == (None, None, None, None, None)
    assert args.no_vcam is False
    assert args.camera_file is None


def test_camera_options_are_parsed() -> None:
    args = cli.parse_args(["--headless", "--camera", "1", "--width", "640", "--height", "480", "--fps", "15"])
    assert (args.camera, args.width, args.height, args.fps) == (1, 640, 480, 15)
    assert cli.parse_args(["--no-virtual-camera"]).no_vcam is True


def test_camera_file_works_with_the_window_and_headless() -> None:
    assert cli.parse_args(["--camera-file", "面接.flv"]).camera_file == Path("面接.flv")
    assert cli.parse_args(["--headless", "--camera-file", "a.flv"]).camera_file == Path("a.flv")


@pytest.mark.parametrize(
    "argv",
    [
        ["--out", "a.png"],
        ["--compare"],
        ["--image", "a.png"],
        ["--width", "640"],
        ["--settings", "s.json"],
        ["--headless", "--image", "a.png", "--out", "b.png"],
        ["--headless", "--preset", "natural", "--settings", "s.json"],
        ["--camera", "-1"],
        ["--fps", "0"],
        ["--width", "abc", "--height", "480"],
        ["--camera", "1", "--camera-file", "a.flv"],
        ["--camera-file", "a.flv", "--image", "a.png", "--out", "b.png"],
        ["--camera-file", "a.flv", "--list-cameras"],
        ["--camera-file", "a.flv", "--download-models"],
        ["--from-photo", "a.jpg"],
        ["--from-photo", "a.jpg", "--name", ""],
        ["--name", "x"],
        ["--bare", "b.jpg"],
        ["--from-photo", "a.jpg", "--name", "x", "--headless"],
        ["--from-photo", "a.jpg", "--name", "x", "--camera-file", "a.flv"],
        ["--video", "a.mp4"],
        ["--video", "a.mp4", "--out", "b.avi"],
        ["--video", "a.mp4", "--out", "a.mp4"],
        ["--video", "a.mp4", "--out", "b.mp4", "--headless"],
        ["--video", "a.mp4", "--out", "b.mp4", "--image", "c.png"],
        ["--video", "a.mp4", "--out", "b.mp4", "--camera-file", "c.mp4"],
        ["--video", "a.mp4", "--out", "b.mp4", "--preset", "natural", "--settings", "s.json"],
    ],
)
def test_misuse_is_a_usage_error(argv: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli.parse_args(argv)
    assert exc_info.value.code == 2
    assert capsys.readouterr().err


ENGLISH_FROM_ARGPARSE = (
    "error:",
    "usage:",
    "not allowed",
    "expected one",
    "unrecognized",
    "ambiguous",
    "ignored explicit",
    "invalid",
    "argument --",
)


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["--camera", "1", "--camera-file", "a.flv"], "--camera-file と --camera は同時に指定できません"),
        (["--headless", "--image", "a.png", "--out", "b.png"], "--image と --headless は同時に指定できません"),
        (["--preset", "a", "--settings", "b", "--headless"], "--settings と --preset は同時に指定できません"),
        (["--camera-file"], "--camera-file には値を指定してください"),
        (["--camera"], "--camera には値を指定してください"),
        (["--bogus"], "使えないオプションです: --bogus"),
        (["--cam", "1"], "--cam だけでは"),
        (["--headless=1"], "--headless には値を指定できません"),
        (["--camera", "abc"], "--camera: 整数を指定してください: abc"),
        (["--fps", "0"], "--fps: 1 以上の整数を指定してください: 0"),
        (["--width", "640"], "--width と --height は両方を指定してください"),  # our own check goes through as it is
    ],
)
def test_misuse_is_explained_in_japanese(argv: list[str], expected: str, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli.parse_args(argv)
    assert exc_info.value.code == 2
    err = capsys.readouterr().err
    assert f"エラー: {expected}" in err
    assert "使い方:" in err
    assert not [english for english in ENGLISH_FROM_ARGPARSE if english in err]


def test_help_has_no_english_headings(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli.parse_args(["--help"])
    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    assert out.startswith("使い方:")
    assert "usage:" not in out and "options:" not in out


def test_builtin_presets_are_complete() -> None:
    files = sorted(config.BUILTIN_PRESETS_DIR.glob("*.json"))
    assert files
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        assert isinstance(data.pop("display_name"), str), path.name  # not a setting: shown to the user
        # Every field is present and valid: nothing is filled in, clamped or replaced on load.
        assert MakeupSettings.from_dict(data).to_dict() == data, path.name


def test_builtin_presets_are_listed_in_the_help() -> None:
    files = config.BUILTIN_PRESETS_DIR.glob("*.json")
    assert sorted(path.stem for path in files) == sorted(cli.BUILTIN_PRESET_LABELS)


def test_the_help_calls_the_presets_what_the_window_calls_them() -> None:
    for name, label in cli.BUILTIN_PRESET_LABELS.items():
        data = json.loads((config.BUILTIN_PRESETS_DIR / f"{name}.json").read_text(encoding="utf-8"))
        assert data["display_name"] == label


# ---- modes with fakes ------------------------------------------------------------


def test_window_mode_hands_the_options_to_the_gui(monkeypatch: pytest.MonkeyPatch) -> None:
    gui = pytest.importorskip("makeup_cam.gui")
    received: list[Any] = []
    monkeypatch.setattr(gui, "run_gui", lambda args: received.append(args) or 7)

    assert cli.main(["--camera", "2", "--preset", "natural", "--no-virtual-camera"]) == 7
    assert (received[0].camera, received[0].preset, received[0].no_vcam) == (2, "natural", True)


def test_list_cameras(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    cameras = [CameraInfo(0, "USB Camera", 1280, 720), CameraInfo(2, "カメラ 2", 0, 0)]
    monkeypatch.setattr(camera, "list_cameras", lambda: cameras)

    assert cli.main(["--list-cameras"]) == cli.EXIT_OK
    out = capsys.readouterr().out
    assert "0: USB Camera（標準 1280x720）" in out
    assert "2: カメラ 2\n" in out


def test_list_cameras_without_cameras(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(camera, "list_cameras", lambda: [])
    assert cli.main(["--list-cameras"]) == cli.EXIT_ERROR
    assert "見つかりませんでした" in capsys.readouterr().out


def test_download_shows_progress(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    def fake_download(progress: Any) -> dict[str, Path]:
        progress("face_landmarker", 0, 2_000_000)
        progress("face_landmarker", 2_000_000, 2_000_000)
        progress("selfie_multiclass", 1_500_000, 0)
        return {}

    monkeypatch.setattr(cli, "ensure_all_models", fake_download)
    assert cli.main(["--download-models"]) == cli.EXIT_OK
    captured = capsys.readouterr()
    assert "ダウンロードしています" in captured.err
    assert "face_landmarker: 2.0 / 2.0 MB" in captured.err
    assert "selfie_multiclass: 1.5 MB" in captured.err
    assert "準備できています" in captured.out


def test_download_failure_is_reported_without_traceback(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def offline(progress: Any) -> dict[str, Path]:
        raise ModelDownloadError("インターネット接続を確認してください。")

    monkeypatch.setattr(cli, "ensure_all_models", offline)
    assert cli.main(["--download-models"]) == cli.EXIT_ERROR
    err = capsys.readouterr().err
    assert "エラー: インターネット接続を確認してください。" in err
    assert "Traceback" not in err


class FakePipeline:
    def __init__(self, settings: MakeupSettings) -> None:
        self.settings = settings
        self.closed = False

    def close(self) -> None:
        self.closed = True


class FakeRunner:
    instances: list[FakeRunner] = []

    def __init__(self, pipeline: FakePipeline, **kwargs: Any) -> None:
        self.pipeline = pipeline
        self.kwargs = kwargs
        self.events: list[str] = []
        FakeRunner.instances.append(self)

    def start(self) -> None:
        self.events.append("start")
        status = RunnerStatus(
            fps=29.5, camera_ok=True, virtual_camera="OBS Virtual Camera", face_found=True, processing_ms=31.0
        )
        self.kwargs["on_status"](status)

    def stop(self, timeout: float = 2.0) -> None:
        self.events.append("stop")
        self.kwargs["on_status"](RunnerStatus())  # the real runner reports idle on stop


def test_headless_runs_until_interrupted(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    FakeRunner.instances.clear()
    monkeypatch.setattr(cli, "prepare_models", lambda stream: None)
    monkeypatch.setattr(cli, "MakeupPipeline", FakePipeline)
    monkeypatch.setattr(runner, "CameraRunner", FakeRunner)
    monkeypatch.setattr(cli, "wait_for_interrupt", lambda _console: None)

    assert cli.main(["--headless", "--preset", "polished", "--camera", "1"]) == cli.EXIT_OK
    fake = FakeRunner.instances[0]
    assert fake.events == ["start", "stop"]
    expected = {"camera_index": 1, "width": 1280, "height": 720, "fps": 30, "virtual_camera": True}
    assert expected.items() <= fake.kwargs.items()
    assert fake.pipeline.settings == cli.load_preset("polished")
    assert fake.pipeline.closed
    out = capsys.readouterr().out
    assert "プリセット「polished」" in out and "Ctrl+C" in out
    assert "カメラ OK ｜ 仮想カメラ OBS Virtual Camera ｜ 顔 あり ｜ 29.5 fps ｜ 処理 31.0 ms" in out
    assert "『OBS Virtual Camera』を選んでください" in out
    assert "カメラ 接続中" not in out  # nothing is printed after stopping


@pytest.fixture
def fake_headless(monkeypatch: pytest.MonkeyPatch) -> type[FakeRunner]:
    FakeRunner.instances.clear()
    monkeypatch.setattr(cli, "prepare_models", lambda stream: None)
    monkeypatch.setattr(cli, "MakeupPipeline", FakePipeline)
    monkeypatch.setattr(runner, "CameraRunner", FakeRunner)
    monkeypatch.setattr(cli, "wait_for_interrupt", lambda _console: None)
    return FakeRunner


def write_clip(path: Path) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), 20.0, (64, 48))
    for _ in range(3):
        writer.write(np.zeros((48, 64, 3), np.uint8))
    writer.release()
    return path


def test_headless_default_camera_is_never_the_virtual_camera(
    fake_headless: type[FakeRunner], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(camera, "_device_names", lambda: {0: "OBS Virtual Camera", 1: "HD Pro Webcam C920"})

    assert cli.main(["--headless", "--preset", "natural"]) == cli.EXIT_OK

    assert fake_headless.instances[0].kwargs["camera_index"] == 1
    assert "カメラ 1「HD Pro Webcam C920」" in capsys.readouterr().out


def test_headless_with_only_a_virtual_camera_still_starts(
    fake_headless: type[FakeRunner], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(camera, "_device_names", lambda: {0: "OBS Virtual Camera"})

    assert cli.main(["--headless", "--preset", "natural"]) == cli.EXIT_OK

    # CameraSource refuses this device; the runner keeps sending its placeholder.
    assert fake_headless.instances[0].kwargs["camera_index"] == 0


def test_headless_plays_a_video_file_instead_of_a_camera(
    fake_headless: type[FakeRunner], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    clip = write_clip(tmp_path / "面接の練習.mp4")

    assert cli.main(["--headless", "--preset", "natural", "--camera-file", str(clip)]) == cli.EXIT_OK

    source = fake_headless.instances[0].kwargs["camera_factory"](index=0, width=640, height=480, fps=30)
    assert isinstance(source, camera.VideoFileSource)
    assert (source.path, source.width, source.height) == (clip, 640, 480)
    assert "動画ファイル「面接の練習.mp4」" in capsys.readouterr().out


def test_window_gets_the_video_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    gui = pytest.importorskip("makeup_cam.gui")
    clip = write_clip(tmp_path / "clip.mp4")
    received: list[Any] = []
    monkeypatch.setattr(gui, "run_gui", lambda args: received.append(args) or 0)

    assert cli.main(["--camera-file", str(clip), "--no-virtual-camera"]) == 0
    assert received[0].camera_file == clip


@pytest.mark.parametrize("content", [None, b"not a video"])
@pytest.mark.parametrize("mode", [["--headless"], []])
def test_unplayable_video_file_stops_early_with_a_friendly_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    mode: list[str],
    content: bytes | None,
) -> None:
    def must_not_run(*_args: Any) -> None:
        raise AssertionError("started despite the bad file")

    monkeypatch.setattr(cli, "prepare_models", must_not_run)
    monkeypatch.setattr(cli, "run_window", must_not_run)
    path = tmp_path / "bad.mp4"
    if content is not None:
        path.write_bytes(content)

    assert cli.main([*mode, "--camera-file", str(path)]) == cli.EXIT_ERROR
    err = capsys.readouterr().err
    assert "エラー: 動画ファイル" in err and str(path) in err
    assert "Traceback" not in err


@pytest.mark.parametrize("mode", [["--headless"], []])
def test_audio_only_file_stops_early_instead_of_freezing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], mode: list[str]
) -> None:
    def must_not_run(*_args: Any) -> None:
        raise AssertionError("started despite the bad file")

    monkeypatch.setattr(cli, "prepare_models", must_not_run)
    monkeypatch.setattr(cli, "run_window", must_not_run)
    path = tmp_path / "音声だけ.wav"
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(8000)
        audio.writeframes(b"\0\0" * 8000)

    started = time.monotonic()
    assert cli.main([*mode, "--camera-file", str(path)]) == cli.EXIT_ERROR

    assert time.monotonic() - started < cli.CAMERA_FILE_CHECK_TIMEOUT_S  # not even the safety net was needed
    err = capsys.readouterr().err
    assert "エラー: 動画ファイルを読み込めませんでした" in err and str(path) in err


class StuckVideoFile:
    """A decoder stuck inside a native call, as far as the caller can tell."""

    release = threading.Event()

    def __init__(self, path: Path) -> None:
        pass

    def open(self) -> None:
        StuckVideoFile.release.wait(30.0)

    def close(self) -> None:
        pass


@pytest.fixture
def stuck_decoder(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setattr(camera, "VideoFileSource", StuckVideoFile)
    StuckVideoFile.release.clear()
    yield
    StuckVideoFile.release.set()


def test_a_stuck_decoder_cannot_freeze_the_start(stuck_decoder: Any, tmp_path: Path) -> None:
    path = tmp_path / "stuck.mp4"

    started = time.monotonic()
    with pytest.raises(cli.CliError, match="動画ファイルを読み込めませんでした"):
        cli.check_camera_file(path, timeout_s=0.1)

    assert time.monotonic() - started < 2.0


def test_ctrl_c_gets_through_while_a_decoder_is_stuck(stuck_decoder: Any, tmp_path: Path) -> None:
    previous = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, signal.default_int_handler)
    timer = threading.Timer(0.1, signal.raise_signal, args=(signal.SIGINT,))
    timer.start()
    started = time.monotonic()
    try:
        with pytest.raises(KeyboardInterrupt):
            cli.check_camera_file(tmp_path / "stuck.mp4", timeout_s=20.0)
    finally:
        timer.join()
        signal.signal(signal.SIGINT, previous)

    assert time.monotonic() - started < 5.0


def test_headless_keeps_the_output_16_to_9_and_passes_the_device_name(
    fake_headless: type[FakeRunner], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(camera, "_device_names", lambda: {0: "OBS Virtual Camera", 1: "HD Pro Webcam C920"})

    assert cli.main(["--headless", "--preset", "natural", "--width", "640", "--height", "480"]) == cli.EXIT_OK

    kwargs = fake_headless.instances[0].kwargs
    assert (kwargs["width"], kwargs["height"]) == (640, 480)
    assert kwargs["output_size"] == (1280, 720)  # OBS would stretch 640x480 to a 16:9 meeting app
    assert (kwargs["camera_index"], kwargs["camera_name"]) == (1, "HD Pro Webcam C920")


def test_wait_for_interrupt_returns_on_ctrl_c(monkeypatch: pytest.MonkeyPatch) -> None:
    def interrupt(_seconds: float) -> None:
        raise KeyboardInterrupt

    handled: list[int] = []
    monkeypatch.setattr(cli.signal, "signal", lambda signum, _handler: handled.append(signum))
    monkeypatch.setattr(cli.time, "sleep", interrupt)
    cli.wait_for_interrupt()
    assert cli.signal.SIGTERM in handled
    if hasattr(cli.signal, "SIGBREAK"):  # Ctrl+Break, and closing the console window on Windows
        assert cli.signal.SIGBREAK in handled


def test_status_printer_repeats_slowly_and_reports_each_problem_once() -> None:
    stream = io.StringIO()
    now = [0.0]
    printer = cli.StatusPrinter(stream, virtual_camera_enabled=False, repeat_s=5.0, clock=lambda: now[0])
    ok = RunnerStatus(fps=30.0, camera_ok=True, face_found=True)

    printer(ok)
    now[0] = 1.0
    printer(ok)
    assert stream.getvalue().count("\n") == 1
    assert "仮想カメラ オフ" in stream.getvalue()

    now[0] = 6.0
    printer(ok)
    lost = RunnerStatus(camera_error="カメラ 0 を開けませんでした。")
    printer(lost)
    now[0] = 12.0
    printer(lost)
    lines = stream.getvalue().splitlines()
    assert len(lines) == 5
    assert lines[3] == "注意: カメラ 0 を開けませんでした。"
    assert lines[4].startswith("カメラ エラー")

    printer.close()
    printer(ok)
    assert len(stream.getvalue().splitlines()) == 5


# ---- still images ------------------------------------------------------------------


def test_side_by_side_keeps_both_images() -> None:
    before = np.full((90, 160, 3), 10, np.uint8)
    after = np.full((90, 160, 3), 200, np.uint8)
    combined = cli.side_by_side(before, after)
    assert combined.shape[0] == 90 and combined.shape[1] > 320
    assert np.array_equal(combined[60:, :160], before[60:])  # below the label
    assert np.array_equal(combined[60:, -160:], after[60:])


def test_images_with_japanese_paths_round_trip(tmp_path: Path) -> None:
    image = np.random.default_rng(0).integers(0, 255, (24, 32, 3), dtype=np.uint8)
    path = tmp_path / "写真" / "メイク後.png"
    cli.write_image(path, image)
    assert np.array_equal(cli.read_image(path), image)


def test_unknown_extension_is_a_friendly_error(tmp_path: Path) -> None:
    with pytest.raises(cli.CliError, match="拡張子"):
        cli.write_image(tmp_path / "out.xyz", np.zeros((4, 4, 3), np.uint8))


def test_missing_image_is_a_friendly_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["--image", str(tmp_path / "missing.png"), "--out", str(tmp_path / "o.png")]) == cli.EXIT_ERROR
    err = capsys.readouterr().err
    assert "画像を読み込めませんでした" in err
    assert "Traceback" not in err


def test_unknown_preset_lists_the_available_ones(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    argv = ["--image", "in.png", "--out", str(tmp_path / "o.png"), "--preset", "nope"]
    assert cli.main(argv) == cli.EXIT_ERROR
    err = capsys.readouterr().err
    assert "『nope』が見つかりません" in err
    assert "natural" in err and "polished" in err


def test_broken_settings_file_is_a_friendly_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    settings = tmp_path / "settings.json"
    settings.write_text("{broken", encoding="utf-8")
    argv = ["--image", "in.png", "--out", str(tmp_path / "o.png"), "--settings", str(settings)]
    assert cli.main(argv) == cli.EXIT_ERROR
    assert "設定ファイルを読み込めませんでした" in capsys.readouterr().err


def test_image_mode_renders_a_before_after_comparison(
    tmp_path: Path, face_frame_720p: np.ndarray, models_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source, target = tmp_path / "in.png", tmp_path / "out" / "compare.png"
    cv2.imwrite(str(source), face_frame_720p)

    assert cli.main(["--image", str(source), "--out", str(target), "--preset", "natural", "--compare"]) == 0
    assert "保存しました" in capsys.readouterr().out
    combined = cli.read_image(target)
    height, width = face_frame_720p.shape[:2]
    assert combined.shape[0] == height and combined.shape[1] > 2 * width
    before, after = combined[:, :width], combined[:, -width:]
    label_free = slice(100, None)  # below the BEFORE / AFTER labels
    assert np.array_equal(before[label_free], face_frame_720p[label_free])
    assert not np.array_equal(after[label_free], face_frame_720p[label_free])


# ---- video files -------------------------------------------------------------------


def test_video_options_are_parsed() -> None:
    args = cli.parse_args(["--video", "面接.flv", "--out", "結果.MP4", "--compare", "--preset", "strong"])
    assert (args.video, args.out, args.compare, args.preset) == (Path("面接.flv"), Path("結果.MP4"), True, "strong")
    assert cli.parse_args(["--video", "a.flv", "--out", "b.mp4", "--settings", "s.json"]).settings == Path("s.json")


@pytest.mark.parametrize(
    ("position_ms", "previous_ms", "expected"),
    [
        (0.0, None, 0),
        (float("nan"), None, 0),
        (33.4, 0, 33),  # the file's own time
        (100.0, 33, 100),  # a gap in the file stays a gap
        (0.0, 33, 66),  # a container without timestamps: the nominal interval
        (float("nan"), 33, 66),
        (33.0, 33, 66),  # equal is not allowed
        (10.0, 33, 66),  # neither is going back
        (33.4, 33, 34),  # later in the file, but the same millisecond once rounded
        (33.6, 33, 34),
    ],
)
def test_video_timestamps_always_increase(position_ms: float, previous_ms: int | None, expected: int) -> None:
    assert cli.next_video_timestamp(position_ms, previous_ms, 33.3) == expected


class FakeFilePipeline:
    """Records what the real pipeline would get, and draws a mark instead of makeup."""

    instances: list[FakeFilePipeline] = []

    def __init__(self) -> None:
        self.calls: list[tuple[np.ndarray, int]] = []
        FakeFilePipeline.instances.append(self)

    def process(self, frame: np.ndarray, ts_ms: int) -> Any:
        self.calls.append((frame.copy(), ts_ms))
        output = frame.copy()
        output[:, : frame.shape[1] // 2] = 255  # an unmistakable change
        return cli.ProcessResult(
            output=output,
            preview=output,
            original_preview=frame,
            face_found=len(self.calls) > 1,
            timings_ms={"total": 2.0},
        )


@pytest.fixture
def fake_file_pipeline(monkeypatch: pytest.MonkeyPatch) -> type[FakeFilePipeline]:
    @cli.contextmanager
    def fake(settings: MakeupSettings) -> Any:
        fake.settings = settings
        yield FakeFilePipeline()

    FakeFilePipeline.instances.clear()
    FakeFilePipeline.settings_seen = fake  # type: ignore[attr-defined]
    monkeypatch.setattr(cli, "prepare_models", lambda stream: None)
    monkeypatch.setattr(cli, "file_pipeline", fake)
    return FakeFilePipeline


TINY_SIZE = (320, 240)
TINY_FPS = 25.0


def write_tiny_video(path: Path, frames: int = 12, size: tuple[int, int] = TINY_SIZE, fps: float = TINY_FPS) -> Path:
    """A moving bright square on a gray ground, so the frames can be told apart."""
    writer = cv2.VideoWriter(str(path), cv2.CAP_FFMPEG, cv2.VideoWriter.fourcc(*"MJPG"), fps, size)
    assert writer.isOpened()
    for index in range(frames):
        frame = np.full((size[1], size[0], 3), 90, np.uint8)
        frame[20:60, 10 + index * 10 : 50 + index * 10] = 230
        writer.write(frame)
    writer.release()
    return path


def read_all_frames(path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


def test_video_goes_through_the_pipeline_frame_by_frame_with_its_own_timestamps(
    tmp_path: Path, fake_file_pipeline: type[FakeFilePipeline], capsys: pytest.CaptureFixture[str]
) -> None:
    clip = write_tiny_video(tmp_path / "in.avi")
    target = tmp_path / "新しいフォルダ" / "面接の結果.mp4"

    assert cli.main(["--video", str(clip), "--out", str(target), "--preset", "polished"]) == cli.EXIT_OK

    pipeline = fake_file_pipeline.instances[0]
    timestamps = [ts for _frame, ts in pipeline.calls]
    assert timestamps == [round(index * 1000 / TINY_FPS) for index in range(12)]  # 25 fps in the file: 40 ms apart
    decoded = read_all_frames(clip)
    assert all(np.array_equal(frame, want) for (frame, _ts), want in zip(pipeline.calls, decoded, strict=True))
    assert fake_file_pipeline.settings_seen.settings == cli.load_preset("polished")

    written = read_all_frames(target)
    assert len(written) == 12
    assert written[0].shape[:2] == (240, 320)
    assert written[0][:, :160].mean() > 240  # the pipeline's output, not the input
    cap = cv2.VideoCapture(str(target), cv2.CAP_FFMPEG)
    assert cap.get(cv2.CAP_PROP_FPS) == pytest.approx(TINY_FPS, abs=0.01)
    cap.release()
    out = capsys.readouterr().out
    assert "保存しました" in out and "12 フレーム" in out and "顔あり 11 フレーム" in out


def test_video_compare_puts_the_original_on_the_left(tmp_path: Path, fake_file_pipeline: Any) -> None:
    clip = write_tiny_video(tmp_path / "in.avi")
    target = tmp_path / "compare.mp4"

    assert cli.main(["--video", str(clip), "--out", str(target), "--preset", "natural", "--compare"]) == cli.EXIT_OK

    written = read_all_frames(target)
    assert len(written) == 12
    height, width = written[0].shape[:2]
    assert height == 240 and width >= 2 * 320
    below_labels = slice(120, None)
    before, after = written[5][below_labels, :320], written[5][below_labels, -320:]
    assert abs(float(before[:, :150].mean()) - 90.0) < 8.0  # untouched ground
    assert after[:, :150].mean() > 240  # the fake's mark


def test_video_uses_a_settings_file(tmp_path: Path, fake_file_pipeline: Any) -> None:
    settings = MakeupSettings()
    settings.lips.opacity = 0.123
    path = tmp_path / "s.json"
    settings.save(path)
    clip = write_tiny_video(tmp_path / "in.avi", frames=2)

    assert cli.main(["--video", str(clip), "--out", str(tmp_path / "o.mp4"), "--settings", str(path)]) == cli.EXIT_OK
    assert fake_file_pipeline.settings_seen.settings.lips.opacity == pytest.approx(0.123)


def test_video_with_a_japanese_input_name(tmp_path: Path, fake_file_pipeline: Any) -> None:
    clip = write_tiny_video(tmp_path / "面接の練習.avi", frames=3)
    target = tmp_path / "結果.mp4"
    assert cli.main(["--video", str(clip), "--out", str(target), "--preset", "natural"]) == cli.EXIT_OK
    assert len(read_all_frames(target)) == 3


def test_video_says_so_when_it_never_finds_a_face(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_file_pipeline: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    original = FakeFilePipeline.process

    def no_face(self: FakeFilePipeline, frame: np.ndarray, ts_ms: int) -> Any:
        result = original(self, frame, ts_ms)
        result.face_found = False
        return result

    monkeypatch.setattr(FakeFilePipeline, "process", no_face)
    clip = write_tiny_video(tmp_path / "in.avi", frames=3)
    assert cli.main(["--video", str(clip), "--out", str(tmp_path / "o.mp4"), "--preset", "natural"]) == cli.EXIT_OK
    assert "顔が見つからなかった" in capsys.readouterr().err


@pytest.mark.parametrize("content", [None, b"not a video"])
def test_unplayable_video_stops_before_the_models_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], content: bytes | None
) -> None:
    def must_not_run(*_args: Any) -> None:
        raise AssertionError("started despite the bad file")

    monkeypatch.setattr(cli, "prepare_models", must_not_run)
    path = tmp_path / "bad.mp4"
    if content is not None:
        path.write_bytes(content)
    target = tmp_path / "o.mp4"

    assert cli.main(["--video", str(path), "--out", str(target), "--preset", "natural"]) == cli.EXIT_ERROR

    err = capsys.readouterr().err
    assert "エラー: 動画ファイル" in err and str(path) in err and "Traceback" not in err
    assert not target.exists()


def test_video_unwritable_target_is_a_friendly_error(
    tmp_path: Path, fake_file_pipeline: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    clip = write_tiny_video(tmp_path / "in.avi", frames=2)
    blocker = tmp_path / "file"
    blocker.write_bytes(b"")  # a file where the folder should be
    target = blocker / "o.mp4"

    assert cli.main(["--video", str(clip), "--out", str(target), "--preset", "natural"]) == cli.EXIT_ERROR

    err = capsys.readouterr().err
    assert "動画を保存できませんでした" in err and "Traceback" not in err


def test_video_stopped_midway_leaves_what_was_written_playable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_file_pipeline: Any
) -> None:
    original = FakeFilePipeline.process

    def interrupted(self: FakeFilePipeline, frame: np.ndarray, ts_ms: int) -> Any:
        if len(self.calls) == 4:
            raise KeyboardInterrupt
        return original(self, frame, ts_ms)

    monkeypatch.setattr(FakeFilePipeline, "process", interrupted)
    clip = write_tiny_video(tmp_path / "in.avi")
    target = tmp_path / "o.mp4"

    assert cli.main(["--video", str(clip), "--out", str(target), "--preset", "natural"]) == cli.EXIT_INTERRUPTED
    assert len(read_all_frames(target)) == 4


def moving_face_clip(path: Path, frame: np.ndarray, frames: int = 12) -> Path:
    """The test portrait sliding sideways, at half size, stored without loss so it can be compared."""
    small = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
    writer = cv2.VideoWriter(str(path), cv2.CAP_FFMPEG, cv2.VideoWriter.fourcc(*"FFV1"), 30.0, (640, 360))
    assert writer.isOpened()
    for index in range(frames):
        writer.write(np.roll(small, index, axis=1))
    writer.release()
    return path


def test_video_mode_with_the_real_models(
    tmp_path: Path, face_frame_720p: np.ndarray, models_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    clip = moving_face_clip(tmp_path / "in.avi", face_frame_720p, frames=20)
    target = tmp_path / "compare.mp4"

    assert cli.main(["--video", str(clip), "--out", str(target), "--preset", "polished", "--compare"]) == 0

    assert "顔あり 20 フレーム" in capsys.readouterr().out
    written = read_all_frames(target)
    assert len(written) == 20
    last = written[-1]
    width = (last.shape[1] - 2) // 2
    before, after = last[60:, :width].astype(np.float32), last[60:, -width:].astype(np.float32)
    assert np.abs(after - before).mean() > 0.5  # the makeup is there at the end of the fade-in


def test_video_mode_with_every_effect_off_keeps_the_picture(
    tmp_path: Path, face_frame_720p: np.ndarray, models_dir: Path
) -> None:
    clip = moving_face_clip(tmp_path / "in.avi", face_frame_720p, frames=6)
    seen: list[tuple[np.ndarray, np.ndarray]] = []

    summary = cli.process_video(
        clip, cli.load_preset("off"), lambda frame, result, _fps: seen.append((frame.copy(), result.output.copy()))
    )

    assert summary.frames == 6 and len(seen) == 6
    assert all(np.array_equal(frame, output) for frame, output in seen)  # off.json is an exact pass-through


# ---- status wording --------------------------------------------------------------


def test_the_status_line_says_whether_there_is_a_face() -> None:
    found = cli.format_status(RunnerStatus(face_found=True), virtual_camera_enabled=False)
    missing = cli.format_status(RunnerStatus(face_found=False), virtual_camera_enabled=False)
    assert "顔 あり" in found and "検出中" not in found
    assert "顔 なし" in missing and "検出中" not in missing


# ---- --from-photo ------------------------------------------------------------------


class FakeAnalysis:
    def __init__(self, confidence: dict[str, float], notes: list[str] | None = None) -> None:
        self.settings = MakeupSettings()
        self.confidence = confidence
        self.notes = notes or []


@pytest.fixture
def fake_photo_tools(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    from makeup_cam import analyze

    seen: dict[str, Any] = {"result": FakeAnalysis({"lips": 0.85, "blush": 0.0}, ["前髪で眉が隠れています"])}

    def fake_analyze(makeup: np.ndarray, bare: np.ndarray | None, *, base: MakeupSettings) -> FakeAnalysis:
        seen.update(makeup=makeup, bare=bare, base=base)
        return seen["result"]

    monkeypatch.setattr(cli, "prepare_models", lambda stream: None)
    monkeypatch.setattr(analyze, "analyze_makeup", fake_analyze)
    return seen


def write_photo(path: Path, value: int = 128) -> Path:
    assert cv2.imwrite(str(path), np.full((8, 8, 3), value, np.uint8))
    return path


def test_from_photo_saves_a_preset_and_reports_what_could_not_be_measured(
    tmp_path: Path, fake_photo_tools: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    makeup, bare = write_photo(tmp_path / "m.png", 10), write_photo(tmp_path / "b.png", 200)

    assert cli.main(["--from-photo", str(makeup), "--bare", str(bare), "--name", "いつものメイク"]) == cli.EXIT_OK

    assert fake_photo_tools["makeup"][0, 0, 0] == 10 and fake_photo_tools["bare"][0, 0, 0] == 200
    assert fake_photo_tools["base"] == cli.load_preset("natural")
    captured = capsys.readouterr()
    assert "リップ: 0.85" in captured.out
    assert "チーク: 測れなかった（natural の値のまま）" in captured.out
    assert "立体感: 測れなかった" in captured.out  # not in the result at all: not measured either
    assert "前髪で眉が隠れています" in captured.out
    assert "--bare" not in captured.err
    assert cli.load_preset("いつものメイク") == MakeupSettings()
    assert sorted(path.name for path in tmp_path.rglob("*.png")) == ["b.png", "m.png"]  # the photos are not kept


def test_from_photo_without_a_bare_photo_urges_to_add_one(
    tmp_path: Path, fake_photo_tools: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["--from-photo", str(write_photo(tmp_path / "m.png")), "--name", "a"]) == cli.EXIT_OK
    assert fake_photo_tools["bare"] is None
    err = capsys.readouterr().err
    assert "--bare" in err and "アイシャドウをほとんど測れません" in err


def test_from_photo_starts_from_the_given_preset(tmp_path: Path, fake_photo_tools: dict[str, Any]) -> None:
    argv = ["--from-photo", str(write_photo(tmp_path / "m.png")), "--name", "a", "--preset", "polished"]
    assert cli.main(argv) == cli.EXIT_OK
    assert fake_photo_tools["base"] == cli.load_preset("polished")


def test_from_photo_saves_nothing_when_nothing_could_be_measured(
    tmp_path: Path, fake_photo_tools: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    fake_photo_tools["result"] = FakeAnalysis({"lips": 0.0}, ["白黒の写真です"])

    assert cli.main(["--from-photo", str(write_photo(tmp_path / "m.png")), "--name", "a"]) == cli.EXIT_ERROR

    captured = capsys.readouterr()
    assert "白黒の写真です" in captured.out
    assert "プリセットは保存していません" in captured.err
    assert find_preset("a") is None


def test_from_photo_refuses_a_builtin_name_with_the_message_as_it_is(
    tmp_path: Path, fake_photo_tools: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["--from-photo", str(write_photo(tmp_path / "m.png")), "--name", "natural"]) == cli.EXIT_ERROR
    assert "『natural』は標準プリセットの名前なので使えません" in capsys.readouterr().err


def test_from_photo_explains_an_unreadable_photo(
    tmp_path: Path, fake_photo_tools: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "broken.jpg").write_bytes(b"not a photo")
    assert cli.main(["--from-photo", str(tmp_path / "broken.jpg"), "--name", "a"]) == cli.EXIT_ERROR
    err = capsys.readouterr().err
    assert "写真を読み込めませんでした" in err and "Traceback" not in err


def test_from_photo_explains_a_photo_without_a_face(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from makeup_cam import analyze

    def no_face(*_args: Any, **_kwargs: Any) -> None:
        raise analyze.AnalysisError("顔が見つかりませんでした。")

    monkeypatch.setattr(cli, "prepare_models", lambda stream: None)
    monkeypatch.setattr(analyze, "analyze_makeup", no_face)
    assert cli.main(["--from-photo", str(write_photo(tmp_path / "m.png")), "--name", "a"]) == cli.EXIT_ERROR
    assert "エラー: 顔が見つかりませんでした。" in capsys.readouterr().err


def test_from_photo_with_the_real_models_end_to_end(
    tmp_path: Path, face_frame_720p: np.ndarray, models_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    photo = tmp_path / "face.png"
    assert cv2.imwrite(str(photo), face_frame_720p)

    code = cli.main(["--from-photo", str(photo), "--bare", str(photo), "--name", "同じ写真2枚"])

    assert code in (cli.EXIT_OK, cli.EXIT_ERROR)  # the same photo twice may measure next to nothing
    captured = capsys.readouterr()
    assert "項目ごとの信頼度" in captured.out and "Traceback" not in captured.err


# ---- stopping, in a real process with its own console ---------------------------------


_APP_IN_ITS_CONSOLE = r"""
import ctypes, sys
ctypes.windll.kernel32.SetConsoleCtrlHandler(None, False)  # as in a console opened by hand
from makeup_cam import __main__ as cli, camera, gui, pipeline, runner
from makeup_cam.pipeline import ProcessResult

log = open(sys.argv[1], "a", encoding="utf-8", buffering=1)
mode, clip = sys.argv[2], sys.argv[3]


class Vcam:
    device = "Fake Virtual Camera"
    sent = 0

    def __init__(self, *, width, height, fps):
        log.write("vcam opened\n")

    def send(self, frame):
        self.sent += 1
        if self.sent == 5:
            log.write("frames flowing\n")

    def close(self):
        log.write("vcam closed\n")


class Pipeline:
    bypass = False
    has_background = False

    def __init__(self, settings):
        pass

    def set_settings(self, settings):
        pass

    def process(self, frame, ts):
        return ProcessResult(output=frame, preview=frame, original_preview=frame, face_found=True, timings_ms={})

    def close(self):
        log.write("pipeline closed\n")


original_init = runner.CameraRunner.__init__


def init(self, *args, **kwargs):
    kwargs["vcam_factory"] = Vcam
    original_init(self, *args, **kwargs)


runner.CameraRunner.__init__ = init
cli.MakeupPipeline = Pipeline
pipeline.MakeupPipeline = Pipeline
cli.prepare_models = lambda stream: None
gui.prepare_models = lambda: None
code = cli.main(["--camera-file", clip] + (["--headless"] if mode == "headless" else []))
log.write(f"main returned {code}\n")
sys.exit(code)
"""

_SEND_INTO_THE_CONSOLE = r"""
import ctypes, sys, time
kernel32 = ctypes.windll.kernel32
kernel32.FreeConsole()
if not kernel32.AttachConsole(int(sys.argv[1])):
    sys.exit(3)
handler = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_uint)(lambda _event: 1)  # this helper stays out of it
kernel32.SetConsoleCtrlHandler(handler, True)
if sys.argv[2] == "close":
    window = kernel32.GetConsoleWindow()
    if not window:
        sys.exit(4)
    ctypes.windll.user32.PostMessageW(window, 0x0010, 0, 0)  # WM_CLOSE
else:
    kernel32.GenerateConsoleCtrlEvent({"ctrl_c": 0, "ctrl_break": 1}[sys.argv[2]], 0)
    time.sleep(0.2)
kernel32.FreeConsole()
"""

STATUS_CONTROL_C_EXIT = 0xC000013A


@pytest.mark.skipif(sys.platform != "win32", reason="Windows consoles")
@pytest.mark.parametrize(
    ("mode", "how", "expected_code"),
    [
        ("headless", "ctrl_c", 0),
        ("headless", "ctrl_break", 0),
        ("headless", "close", STATUS_CONTROL_C_EXIT),  # Windows reports this for a closed console, whatever one does
        ("window", "ctrl_c", 0),
        ("window", "close", STATUS_CONTROL_C_EXIT),
    ],
)
def test_the_app_stops_cleanly_in_a_real_console(tmp_path: Path, mode: str, how: str, expected_code: int) -> None:
    clip = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (64, 48))
    for _ in range(10):
        writer.write(np.zeros((48, 64, 3), np.uint8))
    writer.release()
    log = tmp_path / "log.txt"
    log.write_text("", encoding="utf-8")
    env = {
        **os.environ,
        "QT_QPA_PLATFORM": "offscreen",
        "AI_MAKEUP_CAM_HOME": str(tmp_path / "home"),
        "PYTHONUTF8": "1",
    }
    startup = subprocess.STARTUPINFO()
    startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW  # SW_HIDE: no window pops up during the tests
    app = subprocess.Popen(
        [sys.executable, "-c", _APP_IN_ITS_CONSOLE, str(log), mode, str(clip)],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        creationflags=subprocess.CREATE_NEW_CONSOLE,
        startupinfo=startup,
    )
    try:
        deadline = time.monotonic() + 60.0
        while "frames flowing" not in log.read_text(encoding="utf-8"):
            assert app.poll() is None, log.read_text(encoding="utf-8")
            assert time.monotonic() < deadline
            time.sleep(0.02)
        time.sleep(0.3)  # past the signal handlers being installed
        sent = subprocess.run(
            [sys.executable, "-c", _SEND_INTO_THE_CONSOLE, str(app.pid), how], creationflags=subprocess.CREATE_NO_WINDOW
        )
        if sent.returncode in (3, 4):
            pytest.skip("cannot reach the console of a child process here")
        assert sent.returncode == 0
        code = app.wait(timeout=30.0) & 0xFFFFFFFF
    finally:
        app.kill()
    lines = log.read_text(encoding="utf-8").splitlines()
    assert code == expected_code
    assert "vcam closed" in lines and "pipeline closed" in lines
