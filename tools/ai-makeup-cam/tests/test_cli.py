"""The command line: argument parsing, each mode with fakes, and --image with the real models."""

from __future__ import annotations

import io
import json
import signal
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


def test_builtin_presets_are_complete_and_listed_in_the_help() -> None:
    files = sorted(config.BUILTIN_PRESETS_DIR.glob("*.json"))
    assert sorted(path.stem for path in files) == sorted(cli.BUILTIN_PRESET_LABELS)
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        # Every field is present and valid: nothing is filled in, clamped or replaced on load.
        assert MakeupSettings.from_dict(data).to_dict() == data, path.name


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
    monkeypatch.setattr(cli, "wait_for_interrupt", lambda: None)

    assert cli.main(["--headless", "--preset", "polished", "--camera", "1"]) == cli.EXIT_OK
    fake = FakeRunner.instances[0]
    assert fake.events == ["start", "stop"]
    expected = {"camera_index": 1, "width": 1280, "height": 720, "fps": 30, "virtual_camera": True}
    assert expected.items() <= fake.kwargs.items()
    assert fake.pipeline.settings == cli.load_preset("polished")
    assert fake.pipeline.closed
    out = capsys.readouterr().out
    assert "プリセット「polished」" in out and "Ctrl+C" in out
    assert "カメラ OK ｜ 仮想カメラ OBS Virtual Camera ｜ 顔 検出中 ｜ 29.5 fps ｜ 処理 31.0 ms" in out
    assert "『OBS Virtual Camera』を選んでください" in out
    assert "カメラ 接続中" not in out  # nothing is printed after stopping


@pytest.fixture
def fake_headless(monkeypatch: pytest.MonkeyPatch) -> type[FakeRunner]:
    FakeRunner.instances.clear()
    monkeypatch.setattr(cli, "prepare_models", lambda stream: None)
    monkeypatch.setattr(cli, "MakeupPipeline", FakePipeline)
    monkeypatch.setattr(runner, "CameraRunner", FakeRunner)
    monkeypatch.setattr(cli, "wait_for_interrupt", lambda: None)
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
