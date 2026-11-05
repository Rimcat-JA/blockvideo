"""Command line entry point: ``python -m makeup_cam``.

One mode per run:

- no mode option: the settings window with live preview (``gui.run_gui``),
- ``--headless``: camera -> makeup -> virtual camera without a window,
  printing status lines until Ctrl+C,
- ``--image IN --out OUT``: apply the makeup to a still image (no camera
  needed), optionally as a side-by-side before/after (``--compare``),
- ``--video IN --out OUT``: the same for a video file, frame by frame (for checking
  the makeup in motion; the result is an .mp4 without sound),
- ``--from-photo MAKEUP --name NAME``: make a preset from a photo of the makeup,
- ``--list-cameras`` / ``--download-models``: small maintenance tasks.

Messages for the user are Japanese. Expected failures (missing preset, model
download, unreadable image, ...) end with a short message instead of a
traceback; details go to the log.
"""

from __future__ import annotations

import argparse
import logging
import re
import signal
import sys
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn, TextIO

import cv2
import numpy as np

from . import __version__
from .config import SECTION_LABELS, MakeupSettings
from .models import FACE_LANDMARKER, SELFIE_MULTICLASS, ModelDownloadError, ensure_all_models, ensure_model, models_dir
from .pipeline import MakeupPipeline, ProcessResult
from .presets import DEFAULT_PRESET_NAME, PresetError, initial_settings, last_session_path, list_presets, load_preset
from .runner import ConsoleCloseWatch, RunnerStatus
from .segmentation import Segmenter
from .tracking import FaceTracker

logger = logging.getLogger("makeup_cam")

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_INTERRUPTED = 130

DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
DEFAULT_FPS = 30

# A still image is fed as this many video frames so that the landmark tracker
# (about 10 frames), the segmentation smoothing and the makeup fade-in settle.
STILL_FRAMES = 20
STILL_FRAME_INTERVAL_MS = 33

VIDEO_OUT_SUFFIX = ".mp4"
VIDEO_FOURCC = "mp4v"  # what every OpenCV build can write, without extra codec libraries
VIDEO_FALLBACK_FPS = 30.0  # when a file does not state a believable frame rate
VIDEO_FPS_RANGE = (2.0, 240.0)
VIDEO_PROGRESS_EVERY = 30  # frames

STATUS_REPEAT_S = 5.0
# A video file that gives no first frame in this time is treated as unplayable.
CAMERA_FILE_CHECK_TIMEOUT_S = 15.0
# What the built-in preset files contain (file names stay ASCII so they
# survive zip archives and every file system).
BUILTIN_PRESET_LABELS = {
    "natural": "就活ナチュラル（標準）",
    "bare": "すっぴん風補正",
    "polished": "しっかりメイク",
    "strong": "しっかりメイク＋小顔補正",
    "off": "全部オフ",
}


class CliError(Exception):
    """An expected failure; the message is Japanese and printed as is."""


# argparse's own messages, as written by Python's standard library, in Japanese. Anything else passes unchanged.
_USAGE_ERRORS = (
    (re.compile(r"argument (.+): not allowed with argument (.+)"), "{0} と {1} は同時に指定できません"),
    (re.compile(r"argument (.+): expected one argument"), "{0} には値を指定してください"),
    (re.compile(r"argument (.+): ignored explicit argument (.+)"), "{0} には値を指定できません（{1}）"),
    (re.compile(r"ambiguous option: (.+) could match (.+)"), "{0} だけでは {1} のどれか分かりません"),
    (re.compile(r"unrecognized arguments: (.+)"), "使えないオプションです: {0}"),
    (re.compile(r"argument (.+): (.+)", re.DOTALL), "{0}: {1}"),  # our own type checks word the second part
)


def _translate_usage_error(message: str) -> str:
    for pattern, japanese in _USAGE_ERRORS:
        match = pattern.fullmatch(message)
        if match is not None:
            return japanese.format(*match.groups())
    return message


class _JapaneseHelpFormatter(argparse.HelpFormatter):
    def add_usage(self, usage: Any, actions: Any, groups: Any, prefix: str | None = None) -> None:
        super().add_usage(usage, actions, groups, "使い方: " if prefix is None else prefix)


class _JapaneseArgumentParser(argparse.ArgumentParser):
    """Reports misuse in Japanese."""

    def error(self, message: str) -> NoReturn:
        self.print_usage(sys.stderr)
        self.exit(2, f"エラー: {_translate_usage_error(message)}\n")


# --------------------------------------------------------------------------
# Argument parsing
# --------------------------------------------------------------------------


def _positive_int(text: str) -> int:
    value = _int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"1 以上の整数を指定してください: {text}")
    return value


def _index(text: str) -> int:
    value = _int(text)
    if value < 0:
        raise argparse.ArgumentTypeError(f"0 以上の整数を指定してください: {text}")
    return value


def _int(text: str) -> int:
    try:
        return int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"整数を指定してください: {text}") from None


def build_parser() -> argparse.ArgumentParser:
    presets = "、".join(f"{name}＝{label}" for name, label in BUILTIN_PRESET_LABELS.items())
    parser = _JapaneseArgumentParser(
        prog="python -m makeup_cam",
        formatter_class=_JapaneseHelpFormatter,
        description="Web カメラの映像に自然なメイクをかけて、仮想カメラ（OBS Virtual Camera）に送ります。"
        "オプションなしで起動すると設定画面が開きます。",
        epilog=f"標準プリセット: {presets}",
        add_help=False,
    )
    modes = parser.add_argument_group("動作モード（どれか1つ。指定しなければ設定画面）")
    exclusive = modes.add_mutually_exclusive_group()
    exclusive.add_argument(
        "--headless",
        action="store_true",
        help="画面を出さずに、カメラ → メイク → 仮想カメラを実行します（Ctrl+C で終了）",
    )
    exclusive.add_argument(
        "--image", metavar="IN", type=Path, help="静止画にメイクをかけます（カメラ不要。--out が必要）"
    )
    exclusive.add_argument(
        "--video",
        metavar="IN",
        type=Path,
        help="動画ファイルに 1 フレームずつメイクをかけて .mp4 で保存します"
        "（カメラ不要。動きの確認用。音声は入りません。--out が必要）",
    )
    exclusive.add_argument(
        "--from-photo",
        metavar="MAKEUP",
        type=Path,
        help="メイクをした顔の写真からプリセットを作ります（カメラ不要。--name が必要）",
    )
    exclusive.add_argument("--list-cameras", action="store_true", help="接続されているカメラの番号と名前を表示します")
    exclusive.add_argument("--download-models", action="store_true", help="顔認識モデルをダウンロードして終了します")

    makeup = parser.add_argument_group("メイクの設定")
    source = makeup.add_mutually_exclusive_group()
    source.add_argument(
        "--preset", metavar="NAME", help="使うプリセット名（例: natural）。保存したプリセット名も使えます"
    )
    source.add_argument(
        "--settings",
        metavar="FILE",
        type=Path,
        help="設定ファイル（プリセットの JSON）を使います（--headless / --image / --video 用）",
    )

    image = parser.add_argument_group("静止画・動画（--image / --video）")
    image.add_argument("--out", metavar="OUT", type=Path, help="保存先のファイル（--image は .png / .jpg、--video は .mp4）")
    image.add_argument("--compare", action="store_true", help="左に元の映像、右にメイク後を並べて保存します")

    photo = parser.add_argument_group("写真から作る（--from-photo）")
    photo.add_argument(
        "--bare",
        metavar="BARE",
        type=Path,
        help="同じ場所・同じ照明で撮ったすっぴんの写真（強くおすすめします。1 枚ではアイシャドウをほとんど測れません）",
    )
    photo.add_argument("--name", metavar="NAME", help="保存するプリセットの名前（標準プリセットと同じ名前は使えません）")

    camera = parser.add_argument_group("カメラ")
    camera_source = camera.add_mutually_exclusive_group()
    camera_source.add_argument(
        "--camera", metavar="N", type=_index, help="使うカメラの番号（--list-cameras で確認できます）"
    )
    camera_source.add_argument(
        "--camera-file",
        metavar="PATH",
        type=Path,
        help="カメラの代わりに動画ファイルを繰り返し再生します"
        "（カメラなしでのテスト用。設定画面 / --headless で使えます）",
    )
    camera.add_argument(
        "--width", metavar="W", type=_positive_int, help="カメラの横の解像度（例: 1280。--height と一緒に）"
    )
    camera.add_argument(
        "--height", metavar="H", type=_positive_int, help="カメラの縦の解像度（例: 720。--width と一緒に）"
    )
    camera.add_argument("--fps", metavar="F", type=_positive_int, help="フレームレート（例: 30）")
    camera.add_argument(
        "--no-virtual-camera",
        dest="no_vcam",
        action="store_true",
        help="仮想カメラに出力しません（プレビュー・確認用）",
    )

    other = parser.add_argument_group("その他")
    other.add_argument("-v", "--verbose", action="store_true", help="詳しいログを表示します")
    other.add_argument("--version", action="version", version=f"%(prog)s {__version__}", help="バージョンを表示します")
    other.add_argument("-h", "--help", action="help", help="このヘルプを表示します")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse and cross-check the options; exits with a usage message on misuse."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.image is not None and args.out is None:
        parser.error("--image には保存先の --out も指定してください")
    if args.video is not None and args.out is None:
        parser.error("--video には保存先の --out も指定してください")
    if args.image is None and args.video is None and (args.out is not None or args.compare):
        parser.error("--out と --compare は --image または --video と一緒に使います")
    if args.video is not None and args.out.suffix.lower() != VIDEO_OUT_SUFFIX:
        parser.error(f"--video の保存先は {VIDEO_OUT_SUFFIX} にしてください: {args.out}")
    if args.video is not None and args.video.resolve() == args.out.resolve():
        parser.error("--video の保存先には、元の動画と別のファイルを指定してください")
    if (args.width is None) != (args.height is None):
        parser.error("--width と --height は両方を指定してください")
    if args.settings is not None and not (args.headless or args.image is not None or args.video is not None):
        parser.error(
            "--settings は --headless、--image、--video のどれかと一緒に使います"
            "（設定画面ではプリセットを選んでください）"
        )
    other_mode = (
        args.image is not None
        or args.video is not None
        or args.from_photo is not None
        or args.list_cameras
        or args.download_models
    )
    if args.camera_file is not None and other_mode:
        parser.error("--camera-file は設定画面または --headless で使います")
    if args.from_photo is not None and not args.name:
        parser.error("--from-photo には保存するプリセットの --name も指定してください")
    if args.from_photo is None and (args.bare is not None or args.name is not None):
        parser.error("--bare と --name は --from-photo と一緒に使います")
    return args


# --------------------------------------------------------------------------
# Shared steps
# --------------------------------------------------------------------------


def resolve_settings(args: argparse.Namespace) -> tuple[MakeupSettings, str]:
    """Settings for --headless / --image / --video and a Japanese description of where they came from.

    Without ``--preset`` / ``--settings`` this is what the window would start
    with: the last session, else the default built-in preset.
    """
    if args.settings is not None:
        return _load_settings_file(args.settings), f"設定ファイル「{args.settings}」"
    if args.preset:
        return _load_named_preset(args.preset), f"プリセット「{args.preset}」"
    settings, preset = initial_settings()
    if preset:
        return settings, f"プリセット「{preset}」"
    return settings, "前回の設定" if last_session_path().is_file() else "初期値"


def _load_settings_file(path: Path) -> MakeupSettings:
    try:
        return MakeupSettings.load(path)
    except FileNotFoundError:
        raise CliError(f"設定ファイルが見つかりません: {path}") from None
    except (OSError, ValueError) as exc:
        raise CliError(f"設定ファイルを読み込めませんでした: {path}\n（詳細: {exc}）") from exc


def _load_named_preset(name: str) -> MakeupSettings:
    try:
        return load_preset(name)
    except PresetError as exc:
        names = "、".join(info.name for info in list_presets()) or "（なし）"
        raise CliError(f"{exc}\n使えるプリセット: {names}") from exc


class DownloadProgress:
    """Prints one self-updating line per model while it downloads."""

    def __init__(self, stream: TextIO) -> None:
        self._stream = stream
        self._current: str | None = None

    def __call__(self, name: str, done: int, total: int) -> None:
        if self._current is None:
            self._stream.write("顔認識モデルをダウンロードしています（初回のみ・合計 約 20 MB）…\n")
        elif name != self._current:
            self._stream.write("\n")
        self._current = name
        amount = f"{done / 1e6:.1f} / {total / 1e6:.1f} MB" if total else f"{done / 1e6:.1f} MB"
        self._stream.write(f"\r  {name}: {amount}")
        self._stream.flush()

    def finish(self) -> None:
        if self._current is not None:
            self._stream.write("\n")
            self._stream.flush()


def prepare_models(stream: TextIO) -> None:
    """Make sure both models are present, downloading them with a progress line if needed."""
    progress = DownloadProgress(stream)
    try:
        ensure_all_models(progress)
    except ModelDownloadError as exc:
        raise CliError(str(exc)) from exc
    finally:
        progress.finish()


def _start_failure(exc: Exception) -> CliError:
    logger.debug("pipeline start-up failed", exc_info=exc)
    return CliError(f"顔認識の準備に失敗しました。\n（詳細: {exc}）")


def check_camera_file(path: Path, timeout_s: float | None = None) -> None:
    """Stop right away on a video file that cannot be played, before the models load.

    The check runs on its own thread: a decoder stuck inside a native call could
    otherwise freeze the app with not even Ctrl+C getting through.
    """
    from .camera import CameraError, VideoFileSource

    problems: list[str] = []

    def check() -> None:
        source = VideoFileSource(path)
        try:
            source.open()
        except CameraError as exc:
            problems.append(str(exc))
        except Exception as exc:
            logger.debug("checking the video file failed", exc_info=exc)
            problems.append(f"動画ファイルを読み込めませんでした: {path}\n（詳細: {exc}）")
        finally:
            source.close()

    checker = threading.Thread(target=check, name="camera-file-check", daemon=True)
    checker.start()
    deadline = time.monotonic() + (CAMERA_FILE_CHECK_TIMEOUT_S if timeout_s is None else timeout_s)
    while checker.is_alive() and time.monotonic() < deadline:
        checker.join(0.1)  # short waits, because a Python signal handler runs only between them
    if checker.is_alive():
        raise CliError(f"動画ファイルを読み込めませんでした（読み込みが終わりません）: {path}")
    if problems:
        raise CliError(problems[0])


# --------------------------------------------------------------------------
# --image
# --------------------------------------------------------------------------


@contextmanager
def file_pipeline(settings: MakeupSettings) -> Iterator[MakeupPipeline]:
    """The real pipeline for stills and video files: never mirrored, and segmentation runs
    inline, one run per ``segmentation_interval`` frames, so the same file gives the same result."""
    unmirrored = settings.copy()
    unmirrored.general.mirror_output = False
    general = unmirrored.general
    with ExitStack() as stack:
        tracker = FaceTracker(ensure_model(FACE_LANDMARKER), general.landmark_smoothing)
        stack.callback(tracker.close)
        segmenter = Segmenter(ensure_model(SELFIE_MULTICLASS), general.segmentation_interval, background=False)
        stack.callback(segmenter.close)
        pipeline = MakeupPipeline(unmirrored, tracker=tracker, segmenter=segmenter)
        stack.callback(pipeline.close)
        yield pipeline


def render_still(frame: np.ndarray, settings: MakeupSettings) -> ProcessResult:
    """Apply ``settings`` to one uint8 BGR image with the real models.

    The image is fed as a short static video (``STILL_FRAMES``) so tracking,
    segmentation and the fade-in settle.
    """
    with file_pipeline(settings) as pipeline:
        result = pipeline.process(frame, 0)
        for index in range(1, STILL_FRAMES):
            result = pipeline.process(frame, index * STILL_FRAME_INTERVAL_MS)
        return result


def side_by_side(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """``before | after`` with a thin white divider and ASCII labels (OpenCV fonts have no Japanese)."""
    height, width = before.shape[:2]
    gap = max(2, width // 320)
    canvas = np.full((height, 2 * width + gap, 3), 255, np.uint8)
    canvas[:, :width] = before
    canvas[:, width + gap :] = after
    _draw_label(canvas, "BEFORE", 0, height)
    _draw_label(canvas, "AFTER", width + gap, height)
    return canvas


def _draw_label(canvas: np.ndarray, text: str, x0: int, height: int) -> None:
    scale = max(0.4, height / 900)
    thickness = max(1, round(scale * 2))
    (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    pad = max(4, round(text_h * 0.5))
    top_left = (x0 + pad, pad)
    bottom_right = (x0 + 3 * pad + text_w, 3 * pad + text_h + baseline)
    cv2.rectangle(canvas, top_left, bottom_right, (40, 40, 40), cv2.FILLED)
    origin = (x0 + 2 * pad, 2 * pad + text_h)
    cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def read_image(path: Path) -> np.ndarray:
    """Read an image as uint8 BGR (works with non-ASCII paths on Windows)."""
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except OSError as exc:
        raise CliError(f"画像を読み込めませんでした: {path}\n（詳細: {exc}）") from exc
    image = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    if image is None:
        raise CliError(f"画像を読み込めませんでした（画像ファイルではないか、壊れています）: {path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    """Write an image; the format follows the extension (non-ASCII paths are fine)."""
    suffix = path.suffix.lower() or ".png"
    try:
        ok, encoded = cv2.imencode(suffix, image)
    except cv2.error:
        ok = False
    if not ok:
        raise CliError(f"この拡張子では保存できません: {path}（.png か .jpg を指定してください）")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded.tofile(path)
    except OSError as exc:
        raise CliError(f"画像を保存できませんでした: {path}\n（詳細: {exc}）") from exc


def run_image(args: argparse.Namespace) -> int:
    settings, source = resolve_settings(args)
    frame = read_image(args.image)
    prepare_models(sys.stderr)
    try:
        result = render_still(frame, settings)
    except (OSError, RuntimeError) as exc:
        raise _start_failure(exc) from exc
    write_image(args.out, side_by_side(frame, result.output) if args.compare else result.output)
    print(f"保存しました: {args.out}（{source}）")
    if not result.face_found:
        print("注意: 顔が見つからなかったため、髪以外のメイクはかかっていません。", file=sys.stderr)
    return EXIT_OK


# --------------------------------------------------------------------------
# --video
# --------------------------------------------------------------------------


@dataclass
class VideoSummary:
    """What ``process_video`` went through."""

    frames: int = 0
    frames_with_face: int = 0
    total_ms: float = 0.0  # sum of the pipeline's own time per frame


def _open_video(path: Path) -> tuple[Any, float]:
    """A capture of a file that ``check_camera_file`` accepted, and its frame rate."""
    from .camera import quiet_opencv_logs

    with quiet_opencv_logs:
        for api in (cv2.CAP_FFMPEG, cv2.CAP_ANY):
            cap = cv2.VideoCapture(str(path), api)
            if cap.isOpened():
                fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
                low, high = VIDEO_FPS_RANGE
                return cap, fps if low <= fps <= high else VIDEO_FALLBACK_FPS
            cap.release()
    raise CliError(f"動画ファイルを読み込めませんでした: {path}")


def next_video_timestamp(position_ms: float, previous_ms: int | None, interval_ms: float) -> int:
    """The timestamp of a decoded frame: the one in the file, if it is usable.

    The landmark model needs strictly increasing milliseconds. Containers without
    timestamps report 0 or NaN, and those frames get the nominal frame interval.
    """
    if previous_ms is None:
        return max(0, round(position_ms)) if position_ms == position_ms else 0
    if position_ms != position_ms or position_ms <= previous_ms:  # NaN, or going nowhere
        return previous_ms + max(1, round(interval_ms))
    return max(previous_ms + 1, round(position_ms))


def video_frames(path: Path) -> Iterator[tuple[np.ndarray, int, float]]:
    """Every frame of a video file, in order: (uint8 BGR, timestamp in ms, the file's frame rate)."""
    from .camera import _to_bgr

    cap, fps = _open_video(path)
    interval_ms = 1000.0 / fps
    timestamp = None
    try:
        while True:
            try:
                ok, frame = cap.read()
            except cv2.error as exc:
                logger.debug("reading %s stopped: %s", path, exc)
                return
            if not ok or frame is None or not frame.size:
                return  # the end of the file, or a damaged tail
            timestamp = next_video_timestamp(float(cap.get(cv2.CAP_PROP_POS_MSEC)), timestamp, interval_ms)
            yield _to_bgr(frame), timestamp, fps
    finally:
        cap.release()


def process_video(
    path: Path,
    settings: MakeupSettings,
    on_frame: Callable[[np.ndarray, ProcessResult, float], None],
) -> VideoSummary:
    """Run every frame of a video file through the real pipeline with the file's own timestamps.

    Calls ``on_frame(input frame, result, file frame rate)`` for each frame. Nothing is dropped,
    however slow it is. Live, the masks come from a worker thread and may be one model run older.
    """
    summary = VideoSummary()
    with file_pipeline(settings) as pipeline:
        for frame, timestamp, fps in video_frames(path):
            result = pipeline.process(frame, timestamp)
            summary.frames += 1
            summary.frames_with_face += result.face_found
            summary.total_ms += result.timings_ms.get("total", 0.0)
            on_frame(frame, result, fps)
    return summary


class _Mp4Writer:
    """Opens the file at the first frame, when the size is known, and says so if that fails."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._writer: cv2.VideoWriter | None = None

    def write(self, image: np.ndarray, fps: float) -> None:
        if self._writer is None:
            self._writer = self._open(image, fps)
        self._writer.write(image)

    def _open(self, image: np.ndarray, fps: float) -> cv2.VideoWriter:
        height, width = image.shape[:2]
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise CliError(f"動画を保存できませんでした: {self._path}\n（詳細: {exc}）") from exc
        fourcc = cv2.VideoWriter.fourcc(*VIDEO_FOURCC)
        writer = cv2.VideoWriter(str(self._path), cv2.CAP_FFMPEG, fourcc, fps, (width, height))
        if not writer.isOpened():
            writer.release()
            raise CliError(f"動画を保存できませんでした: {self._path}")
        return writer

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()  # also after an interruption: what was written stays playable
            self._writer = None


def run_video(args: argparse.Namespace) -> int:
    settings, source = resolve_settings(args)
    check_camera_file(args.video)
    prepare_models(sys.stderr)
    writer = _Mp4Writer(args.out)
    count = [0]

    def on_frame(frame: np.ndarray, result: ProcessResult, fps: float) -> None:
        writer.write(side_by_side(frame, result.output) if args.compare else result.output, fps)
        count[0] += 1
        if count[0] % VIDEO_PROGRESS_EVERY == 0:
            print(f"\r処理中… {count[0]} フレーム", end="", file=sys.stderr, flush=True)

    try:
        summary = process_video(args.video, settings, on_frame)
    except (OSError, RuntimeError) as exc:
        raise _start_failure(exc) from exc
    finally:
        writer.close()
        if count[0] >= VIDEO_PROGRESS_EVERY:
            print(file=sys.stderr)
    if summary.frames == 0:
        raise CliError(f"動画ファイルからフレームを読み込めませんでした: {args.video}")
    average_ms = summary.total_ms / summary.frames
    print(
        f"保存しました: {args.out}（{source}、{summary.frames} フレーム、"
        f"顔あり {summary.frames_with_face} フレーム、平均 {average_ms:.1f} ms）"
    )
    if summary.frames_with_face == 0:
        print("注意: どのフレームでも顔が見つからなかったため、髪以外のメイクはかかっていません。", file=sys.stderr)
    return EXIT_OK


# --------------------------------------------------------------------------
# --headless
# --------------------------------------------------------------------------


def format_status(status: RunnerStatus, *, virtual_camera_enabled: bool) -> str:
    """One status line, e.g. ``カメラ OK ｜ 仮想カメラ OBS Virtual Camera ｜ 顔 あり ｜ 29.8 fps ｜ 処理 31.0 ms``."""
    camera = "OK" if status.camera_ok else "エラー" if status.camera_error else "接続中…"
    if not virtual_camera_enabled:
        virtual = "オフ"
    elif status.virtual_camera:
        virtual = status.virtual_camera
    else:
        virtual = "エラー" if status.virtual_camera_error else "準備中…"
    face = "あり" if status.face_found else "なし"
    return (
        f"カメラ {camera} ｜ 仮想カメラ {virtual} ｜ 顔 {face} ｜ "
        f"{status.fps:.1f} fps ｜ 処理 {status.processing_ms:.1f} ms"
    )


class StatusPrinter:
    """Prints runner status lines: at once on a change, otherwise every ``STATUS_REPEAT_S``.

    Each problem message (camera, virtual camera, processing) is printed in
    full once when it appears. Called on the runner's output thread.
    """

    def __init__(
        self,
        stream: TextIO,
        *,
        virtual_camera_enabled: bool,
        repeat_s: float = STATUS_REPEAT_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._stream = stream
        self._virtual_camera_enabled = virtual_camera_enabled
        self._repeat_s = repeat_s
        self._clock = clock
        self._lock = threading.Lock()
        self._state: tuple[Any, ...] | None = None
        self._printed_at = 0.0
        self._problems: set[str] = set()
        self._device_announced = False
        self._closed = False

    def __call__(self, status: RunnerStatus) -> None:
        state = (
            status.camera_ok,
            status.camera_error,
            status.virtual_camera,
            status.virtual_camera_error,
            status.face_found,
            status.processing_error,
        )
        with self._lock:
            now = self._clock()
            if self._closed or (state == self._state and now - self._printed_at < self._repeat_s):
                return
            self._state, self._printed_at = state, now
            lines = [format_status(status, virtual_camera_enabled=self._virtual_camera_enabled)]
            lines += self._new_problems(status)
            lines += self._device_hint(status)
            self._stream.write("".join(f"{line}\n" for line in lines))
            self._stream.flush()

    def close(self) -> None:
        """Stop printing (the runner reports a final idle status when it stops)."""
        with self._lock:
            self._closed = True

    def _new_problems(self, status: RunnerStatus) -> list[str]:
        current = {p for p in (status.camera_error, status.virtual_camera_error, status.processing_error) if p}
        fresh = current - self._problems
        self._problems = current
        return [f"注意: {problem}" for problem in sorted(fresh)]

    def _device_hint(self, status: RunnerStatus) -> list[str]:
        if self._device_announced or not status.virtual_camera:
            return []
        self._device_announced = True
        return [f"→ Zoom・Teams・Google Meet のカメラ設定で『{status.virtual_camera}』を選んでください。"]


def _raise_interrupt(_signum: int, _frame: object) -> None:
    raise KeyboardInterrupt


def wait_for_interrupt(console: ConsoleCloseWatch | None = None) -> None:
    """Block until Ctrl+C, SIGTERM, a closed terminal or, on Windows, Ctrl+Break or a closed console window."""
    if threading.current_thread() is threading.main_thread():
        for name in ("SIGTERM", "SIGBREAK", "SIGHUP"):
            if hasattr(signal, name):
                signal.signal(getattr(signal, name), _raise_interrupt)
    try:
        while console is None or not console.closing.is_set():
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass


def _say_quietly(text: str) -> None:
    """Print without letting a console that is going away stop the cleanup."""
    try:
        print(text, flush=True)
    except (OSError, ValueError):
        pass


def _headless_camera(args: argparse.Namespace) -> tuple[int, str, dict[str, Any]]:
    """The camera index, its description and the extra ``CameraRunner`` options for --headless."""
    from .camera import resolve_camera, video_file_camera_factory

    if args.camera_file is not None:
        factory = video_file_camera_factory(args.camera_file)
        return 0, f"動画ファイル「{args.camera_file.name}」", {"camera_factory": factory}
    if args.camera is not None:
        return args.camera, f"カメラ {args.camera}", {}
    index, name = resolve_camera(None)  # the first real camera, never a virtual one
    return index, f"カメラ {index}「{name}」" if name else f"カメラ {index}", {"camera_name": name}


def run_headless(args: argparse.Namespace) -> int:
    from .camera import virtual_camera_size
    from .runner import CameraRunner

    settings, source = resolve_settings(args)
    prepare_models(sys.stderr)
    try:
        pipeline = MakeupPipeline(settings)
    except (OSError, RuntimeError) as exc:
        raise _start_failure(exc) from exc
    camera_index, camera_label, camera_options = _headless_camera(args)
    width, height = (args.width, args.height) if args.width else (DEFAULT_WIDTH, DEFAULT_HEIGHT)
    fps = args.fps or DEFAULT_FPS
    printer = StatusPrinter(sys.stdout, virtual_camera_enabled=not args.no_vcam)
    if not args.verbose:
        # The printer already shows camera and virtual camera problems; keep unexpected errors only.
        logging.getLogger("makeup_cam.runner").setLevel(logging.ERROR)
    runner = CameraRunner(
        pipeline,
        camera_index=camera_index,
        width=width,
        height=height,
        fps=fps,
        virtual_camera=not args.no_vcam,
        output_size=virtual_camera_size(width, height),
        on_status=printer,
        **camera_options,
    )
    print(f"{source}でメイクを始めます（{camera_label}、{width}x{height}、{fps} fps）。Ctrl+C で終了します。")
    console = ConsoleCloseWatch()
    runner.start()
    try:
        wait_for_interrupt(console)
    finally:
        printer.close()
        _say_quietly("終了しています…")
        runner.stop()
        pipeline.close()
        console.finish()
    return EXIT_OK


# --------------------------------------------------------------------------
# --from-photo
# --------------------------------------------------------------------------


def run_from_photo(args: argparse.Namespace) -> int:
    """Measure the makeup in a photo and save it as a preset. The photos themselves are not kept."""
    from .analyze import AnalysisError, analyze_makeup, read_photo, save_as_preset

    base_name = args.preset or DEFAULT_PRESET_NAME
    base = _load_named_preset(base_name)
    try:
        makeup = read_photo(args.from_photo)
        bare = read_photo(args.bare) if args.bare is not None else None
    except AnalysisError as exc:
        raise CliError(str(exc)) from exc
    prepare_models(sys.stderr)
    if bare is None:
        print(
            "注意: すっぴんの写真（--bare）がありません。メイクの写真 1 枚だけでは、アイシャドウをほとんど測れません。"
            "同じ場所・同じ照明で撮ったすっぴんの写真を入れることを強くおすすめします。",
            file=sys.stderr,
        )
    print("写真を解析しています…", flush=True)
    try:
        result = analyze_makeup(makeup, bare, base=base)
    except (AnalysisError, ModelDownloadError) as exc:
        raise CliError(str(exc)) from exc
    except (OSError, RuntimeError) as exc:
        raise _start_failure(exc) from exc
    if _print_photo_report(result, base_name) == 0:
        raise CliError("どの項目も測れなかったので、プリセットは保存していません。写真を確認して、撮り直してください。")
    try:
        info = save_as_preset(result, args.name)
    except PresetError as exc:
        raise CliError(str(exc)) from exc
    print(f"プリセット『{info.name}』を保存しました: {info.path}")
    return EXIT_OK


def _print_photo_report(result: Any, base_name: str) -> int:
    """Print the confidence per section and the notes; returns how many sections were measured at all."""
    from .analyze import MEASURED_SECTIONS

    measured = 0
    print("項目ごとの信頼度（1 に近いほど確か）:")
    for section in MEASURED_SECTIONS:
        confidence = result.confidence.get(section, 0.0)
        label = SECTION_LABELS.get(section, section)
        if confidence > 0.0:
            measured += 1
            print(f"  {label}: {confidence:.2f}")
        else:
            print(f"  {label}: 測れなかった（{base_name} の値のまま）")
    if result.notes:
        print("メモ:")
        for note in result.notes:
            print(f"  ・{note}")
    return measured


# --------------------------------------------------------------------------
# --list-cameras / --download-models / GUI
# --------------------------------------------------------------------------


def run_list_cameras() -> int:
    from .camera import list_cameras

    print("カメラを探しています…")
    cameras = list_cameras()
    if not cameras:
        print("カメラが見つかりませんでした。接続と、ほかのアプリが使っていないかを確認してください。")
        return EXIT_ERROR
    for info in cameras:
        size = f"（標準 {info.width}x{info.height}）" if info.width and info.height else ""
        print(f"  {info.index}: {info.name}{size}")
    print("使うカメラは --camera 番号 で指定できます（設定画面では上部のメニューで選べます）。")
    return EXIT_OK


def run_download_models() -> int:
    prepare_models(sys.stderr)
    print(f"顔認識モデルは準備できています: {models_dir()}")
    return EXIT_OK


def run_window(args: argparse.Namespace) -> int:
    from .gui import run_gui

    return run_gui(args)


def dispatch(args: argparse.Namespace) -> int:
    if args.list_cameras:
        return run_list_cameras()
    if args.download_models:
        return run_download_models()
    if args.image is not None:
        return run_image(args)
    if args.video is not None:
        return run_video(args)
    if args.from_photo is not None:
        return run_from_photo(args)
    if args.camera_file is not None:
        check_camera_file(args.camera_file)
    if args.headless:
        return run_headless(args)
    return run_window(args)


def _make_console_safe() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:  # never crash on a console that cannot show a character
            reconfigure(errors="replace")


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def main(argv: Sequence[str] | None = None) -> int:
    _make_console_safe()
    args = parse_args(argv)
    _configure_logging(args.verbose)
    try:
        return dispatch(args)
    except CliError as exc:
        print(f"エラー: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except KeyboardInterrupt:
        print("中断しました。", file=sys.stderr)
        return EXIT_INTERRUPTED
    except Exception:
        logger.exception("unexpected error")
        print("エラー: 予期しない問題が発生しました（詳細は上のログを確認してください）。", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
