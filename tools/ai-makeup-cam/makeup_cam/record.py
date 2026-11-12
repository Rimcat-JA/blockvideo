"""Recording tool for the Plan B training footage: ``python -m makeup_cam.record``.

A small window that shows the capture script (``docs/capture-guide.md`` §5.2,
and the evaluation take of §6) in large text while she films herself. Each
step becomes its own video file plus a JSON sidecar.

Threads (the window must never freeze, and a slow disk must never stall the
camera):

- GUI thread: widgets only. It polls ``CaptureEngine.snapshot`` and
  ``TakeRecorder.stats`` on a timer and never waits for the camera or the disk
  during a recording.
- Capture thread: opens and reads the source, publishes a small preview frame
  and offers every frame to the current take without blocking.
- Writer thread, one per take: drains a byte-bounded queue into
  ``cv2.VideoWriter``, checks the finished file and writes the sidecar. When
  the queue is full the frame is dropped and counted; the count is shown live.

Footage is stored exactly as the camera delivers it: no makeup effects, no
mirroring (the preview alone is mirrored) and no audio.

Privacy (these are videos of her own face):

- There is no default output folder, and none is remembered between runs.
- A folder inside a git working tree or inside this project is refused, also
  when it is about to be written to.
- Nothing is ever uploaded: this module opens no network connection. Sidecars
  hold file names only, no absolute paths (they would reveal the user name).

Which take counts (the newest with ``"status": "completed"``):

- ``completed``: stopped at the target time or by hand, played back correctly.
- ``aborted``: stopped for a retake, or by hand within moments of starting.
- ``interrupted``: the window was closed during the take.
- ``failed``: something went wrong, or too many frames are missing.

Files are kept in all four cases; nothing here deletes footage.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import html
import json
import logging
import math
import os
import queue
import re
import shutil
import signal
import statistics
import sys
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from enum import Enum
from itertools import pairwise
from pathlib import Path
from typing import Any, Protocol, TypeVar

import cv2
import numpy as np
from PySide6.QtCore import QEvent, QRectF, QStandardPaths, Qt, QTimer
from PySide6.QtGui import (
    QCloseEvent,
    QColor,
    QFont,
    QFontMetrics,
    QImage,
    QKeySequence,
    QPainter,
    QPaintEvent,
    QResizeEvent,
    QShortcut,
    QShowEvent,
)
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from . import camera as camera_module
from .camera import CameraError, CameraSource, quiet_opencv_logs

logger = logging.getLogger(__name__)

APP_TITLE = "AI メイクカメラ 録画ツール"

COUNTDOWN_S = 5.0
# Memory held by frames waiting for the disk: about 3 s of 1280x720 BGR.
WRITE_QUEUE_BYTES = 256 * 1024 * 1024
# Recording stops cleanly below this, so the video index can still be written.
MIN_FREE_BYTES = 2 * 1024**3
HOUSEKEEPING_INTERVAL_S = 1.0
# A capture gap this long, in frame periods, is a stall: frames were lost. Shorter ones are jitter.
CAMERA_STALL_PERIODS = 2.5
# A frame arriving this soon after the previous one was buffered during a stall.
CAMERA_BURST_PERIODS = 0.5
# A gap this long is one missing frame.
CAMERA_SINGLE_PERIODS = 1.5
# Both only if the gap also stands out from this camera's usual spread of gaps, by about 4 standard
# deviations: this many mean deviations, or median deviations. The running mean follows at this rate
# and takes this many periods at most from one gap, so that an outage leaves it alone.
CAMERA_OUTLIER_SPREADS = 5.0
CAMERA_OUTLIER_MEDIAN_DEVIATIONS = 6.0
CAMERA_SPREAD_RATE = 0.02
CAMERA_SPREAD_MAX_PERIODS = 3.0
# The average gap may be this much longer than the nominal frame period without any frame missing.
CAMERA_RATE_TOLERANCE = 0.04
NO_FRAME_WARNING_S = 1.0
# Without a frame for this long during a take she is alerted, and after the second limit it stops.
CAMERA_ALERT_S = 2.0
CAMERA_LOST_STOP_S = 15.0
ALERT_BEEP_INTERVAL_S = 5.0
# A take missing more than this share of its frames is a failed take, whatever else went right,
# as long as that is more than a moment of it: the steps are 20 s and longer, a hiccup in a few seconds is not.
MAX_LOST_RATIO = 0.10
MIN_LOST_TO_FAIL_S = 1.0
# A share of the first moments of a take says little: no warning about it before this.
LOST_WARNING_AFTER_S = 3.0
# Stopped by hand this soon, it was a slip: the take must not replace a good one.
MIN_USABLE_TAKE_S = 3.0
# cv2.VideoWriter.write reports nothing, so a healthy file must be seen to grow: this long at most
# (FFV1 with a flat image flushes only every 5 s), once this much raw video has been handed over.
WRITE_STALL_S = 8.0
WRITE_STALL_MIN_BYTES = 64 * 1024 * 1024
# The writer has been stuck in one call for this long while frames wait.
WRITE_ALERT_S = 5.0
SAVE_STALL_WARNING_S = 10.0
# Folder checks run on a helper thread, so that a drive that hangs cannot freeze the window.
FOLDER_CHECK_TIMEOUT_S = 5.0
# A new capture thread waits this long for the previous one to release the device.
CAMERA_SWITCH_WAIT_S = 5.0
PREVIEW_WIDTH = 640
FPS_WINDOW_FRAMES = 90
FPS_MIN_FRAMES = 15
FPS_MIN_SPAN_S = 0.5
MAX_PLAUSIBLE_FPS = 120.0
# The measured rate replaces the reported one beyond this: a video then plays at most this far off speed.
FPS_MISMATCH_RATIO = 0.03
# Shown as a warning; the guide asks for at least this (§1.1).
GUIDE_MIN_WIDTH = 1280
GUIDE_MIN_HEIGHT = 720
GUIDE_MIN_FPS = 25.0
DROP_WARNING_RATIO = 0.01
# Shorter than this share of the target time is worth a word; only she knows whether it is enough.
SHORT_TAKE_RATIO = 0.5
UI_INTERVAL_MS = 33
CLOSE_TIMEOUT_S = 20.0
SIDECAR_SCHEMA_VERSION = 1

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Capture timestamps. time.monotonic only ticks every 15.6 ms on Windows.
_now = time.perf_counter


# --------------------------------------------------------------------------
# The script (docs/capture-guide.md §5 and §6)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Cue:
    """One line of a step; ``seconds`` is its share of the step's target time."""

    text: str
    seconds: float


@dataclass(frozen=True)
class ScriptStep:
    key: str  # ASCII, used in file names
    title: str
    instruction: str
    target_s: float
    target_label: str
    cues: tuple[Cue, ...] = ()
    # False: the step ends when she says so (she may be in the middle of a sentence).
    auto_stop: bool = True
    # Stays on screen during the take, under the changing cues.
    note: str = ""


@dataclass(frozen=True)
class Session:
    key: str
    label: str
    lighting: str
    slug: str  # ASCII, used in folder names
    purpose: str  # "training" or "evaluation"
    folder: str  # subfolder of the output folder (§7.1)
    talk_minutes: int = 0
    plan: str = ""


@dataclass(frozen=True)
class Condition:
    key: str  # ASCII, used in folder names
    label: str


CONDITIONS = (Condition("bare", "すっぴん"), Condition("makeup", "メイク"))

SESSIONS = (
    Session("1", "1 回目（1 日目・本番の照明）", "本番の照明", "main", "training", "sessions", 8, "すっぴん 15 分 → メイク 15 分"),
    Session("2", "2 回目（2 日目・昼の窓の光）", "昼の窓の光", "window", "training", "sessions", 5, "すっぴん 10 分 → メイク 10 分"),
    Session("3", "3 回目（3 日目・夜の部屋の照明）", "夜の部屋の照明", "night", "training", "sessions", 5, "すっぴん 10 分 → メイク 10 分"),
    Session(
        "eval",
        "評価用（学習には使わない。別の日に）",
        "本番の照明",
        "mock",
        "evaluation",
        "eval",
        0,
        "模擬面接 20〜30 分（すっぴん）＋ 比較用のメイク 5 分",
    ),
)


def _talk_cues() -> tuple[Cue, ...]:
    return (Cue("自己PR", 1), Cue("ガクチカ", 1), Cue("志望動機", 1), Cue("逆質問", 1))


def training_script(talk_minutes: int) -> tuple[ScriptStep, ...]:
    """The §5.2 script. The talking step fills the time difference between the sessions of §5.1."""
    return (
        ScriptStep(
            "prepare",
            "準備",
            "グレーカードを顔の横に持つ → 下ろす",
            30,
            "30 秒",
            (Cue("グレーカードを顔の横に持つ", 20), Cue("下ろす", 10)),
        ),
        ScriptStep("front", "正面", "無表情でカメラを見る、まばたきは自然に", 20, "20 秒"),
        ScriptStep(
            "head",
            "頭の動き",
            "ゆっくり左右を向く（30° くらいまで）、上下を向く、首をかしげる、うなずく",
            60,
            "1 分",
            (
                Cue("ゆっくり左右を向く（30° くらいまで）", 25),
                Cue("上下を向く", 15),
                Cue("首をかしげる", 10),
                Cue("うなずく", 10),
            ),
        ),
        ScriptStep(
            "bow",
            "お辞儀",
            "「よろしくお願いします」と言って会釈（15°）× 3 回、普通のお辞儀（30°）× 3 回",
            60,
            "1 分",
            (Cue("「よろしくお願いします」と言って会釈（15°）× 3 回", 30), Cue("普通のお辞儀（30°）× 3 回", 30)),
        ),
        ScriptStep(
            "talk",
            "話す",
            "面接の答えを声に出して読む。自己PR、ガクチカ、志望動機、逆質問 など。普段の面接のように、相づち・笑顔・考える仕草も入れて",
            talk_minutes * 60,
            "5〜8 分",
            _talk_cues(),
            auto_stop=False,
            note="普段の面接のように、相づち・笑顔・考える仕草も入れて",
        ),
        ScriptStep(
            "expression",
            "表情",
            "口を閉じて微笑む → 歯を見せて笑う → 声を出して笑う → 驚く → 困った顔 → 目を閉じる",
            60,
            "1 分",
            (
                Cue("口を閉じて微笑む", 10),
                Cue("歯を見せて笑う", 10),
                Cue("声を出して笑う", 10),
                Cue("驚く", 10),
                Cue("困った顔", 10),
                Cue("目を閉じる", 10),
            ),
        ),
        ScriptStep(
            "gaze",
            "視線",
            "手元のメモを見る、横を見る、カメラに戻る",
            30,
            "30 秒",
            (Cue("手元のメモを見る", 10), Cue("横を見る", 10), Cue("カメラに戻る", 10)),
        ),
        ScriptStep(
            "motion",
            "小さな動き",
            "髪を耳にかける、あごに少し手を添える、前髪を直す、顔より下で手振り",
            60,
            "1 分",
            (
                Cue("髪を耳にかける", 15),
                Cue("あごに少し手を添える", 15),
                Cue("前髪を直す", 15),
                Cue("顔より下で手振り", 15),
            ),
        ),
        ScriptStep("closing", "終わり", "「ありがとうございました」とお辞儀", 30, "30 秒"),
    )


def evaluation_script(condition: Condition) -> tuple[ScriptStep, ...]:
    """The §6 takes: one long uninterrupted recording, ended by hand."""
    if condition.key == "bare":
        return (
            ScriptStep(
                "interview",
                "模擬面接（通しで）",
                "最初と最後のお辞儀、話す、笑う、メモを見る、髪を耳にかける、などが自然に入るように",
                20 * 60,
                "20〜30 分",
                auto_stop=False,
            ),
        )
    return (
        ScriptStep(
            "comparison",
            "比較用のメイク",
            "本物のメイクをして、面接のように話す（模擬面接と同じ日に）",
            5 * 60,
            "5 分",
            auto_stop=False,
        ),
    )


def script_for(session: Session, condition: Condition) -> tuple[ScriptStep, ...]:
    if session.purpose == "evaluation":
        return evaluation_script(condition)
    return training_script(session.talk_minutes)


def cue_schedule(step: ScriptStep) -> tuple[tuple[float, str], ...]:
    """(start time in seconds, text) of each cue, spread over the step's target time."""
    if not step.cues:
        return ((0.0, step.instruction),)
    total = sum(cue.seconds for cue in step.cues)
    schedule = []
    start = 0.0
    for cue in step.cues:
        schedule.append((round(start, 3), cue.text))
        start += step.target_s * cue.seconds / total
    return tuple(schedule)


def cue_at(step: ScriptStep, elapsed_s: float) -> tuple[str, str | None]:
    """The cue to show after ``elapsed_s`` seconds and the one after it (None for the last)."""
    schedule = cue_schedule(step)
    current = 0
    for index, (start, _) in enumerate(schedule):
        if elapsed_s >= start:
            current = index
    following = schedule[current + 1][1] if current + 1 < len(schedule) else None
    return schedule[current][1], following


# --------------------------------------------------------------------------
# The output folder
# --------------------------------------------------------------------------


class OutputFolderError(ValueError):
    """The folder cannot hold the footage. The message is Japanese and actionable."""


_OUTSIDE_HINT = "リポジトリの外のフォルダ（暗号化した外付けドライブなど）を選んでください。"
# Whole folder names. Sync clients add suffixes: "OneDrive - 会社", "OneDrive-Personal", "Dropbox (会社)".
_CLOUD_FOLDER = re.compile(
    r"(onedrive|skydrive|dropbox|google ?(drive|ドライブ)|icloud( ?drive)?|nextcloud|owncloud|pcloud|mega|box|sync|"
    r"amazon ?drive|proton ?drive|synology ?drive|tresorit|seafile|resilio ?sync|yandex\.?disk)([ _(（-].*)?"
    r"|megasync|my ?drive|マイドライブ|共有ドライブ|shared ?drives?",
    re.IGNORECASE,
)
_ONEDRIVE_ENV = ("OneDrive", "OneDriveConsumer", "OneDriveCommercial")
# Windows opens these as devices wherever they appear in a path, with or without an extension.
_DEVICE_NAME = re.compile(r"(con|prn|aux|nul|com[0-9¹²³]|lpt[0-9¹²³]|conin\$|conout\$)(\..*)?", re.IGNORECASE)
_WINDOWS_PATH_LIMIT = 248  # for a folder; Windows fails with "not found" beyond it
_PROBE_PREFIX = ".makeup_cam_write_test_"

_T = TypeVar("_T")


def find_git_root(path: Path) -> Path | None:
    """The closest folder at or above ``path`` holding ``.git`` (a folder, or a file in a worktree)."""
    for folder in (path, *path.parents):
        if (folder / ".git").exists():
            return folder
    return None


def _is_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _names_a_device(raw: str) -> bool:
    if sys.platform != "win32":
        return False
    return any(_DEVICE_NAME.fullmatch(part.rstrip(" .")) for part in re.split(r"[\\/]+", raw))


def _os_error_reason(exc: OSError) -> str:
    """Why something failed, in Japanese. Without the path that ``str(exc)`` would add."""
    if exc.errno == errno.ENAMETOOLONG or len(str(exc.filename or "")) >= _WINDOWS_PATH_LIMIT:
        return "パスが長すぎます。"
    if isinstance(exc, PermissionError):
        return "書き込みの権限がないか、書き込み禁止のドライブです。"
    if exc.errno == errno.ENOSPC:
        return "空き容量がありません。"
    if isinstance(exc, FileNotFoundError):
        return "ドライブやフォルダが見つかりません。"
    if isinstance(exc, (FileExistsError, NotADirectoryError)):
        return "途中のフォルダと同じ名前のファイルがあります。"
    return exc.strerror or type(exc).__name__


def validate_output_dir(text: str | os.PathLike[str] | None) -> Path:
    """The absolute output folder, or ``OutputFolderError``. Writes nothing to disk.

    Both the path as written and the path with its links resolved are checked,
    so a link cannot lead footage into a repository. When that cannot be
    established (a locked drive, a folder she may not look into) the answer is no.
    """
    raw = "" if text is None else str(text).strip().strip('"')
    if not raw:
        raise OutputFolderError("保存先のフォルダを指定してください。初期値はありません（顔の動画なので、ご自身で選んでください）。")
    if _names_a_device(raw):
        raise OutputFolderError(f"「{raw}」は Windows が予約している名前なので、フォルダには使えません。別の名前にしてください。")
    try:
        written = Path(os.path.abspath(Path(raw).expanduser()))
        resolved = written.resolve()
    except (OSError, ValueError, RuntimeError) as exc:
        raise OutputFolderError(f"このパスは使えません: {raw}") from exc
    try:
        for candidate in (written, resolved):
            root = find_git_root(candidate)
            if root is not None:
                raise OutputFolderError(
                    f"このフォルダは Git の作業フォルダ（{root}）の中にあるので、保存できません。"
                    f"顔の動画が公開リポジトリに入ってしまうのを防ぐためです。{_OUTSIDE_HINT}"
                )
            if _is_within(candidate, _PROJECT_ROOT):
                raise OutputFolderError(f"このフォルダは AI メイクカメラのプログラムの中にあるので、保存できません。{_OUTSIDE_HINT}")
        is_file = resolved.exists() and not resolved.is_dir()
    except OSError as exc:
        raise OutputFolderError(
            f"このフォルダを確認できませんでした: {raw}\n"
            "ドライブがロックされているか、つながっていないか、見る権限がないのかもしれません。"
        ) from exc
    if is_file:
        raise OutputFolderError(f"これはファイルなので、保存先にできません: {resolved}")
    return resolved


def _remove_probe(remove: Callable[[Path], None], path: Path) -> None:
    """Being able to create it is what counted; a virus scanner may hold it for a moment."""
    for attempt in range(3):
        try:
            remove(path)
            return
        except OSError as exc:
            if attempt == 2:
                logger.warning("could not remove the test item %s: %s", path.name, exc)
            else:
                time.sleep(0.05)


def _probe_file(folder: Path) -> None:
    """Create, write and remove a file in ``folder``; raises ``OSError``.

    Not ``tempfile``: on Windows it takes ``PermissionError`` to mean a directory
    of the same name exists and retries TMP_MAX (over 2 billion) times, a freeze
    of days in a folder she may not write to.
    """
    path = folder / f"{_PROBE_PREFIX}{uuid.uuid4().hex}"
    handle = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0), 0o600)
    try:
        os.write(handle, b"0")
    finally:
        os.close(handle)
        _remove_probe(os.unlink, path)


def _probe_subfolder(folder: Path) -> None:
    """What recording will do below ``folder``; raises ``OSError``.

    Not simply a file in ``folder``: an ordinary user may create folders in
    ``C:\\`` but no files there.
    """
    path = folder / f"{_PROBE_PREFIX}{uuid.uuid4().hex}"
    os.mkdir(path)
    try:
        _probe_file(path)
    finally:
        _remove_probe(os.rmdir, path)


def _cannot_write_message(folder: Path, exc: OSError) -> str:
    return f"このフォルダには書き込めません: {folder}\n{_os_error_reason(exc)}別のフォルダを選んでください。"


def _nearest_existing_folder(folder: Path) -> Path | None:
    for candidate in (folder, *folder.parents):
        if candidate.exists():
            if candidate.is_dir():
                return candidate
            raise OutputFolderError(f"途中のフォルダと同じ名前のファイルがあるので、保存先にできません: {candidate}")
    return None


@dataclass(frozen=True)
class OutputFolderCheck:
    folder: Path
    free_bytes: int | None


def inspect_output_dir(text: str | os.PathLike[str] | None) -> OutputFolderCheck:
    """What the setup page needs to know, or ``OutputFolderError``. Leaves nothing behind on disk."""
    folder = validate_output_dir(text)
    try:
        existing = _nearest_existing_folder(folder)
        if existing is None:
            raise OutputFolderError(f"このドライブやフォルダが見つかりません: {folder}")
        if existing == folder:
            _probe_file(folder)
        else:
            _probe_subfolder(existing)
        return OutputFolderCheck(folder, shutil.disk_usage(existing).free)
    except OSError as exc:
        raise OutputFolderError(_cannot_write_message(folder, exc)) from exc


def prepare_output_dir(folder: Path) -> None:
    """Create the folder and confirm a file can be made in it, or raise ``OutputFolderError``."""
    try:
        folder.mkdir(parents=True, exist_ok=True)
        _probe_file(folder)
    except OSError as exc:
        raise OutputFolderError(_cannot_write_message(folder, exc)) from exc


def _within_time_limit(work: Callable[[], _T]) -> _T:
    """Run disk work for the GUI thread on a helper thread, so that a drive that hangs cannot freeze it.

    The result or exception of ``work`` is passed on. Past the limit the helper
    is left to finish by itself and ``OutputFolderError`` is raised.
    """
    outcome: list[_T] = []
    failure: list[BaseException] = []

    def run() -> None:
        try:
            outcome.append(work())
        except BaseException as exc:
            failure.append(exc)

    thread = threading.Thread(target=run, name="record-folder-check", daemon=True)
    thread.start()
    thread.join(FOLDER_CHECK_TIMEOUT_S)
    if failure:
        raise failure[0]
    if outcome:
        return outcome[0]
    raise OutputFolderError("保存先のドライブから応答がありません。つながっているか確認して、もう一度試してください。")


def cloud_sync_warning(folder: Path) -> str | None:
    """A Japanese warning when the folder looks synced to a cloud service, else None.

    This tool uploads nothing, but a sync client watching the folder would.
    """
    cloud_roots = [Path(os.environ[name]) for name in _ONEDRIVE_ENV if os.environ.get(name)]
    named = any(_CLOUD_FOLDER.fullmatch(part) for part in folder.parts)
    if named or any(_is_within(folder, root) for root in cloud_roots):
        return (
            "このフォルダはクラウド（OneDrive など）と同期されているようです。"
            "このツールは何もアップロードしませんが、同期アプリが顔の動画をクラウドに送ってしまいます。"
            "同期されないフォルダ（暗号化した外付けドライブなど）をおすすめします。"
        )
    return None


def session_folder(output: Path, session: Session, condition: Condition, day: date) -> Path:
    """E.g. ``<output>/sessions/2026-11-10_main_bare`` (the layout of §7.1, one folder per recording)."""
    return output / session.folder / f"{day.isoformat()}_{session.slug}_{condition.key}"


# --------------------------------------------------------------------------
# Video files
# --------------------------------------------------------------------------


class VideoWriterLike(Protocol):
    """The subset of ``cv2.VideoWriter`` used here (injectable for tests)."""

    def isOpened(self) -> bool: ...
    def write(self, frame: np.ndarray) -> None: ...
    def release(self) -> None: ...


WriterFactory = Callable[[Path, str, float, tuple[int, int]], VideoWriterLike]


@dataclass(frozen=True)
class VideoFormat:
    key: str
    fourcc: str
    extension: str
    lossless: bool
    # Share of the raw BGR data rate, measured with a noisy 1280x720 test clip.
    size_ratio: float


# HuffYUV: pixel-exact and about 10 times faster than FFV1 here (4 ms against 35 ms per
# 720p frame), so it keeps up on a busy PC. The pip build of OpenCV has no H.264 encoder.
VIDEO_FORMATS = (
    VideoFormat("hfyu", "HFYU", ".mkv", True, 0.6),
    VideoFormat("ffv1", "FFV1", ".mkv", True, 0.5),
    VideoFormat("mjpg", "MJPG", ".mkv", False, 0.05),
)


def video_format_order(preferred: str | None) -> tuple[VideoFormat, ...]:
    """All formats to try, the preferred one first."""
    return tuple(sorted(VIDEO_FORMATS, key=lambda fmt: fmt.key != preferred))


def cv2_video_writer(path: Path, fourcc: str, fps: float, size: tuple[int, int]) -> VideoWriterLike:
    # Always FFmpeg: unlike OpenCV's built-in MJPEG writer it accepts Japanese folder names.
    return cv2.VideoWriter(str(path), cv2.CAP_FFMPEG, cv2.VideoWriter.fourcc(*fourcc), fps, size)


@dataclass(frozen=True)
class VideoProbe:
    """What a recorded file turns out to contain when opened again."""

    playable: bool
    frames: int
    width: int
    height: int
    fps: float


def probe_video(path: Path) -> VideoProbe:
    """Open a video and decode its first and last frames, without reading it all."""
    with quiet_opencv_logs:
        capture = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    try:
        if not capture.isOpened():
            return VideoProbe(False, 0, 0, 0, 0.0)
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        first_ok = capture.read()[0]
        last_ok = first_ok
        if first_ok and frames > 1:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frames - 1)
            last_ok = capture.read()[0]
        return VideoProbe(bool(first_ok and last_ok), frames, width, height, fps)
    except cv2.error as exc:
        logger.warning("could not check %s: %s", path.name, exc)
        return VideoProbe(False, 0, 0, 0, 0.0)
    finally:
        capture.release()


def choose_container_fps(reported: float, measured: float, requested: float) -> float:
    """The frame rate to store in the file, so the video plays at the right speed.

    Webcams often report 30 fps and deliver far less in dim light. The measured
    rate wins when it clearly disagrees or nothing usable was reported.
    """

    def plausible(fps: float) -> bool:
        return 1.0 <= fps <= MAX_PLAUSIBLE_FPS

    if plausible(measured) and (not plausible(reported) or abs(measured - reported) > FPS_MISMATCH_RATIO * reported):
        return round(measured, 2)
    return float(reported) if plausible(reported) else float(requested)


def _take_stem(step_number: int, step_key: str, take: int) -> str:
    return f"step{step_number:02d}_{step_key}_take{take:02d}"


_TAKE_NAME = re.compile(r"step(\d+)_([a-z0-9_]+)_take(\d+)")


def next_take_number(folder: Path, step_number: int, step_key: str) -> int:
    """One past the highest take of this step in the folder, so nothing is ever overwritten."""
    highest = 0
    for path in folder.glob(f"step{step_number:02d}_{step_key}_take*"):
        match = _TAKE_NAME.fullmatch(path.stem)
        if match is not None:
            highest = max(highest, int(match.group(3)))
    return highest + 1


@dataclass(frozen=True)
class TakeResult:
    """A finished take, as shown to her and as read back from a sidecar."""

    status: str  # "completed", "aborted", "interrupted" or "failed": see the module docstring
    take: int
    video_name: str | None
    sidecar_name: str | None
    frames_written: int
    dropped_frames: int
    width: int
    height: int
    fps: float
    duration_s: float
    verified: bool
    lossless: bool
    error: str | None
    # "manual", "target", "retake", "close", "camera_lost" or "error"
    stop_reason: str = "manual"


def load_completed_takes(folder: Path) -> dict[int, TakeResult]:
    """{step number: its newest completed take whose video is still there}, to continue a session."""
    results: dict[int, TakeResult] = {}
    for path in sorted(folder.glob("step*_take*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("status") != "completed":
                continue
            number = int(data["step"]["number"])
            if not (folder / data["video"]["file"]).is_file():
                logger.warning("ignoring %s: its video is gone", path.name)
                continue
            result = TakeResult(
                status="completed",
                take=int(data["take"]),
                video_name=data["video"]["file"],
                sidecar_name=path.name,
                frames_written=int(data["frames"]["written"]),
                dropped_frames=int(data["frames"]["dropped_total"]),
                width=int(data["resolution"]["width"]),
                height=int(data["resolution"]["height"]),
                fps=float(data["fps"]["container"]),
                duration_s=float(data["timestamps"]["duration_s"]),
                verified=bool(data["verified"]["ok"]),
                lossless=bool(data["video"]["lossless"]),
                error=None,
                stop_reason=str(data.get("stop_reason", "manual")),
            )
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            logger.warning("ignoring unreadable sidecar %s: %s", path.name, exc)
            continue
        if number not in results or result.take > results[number].take:
            results[number] = result
    return results


# --------------------------------------------------------------------------
# One take: the writer thread
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TakeSpec:
    folder: Path
    session: Session
    condition: Condition
    step: ScriptStep
    step_number: int


@dataclass(frozen=True)
class TakeStats:
    take: int
    elapsed_s: float
    captured: int
    written: int
    dropped_queue_full: int
    dropped_camera_gaps: int
    pending: int
    free_bytes: int | None
    camera_stalled: int = 0  # the part of dropped_camera_gaps that was an outage, not a lower frame rate
    frame_age_s: float = 0.0  # since the camera last delivered a frame
    write_age_s: float = 0.0  # since the last frame went to disk

    @property
    def dropped_total(self) -> int:
        return self.dropped_queue_full + self.dropped_camera_gaps

    @property
    def lost_ratio(self) -> float:
        """The share of the take's frames that is really missing, a stall still going on included."""
        lost = self.camera_stalled + self.dropped_queue_full
        return lost / max(self.captured + self.camera_stalled, 1)


def _wall_clock() -> datetime:
    return datetime.now().astimezone()


def _iso(moment: datetime | None) -> str | None:
    return None if moment is None else moment.isoformat(timespec="milliseconds")


_QUOTED_PATH = re.compile(r"""(['"])(?:[A-Za-z]:[\\/]|\\\\|/)[^'"]*\1""")
_BARE_PATH = re.compile(r"""(?:[A-Za-z]:[\\/]|\\\\)[^\s'"）)]*""")


def _strip_paths(text: str) -> str:
    """Take file paths out of a message: in a sidecar they would reveal the user name."""
    return _BARE_PATH.sub("（パス）", _QUOTED_PATH.sub("（パス）", text))


def _error_detail(exc: BaseException) -> str:
    """A short description of an exception for the screen; ``_sidecar_error`` cleans it for the file."""
    if isinstance(exc, OSError):
        return _os_error_reason(exc)
    return str(exc) or type(exc).__name__


class FrameGapCounter:
    """Frames the camera failed to deliver, worked out from capture times alone.

    Timing jitter is not a loss: a late frame is followed by an early one, and a
    driver may buffer through a short stall and deliver in a burst. So three
    things count, each less any burst right after it:

    - Stalls: gaps far longer than a frame period. They are what can make a take
      unusable, so they are also reported on their own.
    - Single missing frames: gaps of about two periods.
    - A lower frame rate: a camera that slows down during a take (exposure in dim
      light) has an average gap clearly longer than the nominal period. The
      footage itself is fine, but it plays too fast and she should know.

    The limits for the first two follow the spread of this camera's gaps, so
    the more jitter it has, the longer a gap must be to count. A camera settling
    at a lower rate soon has the spread to no longer count as stalling either.

    Not thread-safe: the caller holds the lock.
    """

    def __init__(self, fps: float, started_at: float = 0.0) -> None:
        self._period = 1.0 / fps
        self._started_at = started_at
        self._previous: float | None = None
        self._stalled = 0.0
        self._singles = 0.0
        self._burst_left = 0.0  # frames of the last gap that may still arrive late
        self._burst_after_stall = False
        self._spread = 0.5 * self._period  # mean deviation of the gaps; starts out assuming jitter
        self._steady_s = 0.0
        self._steady_gaps = 0

    def _limit(self, periods: float) -> float:
        return max(periods * self._period, self._period + CAMERA_OUTLIER_SPREADS * self._spread)

    def add(self, captured_at: float) -> None:
        first = self._previous is None
        gap = captured_at - (self._started_at if self._previous is None else self._previous)
        self._previous = captured_at
        period = self._period
        if gap > self._limit(CAMERA_STALL_PERIODS):
            self._owe(gap / period - 1.0, stall=True)
        elif first:
            return  # the wait for the first frame says nothing about the timing
        elif self._burst_left > 0.0 and gap < CAMERA_BURST_PERIODS * period:
            self._credit(min(self._burst_left, 1.0 - gap / period))
        elif gap > self._limit(CAMERA_SINGLE_PERIODS):
            self._owe(gap / period - 1.0, stall=False)
        else:
            self._burst_left = 0.0
            self._steady_s += gap
            self._steady_gaps += 1
        if not first:
            deviation = min(abs(gap - period), CAMERA_SPREAD_MAX_PERIODS * period)
            self._spread += (deviation - self._spread) * CAMERA_SPREAD_RATE

    def _owe(self, frames: float, *, stall: bool) -> None:
        if stall:
            self._stalled += frames
        else:
            self._singles += frames
        self._burst_left = frames
        self._burst_after_stall = stall

    def _credit(self, frames: float) -> None:
        if self._burst_after_stall:
            self._stalled -= frames
        else:
            self._singles -= frames
        self._burst_left -= frames

    def age(self, now: float) -> float:
        """Seconds since the last frame, or since the start before the first one."""
        return now - (self._started_at if self._previous is None else self._previous)

    def stalled(self, now: float) -> int:
        """Frames lost in stalls so far, the one still going on included."""
        gap = self.age(now)
        ongoing = gap / self._period - 1.0 if gap > self._limit(CAMERA_STALL_PERIODS) else 0.0
        return max(0, round(self._stalled + ongoing))

    def lost(self, now: float) -> int:
        """All frames missing so far, for the drop count."""
        slow = 0.0
        if self._steady_gaps:
            longest = self._period * (1.0 + CAMERA_RATE_TOLERANCE)
            slow = max(0.0, self._steady_s / longest - self._steady_gaps)
        return self.stalled(now) + round(self._singles + slow)


class TakeRecorder:
    """Writes one take on its own thread; ``offer`` never blocks the caller.

    ``request_stop`` closes the intake, then everything already queued is
    written, the file is checked and the sidecar finalized. ``done`` turns true
    and ``result`` is set whatever happened; nothing here raises to the caller.
    """

    def __init__(
        self,
        spec: TakeSpec,
        *,
        fps: float,
        camera: dict[str, Any],
        writer_factory: WriterFactory = cv2_video_writer,
        formats: Sequence[VideoFormat] = VIDEO_FORMATS,
        queue_bytes: int = WRITE_QUEUE_BYTES,
    ) -> None:
        self._spec = spec
        self._fps = fps
        self._camera = camera
        self._writer_factory = writer_factory
        self._formats = tuple(formats)
        self._queue_bytes = queue_bytes
        self._queue: queue.SimpleQueue[tuple[np.ndarray, float]] = queue.SimpleQueue()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._done = threading.Event()
        self._thread = threading.Thread(target=self._run, name="record-writer", daemon=True)
        self._result: TakeResult | None = None

        # Guarded by _lock.
        self._intake_open = True
        self._pending_bytes = 0
        self._pending_frames = 0
        self._captured = 0
        self._written = 0
        self._dropped_queue_full = 0
        self._stop_reason = "manual"
        self._error: str | None = None
        self._take = 0
        self._free_bytes: int | None = None
        self._gaps = FrameGapCounter(fps)
        self._stopped_t: float | None = None
        self._last_write_t = 0.0

        # Used by the writer thread only.
        self._writer: VideoWriterLike | None = None
        self._format: VideoFormat | None = None
        self._video_path: Path | None = None
        self._sidecar_path: Path | None = None
        self._frame_size: tuple[int, int] | None = None
        self._written_size: tuple[int, int] = (0, 0)
        self._frame_offsets_ms: list[float] = []
        self._next_housekeeping = 0.0
        self._raw_bytes_written = 0
        self._file_size = 0
        self._file_grew_t = 0.0
        self._raw_bytes_at_growth = 0
        self._first_frame_t: float | None = None
        self._last_frame_t: float | None = None
        self._verified: dict[str, Any] = {"ok": False}

        self._started_t = _now()
        self._started_wall = _wall_clock()

    # ---- used by the GUI and capture threads ------------------------------

    def start(self) -> None:
        with self._lock:
            self._started_t = self._last_write_t = _now()
            self._gaps = FrameGapCounter(self._fps, self._started_t)
        self._started_wall = _wall_clock()
        self._thread.start()

    def offer(self, frame: np.ndarray, captured_at: float) -> None:
        """Queue a frame, or count it as dropped when the disk has fallen behind."""
        with self._lock:
            if not self._intake_open:
                return
            self._captured += 1
            self._gaps.add(captured_at)
            if self._pending_frames and self._pending_bytes + frame.nbytes > self._queue_bytes:
                self._dropped_queue_full += 1
                return
            self._pending_bytes += frame.nbytes
            self._pending_frames += 1
            # Inside the lock: once the intake is closed no frame can still be on its way.
            self._queue.put((frame, captured_at))

    def request_stop(self, reason: str = "manual") -> None:
        with self._lock:
            if self._intake_open:
                self._intake_open = False
                self._stop_reason = reason
                self._stopped_t = _now()
        self._stop.set()

    def stats(self) -> TakeStats:
        with self._lock:
            # While the take runs the time without frames counts too: an outage shows up as it happens.
            end = self._stopped_t if self._stopped_t is not None else _now()
            return TakeStats(
                take=self._take,
                elapsed_s=max(0.0, end - self._started_t),
                captured=self._captured,
                written=self._written,
                dropped_queue_full=self._dropped_queue_full,
                dropped_camera_gaps=self._gaps.lost(end),
                pending=self._pending_frames,
                free_bytes=self._free_bytes,
                camera_stalled=self._gaps.stalled(end),
                frame_age_s=max(0.0, self._gaps.age(end)),
                write_age_s=max(0.0, end - self._last_write_t),
            )

    @property
    def done(self) -> bool:
        return self._done.is_set()

    @property
    def result(self) -> TakeResult | None:
        return self._result

    def wait(self, timeout_s: float | None = None) -> bool:
        return self._done.wait(timeout_s)

    # ---- writer thread ----------------------------------------------------

    def _run(self) -> None:
        try:
            self._prepare()
            self._write_loop()
        except Exception as exc:  # the take fails; the window and the camera carry on
            logger.exception("recording failed")
            self._fail(f"録画中に問題が起きました。撮り直してください。（詳細: {_error_detail(exc)}）")
        finally:
            try:
                self._finish()
            except Exception as exc:  # the window waits for ``done``: it must come whatever happens
                logger.exception("could not finish the take")
                self._fail(f"録画を保存できませんでした。撮り直してください。（詳細: {_error_detail(exc)}）")
                self._result = self._build_result("failed")
            finally:
                self._done.set()

    def _prepare(self) -> None:
        spec = self._spec
        try:
            # Checked again at the last moment: the folder may have become a repository since.
            validate_output_dir(spec.folder)
            prepare_output_dir(spec.folder)
        except OutputFolderError as exc:
            self._fail(str(exc))
            return
        take = next_take_number(spec.folder, spec.step_number, spec.step.key)
        with self._lock:
            self._take = take
        self._sidecar_path = spec.folder / f"{_take_stem(spec.step_number, spec.step.key, take)}.json"
        # Written now so that a power cut leaves a trace saying the video is incomplete.
        self._write_sidecar("recording")

    def _write_loop(self) -> None:
        while True:
            # Read before the queue: once this is true nothing more arrives, so empty means finished.
            stopping = self._stop.is_set()
            try:
                frame, captured_at = self._queue.get(timeout=0.05) if not stopping else self._queue.get_nowait()
            except queue.Empty:
                if stopping:
                    return
                self._housekeeping()
                continue
            self._write_frame(frame, captured_at)
            self._housekeeping()

    def _write_frame(self, frame: np.ndarray, captured_at: float) -> None:
        try:
            if self._error is not None:  # failed earlier: only empty the queue
                return
            if self._frame_size is None:
                self._open_video(frame)
                if self._error is not None:
                    return
            if (frame.shape[1], frame.shape[0]) != self._frame_size:
                self._fail("カメラの解像度が途中で変わったので、録画を止めました。撮り直してください。")
                return
            assert self._writer is not None
            width, height = self._written_size
            self._writer.write(frame[:height, :width])
            if self._first_frame_t is None:
                self._first_frame_t = captured_at
            self._last_frame_t = captured_at
            self._frame_offsets_ms.append(round((captured_at - self._first_frame_t) * 1000.0, 1))
            self._raw_bytes_written += frame.nbytes
            with self._lock:
                self._written += 1
                self._last_write_t = _now()
        finally:
            with self._lock:
                self._pending_bytes -= frame.nbytes
                self._pending_frames -= 1

    def _open_video(self, frame: np.ndarray) -> None:
        height, width = frame.shape[:2]
        self._frame_size = (width, height)
        # FFmpeg's writer needs even dimensions; it would cut the last row and column anyway.
        self._written_size = (width & ~1, height & ~1)
        stem = _take_stem(self._spec.step_number, self._spec.step.key, self.stats().take)
        for video_format in self._formats:
            path = self._spec.folder / f"{stem}{video_format.extension}"
            with quiet_opencv_logs:
                writer = self._writer_factory(path, video_format.fourcc, self._fps, self._written_size)
            if writer.isOpened():
                self._writer, self._format, self._video_path = writer, video_format, path
                logger.info("recording %s as %s, %dx%d", path.name, video_format.key, width, height)
                return
            writer.release()
            path.unlink(missing_ok=True)
        self._fail("動画ファイルを作れませんでした。保存先のフォルダと空き容量を確認してください。")

    def _housekeeping(self) -> None:
        """Once a second: stop cleanly when the disk is full, the drive is gone or the camera is.

        ``cv2.VideoWriter.write`` reports no errors, so this is the only way to notice.
        """
        now = _now()
        if now < self._next_housekeeping or self._error is not None:
            return
        self._next_housekeeping = now + HOUSEKEEPING_INTERVAL_S
        with self._lock:
            # Not once stopped: then only the queue is being emptied, however long a slow disk takes.
            camera_gone = self._stopped_t is None and self._gaps.age(now) >= CAMERA_LOST_STOP_S
        if camera_gone:
            logger.warning("no frame for %.0f s; ending the take", CAMERA_LOST_STOP_S)
            self.request_stop("camera_lost")
            return
        try:
            free = shutil.disk_usage(self._spec.folder).free
            file_size = None if self._video_path is None else self._video_path.stat().st_size
        except OSError:
            self._fail("保存先のドライブに書き込めなくなったので、録画を止めました。ドライブを確認して撮り直してください。")
            return
        with self._lock:
            self._free_bytes = free
        if free < MIN_FREE_BYTES:
            self._fail("保存先の空き容量が少なくなったので、録画を止めました。容量を空けてから撮り直してください。")
        elif file_size is not None and self._file_has_stopped_growing(file_size, now):
            self._fail("動画ファイルが増えなくなったので、録画を止めました。保存先のドライブを確認して撮り直してください。")

    def _file_has_stopped_growing(self, file_size: int, now: float) -> bool:
        """The drive reports free space and the file is there, yet the frames are not arriving."""
        if file_size > self._file_size or not self._file_grew_t:
            self._file_size = file_size
            self._file_grew_t = now
            self._raw_bytes_at_growth = self._raw_bytes_written
            return False
        handed_over = self._raw_bytes_written - self._raw_bytes_at_growth
        return handed_over >= WRITE_STALL_MIN_BYTES and now - self._file_grew_t >= WRITE_STALL_S

    def _fail(self, message: str) -> None:
        if self._error is None:
            self._error = message
        logger.warning("take failed: %s", message)
        self.request_stop("error")

    def _finish(self) -> None:
        self._stop.set()
        video_error: str | None = None
        try:
            if self._writer is not None:
                self._writer.release()
            video_error = self._check_video()
        except Exception as exc:
            logger.exception("could not finish the video")
            video_error = f"動画ファイルを閉じられませんでした。撮り直してください。（詳細: {_error_detail(exc)}）"
        if self._error is None:
            self._error = video_error
        if self._error is None and self._stop_reason in ("manual", "target", "camera_lost"):
            self._error = self._lost_frames_error()
        status = self._final_status()
        self._write_sidecar(status)
        if self._error is not None:  # also when that was the sidecar itself
            status = "failed"
        self._result = self._build_result(status)

    def _check_video(self) -> str | None:
        """Open the finished file again and compare it with what was written."""
        written = self.stats().written
        path = self._video_path
        if path is None or written == 0:
            if path is not None:
                path.unlink(missing_ok=True)  # only a header, no footage
                self._video_path = None
            if self._stop_reason == "retake":
                return None  # stopped before the first frame came; she has moved on already
            return "映像が 1 フレームも届かなかったので、動画を保存できませんでした。カメラを確認してください。"
        probe = probe_video(path)
        width, height = self._written_size
        ok = probe.playable and probe.frames == written and (probe.width, probe.height) == (width, height)
        self._verified = {
            "ok": ok,
            "frames": probe.frames,
            "width": probe.width,
            "height": probe.height,
            "fps": probe.fps,
        }
        return None if ok else "保存した動画を確認できませんでした（再生できないか、フレーム数が合いません）。撮り直してください。"

    def _lost_frames_error(self) -> str | None:
        """Such a take must not pass for finished: the newest completed take is the one that gets used."""
        stats = self.stats()
        lost = stats.camera_stalled + stats.dropped_queue_full
        if stats.lost_ratio <= MAX_LOST_RATIO or lost < MIN_LOST_TO_FAIL_S * self._fps:
            return None
        percent = round(stats.lost_ratio * 100)
        if stats.camera_stalled >= stats.dropped_queue_full:
            return (
                f"カメラの映像が途切れて、全体の {percent}% が欠けました。このテイクは使いません。"
                "カメラの接続を確認して撮り直してください。"
            )
        return (
            f"保存が間に合わず、映像の {percent}% が欠けました。このテイクは使いません。"
            "ほかのアプリを閉じるか、速いドライブに変えて撮り直してください。"
        )

    def _stopped_too_soon(self) -> bool:
        limit = min(MIN_USABLE_TAKE_S, self._spec.step.target_s / 2)
        return self._stop_reason == "manual" and self._duration_s() < limit

    def _final_status(self) -> str:
        if self._error is not None:
            return "failed"
        if self._stop_reason == "close":
            return "interrupted"
        return "aborted" if self._stop_reason == "retake" or self._stopped_too_soon() else "completed"

    def _build_result(self, status: str) -> TakeResult:
        stats = self.stats()
        width, height = self._written_size
        return TakeResult(
            status=status,
            take=stats.take,
            video_name=None if self._video_path is None else self._video_path.name,
            sidecar_name=None if self._sidecar_path is None else self._sidecar_path.name,
            frames_written=stats.written,
            dropped_frames=stats.dropped_total,
            width=width,
            height=height,
            fps=self._fps,
            duration_s=self._duration_s(),
            verified=bool(self._verified.get("ok")),
            lossless=bool(self._format and self._format.lossless),
            error=self._error,
            stop_reason=self._stop_reason,
        )

    def _duration_s(self) -> float:
        if self._first_frame_t is None or self._last_frame_t is None:
            return 0.0
        return round(self._last_frame_t - self._first_frame_t, 3)

    # ---- sidecar ----------------------------------------------------------

    def _wall_at(self, moment: float | None) -> datetime | None:
        return None if moment is None else self._started_wall + timedelta(seconds=moment - self._started_t)

    def _sidecar_data(self, status: str) -> dict[str, Any]:
        spec = self._spec
        stats = self.stats()
        width, height = self._written_size
        return {
            "schema_version": SIDECAR_SCHEMA_VERSION,
            "tool": "makeup_cam.record",
            "status": status,
            "stop_reason": self._stop_reason,
            "error": self._sidecar_error(),
            "purpose": spec.session.purpose,
            "session": {"key": spec.session.key, "label": spec.session.label, "lighting": spec.session.lighting},
            "condition": {"key": spec.condition.key, "label": spec.condition.label},
            "step": {
                "number": spec.step_number,
                "key": spec.step.key,
                "title": spec.step.title,
                "instruction": spec.step.instruction,
                "target_s": spec.step.target_s,
                "cues": [{"start_s": start, "text": text} for start, text in cue_schedule(spec.step)],
            },
            "take": stats.take,
            "camera": self._camera,
            "video": {
                "file": None if self._video_path is None else self._video_path.name,
                "codec": None if self._format is None else self._format.fourcc,
                "lossless": bool(self._format and self._format.lossless),
                "audio": False,
                "mirrored": False,
            },
            "resolution": {"width": width, "height": height},
            "fps": {
                "container": self._fps,
                "measured": self._measured_fps(),
            },
            "timestamps": {
                "recording_started_at": _iso(self._started_wall),
                "first_frame_at": _iso(self._wall_at(self._first_frame_t)),
                "last_frame_at": _iso(self._wall_at(self._last_frame_t)),
                "recording_stopped_at": _iso(self._wall_at(self._stopped_t)),
                "duration_s": self._duration_s(),
                # Capture time of each written frame, relative to the first one.
                "frame_offsets_ms": "@FRAME_OFFSETS@",
            },
            "frames": {
                "captured": stats.captured,
                "written": stats.written,
                "dropped_queue_full": stats.dropped_queue_full,
                "dropped_camera_gaps": stats.dropped_camera_gaps,
                "dropped_total": stats.dropped_total,
                "camera_stalled": stats.camera_stalled,
                "lost_ratio": round(stats.lost_ratio, 4),
            },
            "verified": self._verified,
        }

    def _sidecar_error(self) -> str | None:
        """The message as shown on screen may name the folder; the sidecar must not."""
        if self._error is None:
            return None
        text = self._error
        for known in (str(self._spec.folder), str(Path.home())):
            text = text.replace(known, "（パス）")
        text = re.sub(r"""（パス）[\\/][^\s'"）)]*""", "（パス）", text)
        return _strip_paths(text)

    def _measured_fps(self) -> float | None:
        offsets = self._frame_offsets_ms
        if len(offsets) < 2 or offsets[-1] <= 0:
            return None
        return round((len(offsets) - 1) / (offsets[-1] / 1000.0), 3)

    def _write_sidecar(self, status: str) -> None:
        path = self._sidecar_path
        if path is None:
            return
        text = json.dumps(self._sidecar_data(status), ensure_ascii=False, indent=2)
        # One compact line instead of one line per frame.
        text = text.replace('"@FRAME_OFFSETS@"', json.dumps(self._frame_offsets_ms, separators=(",", ":")))
        temporary = path.with_suffix(".json.tmp")
        try:
            temporary.write_text(text + "\n", encoding="utf-8")
            self._replace_file(temporary, path)
        except OSError as exc:
            logger.warning("could not write %s: %s", path.name, exc)
            self._fail(f"記録ファイル（JSON）を保存できませんでした。保存先を確認してください。（{_os_error_reason(exc)}）")

    @staticmethod
    def _replace_file(temporary: Path, path: Path) -> None:
        """A virus scanner or sync client may hold the old file for a moment."""
        for attempt in range(4):
            try:
                os.replace(temporary, path)
                return
            except PermissionError:
                if attempt == 3:
                    raise
                time.sleep(0.1)


# --------------------------------------------------------------------------
# The camera: the capture thread
# --------------------------------------------------------------------------


class FrameSource(Protocol):
    """What ``CameraSource`` and ``VideoFileSource`` have in common.

    ``read`` must return a new array each time; frames are queued without copying.
    """

    actual_width: int
    actual_height: int
    actual_fps: float

    def open(self) -> None: ...
    def read(self) -> np.ndarray | None: ...
    def close(self) -> None: ...


SourceFactory = Callable[[int], FrameSource]


@dataclass(frozen=True)
class CaptureSnapshot:
    state: str  # "idle", "opening", "running", "no_frames" or "failed"
    message: str
    preview: np.ndarray | None
    preview_seq: int
    width: int
    height: int
    fps: float


def camera_description(source: FrameSource) -> dict[str, Any]:
    """For the sidecar. Names only: the full path of a test file would reveal the user name."""
    description: dict[str, Any] = {
        "type": type(source).__name__,
        "reported_width": int(getattr(source, "actual_width", 0)),
        "reported_height": int(getattr(source, "actual_height", 0)),
        "reported_fps": float(getattr(source, "actual_fps", 0.0)),
    }
    for attribute in ("index", "backend"):
        value = getattr(source, attribute, None)
        if value is not None:
            description[attribute] = value
    path = getattr(source, "path", None)
    if path is not None:
        description["file_name"] = Path(path).name
    return description


def steady_fps(times: Iterable[float]) -> float:
    """The frame rate from capture times, leaving out stalls and the burst of late frames after them.

    A hiccup just before a take must not set the speed of the whole video. Only
    the two together may go: dropping only the stalls would overstate the rate
    after a burst, and dropping every short gap would understate it under jitter.
    """
    gaps = [later - earlier for earlier, later in pairwise(times)]
    if not gaps:
        return 0.0
    typical = statistics.median(gaps)
    deviation = statistics.median(abs(gap - typical) for gap in gaps)
    limit = max(CAMERA_STALL_PERIODS * typical, typical + CAMERA_OUTLIER_MEDIAN_DEVIATIONS * deviation)
    steady_s, steady_gaps = 0.0, 0
    after_stall = False
    for gap in gaps:
        if gap > limit:
            after_stall = True
        elif not after_stall or gap >= CAMERA_BURST_PERIODS * typical:
            after_stall = False
            steady_s += gap
            steady_gaps += 1
    return steady_gaps / steady_s if steady_s > 0 else 0.0


class CaptureEngine:
    """Runs the source on a worker thread; every public method returns immediately."""

    def __init__(
        self,
        source_factory: SourceFactory,
        *,
        writer_factory: WriterFactory = cv2_video_writer,
        formats: Sequence[VideoFormat] = VIDEO_FORMATS,
        requested_fps: float = 30.0,
        preview_width: int = PREVIEW_WIDTH,
    ) -> None:
        self._source_factory = source_factory
        self._writer_factory = writer_factory
        self._formats = tuple(formats)
        self._requested_fps = requested_fps
        self._preview_width = preview_width
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop: threading.Event | None = None
        self._recorder: TakeRecorder | None = None
        # Guarded by _lock.
        self._state = "idle"
        self._message = ""
        self._preview: np.ndarray | None = None
        self._preview_seq = 0
        self._frame_size = (0, 0)
        self._last_frame_t = 0.0
        self._frame_times: deque[float] = deque(maxlen=FPS_WINDOW_FRAMES)
        self._camera: dict[str, Any] = {}
        self._reported_fps = 0.0

    def start(self, camera_index: int) -> None:
        """(Re)start capturing. The new thread takes over the device when the old one lets go."""
        self.stop()
        previous = self._thread
        stop = threading.Event()
        with self._lock:
            self._stop = stop
            self._state = "opening"
            self._message = ""
            self._preview = None
            self._frame_size = (0, 0)
            self._frame_times.clear()
            self._camera = {}
            self._reported_fps = 0.0
        self._thread = threading.Thread(
            target=self._capture_loop, args=(camera_index, stop, previous), name="record-capture", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Ask the capture thread to stop; a running take is finished and saved."""
        self.end_take("close")
        if self._stop is not None:
            self._stop.set()

    def join(self, timeout_s: float) -> bool:
        thread = self._thread
        if thread is None:
            return True
        thread.join(timeout_s)
        return not thread.is_alive()

    def snapshot(self) -> CaptureSnapshot:
        now = _now()
        with self._lock:
            state = self._state
            if state == "running" and now - self._last_frame_t > NO_FRAME_WARNING_S:
                state = "no_frames"  # worked out here so that a hung read() is noticed too
            return CaptureSnapshot(
                state=state,
                message=self._message,
                preview=self._preview,
                preview_seq=self._preview_seq,
                width=self._frame_size[0],
                height=self._frame_size[1],
                fps=self._measured_fps(),
            )

    def begin_take(self, spec: TakeSpec) -> TakeRecorder:
        """Start a take. Disk work (folder, take number, file) happens on the writer thread."""
        with self._lock:
            fps = choose_container_fps(self._reported_fps, self._measured_fps(), self._requested_fps)
            camera = dict(self._camera)
        recorder = TakeRecorder(
            spec, fps=fps, camera=camera, writer_factory=self._writer_factory, formats=self._formats
        )
        recorder.start()
        self._recorder = recorder
        return recorder

    def end_take(self, reason: str = "manual") -> None:
        recorder, self._recorder = self._recorder, None
        if recorder is not None:
            recorder.request_stop(reason)

    # ---- capture thread ---------------------------------------------------

    def _measured_fps(self) -> float:
        times = self._frame_times
        if len(times) < FPS_MIN_FRAMES or times[-1] - times[0] < FPS_MIN_SPAN_S:
            return 0.0
        return steady_fps(times)

    def _set_state(self, state: str, message: str = "") -> None:
        with self._lock:
            self._state = state
            self._message = message

    def _capture_loop(self, camera_index: int, stop: threading.Event, previous: threading.Thread | None) -> None:
        if previous is not None:
            previous.join(CAMERA_SWITCH_WAIT_S)
        source = self._open_source(camera_index, stop)
        if source is None:
            return
        try:
            if not stop.is_set():
                self._read_frames(source, stop)
        finally:
            try:
                source.close()
            except Exception:
                logger.exception("closing the camera failed")
            with self._lock:
                if self._state in ("running", "opening") and stop.is_set() and self._stop is stop:
                    self._state = "idle"

    def _open_source(self, camera_index: int, stop: threading.Event) -> FrameSource | None:
        source: FrameSource | None = None
        try:
            source = self._source_factory(camera_index)
            source.open()
        except Exception as exc:
            if isinstance(exc, CameraError):
                message = str(exc)
            else:
                logger.exception("opening the camera failed")
                message = f"カメラを開けませんでした。（詳細: {exc}）"
            with self._lock:
                if self._stop is stop:  # not when this thread has been replaced in the meantime
                    self._state, self._message = "failed", message
            return None
        with self._lock:
            if self._stop is stop:
                self._camera = camera_description(source)
                self._reported_fps = float(getattr(source, "actual_fps", 0.0) or 0.0)
                self._state = "running"
                self._message = ""
                self._last_frame_t = _now()
        return source

    def _read_frames(self, source: FrameSource, stop: threading.Event) -> None:
        while not stop.is_set():
            try:
                frame = source.read()
            except Exception:
                logger.exception("reading from the camera failed")
                frame = None
            captured_at = _now()
            if frame is None:
                stop.wait(0.01)
                continue
            recorder = self._recorder
            if recorder is not None:
                recorder.offer(frame, captured_at)
            preview = self._make_preview(frame)
            with self._lock:
                self._preview = preview
                self._preview_seq += 1
                self._frame_size = (frame.shape[1], frame.shape[0])
                self._last_frame_t = captured_at
                self._frame_times.append(captured_at)

    def _make_preview(self, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        if width <= self._preview_width:
            return frame
        size = (self._preview_width, max(1, round(height * self._preview_width / width)))
        return cv2.resize(frame, size, interpolation=cv2.INTER_LINEAR)


# --------------------------------------------------------------------------
# Small display helpers
# --------------------------------------------------------------------------


def format_clock(seconds: float) -> str:
    total = max(0, int(seconds))
    return f"{total // 60:02d}:{total % 60:02d}"


def format_bytes(count: float) -> str:
    gigabytes = count / 1024**3
    if gigabytes >= 10:
        return f"{gigabytes:.0f} GB"
    if gigabytes >= 1:
        return f"{gigabytes:.1f} GB"
    return f"{count / 1024**2:.0f} MB"


def estimate_bytes(
    script: Sequence[ScriptStep], width: int, height: int, fps: float, video_format: VideoFormat = VIDEO_FORMATS[0]
) -> float:
    """Rough size of a whole session; real footage varies with noise and lighting."""
    seconds = sum(step.target_s for step in script)
    return seconds * fps * width * height * 3 * video_format.size_ratio


def bgr_to_qimage(frame: np.ndarray, *, mirror: bool = False) -> QImage:
    """A QImage owning a copy of the pixels."""
    pixels = np.ascontiguousarray(frame[:, ::-1] if mirror else frame)
    height, width = pixels.shape[:2]
    return QImage(pixels.data, width, height, pixels.strides[0], QImage.Format.Format_BGR888).copy()


def _beep() -> None:
    QApplication.beep()


def _sound_alarm() -> None:
    """Three quick beeps, unlike the single one of a normal start or stop: she may not be looking at the screen."""
    _beep()
    for delay_ms in (250, 500):
        QTimer.singleShot(delay_ms, _beep)


def _set_display_keep_awake(active: bool) -> None:
    """Keep the screen on during a take: the script must not vanish in a long talking step.

    Applies to this thread only and ends with the process; system settings stay as they are.
    """
    if sys.platform != "win32":
        return
    es_continuous, es_system_required, es_display_required = 0x80000000, 0x00000001, 0x00000002
    flags = es_continuous | (es_system_required | es_display_required if active else 0)
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(ctypes.c_uint32(flags))
    except (AttributeError, OSError) as exc:
        logger.debug("could not keep the display on: %s", exc)


# --------------------------------------------------------------------------
# Text and style
# --------------------------------------------------------------------------

# Japanese fonts first: on a non-Japanese Windows the fallback can pick Chinese glyph shapes.
_FONT_FAMILIES = ["Yu Gothic UI", "Meiryo UI", "Meiryo", "Hiragino Sans", "Noto Sans CJK JP", "sans-serif"]

STORAGE_ADVICE = (
    "<b>保存について（大切です）</b><br>"
    "これらは<b>あなたの顔そのもののデータ</b>です。悪用されるとディープフェイクを作れてしまうので、扱いに注意してください。"
    "<ul style='margin-left:-16px'>"
    "<li>公開リポジトリ（GitHub など）には<b>絶対に置かない</b>。Git の作業フォルダの中には保存できないようにしています</li>"
    "<li>外付けドライブやクラウドに置くなら、<b>暗号化する</b>（Windows の BitLocker、Mac の暗号化ディスクイメージなど）</li>"
    "<li>vast.ai などで学習するときは、データセンター型（Secure Cloud）のマシンを選び、終わったらすぐに削除する</li>"
    "</ul>"
    "このツールは<b>インターネットに何も送りません</b>。選んだフォルダの中に sessions（学習用）と eval（評価用）を作って"
    "保存するだけです。音声は保存しません。"
)

BEFORE_SHOOTING = (
    "<b>撮影の前に</b>"
    "<ul style='margin-left:-16px'>"
    "<li>面接で使う PC・Web カメラで、机・椅子・カメラの位置も本番と同じに（顔からカメラまで 50〜70 cm、胸から上）</li>"
    "<li>カメラの自動補正を切る（Windows Studio エフェクト、美肌フィルタ、ポートレートモード）</li>"
    "<li>できれば露出とホワイトバランスを固定する</li>"
    "<li>グレーカード（なければ白いコピー用紙）を手元に用意する</li>"
    "<li>ほかの人が映り込まないようにする</li>"
    "<li>撮影中はカメラと照明を動かさない。メイクで席を立つときもそのまま</li>"
    "</ul>"
)

_STYLE = """
QWidget { background: #202124; color: #e8eaed; }
QLineEdit, QComboBox, QSpinBox {
    background: #303134; border: 1px solid #5f6368; border-radius: 4px; padding: 6px;
}
QComboBox QAbstractItemView { background: #303134; selection-background-color: #1a73e8; }
QPushButton {
    background: #3c4043; border: 1px solid #5f6368; border-radius: 6px; padding: 10px 18px;
}
QPushButton:hover { background: #4a4e52; }
QPushButton:disabled { color: #80868b; background: #2a2b2e; border-color: #3c4043; }
QPushButton[accent="true"] { background: #1a73e8; border-color: #1a73e8; color: white; }
QPushButton[accent="true"]:disabled { color: #80868b; background: #2a2b2e; border-color: #3c4043; }
QPushButton[danger="true"] { background: #c5221f; border-color: #c5221f; color: white; }
QFrame[panel="true"] { background: #292a2d; border-radius: 8px; }
QFrame[panel="true"][alert="true"] { background: #5b1f1c; border: 2px solid #f28b82; }
QFrame[panel="true"] QLabel, QFrame[panel="true"] QCheckBox { background: transparent; }
QProgressBar { background: #303134; border: none; border-radius: 4px; max-height: 8px; }
QProgressBar::chunk { background: #8ab4f8; border-radius: 4px; }
QScrollBar:vertical { background: transparent; width: 10px; }
QScrollBar:horizontal { background: transparent; height: 10px; }
QScrollBar::handle { background: #5f6368; border-radius: 5px; }
QScrollBar::handle:vertical { min-height: 24px; }
QScrollBar::handle:horizontal { min-width: 24px; }
QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; }
QScrollBar::add-page, QScrollBar::sub-page { background: transparent; }
"""

_COLOR_MUTED = "#9aa0a6"
_COLOR_OK = "#81c995"
_COLOR_WARN = "#fdd663"
_COLOR_ERROR = "#f28b82"
_COLOR_ALERT_TEXT = "#ffffff"


def _font(size: float, *, bold: bool = False) -> QFont:
    font = QFont()
    font.setFamilies(_FONT_FAMILIES)
    font.setPointSizeF(size)
    font.setBold(bold)
    return font


def _label(text: str = "", size: float = 11, *, bold: bool = False, color: str | None = None) -> QLabel:
    label = QLabel(text)
    label.setFont(_font(size, bold=bold))
    label.setWordWrap(True)
    if color is not None:
        _set_color(label, color)
    return label


def _set_color(label: QLabel, color: str | None) -> None:
    style = f"color: {color};" if color else ""
    if label.styleSheet() != style:  # setting a style sheet costs a re-polish; this runs on every tick
        label.setStyleSheet(style)


def _panel() -> QFrame:
    frame = QFrame()
    frame.setProperty("panel", True)
    return frame


def _scrolling(content: QWidget) -> QScrollArea:
    """On a small screen the content scrolls, so that the window still fits and its buttons stay visible."""
    area = QScrollArea()
    area.setWidget(content)
    area.setWidgetResizable(True)
    area.setFrameShape(QFrame.Shape.NoFrame)
    area.setFocusPolicy(Qt.FocusPolicy.NoFocus)  # Space must stay with the shortcut
    area.setMinimumWidth(content.minimumSizeHint().width() + area.verticalScrollBar().sizeHint().width())
    return area


class CueLabel(QLabel):
    """The big script line. A long text gets a smaller font instead of more lines, so the layout stays put."""

    SIZES = (32.0, 28.0, 24.0, 20.0)

    def __init__(self) -> None:
        super().__init__()
        self.setFont(_font(self.SIZES[0], bold=True))
        self.setWordWrap(True)
        self.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self._box_height = self.fontMetrics().lineSpacing() * 2
        self.setMinimumHeight(self._box_height)
        self._fitted: tuple[str, int] | None = None

    def setText(self, text: str) -> None:
        super().setText(text)
        self._fit()

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self._fit()

    def _fit(self) -> None:
        key = (self.text(), self.width())
        if key == self._fitted:
            return
        self._fitted = key
        font = self.font()
        for size in self.SIZES:
            font = _font(size, bold=True)
            flags = int(Qt.TextFlag.TextWordWrap)
            needed = QFontMetrics(font).boundingRect(0, 0, max(1, self.width()), 100000, flags, self.text())
            if needed.height() <= self._box_height:
                break
        if font.pointSizeF() != self.font().pointSizeF():
            self.setFont(font)


class PreviewView(QWidget):
    """The live picture with the countdown, the recording mark and camera messages on top."""

    def __init__(self) -> None:
        super().__init__()
        self._image: QImage | None = None
        self._countdown = ""
        self._message = ""
        self._recording = False
        self._stalled = False
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.setMinimumSize(320, 180)

    @property
    def image(self) -> QImage | None:
        return self._image

    def set_image(self, image: QImage | None) -> None:
        self._image = image
        self.update()

    def set_overlay(self, *, countdown: str = "", message: str = "", recording: bool = False, stalled: bool = False) -> None:
        """``stalled``: the picture is the last frame the camera delivered, not what it sees now."""
        state = (countdown, message, recording, stalled)
        if state != (self._countdown, self._message, self._recording, self._stalled):
            self._countdown, self._message, self._recording, self._stalled = state
            self.update()

    def paintEvent(self, _event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#000000"))
        if self._image is not None and not self._image.isNull():
            scaled = self._image.size().scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio)
            x, y = (self.width() - scaled.width()) / 2, (self.height() - scaled.height()) / 2
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
            painter.drawImage(QRectF(x, y, scaled.width(), scaled.height()), self._image)
        if self._stalled:
            painter.fillRect(self.rect(), QColor(0, 0, 0, 170))
        if self._recording:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(0, 0, 0, 150))
            painter.drawRoundedRect(8, 8, 116, 30, 8, 8)
            painter.setBrush(QColor(_COLOR_WARN if self._stalled else "#ea4335"))
            painter.drawEllipse(14, 14, 18, 18)
            painter.setPen(QColor("#ffffff"))
            painter.setFont(_font(12, bold=True))
            painter.drawText(40, 8, 80, 30, int(Qt.AlignmentFlag.AlignVCenter), "映像なし" if self._stalled else "録画中")
        if self._countdown:
            # In a corner: the face stays visible while she settles into position.
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            radius = max(36.0, self.height() * 0.14)
            circle = QRectF(self.width() - 2 * radius - 12, 12, 2 * radius, 2 * radius)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(0, 0, 0, 170))
            painter.drawEllipse(circle)
            painter.setPen(QColor("#ffffff"))
            font = _font(11, bold=True)
            font.setPixelSize(round(radius * 1.3))
            painter.setFont(font)
            painter.drawText(circle, int(Qt.AlignmentFlag.AlignCenter), self._countdown)
        if self._message:
            band = self.rect().adjusted(0, self.height() - 64, 0, 0)
            painter.fillRect(band, QColor(0, 0, 0, 180))
            painter.setPen(QColor(_COLOR_WARN))
            painter.setFont(_font(12, bold=True))
            flags = int(Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap)
            painter.drawText(band.adjusted(12, 0, -12, 0), flags, self._message)
        painter.end()


class Phase(Enum):
    READY = "ready"  # this step has no take yet
    COUNTDOWN = "countdown"
    RECORDING = "recording"
    SAVING = "saving"  # finishing the file on the writer thread
    REVIEW = "review"  # this step has a take; go on or retake it
    FAILED = "failed"  # the last take of this step failed; retaking it is the way forward
    DONE = "done"  # past the last step


# --------------------------------------------------------------------------
# The window
# --------------------------------------------------------------------------


class RecorderWindow(QMainWindow):
    """Setup page (folder, session, camera), then the step-by-step recording page.

    Timing and camera state are all polled in ``_tick``, so nothing here waits
    for the camera. Disk access for this thread is limited to the setup page and
    the moment a session starts, and runs on a helper thread with a time limit.
    """

    def __init__(
        self,
        source_factory: SourceFactory,
        *,
        output: str | None = None,
        camera_index: int = 0,
        camera_label: str | None = None,
        countdown_s: float = COUNTDOWN_S,
        capture_size: tuple[int, int] = (GUIDE_MIN_WIDTH, GUIDE_MIN_HEIGHT),
        capture_fps: float = 30.0,
        writer_factory: WriterFactory = cv2_video_writer,
        preferred_format: str | None = None,
        script_provider: Callable[[Session, Condition], Sequence[ScriptStep]] = script_for,
        today: Callable[[], date] = date.today,
    ) -> None:
        super().__init__()
        self._countdown_s = countdown_s
        self._capture_size = capture_size
        self._capture_fps = capture_fps
        self._script_provider = script_provider
        self._today = today
        self._formats = video_format_order(preferred_format)
        self._engine = CaptureEngine(
            source_factory, writer_factory=writer_factory, formats=self._formats, requested_fps=capture_fps
        )

        self._phase = Phase.READY
        self._session = SESSIONS[0]
        self._condition = CONDITIONS[0]
        self._script: tuple[ScriptStep, ...] = ()
        self._step_index = 0
        self._folder: Path | None = None
        self._recorder: TakeRecorder | None = None
        self._results: dict[int, TakeResult] = {}
        self._failures: dict[int, TakeResult] = {}  # steps whose last take failed
        self._notes: dict[int, tuple[str, str]] = {}  # what to say about the last attempt at a step
        self._alert_on = False
        self._next_alert_beep = 0.0
        self._saving_progress: tuple[int, int] | None = None
        self._saving_progress_at = 0.0
        self._fitted_to_screen = False
        self._countdown_ends_at = 0.0
        self._retake_after_save = False
        self._closing_deadline: float | None = None
        self._force_close = False
        self._preview_seq = -1
        self._output_refused = False

        self.setWindowTitle(APP_TITLE)
        self.setStyleSheet(_STYLE)
        self.setFont(_font(11))
        self._stack = QStackedWidget()
        self.setCentralWidget(self._stack)
        self._build_setup_page(camera_index, camera_label)
        self._build_record_page()
        self._stack.addWidget(self.setup_page)
        self._stack.addWidget(self.record_page)
        self._build_shortcuts()
        self._set_shortcuts_enabled(False)
        self.resize(1200, 780)

        self._timer = QTimer(self)
        self._timer.setInterval(UI_INTERVAL_MS)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

        if output:
            self.output_edit.setText(output)
            self.check_output()

    # ---- read by tests and by the session flow ----------------------------

    @property
    def phase(self) -> Phase:
        return self._phase

    @property
    def step_index(self) -> int:
        return self._step_index

    @property
    def results(self) -> dict[int, TakeResult]:
        return self._results

    @property
    def session_dir(self) -> Path | None:
        return self._folder

    @property
    def engine(self) -> CaptureEngine:
        return self._engine

    # ---- setup page -------------------------------------------------------

    def _build_setup_page(self, camera_index: int, camera_label: str | None) -> None:
        self.setup_page = QWidget()
        outer = QHBoxLayout(self.setup_page)
        outer.setContentsMargins(24, 20, 24, 20)
        outer.setSpacing(20)

        left_content = QWidget()
        left = QVBoxLayout(left_content)
        left.setContentsMargins(0, 0, 0, 0)
        left.setSpacing(12)
        left.addWidget(_label("学習用の動画を撮影します", 20, bold=True))
        left.addWidget(_label("台本を順番に画面に出します。ステップごとに別の動画ファイルとして保存します。", 11, color=_COLOR_MUTED))

        left.addWidget(_label("1. 保存先のフォルダ（必須）", 12, bold=True))
        row = QHBoxLayout()
        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("例: E:\\capture（暗号化したドライブがおすすめ）")
        self.output_edit.textChanged.connect(self._on_output_edited)
        self.output_edit.editingFinished.connect(self.check_output)
        self.browse_button = QPushButton("参照…")
        self.browse_button.clicked.connect(self.browse_output)
        row.addWidget(self.output_edit, 1)
        row.addWidget(self.browse_button)
        left.addLayout(row)
        self.output_message = _label("", 10.5)
        self.output_message.setTextFormat(Qt.TextFormat.PlainText)
        left.addWidget(self.output_message)
        self.cloud_ack = QCheckBox("同期されることを理解したうえで、このフォルダに保存します")
        self.cloud_ack.setVisible(False)
        self.cloud_ack.toggled.connect(lambda _checked: self._update_start_button())
        left.addWidget(self.cloud_ack)

        left.addWidget(_label("2. 撮影の回と条件", 12, bold=True))
        form = QFormLayout()
        form.setHorizontalSpacing(12)
        self.session_combo = QComboBox()
        for session in SESSIONS:
            self.session_combo.addItem(session.label, session)
        self.session_combo.currentIndexChanged.connect(lambda _index: self._on_session_changed())
        self.condition_combo = QComboBox()
        for condition in CONDITIONS:
            self.condition_combo.addItem(condition.label, condition)
        self.condition_combo.currentIndexChanged.connect(lambda _index: self._on_session_changed())
        form.addRow("回", self.session_combo)
        form.addRow("条件", self.condition_combo)
        left.addLayout(form)
        self.session_hint = _label("", 10.5, color=_COLOR_MUTED)
        left.addWidget(self.session_hint)

        left.addWidget(_label("3. カメラ", 12, bold=True))
        camera_row = QHBoxLayout()
        self.camera_spin = QSpinBox()
        self.camera_spin.setRange(0, 15)
        self.camera_spin.setValue(camera_index)
        self.camera_spin.setPrefix("カメラ ")
        camera_row.addWidget(self.camera_spin)
        self.mirror_box = QCheckBox("プレビューを鏡のように表示（保存される動画はそのままの向き）")
        self.mirror_box.setChecked(True)
        camera_row.addWidget(self.mirror_box, 1)
        left.addLayout(camera_row)
        if camera_label:
            self.camera_spin.setEnabled(False)
            left.addWidget(_label(f"カメラの代わりに動画ファイルを使います（テスト用）: {camera_label}", 10.5, color=_COLOR_WARN))
        self.setup_error = _label("", 10.5, color=_COLOR_ERROR)
        self.setup_error.setTextFormat(Qt.TextFormat.PlainText)
        left.addWidget(self.setup_error)
        left.addStretch(1)
        self.start_button = QPushButton("撮影を始める")
        self.start_button.setFont(_font(13, bold=True))
        self.start_button.setProperty("accent", True)
        self.start_button.clicked.connect(self.start_session)
        left_column = QVBoxLayout()
        left_column.setSpacing(12)
        left_column.addWidget(_scrolling(left_content), 1)
        left_column.addWidget(self.start_button)
        outer.addLayout(left_column, 3)

        right_panel = _panel()
        right = QVBoxLayout(right_panel)
        right.setContentsMargins(18, 16, 18, 16)
        right.setSpacing(14)
        self.storage_advice = _label(STORAGE_ADVICE, 10.5)
        self.storage_advice.setTextFormat(Qt.TextFormat.RichText)
        right.addWidget(self.storage_advice)
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("color: #5f6368;")
        right.addWidget(line)
        before = _label(BEFORE_SHOOTING, 10.5)
        before.setTextFormat(Qt.TextFormat.RichText)
        right.addWidget(before)
        right.addStretch(1)
        outer.addWidget(_scrolling(right_panel), 2)

        self._on_session_changed()
        self._update_start_button()

    def _selected_session(self) -> Session:
        return self.session_combo.currentData()

    def _selected_condition(self) -> Condition:
        return self.condition_combo.currentData()

    def _on_session_changed(self) -> None:
        session = self._selected_session()
        total = sum(step.target_s for step in self._script_provider(session, self._selected_condition()))
        rule = (
            "学習用とは別の日に撮ります。"
            if session.purpose == "evaluation"
            else "同じ日に「すっぴん → メイク」の順で、カメラの位置・照明・服・髪型を変えずに撮ります。"
        )
        self.session_hint.setText(
            f"この回：{session.plan}（照明：{session.lighting}）。{rule}\n今回の撮影時間の目安：約 {round(total / 60)} 分"
        )
        self.check_output(show_empty=False)

    def _on_output_edited(self, _text: str) -> None:
        self.output_message.setText("")
        self.cloud_ack.setVisible(False)
        self.cloud_ack.setChecked(False)
        self._output_refused = False
        self._update_start_button()

    def browse_output(self) -> None:
        # An explicit start folder: Qt would otherwise open the current directory, the project.
        start = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.HomeLocation)
        chosen = QFileDialog.getExistingDirectory(self, "保存先のフォルダを選ぶ", start)
        if chosen:
            self.output_edit.setText(chosen)
            self.check_output()

    def check_output(self, *, show_empty: bool = True) -> Path | None:
        """Validate the folder and show the outcome on the setup page; None when refused."""
        text = self.output_edit.text()
        if not text.strip() and not show_empty:
            return None
        try:
            checked = _within_time_limit(lambda: inspect_output_dir(text))
        except OutputFolderError as exc:
            self.output_message.setText(str(exc))
            _set_color(self.output_message, _COLOR_ERROR)
            self.cloud_ack.setVisible(False)
            self._output_refused = True
            self._update_start_button()
            return None
        self._output_refused = False
        folder = checked.folder
        warning = cloud_sync_warning(folder)
        self.cloud_ack.setVisible(warning is not None)
        lines = [f"保存先: {folder}", self._space_line(checked.free_bytes)]
        if warning is not None:
            lines.append(warning)
        self.output_message.setText("\n".join(line for line in lines if line))
        _set_color(self.output_message, _COLOR_WARN if warning or self._space_is_short(checked.free_bytes) else _COLOR_OK)
        self._update_start_button()
        return folder

    def _session_estimate(self) -> float:
        width, height = self._capture_size
        script = self._script_provider(self._selected_session(), self._selected_condition())
        return estimate_bytes(script, width, height, self._capture_fps, self._formats[0])

    def _space_is_short(self, free: int | None) -> bool:
        return free is not None and free < self._session_estimate()

    def _space_line(self, free: int | None) -> str:
        estimate = format_bytes(self._session_estimate())
        if free is None:
            return f"今回の撮影で約 {estimate} 使います（高画質のまま保存するため大きくなります）。"
        short = "。空き容量が足りないかもしれません" if free < self._session_estimate() else ""
        return f"空き容量 {format_bytes(free)}／今回の撮影で約 {estimate} 使います（高画質のまま保存するため大きくなります）{short}。"

    def _update_start_button(self) -> None:
        has_text = bool(self.output_edit.text().strip())
        acknowledged = not self.cloud_ack.isVisibleTo(self.setup_page) or self.cloud_ack.isChecked()
        self.start_button.setEnabled(has_text and acknowledged and not self._output_refused)

    def start_session(self) -> None:
        """Validate everything again, open the camera and show the first step."""
        self.setup_error.setText("")
        folder = self.check_output()
        if folder is None or not self.start_button.isEnabled():
            return
        session, condition = self._selected_session(), self._selected_condition()
        session_dir = session_folder(folder, session, condition, self._today())

        def open_folder() -> dict[int, TakeResult]:
            prepare_output_dir(folder)
            return load_completed_takes(session_dir)

        try:
            completed = _within_time_limit(open_folder)
        except OutputFolderError as exc:
            self.setup_error.setText(str(exc))
            return
        self._session, self._condition = session, condition
        self._script = tuple(self._script_provider(self._session, self._condition))
        self._folder = session_dir
        self._results = {number - 1: take for number, take in completed.items()}
        self._failures = {}
        self._notes = {}
        self._step_index = next((i for i in range(len(self._script)) if i not in self._results), 0)
        self._stack.setCurrentWidget(self.record_page)
        self._set_shortcuts_enabled(True)
        self.session_banner.setText(f"{self._session.label}・{self._condition.label}")
        self._preview_seq = -1
        self.preview.set_image(None)
        self._engine.start(self.camera_spin.value())
        self._show_step()

    def back_to_setup(self) -> None:
        if self._phase in (Phase.COUNTDOWN, Phase.RECORDING, Phase.SAVING):
            return
        self._engine.stop()
        self._set_shortcuts_enabled(False)
        self._stack.setCurrentWidget(self.setup_page)

    # ---- recording page ---------------------------------------------------

    def _build_record_page(self) -> None:
        self.record_page = QWidget()
        page = QVBoxLayout(self.record_page)
        page.setContentsMargins(16, 12, 16, 12)
        page.setSpacing(10)

        # At the top of the screen, close to the webcam, so reading keeps her eyes near the lens.
        self.script_panel = _panel()
        self.script_panel.setProperty("alert", False)
        script = QVBoxLayout(self.script_panel)
        script.setContentsMargins(20, 12, 20, 14)
        script.setSpacing(4)
        self.session_banner = _label("", 10.5, color=_COLOR_MUTED)
        self.step_title = _label("", 18, bold=True)
        self.cue_label = CueLabel()
        self.next_label = _label("", 15, color=_COLOR_MUTED)
        self.next_label.setTextFormat(Qt.TextFormat.PlainText)  # it may show an error message
        self.next_label.setMinimumHeight(self.next_label.fontMetrics().lineSpacing() * 2)
        self.next_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        script.addWidget(self.session_banner)
        script.addWidget(self.step_title)
        script.addWidget(self.cue_label)
        script.addWidget(self.next_label)
        page.addWidget(self.script_panel)

        middle = QHBoxLayout()
        middle.setSpacing(12)
        self.preview = PreviewView()
        middle.addWidget(self.preview, 3)

        side_content = QWidget()
        side = QVBoxLayout(side_content)
        side.setContentsMargins(0, 0, 0, 0)
        side.setSpacing(8)
        self.timer_label = _label("00:00", 34, bold=True)
        self.timer_label.setWordWrap(False)
        self.timer_label.setMinimumWidth(self.timer_label.fontMetrics().horizontalAdvance("00:00 / 00:00"))
        self.timer_progress = QProgressBar()
        self.timer_progress.setRange(0, 1000)
        self.timer_progress.setTextVisible(False)
        side.addWidget(self.timer_label)
        side.addWidget(self.timer_progress)

        info = QGridLayout()
        info.setHorizontalSpacing(10)
        info.setVerticalSpacing(2)
        self.camera_info = _label("カメラ: —", 10.5)
        self.dropped_label = _label("ドロップ: 0 フレーム", 12, bold=True, color=_COLOR_OK)
        self.take_label = _label("", 10.5, color=_COLOR_MUTED)
        info.addWidget(self.camera_info, 0, 0)
        info.addWidget(self.dropped_label, 1, 0)
        info.addWidget(self.take_label, 2, 0)
        side.addLayout(info)

        self.result_label = _label("", 12)
        self.result_label.setTextFormat(Qt.TextFormat.PlainText)
        side.addWidget(self.result_label)

        self.step_list = _label("", 10.5)
        self.step_list.setTextFormat(Qt.TextFormat.RichText)
        self.step_list.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        side.addWidget(self.step_list)
        side.addStretch(1)
        middle.addWidget(_scrolling(side_content), 2)
        page.addLayout(middle, 1)

        buttons = QHBoxLayout()
        buttons.setSpacing(10)
        self.setup_button = QPushButton("設定に戻る")
        self.setup_button.clicked.connect(self.back_to_setup)
        self.back_button = QPushButton("← 戻る")
        self.back_button.clicked.connect(self.go_back)
        self.retake_button = QPushButton("撮り直し (R)")
        self.retake_button.clicked.connect(self.retake)
        self.primary_button = QPushButton("録画開始 (Space)")
        self.primary_button.clicked.connect(self.primary_action)
        self.primary_button.setFont(_font(13, bold=True))
        self.next_button = QPushButton("次へ →")
        self.next_button.clicked.connect(self.go_next)
        self.auto_stop_box = QCheckBox("目安の時間で自動で止める")
        self.auto_stop_box.setChecked(True)
        self.auto_stop_box.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        for button in (self.setup_button, self.back_button, self.retake_button, self.primary_button, self.next_button):
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)  # Space must stay with the shortcut
            button.setMinimumHeight(46)
        buttons.addWidget(self.setup_button)
        buttons.addWidget(self.auto_stop_box)
        buttons.addStretch(1)
        buttons.addWidget(self.back_button)
        buttons.addWidget(self.retake_button)
        buttons.addWidget(self.primary_button, 1)
        buttons.addWidget(self.next_button)
        page.addLayout(buttons)

    def _build_shortcuts(self) -> None:
        self._shortcuts = []
        for key, slot in (
            (Qt.Key.Key_Space, self._on_space),
            (Qt.Key.Key_Right, self.go_next),
            (Qt.Key.Key_Left, self.go_back),
            (Qt.Key.Key_R, self.retake),
            (Qt.Key.Key_Escape, self._on_escape),
        ):
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.setAutoRepeat(False)  # a held key must not run through the steps
            shortcut.activated.connect(slot)
            self._shortcuts.append(shortcut)

    def _set_shortcuts_enabled(self, enabled: bool) -> None:
        for shortcut in self._shortcuts:
            shortcut.setEnabled(enabled)

    def _on_space(self) -> None:
        if self._phase is Phase.REVIEW:
            self.go_next()
        else:
            self.primary_action()

    def _on_escape(self) -> None:
        if self._phase is Phase.COUNTDOWN:
            self._cancel_countdown()

    # ---- step flow --------------------------------------------------------

    @property
    def _step(self) -> ScriptStep:
        return self._script[self._step_index]

    def primary_action(self) -> None:
        if self._phase in (Phase.READY, Phase.FAILED):
            self._start_countdown()
        elif self._phase is Phase.COUNTDOWN:
            self._cancel_countdown()
        elif self._phase is Phase.RECORDING:
            self._stop_recording("manual")

    def go_next(self) -> None:
        if self._phase not in (Phase.READY, Phase.REVIEW, Phase.FAILED):
            return
        if self._step_index + 1 < len(self._script):
            self._step_index += 1
            self._show_step()
        else:
            self._show_done()

    def go_back(self) -> None:
        if self._phase is Phase.DONE:
            self._show_step()
        elif self._phase in (Phase.READY, Phase.REVIEW, Phase.FAILED) and self._step_index > 0:
            self._step_index -= 1
            self._show_step()

    def retake(self) -> None:
        """A new take of this step; earlier files are kept. During a recording it restarts it."""
        if self._phase is Phase.RECORDING:
            self._retake_after_save = True
            self._stop_recording("retake")
        elif self._phase in (Phase.REVIEW, Phase.FAILED):
            self._start_countdown()

    def _show_step(self) -> None:
        step = self._step
        self.step_title.setText(
            f"ステップ {self._step_index + 1} / {len(self._script)}　{step.title}（目安 {step.target_label}）"
        )
        failure = self._failures.get(self._step_index)
        result = self._results.get(self._step_index)
        shown = failure or result
        self.timer_label.setText(f"{format_clock(shown.duration_s if shown else 0)} / {format_clock(step.target_s)}")
        self.timer_progress.setValue(0)
        self._show_dropped(shown.dropped_frames if shown else 0)
        if failure is not None:
            self._show_alert("録画に失敗しました", failure.error or "撮り直してください。")
            self._set_phase(Phase.FAILED)
        else:
            self._clear_alert()
            self._show_cue(0.0, live=False)
            note = self._notes.get(self._step_index)
            warning = None if result is None else self._take_warning(result)
            if note is not None:
                self._show_next(*note)
            elif warning is not None:
                self._show_next(warning, _COLOR_WARN)
            self._set_phase(Phase.REVIEW if result else Phase.READY)
        self._show_result()
        self._refresh_step_list()

    def _show_next(self, text: str, color: str = _COLOR_MUTED) -> None:
        self.next_label.setText(text)
        _set_color(self.next_label, color)

    def _show_alert(self, headline: str, detail: str) -> None:
        """Where she is looking anyway, and in the largest letters there are."""
        self._set_alert_style(True)
        self.cue_label.setText(headline)
        self._show_next(detail, _COLOR_ALERT_TEXT)

    def _clear_alert(self) -> None:
        self._set_alert_style(False)

    def _set_alert_style(self, active: bool) -> None:
        if active == self._alert_on:
            return
        self._alert_on = active
        self.script_panel.setProperty("alert", active)
        self.script_panel.style().unpolish(self.script_panel)
        self.script_panel.style().polish(self.script_panel)
        _set_color(self.cue_label, _COLOR_ALERT_TEXT if active else None)

    def _show_cue(self, elapsed_s: float, *, live: bool, warning: str = "") -> None:
        """``live``: counting down or recording, one cue at a time. Otherwise the whole instruction."""
        step = self._step
        current, following = cue_at(step, elapsed_s)
        if not live:
            self.cue_label.setText(step.instruction)
            self._show_next(f"最初は: {current}" if step.cues else "")
            return
        self.cue_label.setText(current)
        if warning:
            self._show_next(warning, _COLOR_WARN)
            return
        lines = []
        if following is not None:
            lines.append(f"次: {following}")
        elif not step.auto_stop:
            lines.append(
                "目安の時間になりました。区切りのよいところで止めてください。"
                if elapsed_s >= step.target_s
                else "終わったら「停止」を押してください。"
            )
        if step.note:
            lines.append(step.note)
        self._show_next("\n".join(lines))

    def _show_result(self) -> None:
        index = self._step_index
        result = self._results.get(index)
        text, color = "", None
        if index in self._failures:
            text = f"前のテイク {result.take} はそのまま残っています。" if result else "このステップの動画はまだありません。"
        elif result is not None:
            text, color = self._summarize(result)
        self.result_label.setText(text)
        _set_color(self.result_label, color)
        self.take_label.setText("" if result is None else f"保存済みのテイク: {result.take}")

    def _summarize(self, result: TakeResult) -> tuple[str, str]:
        size = f"{result.width}×{result.height}"
        text = f"テイク {result.take} を保存しました：{result.frames_written:,} フレーム、{size}、{round(result.fps, 2):g} fps、再生確認 OK"
        return (text, _COLOR_OK if self._take_warning(result) is None else _COLOR_WARN)

    def _take_warning(self, result: TakeResult) -> str | None:
        """What is worth telling her about a take of the current step that is otherwise fine."""
        total = result.frames_written + result.dropped_frames
        if result.dropped_frames / max(total, 1) > DROP_WARNING_RATIO:
            return f"ドロップが {result.dropped_frames:,} フレームありました。撮り直しをおすすめします。"
        if not result.lossless:
            return "ロスレスの形式で保存できなかったので、画質が少し落ちています。"
        target = self._step.target_s
        if result.duration_s < SHORT_TAKE_RATIO * target:
            return f"目安の {format_clock(target)} よりかなり短いテイクです（{format_clock(result.duration_s)}）。足りなければ撮り直してください。"
        return None

    def _refresh_step_list(self) -> None:
        rows = (len(self._script) + 1) // 2
        cells = [self._step_cell(index) for index in range(len(self._script))]
        lines = []
        for row in range(rows):
            right = cells[row + rows] if row + rows < len(cells) else "<td></td>"
            lines.append(f"<tr>{cells[row]}{right}</tr>")
        self.step_list.setText(f"<table cellpadding='2' width='100%'>{''.join(lines)}</table>")

    def _step_cell(self, index: int) -> str:
        step = self._script[index]
        done = index in self._results
        failed = index in self._failures and not done
        current = index == self._step_index and self._phase is not Phase.DONE
        color = "#8ab4f8" if current else _COLOR_ERROR if failed else _COLOR_OK if done else "#e8eaed"
        weight = "bold" if current else "normal"
        mark = "✕" if failed else "済" if done else "　"
        text = f"{mark} {index + 1}. {html.escape(step.title)}（{html.escape(step.target_label)}）"
        return f"<td style='color:{color}; font-weight:{weight}; padding-right:12px'>{text}</td>"

    def _show_done(self) -> None:
        missing = [step.title for index, step in enumerate(self._script) if index not in self._results]
        self.step_title.setText("この回の撮影はここまでです")
        self._clear_alert()
        if missing:
            self.cue_label.setText("まだ撮っていないステップがあります")
            self._show_next("未撮影: " + "、".join(missing))
        else:
            self.cue_label.setText("おつかれさまでした")
            if self._session.purpose == "training" and self._condition.key == "bare":
                self._show_next("次は同じ日に「メイク」を撮ります。カメラと照明は動かさないでください。")
            else:
                self._show_next("動画は暗号化したドライブで保管してください。")
        self.result_label.setText(f"保存先: {self._folder}")
        _set_color(self.result_label, None)
        self.timer_label.setText("")
        self.timer_progress.setValue(0)
        self.take_label.setText("")
        self.dropped_label.setText("")
        self._set_phase(Phase.DONE)
        self._refresh_step_list()

    def _set_phase(self, phase: Phase) -> None:
        self._phase = phase
        active = phase in (Phase.COUNTDOWN, Phase.RECORDING, Phase.SAVING)
        _set_display_keep_awake(active)
        idle = phase in (Phase.READY, Phase.REVIEW, Phase.FAILED)
        self.setup_button.setEnabled(idle or phase is Phase.DONE)
        self.back_button.setEnabled((idle and self._step_index > 0) or phase is Phase.DONE)
        self.retake_button.setEnabled(phase in (Phase.REVIEW, Phase.RECORDING, Phase.FAILED))
        self.next_button.setEnabled(idle)
        self.next_button.setText("完了 →" if self._step_index + 1 >= len(self._script) else "次へ →")
        self.auto_stop_box.setEnabled(not active)
        self._set_button_role(self.next_button, accent=phase is Phase.REVIEW)
        self._set_button_role(self.setup_button, accent=phase is Phase.DONE)
        if phase is Phase.READY:
            self.primary_button.setText("録画開始 (Space)")
            self.primary_button.setEnabled(True)
            self._set_button_role(self.primary_button, accent=True)
        elif phase is Phase.FAILED:
            self.primary_button.setText("撮り直し (Space)")
            self.primary_button.setEnabled(True)
            self._set_button_role(self.primary_button, accent=True)
        elif phase is Phase.COUNTDOWN:
            self.primary_button.setText("キャンセル (Esc)")
            self.primary_button.setEnabled(True)
            self._set_button_role(self.primary_button)
        elif phase is Phase.RECORDING:
            self.primary_button.setText("停止 (Space)")
            self.primary_button.setEnabled(True)
            self._set_button_role(self.primary_button, danger=True)
        elif phase is Phase.SAVING:
            self.primary_button.setText("保存しています…")
            self.primary_button.setEnabled(False)
            self._set_button_role(self.primary_button)
        else:
            self.primary_button.setText("録画済み" if phase is Phase.REVIEW else "完了")
            self.primary_button.setEnabled(False)
            self._set_button_role(self.primary_button)
        self._update_camera_gate()

    def _set_button_role(self, button: QPushButton, *, accent: bool = False, danger: bool = False) -> None:
        if button.property("accent") == accent and button.property("danger") == danger:
            return
        button.setProperty("accent", accent)
        button.setProperty("danger", danger)
        button.style().unpolish(button)
        button.style().polish(button)

    def _camera_ready(self) -> bool:
        return self._engine.snapshot().state == "running"

    def _update_camera_gate(self) -> None:
        if self._phase in (Phase.READY, Phase.FAILED):
            self.primary_button.setEnabled(self._camera_ready())

    def _start_countdown(self) -> None:
        if not self._camera_ready():
            return
        self._notes.pop(self._step_index, None)
        self._failures.pop(self._step_index, None)
        self._clear_alert()
        self.result_label.setText("")
        self._countdown_ends_at = time.monotonic() + self._countdown_s
        self._set_phase(Phase.COUNTDOWN)
        self._show_cue(0.0, live=True)
        self.timer_progress.setValue(0)
        self._tick_countdown()

    def _cancel_countdown(self) -> None:
        self.preview.set_overlay()
        self._show_step()

    def _begin_recording(self) -> None:
        assert self._folder is not None
        spec = TakeSpec(self._folder, self._session, self._condition, self._step, self._step_index + 1)
        self._recorder = self._engine.begin_take(spec)
        self._next_alert_beep = 0.0
        self._set_phase(Phase.RECORDING)
        self._show_dropped(0)
        _beep()

    def _stop_recording(self, reason: str) -> None:
        self._engine.end_take(reason)
        self._saving_progress = None
        if self._alert_on:  # about the take that has just ended
            self._clear_alert()
            self.cue_label.setText("保存しています…")
            self._show_next("")
        self._set_phase(Phase.SAVING)
        if reason == "target":
            _beep()

    def _on_take_finished(self) -> None:
        assert self._recorder is not None
        result = self._recorder.result
        self._recorder = None
        self._engine.end_take("manual")  # clears the engine's reference when the take failed by itself
        retake = self._retake_after_save
        self._retake_after_save = False
        index = self._step_index
        failed = result is not None and result.status == "failed"
        if result is None:
            logger.error("the take ended without a result")
        elif failed:
            self._failures[index] = result
        elif result.status == "completed":
            self._results[index] = result
            self._notes.pop(index, None)
            if result.stop_reason == "camera_lost":
                saved = format_clock(result.duration_s)
                self._notes[index] = (f"カメラの映像が届かなくなったので、録画を止めました。ここまでの {saved} は保存してあります。", _COLOR_WARN)
        elif result.status == "aborted" and not retake:
            kept = self._results.get(index)
            also = f"前のテイク {kept.take} をそのまま使います。" if kept else ""
            self._notes[index] = (f"テイク {result.take} はすぐに止めたので、使わないテイクとして残しました。{also}", _COLOR_WARN)
        if self._closing_deadline is not None:
            self.close()
            return
        if retake and not failed:
            self._set_phase(Phase.REVIEW)
            self._start_countdown()
            if self._phase is Phase.COUNTDOWN:
                return
        self._show_step()
        if failed or (result is not None and result.stop_reason == "camera_lost"):
            _sound_alarm()

    # ---- polling ----------------------------------------------------------

    def _tick(self) -> None:
        snapshot = self._engine.snapshot()
        if self._stack.currentWidget() is not self.record_page:
            if self._closing_deadline is not None:
                self._tick_saving()
            return
        if snapshot.preview is not None and snapshot.preview_seq != self._preview_seq:
            self._preview_seq = snapshot.preview_seq
            self.preview.set_image(bgr_to_qimage(snapshot.preview, mirror=self.mirror_box.isChecked()))
        self._show_camera_state(snapshot)
        self._update_camera_gate()
        if self._phase is Phase.COUNTDOWN:
            self._tick_countdown()
        elif self._phase is Phase.RECORDING:
            self._tick_recording()
        elif self._phase is Phase.SAVING:
            self._tick_saving()

    def _show_camera_state(self, snapshot: CaptureSnapshot) -> None:
        message = ""
        if snapshot.state == "opening":
            message = "カメラを開いています…"
        elif snapshot.state == "failed":
            message = snapshot.message or "カメラを開けませんでした。"
        elif snapshot.state == "no_frames":
            message = "カメラの映像が届いていません。接続を確認してください。"
        countdown = ""
        if self._phase is Phase.COUNTDOWN:
            countdown = str(max(1, math.ceil(self._countdown_ends_at - time.monotonic())))
            self.timer_label.setText(f"開始まで {countdown}")
        self.preview.set_overlay(
            countdown=countdown,
            message=message,
            recording=self._phase is Phase.RECORDING,
            stalled=snapshot.state == "no_frames",
        )
        if snapshot.width:
            fps = f"{snapshot.fps:.1f} fps" if snapshot.fps else "fps 測定中"
            self.camera_info.setText(f"カメラ: {snapshot.width}×{snapshot.height}、{fps}")
            small = snapshot.width < GUIDE_MIN_WIDTH or snapshot.height < GUIDE_MIN_HEIGHT
            low = small or 0 < snapshot.fps < GUIDE_MIN_FPS
            _set_color(self.camera_info, _COLOR_WARN if low else None)
            self.camera_info.setToolTip("ガイドの目安は 1280×720 以上、30 fps です。" if low else "")
        else:
            self.camera_info.setText("カメラ: —")

    def _tick_countdown(self) -> None:
        if time.monotonic() < self._countdown_ends_at:
            return
        self.preview.set_overlay()
        if self._camera_ready():
            self._begin_recording()
            return
        self._notes[self._step_index] = ("カメラの映像が届いていないので、録画を始めませんでした。接続を確認してください。", _COLOR_ERROR)
        self._show_step()
        _sound_alarm()

    def _tick_recording(self) -> None:
        recorder = self._recorder
        assert recorder is not None
        if recorder.done:  # it failed by itself: disk full, camera changed, ...
            self._on_take_finished()
            return
        stats = recorder.stats()
        step = self._step
        self.timer_label.setText(f"{format_clock(stats.elapsed_s)} / {format_clock(step.target_s)}")
        self.timer_progress.setValue(min(1000, round(1000 * stats.elapsed_s / max(step.target_s, 0.001))))
        problem = self._take_problem(stats)
        if problem is not None:
            self._show_alert(*problem)
            self._repeat_alarm()
        else:
            self._clear_alert()
            self._next_alert_beep = 0.0
            self._show_cue(stats.elapsed_s, live=True, warning=self._lost_frames_warning(stats))
        self._show_dropped(stats.dropped_total, waiting=stats.pending)
        if stats.take:
            free = "" if stats.free_bytes is None else f"（空き容量 {format_bytes(stats.free_bytes)}）"
            self.take_label.setText(f"テイク {stats.take} を録画中{free}")
        if step.auto_stop and self.auto_stop_box.isChecked() and stats.elapsed_s >= step.target_s:
            self._stop_recording("target")

    def _take_problem(self, stats: TakeStats) -> tuple[str, str] | None:
        """Something she must know now, not when she stops talking minutes later."""
        if stats.frame_age_s >= CAMERA_ALERT_S:
            left = max(1, math.ceil(CAMERA_LOST_STOP_S - stats.frame_age_s))
            return ("カメラの映像が止まっています", f"カメラの接続を確認してください。このまま {left} 秒たつと録画を止めます。")
        if stats.pending and stats.write_age_s >= WRITE_ALERT_S:
            return ("保存が止まっています", "保存先のドライブを確認してください。戻らなければ、停止して撮り直してください。")
        return None

    def _lost_frames_warning(self, stats: TakeStats) -> str:
        if stats.elapsed_s < LOST_WARNING_AFTER_S or stats.lost_ratio <= MAX_LOST_RATIO:
            return ""
        return (
            f"映像の {round(stats.lost_ratio * 100)}% が欠けています。"
            f"{round(MAX_LOST_RATIO * 100)}% を超えたまま終わると、使えないテイクになります。"
        )

    def _repeat_alarm(self) -> None:
        now = time.monotonic()
        if now >= self._next_alert_beep:
            self._next_alert_beep = now + ALERT_BEEP_INTERVAL_S
            _sound_alarm()

    def _show_dropped(self, count: int, *, waiting: int = 0) -> None:
        backlog = f"（保存待ち {waiting}）" if waiting > 5 else ""
        self.dropped_label.setText(f"ドロップ: {count:,} フレーム{backlog}")
        _set_color(self.dropped_label, _COLOR_OK if count == 0 else _COLOR_ERROR)

    def _tick_saving(self) -> None:
        recorder = self._recorder
        if recorder is None or recorder.done:
            if recorder is not None:
                self._on_take_finished()
            return
        now = time.monotonic()
        if self._closing_deadline is not None and now >= self._closing_deadline:
            logger.warning("the video is still being written; closing anyway")
            self._force_close = True
            self.close()
            return
        stats = recorder.stats()
        progress = (stats.written, stats.pending)
        if progress != self._saving_progress:
            self._saving_progress = progress
            self._saving_progress_at = now
        elif now - self._saving_progress_at >= SAVE_STALL_WARNING_S:
            self._show_alert("保存が止まっています", "保存先のドライブが応答していません。ドライブを確認してください。窓を閉じると終了できます。")

    # ---- the first size ---------------------------------------------------

    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)
        if not self._fitted_to_screen:
            self._fitted_to_screen = True
            self._fit_to_screen()

    def _fit_to_screen(self) -> None:
        """The preferred size may not fit a small or scaled screen: the buttons at the bottom must stay visible."""
        available = self.screen().availableGeometry()
        margins = self.frameGeometry().size() - self.geometry().size()
        width = min(self.width(), available.width() - margins.width())
        height = min(self.height(), available.height() - margins.height())
        if (width, height) != (self.width(), self.height()):
            self.resize(width, height)
        frame = self.frameGeometry()
        left = max(available.left(), min(frame.left(), available.right() - frame.width() + 1))
        top = max(available.top(), min(frame.top(), available.bottom() - frame.height() + 1))
        if (left, top) != (frame.left(), frame.top()):
            self.move(left, top)

    # ---- closing ----------------------------------------------------------

    def closeEvent(self, event: QCloseEvent) -> None:
        """Closing during a take finishes and saves it first, with a time limit."""
        if self._recorder is not None and not self._force_close:
            if self._closing_deadline is None:
                self._closing_deadline = time.monotonic() + CLOSE_TIMEOUT_S
                if self._phase is Phase.RECORDING:
                    self._stop_recording("close")
                self.setWindowTitle(f"{APP_TITLE}（保存してから終了します…）")
            event.ignore()
            return
        self._timer.stop()
        self._engine.stop()
        _set_display_keep_awake(False)
        event.accept()


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------


class RecorderCliError(Exception):
    """An expected start-up failure; the message is Japanese and printed as is."""


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"1 以上の整数を指定してください: {text}") from exc
    if value <= 0:
        raise argparse.ArgumentTypeError(f"1 以上の整数を指定してください: {text}")
    return value


def _non_negative_float(text: str) -> float:
    try:
        value = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"0 以上の数を指定してください: {text}") from exc
    if not 0 <= value < float("inf"):
        raise argparse.ArgumentTypeError(f"0 以上の数を指定してください: {text}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m makeup_cam.record",
        description="学習用の動画を、台本を画面に出しながら録画します（docs/capture-guide.md の 5・6 章）。",
    )
    parser.add_argument(
        "--output", metavar="DIR", help="保存先のフォルダ。初期値はありません。Git の作業フォルダの中には保存できません"
    )
    parser.add_argument("--camera", type=int, default=0, metavar="N", help="カメラの番号（初期値: 0）")
    parser.add_argument("--camera-file", type=Path, metavar="PATH", help="カメラの代わりに動画ファイルを使う（テスト用）")
    parser.add_argument("--width", type=_positive_int, default=GUIDE_MIN_WIDTH, help="解像度の幅（初期値: 1280）")
    parser.add_argument("--height", type=_positive_int, default=GUIDE_MIN_HEIGHT, help="解像度の高さ（初期値: 720）")
    parser.add_argument("--fps", type=_positive_int, default=30, help="フレームレート（初期値: 30）")
    parser.add_argument(
        "--countdown", type=_non_negative_float, default=COUNTDOWN_S, metavar="SEC", help="録画前のカウントダウンの秒数（初期値: 5）"
    )
    parser.add_argument(
        "--codec",
        choices=[video_format.key for video_format in VIDEO_FORMATS],
        default=VIDEO_FORMATS[0].key,
        help="保存形式。hfyu（初期値）と ffv1 はロスレス、mjpg はファイルが小さい代わりに画質が落ちます",
    )
    return parser


def build_source_factory(args: argparse.Namespace) -> SourceFactory:
    """The real camera, or with ``--camera-file`` the video file standing in for it."""
    width, height, fps = args.width, args.height, args.fps
    if args.camera_file is None:
        return lambda index: CameraSource(index, width, height, fps)
    video_source = getattr(camera_module, "VideoFileSource", None)
    if video_source is None:
        raise RecorderCliError("--camera-file はこのバージョンでは使えません（VideoFileSource がありません）。")
    if not args.camera_file.is_file():
        raise RecorderCliError(f"動画ファイルが見つかりません: {args.camera_file}")
    path = args.camera_file
    return lambda _index: video_source(path, width, height, fps)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        source_factory = build_source_factory(args)
    except RecorderCliError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    app = QApplication.instance() or QApplication(sys.argv[:1])
    window = RecorderWindow(
        source_factory,
        output=args.output,
        camera_index=args.camera,
        camera_label=None if args.camera_file is None else args.camera_file.name,
        countdown_s=args.countdown,
        capture_size=(args.width, args.height),
        capture_fps=float(args.fps),
        preferred_format=args.codec,
    )
    window.show()
    # The window's own timer wakes the event loop, so Python gets to run this handler.
    previous = signal.signal(signal.SIGINT, lambda _signum, _frame: window.close())
    try:
        return app.exec()
    finally:
        signal.signal(signal.SIGINT, previous)
        window.engine.join(2.0)
        # Now, while QApplication exists: left to Python, the widgets may be destroyed after it,
        # and the process then ends in an access violation (seen in over half of the runs).
        window.deleteLater()
        QApplication.sendPostedEvents(window, QEvent.Type.DeferredDelete)


if __name__ == "__main__":
    sys.exit(main())
