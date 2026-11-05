"""Recording tool tests, offscreen with fake sources and tiny temporary videos.

Everything is written to pytest's temp folder, never to the repository, and
the videos are synthetic: each frame's pixel value encodes its number, so a
lossless file can be checked frame by frame.
"""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from datetime import date
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
import pytest
from PySide6.QtCore import QRect, Qt, QTimer
from PySide6.QtWidgets import QApplication

from makeup_cam import camera as camera_module
from makeup_cam import record
from makeup_cam.camera import CameraError
from makeup_cam.record import (
    CONDITIONS,
    SESSIONS,
    CaptureEngine,
    Cue,
    FrameGapCounter,
    OutputFolderError,
    Phase,
    RecorderWindow,
    ScriptStep,
    TakeRecorder,
    TakeSpec,
    choose_container_fps,
    cloud_sync_warning,
    cue_at,
    cue_schedule,
    inspect_output_dir,
    load_completed_takes,
    next_take_number,
    probe_video,
    prepare_output_dir,
    script_for,
    session_folder,
    training_script,
    validate_output_dir,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
GUIDE = REPO_ROOT / "docs" / "capture-guide.md"
TINY = (64, 48)


@pytest.fixture(autouse=True)
def fractions_of_a_second_count(monkeypatch: pytest.MonkeyPatch) -> None:
    """These tests record fractions of a second. The rule against slips of the finger has its own tests."""
    monkeypatch.setattr(record, "MIN_USABLE_TAKE_S", 0.0)


def numbered_frame(number: int, size: tuple[int, int] = TINY) -> np.ndarray:
    return np.full((size[1], size[0], 3), number % 256, np.uint8)


def read_all_frames(path: Path) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    assert capture.isOpened(), path
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    return frames


class FakeSource:
    """A camera delivering numbered frames at ``fps`` in real time."""

    def __init__(self, fps: float = 100.0, size: tuple[int, int] = TINY, *, reported_fps: float | None = None) -> None:
        self.index = 3
        self.backend = "fake"
        self.actual_width, self.actual_height = size
        self.actual_fps = fps if reported_fps is None else reported_fps
        self._period = 1.0 / fps
        self._size = size
        self._count = 0
        self.opened = 0
        self.closed = 0

    def open(self) -> None:
        self.opened += 1

    def read(self) -> np.ndarray | None:
        time.sleep(self._period)
        self._count += 1
        return numbered_frame(self._count, self._size)

    def close(self) -> None:
        self.closed += 1


def short_script(_session: record.Session, _condition: record.Condition) -> tuple[ScriptStep, ...]:
    """Three steps taking a fraction of a second, one of them ended by hand."""
    return (
        ScriptStep("one", "一", "最初の指示", 0.3, "0.3 秒", (Cue("前半", 1), Cue("後半", 1))),
        ScriptStep("two", "二", "次の指示", 0.3, "0.3 秒"),
        ScriptStep("three", "三", "手で止める指示", 60.0, "1 分", auto_stop=False),
    )


def take_spec(output: Path, step: ScriptStep | None = None, number: int = 1) -> TakeSpec:
    session, condition = SESSIONS[0], CONDITIONS[0]
    step = step or training_script(5)[0]
    folder = session_folder(output, session, condition, date(2026, 11, 10))
    return TakeSpec(folder, session, condition, step, number)


def make_recorder(output: Path, **kwargs) -> TakeRecorder:
    kwargs.setdefault("fps", 30.0)
    kwargs.setdefault("camera", {"type": "FakeSource"})
    return TakeRecorder(take_spec(output), **kwargs)


def record_frames(recorder: TakeRecorder, count: int, *, size: tuple[int, int] = TINY, fps: float = 30.0) -> None:
    recorder.start()
    now = time.perf_counter()
    for number in range(count):
        recorder.offer(numbered_frame(number, size), now + number / fps)
    recorder.request_stop("manual")
    assert recorder.wait(20.0), "the writer thread did not finish"


def sidecar_of(output: Path, stem: str = "step01_prepare_take01") -> dict:
    folder = session_folder(output, SESSIONS[0], CONDITIONS[0], date(2026, 11, 10))
    return json.loads((folder / f"{stem}.json").read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# The script
# --------------------------------------------------------------------------


def _guide_section(title: str) -> str:
    text = GUIDE.read_text(encoding="utf-8")
    start = text.index(title)
    end = text.index("\n### ", start + len(title))
    return text[start:end]


def test_script_steps_match_the_guide() -> None:
    entries = re.findall(r"^\d+\. \*\*(.+?)\*\*（(.+?)）：", _guide_section("### 5.2"), re.MULTILINE)
    assert len(entries) == 9
    script = training_script(5)
    assert [(step.title, step.target_label) for step in script] == entries
    for step, duration in zip(script, entries, strict=True):
        if "〜" not in duration[1]:
            value = float(duration[1].split()[0]) * (60 if "分" in duration[1] else 1)
            assert step.target_s == value


def test_script_text_matches_the_guide_word_for_word() -> None:
    section = _guide_section("### 5.2")
    entries = re.findall(r"^\d+\. \*\*.+?\*\*（.+?）：(.+?)(?=^\d+\. |\Z)", section, re.MULTILINE | re.DOTALL)
    assert len(entries) == 9
    for talk_minutes in (5, 8):
        for body, step in zip(entries, training_script(talk_minutes), strict=True):
            sentence = re.sub(r"\s*\n\s*", "", body).strip()
            assert step.instruction == sentence
            position = 0
            for cue in step.cues:  # each cue is a piece of the sentence, in its order
                position = sentence.index(cue.text, position) + len(cue.text)


def test_sessions_match_the_guide_table() -> None:
    section = _guide_section("### 5.1")
    for session in SESSIONS[:3]:
        assert session.lighting in section
        assert session.plan in section
    assert {condition.label for condition in CONDITIONS} == {"すっぴん", "メイク"}


def test_talking_step_fills_the_longer_first_session() -> None:
    first = script_for(SESSIONS[0], CONDITIONS[0])
    second = script_for(SESSIONS[1], CONDITIONS[0])
    talk = {step.key: step for step in first}["talk"]
    assert talk.target_s == 8 * 60 and not talk.auto_stop
    assert {step.key: step for step in second}["talk"].target_s == 5 * 60


def test_evaluation_session_is_one_long_take_kept_apart_from_training() -> None:
    session = next(item for item in SESSIONS if item.key == "eval")
    bare = script_for(session, CONDITIONS[0])
    makeup = script_for(session, CONDITIONS[1])
    assert len(bare) == len(makeup) == 1
    assert bare[0].target_s == 20 * 60 and makeup[0].target_s == 5 * 60
    assert not bare[0].auto_stop
    folder = session_folder(Path("out"), session, CONDITIONS[0], date(2026, 11, 20))
    assert folder.parts[1] == "eval" and session.purpose == "evaluation"


def test_cues_follow_the_timer() -> None:
    step = ScriptStep("x", "x", "全体", 60.0, "1 分", (Cue("a", 30), Cue("b", 15), Cue("c", 15)))
    assert cue_schedule(step) == ((0.0, "a"), (30.0, "b"), (45.0, "c"))
    assert cue_at(step, 0.0) == ("a", "b")
    assert cue_at(step, 29.9) == ("a", "b")
    assert cue_at(step, 30.0) == ("b", "c")
    assert cue_at(step, 45.0) == ("c", None)
    assert cue_at(step, 500.0) == ("c", None)


def test_step_without_cues_shows_its_instruction() -> None:
    step = ScriptStep("x", "x", "全体", 10.0, "10 秒")
    assert cue_at(step, 3.0) == ("全体", None)


def test_every_cue_scales_to_the_step_length() -> None:
    for step in training_script(8):
        schedule = cue_schedule(step)
        assert schedule[0][0] == 0.0
        assert all(0 <= start < step.target_s for start, _ in schedule)


# --------------------------------------------------------------------------
# The output folder
# --------------------------------------------------------------------------


@pytest.mark.parametrize("text", [None, "", "   "])
def test_no_folder_means_no_default(text: str | None) -> None:
    with pytest.raises(OutputFolderError, match="保存先"):
        validate_output_dir(text)


def test_folder_inside_a_git_repository_is_refused(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    for folder in (tmp_path, tmp_path / "videos", tmp_path / "a" / "b" / "not-yet-existing"):
        with pytest.raises(OutputFolderError, match="Git の作業フォルダ"):
            validate_output_dir(folder)


def test_folder_inside_a_git_worktree_is_refused(tmp_path: Path) -> None:
    (tmp_path / ".git").write_text("gitdir: elsewhere\n", encoding="utf-8")  # worktrees and submodules
    with pytest.raises(OutputFolderError, match="Git の作業フォルダ"):
        validate_output_dir(tmp_path / "videos")


@pytest.mark.parametrize("relative", [".", "makeup_cam", ".test-media/renders/recorder", "not/yet/existing"])
def test_this_project_is_refused(relative: str) -> None:
    with pytest.raises(OutputFolderError):
        validate_output_dir(REPO_ROOT / relative)


def test_project_is_refused_even_without_git_metadata(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    project = tmp_path / "unzipped-project"
    project.mkdir()
    monkeypatch.setattr(record, "_PROJECT_ROOT", project.resolve())
    with pytest.raises(OutputFolderError, match="プログラムの中"):
        validate_output_dir(project / "videos")


def test_link_into_a_repository_is_refused(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    link = tmp_path / "innocent-looking-link"
    try:
        if sys.platform == "win32":
            import _winapi

            _winapi.CreateJunction(str(repo), str(link))  # needs no extra rights, unlike a symlink
        else:
            link.symlink_to(repo, target_is_directory=True)
    except OSError:
        pytest.skip("could not create a link on this machine")
    with pytest.raises(OutputFolderError, match="Git の作業フォルダ"):
        validate_output_dir(link / "videos")


def test_a_path_the_os_cannot_parse_is_refused() -> None:
    with pytest.raises(OutputFolderError):
        validate_output_dir("C:\\bad\0name")


def test_a_file_is_not_a_folder(tmp_path: Path) -> None:
    path = tmp_path / "file.txt"
    path.write_text("x", encoding="utf-8")
    with pytest.raises(OutputFolderError, match="ファイル"):
        validate_output_dir(path)


def test_folder_outside_repositories_is_accepted_and_not_created(tmp_path: Path) -> None:
    target = tmp_path / "撮影" / "new"
    assert validate_output_dir(target) == target.resolve()
    assert not target.exists()
    prepare_output_dir(target)
    assert target.is_dir() and not list(target.iterdir())


@pytest.mark.parametrize(
    "name",
    [
        "OneDrive",
        "OneDrive - 株式会社サンプル",
        "OneDrive-Personal",
        "OneDrive(会社)",
        "Dropbox",
        "Dropbox (Personal)",
        "Dropbox (サンプル)",
        "Google Drive",
        "Google ドライブ",
        "GoogleDrive",
        "マイドライブ",
        "My Drive",
        "共有ドライブ",
        "Shared drives",
        "iCloudDrive",
        "iCloud Drive",
        "Box",
        "Box Sync",
        "MEGA",
        "MEGAsync",
        "pCloud Drive",
        "Nextcloud",
        "Sync",
        "Amazon Drive",
        "SkyDrive",
    ],
)
def test_common_sync_folder_names_are_flagged(name: str, tmp_path: Path) -> None:
    assert "同期" in cloud_sync_warning(tmp_path / name / "capture")


@pytest.mark.parametrize("name", ["capture", "撮影データ", "Boxing", "Megan", "Syncopation", "dropbear"])
def test_ordinary_folder_names_are_not_flagged(name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for variable in ("OneDrive", "OneDriveConsumer", "OneDriveCommercial"):
        monkeypatch.delenv(variable, raising=False)
    assert cloud_sync_warning(tmp_path / name) is None


def test_cloud_synced_folders_are_flagged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("OneDrive", "OneDriveConsumer", "OneDriveCommercial"):
        monkeypatch.delenv(name, raising=False)
    assert cloud_sync_warning(tmp_path / "plain") is None
    assert "同期" in cloud_sync_warning(tmp_path / "OneDrive - 会社" / "videos")
    assert "同期" in cloud_sync_warning(tmp_path / "Dropbox" / "videos")
    custom = tmp_path / "custom-sync-root"
    monkeypatch.setenv("OneDrive", str(custom))
    assert "同期" in cloud_sync_warning(custom / "videos")


def finishes_within(seconds: float, work: Callable[[], object]) -> object:
    """Run ``work`` on a thread of its own, so that a regression fails the test instead of hanging the run."""
    outcome: list[object] = []

    def run() -> None:
        try:
            outcome.append(work())
        except Exception as exc:
            outcome.append(exc)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(seconds)
    assert not thread.is_alive(), f"still running after {seconds} s"
    return outcome[0]


def deny_new_files(monkeypatch: pytest.MonkeyPatch) -> None:
    """What Windows answers in a folder she may not write to. Also ``tempfile`` would get this answer."""

    def refuse(*_args: object, **_kwargs: object) -> int:
        raise PermissionError(13, "Permission denied", "somewhere")

    monkeypatch.setattr(record.os, "open", refuse)


def test_unwritable_folder_is_refused_at_once_not_after_days(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    deny_new_files(monkeypatch)
    outcome = finishes_within(5.0, lambda: prepare_output_dir(tmp_path))
    assert isinstance(outcome, OutputFolderError) and "書き込めません" in str(outcome)


@pytest.mark.skipif(sys.platform != "win32", reason="needs a folder this user may not write to")
def test_a_real_protected_folder_is_refused_at_once() -> None:
    folder = Path(os.environ["SystemRoot"])
    outcome = finishes_within(10.0, lambda: inspect_output_dir(folder))
    if not isinstance(outcome, OutputFolderError):
        pytest.skip("this user may write to the Windows folder (running as administrator?)")
    assert "書き込めません" in str(outcome)
    assert not list(folder.glob(f"{record._PROBE_PREFIX}*"))


def test_checking_a_folder_leaves_nothing_behind(tmp_path: Path) -> None:
    existing = inspect_output_dir(tmp_path)
    assert existing.folder == tmp_path.resolve() and existing.free_bytes > 0
    assert not list(tmp_path.iterdir())
    new = inspect_output_dir(tmp_path / "not" / "yet")
    assert new.folder == (tmp_path / "not" / "yet").resolve()
    assert not list(tmp_path.iterdir())


def test_a_new_folder_may_go_where_files_may_not(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Like C:\\ for an ordinary user: folders may be created there, files may not."""
    real_probe = record._probe_file

    def only_below(folder: Path) -> None:
        if folder == tmp_path.resolve():
            raise PermissionError(13, "Permission denied")
        real_probe(folder)

    monkeypatch.setattr(record, "_probe_file", only_below)
    assert inspect_output_dir(tmp_path / "capture").folder == (tmp_path / "capture").resolve()
    with pytest.raises(OutputFolderError, match="書き込めません"):
        inspect_output_dir(tmp_path)


def test_a_missing_drive_is_refused(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(record, "_nearest_existing_folder", lambda _folder: None)
    with pytest.raises(OutputFolderError, match="見つかりません"):
        inspect_output_dir(tmp_path / "x")


def test_a_path_too_long_for_windows_says_so(tmp_path: Path) -> None:
    long_path = str(tmp_path / ("x" * 240) / ("y" * 240))
    exc = FileNotFoundError(3, "The system cannot find the path specified", long_path)
    assert "長すぎます" in record._cannot_write_message(tmp_path, exc)


def test_a_file_in_the_middle_of_the_path_is_refused(tmp_path: Path) -> None:
    (tmp_path / "file.txt").write_text("x", encoding="utf-8")
    with pytest.raises(OutputFolderError, match="ファイル"):
        inspect_output_dir(tmp_path / "file.txt" / "capture")


@pytest.mark.skipif(sys.platform != "win32", reason="Windows device names")
@pytest.mark.parametrize(
    "text", ["CON", "con", "NUL", "AUX", "PRN", "COM1", "LPT9", "CONIN$", "CON ", "nul.txt", "C:/capture/NUL", "C:\\capture\\aux\\x", "\\\\.\\NUL"]
)
def test_windows_device_names_are_refused(text: str) -> None:
    with pytest.raises(OutputFolderError, match="予約"):
        validate_output_dir(text)


@pytest.mark.parametrize("text", ["CONSOLE", "nullable", "auxiliary", "COM", "COM10", "prn2", "C:/capture/console"])
def test_names_merely_like_a_device_are_fine(text: str) -> None:
    assert not record._names_a_device(text)


def test_an_os_error_while_checking_is_a_refusal_not_a_crash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A locked BitLocker drive answers with an error of its own, not with "not found"."""

    def locked(_path: Path) -> Path | None:
        raise OSError(-2144272384, "This drive is locked by BitLocker Drive Encryption")

    monkeypatch.setattr(record, "find_git_root", locked)
    with pytest.raises(OutputFolderError, match="確認できません"):
        validate_output_dir(tmp_path)


def test_the_module_cannot_reach_the_network() -> None:
    tree = ast.parse(Path(record.__file__).read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imported.add(node.module.split(".")[0])
    forbidden = {"socket", "ssl", "http", "urllib", "requests", "ftplib", "smtplib", "httpx", "aiohttp", "webbrowser"}
    assert not imported & forbidden


# --------------------------------------------------------------------------
# One take
# --------------------------------------------------------------------------


def test_take_plays_back_with_every_frame_exact(tmp_path: Path) -> None:
    recorder = make_recorder(tmp_path)
    record_frames(recorder, 40)
    result = recorder.result
    assert result is not None and result.status == "completed" and result.error is None
    assert result.frames_written == 40 and result.verified and result.lossless
    folder = session_folder(tmp_path, SESSIONS[0], CONDITIONS[0], date(2026, 11, 10))
    frames = read_all_frames(folder / result.video_name)
    assert len(frames) == 40
    assert frames[0].shape == (TINY[1], TINY[0], 3)
    assert [int(frame[0, 0, 0]) for frame in frames] == list(range(40))  # same frames, same order, lossless
    probe = probe_video(folder / result.video_name)
    assert (probe.playable, probe.frames, probe.width, probe.height) == (True, 40, *TINY)
    assert probe.fps == pytest.approx(30.0)


def test_sidecar_records_what_was_asked_for(tmp_path: Path) -> None:
    recorder = make_recorder(tmp_path, camera={"type": "FakeSource", "index": 2})
    record_frames(recorder, 12)
    data = sidecar_of(tmp_path)
    assert data["status"] == "completed"
    assert data["step"]["number"] == 1 and data["step"]["key"] == "prepare" and data["step"]["title"] == "準備"
    assert data["step"]["cues"][0] == {"start_s": 0.0, "text": "グレーカードを顔の横に持つ"}
    assert data["condition"] == {"key": "bare", "label": "すっぴん"}
    assert data["session"]["lighting"] == "本番の照明"
    assert data["camera"] == {"type": "FakeSource", "index": 2}
    assert data["resolution"] == {"width": TINY[0], "height": TINY[1]}
    assert data["fps"]["container"] == 30.0
    assert data["fps"]["measured"] == pytest.approx(30.0, abs=0.1)
    assert data["take"] == 1
    assert data["video"]["file"] == "step01_prepare_take01.mkv"
    assert data["video"]["codec"] == "HFYU" and data["video"]["lossless"] and not data["video"]["audio"]
    stamps = data["timestamps"]
    assert stamps["recording_started_at"] and stamps["first_frame_at"] and stamps["recording_stopped_at"]
    assert len(stamps["frame_offsets_ms"]) == 12
    assert stamps["frame_offsets_ms"][0] == 0.0
    assert stamps["frame_offsets_ms"][11] == pytest.approx(11 / 30 * 1000, abs=0.2)
    assert data["frames"]["written"] == 12 and data["frames"]["dropped_total"] == 0
    assert data["verified"]["ok"] and data["verified"]["frames"] == 12


@pytest.mark.parametrize(
    "error",
    [
        lambda path: PermissionError(13, "Permission denied", str(path)),
        lambda path: RuntimeError(f"cannot open '{path}'"),
        lambda path: RuntimeError(f"cannot open {path}"),
    ],
)
def test_a_failed_take_reveals_no_path_in_its_sidecar(tmp_path: Path, error: Callable[[Path], Exception]) -> None:
    output = tmp_path / "secret user name"

    def angry(path: Path, _fourcc: str, _fps: float, _size: tuple[int, int]) -> record.VideoWriterLike:
        raise error(path)

    recorder = make_recorder(output, writer_factory=angry)
    recorder.start()
    recorder.offer(numbered_frame(0), time.perf_counter())
    assert recorder.wait(10.0)
    assert recorder.result.status == "failed"
    data = sidecar_of(output)
    assert data["status"] == "failed" and data["error"]
    text = json.dumps(data, ensure_ascii=False)
    for leak in ("secret", "user name", str(tmp_path), str(tmp_path).replace("\\", "\\\\"), str(Path.home())):
        assert leak not in text


def test_the_reason_for_an_os_error_leaves_the_path_out() -> None:
    reason = record._os_error_reason(OSError(5, "入出力エラー", "C:/Users/someone/capture/take.mkv"))
    assert reason == "入出力エラー"


def test_sidecar_reveals_no_absolute_path(tmp_path: Path) -> None:
    source = tmp_path / "secret-user-name" / "clip.mp4"

    class FileLikeSource(FakeSource):
        path = source

    description = record.camera_description(FileLikeSource())
    assert description["file_name"] == "clip.mp4"
    recorder = make_recorder(tmp_path, camera=description)
    record_frames(recorder, 3)
    folder = session_folder(tmp_path, SESSIONS[0], CONDITIONS[0], date(2026, 11, 10))
    text = (folder / "step01_prepare_take01.json").read_text(encoding="utf-8")
    assert "secret-user-name" not in text
    assert str(tmp_path) not in text and str(tmp_path).replace("\\", "\\\\") not in text


def test_takes_never_overwrite_each_other(tmp_path: Path) -> None:
    for expected in range(1, 4):
        recorder = make_recorder(tmp_path)
        record_frames(recorder, 5 + expected)
        assert recorder.result.take == expected
    folder = session_folder(tmp_path, SESSIONS[0], CONDITIONS[0], date(2026, 11, 10))
    assert sorted(path.name for path in folder.glob("*.mkv")) == [f"step01_prepare_take0{n}.mkv" for n in (1, 2, 3)]
    assert len(read_all_frames(folder / "step01_prepare_take01.mkv")) == 6
    assert next_take_number(folder, 1, "prepare") == 4
    assert next_take_number(folder, 2, "front") == 1
    assert load_completed_takes(folder)[1].take == 3


def test_japanese_output_folder_works(tmp_path: Path) -> None:
    output = tmp_path / "撮影 データ（顔）"
    recorder = make_recorder(output)
    record_frames(recorder, 9)
    assert recorder.result.status == "completed" and recorder.result.verified
    folder = session_folder(output, SESSIONS[0], CONDITIONS[0], date(2026, 11, 10))
    assert len(read_all_frames(folder / recorder.result.video_name)) == 9


def test_720p_recording_is_pixel_exact(tmp_path: Path) -> None:
    size = (1280, 720)
    rng = np.random.default_rng(1)
    frames = [rng.integers(0, 256, (size[1], size[0], 3), np.uint8) for _ in range(3)]
    recorder = make_recorder(tmp_path)
    recorder.start()
    for number, frame in enumerate(frames):
        recorder.offer(frame, time.perf_counter() + number / 30)
    recorder.request_stop()
    assert recorder.wait(20.0)
    folder = session_folder(tmp_path, SESSIONS[0], CONDITIONS[0], date(2026, 11, 10))
    decoded = read_all_frames(folder / recorder.result.video_name)
    assert len(decoded) == 3
    for original, restored in zip(frames, decoded, strict=True):
        assert np.array_equal(original, restored)


def test_odd_camera_size_is_cropped_to_even_and_reported(tmp_path: Path) -> None:
    recorder = make_recorder(tmp_path)
    record_frames(recorder, 6, size=(65, 49))
    result = recorder.result
    assert result.status == "completed" and (result.width, result.height) == (64, 48)
    assert sidecar_of(tmp_path)["resolution"] == {"width": 64, "height": 48}


class SlowWriter:
    """The real writer behind a slow disk."""

    def __init__(self, path: Path, fourcc: str, fps: float, size: tuple[int, int], delay_s: float) -> None:
        self._inner = record.cv2_video_writer(path, fourcc, fps, size)
        self._delay_s = delay_s

    def isOpened(self) -> bool:
        return self._inner.isOpened()

    def write(self, frame: np.ndarray) -> None:
        time.sleep(self._delay_s)
        self._inner.write(frame)

    def release(self) -> None:
        self._inner.release()


def test_slow_disk_drops_and_counts_frames_without_blocking_the_camera(tmp_path: Path) -> None:
    frame_bytes = numbered_frame(0).nbytes
    recorder = make_recorder(
        tmp_path,
        writer_factory=lambda path, fourcc, fps, size: SlowWriter(path, fourcc, fps, size, 0.02),
        queue_bytes=frame_bytes * 3,
    )
    recorder.start()
    start = time.perf_counter()
    for number in range(100):
        recorder.offer(numbered_frame(number), start + number / 30)
    offered_in = time.perf_counter() - start
    assert offered_in < 0.5  # 100 frames at 20 ms each would take 2 s if the caller had to wait
    stats = recorder.stats()
    assert stats.dropped_queue_full > 50
    recorder.request_stop()
    assert recorder.wait(20.0)
    result = recorder.result
    assert result.status == "failed" and "保存が間に合わず" in result.error  # too much of it is missing to use
    assert result.verified  # what did get written is intact
    data = sidecar_of(tmp_path)
    assert data["frames"]["captured"] == 100
    assert data["frames"]["written"] + data["frames"]["dropped_queue_full"] == 100
    folder = session_folder(tmp_path, SESSIONS[0], CONDITIONS[0], date(2026, 11, 10))
    assert len(read_all_frames(folder / result.video_name)) == data["frames"]["written"]


def test_camera_gaps_are_counted_as_lost_frames(tmp_path: Path) -> None:
    recorder = make_recorder(tmp_path)
    recorder.start()
    now = time.perf_counter()
    for number, moment in enumerate([0.0, 1 / 30, 2 / 30, 7 / 30, 8 / 30]):  # frames 3-6 never came
        recorder.offer(numbered_frame(number), now + moment)
    time.sleep(0.01)
    assert recorder.stats().dropped_camera_gaps == 4
    recorder.request_stop()
    assert recorder.wait(20.0)
    assert recorder.result.frames_written == 5


def test_a_late_burst_of_buffered_frames_is_not_counted_as_lost(tmp_path: Path) -> None:
    recorder = make_recorder(tmp_path)
    recorder.start()
    now = time.perf_counter()
    moments = [0.0, 1 / 30, 4 / 30, 4.1 / 30, 4.2 / 30, 5.2 / 30]  # a stall, then the two late frames at once
    for number, moment in enumerate(moments):
        recorder.offer(numbered_frame(number), now + moment)
    assert recorder.stats().dropped_camera_gaps == 0
    recorder.request_stop()
    assert recorder.wait(20.0)


def test_an_outage_counts_in_the_drop_count_while_it_lasts(tmp_path: Path) -> None:
    recorder = make_recorder(tmp_path)
    recorder.start()
    now = time.perf_counter()
    recorder.offer(numbered_frame(0), now)
    time.sleep(0.5)  # no frame arrives now: before, nothing moved until the next one came or the take ended
    stats = recorder.stats()
    assert stats.dropped_camera_gaps >= 10 and stats.camera_stalled == stats.dropped_camera_gaps
    assert stats.frame_age_s >= 0.5 and stats.lost_ratio > 0.5
    recorder.request_stop()
    assert recorder.wait(20.0)


def test_a_take_with_most_frames_missing_is_failed_and_not_used(tmp_path: Path) -> None:
    recorder = make_recorder(tmp_path)
    recorder.start()
    now = time.perf_counter()
    for number in range(15):
        recorder.offer(numbered_frame(number), now + number / 30)
    recorder.offer(numbered_frame(15), now + 4.0)  # gone for 3.5 s, back for the very end
    recorder.request_stop()
    assert recorder.wait(20.0)
    result = recorder.result
    assert result.status == "failed" and "カメラの映像が途切れて" in result.error
    assert result.frames_written == 16 and result.dropped_frames > 100  # the frames there are stay on disk
    data = sidecar_of(tmp_path)
    assert data["frames"]["camera_stalled"] > 100 and data["frames"]["lost_ratio"] > 0.8
    folder = session_folder(tmp_path, SESSIONS[0], CONDITIONS[0], date(2026, 11, 10))
    assert load_completed_takes(folder) == {}


def test_a_small_outage_in_a_long_take_keeps_the_take(tmp_path: Path) -> None:
    recorder = make_recorder(tmp_path)
    recorder.start()
    now = time.perf_counter()
    for number in range(300):
        recorder.offer(numbered_frame(number), now + number / 30 + (0.9 if number >= 150 else 0.0))
    recorder.request_stop()
    assert recorder.wait(20.0)
    result = recorder.result
    assert result.status == "completed" and 24 <= result.dropped_frames <= 29  # about 8 %, shown with a warning


def test_the_time_limit_is_on_the_share_and_on_the_seconds(tmp_path: Path) -> None:
    recorder = make_recorder(tmp_path)
    recorder.start()
    now = time.perf_counter()
    for number in range(10):
        recorder.offer(numbered_frame(number), now + number / 30 + (0.5 if number >= 5 else 0.0))  # half a second of 0.8 s
    recorder.request_stop()
    assert recorder.wait(20.0)
    assert recorder.result.status == "completed"


def test_a_camera_gone_for_good_stops_the_take_by_itself(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(record, "CAMERA_LOST_STOP_S", 0.3)
    monkeypatch.setattr(record, "HOUSEKEEPING_INTERVAL_S", 0.05)
    monkeypatch.setattr(record, "MIN_LOST_TO_FAIL_S", 0.0)
    recorder = make_recorder(tmp_path)
    recorder.start()
    recorder.offer(numbered_frame(0), time.perf_counter())
    assert recorder.wait(10.0)
    assert recorder.result.status == "failed" and "途切れて" in recorder.result.error
    assert recorder.result.stop_reason == "camera_lost"


def test_a_long_good_part_survives_the_camera_going_away(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(record, "HOUSEKEEPING_INTERVAL_S", 0.05)
    recorder = make_recorder(tmp_path)
    recorder.start()
    moment = time.perf_counter() - 10.0
    for number in range(300):  # ten good seconds up to just now
        recorder.offer(numbered_frame(number), moment + number / 30)
    monkeypatch.setattr(record, "CAMERA_LOST_STOP_S", 0.3)  # only from here: those moments lie in the past
    assert recorder.wait(10.0)  # it stops by itself
    assert recorder.result.status == "completed" and recorder.result.stop_reason == "camera_lost"
    assert recorder.result.dropped_frames >= 8
    assert sidecar_of(tmp_path)["stop_reason"] == "camera_lost"


def test_a_slow_disk_finishing_up_is_not_a_lost_camera(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(record, "CAMERA_LOST_STOP_S", 0.1)
    monkeypatch.setattr(record, "HOUSEKEEPING_INTERVAL_S", 0.01)
    recorder = make_recorder(tmp_path, writer_factory=lambda path, fourcc, fps, size: SlowWriter(path, fourcc, fps, size, 0.05))
    recorder.start()
    now = time.perf_counter()
    for number in range(20):  # the queue takes a second to empty, long after the last frame came
        recorder.offer(numbered_frame(number), now + number / 30)
    recorder.request_stop("manual")
    assert recorder.wait(20.0)
    assert recorder.result.status == "completed" and recorder.result.stop_reason == "manual"


def test_a_camera_that_never_delivers_stops_the_take_too(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(record, "CAMERA_LOST_STOP_S", 0.3)
    monkeypatch.setattr(record, "HOUSEKEEPING_INTERVAL_S", 0.05)
    recorder = make_recorder(tmp_path)
    recorder.start()
    assert recorder.wait(10.0)
    assert recorder.result.status == "failed" and recorder.result.frames_written == 0


@pytest.mark.parametrize("sigma_ms", [1, 3, 5, 8, 12])
def test_timing_jitter_is_not_counted_as_lost_frames(sigma_ms: int) -> None:
    rng = np.random.default_rng(5)
    counter = FrameGapCounter(30.0)
    moment, frames = 0.0, 3000
    for number in range(frames):
        moment = max(moment + 0.001, number / 30 + rng.normal(0.0, sigma_ms / 1000))
        counter.add(moment)
    assert counter.lost(moment) / frames < 0.005  # the warning starts at 1 %


def test_jitter_does_not_fail_a_take_through_the_recorder(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    recorder = make_recorder(tmp_path)
    recorder.start()
    now, moment = time.perf_counter(), time.perf_counter()
    for number in range(600):
        moment = max(moment + 0.001, now + number / 30 + rng.normal(0.0, 0.012))
        recorder.offer(numbered_frame(number), moment)
    recorder.request_stop()
    assert recorder.wait(20.0)
    assert recorder.result.status == "completed" and recorder.result.dropped_frames <= 6


def test_single_missing_frames_count_when_the_timing_is_steady() -> None:
    rng = np.random.default_rng(11)
    counter = FrameGapCounter(30.0)
    slot, moment = 0, 0.0
    for number in range(3000):
        slot += 2 if number % 50 == 49 else 1  # every 50th frame missing: 60 in all
        moment = max(moment + 0.001, slot / 30 + rng.normal(0.0, 0.002))
        counter.add(moment)
    assert 50 <= counter.lost(moment) <= 65
    assert counter.stalled(moment) == 0  # they do not make the take unusable


def test_a_late_frame_followed_by_an_early_one_is_not_a_missing_frame() -> None:
    counter = FrameGapCounter(30.0)
    moments = [number / 30 for number in range(600)]
    for number in range(200, 600, 50):
        moments[number] += 0.022  # two thirds of a frame late, the next one on time
    for moment in moments:
        counter.add(moment)
    assert counter.lost(moments[-1]) == 0


def test_a_hiccup_before_the_take_does_not_set_the_speed_of_the_video(tmp_path: Path) -> None:
    class HiccupThenBurst(FakeSource):
        """One stall a second, made up for right after it, like a driver with a buffer."""

        def __init__(self) -> None:
            super().__init__(fps=30.0)
            self._next = 0.0

        def read(self) -> np.ndarray | None:
            now = time.perf_counter()
            self._next = self._next or now
            if self._count % 30 == 29:
                time.sleep(0.2)
                self._next = time.perf_counter() - 0.17
            elif self._next > now:
                time.sleep(self._next - now)
            self._next += 1 / 30
            self._count += 1
            return numbered_frame(self._count)

    engine = CaptureEngine(lambda _index: HiccupThenBurst())
    engine.start(0)
    wait_for(lambda: engine.snapshot().fps > 0, 8.0)
    wait_for(lambda: len(engine._frame_times) >= 80, 8.0)
    recorder = engine.begin_take(take_spec(tmp_path))
    engine.end_take()
    assert recorder.wait(20.0)
    engine.stop()
    assert engine.join(5.0)
    assert sidecar_of(tmp_path)["fps"]["container"] == pytest.approx(30.0, rel=0.03)


def test_a_camera_falling_far_below_its_rate_stops_counting_as_stalling() -> None:
    counter = FrameGapCounter(30.0)
    moment = 0.0
    for number in range(1200):
        moment += 1 / 30 if number < 300 else 0.1  # 10 fps for 30 s: the gaps are 3 periods, and stay so
        counter.add(moment)
    assert counter.stalled(moment) < 300  # about 2 for each of the first gaps only, not for all 900
    assert counter.lost(moment) > 500


def test_an_outage_does_not_blind_the_counter_to_the_next_one() -> None:
    counter = FrameGapCounter(30.0)
    moment = 0.0
    for number in range(600):
        moment += 1 / 30
        if number in (300, 400):
            moment += 3.0
        counter.add(moment)
    assert counter.stalled(moment) == pytest.approx(180, abs=3)


def test_a_short_outage_right_after_a_long_one_still_counts() -> None:
    counter = FrameGapCounter(30.0)
    moment = 0.0
    for number in range(400):
        moment += 1 / 30
        if number == 300:
            moment += 3.0
        elif number == 310:
            moment += 0.2
        counter.add(moment)
    assert counter.stalled(moment) == pytest.approx(96, abs=2)


def test_a_clock_a_little_slower_than_reported_loses_nothing() -> None:
    counter = FrameGapCounter(30.0)
    for number in range(9000):
        counter.add(number / 29.2)
    assert counter.lost(8999 / 29.2) == 0


def test_a_camera_halving_its_rate_shows_as_drops_but_not_as_an_outage(tmp_path: Path) -> None:
    recorder = make_recorder(tmp_path)
    recorder.start()
    now = time.perf_counter()
    moment = now
    for number in range(300):
        moment += 1 / 30 if number < 150 else 2 / 30  # dim light: longer exposure, half the frames
        recorder.offer(numbered_frame(number), moment)
    recorder.request_stop()
    assert recorder.wait(20.0)
    result = recorder.result
    assert result.status == "completed"  # the footage is fine; failing it would only repeat on the retake
    assert result.dropped_frames > 50  # and she is told
    assert sidecar_of(tmp_path)["frames"]["camera_stalled"] == 0


def test_measured_fps_ignores_a_hiccup() -> None:
    engine = CaptureEngine(lambda _index: FakeSource())
    start = 100.0
    times = [start + number / 30 for number in range(60)]
    times += [times[-1] + 0.5 + number / 30 for number in range(1, 30)]  # half a second without frames
    engine._frame_times.extend(times)
    assert engine._measured_fps() == pytest.approx(30.0, abs=0.01)


def test_measured_fps_ignores_a_hiccup_and_the_burst_after_it() -> None:
    times, moment = [], 0.0
    for number in range(90):
        if number % 30 == 29:
            moment += 0.2  # the driver holds frames back ...
        elif number % 30 < 6 and number > 6:
            moment += 0.001  # ... and hands them over at once
        else:
            moment += 1 / 30
        times.append(moment)
    assert record.steady_fps(times) == pytest.approx(30.0, abs=0.3)


@pytest.mark.parametrize("sigma_ms", [3, 8, 12])
def test_measured_fps_stays_true_under_jitter(sigma_ms: int) -> None:
    rng = np.random.default_rng(2)
    times, moment = [], 0.0
    for number in range(90):
        moment = max(moment + 0.001, number / 30 + rng.normal(0.0, sigma_ms / 1000))
        times.append(moment)
    assert record.steady_fps(times) == pytest.approx(30.0, rel=0.03)  # within what still trusts the camera's report


def test_a_file_that_stops_growing_fails_the_take_during_the_recording(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The drive reports free space and the file exists, but the data goes nowhere."""
    monkeypatch.setattr(record, "WRITE_STALL_S", 0.2)
    monkeypatch.setattr(record, "WRITE_STALL_MIN_BYTES", 1)
    monkeypatch.setattr(record, "HOUSEKEEPING_INTERVAL_S", 0.05)

    class Dying:
        def __init__(self, path: Path, fourcc: str, fps: float, size: tuple[int, int]) -> None:
            self._inner = record.cv2_video_writer(path, fourcc, fps, size)

        def isOpened(self) -> bool:
            return self._inner.isOpened()

        def write(self, _frame: np.ndarray) -> None:
            pass  # like cv2.VideoWriter on a dead disk: no error

        def release(self) -> None:
            self._inner.release()

    recorder = make_recorder(tmp_path, writer_factory=Dying)
    recorder.start()
    deadline = time.monotonic() + 10.0
    number = 0
    while not recorder.done and time.monotonic() < deadline:
        number += 1
        recorder.offer(numbered_frame(number), time.perf_counter())
        time.sleep(0.01)
    assert recorder.done, "it should stop without being asked to"
    assert recorder.result.status == "failed" and "増えなく" in recorder.result.error


def test_a_healthy_file_is_not_taken_for_a_stalled_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(record, "WRITE_STALL_S", 0.3)
    monkeypatch.setattr(record, "WRITE_STALL_MIN_BYTES", 1)
    monkeypatch.setattr(record, "HOUSEKEEPING_INTERVAL_S", 0.05)
    rng = np.random.default_rng(3)
    noise = rng.integers(0, 256, (240, 320, 3), np.uint8)  # big enough to reach the file at once
    recorder = make_recorder(tmp_path)
    recorder.start()
    deadline = time.monotonic() + 1.2
    while time.monotonic() < deadline:
        recorder.offer(noise.copy(), time.perf_counter())
        time.sleep(0.02)
    assert not recorder.done
    recorder.request_stop()
    assert recorder.wait(20.0)
    assert recorder.result.status == "completed"


def test_size_change_stops_the_take_and_keeps_what_was_recorded(tmp_path: Path) -> None:
    recorder = make_recorder(tmp_path)
    recorder.start()
    now = time.perf_counter()
    for number in range(5):
        recorder.offer(numbered_frame(number), now + number / 30)
    recorder.offer(numbered_frame(5, (32, 24)), now + 5 / 30)
    assert recorder.wait(20.0)  # it stops by itself
    result = recorder.result
    assert result.status == "failed" and "解像度" in result.error
    assert result.frames_written == 5
    folder = session_folder(tmp_path, SESSIONS[0], CONDITIONS[0], date(2026, 11, 10))
    assert len(read_all_frames(folder / result.video_name)) == 5


def test_no_frames_at_all_fails_and_leaves_no_empty_video(tmp_path: Path) -> None:
    recorder = make_recorder(tmp_path)
    recorder.start()
    recorder.request_stop()
    assert recorder.wait(10.0)
    assert recorder.result.status == "failed" and "フレーム" in recorder.result.error
    folder = session_folder(tmp_path, SESSIONS[0], CONDITIONS[0], date(2026, 11, 10))
    assert not list(folder.glob("*.mkv"))
    assert sidecar_of(tmp_path)["status"] == "failed"


def test_folder_that_became_a_repository_is_refused_at_the_last_moment(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    recorder = make_recorder(tmp_path)
    record_frames(recorder, 5)
    assert recorder.result.status == "failed" and "Git" in recorder.result.error
    assert not list(tmp_path.rglob("*.mkv")) and not list(tmp_path.rglob("*.json"))


def test_a_folder_that_stops_accepting_files_fails_the_take_at_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Between takes: the writer thread used to loop in tempfile for days, stuck on 保存しています."""
    deny_new_files(monkeypatch)
    recorder = make_recorder(tmp_path)
    started = time.perf_counter()
    record_frames(recorder, 5)
    assert time.perf_counter() - started < 10.0
    assert recorder.result.status == "failed" and "書き込めません" in recorder.result.error


def test_a_sidecar_that_cannot_be_written_stops_the_take(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Otherwise every frame would be thrown away while the window says it is recording."""

    def refuse(_source: object, _target: object) -> None:
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(record.os, "replace", refuse)
    recorder = make_recorder(tmp_path)
    recorder.start()
    assert recorder.wait(10.0)  # without being asked to stop
    assert recorder.result.status == "failed" and "JSON" in recorder.result.error


def test_full_disk_stops_cleanly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    usage = record.shutil.disk_usage(tmp_path)
    monkeypatch.setattr(record.shutil, "disk_usage", lambda _path: usage._replace(free=1024))
    recorder = make_recorder(tmp_path)
    recorder.start()
    recorder.offer(numbered_frame(0), time.perf_counter())
    assert recorder.wait(10.0)
    assert recorder.result.status == "failed" and "空き容量" in recorder.result.error


def test_unavailable_writer_is_a_clear_failure(tmp_path: Path) -> None:
    class Closed:
        def isOpened(self) -> bool:
            return False

        def write(self, frame: np.ndarray) -> None:
            raise AssertionError("never written")

        def release(self) -> None:
            pass

    recorder = make_recorder(tmp_path, writer_factory=lambda *_args: Closed())
    record_frames(recorder, 3)
    assert recorder.result.status == "failed" and "動画ファイルを作れません" in recorder.result.error


def test_falls_back_to_the_next_format_and_says_so(tmp_path: Path) -> None:
    used = []

    def only_mjpg(path: Path, fourcc: str, fps: float, size: tuple[int, int]) -> record.VideoWriterLike:
        used.append(fourcc)
        return record.cv2_video_writer(path, fourcc if fourcc == "MJPG" else "????", fps, size)

    recorder = make_recorder(tmp_path, writer_factory=only_mjpg)
    record_frames(recorder, 4)
    assert used == ["HFYU", "FFV1", "MJPG"]
    result = recorder.result
    assert result.status == "completed" and not result.lossless
    assert sidecar_of(tmp_path)["video"]["codec"] == "MJPG"


def test_writer_crash_is_contained(tmp_path: Path) -> None:
    class Crashing:
        def isOpened(self) -> bool:
            return True

        def write(self, frame: np.ndarray) -> None:
            raise cv2.error("boom")

        def release(self) -> None:
            pass

    recorder = make_recorder(tmp_path, writer_factory=lambda *_args: Crashing())
    recorder.start()
    recorder.offer(numbered_frame(0), time.perf_counter())
    assert recorder.wait(10.0)
    assert recorder.result.status == "failed"


def test_a_slip_of_the_finger_does_not_replace_a_good_take(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(record, "MIN_USABLE_TAKE_S", 3.0)
    now = time.perf_counter()
    good = make_recorder(tmp_path)
    good.start()
    for number in range(150):  # 5 s
        good.offer(numbered_frame(number), now + number / 30)
    good.request_stop()
    assert good.wait(20.0)
    assert good.result.status == "completed"

    slip = make_recorder(tmp_path)
    slip.start()
    for number in range(10):  # a third of a second
        slip.offer(numbered_frame(number), now + number / 30)
    slip.request_stop()
    assert slip.wait(20.0)
    assert slip.result.status == "aborted" and slip.result.error is None
    assert sidecar_of(tmp_path, "step01_prepare_take02")["stop_reason"] == "manual"
    folder = session_folder(tmp_path, SESSIONS[0], CONDITIONS[0], date(2026, 11, 10))
    assert load_completed_takes(folder)[1].take == 1
    assert (folder / "step01_prepare_take02.mkv").is_file()  # kept all the same


def test_reaching_the_target_is_never_a_slip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(record, "MIN_USABLE_TAKE_S", 3.0)
    recorder = make_recorder(tmp_path)
    recorder.start()
    now = time.perf_counter()
    for number in range(5):
        recorder.offer(numbered_frame(number), now + number / 30)
    recorder.request_stop("target")
    assert recorder.wait(20.0)
    assert recorder.result.status == "completed"


def test_damaged_sidecars_do_not_stop_a_session_from_opening(tmp_path: Path) -> None:
    recorder = make_recorder(tmp_path)
    record_frames(recorder, 5)
    folder = session_folder(tmp_path, SESSIONS[0], CONDITIONS[0], date(2026, 11, 10))
    for name, content in (
        ("step02_front_take01.json", "[1, 2, 3]"),
        ("step03_head_take01.json", '"just a string"'),
        ("step04_bow_take01.json", '{"status": "completed"}'),
        ("step05_talk_take01.json", '{"status": "completed", "step": {"number": 5}, "video": {"file": null}}'),
    ):
        (folder / name).write_text(content, encoding="utf-8")
    assert list(load_completed_takes(folder)) == [1]


def test_a_take_whose_video_is_gone_does_not_count(tmp_path: Path) -> None:
    for _ in range(2):
        record_frames(make_recorder(tmp_path), 5)
    folder = session_folder(tmp_path, SESSIONS[0], CONDITIONS[0], date(2026, 11, 10))
    assert load_completed_takes(folder)[1].take == 2
    (folder / "step01_prepare_take02.mkv").unlink()
    assert load_completed_takes(folder)[1].take == 1  # back to the one that is still there
    (folder / "step01_prepare_take01.mkv").unlink()
    assert load_completed_takes(folder) == {}


def test_an_interrupted_recording_is_not_counted_as_done(tmp_path: Path) -> None:
    recorder = make_recorder(tmp_path)
    record_frames(recorder, 5)
    folder = session_folder(tmp_path, SESSIONS[0], CONDITIONS[0], date(2026, 11, 10))
    path = folder / "step01_prepare_take01.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["status"] = "recording"  # what a power cut leaves behind
    path.write_text(json.dumps(data), encoding="utf-8")
    assert load_completed_takes(folder) == {}
    assert next_take_number(folder, 1, "prepare") == 2
    (folder / "step02_front_take01.json").write_text("{broken", encoding="utf-8")
    assert load_completed_takes(folder) == {}


@pytest.mark.parametrize(
    ("reported", "measured", "requested", "expected"),
    [
        (30.0, 29.7, 30.0, 30.0),  # close enough: trust the camera
        (30.0, 29.2, 30.0, 30.0),
        (30.0, 28.8, 30.0, 28.8),  # 4 % off: a few seconds' worth in every minute of video
        (30.0, 27.5, 30.0, 27.5),
        (30.0, 15.2, 30.0, 15.2),  # dim light: the video must not play at double speed
        (0.0, 24.0, 30.0, 24.0),  # nothing reported
        (0.0, 0.0, 30.0, 30.0),  # nothing known yet
        (1000.0, 0.0, 30.0, 30.0),  # nonsense reported
        (29.97, 0.0, 30.0, 29.97),
    ],
)
def test_container_fps(reported: float, measured: float, requested: float, expected: float) -> None:
    assert choose_container_fps(reported, measured, requested) == expected


# --------------------------------------------------------------------------
# The capture thread
# --------------------------------------------------------------------------


def wait_for(condition: Callable[[], bool], timeout_s: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("condition not met in time")
        time.sleep(0.005)


def test_engine_previews_and_records(tmp_path: Path) -> None:
    source = FakeSource(fps=100.0, size=(1280, 720))
    engine = CaptureEngine(lambda _index: source, preview_width=320)
    engine.start(0)
    wait_for(lambda: engine.snapshot().preview is not None)
    snapshot = engine.snapshot()
    assert snapshot.state == "running" and snapshot.preview.shape == (180, 320, 3)
    assert (snapshot.width, snapshot.height) == (1280, 720)
    recorder = engine.begin_take(take_spec(tmp_path))
    wait_for(lambda: recorder.stats().written >= 5)
    engine.end_take()
    assert recorder.wait(20.0)
    engine.stop()
    assert engine.join(5.0)
    assert recorder.result.status == "completed" and recorder.result.verified
    assert (recorder.result.width, recorder.result.height) == (1280, 720)
    assert source.closed == 1
    data = sidecar_of(tmp_path)
    assert data["camera"]["type"] == "FakeSource"
    assert data["camera"]["index"] == 3 and data["camera"]["backend"] == "fake"


def test_engine_reports_a_camera_that_will_not_open() -> None:
    class Broken(FakeSource):
        def open(self) -> None:
            raise CameraError("カメラ 7 を開けませんでした。")

    engine = CaptureEngine(lambda _index: Broken())
    engine.start(7)
    wait_for(lambda: engine.snapshot().state == "failed")
    assert "カメラ 7" in engine.snapshot().message
    assert engine.join(5.0)


def test_engine_survives_unexpected_exceptions_from_the_source() -> None:
    class Flaky(FakeSource):
        def read(self) -> np.ndarray | None:
            self._count += 1
            if self._count % 2:
                raise RuntimeError("driver hiccup")
            time.sleep(0.002)
            return numbered_frame(self._count)

    engine = CaptureEngine(lambda _index: Flaky())
    engine.start(0)
    wait_for(lambda: engine.snapshot().preview_seq > 5)
    engine.stop()
    assert engine.join(5.0)


def test_snapshot_stays_fast_while_read_hangs() -> None:
    release = threading.Event()

    class Hanging(FakeSource):
        def read(self) -> np.ndarray | None:
            release.wait(10.0)
            return None

    engine = CaptureEngine(lambda _index: Hanging())
    engine.start(0)
    wait_for(lambda: engine.snapshot().state != "opening")
    deadline = time.monotonic() + 1.2
    slowest = 0.0
    while time.monotonic() < deadline:
        start = time.perf_counter()
        snapshot = engine.snapshot()
        slowest = max(slowest, time.perf_counter() - start)
        time.sleep(0.01)
    assert slowest < 0.1
    assert snapshot.state == "no_frames"  # noticed although read() never came back
    release.set()
    engine.stop()
    assert engine.join(5.0)


def test_engine_can_switch_cameras_and_releases_the_old_one() -> None:
    sources = []

    def factory(index: int) -> FakeSource:
        sources.append(FakeSource())
        return sources[-1]

    engine = CaptureEngine(factory)
    engine.start(0)
    wait_for(lambda: engine.snapshot().preview is not None)
    engine.start(1)
    wait_for(lambda: len(sources) == 2 and engine.snapshot().preview is not None)
    assert sources[0].closed == 1 and sources[1].closed == 0
    engine.stop()
    assert engine.join(5.0)
    assert sources[1].closed == 1


# --------------------------------------------------------------------------
# The window
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    return QApplication.instance() or QApplication([])


@pytest.fixture
def make_window(qapp: QApplication) -> Iterator[Callable[..., RecorderWindow]]:
    windows: list[RecorderWindow] = []

    def make(**kwargs) -> RecorderWindow:
        kwargs.setdefault("countdown_s", 0.05)
        kwargs.setdefault("script_provider", short_script)
        kwargs.setdefault("today", lambda: date(2026, 11, 10))
        source_factory = kwargs.pop("source_factory", lambda _index: FakeSource())
        window = RecorderWindow(source_factory, **kwargs)
        windows.append(window)
        return window

    yield make
    for window in windows:
        window._force_close = True
        window.close()
        window.engine.join(5.0)
        window.deleteLater()
    QApplication.processEvents()


def pump_until(condition: Callable[[], bool], timeout_s: float = 8.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("condition not met in time")
        QApplication.processEvents()
        time.sleep(0.003)


def start_recording_session(window: RecorderWindow, output: Path) -> None:
    window.output_edit.setText(str(output))
    window.check_output()
    window.start_session()
    assert window.record_page is window._stack.currentWidget()
    pump_until(lambda: window.primary_button.isEnabled())


def test_start_needs_an_explicit_folder(make_window, tmp_path: Path) -> None:
    window = make_window()
    assert window.output_edit.text() == ""
    assert not window.start_button.isEnabled()
    window.start_session()
    assert window._stack.currentWidget() is window.setup_page
    window.output_edit.setText(str(tmp_path / "out"))
    assert window.start_button.isEnabled()


def test_repository_folder_is_refused_with_a_clear_message(make_window, tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    window = make_window(output=str(tmp_path / "videos"))
    assert "Git の作業フォルダ" in window.output_message.text()
    assert not window.start_button.isEnabled()
    window.start_session()
    assert window._stack.currentWidget() is window.setup_page
    assert not (tmp_path / "videos").exists()


def test_typing_a_new_folder_clears_the_refusal(make_window, tmp_path: Path) -> None:
    (tmp_path / "repo" / ".git").mkdir(parents=True)
    window = make_window(output=str(tmp_path / "repo"))
    assert not window.start_button.isEnabled()
    window.output_edit.setText(str(tmp_path / "elsewhere"))
    assert window.output_message.text() == ""
    assert window.start_button.isEnabled()
    window.check_output()
    assert "保存先" in window.output_message.text() and "空き容量" in window.output_message.text()


def test_storage_advice_is_on_the_first_page(make_window) -> None:
    window = make_window()
    text = window.storage_advice.text()
    for phrase in ("ディープフェイク", "公開リポジトリ", "暗号化", "BitLocker", "Secure Cloud", "インターネットに何も送りません"):
        assert phrase in text


def test_cloud_folder_needs_acknowledgement(make_window, tmp_path: Path) -> None:
    window = make_window(output=str(tmp_path / "OneDrive" / "videos"))
    assert "同期" in window.output_message.text()
    assert not window.start_button.isEnabled()
    window.cloud_ack.setChecked(True)
    assert window.start_button.isEnabled()


def test_session_and_condition_choices_cover_the_guide(make_window) -> None:
    window = make_window(script_provider=script_for)
    assert window.session_combo.count() == 4
    assert [window.condition_combo.itemText(i) for i in range(2)] == ["すっぴん", "メイク"]
    window.session_combo.setCurrentIndex(0)
    assert "すっぴん 15 分 → メイク 15 分" in window.session_hint.text()
    window.session_combo.setCurrentIndex(3)
    assert "別の日" in window.session_hint.text()


def test_unwritable_folder_is_refused_before_she_can_press_start(make_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    deny_new_files(monkeypatch)
    started = time.perf_counter()
    window = make_window(output=str(tmp_path))
    assert time.perf_counter() - started < 5.0
    assert "書き込めません" in window.output_message.text()
    assert not window.start_button.isEnabled()
    window.start_button.click()
    window.start_session()
    assert window._stack.currentWidget() is window.setup_page


def test_folder_that_turns_unwritable_after_the_check_is_reported(make_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    window = make_window(output=str(tmp_path))
    assert window.start_button.isEnabled()
    monkeypatch.setattr(record, "inspect_output_dir", lambda text: record.OutputFolderCheck(tmp_path, None))
    deny_new_files(monkeypatch)
    window.start_session()
    assert window._stack.currentWidget() is window.setup_page
    assert "書き込めません" in window.setup_error.text()


def test_a_folder_check_that_hangs_does_not_freeze_the_window(make_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    release = threading.Event()

    def hang(_text: str) -> record.OutputFolderCheck:
        release.wait(20.0)
        raise OutputFolderError("too late")

    window = make_window()
    monkeypatch.setattr(record, "FOLDER_CHECK_TIMEOUT_S", 0.1)
    monkeypatch.setattr(record, "inspect_output_dir", hang)
    window.output_edit.setText(str(tmp_path))
    started = time.perf_counter()
    assert window.check_output() is None
    assert time.perf_counter() - started < 2.0
    assert "応答がありません" in window.output_message.text()
    assert not window.start_button.isEnabled()
    release.set()


def test_an_os_error_while_checking_leaves_the_window_working(make_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def locked(_path: Path) -> Path | None:
        raise OSError(1, "locked")

    monkeypatch.setattr(record, "find_git_root", locked)
    window = make_window(output=str(tmp_path))
    assert "確認できません" in window.output_message.text()
    assert not window.start_button.isEnabled()


@pytest.mark.skipif(sys.platform != "win32", reason="Windows device names")
def test_a_device_name_in_the_folder_box_is_refused(make_window) -> None:
    window = make_window()
    for text in ("CON", "NUL", "PRN", "AUX"):
        window.output_edit.setText(text)
        assert window.check_output() is None
        assert "予約" in window.output_message.text() and not window.start_button.isEnabled()


class SmallScreen:
    """Logical 1280x720 less a task bar: a 1920x1080 laptop at 150 %, roughly. The offscreen fonts need the width."""

    def availableGeometry(self) -> QRect:
        return QRect(100, 50, 1280, 672)


def test_first_window_fits_the_screen(make_window) -> None:
    window = make_window()
    window.screen = SmallScreen  # asked for its available geometry when the window is first shown
    window.move(500, 400)
    window.show()
    QApplication.processEvents()
    available = SmallScreen().availableGeometry()
    assert available.contains(window.frameGeometry())
    assert window.height() < 780 and window.width() == 1200


def test_a_window_is_fitted_only_once(make_window) -> None:
    window = make_window()
    window.screen = SmallScreen
    window.show()
    window.resize(1000, 600)  # what she chooses later is her business
    window.hide()
    window.show()
    assert (window.width(), window.height()) == (1000, 600)


def test_both_pages_can_shrink_to_a_small_laptop_screen(make_window, tmp_path: Path) -> None:
    window = make_window(script_provider=script_for)
    window.show()
    assert window.setup_page.minimumSizeHint().height() < 535  # 1366x768 at 125 %
    start_recording_session(window, tmp_path)
    for index in range(len(window._script)):
        window._step_index = index
        window._show_step()
        QApplication.processEvents()
        assert window.record_page.minimumSizeHint().height() < 535


def test_long_cues_get_a_smaller_font_not_more_lines(make_window) -> None:
    window = make_window(script_provider=script_for)
    window.resize(800, 700)
    window.show()
    label = window.cue_label
    label.resize(760, label.height())
    short, long = "下ろす", training_script(5)[4].instruction
    label.setText(short)
    assert label.font().pointSizeF() == record.CueLabel.SIZES[0]
    label.setText(long)
    assert label.font().pointSizeF() < record.CueLabel.SIZES[0]
    label.setText(short)
    assert label.font().pointSizeF() == record.CueLabel.SIZES[0]


def test_held_keys_do_not_run_through_the_steps(make_window, tmp_path: Path) -> None:
    window = make_window()
    assert all(not shortcut.autoRepeat() for shortcut in window._shortcuts)


def test_full_flow_records_each_step_to_its_own_file(make_window, tmp_path: Path) -> None:
    window = make_window()
    start_recording_session(window, tmp_path)
    assert window.session_dir == tmp_path.resolve() / "sessions" / "2026-11-10_main_bare"

    # Step 1: countdown, recording with cues, automatic stop, review.
    assert window.phase is Phase.READY and window.step_index == 0
    assert window.cue_label.text() == "最初の指示"
    window.primary_action()
    assert window.phase is Phase.COUNTDOWN
    assert window.cue_label.text() == "前半"
    pump_until(lambda: window.phase is Phase.RECORDING)
    assert window.primary_button.text().startswith("停止")
    pump_until(lambda: window.cue_label.text() == "後半" or window.phase is not Phase.RECORDING)
    pump_until(lambda: window.phase is Phase.REVIEW)
    first = window.results[0]
    assert first.status == "completed" and first.verified and first.frames_written > 5
    assert "再生確認 OK" in window.result_label.text()
    assert window.next_button.isEnabled() and window.retake_button.isEnabled()

    # Next: step 2.
    window.go_next()
    assert window.step_index == 1 and window.phase is Phase.READY
    window.primary_action()
    pump_until(lambda: window.phase is Phase.REVIEW)

    # Next: step 3 is ended by hand.
    window.go_next()
    window.primary_action()
    pump_until(lambda: window.phase is Phase.RECORDING)
    pump_until(lambda: window._recorder.stats().written > 3)
    window.primary_action()
    assert window.phase is Phase.SAVING
    pump_until(lambda: window.phase is Phase.REVIEW)
    assert window.results[2].status == "completed"

    # Done page.
    window.go_next()
    assert window.phase is Phase.DONE
    assert "おつかれさまでした" in window.cue_label.text()

    folder = window.session_dir
    names = sorted(path.name for path in folder.iterdir())
    assert names == [
        "step01_one_take01.json",
        "step01_one_take01.mkv",
        "step02_two_take01.json",
        "step02_two_take01.mkv",
        "step03_three_take01.json",
        "step03_three_take01.mkv",
    ]
    for stem, index in (("step01_one_take01", 0), ("step02_two_take01", 1), ("step03_three_take01", 2)):
        data = json.loads((folder / f"{stem}.json").read_text(encoding="utf-8"))
        assert data["status"] == "completed" and data["verified"]["ok"]
        frames = read_all_frames(folder / f"{stem}.mkv")
        assert len(frames) == data["frames"]["written"] == window.results[index].frames_written
        assert frames[0].shape == (TINY[1], TINY[0], 3)


def test_the_real_buttons_are_wired(make_window, tmp_path: Path) -> None:
    window = make_window()
    window.output_edit.setText(str(tmp_path))
    window.start_button.click()
    assert window._stack.currentWidget() is window.record_page
    pump_until(lambda: window.primary_button.isEnabled())
    window.primary_button.click()
    pump_until(lambda: window.phase is Phase.REVIEW)
    window.retake_button.click()
    assert window.phase is Phase.COUNTDOWN
    window.primary_button.click()  # cancels
    assert window.phase is Phase.REVIEW
    window.next_button.click()
    assert window.step_index == 1
    window.back_button.click()
    assert window.step_index == 0
    window.setup_button.click()
    assert window._stack.currentWidget() is window.setup_page


def test_back_and_next_move_between_steps(make_window, tmp_path: Path) -> None:
    window = make_window()
    start_recording_session(window, tmp_path)
    assert not window.back_button.isEnabled()
    window.go_back()
    assert window.step_index == 0
    window.go_next()
    assert window.step_index == 1 and window.back_button.isEnabled()
    window.go_back()
    assert window.step_index == 0
    assert "最初の指示" in window.cue_label.text()


def test_retake_from_review_keeps_the_earlier_take(make_window, tmp_path: Path) -> None:
    window = make_window()
    start_recording_session(window, tmp_path)
    window.primary_action()
    pump_until(lambda: window.phase is Phase.REVIEW)
    assert window.results[0].take == 1
    window.retake()
    assert window.phase is Phase.COUNTDOWN
    pump_until(lambda: window.phase is Phase.RECORDING)
    pump_until(lambda: window.phase is Phase.REVIEW)
    assert window.results[0].take == 2
    assert (window.session_dir / "step01_one_take01.mkv").exists()
    assert (window.session_dir / "step01_one_take02.mkv").exists()


def test_retake_during_recording_restarts_the_step(make_window, tmp_path: Path) -> None:
    window = make_window()
    start_recording_session(window, tmp_path)
    window.go_next()
    window.go_next()  # the step ended by hand
    window.primary_action()
    pump_until(lambda: window.phase is Phase.RECORDING and window._recorder.stats().written > 2)
    window.retake()
    pump_until(lambda: window.phase is Phase.RECORDING and window._recorder.stats().take == 2)
    pump_until(lambda: window._recorder.stats().written > 2)
    first = json.loads((window.session_dir / "step03_three_take01.json").read_text(encoding="utf-8"))
    assert first["status"] == "aborted" and first["stop_reason"] == "retake"
    assert 2 not in window.results  # an aborted take does not count as done
    window.primary_action()
    pump_until(lambda: window.phase is Phase.REVIEW)
    assert window.results[2].take == 2


def test_cancel_during_countdown_records_nothing(make_window, tmp_path: Path) -> None:
    window = make_window(countdown_s=30.0)
    start_recording_session(window, tmp_path)
    window.primary_action()
    assert window.phase is Phase.COUNTDOWN
    pump_until(lambda: window.preview._countdown != "")
    window.primary_action()
    assert window.phase is Phase.READY
    assert window.preview._countdown == ""
    assert not list(tmp_path.rglob("*.mkv"))


def test_cannot_start_without_a_working_camera(make_window, tmp_path: Path) -> None:
    class Broken(FakeSource):
        def open(self) -> None:
            raise CameraError("カメラ 0 を開けませんでした。")

    window = make_window(source_factory=lambda _index: Broken())
    window.output_edit.setText(str(tmp_path))
    window.start_session()
    pump_until(lambda: "開けませんでした" in window.preview._message)
    assert not window.primary_button.isEnabled()
    window.primary_action()
    assert window.phase is Phase.READY
    window.back_to_setup()
    assert window._stack.currentWidget() is window.setup_page


def test_dropped_frames_are_shown_live(make_window, tmp_path: Path) -> None:
    def slow(path: Path, fourcc: str, fps: float, size: tuple[int, int]) -> SlowWriter:
        return SlowWriter(path, fourcc, fps, size, 0.05)

    window = make_window(writer_factory=slow, source_factory=lambda _index: FakeSource(fps=100.0))
    start_recording_session(window, tmp_path)
    window.go_next()
    window.go_next()
    window.primary_action()
    pump_until(lambda: window.phase is Phase.RECORDING)
    # Force the bounded queue to overflow by making it tiny for this take.
    window._recorder._queue_bytes = numbered_frame(0).nbytes * 2
    pump_until(lambda: window._recorder.stats().dropped_queue_full > 3)
    pump_until(lambda: re.search(r"ドロップ: [1-9]", window.dropped_label.text()) is not None)
    window.primary_action()
    pump_until(lambda: window.phase is Phase.REVIEW)
    assert window.results[2].dropped_frames > 3


def test_ui_keeps_ticking_while_the_disk_hangs(make_window, tmp_path: Path) -> None:
    release = threading.Event()

    class Stuck(SlowWriter):
        def write(self, frame: np.ndarray) -> None:
            release.wait(10.0)
            self._inner.write(frame)

    window = make_window(writer_factory=lambda path, fourcc, fps, size: Stuck(path, fourcc, fps, size, 0))
    start_recording_session(window, tmp_path)
    window.go_next()
    window.go_next()
    window.primary_action()
    pump_until(lambda: window.phase is Phase.RECORDING)
    slowest = 0.0
    seq = window.engine.snapshot().preview_seq
    deadline = time.monotonic() + 0.5
    while time.monotonic() < deadline:
        start = time.perf_counter()
        QApplication.processEvents()
        slowest = max(slowest, time.perf_counter() - start)
        time.sleep(0.005)
    assert slowest < 0.25
    assert window.engine.snapshot().preview_seq > seq + 5  # the camera and preview kept going
    window.primary_action()
    assert window.phase is Phase.SAVING
    release.set()
    pump_until(lambda: window.phase is Phase.REVIEW, 15.0)


def test_failed_take_is_reported_and_can_be_retaken(make_window, tmp_path: Path) -> None:
    class Shrinking(FakeSource):
        def read(self) -> np.ndarray | None:
            time.sleep(self._period)
            self._count += 1
            return numbered_frame(self._count, (32, 24) if self._count % 30 == 0 else TINY)

    window = make_window(source_factory=lambda _index: Shrinking())
    start_recording_session(window, tmp_path)
    window.go_next()
    window.go_next()
    window.primary_action()
    pump_until(lambda: window.phase is Phase.FAILED, 15.0)
    assert 2 not in window.results
    assert window.cue_label.text() == "録画に失敗しました"
    assert "解像度" in window.next_label.text()
    assert window.retake_button.isEnabled()


class SwitchableSource(FakeSource):
    """A camera that can stop delivering, like one whose cable is pulled.

    At the real frame rate: on a busy machine a period of 10 ms would look like stalls all the time.
    """

    def __init__(self, fps: float = 30.0) -> None:
        super().__init__(fps)
        self.gone = False

    def read(self) -> np.ndarray | None:
        time.sleep(self._period)
        if self.gone:
            return None
        self._count += 1
        return numbered_frame(self._count, self._size)


@pytest.fixture
def beeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    heard: list[float] = []
    monkeypatch.setattr(record, "_beep", lambda: heard.append(time.monotonic()))
    return heard


def one_step(target_s: float, *, auto_stop: bool = True) -> Callable[[record.Session, record.Condition], tuple[ScriptStep, ...]]:
    def provider(_session: record.Session, _condition: record.Condition) -> tuple[ScriptStep, ...]:
        return (ScriptStep("one", "一", "指示", target_s, "目安", auto_stop=auto_stop), ScriptStep("two", "二", "次の指示", target_s, "目安"))

    return provider


def test_a_failed_take_cannot_be_mistaken_for_a_good_one(make_window, tmp_path: Path, beeps: list[float]) -> None:
    source = SwitchableSource()
    window = make_window(source_factory=lambda _index: source, script_provider=one_step(1.6))
    start_recording_session(window, tmp_path)
    window.primary_action()
    pump_until(lambda: window.phase is Phase.RECORDING and window._recorder.stats().written > 5)
    beeps.clear()
    source.gone = True
    pump_until(lambda: window.phase is Phase.FAILED, 15.0)
    assert 0 not in window.results and "カメラの映像が途切れて" in window.next_label.text()
    # In the place and the size she is looking at, in colours nothing else uses, and to be heard.
    assert window.cue_label.text() == "録画に失敗しました"
    assert window.script_panel.property("alert") is True
    assert len(beeps) >= 2  # the stop at the target time, then the alarm
    pump_until(lambda: len(beeps) >= 4)  # the alarm's other two
    # Not the buttons of a good take.
    assert window.primary_button.text() == "撮り直し (Space)"
    assert window.next_button.property("accent") is False and window.primary_button.property("accent") is True
    assert "✕" in window._step_cell(0)
    source.gone = False
    pump_until(lambda: window.primary_button.isEnabled())
    # Space is the retake here, not the next step.
    window._on_space()
    assert window.phase is Phase.COUNTDOWN and window.step_index == 0
    assert window.script_panel.property("alert") is False
    pump_until(lambda: window.phase is Phase.REVIEW, 15.0)
    assert window.results[0].take == 2 and "✕" not in window._step_cell(0)


def test_after_a_failure_she_can_still_move_on_on_purpose(make_window, tmp_path: Path) -> None:
    source = SwitchableSource()
    window = make_window(source_factory=lambda _index: source, script_provider=one_step(1.6))
    start_recording_session(window, tmp_path)
    window.primary_action()
    pump_until(lambda: window.phase is Phase.RECORDING)
    source.gone = True
    pump_until(lambda: window.phase is Phase.FAILED, 15.0)
    source.gone = False
    window.next_button.click()
    assert window.step_index == 1 and window.phase is Phase.READY
    assert window.script_panel.property("alert") is False
    window.back_button.click()
    assert window.step_index == 0 and window.phase is Phase.FAILED  # it has not gone away
    window._step_index = 1
    window._show_step()
    window.go_next()
    assert window.phase is Phase.DONE and "一" in window.next_label.text()  # listed as not filmed yet


def test_a_failed_retake_leaves_the_earlier_take_in_place(make_window, tmp_path: Path) -> None:
    source = SwitchableSource()
    window = make_window(source_factory=lambda _index: source, script_provider=one_step(1.6))
    start_recording_session(window, tmp_path)
    window.primary_action()
    pump_until(lambda: window.phase is Phase.REVIEW, 15.0)
    assert window.results[0].take == 1
    window.retake()
    pump_until(lambda: window.phase is Phase.RECORDING)
    source.gone = True
    pump_until(lambda: window.phase is Phase.FAILED, 15.0)
    assert window.results[0].take == 1
    assert "前のテイク 1" in window.result_label.text()
    assert load_completed_takes(window.session_dir)[1].take == 1


def test_camera_loss_alerts_her_during_the_take(make_window, tmp_path: Path, beeps: list[float], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(record, "CAMERA_ALERT_S", 0.5)
    source = SwitchableSource()
    window = make_window(source_factory=lambda _index: source, script_provider=one_step(600.0, auto_stop=False))
    start_recording_session(window, tmp_path)
    window.primary_action()
    pump_until(lambda: window.phase is Phase.RECORDING and window._recorder.stats().written > 5)
    assert window.script_panel.property("alert") is False
    beeps.clear()
    source.gone = True
    pump_until(lambda: window.script_panel.property("alert") is True)
    assert window.phase is Phase.RECORDING  # it may still come back
    assert window.cue_label.text() == "カメラの映像が止まっています"
    assert re.search(r"ドロップ: [1-9]", window.dropped_label.text())
    assert beeps
    pump_until(lambda: window.preview._stalled, 3.0)  # the frozen picture no longer looks alive

    source.gone = False
    pump_until(lambda: window.script_panel.property("alert") is False and window.cue_label.text() == "指示")
    pump_until(lambda: not window.preview._stalled)
    window.primary_action()
    pump_until(lambda: window.phase in (Phase.REVIEW, Phase.FAILED), 15.0)


def test_a_camera_gone_for_good_ends_in_the_failed_state(make_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(record, "CAMERA_ALERT_S", 0.1)
    monkeypatch.setattr(record, "CAMERA_LOST_STOP_S", 0.4)
    monkeypatch.setattr(record, "HOUSEKEEPING_INTERVAL_S", 0.05)
    monkeypatch.setattr(record, "MIN_LOST_TO_FAIL_S", 0.0)
    source = SwitchableSource()
    window = make_window(source_factory=lambda _index: source, script_provider=one_step(600.0, auto_stop=False))
    start_recording_session(window, tmp_path)
    window.primary_action()
    pump_until(lambda: window.phase is Phase.RECORDING and window._recorder.stats().written > 2)
    source.gone = True
    pump_until(lambda: window.phase is Phase.FAILED, 15.0)
    assert "途切れて" in window.next_label.text()
    assert not window.primary_button.isEnabled() or window._camera_ready()  # the retake waits for the camera


def test_a_camera_lost_late_in_a_long_take_leaves_the_take_to_her(make_window, tmp_path: Path, beeps: list[float], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(record, "CAMERA_ALERT_S", 0.1)
    monkeypatch.setattr(record, "CAMERA_LOST_STOP_S", 0.4)
    monkeypatch.setattr(record, "HOUSEKEEPING_INTERVAL_S", 0.05)
    monkeypatch.setattr(record, "MAX_LOST_RATIO", 1.0)  # that part has its own tests; this take is short
    source = SwitchableSource()
    window = make_window(source_factory=lambda _index: source, script_provider=one_step(600.0, auto_stop=False))
    start_recording_session(window, tmp_path)
    window.primary_action()
    pump_until(lambda: window.phase is Phase.RECORDING and window._recorder.stats().written > 5)
    beeps.clear()
    source.gone = True
    pump_until(lambda: window.phase is Phase.REVIEW, 15.0)
    assert window.results[0].stop_reason == "camera_lost"
    assert "録画を止めました" in window.next_label.text() and "保存してあります" in window.next_label.text()
    assert beeps  # stopped without her asking: she has to hear it


def test_a_take_losing_too_much_says_so_while_it_runs(make_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(record, "LOST_WARNING_AFTER_S", 0.1)
    source = SwitchableSource()
    window = make_window(source_factory=lambda _index: source, script_provider=one_step(600.0, auto_stop=False))
    start_recording_session(window, tmp_path)
    window.primary_action()
    pump_until(lambda: window.phase is Phase.RECORDING and window._recorder.stats().written > 3)
    source.gone = True
    pump_until(lambda: window._recorder.stats().frame_age_s > 0.7)
    source.gone = False
    pump_until(lambda: "欠けています" in window.next_label.text())
    assert window.script_panel.property("alert") is False and window.cue_label.text() == "指示"  # the cues stay
    window.primary_action()
    pump_until(lambda: window.phase in (Phase.REVIEW, Phase.FAILED, Phase.READY), 15.0)


def test_countdown_ending_without_a_camera_starts_nothing(make_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = SwitchableSource()
    window = make_window(source_factory=lambda _index: source, countdown_s=1.5)
    start_recording_session(window, tmp_path)
    window.primary_action()
    assert window.phase is Phase.COUNTDOWN
    source.gone = True
    pump_until(lambda: window.phase is not Phase.COUNTDOWN, 8.0)
    assert window.phase is Phase.READY and "届いていない" in window.next_label.text()
    assert not list(tmp_path.rglob("*.mkv")) and not list(tmp_path.rglob("*.json"))


def test_saving_that_hangs_says_so(make_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(record, "SAVE_STALL_WARNING_S", 0.2)
    release = threading.Event()

    class Stuck(SlowWriter):
        def write(self, frame: np.ndarray) -> None:
            release.wait(30.0)
            self._inner.write(frame)

    window = make_window(writer_factory=lambda path, fourcc, fps, size: Stuck(path, fourcc, fps, size, 0))
    start_recording_session(window, tmp_path)
    window.go_next()
    window.go_next()
    window.primary_action()
    pump_until(lambda: window.phase is Phase.RECORDING and window._recorder.stats().captured > 2)
    window.primary_action()
    assert window.phase is Phase.SAVING
    pump_until(lambda: window.script_panel.property("alert") is True)
    assert window.cue_label.text() == "保存が止まっています"
    release.set()
    pump_until(lambda: window.phase is not Phase.SAVING, 15.0)
    assert window.script_panel.property("alert") is False


def test_a_writer_stuck_during_the_take_alerts_her(make_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(record, "WRITE_ALERT_S", 0.2)
    release = threading.Event()

    class Stuck(SlowWriter):
        def write(self, frame: np.ndarray) -> None:
            release.wait(30.0)
            self._inner.write(frame)

    window = make_window(writer_factory=lambda path, fourcc, fps, size: Stuck(path, fourcc, fps, size, 0))
    start_recording_session(window, tmp_path)
    window.go_next()
    window.go_next()
    window.primary_action()
    pump_until(lambda: window.phase is Phase.RECORDING)
    pump_until(lambda: window.script_panel.property("alert") is True)
    assert window.cue_label.text() == "保存が止まっています"
    release.set()
    pump_until(lambda: window.script_panel.property("alert") is False)
    window.primary_action()
    pump_until(lambda: window.phase in (Phase.REVIEW, Phase.FAILED), 15.0)


def test_a_slip_of_the_finger_in_a_retake_keeps_the_good_take(make_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    window = make_window(script_provider=one_step(600.0, auto_stop=False))
    start_recording_session(window, tmp_path)
    window.primary_action()
    pump_until(lambda: window.phase is Phase.RECORDING and window._recorder.stats().written > 3)
    window.primary_action()
    pump_until(lambda: window.phase is Phase.REVIEW)
    assert window.results[0].take == 1

    monkeypatch.setattr(record, "MIN_USABLE_TAKE_S", 3.0)
    window.retake()
    pump_until(lambda: window.phase is Phase.RECORDING and window._recorder.stats().written > 3)
    window.primary_action()
    pump_until(lambda: window.phase is Phase.REVIEW)
    assert window.results[0].take == 1
    assert "テイク 2 はすぐに止めた" in window.next_label.text() and "前のテイク 1" in window.next_label.text()
    assert "テイク 1 を保存しました" in window.result_label.text()
    first = json.loads((window.session_dir / "step01_one_take02.json").read_text(encoding="utf-8"))
    assert first["status"] == "aborted"


def test_a_slip_of_the_finger_on_the_first_take_leaves_the_step_to_do(make_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(record, "MIN_USABLE_TAKE_S", 3.0)
    window = make_window(script_provider=one_step(600.0, auto_stop=False))
    start_recording_session(window, tmp_path)
    window.primary_action()
    pump_until(lambda: window.phase is Phase.RECORDING and window._recorder.stats().written > 3)
    window.primary_action()
    pump_until(lambda: window.phase is Phase.READY)
    assert 0 not in window.results and "すぐに止めた" in window.next_label.text()


def test_a_much_shorter_take_is_kept_with_a_word_about_it(make_window, tmp_path: Path) -> None:
    window = make_window(script_provider=one_step(600.0, auto_stop=False))
    start_recording_session(window, tmp_path)
    window.primary_action()
    pump_until(lambda: window.phase is Phase.RECORDING and window._recorder.stats().written > 3)
    window.primary_action()
    pump_until(lambda: window.phase is Phase.REVIEW)
    assert "短いテイク" in window.next_label.text()


def test_the_talking_step_keeps_its_reminder_on_screen(make_window, tmp_path: Path) -> None:
    def talk_only(session: record.Session, condition: record.Condition) -> tuple[ScriptStep, ...]:
        return tuple(step for step in script_for(session, condition) if step.key == "talk")

    window = make_window(script_provider=talk_only)
    start_recording_session(window, tmp_path)
    assert "自己PR、ガクチカ、志望動機、逆質問" in window.cue_label.text()
    window.primary_action()
    pump_until(lambda: window.phase is Phase.RECORDING)
    pump_until(lambda: window.cue_label.text() == "自己PR")
    assert "次: ガクチカ" in window.next_label.text()
    assert "相づち・笑顔・考える仕草" in window.next_label.text()
    window.retake()  # to end it; a real press of stop is tested elsewhere
    pump_until(lambda: window.phase is Phase.RECORDING and window._recorder.stats().take == 2)
    window.primary_action()
    pump_until(lambda: window.phase is not Phase.SAVING)


def test_closing_during_a_take_saves_it_first(make_window, tmp_path: Path) -> None:
    window = make_window()
    window.show()
    start_recording_session(window, tmp_path)
    window.go_next()
    window.go_next()
    window.primary_action()
    pump_until(lambda: window.phase is Phase.RECORDING and window._recorder.stats().written > 3)
    folder = window.session_dir
    assert window.close() is False  # not yet: the file is still being finished
    pump_until(lambda: not window.isVisible())
    data = json.loads((folder / "step03_three_take01.json").read_text(encoding="utf-8"))
    assert data["status"] == "interrupted" and data["stop_reason"] == "close"  # saved, but not for use
    assert len(read_all_frames(folder / "step03_three_take01.mkv")) == data["frames"]["written"] > 3
    assert load_completed_takes(folder) == {}
    assert window.engine.join(5.0)


def test_closing_during_a_retake_does_not_replace_the_good_take(make_window, tmp_path: Path) -> None:
    window = make_window()
    window.show()
    start_recording_session(window, tmp_path)
    window.primary_action()
    pump_until(lambda: window.phase is Phase.REVIEW)
    window.retake()
    pump_until(lambda: window.phase is Phase.RECORDING and window._recorder.stats().written > 2)
    folder = window.session_dir
    assert window.close() is False
    pump_until(lambda: not window.isVisible())
    assert json.loads((folder / "step01_one_take02.json").read_text(encoding="utf-8"))["status"] == "interrupted"
    assert load_completed_takes(folder)[1].take == 1


def test_reopening_after_closing_during_a_take_returns_to_that_step(make_window, tmp_path: Path) -> None:
    window = make_window()
    window.show()
    start_recording_session(window, tmp_path)
    window.primary_action()
    pump_until(lambda: window.phase is Phase.RECORDING and window._recorder.stats().written > 2)
    window.close()
    pump_until(lambda: not window.isVisible())
    window.engine.join(5.0)

    again = make_window()
    start_recording_session(again, tmp_path)
    assert again.step_index == 0 and again.phase is Phase.READY and not again.results
    again.primary_action()
    pump_until(lambda: again.phase is Phase.REVIEW)
    assert again.results[0].take == 2  # the closed one stays on disk under its own number


def test_stopping_before_closing_counts_as_stopped(make_window, tmp_path: Path) -> None:
    window = make_window()
    window.show()
    start_recording_session(window, tmp_path)
    window.go_next()
    window.go_next()
    window.primary_action()
    pump_until(lambda: window.phase is Phase.RECORDING and window._recorder.stats().written > 3)
    folder = window.session_dir
    window.primary_action()  # stopped by hand, the window closed while saving
    window.close()
    pump_until(lambda: not window.isVisible())
    assert json.loads((folder / "step03_three_take01.json").read_text(encoding="utf-8"))["status"] == "completed"


def test_closing_gives_up_waiting_for_a_hung_disk(make_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(record, "CLOSE_TIMEOUT_S", 0.1)
    release = threading.Event()

    class Stuck(SlowWriter):
        def write(self, frame: np.ndarray) -> None:
            release.wait(10.0)

    window = make_window(writer_factory=lambda path, fourcc, fps, size: Stuck(path, fourcc, fps, size, 0))
    window.show()
    start_recording_session(window, tmp_path)
    window.go_next()
    window.go_next()
    window.primary_action()
    pump_until(lambda: window.phase is Phase.RECORDING)
    window.close()
    pump_until(lambda: not window.isVisible())
    release.set()


def test_an_earlier_session_can_be_continued(make_window, tmp_path: Path) -> None:
    window = make_window()
    start_recording_session(window, tmp_path)
    window.primary_action()
    pump_until(lambda: window.phase is Phase.REVIEW)
    window.back_to_setup()

    again = make_window()
    start_recording_session(again, tmp_path)
    assert again.step_index == 1  # the first step without a take
    assert 0 in again.results
    again.go_back()
    assert again.phase is Phase.REVIEW
    assert "テイク 1" in again.result_label.text()


def test_makeup_condition_gets_its_own_folder(make_window, tmp_path: Path) -> None:
    window = make_window()
    window.condition_combo.setCurrentIndex(1)
    window.session_combo.setCurrentIndex(1)
    start_recording_session(window, tmp_path)
    assert window.session_dir.name == "2026-11-10_window_makeup"
    assert window.session_dir.parent.name == "sessions"


def test_evaluation_session_goes_to_the_eval_folder(make_window, tmp_path: Path) -> None:
    window = make_window(script_provider=script_for)
    window.session_combo.setCurrentIndex(3)
    start_recording_session(window, tmp_path)
    assert window.session_dir.parent.name == "eval"
    assert window.step_title.text().startswith("ステップ 1 / 1")


def test_keyboard_shortcuts_work_only_on_the_recording_page(make_window, tmp_path: Path) -> None:
    window = make_window(countdown_s=30.0)
    window.show()
    assert not any(shortcut.isEnabled() for shortcut in window._shortcuts)
    start_recording_session(window, tmp_path)
    assert all(shortcut.isEnabled() for shortcut in window._shortcuts)
    keys = {shortcut.key()[0].key(): shortcut for shortcut in window._shortcuts}
    keys[Qt.Key.Key_Space].activated.emit()
    assert window.phase is Phase.COUNTDOWN
    keys[Qt.Key.Key_Escape].activated.emit()
    assert window.phase is Phase.READY
    keys[Qt.Key.Key_Right].activated.emit()
    assert window.step_index == 1
    keys[Qt.Key.Key_Left].activated.emit()
    assert window.step_index == 0
    window.back_to_setup()
    assert not any(shortcut.isEnabled() for shortcut in window._shortcuts)


def test_preview_is_mirrored_but_the_file_is_not(make_window, tmp_path: Path) -> None:
    class Asymmetric(FakeSource):
        def read(self) -> np.ndarray | None:
            time.sleep(self._period)
            frame = numbered_frame(0)
            frame[:, :8] = 255  # white on the left edge
            return frame

    window = make_window(source_factory=lambda _index: Asymmetric())
    window.mirror_box.setChecked(True)
    start_recording_session(window, tmp_path)
    pump_until(lambda: window.preview.image is not None)
    image = window.preview.image
    assert image.pixelColor(image.width() - 1, 0).red() == 255  # shown like a mirror
    window.primary_action()
    pump_until(lambda: window.phase is Phase.REVIEW)
    frames = read_all_frames(window.session_dir / window.results[0].video_name)
    assert frames[0][0, 0, 0] == 255 and frames[0][0, -1, 0] == 0  # stored as the camera saw it


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------


def _args(*argv: str):
    return record.build_parser().parse_args(argv)


def test_cli_has_no_default_output() -> None:
    assert _args().output is None


def test_cli_camera_source_is_used_by_default() -> None:
    source = record.build_source_factory(_args("--width", "640", "--height", "480", "--fps", "15"))(2)
    assert isinstance(source, record.CameraSource)
    assert (source.index, source.width, source.height, source.fps) == (2, 640, 480, 15)


def test_cli_camera_file_uses_video_file_source(tmp_path: Path) -> None:
    path = tmp_path / "clip.mkv"
    writer = record.cv2_video_writer(path, "HFYU", 30.0, TINY)
    for number in range(5):
        writer.write(numbered_frame(number))
    writer.release()
    source = record.build_source_factory(_args("--camera-file", str(path), "--width", "128", "--height", "96"))(0)
    assert isinstance(source, camera_module.VideoFileSource)
    source.open()
    try:
        frame = source.read()
        assert frame is not None and frame.shape == (96, 128, 3)
    finally:
        source.close()


def test_cli_missing_file_is_a_short_message(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert record.main(["--camera-file", str(tmp_path / "missing.mp4")]) == 1
    assert "見つかりません" in capsys.readouterr().err


def test_cli_without_video_file_source_support_is_a_short_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delattr(camera_module, "VideoFileSource", raising=False)
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"x")
    assert record.main(["--camera-file", str(path)]) == 1
    assert "VideoFileSource" in capsys.readouterr().err


def test_main_runs_the_window_until_it_is_closed(qapp: QApplication, tmp_path: Path) -> None:
    clip = tmp_path / "clip.mkv"
    writer = record.cv2_video_writer(clip, "HFYU", 30.0, TINY)
    for number in range(5):
        writer.write(numbered_frame(number))
    writer.release()
    QTimer.singleShot(50, QApplication.closeAllWindows)
    assert record.main(["--camera-file", str(clip), "--output", str(tmp_path / "out"), "--countdown", "0"]) == 0
    assert not (tmp_path / "out").exists()  # nothing was started, so nothing was created


_CLOSE_AT_ONCE = """
import sys
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication
from makeup_cam import record
original = record.RecorderWindow.__init__
def init(self, *args, **kwargs):
    original(self, *args, **kwargs)
    QTimer.singleShot(100, QApplication.closeAllWindows)
record.RecorderWindow.__init__ = init
sys.exit(record.main([]))
"""


def test_the_process_ends_without_crashing() -> None:
    """As a real run does, with main() the last to hold the application: widgets outliving it crash the process.

    It was not every time, so a few at once.
    """
    children = [
        subprocess.Popen(
            [sys.executable, "-c", _CLOSE_AT_ONCE],
            cwd=REPO_ROOT,
            env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        for _ in range(4)
    ]
    assert [child.wait(timeout=60) for child in children] == [0, 0, 0, 0]


@pytest.mark.skipif(sys.platform != "win32", reason="Windows device names")
def test_main_survives_a_device_name_as_the_folder(qapp: QApplication, tmp_path: Path) -> None:
    """It used to end in a traceback before any window appeared."""
    clip = tmp_path / "clip.mkv"
    writer = record.cv2_video_writer(clip, "HFYU", 30.0, TINY)
    writer.write(numbered_frame(0))
    writer.release()
    QTimer.singleShot(50, QApplication.closeAllWindows)
    assert record.main(["--camera-file", str(clip), "--output", "CON", "--countdown", "0"]) == 0


def test_recording_through_a_video_file_source_end_to_end(make_window, tmp_path: Path) -> None:
    clip = tmp_path / "clip.mkv"
    writer = record.cv2_video_writer(clip, "HFYU", 30.0, TINY)
    for number in range(20):
        writer.write(numbered_frame(number))
    writer.release()
    factory = record.build_source_factory(_args("--camera-file", str(clip), "--width", "128", "--height", "96"))
    window = make_window(source_factory=factory, camera_label=clip.name)
    assert not window.camera_spin.isEnabled()
    start_recording_session(window, tmp_path / "out")
    window.primary_action()
    pump_until(lambda: window.phase is Phase.REVIEW)
    result = window.results[0]
    assert result.status == "completed" and result.verified and (result.width, result.height) == (128, 96)
    data = json.loads((window.session_dir / result.sidecar_name).read_text(encoding="utf-8"))
    assert data["camera"]["type"] == "VideoFileSource" and data["camera"]["file_name"] == "clip.mkv"
    assert str(tmp_path) not in json.dumps(data, ensure_ascii=False)
    assert len(read_all_frames(window.session_dir / result.video_name)) == result.frames_written
