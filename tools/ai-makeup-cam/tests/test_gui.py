"""Settings window tests, run offscreen with a fake runner (no camera or models)."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import argparse
import json
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Collection, Iterator
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import cv2
import numpy as np
import pytest
import shiboken6 as Shiboken
from PySide6.QtCore import QByteArray, QPoint, QPointF, Qt
from PySide6.QtGui import QColor, QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QAbstractButton, QApplication, QCheckBox, QMainWindow, QPushButton

from makeup_cam import background as plates
from makeup_cam import camera, config, gui
from makeup_cam.analyze import AnalysisError, AnalysisResult
from makeup_cam.config import SECTION_LABELS, SECTIONS, MakeupSettings
from makeup_cam.gui import (
    APP_TITLE,
    FACE_HINT,
    ChoiceEditor,
    ColorEditor,
    FloatEditor,
    GuiOptions,
    GuiState,
    IntEditor,
    MainWindow,
    bgr_to_qimage,
    options_from_args,
)
from makeup_cam.pipeline import MakeupPipeline
from makeup_cam.presets import find_preset, load_last_session, load_preset, save_preset
from makeup_cam.runner import CameraRunner

# ---- fakes ------------------------------------------------------------------


@dataclass
class FakeResult:
    """Shaped like pipeline.ProcessResult."""

    output: np.ndarray
    preview: np.ndarray
    original_preview: np.ndarray
    face_found: bool = True
    timings_ms: dict[str, float] = field(default_factory=dict)


@dataclass
class FakeStatus:
    """Shaped like runner.RunnerStatus."""

    fps: float = 30.0
    camera_ok: bool = True
    camera_error: str | None = None
    virtual_camera: str | None = "OBS Virtual Camera"
    virtual_camera_error: str | None = None
    face_found: bool = True
    processing_ms: float = 10.0
    processing_error: str | None = None
    background_message: str | None = None


@dataclass
class FakeCamera:
    """Shaped like camera.CameraInfo."""

    index: int
    name: str
    width: int
    height: int


class FakeRunner:
    def __init__(self, on_frame: Callable[[FakeResult], None], on_status: Callable[[FakeStatus], None]) -> None:
        self.on_frame = on_frame
        self.on_status = on_status
        self.started = False
        self.stopped = False
        self.stop_delay_s = 0.0
        self.pushed: list[MakeupSettings] = []
        self.bypass: list[bool] = []
        self.cameras: list[tuple[int, int | None, int | None]] = []
        self.camera_names: list[str | None] = []
        self.virtual_camera: list[bool] = []
        self.has_background = False
        self.capture_answers: list[Callable[[plates.PlateResult], None]] = []
        self.clears = 0

    def capture_background(self, on_done: Callable[[plates.PlateResult], None]) -> None:
        self.capture_answers.append(on_done)

    def clear_background(self) -> plates.PlateResult:
        self.clears += 1
        self.has_background = False
        return plates.PlateResult(True, plates.CLEARED_MESSAGE)

    def start(self) -> None:
        self.started = True

    def stop(self, timeout: float = 2.0) -> None:
        time.sleep(self.stop_delay_s)
        self.stopped = True

    def set_settings(self, settings: MakeupSettings) -> None:
        self.pushed.append(settings)

    def glide_to_settings(self, settings: MakeupSettings) -> None:
        self.pushed.append(settings)

    def set_bypass(self, bypass: bool) -> None:
        self.bypass.append(bypass)

    def set_camera(
        self, index: int, width: int | None = None, height: int | None = None, name: str | None = None
    ) -> None:
        self.cameras.append((index, width, height))
        self.camera_names.append(name)

    def set_virtual_camera(self, enabled: bool) -> None:
        self.virtual_camera.append(enabled)


class FakeDialogs(gui.Dialogs):
    """Answers every question from attributes and records what was shown."""

    def __init__(self) -> None:
        super().__init__()
        self.confirm_answer = True
        self.text_answer: str | None = None
        self.questions: list[str] = []
        self.errors: list[str] = []
        self.photos: gui.PhotoRequest | None = None
        self.reports: list[gui.PhotoReport] = []
        self.more_names: list[str | None] = []

    def ask_photos(self) -> gui.PhotoRequest | None:
        return self.photos

    def show_photo_report(self, report: gui.PhotoReport) -> None:
        self.reports.append(report)

    def confirm(self, title: str, text: str) -> bool:
        self.questions.append(text)
        return self.confirm_answer

    def ask_text(self, title: str, label: str, text: str = "") -> str | None:
        return self.more_names.pop(0) if self.more_names else self.text_answer

    def show_error(self, title: str, text: str) -> None:
        self.errors.append(text)


@dataclass
class Harness:
    window: MainWindow
    runner: FakeRunner
    dialogs: FakeDialogs


# ---- fixtures and helpers ------------------------------------------------------


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def isolated_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point user data and built-in presets at empty temp dirs; returns the built-in dir."""
    builtin = tmp_path / "builtin"
    builtin.mkdir()
    monkeypatch.setattr(config, "BUILTIN_PRESETS_DIR", builtin)
    monkeypatch.setenv("AI_MAKEUP_CAM_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(camera, "_device_names", dict)  # the cameras of the PC running the tests
    return builtin


@pytest.fixture
def make_window(qapp: QApplication) -> Iterator[Callable[..., Harness]]:
    windows: list[MainWindow] = []

    def make(
        settings: MakeupSettings | None = None,
        preset_name: str | None = None,
        camera_lister: Callable[[Collection[int]], list[FakeCamera]] | None = None,
        options: GuiOptions | None = None,
        photo_analyzer: Callable | None = None,
    ) -> Harness:
        runners: list[FakeRunner] = []

        def factory(on_frame: Callable, on_status: Callable) -> FakeRunner:
            runners.append(FakeRunner(on_frame, on_status))
            return runners[-1]

        dialogs = FakeDialogs()
        window = MainWindow(
            factory,
            settings=settings or MakeupSettings(),
            preset_name=preset_name,
            camera_lister=camera_lister,
            options=options or GuiOptions(),
            dialogs=dialogs,
            photo_analyzer=photo_analyzer,
        )
        windows.append(window)
        return Harness(window, runners[0], dialogs)

    yield make
    for window in windows:
        window.close()
        window.deleteLater()
    QApplication.processEvents()


def wait_until(condition: Callable[[], bool], timeout_s: float = 3.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("condition not met in time")
        QApplication.processEvents()
        time.sleep(0.005)


def solid(bgr: tuple[int, int, int], width: int = 64, height: int = 36) -> np.ndarray:
    return np.full((height, width, 3), bgr, np.uint8)


def result(
    preview_bgr: tuple[int, int, int], original_bgr: tuple[int, int, int], face_found: bool = True
) -> FakeResult:
    preview = solid(preview_bgr)
    return FakeResult(output=preview, preview=preview, original_preview=solid(original_bgr), face_found=face_found)


def shown_color(window: MainWindow) -> tuple[int, int, int]:
    image = window.preview.image
    assert image is not None
    color = image.pixelColor(0, 0)
    return (color.blue(), color.green(), color.red())


# ---- conversion -----------------------------------------------------------------


def test_bgr_to_qimage_maps_channels_and_copies_pixels() -> None:
    frame = solid((255, 0, 0), width=5, height=3)
    image = bgr_to_qimage(frame)
    frame[:] = 0
    assert (image.width(), image.height()) == (5, 3)
    assert image.pixelColor(4, 2) == QColor(0, 0, 255)


def test_bgr_to_qimage_handles_non_contiguous_views() -> None:
    frame = np.zeros((4, 8, 3), np.uint8)
    frame[:, ::2] = (0, 255, 0)
    image = bgr_to_qimage(frame[:, ::2])
    assert image.width() == 4
    assert image.pixelColor(3, 3) == QColor(0, 255, 0)


def test_bgr_to_qimage_rejects_float_frames() -> None:
    with pytest.raises(ValueError):
        bgr_to_qimage(np.zeros((2, 2, 3), np.float32))


# ---- generated controls -----------------------------------------------------------


def nav_rows(window: MainWindow) -> list[tuple[str, str | None]]:
    """Each row of the settings list as (shown text, section); group titles have no section."""
    nav = window._nav
    return [
        (nav.item(row).text().strip("　"), nav.item(row).data(Qt.ItemDataRole.UserRole)) for row in range(nav.count())
    ]


def test_window_has_title_and_one_page_per_section(make_window: Callable[..., Harness]) -> None:
    window = make_window().window
    listed = [section for _, section in nav_rows(window) if section]
    assert window.windowTitle() == APP_TITLE
    assert sorted(listed) == sorted(SECTIONS)
    assert all((SECTION_LABELS[section], section) in nav_rows(window) for section in SECTIONS)
    assert window._stack.count() == len(SECTIONS)
    assert isinstance(window.section_page("lips").editor("opacity"), FloatEditor)
    assert isinstance(window.section_page("general").editor("segmentation_interval"), IntEditor)
    assert isinstance(window.section_page("lips").editor("color"), ColorEditor)
    assert isinstance(window.section_page("lips").editor("finish"), ChoiceEditor)


def test_widgets_show_the_initial_settings(make_window: Callable[..., Harness]) -> None:
    settings = MakeupSettings()
    settings.lips.opacity = 0.83
    settings.blush.shape = "lifted"
    window = make_window(settings).window
    opacity = window.section_page("lips").editor("opacity")
    assert opacity.spin_box.value() == pytest.approx(0.83)
    assert opacity.slider.value() == 83
    assert window.section_page("blush").editor("shape").value() == "lifted"


def test_slider_updates_settings_spin_box_and_runner(make_window: Callable[..., Harness]) -> None:
    harness = make_window()
    editor = harness.window.section_page("lips").editor("opacity")

    editor.slider.setValue(70)

    assert editor.spin_box.value() == pytest.approx(0.70)
    assert harness.window.settings().lips.opacity == pytest.approx(0.70)
    wait_until(lambda: bool(harness.runner.pushed))
    assert harness.runner.pushed[-1].lips.opacity == pytest.approx(0.70)


def test_spin_box_updates_slider_and_signed_ranges(make_window: Callable[..., Harness]) -> None:
    harness = make_window()
    editor = harness.window.section_page("skin").editor("brightness")  # range -0.3 .. 0.3

    editor.spin_box.setValue(-0.12)

    assert editor.slider.value() == 18
    assert harness.window.settings().skin.brightness == pytest.approx(-0.12)


def test_int_field_stores_an_int(make_window: Callable[..., Harness]) -> None:
    harness = make_window()
    harness.window.section_page("general").editor("segmentation_interval").spin_box.setValue(4)
    value = harness.window.settings().general.segmentation_interval
    assert value == 4 and isinstance(value, int)


def test_rapid_changes_are_throttled_but_the_last_value_arrives(make_window: Callable[..., Harness]) -> None:
    harness = make_window()
    slider = harness.window.section_page("lips").editor("opacity").slider
    for position in range(10, 60, 5):
        slider.setValue(position)
    wait_until(lambda: bool(harness.runner.pushed) and harness.runner.pushed[-1].lips.opacity == pytest.approx(0.55))
    assert len(harness.runner.pushed) <= 2


def test_color_hex_entry_updates_settings_and_rejects_bad_input(make_window: Callable[..., Harness]) -> None:
    harness = make_window()
    editor = harness.window.section_page("lips").editor("color")

    editor.hex_edit.setText("aa3344")
    editor.hex_edit.editingFinished.emit()
    assert harness.window.settings().lips.color == "#AA3344"
    assert editor.hex_edit.text() == "#AA3344"

    editor.hex_edit.setText("#12")
    editor.hex_edit.editingFinished.emit()
    assert harness.window.settings().lips.color == "#AA3344"
    assert editor.hex_edit.text() == "#AA3344"


def test_choice_updates_settings(make_window: Callable[..., Harness]) -> None:
    harness = make_window()
    combo = harness.window.section_page("lips").editor("finish").combo_box
    combo.setCurrentIndex(combo.findData("gloss"))
    assert harness.window.settings().lips.finish == "gloss"


def test_reset_button_restores_the_default(make_window: Callable[..., Harness]) -> None:
    harness = make_window()
    page = harness.window.section_page("eyeliner")
    default = MakeupSettings().eyeliner.wing
    assert not page.reset_button("wing").isEnabled()

    page.editor("wing").slider.setValue(90)
    assert page.reset_button("wing").isEnabled()
    page.reset_button("wing").click()

    assert harness.window.settings().eyeliner.wing == default
    assert page.editor("wing").value() == default
    assert not page.reset_button("wing").isEnabled()


def test_disabling_a_section_greys_out_its_controls(make_window: Callable[..., Harness]) -> None:
    harness = make_window()
    page = harness.window.section_page("blush")

    page.editor("enabled").check_box.setChecked(False)

    assert harness.window.settings().blush.enabled is False
    assert not page.editor("opacity").isEnabled()
    page.editor("enabled").check_box.setChecked(True)
    assert page.editor("opacity").isEnabled()


def test_wheel_over_an_unfocused_slider_does_not_change_it(make_window: Callable[..., Harness]) -> None:
    harness = make_window()
    slider = harness.window.section_page("lips").editor("opacity").slider
    before = slider.value()
    wheel = QWheelEvent(
        QPointF(5, 5),
        QPointF(5, 5),
        QPoint(0, 0),
        QPoint(0, 120),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    QApplication.sendEvent(slider, wheel)
    assert slider.value() == before


# ---- presets ------------------------------------------------------------------------


def test_preset_save_as_load_and_delete_roundtrip(make_window: Callable[..., Harness]) -> None:
    harness = make_window()
    window, dialogs = harness.window, harness.dialogs
    opacity = window.section_page("lips").editor("opacity")

    opacity.slider.setValue(80)
    dialogs.text_answer = "面接/用"
    window._save_as_button.click()
    assert window.preset_name == "面接用"
    assert load_preset("面接用").lips.opacity == pytest.approx(0.80)
    assert window._preset_combo.currentText() == "面接用"
    assert not window.is_modified()
    assert window._delete_button.isEnabled()

    opacity.slider.setValue(30)
    assert window.is_modified() and window._save_button.isEnabled()
    window._save_button.click()
    assert load_preset("面接用").lips.opacity == pytest.approx(0.30)
    assert not window.is_modified()

    window._defaults_button.click()
    assert window.settings() == MakeupSettings()
    assert window.preset_name is None and window._preset_combo.currentIndex() == -1

    pushes_before = len(harness.runner.pushed)
    combo = window._preset_combo
    combo.activated.emit(combo.findData("面接用"))
    assert window.settings().lips.opacity == pytest.approx(0.30)
    assert opacity.slider.value() == 30
    assert len(harness.runner.pushed) == pushes_before + 1
    assert harness.runner.pushed[-1].lips.opacity == pytest.approx(0.30)

    window._delete_button.click()
    assert find_preset("面接用") is None
    assert window.preset_name is None
    assert combo.findData("面接用") == -1
    assert window.settings().lips.opacity == pytest.approx(0.30)  # current settings are kept


def test_builtin_preset_is_marked_and_cannot_be_overwritten_or_deleted(
    make_window: Callable[..., Harness], isolated_dirs: Path
) -> None:
    (isolated_dirs / "natural.json").write_text(MakeupSettings().to_json(), encoding="utf-8")
    window = make_window(preset_name="natural").window

    assert window._preset_combo.currentText() == "natural（標準）"
    window.section_page("lips").editor("opacity").slider.setValue(90)
    assert window.is_modified()
    assert not window._save_button.isEnabled()
    assert not window._delete_button.isEnabled()


def test_saving_under_a_builtin_name_asks_and_overrides_it(
    make_window: Callable[..., Harness], isolated_dirs: Path
) -> None:
    (isolated_dirs / "natural.json").write_text(MakeupSettings().to_json(), encoding="utf-8")
    harness = make_window(preset_name="natural")
    harness.window.section_page("lips").editor("opacity").slider.setValue(90)
    harness.dialogs.text_answer = "natural"

    harness.dialogs.confirm_answer = False
    harness.window._save_as_button.click()
    assert find_preset("natural").builtin  # declined: still the built-in one

    harness.dialogs.confirm_answer = True
    harness.window._save_as_button.click()
    assert "標準プリセット" in harness.dialogs.questions[-1]
    assert find_preset("natural").builtin is False
    assert harness.window._delete_button.isEnabled()


def test_choosing_a_preset_with_unsaved_changes_asks_first(make_window: Callable[..., Harness]) -> None:
    save_preset("A", MakeupSettings())
    harness = make_window()
    harness.window.section_page("lips").editor("opacity").slider.setValue(95)
    harness.dialogs.confirm_answer = False

    combo = harness.window._preset_combo
    combo.activated.emit(combo.findData("A"))

    assert harness.dialogs.questions
    assert harness.window.settings().lips.opacity == pytest.approx(0.95)
    assert harness.window.preset_name is None
    assert combo.currentIndex() == -1


def test_broken_preset_shows_an_error_and_keeps_settings(make_window: Callable[..., Harness]) -> None:
    save_preset("broken", MakeupSettings())
    find_preset("broken").path.write_text("{", encoding="utf-8")
    harness = make_window()

    assert harness.window.apply_preset("broken") is False
    assert "broken" in harness.dialogs.errors[-1]
    assert harness.window.settings() == MakeupSettings()


# ---- preview ---------------------------------------------------------------------------


def test_frame_updates_preview_and_face_hint(make_window: Callable[..., Harness]) -> None:
    harness = make_window()
    assert harness.window.preview.notice  # "starting" until the first frame

    harness.runner.on_frame(result((10, 20, 30), (0, 0, 0), face_found=False))

    assert shown_color(harness.window) == (10, 20, 30)
    assert harness.window.preview.hint == FACE_HINT
    assert harness.window.preview.notice == ""
    harness.runner.on_frame(result((10, 20, 30), (0, 0, 0), face_found=True))
    assert harness.window.preview.hint == ""


def test_compare_button_shows_the_original_while_pressed(make_window: Callable[..., Harness]) -> None:
    harness = make_window()
    harness.runner.on_frame(result((10, 20, 30), (200, 150, 100)))
    button = harness.window._compare_button

    button.pressed.emit()
    assert shown_color(harness.window) == (200, 150, 100)
    assert harness.window.preview.badge
    harness.runner.on_frame(result((11, 21, 31), (201, 151, 101)))
    assert shown_color(harness.window) == (201, 151, 101)

    button.released.emit()
    assert shown_color(harness.window) == (11, 21, 31)
    assert harness.window.preview.badge == ""


def test_space_key_holds_compare(make_window: Callable[..., Harness]) -> None:
    harness = make_window()
    harness.runner.on_frame(result((10, 20, 30), (200, 150, 100)))

    QTest.keyPress(harness.window, Qt.Key.Key_Space)
    assert shown_color(harness.window) == (200, 150, 100)
    QTest.keyRelease(harness.window, Qt.Key.Key_Space)
    assert shown_color(harness.window) == (10, 20, 30)


def test_frames_from_a_worker_thread_keep_only_the_latest(make_window: Callable[..., Harness]) -> None:
    harness = make_window()
    shown: list[int] = []
    harness.window.preview.set_image = lambda image: shown.append(image.pixelColor(0, 0).red())

    def produce() -> None:
        for value in range(1, 41):
            harness.runner.on_frame(result((0, 0, value), (0, 0, 0)))

    worker = threading.Thread(target=produce)
    worker.start()
    worker.join()
    wait_until(lambda: bool(shown) and shown[-1] == 40)
    assert len(shown) < 40


def test_bypass_and_virtual_camera_toggles_reach_the_runner(make_window: Callable[..., Harness]) -> None:
    harness = make_window()
    harness.runner.on_frame(result((10, 20, 30), (0, 0, 0)))

    harness.window._bypass_button.click()
    assert harness.runner.bypass == [True]
    assert harness.window.preview.badge == "メイクOFF中"

    harness.window._vcam_check.setChecked(False)
    assert harness.runner.virtual_camera == [False]
    assert harness.window._vcam_status.text() == "仮想カメラ OFF"


# ---- status bar -------------------------------------------------------------------------


def test_status_bar_shows_rates_and_detection(make_window: Callable[..., Harness]) -> None:
    harness = make_window()
    harness.runner.on_status(FakeStatus(fps=29.84, processing_ms=12.345, face_found=False))
    window = harness.window
    assert window._fps_status.text() == "FPS 29.8"
    assert window._time_status.text() == "処理 12.3 ms"
    assert "NG" in window._face_status.text()
    assert "OBS Virtual Camera" in window._vcam_status.text()


def test_virtual_camera_error_is_red_with_full_guidance(make_window: Callable[..., Harness]) -> None:
    harness = make_window()
    guidance = "OBS Studio をインストールしてから、このアプリを再起動してください。"
    harness.runner.on_status(FakeStatus(virtual_camera=None, virtual_camera_error=guidance))

    label = harness.window._vcam_status
    assert "使用できません" in label.text() and gui._ERROR_COLOR in label.text()
    assert label.toolTip() == guidance
    label.linkActivated.emit("details")
    assert harness.dialogs.errors == [guidance]


def test_camera_error_is_shown_on_the_preview_and_status_bar(make_window: Callable[..., Harness]) -> None:
    harness = make_window()
    harness.runner.on_frame(result((10, 20, 30), (0, 0, 0)))
    harness.runner.on_status(FakeStatus(camera_ok=False, camera_error="他のアプリがカメラを使用中です"))

    assert "他のアプリがカメラを使用中です" in harness.window.preview.notice
    assert harness.window._camera_status.isVisibleTo(harness.window)
    harness.runner.on_status(FakeStatus())
    assert harness.window.preview.notice == ""
    assert not harness.window._camera_status.isVisibleTo(harness.window)


# ---- camera selection ------------------------------------------------------------------


def test_cameras_are_listed_in_the_background_and_selectable(make_window: Callable[..., Harness]) -> None:
    asked: list[Collection[int]] = []

    def lister(in_use: Collection[int]) -> list[FakeCamera]:
        asked.append(in_use)
        return [FakeCamera(0, "内蔵カメラ", 1280, 720), FakeCamera(2, "USB カメラ", 1920, 1080)]

    harness = make_window(camera_lister=lister)
    harness.window.start()
    combo = harness.window._camera_combo
    wait_until(lambda: combo.count() == 2)
    assert harness.runner.started
    assert list(asked[0]) == [0]
    assert combo.itemText(1) == "USB カメラ"

    combo.setCurrentIndex(1)
    combo.activated.emit(1)
    resolution = harness.window._resolution_combo
    resolution.setCurrentIndex(resolution.findText("640×480"))
    resolution.activated.emit(resolution.currentIndex())
    resolution.activated.emit(resolution.currentIndex())  # unchanged: no reconnect

    assert harness.runner.cameras == [(2, 1280, 720), (2, 640, 480)]


def test_selected_camera_stays_listed_when_the_probe_misses_it(make_window: Callable[..., Harness]) -> None:
    harness = make_window(camera_lister=lambda in_use: [])
    harness.window.start()
    wait_until(lambda: harness.window._rescan_button.isEnabled())
    assert harness.window._camera_combo.itemData(0) == 0


def test_rescan_moves_a_failing_virtual_camera_selection_to_a_real_camera(make_window: Callable[..., Harness]) -> None:
    found: list[FakeCamera] = []
    harness = make_window(camera_lister=lambda in_use: list(found))
    window = harness.window
    window.start()
    wait_until(lambda: window._rescan_button.isEnabled())
    harness.runner.on_status(FakeStatus(camera_ok=False, camera_error="「OBS Virtual Camera」は仮想カメラなので…"))
    assert "仮想カメラ" in window.preview.notice
    assert harness.runner.cameras == []  # nothing real to move to yet

    found.append(FakeCamera(1, "HD Pro Webcam C920", 1280, 720))  # she plugged in the webcam and pressed the button
    window._rescan_button.click()
    wait_until(lambda: window._rescan_button.isEnabled())

    assert window._camera_combo.currentData() == 1
    assert window._camera_combo.currentText() == "HD Pro Webcam C920"
    assert harness.runner.cameras == [(1, 1280, 720)]


def test_a_camera_that_is_still_starting_is_not_replaced(make_window: Callable[..., Harness]) -> None:
    harness = make_window(camera_lister=lambda in_use: [FakeCamera(1, "HD Pro Webcam C920", 1280, 720)])
    harness.window.start()
    wait_until(lambda: harness.window._rescan_button.isEnabled())

    assert harness.window._camera_combo.currentData() == 0
    assert harness.runner.cameras == []


def test_real_camera_name_is_remembered_with_its_index(make_window: Callable[..., Harness]) -> None:
    harness = make_window(camera_lister=lambda in_use: [FakeCamera(0, "HD Pro Webcam C920", 1280, 720)])
    harness.window.start()
    wait_until(lambda: harness.window._rescan_button.isEnabled())

    harness.window.close()

    state = GuiState()
    assert (state.camera_index(), state.camera_name()) == (0, "HD Pro Webcam C920")


def test_a_selection_known_not_to_be_a_real_camera_does_not_replace_the_remembered_one(
    make_window: Callable[..., Harness], qapp: QApplication
) -> None:
    state = GuiState()
    state.save(
        camera_index=1,
        camera_name="HD Pro Webcam C920",
        resolution=(1280, 720),
        preset=None,
        geometry=QByteArray(),
        splitter=QByteArray(),
    )
    # The webcam is unplugged today: index 0 is OBS (hidden), another camera was found at index 3.
    harness = make_window(camera_lister=lambda in_use: [FakeCamera(3, "Other", 640, 480)])
    harness.window.start()
    wait_until(lambda: harness.window._rescan_button.isEnabled())

    harness.window.close()

    state = GuiState()
    assert (state.camera_index(), state.camera_name()) == (1, "HD Pro Webcam C920")


def test_choosing_a_camera_tells_the_runner_its_name(make_window: Callable[..., Harness]) -> None:
    cameras = [FakeCamera(0, "内蔵カメラ", 1280, 720), FakeCamera(2, "USB", 640, 480)]
    harness = make_window(camera_lister=lambda in_use: list(cameras))
    window, combo = harness.window, harness.window._camera_combo
    window.start()
    wait_until(lambda: combo.count() == 2)

    combo.setCurrentIndex(1)
    combo.activated.emit(1)

    assert harness.runner.cameras == [(2, 1280, 720)]
    assert harness.runner.camera_names == ["USB"]


def test_a_placeholder_is_not_passed_on_as_a_device_name(make_window: Callable[..., Harness]) -> None:
    cameras = [FakeCamera(0, "カメラ 0", 640, 480), FakeCamera(1, "カメラ 1", 640, 480)]
    harness = make_window(camera_lister=lambda in_use: list(cameras))
    window, combo = harness.window, harness.window._camera_combo
    window.start()
    wait_until(lambda: combo.count() == 2)

    combo.setCurrentIndex(1)
    combo.activated.emit(1)

    assert harness.runner.cameras == [(1, 1280, 720)]
    assert harness.runner.camera_names == [None]


def test_the_probe_is_told_what_is_in_use_by_name_once_it_is_known(make_window: Callable[..., Harness]) -> None:
    asked: list[list[object]] = []

    def lister(in_use: Collection[object]) -> list[FakeCamera]:
        asked.append(list(in_use))
        return [FakeCamera(1, "USB Camera", 1280, 720)]

    harness = make_window(camera_lister=lister, options=GuiOptions(camera_index=1))
    window = harness.window
    window.start()
    wait_until(lambda: window._rescan_button.isEnabled())
    assert asked == [[1]]  # no name yet: only the index to go on

    window._rescan_button.click()
    wait_until(lambda: window._rescan_button.isEnabled())
    assert asked == [[1], ["USB Camera"]]


def test_a_failing_camera_without_a_name_is_really_probed(make_window: Callable[..., Harness]) -> None:
    asked: list[list[object]] = []

    def lister(in_use: Collection[object]) -> list[FakeCamera]:
        asked.append(list(in_use))
        return []

    harness = make_window(camera_lister=lister, options=GuiOptions(camera_index=1))
    harness.runner.on_status(FakeStatus(camera_ok=False, camera_error="カメラ 1 を開けませんでした"))
    harness.window.start()
    wait_until(lambda: harness.window._rescan_button.isEnabled())

    assert asked == [[]]  # nothing is open, so there is nothing to protect from being probed


def test_rescan_follows_the_selected_camera_to_its_new_number(make_window: Callable[..., Harness]) -> None:
    found = [FakeCamera(0, "Built-in", 1280, 720), FakeCamera(1, "USB Camera", 1280, 720)]
    harness = make_window(camera_lister=lambda in_use: list(found), options=GuiOptions(camera_index=1))
    window, runner, combo = harness.window, harness.runner, harness.window._camera_combo
    window.start()
    wait_until(lambda: combo.count() == 2)
    assert combo.currentText() == "USB Camera"

    found[:] = [FakeCamera(0, "USB Camera", 0, 0), FakeCamera(1, "Other Camera", 640, 480)]  # Built-in went away
    window._rescan_button.click()
    wait_until(lambda: combo.itemText(1) == "Other Camera")

    assert (combo.currentData(), combo.currentText()) == (0, "USB Camera")
    assert combo.count() == 2  # no made-up entry for the old number
    assert runner.cameras == []  # the same camera: no reconnect

    combo.activated.emit(combo.currentIndex())
    assert runner.cameras == []  # choosing it again is still no reason to reconnect

    resolution = window._resolution_combo
    resolution.setCurrentIndex(resolution.findText("640×480"))
    resolution.activated.emit(resolution.currentIndex())
    assert runner.cameras == [(0, 640, 480)]
    assert runner.camera_names == ["USB Camera"]


def test_a_live_camera_the_probe_missed_stays_selected_under_its_own_name(make_window: Callable[..., Harness]) -> None:
    options = GuiOptions(camera_index=1, camera_name="USB Camera")
    harness = make_window(camera_lister=lambda in_use: [FakeCamera(1, "Other Camera", 640, 480)], options=options)
    window, combo = harness.window, harness.window._camera_combo
    assert combo.currentText() == "USB Camera"  # known from the start
    window.start()
    wait_until(lambda: combo.count() == 2)

    assert combo.currentText() == "USB Camera"  # not Other Camera, which has the same number now
    assert harness.runner.cameras == []
    window.close()

    assert GuiState().camera_name() is None  # an unplugged device replaces nothing


def test_a_failing_camera_that_is_gone_moves_to_a_real_one_on_rescan(make_window: Callable[..., Harness]) -> None:
    found = [FakeCamera(1, "Other Camera", 640, 480)]
    options = GuiOptions(camera_index=1, camera_name="USB Camera")
    harness = make_window(camera_lister=lambda in_use: list(found), options=options)
    harness.runner.on_status(FakeStatus(camera_ok=False, camera_error="「USB Camera」が見つかりません"))

    harness.window._rescan_button.click()
    wait_until(lambda: harness.window._rescan_button.isEnabled())

    assert harness.window._camera_combo.currentText() == "Other Camera"
    assert harness.runner.cameras == [(1, 1280, 720)]
    assert harness.runner.camera_names == ["Other Camera"]


def test_two_identical_cameras_keep_their_order_on_rescan(make_window: Callable[..., Harness]) -> None:
    cameras = [FakeCamera(0, "USB Camera", 640, 480), FakeCamera(1, "USB Camera", 640, 480)]
    asked: list[list[object]] = []

    def lister(in_use: Collection[object]) -> list[FakeCamera]:
        asked.append(list(in_use))
        return list(cameras)

    harness = make_window(camera_lister=lister, options=GuiOptions(camera_index=1))
    window, combo = harness.window, harness.window._camera_combo
    window.start()
    wait_until(lambda: combo.count() == 2)
    assert combo.currentData() == 1

    window._rescan_button.click()
    wait_until(lambda: window._rescan_button.isEnabled())

    assert combo.currentData() == 1
    assert asked == [[1], [1]]  # the name cannot tell them apart: only the number says which one is live


class NoDevice:
    """cv2.VideoCapture with nothing there."""

    def __init__(self, index: int, api: int) -> None:
        pass

    def isOpened(self) -> bool:
        return False

    def release(self) -> None:
        pass


def save_camera(index: int, name: str) -> None:
    state = GuiState()
    state.save(
        camera_index=index,
        camera_name=name,
        resolution=(1280, 720),
        preset=None,
        geometry=QByteArray(),
        splitter=QByteArray(),
    )


def test_the_saved_camera_name_survives_a_run_with_the_webcam_unplugged(
    make_window: Callable[..., Harness], qapp: QApplication
) -> None:
    # No device at all is listed: the selected number is listed anyway in case it is the one in use.
    save_camera(1, "HD Pro Webcam C920")
    options = options_from_args(None, GuiState())
    assert (options.camera_index, options.camera_name) == (1, None)
    harness = make_window(
        camera_lister=lambda in_use: camera.list_cameras(in_use=in_use, capture_factory=NoDevice), options=options
    )
    window = harness.window
    window.start()
    wait_until(lambda: window._rescan_button.isEnabled())

    assert window._camera_combo.currentText() == "カメラ 1"
    assert "見つかりませんでした" in window._camera_combo.toolTip()
    window.close()

    state = GuiState()
    assert (state.camera_index(), state.camera_name()) == (1, "HD Pro Webcam C920")


def test_a_bare_number_is_still_remembered_where_there_are_never_names(
    make_window: Callable[..., Harness], qapp: QApplication
) -> None:
    cameras = [FakeCamera(1, "カメラ 1", 640, 480)]
    harness = make_window(camera_lister=lambda in_use: list(cameras), options=GuiOptions(camera_index=1))
    harness.window.start()
    wait_until(lambda: harness.window._rescan_button.isEnabled())
    harness.window.close()

    state = GuiState()
    assert (state.camera_index(), state.camera_name()) == (1, None)


def test_the_not_found_hint_goes_away_when_the_listed_camera_turns_out_to_work(
    make_window: Callable[..., Harness],
) -> None:
    # Where there are no names, a camera in use is listed without being probed: it may or may not exist.
    harness = make_window(camera_lister=lambda in_use: [FakeCamera(0, "カメラ 0", 0, 0)])
    window, combo = harness.window, harness.window._camera_combo
    window.start()
    wait_until(lambda: window._rescan_button.isEnabled())
    assert "見つかりませんでした" in combo.toolTip()

    harness.runner.on_status(FakeStatus())

    assert combo.toolTip() == ""


def test_a_camera_that_was_probed_is_never_reported_as_not_found(make_window: Callable[..., Harness]) -> None:
    harness = make_window(camera_lister=lambda in_use: [FakeCamera(0, "カメラ 0", 640, 480)])
    harness.window.start()
    wait_until(lambda: harness.window._rescan_button.isEnabled())
    assert harness.window._camera_combo.toolTip() == ""


def test_options_look_up_the_remembered_camera_by_name(qapp: QApplication, monkeypatch: pytest.MonkeyPatch) -> None:
    state = GuiState()
    state.save(
        camera_index=1,
        camera_name="HD Pro Webcam C920",
        resolution=(1280, 720),
        preset=None,
        geometry=QByteArray(),
        splitter=QByteArray(),
    )
    # Renumbered since the last run; index 1 is the virtual camera now.
    monkeypatch.setattr(camera, "_device_names", lambda: {0: "HD Pro Webcam C920", 1: "OBS Virtual Camera"})

    options = options_from_args(None, state)
    assert (options.camera_index, options.camera_name) == (0, "HD Pro Webcam C920")

    # --camera is taken literally (CameraSource refuses a virtual camera).
    explicit = options_from_args(argparse.Namespace(camera=1), state)
    assert (explicit.camera_index, explicit.camera_name) == (1, None)


def test_remembered_virtual_camera_index_is_not_used(qapp: QApplication, monkeypatch: pytest.MonkeyPatch) -> None:
    state = GuiState()
    state.save(camera_index=0, resolution=(1280, 720), preset=None, geometry=QByteArray(), splitter=QByteArray())
    monkeypatch.setattr(camera, "_device_names", lambda: {0: "OBS Virtual Camera", 1: "USB"})

    assert options_from_args(None, state).camera_index == 1


# ---- a renumbered camera, with the real runner and camera source -------------------------


DEVICE_VALUES = {"Built-in": 50, "USB Camera": 100, "Other Camera": 150, "OBS Virtual Camera": 250}


class DeviceWorld:
    """Windows as the app sees it: every capture is bound to the device that held the index when it was opened."""

    def __init__(self, names: dict[int, str]) -> None:
        self.names = dict(names)
        self.opens: list[tuple[str, str]] = []  # (device, name of the thread that opened it)
        self.stopped: list[str] = []  # devices stopped by the release of another capture, as on DirectShow
        self.sent_value: int | None = None
        self.device = "Fake Virtual Camera"

    def capture(self, index: int, api: int) -> DeviceCapture:
        return DeviceCapture(self, self.names.get(index))

    def send(self, frame: np.ndarray) -> None:
        self.sent_value = int(frame[0, 0, 0])

    def close(self) -> None:
        pass

    def by_probe(self) -> list[str]:
        return [device for device, thread in self.opens if thread == "camera-probe"]

    def by_runner(self) -> list[str]:
        return [device for device, thread in self.opens if thread != "camera-probe"]


class DeviceCapture:
    def __init__(self, world: DeviceWorld, device: str | None) -> None:
        self.world = world
        self.device = device
        if device is not None:
            world.opens.append((device, threading.current_thread().name))

    def isOpened(self) -> bool:
        return self.device is not None

    def read(self) -> tuple[bool, np.ndarray | None]:
        time.sleep(0.005)
        return True, np.full((48, 64, 3), DEVICE_VALUES[str(self.device)], np.uint8)

    def set(self, _prop: int, _value: float) -> bool:
        return True

    def get(self, _prop: int) -> float:
        return 30.0

    def release(self) -> None:
        if threading.current_thread().name == "camera-probe":
            self.world.stopped.append(str(self.device))


class PassThroughPipeline:
    bypass = False
    has_background = False

    def prepare(self, _frame: np.ndarray, _timestamp_ms: int) -> None:
        pass

    def process(self, frame: np.ndarray, _timestamp_ms: int) -> FakeResult:
        return FakeResult(output=frame, preview=frame, original_preview=frame)

    def set_settings(self, _settings: MakeupSettings) -> None:
        pass

    def glide_to_settings(self, _settings: MakeupSettings) -> None:
        pass


@pytest.fixture
def real_window(qapp: QApplication, monkeypatch: pytest.MonkeyPatch) -> Iterator[Callable[..., tuple]]:
    windows: list[MainWindow] = []

    def make(names: dict[int, str], index: int, name: str) -> tuple[MainWindow, DeviceWorld]:
        world = DeviceWorld(names)
        monkeypatch.setattr(camera, "host_os", lambda: "windows")
        monkeypatch.setattr(camera, "_device_names", lambda: dict(world.names))
        monkeypatch.setattr(camera, "_msmf_device_names", dict)

        def factory(on_frame: Callable, on_status: Callable) -> CameraRunner:
            return CameraRunner(
                PassThroughPipeline(),
                camera_index=index,
                camera_name=name,
                camera_factory=partial(camera.CameraSource, capture_factory=world.capture, reopen_interval_s=0.05),
                vcam_factory=lambda **_: world,
                on_frame=on_frame,
                on_status=on_status,
                retry_interval_s=0.05,
            )

        window = MainWindow(
            factory,
            settings=MakeupSettings(),
            camera_lister=lambda in_use: camera.list_cameras(in_use=in_use, capture_factory=world.capture),
            options=GuiOptions(camera_index=index, camera_name=name),
            dialogs=FakeDialogs(),
        )
        windows.append(window)
        window.start()
        wait_until(lambda: world.sent_value == DEVICE_VALUES[name] and window._rescan_button.isEnabled())
        return window, world

    yield make
    for window in windows:
        window.close()
        window.deleteLater()
    QApplication.processEvents()


@pytest.mark.parametrize(
    "takes_the_old_number",
    ["OBS Virtual Camera", "Other Camera"],  # refused and blanked the call / silently the wrong camera
)
def test_a_new_resolution_keeps_the_camera_that_was_renumbered(real_window: Callable, takes_the_old_number: str) -> None:
    window, world = real_window({0: "Built-in", 1: "USB Camera", 2: takes_the_old_number}, 1, "USB Camera")
    world.names = {0: "USB Camera", 1: takes_the_old_number}  # the built-in camera went away

    resolution = window._resolution_combo
    resolution.setCurrentIndex(resolution.findText("640×480"))
    resolution.activated.emit(resolution.currentIndex())

    wait_until(lambda: len(world.by_runner()) == 2)
    assert world.by_runner() == ["USB Camera", "USB Camera"]
    wait_until(lambda: world.sent_value == DEVICE_VALUES["USB Camera"])
    time.sleep(0.05)
    assert world.sent_value == DEVICE_VALUES["USB Camera"]
    assert window._last_status is None or window._last_status.camera_error is None
    assert window._camera_combo.currentText() == "USB Camera"


def test_a_rescan_never_opens_the_live_camera_under_its_new_number(real_window: Callable) -> None:
    window, world = real_window({0: "Built-in", 1: "USB Camera", 2: "OBS Virtual Camera"}, 1, "USB Camera")
    combo = window._camera_combo
    world.names = {0: "USB Camera", 1: "OBS Virtual Camera"}
    world.opens.clear()
    world.stopped.clear()

    window._rescan_button.click()
    wait_until(lambda: window._rescan_button.isEnabled())

    assert world.by_probe() == []
    assert world.stopped == []  # on DirectShow this would have stopped the live capture
    assert [(combo.itemData(row), combo.itemText(row)) for row in range(combo.count())] == [(0, "USB Camera")]
    assert (combo.currentData(), combo.currentText()) == (0, "USB Camera")
    assert world.by_runner() == []  # no reconnect
    window.close()
    state = GuiState()
    assert (state.camera_index(), state.camera_name()) == (0, "USB Camera")


# ---- a video file instead of a camera ---------------------------------------------------


def test_video_file_is_the_one_camera_entry_and_survives_reselection(make_window: Callable[..., Harness]) -> None:
    options = GuiOptions(camera_file=Path("素材") / "面接の練習.flv")
    harness = make_window(options=options)
    window, runner = harness.window, harness.runner
    window.start()

    combo = window._camera_combo
    assert (combo.count(), combo.currentText()) == (1, "動画ファイル: 面接の練習.flv")
    assert "テスト用" in combo.toolTip()
    assert not window._rescan_button.isEnabled()
    assert runner.started

    combo.activated.emit(0)  # working: choosing it again changes nothing
    assert runner.cameras == []

    runner.on_status(FakeStatus(camera_ok=False, camera_error="動画ファイルを読み込めませんでした"))
    combo.activated.emit(0)  # failing: choosing it again reconnects
    assert runner.cameras == [(0, 1280, 720)]
    assert (combo.count(), combo.currentText()) == (1, "動画ファイル: 面接の練習.flv")

    runner.on_status(FakeStatus())
    resolution = window._resolution_combo
    resolution.setCurrentIndex(resolution.findText("640×480"))
    resolution.activated.emit(resolution.currentIndex())
    assert runner.cameras == [(0, 1280, 720), (0, 640, 480)]
    assert (combo.count(), combo.currentText()) == (1, "動画ファイル: 面接の練習.flv")


def test_video_file_does_not_replace_the_remembered_camera(
    make_window: Callable[..., Harness], qapp: QApplication
) -> None:
    state = GuiState()
    state.save(
        camera_index=2,
        camera_name="USB",
        resolution=(960, 540),
        preset=None,
        geometry=QByteArray(),
        splitter=QByteArray(),
    )
    harness = make_window(options=GuiOptions(camera_file=Path("clip.flv"), width=640, height=480))

    harness.window.close()

    state = GuiState()
    assert (state.camera_index(), state.camera_name(), state.resolution()) == (2, "USB", (960, 540))


def test_options_with_a_video_file_ignore_the_remembered_camera(qapp: QApplication) -> None:
    state = GuiState()
    state.save(camera_index=2, resolution=(960, 540), preset=None, geometry=QByteArray(), splitter=QByteArray())

    options = options_from_args(argparse.Namespace(camera_file=Path("clip.flv")), state)

    assert (options.camera_file, options.camera_index, options.camera_name) == (Path("clip.flv"), 0, None)


# ---- shutdown and restore ----------------------------------------------------------------


def test_close_saves_session_and_stops_runner(make_window: Callable[..., Harness]) -> None:
    harness = make_window()
    harness.window.section_page("hair").editor("gloss").slider.setValue(77)

    harness.window.close()

    assert harness.runner.stopped
    session = load_last_session()
    assert session is not None and session.hair.gloss == pytest.approx(0.77)


def test_close_never_waits_long_for_a_stuck_runner(make_window: Callable[..., Harness]) -> None:
    harness = make_window()
    harness.runner.stop_delay_s = 10.0
    started = time.monotonic()
    assert harness.window.shutdown() is False
    assert time.monotonic() - started < gui.RUNNER_STOP_DEADLINE_S + 0.5


def test_remembered_state_roundtrip(make_window: Callable[..., Harness]) -> None:
    save_preset("A", MakeupSettings())
    harness = make_window(preset_name="A")
    harness.window.close()

    state = GuiState()
    assert state.preset() == "A"
    assert state.camera_index() == 0
    assert state.resolution() == (1280, 720)
    assert state.geometry() is not None


def test_options_merge_command_line_and_remembered_state(qapp: QApplication) -> None:
    state = GuiState()
    state.save(camera_index=2, resolution=(960, 540), preset=None, geometry=QByteArray(), splitter=QByteArray())
    remembered = options_from_args(None, state)
    assert (remembered.camera_index, remembered.resolution, remembered.virtual_camera) == (2, (960, 540), True)

    args = argparse.Namespace(camera=0, width=640, height=480, fps=15, no_vcam=True, preset="面接")
    explicit = options_from_args(args, state)
    assert explicit == GuiOptions(camera_index=0, width=640, height=480, fps=15, virtual_camera=False, preset="面接")


def test_prepare_models_reports_download_errors(
    qapp: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from makeup_cam import models

    calls: list[str] = []

    def failing_download(progress: Callable[[str, int, int], None]) -> None:
        progress("face_landmarker", 1_000_000, 4_000_000)
        calls.append("download")
        raise models.ModelDownloadError("インターネットに接続できません")

    monkeypatch.setattr(models, "models_dir", lambda: tmp_path)
    monkeypatch.setattr(models, "ensure_all_models", failing_download)
    assert gui.prepare_models() == "インターネットに接続できません"
    assert calls == ["download"]

    monkeypatch.setattr(models, "ensure_all_models", lambda progress: calls.append("ok"))
    assert gui.prepare_models() is None
    for spec in models.MODEL_SPECS.values():
        (tmp_path / spec.filename).write_bytes(b"model")
    assert gui.prepare_models() is None
    assert calls == ["download", "ok"]  # present files: no download attempt


def test_ctrl_c_in_the_terminal_closes_the_window(qapp: QApplication) -> None:
    window = QMainWindow()
    window.show()
    previous = signal.getsignal(signal.SIGINT)
    with gui.closed_by_ctrl_c(window):
        signal.raise_signal(signal.SIGINT)
        qapp.processEvents()
        assert not window.isVisible()
    assert signal.getsignal(signal.SIGINT) is previous


# ---- settings list ---------------------------------------------------------------


def test_sections_are_grouped_in_the_list(make_window: Callable[..., Harness]) -> None:
    rows = nav_rows(make_window().window)
    titles = [text for text, section in rows if section is None]
    assert titles == ["肌・立体感", "目元", "口元", "髪"]  # 全体 and 顔の形 are one page each: no title needed
    sections = [section for _, section in rows]
    assert sections.index("skin") == sections.index(None) + 1
    assert [section for section in sections[sections.index("eyeshadow") :][:4]] == [
        "eyeshadow",
        "tearbag",
        "eyeliner",
        "brows",
    ]


def test_a_new_section_gets_a_page_without_touching_the_groups() -> None:
    groups = gui.grouped_sections([*SECTIONS, "hairstyle"])
    assert groups[-1] == (gui.SECTION_OTHER_GROUP, ["hairstyle"])
    assert sorted(section for _, members in groups for section in members) == sorted([*SECTIONS, "hairstyle"])


def test_choosing_in_the_list_shows_that_page(make_window: Callable[..., Harness]) -> None:
    window = make_window().window
    assert window.current_section() == "general"

    window.show_section("tearbag")

    assert window.current_section() == "tearbag"
    assert window._stack.currentWidget() is window.section_page("tearbag")


def test_group_titles_cannot_be_selected(make_window: Callable[..., Harness]) -> None:
    window = make_window().window
    window.show()
    window.show_section("general")
    window._nav.setFocus()
    QTest.keyClick(window._nav, Qt.Key.Key_Down)  # the next row down is the title 肌・立体感

    assert window.current_section() == "skin"
    titles = [row for row, (_, section) in enumerate(nav_rows(window)) if section is None]
    assert all(not (window._nav.item(row).flags() & Qt.ItemFlag.ItemIsSelectable) for row in titles)


def test_the_list_keeps_the_controls_working(make_window: Callable[..., Harness]) -> None:
    harness = make_window()
    harness.window.show_section("teeth")
    harness.window.section_page("teeth").editor("whitening").slider.setValue(55)
    assert harness.window.settings().teeth.whitening == pytest.approx(0.55)


def test_the_window_fits_a_small_screen(make_window: Callable[..., Harness]) -> None:
    window = make_window().window
    window.show()
    QApplication.processEvents()
    minimum = window.minimumSizeHint().expandedTo(window.minimumSize())
    # A 1280x720 screen at 150% is 853x480. Offscreen there are no Japanese fonts, and the boxes
    # drawn instead are wider; with the Windows fonts this is 800.
    assert minimum.width() <= 900 and minimum.height() <= 480


# ---- preset names ------------------------------------------------------------------


def write_builtin(folder: Path, name: str, display_name: str | None) -> None:
    data = MakeupSettings().to_dict()
    if display_name is not None:
        data["display_name"] = display_name
    (folder / f"{name}.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def test_the_menu_shows_display_names_and_keeps_the_file_name_as_its_data(
    make_window: Callable[..., Harness], isolated_dirs: Path
) -> None:
    write_builtin(isolated_dirs, "natural", "就活ナチュラル（標準）")
    write_builtin(isolated_dirs, "strong", "しっかりメイク＋小顔補正")
    harness = make_window(preset_name="natural")
    combo = harness.window._preset_combo

    assert combo.currentText() == "就活ナチュラル（標準）"  # not （標準）（標準）
    assert combo.findText("しっかりメイク＋小顔補正") >= 0
    assert combo.itemData(combo.findText("しっかりメイク＋小顔補正")) == "strong"
    combo.activated.emit(combo.findData("strong"))
    assert harness.window.preset_name == "strong"
    assert "しっかりメイク＋小顔補正" in harness.window.statusBar().currentMessage()


def test_a_user_preset_named_like_a_builtin_one_shows_its_own_name(
    make_window: Callable[..., Harness], isolated_dirs: Path
) -> None:
    write_builtin(isolated_dirs, "natural", "就活ナチュラル（標準）")
    save_preset("natural", MakeupSettings())
    combo = make_window(preset_name="natural").window._preset_combo
    assert combo.currentText() == "natural"


# ---- background capture ---------------------------------------------------------------


def test_capture_button_asks_the_runner_and_waits_for_the_answer(make_window: Callable[..., Harness]) -> None:
    harness = make_window()
    window, runner = harness.window, harness.runner
    button = window._capture_button
    assert "面接の前に撮影してください。撮影中の映像は相手にもそのまま映ります" in button.toolTip()
    assert "カメラの角度を変えたら撮り直してください" in button.toolTip()
    assert button.isEnabled()

    button.click()

    assert len(runner.capture_answers) == 1
    assert not button.isEnabled()
    button.click()
    assert len(runner.capture_answers) == 1


def test_the_countdown_is_shown_without_the_face_hint(make_window: Callable[..., Harness]) -> None:
    harness = make_window()
    harness.runner.on_frame(result((10, 20, 30), (0, 0, 0), face_found=False))
    assert harness.window.preview.hint == FACE_HINT
    message = plates.COUNTDOWN_MESSAGE.format(seconds=3)

    harness.runner.on_status(FakeStatus(background_message=message, face_found=False))

    assert harness.window.preview.banner == message
    assert harness.window.preview.hint == ""  # she is leaving the picture on purpose
    harness.runner.on_status(FakeStatus(background_message=None, face_found=False))
    assert harness.window.preview.banner == ""
    assert harness.window.preview.hint == FACE_HINT


def test_the_answer_from_another_thread_arrives_on_the_gui_thread(make_window: Callable[..., Harness]) -> None:
    harness = make_window()
    window, runner = harness.window, harness.runner
    window._capture_button.click()
    runner.has_background = True
    gui_thread = threading.current_thread()
    seen: list[threading.Thread] = []
    original = window._show_plate_result
    window._show_plate_result = lambda plate: (seen.append(threading.current_thread()), original(plate))

    worker = threading.Thread(target=runner.capture_answers[0], args=(plates.PlateResult(True, plates.SUCCESS_MESSAGE),))
    worker.start()
    worker.join()
    assert not seen  # nothing touched the widgets on the worker
    wait_until(lambda: bool(seen))

    assert seen == [gui_thread]
    assert window.preview.banner == plates.SUCCESS_MESSAGE
    assert window.statusBar().currentMessage() == plates.SUCCESS_MESSAGE
    assert window._capture_button.isEnabled()
    assert window._clear_plate_button.isEnabled()
    assert window._plate_state.text() == "撮影済み"


def test_a_rejected_capture_explains_itself_and_can_be_retried(make_window: Callable[..., Harness]) -> None:
    harness = make_window()
    window, runner = harness.window, harness.runner
    window._capture_button.click()
    message = plates.PERSON_VISIBLE_MESSAGE.format(percent=12.0)

    runner.capture_answers[0](plates.PlateResult(False, message))
    wait_until(lambda: window._capture_button.isEnabled())

    assert window.preview.banner == message
    assert window._plate_state.text() == "未撮影"
    assert not window._clear_plate_button.isEnabled()
    window._capture_button.click()
    assert len(runner.capture_answers) == 2
    assert window.preview.banner == ""


def test_the_result_message_goes_away_after_a_while(make_window: Callable[..., Harness]) -> None:
    harness = make_window()
    window = harness.window
    window._plate_message_timer.setInterval(10)
    window._capture_button.click()
    harness.runner.capture_answers[0](plates.PlateResult(True, plates.SUCCESS_MESSAGE))
    wait_until(lambda: window.preview.banner == plates.SUCCESS_MESSAGE)
    wait_until(lambda: window.preview.banner == "")


def test_clear_button_needs_a_background_and_reports(make_window: Callable[..., Harness]) -> None:
    harness = make_window()
    window, runner = harness.window, harness.runner
    assert not window._clear_plate_button.isEnabled()

    runner.has_background = True
    runner.on_status(FakeStatus())
    assert window._clear_plate_button.isEnabled()
    window._clear_plate_button.click()

    assert runner.clears == 1
    assert window.statusBar().currentMessage() == plates.CLEARED_MESSAGE
    assert window.preview.banner == plates.CLEARED_MESSAGE
    assert not window._clear_plate_button.isEnabled()
    assert window._plate_state.text() == "未撮影"


def test_clear_button_also_reaches_a_plate_of_another_resolution(make_window: Callable[..., Harness]) -> None:
    folder = config.user_data_dir()
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "background_640x480.png").write_bytes(b"room")
    harness = make_window()  # in use: none, for this resolution
    assert harness.window._plate_state.text() == "未撮影"
    assert harness.window._clear_plate_button.isEnabled()


class PlateWorld:
    """A real runner on a real pipeline without models, on a camera that can fail."""

    def __init__(self) -> None:
        self.outage = threading.Event()
        self.released = threading.Event()
        self.released.set()

    def read(self) -> np.ndarray | None:
        time.sleep(0.005)
        return None if self.outage.is_set() else np.full((48, 64, 3), 120, np.uint8)

    def open(self) -> None:
        pass

    def close(self) -> None:
        pass

    def probe(self, _frame: np.ndarray) -> float:
        assert self.released.wait(5.0)
        return 0.0

    device = "Fake Virtual Camera"

    def send(self, _frame: np.ndarray) -> None:
        pass


class NoFaceTracker:
    def detect(self, frame: np.ndarray, ts_ms: int) -> tuple[None, None]:
        return None, None

    def set_smoothing(self, smoothing: float) -> None:
        pass

    def close(self) -> None:
        pass


class NoMasks:
    def update(self, frame: np.ndarray, ts_ms: int, frame_index: int) -> None:
        return None

    def set_interval(self, interval: int) -> None:
        pass

    def close(self) -> None:
        pass


@pytest.fixture
def plate_window(qapp: QApplication) -> Iterator[Callable[..., tuple[MainWindow, PlateWorld]]]:
    windows: list[MainWindow] = []

    def make(countdown_s: float = 0.02, timeout_s: float = 0.3) -> tuple[MainWindow, PlateWorld]:
        world = PlateWorld()
        pipeline = MakeupPipeline(
            MakeupSettings(), tracker=NoFaceTracker(), segmenter=NoMasks(), effects=[], person_probe=world.probe
        )

        def factory(on_frame: Callable, on_status: Callable) -> CameraRunner:
            return CameraRunner(
                pipeline,
                width=64,
                height=48,
                camera_factory=lambda **_: world,
                vcam_factory=lambda **_: world,
                on_frame=on_frame,
                on_status=on_status,
                camera_timeout_s=0.05,
                plate_collector_factory=lambda: plates.PlateCollector(
                    countdown_s=countdown_s, count=6, spacing_s=0.005, timeout_s=timeout_s
                ),
            )

        window = MainWindow(factory, settings=MakeupSettings(), dialogs=FakeDialogs())
        windows.append(window)
        window.start()
        wait_until(lambda: window.preview.image is not None)
        return window, world

    yield make
    for window in windows:
        window.close()
        window.deleteLater()
    QApplication.processEvents()


def test_a_real_capture_through_the_window(plate_window: Callable) -> None:
    window, _ = plate_window()

    window._capture_button.click()
    wait_until(lambda: window.preview.banner == plates.SUCCESS_MESSAGE)

    assert plates.plate_path(64, 48).is_file()
    assert window._capture_button.isEnabled()
    wait_until(lambda: window._plate_state.text() == "撮影済み")
    assert window._clear_plate_button.isEnabled()
    window._clear_plate_button.click()
    assert not plates.plate_path(64, 48).exists()
    assert window._plate_state.text() == "未撮影"


def test_the_camera_lost_during_a_capture_ends_it_with_a_message(plate_window: Callable) -> None:
    window, world = plate_window()
    world.outage.set()
    wait_until(lambda: bool(window._last_status) and not window._last_status.camera_ok)

    window._capture_button.click()
    wait_until(lambda: window._capture_button.isEnabled())

    assert window.statusBar().currentMessage() == plates.TOO_FEW_FRAMES_MESSAGE
    assert not plates.plate_path(64, 48).exists()
    assert window._plate_state.text() == "未撮影"


def test_deleting_during_the_countdown_ends_it_at_once_in_the_window(plate_window: Callable) -> None:
    window, world = plate_window(countdown_s=60.0)
    window._capture_button.click()
    wait_until(lambda: window.preview.banner.endswith("カメラの前から離れてください"))
    window._plates_saved = True  # an older plate, of another resolution

    window._on_clear_background()

    wait_until(lambda: window.preview.banner == plates.CAPTURE_CANCELLED_MESSAGE)
    assert window._capture_button.isEnabled()
    assert not plates.plate_path(64, 48).exists()


def test_closing_during_the_countdown_does_not_hang(plate_window: Callable) -> None:
    window, _ = plate_window(countdown_s=60.0)
    window._capture_button.click()
    wait_until(lambda: window.preview.banner.endswith("カメラの前から離れてください"))
    started = time.monotonic()

    window.close()

    assert time.monotonic() - started < gui.RUNNER_STOP_DEADLINE_S
    QApplication.processEvents()  # the late answer lands on a closed window without harm


def test_the_gui_thread_does_not_wait_for_the_person_check(plate_window: Callable) -> None:
    window, world = plate_window()
    world.released.clear()
    window._capture_button.click()
    wait_until(lambda: window.preview.banner == plates.CAPTURING_MESSAGE)
    started = time.monotonic()
    for _ in range(5):
        QApplication.processEvents()
    assert time.monotonic() - started < 0.2
    assert not window._capture_button.isEnabled()
    world.released.set()
    wait_until(lambda: window._capture_button.isEnabled())


# ---- 写真から作る -------------------------------------------------------------------


def analysis(confidence: dict[str, float], notes: tuple[str, ...] = ()) -> AnalysisResult:
    settings = MakeupSettings()
    settings.lips.opacity = 0.77
    return AnalysisResult(settings=settings, confidence=confidence, notes=list(notes))


class FakeAnalyzer:
    """Stands in for the analysis; can be held open and remembers where it ran."""

    def __init__(self, outcome: AnalysisResult | Exception) -> None:
        self.outcome = outcome
        self.calls: list[tuple[Path, Path | None, MakeupSettings | None]] = []
        self.threads: list[threading.Thread] = []
        self.release = threading.Event()
        self.release.set()

    def __call__(self, makeup: Path, bare: Path | None, base: MakeupSettings | None) -> AnalysisResult:
        self.calls.append((makeup, bare, base))
        self.threads.append(threading.current_thread())
        assert self.release.wait(5.0)
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


GOOD = {"lips": 0.85, "blush": 0.8, "skin": 0.9, "contour": 0.0}


def run_photo_action(harness: Harness) -> None:
    harness.window._photo_button.click()
    wait_until(lambda: harness.window._photo_button.isEnabled())


def test_photo_action_runs_on_a_worker_and_selects_the_new_preset(
    make_window: Callable[..., Harness], isolated_dirs: Path
) -> None:
    analyzer = FakeAnalyzer(analysis(GOOD, ("前髪で眉が隠れています",)))
    analyzer.release.clear()
    harness = make_window(photo_analyzer=analyzer)
    window = harness.window
    harness.dialogs.photos = gui.PhotoRequest(Path("m.jpg"), Path("b.jpg"), "いつものメイク")

    window._photo_button.click()

    wait_until(lambda: bool(analyzer.calls))  # the GUI thread came back without the analysis being over
    assert not window._photo_button.isEnabled() and window._photo_button.text() == "解析中…"
    assert not harness.dialogs.reports
    QApplication.processEvents()
    analyzer.release.set()
    wait_until(lambda: window._photo_button.isEnabled())

    assert analyzer.threads[0] is not threading.current_thread()
    assert analyzer.calls[0][:2] == (Path("m.jpg"), Path("b.jpg"))
    assert window.preset_name == "いつものメイク"
    assert window._preset_combo.currentText() == "いつものメイク"
    assert window.settings().lips.opacity == pytest.approx(0.77)
    assert harness.runner.pushed[-1].lips.opacity == pytest.approx(0.77)
    assert load_preset("いつものメイク").lips.opacity == pytest.approx(0.77)
    report = harness.dialogs.reports[0]
    assert report.notes == ("前髪で眉が隠れています",)
    assert (SECTION_LABELS["lips"], "0.85", True) in report.rows
    assert (SECTION_LABELS["contour"], "測れなかった（natural の値のまま）", False) in report.rows
    assert (SECTION_LABELS["eyeshadow"], "測れなかった（natural の値のまま）", False) in report.rows  # absent: not measured
    assert gui.PHOTO_BARE_ADVICE not in report.summary


def test_photo_action_starts_from_the_selected_preset(make_window: Callable[..., Harness]) -> None:
    base = MakeupSettings()
    base.blush.opacity = 0.11
    save_preset("A", base)
    analyzer = FakeAnalyzer(analysis(GOOD))
    harness = make_window(base, preset_name="A", photo_analyzer=analyzer)
    harness.dialogs.photos = gui.PhotoRequest(Path("m.jpg"), Path("b.jpg"), "new")

    run_photo_action(harness)

    assert analyzer.calls[0][2].blush.opacity == pytest.approx(0.11)
    assert (SECTION_LABELS["contour"], "測れなかった（A の値のまま）", False) in harness.dialogs.reports[0].rows


def test_photo_action_without_a_bare_photo_urges_to_add_one(make_window: Callable[..., Harness]) -> None:
    analyzer = FakeAnalyzer(analysis(GOOD))
    harness = make_window(photo_analyzer=analyzer)
    harness.dialogs.photos = gui.PhotoRequest(Path("m.jpg"), None, "one")

    harness.dialogs.confirm_answer = False
    harness.window._photo_button.click()
    assert not analyzer.calls
    assert "アイシャドウをほとんど測れず" in harness.dialogs.questions[-1]
    assert harness.window._photo_button.isEnabled()

    harness.dialogs.confirm_answer = True
    run_photo_action(harness)
    assert analyzer.calls[0][1] is None
    assert gui.PHOTO_BARE_ADVICE in harness.dialogs.reports[0].summary


def test_photo_action_refuses_a_builtin_name_and_asks_for_another_without_a_new_analysis(
    make_window: Callable[..., Harness], isolated_dirs: Path
) -> None:
    write_builtin(isolated_dirs, "natural", "就活ナチュラル（標準）")
    analyzer = FakeAnalyzer(analysis(GOOD))
    harness = make_window(photo_analyzer=analyzer)
    harness.dialogs.photos = gui.PhotoRequest(Path("m.jpg"), Path("b.jpg"), "natural")
    harness.dialogs.more_names = ["いつもの"]

    run_photo_action(harness)

    assert harness.dialogs.errors == ["『natural』は標準プリセットの名前なので使えません。別の名前を指定してください。"]
    assert len(analyzer.calls) == 1
    assert find_preset("natural").builtin
    assert harness.window.preset_name == "いつもの"


def test_photo_action_gives_up_when_no_other_name_is_given(
    make_window: Callable[..., Harness], isolated_dirs: Path
) -> None:
    write_builtin(isolated_dirs, "natural", "就活ナチュラル（標準）")
    harness = make_window(photo_analyzer=FakeAnalyzer(analysis(GOOD)))
    harness.dialogs.photos = gui.PhotoRequest(Path("m.jpg"), Path("b.jpg"), "natural")
    harness.dialogs.more_names = [None]

    run_photo_action(harness)

    assert harness.window.preset_name is None
    assert not harness.dialogs.reports


def test_photo_action_asks_before_replacing_a_preset_of_hers(make_window: Callable[..., Harness]) -> None:
    save_preset("A", MakeupSettings())
    harness = make_window(photo_analyzer=FakeAnalyzer(analysis(GOOD)))
    harness.dialogs.photos = gui.PhotoRequest(Path("m.jpg"), Path("b.jpg"), "A")
    harness.dialogs.confirm_answer = False
    harness.dialogs.more_names = [None]

    run_photo_action(harness)

    assert "すでにあります" in harness.dialogs.questions[-1]
    assert load_preset("A") == MakeupSettings()


def test_photo_action_keeps_unsaved_changes_if_she_says_so(make_window: Callable[..., Harness]) -> None:
    harness = make_window(photo_analyzer=FakeAnalyzer(analysis(GOOD)))
    harness.window.section_page("blush").editor("opacity").slider.setValue(77)
    harness.dialogs.photos = gui.PhotoRequest(Path("m.jpg"), Path("b.jpg"), "new")
    harness.dialogs.confirm_answer = False

    run_photo_action(harness)

    assert find_preset("new") is not None
    assert harness.window._preset_combo.findData("new") >= 0
    assert harness.window.preset_name is None
    assert harness.window.settings().blush.opacity == pytest.approx(0.77)
    assert harness.dialogs.reports


def test_photo_action_saves_nothing_when_nothing_could_be_measured(make_window: Callable[..., Harness]) -> None:
    outcome = analysis({"lips": 0.0, "skin": 0.0}, ("白黒の写真です",))
    harness = make_window(photo_analyzer=FakeAnalyzer(outcome))
    harness.dialogs.photos = gui.PhotoRequest(Path("m.jpg"), Path("b.jpg"), "bad")

    run_photo_action(harness)

    assert find_preset("bad") is None
    assert harness.window.preset_name is None
    report = harness.dialogs.reports[0]
    assert report.title == gui.PHOTO_NOTHING_MEASURED_TITLE
    assert "プリセットは保存していません" in report.summary
    assert report.notes == ("白黒の写真です",) and report.rows == ()


def test_photo_action_shows_an_analysis_error_as_it_is(make_window: Callable[..., Harness]) -> None:
    harness = make_window(photo_analyzer=FakeAnalyzer(AnalysisError("顔が見つかりませんでした。")))
    harness.dialogs.photos = gui.PhotoRequest(Path("m.jpg"), Path("b.jpg"), "x")

    run_photo_action(harness)

    assert harness.dialogs.errors == ["顔が見つかりませんでした。"]
    assert find_preset("x") is None


def test_photo_action_survives_an_unexpected_error(make_window: Callable[..., Harness]) -> None:
    harness = make_window(photo_analyzer=FakeAnalyzer(RuntimeError("native failure")))
    harness.dialogs.photos = gui.PhotoRequest(Path("m.jpg"), Path("b.jpg"), "x")

    run_photo_action(harness)

    assert "写真を解析できませんでした" in harness.dialogs.errors[0]


def test_closing_during_an_analysis_discards_it(make_window: Callable[..., Harness]) -> None:
    analyzer = FakeAnalyzer(analysis(GOOD))
    analyzer.release.clear()
    harness = make_window(photo_analyzer=analyzer)
    harness.dialogs.photos = gui.PhotoRequest(Path("m.jpg"), Path("b.jpg"), "late")
    harness.window._photo_button.click()
    wait_until(lambda: bool(analyzer.calls))

    harness.window.close()
    assert not harness.window.wait_for_photo_analysis(0.01)
    analyzer.release.set()
    assert harness.window.wait_for_photo_analysis(5.0)
    wait_until(lambda: harness.window._photo_button.isEnabled())

    assert find_preset("late") is None
    assert not harness.dialogs.reports


def test_the_photo_dialog_needs_a_photo_and_a_name(qapp: QApplication) -> None:
    dialog = gui.PhotoPresetDialog()
    assert not dialog.create_button.isEnabled()
    dialog.makeup_edit.setText("m.jpg")
    assert not dialog.create_button.isEnabled()
    dialog.name_edit.setText("いつもの")
    assert dialog.create_button.isEnabled()
    assert dialog.request() == gui.PhotoRequest(Path("m.jpg"), None, "いつもの")
    dialog.bare_edit.setText("b.jpg")
    assert dialog.request().bare == Path("b.jpg")
    text = " ".join(label.text() for label in dialog.findChildren(gui.QLabel))
    for needed in ("同じ場所・同じ照明", "前髪を上げ", "メガネを外す", "構図と服", "アイシャドウをほとんど測れず"):
        assert needed in text
    dialog.deleteLater()


def test_the_report_dialog_marks_what_was_not_measured(qapp: QApplication) -> None:
    report = gui.photo_report(analysis(GOOD, ("<b>メモ</b>",)), "新しい", "natural", used_bare=True)
    dialog = gui.PhotoReportDialog(report)
    text = dialog.body.text()
    assert "測れなかった（natural の値のまま）" in text and gui._ERROR_COLOR in text
    assert "&lt;b&gt;メモ&lt;/b&gt;" in text  # notes are text, not markup
    dialog.deleteLater()


def test_a_real_analysis_of_an_unreadable_file_is_reported_in_japanese(
    make_window: Callable[..., Harness], tmp_path: Path
) -> None:
    (tmp_path / "broken.jpg").write_bytes(b"not a photo")
    harness = make_window()  # the real analyzer
    harness.dialogs.photos = gui.PhotoRequest(tmp_path / "broken.jpg", tmp_path / "broken.jpg", "x")

    run_photo_action(harness)

    assert "写真を読み込めませんでした" in harness.dialogs.errors[0]


# ---- how the process ends --------------------------------------------------------------


_RUN_AND_CLOSE = """
import os, sys
from PySide6.QtCore import QTimer
from makeup_cam import gui, pipeline

class Pipeline:
    bypass = False
    has_background = False
    def __init__(self, settings): pass
    def set_settings(self, settings): pass
    def glide_to_settings(self, settings): pass
    def process(self, frame, ts): raise RuntimeError("no frames in this test")
    def close(self): pass

pipeline.MakeupPipeline = Pipeline
gui.prepare_models = lambda: None
original = gui.MainWindow.start
def start(self):
    original(self)
    QTimer.singleShot(100, self.close)
gui.MainWindow.start = start
import argparse
args = argparse.Namespace(camera_file=sys.argv[1], no_vcam=True)
sys.exit(gui.run_gui(args))
"""


def test_the_process_ends_without_crashing(tmp_path: Path) -> None:
    """A real run to its end: a window destroyed after QApplication ends the process in an access violation."""
    clip = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (64, 48))
    for _ in range(5):
        writer.write(np.zeros((48, 64, 3), np.uint8))
    writer.release()
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "QT_QPA_PLATFORM": "offscreen", "AI_MAKEUP_CAM_HOME": str(tmp_path / "home")}
    children = [
        subprocess.Popen(
            [sys.executable, "-c", _RUN_AND_CLOSE, str(clip)],
            cwd=root,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        for _ in range(4)
    ]
    assert [child.wait(timeout=60) for child in children] == [0, 0, 0, 0]


def test_run_gui_destroys_the_window_while_the_application_lives(
    qapp: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from makeup_cam import pipeline as pipeline_module

    class Pipeline:
        bypass = False
        has_background = False
        closed = False

        def __init__(self, _settings: MakeupSettings) -> None:
            pass

        def set_settings(self, _settings: MakeupSettings) -> None:
            pass

        def glide_to_settings(self, _settings: MakeupSettings) -> None:
            pass

        def process(self, frame: np.ndarray, _ts: int) -> FakeResult:
            return FakeResult(output=frame, preview=frame, original_preview=frame)

        def close(self) -> None:
            Pipeline.closed = True

    made: list[MainWindow] = []
    original_start = MainWindow.start

    def start(self: MainWindow) -> None:
        made.append(self)
        original_start(self)
        gui.QTimer.singleShot(50, self.close)

    clip = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (64, 48))
    for _ in range(5):
        writer.write(np.zeros((48, 64, 3), np.uint8))
    writer.release()
    monkeypatch.setattr(pipeline_module, "MakeupPipeline", Pipeline)
    monkeypatch.setattr(gui, "prepare_models", lambda: None)
    monkeypatch.setattr(MainWindow, "start", start)

    assert gui.run_gui(argparse.Namespace(camera_file=clip, no_vcam=True)) == 0

    assert Pipeline.closed
    assert not Shiboken.isValid(made[0])


def test_ctrl_c_also_closes_an_open_dialog(qapp: QApplication) -> None:
    window = QMainWindow()
    window.show()
    dialog = gui.QDialog(window)
    dialog.show()
    with gui.closed_by_ctrl_c(window):
        signal.raise_signal(signal.SIGINT)
        qapp.processEvents()
    assert not window.isVisible() and not dialog.isVisible()


def test_a_closing_console_closes_the_window(qapp: QApplication) -> None:
    window = QMainWindow()
    window.show()
    closing = threading.Event()
    with gui.closed_by_ctrl_c(window, closing):
        qapp.processEvents()
        assert window.isVisible()
        closing.set()
        wait_until(lambda: not window.isVisible())


@pytest.mark.parametrize("key", [Qt.Key.Key_Space, Qt.Key.Key_Return])
def test_buttons_work_from_the_keyboard(make_window: Callable[..., Harness], key: Qt.Key) -> None:
    harness = make_window()
    harness.window.show()
    button = harness.window.findChild(QPushButton, "bypassButton") or next(
        b for b in harness.window.findChildren(QPushButton) if b.isCheckable() and b.isEnabled()
    )
    before = button.isChecked()
    QTest.keyClick(button, key)
    wait_until(lambda: button.isChecked() != before)


def test_space_on_a_checkbox_toggles_it_and_does_not_compare(make_window: Callable[..., Harness]) -> None:
    harness = make_window()
    harness.runner.on_frame(result((10, 20, 30), (200, 150, 100)))
    box = next(b for b in harness.window.findChildren(QCheckBox) if b.isEnabled())
    before = box.isChecked()
    QTest.keyPress(box, Qt.Key.Key_Space)
    assert shown_color(harness.window) == (10, 20, 30)
    QTest.keyRelease(box, Qt.Key.Key_Space)
    assert box.isChecked() != before


def test_clicking_a_button_leaves_space_to_compare(make_window: Callable[..., Harness]) -> None:
    harness = make_window()
    for button in harness.window.findChildren(QAbstractButton):
        assert button.focusPolicy() == Qt.FocusPolicy.TabFocus


@pytest.mark.parametrize("pasted", ['"C:/写真/メイク.jpg"', '  "C:/写真/メイク.jpg" ', "C:/写真/メイク.jpg"])
def test_a_path_copied_in_explorer_can_be_pasted(qapp, pasted: str) -> None:
    dialog = gui.PhotoPresetDialog()
    dialog.makeup_edit.setText(pasted)
    dialog.bare_edit.setText(pasted)
    dialog.name_edit.setText("x")
    request = dialog.request()
    assert request.makeup == Path("C:/写真/メイク.jpg")
    assert request.bare == Path("C:/写真/メイク.jpg")


def test_face_shape_preset_without_a_plate_tells_about_the_background() -> None:
    strong = MakeupSettings()
    strong.reshape.enabled = True
    strong.reshape.face_slim = 0.4
    assert gui._uses_face_shape(strong)
    assert not gui._uses_face_shape(MakeupSettings())
    strong.reshape.enabled = False
    assert not gui._uses_face_shape(strong)
